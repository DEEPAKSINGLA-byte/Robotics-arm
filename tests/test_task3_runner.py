"""Quick checks for portable settings and validation-only scoring."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from bin_grasp.pipeline import REPO, load_config
from bin_grasp.verify import verify


class RunnerChecks(unittest.TestCase):
    def test_default_paths_are_relative_to_repo(self):
        config = load_config(REPO / 'configs/task3.json')
        self.assertEqual(Path(config['checkpoint']), REPO / 'runs/sam-adaptation-relaxed-300/best.pt')
        self.assertEqual(config['sam_settings']['pred_iou_thresh'], .7)
        self.assertEqual(config['filtering']['max_candidates'], 128)

    def test_absolute_model_paths_are_kept(self):
        keys = ('checkpoint', 'sam_path', 'siglip_path', 'depth_head', 'moge_checkpoint')
        values = {key: '/tmp/task3-test-model' for key in keys}
        with patch('bin_grasp.pipeline.read', return_value=values):
            config = load_config('unused.json')
        self.assertTrue(all(config[key] == '/tmp/task3-test-model' for key in keys))

    def test_verification_rejects_test_manifest(self):
        with patch('bin_grasp.verify.read', return_value={'split': 'test', 'mode': 'sequence'}):
            with self.assertRaisesRegex(ValueError, 'validation'):
                verify([], Path('/tmp'), 'unused.json')

    def test_no_candidates_does_not_need_depth_or_old_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scene = 'scene/rgb/image.png'
            labels = root / 'scene/label/image.png'
            labels.parent.mkdir(parents=True)
            Image.fromarray(np.ones((4, 4), dtype='uint8')).save(labels)
            run = root / 'run'
            (run / 'selection').mkdir(parents=True)
            Image.fromarray(np.zeros((4, 4), dtype='uint8')).save(run / 'selection/selected_mask.png')
            (run / 'result.json').write_text(json.dumps({
                'image': str(root / scene), 'sentence': 'the box', 'status': 'no_candidates'}))
            manifest = root / 'val.json'
            manifest.write_text(json.dumps({'split': 'val', 'mode': 'sequence', 'rows': [
                {'scene': scene, 'sentence': 'the box', 'target_id': 1}]}))
            result = verify([run], root, manifest)
            self.assertEqual(result['scenes'], 1)
            self.assertEqual(result['selected_target_count'], 0)
            self.assertEqual(result['records'][0]['proposals'], 0)
            self.assertFalse(result['records'][0]['robot_execution_ready'])


if __name__ == '__main__':
    unittest.main()
