import unittest
import numpy as np
from PIL import Image
from compare_sam_selectors import remaining_index
from run_experiment2 import select_rows, generate_masks, tiled_boxes
from run_sam_settings_study import VARIANTS
from transformers.pipelines.mask_generation import MaskGenerationPipeline


class SettingsStudyTests(unittest.TestCase):
    def test_subset_excludes_used_scenes_not_just_sentences(self):
        rows = [dict(key=f'{s}-{e}', scene=str(s), group=str(s % 2))
                for s in range(8) for e in range(3)]
        index = dict(split='val', mode='sequence', rows=rows)
        subset = remaining_index(index, [rows[0]])
        self.assertNotIn('0', {r['scene'] for r in subset['rows']})
        selected = select_rows(subset, 4, 42)
        frozen = dict(index, rows=selected)
        self.assertEqual({r['key'] for r in selected},
                         {r['key'] for r in select_rows(frozen, 4, 42)})
        with self.assertRaises(ValueError):
            remaining_index(dict(index, split='test'), [rows[0]])

    def test_installed_pipeline_accepts_settings(self):
        obj = object.__new__(MaskGenerationPipeline)
        preprocess, forward, _ = obj._sanitize_parameters(
            crops_n_layers=1, points_per_crop=64,
            pred_iou_thresh=.7, stability_score_thresh=.9)
        self.assertEqual(preprocess, dict(crops_n_layers=1, points_per_crop=64))
        self.assertEqual(forward, dict(pred_iou_thresh=.7, stability_score_thresh=.9))

    def test_predeclared_variants(self):
        self.assertEqual(list(VARIANTS), ['baseline', 'dense64', 'crops1', 'relaxed'])
        self.assertEqual(VARIANTS['baseline'], [])

    def test_explicit_crops_all_contribute_in_full_image_coordinates(self):
        calls = []
        def fake(image, **settings):
            self.assertEqual(settings['crops_n_layers'], 0)
            calls.append(image.size)
            mask = np.zeros((image.height, image.width), bool)
            mask[3:6, 3:6] = True
            return dict(masks=[mask], scores=[.9])
        result = generate_masks(fake, Image.new('RGB', (100, 80)),
                                dict(crops_n_layers=1, crops_nms_thresh=.7))
        self.assertEqual(len(calls), 5)
        self.assertEqual(len(result['masks']), 4)  # identical full-image/top-left mask merges
        self.assertTrue(all(m.shape == (80, 100) for m in result['masks']))
        bottom_right = tiled_boxes(100, 80)[-1]
        self.assertTrue(any(m[bottom_right[1]+3, bottom_right[0]+3] for m in result['masks']))


if __name__ == '__main__':
    unittest.main()
