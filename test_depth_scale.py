import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from run_depth_scale import ScaleHead, target_scale, metric, catalog, exclude_training


class DepthScaleTests(unittest.TestCase):
    def test_exclusions_are_explicit_and_train_only(self):
        original = dict(train=[dict(scene='a'), dict(scene='b')],
                        val=[dict(scene='v')], test=[dict(scene='t')])
        result = exclude_training(original, ['a'])
        self.assertEqual(result['train'], [dict(scene='b')])
        self.assertEqual(len(original['train']), 2)
        self.assertIs(result['val'], original['val'])
        self.assertIs(result['test'], original['test'])
        for invalid in (['v'], ['t'], ['typo'], ['a', 'b']):
            with self.assertRaises(ValueError):
                exclude_training(original, invalid)

    def test_identity_initialization_and_positive_scale(self):
        head = ScaleHead(8)
        features = torch.randn(3, 8)
        self.assertTrue(torch.equal(head(features).exp(), torch.ones(3)))
        head(features).sum().backward()
        self.assertIsNotNone(head.net[-1].weight.grad)

    def test_known_scale_ignores_missing_depth(self):
        pred = np.ones((20, 20), np.float32) * 2
        depth = np.ones((20, 20), np.uint16) * 1000
        depth[0] = 0
        self.assertAlmostEqual(np.exp(target_scale(pred, depth)), .5)

    def test_full_resolution_error_and_coverage(self):
        pred = np.array([[2., np.nan], [2., 2.]])
        depth = np.ones((2, 2), np.uint16) * 1000
        scores = metric(pred, depth, np.ones((2, 2), bool), .5)
        self.assertEqual(scores['object_mae_m'], 0)
        self.assertEqual(scores['object_coverage'], .75)
        self.assertEqual(scores['object_delta1'], .75)

    def test_reject_invalid_target(self):
        with self.assertRaises(ValueError):
            target_scale(np.zeros((20, 20)), np.zeros((20, 20)))

    def test_deduplication_and_sequence_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            for split in ('train', 'val', 'test'):
                rows = [dict(scene=f'{split}/rgb/x.png', group=split, target_id=i) for i in (1, 2)]
                (Path(folder)/f'{split}.json').write_text(json.dumps(dict(
                    mode='sequence', split=split, fingerprint='same', rows=rows)))
            catalogs, _ = catalog(folder)
            self.assertEqual(len(catalogs['train']), 1)
            self.assertEqual(catalogs['train'][0]['ids'], [1, 2])
            p = Path(folder)/'val.json'
            manifest = json.loads(p.read_text())
            for row in manifest['rows']:
                row['group'] = 'train'
            p.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'overlap'):
                catalog(folder)

    def test_head_can_learn_image_dependent_factors(self):
        torch.manual_seed(42)
        x = torch.randn(32, 4)
        target = x[:, 0] * .3
        head = ScaleHead(4, hidden=8)
        optimizer = torch.optim.Adam(head.parameters(), lr=.03)
        initial = float(((head(x)-target)**2).mean().detach())
        for _ in range(100):
            optimizer.zero_grad()
            loss = ((head(x)-target)**2).mean()
            loss.backward()
            optimizer.step()
        self.assertLess(float(((head(x)-target)**2).mean().detach()), initial*.1)


if __name__ == '__main__':
    unittest.main()
