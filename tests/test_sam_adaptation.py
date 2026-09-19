import unittest
import numpy as np
import torch
from experiments.run_sam_adaptation import check_indexes, training_rows, positive_loss, evaluate


class AdaptationTests(unittest.TestCase):
    def test_only_disjoint_sequence_splits(self):
        common = dict(cache_id='a', fingerprint='b', mode='sequence', recipe={})
        train = dict(common, split='train', rows=[dict(key='a', scene='a', group='a')])
        val = dict(common, split='val', rows=[dict(key='b', scene='b', group='b')])
        check_indexes(train, val)
        with self.assertRaises(ValueError):
            check_indexes(dict(train, split='test'), val)
        with self.assertRaises(ValueError):
            check_indexes(train, dict(val, rows=[dict(key='b', scene='b', group='a')]))

    def test_sampling_reproducible_without_answers(self):
        rows = [dict(scene=str(s), key=f'{s}:{e}') for s in range(5) for e in range(8)]
        index = dict(split='train', rows=rows)
        selected = training_rows(index, 3, 4, 42)
        self.assertEqual(selected, training_rows(index, 3, 4, 42))
        self.assertEqual(len(selected), 12)
        self.assertEqual(len({r['scene'] for r in selected}), 3)
        with self.assertRaises(ValueError):
            training_rows(dict(index, split='val'), 3, 4, 42)

    def test_multiple_good_candidates_and_padding(self):
        scores = torch.tensor([[0., 0., 0., float('-inf')]], requires_grad=True)
        positive = torch.tensor([[True, True, False, False]])
        loss = positive_loss(scores, positive)
        self.assertAlmostEqual(loss.item(), -np.log(2/3), places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertEqual(scores.grad[0, 3].item(), 0.)

    def test_missing_target_cannot_train(self):
        with self.assertRaises(ValueError):
            positive_loss(torch.zeros(1, 3), torch.zeros(1, 3, dtype=torch.bool))

    def test_empty_validation_proposals_remain_failures(self):
        model = torch.nn.Linear(1, 1)
        result, predictions = evaluate(model, [({}, torch.empty(0), 'missing')], 4)
        self.assertEqual(result['samples'], 1)
        self.assertEqual(result['proposal_miss'], 1)
        self.assertEqual(result['accuracy'], 0)
        self.assertIsNone(predictions[0]['selected_index'])


if __name__ == '__main__':
    unittest.main()
