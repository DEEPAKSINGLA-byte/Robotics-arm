import unittest
from prepare_splits import digest
from evaluate_final_test import aggregate, verify_test_split


class FinalTestTests(unittest.TestCase):
    def fixture(self):
        rows = {s: [dict(key=s, scene=s, group=s)] for s in ('train', 'val', 'test')}
        fingerprint = digest(dict(mode='sequence', seed=42, splits=rows))
        summary = dict(mode='sequence', seed=42, fingerprint=fingerprint)
        docs = {s: dict(mode='sequence', split=s, fingerprint=fingerprint, rows=r) for s, r in rows.items()}
        checkpoint = dict(mode='sequence', overfit=False, fingerprint=fingerprint, training_groups=['train'])
        return docs, summary, checkpoint

    def test_test_manifest_and_model_guards(self):
        docs, summary, checkpoint = self.fixture()
        verify_test_split(docs, summary, checkpoint)
        with self.assertRaises(ValueError):
            verify_test_split(docs, summary, dict(checkpoint, training_groups=['test']))
        with self.assertRaises(ValueError):
            verify_test_split(docs, summary, dict(checkpoint, overfit=True))
        docs['test']['rows'][0]['scene'] = 'train'
        with self.assertRaises(ValueError):
            verify_test_split(docs, summary, checkpoint)

    def test_full_denominator_keeps_missing_targets(self):
        rows = [dict(scene='a', group='g'), dict(scene='a', group='g'), dict(scene='b', group='g')]
        results = [dict(success=True, available=True, selected_iou=.8),
                   dict(success=False, available=True, selected_iou=.1),
                   dict(success=False, available=False, selected_iou=0.)]
        result = aggregate(rows, results)
        self.assertEqual(result['expressions'], 3)
        self.assertEqual(result['images'], 2)
        self.assertEqual(result['accuracy'], 1/3)
        self.assertEqual(result['scene_macro_accuracy'], .25)
        self.assertEqual(result['proposal_miss'], 1)
        self.assertEqual(result['selection_miss'], 1)
        self.assertTrue(result['test_evaluated'])
        with self.assertRaises(ValueError):
            aggregate(rows, results[:-1])


if __name__ == '__main__':
    unittest.main()
