"""Checks for padding, candidate order, leakage boundaries, and sequence splits."""
import unittest
import torch
from bin_grasp.model import GroundingModel
from bin_grasp.data import collate, verify_pair, ResidentFeatures
from experiments.evaluate_experiment1 import evaluate
from experiments.train_experiment1 import resolve_model_config
from experiments.prepare_splits import make_splits
from types import SimpleNamespace


class GroundingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def inputs(self, n):
        return dict(visual=torch.randn(n,768), geometry=torch.rand(n,7),
                    text=torch.randn(768), scene=torch.randn(768))

    def test_experiment_configuration_and_checkpoint_guard(self):
        args=SimpleNamespace(hidden=128,layers=None,dropout=None,overfit=False)
        small=resolve_model_config(args)
        self.assertEqual(small['hidden'],128)
        self.assertEqual(small['layers'],2)
        self.assertEqual(small['dropout'],.1)
        initial={'config':GroundingModel().config}
        with self.assertRaisesRegex(ValueError,'Cannot change hidden'):
            resolve_model_config(args,initial)
        args.hidden=None
        self.assertEqual(resolve_model_config(args,initial),initial['config'])
        args.dropout=1.0
        with self.assertRaisesRegex(ValueError,'dropout'):
            resolve_model_config(args)
        args.dropout=.3
        cfg=resolve_model_config(args)
        model=GroundingModel(**cfg).eval()
        batch,_,_=collate([(self.inputs(3),1,{})])
        with torch.no_grad():
            self.assertTrue(torch.equal(model(**batch),model(**batch)))

    def test_padding_cannot_be_selected_or_change_real_scores(self):
        model = GroundingModel(hidden=32, layers=1, heads=4, dropout=0).eval()
        a, b = self.inputs(3), self.inputs(8)
        alone, _, _ = collate([(a,1,{})])
        together, targets, _ = collate([(a,1,{}),(b,6,{})])
        with torch.no_grad():
            first, batch = model(**alone), model(**together)
        self.assertTrue(torch.allclose(first[0],batch[0,:3], atol=1e-5))
        self.assertTrue(torch.isneginf(batch[0,3:]).all())
        self.assertTrue(torch.isfinite(torch.nn.functional.cross_entropy(batch,targets)))

    def test_candidate_order_does_not_supply_the_answer(self):
        model = GroundingModel(hidden=32,layers=1,heads=4,dropout=0).eval()
        a = self.inputs(5)
        batch,_,_ = collate([(a,2,{})])
        order = torch.tensor([4,1,3,0,2])
        shuffled = dict(batch)
        for key in ('visual','geometry','valid'):
            shuffled[key] = batch[key][:,order]
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(**batch)[:,order],model(**shuffled),atol=1e-5))

    def test_answers_and_metadata_stay_outside_forward_inputs(self):
        batch,targets,metadata = collate([(self.inputs(3),2,{'target_class':'cup'})])
        self.assertEqual(set(batch),{'visual','geometry','text','scene','valid'})
        self.assertEqual(targets.tolist(),[2])

    def test_sequence_membership_is_disjoint(self):
        sources = {s:[dict(group=f'g{i}',key=f'{s}{i}') for i in range(20)]
                   for s in ('train','val','test')}
        splits = make_splits(sources,'sequence',42)
        groups = {s:{r['group'] for r in rows} for s,rows in splits.items()}
        self.assertFalse(groups['train'] & groups['val'])
        self.assertFalse(groups['train'] & groups['test'])
        self.assertFalse(groups['val'] & groups['test'])
        self.assertEqual(sum(map(len,splits.values())),60)
        self.assertEqual(splits,make_splits(sources,'sequence',42))

    def test_mismatched_and_leaking_splits_rejected(self):
        train = SimpleNamespace(index=dict(cache_id='x',mode='sequence',fingerprint='s',split='train'),rows=[dict(key='a',group='g')])
        val = SimpleNamespace(index=dict(cache_id='x',mode='sequence',fingerprint='s',split='val'),rows=[dict(key='b',group='g')])
        with self.assertRaisesRegex(ValueError,'leakage'):
            verify_pair(train,val)
        val.rows[0]['group']='h'
        verify_pair(train,val)
        val.index['split']='test'
        with self.assertRaisesRegex(ValueError,'never use test'):
            verify_pair(train,val)

    def test_resident_batches_preserve_features_answers_and_partial_batch(self):
        records = {
            's1': dict(ids=[2,8],visual=torch.randn(2,768),geometry=torch.rand(2,7),scene=torch.randn(768)),
            's2': dict(ids=[1,4,9],visual=torch.randn(3,768),geometry=torch.rand(3,7),scene=torch.randn(768))}
        rows = [dict(key='a',scene='s1',target_id=8,target_class='cup',sentence='cup left'),
                dict(key='b',scene='s2',target_id=9,target_class='box',sentence='box right'),
                dict(key='c',scene='s1',target_id=2,target_class='cup',sentence='other cup')]
        class FakeDataset:
            def __len__(self):
                return 3
        data = FakeDataset()
        data.rows=rows
        data.texts=torch.randn(3,768).numpy()
        data.classes={'s1':{2:'cup',8:'cup'},'s2':{9:'box'}}
        data.scene_cache={}
        data.scene_record=lambda s:records[s]
        resident=ResidentFeatures(data,storage='cpu',device='cpu')
        batches=list(resident.loader(2))
        self.assertEqual([len(b[1]) for b in batches],[2,1])
        self.assertEqual(batches[0][1].tolist(),[1,2])
        self.assertEqual(batches[1][1].tolist(),[0])
        self.assertTrue(torch.equal(batches[0][0]['visual'][0,:2],records['s1']['visual']))
        self.assertEqual(batches[0][2][0]['candidate_ids'],[2,8])
        self.assertTrue(batches[0][2][0]['duplicate'])
        self.assertFalse(batches[0][0]['valid'][0,2])
        seen=[m['key'] for _,_,meta in resident.loader(2,shuffle=True) for m in meta]
        self.assertEqual(sorted(seen),['a','b','c'])
        self.assertEqual(next(iter(resident.loader(2,metadata=False)))[2],[])
        result=evaluate(None,resident.loader(2),'cpu')
        self.assertEqual(result['groups']['all']['count'],3)


if __name__=='__main__':
    unittest.main()
