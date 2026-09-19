import unittest
import numpy as np
from experiments.build_error_review import select_errors, outline, inside
from pathlib import Path


class ReviewTests(unittest.TestCase):
    def fixture(self):
        rows=[]
        predictions=[]
        for group in range(7):
            for scene in range(10):
                for expression in range(2):
                    key=f'{group}-{scene}-{expression}'
                    rows.append(dict(key=key,group=str(group),scene=f'{group}/{scene}',target_id=1))
                    predictions.append(dict(key=key,correct=False,target_index=0,predicted_index=1,predicted_scene_instance_id=2))
        return dict(split='val',rows=rows),predictions

    def test_distinct_balanced_repeatable(self):
        index,predictions=self.fixture()
        first,stats=select_errors(index,predictions)
        second,_=select_errors(index,predictions)
        self.assertEqual(first,second)
        self.assertEqual(len({r['scene'] for r,p in first}),50)
        self.assertEqual(sorted(stats['selected_by_group'].values()),[7,7,7,7,7,7,8])

    def test_test_split_and_bad_predictions_rejected(self):
        index,predictions=self.fixture()
        index['split']='test'
        with self.assertRaises(ValueError): select_errors(index,predictions)
        index['split']='val'
        with self.assertRaises(ValueError): select_errors(index,predictions[:-1])
        with self.assertRaises(ValueError): select_errors(index,predictions+[predictions[0]])
        predictions[0]['predicted_scene_instance_id']=1
        with self.assertRaises(ValueError): select_errors(index,predictions)

    def test_outline_exact_edges_and_box(self):
        mask=np.zeros((8,9),dtype=bool)
        mask[2:6,3:7]=True
        runs,box=outline(mask)
        edge=np.zeros_like(mask)
        for x,y,n in runs: edge[y,x:x+n]=True
        expected=mask.copy()
        expected[3:5,4:6]=False
        np.testing.assert_array_equal(edge,expected)
        self.assertEqual(box,[3,2,7,6])
        with self.assertRaises(ValueError): outline(np.zeros((2,2),dtype=bool))
        with self.assertRaises(ValueError): inside(Path('/tmp/report-root'),'../escape')


if __name__=='__main__': unittest.main()
