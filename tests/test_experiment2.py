import unittest
from unittest.mock import patch
import numpy as np
import torch
from experiments.run_experiment2 import select_rows, validate_checkpoint, filter_proposals, prepare_candidates, summarize_evaluation, StableMaskGenerationPipeline, MaskGenerationPipeline


class Experiment2Tests(unittest.TestCase):
    def test_mixed_precision_cleanup_uses_matching_fp32(self):
        outputs=[dict(iou_scores=torch.tensor([.9],dtype=torch.bfloat16),boxes=torch.ones(1,4,dtype=torch.int64))]
        wrapper=object.__new__(StableMaskGenerationPipeline)
        with patch.object(MaskGenerationPipeline,'postprocess',return_value={'ok':True}) as parent:
            result=wrapper.postprocess(outputs,crops_nms_thresh=.7)
        self.assertEqual(result,{'ok':True})
        self.assertEqual(outputs[0]['iou_scores'].dtype,torch.float32)
        self.assertEqual(outputs[0]['boxes'].dtype,torch.float32)
        parent.assert_called_once_with(outputs,crops_nms_thresh=.7)

    def test_distinct_scene_selection_without_correctness(self):
        rows=[dict(scene=f'{g}/{s}',group=str(g),key=f'{g}/{s}/{t}') for g in range(7) for s in range(5) for t in range(2)]
        index=dict(split='val',mode='sequence',rows=rows)
        selected=select_rows(index,20,42)
        self.assertEqual(selected,select_rows(index,20,42))
        self.assertEqual(len({r['scene'] for r in selected}),20)
        self.assertEqual(len({r['group'] for r in selected}),7)
        with self.assertRaises(ValueError): select_rows(dict(index,split='test'),20,42)

    def test_checkpoint_split_guard(self):
        index=dict(cache_id='a',fingerprint='b',mode='sequence',rows=[dict(group='val')])
        checkpoint=dict(cache_id='a',fingerprint='b',mode='sequence',overfit=False,training_groups=['train'])
        validate_checkpoint(checkpoint,index)
        with self.assertRaises(ValueError): validate_checkpoint(dict(checkpoint,training_groups=['val']),index)
        with self.assertRaises(ValueError): validate_checkpoint(dict(checkpoint,overfit=True),index)

    def test_proposal_filter_keeps_distinct_overlapping_objects(self):
        masks=np.zeros((5,10,10),dtype=bool)
        masks[0,1:5,1:5]=True
        masks[1]=masks[0]
        masks[2,3:7,3:7]=True
        masks[3]=True
        masks[4,0,0]=True
        keep,stats=filter_proposals(masks,np.array([.9,.8,.7,.95,.6]),(10,10),min_area=2)
        self.assertEqual(keep,[0,2])
        self.assertEqual(stats['duplicate_masks_removed'],1)
        self.assertEqual(stats['rejected_area_or_nonfinite'],2)
        empty,stats=filter_proposals(np.zeros((0,10,10),dtype=bool),np.zeros(0),(10,10))
        self.assertEqual(empty,[])

    def test_crop_geometry_matches_training_convention(self):
        rgb=np.full((8,10,3),240,dtype=np.uint8)
        mask=np.zeros((1,8,10),dtype=bool)
        mask[0,2:6,3:5]=True
        crops,geometry=prepare_candidates(rgb,mask)
        self.assertEqual(crops[0].size,(4,4))
        pixels=np.asarray(crops[0])
        self.assertTrue((pixels[:,0]==127).all())
        self.assertTrue((pixels[:,1:3]==240).all())
        np.testing.assert_allclose(geometry.numpy()[0],[.3,.25,.5,.75,.4,.5,.1],rtol=1e-6)

    def test_failure_separation_and_empty_proposals(self):
        masks=np.zeros((2,8,8),dtype=bool)
        masks[0,1:3,1:3]=True
        masks[1,5:7,5:7]=True
        target=masks[0]
        good=summarize_evaluation(masks,[0,1],0,1,target)
        self.assertEqual(good['failure'],'success')
        bad=summarize_evaluation(masks,[0,1],1,0,target)
        self.assertEqual(bad['failure'],'selection_miss')
        missing=summarize_evaluation(masks,[1],0,0,target)
        self.assertEqual(missing['failure'],'proposal_miss')
        self.assertEqual(missing['raw_best_iou'],1)
        empty=summarize_evaluation(np.zeros((0,8,8),dtype=bool),[],None,None,target)
        self.assertEqual(empty['failure'],'proposal_miss')
        self.assertEqual(empty['selected_iou'],0)


if __name__=='__main__': unittest.main()
