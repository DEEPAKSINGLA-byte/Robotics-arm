"""Shared masked-crop recipe for oracle and predicted object candidates.

Keep this recipe identical in feature extraction and inference so existing
cached features and trained checkpoints remain compatible.
"""
import numpy as np
from PIL import Image

from phase1_dataset import tight_mask_bbox
from phase2_experiment1 import OracleGroundingDataset


def candidate_inputs(rgb, masks):
    """Return square gray-padded PIL crops and normalized NumPy geometry."""
    if not len(masks):
        raise ValueError('Cannot encode an empty proposal set')
    if masks.shape[1:] != rgb.shape[:2]:
        raise ValueError('Candidate masks and RGB dimensions differ')
    boxes = np.asarray([tight_mask_bbox(mask) for mask in masks], dtype=np.float32)
    geometry = OracleGroundingDataset._candidate_geometry(
        masks, boxes, rgb.shape[1], rgb.shape[0])
    crops = []
    for mask, box in zip(masks, boxes):
        x1, y1, x2, y2 = box.astype(int)
        crop = rgb[y1:y2, x1:x2].copy()
        crop[~mask[y1:y2, x1:x2]] = 127
        side = max(crop.shape[:2])
        canvas = np.full((side, side, 3), 127, dtype=np.uint8)
        y, x = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
        canvas[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
        crops.append(Image.fromarray(canvas))
    return crops, geometry
