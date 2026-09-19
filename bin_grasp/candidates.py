"""Phase 2, Experiment 1: oracle-candidate language grounding.

The OCID integer label image supplies every candidate mask. The referring
expression is a model input, while the annotated target scene-instance ID is
kept only in supervision/debug data. Nothing is cached or written to disk.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from bin_grasp.dataset import OCIDRefDataset, tight_mask_bbox


FORBIDDEN_MODEL_INPUT_KEYS = {
    "scene_instance_id",
    "target_candidate_index",
    "target_mask",
    "target_class_name",
}


class OracleGroundingDataset(Dataset):
    """Return RGB, expression, and every perfect instance-mask candidate.

    ``model_inputs`` is safe to pass to a grounding model. The correct answer
    is available only under ``supervision`` and ``debug_metadata``.
    Candidate masks are generated in RAM and are never saved as PNG files.
    """

    def __init__(
        self,
        annotations_path: str | Path,
        ocid_root: str | Path,
        *,
        image_size: tuple[int, int] | None = None,
        label_cache_items: int = 8,
        validate_files: bool = False,
    ) -> None:
        self.base = OCIDRefDataset(
            annotations_path,
            ocid_root,
            image_size=None,
            return_depth_supervision=False,
            label_cache_items=label_cache_items,
            validate_files=validate_files,
        )
        self.image_size = image_size

        # Used only for human-readable diagnostics, never as a model input.
        classes: dict[str, dict[int, str]] = defaultdict(dict)
        for sample_id in self.base.sample_ids:
            row = self.base.annotations[sample_id]
            classes[row["scene_path"]].setdefault(
                int(row["scene_instance_id"]), row["class"]
            )
        self.scene_classes = dict(classes)

    def __len__(self) -> int:
        return len(self.base)

    @staticmethod
    def _candidate_geometry(
        masks: np.ndarray, boxes: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        """Return normalized [x1,y1,x2,y2,cx,cy,area] per candidate."""

        normalized = boxes.astype(np.float32).copy()
        normalized[:, [0, 2]] /= width
        normalized[:, [1, 3]] /= height
        centers = np.column_stack(
            (
                (normalized[:, 0] + normalized[:, 2]) / 2,
                (normalized[:, 1] + normalized[:, 3]) / 2,
            )
        )
        areas = masks.reshape(masks.shape[0], -1).mean(axis=1, dtype=np.float32)
        return np.column_stack((normalized, centers, areas)).astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        native = self.base._load_native(index)
        annotation = native["annotation"]
        rgb_image: Image.Image = native["rgb_image"]
        label = self.base.label_cache.get(native["paths"].label)

        if self.image_size is not None:
            rgb_image = rgb_image.resize(self.image_size, Image.Resampling.BILINEAR)
            label = np.asarray(
                Image.fromarray(label).resize(
                    self.image_size, Image.Resampling.NEAREST
                )
            )

        height, width = label.shape[:2]
        candidate_ids = [int(value) for value in np.unique(label) if int(value) != 0]
        if not candidate_ids:
            raise ValueError(f"No foreground candidates in {native['paths'].label}")

        candidate_masks = np.stack(
            [label == candidate_id for candidate_id in candidate_ids]
        )
        candidate_boxes = np.asarray(
            [tight_mask_bbox(mask) for mask in candidate_masks], dtype=np.float32
        )
        candidate_geometry = self._candidate_geometry(
            candidate_masks, candidate_boxes, width, height
        )

        target_id = int(annotation["scene_instance_id"])
        if target_id not in candidate_ids:
            raise ValueError(
                f"Target ID {target_id} disappeared from candidate set for "
                f"sample {native['sample_id']}"
            )
        target_index = candidate_ids.index(target_id)

        rgb = np.asarray(rgb_image, dtype=np.uint8).copy()
        image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
        masks_tensor = torch.from_numpy(candidate_masks.copy())

        model_inputs = {
            "image": image_tensor,
            "sentence": annotation["sentence"],
            "candidate_masks": masks_tensor,
            "candidate_boxes": torch.from_numpy(candidate_boxes),
            "candidate_geometry": torch.from_numpy(candidate_geometry),
        }
        leaked = FORBIDDEN_MODEL_INPUT_KEYS.intersection(model_inputs)
        if leaked:
            raise RuntimeError(f"Target leakage in model inputs: {sorted(leaked)}")

        class_catalog = self.scene_classes.get(annotation["scene_path"], {})
        candidate_class_names = [
            class_catalog.get(candidate_id, "unknown") for candidate_id in candidate_ids
        ]

        return {
            "model_inputs": model_inputs,
            "supervision": {
                "target_candidate_index": torch.tensor(target_index, dtype=torch.long),
                "target_mask": masks_tensor[target_index],
            },
            "debug_metadata": {
                "sample_id": native["sample_id"],
                "scene_path": annotation["scene_path"],
                "candidate_scene_instance_ids": candidate_ids,
                "candidate_class_names": candidate_class_names,
                "target_scene_instance_id": target_id,
                "target_class_name": annotation["class"],
            },
        }


def experiment1_collate(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep samples as a list because scenes have different candidate counts."""

    return samples


def inspect_sample(dataset: OracleGroundingDataset, sample_id: str) -> dict[str, Any]:
    try:
        index = dataset.base.sample_ids.index(str(sample_id))
    except ValueError as exc:
        raise KeyError(f"Unknown sample ID: {sample_id}") from exc

    sample = dataset[index]
    inputs = sample["model_inputs"]
    supervision = sample["supervision"]
    debug = sample["debug_metadata"]
    target_index = int(supervision["target_candidate_index"])
    selected_candidate = inputs["candidate_masks"][target_index]

    return {
        "sample_id": debug["sample_id"],
        "sentence_given_to_model": inputs["sentence"],
        "model_input_keys": sorted(inputs),
        "candidate_count": int(inputs["candidate_masks"].shape[0]),
        "candidate_scene_instance_ids_debug_only": debug[
            "candidate_scene_instance_ids"
        ],
        "candidate_class_names_debug_only": debug["candidate_class_names"],
        "target_candidate_index_supervision_only": target_index,
        "target_scene_instance_id_debug_only": debug["target_scene_instance_id"],
        "candidate_at_target_index_matches_target_mask": bool(
            torch.equal(selected_candidate, supervision["target_mask"])
        ),
        "target_leakage_detected": bool(
            FORBIDDEN_MODEL_INPUT_KEYS.intersection(inputs)
        ),
        "disk_outputs_created": 0,
    }


def audit(dataset: OracleGroundingDataset, count: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(count, len(dataset)))
    failures: list[dict[str, str]] = []
    candidate_counts: list[int] = []
    unknown_classes = 0
    total_candidates = 0

    for index in indices:
        try:
            sample = dataset[index]
            inputs = sample["model_inputs"]
            supervision = sample["supervision"]
            debug = sample["debug_metadata"]
            target_index = int(supervision["target_candidate_index"])
            if FORBIDDEN_MODEL_INPUT_KEYS.intersection(inputs):
                raise AssertionError("Target supervision leaked into model_inputs")
            if not torch.equal(
                inputs["candidate_masks"][target_index],
                supervision["target_mask"],
            ):
                raise AssertionError("Target index does not select target mask")
            if (
                debug["candidate_scene_instance_ids"][target_index]
                != debug["target_scene_instance_id"]
            ):
                raise AssertionError("Target index does not select target ID")

            candidate_count = int(inputs["candidate_masks"].shape[0])
            candidate_counts.append(candidate_count)
            total_candidates += candidate_count
            unknown_classes += debug["candidate_class_names"].count("unknown")
        except Exception as exc:
            failures.append({"dataset_index": str(index), "error": str(exc)})

    return {
        "requested_samples": count,
        "evaluated_samples": len(candidate_counts),
        "failed_samples": len(failures),
        "target_leakage_failures": sum(
            "leaked" in row["error"].lower() for row in failures
        ),
        "candidate_count_min": min(candidate_counts) if candidate_counts else 0,
        "candidate_count_mean": (
            sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0
        ),
        "candidate_count_max": max(candidate_counts) if candidate_counts else 0,
        "candidate_classes_known_ratio_debug_only": (
            (total_candidates - unknown_classes) / total_candidates
            if total_candidates
            else 0
        ),
        "failures_preview": failures[:20],
        "storage_policy": "Candidates were generated in RAM; no masks were saved.",
        "important_note": (
            "This validates the Experiment 1 data interface. A grounding model "
            "must still be trained before expression-selection accuracy exists."
        ),
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the storage-efficient oracle-candidate grounding interface."
    )
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--ocid-root", required=True, type=Path)
    parser.add_argument("--inspect-sample")
    parser.add_argument("--audit", type=int, default=0, metavar="N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--label-cache-items", type=int, default=8)
    args = parser.parse_args()

    dataset = OracleGroundingDataset(
        args.annotations,
        args.ocid_root,
        image_size=tuple(args.image_size) if args.image_size else None,
        label_cache_items=args.label_cache_items,
        validate_files=True,
    )
    print(f"Loaded {len(dataset):,} referring expressions")

    if args.inspect_sample is not None:
        print(json.dumps(inspect_sample(dataset, args.inspect_sample), indent=2))
    if args.audit:
        print(json.dumps(audit(dataset, args.audit, args.seed), indent=2))


if __name__ == "__main__":
    main()
