"""Storage-efficient Phase 1 loader for OCID-Ref.

This module never duplicates RGB, depth, label, or point-cloud files. Target
masks are generated in memory from the OCID integer label image and the
OCID-Ref ``scene_instance_id`` field.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class ResolvedPaths:
    rgb: Path
    label: Path
    depth: Path
    pcd: Path


class ArrayLRU:
    """Small in-memory cache; it never writes cache files to disk."""

    def __init__(self, max_items: int = 8) -> None:
        self.max_items = max(0, int(max_items))
        self._items: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, path: Path) -> np.ndarray:
        key = str(path)
        if self.max_items and key in self._items:
            value = self._items.pop(key)
            self._items[key] = value
            return value

        value = np.asarray(Image.open(path))
        if self.max_items:
            self._items[key] = value
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)
        return value


def parse_bbox(value: Any) -> tuple[int, int, int, int]:
    """Parse the actual OCID-Ref format: [x1, y1, x2, y2]."""

    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"Expected four bbox values, received {value!r}")
    x1, y1, x2, y2 = map(int, value)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid xyxy bbox: {value!r}")
    return x1, y1, x2, y2


def resolve_paths(ocid_root: Path, scene_path: str) -> ResolvedPaths:
    """Resolve one annotation without copying any dataset asset."""

    relative = Path(scene_path.replace("\\", "/"))
    parts = list(relative.parts)
    try:
        modality_index = parts.index("rgb")
    except ValueError as exc:
        raise ValueError(f"scene_path has no rgb directory: {scene_path}") from exc

    def replace_modality(name: str, suffix: str = ".png") -> Path:
        replaced = parts.copy()
        replaced[modality_index] = name
        result = Path(*replaced)
        return result.with_suffix(suffix)

    return ResolvedPaths(
        rgb=ocid_root / relative,
        label=ocid_root / replace_modality("label"),
        depth=ocid_root / replace_modality("depth"),
        pcd=ocid_root / replace_modality("pcd", ".pcd"),
    )


def clamp_bbox(
    bbox: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )


def tight_mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("Cannot calculate a bounding box for an empty mask")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def rectangle_iou(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


class OCIDRefDataset(Dataset):
    """RGB-language dataset that creates target masks only in RAM.

    The complete JSON is loaded once. No processed image dataset, duplicate
    masks, token cache, or preview directory is created.
    """

    def __init__(
        self,
        annotations_path: str | Path,
        ocid_root: str | Path,
        *,
        image_size: tuple[int, int] | None = None,
        return_depth_supervision: bool = False,
        label_cache_items: int = 8,
        validate_files: bool = False,
    ) -> None:
        self.annotations_path = Path(annotations_path).expanduser().resolve()
        self.ocid_root = Path(ocid_root).expanduser().resolve()
        self.image_size = image_size
        self.return_depth_supervision = return_depth_supervision
        self.label_cache = ArrayLRU(label_cache_items)

        with self.annotations_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)

        if isinstance(raw, dict):
            self.sample_ids = list(raw.keys())
            self.annotations = raw
        elif isinstance(raw, list):
            self.sample_ids = [str(item.get("sentence_id", i)) for i, item in enumerate(raw)]
            self.annotations = dict(zip(self.sample_ids, raw))
        else:
            raise TypeError("Annotation JSON must contain an object or a list")

        if validate_files and self.sample_ids:
            for index in (0, len(self.sample_ids) // 2, len(self.sample_ids) - 1):
                annotation = self.annotations[self.sample_ids[index]]
                paths = resolve_paths(self.ocid_root, annotation["scene_path"])
                if not paths.rgb.is_file() or not paths.label.is_file():
                    raise FileNotFoundError(
                        f"Dataset path check failed for {annotation['scene_path']}"
                    )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _load_native(self, index: int) -> dict[str, Any]:
        sample_id = self.sample_ids[index]
        annotation = self.annotations[sample_id]
        paths = resolve_paths(self.ocid_root, annotation["scene_path"])

        if not paths.rgb.is_file():
            raise FileNotFoundError(paths.rgb)
        if not paths.label.is_file():
            raise FileNotFoundError(paths.label)

        rgb_image = Image.open(paths.rgb).convert("RGB")
        label = self.label_cache.get(paths.label)
        if label.shape[:2] != (rgb_image.height, rgb_image.width):
            raise ValueError(
                f"RGB/label shape mismatch: RGB={rgb_image.size}, label={label.shape}"
            )

        local_id = int(annotation["scene_instance_id"])
        target_mask = label == local_id
        if not target_mask.any():
            raise ValueError(
                f"scene_instance_id={local_id} is absent from {paths.label}"
            )

        bbox = parse_bbox(annotation["bbox"])
        bounded_bbox = clamp_bbox(bbox, rgb_image.width, rgb_image.height)
        if bounded_bbox[2] <= bounded_bbox[0] or bounded_bbox[3] <= bounded_bbox[1]:
            raise ValueError(f"Bounding box is outside the image: {bbox}")

        return {
            "sample_id": sample_id,
            "annotation": annotation,
            "paths": paths,
            "rgb_image": rgb_image,
            "target_mask": target_mask,
            "bbox": bounded_bbox,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        native = self._load_native(index)
        annotation = native["annotation"]
        rgb_image: Image.Image = native["rgb_image"]
        target_mask: np.ndarray = native["target_mask"]
        x1, y1, x2, y2 = native["bbox"]
        original_size = (rgb_image.width, rgb_image.height)

        if self.image_size is not None:
            target_width, target_height = self.image_size
            scale_x = target_width / rgb_image.width
            scale_y = target_height / rgb_image.height
            rgb_image = rgb_image.resize(
                (target_width, target_height), Image.Resampling.BILINEAR
            )
            mask_image = Image.fromarray(target_mask.astype(np.uint8) * 255)
            mask_image = mask_image.resize(
                (target_width, target_height), Image.Resampling.NEAREST
            )
            target_mask = np.asarray(mask_image) > 0
            x1, x2 = round(x1 * scale_x), round(x2 * scale_x)
            y1, y2 = round(y1 * scale_y), round(y2 * scale_y)

        rgb_array = np.asarray(rgb_image, dtype=np.uint8).copy()
        image_tensor = torch.from_numpy(rgb_array).permute(2, 0, 1).float().div_(255.0)
        mask_tensor = torch.from_numpy(target_mask.copy()).unsqueeze(0)

        output: dict[str, Any] = {
            "image": image_tensor,
            "sentence": annotation["sentence"],
            "target_mask": mask_tensor,
            "bbox": torch.tensor([x1, y1, x2, y2], dtype=torch.float32),
            "sample_id": native["sample_id"],
            "scene_path": annotation["scene_path"],
            "instance_id": int(annotation["instance_id"]),
            "scene_instance_id": int(annotation["scene_instance_id"]),
            "class_name": annotation["class"],
            "class_instance": annotation.get("class_instance", ""),
            "sub_dataset": annotation.get("sub_dataset", ""),
            "sequence_path": annotation.get("sequence_path", ""),
            "original_size": torch.tensor(original_size, dtype=torch.int32),
            "rgb_path": str(native["paths"].rgb),
            "label_path": str(native["paths"].label),
            "depth_path": str(native["paths"].depth),
            "pcd_path": str(native["paths"].pcd),
        }

        if self.return_depth_supervision:
            depth_path = native["paths"].depth
            if not depth_path.is_file():
                raise FileNotFoundError(depth_path)
            depth = np.asarray(Image.open(depth_path), dtype=np.uint16)
            if self.image_size is not None:
                depth_image = Image.fromarray(depth)
                depth_image = depth_image.resize(
                    self.image_size, Image.Resampling.NEAREST
                )
                depth = np.asarray(depth_image, dtype=np.uint16)
            output["depth_mm"] = torch.from_numpy(depth.copy()).unsqueeze(0)

        return output

    def validation_metrics(self, index: int) -> dict[str, Any]:
        native = self._load_native(index)
        mask = native["target_mask"]
        bbox = native["bbox"]
        x1, y1, x2, y2 = bbox
        target_pixels = int(mask.sum())
        target_inside = int(mask[y1:y2, x1:x2].sum())
        bbox_area = (x2 - x1) * (y2 - y1)
        mask_bbox = tight_mask_bbox(mask)

        return {
            "sample_id": native["sample_id"],
            "scene_path": native["annotation"]["scene_path"],
            "scene_instance_id": int(native["annotation"]["scene_instance_id"]),
            "class_name": native["annotation"]["class"],
            "target_pixels": target_pixels,
            "mask_inside_bbox_ratio": target_inside / target_pixels,
            "bbox_covered_by_mask_ratio": target_inside / bbox_area,
            "bbox_vs_tight_bbox_iou": rectangle_iou(bbox, mask_bbox),
            "bbox": list(bbox),
            "tight_mask_bbox": list(mask_bbox),
        }


def make_dataloader(
    dataset: OCIDRefDataset,
    *,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a conservative loader suitable for a removable USB drive."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def audit_dataset(
    dataset: OCIDRefDataset, sample_count: int, seed: int
) -> dict[str, Any]:
    rng = random.Random(seed)
    count = min(sample_count, len(dataset))
    indices = rng.sample(range(len(dataset)), count)

    metrics: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index in indices:
        try:
            metrics.append(dataset.validation_metrics(index))
        except Exception as exc:  # Audit should report bad rows instead of stopping.
            failures.append({"index": str(index), "error": str(exc)})

    containment = [row["mask_inside_bbox_ratio"] for row in metrics]
    box_iou = [row["bbox_vs_tight_bbox_iou"] for row in metrics]
    unique_targets = {
        (row["scene_path"], row["scene_instance_id"]) for row in metrics
    }
    unique_images = {row["scene_path"] for row in metrics}

    summary = {
        "requested_samples": sample_count,
        "evaluated_samples": len(metrics),
        "failed_samples": len(failures),
        "unique_targets": len(unique_targets),
        "unique_images": len(unique_images),
        "duplicate_expression_rows": len(metrics) - len(unique_targets),
        "mask_inside_bbox_mean": mean(containment),
        "mask_inside_bbox_min": min(containment) if containment else 0.0,
        "bbox_vs_tight_bbox_iou_mean": mean(box_iou),
        "bbox_vs_tight_bbox_iou_min": min(box_iou) if box_iou else 0.0,
        "below_0_98_containment": sum(value < 0.98 for value in containment),
        "seed": seed,
        "failures_preview": failures[:20],
        "storage_policy": "No images, masks, previews, or CSV reports were written.",
    }
    return summary


def compare_annotation_splits(paths: list[Path]) -> dict[str, Any]:
    """Measure scene/target overlap between annotation files without disk output."""

    target_sets: dict[str, set[tuple[str, int]]] = {}
    image_sets: dict[str, set[str]] = {}
    inventories: dict[str, dict[str, int]] = {}

    for path in paths:
        resolved = path.expanduser().resolve()
        with resolved.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        rows = raw.values() if isinstance(raw, dict) else raw
        targets = {
            (row["scene_path"], int(row["scene_instance_id"])) for row in rows
        }
        name = resolved.stem.removesuffix("_expressions")
        target_sets[name] = targets
        image_sets[name] = {scene_path for scene_path, _ in targets}
        inventories[name] = {
            "expressions": len(raw),
            "unique_targets": len(targets),
            "unique_images": len(image_sets[name]),
        }
        del raw

    names = list(target_sets)
    overlaps: dict[str, dict[str, int]] = {}
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            overlaps[f"{first}_vs_{second}"] = {
                "shared_targets": len(target_sets[first] & target_sets[second]),
                "shared_images": len(image_sets[first] & image_sets[second]),
            }

    return {
        "splits": inventories,
        "pairwise_overlap": overlaps,
        "interpretation": (
            "These are expression-level splits when shared targets/images are non-zero. "
            "Do not describe their scores as unseen-scene generalization."
        ),
        "storage_policy": "The comparison is calculated in RAM and writes no files.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Storage-efficient OCID-Ref Phase 1 dataset loader and auditor."
    )
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--ocid-root", type=Path)
    parser.add_argument(
        "--check-splits",
        nargs="+",
        type=Path,
        metavar="JSON",
        help="Compare two or more annotation files, print overlap, and exit.",
    )
    parser.add_argument("--audit", type=int, default=0, metavar="N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inspect-sample", default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--label-cache-items", type=int, default=8)
    args = parser.parse_args()

    if args.check_splits:
        if len(args.check_splits) < 2:
            parser.error("--check-splits requires at least two JSON files")
        print(json.dumps(compare_annotation_splits(args.check_splits), indent=2))
        return

    if args.annotations is None or args.ocid_root is None:
        parser.error("--annotations and --ocid-root are required outside split-check mode")

    dataset = OCIDRefDataset(
        args.annotations,
        args.ocid_root,
        label_cache_items=args.label_cache_items,
        validate_files=True,
    )
    print(f"Loaded {len(dataset):,} referring expressions")

    if args.inspect_sample is not None:
        try:
            index = dataset.sample_ids.index(str(args.inspect_sample))
        except ValueError as exc:
            raise KeyError(f"Unknown sample ID: {args.inspect_sample}") from exc
        sample = dataset[index]
        metrics = dataset.validation_metrics(index)
        print(
            json.dumps(
                {
                    "sample_id": sample["sample_id"],
                    "sentence": sample["sentence"],
                    "class_name": sample["class_name"],
                    "image_shape": list(sample["image"].shape),
                    "mask_shape": list(sample["target_mask"].shape),
                    "metrics": metrics,
                },
                indent=2,
            )
        )

    if args.audit:
        summary = audit_dataset(dataset, args.audit, args.seed)
        print(json.dumps(summary, indent=2))
        if args.summary_json is not None:
            args.summary_json.parent.mkdir(parents=True, exist_ok=True)
            with args.summary_json.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
                handle.write("\n")
            print(f"Wrote compact summary: {args.summary_json}")


if __name__ == "__main__":
    main()
