"""Build full-frame val/holdout splits for centerline heatmap experiments."""

from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path
import random
from typing import Any

logger = logging.getLogger(__name__)


def _load_coco(path: Path) -> dict[str, Any]:
    """Load one COCO annotation file."""
    return json.loads(path.read_text())


def _merge_coco(paths: list[Path]) -> dict[str, Any]:
    """Merge COCO files while assigning fresh image and annotation IDs."""
    merged: dict[str, Any] = {
        "info": {},
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": [],
    }
    next_image_id = 1
    next_ann_id = 1
    seen_categories: dict[int, dict[str, Any]] = {}
    for path in paths:
        coco = _load_coco(path)
        for category in coco.get("categories", []):
            seen_categories[int(category["id"])] = category
        id_map: dict[int, int] = {}
        for image in coco.get("images", []):
            new_image = deepcopy(image)
            old_id = int(image["id"])
            id_map[old_id] = next_image_id
            new_image["id"] = next_image_id
            next_image_id += 1
            merged["images"].append(new_image)
        for ann in coco.get("annotations", []):
            old_image_id = int(ann["image_id"])
            if old_image_id not in id_map:
                continue
            new_ann = deepcopy(ann)
            new_ann["id"] = next_ann_id
            next_ann_id += 1
            new_ann["image_id"] = id_map[old_image_id]
            merged["annotations"].append(new_ann)
    merged["categories"] = list(seen_categories.values()) or [{"id": 1, "name": "streak"}]
    return merged


def _subset_coco(coco: dict[str, Any], image_ids: set[int]) -> dict[str, Any]:
    """Return a COCO subset containing only selected image IDs."""
    subset = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco.get("categories", [{"id": 1, "name": "streak"}]),
        "images": [image for image in coco.get("images", []) if int(image["id"]) in image_ids],
        "annotations": [
            ann for ann in coco.get("annotations", []) if int(ann["image_id"]) in image_ids
        ],
    }
    return subset


def _sample_ids(
    pool: list[int],
    count: int,
    rng: random.Random,
    label: str,
    allow_shortfall: bool,
) -> list[int]:
    """Sample IDs or raise a clear insufficient-data error."""
    if len(pool) < count:
        if allow_shortfall:
            logger.warning(
                "Not enough %s images: requested %d, available %d; using all available",
                label,
                count,
                len(pool),
            )
            selected = list(pool)
            rng.shuffle(selected)
            return selected
        raise ValueError(
            f"Not enough {label} images: requested {count}, available {len(pool)}. "
            "Provide more negative full-frame source images or reduce the requested split counts."
        )
    selected = list(pool)
    rng.shuffle(selected)
    return selected[:count]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/annotations/train.json",
            "data/annotations/val.json",
            "data/annotations/test.json",
        ],
    )
    parser.add_argument("--output-dir", default="data/annotations/centerline_fullframe_splits")
    parser.add_argument("--val-positive", type=int, default=78)
    parser.add_argument("--val-negative", type=int, default=160)
    parser.add_argument("--holdout-positive", type=int, default=71)
    parser.add_argument("--holdout-negative", type=int, default=216)
    parser.add_argument("--allow-shortfall", action="store_true")
    parser.add_argument("--seed", type=int, default=20260524)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rng = random.Random(args.seed)
    coco = _merge_coco([Path(path) for path in args.inputs])
    ann_image_ids = {int(ann["image_id"]) for ann in coco.get("annotations", [])}
    positive_ids = [int(image["id"]) for image in coco["images"] if int(image["id"]) in ann_image_ids]
    negative_ids = [int(image["id"]) for image in coco["images"] if int(image["id"]) not in ann_image_ids]
    logger.info(
        "source pool images=%d positives=%d negatives=%d",
        len(coco["images"]),
        len(positive_ids),
        len(negative_ids),
    )

    val_pos = _sample_ids(positive_ids, args.val_positive, rng, "positive val", args.allow_shortfall)
    remaining_pos = [image_id for image_id in positive_ids if image_id not in set(val_pos)]
    holdout_pos = _sample_ids(
        remaining_pos,
        args.holdout_positive,
        rng,
        "positive holdout",
        args.allow_shortfall,
    )

    val_neg = _sample_ids(negative_ids, args.val_negative, rng, "negative val", args.allow_shortfall)
    remaining_neg = [image_id for image_id in negative_ids if image_id not in set(val_neg)]
    holdout_neg = _sample_ids(
        remaining_neg,
        args.holdout_negative,
        rng,
        "negative holdout",
        args.allow_shortfall,
    )

    val_ids = set(val_pos + val_neg)
    holdout_ids = set(holdout_pos + holdout_neg)
    train_ids = {int(image["id"]) for image in coco["images"]} - val_ids - holdout_ids

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / "train_source.json",
        "val": output_dir / "val_fullframe.json",
        "holdout": output_dir / "holdout_fullframe.json",
    }
    paths["train"].write_text(json.dumps(_subset_coco(coco, train_ids), indent=2))
    paths["val"].write_text(json.dumps(_subset_coco(coco, val_ids), indent=2))
    paths["holdout"].write_text(json.dumps(_subset_coco(coco, holdout_ids), indent=2))
    report = {
        "source_images": len(coco["images"]),
        "source_positive": len(positive_ids),
        "source_negative": len(negative_ids),
        "train_images": len(train_ids),
        "val_positive": len(val_pos),
        "val_negative": len(val_neg),
        "holdout_positive": len(holdout_pos),
        "holdout_negative": len(holdout_neg),
        "paths": {key: str(path) for key, path in paths.items()},
    }
    (output_dir / "split_report.json").write_text(json.dumps(report, indent=2))
    logger.info("wrote centerline full-frame splits to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
