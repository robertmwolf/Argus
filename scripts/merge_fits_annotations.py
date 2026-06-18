"""Merge GTImages and a reviewed Frigate COCO file into a combined corpus.

Combines:
  - GTImages real training annotations (positives + existing negatives)
  - A reviewed Frigate corpus

The old ``gtimages_plus_frigate_train.json`` was built from stale Frigate
negative labels and has been archived. Do not recreate that artifact from
unreviewed Frigate frames.

Validation and test splits are kept as GTImages-only; Frigate is never mixed
into held-out evaluation sets so the test distribution stays consistent across
all tracks.

Outputs:
  /Volumes/External/TrainingData/annotations/gtimages_plus_frigate_reviewed.json

Usage:
  python scripts/merge_fits_annotations.py
  python scripts/merge_fits_annotations.py \\
      --gtimages data/annotations/gtimages_train_real.json \\
      --frigate  /Volumes/External/TrainingData/annotations/frigate_streaks.json \\
      --output   /Volumes/External/TrainingData/annotations/gtimages_plus_frigate_reviewed.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def merge(
    gtimages_path: Path,
    frigate_path: Path,
) -> dict:
    """Merge two COCO dicts, reassigning image and annotation IDs.

    Args:
        gtimages_path: Path to GTImages COCO training JSON.
        frigate_path: Path to reviewed Frigate COCO JSON.

    Returns:
        Merged COCO dict with unified, sequential IDs.
    """
    gt_coco = _read_json(gtimages_path)
    fr_coco = _read_json(frigate_path)

    merged_images: list[dict] = []
    merged_annotations: list[dict] = []

    next_img_id = 1
    next_ann_id = 1

    # GTImages — carries over both positives and its own negatives.
    gt_img_remap: dict[int, int] = {}
    for img in gt_coco.get("images", []):
        new_id = next_img_id
        gt_img_remap[int(img["id"])] = new_id
        merged_images.append({**img, "id": new_id, "source": "GTImages"})
        next_img_id += 1

    for ann in gt_coco.get("annotations", []):
        new_img_id = gt_img_remap[int(ann["image_id"])]
        merged_annotations.append({**ann, "id": next_ann_id, "image_id": new_img_id})
        next_ann_id += 1

    # Frigate — carry over reviewed positives and any reviewed negatives.
    fr_img_remap: dict[int, int] = {}
    for img in fr_coco.get("images", []):
        new_id = next_img_id
        fr_img_remap[int(img["id"])] = new_id
        merged_images.append({
            "id": new_id,
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
            "source": "Frigate",
        })
        next_img_id += 1

    for ann in fr_coco.get("annotations", []):
        new_img_id = fr_img_remap[int(ann["image_id"])]
        merged_annotations.append({**ann, "id": next_ann_id, "image_id": new_img_id})
        next_ann_id += 1

    gt_positives = sum(
        1 for a in gt_coco.get("annotations", []) if a.get("category_id", 1) == 1
    )
    gt_images = len(gt_coco.get("images", []))
    fr_positives = sum(
        1 for a in fr_coco.get("annotations", []) if a.get("category_id", 1) == 1
    )
    fr_images = len(fr_coco.get("images", []))

    logger.info(
        "Merged: %d GTImages frames (%d streak annotations) + "
        "%d Frigate frames (%d streak annotations) "
        "= %d total frames, %d annotations",
        gt_images, gt_positives, fr_images, fr_positives,
        len(merged_images), len(merged_annotations),
    )

    return {
        "info": {
            "description": "GTImages (real, streak-annotated) + reviewed Frigate",
            "gtimages_source": str(gtimages_path),
            "frigate_source": str(frigate_path),
            "gtimages_frames": gt_images,
            "gtimages_streak_annotations": gt_positives,
            "frigate_frames": fr_images,
            "frigate_streak_annotations": fr_positives,
        },
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": gt_coco.get("categories", [
            {"id": 1, "name": "satellite_streak", "supercategory": "streak"}
        ]),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gtimages",
        type=Path,
        default=Path("data/annotations/gtimages_train_real.json"),
        help="GTImages training COCO JSON (real annotations only, no synthetic).",
    )
    parser.add_argument(
        "--frigate",
        type=Path,
        default=Path("/Volumes/External/TrainingData/annotations/frigate_streaks.json"),
        help="Reviewed Frigate COCO JSON. Do not pass archived stale negatives.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/Volumes/External/TrainingData/annotations/"
            "gtimages_plus_frigate_reviewed.json"
        ),
        help="Output merged COCO JSON path.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    for p, label in [(args.gtimages, "GTImages"), (args.frigate, "Frigate")]:
        if not p.exists():
            raise SystemExit(
                f"{label} annotation file not found: {p}\n"
                f"  GTImages: run  python scripts/convert_gtimages.py\n"
                f"  Frigate:  run  python scripts/annotate_frigate.py"
            )

    merged = merge(args.gtimages, args.frigate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2))
    logger.info("Wrote merged corpus → %s", args.output)


if __name__ == "__main__":
    main()
