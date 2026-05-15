"""Generate augmented training data targeting short and medium satellite streaks.

Reads train.json, injects synthetic short and medium streaks into a subset
of training images, and writes:
  - Augmented images  → data/augmented/
  - Combined COCO JSON → data/annotations/train_augmented.json

The combined JSON (original + augmented entries) is a drop-in replacement
for train.json: set USE_DEV_SUBSET=false and the train_dino.py path override
  cfg.train_dataloader.dataset.ann_file = "annotations/train_augmented.json"

Short band  : streaks 50 – 400 px at original image scale
Medium band : streaks 400 – 1000 px at original image scale

Both ranges are well below the typical full-width streaks that dominate the
existing training set, so they directly address the zero-recall gap on short
and medium detections shown in the Phase E per-band evaluation.

Usage:
    python scripts/augment_short_medium.py
    python scripts/augment_short_medium.py --n-images 300 --seed 42
    python scripts/augment_short_medium.py --out-dir data/augmented \
        --ann-out data/annotations/train_augmented.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np

from training.augmentations import SyntheticStreakInject

logger = logging.getLogger(__name__)


def _load_image(img_path: Path) -> np.ndarray | None:
    """Load a JPEG image as uint8 RGB; return None if unreadable."""
    img = cv2.imread(str(img_path))
    if img is None:
        logger.warning("Could not read %s; skipping", img_path)
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _inject_and_save(
    image: np.ndarray,
    injectors: list[SyntheticStreakInject],
    out_path: Path,
) -> list[tuple[float, float, float, float]]:
    """Apply all injectors to the image, save result, return all new bboxes."""
    bboxes: list[tuple[float, float, float, float]] = []
    labels: list[int] = []
    img = image.copy()
    for inj in injectors:
        img, bboxes, labels = inj.inject(img, bboxes, labels)

    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return bboxes


def build_augmented_dataset(
    data_root: Path,
    ann_in: Path,
    ann_out: Path,
    out_dir: Path,
    n_images: int,
    seed: int,
) -> None:
    """Generate augmented images and write combined COCO JSON.

    Args:
        data_root: Root of the data directory (contains ``satstreaks/`` and
            ``annotations/``).
        ann_in: Source COCO annotation file (typically train.json).
        ann_out: Destination for the combined annotation file.
        out_dir: Directory for augmented JPEG images.
        n_images: Number of training images to augment (random sample).
        seed: Random seed for reproducibility.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ann_out.parent.mkdir(parents=True, exist_ok=True)

    with open(ann_in) as f:
        coco = json.load(f)

    rng = random.Random(seed)
    images_pool = coco["images"]
    rng.shuffle(images_pool)
    selected = images_pool[:n_images]

    # Short-streak injector: 50–400 px at original resolution, p=1 always injects
    short_injector = SyntheticStreakInject(p=1.0, min_length_px=50.0, max_length_fraction=0.1)
    # Medium-streak injector: 400–1000 px, fraction of 4096 diagonal ≈ 5792 px
    medium_injector = SyntheticStreakInject(p=1.0, min_length_px=400.0, max_length_fraction=0.17)

    max_img_id = max(img["id"] for img in coco["images"])
    max_ann_id = max((a["id"] for a in coco["annotations"]), default=0)
    next_img_id = max_img_id + 1
    next_ann_id = max_ann_id + 1

    new_images: list[dict] = []
    new_annotations: list[dict] = []

    for i, img_info in enumerate(selected):
        img_path = data_root / img_info["file_name"]
        image = _load_image(img_path)
        if image is None:
            continue

        stem = Path(img_info["file_name"]).stem
        aug_filename = f"augmented/aug_{stem}_{i:04d}.jpg"
        out_path = data_root / aug_filename

        bboxes = _inject_and_save(image, [short_injector, medium_injector], out_path)

        new_img = {
            "id": next_img_id,
            "file_name": aug_filename,
            "width": img_info["width"],
            "height": img_info["height"],
        }
        new_images.append(new_img)

        for x_min, y_min, x_max, y_max in bboxes:
            w = x_max - x_min
            h = y_max - y_min
            cx = x_min + w / 2.0
            cy = y_min + h / 2.0
            new_annotations.append({
                "id": next_ann_id,
                "image_id": next_img_id,
                "category_id": 1,
                "iscrowd": 0,
                "bbox": [float(x_min), float(y_min), float(w), float(h)],
                "area": float(w * h),
                "obb": [cx, cy, max(w, h), min(w, h), 0.0],
                "segmentation_mask": "",
            })
            next_ann_id += 1

        next_img_id += 1

        if (i + 1) % 50 == 0:
            logger.info("Augmented %d / %d images", i + 1, len(selected))

    combined = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": coco["images"] + new_images,
        "annotations": coco["annotations"] + new_annotations,
    }

    with open(ann_out, "w") as f:
        json.dump(combined, f)

    logger.info(
        "Done. Original: %d images / %d annotations. "
        "Added: %d images / %d annotations. "
        "Written to %s",
        len(coco["images"]), len(coco["annotations"]),
        len(new_images), len(new_annotations),
        ann_out,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root", type=Path, default=Path("data"),
        help="Root data directory (default: data/)",
    )
    p.add_argument(
        "--ann-in", type=Path, default=Path("data/annotations/train.json"),
        help="Source COCO annotation file (default: data/annotations/train.json)",
    )
    p.add_argument(
        "--ann-out", type=Path, default=Path("data/annotations/train_augmented.json"),
        help="Output combined annotation file",
    )
    p.add_argument(
        "--out-dir", type=Path, default=Path("data/augmented"),
        help="Directory for augmented images",
    )
    p.add_argument(
        "--n-images", type=int, default=500,
        help="Number of training images to augment (default: 500)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    build_augmented_dataset(
        data_root=args.data_root,
        ann_in=args.ann_in,
        ann_out=args.ann_out,
        out_dir=args.out_dir,
        n_images=args.n_images,
        seed=args.seed,
    )
