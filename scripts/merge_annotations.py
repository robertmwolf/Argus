"""Merge SatStreaks and GTImages annotations into train/val COCO JSON splits.

Reads:
  data/satstreaks/Data/labels.json   — SatStreaks split (train/val/test keys)
  data/annotations/gtimages.json     — GTImages labeled streaks (convert first)
  data/annotations/gtimages_negatives.json — GTImages no-streak images

Writes:
  data/annotations/train.json        — 80% split for training
  data/annotations/val.json          — 20% split for validation
  data/annotations/test.json         — held-out test set (SatStreaks test split)

SatStreaks images reference paths like ``Images/jXXXXXX_flc.fits.jpg``.
GTImages images are written with paths relative to ``data/``, for example
``GTImages/Streak_NNNN_HHMMSS.fits``.

Image ID space:
  SatStreaks images: IDs start at 1
  GTImages images:   IDs start at 1_000_000 (avoids collision)

Usage::

    python scripts/merge_annotations.py [--seed 42] [--val-fraction 0.2]
    python scripts/merge_annotations.py --satstreaks-only   # skip GTImages
    python scripts/merge_annotations.py --gtimages-only     # skip SatStreaks
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SATSTREAKS_LABELS  = Path("data/satstreaks/Data/labels.json")
SATSTREAKS_IMG_DIR = Path("data/satstreaks/Data/Images")
SATSTREAKS_MSK_DIR = Path("data/satstreaks/Data/Masks")
GTIMAGES_JSON      = Path("data/annotations/gtimages.json")
GTIMAGES_NEG_JSON  = Path("data/annotations/gtimages_negatives.json")
OUT_DIR            = Path("data/annotations")

_GTIMAGES_ID_OFFSET = 1_000_000  # keeps GTImages IDs distinct from SatStreaks


# ---------------------------------------------------------------------------
# SatStreaks loader
# ---------------------------------------------------------------------------

def _image_size(image_path: Path) -> tuple[int, int]:
    """Return image dimensions as ``(width, height)``.

    Args:
        image_path: Path to a SatStreaks JPEG/PNG image.

    Returns:
        Image width and height in pixels.
    """
    with Image.open(image_path) as img:
        return img.size


def _bbox_from_mask(mask_path: Path) -> tuple[list[float], float] | None:
    """Derive a COCO bounding box from a SatStreaks segmentation mask.

    Source: SatStreaks — segmentation masks converted to detection boxes for
    DINO fine-tuning.
    Ref: https://github.com/jijup/SatStreaks

    Args:
        mask_path: Path to the binary/object mask image.

    Returns:
        Tuple of ``([x, y, width, height], area)`` when the mask has foreground,
        otherwise ``None``.
    """
    with Image.open(mask_path) as mask_img:
        mask = np.asarray(mask_img.convert("L"))

    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    area = float((mask > 0).sum())
    return [float(x_min), float(y_min), float(width), float(height)], area

def _load_satstreaks(
    labels_path: Path,
    split: str,
    id_offset: int = 0,
    ann_id_offset: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Load one SatStreaks split into COCO images + annotations lists.

    Args:
        labels_path: Path to SatStreaks Data/labels.json.
        split: One of ``"train"``, ``"val"``, ``"test"``.
        id_offset: Added to every image ID (for ID-space separation).
        ann_id_offset: Added to every annotation ID.

    Returns:
        Tuple of (images, annotations) lists in COCO format.
    """
    with open(labels_path) as f:
        data = json.load(f)

    entries = data.get(split, [])
    images: list[dict] = []
    annotations: list[dict] = []
    ann_id = ann_id_offset

    for img_idx, entry in enumerate(entries):
        img_id = id_offset + img_idx + 1
        filename = entry.get("filename", "")
        # Resolve to path relative to data root so the dataloader can find it
        img_rel = str(Path("satstreaks/Data") / filename)
        img_path = Path("data") / img_rel

        # SatStreaks doesn't provide image dimensions — use a placeholder.
        # MMDetection will read actual dims at load time.
        mask_rel = entry.get("annotation", "")
        mask_path = Path("data/satstreaks/Data") / mask_rel if mask_rel else None

        if not img_path.exists():
            logger.debug("Skipping SatStreaks image with missing file: %s", img_path)
            continue
        if mask_rel and (mask_path is None or not mask_path.exists()):
            logger.debug("Skipping SatStreaks image with missing mask: %s", mask_path)
            continue
        if mask_path is None:
            logger.debug("Skipping SatStreaks entry without mask: %s", filename)
            continue

        bbox_area = _bbox_from_mask(mask_path)
        if bbox_area is None:
            logger.debug("Skipping SatStreaks image with empty mask: %s", mask_path)
            continue
        bbox, area = bbox_area
        width, height = _image_size(img_path)
        x, y, bw, bh = bbox

        images.append({
            "id": img_id,
            "file_name": img_rel,
            "width": width,
            "height": height,
        })

        annotations.append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "iscrowd": 0,
            "bbox": bbox,
            "area": area,
            "obb": [x + bw / 2.0, y + bh / 2.0, bw, bh, 0.0],
            "segmentation_mask": str(Path("satstreaks/Data") / mask_rel),
        })
        ann_id += 1

    return images, annotations


# ---------------------------------------------------------------------------
# GTImages loader (already COCO JSON)
# ---------------------------------------------------------------------------

def _load_gtimages(
    json_path: Path,
    id_offset: int = _GTIMAGES_ID_OFFSET,
    ann_id_offset: int = 0,
    image_prefix: str = "GTImages",
) -> tuple[list[dict], list[dict]]:
    """Load GTImages COCO JSON with remapped IDs.

    Args:
        json_path: Path to gtimages.json or gtimages_negatives.json.
        id_offset: Added to every image ID to avoid collision with SatStreaks.
        ann_id_offset: Added to every annotation ID.
        image_prefix: Directory prefix, relative to ``data/``, for bare FITS
            filenames emitted by ``convert_gtimages.py``.

    Returns:
        Tuple of (images, annotations) in COCO format.
    """
    with open(json_path) as f:
        coco = json.load(f)

    old_to_new_img_id: dict[int, int] = {}
    images: list[dict] = []
    for img in coco.get("images", []):
        new_id = id_offset + img["id"]
        old_to_new_img_id[img["id"]] = new_id
        file_name = str(img.get("file_name", ""))
        if image_prefix and "/" not in file_name and not Path(file_name).is_absolute():
            file_name = str(Path(image_prefix) / file_name)
        images.append({**img, "id": new_id, "file_name": file_name})

    annotations: list[dict] = []
    for i, ann in enumerate(coco.get("annotations", [])):
        new_img_id = old_to_new_img_id.get(ann["image_id"], ann["image_id"] + id_offset)
        annotations.append({
            **ann,
            "id": ann_id_offset + i,
            "image_id": new_img_id,
        })

    return images, annotations


# ---------------------------------------------------------------------------
# COCO writer
# ---------------------------------------------------------------------------

def _write_coco(
    images: list[dict],
    annotations: list[dict],
    output_path: Path,
    description: str = "",
) -> None:
    """Write a COCO JSON file.

    Args:
        images: List of COCO image dicts.
        annotations: List of COCO annotation dicts.
        output_path: Destination path.
        description: Optional description string for the ``info`` block.
    """
    coco = {
        "info": {"description": description, "version": "1.0"},
        "licenses": [],
        "categories": [{"id": 1, "name": "streak", "supercategory": "satellite"}],
        "images": images,
        "annotations": annotations,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)
    logger.info(
        "Wrote %s  (%d images, %d annotations)",
        output_path, len(images), len(annotations),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def merge(
    val_fraction: float = 0.2,
    seed: int = 42,
    satstreaks_only: bool = False,
    gtimages_only: bool = False,
) -> None:
    """Build train/val/test COCO splits from SatStreaks and GTImages.

    Args:
        val_fraction: Fraction of the combined pool to use for validation.
        seed: Random seed for the train/val split.
        satstreaks_only: If True, skip GTImages.
        gtimages_only: If True, skip SatStreaks.
    """
    rng = random.Random(seed)

    train_images: list[dict] = []
    train_annotations: list[dict] = []
    val_images: list[dict] = []
    val_annotations: list[dict] = []
    test_images: list[dict] = []
    test_annotations: list[dict] = []

    ann_counter = 0

    # ---- SatStreaks --------------------------------------------------------
    if not gtimages_only:
        if not SATSTREAKS_LABELS.exists():
            logger.warning(
                "SatStreaks labels not found at %s — skipping SatStreaks",
                SATSTREAKS_LABELS,
            )
        else:
            logger.info("Loading SatStreaks…")
            ss_train_imgs, ss_train_anns = _load_satstreaks(
                SATSTREAKS_LABELS, "train", id_offset=0, ann_id_offset=ann_counter,
            )
            ann_counter += len(ss_train_anns)

            ss_val_imgs, ss_val_anns = _load_satstreaks(
                SATSTREAKS_LABELS, "val", id_offset=len(ss_train_imgs),
                ann_id_offset=ann_counter,
            )
            ann_counter += len(ss_val_anns)

            ss_test_imgs, ss_test_anns = _load_satstreaks(
                SATSTREAKS_LABELS, "test",
                id_offset=len(ss_train_imgs) + len(ss_val_imgs),
                ann_id_offset=ann_counter,
            )
            ann_counter += len(ss_test_anns)

            # SatStreaks already ships train/val/test — honour their split
            train_images.extend(ss_train_imgs)
            train_annotations.extend(ss_train_anns)
            val_images.extend(ss_val_imgs)
            val_annotations.extend(ss_val_anns)
            test_images.extend(ss_test_imgs)
            test_annotations.extend(ss_test_anns)

            logger.info(
                "SatStreaks: train=%d, val=%d, test=%d images",
                len(ss_train_imgs), len(ss_val_imgs), len(ss_test_imgs),
            )

    # ---- GTImages ----------------------------------------------------------
    if not satstreaks_only:
        gt_loaded = False
        if GTIMAGES_JSON.exists():
            logger.info("Loading GTImages labeled streaks…")
            gt_imgs, gt_anns = _load_gtimages(
                GTIMAGES_JSON,
                id_offset=_GTIMAGES_ID_OFFSET,
                ann_id_offset=ann_counter,
            )
            ann_counter += len(gt_anns)
            gt_loaded = True
        else:
            logger.warning(
                "GTImages JSON not found at %s — run scripts/convert_gtimages.py first",
                GTIMAGES_JSON,
            )
            gt_imgs, gt_anns = [], []

        if GTIMAGES_NEG_JSON.exists():
            logger.info("Loading GTImages negatives…")
            gt_neg_imgs, gt_neg_anns = _load_gtimages(
                GTIMAGES_NEG_JSON,
                id_offset=_GTIMAGES_ID_OFFSET + 500_000,
                ann_id_offset=ann_counter,
            )
            ann_counter += len(gt_neg_anns)
            gt_imgs.extend(gt_neg_imgs)
            gt_anns.extend(gt_neg_anns)
        else:
            logger.warning("GTImages negatives JSON not found — skipping")

        if gt_loaded and gt_imgs:
            # Randomly split GTImages 80/20 into train/val
            rng.shuffle(gt_imgs)
            split_idx = int(len(gt_imgs) * (1.0 - val_fraction))
            gt_train_imgs = gt_imgs[:split_idx]
            gt_val_imgs   = gt_imgs[split_idx:]

            gt_train_ids = {img["id"] for img in gt_train_imgs}
            gt_val_ids   = {img["id"] for img in gt_val_imgs}

            for ann in gt_anns:
                if ann["image_id"] in gt_train_ids:
                    train_annotations.append(ann)
                elif ann["image_id"] in gt_val_ids:
                    val_annotations.append(ann)

            train_images.extend(gt_train_imgs)
            val_images.extend(gt_val_imgs)

            logger.info(
                "GTImages: train=%d, val=%d images (80/20 random split, seed=%d)",
                len(gt_train_imgs), len(gt_val_imgs), seed,
            )

    # ---- Write splits ------------------------------------------------------
    if not train_images:
        logger.error("No training images found — aborting")
        return

    _write_coco(
        train_images, train_annotations,
        OUT_DIR / "train.json",
        description="ARGUS training split (SatStreaks train + GTImages 80%)",
    )
    _write_coco(
        val_images, val_annotations,
        OUT_DIR / "val.json",
        description="ARGUS validation split (SatStreaks val + GTImages 20%)",
    )
    if test_images:
        _write_coco(
            test_images, test_annotations,
            OUT_DIR / "test.json",
            description="ARGUS test split (SatStreaks test — held out)",
        )

    # Summary
    logger.info(
        "Done.  Total: train=%d, val=%d, test=%d images",
        len(train_images), len(val_images), len(test_images),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge SatStreaks + GTImages into train/val/test splits")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Fraction of GTImages to reserve for validation (default 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for GTImages train/val split (default 42)")
    parser.add_argument("--satstreaks-only", action="store_true",
                        help="Use SatStreaks only (skip GTImages)")
    parser.add_argument("--gtimages-only", action="store_true",
                        help="Use GTImages only (skip SatStreaks)")
    args = parser.parse_args()

    merge(
        val_fraction=args.val_fraction,
        seed=args.seed,
        satstreaks_only=args.satstreaks_only,
        gtimages_only=args.gtimages_only,
    )
