"""Build merged train annotation for Run 17 ViT-B training.

The source annotations use virtual tile paths (e.g. Streak_foo__tx1800_ty900_ts1800.fits)
that were used with an older tiled-PNG pipeline.  The cache script (cache_dinov3_heatmap_features.py)
performs its own tiling from the full image, so it needs the real FITS path and
full-image-coordinate bounding boxes — not pre-tiled crops.

This script:
  1. Strips the __txN_tyN_tsN suffix from all_train_run5_tiled_ts1800.json to recover
     real FITS paths, then de-duplicates to one entry per unique FITS file.
  2. Verifies each real FITS path exists.
  3. Merges with synth_short + synth_medium NPY from all_train_run13_npy.json (those
     files live on the external drive and need no path remapping).
  4. Writes all_train_run17_merged.json alongside the source annotations.

Output: /Volumes/External/TrainingData/annotations/all_train_run17_merged.json
Val annotation: val_atwood_tiled_ts1800.json needs the same treatment — a separate
output is written: val_run17_fits.json
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ANN_DIR = Path("/Volumes/External/TrainingData/annotations")
TRAIN_OUTPUT = ANN_DIR / "all_train_run17_merged.json"
VAL_OUTPUT   = ANN_DIR / "val_run17_fits.json"

RUN5_ANN  = ANN_DIR / "all_train_run5_tiled_ts1800.json"
VAL_ANN   = ANN_DIR / "val_atwood_tiled_ts1800.json"
RUN13_ANN = ANN_DIR / "all_train_run13_npy.json"

_TILE_RE = re.compile(r"^(.+?)__tx\d+_ty\d+_ts\d+$")


def _strip_tile_suffix(fname: str) -> str:
    """Return the real (un-tiled) file path for a virtual tile path."""
    p = Path(fname)
    m = _TILE_RE.match(p.stem)
    if m:
        return str(p.parent / (m.group(1) + p.suffix))
    return fname


def _deduplicate_to_full_images(coco: dict) -> dict:
    """Collapse virtual-tile image entries to one entry per real FITS file.

    Annotations are kept as-is since bboxes are in full-image coordinates.
    Multiple tile entries for the same FITS file are merged into one image entry
    with a single shared image_id.  Any annotations referencing the old tile
    image_ids are remapped to the canonical id for that real file.
    """
    # Map: real_path → canonical image entry
    real_path_to_img: dict[str, dict] = {}
    old_id_to_new_id: dict[int, int] = {}

    for img in coco["images"]:
        real = _strip_tile_suffix(img["file_name"])
        if real not in real_path_to_img:
            new_img = dict(img)
            new_img["file_name"] = real
            real_path_to_img[real] = new_img
        canonical_id = real_path_to_img[real]["id"]
        old_id_to_new_id[img["id"]] = canonical_id

    new_images = list(real_path_to_img.values())
    new_anns = []
    seen_ann_keys: set[tuple] = set()
    ann_id = 1
    for ann in coco["annotations"]:
        new_img_id = old_id_to_new_id.get(ann["image_id"], ann["image_id"])
        key = (new_img_id, tuple(ann.get("bbox", [])), ann.get("category_id"))
        if key in seen_ann_keys:
            continue
        seen_ann_keys.add(key)
        new_ann = dict(ann)
        new_ann["id"] = ann_id
        new_ann["image_id"] = new_img_id
        new_anns.append(new_ann)
        ann_id += 1

    return {
        "info": coco.get("info", {}),
        "categories": coco.get("categories", []),
        "images": new_images,
        "annotations": new_anns,
    }


def _verify_files(images: list[dict], key: str = "file_name") -> tuple[int, int]:
    ok = missing = 0
    for img in images:
        if Path(img[key]).exists():
            ok += 1
        else:
            missing += 1
    return ok, missing


def main() -> int:
    for p in [RUN5_ANN, VAL_ANN, RUN13_ANN]:
        if not p.exists():
            logger.error("Missing: %s", p)
            return 1

    # ── Train ──────────────────────────────────────────────────────────────────
    run5  = json.loads(RUN5_ANN.read_text())
    run13 = json.loads(RUN13_ANN.read_text())

    run5_dedup = _deduplicate_to_full_images(run5)
    ok, missing = _verify_files(run5_dedup["images"])
    if missing:
        logger.error("run5: %d/%d real FITS files missing — aborting", missing, ok + missing)
        return 1
    logger.info("run5 deduplicated: %d unique FITS files (%d annotations), all present",
                len(run5_dedup["images"]), len(run5_dedup["annotations"]))

    # Extract synth images from run13
    synth_imgs = [
        img for img in run13["images"]
        if "synth" in img.get("npy_path", img.get("file_name", ""))
    ]
    synth_img_ids = {img["id"] for img in synth_imgs}
    synth_anns = [ann for ann in run13["annotations"] if ann["image_id"] in synth_img_ids]

    ok_s, missing_s = _verify_files(
        [{**img, "file_name": img.get("npy_path", img.get("file_name", ""))} for img in synth_imgs]
    )
    if missing_s:
        logger.error("synth: %d/%d NPY files missing — aborting", missing_s, ok_s + missing_s)
        return 1
    logger.info("synth: %d images (%d annotations), all present", len(synth_imgs), len(synth_anns))

    # Remap synth IDs to avoid conflict with run5 IDs
    max_img_id = max(img["id"] for img in run5_dedup["images"])
    max_ann_id = max((ann["id"] for ann in run5_dedup["annotations"]), default=0)
    img_offset = max_img_id + 1
    ann_offset = max_ann_id + 1

    old_to_new = {img["id"]: img["id"] + img_offset for img in synth_imgs}
    remapped_imgs = []
    for img in synth_imgs:
        new = dict(img)
        new["id"] = old_to_new[img["id"]]
        new["file_name"] = img.get("npy_path", img.get("file_name", ""))
        remapped_imgs.append(new)

    remapped_anns = []
    for i, ann in enumerate(synth_anns):
        new = dict(ann)
        new["id"] = ann_offset + i
        new["image_id"] = old_to_new[ann["image_id"]]
        remapped_anns.append(new)

    train_merged = {
        "info": {
            "description": "Run 17 train — run5 FITS (de-tiled) + run13 synth NPY",
            "source_run5": str(RUN5_ANN),
            "source_run13_synth": str(RUN13_ANN),
        },
        "categories": run5_dedup.get("categories", []),
        "images": run5_dedup["images"] + remapped_imgs,
        "annotations": run5_dedup["annotations"] + remapped_anns,
    }
    TRAIN_OUTPUT.write_text(json.dumps(train_merged, indent=2))
    logger.info("Train written: %s  (%d images, %d annotations)",
                TRAIN_OUTPUT, len(train_merged["images"]), len(train_merged["annotations"]))

    # ── Val ────────────────────────────────────────────────────────────────────
    val = json.loads(VAL_ANN.read_text())
    val_dedup = _deduplicate_to_full_images(val)
    ok_v, missing_v = _verify_files(val_dedup["images"])
    if missing_v:
        logger.error("val: %d/%d real FITS files missing — aborting", missing_v, ok_v + missing_v)
        return 1
    logger.info("val deduplicated: %d unique FITS files (%d annotations), all present",
                len(val_dedup["images"]), len(val_dedup["annotations"]))

    val_dedup["info"] = {"description": "Run 17 val — val_atwood_tiled_ts1800 de-tiled FITS"}
    VAL_OUTPUT.write_text(json.dumps(val_dedup, indent=2))
    logger.info("Val written: %s  (%d images, %d annotations)",
                VAL_OUTPUT, len(val_dedup["images"]), len(val_dedup["annotations"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
