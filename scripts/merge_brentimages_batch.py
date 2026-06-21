"""Merge a new BrentImages annotation batch into the canonical training source.

Takes the current merged annotation (``--base``) and a new per-batch COCO JSON
(``--add``) and produces an updated merged file (``--output``).  Frames listed
in the satellite-train exclusion manifest (``--exclude-manifest``) are silently
skipped so the satellite-train exclusion policy is enforced automatically.

Image and annotation IDs are reassigned globally to avoid collisions.  The
output carries a ``provenance`` key that records every source file and the
exclusion manifest used, so the merge is auditable.

Typical usage::

    python scripts/merge_brentimages_batch.py \\
        --base  /Volumes/External/TrainingData/annotations/all_train_run17_merged_no_sattrains.json \\
        --add   /Volumes/External/TrainingData/raw/BrentImages/Img_YYYYMMDD_Atwood/annotations.json \\
        --exclude-manifest /Volumes/External/TrainingData/annotations/sat_train_excluded.json \\
        --output /Volumes/External/TrainingData/annotations/all_train_run18_merged_no_sattrains.json

After merging, rebuild the tile dataset with ``scripts/build_atwood_window_dataset.py``
(see ``docs/data_strategy.md`` for the full workflow).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _normalize_to_tile_local(ann: dict, img: dict) -> dict | None:
    """Translate annotation OBB to tile-local coordinates if needed.

    Annotations must be in tile-local space (cx/cy relative to the crop window,
    not the full FITS frame). If an annotation's center lies outside the tile
    bounds but subtracting tile_origin brings it in-bounds, it was saved in
    full-frame space and is fixed in-place. If it remains out-of-bounds after
    translation it is genuinely wrong and dropped (returns None).

    Images without tile_origin or with unknown dimensions (width/height == 0)
    are assumed to be full-frame annotations and pass through unchanged.
    """
    import math

    tile_origin = img.get("tile_origin")
    tile_w = int(img.get("width", 0))
    tile_h = int(img.get("height", 0))

    if not tile_origin or tile_w == 0 or tile_h == 0:
        return ann

    ox, oy = int(tile_origin[0]), int(tile_origin[1])
    obb = ann.get("obb", {})
    cx, cy = float(obb.get("cx", 0.0)), float(obb.get("cy", 0.0))

    if 0.0 <= cx <= tile_w and 0.0 <= cy <= tile_h:
        return ann  # already tile-local

    ncx, ncy = cx - ox, cy - oy
    if not (0.0 <= ncx <= tile_w and 0.0 <= ncy <= tile_h):
        logger.warning(
            "Dropping ann id=%s for %s: center (%.0f,%.0f) → (%.0f,%.0f) "
            "still OOB in %dx%d tile (origin %d,%d)",
            ann.get("id"), img.get("file_name"),
            cx, cy, ncx, ncy, tile_w, tile_h, ox, oy,
        )
        return None

    logger.warning(
        "Ann id=%s for %s: translated OBB from full-frame (%.0f,%.0f) to tile-local (%.0f,%.0f)",
        ann.get("id"), img.get("file_name"), cx, cy, ncx, ncy,
    )
    ann = dict(ann)
    new_obb = dict(obb)
    new_obb["cx"] = round(ncx, 2)
    new_obb["cy"] = round(ncy, 2)
    ann["obb"] = new_obb

    # Recompute derived endpoint fields.
    wl, hl = float(new_obb.get("w", 0)), float(new_obb.get("h", 0))
    ang = math.radians(float(new_obb.get("angle_deg", 0)))
    hx, hy = wl / 2 * math.cos(ang), wl / 2 * math.sin(ang)
    x1, y1, x2, y2 = ncx - hx, ncy - hy, ncx + hx, ncy + hy
    ann["x1"] = round(x1, 2); ann["y1"] = round(y1, 2)
    ann["x2"] = round(x2, 2); ann["y2"] = round(y2, 2)
    ann["bbox"] = [round(min(x1, x2), 2), round(min(y1, y2), 2),
                   round(abs(x2 - x1) + hl, 2), round(abs(y2 - y1) + hl, 2)]
    # Recompute segmentation (4 corners of the OBB).
    hx2, hy2 = hl / 2 * math.sin(ang), hl / 2 * math.cos(ang)
    corners = [
        (ncx - hx + hy2, ncy - hy - hx2),
        (ncx + hx + hy2, ncy + hy - hx2),
        (ncx + hx - hy2, ncy + hy + hx2),
        (ncx - hx - hy2, ncy - hy + hx2),
    ]
    ann["segmentation"] = [[coord for corner in corners for coord in corner]]
    return ann


def merge(
    base_path: str | Path,
    add_path: str | Path,
    output_path: str | Path,
    exclude_manifest_path: str | Path | None = None,
) -> None:
    base = _load(base_path)
    batch = _load(add_path)

    # Build exclusion set from the satellite-train manifest
    excluded_fnames: set[str] = set()
    if exclude_manifest_path and Path(exclude_manifest_path).exists():
        manifest = _load(exclude_manifest_path)
        excluded_fnames = {e["file_name"] for e in manifest.get("excluded", [])}
        logger.info("Exclusion manifest: %d satellite-train frames", len(excluded_fnames))
    else:
        logger.warning("No exclusion manifest supplied — no frames will be filtered")

    # Collect existing file_names to skip true duplicates
    existing_fnames = {img["file_name"] for img in base.get("images", [])}

    # Filter the batch: skip excluded frames and exact duplicates
    batch_imgs_filtered = []
    skip_excluded = skip_dup = 0
    for img in batch.get("images", []):
        fn = img["file_name"]
        if fn in excluded_fnames:
            skip_excluded += 1
        elif fn in existing_fnames:
            skip_dup += 1
            logger.debug("Skipping duplicate: %s", fn)
        else:
            batch_imgs_filtered.append(img)

    if skip_excluded:
        logger.info("Skipped %d satellite-train frames (in exclusion manifest)", skip_excluded)
    if skip_dup:
        logger.info("Skipped %d duplicate frames (already in base)", skip_dup)

    batch_img_ids_kept = {int(img["id"]) for img in batch_imgs_filtered}
    batch_anns_filtered = [
        ann for ann in batch.get("annotations", [])
        if int(ann["image_id"]) in batch_img_ids_kept
    ]

    logger.info(
        "Adding %d images and %d annotations from batch",
        len(batch_imgs_filtered), len(batch_anns_filtered),
    )

    # Reassign IDs globally: images start from 1, annotations from 1
    all_imgs: list[dict] = []
    all_anns: list[dict] = []

    # Pass 1: base images (IDs already contiguous — just collect them)
    base_id_map: dict[int, int] = {}
    for new_id, img in enumerate(base.get("images", []), start=1):
        old_id = int(img["id"])
        base_id_map[old_id] = new_id
        all_imgs.append({**img, "id": new_id})

    # Pass 2: batch images (new IDs follow base)
    batch_id_map: dict[int, int] = {}
    for img in batch_imgs_filtered:
        new_id = len(all_imgs) + 1
        batch_id_map[int(img["id"])] = new_id
        all_imgs.append({**img, "id": new_id})

    # Pass 3: annotations — remap image_id, validate tile-local coords, assign new IDs
    img_by_new_id: dict[int, dict] = {img["id"]: img for img in all_imgs}
    ann_id_counter = 1
    n_oob_dropped = 0
    for ann in base.get("annotations", []):
        old_img_id = int(ann["image_id"])
        new_img_id = base_id_map.get(old_img_id)
        if new_img_id is None:
            continue  # image was somehow absent from base — skip
        ann = _normalize_to_tile_local({**ann, "id": ann_id_counter, "image_id": new_img_id},
                                       img_by_new_id[new_img_id])
        if ann is None:
            n_oob_dropped += 1
            continue
        ann["id"] = ann_id_counter
        all_anns.append(ann)
        ann_id_counter += 1

    for ann in batch_anns_filtered:
        old_img_id = int(ann["image_id"])
        new_img_id = batch_id_map.get(old_img_id)
        if new_img_id is None:
            continue
        ann = _normalize_to_tile_local({**ann, "id": ann_id_counter, "image_id": new_img_id},
                                       img_by_new_id[new_img_id])
        if ann is None:
            n_oob_dropped += 1
            continue
        ann["id"] = ann_id_counter
        all_anns.append(ann)
        ann_id_counter += 1

    if n_oob_dropped:
        logger.warning("Dropped %d annotations with OOB coordinates during merge", n_oob_dropped)

    out = {
        "images": all_imgs,
        "annotations": all_anns,
        "categories": base.get("categories") or batch.get("categories") or [
            {"id": 1, "name": "streak", "supercategory": "satellite"}
        ],
        "provenance": {
            "builder": "scripts/merge_brentimages_batch.py",
            "base": str(base_path),
            "added": str(add_path),
            "exclude_manifest": str(exclude_manifest_path) if exclude_manifest_path else None,
            "base_images": len(base.get("images", [])),
            "added_images": len(batch_imgs_filtered),
            "skipped_excluded": skip_excluded,
            "skipped_duplicate": skip_dup,
            "total_images": len(all_imgs),
            "total_annotations": len(all_anns),
            "dropped_oob_annotations": n_oob_dropped,
        },
    }

    Path(output_path).write_text(json.dumps(out))
    logger.info(
        "Written: %s  (%d images, %d annotations)",
        output_path, len(all_imgs), len(all_anns),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True,
                   help="Current canonical merged annotation JSON")
    p.add_argument("--add", required=True,
                   help="New batch COCO annotation JSON to merge in")
    p.add_argument("--exclude-manifest", default="",
                   help="sat_train_excluded.json; frames listed here are skipped")
    p.add_argument("--output", required=True,
                   help="Output path for the new merged annotation")
    args = p.parse_args()

    for path, label in [(args.base, "--base"), (args.add, "--add")]:
        if not Path(path).exists():
            logger.error("%s not found: %s", label, path)
            sys.exit(1)

    merge(
        base_path=args.base,
        add_path=args.add,
        output_path=args.output,
        exclude_manifest_path=args.exclude_manifest or None,
    )


if __name__ == "__main__":
    main()
