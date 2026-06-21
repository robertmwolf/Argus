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

    # Pass 3: annotations — remap image_id, assign new ann IDs
    ann_id_counter = 1
    for ann in base.get("annotations", []):
        old_img_id = int(ann["image_id"])
        new_img_id = base_id_map.get(old_img_id)
        if new_img_id is None:
            continue  # image was somehow absent from base — skip
        all_anns.append({**ann, "id": ann_id_counter, "image_id": new_img_id})
        ann_id_counter += 1

    for ann in batch_anns_filtered:
        old_img_id = int(ann["image_id"])
        new_img_id = batch_id_map.get(old_img_id)
        if new_img_id is None:
            continue
        all_anns.append({**ann, "id": ann_id_counter, "image_id": new_img_id})
        ann_id_counter += 1

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
