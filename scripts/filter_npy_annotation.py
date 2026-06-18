#!/usr/bin/env python3
"""Filter an NPY annotation to only include image IDs present in a reference JSON.

Used after convert_tiles_to_npy.py to strip duplicate or unwanted entries when
the input JSON was larger than intended (e.g. duplicated synthetics).

Usage:
    python scripts/filter_npy_annotation.py \\
        --npy-ann /Volumes/External/TrainingData/annotations/all_train_run6_tiled_npy.json \\
        --ref-ann /Volumes/External/TrainingData/annotations/all_train_run6_tiled.json \\
        --output  /Volumes/External/TrainingData/annotations/all_train_run6_tiled_npy_filtered.json
"""
from __future__ import annotations

import argparse
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npy-ann", required=True)
    parser.add_argument("--ref-ann", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.ref_ann) as f:
        ref = json.load(f)
    ref_ids = {img["id"] for img in ref["images"]}
    logger.info("Reference annotation has %d image IDs", len(ref_ids))

    with open(args.npy_ann) as f:
        npy = json.load(f)

    kept_images = [img for img in npy["images"] if img["id"] in ref_ids]
    kept_ann_ids = {img["id"] for img in kept_images}
    kept_anns = [a for a in npy.get("annotations", []) if a["image_id"] in kept_ann_ids]

    out = dict(npy)
    out["images"] = kept_images
    out["annotations"] = kept_anns
    with open(args.output, "w") as f:
        json.dump(out, f)

    logger.info(
        "Filtered %d → %d images, %d → %d annotations → %s",
        len(npy["images"]), len(kept_images),
        len(npy.get("annotations", [])), len(kept_anns),
        args.output,
    )


if __name__ == "__main__":
    main()
