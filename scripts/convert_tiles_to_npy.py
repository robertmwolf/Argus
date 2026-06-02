#!/usr/bin/env python3
"""Pre-convert FITS/PNG virtual tile annotations to raw float32 .npy files on disk.

Groups tiles by parent image so each source file is opened exactly once.
Saves each tile as a float32 .npy (shape H×W, before normalisation)
so training can skip expensive FITS I/O on every step.

Supports both FITS parent files (BrentImages) and PNG parent files (Frigate).

After conversion, a new annotation JSON is written with file_name
pointing to the .npy tile rather than the virtual FITS tile path.

Usage
-----
    python scripts/convert_tiles_to_npy.py \\
        --annotations /Volumes/External/TrainingData/annotations/all_train_run5_tiled.json \\
        --output-dir  /Volumes/External/TrainingData/tiles_npy/train \\
        --output-ann  /Volumes/External/TrainingData/annotations/all_train_run5_tiled_npy.json

    python scripts/convert_tiles_to_npy.py \\
        --annotations /Volumes/External/TrainingData/annotations/val_atwood_tiled_400.json \\
        --output-dir  /Volumes/External/TrainingData/tiles_npy/val \\
        --output-ann  /Volumes/External/TrainingData/annotations/val_atwood_tiled_400_npy.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_TILE_RE = re.compile(r"^(.+?)__tx(\d+)_ty(\d+)_ts(\d+)$")


def _parse_virtual(file_name: str):
    """Return (real_path, x0, y0, ts) for a virtual tile path, or None."""
    p = Path(file_name)
    m = _TILE_RE.match(p.stem)
    if not m:
        return None
    real = str(p.parent / (m.group(1) + p.suffix))
    return real, int(m.group(2)), int(m.group(3)), int(m.group(4))


def _load_source_raw(source_path: str) -> np.ndarray:
    """Load a source image (FITS or PNG/JPEG) as float32 (H, W) array.

    FITS: returns the raw primary HDU data (e.g. 16-bit counts).
    PNG/JPEG: converts to grayscale luminance as float32 [0, 255].
    """
    suffix = Path(source_path).suffix.lower()
    if suffix in {".fits", ".fit", ".fts"}:
        with fits.open(source_path) as hdul:
            data = hdul[0].data
            if data is None:
                raise ValueError(f"No data in {source_path}")
            return data.astype(np.float32)
    else:
        with Image.open(source_path) as im:
            return np.asarray(im.convert("L"), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True,
                        help="Input COCO annotation JSON with virtual tile paths")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write .npy tile files")
    parser.add_argument("--output-ann", required=True,
                        help="Output COCO annotation JSON with .npy paths")
    parser.add_argument("--workers", type=int, default=0,
                        help="(unused, sequential for now)")
    args = parser.parse_args()

    ann_path = Path(args.annotations)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading annotations from %s", ann_path)
    with open(ann_path) as f:
        coco = json.load(f)

    images = coco["images"]

    # Group images by parent FITS path
    groups: dict[str, list[dict]] = defaultdict(list)
    non_virtual: list[dict] = []

    for img in images:
        parsed = _parse_virtual(img["file_name"])
        if parsed is None:
            non_virtual.append(img)
        else:
            real_path, x0, y0, ts = parsed
            groups[real_path].append((img, x0, y0, ts))

    if non_virtual:
        logger.warning(
            "%d non-virtual (already real) paths will be skipped — "
            "copy them manually if needed", len(non_virtual)
        )

    # id → new file_name mapping
    id_to_npy: dict[int, str] = {}
    total = len(images) - len(non_virtual)
    done = 0
    errors = 0

    logger.info(
        "Converting %d tiles from %d parent source files (FITS+PNG)...",
        total, len(groups)
    )

    for fits_idx, (fits_path, tile_list) in enumerate(groups.items()):
        try:
            raw = _load_source_raw(fits_path)
        except Exception as exc:
            logger.warning("Failed to load %s: %s — skipping %d tiles",
                           fits_path, exc, len(tile_list))
            errors += len(tile_list)
            done += len(tile_list)
            continue

        h, w = raw.shape

        for img, x0, y0, ts in tile_list:
            # Pad if tile extends beyond image edge
            pad_h = max(0, y0 + ts - h)
            pad_w = max(0, x0 + ts - w)
            arr = raw
            if pad_h > 0 or pad_w > 0:
                arr = np.pad(raw, ((0, pad_h), (0, pad_w)), mode="edge")

            tile = arr[y0:y0 + ts, x0:x0 + ts]  # (ts, ts) float32

            # Unique filename: image_id
            npy_name = f"{img['id']}.npy"
            npy_path = out_dir / npy_name
            np.save(str(npy_path), tile)
            id_to_npy[img["id"]] = str(npy_path)
            done += 1

        if (fits_idx + 1) % 100 == 0 or fits_idx == len(groups) - 1:
            logger.info(
                "Source %d/%d  tiles %d/%d  errors=%d",
                fits_idx + 1, len(groups), done, total, errors
            )

    # Build new annotation JSON
    new_images = []
    for img in images:
        new_img = dict(img)
        if img["id"] in id_to_npy:
            new_img["file_name"] = id_to_npy[img["id"]]
        new_images.append(new_img)

    new_coco = dict(coco)
    new_coco["images"] = new_images

    out_ann = Path(args.output_ann)
    out_ann.parent.mkdir(parents=True, exist_ok=True)
    with open(out_ann, "w") as f:
        json.dump(new_coco, f)

    logger.info(
        "Done. %d/%d tiles converted, %d errors. "
        "New annotations: %s",
        done - errors, total, errors, out_ann
    )


if __name__ == "__main__":
    main()
