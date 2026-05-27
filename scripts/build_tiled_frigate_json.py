"""Tile Frigate 2325×1555 images into crops for short-streak training.

Frigate streaks are 20–80 px diagonal on 2325×1555 frames — too small for the
full-image resize path.  This script tiles each image with a sliding window of
``native_tile_size`` pixels (default 400, use 110 for 3.6× magnification) and
emits:
  - every tile that contains ≥1 annotation centre
  - one random negative tile per unannotated image (domain adaptation)

Output: data/annotations/frigate_tiled_train.json  (COCO format)

Tile coordinates are embedded in file_name as a virtual path::

    /path/to/frigate/<stem>__tx<x0>_ty<y0>_ts<native_tile_size>.png

The data loader is expected to crop ``native_tile_size × native_tile_size`` px
at ``(x0, y0)`` from the source image.  With the default ``native_tile_size=400``
the image dimensions and annotation bboxes in the JSON are in 400 px space
(1:1 ratio, same as existing training data).  With ``native_tile_size=110`` the
image dimensions and annotation bboxes are in 110 px space; the MMDetection
training pipeline's resize step will upscale to ``model_input_size=400`` and
scale the annotations accordingly.

Usage::

    # 1:1 tiles (same as previous behaviour)
    python scripts/build_tiled_frigate_json.py

    # Zoom-in tiles for short-streak training (~3.6× magnification)
    python scripts/build_tiled_frigate_json.py --native-tile-size 110 --overlap 0.5

Source: adaptive_tiling_plan.md §5 — files to create / modify
Ref: docs/adaptive_tiling_plan.md
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ANN_DIR = _REPO_ROOT / "data/annotations"
_CANONICAL_CATEGORIES = [{"id": 1, "name": "streak", "supercategory": "satellite"}]

# Defaults — can be overridden via CLI.
_DEFAULT_TILE_SIZE = 400
_DEFAULT_OVERLAP = 0.25
# Minimum fraction of original bbox area that must survive clipping to keep it.
MIN_AREA_FRACTION = 0.25
NEGATIVE_TILES_PER_IMAGE = 1
RANDOM_SEED = 42


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a COCO-format tiled Frigate training JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--native-tile-size",
        type=int,
        default=_DEFAULT_TILE_SIZE,
        help=(
            "Crop footprint in source-image pixels.  Use 110 for ~3.6× "
            "upsampling magnification to bring 20–80 px Frigate streaks into "
            "the model's training sweet-spot."
        ),
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=_DEFAULT_OVERLAP,
        help="Fractional tile overlap in [0, 1).",
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=_ANN_DIR / "frigate_streaks.json",
        help="Source COCO annotation file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output JSON path.  Defaults to "
            "data/annotations/frigate_tiled_train_ts<native_tile_size>.json "
            "(or frigate_tiled_train.json when native_tile_size=400)."
        ),
    )
    return parser.parse_args()


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    n = int(math.ceil((length - tile_size) / stride)) + 1
    return [i * stride for i in range(n)]


def _clip_bbox(
    bbox: list[float],
    x0: int,
    y0: int,
    tile_size: int,
) -> list[float] | None:
    """Clip ``[x, y, w, h]`` to tile; return in tile-local coords, or None."""
    bx, by, bw, bh = bbox
    orig_area = bw * bh
    if orig_area <= 0:
        return None

    x1 = max(bx, float(x0))
    y1 = max(by, float(y0))
    x2 = min(bx + bw, float(x0 + tile_size))
    y2 = min(by + bh, float(y0 + tile_size))

    if x2 <= x1 or y2 <= y1:
        return None
    clipped_area = (x2 - x1) * (y2 - y1)
    if clipped_area / orig_area < MIN_AREA_FRACTION:
        return None

    return [x1 - x0, y1 - y0, x2 - x1, y2 - y1]


def _tile_file_name(original_path: str, x0: int, y0: int, tile_size: int) -> str:
    """Encode tile position and native crop size into the filename."""
    stem = Path(original_path).stem
    suffix = Path(original_path).suffix
    parent = str(Path(original_path).parent)
    return f"{parent}/{stem}__tx{x0}_ty{y0}_ts{tile_size}{suffix}"


def build_tiled_frigate_json(
    native_tile_size: int = _DEFAULT_TILE_SIZE,
    overlap: float = _DEFAULT_OVERLAP,
    src_path: Path | None = None,
    out_path: Path | None = None,
) -> None:
    """Generate a tiled Frigate COCO training JSON.

    Args:
        native_tile_size: Crop footprint in source-image pixels.  Image
            dimensions and annotation bboxes in the output JSON are in this
            pixel space.  The MMDetection training pipeline's resize step
            upscales tiles (and scales annotations) to the model input size.
        overlap: Fractional tile overlap.  Increase to 0.5 for small tiles
            (``native_tile_size < 200``) to reduce missed-streak rate.
        src_path: Source COCO annotation file.  Defaults to
            ``data/annotations/frigate_streaks.json``.
        out_path: Output path.  Defaults to
            ``data/annotations/frigate_tiled_train.json`` (for 400 px tiles)
            or ``frigate_tiled_train_ts<size>.json`` for other sizes.
    """
    if src_path is None:
        src_path = _ANN_DIR / "frigate_streaks.json"
    if out_path is None:
        if native_tile_size == 400:
            out_path = _ANN_DIR / "frigate_tiled_train.json"
        else:
            out_path = _ANN_DIR / f"frigate_tiled_train_ts{native_tile_size}.json"

    rng = random.Random(RANDOM_SEED)
    stride = max(1, int(round(native_tile_size * (1.0 - overlap))))

    logger.info(
        "Building tiled Frigate JSON: native_tile_size=%d, overlap=%.2f, stride=%d",
        native_tile_size, overlap, stride,
    )

    with open(src_path) as f:
        src = json.load(f)

    ann_by_image: dict[int, list[dict]] = {}
    for ann in src.get("annotations", []):
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    out_images: list[dict] = []
    out_annotations: list[dict] = []
    next_img_id = 1
    next_ann_id = 1

    positive_count = 0
    negative_count = 0

    for img in src["images"]:
        W = img["width"]
        H = img["height"]
        orig_path = img["file_name"]
        anns = ann_by_image.get(img["id"], [])

        xs = _tile_starts(W, native_tile_size, stride)
        ys = _tile_starts(H, native_tile_size, stride)

        if anns:
            # Emit every tile that captures ≥1 annotation centre.
            ann_centres = [
                (a["bbox"][0] + a["bbox"][2] / 2, a["bbox"][1] + a["bbox"][3] / 2, a)
                for a in anns
            ]

            for y0 in ys:
                for x0 in xs:
                    tile_anns = []
                    for cx, cy, ann in ann_centres:
                        if (x0 <= cx < x0 + native_tile_size and
                                y0 <= cy < y0 + native_tile_size):
                            clipped = _clip_bbox(ann["bbox"], x0, y0, native_tile_size)
                            if clipped is not None:
                                tile_anns.append((clipped, ann.get("category_id", 1)))

                    if not tile_anns:
                        continue

                    fname = _tile_file_name(orig_path, x0, y0, native_tile_size)
                    out_images.append({
                        "id": next_img_id,
                        "file_name": fname,
                        # Dimensions in native-tile-size pixel space.
                        # MMDetection resize handles upscaling to model_input_size.
                        "width": native_tile_size,
                        "height": native_tile_size,
                    })
                    for clipped_bbox, _ in tile_anns:
                        out_annotations.append({
                            "id": next_ann_id,
                            "image_id": next_img_id,
                            "category_id": 1,
                            "bbox": clipped_bbox,
                            "area": clipped_bbox[2] * clipped_bbox[3],
                            "iscrowd": 0,
                        })
                        next_ann_id += 1
                    next_img_id += 1
                    positive_count += 1

        else:
            # Negative image — emit one random tile for domain adaptation.
            x0 = rng.choice(xs)
            y0 = rng.choice(ys)
            fname = _tile_file_name(orig_path, x0, y0, native_tile_size)
            out_images.append({
                "id": next_img_id,
                "file_name": fname,
                "width": native_tile_size,
                "height": native_tile_size,
            })
            next_img_id += 1
            negative_count += 1

    result = {
        "info": {
            "description": f"Frigate tiled training crops (native_tile_size={native_tile_size})",
            "version": "1.0",
        },
        "licenses": [],
        "categories": _CANONICAL_CATEGORIES,
        "images": out_images,
        "annotations": out_annotations,
    }

    with open(out_path, "w") as f:
        json.dump(result, f)

    logger.info(
        "%s: %d positive tiles (%d annotations) + %d negative tiles = %d total images",
        out_path.name,
        positive_count,
        len(out_annotations),
        negative_count,
        len(out_images),
    )


if __name__ == "__main__":
    args = _parse_args()
    build_tiled_frigate_json(
        native_tile_size=args.native_tile_size,
        overlap=args.overlap,
        src_path=args.src,
        out_path=args.out,
    )
