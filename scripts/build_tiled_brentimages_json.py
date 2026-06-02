"""Tile BrentImages full-frame FITS into 400×400 training crops.

BrentImages FITS frames are 6248×4176 px.  Passing them full-frame to the
model (400 px input) downscales streaks from ~107–1540 px native to 6–99 px
at model input — too small to detect reliably.  Tiling at native_tile_size=400
presents the model with the full-resolution crops it was designed for.

  native streak (107–1540 px) × magnification (400/400 = 1×) = 107–1540 px

This is the same mechanism used for Frigate tiles; see
``scripts/build_tiled_frigate_json.py`` and ``docs/adaptive_tiling_plan.md``.

Tile selection strategy
-----------------------
Frigate streaks (20–80 px) fit entirely in one tile, so centre-based
selection (emit only the tile containing the streak centre) captures the full
annotation.

BrentImages streaks (median 652 px) are often longer than one 400 px tile.
This script uses **area-based selection**: emit every tile where the clipped
annotation bbox retains ≥ MIN_AREA_FRACTION (default 25%) of its original
area.  This ensures fragments from all significant tiles are in the training
set, which is important for the model to learn to detect long-streak fragments
and for the collinear stitcher to receive training signal.

Usage::

    # Night 1:
    python scripts/build_tiled_brentimages_json.py \\
        --src /Volumes/External/TrainingData/annotations/brentimages_night1_full.json \\
        --out /Volumes/External/TrainingData/annotations/brentimages_night1_tiled_train.json

    # Night 2:
    python scripts/build_tiled_brentimages_json.py \\
        --src /Volumes/External/TrainingData/annotations/brentimages_night2_full.json \\
        --out /Volumes/External/TrainingData/annotations/brentimages_night2_tiled_train.json

    # Larger tiles for very long streaks:
    python scripts/build_tiled_brentimages_json.py \\
        --src data/annotations/brentimages_night1_full.json \\
        --native-tile-size 600 --overlap 0.5

Source: adaptive_tiling_plan.md §5 — files to create / modify
Ref: docs/adaptive_tiling_plan.md, docs/training_methods.md §2
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_ANN_DIR = Path(
    os.environ.get("ARGUS_ANNOTATIONS_DIR", "/Volumes/External/TrainingData/annotations")
)
_CANONICAL_CATEGORIES = [{"id": 1, "name": "streak", "supercategory": "satellite"}]

# Defaults — can be overridden via CLI.
_DEFAULT_TILE_SIZE = 400
_DEFAULT_OVERLAP = 0.5
# Minimum fraction of original bbox area that must survive clipping to keep it.
MIN_AREA_FRACTION = 0.25
# Number of random negative tiles emitted per negative (unannotated) image.
NEGATIVE_TILES_PER_IMAGE = 2
# Hard-negative tiles emitted per positive image (annotation-free tiles from
# images that DO contain streaks — the hardest negatives because they share
# the same star-field background as the positive tiles).
HARD_NEG_PER_POS = 0
RANDOM_SEED = 42


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a COCO-format tiled BrentImages training JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Source COCO annotation file with full-frame BrentImages entries.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output JSON path.  Defaults to "
            "$ARGUS_ANNOTATIONS_DIR/<src-stem>_tiled_ts<native_tile_size>.json."
        ),
    )
    parser.add_argument(
        "--native-tile-size",
        type=int,
        default=_DEFAULT_TILE_SIZE,
        help=(
            "Crop footprint in source-image pixels.  Image dimensions and "
            "annotation bboxes in the output JSON are in this pixel space; "
            "the MMDetection resize step upscales to model_input_size (400)."
        ),
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=_DEFAULT_OVERLAP,
        help="Fractional tile overlap in [0, 1).",
    )
    parser.add_argument(
        "--min-area-fraction",
        type=float,
        default=MIN_AREA_FRACTION,
        help=(
            "Minimum fraction of original bbox area that must be visible "
            "in a tile for it to be included (area-based selection)."
        ),
    )
    parser.add_argument(
        "--neg-tiles-per-image",
        type=int,
        default=NEGATIVE_TILES_PER_IMAGE,
        help="Number of random negative tiles emitted per unannotated image.",
    )
    parser.add_argument(
        "--hard-neg-per-pos",
        type=int,
        default=HARD_NEG_PER_POS,
        help=(
            "Hard-negative tiles emitted per positive (annotated) image.  "
            "These are annotation-free tiles drawn from images that contain "
            "at least one streak — the hardest negatives because they share "
            "the same star-field background as the positive tiles.  "
            "Recommended: 5 for Run 6+ to reach ~50%% negative ratio."
        ),
    )
    return parser.parse_args()


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    """Return all tile start positions along one axis."""
    if length <= tile_size:
        return [0]
    n = int(math.ceil((length - tile_size) / stride)) + 1
    return [i * stride for i in range(n)]


def _clip_bbox(
    bbox: list[float],
    x0: int,
    y0: int,
    tile_size: int,
    min_area_fraction: float,
) -> list[float] | None:
    """Clip ``[x, y, w, h]`` to tile; return tile-local coords, or None.

    Returns None if the clipped area is below ``min_area_fraction`` of the
    original area (avoids near-empty boxes at tile boundaries).
    """
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
    if clipped_area / orig_area < min_area_fraction:
        return None

    return [x1 - x0, y1 - y0, x2 - x1, y2 - y1]


def _transform_obb(
    obb: dict | list | None,
    x0: int,
    y0: int,
) -> dict | None:
    """Translate an OBB's centre to tile-local coordinates.

    Width, height, and angle are translation-invariant and are preserved
    unchanged.  The centre is shifted by ``(x0, y0)`` — the tile's top-left
    corner in the source image.

    Args:
        obb: Source OBB as a dict (``cx``, ``cy``, ``w``, ``h``,
            ``angle_deg``) or a 5-element list, or None.
        x0: Tile left edge in source-image pixels.
        y0: Tile top edge in source-image pixels.

    Returns:
        Tile-local OBB dict, or None if ``obb`` is None.
    """
    if obb is None:
        return None
    if isinstance(obb, dict):
        return {
            "cx":        float(obb["cx"]) - x0,
            "cy":        float(obb["cy"]) - y0,
            "w":         float(obb["w"]),
            "h":         float(obb["h"]),
            "angle_deg": float(obb.get("angle_deg", 0.0)),
        }
    # list format: [cx, cy, w, h, angle_deg]
    return {
        "cx":        float(obb[0]) - x0,
        "cy":        float(obb[1]) - y0,
        "w":         float(obb[2]),
        "h":         float(obb[3]),
        "angle_deg": float(obb[4]) if len(obb) > 4 else 0.0,
    }


def _tile_file_name(original_path: str, x0: int, y0: int, tile_size: int) -> str:
    """Encode tile position and native crop size into the virtual filename."""
    stem = Path(original_path).stem
    suffix = Path(original_path).suffix
    parent = str(Path(original_path).parent)
    return f"{parent}/{stem}__tx{x0}_ty{y0}_ts{tile_size}{suffix}"


def build_tiled_brentimages_json(
    src_path: Path,
    out_path: Path | None = None,
    native_tile_size: int = _DEFAULT_TILE_SIZE,
    overlap: float = _DEFAULT_OVERLAP,
    min_area_fraction: float = MIN_AREA_FRACTION,
    neg_tiles_per_image: int = NEGATIVE_TILES_PER_IMAGE,
    hard_neg_per_pos: int = HARD_NEG_PER_POS,
) -> Path:
    """Generate a tiled BrentImages COCO training JSON.

    Args:
        src_path: Source COCO file with full-frame BrentImages images and
            annotations.  Image ``width``/``height`` must be correct (they are
            used to compute tile positions; no actual FITS files are loaded).
        out_path: Output path.  Defaults to
            ``$ARGUS_ANNOTATIONS_DIR/<src-stem>_tiled_ts<size>.json``.
        native_tile_size: Crop footprint in source pixels (default 400).
            The output JSON encodes images as ``native_tile_size × native_tile_size``;
            the MMDetection pipeline's resize step upscales to model input size.
        overlap: Fractional tile overlap (default 0.5).
        min_area_fraction: Minimum fraction of bbox area required after
            clipping to include a tile (area-based selection for long streaks).
        neg_tiles_per_image: Random tiles emitted per unannotated image.
        hard_neg_per_pos: Hard-negative tiles emitted per positive image.
            These are annotation-free tiles from images that contain streaks —
            the hardest negatives because they share the same star-field context
            as the positive tiles.  Set to ≥5 for Run 6+ builds.

    Returns:
        Path to the written JSON.
    """
    if out_path is None:
        stem = src_path.stem
        out_path = _ANN_DIR / f"{stem}_tiled_ts{native_tile_size}.json"

    rng = random.Random(RANDOM_SEED)
    stride = max(1, int(round(native_tile_size * (1.0 - overlap))))

    logger.info(
        "Building tiled BrentImages JSON: src=%s, native_tile_size=%d, "
        "overlap=%.2f, stride=%d",
        src_path.name, native_tile_size, overlap, stride,
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

    positive_tile_count = 0
    negative_tile_count = 0
    hard_neg_tile_count = 0
    skipped_images = 0

    for img in src["images"]:
        W = img.get("width")
        H = img.get("height")
        if not W or not H:
            logger.warning("Image %s missing width/height — skipping", img["file_name"])
            skipped_images += 1
            continue

        orig_path = img["file_name"]
        anns = ann_by_image.get(img["id"], [])

        xs = _tile_starts(W, native_tile_size, stride)
        ys = _tile_starts(H, native_tile_size, stride)

        if anns:
            # Area-based selection: emit every tile that sees ≥25% of any annotation.
            # This captures all significant fragments of long streaks (median 652 px),
            # unlike centre-only selection which misses the streak ends.
            for y0 in ys:
                for x0 in xs:
                    tile_anns: list[tuple[list[float], int, dict]] = []
                    for ann in anns:
                        clipped = _clip_bbox(
                            ann["bbox"], x0, y0, native_tile_size, min_area_fraction
                        )
                        if clipped is not None:
                            tile_anns.append((clipped, ann.get("category_id", 1), ann))

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
                    for clipped_bbox, cat_id, src_ann in tile_anns:
                        out_ann: dict = {
                            "id":          next_ann_id,
                            "image_id":    next_img_id,
                            "category_id": cat_id,
                            "bbox":        clipped_bbox,
                            "area":        clipped_bbox[2] * clipped_bbox[3],
                            "iscrowd":     0,
                        }
                        tile_obb = _transform_obb(src_ann.get("obb"), x0, y0)
                        if tile_obb is not None:
                            out_ann["obb"] = tile_obb
                        if "attributes" in src_ann:
                            out_ann["attributes"] = src_ann["attributes"]
                        out_annotations.append(out_ann)
                        next_ann_id += 1
                    next_img_id += 1
                    positive_tile_count += 1

            # Hard negatives: annotation-free tiles from this positive image.
            # These share the star-field background of the positive tiles and
            # are the hardest negatives to suppress without explicit examples.
            if hard_neg_per_pos > 0:
                all_positions = [(x0, y0) for y0 in ys for x0 in xs]
                # Exclude positions that contributed at least one positive tile.
                hard_neg_candidates = []
                for x0, y0 in all_positions:
                    tile_has_ann = any(
                        _clip_bbox(ann["bbox"], x0, y0, native_tile_size, min_area_fraction)
                        is not None
                        for ann in anns
                    )
                    if not tile_has_ann:
                        hard_neg_candidates.append((x0, y0))
                chosen_hn = rng.sample(
                    hard_neg_candidates,
                    k=min(hard_neg_per_pos, len(hard_neg_candidates)),
                )
                for x0, y0 in chosen_hn:
                    fname = _tile_file_name(orig_path, x0, y0, native_tile_size)
                    out_images.append({
                        "id": next_img_id,
                        "file_name": fname,
                        "width": native_tile_size,
                        "height": native_tile_size,
                    })
                    next_img_id += 1
                    hard_neg_tile_count += 1

        else:
            # Negative image: emit a small number of random tiles for domain
            # adaptation (teaches the model what BrentImages background looks like).
            chosen = rng.sample(
                [(x0, y0) for y0 in ys for x0 in xs],
                k=min(neg_tiles_per_image, len(xs) * len(ys)),
            )
            for x0, y0 in chosen:
                fname = _tile_file_name(orig_path, x0, y0, native_tile_size)
                out_images.append({
                    "id": next_img_id,
                    "file_name": fname,
                    "width": native_tile_size,
                    "height": native_tile_size,
                })
                next_img_id += 1
                negative_tile_count += 1

    result = {
        "info": {
            "description": (
                f"BrentImages tiled training crops "
                f"(native_tile_size={native_tile_size}, overlap={overlap})"
            ),
            "version": "1.0",
            "source": str(src_path),
        },
        "licenses": [],
        "categories": _CANONICAL_CATEGORIES,
        "images": out_images,
        "annotations": out_annotations,
    }

    with open(out_path, "w") as f:
        json.dump(result, f)

    logger.info(
        "%s: %d positive + %d hard-neg + %d negative = %d total tiles  "
        "(%d annotations, %d source images skipped)",
        out_path.name,
        positive_tile_count,
        hard_neg_tile_count,
        negative_tile_count,
        len(out_images),
        len(out_annotations),
        skipped_images,
    )
    return out_path


if __name__ == "__main__":
    args = _parse_args()
    if args.out is None:
        stem = args.src.stem
        args.out = _ANN_DIR / f"{stem}_tiled_ts{args.native_tile_size}.json"
    build_tiled_brentimages_json(
        src_path=args.src,
        out_path=args.out,
        native_tile_size=args.native_tile_size,
        overlap=args.overlap,
        min_area_fraction=args.min_area_fraction,
        neg_tiles_per_image=args.neg_tiles_per_image,
        hard_neg_per_pos=args.hard_neg_per_pos,
    )
