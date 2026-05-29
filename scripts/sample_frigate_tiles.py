"""Diversity-maximising subset selection from Frigate tiled training crops.

Frigate provides 717 400×400 px tiles from a single QHY600M observation night.
Using all 717 tiles risks dominating the training distribution with a single
instrument and single night of sky conditions.  This script selects a
geometry-diverse subset of ~150–200 tiles that covers the full range of:

  - Streak orientation (0–180°)
  - Streak length within the Frigate short-band range (~20–80 px)
  - Streak aspect ratio (thin vs. moderate)
  - Tile position within the original frame (spatial diversity)

Algorithm: Furthest-Point Sampling (greedy max-min-distance) in a normalised
feature space.  At each step the tile that maximises its minimum distance to
any already-selected tile is added.

Negative tiles (no streak annotation) are handled separately: a random sample
of up to ``--max-negatives`` tiles is included alongside the positive subset.

Input
-----
  data/annotations/frigate_tiled_train.json

Output
------
  data/annotations/frigate_diversity_<N>.json  (N = actual tile count)
  data/features/frigate_tile_features.csv      (features for all tiles)

Usage
-----
python scripts/sample_frigate_tiles.py

# With explicit target and seed:
python scripts/sample_frigate_tiles.py --n-tiles 200 --seed 42

# Inspect features without writing the subset JSON:
python scripts/sample_frigate_tiles.py --features-only
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_INPUT = _REPO_ROOT / "data/annotations/frigate_tiled_train.json"
_FEATURES_DIR = _REPO_ROOT / "data/features"
_ANNOTATIONS_DIR = _REPO_ROOT / "data/annotations"
_CANONICAL_CATEGORIES = [{"id": 1, "name": "streak", "supercategory": "satellite"}]

# Frigate tile geometry is already in model-input space (400×400 px)
_TILE_SIZE = 400.0


# ---------------------------------------------------------------------------
# Feature extraction from COCO bbox
# ---------------------------------------------------------------------------

def _bbox_features(bbox: list[float]) -> dict[str, float]:
    """Extract geometry features from a COCO [x, y, w, h] bbox.

    For Frigate tiles, the bbox is already in tile coordinates (400×400 px).
    Streak length is the diagonal; angle is approximated from the bbox
    major axis.

    Returns:
        Dict with length_px, aspect_ratio, angle_deg.
    """
    x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    length_px = math.hypot(w, h)
    # aspect_ratio: major / minor axis of bbox (same convention as Atwood OBB)
    major = max(w, h)
    minor = min(w, h)
    aspect_ratio = major / max(minor, 1e-6)
    # Angle of major axis from horizontal (approximate for axis-aligned bbox)
    angle_deg = math.degrees(math.atan2(h, w)) % 180.0
    return {
        "length_px": round(length_px, 2),
        "aspect_ratio": round(aspect_ratio, 3),
        "angle_deg": round(angle_deg, 2),
    }


def _tile_position(file_name: str) -> tuple[float, float]:
    """Parse tile origin (tx, ty) from a Frigate tile filename.

    Virtual filename format:  stem__tx{x}_ty{y}_ts{size}.ext
    Returns (0.0, 0.0) if the pattern does not match.
    """
    import re
    m = re.search(r"__tx(\d+)_ty(\d+)_ts\d+", Path(file_name).name)
    if m:
        return float(m.group(1)), float(m.group(2))
    return 0.0, 0.0


# ---------------------------------------------------------------------------
# Furthest-Point Sampling
# ---------------------------------------------------------------------------

def _fps(
    features: np.ndarray,
    n_select: int,
    seed: int = 42,
) -> list[int]:
    """Greedy Furthest-Point Sampling (FPS) in normalised feature space.

    Selects ``n_select`` indices from ``features`` that maximally cover
    the feature distribution.

    Args:
        features: Array of shape (N, D) — rows are tiles, cols are features.
        n_select: Number of tiles to select.
        seed: Random seed for the initial point.

    Returns:
        List of selected row indices (length = min(n_select, N)).
    """
    N = features.shape[0]
    n_select = min(n_select, N)
    if n_select == N:
        return list(range(N))

    rng = np.random.default_rng(seed)
    selected: list[int] = []

    # Initialise: start from the point nearest the feature-space centroid
    centroid = features.mean(axis=0)
    dists_to_centroid = np.linalg.norm(features - centroid, axis=1)
    first = int(np.argmin(dists_to_centroid))
    selected.append(first)

    # min_dist[i] = minimum distance from point i to the selected set so far
    min_dist = np.linalg.norm(features - features[first], axis=1)
    min_dist[first] = 0.0

    for _ in range(n_select - 1):
        # Furthest point from the current selected set
        farthest = int(np.argmax(min_dist))
        selected.append(farthest)

        # Update minimum distances
        new_dists = np.linalg.norm(features - features[farthest], axis=1)
        min_dist = np.minimum(min_dist, new_dists)
        min_dist[farthest] = 0.0

    return selected


def _normalise_features(matrix: np.ndarray) -> np.ndarray:
    """Normalise each column to [0, 1]; handle zero-range columns."""
    lo = matrix.min(axis=0)
    hi = matrix.max(axis=0)
    rng = hi - lo
    rng[rng < 1e-9] = 1.0   # avoid division by zero for constant features
    return (matrix - lo) / rng


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diversity-maximising Frigate tile subset selection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        help="Frigate tiled train COCO JSON "
             "(default: data/annotations/frigate_tiled_train.json)",
    )
    p.add_argument(
        "--n-tiles",
        type=int,
        default=200,
        help="Target number of positive tiles to select (default: 200)",
    )
    p.add_argument(
        "--max-negatives",
        type=int,
        default=50,
        help="Maximum negative (no-annotation) tiles to include (default: 50)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_ANNOTATIONS_DIR,
        help="Directory for the output COCO JSON "
             "(default: data/annotations)",
    )
    p.add_argument(
        "--features-only",
        action="store_true",
        help="Extract and write features CSV only; do not write subset JSON.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    with open(args.input) as fh:
        coco = json.load(fh)

    images: list[dict] = coco["images"]
    annotations: list[dict] = coco.get("annotations", [])

    # Map image_id → list of annotations
    from collections import defaultdict
    anns_by_img: dict[int, list[dict]] = defaultdict(list)
    for ann in annotations:
        anns_by_img[ann["image_id"]].append(ann)

    # Partition into positive (has annotation) and negative (no annotation)
    positive_imgs: list[dict] = []
    negative_imgs: list[dict] = []
    for img in images:
        if anns_by_img[img["id"]]:
            positive_imgs.append(img)
        else:
            negative_imgs.append(img)

    logger.info(
        "Input: %d total tiles  (%d positive, %d negative)",
        len(images), len(positive_imgs), len(negative_imgs),
    )

    # ---------------------------------------------------------------------------
    # Extract features for positive tiles
    # ---------------------------------------------------------------------------
    tile_features: list[dict] = []

    for img in positive_imgs:
        img_anns = anns_by_img[img["id"]]
        # Use the longest annotation's bbox as the representative geometry
        longest_ann = max(
            img_anns,
            key=lambda a: math.hypot(a["bbox"][2], a["bbox"][3]),
        )
        geom = _bbox_features(longest_ann["bbox"])
        tx, ty = _tile_position(img["file_name"])

        # Dominant orientation angle — use circular (sin, cos) encoding so that
        # 0° and 180° are treated as close (streak orientation is undirected)
        angle_rad = math.radians(geom["angle_deg"])
        angle_sin = math.sin(2 * angle_rad)   # period π (undirected angle)
        angle_cos = math.cos(2 * angle_rad)

        tile_features.append({
            "image_id": img["id"],
            "file_name": img["file_name"],
            "tile_x": tx,
            "tile_y": ty,
            "length_px": geom["length_px"],
            "aspect_ratio": geom["aspect_ratio"],
            "angle_deg": geom["angle_deg"],
            "angle_sin": round(angle_sin, 4),
            "angle_cos": round(angle_cos, 4),
            "n_annotations": len(img_anns),
        })

    # ---------------------------------------------------------------------------
    # Write full features CSV (before subsetting)
    # ---------------------------------------------------------------------------
    _FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    features_path = _FEATURES_DIR / "frigate_tile_features.csv"
    _CSV_FIELDS = [
        "image_id", "file_name", "tile_x", "tile_y",
        "length_px", "aspect_ratio", "angle_deg",
        "angle_sin", "angle_cos", "n_annotations",
    ]
    with open(features_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(tile_features)
    logger.info("Feature CSV written: %s  (%d rows)", features_path, len(tile_features))

    if args.features_only:
        return

    # ---------------------------------------------------------------------------
    # Build feature matrix for FPS
    # ---------------------------------------------------------------------------
    # Feature dimensions: length_px, log(aspect_ratio), angle_sin, angle_cos,
    # tile_x_norm, tile_y_norm
    #
    # Spatial position (tile_x, tile_y) captures frame-level diversity — tiles
    # from different parts of the original sensor have different backgrounds.
    # Circular angle encoding ensures wraparound continuity at 0°/180°.
    # Log-aspect captures the wide dynamic range of aspect ratios compactly.

    feat_matrix = np.array([
        [
            f["length_px"],
            math.log1p(f["aspect_ratio"]),
            f["angle_sin"],
            f["angle_cos"],
            f["tile_x"],
            f["tile_y"],
        ]
        for f in tile_features
    ], dtype=np.float32)

    feat_norm = _normalise_features(feat_matrix)

    # ---------------------------------------------------------------------------
    # FPS selection
    # ---------------------------------------------------------------------------
    n_target = min(args.n_tiles, len(positive_imgs))
    selected_indices = _fps(feat_norm, n_target, seed=args.seed)
    selected_img_ids = {tile_features[i]["image_id"] for i in selected_indices}

    logger.info(
        "FPS selected %d / %d positive tiles  (target=%d)",
        len(selected_img_ids), len(positive_imgs), n_target,
    )

    # ---------------------------------------------------------------------------
    # Add negative tiles (random sample up to max-negatives)
    # ---------------------------------------------------------------------------
    rng = random.Random(args.seed)
    neg_sample = negative_imgs[:args.max_negatives]
    if len(negative_imgs) > args.max_negatives:
        neg_sample = rng.sample(negative_imgs, args.max_negatives)
    neg_ids = {img["id"] for img in neg_sample}

    total_ids = selected_img_ids | neg_ids
    logger.info(
        "Output set: %d positive + %d negative = %d total tiles",
        len(selected_img_ids), len(neg_ids), len(total_ids),
    )

    # ---------------------------------------------------------------------------
    # Build output COCO JSON
    # ---------------------------------------------------------------------------
    selected_images = [img for img in images if img["id"] in total_ids]
    selected_anns = [ann for ann in annotations if ann["image_id"] in total_ids]

    # Re-number IDs sequentially
    old_to_new: dict[int, int] = {}
    new_images = []
    for new_id, img in enumerate(selected_images, start=1):
        old_to_new[img["id"]] = new_id
        new_img = dict(img)
        new_img["id"] = new_id
        new_images.append(new_img)

    new_anns = []
    for new_id, ann in enumerate(selected_anns, start=1):
        new_ann = dict(ann)
        new_ann["id"] = new_id
        new_ann["image_id"] = old_to_new[ann["image_id"]]
        new_anns.append(new_ann)

    from datetime import date
    output_coco = {
        "info": {
            "description": (
                f"Frigate diversity subset — {len(new_images)} tiles "
                f"({len(selected_img_ids)} positive, {len(neg_ids)} negative).  "
                "Selected by FPS in length×aspect×angle×position feature space."
            ),
            "version": "1.0",
            "date_created": date.today().isoformat(),
        },
        "licenses": [],
        "categories": _CANONICAL_CATEGORIES,
        "images": new_images,
        "annotations": new_anns,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Write a count-suffixed copy for human inspection
    versioned_path = args.output_dir / f"frigate_diversity_{len(new_images)}.json"
    with open(versioned_path, "w") as fh:
        json.dump(output_coco, fh)
    # Also write to the stable canonical name used by the session manifest
    stable_path = args.output_dir / "frigate_diversity.json"
    with open(stable_path, "w") as fh:
        json.dump(output_coco, fh)

    logger.info(
        "Subset JSON written: %s  (%d images, %d annotations)",
        versioned_path, len(new_images), len(new_anns),
    )
    logger.info("Stable symlink: %s", stable_path)

    # ---------------------------------------------------------------------------
    # Distribution summary
    # ---------------------------------------------------------------------------
    selected_feats = [tile_features[i] for i in selected_indices]
    lengths = [f["length_px"] for f in selected_feats]
    angles = [f["angle_deg"] for f in selected_feats]
    aspects = [f["aspect_ratio"] for f in selected_feats]

    logger.info(
        "Length distribution — min=%.0f  median=%.0f  max=%.0f",
        np.min(lengths), np.median(lengths), np.max(lengths),
    )
    angle_arr = np.array(angles)
    logger.info(
        "Angle distribution  — min=%.0f  median=%.0f  max=%.0f  (°)",
        np.min(angle_arr), np.median(angle_arr), np.max(angle_arr),
    )
    logger.info(
        "Aspect ratio        — min=%.1f  median=%.1f  max=%.1f",
        np.min(aspects), np.median(aspects), np.max(aspects),
    )


if __name__ == "__main__":
    main()
