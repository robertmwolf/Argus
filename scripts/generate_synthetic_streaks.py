#!/usr/bin/env python3
"""Augment negative NPY tiles with synthetic satellite streaks.

Reads negative tiles from an existing NPY-based annotation JSON, renders
physically-motivated streaks (Gaussian PSF, realistic SNR, random geometry),
and writes augmented NPY files + a new COCO annotation JSON.

The output JSON can be merged with real endpoint annotations
merge helpers, or passed directly to cache_dinov3_heatmap_features.py.

Streak model
------------
A streak is a 1-D Gaussian smeared along a line:

    I(x, y) = A · exp(-(d_perp)² / (2σ²))

where d_perp is perpendicular distance from the streak axis, A is peak
amplitude (chosen to match target SNR relative to the tile's local noise),
and σ controls the PSF width (~1.5 px → FWHM ~3.5 px, matching real data).

SNR is defined as A / mad_std(tile), where mad_std = 1.4826 × median(|x - median(x)|).
This matches how peak_snr is computed for real annotations.

Usage
-----
    python scripts/generate_synthetic_streaks.py \\
        --neg-annotations /Volumes/External/TrainingData/annotations/all_train_run10a_tiled_npy.json \\
        --output-dir      /tmp/argus_synth/tiles \\
        --output-ann      /Volumes/External/TrainingData/annotations/synth_run10a_npy.json \\
        --n-per-neg 2 --seed 42

    # Then cache + merge as normal:
    python scripts/cache_dinov3_heatmap_features.py \\
        --annotations .../synth_run10a_npy.json \\
        --output-dir  .../synth_run10a_train ...
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Streak rendering ──────────────────────────────────────────────────────────

def _mad_std(arr: np.ndarray) -> float:
    """Robust noise estimate: 1.4826 × MAD."""
    med = float(np.median(arr))
    return 1.4826 * float(np.median(np.abs(arr - med)))


def _render_streak(
    tile: np.ndarray,
    cx: float,
    cy: float,
    length: float,
    angle_deg: float,
    snr: float,
    psf_sigma: float = 1.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a copy of *tile* with one synthetic streak added.

    Parameters
    ----------
    cx, cy      : streak centre in pixel coordinates
    length      : full streak length in pixels
    angle_deg   : CCW angle from +x axis (0 = horizontal)
    snr         : peak signal / mad_std(tile) — controls brightness
    psf_sigma   : Gaussian cross-section sigma in pixels
    """
    if rng is None:
        rng = np.random.default_rng()

    h, w = tile.shape
    noise_std = _mad_std(tile)
    if noise_std <= 0:
        noise_std = float(np.std(tile)) or 1.0
    amplitude = snr * noise_std

    θ = math.radians(angle_deg)
    cos_θ, sin_θ = math.cos(θ), math.sin(θ)

    # half-length end points
    half = length / 2.0
    x0, y0 = cx - half * cos_θ, cy - half * sin_θ
    x1, y1 = cx + half * cos_θ, cy + half * sin_θ

    # bounding box with margin
    margin = int(math.ceil(psf_sigma * 4))
    xmin = max(0, int(min(x0, x1)) - margin)
    xmax = min(w, int(max(x0, x1)) + margin + 1)
    ymin = max(0, int(min(y0, y1)) - margin)
    ymax = min(h, int(max(y0, y1)) + margin + 1)

    if xmax <= xmin or ymax <= ymin:
        return tile.copy()

    xs = np.arange(xmin, xmax, dtype=np.float32)
    ys = np.arange(ymin, ymax, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)

    # perpendicular distance from each pixel to the streak axis
    dx = XX - cx
    dy = YY - cy
    # parallel projection along streak direction
    d_par = dx * cos_θ + dy * sin_θ
    # perpendicular distance
    d_perp = -dx * sin_θ + dy * cos_θ

    # clip to streak length (streak doesn't extend beyond endpoints)
    d_par_clipped = np.clip(d_par, -half, half)
    # distance from the nearest point on the segment
    nearest_x = cx + d_par_clipped * cos_θ
    nearest_y = cy + d_par_clipped * sin_θ
    dist2 = (XX - nearest_x) ** 2 + (YY - nearest_y) ** 2

    streak_patch = amplitude * np.exp(-dist2 / (2 * psf_sigma ** 2))

    out = tile.copy()
    out[ymin:ymax, xmin:xmax] += streak_patch.astype(np.float32)
    return out


# ── Endpoint annotation helper ───────────────────────────────────────────────

def _make_annotation(
    ann_id: int,
    img_id: int,
    cx: float,
    cy: float,
    length: float,
    angle_deg: float,
) -> dict[str, Any]:
    theta = math.radians(angle_deg)
    half_dx = 0.5 * length * math.cos(theta)
    half_dy = 0.5 * length * math.sin(theta)
    return {
        "id": ann_id,
        "image_id": img_id,
        "category_id": 1,
        "iscrowd": 0,
        "x1": cx - half_dx,
        "y1": cy - half_dy,
        "x2": cx + half_dx,
        "y2": cy + half_dy,
        "attributes": {
            "synthetic": True,
            "length_px": length,
        },
    }


# ── Length distribution matching real data ───────────────────────────────────

def _sample_length(rng: np.random.Generator, tile_size: int = 400) -> float:
    """Sample streak length with ~realistic mix: 40% short, 40% medium, 20% long."""
    r = rng.random()
    if r < 0.40:
        return float(rng.uniform(20, 80))    # short
    elif r < 0.80:
        return float(rng.uniform(80, 200))   # medium
    else:
        return float(rng.uniform(200, min(320, tile_size - 20)))  # long


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--neg-annotations", required=True,
                        help="COCO JSON (NPY-based) to source negative tiles from")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write synthetic NPY tiles")
    parser.add_argument("--output-ann", required=True,
                        help="Path to write output COCO JSON")
    parser.add_argument("--n-per-neg", type=int, default=2,
                        help="Synthetic tiles to generate per negative source tile (default: 2)")
    parser.add_argument("--snr-min", type=float, default=3.0,
                        help="Minimum streak SNR (default: 3.0)")
    parser.add_argument("--snr-max", type=float, default=12.0,
                        help="Maximum streak SNR (default: 12.0)")
    parser.add_argument("--psf-sigma", type=float, default=1.5,
                        help="Gaussian cross-section sigma in pixels (default: 1.5)")
    parser.add_argument("--multi-streak-prob", type=float, default=0.15,
                        help="Probability of placing a second streak on the same tile (default: 0.15)")
    parser.add_argument("--length-range", type=float, nargs=2, metavar=("LO", "HI"),
                        default=None,
                        help="Override length distribution with uniform [LO, HI] px in tile space. "
                             "When omitted uses the default 40/40/20 short/medium/long bucketed mix. "
                             "For tiles that will be resized to model input, set LO/HI in tile pixels: "
                             "apparent_length = drawn_length * (model_input_size / tile_size).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading annotations from %s", args.neg_annotations)
    with open(args.neg_annotations) as f:
        coco = json.load(f)

    ann_img_ids = {a["image_id"] for a in coco.get("annotations", [])}
    neg_imgs = [img for img in coco["images"] if img["id"] not in ann_img_ids]
    logger.info("Negative tiles available: %d", len(neg_imgs))

    out_images: list[dict] = []
    out_annotations: list[dict] = []
    next_img_id = 1
    next_ann_id = 1
    missing = 0
    written = 0

    for src_img in neg_imgs:
        src_path = Path(src_img["file_name"])
        if not src_path.exists():
            missing += 1
            if missing <= 3:
                logger.warning("Missing NPY: %s — skipping", src_path)
            continue

        try:
            raw = np.load(src_path)
        except Exception as e:
            logger.warning("Failed to load %s: %s", src_path, e)
            continue

        if raw.dtype == np.uint8:
            # Full-image normalisation was applied at convert_tiles_to_npy time.
            base_tile = raw.astype(np.float32)
        else:
            base_tile = raw

        h, w = base_tile.shape[:2]

        for _ in range(args.n_per_neg):
            if args.length_range is not None:
                length = float(rng.uniform(args.length_range[0], args.length_range[1]))
            else:
                length = _sample_length(rng, tile_size=min(h, w))
            angle_deg = float(rng.uniform(0, 180))

            # keep streak centre far enough from edges that full length fits
            margin = int(length / 2) + 10
            if margin * 2 >= w or margin * 2 >= h:
                margin = min(w, h) // 4
            cx = float(rng.uniform(margin, w - margin))
            cy = float(rng.uniform(margin, h - margin))
            snr = float(rng.uniform(args.snr_min, args.snr_max))

            aug_tile = _render_streak(base_tile, cx, cy, length, angle_deg, snr, args.psf_sigma, rng)

            streaks = [(cx, cy, length, angle_deg)]

            # optional second streak
            if rng.random() < args.multi_streak_prob:
                l2 = _sample_length(rng, tile_size=min(h, w))
                a2 = float(rng.uniform(0, 180))
                m2 = int(l2 / 2) + 10
                if m2 * 2 < w and m2 * 2 < h:
                    cx2 = float(rng.uniform(m2, w - m2))
                    cy2 = float(rng.uniform(m2, h - m2))
                    snr2 = float(rng.uniform(args.snr_min, args.snr_max))
                    aug_tile = _render_streak(aug_tile, cx2, cy2, l2, a2, snr2, args.psf_sigma, rng)
                    streaks.append((cx2, cy2, l2, a2))

            out_path = out_dir / f"{next_img_id}.npy"
            np.save(out_path, aug_tile)

            img_entry: dict[str, Any] = {
                "id": next_img_id,
                "file_name": str(out_path),
                "width": w,
                "height": h,
                "orig_image_id": src_img["id"],
                "tile_origin": src_img.get("tile_origin", [0, 0]),
                "synthetic": True,
            }
            out_images.append(img_entry)

            for (scx, scy, sl, sa) in streaks:
                out_annotations.append(_make_annotation(next_ann_id, next_img_id, scx, scy, sl, sa))
                next_ann_id += 1

            next_img_id += 1
            written += 1

    if missing > 3:
        logger.warning("... and %d more missing NPY files", missing - 3)
    logger.info("Wrote %d synthetic tiles (%d annotations) to %s", written, len(out_annotations), out_dir)

    out_coco = {
        "info": {"description": "ARGUS synthetic streak augmentation", "version": "1.0"},
        "licenses": [],
        "categories": coco.get("categories", [{"id": 1, "name": "streak", "supercategory": "satellite"}]),
        "images": out_images,
        "annotations": out_annotations,
    }
    Path(args.output_ann).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_ann, "w") as f:
        json.dump(out_coco, f)
    logger.info("Annotation JSON written to %s", args.output_ann)


if __name__ == "__main__":
    main()
