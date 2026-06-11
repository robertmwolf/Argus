#!/usr/bin/env python3
"""Export YOLO OBB dataset for Run 17 from real FITS + synthetic NPY sources.

Sources (all verified present on external drive):
  Train real   : all_train_run5.json           — 1989 full FITS (6248×4176)
  Train synth  : synth_run13_short_npy.json    — 2810 NPY (1800×1800)
               + synth_run13_medium_npy.json   — 1124 NPY (1800×1800)
  Val          : val_atwood.json               — 240 full FITS (6248×4176)

Both FITS and NPY sources are sub-tiled to TILE_SIZE×TILE_SIZE (default 400 px)
with OVERLAP fraction (default 0.5) to match the Run 15 training scale.

Normalisation: ZScore → clip [-3σ, +3σ] → rescale [0, 255] uint8 → 3-channel PNG.

YOLO OBB label format per object (one line, DOTA polygon):
    class_id  x1 y1 x2 y2 x3 y3 x4 y4
where coordinates are the 4 OBB corners normalised to [0,1] by sub-tile dimensions.

Usage:
    # Full export (overnight):
    python scripts/export_yolo_obb_dataset.py \\
        --output-dir /Volumes/External/TrainingData/yolo_run17_dataset

    # Dry run (count tiles, no writes):
    python scripts/export_yolo_obb_dataset.py --output-dir /tmp/dry --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

ANN_ROOT = Path("/Volumes/External/TrainingData/annotations")

TRAIN_SOURCES = [
    ANN_ROOT / "all_train_run5.json",
    ANN_ROOT / "synth_run13_short_npy.json",
    ANN_ROOT / "synth_run13_medium_npy.json",
]
VAL_SOURCES = [
    ANN_ROOT / "val_atwood.json",
]

DEFAULT_TILE_SIZE = 400
DEFAULT_OVERLAP   = 0.5


# ── Normalisation ─────────────────────────────────────────────────────────────

def array_to_uint8(arr: np.ndarray) -> np.ndarray:
    """ZScore-normalise float array → uint8 grayscale (2-D)."""
    std = float(arr.std())
    if std < 1e-6:
        return np.zeros(arr.shape, dtype=np.uint8)
    z = (arr - arr.mean()) / std
    z = np.clip(z, -3.0, 3.0)
    return ((z + 3.0) / 6.0 * 255.0).astype(np.uint8)


def save_png(gray: np.ndarray, path: Path) -> None:
    """Write grayscale uint8 array as 3-channel PNG for YOLO."""
    import cv2
    cv2.imwrite(str(path), np.stack([gray, gray, gray], axis=-1))


# ── Source loaders ────────────────────────────────────────────────────────────

def load_fits(path: Path) -> np.ndarray:
    """Load a FITS file and return a 2-D float32 array."""
    from astropy.io import fits as astrofits
    with astrofits.open(str(path)) as hdul:
        data = hdul[0].data
    if data is None:
        raise ValueError(f"No data in primary HDU: {path}")
    arr = np.squeeze(data).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D FITS, got shape {arr.shape}: {path}")
    return arr


def load_source(file_name: str) -> np.ndarray:
    """Load a source file (FITS or NPY) as float32 2-D array."""
    p = Path(file_name)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.suffix.lower() in (".fits", ".fit"):
        return load_fits(p)
    return np.load(str(p)).astype(np.float32)


# ── OBB helpers ───────────────────────────────────────────────────────────────

def _clip_segment_to_tile(
    cx: float, cy: float, half_len: float, angle_deg: float,
    tw: int, th: int,
) -> float:
    """Return the visible half-length of a streak segment clipped to [0,tw]×[0,th].

    The streak runs from (cx-half_len*cos θ, cy-half_len*sin θ) to the opposite
    end.  We parametrise points along the streak as p(t) = (cx,cy) + t*(cos θ, sin θ)
    for t ∈ [-half_len, +half_len] and intersect with the tile rectangle.
    """
    theta = math.radians(angle_deg)
    dx, dy = math.cos(theta), math.sin(theta)

    # Slab intersections for each tile edge
    t_min, t_max = -half_len, half_len

    for (d, c, lo, hi) in [(dx, cx, 0.0, float(tw)), (dy, cy, 0.0, float(th))]:
        if abs(d) < 1e-9:
            if c < lo or c > hi:
                return 0.0  # parallel and outside — no intersection
        else:
            ta = (lo - c) / d
            tb = (hi - c) / d
            t_min = max(t_min, min(ta, tb))
            t_max = min(t_max, max(ta, tb))

    if t_max <= t_min:
        return 0.0
    return (t_max - t_min) / 2.0  # half-length of visible segment


def clip_obb(
    obb: dict[str, float], x0: int, y0: int, tw: int, th: int
) -> dict[str, float] | None:
    """Translate OBB into sub-tile coords and clip its length to the tile boundary.

    Returns None if the centre lies outside the tile or the visible length
    is zero.  The OBB height (streak thickness) is unchanged; the width
    (streak length) is clipped to the visible segment within the tile.
    """
    cx = obb["cx"] - x0
    cy = obb["cy"] - y0
    if cx < 0 or cx >= tw or cy < 0 or cy >= th:
        return None

    half_len = _clip_segment_to_tile(cx, cy, obb["w"] / 2.0, obb["angle_deg"], tw, th)
    if half_len <= 0:
        return None

    return {
        "cx": cx,
        "cy": cy,
        "w": half_len * 2.0,   # visible streak length, clipped to tile
        "h": obb["h"],          # streak thickness — unchanged
        "angle_deg": obb["angle_deg"],
    }


def obb_to_yolo(obb: dict[str, float], tw: int, th: int, class_id: int = 0) -> str:
    """Format OBB as a YOLO OBB label in DOTA polygon format: class x1 y1 x2 y2 x3 y3 x4 y4."""
    cx, cy = obb["cx"], obb["cy"]
    hw, hh = obb["w"] / 2.0, obb["h"] / 2.0
    t = math.radians(obb["angle_deg"])
    cos_t, sin_t = math.cos(t), math.sin(t)
    offs = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    pts = []
    for dx, dy in offs:
        px = max(0.0, min(1.0, (cx + dx * cos_t - dy * sin_t) / tw))
        py = max(0.0, min(1.0, (cy + dx * sin_t + dy * cos_t) / th))
        pts.extend([px, py])
    return f"{class_id} " + " ".join(f"{p:.6f}" for p in pts)


# ── Sub-tiling ────────────────────────────────────────────────────────────────

def sub_tile_origins(
    src_w: int, src_h: int, tile_size: int, overlap: float
) -> list[tuple[int, int]]:
    """Return deduplicated (x, y) origins covering the full source image."""
    stride = max(1, int(tile_size * (1.0 - overlap)))
    pts: list[tuple[int, int]] = []

    def row_origins(dim: int) -> list[int]:
        os: list[int] = []
        x = 0
        while x + tile_size <= dim:
            os.append(x)
            x += stride
        if not os or os[-1] + tile_size < dim:
            os.append(max(0, dim - tile_size))
        return os

    for y in row_origins(src_h):
        for x in row_origins(src_w):
            pts.append((x, y))
    return list(dict.fromkeys(pts))


# ── Per-image export ──────────────────────────────────────────────────────────

def export_image(
    img_meta: dict[str, Any],
    annotations: list[dict[str, Any]],
    img_dir: Path,
    lbl_dir: Path,
    tile_size: int,
    overlap: float,
    dry_run: bool,
) -> tuple[int, int]:
    """Tile one source image into 400 px crops and write PNG + YOLO labels.

    Returns (n_tiles_written, n_annotations_written).
    """
    file_name = img_meta["file_name"]
    src_w = img_meta["width"]
    src_h = img_meta["height"]

    if not dry_run:
        try:
            arr = load_source(file_name)
            src_h, src_w = arr.shape  # authoritative dims from file
            uint8 = array_to_uint8(arr)
        except (FileNotFoundError, ValueError) as exc:
            log.warning("Skipping %s — %s", file_name, exc)
            return 0, 0
    else:
        # Dry run: use metadata dims, skip file I/O
        if not Path(file_name).exists():
            return 0, 0

    needs_subtile = src_w > tile_size or src_h > tile_size
    origins = sub_tile_origins(src_w, src_h, tile_size, overlap) if needs_subtile else [(0, 0)]

    stem_base = f"{img_meta['id']:07d}"
    has_anns = len(annotations) > 0
    n_tiles = n_anns = 0

    for x0, y0 in origins:
        tw = min(tile_size, src_w - x0)
        th = min(tile_size, src_h - y0)

        sub_obbs = [
            c for ann in annotations
            for c in [clip_obb(ann["obb"], x0, y0, tw, th)]
            if c is not None and "obb" in ann
        ]

        # Skip streak-free sub-tiles of positive source images.
        # Pure-negative source images (no annotations) are exported as background tiles.
        if has_anns and not sub_obbs:
            continue

        stem = f"{stem_base}_{x0}_{y0}" if needs_subtile else stem_base

        if not dry_run:
            save_png(uint8[y0: y0 + th, x0: x0 + tw], img_dir / f"{stem}.png")
            (lbl_dir / f"{stem}.txt").write_text(
                "\n".join(obb_to_yolo(o, tw, th) for o in sub_obbs)
            )

        n_tiles += 1
        n_anns += len(sub_obbs)

    return n_tiles, n_anns


# ── Split export ──────────────────────────────────────────────────────────────

def export_sources(
    ann_paths: list[Path],
    out_root: Path,
    split: str,
    tile_size: int,
    overlap: float,
    dry_run: bool,
) -> None:
    """Merge multiple COCO annotation files and export a single split."""
    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    if not dry_run:
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

    total_tiles = total_anns = total_skipped = 0

    for ann_path in ann_paths:
        log.info("[%s] Loading %s", split, ann_path.name)
        data = json.loads(ann_path.read_text())
        images = {img["id"]: img for img in data["images"]}
        anns_by_img: dict[int, list] = {iid: [] for iid in images}
        for ann in data["annotations"]:
            anns_by_img[ann["image_id"]].append(ann)

        n = len(images)
        file_tiles = file_anns = file_skipped = 0
        for idx, (img_id, img_meta) in enumerate(images.items()):
            if idx % 200 == 0:
                log.info("  [%s/%s] %d / %d ...", split, ann_path.stem, idx, n)
            t, a = export_image(
                img_meta, anns_by_img[img_id],
                img_dir, lbl_dir, tile_size, overlap, dry_run,
            )
            if t == 0:
                file_skipped += 1
            file_tiles += t
            file_anns  += a

        log.info(
            "  [%s/%s] → %d tiles, %d annotations, %d source images skipped",
            split, ann_path.stem, file_tiles, file_anns, file_skipped,
        )
        total_tiles   += file_tiles
        total_anns    += file_anns
        total_skipped += file_skipped

    log.info("[%s] TOTAL — %d tiles, %d annotations, %d skipped", split, total_tiles, total_anns, total_skipped)


# ── dataset.yaml ──────────────────────────────────────────────────────────────

def write_yaml(out_root: Path) -> None:
    yaml_path = out_root / "dataset.yaml"
    yaml_path.write_text(
        "# YOLO OBB dataset — Run 17 (Run 15 distribution, 400px tiles, zscore norm)\n"
        "# Train: all_train_run5 (real FITS) + synth_run13_short + synth_run13_medium (NPY)\n"
        "# Val:   val_atwood (real FITS)\n"
        "\n"
        f"path: {out_root.resolve()}\n"
        "train: images/train\n"
        "val:   images/val\n"
        "\n"
        "nc: 1\n"
        "names:\n"
        "  - streak\n"
    )
    log.info("Wrote %s", yaml_path)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--overlap",   type=float, default=DEFAULT_OVERLAP)
    parser.add_argument("--dry-run",   action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        log.info("DRY RUN — no files will be written")

    export_sources(TRAIN_SOURCES, args.output_dir, "train", args.tile_size, args.overlap, args.dry_run)
    export_sources(VAL_SOURCES,   args.output_dir, "val",   args.tile_size, args.overlap, args.dry_run)

    if not args.dry_run:
        write_yaml(args.output_dir)
        log.info("Dataset ready: %s", args.output_dir)
    else:
        log.info("Dry run complete.")


if __name__ == "__main__":
    main()
