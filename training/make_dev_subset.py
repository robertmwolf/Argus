"""Build the 50-image annotated dev subset for local training.

Generates synthetic FITS images with known streak positions, writes YOLO OBB
label files, and produces data/annotations/dev_subset.json in COCO format.

Subset composition (matches CLAUDE.md spec):
  20 images  — no streak (background only)
  20 images  — short streak  (<269 px)
  10 images  — long  streak  (≥269 px)
  ─────────────────────────────────────
  50 images total

Usage::

    python -m training.make_dev_subset               # writes to default paths
    python -m training.make_dev_subset --output-dir /tmp/dev --small
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

# Repo root so imports work when run as script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.make_test_fits import _make_background, _add_stars, _make_wcs_header
from astropy.io import fits

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants matching CLAUDE.md spec
# ---------------------------------------------------------------------------
_N_BLANK = 20
_N_SHORT = 20   # streak length < 269 px
_N_LONG  = 10   # streak length ≥ 269 px

_SHORT_MAX_PX = 268.9   # exclusive upper bound for "short"
_LONG_MIN_PX  = 269.0   # inclusive lower bound for "long"

_DEFAULT_WIDTH  = 512    # small-ish for fast local iteration
_DEFAULT_HEIGHT = 512

_STREAK_WIDTH_PX = 3.0   # OBB minor axis when creating labels


# ---------------------------------------------------------------------------
# Synthetic FITS + label generation
# ---------------------------------------------------------------------------

def _add_streak_controlled(
    image: np.ndarray,
    rng: np.random.Generator,
    target_min_px: float,
    target_max_px: float,
    streak_brightness: float = 6000.0,
    streak_width: float = 1.5,
) -> dict:
    """Inject a streak with length in [target_min_px, target_max_px].

    Returns dict with: cx, cy, w, h, angle_deg, x_start, y_start, x_end, y_end.
    """
    h_img, w_img = image.shape
    margin = max(10, min(30, w_img // 10, h_img // 10))

    for _attempt in range(50):
        angle_deg = rng.uniform(10, 170)
        angle_rad = np.radians(angle_deg)

        length = rng.uniform(target_min_px, target_max_px)

        # Start inside the image, leaving room for the full streak
        x0 = rng.uniform(margin, w_img - margin)
        y0 = rng.uniform(margin, h_img - margin)

        x1 = x0 + length * np.cos(angle_rad)
        y1 = y0 + length * np.sin(angle_rad)

        if not (0 <= x1 < w_img and 0 <= y1 < h_img):
            continue

        # Rasterise
        n_samples = max(int(length * 2), 2)
        xs = np.linspace(x0, x1, n_samples)
        ys = np.linspace(y0, y1, n_samples)
        for xf, yf in zip(xs, ys):
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    px, py = int(round(xf)) + dx, int(round(yf)) + dy
                    if 0 <= py < h_img and 0 <= px < w_img:
                        dist = np.hypot(dx, dy)
                        val = streak_brightness * np.exp(-0.5 * (dist / streak_width) ** 2)
                        image[py, px] += val

        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        return dict(
            cx=cx, cy=cy,
            w=length, h=_STREAK_WIDTH_PX,
            angle_deg=angle_deg,
            x_start=x0, y_start=y0,
            x_end=x1, y_end=y1,
            length_px=length,
        )

    # Fallback: horizontal streak in the centre
    x0, y0 = margin, h_img // 2
    length = (target_min_px + target_max_px) / 2.0
    x1 = x0 + length
    cx = (x0 + x1) / 2.0
    for xi in range(int(x0), int(x1) + 1):
        if 0 <= xi < w_img:
            image[h_img // 2, xi] = streak_brightness
    return dict(
        cx=cx, cy=float(h_img // 2),
        w=length, h=_STREAK_WIDTH_PX,
        angle_deg=0.0,
        x_start=x0, y_start=float(h_img // 2),
        x_end=x1, y_end=float(h_img // 2),
        length_px=length,
    )


def _make_one(
    output_dir: Path,
    filename: str,
    has_streak: bool,
    target_min_px: float = 0.0,
    target_max_px: float = 0.0,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
    obs_time: datetime | None = None,
    seed: int | None = None,
) -> dict | None:
    """Generate one synthetic FITS file and optional YOLO OBB label.

    Returns streak metadata dict or None (blank image).
    """
    if obs_time is None:
        obs_time = datetime(2024, 4, 2, 2, 0, 0, tzinfo=timezone.utc)

    rng = np.random.default_rng(seed)
    image = _make_background(rng, height, width)
    _add_stars(image, rng)

    streak_meta = None
    if has_streak:
        streak_meta = _add_streak_controlled(
            image, rng, target_min_px, target_max_px
        )

    image = np.clip(image, 0, 65535).astype(np.uint16)

    fits_path = output_dir / filename
    hdu = fits.PrimaryHDU(image)
    hdr = hdu.header
    hdr["DATE-OBS"] = obs_time.strftime("%Y-%m-%dT%H:%M:%S.000")
    hdr["EXPTIME"]  = 10.0
    hdr["NAXIS1"]   = width
    hdr["NAXIS2"]   = height
    hdr["PIXSCALE"] = 1.36
    hdr["SITELAT"]  = 49.61
    hdr["SITELONG"] = 6.13
    hdr["SITEELEV"] = 280.0
    _make_wcs_header(hdr, width, height)
    hdu.writeto(str(fits_path), overwrite=True)

    # Write YOLO OBB label file
    label_path = output_dir / (Path(filename).stem + ".txt")
    if has_streak and streak_meta is not None:
        cx_n = streak_meta["cx"] / width
        cy_n = streak_meta["cy"] / height
        w_n  = streak_meta["w"]  / width
        h_n  = streak_meta["h"]  / height
        with open(label_path, "w") as f:
            f.write(f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f} {streak_meta['angle_deg']:.4f}\n")
    else:
        label_path.write_text("")  # empty label for blank image

    logger.debug("Wrote %s (streak=%s)", fits_path.name, has_streak)
    return streak_meta


# ---------------------------------------------------------------------------
# OBB → COCO conversion (inlined to avoid requiring FITS re-read)
# ---------------------------------------------------------------------------

def _streak_to_coco_ann(
    ann_id: int,
    img_id: int,
    meta: dict,
    img_w: int,
    img_h: int,
) -> dict:
    """Convert streak metadata dict → COCO annotation."""
    import numpy as np
    cx, cy, w, h, angle_deg = (
        meta["cx"], meta["cy"], meta["w"], meta["h"], meta["angle_deg"]
    )
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    hw, hh = w / 2.0, h / 2.0
    corners_local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    xs = [cx + dx * cos_a - dy * sin_a for dx, dy in corners_local]
    ys = [cy + dx * sin_a + dy * cos_a for dx, dy in corners_local]
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    bw, bh = x2 - x1, y2 - y1

    return {
        "id": ann_id,
        "image_id": img_id,
        "category_id": 0,
        "bbox": [x1, y1, bw, bh],
        "area": float(bw * bh),
        "obb": [cx, cy, w, h, angle_deg],
        "iscrowd": 0,
    }


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_dev_subset(
    fits_dir: Path,
    annotations_path: Path,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
    seed: int = 42,
) -> None:
    """Generate all 50 images and write data/annotations/dev_subset.json.

    Args:
        fits_dir: Directory to write FITS and label .txt files.
        annotations_path: Output COCO JSON path.
        width: Image width in pixels.
        height: Image height in pixels.
        seed: Base RNG seed.
    """
    fits_dir.mkdir(parents=True, exist_ok=True)
    annotations_path.parent.mkdir(parents=True, exist_ok=True)

    base_time = datetime(2024, 4, 2, 2, 0, 0, tzinfo=timezone.utc)

    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    categories = [{"id": 0, "name": "streak"}]

    img_id = 1
    ann_id = 1
    stats = {"blank": 0, "short": 0, "long": 0}

    # Scale thresholds to image size so small images work too
    img_short_axis = min(width, height)
    short_max = min(_SHORT_MAX_PX, img_short_axis * 0.50)
    long_min  = min(_LONG_MIN_PX,  img_short_axis * 0.55)
    long_max  = img_short_axis * 0.85

    # --- Blank images ---
    for i in range(_N_BLANK):
        fname = f"dev_blank_{i:03d}.fits"
        obs_time = base_time + timedelta(minutes=i * 5)
        _make_one(fits_dir, fname, has_streak=False,
                  width=width, height=height, obs_time=obs_time, seed=seed + i)
        # MMDetection resolves COCO file_name relative to data_root.  The
        # default annotations live in data/annotations, so make file_name
        # relative to data/ rather than data/annotations.
        data_root = annotations_path.parent.parent
        rel = Path(os.path.relpath(fits_dir, data_root))
        coco_images.append({"id": img_id, "file_name": str(rel / fname),
                             "width": width, "height": height})
        img_id += 1
        stats["blank"] += 1

    # --- Short streaks ---
    for i in range(_N_SHORT):
        fname = f"dev_short_{i:03d}.fits"
        obs_time = base_time + timedelta(minutes=(_N_BLANK + i) * 5)
        meta = _make_one(
            fits_dir, fname, has_streak=True,
            target_min_px=40.0, target_max_px=short_max,
            width=width, height=height, obs_time=obs_time,
            seed=seed + _N_BLANK + i,
        )
        coco_images.append({"id": img_id, "file_name": str(rel / fname),
                             "width": width, "height": height})
        if meta is not None:
            coco_annotations.append(
                _streak_to_coco_ann(ann_id, img_id, meta, width, height)
            )
            ann_id += 1
        img_id += 1
        stats["short"] += 1

    # --- Long streaks ---
    for i in range(_N_LONG):
        fname = f"dev_long_{i:03d}.fits"
        obs_time = base_time + timedelta(minutes=(_N_BLANK + _N_SHORT + i) * 5)
        meta = _make_one(
            fits_dir, fname, has_streak=True,
            target_min_px=long_min, target_max_px=long_max,
            width=width, height=height, obs_time=obs_time,
            seed=seed + _N_BLANK + _N_SHORT + i,
        )
        coco_images.append({"id": img_id, "file_name": str(rel / fname),
                             "width": width, "height": height})
        if meta is not None:
            coco_annotations.append(
                _streak_to_coco_ann(ann_id, img_id, meta, width, height)
            )
            ann_id += 1
        img_id += 1
        stats["long"] += 1

    coco = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }
    annotations_path.write_text(json.dumps(coco, indent=2))

    print(f"Dev subset built:")
    print(f"  Images  : {len(coco_images)}  ({stats['blank']} blank, "
          f"{stats['short']} short, {stats['long']} long)")
    print(f"  Streaks : {len(coco_annotations)}")
    print(f"  FITS    : {fits_dir}/")
    print(f"  COCO    : {annotations_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="data/dev_subset",
        help="Directory for generated FITS files (default: data/dev_subset)",
    )
    parser.add_argument(
        "--annotations",
        default="data/annotations/dev_subset.json",
        help="Output COCO JSON path (default: data/annotations/dev_subset.json)",
    )
    parser.add_argument(
        "--small", action="store_true",
        help="Use 256×256 images (faster, for CI)"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    w = 256 if args.small else _DEFAULT_WIDTH
    h = 256 if args.small else _DEFAULT_HEIGHT

    build_dev_subset(
        fits_dir=Path(args.output_dir),
        annotations_path=Path(args.annotations),
        width=w,
        height=h,
        seed=args.seed,
    )
