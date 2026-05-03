"""Generate minimal synthetic FITS files for pipeline testing.

Produces realistic-looking FITS images with Poisson noise, Gaussian
star PSFs, and optional synthetic satellite streaks.  No real telescope
data required — useful for smoke-testing the full pipeline on a Mac
before the MILAN dataset is downloaded.

Generated files are written to data/sample/ by default.

Usage::

    python scripts/make_test_fits.py                   # default: 5 with streak, 5 without
    python scripts/make_test_fits.py --n-streak 10 --n-blank 5
    python scripts/make_test_fits.py --output-dir /tmp/test_fits
    python scripts/make_test_fits.py --small           # 512×512 (fast)
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)

# Stellina telescope defaults (MILAN survey parameters)
_DEFAULT_WIDTH  = 3096
_DEFAULT_HEIGHT = 2080
_SMALL_WIDTH    = 512
_SMALL_HEIGHT   = 512
_PIXSCALE_ARCSEC = 1.36   # arcsec/pixel


# ---------------------------------------------------------------------------
# Core generators
# ---------------------------------------------------------------------------

def _make_background(
    rng: np.random.Generator,
    height: int,
    width: int,
    sky_level: float = 100.0,
) -> np.ndarray:
    """Poisson sky background with read-noise (Gaussian)."""
    background = rng.poisson(sky_level, size=(height, width)).astype(np.float32)
    read_noise = rng.normal(0, 5.0, size=(height, width)).astype(np.float32)
    return background + read_noise


def _add_stars(
    image: np.ndarray,
    rng: np.random.Generator,
    n_stars: int = 200,
    psf_sigma: float = 1.5,
) -> None:
    """Add Gaussian PSF stars in-place."""
    h, w = image.shape
    margin = 10
    for _ in range(n_stars):
        cx = rng.integers(margin, w - margin)
        cy = rng.integers(margin, h - margin)
        brightness = rng.integers(500, 8000)
        radius = max(3, int(psf_sigma * 3))
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                py, px = cy + dy, cx + dx
                if 0 <= py < h and 0 <= px < w:
                    r2 = (dx / psf_sigma) ** 2 + (dy / psf_sigma) ** 2
                    image[py, px] += brightness * np.exp(-0.5 * r2)


def _add_streak(
    image: np.ndarray,
    rng: np.random.Generator,
    streak_brightness: float = 6000.0,
    streak_width: float = 1.5,
) -> dict:
    """Add a linear satellite streak in-place.

    Returns:
        Dict with keys: x_start, y_start, x_end, y_end, angle_deg,
        length_px.  Useful for building ground-truth annotations.
    """
    h, w = image.shape
    margin = max(5, min(50, w // 8, h // 8))   # proportional, safe for small images

    # Random start point near an edge, angle ∈ [10°, 170°]
    angle_deg = rng.uniform(10, 170)
    angle_rad = np.radians(angle_deg)

    # Start from left/right/top/bottom margin randomly
    side = rng.integers(0, 4)
    if side == 0:   # left
        x0, y0 = rng.integers(0, margin), rng.integers(margin, h - margin)
    elif side == 1: # right
        x0, y0 = rng.integers(w - margin, w), rng.integers(margin, h - margin)
    elif side == 2: # top
        x0, y0 = rng.integers(margin, w - margin), rng.integers(0, margin)
    else:           # bottom
        x0, y0 = rng.integers(margin, w - margin), rng.integers(h - margin, h)

    # Streak length: 10–60 % of image diagonal
    diag = np.hypot(w, h)
    length = rng.uniform(0.10 * diag, 0.60 * diag)
    x1 = x0 + length * np.cos(angle_rad)
    y1 = y0 + length * np.sin(angle_rad)

    # Rasterise the streak with anti-aliasing width
    n_samples = int(length * 2)
    xs = np.linspace(x0, x1, n_samples)
    ys = np.linspace(y0, y1, n_samples)

    for xf, yf in zip(xs, ys):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                px, py = int(round(xf)) + dx, int(round(yf)) + dy
                if 0 <= py < h and 0 <= px < w:
                    dist = np.hypot(dx, dy)
                    val = streak_brightness * np.exp(
                        -0.5 * (dist / streak_width) ** 2
                    )
                    image[py, px] += val

    return dict(
        x_start=float(x0),
        y_start=float(y0),
        x_end=float(min(x1, w - 1)),
        y_end=float(min(y1, h - 1)),
        angle_deg=float(angle_deg),
        length_px=float(length),
    )


def _make_wcs_header(
    header: fits.Header,
    width: int,
    height: int,
    ra_center: float = 83.82,
    dec_center: float = -5.39,
    pixscale_deg: float = _PIXSCALE_ARCSEC / 3600.0,
) -> None:
    """Populate a minimal TAN-projection WCS into *header* in-place."""
    header["CTYPE1"]  = "RA---TAN"
    header["CTYPE2"]  = "DEC--TAN"
    header["CRVAL1"]  = ra_center
    header["CRVAL2"]  = dec_center
    header["CRPIX1"]  = width  / 2.0
    header["CRPIX2"]  = height / 2.0
    header["CDELT1"]  = -pixscale_deg   # RA increases to the left
    header["CDELT2"]  =  pixscale_deg
    header["CD1_1"]   = -pixscale_deg
    header["CD1_2"]   = 0.0
    header["CD2_1"]   = 0.0
    header["CD2_2"]   =  pixscale_deg


def make_test_fits(
    output_path: str | Path,
    with_streak: bool = True,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
    obs_time: datetime | None = None,
    seed: int | None = None,
) -> dict | None:
    """Generate a single synthetic FITS file.

    Args:
        output_path: Destination ``.fits`` file path.
        with_streak: If True, inject one satellite streak.
        width: Image width in pixels.
        height: Image height in pixels.
        obs_time: UTC observation time.  Defaults to 2024-04-02 02:55:24 UTC.
        seed: RNG seed for reproducibility.

    Returns:
        Streak metadata dict (x_start, y_start, …) if ``with_streak`` else None.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    if obs_time is None:
        obs_time = datetime(2024, 4, 2, 2, 55, 24, tzinfo=timezone.utc)

    # Build image array
    image = _make_background(rng, height, width)
    _add_stars(image, rng)

    streak_meta = None
    if with_streak:
        streak_meta = _add_streak(image, rng)

    # Clip to uint16 range and cast
    image = np.clip(image, 0, 65535).astype(np.uint16)

    # Build FITS header
    hdu = fits.PrimaryHDU(image)
    h = hdu.header
    h["DATE-OBS"] = obs_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    h["EXPTIME"]  = 10.0
    h["NAXIS1"]   = width
    h["NAXIS2"]   = height
    h["PIXSCALE"] = _PIXSCALE_ARCSEC
    h["SITELAT"]  = 49.61    # Luxembourg (MILAN survey site)
    h["SITELONG"] = 6.13
    h["SITEELEV"] = 280.0
    _make_wcs_header(h, width, height)

    hdu.writeto(str(output_path), overwrite=True)
    logger.info("Written: %s  (streak=%s)", output_path, with_streak)
    return streak_meta


def generate_test_set(
    output_dir: Path,
    n_streak: int = 5,
    n_blank: int = 5,
    small: bool = False,
    seed: int = 42,
) -> list[dict]:
    """Generate a numbered set of synthetic FITS files.

    Args:
        output_dir: Directory to write files into.
        n_streak: Number of images with a streak.
        n_blank: Number of images without a streak.
        small: If True, use 512×512 instead of full MILAN resolution.
        seed: Base RNG seed (each file uses seed + index).

    Returns:
        List of metadata dicts with keys: path, has_streak, streak_info.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    w = _SMALL_WIDTH  if small else _DEFAULT_WIDTH
    h = _SMALL_HEIGHT if small else _DEFAULT_HEIGHT

    base_time = datetime(2024, 4, 2, 2, 0, 0, tzinfo=timezone.utc)
    results = []

    for i in range(n_streak):
        obs_time = base_time + timedelta(minutes=i * 10)
        path = output_dir / f"synth_streak_{i:03d}.fits"
        streak_info = make_test_fits(
            path, with_streak=True, width=w, height=h,
            obs_time=obs_time, seed=seed + i,
        )
        results.append({"path": str(path), "has_streak": True,
                        "streak_info": streak_info})

    for i in range(n_blank):
        obs_time = base_time + timedelta(minutes=(n_streak + i) * 10)
        path = output_dir / f"synth_blank_{i:03d}.fits"
        make_test_fits(
            path, with_streak=False, width=w, height=h,
            obs_time=obs_time, seed=seed + n_streak + i,
        )
        results.append({"path": str(path), "has_streak": False,
                        "streak_info": None})

    return results


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/sample",
                        help="Directory to write FITS files (default: data/sample)")
    parser.add_argument("--n-streak", type=int, default=5,
                        help="Number of images with a streak (default: 5)")
    parser.add_argument("--n-blank",  type=int, default=5,
                        help="Number of images without a streak (default: 5)")
    parser.add_argument("--small", action="store_true",
                        help="Use 512×512 pixels instead of full 3096×2080")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed (default: 42)")
    args = parser.parse_args()

    results = generate_test_set(
        output_dir=Path(args.output_dir),
        n_streak=args.n_streak,
        n_blank=args.n_blank,
        small=args.small,
        seed=args.seed,
    )

    print(f"\nGenerated {len(results)} FITS files in {args.output_dir}/")
    for r in results:
        tag = "streak" if r["has_streak"] else "blank "
        info = ""
        if r["streak_info"]:
            s = r["streak_info"]
            info = (f"  angle={s['angle_deg']:.1f}°  "
                    f"length={s['length_px']:.0f}px")
        print(f"  [{tag}] {Path(r['path']).name}{info}")
