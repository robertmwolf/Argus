"""Extract per-annotation streak geometry and SNR features from Atwood FITS.

Reads all sessions matching ``--scope`` from the session manifest, loads each
annotation JSON, and computes per-annotation features:

  - Geometry (length, aspect ratio, angle) — from OBB stored in annotations
  - SNR — from annotation attributes when available (Night 1), computed from
    FITS pixel data otherwise (Nights 2+)

Output is a CSV with one row per annotation.  This CSV is the primary input
to ``build_stratified_splits.py``.

Band thresholds (pixels in ORIGINAL IMAGE coordinate space):
  Short   : length < 269 px   (Atwood: < 342 arcsec)
  Medium  : 269 – 800 px      (Atwood: 342 – 1016 arcsec)
  Long    : length ≥ 800 px   (Atwood: > 1016 arcsec)

SNR classes:
  bright  : SNR > 20
  medium  : 5 < SNR ≤ 20
  faint   : SNR ≤ 5
  null    : SNR unavailable

Aspect ratio classes:
  thin    : OBB w/h > 20  (clean, well-focused streak)
  normal  : 5 – 20        (typical LEO streak)
  chunky  : < 5           (wide, defocused, or very short)

Usage
-----
# Full run — computes FITS-based SNR for sessions without attr SNR:
python scripts/extract_streak_features.py \\
    --output data/features/atwood_streak_features.csv

# Skip FITS loading (SNR=null for Night 2 / Geo):
python scripts/extract_streak_features.py --no-fits

# Parallel FITS loading (faster on slow drives):
python scripts/extract_streak_features.py --workers 4
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML required: pip install pyyaml")

try:
    from tqdm import tqdm as _tqdm

    def _progress(iterable, **kwargs):
        return _tqdm(iterable, **kwargs)

except ImportError:
    def _progress(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        total = kwargs.get("total", None)
        if desc:
            logger.info("%s (%s items)...", desc, total if total else "?")
        return iterable

try:
    from astropy.io import fits as _astropy_fits
    _ASTROPY_OK = True
except ImportError:
    _ASTROPY_OK = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST_PATH = _REPO_ROOT / "data/sessions/manifest.yaml"
_DEFAULT_OUTPUT = _REPO_ROOT / "data/features/atwood_streak_features.csv"

# Band thresholds — pixels in ORIGINAL IMAGE coordinate space (not arcseconds)
SHORT_MAX = 269.0
LONG_MIN = 800.0

# SNR classification thresholds
_SNR_BRIGHT = 20.0
_SNR_FAINT = 5.0

# Aspect ratio classification thresholds
_ASPECT_THIN = 20.0
_ASPECT_CHUNKY = 5.0

_CSV_FIELDS = [
    "annotation_id",
    "image_id",
    "session_id",
    "scope_id",
    "file_name",
    "fits_path",
    "img_width",
    "img_height",
    "length_px",
    "aspect_ratio",
    "angle_deg",
    "snr",
    "band",
    "snr_class",
    "aspect_class",
]


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def _band(length_px: float) -> str:
    if length_px < SHORT_MAX:
        return "short"
    if length_px < LONG_MIN:
        return "medium"
    return "long"


def _snr_class(snr: float | None) -> str:
    if snr is None:
        return "null"
    if snr > _SNR_BRIGHT:
        return "bright"
    if snr > _SNR_FAINT:
        return "medium"
    return "faint"


def _aspect_class(aspect_ratio: float) -> str:
    if aspect_ratio > _ASPECT_THIN:
        return "thin"
    if aspect_ratio >= _ASPECT_CHUNKY:
        return "normal"
    return "chunky"


# ---------------------------------------------------------------------------
# FITS-based SNR computation
# ---------------------------------------------------------------------------

def compute_snr_from_fits(
    fits_path: Path,
    bbox: list[float],
) -> float | None:
    """Compute streak SNR from raw FITS pixel data.

    SNR = (mean_flux_in_bbox - background_median) / background_sigma

    Background is sampled from an annulus surrounding the streak bounding
    box (expanded by 50% on each side).  Noise is estimated as the robust
    sigma (MAD × 1.4826) of the annulus pixels.

    Args:
        fits_path: Absolute path to the FITS file.
        bbox: COCO-format ``[x, y, w, h]`` in native pixel coordinates.

    Returns:
        SNR as float, or None if the file is unreadable or region invalid.
    """
    if not _ASTROPY_OK:
        logger.warning("astropy not available; cannot compute FITS SNR")
        return None

    x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

    # Expand to include background annulus in a single sub-image read
    pad = max(int(max(w, h) * 0.5), 30)

    try:
        with _astropy_fits.open(str(fits_path), memmap=False) as hdul:
            # Find the first 2-D image HDU
            img_hdu = None
            for hdu in hdul:
                if hasattr(hdu, "shape") and len(hdu.shape) == 2:
                    img_hdu = hdu
                    break
            if img_hdu is None:
                return None

            H, W = img_hdu.shape

            # Clip combined region to image bounds
            bx1 = max(0, int(math.floor(x)) - pad)
            by1 = max(0, int(math.floor(y)) - pad)
            bx2 = min(W, int(math.ceil(x + w)) + pad)
            by2 = min(H, int(math.ceil(y + h)) + pad)

            if bx2 - bx1 < 5 or by2 - by1 < 5:
                return None

            # Partial read via astropy Section — avoids loading the full FITS
            # frame (52 MB per Night 1 file).  ~28× faster than memmap=False
            # full load when only a small bbox+annulus region is needed.
            subarray = np.array(img_hdu.section[by1:by2, bx1:bx2], dtype=np.float32)

    except Exception as exc:
        logger.debug("FITS section read failed %s: %s", fits_path.name, exc)
        return None

    # Inner bbox coordinates relative to the loaded subarray
    ix1 = int(math.floor(x)) - bx1
    iy1 = int(math.floor(y)) - by1
    ix2 = ix1 + int(math.ceil(w))
    iy2 = iy1 + int(math.ceil(h))
    ix1 = max(0, ix1)
    iy1 = max(0, iy1)
    ix2 = min(subarray.shape[1], ix2)
    iy2 = min(subarray.shape[0], iy2)

    if ix2 <= ix1 or iy2 <= iy1:
        return None

    # 99th-percentile pixel as the signal estimator.
    # The axis-aligned bbox encloses mostly background for angled streaks
    # (45° streak fills ~14% of its bbox).  The mean is dominated by background;
    # p99 reliably captures the streak peak while being robust against single
    # hot pixels.  This closely matches SkyTrack's peak_snr annotation metric.
    streak_signal = float(np.percentile(subarray[iy1:iy2, ix1:ix2], 99))

    # Background annulus: mask the inner bbox and compute robust noise estimate
    annulus = subarray.copy()
    annulus[iy1:iy2, ix1:ix2] = np.nan
    bg_pixels = annulus[~np.isnan(annulus)]

    if len(bg_pixels) < 20:
        return None

    bg_median = float(np.median(bg_pixels))
    mad = float(np.median(np.abs(bg_pixels - bg_median)))
    bg_sigma = mad * 1.4826

    if bg_sigma < 1.0:
        bg_sigma = max(float(np.std(bg_pixels)), 1.0)

    return (streak_signal - bg_median) / bg_sigma


# ---------------------------------------------------------------------------
# Manifest and annotation loading
# ---------------------------------------------------------------------------

def _load_manifest(path: Path) -> list[dict]:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return data["sources"]


def _fits_path_for_image(
    file_name: str,
    raw_dir: str | None,
) -> Path | None:
    """Resolve a FITS image path from a file_name + optional raw_dir."""
    p = Path(file_name)
    if p.is_absolute():
        return p
    if raw_dir:
        return Path(raw_dir) / p.name
    return None


# ---------------------------------------------------------------------------
# Per-annotation feature extraction
# ---------------------------------------------------------------------------

def _extract_one(
    ann: dict,
    image: dict,
    session_id: str,
    scope_id: str,
    raw_dir: str | None,
    use_fits: bool,
) -> dict:
    """Extract feature row for a single annotation."""
    obb = ann.get("obb")
    bbox = ann.get("bbox", [0.0, 0.0, 0.0, 0.0])

    if obb and obb.get("w", 0) > 0:
        length_px = float(obb["w"])
        obb_h = float(obb.get("h", 1.0))
        aspect_ratio = length_px / max(obb_h, 1e-6)
        angle_deg = float(obb.get("angle_deg", 0.0)) % 180.0
    else:
        # Fallback: geometry from axis-aligned bbox
        bw, bh = float(bbox[2]), float(bbox[3])
        length_px = math.hypot(bw, bh)
        aspect_ratio = max(bw, bh) / max(min(bw, bh), 1e-6)
        angle_deg = math.degrees(math.atan2(bh, bw)) % 180.0

    # SNR — always compute from FITS for consistency across all sessions.
    # Night 1's annotation attributes carry SkyTrack "peak_snr" (peak pixel /
    # noise), which is on a very different scale from our computed mean-SNR.
    # Using a single method (FITS-derived mean SNR) ensures that band × snr_class
    # stratification cells are comparable across nights.
    #
    # Fallback order:
    #   1. Compute from FITS if file is accessible (preferred — consistent method)
    #   2. Use annotation attribute peak_snr if > 0  (Night 1 fallback)
    #   3. None (SNR unknown)
    attrs = ann.get("attributes", {})
    attr_snr = float(attrs.get("peak_snr", 0.0))
    snr: float | None = None

    fits_path: Path | None = _fits_path_for_image(image["file_name"], raw_dir)

    if use_fits and fits_path is not None and fits_path.exists():
        snr = compute_snr_from_fits(fits_path, bbox)
    elif use_fits and fits_path:
        logger.debug("FITS not found: %s", fits_path)

    # Fallback to attribute SNR if FITS computation failed or was skipped
    if snr is None and attr_snr > 0:
        snr = attr_snr

    return {
        "annotation_id": ann["id"],
        "image_id": ann["image_id"],
        "session_id": session_id,
        "scope_id": scope_id,
        "file_name": image["file_name"],
        "fits_path": str(fits_path) if fits_path else "",
        "img_width": image.get("width", 0),
        "img_height": image.get("height", 0),
        "length_px": round(length_px, 2),
        "aspect_ratio": round(aspect_ratio, 3),
        "angle_deg": round(angle_deg, 2),
        "snr": round(snr, 2) if snr is not None else "",
        "band": _band(length_px),
        "snr_class": _snr_class(snr),
        "aspect_class": _aspect_class(aspect_ratio),
    }


def _process_session(
    entry: dict,
    use_fits: bool,
    workers: int,
) -> list[dict]:
    """Load one manifest session and return feature rows for all annotations."""
    session_id = entry["session_id"]
    scope_id = entry.get("scope_id", session_id)
    raw_dir = entry.get("raw_dir")

    ann_path = _REPO_ROOT / entry["annotation_file"]
    if not ann_path.exists():
        logger.warning("Annotation file missing, skipping: %s", ann_path)
        return []

    with open(ann_path) as fh:
        coco = json.load(fh)

    images_by_id: dict[int, dict] = {img["id"]: img for img in coco["images"]}
    annotations = coco.get("annotations", [])

    logger.info(
        "Session %-25s  %d annotations  raw_dir=%s",
        session_id,
        len(annotations),
        raw_dir or "(none)",
    )

    rows: list[dict] = []

    if workers > 1 and use_fits:
        # Parallel FITS loading
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for ann in annotations:
                img = images_by_id.get(ann["image_id"])
                if img is None:
                    continue
                fut = pool.submit(
                    _extract_one,
                    ann, img, session_id, scope_id, raw_dir, use_fits,
                )
                futures[fut] = ann["id"]

            for fut in _progress(
                as_completed(futures),
                total=len(futures),
                desc=f"  {session_id}",
            ):
                try:
                    rows.append(fut.result())
                except Exception as exc:
                    logger.error("Failed ann_id=%s: %s", futures[fut], exc)
    else:
        for ann in _progress(
            annotations,
            total=len(annotations),
            desc=f"  {session_id}",
        ):
            img = images_by_id.get(ann["image_id"])
            if img is None:
                continue
            rows.append(
                _extract_one(ann, img, session_id, scope_id, raw_dir, use_fits)
            )

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract per-annotation streak features from Atwood FITS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=_MANIFEST_PATH,
        help="Session manifest YAML (default: data/sessions/manifest.yaml)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Output CSV path (default: data/features/atwood_streak_features.csv)",
    )
    p.add_argument(
        "--scope",
        default="atwood",
        help="scope_id to process; 'all' for every training scope (default: atwood)",
    )
    p.add_argument(
        "--no-fits",
        action="store_true",
        help="Skip FITS pixel loading; SNR will be null for sessions without "
             "attribute SNR (Night 2, Geo).  Much faster but incomplete.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for FITS I/O (default: 1)",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "holdout"],
        help="Manifest splits to include (default: train holdout).  "
             "Pass 'all' to include every split.",
    )
    p.add_argument(
        "--sessions",
        nargs="+",
        default=None,
        help="Restrict to specific session_id(s).  Default: all sessions "
             "matching --scope and --splits.",
    )
    p.add_argument(
        "--append",
        action="store_true",
        help="Append rows to an existing CSV instead of overwriting it.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.manifest) as fh:
        manifest_data = yaml.safe_load(fh)
    sources = manifest_data["sources"]

    # Filter by scope
    if args.scope != "all":
        sources = [s for s in sources if s.get("scope_id") == args.scope]
        if not sources:
            raise SystemExit(
                f"No sources found for scope_id={args.scope!r}. "
                "Check manifest or use --scope all."
            )

    # Filter by split
    if args.splits != ["all"]:
        split_set = set(args.splits)
        sources = [s for s in sources if s.get("split") in split_set]
    if not sources:
        raise SystemExit("No sources match the requested splits. Check --splits.")

    # Filter by explicit session list
    if args.sessions:
        session_set = set(args.sessions)
        sources = [s for s in sources if s.get("session_id") in session_set]
        if not sources:
            raise SystemExit(f"No sources found for sessions={args.sessions!r}.")

    logger.info(
        "Processing %d session(s) for scope=%s  use_fits=%s  workers=%d",
        len(sources),
        args.scope,
        not args.no_fits,
        args.workers,
    )

    all_rows: list[dict] = []
    for entry in sources:
        rows = _process_session(entry, use_fits=not args.no_fits, workers=args.workers)
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("No feature rows extracted — check annotation files and paths.")

    # Summary statistics
    bands = {"short": 0, "medium": 0, "long": 0}
    snr_classes = {"bright": 0, "medium": 0, "faint": 0, "null": 0}
    for row in all_rows:
        bands[row["band"]] = bands.get(row["band"], 0) + 1
        snr_classes[row["snr_class"]] = snr_classes.get(row["snr_class"], 0) + 1

    logger.info("Total annotations: %d", len(all_rows))
    logger.info(
        "  Bands:      short=%d  medium=%d  long=%d",
        bands.get("short", 0), bands.get("medium", 0), bands.get("long", 0),
    )
    logger.info(
        "  SNR class:  bright=%d  medium=%d  faint=%d  null=%d",
        snr_classes.get("bright", 0), snr_classes.get("medium", 0),
        snr_classes.get("faint", 0), snr_classes.get("null", 0),
    )

    # Write CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_mode = "a" if args.append and args.output.exists() else "w"
    write_header = write_mode == "w"
    with open(args.output, write_mode, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(all_rows)

    logger.info("Written: %s  (%d rows)", args.output, len(all_rows))


if __name__ == "__main__":
    main()
