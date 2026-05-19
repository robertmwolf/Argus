"""Build a negative-example COCO corpus from the Frigate FITS dataset.

Frigate (DanSRoll/frigate, Nature Scientific Data 2025) is a 2,000-frame raw FITS
time series (QHY600M, 9600×6422 px, 0.5 s, single site). Auto-detection attempts
using threshold-based contour detection, z-score normalisation on raw FITS, and
frame-differencing all produce zero reliable streak annotations — either no
satellite transited during the observation window, or the streaks are below any
threshold that remains selective against background features.

This script therefore uses Frigate as a **negative-example corpus**: all FITS
frames are registered in the output COCO JSON with no annotations. The
YOLO tiling pipeline (train_compare_streakmind_yolo.py) will tile each FITS
frame and treat unannotated tiles as background training examples.

Value: GTImages provides only 93 no-streak frames from a single site. Frigate
adds ~1,980 frames from a different camera (QHY600M), different site, and
different sky conditions — improving background diversity and model precision.

Outputs (written to data/annotations/):
  frigate_negatives.json   — COCO format, file_name = absolute FITS path,
                             zero annotations.

Usage:
  # Full corpus (all 1,980 frames with matched processed PNGs):
  python scripts/annotate_frigate.py

  # Limit to N frames (smoke test):
  python scripts/annotate_frigate.py --max-frames 50

  # Only use raw FITS (no processed PNG check):
  python scripts/annotate_frigate.py --raw-only
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from astropy.io import fits

logger = logging.getLogger(__name__)

_CATEGORIES = [{"id": 1, "name": "satellite_streak", "supercategory": "streak"}]


def _fits_dimensions(fits_path: Path) -> tuple[int, int]:
    """Return (width, height) from FITS header without loading pixel data."""
    with fits.open(fits_path, memmap=True) as hdul:
        hdr = hdul[0].header
        return int(hdr["NAXIS1"]), int(hdr["NAXIS2"])


def build_negatives_coco(
    raw_dir: Path,
    processed_dir: Path | None,
    max_frames: int | None,
) -> dict:
    """Return a COCO dict with all Frigate FITS registered but no annotations.

    Args:
        raw_dir: Directory containing raw FITS files.
        processed_dir: If provided, only include FITS files that have a
            corresponding processed PNG (avoids the windowed-out first/last
            10 frames that lack processed counterparts).
        max_frames: Cap the number of images included (for smoke tests).

    Returns:
        COCO dict with images list and empty annotations list.
    """
    fits_files = sorted(raw_dir.glob("*.fits"))

    if processed_dir is not None:
        png_stems = {p.stem for p in processed_dir.glob("*.png")}
        fits_files = [f for f in fits_files if f.stem in png_stems]
        logger.info(
            "Filtered to %d FITS files with matching processed PNGs", len(fits_files)
        )

    if max_frames is not None:
        fits_files = fits_files[:max_frames]

    # Read dimensions from first file; assume all frames are the same size.
    if not fits_files:
        raise FileNotFoundError(f"No FITS files found in {raw_dir}")
    fits_w, fits_h = _fits_dimensions(fits_files[0])
    logger.info(
        "FITS dimensions: %d × %d — registering %d frames as negatives",
        fits_w, fits_h, len(fits_files),
    )

    images = [
        {
            "id": i + 1,
            "file_name": str(f.resolve()),
            "width": fits_w,
            "height": fits_h,
        }
        for i, f in enumerate(fits_files)
    ]

    return {
        "info": {
            "description": (
                "Frigate negative-example corpus — no streak annotations. "
                "All frames are background-only training examples. "
                "Auto-detection (threshold/contour, z-score norm, frame-diff) "
                "found no reliable streak signal in this observation sequence."
            ),
            "source": "github.com/DanSRoll/frigate",
            "reference": "Nature Scientific Data 2025 — https://doi.org/10.1038/s41597-025-06220-0",
        },
        "images": images,
        "annotations": [],
        "categories": _CATEGORIES,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("/Volumes/External/frigate/raw"),
        help="Directory containing Frigate raw FITS files.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("/Volumes/External/frigate/processed"),
        help="Directory containing Frigate processed PNGs (used to filter "
             "the first/last frames that lack processed counterparts). "
             "Pass --raw-only to skip this filter.",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Include all raw FITS files without cross-checking against "
             "processed PNGs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/annotations/frigate_negatives.json"),
        help="Output COCO JSON path.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit to this many frames (for smoke testing).",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.raw_dir.exists():
        raise SystemExit(f"Raw FITS dir not found: {args.raw_dir}")

    processed_dir: Path | None = None
    if not args.raw_only:
        if args.processed_dir.exists():
            processed_dir = args.processed_dir
        else:
            logger.warning(
                "Processed dir not found (%s); including all raw FITS without filter.",
                args.processed_dir,
            )

    coco = build_negatives_coco(args.raw_dir, processed_dir, args.max_frames)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(coco, indent=2))
    logger.info(
        "Wrote %d negative images, 0 annotations → %s",
        len(coco["images"]), args.output,
    )


if __name__ == "__main__":
    main()
