"""ASTRiDE-based classical streak detector for the ARGUS pipeline.

Accepts a FITSImage, runs sep background subtraction, then ASTRiDE contour
detection, and returns a list of StreakDetection dataclasses ready for
downstream astrometry and matching.

# Source: Kim et al. — ASTRiDE streak detection algorithm (contour + shape filter)
# Ref: https://github.com/dwkim78/ASTRiDE
"""

from __future__ import annotations

import logging
import math
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import sep
from astropy.io import fits
from astride.detect import Streak

from src.ingest.fits_parser import FITSImage, parse_fits

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class StreakDetection:
    """One detected streak in pixel coordinates.

    Sky-coordinate fields (ra_*, dec_*, angular_velocity_arcsec_s,
    position_angle_deg) are None until populated by the plate solver.

    Attributes:
        x_start: Streak start column (pixels).
        y_start: Streak start row (pixels).
        x_end: Streak end column (pixels).
        y_end: Streak end row (pixels).
        x_center: Midpoint column (pixels).
        y_center: Midpoint row (pixels).
        angle_deg: Image-plane angle of the streak vector, degrees,
            measured counter-clockwise from the positive x-axis (east).
            Range: (-180, 180].
        length_px: Euclidean length start→end in pixels.
        width_px: Estimated cross-streak width (area / length).
        shape_factor: ASTRiDE elongation metric (lower → more streak-like).
        area_px: Contour area in pixels.
        ra_start: Sky RA of start point (degrees). None before plate solve.
        dec_start: Sky Dec of start point (degrees). None before plate solve.
        ra_end: Sky RA of end point (degrees). None before plate solve.
        dec_end: Sky Dec of end point (degrees). None before plate solve.
        ra_center: Sky RA of midpoint (degrees). None before plate solve.
        dec_center: Sky Dec of midpoint (degrees). None before plate solve.
        angular_velocity_arcsec_s: Angular speed (arcsec/s). None before solve.
        position_angle_deg: Celestial position angle (degrees). None before solve.
    """

    x_start: float
    y_start: float
    x_end: float
    y_end: float
    x_center: float
    y_center: float
    angle_deg: float
    length_px: float
    width_px: float
    shape_factor: float
    area_px: float
    ra_start: float | None = None
    dec_start: float | None = None
    ra_end: float | None = None
    dec_end: float | None = None
    ra_center: float | None = None
    dec_center: float | None = None
    angular_velocity_arcsec_s: float | None = None
    position_angle_deg: float | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _preprocess(data: np.ndarray) -> np.ndarray:
    """Subtract background, clip extremes, and scale to uint16.

    Steps:
    1. sep median background subtraction.
    2. Clip negatives to 0; clip top at 99.9th percentile.
    3. Scale to [0, 65535] and cast to uint16.

    Args:
        data: Raw float32 image array (height × width).

    Returns:
        Preprocessed uint16 array, same shape.
    """
    # sep requires a C-contiguous array; keep float32 to avoid an
    # unnecessary upcast that increases memory use and slows processing.
    work = np.ascontiguousarray(data, dtype=np.float32)
    bkg = sep.Background(work)
    work -= bkg.back()

    np.clip(work, 0, None, out=work)
    p999 = np.percentile(work, 99.9)
    if p999 > 0:
        np.clip(work, 0, p999, out=work)
        work = work / p999 * 65535.0

    return work.astype(np.uint16)


def _endpoints_from_contour(
    x_contour: np.ndarray,
    y_contour: np.ndarray,
    slope_angle_deg: float,
) -> tuple[float, float, float, float]:
    """Find streak endpoints by projecting contour onto the streak axis.

    # Source: Kim et al. — ASTRiDE slope_angle convention
    # Ref: https://github.com/dwkim78/ASTRiDE

    Projects every contour point onto the unit direction vector defined by
    slope_angle_deg, then takes the min- and max-projection points as the
    two endpoints.

    Args:
        x_contour: Column coordinates of contour boundary.
        y_contour: Row coordinates of contour boundary.
        slope_angle_deg: ASTRiDE slope_angle (degrees from x-axis).

    Returns:
        (x_start, y_start, x_end, y_end) — the two extreme contour points.
    """
    angle_rad = math.radians(slope_angle_deg)
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)
    proj = x_contour * dx + y_contour * dy
    i0 = int(np.argmin(proj))
    i1 = int(np.argmax(proj))
    return (
        float(x_contour[i0]), float(y_contour[i0]),
        float(x_contour[i1]), float(y_contour[i1]),
    )


def _edge_to_detection(
    edge: dict,
    min_length_px: float,
) -> StreakDetection | None:
    """Convert one ASTRiDE edge dict to a StreakDetection.

    Returns None if the detection is shorter than min_length_px.

    Args:
        edge: ASTRiDE edge dictionary from Streak.streaks.
        min_length_px: Minimum streak length to keep.

    Returns:
        StreakDetection or None.
    """
    x_arr = np.asarray(edge["x"], dtype=np.float64)
    y_arr = np.asarray(edge["y"], dtype=np.float64)
    slope_angle = float(edge.get("slope_angle", 0.0))

    x0, y0, x1, y1 = _endpoints_from_contour(x_arr, y_arr, slope_angle)

    length_px = math.hypot(x1 - x0, y1 - y0)
    if length_px < min_length_px:
        return None

    angle_deg = math.degrees(math.atan2(y1 - y0, x1 - x0))
    x_center = (x0 + x1) / 2.0
    y_center = (y0 + y1) / 2.0
    area_px = float(edge.get("area", 0.0))
    width_px = area_px / length_px if length_px > 0 else 0.0

    return StreakDetection(
        x_start=x0,
        y_start=y0,
        x_end=x1,
        y_end=y1,
        x_center=x_center,
        y_center=y_center,
        angle_deg=angle_deg,
        length_px=length_px,
        width_px=width_px,
        shape_factor=float(edge.get("shape_factor", 0.0)),
        area_px=area_px,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_streaks(
    image: FITSImage,
    contour_threshold: float = 3.0,
    min_length_px: float = 20.0,
) -> list[StreakDetection]:
    """Detect satellite streaks in a FITS image using ASTRiDE.

    Preprocessing: sep background subtraction → 99.9th-percentile clip →
    uint16 scaling. ASTRiDE then runs its own constant-background removal
    and contour-based streak filter.

    # Source: Kim et al. — ASTRiDE pipeline integration
    # Ref: https://github.com/dwkim78/ASTRiDE

    Args:
        image: Parsed FITSImage from fits_parser.parse_fits().
        contour_threshold: ASTRiDE sigma threshold for contour search.
            Lower values find fainter streaks but increase false positives.
            Default 3.0 (ASTRiDE upstream default).
        min_length_px: Minimum streak length in pixels. Detections shorter
            than this are discarded. Default 20.

    Returns:
        List of StreakDetection objects. Empty list if none found.
    """
    preprocessed = _preprocess(image.data)

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        hdu = fits.PrimaryHDU(preprocessed)
        hdu.writeto(tmp_path, overwrite=True)

        streak = Streak(
            tmp_path,
            remove_bkg="constant",
            contour_threshold=contour_threshold,
        )
        streak.detect()
        raw_edges = streak.streaks or []
    except Exception:
        logger.exception("ASTRiDE failed on %s", image.filepath.name)
        return []
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass

    results: list[StreakDetection] = []
    for edge in raw_edges:
        det = _edge_to_detection(edge, min_length_px)
        if det is not None:
            results.append(det)

    logger.info(
        "%s: %d streak(s) detected (threshold=%.1f, min_len=%g px)",
        image.filepath.name,
        len(results),
        contour_threshold,
        min_length_px,
    )
    return results


# ---------------------------------------------------------------------------
# Visualisation helper (used by __main__ and tests)
# ---------------------------------------------------------------------------

def annotate_image(
    image: FITSImage,
    detections: list[StreakDetection],
    output_path: Path,
) -> None:
    """Save a PNG of the image with detected streaks annotated.

    Converts the float32 image to 8-bit (log-stretch) and draws a line and
    label for each detection.

    Args:
        image: Source FITSImage.
        detections: Detections to annotate.
        output_path: Where to write the PNG.
    """
    data = image.data.astype(np.float32)
    data = np.clip(data, 0, None)
    # Log stretch for visibility
    data = np.log1p(data)
    vmax = np.percentile(data, 99.9) or 1.0
    display = (data / vmax * 255).clip(0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

    for i, d in enumerate(detections, start=1):
        pt0 = (int(round(d.x_start)), int(round(d.y_start)))
        pt1 = (int(round(d.x_end)), int(round(d.y_end)))
        cv2.line(rgb, pt0, pt1, (0, 0, 255), 2)
        label = f"#{i} {d.length_px:.0f}px"
        cv2.putText(
            rgb, label,
            (int(round(d.x_center)), int(round(d.y_center))),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), rgb)
    logger.info("Annotated image saved: %s", output_path)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) != 2:
        print("Usage: python classical_detector.py <path/to/image.fits>")
        sys.exit(1)

    fits_path = Path(sys.argv[1])
    img = parse_fits(fits_path)

    t0 = time.perf_counter()
    dets = detect_streaks(img)
    elapsed = time.perf_counter() - t0

    print(f"\n=== {fits_path.name} — {len(dets)} streak(s) in {elapsed:.2f}s ===")
    for i, d in enumerate(dets, 1):
        print(
            f"  #{i:2d}  center=({d.x_center:.1f}, {d.y_center:.1f})  "
            f"length={d.length_px:.1f}px  angle={d.angle_deg:.1f}°  "
            f"shape={d.shape_factor:.3f}"
        )

    if dets:
        out_png = fits_path.with_suffix(".annotated.png")
        annotate_image(img, dets, out_png)
        print(f"\nAnnotated image: {out_png}")
