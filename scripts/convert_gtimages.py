"""Convert GTImages .strk annotation files to COCO JSON format.

GTImages is a purpose-built satellite streak dataset captured by SkyTrack 1.9.8.
Each .strk file covers one tracked NORAD ID and lists per-image observations with
pixel-precise start/end streak coordinates, SNR metrics, and embedded TLE elements.

This script produces two COCO JSON files:
  - Primary: usable labeled images (reject=0) with OBB annotations
  - Negatives: images where no streak was found (reject=-1), annotations empty

The OBB is constructed from start/end pixel coordinates as a tight rotated box
of fixed width (derived from the streak's elongation field, floored at 10 px).

Source: GTImages dataset, SkyTrack 1.9.8
"""

import argparse
import json
import logging
import math
import pathlib
from datetime import datetime
from typing import Any

from astropy.io import fits

logger = logging.getLogger(__name__)

# Fixed half-width of the streak OBB perpendicular to the streak axis (pixels).
# GTImages does not record a PSF width; 8 px is conservative for a 1.2"/px camera.
_STREAK_HALF_WIDTH_PX = 8.0


def _parse_strk_file(path: pathlib.Path) -> dict[str, Any]:
    """Parse a single .strk file and return structured data.

    Args:
        path: Path to the .strk annotation file.

    Returns:
        Dict with keys: norad_id (int), name (str), object_id (str),
        incl_deg (float), observations (list of dicts).
        Each observation dict has: filename, datetime_utc, jd_mid,
        x_start, y_start, x_end, y_end, peak_snr, mean_snr,
        elongation, length_px, reject, comment.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    norad_id: int | None = None
    name = ""
    object_id = ""
    incl_deg = 0.0
    observations: list[dict[str, Any]] = []

    in_obs_section = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("["):
            if stripped == "[OBS]":
                in_obs_section = False  # header row next
            continue

        # TLE summary row: first column is a bare integer NORAD ID
        parts = stripped.split("\t")
        if (
            not in_obs_section
            and len(parts) >= 16
            and parts[0].strip().isdigit()
            and not parts[0].strip().startswith("Image")
        ):
            norad_id = int(parts[0].strip())
            incl_deg = float(parts[3].strip())
            name = parts[15].strip()
            object_id = parts[16].strip() if len(parts) > 16 else ""
            continue

        # Column header row
        if stripped.startswith("Image\t"):
            in_obs_section = True
            continue

        if in_obs_section and stripped:
            if len(parts) < 14:
                continue
            try:
                obs: dict[str, Any] = {
                    "filename": parts[0].strip(),
                    "datetime_utc": parts[1].strip(),
                    "jd_mid": float(parts[2].strip()),
                    "x_start": float(parts[3].strip()),
                    "y_start": float(parts[4].strip()),
                    "x_end": float(parts[5].strip()),
                    "y_end": float(parts[6].strip()),
                    "x_mid": float(parts[7].strip()),
                    "y_mid": float(parts[8].strip()),
                    "peak_snr": float(parts[9].strip()),
                    "mean_snr": float(parts[10].strip()),
                    "elongation": float(parts[11].strip()),
                    "length_px": float(parts[12].strip()),
                    "reject": parts[13].strip(),
                    "comment": parts[-1].strip() if len(parts) > 25 else "",
                }
                observations.append(obs)
            except (ValueError, IndexError):
                logger.debug("Skipping malformed obs line in %s: %s", path.name, stripped[:80])

    return {
        "norad_id": norad_id,
        "name": name,
        "object_id": object_id,
        "incl_deg": incl_deg,
        "observations": observations,
    }


def _obs_to_coco_annotation(
    obs: dict[str, Any],
    image_id: int,
    annotation_id: int,
    norad_id: int,
    sat_name: str,
) -> dict[str, Any]:
    """Convert one usable observation to a COCO annotation dict.

    The OBB is represented as a rotated bounding box stored in COCO's
    ``segmentation`` field as a flat 8-element polygon [x1,y1,...,x4,y4]
    and in a custom ``obb`` field as {cx, cy, w, h, angle_deg}.

    Args:
        obs: Parsed observation dict from _parse_strk_file.
        image_id: COCO image id for this file.
        annotation_id: Unique annotation id.
        norad_id: NORAD catalog ID of the satellite.
        sat_name: Human-readable satellite name.

    Returns:
        COCO annotation dict.
    """
    x0, y0 = obs["x_start"], obs["y_start"]
    x1, y1 = obs["x_end"], obs["y_end"]
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)

    # Angle of the streak axis (degrees, measured from +X axis)
    angle_deg = math.degrees(math.atan2(dy, dx))

    # OBB centre
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    w = length
    h = _STREAK_HALF_WIDTH_PX * 2.0

    # Four corners of the rotated box
    cos_a = math.cos(math.radians(angle_deg))
    sin_a = math.sin(math.radians(angle_deg))
    hw, hh = w / 2.0, h / 2.0
    corners = [
        (cx + hw * cos_a - hh * sin_a, cy + hw * sin_a + hh * cos_a),
        (cx - hw * cos_a - hh * sin_a, cy - hw * sin_a + hh * cos_a),
        (cx - hw * cos_a + hh * sin_a, cy - hw * sin_a - hh * cos_a),
        (cx + hw * cos_a + hh * sin_a, cy + hw * sin_a - hh * cos_a),
    ]
    flat_poly = [coord for pt in corners for coord in pt]

    # Axis-aligned bounding box (COCO required field)
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    bbox_x = min(xs)
    bbox_y = min(ys)
    bbox_w = max(xs) - bbox_x
    bbox_h = max(ys) - bbox_y

    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "segmentation": [flat_poly],
        "bbox": [bbox_x, bbox_y, bbox_w, bbox_h],
        "area": w * h,
        "iscrowd": 0,
        "obb": {
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "angle_deg": angle_deg,
        },
        "attributes": {
            "norad_id": norad_id,
            "satellite_name": sat_name,
            "peak_snr": obs["peak_snr"],
            "mean_snr": obs["mean_snr"],
            "length_px": obs["length_px"],
            "elongation": obs["elongation"],
            "jd_mid": obs["jd_mid"],
            "datetime_utc": obs["datetime_utc"],
        },
    }


def _fits_dimensions(fits_path: pathlib.Path) -> tuple[int, int]:
    """Return (width, height) of a FITS primary HDU without loading pixel data.

    Args:
        fits_path: Path to the FITS file.

    Returns:
        Tuple of (width_px, height_px).
    """
    with fits.open(fits_path, memmap=True) as hdul:
        h = hdul[0].header
        return int(h["NAXIS1"]), int(h["NAXIS2"])


def convert(
    strk_dir: pathlib.Path,
    output_path: pathlib.Path,
    negatives_output_path: pathlib.Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert all .strk files in strk_dir to COCO JSON.

    Args:
        strk_dir: Directory containing .strk and .fits files.
        output_path: Destination for the labeled (reject=0) COCO JSON.
        negatives_output_path: Destination for the no-streak (reject=-1) COCO JSON.
            If None, negatives are not written.

    Returns:
        Tuple of (labeled_coco_dict, negatives_coco_dict).
    """
    category = [{"id": 1, "name": "satellite_streak", "supercategory": "streak"}]
    info = {
        "description": "GTImages satellite streak dataset — SkyTrack 1.9.8",
        "version": "1.0",
        "year": 2026,
        "contributor": "GTImages ground station (43.67N, 81.02W)",
        "date_created": datetime.utcnow().strftime("%Y/%m/%d"),
    }

    labeled: dict[str, Any] = {"info": info, "categories": category, "images": [], "annotations": []}
    negatives: dict[str, Any] = {"info": info, "categories": category, "images": [], "annotations": []}

    image_id = 0
    annotation_id = 0
    missing = 0
    skipped_reject = 0

    strk_files = sorted(strk_dir.glob("*.strk"))
    if not strk_files:
        raise FileNotFoundError(f"No .strk files found in {strk_dir}")

    for strk_path in strk_files:
        sat = _parse_strk_file(strk_path)
        if sat["norad_id"] is None:
            logger.warning("Could not parse NORAD ID from %s — skipping", strk_path.name)
            continue

        norad_id: int = sat["norad_id"]
        sat_name: str = sat["name"]

        for obs in sat["observations"]:
            fits_path = strk_dir / obs["filename"]
            reject = obs["reject"]

            if reject not in ("0", "-1"):
                skipped_reject += 1
                continue

            if not fits_path.exists():
                logger.warning("Missing FITS file: %s", fits_path.name)
                missing += 1
                continue

            try:
                width, height = _fits_dimensions(fits_path)
            except Exception as exc:
                logger.warning("Could not read FITS dims for %s: %s", fits_path.name, exc)
                missing += 1
                continue

            image_id += 1
            image_entry = {
                "id": image_id,
                "file_name": obs["filename"],
                "width": width,
                "height": height,
                "date_captured": obs["datetime_utc"],
            }

            if reject == "0":
                labeled["images"].append(image_entry)
                annotation_id += 1
                labeled["annotations"].append(
                    _obs_to_coco_annotation(obs, image_id, annotation_id, norad_id, sat_name)
                )
            else:  # reject == "-1"
                negatives["images"].append(image_entry)
                # No annotations for negative images (empty list is COCO-valid)

    logger.info(
        "Labeled: %d images, %d annotations",
        len(labeled["images"]),
        len(labeled["annotations"]),
    )
    logger.info("Negatives: %d images", len(negatives["images"]))
    if missing:
        logger.warning("%d observations skipped (FITS file missing)", missing)
    if skipped_reject:
        logger.debug("%d observations skipped (reject flag not 0 or -1)", skipped_reject)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(labeled, indent=2))
    logger.info("Written: %s", output_path)

    if negatives_output_path is not None:
        negatives_output_path.parent.mkdir(parents=True, exist_ok=True)
        negatives_output_path.write_text(json.dumps(negatives, indent=2))
        logger.info("Written: %s", negatives_output_path)

    return labeled, negatives


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GTImages .strk files to COCO JSON")
    parser.add_argument("--strk-dir", required=True, type=pathlib.Path, help="Directory containing .strk and .fits files")
    parser.add_argument("--output", required=True, type=pathlib.Path, help="Output COCO JSON for labeled images")
    parser.add_argument("--negatives-output", type=pathlib.Path, default=None, help="Output COCO JSON for no-streak images")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    convert(args.strk_dir, args.output, args.negatives_output)


if __name__ == "__main__":
    main()
