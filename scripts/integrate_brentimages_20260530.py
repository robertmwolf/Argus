#!/usr/bin/env python3
"""Integrate 20260530_Atwood BrentImages annotations into all_train_run17_merged.json.

Reads each .strk file in the source directory, creates 1800×1800 windowed crops
centered on each valid streak midpoint, converts OBBs to window-local coordinates,
and merges the new image/annotation entries into the target JSON.

Only Reject=0 entries (manual annotations) are included. Reject=-1 (confirmed no
streak) and Reject=5 (unusable) are skipped.

Usage:
    python scripts/integrate_brentimages_20260530.py \\
        [--source-dir /Volumes/External/TrainingData/raw/BrentImages/20260530_Atwood] \\
        [--merged data/annotations/all_train_run17_merged.json] \\
        [--output data/annotations/all_train_run17_merged.json]  # in-place by default
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logger = logging.getLogger(__name__)

FRAME_W, FRAME_H = 6248, 4176
WINDOW_SIZE = 1800
STRK_WIDTH = 16.0  # default streak cross-section width in pixels


def _parse_strk(strk_path: Path) -> tuple[str, list[dict]]:
    """Parse a .strk file. Returns (satellite_name, list of obs dicts)."""
    text = strk_path.read_text()
    sections = {}
    current = None
    lines_by_section: dict[str, list[str]] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1]
            lines_by_section[current] = []
        elif current is not None:
            lines_by_section[current].append(line)

    # Parse satellite name from TLE section
    satellite_name = ""
    tle_lines = [l for l in lines_by_section.get("TLE", []) if l.strip()]
    if len(tle_lines) >= 2:
        # Header is first non-empty line, data is second
        header = tle_lines[0].split("\t")
        data = tle_lines[1].split("\t")
        col = {h.strip(): i for i, h in enumerate(header)}
        if "Name" in col and col["Name"] < len(data):
            satellite_name = data[col["Name"]].strip()

    # Parse OBS section
    obs_lines = [l for l in lines_by_section.get("OBS", []) if l.strip()]
    if len(obs_lines) < 2:
        return satellite_name, []

    header_fields = [f.strip() for f in obs_lines[0].split("\t")]
    col = {name: i for i, name in enumerate(header_fields)}

    def _g(row: list[str], name: str, default=None):
        i = col.get(name)
        if i is None or i >= len(row):
            return default
        v = row[i].strip()
        return v if v else default

    obs: list[dict] = []
    for line in obs_lines[1:]:
        if not line.strip():
            continue
        row = line.split("\t")
        reject_str = _g(row, "Reject", "5")
        try:
            reject = int(reject_str)
        except ValueError:
            continue
        if reject != 0:
            continue  # skip unusable (5) and confirmed-no-streak (-1)

        try:
            start_x = float(_g(row, "Start X Pixel", "0"))
            start_y = float(_g(row, "Start Y Pixel", "0"))
            end_x = float(_g(row, "End X Pixel", "0"))
            end_y = float(_g(row, "End Y Pixel", "0"))
            mid_x = float(_g(row, "Mid X Pixel", "0"))
            mid_y = float(_g(row, "Mid Y Pixel", "0"))
            peak_snr = float(_g(row, "Peak SNR", "0"))
            mean_snr = float(_g(row, "Mean SNR", "0"))
            elongation = float(_g(row, "Elongation", "0"))
            length_px = float(_g(row, "Length", "0"))
            jd_mid = float(_g(row, "JD Midpoint", "0"))
            datetime_utc = _g(row, "Date Time(UTC)", "")
            image_fname = _g(row, "Image", "")
        except (ValueError, TypeError):
            continue

        if not image_fname:
            continue

        obs.append({
            "image": image_fname,
            "start_x": start_x,
            "start_y": start_y,
            "end_x": end_x,
            "end_y": end_y,
            "mid_x": mid_x,
            "mid_y": mid_y,
            "peak_snr": peak_snr,
            "mean_snr": mean_snr,
            "elongation": elongation,
            "length_px": length_px,
            "jd_mid": jd_mid,
            "datetime_utc": datetime_utc,
        })

    return satellite_name, obs


def _obb_from_endpoints(
    x1: float, y1: float, x2: float, y2: float, width: float = STRK_WIDTH
) -> dict:
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    angle_deg = math.degrees(math.atan2(dy, dx))
    return {"cx": cx, "cy": cy, "w": length, "h": width, "angle_deg": angle_deg}


def _aabb_from_obb(obb: dict) -> tuple[float, float, float, float]:
    """Axis-aligned bbox [x0, y0, w, h] of an oriented bounding box."""
    cx, cy = obb["cx"], obb["cy"]
    w, h = obb["w"], obb["h"]
    theta = math.radians(obb["angle_deg"])
    cos_t, sin_t = abs(math.cos(theta)), abs(math.sin(theta))
    half_W = (w / 2) * cos_t + (h / 2) * sin_t
    half_H = (w / 2) * sin_t + (h / 2) * cos_t
    return (cx - half_W, cy - half_H, 2 * half_W, 2 * half_H)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--source-dir",
        default="/Volumes/External/TrainingData/raw/BrentImages/20260530_Atwood",
        help="directory containing .strk and .fits files",
    )
    ap.add_argument(
        "--merged",
        default="data/annotations/all_train_run17_merged.json",
        help="existing merged annotation JSON to extend",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="output path (default: overwrite --merged in-place)",
    )
    ap.add_argument("--dry-run", action="store_true",
                    help="print stats without modifying any files")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    source_dir = Path(args.source_dir)
    merged_path = _REPO / args.merged
    out_path = Path(args.output) if args.output else merged_path

    # Load merged JSON
    logger.info("Loading merged JSON from %s", merged_path)
    merged = json.loads(merged_path.read_text())
    max_image_id = max(im["id"] for im in merged["images"]) if merged["images"] else 0
    max_ann_id = max(a["id"] for a in merged["annotations"]) if merged["annotations"] else 0
    logger.info("Existing: %d images, %d annotations (max_img_id=%d, max_ann_id=%d)",
                len(merged["images"]), len(merged["annotations"]), max_image_id, max_ann_id)

    # Check for already-integrated entries (idempotency guard)
    existing_fnames = {im["file_name"] for im in merged["images"]}

    strk_files = sorted(source_dir.glob("*.strk"))
    logger.info("Found %d .strk files in %s", len(strk_files), source_dir)

    new_images: list[dict] = []
    new_annotations: list[dict] = []
    skipped_missing = 0
    skipped_duplicate = 0
    skipped_zero_length = 0

    next_image_id = max_image_id + 1
    next_ann_id = max_ann_id + 1

    for strk_path in strk_files:
        norad_id_str = strk_path.stem
        try:
            norad_id = int(norad_id_str)
        except ValueError:
            logger.warning("Non-integer .strk stem, skipping: %s", strk_path.name)
            continue

        sat_name, obs_list = _parse_strk(strk_path)
        if not obs_list:
            continue

        logger.info("NORAD %d (%s): %d valid observations", norad_id, sat_name, len(obs_list))

        for obs in obs_list:
            fits_path = str(source_dir / obs["image"])

            if fits_path in existing_fnames:
                skipped_duplicate += 1
                continue

            # Verify FITS exists
            if not Path(fits_path).exists():
                logger.warning("FITS not found, skipping: %s", fits_path)
                skipped_missing += 1
                continue

            # Compute OBB in full-frame coords
            obb_full = _obb_from_endpoints(
                obs["start_x"], obs["start_y"],
                obs["end_x"], obs["end_y"],
            )

            if obb_full["w"] < 1.0:
                skipped_zero_length += 1
                continue

            # Window centered on streak midpoint
            mid_x = obs["mid_x"] if obs["mid_x"] != 0 else obb_full["cx"]
            mid_y = obs["mid_y"] if obs["mid_y"] != 0 else obb_full["cy"]
            x0 = int(max(0, min(round(mid_x) - WINDOW_SIZE // 2, FRAME_W - WINDOW_SIZE)))
            y0 = int(max(0, min(round(mid_y) - WINDOW_SIZE // 2, FRAME_H - WINDOW_SIZE)))

            # Convert OBB to window-local coordinates
            obb_local = {
                "cx": obb_full["cx"] - x0,
                "cy": obb_full["cy"] - y0,
                "w": obb_full["w"],
                "h": obb_full["h"],
                "angle_deg": obb_full["angle_deg"],
            }

            bbox_x0, bbox_y0, bbox_w, bbox_h = _aabb_from_obb(obb_local)

            image_entry = {
                "id": next_image_id,
                "file_name": fits_path,
                "width": WINDOW_SIZE,
                "height": WINDOW_SIZE,
                "orig_image_id": next_image_id,
                "tile_origin": [x0, y0],
            }

            ann_entry = {
                "id": next_ann_id,
                "image_id": next_image_id,
                "category_id": 1,
                "bbox": [bbox_x0, bbox_y0, bbox_w, bbox_h],
                "area": float(bbox_w * bbox_h),
                "iscrowd": 0,
                "obb": obb_local,
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

            new_images.append(image_entry)
            new_annotations.append(ann_entry)
            next_image_id += 1
            next_ann_id += 1

    logger.info(
        "New entries: %d images, %d annotations | skipped: %d missing FITS, "
        "%d duplicates, %d zero-length",
        len(new_images), len(new_annotations),
        skipped_missing, skipped_duplicate, skipped_zero_length,
    )

    if not new_images:
        logger.info("Nothing to add — merged JSON unchanged.")
        return

    if args.dry_run:
        logger.info("Dry run — no files written.")
        return

    merged["images"].extend(new_images)
    merged["annotations"].extend(new_annotations)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2))
    logger.info(
        "Saved to %s (%d images, %d annotations total)",
        out_path, len(merged["images"]), len(merged["annotations"]),
    )


if __name__ == "__main__":
    main()
