"""Convert YOLO OBB label files to COCO JSON format for ARGUS training.

YOLO OBB format (one line per annotation):
  class_id cx cy w h angle_deg
Where cx, cy, w, h are normalised 0-1 relative to image dimensions,
and angle_deg is in degrees.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)

# Supported FITS extensions
_FITS_EXTS = (".fits", ".fit", ".fts")


def compute_obb_corners(
    cx: float, cy: float, w: float, h: float, angle_deg: float
) -> np.ndarray:
    """Compute the 4 corners of an oriented bounding box (OBB).

    The box is defined by its centre (cx, cy), full width w, full height h,
    and counter-clockwise rotation angle_deg. Coordinates are in the same
    space as cx/cy (pixels or normalised — the caller's responsibility).

    Args:
        cx: Centre x coordinate.
        cy: Centre y coordinate.
        w: Full width of the box.
        h: Full height of the box.
        angle_deg: Rotation angle in degrees (counter-clockwise).

    Returns:
        np.ndarray of shape (4, 2) with the four corner coordinates in order:
        top-left, top-right, bottom-right, bottom-left.
    """
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Half extents
    hw, hh = w / 2.0, h / 2.0

    # Unrotated corners relative to centre
    corners_local = np.array(
        [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float64
    )

    # Rotation matrix (counter-clockwise)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    corners_rotated = corners_local @ rot.T
    corners_rotated[:, 0] += cx
    corners_rotated[:, 1] += cy
    return corners_rotated


def convert_yolo_obb_to_coco(
    yolo_label_dir: str | Path,
    fits_dir: str | Path,
    output_json: str | Path,
) -> None:
    """Convert a directory of YOLO OBB label files to a COCO JSON file.

    For each .txt file in yolo_label_dir:
      - Find matching FITS (same stem, .fits/.fit/.fts) in fits_dir.
      - Read image dims from FITS header (NAXIS1=width, NAXIS2=height).
      - Denormalise cx, cy, w, h to pixel space.
      - Compute axis-aligned bbox from OBB corners.
      - Store obb as annotation["obb"] = [cx, cy, w, h, angle_deg] (pixels).

    COCO output format:
      images: [{id, file_name, width, height}]
      annotations: [{id, image_id, category_id=0, bbox=[x1,y1,w,h],
                      area, obb=[cx,cy,w,h,angle_deg], iscrowd=0}]
      categories: [{"id": 0, "name": "streak"}]

    Prints to stdout:
      Total images, total streaks, streak length stats (min/mean/max/p75 px).

    Args:
        yolo_label_dir: Directory containing YOLO OBB .txt label files.
        fits_dir: Directory to search for matching FITS images.
        output_json: Destination path for the COCO JSON file.
    """
    yolo_label_dir = Path(yolo_label_dir)
    fits_dir = Path(fits_dir)
    output_json = Path(output_json)

    images: list[dict] = []
    annotations: list[dict] = []
    categories = [{"id": 0, "name": "streak"}]

    ann_id = 1
    img_id = 1
    streak_lengths: list[float] = []

    label_files = sorted(yolo_label_dir.glob("*.txt"))

    for label_path in label_files:
        # Find matching FITS file
        fits_path: Path | None = None
        for ext in _FITS_EXTS:
            candidate = fits_dir / (label_path.stem + ext)
            if candidate.exists():
                fits_path = candidate
                break

        if fits_path is None:
            logger.warning(
                "No FITS file found for label %s in %s — skipping",
                label_path.name,
                fits_dir,
            )
            continue

        # Read image dimensions from FITS header
        try:
            with fits.open(fits_path) as hdul:
                header = hdul[0].header
                img_w = int(header["NAXIS1"])
                img_h = int(header["NAXIS2"])
        except Exception as exc:
            logger.warning(
                "Cannot read FITS header from %s: %s — skipping", fits_path.name, exc
            )
            continue

        images.append(
            {
                "id": img_id,
                "file_name": fits_path.name,
                "width": img_w,
                "height": img_h,
            }
        )

        # Parse label file
        lines = label_path.read_text().strip().splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                logger.warning(
                    "Malformed label line in %s: '%s' — skipping", label_path.name, line
                )
                continue

            _class_id = int(parts[0])
            cx_n = float(parts[1])
            cy_n = float(parts[2])
            w_n = float(parts[3])
            h_n = float(parts[4])
            angle_deg = float(parts[5])

            # Denormalise to pixel space
            cx_px = cx_n * img_w
            cy_px = cy_n * img_h
            w_px = w_n * img_w
            h_px = h_n * img_h

            # Axis-aligned bbox from OBB corners
            corners = compute_obb_corners(cx_px, cy_px, w_px, h_px, angle_deg)
            x1 = float(corners[:, 0].min())
            y1 = float(corners[:, 1].min())
            x2 = float(corners[:, 0].max())
            y2 = float(corners[:, 1].max())
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            area = float(bbox_w * bbox_h)

            # Streak length ≈ max OBB dimension
            streak_len = max(w_px, h_px)
            streak_lengths.append(streak_len)

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 0,
                    "bbox": [x1, y1, bbox_w, bbox_h],
                    "area": area,
                    "obb": [cx_px, cy_px, w_px, h_px, angle_deg],
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        img_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(coco, indent=2))
    logger.info("Wrote COCO JSON to %s", output_json)

    # Print stats to stdout
    total_images = len(images)
    total_streaks = len(streak_lengths)
    print(f"Total images  : {total_images}")
    print(f"Total streaks : {total_streaks}")
    if streak_lengths:
        arr = np.array(streak_lengths)
        print(f"Streak length : min={arr.min():.1f} mean={arr.mean():.1f} "
              f"max={arr.max():.1f} p75={float(np.percentile(arr, 75)):.1f} px")
    else:
        print("Streak length : N/A (no annotations)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Convert YOLO OBB labels to COCO JSON")
    parser.add_argument("--yolo-labels", required=True, help="Directory of YOLO .txt labels")
    parser.add_argument("--fits-dir", required=True, help="Directory of FITS images")
    parser.add_argument("--output", required=True, help="Output COCO JSON path")
    args = parser.parse_args()

    convert_yolo_obb_to_coco(
        yolo_label_dir=args.yolo_labels,
        fits_dir=args.fits_dir,
        output_json=args.output,
    )
