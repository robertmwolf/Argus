"""Rank Frigate processed PNGs by likelihood of containing a satellite streak.

Frigate (DanSRoll/frigate, Nature Scientific Data 2025) is a 2,000-frame FITS
time series (QHY600M, 9600×6422, 0.5s exposures) with processed PNGs at
2325×1555 px. Short satellite streaks (~50–150px) are expected but were missed
by previous auto-detection attempts tuned for longer streaks.

Strategy — frame differencing:
    Stars are effectively fixed between adjacent 0.5-second frames; satellites
    are not.  abs(frame[t] − frame[t−1]) suppresses static background and
    highlights transients.  This script uses a permissive Hough pass on the
    diff image as a *triage* screener only — false positives are acceptable
    (they cost one human glance); false negatives are expensive (missed frames
    never get annotated).

Output:
    data/annotations/frigate_screen.json  — all frames ranked by score
    Optional contact sheet PNG of top-N colourised diff thumbnails.

Usage:
    # Full run (all 1,980 frames, 8 workers):
    python scripts/screen_frigate.py \\
        --workers 8 \\
        --contact-sheet data/annotations/frigate_contacts.png

    # Smoke test (first 50 frames):
    python scripts/screen_frigate.py --max-frames 50 --workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Regex to parse "Capture_NNNNN HH_MM_SSZ.png"
_FNAME_PAT = re.compile(r"Capture_(\d{5}) (\d{2})_(\d{2})_(\d{2})Z\.png", re.IGNORECASE)

# Hough / scoring parameters (diff images, 2325×1555, targeting 50–150px streaks)
# NOTE: threshold and minLineLength must be high enough to reject star-cancellation
# residuals. Star dipoles in a diff image are point-like (<15px); satellite streaks
# are ≥50px. Setting minLineLength=55 and threshold=80 eliminates dipole noise while
# retaining genuine streak signals.
_HOUGH_THRESHOLD = 80       # votes required; rejects star dipoles (<15px residuals)
_HOUGH_MIN_LINE = 55        # px; above typical star-dipole extent (~10–15px)
_HOUGH_MAX_GAP = 8          # px; tight bridging so dipole pairs don't merge into lines
_BLOB_THRESH = 40           # diff pixel value threshold for connected-component analysis
_BLOB_MIN_DIM = 50          # px; minimum max_dim of an elongated blob
_BLOB_MAX_DIM = 150         # px; maximum max_dim (above = not a short streak)
_BLOB_ASPECT = 4.0          # minimum aspect ratio — streaks are more elongated than dipoles
_AMPLIFY = 4                # multiplier for contact-sheet diff display


def _parse_filename(stem: str) -> tuple[int, str]:
    """Return (frame_number, timestamp_utc) from a Frigate PNG stem.

    Examples:
        "Capture_00245 03_02_04Z" -> (245, "03:02:04Z")
    """
    m = _FNAME_PAT.match(stem + ".png")
    if not m:
        return (0, "")
    frame_num = int(m.group(1))
    ts = f"{m.group(2)}:{m.group(3)}:{m.group(4)}Z"
    return frame_num, ts


def _load_gray_float(path: Path) -> np.ndarray | None:
    """Load PNG as float32 grayscale [0, 255].  Returns None if unreadable."""
    try:
        import cv2 as cv
        img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
        if img is None:
            return None
        return img.astype(np.float32)
    except Exception:
        return None


def _compute_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Absolute difference clamped to uint8.

    Uses float32 arithmetic to avoid wrap-around (e.g. 5 − 200 = 51 in uint8).
    """
    return np.clip(np.abs(a - b), 0, 255).astype(np.uint8)


def _enhance_diff(diff_u8: np.ndarray) -> np.ndarray:
    """CLAHE + Gaussian blur + Canny edge map optimised for diff images.

    Lighter preprocessing than the annotator's source-image pipeline because
    star cancellation already removes large-scale structure; smaller CLAHE tiles
    and Canny thresholds expose faint transient edges.
    """
    try:
        import cv2 as cv
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(diff_u8)
        blurred = cv.GaussianBlur(enhanced, (3, 3), 0)
        return cv.Canny(blurred, 20, 80)
    except Exception:
        return np.zeros_like(diff_u8)


def _score_diff(diff_u8: np.ndarray, edges: np.ndarray) -> dict:
    """Extract scoring components from a diff image and its edge map.

    Returns a dict with:
        hough_count:       number of Hough lines passing the filter
        elongated_blobs:   number of connected components with aspect ≥ _BLOB_ASPECT
        max_diff_response: maximum pixel value in diff_u8
        score:             composite float (higher = more likely to contain a streak)
        candidates:        list of {x1,y1,x2,y2,length_px,angle_deg} dicts
    """
    try:
        import cv2 as cv
    except ImportError:
        return {"hough_count": 0, "elongated_blobs": 0,
                "max_diff_response": 0.0, "score": 0.0, "candidates": []}

    h, w = diff_u8.shape
    border = 20  # smaller than annotator (40) because diff images are smaller

    # ---- Hough on edge map -------------------------------------------------
    lines = cv.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=_HOUGH_THRESHOLD,
        minLineLength=_HOUGH_MIN_LINE,
        maxLineGap=_HOUGH_MAX_GAP,
    )
    candidates: list[dict] = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = map(float, line[0])
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            if cx < border or cx > w - border or cy < border or cy > h - border:
                continue
            length = math.hypot(x2 - x1, y2 - y1)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            candidates.append({
                "x1": round(x1), "y1": round(y1),
                "x2": round(x2), "y2": round(y2),
                "length_px": round(length, 1),
                "angle_deg": round(angle, 1),
            })

    # ---- Elongated blob filter on thresholded diff -------------------------
    _, binary = cv.threshold(diff_u8, _BLOB_THRESH, 255, cv.THRESH_BINARY)
    n_labels, labels, stats, _ = cv.connectedComponentsWithStats(binary)
    elongated_blobs = 0
    for lbl in range(1, n_labels):
        left  = stats[lbl, cv.CC_STAT_LEFT]
        top   = stats[lbl, cv.CC_STAT_TOP]
        bw    = stats[lbl, cv.CC_STAT_WIDTH]
        bh    = stats[lbl, cv.CC_STAT_HEIGHT]
        max_d = max(bw, bh)
        min_d = min(bw, bh) or 1
        cx    = left + bw / 2
        cy    = top  + bh / 2
        if (max_d >= _BLOB_MIN_DIM and max_d <= _BLOB_MAX_DIM
                and max_d / min_d >= _BLOB_ASPECT
                and cx >= border and cx <= w - border
                and cy >= border and cy <= h - border):
            elongated_blobs += 1

    max_resp = float(diff_u8.max())
    norm_resp = max(0.0, max_resp - 25.0) / 230.0
    # Cap hough_count at 10 so a flood of false detections can't dominate the score
    score = 4.0 * elongated_blobs + 2.0 * min(len(candidates), 10) + 3.0 * norm_resp

    return {
        "hough_count": len(candidates),
        "elongated_blobs": elongated_blobs,
        "max_diff_response": round(max_resp, 1),
        "score": round(score, 3),
        "candidates": candidates,
    }


def _process_pair(args: tuple[int, Path, Path]) -> dict:
    """Score one frame by differencing with its neighbour.

    Module-level so multiprocessing.Pool can pickle it.

    Args:
        args: (frame_index, this_path, prev_path)
    """
    idx, this_path, prev_path = args

    frame_num, ts = _parse_filename(this_path.stem)

    cur  = _load_gray_float(this_path)
    prev = _load_gray_float(prev_path)

    if cur is None or prev is None:
        return {
            "frame": this_path.name,
            "path": str(this_path),
            "frame_number": frame_num,
            "timestamp_utc": ts,
            "prev_frame": prev_path.name,
            "score": 0.0,
            "hough_count": 0,
            "elongated_blobs": 0,
            "max_diff_response": 0.0,
            "candidates": [],
            "error": "unreadable",
        }

    diff_u8 = _compute_diff(cur, prev)
    edges   = _enhance_diff(diff_u8)
    scoring = _score_diff(diff_u8, edges)

    return {
        "frame": this_path.name,
        "path": str(this_path),
        "frame_number": frame_num,
        "timestamp_utc": ts,
        "prev_frame": prev_path.name,
        **scoring,
    }


def _make_contact_sheet(
    results: list[dict],
    top_n: int,
    out_path: Path,
) -> None:
    """Write a contact sheet of colourised diff thumbnails for the top-N frames.

    Layout: 5 columns × ceil(top_n / 5) rows, thumbnails at 300×200 px,
    10 px margins.  Diff is amplified ×_AMPLIFY and colourised with COLORMAP_HOT.
    """
    try:
        import cv2 as cv
    except ImportError:
        logger.warning("opencv-python not available — skipping contact sheet")
        return

    top = results[:top_n]
    cols = 5
    rows = math.ceil(len(top) / cols)
    thumb_w, thumb_h = 300, 200
    margin = 10
    sheet_w = cols * thumb_w + (cols + 1) * margin
    sheet_h = rows * thumb_h + (rows + 1) * margin
    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

    for i, entry in enumerate(top):
        col = i % cols
        row = i // cols
        x0 = margin + col * (thumb_w + margin)
        y0 = margin + row * (thumb_h + margin)

        this_path = Path(entry["path"])
        prev_path = this_path.parent / entry["prev_frame"]

        cur  = _load_gray_float(this_path)
        prev = _load_gray_float(prev_path)
        if cur is None or prev is None:
            continue

        diff_u8 = _compute_diff(cur, prev)
        amplified = np.clip(diff_u8.astype(np.float32) * _AMPLIFY, 0, 255).astype(np.uint8)
        coloured = cv.applyColorMap(amplified, cv.COLORMAP_HOT)
        thumb = cv.resize(coloured, (thumb_w, thumb_h), interpolation=cv.INTER_AREA)

        # Overlay score and frame number
        label = f"#{entry['frame_number']}  s={entry['score']:.1f}"
        cv.putText(thumb, label, (4, 16),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv.LINE_AA)

        sheet[y0:y0 + thumb_h, x0:x0 + thumb_w] = thumb

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(out_path), sheet)
    logger.info("Contact sheet written → %s  (%d × %d px)", out_path, sheet_w, sheet_h)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--processed-dir", type=Path,
        default=Path("/Volumes/External/frigate/processed"),
        help="Directory containing Frigate processed PNGs.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/annotations/frigate_screen.json"),
        help="Output ranking JSON.",
    )
    parser.add_argument(
        "--workers", type=int,
        default=multiprocessing.cpu_count(),
        help="Number of worker processes (default: cpu_count).",
    )
    parser.add_argument(
        "--top-n", type=int, default=50,
        help="Number of frames to include in the contact sheet.",
    )
    parser.add_argument(
        "--contact-sheet", type=Path, default=None,
        metavar="PATH",
        help="If set, write a contact sheet PNG of the top-N frames.",
    )
    parser.add_argument(
        "--min-score", type=float, default=0.0,
        help="Exclude frames with score below this from the output JSON. "
             "Default 0.0 = include all frames.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Cap the number of frames processed (for smoke testing).",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    processed_dir: Path = args.processed_dir
    if not processed_dir.exists():
        raise SystemExit(f"Processed dir not found: {processed_dir}")

    pngs = sorted(processed_dir.glob("*.png"))
    if not pngs:
        raise SystemExit(f"No PNG files found in {processed_dir}")

    if args.max_frames is not None:
        pngs = pngs[: args.max_frames]

    logger.info("Found %d PNG frames in %s", len(pngs), processed_dir)

    # Build pairs: each frame diffs against its predecessor.
    # Frame 0 uses a forward diff (against frame 1) to avoid being scored as zero.
    pairs: list[tuple[int, Path, Path]] = []
    for i, path in enumerate(pngs):
        prev = pngs[i - 1] if i > 0 else (pngs[1] if len(pngs) > 1 else pngs[0])
        pairs.append((i, path, prev))

    logger.info("Scoring %d pairs with %d workers…", len(pairs), args.workers)
    with multiprocessing.Pool(processes=args.workers) as pool:
        results: list[dict] = pool.map(_process_pair, pairs)

    results.sort(key=lambda r: -r["score"])

    n_with_signal = sum(1 for r in results if r["score"] >= 1.0)
    logger.info(
        "Done. %d / %d frames have score ≥ 1.0", n_with_signal, len(results)
    )

    if args.min_score > 0.0:
        before = len(results)
        results = [r for r in results if r["score"] >= args.min_score]
        logger.info(
            "Filtered to %d frames with score ≥ %.2f (excluded %d)",
            len(results), args.min_score, before - len(results),
        )

    output = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "processed_dir": str(processed_dir),
            "n_frames": len(pairs),
            "n_with_signal": n_with_signal,
            "hough_params": {
                "threshold": _HOUGH_THRESHOLD,
                "min_line_length": _HOUGH_MIN_LINE,
                "max_line_gap": _HOUGH_MAX_GAP,
            },
            "scoring_weights": {
                "elongated_blobs": 4.0,
                "hough_count": 2.0,
                "normalized_max_response": 3.0,
            },
        },
        "frames": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    logger.info("Ranking written → %s  (%d frames)", args.output, len(results))

    if args.contact_sheet is not None:
        logger.info("Generating contact sheet (top %d)…", args.top_n)
        _make_contact_sheet(results, args.top_n, args.contact_sheet)


if __name__ == "__main__":
    main()
