"""Build the architecture-aligned balanced evaluation set ``val_balanced_v1``.

Bands (eval/streak_metrics.py): short[50,400) / medium[400,1000) / long[1000+).
Frigate (<50px) is excluded entirely — a different, sub-resolvable regime.

Real medium/long/short streaks come from single/double-streak Atwood frames
(val_atwood + test_atwood), cropped to a per-frame window (``tile_origin``) so
the eval processes a small region instead of the full 6248x4176 frame.

The short band is thin in real Atwood (~36), so it is backfilled with synthetic
short streaks rendered onto real Atwood FITS backgrounds (saved as FITS, so they
traverse the same tiled path as everything else — no NPY tile-edge artifacts).

Output: data/annotations/val_balanced_v1.json + a README with band counts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.generate_synthetic_streaks import _render_streak  # noqa: E402

logger = logging.getLogger(__name__)

FRAME_W, FRAME_H = 6248, 4176
WINDOW_MARGIN = 250          # px padding around streak bbox for the crop window
SHORT_MAX, LONG_MIN, MIN_LEN = 400.0, 1000.0, 50.0


def _band(L: float) -> str:
    if L < MIN_LEN:
        return "excluded"
    if L < SHORT_MAX:
        return "short"
    if L < LONG_MIN:
        return "medium"
    return "long"


def _streak_len(obb: dict) -> float:
    return float(max(obb["w"], obb["h"]))


def _window_for(bboxes: list[list[float]]) -> tuple[int, int, int, int]:
    """Union of axis-aligned bboxes, padded and clamped to the frame."""
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[0] + b[2] for b in bboxes)
    y1 = max(b[1] + b[3] for b in bboxes)
    wx0 = max(0, int(x0 - WINDOW_MARGIN))
    wy0 = max(0, int(y0 - WINDOW_MARGIN))
    wx1 = min(FRAME_W, int(x1 + WINDOW_MARGIN))
    wy1 = min(FRAME_H, int(y1 + WINDOW_MARGIN))
    return wx0, wy0, wx1 - wx0, wy1 - wy0


def _load_real_frames(ann_files: list[str]) -> list[dict]:
    """Return per-frame records {file_name, window, streaks(local obb)} for
    single/double-streak frames whose every streak is >= MIN_LEN."""
    frames: list[dict] = []
    for f in ann_files:
        coco = json.loads(Path(f).read_text())
        img_by_id = {im["id"]: im for im in coco["images"]}
        anns_by_img: dict[int, list[dict]] = {}
        for a in coco["annotations"]:
            anns_by_img.setdefault(a["image_id"], []).append(a)
        for iid, anns in anns_by_img.items():
            if not (1 <= len(anns) <= 2):
                continue
            if any(_streak_len(a["obb"]) < MIN_LEN for a in anns):
                continue
            x0, y0, cw, ch = _window_for([a["bbox"] for a in anns])
            local = []
            for a in anns:
                o = a["obb"]
                local.append({
                    "obb": {"cx": o["cx"] - x0, "cy": o["cy"] - y0,
                            "w": o["w"], "h": o["h"], "angle_deg": o["angle_deg"]},
                    "len": _streak_len(o),
                })
            frames.append({
                "file_name": img_by_id[iid]["file_name"],
                "tile_origin": [x0, y0],
                "width": cw, "height": ch,
                "streaks": local,
            })
    return frames


def _frame_band(frame: dict) -> str:
    """Dominant band of a frame (longest streak), for selection bucketing."""
    return _band(max(s["len"] for s in frame["streaks"]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/annotations/val_balanced_v1.json")
    p.add_argument("--synth-dir", default="data/synthetic_eval/balanced_v1")
    p.add_argument("--n-medium", type=int, default=100)
    p.add_argument("--n-long", type=int, default=80)
    p.add_argument("--n-synth-short", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rng = np.random.default_rng(args.seed)

    frames = _load_real_frames([
        "data/annotations/val_atwood.json",
        "data/annotations/test_atwood.json",
    ])
    logger.info("Loaded %d candidate real frames", len(frames))

    # Bucket frames by dominant band and select to targets.
    by_band: dict[str, list[dict]] = {"short": [], "medium": [], "long": []}
    for fr in frames:
        b = _frame_band(fr)
        if b in by_band:
            by_band[b].append(fr)
    for b in by_band:
        rng.shuffle(by_band[b])

    selected: list[dict] = []
    counts = {"short": 0, "medium": 0, "long": 0}

    def _take(bucket: str, target_streaks: int) -> None:
        for fr in by_band[bucket]:
            if counts[bucket] >= target_streaks:
                break
            selected.append(fr)
            for s in fr["streaks"]:
                counts[_band(s["len"])] += 1

    _take("long", args.n_long)
    _take("medium", args.n_medium)
    # take ALL real short frames (thin band)
    for fr in by_band["short"]:
        selected.append(fr)
        for s in fr["streaks"]:
            counts[_band(s["len"])] += 1

    logger.info("Real selection: %d frames, band streak counts=%s", len(selected), counts)

    # ---- Build COCO entries from real selection ----
    images: list[dict] = []
    annotations: list[dict] = []
    next_img, next_ann = 1, 1
    for fr in selected:
        images.append({
            "id": next_img, "file_name": fr["file_name"],
            "width": fr["width"], "height": fr["height"],
            "tile_origin": fr["tile_origin"], "synthetic": False,
        })
        for s in fr["streaks"]:
            o = s["obb"]
            annotations.append(_coco_ann(next_ann, next_img, o))
            next_ann += 1
        next_img += 1

    # ---- Synthetic short backfill on real FITS backgrounds ----
    from astropy.io import fits
    synth_dir = Path(args.synth_dir)
    synth_dir.mkdir(parents=True, exist_ok=True)
    # use long/medium source frames (they have clean regions away from the streak)
    src_pool = [fr for fr in frames if _frame_band(fr) in ("long", "medium")]
    rng.shuffle(src_pool)
    made = 0
    for fr in src_pool:
        if made >= args.n_synth_short:
            break
        try:
            with fits.open(fr["file_name"]) as hdul:
                data = np.asarray(hdul[0].data, dtype=np.float32)
        except Exception as exc:
            logger.warning("skip synth src %s: %s", fr["file_name"], exc)
            continue
        # real streak bbox in full-frame coords (to avoid overlap)
        x0w, y0w = fr["tile_origin"]
        real_boxes = [(s["obb"]["cx"] + x0w, s["obb"]["cy"] + y0w,
                       max(s["obb"]["w"], s["obb"]["h"])) for s in fr["streaks"]]
        win = 800
        placed = False
        for _ in range(20):
            wx = int(rng.integers(0, FRAME_W - win))
            wy = int(rng.integers(0, FRAME_H - win))
            # reject windows overlapping a real streak centre (+ generous buffer)
            if any(wx - 300 < rx < wx + win + 300 and wy - 300 < ry < wy + win + 300
                   for rx, ry, _ in real_boxes):
                continue
            placed = True
            break
        if not placed:
            continue
        crop = data[wy:wy + win, wx:wx + win].copy()
        length = float(rng.uniform(60, 380))           # short band, with margin
        angle = float(rng.uniform(0, 180))
        cx = float(rng.uniform(length / 2 + 20, win - length / 2 - 20))
        cy = float(rng.uniform(length / 2 + 20, win - length / 2 - 20))
        snr = float(rng.uniform(5.0, 10.0))
        aug = _render_streak(crop, cx, cy, length, angle, snr, 1.5, rng)

        out_fits = synth_dir / f"synth_short_{made:03d}.fits"
        fits.PrimaryHDU(aug.astype(np.float32)).writeto(out_fits, overwrite=True)
        images.append({
            "id": next_img, "file_name": str(out_fits),
            "width": win, "height": win,
            "tile_origin": [0, 0], "synthetic": True,
        })
        annotations.append(_coco_ann(next_ann, next_img,
                                     {"cx": cx, "cy": cy, "w": length, "h": 8.0,
                                      "angle_deg": angle}))
        next_ann += 1
        next_img += 1
        made += 1
    logger.info("Synthetic short streaks rendered: %d", made)

    # ---- Final band tally ----
    final = {"short": 0, "medium": 0, "long": 0, "excluded": 0}
    for a in annotations:
        final[_band(_streak_len(a["obb"]))] += 1
    logger.info("FINAL band counts: %s  (images=%d)", final, len(images))

    out = {"images": images, "annotations": annotations,
           "categories": [{"id": 1, "name": "streak"}]}
    Path(args.out).write_text(json.dumps(out, indent=2))
    logger.info("Wrote %s", args.out)

    readme = Path(args.out).parent / "val_balanced_v1_README.md"
    readme.write_text(
        f"# val_balanced_v1\n\n"
        f"Architecture-aligned balanced eval set. Bands: short[50,400) / "
        f"medium[400,1000) / long[1000+). Frigate (<50px) excluded.\n\n"
        f"| Band | Count |\n|------|------:|\n"
        f"| short  | {final['short']} ({made} synthetic on real FITS, rest real Atwood) |\n"
        f"| medium | {final['medium']} (real Atwood) |\n"
        f"| long   | {final['long']} (real Atwood) |\n\n"
        f"Images: {len(images)} ({sum(1 for i in images if i['synthetic'])} synthetic). "
        f"Real frames are full Atwood FITS with a per-streak `tile_origin` crop window; "
        f"synthetic are 800px FITS windows on real Atwood backgrounds.\n\n"
        f"Built by `scripts/build_balanced_eval.py` (seed {args.seed}).\n"
    )
    logger.info("Wrote %s", readme)


def _coco_ann(ann_id: int, img_id: int, obb: dict) -> dict:
    import math
    w, h, ang = obb["w"], obb["h"], math.radians(obb["angle_deg"])
    hx, hy = w / 2 * math.cos(ang), w / 2 * math.sin(ang)
    x1, y1 = obb["cx"] - hx, obb["cy"] - hy
    x2, y2 = obb["cx"] + hx, obb["cy"] + hy
    return {
        "id": ann_id, "image_id": img_id, "category_id": 1,
        "obb": obb,
        "bbox": [min(x1, x2), min(y1, y2), abs(x2 - x1) + h, abs(y2 - y1) + h],
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "streak_length_px": float(max(w, h)),
        "iscrowd": 0,
    }


if __name__ == "__main__":
    main()
