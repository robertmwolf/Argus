"""Build the Run 18 ViT-B retrain split — consistent real-FITS distribution.

Root cause of Run 17's failure: the training set mixed three normalisation
regimes (raw FITS + PNG + synthetic NPY, 70% synthetic) while validation was
pure FITS, so the head underfit (train_dice 0.37) and val_dice (0.105) was
measuring a different distribution. Run 15 (ViT-S, dice 0.77) succeeded because
train and val were one consistent NPY distribution.

This builder produces ONE consistent zscore-FITS distribution:
  * Positive windows: real Atwood FITS streak windows (non-eval), streaks >=50px.
  * Synthetic short: streaks rendered ON real FITS windows (FITS output) to
    strengthen the thin short band (the Run 18 gate metric).
  * Negative-sky windows: clean crops (no annotations) to hit ~38% negative
    tiles at cache time (Run 10 finding: 38% neg is best; 42% hurts).

Split is by SOURCE FRAME (no leakage) and excludes all val_atwood/test_atwood
frames (val_balanced_v1, the Run 18 eval gate, is built from those).

Output: data/annotations/{train,val}_run18.json + README.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.generate_synthetic_streaks import _render_streak  # noqa: E402

logger = logging.getLogger(__name__)
FRAME_W, FRAME_H = 6248, 4176
MIN_LEN, SHORT_MAX, LONG_MIN = 50.0, 400.0, 1000.0


def _band(L: float) -> str:
    if L < SHORT_MAX:
        return "short"
    if L < LONG_MIN:
        return "medium"
    return "long"


def _coco_ann(ann_id, img_id, obb):
    import math
    w, h, ang = obb["w"], obb["h"], math.radians(obb["angle_deg"])
    hx, hy = w / 2 * math.cos(ang), w / 2 * math.sin(ang)
    x1, y1, x2, y2 = obb["cx"] - hx, obb["cy"] - hy, obb["cx"] + hx, obb["cy"] + hy
    return {"id": ann_id, "image_id": img_id, "category_id": 1, "obb": obb,
            "bbox": [min(x1, x2), min(y1, y2), abs(x2 - x1) + h, abs(y2 - y1) + h],
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "streak_length_px": float(max(w, h)), "iscrowd": 0}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--n-synth-short", type=int, default=400)
    p.add_argument("--neg-frac", type=float, default=0.38,
                   help="target negative TILE fraction (Run 10: 0.38 best)")
    p.add_argument("--pos-tiles-per-window", type=float, default=3.0,
                   help="approx positive tiles the cacher keeps per window (for neg sizing)")
    p.add_argument("--neg-tiles-per-image", type=int, default=4,
                   help="must match the cacher's --neg-tiles-per-image")
    p.add_argument("--synth-dir", default="data/synthetic_eval/run18_synth_short")
    p.add_argument("--seed", type=int, default=18)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rng = np.random.default_rng(args.seed)

    eval_frames = set()
    for f in ["val_atwood", "test_atwood"]:
        for im in json.load(open(f"data/annotations/{f}.json"))["images"]:
            eval_frames.add(im["file_name"])

    merged = json.load(open("data/annotations/all_train_run17_merged.json"))
    anns_by_img = defaultdict(list)
    for a in merged["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    # Collect positive real-FITS windows (non-eval, streaks >=50px).
    windows = []  # {file_name, tile_origin, w, h, anns(window-local obb)}
    for im in merged["images"]:
        fn = im["file_name"]
        if not fn.lower().endswith(".fits") or fn in eval_frames:
            continue
        keep = [a for a in anns_by_img[im["id"]]
                if max(a["obb"]["w"], a["obb"]["h"]) >= MIN_LEN]
        if not keep:
            continue
        windows.append({"file_name": fn, "tile_origin": im.get("tile_origin", [0, 0]),
                        "w": im["width"], "h": im["height"],
                        "anns": [{"obb": a["obb"], "len": max(a["obb"]["w"], a["obb"]["h"])}
                                 for a in keep]})
    logger.info("Positive real-FITS windows: %d", len(windows))

    # Split by source frame (no leakage), stratified by dominant band.
    by_frame = defaultdict(list)
    for w in windows:
        by_frame[w["file_name"]].append(w)
    frames = list(by_frame)
    frame_band = {fr: _band(max(max(a["len"] for a in w["anns"]) for w in by_frame[fr]))
                  for fr in frames}
    val_frames = set()
    for band in ("short", "medium", "long"):
        fl = [fr for fr in frames if frame_band[fr] == band]
        rng.shuffle(fl)
        val_frames.update(fl[:max(1, int(len(fl) * args.val_frac))])

    def emit(window_list, img0, ann0):
        images, anns = [], []
        iid, aid = img0, ann0
        for w in window_list:
            images.append({"id": iid, "file_name": w["file_name"], "width": w["w"],
                           "height": w["h"], "tile_origin": w["tile_origin"],
                           "synthetic": w.get("synthetic", False)})
            for a in w["anns"]:
                anns.append(_coco_ann(aid, iid, a["obb"]))
                aid += 1
            iid += 1
        return images, anns, iid, aid

    train_w = [w for w in windows if w["file_name"] not in val_frames]
    val_w = [w for w in windows if w["file_name"] in val_frames]

    # ---- Synthetic short streaks on real FITS windows (train only) ----
    from astropy.io import fits
    synth_dir = Path(args.synth_dir)
    synth_dir.mkdir(parents=True, exist_ok=True)
    synth_windows = []
    src = [w for w in train_w]
    rng.shuffle(src)
    made = 0
    for w in src:
        if made >= args.n_synth_short:
            break
        try:
            with fits.open(w["file_name"]) as h:
                data = np.asarray(h[0].data, dtype=np.float32)
        except Exception:
            continue
        x0, y0 = int(w["tile_origin"][0]), int(w["tile_origin"][1])
        crop = data[y0:y0 + w["h"], x0:x0 + w["w"]].copy()
        if crop.shape[0] < 200 or crop.shape[1] < 200:
            continue
        ch, cw = crop.shape
        length = float(rng.uniform(60, 380))
        angle = float(rng.uniform(0, 180))
        cx = float(rng.uniform(length / 2 + 20, cw - length / 2 - 20))
        cy = float(rng.uniform(length / 2 + 20, ch - length / 2 - 20))
        aug = _render_streak(crop, cx, cy, length, angle, float(rng.uniform(5, 10)), 1.5, rng)
        out = synth_dir / f"synth_{made:04d}.fits"
        fits.PrimaryHDU(aug.astype(np.float32)).writeto(out, overwrite=True)
        synth_windows.append({"file_name": str(out), "tile_origin": [0, 0], "w": cw, "h": ch,
                              "synthetic": True,
                              "anns": [{"obb": {"cx": cx, "cy": cy, "w": length, "h": 8.0,
                                                "angle_deg": angle}, "len": length}]})
        made += 1
    logger.info("Synthetic short windows: %d", made)
    train_w = train_w + synth_windows

    # ---- Negative-sky windows (no annotations) for ~38% neg tiles ----
    # neg_tiles = neg_frac/(1-neg_frac) * pos_tiles; pos_tiles ~= pos_windows * pos_per_window
    def neg_windows_for(pos_windows):
        pos_tiles = len(pos_windows) * args.pos_tiles_per_window
        neg_tiles = args.neg_frac / (1 - args.neg_frac) * pos_tiles
        return int(round(neg_tiles / max(1, args.neg_tiles_per_image)))

    def make_negs(pos_windows, n):
        out = []
        pool = [w for w in pos_windows if not w.get("synthetic")]
        for i in range(n):
            w = pool[i % len(pool)]
            # a window elsewhere in the same full frame, away from the streak window
            sx, sy = int(w["tile_origin"][0]), int(w["tile_origin"][1])
            for _ in range(20):
                nx = int(rng.integers(0, FRAME_W - 1800))
                ny = int(rng.integers(0, FRAME_H - 1800))
                if abs(nx - sx) > 1800 or abs(ny - sy) > 1800:
                    out.append({"file_name": w["file_name"], "tile_origin": [nx, ny],
                                "w": 1800, "h": 1800, "anns": []})
                    break
        return out

    train_negs = make_negs(train_w, neg_windows_for(train_w))
    val_negs = make_negs(val_w, neg_windows_for(val_w))
    logger.info("Negative-sky windows: train=%d val=%d", len(train_negs), len(val_negs))

    tr_imgs, tr_anns, nid, naid = emit(train_w + train_negs, 1, 1)
    va_imgs, va_anns, _, _ = emit(val_w + val_negs, nid, naid)

    for name, imgs, anns in [("train_run18", tr_imgs, tr_anns), ("val_run18", va_imgs, va_anns)]:
        out = {"images": imgs, "annotations": anns, "categories": [{"id": 1, "name": "streak"}]}
        Path(f"data/annotations/{name}.json").write_text(json.dumps(out))
        b = Counter(_band(a["streak_length_px"]) for a in anns)
        ann_img_ids = {a["image_id"] for a in anns}
        nneg = sum(1 for im in imgs if im["id"] not in ann_img_ids)
        logger.info("%s: %d images (%d neg windows), streaks short=%d med=%d long=%d",
                    name, len(imgs), nneg, b["short"], b["medium"], b["long"])

    Path("data/annotations/run18_split_README.md").write_text(
        f"# Run 18 split (consistent real-FITS)\n\n"
        f"Fixes Run 17's format mismatch: real FITS only (no PNG/NPY), one zscore\n"
        f"distribution for train+val. Synthetic short streaks rendered on real FITS.\n"
        f"~{int(args.neg_frac*100)}% negative tiles (Run 10 best). Split by frame, no\n"
        f"leakage; val_atwood/test_atwood frames excluded (val_balanced_v1 is the gate).\n\n"
        f"Built by scripts/build_run18_split.py (seed {args.seed}).\n")
    logger.info("Wrote train_run18.json, val_run18.json, README")


if __name__ == "__main__":
    main()
