"""Build the Atwood streak-window training dataset (self-contained, versioned).

Supersedes ``build_run18_split.py`` (run-scoped name + a coordinate bug). See
``agent_docs/dataset_naming.md`` for the naming convention this follows.

## The bug this fixes (v1)

The source annotations (``all_train_run17_merged.json``) use a *windowed*
convention: ``file_name`` is the FULL 6248x4176 frame, ``tile_origin`` is the
window's offset into that frame, ``width``/``height`` are the window size, and
each ``obb`` is **window-local** (cx/cy relative to the window, not the frame).

``build_run18_split.py`` copied ``file_name`` (full frame) + window-local obb
straight through WITHOUT materialising the crop, so the heatmap cacher
(``_cache_tiled``) tiled the full frame and placed every target ~``tile_origin``
(~1800 px) off the real streak — on empty sky. Proven: a line-profile of streak
brightness along the obb centerline was at chance for these windows, and jumped
from contrast -0.02 to +15.87 once ``tile_origin`` was added. Both ViT-S and
ViT-B trained to ~0.12 val_dice on the mislabelled data (Run 20 control).

The synthetic-short path already did it right (it crops the FITS to a window and
uses window-local obbs). This builder makes the REAL-window path do the same:
**every output image is a materialised crop with window-local obbs and
``tile_origin=[0,0]``**, so coords and pixels always agree.

## v2 changes (2026-06-14)

1. **Per-frame z-score normalisation**: mean/std computed over the full
   6248×4176 FITS frame (finite pixels only), then applied to each crop before
   saving.  The cacher must be called with ``--norm-mode none`` so it does not
   re-normalise.  This matches the eval pipeline, which normalises full frames
   before tiling, closing the train/eval distribution gap that caused high FP
   rates in v1 ViT-B.

2. **Corpus negative frames**: the 576 unannotated FITS frames in the merged
   corpus are used as pure negative-sky sources, replacing the per-annotated-
   frame random crops.  Neg tile fraction raised conservatively to ~0.42
   (was 0.38).  Going above 0.45 has caused training collapse in prior runs.

## v3 changes

3. **Per-annotated-frame background tiles**: for each training frame that has
   streak annotations, ``--bg-per-frame`` (default 3) additional 1800×1800
   background windows are sampled from regions that do NOT overlap any
   annotation (with a 400 px clearance margin).  This closes the remaining
   train/eval distribution gap: at inference the model tiles the full frame and
   sees mostly empty-sky tiles from frames that *do* contain streaks — exactly
   what v3 negatives now represent.  Corpus negatives remain but their count is
   reduced proportionally so total neg fraction stays ≈ 0.42.

## v5 changes

4. **Hard negative mining**: ``--hard-negs-json`` points to the JSON output of
   ``scripts/mine_hard_negatives.py``.  These are 400×400 tiles where the
   trained vits_window_v4 model fires with peak > 0.85 but no GT annotation
   exists — the exact FP patterns that inflate false positives.  Up to
   ``n_total_negs // 3`` hard negatives are added to the negative pool (after
   random shuffle and cap), with the remaining 2/3 split between corpus and
   frame-bg as before.

## Output (per agent_docs/dataset_naming.md)

Self-contained, content-named, versioned directories under TrainingData root:

    {root}/train_atwood_synth_window_v{N}/   real Atwood windows + synthetic short
    {root}/val_atwood_window_v{N}/           real Atwood windows (held-out frames)
        annotation.json   COCO; file_names are RELATIVE paths into tiles/
        tiles/*.npy       per-frame-zscore float32 crops (norm already applied)

v2+ crops are stored **already normalised** (per-frame zscore, 3-sigma clip →
float32, NOT uint8).  The cacher must use ``--norm-mode none``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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
NEG_WIN = 1800  # negative-sky crop size (>= 4 tiles at 400px)


def _band(L: float) -> str:
    if L < SHORT_MAX:
        return "short"
    if L < LONG_MIN:
        return "medium"
    return "long"


def _coco_ann(ann_id, img_id, obb):
    w, h, ang = obb["w"], obb["h"], math.radians(obb["angle_deg"])
    hx, hy = w / 2 * math.cos(ang), w / 2 * math.sin(ang)
    x1, y1, x2, y2 = obb["cx"] - hx, obb["cy"] - hy, obb["cx"] + hx, obb["cy"] + hy
    return {"id": ann_id, "image_id": img_id, "category_id": 1, "obb": obb,
            "bbox": [min(x1, x2), min(y1, y2), abs(x2 - x1) + h, abs(y2 - y1) + h],
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "streak_length_px": float(max(w, h)), "iscrowd": 0}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="/Volumes/External/TrainingData",
                   help="root for self-contained dataset directories")
    p.add_argument("--version", type=int, default=1, help="dataset version N (v{N})")
    p.add_argument("--source", default="data/annotations/all_train_run17_merged.json")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--n-synth-short", type=int, default=400)
    p.add_argument("--neg-frac", type=float, default=0.42,
                   help="target negative TILE fraction (v1: 0.38; v2: 0.42; do not exceed 0.45)")
    p.add_argument("--pos-tiles-per-window", type=float, default=3.0)
    p.add_argument("--neg-tiles-per-image", type=int, default=4,
                   help="must match the cacher's --neg-tiles-per-image")
    p.add_argument("--bg-per-frame", type=int, default=3,
                   help="(v3+) background windows sampled per annotated training frame")
    p.add_argument("--hard-negs-json", default="",
                   help="(v5+) path to hard_negatives_*.json from mine_hard_negatives.py")
    p.add_argument("--eval-frames-json", default="",
                   help="COCO JSON whose images should be excluded from training "
                        "(replaces the legacy val_atwood.json / test_atwood.json lookup "
                        "when those files are absent).  Pass val_balanced_v1_no_sattrains.json "
                        "or similar.")
    p.add_argument("--seed", type=int, default=18)
    p.add_argument("--limit", type=int, default=0,
                   help="smoke test: cap positive windows (0 = all)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rng = np.random.default_rng(args.seed)

    from astropy.io import fits

    root = Path(args.dataset_root)
    train_name = f"train_atwood_synth_window_v{args.version}"
    val_name = f"val_atwood_window_v{args.version}"
    train_dir, val_dir = root / train_name, root / val_name
    for d in (train_dir, val_dir):
        (d / "tiles").mkdir(parents=True, exist_ok=True)

    # Load each FITS frame on demand, apply per-frame zscore, discard after use.
    # Do NOT cache: 1600+ frames × 104 MB = 167 GB, causes OOM kill.
    # Per-frame zscore: mean/std over all finite pixels in the full frame,
    # clip to ±3σ, then (x - mean) / std → float32.  Crops saved already
    # normalised so the cacher must use --norm-mode none.
    def load_frame(fn: str) -> np.ndarray | None:
        try:
            with fits.open(fn, memmap=False) as h:
                raw = np.asarray(h[0].data, dtype=np.float32)
        except Exception as exc:
            logger.warning("could not load %s: %s", fn, exc)
            return None
        finite = raw[np.isfinite(raw)]
        if finite.size == 0:
            logger.warning("frame has no finite pixels: %s", fn)
            return None
        mean, std = float(finite.mean()), float(finite.std())
        if std < 1e-6:
            std = 1.0
        clipped = np.clip(raw, mean - 3.0 * std, mean + 3.0 * std)
        return ((clipped - mean) / std).astype(np.float32)

    eval_frames = set()
    if args.eval_frames_json:
        for im in json.load(open(args.eval_frames_json))["images"]:
            eval_frames.add(im["file_name"])
    else:
        for f in ["val_atwood", "test_atwood"]:
            p_legacy = Path(f"data/annotations/{f}.json")
            if p_legacy.exists():
                for im in json.load(open(p_legacy))["images"]:
                    eval_frames.add(im["file_name"])
    logger.info("eval_frames hold-out: %d files", len(eval_frames))

    merged = json.load(open(args.source))
    anns_by_img = defaultdict(list)
    for a in merged["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    # Collect positive real-FITS windows (non-eval, streaks >=50px). obb stays
    # WINDOW-LOCAL; tile_origin/width/height describe the crop to materialise.
    windows = []
    for im in merged["images"]:
        fn = im["file_name"]
        if not fn.lower().endswith(".fits") or fn in eval_frames:
            continue
        keep = [a for a in anns_by_img[im["id"]]
                if max(a["obb"]["w"], a["obb"]["h"]) >= MIN_LEN]
        if not keep:
            continue
        windows.append({"frame": fn, "x0": int(im.get("tile_origin", [0, 0])[0]),
                        "y0": int(im.get("tile_origin", [0, 0])[1]),
                        "w": int(im["width"]), "h": int(im["height"]),
                        "anns": [{"obb": a["obb"], "len": max(a["obb"]["w"], a["obb"]["h"])}
                                 for a in keep]})
    if args.limit:
        windows = windows[:args.limit]
    logger.info("Positive real-FITS windows: %d", len(windows))

    # Split by source frame (no leakage), stratified by dominant band.
    by_frame = defaultdict(list)
    for w in windows:
        by_frame[w["frame"]].append(w)
    frames = list(by_frame)
    frame_band = {fr: _band(max(max(a["len"] for a in w["anns"]) for w in by_frame[fr]))
                  for fr in frames}
    val_frames = set()
    for band in ("short", "medium", "long"):
        fl = [fr for fr in frames if frame_band[fr] == band]
        rng.shuffle(fl)
        val_frames.update(fl[:max(1, int(len(fl) * args.val_frac))])

    train_w = [w for w in windows if w["frame"] not in val_frames]
    val_w = [w for w in windows if w["frame"] in val_frames]

    # ---- Synthetic short streaks rendered ON real-FITS crops (train only) ----
    synth_specs = []  # {frame,x0,y0,w,h, render:(cx,cy,length,angle)}
    src = list(train_w)
    rng.shuffle(src)
    for w in src:
        if len(synth_specs) >= args.n_synth_short:
            break
        if w["w"] < 200 or w["h"] < 200:
            continue
        length = float(rng.uniform(60, 380))
        cx = float(rng.uniform(length / 2 + 20, w["w"] - length / 2 - 20))
        cy = float(rng.uniform(length / 2 + 20, w["h"] - length / 2 - 20))
        synth_specs.append({"frame": w["frame"], "x0": w["x0"], "y0": w["y0"],
                            "w": w["w"], "h": w["h"],
                            "render": (cx, cy, length, float(rng.uniform(0, 180)))})
    logger.info("Synthetic short windows planned: %d", len(synth_specs))

    # ---- Corpus frames with no annotations (pure negative-sky sources) ----
    # These 576 unannotated FITS frames give the model varied empty-sky tiles
    # across the full frame area, independent of the positive-window regions.
    annotated_frames = {w["frame"] for w in windows} | eval_frames
    corpus_neg_frames = [
        im["file_name"] for im in merged["images"]
        if im["file_name"].lower().endswith(".fits")
        and im["file_name"] not in annotated_frames
    ]
    rng.shuffle(corpus_neg_frames)
    logger.info("Unannotated corpus frames available for negatives: %d", len(corpus_neg_frames))

    # ---- Negative-sky windows (no annotations) for ~neg_frac neg tiles ----
    def neg_count(pos_windows):
        pos_tiles = len(pos_windows) * args.pos_tiles_per_window
        neg_tiles = args.neg_frac / (1 - args.neg_frac) * pos_tiles
        return int(round(neg_tiles / max(1, args.neg_tiles_per_image)))

    # (v3+) Sample background windows from annotated frames, avoiding all annotation
    # regions.  Each window is NEG_WIN×NEG_WIN with a BG_MARGIN clearance from
    # every annotation window bbox; produces neg_tiles_per_image tiles when cached.
    _BG_MARGIN = 400  # px clearance beyond each annotation window
    def plan_frame_bg(pos_windows):
        if args.version < 3 or args.bg_per_frame <= 0:
            return []
        by_frame = defaultdict(list)
        for w in pos_windows:
            by_frame[w["frame"]].append(w)
        out = []
        for frame, wins in by_frame.items():
            excl = [(max(0, w["x0"] - _BG_MARGIN), max(0, w["y0"] - _BG_MARGIN),
                     min(FRAME_W, w["x0"] + w["w"] + _BG_MARGIN),
                     min(FRAME_H, w["y0"] + w["h"] + _BG_MARGIN))
                    for w in wins]
            sampled, attempts = 0, 0
            while sampled < args.bg_per_frame and attempts < args.bg_per_frame * 40:
                attempts += 1
                nx = int(rng.integers(0, FRAME_W - NEG_WIN))
                ny = int(rng.integers(0, FRAME_H - NEG_WIN))
                nx1, ny1 = nx + NEG_WIN, ny + NEG_WIN
                if not any(nx < rx1 and nx1 > rx0 and ny < ry1 and ny1 > ry0
                           for rx0, ry0, rx1, ry1 in excl):
                    out.append({"frame": frame, "x0": nx, "y0": ny,
                                "w": NEG_WIN, "h": NEG_WIN})
                    sampled += 1
        return out

    def plan_negs(pos_windows, n_corpus):
        # Draw from corpus_neg_frames first (diverse sky); fall back to sampling
        # from positive-window frames if corpus is exhausted.
        out = []
        corpus_pool = list(corpus_neg_frames)
        fallback_pool = [w["frame"] for w in pos_windows]
        for i in range(n_corpus):
            if corpus_pool:
                frame = corpus_pool[i % len(corpus_pool)]
            elif fallback_pool:
                frame = fallback_pool[i % len(fallback_pool)]
            else:
                break
            nx = int(rng.integers(0, FRAME_W - NEG_WIN))
            ny = int(rng.integers(0, FRAME_H - NEG_WIN))
            out.append({"frame": frame, "x0": nx, "y0": ny,
                        "w": NEG_WIN, "h": NEG_WIN})
        return out

    # v5+: load mined hard negatives (400×400 FP-prone tiles from vits_window_v4).
    hard_negs_pool: list[dict] = []
    if args.hard_negs_json and args.version >= 5:
        hn_path = Path(args.hard_negs_json)
        if hn_path.exists():
            hard_negs_pool = json.loads(hn_path.read_text())
            rng.shuffle(hard_negs_pool)
            logger.info("Hard negatives pool: %d tiles from %s", len(hard_negs_pool), hn_path)
        else:
            logger.warning("--hard-negs-json path not found: %s (ignored)", hn_path)

    # v3: frame_bg can exceed the neg budget if there are many annotated frames
    # (e.g. 879 frames × 3 bg_per_frame = 2637 but budget is ~561 windows).
    # Cap frame_bg at half the total neg budget so corpus still contributes;
    # shuffle first so the cap is a random subset rather than the first N frames.
    # v5/v7+: reserve up to 1/3 of the neg budget for hard negatives.
    # v6 experiment raised this to 1/2 — caused precision regression; reverted for v7+.
    hard_neg_frac = 2 if args.version == 6 else 3  # denominator: 1/2 (v6 only) vs 1/3
    train_frame_bg_all = plan_frame_bg(train_w)
    n_train_total = neg_count(train_w)
    if hard_negs_pool:
        n_hard_train = min(len(hard_negs_pool), n_train_total // hard_neg_frac)
        train_hard_negs = list(hard_negs_pool[:n_hard_train])
        n_remaining = n_train_total - n_hard_train
    else:
        train_hard_negs = []
        n_remaining = n_train_total
    n_frame_bg_cap = n_remaining // 2
    rng.shuffle(train_frame_bg_all)
    train_frame_bg = train_frame_bg_all[:n_frame_bg_cap]
    n_train_corpus = n_remaining - len(train_frame_bg)
    train_negs = plan_negs(train_w, n_train_corpus) + train_frame_bg + train_hard_negs

    val_frame_bg_all = plan_frame_bg(val_w)
    n_val_total = neg_count(val_w)
    n_val_frame_bg_cap = n_val_total // 2
    rng.shuffle(val_frame_bg_all)
    val_frame_bg = val_frame_bg_all[:n_val_frame_bg_cap]
    n_val_corpus = n_val_total - len(val_frame_bg)
    val_negs = plan_negs(val_w, n_val_corpus) + val_frame_bg

    logger.info("Negative-sky windows: train=%d (%d corpus + %d frame-bg + %d hard-neg)  val=%d (%d corpus + %d frame-bg)",
                len(train_negs), n_train_corpus, len(train_frame_bg), len(train_hard_negs),
                len(val_negs), n_val_corpus, len(val_frame_bg))

    # ---- Materialise crops -> NPY, emit COCO with relative file_names ----
    # Group all jobs by source frame so each 104 MB FITS is loaded exactly once,
    # all its crops extracted and saved, then the array discarded before the next frame.
    def build(dir_, pos_windows, synth, negs):
        from collections import defaultdict as _dd

        # Each job: dict with keys frame, x0, y0, w, h, and type-specific extras.
        # Order within a frame group doesn't matter — iid is assigned at write time.
        jobs_by_frame = _dd(list)
        for w in pos_windows:
            jobs_by_frame[w["frame"]].append({"kind": "pos", **w})
        for s in synth:
            jobs_by_frame[s["frame"]].append({"kind": "synth", **s})
        for w in negs:
            jobs_by_frame[w["frame"]].append({"kind": "neg", **w})

        images, anns = [], []
        iid, aid = 1, 1
        n_frames = len(jobs_by_frame)
        for fi, (frame, jobs) in enumerate(jobs_by_frame.items(), 1):
            data = load_frame(frame)
            if data is None:
                logger.warning("skipping frame (load failed): %s", frame)
                continue
            if fi % 50 == 0 or fi == n_frames:
                logger.info("frame %d/%d  tiles so far: %d", fi, n_frames, iid - 1)
            for job in jobs:
                x0, y0, w, h = job["x0"], job["y0"], job["w"], job["h"]
                crop = data[y0:y0 + h, x0:x0 + w].astype(np.float32).copy()
                if crop.shape[0] < 64 or crop.shape[1] < 64:
                    continue
                if job["kind"] == "synth":
                    cx, cy, length, angle = job["render"]
                    crop = _render_streak(crop, cx, cy, length, angle,
                                         float(rng.uniform(5, 10)), 1.5, rng).astype(np.float32)
                rel = f"tiles/{iid}.npy"
                np.save(dir_ / rel, crop)
                rec = {"id": iid, "file_name": rel, "width": w, "height": h,
                       "tile_origin": [0, 0], "source_frame": frame,
                       "source_origin": [x0, y0], "synthetic": job["kind"] == "synth"}
                images.append(rec)
                if job["kind"] == "pos":
                    for a in job["anns"]:
                        anns.append(_coco_ann(aid, iid, a["obb"])); aid += 1
                elif job["kind"] == "synth":
                    cx, cy, length, angle = job["render"]
                    anns.append(_coco_ann(aid, iid, {"cx": cx, "cy": cy, "w": length,
                                                     "h": 8.0, "angle_deg": angle})); aid += 1
                iid += 1
            del data  # release 104 MB before loading the next frame
        return images, anns

    tr_imgs, tr_anns = build(train_dir, train_w, synth_specs, train_negs)
    va_imgs, va_anns = build(val_dir, val_w, [], val_negs)

    for dir_, name, imgs, anns in [(train_dir, train_name, tr_imgs, tr_anns),
                                   (val_dir, val_name, va_imgs, va_anns)]:
        prov = {"builder": "scripts/build_atwood_window_dataset.py",
                "source": args.source, "seed": args.seed, "version": args.version,
                "norm": "per-frame zscore (3-sigma clip) applied at build time; cacher must use --norm-mode none",
                "obb_frame": "window-local (crop materialised; tile_origin=[0,0])"}
        if args.hard_negs_json:
            prov["hard_negs_json"] = args.hard_negs_json
            prov["n_hard_negs_train"] = len(train_hard_negs)
        out = {"images": imgs, "annotations": anns,
               "categories": [{"id": 1, "name": "streak"}],
               "provenance": prov}
        (dir_ / "annotation.json").write_text(json.dumps(out))
        b = Counter(_band(a["streak_length_px"]) for a in anns)
        ann_ids = {a["image_id"] for a in anns}
        nneg = sum(1 for im in imgs if im["id"] not in ann_ids)
        logger.info("%s: %d images (%d neg), streaks short=%d med=%d long=%d  -> %s",
                    name, len(imgs), nneg, b["short"], b["medium"], b["long"],
                    dir_ / "annotation.json")


if __name__ == "__main__":
    main()
