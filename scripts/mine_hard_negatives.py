#!/usr/bin/env python3
"""Mine hard negative tiles from training frames using a trained checkpoint.

For each FITS training frame that has streak annotations, tile the full frame at
400px / 0% overlap, run the ViT-S heatmap model, and collect tiles where:
  (a) peak heatmap probability exceeds --peak-threshold, AND
  (b) the tile does not overlap any ground-truth annotation window
      (with --margin px clearance).

These are the specific false-positive patterns that fool the current model.
Adding them as explicit negatives in v5 training should improve precision.

Output: JSON list of {"frame": path, "x0": int, "y0": int, "w": 400, "h": 400}.

Usage:
    python scripts/mine_hard_negatives.py \\
        --checkpoint weights/vits_window_v4/best.pt \\
        --annotations data/annotations/all_train_run17_merged.json \\
        --output data/annotations/hard_negatives_vits_window_v4.json \\
        [--peak-threshold 0.85] [--margin 400] [--tile-size 400] [--limit 10]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logger = logging.getLogger(__name__)

FRAME_W, FRAME_H = 6248, 4176


def _build_excl_zones(annotation_windows: list[dict], margin: int) -> list[tuple[int, int, int, int]]:
    """Return (x0, y0, x1, y1) exclusion zones expanded by margin."""
    zones = []
    for win in annotation_windows:
        x0, y0 = win["x0"], win["y0"]
        w, h = win["w"], win["h"]
        zones.append((
            max(0, x0 - margin),
            max(0, y0 - margin),
            min(FRAME_W, x0 + w + margin),
            min(FRAME_H, y0 + h + margin),
        ))
    return zones


def _overlaps_any(tx: int, ty: int, ts: int, zones: list[tuple]) -> bool:
    tx1, ty1 = tx + ts, ty + ts
    return any(tx < x1 and tx1 > x0 and ty < y1 and ty1 > y0 for x0, y0, x1, y1 in zones)


def _random_tile_positions(
    frame_h: int, frame_w: int, tile_size: int, n: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    """Sample n random tile top-left positions within frame bounds."""
    max_x = max(0, frame_w - tile_size)
    max_y = max(0, frame_h - tile_size)
    xs = rng.integers(0, max_x + 1, size=n)
    ys = rng.integers(0, max_y + 1, size=n)
    return list(zip(xs.tolist(), ys.tolist()))


def mine_frame(
    frame_path: str,
    excl_zones: list[tuple],
    model: object,
    image_size: int,
    device: object,
    peak_threshold: float,
    tile_size: int,
    random_tiles: int = 0,
    rng: "np.random.Generator | None" = None,
) -> list[dict]:
    """Run tiled inference on one FITS frame; return hard negative tile specs.

    If random_tiles > 0, randomly sample that many tile positions instead of
    exhaustively tiling the frame (faster, broader spatial coverage).
    """
    from astropy.io import fits as _fits

    from inference.convnext_heatmap_detector import _run_single_tile_probs
    from inference.fits_loader import _normalise_zscore
    from inference.tiled_pipeline import tile_image

    # Load full FITS frame and apply per-frame zscore (same as eval pipeline).
    try:
        with _fits.open(frame_path, memmap=False) as h:
            raw = np.asarray(h[0].data, dtype=np.float32)
    except Exception as exc:
        logger.warning("could not load %s: %s", frame_path, exc)
        return []

    grey = _normalise_zscore(raw)            # uint8 (H, W)
    arr  = np.stack([grey, grey, grey], axis=-1)   # uint8 (H, W, 3)
    h, w = arr.shape[:2]

    if random_tiles > 0 and rng is not None:
        positions = _random_tile_positions(h, w, tile_size, random_tiles, rng)
        tile_iter = ((arr[ty:ty + tile_size, tx:tx + tile_size], tx, ty)
                     for tx, ty in positions)
    else:
        tile_iter = tile_image(arr, tile_size, 0.0)

    hard_negs = []
    for tile, tx, ty in tile_iter:
        if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
            continue
        heat, _, _, _ = _run_single_tile_probs(tile, model, image_size, device)
        if float(heat.max()) < peak_threshold:
            continue
        if not _overlaps_any(tx, ty, tile_size, excl_zones):
            hard_negs.append({"frame": frame_path, "x0": int(tx), "y0": int(ty),
                               "w": tile_size, "h": tile_size})

    return hard_negs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="weights/vits_window_v4/best.pt",
                    help="path to vits_window_v4 best.pt")
    ap.add_argument("--annotations", default="data/annotations/all_train_run17_merged.json",
                    help="merged training annotation JSON")
    ap.add_argument("--output", required=True,
                    help="output JSON file for mined hard negative tile specs")
    ap.add_argument("--peak-threshold", type=float, default=0.85,
                    help="min peak heatmap probability to count as a model firing (default 0.85)")
    ap.add_argument("--margin", type=int, default=400,
                    help="exclusion margin around GT annotation windows in px (default 400)")
    ap.add_argument("--tile-size", type=int, default=400,
                    help="mining tile size in source px (must match training tile size, default 400)")
    ap.add_argument("--max-hard-negs", type=int, default=0,
                    help="stop mining once this many hard negatives are collected (0 = all frames). "
                         "Set to ~3× the negative budget to avoid mining 30× more than needed.")
    ap.add_argument("--random-tiles", type=int, default=0,
                    help="sample this many random tile positions per frame instead of exhaustive "
                         "tiling (0 = exhaustive). Use ~30 for 5× speedup with broader coverage.")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke-test: cap number of frames processed (0 = all)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # --- Load model ---
    from inference.vits_heatmap_detector import _load_model
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)
    logger.info("Loading model from %s", ckpt_path)
    model, image_size, device, _ = _load_model(ckpt_path)
    logger.info("Model loaded (image_size=%d, device=%s)", image_size, device)

    # --- Load annotations ---
    merged = json.loads(Path(args.annotations).read_text())

    # Eval frames: exclude these from mining (no leakage)
    eval_frames: set[str] = set()
    for fname in ["data/annotations/val_atwood.json", "data/annotations/test_atwood.json"]:
        p = _REPO / fname
        if p.exists():
            for im in json.loads(p.read_text())["images"]:
                eval_frames.add(im["file_name"])
    logger.info("Eval frames excluded: %d", len(eval_frames))

    # Group annotation windows by source FITS frame.
    anns_by_img = defaultdict(list)
    for a in merged["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    windows_by_frame: dict[str, list[dict]] = defaultdict(list)
    for im in merged["images"]:
        fn = im["file_name"]
        if not fn.lower().endswith((".fits", ".fit", ".fts")):
            continue
        if fn in eval_frames:
            continue
        if not anns_by_img.get(im["id"]):
            continue  # mine annotated frames only
        windows_by_frame[fn].append({
            "x0": int(im.get("tile_origin", [0, 0])[0]),
            "y0": int(im.get("tile_origin", [0, 0])[1]),
            "w":  int(im["width"]),
            "h":  int(im["height"]),
        })

    import random as _random
    all_frames = sorted(windows_by_frame)
    _random.seed(42)
    _random.shuffle(all_frames)
    if args.limit:
        all_frames = all_frames[:args.limit]
    logger.info("Frames to mine: %d (of %d annotated training frames) — randomised order",
                len(all_frames), len(windows_by_frame))

    # Per-frame RNG for reproducible random tile sampling (seeded from global seed + frame index).
    base_rng = np.random.default_rng(42)

    # --- Mine ---
    all_hard_negs: list[dict] = []
    for fi, frame in enumerate(all_frames, 1):
        frame_rng = np.random.default_rng(base_rng.integers(2**31) + fi)
        excl_zones = _build_excl_zones(windows_by_frame[frame], args.margin)
        negs = mine_frame(frame, excl_zones, model, image_size, device,
                          args.peak_threshold, args.tile_size,
                          random_tiles=args.random_tiles, rng=frame_rng)
        all_hard_negs.extend(negs)
        if fi % 20 == 0 or fi == len(all_frames):
            logger.info("frame %d/%d  hard negs so far: %d", fi, len(all_frames), len(all_hard_negs))
        if args.max_hard_negs and len(all_hard_negs) >= args.max_hard_negs:
            logger.info("Reached --max-hard-negs=%d after %d frames — stopping early",
                        args.max_hard_negs, fi)
            break

    logger.info("Total hard negatives mined: %d from %d frames", len(all_hard_negs), len(all_frames))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_hard_negs, indent=2))
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
