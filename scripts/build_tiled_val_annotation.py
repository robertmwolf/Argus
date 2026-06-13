"""Build a sparse pre-tiled evaluation annotation from a full-frame annotation.

Motivation
----------
Full-frame tiled inference evaluates every 400-px tile in each 6248×4176 image
(~176 tiles/frame, 42 k tiles total for the val set), ~97 % of which are empty
sky.  This script builds a compact, deterministic eval set containing only:

  * **Positive tiles** — the tile whose centre contains the GT annotation centre
    (one tile per annotation, no double-counting across GT boundaries).
  * **Context tiles** — the N neighbouring tiles closest to the annotation
    centre, ordered by proximity.  These are the tiles where bleed-over false
    positives first appear (the streak is heading in that direction) and where
    the background immediately adjacent to a real detection lives.

There is no random sampling; the same annotation file always produces the same
tile set.

Typical counts (val_run17_fits.json, 240 images, 1 156 GT annotations):

  --context 0 →  ~1 344 tiles  (  0 % neg)
  --context 2 →  ~3 744 tiles  ( 64 % neg)   ← good for fast iteration
  --context 4 →  ~5 424 tiles  ( 75 % neg)   ← default; catches bleed FPs
  --context 8 →  ~6 912 tiles  ( 81 % neg)   ← near-full ring
  full tiling →  42 240 tiles  ( 97 % neg)

Output
------
``<output-dir>/annotation.json``
    COCO annotation; each image entry is a 400-px tile.  ``file_name`` is a
    relative path into ``tiles/``.  Bounding boxes and OBBs are in tile-local
    coordinates.

``<output-dir>/tiles/``
    Single-channel float32 NPY files (raw pixel values, no normalisation baked
    in).  Evaluate with ``--norm-mode zscore`` to match Run 15/17 training.

Usage
-----
    python scripts/build_tiled_val_annotation.py \\
        --annotations /Volumes/External/TrainingData/annotations/val_run17_fits.json \\
        --output-dir  /Volumes/External/TrainingData/val_tiled_eval_400 \\
        --tile-size 400 \\
        --context 4

Evaluate the result with (no --tiled flag — each file is already a 400-px crop):

    ARGUS_NORM=zscore \\
    python scripts/evaluate_dinov3_heatmap.py \\
        --annotations <output-dir>/annotation.json \\
        --checkpoint  weights/run17_vitb/best.pt \\
        --output      results/run17_vitb/fast_eval/metrics.json \\
        --norm-mode   zscore
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inference.fits_loader import FITSLoader
from inference.tiled_pipeline import tile_image

logger = logging.getLogger(__name__)


# ── Annotation helpers ────────────────────────────────────────────────────────

def _ann_center(ann: dict) -> tuple[float, float]:
    obb = ann.get("obb")
    if obb:
        cx = float(obb["cx"] if isinstance(obb, dict) else obb[0])
        cy = float(obb["cy"] if isinstance(obb, dict) else obb[1])
    else:
        b = ann.get("bbox", [0, 0, 0, 0])
        cx, cy = float(b[0]) + float(b[2]) / 2.0, float(b[1]) + float(b[3]) / 2.0
    return cx, cy


def _center_in_tile(cx: float, cy: float, x0: int, y0: int, ts: int) -> bool:
    return x0 <= cx < x0 + ts and y0 <= cy < y0 + ts


def _tile_local_ann(ann: dict, x0: int, y0: int) -> dict:
    """Return copy of ann with OBB/bbox coordinates shifted to tile-local origin."""
    local = dict(ann)
    obb = local.get("obb")
    if obb:
        if isinstance(obb, dict):
            local["obb"] = {**obb,
                            "cx": float(obb["cx"]) - x0,
                            "cy": float(obb["cy"]) - y0}
        else:
            lst = list(obb)
            lst[0] = float(lst[0]) - x0
            lst[1] = float(lst[1]) - y0
            local["obb"] = lst
    bbox = local.get("bbox")
    if bbox:
        local["bbox"] = [float(bbox[0]) - x0, float(bbox[1]) - y0,
                         float(bbox[2]), float(bbox[3])]
    return local


def _closest_neighbors(
    x0: int,
    y0: int,
    cx: float,
    cy: float,
    valid_origins: set[tuple[int, int]],
    ts: int,
    n: int,
) -> list[tuple[int, int]]:
    """Return up to N grid-neighbours of tile (x0, y0) sorted by proximity to (cx, cy).

    Proximity is measured from the annotation centre to the *centre of the
    neighbouring tile*, so the tiles the streak is heading toward (in the
    direction of its nearest edge) naturally rank first.
    """
    tile_cx, tile_cy = x0 + ts / 2.0, y0 + ts / 2.0
    candidates: list[tuple[float, tuple[int, int]]] = []
    for dx in (-ts, 0, ts):
        for dy in (-ts, 0, ts):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x0 + dx, y0 + dy
            if (nx, ny) not in valid_origins:
                continue
            # neighbour tile centre
            ncx, ncy = nx + ts / 2.0, ny + ts / 2.0
            dist = math.hypot(cx - ncx, cy - ncy)
            candidates.append((dist, (nx, ny)))
    candidates.sort()
    return [origin for _, origin in candidates[:n]]


# ── Image loading ─────────────────────────────────────────────────────────────

def _load_raw_float32(path: Path, loader: FITSLoader) -> np.ndarray | None:
    suffix = path.suffix.lower()
    try:
        if suffix in {".fits", ".fit", ".fts"}:
            result = loader.load(path)
            raw = result.get("raw_float32")
            if raw is not None:
                return raw.astype(np.float32)
            arr = np.asarray(result["array"], dtype=np.float32)
            return arr[..., 0] if arr.ndim == 3 else arr
        if suffix == ".npy":
            raw = np.load(str(path))
            if raw.ndim == 3:
                raw = raw[..., 0]
            return raw.astype(np.float32)
        from PIL import Image
        with Image.open(path) as im:
            return np.asarray(im.convert("L"), dtype=np.float32)
    except Exception as exc:
        logger.warning("Could not load %s: %s", path, exc)
        return None


def _resolve_path(fname: str, ann_dir: Path) -> Path:
    p = Path(fname)
    if p.is_absolute() and p.exists():
        return p
    for base in (ann_dir, ann_dir.parent, Path(".")):
        candidate = (base / fname).resolve()
        if candidate.exists():
            return candidate
    return p


# ── Core builder ─────────────────────────────────────────────────────────────

def build_tiled_eval_annotation(
    source_ann: dict[str, Any],
    output_dir: Path,
    tile_size: int = 400,
    context_tiles: int = 4,
    ann_dir: Path | None = None,
) -> dict[str, Any]:
    """Create a sparse tiled annotation from full-frame COCO data.

    Args:
        source_ann:     Parsed COCO annotation dict (full-frame images).
        output_dir:     Directory to write tile NPY files and annotation.json.
        tile_size:      Native tile edge length in pixels.
        context_tiles:  Number of neighbouring tiles to include per positive
                        tile, chosen by proximity to the annotation centre.
        ann_dir:        Directory that contains the source annotation file
                        (used for relative path resolution).

    Returns:
        COCO annotation dict for the tiled dataset.
    """
    loader = FITSLoader()
    tiles_dir = output_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    if ann_dir is None:
        ann_dir = Path(".")

    images_meta = source_ann.get("images", [])
    id_to_anns: dict[int, list[dict]] = {}
    for a in source_ann.get("annotations", []):
        id_to_anns.setdefault(int(a["image_id"]), []).append(a)

    out_images: list[dict] = []
    out_anns: list[dict] = []
    ann_id = 1
    n_images_processed = 0
    n_tiles_total = 0
    n_tiles_pos = 0

    for img_meta in images_meta:
        img_id = int(img_meta["id"])
        img_path = _resolve_path(str(img_meta["file_name"]), ann_dir)
        raw = _load_raw_float32(img_path, loader)
        if raw is None:
            logger.warning("Skipping unloadable image %s", img_meta["file_name"])
            continue

        orig_anns = id_to_anns.get(img_id, [])
        all_tiles = list(tile_image(raw, tile_size, 0.0))
        valid_origins: set[tuple[int, int]] = {(x0, y0) for _, x0, y0 in all_tiles}

        # ── Select tiles ──────────────────────────────────────────────────────
        # Maps (x0, y0) → list of matching GT annotations (empty for context tiles)
        selected: dict[tuple[int, int], list[dict]] = {}

        for _, x0, y0 in all_tiles:
            matching = [a for a in orig_anns
                        if _center_in_tile(*_ann_center(a), x0, y0, tile_size)]
            if not matching:
                continue
            # Positive tile: keep it and add context neighbours
            selected.setdefault((x0, y0), []).extend(matching)
            for a in matching:
                cx, cy = _ann_center(a)
                for nb in _closest_neighbors(x0, y0, cx, cy, valid_origins,
                                             tile_size, context_tiles):
                    if nb not in selected:
                        selected[nb] = []  # context tile — no GT annotations

        # ── Write tiles ───────────────────────────────────────────────────────
        tile_arr_map = {(x0, y0): tile_arr for tile_arr, x0, y0 in all_tiles}
        img_pos = 0
        for tile_seq, ((x0, y0), tile_anns) in enumerate(sorted(selected.items())):
            tile_arr = tile_arr_map[(x0, y0)]
            tile_id = img_id * 10000 + tile_seq
            npy_name = f"{img_id:06d}_t{tile_seq:04d}.npy"
            np.save(str(tiles_dir / npy_name), tile_arr.astype(np.float32))

            out_images.append({
                "id": tile_id,
                "file_name": str(Path("tiles") / npy_name),
                "width": tile_arr.shape[1],
                "height": tile_arr.shape[0],
                "orig_image_id": img_id,
                "tile_origin": [x0, y0],
                "tile_size": tile_size,
            })

            for src_ann in tile_anns:
                local = _tile_local_ann(src_ann, x0, y0)
                local["id"] = ann_id
                local["image_id"] = tile_id
                ann_id += 1
                out_anns.append(local)

            if tile_anns:
                img_pos += 1

        n_images_processed += 1
        n_tiles_total += len(selected)
        n_tiles_pos += img_pos
        logger.info(
            "img_id=%d  total=%d  pos=%d  neg=%d  anns=%d",
            img_id, len(selected), img_pos, len(selected) - img_pos, len(orig_anns),
        )

    n_neg = n_tiles_total - n_tiles_pos
    actual_neg_ratio = n_neg / n_tiles_total if n_tiles_total else 0.0
    return {
        "info": {
            **source_ann.get("info", {}),
            "tiled_eval": True,
            "tile_size": tile_size,
            "context_tiles": context_tiles,
            "n_pos_tiles": n_tiles_pos,
            "n_neg_tiles": n_neg,
            "neg_ratio_actual": actual_neg_ratio,
            "n_images": n_images_processed,
        },
        "categories": source_ann.get("categories", []),
        "images": out_images,
        "annotations": out_anns,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", required=True,
                        help="Full-frame COCO annotation JSON (e.g. val_run17_fits.json).")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write tiles/ and annotation.json.")
    parser.add_argument("--tile-size", type=int, default=400,
                        help="Native tile edge length in pixels (default: 400).")
    parser.add_argument("--context", type=int, default=4, dest="context_tiles",
                        help="Neighbouring tiles to include per positive tile, "
                             "sorted by proximity to the annotation centre "
                             "(default: 4).  0 = positive tiles only.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    ann_path = Path(args.annotations)
    source_ann = json.loads(ann_path.read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_coco = build_tiled_eval_annotation(
        source_ann=source_ann,
        output_dir=output_dir,
        tile_size=args.tile_size,
        context_tiles=args.context_tiles,
        ann_dir=ann_path.parent,
    )

    out_path = output_dir / "annotation.json"
    out_path.write_text(json.dumps(out_coco, indent=2))

    info = out_coco["info"]
    n_pos  = info["n_pos_tiles"]
    n_neg  = info["n_neg_tiles"]
    total  = n_pos + n_neg
    ratio  = 100.0 * info["neg_ratio_actual"]

    print(
        f"\nSummary\n"
        f"  Output:          {out_path}\n"
        f"  Images:          {info['n_images']}\n"
        f"  Total tiles:     {total}  ({n_pos} positive, {n_neg} context/neg)\n"
        f"  Neg ratio:       {ratio:.1f}%\n"
        f"  Annotations:     {len(out_coco['annotations'])}\n"
        f"\n"
        f"Evaluate with (no --tiled flag; each file is a {args.tile_size}-px crop):\n"
        f"  ARGUS_NORM=zscore \\\n"
        f"  python scripts/evaluate_dinov3_heatmap.py \\\n"
        f"    --annotations {out_path} \\\n"
        f"    --checkpoint  <path/to/best.pt> \\\n"
        f"    --output      <results_dir>/metrics.json \\\n"
        f"    --norm-mode   zscore\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
