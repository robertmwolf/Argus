"""Build frigate_cluster2_tiled_110.json — cluster-2 frigate annotations tiled at 110px.

Cluster 2 definition: annotations with diagonal >= 35px AND aspect_ratio >= 2.0.
These are the ~13% of frigate annotations that represent actual linear streak
morphology (35-66px native), as opposed to the 86% near-circular blobs (<25px, AR~1).

Tiling strategy:
  - native_tile_size = 110px  →  3.64x magnification at 400px model input
  - A 35-66px streak at native appears as 127-240px at model input
  - Overlap = 0 (one tile per streak, centred on annotation)
  - For each cluster-2 streak: one positive tile centred on the streak
  - Per frame: add negative tiles from regions with no annotation
  - Tile coordinates stored as virtual paths: <stem>__tx{x}_ty{y}_ts110.png
    resolved at load time by training/transforms.py:LoadFITSFromFile

Output: data/annotations/frigate_cluster2_tiled_110.json

Usage:
    python scripts/build_frigate_cluster2.py
    python scripts/build_frigate_cluster2.py --neg-per-frame 3 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ANN_SRC   = _REPO_ROOT / "data/annotations/frigate_streaks.json"
_OUT       = _REPO_ROOT / "data/annotations/frigate_cluster2_tiled_110.json"

TILE_SIZE  = 110
MIN_DIAG   = 35.0
MIN_AR     = 2.0
_CATEGORIES = [{"id": 1, "name": "streak", "supercategory": "satellite"}]


def _diag(bbox: list) -> float:
    _, _, w, h = bbox
    return math.sqrt(w * w + h * h)


def _ar(bbox: list) -> float:
    _, _, w, h = bbox
    return max(w, h) / min(w, h) if min(w, h) > 0 else 0.0


def _tile_path(parent_path: str, x0: int, y0: int, ts: int) -> str:
    """Return virtual tile path: <parent_stem>__tx{x0}_ty{y0}_ts{ts}<ext>."""
    p = Path(parent_path)
    return str(p.parent / f"{p.stem}__tx{x0}_ty{y0}_ts{ts}{p.suffix}")


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def build(neg_per_frame: int, seed: int) -> dict:
    rng = random.Random(seed)
    src = json.loads(_ANN_SRC.read_text())
    img_map = {i["id"]: i for i in src["images"]}

    # Group cluster-2 annotations by image
    cluster2_by_img: dict[int, list] = {}
    for ann in src["annotations"]:
        if _diag(ann["bbox"]) >= MIN_DIAG and _ar(ann["bbox"]) >= MIN_AR:
            cluster2_by_img.setdefault(ann["image_id"], []).append(ann)

    out_images: list[dict] = []
    out_annotations: list[dict] = []
    img_id_counter = 1
    ann_id_counter = 1

    for src_img_id, anns in cluster2_by_img.items():
        src_img = img_map[src_img_id]
        img_w   = src_img["width"]
        img_h   = src_img["height"]
        src_path = src_img["file_name"]

        # ── positive tiles ────────────────────────────────────────────────
        for ann in anns:
            x, y, w, h = ann["bbox"]
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            x0 = _clamp(cx - TILE_SIZE // 2, 0, img_w - TILE_SIZE)
            y0 = _clamp(cy - TILE_SIZE // 2, 0, img_h - TILE_SIZE)

            tile_path = _tile_path(src_path, x0, y0, TILE_SIZE)

            # Remap bbox to tile coordinate space
            bx = x - x0
            by = y - y0
            # Clip to tile bounds
            bx2 = min(bx + w, TILE_SIZE)
            by2 = min(by + h, TILE_SIZE)
            bx  = max(bx, 0)
            by  = max(by, 0)
            bw  = bx2 - bx
            bh  = by2 - by
            if bw <= 0 or bh <= 0:
                continue

            tile_img = {
                "id": img_id_counter,
                "file_name": tile_path,
                "width": TILE_SIZE,
                "height": TILE_SIZE,
            }
            out_images.append(tile_img)

            out_annotations.append({
                "id": ann_id_counter,
                "image_id": img_id_counter,
                "category_id": 1,
                "bbox": [bx, by, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            })
            img_id_counter += 1
            ann_id_counter += 1

        # ── negative tiles ────────────────────────────────────────────────
        # Collect occupied regions to avoid placing negative tiles over streaks
        occupied = [(int(a["bbox"][0]), int(a["bbox"][1]),
                     int(a["bbox"][0] + a["bbox"][2]),
                     int(a["bbox"][1] + a["bbox"][3]))
                    for a in src["annotations"] if a["image_id"] == src_img_id]

        added_neg = 0
        attempts  = 0
        while added_neg < neg_per_frame and attempts < 200:
            attempts += 1
            nx = rng.randint(0, max(0, img_w - TILE_SIZE))
            ny = rng.randint(0, max(0, img_h - TILE_SIZE))
            # Check no overlap with any annotation on this frame
            nx2, ny2 = nx + TILE_SIZE, ny + TILE_SIZE
            overlap = any(
                nx < ox2 and nx2 > ox1 and ny < oy2 and ny2 > oy1
                for ox1, oy1, ox2, oy2 in occupied
            )
            if overlap:
                continue
            tile_path = _tile_path(src_path, nx, ny, TILE_SIZE)
            out_images.append({
                "id": img_id_counter,
                "file_name": tile_path,
                "width": TILE_SIZE,
                "height": TILE_SIZE,
            })
            img_id_counter += 1
            added_neg += 1

    return {
        "info": {
            "description": "Frigate cluster-2 tiled at 110px (3.64x magnification)",
            "version": "1.0",
            "tile_size": TILE_SIZE,
            "min_diag_px": MIN_DIAG,
            "min_aspect_ratio": MIN_AR,
        },
        "categories": _CATEGORIES,
        "images": out_images,
        "annotations": out_annotations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--neg-per-frame", type=int, default=3,
                        help="Negative tiles per source frame (default: 3).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=_OUT)
    args = parser.parse_args()

    coco = build(neg_per_frame=args.neg_per_frame, seed=args.seed)

    n_pos  = len(coco["annotations"])
    n_imgs = len(coco["images"])
    n_neg  = n_imgs - n_pos

    print(f"Cluster-2 tiles: {n_imgs} images ({n_pos} positive, {n_neg} negative), {n_pos} annotations")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(coco, indent=2))
    print(f"Written → {args.output}")


if __name__ == "__main__":
    main()
