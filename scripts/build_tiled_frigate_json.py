"""Tile Frigate 2325×1555 images into 400 px crops for short-streak training.

Frigate streaks are 9–66 px diagonal on 2325×1555 frames — sub-pixel for the
full-image resize path.  This script tiles each image with a sliding 400 px
window (25 % overlap) and emits:
  - every tile that contains ≥1 annotation centre
  - one random negative tile per unannotated image (domain adaptation)

Output: data/annotations/frigate_tiled_train.json  (COCO format)

Tile coordinates are embedded in file_name as a virtual path:
  /Volumes/External/frigate/processed/<stem>__tx<x0>_ty<y0>_ts400.png
The LoadFITSFromFile transform is extended to honour this suffix via
training/transforms.py (handled separately at load time by the tile crop).
"""

from __future__ import annotations

import json
import logging
import math
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ANN_DIR = _REPO_ROOT / "data/annotations"
_CANONICAL_CATEGORIES = [{"id": 1, "name": "streak", "supercategory": "satellite"}]

TILE_SIZE = 400
OVERLAP = 0.25
# Minimum fraction of original bbox area that must survive clipping to keep an annotation.
MIN_AREA_FRACTION = 0.25
NEGATIVE_TILES_PER_IMAGE = 1
RANDOM_SEED = 42


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    n = int(math.ceil((length - tile_size) / stride)) + 1
    return [i * stride for i in range(n)]


def _clip_bbox(bbox: list[float], x0: int, y0: int, tile_size: int) -> list[float] | None:
    """Clip [x, y, w, h] to tile and return clipped [x, y, w, h] in tile coords, or None."""
    bx, by, bw, bh = bbox
    orig_area = bw * bh
    if orig_area <= 0:
        return None

    x1 = max(bx, float(x0))
    y1 = max(by, float(y0))
    x2 = min(bx + bw, float(x0 + tile_size))
    y2 = min(by + bh, float(y0 + tile_size))

    if x2 <= x1 or y2 <= y1:
        return None
    clipped_area = (x2 - x1) * (y2 - y1)
    if clipped_area / orig_area < MIN_AREA_FRACTION:
        return None

    return [x1 - x0, y1 - y0, x2 - x1, y2 - y1]


def _tile_file_name(original_path: str, x0: int, y0: int, tile_size: int) -> str:
    """Encode tile position into the filename so the data loader can crop."""
    stem = Path(original_path).stem
    suffix = Path(original_path).suffix
    parent = str(Path(original_path).parent)
    return f"{parent}/{stem}__tx{x0}_ty{y0}_ts{tile_size}{suffix}"


def build_tiled_frigate_json() -> None:
    rng = random.Random(RANDOM_SEED)
    stride = max(1, int(round(TILE_SIZE * (1.0 - OVERLAP))))

    src_path = _ANN_DIR / "frigate_streaks.json"
    with open(src_path) as f:
        src = json.load(f)

    ann_by_image: dict[int, list[dict]] = {}
    for ann in src.get("annotations", []):
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    out_images: list[dict] = []
    out_annotations: list[dict] = []
    next_img_id = 1
    next_ann_id = 1

    positive_count = 0
    negative_count = 0

    for img in src["images"]:
        W = img["width"]
        H = img["height"]
        orig_path = img["file_name"]
        anns = ann_by_image.get(img["id"], [])

        xs = _tile_starts(W, TILE_SIZE, stride)
        ys = _tile_starts(H, TILE_SIZE, stride)

        if anns:
            # Emit every tile that captures ≥1 annotation centre.
            ann_centres = [(a["bbox"][0] + a["bbox"][2] / 2, a["bbox"][1] + a["bbox"][3] / 2, a) for a in anns]
            emitted_tiles: set[tuple[int, int]] = set()

            for y0 in ys:
                for x0 in xs:
                    tile_anns = []
                    for cx, cy, ann in ann_centres:
                        if x0 <= cx < x0 + TILE_SIZE and y0 <= cy < y0 + TILE_SIZE:
                            clipped = _clip_bbox(ann["bbox"], x0, y0, TILE_SIZE)
                            if clipped is not None:
                                tile_anns.append((clipped, ann.get("category_id", 1)))

                    if not tile_anns:
                        continue

                    emitted_tiles.add((x0, y0))
                    fname = _tile_file_name(orig_path, x0, y0, TILE_SIZE)
                    out_images.append({
                        "id": next_img_id,
                        "file_name": fname,
                        "width": TILE_SIZE,
                        "height": TILE_SIZE,
                    })
                    for clipped_bbox, _ in tile_anns:
                        out_annotations.append({
                            "id": next_ann_id,
                            "image_id": next_img_id,
                            "category_id": 1,
                            "bbox": clipped_bbox,
                            "area": clipped_bbox[2] * clipped_bbox[3],
                            "iscrowd": 0,
                        })
                        next_ann_id += 1
                    next_img_id += 1
                    positive_count += 1

        else:
            # Negative image — emit one random tile for domain adaptation.
            x0 = rng.choice(xs)
            y0 = rng.choice(ys)
            fname = _tile_file_name(orig_path, x0, y0, TILE_SIZE)
            out_images.append({
                "id": next_img_id,
                "file_name": fname,
                "width": TILE_SIZE,
                "height": TILE_SIZE,
            })
            next_img_id += 1
            negative_count += 1

    result = {
        "info": {"description": "Frigate tiled training crops", "version": "1.0"},
        "licenses": [],
        "categories": _CANONICAL_CATEGORIES,
        "images": out_images,
        "annotations": out_annotations,
    }

    out_path = _ANN_DIR / "frigate_tiled_train.json"
    with open(out_path, "w") as f:
        json.dump(result, f)

    logger.info(
        "frigate_tiled_train.json: %d positive tiles (%d annotations) + %d negative tiles = %d total images",
        positive_count,
        len(out_annotations),
        negative_count,
        len(out_images),
    )


if __name__ == "__main__":
    build_tiled_frigate_json()
