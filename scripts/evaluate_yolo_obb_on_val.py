#!/usr/bin/env python3
"""Run YOLO OBB model on val_balanced_v1 and produce geometry_metrics predictions.

Sub-tiles each val image at 416px (matching training), maps detections back to
full-image coordinates, deduplicates with NMS, then writes a predictions.json
compatible with eval/geometry_metrics.py.

Usage:
    python scripts/evaluate_yolo_obb_on_val.py \
        --weights runs/obb/weights/yolo_run17/run/weights/best.pt \
        --annotations data/annotations/val_balanced_v1.json \
        --output results/yolo_run17/balanced_v1/pf85/predictions.json \
        [--conf 0.25] [--iou 0.45] [--tile-size 416] [--tile-overlap 0.25]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load_image(img_meta: dict) -> np.ndarray:
    """Load image tile from metadata, zscore-normalise, return uint8 HxWx3 BGR.

    Handles tile_origin: if present, crops the full FITS to the annotation window
    before returning, matching how evaluate_dinov3_heatmap.py loads val images.
    """
    from inference.fits_loader import apply_norm

    file_name = img_meta["file_name"]
    path = Path(file_name)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        raw = np.load(str(path))
        if raw.dtype == np.uint8:
            gray = raw if raw.ndim == 2 else raw[:, :, 0]
        else:
            gray = apply_norm(raw.astype(np.float32), "zscore")[:, :, 0]
    elif suffix in {".fits", ".fit", ".fts"}:
        from astropy.io import fits as _fits
        with _fits.open(str(path)) as hdul:
            raw = hdul[0].data.astype(np.float32)
        gray = apply_norm(raw, "zscore")[:, :, 0]
        # Crop to annotation window if tile_origin is specified
        tile_origin = img_meta.get("tile_origin")
        if tile_origin is not None:
            x0, y0 = int(tile_origin[0]), int(tile_origin[1])
            crop_w = int(img_meta.get("width", gray.shape[1]))
            crop_h = int(img_meta.get("height", gray.shape[0]))
            gray = gray[y0:y0 + crop_h, x0:x0 + crop_w]
    else:
        raise ValueError(f"Unsupported image extension: {suffix}")

    return np.stack([gray, gray, gray], axis=-1).astype(np.uint8)


def _subtile_coords(h: int, w: int, tile_size: int, overlap: float):
    """Yield (x0, y0) top-left corners for a sliding window."""
    stride = max(1, int(tile_size * (1 - overlap)))
    ys = list(range(0, max(1, h - tile_size), stride)) + [max(0, h - tile_size)]
    xs = list(range(0, max(1, w - tile_size), stride)) + [max(0, w - tile_size)]
    seen = set()
    for y0 in ys:
        for x0 in xs:
            key = (x0, y0)
            if key not in seen:
                seen.add(key)
                yield x0, y0


def _obb_corners(cx, cy, w, h, angle_deg):
    """Return 4x2 corner array for an OBB."""
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    hw, hh = w / 2, h / 2
    offsets = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    return np.array([
        [cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a]
        for dx, dy in offsets
    ])


def _iou_obb(a, b):
    """Approximate OBB IoU via bounding-box of corners."""
    ca = _obb_corners(**a)
    cb = _obb_corners(**b)
    ax0, ay0 = ca.min(0); ax1, ay1 = ca.max(0)
    bx0, by0 = cb.min(0); bx1, by1 = cb.max(0)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms(dets: list[dict], iou_thresh: float) -> list[dict]:
    """Greedy NMS on OBB detections sorted by confidence."""
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)
    keep = []
    for d in dets:
        if all(_iou_obb(d["obb"], k["obb"]) < iou_thresh for k in keep):
            keep.append(d)
    return keep


def run_inference(weights: str, annotations: str, output: str,
                  conf: float, iou: float, tile_size: int, overlap: float):
    from ultralytics import YOLO
    import torch

    model = YOLO(weights)
    ann = json.loads(Path(annotations).read_text())
    images = {img["id"]: img for img in ann["images"]}

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_predictions = []

    for img_meta in ann["images"]:
        img_id = img_meta["id"]
        file_name = img_meta["file_name"]

        try:
            bgr = _load_image(img_meta)
        except Exception as e:
            print(f"  [WARN] failed to load {file_name}: {e}")
            continue

        h, w = bgr.shape[:2]
        tile_dets: list[dict] = []

        for x0, y0 in _subtile_coords(h, w, tile_size, overlap):
            x1 = min(x0 + tile_size, w)
            y1 = min(y0 + tile_size, h)
            crop = bgr[y0:y1, x0:x1]

            results = model.predict(
                crop, conf=conf, iou=iou, verbose=False, device="mps"
            )

            for r in results:
                if r.obb is None or len(r.obb) == 0:
                    continue
                boxes = r.obb
                for i in range(len(boxes)):
                    xywhr = boxes.xywhr[i].cpu().numpy()  # cx,cy,w,h,angle_rad
                    score = float(boxes.conf[i].cpu().numpy())
                    cx_crop, cy_crop, bw, bh, angle_rad = xywhr
                    angle_deg = math.degrees(float(angle_rad))
                    # map back to full image coords
                    cx_full = float(cx_crop) + x0
                    cy_full = float(cy_crop) + y0
                    tile_dets.append({
                        "image_id": img_id,
                        "score": score,
                        "obb": {
                            "cx": cx_full,
                            "cy": cy_full,
                            "w": float(bw),
                            "h": float(bh),
                            "angle_deg": angle_deg,
                        },
                    })

        # NMS across tiles
        kept = _nms(tile_dets, iou_thresh=iou)
        all_predictions.extend(kept)
        print(f"  {Path(file_name).name}: {len(tile_dets)} raw → {len(kept)} after NMS")

    out_path.write_text(json.dumps(all_predictions, indent=2))
    print(f"\nWrote {len(all_predictions)} predictions to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--tile-size", type=int, default=416)
    ap.add_argument("--tile-overlap", type=float, default=0.25)
    args = ap.parse_args()

    run_inference(
        weights=args.weights,
        annotations=args.annotations,
        output=args.output,
        conf=args.conf,
        iou=args.iou,
        tile_size=args.tile_size,
        overlap=args.tile_overlap,
    )


if __name__ == "__main__":
    main()
