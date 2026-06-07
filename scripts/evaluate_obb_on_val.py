#!/usr/bin/env python3
"""Run MMDet ViT-S OBB model on 1800px heatmap val tiles.

Each 1800px tile is sub-tiled into 400px crops matching the OBB training
scale (model was trained on 400px Atwood crops).  Detections are mapped
back to 1800px tile coordinates and deduplicated with NMS.  The output
predictions.json is compatible with run_posthoc_threshold_analysis.py so
OBB and heatmap results can be compared on the same eval set.

Usage
-----
    python scripts/evaluate_obb_on_val.py \\
        --annotations /Volumes/External/TrainingData/annotations/val_run12_1800_npy.json \\
        --checkpoint  weights/run5_vits_mmdet/best_coco_bbox_mAP_epoch_15.pth \\
        --config      weights/run5_vits_mmdet/streak_dinov3_vits_400px_run5.py \\
        --output-dir  results/obb_vs_heatmap

    # Quick smoke-test on 20 images:
    python scripts/evaluate_obb_on_val.py ... --max-images 20
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_TILE_RE = re.compile(r"^(.+?)__tx(\d+)_ty(\d+)_ts(\d+)$")


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_bgr(file_name: str, norm_mode: str) -> np.ndarray:
    """Return uint8 BGR (H, W, 3) array from FITS virtual tile or NPY path."""
    from inference.fits_loader import apply_norm

    path = Path(file_name)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        raw = np.load(str(path))
        if raw.dtype == np.uint8:
            gray = raw  # already normalised at tile-convert time
        else:
            gray = apply_norm(raw.astype(np.float32), norm_mode)[:, :, 0]
    elif suffix in {".fits", ".fit", ".fts"}:
        m = _TILE_RE.match(path.stem)
        if m:
            real = str(path.parent / (m.group(1) + suffix))
            x0, y0, ts = int(m.group(2)), int(m.group(3)), int(m.group(4))
            from astropy.io import fits as _fits
            with _fits.open(real) as hdul:
                raw = hdul[0].data.astype(np.float32)
            normed = apply_norm(raw, norm_mode)[:, :, 0]
            gray = normed[y0:y0 + ts, x0:x0 + ts]
        else:
            from astropy.io import fits as _fits
            with _fits.open(str(path)) as hdul:
                raw = hdul[0].data.astype(np.float32)
            gray = apply_norm(raw, norm_mode)[:, :, 0]
    else:
        raise ValueError(f"Unsupported image extension: {suffix}")

    # Stack grayscale to BGR (all channels equal; data_preprocessor does bgr→rgb
    # but since channels are identical the conversion is a no-op)
    return np.stack([gray, gray, gray], axis=-1).astype(np.uint8)


# ---------------------------------------------------------------------------
# Sub-tiling helpers
# ---------------------------------------------------------------------------

def _crop_positions(img_size: int, crop_size: int, stride: int) -> list[int]:
    """Sliding-window positions that fully cover [0, img_size)."""
    if img_size <= crop_size:
        return [0]
    positions = list(range(0, img_size - crop_size, stride))
    # Ensure last window reaches the right/bottom edge
    if positions[-1] + crop_size < img_size:
        positions.append(img_size - crop_size)
    return positions


def _pad_to(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Zero-pad array to (h, w, C) — only called for edge tiles."""
    pad = np.zeros((h, w, arr.shape[2]), dtype=arr.dtype)
    pad[:arr.shape[0], :arr.shape[1]] = arr
    return pad


# ---------------------------------------------------------------------------
# NMS (axis-aligned)
# ---------------------------------------------------------------------------

def _iou_aa(a: list[float], b: list[float]) -> float:
    xi1 = max(a[0], b[0]); yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2]); yi2 = min(a[3], b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _nms(boxes: list[list[float]], scores: list[float],
         iou_thresh: float = 0.5) -> list[int]:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    kept: list[int] = []
    while order:
        i = order.pop(0)
        kept.append(i)
        order = [j for j in order if _iou_aa(boxes[i], boxes[j]) < iou_thresh]
    return kept


# ---------------------------------------------------------------------------
# Model inference (bypasses inference_detector to avoid pipeline rebuild)
# ---------------------------------------------------------------------------

def _run_batch(
    model,
    crops: list[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run MMDet model on a batch of uint8 BGR (H, W, 3) crops.

    Sends all crops in one forward pass to amortise Python/model overhead.
    Returns a list of (bboxes, scores) per crop — (N, 4) and (N,) arrays.
    Empty crops return (empty_array, empty_array).
    """
    from mmdet.structures import DetDataSample

    tensors = []
    data_samples = []
    for crop in crops:
        h, w = crop.shape[:2]
        tensors.append(torch.from_numpy(crop).permute(2, 0, 1).float())
        ds = DetDataSample()
        ds.set_metainfo({
            "img_shape": (h, w),
            "ori_shape": (h, w),
            "scale_factor": np.array([1.0, 1.0], dtype=np.float32),
            "img_id": 0,
        })
        data_samples.append(ds)

    data = {"inputs": tensors, "data_samples": data_samples}
    with torch.no_grad():
        results = model.test_step(data)

    out = []
    for r in results:
        inst = r.pred_instances
        if len(inst) == 0:
            out.append((np.empty((0, 4), dtype=np.float32),
                        np.empty((0,), dtype=np.float32)))
        else:
            out.append((inst.bboxes.cpu().numpy(),
                        inst.scores.cpu().numpy()))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", required=True,
                    help="COCO annotation JSON with 1800px tile paths "
                         "(val_run12_1800_npy.json or val_atwood_tiled_ts1800.json)")
    ap.add_argument("--checkpoint", required=True,
                    help="MMDet checkpoint (.pth)")
    ap.add_argument("--config", required=True,
                    help="MMDet config (.py) stored alongside the checkpoint")
    ap.add_argument("--output-dir", required=True,
                    help="Directory to write obb_predictions.json")
    ap.add_argument("--norm-mode", default="zscore",
                    choices=["zscore", "autostretch", "zscale"],
                    help="Normalisation applied to raw tiles before model input "
                         "(default: zscore — must match heatmap eval norm)")
    ap.add_argument("--crop-size", type=int, default=400,
                    help="Sub-tile size in px (default: 400 — OBB training tile size)")
    ap.add_argument("--crop-stride", type=int, default=350,
                    help="Sub-tile stride in px (default: 350 — ~12.5%% overlap)")
    ap.add_argument("--min-conf", type=float, default=0.01,
                    help="Minimum score to include in output (default: 0.01)")
    ap.add_argument("--nms-iou", type=float, default=0.5,
                    help="IoU threshold for within-image NMS (default: 0.5)")
    ap.add_argument("--max-images", type=int, default=None,
                    help="Process only first N images (for smoke-testing)")
    ap.add_argument("--device", default=None,
                    help="Torch device (default: mps if available, else cpu)")
    args = ap.parse_args()

    if args.device is None:
        args.device = "mps" if torch.backends.mps.is_available() else "cpu"

    logger.info("Device: %s", args.device)

    # ── Load MMDet model ────────────────────────────────────────────────────
    # PyTorch 2.6 changed weights_only default to True but MMEngine checkpoints
    # contain numpy arrays and HistoryBuffer objects.  Temporarily patch
    # torch.load to use weights_only=False for the duration of init_detector.
    import functools as _functools
    _orig_torch_load = torch.load
    torch.load = _functools.partial(_orig_torch_load, weights_only=False)
    try:
        from mmdet.apis import init_detector
        logger.info("Loading OBB checkpoint: %s", args.checkpoint)
        model = init_detector(args.config, args.checkpoint, device=args.device)
    finally:
        torch.load = _orig_torch_load  # always restore
    model.eval()
    _device = args.device


    # ── Load annotations ────────────────────────────────────────────────────
    ann_path = Path(args.annotations)
    coco = json.loads(ann_path.read_text())
    images = coco["images"]
    if args.max_images:
        images = images[: args.max_images]
        logger.info("--max-images %d: processing subset", args.max_images)

    logger.info(
        "Eval set: %s  (%d images)", ann_path.name, len(images)
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions: list[dict] = []
    skipped = 0

    for idx, img_info in enumerate(images):
        image_id = int(img_info["id"])
        file_name = img_info["file_name"]

        try:
            bgr = _load_bgr(file_name, args.norm_mode)
        except Exception as exc:
            logger.warning("  skip %s: %s", Path(file_name).name, exc)
            skipped += 1
            continue

        img_h, img_w = bgr.shape[:2]
        xs = _crop_positions(img_w, args.crop_size, args.crop_stride)
        ys = _crop_positions(img_h, args.crop_size, args.crop_stride)

        # Collect all sub-tile crops and their offsets
        crops_for_batch: list[np.ndarray] = []
        offsets: list[tuple[int, int]] = []

        for y0 in ys:
            for x0 in xs:
                crop = bgr[y0:y0 + args.crop_size, x0:x0 + args.crop_size]
                if crop.shape[0] < 4 or crop.shape[1] < 4:
                    continue
                if crop.shape[0] < args.crop_size or crop.shape[1] < args.crop_size:
                    crop = _pad_to(crop, args.crop_size, args.crop_size)
                crops_for_batch.append(crop)
                offsets.append((x0, y0))

        all_boxes: list[list[float]] = []
        all_scores: list[float] = []

        if crops_for_batch:
            batch_results = _run_batch(model, crops_for_batch)
            for (bboxes, scores), (x0, y0) in zip(batch_results, offsets):
                if len(bboxes) == 0:
                    continue

                # Map sub-tile coordinates → 1800px tile coordinates
                bboxes[:, 0] += x0
                bboxes[:, 2] += x0
                bboxes[:, 1] += y0
                bboxes[:, 3] += y0

                mask = scores >= args.min_conf
                all_boxes.extend(bboxes[mask].tolist())
                all_scores.extend(scores[mask].tolist())

        # NMS to deduplicate overlapping sub-tile predictions
        if all_boxes:
            kept = _nms(all_boxes, all_scores, iou_thresh=args.nms_iou)
            for k in kept:
                x1, y1, x2, y2 = all_boxes[k]
                score = all_scores[k]
                bw = max(x2 - x1, 1.0)
                bh = max(y2 - y1, 1.0)
                predictions.append({
                    "image_id": image_id,
                    "confidence": float(score),
                    "obb": {
                        "cx": float((x1 + x2) / 2),
                        "cy": float((y1 + y2) / 2),
                        "w": float(bw),
                        "h": float(bh),
                        "angle_deg": 0.0,
                    },
                    "streak_length_px": float(max(bw, bh)),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                })

        if (idx + 1) % 100 == 0 or idx + 1 == len(images):
            logger.info(
                "  %d/%d images  %d predictions  %d skipped",
                idx + 1, len(images), len(predictions), skipped,
            )

    out_path = out_dir / "obb_predictions.json"
    out_path.write_text(json.dumps(predictions))
    logger.info("Saved %d predictions → %s", len(predictions), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
