"""Evaluate a plain PyTorch DINOv3 heatmap checkpoint as ARGUS detections."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from scipy import ndimage
from skimage.transform import radon
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.metrics import evaluate
from inference.device import get_device
from inference.fits_loader import FITSLoader
from models.plain_dinov3.streak_heatmap import (
    ConvNeXtStreakHeatmap,
    DINOv3StreakHeatmap,
    decode_geometry,
    imagenet_normalize,
)
from training.dinov3_heatmap_dataset import StreakHeatmapDataset, collate_heatmap_batch
from training.train_dinov3_heatmap_cached import HeatmapHead

logger = logging.getLogger(__name__)


def _component_to_obb(
    mask: np.ndarray,
    score_map: np.ndarray,
    patch_size: int,
    geometry_map: np.ndarray | None = None,
    image_size: int | None = None,
) -> dict[str, Any] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return None
    pts = np.column_stack([(xs + 0.5) * patch_size, (ys + 0.5) * patch_size]).astype(np.float32)
    center = pts.mean(axis=0)
    cov = np.cov((pts - center).T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    major = vecs[:, order[0]]
    minor = vecs[:, order[1]]
    rel = pts - center
    along = rel @ major
    across = rel @ minor
    length = max(float(along.max() - along.min()) + patch_size, patch_size)
    width = max(float(across.max() - across.min()) + patch_size, patch_size)
    angle = math.degrees(math.atan2(float(major[1]), float(major[0]))) % 180.0
    if geometry_map is not None and image_size is not None:
        geom_vals = geometry_map[:, mask]
        if geom_vals.shape[1] > 0:
            cos2 = float(geom_vals[0].mean())
            sin2 = float(geom_vals[1].mean())
            if abs(cos2) + abs(sin2) > 1e-3:
                angle = (0.5 * math.degrees(math.atan2(sin2, cos2))) % 180.0
            length = max(float(geom_vals[2].mean()) * image_size, patch_size)
            width = max(float(geom_vals[3].mean()) * image_size, patch_size)
    confidence = float(score_map[mask].mean())
    return {
        "confidence": confidence,
        "obb": {
            "cx": float(center[0]),
            "cy": float(center[1]),
            "w": length,
            "h": width,
            "angle_deg": angle,
        },
        "streak_length_px": length,
    }


def heatmap_to_detections(
    probs: np.ndarray,
    image_id: int,
    threshold: float,
    patch_size: int,
    min_pixels: int,
    image_size: int,
    letterbox: tuple[float, float, float],
    geometry_map: np.ndarray | None = None,
    image_array: np.ndarray | None = None,
    refine_geometry: bool = True,
) -> list[dict[str, Any]]:
    """Convert one heatmap to ARGUS-style detection dictionaries."""
    binary = probs >= threshold
    labels, n_labels = ndimage.label(binary)
    detections: list[dict[str, Any]] = []
    for label_id in range(1, n_labels + 1):
        mask = labels == label_id
        if int(mask.sum()) < min_pixels:
            continue
        det = _component_to_obb(mask, probs, patch_size, geometry_map=geometry_map, image_size=image_size)
        if det is None:
            continue
        scale, pad_x, pad_y = letterbox
        obb = det["obb"]
        obb["cx"] = (obb["cx"] - pad_x) / scale
        obb["cy"] = (obb["cy"] - pad_y) / scale
        obb["w"] /= scale
        obb["h"] /= scale
        det["streak_length_px"] = max(float(obb["w"]), float(obb["h"]))
        if refine_geometry and image_array is not None:
            _refine_detection_geometry(det, image_array)
        det["image_id"] = image_id
        detections.append(det)
    return detections


def _refine_detection_geometry(det: dict[str, Any], image_array: np.ndarray) -> None:
    """Refine component OBB angle and length using original-image pixels."""
    obb = det["obb"]
    h, w = image_array.shape[:2]
    cx = float(np.clip(obb["cx"], 0, max(w - 1, 0)))
    cy = float(np.clip(obb["cy"], 0, max(h - 1, 0)))
    half = max(float(obb["w"]), float(obb["h"]), 64.0) / 2.0 + 24.0
    half = min(half, 384.0)
    x1 = int(max(0, math.floor(cx - half)))
    x2 = int(min(w, math.ceil(cx + half)))
    y1 = int(max(0, math.floor(cy - half)))
    y2 = int(min(h, math.ceil(cy + half)))
    crop = image_array[y1:y2, x1:x2]
    if crop.size == 0 or min(crop.shape[:2]) < 8:
        return
    gray = crop[..., 0].astype(np.float32) if crop.ndim == 3 else crop.astype(np.float32)
    gray = gray - float(np.median(gray))
    gray[gray < 0] = 0
    if float(gray.max()) <= 0:
        return
    step = 1
    if max(gray.shape[:2]) > 512:
        step = int(math.ceil(max(gray.shape[:2]) / 512))
        gray = gray[::step, ::step]

    seed = float(obb.get("angle_deg", 0.0))
    theta = (90.0 - np.arange(seed - 25.0, seed + 25.0, 2.0)) % 180.0
    sinogram = radon(gray, theta=theta, circle=False)
    variances = sinogram.var(axis=0)
    best_theta = float(theta[int(np.argmax(variances))])
    angle = (90.0 - best_theta) % 180.0
    obb["angle_deg"] = angle

    ux, uy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    yy, xx = np.nonzero(gray > max(float(gray.mean() + gray.std()), float(np.percentile(gray, 95))))
    if len(xx) >= 2:
        pts_x = xx.astype(np.float32) * step + x1
        pts_y = yy.astype(np.float32) * step + y1
        along = (pts_x - cx) * ux + (pts_y - cy) * uy
        across = np.abs(-(pts_x - cx) * uy + (pts_y - cy) * ux)
        keep = across <= max(float(obb["h"]), 8.0)
        if int(keep.sum()) >= 2:
            along_kept = along[keep]
            obb["w"] = max(float(along_kept.max() - along_kept.min()), float(obb["w"]))
            obb["h"] = max(float(np.percentile(across[keep], 90) * 2.0), 4.0)
            det["streak_length_px"] = float(obb["w"])


def coco_ground_truth(annotation_file: Path) -> list[dict[str, Any]]:
    """Read COCO annotations into ARGUS metric ground truth format."""
    coco = json.loads(annotation_file.read_text())
    id_to_file = {int(img["id"]): img["file_name"] for img in coco.get("images", [])}
    _ = id_to_file
    gts: list[dict[str, Any]] = []
    for ann in coco.get("annotations", []):
        image_id = int(ann["image_id"])
        obb = ann.get("obb")
        if isinstance(obb, list):
            obb_dict = {"cx": obb[0], "cy": obb[1], "w": obb[2], "h": obb[3], "angle_deg": obb[4]}
        elif isinstance(obb, dict):
            obb_dict = obb
        else:
            x, y, w, h = ann["bbox"]
            obb_dict = {"cx": x + w / 2, "cy": y + h / 2, "w": w, "h": h, "angle_deg": 0.0}
        gts.append({
            "image_id": image_id,
            "obb": obb_dict,
            "streak_length_px": max(float(obb_dict["w"]), float(obb_dict["h"])),
        })
    return gts


def _load_eval_image(path: Path, loader: FITSLoader) -> np.ndarray | None:
    suffix = path.suffix.lower()
    try:
        if suffix in {".fits", ".fit", ".fts"}:
            return np.asarray(loader.load(path)["array"], dtype=np.uint8)
        from PIL import Image
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        logger.warning("Could not reload %s for geometry refinement: %s", path, exc)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/annotations/test.json")
    parser.add_argument("--checkpoint", default="weights/run_plain_dinov3_heatmap/best.pt")
    parser.add_argument("--weights", default=None, help="Override DINOv3 backbone weights from checkpoint args")
    parser.add_argument("--output", default="results/plain_dinov3_heatmap/metrics.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-pixels", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--norm-mode", choices=["autostretch", "zscore", "zscale"],
                        default="autostretch",
                        help="Pixel normalisation for raw FITS/NPY tiles. Must match "
                             "the mode used when the feature cache was built.")
    parser.add_argument("--no-refine-geometry", action="store_true")
    parser.add_argument("--tiled", action="store_true",
                        help="Use the pipeline detector (with tiling) instead of the "
                             "full-image letterbox path. Required for accurate medium-band "
                             "recall on large images. Only supported for convnext checkpoints.")
    parser.add_argument("--stitch", action="store_true",
                        help="Merge collinear tile fragments after per-image NMS "
                             "(tiled mode only). Reconstructs OBB from merged bbox so "
                             "IoU matching against full-extent GT annotations works.")
    parser.add_argument("--stitch-max-gap", type=float, default=400.0,
                        help="Max gap in px between collinear fragments to merge (default: 400).")
    parser.add_argument("--scales", type=int, nargs="+", default=None, metavar="PX",
                        help="Run multi-scale inference at these native tile sizes (px) and "
                             "merge via NMS.  Implies --tiled.  Example: --scales 1800 518 110. "
                             "When omitted, falls back to single-scale tiling controlled by "
                             "VITS_HEATMAP_NATIVE_TILE_SIZE / CONVNEXT_HEATMAP_NATIVE_TILE_SIZE.")
    args = parser.parse_args()
    if args.scales:
        args.tiled = True  # --scales implies tiled

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = ckpt.get("args", {})
    is_cached_head = "head" in ckpt and "model" not in ckpt
    if is_cached_head:
        train_meta = ckpt["train_cache_metadata"]
        weights = args.weights or train_meta.get("weights", "weights/dinov3_vitb16_lvd1689m.pth")
        model_size = train_meta.get("model_size", "base")
        image_size = int(train_meta.get("image_size", 512))
        backbone = train_meta.get("backbone", "vit")
        convnext_stage = int(train_meta.get("convnext_stage") or 3)

        # POLICY: heatmap models trained on cached full-image features at ≤512 px
        # must be evaluated with --tiled when the target images are larger than the
        # cache image_size.  Without tiling, medium streaks (150–400 px native) span
        # <2 feature patches in the resized image, producing blob OBBs that fail all
        # IoU thresholds.  This is a measurement error, not a model failure.
        # The rule: if cache image_size < 600 px, always pass --tiled.
        if not args.tiled and image_size < 600:
            logger.warning(
                "WARNING: evaluating a heatmap checkpoint cached at %d px without "
                "--tiled. Medium-band recall will be artificially near zero because "
                "medium streaks span <2 feature patches at full-image resize. "
                "Re-run with --tiled for valid results.", image_size
            )
    else:
        weights = args.weights or train_args.get("weights", "weights/dinov3_vitb16_lvd1689m.pth")
        model_size = train_args.get("model_size", "base")
        image_size = int(train_args.get("image_size", 512))
        backbone = "vit"
        convnext_stage = 3

    # Spatial stride of the feature map: ViT patch-16 → 16, ConvNeXt stage≤2 → 16, stage3 → 32
    if backbone == "convnext":
        patch_size = 16 if convnext_stage <= 2 else 32
    else:
        patch_size = 16

    if backbone == "convnext":
        model = ConvNeXtStreakHeatmap(
            model_size=model_size, weights=weights, extract_stage=convnext_stage
        ).to(device)
    else:
        model = DINOv3StreakHeatmap(model_size=model_size, weights=weights).to(device)
    if is_cached_head:
        hidden = int(ckpt.get("args", {}).get("hidden_channels", 256))
        cached_head = HeatmapHead(int(ckpt["in_channels"]), hidden)
        cached_head.load_state_dict(ckpt["head"])
        model.head = cached_head.net.to(device)
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()

    # --- Tiled path: delegate to pipeline detector (handles tiling + NMS) ---
    if args.tiled:
        import os as _os
        if args.scales:
            # Multi-scale: run at each requested tile size, merge via NMS
            from inference.multiscale_detector import run_multiscale_detector as _ms_det
            def _run_tiled(arr):  # type: ignore[misc]
                return _ms_det(
                    arr,
                    checkpoint=Path(args.checkpoint),
                    backbone=backbone,
                    scales=args.scales,
                    threshold=args.threshold,
                    min_pixels=args.min_pixels,
                )
        elif backbone == "convnext":
            _os.environ["CONVNEXT_HEATMAP_CHECKPOINT"] = args.checkpoint
            _os.environ["CONVNEXT_HEATMAP_THRESHOLD"]  = str(args.threshold)
            _os.environ["CONVNEXT_HEATMAP_MIN_PIXELS"] = str(args.min_pixels)
            from inference.convnext_heatmap_detector import run_convnext_heatmap_detector as _run_tiled
        elif backbone == "vit":
            _os.environ["VITS_HEATMAP_THRESHOLD"]  = str(args.threshold)
            _os.environ["VITS_HEATMAP_MIN_PIXELS"] = str(args.min_pixels)
            from inference.vits_heatmap_detector import run_vits_heatmap_detector as _run_tiled_vit
            def _run_tiled(arr):  # type: ignore[misc]
                return _run_tiled_vit(arr, checkpoint=Path(args.checkpoint))
        else:
            logger.error("--tiled is only supported for convnext and vit checkpoints")
            return 1

        from inference.tiled_pipeline import stitch_collinear_fragments as _stitch_frags
        from inference.fits_loader import FITSLoader as _FITSLoader
        _os.environ["ARGUS_NORM"] = args.norm_mode  # match training normalization
        _fits_loader = _FITSLoader()
        coco_data = json.loads(Path(args.annotations).read_text())
        images_meta = coco_data.get("images", [])
        if args.max_samples:
            images_meta = images_meta[:args.max_samples]

        # Resolve image paths the same way StreakHeatmapDataset does
        _ann_dir = Path(args.annotations).resolve().parent
        def _resolve(fname: str) -> Path:
            for base in [Path("."), _ann_dir, _ann_dir.parent]:
                p = (base / fname).resolve()
                if p.exists():
                    return p
            return Path(fname)

        predictions: list[dict[str, Any]] = []
        for meta in images_meta:
            img_path = _resolve(meta["file_name"])
            try:
                arr = _load_eval_image(img_path, _fits_loader)
                if arr is None:
                    continue
            except Exception as exc:
                logger.warning("Could not load %s: %s", img_path, exc)
                continue
            dets = _run_tiled(arr)
            if args.stitch and len(dets) > 1:
                # stitch_collinear_fragments expects bbox=[x,y,w,h] and "score".
                # Heatmap detectors return bbox=[x1,y1,x2,y2] with "confidence".
                # OBB reconstruction for merged fragments is handled inside
                # tiled_pipeline._merge() via _merge_obb().
                n_before = len(dets)
                stitch_in = []
                for d in dets:
                    x1, y1, x2, y2 = d["bbox"]
                    stitch_in.append({**d,
                                      "bbox": [x1, y1, x2 - x1, y2 - y1],
                                      "score": d["confidence"]})
                stitched = _stitch_frags(stitch_in, max_gap_px=args.stitch_max_gap)
                dets = []
                for s in stitched:
                    x, y, w, h = s["bbox"]
                    dets.append({**s,
                                 "bbox": [x, y, x + w, y + h],
                                 "confidence": s.get("confidence", s.get("score", 0.0))})
                logger.debug("stitch img_id=%d  %d → %d", meta["id"], n_before, len(dets))
            for det in dets:
                det["image_id"] = int(meta["id"])
            predictions.extend(dets)
            logger.info("tiled eval img_id=%d  dets=%d", meta["id"], len(dets))

        ground_truth = coco_ground_truth(Path(args.annotations))
        if args.max_samples:
            allowed = {int(m["id"]) for m in images_meta}
            ground_truth = [g for g in ground_truth if int(g["image_id"]) in allowed]
        metrics = evaluate(predictions, ground_truth)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"metrics": metrics, "n_predictions": len(predictions),
                   "tiled": True, "stitch": args.stitch}
        out_path.write_text(json.dumps(payload, indent=2))
        (out_path.parent / "predictions.json").write_text(json.dumps(predictions, indent=2))
        logger.info("wrote %s (%d predictions, tiled)", out_path, len(predictions))
        return 0
    # --- End tiled path ---

    ds = StreakHeatmapDataset(args.annotations, image_size=image_size, max_samples=args.max_samples, norm_mode=args.norm_mode)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_heatmap_batch)
    image_loader = FITSLoader()

    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            images = imagenet_normalize(batch["image"].to(device))
            output = model(images)
            logits = output[:, :1]
            probs = torch.sigmoid(logits).cpu().numpy()
            geometry = None
            if output.shape[1] >= 5:
                geometry = decode_geometry(output[:, 1:5]).cpu().numpy()
            image_ids = batch["image_id"].cpu().numpy().tolist()
            letterboxes = batch["letterbox"].cpu().numpy().tolist()
            file_names = batch["file_name"]
            for idx, (prob, image_id, letterbox, file_name) in enumerate(zip(probs[:, 0], image_ids, letterboxes, file_names)):
                image_path = ds._resolve_image_path(str(file_name))
                image_array = None if args.no_refine_geometry else _load_eval_image(image_path, image_loader)
                predictions.extend(
                    heatmap_to_detections(
                        prob,
                        int(image_id),
                        args.threshold,
                        patch_size,
                        args.min_pixels,
                        image_size,
                        (float(letterbox[0]), float(letterbox[1]), float(letterbox[2])),
                        geometry_map=None if geometry is None else geometry[idx],
                        image_array=image_array,
                        refine_geometry=not args.no_refine_geometry,
                    )
                )

    ground_truth = coco_ground_truth(Path(args.annotations))
    if args.max_samples:
        allowed_ids = {int(meta["id"]) for meta in ds.images}
        ground_truth = [gt for gt in ground_truth if int(gt["image_id"]) in allowed_ids]

    metrics = evaluate(predictions, ground_truth)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": metrics, "n_predictions": len(predictions)}
    out_path.write_text(json.dumps(payload, indent=2))
    (out_path.parent / "predictions.json").write_text(json.dumps(predictions, indent=2))
    logger.info("wrote %s (%d predictions)", out_path, len(predictions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
