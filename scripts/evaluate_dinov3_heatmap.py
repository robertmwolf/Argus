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
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.streak_metrics import evaluate_segments
from inference.heatmap_detector_base import _component_to_segment
from inference.device import get_device
from models.plain_dinov3.streak_heatmap import (
    ConvNeXtStreakHeatmap,
    DINOv3StreakHeatmap,
    imagenet_normalize,
)
from training.annotation_endpoints import annotation_to_endpoints
from training.dinov3_heatmap_dataset import StreakHeatmapDataset, collate_heatmap_batch
from training.train_dinov3_heatmap_cached import HeatmapHead

_HEATMAP_PATCH_SIZE = 16  # ViT-S/16 and ViT-B/16

logger = logging.getLogger(__name__)


def heatmap_to_detections(
    probs: np.ndarray,
    image_id: int,
    threshold: float,
    patch_size: int,
    min_pixels: int,
    image_size: int,
    letterbox: tuple[float, float, float],
) -> list[dict[str, Any]]:
    """Convert one heatmap to segment-based detection dicts (x1, y1, x2, y2)."""
    binary = probs >= threshold
    labels, n_labels = ndimage.label(binary)
    scale, pad_x, pad_y = letterbox
    detections: list[dict[str, Any]] = []
    for label_id in range(1, n_labels + 1):
        mask = labels == label_id
        if int(mask.sum()) < min_pixels:
            continue
        det = _component_to_segment(mask, probs, patch_size, image_size)
        if det is None:
            continue
        # Unscale from letterbox canvas to source-tile pixels
        det["x1"] = (det["x1"] - pad_x) / scale
        det["y1"] = (det["y1"] - pad_y) / scale
        det["x2"] = (det["x2"] - pad_x) / scale
        det["y2"] = (det["y2"] - pad_y) / scale
        det["streak_length_px"] = math.sqrt(
            (det["x2"] - det["x1"]) ** 2 + (det["y2"] - det["y1"]) ** 2
        )
        det["image_id"] = image_id
        detections.append(det)
    return detections


def coco_ground_truth(annotation_file: Path) -> list[dict[str, Any]]:
    """Read COCO annotations into segment-based ground truth format."""
    coco = json.loads(annotation_file.read_text())
    gts: list[dict[str, Any]] = []
    for ann in coco.get("annotations", []):
        image_id = int(ann["image_id"])
        x1, y1, x2, y2 = annotation_to_endpoints(ann)
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        gts.append({
            "image_id": image_id,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "streak_length_px": length,
        })
    return gts



def _detections_from_feat_heatmap(
    feat_probs: np.ndarray,
    threshold: float,
    min_pixels: int,
    image_id: int,
    patch_size: int = _HEATMAP_PATCH_SIZE,
) -> list[dict[str, Any]]:
    """Convert a feature-resolution probability map to detection dicts.

    Args:
        feat_probs: Float32 array shaped (H_feat, W_feat) loaded from the
            heatmap cache.  Each element represents one 16-px patch in the
            source image.
        threshold: Binarisation threshold in [0, 1].
        min_pixels: Minimum connected component size in feature-map pixels.
        image_id: COCO image id to stamp on each detection.
        patch_size: Feature stride in source pixels (default 16).

    Returns:
        Detection dicts with bbox/confidence/x1/y1/x2/y2/streak_length_px
        in source-image pixel coordinates, consistent with the tiled eval path.
    """
    from scipy import ndimage as _ndimage

    feat_h = feat_probs.shape[0]
    binary = feat_probs >= threshold
    labels, n_labels = _ndimage.label(binary)
    detections: list[dict[str, Any]] = []
    for label_id in range(1, n_labels + 1):
        mask = labels == label_id
        if int(mask.sum()) < min_pixels:
            continue
        det = _component_to_segment(mask, feat_probs, patch_size, feat_h * patch_size)
        if det is None:
            continue
        detections.append({
            "confidence":       det["confidence"],
            "peak_confidence":  det.get("peak_confidence", det["confidence"]),
            "x1":               det["x1"],
            "y1":               det["y1"],
            "x2":               det["x2"],
            "y2":               det["y2"],
            "streak_length_px": det["streak_length_px"],
            "method":           "vits_heatmap",
            "image_id":         image_id,
        })
    return detections


def _eval_from_heatmap_cache(args: "argparse.Namespace") -> int:
    """Threshold-sweep eval using pre-cached feature-resolution heatmaps.

    Loads NPY heatmaps written by ``cache_heatmap_maps.py`` and re-applies
    thresholding + connected components + optional stitch at each requested
    threshold without touching the GPU.  All 240 val image heatmaps fit in
    ~100 MB RAM so they are held in memory for efficient multi-threshold sweeps.

    Called from ``main()`` when ``--heatmap-cache`` is provided with ``--tiled``.
    """
    cache_dir = Path(args.heatmap_cache)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        logger.error("manifest.json not found in --heatmap-cache dir: %s", cache_dir)
        return 1
    manifest_data = json.loads(manifest_path.read_text())
    logger.info(
        "Heatmap cache: %d images  checkpoint=%s",
        manifest_data.get("n_images", "?"),
        manifest_data.get("checkpoint", "?"),
    )

    coco_data = json.loads(Path(args.annotations).read_text())
    images_meta = coco_data.get("images", [])
    if args.max_samples:
        images_meta = images_meta[: args.max_samples]
    image_ids_in_split = {int(m["id"]) for m in images_meta}

    ground_truth = coco_ground_truth(Path(args.annotations))
    if args.max_samples:
        ground_truth = [g for g in ground_truth if int(g["image_id"]) in image_ids_in_split]

    # Load all heatmaps into memory (~100 MB for 240 val images at 16× downsample)
    heatmaps: dict[int, np.ndarray] = {}
    n_missing = 0
    for meta in images_meta:
        image_id = int(meta["id"])
        npy_name = manifest_data["images"].get(str(image_id))
        if npy_name is None:
            logger.warning("image_id=%d not in manifest, skipping", image_id)
            n_missing += 1
            continue
        npy_path = cache_dir / npy_name
        if not npy_path.exists():
            logger.warning("NPY missing: %s", npy_path)
            n_missing += 1
            continue
        heatmaps[image_id] = np.load(str(npy_path))
    logger.info("Loaded %d/%d heatmaps (%d missing)", len(heatmaps),
                len(images_meta), n_missing)

    from inference.postprocess import stitch_collinear_segments as _stitch_frags

    def _stitch_dets(dets: list[dict]) -> list[dict]:
        if len(dets) <= 1:
            return dets
        return _stitch_frags(
            dets,
            max_gap_px=args.stitch_max_gap,
            max_growth_ratio=args.stitch_max_growth_ratio,
        )

    def _detect_all(threshold: float) -> list[dict]:
        preds: list[dict] = []
        for meta in images_meta:
            image_id = int(meta["id"])
            feat = heatmaps.get(image_id)
            if feat is None:
                continue
            dets = _detections_from_feat_heatmap(
                feat, threshold, args.min_pixels, image_id
            )
            # Peak-floor gate BEFORE stitch (drops noise components so they
            # cannot seed chains); top-K AFTER stitch (keep the K strongest
            # final streaks per image).
            if args.peak_floor > 0.0:
                dets = [d for d in dets
                        if d.get("peak_confidence", d["confidence"]) >= args.peak_floor]
            if args.stitch and dets:
                dets = _stitch_dets(dets)
            if args.top_k > 0 and len(dets) > args.top_k:
                dets = sorted(dets, key=lambda d: d.get("peak_confidence", d["confidence"]),
                              reverse=True)[:args.top_k]
            preds.extend(dets)
            logger.debug("cache eval img_id=%d t=%.2f dets=%d", image_id, threshold, len(dets))
        return preds

    def _write_metrics(preds: list[dict], threshold_val: float, out_path: Path) -> None:
        metrics = evaluate_segments(preds, ground_truth)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metrics": metrics,
            "n_predictions": len(preds),
            "tiled": True,
            "stitch": args.stitch,
            "threshold": threshold_val,
            "source": "heatmap_cache",
        }
        out_path.write_text(json.dumps(payload, indent=2))
        (out_path.parent / f"predictions_t{int(round(threshold_val * 100)):03d}.json").write_text(
            json.dumps(preds, indent=2)
        )
        logger.info("wrote %s (%d predictions, heatmap-cache)", out_path, len(preds))

    base_out = Path(args.output)

    if args.threshold_sweep:
        for t in args.threshold_sweep:
            tag = f"t{int(round(t * 100)):03d}"
            sweep_out = base_out.parent / f"metrics_{tag}.json"
            preds = _detect_all(t)
            _write_metrics(preds, t, sweep_out)
        return 0

    preds = _detect_all(args.threshold)
    _write_metrics(preds, args.threshold, base_out)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--data-root", default=None,
                        help="Durable dataset root (or set ARGUS_DATA_ROOT).")
    parser.add_argument("--scratch-root", default=None,
                        help="Optional staged local mirror (or set ARGUS_SCRATCH_ROOT).")
    parser.add_argument("--checkpoint", default="weights/run_plain_dinov3_heatmap/best.pt")
    parser.add_argument("--weights", default=None, help="Override DINOv3 backbone weights from checkpoint args")
    parser.add_argument("--output", default="results/plain_dinov3_heatmap/metrics.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold-sweep", type=float, nargs="+", default=None,
                        metavar="T",
                        help="Run a threshold sweep instead of a single eval. Load the model "
                             "and run inference once at --threshold (use a low value like 0.05 "
                             "to keep all candidates), then re-filter by each sweep threshold "
                             "and write metrics to --output with the threshold embedded in the "
                             "filename, e.g. metrics_t050.json. Example: "
                             "--threshold 0.05 --threshold-sweep 0.2 0.3 0.4 0.5 0.6 0.7")
    parser.add_argument("--min-pixels", type=int, default=2)
    parser.add_argument("--peak-floor", type=float, default=0.0,
                        help="Drop detections whose peak activation is below this "
                             "(applied before stitch). 0.0 = off. The true streak "
                             "has a sharp peak (~1.0); diffuse noise blobs are softer.")
    parser.add_argument("--top-k", type=int, default=0,
                        help="Keep only the K highest-peak detections per image "
                             "(applied after stitch). 0 = off.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--norm-mode", choices=["autostretch", "zscore", "zscale"],
                        default="autostretch",
                        help="Pixel normalisation for raw FITS/NPY tiles. Must match "
                             "the mode used when the feature cache was built.")
    parser.add_argument("--tiled", action="store_true",
                        help="Use the pipeline detector (with tiling) instead of the "
                             "full-image letterbox path. Required for accurate medium-band "
                             "recall on large images. Only supported for convnext checkpoints.")
    parser.add_argument("--stitch", action="store_true",
                        help="Merge collinear tile fragments after per-image NMS "
                             "(tiled mode only). The merged segment spans the outer "
                             "compatible endpoints.")
    parser.add_argument("--stitch-max-gap", type=float, default=200.0,
                        help="Max gap in px between collinear fragments to merge (default: 200).")
    parser.add_argument("--stitch-max-growth-ratio", type=float, default=3.0,
                        help="Max merged-span / longer-input-span ratio for stitching (default: 3.0).")
    parser.add_argument("--pretiled", action="store_true",
                        help="Annotation contains pre-tiled crops at the model's native tile size "
                             "(e.g. val_atwood_near_ctx_t400_c4_v1). Suppresses the image_size<600 "
                             "warning — tiles are already at training scale so letterbox is correct.")
    parser.add_argument("--heatmap-cache", default=None, metavar="DIR",
                        help="Directory of pre-cached feature-resolution heatmaps produced by "
                             "scripts/cache_heatmap_maps.py.  When provided with --tiled, skips "
                             "model loading and GPU inference entirely — threshold sweeps run "
                             "from cached NPY probability maps in seconds.  The cache must have "
                             "been built with the same checkpoint and tiling params as this eval.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Fast path: load pre-cached heatmaps and sweep thresholds without GPU inference
    if args.heatmap_cache is not None:
        if not args.tiled:
            logger.error("--heatmap-cache requires --tiled (heatmaps were built with tiled inference)")
            return 1
        return _eval_from_heatmap_cache(args)

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
        # <2 feature patches in the resized image, producing poorly resolved
        # segments. This is a measurement error, not a model failure.
        # The rule: if cache image_size < 600 px, always pass --tiled.
        if not args.tiled and not args.pretiled and image_size < 600:
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
        if backbone == "convnext":
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

        from inference.postprocess import stitch_collinear_segments as _stitch_frags
        from inference.fits_loader import FITSLoader as _FITSLoader, apply_norm as _apply_norm
        _os.environ["ARGUS_NORM"] = args.norm_mode  # match training normalization
        _fits_loader = _FITSLoader()

        def _load_eval_image(path: Path) -> np.ndarray | None:
            try:
                suffix = path.suffix.lower()
                if suffix == ".npy":
                    raw = np.load(str(path))
                    if raw.dtype == np.uint8:
                        return np.stack([raw, raw, raw], axis=-1)
                    return _apply_norm(raw.astype(np.float32), args.norm_mode)
                if suffix in {".fits", ".fit", ".fts"}:
                    return np.asarray(_fits_loader.load(path)["array"], dtype=np.uint8)
                from PIL import Image as _PIL_Image
                with _PIL_Image.open(path) as im:
                    return np.asarray(im.convert("RGB"), dtype=np.uint8)
            except Exception as exc:
                logger.warning("Could not load %s: %s", path, exc)
                return None
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
                arr = _load_eval_image(img_path)
                if arr is None:
                    continue
            except Exception as exc:
                logger.warning("Could not load %s: %s", img_path, exc)
                continue
            # Crop to the annotation window if tile_origin is present.
            # GT coordinates are tile-local so inference must run in the same space.
            tile_origin = meta.get("tile_origin")
            if tile_origin is not None:
                x0, y0 = int(tile_origin[0]), int(tile_origin[1])
                crop_w = int(meta.get("width",  arr.shape[1]))
                crop_h = int(meta.get("height", arr.shape[0]))
                arr = arr[y0:y0 + crop_h, x0:x0 + crop_w]
            dets = _run_tiled(arr)
            if args.stitch and len(dets) > 1:
                n_before = len(dets)
                dets = _stitch_frags(
                    dets,
                    max_gap_px=args.stitch_max_gap,
                    max_growth_ratio=args.stitch_max_growth_ratio,
                )
                logger.debug("stitch img_id=%d  %d → %d", meta["id"], n_before, len(dets))
            for det in dets:
                det["image_id"] = int(meta["id"])
            predictions.extend(dets)
            logger.info("tiled eval img_id=%d  dets=%d", meta["id"], len(dets))

        ground_truth = coco_ground_truth(Path(args.annotations))
        if args.max_samples:
            allowed = {int(m["id"]) for m in images_meta}
            ground_truth = [g for g in ground_truth if int(g["image_id"]) in allowed]

        def _write_metrics(preds: list[dict], threshold_val: float, out_path: Path) -> None:
            metrics = evaluate_segments(preds, ground_truth)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"metrics": metrics, "n_predictions": len(preds),
                       "tiled": True, "stitch": args.stitch, "threshold": threshold_val}
            out_path.write_text(json.dumps(payload, indent=2))
            (out_path.parent / "predictions.json").write_text(json.dumps(preds, indent=2))
            logger.info("wrote %s (%d predictions, tiled)", out_path, len(preds))

        if args.threshold_sweep:
            # Re-filter the already-run predictions at each sweep threshold — no model reload.
            base_out = Path(args.output)
            for t in args.threshold_sweep:
                tag = f"t{int(round(t * 100)):03d}"
                sweep_out = base_out.parent / f"metrics_{tag}.json"
                filtered = [p for p in predictions if p.get("confidence", 0.0) >= t]
                _write_metrics(filtered, t, sweep_out)
            return 0

        out_path = Path(args.output)
        _write_metrics(predictions, args.threshold, out_path)
        return 0
    # --- End tiled path ---

    ds = StreakHeatmapDataset(
        args.annotations,
        image_size=image_size,
        max_samples=args.max_samples,
        norm_mode=args.norm_mode,
        data_root=args.data_root,
        scratch_root=args.scratch_root,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_heatmap_batch)

    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            images = imagenet_normalize(batch["image"].to(device))
            output = model(images)
            logits = output[:, :1]
            probs = torch.sigmoid(logits).cpu().numpy()
            image_ids = batch["image_id"].cpu().numpy().tolist()
            letterboxes = batch["letterbox"].cpu().numpy().tolist()
            for idx, (prob, image_id, letterbox) in enumerate(zip(probs[:, 0], image_ids, letterboxes)):
                predictions.extend(
                    heatmap_to_detections(
                        prob,
                        int(image_id),
                        args.threshold,
                        patch_size,
                        args.min_pixels,
                        image_size,
                        (float(letterbox[0]), float(letterbox[1]), float(letterbox[2])),
                    )
                )

    ground_truth = coco_ground_truth(Path(args.annotations))
    if args.max_samples:
        allowed_ids = {int(meta["id"]) for meta in ds.images}
        ground_truth = [gt for gt in ground_truth if int(gt["image_id"]) in allowed_ids]

    metrics = evaluate_segments(predictions, ground_truth)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": metrics, "n_predictions": len(predictions)}
    out_path.write_text(json.dumps(payload, indent=2))
    (out_path.parent / "predictions.json").write_text(json.dumps(predictions, indent=2))
    logger.info("wrote %s (%d predictions)", out_path, len(predictions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
