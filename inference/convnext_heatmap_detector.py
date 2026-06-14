"""ConvNeXt-S Stage-2 heatmap detector for the ARGUS pipeline.

Loads the Run 5/6 frozen ConvNeXt-Small backbone + trained HeatmapHead and
converts the output probability map to pipeline-compatible OBB detections.

Checkpoint format is the cached-head format produced by
``training/train_dinov3_heatmap_cached.py``.

Environment variables
---------------------
CONVNEXT_HEATMAP_CHECKPOINT
    Path to the ``best.pt`` checkpoint.
    Default: ``weights/run5_convnext_small_s2_heatmap/best.pt``
CONVNEXT_HEATMAP_THRESHOLD
    Heatmap binarisation threshold (float, default 0.5).
CONVNEXT_HEATMAP_IMAGE_SIZE
    Square input size in pixels (int, default 384).
CONVNEXT_HEATMAP_MIN_PIXELS
    Minimum component size in feature-map pixels (int, default 2).
CONVNEXT_HEATMAP_NATIVE_TILE_SIZE
    Inference tile size in source pixels (int, default 1562).
    **Must match the tile size used when building the training cache.**
    For Run 5/6 models trained on 400px pre-tiled NPY crops, set to 400.
    The 1562px default is only appropriate for models cached with
    ``--native-tile-size 1562``.
CONVNEXT_HEATMAP_TILE_OVERLAP
    Fractional tile overlap (float, default 0.5).
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from inference.device import get_device
from models.plain_dinov3.streak_heatmap import (
    ConvNeXtStreakHeatmap,
    decode_geometry,
    imagenet_normalize,
)

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, tuple[Any, int, torch.device]] = {}

_DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "weights" / "run5_convnext_small_s2_heatmap_pretiled" / "best.pt"


def _default_checkpoint() -> Path:
    return Path(os.environ.get("CONVNEXT_HEATMAP_CHECKPOINT", str(_DEFAULT_CHECKPOINT)))


def get_convnext_heatmap_status() -> dict[str, str]:
    """Return availability metadata for the ConvNeXt heatmap detector."""
    ckpt = _default_checkpoint()
    return {
        "id":      "convnext_heatmap",
        "name":    "ConvNeXt-S HeatMap",
        "type":    "ml",
        "dataset": "Atwood+Frigate Run5",
        "status":  "active" if ckpt.exists() else "no_weights",
    }


def _load_model(checkpoint_path: Path) -> tuple[Any, int, torch.device]:
    """Load and cache the ConvNeXt heatmap model."""
    device = get_device()
    cache_key = str(checkpoint_path.resolve())
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"ConvNeXt heatmap checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_meta = ckpt.get("train_cache_metadata", {})
    weights    = train_meta.get("weights", "weights/dinov3_convnext_small_pretrain_lvd1689m.pth")
    model_size = train_meta.get("model_size", "small")
    stage      = int(train_meta.get("convnext_stage") or 3)
    image_size = int(train_meta.get("image_size", 384))

    from training.train_dinov3_heatmap_cached import HeatmapHead

    backbone = ConvNeXtStreakHeatmap(
        model_size=model_size,
        weights=weights,
        extract_stage=stage,
        freeze_backbone=True,
    ).to(device)

    hidden     = int(ckpt.get("args", {}).get("hidden_channels", 256))
    in_channels = int(ckpt["in_channels"])
    head = HeatmapHead(in_channels, hidden)
    head.load_state_dict(ckpt["head"])
    backbone.head = head.net.to(device)
    backbone.eval()

    result = (backbone, image_size, device)
    _MODEL_CACHE[cache_key] = result
    return result


def _letterbox(array: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
    """Resize image to square canvas with aspect-ratio-preserving letterbox.

    Returns:
        Tuple of (canvas_rgb_uint8, scale, pad_x, pad_y).
    """
    h, w = array.shape[:2]
    scale = min(size / w, size / h)
    new_w = round(w * scale)
    new_h = round(h * scale)
    pad_x = (size - new_w) / 2.0
    pad_y = (size - new_h) / 2.0
    resized = np.array(Image.fromarray(array).resize((new_w, new_h), Image.BILINEAR))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0, x0 = round(pad_y), round(pad_x)
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas, float(scale), float(pad_x), float(pad_y)


def _component_to_segment(
    mask: np.ndarray,
    score_map: np.ndarray,
    patch_size: int,
    geometry_map: np.ndarray | None,
    image_size: int,
) -> dict[str, Any] | None:
    """Fit a line segment to a connected heatmap component via PCA.

    Returns a dict with both the new endpoint fields (x1, y1, x2, y2) and a
    backward-compat obb sub-dict derived from the segment geometry.

    Args:
        mask: Boolean mask of active feature-map pixels for this component.
        score_map: Per-pixel probability map from the heatmap head.
        patch_size: Feature-map patch stride in source pixels.
        geometry_map: Optional 4-channel geometry prediction (cos2, sin2,
            length_norm, width_norm).  When present, overrides PCA angle and
            length.
        image_size: Square canvas side length (used to de-normalise geometry).

    Returns:
        Detection dict or None if the component has fewer than 2 pixels.
    """
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
    length = max(float((rel @ major).ptp()) + patch_size, patch_size)
    width  = max(float((rel @ minor).ptp()) + patch_size, patch_size)
    angle  = math.degrees(math.atan2(float(major[1]), float(major[0]))) % 180.0

    if geometry_map is not None:
        geom = geometry_map[:, mask]
        if geom.shape[1] > 0:
            cos2, sin2 = float(geom[0].mean()), float(geom[1].mean())
            if abs(cos2) + abs(sin2) > 1e-3:
                angle = (0.5 * math.degrees(math.atan2(sin2, cos2))) % 180.0
                # Recompute major axis unit vector from refined angle
                rad = math.radians(angle)
                major = np.array([math.cos(rad), math.sin(rad)], dtype=np.float32)
            length = max(float(geom[2].mean()) * image_size, patch_size)
            width  = max(float(geom[3].mean()) * image_size, patch_size)

    # Compute endpoints from centre + half-length along major axis
    half = length / 2.0
    x1 = float(center[0]) - half * float(major[0])
    y1 = float(center[1]) - half * float(major[1])
    x2 = float(center[0]) + half * float(major[0])
    y2 = float(center[1]) + half * float(major[1])

    return {
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "confidence": float(score_map[mask].mean()),
        # Peak (max) activation in the component. A real streak has a sharp
        # high peak; diffuse background blobs are softer even when their mean
        # clears the threshold. Used by the detector's peak-floor / top-K filter.
        "peak_confidence": float(score_map[mask].max()),
        "streak_length_px": length,
        # Backward-compat OBB derived from segment geometry
        "obb": {
            "cx": float(center[0]),
            "cy": float(center[1]),
            "w":  length,
            "h":  max(width, 3.0),
            "angle_deg": angle,
        },
    }


# Keep old name as an alias for any external callers
_component_to_obb = _component_to_segment


def _run_single_tile(
    array: np.ndarray,
    model: Any,
    image_size: int,
    device: torch.device,
    threshold: float,
    min_pixels: int,
    use_geometry: bool = True,
) -> list[dict[str, Any]]:
    """Run detector on one tile; return detections in tile-local coordinates."""
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=2)
    canvas, scale, pad_x, pad_y = _letterbox(array, image_size)
    img_tensor = torch.from_numpy(canvas.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output   = model(imagenet_normalize(img_tensor))
        logits   = output[:, :1]
        probs    = torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)
        geometry = (
            decode_geometry(output[:, 1:5])[0].cpu().numpy()
            if use_geometry and output.shape[1] >= 5
            else None
        )

    patch_size = 16  # ConvNeXt stage-2 stride equals ViT-S/16 patch stride
    binary = probs >= threshold
    labels, n_labels = ndimage.label(binary)
    detections: list[dict[str, Any]] = []

    for label_id in range(1, n_labels + 1):
        mask = labels == label_id
        if int(mask.sum()) < min_pixels:
            continue
        det = _component_to_segment(mask, probs, patch_size, geometry, image_size)
        if det is None:
            continue
        obb = det["obb"]
        # Unscale OBB centre and dimensions from letterbox canvas to tile-local pixels
        obb["cx"]  = (obb["cx"] - pad_x) / scale
        obb["cy"]  = (obb["cy"] - pad_y) / scale
        obb["w"]  /= scale
        obb["h"]  /= scale
        det["streak_length_px"] = float(obb["w"])
        # Unscale explicit endpoints
        det["x1"] = (det["x1"] - pad_x) / scale
        det["y1"] = (det["y1"] - pad_y) / scale
        det["x2"] = (det["x2"] - pad_x) / scale
        det["y2"] = (det["y2"] - pad_y) / scale
        cx, cy = obb["cx"], obb["cy"]
        hw, hh = obb["w"] / 2, obb["h"] / 2
        detections.append({
            "bbox":             [cx - hw, cy - hh, cx + hw, cy + hh],
            "confidence":       det["confidence"],
            "method":           "convnext_heatmap",
            "x1":               det["x1"],
            "y1":               det["y1"],
            "x2":               det["x2"],
            "y2":               det["y2"],
            "obb":              obb,
            "streak_length_px": det["streak_length_px"],
        })
    return detections


def _run_single_tile_probs(
    array: np.ndarray,
    model: Any,
    image_size: int,
    device: torch.device,
) -> tuple[np.ndarray, float, float, float]:
    """Run model on one tile and return the raw probability map in tile-pixel space.

    Returns:
        Tuple of (heat_tile, scale, pad_x, pad_y) where ``heat_tile`` is a
        float32 array shaped ``(H_tile, W_tile)`` with heatmap probabilities
        upsampled to the original tile resolution.
    """
    import cv2 as _cv2
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=2)
    h, w = array.shape[:2]
    canvas, scale, pad_x, pad_y = _letterbox(array, image_size)
    img_tensor = (
        torch.from_numpy(canvas.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )
    with torch.no_grad():
        output = model(imagenet_normalize(img_tensor))
        probs = torch.sigmoid(output[:, :1])[0, 0].cpu().numpy().astype(np.float32)

    # Upsample feature-map probs → letterbox canvas size → tile pixel size
    probs_canvas = _cv2.resize(probs, (image_size, image_size), interpolation=_cv2.INTER_LINEAR)
    y0r, x0r = round(pad_y), round(pad_x)
    new_h, new_w = round(h * scale), round(w * scale)
    probs_content = probs_canvas[y0r:y0r + new_h, x0r:x0r + new_w]
    heat_tile = _cv2.resize(probs_content, (w, h), interpolation=_cv2.INTER_LINEAR)
    return heat_tile, scale, pad_x, pad_y


def _run_tile_batch_full(
    tiles: list[tuple["np.ndarray", int, int]],
    model: Any,
    image_size: int,
    device: torch.device,
    threshold: float,
    min_pixels: int,
    use_geometry: bool = True,
    batch_size: int = 4,
) -> list[tuple[list[dict[str, Any]], "np.ndarray", int, int, int, int]]:
    """Process tiles in batches with a single forward pass per batch.

    Eliminates the double forward pass that the *_and_heatmap functions
    previously required (one pass for detections, one for the heatmap overlay).
    Returns (dets, heat_tile, x0, y0, tile_h, tile_w) for each input tile in
    the same order they were supplied.
    """
    import cv2 as _cv2

    # Pre-process all tiles to letterbox canvases; keep metadata for unscaling.
    prepped: list[tuple[torch.Tensor, int, int, int, int, float, float, float]] = []
    for tile, x0, y0 in tiles:
        if tile.ndim == 2:
            tile = np.stack([tile] * 3, axis=2)
        th, tw = tile.shape[:2]
        canvas, scale, pad_x, pad_y = _letterbox(tile, image_size)
        t = torch.from_numpy(canvas.astype(np.float32) / 255.0).permute(2, 0, 1)
        prepped.append((t, x0, y0, th, tw, scale, pad_x, pad_y))

    patch_size = 16
    results: list[tuple[list[dict[str, Any]], np.ndarray, int, int, int, int]] = []

    for i in range(0, len(prepped), batch_size):
        chunk = prepped[i:i + batch_size]
        batch = torch.stack([c[0] for c in chunk]).to(device)

        with torch.no_grad():
            output = model(imagenet_normalize(batch))
            logits = output[:, :1]
            probs_np = torch.sigmoid(logits).cpu().numpy().astype(np.float32)  # (N,1,hf,wf)
            if use_geometry and output.shape[1] >= 5:
                geom_np = decode_geometry(output[:, 1:5]).cpu().numpy()  # (N,4,hf,wf)
            else:
                geom_np = None

        for j, (_, x0, y0, th, tw, scale, pad_x, pad_y) in enumerate(chunk):
            probs = probs_np[j, 0]
            geometry = geom_np[j] if geom_np is not None else None

            # Detections
            binary = probs >= threshold
            labels, n_labels = ndimage.label(binary)
            dets: list[dict[str, Any]] = []
            for label_id in range(1, n_labels + 1):
                mask = labels == label_id
                if int(mask.sum()) < min_pixels:
                    continue
                det = _component_to_segment(mask, probs, patch_size, geometry, image_size)
                if det is None:
                    continue
                obb = det["obb"]
                obb["cx"] = (obb["cx"] - pad_x) / scale
                obb["cy"] = (obb["cy"] - pad_y) / scale
                obb["w"] /= scale
                obb["h"] /= scale
                det["streak_length_px"] = float(obb["w"])
                det["x1"] = (det["x1"] - pad_x) / scale
                det["y1"] = (det["y1"] - pad_y) / scale
                det["x2"] = (det["x2"] - pad_x) / scale
                det["y2"] = (det["y2"] - pad_y) / scale
                cx, cy = obb["cx"], obb["cy"]
                hw, hh = obb["w"] / 2, obb["h"] / 2
                dets.append({
                    "bbox": [cx - hw, cy - hh, cx + hw, cy + hh],
                    "confidence": det["confidence"],
                    "method": "convnext_heatmap",
                    "x1": det["x1"], "y1": det["y1"], "x2": det["x2"], "y2": det["y2"],
                    "obb": obb,
                    "streak_length_px": det["streak_length_px"],
                })

            # Heatmap — upsample feature-map probs → tile pixel resolution
            probs_canvas = _cv2.resize(probs, (image_size, image_size), interpolation=_cv2.INTER_LINEAR)
            y0r, x0r = round(pad_y), round(pad_x)
            new_h, new_w = round(th * scale), round(tw * scale)
            probs_content = probs_canvas[y0r:y0r + new_h, x0r:x0r + new_w]
            heat_tile = _cv2.resize(probs_content, (tw, th), interpolation=_cv2.INTER_LINEAR)

            results.append((dets, heat_tile, x0, y0, th, tw))

    return results


def _rescale_detections(dets: list[dict[str, Any]], scale: float) -> list[dict[str, Any]]:
    """Scale detection coordinates from a downscaled image back to original size.

    Args:
        dets: Detections in downscaled-image coordinates.
        scale: Factor to multiply all coordinates by (orig_size / scaled_size).

    Returns:
        New list of dicts with all coordinate and dimension fields scaled.
    """
    out = []
    for det in dets:
        det = dict(det)
        b = det["bbox"]
        det["bbox"] = [b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale]
        obb = dict(det["obb"])
        obb["cx"] *= scale
        obb["cy"] *= scale
        obb["w"]  *= scale
        obb["h"]  *= scale
        det["obb"] = obb
        if "streak_length_px" in det:
            det["streak_length_px"] = det["streak_length_px"] * scale
        if "x1" in det:
            det["x1"] = det["x1"] * scale
            det["y1"] = det["y1"] * scale
            det["x2"] = det["x2"] * scale
            det["y2"] = det["y2"] * scale
        out.append(det)
    return out


def _remap_detection(det: dict[str, Any], x0: int, y0: int) -> dict[str, Any]:
    """Shift tile-local detection coordinates to full-image coordinates.

    Args:
        det: Detection dict in tile-local pixel coordinates.
        x0: Tile left edge in full-image pixels.
        y0: Tile top edge in full-image pixels.

    Returns:
        Detection dict with all coordinate fields shifted by (x0, y0).
    """
    det = dict(det)
    b = det["bbox"]
    det["bbox"] = [b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0]
    obb = {**det["obb"], "cx": det["obb"]["cx"] + x0, "cy": det["obb"]["cy"] + y0}
    det["obb"] = obb
    # Shift explicit endpoints when present
    if "x1" in det:
        det["x1"] = det["x1"] + x0
        det["y1"] = det["y1"] + y0
        det["x2"] = det["x2"] + x0
        det["y2"] = det["y2"] + y0
    return det


def run_convnext_heatmap_detector(array: np.ndarray) -> list[dict[str, Any]]:
    """Run the ConvNeXt heatmap detector on a single image with tiling.

    Tiles the image using ``CONVNEXT_HEATMAP_NATIVE_TILE_SIZE`` (default 1562 px)
    so that medium streaks (~150–400 px native) span 3–8 feature patches rather
    than <2 patches in a full-image 384 px resize.  Uses 50 % overlap between
    tiles and NMS to deduplicate cross-tile detections.

    Args:
        array: uint8 RGB array, shape ``(H, W, 3)``.

    Returns:
        Pipeline-compatible detection dicts with ``obb``, ``confidence``,
        ``bbox``, ``streak_length_px``, and ``method`` keys.
    """
    checkpoint = _default_checkpoint()
    if not checkpoint.exists():
        logger.debug("ConvNeXt heatmap checkpoint not found at %s; skipping", checkpoint)
        return []

    threshold        = float(os.environ.get("CONVNEXT_HEATMAP_THRESHOLD", "0.5"))
    min_pixels       = int(os.environ.get("CONVNEXT_HEATMAP_MIN_PIXELS", "2"))
    # WARNING: tile size at eval must match tile size used during training.
    # Run 5 cache: native_tile_size=400 (pre-tiled 400px NPY crops).
    # Run 6 cache: native_tile_size=400 (pre-tiled 400px NPY crops).
    # The 1562px default below is only correct when the model was cached with
    # --native-tile-size 1562 (not the case for Run 5/6 pretiled models).
    # For Run 5/6 checkpoints always set CONVNEXT_HEATMAP_NATIVE_TILE_SIZE=400.
    native_tile_size = int(os.environ.get("CONVNEXT_HEATMAP_NATIVE_TILE_SIZE", "1562"))
    tile_overlap     = float(os.environ.get("CONVNEXT_HEATMAP_TILE_OVERLAP", "0.5"))

    try:
        model, image_size, device = _load_model(checkpoint)
    except Exception as exc:
        logger.warning("ConvNeXt heatmap model load failed: %s", exc)
        return []

    if array.ndim == 2:
        array = np.stack([array] * 3, axis=2)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    h, w = array.shape[:2]

    if max(h, w) <= native_tile_size:
        dets = _run_single_tile(array, model, image_size, device, threshold, min_pixels)
        logger.debug("ConvNeXt heatmap (single shot): %d detection(s)", len(dets))
        return dets

    from inference.tiled_pipeline import tile_image, _torchvision_nms, _numpy_nms

    all_dets: list[dict[str, Any]] = []
    for tile, x0, y0 in tile_image(array, native_tile_size, tile_overlap):
        for det in _run_single_tile(tile, model, image_size, device, threshold, min_pixels):
            all_dets.append(_remap_detection(det, x0, y0))

    if len(all_dets) <= 1:
        logger.debug("ConvNeXt heatmap (tiled): %d detection(s)", len(all_dets))
        return all_dets

    preds_xywh = [
        {
            "bbox":        [d["bbox"][0], d["bbox"][1],
                            d["bbox"][2] - d["bbox"][0], d["bbox"][3] - d["bbox"][1]],
            "score":       float(d["confidence"]),
            "category_id": 1,
        }
        for d in all_dets
    ]
    try:
        kept = _torchvision_nms(preds_xywh, iou_threshold=0.3)
    except Exception:
        kept = _numpy_nms(preds_xywh, iou_threshold=0.3)

    result = [all_dets[i] for i in kept]
    logger.debug("ConvNeXt heatmap (tiled): %d raw → %d after NMS", len(all_dets), len(result))
    return result
