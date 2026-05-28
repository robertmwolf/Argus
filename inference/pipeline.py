"""End-to-end ARGUS inference pipeline.

Accepts a FITS file path and returns a list of satellite streak detections
with optional oriented bounding boxes, sky coordinates, and cross-identifications.

Pipeline stages:
  1. FITS loading and Z-score normalisation   (inference/fits_loader.py)
  2. DINO inference                            (mmdet.apis)
  3. Radon angle refinement + OBB NMS          (inference/postprocess.py)
  4. WCS pixel → RA/Dec conversion             (astropy.wcs)
  5. TLE cross-identification                  (inference/crossid.py)

Fast mode (fast=True or FAST_MODE=true):
  - Skips Radon refinement (uses bbox aspect-ratio angle estimate)
  - Skips cross-identification (identifications=[])
  - Forces image_size=256
  - Target: <60 seconds per image on Mac M3

Timing logged at DEBUG level:
  fits_load_ms, inference_ms, postprocess_ms, crossid_ms, db_write_ms
"""

from __future__ import annotations

import copy
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PyTorch 2.6 compatibility — checkpoint loading patch
# ---------------------------------------------------------------------------

def _patch_torch_load_weights_only() -> None:
    try:
        import inspect
        import torch
        import mmengine.runner.checkpoint as ckpt_mod
        sig = inspect.signature(torch.load)
        if "weights_only" not in sig.parameters:
            return
        import functools
        _orig = torch.load

        @functools.wraps(_orig)
        def _patched(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig(*args, **kwargs)

        ckpt_mod.torch.load = _patched  # type: ignore[attr-defined]
    except Exception:
        pass


_patch_torch_load_weights_only()

# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

_DEFAULT_CONFIDENCE_THRESHOLD = 0.05
_FAST_IMAGE_SIZE = 256

# Minimum inference resolution for large sensor images.
# The device_config image_size (400 px on CPU/MPS) was chosen for training
# memory budgets.  For inference on full-frame telescope images (typically
# 4–6 k pixels wide) it produces a scale of ~6 %, shrinking a 500 px streak
# to ~30 px — too small for reliable detection.  We clamp the inference size
# to at least this value so a 500 px streak stays above ~80 px after scaling.
_MIN_INFERENCE_IMAGE_SIZE = 1280

# ASTRiDE's contour finder can take minutes on full-frame 20+ MP FITS images,
# which blocks the API worker because detector results are joined before
# post-processing. Run it at full resolution below this cap and downsample above
# it by default. If downsampling is disabled, ASTRiDE still runs at full
# resolution because selecting the detector should always execute it.
_DEFAULT_ASTRIDE_MAX_PIXELS = 8_000_000
_DEFAULT_ASTRIDE_DOWNSAMPLE_MAX_PIXELS = 4_000_000
_DEFAULT_ASTRIDE_DOWNSAMPLE_MIN_SCALE = 0.25
_DEFAULT_ASTRIDE_MAX_DETECTIONS = 50

# ASTRiDE is disabled by default: near-zero recall on JPEG inputs (most test
# sets), 600+ FP detections per image on real FITS, and max corroboration boost
# of 4 %.  Set ARGUS_ENABLE_ASTRIDE=1 or pass "astride" in enabled_detectors
# to re-enable it for research/comparison runs.
_ASTRIDE_ENABLED_BY_DEFAULT: bool = (
    os.environ.get("ARGUS_ENABLE_ASTRIDE", "0").strip() not in ("", "0", "false", "False")
)

# Radon refinement is intentionally CPU-bound. With a low detector threshold a
# noisy large frame can yield 100+ DINO boxes, turning postprocess into minutes
# of work. Keep only the strongest DINO candidates before the expensive OBB
# refinement; set DINO_MAX_POSTPROCESS_DETECTIONS=0 to disable this cap.
_DEFAULT_DINO_MAX_POSTPROCESS_DETECTIONS = 10

# Axis-aligned boxes do not encode slope sign, and very loose DINO boxes can
# seed the Radon search tens of degrees away from the actual streak. 60 degrees
# keeps the API CPU budget bounded after the 512 px Radon crop cap while covering
# the common mirrored/loose-box failure mode seen on long diagonal streaks.
_DEFAULT_RADON_ANGLE_SEARCH_RANGE = 60.0

# SGP4 cross-identification can be expensive because each detection is scored
# against a catalog window. For interactive API use, identify the strongest
# detections first and leave lower-confidence candidates as un-identified.
_DEFAULT_CROSSID_MAX_DETECTIONS = 3


# ---------------------------------------------------------------------------
# Static model registry — all known DINO variants
# ---------------------------------------------------------------------------

def _model_registry() -> list[dict]:
    """Return metadata for every registered DINO model variant.

    This is the canonical catalog used by get_detector_statuses() and
    resolve_model_specs(). It is independent of active env-var config so
    the detectors endpoint always reflects what is *available*, not just
    what is currently loaded.

    Returns:
        List of dicts with keys: id, size, label, dataset, weights, config.
    """
    root = Path(__file__).resolve().parent.parent
    return [
        {
            "id":      "dinov3_vitb",
            "size":    "dinov3_vitb",
            "label":   "DINOv3 ViT-B",
            "dataset": "SatStreaks+GTImages",
            "weights": root / "weights" / "dinov3_vitb_augmented" / "best_coco_bbox_mAP_epoch_10.pth",
            "config":  root / "models" / "dino" / "streak_dinov3_vitb.py",
        },
        {
            "id":      "dinov3_vitb_run3",
            "size":    "dinov3_vitb_run3",
            "label":   "DINOv3 ViT-B Run 3 (cold-start, nodm)",
            "dataset": "SatStreaks+BrentImages+Frigate",
            "weights": root / "weights" / "run3_cold_nodm" / "best.pth",
            "config":  root / "models" / "dino" / "streak_dinov3_vitb_400px_run3.py",
        },
        {
            "id":      "dinov3_vitl",
            "size":    "dinov3_vitl",
            "label":   "DINOv3 ViT-L",
            "dataset": "SatStreaks+GTImages",
            "weights": root / "weights" / "run_5070ti_dinov3_vitl" / "best_coco_bbox_mAP_epoch_50.pth",
            "config":  root / "models" / "dino" / "streak_dinov3_vitl.py",
        },
        {
            "id":      "tiny",
            "size":    "tiny",
            "label":   "DINO Swin-Tiny",
            "dataset": "SatStreaks",
            "weights": root / "weights" / "dino_tiny.pth",
            "config":  root / "models" / "dino" / "streak_codino_swin_t.py",
        },
        {
            "id":      "large",
            "size":    "large",
            "label":   "DINO Swin-Large",
            "dataset": "SatStreaks",
            "weights": root / "weights" / "dino_large.pth",
            "config":  root / "models" / "dino" / "streak_codino_swin_l.py",
        },
    ]


# ---------------------------------------------------------------------------
# Model config selection
# ---------------------------------------------------------------------------

def _select_config(model_size: str) -> Path:
    """Return the MMDetection config path for the given model size.

    Args:
        model_size: 'tiny' | 'large' | 'dinov3_vitb' | 'dinov3_vitl'

    Returns:
        Absolute path to the MMDetection config file.

    Raises:
        EnvironmentError: If a CUDA-only size is requested on a non-CUDA device.
        ValueError: If an unknown model_size string is given.
    """
    root = Path(__file__).resolve().parent.parent
    configs = {
        "tiny":                    root / "models" / "dino" / "streak_codino_swin_t.py",
        "large":                   root / "models" / "dino" / "streak_codino_swin_l.py",
        "dinov3_vitb":             root / "models" / "dino" / "streak_dinov3_vitb.py",
        "dinov3_vitl":             root / "models" / "dino" / "streak_dinov3_vitl.py",
        "dinov3_gt_dm_satstreaks": root / "models" / "dino" / "streak_dinov3_vitb.py",
        "dinov3_vitb_multisource": root / "models" / "dino" / "streak_dinov3_vitb_400px.py",
        "dinov3_vitb_run3":        root / "models" / "dino" / "streak_dinov3_vitb_400px_run3.py",
    }
    if model_size not in configs:
        raise ValueError(
            f"Unknown MODEL_SIZE '{model_size}'. "
            f"Choose from: {sorted(configs)}"
        )
    cuda_only = {"large", "dinov3_vitl"}
    if model_size in cuda_only:
        from inference.device import get_device
        device = get_device()
        if device.type != "cuda":
            raise EnvironmentError(
                f"MODEL_SIZE={model_size} requires a CUDA GPU. "
                f"Current device: {device.type}. Use 'tiny' or 'dinov3_vitb' on Mac."
            )
    return configs[model_size]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(
    config_path: Path,
    weights_path: Path,
    device: Any,
) -> Any:
    """Initialise an MMDetection DINO detector from a config + checkpoint.

    Args:
        config_path: Path to the MMDetection Python config file.
        weights_path: Path to the model checkpoint (.pth file).
        device: torch.device to place the model on.

    Returns:
        MMDetection detector model (mmdet.models.detectors base class).

    Raises:
        FileNotFoundError: If the weights file does not exist.
        ImportError: If mmdet is not installed.
    """
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}\n"
            "Download pretrained weights first:\n"
            "  MODEL_SIZE=tiny python scripts/download_weights.py"
        )
    try:
        from mmdet.apis import init_detector  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "mmdet is required for inference. "
            "Install with: pip install mmdet"
        ) from exc

    logger.debug("Loading model: config=%s  weights=%s  device=%s",
                 config_path, weights_path, device)
    model = init_detector(str(config_path), str(weights_path), device=str(device))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# DINO inference
# ---------------------------------------------------------------------------

def _run_inference(
    model: Any,
    array: "np.ndarray",
    image_size: int,
    confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
    model_name: str = "ml",
) -> list[dict]:
    """Run DINO inference on an image array, return raw bbox detections.

    Args:
        model: Loaded MMDetection model.
        array: uint8 (H, W, 3) image array.
        image_size: Longest edge to which the image is rescaled for inference.
        confidence_threshold: Minimum score to keep a detection.
        model_name: Value to set on the 'method' key of each detection dict
            (e.g. 'dinov3_vitb', 'tiny').

    Returns:
        List of dicts with keys: bbox ([x1, y1, x2, y2] floats), confidence.
    """
    import numpy as np
    try:
        from mmdet.apis import inference_detector  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise ImportError("mmdet is required for inference.") from exc

    # Resize for inference if needed
    h, w = array.shape[:2]
    scale = image_size / max(h, w)
    if scale < 1.0:
        import cv2
        array = cv2.resize(array, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0

    result = inference_detector(model, array)

    detections: list[dict] = []
    try:
        # MMDet 3.x: result is DetDataSample
        pred = result.pred_instances
        bboxes    = pred.bboxes.cpu().numpy()      # (N, 4) float32
        scores    = pred.scores.cpu().numpy()      # (N,)
    except AttributeError:
        # Fallback: older API returned list/tuple
        bboxes_scores = result[0] if isinstance(result, (list, tuple)) else result
        import numpy as np
        bboxes_scores = np.asarray(bboxes_scores)
        if bboxes_scores.ndim < 2 or bboxes_scores.shape[0] == 0:
            return []
        bboxes = bboxes_scores[:, :4]
        scores = bboxes_scores[:, 4]

    for bbox, score in zip(bboxes, scores):
        if float(score) < confidence_threshold:
            continue
        # Scale bbox back to original image coordinates
        x1, y1, x2, y2 = (float(bbox[0] / scale),
                           float(bbox[1] / scale),
                           float(bbox[2] / scale),
                           float(bbox[3] / scale))
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "confidence": float(score),
            "method": model_name,
        })

    logger.debug("DINO: %d raw detections above threshold %.2f",
                 len(detections), confidence_threshold)
    return detections


def _run_classical_detector(
    array: "np.ndarray",
    min_length_px: float = 80.0,
    min_aspect_ratio: float = 5.0,
    max_detections: int = 20,
) -> list[dict]:
    """Find bright elongated streaks with classical image processing.

    This bounded detector complements the DINO model in local development.  It
    uses no learned weights: threshold the bright tail of the normalised image,
    close short gaps, then keep elongated connected components.

    Args:
        array: uint8 RGB image produced by ``FITSLoader``.
        min_length_px: Minimum long-axis component length to keep.
        min_aspect_ratio: Minimum long/short axis ratio to keep.
        max_detections: Maximum number of classical detections to return.

    Returns:
        Detection dictionaries compatible with the rest of the pipeline.
    """
    import cv2
    import numpy as np

    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY) if array.ndim == 3 else array
    finite = gray[np.isfinite(gray)]
    if finite.size == 0:
        return []

    threshold = max(float(np.percentile(finite, 99.5)), 180.0)
    mask = (gray >= threshold).astype("uint8") * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    detections: list[dict] = []

    for label in range(1, n_labels):
        x, y, w, h, area = stats[label]
        if area < 20:
            continue

        long_axis = float(max(w, h))
        short_axis = float(max(1, min(w, h)))
        aspect = long_axis / short_axis
        if long_axis < min_length_px or aspect < min_aspect_ratio:
            continue

        ys, xs = np.where(labels == label)
        if xs.size < 2:
            continue

        coords = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
        center = coords.mean(axis=0)
        centered = coords - center
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        direction = vt[0]
        projections = centered @ direction
        proj_min = float(projections.min())
        proj_max = float(projections.max())
        length = proj_max - proj_min
        if length < min_length_px:
            continue

        start = center + proj_min * direction
        end = center + proj_max * direction
        angle = math.degrees(math.atan2(direction[1], direction[0])) % 180.0
        area_width = max(3.0, float(area) / max(length, 1.0))
        confidence = min(0.99, max(0.2, aspect / 20.0))

        detections.append({
            "bbox": [float(x), float(y), float(x + w), float(y + h)],
            "confidence": confidence,
            "method": "opencv",
            "obb": {
                "cx": float((start[0] + end[0]) / 2.0),
                "cy": float((start[1] + end[1]) / 2.0),
                "w": float(length),
                "h": float(area_width),
                "angle_deg": float(angle),
            },
            "streak_length_px": float(length),
        })

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    logger.debug("Classical detector: %d candidate streak(s)", len(detections))
    return detections[:max_detections]


def _run_tta_inference(
    model: Any,
    array: "np.ndarray",
    image_size: int,
    confidence_threshold: float,
    model_name: str,
) -> list[dict]:
    """Run inference on original, H-flip, and V-flip views; merge all detections.

    Bounding boxes from flipped views are mapped back to original image
    coordinates before merging.  The caller is responsible for NMS to remove
    near-duplicate detections that arise across views.

    Args:
        model: Loaded MMDetection model.
        array: uint8 (H, W, 3) image array in original orientation.
        image_size: Longest-edge resize target for inference.
        confidence_threshold: Minimum score to keep a detection.
        model_name: Value for the 'method' key on each detection dict.

    Returns:
        Combined list of detection dicts from all three views.
    """
    import numpy as np

    h, w = array.shape[:2]
    all_dets: list[dict] = []

    all_dets.extend(_run_inference(model, array, image_size, confidence_threshold, model_name))

    hflip = array[:, ::-1, :].copy()
    for det in _run_inference(model, hflip, image_size, confidence_threshold, model_name):
        x1, y1, x2, y2 = det["bbox"]
        det["bbox"] = [w - x2, y1, w - x1, y2]
        all_dets.append(det)

    logger.debug("TTA: %d detections from 2 augmented views", len(all_dets))
    return all_dets


def _downsample_fits_image_for_astride(fits_image: Any, max_pixels: int) -> tuple[Any, float]:
    """Return a downsampled FITSImage-like object and its linear scale.

    Args:
        fits_image: Parsed FITSImage or compatible object.
        max_pixels: Target maximum pixel count for ASTRiDE input.

    Returns:
        ``(downsampled_image, scale)`` where ``scale`` maps original pixels to
        downsampled pixels. If downsampling is not needed, returns the original
        image and scale 1.0.
    """
    import cv2
    import numpy as np

    image_pixels = int(fits_image.width_px) * int(fits_image.height_px)
    if max_pixels <= 0 or image_pixels <= max_pixels:
        return fits_image, 1.0

    min_scale = float(os.environ.get(
        "ASTRIDE_DOWNSAMPLE_MIN_SCALE",
        str(_DEFAULT_ASTRIDE_DOWNSAMPLE_MIN_SCALE),
    ))
    scale = math.sqrt(max_pixels / image_pixels)
    scale = max(min_scale, min(1.0, scale))

    new_width = max(1, int(round(int(fits_image.width_px) * scale)))
    new_height = max(1, int(round(int(fits_image.height_px) * scale)))
    data = np.asarray(fits_image.data, dtype=np.float32)
    resized = cv2.resize(
        data,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32, copy=False)

    downsampled = copy.copy(fits_image)
    downsampled.data = resized
    downsampled.width_px = new_width
    downsampled.height_px = new_height
    if hasattr(downsampled, "header") and downsampled.header is not None:
        downsampled.header = downsampled.header.copy()
        downsampled.header["NAXIS1"] = new_width
        downsampled.header["NAXIS2"] = new_height

    logger.info(
        "Downsampled ASTRiDE input %dx%d -> %dx%d (scale=%.3f)",
        fits_image.width_px,
        fits_image.height_px,
        new_width,
        new_height,
        scale,
    )
    return downsampled, scale


def _scale_astride_detection(det: Any, scale: float) -> Any:
    """Scale an ASTRiDE StreakDetection from downsampled to original pixels.

    Args:
        det: StreakDetection or compatible object.
        scale: Linear scale used to downsample the input image.

    Returns:
        A detection object in original image pixel coordinates.
    """
    if scale == 1.0:
        return det

    scaled = copy.copy(det)
    for attr in ("x_start", "y_start", "x_end", "y_end", "x_center", "y_center"):
        setattr(scaled, attr, float(getattr(det, attr)) / scale)
    scaled.length_px = float(det.length_px) / scale
    scaled.width_px = float(det.width_px) / scale
    scaled.area_px = float(det.area_px) / (scale * scale)
    return scaled


def _run_astride_detector(
    fits_path: Path,
    min_length_px: float = 20.0,
    contour_threshold: float = 3.0,
) -> list[dict]:
    """Run the Phase-0 ASTRiDE detector on the raw FITS file.

    Calls src.detection.classical_detector.detect_streaks, then converts each
    StreakDetection to the standard pipeline dict format so it can enter the
    shared NMS pool alongside DINO and OpenCV detections.

    Args:
        fits_path: Path to the FITS file (same path passed to run()).
        min_length_px: Minimum streak length forwarded to ASTRiDE.
        contour_threshold: ASTRiDE sigma threshold for contour search.

    Returns:
        Detection dicts compatible with the rest of the pipeline, each with
        method='astride'.
    """
    try:
        from src.ingest.fits_parser import parse_fits
        from src.detection.classical_detector import detect_streaks
    except ImportError:
        logger.debug("ASTRiDE detector unavailable (src package not importable); skipping")
        return []

    try:
        fits_image = parse_fits(fits_path)
        max_pixels = int(os.environ.get("ASTRIDE_MAX_PIXELS", str(_DEFAULT_ASTRIDE_MAX_PIXELS)))
        image_pixels = int(fits_image.width_px) * int(fits_image.height_px)
        if max_pixels > 0 and image_pixels > max_pixels:
            downsample_max_pixels = int(os.environ.get(
                "ASTRIDE_DOWNSAMPLE_MAX_PIXELS",
                str(_DEFAULT_ASTRIDE_DOWNSAMPLE_MAX_PIXELS),
            ))
            if downsample_max_pixels <= 0:
                logger.warning(
                    "Running ASTRiDE at full resolution for %s: image has %d "
                    "pixels, above ASTRIDE_MAX_PIXELS=%d, and downsampling is "
                    "disabled",
                    fits_path.name,
                    image_pixels,
                    max_pixels,
                )
                astride_scale = 1.0
            else:
                fits_image, astride_scale = _downsample_fits_image_for_astride(
                    fits_image,
                    downsample_max_pixels,
                )
        else:
            astride_scale = 1.0
        streak_dets = detect_streaks(
            fits_image,
            contour_threshold=contour_threshold,
            min_length_px=max(1.0, min_length_px * astride_scale),
        )
        streak_dets = [
            _scale_astride_detection(d, astride_scale)
            for d in streak_dets
        ]
    except Exception:
        logger.exception("ASTRiDE detector failed on %s", fits_path)
        return []

    detections: list[dict] = []
    for d in streak_dets:
        aspect = d.length_px / max(d.width_px, 1.0)
        confidence = min(0.99, max(0.2, aspect / 20.0))
        detections.append({
            "bbox": [
                float(min(d.x_start, d.x_end)),
                float(min(d.y_start, d.y_end)),
                float(max(d.x_start, d.x_end)),
                float(max(d.y_start, d.y_end)),
            ],
            "confidence": confidence,
            "method": "astride",
            "obb": {
                "cx": float(d.x_center),
                "cy": float(d.y_center),
                "w": float(d.length_px),
                "h": float(d.width_px),
                "angle_deg": float(d.angle_deg),
            },
            "streak_length_px": float(d.length_px),
        })

    max_detections = int(os.environ.get(
        "ASTRIDE_MAX_DETECTIONS",
        str(_DEFAULT_ASTRIDE_MAX_DETECTIONS),
    ))
    if max_detections > 0 and len(detections) > max_detections:
        before_count = len(detections)
        detections = sorted(
            detections,
            key=lambda d: d.get("confidence", 0.0),
            reverse=True,
        )[:max_detections]
        logger.warning(
            "Capped ASTRiDE detections from %d to %d; set "
            "ASTRIDE_MAX_DETECTIONS=0 to process all",
            before_count,
            max_detections,
        )

    logger.debug("ASTRiDE detector: %d candidate streak(s)", len(detections))
    return detections


def _lower_astride_only_confidence(detections: list[dict]) -> list[dict]:
    """Set ASTRiDE-only grouped streaks to a conservative confidence.

    ASTRiDE-only detections are useful enough to show, but they should not carry
    the high aspect-ratio confidence used internally for ranking candidates.
    Once detections have been grouped across methods, this helper can tell
    whether an ASTRiDE candidate has independent support and lower only the
    ASTRiDE-only groups.

    Args:
        detections: Grouped detections with ``streak_id`` and ``method`` keys.

    Returns:
        Detections with ASTRiDE-only group confidence set to 0.30.
    """
    methods_by_group: dict[Any, set[str]] = {}
    for det in detections:
        group_key = det.get("streak_id")
        method = str(det.get("method", "")).lower()
        methods_by_group.setdefault(group_key, set()).add(method)

    lowered = 0
    for det in detections:
        if methods_by_group.get(det.get("streak_id"), set()) == {"astride"}:
            det["confidence"] = 0.30
            lowered += 1
    if lowered:
        logger.debug("Set %d ASTRiDE-only detection(s) to 0.30 confidence", lowered)
    return detections


# ---------------------------------------------------------------------------
# Pixel → sky coordinate conversion
# ---------------------------------------------------------------------------

def _pixel_to_sky(
    cx_px: float,
    cy_px: float,
    wcs: Any,
) -> tuple[float | None, float | None]:
    """Convert a pixel coordinate to RA/Dec using the image WCS.

    Args:
        cx_px: X pixel coordinate (column, 0-based).
        cy_px: Y pixel coordinate (row, 0-based).
        wcs: astropy.wcs.WCS object, or None if no WCS available.

    Returns:
        (ra_deg, dec_deg) in degrees, or (None, None) if conversion fails
        or WCS is None.
    """
    if wcs is None:
        return None, None
    try:
        import numpy as np
        sky = wcs.all_pix2world(np.array([[cx_px, cy_px]]), 0)
        ra_deg  = float(sky[0, 0]) % 360.0
        dec_deg = float(sky[0, 1])
        return ra_deg, dec_deg
    except Exception as exc:
        logger.debug("WCS pixel→sky failed at (%.1f, %.1f): %s", cx_px, cy_px, exc)
        return None, None


# ---------------------------------------------------------------------------
# Angle estimate from bbox when Radon is skipped
# ---------------------------------------------------------------------------

def _angle_from_bbox(bbox: list[float]) -> float:
    """Estimate streak angle from the diagonal of an axis-aligned bbox.

    Uses atan2(height, width) as the seed angle for Radon refinement.
    This is more accurate than snapping to 0/90° for diagonal streaks.

    Args:
        bbox: [x1, y1, x2, y2]

    Returns:
        Estimated angle in degrees [0, 180).
    """
    x1, y1, x2, y2 = bbox
    bw = abs(x2 - x1)
    bh = abs(y2 - y1)
    if bw < 1e-6 and bh < 1e-6:
        return 0.0
    return math.degrees(math.atan2(bh, bw)) % 180.0


# ---------------------------------------------------------------------------
# Public model loader (for batch inference — load once, run many)
# ---------------------------------------------------------------------------

def load_model(
    model_size: str | None = None,
    weights_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """Load the DINO detector once for reuse across multiple images.

    Call this before a batch inference loop and pass the returned values to
    ``run()`` via the ``model`` and ``inference_device`` parameters to avoid
    reloading the checkpoint on every image.

    Args:
        model_size: 'tiny' or 'large'.  Defaults to the MODEL_SIZE env var
            (or 'tiny' if unset).
        weights_path: Explicit path to a .pth checkpoint.  Defaults to the
            MODEL_WEIGHTS env var, then ``weights/dino_{model_size}.pth``.

    Returns:
        (model, inference_device) — pass both to ``run()``.

    Raises:
        FileNotFoundError: If the weights file does not exist.
        EnvironmentError: If model_size='large' on a non-CUDA device.
    """
    if model_size is None:
        model_size = os.environ.get("MODEL_SIZE", "tiny")

    from inference.device import get_device
    device = get_device()

    import torch as _torch
    inference_device = device
    if device.type == "mps" and not _torch.cuda.is_available():
        inference_device = _torch.device("cpu")
        logger.debug("MPS device detected — forcing DINO inference to CPU")

    config_path = _select_config(model_size)

    if weights_path is None:
        weights_env = os.environ.get("MODEL_WEIGHTS", "")
        if weights_env:
            weights_path = Path(weights_env)
        else:
            root = Path(__file__).resolve().parent.parent
            # DINOv3 variants have dedicated checkpoint directories
            _dinov3_defaults = {
                "dinov3_vitb":             root / "weights" / "dinov3_vitb_augmented" / "best_coco_bbox_mAP_epoch_10.pth",
                "dinov3_vitl":             root / "weights" / "run_5070ti_dinov3_vitl" / "best_coco_bbox_mAP_epoch_50.pth",
                "dinov3_gt_dm_satstreaks": root / "weights" / "run_gt_dm_satstreaks_dinov3_vitb" / "best_coco_bbox_mAP_epoch_4.pth",
                # Run 3: best.pth is a symlink updated each night to the current best checkpoint.
                "dinov3_vitb_run3":        root / "weights" / "run3_cold_nodm" / "best.pth",
            }
            if model_size in _dinov3_defaults:
                weights_path = _dinov3_defaults[model_size]
            else:
                weights_path = root / "weights" / f"dino_{model_size}.pth"

    model = _load_model(config_path, Path(weights_path), inference_device)
    return model, inference_device


# ---------------------------------------------------------------------------
# Multi-model configuration helpers
# ---------------------------------------------------------------------------

def resolve_model_specs() -> list[dict]:
    """Return normalized DINO model specs from ARGUS_MODEL_CONFIGS or MODEL_SIZE fallback.

    Returns:
        List of dicts with keys: id, size, weights, label, dataset.
        Safe to call without loading any model or weights.
    """
    import json as _json

    raw = os.environ.get("ARGUS_MODEL_CONFIGS", "")
    root = Path(__file__).resolve().parent.parent
    if raw:
        specs = _json.loads(raw)
        result = []
        for s in specs:
            w = Path(s["weights"])
            weights_str = str(w if w.is_absolute() else root / w)
            result.append({
                "id":      s["id"],
                "size":    s["size"],
                "weights": weights_str,
                "label":   s.get("label", s["id"]),
                "dataset": s.get("dataset", ""),
            })
        return result

    # Single-model fallback
    model_size = os.environ.get("MODEL_SIZE", "tiny")
    weights_env = os.environ.get("MODEL_WEIGHTS", "")
    _dinov3_defaults: dict[str, Path] = {
        "dinov3_vitb":             root / "weights" / "dinov3_vitb_augmented" / "best_coco_bbox_mAP_epoch_10.pth",
        "dinov3_vitl":             root / "weights" / "run_5070ti_dinov3_vitl" / "best_coco_bbox_mAP_epoch_50.pth",
        "dinov3_gt_dm_satstreaks": root / "weights" / "run_gt_dm_satstreaks_dinov3_vitb" / "best_coco_bbox_mAP_epoch_4.pth",
        "dinov3_vitb_multisource": root / "weights" / "run3_cold_nodm" / "best.pth",
        # Run 3: best.pth symlink is updated each night to the current best checkpoint.
        "dinov3_vitb_run3":        root / "weights" / "run3_cold_nodm" / "best.pth",
    }
    _meta: dict[str, tuple[str, str]] = {
        "tiny":                    ("DINO Swin-Tiny",                  "SatStreaks"),
        "large":                   ("DINO Swin-Large",                 "SatStreaks"),
        "dinov3_vitb":             ("DINOv3 ViT-B",                    "SatStreaks+GTImages"),
        "dinov3_vitl":             ("DINOv3 ViT-L",                    "SatStreaks+GTImages"),
        "dinov3_vitb_multisource": ("DINOv3 Base - Multi-source",      "SatStreaks+BrentImages+Frigate"),
        "dinov3_vitb_run3":        ("DINOv3 ViT-B Run 3 (cold-start, nodm)", "SatStreaks+BrentImages+Frigate"),
    }
    label, dataset = _meta.get(model_size, (model_size, ""))
    if weights_env:
        weights_str = weights_env
    elif model_size in _dinov3_defaults:
        weights_str = str(_dinov3_defaults[model_size])
    else:
        weights_str = str(root / "weights" / f"dino_{model_size}.pth")
    return [{"id": model_size, "size": model_size, "weights": weights_str,
             "label": label, "dataset": dataset}]


def load_models() -> list[tuple[Any, Any, dict]]:
    """Load all DINO model checkpoints defined by ARGUS_MODEL_CONFIGS or MODEL_SIZE.

    Returns:
        List of (model, inference_device, spec) tuples ready for run_with_array().

    Raises:
        FileNotFoundError: If any weights file does not exist.
        EnvironmentError: If a CUDA-only size is requested on a non-CUDA device.
    """
    from inference.device import get_device
    import torch as _torch

    device = get_device()
    inference_device = device
    if device.type == "mps" and not _torch.cuda.is_available():
        inference_device = _torch.device("cpu")
        logger.debug("MPS device detected — forcing DINO inference to CPU")

    result = []
    for spec in resolve_model_specs():
        config_path = _select_config(spec["size"])
        model = _load_model(config_path, Path(spec["weights"]), inference_device)
        logger.info("Loaded model %s from %s on %s", spec["id"], spec["weights"], inference_device)
        result.append((model, inference_device, spec))
    return result


def get_detector_statuses() -> list[dict]:
    """Return availability metadata for every detector without loading any model.

    Checks weights file existence and required imports; never initialises a model.

    Returns:
        List of dicts with keys: id, name, type, dataset, status.
        status is one of 'active' | 'no_weights' | 'unavailable'.
    """
    statuses: list[dict] = []

    # All registered DINO variants — show everything, active or not.
    seen_ids: set[str] = set()
    for entry in _model_registry():
        seen_ids.add(entry["id"])
        weight_ok = Path(entry["weights"]).exists()
        statuses.append({
            "id":      entry["id"],
            "name":    entry["label"],
            "type":    "ml",
            "dataset": entry["dataset"],
            "status":  "active" if weight_ok else "no_weights",
        })

    # Any extra models injected via ARGUS_MODEL_CONFIGS that aren't in the registry.
    for spec in resolve_model_specs():
        if spec["id"] not in seen_ids:
            weight_ok = Path(spec["weights"]).exists()
            statuses.append({
                "id":      spec["id"],
                "name":    spec["label"],
                "type":    "ml",
                "dataset": spec["dataset"],
                "status":  "active" if weight_ok else "no_weights",
            })

    # Classical / ASTRiDE (share the same import gate)
    try:
        from src.detection.classical_detector import detect_streaks as _ds  # noqa: F401
        classical_status = "active"
    except ImportError:
        classical_status = "unavailable"

    statuses.append({
        "id": "classical", "name": "OpenCV Morphological",
        "type": "classical", "dataset": "—", "status": classical_status,
    })
    statuses.append({
        "id": "astride", "name": "ASTRiDE",
        "type": "classical", "dataset": "—", "status": classical_status,
    })
    return statuses


# ---------------------------------------------------------------------------
# Parallel detector runner
# ---------------------------------------------------------------------------

def _run_all_detectors(
    models_with_specs: list[tuple[Any, Any, dict]],
    array: "np.ndarray",
    fits_path: Path,
    image_size: int,
    confidence_threshold: float,
    tta_enabled: bool,
    enabled_detectors: set[str] | None,
) -> dict[str, list[dict]]:
    """Run all enabled detectors concurrently via a ThreadPoolExecutor.

    DINO models and secondary detectors (classical, ASTRiDE) are
    submitted in parallel.  Detectors whose ID is absent from
    *enabled_detectors* are skipped entirely (None = all enabled).

    Args:
        models_with_specs: List of (model, device, spec) from load_models().
        array: uint8 (H, W, 3) image array.
        fits_path: Path to the FITS file (needed by ASTRiDE).
        image_size: Inference resolution for DINO.
        confidence_threshold: Minimum detection score.
        tta_enabled: Whether to run 3x TTA inference for each DINO model.
        enabled_detectors: Set of detector IDs to run; None means all.

    Returns:
        Dict mapping detector ID → raw detection list.
    """
    def _timed_detector(det_id: str, fn: Any, *args: Any) -> tuple[list[dict], float]:
        start = time.perf_counter()
        detections = fn(*args)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return detections, elapsed_ms

    def _enabled(det_id: str) -> bool:
        if enabled_detectors is not None:
            return det_id in enabled_detectors
        # Default (None = "all"): skip astride unless explicitly opted in.
        if det_id == "astride":
            return _ASTRIDE_ENABLED_BY_DEFAULT
        return True

    tasks: dict[Any, str] = {}
    results: dict[str, list[dict]] = {}
    enabled_label = (
        "all"
        if enabled_detectors is None
        else ",".join(sorted(enabled_detectors)) or "none"
    )
    logger.info("Enabled detectors: %s", enabled_label)

    n_workers = len(models_with_specs) + 5
    with ThreadPoolExecutor(max_workers=max(1, n_workers)) as pool:
        # DINO models
        for model, _dev, spec in models_with_specs:
            if not _enabled(spec["id"]):
                results[spec["id"]] = []
                continue
            fn = _run_tta_inference if tta_enabled else _run_inference
            f = pool.submit(
                _timed_detector,
                spec["id"],
                fn,
                model,
                array,
                image_size,
                confidence_threshold,
                spec["id"],
            )
            tasks[f] = spec["id"]

        # Secondary detectors
        if _enabled("classical"):
            tasks[pool.submit(_timed_detector, "classical", _run_classical_detector, array)] = "classical"
        else:
            results["classical"] = []

        if _enabled("astride"):
            tasks[pool.submit(_timed_detector, "astride", _run_astride_detector, fits_path)] = "astride"
        else:
            results["astride"] = []

        for f in as_completed(tasks):
            key = tasks[f]
            try:
                dets, elapsed_ms = f.result()
                for det in dets:
                    det.setdefault("method", key)
                results[key] = dets
                logger.info(
                    "detector_timing detector=%s elapsed_ms=%.1f detections=%d",
                    key, elapsed_ms, len(dets),
                )
            except Exception:
                logger.exception("Detector %s failed", key)
                results[key] = []

    return results


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_with_array(
    fits_path: str | Path,
    fast: bool = False,
    model: Any | None = None,
    inference_device: Any | None = None,
    models: list[tuple[Any, Any, dict]] | None = None,
    enabled_detectors: set[str] | None = None,
    raw_mode: bool = False,
) -> tuple[list[dict], "np.ndarray"]:
    """Run the full ARGUS inference pipeline on a single FITS image.

    Args:
        fits_path: Path to the input FITS file.
        fast: If True, skip Radon refinement, cross-ID, and DB write.
            Equivalent to setting FAST_MODE=true.  Uses image_size=256.
            Target: <60 s wall time per image on Mac.
        model: Pre-loaded MMDetection model from ``load_model()``.  When
            provided, the checkpoint is not reloaded — use this for batch
            eval loops to avoid loading 187 MB per image.
        inference_device: The device the pre-loaded model lives on.
            Required when ``model`` is passed; ignored otherwise.
        models: List of (model, device, spec) from ``load_models()``.
            Takes priority over ``model``/``inference_device`` when provided.
            Enables running multiple DINO checkpoints in parallel.
        enabled_detectors: Set of detector IDs to run (e.g. {"classical",
            "astride"}).  None (default) enables all detectors.
        raw_mode: If True, skip Radon angle refinement, OBB extent tracing,
            per-detector NMS, and cross-detector grouping.  Every detection
            from every model is returned as a unique streak with the raw OBB
            produced by that detector.  If False (default), the full
            postprocessing pipeline runs.

    Returns:
        Tuple of (detections, array).  ``array`` is the uint8 (H, W, 3) image
        produced by FITSLoader — callers can pass it to downstream rendering
        steps to avoid a second FITS parse.  Use ``run()`` for the list-only
        interface.  Each detection dict has keys:
          confidence        — float, DINO score 0–1
          bbox              — [x1, y1, x2, y2] pixel coords
          obb               — {cx, cy, w, h, angle_deg}
          streak_length_px  — float, long axis of OBB
          ra_tip1_deg       — float or None, sky RA of OBB tip 1
          dec_tip1_deg      — float or None, sky Dec of OBB tip 1
          ra_tip2_deg       — float or None, sky RA of OBB tip 2
          dec_tip2_deg      — float or None, sky Dec of OBB tip 2
          quality_flag      — int 0–4 (0=good, 1=edge, 2=low_conf,
                               3=too_short, 4=no_wcs)
          edge_clipped      — bool, True when a tip touches/crosses an image
                               border and catalogue scoring should treat the
                               streak as a partial visible segment
          streak_direction_swapped — bool, True if tip1/tip2 were swapped
                               to assign start→end direction (set only when
                               exposure_time is in the FITS header)
          identifications   — list of up to 3 {satellite_name, norad_id,
                               confidence, separation_arcsec, atrk_arcsec,
                               xtrk_arcsec, rank} dicts (empty in fast mode)

    Raises:
        FileNotFoundError: If the FITS file or model weights are not found.
        ValueError: If MODEL_SIZE is unknown.
        EnvironmentError: If MODEL_SIZE=large on a non-CUDA machine.
    """
    fits_path = Path(fits_path)

    # Respect FAST_MODE env var
    if os.environ.get("FAST_MODE", "").lower() == "true":
        fast = True

    model_size    = os.environ.get("MODEL_SIZE", "tiny")
    weights_env   = os.environ.get("MODEL_WEIGHTS", "")
    confidence_threshold = float(
        os.environ.get("CONFIDENCE_THRESHOLD", str(_DEFAULT_CONFIDENCE_THRESHOLD))
    )
    tta_enabled = os.environ.get("TTA_ENABLED", "").lower() in {"1", "true", "yes"}

    # --- 1. Load FITS --------------------------------------------------------
    t0 = time.perf_counter()
    from inference.fits_loader import FITSLoader
    loader = FITSLoader()
    fits_data = loader.load(fits_path)
    array    = fits_data["array"]       # uint8 (H, W, 3)
    wcs      = fits_data["wcs"]
    obs_time = fits_data.get("obs_time")
    fits_load_ms = (time.perf_counter() - t0) * 1000
    logger.debug("fits_load_ms=%.1f", fits_load_ms)
    logger.info("fits_load_ms=%.1f", fits_load_ms)

    # --- 2. Build models list then run all detectors in parallel -------------
    t1 = time.perf_counter()
    from inference.device import get_device, get_device_config
    device     = get_device()
    dev_config = get_device_config()
    if fast:
        image_size = _FAST_IMAGE_SIZE
    else:
        # Use at least _MIN_INFERENCE_IMAGE_SIZE so that streaks in large
        # sensor images (~6 k px) are not crushed to ~30 px before detection.
        image_size = max(dev_config["image_size"], _MIN_INFERENCE_IMAGE_SIZE)

    # Resolve which DINO model(s) to run:
    #  - models (plural) is the preferred multi-model path from load_models()
    #  - model (singular) is the legacy single-model path from load_model()
    #  - if neither is given, load inline from MODEL_SIZE env
    if models is not None:
        _models_with_specs = models
    elif model is not None:
        _spec = {"id": model_size, "size": model_size, "label": model_size, "dataset": ""}
        if inference_device is None:
            inference_device = device
        _models_with_specs = [(model, inference_device, _spec)]
    else:
        # Load inline — same logic as load_model() but embedded here for the
        # legacy single-image call path (e.g. CLI eval scripts).
        config_path = _select_config(model_size)
        if weights_env:
            weights_path = Path(weights_env)
        else:
            root = Path(__file__).resolve().parent.parent
            _dinov3_inline: dict[str, Path] = {
                "dinov3_vitb":             root / "weights" / "dinov3_vitb_augmented" / "best_coco_bbox_mAP_epoch_10.pth",
                "dinov3_vitl":             root / "weights" / "run_5070ti_dinov3_vitl" / "best_coco_bbox_mAP_epoch_50.pth",
                "dinov3_gt_dm_satstreaks": root / "weights" / "run_gt_dm_satstreaks_dinov3_vitb" / "best_coco_bbox_mAP_epoch_4.pth",
                "dinov3_vitb_run3":        root / "weights" / "run3_cold_nodm" / "best.pth",
            }
            if model_size in _dinov3_inline:
                weights_path = _dinov3_inline[model_size]
            else:
                weights_path = root / "weights" / f"dino_{model_size}.pth"

        # DINO multi-scale deformable attention exceeds MPS's 4 GB per-allocation
        # limit.  Force CPU on Mac until a memory-efficient MPS path is available.
        import torch as _torch
        inference_device = device
        if device.type == "mps" and not _torch.cuda.is_available():
            inference_device = _torch.device("cpu")
            logger.debug("MPS device detected — forcing DINO inference to CPU")

        model = _load_model(config_path, weights_path, inference_device)
        _meta = {"id": model_size, "size": model_size, "label": model_size, "dataset": ""}
        _models_with_specs = [(model, inference_device, _meta)]

    # Run all detectors (DINO variants + classical + ASTRiDE) in parallel.
    all_det_results = _run_all_detectors(
        _models_with_specs, array, fits_path, image_size,
        confidence_threshold, tta_enabled, enabled_detectors,
    )

    # Collect results — DINO detections from all models are merged before NMS.
    raw_dets: list[dict] = []
    for _, _, _s in _models_with_specs:
        raw_dets.extend(all_det_results.get(_s["id"], []))
    max_dino_post = int(os.environ.get(
        "DINO_MAX_POSTPROCESS_DETECTIONS",
        str(_DEFAULT_DINO_MAX_POSTPROCESS_DETECTIONS),
    ))
    if max_dino_post > 0 and len(raw_dets) > max_dino_post:
        before_count = len(raw_dets)
        raw_dets = sorted(
            raw_dets,
            key=lambda d: d.get("confidence", 0.0),
            reverse=True,
        )[:max_dino_post]
        logger.warning(
            "Capped DINO postprocess candidates from %d to %d; set "
            "DINO_MAX_POSTPROCESS_DETECTIONS=0 to process all",
            before_count,
            max_dino_post,
        )

    classical_dets = all_det_results.get("classical", [])
    astride_dets   = all_det_results.get("astride", [])

    inference_ms = (time.perf_counter() - t1) * 1000
    logger.debug("inference_ms=%.1f  raw_dets=%d  tta=%s  threshold=%.2f",
                 inference_ms, len(raw_dets), tta_enabled, confidence_threshold)
    logger.info(
        "inference_ms=%.1f raw_dets=%d tta=%s threshold=%.2f",
        inference_ms, len(raw_dets), tta_enabled, confidence_threshold,
    )

    # --- 3. Postprocess: angle refinement + OBB (full or raw-mode) -----------
    t2 = time.perf_counter()
    import numpy as np
    from inference.postprocess import (
        bbox_to_obb, classify_detection_quality,
        refine_angle, extend_obb_to_streak_extent,
        nms_detections, group_detections, fuse_group_geometries,
    )

    h_img = array.shape[0]
    w_img = array.shape[1]

    if raw_mode:
        # Raw mode: no Radon, no extent tracing, no NMS, no grouping.
        # Every detection from every model is an independent unique streak.
        for det in raw_dets:
            if det.get("obb") is None:
                seed_angle = _angle_from_bbox(det["bbox"])
                det["obb"] = bbox_to_obb(det["bbox"], seed_angle)
            det["streak_length_px"] = float(det["obb"]["w"])

        for det in classical_dets + astride_dets:
            if det.get("obb"):
                det.setdefault("streak_length_px", float(det["obb"]["w"]))
            elif det.get("bbox"):
                seed_angle = _angle_from_bbox(det["bbox"])
                det["obb"] = bbox_to_obb(det["bbox"], seed_angle)
                det["streak_length_px"] = float(det["obb"]["w"])

        detections = raw_dets + classical_dets + astride_dets
        for idx, det in enumerate(detections):
            det["streak_id"] = idx + 1

    else:
        # Full mode: Radon angle refinement, extent tracing, per-detector NMS,
        # cross-detector grouping, and geometry fusion.
        _gray_f32 = np.asarray(array, dtype=np.float32)
        if _gray_f32.ndim == 3:
            _gray_f32 = _gray_f32.mean(axis=2)
        _bg = float(np.median(_gray_f32))
        _extent_threshold = _bg + 3.0 * float(_gray_f32.std())

        for det in raw_dets:
            x1, y1, x2, y2 = det["bbox"]
            px1 = max(0, int(math.floor(x1)))
            py1 = max(0, int(math.floor(y1)))
            px2 = min(w_img, int(math.ceil(x2)))
            py2 = min(h_img, int(math.ceil(y2)))
            crop = array[py1:py2, px1:px2]
            seed_angle = _angle_from_bbox(det["bbox"])
            initial_obb = bbox_to_obb(det["bbox"], seed_angle)
            angle_range = float(os.environ.get(
                "RADON_ANGLE_SEARCH_RANGE",
                str(_DEFAULT_RADON_ANGLE_SEARCH_RANGE),
            ))
            angle = refine_angle(crop, initial_obb, angle_search_range=angle_range)
            obb = bbox_to_obb(det["bbox"], angle)
            obb = extend_obb_to_streak_extent(array, obb, _gray=_gray_f32, _threshold=_extent_threshold)
            det["obb"] = obb
            det["streak_length_px"] = float(obb["w"])

        for det in classical_dets + astride_dets:
            if det.get("obb"):
                obb = extend_obb_to_streak_extent(array, det["obb"], _gray=_gray_f32, _threshold=_extent_threshold)
                det["obb"] = obb
                det["streak_length_px"] = float(obb["w"])

        raw_dets             = nms_detections(raw_dets,             iou_threshold=0.5)
        classical_dets       = nms_detections(classical_dets,       iou_threshold=0.5)
        astride_dets         = nms_detections(astride_dets,         iou_threshold=0.5)
        combined = (
            raw_dets + classical_dets + astride_dets
        )
        detections = group_detections(combined, iou_threshold=0.5)
        detections = fuse_group_geometries(detections)
        detections = _lower_astride_only_confidence(detections)

    postprocess_ms = (time.perf_counter() - t2) * 1000
    logger.debug("postprocess_ms=%.1f  dets=%d  raw_mode=%s", postprocess_ms, len(detections), raw_mode)
    logger.info("postprocess_ms=%.1f dets=%d raw_mode=%s", postprocess_ms, len(detections), raw_mode)

    # --- 4. WCS: pixel → sky coordinates (both streak endpoints) ------------
    for det in detections:
        obb = det["obb"]
        cx, cy = obb["cx"], obb["cy"]
        half   = obb["w"] / 2.0
        angle_rad = math.radians(obb["angle_deg"])
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

        tip1_x = cx - half * cos_a
        tip1_y = cy - half * sin_a
        tip2_x = cx + half * cos_a
        tip2_y = cy + half * sin_a

        det["ra_tip1_deg"],  det["dec_tip1_deg"]  = _pixel_to_sky(tip1_x, tip1_y, wcs)
        det["ra_tip2_deg"],  det["dec_tip2_deg"]  = _pixel_to_sky(tip2_x, tip2_y, wcs)

        # Quality flag — assigned after sky coords are available
        det["quality_flag"] = classify_detection_quality(
            det,
            image_shape=(array.shape[0], array.shape[1]),
        )

    # --- 5. Cross-identification (skipped in fast mode) ----------------------
    t3 = time.perf_counter()
    if fast:
        for det in detections:
            det["identifications"] = []
        crossid_ms = 0.0
    elif wcs is None:
        for det in detections:
            det["identifications"] = []
        crossid_ms = 0.0
        logger.info(
            "Skipping cross-ID: plate solve failed or no WCS available for %s",
            fits_path.name,
        )
    elif not any(
        det.get("ra_tip1_deg") is not None or det.get("ra_tip2_deg") is not None
        for det in detections
    ):
        for det in detections:
            det["identifications"] = []
        crossid_ms = 0.0
        logger.info("Skipping cross-ID: no detections have sky coordinates")
    else:
        max_crossid = int(os.environ.get(
            "CROSSID_MAX_DETECTIONS",
            str(_DEFAULT_CROSSID_MAX_DETECTIONS),
        ))
        if max_crossid <= 0:
            for det in detections:
                det["identifications"] = []
            crossid_ms = 0.0
            logger.info(
                "Skipping cross-ID: CROSSID_MAX_DETECTIONS=%d. Set a positive "
                "value to identify top detections.",
                max_crossid,
            )
        else:
            from inference.crossid import cross_identify
            # Extract observer location from FITS header (may be None)
            observer_lat = fits_data.get("observer_lat") or 0.0
            observer_lon = fits_data.get("observer_lon") or 0.0
            observer_alt = fits_data.get("observer_alt_m") or 0.0
            if obs_time is None:
                obs_time = datetime.now(tz=timezone.utc)
            crossid_candidates = [
                det for det in detections
                if det.get("ra_tip1_deg") is not None or det.get("ra_tip2_deg") is not None
            ]
            if len(crossid_candidates) > max_crossid:
                crossid_candidates = sorted(
                    crossid_candidates,
                    key=lambda d: d.get("confidence", 0.0),
                    reverse=True,
                )[:max_crossid]
                logger.warning(
                    "Capped cross-ID to top %d/%d detection(s)",
                    max_crossid,
                    len(detections),
                )
            crossid_ids = {id(det) for det in crossid_candidates}
            for det in detections:
                if id(det) not in crossid_ids:
                    det["identifications"] = []
            cross_identify(
                crossid_candidates, obs_time,
                observer_lat, observer_lon, observer_alt,
                exposure_time=fits_data.get("exposure_time"),
            )
            crossid_ms = (time.perf_counter() - t3) * 1000

    logger.debug("crossid_ms=%.1f", crossid_ms)
    logger.info("crossid_ms=%.1f", crossid_ms)
    logger.debug("db_write_ms=0.0  (Phase 4 — not yet implemented)")

    logger.info(
        "Pipeline complete: %d detections  "
        "[load=%.0fms  infer=%.0fms  post=%.0fms  crossid=%.0fms]",
        len(detections), fits_load_ms, inference_ms, postprocess_ms, crossid_ms,
    )
    return detections, array


def run(
    fits_path: str | Path,
    fast: bool = False,
    model: Any | None = None,
    inference_device: Any | None = None,
    raw_mode: bool = False,
) -> list[dict]:
    """Run the full ARGUS inference pipeline on a single FITS image.

    Thin wrapper around ``run_with_array()`` for callers that only need the
    detection list.  See ``run_with_array()`` for full parameter documentation.
    """
    detections, _ = run_with_array(
        fits_path, fast=fast, model=model, inference_device=inference_device,
        raw_mode=raw_mode,
    )
    return detections


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(
        description="Run ARGUS inference pipeline on a FITS image."
    )
    parser.add_argument("--image", required=True, help="Path to FITS file")
    parser.add_argument("--fast", action="store_true",
                        help="Skip Radon refinement and cross-ID (fast mode)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable DEBUG logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        results = run(fits_path=args.image, fast=args.fast)
        print(f"\nDetected {len(results)} streak(s):\n")
        for i, det in enumerate(results, 1):
            obb = det.get("obb", {})
            ids = det.get("identifications", [])
            best_id = ids[0]["satellite_name"] if ids else "—"
            print(
                f"  [{i}] conf={det['confidence']:.3f}  "
                f"len={det.get('streak_length_px', 0):.0f}px  "
                f"angle={obb.get('angle_deg', 0):.1f}°  "
                f"Tip1 RA/Dec=({det.get('ra_tip1_deg')}, {det.get('dec_tip1_deg')})  "
                f"Tip2 RA/Dec=({det.get('ra_tip2_deg')}, {det.get('dec_tip2_deg')})  "
                f"best_id={best_id}"
            )
        if not results:
            print("  (no detections above threshold)")
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except EnvironmentError as exc:
        print(f"ENVIRONMENT ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
