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

import logging
import math
import os
import time
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
# Constants / defaults
# ---------------------------------------------------------------------------

_DEFAULT_CONFIDENCE_THRESHOLD = 0.10  # raised to 0.30 once cloud-trained Swin-L is in use
_FAST_IMAGE_SIZE = 256


# ---------------------------------------------------------------------------
# Model config selection
# ---------------------------------------------------------------------------

def _select_config(model_size: str) -> Path:
    """Return the MMDetection config path for the given model size.

    Args:
        model_size: 'tiny' (Swin-T, Mac dev) or 'large' (Swin-L, A100 cloud).

    Returns:
        Absolute path to the MMDetection config file.

    Raises:
        EnvironmentError: If 'large' is requested on a non-CUDA device.
        ValueError: If an unknown model_size string is given.
    """
    root = Path(__file__).resolve().parent.parent
    configs = {
        "tiny":  root / "models" / "dino" / "streak_codino_swin_t.py",
        "large": root / "models" / "dino" / "streak_codino_swin_l.py",
    }
    if model_size not in configs:
        raise ValueError(
            f"Unknown MODEL_SIZE '{model_size}'. Choose 'tiny' or 'large'."
        )
    if model_size == "large":
        from inference.device import get_device
        device = get_device()
        if device.type != "cuda":
            raise EnvironmentError(
                "MODEL_SIZE=large requires a CUDA GPU. "
                f"Current device: {device.type}. Use MODEL_SIZE=tiny on Mac."
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
) -> list[dict]:
    """Run DINO inference on an image array, return raw bbox detections.

    Args:
        model: Loaded MMDetection model.
        array: uint8 (H, W, 3) image array.
        image_size: Longest edge to which the image is rescaled for inference.
        confidence_threshold: Minimum score to keep a detection.

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
        })

    logger.debug("DINO: %d raw detections above threshold %.2f",
                 len(detections), confidence_threshold)
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
            weights_path = root / "weights" / f"dino_{model_size}.pth"

    model = _load_model(config_path, Path(weights_path), inference_device)
    return model, inference_device


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run(
    fits_path: str | Path,
    fast: bool = False,
    model: Any | None = None,
    inference_device: Any | None = None,
) -> list[dict]:
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

    Returns:
        List of detection dicts.  Each dict has keys:
          confidence        — float, DINO score 0–1
          bbox              — [x1, y1, x2, y2] pixel coords
          obb               — {cx, cy, w, h, angle_deg}
          streak_length_px  — float, long axis of OBB
          ra_deg            — float or None, sky coord of streak midpoint
          dec_deg           — float or None
          identifications   — list of up to 3 {satellite_name, norad_id,
                               confidence, rank} dicts (empty in fast mode)

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

    # --- 2. Load model + run DINO inference ----------------------------------
    t1 = time.perf_counter()
    from inference.device import get_device, get_device_config
    device     = get_device()
    dev_config = get_device_config()
    image_size = _FAST_IMAGE_SIZE if fast else dev_config["image_size"]

    if model is None:
        config_path = _select_config(model_size)
        if weights_env:
            weights_path = Path(weights_env)
        else:
            root = Path(__file__).resolve().parent.parent
            weights_path = root / "weights" / f"dino_{model_size}.pth"

        # DINO multi-scale deformable attention exceeds MPS's 4 GB per-allocation
        # limit.  Force CPU on Mac until a memory-efficient MPS path is available.
        import torch as _torch
        inference_device = device
        if device.type == "mps" and not _torch.cuda.is_available():
            inference_device = _torch.device("cpu")
            logger.debug("MPS device detected — forcing DINO inference to CPU")

        model = _load_model(config_path, weights_path, inference_device)
    elif inference_device is None:
        inference_device = device

    raw_dets   = _run_inference(model, array, image_size, confidence_threshold)
    inference_ms = (time.perf_counter() - t1) * 1000
    logger.debug("inference_ms=%.1f  raw_dets=%d  threshold=%.2f",
                 inference_ms, len(raw_dets), confidence_threshold)

    if not raw_dets:
        logger.debug("No detections above threshold — returning empty list")
        return []

    # --- 3. Postprocess: angle refinement + OBB + NMS -----------------------
    t2 = time.perf_counter()
    from inference.postprocess import bbox_to_obb, refine_angle, nms_detections, extend_obb_to_streak_extent

    h_img = array.shape[0]
    w_img = array.shape[1]

    for det in raw_dets:
        x1, y1, x2, y2 = det["bbox"]

        # Always run Radon refinement — it operates on a small crop and is fast.
        # FAST_MODE only skips cross-ID, not angle refinement.
        px1 = max(0, int(math.floor(x1)))
        py1 = max(0, int(math.floor(y1)))
        px2 = min(w_img, int(math.ceil(x2)))
        py2 = min(h_img, int(math.ceil(y2)))
        crop = array[py1:py2, px1:px2]
        seed_angle = _angle_from_bbox(det["bbox"])
        initial_obb = bbox_to_obb(det["bbox"], seed_angle)
        # Use ±45° search so diagonal streaks aren't missed by a coarse seed
        angle = refine_angle(crop, initial_obb, angle_search_range=45.0)

        obb = bbox_to_obb(det["bbox"], angle)
        # Extend OBB endpoints along the streak axis to the true streak tips —
        # DINO bboxes often cover only a fraction of a long streak.
        obb = extend_obb_to_streak_extent(array, obb)
        det["obb"] = obb
        det["streak_length_px"] = float(obb["w"])

    detections = nms_detections(raw_dets, iou_threshold=0.5)
    postprocess_ms = (time.perf_counter() - t2) * 1000
    logger.debug("postprocess_ms=%.1f  after_nms=%d", postprocess_ms, len(detections))

    # --- 4. WCS: pixel → sky coordinates ------------------------------------
    for det in detections:
        cx = det["obb"]["cx"]
        cy = det["obb"]["cy"]
        ra, dec = _pixel_to_sky(cx, cy, wcs)
        det["ra_deg"]  = ra
        det["dec_deg"] = dec

    # --- 5. Cross-identification (skipped in fast mode) ----------------------
    t3 = time.perf_counter()
    if fast:
        for det in detections:
            det["identifications"] = []
        crossid_ms = 0.0
    else:
        from inference.crossid import cross_identify
        # Extract observer location from FITS header (may be None)
        observer_lat = fits_data.get("observer_lat") or 0.0
        observer_lon = fits_data.get("observer_lon") or 0.0
        observer_alt = fits_data.get("observer_alt_m") or 0.0
        if obs_time is None:
            obs_time = datetime.now(tz=timezone.utc)
        cross_identify(
            detections, obs_time,
            observer_lat, observer_lon, observer_alt,
        )
        crossid_ms = (time.perf_counter() - t3) * 1000

    logger.debug("crossid_ms=%.1f", crossid_ms)
    logger.debug("db_write_ms=0.0  (Phase 4 — not yet implemented)")

    logger.info(
        "Pipeline complete: %d detections  "
        "[load=%.0fms  infer=%.0fms  post=%.0fms  crossid=%.0fms]",
        len(detections), fits_load_ms, inference_ms, postprocess_ms, crossid_ms,
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
                f"RA={det.get('ra_deg')}  Dec={det.get('dec_deg')}  "
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
