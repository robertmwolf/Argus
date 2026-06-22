"""Endpoint-native ARGUS inference pipeline."""

from __future__ import annotations

import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_RADON_ANGLE_SEARCH_RANGE = 60.0
_DEFAULT_CROSSID_MAX_DETECTIONS = 3


def resolve_model_specs() -> list[dict]:
    """Return no legacy detector specifications."""
    return []


def load_models() -> list[tuple[Any, Any, dict]]:
    """Return no legacy detector instances."""
    return []


def load_model(*_args: Any, **_kwargs: Any) -> tuple[None, None]:
    """Return an empty compatibility tuple for retired detector callers."""
    logger.warning("load_model() is retired; endpoint heatmap models load on demand")
    return None, None


def _pixel_to_sky(x: float, y: float, wcs: Any) -> tuple[float | None, float | None]:
    """Convert one pixel coordinate to sky coordinates when WCS is available."""
    if wcs is None:
        return None, None
    try:
        sky = wcs.pixel_to_world(x, y)
        return float(sky.ra.deg), float(sky.dec.deg)
    except Exception as exc:
        logger.debug("Pixel-to-sky conversion failed: %s", exc)
        return None, None


def get_detector_statuses() -> list[dict]:
    """Return availability metadata for endpoint heatmap detectors."""
    import importlib
    try:
        getter = getattr(
            importlib.import_module("inference.vits_window_v11_detector"),
            "get_vits_v11_heatmap_status",
        )
        return [getter()]
    except (ImportError, AttributeError):
        return [{
            "id": "vits_heatmap_v11",
            "name": "DINOv3 ViT-S HeatMap v11 (asl_cldice, coord-validated)",
            "type": "ml",
            "dataset": "Atwood window_v11",
            "status": "unavailable",
        }]


def _detector_runners() -> dict[str, tuple[Callable[[np.ndarray], tuple[list[dict], np.ndarray | None]], str, str]]:
    """Return detector callables with their normalization configuration."""
    from inference.vits_window_v11_detector import run_vits_v11_heatmap_detector_and_heatmap

    return {
        "vits_heatmap_v11": (run_vits_v11_heatmap_detector_and_heatmap, "VITS_V11_HEATMAP_NORM", "zscore"),
    }


def _run_all_detectors(
    models_with_specs: list[tuple[Any, Any, dict]],
    array: np.ndarray,
    fits_path: Path,
    image_size: int,
    confidence_threshold: float,
    tta_enabled: bool,
    enabled_detectors: set[str] | None,
    raw_array_f32: np.ndarray | None = None,
    default_norm_mode: str = "",
) -> tuple[dict[str, list[dict]], dict[str, np.ndarray], list[str]]:
    """Run enabled endpoint heatmap detectors concurrently."""
    del models_with_specs, fits_path, image_size, confidence_threshold, tta_enabled
    from inference.fits_loader import apply_norm
    from inference.postprocess import stitch_collinear_segments

    runners = _detector_runners()
    selected = {
        key: value for key, value in runners.items()
        if enabled_detectors is None or key in enabled_detectors
    }
    results: dict[str, list[dict]] = {key: [] for key in runners if key not in selected}
    heatmaps: dict[str, np.ndarray] = {}

    def execute(
        detector_id: str,
        runner: Callable[[np.ndarray], tuple[list[dict], np.ndarray | None]],
        norm_env: str,
        norm_default: str,
    ) -> tuple[str, list[dict], np.ndarray | None, float]:
        norm = os.environ.get(norm_env, norm_default).lower()
        model_array = (
            apply_norm(raw_array_f32, norm)
            if raw_array_f32 is not None and norm != default_norm_mode
            else array
        )
        started = time.perf_counter()
        detections, heatmap = runner(model_array)
        detections = stitch_collinear_segments(detections) if len(detections) > 1 else detections
        for detection in detections:
            detection["method"] = detector_id
            detection["norm_mode"] = norm
        return detector_id, detections, heatmap, (time.perf_counter() - started) * 1000.0

    with ThreadPoolExecutor(max_workers=max(1, len(selected))) as pool:
        futures = {
            pool.submit(execute, detector_id, *spec): detector_id
            for detector_id, spec in selected.items()
        }
        for future in as_completed(futures):
            detector_id = futures[future]
            try:
                key, detections, heatmap, elapsed_ms = future.result()
                results[key] = detections
                if heatmap is not None:
                    heatmaps[key] = heatmap
                logger.info(
                    "detector_timing detector=%s elapsed_ms=%.1f detections=%d",
                    key, elapsed_ms, len(detections),
                )
            except Exception:
                logger.exception("Detector %s failed", detector_id)
                results[detector_id] = []
    return results, heatmaps, []


def run_with_array(
    fits_path: str | Path,
    fast: bool = False,
    model: Any | None = None,
    inference_device: Any | None = None,
    models: list[tuple[Any, Any, dict]] | None = None,
    enabled_detectors: set[str] | None = None,
    raw_mode: bool = False,
) -> tuple[list[dict], np.ndarray, dict[str, np.ndarray]]:
    """Run endpoint detection, refinement, WCS conversion, and cross-ID."""
    del model, inference_device
    from inference.fits_loader import FITSLoader
    from inference.postprocess import (
        classify_detection_quality,
        extend_segment_to_streak_extent,
        fuse_group_geometries,
        group_detections,
        nms_detections,
        refine_segment_angle,
    )
    from inference.streak_segment import apply_segment_geometry

    path = Path(fits_path)
    load_started = time.perf_counter()
    fits_data = FITSLoader().load(path, skip_plate_solve=fast)
    array = fits_data["array"]
    raw_array = fits_data.get("raw_float32")
    wcs = fits_data.get("wcs")
    load_ms = (time.perf_counter() - load_started) * 1000.0

    inference_started = time.perf_counter()
    detector_results, heatmaps, _ = _run_all_detectors(
        models or [], array, path, 0, 0.0, False, enabled_detectors,
        raw_array_f32=raw_array,
        default_norm_mode=fits_data.get("norm_mode", "zscore"),
    )
    detections = [item for group in detector_results.values() for item in group]
    inference_ms = (time.perf_counter() - inference_started) * 1000.0

    post_started = time.perf_counter()
    gray = np.asarray(array, dtype=np.float32)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    threshold = float(np.median(gray)) + 3.0 * float(gray.std())
    angle_range = float(os.environ.get("RADON_ANGLE_SEARCH_RANGE", _DEFAULT_RADON_ANGLE_SEARCH_RANGE))
    height, width = gray.shape

    for detection in detections:
        apply_segment_geometry(detection)
        if not fast and not raw_mode:
            padding = 20
            left = max(0, math.floor(min(detection["x1"], detection["x2"])) - padding)
            top = max(0, math.floor(min(detection["y1"], detection["y2"])) - padding)
            right = min(width, math.ceil(max(detection["x1"], detection["x2"])) + padding)
            bottom = min(height, math.ceil(max(detection["y1"], detection["y2"])) + padding)
            angle = refine_segment_angle(
                array[top:bottom, left:right], detection["angle_deg"], angle_range
            )
            half = detection["streak_length_px"] / 2.0
            radians = math.radians(angle)
            detection["x1"] = detection["cx"] - half * math.cos(radians)
            detection["y1"] = detection["cy"] - half * math.sin(radians)
            detection["x2"] = detection["cx"] + half * math.cos(radians)
            detection["y2"] = detection["cy"] + half * math.sin(radians)
            detection.update(extend_segment_to_streak_extent(
                array, detection, sample_halfwidth=15, _gray=gray, _threshold=threshold
            ))

    if raw_mode:
        for index, detection in enumerate(detections, 1):
            detection["streak_id"] = index
    else:
        by_method: dict[str, list[dict]] = {}
        for detection in detections:
            by_method.setdefault(detection.get("method", "unknown"), []).append(detection)
        detections = [item for group in by_method.values() for item in nms_detections(group)]
        detections = fuse_group_geometries(group_detections(detections))
    postprocess_ms = (time.perf_counter() - post_started) * 1000.0

    for detection in detections:
        apply_segment_geometry(detection)
        detection["ra_tip1_deg"], detection["dec_tip1_deg"] = _pixel_to_sky(
            detection["x1"], detection["y1"], wcs
        )
        detection["ra_tip2_deg"], detection["dec_tip2_deg"] = _pixel_to_sky(
            detection["x2"], detection["y2"], wcs
        )
        detection["quality_flag"] = classify_detection_quality(detection, (height, width))

    crossid_started = time.perf_counter()
    if fast or wcs is None:
        for detection in detections:
            detection["identifications"] = []
    else:
        max_detections = int(os.environ.get("CROSSID_MAX_DETECTIONS", _DEFAULT_CROSSID_MAX_DETECTIONS))
        candidates = sorted(detections, key=lambda item: item.get("confidence", 0.0), reverse=True)
        candidates = candidates[:max(0, max_detections)]
        selected_ids = {id(item) for item in candidates}
        for detection in detections:
            if id(detection) not in selected_ids:
                detection["identifications"] = []
        if candidates:
            from inference.crossid import cross_identify
            cross_identify(
                candidates,
                fits_data.get("obs_time") or datetime.now(tz=timezone.utc),
                fits_data.get("observer_lat") or 0.0,
                fits_data.get("observer_lon") or 0.0,
                fits_data.get("observer_alt_m") or 0.0,
                exposure_time=fits_data.get("exposure_time"),
            )
    crossid_ms = (time.perf_counter() - crossid_started) * 1000.0
    logger.info(
        "Pipeline complete: %d detections [load=%.0fms infer=%.0fms post=%.0fms crossid=%.0fms]",
        len(detections), load_ms, inference_ms, postprocess_ms, crossid_ms,
    )
    return detections, array, heatmaps


def run(
    fits_path: str | Path,
    fast: bool = False,
    model: Any | None = None,
    inference_device: Any | None = None,
    raw_mode: bool = False,
) -> list[dict]:
    """Run the endpoint pipeline and return detections only."""
    detections, _, _ = run_with_array(
        fits_path,
        fast=fast,
        model=model,
        inference_device=inference_device,
        raw_mode=raw_mode,
    )
    return detections


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.image, fast=args.fast), indent=2))
