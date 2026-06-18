"""Tests for the endpoint-native inference pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np


def _detection() -> dict:
    return {
        "x1": 30.0,
        "y1": 50.0,
        "x2": 180.0,
        "y2": 50.0,
        "confidence": 0.9,
        "method": "vits_heatmap",
    }


def _fits_payload() -> dict:
    array = np.zeros((256, 256, 3), dtype=np.uint8)
    return {
        "array": array,
        "raw_float32": np.zeros((256, 256), dtype=np.float32),
        "norm_mode": "zscore",
        "wcs": None,
        "obs_time": None,
    }


def test_model_registry_is_retired() -> None:
    from inference.pipeline import load_models, resolve_model_specs
    assert resolve_model_specs() == []
    assert load_models() == []


def test_fast_pipeline_returns_endpoint_only_detection(tmp_path: Path) -> None:
    from inference.pipeline import run_with_array

    image_path = tmp_path / "image.fits"
    image_path.touch()
    with patch("inference.fits_loader.FITSLoader.load", return_value=_fits_payload()), \
         patch("inference.pipeline._run_all_detectors", return_value=(
             {"vits_heatmap": [_detection()]}, {}, []
         )):
        detections, array, heatmaps = run_with_array(image_path, fast=True)

    assert array.shape == (256, 256, 3)
    assert heatmaps == {}
    assert len(detections) == 1
    detection = detections[0]
    assert detection["x1"] == 30.0
    assert detection["x2"] == 180.0
    assert detection["streak_length_px"] == 150.0
    assert detection["angle_deg"] == 0.0
    assert set(detection) >= {"x1", "y1", "x2", "y2"}
    assert "bbox" not in detection


def test_raw_mode_assigns_independent_streak_ids(tmp_path: Path) -> None:
    from inference.pipeline import run_with_array

    image_path = tmp_path / "image.fits"
    image_path.touch()
    second = {**_detection(), "x1": 40.0, "x2": 190.0, "confidence": 0.8}
    with patch("inference.fits_loader.FITSLoader.load", return_value=_fits_payload()), \
         patch("inference.pipeline._run_all_detectors", return_value=(
             {"vits_heatmap": [_detection(), second]}, {}, []
         )):
        detections, _, _ = run_with_array(image_path, fast=True, raw_mode=True)
    assert [item["streak_id"] for item in detections] == [1, 2]
