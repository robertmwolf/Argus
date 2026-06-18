"""Tests for inference/pipeline.py — end-to-end inference orchestrator.

All mmdet model loading and inference are mocked so no GPU or weights file
is required to run these tests.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

_SYNTH_FITS = Path("data/sample/synth_streak_000.fits")

# Required keys in every detection dict returned by run()
_REQUIRED_KEYS = {
    "method",
    "confidence",
    "bbox",
    "obb",
    "streak_length_px",
    "ra_tip1_deg",
    "dec_tip1_deg",
    "ra_tip2_deg",
    "dec_tip2_deg",
    "identifications",
}
_OBB_KEYS = {"cx", "cy", "w", "h", "angle_deg"}

# Override heatmap checkpoint paths so detectors gracefully skip when weights
# aren't needed for a given test.
_NO_CKPT = "/nonexistent/checkpoint.pt"
_DISABLE_HEATMAPS = {
    "VITS_HEATMAP_CHECKPOINT":    _NO_CKPT,
    "VITB_HEATMAP_CHECKPOINT":    _NO_CKPT,
    "VITS_V4_HEATMAP_CHECKPOINT": _NO_CKPT,
    "VITB_V4_HEATMAP_CHECKPOINT": _NO_CKPT,
    "VITS_V9_HEATMAP_CHECKPOINT": _NO_CKPT,
}

# Two fake raw detections returned by the mocked _run_inference
_FAKE_DETS = [
    {"bbox": [50.0, 60.0, 300.0, 80.0], "confidence": 0.92},
    {"bbox": [10.0, 200.0, 20.0, 450.0], "confidence": 0.77},
]


def _ensure_synth_fits():
    if not _SYNTH_FITS.exists():
        from scripts.make_test_fits import generate_test_set
        generate_test_set(Path("data/sample"), n_streak=1, n_blank=0, small=True, seed=42)


# ---------------------------------------------------------------------------
# _select_config
# ---------------------------------------------------------------------------

class TestSelectConfig:
    def test_tiny_returns_swin_t_path(self):
        from inference.pipeline import _select_config
        p = _select_config("tiny")
        assert "swin_t" in p.name

    def test_large_raises_on_non_cuda(self):
        from inference.pipeline import _select_config
        # get_device is imported inside the function from inference.device
        with patch("inference.device.get_device") as mock_dev:
            mock_dev.return_value = MagicMock(type="mps")
            with pytest.raises(EnvironmentError, match="CUDA"):
                _select_config("large")

    def test_unknown_size_raises_value_error(self):
        from inference.pipeline import _select_config
        with pytest.raises(ValueError, match="Unknown MODEL_SIZE"):
            _select_config("xlarge")


# ---------------------------------------------------------------------------
# _load_model
# ---------------------------------------------------------------------------

class TestLoadModel:
    def test_missing_weights_raises_file_not_found(self, tmp_path):
        from inference.pipeline import _load_model
        import torch
        with pytest.raises(FileNotFoundError, match="weights not found"):
            _load_model(
                tmp_path / "config.py",
                tmp_path / "nonexistent.pth",
                torch.device("cpu"),
            )


# ---------------------------------------------------------------------------
# run() — fast mode with fully mocked internals
# ---------------------------------------------------------------------------

class TestRunFastMode:

    @pytest.fixture(autouse=True)
    def ensure_synth_fits(self):
        _ensure_synth_fits()

    def _run_patched(self, fast=True, n_dets=2, env=None):
        """Run pipeline.run() with _load_model and _run_inference mocked."""
        import inference.pipeline as pl

        mock_model = MagicMock()
        fake_dets  = _FAKE_DETS[:n_dets]
        extra_env  = {"MODEL_SIZE": "tiny", "MODEL_WEIGHTS": str(_SYNTH_FITS), **_DISABLE_HEATMAPS}
        if env:
            extra_env.update(env)

        with patch.object(pl, "_load_model", return_value=mock_model), \
             patch.object(pl, "_run_inference", return_value=list(fake_dets)), \
             patch.object(pl, "_run_classical_detector", return_value=[]), \
             patch.dict(os.environ, extra_env, clear=False):
            return pl.run(_SYNTH_FITS, fast=fast)

    def test_returns_list(self):
        assert isinstance(self._run_patched(), list)

    def test_each_detection_has_required_keys(self):
        for det in self._run_patched():
            missing = _REQUIRED_KEYS - set(det.keys())
            assert not missing, f"Detection missing keys: {missing}"

    def test_obb_has_required_keys(self):
        for det in self._run_patched():
            missing = _OBB_KEYS - set(det.get("obb", {}).keys())
            assert not missing, f"OBB missing keys: {missing}"

    def test_identifications_empty_in_fast_mode(self):
        for det in self._run_patched(fast=True):
            assert det["identifications"] == []

    def test_streak_length_px_is_positive(self):
        for det in self._run_patched():
            assert det["streak_length_px"] > 0.0

    def test_confidence_between_0_and_1(self):
        for det in self._run_patched():
            assert 0.0 <= det["confidence"] <= 1.0

    def test_ml_detections_have_method(self):
        for det in self._run_patched():
            assert det["method"] == "tiny"

    def test_empty_detections_returns_empty_list(self):
        result = self._run_patched(n_dets=0)
        assert result == []

    def test_missing_weights_raises_file_not_found(self, tmp_path):
        from inference.pipeline import _load_model
        import torch
        with pytest.raises(FileNotFoundError):
            _load_model(tmp_path / "cfg.py", tmp_path / "missing.pth", torch.device("cpu"))

    def test_fast_mode_env_var_respected(self):
        """FAST_MODE=true should activate fast mode even when fast=False is passed."""
        import inference.pipeline as pl

        with patch.object(pl, "_load_model", return_value=MagicMock()), \
             patch.object(pl, "_run_inference", return_value=list(_FAKE_DETS[:1])), \
             patch.object(pl, "_run_classical_detector", return_value=[]), \
             patch.dict(os.environ,
                        {"MODEL_SIZE": "tiny", "MODEL_WEIGHTS": str(_SYNTH_FITS),
                         "FAST_MODE": "true", **_DISABLE_HEATMAPS},
                        clear=False):
            result = pl.run(_SYNTH_FITS, fast=False)  # fast=False but env says true
        # In fast mode identifications must be empty
        assert all(det["identifications"] == [] for det in result)


# ---------------------------------------------------------------------------
# Cross-ID candidate cap
# ---------------------------------------------------------------------------

class TestCrossIdCap:
    def test_crossid_candidate_cap_limits_identification_work(self):
        import inference.crossid as crossid
        import inference.pipeline as pl
        import inference.postprocess as pp

        endpoint_dets = [
            {
                "x1": float(10 + i * 40), "y1": 20.0,
                "x2": float(35 + i * 40), "y2": 28.0,
                "confidence": 1.0 - i / 20.0,
            }
            for i in range(8)
        ]

        def _mark_identified(detections, *args, **kwargs):
            for det in detections:
                det["identifications"] = [{"rank": 1}]
            return detections

        with patch.object(pl, "_run_all_detectors", return_value=(
                 {"vits_heatmap": endpoint_dets}, {}, [])), \
             patch.object(pl, "_pixel_to_sky", return_value=(10.0, 20.0)), \
             patch.object(pp, "refine_segment_angle", return_value=0.0), \
             patch.object(crossid, "cross_identify", side_effect=_mark_identified) as mock_crossid, \
             patch.dict(os.environ,
                        {"MODEL_SIZE": "tiny", "MODEL_WEIGHTS": str(_SYNTH_FITS),
                         "FAST_MODE": "false", "CROSSID_MAX_DETECTIONS": "3",
                         "DINO_MAX_POSTPROCESS_DETECTIONS": "0"},
                        clear=False):
            result = pl.run(_SYNTH_FITS, fast=False)

        assert len(mock_crossid.call_args.args[0]) == 3
        assert sum(bool(det.get("identifications")) for det in result) == 3
