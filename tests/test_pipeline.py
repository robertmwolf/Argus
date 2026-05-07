"""Tests for inference/pipeline.py — end-to-end inference orchestrator.

All mmdet model loading and inference are mocked so no GPU or weights file
is required to run these tests.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

_SYNTH_FITS = Path("data/sample/synth_streak_000.fits")

# Required keys in every detection dict returned by run()
_REQUIRED_KEYS = {
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
        extra_env  = {"MODEL_SIZE": "tiny", "MODEL_WEIGHTS": str(_SYNTH_FITS)}
        if env:
            extra_env.update(env)

        with patch.object(pl, "_load_model", return_value=mock_model), \
             patch.object(pl, "_run_inference", return_value=list(fake_dets)), \
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
             patch.dict(os.environ,
                        {"MODEL_SIZE": "tiny", "MODEL_WEIGHTS": str(_SYNTH_FITS),
                         "FAST_MODE": "true"},
                        clear=False):
            result = pl.run(_SYNTH_FITS, fast=False)  # fast=False but env says true
        # In fast mode identifications must be empty
        assert all(det["identifications"] == [] for det in result)


# ---------------------------------------------------------------------------
# refine_angle NOT called in fast mode
# ---------------------------------------------------------------------------

class TestAngleRefinement:
    """Radon refinement now always runs; FAST_MODE only skips cross-ID."""

    @pytest.fixture(autouse=True)
    def ensure_synth_fits(self):
        _ensure_synth_fits()

    def test_refine_angle_always_called(self):
        """refine_angle must be invoked for every detection regardless of fast mode."""
        import inference.pipeline as pl
        import inference.postprocess as pp

        with patch.object(pl, "_load_model", return_value=MagicMock()), \
             patch.object(pl, "_run_inference", return_value=[
                 {"bbox": [50.0, 60.0, 200.0, 80.0], "confidence": 0.92},
             ]), \
             patch.object(pp, "refine_angle", return_value=45.0) as mock_refine, \
             patch.dict(os.environ,
                        {"MODEL_SIZE": "tiny", "MODEL_WEIGHTS": str(_SYNTH_FITS)},
                        clear=False):
            pl.run(_SYNTH_FITS, fast=True)
        assert mock_refine.call_count >= 1

    def test_refine_angle_called_in_non_fast_mode(self):
        """refine_angle must be invoked for each detection when fast=False."""
        import inference.crossid as crossid
        import inference.pipeline as pl
        import inference.postprocess as pp

        with patch.object(pl, "_load_model", return_value=MagicMock()), \
             patch.object(pl, "_run_inference", return_value=[
                 {"bbox": [50.0, 60.0, 200.0, 80.0], "confidence": 0.92},
             ]), \
             patch.object(pp, "refine_angle", return_value=45.0) as mock_refine, \
             patch.object(crossid, "cross_identify", return_value=None), \
             patch.dict(os.environ,
                        {"MODEL_SIZE": "tiny", "MODEL_WEIGHTS": str(_SYNTH_FITS),
                         "FAST_MODE": "false"},
                        clear=False):
            pl.run(_SYNTH_FITS, fast=False)
        assert mock_refine.call_count >= 1
