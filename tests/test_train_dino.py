"""Tests for training/train_dino.py.

All tests are unit-level — no actual training is run, no GPU needed.
MMDet Runner is fully mocked.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from training.train_dino import (
    _select_config,
    _make_stage2_hook,
    _make_cost_hook,
    _CONFIG_MAP,
    _STAGE2_EPOCH,
)


# ---------------------------------------------------------------------------
# _select_config
# ---------------------------------------------------------------------------

class TestSelectConfig:
    def test_tiny_returns_swin_t_path(self):
        path = _select_config("tiny")
        assert "swin_t" in path

    def test_large_returns_swin_l_path(self):
        with patch("torch.cuda.is_available", return_value=True):
            path = _select_config("large")
        assert "swin_l" in path

    def test_unknown_model_size_raises(self):
        with pytest.raises(ValueError, match="MODEL_SIZE"):
            _select_config("medium")

    def test_large_without_cuda_raises(self):
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(EnvironmentError, match="CUDA"):
                _select_config("large")

    def test_tiny_never_requires_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            path = _select_config("tiny")   # must not raise
        assert Path(path).suffix == ".py"

    def test_config_files_exist_on_disk(self):
        for size, path in _CONFIG_MAP.items():
            assert Path(path).exists(), (
                f"Config for MODEL_SIZE={size} not found at {path}"
            )


# ---------------------------------------------------------------------------
# Stage2UnfreezeHook
# ---------------------------------------------------------------------------

class TestStage2Hook:
    def _make_runner(self, epoch: int):
        runner = MagicMock()
        runner.epoch = epoch - 1   # mmengine stores 0-indexed internally

        # Set up a mock param group with backbone flag
        backbone_group = {
            "lr": 0.0,
            "_is_backbone": True,
            "_base_lr": 1e-5,
        }
        non_backbone = {"lr": 1e-4, "_is_backbone": False}
        runner.optim_wrapper.optimizer.param_groups = [
            backbone_group, non_backbone
        ]
        return runner

    def test_hook_returns_not_none(self):
        hook = _make_stage2_hook()
        assert hook is not None

    def test_fires_at_stage2_epoch(self):
        hook = _make_stage2_hook(stage2_epoch=21)
        runner = self._make_runner(epoch=21)
        hook.before_train_epoch(runner)
        # Backbone LR should be updated to base_lr * 0.1
        backbone_group = runner.optim_wrapper.optimizer.param_groups[0]
        assert backbone_group["lr"] == pytest.approx(1e-5 * 0.1)

    def test_does_not_fire_at_other_epochs(self):
        hook = _make_stage2_hook(stage2_epoch=21)
        for epoch in (1, 10, 20, 22, 50):
            runner = self._make_runner(epoch=epoch)
            original_lr = runner.optim_wrapper.optimizer.param_groups[0]["lr"]
            hook.before_train_epoch(runner)
            assert runner.optim_wrapper.optimizer.param_groups[0]["lr"] == original_lr

    def test_non_backbone_group_untouched(self):
        hook = _make_stage2_hook(stage2_epoch=21)
        runner = self._make_runner(epoch=21)
        original_lr = runner.optim_wrapper.optimizer.param_groups[1]["lr"]
        hook.before_train_epoch(runner)
        assert runner.optim_wrapper.optimizer.param_groups[1]["lr"] == original_lr


# ---------------------------------------------------------------------------
# CostGuardrailHook
# ---------------------------------------------------------------------------

class TestCostHook:
    def test_hook_returns_not_none(self):
        hook = _make_cost_hook()
        assert hook is not None

    def test_records_start_time_on_epoch_0(self):
        hook = _make_cost_hook(max_epochs=50)
        runner = MagicMock()
        runner.epoch = 0

        before = time.time()
        hook.before_train_epoch(runner)
        after = time.time()

        assert before <= hook._epoch1_start <= after

    def test_prints_and_sleeps_after_epoch_1(self, capsys):
        hook = _make_cost_hook(max_epochs=2, cost_per_hour=1.29)
        hook._epoch1_start = time.time() - 10.0  # pretend epoch took 10s

        runner = MagicMock()
        runner.epoch = 0   # after_train_epoch fires when epoch==0

        with patch("time.sleep") as mock_sleep:
            hook.after_train_epoch(runner)

        captured = capsys.readouterr()
        assert "cost" in captured.out.lower() or "epoch" in captured.out.lower()
        mock_sleep.assert_called_once_with(30)

    def test_does_not_sleep_after_later_epochs(self):
        hook = _make_cost_hook(max_epochs=50)
        runner = MagicMock()

        with patch("time.sleep") as mock_sleep:
            for epoch in (1, 2, 10, 49):
                runner.epoch = epoch
                hook.after_train_epoch(runner)

        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# CLI argument parsing (import-level smoke test)
# ---------------------------------------------------------------------------

class TestTrainDinoImport:
    def test_module_imports_cleanly(self):
        """train_dino must import without side-effects."""
        import training.train_dino  # noqa: F401

    def test_config_map_has_both_sizes(self):
        assert "tiny" in _CONFIG_MAP
        assert "large" in _CONFIG_MAP

    def test_stage2_epoch_is_reasonable(self):
        assert 1 < _STAGE2_EPOCH < 50


# ---------------------------------------------------------------------------
# make_test_fits (smoke-test the test data generator)
# ---------------------------------------------------------------------------

class TestMakeTestFits:
    def test_generates_fits_file(self, tmp_path):
        from scripts.make_test_fits import make_test_fits
        from astropy.io import fits as astrofits

        out = tmp_path / "test.fits"
        meta = make_test_fits(out, with_streak=True, width=64, height=64, seed=0)

        assert out.exists()
        assert meta is not None
        assert "angle_deg" in meta
        assert "length_px" in meta

        hdul = astrofits.open(str(out))
        data = hdul[0].data
        assert data.shape == (64, 64)
        hdul.close()

    def test_blank_image_returns_none(self, tmp_path):
        from scripts.make_test_fits import make_test_fits

        out = tmp_path / "blank.fits"
        meta = make_test_fits(out, with_streak=False, width=64, height=64, seed=1)
        assert meta is None
        assert out.exists()

    def test_generate_test_set(self, tmp_path):
        from scripts.make_test_fits import generate_test_set

        results = generate_test_set(
            tmp_path, n_streak=2, n_blank=1, small=True, seed=7
        )
        assert len(results) == 3
        streak_count = sum(1 for r in results if r["has_streak"])
        blank_count  = sum(1 for r in results if not r["has_streak"])
        assert streak_count == 2
        assert blank_count  == 1
        for r in results:
            assert Path(r["path"]).exists()

    def test_wcs_headers_present(self, tmp_path):
        from scripts.make_test_fits import make_test_fits
        from astropy.io import fits as astrofits

        out = tmp_path / "wcs.fits"
        make_test_fits(out, width=64, height=64, seed=2)
        hdul = astrofits.open(str(out))
        h = hdul[0].header
        for key in ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CTYPE1"):
            assert key in h, f"Missing WCS header key: {key}"
        hdul.close()

    def test_reproducible_with_same_seed(self, tmp_path):
        from scripts.make_test_fits import make_test_fits
        from astropy.io import fits as astrofits
        import numpy as np

        p1 = tmp_path / "a.fits"
        p2 = tmp_path / "b.fits"
        make_test_fits(p1, width=64, height=64, seed=99)
        make_test_fits(p2, width=64, height=64, seed=99)

        d1 = astrofits.open(str(p1))[0].data
        d2 = astrofits.open(str(p2))[0].data
        np.testing.assert_array_equal(d1, d2)
