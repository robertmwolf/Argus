"""Tests for inference/device.py."""

import os
from contextlib import contextmanager
from unittest.mock import patch, MagicMock

import pytest
import torch

from inference.device import get_device, get_device_config, safe_autocast, _mps_available


# ---------------------------------------------------------------------------
# Helpers: patch torch backend availability
# ---------------------------------------------------------------------------

@contextmanager
def _mock_backends(cuda: bool = False, mps: bool = False):
    """Patch torch availability flags for the duration of the block."""
    with patch("torch.cuda.is_available", return_value=cuda), \
         patch("torch.backends.mps.is_available", return_value=mps):
        yield


# ---------------------------------------------------------------------------
# get_device
# ---------------------------------------------------------------------------

class TestGetDevice:
    def test_returns_torch_device(self):
        device = get_device()
        assert isinstance(device, torch.device)

    def test_cuda_preferred_over_mps(self):
        with _mock_backends(cuda=True, mps=True):
            device = get_device()
        assert device.type == "cuda"

    def test_mps_preferred_over_cpu(self):
        with _mock_backends(cuda=False, mps=True):
            device = get_device()
        assert device.type == "mps"

    def test_cpu_fallback_when_no_gpu(self):
        with _mock_backends(cuda=False, mps=False):
            device = get_device()
        assert device.type == "cpu"

    def test_mps_disabled_by_env_var(self):
        with _mock_backends(cuda=False, mps=True), \
             patch.dict(os.environ, {"DISABLE_MPS": "1"}):
            device = get_device()
        assert device.type == "cpu"

    def test_device_type_is_string(self):
        device = get_device()
        assert device.type in ("cuda", "mps", "cpu")


# ---------------------------------------------------------------------------
# get_device_config
# ---------------------------------------------------------------------------

class TestGetDeviceConfig:
    def _config_for(self, device_type: str) -> dict:
        with _mock_backends(cuda=(device_type == "cuda"),
                            mps=(device_type == "mps")):
            return get_device_config()

    def test_returns_dict(self):
        cfg = get_device_config()
        assert isinstance(cfg, dict)

    def test_required_keys_present(self):
        required = {
            "batch_size", "num_workers", "pin_memory",
            "image_size", "mixed_precision", "gradient_checkpointing",
        }
        cfg = get_device_config()
        assert required.issubset(cfg.keys())

    # CUDA config
    def test_cuda_batch_size(self):
        assert self._config_for("cuda")["batch_size"] == 2

    def test_cuda_num_workers(self):
        assert self._config_for("cuda")["num_workers"] == 4

    def test_cuda_pin_memory(self):
        assert self._config_for("cuda")["pin_memory"] is True

    def test_cuda_image_size(self):
        assert self._config_for("cuda")["image_size"] == 800

    def test_cuda_mixed_precision(self):
        assert self._config_for("cuda")["mixed_precision"] is True

    # MPS config
    def test_mps_num_workers_is_zero(self):
        """MPS crashes with >0 DataLoader workers."""
        assert self._config_for("mps")["num_workers"] == 0

    def test_mps_pin_memory_false(self):
        """pin_memory is not supported on MPS."""
        assert self._config_for("mps")["pin_memory"] is False

    def test_mps_mixed_precision_false(self):
        """MPS autocast support is incomplete."""
        assert self._config_for("mps")["mixed_precision"] is False

    def test_mps_image_size_reduced(self):
        """Image size is halved on Mac to fit 16 GB unified memory."""
        assert self._config_for("mps")["image_size"] == 400

    # CPU config
    def test_cpu_pin_memory_false(self):
        assert self._config_for("cpu")["pin_memory"] is False

    def test_cpu_mixed_precision_false(self):
        assert self._config_for("cpu")["mixed_precision"] is False

    def test_cpu_image_size_reduced(self):
        assert self._config_for("cpu")["image_size"] == 400

    # General
    def test_gradient_checkpointing_always_true(self):
        for device_type in ("cuda", "mps", "cpu"):
            cfg = self._config_for(device_type)
            assert cfg["gradient_checkpointing"] is True, device_type

    def test_batch_size_is_positive_int(self):
        cfg = get_device_config()
        assert isinstance(cfg["batch_size"], int)
        assert cfg["batch_size"] >= 1


# ---------------------------------------------------------------------------
# safe_autocast
# ---------------------------------------------------------------------------

class TestSafeAutocast:
    def test_does_not_raise_on_cpu(self):
        device = torch.device("cpu")
        with safe_autocast(device):
            t = torch.tensor([1.0, 2.0]) + 1.0
        # [1+1, 2+1] = [2, 3], sum = 5
        assert t.sum().item() == pytest.approx(5.0)

    def test_returns_context_manager(self):
        device = torch.device("cpu")
        ctx = safe_autocast(device)
        # Must support __enter__ / __exit__
        assert hasattr(ctx, "__enter__")
        assert hasattr(ctx, "__exit__")

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_does_not_raise_on_cuda(self):
        device = torch.device("cuda")
        with safe_autocast(device):
            t = torch.tensor([1.0], device=device) + 1.0
        assert t.item() == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# _mps_available (internal helper)
# ---------------------------------------------------------------------------

class TestMpsAvailable:
    def test_false_when_mps_not_available(self):
        with patch("torch.backends.mps.is_available", return_value=False):
            assert _mps_available() is False

    def test_false_when_disable_mps_env_set(self):
        with patch("torch.backends.mps.is_available", return_value=True), \
             patch.dict(os.environ, {"DISABLE_MPS": "true"}):
            assert _mps_available() is False

    def test_true_when_available_and_not_disabled(self):
        with patch("torch.backends.mps.is_available", return_value=True), \
             patch.dict(os.environ, {}, clear=False):
            # Remove DISABLE_MPS if set in environment
            env = {k: v for k, v in os.environ.items() if k != "DISABLE_MPS"}
            with patch.dict(os.environ, env, clear=True):
                assert _mps_available() is True
