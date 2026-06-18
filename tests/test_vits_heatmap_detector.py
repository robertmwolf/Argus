"""Checkpoint compatibility tests for the ViT-S heatmap detector."""

import torch

from inference.heatmap_detector_base import _centerline_head_state


def test_centerline_head_state_normalizes_keys_and_drops_extra_channels() -> None:
    state = {
        "4.weight": torch.arange(10, dtype=torch.float32).reshape(5, 2, 1, 1),
        "4.bias": torch.arange(5, dtype=torch.float32),
    }
    runtime = _centerline_head_state(state)
    assert set(runtime) == {"net.4.weight", "net.4.bias"}
    assert runtime["net.4.weight"].shape == (1, 2, 1, 1)
    assert runtime["net.4.bias"].tolist() == [0.0]
