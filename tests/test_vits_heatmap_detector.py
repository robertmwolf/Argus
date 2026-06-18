"""Checkpoint compatibility tests for the ViT-S heatmap detector."""

import torch

from inference.vits_heatmap_detector import _centerline_head_state, _head_out_channels


def test_head_out_channels_reads_wrapped_head_state() -> None:
    state = {"net.4.weight": torch.zeros((5, 128, 1, 1))}
    assert _head_out_channels(state) == 5


def test_head_out_channels_reads_sequential_head_state() -> None:
    state = {"4.weight": torch.zeros((1, 128, 1, 1))}
    assert _head_out_channels(state) == 1


def test_centerline_head_state_normalizes_keys_and_drops_extra_channels() -> None:
    state = {
        "4.weight": torch.arange(10, dtype=torch.float32).reshape(5, 2, 1, 1),
        "4.bias": torch.arange(5, dtype=torch.float32),
    }
    runtime = _centerline_head_state(state)
    assert set(runtime) == {"net.4.weight", "net.4.bias"}
    assert runtime["net.4.weight"].shape == (1, 2, 1, 1)
    assert runtime["net.4.bias"].tolist() == [0.0]
