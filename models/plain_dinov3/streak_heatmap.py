"""Plain PyTorch DINOv3 heatmap model for satellite streak detection.

This module deliberately avoids MMDetection, MMEngine, and mmcv. It treats
DINOv3 as a frozen feature encoder and trains a small convolutional head that
predicts a low-resolution streak probability heatmap.

# Source: DINOv3 (Meta AI, 2025) — ViT feature extraction pattern
# Ref: https://github.com/facebookresearch/dinov3
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_IMAGENET_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
_IMAGENET_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)

_MODEL_CONFIGS: dict[str, tuple[str, int, dict]] = {
    "base": (
        "vit_base",
        768,
        dict(patch_size=16, img_size=518, n_storage_tokens=4,
             layerscale_init=1e-4, mask_k_bias=True),
    ),
    "large": (
        "vit_large",
        1024,
        dict(patch_size=16, img_size=518, n_storage_tokens=4,
             layerscale_init=1e-4, mask_k_bias=True),
    ),
}


def load_dinov3_encoder(model_size: str, weights_path: Path) -> nn.Module:
    """Load a DINOv3 ViT encoder without OpenMMLab dependencies.

    Args:
        model_size: ``base`` for ViT-B/16 or ``large`` for ViT-L/16.
        weights_path: Local DINOv3 checkpoint path.

    Returns:
        Frozen DINOv3 ViT module.
    """
    if model_size not in _MODEL_CONFIGS:
        raise KeyError(f"model_size must be one of {sorted(_MODEL_CONFIGS)}, got {model_size!r}")
    if not weights_path.exists():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights_path}")

    try:
        import dinov3.models.vision_transformer as vits  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "dinov3 package not installed. Run: "
            "pip install git+https://github.com/facebookresearch/dinov3.git"
        ) from exc

    fn_name, _, kwargs = _MODEL_CONFIGS[model_size]
    model = getattr(vits, fn_name)(**kwargs)

    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("DINOv3 load: %d missing keys, first=%s", len(missing), missing[:3])
    if unexpected:
        logger.warning("DINOv3 load: %d unexpected keys, first=%s", len(unexpected), unexpected[:3])

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


class DINOv3StreakHeatmap(nn.Module):
    """Frozen DINOv3 encoder plus trainable low-resolution streak heatmap head."""

    def __init__(
        self,
        model_size: str = "base",
        weights: str | Path = "weights/dinov3_vitb16_lvd1689m.pth",
        hidden_channels: int = 256,
        out_channels: int = 5,
        freeze_backbone: bool = True,
    ) -> None:
        """Initialise the model.

        Args:
            model_size: ``base`` or ``large``.
            weights: Local DINOv3 checkpoint.
            hidden_channels: Width of the trainable heatmap head.
            out_channels: 1 heatmap channel plus 4 geometry channels by default.
            freeze_backbone: Keep True for the initial spike.
        """
        super().__init__()
        self.model_size = model_size
        self.freeze_backbone = freeze_backbone
        _, embed_dim, _ = _MODEL_CONFIGS[model_size]
        self.encoder = load_dinov3_encoder(model_size, Path(weights))
        if not freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad_(True)

        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels // 2, out_channels, kernel_size=1),
        )

    def train(self, mode: bool = True) -> "DINOv3StreakHeatmap":
        """Keep the encoder in eval mode when frozen."""
        super().train(mode)
        if self.freeze_backbone:
            self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict patch-grid heatmap logits.

        Args:
            x: Float tensor, shape ``(B, 3, H, W)``, already ImageNet-normalised.

        Returns:
            Tensor, shape ``(B, 5, H/16, W/16)`` by default. Channel 0 is the
            heatmap logit; channels 1-4 are geometry predictions.
        """
        feats = self.extract_features(x)
        return self.head(feats)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the final DINOv3 spatial feature map."""
        if self.freeze_backbone:
            with torch.no_grad():
                return self.encoder.get_intermediate_layers(x, n=1, reshape=True)[0]
        return self.encoder.get_intermediate_layers(x, n=1, reshape=True)[0]

    def predict_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """Predict heatmap logits from cached DINOv3 features."""
        return self.head(features)


def imagenet_normalize(batch: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalisation to a ``[0, 1]`` RGB batch."""
    mean = batch.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = batch.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    return (batch - mean) / std


def decode_geometry(raw_geometry: torch.Tensor) -> torch.Tensor:
    """Decode raw geometry channels to target scale.

    Channels are: cos(2θ), sin(2θ), length/image_size, width/image_size.
    """
    angle_vec = torch.tanh(raw_geometry[:, 0:2])
    length = 2.0 * torch.sigmoid(raw_geometry[:, 2:3])
    width = torch.sigmoid(raw_geometry[:, 3:4])
    return torch.cat([angle_vec, length, width], dim=1)


def decode_box(raw_box: torch.Tensor) -> torch.Tensor:
    """Decode raw direct-box channels to target scale.

    Channels are: dx, dy, cos(2θ), sin(2θ), length/image_size,
    width/image_size.
    """
    offset = 2.0 * torch.tanh(raw_box[:, 0:2])
    angle_vec = torch.tanh(raw_box[:, 2:4])
    length = 2.0 * torch.sigmoid(raw_box[:, 4:5])
    width = torch.sigmoid(raw_box[:, 5:6])
    return torch.cat([offset, angle_vec, length, width], dim=1)
