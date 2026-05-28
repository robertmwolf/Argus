"""DINOv3 ViT backbone adapter for MMDetection.

DINOv3 is an isotropic ViT — every transformer block operates at the same
spatial resolution (H/16 × W/16 for patch_size=16).  MMDetection's neck
(ChannelMapper) expects a feature pyramid at strides [8, 16, 32, 64].

This module bridges the gap:
  1. Loads DINOv3 ViT from a local .pth checkpoint with correct constructor args.
  2. Extracts the last intermediate layer as spatial feature maps (B, C, H/16, W/16).
  3. Builds a 4-level pseudo-pyramid by bilinear upsampling and average pooling.
  4. Registers as 'DINOv3Backbone' in MMDetection's MODELS registry so it can be
     referenced by name in MMDet config dicts.

Usage in an MMDet config:
    backbone=dict(
        type='DINOv3Backbone',
        model_size='base',           # 'small' (ViT-S), 'base' (ViT-B), or 'large' (ViT-L)
        weights='weights/dinov3_vitb16_lvd1689m.pth',
        frozen=True,                 # keep False only for optional Stage F unfreeze
        out_channels=768,            # 768 for ViT-B, 1024 for ViT-L
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[768, 768, 768, 768],   # 4 pyramid levels, all same width
        ...
    ),

# Source: DINOv3 (Meta AI, 2025) — ViT feature extraction pattern
# Ref: https://github.com/facebookresearch/dinov3
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Map model_size → (constructor fn name, embed_dim, constructor kwargs)
_MODEL_CONFIGS: dict[str, tuple[str, int, dict]] = {
    "small": (
        "vit_small",
        384,
        dict(patch_size=16, img_size=518, n_storage_tokens=4,
             layerscale_init=1e-4, mask_k_bias=True),
    ),
    "base": (
        "vit_base",
        768,
        dict(patch_size=16, img_size=518, n_storage_tokens=4,
             layerscale_init=1e-4, mask_k_bias=True),
    ),
    "large": (
        "vit_large",
        1024,
        # ViT-L checkpoint config — verify with:
        #   torch.load(path)['storage_tokens'].shape  → (1, N, 1024)
        #   'blocks.0.ls1.gamma' present → layerscale_init needed
        dict(patch_size=16, img_size=518, n_storage_tokens=4,
             layerscale_init=1e-4, mask_k_bias=True),
    ),
}


def _load_dinov3(model_size: str, weights_path: Path) -> nn.Module:
    """Instantiate and load a DINOv3 ViT model from a local .pth file.

    Args:
        model_size: 'base' or 'large'.
        weights_path: Path to the pretrain .pth checkpoint.

    Returns:
        DINOv3 ViT in eval mode with frozen parameters.

    Raises:
        ImportError: dinov3 package not installed.
        FileNotFoundError: weights_path does not exist.
        KeyError: unrecognised model_size.
    """
    if model_size not in _MODEL_CONFIGS:
        raise KeyError(f"model_size must be 'base' or 'large', got '{model_size}'")
    if not Path(weights_path).exists():
        raise FileNotFoundError(
            f"DINOv3 weights not found: {weights_path}\n"
            "Download from the Meta portal and place in weights/."
        )
    try:
        import dinov3.models.vision_transformer as vits  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "dinov3 package not installed.\n"
            "Run: pip install git+https://github.com/facebookresearch/dinov3.git"
        ) from exc

    fn_name, _, kwargs = _MODEL_CONFIGS[model_size]
    constructor = getattr(vits, fn_name)
    model = constructor(**kwargs)

    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("DINOv3 load: %d missing keys — %s ...", len(missing), missing[:3])
    if unexpected:
        logger.warning("DINOv3 load: %d unexpected keys — %s ...", len(unexpected), unexpected[:3])

    model.eval()
    return model


class DINOv3Backbone(nn.Module):
    """MMDetection-compatible backbone wrapping a frozen DINOv3 ViT.

    Produces a 4-level feature pyramid from a single-scale isotropic ViT by
    spatially upsampling and downsampling the last transformer layer's patch
    features:

        Level 0  stride  8  → upsample ×2   (highest resolution)
        Level 1  stride 16  → identity       (native ViT output)
        Level 2  stride 32  → avg-pool ×2
        Level 3  stride 64  → avg-pool ×4

    All levels share the same channel width (768 for ViT-B, 1024 for ViT-L),
    which is then projected to 256 by the downstream ChannelMapper neck.

    Args:
        model_size: 'base' (ViT-B/16, 768-dim) or 'large' (ViT-L/16, 1024-dim).
        weights: Path to the DINOv3 pretrain .pth file.
        frozen: If True (default), backbone parameters are frozen and the module
            runs in eval mode even during model.train().  Set False only for the
            optional Stage F partial-unfreeze experiment.
        out_channels: Feature channel width exposed to the neck.  Must match
            the ViT embed_dim for the chosen model_size (768 / 1024).
    """

    def __init__(
        self,
        model_size: str = "base",
        weights: str = "weights/dinov3_vitb16_lvd1689m.pth",
        frozen: bool = True,
        out_channels: int = 768,
    ) -> None:
        super().__init__()
        self.model_size = model_size
        self.frozen = frozen
        self.out_channels = out_channels
        _, embed_dim, _ = _MODEL_CONFIGS[model_size]
        if out_channels != embed_dim:
            raise ValueError(
                f"out_channels={out_channels} does not match embed_dim={embed_dim} "
                f"for model_size='{model_size}'. They must be equal."
            )

        self.vit = _load_dinov3(model_size, Path(weights))
        if frozen:
            for p in self.vit.parameters():
                p.requires_grad_(False)
            logger.info("DINOv3Backbone (%s): backbone frozen", model_size)
        else:
            logger.info("DINOv3Backbone (%s): backbone trainable", model_size)

    def init_weights(self) -> None:
        """No-op: weights are already loaded from the pretrain checkpoint."""

    def train(self, mode: bool = True) -> "DINOv3Backbone":
        """Keep the ViT in eval mode when frozen to disable dropout/BN updates."""
        super().train(mode)
        if self.frozen:
            self.vit.eval()
        return self

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Extract a 4-level feature pyramid from a batch of images.

        Args:
            x: Float tensor (B, 3, H, W), ImageNet-normalised.
               H and W must be divisible by 16.

        Returns:
            Tuple of 4 tensors at strides (8, 16, 32, 64):
                (B, C, H/8,  W/8)
                (B, C, H/16, W/16)
                (B, C, H/32, W/32)
                (B, C, H/64, W/64)
            where C = out_channels (768 or 1024).
        """
        # get_intermediate_layers with reshape=True returns spatial tensors
        # (B, embed_dim, H/patch, W/patch) — one per requested layer.
        # We use n=1 (last layer only) since all isotropic layers share the
        # same spatial resolution; depth variation adds little pyramid benefit.
        # Source: DINOv3 — get_intermediate_layers API
        # Ref: https://github.com/facebookresearch/dinov3
        if self.frozen:
            with torch.no_grad():
                feats = self.vit.get_intermediate_layers(x, n=1, reshape=True)
        else:
            feats = self.vit.get_intermediate_layers(x, n=1, reshape=True)

        base: torch.Tensor = feats[0]  # (B, C, H/16, W/16)

        p0 = F.interpolate(base, scale_factor=2.0, mode="bilinear", align_corners=False)
        p1 = base
        p2 = F.avg_pool2d(base, kernel_size=2, stride=2)
        p3 = F.avg_pool2d(base, kernel_size=4, stride=4)

        return (p0, p1, p2, p3)


# ---------------------------------------------------------------------------
# MMDetection registry
# ---------------------------------------------------------------------------

def register_dinov3_backbone() -> None:
    """Register DINOv3Backbone with MMDetection's MODELS registry.

    Called once at import time.  Safe to call multiple times (re-registration
    is a no-op if the key already exists).
    """
    try:
        from mmdet.registry import MODELS  # type: ignore[import]
        if "DINOv3Backbone" not in MODELS._module_dict:
            MODELS.register_module(module=DINOv3Backbone)
            logger.debug("DINOv3Backbone registered with MMDetection MODELS registry")
    except ImportError:
        logger.debug("mmdet not available — DINOv3Backbone not registered (standalone mode ok)")


register_dinov3_backbone()


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    weights = sys.argv[1] if len(sys.argv) > 1 else "weights/dinov3_vitb16_lvd1689m.pth"
    size    = sys.argv[2] if len(sys.argv) > 2 else "base"
    out_ch  = 768 if size == "base" else 1024

    print(f"\nTesting DINOv3Backbone (model_size='{size}', weights='{weights}')")
    backbone = DINOv3Backbone(model_size=size, weights=weights, frozen=True, out_channels=out_ch)
    backbone.eval()

    h, w = 416, 416
    x = torch.zeros(1, 3, h, w)
    with torch.no_grad():
        pyramid = backbone(x)

    print(f"\nInput:  {tuple(x.shape)}")
    strides = [8, 16, 32, 64]
    for i, (feat, stride) in enumerate(zip(pyramid, strides)):
        expected_h = h // stride
        expected_w = w // stride
        status = "✓" if feat.shape == (1, out_ch, expected_h, expected_w) else "✗ MISMATCH"
        print(f"  Level {i} (stride {stride:2d}): {tuple(feat.shape)}  {status}")

    print(f"\nAll grads zero (frozen check): {all(p.grad is None for p in backbone.vit.parameters())}")
    print("DINOv3Backbone standalone test complete.")
