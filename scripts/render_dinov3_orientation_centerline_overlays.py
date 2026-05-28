"""Render visual QA panels for DINOv3 orientation-centerline checkpoints."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inference.device import get_device
from models.plain_dinov3.streak_heatmap import imagenet_normalize
from scripts.evaluate_dinov3_orientation_centerline import _load_checkpoint_model
from training.dinov3_orientation_centerline_dataset import (
    DINOv3OrientationCenterlineDataset,
    collate_centerline_batch,
)

logger = logging.getLogger(__name__)


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a CHW or HWC float image in [0, 1] to uint8 RGB."""
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def _heat_to_rgb(heat: np.ndarray) -> np.ndarray:
    """Colorize one heatmap as RGB."""
    heat_u8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(heat_u8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Render a binary mask as RGB."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask] = np.array([255, 255, 255], dtype=np.uint8)
    return out


def _orientation_to_rgb(heat: np.ndarray, bins: np.ndarray, n_bins: int) -> np.ndarray:
    """Render orientation bins as hue and confidence as value."""
    hue = (bins.astype(np.float32) / max(n_bins, 1) * 179.0).astype(np.uint8)
    saturation = np.full_like(hue, 220, dtype=np.uint8)
    value = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    hsv = np.stack([hue, saturation, value], axis=-1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _blend(base: np.ndarray, overlay: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Alpha blend two RGB images."""
    return np.clip(base.astype(np.float32) * (1.0 - alpha) + overlay.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def _add_label(panel: np.ndarray, label: str) -> Image.Image:
    """Add a compact label above a panel."""
    image = Image.fromarray(panel)
    canvas = Image.new("RGB", (image.width, image.height + 28), (18, 18, 18))
    canvas.paste(image, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), label, fill=(245, 245, 245))
    return canvas


def _save_panel(
    output_path: Path,
    source: np.ndarray,
    gt_heat: np.ndarray,
    pred_heat: np.ndarray,
    pred_mask: np.ndarray,
    pred_bins: np.ndarray,
    n_bins: int,
) -> None:
    """Save one multi-panel QA image."""
    source_rgb = _to_uint8_rgb(source)
    gt_rgb = _heat_to_rgb(gt_heat)
    pred_rgb = _heat_to_rgb(pred_heat)
    mask_rgb = _mask_to_rgb(pred_mask)
    orientation_rgb = _orientation_to_rgb(pred_heat, pred_bins, n_bins)
    panels = [
        _add_label(source_rgb, "source"),
        _add_label(_blend(source_rgb, gt_rgb), "gt centerline"),
        _add_label(_blend(source_rgb, pred_rgb), "pred heat"),
        _add_label(_blend(source_rgb, mask_rgb), "thresholded pred"),
        _add_label(_blend(source_rgb, orientation_rgb), "orientation hue"),
    ]
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--output-dir", default="results/dinov3_orientation_centerline_overlays")
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--target-threshold", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--preserve-image-bit-depth", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = get_device()
    model, train_args = _load_checkpoint_model(Path(args.checkpoint), args.weights, device)
    annotations = args.annotations or train_args.get("val_annotations", "data/annotations/val.json")
    preserve_bit_depth = bool(args.preserve_image_bit_depth or train_args.get("preserve_image_bit_depth", False))
    orientation_bins = int(train_args.get("orientation_bins", 18))
    ds = DINOv3OrientationCenterlineDataset(
        annotation_file=annotations,
        split="val",
        tile_size=int(train_args.get("tile_size", 2560)),
        image_size=int(train_args.get("image_size", 1024)),
        orientation_bins=orientation_bins,
        centerline_width=float(train_args.get("centerline_width", 2.0)),
        centerline_sigma=float(train_args.get("centerline_sigma", 1.4)),
        neighbor_bin_weight=float(train_args.get("neighbor_bin_weight", 0.35)),
        second_neighbor_weight=float(train_args.get("second_neighbor_weight", 0.0)),
        positive_tiles=None,
        negative_tiles=None,
        preserve_image_bit_depth=preserve_bit_depth,
        seed=int(train_args.get("seed", 20260524)),
        max_samples=args.max_samples,
    )
    workers = args.workers if device.type != "mps" else 0
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_centerline_batch,
    )

    output_dir = Path(args.output_dir)
    saved = 0
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            target = batch["target"].to(device)
            logits = model(imagenet_normalize(image))
            if logits.shape[-2:] != target.shape[-2:]:
                logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
            target_np = target[0].cpu().numpy()
            pred_heat = probs.max(axis=0)
            gt_heat = target_np.max(axis=0)
            pred_bins = probs.argmax(axis=0)
            pred_mask = pred_heat >= args.threshold
            image_id = int(batch["image_id"][0].item())
            positive = bool((gt_heat > args.target_threshold).any())
            output_path = output_dir / f"{saved:04d}_image{image_id}_pos{int(positive)}.png"
            _save_panel(
                output_path=output_path,
                source=image[0].cpu().numpy(),
                gt_heat=gt_heat,
                pred_heat=pred_heat,
                pred_mask=pred_mask,
                pred_bins=pred_bins,
                n_bins=orientation_bins,
            )
            saved += 1
    logger.info("saved %d overlays to %s", saved, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
