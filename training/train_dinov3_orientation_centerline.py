"""Train a DINOv3 orientation-binned centerline heatmap model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inference.device import get_device
from models.plain_dinov3.streak_heatmap import (
    DINOv3OrientationCenterline,
    imagenet_normalize,
)
from training.dinov3_orientation_centerline_dataset import (
    DINOv3OrientationCenterlineDataset,
    collate_centerline_batch,
)

logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable spike runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dice_score_from_heat(logit_heat: torch.Tensor, target_heat: torch.Tensor) -> torch.Tensor:
    """Soft Dice on bin-collapsed centerline heatmaps."""
    probs = torch.sigmoid(logit_heat)
    intersection = (probs * target_heat).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target_heat.sum(dim=(1, 2, 3)) + 1e-6
    return ((2.0 * intersection + 1e-6) / denom).mean()


def _centerline_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float,
    dice_weight: float,
    bce_weight: float,
    orientation_ce_weight: float,
    manual_positive_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the colleague-style centerline loss mix."""
    target_heat = target.amax(dim=1, keepdim=True)
    logit_heat = logits.amax(dim=1, keepdim=True)
    pos_weight_tensor = torch.tensor([pos_weight], device=logits.device)
    pixel_weight = torch.ones_like(target)
    pixel_weight = pixel_weight + (manual_positive_weight - 1.0) * (target > 0).float()
    bce_raw = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight_tensor,
        reduction="none",
    )
    bce = (bce_raw * pixel_weight).mean()
    dice = _dice_score_from_heat(logit_heat, target_heat)
    dice_loss = 1.0 - dice

    positive_mask = (target_heat.squeeze(1) > 0.05).float()
    labels = target.argmax(dim=1)
    ce_raw = F.cross_entropy(logits, labels, reduction="none")
    orientation_ce = (ce_raw * positive_mask).sum() / positive_mask.sum().clamp_min(1.0)

    loss = dice_weight * dice_loss + bce_weight * bce + orientation_ce_weight * orientation_ce
    metrics = {
        "loss": float(loss.detach().cpu()),
        "dice": float(dice.detach().cpu()),
        "bce": float(bce.detach().cpu()),
        "orientation_ce": float(orientation_ce.detach().cpu()),
    }
    return loss, metrics


def _catchment_loss(
    logits: torch.Tensor,
    catchment_target: torch.Tensor,
    pos_weight: float,
    dice_weight: float,
    bce_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute a wider seed-zone loss on bin-collapsed heatmaps."""
    catchment_heat = catchment_target.amax(dim=1, keepdim=True)
    logit_heat = logits.amax(dim=1, keepdim=True)
    pos_weight_tensor = torch.tensor([pos_weight], device=logits.device)
    bce = F.binary_cross_entropy_with_logits(
        logit_heat,
        catchment_heat,
        pos_weight=pos_weight_tensor,
    )
    dice = _dice_score_from_heat(logit_heat, catchment_heat)
    loss = dice_weight * (1.0 - dice) + bce_weight * bce
    metrics = {
        "catchment_dice": float(dice.detach().cpu()),
        "catchment_bce": float(bce.detach().cpu()),
        "catchment_loss": float(loss.detach().cpu()),
    }
    return loss, metrics


def _run_epoch(
    model: DINOv3OrientationCenterline,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    args: argparse.Namespace,
    scaler: torch.cuda.amp.GradScaler | None,
) -> dict[str, float]:
    """Run one train or validation epoch."""
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "dice": 0.0,
        "bce": 0.0,
        "orientation_ce": 0.0,
        "image_bce": 0.0,
        "catchment_dice": 0.0,
        "catchment_bce": 0.0,
        "catchment_loss": 0.0,
    }
    batches = 0

    for batch_idx, batch in enumerate(loader, start=1):
        images = imagenet_normalize(batch["image"].to(device))
        target = batch["target"].to(device)
        catchment_target = batch["catchment_target"].to(device)
        image_target = batch["positive"].to(device).view(-1, 1)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda" and args.amp):
            if args.image_loss_weight > 0.0:
                logits, image_logits = model.forward_with_image_logit(images)
            else:
                logits = model(images)
                image_logits = None
            if logits.shape[-2:] != target.shape[-2:]:
                logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
            loss, metrics = _centerline_loss(
                logits=logits,
                target=target,
                pos_weight=args.pos_weight,
                dice_weight=args.dice_weight,
                bce_weight=args.bce_weight,
                orientation_ce_weight=args.orientation_ce_weight,
                manual_positive_weight=args.manual_positive_weight,
            )
            if args.catchment_loss_weight > 0.0:
                catch_loss, catch_metrics = _catchment_loss(
                    logits=logits,
                    catchment_target=catchment_target,
                    pos_weight=args.catchment_pos_weight,
                    dice_weight=args.catchment_dice_weight,
                    bce_weight=args.catchment_bce_weight,
                )
                loss = loss + args.catchment_loss_weight * catch_loss
                metrics.update(catch_metrics)
                metrics["loss"] = float(loss.detach().cpu())
            if image_logits is not None:
                image_loss = F.binary_cross_entropy_with_logits(image_logits, image_target)
                loss = loss + args.image_loss_weight * image_loss
                metrics["loss"] = float(loss.detach().cpu())
                metrics["image_bce"] = float(image_loss.detach().cpu())
        if training and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
        for key, value in metrics.items():
            if key not in totals:
                totals[key] = 0.0
            totals[key] += value
        batches += 1
        if training and args.log_interval > 0 and batch_idx % args.log_interval == 0:
            running = {key: value / max(batches, 1) for key, value in totals.items()}
            logger.info(
                "batch=%d/%d train_loss=%.4f train_dice=%.4f",
                batch_idx,
                len(loader),
                running["loss"],
                running["dice"],
            )

    denom = max(batches, 1)
    return {key: value / denom for key, value in totals.items()}


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    args: argparse.Namespace,
) -> torch.optim.lr_scheduler.LRScheduler:
    total_steps = max(len(train_loader) * args.epochs, 1)
    min_factor = args.min_lr / args.lr
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.min_lr,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-annotations", default="data/annotations/train.json")
    parser.add_argument("--val-annotations", default="data/annotations/val.json")
    parser.add_argument("--holdout-annotations", default=None)
    parser.add_argument("--work-dir", default="weights/run_dinov3_orientation_centerline")
    parser.add_argument("--weights", default="weights/dinov3_vitb16_lvd1689m.pth")
    parser.add_argument("--model-size", choices=["small", "base", "large"], default="base")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--tile-size", type=int, default=2560)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--positive-train-tiles", type=int, default=1236)
    parser.add_argument("--negative-train-tiles", type=int, default=1400)
    parser.add_argument("--orientation-bins", type=int, default=18)
    parser.add_argument("--decoder-channels", type=int, default=192)
    parser.add_argument("--last-layers", type=int, default=4)
    parser.add_argument("--centerline-width", type=float, default=2.0)
    parser.add_argument("--centerline-sigma", type=float, default=1.4)
    parser.add_argument("--catchment-width", type=float, default=0.0)
    parser.add_argument("--catchment-sigma", type=float, default=6.0)
    parser.add_argument("--neighbor-bin-weight", type=float, default=0.35)
    parser.add_argument("--second-neighbor-weight", type=float, default=0.0)
    parser.add_argument("--pos-weight", type=float, default=120.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=0.35)
    parser.add_argument("--orientation-ce-weight", type=float, default=0.20)
    parser.add_argument("--manual-positive-weight", type=float, default=8.0)
    parser.add_argument("--catchment-loss-weight", type=float, default=0.0)
    parser.add_argument("--catchment-pos-weight", type=float, default=20.0)
    parser.add_argument("--catchment-dice-weight", type=float, default=1.0)
    parser.add_argument("--catchment-bce-weight", type=float, default=0.20)
    parser.add_argument("--image-loss-weight", type=float, default=0.0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--preserve-image-bit-depth", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _set_seed(args.seed)
    device = get_device()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    train_ds = DINOv3OrientationCenterlineDataset(
        annotation_file=args.train_annotations,
        split="train",
        tile_size=args.tile_size,
        image_size=args.image_size,
        orientation_bins=args.orientation_bins,
        centerline_width=args.centerline_width,
        centerline_sigma=args.centerline_sigma,
        catchment_width=args.catchment_width,
        catchment_sigma=args.catchment_sigma,
        neighbor_bin_weight=args.neighbor_bin_weight,
        second_neighbor_weight=args.second_neighbor_weight,
        positive_tiles=args.positive_train_tiles,
        negative_tiles=args.negative_train_tiles,
        preserve_image_bit_depth=args.preserve_image_bit_depth,
        seed=args.seed,
        max_samples=args.max_train_samples,
    )
    val_ds = DINOv3OrientationCenterlineDataset(
        annotation_file=args.val_annotations,
        split="val",
        tile_size=args.tile_size,
        image_size=args.image_size,
        orientation_bins=args.orientation_bins,
        centerline_width=args.centerline_width,
        centerline_sigma=args.centerline_sigma,
        catchment_width=args.catchment_width,
        catchment_sigma=args.catchment_sigma,
        neighbor_bin_weight=args.neighbor_bin_weight,
        second_neighbor_weight=args.second_neighbor_weight,
        positive_tiles=None,
        negative_tiles=None,
        preserve_image_bit_depth=args.preserve_image_bit_depth,
        seed=args.seed,
        max_samples=args.max_val_samples,
    )
    pin_memory = device.type == "cuda"
    workers = args.workers if device.type != "mps" else 0
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        collate_fn=collate_centerline_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        collate_fn=collate_centerline_batch,
    )

    model = DINOv3OrientationCenterline(
        model_size=args.model_size,
        weights=args.weights,
        decoder_channels=args.decoder_channels,
        orientation_bins=args.orientation_bins,
        last_layers=args.last_layers,
        freeze_backbone=True,
    ).to(device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        logger.info("Resumed model from %s", args.resume)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = _build_scheduler(optimizer, train_loader, args)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and args.amp)
    if not scaler.is_enabled():
        scaler = None

    best_dice = -1.0
    history: list[dict[str, Any]] = []
    metadata = {
        "args": vars(args),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "model_name": "DINOv3OrientationCenterline",
    }
    (work_dir / "config.json").write_text(json.dumps(metadata, indent=2))
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, scheduler, device, args, scaler)
        with torch.no_grad():
            val_metrics = _run_epoch(model, val_loader, None, None, device, args, None)
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        logger.info(
            "epoch=%d train_loss=%.4f train_dice=%.4f val_loss=%.4f val_dice=%.4f lr=%.2e",
            epoch,
            row["train_loss"],
            row["train_dice"],
            row["val_loss"],
            row["val_dice"],
            row["lr"],
        )
        payload = {
            **metadata,
            "model": model.state_dict(),
            "epoch": epoch,
            "val_dice": val_metrics["dice"],
        }
        torch.save(payload, work_dir / "latest.pt")
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(payload, work_dir / "best.pt")
    (work_dir / "history.json").write_text(json.dumps(history, indent=2))
    logger.info("best val dice %.4f saved to %s", best_dice, work_dir / "best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
