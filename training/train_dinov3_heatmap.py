"""Train the plain PyTorch DINOv3 streak heatmap spike.

This is intentionally independent of OpenMMLab. It is meant to test whether
ARGUS can replace the MMDetection/DETR training path with a smaller model that
is easier to run on native Windows and ordinary PyTorch environments.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from inference.device import get_device
from models.plain_dinov3.streak_heatmap import DINOv3StreakHeatmap, imagenet_normalize
from training.dinov3_heatmap_dataset import StreakHeatmapDataset, collate_heatmap_batch

logger = logging.getLogger(__name__)


def _dice_score(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
    return ((2 * intersection + 1e-6) / denom).mean()


def _run_epoch(
    model: DINOv3StreakHeatmap,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    pos_weight: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_dice = 0.0
    batches = 0
    pos_weight_tensor = torch.tensor([pos_weight], device=device)

    for batch in loader:
        image = imagenet_normalize(batch["image"].to(device))
        target = batch["heatmap"].to(device)
        output = model(image)
        logits = output
        if logits.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")

        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight_tensor)
        dice_loss = 1.0 - _dice_score(logits, target)
        loss = bce + dice_loss

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_dice += float((1.0 - dice_loss).detach().cpu())
        batches += 1

    denom = max(batches, 1)
    return {"loss": total_loss / denom, "dice": total_dice / denom}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-annotations", default="data/annotations/train.json")
    parser.add_argument("--val-annotations", default="data/annotations/val.json")
    parser.add_argument("--weights", default="weights/dinov3_vitb16_lvd1689m.pth")
    parser.add_argument("--model-size", choices=["base", "large"], default="base")
    parser.add_argument("--work-dir", default="weights/run_plain_dinov3_heatmap")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float, default=20.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--norm-mode", choices=["autostretch", "zscore", "zscale"],
                        default="autostretch",
                        help="Pixel normalisation applied to raw FITS/NPY tiles. "
                             "'autostretch' (default) removes sky background so streak "
                             "signal is consistent across tiles. 'zscore' clips at ±3σ. "
                             "'zscale' uses IRAF ZScale.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.smoke_test:
        args.epochs = 1
        args.max_samples = args.max_samples or 4
        args.batch_size = 1

    device = get_device()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    train_ds = StreakHeatmapDataset(args.train_annotations, args.image_size, max_samples=args.max_samples, norm_mode=args.norm_mode)
    val_ds = StreakHeatmapDataset(args.val_annotations, args.image_size, max_samples=args.max_samples, norm_mode=args.norm_mode)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_heatmap_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_heatmap_batch,
    )

    model = DINOv3StreakHeatmap(args.model_size, args.weights).to(device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=1e-4)

    best_dice = -1.0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, device, args.pos_weight)
        with torch.no_grad():
            val_metrics = _run_epoch(model, val_loader, None, device, args.pos_weight)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
        }
        history.append(row)
        logger.info("epoch=%d train_loss=%.4f train_dice=%.3f val_loss=%.4f val_dice=%.3f",
                    epoch, row["train_loss"], row["train_dice"], row["val_loss"], row["val_dice"])

        latest = {
            "model": model.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "val_dice": val_metrics["dice"],
        }
        torch.save(latest, work_dir / "latest.pt")
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(latest, work_dir / "best.pt")

    (work_dir / "history.json").write_text(json.dumps(history, indent=2))
    logger.info("best val dice %.3f saved to %s", best_dice, work_dir / "best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
