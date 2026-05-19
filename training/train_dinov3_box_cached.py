"""Train a plain DINOv3 center/box head from cached frozen features."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from inference.device import get_device
from models.plain_dinov3.streak_heatmap import decode_box

logger = logging.getLogger(__name__)


class CachedBoxDataset(Dataset):
    """Dataset backed by feature `.pt` files with center/box targets."""

    def __init__(self, cache_dir: str | Path) -> None:
        """Initialise the cached dataset.

        Args:
            cache_dir: Directory written by ``cache_dinov3_heatmap_features.py``.
        """
        self.cache_dir = Path(cache_dir)
        metadata = json.loads((self.cache_dir / "manifest.json").read_text())
        self.items = metadata["manifest"]
        self.metadata = metadata

    def __len__(self) -> int:
        """Return number of cached samples."""
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load one cached feature sample."""
        item = self.items[idx]
        sample = torch.load(self.cache_dir / item["path"], map_location="cpu", weights_only=False)
        if "center_heatmap" not in sample or "box_target" not in sample or "box_mask" not in sample:
            raise KeyError(
                "Cache is missing center/box targets. Rebuild it with the updated "
                "scripts/cache_dinov3_heatmap_features.py."
            )
        return {
            "features": sample["features"].float(),
            "center_heatmap": sample["center_heatmap"].float(),
            "box_target": sample["box_target"].float(),
            "box_mask": sample["box_mask"].float(),
            "image_id": torch.tensor(sample["image_id"], dtype=torch.int64),
        }


class CenterBoxHead(nn.Module):
    """Small convolutional center heatmap plus direct box-regression head."""

    def __init__(self, in_channels: int, hidden_channels: int = 256) -> None:
        """Initialise the head.

        Args:
            in_channels: DINOv3 feature channel count.
            hidden_channels: Width of the trainable head.
        """
        super().__init__()
        mid_channels = hidden_channels // 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
        )
        self.center_head = nn.Sequential(
            nn.Conv2d(hidden_channels, mid_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
        )
        self.box_head = nn.Sequential(
            nn.Conv2d(hidden_channels, mid_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_channels, 6, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict center logits and raw box channels."""
        stem = self.stem(features)
        return torch.cat([self.center_head(stem), self.box_head(stem)], dim=1)


def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Collate cached box samples."""
    return {
        "features": torch.stack([item["features"] for item in batch]),
        "center_heatmap": torch.stack([item["center_heatmap"] for item in batch]),
        "box_target": torch.stack([item["box_target"] for item in batch]),
        "box_mask": torch.stack([item["box_mask"] for item in batch]),
        "image_id": torch.stack([item["image_id"] for item in batch]),
    }


def _dice_score(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute soft Dice score for the center heatmap."""
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
    return ((2 * intersection + 1e-6) / denom).mean()


def _run_epoch(
    head: CenterBoxHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    pos_weight: float,
    box_weight: float,
) -> dict[str, float]:
    """Run one train or validation epoch."""
    training = optimizer is not None
    head.train(training)
    total_loss = 0.0
    total_center_dice = 0.0
    total_box_loss = 0.0
    batches = 0
    pos_weight_tensor = torch.tensor([pos_weight], device=device)

    for batch in loader:
        features = batch["features"].to(device)
        center_target = batch["center_heatmap"].to(device)
        box_target = batch["box_target"].to(device)
        box_mask = batch["box_mask"].to(device)

        output = head(features)
        center_logits = output[:, :1]
        box_pred = decode_box(output[:, 1:7])
        if center_logits.shape[-2:] != center_target.shape[-2:]:
            center_target = F.interpolate(center_target, size=center_logits.shape[-2:], mode="nearest")
            box_target = F.interpolate(box_target, size=center_logits.shape[-2:], mode="nearest")
            box_mask = F.interpolate(box_mask, size=center_logits.shape[-2:], mode="nearest")

        bce = F.binary_cross_entropy_with_logits(center_logits, center_target, pos_weight=pos_weight_tensor)
        dice_loss = 1.0 - _dice_score(center_logits, center_target)
        reg_mask = box_mask.expand_as(box_target) > 0
        if bool(reg_mask.any()):
            raw_loss = F.smooth_l1_loss(box_pred, box_target, beta=0.02, reduction="none")
            channel_weights = box_target.new_tensor([2.0, 2.0, 1.0, 1.0, 4.0, 20.0]).view(1, 6, 1, 1)
            box_loss = (raw_loss * channel_weights * box_mask).sum() / ((box_mask.sum() * channel_weights.sum()) + 1e-6)
        else:
            box_loss = torch.zeros((), device=device)
        loss = bce + dice_loss + box_weight * box_loss

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_center_dice += float((1.0 - dice_loss).detach().cpu())
        total_box_loss += float(box_loss.detach().cpu())
        batches += 1

    denom = max(batches, 1)
    return {
        "loss": total_loss / denom,
        "center_dice": total_center_dice / denom,
        "box_loss": total_box_loss / denom,
    }


def main() -> int:
    """Run cached direct center/box training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--work-dir", default="weights/run_plain_dinov3_box_cached")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float, default=20.0)
    parser.add_argument("--box-weight", type=float, default=1.0)
    parser.add_argument("--hidden-channels", type=int, default=256)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = get_device()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CachedBoxDataset(args.train_cache)
    val_ds = CachedBoxDataset(args.val_cache)
    first = torch.load(Path(args.train_cache) / train_ds.items[0]["path"], map_location="cpu", weights_only=False)
    in_channels = int(first["features"].shape[0])
    head = CenterBoxHead(in_channels, args.hidden_channels).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate)

    best_score = -1.0
    history: list[dict[str, float | int]] = []
    metadata = {
        "args": vars(args),
        "train_cache_metadata": train_ds.metadata,
        "val_cache_metadata": val_ds.metadata,
        "in_channels": in_channels,
        "head_type": "center_box",
    }
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(head, train_loader, optimizer, device, args.pos_weight, args.box_weight)
        with torch.no_grad():
            val_metrics = _run_epoch(head, val_loader, None, device, args.pos_weight, args.box_weight)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_center_dice": train_metrics["center_dice"],
            "train_box_loss": train_metrics["box_loss"],
            "val_loss": val_metrics["loss"],
            "val_center_dice": val_metrics["center_dice"],
            "val_box_loss": val_metrics["box_loss"],
        }
        history.append(row)
        logger.info(
            "epoch=%d train_loss=%.4f train_center_dice=%.3f train_box=%.4f "
            "val_loss=%.4f val_center_dice=%.3f val_box=%.4f",
            epoch,
            row["train_loss"],
            row["train_center_dice"],
            row["train_box_loss"],
            row["val_loss"],
            row["val_center_dice"],
            row["val_box_loss"],
        )
        score = float(val_metrics["center_dice"] - 0.05 * val_metrics["box_loss"])
        payload = {"head": head.state_dict(), "epoch": epoch, "score": score, **metadata}
        torch.save(payload, work_dir / "latest.pt")
        if score > best_score:
            best_score = score
            torch.save(payload, work_dir / "best.pt")

    (work_dir / "history.json").write_text(json.dumps(history, indent=2))
    logger.info("best score %.3f saved to %s", best_score, work_dir / "best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
