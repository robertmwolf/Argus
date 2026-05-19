"""Train the plain DINOv3 heatmap head from cached frozen features."""

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

from models.plain_dinov3.streak_heatmap import decode_geometry

logger = logging.getLogger(__name__)


class CachedHeatmapDataset(Dataset):
    """Dataset backed by feature `.pt` files from cache_dinov3_heatmap_features."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        metadata = json.loads((self.cache_dir / "manifest.json").read_text())
        self.items = metadata["manifest"]
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        sample = torch.load(self.cache_dir / item["path"], map_location="cpu", weights_only=False)
        return {
            "features": sample["features"].float(),
            "heatmap": sample["heatmap"].float(),
            "geometry": sample.get("geometry", torch.zeros((4, *sample["heatmap"].shape[-2:]))).float(),
            "image_id": torch.tensor(sample["image_id"], dtype=torch.int64),
        }


class HeatmapHead(nn.Module):
    """Small convolutional heatmap and geometry head for cached DINOv3 features."""

    def __init__(self, in_channels: int, hidden_channels: int = 256, out_channels: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels // 2, out_channels, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {
        "features": torch.stack([item["features"] for item in batch]),
        "heatmap": torch.stack([item["heatmap"] for item in batch]),
        "geometry": torch.stack([item["geometry"] for item in batch]),
        "image_id": torch.stack([item["image_id"] for item in batch]),
    }


def _dice_score(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
    return ((2 * intersection + 1e-6) / denom).mean()


def _run_epoch(
    head: HeatmapHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    pos_weight: float,
    geometry_weight: float,
) -> dict[str, float]:
    training = optimizer is not None
    head.train(training)
    total_loss = 0.0
    total_dice = 0.0
    batches = 0
    pos_weight_tensor = torch.tensor([pos_weight], device=device)

    for batch in loader:
        features = batch["features"].to(device)
        target = batch["heatmap"].to(device)
        geometry = batch["geometry"].to(device)
        output = head(features)
        logits = output[:, :1]
        geom_pred = decode_geometry(output[:, 1:5])
        if logits.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
            geometry = F.interpolate(geometry, size=logits.shape[-2:], mode="nearest")
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight_tensor)
        dice_loss = 1.0 - _dice_score(logits, target)
        mask = target.expand_as(geometry) > 0
        if bool(mask.any()):
            geom_loss = F.smooth_l1_loss(geom_pred[mask], geometry[mask])
        else:
            geom_loss = torch.zeros((), device=device)
        loss = bce + dice_loss + geometry_weight * geom_loss
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
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--work-dir", default="weights/run_plain_dinov3_heatmap_cached")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float, default=20.0)
    parser.add_argument("--geometry-weight", type=float, default=0.25)
    parser.add_argument("--hidden-channels", type=int, default=256)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CachedHeatmapDataset(args.train_cache)
    val_ds = CachedHeatmapDataset(args.val_cache)
    in_channels = int(torch.load(Path(args.train_cache) / train_ds.items[0]["path"], map_location="cpu", weights_only=False)["features"].shape[0])
    head = HeatmapHead(in_channels, args.hidden_channels).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate)

    best_dice = -1.0
    history: list[dict[str, float | int]] = []
    metadata = {
        "args": vars(args),
        "train_cache_metadata": train_ds.metadata,
        "val_cache_metadata": val_ds.metadata,
        "in_channels": in_channels,
    }
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(head, train_loader, optimizer, device, args.pos_weight, args.geometry_weight)
        with torch.no_grad():
            val_metrics = _run_epoch(head, val_loader, None, device, args.pos_weight, args.geometry_weight)
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
        payload = {"head": head.state_dict(), "epoch": epoch, "val_dice": val_metrics["dice"], **metadata}
        torch.save(payload, work_dir / "latest.pt")
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(payload, work_dir / "best.pt")

    (work_dir / "history.json").write_text(json.dumps(history, indent=2))
    logger.info("best val dice %.3f saved to %s", best_dice, work_dir / "best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
