"""Train the plain DINOv3 heatmap head from cached frozen features."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inference.device import get_device
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


def _focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    """Sigmoid focal loss (Lin et al. 2017) for binary heatmap pixels.

    Replaces BCE+pos_weight when --focal-gamma > 0.  The (1-p_t)^gamma term
    down-weights easy examples so the head is penalised more for confident
    wrong predictions (i.e. bright background activations).
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = p * target + (1.0 - p) * (1.0 - target)
    focal_weight = (1.0 - p_t) ** gamma
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (alpha_t * focal_weight * ce).mean()


def _run_epoch(
    head: HeatmapHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    pos_weight: float,
    geometry_weight: float,
    focal_gamma: float = 0.0,
    focal_alpha: float = 0.85,
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
        if focal_gamma > 0.0:
            heatmap_loss = _focal_loss(logits, target, focal_alpha, focal_gamma)
        else:
            heatmap_loss = F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=pos_weight_tensor
            )
        dice_loss = 1.0 - _dice_score(logits, target)
        mask = target.expand_as(geometry) > 0
        if bool(mask.any()):
            geom_loss = F.smooth_l1_loss(geom_pred[mask], geometry[mask])
        else:
            geom_loss = torch.zeros((), device=device)
        loss = heatmap_loss + dice_loss + geometry_weight * geom_loss
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
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers.  0=synchronous (safe for MPS). "
                             "Set >0 only on CUDA where fork/spawn is stable.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest.pt in --work-dir. Restores head weights, "
                             "optimizer state, best_dice, and history, then continues "
                             "training for the remaining epochs (total --epochs minus "
                             "already-completed epochs).")
    parser.add_argument("--lr-scheduler", choices=["none", "cosine"], default="none",
                        help="LR scheduler. 'cosine' uses CosineAnnealingLR(T_max=epochs) "
                             "for faster convergence (~30%% fewer epochs). Default: none "
                             "(constant LR, Run 8 behaviour).")
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="Linear LR warmup epochs before cosine decay (only with "
                             "--lr-scheduler cosine). Lets a hotter peak LR run without "
                             "an early loss spike. Default: 0 (no warmup).")
    parser.add_argument("--focal-gamma", type=float, default=0.0,
                        help="Focal loss gamma (focusing parameter). 0 = standard BCE "
                             "(default). 2.0 = standard RetinaNet focal loss. When > 0, "
                             "replaces BCE+pos_weight with focal loss.")
    parser.add_argument("--focal-alpha", type=float, default=0.85,
                        help="Focal loss alpha (positive-class weight, analogous to "
                             "pos_weight). Only used when --focal-gamma > 0. Default 0.85.")
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Stop training if val_loss does not improve for this many "
                             "consecutive epochs. 0 = disabled (default). best.pt is "
                             "always the lowest-val-loss checkpoint seen so far.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = get_device()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CachedHeatmapDataset(args.train_cache)
    val_ds = CachedHeatmapDataset(args.val_cache)
    in_channels = int(torch.load(Path(args.train_cache) / train_ds.items[0]["path"], map_location="cpu", weights_only=False)["features"].shape[0])
    head = HeatmapHead(in_channels, args.hidden_channels).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=_collate,
    )

    start_epoch = 1
    best_dice = -1.0
    history: list[dict[str, float | int]] = []

    if args.resume:
        resume_path = work_dir / "latest.pt"
        best_path = work_dir / "best.pt"
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume requested but {resume_path} not found")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        head.load_state_dict(ckpt["head"])
        start_epoch = int(ckpt["epoch"]) + 1
        if best_path.exists():
            best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            best_dice = float(best_ckpt["val_dice"])
        history_path = work_dir / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text())
        logger.info("Resumed from %s (epoch %d, best_dice=%.3f, continuing to epoch %d)",
                    resume_path, start_epoch - 1, best_dice, args.epochs)

    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    if args.lr_scheduler == "cosine":
        warmup = max(0, int(args.warmup_epochs))
        if warmup > 0:
            # Linear warmup (0.1*lr -> lr over `warmup` epochs) then cosine decay
            # over the remainder. Lets us run a hotter peak LR without an early spike.
            warm = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup
            )
            cos = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - warmup)
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warm, cos], milestones=[warmup]
            )
            # Advance the schedule to the resume point if continuing a run.
            for _ in range(max(0, start_epoch - 1)):
                scheduler.step()
            logger.info("Warmup(%d)+Cosine scheduler active (epochs=%d, start_epoch=%d)",
                        warmup, args.epochs, start_epoch)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, last_epoch=start_epoch - 2
            )
            logger.info("CosineAnnealingLR scheduler active (T_max=%d, start_epoch=%d)",
                        args.epochs, start_epoch)

    if args.focal_gamma > 0.0:
        logger.info("Using focal loss (gamma=%.2f, alpha=%.2f) — pos_weight ignored",
                    args.focal_gamma, args.focal_alpha)
    if args.early_stopping_patience > 0:
        logger.info("Early stopping patience=%d epochs (tracking val_loss)",
                    args.early_stopping_patience)

    metadata = {
        "args": vars(args),
        "train_cache_metadata": train_ds.metadata,
        "val_cache_metadata": val_ds.metadata,
        "in_channels": in_channels,
    }

    best_val_loss = float("inf")
    no_improve_epochs = 0

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = _run_epoch(
            head, train_loader, optimizer, device,
            args.pos_weight, args.geometry_weight,
            focal_gamma=args.focal_gamma, focal_alpha=args.focal_alpha,
        )
        if scheduler is not None:
            scheduler.step()
        with torch.no_grad():
            val_metrics = _run_epoch(
                head, val_loader, None, device,
                args.pos_weight, args.geometry_weight,
                focal_gamma=args.focal_gamma, focal_alpha=args.focal_alpha,
            )
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
        # Persist history every epoch so an early stop still leaves the dice curve.
        (work_dir / "history.json").write_text(json.dumps(history, indent=2))

        # Early stopping: track val_loss (diverges before val_dice visibly degrades).
        if args.early_stopping_patience > 0:
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= args.early_stopping_patience:
                    logger.info(
                        "Early stopping at epoch %d (val_loss no improvement for %d epochs, "
                        "best=%.4f)",
                        epoch, args.early_stopping_patience, best_val_loss,
                    )
                    break

    (work_dir / "history.json").write_text(json.dumps(history, indent=2))
    logger.info("best val dice %.3f saved to %s", best_dice, work_dir / "best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
