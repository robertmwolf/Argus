"""Train the plain DINOv3 heatmap head from cached frozen features."""

from __future__ import annotations

import argparse
import dataclasses
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


# ── Loss functions ────────────────────────────────────────────────────────────

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


def _asl_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma_neg: float = 4.0,
    gamma_pos: float = 0.0,
    margin: float = 0.05,
) -> torch.Tensor:
    """Asymmetric Loss (Ridnik et al. 2021) for binary heatmap pixels.

    Applies a probability margin to negative pixels: any negative predicted below
    `margin` contributes zero loss (it is already handled).  Only hard negatives
    — pixels the model confidently mispredicts — are penalised.  This directly
    trains the model to reject the borderline background patches that become FPs.

    # Source: Ridnik et al. 2021 — Asymmetric Loss For Multi-Label Classification
    # Ref: https://arxiv.org/abs/2009.14119
    """
    p = torch.sigmoid(logits)
    # Shift negatives down by margin; easy negatives (p < margin) → zero loss.
    p_m = (p - margin).clamp(min=0.0)
    # Positive branch: mild focusing on hard positives.
    loss_pos = -((1.0 - p) ** gamma_pos) * torch.log(p.clamp(min=1e-8))
    # Negative branch: strong focusing on hard negatives after the shift.
    loss_neg = -(p_m ** gamma_neg) * torch.log((1.0 - p_m).clamp(min=1e-8))
    return (target * loss_pos + (1.0 - target) * loss_neg).mean()


def _tversky_score(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.6,
) -> torch.Tensor:
    """Tversky index with FP weight alpha and FN weight (1-alpha).

    alpha > 0.5 penalises false positives more than false negatives, directly
    trading recall for precision.
    """
    probs = torch.sigmoid(logits)
    beta = 1.0 - alpha
    tp = (probs * target).sum(dim=(1, 2, 3))
    fp = (probs * (1.0 - target)).sum(dim=(1, 2, 3))
    fn = ((1.0 - probs) * target).sum(dim=(1, 2, 3))
    return ((tp + 1e-6) / (tp + alpha * fp + beta * fn + 1e-6)).mean()


def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, (3, 1), stride=(1, 1), padding=(1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), stride=(1, 1), padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, (3, 3), stride=(1, 1), padding=(1, 1))


def _soft_skeleton(img: torch.Tensor, num_iters: int = 3) -> torch.Tensor:
    """Differentiable morphological skeleton via iterative open-subtract.

    # Source: Shit et al. 2021 — clDice: a Novel Topology-Preserving Loss
    # Ref: https://arxiv.org/abs/2003.07311
    """
    img1 = _soft_dilate(_soft_erode(img))
    skel = F.relu(img - img1)
    for _ in range(num_iters - 1):
        img = _soft_erode(img)
        img1 = _soft_dilate(_soft_erode(img))
        skel = skel + F.relu(img - img1)
    return skel


def _cldice_score(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_iters: int = 3,
) -> torch.Tensor:
    """clDice score: topology-preserving metric for tubular/linear structures.

    Computes Dice on the *skeletons* of the prediction and ground truth rather
    than the filled masks.  Explicitly rewards connected, linear predictions and
    penalises false-positive blobs that do not correspond to a real streak.

    # Source: Shit et al. 2021 — clDice: a Novel Topology-Preserving Loss
    # Ref: https://arxiv.org/abs/2003.07311
    """
    probs = torch.sigmoid(logits).clamp(1e-4, 1.0 - 1e-4)
    gt = target.clamp(1e-4, 1.0 - 1e-4)

    skel_pred = _soft_skeleton(probs, num_iters)
    skel_gt = _soft_skeleton(gt, num_iters)

    # T_prec: fraction of the predicted skeleton that overlaps the GT mask.
    tprec = ((skel_pred * gt).sum(dim=(1, 2, 3)) + 1e-6) / (skel_pred.sum(dim=(1, 2, 3)) + 1e-6)
    # T_sens: fraction of the GT skeleton that overlaps the predicted mask.
    tsens = ((skel_gt * probs).sum(dim=(1, 2, 3)) + 1e-6) / (skel_gt.sum(dim=(1, 2, 3)) + 1e-6)

    return (2.0 * tprec * tsens / (tprec + tsens + 1e-6)).mean()


# ── Loss configuration ────────────────────────────────────────────────────────

@dataclasses.dataclass
class LossConfig:
    """All hyperparameters for the heatmap loss computation."""

    mode: str = "focal_dice"
    """One of: focal_dice, asl_dice, focal_cldice, tversky, asl_cldice."""

    # focal / BCE
    focal_gamma: float = 2.0
    focal_alpha: float = 0.85
    pos_weight: float = 20.0

    # ASL
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 0.0
    asl_margin: float = 0.05

    # Tversky (FP weight; FN weight = 1 - tversky_alpha)
    tversky_alpha: float = 0.6

    # clDice
    cldice_iters: int = 3

    # geometry head
    geometry_weight: float = 0.25


# ── Training loop ─────────────────────────────────────────────────────────────

def _run_epoch(
    head: HeatmapHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    loss_cfg: LossConfig,
) -> dict[str, float]:
    training = optimizer is not None
    head.train(training)
    total_loss = 0.0
    total_seg = 0.0    # seg score (dice / tversky / cldice depending on mode)
    total_prec = 0.0   # pixel-level precision at t=0.5 (proxy for FP rate)
    batches = 0
    pos_weight_tensor = torch.tensor([loss_cfg.pos_weight], device=device)

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

        mode = loss_cfg.mode
        if mode == "focal_dice":
            heatmap_loss = _focal_loss(logits, target, loss_cfg.focal_alpha, loss_cfg.focal_gamma) \
                if loss_cfg.focal_gamma > 0.0 \
                else F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight_tensor)
            seg_score = _dice_score(logits, target)
        elif mode == "asl_dice":
            heatmap_loss = _asl_loss(logits, target, loss_cfg.asl_gamma_neg, loss_cfg.asl_gamma_pos, loss_cfg.asl_margin)
            seg_score = _dice_score(logits, target)
        elif mode == "focal_cldice":
            heatmap_loss = _focal_loss(logits, target, loss_cfg.focal_alpha, loss_cfg.focal_gamma)
            seg_score = _cldice_score(logits, target, loss_cfg.cldice_iters)
        elif mode == "tversky":
            heatmap_loss = _focal_loss(logits, target, loss_cfg.focal_alpha, loss_cfg.focal_gamma)
            seg_score = _tversky_score(logits, target, loss_cfg.tversky_alpha)
        elif mode == "asl_cldice":
            heatmap_loss = _asl_loss(logits, target, loss_cfg.asl_gamma_neg, loss_cfg.asl_gamma_pos, loss_cfg.asl_margin)
            seg_score = _cldice_score(logits, target, loss_cfg.cldice_iters)
        else:
            raise ValueError(f"Unknown loss mode: {mode!r}")

        seg_loss = 1.0 - seg_score

        mask = target.expand_as(geometry) > 0
        if bool(mask.any()):
            geom_loss = F.smooth_l1_loss(geom_pred[mask], geometry[mask])
        else:
            geom_loss = torch.zeros((), device=device)

        loss = heatmap_loss + seg_loss + loss_cfg.geometry_weight * geom_loss

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_seg += float(seg_score.detach().cpu())

        # Pixel-level precision at threshold 0.5: tracks FP activation rate.
        with torch.no_grad():
            preds = (torch.sigmoid(logits.detach()) > 0.5).float()
            tp_px = (preds * target).sum()
            fp_px = (preds * (1.0 - target)).sum()
            total_prec += float(tp_px / (tp_px + fp_px + 1e-6))

        batches += 1

    denom = max(batches, 1)
    return {
        "loss": total_loss / denom,
        "seg": total_seg / denom,    # dice / tversky / cldice score
        "prec": total_prec / denom,  # pixel precision at t=0.5
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--work-dir", default="weights/run_plain_dinov3_heatmap_cached")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers.  0=synchronous (safe for MPS). "
                             "Set >0 only on CUDA where fork/spawn is stable.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest.pt in --work-dir. Restores head weights, "
                             "optimizer state, best_seg, and history, then continues "
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
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Stop training if val_loss does not improve for this many "
                             "consecutive epochs. 0 = disabled (default). best.pt is "
                             "always the lowest-val-loss checkpoint seen so far.")

    # ── Loss mode ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--loss-mode",
        choices=["focal_dice", "asl_dice", "focal_cldice", "tversky", "asl_cldice"],
        default="focal_dice",
        help=(
            "Loss function combination.\n"
            "  focal_dice   — focal + Dice (v8 baseline)\n"
            "  asl_dice     — Asymmetric Loss + Dice (precision-targeting)\n"
            "  focal_cldice — focal + clDice (topology-aware; rewards linear connectivity)\n"
            "  tversky      — focal + Tversky(FP-penalising, alpha>0.5)\n"
            "  asl_cldice   — ASL + clDice (combined precision + topology)"
        ),
    )

    # focal / BCE params (used by focal_dice, focal_cldice, tversky)
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma. 0 = BCE+pos_weight. Default 2.0.")
    parser.add_argument("--focal-alpha", type=float, default=0.85,
                        help="Focal loss alpha (positive-class weight). Default 0.85.")
    parser.add_argument("--pos-weight", type=float, default=20.0,
                        help="BCE pos_weight (only when --focal-gamma 0). Default 20.0.")

    # ASL params (used by asl_dice, asl_cldice)
    parser.add_argument("--asl-gamma-neg", type=float, default=4.0,
                        help="ASL focusing parameter for negatives. Default 4.0.")
    parser.add_argument("--asl-gamma-pos", type=float, default=0.0,
                        help="ASL focusing parameter for positives. Default 0.0.")
    parser.add_argument("--asl-margin", type=float, default=0.05,
                        help="ASL probability margin (easy negatives below this → zero loss). "
                             "Default 0.05.")

    # Tversky param (used by tversky mode)
    parser.add_argument("--tversky-alpha", type=float, default=0.6,
                        help="Tversky FP weight. beta = 1 - alpha. alpha > 0.5 penalises "
                             "false positives more than false negatives. Default 0.6.")

    # clDice param (used by focal_cldice, asl_cldice)
    parser.add_argument("--cldice-iters", type=int, default=3,
                        help="Soft-skeleton iterations for clDice. Default 3 (appropriate "
                             "for 28×28 feature maps; increase for higher-res heatmaps).")

    parser.add_argument("--geometry-weight", type=float, default=0.25)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = get_device()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    loss_cfg = LossConfig(
        mode=args.loss_mode,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        pos_weight=args.pos_weight,
        asl_gamma_neg=args.asl_gamma_neg,
        asl_gamma_pos=args.asl_gamma_pos,
        asl_margin=args.asl_margin,
        tversky_alpha=args.tversky_alpha,
        cldice_iters=args.cldice_iters,
        geometry_weight=args.geometry_weight,
    )

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
    best_seg = -1.0
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
            best_seg = float(best_ckpt.get("val_seg", best_ckpt.get("val_dice", -1.0)))
        history_path = work_dir / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text())
        logger.info("Resumed from %s (epoch %d, best_seg=%.3f, continuing to epoch %d)",
                    resume_path, start_epoch - 1, best_seg, args.epochs)

    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    if args.lr_scheduler == "cosine":
        warmup = max(0, int(args.warmup_epochs))
        if warmup > 0:
            warm = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup
            )
            cos = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - warmup)
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warm, cos], milestones=[warmup]
            )
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

    logger.info("Loss mode: %s", loss_cfg.mode)
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
        train_metrics = _run_epoch(head, train_loader, optimizer, device, loss_cfg)
        if scheduler is not None:
            scheduler.step()
        with torch.no_grad():
            val_metrics = _run_epoch(head, val_loader, None, device, loss_cfg)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_seg": train_metrics["seg"],
            "val_loss": val_metrics["loss"],
            "val_seg": val_metrics["seg"],
            "val_prec": val_metrics["prec"],
        }
        history.append(row)
        logger.info(
            "epoch=%d train_loss=%.4f train_seg=%.3f "
            "val_loss=%.4f val_seg=%.3f val_prec=%.3f",
            epoch, row["train_loss"], row["train_seg"],
            row["val_loss"], row["val_seg"], row["val_prec"],
        )
        payload = {
            "head": head.state_dict(),
            "epoch": epoch,
            "val_seg": val_metrics["seg"],
            "val_dice": val_metrics["seg"],  # keep old key for backward compat
            **metadata,
        }
        torch.save(payload, work_dir / "latest.pt")
        if val_metrics["seg"] > best_seg:
            best_seg = val_metrics["seg"]
            torch.save(payload, work_dir / "best.pt")
        (work_dir / "history.json").write_text(json.dumps(history, indent=2))

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
    logger.info("best val seg %.3f saved to %s", best_seg, work_dir / "best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
