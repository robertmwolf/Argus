"""Evaluate DINOv3 orientation-centerline checkpoints as endpoint segments."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
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


def _parse_thresholds(raw: str) -> list[float]:
    """Parse a comma-separated threshold list."""
    thresholds = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not thresholds:
        raise ValueError("At least one threshold is required")
    return thresholds


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Return precision, recall, and F1 from integer counts."""
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _component_counts(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    min_pixels: int,
    iou_threshold: float,
) -> tuple[int, int, int]:
    """Match predicted and target centerline components by mask IoU."""
    pred_labels, pred_count = ndimage.label(pred_mask)
    gt_labels, gt_count = ndimage.label(gt_mask)
    pred_ids = [
        label_id
        for label_id in range(1, pred_count + 1)
        if int((pred_labels == label_id).sum()) >= min_pixels
    ]
    gt_ids = [
        label_id
        for label_id in range(1, gt_count + 1)
        if int((gt_labels == label_id).sum()) >= min_pixels
    ]
    matched_gt: set[int] = set()
    tp = 0
    for pred_id in pred_ids:
        pred_component = pred_labels == pred_id
        best_iou = 0.0
        best_gt = -1
        for gt_id in gt_ids:
            if gt_id in matched_gt:
                continue
            gt_component = gt_labels == gt_id
            intersection = int(np.logical_and(pred_component, gt_component).sum())
            if intersection == 0:
                continue
            union = int(np.logical_or(pred_component, gt_component).sum())
            iou = intersection / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_id
        if best_iou >= iou_threshold and best_gt >= 0:
            tp += 1
            matched_gt.add(best_gt)
    fp = len(pred_ids) - tp
    fn = len(gt_ids) - tp
    return tp, fp, fn


def _distance_tolerant_pixel_counts(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    tolerance_px: float,
) -> tuple[int, int, int]:
    """Count centerline pixels with distance tolerance instead of exact overlap."""
    if pred_mask.any():
        gt_distance = ndimage.distance_transform_edt(~gt_mask)
        tp = int(np.logical_and(pred_mask, gt_distance <= tolerance_px).sum())
        fp = int(np.logical_and(pred_mask, gt_distance > tolerance_px).sum())
    else:
        tp = 0
        fp = 0
    if gt_mask.any():
        pred_distance = ndimage.distance_transform_edt(~pred_mask)
        fn = int(np.logical_and(gt_mask, pred_distance > tolerance_px).sum())
    else:
        fn = 0
    return tp, fp, fn


def _distance_tolerant_component_counts(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    min_pixels: int,
    tolerance_px: float,
    coverage_threshold: float,
) -> tuple[int, int, int]:
    """Match centerline components by mutual distance-tolerant coverage."""
    pred_labels, pred_count = ndimage.label(pred_mask)
    gt_labels, gt_count = ndimage.label(gt_mask)
    pred_ids = [
        label_id
        for label_id in range(1, pred_count + 1)
        if int((pred_labels == label_id).sum()) >= min_pixels
    ]
    gt_ids = [
        label_id
        for label_id in range(1, gt_count + 1)
        if int((gt_labels == label_id).sum()) >= min_pixels
    ]
    gt_distances: dict[int, np.ndarray] = {}
    pred_distances: dict[int, np.ndarray] = {}
    for gt_id in gt_ids:
        gt_distances[gt_id] = ndimage.distance_transform_edt(gt_labels != gt_id)
    for pred_id in pred_ids:
        pred_distances[pred_id] = ndimage.distance_transform_edt(pred_labels != pred_id)

    matched_gt: set[int] = set()
    tp = 0
    for pred_id in pred_ids:
        pred_component = pred_labels == pred_id
        best_score = 0.0
        best_gt = -1
        for gt_id in gt_ids:
            if gt_id in matched_gt:
                continue
            gt_component = gt_labels == gt_id
            pred_coverage = float((gt_distances[gt_id][pred_component] <= tolerance_px).mean())
            gt_coverage = float((pred_distances[pred_id][gt_component] <= tolerance_px).mean())
            score = min(pred_coverage, gt_coverage)
            if score > best_score:
                best_score = score
                best_gt = gt_id
        if best_score >= coverage_threshold and best_gt >= 0:
            tp += 1
            matched_gt.add(best_gt)
    fp = len(pred_ids) - tp
    fn = len(gt_ids) - tp
    return tp, fp, fn


def _empty_threshold_accumulator() -> dict[str, int]:
    """Create integer counters for one threshold."""
    return {
        "pixel_tp": 0,
        "pixel_fp": 0,
        "pixel_fn": 0,
        "image_tp": 0,
        "image_fp": 0,
        "image_fn": 0,
        "image_tn": 0,
        "component_tp": 0,
        "component_fp": 0,
        "component_fn": 0,
        "tolerant_pixel_tp": 0,
        "tolerant_pixel_fp": 0,
        "tolerant_pixel_fn": 0,
        "tolerant_component_tp": 0,
        "tolerant_component_fp": 0,
        "tolerant_component_fn": 0,
        "orientation_correct_pred_gt": 0,
        "orientation_total_pred_gt": 0,
    }


def _finalize_threshold_metrics(threshold: float, counts: dict[str, int]) -> dict[str, Any]:
    """Convert raw counts into reported metrics for one threshold."""
    pixel = _prf(counts["pixel_tp"], counts["pixel_fp"], counts["pixel_fn"])
    image = _prf(counts["image_tp"], counts["image_fp"], counts["image_fn"])
    component = _prf(counts["component_tp"], counts["component_fp"], counts["component_fn"])
    tolerant_pixel = _prf(
        counts["tolerant_pixel_tp"],
        counts["tolerant_pixel_fp"],
        counts["tolerant_pixel_fn"],
    )
    tolerant_component = _prf(
        counts["tolerant_component_tp"],
        counts["tolerant_component_fp"],
        counts["tolerant_component_fn"],
    )
    total_images = counts["image_tp"] + counts["image_fp"] + counts["image_fn"] + counts["image_tn"]
    image_accuracy = (
        (counts["image_tp"] + counts["image_tn"]) / total_images if total_images > 0 else 0.0
    )
    orientation_total = counts["orientation_total_pred_gt"]
    orientation_accuracy = (
        counts["orientation_correct_pred_gt"] / orientation_total if orientation_total > 0 else 0.0
    )
    return {
        "threshold": threshold,
        "pixel": pixel,
        "distance_tolerant_pixel": tolerant_pixel,
        "image": {**image, "accuracy": round(image_accuracy, 4)},
        "components": component,
        "distance_tolerant_components": tolerant_component,
        "counts": counts,
        "orientation_accuracy_on_predicted_gt_pixels": round(orientation_accuracy, 4),
    }


def _load_checkpoint_model(
    checkpoint_path: Path,
    weights_override: str | None,
    device: torch.device,
) -> tuple[DINOv3OrientationCenterline, dict[str, Any]]:
    """Load a centerline checkpoint and reconstruct its model."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    train_args = dict(ckpt.get("args", {}))
    weights = weights_override or train_args.get("weights", "weights/dinov3_vitb16_lvd1689m.pth")
    model = DINOv3OrientationCenterline(
        model_size=train_args.get("model_size", "base"),
        weights=weights,
        decoder_channels=int(train_args.get("decoder_channels", 192)),
        orientation_bins=int(train_args.get("orientation_bins", 18)),
        last_layers=int(train_args.get("last_layers", 4)),
        freeze_backbone=True,
    ).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        logger.info("Checkpoint missing %d model keys, first=%s", len(missing), missing[:3])
    if unexpected:
        logger.info("Checkpoint has %d unexpected model keys, first=%s", len(unexpected), unexpected[:3])
    model.eval()
    return model, train_args


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", default=None, help="Override DINOv3 backbone weights from checkpoint metadata")
    parser.add_argument("--output", default="results/dinov3_orientation_centerline/metrics.json")
    parser.add_argument("--thresholds", default="0.30,0.50,0.70,0.85")
    parser.add_argument("--target-threshold", type=float, default=0.05)
    parser.add_argument("--component-iou", type=float, default=0.10)
    parser.add_argument("--distance-tolerance-px", type=float, default=3.0)
    parser.add_argument("--component-coverage", type=float, default=0.30)
    parser.add_argument("--min-component-pixels", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--preserve-image-bit-depth", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    thresholds = _parse_thresholds(args.thresholds)
    device = get_device()
    checkpoint_path = Path(args.checkpoint)
    model, train_args = _load_checkpoint_model(checkpoint_path, args.weights, device)

    annotations = args.annotations or train_args.get("val_annotations", "data/annotations/val.json")
    preserve_bit_depth = bool(args.preserve_image_bit_depth or train_args.get("preserve_image_bit_depth", False))
    ds = DINOv3OrientationCenterlineDataset(
        annotation_file=annotations,
        split="val",
        tile_size=int(train_args.get("tile_size", 2560)),
        image_size=int(train_args.get("image_size", 1024)),
        orientation_bins=int(train_args.get("orientation_bins", 18)),
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
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_centerline_batch,
    )

    threshold_counts = {threshold: _empty_threshold_accumulator() for threshold in thresholds}
    soft_dice_values: list[float] = []
    orientation_correct_gt = 0
    orientation_neighbor_gt = 0
    orientation_total_gt = 0
    max_scores: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            images = imagenet_normalize(batch["image"].to(device))
            target = batch["target"].to(device)
            logits = model(images)
            if logits.shape[-2:] != target.shape[-2:]:
                logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
            probs = torch.sigmoid(logits)
            probs_heat = probs.amax(dim=1)
            target_heat = target.amax(dim=1)
            intersection = (probs_heat * target_heat).sum(dim=(1, 2))
            denom = probs_heat.sum(dim=(1, 2)) + target_heat.sum(dim=(1, 2)) + 1e-6
            soft_dice_values.extend(((2.0 * intersection + 1e-6) / denom).cpu().numpy().tolist())

            probs_np = probs.cpu().numpy()
            probs_heat_np = probs_heat.cpu().numpy()
            target_np = target.cpu().numpy()
            target_heat_np = target_heat.cpu().numpy()
            image_ids = batch["image_id"].cpu().numpy().tolist()
            file_names = batch["file_name"]
            for idx in range(probs_heat_np.shape[0]):
                gt_mask = target_heat_np[idx] > args.target_threshold
                pred_bins = probs_np[idx].argmax(axis=0)
                gt_bins = target_np[idx].argmax(axis=0)
                gt_positive = bool(gt_mask.any())
                if gt_positive:
                    correct = pred_bins[gt_mask] == gt_bins[gt_mask]
                    diff = np.abs(pred_bins[gt_mask] - gt_bins[gt_mask])
                    wrapped_diff = np.minimum(diff, int(train_args.get("orientation_bins", 18)) - diff)
                    orientation_correct_gt += int(correct.sum())
                    orientation_neighbor_gt += int((wrapped_diff <= 1).sum())
                    orientation_total_gt += int(gt_mask.sum())
                max_scores.append({
                    "image_id": int(image_ids[idx]),
                    "file_name": str(file_names[idx]),
                    "target_positive": gt_positive,
                    "max_score": float(probs_heat_np[idx].max()),
                })

                for threshold in thresholds:
                    pred_mask = probs_heat_np[idx] >= threshold
                    counts = threshold_counts[threshold]
                    counts["pixel_tp"] += int(np.logical_and(pred_mask, gt_mask).sum())
                    counts["pixel_fp"] += int(np.logical_and(pred_mask, ~gt_mask).sum())
                    counts["pixel_fn"] += int(np.logical_and(~pred_mask, gt_mask).sum())
                    pred_positive = int(pred_mask.sum()) >= args.min_component_pixels
                    if pred_positive and gt_positive:
                        counts["image_tp"] += 1
                    elif pred_positive and not gt_positive:
                        counts["image_fp"] += 1
                    elif not pred_positive and gt_positive:
                        counts["image_fn"] += 1
                    else:
                        counts["image_tn"] += 1
                    ctp, cfp, cfn = _component_counts(
                        pred_mask,
                        gt_mask,
                        min_pixels=args.min_component_pixels,
                        iou_threshold=args.component_iou,
                    )
                    counts["component_tp"] += ctp
                    counts["component_fp"] += cfp
                    counts["component_fn"] += cfn
                    dtp, dfp, dfn = _distance_tolerant_pixel_counts(
                        pred_mask,
                        gt_mask,
                        tolerance_px=args.distance_tolerance_px,
                    )
                    counts["tolerant_pixel_tp"] += dtp
                    counts["tolerant_pixel_fp"] += dfp
                    counts["tolerant_pixel_fn"] += dfn
                    dctp, dcfp, dcfn = _distance_tolerant_component_counts(
                        pred_mask,
                        gt_mask,
                        min_pixels=args.min_component_pixels,
                        tolerance_px=args.distance_tolerance_px,
                        coverage_threshold=args.component_coverage,
                    )
                    counts["tolerant_component_tp"] += dctp
                    counts["tolerant_component_fp"] += dcfp
                    counts["tolerant_component_fn"] += dcfn
                    pred_gt_mask = np.logical_and(pred_mask, gt_mask)
                    if pred_gt_mask.any():
                        counts["orientation_correct_pred_gt"] += int(
                            (pred_bins[pred_gt_mask] == gt_bins[pred_gt_mask]).sum()
                        )
                        counts["orientation_total_pred_gt"] += int(pred_gt_mask.sum())

    threshold_metrics = [
        _finalize_threshold_metrics(threshold, threshold_counts[threshold])
        for threshold in thresholds
    ]
    best_by_pixel_f1 = max(threshold_metrics, key=lambda item: item["pixel"]["f1"])
    best_by_tolerant_pixel_f1 = max(
        threshold_metrics,
        key=lambda item: item["distance_tolerant_pixel"]["f1"],
    )
    best_by_image_f1 = max(threshold_metrics, key=lambda item: item["image"]["f1"])
    orientation_accuracy_gt = (
        orientation_correct_gt / orientation_total_gt if orientation_total_gt > 0 else 0.0
    )
    orientation_neighbor_accuracy_gt = (
        orientation_neighbor_gt / orientation_total_gt if orientation_total_gt > 0 else 0.0
    )
    payload = {
        "checkpoint": str(checkpoint_path),
        "annotations": str(annotations),
        "n_samples": len(ds),
        "soft_dice": round(float(np.mean(soft_dice_values)) if soft_dice_values else 0.0, 4),
        "orientation_accuracy_on_gt_pixels": round(orientation_accuracy_gt, 4),
        "orientation_within_1_bin_on_gt_pixels": round(orientation_neighbor_accuracy_gt, 4),
        "threshold_metrics": threshold_metrics,
        "best_by_pixel_f1": best_by_pixel_f1,
        "best_by_distance_tolerant_pixel_f1": best_by_tolerant_pixel_f1,
        "best_by_image_f1": best_by_image_f1,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    (out_path.parent / "image_scores.json").write_text(json.dumps(max_scores, indent=2))
    logger.info(
        "wrote %s soft_dice=%.4f best_pixel_t=%.2f best_image_t=%.2f",
        out_path,
        payload["soft_dice"],
        best_by_pixel_f1["threshold"],
        best_by_image_f1["threshold"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
