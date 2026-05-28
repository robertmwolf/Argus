"""Sweep no-OBB centerline segment extraction settings for DINOv3 heatmaps.

This is the tuning companion to ``propose_dinov3_centerline_segments.py``.  It
runs the model once per validation image, then evaluates multiple seed
threshold/orientation-consistency settings from the same heatmap so proposal
tuning does not repeatedly pay the DINOv3 inference cost.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inference.device import get_device
from models.plain_dinov3.streak_heatmap import imagenet_normalize
from scripts.evaluate_dinov3_orientation_centerline import _load_checkpoint_model
from scripts.propose_dinov3_centerline_segments import (
    _extract_proposals,
    _proposal_gt_coverage,
    _to_gray,
)
from training.dinov3_orientation_centerline_dataset import DINOv3OrientationCenterlineDataset

logger = logging.getLogger(__name__)


def _parse_float_list(raw: str) -> list[float]:
    """Parse a comma-separated float list."""
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one float value")
    return values


def _parse_int_list(raw: str) -> list[int]:
    """Parse a comma-separated integer list."""
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def _new_counts() -> dict[str, Any]:
    """Create metric counters for one sweep configuration."""
    return {
        "positive_images": 0,
        "negative_images": 0,
        "positive_images_with_segment": 0,
        "negative_images_with_segment": 0,
        "positive_images_matched": 0,
        "n_segments": 0,
        "coverage_values": [],
        "positive_best_coverages": [],
        "negative_segment_counts": [],
        "positive_segment_counts": [],
    }


def _finalize_counts(counts: dict[str, Any]) -> dict[str, Any]:
    """Convert raw counters into compact sweep metrics."""
    positive_images = int(counts["positive_images"])
    negative_images = int(counts["negative_images"])
    n_images = positive_images + negative_images
    n_segments = int(counts["n_segments"])
    positive_with_segment = int(counts["positive_images_with_segment"])
    negative_with_segment = int(counts["negative_images_with_segment"])
    positive_matched = int(counts["positive_images_matched"])
    coverage_values = np.asarray(counts["coverage_values"], dtype=np.float32)
    positive_best = np.asarray(counts["positive_best_coverages"], dtype=np.float32)
    negative_segments = np.asarray(counts["negative_segment_counts"], dtype=np.float32)
    positive_segments = np.asarray(counts["positive_segment_counts"], dtype=np.float32)

    return {
        "n_images": n_images,
        "positive_images": positive_images,
        "negative_images": negative_images,
        "n_segments": n_segments,
        "segments_per_image": round(n_segments / n_images, 4) if n_images else 0.0,
        "positive_images_with_segment": positive_with_segment,
        "positive_segment_recall": round(positive_with_segment / positive_images, 4)
        if positive_images
        else 0.0,
        "positive_images_matched": positive_matched,
        "positive_matched_recall": round(positive_matched / positive_images, 4)
        if positive_images
        else 0.0,
        "negative_images_with_segment": negative_with_segment,
        "negative_segment_rate": round(negative_with_segment / negative_images, 4)
        if negative_images
        else 0.0,
        "positive_segments_per_image": round(float(positive_segments.mean()), 4)
        if positive_segments.size
        else 0.0,
        "negative_segments_per_image": round(float(negative_segments.mean()), 4)
        if negative_segments.size
        else 0.0,
        "best_coverage_mean": round(float(positive_best.mean()), 4) if positive_best.size else 0.0,
        "best_coverage_median": round(float(np.median(positive_best)), 4)
        if positive_best.size
        else 0.0,
        "best_coverage_p25": round(float(np.percentile(positive_best, 25)), 4)
        if positive_best.size
        else 0.0,
        "best_coverage_p75": round(float(np.percentile(positive_best, 75)), 4)
        if positive_best.size
        else 0.0,
        "proposal_coverage_mean": round(float(coverage_values.mean()), 4)
        if coverage_values.size
        else 0.0,
    }


def _config_key(
    threshold: float,
    min_orientation_consistency: float,
    min_component_pixels: int,
    max_components: int,
) -> str:
    """Create a stable JSON key for one sweep setting."""
    return (
        f"t{threshold:.3f}_oc{min_orientation_consistency:.3f}"
        f"_min{min_component_pixels}_max{max_components}"
    )


def main() -> int:
    """Run a cached segment proposal parameter sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--output-json", default="results/dinov3_centerline_segment_sweep/sweep.json")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--thresholds", default="0.30,0.40,0.50,0.60,0.70,0.80,0.85")
    parser.add_argument("--min-orientation-consistencies", default="0.45,0.55,0.65,0.75")
    parser.add_argument("--min-component-pixels", default="4,8,16")
    parser.add_argument("--max-components-per-image", default="4,8")
    parser.add_argument("--target-threshold", type=float, default=0.05)
    parser.add_argument("--proposal-tolerance-px", type=int, default=6)
    parser.add_argument("--proposal-gt-coverage", type=float, default=0.10)
    parser.add_argument("--orientation-neighbor-bins", type=int, default=1)
    parser.add_argument("--crop-padding", type=int, default=48)
    parser.add_argument("--radon-search-degrees", type=float, default=12.0)
    parser.add_argument("--radon-step-degrees", type=float, default=0.5)
    parser.add_argument("--min-length-px", type=float, default=16.0)
    parser.add_argument("--extension-px", type=float, default=8.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--preserve-image-bit-depth", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    thresholds = _parse_float_list(args.thresholds)
    consistencies = _parse_float_list(args.min_orientation_consistencies)
    min_pixels_values = _parse_int_list(args.min_component_pixels)
    max_component_values = _parse_int_list(args.max_components_per_image)

    config_order: list[dict[str, Any]] = []
    counts_by_key: dict[str, dict[str, Any]] = {}
    for threshold in thresholds:
        for consistency in consistencies:
            for min_pixels in min_pixels_values:
                for max_components in max_component_values:
                    key = _config_key(threshold, consistency, min_pixels, max_components)
                    config = {
                        "key": key,
                        "threshold": threshold,
                        "min_orientation_consistency": consistency,
                        "min_component_pixels": min_pixels,
                        "max_components_per_image": max_components,
                    }
                    config_order.append(config)
                    counts_by_key[key] = _new_counts()

    device = get_device()
    model, train_args = _load_checkpoint_model(Path(args.checkpoint), args.weights, device)
    annotations = args.annotations or train_args.get("val_annotations", "data/annotations/val.json")
    preserve_bit_depth = bool(args.preserve_image_bit_depth or train_args.get("preserve_image_bit_depth", False))
    orientation_bins = int(train_args.get("orientation_bins", 18))
    dataset = DINOv3OrientationCenterlineDataset(
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

    logger.info("sweeping %d configs on %d images", len(config_order), len(dataset))
    with torch.no_grad():
        for idx in range(len(dataset)):
            item = dataset[idx]
            sample = dataset.samples[idx]
            image = item["image"].unsqueeze(0).to(device)  # type: ignore[union-attr]
            logits = model(imagenet_normalize(image))
            if logits.shape[-2:] != image.shape[-2:]:
                logits = F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
            heat = probs.max(axis=0).astype(np.float32)
            bins = probs.argmax(axis=0).astype(np.int32)
            gray = _to_gray(item["image"].numpy())  # type: ignore[union-attr]
            target_heat = item["target"].max(dim=0).values.numpy()  # type: ignore[union-attr]
            gt_mask = target_heat > args.target_threshold
            gt_positive = bool(gt_mask.any())

            for config in config_order:
                proposals = _extract_proposals(
                    heat=heat,
                    bins=bins,
                    gray=gray,
                    sample=sample,
                    threshold=float(config["threshold"]),
                    min_component_pixels=int(config["min_component_pixels"]),
                    orientation_neighbor_bins=args.orientation_neighbor_bins,
                    min_orientation_consistency=float(config["min_orientation_consistency"]),
                    crop_padding=args.crop_padding,
                    radon_search_degrees=args.radon_search_degrees,
                    radon_step_degrees=args.radon_step_degrees,
                    min_length_px=args.min_length_px,
                    extension_px=args.extension_px,
                    max_components=int(config["max_components_per_image"]),
                    n_bins=orientation_bins,
                )
                coverages = [
                    _proposal_gt_coverage(
                        proposal,
                        gt_mask,
                        tolerance_px=args.proposal_tolerance_px,
                    )
                    for proposal in proposals
                ]
                matched = bool(coverages and max(coverages) >= args.proposal_gt_coverage)
                counts = counts_by_key[str(config["key"])]
                counts["n_segments"] += len(proposals)
                counts["coverage_values"].extend(coverages)
                if gt_positive:
                    counts["positive_images"] += 1
                    counts["positive_segment_counts"].append(len(proposals))
                    counts["positive_best_coverages"].append(max(coverages) if coverages else 0.0)
                    if proposals:
                        counts["positive_images_with_segment"] += 1
                    if matched:
                        counts["positive_images_matched"] += 1
                else:
                    counts["negative_images"] += 1
                    counts["negative_segment_counts"].append(len(proposals))
                    if proposals:
                        counts["negative_images_with_segment"] += 1

            if (idx + 1) % 25 == 0:
                logger.info("processed %d/%d images", idx + 1, len(dataset))

    rows: list[dict[str, Any]] = []
    for config in config_order:
        row = {
            **config,
            **_finalize_counts(counts_by_key[str(config["key"])]),
        }
        rows.append(row)
    rows.sort(
        key=lambda row: (
            float(row["positive_matched_recall"]),
            -float(row["negative_segment_rate"]),
            float(row["best_coverage_median"]),
        ),
        reverse=True,
    )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.checkpoint),
        "annotations": str(annotations),
        "n_configs": len(config_order),
        "n_images": len(dataset),
        "proposal_tolerance_px": args.proposal_tolerance_px,
        "proposal_gt_coverage": args.proposal_gt_coverage,
        "target_threshold": args.target_threshold,
        "results": rows,
    }
    output_json.write_text(json.dumps(payload, indent=2))

    output_csv = Path(args.output_csv) if args.output_csv else output_json.with_suffix(".csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    logger.info("wrote %s and %s", output_json, output_csv)
    if rows:
        best = rows[0]
        logger.info(
            "best: threshold=%.3f orient=%.3f min_pixels=%d max_components=%d "
            "matched_recall=%.4f neg_rate=%.4f segs/image=%.4f",
            best["threshold"],
            best["min_orientation_consistency"],
            best["min_component_pixels"],
            best["max_components_per_image"],
            best["positive_matched_recall"],
            best["negative_segment_rate"],
            best["segments_per_image"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
