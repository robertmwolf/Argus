"""Endpoint error distribution analysis for ARGUS heatmap models.

Matches predictions to ground truth (same three-criterion matcher used by
geometry_metrics.py), then decomposes endpoint error into signed along-track
components so you can tell whether the model is *too long* or *too short* in
the aggregate.

Outputs
-------
- Console summary with distribution statistics and directional bias.
- ``--output PATH``  — JSON file with per-pair detail and aggregate stats.
- ``--top-n N``      — list the N worst matched pairs (by symmetric endpoint
  error) to help identify labelling errors vs model errors.
- ``--patch-annotations PATH``  — non-destructively adds
  ``"review_status": "pending"`` to the worst-N annotations in the COCO JSON
  so they surface in your annotation review tool.  Writes a backup (.bak)
  before modifying.

Signed error convention
-----------------------
For each matched pair, both prediction endpoints are projected onto the GT axis.
Positive signed error means the prediction *extends beyond* the GT endpoint
(too long on that side); negative means it falls *short* (too short).

Usage
-----
  python scripts/analyze_endpoint_errors.py \\
      --predictions results/run15_vits/balanced_v1/pf85/predictions_t070.json \\
      --annotations $ARGUS_DATA_ROOT/annotations/val_balanced_v1.json \\
      --output results/run15_vits/balanced_v1/pf85/endpoint_error_analysis.json \\
      --top-n 20

  # also flag the worst 20 for re-annotation:
      --patch-annotations $ARGUS_DATA_ROOT/annotations/val_balanced_v1.json
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.streak_metrics import (
    _angle_error_deg,
    _greedy_match_segments,
    _load_gt_from_coco,
    _perpendicular_offset,
    _length_1d_iou,
)
from inference.streak_segment import StreakSegment, detection_dict_to_segment
from training.annotation_endpoints import annotation_to_endpoints

# ---------------------------------------------------------------------------
# Signed along-track endpoint error
# ---------------------------------------------------------------------------

def _signed_endpoint_errors(
    pred: StreakSegment, gt: StreakSegment
) -> tuple[float, float]:
    """Signed along-track error for each GT endpoint.

    Projects both prediction endpoints onto the GT axis, picks the assignment
    that minimises total error, then returns signed residuals.

    Positive = prediction extends *beyond* the GT endpoint (too long).
    Negative = prediction falls *short* of the GT endpoint (too short).

    Returns:
        (err_p1, err_p2) — signed errors for the two GT endpoints.
    """
    rad = math.radians(gt.angle_deg)
    cos_g = math.cos(rad)
    sin_g = math.sin(rad)

    # Project each prediction endpoint onto the GT axis (relative to GT centre)
    def proj(px: float, py: float) -> float:
        return (px - gt.cx) * cos_g + (py - gt.cy) * sin_g

    t1 = proj(pred.x1, pred.y1)
    t2 = proj(pred.x2, pred.y2)

    # GT endpoints in the same axis frame
    half = gt.length_px / 2.0
    gt_lo = -half  # maps to whichever GT endpoint is "first" along the axis
    gt_hi = +half

    # Direct assignment: pred p1 ↔ gt_lo, pred p2 ↔ gt_hi
    direct_err = abs(t1 - gt_lo) + abs(t2 - gt_hi)
    # Swapped assignment
    swap_err = abs(t1 - gt_hi) + abs(t2 - gt_lo)

    if direct_err <= swap_err:
        return t1 - gt_lo, t2 - gt_hi
    else:
        return t1 - gt_hi, t2 - gt_lo


# ---------------------------------------------------------------------------
# Per-pair record
# ---------------------------------------------------------------------------

def _analyse_pair(
    pred: StreakSegment, gt: StreakSegment, ann_id: int
) -> dict[str, Any]:
    """Build a per-pair error record."""
    se1, se2 = _signed_endpoint_errors(pred, gt)
    sym_err = (abs(se1) + abs(se2)) / 2.0
    signed_length_err = pred.length_px - gt.length_px
    return {
        "image_id": gt.image_id,
        "annotation_id": ann_id,
        "gt_length_px": round(gt.length_px, 2),
        "pred_length_px": round(pred.length_px, 2),
        "signed_length_err_px": round(signed_length_err, 3),
        "symmetric_endpoint_err_px": round(sym_err, 3),
        "endpoint1_signed_err_px": round(se1, 3),
        "endpoint2_signed_err_px": round(se2, 3),
        "angle_err_deg": round(_angle_error_deg(pred.angle_deg, gt.angle_deg), 4),
        "perp_offset_px": round(_perpendicular_offset(pred, gt), 3),
        "pred_endpoints": {"x1": pred.x1, "y1": pred.y1, "x2": pred.x2, "y2": pred.y2},
        "gt_endpoints": {"x1": gt.x1, "y1": gt.y1, "x2": gt.x2, "y2": gt.y2},
    }


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def _dist_stats(values: list[float], label: str) -> dict[str, float]:
    if not values:
        return {}
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "mean": round(float(a.mean()), 3),
        "std": round(float(a.std()), 3),
        "p25": round(float(np.percentile(a, 25)), 3),
        "median": round(float(np.median(a)), 3),
        "p75": round(float(np.percentile(a, 75)), 3),
        "p90": round(float(np.percentile(a, 90)), 3),
        "p95": round(float(np.percentile(a, 95)), 3),
        "p99": round(float(np.percentile(a, 99)), 3),
        "max": round(float(a.max()), 3),
    }


def _bias_summary(signed_values: list[float]) -> dict[str, Any]:
    """Signed aggregate: positive mean = model too long, negative = too short."""
    if not signed_values:
        return {}
    a = np.asarray(signed_values, dtype=np.float64)
    n_long = int((a > 0).sum())
    n_short = int((a < 0).sum())
    return {
        "mean_signed": round(float(a.mean()), 3),
        "median_signed": round(float(np.median(a)), 3),
        "std": round(float(a.std()), 3),
        "pct_too_long": round(100.0 * n_long / len(signed_values), 1),
        "pct_too_short": round(100.0 * n_short / len(signed_values), 1),
        "interpretation": (
            "model predicts too LONG on average" if a.mean() > 2.0
            else "model predicts too SHORT on average" if a.mean() < -2.0
            else "model length bias is approximately neutral (< 2 px)"
        ),
    }


def _band(length: float) -> str:
    if length < 400.0:
        return "short"
    if length < 1000.0:
        return "medium"
    return "long"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyse(
    predictions: list[dict],
    ground_truth_coco: Path,
    perp_threshold_px: float = 10.0,
    angle_tol_deg: float = 15.0,
    length_iou_min: float = 0.1,
) -> tuple[list[dict], dict]:
    """Run matching and return (per_pair_records, aggregate_stats).

    We need annotation IDs to support flagging, so we load GT directly here
    rather than via _load_gt_from_coco.
    """
    with ground_truth_coco.open() as f:
        coco = json.load(f)

    # Build image_id → file_name map
    id_to_file: dict[int, str] = {
        img["id"]: img.get("file_name", "")
        for img in coco.get("images", [])
    }

    # Build per-image GT list, preserving annotation_id
    gt_segments: list[StreakSegment] = []
    ann_id_by_seg_idx: list[int] = []
    gt_dicts: list[dict] = []

    for ann in coco.get("annotations", []):
        x1, y1, x2, y2 = annotation_to_endpoints(ann)
        length_px = math.hypot(x2 - x1, y2 - y1)
        seg = StreakSegment(
            x1=x1, y1=y1, x2=x2, y2=y2,
            confidence=1.0,
            image_id=ann["image_id"],
        )
        gt_segments.append(seg)
        ann_id_by_seg_idx.append(ann["id"])
        gt_dicts.append({
            "image_id": ann["image_id"],
            "annotation_id": ann["id"],
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "streak_length_px": length_px,
        })

    pred_segments = [detection_dict_to_segment(p) for p in predictions]

    # Group by image_id
    preds_by_img: dict[Any, list[tuple[int, StreakSegment]]] = defaultdict(list)
    gts_by_img: dict[Any, list[tuple[int, StreakSegment]]] = defaultdict(list)
    for i, seg in enumerate(pred_segments):
        preds_by_img[seg.image_id].append((i, seg))
    for i, seg in enumerate(gt_segments):
        gts_by_img[seg.image_id].append((i, seg))

    all_image_ids = set(preds_by_img) | set(gts_by_img)
    matched_pairs: list[tuple[StreakSegment, StreakSegment, int]] = []  # (pred, gt, ann_id)

    for img_id in all_image_ids:
        img_pred_entries = sorted(
            preds_by_img[img_id], key=lambda x: x[1].confidence, reverse=True
        )
        img_gt_entries = gts_by_img[img_id]

        img_preds = [seg for _, seg in img_pred_entries]
        img_gts = [seg for _, seg in img_gt_entries]
        img_gt_indices = [gi for gi, _ in img_gt_entries]

        pairs, _ = _greedy_match_segments(
            img_preds, img_gts,
            angle_tol_deg=angle_tol_deg,
            perp_tol_px=perp_threshold_px,
            length_iou_min=length_iou_min,
        )
        for pi, gi in pairs:
            gt_seg_global_idx = img_gt_indices[gi]
            ann_id = ann_id_by_seg_idx[gt_seg_global_idx]
            matched_pairs.append((img_preds[pi], img_gts[gi], ann_id))

    # Per-pair records
    per_pair: list[dict] = []
    for pred, gt, ann_id in matched_pairs:
        record = _analyse_pair(pred, gt, ann_id)
        record["file_name"] = id_to_file.get(int(gt.image_id), "")
        record["band"] = _band(gt.length_px)
        per_pair.append(record)

    # Aggregate stats
    sym_errs = [r["symmetric_endpoint_err_px"] for r in per_pair]
    length_errs = [r["signed_length_err_px"] for r in per_pair]
    ep1_errs = [r["endpoint1_signed_err_px"] for r in per_pair]
    ep2_errs = [r["endpoint2_signed_err_px"] for r in per_pair]
    # "endpoint overrun" = signed error when positive (beyond GT), underrun when negative
    all_ep_signed = ep1_errs + ep2_errs  # both endpoints pooled for bias

    bands = ("short", "medium", "long")
    per_band_stats: dict[str, Any] = {}
    for band in bands:
        band_pairs = [r for r in per_pair if r["band"] == band]
        if band_pairs:
            per_band_stats[band] = {
                "n": len(band_pairs),
                "symmetric_endpoint_err_px": _dist_stats(
                    [r["symmetric_endpoint_err_px"] for r in band_pairs], band
                ),
                "signed_length_err_px": _bias_summary(
                    [r["signed_length_err_px"] for r in band_pairs]
                ),
            }

    aggregate = {
        "n_matched_pairs": len(per_pair),
        "n_gt": len(gt_segments),
        "symmetric_endpoint_err_px": _dist_stats(sym_errs, "all"),
        "signed_length_err_px": _bias_summary(length_errs),
        "pooled_endpoint_signed_err_px": _bias_summary(all_ep_signed),
        "per_band": per_band_stats,
    }

    return per_pair, aggregate


# ---------------------------------------------------------------------------
# Patch annotations to pending
# ---------------------------------------------------------------------------

def patch_pending(ann_path: Path, annotation_ids: set[int]) -> int:
    """Add review_status=pending to the given annotation IDs.

    Writes a .bak backup before modifying. Returns the count modified.
    """
    backup = ann_path.with_suffix(ann_path.suffix + ".bak")
    shutil.copy2(ann_path, backup)
    print(f"  Backup written to {backup}")

    with ann_path.open() as f:
        coco = json.load(f)

    count = 0
    for ann in coco.get("annotations", []):
        if ann["id"] in annotation_ids:
            ann["review_status"] = "pending"
            count += 1

    ann_path.write_text(json.dumps(coco, indent=2))
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_predictions(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data["predictions"] if isinstance(data, dict) else data


def _print_section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _print_dist(label: str, stats: dict) -> None:
    if not stats:
        print(f"  {label}: (no data)")
        return
    print(f"  {label}:")
    print(f"    n={stats['n']}  mean={stats['mean']:+.1f}  std={stats['std']:.1f}  "
          f"median={stats['median']:+.1f}")
    print(f"    p25={stats['p25']:+.1f}  p75={stats['p75']:+.1f}  "
          f"p90={stats['p90']:+.1f}  p95={stats['p95']:+.1f}  "
          f"p99={stats['p99']:+.1f}  max={stats['max']:+.1f}")


def _print_bias(label: str, bias: dict) -> None:
    if not bias:
        print(f"  {label}: (no data)")
        return
    print(f"  {label}:")
    print(f"    mean={bias['mean_signed']:+.1f}  median={bias['median_signed']:+.1f}  "
          f"std={bias['std']:.1f}")
    print(f"    too long: {bias['pct_too_long']:.0f}%   "
          f"too short: {bias['pct_too_short']:.0f}%")
    print(f"    → {bias['interpretation']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path,
                        help="Flat predictions JSON (list or {predictions: [...]})")
    parser.add_argument("--annotations", required=True, type=Path,
                        help="COCO-style ground-truth annotation JSON")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write full per-pair detail + aggregate to this JSON path")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Print (and optionally flag) the N worst matched pairs")
    parser.add_argument("--patch-annotations", type=Path, default=None,
                        help="COCO JSON to patch with review_status=pending for top-N worst cases. "
                             "Defaults to --annotations if you just pass the flag without a value.")
    parser.add_argument("--perp-threshold-px", type=float, default=10.0)
    parser.add_argument("--angle-tol-deg", type=float, default=15.0)
    parser.add_argument("--length-iou-min", type=float, default=0.1)
    args = parser.parse_args()

    predictions = _load_predictions(args.predictions)
    print(f"Loaded {len(predictions)} predictions from {args.predictions}")

    per_pair, aggregate = analyse(
        predictions,
        args.annotations,
        perp_threshold_px=args.perp_threshold_px,
        angle_tol_deg=args.angle_tol_deg,
        length_iou_min=args.length_iou_min,
    )

    # ── Overall stats ─────────────────────────────────────────────────────
    _print_section("SYMMETRIC ENDPOINT ERROR  (absolute, lower is better)")
    _print_dist("all bands", aggregate["symmetric_endpoint_err_px"])
    for band, bstats in aggregate["per_band"].items():
        _print_dist(f"  {band} ({bstats['n']} pairs)",
                    bstats["symmetric_endpoint_err_px"])

    _print_section("SIGNED LENGTH BIAS  (pred_length − gt_length)")
    _print_bias("all bands", aggregate["signed_length_err_px"])
    for band, bstats in aggregate["per_band"].items():
        _print_bias(f"  {band}", bstats["signed_length_err_px"])

    _print_section("POOLED ENDPOINT SIGNED ERROR  (per-endpoint, both ends pooled)")
    _print_bias("all endpoints", aggregate["pooled_endpoint_signed_err_px"])

    # ── Top-N worst ───────────────────────────────────────────────────────
    worst = sorted(per_pair, key=lambda r: r["symmetric_endpoint_err_px"], reverse=True)
    top_n = worst[: args.top_n]

    _print_section(f"TOP-{args.top_n} WORST MATCHED PAIRS (by symmetric endpoint error)")
    fmt = "{:>4}  {:>6}  {:>6}  {:>8}  {:>8}  {:>8}  {:>7}  {}"
    print(fmt.format(
        "rank", "ann_id", "img_id", "sym_err", "len_err", "ang_err", "band", "file"
    ))
    print("  " + "─" * 100)
    for rank, r in enumerate(top_n, 1):
        fname = Path(r["file_name"]).name if r["file_name"] else "?"
        print(fmt.format(
            rank,
            r["annotation_id"],
            r["image_id"],
            f"{r['symmetric_endpoint_err_px']:+.1f}px",
            f"{r['signed_length_err_px']:+.1f}px",
            f"{r['angle_err_deg']:.3f}°",
            r["band"],
            fname,
        ))

    # ── Write output ──────────────────────────────────────────────────────
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "aggregate": aggregate,
            "top_n_worst": top_n,
            "per_pair": per_pair,
        }
        args.output.write_text(json.dumps(output_data, indent=2))
        print(f"\nFull per-pair report written to {args.output}")

    # ── Patch pending ─────────────────────────────────────────────────────
    patch_target = args.patch_annotations
    if patch_target is not None:
        worst_ids = {r["annotation_id"] for r in top_n}
        print(f"\nPatching {len(worst_ids)} annotations to review_status=pending in {patch_target}")
        n_patched = patch_pending(patch_target, worst_ids)
        print(f"  {n_patched} annotations updated.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
