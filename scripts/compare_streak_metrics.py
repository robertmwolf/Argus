"""Side-by-side comparison of OBB IoU and line-segment streak metrics.

Loads predictions and ground truth from JSON files, then evaluates the same
prediction set under two matching criteria:

  * OBB IoU (existing ``eval.metrics.evaluate``)
  * Segment-match (new ``eval.streak_metrics.evaluate_segments``)

Prints a formatted comparison table for each requested confidence threshold.

Usage::

    python scripts/compare_streak_metrics.py \\
        --predictions results/run15_vits/t0.05_nostitch/predictions.json \\
        --annotations data/annotations/val_run12_1800_npy.json \\
        --thresholds 0.3 0.5 0.7 0.85 0.9

Optional flags::

    --angle-tol    Angle tolerance in degrees for segment match (default 5)
    --perp-tol     Perpendicular offset tolerance in pixels (default 5)
    --len-iou-min  Minimum 1-D IoU for segment match (default 0.5)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Allow running from the repo root without installing the package
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.metrics import evaluate as obb_evaluate  # noqa: E402
from eval.streak_metrics import (  # noqa: E402
    _load_gt_from_coco,
    _load_predictions_json,
    evaluate_segments,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_data(
    predictions_path: Path,
    annotations_path: Path,
) -> tuple[list[dict], list[dict]]:
    """Load predictions and ground-truth from paths.

    Args:
        predictions_path: Path to flat predictions JSON list.
        annotations_path: Path to COCO-style annotations JSON.

    Returns:
        Tuple of (all_predictions, ground_truth).
    """
    all_preds = _load_predictions_json(predictions_path)
    gts = _load_gt_from_coco(annotations_path)
    return all_preds, gts


def _fmt_row(label: str, metrics: dict) -> str:
    """Format a single metric row as a compact string.

    Args:
        label: Label prefix for the row.
        metrics: Metric dict from evaluate() or evaluate_segments().

    Returns:
        Formatted multi-line string.
    """
    pb = metrics.get("per_band", {})
    short_r  = pb.get("short",  {}).get("recall", 0.0)
    medium_r = pb.get("medium", {}).get("recall", 0.0)
    long_r   = pb.get("long",   {}).get("recall", 0.0)

    lines = [
        f"  {label}",
        (
            f"    recall={metrics.get('recall', 0):.4f}  "
            f"precision={metrics.get('precision', 0):.4f}  "
            f"f1={metrics.get('f1', 0):.4f}"
        ),
        (
            f"    short_recall={short_r:.4f}  "
            f"medium_recall={medium_r:.4f}  "
            f"long_recall={long_r:.4f}"
        ),
    ]

    # Append segment-match-specific diagnostics when present
    if "mean_perp_offset_px" in metrics:
        lines.append(
            f"    mean_angle_error={metrics.get('mean_angle_error_deg', 0):.2f}°  "
            f"mean_perp_offset={metrics.get('mean_perp_offset_px', 0):.2f}px  "
            f"mean_length_1d_iou={metrics.get('mean_length_1d_iou', 0):.4f}"
        )

    return "\n".join(lines)


def _delta_marker(seg_val: float, obb_val: float) -> str:
    """Return a delta string comparing segment vs OBB metric values.

    Args:
        seg_val: Value from segment-match evaluation.
        obb_val: Value from OBB IoU evaluation.

    Returns:
        Formatted delta string, e.g. '+0.1234' or '-0.0056'.
    """
    delta = seg_val - obb_val
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.4f}"


def run_comparison(
    all_preds: list[dict],
    gts: list[dict],
    thresholds: list[float],
    angle_tol_deg: float,
    perp_tol_px: float,
    length_iou_min: float,
    label: str = "",
) -> list[dict]:
    """Run OBB and segment evaluations at each threshold and print results.

    Args:
        all_preds: All prediction dicts (unfiltered).
        gts: Ground-truth annotation dicts.
        thresholds: Confidence thresholds to evaluate at.
        angle_tol_deg: Angle tolerance for segment match (degrees).
        perp_tol_px: Perpendicular offset tolerance (pixels).
        length_iou_min: Minimum 1-D IoU for segment match.
        label: Optional label prefix for the output header.

    Returns:
        List of comparison row dicts (one per threshold) for serialisation.
    """
    rows: list[dict] = []
    header = f"{'=' * 70}"
    if label:
        print(f"\n{header}")
        print(f"  {label}")
    print(header)
    print(
        f"  Segment-match parameters: angle<{angle_tol_deg}°  "
        f"perp<{perp_tol_px}px  len_iou>{length_iou_min}"
    )
    print(header)

    for t in sorted(thresholds):
        preds = [p for p in all_preds if float(p.get("confidence", 0.0)) >= t]
        n_pred = len(preds)

        print(f"\n--- threshold = {t:.2f}  (n_preds={n_pred}  n_gt={len(gts)}) ---")

        # OBB IoU
        obb_result = obb_evaluate(preds, gts, iou_threshold=0.5)
        print(f"\n=== OBB IoU (iou_threshold=0.50) ===")
        print(_fmt_row("", obb_result))

        # Segment match
        seg_result = evaluate_segments(
            preds, gts,
            angle_tol_deg=angle_tol_deg,
            perp_tol_px=perp_tol_px,
            length_iou_min=length_iou_min,
        )
        print(
            f"\n=== Streak-Match "
            f"(angle<{angle_tol_deg}°, perp<{perp_tol_px}px, "
            f"len_iou>{length_iou_min}) ==="
        )
        print(_fmt_row("", seg_result))

        # Delta summary
        recall_delta  = _delta_marker(seg_result["recall"],    obb_result["recall"])
        f1_delta      = _delta_marker(seg_result["f1"],        obb_result["f1"])
        prec_delta    = _delta_marker(seg_result["precision"], obb_result["precision"])
        pb_obb = obb_result.get("per_band", {})
        pb_seg = seg_result.get("per_band", {})
        short_delta  = _delta_marker(
            pb_seg.get("short", {}).get("recall", 0),
            pb_obb.get("short", {}).get("recall", 0),
        )
        medium_delta = _delta_marker(
            pb_seg.get("medium", {}).get("recall", 0),
            pb_obb.get("medium", {}).get("recall", 0),
        )
        long_delta   = _delta_marker(
            pb_seg.get("long", {}).get("recall", 0),
            pb_obb.get("long", {}).get("recall", 0),
        )
        print(
            f"\n  [DELTA seg - obb]  "
            f"recall={recall_delta}  precision={prec_delta}  f1={f1_delta}  "
            f"short_recall={short_delta}  medium_recall={medium_delta}  "
            f"long_recall={long_delta}"
        )

        rows.append({
            "threshold": t,
            "n_preds": n_pred,
            "n_gt": len(gts),
            "obb": obb_result,
            "segment": seg_result,
        })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the comparison script."""
    parser = argparse.ArgumentParser(
        description="Compare OBB IoU vs segment-match metrics side by side."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="Path to flat predictions JSON (list of {image_id, confidence, obb, …}).",
    )
    parser.add_argument(
        "--annotations",
        required=True,
        type=Path,
        help="Path to COCO-style annotations JSON.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.3, 0.5, 0.7, 0.85, 0.9],
        help="Confidence thresholds to evaluate at.",
    )
    parser.add_argument(
        "--angle-tol",
        type=float,
        default=5.0,
        dest="angle_tol_deg",
        help="Angle tolerance in degrees for segment match (default 5).",
    )
    parser.add_argument(
        "--perp-tol",
        type=float,
        default=5.0,
        dest="perp_tol_px",
        help="Perpendicular offset tolerance in pixels (default 5).",
    )
    parser.add_argument(
        "--len-iou-min",
        type=float,
        default=0.5,
        dest="length_iou_min",
        help="Minimum 1-D IoU for a segment match (default 0.5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save JSON summary.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Resolve relative paths from repo root
    repo_root = Path(__file__).resolve().parent.parent
    pred_path = args.predictions if args.predictions.is_absolute() else repo_root / args.predictions
    ann_path  = args.annotations  if args.annotations.is_absolute()  else repo_root / args.annotations

    if not pred_path.exists():
        logger.error("Predictions file not found: %s", pred_path)
        sys.exit(1)
    if not ann_path.exists():
        logger.error("Annotations file not found: %s", ann_path)
        sys.exit(1)

    print(f"\nLoading predictions : {pred_path}")
    print(f"Loading annotations : {ann_path}")
    all_preds, gts = _load_data(pred_path, ann_path)
    print(f"Total predictions   : {len(all_preds)}")
    print(f"Total GT streaks    : {len(gts)}")

    all_rows: dict[str, list[dict]] = {}

    # Primary comparison (nostitch predictions)
    rows = run_comparison(
        all_preds, gts,
        thresholds=args.thresholds,
        angle_tol_deg=args.angle_tol_deg,
        perp_tol_px=args.perp_tol_px,
        length_iou_min=args.length_iou_min,
        label=f"Source: {pred_path.parent.name}",
    )
    all_rows[str(pred_path)] = rows

    # Also compare stitch results if they exist
    stitch_sweep = pred_path.parent.parent / "threshold_sweep_stitchfix" / "threshold_sweep.json"
    stitch_preds_path = pred_path.parent.parent / "t0.05_nostitch" / "predictions.json"
    # The stitchfix sweep uses the same predictions.json but applies stitch logic —
    # since we don't have stitch post-processing here, we note its metrics directly
    # from the existing threshold_sweep_stitchfix/threshold_sweep.json when available.
    if stitch_sweep.exists():
        print(f"\n{'=' * 70}")
        print(f"  Also found: {stitch_sweep}")
        print(f"  (Pre-computed OBB IoU rows from threshold_sweep_stitchfix)")
        print(f"{'=' * 70}")
        with stitch_sweep.open() as f:
            sweep_data = json.load(f)
        stitch_rows = [r for r in sweep_data.get("rows", []) if r.get("stitch")]
        for row in stitch_rows:
            t = row.get("threshold")
            if t not in [r["threshold"] for r in rows]:
                continue
            seg_rows_at_t = [r for r in rows if r["threshold"] == t]
            seg_at_t = seg_rows_at_t[0]["segment"] if seg_rows_at_t else {}
            print(
                f"\n  threshold={t:.2f}  [stitch OBB IoU (pre-computed)]  "
                f"recall={row.get('recall', 0):.4f}  "
                f"precision={row.get('precision', 0):.4f}  "
                f"f1={row.get('f1', 0):.4f}  "
                f"short={row.get('short_recall', 0):.4f}  "
                f"medium={row.get('medium_recall', 0):.4f}  "
                f"long={row.get('long_recall', 0):.4f}"
            )
            if seg_at_t:
                pb_seg = seg_at_t.get("per_band", {})
                print(
                    f"  threshold={t:.2f}  [nostitch segment-match]          "
                    f"recall={seg_at_t.get('recall', 0):.4f}  "
                    f"precision={seg_at_t.get('precision', 0):.4f}  "
                    f"f1={seg_at_t.get('f1', 0):.4f}  "
                    f"short={pb_seg.get('short', {}).get('recall', 0):.4f}  "
                    f"medium={pb_seg.get('medium', {}).get('recall', 0):.4f}  "
                    f"long={pb_seg.get('long', {}).get('recall', 0):.4f}"
                )

    # Save JSON summary
    if args.output:
        out_path = args.output if args.output.is_absolute() else repo_root / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(all_rows, f, indent=2)
        print(f"\nJSON summary saved to: {out_path}")
    else:
        # Auto-save alongside the predictions file
        auto_out = pred_path.parent.parent / "streak_match_comparison.txt"
        # (text output captured externally by the caller — no auto-write here)

    print(f"\n{'=' * 70}")
    print("Comparison complete.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
