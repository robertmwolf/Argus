"""Post-hoc threshold × stitch sweep on an existing predictions.json.

Filters predictions by confidence threshold and optionally applies collinear
stitching, then recomputes precision/recall/per-band metrics via eval/metrics.py.
No model inference is needed — runs in seconds from cached predictions.

Usage:
    python scripts/run_posthoc_threshold_analysis.py \\
        --predictions results/run9_convnext_s2/predictions.json \\
        --annotations data/annotations/test_atwood.json \\
        --output-dir  results/run9_convnext_s2/threshold_sweep \\
        --stitch

    # Custom thresholds:
    python scripts/run_posthoc_threshold_analysis.py ... --thresholds 0.7 0.8 0.9
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from eval.metrics import evaluate
from inference.tiled_pipeline import stitch_collinear_fragments
from scripts.evaluate_dinov3_heatmap import coco_ground_truth

_DEFAULT_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]

# Success gate used for exit-code and row highlighting
_GATE_PRECISION = 0.01   # >1%
_GATE_RECALL    = 0.60   # ≥60%


def _apply_stitch(dets: list[dict], max_gap: float = 400.0) -> list[dict]:
    if len(dets) <= 1:
        return dets
    stitch_in = []
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        stitch_in.append({**d, "bbox": [x1, y1, x2 - x1, y2 - y1],
                          "score": d["confidence"]})
    stitched = stitch_collinear_fragments(stitch_in, max_gap_px=max_gap)
    out = []
    for s in stitched:
        x, y, w, h = s["bbox"]
        out.append({**s,
                    "bbox": [x, y, x + w, y + h],
                    "confidence": s.get("confidence", s.get("score", 0.0))})
    return out


def run_sweep(
    predictions: list[dict],
    ground_truth: list[dict],
    thresholds: list[float],
    run_stitch: bool,
    stitch_max_gap: float = 400.0,
    iou_threshold: float = 0.1,
) -> list[dict]:
    """Return a list of result rows, one per (threshold, stitch) combination."""
    by_img: dict[int, list[dict]] = defaultdict(list)
    for p in predictions:
        by_img[int(p["image_id"])].append(p)

    rows = []
    for thresh in thresholds:
        for do_stitch in ([False, True] if run_stitch else [False]):
            filtered: list[dict] = []
            for dets in by_img.values():
                img_dets = [d for d in dets if float(d.get("confidence", 0)) >= thresh]
                if do_stitch and img_dets:
                    img_dets = _apply_stitch(img_dets, stitch_max_gap)
                filtered.extend(img_dets)

            m = evaluate(filtered, ground_truth, iou_threshold=iou_threshold)
            pb = m.get("per_band", {})
            rows.append({
                "threshold": thresh,
                "stitch": do_stitch,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "map_50": m.get("map_50", 0.0),
                "short_recall": pb.get("short",  {}).get("recall", 0.0),
                "medium_recall": pb.get("medium", {}).get("recall", 0.0),
                "long_recall": pb.get("long",   {}).get("recall", 0.0),
                "n_predictions": len(filtered),
                "meets_gate": (
                    m["precision"] >= _GATE_PRECISION and
                    m["recall"]    >= _GATE_RECALL
                ),
            })
    return rows


def _print_table(rows: list[dict], n_images: int) -> None:
    hdr = (
        f"{'thresh':>6}  {'stitch':>6}  {'precision':>10}  {'recall':>7}  "
        f"{'F1':>5}  {'short_R':>7}  {'med_R':>6}  {'long_R':>7}  {'p/img':>6}"
    )
    sep = "─" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)
    for r in rows:
        p_img = r["n_predictions"] / max(n_images, 1)
        gate  = " ◀ GATE" if r["meets_gate"] else ""
        print(
            f"{r['threshold']:>6.2f}  {'yes' if r['stitch'] else 'no':>6}  "
            f"{r['precision']*100:>9.2f}%  "
            f"{r['recall']*100:>6.1f}%  "
            f"{r['f1']*100:>4.1f}%  "
            f"{r['short_recall']*100:>6.1f}%  "
            f"{r['medium_recall']*100:>6.1f}%  "
            f"{r['long_recall']*100:>6.1f}%  "
            f"{p_img:>6.1f}"
            f"{gate}"
        )
    print(sep)
    gate_rows = [r for r in rows if r["meets_gate"]]
    print(f"Success gate (prec>{_GATE_PRECISION*100:.0f}% AND recall≥{_GATE_RECALL*100:.0f}%): "
          f"{len(gate_rows)}/{len(rows)} rows pass")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True,
                    help="Path to predictions.json produced by evaluate_dinov3_heatmap.py")
    ap.add_argument("--annotations", required=True,
                    help="COCO annotation file (GT, e.g. test_atwood.json)")
    ap.add_argument("--output-dir", required=True,
                    help="Directory for threshold_sweep.json output")
    ap.add_argument("--stitch", action="store_true",
                    help="Run both no-stitch and stitch variants (default: no-stitch only)")
    ap.add_argument("--stitch-max-gap", type=float, default=400.0,
                    help="Max gap in px for collinear stitching (default: 400)")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=_DEFAULT_THRESHOLDS,
                    help="Confidence thresholds to sweep (default: 0.50 0.60 … 0.95)")
    ap.add_argument("--iou-threshold", type=float, default=0.1,
                    help="IoU threshold for TP matching (default: 0.10 — appropriate for "
                         "thin-streak detections where min pred width ~55px vs GT ~16px)")
    args = ap.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        print(f"ERROR: {pred_path} not found", file=sys.stderr)
        return 1

    predictions = json.loads(pred_path.read_text())
    ground_truth = coco_ground_truth(Path(args.annotations))

    ann_data = json.loads(Path(args.annotations).read_text())
    n_images = len(ann_data.get("images", []))

    print(f"\nPost-hoc threshold sweep: {pred_path.name}")
    print(f"  {len(predictions)} predictions  |  {len(ground_truth)} GT annotations  "
          f"|  {n_images} images  |  IoU>={args.iou_threshold:.2f}")
    if args.stitch:
        print(f"  Stitch variants: no-stitch + stitch (max_gap={args.stitch_max_gap:.0f}px)")
    print()

    rows = run_sweep(
        predictions, ground_truth,
        thresholds=sorted(args.thresholds),
        run_stitch=args.stitch,
        stitch_max_gap=args.stitch_max_gap,
        iou_threshold=args.iou_threshold,
    )

    _print_table(rows, n_images)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "threshold_sweep.json"
    out_path.write_text(json.dumps({
        "predictions_file": str(pred_path),
        "annotations_file": args.annotations,
        "n_predictions_total": len(predictions),
        "n_gt": len(ground_truth),
        "n_images": n_images,
        "gate_precision": _GATE_PRECISION,
        "gate_recall": _GATE_RECALL,
        "iou_threshold": args.iou_threshold,
        "rows": rows,
    }, indent=2))
    print(f"\nResults saved → {out_path}")

    gate_met = any(r["meets_gate"] for r in rows)
    return 0 if gate_met else 1


if __name__ == "__main__":
    sys.exit(main())
