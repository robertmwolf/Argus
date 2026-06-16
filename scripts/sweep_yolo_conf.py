#!/usr/bin/env python3
"""Offline confidence sweep for a YOLO OBB predictions file.

Runs inference ONCE at a low conf floor (e.g. 0.05) to capture all candidate
boxes, then re-thresholds offline at each level and re-scores with
eval.geometry_metrics — no extra model passes.  Prints a recall/precision/F1
table per threshold and writes a sweep.json.

Usage:
    python scripts/sweep_yolo_conf.py \
        --predictions results/<run>/sweep/predictions_conf05.json \
        --annotations data/annotations/val_balanced_v1.json \
        --output      results/<run>/sweep/sweep.json \
        --thresholds 0.05 0.10 0.15 0.20 0.25 0.30 0.40 0.50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from eval.geometry_metrics import evaluate_geometry, _load_ground_truth


def _f1(p: float, r: float) -> float:
    return round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="Low-floor predictions JSON")
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
    args = ap.parse_args()

    raw = json.loads(Path(args.predictions).read_text())
    # Normalize: matcher sorts by "confidence"; our inference writes "score".
    for p in raw:
        p["confidence"] = float(p.get("score", p.get("confidence", 0.0)))
        obb = p["obb"]
        p["streak_length_px"] = max(float(obb["w"]), float(obb["h"]))

    gt = _load_ground_truth(Path(args.annotations))

    rows = []
    for t in sorted(args.thresholds):
        kept = [p for p in raw if p["confidence"] >= t]
        res = evaluate_geometry(kept, gt)
        t1 = res["tier1_detection"]
        pb = t1["per_band"]
        ang = res["tier2_raw_geometry"]["angle_err_deg"]["mean"]
        rows.append({
            "conf": t,
            "recall": t1["detection_recall"],
            "precision": t1["detection_precision"],
            "f1": _f1(t1["detection_precision"], t1["detection_recall"]),
            "short": pb["short"]["recall"],
            "medium": pb["medium"]["recall"],
            "long": pb["long"]["recall"],
            "angle_deg": ang,
            "n_pred": t1["n_found"] + t1["n_false_positives"],
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(rows, indent=2))

    # Pretty table
    print(f"\nConfidence sweep — {Path(args.predictions).parent}")
    print(f"{'conf':>5} {'recall':>7} {'prec':>6} {'F1':>6} {'short':>6} {'med':>6} {'long':>6} {'ang°':>6} {'npred':>6}")
    print("-" * 62)
    best_f1 = max(rows, key=lambda r: r["f1"])
    for r in rows:
        star = "  <- best F1" if r is best_f1 else ""
        print(f"{r['conf']:>5.2f} {r['recall']:>7.3f} {r['precision']:>6.3f} {r['f1']:>6.3f} "
              f"{r['short']:>6.3f} {r['medium']:>6.3f} {r['long']:>6.3f} {r['angle_deg']:>6.2f} {r['n_pred']:>6d}{star}")
    print(f"\nbest F1 = {best_f1['f1']:.3f} at conf={best_f1['conf']:.2f} "
          f"(R={best_f1['recall']:.3f}, P={best_f1['precision']:.3f})")


if __name__ == "__main__":
    main()
