"""Compare no-OBB heatmap line detections against OBB detector predictions.

The comparison uses native line-segment metrics for both methods: heatmap
predictions use their ``line_segment`` output, while OBB predictions are
converted to their centerline.  This avoids rewarding or punishing the heatmap
method for box geometry it does not actually produce.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.line_metrics import evaluate_line_segments, load_line_ground_truth


def _load_predictions(path: str | Path, method: str | None = None) -> list[dict[str, Any]]:
    """Load flat predictions, method-prediction JSON, or proposal-script JSON."""
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "images" in payload:
        predictions: list[dict[str, Any]] = []
        for image in payload.get("images", []):
            image_id = image.get("file_name") or image.get("image_id")
            for segment in image.get("segments", []):
                native_start = segment.get("native_start") or {}
                native_end = segment.get("native_end") or {}
                predictions.append(
                    {
                        "image_id": image_id,
                        "confidence": float(segment.get("score", 0.0)),
                        "method": method or "dinov3_heatmap_centerline",
                        "line_segment": {
                            "x1": float(native_start.get("x", 0.0)),
                            "y1": float(native_start.get("y", 0.0)),
                            "x2": float(native_end.get("x", 0.0)),
                            "y2": float(native_end.get("y", 0.0)),
                            "angle_deg": float(segment.get("radon_angle_deg", 0.0)),
                            "length_px": float(
                                (
                                    (float(native_end.get("x", 0.0)) - float(native_start.get("x", 0.0))) ** 2
                                    + (float(native_end.get("y", 0.0)) - float(native_start.get("y", 0.0))) ** 2
                                )
                                ** 0.5
                            ),
                        },
                    }
                )
        return predictions
    if isinstance(payload, dict):
        if method:
            return list(payload.get(method, []))
        if "methods" in payload:
            raise ValueError("Benchmark result JSON does not contain raw predictions.")
        raise ValueError("For dict prediction JSON, pass --method or use proposal JSON with an images key.")
    raise ValueError(f"Unsupported prediction JSON format: {path}")


def main() -> int:
    """Run line-native comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--heatmap-predictions", required=True)
    parser.add_argument("--obb-predictions", default=None)
    parser.add_argument("--obb-method", default=None)
    parser.add_argument("--output", default="results/heatmap_vs_obb_line_comparison.json")
    parser.add_argument("--tolerance-px", type=float, default=6.0)
    parser.add_argument("--coverage-threshold", type=float, default=0.10)
    args = parser.parse_args()

    ground_truth = load_line_ground_truth(args.annotations)
    heatmap_predictions = _load_predictions(args.heatmap_predictions, method="dinov3_heatmap_centerline")
    results: dict[str, Any] = {
        "annotations": str(args.annotations),
        "tolerance_px": args.tolerance_px,
        "coverage_threshold": args.coverage_threshold,
        "methods": {
            "dinov3_heatmap_centerline": {
                **evaluate_line_segments(
                    heatmap_predictions,
                    ground_truth,
                    tolerance_px=args.tolerance_px,
                    coverage_threshold=args.coverage_threshold,
                ),
                "n_predictions_input": len(heatmap_predictions),
            }
        },
    }

    if args.obb_predictions:
        method = args.obb_method or "obb_detector"
        obb_predictions = _load_predictions(args.obb_predictions, method=args.obb_method)
        results["methods"][method] = {
            **evaluate_line_segments(
                obb_predictions,
                ground_truth,
                tolerance_px=args.tolerance_px,
                coverage_threshold=args.coverage_threshold,
            ),
            "n_predictions_input": len(obb_predictions),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results["methods"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
