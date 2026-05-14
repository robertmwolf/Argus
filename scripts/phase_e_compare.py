"""Phase E: Head-to-head comparison — DINOv3 ViT-B vs Swin-T baseline.

Evaluates both models on the same held-out val split using MMDetection's
CocoMetric, then renders a side-by-side Markdown table.

Usage::

    # Evaluate both models (runs whichever checkpoints are present):
    python scripts/phase_e_compare.py

    # Evaluate one model only:
    python scripts/phase_e_compare.py --model swin
    python scripts/phase_e_compare.py --model dinov3_vitb

    # Use a specific DINOv3 checkpoint (default: auto-detect best):
    python scripts/phase_e_compare.py --dinov3-checkpoint weights/dinov3_vitb_dev/best_coco_bbox_mAP_epoch_50.pth

    # Override the val split (default: data/annotations/val.json):
    python scripts/phase_e_compare.py --split dev_subset

Results are saved to results/phase_e/ as JSON and printed as Markdown.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

_MODELS = {
    "swin_t": {
        "label": "Co-DINO Swin-T (baseline)",
        "config": "models/dino/streak_codino_swin_t.py",
        "checkpoint": "weights/local_run/best_coco_bbox_mAP_epoch_50.pth",
    },
    "dinov3_vitb": {
        "label": "DINOv3 ViT-B frozen (Phase C)",
        "config": "models/dino/streak_dinov3_vitb.py",
        "checkpoint": None,  # auto-detected from weights/dinov3_vitb_dev/
    },
}

_DINOV3_CHECKPOINT_DIR = Path("weights/dinov3_vitb_dev")


def _find_best_dinov3_checkpoint(work_dir: Path) -> Path | None:
    """Return the best checkpoint from the DINOv3 ViT-B training run, or None."""
    candidates = sorted(work_dir.glob("**/best_coco_bbox_mAP_epoch_*.pth"))
    if candidates:
        # Pick the one with the highest epoch number as a tiebreaker
        def _epoch(p: Path) -> int:
            try:
                return int(p.stem.split("epoch_")[-1])
            except ValueError:
                return 0
        return max(candidates, key=_epoch)
    return None


# ---------------------------------------------------------------------------
# Single-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_key: str,
    split: str,
    work_dir: Path,
    dinov3_checkpoint_override: Path | None = None,
) -> dict | None:
    """Run MMDetection test loop for one model on the given split.

    Returns:
        Metrics dict, or None if checkpoint not found.
    """
    from scripts.evaluate_dino_checkpoint import evaluate_checkpoint

    spec = _MODELS[model_key]
    config_path = Path(spec["config"])

    if model_key == "dinov3_vitb":
        ckpt_path = dinov3_checkpoint_override or _find_best_dinov3_checkpoint(_DINOV3_CHECKPOINT_DIR)
        if ckpt_path is None:
            logger.warning(
                "DINOv3 ViT-B checkpoint not found in %s — Phase C may still be running. "
                "Run again after training completes.",
                _DINOV3_CHECKPOINT_DIR,
            )
            return None
    else:
        ckpt_path = Path(spec["checkpoint"])
        if not ckpt_path.exists():
            logger.warning("Checkpoint not found: %s — skipping", ckpt_path)
            return None

    logger.info("Evaluating %s on split=%s …", spec["label"], split)
    logger.info("  config:     %s", config_path)
    logger.info("  checkpoint: %s", ckpt_path)

    model_work_dir = work_dir / model_key
    output_path = work_dir / f"{model_key}_{split}_metrics.json"

    metrics = evaluate_checkpoint(
        config_path=config_path,
        checkpoint_path=ckpt_path,
        split=split,
        work_dir=model_work_dir,
        output_path=output_path,
    )
    return metrics


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

_METRIC_LABELS = [
    ("coco/bbox_mAP",    "mAP@0.5:0.95"),
    ("coco/bbox_mAP_50", "mAP@0.5"),
    ("coco/bbox_mAP_75", "mAP@0.75"),
    ("coco/bbox_mAP_s",  "mAP (small)"),
    ("coco/bbox_mAP_m",  "mAP (medium)"),
    ("coco/bbox_mAP_l",  "mAP (large)"),
]

_TARGETS = {
    "coco/bbox_mAP_50": "—",
    "coco/bbox_mAP_75": "—",
}


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def _delta(a: float | None, b: float | None) -> str:
    """Signed delta (a − b), formatted with + sign."""
    if a is None or b is None:
        return "—"
    d = a - b
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.3f}"


def render_markdown_table(results: dict[str, dict | None]) -> str:
    swin = results.get("swin_t")
    dino = results.get("dinov3_vitb")

    lines = [
        "## Phase E — DINOv3 ViT-B vs Swin-T Baseline",
        "",
        f"Evaluated on: `{results.get('split', 'val')}` split  ",
        f"Date: {results.get('date', '—')}",
        "",
        f"| Metric | {_MODELS['swin_t']['label']} | {_MODELS['dinov3_vitb']['label']} | Δ (DINOv3 − Swin) |",
        "|--------|------|------|------|",
    ]

    for key, label in _METRIC_LABELS:
        swin_val = swin.get(key) if swin else None
        dino_val = dino.get(key) if dino else None
        lines.append(
            f"| {label} | {_fmt(swin_val)} | {_fmt(dino_val)} | {_delta(dino_val, swin_val)} |"
        )

    lines += [
        "",
        "### Notes",
        "- Both models evaluated on the same held-out val split with MMDetection CocoMetric.",
        "- Swin-T: Co-DINO Swin-T fine-tuned 50 epochs on dev subset (50 images).",
        "- DINOv3 ViT-B: frozen backbone + DETR head only, 50 epochs on dev subset.",
        "- Phase D (ViT-L, full dataset) and Phase E full comparison pending workstation run.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        choices=list(_MODELS) + ["all"],
        default="all",
        help="Which model to evaluate (default: all)",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Annotation split name (default: val → data/annotations/val.json)",
    )
    parser.add_argument(
        "--dinov3-checkpoint",
        type=Path,
        default=None,
        help="Override DINOv3 ViT-B checkpoint path (default: auto-detect best in weights/dinov3_vitb_dev/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase_e"),
        help="Directory to save evaluation outputs (default: results/phase_e/)",
    )
    args = parser.parse_args()

    work_dir = args.output_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    models_to_run = list(_MODELS) if args.model == "all" else [args.model]

    results: dict = {
        "split": args.split,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    any_ran = False
    for model_key in models_to_run:
        metrics = evaluate_model(
            model_key=model_key,
            split=args.split,
            work_dir=work_dir,
            dinov3_checkpoint_override=args.dinov3_checkpoint,
        )
        results[model_key] = metrics
        if metrics is not None:
            any_ran = True

    # Save combined results
    combined_path = work_dir / f"phase_e_comparison_{args.split}.json"
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    logger.info("Combined results saved to %s", combined_path)

    # Print Markdown table
    table = render_markdown_table(results)
    print("\n" + table + "\n")

    if not any_ran:
        logger.error("No models could be evaluated — check checkpoint paths above.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
