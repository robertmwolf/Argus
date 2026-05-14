"""Phase E: Head-to-head comparison — DINOv3 ViT-B vs Swin-T baseline.

Evaluates both models on the same annotation split using MMDetection's
CocoMetric, then renders a side-by-side Markdown table.

Split guidance:
  --split dev_subset  Phase C local comparison (both models trained on dev_subset)
  --split val         Phase D full comparison (both models trained on full dataset)

Usage::

    # Phase C local comparison (dev_subset models, evaluate on dev_subset):
    python scripts/phase_e_compare.py --split dev_subset

    # Phase D full comparison (full-dataset models, evaluate on val):
    python scripts/phase_e_compare.py --split val

    # Evaluate one model only:
    python scripts/phase_e_compare.py --model swin_t --split dev_subset
    python scripts/phase_e_compare.py --model dinov3_vitb --split dev_subset

    # Use a specific DINOv3 checkpoint (default: auto-detect best):
    python scripts/phase_e_compare.py --dinov3-checkpoint weights/dinov3_vitb_dev/best_coco_bbox_mAP_epoch_50.pth --split dev_subset

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
    # Full-dataset Swin-T baseline (SatStreaks + GTImages merged)
    # Known mAP@0.5 on test.json: 0.190 (MMDet CocoMetric)
    "swin_t": {
        "label": "Co-DINO Swin-T (full-dataset baseline)",
        "config": "models/dino/streak_codino_swin_t.py",
        "checkpoint": "weights/retrain_satstreaks_gtimages_20260507_1050/best_coco_bbox_mAP_epoch_4.pth",
    },
    # Phase C: frozen ViT-B, 50-image dev subset — local sanity check only
    "dinov3_vitb": {
        "label": "DINOv3 ViT-B frozen (Phase C, dev subset)",
        "config": "models/dino/streak_dinov3_vitb.py",
        "checkpoint": None,  # auto-detected from weights/dinov3_vitb_dev/
    },
    # Phase D: frozen ViT-L, full SatStreaks dataset — primary workstation run
    "dinov3_vitl": {
        "label": "DINOv3 ViT-L frozen (Phase D, full dataset)",
        "config": "models/dino/streak_dinov3_vitl.py",
        "checkpoint": None,  # auto-detected from weights/run_5070ti_dinov3_vitl/
    },
}

_DINOV3_CHECKPOINT_DIR = Path("weights/dinov3_vitb_dev")
_DINOV3_VITL_CHECKPOINT_DIR = Path("weights/run_5070ti_dinov3_vitl")


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
    elif model_key == "dinov3_vitl":
        ckpt_path = dinov3_checkpoint_override or _find_best_dinov3_checkpoint(_DINOV3_VITL_CHECKPOINT_DIR)
        if ckpt_path is None:
            logger.warning(
                "DINOv3 ViT-L checkpoint not found in %s — Phase D workstation run not yet complete. "
                "Copy results from workstation and run again.",
                _DINOV3_VITL_CHECKPOINT_DIR,
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
    swin  = results.get("swin_t")
    vitb  = results.get("dinov3_vitb")
    vitl  = results.get("dinov3_vitl")
    split = results.get("split", "val")

    lines = [
        "## Phase E — DINOv3 vs Swin-T Baseline",
        "",
        f"Evaluated on: `{split}` split  ",
        f"Date: {results.get('date', '—')}",
        "",
        f"| Metric | {_MODELS['swin_t']['label']} | {_MODELS['dinov3_vitb']['label']} | {_MODELS['dinov3_vitl']['label']} |",
        "|--------|------|------|------|",
    ]

    for key, label in _METRIC_LABELS:
        swin_val = swin.get(key) if swin else None
        vitb_val = vitb.get(key) if vitb else None
        vitl_val = vitl.get(key) if vitl else None
        lines.append(
            f"| {label} | {_fmt(swin_val)} | {_fmt(vitb_val)} | {_fmt(vitl_val)} |"
        )

    lines += ["", "**Δ vs Swin-T baseline:**", ""]
    lines += [
        f"| Metric | ViT-B Δ | ViT-L Δ |",
        "|--------|---------|---------|",
    ]
    for key, label in _METRIC_LABELS:
        swin_val = swin.get(key) if swin else None
        vitb_val = vitb.get(key) if vitb else None
        vitl_val = vitl.get(key) if vitl else None
        lines.append(f"| {label} | {_delta(vitb_val, swin_val)} | {_delta(vitl_val, swin_val)} |")

    if split == "dev_subset":
        context = (
            "Phase C dev-subset check: Swin-T and ViT-B both trained on 50-image dev_subset. "
            "Run `--split val` with full-dataset checkpoints for the Phase D meaningful comparison."
        )
    else:
        context = (
            "Phase D full comparison: Swin-T (retrain_satstreaks_gtimages) vs ViT-L (Phase D workstation run) "
            "both trained on full SatStreaks + GTImages merged dataset."
        )

    lines += [
        "",
        "### Notes",
        f"- {context}",
        "- Swin-T baseline known mAP@0.5 on test.json: 0.190 (MMDet CocoMetric).",
        "- Phase E gate: ViT-L mAP@0.5 > Swin-T mAP@0.5 → Phase D succeeded.",
        "  If within ±5 pp: acceptable. If > 5 pp below: consider Phase F partial unfreeze.",
        "- Phase 8 hard targets (test.json): ≥94% precision, ≥97% recall.",
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
        help="Which model to evaluate: swin_t | dinov3_vitb | dinov3_vitl | all (default: all)",
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
