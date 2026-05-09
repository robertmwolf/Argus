"""Evaluate a DINO checkpoint on an ARGUS COCO split.

This helper keeps checkpoint comparisons apples-to-apples by forcing the same
annotation file, image pipeline, and MMDetection CocoMetric setup for each run.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from training.train_dino import _patch_torch_load_weights_only

logger = logging.getLogger(__name__)


def _force_cpu_when_no_cuda() -> None:
    """Match local training behavior: avoid MPS for DINO deformable attention."""
    import torch

    if torch.cuda.is_available():
        return
    try:
        import mmengine.device as device_mod
        import mmengine.device.utils as device_utils

        device_utils.DEVICE = "cpu"
        device_mod.DEVICE = "cpu"  # type: ignore[attr-defined]
        if hasattr(device_mod, "get_device"):
            device_mod.get_device = lambda: "cpu"  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - defensive runtime patch
        logger.warning("Could not force MMEngine device to CPU: %s", exc)


def evaluate_checkpoint(
    config_path: Path,
    checkpoint_path: Path,
    split: str,
    work_dir: Path,
    output_path: Path | None,
) -> dict[str, Any]:
    """Run MMDetection test loop for one checkpoint and split.

    Args:
        config_path: MMDetection config path.
        checkpoint_path: Checkpoint to evaluate.
        split: Annotation split basename, e.g. ``"val"`` or ``"test"``.
        work_dir: Directory for MMEngine logs and outputs.
        output_path: Optional JSON metrics output path.

    Returns:
        Metric dictionary returned by MMEngine.
    """
    _patch_torch_load_weights_only()
    _force_cpu_when_no_cuda()

    from mmengine.config import Config
    from mmengine.runner import Runner
    import mmdet.models  # noqa: F401 - register DINO components

    ann_file = f"annotations/{split}.json"
    evaluator_ann_file = f"data/annotations/{split}.json"

    cfg = Config.fromfile(config_path)
    cfg.work_dir = str(work_dir)
    cfg.load_from = str(checkpoint_path)
    cfg.resume = False
    cfg.test_dataloader.dataset.ann_file = ann_file
    cfg.test_dataloader.dataset.data_root = "data/"
    cfg.test_dataloader.num_workers = 0
    cfg.test_dataloader.persistent_workers = False
    cfg.test_evaluator.ann_file = evaluator_ann_file
    cfg.test_evaluator.outfile_prefix = str(work_dir / f"{checkpoint_path.stem}_{split}")
    cfg.log_level = "INFO"

    work_dir.mkdir(parents=True, exist_ok=True)
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
            "split": split,
            "metrics": metrics,
        }
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("models/dino/streak_codino_swin_t.py"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    metrics = evaluate_checkpoint(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        split=args.split,
        work_dir=args.work_dir,
        output_path=args.output,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
