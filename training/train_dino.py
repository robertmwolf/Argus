"""Co-DINO / DINO Swin training script for satellite streak detection.

Wraps MMDetection's Runner to add:
  - Hardware-aware device / config selection via MODEL_SIZE env var
  - Two-stage fine-tuning (backbone frozen epochs 1-20, unfrozen 21-50)
  - Cost guardrails: after epoch 1 print estimated total cost, sleep 30 s
  - --smoke-test mode: 2 epochs on 10 images, asserts loss decreasing, exits

Usage::

    # Local Mac dev (Swin-T, dev subset):
    MODEL_SIZE=tiny python -m training.train_dino

    # Cloud A100 (Swin-L, full dataset):
    MODEL_SIZE=large python -m training.train_dino --work-dir weights/run_001

    # Verify cloud setup before full run:
    MODEL_SIZE=large python -m training.train_dino --smoke-test

    # Custom config:
    python -m training.train_dino --config models/dino/streak_codino_swin_t.py
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config selection
# ---------------------------------------------------------------------------

_CONFIG_MAP: dict[str, str] = {
    "tiny":  "models/dino/streak_codino_swin_t.py",
    "large": "models/dino/streak_codino_swin_l.py",
}

_STAGE2_EPOCH = 21   # epoch at which backbone is unfrozen


def _select_config(model_size: str) -> str:
    """Return config path for the given MODEL_SIZE.

    Args:
        model_size: ``'tiny'`` or ``'large'``.

    Returns:
        Path to the MMDetection config file.

    Raises:
        ValueError: If model_size is not recognised.
        EnvironmentError: If model_size='large' and CUDA is not available.
    """
    if model_size not in _CONFIG_MAP:
        raise ValueError(
            f"Unknown MODEL_SIZE={model_size!r}. Choose 'tiny' or 'large'."
        )

    if model_size == "large":
        import torch
        if not torch.cuda.is_available():
            raise EnvironmentError(
                "MODEL_SIZE=large requires CUDA (Lambda Labs A100).\n"
                "On Mac, use MODEL_SIZE=tiny for development."
            )

    return _CONFIG_MAP[model_size]


# ---------------------------------------------------------------------------
# Two-stage backbone schedule hook
# ---------------------------------------------------------------------------

def _make_stage2_hook(stage2_epoch: int = _STAGE2_EPOCH):
    """Return an MMEngine hook that unfreezes the backbone at *stage2_epoch*.

    Source: StreakMind — two-stage fine-tuning strategy
    Stage 1 (epochs 1-20):  backbone lr_mult=0.0 (frozen)
    Stage 2 (epochs 21-50): backbone lr_mult=0.1 (fine-tune)
    """
    try:
        from mmengine.hooks import Hook

        class Stage2UnfreezeHook(Hook):
            """Unfreeze the backbone at the start of epoch *stage2_epoch*."""

            priority = "NORMAL"

            def before_train_epoch(self, runner) -> None:  # type: ignore[override]
                epoch = runner.epoch + 1   # 1-indexed
                if epoch != stage2_epoch:
                    return

                logger.info(
                    "Epoch %d — Stage 2: unfreezing backbone (lr_mult=0.0→0.1)",
                    epoch,
                )
                optim_wrapper = runner.optim_wrapper
                for group in optim_wrapper.optimizer.param_groups:
                    if group.get("_is_backbone", False):
                        old_lr = group["lr"]
                        # Restore backbone LR to 10% of base LR
                        base_lr = group.get("_base_lr", group["lr"])
                        group["lr"] = base_lr * 0.1
                        logger.info(
                            "Backbone param group LR: %.2e → %.2e",
                            old_lr, group["lr"],
                        )

        return Stage2UnfreezeHook()

    except ImportError:
        logger.warning("mmengine not available — Stage2UnfreezeHook skipped")
        return None


# ---------------------------------------------------------------------------
# Cost guardrails hook
# ---------------------------------------------------------------------------

def _make_cost_hook(max_epochs: int = 50, cost_per_hour: float = 1.29):
    """Return an MMEngine hook that prints cost estimate after epoch 1.

    Prints estimated total training time and Lambda Labs cost, then
    sleeps 30 seconds so the operator can abort before epoch 2 starts.
    """
    try:
        from mmengine.hooks import Hook

        class CostGuardrailHook(Hook):
            """Print cost estimate + sleep 30 s after the first epoch."""

            priority = "LOW"
            _epoch1_start: float = 0.0

            def before_train_epoch(self, runner) -> None:  # type: ignore[override]
                if runner.epoch == 0:
                    self._epoch1_start = time.time()

            def after_train_epoch(self, runner) -> None:  # type: ignore[override]
                if runner.epoch != 0:
                    return

                elapsed = time.time() - self._epoch1_start
                total_est = elapsed * max_epochs
                cost_est  = (total_est / 3600.0) * cost_per_hour

                mins, secs = divmod(int(elapsed), 60)
                total_h, total_m = divmod(int(total_est / 60), 60)

                print(
                    f"\n{'='*60}\n"
                    f"  Epoch 1/{max_epochs} complete in {mins}m {secs}s.\n"
                    f"  Estimated total training time : {total_h}h {total_m}m\n"
                    f"  Estimated cost at ${cost_per_hour:.2f}/hr (Lambda A100): "
                    f"${cost_est:.2f}\n"
                    f"\n"
                    f"  Press Ctrl+C within 30 seconds to abort if this looks wrong.\n"
                    f"{'='*60}\n"
                )
                time.sleep(30)

        return CostGuardrailHook()

    except ImportError:
        logger.warning("mmengine not available — CostGuardrailHook skipped")
        return None


# ---------------------------------------------------------------------------
# Smoke-test helper
# ---------------------------------------------------------------------------

def _run_smoke_test(cfg_path: str, work_dir: Path) -> None:
    """Run 2 epochs on 10 images and assert loss is decreasing.

    Modifies the config in-place (epochs=2, subset of first 10 images).
    Completes in <5 minutes on an A100.  Used to verify cloud setup before
    committing to a full run.

    Args:
        cfg_path: Path to the MMDetection config file.
        work_dir: Directory for checkpoints and logs.

    Raises:
        SystemExit(1): If loss does not decrease over the 2 epochs.
    """
    try:
        from mmengine.config import Config
        from mmengine.runner import Runner
        import mmdet.models  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(f"mmdet/mmengine not installed: {exc}") from exc

    logger.info("=== SMOKE TEST: 2 epochs, 10 images ===")

    cfg = Config.fromfile(cfg_path)

    # Limit to 2 epochs, 10 training images, val every epoch
    cfg.train_cfg.max_epochs = 2
    cfg.train_cfg.val_interval = 1
    cfg.train_dataloader.dataset["indices"] = list(range(10))
    cfg.work_dir = str(work_dir / "smoke_test")

    losses: list[float] = []

    from mmengine.hooks import Hook

    class LossRecorderHook(Hook):
        priority = "LOW"

        def after_train_epoch(self, runner) -> None:  # type: ignore[override]
            # runner.message_hub stores the latest logged scalars
            loss = runner.message_hub.get_scalar("train/loss").current()
            losses.append(float(loss))
            logger.info("Smoke test epoch %d loss: %.4f", runner.epoch, loss)

    cfg.custom_hooks = [
        dict(type="LossRecorderHook"),
    ]

    runner = Runner.from_cfg(cfg)
    runner.register_hook(LossRecorderHook())
    runner.train()

    if len(losses) >= 2 and losses[-1] >= losses[0]:
        logger.error(
            "Smoke test FAILED: loss did not decrease "
            "(epoch 1=%.4f, epoch 2=%.4f)",
            losses[0], losses[-1],
        )
        raise SystemExit(1)

    logger.info(
        "Smoke test PASSED: loss %.4f → %.4f", losses[0], losses[-1]
    )
    print("\n✓  Smoke test passed. Ready to start full training.")


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(
    config_path: str,
    work_dir: Path,
    resume: bool = False,
    smoke_test: bool = False,
) -> None:
    """Launch DINO training with two-stage fine-tuning and cost guardrails.

    Args:
        config_path: Path to the MMDetection config file.
        work_dir: Directory for checkpoints, logs, and val results.
        resume: If True, resume from the latest checkpoint in work_dir.
        smoke_test: If True, run 2-epoch sanity check and exit.
    """
    try:
        from mmengine.config import Config
        from mmengine.runner import Runner
        import mmdet.models  # noqa: F401 — trigger model registration
    except ImportError as exc:
        raise RuntimeError(
            f"mmdet/mmengine not installed: {exc}\n"
            "Run: pip install mmengine mmdet"
        ) from exc

    from inference.device import get_device

    device = get_device()
    logger.info("Device: %s", device)

    if smoke_test:
        _run_smoke_test(config_path, work_dir)
        return

    cfg = Config.fromfile(config_path)
    cfg.work_dir = str(work_dir)
    cfg.resume = resume

    # Log setup
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    cfg.log_level = "INFO"

    # Inject cost guardrail hook (cloud only — skip on Mac)
    max_epochs = cfg.train_cfg.max_epochs
    custom_hooks = list(cfg.get("custom_hooks", []))
    if device.type == "cuda":
        custom_hooks.append(dict(type="CostGuardrailHook",
                                  max_epochs=max_epochs))
    cfg.custom_hooks = custom_hooks

    logger.info(
        "Starting training: config=%s  work_dir=%s  epochs=%d  device=%s",
        config_path, work_dir, max_epochs, device,
    )

    runner = Runner.from_cfg(cfg)

    # Register two-stage hook manually
    stage2_hook = _make_stage2_hook(_STAGE2_EPOCH)
    if stage2_hook is not None:
        runner.register_hook(stage2_hook)

    # Register cost guardrail hook manually (if not in custom_hooks)
    if device.type == "cuda":
        cost_hook = _make_cost_hook(max_epochs)
        if cost_hook is not None:
            runner.register_hook(cost_hook)

    runner.train()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to MMDetection config file. "
            "Defaults to the config for the current MODEL_SIZE env var."
        ),
    )
    parser.add_argument(
        "--work-dir",
        default="weights/run",
        help="Directory for checkpoints and logs (default: weights/run)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint in --work-dir",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run 2 epochs on 10 images to verify setup, then exit. "
            "Completes in <5 minutes on an A100."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()

    model_size = os.environ.get("MODEL_SIZE", "tiny").lower()

    if args.config:
        config_path = args.config
        logger.info("Using explicit config: %s", config_path)
    else:
        try:
            config_path = _select_config(model_size)
        except (ValueError, EnvironmentError) as exc:
            logger.error("%s", exc)
            raise SystemExit(1) from exc
        logger.info("MODEL_SIZE=%s → %s", model_size, config_path)

    work_dir = Path(args.work_dir)

    try:
        train(
            config_path=config_path,
            work_dir=work_dir,
            resume=args.resume,
            smoke_test=args.smoke_test,
        )
    except RuntimeError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
