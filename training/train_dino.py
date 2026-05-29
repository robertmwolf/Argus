"""DINO training script for satellite streak detection.

Wraps MMDetection's Runner to add:
  - Hardware-aware device / config selection via MODEL_SIZE env var or --backbone
  - Swin backbones: two-stage fine-tuning (backbone frozen epochs 1-20, unfrozen 21-50)
  - DINOv3 backbones: backbone permanently frozen; only neck + DETR head train
  - Cost guardrails: after epoch 1 print estimated total cost, sleep 30 s (CUDA only)
  - --smoke-test mode: 2 epochs on 10 images, asserts loss decreasing, exits

Usage::

    # Local Mac dev (Swin-T, dev subset):
    MODEL_SIZE=tiny python -m training.train_dino

    # Local Mac dev (DINOv3 ViT-B, dev subset):
    MODEL_SIZE=dinov3_vitb python -m training.train_dino
    # equivalently:
    python -m training.train_dino --backbone dinov3_vitb

    # Workstation RTX 5070 Ti (DINOv3 ViT-L, full dataset):
    MODEL_SIZE=dinov3_vitl USE_DEV_SUBSET=false python -m training.train_dino

    # Cloud A100 (Swin-L, full dataset):
    MODEL_SIZE=large python -m training.train_dino --work-dir weights/run_001

    # Verify setup before full run:
    python -m training.train_dino --backbone dinov3_vitb --smoke-test

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
# PyTorch 2.6 compatibility — checkpoint loading patch
# ---------------------------------------------------------------------------
# PyTorch 2.6 changed torch.load to default weights_only=True. mmengine
# checkpoints embed numpy arrays and custom objects that require weights_only=False.
# We monkey-patch torch.load to restore the pre-2.6 behaviour; this is safe
# because we only load checkpoints from MMDetection's official OpenMMLab releases.
def _patch_torch_load_weights_only() -> None:
    """Patch mmengine's torch.load reference to use weights_only=False.

    PyTorch 2.6 defaulted weights_only=True.  mmengine checkpoints embed
    numpy arrays and custom objects; loading them requires weights_only=False.
    We patch the torch reference *inside* mmengine.runner.checkpoint so that
    its load_from_local() call opts out of the new default safely — the
    checkpoint files come from OpenMMLab's official release.
    """
    try:
        import inspect
        import torch
        import mmengine.runner.checkpoint as ckpt_mod

        sig = inspect.signature(torch.load)
        if "weights_only" not in sig.parameters:
            return  # torch < 2.0 — not applicable
        # Patch whenever default is not explicitly False (covers True and None)

        import functools
        _orig = torch.load

        @functools.wraps(_orig)
        def _patched(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig(*args, **kwargs)

        # Replace the torch.load reference used inside mmengine's checkpoint module
        ckpt_mod.torch.load = _patched  # type: ignore[attr-defined]
    except Exception:
        pass  # safe to skip — worst case the user sees the same error as before


_patch_torch_load_weights_only()

# ---------------------------------------------------------------------------
# Config selection
# ---------------------------------------------------------------------------

_CONFIG_MAP: dict[str, str] = {
    "tiny":                    "models/dino/streak_codino_swin_t.py",
    "large":                   "models/dino/streak_codino_swin_l.py",
    "dinov3_vitb":             "models/dino/streak_dinov3_vitb.py",
    "dinov3_vitl":             "models/dino/streak_dinov3_vitl.py",
}

# Backbones that are permanently frozen — Stage2UnfreezeHook is skipped.
_FROZEN_BACKBONES: frozenset[str] = frozenset({"dinov3_vitb", "dinov3_vitl"})

_STAGE2_EPOCH = 21   # epoch at which Swin backbone is unfrozen (not used for DINOv3)


def _select_config(model_size: str) -> str:
    """Return config path for the given MODEL_SIZE / backbone key.

    Args:
        model_size: One of 'tiny', 'large', 'dinov3_vitb', 'dinov3_vitl'.

    Returns:
        Path to the MMDetection config file.

    Raises:
        ValueError: If model_size is not recognised.
        EnvironmentError: If model_size requires CUDA and it is not available.
    """
    if model_size not in _CONFIG_MAP:
        raise ValueError(
            f"Unknown MODEL_SIZE={model_size!r}. "
            f"Choose from: {sorted(_CONFIG_MAP)}"
        )

    import torch
    if model_size == "large" and not torch.cuda.is_available():
        raise EnvironmentError(
            "MODEL_SIZE=large (Swin-L) requires CUDA.\n"
            "On Mac use MODEL_SIZE=tiny or MODEL_SIZE=dinov3_vitb."
        )
    if model_size == "dinov3_vitl" and not torch.cuda.is_available():
        raise EnvironmentError(
            "MODEL_SIZE=dinov3_vitl (ViT-L, 512px) requires CUDA (RTX 5070 Ti or A100).\n"
            "On Mac use MODEL_SIZE=dinov3_vitb (ViT-B, 256px)."
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

    # Force CPU for smoke test — DINO deformable attention peaks above the
    # 4 GB MPS NDArray limit even at small image sizes.  CPU is slower
    # (~5 min for 2 epochs) but exercises the full training codepath.
    import torch
    if not torch.cuda.is_available():
        # Patch mmengine's device detection to report CPU so the Runner
        # places the model on CPU instead of MPS.
        try:
            import mmengine.device.utils as _dev
            _dev.DEVICE = "cpu"
            import mmengine.device as _devmod
            _devmod.DEVICE = "cpu"  # type: ignore[attr-defined]
            if hasattr(_devmod, "get_device"):
                _devmod.get_device = lambda: "cpu"  # type: ignore[attr-defined]
        except Exception:
            pass
        cfg.train_dataloader.num_workers = 0
        cfg.val_dataloader.num_workers = 0

    losses: list[float] = []

    from mmengine.hooks import Hook

    class LossRecorderHook(Hook):
        priority = "LOW"

        def after_train_epoch(self, runner) -> None:  # type: ignore[override]
            # runner.message_hub stores the latest logged scalars
            loss = runner.message_hub.get_scalar("train/loss").current()
            losses.append(float(loss))
            logger.info("Smoke test epoch %d loss: %.4f", runner.epoch, loss)

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
    max_epochs: int | None = None,
    val_interval: int | None = None,
    checkpoint_interval: int | None = None,
    load_from: str | None = None,
) -> None:
    """Launch DINO training with two-stage fine-tuning and cost guardrails.

    Args:
        config_path: Path to the MMDetection config file.
        work_dir: Directory for checkpoints, logs, and val results.
        resume: If True, resume from the latest checkpoint in work_dir.
        smoke_test: If True, run 2-epoch sanity check and exit.
        max_epochs: Optional override for local timeboxed retraining.
        val_interval: Optional validation interval override, in epochs.
        checkpoint_interval: Optional checkpoint interval override, in epochs.
        load_from: Optional checkpoint to initialise training from.
    """
    # ---------------------------------------------------------------------------
    # Training guardrails — disable expensive inference-time stages
    # ---------------------------------------------------------------------------
    # Plate solving (ASTAP) and TLE cross-identification add 10–60 s per image
    # and are irrelevant to detection training.  Enforce their absence here so
    # dataloader workers (which inherit os.environ) cannot accidentally run them.
    if os.environ.get("ARGUS_ENABLE_PLATE_SOLVE", "").strip().lower() in (
        "1", "true", "yes", "on"
    ):
        logger.warning(
            "ARGUS_ENABLE_PLATE_SOLVE is set — ASTAP will run on every training "
            "image and will significantly slow training.  Unset this variable "
            "unless you intentionally need WCS during training."
        )
    else:
        os.environ["ARGUS_ENABLE_PLATE_SOLVE"] = "0"

    # Skip sidecar WCS lookup and plate solving in FITSLoader during training.
    # WCS is not needed for detection training; cross-ID is gated on WCS anyway.
    os.environ.setdefault("ARGUS_SKIP_WCS", "1")

    # Belt-and-suspenders: even if pipeline.run() were called during eval,
    # CROSSID_MAX_DETECTIONS=0 prevents any TLE catalog queries.
    os.environ.setdefault("CROSSID_MAX_DETECTIONS", "0")

    logger.info(
        "Training guardrails active: ARGUS_SKIP_WCS=1  CROSSID_MAX_DETECTIONS=0"
    )

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

    # DINO multi-scale deformable attention exceeds MPS's 4 GB per-allocation
    # limit.  Force CPU on Mac until a memory-efficient MPS path is available.
    import torch as _torch
    if not _torch.cuda.is_available():
        try:
            import mmengine.device.utils as _dev
            _dev.DEVICE = "cpu"
            import mmengine.device as _devmod
            _devmod.DEVICE = "cpu"  # type: ignore[attr-defined]
            if hasattr(_devmod, "get_device"):
                _devmod.get_device = lambda: "cpu"  # type: ignore[attr-defined]
        except Exception:
            pass

    cfg = Config.fromfile(config_path)
    cfg.work_dir = str(work_dir)
    cfg.resume = resume
    if max_epochs is not None:
        cfg.train_cfg.max_epochs = max_epochs
        logger.info("Overriding max_epochs=%d", max_epochs)
    if val_interval is not None:
        cfg.train_cfg.val_interval = val_interval
        logger.info("Overriding val_interval=%d", val_interval)
    if checkpoint_interval is not None:
        cfg.default_hooks.checkpoint.interval = checkpoint_interval
        logger.info("Overriding checkpoint_interval=%d", checkpoint_interval)
    if load_from:
        cfg.load_from = load_from
        logger.info("Initialising from checkpoint: %s", load_from)
    if os.environ.get("USE_DEV_SUBSET", "true").lower() in {"0", "false", "no"}:
        # TRAIN_ANN_FILE / VAL_ANN_FILE let callers substitute annotation files
        # (e.g. all_train_nodm.json) without changing any other config.  Paths
        # may be relative to data_root or absolute external-drive paths.
        train_ann = os.environ.get("TRAIN_ANN_FILE", "annotations/train.json")
        val_ann = os.environ.get("VAL_ANN_FILE", "annotations/val.json")
        cfg.train_dataloader.dataset.ann_file = train_ann
        cfg.val_dataloader.dataset.ann_file = val_ann
        cfg.test_dataloader = cfg.val_dataloader
        cfg.val_evaluator.ann_file = (
            val_ann if Path(val_ann).is_absolute() else f"data/{val_ann}"
        )
        cfg.test_evaluator = cfg.val_evaluator
        logger.info(
            "USE_DEV_SUBSET=false → train=%s  val=%s",
            train_ann if Path(train_ann).is_absolute() else f"data/{train_ann}",
            val_ann if Path(val_ann).is_absolute() else f"data/{val_ann}",
        )
    if not _torch.cuda.is_available():
        cfg.train_dataloader.num_workers = 0
        cfg.val_dataloader.num_workers = 0

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

    # Stage2UnfreezeHook: Swin backbones only.
    # DINOv3 backbones are permanently frozen in the config; the hook is skipped
    # to avoid misleading log messages about "unfreezing" a frozen backbone.
    is_frozen_backbone = any(tag in config_path for tag in ("dinov3",))
    if not is_frozen_backbone:
        stage2_hook = _make_stage2_hook(_STAGE2_EPOCH)
        if stage2_hook is not None:
            runner.register_hook(stage2_hook)
    else:
        logger.info("DINOv3 backbone: Stage2UnfreezeHook skipped (backbone permanently frozen)")

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
        "--backbone",
        default=None,
        choices=sorted(_CONFIG_MAP),
        help=(
            "Backbone / config preset. Equivalent to setting MODEL_SIZE env var. "
            "Options: tiny (Swin-T, Mac), large (Swin-L, A100), "
            "dinov3_vitb (frozen ViT-B, Mac), dinov3_vitl (frozen ViT-L, workstation)."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to MMDetection config file. "
            "Defaults to the config for the current MODEL_SIZE env var or --backbone."
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
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override config max_epochs, useful for local timeboxed retraining.",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=None,
        help="Override config validation interval in epochs.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Override config checkpoint interval in epochs.",
    )
    parser.add_argument(
        "--load-from",
        default=None,
        help="Initial checkpoint path for fine-tuning.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()

    # --backbone flag takes precedence over MODEL_SIZE env var
    model_size = (args.backbone or os.environ.get("MODEL_SIZE", "tiny")).lower()

    if args.config:
        config_path = args.config
        logger.info("Using explicit config: %s", config_path)
    else:
        try:
            config_path = _select_config(model_size)
        except (ValueError, EnvironmentError) as exc:
            logger.error("%s", exc)
            raise SystemExit(1) from exc
        logger.info("backbone=%s → %s", model_size, config_path)

    work_dir = Path(args.work_dir)

    try:
        train(
            config_path=config_path,
            work_dir=work_dir,
            resume=args.resume,
            smoke_test=args.smoke_test,
            max_epochs=args.max_epochs,
            val_interval=args.val_interval,
            checkpoint_interval=args.checkpoint_interval,
            load_from=args.load_from,
        )
    except RuntimeError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc
