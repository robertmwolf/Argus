"""Pre-flight checklist before renting a GPU for Swin-L training.

Validates every precondition for cloud training.  Exits 0 only when all
checks pass.  Run this on the training machine after cloud_setup.sh, before
launching train_dino.py.

Usage::

    python scripts/prepare_cloud_training.py

All checks are self-contained and offline (no Space-Track credentials needed).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Ensure the repo root is on sys.path so mmengine can find training.transforms
# and other project modules when the script is run from any directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
WARN = "\033[33m!\033[0m"

_results: list[tuple[bool, str]] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    """Record and print one checklist item.

    Args:
        ok: Whether the check passed.
        label: Short description of what was checked.
        detail: Optional extra context printed on failure.

    Returns:
        ok (passed through for caller use).
    """
    icon = PASS if ok else FAIL
    print(f"  {icon}  {label}")
    if not ok and detail:
        print(f"       {detail}")
    _results.append((ok, label))
    return ok


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_cuda() -> bool:
    """CUDA must be available and at least one GPU present."""
    try:
        import torch
        ok = torch.cuda.is_available()
        if ok:
            name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            return check(True, f"CUDA available — {name} ({vram_gb:.1f} GB)")
        return check(False, "CUDA available", "torch.cuda.is_available() returned False")
    except ImportError:
        return check(False, "CUDA available", "PyTorch not installed")


def check_pytorch_version() -> bool:
    """PyTorch >= 2.6 is needed for Blackwell (RTX 50xx) support."""
    try:
        import torch
        major, minor = (int(x) for x in torch.__version__.split(".")[:2])
        ok = (major, minor) >= (2, 6)
        return check(ok, f"PyTorch >= 2.6  (found {torch.__version__})",
                     "RTX 50xx (Blackwell) requires PyTorch 2.6+. Run: pip install torch --upgrade")
    except Exception as exc:
        return check(False, "PyTorch version", str(exc))


def check_mmdet() -> bool:
    """MMDetection must be importable."""
    try:
        import mmdet
        return check(True, f"mmdet importable (version {mmdet.__version__})")
    except ImportError:
        return check(False, "mmdet importable",
                     "pip install mmdet mmengine mmcv")


def check_mmcv_ops() -> bool:
    """mmcv compiled ops must be present for Co-DINO CUDA training."""
    try:
        import mmcv.ops  # noqa: F401
        return check(True, "mmcv CUDA/C++ ops importable")
    except Exception as exc:
        return check(False, "mmcv CUDA/C++ ops importable",
                     f"{exc}. Reinstall mmcv from the platform-specific OpenMMLab wheel index.")


def check_dinov3_package() -> bool:
    """DINOv3 Python package must be importable for the ViT backbone adapter."""
    try:
        import dinov3.models.vision_transformer  # noqa: F401
        return check(True, "dinov3 package importable")
    except Exception as exc:
        return check(False, "dinov3 package importable",
                     f"{exc}. Run: pip install git+https://github.com/facebookresearch/dinov3.git")


def check_config_tiny() -> bool:
    """Swin-T config must parse cleanly via mmengine."""
    return _check_config("models/dino/streak_codino_swin_t.py", "Swin-T config parses")


def check_config_large() -> bool:
    """Swin-L config must parse cleanly via mmengine."""
    return _check_config("models/dino/streak_codino_swin_l.py", "Swin-L config parses")


def check_config_dinov3_vitb() -> bool:
    """DINOv3 ViT-B config must parse cleanly via mmengine."""
    return _check_config("models/dino/streak_dinov3_vitb.py", "DINOv3 ViT-B config parses")


def check_config_dinov3_vitl() -> bool:
    """DINOv3 ViT-L config must parse cleanly via mmengine."""
    return _check_config("models/dino/streak_dinov3_vitl.py", "DINOv3 ViT-L config parses")


def _check_config(config_path: str, label: str) -> bool:
    try:
        from mmengine.config import Config
        cfg = Config.fromfile(config_path)
        _ = cfg.model  # access to trigger any lazy-load errors
        return check(True, label)
    except Exception as exc:
        return check(False, label, str(exc))


def check_weights_large() -> bool:
    """Swin-L pretrain weights must exist in weights/."""
    path = Path("weights/co_dino_swin_l_coco.pth")
    ok = path.exists() and path.stat().st_size > 100_000_000
    return check(ok, f"Swin-L weights present ({path})",
                 f"Run: MODEL_SIZE=large python scripts/download_weights.py")


def check_weights_dinov3_vitl() -> bool:
    """DINOv3 ViT-L pretrain weights must exist in weights/."""
    path = Path("weights/dinov3_vitl16_lvd1689m.pth")
    ok = path.exists() and path.stat().st_size > 500_000_000
    return check(ok, f"DINOv3 ViT-L weights present ({path})",
                 "Copy from Mac: scp mac:~/Argus/weights/dinov3_vitl16_lvd1689m.pth weights/")


def check_annotations() -> bool:
    """Train and val annotation files must exist and have images + annotations."""
    all_ok = True
    for split in ("train", "val"):
        path = Path(f"data/annotations/{split}.json")
        if not path.exists():
            all_ok = check(False, f"data/annotations/{split}.json exists",
                           "Run scripts/merge_annotations.py to generate train/val splits") and all_ok
            continue
        try:
            with open(path) as f:
                coco = json.load(f)
            n_images = len(coco.get("images", []))
            n_anns = len(coco.get("annotations", []))
            ok = n_images >= 10
            label = f"data/annotations/{split}.json  ({n_images} images, {n_anns} annotations)"
            if ok and n_anns == 0:
                # Annotations are derived from masks at load time; 0 here is fine
                # when SatStreaks masks are present in data/satstreaks/Data/Masks/
                print(f"  {WARN}  {label} — 0 OBB annotations (masks read at load time)")
                _results.append((True, label))
            else:
                all_ok = check(ok, label, "File exists but has < 10 images — re-run merge_annotations.py") and all_ok
        except Exception as exc:
            all_ok = check(False, f"data/annotations/{split}.json readable", str(exc)) and all_ok
    return all_ok


def check_gtimages_converted() -> bool:
    """GTImages COCO JSON should exist; warn if missing (not fatal)."""
    path = Path("data/annotations/gtimages.json")
    if path.exists():
        try:
            with open(path) as f:
                coco = json.load(f)
            n = len(coco.get("images", []))
            return check(True, f"GTImages annotations present ({n} images)")
        except Exception:
            pass
    # Soft warning — GTImages can be converted now if the FITS are present
    fits_count = len(list(Path("data/GTImages").glob("*.fits"))) if Path("data/GTImages").exists() else 0
    if fits_count > 0:
        print(f"  {WARN}  GTImages not converted yet ({fits_count} FITS found) — run:")
        print("       python scripts/convert_gtimages.py \\")
        print("           --strk-dir data/GTImages \\")
        print("           --output data/annotations/gtimages.json")
        _results.append((True, "GTImages (soft warning — not converted yet)"))
    else:
        print(f"  {WARN}  data/GTImages/ missing or empty — download and unzip GTImages.zip")
        _results.append((True, "GTImages (soft warning — files not found)"))
    return True  # non-fatal


def check_dev_subset() -> bool:
    """dev_subset.json must exist (committed to repo)."""
    path = Path("data/annotations/dev_subset.json")
    ok = path.exists()
    return check(ok, "dev_subset.json present", "git checkout data/annotations/dev_subset.json")


def check_fast_pipeline() -> bool:
    """pipeline.run() in fast mode must complete on a synthetic FITS."""
    try:
        import tempfile
        from pathlib import Path as _Path
        sys.path.insert(0, str(_Path(__file__).parent.parent))
        from scripts.make_test_fits import make_test_fits  # type: ignore[import]
        from inference.pipeline import run as pipeline_run

        with tempfile.TemporaryDirectory() as tmp:
            fits_path = str(_Path(tmp) / "smoke.fits")
            make_test_fits(fits_path, with_streak=True)
            dets = pipeline_run(fits_path, fast=True)

        return check(True, f"Fast pipeline smoke test  ({len(dets)} detections returned)")
    except Exception as exc:
        return check(False, "Fast pipeline smoke test", str(exc))


def check_smoke_test_train() -> bool:
    """Training smoke test (2 epochs, 10 images) must exit cleanly.

    Skipped when SKIP_SMOKE_TRAIN=1 is set (takes ~5 min on GPU).
    Uses the active MODEL_SIZE backbone so the right config is exercised.
    """
    if os.environ.get("SKIP_SMOKE_TRAIN") == "1":
        print(f"  {WARN}  Training smoke test skipped (SKIP_SMOKE_TRAIN=1)")
        _results.append((True, "Training smoke test (skipped)"))
        return True

    model_size = os.environ.get("MODEL_SIZE", "large").lower()
    cmd = [
        sys.executable, "-m", "training.train_dino",
        "--backbone", model_size,
        "--smoke-test",
    ]
    env = {**os.environ, "MODEL_SIZE": model_size}
    label = f"Training smoke test --backbone {model_size} (2 epochs)"
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    ok = result.returncode == 0
    detail = (result.stderr or result.stdout)[-500:] if not ok else ""
    return check(ok, label, detail)


def check_requirements_pinned() -> bool:
    """requirements.txt must exist."""
    path = Path("requirements.txt")
    ok = path.exists() and path.stat().st_size > 100
    return check(ok, "requirements.txt present", "Missing requirements.txt")


def check_env_vars() -> bool:
    """Required env vars for training must be set."""
    all_ok = True
    model_size = os.environ.get("MODEL_SIZE", "").lower()
    valid_sizes = {"large", "dinov3_vitl", "dinov3_vitb", "tiny"}
    ms_ok = model_size in valid_sizes
    all_ok = check(ms_ok,
                   f"$MODEL_SIZE={model_size!r}  (valid: {sorted(valid_sizes)})",
                   "export MODEL_SIZE=dinov3_vitl") and all_ok

    val = os.environ.get("USE_DEV_SUBSET", "")
    ok = val.lower() == "false"
    all_ok = check(ok, f"$USE_DEV_SUBSET='false'  (found {val!r})",
                   "export USE_DEV_SUBSET=false") and all_ok
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run all checks and return exit code.

    Returns:
        0 if all checks pass, 1 otherwise.
    """
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║      ARGUS Cloud Training Pre-flight Checklist       ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    print("── Environment ──────────────────────────────────────────")
    check_cuda()
    check_pytorch_version()
    check_env_vars()

    model_size = os.environ.get("MODEL_SIZE", "large").lower()
    is_dinov3 = model_size.startswith("dinov3")

    print("\n── Dependencies ─────────────────────────────────────────")
    check_mmdet()
    check_mmcv_ops()
    if is_dinov3:
        check_dinov3_package()
    check_requirements_pinned()

    print("\n── Model configs ────────────────────────────────────────")
    check_config_tiny()
    if is_dinov3:
        check_config_dinov3_vitb()
        check_config_dinov3_vitl()
        check_weights_dinov3_vitl()
    else:
        check_config_large()
        check_weights_large()

    print("\n── Data ─────────────────────────────────────────────────")
    check_dev_subset()
    check_gtimages_converted()
    check_annotations()

    print("\n── Pipeline smoke tests ─────────────────────────────────")
    check_fast_pipeline()
    check_smoke_test_train()

    # Summary
    failures = [label for ok, label in _results if not ok]
    total = len(_results)
    passed = total - len(failures)

    print(f"\n{'─' * 56}")
    print(f"  {passed}/{total} checks passed")
    if failures:
        print("\n  Failed checks:")
        for label in failures:
            print(f"    {FAIL}  {label}")
        print("\n  Fix the failures above, then re-run this script.")
        return 1
    else:
        print(f"\n  {PASS}  All checks passed — ready for GPU training.")
        print("\n  Next step:")
        if is_dinov3:
            print("    python -m training.train_dino \\")
            print(f"        --backbone {model_size} \\")
            print(f"        --work-dir weights/run_5070ti_{model_size}\n")
        else:
            print("    MODEL_SIZE=large python -m training.train_dino \\")
            print("        --work-dir weights/run_001\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
