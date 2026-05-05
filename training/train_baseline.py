"""YOLO11-OBB baseline training script for satellite streak detection.

Trains Ultralytics YOLO11-OBB on the synthetic dev subset.  Used as the
comparison baseline against the Co-DINO model in eval/benchmark.py.

Results are written to weights/yolo_baseline/.

Usage::

    # Train on dev subset (default, ~30 min on M3 MPS):
    python -m training.train_baseline

    # Smoke-test: 2 epochs only:
    python -m training.train_baseline --smoke-test

    # Full dataset (cloud only):
    USE_DEV_SUBSET=false python -m training.train_baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_YOLO_MODEL = "yolo11n-obb.pt"  # nano OBB — fastest for dev
_IMGSZ      = 256                # matches our synthetic FITS size
_EPOCHS_FULL  = 50
_EPOCHS_SMOKE = 2
_BATCH_SIZE   = 4

_DEV_ANN  = Path("data/annotations/dev_subset.json")
_FULL_ANN = Path("data/annotations/full_dataset.json")
_WORK_DIR = Path("weights/yolo_baseline")


# ---------------------------------------------------------------------------
# COCO → YOLO OBB dataset layout
# ---------------------------------------------------------------------------

def _coco_to_yolo_dataset(
    coco_path: Path,
    out_dir: Path,
    val_frac: float = 0.2,
    seed: int = 42,
) -> Path:
    """Convert a COCO JSON annotation file to a YOLO OBB dataset directory.

    Writes:
      out_dir/
        images/train/   ← symlinks to FITS files (YOLO reads them)
        images/val/
        labels/train/   ← YOLO OBB .txt files
        labels/val/
        dataset.yaml

    Args:
        coco_path: Path to the COCO JSON annotation file.
        out_dir: Destination directory (created if absent).
        val_frac: Fraction of images to reserve for validation.
        seed: RNG seed for train/val split.

    Returns:
        Path to the generated dataset.yaml.
    """
    import random
    random.seed(seed)

    with open(coco_path) as f:
        coco = json.load(f)

    images   = coco["images"]
    ann_root = coco_path.parent  # FITS paths are relative to here

    # Build image_id → annotations lookup
    id_to_anns: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        id_to_anns.setdefault(ann["image_id"], []).append(ann)

    # Train / val split
    shuffled = images[:]
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    val_ids   = {img["id"] for img in shuffled[:n_val]}
    train_ids = {img["id"] for img in shuffled[n_val:]}

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # YOLO requires standard image formats; convert FITS → PNG via FITSLoader
    from inference.fits_loader import FITSLoader
    import cv2
    fits_loader = FITSLoader()

    for img_meta in images:
        img_id   = img_meta["id"]
        fname    = img_meta["file_name"]
        split    = "train" if img_id in train_ids else "val"
        img_w    = img_meta["width"]
        img_h    = img_meta["height"]

        src = (ann_root / fname).resolve()
        # YOLO needs a standard image extension; save as .png
        stem = Path(fname).stem
        dst = (out_dir / "images" / split / (stem + ".png")).resolve()

        if not dst.exists():
            if src.exists():
                try:
                    loaded = fits_loader.load(str(src))
                    arr = loaded["array"]  # uint8 (H, W, 3)
                    cv2.imwrite(str(dst), arr)
                except Exception as exc:
                    logger.warning("Failed to convert %s: %s — creating zero PNG", src, exc)
                    import numpy as np
                    cv2.imwrite(str(dst), np.zeros((img_h, img_w, 3), dtype=np.uint8))
            else:
                logger.warning("FITS not found: %s — creating zero PNG", src)
                import numpy as np
                cv2.imwrite(str(dst), np.zeros((img_h, img_w, 3), dtype=np.uint8))

        # YOLO OBB label: one line per annotation (stem matches the .png image)
        # Format: class_id x1_n y1_n x2_n y2_n x3_n y3_n x4_n y4_n
        # (normalized corner coordinates of the rotated bbox, same as DOTA format)
        label_path = out_dir / "labels" / split / (stem + ".txt")
        anns = id_to_anns.get(img_id, [])
        lines = []
        import math as _math
        for ann in anns:
            obb = ann["obb"]  # [cx_px, cy_px, w_px, h_px, angle_deg]
            cx, cy, w, h, angle_deg = obb[0], obb[1], obb[2], obb[3], obb[4]
            rad = _math.radians(angle_deg)
            cos_a, sin_a = _math.cos(rad), _math.sin(rad)
            hw, hh = w / 2, h / 2
            # Four corners in local frame, rotated, then shifted to image coords
            corners = [
                (cx + (-hw) * cos_a - (-hh) * sin_a, cy + (-hw) * sin_a + (-hh) * cos_a),
                (cx + ( hw) * cos_a - (-hh) * sin_a, cy + ( hw) * sin_a + (-hh) * cos_a),
                (cx + ( hw) * cos_a - ( hh) * sin_a, cy + ( hw) * sin_a + ( hh) * cos_a),
                (cx + (-hw) * cos_a - ( hh) * sin_a, cy + (-hw) * sin_a + ( hh) * cos_a),
            ]
            coords = " ".join(
                f"{x / img_w:.6f} {y / img_h:.6f}" for x, y in corners
            )
            lines.append(f"0 {coords}")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    # Write dataset.yaml
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\nnc: 1\n"
        f"names: ['streak']\n"
    )

    n_train = len(train_ids)
    n_val_actual = len(val_ids)
    logger.info(
        "YOLO dataset: %d train / %d val → %s",
        n_train, n_val_actual, yaml_path,
    )
    return yaml_path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    smoke_test: bool = False,
    work_dir: Path = _WORK_DIR,
    imgsz: int = _IMGSZ,
) -> None:
    """Train YOLO11-OBB on the dev subset (or full dataset if USE_DEV_SUBSET=false).

    Args:
        smoke_test: If True, run 2 epochs and exit.
        work_dir: Directory for weights and logs.
        imgsz: Image size for training.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics required: pip install ultralytics"
        ) from exc

    use_dev = os.environ.get("USE_DEV_SUBSET", "true").lower() != "false"
    ann_path = _DEV_ANN if use_dev else _FULL_ANN

    if not ann_path.exists():
        if use_dev:
            logger.info("dev_subset.json not found — building it now …")
            from training.make_dev_subset import build_dev_subset
            build_dev_subset(
                fits_dir=Path("data/dev_subset"),
                annotations_path=_DEV_ANN,
            )
        else:
            raise FileNotFoundError(
                f"Annotation file not found: {ann_path}\n"
                "Run training/make_dev_subset.py first, or set USE_DEV_SUBSET=true."
            )

    work_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = work_dir / "dataset"
    logger.info("Converting COCO → YOLO dataset …")
    yaml_path = _coco_to_yolo_dataset(ann_path, dataset_dir)

    epochs = _EPOCHS_SMOKE if smoke_test else _EPOCHS_FULL

    logger.info(
        "Training YOLO11n-OBB: %d epochs, imgsz=%d, data=%s",
        epochs, imgsz, yaml_path,
    )

    model = YOLO(_YOLO_MODEL)
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=_BATCH_SIZE,
        # Use absolute path so YOLO saves directly to work_dir/run/ rather
        # than under Ultralytics' default runs/obb/{project}/ directory.
        project=str(work_dir.resolve()),
        name="run",
        exist_ok=True,
        # MPS crashes with 0-instance batches (blank images) due to a PyTorch
        # MPS limitation with zero-size tensors in OBB loss computation.
        device="cuda" if _device_is_cuda() else "cpu",
        workers=0,
        verbose=True,
        save=True,
        val=True,
    )

    best_path = work_dir.resolve() / "run" / "weights" / "best.pt"
    if best_path.exists():
        print(f"\nBest weights: {best_path}")
    else:
        print(f"\nTraining complete. Weights in {work_dir.resolve()}/run/weights/")

    if smoke_test:
        print("Smoke test passed.")


def _device_is_mps() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


def _device_is_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run 2 epochs and exit (verify environment only)",
    )
    parser.add_argument(
        "--work-dir", default=str(_WORK_DIR),
        help=f"Output directory for weights and logs (default: {_WORK_DIR})",
    )
    parser.add_argument(
        "--imgsz", type=int, default=_IMGSZ,
        help=f"Training image size (default: {_IMGSZ})",
    )
    args = parser.parse_args()

    train(
        smoke_test=args.smoke_test,
        work_dir=Path(args.work_dir),
        imgsz=args.imgsz,
    )
