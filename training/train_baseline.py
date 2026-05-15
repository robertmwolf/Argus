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

def _obb_fields(obb: list | dict) -> tuple[float, float, float, float, float]:
    """Return (cx, cy, w, h, angle_deg) from either list or dict OBB."""
    if isinstance(obb, dict):
        return obb["cx"], obb["cy"], obb["w"], obb["h"], obb["angle_deg"]
    return float(obb[0]), float(obb[1]), float(obb[2]), float(obb[3]), float(obb[4])


def _clip_obb_to_tile(
    cx: float, cy: float, w: float, h: float, angle_deg: float,
    tx: int, ty: int, tile_size: int,
    min_visible_px: float = 20.0,
) -> tuple[float, float, float, float, float] | None:
    """Clip a streak OBB to a tile; return OBB in tile-local coords or None.

    Models the streak as a parametric line through (cx, cy) in direction
    (cos θ, sin θ).  Clips the parameter range [-w/2, w/2] to the tile
    bounds, then returns a new OBB centred on the visible segment.

    Args:
        cx, cy: Streak centre in full-image pixel coordinates.
        w: Streak length (long axis) in pixels.
        h: Streak width (short axis) in pixels.
        angle_deg: Streak orientation in degrees.
        tx, ty: Top-left corner of the tile in full-image coordinates.
        tile_size: Tile width and height in pixels.
        min_visible_px: Minimum clipped length to keep the annotation.

    Returns:
        (cx_tile, cy_tile, w_clipped, h, angle_deg) in tile-local pixels,
        or None if the streak is not meaningfully visible in this tile.
    """
    import math as _m
    rad = _m.radians(angle_deg)
    cos_a, sin_a = _m.cos(rad), _m.sin(rad)

    tx2, ty2 = tx + tile_size, ty + tile_size
    t_lo, t_hi = -w / 2.0, w / 2.0
    eps = 1e-9

    if abs(cos_a) > eps:
        ta, tb = (tx - cx) / cos_a, (tx2 - cx) / cos_a
        t_lo = max(t_lo, min(ta, tb))
        t_hi = min(t_hi, max(ta, tb))
    elif not (tx <= cx < tx2):
        return None

    if abs(sin_a) > eps:
        ta, tb = (ty - cy) / sin_a, (ty2 - cy) / sin_a
        t_lo = max(t_lo, min(ta, tb))
        t_hi = min(t_hi, max(ta, tb))
    elif not (ty <= cy < ty2):
        return None

    if t_hi - t_lo < min_visible_px:
        return None

    t_mid = (t_lo + t_hi) / 2.0
    return (
        cx + t_mid * cos_a - tx,   # cx in tile coords
        cy + t_mid * sin_a - ty,   # cy in tile coords
        t_hi - t_lo,               # clipped length
        h,
        angle_deg,
    )


def _obb_label_line(cx: float, cy: float, w: float, h: float,
                    angle_deg: float, norm: float) -> str:
    """Render one YOLO OBB label line (4 normalised corner points)."""
    import math as _m
    rad = _m.radians(angle_deg)
    cos_a, sin_a = _m.cos(rad), _m.sin(rad)
    hw, hh = w / 2, h / 2
    corners = [
        (cx + (-hw) * cos_a - (-hh) * sin_a, cy + (-hw) * sin_a + (-hh) * cos_a),
        (cx + ( hw) * cos_a - (-hh) * sin_a, cy + ( hw) * sin_a + (-hh) * cos_a),
        (cx + ( hw) * cos_a - ( hh) * sin_a, cy + ( hw) * sin_a + ( hh) * cos_a),
        (cx + (-hw) * cos_a - ( hh) * sin_a, cy + (-hw) * sin_a + ( hh) * cos_a),
    ]
    coords = " ".join(f"{x / norm:.6f} {y / norm:.6f}" for x, y in corners)
    return f"0 {coords}"


def _coco_to_yolo_dataset(
    coco_path: Path,
    out_dir: Path,
    val_frac: float = 0.2,
    seed: int = 42,
    tile_size: int = 256,
    max_bg_tiles: int = 2,
) -> Path:
    """Convert a COCO JSON annotation file to a tiled YOLO OBB dataset.

    Full-resolution images (larger than tile_size) are sliced into
    tile_size × tile_size patches.  Each patch that contains a visible
    streak segment is saved with clipped OBB labels.  Up to max_bg_tiles
    background patches per image are also included for hard-negative
    training.  Images already at or below tile_size are saved as-is.

    Writes:
      out_dir/
        images/train/   ← tile PNGs
        images/val/
        labels/train/   ← YOLO OBB .txt files (one per tile)
        labels/val/
        dataset.yaml

    Args:
        coco_path: Path to the COCO JSON annotation file.
        out_dir: Destination directory (created if absent).
        val_frac: Fraction of *source images* to reserve for validation.
        seed: RNG seed for train/val split and background tile sampling.
        tile_size: Tile edge length in pixels (should match YOLO imgsz).
        max_bg_tiles: Maximum background (empty) tiles kept per source image.

    Returns:
        Path to the generated dataset.yaml.
    """
    import math as _math
    import random

    random.seed(seed)

    with open(coco_path) as f:
        coco = json.load(f)

    images   = coco["images"]
    ann_root = coco_path.parent.parent  # annotations/ → data/

    id_to_anns: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        id_to_anns.setdefault(ann["image_id"], []).append(ann)

    shuffled = images[:]
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    val_ids   = {img["id"] for img in shuffled[:n_val]}
    train_ids = {img["id"] for img in shuffled[n_val:]}

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    from inference.fits_loader import FITSLoader
    import cv2
    import numpy as np

    fits_loader = FITSLoader()
    n_tiles_written = 0

    for img_meta in images:
        img_id = img_meta["id"]
        fname  = img_meta["file_name"]
        split  = "train" if img_id in train_ids else "val"
        stem   = Path(fname).stem

        src = (ann_root / fname).resolve()
        anns = id_to_anns.get(img_id, [])

        if src.exists():
            try:
                # Use cv2 directly for JPEG/PNG files — FITSLoader tries astropy
                # first and raises on files whose extension is .fits.jpg.
                if src.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    bgr = cv2.imread(str(src))
                    if bgr is None:
                        raise OSError(f"cv2.imread returned None for {src}")
                    arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                else:
                    loaded = fits_loader.load(str(src))
                    arr = loaded["array"]  # uint8 (H, W, 3)
            except Exception as exc:
                logger.warning("Failed to load %s: %s — skipping", src, exc)
                continue
        else:
            logger.warning("Image not found: %s — skipping", src)
            continue

        h_arr, w_arr = arr.shape[:2]

        # Small images: save as-is with full-image labels
        if max(h_arr, w_arr) <= tile_size:
            dst = out_dir / "images" / split / (stem + ".png")
            if not dst.exists():
                cv2.imwrite(str(dst), arr)
            lines = []
            for ann in anns:
                cx, cy, w, h, angle_deg = _obb_fields(ann["obb"])
                lines.append(_obb_label_line(cx, cy, w, h, angle_deg,
                                             norm=float(max(w_arr, h_arr))))
            label_path = out_dir / "labels" / split / (stem + ".txt")
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
            n_tiles_written += 1
            continue

        # Large images: tile and clip annotations
        rng = random.Random(img_id)
        bg_tile_coords: list[tuple[int, int]] = []
        pos_tile_coords: list[tuple[int, int]] = []

        xs = range(0, w_arr - tile_size + 1, tile_size)
        ys = range(0, h_arr - tile_size + 1, tile_size)

        for ty in ys:
            for tx in xs:
                clipped = []
                for ann in anns:
                    cx, cy, w, h, angle_deg = _obb_fields(ann["obb"])
                    result = _clip_obb_to_tile(cx, cy, w, h, angle_deg,
                                               tx, ty, tile_size)
                    if result is not None:
                        clipped.append(result)
                if clipped:
                    pos_tile_coords.append((tx, ty, clipped))  # type: ignore[arg-type]
                else:
                    bg_tile_coords.append((tx, ty))

        # Sample background tiles
        rng.shuffle(bg_tile_coords)
        bg_sample = bg_tile_coords[:max_bg_tiles]

        for entry in pos_tile_coords:
            tx, ty, clipped = entry  # type: ignore[misc]
            tile_id = f"{stem}_t{tx:04d}_{ty:04d}"
            dst = out_dir / "images" / split / (tile_id + ".png")
            if not dst.exists():
                patch = arr[ty:ty + tile_size, tx:tx + tile_size]
                cv2.imwrite(str(dst), patch)
            lines = [_obb_label_line(cx, cy, w, h, ad, norm=float(tile_size))
                     for cx, cy, w, h, ad in clipped]
            lbl = out_dir / "labels" / split / (tile_id + ".txt")
            lbl.write_text("\n".join(lines) + "\n")
            n_tiles_written += 1

        for tx, ty in bg_sample:
            tile_id = f"{stem}_bg{tx:04d}_{ty:04d}"
            dst = out_dir / "images" / split / (tile_id + ".png")
            if not dst.exists():
                patch = arr[ty:ty + tile_size, tx:tx + tile_size]
                cv2.imwrite(str(dst), patch)
            lbl = out_dir / "labels" / split / (tile_id + ".txt")
            lbl.write_text("")
            n_tiles_written += 1

    logger.info("Tiled dataset: %d tiles written to %s", n_tiles_written, out_dir)

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
    epochs: int | None = None,
) -> None:
    """Train YOLO11-OBB on the dev subset (or full dataset if USE_DEV_SUBSET=false).

    Args:
        smoke_test: If True, run 2 epochs and exit.
        work_dir: Directory for weights and logs.
        imgsz: Image size for training.
        epochs: Number of training epochs. Defaults to _EPOCHS_FULL (50).
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics required: pip install ultralytics"
        ) from exc

    use_dev = os.environ.get("USE_DEV_SUBSET", "true").lower() != "false"
    if use_dev:
        ann_path = _DEV_ANN
    else:
        train_ann_env = os.environ.get("TRAIN_ANN_FILE", "")
        ann_path = (Path("data/annotations") / train_ann_env
                    if train_ann_env else Path("data/annotations/train.json"))

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
                "Set TRAIN_ANN_FILE to a valid annotations filename, or set USE_DEV_SUBSET=true."
            )

    work_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = work_dir / "dataset"
    yaml_path = dataset_dir / "dataset.yaml"
    if yaml_path.exists():
        logger.info("Reusing existing tiled dataset at %s", dataset_dir)
    else:
        logger.info("Converting COCO → tiled YOLO dataset (tile_size=%d) …", imgsz)
        yaml_path = _coco_to_yolo_dataset(ann_path, dataset_dir, tile_size=imgsz)

    epochs = _EPOCHS_SMOKE if smoke_test else (epochs if epochs is not None else _EPOCHS_FULL)

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
    parser.add_argument(
        "--epochs", type=int, default=None,
        help=f"Number of training epochs (default: {_EPOCHS_FULL})",
    )
    args = parser.parse_args()

    train(
        smoke_test=args.smoke_test,
        work_dir=Path(args.work_dir),
        imgsz=args.imgsz,
        epochs=args.epochs,
    )
