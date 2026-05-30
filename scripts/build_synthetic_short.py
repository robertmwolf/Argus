"""Generate synthetic short-band training samples from real Atwood backgrounds.

Loads full-frame Atwood FITS images from atwood_train.json, extracts 400px crops
from random positions, injects 1-3 synthetic short streaks (50-269px at NATIVE
Atwood resolution, which corresponds to ~3-17px at 400px model input), then saves
as PNG with a standalone COCO annotation JSON.

Why native-resolution injection:
  The MMDet training pipeline resizes full Atwood frames (6248px) to 400px model
  input (scale ~0.064x).  Injecting streaks at native scale means the training
  pipeline sees them at the correct apparent size: 50-269px native → 3-17px model
  input.  This trains the model to detect real Atwood short-band objects.

Why crop then save (not full-frame):
  FITS images are large (6248×4176).  Saving 200 augmented FITS would be ~2GB.
  Instead we extract 400px crops that already contain the injected streak, saving
  as PNG.  The resulting file is ~480KB and loads as a standard image in MMDet.

Output:
  data/augmented/synthetic_short/<stem>_crop_<n>.png
  data/annotations/synthetic_short_atwood.json   (standalone COCO, no original anns)

Usage:
    python scripts/build_synthetic_short.py
    python scripts/build_synthetic_short.py --n-images 200 --crops-per-image 2 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.augmentations import SyntheticStreakInject

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_ANN_IN   = _REPO_ROOT / "data/annotations/atwood_train.json"
_OUT_DIR  = _REPO_ROOT / "data/augmented/synthetic_short"
_ANN_OUT  = _REPO_ROOT / "data/annotations/synthetic_short_atwood.json"
_CATEGORIES = [{"id": 1, "name": "streak", "supercategory": "satellite"}]

# Atwood native image size — used to clamp injection length
ATWOOD_W = 6248
ATWOOD_H = 4176
ATWOOD_DIAG = math.sqrt(ATWOOD_W**2 + ATWOOD_H**2)

# Short-band: 50–269px at native Atwood resolution
SHORT_MIN_PX  = 50.0
SHORT_MAX_PX  = 269.0

# Crop size fed to the model (must match MMDet resize target)
CROP_SIZE = 400


def _load_fits_as_uint8(fits_path: Path) -> np.ndarray | None:
    """Load a FITS file and normalise to uint8 RGB using z-score stretch."""
    try:
        import astropy.io.fits as astrofits
        with astrofits.open(str(fits_path)) as hdul:
            data = hdul[0].data.astype(np.float32)
        if data.ndim == 3:
            data = data[0]
        mu  = float(np.mean(data))
        sig = float(np.std(data))
        if sig < 1e-6:
            return None
        norm = np.clip((data - mu) / sig, -3.0, 3.0)
        uint8 = ((norm + 3.0) / 6.0 * 255.0).astype(np.uint8)
        return cv2.cvtColor(uint8, cv2.COLOR_GRAY2BGR)
    except Exception as e:
        logger.warning("Failed to load %s: %s", fits_path, e)
        return None


def _random_crop(
    img: np.ndarray,
    crop_size: int,
    rng: random.Random,
    streak_bbox: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, int, int] | None:
    """Extract a crop_size×crop_size crop that contains the streak bbox.

    Returns (crop, x_offset, y_offset) or None if image is too small.
    """
    h, w = img.shape[:2]
    if h < crop_size or w < crop_size:
        return None

    if streak_bbox is not None:
        sx, sy, sw, sh = streak_bbox
        # Pick a crop that contains the streak centre
        cx = sx + sw / 2.0
        cy = sy + sh / 2.0
        x0_lo = int(max(0, cx - crop_size + sw / 2.0 + 1))
        x0_hi = int(min(w - crop_size, cx - sw / 2.0 - 1))
        y0_lo = int(max(0, cy - crop_size + sh / 2.0 + 1))
        y0_hi = int(min(h - crop_size, cy - sh / 2.0 - 1))
        if x0_lo > x0_hi or y0_lo > y0_hi:
            # Fallback: centre the crop on the streak
            x0 = int(max(0, min(w - crop_size, cx - crop_size // 2)))
            y0 = int(max(0, min(h - crop_size, cy - crop_size // 2)))
        else:
            x0 = rng.randint(x0_lo, x0_hi)
            y0 = rng.randint(y0_lo, y0_hi)
    else:
        x0 = rng.randint(0, w - crop_size)
        y0 = rng.randint(0, h - crop_size)

    return img[y0:y0 + crop_size, x0:x0 + crop_size].copy(), x0, y0


def build(
    n_images: int,
    crops_per_image: int,
    seed: int,
    snr_scale: float,
    args_ns=None,
) -> None:
    rng = random.Random(seed)
    _OUT_DIR.mkdir(exist_ok=True)

    src = json.loads(_ANN_IN.read_text())
    pool = src["images"].copy()
    rng.shuffle(pool)
    selected = (pool * math.ceil(n_images / max(len(pool), 1)))[:n_images]

    # Short-band injector at native Atwood resolution.
    # When snr_scale_max > snr_scale, each image gets a freshly-sampled scale
    # drawn uniformly from [snr_scale, snr_scale_max] to cover bright→faint range.
    snr_scale_max = getattr(args_ns, 'snr_scale_max', snr_scale) if hasattr(args_ns, 'snr_scale_max') else snr_scale
    max_frac = SHORT_MAX_PX / ATWOOD_DIAG

    def _make_injector() -> SyntheticStreakInject:
        scale = snr_scale if snr_scale_max <= snr_scale else rng.uniform(snr_scale, snr_scale_max)
        return SyntheticStreakInject(
            p=1.0,
            min_length_px=SHORT_MIN_PX,
            max_length_fraction=max_frac,
            snr_scale=scale,
        )

    out_images: list[dict] = []
    out_annotations: list[dict] = []
    img_id = 1
    ann_id = 1

    for idx, img_info in enumerate(selected):
        fits_path = Path(img_info["file_name"])
        if not fits_path.exists():
            logger.warning("FITS not found: %s", fits_path)
            continue

        img = _load_fits_as_uint8(fits_path)
        if img is None:
            continue

        for crop_n in range(crops_per_image):
            # Inject streak into the full-frame image then crop around it
            injector = _make_injector()
            injected, new_bboxes, _ = injector.inject(img, [], [])

            if not new_bboxes:
                continue

            # Use the first injected bbox to centre the crop
            first_bbox = new_bboxes[0]  # (x_min, y_min, x_max, y_max)
            bbox_xywh = (
                first_bbox[0],
                first_bbox[1],
                first_bbox[2] - first_bbox[0],
                first_bbox[3] - first_bbox[1],
            )

            result = _random_crop(injected, CROP_SIZE, rng, bbox_xywh)
            if result is None:
                continue
            crop, x0, y0 = result

            stem   = fits_path.stem
            fname  = f"{stem}_syn_{crop_n:02d}_{idx:04d}.png"
            out_path = _OUT_DIR / fname
            cv2.imwrite(str(out_path), crop)

            out_images.append({
                "id": img_id,
                "file_name": str(out_path),
                "width": CROP_SIZE,
                "height": CROP_SIZE,
            })

            # Add all bboxes that fall (even partially) within the crop
            for x_min, y_min, x_max, y_max in new_bboxes:
                bx = x_min - x0
                by = y_min - y0
                bw = x_max - x_min
                bh = y_max - y_min
                # Clip to crop bounds
                bx2 = min(bx + bw, CROP_SIZE)
                by2 = min(by + bh, CROP_SIZE)
                bx  = max(bx, 0.0)
                by  = max(by, 0.0)
                bw  = bx2 - bx
                bh  = by2 - by
                if bw < 5 or bh < 5:
                    continue
                diag = math.sqrt(bw**2 + bh**2)
                if diag < 5:
                    continue
                out_annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [float(bx), float(by), float(bw), float(bh)],
                    "area": float(bw * bh),
                    "iscrowd": 0,
                })
                ann_id += 1

            img_id += 1

        if (idx + 1) % 20 == 0:
            logger.info("Processed %d / %d source images (%d crops so far)",
                        idx + 1, len(selected), len(out_images))

    coco = {
        "info": {
            "description": "Synthetic short-band streaks on Atwood backgrounds",
            "short_min_px": SHORT_MIN_PX,
            "short_max_px": SHORT_MAX_PX,
            "snr_scale": snr_scale,
            "crop_size": CROP_SIZE,
        },
        "categories": _CATEGORIES,
        "images": out_images,
        "annotations": out_annotations,
    }

    ann_out = args_ns.output if (args_ns and hasattr(args_ns, 'output') and args_ns.output) else _ANN_OUT

    # Append mode: merge with existing JSON
    if args_ns and getattr(args_ns, 'append', False) and ann_out.exists():
        existing = json.loads(ann_out.read_text())
        max_img_id = max((i['id'] for i in existing['images']), default=0)
        max_ann_id = max((a['id'] for a in existing['annotations']), default=0)
        for img in out_images:
            img['id'] += max_img_id
        for a in out_annotations:
            a['id'] += max_ann_id
            a['image_id'] += max_img_id
        coco['images']       = existing['images']       + out_images
        coco['annotations']  = existing['annotations']  + out_annotations

    ann_out.write_text(json.dumps(coco, indent=2))
    logger.info(
        "Done. %d crop images, %d synthetic annotations → %s",
        len(out_images), len(out_annotations), ann_out,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-images", type=int, default=200,
                        help="Number of source FITS images to augment (default: 200).")
    parser.add_argument("--crops-per-image", type=int, default=1,
                        help="Crops (and injections) per source image (default: 1).")
    parser.add_argument("--snr-scale", type=float, default=0.2,
                        help="Minimum streak brightness multiplier (default: 0.2 = faint).")
    parser.add_argument("--snr-scale-max", type=float, default=1.0,
                        help="Maximum brightness multiplier; each image samples uniformly "
                             "from [snr-scale, snr-scale-max] (default: 1.0 = full range).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output annotation JSON (default: data/annotations/synthetic_short_atwood.json).")
    parser.add_argument("--append", action="store_true",
                        help="Append to an existing annotation JSON instead of overwriting.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build(
        n_images=args.n_images,
        crops_per_image=args.crops_per_image,
        seed=args.seed,
        snr_scale=args.snr_scale,
        args_ns=args,
    )


if __name__ == "__main__":
    main()
