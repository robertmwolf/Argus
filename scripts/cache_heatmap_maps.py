"""Cache per-image heatmap probability maps for fast threshold sweeps.

Runs the ViT-S/B backbone once per val image and saves the composited
full-image probability heatmap (downsampled to feature-map resolution)
as an NPY file.  Future threshold sweeps can load these cached heatmaps
and re-apply thresholding + NMS in seconds without touching the GPU.

Each output NPY is float32 shaped (H // 16, W // 16) — one value per
ViT-S/16 or ViT-B/16 feature patch in the source image.  Overlapping
tile heatmaps are merged via pixel-wise max.

Usage:
    python scripts/cache_heatmap_maps.py \\
        --annotations data/annotations/val_run17_fits.json \\
        --checkpoint weights/run15_vits/best.pt \\
        --output-dir /tmp/argus_run15_heatmap_cache \\
        --norm-mode zscore

    # ViT-B:
    python scripts/cache_heatmap_maps.py \\
        --annotations data/annotations/val_run17_fits.json \\
        --checkpoint weights/run17_vitb/best.pt \\
        --output-dir /tmp/argus_run17_heatmap_cache \\
        --norm-mode zscore

The manifest.json written to --output-dir records the checkpoint path,
tiling parameters, and a mapping from image_id → npy filename.
Pass --output-dir to ``evaluate_dinov3_heatmap.py --heatmap-cache`` to
run zero-GPU threshold sweeps from the cached maps.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inference.convnext_heatmap_detector import _run_single_tile_probs
from inference.fits_loader import FITSLoader, apply_norm
from inference.tiled_pipeline import tile_image
from inference.vits_heatmap_detector import _load_model

logger = logging.getLogger(__name__)

# ViT-S/16 and ViT-B/16 both have patch stride 16
PATCH_SIZE = 16


def _build_feat_heatmap(
    array: np.ndarray,
    model: Any,
    image_size: int,
    device: Any,
    native_tile_size: int,
    tile_overlap: float,
) -> np.ndarray:
    """Run tiled inference and return a feature-resolution probability heatmap.

    Args:
        array: uint8 RGB image, shape (H, W, 3).
        model: Loaded heatmap model.
        image_size: Model input size (from checkpoint metadata).
        device: Torch device.
        native_tile_size: Tile size in source pixels (default 400).
        tile_overlap: Fractional overlap between tiles (default 0.5).

    Returns:
        Float32 heatmap at feature-map resolution (ceil(H/16), ceil(W/16))
        with values in [0, 1].  Overlapping tiles are merged via max-pool.
    """
    h_full, w_full = array.shape[:2]
    # Feature grid dimensions (ceiling so every pixel is covered)
    h_feat = (h_full + PATCH_SIZE - 1) // PATCH_SIZE
    w_feat = (w_full + PATCH_SIZE - 1) // PATCH_SIZE
    heat_feat = np.zeros((h_feat, w_feat), dtype=np.float32)

    if max(h_full, w_full) <= native_tile_size:
        heat_px, _, _, _ = _run_single_tile_probs(array, model, image_size, device)
        return cv2.resize(heat_px, (w_feat, h_feat), interpolation=cv2.INTER_LINEAR)

    for tile, x0, y0 in tile_image(array, native_tile_size, tile_overlap):
        th, tw = tile.shape[:2]
        heat_px, _, _, _ = _run_single_tile_probs(tile, model, image_size, device)
        # Downsample tile's pixel-space heatmap to feature resolution
        tile_h_feat = (th + PATCH_SIZE - 1) // PATCH_SIZE
        tile_w_feat = (tw + PATCH_SIZE - 1) // PATCH_SIZE
        tile_feat = cv2.resize(
            heat_px, (tile_w_feat, tile_h_feat), interpolation=cv2.INTER_LINEAR
        )
        # Place into the full-image feature grid via max (handles overlap)
        x0f = x0 // PATCH_SIZE
        y0f = y0 // PATCH_SIZE
        x1f = min(x0f + tile_w_feat, w_feat)
        y1f = min(y0f + tile_h_feat, h_feat)
        dw = x1f - x0f
        dh = y1f - y0f
        np.maximum(
            heat_feat[y0f:y1f, x0f:x1f],
            tile_feat[:dh, :dw],
            out=heat_feat[y0f:y1f, x0f:x1f],
        )

    return heat_feat


def _resolve_path(fname: str, ann_dir: Path) -> Path:
    for base in [Path("."), ann_dir, ann_dir.parent]:
        p = (base / fname).resolve()
        if p.exists():
            return p
    return Path(fname)


def _load_image(path: Path, fits_loader: FITSLoader, norm_mode: str) -> np.ndarray | None:
    try:
        suffix = path.suffix.lower()
        if suffix in {".fits", ".fit", ".fts"}:
            return np.asarray(fits_loader.load(path)["array"], dtype=np.uint8)
        if suffix == ".npy":
            raw = np.load(str(path))
            if raw.dtype == np.uint8:
                return np.stack([raw, raw, raw], axis=-1)
            return apply_norm(raw.astype(np.float32), norm_mode)
        from PIL import Image
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        logger.warning("Could not load %s: %s", path, exc)
        return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--annotations", required=True,
                    help="COCO annotation file (val or test FITS-based)")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to best.pt checkpoint (ViT-S or ViT-B cached-head format)")
    ap.add_argument("--output-dir", required=True,
                    help="Directory to write per-image NPY heatmaps and manifest.json")
    ap.add_argument("--norm-mode", choices=["autostretch", "zscore", "zscale"],
                    default="zscore",
                    help="Pixel normalisation (must match training run, default: zscore)")
    ap.add_argument("--native-tile-size", type=int, default=400,
                    help="Tile size in source pixels (default: 400, matches Run 15/17)")
    ap.add_argument("--tile-overlap", type=float, default=0.5,
                    help="Fractional tile overlap (default: 0.5)")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Process only the first N images (for testing)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["ARGUS_NORM"] = args.norm_mode

    logger.info("Loading model from %s", ckpt_path)
    model, image_size, device, _use_geo = _load_model(ckpt_path)
    logger.info("Model loaded (image_size=%d, device=%s)", image_size, device)

    coco = json.loads(Path(args.annotations).read_text())
    images_meta = coco.get("images", [])
    if args.max_samples:
        images_meta = images_meta[: args.max_samples]

    ann_dir = Path(args.annotations).resolve().parent
    fits_loader = FITSLoader()

    manifest: dict[str, str] = {}
    n_done = 0
    n_skip = 0

    for meta in images_meta:
        image_id = int(meta["id"])
        img_path = _resolve_path(meta["file_name"], ann_dir)
        arr = _load_image(img_path, fits_loader, args.norm_mode)
        if arr is None:
            logger.warning("Skipping image_id=%d (load failed: %s)", image_id, img_path)
            n_skip += 1
            continue
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=2)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        # Crop to tile_origin window if present (matches evaluate_dinov3_heatmap.py)
        tile_origin = meta.get("tile_origin")
        if tile_origin is not None:
            x0o, y0o = int(tile_origin[0]), int(tile_origin[1])
            crop_w = int(meta.get("width", arr.shape[1]))
            crop_h = int(meta.get("height", arr.shape[0]))
            arr = arr[y0o : y0o + crop_h, x0o : x0o + crop_w]

        heat_feat = _build_feat_heatmap(
            arr,
            model,
            image_size,
            device,
            native_tile_size=args.native_tile_size,
            tile_overlap=args.tile_overlap,
        )

        npy_name = f"{image_id:06d}.npy"
        np.save(str(out_dir / npy_name), heat_feat)
        manifest[str(image_id)] = npy_name
        n_done += 1
        logger.info(
            "Cached image_id=%d  feat_shape=%s  (%d/%d)",
            image_id, heat_feat.shape, n_done, len(images_meta),
        )

    manifest_data = {
        "checkpoint": str(ckpt_path.resolve()),
        "annotations": str(Path(args.annotations).resolve()),
        "norm_mode": args.norm_mode,
        "native_tile_size": args.native_tile_size,
        "tile_overlap": args.tile_overlap,
        "patch_size": PATCH_SIZE,
        "n_images": n_done,
        "images": manifest,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, indent=2))
    logger.info(
        "Manifest written → %s  (%d cached, %d skipped)",
        manifest_path, n_done, n_skip,
    )
    return 0 if n_done > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
