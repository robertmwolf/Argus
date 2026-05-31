"""Cache frozen DINOv3 features for the plain heatmap spike.

Supports two modes:

Full-image mode (default, --native-tile-size 0):
    Each image is letterboxed to ``--image-size`` and cached as one entry.
    Suitable for whole-frame inference, but medium streaks in large images
    (e.g. 6248 px Atwood) span <2 feature patches at 384 px — the model
    learns blob detections and cannot produce tight OBBs.

Tiled mode (--native-tile-size N):
    Each image is partitioned into overlapping N×N px crops, each letterboxed
    to ``--image-size``.  Annotations are transformed to tile-local coordinates
    and filtered to tiles where the streak centre falls inside.  Use
    ``--native-tile-size 1562`` for Atwood 6248 px images (4 tiles/row) —
    a 300 px medium streak then spans ~4.6 feature patches rather than ~1.2.

    IMPORTANT: a model trained in tiled mode must be evaluated with
    ``--tiled`` in ``evaluate_dinov3_heatmap.py``.  Mixing training and
    inference modes produces distribution mismatch and inflated false
    positives (see docs/training_methods.md §6).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inference.device import get_device
from inference.fits_loader import FITSLoader
from models.plain_dinov3.streak_heatmap import (
    ConvNeXtStreakHeatmap,
    DINOv3StreakHeatmap,
    imagenet_normalize,
)
from training.dinov3_heatmap_dataset import StreakHeatmapDataset, collate_heatmap_batch

logger = logging.getLogger(__name__)


def _letterbox_array(array: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
    """Letterbox ``array`` into a square ``size×size`` canvas.

    Returns:
        (canvas_uint8, scale, pad_x, pad_y)
    """
    h, w = array.shape[:2]
    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    pad_x = (size - new_w) / 2.0
    pad_y = (size - new_h) / 2.0
    resized = np.array(Image.fromarray(array).resize((new_w, new_h), Image.BILINEAR))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[round(pad_y):round(pad_y) + new_h, round(pad_x):round(pad_x) + new_w] = resized
    return canvas, float(scale), float(pad_x), float(pad_y)


def _build_tile_targets(
    anns: list[dict[str, Any]],
    tile_w: int,
    tile_h: int,
    image_size: int,
    scale: float,
    pad_x: float,
    pad_y: float,
    patch_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build heatmap + geometry targets for one tile.

    Annotations are given in **full-image** coordinates; ``tile_x0`` /
    ``tile_y0`` have already been subtracted before this call so all ``cx/cy``
    values are in tile-local pixels.

    Returns:
        (heatmap, geometry) tensors matching StreakHeatmapDataset conventions.
    """
    grid = image_size // patch_size
    target = np.zeros((grid, grid), dtype=np.float32)
    geom   = np.zeros((4, grid, grid), dtype=np.float32)

    yy, xx = np.mgrid[0:grid, 0:grid].astype(np.float32)
    px = (xx + 0.5) * patch_size
    py = (yy + 0.5) * patch_size

    for ann in anns:
        obb = ann.get("obb")
        if not obb:
            continue
        if isinstance(obb, dict):
            cx   = float(obb["cx"])   * scale + pad_x
            cy   = float(obb["cy"])   * scale + pad_y
            w    = float(obb["w"])    * scale
            h    = float(obb["h"])    * scale
            adeg = float(obb.get("angle_deg", 0.0))
        else:
            cx   = float(obb[0]) * scale + pad_x
            cy   = float(obb[1]) * scale + pad_y
            w    = float(obb[2]) * scale
            h    = float(obb[3]) * scale
            adeg = float(obb[4])

        length = max(w, h)
        width  = max(min(w, h), patch_size)
        angle  = math.radians(adeg)
        ux, uy = math.cos(angle), math.sin(angle)

        dx = px - cx; dy = py - cy
        along  = dx * ux + dy * uy
        across = np.abs(-dx * uy + dy * ux)
        mask   = (np.abs(along) <= length / 2 + 8.0) & (across <= width / 2 + 8.0)
        target[mask] = 1.0
        geom[0, mask] = math.cos(2.0 * angle)
        geom[1, mask] = math.sin(2.0 * angle)
        geom[2, mask] = min(length / image_size, 2.0)
        geom[3, mask] = min(width  / image_size, 1.0)

    geom[:, target <= 0] = 0.0
    return torch.from_numpy(target).unsqueeze(0), torch.from_numpy(geom)


def _load_image_array(path: Path, loader: FITSLoader) -> np.ndarray | None:
    suffix = path.suffix.lower()
    try:
        if suffix in {".fits", ".fit", ".fts"}:
            arr = np.asarray(loader.load(path)["array"], dtype=np.uint8)
        else:
            with Image.open(path) as im:
                arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=2)
        return arr
    except Exception as exc:
        logger.warning("Could not load %s: %s", path, exc)
        return None


def _cache_tiled(
    coco_data: dict[str, Any],
    model: Any,
    device: torch.device,
    image_size: int,
    feature_dir: Path,
    native_tile_size: int,
    tile_overlap: float,
    max_samples: int | None,
    ann_dir: Path,
    min_area_fraction: float = 0.25,
    neg_tiles_per_image: int = 2,
    random_seed: int = 42,
) -> list[dict]:
    """Cache features by tiling each image, selecting only annotation-covered tiles.

    Tile selection mirrors ``build_tiled_brentimages_json.py``:

    * **Positive tiles** — included when ≥1 annotation bbox retains at least
      ``min_area_fraction`` (default 25 %) of its original area after clipping
      to the tile.  This is checked *before* running the backbone so we never
      pay for tiles that carry no training signal.
    * **Negative tiles** — ``neg_tiles_per_image`` random tiles per image that
      has no annotations at all (domain adaptation; background appearance).
    * **Background tiles inside positive images** are *not* added.  The heatmap
      loss treats missing patches as negatives automatically.

    This keeps tile counts at roughly 3–6 per positive image (vs 620 with the
    old exhaustive approach), making the cache feasible on a single machine.
    """
    import random as _random
    from inference.tiled_pipeline import tile_image

    rng = _random.Random(random_seed)
    loader = FITSLoader()
    images_meta = coco_data.get("images", [])
    if max_samples:
        images_meta = images_meta[:max_samples]

    id_to_anns: dict[int, list] = {}
    for ann in coco_data.get("annotations", []):
        id_to_anns.setdefault(int(ann["image_id"]), []).append(ann)

    def _resolve(fname: str) -> Path:
        raw = Path(fname)
        if raw.is_absolute():
            return raw
        for base in [ann_dir, ann_dir.parent, Path("data")]:
            p = (base / raw).resolve()
            if p.exists():
                return p
        return ann_dir / raw

    def _ann_area_in_tile(ann: dict, x0: int, y0: int, ts: int) -> float:
        """Return fraction of annotation bbox visible inside the tile."""
        bbox = ann.get("bbox")
        if not bbox:
            return 0.0
        bx, by, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        orig_area = bw * bh
        if orig_area <= 0:
            return 0.0
        x1 = max(bx, float(x0));  y1 = max(by, float(y0))
        x2 = min(bx + bw, float(x0 + ts)); y2 = min(by + bh, float(y0 + ts))
        if x2 <= x1 or y2 <= y1:
            return 0.0
        return (x2 - x1) * (y2 - y1) / orig_area

    manifest: list[dict] = []
    for img_meta in images_meta:
        img_id   = int(img_meta["id"])
        img_path = _resolve(str(img_meta["file_name"]))
        array    = _load_image_array(img_path, loader)
        if array is None:
            continue

        orig_anns = id_to_anns.get(img_id, [])

        # Collect all tiles first; decide which to keep without running backbone.
        all_tiles = list(tile_image(array, native_tile_size, tile_overlap))

        if orig_anns:
            # Positive image: keep tiles with ≥min_area_fraction of any annotation.
            selected = []
            for tile, x0, y0 in all_tiles:
                ts = native_tile_size
                qualifying = [
                    ann for ann in orig_anns
                    if _ann_area_in_tile(ann, x0, y0, ts) >= min_area_fraction
                ]
                if qualifying:
                    selected.append((tile, x0, y0, qualifying))
        else:
            # Negative image: a few random tiles for domain adaptation.
            chosen = rng.sample(all_tiles, k=min(neg_tiles_per_image, len(all_tiles)))
            selected = [(tile, x0, y0, []) for tile, x0, y0 in chosen]

        tile_idx = 0
        for tile, x0, y0, qualifying_anns in selected:
            tw, th = tile.shape[1], tile.shape[0]

            # Shift qualifying annotation OBBs to tile-local coordinates.
            tile_anns = []
            for ann in qualifying_anns:
                obb = ann.get("obb")
                if not obb:
                    continue
                cx_full = float(obb["cx"] if isinstance(obb, dict) else obb[0])
                cy_full = float(obb["cy"] if isinstance(obb, dict) else obb[1])
                local = dict(ann)
                raw_obb = local["obb"]
                if isinstance(raw_obb, dict):
                    local["obb"] = {**raw_obb, "cx": cx_full - x0, "cy": cy_full - y0}
                else:
                    lo = list(raw_obb); lo[0] -= x0; lo[1] -= y0
                    local["obb"] = lo
                tile_anns.append(local)

            canvas, scale, pad_x, pad_y = _letterbox_array(tile, image_size)
            img_tensor = (torch.from_numpy(canvas.astype(np.float32) / 255.0)
                          .permute(2, 0, 1).unsqueeze(0).to(device))

            with torch.no_grad():
                features = model.extract_features(
                    imagenet_normalize(img_tensor)
                ).cpu().to(torch.float16).squeeze(0)

            heatmap, geometry = _build_tile_targets(
                tile_anns, tw, th, image_size, scale, pad_x, pad_y
            )

            sample_id = img_id * 10000 + tile_idx
            rel_path  = Path("features") / f"{sample_id}.pt"
            torch.save(
                {
                    "features":      features,
                    "heatmap":       heatmap.to(torch.float16),
                    "center_heatmap": torch.zeros_like(heatmap, dtype=torch.float16),
                    "box_target":    torch.zeros((6, features.shape[1], features.shape[2]),
                                                 dtype=torch.float16),
                    "box_mask":      torch.zeros((1, features.shape[1], features.shape[2]),
                                                 dtype=torch.float16),
                    "geometry":      geometry.to(torch.float16),
                    "image_id":      sample_id,
                    "orig_size":     [th, tw],
                    "letterbox":     [scale, pad_x, pad_y],
                    "file_name":     str(img_meta["file_name"]),
                    "tile_origin":   [x0, y0],
                    "orig_image_id": img_id,
                },
                feature_dir.parent / rel_path,
            )
            manifest.append({
                "image_id":      sample_id,
                "path":          str(rel_path),
                "file_name":     str(img_meta["file_name"]),
                "tile_origin":   [x0, y0],
                "orig_image_id": img_id,
            })
            tile_idx += 1

        logger.info("tiled img_id=%d  tiles=%d  anns=%d  total_cached=%d",
                    img_id, tile_idx, len(orig_anns), len(manifest))

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weights", default=None,
                        help="Backbone checkpoint. Defaults to the canonical weight file "
                             "for the selected backbone/model-size.")
    parser.add_argument("--backbone", choices=["vit", "convnext"], default="vit",
                        help="Feature encoder family (default: vit)")
    parser.add_argument("--model-size", choices=["small", "base", "large"], default="small",
                        help="Backbone size (default: small). "
                             "For vit: small=ViT-S/16, base=ViT-B/16. "
                             "For convnext: small=ConvNeXt-S.")
    parser.add_argument("--convnext-stage", type=int, default=3, choices=[0, 1, 2, 3],
                        help="ConvNeXt stage whose output is used as the feature map "
                             "(0-3, default 3 = full backbone at stride 32, 768 ch). "
                             "Stage 2 gives stride 16, 384 ch — same as ViT-S/16.")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--native-tile-size", type=int, default=0,
                        help="Tile the image into overlapping N×N px crops before "
                             "caching (0 = full-image letterbox, the old default). "
                             "For Atwood 6248 px images use 400 (matching the OBB "
                             "training data scale). Only tiles with ≥25 %% of any "
                             "annotation bbox visible are cached (same gate as "
                             "build_tiled_brentimages_json.py). "
                             "Models trained on tiled caches MUST be evaluated with "
                             "--tiled in evaluate_dinov3_heatmap.py.")
    parser.add_argument("--tile-overlap", type=float, default=0.5,
                        help="Fractional overlap between tiles (default 0.5).")
    parser.add_argument("--min-area-fraction", type=float, default=0.25,
                        help="Min fraction of annotation bbox area that must be "
                             "visible in a tile to include it (default 0.25).")
    parser.add_argument("--neg-tiles-per-image", type=int, default=2,
                        help="Random background tiles to cache per unannotated "
                             "image for domain adaptation (default 2).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = get_device()
    out_dir = Path(args.output_dir)
    feature_dir = out_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    ds = StreakHeatmapDataset(args.annotations, image_size=args.image_size, max_samples=args.max_samples)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_heatmap_batch,
    )
    _DEFAULT_WEIGHTS = {
        ("vit", "small"): "weights/dinov3_vits16_lvd1689m.pth",
        ("vit", "base"): "weights/dinov3_vitb16_lvd1689m.pth",
        ("vit", "large"): "weights/dinov3_vitl16_lvd1689m.pth",
        ("convnext", "small"): "weights/dinov3_convnext_small_pretrain_lvd1689m.pth",
    }
    weights = args.weights or _DEFAULT_WEIGHTS.get((args.backbone, args.model_size))
    if weights is None:
        raise ValueError(
            f"No default weight path for backbone={args.backbone}, model_size={args.model_size}. "
            "Pass --weights explicitly."
        )

    if args.backbone == "convnext":
        model = ConvNeXtStreakHeatmap(
            model_size=args.model_size,
            weights=weights,
            extract_stage=args.convnext_stage,
        ).to(device)
    else:
        model = DINOv3StreakHeatmap(model_size=args.model_size, weights=weights).to(device)
    model.eval()

    # --- Tiled path ---
    if args.native_tile_size > 0:
        logger.info(
            "Tiled caching: native_tile_size=%d overlap=%.2f  "
            "(models trained this way MUST be evaluated with --tiled)",
            args.native_tile_size, args.tile_overlap,
        )
        coco_data = json.loads(Path(args.annotations).read_text())
        manifest = _cache_tiled(
            coco_data=coco_data,
            model=model,
            device=device,
            image_size=args.image_size,
            feature_dir=feature_dir,
            native_tile_size=args.native_tile_size,
            tile_overlap=args.tile_overlap,
            max_samples=args.max_samples,
            ann_dir=Path(args.annotations).resolve().parent,
            min_area_fraction=args.min_area_fraction,
            neg_tiles_per_image=args.neg_tiles_per_image,
        )
        metadata = {
            "annotations":    args.annotations,
            "weights":        weights,
            "backbone":       args.backbone,
            "model_size":     args.model_size,
            "convnext_stage": args.convnext_stage if args.backbone == "convnext" else None,
            "image_size":     args.image_size,
            "native_tile_size": args.native_tile_size,
            "tile_overlap":   args.tile_overlap,
            "n_samples":      len(manifest),
            "manifest":       manifest,
        }
        (out_dir / "manifest.json").write_text(json.dumps(metadata, indent=2))
        logger.info("wrote tiled cache manifest: %s (%d tiles)", out_dir / "manifest.json", len(manifest))
        return 0
    # --- End tiled path ---

    manifest: list[dict] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            images = imagenet_normalize(batch["image"].to(device))
            features = model.extract_features(images).cpu().to(torch.float16)
            heatmaps = batch["heatmap"].cpu().to(torch.float16)
            center_heatmaps = batch["center_heatmap"].cpu().to(torch.float16)
            box_targets = batch["box_target"].cpu().to(torch.float16)
            box_masks = batch["box_mask"].cpu().to(torch.float16)
            geometries = batch["geometry"].cpu().to(torch.float16)
            image_ids = batch["image_id"].cpu().tolist()
            orig_sizes = batch["orig_size"].cpu().tolist()
            letterboxes = batch["letterbox"].cpu().tolist()
            file_names = batch["file_name"]

            for i, image_id in enumerate(image_ids):
                rel_path = Path("features") / f"{int(image_id)}.pt"
                torch.save(
                    {
                        "features": features[i],
                        "heatmap": heatmaps[i],
                        "center_heatmap": center_heatmaps[i],
                        "box_target": box_targets[i],
                        "box_mask": box_masks[i],
                        "geometry": geometries[i],
                        "image_id": int(image_id),
                        "orig_size": orig_sizes[i],
                        "letterbox": letterboxes[i],
                        "file_name": file_names[i],
                    },
                    out_dir / rel_path,
                )
                manifest.append({
                    "image_id": int(image_id),
                    "path": str(rel_path),
                    "file_name": file_names[i],
                })
            logger.info("cached batch %d/%d (%d samples)", batch_idx, len(loader), len(manifest))

    metadata = {
        "annotations": args.annotations,
        "weights": weights,
        "backbone": args.backbone,
        "model_size": args.model_size,
        "convnext_stage": args.convnext_stage if args.backbone == "convnext" else None,
        "image_size": args.image_size,
        "n_samples": len(manifest),
        "manifest": manifest,
    }
    (out_dir / "manifest.json").write_text(json.dumps(metadata, indent=2))
    logger.info("wrote cache manifest: %s", out_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
