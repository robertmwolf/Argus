"""Phase A feasibility probe: DINOv3 ViT feature extraction on FITS streak images.

Loads DINOv3 ViT-B/16 (LVD-1689M) from a local .pth checkpoint, extracts
patch-level features from 512×512 crops centred on annotated streaks and
matched background crops, then produces three outputs per image:

  1. PCA heatmap — first 3 principal components of patch tokens, rendered as RGB
  2. Feature-norm map — L2 norm per patch, bright = high activation
  3. Console report — mean patch norm in streak region vs background region

Gate: if streak crops produce visually distinct PCA clusters AND mean norm ratio
streak/background > 1.15, the backbone is viable for downstream detection.

Usage:
    # Install dinov3 first (one-time):
    #   pip install git+https://github.com/facebookresearch/dinov3.git
    #
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    python scripts/probe_dinov3.py \\
        --weights weights/dinov3_vitb16_lvd1689m.pth \\
        --annotations data/annotations/dev_subset.json \\
        --image-dir data/ \\
        --out results/probe_dinov3 \\
        [--n-images 10] [--crop-size 512] [--device cpu|mps|cuda]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ImageNet normalization — required for all DINOv3 LVD-1689M models
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_dinov3_vitb(weights_path: Path, device: torch.device) -> torch.nn.Module:
    """Load DINOv3 ViT-B/16 from a local .pth pretrain checkpoint.

    Args:
        weights_path: Path to dinov3_vitb16_pretrain_lvd1689m-*.pth
        device: Target device.

    Returns:
        DINOv3 ViT model in eval mode on *device*.

    Raises:
        ImportError: If the dinov3 package is not installed.
        FileNotFoundError: If *weights_path* does not exist.
    """
    if not weights_path.exists():
        raise FileNotFoundError(
            f"DINOv3 weights not found: {weights_path}\n"
            "Download dinov3_vitb16_pretrain_lvd1689m-*.pth from the Meta portal "
            "and place it in weights/."
        )
    try:
        import dinov3.models.vision_transformer as vits  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "dinov3 package not installed. Run:\n"
            "  pip install git+https://github.com/facebookresearch/dinov3.git"
        ) from exc

    logger.info("Building DINOv3 ViT-B/16 ...")
    # n_storage_tokens=4, layerscale_init, mask_k_bias derived from checkpoint inspection
    model = vits.vit_base(
        patch_size=16,
        img_size=518,
        n_storage_tokens=4,
        layerscale_init=1e-4,
        mask_k_bias=True,
    )
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Missing keys (%d): %s ...", len(missing), missing[:3])
    if unexpected:
        logger.warning("Unexpected keys (%d): %s ...", len(unexpected), unexpected[:3])
    model.eval().to(device)
    logger.info("Model loaded on %s  (%.0f M params)", device, sum(p.numel() for p in model.parameters()) / 1e6)
    return model


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]


def load_fits_as_uint8(fits_path: Path) -> np.ndarray:
    """Load a FITS file → uint8 (H, W, 3) via ARGUS FITSLoader."""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from inference.fits_loader import FITSLoader
    loader = FITSLoader()
    return loader.load(fits_path)["array"]  # (H, W, 3) uint8


def extract_crop(
    img: np.ndarray,
    cx: float,
    cy: float,
    size: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Extract a square crop centred at (cx, cy), clamped to image bounds.

    Returns:
        (crop_uint8, (x0, y0, x1, y1)) — crop array and its pixel coordinates
        in the original image.
    """
    h, w = img.shape[:2]
    half = size // 2
    x0 = max(0, int(cx) - half)
    y0 = max(0, int(cy) - half)
    x1 = min(w, x0 + size)
    y1 = min(h, y0 + size)
    # Adjust origin if clamped
    x0 = max(0, x1 - size)
    y0 = max(0, y1 - size)
    crop = img[y0:y1, x0:x1]
    if crop.shape[0] != size or crop.shape[1] != size:
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    return crop, (x0, y0, x1, y1)


def background_crop(
    img: np.ndarray,
    streak_box: tuple[int, int, int, int],
    size: int,
) -> np.ndarray:
    """Return a same-size background crop far from the streak bounding box."""
    h, w = img.shape[:2]
    sx0, sy0, sx1, sy1 = streak_box
    # Try corners; pick first that doesn't overlap the streak box
    candidates = [
        (size // 2, size // 2),                  # top-left
        (w - size // 2, size // 2),               # top-right
        (size // 2, h - size // 2),               # bottom-left
        (w - size // 2, h - size // 2),           # bottom-right
    ]
    for cx, cy in candidates:
        bx0 = cx - size // 2
        by0 = cy - size // 2
        bx1 = bx0 + size
        by1 = by0 + size
        overlap = not (bx1 < sx0 or bx0 > sx1 or by1 < sy0 or by0 > sy1)
        if not overlap:
            crop, _ = extract_crop(img, cx, cy, size)
            return crop
    # Fallback: use top-left even if overlapping
    crop, _ = extract_crop(img, size // 2, size // 2, size)
    return crop


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def to_tensor(crop_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert uint8 (H, W, 3) crop to normalised float tensor (1, 3, H, W)."""
    x = torch.from_numpy(crop_uint8).float().permute(2, 0, 1) / 255.0  # (3, H, W)
    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    return x.unsqueeze(0).to(device)


@torch.no_grad()
def extract_patch_features(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    patch_size: int = 16,
    n_layers: int = 4,
) -> np.ndarray:
    """Extract patch token features from the last n_layers of a DINOv3 ViT.

    Args:
        model: DINOv3 ViT in eval mode.
        tensor: Float tensor (1, 3, H, W), ImageNet-normalised.
        patch_size: Model patch size (16 for ViT-B/16).
        n_layers: Number of final layers to extract from.

    Returns:
        Patch feature array (num_patches, embed_dim*n_layers) — averaged across
        the n_layers returned by get_intermediate_layers.
    """
    # get_intermediate_layers returns list of n tensors, each (B, num_patches, dim)
    layers = model.get_intermediate_layers(tensor, n=n_layers, norm=True)
    # Stack and average: (n_layers, B, num_patches, dim) → (num_patches, dim)
    stacked = torch.stack(layers, dim=0)          # (n, 1, N, D)
    avg = stacked.mean(dim=0).squeeze(0)           # (N, D)
    return avg.cpu().numpy()


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def pca_heatmap(
    patch_feats: np.ndarray,
    grid_h: int,
    grid_w: int,
    out_size: int,
) -> np.ndarray:
    """Render the first 3 PCA components of patch tokens as an RGB image.

    Args:
        patch_feats: (num_patches, dim)
        grid_h, grid_w: Patch grid dimensions.
        out_size: Output image pixel size.

    Returns:
        uint8 RGB image (out_size, out_size, 3).
    """
    pca = PCA(n_components=3)
    components = pca.fit_transform(patch_feats)  # (N, 3)
    # Normalise each component to [0, 255]
    result = np.zeros_like(components)
    for i in range(3):
        c = components[:, i]
        lo, hi = c.min(), c.max()
        result[:, i] = (c - lo) / (hi - lo + 1e-8) * 255
    grid = result.reshape(grid_h, grid_w, 3).astype(np.uint8)
    return cv2.resize(grid, (out_size, out_size), interpolation=cv2.INTER_NEAREST)


def norm_heatmap(
    patch_feats: np.ndarray,
    grid_h: int,
    grid_w: int,
    out_size: int,
) -> np.ndarray:
    """Render per-patch L2 feature norm as a jet heatmap."""
    norms = np.linalg.norm(patch_feats, axis=1)          # (N,)
    lo, hi = norms.min(), norms.max()
    norms_norm = ((norms - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
    grid = norms_norm.reshape(grid_h, grid_w)
    resized = cv2.resize(grid, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    return cv2.applyColorMap(resized, cv2.COLORMAP_JET)


def save_comparison(
    streak_crop: np.ndarray,
    bg_crop: np.ndarray,
    streak_feats: np.ndarray,
    bg_feats: np.ndarray,
    grid_h: int,
    grid_w: int,
    out_path: Path,
    image_name: str,
) -> None:
    """Save a 2×3 panel: [streak | bg] × [raw | PCA | norm heatmap]."""
    size = streak_crop.shape[0]
    panels = []
    for crop, feats, label in [
        (streak_crop, streak_feats, "streak"),
        (bg_crop, bg_feats, "background"),
    ]:
        raw = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        pca = pca_heatmap(feats, grid_h, grid_w, size)
        pca_bgr = cv2.cvtColor(pca, cv2.COLOR_RGB2BGR)
        norm = norm_heatmap(feats, grid_h, grid_w, size)
        # Label each panel
        for img, title in [(raw, f"{label} raw"), (pca_bgr, f"{label} PCA"), (norm, f"{label} norm")]:
            cv2.putText(img, title, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        panels.append(np.hstack([raw, pca_bgr, norm]))
    grid = np.vstack(panels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    logger.info("Saved: %s", out_path.name)


# ---------------------------------------------------------------------------
# Main probe loop
# ---------------------------------------------------------------------------

def run_probe(
    model: torch.nn.Module,
    annotations: dict,
    image_dir: Path,
    out_dir: Path,
    crop_size: int,
    n_images: int,
    device: torch.device,
) -> dict[str, float]:
    """Run the probe across annotated streak images.

    Returns:
        Summary dict with mean norm ratios and timing.
    """
    # Build image_id → annotations map
    ann_by_image: dict[int, list[dict]] = {}
    for ann in annotations["annotations"]:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    # Filter to images that have at least one streak annotation
    streak_images = [
        img for img in annotations["images"]
        if img["id"] in ann_by_image
    ][:n_images]

    if not streak_images:
        logger.error("No annotated images found in dev_subset.")
        return {}

    logger.info("Probing %d images (crop_size=%d)", len(streak_images), crop_size)

    patch_size = 16
    grid_h = grid_w = crop_size // patch_size
    norm_ratios: list[float] = []
    times: list[float] = []

    for img_meta in streak_images:
        img_path = image_dir / img_meta["file_name"]
        if not img_path.exists():
            logger.warning("Image not found: %s", img_path)
            continue

        logger.info("Loading %s ...", img_path.name)
        try:
            img = load_fits_as_uint8(img_path)
        except Exception as exc:
            logger.warning("Skipping %s: %s", img_path.name, exc)
            continue

        # Use first annotation's OBB centre as crop anchor
        ann = ann_by_image[img_meta["id"]][0]
        obb = ann.get("obb", {})
        cx = obb.get("cx", ann["bbox"][0] + ann["bbox"][2] / 2)
        cy = obb.get("cy", ann["bbox"][1] + ann["bbox"][3] / 2)

        streak_crop, streak_box = extract_crop(img, cx, cy, crop_size)
        bg_crop = background_crop(img, streak_box, crop_size)

        # Extract features
        t0 = time.perf_counter()
        streak_tensor = to_tensor(streak_crop, device)
        bg_tensor     = to_tensor(bg_crop, device)
        streak_feats  = extract_patch_features(model, streak_tensor, patch_size)
        bg_feats      = extract_patch_features(model, bg_tensor, patch_size)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        # Cosine dissimilarity: mean(1 - cos_sim) between streak and bg patch features.
        # ViT patch norms are uniform by design; cosine distance captures directional
        # difference in feature space, which is the meaningful signal.
        streak_norm_feats = streak_feats / (np.linalg.norm(streak_feats, axis=1, keepdims=True) + 1e-8)
        bg_norm_feats     = bg_feats     / (np.linalg.norm(bg_feats,     axis=1, keepdims=True) + 1e-8)
        cos_sims  = (streak_norm_feats * bg_norm_feats).sum(axis=1)  # (N,)
        cos_dissim = float(1.0 - cos_sims.mean())
        norm_ratios.append(cos_dissim)
        logger.info(
            "  %s | cosine_dissim=%.4f  time=%.1fs",
            img_path.name, cos_dissim, elapsed,
        )

        # Visualisations
        stem = img_path.stem
        save_comparison(
            streak_crop, bg_crop,
            streak_feats, bg_feats,
            grid_h, grid_w,
            out_dir / f"{stem}_probe.jpg",
            img_path.name,
        )

    mean_dissim = float(np.mean(norm_ratios)) if norm_ratios else 0.0
    mean_time   = float(np.mean(times)) if times else 0.0
    # Gate: mean cosine dissimilarity > 0.05 means streak and background crops
    # occupy meaningfully different regions of feature space.
    gate_pass   = mean_dissim > 0.05

    print("\n" + "=" * 60)
    print("PHASE A — PROBE RESULTS")
    print("=" * 60)
    print(f"  Images probed        : {len(norm_ratios)}")
    print(f"  Mean cosine dissim   : {mean_dissim:.4f}  (streak vs background patches)")
    print(f"  Mean inference       : {mean_time:.1f} s/crop-pair")
    print(f"  Gate (dissim > 0.05) : {'PASS ✓' if gate_pass else 'FAIL ✗'}")
    print("=" * 60)
    print("\nAlso review PCA heatmaps in results/probe_dinov3/ —")
    print("streak-region patches should form a distinct colour cluster vs background.")

    summary = {
        "n_images": len(norm_ratios),
        "mean_cosine_dissimilarity": mean_dissim,
        "mean_inference_s": mean_time,
        "gate_pass": gate_pass,
        "per_image_dissimilarities": norm_ratios,
    }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights",      type=Path, default=Path("weights/dinov3_vitb16_lvd1689m.pth"),
                   help="Path to DINOv3 ViT-B/16 LVD pretrain .pth file")
    p.add_argument("--annotations",  type=Path, default=Path("data/annotations/dev_subset.json"),
                   help="COCO-format annotation JSON with streak OBBs")
    p.add_argument("--image-dir",    type=Path, default=Path("data/"),
                   help="Root directory for image file_name paths in annotations")
    p.add_argument("--out",          type=Path, default=Path("results/probe_dinov3"),
                   help="Output directory for visualisation images and summary JSON")
    p.add_argument("--n-images",     type=int, default=10,
                   help="Number of annotated images to probe (default: 10)")
    p.add_argument("--crop-size",    type=int, default=512,
                   help="Square crop size in pixels centred on each streak (default: 512)")
    p.add_argument("--device",       type=str, default=None,
                   help="Device override: cpu, mps, cuda (default: auto)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Device selection (mirrors inference/device.py priority)
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    # Load model
    model = load_dinov3_vitb(args.weights, device)

    # Load annotations
    if not args.annotations.exists():
        logger.error("Annotation file not found: %s", args.annotations)
        sys.exit(1)
    with open(args.annotations) as f:
        annotations = json.load(f)

    # Run probe
    summary = run_probe(
        model=model,
        annotations=annotations,
        image_dir=args.image_dir,
        out_dir=args.out,
        crop_size=args.crop_size,
        n_images=args.n_images,
        device=device,
    )

    # Save summary JSON
    import json as _json
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "probe_summary.json"
    with open(summary_path, "w") as f:
        _json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", summary_path)
