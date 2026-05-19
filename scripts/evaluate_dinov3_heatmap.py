"""Evaluate a plain PyTorch DINOv3 heatmap checkpoint as ARGUS detections."""

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
from scipy import ndimage
from skimage.transform import radon
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.metrics import evaluate
from inference.device import get_device
from inference.fits_loader import FITSLoader
from models.plain_dinov3.streak_heatmap import DINOv3StreakHeatmap, decode_geometry, imagenet_normalize
from training.dinov3_heatmap_dataset import StreakHeatmapDataset, collate_heatmap_batch
from training.train_dinov3_heatmap_cached import HeatmapHead

logger = logging.getLogger(__name__)


def _component_to_obb(
    mask: np.ndarray,
    score_map: np.ndarray,
    patch_size: int,
    geometry_map: np.ndarray | None = None,
    image_size: int | None = None,
) -> dict[str, Any] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return None
    pts = np.column_stack([(xs + 0.5) * patch_size, (ys + 0.5) * patch_size]).astype(np.float32)
    center = pts.mean(axis=0)
    cov = np.cov((pts - center).T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    major = vecs[:, order[0]]
    minor = vecs[:, order[1]]
    rel = pts - center
    along = rel @ major
    across = rel @ minor
    length = max(float(along.max() - along.min()) + patch_size, patch_size)
    width = max(float(across.max() - across.min()) + patch_size, patch_size)
    angle = math.degrees(math.atan2(float(major[1]), float(major[0]))) % 180.0
    if geometry_map is not None and image_size is not None:
        geom_vals = geometry_map[:, mask]
        if geom_vals.shape[1] > 0:
            cos2 = float(geom_vals[0].mean())
            sin2 = float(geom_vals[1].mean())
            if abs(cos2) + abs(sin2) > 1e-3:
                angle = (0.5 * math.degrees(math.atan2(sin2, cos2))) % 180.0
            length = max(float(geom_vals[2].mean()) * image_size, patch_size)
            width = max(float(geom_vals[3].mean()) * image_size, patch_size)
    confidence = float(score_map[mask].mean())
    return {
        "confidence": confidence,
        "obb": {
            "cx": float(center[0]),
            "cy": float(center[1]),
            "w": length,
            "h": width,
            "angle_deg": angle,
        },
        "streak_length_px": length,
    }


def heatmap_to_detections(
    probs: np.ndarray,
    image_id: int,
    threshold: float,
    patch_size: int,
    min_pixels: int,
    image_size: int,
    letterbox: tuple[float, float, float],
    geometry_map: np.ndarray | None = None,
    image_array: np.ndarray | None = None,
    refine_geometry: bool = True,
) -> list[dict[str, Any]]:
    """Convert one heatmap to ARGUS-style detection dictionaries."""
    binary = probs >= threshold
    labels, n_labels = ndimage.label(binary)
    detections: list[dict[str, Any]] = []
    for label_id in range(1, n_labels + 1):
        mask = labels == label_id
        if int(mask.sum()) < min_pixels:
            continue
        det = _component_to_obb(mask, probs, patch_size, geometry_map=geometry_map, image_size=image_size)
        if det is None:
            continue
        scale, pad_x, pad_y = letterbox
        obb = det["obb"]
        obb["cx"] = (obb["cx"] - pad_x) / scale
        obb["cy"] = (obb["cy"] - pad_y) / scale
        obb["w"] /= scale
        obb["h"] /= scale
        det["streak_length_px"] = max(float(obb["w"]), float(obb["h"]))
        if refine_geometry and image_array is not None:
            _refine_detection_geometry(det, image_array)
        det["image_id"] = image_id
        detections.append(det)
    return detections


def _refine_detection_geometry(det: dict[str, Any], image_array: np.ndarray) -> None:
    """Refine component OBB angle and length using original-image pixels."""
    obb = det["obb"]
    h, w = image_array.shape[:2]
    cx = float(np.clip(obb["cx"], 0, max(w - 1, 0)))
    cy = float(np.clip(obb["cy"], 0, max(h - 1, 0)))
    half = max(float(obb["w"]), float(obb["h"]), 64.0) / 2.0 + 24.0
    half = min(half, 384.0)
    x1 = int(max(0, math.floor(cx - half)))
    x2 = int(min(w, math.ceil(cx + half)))
    y1 = int(max(0, math.floor(cy - half)))
    y2 = int(min(h, math.ceil(cy + half)))
    crop = image_array[y1:y2, x1:x2]
    if crop.size == 0 or min(crop.shape[:2]) < 8:
        return
    gray = crop[..., 0].astype(np.float32) if crop.ndim == 3 else crop.astype(np.float32)
    gray = gray - float(np.median(gray))
    gray[gray < 0] = 0
    if float(gray.max()) <= 0:
        return
    step = 1
    if max(gray.shape[:2]) > 512:
        step = int(math.ceil(max(gray.shape[:2]) / 512))
        gray = gray[::step, ::step]

    seed = float(obb.get("angle_deg", 0.0))
    theta = (90.0 - np.arange(seed - 25.0, seed + 25.0, 2.0)) % 180.0
    sinogram = radon(gray, theta=theta, circle=False)
    variances = sinogram.var(axis=0)
    best_theta = float(theta[int(np.argmax(variances))])
    angle = (90.0 - best_theta) % 180.0
    obb["angle_deg"] = angle

    ux, uy = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    yy, xx = np.nonzero(gray > max(float(gray.mean() + gray.std()), float(np.percentile(gray, 95))))
    if len(xx) >= 2:
        pts_x = xx.astype(np.float32) * step + x1
        pts_y = yy.astype(np.float32) * step + y1
        along = (pts_x - cx) * ux + (pts_y - cy) * uy
        across = np.abs(-(pts_x - cx) * uy + (pts_y - cy) * ux)
        keep = across <= max(float(obb["h"]), 8.0)
        if int(keep.sum()) >= 2:
            along_kept = along[keep]
            obb["w"] = max(float(along_kept.max() - along_kept.min()), float(obb["w"]))
            obb["h"] = max(float(np.percentile(across[keep], 90) * 2.0), 4.0)
            det["streak_length_px"] = float(obb["w"])


def coco_ground_truth(annotation_file: Path) -> list[dict[str, Any]]:
    """Read COCO annotations into ARGUS metric ground truth format."""
    coco = json.loads(annotation_file.read_text())
    id_to_file = {int(img["id"]): img["file_name"] for img in coco.get("images", [])}
    _ = id_to_file
    gts: list[dict[str, Any]] = []
    for ann in coco.get("annotations", []):
        image_id = int(ann["image_id"])
        obb = ann.get("obb")
        if isinstance(obb, list):
            obb_dict = {"cx": obb[0], "cy": obb[1], "w": obb[2], "h": obb[3], "angle_deg": obb[4]}
        elif isinstance(obb, dict):
            obb_dict = obb
        else:
            x, y, w, h = ann["bbox"]
            obb_dict = {"cx": x + w / 2, "cy": y + h / 2, "w": w, "h": h, "angle_deg": 0.0}
        gts.append({
            "image_id": image_id,
            "obb": obb_dict,
            "streak_length_px": max(float(obb_dict["w"]), float(obb_dict["h"])),
        })
    return gts


def _load_eval_image(path: Path, loader: FITSLoader) -> np.ndarray | None:
    suffix = path.suffix.lower()
    try:
        if suffix in {".fits", ".fit", ".fts"}:
            return np.asarray(loader.load(path)["array"], dtype=np.uint8)
        from PIL import Image
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        logger.warning("Could not reload %s for geometry refinement: %s", path, exc)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/annotations/test.json")
    parser.add_argument("--checkpoint", default="weights/run_plain_dinov3_heatmap/best.pt")
    parser.add_argument("--weights", default=None, help="Override DINOv3 backbone weights from checkpoint args")
    parser.add_argument("--output", default="results/plain_dinov3_heatmap/metrics.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-pixels", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-refine-geometry", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = ckpt.get("args", {})
    is_cached_head = "head" in ckpt and "model" not in ckpt
    if is_cached_head:
        train_meta = ckpt["train_cache_metadata"]
        weights = args.weights or train_meta.get("weights", "weights/dinov3_vitb16_lvd1689m.pth")
        model_size = train_meta.get("model_size", "base")
        image_size = int(train_meta.get("image_size", 512))
    else:
        weights = args.weights or train_args.get("weights", "weights/dinov3_vitb16_lvd1689m.pth")
        model_size = train_args.get("model_size", "base")
        image_size = int(train_args.get("image_size", 512))
    patch_size = 16

    model = DINOv3StreakHeatmap(model_size=model_size, weights=weights).to(device)
    if is_cached_head:
        hidden = int(ckpt.get("args", {}).get("hidden_channels", 256))
        cached_head = HeatmapHead(int(ckpt["in_channels"]), hidden)
        cached_head.load_state_dict(ckpt["head"])
        model.head = cached_head.net.to(device)
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()

    ds = StreakHeatmapDataset(args.annotations, image_size=image_size, max_samples=args.max_samples)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_heatmap_batch)
    image_loader = FITSLoader()

    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            images = imagenet_normalize(batch["image"].to(device))
            output = model(images)
            logits = output[:, :1]
            probs = torch.sigmoid(logits).cpu().numpy()
            geometry = None
            if output.shape[1] >= 5:
                geometry = decode_geometry(output[:, 1:5]).cpu().numpy()
            image_ids = batch["image_id"].cpu().numpy().tolist()
            letterboxes = batch["letterbox"].cpu().numpy().tolist()
            file_names = batch["file_name"]
            for idx, (prob, image_id, letterbox, file_name) in enumerate(zip(probs[:, 0], image_ids, letterboxes, file_names)):
                image_path = ds._resolve_image_path(str(file_name))
                image_array = None if args.no_refine_geometry else _load_eval_image(image_path, image_loader)
                predictions.extend(
                    heatmap_to_detections(
                        prob,
                        int(image_id),
                        args.threshold,
                        patch_size,
                        args.min_pixels,
                        image_size,
                        (float(letterbox[0]), float(letterbox[1]), float(letterbox[2])),
                        geometry_map=None if geometry is None else geometry[idx],
                        image_array=image_array,
                        refine_geometry=not args.no_refine_geometry,
                    )
                )

    ground_truth = coco_ground_truth(Path(args.annotations))
    if args.max_samples:
        allowed_ids = {int(meta["id"]) for meta in ds.images}
        ground_truth = [gt for gt in ground_truth if int(gt["image_id"]) in allowed_ids]

    metrics = evaluate(predictions, ground_truth)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": metrics, "n_predictions": len(predictions)}
    out_path.write_text(json.dumps(payload, indent=2))
    (out_path.parent / "predictions.json").write_text(json.dumps(predictions, indent=2))
    logger.info("wrote %s (%d predictions)", out_path, len(predictions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
