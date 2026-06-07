"""Evaluate a plain PyTorch DINOv3 direct center/box checkpoint."""

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
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.metrics import evaluate
from inference.device import get_device
from models.plain_dinov3.streak_heatmap import DINOv3StreakHeatmap, decode_box, imagenet_normalize
from training.dinov3_heatmap_dataset import StreakHeatmapDataset, collate_heatmap_batch
from training.train_dinov3_box_cached import CenterBoxHead
from scripts.evaluate_dinov3_heatmap import coco_ground_truth

logger = logging.getLogger(__name__)


def center_box_to_detections(
    center_probs: np.ndarray,
    box_map: np.ndarray,
    image_id: int,
    threshold: float,
    patch_size: int,
    image_size: int,
    letterbox: tuple[float, float, float],
    max_detections: int,
    nms_kernel: int,
) -> list[dict[str, Any]]:
    """Convert center/box grid predictions to ARGUS detection dictionaries."""
    local_max = center_probs == ndimage.maximum_filter(center_probs, size=nms_kernel, mode="constant")
    ys, xs = np.nonzero((center_probs >= threshold) & local_max)
    if len(xs) == 0:
        return []

    order = np.argsort(center_probs[ys, xs])[::-1][:max_detections]
    scale, pad_x, pad_y = letterbox
    detections: list[dict[str, Any]] = []
    for idx in order:
        y = int(ys[idx])
        x = int(xs[idx])
        score = float(center_probs[y, x])
        dx, dy, cos2, sin2, length_norm, width_norm = (float(v) for v in box_map[:, y, x])
        cx = (x + 0.5 + dx) * patch_size
        cy = (y + 0.5 + dy) * patch_size
        angle = (0.5 * math.degrees(math.atan2(sin2, cos2))) % 180.0
        length = max(length_norm * image_size, patch_size)
        width = max(width_norm * image_size, 1e-3)

        obb = {
            "cx": (cx - pad_x) / scale,
            "cy": (cy - pad_y) / scale,
            "w": length / scale,
            "h": width / scale,
            "angle_deg": angle,
        }
        detections.append({
            "image_id": image_id,
            "confidence": score,
            "obb": obb,
            "streak_length_px": max(float(obb["w"]), float(obb["h"])),
        })
    return detections


def main() -> int:
    """Run direct center/box evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/annotations/test.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", default=None, help="Override DINOv3 backbone weights from checkpoint metadata")
    parser.add_argument("--output", default="results/plain_dinov3_box/metrics.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--norm-mode", choices=["autostretch", "zscore", "zscale"],
                        default="autostretch",
                        help="Pixel normalisation for raw FITS/NPY tiles. Must match "
                             "the mode used during training.")
    parser.add_argument("--max-detections", type=int, default=20)
    parser.add_argument("--nms-kernel", type=int, default=7)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ckpt.get("head_type") != "center_box":
        raise ValueError(f"Expected a center_box checkpoint, got {ckpt.get('head_type')!r}")

    train_meta = ckpt["train_cache_metadata"]
    weights = args.weights or train_meta.get("weights", "weights/dinov3_vitb16_lvd1689m.pth")
    model_size = train_meta.get("model_size", "base")
    image_size = int(train_meta.get("image_size", 512))
    patch_size = 16

    model = DINOv3StreakHeatmap(model_size=model_size, weights=weights, out_channels=7).to(device)
    hidden = int(ckpt.get("args", {}).get("hidden_channels", 256))
    cached_head = CenterBoxHead(int(ckpt["in_channels"]), hidden)
    cached_head.load_state_dict(ckpt["head"])
    model.head = cached_head.to(device)
    model.eval()

    ds = StreakHeatmapDataset(args.annotations, image_size=image_size, max_samples=args.max_samples, norm_mode=args.norm_mode)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_heatmap_batch)

    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            images = imagenet_normalize(batch["image"].to(device))
            output = model(images)
            center_probs = torch.sigmoid(output[:, :1]).cpu().numpy()
            box_map = decode_box(output[:, 1:7]).cpu().numpy()
            image_ids = batch["image_id"].cpu().numpy().tolist()
            letterboxes = batch["letterbox"].cpu().numpy().tolist()
            for idx, (prob, image_id, letterbox) in enumerate(zip(center_probs[:, 0], image_ids, letterboxes)):
                predictions.extend(
                    center_box_to_detections(
                        prob,
                        box_map[idx],
                        int(image_id),
                        args.threshold,
                        patch_size,
                        image_size,
                        (float(letterbox[0]), float(letterbox[1]), float(letterbox[2])),
                        args.max_detections,
                        args.nms_kernel,
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

    # Resolve integer COCO image IDs → file_name strings for the comparison
    # script (compare_heatmap_centerline_to_obb.py / eval/line_metrics.py),
    # which keys ground truth by filename rather than integer ID.
    coco = json.loads(Path(args.annotations).read_text())
    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}
    filename_predictions = [
        {**pred, "image_id": id_to_filename.get(int(pred["image_id"]), str(pred["image_id"]))}
        for pred in predictions
    ]
    (out_path.parent / "predictions.json").write_text(json.dumps(filename_predictions, indent=2))
    logger.info("wrote %s (%d predictions)", out_path, len(predictions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
