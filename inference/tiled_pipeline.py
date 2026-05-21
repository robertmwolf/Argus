"""Tiled inference for ARGUS satellite streak detection.

Large Frigate frames can contain very short streaks that disappear when the
whole image is resized before inference.  This module runs the existing ARGUS
DINO inference path on native-resolution square tiles, remaps tile detections
back into full-image coordinates, and merges cross-tile duplicates with NMS.

The tile_size and overlap parameters are explicit because the same values
should be used when Frigate images are later re-annotated into training tiles.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

logger = logging.getLogger(__name__)

Prediction = dict[str, Any]


def tile_image(
    img_array: "np.ndarray",
    tile_size: int,
    overlap: float,
) -> Iterator[tuple["np.ndarray", int, int]]:
    """Yield square, padded image tiles that cover the full image.

    Args:
        img_array: Image array with shape ``(H, W)`` or ``(H, W, C)``.
        tile_size: Square tile edge length in pixels.
        overlap: Fractional overlap between neighboring tiles in ``[0, 1)``.

    Yields:
        Tuples of ``(tile_array, x0, y0)`` where ``x0``/``y0`` are the tile's
        top-left coordinate in the original image coordinate system.

    Raises:
        ValueError: If tile parameters are invalid.
    """
    import numpy as np

    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in the range [0, 1)")
    if img_array.ndim not in {2, 3}:
        raise ValueError("img_array must have shape (H, W) or (H, W, C)")

    h_img, w_img = img_array.shape[:2]
    if h_img <= 0 or w_img <= 0:
        raise ValueError("img_array must be non-empty")

    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    x_starts = _tile_starts(w_img, tile_size, stride)
    y_starts = _tile_starts(h_img, tile_size, stride)

    padded_h = y_starts[-1] + tile_size
    padded_w = x_starts[-1] + tile_size
    pad_h = max(0, padded_h - h_img)
    pad_w = max(0, padded_w - w_img)
    pad_spec = ((0, pad_h), (0, pad_w))
    if img_array.ndim == 3:
        pad_spec = (*pad_spec, (0, 0))
    padded = np.pad(img_array, pad_spec, mode="edge")

    for y0 in y_starts:
        for x0 in x_starts:
            yield padded[y0:y0 + tile_size, x0:x0 + tile_size].copy(), x0, y0


def remap_predictions(preds: list[Prediction], x0: int, y0: int) -> list[Prediction]:
    """Add tile offsets to ``[x, y, w, h]`` prediction boxes.

    Args:
        preds: Predictions with COCO-style ``bbox`` values ``[x, y, w, h]``.
        x0: Tile X offset in the full image.
        y0: Tile Y offset in the full image.

    Returns:
        New prediction dictionaries in full-image coordinates.
    """
    remapped: list[Prediction] = []
    for pred in preds:
        x, y, w, h = [float(v) for v in pred["bbox"]]
        updated = dict(pred)
        updated["bbox"] = [x + float(x0), y + float(y0), w, h]
        remapped.append(updated)
    return remapped


def nms_predictions(
    preds: list[Prediction],
    iou_threshold: float = 0.4,
) -> list[Prediction]:
    """Apply cross-tile IoU NMS to COCO-style predictions.

    Args:
        preds: Predictions with keys ``bbox`` (``[x, y, w, h]``), ``score``,
            and ``category_id``.
        iou_threshold: IoU threshold above which lower-scored boxes are
            suppressed, independently per category.

    Returns:
        Predictions kept after NMS, sorted by descending score.
    """
    if not preds:
        return []
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in the range [0, 1]")

    try:
        keep_indices = _torchvision_nms(preds, iou_threshold)
    except Exception as exc:
        logger.debug("torchvision NMS unavailable; using NumPy fallback: %s", exc)
        keep_indices = _numpy_nms(preds, iou_threshold)

    return [preds[i] for i in keep_indices]


def run_tiled_inference(
    image_path: str | Path,
    model_config: str | Path,
    checkpoint: str | Path,
    tile_size: int = 400,
    overlap: float = 0.5,
    conf_threshold: float = 0.3,
) -> list[Prediction]:
    """Run DINO inference over native-resolution overlapping tiles.

    Args:
        image_path: FITS, PNG, or JPEG image path.
        model_config: MMDetection config path, or an ARGUS model size alias
            accepted by ``inference.pipeline._select_config``.
        checkpoint: Model checkpoint path.
        tile_size: Square tile edge length in pixels. Use the same value for
            later tiled Frigate training annotations.
        overlap: Fractional tile overlap in ``[0, 1)``. Use the same value for
            later tiled Frigate training annotations.
        conf_threshold: Minimum model score to keep before cross-tile NMS.

    Returns:
        Final predictions as dictionaries with ``bbox`` in ``[x, y, w, h]``
        full-image coordinates plus ``score`` and ``category_id``.
    """
    t0 = time.perf_counter()
    image_path = Path(image_path)
    checkpoint = Path(checkpoint)

    from inference.fits_loader import FITSLoader
    from inference.pipeline import _load_model, _run_inference, _select_config

    loaded = FITSLoader().load(image_path)
    img_array = loaded["array"]

    config_path = Path(model_config)
    if not config_path.exists():
        config_path = _select_config(str(model_config))

    from inference.device import get_device
    import torch

    device = get_device()
    inference_device = device
    if device.type == "mps" and not torch.cuda.is_available():
        inference_device = torch.device("cpu")
        logger.debug("MPS device detected; forcing tiled DINO inference to CPU")

    model = _load_model(config_path, checkpoint, inference_device)

    predictions: list[Prediction] = []
    tiles = list(tile_image(img_array, tile_size=tile_size, overlap=overlap))
    logger.info(
        "Running tiled inference on %s: %d tiles, tile_size=%d, overlap=%.2f",
        image_path.name, len(tiles), tile_size, overlap,
    )

    for idx, (tile, x0, y0) in enumerate(tiles, 1):
        tile_dets = _run_inference(
            model,
            tile,
            image_size=tile_size,
            confidence_threshold=conf_threshold,
            model_name="tiled_dino",
        )
        tile_preds = [_pipeline_det_to_prediction(det) for det in tile_dets]
        predictions.extend(remap_predictions(tile_preds, x0, y0))
        logger.debug(
            "tile %d/%d at (%d,%d): %d detections",
            idx, len(tiles), x0, y0, len(tile_preds),
        )

    final = _clip_predictions_to_image(
        nms_predictions(predictions, iou_threshold=0.4),
        width=img_array.shape[1],
        height=img_array.shape[0],
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info("Tiled inference complete: %d detections in %.0f ms", len(final), elapsed_ms)
    return final


def save_visualization(
    image_path: str | Path,
    predictions: list[Prediction],
    output_dir: str | Path,
    tile_size: int,
    overlap: float,
) -> Path:
    """Save a PNG visualization with tile grid and detections drawn.

    Args:
        image_path: Source image path.
        predictions: Full-image predictions with ``[x, y, w, h]`` boxes.
        output_dir: Directory where the visualization should be written.
        tile_size: Tile size used for inference.
        overlap: Tile overlap used for inference.

    Returns:
        Path to the saved visualization.
    """
    import cv2
    from inference.fits_loader import FITSLoader

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = FITSLoader().load(image_path)["array"]
    vis = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    for x0 in _tile_starts(image.shape[1], tile_size, stride):
        cv2.line(vis, (x0, 0), (x0, image.shape[0] - 1), (80, 80, 80), 1)
        cv2.line(
            vis,
            (min(x0 + tile_size, image.shape[1] - 1), 0),
            (min(x0 + tile_size, image.shape[1] - 1), image.shape[0] - 1),
            (80, 80, 80),
            1,
        )
    for y0 in _tile_starts(image.shape[0], tile_size, stride):
        cv2.line(vis, (0, y0), (image.shape[1] - 1, y0), (80, 80, 80), 1)
        cv2.line(
            vis,
            (0, min(y0 + tile_size, image.shape[0] - 1)),
            (image.shape[1] - 1, min(y0 + tile_size, image.shape[0] - 1)),
            (80, 80, 80),
            1,
        )

    for pred in predictions:
        x, y, w, h = pred["bbox"]
        x1, y1 = int(round(x)), int(round(y))
        x2, y2 = int(round(x + w)), int(round(y + h))
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            vis,
            f"{float(pred.get('score', 0.0)):.2f}",
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    out_path = output_dir / f"{image_path.stem}_tiled.png"
    cv2.imwrite(str(out_path), vis)
    return out_path


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    """Return fixed-stride starts whose final tile may extend into padding."""
    import math

    if length <= tile_size:
        return [0]
    n_tiles = int(math.ceil((length - tile_size) / stride)) + 1
    return [idx * stride for idx in range(n_tiles)]


def _pipeline_det_to_prediction(det: dict[str, Any]) -> Prediction:
    """Convert existing pipeline ``[x1, y1, x2, y2]`` output to ``[x, y, w, h]``."""
    x1, y1, x2, y2 = [float(v) for v in det["bbox"]]
    return {
        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
        "score": float(det.get("confidence", det.get("score", 0.0))),
        "category_id": int(det.get("category_id", 1)),
    }


def _xywh_to_xyxy(pred: Prediction) -> list[float]:
    """Convert one prediction bbox to ``[x1, y1, x2, y2]``."""
    x, y, w, h = [float(v) for v in pred["bbox"]]
    return [x, y, x + w, y + h]


def _torchvision_nms(preds: list[Prediction], iou_threshold: float) -> list[int]:
    """Run torchvision NMS independently per category."""
    import torch
    from torchvision.ops import nms

    keep: list[int] = []
    categories = sorted({int(pred.get("category_id", 1)) for pred in preds})
    for category in categories:
        indices = [
            i for i, pred in enumerate(preds)
            if int(pred.get("category_id", 1)) == category
        ]
        boxes = torch.tensor([_xywh_to_xyxy(preds[i]) for i in indices], dtype=torch.float32)
        scores = torch.tensor(
            [float(preds[i].get("score", 0.0)) for i in indices],
            dtype=torch.float32,
        )
        kept = nms(boxes, scores, iou_threshold).cpu().tolist()
        keep.extend(indices[int(i)] for i in kept)
    keep.sort(key=lambda i: float(preds[i].get("score", 0.0)), reverse=True)
    return keep


def _numpy_nms(preds: list[Prediction], iou_threshold: float) -> list[int]:
    """Pure-NumPy NMS fallback, independently per category."""
    import numpy as np

    keep: list[int] = []
    categories = sorted({int(pred.get("category_id", 1)) for pred in preds})
    for category in categories:
        indices = np.array(
            [i for i, pred in enumerate(preds) if int(pred.get("category_id", 1)) == category],
            dtype=np.int64,
        )
        boxes = np.array([_xywh_to_xyxy(preds[int(i)]) for i in indices], dtype=np.float32)
        scores = np.array(
            [float(preds[int(i)].get("score", 0.0)) for i in indices],
            dtype=np.float32,
        )
        order = scores.argsort()[::-1]

        while order.size > 0:
            current = int(order[0])
            keep.append(int(indices[current]))
            if order.size == 1:
                break
            ious = _box_iou_xyxy(boxes[current], boxes[order[1:]])
            order = order[1:][ious <= iou_threshold]

    keep.sort(key=lambda i: float(preds[i].get("score", 0.0)), reverse=True)
    return keep


def _box_iou_xyxy(box: "np.ndarray", boxes: "np.ndarray") -> "np.ndarray":
    """Compute IoU between one ``xyxy`` box and many ``xyxy`` boxes."""
    import numpy as np

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    box_area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    boxes_area = (
        np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    )
    union = box_area + boxes_area - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0.0)


def _clip_predictions_to_image(
    preds: list[Prediction],
    width: int,
    height: int,
) -> list[Prediction]:
    """Clip prediction boxes to image bounds and drop degenerate boxes."""
    clipped: list[Prediction] = []
    for pred in preds:
        x, y, w, h = [float(v) for v in pred["bbox"]]
        x1 = min(max(x, 0.0), float(width))
        y1 = min(max(y, 0.0), float(height))
        x2 = min(max(x + w, 0.0), float(width))
        y2 = min(max(y + h, 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            continue
        updated = dict(pred)
        updated["bbox"] = [x1, y1, x2 - x1, y2 - y1]
        clipped.append(updated)
    return clipped


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for tiled inference."""
    parser = argparse.ArgumentParser(description="Run ARGUS tiled DINO inference.")
    parser.add_argument("--image", required=True, help="Path to FITS/PNG/JPEG image")
    parser.add_argument(
        "--model-config",
        default="dinov3_gt_dm_satstreaks",
        help="MMDetection config path or ARGUS model size alias",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--tile-size", type=int, default=400, help="Square tile size in pixels")
    parser.add_argument("--overlap", type=float, default=0.5, help="Fractional tile overlap")
    parser.add_argument(
        "--conf-threshold", type=float, default=0.3, help="Minimum detection score"
    )
    parser.add_argument(
        "--output", required=True, help="Directory for visualization and JSON output"
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    return parser.parse_args()


def main() -> None:
    """Run the tiled inference CLI."""
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    predictions = run_tiled_inference(
        image_path=args.image,
        model_config=args.model_config,
        checkpoint=args.checkpoint,
        tile_size=args.tile_size,
        overlap=args.overlap,
        conf_threshold=args.conf_threshold,
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_path = save_visualization(
        args.image, predictions, output_dir, args.tile_size, args.overlap
    )
    json_path = output_dir / f"{Path(args.image).stem}_tiled_predictions.json"
    json_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

    print(f"Detected {len(predictions)} streak candidate(s)")
    print(f"Visualization: {vis_path}")
    print(f"Predictions:   {json_path}")


if __name__ == "__main__":
    main()
