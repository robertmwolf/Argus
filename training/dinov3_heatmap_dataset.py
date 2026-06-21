"""Dataset utilities for endpoint-centerline heatmap training."""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from inference.fits_loader import FITSLoader, apply_norm
from training.annotation_endpoints import annotation_to_endpoints
from training.data_paths import resolve_source_path

logger = logging.getLogger(__name__)


def rasterise_streak_heatmap(
    along_abs: np.ndarray,
    across: np.ndarray,
    half_len: float,
    patch_size: float,
    endpoint_taper_px: float = 16.0,
) -> np.ndarray:
    """Return per-cell heatmap values for one streak.

    Cells within the inner region receive 1.0. Cells in the endpoint taper zone
    receive a cosine rolloff from 1.0 (at ``half_len - effective_taper``) to 0.0
    (at ``half_len + patch_size / 2``). Cells outside or beyond the across limit
    receive 0.0 and should be masked by the caller.

    The taper length is capped at 30 % of the half-length so very short streaks
    do not collapse entirely.
    """
    effective_taper = min(endpoint_taper_px, half_len * 0.3)
    taper_start = half_len - effective_taper
    taper_end = half_len + patch_size / 2.0
    span = max(taper_end - taper_start, 1e-6)

    in_taper = (along_abs > taper_start) & (along_abs <= taper_end)
    t = np.where(in_taper, (along_abs - taper_start) / span, 0.0)
    taper_val = np.where(in_taper, 0.5 * (1.0 + np.cos(np.pi * t)), 0.0)
    inner_val = (along_abs <= taper_start).astype(np.float32)
    return np.maximum(inner_val, taper_val).astype(np.float32)


class StreakHeatmapDataset(Dataset):
    """Load source annotations as image tensors plus centerline heatmaps."""

    def __init__(
        self,
        annotation_file: str | Path,
        image_size: int = 512,
        patch_size: int = 16,
        max_samples: int | None = None,
        norm_mode: str = "autostretch",
        data_root: str | Path | None = None,
        scratch_root: str | Path | None = None,
        endpoint_taper_px: float = 16.0,
    ) -> None:
        """Initialise the endpoint heatmap dataset."""
        self.annotation_file = Path(annotation_file)
        self.image_size = image_size
        self.patch_size = patch_size
        self.norm_mode = norm_mode
        self.data_root = data_root
        self.scratch_root = scratch_root
        self.endpoint_taper_px = endpoint_taper_px
        self.loader = FITSLoader()

        source = json.loads(self.annotation_file.read_text())
        images = list(source.get("images", []))
        self.images = images[:max_samples] if max_samples else images
        self.id_to_anns: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in source.get("annotations", []):
            self.id_to_anns[int(annotation["image_id"])].append(annotation)

    def __len__(self) -> int:
        """Return the number of images."""
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        """Load one image and rasterise its endpoint centerlines."""
        meta = self.images[idx]
        image_id = int(meta["id"])
        array = self._load_image(self._resolve_image_path(str(meta["file_name"])))
        orig_h, orig_w = array.shape[:2]
        image, scale, pad_x, pad_y = self._letterbox_image(array)
        heatmap = self._build_heatmap(
            self.id_to_anns.get(image_id, []),
            orig_w,
            orig_h,
            scale,
            pad_x,
            pad_y,
        )
        return {
            "image": image,
            "heatmap": heatmap,
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.float32),
            "letterbox": torch.tensor([scale, pad_x, pad_y], dtype=torch.float32),
            "file_name": str(meta["file_name"]),
        }

    def _resolve_image_path(self, file_name: str) -> Path:
        """Resolve source image paths used by annotation manifests."""
        return resolve_source_path(
            file_name, self.annotation_file, self.data_root, self.scratch_root
        )

    def _load_image(self, path: Path) -> np.ndarray:
        """Load and normalize a FITS, NPY, or ordinary image."""
        try:
            if path.suffix.lower() in {".fits", ".fit", ".fts"}:
                loaded = self.loader.load(path)
                raw = loaded.get("raw_float32")
                return apply_norm(raw, self.norm_mode) if raw is not None else np.asarray(loaded["array"])
            if path.suffix.lower() == ".npy":
                raw = np.load(path)
                if raw.dtype == np.uint8:
                    return np.stack([raw, raw, raw], axis=-1)
                return apply_norm(raw.astype(np.float32), self.norm_mode)
            with Image.open(path) as image:
                return np.asarray(image.convert("RGB"), dtype=np.uint8)
        except Exception as exc:
            logger.warning("Failed to load %s: %s; using blank image", path, exc)
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

    def _letterbox_image(self, array: np.ndarray) -> tuple[torch.Tensor, float, float, float]:
        """Resize an image preserving aspect ratio and pad to a square."""
        orig_h, orig_w = array.shape[:2]
        if orig_h <= 0 or orig_w <= 0:
            return torch.zeros((3, self.image_size, self.image_size)), 1.0, 0.0, 0.0
        scale = min(self.image_size / orig_w, self.image_size / orig_h)
        new_w = max(1, round(orig_w * scale))
        new_h = max(1, round(orig_h * scale))
        pad_x = (self.image_size - new_w) / 2.0
        pad_y = (self.image_size - new_h) / 2.0
        resized = Image.fromarray(array).resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (self.image_size, self.image_size))
        canvas.paste(resized, (round(pad_x), round(pad_y)))
        values = np.asarray(canvas, dtype=np.uint8)
        tensor = torch.from_numpy(values.transpose(2, 0, 1).astype(np.float32) / 255.0)
        return tensor, float(scale), float(pad_x), float(pad_y)

    def _build_heatmap(
        self,
        annotations: list[dict[str, Any]],
        orig_w: int,
        orig_h: int,
        scale: float,
        pad_x: float,
        pad_y: float,
    ) -> torch.Tensor:
        """Rasterise annotation endpoints onto the feature grid."""
        grid_size = self.image_size // self.patch_size
        target = np.zeros((grid_size, grid_size), dtype=np.float32)
        if orig_w <= 0 or orig_h <= 0:
            return torch.from_numpy(target).unsqueeze(0)
        yy, xx = np.mgrid[0:grid_size, 0:grid_size].astype(np.float32)
        px = (xx + 0.5) * self.patch_size
        py = (yy + 0.5) * self.patch_size
        for annotation in annotations:
            x1, y1, x2, y2 = annotation_to_endpoints(annotation)
            x1, y1 = x1 * scale + pad_x, y1 * scale + pad_y
            x2, y2 = x2 * scale + pad_x, y2 * scale + pad_y
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            length = math.hypot(x2 - x1, y2 - y1)
            angle = math.atan2(y2 - y1, x2 - x1)
            ux, uy = math.cos(angle), math.sin(angle)
            dx, dy = px - cx, py - cy
            along_abs = np.abs(dx * ux + dy * uy)
            across = np.abs(-dx * uy + dy * ux)
            half_len = length / 2.0
            across_mask = across <= self.patch_size / 2.0
            values = rasterise_streak_heatmap(
                along_abs, across, half_len, self.patch_size, self.endpoint_taper_px
            )
            target = np.maximum(target, np.where(across_mask, values, 0.0))
        return torch.from_numpy(target).unsqueeze(0)


def collate_heatmap_batch(batch: list[dict[str, torch.Tensor | str | int]]) -> dict[str, Any]:
    """Collate endpoint heatmap samples into a training batch."""
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "heatmap": torch.stack([item["heatmap"] for item in batch]),
        "image_id": torch.stack([item["image_id"] for item in batch]),
        "orig_size": torch.stack([item["orig_size"] for item in batch]),
        "letterbox": torch.stack([item["letterbox"] for item in batch]),
        "file_name": [item["file_name"] for item in batch],
    }


if __name__ == "__main__":
    print("Use training/train_dinov3_heatmap.py to train the dataset.")
