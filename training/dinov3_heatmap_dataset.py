"""Dataset utilities for the plain PyTorch DINOv3 heatmap spike."""

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

logger = logging.getLogger(__name__)


class StreakHeatmapDataset(Dataset):
    """Load COCO streak annotations as image tensors plus patch-grid heatmaps."""

    def __init__(
        self,
        annotation_file: str | Path,
        image_size: int = 512,
        patch_size: int = 16,
        max_samples: int | None = None,
        norm_mode: str = "autostretch",
    ) -> None:
        """Initialise the dataset.

        Args:
            annotation_file: COCO-format annotation JSON.
            image_size: Square letterbox canvas size used for the spike.
            patch_size: DINOv3 patch size; heatmaps are ``image_size / patch``.
            max_samples: Optional quick-run cap.
            norm_mode: Normalisation applied to raw float32 FITS/NPY pixels before
                feeding the backbone. One of ``'autostretch'`` (PixInsight AutoSTF,
                background-subtracting), ``'zscore'`` (3-sigma clip), or
                ``'zscale'`` (IRAF ZScale). Autostretch is the default because it
                removes the sky background so streak signal is consistently visible
                across tiles regardless of local star brightness.
        """
        self.annotation_file = Path(annotation_file)
        self.image_size = image_size
        self.patch_size = patch_size
        self.norm_mode = norm_mode
        self.loader = FITSLoader()

        coco = json.loads(self.annotation_file.read_text())
        images = list(coco.get("images", []))
        self.images = images[:max_samples] if max_samples else images
        self.id_to_anns: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ann in coco.get("annotations", []):
            self.id_to_anns[int(ann["image_id"])].append(ann)

    def __len__(self) -> int:
        """Return number of images."""
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        """Load one sample."""
        meta = self.images[idx]
        image_id = int(meta["id"])
        image_path = self._resolve_image_path(str(meta["file_name"]))
        array = self._load_image(image_path)
        orig_h, orig_w = array.shape[:2]

        image, scale, pad_x, pad_y = self._letterbox_image(array)

        heatmap = self._build_heatmap(
            self.id_to_anns.get(image_id, []),
            orig_w=orig_w,
            orig_h=orig_h,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
        )
        center_heatmap, box_target, box_mask = self._build_box_targets(
            self.id_to_anns.get(image_id, []),
            orig_w=orig_w,
            orig_h=orig_h,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
        )

        return {
            "image": image,
            "heatmap": heatmap,
            "center_heatmap": center_heatmap,
            "box_target": box_target,
            "box_mask": box_mask,
            "geometry": self._build_geometry(
                self.id_to_anns.get(image_id, []),
                heatmap=heatmap,
                orig_w=orig_w,
                orig_h=orig_h,
                scale=scale,
                pad_x=pad_x,
                pad_y=pad_y,
            ),
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.float32),
            "letterbox": torch.tensor([scale, pad_x, pad_y], dtype=torch.float32),
            "file_name": str(meta["file_name"]),
        }

    def _resolve_image_path(self, file_name: str) -> Path:
        raw_path = Path(file_name)
        if raw_path.is_absolute():
            return raw_path
        candidates = [
            self.annotation_file.parent / raw_path,
            self.annotation_file.parent.parent / raw_path,
            Path("data") / raw_path,
        ]
        if "/" not in file_name:
            candidates.append(Path("data/GTImages") / raw_path)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _load_image(self, path: Path) -> np.ndarray:
        suffix = path.suffix.lower()
        try:
            if suffix in {".fits", ".fit", ".fts"}:
                loaded = self.loader.load(path)
                raw = loaded.get("raw_float32")
                if raw is not None:
                    return apply_norm(raw, self.norm_mode)  # (H, W, 3) uint8
                return np.asarray(loaded["array"], dtype=np.uint8)
            elif suffix == ".npy":
                raw = np.load(str(path))
                if raw.dtype == np.uint8:
                    # Full-image normalisation was applied at convert_tiles_to_npy time.
                    return np.stack([raw, raw, raw], axis=-1)
                return apply_norm(raw.astype(np.float32), self.norm_mode)  # (H, W, 3) uint8
            else:
                with Image.open(path) as im:
                    return np.asarray(im.convert("RGB"), dtype=np.uint8)
        except Exception as exc:
            logger.warning("Failed to load %s: %s; using blank image", path, exc)
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

    def _letterbox_image(self, array: np.ndarray) -> tuple[torch.Tensor, float, float, float]:
        """Resize image preserving aspect ratio and pad to square canvas."""
        orig_h, orig_w = array.shape[:2]
        if orig_h <= 0 or orig_w <= 0:
            blank = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
            return blank, 1.0, 0.0, 0.0

        scale = min(self.image_size / orig_w, self.image_size / orig_h)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        pad_x = (self.image_size - new_w) / 2.0
        pad_y = (self.image_size - new_h) / 2.0

        image = Image.fromarray(array).resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (self.image_size, self.image_size), color=(0, 0, 0))
        canvas.paste(image, (int(round(pad_x)), int(round(pad_y))))
        canvas_np = np.asarray(canvas, dtype=np.uint8)
        tensor = torch.from_numpy(canvas_np.transpose(2, 0, 1).astype(np.float32) / 255.0)
        return tensor, float(scale), float(pad_x), float(pad_y)

    def _build_heatmap(
        self,
        anns: list[dict[str, Any]],
        orig_w: int,
        orig_h: int,
        scale: float,
        pad_x: float,
        pad_y: float,
    ) -> torch.Tensor:
        grid_size = self.image_size // self.patch_size
        target = np.zeros((grid_size, grid_size), dtype=np.float32)
        if orig_w <= 0 or orig_h <= 0:
            return torch.from_numpy(target).unsqueeze(0)

        yy, xx = np.mgrid[0:grid_size, 0:grid_size].astype(np.float32)
        px = (xx + 0.5) * self.patch_size
        py = (yy + 0.5) * self.patch_size

        for ann in anns:
            obb = ann.get("obb")
            if obb:
                if isinstance(obb, dict):
                    cx = float(obb["cx"]) * scale + pad_x
                    cy = float(obb["cy"]) * scale + pad_y
                    length = max(float(obb["w"]) * scale, float(obb["h"]) * scale)
                    width = max(min(float(obb["w"]) * scale, float(obb["h"]) * scale), self.patch_size)
                    angle = math.radians(float(obb.get("angle_deg", 0.0)))
                else:
                    cx = float(obb[0]) * scale + pad_x
                    cy = float(obb[1]) * scale + pad_y
                    length = max(float(obb[2]) * scale, float(obb[3]) * scale)
                    width = max(min(float(obb[2]) * scale, float(obb[3]) * scale), self.patch_size)
                    angle = math.radians(float(obb[4]))
                self._rasterise_line(target, px, py, cx, cy, length, width, angle)
            else:
                x, y, w, h = (float(v) for v in ann["bbox"])
                x1, y1 = x * scale + pad_x, y * scale + pad_y
                x2, y2 = (x + w) * scale + pad_x, (y + h) * scale + pad_y
                inside = (px >= x1) & (px <= x2) & (py >= y1) & (py <= y2)
                target[inside] = 1.0

        return torch.from_numpy(target).unsqueeze(0)

    def _build_geometry(
        self,
        anns: list[dict[str, Any]],
        heatmap: torch.Tensor,
        orig_w: int,
        orig_h: int,
        scale: float,
        pad_x: float,
        pad_y: float,
    ) -> torch.Tensor:
        """Build per-cell geometry targets aligned with the heatmap.

        Channels are: cos(2θ), sin(2θ), length/image_size, width/image_size.
        """
        grid_size = self.image_size // self.patch_size
        geom = np.zeros((4, grid_size, grid_size), dtype=np.float32)
        if orig_w <= 0 or orig_h <= 0:
            return torch.from_numpy(geom)

        yy, xx = np.mgrid[0:grid_size, 0:grid_size].astype(np.float32)
        px = (xx + 0.5) * self.patch_size
        py = (yy + 0.5) * self.patch_size

        for ann in anns:
            obb = ann.get("obb")
            if obb:
                if isinstance(obb, dict):
                    cx = float(obb["cx"]) * scale + pad_x
                    cy = float(obb["cy"]) * scale + pad_y
                    length = max(float(obb["w"]) * scale, float(obb["h"]) * scale)
                    width = max(min(float(obb["w"]) * scale, float(obb["h"]) * scale), self.patch_size)
                    angle_deg = float(obb.get("angle_deg", 0.0))
                else:
                    cx = float(obb[0]) * scale + pad_x
                    cy = float(obb[1]) * scale + pad_y
                    length = max(float(obb[2]) * scale, float(obb[3]) * scale)
                    width = max(min(float(obb[2]) * scale, float(obb[3]) * scale), self.patch_size)
                    angle_deg = float(obb[4])
                angle = math.radians(angle_deg)
                ux, uy = math.cos(angle), math.sin(angle)
                dx = px - cx
                dy = py - cy
                along = dx * ux + dy * uy
                across = np.abs(-dx * uy + dy * ux)
                mask = (np.abs(along) <= length / 2 + 8.0) & (across <= width / 2 + 8.0)
                geom[0, mask] = math.cos(2.0 * angle)
                geom[1, mask] = math.sin(2.0 * angle)
                geom[2, mask] = min(length / self.image_size, 2.0)
                geom[3, mask] = min(width / self.image_size, 1.0)
            else:
                x, y, w, h = (float(v) for v in ann["bbox"])
                x1, y1 = x * scale + pad_x, y * scale + pad_y
                x2, y2 = (x + w) * scale + pad_x, (y + h) * scale + pad_y
                mask = (px >= x1) & (px <= x2) & (py >= y1) & (py <= y2)
                length = max((x2 - x1), (y2 - y1))
                width = max(min((x2 - x1), (y2 - y1)), self.patch_size)
                geom[0, mask] = 1.0
                geom[1, mask] = 0.0
                geom[2, mask] = min(length / self.image_size, 2.0)
                geom[3, mask] = min(width / self.image_size, 1.0)

        geom[:, heatmap.squeeze(0).numpy() <= 0] = 0.0
        return torch.from_numpy(geom)

    def _build_box_targets(
        self,
        anns: list[dict[str, Any]],
        orig_w: int,
        orig_h: int,
        scale: float,
        pad_x: float,
        pad_y: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build sparse center and box-regression targets.

        Box channels are: dx, dy, cos(2θ), sin(2θ), length/image_size,
        width/image_size. Offsets are relative to the patch-cell center and
        divided by patch size.
        """
        grid_size = self.image_size // self.patch_size
        center = np.zeros((1, grid_size, grid_size), dtype=np.float32)
        box = np.zeros((6, grid_size, grid_size), dtype=np.float32)
        mask = np.zeros((1, grid_size, grid_size), dtype=np.float32)
        if orig_w <= 0 or orig_h <= 0:
            return torch.from_numpy(center), torch.from_numpy(box), torch.from_numpy(mask)

        for ann in anns:
            parsed = self._parse_annotation_box(ann, scale, pad_x, pad_y)
            if parsed is None:
                continue
            cx, cy, length, width, angle_deg = parsed
            if not (0 <= cx < self.image_size and 0 <= cy < self.image_size):
                continue
            center_gx = int(np.clip(cx / self.patch_size, 0, grid_size - 1))
            center_gy = int(np.clip(cy / self.patch_size, 0, grid_size - 1))
            if center[0, center_gy, center_gx] > 0 and box[4, center_gy, center_gx] >= min(length / self.image_size, 2.0):
                continue

            angle = math.radians(angle_deg)
            gaussian = self._draw_gaussian(center[0], center_gx, center_gy, sigma=1.0)
            ys, xs = np.nonzero(gaussian >= 0.25)
            for gy, gx in zip(ys.tolist(), xs.tolist()):
                weight = float(gaussian[gy, gx])
                if weight <= mask[0, gy, gx]:
                    continue
                cell_cx = (gx + 0.5) * self.patch_size
                cell_cy = (gy + 0.5) * self.patch_size
                mask[0, gy, gx] = weight
                box[0, gy, gx] = float(np.clip((cx - cell_cx) / self.patch_size, -2.0, 2.0))
                box[1, gy, gx] = float(np.clip((cy - cell_cy) / self.patch_size, -2.0, 2.0))
                box[2, gy, gx] = math.cos(2.0 * angle)
                box[3, gy, gx] = math.sin(2.0 * angle)
                box[4, gy, gx] = min(length / self.image_size, 2.0)
                box[5, gy, gx] = min(width / self.image_size, 1.0)

        return torch.from_numpy(center), torch.from_numpy(box), torch.from_numpy(mask)

    @staticmethod
    def _draw_gaussian(target: np.ndarray, cx: int, cy: int, sigma: float) -> np.ndarray:
        """Draw a clipped Gaussian center target on the patch grid."""
        full_gaussian = np.zeros_like(target, dtype=np.float32)
        radius = max(1, int(math.ceil(3.0 * sigma)))
        h, w = target.shape
        x1 = max(0, cx - radius)
        x2 = min(w, cx + radius + 1)
        y1 = max(0, cy - radius)
        y2 = min(h, cy + radius + 1)
        if x1 >= x2 or y1 >= y2:
            return full_gaussian
        yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)
        gaussian = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma)))
        gaussian = gaussian.astype(np.float32)
        full_gaussian[y1:y2, x1:x2] = gaussian
        target[y1:y2, x1:x2] = np.maximum(target[y1:y2, x1:x2], gaussian)
        return full_gaussian

    def _parse_annotation_box(
        self,
        ann: dict[str, Any],
        scale: float,
        pad_x: float,
        pad_y: float,
    ) -> tuple[float, float, float, float, float] | None:
        """Parse an annotation into letterboxed center, size, and angle."""
        obb = ann.get("obb")
        if obb:
            if isinstance(obb, dict):
                cx = float(obb["cx"]) * scale + pad_x
                cy = float(obb["cy"]) * scale + pad_y
                w = float(obb["w"]) * scale
                h = float(obb["h"]) * scale
                angle_deg = float(obb.get("angle_deg", 0.0))
            else:
                cx = float(obb[0]) * scale + pad_x
                cy = float(obb[1]) * scale + pad_y
                w = float(obb[2]) * scale
                h = float(obb[3]) * scale
                angle_deg = float(obb[4])
            return cx, cy, max(w, h), max(min(w, h), 1e-3), angle_deg

        bbox = ann.get("bbox")
        if not bbox:
            return None
        x, y, w, h = (float(v) for v in bbox)
        x1, y1 = x * scale + pad_x, y * scale + pad_y
        x2, y2 = (x + w) * scale + pad_x, (y + h) * scale + pad_y
        return (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1), max(min(x2 - x1, y2 - y1), 1e-3), 0.0

    @staticmethod
    def _rasterise_line(
        target: np.ndarray,
        px: np.ndarray,
        py: np.ndarray,
        cx: float,
        cy: float,
        length: float,
        width: float,
        angle: float,
    ) -> None:
        ux, uy = math.cos(angle), math.sin(angle)
        dx = px - cx
        dy = py - cy
        along = dx * ux + dy * uy
        across = np.abs(-dx * uy + dy * ux)
        mask = (np.abs(along) <= length / 2 + 8.0) & (across <= width / 2 + 8.0)
        target[mask] = 1.0


def collate_heatmap_batch(batch: list[dict[str, torch.Tensor | str | int]]) -> dict[str, Any]:
    """Collate heatmap samples into a batch dict."""
    return {
        "image": torch.stack([item["image"] for item in batch]),  # type: ignore[arg-type]
        "heatmap": torch.stack([item["heatmap"] for item in batch]),  # type: ignore[arg-type]
        "center_heatmap": torch.stack([item["center_heatmap"] for item in batch]),  # type: ignore[arg-type]
        "box_target": torch.stack([item["box_target"] for item in batch]),  # type: ignore[arg-type]
        "box_mask": torch.stack([item["box_mask"] for item in batch]),  # type: ignore[arg-type]
        "geometry": torch.stack([item["geometry"] for item in batch]),  # type: ignore[arg-type]
        "image_id": torch.stack([item["image_id"] for item in batch]),  # type: ignore[arg-type]
        "orig_size": torch.stack([item["orig_size"] for item in batch]),  # type: ignore[arg-type]
        "letterbox": torch.stack([item["letterbox"] for item in batch]),  # type: ignore[arg-type]
        "file_name": [item["file_name"] for item in batch],
    }
