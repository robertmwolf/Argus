"""Orientation-binned centerline targets for the DINOv3 heatmap spike."""

from __future__ import annotations

import json
import logging
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Virtual tile path format shared with training/transforms.py and
# scripts/build_tiled_frigate_json.py:
#   <real_stem>__tx<x0>_ty<y0>_ts<size>.<ext>
# The loader detects this pattern, loads the *parent* file, and crops the
# requested tile before any further processing.
_TILE_RE = re.compile(r"^(.+?)__tx(\d+)_ty(\d+)_ts(\d+)$")


@dataclass(frozen=True)
class CenterlineSample:
    """One full-frame or tiled training sample."""

    image_id: int
    file_name: str
    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int
    positive: bool


class DINOv3OrientationCenterlineDataset(Dataset):
    """Load COCO images as orientation-binned soft centerline targets."""

    def __init__(
        self,
        annotation_file: str | Path,
        split: Literal["train", "val", "holdout"] = "train",
        tile_size: int = 2560,
        image_size: int = 1024,
        orientation_bins: int = 18,
        centerline_width: float = 2.0,
        centerline_sigma: float = 1.4,
        catchment_width: float = 0.0,
        catchment_sigma: float = 6.0,
        neighbor_bin_weight: float = 0.35,
        second_neighbor_weight: float = 0.0,
        positive_tiles: int | None = 1236,
        negative_tiles: int | None = 1400,
        preserve_image_bit_depth: bool = False,
        seed: int = 20260524,
        max_samples: int | None = None,
    ) -> None:
        """Initialise the dataset.

        Args:
            annotation_file: COCO-format annotation JSON.
            split: ``train`` builds tiles; validation/holdout uses full frames.
            tile_size: Native-pixel square tile size for train split.
            image_size: Model input size after bilinear resize.
            orientation_bins: Number of half-circle orientation bins.
            centerline_width: Hard support around the centerline in pixels.
            centerline_sigma: Soft Gaussian falloff across the centerline.
            catchment_width: Optional wider recoverable seed-zone support.
            catchment_sigma: Optional wider recoverable seed-zone falloff.
            neighbor_bin_weight: Weight for adjacent orientation bins.
            second_neighbor_weight: Weight for second-neighbor bins.
            positive_tiles: Optional cap on positive train tiles.
            negative_tiles: Optional cap on negative train tiles.
            preserve_image_bit_depth: Preserve 16-bit PNG dynamic range on load.
            seed: Deterministic tile sampling seed.
            max_samples: Optional quick-run cap after sample construction.
        """
        self.annotation_file = Path(annotation_file)
        self.split = split
        self.tile_size = tile_size
        self.image_size = image_size
        self.orientation_bins = orientation_bins
        self.centerline_width = centerline_width
        self.centerline_sigma = centerline_sigma
        self.catchment_width = catchment_width
        self.catchment_sigma = catchment_sigma
        self.neighbor_bin_weight = neighbor_bin_weight
        self.second_neighbor_weight = second_neighbor_weight
        self.preserve_image_bit_depth = preserve_image_bit_depth

        coco = json.loads(self.annotation_file.read_text())
        self.images: list[dict[str, Any]] = list(coco.get("images", []))
        self.image_by_id = {int(item["id"]): item for item in self.images}
        self.id_to_anns: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ann in coco.get("annotations", []):
            self.id_to_anns[int(ann["image_id"])].append(ann)

        rng = random.Random(seed)
        if split == "train":
            self.samples = self._build_train_tiles(rng, positive_tiles, negative_tiles)
        else:
            self.samples = self._build_full_frame_samples()
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        logger.info(
            "Built %s centerline dataset: samples=%d positives=%d negatives=%d",
            split,
            len(self.samples),
            sum(1 for sample in self.samples if sample.positive),
            sum(1 for sample in self.samples if not sample.positive),
        )

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        """Load one image tile and its orientation-binned centerline target."""
        sample = self.samples[idx]
        image_path = self._resolve_image_path(sample.file_name)
        image = self._load_resized_crop(image_path, sample)
        target = self._build_target(sample)
        catchment = self._build_target(
            sample,
            centerline_width=self.catchment_width,
            centerline_sigma=self.catchment_sigma,
        ) if self.catchment_width > 0.0 else target
        return {
            "image": image,
            "target": target,
            "catchment_target": catchment,
            "positive": torch.tensor(float(sample.positive), dtype=torch.float32),
            "image_id": torch.tensor(sample.image_id, dtype=torch.int64),
            "file_name": sample.file_name,
        }

    def _load_resized_crop(self, path: Path, sample: CenterlineSample) -> torch.Tensor:
        """Load only the needed crop where possible, then resize to model input.

        Handles *virtual tile paths* of the form::

            <real_stem>__tx<x0>_ty<y0>_ts<size>.<ext>

        These paths are not real files on disk.  The tile suffix encodes the
        (x0, y0, tile_size) crop into the filename so that a single parent
        image can be shared across many tiles without pre-slicing it.  When
        detected, the loader opens the *parent* file and offsets the crop box
        by (x0, y0) before extracting the region needed for this sample.
        """
        # ── virtual tile detection ─────────────────────────────────────────
        tile_offset: tuple[int, int] | None = None
        m = _TILE_RE.match(path.stem)
        if m:
            real_stem = m.group(1)
            x0, y0, ts = int(m.group(2)), int(m.group(3)), int(m.group(4))
            path = path.parent / (real_stem + path.suffix)
            tile_offset = (x0, y0)

        suffix = path.suffix.lower()

        # ── FITS branch ────────────────────────────────────────────────────
        if suffix in {".fits", ".fit", ".fts"}:
            array = self._load_image(path)
            if tile_offset is not None:
                ox, oy = tile_offset
                h, w = array.shape[:2]
                pad_h = max(0, oy + ts - h)
                pad_w = max(0, ox + ts - w)
                if pad_h > 0 or pad_w > 0:
                    array = np.pad(array, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
                array = array[oy:oy + ts, ox:ox + ts].copy()
            crop = self._crop_with_padding(array, sample)
            return self._resize_to_tensor(crop)

        # ── raster branch (PNG / TIFF / …) ────────────────────────────────
        try:
            with Image.open(path) as im:
                # Offset the sample's crop into the parent image coordinate space.
                ox, oy = tile_offset if tile_offset is not None else (0, 0)
                box = (
                    ox + int(sample.crop_x),
                    oy + int(sample.crop_y),
                    ox + int(sample.crop_x + sample.crop_w),
                    oy + int(sample.crop_y + sample.crop_h),
                )
                if self.preserve_image_bit_depth and im.mode in {"I;16", "I;16B", "I", "F"}:
                    arr = np.asarray(im.crop(box), dtype=np.float32)
                    arr = self._scale_float_image(arr)
                    array = np.repeat(arr[..., None], 3, axis=2)
                    return self._resize_to_tensor(array)
                resized = im.convert("RGB").crop(box).resize(
                    (self.image_size, self.image_size),
                    Image.BILINEAR,
                )
                arr = np.asarray(resized, dtype=np.float32) / 255.0
                return torch.from_numpy(np.transpose(arr, (2, 0, 1)).copy())
        except Exception as exc:
            logger.warning("Failed to load %s: %s; using blank image", path, exc)
            return torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)

    def _build_train_tiles(
        self,
        rng: random.Random,
        positive_tiles: int | None,
        negative_tiles: int | None,
    ) -> list[CenterlineSample]:
        positives: list[CenterlineSample] = []
        negatives: list[CenterlineSample] = []
        for meta in self.images:
            image_id = int(meta["id"])
            width = int(meta.get("width", self.tile_size))
            height = int(meta.get("height", self.tile_size))
            anns = self.id_to_anns.get(image_id, [])
            x_starts = self._tile_starts(width)
            y_starts = self._tile_starts(height)
            for crop_y in y_starts:
                for crop_x in x_starts:
                    sample = CenterlineSample(
                        image_id=image_id,
                        file_name=str(meta["file_name"]),
                        crop_x=crop_x,
                        crop_y=crop_y,
                        crop_w=min(self.tile_size, width - crop_x),
                        crop_h=min(self.tile_size, height - crop_y),
                        positive=self._tile_has_streak(anns, crop_x, crop_y),
                    )
                    if sample.positive:
                        positives.append(sample)
                    else:
                        negatives.append(sample)
        rng.shuffle(positives)
        rng.shuffle(negatives)
        if positive_tiles is not None:
            positives = positives[:positive_tiles]
        if negative_tiles is not None:
            negatives = negatives[:negative_tiles]
        samples = positives + negatives
        rng.shuffle(samples)
        return samples

    def _build_full_frame_samples(self) -> list[CenterlineSample]:
        samples: list[CenterlineSample] = []
        for meta in self.images:
            image_id = int(meta["id"])
            width = int(meta.get("width", self.tile_size))
            height = int(meta.get("height", self.tile_size))
            samples.append(
                CenterlineSample(
                    image_id=image_id,
                    file_name=str(meta["file_name"]),
                    crop_x=0,
                    crop_y=0,
                    crop_w=width,
                    crop_h=height,
                    positive=bool(self.id_to_anns.get(image_id)),
                )
            )
        return samples

    def _tile_starts(self, extent: int) -> list[int]:
        if extent <= self.tile_size:
            return [0]
        starts = list(range(0, max(extent - self.tile_size + 1, 1), self.tile_size))
        last = extent - self.tile_size
        if starts[-1] != last:
            starts.append(last)
        return starts

    def _tile_has_streak(self, anns: list[dict[str, Any]], crop_x: int, crop_y: int) -> bool:
        crop = (crop_x, crop_y, crop_x + self.tile_size, crop_y + self.tile_size)
        for ann in anns:
            x, y, w, h = (float(v) for v in ann.get("bbox", [0, 0, 0, 0]))
            if self._boxes_intersect(crop, (x, y, x + w, y + h)):
                return True
        return False

    @staticmethod
    def _boxes_intersect(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
        return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]

    def _resolve_image_path(self, file_name: str) -> Path:
        raw_path = Path(file_name)
        if raw_path.is_absolute():
            return raw_path
        candidates = [
            self.annotation_file.parent / raw_path,
            self.annotation_file.parent.parent / raw_path,
            Path("data") / raw_path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _load_image(self, path: Path) -> np.ndarray:
        suffix = path.suffix.lower()
        try:
            if suffix in {".fits", ".fit", ".fts"}:
                return self._load_fits_pixels(path)
            with Image.open(path) as im:
                if self.preserve_image_bit_depth and im.mode in {"I;16", "I;16B", "I", "F"}:
                    arr = np.asarray(im, dtype=np.float32)
                    arr = self._scale_float_image(arr)
                    return np.repeat(arr[..., None], 3, axis=2)
                arr = np.asarray(im.convert("RGB"), dtype=np.float32)
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                return arr / 255.0
        except Exception as exc:
            logger.warning("Failed to load %s: %s; using blank image", path, exc)
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)

    def _load_fits_pixels(self, path: Path) -> np.ndarray:
        """Load FITS pixels directly without WCS or plate solving."""
        from astropy.io import fits

        with fits.open(path, memmap=False) as hdul:
            raw = hdul[0].data
            if raw is None:
                raise ValueError(f"Primary HDU in {path.name} contains no image data")
            arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim > 2:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Unsupported FITS image shape for {path.name}: {arr.shape}")
        scaled = self._scale_zscore(arr)
        return np.repeat(scaled[..., None], 3, axis=2)

    @staticmethod
    def _scale_zscore(arr: np.ndarray, sigma: float = 3.0) -> np.ndarray:
        """Z-score normalise FITS pixels to float32 [0, 1].

        Clips to [mean − sigma*std, mean + sigma*std] then rescales linearly.
        Matches the zscore mode in inference.fits_loader._normalise_zscore.
        """
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros_like(arr, dtype=np.float32)
        mean = float(finite.mean())
        std = float(finite.std())
        if std == 0.0:
            return np.zeros_like(arr, dtype=np.float32)
        lo = mean - sigma * std
        hi = mean + sigma * std
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _scale_float_image(arr: np.ndarray) -> np.ndarray:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros_like(arr, dtype=np.float32)
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
        return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    def _crop_with_padding(self, array: np.ndarray, sample: CenterlineSample) -> np.ndarray:
        h, w = array.shape[:2]
        x1 = max(sample.crop_x, 0)
        y1 = max(sample.crop_y, 0)
        x2 = min(sample.crop_x + sample.crop_w, w)
        y2 = min(sample.crop_y + sample.crop_h, h)
        crop = array[y1:y2, x1:x2]
        if crop.shape[0] == self.tile_size and crop.shape[1] == self.tile_size:
            return crop
        canvas = np.zeros((sample.crop_h, sample.crop_w, 3), dtype=np.float32)
        canvas[: crop.shape[0], : crop.shape[1]] = crop
        return canvas

    def _resize_to_tensor(self, array: np.ndarray) -> torch.Tensor:
        import cv2

        resized = cv2.resize(
            array.astype(np.float32),
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        if resized.ndim == 2:
            resized = np.repeat(resized[..., None], 3, axis=2)
        rgb = np.transpose(np.clip(resized, 0.0, 1.0), (2, 0, 1))
        return torch.from_numpy(rgb.copy())

    def _build_target(
        self,
        sample: CenterlineSample,
        centerline_width: float | None = None,
        centerline_sigma: float | None = None,
    ) -> torch.Tensor:
        target = np.zeros(
            (self.orientation_bins, self.image_size, self.image_size),
            dtype=np.float32,
        )
        line_width = self.centerline_width if centerline_width is None else centerline_width
        line_sigma = self.centerline_sigma if centerline_sigma is None else centerline_sigma
        sx = self.image_size / max(sample.crop_w, 1)
        sy = self.image_size / max(sample.crop_h, 1)
        for ann in self.id_to_anns.get(sample.image_id, []):
            parsed = self._parse_annotation(ann)
            if parsed is None:
                continue
            cx, cy, length, width, angle = parsed
            cx = (cx - sample.crop_x) * sx
            cy = (cy - sample.crop_y) * sy
            length = length * (sx + sy) * 0.5
            width = max(width * (sx + sy) * 0.5, line_width)
            if not self._line_intersects_input(cx, cy, length, angle, line_width, line_sigma):
                continue
            self._rasterise_centerline(target, cx, cy, length, max(width, line_width), angle, line_width, line_sigma)
        return torch.from_numpy(target)

    def _parse_annotation(self, ann: dict[str, Any]) -> tuple[float, float, float, float, float] | None:
        obb = ann.get("obb")
        if obb:
            if isinstance(obb, dict):
                cx = float(obb["cx"])
                cy = float(obb["cy"])
                w = float(obb["w"])
                h = float(obb["h"])
                angle = math.radians(float(obb.get("angle_deg", 0.0)))
            else:
                cx = float(obb[0])
                cy = float(obb[1])
                w = float(obb[2])
                h = float(obb[3])
                angle = math.radians(float(obb[4]))
            if h > w:
                angle += math.pi / 2.0
            return cx, cy, max(w, h), max(min(w, h), 1e-3), angle % math.pi

        bbox = ann.get("bbox")
        if not bbox:
            return None
        x, y, w, h = (float(v) for v in bbox)
        angle = 0.0 if w >= h else math.pi / 2.0
        return x + w / 2.0, y + h / 2.0, max(w, h), max(min(w, h), 1e-3), angle

    def _line_intersects_input(
        self,
        cx: float,
        cy: float,
        length: float,
        angle: float,
        centerline_width: float,
        centerline_sigma: float,
    ) -> bool:
        radius = length / 2.0 + 4.0 * centerline_sigma + centerline_width
        return -radius <= cx <= self.image_size + radius and -radius <= cy <= self.image_size + radius

    def _rasterise_centerline(
        self,
        target: np.ndarray,
        cx: float,
        cy: float,
        length: float,
        width: float,
        angle: float,
        centerline_width: float,
        centerline_sigma: float,
    ) -> None:
        ux = math.cos(angle)
        uy = math.sin(angle)
        radius = int(math.ceil(max(width, centerline_width) + 4.0 * centerline_sigma))
        x1 = max(0, int(math.floor(cx - abs(ux) * length / 2.0 - radius)))
        x2 = min(self.image_size, int(math.ceil(cx + abs(ux) * length / 2.0 + radius)))
        y1 = max(0, int(math.floor(cy - abs(uy) * length / 2.0 - radius)))
        y2 = min(self.image_size, int(math.ceil(cy + abs(uy) * length / 2.0 + radius)))
        if x1 >= x2 or y1 >= y2:
            return
        yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)
        dx = xx - cx
        dy = yy - cy
        along = dx * ux + dy * uy
        across = np.abs(-dx * uy + dy * ux)
        support = (np.abs(along) <= length / 2.0) & (across <= width / 2.0 + 3.0 * centerline_sigma)
        values = np.exp(-(across**2) / (2.0 * centerline_sigma**2)).astype(np.float32)
        values[~support] = 0.0
        bin_idx = int(round((angle % math.pi) / math.pi * self.orientation_bins)) % self.orientation_bins
        for offset, weight in (
            (0, 1.0),
            (-1, self.neighbor_bin_weight),
            (1, self.neighbor_bin_weight),
            (-2, self.second_neighbor_weight),
            (2, self.second_neighbor_weight),
        ):
            if weight <= 0.0:
                continue
            channel = (bin_idx + offset) % self.orientation_bins
            patch = target[channel, y1:y2, x1:x2]
            np.maximum(patch, values * weight, out=patch)


def collate_centerline_batch(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, Any]:
    """Collate centerline samples into a batch dict."""
    return {
        "image": torch.stack([item["image"] for item in batch]),  # type: ignore[arg-type]
        "target": torch.stack([item["target"] for item in batch]),  # type: ignore[arg-type]
        "catchment_target": torch.stack([item["catchment_target"] for item in batch]),  # type: ignore[arg-type]
        "positive": torch.stack([item["positive"] for item in batch]),  # type: ignore[arg-type]
        "image_id": torch.stack([item["image_id"] for item in batch]),  # type: ignore[arg-type]
        "file_name": [str(item["file_name"]) for item in batch],
    }
