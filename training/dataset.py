"""PyTorch Dataset for FITS-based satellite streak detection.

Loads FITS images via FITSLoader and wraps COCO-format annotations for
use with PyTorch training loops.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as e:
    raise ImportError("torch required: pip install torch>=2.2.0") from e

from inference.fits_loader import FITSLoader

logger = logging.getLogger(__name__)


class FITSStreakDataset(Dataset):
    """PyTorch Dataset loading FITS files via FITSLoader.

    Reads a COCO-format annotation JSON and loads source FITS files on
    demand. Never loads from cached PNG — always reads from FITS.

    Args:
        annotation_file: Path to a COCO-format JSON annotation file.
        transforms: Optional callable applied to (image_array, target).
    """

    def __init__(self, annotation_file: str | Path, transforms: Any = None) -> None:
        """Load COCO JSON; build image_id→meta and image_id→annotations lookups.

        Args:
            annotation_file: Path to COCO JSON file.
            transforms: Optional transform applied to (image_tensor, target).
        """
        self.annotation_file = Path(annotation_file)
        self.transforms = transforms
        self._loader = FITSLoader()

        with open(self.annotation_file) as f:
            coco = json.load(f)

        self._images: list[dict] = coco.get("images", [])
        self._id_to_meta: dict[int, dict] = {
            img["id"]: img for img in self._images
        }
        self._id_to_anns: dict[int, list[dict]] = defaultdict(list)
        for ann in coco.get("annotations", []):
            self._id_to_anns[ann["image_id"]].append(ann)

    def __len__(self) -> int:
        """Return number of images in the dataset."""
        return len(self._images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Load FITS image and build target dict.

        Image tensor: float32, shape (3, H, W), values in [0, 1].

        Target dict:
          boxes      — FloatTensor [N, 4] as [x1, y1, x2, y2]
          labels     — LongTensor  [N] all zeros (single class: streak)
          image_id   — IntTensor scalar
          obb_params — FloatTensor [N, 5] as [cx, cy, w, h, angle_deg]

        FITS is searched relative to annotation_file's parent directory.
        If FITS is not found: logs a warning and returns a zero tensor with
        an empty target dict (no crash).

        Args:
            idx: Dataset index.

        Returns:
            Tuple of (image_tensor, target_dict).
        """
        meta = self._images[idx]
        img_id: int = meta["id"]
        file_name: str = meta["file_name"]

        fits_path = self._resolve_image_path(file_name)
        h: int = meta.get("height", 0)
        w: int = meta.get("width", 0)

        # --- Load image ---
        if not fits_path.exists():
            logger.warning(
                "FITS file not found: %s — returning zero tensor for idx=%d",
                fits_path,
                idx,
            )
            zero_tensor = torch.zeros((3, max(h, 1), max(w, 1)), dtype=torch.float32)
            empty_target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "image_id": torch.tensor(img_id, dtype=torch.int32),
                "obb_params": torch.zeros((0, 5), dtype=torch.float32),
            }
            return zero_tensor, empty_target

        try:
            loaded = self._loader.load(fits_path)
        except Exception as exc:
            logger.warning(
                "Failed to load FITS %s: %s — returning zero tensor for idx=%d",
                fits_path,
                exc,
                idx,
            )
            zero_tensor = torch.zeros((3, max(h, 1), max(w, 1)), dtype=torch.float32)
            empty_target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "image_id": torch.tensor(img_id, dtype=torch.int32),
                "obb_params": torch.zeros((0, 5), dtype=torch.float32),
            }
            return zero_tensor, empty_target

        # Convert uint8 (H, W, 3) → float32 (3, H, W) in [0, 1]
        array = loaded["array"]  # uint8 (H, W, 3)
        img_tensor = torch.from_numpy(
            array.transpose(2, 0, 1).astype(np.float32) / 255.0
        )

        # --- Build target ---
        anns = self._id_to_anns.get(img_id, [])
        boxes_list: list[list[float]] = []
        obb_list: list[list[float]] = []

        for ann in anns:
            x1, y1, bw, bh = ann["bbox"]
            boxes_list.append([x1, y1, x1 + bw, y1 + bh])
            obb = ann["obb"]
            if isinstance(obb, dict):
                obb_list.append([
                    obb["cx"],
                    obb["cy"],
                    obb["w"],
                    obb["h"],
                    obb["angle_deg"],
                ])
            else:
                obb_list.append(obb)

        n = len(boxes_list)
        boxes = torch.tensor(boxes_list, dtype=torch.float32).reshape(n, 4)
        labels = torch.zeros(n, dtype=torch.int64)
        obb_params = torch.tensor(obb_list, dtype=torch.float32).reshape(n, 5)

        target: dict[str, torch.Tensor] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(img_id, dtype=torch.int32),
            "obb_params": obb_params,
        }

        if self.transforms is not None:
            img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target

    def _resolve_image_path(self, file_name: str) -> Path:
        """Resolve an image path from COCO metadata.

        COCO files in this project are used in two contexts:
        generated dev subsets store paths relative to ``data/annotations/``,
        while merged training splits store paths relative to ``data/``.  Older
        GTImages conversion outputs used bare FITS filenames, so keep a final
        fallback to ``data/GTImages`` for compatibility.

        Args:
            file_name: COCO ``images[].file_name`` value.

        Returns:
            Existing image path when found, otherwise the annotation-relative
            candidate so callers get the usual missing-file warning.
        """
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


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python dataset.py <coco_annotation.json>")
        sys.exit(1)

    ann_file = Path(sys.argv[1])
    ds = FITSStreakDataset(ann_file)
    print(f"Dataset loaded: {len(ds)} images")

    if len(ds) > 0:
        img, tgt = ds[0]
        print(f"  image tensor shape : {img.shape}")
        print(f"  image dtype        : {img.dtype}")
        print(f"  boxes              : {tgt['boxes'].shape}")
        print(f"  labels             : {tgt['labels']}")
        print(f"  obb_params         : {tgt['obb_params'].shape}")
