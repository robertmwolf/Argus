"""Augmentation pipelines for StreakMind training.

Provides training and validation transforms via albumentations, plus a
custom synthetic streak injection transform for class balancing.

# Source: StreakMind — augmentation pipeline for streak detection training
# Ref: StreakMind paper/repo (cite per published source)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

try:
    import albumentations as A
    from albumentations.core.transforms_interface import DualTransform
except ImportError as e:
    raise ImportError(
        "albumentations required: pip install albumentations>=1.3.0"
    ) from e

logger = logging.getLogger(__name__)


def get_train_transforms() -> A.Compose:
    """Return training augmentation pipeline.

    # Source: StreakMind — augmentation pipeline for streak detection training
    # Ref: StreakMind paper/repo (cite per published source)

    Transform order:
      1. HorizontalFlip(p=0.5)
      2. VerticalFlip(p=0.5)
      3. RandomRotate90(p=0.5)
      4. ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=180, p=0.8)
      5. RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7)
      6. GaussNoise(var_limit=(10, 50), p=0.5)
      7. Blur(blur_limit=3, p=0.3)

    Returns:
        An albumentations Compose pipeline with pascal_voc bbox_params.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=180,
                p=0.8,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.7,
            ),
            A.GaussNoise(std_range=(0.02, 0.1), p=0.5),
            A.Blur(blur_limit=3, p=0.3),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


def get_val_transforms() -> A.Compose:
    """Return validation transform pipeline (identity — no augmentation).

    Returns:
        An albumentations Compose pipeline with pascal_voc bbox_params
        that applies no augmentations.
    """
    return A.Compose(
        [],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


class SyntheticStreakInject(DualTransform):
    """Inject 1–3 synthetic linear streaks into an astronomical image.

    Streaks simulate satellite trails for class balancing during training.
    Each streak has a random angle, length, brightness, and Gaussian
    cross-section profile.

    # Source: StreakMind — synthetic streak injection for class balancing
    # Ref: StreakMind paper/repo (cite per published source)

    Per streak:
      - angle: uniform [0, 180)°
      - length: uniform [50px, image_diagonal]
      - brightness: mean(image) + uniform(1, 5) * std(image)
      - cross-section: Gaussian profile sigma=1.5px via cv2.GaussianBlur
      - start pos: random, ensuring full streak stays in-frame

    Usage:
        Use inject() for direct image+bbox modification (recommended).
        When used inside an albumentations Compose, only the image pixels
        are modified; use inject() to also get updated bboxes/labels.

    Args:
        p: Probability of applying the transform.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)

    # ------------------------------------------------------------------
    # Public interface: direct injection (handles image + bboxes together)
    # ------------------------------------------------------------------

    def inject(
        self,
        image: np.ndarray,
        bboxes: list[tuple[float, float, float, float]],
        labels: list[int],
    ) -> tuple[np.ndarray, list[tuple[float, float, float, float]], list[int]]:
        """Apply synthetic streak injection directly to image and bbox lists.

        This method bypasses albumentations' internal bbox pipeline and is
        the recommended way to use this transform in training loops.

        Args:
            image: uint8 image array (H, W, 3) or (H, W).
            bboxes: List of (x_min, y_min, x_max, y_max) in pixel coords.
            labels: List of integer class labels (same length as bboxes).

        Returns:
            Tuple of (modified_image, new_bboxes, new_labels) where
            new_bboxes and new_labels include any injected streaks (class 0).
        """
        import random as _random
        if not _random.random() < self.p:
            return image, bboxes, labels

        streaks = self._generate_streaks(image)
        modified = self._draw_streaks(image, streaks)

        new_bboxes = list(bboxes)
        new_labels = list(labels)
        h, w = image.shape[:2]
        for x0, y0, x1, y1, _ in streaks:
            x_min = max(0, int(min(x0, x1)))
            y_min = max(0, int(min(y0, y1)))
            x_max = min(w, int(max(x0, x1)) + 1)
            y_max = min(h, int(max(y0, y1)) + 1)
            if x_max > x_min and y_max > y_min:
                new_bboxes.append((float(x_min), float(y_min), float(x_max), float(y_max)))
                new_labels.append(0)

        return modified, new_bboxes, new_labels

    # ------------------------------------------------------------------
    # albumentations DualTransform interface (image-only inside Compose)
    # ------------------------------------------------------------------

    def apply(self, image: np.ndarray, **params: Any) -> np.ndarray:
        """Draw synthetic streaks on the image.

        Args:
            image: Input uint8 image array (H, W, C) or (H, W).
            **params: Includes injected_streaks from get_params_dependent_on_data.

        Returns:
            Image with streaks drawn, same shape and dtype.
        """
        streaks = params.get("injected_streaks", [])
        return self._draw_streaks(image, streaks)

    def apply_to_bboxes(self, bboxes: Any, **params: Any) -> Any:
        """Pass through bboxes unchanged (bbox addition via inject())."""
        return bboxes

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate random streak parameters from image dimensions.

        Args:
            params: Current transform parameters.
            data: Transform data dict containing the image.

        Returns:
            Dict with 'injected_streaks': list of (x0, y0, x1, y1, brightness).
        """
        image = data["image"]
        streaks = self._generate_streaks(image)
        return {"injected_streaks": streaks}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Return names of init args for serialisation."""
        return ("p",)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_streaks(
        self, image: np.ndarray
    ) -> list[tuple[float, float, float, float, float]]:
        """Generate 1–3 random streak parameter tuples for the given image.

        Args:
            image: Image array used for size and brightness statistics.

        Returns:
            List of (x0, y0, x1, y1, brightness) tuples.
        """
        h, w = image.shape[:2]
        diagonal = float(np.sqrt(h ** 2 + w ** 2))
        rng = np.random.default_rng()
        n_streaks = int(rng.integers(1, 4))

        finite_px = image[np.isfinite(image)]
        if finite_px.size == 0:
            img_mean, img_std = 128.0, 20.0
        else:
            img_mean = float(finite_px.mean())
            img_std = float(finite_px.std()) if finite_px.std() > 0 else 20.0

        streaks = []
        for _ in range(n_streaks):
            angle_rad = np.deg2rad(float(rng.uniform(0, 180)))
            length = float(rng.uniform(50, diagonal))
            brightness = float(np.clip(img_mean + float(rng.uniform(1, 5)) * img_std, 0, 255))
            dx = np.cos(angle_rad) * length / 2.0
            dy = np.sin(angle_rad) * length / 2.0
            margin_x, margin_y = abs(dx) + 1, abs(dy) + 1
            if 2 * margin_x >= w or 2 * margin_y >= h:
                cx, cy = w / 2.0, h / 2.0
            else:
                cx = float(rng.uniform(margin_x, w - margin_x))
                cy = float(rng.uniform(margin_y, h - margin_y))
            streaks.append((cx - dx, cy - dy, cx + dx, cy + dy, brightness))
        return streaks

    def _draw_streaks(
        self, image: np.ndarray, streaks: list[tuple]
    ) -> np.ndarray:
        """Draw streak lines with Gaussian cross-section onto image.

        Args:
            image: Source image array.
            streaks: List of (x0, y0, x1, y1, brightness) tuples.

        Returns:
            Modified image with same shape and dtype as input.
        """
        img = image.copy().astype(np.float32)
        is_gray = img.ndim == 2
        for x0, y0, x1, y1, brightness in streaks:
            canvas = np.zeros_like(img)
            color = float(brightness) if is_gray else (float(brightness),) * img.shape[2]
            cv2.line(canvas, (int(x0), int(y0)), (int(x1), int(y1)), color, thickness=1)
            canvas = cv2.GaussianBlur(canvas, (5, 5), sigmaX=1.5)
            img = np.clip(img + canvas, 0, 255)
        return img.astype(image.dtype)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Quick smoke test on a synthetic image
    rng = np.random.default_rng(1)
    image = rng.integers(50, 200, (256, 256, 3), dtype=np.uint8)
    bboxes: list[tuple[float, float, float, float]] = []
    labels: list[int] = []

    train_tf = get_train_transforms()
    out = train_tf(image=image, bboxes=bboxes, labels=labels)
    print(f"Training transform output shape: {out['image'].shape}")

    inject = SyntheticStreakInject(p=1.0)
    inject_tf = A.Compose(
        [inject],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )
    out2 = inject_tf(image=image, bboxes=[], labels=[])
    print(f"Streak inject output shape: {out2['image'].shape}")
    print(f"Injected bboxes: {len(out2['bboxes'])}")
