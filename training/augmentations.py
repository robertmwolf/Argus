"""Augmentation pipelines for ARGUS training.

Provides training and validation transforms via albumentations, plus a
custom synthetic streak injection transform for class balancing.

# Source: StreakMind — augmentation pipeline for streak detection training
# Ref: StreakMind paper/repo (cite per published source)
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class SyntheticStreakGeometry:
    """Visible geometry and rendering parameters for one synthetic streak."""

    x0: float
    y0: float
    x1: float
    y1: float
    brightness: float
    brightness_level: int
    width_px: float

    @property
    def length_px(self) -> float:
        return float(np.hypot(self.x1 - self.x0, self.y1 - self.y0))

    @property
    def angle_deg(self) -> float:
        return float(np.degrees(np.arctan2(self.y1 - self.y0, self.x1 - self.x0)) % 180.0)

    @property
    def obb(self) -> dict[str, float]:
        return {
            "cx": (self.x0 + self.x1) / 2.0,
            "cy": (self.y0 + self.y1) / 2.0,
            "w": self.length_px,
            "h": self.width_px,
            "angle_deg": self.angle_deg,
        }

    @property
    def render_tuple(self) -> tuple[float, float, float, float, float]:
        return self.x0, self.y0, self.x1, self.y1, self.brightness

    def segmentation(self) -> list[float]:
        """Return the OBB polygon as COCO's flat 8-value segmentation."""
        angle_rad = np.deg2rad(self.angle_deg)
        cos_a = float(np.cos(angle_rad))
        sin_a = float(np.sin(angle_rad))
        cx = (self.x0 + self.x1) / 2.0
        cy = (self.y0 + self.y1) / 2.0
        half_w = self.length_px / 2.0
        half_h = self.width_px / 2.0
        corners = [
            (cx + half_w * cos_a - half_h * sin_a, cy + half_w * sin_a + half_h * cos_a),
            (cx - half_w * cos_a - half_h * sin_a, cy - half_w * sin_a + half_h * cos_a),
            (cx - half_w * cos_a + half_h * sin_a, cy - half_w * sin_a - half_h * cos_a),
            (cx + half_w * cos_a + half_h * sin_a, cy + half_w * sin_a - half_h * cos_a),
        ]
        return [float(coord) for point in corners for coord in point]

    def bbox(self) -> list[float]:
        """Return the COCO axis-aligned bbox enclosing the OBB polygon."""
        poly = self.segmentation()
        xs = poly[0::2]
        ys = poly[1::2]
        x_min = min(xs)
        y_min = min(ys)
        return [float(x_min), float(y_min), float(max(xs) - x_min), float(max(ys) - y_min)]


def get_train_transforms() -> A.Compose:
    """Return training augmentation pipeline.

    # Source: StreakMind — augmentation pipeline for streak detection training
    # Ref: StreakMind paper/repo (cite per published source)

    Transform order:
      1. HorizontalFlip(p=0.5)
      2. VerticalFlip(p=0.5)
      3. RandomRotate90(p=0.5)
      4. ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=180, p=0.8)
      5. RandomScale(scale_limit=(-0.25, 0.25), p=0.5)          [cross-scope pixel scale jitter]
      6. RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7)
      7. GaussNoise(std_range=(0.02, 0.1), p=0.5)
      8. GaussianBlur(sigma_limit=(0.5, 2.0), p=0.5)             [PSF / seeing / focus variation]
      9. Blur(blur_limit=3, p=0.3)

    Cross-scope generalisation additions (items 5 and 8):

    RandomScale ±25% simulates pixel scales from 0.95 to 1.6 arcsec/px around
    Atwood's 1.27 arcsec/px baseline, making the model robust to instruments
    with slightly different optics.  (Does not change the tile size — the tile
    is resized and then re-cropped to the original dimensions by albumentations.)

    GaussianBlur σ 0.5–2.0 px simulates variation in atmospheric seeing (0.5–3"),
    PSF quality, and focus accuracy across different scopes and nights.

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
            # Cross-scope pixel scale jitter: simulates 0.95–1.6 arcsec/px instruments
            A.RandomScale(scale_limit=(-0.25, 0.25), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.7,
            ),
            A.GaussNoise(std_range=(0.02, 0.1), p=0.5),
            # PSF / seeing / focus blur: simulates different scopes and observing conditions
            A.GaussianBlur(blur_limit=0, sigma_limit=(0.5, 2.0), p=0.5),
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
      - brightness: snr_scale × (mean(image) + uniform(1, 5) × std(image))
      - cross-section: Gaussian profile sigma=1.5px via cv2.GaussianBlur
      - start pos: random, ensuring full streak stays in-frame

    Usage:
        Use inject() for direct image+bbox modification (recommended).
        When used inside an albumentations Compose, only the image pixels
        are modified; use inject() to also get updated bboxes/labels.

    Args:
        p: Probability of applying the transform.
        min_length_px: Minimum streak length in pixels.
        max_length_fraction: Maximum streak length as a fraction of image diagonal.
        snr_scale: Brightness multiplier applied to injected streaks (0.0–1.0).
            1.0 = full brightness (default, existing behaviour).
            0.1–0.3 = near-threshold faint streaks that target the long-band
            false-negative gap identified in Run 3 (49 missed long streaks).
            Values below ~0.05 produce streaks that are invisible against noise
            and should be avoided.
    """

    def __init__(
        self,
        p: float = 0.5,
        min_length_px: float = 50.0,
        max_length_fraction: float = 1.0,
        snr_scale: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.min_length_px = min_length_px
        self.max_length_fraction = max_length_fraction
        if not (0.0 < snr_scale <= 1.0):
            raise ValueError(
                f"snr_scale must be in (0, 1]; got {snr_scale}.  "
                "Use values in 0.1–0.3 for faint-streak injection."
            )
        self.snr_scale = float(snr_scale)

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

    def inject_with_geometry(
        self,
        image: np.ndarray,
        rng: np.random.Generator | None = None,
        n_streaks: int | None = None,
        min_length_px: float | None = None,
        max_length_px: float | None = None,
        angle_choices_deg: Sequence[float] | None = None,
        brightness_level: int | None = None,
        width_px: float = 16.0,
        full_crossing_probability: float = 0.25,
    ) -> tuple[np.ndarray, list[SyntheticStreakGeometry]]:
        """Draw synthetic streaks and return exact visible OBB geometry.

        This is the dataset-building path used by the GTImages reproduction
        pipeline. It keeps the older ``inject()`` API intact while exposing
        true line endpoints, angle, length, and brightness provenance for COCO
        annotations.

        Args:
            image: uint8 image array, shape ``(H, W, C)`` or ``(H, W)``.
            rng: NumPy generator for deterministic dataset creation.
            n_streaks: Number of streaks to draw. Defaults to 1–3.
            min_length_px: Minimum requested streak length.
            max_length_px: Maximum requested streak length.
            angle_choices_deg: Optional observed-angle population to sample from.
            brightness_level: Optional level 1–5. If omitted, sampled uniformly.
            width_px: OBB short-axis width in pixels.
            full_crossing_probability: Chance that a streak is deliberately
                made longer than the frame diagonal.

        Returns:
            Tuple of modified image and visible streak geometries.
        """
        rng = rng or np.random.default_rng()
        h_img, w_img = image.shape[:2]
        diagonal = float(np.hypot(h_img, w_img))
        min_len = float(min_length_px if min_length_px is not None else self.min_length_px)
        max_len = float(max_length_px if max_length_px is not None else self.max_length_fraction * diagonal)
        max_len = max(min_len + 1.0, min(max_len, diagonal * 1.5))
        count = n_streaks if n_streaks is not None else int(rng.integers(1, 4))

        finite_px = image[np.isfinite(image)]
        if finite_px.size == 0:
            img_mean, img_std = 128.0, 20.0
        else:
            img_mean = float(finite_px.mean())
            img_std = float(finite_px.std()) if finite_px.std() > 0 else 20.0

        geometries: list[SyntheticStreakGeometry] = []
        for _ in range(count):
            for _attempt in range(100):
                if angle_choices_deg is not None and len(angle_choices_deg) > 0:
                    angle_deg = float(rng.choice(np.asarray(angle_choices_deg, dtype=np.float64)))
                else:
                    angle_deg = float(rng.uniform(0.0, 180.0))
                angle_rad = float(np.deg2rad(angle_deg))

                requested_len = float(rng.uniform(min_len, max_len))
                if rng.random() < full_crossing_probability:
                    requested_len = max(requested_len, diagonal * float(rng.uniform(1.05, 1.35)))

                dx = float(np.cos(angle_rad) * requested_len / 2.0)
                dy = float(np.sin(angle_rad) * requested_len / 2.0)
                margin = requested_len * 0.2
                cx = float(rng.uniform(-margin, w_img + margin))
                cy = float(rng.uniform(-margin, h_img + margin))

                clipped = self._clip_segment_to_image(
                    cx - dx, cy - dy, cx + dx, cy + dy, w_img, h_img
                )
                if clipped is None:
                    continue
                x0, y0, x1, y1 = clipped
                visible_len = float(np.hypot(x1 - x0, y1 - y0))
                if visible_len < max(8.0, min_len * 0.25):
                    continue

                level = int(brightness_level if brightness_level is not None else rng.integers(1, 6))
                level = max(1, min(5, level))
                brightness = float(
                    np.clip(self.snr_scale * (0.65 + 0.55 * level) * img_std, 8.0, 255.0)
                )
                geometries.append(
                    SyntheticStreakGeometry(
                        x0=float(x0),
                        y0=float(y0),
                        x1=float(x1),
                        y1=float(y1),
                        brightness=brightness,
                        brightness_level=level,
                        width_px=float(width_px),
                    )
                )
                break

        modified = self._draw_streaks(image, [g.render_tuple for g in geometries])
        return modified, geometries

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
        return ("p", "min_length_px", "max_length_fraction", "snr_scale")

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

        max_length = min(self.max_length_fraction * diagonal, diagonal)
        min_length = max(self.min_length_px, 1.0)

        streaks = []
        for _ in range(n_streaks):
            angle_rad = np.deg2rad(float(rng.uniform(0, 180)))
            length = float(rng.uniform(min_length, max(min_length + 1.0, max_length)))
            brightness = float(
                np.clip(
                    self.snr_scale * (img_mean + float(rng.uniform(1, 5)) * img_std),
                    0, 255,
                )
            )
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

    @staticmethod
    def _clip_segment_to_image(
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        width: int,
        height: int,
    ) -> tuple[float, float, float, float] | None:
        """Clip a line segment to image bounds using Liang-Barsky clipping."""
        x_min, y_min = 0.0, 0.0
        x_max, y_max = float(width - 1), float(height - 1)
        dx = x1 - x0
        dy = y1 - y0
        p = [-dx, dx, -dy, dy]
        q = [x0 - x_min, x_max - x0, y0 - y_min, y_max - y0]
        u1, u2 = 0.0, 1.0

        for pi, qi in zip(p, q):
            if abs(pi) < 1e-12:
                if qi < 0:
                    return None
                continue
            t = qi / pi
            if pi < 0:
                u1 = max(u1, t)
            else:
                u2 = min(u2, t)
            if u1 > u2:
                return None

        return x0 + u1 * dx, y0 + u1 * dy, x0 + u2 * dx, y0 + u2 * dy

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
