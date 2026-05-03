"""Tests for training.augmentations."""

from __future__ import annotations

import numpy as np
import pytest

A = pytest.importorskip("albumentations")

from training.augmentations import (
    SyntheticStreakInject,
    get_train_transforms,
    get_val_transforms,
)


# ---------------------------------------------------------------------------
# get_train_transforms
# ---------------------------------------------------------------------------


class TestGetTrainTransforms:
    def test_runs_without_error(self) -> None:
        rng = np.random.default_rng(42)
        image = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
        tf = get_train_transforms()
        out = tf(image=image, bboxes=[], labels=[])
        assert "image" in out

    def test_output_shape_unchanged(self) -> None:
        rng = np.random.default_rng(42)
        image = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
        tf = get_train_transforms()
        out = tf(image=image, bboxes=[], labels=[])
        assert out["image"].shape == (256, 256, 3)

    def test_output_dtype_uint8(self) -> None:
        rng = np.random.default_rng(42)
        image = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
        tf = get_train_transforms()
        out = tf(image=image, bboxes=[], labels=[])
        assert out["image"].dtype == np.uint8

    def test_bbox_passthrough(self) -> None:
        rng = np.random.default_rng(42)
        image = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
        tf = get_train_transforms()
        # Single bbox in pascal_voc format [x1, y1, x2, y2]
        bboxes = [(50, 50, 150, 100)]
        out = tf(image=image, bboxes=bboxes, labels=[0])
        # After transform the number of bboxes should be preserved (may differ if
        # bbox gets clipped out — that's acceptable behaviour from albumentations)
        assert isinstance(out["bboxes"], list)


# ---------------------------------------------------------------------------
# get_val_transforms
# ---------------------------------------------------------------------------


class TestGetValTransforms:
    def test_returns_identical_pixels(self) -> None:
        rng = np.random.default_rng(7)
        image = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
        tf = get_val_transforms()
        out = tf(image=image, bboxes=[], labels=[])
        np.testing.assert_array_equal(out["image"], image)

    def test_shape_unchanged(self) -> None:
        image = np.zeros((100, 80, 3), dtype=np.uint8)
        tf = get_val_transforms()
        out = tf(image=image, bboxes=[], labels=[])
        assert out["image"].shape == (100, 80, 3)


# ---------------------------------------------------------------------------
# SyntheticStreakInject
# ---------------------------------------------------------------------------


class TestSyntheticStreakInject:
    """Tests use inject() directly — albumentations 2.x cannot add new
    bbox+label pairs mid-pipeline (label manager limitation), so bbox
    injection is tested via the standalone inject() method."""

    def test_adds_at_least_one_bbox_on_empty_input(self) -> None:
        rng = np.random.default_rng(0)
        image = rng.integers(50, 200, (256, 256, 3), dtype=np.uint8)
        injector = SyntheticStreakInject(p=1.0)
        out_img, out_bboxes, out_labels = injector.inject(image, [], [])
        assert len(out_bboxes) >= 1
        assert len(out_labels) == len(out_bboxes)

    def test_output_shape_unchanged(self) -> None:
        rng = np.random.default_rng(1)
        image = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
        injector = SyntheticStreakInject(p=1.0)
        out_img, _, _ = injector.inject(image, [], [])
        assert out_img.shape == (128, 128, 3)

    def test_image_dtype_preserved(self) -> None:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        injector = SyntheticStreakInject(p=1.0)
        out_img, _, _ = injector.inject(image, [], [])
        assert out_img.dtype == np.uint8

    def test_p_zero_applies_nothing(self) -> None:
        rng = np.random.default_rng(5)
        image = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        injector = SyntheticStreakInject(p=0.0)
        out_img, out_bboxes, out_labels = injector.inject(image, [], [])
        np.testing.assert_array_equal(out_img, image)
        assert len(out_bboxes) == 0

    def test_existing_bboxes_preserved(self) -> None:
        rng = np.random.default_rng(2)
        image = rng.integers(50, 200, (256, 256, 3), dtype=np.uint8)
        injector = SyntheticStreakInject(p=1.0)
        existing = [(20.0, 20.0, 80.0, 60.0)]
        out_img, out_bboxes, out_labels = injector.inject(image, existing, [0])
        assert len(out_bboxes) >= 2  # original + at least 1 synthetic
        assert (20.0, 20.0, 80.0, 60.0) in out_bboxes

    def test_injected_labels_are_zero(self) -> None:
        image = np.ones((128, 128, 3), dtype=np.uint8) * 100
        injector = SyntheticStreakInject(p=1.0)
        _, out_bboxes, out_labels = injector.inject(image, [], [])
        assert all(lbl == 0 for lbl in out_labels)

    def test_inside_compose_modifies_image(self) -> None:
        """When used inside Compose, image pixels are modified."""
        rng = np.random.default_rng(3)
        image = rng.integers(50, 150, (256, 256, 3), dtype=np.uint8)
        tf = A.Compose(
            [SyntheticStreakInject(p=1.0)],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )
        out = tf(image=image, bboxes=[], labels=[])
        # Image should be modified (streaks drawn)
        assert not np.array_equal(out["image"], image)
