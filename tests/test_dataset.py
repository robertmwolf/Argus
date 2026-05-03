"""Tests for training.dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from astropy.io import fits

from training.dataset import FITSStreakDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fits(directory: Path, stem: str, width: int = 64, height: int = 64) -> Path:
    rng = np.random.default_rng(0)
    data = rng.normal(1000.0, 100.0, (height, width)).astype(np.float32)
    hdu = fits.PrimaryHDU(data)
    hdu.header["NAXIS1"] = width
    hdu.header["NAXIS2"] = height
    out = directory / f"{stem}.fits"
    hdu.writeto(out, overwrite=True)
    return out


def _make_coco_json(
    directory: Path,
    images: list[dict],
    annotations: list[dict],
    out_name: str = "coco.json",
) -> Path:
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": "streak"}],
    }
    out = directory / out_name
    out.write_text(json.dumps(coco))
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFITSStreakDatasetLen:
    def test_len_matches_image_count(self, tmp_path: Path) -> None:
        _make_fits(tmp_path, "img1")
        _make_fits(tmp_path, "img2")
        coco = _make_coco_json(
            tmp_path,
            images=[
                {"id": 1, "file_name": "img1.fits", "width": 64, "height": 64},
                {"id": 2, "file_name": "img2.fits", "width": 64, "height": 64},
            ],
            annotations=[],
        )
        ds = FITSStreakDataset(coco)
        assert len(ds) == 2

    def test_len_zero_for_empty_json(self, tmp_path: Path) -> None:
        coco = _make_coco_json(tmp_path, images=[], annotations=[])
        ds = FITSStreakDataset(coco)
        assert len(ds) == 0


class TestFITSStreakDatasetGetItem:
    def test_returns_tensor_and_dict(self, tmp_path: Path) -> None:
        _make_fits(tmp_path, "img1")
        coco = _make_coco_json(
            tmp_path,
            images=[{"id": 1, "file_name": "img1.fits", "width": 64, "height": 64}],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 0,
                    "bbox": [10.0, 10.0, 20.0, 5.0],
                    "area": 100.0,
                    "obb": [20.0, 12.5, 20.0, 5.0, 0.0],
                    "iscrowd": 0,
                }
            ],
        )
        ds = FITSStreakDataset(coco)
        img, tgt = ds[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(tgt, dict)

    def test_image_tensor_shape_3_h_w(self, tmp_path: Path) -> None:
        _make_fits(tmp_path, "img1", width=64, height=48)
        coco = _make_coco_json(
            tmp_path,
            images=[{"id": 1, "file_name": "img1.fits", "width": 64, "height": 48}],
            annotations=[],
        )
        ds = FITSStreakDataset(coco)
        img, _ = ds[0]
        assert img.shape == (3, 48, 64)

    def test_image_tensor_dtype_float32(self, tmp_path: Path) -> None:
        _make_fits(tmp_path, "img1")
        coco = _make_coco_json(
            tmp_path,
            images=[{"id": 1, "file_name": "img1.fits", "width": 64, "height": 64}],
            annotations=[],
        )
        ds = FITSStreakDataset(coco)
        img, _ = ds[0]
        assert img.dtype == torch.float32

    def test_image_values_in_0_1(self, tmp_path: Path) -> None:
        _make_fits(tmp_path, "img1")
        coco = _make_coco_json(
            tmp_path,
            images=[{"id": 1, "file_name": "img1.fits", "width": 64, "height": 64}],
            annotations=[],
        )
        ds = FITSStreakDataset(coco)
        img, _ = ds[0]
        assert img.min().item() >= 0.0
        assert img.max().item() <= 1.0

    def test_target_keys_present(self, tmp_path: Path) -> None:
        _make_fits(tmp_path, "img1")
        coco = _make_coco_json(
            tmp_path,
            images=[{"id": 1, "file_name": "img1.fits", "width": 64, "height": 64}],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 0,
                    "bbox": [5.0, 5.0, 10.0, 3.0],
                    "area": 30.0,
                    "obb": [10.0, 6.5, 10.0, 3.0, 0.0],
                    "iscrowd": 0,
                }
            ],
        )
        ds = FITSStreakDataset(coco)
        _, tgt = ds[0]
        assert set(tgt.keys()) == {"boxes", "labels", "image_id", "obb_params"}

    def test_boxes_dtype_float32_shape_n_4(self, tmp_path: Path) -> None:
        _make_fits(tmp_path, "img1")
        coco = _make_coco_json(
            tmp_path,
            images=[{"id": 1, "file_name": "img1.fits", "width": 64, "height": 64}],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 0,
                    "bbox": [5.0, 5.0, 10.0, 3.0],
                    "area": 30.0,
                    "obb": [10.0, 6.5, 10.0, 3.0, 0.0],
                    "iscrowd": 0,
                }
            ],
        )
        ds = FITSStreakDataset(coco)
        _, tgt = ds[0]
        assert tgt["boxes"].dtype == torch.float32
        assert tgt["boxes"].shape == (1, 4)

    def test_labels_dtype_int64_all_zeros(self, tmp_path: Path) -> None:
        _make_fits(tmp_path, "img1")
        coco = _make_coco_json(
            tmp_path,
            images=[{"id": 1, "file_name": "img1.fits", "width": 64, "height": 64}],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 0,
                    "bbox": [5.0, 5.0, 10.0, 3.0],
                    "area": 30.0,
                    "obb": [10.0, 6.5, 10.0, 3.0, 0.0],
                    "iscrowd": 0,
                }
            ],
        )
        ds = FITSStreakDataset(coco)
        _, tgt = ds[0]
        assert tgt["labels"].dtype == torch.int64
        assert (tgt["labels"] == 0).all()

    def test_obb_params_shape_n_5(self, tmp_path: Path) -> None:
        _make_fits(tmp_path, "img1")
        coco = _make_coco_json(
            tmp_path,
            images=[{"id": 1, "file_name": "img1.fits", "width": 64, "height": 64}],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 0,
                    "bbox": [5.0, 5.0, 10.0, 3.0],
                    "area": 30.0,
                    "obb": [10.0, 6.5, 10.0, 3.0, 45.0],
                    "iscrowd": 0,
                }
            ],
        )
        ds = FITSStreakDataset(coco)
        _, tgt = ds[0]
        assert tgt["obb_params"].shape == (1, 5)

    def test_missing_fits_returns_zero_tensor_no_crash(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # No actual FITS file created
        coco = _make_coco_json(
            tmp_path,
            images=[{"id": 1, "file_name": "missing.fits", "width": 32, "height": 32}],
            annotations=[],
        )
        ds = FITSStreakDataset(coco)
        import logging
        with caplog.at_level(logging.WARNING):
            img, tgt = ds[0]
        assert img.dtype == torch.float32
        assert tgt["boxes"].shape[0] == 0
        assert tgt["labels"].shape[0] == 0
        assert tgt["obb_params"].shape[0] == 0
