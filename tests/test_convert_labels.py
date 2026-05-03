"""Tests for training.convert_labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from training.convert_labels import compute_obb_corners, convert_yolo_obb_to_coco


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fits(directory: Path, stem: str, width: int = 128, height: int = 64) -> Path:
    """Create a minimal FITS file with given dims."""
    data = np.zeros((height, width), dtype=np.float32)
    hdu = fits.PrimaryHDU(data)
    hdu.header["NAXIS1"] = width
    hdu.header["NAXIS2"] = height
    out = directory / f"{stem}.fits"
    hdu.writeto(out, overwrite=True)
    return out


def _make_label(directory: Path, stem: str, lines: list[str]) -> Path:
    """Write a YOLO OBB label file."""
    out = directory / f"{stem}.txt"
    out.write_text("\n".join(lines))
    return out


# ---------------------------------------------------------------------------
# compute_obb_corners
# ---------------------------------------------------------------------------


class TestComputeObbCorners:
    def test_returns_shape_4_2(self) -> None:
        corners = compute_obb_corners(50.0, 50.0, 40.0, 10.0, 0.0)
        assert corners.shape == (4, 2)

    def test_axis_aligned_zero_rotation(self) -> None:
        """With angle=0, corners should form an axis-aligned rectangle."""
        cx, cy, w, h = 50.0, 50.0, 40.0, 10.0
        corners = compute_obb_corners(cx, cy, w, h, 0.0)
        xs = corners[:, 0]
        ys = corners[:, 1]
        assert xs.min() == pytest.approx(cx - w / 2, abs=1e-6)
        assert xs.max() == pytest.approx(cx + w / 2, abs=1e-6)
        assert ys.min() == pytest.approx(cy - h / 2, abs=1e-6)
        assert ys.max() == pytest.approx(cy + h / 2, abs=1e-6)

    def test_centre_preserved(self) -> None:
        """The centroid of corners should equal (cx, cy)."""
        cx, cy = 30.0, 70.0
        corners = compute_obb_corners(cx, cy, 20.0, 5.0, 37.0)
        assert corners[:, 0].mean() == pytest.approx(cx, abs=1e-5)
        assert corners[:, 1].mean() == pytest.approx(cy, abs=1e-5)

    def test_90_degree_swaps_dims(self) -> None:
        """At 90° rotation, width and height roles swap in bounding box."""
        w, h = 40.0, 10.0
        corners = compute_obb_corners(50.0, 50.0, w, h, 90.0)
        bbox_w = corners[:, 0].max() - corners[:, 0].min()
        bbox_h = corners[:, 1].max() - corners[:, 1].min()
        # After 90° rotation, the bbox width should be ~ original h
        assert bbox_w == pytest.approx(h, abs=1e-5)
        assert bbox_h == pytest.approx(w, abs=1e-5)


# ---------------------------------------------------------------------------
# convert_yolo_obb_to_coco
# ---------------------------------------------------------------------------


class TestConvertYoloObbToCoco:
    def test_basic_structure(self, tmp_path: Path) -> None:
        label_dir = tmp_path / "labels"
        fits_dir = tmp_path / "fits"
        label_dir.mkdir()
        fits_dir.mkdir()

        _make_fits(fits_dir, "img1", width=200, height=100)
        _make_label(label_dir, "img1", ["0 0.5 0.5 0.3 0.1 15.0"])

        out_json = tmp_path / "coco.json"
        convert_yolo_obb_to_coco(label_dir, fits_dir, out_json)

        data = json.loads(out_json.read_text())
        assert "images" in data
        assert "annotations" in data
        assert "categories" in data

    def test_categories_streak(self, tmp_path: Path) -> None:
        label_dir = tmp_path / "labels"
        fits_dir = tmp_path / "fits"
        label_dir.mkdir()
        fits_dir.mkdir()

        _make_fits(fits_dir, "img1")
        _make_label(label_dir, "img1", ["0 0.5 0.5 0.2 0.1 0.0"])

        out_json = tmp_path / "coco.json"
        convert_yolo_obb_to_coco(label_dir, fits_dir, out_json)

        data = json.loads(out_json.read_text())
        assert data["categories"] == [{"id": 0, "name": "streak"}]

    def test_annotation_has_obb_field(self, tmp_path: Path) -> None:
        label_dir = tmp_path / "labels"
        fits_dir = tmp_path / "fits"
        label_dir.mkdir()
        fits_dir.mkdir()

        _make_fits(fits_dir, "img1", width=200, height=100)
        _make_label(label_dir, "img1", ["0 0.5 0.5 0.3 0.1 45.0"])

        out_json = tmp_path / "coco.json"
        convert_yolo_obb_to_coco(label_dir, fits_dir, out_json)

        data = json.loads(out_json.read_text())
        ann = data["annotations"][0]
        assert "obb" in ann
        assert len(ann["obb"]) == 5

    def test_bbox_in_pixel_space(self, tmp_path: Path) -> None:
        """bbox values must be > 1 (pixel-space, not normalised 0-1)."""
        label_dir = tmp_path / "labels"
        fits_dir = tmp_path / "fits"
        label_dir.mkdir()
        fits_dir.mkdir()

        _make_fits(fits_dir, "img1", width=200, height=100)
        # cx=0.5 → 100px, w=0.3 → 60px: bbox should be in tens-of-pixels range
        _make_label(label_dir, "img1", ["0 0.5 0.5 0.3 0.1 0.0"])

        out_json = tmp_path / "coco.json"
        convert_yolo_obb_to_coco(label_dir, fits_dir, out_json)

        data = json.loads(out_json.read_text())
        bbox = data["annotations"][0]["bbox"]
        # x1 in pixel space should be >> 1
        assert bbox[0] > 1.0
        assert bbox[2] > 1.0  # width

    def test_empty_label_dir(self, tmp_path: Path) -> None:
        label_dir = tmp_path / "labels"
        fits_dir = tmp_path / "fits"
        label_dir.mkdir()
        fits_dir.mkdir()

        out_json = tmp_path / "coco.json"
        convert_yolo_obb_to_coco(label_dir, fits_dir, out_json)

        data = json.loads(out_json.read_text())
        assert data["annotations"] == []
        assert data["images"] == []
        assert data["categories"] == [{"id": 0, "name": "streak"}]

    def test_multiple_annotations_per_image(self, tmp_path: Path) -> None:
        label_dir = tmp_path / "labels"
        fits_dir = tmp_path / "fits"
        label_dir.mkdir()
        fits_dir.mkdir()

        _make_fits(fits_dir, "img1", width=200, height=100)
        _make_label(label_dir, "img1", [
            "0 0.3 0.3 0.2 0.1 10.0",
            "0 0.7 0.7 0.2 0.1 80.0",
        ])

        out_json = tmp_path / "coco.json"
        convert_yolo_obb_to_coco(label_dir, fits_dir, out_json)

        data = json.loads(out_json.read_text())
        assert len(data["annotations"]) == 2

    def test_empty_label_file_no_annotations(self, tmp_path: Path) -> None:
        label_dir = tmp_path / "labels"
        fits_dir = tmp_path / "fits"
        label_dir.mkdir()
        fits_dir.mkdir()

        _make_fits(fits_dir, "img1")
        _make_label(label_dir, "img1", [])

        out_json = tmp_path / "coco.json"
        convert_yolo_obb_to_coco(label_dir, fits_dir, out_json)

        data = json.loads(out_json.read_text())
        assert data["annotations"] == []
        assert len(data["images"]) == 1

    def test_missing_fits_skips_label(self, tmp_path: Path) -> None:
        label_dir = tmp_path / "labels"
        fits_dir = tmp_path / "fits"
        label_dir.mkdir()
        fits_dir.mkdir()

        # Label with no matching FITS
        _make_label(label_dir, "missing_img", ["0 0.5 0.5 0.2 0.1 0.0"])

        out_json = tmp_path / "coco.json"
        convert_yolo_obb_to_coco(label_dir, fits_dir, out_json)

        data = json.loads(out_json.read_text())
        assert data["images"] == []
        assert data["annotations"] == []
