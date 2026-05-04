"""Tests that run against real FITS images dropped into data/test/.

All tests in this file are marked @pytest.mark.real_data and are skipped
automatically when data/test/ contains no .fits/.fit files, so the suite
stays green without any committed images.

To run these tests:
    1. Drop one or more FITS files into data/test/
    2. pytest -m real_data -v

Naming convention (optional — tests use it for tighter assertions):
    *streak*.fits  — image is expected to contain at least one satellite streak
    *blank*.fits   — image is expected to contain no streaks
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import DATA_TEST_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_streak_in_name(p: Path) -> bool:
    return "streak" in p.stem.lower()

def _has_blank_in_name(p: Path) -> bool:
    return "blank" in p.stem.lower() or "empty" in p.stem.lower()


def _load_fits_image(path: Path):
    """Parse a real FITS file into a FITSImage via the production parser."""
    from src.ingest.fits_parser import parse_fits
    return parse_fits(path)


# ---------------------------------------------------------------------------
# Session-scoped skip guard
# ---------------------------------------------------------------------------

def _require_real_images(files: list[Path]) -> None:
    """Skip the entire test if no real images are available."""
    if not files:
        pytest.skip(
            f"No FITS files found in {DATA_TEST_DIR}. "
            "Drop .fits files there to run real-data tests."
        )


# ---------------------------------------------------------------------------
# FITS parsing
# ---------------------------------------------------------------------------

@pytest.mark.real_data
class TestRealFitsParsing:
    """Verify the FITS parser handles real telescope images correctly."""

    def test_all_images_parse_without_error(self, real_fits_files):
        _require_real_images(real_fits_files)
        from src.ingest.fits_parser import parse_fits
        for path in real_fits_files:
            result = parse_fits(path)
            assert result is not None, f"parse_fits returned None for {path.name}"

    def test_image_data_is_2d_float32(self, real_fits_files):
        _require_real_images(real_fits_files)
        from src.ingest.fits_parser import parse_fits
        for path in real_fits_files:
            img = parse_fits(path)
            assert img.data.ndim == 2, \
                f"{path.name}: expected 2D array, got shape {img.data.shape}"
            assert img.data.dtype == np.float32, \
                f"{path.name}: expected float32, got {img.data.dtype}"

    def test_image_dimensions_are_positive(self, real_fits_files):
        _require_real_images(real_fits_files)
        from src.ingest.fits_parser import parse_fits
        for path in real_fits_files:
            img = parse_fits(path)
            assert img.width_px > 0 and img.height_px > 0, \
                f"{path.name}: non-positive dimensions {img.width_px}×{img.height_px}"

    def test_obs_time_is_present(self, real_fits_files):
        _require_real_images(real_fits_files)
        from src.ingest.fits_parser import parse_fits
        for path in real_fits_files:
            img = parse_fits(path)
            assert img.obs_time is not None, \
                f"{path.name}: obs_time is None — DATE-OBS header may be missing"

    def test_pixel_data_has_non_zero_variance(self, real_fits_files):
        _require_real_images(real_fits_files)
        from src.ingest.fits_parser import parse_fits
        for path in real_fits_files:
            img = parse_fits(path)
            assert img.data.std() > 0, \
                f"{path.name}: image data is entirely flat (std=0) — likely a corrupt file"


# ---------------------------------------------------------------------------
# FITS loader (ML tensor conversion)
# ---------------------------------------------------------------------------

@pytest.mark.real_data
class TestRealFitsLoader:
    """Verify FITSLoader produces correctly shaped, normalised tensors."""

    def test_all_images_load_to_tensor(self, real_fits_files):
        _require_real_images(real_fits_files)
        import torch
        from inference.fits_loader import FITSLoader
        loader = FITSLoader()
        for path in real_fits_files:
            tensor, meta = loader.load(path)
            assert isinstance(tensor, torch.Tensor), \
                f"{path.name}: expected Tensor, got {type(tensor)}"

    def test_tensor_is_3d_chw(self, real_fits_files):
        _require_real_images(real_fits_files)
        from inference.fits_loader import FITSLoader
        loader = FITSLoader()
        for path in real_fits_files:
            tensor, _ = loader.load(path)
            assert tensor.ndim == 3, \
                f"{path.name}: expected CHW tensor (3D), got {tensor.ndim}D"
            assert tensor.shape[0] == 1, \
                f"{path.name}: expected 1 channel, got {tensor.shape[0]}"

    def test_tensor_is_normalised(self, real_fits_files):
        _require_real_images(real_fits_files)
        from inference.fits_loader import FITSLoader
        loader = FITSLoader()
        for path in real_fits_files:
            tensor, _ = loader.load(path)
            arr = tensor.numpy()
            mean = float(arr.mean())
            std  = float(arr.std())
            assert abs(mean) < 1.0, \
                f"{path.name}: Z-score mean should be ~0, got {mean:.3f}"
            assert 0.5 < std < 2.0, \
                f"{path.name}: Z-score std should be ~1, got {std:.3f}"

    def test_metadata_contains_required_keys(self, real_fits_files):
        _require_real_images(real_fits_files)
        from inference.fits_loader import FITSLoader
        loader = FITSLoader()
        for path in real_fits_files:
            _, meta = loader.load(path)
            for key in ("obs_time", "width_px", "height_px"):
                assert key in meta, \
                    f"{path.name}: metadata missing key '{key}'"


# ---------------------------------------------------------------------------
# Classical detector
# ---------------------------------------------------------------------------

@pytest.mark.real_data
class TestRealClassicalDetector:
    """Verify the ASTRiDE classical detector runs cleanly on real images."""

    def test_detector_returns_list(self, real_fits_files):
        _require_real_images(real_fits_files)
        from src.ingest.fits_parser import parse_fits
        from src.detection.classical_detector import detect_streaks
        for path in real_fits_files:
            img = parse_fits(path)
            detections = detect_streaks(img)
            assert isinstance(detections, list), \
                f"{path.name}: detect_streaks should return a list"

    def test_streak_images_produce_at_least_one_detection(self, real_fits_files):
        """Files with 'streak' in their name should yield ≥1 detection."""
        _require_real_images(real_fits_files)
        from src.ingest.fits_parser import parse_fits
        from src.detection.classical_detector import detect_streaks
        streak_files = [p for p in real_fits_files if _has_streak_in_name(p)]
        if not streak_files:
            pytest.skip("No *streak* files in data/test/ — skipping detection assertion")
        for path in streak_files:
            img = parse_fits(path)
            detections = detect_streaks(img)
            assert len(detections) >= 1, \
                f"{path.name}: expected ≥1 detection in a streak image, got 0"

    def test_blank_images_produce_no_detections(self, real_fits_files):
        """Files with 'blank' or 'empty' in their name should yield 0 detections."""
        _require_real_images(real_fits_files)
        from src.ingest.fits_parser import parse_fits
        from src.detection.classical_detector import detect_streaks
        blank_files = [p for p in real_fits_files if _has_blank_in_name(p)]
        if not blank_files:
            pytest.skip("No *blank* files in data/test/ — skipping blank-image assertion")
        for path in blank_files:
            img = parse_fits(path)
            detections = detect_streaks(img)
            assert len(detections) == 0, \
                f"{path.name}: expected 0 detections in a blank image, got {len(detections)}"

    def test_detection_fields_are_valid(self, real_fits_files):
        """Every returned StreakDetection must have plausible field values."""
        _require_real_images(real_fits_files)
        from src.ingest.fits_parser import parse_fits
        from src.detection.classical_detector import detect_streaks
        for path in real_fits_files:
            img = parse_fits(path)
            for det in detect_streaks(img):
                assert det.length_px > 0, \
                    f"{path.name}: streak length must be positive"
                assert 0.0 <= det.angle_deg <= 180.0, \
                    f"{path.name}: angle {det.angle_deg} outside [0, 180]"


# ---------------------------------------------------------------------------
# Postprocessor (Radon angle refinement)
# ---------------------------------------------------------------------------

@pytest.mark.real_data
class TestRealPostprocess:
    """Verify Radon refinement runs without error on real image cutouts."""

    def test_refine_angle_completes_on_streak_images(self, real_fits_files):
        _require_real_images(real_fits_files)
        from src.ingest.fits_parser import parse_fits
        from src.detection.classical_detector import detect_streaks
        from inference.postprocess import refine_angle
        streak_files = [p for p in real_fits_files if _has_streak_in_name(p)]
        if not streak_files:
            pytest.skip("No *streak* files in data/test/")
        for path in streak_files:
            img = parse_fits(path)
            detections = detect_streaks(img)
            for det in detections[:3]:  # cap at 3 per image to keep tests fast
                refined = refine_angle(det, img.data)
                assert isinstance(refined, float), \
                    f"{path.name}: refine_angle should return float, got {type(refined)}"
                assert 0.0 <= refined <= 180.0, \
                    f"{path.name}: refined angle {refined} outside [0, 180]"
