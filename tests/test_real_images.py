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
    """Verify FITSLoader produces correctly shaped, normalised arrays."""

    def test_all_images_load_to_array(self, real_fits_files):
        _require_real_images(real_fits_files)
        from inference.fits_loader import FITSLoader
        loader = FITSLoader()
        for path in real_fits_files:
            result = loader.load(path)
            assert "array" in result, \
                f"{path.name}: result missing 'array' key"
            assert isinstance(result["array"], np.ndarray), \
                f"{path.name}: expected ndarray, got {type(result['array'])}"

    def test_array_is_hwc_uint8(self, real_fits_files):
        _require_real_images(real_fits_files)
        from inference.fits_loader import FITSLoader
        loader = FITSLoader()
        for path in real_fits_files:
            result = loader.load(path)
            arr = result["array"]
            assert arr.ndim == 3, \
                f"{path.name}: expected HWC array (3D), got {arr.ndim}D"
            assert arr.shape[2] == 3, \
                f"{path.name}: expected 3 channels, got {arr.shape[2]}"
            assert arr.dtype == np.uint8, \
                f"{path.name}: expected uint8, got {arr.dtype}"

    def test_array_has_non_zero_variance(self, real_fits_files):
        _require_real_images(real_fits_files)
        from inference.fits_loader import FITSLoader
        loader = FITSLoader()
        for path in real_fits_files:
            result = loader.load(path)
            assert result["array"].std() > 0, \
                f"{path.name}: array is entirely flat (std=0) — likely corrupt"

    def test_metadata_contains_required_keys(self, real_fits_files):
        _require_real_images(real_fits_files)
        from inference.fits_loader import FITSLoader
        loader = FITSLoader()
        for path in real_fits_files:
            result = loader.load(path)
            for key in ("obs_time", "shape", "filename"):
                assert key in result, \
                    f"{path.name}: result missing key '{key}'"
            h, w = result["shape"]
            assert h > 0 and w > 0, \
                f"{path.name}: non-positive shape {result['shape']}"


# ---------------------------------------------------------------------------
# Classical detector
# ---------------------------------------------------------------------------

@pytest.mark.real_data
class TestRealClassicalDetector:
    """Verify the classical (src/) detector runs cleanly on real images."""

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
        """At least one *streak* file must yield ≥1 detection.

        The classical detector may miss faint or short synthetic streaks, so we
        require the aggregate count across all streak-named files to be ≥ 1
        rather than requiring every individual file to produce a detection.
        """
        _require_real_images(real_fits_files)
        from src.ingest.fits_parser import parse_fits
        from src.detection.classical_detector import detect_streaks
        streak_files = [p for p in real_fits_files if _has_streak_in_name(p)]
        if not streak_files:
            pytest.skip("No *streak* files in data/test/ — skipping detection assertion")
        total = sum(len(detect_streaks(parse_fits(p))) for p in streak_files)
        assert total >= 1, \
            f"Expected ≥1 detection across {len(streak_files)} streak file(s), got 0"

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
        from inference.postprocess import refine_segment_angle
        streak_files = [p for p in real_fits_files if _has_streak_in_name(p)]
        if not streak_files:
            pytest.skip("No *streak* files in data/test/")
        for path in streak_files:
            img = parse_fits(path)
            detections = detect_streaks(img)
            for det in detections[:3]:  # cap at 3 per image to keep tests fast
                margin = max(10, int(det.length_px * 0.2))
                x1 = max(0, int(min(det.x_start, det.x_end)) - margin)
                y1 = max(0, int(min(det.y_start, det.y_end)) - margin)
                x2 = min(img.data.shape[1], int(max(det.x_start, det.x_end)) + margin)
                y2 = min(img.data.shape[0], int(max(det.y_start, det.y_end)) + margin)
                crop = img.data[y1:y2, x1:x2]
                refined = refine_segment_angle(crop, det.angle_deg)
                assert isinstance(refined, float), \
                    f"{path.name}: refine_segment_angle should return float, got {type(refined)}"
                assert 0.0 <= refined <= 180.0, \
                    f"{path.name}: refined angle {refined} outside [0, 180]"
