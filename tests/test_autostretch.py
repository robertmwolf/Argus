"""Tests for inference/autostretch.py — PixInsight AutoSTF implementation."""

from __future__ import annotations

import numpy as np
import pytest

from inference.autostretch import _find_midtone, _mtf, autostretch


# ---------------------------------------------------------------------------
# MTF unit tests
# ---------------------------------------------------------------------------

class TestMTF:
    def test_zero_maps_to_zero(self):
        x = np.array([0.0], dtype=np.float32)
        assert float(_mtf(0.5, x)[0]) == pytest.approx(0.0)

    def test_one_maps_to_one(self):
        x = np.array([1.0], dtype=np.float32)
        assert float(_mtf(0.5, x)[0]) == pytest.approx(1.0)

    def test_midtone_maps_to_half(self):
        m = 0.3
        x = np.array([m], dtype=np.float32)
        assert float(_mtf(m, x)[0]) == pytest.approx(0.5, abs=1e-4)

    def test_monotonically_increasing(self):
        x = np.linspace(0.0, 1.0, 100, dtype=np.float32)
        y = _mtf(0.25, x)
        assert np.all(np.diff(y) >= 0)


# ---------------------------------------------------------------------------
# _find_midtone unit tests
# ---------------------------------------------------------------------------

class TestFindMidtone:
    def test_v1_zero_returns_zero(self):
        assert _find_midtone(0.25, 0.0) == pytest.approx(0.0)

    def test_v1_one_returns_one(self):
        assert _find_midtone(0.25, 1.0) == pytest.approx(1.0)

    def test_mtf_roundtrip(self):
        """MTF applied with found m should map v1 to target."""
        target = 0.25
        v1 = 0.15
        m = _find_midtone(target, v1)
        result = float(_mtf(m, np.array([v1], dtype=np.float64))[0])
        assert result == pytest.approx(target, abs=1e-4)

    def test_different_targets(self):
        for target in [0.1, 0.25, 0.5, 0.75]:
            v1 = 0.2
            m = _find_midtone(target, v1)
            result = float(_mtf(m, np.array([v1], dtype=np.float64))[0])
            assert result == pytest.approx(target, abs=1e-3)


# ---------------------------------------------------------------------------
# autostretch integration tests
# ---------------------------------------------------------------------------

class TestAutostretch:
    def _sky_image(self, shape=(128, 128), sky=1000.0, noise=20.0, seed=42):
        """Synthetic low-contrast sky background image."""
        rng = np.random.default_rng(seed)
        return rng.normal(sky, noise, shape).astype(np.float32)

    def test_output_shape_2d(self):
        img = self._sky_image()
        out = autostretch(img)
        assert out.shape == img.shape

    def test_output_shape_3channel(self):
        img = np.stack([self._sky_image()] * 3, axis=-1)
        out = autostretch(img)
        assert out.shape == img.shape

    def test_output_dtype_float32(self):
        out = autostretch(self._sky_image())
        assert out.dtype == np.float32

    def test_output_range_01(self):
        out = autostretch(self._sky_image())
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_median_near_target_bg(self):
        """Stretched background median should be close to target_bg=0.25."""
        img = self._sky_image(shape=(256, 256))
        out = autostretch(img, target_bg=0.25)
        # Background dominates; median of output should land near target.
        # Allow ±0.08 tolerance since small images have statistical noise.
        assert float(np.median(out)) == pytest.approx(0.25, abs=0.08)

    def test_flat_image_returns_valid_output(self):
        """Flat (zero-variance) image should not crash and return [0,1] output."""
        img = np.full((64, 64), 500.0, dtype=np.float32)
        out = autostretch(img)
        assert out.dtype == np.float32
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_no_finite_pixels_raises(self):
        img = np.full((32, 32), np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="finite pixels"):
            autostretch(img)

    def test_wrong_ndim_raises(self):
        img = np.zeros((4, 4, 4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="2D or 3D"):
            autostretch(img)

    def test_streak_brighter_than_background(self):
        """A bright streak region should be brighter than background after stretch."""
        rng = np.random.default_rng(7)
        img = rng.normal(1000.0, 20.0, (128, 128)).astype(np.float32)
        img[60:65, 10:120] += 500.0  # bright streak

        out = autostretch(img)
        streak_mean = float(out[60:65, 10:120].mean())
        bg_mean = float(out[0:50, :].mean())
        assert streak_mean > bg_mean

    def test_linked_channels_identical_stretch(self):
        """All 3 channels of a grayscale-stacked image should be identical after stretch."""
        gray = self._sky_image(shape=(64, 64))
        img_3ch = np.stack([gray, gray, gray], axis=-1)
        out = autostretch(img_3ch)
        np.testing.assert_array_equal(out[:, :, 0], out[:, :, 1])
        np.testing.assert_array_equal(out[:, :, 0], out[:, :, 2])

    def test_custom_shadows_clip(self):
        """A less aggressive shadows_clip should raise the black point less."""
        img = self._sky_image(shape=(128, 128))
        out_default = autostretch(img, shadows_clip=-2.8)
        out_shallow = autostretch(img, shadows_clip=-1.0)
        # Shallower clip → higher black point → more pixels clipped to black
        assert float(out_shallow.min()) <= float(out_default.min()) + 0.01


# ---------------------------------------------------------------------------
# FITSLoader integration: autostretch is applied end-to-end
# ---------------------------------------------------------------------------

class TestFITSLoaderAutostretch:
    def _write_fits(self, path, sky=1000.0, noise=20.0):
        from astropy.io import fits
        rng = np.random.default_rng(99)
        data = rng.normal(sky, noise, (64, 64)).astype(np.float32)
        hdu = fits.PrimaryHDU(data)
        hdu.header["DATE-OBS"] = "2025-01-01T00:00:00"
        hdu.writeto(path, overwrite=True)
        return path

    def test_loader_output_dtype_uint8(self, tmp_path):
        from inference.fits_loader import FITSLoader
        f = self._write_fits(tmp_path / "test.fits")
        result = FITSLoader().load(f)
        assert result["array"].dtype == np.uint8

    def test_loader_output_shape_3ch(self, tmp_path):
        from inference.fits_loader import FITSLoader
        f = self._write_fits(tmp_path / "test.fits")
        result = FITSLoader().load(f)
        h, w, c = result["array"].shape
        assert c == 3

    def test_loader_uses_full_uint8_range(self, tmp_path):
        """AutoSTF should produce a wide dynamic range, not a narrow band."""
        from inference.fits_loader import FITSLoader
        f = self._write_fits(tmp_path / "test.fits")
        result = FITSLoader().load(f)
        arr = result["array"]
        dynamic_range = int(arr.max()) - int(arr.min())
        assert dynamic_range > 50, f"Dynamic range too narrow: {dynamic_range}"
