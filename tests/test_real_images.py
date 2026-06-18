"""Optional smoke tests for real FITS files placed in ``data/test``."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import DATA_TEST_DIR


def _require_real_images(files) -> None:
    if not files:
        pytest.skip(f"No FITS files found in {DATA_TEST_DIR}")


@pytest.mark.real_data
def test_real_fits_files_parse(real_fits_files) -> None:
    _require_real_images(real_fits_files)
    from src.ingest.fits_parser import parse_fits

    for path in real_fits_files:
        image = parse_fits(path)
        assert image.data.ndim == 2
        assert image.data.dtype == np.float32
        assert image.width_px > 0 and image.height_px > 0
        assert image.obs_time is not None


@pytest.mark.real_data
def test_real_fits_files_load_for_inference(real_fits_files) -> None:
    _require_real_images(real_fits_files)
    from inference.fits_loader import FITSLoader

    loader = FITSLoader()
    for path in real_fits_files:
        result = loader.load(path)
        array = result["array"]
        assert array.ndim == 3 and array.shape[2] == 3
        assert array.dtype == np.uint8
        assert array.std() > 0
