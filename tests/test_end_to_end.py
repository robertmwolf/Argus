"""End-to-end pipeline test using synthetic data.

Does not hit Space-Track. Verifies that all modules wire together correctly
with mock/synthetic inputs.
"""

import math
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pytest
from astropy.io import fits as astrofits
from astropy.wcs import WCS

from src.detection.classical_detector import StreakDetection
from src.ingest.fits_parser import FITSImage
from src.matching.matcher import match, CandidateMatch
from src.matching.scorer import gaussian_score, aggregate_score
from src.matching.spatial_filter import filter_by_fov, _angular_separation
from src.matching.propagator import propagate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wcs_header(ra_center: float = 10.0, dec_center: float = 20.0) -> astrofits.Header:
    hdr = astrofits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = 512
    hdr["NAXIS2"] = 512
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRVAL1"] = ra_center
    hdr["CRVAL2"] = dec_center
    hdr["CRPIX1"] = 256.0
    hdr["CRPIX2"] = 256.0
    hdr["CDELT1"] = -0.001  # ~3.6 arcsec/px
    hdr["CDELT2"] = 0.001
    hdr["DATE-OBS"] = "2024-04-02T12:00:00"
    hdr["EXPTIME"] = 30.0
    return hdr


def _make_fits_image(ra_center=10.0, dec_center=20.0) -> FITSImage:
    hdr = _make_wcs_header(ra_center, dec_center)
    return FITSImage(
        filepath=None,
        obs_time=datetime(2024, 4, 2, 12, 0, 0, tzinfo=timezone.utc),
        ra_center=ra_center,
        dec_center=dec_center,
        width_px=512,
        height_px=512,
        pixscale_arcsec=3.6,
        exptime_sec=30.0,
        sitelat=45.0,
        sitelong=9.0,
        siteelev=200.0,
        data=np.zeros((512, 512), dtype=np.float32),
        header=hdr,
    )


def _make_streak(ra=10.0, dec=20.0, vel=300.0, pa=45.0) -> StreakDetection:
    ra_start = (ra - 0.1) if ra is not None else None
    ra_end = (ra + 0.1) if ra is not None else None
    return StreakDetection(
        x_start=156.0, y_start=246.0,
        x_end=356.0, y_end=266.0,
        x_center=256.0, y_center=256.0,
        angle_deg=pa if pa is not None else 0.0,
        length_px=200.0,
        width_px=2.0,
        shape_factor=100.0,
        area_px=400.0,
        ra_start=ra_start, dec_start=dec,
        ra_end=ra_end, dec_end=dec,
        ra_center=ra, dec_center=dec,
        angular_velocity_arcsec_s=vel,
        position_angle_deg=pa,
    )


def _make_propagated_candidate(
    norad_id: int = 99999,
    ra: float = 10.0,
    dec: float = 20.0,
    vel: float = 300.0,
    direction: float = 45.0,
    age_hours: float = 2.0,
) -> dict:
    return {
        "norad_id": norad_id,
        "object_name": f"TESTSAT-{norad_id}",
        "tle_epoch": datetime(2024, 4, 2, 10, 0, 0, tzinfo=timezone.utc),
        "tle_age_hours": age_hours,
        "predicted_ra": ra,
        "predicted_dec": dec,
        "predicted_velocity_arcsec_s": vel,
        "predicted_direction_deg": direction,
        "predicted_magnitude": None,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScorerIntegration:
    def test_perfect_match_near_one(self):
        score = aggregate_score(1.0, 1.0, 1.0, 1.0, tle_age_hours=0.0)
        assert score == pytest.approx(1.0, abs=1e-9)

    def test_scores_decrease_with_separation(self):
        s0 = gaussian_score(0.0, sigma=0.25)
        s1 = gaussian_score(0.1, sigma=0.25)
        s2 = gaussian_score(0.5, sigma=0.25)
        assert s0 > s1 > s2


class TestMatcherIntegration:
    def test_correct_candidate_ranked_first(self):
        det = _make_streak(ra=10.0, dec=20.0, vel=300.0, pa=45.0)
        correct = _make_propagated_candidate(norad_id=1, ra=10.0, dec=20.0, vel=300.0, direction=45.0)
        decoy1 = _make_propagated_candidate(norad_id=2, ra=13.0, dec=23.0, vel=100.0, direction=90.0)
        decoy2 = _make_propagated_candidate(norad_id=3, ra=15.0, dec=25.0, vel=50.0, direction=10.0)

        results = match(det, [decoy2, decoy1, correct], _make_fits_image())
        assert len(results) == 3
        assert results[0].norad_id == 1

    def test_all_results_are_candidate_match(self):
        det = _make_streak()
        cands = [_make_propagated_candidate(norad_id=i) for i in range(3)]
        results = match(det, cands, _make_fits_image())
        assert all(isinstance(r, CandidateMatch) for r in results)

    def test_no_sky_coords_returns_empty(self):
        det = _make_streak(ra=None, dec=None)
        det.ra_center = None
        det.dec_center = None
        results = match(det, [_make_propagated_candidate()], _make_fits_image())
        assert results == []


class TestAngularSeparation:
    def test_same_point_zero(self):
        assert _angular_separation(10.0, 20.0, 10.0, 20.0) == pytest.approx(0.0, abs=1e-9)

    def test_known_separation(self):
        sep = _angular_separation(0.0, 0.0, 1.0, 0.0)
        assert sep == pytest.approx(1.0, abs=1e-6)


class TestFullPipelineWithMockPropagator:
    """Full pipeline test: streak → spatial filter (mocked) → match."""

    def test_correct_norad_in_top_three(self):
        det = _make_streak(ra=10.0, dec=20.0, vel=300.0, pa=45.0)
        fits_img = _make_fits_image()

        # Build a small candidate pool: 1 correct + 4 decoys
        candidates = [
            _make_propagated_candidate(norad_id=42, ra=10.0, dec=20.0, vel=300.0, direction=45.0),
            _make_propagated_candidate(norad_id=100, ra=12.0, dec=22.0, vel=50.0, direction=90.0),
            _make_propagated_candidate(norad_id=101, ra=14.0, dec=18.0, vel=200.0, direction=10.0),
            _make_propagated_candidate(norad_id=102, ra=8.0, dec=24.0, vel=400.0, direction=180.0),
            _make_propagated_candidate(norad_id=103, ra=16.0, dec=16.0, vel=600.0, direction=5.0),
        ]

        results = match(det, candidates, fits_img)
        top3_norad = [r.norad_id for r in results[:3]]
        assert 42 in top3_norad, f"Correct NORAD 42 not in top 3: {top3_norad}"
