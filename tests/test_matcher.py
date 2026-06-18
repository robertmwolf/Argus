"""Tests for src/matching/matcher.py."""

from datetime import datetime, timezone

import pytest

from src.matching.matcher import match, CandidateMatch, _resolve_direction_delta
from src.detection.streak import StreakDetection
from src.ingest.fits_parser import FITSImage
import numpy as np
from astropy.io import fits as astrofits


def _make_detection(
    ra_center=10.0,
    dec_center=20.0,
    angular_velocity_arcsec_s=300.0,
    position_angle_deg=45.0,
) -> StreakDetection:
    return StreakDetection(
        x_start=100.0,
        y_start=100.0,
        x_end=200.0,
        y_end=150.0,
        ra_start=9.9,
        dec_start=19.9,
        ra_end=10.1,
        dec_end=20.1,
        ra_center=ra_center,
        dec_center=dec_center,
        angular_velocity_arcsec_s=angular_velocity_arcsec_s,
        position_angle_deg=position_angle_deg,
    )


def _make_fits_image() -> FITSImage:
    hdr = astrofits.Header()
    hdr["NAXIS1"] = 512
    hdr["NAXIS2"] = 512
    return FITSImage(
        filepath=None,
        obs_time=datetime(2024, 4, 2, 12, 0, 0, tzinfo=timezone.utc),
        ra_center=10.0,
        dec_center=20.0,
        width_px=512,
        height_px=512,
        pixscale_arcsec=None,
        exptime_sec=30.0,
        sitelat=None,
        sitelong=None,
        siteelev=None,
        data=np.zeros((512, 512), dtype=np.float32),
        header=hdr,
    )


def _make_candidate(
    norad_id=12345,
    predicted_ra=10.0,
    predicted_dec=20.0,
    predicted_velocity_arcsec_s=300.0,
    predicted_direction_deg=45.0,
    tle_age_hours=1.0,
) -> dict:
    return {
        "norad_id": norad_id,
        "object_name": f"SAT-{norad_id}",
        "tle_epoch": datetime(2024, 4, 2, 11, 0, 0, tzinfo=timezone.utc),
        "tle_age_hours": tle_age_hours,
        "predicted_ra": predicted_ra,
        "predicted_dec": predicted_dec,
        "predicted_velocity_arcsec_s": predicted_velocity_arcsec_s,
        "predicted_direction_deg": predicted_direction_deg,
        "predicted_magnitude": None,
    }


class TestMatchNoSkyCoords:
    def test_no_ra_returns_empty(self):
        det = _make_detection(ra_center=None, dec_center=None)
        result = match(det, [_make_candidate()], _make_fits_image())
        assert result == []

    def test_no_dec_returns_empty(self):
        det = _make_detection(ra_center=10.0, dec_center=None)
        result = match(det, [_make_candidate()], _make_fits_image())
        assert result == []


class TestMatchScoring:
    def test_close_candidate_scores_higher_than_distant(self):
        det = _make_detection(ra_center=10.0, dec_center=20.0)
        close = _make_candidate(norad_id=1, predicted_ra=10.001, predicted_dec=20.001)
        distant = _make_candidate(norad_id=2, predicted_ra=15.0, predicted_dec=25.0)
        results = match(det, [close, distant], _make_fits_image())
        assert results[0].norad_id == 1
        assert results[0].weighted_score > results[1].weighted_score

    def test_returns_candidate_match_dataclass(self):
        det = _make_detection()
        results = match(det, [_make_candidate()], _make_fits_image())
        assert len(results) == 1
        assert isinstance(results[0], CandidateMatch)

    def test_sorted_descending_by_score(self):
        det = _make_detection(ra_center=10.0, dec_center=20.0)
        cands = [
            _make_candidate(norad_id=i, predicted_ra=10.0 + i * 0.5, predicted_dec=20.0)
            for i in range(5)
        ]
        results = match(det, cands, _make_fits_image())
        scores = [r.weighted_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_candidates_returns_empty(self):
        det = _make_detection()
        assert match(det, [], _make_fits_image()) == []


class TestAmbiguityFlag:
    def test_ambiguous_when_scores_within_threshold(self):
        det = _make_detection(ra_center=10.0, dec_center=20.0)
        # Two nearly identical candidates
        c1 = _make_candidate(norad_id=1, predicted_ra=10.001, predicted_dec=20.0)
        c2 = _make_candidate(norad_id=2, predicted_ra=10.002, predicted_dec=20.0)
        results = match(det, [c1, c2], _make_fits_image())
        assert results[0].ambiguous is True

    def test_not_ambiguous_when_scores_far_apart(self):
        det = _make_detection(ra_center=10.0, dec_center=20.0)
        close = _make_candidate(norad_id=1, predicted_ra=10.0, predicted_dec=20.0)
        distant = _make_candidate(norad_id=2, predicted_ra=20.0, predicted_dec=30.0)
        results = match(det, [close, distant], _make_fits_image())
        assert results[0].ambiguous is False


class TestDirectionDelta:
    def test_same_angle_returns_zero(self):
        assert _resolve_direction_delta(45.0, 45.0) == pytest.approx(0.0)

    def test_opposite_angle_resolves_to_zero(self):
        """Observed PA + 180° should also resolve to near-zero delta."""
        assert _resolve_direction_delta(225.0, 45.0) == pytest.approx(0.0, abs=1e-9)

    def test_small_delta_returned(self):
        delta = _resolve_direction_delta(46.0, 45.0)
        assert delta == pytest.approx(1.0, abs=1e-9)

    def test_never_exceeds_90_degrees(self):
        """After ambiguity resolution, max meaningful delta is 90°."""
        import random
        for _ in range(100):
            obs = random.uniform(0, 360)
            pred = random.uniform(0, 360)
            delta = _resolve_direction_delta(obs, pred)
            assert delta <= 90.0 + 1e-9
