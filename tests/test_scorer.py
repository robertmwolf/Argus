"""Tests for src/matching/scorer.py."""

import math
import pytest
from src.matching.scorer import gaussian_score, tle_age_penalty, aggregate_score


class TestGaussianScore:
    def test_zero_delta_returns_one(self):
        assert gaussian_score(0.0, sigma=1.0) == pytest.approx(1.0)

    def test_one_sigma_returns_approx_point_six(self):
        assert gaussian_score(1.0, sigma=1.0) == pytest.approx(math.exp(-0.5), rel=1e-6)

    def test_large_delta_approaches_zero(self):
        assert gaussian_score(100.0, sigma=1.0) < 1e-10

    def test_score_in_zero_one_range(self):
        for delta in [0.0, 0.5, 1.0, 5.0, 10.0]:
            s = gaussian_score(delta, sigma=2.0)
            assert 0.0 <= s <= 1.0


class TestTleAgePenalty:
    def test_zero_age_returns_one(self):
        assert tle_age_penalty(0.0) == pytest.approx(1.0)

    def test_penalty_decreases_with_age(self):
        assert tle_age_penalty(24.0) < tle_age_penalty(6.0)
        assert tle_age_penalty(72.0) < tle_age_penalty(24.0)

    def test_very_old_tle_near_zero(self):
        assert tle_age_penalty(200.0) < 0.01

    def test_fresh_tle_near_one(self):
        assert tle_age_penalty(1.0) > 0.99


class TestAggregateScore:
    def test_perfect_scores_high(self):
        score = aggregate_score(1.0, 1.0, 1.0, 1.0, tle_age_hours=0.0)
        assert score == pytest.approx(1.0, rel=1e-6)

    def test_zero_scores_return_zero_or_small(self):
        score = aggregate_score(0.0, 0.0, 0.0, 0.0, tle_age_hours=0.0)
        assert score == pytest.approx(0.0, abs=1e-10)

    def test_old_tle_penalises_position(self):
        fresh = aggregate_score(1.0, 0.0, 0.0, 0.0, tle_age_hours=0.0)
        old = aggregate_score(1.0, 0.0, 0.0, 0.0, tle_age_hours=72.0)
        assert old < fresh

    def test_weights_sum_to_one_at_perfect_fresh(self):
        score = aggregate_score(1.0, 1.0, 1.0, 1.0, tle_age_hours=0.0)
        assert score == pytest.approx(1.0, abs=1e-10)
