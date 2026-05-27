"""Tests for inference/confidence.py — Unified Confidence Score."""
import pytest

from inference.confidence import (
    DetectorProfile,
    DETECTOR_PROFILES,
    compute_unified_confidence,
    fbeta_weight,
)


# ---------------------------------------------------------------------------
# fbeta_weight
# ---------------------------------------------------------------------------

class TestFbetaWeight:
    def test_perfect_detector(self):
        assert fbeta_weight(1.0, 1.0) == pytest.approx(1.0)

    def test_zero_both(self):
        assert fbeta_weight(0.0, 0.0) == 0.0

    def test_precision_heavy_beta_half(self):
        # F-0.5 weights precision more than recall
        # High precision, low recall → higher than high recall, low precision
        high_p = fbeta_weight(0.9, 0.3)
        high_r = fbeta_weight(0.3, 0.9)
        assert high_p > high_r

    def test_known_dino_swin_t(self):
        # From phase8_benchmark.json: P=0.6667, R=0.7333
        w = fbeta_weight(0.6667, 0.7333)
        # (1.25 × 0.6667 × 0.7333) / (0.25 × 0.6667 + 0.7333)
        expected = (1.25 * 0.6667 * 0.7333) / (0.25 * 0.6667 + 0.7333)
        assert w == pytest.approx(expected, rel=1e-4)

    def test_result_in_unit_interval(self):
        for p in [0.0, 0.3, 0.5, 0.7, 1.0]:
            for r in [0.0, 0.3, 0.5, 0.7, 1.0]:
                w = fbeta_weight(p, r)
                assert 0.0 <= w <= 1.0


# ---------------------------------------------------------------------------
# DetectorProfile
# ---------------------------------------------------------------------------

class TestDetectorProfile:
    def test_f_beta_weight_property(self):
        p = DetectorProfile(name="Test", precision=0.6667, recall=0.7333)
        assert p.f_beta_weight == pytest.approx(fbeta_weight(0.6667, 0.7333), rel=1e-4)

    def test_registered_profiles_have_valid_weights(self):
        for key, profile in DETECTOR_PROFILES.items():
            w = profile.f_beta_weight
            assert 0.0 <= w <= 1.0, f"{key}: weight {w} out of range"

    def test_confidence_ceiling_default_is_none(self):
        p = DetectorProfile(name="Test", precision=0.5, recall=0.5)
        assert p.confidence_ceiling is None

    def test_astride_has_ceiling(self):
        profile = DETECTOR_PROFILES["astride"]
        assert profile.confidence_ceiling is not None
        assert 0.0 < profile.confidence_ceiling < 1.0

    def test_ceiling_in_unit_interval_when_set(self):
        for key, profile in DETECTOR_PROFILES.items():
            if profile.confidence_ceiling is not None:
                assert 0.0 < profile.confidence_ceiling <= 1.0, (
                    f"{key}: ceiling {profile.confidence_ceiling} out of range"
                )


# ---------------------------------------------------------------------------
# compute_unified_confidence
# ---------------------------------------------------------------------------

class TestComputeUnifiedConfidence:
    def test_empty_sources(self):
        result = compute_unified_confidence([])
        assert result["score"] == 0.0
        assert result["components"] == []

    def test_only_unified_source_filtered(self):
        # "unified" entries should be filtered; with nothing left, score = 0
        result = compute_unified_confidence([{"method": "unified", "confidence": 0.95}])
        assert result["score"] == 0.0
        assert result["components"] == []

    def test_unified_entry_ignored_alongside_real_detector(self):
        # The stale "unified" entry should be stripped; only "tiny" counts
        r_with = compute_unified_confidence([
            {"method": "unified", "confidence": 0.99},
            {"method": "tiny", "confidence": 0.85},
        ])
        r_without = compute_unified_confidence([
            {"method": "tiny", "confidence": 0.85},
        ])
        assert r_with["score"] == pytest.approx(r_without["score"])

    def test_single_detector_score(self):
        # Score for one detector preserves the detector's own confidence.
        result = compute_unified_confidence([{"method": "tiny", "confidence": 0.90}])
        assert result["score"] == pytest.approx(0.90)
        assert result["fn_penalty"] == 0.0

    def test_two_agreeing_detectors_higher_than_one(self):
        single = compute_unified_confidence([{"method": "dinov3_vitb", "confidence": 0.90}])
        double = compute_unified_confidence([
            {"method": "dinov3_vitb", "confidence": 0.90},
            {"method": "classical", "confidence": 0.90},
        ])
        assert double["score"] > single["score"]

    def test_two_strongly_disagreeing_detectors_lower_than_two_agreeing(self):
        agree = compute_unified_confidence([
            {"method": "dinov3_vitb", "confidence": 0.90},
            {"method": "classical", "confidence": 0.90},
        ])
        disagree = compute_unified_confidence([
            {"method": "dinov3_vitb", "confidence": 0.90},
            {"method": "classical", "confidence": 0.10},
        ])
        assert disagree["score"] < agree["score"]

    def test_low_confidence_second_detector_does_not_penalize_best_detector(self):
        single = compute_unified_confidence([{"method": "dinov3_vitb", "confidence": 0.85}])
        with_low_confidence = compute_unified_confidence([
            {"method": "dinov3_vitb", "confidence": 0.85},
            {"method": "classical", "confidence": 0.05},
        ])
        assert with_low_confidence["score"] >= single["score"]

    def test_all_zero_confidence(self):
        result = compute_unified_confidence([
            {"method": "tiny", "confidence": 0.0},
            {"method": "classical", "confidence": 0.0},
        ])
        assert result["score"] == pytest.approx(0.0, abs=1e-6)

    def test_low_confidence_non_astride_detector_lowers_score(self):
        # Low-confidence ML detectors still provide small reliability-weighted boosts.
        # "tiny" IS in DETECTOR_PROFILES (higher weight than default); "mystery_detector" is not.
        low_known = compute_unified_confidence([
            {"method": "dinov3_vitb", "confidence": 0.85},
            {"method": "tiny", "confidence": 0.05},
        ])
        low_unknown = compute_unified_confidence([
            {"method": "dinov3_vitb", "confidence": 0.85},
            {"method": "mystery_detector", "confidence": 0.05},
        ])
        assert low_unknown["score"] < low_known["score"]

    def test_score_capped_at_0_99(self):
        # Stacking many high-confidence detectors should never exceed 0.99
        sources = [{"method": "dinov3_vitb", "confidence": 1.0} for _ in range(10)]
        result = compute_unified_confidence(sources)
        assert result["score"] <= 0.99

    def test_score_in_unit_interval(self):
        import random
        random.seed(42)
        for _ in range(50):
            sources = [
                {"method": random.choice(list(DETECTOR_PROFILES.keys())),
                 "confidence": random.random()}
                for _ in range(random.randint(1, 4))
            ]
            r = compute_unified_confidence(sources)
            assert 0.0 <= r["score"] <= 0.99

    def test_components_returned(self):
        result = compute_unified_confidence([
            {"method": "tiny", "confidence": 0.80},
            {"method": "classical", "confidence": 0.70},
        ])
        assert len(result["components"]) == 2
        methods = {c["method"] for c in result["components"]}
        assert methods == {"tiny", "classical"}
        for c in result["components"]:
            assert 0.0 <= c["weight"] <= 1.0
            assert 0.0 <= c["contribution"] <= 1.0

    def test_divergence_zero_for_single_detector(self):
        result = compute_unified_confidence([{"method": "tiny", "confidence": 0.80}])
        assert result["divergence"] == 0.0

    def test_divergence_nonzero_for_disagreeing_detectors(self):
        result = compute_unified_confidence([
            {"method": "dinov3_vitb", "confidence": 0.95},
            {"method": "classical", "confidence": 0.10},
        ])
        assert result["divergence"] > 0.0

    def test_unknown_method_uses_default_weight(self):
        # An unrecognised method should still produce a valid score (default P=R=0.5)
        result = compute_unified_confidence([{"method": "mystery_detector", "confidence": 0.75}])
        assert 0.0 < result["score"] <= 0.99


# ---------------------------------------------------------------------------
# Confidence ceiling
# ---------------------------------------------------------------------------

class TestConfidenceCeiling:
    def test_astride_only_scores_zero(self):
        # ASTRiDE-only detections are disregarded upstream; scorer returns 0
        # for legacy callers that still pass one through.
        result = compute_unified_confidence([{"method": "astride", "confidence": 0.99}])
        assert result["score"] == 0.0
        assert result["components"] == []

    def test_astride_corroborates_without_lowering_score(self):
        single = compute_unified_confidence([{"method": "dinov3_vitb", "confidence": 0.80}])
        with_astride = compute_unified_confidence([
            {"method": "dinov3_vitb", "confidence": 0.80},
            {"method": "astride", "confidence": 0.60},
        ])
        assert with_astride["score"] > single["score"]
        assert with_astride["score"] >= 0.80

    def test_ml_plus_high_astride_lands_near_ninety_percent(self):
        result = compute_unified_confidence([
            {"method": "ml", "confidence": 0.86},
            {"method": "astride", "confidence": 0.99},
        ])
        assert result["score"] == pytest.approx(0.8996)

    def test_astride_component_marked_corroboration_only(self):
        result = compute_unified_confidence([
            {"method": "dinov3_vitb", "confidence": 0.87},
            {"method": "astride", "confidence": 0.99},
        ])
        comp = next(c for c in result["components"] if c["method"] == "astride")
        assert comp["role"] == "corroboration_only"
        assert comp["weight"] == 0.0

    def test_no_ceiling_detector_eff_conf_equals_raw(self):
        result = compute_unified_confidence([{"method": "dinov3_vitb", "confidence": 0.87}])
        comp = result["components"][0]
        assert comp["eff_conf"] == pytest.approx(comp["raw_conf"])
        assert comp["ceiling"] is None

    def test_astride_does_not_create_divergence_penalty(self):
        # ASTRiDE is excluded from divergence, so it cannot drag down a strong
        # corroborated ML detection.
        result = compute_unified_confidence([
            {"method": "dinov3_vitb", "confidence": 0.85},
            {"method": "astride", "confidence": 0.10},
        ])
        assert result["divergence"] == 0.0
        assert result["score"] >= 0.85
