"""Offline tests for scripts/evaluate_candidate_provider.py."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch


def test_compare_to_baseline_metrics() -> None:
    from scripts.evaluate_candidate_provider import _compare_to_baseline

    rows = [
        {
            "provider": "local",
            "top1_norad": 42,
            "top1_confidence": 0.8,
            "top3_norads": [42, 43, 44],
            "candidate_count": 100,
            "error": None,
        },
        {
            "provider": "satchecker",
            "top1_norad": 43,
            "top1_confidence": 0.7,
            "top3_norads": [43, 42, 99],
            "candidate_count": 5,
            "error": None,
        },
    ]

    comparisons = _compare_to_baseline(rows, "local")

    assert comparisons["satchecker"]["top1_match_same"] is False
    assert comparisons["satchecker"]["top3_contains_baseline_top1"] is True
    assert comparisons["satchecker"]["top3_overlap"] == [42, 43]
    assert comparisons["satchecker"]["confidence_delta"] == -0.1


def test_summarize_provider_and_comparison_rates() -> None:
    from scripts.evaluate_candidate_provider import summarize

    images = [
        {
            "provider_results": [
                {
                    "provider": "local",
                    "candidate_count": 10,
                    "error": None,
                    "fetch_ms": 1.0,
                    "total_ms": 5.0,
                    "top1_norad": 1,
                },
                {
                    "provider": "satchecker",
                    "candidate_count": 2,
                    "error": None,
                    "fetch_ms": 100.0,
                    "total_ms": 120.0,
                    "top1_norad": 1,
                },
            ],
            "comparisons": {
                "satchecker": {
                    "top1_match_same": True,
                    "top3_contains_baseline_top1": True,
                    "confidence_delta": 0.05,
                }
            },
        },
        {
            "provider_results": [
                {
                    "provider": "local",
                    "candidate_count": 10,
                    "error": None,
                    "fetch_ms": 1.0,
                    "total_ms": 5.0,
                    "top1_norad": 1,
                },
                {
                    "provider": "satchecker",
                    "candidate_count": 0,
                    "error": "timeout",
                    "fetch_ms": 8.0,
                    "total_ms": 8.0,
                    "top1_norad": None,
                },
            ],
            "comparisons": {
                "satchecker": {
                    "top1_match_same": False,
                    "top3_contains_baseline_top1": False,
                    "confidence_delta": None,
                }
            },
        },
    ]

    summary = summarize(images, ["local", "satchecker"], "local")

    sat = summary["provider_summary"]["satchecker"]
    assert sat["empty_catalog_rate"] == 0.5
    assert sat["error_rate"] == 0.5
    assert sat["identified_rate"] == 0.5
    assert summary["comparison_summary"]["satchecker"]["top1_agreement_rate"] == 0.5
    assert summary["comparison_summary"]["satchecker"]["mean_confidence_delta"] == 0.05


def test_evaluate_provider_fetches_once_and_scores_catalog() -> None:
    from scripts.evaluate_candidate_provider import evaluate_provider

    obs_time = datetime(2024, 4, 2, 12, 0, 0, tzinfo=timezone.utc)
    detections = [{"confidence": 0.9, "ra_tip1_deg": 10.0, "dec_tip1_deg": 20.0}]
    metadata = {
        "obs_time": obs_time,
        "observer_lat": 45.0,
        "observer_lon": 9.0,
        "observer_alt_m": 200.0,
        "exposure_time": 10.0,
    }
    catalog = [{"name": "TEST", "line1": "1 fake", "line2": "2 fake"}]

    def fake_cross_identify(dets, *_args, **_kwargs):
        dets[0]["identifications"] = [{
            "satellite_name": "TEST",
            "norad_id": 42,
            "confidence": 0.75,
            "rank": 1,
        }]
        return dets

    with patch(
        "inference.crossid._fetch_candidate_catalog",
        return_value=catalog,
    ) as mock_fetch:
        with patch("inference.crossid.cross_identify", side_effect=fake_cross_identify) as mock_score:
            row = evaluate_provider(
                "local",
                detections,
                metadata,
                epoch_window_days=3,
                min_mean_motion=0.0,
                max_detections=1,
            )

    mock_fetch.assert_called_once()
    mock_score.assert_called_once()
    assert row["candidate_count"] == 1
    assert row["top1_norad"] == 42
    assert row["top1_confidence"] == 0.75
