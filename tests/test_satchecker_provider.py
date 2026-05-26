"""Tests for the optional SatChecker candidate provider.

All tests run offline.  No live SatChecker API calls are made.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

_OBS_TIME = datetime(2024, 11, 8, 21, 28, 28, tzinfo=timezone.utc)
_LINE1 = "1 25544U 98067A   24313.59901620  .00020137  00000+0  35437-3 0  9990"
_LINE2 = "2 25544  51.6389 331.1001 0007303  38.0131  94.6198 15.50387645480110"


def test_derive_fov_handles_ra_wrap() -> None:
    """A streak crossing RA=0 should produce a center near 0, not 180 deg."""
    from src.matching.satchecker_provider import derive_fov_from_detections

    fov = derive_fov_from_detections(
        [{
            "ra_tip1_deg": 359.8,
            "dec_tip1_deg": 10.0,
            "ra_tip2_deg": 0.2,
            "dec_tip2_deg": 10.0,
        }],
        min_radius_deg=0.1,
        margin_deg=0.0,
        max_radius_deg=2.0,
    )

    assert fov is not None
    ra, dec, radius = fov
    assert ra == pytest.approx(0.0, abs=0.3)
    assert dec == pytest.approx(10.0)
    assert 0.1 <= radius <= 0.25


def test_catalog_from_satchecker_response_parses_satellite_tle_data() -> None:
    """Documented FOV response shape plus included TLEs maps to cross-ID entries."""
    from src.matching.satchecker_provider import catalog_from_satchecker_response

    payload = [{
        "data": {
            "satellites": {
                "ISS (ZARYA) (25544)": {
                    "name": "ISS (ZARYA)",
                    "norad_id": 25544,
                    "positions": [{"ra": 10.0, "dec": 20.0}],
                    "tle_data": [{
                        "data_source": "spacetrack",
                        "epoch": "2024-11-08 14:22:35 UTC",
                        "satellite_id": 25544,
                        "satellite_name": "ISS (ZARYA)",
                        "tle_line1": _LINE1,
                        "tle_line2": _LINE2,
                    }],
                }
            }
        }
    }]

    catalog = catalog_from_satchecker_response(payload)

    assert catalog == [{
        "name": "ISS (ZARYA)",
        "line1": _LINE1,
        "line2": _LINE2,
        "tle_epoch": "2024-11-08 14:22:35 UTC",
        "source": "satchecker:spacetrack",
        "tle_search_mode": "satchecker_fov",
        "tle_data_fresh_at": None,
    }]


def test_satchecker_provider_fetches_with_include_tles(monkeypatch) -> None:
    """Provider calls the FOV endpoint with include_tles and sync mode."""
    from src.matching.satchecker_provider import SatCheckerCandidateProvider

    response = Mock()
    response.json.return_value = [{
        "data": {
            "satellites": {
                "ISS (ZARYA) (25544)": {
                    "name": "ISS (ZARYA)",
                    "norad_id": 25544,
                    "tle_data": [{"tle_line1": _LINE1, "tle_line2": _LINE2}],
                }
            }
        }
    }]
    response.raise_for_status.return_value = None

    detections = [{
        "ra_tip1_deg": 10.0,
        "dec_tip1_deg": 20.0,
        "ra_tip2_deg": 10.2,
        "dec_tip2_deg": 20.1,
    }]
    with patch("src.matching.satchecker_provider.requests.get", return_value=response) as mock_get:
        provider = SatCheckerCandidateProvider(base_url="https://example.test", timeout_s=3)
        catalog = provider.get_catalog(detections, _OBS_TIME, 33.0, -117.0, 100.0, exposure_time=2.0)

    assert len(catalog) == 1
    url = mock_get.call_args.args[0]
    params = mock_get.call_args.kwargs["params"]
    assert url == "https://example.test/fov/satellite-passes/"
    assert params["include_tles"] == "true"
    assert params["async"] == "false"
    assert params["group_by"] == "satellite"
    assert mock_get.call_args.kwargs["timeout"] == 3


def test_crossid_satchecker_provider_falls_back_to_local(monkeypatch) -> None:
    """ARGUS_CANDIDATE_PROVIDER=satchecker keeps local fallback by default."""
    import inference.crossid as crossid

    monkeypatch.setenv("ARGUS_CANDIDATE_PROVIDER", "satchecker")
    with patch("inference.crossid._fetch_satchecker_catalog", return_value=[]):
        with patch("inference.crossid._fetch_tle_catalog", return_value=[{
            "name": "LOCAL",
            "line1": _LINE1,
            "line2": _LINE2,
        }]) as mock_local:
            catalog = crossid._fetch_candidate_catalog(
                [{"ra_tip1_deg": 10.0, "dec_tip1_deg": 20.0}],
                _OBS_TIME,
                3,
                0,
                33.0,
                -117.0,
                100.0,
                2.0,
            )

    mock_local.assert_called_once_with(_OBS_TIME, 3, 0)
    assert catalog[0]["name"] == "LOCAL"
