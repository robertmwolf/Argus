"""Tests for scripts/convert_gtimages.py."""

import json
import math
import pathlib
import sys

import numpy as np
import pytest
from astropy.io import fits

# Make scripts/archive/ importable without installing
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts" / "archive"))
from convert_gtimages import (
    _fits_dimensions,
    _obs_to_coco_annotation,
    _parse_strk_file,
    convert,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_fits(path: pathlib.Path, width: int = 200, height: int = 150) -> pathlib.Path:
    data = np.zeros((height, width), dtype=np.uint16)
    hdu = fits.PrimaryHDU(data)
    hdu.header["NAXIS1"] = width
    hdu.header["NAXIS2"] = height
    hdu.header["DATE-OBS"] = "2026-04-27T01:53:37.305"
    hdu.writeto(path, overwrite=True)
    return path


def _write_strk(path: pathlib.Path, obs_rows: list[str], norad: int = 48381) -> pathlib.Path:
    """Write a minimal .strk file with the given observation rows."""
    tle_row = (
        f"{norad}\t26\t116.74\t53.16\t124.58\t0001333\t94.56\t265.55\t"
        f"15.30197802\t24157-4\t.00000356\t00000+0\t0\t999\t27458\t"
        f"STARLINK-2630\t2021-038AE\tU\n"
    )
    header_row = (
        "Image\tDate Time(UTC)\tJD Midpoint\tStart X Pixel\tStart Y Pixel\t"
        "End X Pixel\tEnd Y Pixel\tMid X Pixel\tMid Y Pixel\tPeak SNR\t"
        "Mean SNR\tElongation\tLength\tReject\tMid RA\tMid Dec\tStart RA\t"
        "Start Dec\tEnd RA\tEnd Dec\tExpected RA\tExpected Dec\t"
        "Expected Range\tExposure\tGain\tComment\n"
    )
    content = (
        "[VERSION]\nSkytrack\t1.9.8\n"
        "[SITE]\n\tLatitude(deg)\tLongitude(deg, East=+)\tElevation(m)\n"
        "43.6735556\t-81.0204722\t365.0\n"
        "[TLE]\n"
        "NORAD\tEpochYear\tEpochDay\tIncl\tRAAN\tECC\tARGP\tMA\tMM\t"
        "BSTAR\tMM1\tMM2\tEphemType\tElset\tRev\tName\tObject ID\tClass\n"
        + tle_row
        + "[OBS]\n"
        + header_row
        + "".join(obs_rows)
    )
    path.write_text(content)
    return path


def _usable_row(fits_name: str, x0=10, y0=20, x1=110, y1=70) -> str:
    return (
        f"{fits_name}\t2026-04-27T01:53:37\t2461157.57\t"
        f"{x0}\t{y0}\t{x1}\t{y1}\t"
        f"{(x0+x1)//2}\t{(y0+y1)//2}\t"
        f"38\t7\t51\t515\t0\t"
        f"139.69\t-10.27\t139.62\t-10.31\t139.76\t-10.22\t"
        f"139.69\t-10.27\t828.30\t0.5000\t300.0\t\n"
    )


def _negative_row(fits_name: str) -> str:
    return (
        f"{fits_name}\t2026-04-27T01:55:33\t2461157.58\t"
        f"0\t0\t0\t0\t0\t0\t0\t0\t1\t0\t-1\t"
        f"0\t0\t0\t0\t0\t0\t0\t0\t0\t0.5000\t300.0\tNo streak found\n"
    )


def _rejected_row(fits_name: str) -> str:
    return (
        f"{fits_name}\t2026-04-27T01:55:50\t2461157.58\t"
        f"100\t100\t200\t150\t150\t125\t5\t2\t30\t200\t4\t"
        f"0\t0\t0\t0\t0\t0\t0\t0\t0\t0.5000\t300.0\tCould not platesolve\n"
    )


# ---------------------------------------------------------------------------
# _parse_strk_file
# ---------------------------------------------------------------------------

class TestParseStrkFile:
    def test_extracts_norad_id(self, tmp_path):
        path = _write_strk(tmp_path / "48381.strk", [_usable_row("img.fits")], norad=48381)
        result = _parse_strk_file(path)
        assert result["norad_id"] == 48381

    def test_extracts_satellite_name(self, tmp_path):
        path = _write_strk(tmp_path / "48381.strk", [_usable_row("img.fits")])
        result = _parse_strk_file(path)
        assert result["name"] == "STARLINK-2630"

    def test_parses_usable_observation(self, tmp_path):
        path = _write_strk(tmp_path / "48381.strk", [_usable_row("img.fits", x0=10, y0=20, x1=110, y1=70)])
        result = _parse_strk_file(path)
        obs = result["observations"][0]
        assert obs["reject"] == "0"
        assert obs["x_start"] == pytest.approx(10.0)
        assert obs["y_start"] == pytest.approx(20.0)
        assert obs["x_end"] == pytest.approx(110.0)
        assert obs["y_end"] == pytest.approx(70.0)

    def test_parses_negative_observation(self, tmp_path):
        path = _write_strk(tmp_path / "48381.strk", [_negative_row("neg.fits")])
        result = _parse_strk_file(path)
        obs = result["observations"][0]
        assert obs["reject"] == "-1"

    def test_multiple_observations(self, tmp_path):
        rows = [_usable_row(f"img_{i}.fits", x0=i*10, y0=0, x1=i*10+50, y1=30) for i in range(3)]
        path = _write_strk(tmp_path / "48381.strk", rows)
        result = _parse_strk_file(path)
        assert len(result["observations"]) == 3

    def test_mixed_reject_flags(self, tmp_path):
        rows = [_usable_row("good.fits"), _negative_row("neg.fits"), _rejected_row("bad.fits")]
        path = _write_strk(tmp_path / "48381.strk", rows)
        result = _parse_strk_file(path)
        rejects = [o["reject"] for o in result["observations"]]
        assert "0" in rejects
        assert "-1" in rejects
        assert "4" in rejects


# ---------------------------------------------------------------------------
# _obs_to_coco_annotation
# ---------------------------------------------------------------------------

class TestObsToCOCOAnnotation:
    def _make_obs(self, x0=0.0, y0=0.0, x1=100.0, y1=0.0) -> dict:
        return {
            "x_start": x0, "y_start": y0, "x_end": x1, "y_end": y1,
            "x_mid": (x0 + x1) / 2, "y_mid": (y0 + y1) / 2,
            "peak_snr": 50.0, "mean_snr": 10.0,
            "elongation": 40.0, "length_px": math.hypot(x1 - x0, y1 - y0),
            "jd_mid": 2461157.58, "datetime_utc": "2026-04-27T01:53:37",
        }

    def test_category_id_is_one(self):
        ann = _obs_to_coco_annotation(self._make_obs(), 1, 1, 48381, "STARLINK-2630")
        assert ann["category_id"] == 1

    def test_image_id_propagated(self):
        ann = _obs_to_coco_annotation(self._make_obs(), 42, 1, 48381, "STARLINK-2630")
        assert ann["image_id"] == 42

    def test_horizontal_streak_angle(self):
        ann = _obs_to_coco_annotation(self._make_obs(x0=0, y0=50, x1=100, y1=50), 1, 1, 1, "SAT")
        assert ann["x1"] == pytest.approx(0.0)
        assert ann["y1"] == pytest.approx(50.0)
        assert ann["x2"] == pytest.approx(100.0)
        assert ann["y2"] == pytest.approx(50.0)

    def test_diagonal_streak_angle(self):
        ann = _obs_to_coco_annotation(self._make_obs(x0=0, y0=0, x1=100, y1=100), 1, 1, 1, "SAT")
        assert ann["x1"] == pytest.approx(0.0)
        assert ann["y1"] == pytest.approx(0.0)
        assert ann["x2"] == pytest.approx(100.0)
        assert ann["y2"] == pytest.approx(100.0)

    def test_endpoint_centre_correct(self):
        ann = _obs_to_coco_annotation(self._make_obs(x0=10, y0=20, x1=110, y1=60), 1, 1, 1, "SAT")
        assert (ann["x1"] + ann["x2"]) / 2 == pytest.approx(60.0)
        assert (ann["y1"] + ann["y2"]) / 2 == pytest.approx(40.0)

    def test_endpoint_length_correct(self):
        ann = _obs_to_coco_annotation(self._make_obs(x0=0, y0=0, x1=100, y1=0), 1, 1, 1, "SAT")
        length = math.hypot(ann["x2"] - ann["x1"], ann["y2"] - ann["y1"])
        assert length == pytest.approx(100.0)

    def test_segmentation_has_eight_points(self):
        ann = _obs_to_coco_annotation(self._make_obs(), 1, 1, 1, "SAT")
        assert len(ann["segmentation"]) == 1
        assert len(ann["segmentation"][0]) == 8

    def test_bbox_encloses_segmentation(self):
        ann = _obs_to_coco_annotation(self._make_obs(x0=20, y0=20, x1=120, y1=20), 1, 1, 1, "SAT")
        poly = ann["segmentation"][0]
        xs = poly[0::2]
        ys = poly[1::2]
        bx, by, bw, bh = ann["bbox"]
        assert bx <= min(xs) + 1e-6
        assert by <= min(ys) + 1e-6
        assert bx + bw >= max(xs) - 1e-6
        assert by + bh >= max(ys) - 1e-6

    def test_norad_id_in_attributes(self):
        ann = _obs_to_coco_annotation(self._make_obs(), 1, 1, 48381, "STARLINK-2630")
        assert ann["attributes"]["norad_id"] == 48381
        assert ann["attributes"]["satellite_name"] == "STARLINK-2630"


# ---------------------------------------------------------------------------
# _fits_dimensions
# ---------------------------------------------------------------------------

class TestFitsDimensions:
    def test_returns_correct_width_height(self, tmp_path):
        p = _write_fits(tmp_path / "img.fits", width=320, height=240)
        w, h = _fits_dimensions(p)
        assert w == 320
        assert h == 240


# ---------------------------------------------------------------------------
# convert (integration)
# ---------------------------------------------------------------------------

class TestConvert:
    def test_labeled_images_count(self, tmp_path):
        _write_fits(tmp_path / "good1.fits")
        _write_fits(tmp_path / "good2.fits")
        _write_strk(tmp_path / "48381.strk", [_usable_row("good1.fits"), _usable_row("good2.fits")])
        labeled, _ = convert(tmp_path, tmp_path / "out.json")
        assert len(labeled["images"]) == 2
        assert len(labeled["annotations"]) == 2

    def test_negative_images_count(self, tmp_path):
        _write_fits(tmp_path / "neg.fits")
        _write_strk(tmp_path / "48381.strk", [_negative_row("neg.fits")])
        _, negatives = convert(tmp_path, tmp_path / "out.json", tmp_path / "neg.json")
        assert len(negatives["images"]) == 1
        assert len(negatives["annotations"]) == 0

    def test_rejected_obs_excluded(self, tmp_path):
        _write_fits(tmp_path / "bad.fits")
        _write_strk(tmp_path / "48381.strk", [_rejected_row("bad.fits")])
        labeled, negatives = convert(tmp_path, tmp_path / "out.json", tmp_path / "neg.json")
        assert len(labeled["images"]) == 0
        assert len(negatives["images"]) == 0

    def test_missing_fits_skipped_gracefully(self, tmp_path):
        _write_strk(tmp_path / "48381.strk", [_usable_row("ghost.fits")])
        # ghost.fits does not exist — should not raise
        labeled, _ = convert(tmp_path, tmp_path / "out.json")
        assert len(labeled["images"]) == 0

    def test_output_json_is_valid_coco(self, tmp_path):
        _write_fits(tmp_path / "img.fits")
        _write_strk(tmp_path / "48381.strk", [_usable_row("img.fits")])
        out = tmp_path / "out.json"
        convert(tmp_path, out)
        data = json.loads(out.read_text())
        assert "images" in data
        assert "annotations" in data
        assert "categories" in data
        assert data["categories"][0]["name"] == "satellite_streak"

    def test_annotation_ids_are_unique(self, tmp_path):
        for i in range(5):
            _write_fits(tmp_path / f"img{i}.fits")
        rows = [_usable_row(f"img{i}.fits", x0=i*10, y0=0, x1=i*10+50, y1=30) for i in range(5)]
        _write_strk(tmp_path / "48381.strk", rows)
        labeled, _ = convert(tmp_path, tmp_path / "out.json")
        ids = [a["id"] for a in labeled["annotations"]]
        assert len(ids) == len(set(ids))

    def test_multiple_strk_files(self, tmp_path):
        _write_fits(tmp_path / "imgA.fits")
        _write_fits(tmp_path / "imgB.fits")
        _write_strk(tmp_path / "48381.strk", [_usable_row("imgA.fits")], norad=48381)
        _write_strk(tmp_path / "57166.strk", [_usable_row("imgB.fits")], norad=57166)
        labeled, _ = convert(tmp_path, tmp_path / "out.json")
        assert len(labeled["images"]) == 2
        norads = {a["attributes"]["norad_id"] for a in labeled["annotations"]}
        assert 48381 in norads
        assert 57166 in norads

    def test_no_strk_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            convert(tmp_path, tmp_path / "out.json")

    def test_negatives_output_none_does_not_write(self, tmp_path):
        _write_fits(tmp_path / "neg.fits")
        _write_strk(tmp_path / "48381.strk", [_negative_row("neg.fits")])
        convert(tmp_path, tmp_path / "out.json", negatives_output_path=None)
        assert not (tmp_path / "neg.json").exists()
