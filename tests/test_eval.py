"""Tests for eval/metrics.py and eval/benchmark.py."""

import json
import math
import pytest
from pathlib import Path

from eval.metrics import (
    _angle_error_deg,
    _compute_ap,
    _compute_mean_angle_error,
    _compute_prf,
    _obb_iou,
    evaluate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obb(cx=100, cy=100, w=200, h=10, angle_deg=0.0):
    return {"cx": cx, "cy": cy, "w": w, "h": h, "angle_deg": angle_deg}


def _pred(image_id="img1", confidence=0.9, w=200, h=10, angle_deg=0.0, length=200.0, **kwargs):
    return {
        "image_id": image_id,
        "confidence": confidence,
        "obb": _make_obb(w=w, h=h, angle_deg=angle_deg, **kwargs),
        "streak_length_px": length,
    }


def _gt(image_id="img1", w=200, h=10, angle_deg=0.0, length=200.0, **kwargs):
    return {
        "image_id": image_id,
        "obb": _make_obb(w=w, h=h, angle_deg=angle_deg, **kwargs),
        "streak_length_px": length,
    }


# ---------------------------------------------------------------------------
# OBB IoU
# ---------------------------------------------------------------------------

class TestObbIou:
    def test_identical_obbs_have_iou_one(self):
        obb = _make_obb(cx=100, cy=100, w=200, h=10, angle_deg=0)
        assert _obb_iou(obb, obb) == pytest.approx(1.0, abs=1e-4)

    def test_non_overlapping_obbs_have_iou_zero(self):
        a = _make_obb(cx=0, cy=0, w=10, h=10, angle_deg=0)
        b = _make_obb(cx=1000, cy=1000, w=10, h=10, angle_deg=0)
        assert _obb_iou(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_partial_overlap_is_between_zero_and_one(self):
        a = _make_obb(cx=100, cy=100, w=100, h=10, angle_deg=0)
        b = _make_obb(cx=150, cy=100, w=100, h=10, angle_deg=0)
        iou = _obb_iou(a, b)
        assert 0.0 < iou < 1.0

    def test_iou_is_symmetric(self):
        a = _make_obb(cx=100, cy=100, w=200, h=10, angle_deg=5)
        b = _make_obb(cx=110, cy=105, w=180, h=12, angle_deg=3)
        assert _obb_iou(a, b) == pytest.approx(_obb_iou(b, a), abs=1e-6)


# ---------------------------------------------------------------------------
# Angle error
# ---------------------------------------------------------------------------

class TestAngleError:
    def test_identical_angles_give_zero_error(self):
        assert _angle_error_deg(45.0, 45.0) == pytest.approx(0.0)

    def test_error_accounts_for_180_degree_symmetry(self):
        # 5° and 185° are the same streak direction
        assert _angle_error_deg(5.0, 185.0) == pytest.approx(0.0, abs=1e-6)

    def test_small_angle_difference_is_correct(self):
        assert _angle_error_deg(10.0, 15.0) == pytest.approx(5.0)

    def test_error_clamped_to_90_degrees(self):
        err = _angle_error_deg(0.0, 90.0)
        assert err == pytest.approx(90.0)
        # 91° error wraps to 89°
        assert _angle_error_deg(0.0, 91.0) == pytest.approx(89.0)


# ---------------------------------------------------------------------------
# Precision / Recall / F1
# ---------------------------------------------------------------------------

class TestComputePRF:
    def test_perfect_predictions_give_precision_recall_one(self):
        preds = [_pred()]
        gts = [_gt()]
        p, r, f = _compute_prf(preds, gts, iou_threshold=0.5)
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)
        assert f == pytest.approx(1.0)

    def test_empty_predictions_give_zero_metrics(self):
        p, r, f = _compute_prf([], [_gt()], iou_threshold=0.5)
        assert p == 0.0
        assert r == 0.0
        assert f == 0.0

    def test_no_overlapping_predictions_give_zero_recall(self):
        preds = [_pred(cx=500, cy=500)]   # far from GT
        gts = [_gt(cx=0, cy=0)]
        p, r, f = _compute_prf(preds, gts, iou_threshold=0.5)
        assert r == pytest.approx(0.0)
        assert p == pytest.approx(0.0)

    def test_one_tp_one_fp_gives_half_precision(self):
        # Two predictions, only one matches the single GT
        preds = [
            _pred(cx=100, cy=100, confidence=0.9),          # matches GT
            _pred(cx=900, cy=900, confidence=0.5),           # no GT nearby
        ]
        gts = [_gt(cx=100, cy=100)]
        p, r, f = _compute_prf(preds, gts, iou_threshold=0.5)
        assert p == pytest.approx(0.5)
        assert r == pytest.approx(1.0)

    def test_one_tp_one_fn_gives_half_recall(self):
        preds = [_pred(cx=100, cy=100)]                      # matches first GT only
        gts = [_gt(cx=100, cy=100), _gt(cx=500, cy=500)]    # second GT undetected
        p, r, f = _compute_prf(preds, gts, iou_threshold=0.5)
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Average Precision (mAP)
# ---------------------------------------------------------------------------

class TestComputeAP:
    def test_perfect_predictions_give_ap_one(self):
        preds = [_pred(confidence=0.9)]
        gts = [_gt()]
        ap = _compute_ap(preds, gts, iou_threshold=0.5)
        assert ap == pytest.approx(1.0, abs=1e-4)

    def test_empty_inputs_give_ap_zero(self):
        assert _compute_ap([], [_gt()], 0.5) == 0.0
        assert _compute_ap([_pred()], [], 0.5) == 0.0

    def test_ap50_greater_than_or_equal_ap75(self):
        preds = [_pred(cx=100, cy=100, confidence=0.9)]
        gts = [_gt(cx=105, cy=100)]   # slight offset reduces IoU
        ap50 = _compute_ap(preds, gts, 0.5)
        ap75 = _compute_ap(preds, gts, 0.75)
        assert ap50 >= ap75


# ---------------------------------------------------------------------------
# Mean angle error
# ---------------------------------------------------------------------------

class TestMeanAngleError:
    def test_matched_predictions_with_known_offset(self):
        # Use a compact box (80×60) so a 5° rotation does not drop IoU below 0.5
        preds = [_pred(angle_deg=10.0, w=80, h=60, length=80)]
        gts = [_gt(angle_deg=5.0, w=80, h=60, length=80)]
        err = _compute_mean_angle_error(preds, gts, iou_threshold=0.5)
        assert err == pytest.approx(5.0, abs=0.1)

    def test_no_matches_gives_zero(self):
        preds = [_pred(cx=1000, cy=1000)]
        gts = [_gt(cx=0, cy=0)]
        err = _compute_mean_angle_error(preds, gts, iou_threshold=0.5)
        assert err == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Full evaluate()
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_returns_all_required_keys(self):
        result = evaluate([_pred()], [_gt()])
        required = {"precision", "recall", "f1", "map_50", "map_75",
                    "mean_angle_error_deg", "per_band"}
        assert required.issubset(result.keys())
        assert set(result["per_band"].keys()) == {"short", "medium", "long"}

    def test_perfect_predictions_give_full_scores(self):
        result = evaluate([_pred()], [_gt()])
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        assert result["map_50"] == pytest.approx(1.0, abs=1e-4)

    def test_per_band_splits_by_streak_length(self):
        preds = [
            _pred(cx=100, cy=100, length=80.0),    # short
            _pred(cx=300, cy=100, length=250.0),   # medium
            _pred(cx=500, cy=100, length=500.0),   # long
        ]
        gts = [
            _gt(cx=100, cy=100, length=80.0),
            _gt(cx=300, cy=100, length=250.0),
            _gt(cx=500, cy=100, length=500.0),
        ]
        result = evaluate(preds, gts)
        for band in ("short", "medium", "long"):
            assert result["per_band"][band]["recall"] == pytest.approx(1.0), band

    def test_empty_predictions_all_zeros(self):
        result = evaluate([], [_gt()])
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["map_50"] == 0.0


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

class TestBenchmarkHelpers:
    def test_load_ground_truth_from_coco_json(self, tmp_path):
        from eval.benchmark import load_ground_truth

        coco = {
            "images": [{"id": 1, "file_name": "img001.fits", "width": 512, "height": 512}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 0, "iscrowd": 0,
                 "bbox": [50, 90, 200, 15],
                 "obb": [150.0, 97.5, 200.0, 15.0, 8.0],
                 "area": 3000},
            ],
            "categories": [{"id": 0, "name": "streak"}],
        }
        ann_file = tmp_path / "test.json"
        ann_file.write_text(json.dumps(coco))

        gts = load_ground_truth(ann_file)
        assert len(gts) == 1
        assert gts[0]["image_id"] == "img001.fits"
        assert gts[0]["obb"]["angle_deg"] == pytest.approx(8.0)
        assert gts[0]["streak_length_px"] == pytest.approx(200.0)

    def test_format_markdown_table_contains_headers(self):
        from eval.benchmark import format_markdown_table

        metrics = evaluate([_pred()], [_gt()])
        table = format_markdown_table(metrics, None)
        assert "Precision" in table
        assert "Recall" in table
        assert "mAP@0.5" in table
        assert "Target" in table

    def test_run_benchmark_saves_json(self, tmp_path):
        from eval.benchmark import run_benchmark

        coco = {
            "images": [{"id": 1, "file_name": "img001.fits", "width": 512, "height": 512}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 0, "iscrowd": 0,
                 "bbox": [50, 90, 200, 15],
                 "obb": [100.0, 100.0, 200.0, 10.0, 0.0],
                 "area": 2000},
            ],
            "categories": [{"id": 0, "name": "streak"}],
        }
        ann_file = tmp_path / "test.json"
        ann_file.write_text(json.dumps(coco))

        dino_preds = [
            {"image_id": "img001.fits", "confidence": 0.92,
             "obb": {"cx": 100, "cy": 100, "w": 200, "h": 10, "angle_deg": 0.0},
             "streak_length_px": 200.0}
        ]
        out_file = tmp_path / "benchmark.json"

        result = run_benchmark(
            annotations_path=ann_file,
            dino_predictions=dino_preds,
            output_path=out_file,
        )

        assert out_file.exists()
        saved = json.loads(out_file.read_text())
        assert saved["precision"] == pytest.approx(1.0)
        assert saved["recall"] == pytest.approx(1.0)
        assert "yolo_baseline" in saved


# ---------------------------------------------------------------------------
# evaluate_crossid
# Source: SkyTrack (colleague) — ComputeRMSResidual
# ---------------------------------------------------------------------------

def _make_crossid_det(
    identified: bool = True,
    atrk: float | None = 0.0,
    xtrk: float | None = 0.0,
    confidence: float = 0.85,
) -> dict:
    """Build a minimal detection with identification data for evaluate_crossid tests."""
    identifications = []
    if identified:
        identifications.append({
            "rank": 1,
            "satellite_name": "STARLINK-1007",
            "norad_id": 44713,
            "confidence": confidence,
            "atrk_arcsec": atrk,
            "xtrk_arcsec": xtrk,
        })
    return {"identifications": identifications}


class TestEvaluateCrossid:
    def test_returns_required_keys(self):
        from eval.metrics import evaluate_crossid
        result = evaluate_crossid([_make_crossid_det()])
        for key in (
            "n_detections", "n_identified", "identification_rate",
            "n_with_residuals", "atrk_rms_arcsec", "xtrk_rms_arcsec",
            "total_rms_arcsec", "top1_confidence_mean",
        ):
            assert key in result, f"Missing key: {key}"

    def test_empty_input(self):
        from eval.metrics import evaluate_crossid
        result = evaluate_crossid([])
        assert result["n_detections"] == 0
        assert result["identification_rate"] == pytest.approx(0.0)

    def test_all_identified(self):
        from eval.metrics import evaluate_crossid
        dets = [_make_crossid_det(identified=True) for _ in range(4)]
        result = evaluate_crossid(dets)
        assert result["n_detections"] == 4
        assert result["n_identified"] == 4
        assert result["identification_rate"] == pytest.approx(1.0)

    def test_none_identified(self):
        from eval.metrics import evaluate_crossid
        dets = [_make_crossid_det(identified=False) for _ in range(3)]
        result = evaluate_crossid(dets)
        assert result["n_identified"] == 0
        assert result["identification_rate"] == pytest.approx(0.0)

    def test_partial_identification(self):
        from eval.metrics import evaluate_crossid
        dets = [_make_crossid_det(identified=True), _make_crossid_det(identified=False)]
        result = evaluate_crossid(dets)
        assert result["identification_rate"] == pytest.approx(0.5)

    def test_atrk_xtrk_rms_zero_residuals(self):
        """Perfect prediction → both RMS residuals are zero."""
        from eval.metrics import evaluate_crossid
        dets = [_make_crossid_det(atrk=0.0, xtrk=0.0) for _ in range(3)]
        result = evaluate_crossid(dets)
        assert result["atrk_rms_arcsec"] == pytest.approx(0.0)
        assert result["xtrk_rms_arcsec"] == pytest.approx(0.0)

    def test_atrk_rms_known_value(self):
        """3 detections with Atrk=[3,4,0], Xtrk=0 → atrk_rms = sqrt((9+16+0)/3) ≈ 2.89.

        evaluate_crossid rounds results to 2 decimal places, so use abs=0.01 tolerance.
        """
        from eval.metrics import evaluate_crossid
        import math
        dets = [
            _make_crossid_det(atrk=3.0, xtrk=0.0),
            _make_crossid_det(atrk=4.0, xtrk=0.0),
            _make_crossid_det(atrk=0.0, xtrk=0.0),
        ]
        result = evaluate_crossid(dets)
        expected = math.sqrt((9 + 16 + 0) / 3)
        assert result["atrk_rms_arcsec"] == pytest.approx(expected, abs=0.01)

    def test_total_rms_combines_both_axes(self):
        """total_rms = sqrt((atrk² + xtrk²) / n)."""
        from eval.metrics import evaluate_crossid
        import math
        dets = [_make_crossid_det(atrk=3.0, xtrk=4.0)]
        result = evaluate_crossid(dets)
        assert result["total_rms_arcsec"] == pytest.approx(5.0, rel=1e-4)

    def test_none_residuals_excluded(self):
        """Identifications without atrk/xtrk keys are excluded from RMS computation."""
        from eval.metrics import evaluate_crossid
        det_no_resid = {"identifications": [{"rank": 1, "confidence": 0.9}]}
        dets = [_make_crossid_det(atrk=4.0, xtrk=3.0), det_no_resid]
        result = evaluate_crossid(dets)
        assert result["n_with_residuals"] == 1
        assert result["total_rms_arcsec"] == pytest.approx(5.0, rel=1e-4)

    def test_top1_confidence_mean(self):
        from eval.metrics import evaluate_crossid
        dets = [
            _make_crossid_det(confidence=0.8),
            _make_crossid_det(confidence=0.6),
        ]
        result = evaluate_crossid(dets)
        assert result["top1_confidence_mean"] == pytest.approx(0.7, abs=1e-6)

    def test_det_without_identifications_key_handled(self):
        """Detections missing the 'identifications' key entirely should not crash."""
        from eval.metrics import evaluate_crossid
        dets = [{"confidence": 0.9, "obb": {}}, _make_crossid_det()]
        result = evaluate_crossid(dets)
        assert result["n_detections"] == 2
