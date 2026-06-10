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

    def test_load_ground_truth_accepts_dict_obb(self, tmp_path):
        from eval.benchmark import load_ground_truth

        coco = {
            "images": [{"id": 1, "file_name": "img001.fits", "width": 512, "height": 512}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0,
                 "bbox": [50, 90, 200, 15],
                 "obb": {"cx": 150.0, "cy": 97.5, "w": 200.0, "h": 15.0, "angle_deg": 8.0},
                 "area": 3000},
            ],
            "categories": [{"id": 1, "name": "streak"}],
        }
        ann_file = tmp_path / "test.json"
        ann_file.write_text(json.dumps(coco))

        gts = load_ground_truth(ann_file)

        assert len(gts) == 1
        assert gts[0]["obb"]["cx"] == pytest.approx(150.0)
        assert gts[0]["streak_length_px"] == pytest.approx(200.0)

    def test_format_markdown_table_contains_headers(self):
        from eval.benchmark import format_markdown_table

        metrics = evaluate([_pred()], [_gt()])
        table = format_markdown_table(metrics)
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


# ---------------------------------------------------------------------------
# extract_method_predictions
# ---------------------------------------------------------------------------

def _make_group(method, confidence, cx=100, cy=100, w=200, h=10, angle=5.0, length=200.0):
    """Build a minimal pipeline-style grouped detection dict."""
    return {
        "obb": {"cx": cx, "cy": cy, "w": w, "h": h, "angle_deg": angle},
        "streak_length_px": length,
        "sources": [{"method": method, "confidence": confidence}],
    }


class TestExtractMethodPredictions:
    def test_single_method_group_produces_that_method_and_unified(self):
        from eval.metrics import extract_method_predictions
        groups = [_make_group("dinov3_vitb", 0.90)]
        result = extract_method_predictions(groups, "img1.fits")
        assert "dinov3_vitb" in result
        assert "unified" in result
        assert len(result["dinov3_vitb"]) == 1
        assert len(result["unified"]) == 1

    def test_unified_confidence_matches_weighted_formula_single_source(self):
        from eval.metrics import extract_method_predictions
        from inference.confidence import compute_unified_confidence
        groups = [_make_group("dinov3_vitb", 0.85)]
        result = extract_method_predictions(groups, "img1.fits")
        expected = compute_unified_confidence([{"method": "dinov3_vitb", "confidence": 0.85}])["score"]
        assert result["unified"][0]["confidence"] == pytest.approx(expected)

    def test_two_methods_produce_higher_unified_than_one(self):
        from eval.metrics import extract_method_predictions
        from inference.confidence import compute_unified_confidence
        group = {
            "obb": {"cx": 100, "cy": 100, "w": 200, "h": 10, "angle_deg": 5.0},
            "streak_length_px": 200.0,
            "sources": [
                {"method": "dinov3_vitb", "confidence": 0.80},
                {"method": "astride",     "confidence": 0.60},
            ],
        }
        result = extract_method_predictions([group], "img1.fits")
        expected = compute_unified_confidence([
            {"method": "dinov3_vitb", "confidence": 0.80},
            {"method": "astride",     "confidence": 0.60},
        ])["score"]
        assert result["unified"][0]["confidence"] == pytest.approx(expected, abs=1e-6)
        # Two detectors should produce a higher score than either alone
        single = compute_unified_confidence([{"method": "dinov3_vitb", "confidence": 0.80}])["score"]
        assert result["unified"][0]["confidence"] > single

    def test_multiple_groups_accumulate_correctly(self):
        from eval.metrics import extract_method_predictions
        groups = [
            _make_group("dinov3_vitb", 0.90, cx=100),
            _make_group("opencv",      0.70, cx=500),
        ]
        result = extract_method_predictions(groups, "img1.fits")
        assert len(result["dinov3_vitb"]) == 1
        assert len(result["opencv"]) == 1
        assert len(result["unified"]) == 2

    def test_synthetic_unified_source_in_input_is_ignored(self):
        """If pipeline already added a 'unified' source, it should not double-count."""
        from eval.metrics import extract_method_predictions
        group = {
            "obb": {"cx": 100, "cy": 100, "w": 200, "h": 10, "angle_deg": 5.0},
            "streak_length_px": 200.0,
            "sources": [
                {"method": "unified",     "confidence": 0.92},  # pre-existing
                {"method": "dinov3_vitb", "confidence": 0.80},
                {"method": "astride",     "confidence": 0.60},
            ],
        }
        result = extract_method_predictions([group], "img1.fits")
        # Individual methods should be extracted
        assert len(result["dinov3_vitb"]) == 1
        assert len(result["astride"]) == 1
        # Unified should be recomputed from the two individual methods only
        from inference.confidence import compute_unified_confidence
        expected = compute_unified_confidence([
            {"method": "dinov3_vitb", "confidence": 0.80},
            {"method": "astride",     "confidence": 0.60},
        ])["score"]
        assert result["unified"][0]["confidence"] == pytest.approx(expected, abs=1e-6)

    def test_group_without_obb_is_skipped(self):
        from eval.metrics import extract_method_predictions
        group = {"streak_length_px": 100.0, "sources": [{"method": "opencv", "confidence": 0.8}]}
        result = extract_method_predictions([group], "img1.fits")
        assert result["unified"] == []
        assert "opencv" not in result

    def test_image_id_is_stamped_on_all_predictions(self):
        from eval.metrics import extract_method_predictions
        groups = [_make_group("opencv", 0.75)]
        result = extract_method_predictions(groups, "frame42.fits")
        assert result["opencv"][0]["image_id"] == "frame42.fits"
        assert result["unified"][0]["image_id"] == "frame42.fits"

    def test_unified_confidence_capped_at_0_99(self):
        from eval.metrics import extract_method_predictions
        # Four very high-confidence detections — noisy-OR would exceed 0.99
        group = {
            "obb": {"cx": 100, "cy": 100, "w": 200, "h": 10, "angle_deg": 0.0},
            "streak_length_px": 200.0,
            "sources": [
                {"method": "dinov3_vitb", "confidence": 0.99},
                {"method": "opencv",      "confidence": 0.98},
                {"method": "astride",     "confidence": 0.97},
            ],
        }
        result = extract_method_predictions([group], "img1.fits")
        assert result["unified"][0]["confidence"] <= 0.99


# ---------------------------------------------------------------------------
# format_comparison_table
# ---------------------------------------------------------------------------

class TestFormatComparisonTable:
    def _make_metrics(self, precision=0.9, recall=0.8, f1=0.85, map50=0.75, map75=0.60):
        return {
            "precision": precision, "recall": recall, "f1": f1,
            "map_50": map50, "map_75": map75,
            "mean_angle_error_deg": 2.5,
            "per_band": {
                "short":  {"precision": 0.8, "recall": 0.7, "f1": 0.75},
                "medium": {"precision": 0.9, "recall": 0.85, "f1": 0.87},
                "long":   {"precision": 0.95, "recall": 0.90, "f1": 0.92},
            },
        }

    def test_unified_appears_first(self):
        from eval.benchmark import format_comparison_table
        method_metrics = {
            "opencv":  self._make_metrics(),
            "unified": self._make_metrics(precision=0.95),
        }
        table = format_comparison_table(method_metrics)
        unified_pos = table.index("Unified")
        opencv_pos = table.index("OpenCV")
        assert unified_pos < opencv_pos

    def test_table_contains_all_methods(self):
        from eval.benchmark import format_comparison_table
        method_metrics = {
            "unified":     self._make_metrics(),
            "dinov3_vitb": self._make_metrics(),
            "opencv":      self._make_metrics(),
        }
        table = format_comparison_table(method_metrics)
        assert "Unified" in table
        assert "DINOv3 ViT-B" in table
        assert "OpenCV" in table

    def test_table_contains_metric_rows(self):
        from eval.benchmark import format_comparison_table
        table = format_comparison_table({"unified": self._make_metrics()})
        for label in ("Precision", "Recall", "F1", "mAP@0.5", "mAP@0.75", "Angle error"):
            assert label in table

    def test_per_band_section_present(self):
        from eval.benchmark import format_comparison_table
        table = format_comparison_table({"unified": self._make_metrics()})
        assert "Per-band" in table
        assert "Short" in table
        assert "Medium" in table
        assert "Long" in table


# ---------------------------------------------------------------------------
# run_multi_method_benchmark
# ---------------------------------------------------------------------------

class TestRunMultiMethodBenchmark:
    def _make_coco(self, tmp_path):
        coco = {
            "images": [{"id": 1, "file_name": "img001.fits", "width": 512, "height": 512}],
            "annotations": [{
                "id": 1, "image_id": 1, "category_id": 0, "iscrowd": 0,
                "bbox": [50, 90, 200, 15],
                "obb": [150.0, 97.5, 200.0, 15.0, 0.0],
                "area": 3000,
            }],
            "categories": [{"id": 0, "name": "streak"}],
        }
        ann_file = tmp_path / "test.json"
        ann_file.write_text(json.dumps(coco))
        return ann_file

    def _make_method_preds(self, image_id="img001.fits"):
        obb = {"cx": 150.0, "cy": 97.5, "w": 200.0, "h": 15.0, "angle_deg": 0.0}
        def _p(conf):
            return {"image_id": image_id, "confidence": conf, "obb": obb, "streak_length_px": 200.0}
        return {
            "unified":     [_p(0.95)],
            "dinov3_vitb": [_p(0.90)],
            "astride":     [_p(0.75)],
        }

    def test_saves_json_with_methods_key(self, tmp_path):
        from eval.benchmark import run_multi_method_benchmark
        ann = self._make_coco(tmp_path)
        out = tmp_path / "multi.json"
        result = run_multi_method_benchmark(
            annotations_path=ann,
            method_predictions=self._make_method_preds(),
            output_path=out,
        )
        assert out.exists()
        saved = json.loads(out.read_text())
        assert "methods" in saved
        assert "unified" in saved["methods"]
        assert "dinov3_vitb" in saved["methods"]

    def test_unified_metrics_present(self, tmp_path):
        from eval.benchmark import run_multi_method_benchmark
        ann = self._make_coco(tmp_path)
        out = tmp_path / "multi.json"
        result = run_multi_method_benchmark(
            annotations_path=ann,
            method_predictions=self._make_method_preds(),
            output_path=out,
        )
        assert "unified" in result["methods"]
        u = result["methods"]["unified"]
        assert "precision" in u
        assert "recall" in u
        assert "confusion_matrix_png" in u

    def test_confusion_matrix_pngs_created(self, tmp_path):
        from eval.benchmark import run_multi_method_benchmark
        ann = self._make_coco(tmp_path)
        out = tmp_path / "multi.json"
        run_multi_method_benchmark(
            annotations_path=ann,
            method_predictions=self._make_method_preds(),
            output_path=out,
        )
        cm_dir = tmp_path / "confusion_matrices"
        assert cm_dir.exists()
        pngs = list(cm_dir.glob("confusion_matrix_*.png"))
        assert len(pngs) == 3  # unified, dinov3_vitb, opencv (or similar 3 methods)

    def test_raises_without_predictions_or_pipeline_flag(self, tmp_path):
        from eval.benchmark import run_multi_method_benchmark
        ann = self._make_coco(tmp_path)
        with pytest.raises(ValueError, match="method_predictions"):
            run_multi_method_benchmark(
                annotations_path=ann,
                method_predictions=None,
                run_pipeline=False,
            )


# ---------------------------------------------------------------------------
# line_metrics
# ---------------------------------------------------------------------------

class TestLineMetrics:
    def test_line_segment_prediction_matches_centerline_gt(self):
        from eval.line_metrics import evaluate_line_segments

        ground_truth = [{
            "image_id": "img001.fits",
            "line_segment": {
                "x1": 10.0, "y1": 20.0,
                "x2": 110.0, "y2": 20.0,
                "angle_deg": 0.0,
                "length_px": 100.0,
            },
            "streak_length_px": 100.0,
        }]
        predictions = [{
            "image_id": "img001.fits",
            "confidence": 0.9,
            "line_segment": {
                "x1": 11.0, "y1": 22.0,
                "x2": 109.0, "y2": 22.0,
                "angle_deg": 0.0,
                "length_px": 98.0,
            },
        }]

        metrics = evaluate_line_segments(predictions, ground_truth, tolerance_px=3.0)

        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)
        assert metrics["mean_angle_error_deg"] == pytest.approx(0.0)

    def test_obb_prediction_can_be_compared_as_centerline(self):
        from eval.line_metrics import evaluate_line_segments

        ground_truth = [{
            "image_id": "img001.fits",
            "line_segment": {
                "x1": 50.0, "y1": 100.0,
                "x2": 150.0, "y2": 100.0,
                "angle_deg": 0.0,
                "length_px": 100.0,
            },
            "streak_length_px": 100.0,
        }]
        predictions = [{
            "image_id": "img001.fits",
            "confidence": 0.8,
            "obb": {"cx": 100.0, "cy": 100.0, "w": 100.0, "h": 10.0, "angle_deg": 0.0},
            "streak_length_px": 100.0,
        }]

        metrics = evaluate_line_segments(predictions, ground_truth, tolerance_px=1.0)

        assert metrics["tp"] == 1
        assert metrics["fp"] == 0
        assert metrics["fn"] == 0
