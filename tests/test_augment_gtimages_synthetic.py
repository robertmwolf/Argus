"""Tests for scripts/augment_gtimages_synthetic.py."""

from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))

from augment_gtimages_synthetic import build_gtimages_synthetic_dataset


def _write_rgb(path: pathlib.Path, value: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.ones((96, 96, 3), dtype=np.uint8) * value
    cv2.imwrite(str(path), image)


def _make_obb(length: float, angle: float = 20.0) -> dict:
    return {"cx": 48.0, "cy": 48.0, "w": length, "h": 16.0, "angle_deg": angle}


def _write_fixture(data_root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    gt_dir = data_root / "GTImages"
    for idx in range(1, 7):
        _write_rgb(gt_dir / f"img{idx}.png", value=70 + idx)
    for idx in range(1, 4):
        _write_rgb(gt_dir / f"neg{idx}.png", value=60 + idx)

    labeled = {
        "images": [
            {"id": 1, "file_name": "img1.png", "width": 96, "height": 96, "date_captured": "2026-04-27T01:00:00"},
            {"id": 2, "file_name": "img2.png", "width": 96, "height": 96, "date_captured": "2026-04-27T01:01:00"},
            {"id": 3, "file_name": "img3.png", "width": 96, "height": 96, "date_captured": "2026-04-27T01:02:00"},
            {"id": 4, "file_name": "img4.png", "width": 96, "height": 96, "date_captured": "2026-04-27T01:03:00"},
            {"id": 5, "file_name": "img5.png", "width": 96, "height": 96, "date_captured": "2026-04-27T01:04:00"},
            {"id": 6, "file_name": "img6.png", "width": 96, "height": 96, "date_captured": "2026-04-27T01:05:00"},
        ],
        "annotations": [],
        "categories": [{"id": 1, "name": "satellite_streak"}],
    }
    lengths = [80.0, 85.0, 90.0, 30.0, 35.0, 40.0]
    for ann_id, length in enumerate(lengths, start=1):
        labeled["annotations"].append({
            "id": ann_id,
            "image_id": ann_id,
            "category_id": 1,
            "bbox": [8.0, 40.0, length, 16.0],
            "area": length * 16.0,
            "iscrowd": 0,
            "obb": _make_obb(length, angle=10.0 + ann_id),
            "attributes": {"norad_id": 40000 + ann_id},
        })

    negatives = {
        "images": [
            {"id": 1, "file_name": "neg1.png", "width": 96, "height": 96, "date_captured": "2026-04-27T02:00:00"},
            {"id": 2, "file_name": "neg2.png", "width": 96, "height": 96, "date_captured": "2026-04-27T02:01:00"},
            {"id": 3, "file_name": "neg3.png", "width": 96, "height": 96, "date_captured": "2026-04-27T02:02:00"},
        ],
        "annotations": [],
        "categories": [{"id": 1, "name": "satellite_streak"}],
    }

    ann_dir = data_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    labeled_path = ann_dir / "gtimages.json"
    negatives_path = ann_dir / "gtimages_negatives.json"
    labeled_path.write_text(json.dumps(labeled))
    negatives_path.write_text(json.dumps(negatives))
    return labeled_path, negatives_path


def test_builds_real_and_synthetic_tracks(tmp_path: pathlib.Path) -> None:
    data_root = tmp_path / "data"
    labeled_path, negatives_path = _write_fixture(data_root)
    output_dir = data_root / "annotations_out"
    synthetic_dir = data_root / "gtimages_synthetic"

    manifest = build_gtimages_synthetic_dataset(
        data_root=data_root,
        labeled_json=labeled_path,
        negatives_json=negatives_path,
        output_dir=output_dir,
        synthetic_dir=synthetic_dir,
        train_fraction=0.67,
        val_fraction=0.22,
        seed=123,
        synthetic_ratio=0.5,
        results_dir=tmp_path / "results",
    )

    train_real = json.loads((output_dir / "gtimages_train_real.json").read_text())
    paper = json.loads((output_dir / "gtimages_train_synth_paper_long.json").read_text())
    adapted = json.loads((output_dir / "gtimages_train_synth_adapted.json").read_text())
    val = json.loads((output_dir / "gtimages_val.json").read_text())
    test = json.loads((output_dir / "gtimages_test.json").read_text())

    assert manifest["normalization"] == "zscale"
    assert len(paper["images"]) > len(train_real["images"])
    assert len(adapted["images"]) > len(train_real["images"])
    assert all(not img["attributes"]["synthetic"] for img in val["images"])
    assert all(not img["attributes"]["synthetic"] for img in test["images"])

    synthetic_paper_anns = [
        ann for ann in paper["annotations"]
        if ann.get("attributes", {}).get("synthetic")
    ]
    assert synthetic_paper_anns
    attrs = synthetic_paper_anns[0]["attributes"]
    assert attrs["synthetic_track"] == "paper_long"
    assert attrs["source_dataset"] == "gtimages"
    assert "parent_image" in attrs
    assert "brightness_level" in attrs
    assert synthetic_dir.exists()


def test_generation_is_deterministic_for_seed(tmp_path: pathlib.Path) -> None:
    data_root = tmp_path / "data"
    labeled_path, negatives_path = _write_fixture(data_root)
    output_dir = data_root / "annotations_out"
    synthetic_dir = data_root / "gtimages_synthetic"

    build_gtimages_synthetic_dataset(
        data_root=data_root,
        labeled_json=labeled_path,
        negatives_json=negatives_path,
        output_dir=output_dir,
        synthetic_dir=synthetic_dir,
        train_fraction=0.67,
        val_fraction=0.22,
        seed=777,
        synthetic_ratio=0.5,
        results_dir=tmp_path / "results",
    )
    first = (output_dir / "gtimages_train_synth_adapted.json").read_text()

    build_gtimages_synthetic_dataset(
        data_root=data_root,
        labeled_json=labeled_path,
        negatives_json=negatives_path,
        output_dir=output_dir,
        synthetic_dir=synthetic_dir,
        train_fraction=0.67,
        val_fraction=0.22,
        seed=777,
        synthetic_ratio=0.5,
        results_dir=tmp_path / "results",
    )
    second = (output_dir / "gtimages_train_synth_adapted.json").read_text()

    assert first == second
