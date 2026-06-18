"""Tests for explicit durable and scratch dataset roots."""

import json
from pathlib import Path

from scripts.stage_dataset_files import main as stage_main
from training.data_paths import relative_source_path, resolve_source_path


def test_resolver_prefers_scratch_copy(tmp_path: Path) -> None:
    durable = tmp_path / "external"
    scratch = tmp_path / "scratch"
    relative = Path("raw/atwood/frame.fits")
    (durable / relative).parent.mkdir(parents=True)
    (scratch / relative).parent.mkdir(parents=True)
    (durable / relative).write_bytes(b"external")
    (scratch / relative).write_bytes(b"local")
    annotation = tmp_path / "annotations/train.json"

    resolved = resolve_source_path(str(relative), annotation, durable, scratch)

    assert resolved == scratch / relative


def test_resolver_falls_back_to_durable_root(tmp_path: Path) -> None:
    durable = tmp_path / "external"
    relative = Path("raw/atwood/frame.fits")
    (durable / relative).parent.mkdir(parents=True)
    (durable / relative).touch()

    resolved = resolve_source_path(
        str(relative), tmp_path / "train.json", durable, tmp_path / "empty"
    )

    assert resolved == durable / relative


def test_legacy_absolute_path_maps_to_scratch(tmp_path: Path) -> None:
    durable = tmp_path / "external"
    scratch = tmp_path / "scratch"
    relative = Path("raw/frame.fits")
    (scratch / relative).parent.mkdir(parents=True)
    (scratch / relative).touch()

    resolved = resolve_source_path(
        str(durable / relative), tmp_path / "train.json", durable, scratch
    )

    assert resolved == scratch / relative
    assert relative_source_path(str(durable / relative), durable) == relative


def test_stage_command_preserves_relative_layout(
    tmp_path: Path, monkeypatch
) -> None:
    durable = tmp_path / "external"
    scratch = tmp_path / "scratch"
    relative = Path("raw/atwood/frame.fits")
    (durable / relative).parent.mkdir(parents=True)
    (durable / relative).write_bytes(b"fits-data")
    annotations = tmp_path / "train.json"
    annotations.write_text(json.dumps({"images": [{"id": 1, "file_name": str(relative)}]}))
    monkeypatch.setattr(
        "sys.argv",
        [
            "stage_dataset_files.py",
            "--annotations", str(annotations),
            "--data-root", str(durable),
            "--scratch-root", str(scratch),
        ],
    )

    assert stage_main() == 0
    assert (scratch / relative).read_bytes() == b"fits-data"
