#!/usr/bin/env python3
"""Copy files referenced by annotation manifests into a local scratch mirror."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.data_paths import configured_root, relative_source_path, resolve_source_path


def _referenced_files(annotation_files: list[Path], data_root: Path) -> dict[Path, Path]:
    """Return source-to-relative-path mappings for all referenced images."""
    files: dict[Path, Path] = {}
    for annotation_file in annotation_files:
        coco = json.loads(annotation_file.read_text())
        for image in coco.get("images", []):
            file_name = str(image["file_name"])
            relative = relative_source_path(file_name, data_root)
            if relative is None or relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    f"{annotation_file}: {file_name!r} is not relative to data root {data_root}"
                )
            source = resolve_source_path(file_name, annotation_file, data_root=data_root)
            if not source.is_file():
                raise FileNotFoundError(f"Referenced source file does not exist: {source}")
            files[source.resolve()] = relative
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", nargs="+", required=True, type=Path,
                        help="Train/validation COCO manifests to stage together.")
    parser.add_argument("--data-root", default=None,
                        help="Durable dataset root (or set ARGUS_DATA_ROOT).")
    parser.add_argument("--scratch-root", default=None,
                        help="Local mirror destination (or set ARGUS_SCRATCH_ROOT).")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--refresh", action="store_true",
                        help="Replace staged files even when size already matches.")
    args = parser.parse_args()

    data_root = configured_root(args.data_root, "ARGUS_DATA_ROOT")
    scratch_root = configured_root(args.scratch_root, "ARGUS_SCRATCH_ROOT")
    if data_root is None or scratch_root is None:
        parser.error("configure both --data-root/ARGUS_DATA_ROOT and --scratch-root/ARGUS_SCRATCH_ROOT")
    if data_root == scratch_root:
        parser.error("data root and scratch root must be different directories")

    files = _referenced_files(args.annotations, data_root)

    def copy(item: tuple[Path, Path]) -> bool:
        source, relative = item
        destination = scratch_root / relative
        if not args.refresh and destination.is_file() and destination.stat().st_size == source.stat().st_size:
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return True

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        copied = sum(pool.map(copy, files.items()))
    print(f"Staged {copied} of {len(files)} referenced files in {scratch_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
