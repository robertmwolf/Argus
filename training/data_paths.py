"""Portable source-data path resolution for training and evaluation."""

from __future__ import annotations

import os
from pathlib import Path


def configured_root(value: str | Path | None, env_name: str) -> Path | None:
    """Return an explicit, exported, or project-.env directory."""
    raw = str(value) if value is not None else os.environ.get(env_name, "")
    if not raw:
        try:
            from dotenv import dotenv_values

            raw = str(dotenv_values(Path(__file__).resolve().parents[1] / ".env").get(env_name) or "")
        except ImportError:
            pass
    return Path(raw).expanduser().resolve() if raw else None


def relative_source_path(file_name: str, data_root: str | Path | None = None) -> Path | None:
    """Return the portable path for a manifest entry when it is under data_root."""
    source = Path(file_name).expanduser()
    if not source.is_absolute():
        return source
    root = configured_root(data_root, "ARGUS_DATA_ROOT")
    if root is None:
        return None
    try:
        return source.resolve().relative_to(root)
    except ValueError:
        return None


def resolve_source_path(
    file_name: str,
    annotation_file: str | Path,
    data_root: str | Path | None = None,
    scratch_root: str | Path | None = None,
) -> Path:
    """Resolve a manifest image, preferring a staged local copy.

    Relative manifest paths are interpreted against ``ARGUS_SCRATCH_ROOT`` first,
    then ``ARGUS_DATA_ROOT``. Absolute legacy paths under the configured data root
    receive the same scratch-first treatment.
    """
    source = Path(file_name).expanduser()
    durable = configured_root(data_root, "ARGUS_DATA_ROOT")
    scratch = configured_root(scratch_root, "ARGUS_SCRATCH_ROOT")
    relative = relative_source_path(file_name, durable)

    candidates: list[Path] = []
    if relative is not None:
        if scratch is not None:
            candidates.append(scratch / relative)
        if durable is not None:
            candidates.append(durable / relative)
    if source.is_absolute():
        candidates.append(source)
    else:
        annotation_dir = Path(annotation_file).expanduser().resolve().parent
        candidates.extend((annotation_dir / source, annotation_dir.parent / source))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else source
