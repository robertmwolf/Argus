"""Build a training annotation JSON from the session manifest.

Reads ``data/sessions/manifest.yaml`` and produces a merged COCO JSON
from all sources whose ``split`` field is ``train``.

This replaces the hard-coded ``build_all_train_json.py`` with a
manifest-driven approach that supports:

  - Per-scope mix ratios  (--mix-ratio scope_id:N)
  - Per-session include/exclude  (--include / --exclude session_id)
  - Versioned output filenames   (all_train_nodm_v3.json)
  - Dry-run mode to preview what will be merged without writing

Usage
-----
# Equivalent to the old build_all_train_json.py (all train sources, weight 1):
python scripts/build_training_json.py --output data/annotations/all_train_nodm_v3.json

# Fine-tune run: oversample a new scope 2× to compensate for small dataset size:
python scripts/build_training_json.py \\
    --mix-ratio newscope:2.0 \\
    --output data/annotations/all_train_ft_newscope.json

# Exclude a source for ablation:
python scripts/build_training_json.py \\
    --exclude frigate_train \\
    --output data/annotations/all_train_nodm_no_frigate.json

# Preview without writing:
python scripts/build_training_json.py --dry-run

Notes
-----
* Sources with bare filenames in their annotation JSON have those paths
  resolved to absolute paths using the ``raw_dir`` field in the manifest.
  This mirrors the path-fixing logic in build_all_train_json.py.

* Mix ratios are applied *before* the global merge.  A weight of 2.0
  duplicates the source's images and annotations.  A weight of 0.5
  samples 50% at random (seeded by --seed, default 42).  Fractional
  weights between 0 and 1 are supported; weights > 1 are rounded to
  the nearest integer repetition count with the remainder sampled.

* The manifest ``mix_weight`` field is the default; ``--mix-ratio``
  overrides it for that scope_id.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required: pip install pyyaml")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST_PATH = _REPO_ROOT / "data/sessions/manifest.yaml"
_ANN_DIR = _REPO_ROOT / "data/annotations"
_CANONICAL_CATEGORIES = [{"id": 1, "name": "streak", "supercategory": "satellite"}]

# Splits that are eligible to be included in training builds
_TRAIN_SPLITS = {"train"}


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def load_manifest(path: Path = _MANIFEST_PATH) -> list[dict]:
    """Return the list of source entries from the session manifest."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "sources" not in data:
        raise ValueError(f"Manifest at {path} must have a top-level 'sources' key")
    return data["sources"]


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _fix_bare_paths(images: list[dict], raw_dir: str) -> list[dict]:
    """Resolve bare filenames to absolute paths under raw_dir."""
    fixed = []
    for img in images:
        new = dict(img)
        fname = img["file_name"]
        if not fname.startswith("/"):
            new["file_name"] = f"{raw_dir}/{fname}"
        fixed.append(new)
    return fixed


def _load_source(entry: dict) -> dict:
    """Load and return a COCO dict for a single manifest entry.

    Applies path-fixing if the entry has a ``raw_dir``.  Merges the
    positives annotation_file and optional negatives_file into one dict.
    """
    ann_file = _REPO_ROOT / entry["annotation_file"]
    if not ann_file.exists():
        raise FileNotFoundError(
            f"Annotation file not found for session '{entry['session_id']}': {ann_file}"
        )

    data = _load_json(ann_file)
    raw_dir = entry.get("raw_dir")

    if raw_dir:
        data["images"] = _fix_bare_paths(data["images"], raw_dir)

    # Merge optional negatives file
    neg_file_rel = entry.get("negatives_file")
    if neg_file_rel:
        neg_path = _REPO_ROOT / neg_file_rel
        if neg_path.exists():
            neg = _load_json(neg_path)
            if raw_dir:
                neg["images"] = _fix_bare_paths(neg["images"], raw_dir)
            # Append images (negatives have no annotations)
            data = dict(data)
            data["images"] = list(data["images"]) + list(neg["images"])
        else:
            logger.warning("negatives_file not found, skipping: %s", neg_path)

    return data


# ---------------------------------------------------------------------------
# Mix-ratio application
# ---------------------------------------------------------------------------

def _apply_mix_ratio(coco: dict, weight: float, seed: int) -> dict:
    """Return a new COCO dict with the mix ratio applied.

    weight == 1.0  → unchanged
    weight == 2.0  → images and annotations duplicated (2 copies)
    weight == 0.5  → 50% random sample of images (with matching annotations)
    weight == 1.5  → 1 full copy + 50% random sample

    Annotation image_ids are adjusted for duplicated images to avoid
    collisions within the copy (the global merge will reassign all IDs).
    """
    if abs(weight - 1.0) < 1e-6:
        return coco  # no-op

    images = coco["images"]
    annotations = coco.get("annotations", [])

    if not images:
        return coco

    rng = random.Random(seed)

    # Build a lookup from image_id → annotations
    ann_by_img: dict[int, list[dict]] = {}
    for ann in annotations:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    full_copies = int(weight)          # whole repetitions
    fraction = weight - full_copies    # remainder (0.0 – 1.0)

    result_images: list[dict] = []
    result_annotations: list[dict] = []

    def _append_copy(imgs: list[dict], id_offset: int) -> None:
        for img in imgs:
            new_img = dict(img)
            old_id = img["id"]
            new_id = old_id + id_offset
            new_img["id"] = new_id
            result_images.append(new_img)
            for ann in ann_by_img.get(old_id, []):
                new_ann = dict(ann)
                new_ann["image_id"] = new_id
                result_annotations.append(new_ann)

    # Compute a per-source ID offset large enough not to collide within copies.
    # The global merge step will reassign all IDs sequentially anyway.
    max_id = max((img["id"] for img in images), default=0)
    id_step = max_id + 1

    for copy_idx in range(full_copies):
        _append_copy(images, id_offset=copy_idx * id_step)

    if fraction > 1e-6:
        sample_n = max(1, round(len(images) * fraction))
        sampled = rng.sample(images, min(sample_n, len(images)))
        _append_copy(sampled, id_offset=full_copies * id_step)

    return {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco.get("categories", _CANONICAL_CATEGORIES),
        "images": result_images,
        "annotations": result_annotations,
    }


# ---------------------------------------------------------------------------
# Global merge
# ---------------------------------------------------------------------------

def merge(sources: list[dict]) -> dict:
    """Merge multiple COCO dicts, reassigning all IDs sequentially."""
    all_images: list[dict] = []
    all_annotations: list[dict] = []
    next_img_id = 1
    next_ann_id = 1

    for src in sources:
        old_to_new: dict[int, int] = {}
        for img in src["images"]:
            new_img = dict(img)
            old_id = img["id"]
            new_img["id"] = next_img_id
            old_to_new[old_id] = next_img_id
            all_images.append(new_img)
            next_img_id += 1

        for ann in src.get("annotations", []):
            if ann["image_id"] not in old_to_new:
                continue  # orphaned — skip
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = old_to_new[ann["image_id"]]
            all_annotations.append(new_ann)
            next_ann_id += 1

    return {
        "info": {"description": "ARGUS merged training split", "version": "3.0"},
        "licenses": [],
        "categories": _CANONICAL_CATEGORIES,
        "images": all_images,
        "annotations": all_annotations,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a training JSON from the ARGUS session manifest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=_MANIFEST_PATH,
        help="Path to the session manifest YAML (default: data/sessions/manifest.yaml)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=_ANN_DIR / "all_train_nodm.json",
        help="Output path for the merged COCO JSON "
             "(default: data/annotations/all_train_nodm.json)",
    )
    p.add_argument(
        "--include",
        metavar="SESSION_ID",
        nargs="+",
        default=None,
        help="Only include these session_ids (whitelist).  "
             "Mutually exclusive with --exclude.",
    )
    p.add_argument(
        "--exclude",
        metavar="SESSION_ID",
        nargs="+",
        default=None,
        help="Exclude these session_ids (blacklist).  "
             "Mutually exclusive with --include.",
    )
    p.add_argument(
        "--mix-ratio",
        metavar="SCOPE_ID:WEIGHT",
        nargs="+",
        default=[],
        help="Override mix_weight for a scope.  E.g. --mix-ratio atwood:2.0 frigate:0.5.  "
             "Weight 2.0 duplicates the source; 0.5 keeps 50%%.  "
             "Overrides the manifest mix_weight for that scope.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for fractional mix-ratio sampling (default: 42)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be merged without writing any files.",
    )
    return p.parse_args()


def _parse_mix_ratios(mix_ratio_args: list[str]) -> dict[str, float]:
    """Parse ['scope:1.5', 'other:0.5'] → {'scope': 1.5, 'other': 0.5}."""
    result: dict[str, float] = {}
    for token in mix_ratio_args:
        parts = token.rsplit(":", 1)
        if len(parts) != 2:
            raise ValueError(
                f"--mix-ratio must be in scope_id:weight format, got: {token!r}"
            )
        scope_id, weight_str = parts
        try:
            weight = float(weight_str)
        except ValueError:
            raise ValueError(
                f"Weight must be a number, got: {weight_str!r} in {token!r}"
            )
        if weight <= 0:
            raise ValueError(f"Weight must be > 0, got {weight} in {token!r}")
        result[scope_id] = weight
    return result


def main() -> None:
    args = parse_args()

    if args.include and args.exclude:
        raise SystemExit("--include and --exclude are mutually exclusive")

    sources = load_manifest(args.manifest)
    mix_overrides = _parse_mix_ratios(args.mix_ratio)

    # Filter to train-eligible splits
    sources = [s for s in sources if s.get("split") in _TRAIN_SPLITS]

    # Apply whitelist / blacklist
    if args.include:
        include_set = set(args.include)
        sources = [s for s in sources if s["session_id"] in include_set]
        missing = include_set - {s["session_id"] for s in sources}
        if missing:
            raise SystemExit(
                f"--include referenced unknown or non-train session_ids: {missing}"
            )

    if args.exclude:
        exclude_set = set(args.exclude)
        sources = [s for s in sources if s["session_id"] not in exclude_set]

    if not sources:
        raise SystemExit("No sources selected — check manifest and filters.")

    logger.info("Sources to merge (%d):", len(sources))
    coco_sources: list[dict] = []

    for entry in sources:
        sid = entry["session_id"]
        scope = entry.get("scope_id", sid)

        # Determine effective mix weight
        weight = mix_overrides.get(scope, entry.get("mix_weight", 1.0))

        if args.dry_run:
            logger.info(
                "  [dry-run] %-30s  scope=%-15s  weight=%.2f",
                sid, scope, weight,
            )
            continue

        logger.info("Loading %-30s  scope=%-15s  weight=%.2f", sid, scope, weight)
        try:
            coco = _load_source(entry)
        except FileNotFoundError as e:
            logger.error("Skipping %s: %s", sid, e)
            continue

        n_img = len(coco["images"])
        n_ann = len(coco.get("annotations", []))
        logger.info("  → %d images, %d annotations (before mix ratio)", n_img, n_ann)

        coco = _apply_mix_ratio(coco, weight, seed=args.seed)

        n_img_after = len(coco["images"])
        if n_img_after != n_img:
            logger.info(
                "  → %d images after mix ratio %.2f", n_img_after, weight
            )

        coco_sources.append(coco)

    if args.dry_run:
        logger.info("[dry-run] No files written.")
        return

    if not coco_sources:
        raise SystemExit("All sources failed to load — check file paths.")

    logger.info("Merging %d source(s)...", len(coco_sources))
    merged = merge(coco_sources)

    n_img_total = len(merged["images"])
    n_ann_total = len(merged["annotations"])
    logger.info(
        "Merged: %d images, %d annotations", n_img_total, n_ann_total
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(merged, f)

    logger.info("Written: %s", args.output)


if __name__ == "__main__":
    main()
