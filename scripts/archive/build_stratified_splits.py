"""Build geometry-stratified train / val / test splits from the Atwood corpus.

Reads the feature table produced by ``extract_streak_features.py`` and
produces three COCO JSON files with balanced representation across the full
morphology space (streak length band × SNR class).

Strategy
--------
* All Atwood annotated images are pooled across all sessions (nights 1+2+Geo).
  Splits are NOT tied to individual nights — a single image from Night 1 may
  end up in val while another from Night 1 goes to train.
* Primary stratification cell: ``band × snr_class``  (e.g. long×bright,
  medium×null, short×medium …).
* Within each cell images are shuffled with a fixed seed and split 70/15/15.
* Cells with fewer than 3 images are assigned entirely to training.
* Negative images (no annotation) are split randomly at the same ratio using
  the same seed.
* Split assignments are written at the *image* level: all annotations
  belonging to an image travel with it.

Output files (relative to repo root)
--------------------------------------
  data/annotations/atwood_train.json
  data/annotations/val_atwood.json
  data/annotations/test_atwood.json
  data/features/atwood_split_summary.json   (cell-by-cell breakdown)

Usage
-----
python scripts/build_stratified_splits.py \\
    --features data/features/atwood_streak_features.csv

# Custom output directory or ratio:
python scripts/build_stratified_splits.py \\
    --features data/features/atwood_streak_features.csv \\
    --output-dir data/annotations \\
    --ratio 0.70 0.15 0.15 \\
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML required: pip install pyyaml")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST_PATH = _REPO_ROOT / "data/sessions/manifest.yaml"
_DEFAULT_FEATURES = _REPO_ROOT / "data/features/atwood_streak_features.csv"
_DEFAULT_OUT_DIR = _REPO_ROOT / "data/annotations"
_CANONICAL_CATEGORIES = [{"id": 1, "name": "streak", "supercategory": "satellite"}]

# SNR cell order for deterministic iteration
_BAND_ORDER = ["short", "medium", "long"]
_SNR_ORDER = ["faint", "medium", "bright", "null"]


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _load_features(path: Path) -> list[dict]:
    import csv
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    logger.info("Loaded %d feature rows from %s", len(rows), path)
    return rows


# ---------------------------------------------------------------------------
# Manifest and COCO loading
# ---------------------------------------------------------------------------

def _load_manifest(path: Path) -> list[dict]:
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return data["sources"]


def _load_source(entry: dict) -> dict:
    """Load a COCO dict for one manifest entry, resolving bare paths."""
    ann_path = _REPO_ROOT / entry["annotation_file"]
    if not ann_path.exists():
        logger.warning("Annotation file missing, skipping: %s", ann_path)
        return {"images": [], "annotations": []}

    with open(ann_path) as fh:
        coco = json.load(fh)

    raw_dir = entry.get("raw_dir")
    if raw_dir:
        for img in coco["images"]:
            fn = img["file_name"]
            if not fn.startswith("/"):
                img["file_name"] = f"{raw_dir}/{fn}"

    return coco


def _load_negatives(entry: dict) -> list[dict]:
    """Return a list of negative image dicts for an entry (no annotations)."""
    neg_rel = entry.get("negatives_file")
    if not neg_rel:
        return []
    neg_path = _REPO_ROOT / neg_rel
    if not neg_path.exists():
        return []

    with open(neg_path) as fh:
        data = json.load(fh)

    raw_dir = entry.get("raw_dir")
    images = data.get("images", [])
    if raw_dir:
        for img in images:
            fn = img["file_name"]
            if not fn.startswith("/"):
                img["file_name"] = f"{raw_dir}/{fn}"

    return images


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def _stratified_split(
    image_ids: list[int],
    ratio: tuple[float, float, float],
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Split a list of image IDs into (train, val, test) using the given ratio.

    Args:
        image_ids: List of image IDs to split.
        ratio: (train_frac, val_frac, test_frac) — must sum to ≤ 1.
        seed: Random seed for reproducibility.

    Returns:
        Three lists: (train_ids, val_ids, test_ids).
    """
    rng = random.Random(seed)
    ids = list(image_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_val = max(1, round(n * ratio[1])) if n >= 3 else 0
    n_test = max(1, round(n * ratio[2])) if n >= 3 else 0
    n_train = n - n_val - n_test

    return ids[:n_train], ids[n_train: n_train + n_val], ids[n_train + n_val:]


# ---------------------------------------------------------------------------
# COCO JSON assembly
# ---------------------------------------------------------------------------

def _build_coco(
    image_ids: set[int],
    images: list[dict],
    annotations: list[dict],
    split_name: str,
) -> dict:
    """Build a COCO dict from a selected set of image IDs."""
    selected_images = [img for img in images if img["id"] in image_ids]
    selected_anns = [ann for ann in annotations if ann["image_id"] in image_ids]

    # Re-number IDs sequentially (keeps file sizes smaller and avoids ID gaps)
    old_to_new_img: dict[int, int] = {}
    new_images = []
    for new_id, img in enumerate(selected_images, start=1):
        new_img = dict(img)
        old_to_new_img[img["id"]] = new_id
        new_img["id"] = new_id
        new_images.append(new_img)

    new_annotations = []
    for new_id, ann in enumerate(selected_anns, start=1):
        new_ann = dict(ann)
        new_ann["id"] = new_id
        new_ann["image_id"] = old_to_new_img[ann["image_id"]]
        new_annotations.append(new_ann)

    return {
        "info": {
            "description": f"ARGUS Atwood geometry-stratified split: {split_name}",
            "version": "1.0",
            "date_created": _today(),
        },
        "licenses": [],
        "categories": _CANONICAL_CATEGORIES,
        "images": new_images,
        "annotations": new_annotations,
    }


def _today() -> str:
    from datetime import date
    return date.today().isoformat()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build geometry-stratified Atwood train/val/test splits.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--features",
        type=Path,
        default=_DEFAULT_FEATURES,
        help="Feature CSV from extract_streak_features.py "
             "(default: data/features/atwood_streak_features.csv)",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=_MANIFEST_PATH,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        help="Directory to write output COCO JSONs (default: data/annotations)",
    )
    p.add_argument(
        "--ratio",
        type=float,
        nargs=3,
        default=[0.70, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split ratio (default: 0.70 0.15 0.15)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).  The test split is frozen at this seed.",
    )
    p.add_argument(
        "--scope",
        default="atwood",
        help="scope_id to process (default: atwood)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print split statistics without writing files.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ratio = tuple(args.ratio)
    if abs(sum(ratio) - 1.0) > 0.01:
        raise SystemExit(
            f"Ratios must sum to 1.0; got {ratio} (sum={sum(ratio):.3f})"
        )

    if not args.features.exists():
        raise SystemExit(
            f"Feature CSV not found: {args.features}\n"
            "Run: python scripts/extract_streak_features.py first."
        )

    feature_rows = _load_features(args.features)

    # ---------------------------------------------------------------------------
    # Load all Atwood annotation sources
    # ---------------------------------------------------------------------------
    sources = _load_manifest(args.manifest)
    atwood_sources = [
        s for s in sources
        if s.get("scope_id") == args.scope
        and s.get("split") in {"train", "holdout", "val"}
    ]
    if not atwood_sources:
        raise SystemExit(
            f"No sources found for scope_id={args.scope!r} in the manifest."
        )

    all_images: list[dict] = []
    all_annotations: list[dict] = []
    all_negatives: list[dict] = []

    # Accumulate with globally unique IDs across sessions
    next_img_id = 1
    next_ann_id = 1

    for entry in atwood_sources:
        coco = _load_source(entry)

        old_to_new: dict[int, int] = {}
        for img in coco["images"]:
            old_to_new[img["id"]] = next_img_id
            new_img = dict(img)
            new_img["id"] = next_img_id
            new_img["_session_id"] = entry["session_id"]
            all_images.append(new_img)
            next_img_id += 1

        for ann in coco.get("annotations", []):
            if ann["image_id"] not in old_to_new:
                continue
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = old_to_new[ann["image_id"]]
            all_annotations.append(new_ann)
            next_ann_id += 1

        # Negatives — carry session tag but no annotations
        for neg_img in _load_negatives(entry):
            new_neg = dict(neg_img)
            new_neg["id"] = next_img_id
            new_neg["_session_id"] = entry["session_id"]
            all_negatives.append(new_neg)
            next_img_id += 1

    logger.info(
        "Loaded %d positive images, %d annotations, %d negative images "
        "from %d session(s).",
        len(all_images), len(all_annotations), len(all_negatives),
        len(atwood_sources),
    )

    # ---------------------------------------------------------------------------
    # Build annotation-to-image lookup
    # ---------------------------------------------------------------------------
    anns_by_image: dict[int, list[int]] = defaultdict(list)
    for ann in all_annotations:
        anns_by_image[ann["image_id"]].append(ann["id"])

    # ---------------------------------------------------------------------------
    # Build per-annotation cell lookup from feature CSV
    # ---------------------------------------------------------------------------
    # The feature CSV uses original annotation IDs before global re-numbering.
    # We need to match on (session_id, original_ann_id → new_ann_id).
    # Simpler: rebuild the cell lookup using global IDs by matching file_name.

    # file_name → cell (we assume file_name is unique across sessions — it is
    # for Atwood since each night is a separate NORAD-ID/time directory).
    fn_to_cell: dict[str, tuple[str, str]] = {}
    for row in feature_rows:
        fn = row["file_name"]
        cell = (row["band"], row["snr_class"])
        # Prefer the cell if the file appears only once; for duplicates take first.
        if fn not in fn_to_cell:
            fn_to_cell[fn] = cell

    # image_id (global, remapped) → cell
    img_id_to_cell: dict[int, tuple[str, str]] = {}

    for img in all_images:
        fn = img["file_name"]
        # Normalise: the feature CSV may have the bare filename; images may have
        # an absolute path after raw_dir resolution.  Match on Path.name.
        fn_base = Path(fn).name
        # Try exact match first, then basename match
        cell = fn_to_cell.get(fn) or fn_to_cell.get(fn_base)
        if cell is None:
            # Not in feature table → no annotation or feature extraction skipped
            continue
        img_id_to_cell[img["id"]] = cell

    # Remaining: images with annotations but no feature row (edge case)
    for img in all_images:
        if img["id"] in img_id_to_cell:
            continue
        aids = anns_by_image.get(img["id"], [])
        if aids:
            img_id_to_cell[img["id"]] = ("unknown", "null")

    # ---------------------------------------------------------------------------
    # Group images by stratification cell
    # ---------------------------------------------------------------------------
    cells: dict[tuple[str, str], list[int]] = defaultdict(list)
    for img_id, cell in img_id_to_cell.items():
        cells[cell].append(img_id)

    # ---------------------------------------------------------------------------
    # Split within each cell
    # ---------------------------------------------------------------------------
    train_ids: set[int] = set()
    val_ids: set[int] = set()
    test_ids: set[int] = set()

    split_summary: dict[str, Any] = {"cells": {}, "totals": {}}

    for band in _BAND_ORDER + ["unknown"]:
        for snr_cls in _SNR_ORDER:
            cell = (band, snr_cls)
            ids = cells.get(cell, [])
            if not ids:
                continue

            if len(ids) < 3:
                # Too small to split — all to training
                t, v, te = ids, [], []
                note = "all_to_train (n<3)"
            else:
                t, v, te = _stratified_split(ids, ratio, seed=args.seed)
                note = f"n={len(ids)}"

            train_ids.update(t)
            val_ids.update(v)
            test_ids.update(te)

            cell_key = f"{band}×{snr_cls}"
            split_summary["cells"][cell_key] = {
                "total": len(ids),
                "train": len(t),
                "val": len(v),
                "test": len(te),
                "note": note,
            }
            logger.info(
                "  %-22s  total=%-4d  train=%-4d  val=%-4d  test=%-4d  %s",
                cell_key, len(ids), len(t), len(v), len(te), note,
            )

    # Sanity: images with annotations but not yet assigned (e.g. "unknown×null")
    assigned = train_ids | val_ids | test_ids
    annotated_ids = {img["id"] for img in all_images if anns_by_image.get(img["id"])}
    unassigned = annotated_ids - assigned
    if unassigned:
        logger.warning(
            "%d annotated images not assigned to any cell — adding to training.",
            len(unassigned),
        )
        train_ids.update(unassigned)

    # ---------------------------------------------------------------------------
    # Split negatives randomly at the same ratio
    # ---------------------------------------------------------------------------
    neg_ids = [img["id"] for img in all_negatives]
    neg_train, neg_val, neg_test = _stratified_split(neg_ids, ratio, seed=args.seed + 1)
    n_neg_train = set(neg_train)
    n_neg_val = set(neg_val)
    n_neg_test = set(neg_test)

    logger.info(
        "Negatives: total=%d  train=%d  val=%d  test=%d",
        len(neg_ids), len(neg_train), len(neg_val), len(neg_test),
    )

    # ---------------------------------------------------------------------------
    # Final counts
    # ---------------------------------------------------------------------------
    logger.info(
        "Positive split totals — train=%d  val=%d  test=%d  (of %d annotated images)",
        len(train_ids), len(val_ids), len(test_ids), len(annotated_ids),
    )

    split_summary["totals"] = {
        "train": {"positive": len(train_ids), "negative": len(neg_train)},
        "val":   {"positive": len(val_ids),   "negative": len(neg_val)},
        "test":  {"positive": len(test_ids),  "negative": len(neg_test)},
    }

    if args.dry_run:
        logger.info("[dry-run] No files written.")
        import json as _json
        print(_json.dumps(split_summary, indent=2))
        return

    # ---------------------------------------------------------------------------
    # Build and write COCO JSONs
    # ---------------------------------------------------------------------------
    all_combined = all_images + all_negatives
    # Remove the internal _session_id tag before writing
    def _clean(imgs: list[dict]) -> list[dict]:
        return [{k: v for k, v in img.items() if k != "_session_id"} for img in imgs]

    splits = {
        "atwood_train": (train_ids | n_neg_train, "train"),
        "val_atwood":   (val_ids   | n_neg_val,   "val"),
        "test_atwood":  (test_ids  | n_neg_test,  "test"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for fname, (ids, split_name) in splits.items():
        coco = _build_coco(
            image_ids=ids,
            images=_clean(all_combined),
            annotations=all_annotations,
            split_name=split_name,
        )
        out_path = args.output_dir / f"{fname}.json"
        with open(out_path, "w") as fh:
            json.dump(coco, fh)
        logger.info(
            "Written: %s  (%d images, %d annotations)",
            out_path, len(coco["images"]), len(coco["annotations"]),
        )

    # Write split summary
    summary_dir = _REPO_ROOT / "data/features"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "atwood_split_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(split_summary, fh, indent=2)
    logger.info("Summary written: %s", summary_path)


if __name__ == "__main__":
    main()
