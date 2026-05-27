"""Merge DarkMatters OBB annotations into the existing GT+SatStreaks COCO splits.

Creates three new annotation files:
  data/annotations/dm_merged_train.json
  data/annotations/dm_merged_val.json
  data/annotations/dm_merged_test.json

Each file is the union of the corresponding existing split (GTImages + SatStreaks)
and a stratified portion of the 239 annotated DarkMatters images.

Unannotated DarkMatters images (44 total) are discarded — they are neither
confirmed positives nor confirmed negatives.

Stratification: images are grouped by ``set_id`` so that images from the same
DarkMatters observation set stay in the same split (no set-level leakage).
Sets that contain only 1 image are assigned to train.

Split ratio: 80/10/10 (train/val/test), fixed random seed 42.

Google-style docstrings and type hints throughout.

Usage::

    python scripts/merge_darkmatters_annotations.py

Writes files to ``data/annotations/`` relative to the project root (the parent
of the directory containing this script).
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DM_SOURCE = Path(
    "/Users/robert/Argus/results/darkmatters_eval/streak_annotations.json"
)
ANN_DIR = PROJECT_ROOT / "data" / "annotations"

EXISTING_SPLITS = {
    "train": ANN_DIR / "train.json",
    "val":   ANN_DIR / "val.json",
    "test":  ANN_DIR / "test.json",
}

OUTPUT_SPLITS = {
    "train": ANN_DIR / "dm_merged_train.json",
    "val":   ANN_DIR / "dm_merged_val.json",
    "test":  ANN_DIR / "dm_merged_test.json",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_coco(path: Path) -> dict[str, Any]:
    """Load a COCO-format JSON file.

    Args:
        path: Absolute path to the JSON file.

    Returns:
        Parsed COCO dictionary with keys ``info``, ``images``,
        ``annotations``, and ``categories``.
    """
    with path.open() as fh:
        return json.load(fh)


def filter_annotated(coco: dict[str, Any]) -> dict[str, Any]:
    """Return a new COCO dict containing only images that have at least one annotation.

    Images that were visited but left unannotated are excluded entirely —
    they are neither confirmed positives nor confirmed negatives.

    Args:
        coco: Full COCO annotation dictionary.

    Returns:
        Filtered COCO dictionary; ``images`` and ``annotations`` are new
        lists; other keys are shared references.
    """
    annotated_image_ids = {ann["image_id"] for ann in coco["annotations"]}
    filtered_images = [
        img for img in coco["images"] if img["id"] in annotated_image_ids
    ]
    return {**coco, "images": filtered_images}


def stratified_split_by_set(
    images: list[dict[str, Any]],
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split images into train/val/test with stratification by ``set_id``.

    Images from the same ``set_id`` stay together in one split so that no
    observation set leaks across splits.  Sets with only 1 image always go
    to train.

    Args:
        images: List of COCO image dicts, each with a ``set_id`` field.
        val_frac: Fraction of images to allocate to the validation split.
        test_frac: Fraction of images to allocate to the test split.
        seed: Random seed for reproducibility.

    Returns:
        A tuple ``(train_images, val_images, test_images)``.
    """
    rng = random.Random(seed)

    # Group by set_id
    by_set: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for img in images:
        by_set[img["set_id"]].append(img)

    # Sort set keys for determinism, then shuffle
    set_keys = sorted(by_set.keys())
    rng.shuffle(set_keys)

    total = len(images)
    target_val = math.ceil(total * val_frac)
    target_test = math.ceil(total * test_frac)

    val_images: list[dict[str, Any]] = []
    test_images: list[dict[str, Any]] = []
    train_images: list[dict[str, Any]] = []

    for key in set_keys:
        group = by_set[key]
        if len(group) == 1:
            # Singletons always go to train
            train_images.extend(group)
            continue

        if len(val_images) < target_val:
            val_images.extend(group)
        elif len(test_images) < target_test:
            test_images.extend(group)
        else:
            train_images.extend(group)

    return train_images, val_images, test_images


def extract_annotations_for_images(
    annotations: list[dict[str, Any]],
    image_ids: set[int],
) -> list[dict[str, Any]]:
    """Return annotations whose ``image_id`` is in ``image_ids``.

    Args:
        annotations: Full list of COCO annotation dicts.
        image_ids: Set of image IDs to retain.

    Returns:
        Filtered list of annotation dicts.
    """
    return [ann for ann in annotations if ann["image_id"] in image_ids]


def renumber_and_merge(
    existing: dict[str, Any],
    dm_images: list[dict[str, Any]],
    dm_annotations: list[dict[str, Any]],
    split_name: str,
) -> dict[str, Any]:
    """Merge DarkMatters images/annotations into an existing COCO split.

    Re-numbers all IDs so they are globally unique within the merged file:
    - Existing image/annotation IDs are kept as-is (1 … N).
    - DarkMatters IDs are offset so they start immediately after the existing
      maximum IDs.

    All annotation fields (including ``"obb"``) are preserved.

    Args:
        existing: Loaded COCO dict for the existing split.
        dm_images: DarkMatters image dicts for this split partition.
        dm_annotations: DarkMatters annotation dicts for this split partition.
        split_name: Human-readable split name used in the ``info`` description.

    Returns:
        Merged COCO dict suitable for writing to disk.
    """
    existing_images = existing["images"]
    existing_annotations = existing["annotations"]

    # Find max IDs in existing split
    max_img_id = max((img["id"] for img in existing_images), default=0)
    max_ann_id = max((ann["id"] for ann in existing_annotations), default=0)

    # Build ID remapping for DarkMatters images
    dm_img_id_map: dict[int, int] = {}
    new_dm_images: list[dict[str, Any]] = []
    for i, img in enumerate(dm_images):
        new_id = max_img_id + i + 1
        dm_img_id_map[img["id"]] = new_id
        new_dm_images.append({**img, "id": new_id})

    # Build ID remapping for DarkMatters annotations
    new_dm_annotations: list[dict[str, Any]] = []
    for j, ann in enumerate(dm_annotations):
        new_ann_id = max_ann_id + j + 1
        new_img_id = dm_img_id_map[ann["image_id"]]
        new_dm_annotations.append({**ann, "id": new_ann_id, "image_id": new_img_id})

    merged_images = existing_images + new_dm_images
    merged_annotations = existing_annotations + new_dm_annotations

    return {
        "info": {
            "description": (
                f"ARGUS {split_name} split — GTImages + SatStreaks + DarkMatters"
            ),
            "version": "2.0",
            "sources": [
                "GTImages (synthetic)",
                "SatStreaks (HST FITS)",
                "DarkMatters CDK20 (real JPEG, 239 annotated images, 304 OBBs)",
            ],
        },
        "licenses": [],
        "categories": existing.get("categories", [
            {"id": 1, "name": "streak", "supercategory": "satellite"}
        ]),
        "images": merged_images,
        "annotations": merged_annotations,
    }


def validate_coco(coco: dict[str, Any], fname: str) -> None:
    """Assert referential integrity of a COCO annotation file.

    Checks that:
    - Every annotation's ``image_id`` refers to a known image.
    - All annotation IDs are unique.

    Args:
        coco: Loaded COCO dictionary to validate.
        fname: File name used in assertion messages.

    Raises:
        AssertionError: If any integrity check fails.
    """
    img_ids = {img["id"] for img in coco["images"]}
    ann_img_ids = {ann["image_id"] for ann in coco["annotations"]}
    assert ann_img_ids <= img_ids, (
        f"{fname}: orphan annotations — image IDs in annotations not in images: "
        f"{ann_img_ids - img_ids}"
    )
    all_ann_ids = [ann["id"] for ann in coco["annotations"]]
    assert len(all_ann_ids) == len(set(all_ann_ids)), (
        f"{fname}: duplicate annotation IDs"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Produce dm_merged_{{train,val,test}}.json in data/annotations/.

    Steps:
    1. Load the DarkMatters source JSON; discard unannotated images.
    2. Stratified 80/10/10 split of DM annotated images by set_id.
    3. For each split: load existing JSON, renumber + merge, validate, write.
    """
    print("Loading DarkMatters source annotations …")
    dm_full = load_coco(DM_SOURCE)
    dm_filtered = filter_annotated(dm_full)
    dm_images = dm_filtered["images"]
    dm_annotations = dm_full["annotations"]

    print(
        f"  {len(dm_full['images'])} total DM images; "
        f"{len(dm_images)} annotated, "
        f"{len(dm_full['images']) - len(dm_images)} unannotated (discarded)"
    )
    print(f"  {len(dm_annotations)} OBB annotations retained")

    # --- DM split ---------------------------------------------------------
    print("\nSplitting DarkMatters images (80/10/10, stratify by set_id, seed=42) …")
    dm_train, dm_val, dm_test = stratified_split_by_set(
        dm_images, val_frac=0.10, test_frac=0.10, seed=42
    )
    print(
        f"  DM train={len(dm_train)}  val={len(dm_val)}  test={len(dm_test)}"
        f"  (total={len(dm_train)+len(dm_val)+len(dm_test)})"
    )

    # --- Merge and write --------------------------------------------------
    dm_splits = {"train": dm_train, "val": dm_val, "test": dm_test}

    for split_name, dm_split_images in dm_splits.items():
        print(f"\nMerging {split_name} split …")

        existing = load_coco(EXISTING_SPLITS[split_name])
        print(
            f"  Existing: {len(existing['images'])} images, "
            f"{len(existing['annotations'])} annotations"
        )

        dm_img_ids = {img["id"] for img in dm_split_images}
        dm_split_anns = extract_annotations_for_images(dm_annotations, dm_img_ids)

        merged = renumber_and_merge(existing, dm_split_images, dm_split_anns, split_name)

        print(
            f"  DarkMatters contrib: {len(dm_split_images)} images, "
            f"{len(dm_split_anns)} annotations"
        )
        print(
            f"  Merged total: {len(merged['images'])} images, "
            f"{len(merged['annotations'])} annotations"
        )

        validate_coco(merged, OUTPUT_SPLITS[split_name].name)
        print(f"  Validation: OK")

        out_path = OUTPUT_SPLITS[split_name]
        with out_path.open("w") as fh:
            json.dump(merged, fh, indent=2)
        print(f"  Written → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
