"""Build the canonical external-drive training annotation JSON.

Produces ``all_train_external_abs.json`` under ``ARGUS_ANNOTATIONS_DIR``
(default: ``/Volumes/External/TrainingData/annotations``).  The resulting COCO
file is assembled from external annotation components and uses external-drive
image paths, so training does not depend on repo-local annotation or image
copies. Set ``ARGUS_ALL_TRAIN_OUT`` to override the output path.

Note: """

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_ANN_DIR = Path(
    os.environ.get("ARGUS_ANNOTATIONS_DIR", "/Volumes/External/TrainingData/annotations")
)
_RAW_DIR = Path(os.environ.get("ARGUS_RAW_DATA_DIR", "/Volumes/External/TrainingData/raw"))
_CANONICAL_CATEGORIES = [{"id": 1, "name": "streak", "supercategory": "satellite"}]


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _external_image_path(file_name: str) -> str:
    """Resolve known training image paths to the external raw-data tree."""
    path = Path(file_name)
    if path.is_absolute():
        return str(path)

    parts = path.parts
    if parts and parts[0] == "satstreaks":
        return str(_RAW_DIR / path)
    if parts and parts[0] == "BrentImages":
        return str(_RAW_DIR / path)
    if parts and parts[0] == "frigate":
        return str(_RAW_DIR / path)
    return str(path)


def _normalise_image_paths(coco: dict) -> dict:
    """Return a shallow COCO copy whose image paths are external-drive paths."""
    fixed = {**coco, "images": []}
    for img in coco.get("images", []):
        new = dict(img)
        new["file_name"] = _external_image_path(new["file_name"])
        fixed["images"].append(new)
    return fixed


def _filter_images(coco: dict, predicate) -> dict:
    """Filter COCO images and annotations while preserving original IDs."""
    images = [img for img in coco.get("images", []) if predicate(img)]
    image_ids = {img["id"] for img in images}
    return {
        **coco,
        "images": images,
        "annotations": [
            ann for ann in coco.get("annotations", []) if ann["image_id"] in image_ids
        ],
    }


def merge(sources: list[dict]) -> dict:
    """Merge multiple COCO annotation dicts, reassigning all IDs sequentially."""
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
                continue  # orphaned annotation — skip
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = old_to_new[ann["image_id"]]
            all_annotations.append(new_ann)
            next_ann_id += 1

    return {
        "info": {"description": "ARGUS merged training split", "version": "2.0"},
        "licenses": [],
        "categories": _CANONICAL_CATEGORIES,
        "images": all_images,
        "annotations": all_annotations,
    }


def main() -> None:
    # Base train.json contains SatStreaks plus full-frame Night 1.  For the
    # canonical adaptive-tiling set, keep only SatStreaks and use tiled
    # BrentImages components below.
    base = _normalise_image_paths(_load(_ANN_DIR / "train.json"))
    satstreaks = _filter_images(
        base,
        lambda img: "/satstreaks/" in img["file_name"],
    )
    logger.info(
        "SatStreaks train: %d images, %d annotations",
        len(satstreaks["images"]),
        len(satstreaks.get("annotations", [])),
    )

    brent_n1_tiled = _normalise_image_paths(
        _load(_ANN_DIR / "brentimages_night1_tiled_train.json")
    )
    brent_n2_tiled = _normalise_image_paths(
        _load(_ANN_DIR / "brentimages_night2_tiled_train.json")
    )
    frigate_tiled = _normalise_image_paths(_load(_ANN_DIR / "frigate_tiled_train_ts110.json"))
    logger.info(
        "Brent N1 tiled: %d images, %d annotations",
        len(brent_n1_tiled["images"]),
        len(brent_n1_tiled.get("annotations", [])),
    )
    logger.info(
        "Brent N2 tiled: %d images, %d annotations",
        len(brent_n2_tiled["images"]),
        len(brent_n2_tiled.get("annotations", [])),
    )
    logger.info(
        "Frigate ts110 tiled: %d images, %d annotations",
        len(frigate_tiled["images"]),
        len(frigate_tiled.get("annotations", [])),
    )

    # Canonical training set (no     sources = [satstreaks, brent_n1_tiled, brent_n2_tiled, frigate_tiled]
    merged = merge(sources)
    out = Path(
        os.environ.get("ARGUS_ALL_TRAIN_OUT", _ANN_DIR / "all_train_external_abs.json")
    )
    with open(out, "w") as f:
        json.dump(merged, f)
    logger.info(
        "%s: %d images, %d annotations",
        out,
        len(merged["images"]),
        len(merged["annotations"]),
    )


if __name__ == "__main__":
    main()
