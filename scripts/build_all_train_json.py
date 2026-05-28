"""Build the canonical training annotation JSON.

Produces:
  data/annotations/all_train_nodm.json   — SatStreaks + Night1 + Night2 + Geo_20260520 + Frigate tiles

Night 2 (brentimages_20260515.json) has bare filenames that are resolved to
absolute paths under /Volumes/External/TrainingData/raw/BrentImages/Img_20260515_Atwood/.

Frigate tiles come from frigate_tiled_train.json (built by build_tiled_frigate_json.py).
The virtual paths encoded in that file (stem__tx<x>_ty<y>_ts400.png) are handled at
load time by training/transforms.py LoadFITSFromFile.

Note: DarkMatters data is excluded from this project and must not be added.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ANN_DIR = _REPO_ROOT / "data/annotations"
_BRENT_N2_DIR = "/Volumes/External/TrainingData/raw/BrentImages/Img_20260515_Atwood"
_BRENT_GEO_DIR = "/Volumes/External/TrainingData/raw/BrentImages/Geo_20260520_Atwood"
_CANONICAL_CATEGORIES = [{"id": 1, "name": "streak", "supercategory": "satellite"}]


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _fix_brent_paths(images: list[dict], base_dir: str) -> list[dict]:
    """Resolve bare filenames to absolute paths under base_dir."""
    fixed = []
    for img in images:
        new = dict(img)
        fname = img["file_name"]
        if not fname.startswith("/"):
            new["file_name"] = f"{base_dir}/{fname}"
        fixed.append(new)
    return fixed


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
    # Load base split (SatStreaks + BrentImages Night 1)
    base_nodm = _load(_ANN_DIR / "train.json")          # 3,023 images

    # Load Night 2 — fix bare filenames to absolute paths
    n2_streaks = _load(_ANN_DIR / "brentimages_20260515.json")
    n2_neg = _load(_ANN_DIR / "brentimages_20260515_negatives.json")

    n2_streaks["images"] = _fix_brent_paths(n2_streaks["images"], _BRENT_N2_DIR)
    n2_neg["images"] = _fix_brent_paths(n2_neg["images"], _BRENT_N2_DIR)

    n2_images_count = len(n2_streaks["images"]) + len(n2_neg["images"])
    n2_ann_count = len(n2_streaks.get("annotations", []))
    logger.info(
        "Night 2: %d images (%d annotated, %d negatives), %d annotations",
        n2_images_count,
        len(n2_streaks["images"]),
        len(n2_neg["images"]),
        n2_ann_count,
    )

    # Load Geo_20260520 — geostationary streak captures
    geo_path = _ANN_DIR / "geo_20260520.json"
    if geo_path.exists():
        geo = _load(geo_path)
        geo["images"] = _fix_brent_paths(geo["images"], _BRENT_GEO_DIR)
        logger.info(
            "Geo_20260520: %d images, %d annotations",
            len(geo["images"]),
            len(geo.get("annotations", [])),
        )
    else:
        geo = None
        logger.warning("geo_20260520.json not found — skipping Geo_20260520_Atwood")

    # Load Frigate tiled crops (built by build_tiled_frigate_json.py)
    frigate_tiled_path = _ANN_DIR / "frigate_tiled_train.json"
    if not frigate_tiled_path.exists():
        raise FileNotFoundError(
            f"{frigate_tiled_path} not found — run scripts/build_tiled_frigate_json.py first"
        )
    frigate_tiled = _load(frigate_tiled_path)
    logger.info(
        "Frigate tiled: %d images, %d annotations",
        len(frigate_tiled["images"]),
        len(frigate_tiled.get("annotations", [])),
    )

    # Canonical training set (no DarkMatters)
    sources = [base_nodm, n2_streaks, n2_neg, frigate_tiled]
    if geo is not None:
        sources.append(geo)
    merged_nodm = merge(sources)
    out_nodm = _ANN_DIR / "all_train_nodm.json"
    with open(out_nodm, "w") as f:
        json.dump(merged_nodm, f)
    logger.info(
        "all_train_nodm.json: %d images, %d annotations",
        len(merged_nodm["images"]),
        len(merged_nodm["annotations"]),
    )


if __name__ == "__main__":
    main()
