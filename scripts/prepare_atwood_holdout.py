"""Prepare reviewed Atwood annotation JSONs for zero-shot evaluation.

The interactive annotator writes one ``brentimages_annotations.json`` per night
containing positives, confirmed blanks, rejected frames, and still-pending
frames.  This script exports only reviewed usable data:

* positives: images with at least one OBB annotation
* negatives: images marked ``blank``
* excluded: rejected or unreviewed images

Outputs are COCO JSONs suitable for ``scripts/zero_shot_eval.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_OUT_DIR = _REPO_ROOT / "data/annotations"
_EXTERNAL_OUT_DIR = Path("/Volumes/External/TrainingData/annotations")
_CATEGORIES = [{"id": 1, "name": "streak", "supercategory": "satellite"}]


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    logger.info("Wrote %s", path)


def _renumber_coco(images: list[dict[str, Any]], annotations: list[dict[str, Any]]) -> dict[str, Any]:
    old_to_new: dict[int, int] = {}
    out_images: list[dict[str, Any]] = []
    out_annotations: list[dict[str, Any]] = []

    for new_id, image in enumerate(images, start=1):
        old_id = int(image["id"])
        old_to_new[old_id] = new_id
        out_image = dict(image)
        out_image["id"] = new_id
        out_images.append(out_image)

    next_ann_id = 1
    for annotation in annotations:
        old_image_id = int(annotation["image_id"])
        if old_image_id not in old_to_new:
            continue
        out_annotation = dict(annotation)
        out_annotation["id"] = next_ann_id
        out_annotation["image_id"] = old_to_new[old_image_id]
        out_annotations.append(out_annotation)
        next_ann_id += 1

    return {
        "info": {"description": "ARGUS reviewed Atwood zero-shot holdout"},
        "licenses": [],
        "images": out_images,
        "annotations": out_annotations,
        "categories": _CATEGORIES,
    }


def prepare(input_path: Path, session_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    """Build positive and negative COCO dicts from one annotator JSON."""
    coco = _load_json(input_path)
    images = list(coco.get("images", []))
    annotations = list(coco.get("annotations", []))

    anns_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in annotations:
        anns_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    positive_images: list[dict[str, Any]] = []
    positive_annotations: list[dict[str, Any]] = []
    negative_images: list[dict[str, Any]] = []
    rejected = 0
    pending = 0

    for image in images:
        image_id = int(image["id"])
        image_annotations = anns_by_image.get(image_id, [])
        if image.get("rejected"):
            rejected += 1
            continue
        if image_annotations:
            out_image = dict(image)
            out_image["source"] = session_id
            positive_images.append(out_image)
            positive_annotations.extend(image_annotations)
            continue
        if image.get("blank"):
            out_image = dict(image)
            out_image["source"] = session_id
            negative_images.append(out_image)
            continue
        pending += 1

    positives = _renumber_coco(positive_images, positive_annotations)
    negatives = _renumber_coco(negative_images, [])
    summary = {
        "source_images": len(images),
        "positive_images": len(positive_images),
        "negative_images": len(negative_images),
        "annotations": len(positive_annotations),
        "rejected_images": rejected,
        "pending_images": pending,
        "norad_ids": len({
            image.get("norad_id")
            for image in positive_images + negative_images
            if image.get("norad_id") is not None
        }),
    }
    return positives, negatives, summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUT_DIR)
    parser.add_argument(
        "--mirror-external",
        action="store_true",
        help="Also write outputs under /Volumes/External/TrainingData/annotations.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    positives, negatives, summary = prepare(args.input, args.session_id)

    pos_name = f"{args.session_id}.json"
    neg_name = f"{args.session_id}_negatives.json"
    summary_name = f"{args.session_id}_summary.json"

    outputs = [args.output_dir]
    if args.mirror_external:
        outputs.append(_EXTERNAL_OUT_DIR)

    for out_dir in outputs:
        _write_json(out_dir / pos_name, positives)
        _write_json(out_dir / neg_name, negatives)
        _write_json(out_dir / summary_name, summary)

    logger.info(
        "%s: %d positives, %d annotations, %d negatives, %d rejected, %d pending",
        args.session_id,
        summary["positive_images"],
        summary["annotations"],
        summary["negative_images"],
        summary["rejected_images"],
        summary["pending_images"],
    )


if __name__ == "__main__":
    main()
