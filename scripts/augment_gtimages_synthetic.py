"""Build GTImages real and synthetic training splits for StreakMind reproduction.

Outputs real-only train/val/test COCO files plus two synthetic training tracks:

  - gtimages_train_real.json
  - gtimages_train_synth_paper_long.json
  - gtimages_train_synth_adapted.json
  - gtimages_val.json
  - gtimages_test.json

Synthetic images are generated only from the training split. Validation and
test remain real GTImages frames.

# Source: StreakMind — hybrid real/synthetic streak training methodology
# Ref: https://arxiv.org/abs/2605.03429
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inference.fits_loader import FITSLoader
from training.augmentations import SyntheticStreakGeometry, SyntheticStreakInject

logger = logging.getLogger(__name__)

_CATEGORIES = [{"id": 1, "name": "satellite_streak", "supercategory": "streak"}]


def _read_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _prefixed_file_name(file_name: str, prefix: str = "GTImages") -> str:
    if "/" in file_name or Path(file_name).is_absolute():
        return file_name
    return str(Path(prefix) / file_name)


def _obb_dict(obb: dict[str, Any] | list[float]) -> dict[str, float]:
    if isinstance(obb, dict):
        return {
            "cx": float(obb["cx"]),
            "cy": float(obb["cy"]),
            "w": float(obb["w"]),
            "h": float(obb["h"]),
            "angle_deg": float(obb["angle_deg"]),
        }
    return {
        "cx": float(obb[0]),
        "cy": float(obb[1]),
        "w": float(obb[2]),
        "h": float(obb[3]),
        "angle_deg": float(obb[4]),
    }


def _load_gtimages_records(
    labeled_json: Path,
    negatives_json: Path,
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    labeled = _read_json(labeled_json)
    negatives = _read_json(negatives_json) if negatives_json.exists() else {"images": []}

    records: list[dict[str, Any]] = []
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)

    source_images: dict[int, dict[str, Any]] = {}
    for img in labeled.get("images", []):
        source_images[int(img["id"])] = {
            **img,
            "file_name": _prefixed_file_name(str(img["file_name"])),
            "source_kind": "labeled",
        }
    for ann in labeled.get("annotations", []):
        anns_by_image[int(ann["image_id"])].append(ann)

    neg_offset = 1_000_000
    for img in negatives.get("images", []):
        img_id = neg_offset + int(img["id"])
        source_images[img_id] = {
            **img,
            "id": img_id,
            "file_name": _prefixed_file_name(str(img["file_name"])),
            "source_kind": "negative",
        }

    lengths = [
        max(float(_obb_dict(ann["obb"])["w"]), float(_obb_dict(ann["obb"])["h"]))
        for ann_list in anns_by_image.values()
        for ann in ann_list
        if ann.get("obb")
    ]
    long_threshold = float(np.percentile(lengths, 75)) if lengths else 400.0

    for img_id, img in source_images.items():
        ann_list = anns_by_image.get(img_id, [])
        max_len = max(
            (max(_obb_dict(ann["obb"])["w"], _obb_dict(ann["obb"])["h"]) for ann in ann_list),
            default=0.0,
        )
        if not ann_list:
            stratum = "no_streak"
        elif max_len >= long_threshold:
            stratum = "long_streak"
        else:
            stratum = "short_streak"
        records.append({**img, "stratum": stratum, "max_streak_length_px": max_len})

    return records, anns_by_image


def _stratified_split(
    records: list[dict[str, Any]],
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_stratum[str(rec["stratum"])].append(rec)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []

    for group in by_stratum.values():
        rng.shuffle(group)
        n_total = len(group)
        n_train = int(round(n_total * train_fraction))
        n_val = int(round(n_total * val_fraction))
        n_train = min(n_train, n_total)
        n_val = min(n_val, n_total - n_train)
        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _copy_real_annotation(ann: dict[str, Any], new_id: int, new_image_id: int) -> dict[str, Any]:
    attrs = dict(ann.get("attributes", {}))
    attrs.update({
        "source_dataset": "gtimages",
        "synthetic": False,
        "synthetic_track": None,
    })
    return {
        **ann,
        "id": new_id,
        "image_id": new_image_id,
        "obb": _obb_dict(ann["obb"]) if ann.get("obb") else ann.get("obb"),
        "attributes": attrs,
    }


def _build_real_coco(
    records: list[dict[str, Any]],
    anns_by_source_image: dict[int, list[dict[str, Any]]],
    description: str,
) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    ann_id = 1
    for new_image_id, rec in enumerate(records, start=1):
        source_id = int(rec["id"])
        images.append({
            "id": new_image_id,
            "file_name": rec["file_name"],
            "width": int(rec["width"]),
            "height": int(rec["height"]),
            "date_captured": rec.get("date_captured"),
            "attributes": {
                "source_dataset": "gtimages",
                "synthetic": False,
                "stratum": rec["stratum"],
                "source_image_id": source_id,
            },
        })
        for ann in anns_by_source_image.get(source_id, []):
            annotations.append(_copy_real_annotation(ann, ann_id, new_image_id))
            ann_id += 1
    return {
        "info": {"description": description, "version": "1.0"},
        "licenses": [],
        "categories": _CATEGORIES,
        "images": images,
        "annotations": annotations,
    }


def _observed_angles_and_lengths(anns_by_image: dict[int, list[dict[str, Any]]]) -> tuple[list[float], list[float]]:
    angles: list[float] = []
    lengths: list[float] = []
    for ann_list in anns_by_image.values():
        for ann in ann_list:
            if not ann.get("obb"):
                continue
            obb = _obb_dict(ann["obb"])
            angles.append(float(obb["angle_deg"]) % 180.0)
            lengths.append(max(float(obb["w"]), float(obb["h"])))
    return angles, lengths


def _load_base_image(data_root: Path, file_name: str, loader: FITSLoader) -> np.ndarray:
    path = data_root / file_name
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"Could not read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return loader.load(path)["array"]


def _clip_bbox(bbox: list[float], width: int, height: int) -> list[float]:
    x, y, w, h = bbox
    x1 = max(0.0, min(float(width), x))
    y1 = max(0.0, min(float(height), y))
    x2 = max(0.0, min(float(width), x + w))
    y2 = max(0.0, min(float(height), y + h))
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def _synthetic_annotation(
    geometry: SyntheticStreakGeometry,
    ann_id: int,
    image_id: int,
    image_width: int,
    image_height: int,
    track: str,
    parent_image: str,
) -> dict[str, Any]:
    obb = geometry.obb
    attrs = {
        "source_dataset": "gtimages",
        "synthetic": True,
        "synthetic_track": track,
        "parent_image": parent_image,
        "length_px": round(geometry.length_px, 3),
        "angle_deg": round(geometry.angle_deg, 3),
        "brightness_level": geometry.brightness_level,
    }
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": 1,
        "iscrowd": 0,
        "segmentation": [geometry.segmentation()],
        "bbox": _clip_bbox(geometry.bbox(), image_width, image_height),
        "area": float(obb["w"] * obb["h"]),
        "obb": obb,
        "attributes": attrs,
    }


def _make_synthetic_track(
    real_train_coco: dict[str, Any],
    data_root: Path,
    synthetic_dir: Path,
    track: str,
    observed_angles: list[float],
    observed_lengths: list[float],
    seed: int,
    synthetic_ratio: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    loader = FITSLoader()
    injector = SyntheticStreakInject(p=1.0)

    base_images = list(real_train_coco["images"])
    n_synthetic = int(round(len(base_images) * synthetic_ratio))
    if n_synthetic <= 0:
        return real_train_coco
    if n_synthetic > len(base_images):
        choices = list(rng.choice(base_images, size=n_synthetic, replace=True))
    else:
        choices = list(rng.choice(base_images, size=n_synthetic, replace=False))

    p75 = float(np.percentile(observed_lengths, 75)) if observed_lengths else 400.0
    images = list(real_train_coco["images"])
    annotations = list(real_train_coco["annotations"])
    next_img_id = max((int(img["id"]) for img in images), default=0) + 1
    next_ann_id = max((int(ann["id"]) for ann in annotations), default=0) + 1

    synthetic_dir.mkdir(parents=True, exist_ok=True)
    try:
        synthetic_rel_root = synthetic_dir.resolve().relative_to(data_root.resolve())
    except ValueError:
        synthetic_rel_root = Path(synthetic_dir.name)
    for idx, parent in enumerate(choices):
        parent_file = str(parent["file_name"])
        base = _load_base_image(data_root, parent_file, loader)
        h, w = base.shape[:2]

        if track == "paper_long":
            min_len = p75
            max_len = float(np.hypot(h, w) * 1.25)
            angles = observed_angles
        elif track == "gtimages_short_medium":
            min_len = 50.0
            max_len = max(150.0, min(p75, 650.0))
            if observed_angles:
                random_angles = [float(v) for v in rng.uniform(0.0, 180.0, size=len(observed_angles))]
                angles = observed_angles + random_angles
            else:
                angles = []
        else:
            raise ValueError(f"Unknown synthetic track: {track}")

        n_streaks = 2 if rng.random() < 0.10 else 1
        level = int(rng.integers(1, 6))
        synthetic_img, geometries = injector.inject_with_geometry(
            base,
            rng=rng,
            n_streaks=n_streaks,
            min_length_px=min_len,
            max_length_px=max_len,
            angle_choices_deg=angles,
            brightness_level=level,
            width_px=16.0,
            full_crossing_probability=0.35 if track == "paper_long" else 0.10,
        )
        if not geometries:
            logger.warning("No synthetic geometry generated for %s; skipping", parent_file)
            continue

        out_rel = synthetic_rel_root / track / f"{Path(parent_file).stem}_synth_{idx:05d}.png"
        out_path = data_root / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), cv2.cvtColor(synthetic_img, cv2.COLOR_RGB2BGR))

        images.append({
            "id": next_img_id,
            "file_name": str(out_rel),
            "width": w,
            "height": h,
            "attributes": {
                "source_dataset": "gtimages",
                "synthetic": True,
                "synthetic_track": track,
                "parent_image": parent_file,
            },
        })
        for geometry in geometries:
            annotations.append(
                _synthetic_annotation(
                    geometry, next_ann_id, next_img_id, w, h, track, parent_file
                )
            )
            next_ann_id += 1
        next_img_id += 1

    return {
        **real_train_coco,
        "info": {
            **real_train_coco.get("info", {}),
            "description": f"GTImages training split + synthetic track {track}",
            "synthetic_track": track,
            "synthetic_ratio": synthetic_ratio,
            "gtimages_length_p75_px": p75,
        },
        "images": images,
        "annotations": annotations,
    }


def _write_coco(path: Path, coco: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coco, indent=2))
    logger.info("Wrote %s (%d images, %d annotations)", path, len(coco["images"]), len(coco["annotations"]))


def _count_strata(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for rec in records:
        counts[str(rec["stratum"])] += 1
    return dict(sorted(counts.items()))


def build_gtimages_synthetic_dataset(
    data_root: Path = Path("data"),
    labeled_json: Path = Path("data/annotations/gtimages.json"),
    negatives_json: Path = Path("data/annotations/gtimages_negatives.json"),
    output_dir: Path = Path("data/annotations"),
    synthetic_dir: Path = Path("data/gtimages_synthetic"),
    train_fraction: float = 0.70,
    val_fraction: float = 0.20,
    seed: int = 42,
    synthetic_ratio: float = 1.0,
    results_dir: Path = Path("results/streakmind_gtimages"),
) -> dict[str, Any]:
    """Create real and synthetic GTImages COCO datasets.

    Args:
        data_root: Root data directory.
        labeled_json: GTImages labeled COCO JSON.
        negatives_json: GTImages negative COCO JSON.
        output_dir: Directory for generated COCO annotation files.
        synthetic_dir: Directory for generated PNG images.
        train_fraction: Fraction of each stratum assigned to training.
        val_fraction: Fraction of each stratum assigned to validation.
        seed: Random seed for split and synthetic generation.
        synthetic_ratio: Synthetic images per real training image.

    Returns:
        Manifest dictionary also written to ``results/streakmind_gtimages``.
    """
    old_norm = os.environ.get("ARGUS_NORM")
    os.environ["ARGUS_NORM"] = "zscale"
    try:
        records, anns_by_image = _load_gtimages_records(labeled_json, negatives_json)
        train_records, val_records, test_records = _stratified_split(
            records, train_fraction, val_fraction, seed
        )

        real_train = _build_real_coco(
            train_records, anns_by_image, "GTImages real-only training split"
        )
        val_coco = _build_real_coco(val_records, anns_by_image, "GTImages real-only validation split")
        test_coco = _build_real_coco(test_records, anns_by_image, "GTImages real-only test split")

        observed_angles, observed_lengths = _observed_angles_and_lengths(anns_by_image)
        paper = _make_synthetic_track(
            real_train,
            data_root,
            synthetic_dir,
            "paper_long",
            observed_angles,
            observed_lengths,
            seed + 1001,
            synthetic_ratio,
        )
        adapted = _make_synthetic_track(
            real_train,
            data_root,
            synthetic_dir,
            "gtimages_short_medium",
            observed_angles,
            observed_lengths,
            seed + 2002,
            synthetic_ratio,
        )

        _write_coco(output_dir / "gtimages_train_real.json", real_train)
        _write_coco(output_dir / "gtimages_train_synth_paper_long.json", paper)
        _write_coco(output_dir / "gtimages_train_synth_adapted.json", adapted)
        _write_coco(output_dir / "gtimages_val.json", val_coco)
        _write_coco(output_dir / "gtimages_test.json", test_coco)

        p75 = float(np.percentile(observed_lengths, 75)) if observed_lengths else 400.0
        manifest = {
            "seed": seed,
            "normalization": "zscale",
            "train_fraction": train_fraction,
            "val_fraction": val_fraction,
            "test_fraction": round(1.0 - train_fraction - val_fraction, 4),
            "synthetic_ratio": synthetic_ratio,
            "gtimages_length_p75_px": round(p75, 3),
            "counts": {
                "all": _count_strata(records),
                "train": _count_strata(train_records),
                "val": _count_strata(val_records),
                "test": _count_strata(test_records),
                "train_real_images": len(real_train["images"]),
                "train_paper_long_images": len(paper["images"]),
                "train_adapted_images": len(adapted["images"]),
            },
            "outputs": {
                "train_real": str(output_dir / "gtimages_train_real.json"),
                "train_synth_paper_long": str(output_dir / "gtimages_train_synth_paper_long.json"),
                "train_synth_adapted": str(output_dir / "gtimages_train_synth_adapted.json"),
                "val": str(output_dir / "gtimages_val.json"),
                "test": str(output_dir / "gtimages_test.json"),
                "synthetic_dir": str(synthetic_dir),
            },
        }

        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
        (results_dir / "summary.md").write_text(_summary_markdown(manifest))
        logger.info("Wrote %s and %s", results_dir / "dataset_manifest.json", results_dir / "summary.md")
        return manifest
    finally:
        if old_norm is None:
            os.environ.pop("ARGUS_NORM", None)
        else:
            os.environ["ARGUS_NORM"] = old_norm


def _summary_markdown(manifest: dict[str, Any]) -> str:
    counts = manifest["counts"]
    return "\n".join([
        "# GTImages Synthetic Dataset Manifest",
        "",
        f"- Seed: `{manifest['seed']}`",
        f"- FITS normalization: `{manifest['normalization']}`",
        f"- Synthetic ratio: `{manifest['synthetic_ratio']}`",
        f"- GTImages P75 streak length: `{manifest['gtimages_length_p75_px']}` px",
        f"- All strata: `{counts['all']}`",
        f"- Train strata: `{counts['train']}`",
        f"- Val strata: `{counts['val']}`",
        f"- Test strata: `{counts['test']}`",
        "",
        "## Outputs",
        "",
        *[f"- `{key}`: `{value}`" for key, value in manifest["outputs"].items()],
        "",
    ])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--labeled-json", type=Path, default=Path("data/annotations/gtimages.json"))
    parser.add_argument("--negatives-json", type=Path, default=Path("data/annotations/gtimages_negatives.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/annotations"))
    parser.add_argument("--synthetic-dir", type=Path, default=Path("data/gtimages_synthetic"))
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-ratio", type=float, default=1.0)
    parser.add_argument("--results-dir", type=Path, default=Path("results/streakmind_gtimages"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )
    build_gtimages_synthetic_dataset(
        data_root=args.data_root,
        labeled_json=args.labeled_json,
        negatives_json=args.negatives_json,
        output_dir=args.output_dir,
        synthetic_dir=args.synthetic_dir,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
        synthetic_ratio=args.synthetic_ratio,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
