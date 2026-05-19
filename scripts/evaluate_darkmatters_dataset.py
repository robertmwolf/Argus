"""Evaluate the DarkMatters telescope image dataset for ARGUS integration.

Runs two integration probes:
  A) YOLO OBB inference on streak-positive images → pseudo-annotation feasibility
  B) Negative image inventory → COCO JSON augmentation candidate list

Outputs:
  <output-dir>/report.md        — human-readable summary
  <output-dir>/report.json      — machine-readable stats
  <output-dir>/yolo_probe.json  — per-image YOLO detection results
  <output-dir>/negatives.json   — COCO JSON skeleton for negative pool

Usage:
    python scripts/evaluate_darkmatters_dataset.py \
        --dataset-dir /Volumes/External/DarkMatters/exports \
        --weights weights/run_full_yolo_obb/run/weights/best.pt \
        --output-dir results/darkmatters_eval
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def win_preview_to_mac(win_path: str, dataset_dir: pathlib.Path) -> pathlib.Path | None:
    """Convert a Windows preview path to the local Mac absolute path.

    Windows paths look like:
      C:\\minerva_CDK20\\assess_ai\\exports\\set_3\\previews\\filename.jpg
    or:
      C:\\minerva\\assess_ai\\exports\\set_3\\previews\\filename.jpg

    We extract `set_X/previews/filename` and resolve under dataset_dir.
    """
    # Normalise separators
    norm = win_path.replace("\\", "/")
    # Match set_N/previews/filename
    m = re.search(r"(set_\d+/previews/[^/]+)$", norm)
    if not m:
        return None
    return dataset_dir / m.group(1)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_streak_csv(dataset_dir: pathlib.Path) -> list[dict]:
    """Load the most recent curated_streak CSV. Returns list of row dicts."""
    import csv

    # Pick highest version number
    candidates = sorted(dataset_dir.glob("curated_streak_v*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No curated_streak_v*.csv found in {dataset_dir}")
    csv_path = candidates[-1]
    log.info("Loading streak labels from %s", csv_path.name)

    rows: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    log.info("Loaded %d labeled rows", len(rows))
    return rows


def load_all_frames_metadata(dataset_dir: pathlib.Path) -> dict[str, dict]:
    """Load frames.csv from every set_* directory. Returns dict keyed by file_name."""
    import csv

    metadata: dict[str, dict] = {}
    for frames_csv in sorted(dataset_dir.glob("set_*/frames.csv")):
        with open(frames_csv, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                metadata[row["file_name"]] = row
    log.info("Loaded frames metadata for %d images", len(metadata))
    return metadata


# ---------------------------------------------------------------------------
# Image compatibility probe
# ---------------------------------------------------------------------------

def check_images(paths: list[pathlib.Path], sample_size: int = 20) -> dict:
    """Check a random sample of images for size, mode, and brightness stats."""
    from PIL import Image

    rng = np.random.default_rng(42)
    sample = [paths[i] for i in rng.choice(len(paths), min(sample_size, len(paths)), replace=False)]

    sizes: list[tuple[int, int]] = []
    modes: list[str] = []
    means: list[float] = []
    missing = 0

    for p in sample:
        if not p.exists():
            missing += 1
            continue
        try:
            img = Image.open(p)
            arr = np.array(img).astype(np.float32) / 255.0
            sizes.append(img.size)
            modes.append(img.mode)
            means.append(float(arr.mean()))
        except Exception as exc:
            log.warning("Cannot open %s: %s", p, exc)
            missing += 1

    return {
        "sampled": len(sample),
        "missing": missing,
        "readable": len(sample) - missing,
        "unique_sizes": list({str(s) for s in sizes}),
        "unique_modes": list(set(modes)),
        "mean_brightness_mean": round(statistics.mean(means), 4) if means else None,
        "mean_brightness_stdev": round(statistics.stdev(means), 4) if len(means) > 1 else None,
    }


# ---------------------------------------------------------------------------
# YOLO inference probe
# ---------------------------------------------------------------------------

def run_yolo_probe(
    positive_paths: list[pathlib.Path],
    weights: pathlib.Path,
    conf_threshold: float = 0.25,
    high_conf_threshold: float = 0.50,
) -> tuple[list[dict], dict]:
    """Run YOLO OBB inference on streak-positive images.

    Returns:
        per_image: list of per-image result dicts
        summary: aggregate stats dict
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics not installed — skipping YOLO probe")
        return [], {"error": "ultralytics not available"}

    if not weights.exists():
        log.error("Weights not found: %s", weights)
        return [], {"error": f"weights not found: {weights}"}

    log.info("Loading YOLO model from %s", weights)
    model = YOLO(str(weights))

    per_image: list[dict] = []
    n_detected = 0
    n_high_conf = 0
    all_confs: list[float] = []
    all_angles: list[float] = []

    existing = [p for p in positive_paths if p.exists()]
    missing_count = len(positive_paths) - len(existing)
    if missing_count:
        log.warning("%d positive images not found on disk (skipped)", missing_count)

    log.info("Running YOLO on %d positive images...", len(existing))
    t0 = time.time()

    for img_path in existing:
        try:
            results = model.predict(
                str(img_path),
                imgsz=640,
                conf=conf_threshold,
                verbose=False,
            )
            r = results[0]
            dets: list[dict] = []
            if r.obb is not None and len(r.obb) > 0:
                for box in r.obb:
                    conf = float(box.conf[0])
                    # xywhr format: cx, cy, w, h, angle_rad
                    xywhr = box.xywhr[0].cpu().tolist()
                    angle_deg = float(np.degrees(xywhr[4]))
                    dets.append({
                        "conf": round(conf, 4),
                        "cx": round(xywhr[0], 1),
                        "cy": round(xywhr[1], 1),
                        "w": round(xywhr[2], 1),
                        "h": round(xywhr[3], 1),
                        "angle_deg": round(angle_deg, 2),
                    })
                    all_confs.append(conf)
                    all_angles.append(angle_deg)
                n_detected += 1
                if any(d["conf"] >= high_conf_threshold for d in dets):
                    n_high_conf += 1

            per_image.append({
                "path": str(img_path),
                "detections": dets,
                "n_det": len(dets),
            })
        except Exception as exc:
            log.warning("YOLO failed on %s: %s", img_path.name, exc)
            per_image.append({"path": str(img_path), "detections": [], "n_det": 0, "error": str(exc)})

    elapsed = time.time() - t0
    detection_rate = n_detected / len(existing) if existing else 0.0
    high_conf_rate = n_high_conf / len(existing) if existing else 0.0

    summary = {
        "images_probed": len(existing),
        "images_missing": missing_count,
        "images_with_any_detection": n_detected,
        "images_with_high_conf_detection": n_high_conf,
        "detection_rate": round(detection_rate, 3),
        "high_conf_detection_rate": round(high_conf_rate, 3),
        "conf_threshold": conf_threshold,
        "high_conf_threshold": high_conf_threshold,
        "mean_confidence": round(statistics.mean(all_confs), 4) if all_confs else None,
        "mean_angle_deg": round(statistics.mean(all_angles), 2) if all_angles else None,
        "elapsed_s": round(elapsed, 1),
    }
    log.info(
        "YOLO probe complete: %.1f%% detection rate, %.1f%% high-conf",
        detection_rate * 100,
        high_conf_rate * 100,
    )
    return per_image, summary


# ---------------------------------------------------------------------------
# COCO JSON negative skeleton builder
# ---------------------------------------------------------------------------

def build_negative_coco(
    negative_rows: list[dict],
    dataset_dir: pathlib.Path,
) -> dict[str, Any]:
    """Build a COCO JSON skeleton (empty annotations) for negative images."""
    images: list[dict] = []
    now_str = datetime.now(timezone.utc).isoformat()

    for i, row in enumerate(negative_rows):
        mac_path = win_preview_to_mac(row["preview_path"], dataset_dir)
        if mac_path is None:
            continue
        images.append({
            "id": i + 1,
            "file_name": str(mac_path),
            "width": 3000,
            "height": 2001,
            "date_captured": now_str,
            "source": "DarkMatters_CDK20",
            "source_group": row.get("source_group", ""),
            "set_id": row.get("set_id", ""),
        })

    return {
        "info": {
            "description": "DarkMatters negative images for ARGUS training augmentation",
            "date_created": now_str,
            "version": "1.0",
        },
        "licenses": [],
        "categories": [{"id": 1, "name": "satellite_streak", "supercategory": "streak"}],
        "images": images,
        "annotations": [],
    }


# ---------------------------------------------------------------------------
# Metadata enrichment
# ---------------------------------------------------------------------------

def enrich_with_metadata(
    rows: list[dict],
    frames_meta: dict[str, dict],
) -> dict[str, Any]:
    """Compute aggregate metadata stats for streak-positive images."""
    fwhm_vals: list[float] = []
    ecc_vals: list[float] = []
    filters: dict[str, int] = {}
    objects: dict[str, int] = {}

    for row in rows:
        key = row.get("file_name", "")
        meta = frames_meta.get(key)
        if meta is None:
            continue
        try:
            fwhm_vals.append(float(meta["fwhm"]))
        except (ValueError, KeyError):
            pass
        try:
            ecc_vals.append(float(meta["eccentricity"]))
        except (ValueError, KeyError):
            pass
        fn = meta.get("filter_name", "unknown")
        filters[fn] = filters.get(fn, 0) + 1
        obj = meta.get("object_name", "unknown")
        objects[obj] = objects.get(obj, 0) + 1

    return {
        "fwhm_mean": round(statistics.mean(fwhm_vals), 3) if fwhm_vals else None,
        "fwhm_stdev": round(statistics.stdev(fwhm_vals), 3) if len(fwhm_vals) > 1 else None,
        "eccentricity_mean": round(statistics.mean(ecc_vals), 3) if ecc_vals else None,
        "filter_distribution": dict(sorted(filters.items(), key=lambda x: -x[1])),
        "top_objects": dict(sorted(objects.items(), key=lambda x: -x[1])[:10]),
        "metadata_matched": len(fwhm_vals),
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def render_report(
    label_counts: dict[str, int],
    image_check_pos: dict,
    image_check_neg: dict,
    yolo_summary: dict,
    neg_coco_count: int,
    meta_pos: dict,
    meta_neg: dict,
    weights_path: str,
) -> str:
    """Render a human-readable Markdown evaluation report."""
    total = sum(label_counts.values())
    positives = label_counts.get("positive", 0) + label_counts.get("hard_positive", 0)
    negatives = label_counts.get("good_negative", 0) + label_counts.get("hard_negative", 0)

    det_rate = yolo_summary.get("detection_rate", 0)
    high_conf_rate = yolo_summary.get("high_conf_detection_rate", 0)
    yolo_err = yolo_summary.get("error")

    # Estimate usable pseudo-annotations
    probed = yolo_summary.get("images_probed", 0)
    usable_est = yolo_summary.get("images_with_high_conf_detection", 0) if not yolo_err else "N/A"

    if not yolo_err and probed > 0:
        verdict = (
            "**VIABLE** — detection rate is high enough to generate usable pseudo-annotations."
            if det_rate >= 0.40
            else "**MARGINAL** — low detection rate; manual annotation recommended for positives."
            if det_rate >= 0.15
            else "**NOT VIABLE via YOLO** — detector fires on <15% of positive images; "
            "instrument/resolution mismatch is likely. Manual annotation or discard."
        )
    else:
        verdict = "**UNKNOWN** — YOLO probe did not run (check weights path or ultralytics install)."

    lines = [
        "# DarkMatters Dataset Evaluation Report",
        f"\n_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        f"_YOLO weights: `{weights_path}`_",
        "",
        "## 1. Label Distribution",
        "",
        f"| Label | Count |",
        f"|---|---|",
    ]
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {label} | {count} |")
    lines += [
        f"| **Total** | **{total}** |",
        "",
        f"- **Streak positives (positive + hard_positive):** {positives}",
        f"- **Negatives (good + hard):** {negatives}",
        "",
        "## 2. Image Compatibility",
        "",
        "### Positive images (streak present)",
        f"- Sampled: {image_check_pos['sampled']} / Missing: {image_check_pos['missing']}",
        f"- Unique sizes: {image_check_pos['unique_sizes']}",
        f"- Color modes: {image_check_pos['unique_modes']}",
        f"- Mean brightness: {image_check_pos['mean_brightness_mean']} "
        f"± {image_check_pos['mean_brightness_stdev']}",
        "",
        "### Negative images (no streak)",
        f"- Sampled: {image_check_neg['sampled']} / Missing: {image_check_neg['missing']}",
        f"- Unique sizes: {image_check_neg['unique_sizes']}",
        f"- Color modes: {image_check_neg['unique_modes']}",
        f"- Mean brightness: {image_check_neg['mean_brightness_mean']} "
        f"± {image_check_neg['mean_brightness_stdev']}",
        "",
        "## 3. YOLO OBB Probe (Option B feasibility)",
        "",
    ]

    if yolo_err:
        lines += [f"**Probe skipped:** {yolo_err}", ""]
    else:
        lines += [
            f"| Metric | Value |",
            f"|---|---|",
            f"| Images probed | {yolo_summary['images_probed']} |",
            f"| Images missing | {yolo_summary.get('images_missing', 0)} |",
            f"| With any detection (conf ≥ {yolo_summary['conf_threshold']}) "
            f"| {yolo_summary['images_with_any_detection']} ({det_rate:.1%}) |",
            f"| With high-conf detection (conf ≥ {yolo_summary['high_conf_threshold']}) "
            f"| {yolo_summary['images_with_high_conf_detection']} ({high_conf_rate:.1%}) |",
            f"| Mean detection confidence | {yolo_summary['mean_confidence']} |",
            f"| Mean streak angle | {yolo_summary['mean_angle_deg']}° |",
            f"| Inference time | {yolo_summary['elapsed_s']} s |",
            "",
            f"**Verdict:** {verdict}",
            f"**Estimated high-quality pseudo-annotations:** {usable_est}",
            "",
        ]

    lines += [
        "## 4. Negative Pool (Option A)",
        "",
        f"- COCO JSON negatives built: **{neg_coco_count}** images (zero annotations each)",
        f"- ARGUS already has: ~91 GTImages negatives",
        f"- Net gain: +{neg_coco_count} negatives ({neg_coco_count / 91:.1f}× current pool)",
        f"- Caveat: JPEG previews at 3000×2001 px vs FITS originals; "
        "instrument response differs.",
        "",
        "## 5. Instrument / Seeing Characterization",
        "",
        "### Positive images",
        f"- Metadata matched: {meta_pos.get('metadata_matched', 0)} / {positives}",
        f"- FWHM: {meta_pos.get('fwhm_mean')} ± {meta_pos.get('fwhm_stdev')} (arcsec/px proxy)",
        f"- Eccentricity: {meta_pos.get('eccentricity_mean')}",
        f"- Filter distribution: {meta_pos.get('filter_distribution')}",
        f"- Top objects: {list(meta_pos.get('top_objects', {}).keys())[:5]}",
        "",
        "### Negative images",
        f"- Metadata matched: {meta_neg.get('metadata_matched', 0)} / {negatives}",
        f"- FWHM: {meta_neg.get('fwhm_mean')} ± {meta_neg.get('fwhm_stdev')}",
        f"- Eccentricity: {meta_neg.get('eccentricity_mean')}",
        "",
        "## 6. Recommendation",
        "",
        "### Option A — Add negatives to training pool",
        f"**Proceed.** {neg_coco_count} JPEG negatives are ready as "
        "`results/darkmatters_eval/negatives.json`. Run "
        "`training/convert_labels.py` or merge directly into `data/annotations/` "
        "if the image dimensions match your tiling pipeline. "
        "Flag these with `source=darkmatters` in annotation attributes.",
        "",
        "### Option B — Pseudo-annotate positives with YOLO",
    ]

    if yolo_err:
        lines.append(
            f"Re-run with `--weights` pointing to a valid YOLO OBB checkpoint to get this result."
        )
    elif det_rate >= 0.40:
        lines.append(
            f"**Proceed.** {usable_est} images have high-confidence YOLO detections. "
            "Review `yolo_probe.json`, keep detections with conf ≥ 0.50, "
            "spot-check 20 random examples visually, then merge into `train.json`."
        )
    elif det_rate >= 0.15:
        lines.append(
            f"**Proceed with caution.** Only {det_rate:.0%} detection rate — "
            "the instrument FOV / PSF may differ enough that YOLO pseudo-labels "
            "are unreliable. Manually annotate a 30-image sample to calibrate before "
            "bulk-adding."
        )
    else:
        lines.append(
            f"**Do not use YOLO pseudo-annotations.** Detection rate ({det_rate:.0%}) is too low. "
            "Options: (1) manually annotate positives in LabelImg/CVAT (~3 hr), "
            "or (2) discard positives and use only the negatives (Option A)."
        )

    lines += [
        "",
        "---",
        "_Report produced by `scripts/evaluate_darkmatters_dataset.py`_",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default="/Volumes/External/DarkMatters/exports",
        type=pathlib.Path,
        help="Root of the DarkMatters exports directory",
    )
    parser.add_argument(
        "--weights",
        default="weights/run_full_yolo_obb/run/weights/best.pt",
        type=pathlib.Path,
        help="YOLO11n-OBB weights file (best.pt)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/darkmatters_eval",
        type=pathlib.Path,
        help="Directory for evaluation outputs",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold for any detection",
    )
    parser.add_argument(
        "--high-conf",
        type=float,
        default=0.50,
        help="YOLO confidence threshold for 'high quality' pseudo-annotation",
    )
    parser.add_argument(
        "--skip-yolo",
        action="store_true",
        help="Skip YOLO inference (for fast dry-run)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load labels
    rows = load_streak_csv(args.dataset_dir)
    label_counts: dict[str, int] = {}
    for row in rows:
        sg = row.get("source_group", "unknown")
        label_counts[sg] = label_counts.get(sg, 0) + 1

    positive_rows = [r for r in rows if r.get("source_group", "").startswith("positive")]
    negative_rows = [r for r in rows if "negative" in r.get("source_group", "")]

    log.info("Positives: %d  Negatives: %d", len(positive_rows), len(negative_rows))

    # 2. Resolve Mac paths
    positive_paths = [
        p for r in positive_rows
        if (p := win_preview_to_mac(r["preview_path"], args.dataset_dir)) is not None
    ]
    negative_paths = [
        p for r in negative_rows
        if (p := win_preview_to_mac(r["preview_path"], args.dataset_dir)) is not None
    ]

    # 3. Image compatibility check
    log.info("Checking image compatibility (positives)...")
    image_check_pos = check_images(positive_paths, sample_size=30)
    log.info("Checking image compatibility (negatives)...")
    image_check_neg = check_images(negative_paths, sample_size=30)

    # 4. YOLO probe
    if args.skip_yolo:
        log.info("Skipping YOLO probe (--skip-yolo)")
        yolo_per_image: list[dict] = []
        yolo_summary: dict = {"error": "skipped via --skip-yolo"}
    else:
        yolo_per_image, yolo_summary = run_yolo_probe(
            positive_paths,
            args.weights,
            conf_threshold=args.conf,
            high_conf_threshold=args.high_conf,
        )

    # 5. Build negative COCO JSON
    log.info("Building negative COCO JSON skeleton...")
    neg_coco = build_negative_coco(negative_rows, args.dataset_dir)
    neg_coco_count = len(neg_coco["images"])

    # 6. Metadata enrichment
    log.info("Loading frames metadata...")
    frames_meta = load_all_frames_metadata(args.dataset_dir)
    meta_pos = enrich_with_metadata(positive_rows, frames_meta)
    meta_neg = enrich_with_metadata(negative_rows, frames_meta)

    # 7. Assemble report dict
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(args.dataset_dir),
        "weights": str(args.weights),
        "label_counts": label_counts,
        "total_labeled": len(rows),
        "positive_count": len(positive_rows),
        "negative_count": len(negative_rows),
        "image_check_positives": image_check_pos,
        "image_check_negatives": image_check_neg,
        "yolo_probe": yolo_summary,
        "negative_coco_images": neg_coco_count,
        "metadata_positives": meta_pos,
        "metadata_negatives": meta_neg,
    }

    # 8. Write outputs
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    log.info("Wrote report.json")

    (args.output_dir / "yolo_probe.json").write_text(json.dumps(yolo_per_image, indent=2))
    log.info("Wrote yolo_probe.json (%d entries)", len(yolo_per_image))

    (args.output_dir / "negatives.json").write_text(json.dumps(neg_coco, indent=2))
    log.info("Wrote negatives.json (%d images)", neg_coco_count)

    md = render_report(
        label_counts=label_counts,
        image_check_pos=image_check_pos,
        image_check_neg=image_check_neg,
        yolo_summary=yolo_summary,
        neg_coco_count=neg_coco_count,
        meta_pos=meta_pos,
        meta_neg=meta_neg,
        weights_path=str(args.weights),
    )
    (args.output_dir / "report.md").write_text(md)
    log.info("Wrote report.md")

    print("\n" + "=" * 60)
    print(md)
    print("=" * 60)
    print(f"\nAll outputs in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
