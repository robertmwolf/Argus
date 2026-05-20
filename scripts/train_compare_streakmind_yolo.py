"""Train and compare StreakMindYOLO tracks on GTImages (+ Frigate).

This driver consumes the annotation files produced by
``scripts/augment_gtimages_synthetic.py`` (GTImages tracks) and
``scripts/merge_fits_annotations.py`` (combined track) and trains/evaluates
four YOLO11-OBB tracks against the same real GTImages validation/test splits:

  - real_only          — GTImages real annotations only
  - paper_long         — GTImages real + synthetic long streaks (StreakMind style)
  - adapted            — GTImages real + synthetic adapted to local streak distribution
  - gtimages_plus_frigate — GTImages real + auto-annotated Frigate FITS frames

All tracks are evaluated at both IoU=0.5 and IoU=0.8 (StreakMind's threshold).

Outputs are written under ``results/streakmind_yolo/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.metrics import evaluate
from inference.fits_loader import FITSLoader
from training.train_baseline import _clip_obb_to_tile, _obb_fields, _obb_label_line

logger = logging.getLogger(__name__)

TRACKS = {
    "real_only": {
        "ann": Path("data/annotations/gtimages_train_real.json"),
        "work_dir": Path("weights/streakmind_yolo_real"),
        "method": "streakmind_yolo",
    },
    "paper_long": {
        "ann": Path("data/annotations/gtimages_train_synth_paper_long.json"),
        "work_dir": Path("weights/streakmind_yolo_paper_long"),
        "method": "streakmind_yolo",
    },
    "adapted": {
        "ann": Path("data/annotations/gtimages_train_synth_adapted.json"),
        "work_dir": Path("weights/streakmind_yolo_adapted"),
        "method": "streakmind_yolo",
    },
    "gtimages_plus_frigate": {
        "ann": Path("data/annotations/gtimages_plus_frigate_train.json"),
        "work_dir": Path("weights/streakmind_yolo_combined"),
        "method": "streakmind_yolo",
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _load_image(path: Path, loader: FITSLoader) -> np.ndarray:
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"Could not read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return loader.load(path)["array"]


def _anns_by_image(coco: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    for ann in coco.get("annotations", []):
        result.setdefault(int(ann["image_id"]), []).append(ann)
    return result


def _write_tile(
    arr: np.ndarray,
    out_img: Path,
    out_label: Path,
    labels: list[str],
) -> None:
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_label.parent.mkdir(parents=True, exist_ok=True)
    if not out_img.exists():
        cv2.imwrite(str(out_img), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    out_label.write_text("\n".join(labels) + ("\n" if labels else ""))


def _materialize_split(
    coco: dict[str, Any],
    data_root: Path,
    out_dir: Path,
    split: str,
    tile_size: int,
    max_bg_tiles: int,
    seed: int,
    centered: bool = True,
) -> int:
    """Materialize one YOLO-OBB split.

    When ``centered=True`` (default), annotated frames use streak-centered
    crops instead of blind uniform tiling.  Each annotation gets its own
    tile_size × tile_size crop centered on the streak, so the model always
    sees the complete streak rather than a clipped fragment.  This is the key
    fix for the angle-prediction failure observed with 640px tiles and long
    streaks (median 636px in GTImages test set).

    Background-only frames (e.g. Frigate negatives) always use random sampling
    regardless of the ``centered`` flag.
    """
    loader = FITSLoader()
    anns_by_img = _anns_by_image(coco)
    rng = np.random.default_rng(seed)
    n_tiles = 0

    for img_info in coco.get("images", []):
        img_id = int(img_info["id"])
        fname = str(img_info["file_name"])
        src = data_root / fname
        try:
            arr = _load_image(src, loader)
        except Exception as exc:
            logger.warning("Skipping unreadable image %s: %s", src, exc)
            continue

        h_arr, w_arr = arr.shape[:2]
        anns = anns_by_img.get(img_id, [])
        stem = Path(fname).stem

        if max(h_arr, w_arr) <= tile_size:
            labels = []
            for ann in anns:
                cx, cy, w, h, angle_deg = _obb_fields(ann["obb"])
                labels.append(_obb_label_line(cx, cy, w, h, angle_deg, norm=float(max(w_arr, h_arr))))
            _write_tile(
                arr,
                out_dir / "images" / split / f"{stem}.png",
                out_dir / "labels" / split / f"{stem}.txt",
                labels,
            )
            n_tiles += 1
            continue

        if centered and anns:
            # --- Streak-centered crops (Fix 2) ---
            # One tile per annotation, centered on the streak so the model
            # always trains on the complete streak and can learn the true angle.
            for i, ann in enumerate(anns):
                cx, cy, _w, _h, _a = _obb_fields(ann["obb"])
                x0 = int(max(0, min(cx - tile_size / 2, w_arr - tile_size)))
                y0 = int(max(0, min(cy - tile_size / 2, h_arr - tile_size)))
                patch = arr[y0:y0 + tile_size, x0:x0 + tile_size]
                # Include all annotations that fall inside this crop
                labels = []
                for ann2 in anns:
                    cx2, cy2, w2, h2, ang2 = _obb_fields(ann2["obb"])
                    result = _clip_obb_to_tile(cx2, cy2, w2, h2, ang2, x0, y0, tile_size)
                    if result is not None:
                        cx_t, cy_t, w_t, h_t, ang_t = result
                        labels.append(_obb_label_line(cx_t, cy_t, w_t, h_t, ang_t, norm=float(tile_size)))
                tile_id = f"{stem}_c{i:04d}"
                _write_tile(
                    patch,
                    out_dir / "images" / split / f"{tile_id}.png",
                    out_dir / "labels" / split / f"{tile_id}.txt",
                    labels,
                )
                n_tiles += 1

            # Random background tiles from the same frame (no annotation check —
            # occasional overlap is harmless and helps background diversity).
            for bg_i in range(max_bg_tiles):
                bx0 = int(rng.integers(0, max(1, w_arr - tile_size)))
                by0 = int(rng.integers(0, max(1, h_arr - tile_size)))
                tile_id = f"{stem}_bg{bg_i:04d}"
                patch = arr[by0:by0 + tile_size, bx0:bx0 + tile_size]
                _write_tile(
                    patch,
                    out_dir / "images" / split / f"{tile_id}.png",
                    out_dir / "labels" / split / f"{tile_id}.txt",
                    [],
                )
                n_tiles += 1

        else:
            # --- Original blind tiling (background-only frames / Frigate) ---
            bg_tiles: list[tuple[int, int]] = []
            for ty in range(0, h_arr - tile_size + 1, tile_size):
                for tx in range(0, w_arr - tile_size + 1, tile_size):
                    clipped = []
                    for ann in anns:
                        cx, cy, w, h, angle_deg = _obb_fields(ann["obb"])
                        result = _clip_obb_to_tile(cx, cy, w, h, angle_deg, tx, ty, tile_size)
                        if result is not None:
                            clipped.append(result)
                    if clipped:
                        tile_id = f"{stem}_t{tx:04d}_{ty:04d}"
                        patch = arr[ty:ty + tile_size, tx:tx + tile_size]
                        labels = [_obb_label_line(cx, cy, w, h, ad, norm=float(tile_size)) for cx, cy, w, h, ad in clipped]
                        _write_tile(
                            patch,
                            out_dir / "images" / split / f"{tile_id}.png",
                            out_dir / "labels" / split / f"{tile_id}.txt",
                            labels,
                        )
                        n_tiles += 1
                    else:
                        bg_tiles.append((tx, ty))

            rng.shuffle(bg_tiles)
            for tx, ty in bg_tiles[:max_bg_tiles]:
                tile_id = f"{stem}_bg{tx:04d}_{ty:04d}"
                patch = arr[ty:ty + tile_size, tx:tx + tile_size]
                _write_tile(
                    patch,
                    out_dir / "images" / split / f"{tile_id}.png",
                    out_dir / "labels" / split / f"{tile_id}.txt",
                    [],
                )
                n_tiles += 1

    return n_tiles


def materialize_yolo_dataset(
    train_ann: Path,
    val_ann: Path,
    out_dir: Path,
    data_root: Path,
    tile_size: int,
    max_bg_tiles: int,
    seed: int,
    rebuild: bool = False,
    centered: bool = True,
) -> Path:
    """Materialize explicit train/val YOLO-OBB tiles."""
    yaml_path = out_dir / "dataset.yaml"
    if yaml_path.exists() and not rebuild:
        logger.info("Reusing YOLO dataset at %s", out_dir)
        return yaml_path
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_coco = _read_json(train_ann)
    val_coco = _read_json(val_ann)
    n_train = _materialize_split(train_coco, data_root, out_dir, "train", tile_size, max_bg_tiles, seed, centered=centered)
    n_val = _materialize_split(val_coco, data_root, out_dir, "val", tile_size, max_bg_tiles, seed + 1, centered=centered)
    yaml_path.write_text(
        f"path: {out_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\nnc: 1\n"
        "names: ['streak']\n"
    )
    logger.info("Materialized %s: %d train tiles, %d val tiles", out_dir, n_train, n_val)
    return yaml_path


def train_track(
    track: str,
    epochs: int,
    imgsz: int,
    batch: int,
    model_size: str,
    data_root: Path,
    val_ann: Path,
    rebuild_dataset: bool,
    centered: bool = True,
) -> Path:
    """Train one StreakMindYOLO track and return best checkpoint path."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required to train StreakMindYOLO") from exc

    cfg = TRACKS[track]
    work_dir = cfg["work_dir"]
    dataset_dir = work_dir / "dataset"
    yaml_path = materialize_yolo_dataset(
        cfg["ann"],
        val_ann,
        dataset_dir,
        data_root=data_root,
        tile_size=imgsz,
        max_bg_tiles=2,
        seed=42,
        rebuild=rebuild_dataset,
        centered=centered,
    )

    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        else:
            # Ultralytics 8.4.x OBB loss has a MPS-specific bug where
            # counts.max() returns a negative value, crashing torch.zeros().
            # CPU is slower but is the only reliable path on Apple Silicon
            # for YOLO OBB training until upstream fixes the MPS kernel.
            device = "cpu"
    except Exception:
        device = "cpu"
    workers = min(2, os.cpu_count() or 1) if device == "cpu" else min(4, os.cpu_count() or 1)

    model = YOLO(f"yolo11{model_size}-obb.pt")
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(work_dir.resolve()),
        name="run",
        exist_ok=True,
        device=device,
        workers=workers,
        verbose=True,
        save=True,
        val=True,
    )
    return work_dir / "run" / "weights" / "best.pt"


def load_ground_truth(ann_path: Path) -> list[dict[str, Any]]:
    coco = _read_json(ann_path)
    id_to_name = {int(img["id"]): img["file_name"] for img in coco["images"]}
    gts: list[dict[str, Any]] = []
    for ann in coco.get("annotations", []):
        cx, cy, w, h, angle_deg = _obb_fields(ann["obb"])
        gts.append({
            "image_id": id_to_name[int(ann["image_id"])],
            "obb": {"cx": cx, "cy": cy, "w": w, "h": h, "angle_deg": angle_deg},
            "streak_length_px": max(w, h),
        })
    return gts


def predict_yolo(
    weights: Path,
    ann_path: Path,
    data_root: Path,
    imgsz: int,
    conf: float,
    batch: int,
    stride: int | None = None,
    method: str = "streakmind_yolo",
) -> list[dict[str, Any]]:
    """Run tiled YOLO-OBB inference over a COCO file."""
    from ultralytics import YOLO

    coco = _read_json(ann_path)
    loader = FITSLoader()
    yolo = YOLO(str(weights))
    stride = stride or imgsz
    predictions: list[dict[str, Any]] = []

    images = coco.get("images", [])
    for idx, img_info in enumerate(images, start=1):
        fname = str(img_info["file_name"])
        arr = _load_image(data_root / fname, loader)
        h_img, w_img = arr.shape[:2]
        img_preds: list[dict[str, Any]] = []
        if max(h_img, w_img) <= imgsz:
            tiles = [(arr, 0, 0)]
        else:
            tiles = []
            for y0 in range(0, h_img, stride):
                for x0 in range(0, w_img, stride):
                    tile = arr[y0:min(y0 + imgsz, h_img), x0:min(x0 + imgsz, w_img)]
                    if tile.shape[0] >= 32 and tile.shape[1] >= 32:
                        tiles.append((tile, x0, y0))

        logger.info("Evaluating %s (%d/%d, %d tiles)", fname, idx, len(images), len(tiles))
        for start in range(0, len(tiles), batch):
            chunk = tiles[start:start + batch]
            chunk_imgs = [tile for tile, _, _ in chunk]
            results = yolo.predict(chunk_imgs, verbose=False, conf=conf, imgsz=imgsz, batch=batch)
            for result, (_, x_off, y_off) in zip(results, chunk):
                if result.obb is None:
                    continue
                for box, score in zip(result.obb.xywhr, result.obb.conf):
                    cx, cy, bw, bh, angle_rad = box.tolist()
                    cx += x_off
                    cy += y_off
                    img_preds.append({
                        "image_id": fname,
                        "confidence": float(score),
                        "method": method,
                        "obb": {
                            "cx": cx,
                            "cy": cy,
                            "w": bw,
                            "h": bh,
                            "angle_deg": float(angle_rad) * 180.0 / np.pi,
                        },
                        "streak_length_px": float(max(bw, bh)),
                    })

        img_preds.sort(key=lambda p: p["confidence"], reverse=True)
        kept: list[dict[str, Any]] = []
        for pred in img_preds:
            suppress = False
            for existing in kept:
                dx = pred["obb"]["cx"] - existing["obb"]["cx"]
                dy = pred["obb"]["cy"] - existing["obb"]["cy"]
                if float(np.hypot(dx, dy)) < imgsz / 2:
                    suppress = True
                    break
            if not suppress:
                kept.append(pred)
        predictions.extend(kept)

    return predictions


def run_comparison(
    tracks: list[str],
    epochs: int,
    imgsz: int,
    batch: int,
    model_size: str,
    data_root: Path,
    val_ann: Path,
    test_ann: Path,
    results_dir: Path,
    conf: float,
    eval_batch: int,
    eval_stride: int | None,
    rebuild_dataset: bool,
    skip_train: bool,
    centered: bool = True,
) -> dict[str, Any]:
    os.environ["ARGUS_NORM"] = "zscale"
    results_dir.mkdir(parents=True, exist_ok=True)
    gt = load_ground_truth(test_ann)

    summary: dict[str, Any] = {
        "model": "StreakMindYOLO",
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "tracks": {},
    }
    for track in tracks:
        cfg = TRACKS[track]
        ann_path = cfg["ann"]
        if not ann_path.exists():
            logger.warning(
                "Skipping track %r — annotation file not found: %s. "
                "Run the appropriate preparation script first.",
                track, ann_path,
            )
            continue
        weights = cfg["work_dir"] / "run" / "weights" / "best.pt"
        if not skip_train or not weights.exists():
            weights = train_track(track, epochs, imgsz, batch, model_size, data_root, val_ann, rebuild_dataset, centered=centered)
        preds = predict_yolo(weights, test_ann, data_root, imgsz, conf, eval_batch, eval_stride)
        metrics_50 = evaluate(preds, gt, iou_threshold=0.5)
        metrics_80 = evaluate(preds, gt, iou_threshold=0.8)
        track_dir = results_dir / track
        track_dir.mkdir(parents=True, exist_ok=True)
        (track_dir / "predictions.json").write_text(json.dumps(preds, indent=2))
        (track_dir / "metrics_iou50.json").write_text(json.dumps(metrics_50, indent=2))
        (track_dir / "metrics_iou80.json").write_text(json.dumps(metrics_80, indent=2))
        summary["tracks"][track] = {
            "weights": str(weights),
            "predictions": len(preds),
            "metrics_iou50": metrics_50,
            "metrics_iou80": metrics_80,
            # keep 'metrics' key for backward compatibility with existing tooling
            "metrics": metrics_50,
        }
        logger.info(
            "%s: [IoU=0.5] P=%.3f R=%.3f F1=%.3f  [IoU=0.8] P=%.3f R=%.3f F1=%.3f",
            track,
            metrics_50["precision"], metrics_50["recall"], metrics_50["f1"],
            metrics_80["precision"], metrics_80["recall"], metrics_80["f1"],
        )

    (results_dir / "comparison.json").write_text(json.dumps(summary, indent=2))
    (results_dir / "comparison.md").write_text(_comparison_markdown(summary))
    return summary


def _comparison_markdown(summary: dict[str, Any]) -> str:
    header = [
        "# StreakMindYOLO GTImages Comparison",
        "",
        f"- Epochs: `{summary['epochs']}`",
        f"- Image size: `{summary['imgsz']}`",
        f"- Batch: `{summary['batch']}`",
        "- Training domain: raw FITS (GTImages) — methodology-matched to StreakMind",
        "- Evaluation: held-out GTImages test split",
        "",
    ]

    def _table(iou_key: str, iou_label: str) -> list[str]:
        rows = [
            f"## IoU = {iou_label}",
            "",
            "| Track | Precision | Recall | F1 | mAP@0.5 | Angle error | Predictions |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for track, payload in summary["tracks"].items():
            m = payload.get(iou_key, payload.get("metrics", {}))
            rows.append(
                f"| {track} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
                f"{m['map_50']:.3f} | {m['mean_angle_error_deg']:.2f} | "
                f"{payload['predictions']} |"
            )
        # StreakMind reference row (reported in their paper at IoU=0.8)
        if iou_label == "0.8":
            rows.append(
                "| **StreakMind** *(reference)* | **0.940** | **0.970** | **0.955** | — | — | — |"
            )
        rows.append("")
        return rows

    lines = header + _table("metrics_iou50", "0.5") + _table("metrics_iou80", "0.8")
    lines += [
        "## Notes",
        "",
        "- StreakMind reference figures (P=94%, R=97%) are from arXiv:2605.03429, evaluated on",
        "  La Sagra Observatory FITS frames with IoU=0.8. Direct comparison is approximate:",
        "  different site, different epoch, different sky background diversity.",
        "- GTImages is single-night single-site; the combined track adds Frigate FITS frames",
        "  for additional background diversity.",
        "",
    ]
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", nargs="+", default=["real_only", "paper_long", "adapted", "gtimages_plus_frigate"], choices=sorted(TRACKS))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--no-centered", action="store_true", help="Disable streak-centered crops (use legacy uniform tiling)")
    parser.add_argument("--model", default="n", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--val-ann", type=Path, default=Path("data/annotations/gtimages_val.json"))
    parser.add_argument("--test-ann", type=Path, default=Path("data/annotations/gtimages_test.json"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/streakmind_yolo"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument(
        "--eval-stride",
        type=int,
        default=None,
        help="Pixel stride for held-out tiled inference. Defaults to imgsz for non-overlapping smoke evaluation.",
    )
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_comparison(
        tracks=args.tracks,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        model_size=args.model,
        data_root=args.data_root,
        val_ann=args.val_ann,
        test_ann=args.test_ann,
        results_dir=args.results_dir,
        conf=args.conf,
        eval_batch=args.eval_batch,
        eval_stride=args.eval_stride,
        rebuild_dataset=args.rebuild_dataset,
        skip_train=args.skip_train,
        centered=not args.no_centered,
    )


if __name__ == "__main__":
    main()
