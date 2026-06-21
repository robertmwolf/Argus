"""Visualize false negatives and false positives from a heatmap eval run.

Uses the same matching logic and band thresholds as eval/geometry_metrics.py
(perp_tol=5px, short<400px / medium 400–1000px / long≥1000px).

Coordinates in val_balanced_v1* annotations and predictions are in tile-local
space (relative to each image's tile_origin in the full FITS).  We crop the
FITS to the tile region so that annotations and predictions are displayed at
the correct location.

Produces one PNG per missed annotation and one per false-positive image:
  Left:   tile region of the FITS (z-score display)
  Centre: heatmap overlay
  Right:  ALL GT (green) + ALL predictions (red) — TP and FP both shown

Usage:
    PYTHONPATH=/Users/robert/Argus python scripts/visualize_fn_fp.py \\
        --predictions results/window_v9/vits_v9_asl_cldice/pf85_no_sattrains/predictions_t085.json \\
        --annotations /Volumes/External/TrainingData/annotations/val_balanced_v1_no_sattrains.json \\
        --checkpoint weights/vits_v9_asl_cldice/best.pt \\
        --output-dir results/window_v9/vits_v9_asl_cldice/pf85_no_sattrains/viz
"""

from __future__ import annotations
import argparse, json, math, os
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from astropy.io import fits

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("VITS_V9_HEATMAP_THRESHOLD", "0.85")
os.environ.setdefault("VITS_V9_HEATMAP_NATIVE_TILE_SIZE", "400")
os.environ.setdefault("VITS_V9_HEATMAP_TILE_OVERLAP", "0.0")
os.environ.setdefault("VITS_V9_HEATMAP_PEAK_FLOOR", "0.85")


# Match geometry_metrics.py band definitions exactly
def _band(length_px: float) -> str:
    if length_px < 400.0:
        return "short"
    if length_px < 1000.0:
        return "medium"
    return "long"


def load_fits_tile(path: str, tile_origin: list[int], tile_w: int, tile_h: int) -> np.ndarray:
    """Load and z-score normalise a tile sub-region from a FITS file."""
    with fits.open(path, memmap=False) as h:
        data = np.asarray(h[0].data, dtype=np.float32)
    ox, oy = int(tile_origin[0]), int(tile_origin[1])
    tile = data[oy:oy + tile_h, ox:ox + tile_w]
    finite = tile[np.isfinite(tile)]
    if finite.size == 0:
        return np.zeros_like(tile)
    m, s = float(finite.mean()), float(finite.std())
    if s < 1e-6:
        s = 1.0
    return np.clip((tile - m) / s, -3, 3)


def zscore_to_display(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


def draw_segment(ax, x1, y1, x2, y2, color, lw=2, label=None):
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, solid_capstyle="round")
    ax.plot([x1, x2], [y1, y2], "o", color=color, markersize=4)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my, label, color=color, fontsize=7, ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5, ec="none"))


def make_panel(gray: np.ndarray, heatmap: np.ndarray | None,
               gt_segs: list[dict], pred_segs: list[dict],
               title: str, out_path: Path) -> None:
    disp = zscore_to_display(gray)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="black")
    fig.suptitle(title, color="white", fontsize=9, y=1.01)

    for ax in axes:
        ax.set_facecolor("black")
        ax.tick_params(colors="gray", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("gray")

    # Panel 1: tile image with GT
    axes[0].imshow(disp, cmap="gray", origin="upper", vmin=0, vmax=255)
    axes[0].set_title("Tile (z-score) + GT", color="white", fontsize=9)
    for g in gt_segs:
        draw_segment(axes[0], g["x1"], g["y1"], g["x2"], g["y2"], "#00ff88", lw=1.5)

    # Panel 2: heatmap overlay
    axes[1].imshow(disp, cmap="gray", origin="upper", vmin=0, vmax=255, alpha=0.6)
    if heatmap is not None:
        axes[1].imshow(heatmap, cmap="inferno", origin="upper", alpha=0.6, vmin=0, vmax=1)
    axes[1].set_title("Heatmap overlay", color="white", fontsize=9)
    for g in gt_segs:
        draw_segment(axes[1], g["x1"], g["y1"], g["x2"], g["y2"], "#00ff88", lw=1.5)
    for p in pred_segs:
        draw_segment(axes[1], p["x1"], p["y1"], p["x2"], p["y2"], "#ff4444", lw=1.5)

    # Panel 3: all GT and all predictions
    axes[2].imshow(disp, cmap="gray", origin="upper", vmin=0, vmax=255)
    axes[2].set_title("GT (green) vs All Predictions (red)", color="white", fontsize=9)
    for g in gt_segs:
        lbl = f"GT {round(math.hypot(g['x2']-g['x1'], g['y2']-g['y1']))}px"
        draw_segment(axes[2], g["x1"], g["y1"], g["x2"], g["y2"], "#00ff88", lw=2, label=lbl)
    for p in pred_segs:
        lbl = f"P {round(p.get('streak_length_px', 0))}px c={p.get('confidence', 0):.2f}"
        draw_segment(axes[2], p["x1"], p["y1"], p["x2"], p["y2"], "#ff4444", lw=2, label=lbl)

    legend_els = [Line2D([0], [0], color="#00ff88", lw=2, label="Ground truth"),
                  Line2D([0], [0], color="#ff4444", lw=2, label="Prediction")]
    axes[2].legend(handles=legend_els, loc="upper right", fontsize=7,
                   facecolor="black", labelcolor="white", edgecolor="gray")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"  → {out_path}")


def resolve_fits(file_name: str) -> str:
    if Path(file_name).exists():
        return file_name
    candidate = str(Path("/Volumes/External/TrainingData") / file_name)
    if Path(candidate).exists():
        return candidate
    return file_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True,
                    help="predictions_t085.json from evaluate_dinov3_heatmap.py")
    ap.add_argument("--annotations",
                    default="/Volumes/External/TrainingData/annotations/val_balanced_v1_no_sattrains.json")
    ap.add_argument("--checkpoint", default="weights/vits_v9_asl_cldice/best.pt")
    ap.add_argument("--output-dir",
                    default="results/window_v9/vits_v9_asl_cldice/pf85_no_sattrains/viz")
    ap.add_argument("--perp-tol", type=float, default=20.0,
                    help="Perpendicular match tolerance in px (matches geometry_metrics default)")
    args = ap.parse_args()

    from eval.streak_metrics import _match_all_segments
    from inference.streak_segment import StreakSegment, detection_dict_to_segment
    from training.annotation_endpoints import annotation_to_endpoints

    all_preds_raw = json.loads(Path(args.predictions).read_text())
    all_anns = json.loads(Path(args.annotations).read_text())

    id_to_img = {int(img["id"]): img for img in all_anns["images"]}

    # Build GT segments
    gt_segs: list[StreakSegment] = []
    for ann in all_anns["annotations"]:
        x1, y1, x2, y2 = annotation_to_endpoints(ann)
        img_id = int(ann["image_id"])
        gt_segs.append(StreakSegment(
            image_id=img_id,
            x1=x1, y1=y1, x2=x2, y2=y2,
            confidence=1.0,
        ))

    # Build prediction segments
    pred_segs: list[StreakSegment] = []
    for p in all_preds_raw:
        pred_segs.append(detection_dict_to_segment(p))

    # Run the same matching as geometry_metrics
    is_tp, n_gt, matched_pairs = _match_all_segments(
        pred_segs, gt_segs,
        angle_tol_deg=10.0,
        perp_tol_px=args.perp_tol,
        length_iou_min=0.3,
    )

    # Identify unmatched GTs (false negatives)
    matched_gt_ids: set[int] = set()
    for _pred, gt in matched_pairs:
        matched_gt_ids.add(id(gt))

    fn_gts: list[StreakSegment] = [g for g in gt_segs if id(g) not in matched_gt_ids]

    # Identify unmatched predictions (false positives)
    fp_preds: list[StreakSegment] = [p for p, tp in zip(pred_segs, is_tp) if not tp]

    print(f"\nMatching results (perp_tol={args.perp_tol}px):")
    print(f"  GT: {len(gt_segs)}, matched: {len(matched_gt_ids)}, FN: {len(fn_gts)}")
    print(f"  Preds: {len(pred_segs)}, TP: {sum(is_tp)}, FP: {len(fp_preds)}")

    for band in ("short", "medium", "long"):
        fn_band = [g for g in fn_gts if _band(g.length_px) == band]
        gt_band = [g for g in gt_segs if _band(g.length_px) == band]
        recall = (len(gt_band) - len(fn_band)) / len(gt_band) if gt_band else float("nan")
        print(f"  {band:6s}: {len(gt_band):3d} GT, {len(fn_band):3d} FN, recall={recall:.3f}")

    # Collect image IDs needing heatmaps
    fn_img_ids = {g.image_id for g in fn_gts}
    fp_img_ids = {p.image_id for p in fp_preds}
    all_img_ids = fn_img_ids | fp_img_ids

    print(f"\nGenerating heatmaps for {len(all_img_ids)} images...")
    from inference.vits_window_v9_detector import run_vits_v9_heatmap_detector_and_heatmap

    ckpt = Path(args.checkpoint)
    heatmaps: dict[int, np.ndarray | None] = {}
    gray_tiles: dict[int, np.ndarray] = {}

    for i, img_id in enumerate(sorted(all_img_ids), 1):
        img_meta = id_to_img[img_id]
        fits_path = resolve_fits(img_meta["file_name"])
        tile_origin = img_meta.get("tile_origin", [0, 0])
        tile_w = img_meta.get("width")
        tile_h = img_meta.get("height")
        print(f"  [{i}/{len(all_img_ids)}] {Path(fits_path).name}  "
              f"tile=({tile_origin[0]},{tile_origin[1]}) {tile_w}x{tile_h}")

        gray = load_fits_tile(fits_path, tile_origin, tile_w, tile_h)
        gray_tiles[img_id] = gray

        disp_arr = zscore_to_display(gray)
        rgb = np.stack([disp_arr] * 3, axis=2)
        _, hmap = run_vits_v9_heatmap_detector_and_heatmap(rgb, checkpoint=ckpt)
        heatmaps[img_id] = hmap

    # Build display dicts by image
    def seg_to_dict(s: StreakSegment) -> dict:
        return {"x1": s.x1, "y1": s.y1, "x2": s.x2, "y2": s.y2,
                "streak_length_px": s.length_px,
                "confidence": s.confidence}

    gt_by_img: dict[int, list[dict]] = defaultdict(list)
    for g in gt_segs:
        gt_by_img[g.image_id].append(seg_to_dict(g))

    pred_by_img: dict[int, list[dict]] = defaultdict(list)
    for p in pred_segs:
        pred_by_img[p.image_id].append(seg_to_dict(p))

    out_dir = Path(args.output_dir)

    # --- False negatives ---
    print(f"\nRendering {len(fn_gts)} false-negative panels...")
    for g in sorted(fn_gts, key=lambda s: s.length_px):
        img_id = g.image_id
        fits_path = resolve_fits(id_to_img[img_id]["file_name"])
        fname = Path(fits_path).stem
        band = _band(g.length_px)
        length = round(g.length_px)
        n_pred = len(pred_by_img[img_id])
        title = (f"FALSE NEGATIVE  |  {fname}  |  band={band}  len={length}px  "
                 f"|  {len(gt_by_img[img_id])} GT / {n_pred} pred  "
                 f"|  perp_tol={args.perp_tol}px")
        out_path = out_dir / "false_negatives" / f"fn_{band}_{length:04d}px_{fname}.png"
        make_panel(gray_tiles[img_id], heatmaps[img_id],
                   gt_by_img[img_id], pred_by_img[img_id], title, out_path)

    # --- False positives (one panel per image) ---
    print(f"\nRendering {len(fp_img_ids)} false-positive image panels...")
    for img_id in sorted(fp_img_ids):
        fits_path = resolve_fits(id_to_img[img_id]["file_name"])
        fname = Path(fits_path).stem
        fp_count = sum(1 for p in fp_preds if p.image_id == img_id)
        title = (f"FALSE POSITIVE  |  {fname}  "
                 f"|  {len(gt_by_img[img_id])} GT / {len(pred_by_img[img_id])} pred  "
                 f"|  {fp_count} unmatched")
        out_path = out_dir / "false_positives" / f"fp_{fname}.png"
        make_panel(gray_tiles[img_id], heatmaps[img_id],
                   gt_by_img[img_id], pred_by_img[img_id], title, out_path)

    print(f"\nDone. Panels written to {out_dir}/")


if __name__ == "__main__":
    main()
