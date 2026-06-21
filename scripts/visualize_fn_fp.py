"""Visualize false negatives and false positives from v9 heatmap eval.

Produces one PNG per missed annotation and one per false positive, showing:
  Left:   original FITS image (z-score display)
  Centre: heatmap overlay
  Right:  annotation vs prediction overlay

Usage:
    python scripts/visualize_fn_fp.py \
        --fn-fp-json /tmp/v9_fn_fp.json \
        --checkpoint weights/vits_v9_asl_cldice/best.pt \
        --predictions results/window_v9/vits_v9_asl_cldice/pf85_no_sattrains/predictions_t085.json \
        --output-dir results/window_v9/vits_v9_asl_cldice/pf85_no_sattrains/viz
"""

from __future__ import annotations
import argparse, json, math, os, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from astropy.io import fits

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("VITS_V9_HEATMAP_THRESHOLD", "0.85")
os.environ.setdefault("VITS_V9_HEATMAP_NATIVE_TILE_SIZE", "400")
os.environ.setdefault("VITS_V9_HEATMAP_TILE_OVERLAP", "0.0")
os.environ.setdefault("VITS_V9_HEATMAP_PEAK_FLOOR", "0.85")


def load_fits(path: str) -> np.ndarray:
    with fits.open(path, memmap=False) as h:
        data = np.asarray(h[0].data, dtype=np.float32)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data)
    m, s = float(finite.mean()), float(finite.std())
    if s < 1e-6: s = 1.0
    return np.clip((data - m) / s, -3, 3)


def zscore_to_display(arr: np.ndarray) -> np.ndarray:
    """Map [-3, 3] z-score to [0, 255] uint8 for display."""
    lo, hi = arr.min(), arr.max()
    if hi == lo: return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


def draw_segment(ax, x1, y1, x2, y2, color, lw=2, label=None):
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, solid_capstyle='round')
    # endpoints
    ax.plot([x1, x2], [y1, y2], 'o', color=color, markersize=4)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my, label, color=color, fontsize=7, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.5, ec='none'))


def make_panel(fits_path: str, heatmap: np.ndarray,
               gt_segs: list[dict], pred_segs: list[dict],
               title: str, out_path: Path) -> None:
    gray = load_fits(fits_path)
    disp = zscore_to_display(gray)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='black')
    fig.suptitle(title, color='white', fontsize=10, y=1.01)

    for ax in axes:
        ax.set_facecolor('black')
        ax.tick_params(colors='gray', labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')

    # Panel 1: raw image
    axes[0].imshow(disp, cmap='gray', origin='upper', vmin=0, vmax=255)
    axes[0].set_title('Original (z-score)', color='white', fontsize=9)
    for g in gt_segs:
        draw_segment(axes[0], g['x1'],g['y1'],g['x2'],g['y2'], '#00ff88', lw=1.5)

    # Panel 2: heatmap overlay
    axes[1].imshow(disp, cmap='gray', origin='upper', vmin=0, vmax=255, alpha=0.6)
    if heatmap is not None:
        axes[1].imshow(heatmap, cmap='inferno', origin='upper',
                       alpha=0.6, vmin=0, vmax=1)
    axes[1].set_title('Heatmap overlay', color='white', fontsize=9)
    for g in gt_segs:
        draw_segment(axes[1], g['x1'],g['y1'],g['x2'],g['y2'], '#00ff88', lw=1.5)
    for p in pred_segs:
        draw_segment(axes[1], p['x1'],p['y1'],p['x2'],p['y2'], '#ff4444', lw=1.5)

    # Panel 3: annotation vs prediction
    axes[2].imshow(disp, cmap='gray', origin='upper', vmin=0, vmax=255)
    axes[2].set_title('GT (green) vs Predictions (red)', color='white', fontsize=9)
    for g in gt_segs:
        lbl = f"GT {round(math.hypot(g['x2']-g['x1'],g['y2']-g['y1']))}px"
        draw_segment(axes[2], g['x1'],g['y1'],g['x2'],g['y2'], '#00ff88', lw=2, label=lbl)
    for p in pred_segs:
        lbl = f"P {round(p.get('streak_length_px',0))}px c={p.get('confidence',0):.2f}"
        draw_segment(axes[2], p['x1'],p['y1'],p['x2'],p['y2'], '#ff4444', lw=2, label=lbl)

    legend_els = [Line2D([0],[0], color='#00ff88', lw=2, label='Ground truth'),
                  Line2D([0],[0], color='#ff4444', lw=2, label='Prediction')]
    axes[2].legend(handles=legend_els, loc='upper right', fontsize=7,
                   facecolor='black', labelcolor='white', edgecolor='gray')

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    print(f"  → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fn-fp-json', required=True)
    ap.add_argument('--checkpoint', default='weights/vits_v9_asl_cldice/best.pt')
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--annotations',
                    default='/Volumes/External/TrainingData/annotations/val_balanced_v1_no_sattrains.json')
    ap.add_argument('--output-dir', default='results/window_v9/vits_v9_asl_cldice/pf85_no_sattrains/viz')
    args = ap.parse_args()

    fn_fp   = json.loads(Path(args.fn_fp_json).read_text())
    all_preds = json.loads(Path(args.predictions).read_text())
    all_anns  = json.loads(Path(args.annotations).read_text())

    id_to_img    = {int(img['id']): img for img in all_anns['images']}
    preds_by_img = defaultdict(list)
    for p in all_preds:
        preds_by_img[int(p['image_id'])].append(p)

    from training.annotation_endpoints import annotation_to_endpoints
    anns_by_img = defaultdict(list)
    for ann in all_anns['annotations']:
        x1,y1,x2,y2 = annotation_to_endpoints(ann)
        anns_by_img[int(ann['image_id'])].append(
            {**ann, 'x1':x1,'y1':y1,'x2':x2,'y2':y2})

    # Collect unique image IDs needed
    fn_img_ids = {int(f['img']['id']) for f in fn_fp['fn']}
    fp_img_ids = {int(f['img']['id']) for f in fn_fp['fp']}
    all_img_ids = fn_img_ids | fp_img_ids

    print(f"Generating heatmaps for {len(all_img_ids)} images...")
    from inference.vits_window_v9_detector import run_vits_v9_heatmap_detector_and_heatmap

    ckpt = Path(args.checkpoint)
    heatmaps: dict[int, np.ndarray | None] = {}
    for i, img_id in enumerate(sorted(all_img_ids), 1):
        img_meta = id_to_img[img_id]
        fits_path = img_meta['file_name']
        if not Path(fits_path).exists():
            fits_path = str(Path('/Volumes/External/TrainingData') / fits_path)
        print(f"  [{i}/{len(all_img_ids)}] {Path(fits_path).name}")
        gray = load_fits(fits_path)
        disp_arr = zscore_to_display(gray)
        rgb = np.stack([disp_arr]*3, axis=2)
        _, hmap = run_vits_v9_heatmap_detector_and_heatmap(rgb, checkpoint=ckpt)
        heatmaps[img_id] = hmap

    out_dir = Path(args.output_dir)

    # --- False negatives ---
    print(f"\nRendering {len(fn_fp['fn'])} false-negative panels...")
    for entry in fn_fp['fn']:
        img_id  = int(entry['img']['id'])
        img_meta = id_to_img[img_id]
        fits_path = img_meta['file_name']
        if not Path(fits_path).exists():
            fits_path = str(Path('/Volumes/External/TrainingData') / fits_path)

        fname = Path(fits_path).stem
        band  = entry['band']
        length = round(entry['length'])

        gt_segs  = anns_by_img[img_id]
        pred_segs = preds_by_img[img_id]

        title = (f"FALSE NEGATIVE  |  {fname}  |  band={band}  len={length}px  "
                 f"|  {len(gt_segs)} GT / {len(pred_segs)} pred on image")

        out_path = out_dir / 'false_negatives' / f"fn_{band}_{length:04d}px_{fname}.png"
        make_panel(fits_path, heatmaps[img_id], gt_segs, pred_segs, title, out_path)

    # --- False positives ---
    print(f"\nRendering {len(fp_img_ids)} false-positive image panels...")
    for img_id in sorted(fp_img_ids):
        img_meta = id_to_img[img_id]
        fits_path = img_meta['file_name']
        if not Path(fits_path).exists():
            fits_path = str(Path('/Volumes/External/TrainingData') / fits_path)

        fname = Path(fits_path).stem
        gt_segs   = anns_by_img[img_id]
        pred_segs = preds_by_img[img_id]
        fp_preds  = [p for i,p in enumerate(all_preds)
                     if int(p['image_id'])==img_id and
                     not any(f['pred']['image_id']==img_id and
                             abs(f['pred']['x1']-p['x1'])<1 for f in fn_fp['fp'])]

        title = (f"FALSE POSITIVE  |  {fname}  "
                 f"|  {len(gt_segs)} GT / {len(pred_segs)} pred on image")
        out_path = out_dir / 'false_positives' / f"fp_{fname}.png"
        make_panel(fits_path, heatmaps[img_id], gt_segs, pred_segs, title, out_path)

    print(f"\nDone. Panels written to {out_dir}/")


if __name__ == '__main__':
    main()
