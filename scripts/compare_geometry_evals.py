#!/usr/bin/env python3
"""Print a comparison table of geometry_eval.json files across all models.

Reads geometry_eval.json files from a predefined list of result dirs,
then prints a formatted table with Tier 1 (detection) and Tier 2 (raw geometry)
columns.  Run with no args.

Usage:
    python scripts/compare_geometry_evals.py [--md]

    --md   Emit GitHub-flavored Markdown table instead of plain text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# (display_name, path_to_geometry_eval)
MODELS: list[tuple[str, str]] = [
    ("run15_seg_nms",    "results/run15_vits_segment_nms/balanced_v1/pf85/geometry_eval_tier3.json"),
    ("run15_vits",       "results/run15_vits/balanced_v1/pf85/geometry_eval_tier3.json"),
    ("run17_vitb",       "results/run17_vitb/balanced_v1/pf85/geometry_eval_tier3.json"),
    ("run18_vitb",       "results/run18_vitb/balanced_v1/pf85/geometry_eval.json"),
    ("run19_vitb",       "results/run19_vitb/balanced_v1/pf85/geometry_eval.json"),
    ("run20_vits_ctrl",  "results/run20_control/run20_vits_control/pf85/geometry_eval.json"),
    ("run20_vitb_ctrl",  "results/run20_control/run20_vitb_control/pf85/geometry_eval.json"),
    ("vits_window_v1",   "results/window_v1/vits_window_v1/pf85/geometry_eval_tier3.json"),
    ("vitb_window_v1",   "results/window_v1/vitb_window_v1/pf85/geometry_eval_tier3.json"),
    ("vits_window_v2",   "results/window_v2/vits_window_v2/pf85/geometry_eval_tier3.json"),
    ("vitb_window_v2",   "results/window_v2/vitb_window_v2/pf85/geometry_eval_tier3.json"),
    ("vits_window_v3",   "results/window_v3/vits_window_v3/pf85/geometry_eval.json"),
    ("vitb_window_v3",   "results/window_v3/vitb_window_v3/pf85/geometry_eval.json"),
    ("vits_window_v4",   "results/window_v4/vits_window_v4/pf85/geometry_eval.json"),
    ("vitb_window_v4",   "results/window_v4/vitb_window_v4/pf85/geometry_eval.json"),
    ("vits_window_v5",   "results/window_v5/vits_window_v5/pf85/geometry_eval.json"),
    ("vits_window_v6",   "results/window_v6/vits_window_v6/pf85/geometry_eval.json"),
    ("vits_window_v7",   "results/window_v7/vits_window_v7/pf85/geometry_eval.json"),
    ("vits_window_v8",   "results/window_v8/vits_window_v8/pf85/geometry_eval.json"),
    ("vitb_window_v8",   "results/window_v8/vitb_window_v8/pf85/geometry_eval.json"),
    # v9 loss ablation (ViT-S, 5 loss modes)
    ("v9_focal_dice",    "results/window_v9/vits_v9_focal_dice/pf85/geometry_eval.json"),
    ("v9_asl_dice",      "results/window_v9/vits_v9_asl_dice/pf85/geometry_eval.json"),
    ("v9_focal_cldice",  "results/window_v9/vits_v9_focal_cldice/pf85/geometry_eval.json"),
    ("v9_tversky",       "results/window_v9/vits_v9_tversky/pf85/geometry_eval.json"),
    ("v9_asl_cldice",    "results/window_v9/vits_v9_asl_cldice/pf85/geometry_eval.json"),
    ("yolo_streakmind",  "results/streakmind_yolo_real/balanced_v1/pf85/geometry_eval.json"),
    ("yolo_run17_cpu",   "results/yolo_run17_cpu/balanced_v1/bestf1_conf20/geometry_eval.json"),
]


def _na(value: float | None, fmt: str = ".3f") -> str:
    if value is None:
        return "n/a"
    return format(value, fmt)


def load_row(name: str, path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    t1 = d["tier1_detection"]
    t2 = d["tier2_raw_geometry"]
    t3 = d.get("tier3_refined_geometry")
    pb = t1["per_band"]
    n_gt = t1["n_gt"]
    n_found = t1["n_found"]

    # T3 is valid only when images were loaded (improvement != 0.0 OR model has 0 recall)
    t3_valid = False
    t3_stale = False
    t3_ang: float | None = None
    t3_ep:  float | None = None
    if t3 and t3.get("n_pairs", 0) > 0:
        improvement = t3.get("angle_improvement_deg", 0.0)
        t2_ang_val = t2["angle_err_deg"]["mean"]
        t3_ang = t3["angle_err_deg"]["mean"]
        t3_ep  = t3["endpoint_err_px"]["mean"]
        if improvement != 0.0 or t2_ang_val == 0.0:
            t3_valid = True
        else:
            t3_stale = True

    return {
        "name":        name,
        "n_gt":        n_gt,
        "t1_recall":   t1["detection_recall"],
        "t1_prec":     t1["detection_precision"],
        "t1_short":    pb["short"]["recall"],
        "t1_medium":   pb["medium"]["recall"],
        "t1_long":     pb["long"]["recall"],
        "t2_ang":      t2["angle_err_deg"]["mean"],
        "t2_ep":       t2["endpoint_err_px"]["mean"],
        "t3_ang":      t3_ang,
        "t3_ep":       t3_ep,
        "t3_valid":    t3_valid,
        "t3_stale":    t3_stale,
    }


def print_table(rows: list[dict], md: bool) -> None:
    cols = [
        ("Model",         "name",      "<22", "s"),
        ("T1 Rcl",        "t1_recall", ">7",  ".3f"),
        ("T1 Prec",       "t1_prec",   ">7",  ".3f"),
        ("Short",         "t1_short",  ">6",  ".3f"),
        ("Med",           "t1_medium", ">6",  ".3f"),
        ("Long",          "t1_long",   ">6",  ".3f"),
        ("T2 Ang°",       "t2_ang",    ">8",  ".2f"),
        ("T2 EndPx",      "t2_ep",     ">9",  ".1f"),
        ("T3 Ang°",       "t3_ang",    ">8",  ""),
        ("T3 EndPx",      "t3_ep",     ">9",  ""),
    ]

    def _fmt_t3(r: dict, key: str, label: str) -> str:
        """Format a T3 value; mark as stale if images weren't loaded."""
        v = r.get(key)
        if v is None:
            return "n/a"
        if r.get("t3_stale"):
            return "(stale)"
        if not r.get("t3_valid"):
            return "n/a"
        return _na(v, ".2f") if "Ang" in label else _na(v, ".1f")

    if md:
        header = "| " + " | ".join(c[0] for c in cols) + " |"
        sep    = "| " + " | ".join("---" for _ in cols) + " |"
        print(header)
        print(sep)
        for r in rows:
            vals = []
            for label, key, width, fmt in cols:
                if key in ("t3_ang", "t3_ep"):
                    vals.append(_fmt_t3(r, key, label))
                    continue
                v = r.get(key)
                if v is None:
                    vals.append("n/a")
                elif isinstance(v, str):
                    vals.append(v)
                elif fmt:
                    vals.append(format(v, fmt))
                else:
                    vals.append(_na(v, ".2f") if "Ang" in label else _na(v, ".1f"))
            print("| " + " | ".join(vals) + " |")
    else:
        header = "  ".join(format(c[0], c[2]) for c in cols)
        sep    = "  ".join("-" * len(format(c[0], c[2])) for c in cols)
        print(header)
        print(sep)
        for r in rows:
            parts = []
            for label, key, width, fmt in cols:
                if key in ("t3_ang", "t3_ep"):
                    parts.append(format(_fmt_t3(r, key, label), width))
                    continue
                v = r.get(key)
                if v is None:
                    parts.append(format("n/a", width))
                elif isinstance(v, str):
                    parts.append(format(v, width))
                elif fmt:
                    parts.append(format(v, width.lstrip("<>") + fmt))
                else:
                    s = _na(v, ".2f") if "Ang" in label else _na(v, ".1f")
                    parts.append(format(s, width))
            print("  ".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare geometry eval results across models")
    ap.add_argument("--md", action="store_true", help="Output Markdown table")
    args = ap.parse_args()

    rows = []
    for name, path in MODELS:
        row = load_row(name, path)
        if row is None:
            print(f"  [missing] {name}: {path}")
        else:
            rows.append(row)

    print(f"\nGeometry evaluation on val_balanced_v1.json  (threshold=0.70, peak_floor=0.85)\n")
    print_table(rows, args.md)
    print(f"\nT1 = Tier 1 detection (did model find the streak?)  Rcl=recall  Prec=precision")
    print(f"T2 = Raw model output geometry (no post-processing)")
    print(f"T3 = After Radon angle + endpoint refinement (n/a if --images-dir not provided)")


if __name__ == "__main__":
    main()
