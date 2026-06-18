"""Print a summary table of threshold-sweep eval results.

Usage:
    python scripts/print_sweep_results.py results/run8_sweep
    python scripts/print_sweep_results.py results/run8_sweep --sort recall
    python scripts/print_sweep_results.py results/run8_sweep --gate-precision 0.01 --gate-recall 0.60
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


_DIR_RE = re.compile(
    r"(?P<backbone>convnext|vits)_t(?P<thresh>[0-9.]+)_(?P<stitch>stitch|nostitch)$"
)


def _load_results(sweep_dir: Path) -> list[dict]:
    rows = []
    for metrics_file in sorted(sweep_dir.rglob("metrics.json")):
        m = _DIR_RE.search(metrics_file.parent.name)
        if m is None:
            continue
        try:
            payload = json.loads(metrics_file.read_text())
        except Exception:
            continue
        met = payload.get("metrics", {})
        per_band = met.get("per_band", {})
        rows.append({
            "backbone": m.group("backbone"),
            "thresh":   float(m.group("thresh")),
            "stitch":   m.group("stitch") == "stitch",
            "precision": met.get("precision", 0.0),
            "recall":    met.get("recall", 0.0),
            "f1":        met.get("f1", 0.0),
            "med_recall": per_band.get("medium", {}).get("recall", 0.0),
            "long_recall": per_band.get("long", {}).get("recall", 0.0),
            "n_preds":   payload.get("n_predictions", 0),
            "path":      str(metrics_file),
        })
    return rows


def _fmt(v: float, pct: bool = False, decimals: int = 1) -> str:
    if pct:
        return f"{v * 100:.{decimals}f}%"
    return f"{v:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sweep_dir", help="Directory produced by run_threshold_sweep.sh")
    ap.add_argument("--sort", choices=["thresh", "recall", "precision", "f1"],
                    default="thresh", help="Sort key (default: thresh)")
    ap.add_argument("--gate-precision", type=float, default=None,
                    help="Highlight rows meeting this precision threshold")
    ap.add_argument("--gate-recall", type=float, default=None,
                    help="Highlight rows meeting this recall threshold")
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.is_dir():
        print(f"ERROR: {sweep_dir} is not a directory", file=sys.stderr)
        return 1

    rows = _load_results(sweep_dir)
    if not rows:
        print(f"No metrics.json files found under {sweep_dir}", file=sys.stderr)
        return 1

    sort_key = {
        "thresh":    lambda r: (r["backbone"], r["thresh"], r["stitch"]),
        "recall":    lambda r: -r["recall"],
        "precision": lambda r: -r["precision"],
        "f1":        lambda r: -r["f1"],
    }[args.sort]
    rows.sort(key=sort_key)

    gate_p = args.gate_precision
    gate_r = args.gate_recall

    header = f"{'backbone':<10} {'thresh':>6}  {'stitch':>6}  {'precision':>10}  {'recall':>7}  {'F1':>7}  {'med_R':>6}  {'long_R':>7}  {'preds':>7}"
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    n_gate = 0
    for r in rows:
        meets_gate = (
            (gate_p is None or r["precision"] >= gate_p) and
            (gate_r is None or r["recall"] >= gate_r)
        )
        if meets_gate and (gate_p is not None or gate_r is not None):
            n_gate += 1
            flag = " ◀ GATE"
        else:
            flag = ""

        print(
            f"{r['backbone']:<10} {r['thresh']:>6.2f}  {'yes' if r['stitch'] else 'no':>6}  "
            f"{_fmt(r['precision'], pct=True):>10}  "
            f"{_fmt(r['recall'],    pct=True):>7}  "
            f"{_fmt(r['f1'],        pct=True):>7}  "
            f"{_fmt(r['med_recall'],pct=True):>6}  "
            f"{_fmt(r['long_recall'],pct=True):>7}  "
            f"{r['n_preds']:>7}"
            f"{flag}"
        )

    print(sep)
    print(f"{len(rows)} runs loaded from {sweep_dir}")
    if gate_p is not None or gate_r is not None:
        gate_desc = []
        if gate_p is not None:
            gate_desc.append(f"precision≥{gate_p*100:.1f}%")
        if gate_r is not None:
            gate_desc.append(f"recall≥{gate_r*100:.1f}%")
        print(f"Rows meeting gate ({' AND '.join(gate_desc)}): {n_gate}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
