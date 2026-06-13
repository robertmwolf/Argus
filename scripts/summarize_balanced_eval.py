"""Format the val_balanced_v1 sweep results into a comparison table.

Reads metrics_t*.json written by evaluate_dinov3_heatmap.py --heatmap-cache
across one or more result dirs (each dir = one model × peak-floor setting) and
prints a per-band P/R/F1 table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _row(tag: str, m: dict, n: int) -> str:
    pb = m["per_band"]
    return (f"{tag:<22} {m['precision']:>5.3f} {m['recall']:>6.3f} {m['f1']:>5.3f}  | "
            f"{pb['short']['recall']:>5.3f} {pb['medium']['recall']:>6.3f} {pb['long']['recall']:>5.3f}  | "
            f"{pb['short']['f1']:>5.3f} {pb['medium']['f1']:>6.3f} {pb['long']['f1']:>5.3f}  {n:>5}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="result dirs containing metrics_t*.json")
    ap.add_argument("--labels", nargs="+", default=None, help="label per dir")
    args = ap.parse_args()
    labels = args.labels or [Path(d).name for d in args.dirs]

    print(f"{'model/setting @ thr':<22} {'Prec':>5} {'Rec':>6} {'F1':>5}  | "
          f"{'sh-R':>5} {'med-R':>6} {'lg-R':>5}  | {'sh-F':>5} {'med-F':>6} {'lg-F':>5}  {'Npred':>5}")
    print("-" * 92)
    for d, label in zip(args.dirs, labels):
        files = sorted(Path(d).glob("metrics_t*.json"))
        for f in files:
            data = json.loads(f.read_text())
            thr = data.get("threshold", f.stem.split("_t")[-1])
            print(_row(f"{label}@{thr}", data["metrics"], data.get("n_predictions", 0)))
        print()


if __name__ == "__main__":
    main()
