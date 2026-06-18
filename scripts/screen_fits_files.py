"""Pre-screen FITS files in a COCO annotation for hang-inducing files.

Spawns a subprocess per file and kills it after --timeout seconds.
Files that are killed are written to a skip-list JSON so that
build_clean_annotation.py (or a manual edit) can exclude them.

Usage:
    python scripts/screen_fits_files.py \\
        --annotations data/annotations/test_atwood.json \\
        --timeout 60 \\
        --skip-list /tmp/bad_fits.json

    # Then build a clean annotation:
    python scripts/screen_fits_files.py \\
        --annotations data/annotations/test_atwood.json \\
        --skip-list /tmp/bad_fits.json \\
        --output    /tmp/test_atwood_clean.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_LOADER_SCRIPT = """
import sys
from pathlib import Path
sys.path.insert(0, '{repo}')
from inference.fits_loader import FITSLoader
import numpy as np
loader = FITSLoader()
path = Path(sys.argv[1])
arr = np.asarray(loader.load(path)['array'], dtype=np.uint8)
print(f'OK {{arr.shape}}')
"""


def _test_file(path: str, python: str, timeout: int) -> bool:
    """Return True if the file loads within timeout seconds."""
    script = _LOADER_SCRIPT.format(repo=str(_REPO))
    try:
        result = subprocess.run(
            [python, "-c", script, path],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--timeout", type=int, default=60,
                    help="Seconds before a file is declared bad (default: 60)")
    ap.add_argument("--skip-list", default=None,
                    help="Path to write/read the bad-file list JSON")
    ap.add_argument("--output", default=None,
                    help="If set, write a clean annotation JSON excluding bad files")
    ap.add_argument("--python", default=sys.executable,
                    help="Python interpreter to use for subprocess tests")
    args = ap.parse_args()

    ann = json.loads(Path(args.annotations).read_text())
    images = ann["images"]

    # ── Mode: build clean annotation from existing skip-list ────────────────
    if args.skip_list and args.output and Path(args.skip_list).exists():
        bad = set(json.loads(Path(args.skip_list).read_text()))
        clean_imgs = [img for img in images if img["file_name"] not in bad]
        clean_ids = {img["id"] for img in clean_imgs}
        clean_anns = [a for a in ann.get("annotations", [])
                      if a["image_id"] in clean_ids]
        out = {**ann, "images": clean_imgs, "annotations": clean_anns}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out))
        print(f"Clean annotation: {len(clean_imgs)}/{len(images)} images "
              f"({len(images)-len(clean_imgs)} excluded)")
        print(f"Wrote → {args.output}")
        return 0

    # ── Mode: screen files ──────────────────────────────────────────────────
    bad: list[str] = []
    for i, img in enumerate(images):
        path = img["file_name"]
        suffix = Path(path).suffix.lower()
        if suffix not in {".fits", ".fit", ".fts"}:
            continue  # PNGs don't hang
        ok = _test_file(path, args.python, args.timeout)
        status = "OK" if ok else "BAD"
        if not ok:
            bad.append(path)
        if (i + 1) % 20 == 0 or not ok:
            print(f"[{i+1}/{len(images)}] {status}  {Path(path).name}")

    print(f"\nScreening complete: {len(bad)} bad files out of {len(images)}")
    for b in bad:
        print(f"  BAD: {b}")

    if args.skip_list:
        Path(args.skip_list).parent.mkdir(parents=True, exist_ok=True)
        Path(args.skip_list).write_text(json.dumps(bad, indent=2))
        print(f"\nSkip-list written → {args.skip_list}")
        print("To build clean annotation:")
        print(f"  python scripts/screen_fits_files.py "
              f"--annotations {args.annotations} "
              f"--skip-list {args.skip_list} "
              f"--output /tmp/test_atwood_clean.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
