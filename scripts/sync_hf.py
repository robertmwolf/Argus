"""Sync ARGUS weights and annotations to/from Hugging Face Hub.

Usage:
    # Upload everything (run from repo root):
    python scripts/sync_hf.py --upload

    # Upload only weights or only annotations:
    python scripts/sync_hf.py --upload --weights-only
    python scripts/sync_hf.py --upload --annotations-only

    # Upload weights + LICENSE + README:
    python scripts/sync_hf.py --upload --weights-only

    # Remove stale files from the HF model repo:
    python scripts/sync_hf.py --cleanup

    # Download to a fresh machine (colleague setup):
    python scripts/sync_hf.py --download --weights-dir weights/ --annotations-dir /path/to/annotations/

Repos:
    lonewolfman22/argus-weights   (model repo)     — flat weights/<name>/best.pt
    lonewolfman22/argus-dataset   (dataset repo)   — flat annotations/<name>.json

Authentication:
    Run `huggingface-cli login` once, or set HF_TOKEN env var.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

HF_USER = "lonewolfman22"
WEIGHTS_REPO = f"{HF_USER}/argus-weights"
DATASET_REPO = f"{HF_USER}/argus-dataset"

REPO_ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = REPO_ROOT / "weights"
EXTERNAL_ANNOTATIONS = Path("/Volumes/External/TrainingData/annotations")

# Explicit allowlist — only current/active weights.
# Superseded runs (8–12, 15, 17, run_plain_dinov3) are excluded.
# DINOv3 pretrained backbones are NOT hosted here; see DEVELOPER.md for how
# to obtain them from Meta Research (github.com/facebookresearch/dinov3).
WEIGHTS_INCLUDE = [
    # Production heatmap detector (ViT-S, v11 dataset, ASL + clDice, 400 px tiles)
    "vits_v11_asl_cldice/best.pt",
    "vits_v11_asl_cldice/history.json",
]

# Files present in the HF model repo that are no longer current.
# Run `python scripts/sync_hf.py --cleanup` to remove them.
WEIGHTS_STALE = [
    "run15_vits/best.pt",
    "run15_vits/history.json",
    "run17_vitb/best.pt",
    "run17_vitb/history.json",
    "run5_vits_mmdet/best_coco_bbox_mAP_epoch_15.pth",
    # Superseded by vits_v11_asl_cldice
    "vits_v9_asl_cldice/best.pt",
    "vits_v9_asl_cldice/history.json",
    # Backbone comparison experiment — ViT-S wins; retired
    "vitb_v10_asl_cldice/best.pt",
    "vitb_v10_asl_cldice/history.json",
    # DINOv3 pretrained backbones — redistributed from Meta Research; removed
    # from this repo. Users must download directly from facebookresearch/dinov3.
    "dinov3_vits16_lvd1689m.pth",
    "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "dinov3_vitl16_lvd1689m.pth",
]
# Canonical v11 annotations — the only files needed to reproduce training and eval.
# Uploaded flat to annotations/ in the dataset repo.
ANNOTATIONS_CANONICAL = [
    "all_train_run17_merged_no_sattrains.json",  # canonical training source (v11 input)
    "val_balanced_v1_no_sattrains.json",          # canonical eval set
    "sat_train_excluded.json",                    # satellite-train exclusion manifest
]

# Annotation paths present in the HF dataset repo that are no longer current.
# Run `python scripts/sync_hf.py --cleanup` to remove them.
ANNOTATIONS_STALE = [
    # Old source/ subdir layout — superseded by flat annotations/
    "annotations/source/all_train_run5_tiled_ts1800.json",
    "annotations/source/val_run17_fits.json",
    "annotations/source/val_atwood.json",
    "annotations/source/test_atwood.json",
    "annotations/source/frigate_streaks.json",
    # Old derived/ subdir layout
    "annotations/derived/all_train_run17_merged.json",
    "annotations/derived/synth_run13_short_npy.json",
    "annotations/derived/synth_run13_medium_npy.json",
    "annotations/derived/dev_subset.json",
]


HF_README_FRONTMATTER = """\
---
license: other
license_name: polyform-noncommercial-1.0.0
license_link: https://polyformproject.org/licenses/noncommercial/1.0.0
---

"""


def upload_weights(api: HfApi) -> None:
    """Upload allowlisted weight files, LICENSE, and README to the model repo."""
    log.info("Uploading weights/ → %s", WEIGHTS_REPO)
    for rel in WEIGHTS_INCLUDE:
        path = WEIGHTS_DIR / rel
        if not path.exists():
            log.warning("  MISSING (skipping): %s", rel)
            continue
        log.info("  uploading %s", rel)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=rel,
            repo_id=WEIGHTS_REPO,
            repo_type="model",
        )

    # Keep LICENSE in sync with the GitHub repo.
    license_path = REPO_ROOT / "LICENSE"
    if license_path.exists():
        log.info("  uploading LICENSE")
        api.upload_file(
            path_or_fileobj=str(license_path),
            path_in_repo="LICENSE",
            repo_id=WEIGHTS_REPO,
            repo_type="model",
        )
    else:
        log.warning("  LICENSE not found at %s — skipping", license_path)

    # Upload README with HF YAML frontmatter prepended.
    readme_path = REPO_ROOT / "README.md"
    if readme_path.exists():
        log.info("  uploading README.md")
        readme_body = readme_path.read_text(encoding="utf-8")
        readme_content = HF_README_FRONTMATTER + readme_body
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=WEIGHTS_REPO,
            repo_type="model",
        )
    else:
        log.warning("  README.md not found at %s — skipping", readme_path)

    log.info("Weights upload complete.")


def cleanup_stale(api: HfApi) -> None:
    """Delete stale files from the HF model and dataset repos."""
    log.info("Cleaning up stale weights from %s", WEIGHTS_REPO)
    existing_weights = set(api.list_repo_files(WEIGHTS_REPO, repo_type="model"))
    for rel in WEIGHTS_STALE:
        if rel not in existing_weights:
            log.info("  already absent: %s", rel)
            continue
        log.info("  deleting %s", rel)
        api.delete_file(path_in_repo=rel, repo_id=WEIGHTS_REPO, repo_type="model")

    log.info("Cleaning up stale annotations from %s", DATASET_REPO)
    existing_ann = set(api.list_repo_files(DATASET_REPO, repo_type="dataset"))
    for rel in ANNOTATIONS_STALE:
        if rel not in existing_ann:
            log.info("  already absent: %s", rel)
            continue
        log.info("  deleting %s", rel)
        api.delete_file(path_in_repo=rel, repo_id=DATASET_REPO, repo_type="dataset")

    log.info("Cleanup complete.")


def upload_annotations(api: HfApi) -> None:
    """Upload canonical annotation JSONs from the external drive to the dataset repo."""
    if not EXTERNAL_ANNOTATIONS.exists():
        log.error(
            "External annotations directory not found: %s — is the drive mounted?",
            EXTERNAL_ANNOTATIONS,
        )
        return

    log.info("Uploading annotations/ → %s", DATASET_REPO)
    for name in ANNOTATIONS_CANONICAL:
        path = EXTERNAL_ANNOTATIONS / name
        if not path.exists():
            log.warning("  MISSING (skipping): %s", name)
            continue
        repo_path = f"annotations/{name}"
        log.info("  uploading %s → %s", name, repo_path)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=repo_path,
            repo_id=DATASET_REPO,
            repo_type="dataset",
        )
    log.info("Annotations upload complete.")


def download_weights(dest: Path) -> None:
    """Download all weights from the model repo."""
    dest.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s → %s", WEIGHTS_REPO, dest)
    snapshot_download(
        repo_id=WEIGHTS_REPO,
        repo_type="model",
        local_dir=str(dest),
        ignore_patterns=["*.DS_Store"],
    )
    log.info("Weights download complete → %s", dest)


def download_annotations(dest: Path) -> None:
    """Download canonical annotation JSONs from the dataset repo into dest/."""
    dest.mkdir(parents=True, exist_ok=True)
    log.info("Downloading annotations from %s → %s", DATASET_REPO, dest)
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        local_dir=str(dest),
        ignore_patterns=["*.DS_Store"],
    )
    log.info("Annotations download complete → %s", dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--upload", action="store_true", help="Push local files to HF")
    mode.add_argument("--download", action="store_true", help="Pull files from HF")
    mode.add_argument("--cleanup", action="store_true", help="Delete stale files from the HF model repo")

    parser.add_argument("--weights-only", action="store_true", help="Only sync weights")
    parser.add_argument("--annotations-only", action="store_true", help="Only sync annotations")

    # Download destinations (override defaults for colleague machines)
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=WEIGHTS_DIR,
        help="Local weights directory (default: weights/)",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=EXTERNAL_ANNOTATIONS,
        help="Local annotations directory (default: /Volumes/External/TrainingData/annotations/)",
    )

    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    do_weights = not args.annotations_only
    do_annotations = not args.weights_only

    if args.cleanup:
        cleanup_stale(api)
    elif args.upload:
        if do_weights:
            upload_weights(api)
        if do_annotations:
            upload_annotations(api)
    else:
        if do_weights:
            download_weights(args.weights_dir)
        if do_annotations:
            download_annotations(args.annotations_dir)


if __name__ == "__main__":
    main()
