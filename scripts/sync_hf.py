"""Sync ARGUS annotations to/from Hugging Face Hub.

Trained model weights are committed to the git repository and do not need
syncing. This script handles only the annotation JSONs, which live on the
external drive and are too large to commit to git.

Usage:
    # Upload canonical annotations:
    python scripts/sync_hf.py --upload

    # Remove stale files from the HF dataset repo:
    python scripts/sync_hf.py --cleanup

    # Download to a fresh machine:
    python scripts/sync_hf.py --download --annotations-dir /path/to/annotations/

Repo:
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
DATASET_REPO = f"{HF_USER}/argus-dataset"

EXTERNAL_ANNOTATIONS = Path("/Volumes/External/TrainingData/annotations")

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


def cleanup_stale(api: HfApi) -> None:
    """Delete stale annotation files from the HF dataset repo."""
    log.info("Cleaning up stale annotations from %s", DATASET_REPO)
    existing = set(api.list_repo_files(DATASET_REPO, repo_type="dataset"))
    for rel in ANNOTATIONS_STALE:
        if rel not in existing:
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
    mode.add_argument("--upload", action="store_true", help="Push annotations to HF")
    mode.add_argument("--download", action="store_true", help="Pull annotations from HF")
    mode.add_argument("--cleanup", action="store_true", help="Delete stale files from the HF dataset repo")

    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=EXTERNAL_ANNOTATIONS,
        help="Local annotations directory (default: /Volumes/External/TrainingData/annotations/)",
    )

    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    if args.cleanup:
        cleanup_stale(api)
    elif args.upload:
        upload_annotations(api)
    else:
        download_annotations(args.annotations_dir)


if __name__ == "__main__":
    main()
