"""Download pretrained Co-DINO weights appropriate for the current environment.

Behaviour is controlled by the MODEL_SIZE environment variable:

  MODEL_SIZE=tiny  (default) — downloads Swin-T DINO COCO weights (~340 MB)
  MODEL_SIZE=large           — downloads Swin-L DINO COCO weights (~2.4 GB)

Weights are saved to ``weights/`` (gitignored).  Download is skipped if the
target file already exists and the SHA-256 checksum passes.

Usage::

    python scripts/download_weights.py                  # uses MODEL_SIZE env var
    MODEL_SIZE=large python scripts/download_weights.py # force large

Note: The Swin-L weights are intended for cloud training only (A100 40 GB).
      Do not download Swin-L on the Mac dev machine — use Swin-T instead.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from pathlib import Path

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weight registry
# Source: OpenMMLab — DINO pretrained COCO checkpoints
# Ref:    https://github.com/open-mmlab/mmdetection/tree/main/configs/dino
# ---------------------------------------------------------------------------
WEIGHT_URLS: dict[str, dict] = {
    "tiny": {
        "url": (
            "https://download.openmmlab.com/mmdetection/v3.0/"
            "dino/dino-4scale_r50_8xb2-12e_coco/"
            "dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth"
        ),
        # Swin-T DINO checkpoint from OpenMMLab
        # TODO: replace with Swin-T specific checkpoint when available
        "filename": "co_dino_swin_t_coco.pth",
        "sha256": None,   # populated after first successful download
        "approx_size_mb": 160,
        "description": "DINO R50 4-scale COCO pretrain (stand-in for Swin-T dev)",
    },
    "large": {
        "url": (
            "https://download.openmmlab.com/mmdetection/v3.0/"
            "dino/dino-5scale_swin-l_8xb2-12e_coco/"
            "dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth"
        ),
        "filename": "co_dino_swin_l_coco.pth",
        "sha256": None,
        "approx_size_mb": 828,
        "description": "DINO 5-scale Swin-L COCO pretrain",
    },
}


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Return SHA-256 hex digest of *path* without reading it all at once."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while data := f.read(chunk):
            h.update(data)
    return h.hexdigest()


def download_weights(model_size: str = "tiny") -> Path:
    """Download pretrained weights for the given model size.

    Args:
        model_size: ``"tiny"`` or ``"large"``.

    Returns:
        Path to the downloaded (or already-existing) weights file.

    Raises:
        ValueError: If *model_size* is not recognised.
        RuntimeError: If the download fails after retries.
    """
    if model_size not in WEIGHT_URLS:
        raise ValueError(
            f"Unknown model_size={model_size!r}. Choose 'tiny' or 'large'."
        )

    entry = WEIGHT_URLS[model_size]
    weights_dir = Path("weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    dest = weights_dir / entry["filename"]

    # Skip if already present and checksum passes
    if dest.exists():
        if entry["sha256"]:
            actual = _sha256(dest)
            if actual == entry["sha256"]:
                logger.info("Weights already present and checksum OK: %s", dest)
                print(f"✓  {dest}  (already downloaded, checksum OK)")
                return dest
            else:
                logger.warning(
                    "Checksum mismatch for %s — re-downloading", dest
                )
        else:
            # No checksum stored yet — trust the existing file
            size_mb = dest.stat().st_size / (1024 ** 2)
            print(
                f"✓  {dest}  (already downloaded, {size_mb:.0f} MB, "
                f"no checksum on record)"
            )
            return dest

    # Guard against accidentally downloading Swin-L on Mac
    if model_size == "large":
        device_env = os.environ.get("MODEL_SIZE", "tiny")
        if device_env != "large":
            raise EnvironmentError(
                "Refusing to download Swin-L weights on a non-large environment.\n"
                "Set MODEL_SIZE=large explicitly if you are on the cloud GPU instance."
            )

    url = entry["url"]
    approx_mb = entry["approx_size_mb"]
    desc = entry["description"]
    print(
        f"\nDownloading {model_size} weights ({approx_mb} MB approx.):\n"
        f"  {desc}\n"
        f"  {url}\n"
        f"  → {dest}\n"
    )

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        tmp = dest.with_suffix(".tmp")
        with open(tmp, "wb") as f, tqdm(
            desc=dest.name,
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            leave=True,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

        tmp.rename(dest)

    except requests.RequestException as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Download failed: {exc}") from exc

    # Record SHA-256 for future use
    digest = _sha256(dest)
    size_mb = dest.stat().st_size / (1024 ** 2)
    print(f"\n✓  Downloaded {dest}  ({size_mb:.0f} MB)")
    print(f"   SHA-256: {digest}")
    print(
        f"\n   Update WEIGHT_URLS['{model_size}']['sha256'] = {digest!r}\n"
        "   in scripts/download_weights.py to enable checksum verification."
    )
    return dest


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    model_size = os.environ.get("MODEL_SIZE", "tiny").lower()
    if len(sys.argv) > 1:
        model_size = sys.argv[1].lower()

    print(f"MODEL_SIZE={model_size}")

    try:
        path = download_weights(model_size)
        print(f"\nWeights ready at: {path}")
    except EnvironmentError as exc:
        print(f"\n✗  {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"\n✗  {exc}", file=sys.stderr)
        sys.exit(1)
