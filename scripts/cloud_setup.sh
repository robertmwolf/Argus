#!/usr/bin/env bash
# cloud_setup.sh — one-time environment setup for ARGUS GPU training
#
# Tested on: Ubuntu 22.04 LTS, NVIDIA RTX 5070 Ti (Blackwell, 16 GB)
# Run once after cloning the repo and before launching training.
#
# Usage:
#   chmod +x scripts/cloud_setup.sh
#   ./scripts/cloud_setup.sh
#
# What this does:
#   1. Installs Miniconda if not already present
#   2. Creates the 'satid' conda environment
#   3. Installs PyTorch 2.6+ (Blackwell-compatible) + all dependencies
#   4. Downloads Swin-L pretrain weights (~828 MB)
#   5. Verifies CUDA is visible from Python
#   6. Sets persistent environment variables in ~/.bashrc
#   7. Runs the pre-flight checklist (SKIP_SMOKE_TRAIN=1 for speed)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="satid"
PYTHON="/root/miniconda3/envs/${CONDA_ENV}/bin/python"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║          ARGUS Cloud Training Environment Setup      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ---------------------------------------------------------------------------
# 1. Miniconda
# ---------------------------------------------------------------------------
echo "── Step 1: Miniconda ────────────────────────────────────"
if command -v conda &>/dev/null; then
    echo "  ✓  conda already installed: $(conda --version)"
else
    echo "  Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /root/miniconda3
    rm /tmp/miniconda.sh
    export PATH="/root/miniconda3/bin:$PATH"
    conda init bash
    echo "  ✓  Miniconda installed"
fi

# Activate conda for this script
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# ---------------------------------------------------------------------------
# 2. Conda environment
# ---------------------------------------------------------------------------
echo ""
echo "── Step 2: Conda environment '${CONDA_ENV}' ─────────────"
if conda env list | grep -q "^${CONDA_ENV} "; then
    echo "  ✓  Environment '${CONDA_ENV}' already exists"
else
    echo "  Creating environment (Python 3.11)..."
    conda create -n "${CONDA_ENV}" python=3.11 -y
    echo "  ✓  Environment created"
fi
conda activate "${CONDA_ENV}"

# ---------------------------------------------------------------------------
# 3. PyTorch 2.6+ (Blackwell / CUDA 12.8)
# ---------------------------------------------------------------------------
echo ""
echo "── Step 3: PyTorch 2.6+ (Blackwell-compatible) ─────────"
cd "${REPO_ROOT}"

# Install PyTorch first (pinned for Blackwell / CUDA 12.8 compatibility)
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128 \
    --quiet

# Verify CUDA is visible
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available after PyTorch install!"
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  ✓  PyTorch {torch.__version__}  —  {name} ({vram:.1f} GB VRAM)")
EOF

# Install remaining dependencies
echo "  Installing project dependencies from requirements.txt..."
pip install -r requirements.txt --quiet
pip install mmdet mmengine mmcv --quiet

echo "  ✓  All dependencies installed"

# ---------------------------------------------------------------------------
# 4. Download Swin-L pretrain weights
# ---------------------------------------------------------------------------
echo ""
echo "── Step 4: Download Swin-L weights (~828 MB) ────────────"
MODEL_SIZE=large python scripts/download_weights.py
echo "  ✓  Weights ready"

# ---------------------------------------------------------------------------
# 5. GTImages conversion (if data/GTImages/ is present)
# ---------------------------------------------------------------------------
echo ""
echo "── Step 5: GTImages annotation conversion ───────────────"
if [ -d "${REPO_ROOT}/data/GTImages" ] && \
   [ "$(ls "${REPO_ROOT}/data/GTImages/"*.fits 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  Found GTImages FITS files — converting to COCO JSON..."
    python scripts/convert_gtimages.py \
        --strk-dir data/GTImages \
        --output data/annotations/gtimages.json \
        --negatives-output data/annotations/gtimages_negatives.json
    echo "  ✓  GTImages converted"
else
    echo "  ! data/GTImages/ not found or empty"
    echo "    Download GTImages.zip, unzip it into data/GTImages/, then re-run:"
    echo "      python scripts/convert_gtimages.py \\"
    echo "          --strk-dir data/GTImages \\"
    echo "          --output data/annotations/gtimages.json"
fi

# ---------------------------------------------------------------------------
# 6. SatStreaks check
# ---------------------------------------------------------------------------
echo ""
echo "── Step 6: SatStreaks dataset check ─────────────────────"
SATSTREAKS_IMGS="${REPO_ROOT}/data/satstreaks/Data/Images"
if [ -d "${SATSTREAKS_IMGS}" ] && \
   [ "$(ls "${SATSTREAKS_IMGS}"/*.jpg 2>/dev/null | wc -l)" -gt 100 ]; then
    COUNT=$(ls "${SATSTREAKS_IMGS}"/*.jpg | wc -l)
    echo "  ✓  SatStreaks images found (${COUNT} images)"
else
    echo "  ! SatStreaks images not found at ${SATSTREAKS_IMGS}"
    echo "    Download from the link in data/satstreaks/README.md"
    echo "    Extract so that data/satstreaks/Data/Images/*.jpg exists"
fi

# ---------------------------------------------------------------------------
# 7. Merge annotations into train/val splits
# ---------------------------------------------------------------------------
echo ""
echo "── Step 7: Merge annotations → train/val splits ─────────"
if [ -f "${REPO_ROOT}/data/annotations/train.json" ] && \
   [ -f "${REPO_ROOT}/data/annotations/val.json" ]; then
    echo "  ✓  train.json and val.json already exist — skipping merge"
else
    if [ -f "${REPO_ROOT}/data/annotations/gtimages.json" ] && \
       [ -d "${SATSTREAKS_IMGS}" ]; then
        python scripts/merge_annotations.py
    else
        echo "  ! Cannot merge — waiting for GTImages and/or SatStreaks"
        echo "    Once both datasets are in place, run:"
        echo "      python scripts/merge_annotations.py"
    fi
fi

# ---------------------------------------------------------------------------
# 8. Persistent environment variables
# ---------------------------------------------------------------------------
echo ""
echo "── Step 8: Persistent environment variables ─────────────"
BASHRC="${HOME}/.bashrc"

_add_env() {
    local var="$1" val="$2"
    if grep -q "^export ${var}=" "${BASHRC}" 2>/dev/null; then
        sed -i "s|^export ${var}=.*|export ${var}=${val}|" "${BASHRC}"
    else
        echo "export ${var}=${val}" >> "${BASHRC}"
    fi
}

_add_env "MODEL_SIZE"                     "large"
_add_env "USE_DEV_SUBSET"                 "false"
_add_env "DATABASE_URL"                   "sqlite+aiosqlite:///./argus.db"
_add_env "PYTORCH_CUDA_ALLOC_CONF"        "expandable_segments:True"
_add_env "PYTORCH_ENABLE_MPS_FALLBACK"    "0"

echo "  ✓  Variables written to ${BASHRC}"
echo "     Run 'source ~/.bashrc' or start a new shell to apply them."

# ---------------------------------------------------------------------------
# 9. Pre-flight checklist
# ---------------------------------------------------------------------------
echo ""
echo "── Step 9: Pre-flight checklist (SKIP_SMOKE_TRAIN=1) ────"
cd "${REPO_ROOT}"
SKIP_SMOKE_TRAIN=1 \
MODEL_SIZE=large \
USE_DEV_SUBSET=false \
python scripts/prepare_cloud_training.py || true

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Setup complete.  Next steps:                        ║"
echo "║                                                      ║"
echo "║  1. source ~/.bashrc                                 ║"
echo "║  2. (If prompted) complete data downloads above     ║"
echo "║  3. python scripts/merge_annotations.py             ║"
echo "║  4. python scripts/prepare_cloud_training.py        ║"
echo "║  5. MODEL_SIZE=large python -m training.train_dino \\ ║"
echo "║         --work-dir weights/run_001                   ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
