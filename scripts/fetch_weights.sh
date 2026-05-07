#!/usr/bin/env bash
# fetch_weights.sh — pull trained weights and results from the training machine
#
# Usage:
#   ./scripts/fetch_weights.sh user@<training-machine-ip>
#
# Example:
#   ./scripts/fetch_weights.sh robert@192.168.1.42
#   ./scripts/fetch_weights.sh ubuntu@10.0.0.5
#
# What is copied back:
#   weights/run_001/          — all checkpoints (best.pth, latest.pth, epochNN.pth)
#   results/                  — benchmark JSON, confusion matrix PNG, per-image CSV
#   eval/results/             — per-image prediction dumps
#   training/logs/            — TensorBoard / MMDet log files
#
# The local weights/ directory is gitignored; results/ JSON files are committed.

set -euo pipefail

REMOTE="${1:-}"
if [ -z "${REMOTE}" ]; then
    echo "Usage: $0 user@<host>"
    echo ""
    echo "Example:"
    echo "  $0 ubuntu@192.168.1.42"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_REPO="${2:-~/Argus}"   # path on the remote machine (default ~/Argus)

echo ""
echo "Fetching training outputs from ${REMOTE}:${REMOTE_REPO}"
echo ""

# ---------------------------------------------------------------------------
# Pull weights
# ---------------------------------------------------------------------------
echo "── Weights ──────────────────────────────────────────────"
mkdir -p "${REPO_ROOT}/weights"
rsync -avz --progress \
    "${REMOTE}:${REMOTE_REPO}/weights/run_001/" \
    "${REPO_ROOT}/weights/run_001/"
echo ""

# Also pull a canonical best.pth to weights/ root for easy pipeline use
BEST_REMOTE="${REMOTE}:${REMOTE_REPO}/weights/run_001/best_coco_bbox_mAP*.pth"
# shellcheck disable=SC2029
BEST_NAME=$(ssh "${REMOTE}" "ls ${REMOTE_REPO}/weights/run_001/best_coco_bbox_mAP*.pth 2>/dev/null | tail -1" || true)
if [ -n "${BEST_NAME}" ]; then
    rsync -avz --progress \
        "${REMOTE}:${BEST_NAME}" \
        "${REPO_ROOT}/weights/best.pth"
    echo "  ✓  Best checkpoint saved as weights/best.pth"
fi

# ---------------------------------------------------------------------------
# Pull results / eval
# ---------------------------------------------------------------------------
echo "── Results & eval ───────────────────────────────────────"
mkdir -p "${REPO_ROOT}/results" "${REPO_ROOT}/eval/results"

rsync -avz --progress \
    "${REMOTE}:${REMOTE_REPO}/results/" \
    "${REPO_ROOT}/results/"

rsync -avz --progress \
    "${REMOTE}:${REMOTE_REPO}/eval/results/" \
    "${REPO_ROOT}/eval/results/"

# ---------------------------------------------------------------------------
# Pull training logs
# ---------------------------------------------------------------------------
echo "── Training logs ────────────────────────────────────────"
mkdir -p "${REPO_ROOT}/training/logs"
rsync -avz --progress \
    "${REMOTE}:${REMOTE_REPO}/training/logs/" \
    "${REPO_ROOT}/training/logs/" 2>/dev/null || true

rsync -avz --progress \
    "${REMOTE}:${REMOTE_REPO}/weights/run_001/*.log" \
    "${REPO_ROOT}/weights/run_001/" 2>/dev/null || true

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Fetch complete.  Local paths:                       ║"
echo "║    weights/best.pth           — best checkpoint      ║"
echo "║    weights/run_001/           — all checkpoints      ║"
echo "║    results/                   — benchmark JSON/PNG   ║"
echo "║    training/logs/             — TensorBoard logs     ║"
echo "║                                                      ║"
echo "║  To run final evaluation locally:                    ║"
echo "║    python -m eval.benchmark \\                        ║"
echo "║        --run-pipeline \\                              ║"
echo "║        --annotations data/annotations/val.json \\    ║"
echo "║        --output results/final_eval.json              ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
