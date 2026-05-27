#!/usr/bin/env bash
# Overnight training orchestration — DINOv3 ViT-B single best run + optional A/B
#
# Default mode (nodm, 15 epochs, 256px): fits in ~22h on Mac MPS/CPU.
#   Val checkpoints at epochs 5, 10, 15.  Best checkpoint goes to eval.
#
# A/B mode available if DM data consent is resolved:
#   Phase 1A (~21h): no-DM model, 256px, 15 epochs
#   Phase 1B (~21h): with-DM model, 256px, 15 epochs  [skipped if 1A mAP@50 >= SKIP_1B_THRESHOLD]
#
# Usage:
#   bash scripts/train_overnight.sh          # single best run (nodm, 15ep) — fits 24h
#   bash scripts/train_overnight.sh phase1   # A/B comparison (both variants, sequential)
#   bash scripts/train_overnight.sh phase2 nodm   # 400px quality run with no-DM data
#   bash scripts/train_overnight.sh phase2 withdm # 400px quality run with DM data
#
# Logs: weights/run_15ep_nodm/overnight_nodm.log
#       weights/run_15ep_withdm/overnight_withdm.log
#       weights/run_best_400px_{nodm,withdm}/overnight_400px_{nodm,withdm}.log

set -euo pipefail

PYTHON=/Users/robert/miniconda3/envs/satid/bin/python
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WARM_START="weights/run_gt_dm_satstreaks_dinov3_vitb/best_coco_bbox_mAP_epoch_4.pth"
# Skip Phase 1B if Phase 1A mAP@50 at final epoch meets this threshold.
SKIP_1B_THRESHOLD="${SKIP_1B_THRESHOLD:-0.82}"

cd "$REPO_ROOT"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Extract the last reported coco/bbox_mAP_50 value from a log file.
last_map50() {
    local logfile="$1"
    grep -o "coco/bbox_mAP_50: [0-9.]*" "$logfile" 2>/dev/null \
        | tail -1 \
        | awk '{print $2}'
}

# Return 0 (true) if $1 >= $2 (float comparison via python).
float_ge() {
    "$PYTHON" -c "import sys; sys.exit(0 if float('${1:-0}') >= float('${2}') else 1)" 2>/dev/null
}

run_phase1a() {
    log "=== PHASE 1A: no-DM, 256px, 15 epochs ==="
    mkdir -p weights/run_15ep_nodm
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    USE_DEV_SUBSET=false \
    TRAIN_ANN_FILE=annotations/all_train_nodm.json \
    VAL_ANN_FILE=annotations/dm_merged_val.json \
    "$PYTHON" -m training.train_dino \
        --config models/dino/streak_dinov3_vitb_longrun.py \
        --work-dir weights/run_15ep_nodm \
        --load-from "$WARM_START" \
        2>&1 | tee weights/run_15ep_nodm/overnight_nodm.log
    log "=== PHASE 1A COMPLETE ==="
}

run_phase1b() {
    log "=== PHASE 1B: with-DM, 256px, 15 epochs ==="
    mkdir -p weights/run_15ep_withdm
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    USE_DEV_SUBSET=false \
    TRAIN_ANN_FILE=annotations/all_train_withdm.json \
    VAL_ANN_FILE=annotations/dm_merged_val.json \
    "$PYTHON" -m training.train_dino \
        --config models/dino/streak_dinov3_vitb_longrun.py \
        --work-dir weights/run_15ep_withdm \
        --load-from "$WARM_START" \
        2>&1 | tee weights/run_15ep_withdm/overnight_withdm.log
    log "=== PHASE 1B COMPLETE ==="
}

run_phase2() {
    local variant="${1:-nodm}"
    if [[ "$variant" == "withdm" ]]; then
        TRAIN_ANN="annotations/all_train_withdm.json"
        WORK_DIR="weights/run_best_400px_withdm"
        LOG_TAG="withdm"
    else
        TRAIN_ANN="annotations/all_train_nodm.json"
        WORK_DIR="weights/run_best_400px_nodm"
        LOG_TAG="nodm"
    fi
    log "=== PHASE 2: ${variant}, 400px, 50 epochs ==="
    mkdir -p "$WORK_DIR"
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    USE_DEV_SUBSET=false \
    TRAIN_ANN_FILE="$TRAIN_ANN" \
    VAL_ANN_FILE=annotations/dm_merged_val.json \
    "$PYTHON" -m training.train_dino \
        --config models/dino/streak_dinov3_vitb_400px.py \
        --work-dir "$WORK_DIR" \
        --load-from "$WARM_START" \
        2>&1 | tee "${WORK_DIR}/overnight_400px_${LOG_TAG}.log"
    log "=== PHASE 2 COMPLETE — best checkpoint in $WORK_DIR ==="
}

report_best() {
    log "=== PHASE 1 SUMMARY ==="
    local map50_nodm="" map50_withdm=""
    for dir in weights/run_15ep_nodm weights/run_15ep_withdm; do
        if [[ -d "$dir" ]]; then
            best=$(ls "$dir"/best_coco_bbox_mAP_epoch_*.pth 2>/dev/null | tail -1 || echo "none")
            log "$dir → best checkpoint: $best"
            logfile=$(ls "$dir"/overnight_*.log 2>/dev/null | head -1 || true)
            if [[ -n "$logfile" && -f "$logfile" ]]; then
                map50=$(last_map50 "$logfile")
                log "  Last val mAP@50: ${map50:-N/A}"
                [[ "$dir" == *nodm* ]] && map50_nodm="$map50"
                [[ "$dir" == *withdm* ]] && map50_withdm="$map50"
            fi
        fi
    done

    if [[ -n "$map50_nodm" && -n "$map50_withdm" ]]; then
        log ""
        log "no-DM mAP@50: $map50_nodm  |  with-DM mAP@50: $map50_withdm"
        if float_ge "$map50_nodm" "$map50_withdm"; then
            log "WINNER: no-DM (nodm)"
            echo "nodm"
        else
            log "WINNER: with-DM (withdm)"
            echo "withdm"
        fi
    elif [[ -n "$map50_nodm" ]]; then
        log "Only Phase 1A completed — using nodm for Phase 2"
        echo "nodm"
    else
        log "WARNING: could not determine winner — defaulting to nodm"
        echo "nodm"
    fi
    log ""
}

# --- Main ---
MODE="${1:-all}"

case "$MODE" in
    all)
        run_phase1a
        run_phase1b
        WINNER=$(report_best)
        log "Auto-proceeding to Phase 2 with winner: $WINNER"
        run_phase2 "$WINNER"
        ;;
    phase1)
        run_phase1a
        run_phase1b
        report_best > /dev/null
        log "Phase 1 complete. Run: bash scripts/train_overnight.sh phase2 [nodm|withdm]"
        ;;
    phase2)
        VARIANT="${2:-nodm}"
        run_phase2 "$VARIANT"
        ;;
    *)
        echo "Usage: $0 [all|phase1|phase2 [nodm|withdm]]"
        exit 1
        ;;
esac
