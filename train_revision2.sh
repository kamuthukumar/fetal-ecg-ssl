#!/usr/bin/env bash
# Second-round revision runs (Reviewer 2 Q3, Reviewer 3 R10 and R12).
set -euo pipefail
cd "$(dirname "$0")"
DEVICE="${DEVICE:-mps}"
DS=NInFEA,NI_FECG_DB
PY="${PY:-python3}"
LOG=results/revision_train2.log
mkdir -p results
echo "=== revision round-2 training $(date) device=$DEVICE ===" | tee -a "$LOG"

run() {
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] START $name  $*" | tee -a "$LOG"
  "$PY" src/train_foundation.py --device "$DEVICE" --pretrain_datasets "$DS" \
      --epochs 50 --lr 1e-4 "$@" --out_dir "results/$name" \
      2>&1 | tee -a "$LOG"
  echo "[$(date +%H:%M:%S)] DONE  $name" | tee -a "$LOG"
}

# Masked modelling without instance normalisation (Reviewer 2, Q3)
run rev_mae_nonorm   --mode mae    --seed 42 --batch_size 24 --input_norm none

# VICReg with a larger batch for the covariance estimate (Reviewer 3, R10)
run rev_vicreg_b64   --mode vicreg --seed 42 --batch_size 64

echo "=== round-2 training finished $(date) ===" | tee -a "$LOG"
