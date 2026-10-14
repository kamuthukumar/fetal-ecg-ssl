#!/usr/bin/env bash
# Extra seeds for masked modelling without instance normalisation.
set -euo pipefail
cd "$(dirname "$0")"
DEVICE="${DEVICE:-mps}"
DS=NInFEA,NI_FECG_DB
PY="${PY:-python3}"
LOG=results/revision_train3.log
mkdir -p results
echo "=== revision round-3 training $(date) device=$DEVICE ===" | tee -a "$LOG"

run() {
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] START $name  $*" | tee -a "$LOG"
  "$PY" src/train_foundation.py --device "$DEVICE" --pretrain_datasets "$DS" \
      --batch_size 24 --epochs 50 --lr 1e-4 "$@" --out_dir "results/$name" \
      2>&1 | tee -a "$LOG"
  echo "[$(date +%H:%M:%S)] DONE  $name" | tee -a "$LOG"
}

run rev_mae_nonorm_s7    --mode mae --seed 7   --input_norm none
run rev_mae_nonorm_s123  --mode mae --seed 123 --input_norm none

echo "=== round-3 training finished $(date) ===" | tee -a "$LOG"
