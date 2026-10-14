#!/usr/bin/env bash
# Extra SimCLR seeds so masked-vs-SimCLR can be compared across seeds (Reviewer 3, R4/R6).
set -euo pipefail
cd "$(dirname "$0")"
DEVICE="${DEVICE:-mps}"
DS=NInFEA,NI_FECG_DB
PY="${PY:-python3}"
LOG=results/revision_train_seeds.log
mkdir -p results
echo "=== seed training $(date) device=$DEVICE ===" | tee -a "$LOG"

run() {
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] START $name  $*" | tee -a "$LOG"
  "$PY" src/train_foundation.py --device "$DEVICE" --pretrain_datasets "$DS" \
      --batch_size 24 --epochs 50 --lr 1e-4 "$@" --out_dir "results/$name" \
      2>&1 | tee -a "$LOG"
  echo "[$(date +%H:%M:%S)] DONE  $name" | tee -a "$LOG"
}

run rev_simclr_s7    --mode simclr --seed 7
run rev_simclr_s123  --mode simclr --seed 123

echo "=== all seed training finished $(date) ===" | tee -a "$LOG"
