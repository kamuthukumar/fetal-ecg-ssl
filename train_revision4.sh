#!/usr/bin/env bash
# Masked modelling on the expanded open corpus (Reviewer 3, R12) and VICReg
# with a larger batch (Reviewer 3, R10).
# VICReg at batch 64 exceeds 16 GB unified memory on Apple silicon; batch 48
# needs about 11 GB and must run without other large jobs.
set -euo pipefail
cd "$(dirname "$0")"
DEVICE="${DEVICE:-mps}"
PY="${PY:-python3}"
LOG=results/revision_train4.log
mkdir -p results
echo "=== revision round-4 training $(date) device=$DEVICE ===" | tee -a "$LOG"

run() {
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] START $name  $*" | tee -a "$LOG"
  "$PY" -u src/train_foundation.py --device "$DEVICE" \
      --epochs 50 --lr 1e-4 "$@" --out_dir "results/$name" \
      2>&1 | tee -a "$LOG"
  echo "[$(date +%H:%M:%S)] DONE  $name" | tee -a "$LOG"
}

"$PY" src/build_cache_setb.py 2>&1 | tee -a "$LOG"
export FECG_CACHE=results/cache/records_setb.pkl
run rev_mae_setb --pretrain_datasets NInFEA,NI_FECG_DB,CinC2013b --mode mae --seed 42 --batch_size 24
unset FECG_CACHE

run rev_vicreg_b48 --pretrain_datasets NInFEA,NI_FECG_DB --mode vicreg --seed 42 --batch_size 48

echo "=== round-4 training finished $(date) ===" | tee -a "$LOG"
