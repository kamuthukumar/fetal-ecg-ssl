#!/usr/bin/env bash
# Chain: wait for SSL pretraining -> SimCLR pretraining -> full benchmark -> figures.
# All steps run sequentially so only one MPS job is active at a time.
set -uo pipefail
cd "$(dirname "$0")"
LOG=/tmp/drive_all.log
: > "$LOG"
echo "[drive] waiting for SSL checkpoint..." | tee -a "$LOG"
until [ -f results/foundation/final_model.pt ]; do sleep 15; done
echo "[drive] SSL done at $(date). Starting SimCLR..." | tee -a "$LOG"

python3 -u src/train_foundation.py --epochs 60 --batch_size 24 --lr 1e-4 --device mps \
  --simclr --pretrain_datasets NInFEA,NI_FECG_DB --out_dir results/foundation_simclr \
  >> "$LOG" 2>&1
echo "[drive] SimCLR done at $(date). Running full benchmark..." | tee -a "$LOG"

python3 -u src/run_all_evals.py >> "$LOG" 2>&1
echo "[drive] benchmark done. Generating figures..." | tee -a "$LOG"

python3 -u src/make_real_figures.py >> "$LOG" 2>&1
echo "[drive] ALL DONE at $(date)." | tee -a "$LOG"
