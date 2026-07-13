#!/usr/bin/env bash
# Pre-train the principled fixes (MAE, VICReg) + a SimCLR control on the current
# (instance-normalised) architecture, then run the full benchmark and figures.
# Sequential so only one MPS job runs at a time.
set -uo pipefail
cd "$(dirname "$0")"
LOG=/tmp/drive_methods.log
: > "$LOG"
DS="NInFEA,NI_FECG_DB"

echo "[drive] MAE pretraining ($(date))" | tee -a "$LOG"
python3 -u src/train_foundation.py --mode mae --epochs 50 --batch_size 24 --lr 2e-4 \
  --device mps --pretrain_datasets "$DS" --out_dir results/foundation_mae >> "$LOG" 2>&1

echo "[drive] VICReg pretraining ($(date))" | tee -a "$LOG"
python3 -u src/train_foundation.py --mode vicreg --epochs 40 --batch_size 24 --lr 1e-4 \
  --device mps --pretrain_datasets "$DS" --out_dir results/foundation_vicreg >> "$LOG" 2>&1

echo "[drive] SimCLR control ($(date))" | tee -a "$LOG"
python3 -u src/train_foundation.py --mode simclr --epochs 40 --batch_size 24 --lr 1e-4 \
  --device mps --pretrain_datasets "$DS" --out_dir results/foundation_simclr >> "$LOG" 2>&1

echo "[drive] full benchmark ($(date))" | tee -a "$LOG"
python3 -u src/run_all_evals.py >> "$LOG" 2>&1

echo "[drive] figures ($(date))" | tee -a "$LOG"
python3 -u src/make_real_figures.py >> "$LOG" 2>&1

echo "[drive] ALL DONE ($(date))" | tee -a "$LOG"
