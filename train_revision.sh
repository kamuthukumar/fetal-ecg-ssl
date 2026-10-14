#!/usr/bin/env bash
# Equal-budget re-training and ablations requested by JBHI reviewers.
# Sequential on MPS; ~20 min per run. Logs to results/revision_train.log
set -euo pipefail
cd "$(dirname "$0")"
DEVICE="${DEVICE:-mps}"
DS=NInFEA,NI_FECG_DB
PY="${PY:-python3}"
LOG=results/revision_train.log
mkdir -p results
echo "=== revision training $(date) device=$DEVICE ===" | tee -a "$LOG"

run() {
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] START $name  $*" | tee -a "$LOG"
  "$PY" src/train_foundation.py --device "$DEVICE" --pretrain_datasets "$DS" \
      --batch_size 24 --epochs 50 --lr 1e-4 "$@" --out_dir "results/$name" \
      2>&1 | tee -a "$LOG"
  echo "[$(date +%H:%M:%S)] DONE  $name" | tee -a "$LOG"
}

# 1. Equal training budget (the comparison the paper's conclusion rests on)
run rev_equal_mae     --mode mae        --seed 42
run rev_equal_simclr  --mode simclr     --seed 42
run rev_equal_vicreg  --mode vicreg     --seed 42

# 2. Extra MAE seeds (variability)
run rev_mae_s7        --mode mae        --seed 7
run rev_mae_s123      --mode mae        --seed 123

# 3. Mask-ratio ablation
run rev_mae_m25       --mode mae        --seed 42 --mask_ratio 0.25
run rev_mae_m75       --mode mae        --seed 42 --mask_ratio 0.75

# 4. Physiological contrastive (Eq. 1 + physics augs), actually trained
run rev_physics       --mode contrastive --seed 42

# 5. Collapse mechanism: SimCLR without instance normalisation
run rev_simclr_nonorm --mode simclr     --seed 42 --input_norm none

# 6. Architecture: CNN-only masked modelling (no Transformer)
run rev_mae_cnn       --mode mae        --seed 42 --no_transformer

echo "=== all training finished $(date) ===" | tee -a "$LOG"
