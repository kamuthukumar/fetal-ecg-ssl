#!/usr/bin/env bash
# Full JBHI revision compute pipeline. Sequential (single MPS GPU).
set -euo pipefail
cd "$(dirname "$0")"
DEVICE="${DEVICE:-mps}"
PY="${PY:-python3}"
export DEVICE
mkdir -p results/revision
echo "===== JBHI revision pipeline $(date) ====="

echo "[A] JSON-level stats (CPU)"
"$PY" src/revision_experiments.py --skip_slow --device "$DEVICE"

echo "[B] GPU analyses on existing checkpoints"
"$PY" src/revision_experiments.py --device "$DEVICE"

echo "[C] Few-shot end-to-end fine-tuning (5% and 10% labels)"
for frac in 0.05 0.10; do
  for init in random mae; do
    ck=results/foundation_mae/final_model.pt
    tag="${init}_f$(echo $frac | tr -d .)"
    "$PY" src/eval_finetune.py --init "$init" --ckpt "$ck" --device "$DEVICE" \
      --epochs 12 --label_frac "$frac" \
      --out_dir results/revision
    mv -f results/revision/finetune_${init}.json \
          results/revision/finetune_${tag}.json 2>/dev/null || true
  done
done

echo "[D] Equal-budget retrain + ablations"
bash train_revision.sh

echo "[E] Score new checkpoints"
"$PY" src/revision_experiments.py --only_new --device "$DEVICE"

echo "[F] Figures"
"$PY" src/make_real_figures.py
"$PY" src/make_revision_figures.py || true

echo "===== pipeline done $(date) ====="
