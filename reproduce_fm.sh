#!/usr/bin/env bash
# Reproduce the self-supervised fetal-ECG results end-to-end from open data.
# Every number and figure in the manuscript is produced by this script.
#
# Device: defaults to Apple-silicon "mps". Override for CUDA/CPU, e.g.:
#     DEVICE=cuda bash reproduce_fm.sh
#     DEVICE=cpu  bash reproduce_fm.sh
set -euo pipefail
cd "$(dirname "$0")"
DEVICE="${DEVICE:-mps}"

echo "[1/7] Install dependencies"
python3 -m pip install --quiet -r requirements.txt

echo "[2/7] Download open-access datasets (NInFEA, CinC 2013, NI-FECG DB)"
python3 src/download_data.py ninfea cinc2013 nifecgdb

echo "[3/7] Build preprocessing cache (4-channel, 1 kHz, filtered)"
python3 src/build_cache.py

echo "[4/7] Pre-train the three self-supervised objectives (CinC 2013 held out)"
DS=NInFEA,NI_FECG_DB
python3 src/train_foundation.py --mode mae    --epochs 50 --batch_size 24 --lr 2e-4 \
  --device "$DEVICE" --pretrain_datasets "$DS" --out_dir results/foundation_mae
python3 src/train_foundation.py --mode vicreg --epochs 40 --batch_size 24 --lr 1e-4 \
  --device "$DEVICE" --pretrain_datasets "$DS" --out_dir results/foundation_vicreg
python3 src/train_foundation.py --mode simclr --epochs 40 --batch_size 24 --lr 1e-4 \
  --device "$DEVICE" --pretrain_datasets "$DS" --out_dir results/foundation_simclr

echo "[5/7] Full benchmark: QRS + structure + few-shot + effective rank (random/simclr/vicreg/mae)"
python3 src/run_all_evals.py

echo "[6/7] End-to-end fine-tuning comparison (random vs mae vs simclr initialisation)"
for init in random mae simclr; do
  ck=results/foundation_mae/final_model.pt
  [ "$init" = simclr ] && ck=results/foundation_simclr/final_model.pt
  python3 src/eval_finetune.py --init "$init" --ckpt "$ck" --device "$DEVICE" --epochs 12
done

echo "[7/7] Figures from measured results"
python3 src/make_real_figures.py

echo "Done. See results/benchmark/summary.json and results/real_figs/*.png"
