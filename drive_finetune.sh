#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
LOG=/tmp/drive_finetune.log; : > "$LOG"
for init in random mae simclr; do
  case $init in
    mae) CK=results/foundation_mae/final_model.pt;;
    simclr) CK=results/foundation_simclr/final_model.pt;;
    *) CK=results/foundation_mae/final_model.pt;;
  esac
  echo "[drive] fine-tune init=$init ($(date))" | tee -a "$LOG"
  python3 -u src/eval_finetune.py --init "$init" --ckpt "$CK" --device mps \
    --epochs 12 --out_dir results/benchmark >> "$LOG" 2>&1
done
echo "[drive] ALL DONE ($(date))" | tee -a "$LOG"
