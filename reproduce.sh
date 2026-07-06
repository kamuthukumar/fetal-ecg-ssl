#!/usr/bin/env bash
# Reproduce every number and figure in the paper from the raw open-access data.
# Usage:  bash reproduce.sh
set -euo pipefail

cd "$(dirname "$0")"

echo "[1/6] Creating virtual environment"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo "[2/6] Downloading the three PhysioNet datasets (~0.5 GB)"
python src/download_data.py            # ADfECGDB, NInFEA, CinC 2013
# If wfdb.dl_database is rate-limited, see README for the direct-curl fallback.

echo "[3/6] Evaluating fetal QRS extraction (ADfECGDB + CinC 2013)"
python src/eval_extraction.py

echo "[4/6] Building gestational-age model on NInFEA (subject-grouped LOSO)"
python src/build_ga_model.py

echo "[5/6] Hardening the GA result (permutation null, ablations, quality gate)"
python src/robust_ga_analysis.py
python src/leakage_diagnostic.py

echo "[6/6] Generating figures"
python src/make_figures.py

echo "Done. See results/*.json, results/*.csv and figures/*.png"
