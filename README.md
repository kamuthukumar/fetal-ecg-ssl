# Reproducible fetal-ECG benchmark and gestational-age analysis

Code and evaluation pipeline for the study *"What non-invasive fetal ECG can and
cannot tell us about gestational age: a reproducible multi-cohort benchmark."*

Everything here runs on three **open-access PhysioNet datasets** and reproduces
every number and figure in the manuscript from the raw signals. No proprietary
data, no pretrained weights, fixed random seed (42).

## What this repository shows

1. **A reproducible non-invasive fetal QRS detection benchmark** on two datasets
   with genuine reference annotations:
   - **ADfECGDB** (direct fetal scalp electrode): F1 = **0.91** (median 0.93).
   - **PhysioNet/CinC 2013 set-a** (75 records, expert `.fqrs`): F1 = **0.68**
     mean / **0.82** median.
   The method is a fully classical, transparent pipeline (band-pass, adaptive
   maternal-beat template subtraction, energy-envelope fetal QRS detection with
   RR-consistency cleanup). No ground truth is used during extraction.

2. **A rigorous negative result on gestational-age (GA) estimation.** Using
   interpretable HRV + morphology features from the extracted fetal ECG on
   **NInFEA** (60 records, 39 subjects, 21–27 wk), GA is **not** predictable
   above the mean-only baseline under **subject-grouped** cross-validation:
   - MAE = **1.8 weeks** (~12.6 days) vs mean-baseline MAE = 1.7–1.8 weeks.
   - Pearson r ≈ **0** across five different estimators.
   - Subject-level **permutation test**: observed MAE sits inside the
     label-shuffled null (p ≈ 0.84).
   We also demonstrate how **record-level** cross-validation leaks identical GA
   labels between recordings of the same mother — the most likely origin of
   previously reported sub-day accuracies.

Per-record GA labels for NInFEA are transcribed from the dataset's companion
paper (Sulas et al., *Scientific Data* 8:30, 2021, Table 3) in
`src/ninfea_labels.py`.

## Quick start

```bash
bash reproduce.sh
```

This creates a virtual environment, downloads the data (~0.5 GB), and writes all
results to `results/` and figures to `figures/`.

### Manual steps

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/download_data.py          # ADfECGDB, NInFEA, CinC 2013
python src/eval_extraction.py        # -> results/extraction_*.{csv,json}
python src/build_ga_model.py         # -> results/ninfea_*.csv, ga_model_summary.json
python src/robust_ga_analysis.py     # -> results/robust_ga_report.json
python src/leakage_diagnostic.py     # subject-grouped vs record-level CV
python src/make_figures.py           # -> figures/*.png
```

If PhysioNet rate-limits bulk downloads, re-run `download_data.py`; it skips
files already present. Verify integrity against each dataset's `SHA256SUMS.txt`.

## Layout

| Path | Purpose |
|------|---------|
| `src/download_data.py` | fetch the three PhysioNet datasets |
| `src/edf_io.py` | dependency-free EDF/EDF+ reader (ADfECGDB + fetal beat annotations) |
| `src/fecg_extraction.py` | maternal cancellation + fetal QRS detection + F1 scoring |
| `src/features.py` | HRV and morphology features from the extracted fetal ECG |
| `src/ninfea_labels.py` | per-record NInFEA gestational-age labels (companion paper) |
| `src/build_ga_model.py` | subject-grouped GA regression |
| `src/robust_ga_analysis.py` | permutation null, ablations, quality gating |
| `src/leakage_diagnostic.py` | subject-grouped vs leaky record-level CV |
| `src/make_figures.py` | manuscript figures |

## Data availability

- ADfECGDB — https://physionet.org/content/adfecgdb/1.0.0/
- NInFEA — https://physionet.org/content/ninfea/1.0.0/
- PhysioNet/CinC 2013 — https://physionet.org/content/challenge-2013/1.0.0/

## License

MIT (see `LICENSE`). PhysioNet datasets carry their own licenses.
