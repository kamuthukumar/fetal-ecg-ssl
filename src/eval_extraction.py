"""Evaluate fetal QRS detection on datasets with genuine reference annotations:
   ADfECGDB (direct scalp fetal ECG -> beat annotations) and
   CinC 2013 set-a (.fqrs expert reference).

Outputs a per-record CSV and a summary to code/results/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from edf_io import read_edf                       # noqa: E402
from fecg_extraction import extract_fetal, match_f1  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)


def eval_adfecgdb():
    rows = []
    for edf in sorted((DATA / "adfecgdb").glob("*.edf")):
        rec = read_edf(edf)
        fs = rec.fs
        abd = np.array([rec.signals[c] for c in rec.labels if c.startswith("Abdomen")])
        ref = np.round(rec.ann_times * fs).astype(int)   # fetal beats from scalp channel
        out = extract_fetal(abd, fs)
        m = match_f1(out.fetal_peaks, ref, fs, tol_ms=50)
        rows.append(dict(dataset="ADfECGDB", record=edf.stem, fs=fs,
                         n_ref=len(ref), n_det=len(out.fetal_peaks),
                         snr_db=out.snr_improvement_db, **m))
        print(f"ADfECGDB {edf.stem}: F1={m['f1']:.3f} Se={m['se']:.3f} "
              f"PPV={m['ppv']:.3f} (ref={len(ref)}, det={len(out.fetal_peaks)})")
    return rows


def eval_cinc():
    rows = []
    recs = sorted(p.stem for p in (DATA / "cinc2013").glob("*.dat"))
    for r in recs:
        base = str(DATA / "cinc2013" / r)
        rec = wfdb.rdrecord(base)
        fs = rec.fs
        abd = rec.p_signal.T                       # 4 abdominal channels
        try:
            ann = wfdb.rdann(base, "fqrs")
        except Exception:
            continue
        ref = ann.sample
        if len(ref) < 5:
            continue
        out = extract_fetal(abd, fs)
        m = match_f1(out.fetal_peaks, ref, fs, tol_ms=50)
        rows.append(dict(dataset="CinC2013", record=r, fs=fs,
                         n_ref=len(ref), n_det=len(out.fetal_peaks),
                         snr_db=out.snr_improvement_db, **m))
        print(f"CinC {r}: F1={m['f1']:.3f} Se={m['se']:.3f} PPV={m['ppv']:.3f} "
              f"(ref={len(ref)}, det={len(out.fetal_peaks)})")
    return rows


def summarize(df):
    out = {}
    for ds, g in df.groupby("dataset"):
        f1 = g["f1"].dropna()
        out[ds] = dict(
            n_records=int(len(g)),
            f1_mean=float(f1.mean()), f1_std=float(f1.std()),
            f1_median=float(f1.median()),
            se_mean=float(g["se"].dropna().mean()),
            ppv_mean=float(g["ppv"].dropna().mean()),
            snr_db_mean=float(g["snr_db"].mean()),
        )
    return out


if __name__ == "__main__":
    rows = []
    rows += eval_adfecgdb()
    rows += eval_cinc()
    df = pd.DataFrame(rows)
    df.to_csv(RES / "extraction_per_record.csv", index=False)
    summary = summarize(df)
    (RES / "extraction_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2))
