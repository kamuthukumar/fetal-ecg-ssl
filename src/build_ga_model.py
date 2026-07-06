"""Real gestational-age regression on NInFEA extracted fetal ECG.

Steps:
  1. For each of the 60 records: load 24 abdominal channels, downsample 2048->256 Hz,
     pick the 4 highest-quality channels, extract fetal QRS (unsupervised), compute
     interpretable HRV + morphology features.
  2. Regress GA (weeks) from features with SUBJECT-grouped cross-validation
     (LOSO by participant) so no subject appears in both train and test.
  3. Report honest MAE (weeks AND days) + baseline (predict-mean) for context.

No ground-truth GA is used at any point during feature extraction.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import decimate
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fecg_extraction import extract_fetal          # noqa: E402
from features import compute_features, FEATURE_NAMES  # noqa: E402
from ninfea_labels import GA_WEEKS, SUBJECT          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ninfea"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)
SEED = 42
np.random.seed(SEED)

TARGET_FS = 256


def build_feature_table():
    rows = []
    for rid in sorted(GA_WEEKS):
        hea = DATA / f"{rid}.hea"
        if not hea.exists():
            continue
        rec = wfdb.rdrecord(str(DATA / str(rid)))
        names = rec.sig_name
        abd_idx = [i for i, n in enumerate(names) if n.startswith("uni_abd")]
        sig = rec.p_signal[:, abd_idx].T           # (24, N) at 2048 Hz
        fs = rec.fs
        q = max(1, int(round(fs / TARGET_FS)))
        sig = np.array([decimate(c, q, ftype="fir") for c in sig])
        fs = fs / q
        # pick 4 channels with the highest fetal-band kurtosis (quality proxy)
        from scipy.stats import kurtosis
        from fecg_extraction import bandpass
        kurt = [kurtosis(bandpass(c, fs, 10, min(70, fs/2*0.95))) for c in sig]
        best4 = np.argsort(kurt)[-4:]
        out = extract_fetal(sig[best4], fs, fhr=(100, 180), mhr=(45, 110))
        feats = compute_features(out.fetal_peaks, out.residual, fs)
        row = dict(record=rid, subject=SUBJECT[rid], ga_weeks=GA_WEEKS[rid],
                   n_beats=len(out.fetal_peaks), **feats)
        rows.append(row)
        print(f"rec {rid:2d}: GA={GA_WEEKS[rid]:.2f}  beats={len(out.fetal_peaks):3d}  "
              f"FHR={feats['mean_fhr'] if not np.isnan(feats['mean_fhr']) else float('nan'):.0f}")
    return pd.DataFrame(rows)


def evaluate(df):
    X = df[FEATURE_NAMES].values
    y = df["ga_weeks"].values
    groups = df["subject"].values
    logo = LeaveOneGroupOut()

    models = {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "ridge": Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc", StandardScaler()),
                           ("m", Ridge(alpha=10.0))]),
        "random_forest": Pipeline([("imp", SimpleImputer(strategy="median")),
                                   ("m", RandomForestRegressor(
                                       n_estimators=400, max_depth=4,
                                       min_samples_leaf=3, random_state=SEED))]),
    }
    results = {}
    preds_out = {}
    for name, mdl in models.items():
        pred = cross_val_predict(mdl, X, y, groups=groups, cv=logo)
        mae_w = float(np.mean(np.abs(pred - y)))
        rmse_w = float(np.sqrt(np.mean((pred - y) ** 2)))
        r = float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) > 0 else float("nan")
        results[name] = dict(mae_weeks=mae_w, mae_days=mae_w * 7,
                             rmse_weeks=rmse_w, pearson_r=r)
        preds_out[name] = pred.tolist()
        print(f"{name:15s}: MAE={mae_w:.2f} wk ({mae_w*7:.1f} d)  "
              f"RMSE={rmse_w:.2f} wk  r={r:.2f}")
    return results, preds_out


if __name__ == "__main__":
    df = build_feature_table()
    df.to_csv(RES / "ninfea_features.csv", index=False)
    results, preds = evaluate(df)
    df_out = df[["record", "subject", "ga_weeks"]].copy()
    for name, p in preds.items():
        df_out[f"pred_{name}"] = p
    df_out.to_csv(RES / "ninfea_ga_predictions.csv", index=False)
    (RES / "ga_model_summary.json").write_text(json.dumps({
        "n_records": int(len(df)),
        "n_subjects": int(df["subject"].nunique()),
        "ga_range_weeks": [float(df.ga_weeks.min()), float(df.ga_weeks.max())],
        "ga_std_weeks": float(df.ga_weeks.std()),
        "cv": "LeaveOneSubjectOut (grouped by participant)",
        "results": results,
    }, indent=2))
    print("\nSaved features + predictions + summary to results/")
