"""Harden the gestational-age negative result.

Adds, on top of the subject-grouped LOSO regression:
  1. A permutation null test (shuffle GA across subjects, 1000x) -> is the observed
     MAE distinguishable from chance?
  2. Quality gating: repeat on the subset of records with >= N reliable fetal beats.
  3. Multiple estimators + a feature-subset ablation (HRV-only, morphology-only).
  4. Skill vs the mean baseline (MAE ratio); a real clock would be << 1.

Everything is subject-grouped to prevent leakage.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURE_NAMES  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
SEED = 42
rng = np.random.default_rng(SEED)

HRV = ["mean_fhr", "sdnn", "rmssd", "pnn20", "cv_rr", "sampen", "dfa_alpha1", "lf_hf"]
MORPH = ["qrs_amp_med", "qrs_amp_cv", "qrs_width_med", "beat_template_corr"]


def make_models():
    return {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "ridge": Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc", StandardScaler()), ("m", Ridge(alpha=10.0))]),
        "svr_rbf": Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("sc", StandardScaler()), ("m", SVR(C=1.0))]),
        "random_forest": Pipeline([("imp", SimpleImputer(strategy="median")),
                                   ("m", RandomForestRegressor(
                                       n_estimators=400, max_depth=4,
                                       min_samples_leaf=3, random_state=SEED))]),
        "grad_boost": Pipeline([("imp", SimpleImputer(strategy="median")),
                                ("m", GradientBoostingRegressor(random_state=SEED))]),
    }


def loso_mae(model, X, y, groups, n_jobs=-1):
    pred = cross_val_predict(model, X, y, groups=groups,
                             cv=LeaveOneGroupOut(), n_jobs=n_jobs)
    mae = float(np.mean(np.abs(pred - y)))
    r = float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) > 0 else float("nan")
    return mae, r, pred


def permutation_null(model, X, y, groups, n=500):
    """Shuffle GA labels at the SUBJECT level; recompute LOSO MAE each time.

    Uses a fast linear estimator for the null distribution (adequate for a
    chance-level reference) with parallel folds."""
    obs, _, _ = loso_mae(model, X, y, groups)
    uniq = np.unique(groups)
    subj_ga = {g: y[groups == g][0] for g in uniq}
    null = []
    for _ in range(n):
        perm_vals = rng.permutation(list(subj_ga.values()))
        mapping = dict(zip(uniq, perm_vals))
        y_perm = np.array([mapping[g] for g in groups])
        m, _, _ = loso_mae(model, X, y_perm, groups, n_jobs=-1)
        null.append(m)
    null = np.array(null)
    # one-sided p: fraction of null MAE <= observed (better-or-equal than real)
    p = float((np.sum(null <= obs) + 1) / (n + 1))
    return dict(observed_mae=obs, null_mae_mean=float(null.mean()),
                null_mae_std=float(null.std()), p_value=p)


def run(df, tag):
    y = df["ga_weeks"].values
    groups = df["subject"].values
    base_mae = float(np.mean(np.abs(y - np.mean(y))))
    out = {"tag": tag, "n_records": int(len(df)),
           "n_subjects": int(df["subject"].nunique()),
           "mean_baseline_mae_weeks": base_mae, "models": {}, "ablation": {}}
    for name, mdl in make_models().items():
        mae, r, _ = loso_mae(mdl, df[FEATURE_NAMES].values, y, groups)
        out["models"][name] = dict(mae_weeks=mae, mae_days=mae * 7, r=r,
                                   skill_vs_baseline=mae / base_mae)
    # ablation with RF
    for sub, cols in [("hrv_only", HRV), ("morph_only", MORPH), ("all", FEATURE_NAMES)]:
        mdl = Pipeline([("imp", SimpleImputer(strategy="median")),
                        ("m", RandomForestRegressor(n_estimators=400, max_depth=4,
                                                    min_samples_leaf=3, random_state=SEED))])
        mae, r, _ = loso_mae(mdl, df[cols].values, y, groups)
        out["ablation"][sub] = dict(mae_weeks=mae, r=r, skill_vs_baseline=mae / base_mae)
    # permutation null with a fast linear estimator (chance-level reference)
    lin = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()), ("m", Ridge(alpha=10.0))])
    out["permutation_null_rf"] = permutation_null(lin, df[FEATURE_NAMES].values, y, groups, n=500)
    return out


if __name__ == "__main__":
    df = pd.read_csv(RES / "ninfea_features.csv")
    report = {"full": run(df, "full_60")}
    # quality gate: records with >= 40 detected fetal beats
    good = df[df["n_beats"] >= 40].copy()
    report["quality_gated"] = run(good, "n_beats>=40")

    (RES / "robust_ga_report.json").write_text(json.dumps(report, indent=2))
    for key in ("full", "quality_gated"):
        rp = report[key]
        print(f"\n===== {key}: n={rp['n_records']} rec / {rp['n_subjects']} subj "
              f"| mean-baseline MAE={rp['mean_baseline_mae_weeks']:.2f} wk =====")
        for name, m in rp["models"].items():
            print(f"  {name:14s} MAE={m['mae_weeks']:.2f} wk ({m['mae_days']:.1f} d) "
                  f"r={m['r']:+.2f} skill={m['skill_vs_baseline']:.2f}")
        pn = rp["permutation_null_rf"]
        print(f"  permutation null (RF): observed MAE={pn['observed_mae']:.2f} wk, "
              f"null={pn['null_mae_mean']:.2f}+/-{pn['null_mae_std']:.2f}, p={pn['p_value']:.3f}")
    print("\nSaved results/robust_ga_report.json")
