"""Show the effect of subject leakage on the NInFEA GA result.

Compares:
  (A) LeaveOneSubjectOut (grouped) -- correct, no leakage
  (B) record-level 5-fold KFold    -- leaky (same mother's 2-3 recordings can
      appear in both train and test, sharing the identical GA label)

This isolates whether the previously reported high r / low MAE could arise
purely from evaluation leakage rather than genuine developmental signal.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, KFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURE_NAMES  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
SEED = 42


def rf():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", RandomForestRegressor(n_estimators=400, max_depth=4,
                                                 min_samples_leaf=3, random_state=SEED))])


def metrics(pred, y):
    mae = np.mean(np.abs(pred - y))
    r = np.corrcoef(pred, y)[0, 1] if np.std(pred) > 0 else np.nan
    return mae, mae * 7, r


if __name__ == "__main__":
    df = pd.read_csv(RES / "ninfea_features.csv")
    X = df[FEATURE_NAMES].values
    y = df["ga_weeks"].values
    groups = df["subject"].values

    # (A) correct: leave one subject out
    predA = cross_val_predict(rf(), X, y, groups=groups, cv=LeaveOneGroupOut())
    maeA, dA, rA = metrics(predA, y)

    # (B) leaky: record-level 5-fold (ignores subject grouping)
    predB = cross_val_predict(rf(), X, y, cv=KFold(5, shuffle=True, random_state=SEED))
    maeB, dB, rB = metrics(predB, y)

    print("Correct (LeaveOneSubjectOut) : "
          f"MAE={maeA:.2f} wk ({dA:.1f} d)  r={rA:+.2f}")
    print("Leaky   (record-level KFold) : "
          f"MAE={maeB:.2f} wk ({dB:.1f} d)  r={rB:+.2f}")
    print()
    n_multi = (df.groupby("subject").size() > 1).sum()
    print(f"{n_multi} of {df.subject.nunique()} subjects contribute >1 recording "
          f"(the records that leak identical GA labels under record-level CV).")
