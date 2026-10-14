"""Reviewer 3, R5: does the collapsed (no instance norm) encoder encode amplitude?

For SimCLR trained with and without instance normalisation (matched budget),
we (i) regress each CinC window's log RMS amplitude on its pooled embedding
with record-grouped 5-fold ridge regression and report R^2, (ii) report how
much of the embedding variance the first principal component carries and how
strongly that component correlates with log amplitude, and (iii) measure the
cosine similarity between embeddings of x and alpha*x.

    python src/amplitude_causal.py --device mps
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from eval_qrs import build_labeled_windows, make_encoder, extract_token_features  # noqa: E402

RUNS = {
    "simclr_no_norm": (ROOT / "results" / "rev_simclr_nonorm" / "final_model.pt", "none"),
    "simclr_instance_norm": (ROOT / "results" / "rev_equal_simclr" / "final_model.pt", "instance"),
}
SCALES = (0.25, 0.5, 2.0, 4.0)


def pooled(model, items, device, scale=1.0):
    its = items if scale == 1.0 else [dict(it, signal=it["signal"] * scale) for it in items]
    return np.stack([f.mean(0) for f in extract_token_features(model, its, device)])


def grouped_r2(X, y, groups):
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def unit(z):
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)


def main(device_str):
    import torch
    import eval_qrs
    device = torch.device(device_str)
    items = build_labeled_windows()
    groups = np.array([it["record"] for it in items])
    log_rms = np.log10(np.array([np.sqrt(np.mean(it["signal"] ** 2)) + 1e-12 for it in items]))
    out = {}
    for name, (ckpt, norm) in RUNS.items():
        model = eval_qrs.make_encoder("ssl", str(ckpt), device, input_norm=norm)
        Z = pooled(model, items, device)
        Zc = Z - Z.mean(0)
        u, s, vt = np.linalg.svd(Zc, full_matrices=False)
        pc1 = Zc @ vt[0]
        cos = {str(a): float((unit(Z) * unit(pooled(model, items, device, a))).sum(1).mean())
               for a in SCALES}
        out[name] = dict(
            r2_log_amplitude=grouped_r2(Z, log_rms, groups),
            pc1_variance_fraction=float(s[0] ** 2 / np.sum(s ** 2)),
            pc1_abs_corr_log_amplitude=float(abs(np.corrcoef(pc1, log_rms)[0, 1])),
            cosine_vs_unscaled=cos,
        )
        print(name, out[name])
    dest = ROOT / "results" / "revision" / "amplitude_causal.json"
    dest.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    main(p.parse_args().device)
