"""Embedding-structure evaluation for a (frozen) Fetal-ECG-FM encoder.

Measures, on record-level mean embeddings:
  1. Gestational-age structure on NInFEA  (silhouette over GA bins, permutation p)
  2. Hardware/dataset invariance          (dataset silhouette + linear-probe acc)

Record-level (not window-level) embeddings are used so per-record window counts
cannot inflate clustering. All p-values come from label-permutation nulls.

Usage:
    python src/eval_structure.py --encoder ssl --ckpt results/foundation/final_model.pt
    python src/eval_structure.py --encoder random
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

SEED = 42
POOL = 32


def record_embeddings(model, records, device, win_sec=10.0, fs=1000.0, batch_size=16):
    """Mean encoder embedding per record (averaged over its 10 s windows)."""
    import torch
    from train_foundation import _window_signal
    embs, meta = [], []
    with torch.no_grad():
        for rec in records:
            abd = np.nan_to_num(rec["abdominal"].astype(np.float32))
            wins = _window_signal(abd, fs, win_sec, win_sec)
            if not wins:
                continue
            ws = []
            for i in range(0, len(wins), batch_size):
                x = torch.from_numpy(np.stack(wins[i:i + batch_size])).to(device)
                _, emb = model(x, return_embedding=True)
                ws.append(emb.cpu().numpy())
            embs.append(np.concatenate(ws, axis=0).mean(axis=0))
            meta.append(dict(dataset=rec["dataset"], ga=rec.get("ga_weeks"),
                             subject=rec.get("subject"), record=rec["record"]))
    return np.array(embs), meta


def _silhouette_perm(X, labels, n_perm=1000, rng=None):
    from sklearn.metrics import silhouette_score
    rng = rng or np.random.default_rng(SEED)
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2 or len(labels) < 3:
        return None, None
    obs = float(silhouette_score(X, labels))
    null = np.array([silhouette_score(X, rng.permutation(labels)) for _ in range(n_perm)])
    p = float((null >= obs).mean())
    return obs, p


def evaluate(encoder_kind, ckpt, device_str="mps", out_dir=None):
    import torch
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from data_loader import load_all_records
    from eval_qrs import make_encoder

    device = torch.device(device_str)
    rng = np.random.default_rng(SEED)

    # Datasets with meaningful structure to test.
    records = load_all_records(datasets=["NInFEA", "CinC2013", "NI_FECG_DB", "NIFEA_ARR"])
    model = make_encoder(encoder_kind, ckpt, device)
    X, meta = record_embeddings(model, records, device)
    # Drop any record whose embedding is non-finite (degenerate recording).
    finite = np.all(np.isfinite(X), axis=1)
    if not finite.all():
        print(f"[{encoder_kind}] dropping {int((~finite).sum())} non-finite record embeddings")
        X = X[finite]; meta = [m for m, f in zip(meta, finite) if f]
    scaler = StandardScaler().fit(X)
    scaler.scale_[scaler.scale_ == 0] = 1.0        # guard zero-variance features
    Xs = np.nan_to_num(scaler.transform(X))
    print(f"[{encoder_kind}] {len(meta)} record embeddings, dim {X.shape[1]}")

    results = {"encoder": encoder_kind, "n_records": len(meta)}

    # ---- 1. Gestational-age structure on NInFEA ----
    ninfea = [i for i, m in enumerate(meta) if m["dataset"] == "NInFEA" and m["ga"] is not None]
    if len(ninfea) >= 6:
        Xg = Xs[ninfea]
        ga = np.array([meta[i]["ga"] for i in ninfea])
        # tertile GA bins (early / mid / late within 21-27 wk)
        edges = np.quantile(ga, [1/3, 2/3])
        bins = np.digitize(ga, edges)
        sil, p = _silhouette_perm(Xg, bins, rng=rng)
        results["ga_structure"] = dict(n=len(ninfea), silhouette=sil, p_perm=p,
                                       ga_range=[float(ga.min()), float(ga.max())])

    # ---- 2. Hardware / dataset invariance ----
    ds = np.array([m["dataset"] for m in meta])
    if len(np.unique(ds)) >= 2:
        sil, p = _silhouette_perm(Xs, LabelEncoder().fit_transform(ds), rng=rng)
        # linear-probe dataset classification accuracy (lower = more invariant)
        y = LabelEncoder().fit_transform(ds)
        k = len(np.unique(y))
        acc = float(cross_val_score(LinearSVC(dual="auto", max_iter=5000),
                                    Xs, y, cv=min(5, np.bincount(y).min())).mean()) \
            if np.bincount(y).min() >= 2 else None
        results["hardware"] = dict(silhouette_dataset=sil, p_perm=p,
                                   linear_probe_acc=acc, chance=1.0 / k, n_datasets=k)

    if out_dir:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        (out / f"structure_{encoder_kind}.json").write_text(json.dumps(results, indent=2))
        np.save(out / f"emb_{encoder_kind}.npy", X)
        (out / f"meta_{encoder_kind}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(results, indent=2))
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", choices=["random", "ssl", "simclr"], required=True)
    p.add_argument("--ckpt", type=str, default=str(ROOT / "results" / "foundation" / "final_model.pt"))
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--out_dir", type=str, default=str(ROOT / "results" / "structure"))
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    evaluate(a.encoder, a.ckpt, a.device, a.out_dir)
