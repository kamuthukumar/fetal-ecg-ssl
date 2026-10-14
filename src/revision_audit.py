"""Second-round revision checks.

1. Alternative fold assignments (Reviewer 2, Q2): the frozen-probe benchmark is
   rescored under three extra record-to-fold shuffles for random, SimCLR and MAE.
2. Maternal-QRS coincidence of frozen-probe false positives (Reviewer 3, R7),
   using the default folds. Maternal QRS times are estimated with the same
   energy detector as the classical pipeline (CinC set-a has no maternal labels).

    python src/revision_audit.py --device mps
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from eval_qrs import (FS, WIN_SEC, build_labeled_windows, make_encoder,  # noqa: E402
                      extract_token_features, _fit_head, _best_threshold,
                      _predict_prob_by_record, score_records)

OUT = ROOT / "results" / "revision"
ENCODERS = {
    "random": ("random", None),
    "simclr": ("ssl", ROOT / "results" / "foundation_simclr" / "final_model.pt"),
    "mae": ("ssl", ROOT / "results" / "foundation_mae" / "final_model.pt"),
}
SPLIT_SEEDS = (1, 2, 3)


def folds(records, seed, n_splits=5):
    """Record-grouped folds; seed None reproduces sklearn GroupKFold."""
    if seed is None:
        from sklearn.model_selection import GroupKFold
        return GroupKFold(n_splits=n_splits)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(records)
    return [set(perm[k::n_splits]) for k in range(n_splits)]


def probe(feats, items, groups, ref_by_record, device, split_seed, want_peaks=False):
    d = feats[0].shape[1]
    records = np.array(sorted(set(groups)))
    if split_seed is None:
        gkf = folds(records, None)
        splits = list(gkf.split(np.arange(len(items)), groups=groups))
    else:
        splits = []
        for test_recs in folds(records, split_seed):
            te = np.array([i for i, g in enumerate(groups) if g in test_recs])
            tr = np.array([i for i, g in enumerate(groups) if g not in test_recs])
            splits.append((tr, te))
    per_rec, peaks = {}, {}
    for tr, te in splits:
        Xtr = np.concatenate([feats[i] for i in tr], axis=0)
        ytr = np.concatenate([items[i]["token_label"] for i in tr], axis=0)
        head = _fit_head(Xtr, ytr, d, device)
        thr = _best_threshold(head, feats, items, tr, groups, device)
        prob = _predict_prob_by_record(head, feats, items, te, device)
        ref = {r: ref_by_record[r] for r in prob}
        if want_peaks:
            f1, pk = score_records(prob, ref, thr, return_peaks=True)
            peaks.update(pk)
        else:
            f1 = score_records(prob, ref, thr)
        per_rec.update(f1)
    return per_rec, peaks


def maternal_times():
    from fecg_extraction import detect_qrs, _refine_to_local_extreme
    from data_loader import load_cinc2013
    out = {}
    for i in range(1, 76):
        rec = load_cinc2013(f"a{i:02d}")
        if rec is None:
            continue
        sig, fs = rec["abdominal"], rec["fs"]
        ch = int(np.argmax(np.var(sig, axis=1)))
        mat, _ = detect_qrs(sig[ch], fs, 45, 110, thr_scale=0.6)
        mat = _refine_to_local_extreme(sig[ch], mat, fs)
        out[rec["record"]] = np.asarray(mat, float) * (FS / fs)
    return out


def fp_on_maternal(peaks, ref_by_record, mat, tol_ms=50.0):
    tol = tol_ms / 1000.0 * FS
    n_fp = n_hit = 0
    n_rec = 0
    for rec, det in peaks.items():
        ref = np.asarray(ref_by_record[rec], float)
        m = mat.get(rec)
        if m is None or len(m) == 0:
            continue
        used = np.zeros(len(ref), bool)
        fps = []
        for x in det:
            dist = np.abs(ref - x)
            dist[used] = np.inf
            j = int(np.argmin(dist)) if len(ref) else 0
            if len(ref) and dist[j] <= tol:
                used[j] = True
            else:
                fps.append(x)
        n_rec += 1
        n_fp += len(fps)
        n_hit += sum(np.min(np.abs(m - x)) <= tol for x in fps)
    return dict(n_records=n_rec, n_fp=int(n_fp), n_fp_on_maternal=int(n_hit),
                frac=float(n_hit / max(n_fp, 1)))


def main(device_str):
    import torch
    from scipy.stats import wilcoxon
    device = torch.device(device_str)
    items = build_labeled_windows()
    groups = np.array([it["record"] for it in items])
    ref_by_record = {}
    for it in items:
        s0 = it["win"] * int(WIN_SEC * FS)
        ref_by_record.setdefault(it["record"], []).extend((it["beats_in_win"] + s0).tolist())
    mat = maternal_times()

    res = {"splits": {}, "maternal_fp": {}}
    scores = {}
    for name, (kind, ckpt) in ENCODERS.items():
        model = make_encoder(kind, None if ckpt is None else str(ckpt), device)
        feats = extract_token_features(model, items, device)
        f1, pk = probe(feats, items, groups, ref_by_record, device, None, want_peaks=True)
        res["maternal_fp"][name] = fp_on_maternal(pk, ref_by_record, mat)
        print(name, "maternal FP", res["maternal_fp"][name])
        scores[name] = {"default": f1}
        for s in SPLIT_SEEDS:
            scores[name][s], _ = probe(feats, items, groups, ref_by_record, device, s)
            print(name, "split", s, round(np.mean(list(scores[name][s].values())), 3))

    keys = sorted(scores["random"]["default"])
    for split in ["default", *SPLIT_SEEDS]:
        v = {n: np.array([scores[n][split][k] for k in keys]) for n in scores}
        row = {n: round(float(v[n].mean()), 4) for n in v}
        for a, b in [("mae", "random"), ("simclr", "random"), ("mae", "simclr")]:
            x = v[a] - v[b]
            row[f"{a}-{b}"] = dict(delta=round(float(x.mean()), 4),
                                   dz=round(float(x.mean() / x.std(ddof=1)), 2),
                                   p=float(f"{wilcoxon(x).pvalue:.2g}"))
        res["splits"][str(split)] = row
        print(split, row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit_splits_maternal.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    main(p.parse_args().device)
