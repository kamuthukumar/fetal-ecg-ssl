"""Second-round revision analyses on CinC 2013 set-a.

  quality   : per-record F1 stratified by abdominal amplitude and noise level,
              plus additive-noise robustness of the frozen probe
              (Reviewer 1, Concern 2).
  rhythm    : per-record F1 stratified by reference RR irregularity and by
              missing-beat gaps (Reviewer 1, Concern 3).
  drift     : linear CKA between token features before and after end-to-end
              fine-tuning, and between the two fine-tuned models
              (Reviewer 2, Q6).

    python src/revision_round2.py --which quality rhythm drift
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
FS = 1000.0
WIN_SEC = 10.0
OUT = ROOT / "results" / "revision"
MAE_CKPT = ROOT / "results" / "foundation_mae" / "final_model.pt"


def _per_record_sets():
    bench = ROOT / "results" / "benchmark"
    db = json.loads((OUT / "direct_baselines.json").read_text())["results"]
    return {
        "random_probe": json.loads((bench / "qrs_random.json").read_text())["per_record"],
        "masked_probe": json.loads((bench / "qrs_mae.json").read_text())["per_record"],
        "classical_ts": json.loads((OUT / "classical_baseline.json").read_text())["classical"]["per_record"],
        "unet": db["unet"]["per_record"],
    }


def _strata(values, recs, sets, labels=("low", "mid", "high")):
    v = np.array([values[r] for r in recs])
    order = np.argsort(v, kind="stable")
    bins = np.empty(len(v), int)
    for b, chunk in enumerate(np.array_split(order, 3)):
        bins[chunk] = b
    out = {}
    for b, lab in enumerate(labels):
        rs = [r for r, k in zip(recs, bins) if k == b]
        out[lab] = dict(n=len(rs), range=[float(v[bins == b].min()), float(v[bins == b].max())],
                        f1={k: float(np.mean([s[r] for r in rs])) for k, s in sets.items()})
    return out


def _paired(a, b):
    from scipy.stats import wilcoxon
    a, b = np.asarray(a), np.asarray(b)
    return dict(delta=float((b - a).mean()), p=float(wilcoxon(a, b).pvalue))


# --------------------------------------------------------------------------- #
def quality(device_str="mps"):
    from data_loader import load_cinc2013
    from fecg_extraction import bandpass
    sets = _per_record_sets()
    amp, noise = {}, {}
    for i in range(1, 76):
        rec = load_cinc2013(f"a{i:02d}")
        x = np.nan_to_num(rec["abdominal"].astype(np.float64))
        amp[rec["record"]] = float(np.median(np.sqrt((x ** 2).mean(axis=1))))
        hf = np.array([bandpass(ch, FS, 45, 90) for ch in x])
        lf = np.array([bandpass(ch, FS, 1, 45) for ch in x])
        noise[rec["record"]] = float(np.median((hf ** 2).mean(axis=1) / ((lf ** 2).mean(axis=1) + 1e-12)))
    recs = sorted(sets["masked_probe"])
    res = dict(amplitude=_strata(amp, recs, sets),
               noise=_strata(noise, recs, sets),
               noise_injection=noise_injection(device_str))
    lo = sorted(recs, key=lambda r: amp[r])[:25]
    res["amplitude"]["low"]["masked_vs_random"] = _paired(
        [sets["random_probe"][r] for r in lo], [sets["masked_probe"][r] for r in lo])
    hi = sorted(recs, key=lambda r: noise[r])[-25:]
    res["noise"]["high"]["masked_vs_random"] = _paired(
        [sets["random_probe"][r] for r in hi], [sets["masked_probe"][r] for r in hi])
    return res


def noise_injection(device_str, snrs_db=(12, 6, 0)):
    """Train the probe on clean folds; test on held-out windows corrupted by
    white noise plus 0.3 Hz baseline wander at a given SNR."""
    import torch
    from sklearn.model_selection import GroupKFold
    from eval_qrs import (build_labeled_windows, make_encoder, extract_token_features,
                          _fit_head, _best_threshold, _predict_prob_by_record, score_records)
    device = torch.device(device_str)
    items = build_labeled_windows()
    groups = np.array([it["record"] for it in items])
    ref = {}
    for it in items:
        s0 = it["win"] * int(WIN_SEC * FS)
        ref.setdefault(it["record"], []).extend((it["beats_in_win"] + s0).tolist())
    rng = np.random.default_rng(SEED)
    t = np.arange(int(WIN_SEC * FS)) / FS
    noisy = {}
    for snr in snrs_db:
        its = []
        for it in items:
            x = it["signal"].astype(np.float64)
            p = (x ** 2).mean(axis=1, keepdims=True)
            n = rng.standard_normal(x.shape) + 2.0 * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, 2 * np.pi))
            n *= np.sqrt(p / ((n ** 2).mean(axis=1, keepdims=True) * 10 ** (snr / 10)))
            its.append(dict(it, signal=(x + n).astype(np.float32)))
        noisy[snr] = its
    out = {}
    for kind, ckpt in (("random", None), ("ssl", str(MAE_CKPT))):
        model = make_encoder(kind, ckpt, device)
        clean = extract_token_features(model, items, device)
        feats_noisy = {s: extract_token_features(model, noisy[s], device) for s in snrs_db}
        per = {"clean": {}, **{f"snr{s}": {} for s in snrs_db}}
        for tr, te in GroupKFold(n_splits=5).split(np.arange(len(items)), groups=groups):
            Xtr = np.concatenate([clean[i] for i in tr]); ytr = np.concatenate([items[i]["token_label"] for i in tr])
            head = _fit_head(Xtr, ytr, Xtr.shape[1], device)
            thr = _best_threshold(head, clean, items, tr, groups, device)
            for key, f in [("clean", clean)] + [(f"snr{s}", feats_noisy[s]) for s in snrs_db]:
                pb = _predict_prob_by_record(head, f, items, te, device)
                per[key].update(score_records(pb, {r: ref[r] for r in pb}, thr))
        out["masked" if kind == "ssl" else "random"] = per
    summ = {}
    for key in out["random"]:
        recs = sorted(out["random"][key])
        a = [out["random"][key][r] for r in recs]; b = [out["masked"][key][r] for r in recs]
        summ[key] = dict(random=float(np.mean(a)), masked=float(np.mean(b)), **_paired(a, b))
        print(f"  noise {key}: random {np.mean(a):.3f}  masked {np.mean(b):.3f}", flush=True)
    return summ


# --------------------------------------------------------------------------- #
def rhythm():
    from data_loader import load_cinc2013
    sets = _per_record_sets()
    irr, gaps = {}, {}
    for i in range(1, 76):
        rec = load_cinc2013(f"a{i:02d}")
        b = np.sort(np.asarray(rec["fetal_beats"], float))
        rr = np.diff(b) / FS
        med = np.median(rr)
        irr[rec["record"]] = float(np.mean(np.abs(rr - med) > 0.2 * med))
        gaps[rec["record"]] = int(np.sum(rr > 1.5 * med))
    recs = sorted(sets["masked_probe"])
    res = dict(rr_irregularity=_strata(irr, recs, sets))
    with_gap = [r for r in recs if gaps[r] > 0]
    no_gap = [r for r in recs if gaps[r] == 0]
    res["missing_beat_gaps"] = {
        name: dict(n=len(rs), f1={k: float(np.mean([s[r] for r in rs])) for k, s in sets.items()})
        for name, rs in (("records_with_gaps", with_gap), ("records_without_gaps", no_gap))}
    top = sorted(recs, key=lambda r: irr[r])[-25:]
    res["rr_irregularity"]["high"]["masked_vs_random"] = _paired(
        [sets["random_probe"][r] for r in top], [sets["masked_probe"][r] for r in top])
    return res


# --------------------------------------------------------------------------- #
def _cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    hsic = np.linalg.norm(X.T @ Y) ** 2
    return float(hsic / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def drift(device_str="mps", epochs=12):
    import torch
    from sklearn.model_selection import GroupKFold
    from eval_finetune import Detector, train_fold
    from eval_qrs import build_labeled_windows
    device = torch.device(device_str)
    items = build_labeled_windows()
    groups = np.array([it["record"] for it in items])

    def feats(enc, idx):
        enc.eval()
        with torch.no_grad():
            x = torch.from_numpy(np.stack([items[i]["signal"] for i in idx])).to(device)
            out = []
            for j in range(0, len(x), 16):
                out.append(enc.token_features(x[j:j + 16]).cpu().numpy())
        f = np.concatenate(out).reshape(-1, out[0].shape[-1])
        rng = np.random.default_rng(SEED)
        return f[rng.choice(len(f), size=min(20000, len(f)), replace=False)]

    rows = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(np.arange(len(items)), groups=groups)):
        dets, pre = {}, {}
        for init in ("random", "mae"):
            d = Detector(init, str(MAE_CKPT), device)
            pre[init] = feats(d.encoder, te)
            train_fold(d, items, tr, device, epochs=epochs)
            dets[init] = d
        post = {k: feats(d.encoder, te) for k, d in dets.items()}
        row = dict(fold=fold,
                   cka_mae_pre_post=_cka(pre["mae"], post["mae"]),
                   cka_random_pre_post=_cka(pre["random"], post["random"]),
                   cka_pre_mae_vs_random=_cka(pre["mae"], pre["random"]),
                   cka_post_mae_vs_random=_cka(post["mae"], post["random"]))
        rows.append(row)
        print(f"  drift fold {fold}: " + "  ".join(f"{k}={v:.3f}" for k, v in row.items() if k != "fold"),
              flush=True)
    keys = [k for k in rows[0] if k != "fold"]
    return dict(epochs=epochs, folds=rows,
                mean={k: float(np.mean([r[k] for r in rows])) for k in keys})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["quality", "rhythm", "drift"])
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    path = OUT / "round2_analyses.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for w in a.which:
        print(f"[{w}]", flush=True)
        res[w] = quality(a.device) if w == "quality" else rhythm() if w == "rhythm" else drift(a.device)
        path.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "drift"}, indent=1)[:4000])
    if "drift" in res:
        print("drift mean:", res["drift"]["mean"])


if __name__ == "__main__":
    main()
