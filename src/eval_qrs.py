"""Honest fetal-QRS-detection benchmark on PhysioNet/CinC 2013 set-a.

This is the core downstream evaluation. It tests whether a *frozen* encoder's
per-token features let a shallow detection head localise fetal QRS complexes,
comparing encoders that differ only in how they were pre-trained:

    * random         : randomly initialised encoder (random-feature baseline)
    * ssl            : Fetal-ECG-FM, self-supervised (physics augs + GA-aware NCE)
    * simclr         : same architecture, generic augmentations, plain InfoNCE

The encoder is frozen; only a small detection head (a linear probe over token
features) is trained, with record-grouped cross-validation so no recording
appears in both train and test. Beats are matched to the expert `.fqrs`
reference with a +/-50 ms tolerance and scored by F1. Everything is seeded.

Usage:
    python src/eval_qrs.py --encoder ssl   --ckpt results/foundation/final_model.pt
    python src/eval_qrs.py --encoder random
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
POOL = 32          # encoder front-end downsampling; token rate = fs / POOL
FS = 1000.0
WIN_SEC = 10.0
TOL_MS = 50.0      # matching tolerance for fetal QRS


def build_labeled_windows(win_sec=WIN_SEC, fs=FS, pool=POOL):
    """Non-overlapping labelled windows from CinC 2013 set-a.

    Returns a list of dicts: signal (4, win), token_label (T',), record, and the
    absolute reference-beat sample indices per window (for scoring).
    """
    from data_loader import load_cinc2013
    n_samp = int(win_sec * fs)
    n_tok = n_samp // pool
    items = []
    for i in range(1, 76):
        rec = load_cinc2013(f"a{i:02d}")
        if rec is None:
            continue
        sig = np.nan_to_num(rec["abdominal"].astype(np.float32))
        beats = np.asarray(rec.get("fetal_beats", []), dtype=float)
        if beats.size < 5:
            continue
        n_win = sig.shape[1] // n_samp
        for w in range(n_win):
            s0 = w * n_samp
            seg = sig[:, s0:s0 + n_samp]
            # per-token binary label: does token cover a reference beat?
            in_win = beats[(beats >= s0) & (beats < s0 + n_samp)] - s0
            lab = np.zeros(n_tok, dtype=np.float32)
            tok_idx = np.clip((in_win // pool).astype(int), 0, n_tok - 1)
            lab[tok_idx] = 1.0
            items.append(dict(signal=seg, token_label=lab, record=rec["record"],
                              beats_in_win=in_win, win=w))
    return items


def make_encoder(kind, ckpt, device):
    import torch
    from model import FetalECGFM
    torch.manual_seed(SEED)
    model = FetalECGFM(n_input_channels=4, d_model=256, proj_dim=64, pool=POOL).to(device)
    if kind in ("ssl", "simclr"):
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        state = torch.load(ckpt, map_location=device)
        state = state.get("model_state", state)
        model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def extract_token_features(model, items, device, batch_size=16):
    """Return list of (T', d_model) feature arrays, one per window."""
    import torch
    feats = []
    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            x = torch.from_numpy(np.stack([it["signal"] for it in batch])).to(device)
            tok = model.token_features(x).cpu().numpy()  # (B, T', d)
            feats.extend(list(tok))
    return feats


def _match_f1(pred_samples, ref_samples, tol_samples):
    """Greedy one-to-one matching within tolerance -> (tp, fp, fn, f1)."""
    if len(ref_samples) == 0:
        return 0, len(pred_samples), 0, 0.0
    if len(pred_samples) == 0:
        return 0, 0, len(ref_samples), 0.0
    ref = np.sort(np.asarray(ref_samples, float))
    pred = np.sort(np.asarray(pred_samples, float))
    used = np.zeros(len(ref), dtype=bool)
    tp = 0
    for p in pred:
        d = np.abs(ref - p)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol_samples:
            used[j] = True
            tp += 1
    fp = len(pred) - tp
    fn = len(ref) - int(used.sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return tp, fp, fn, f1


def _peak_pick(prob, thr, min_dist_tok):
    """Simple 1-D peak picker on a per-token probability trace."""
    idx = []
    i = 1
    n = len(prob)
    while i < n - 1:
        if prob[i] >= thr and prob[i] >= prob[i - 1] and prob[i] >= prob[i + 1]:
            if idx and (i - idx[-1]) < min_dist_tok:
                if prob[i] > prob[idx[-1]]:
                    idx[-1] = i
            else:
                idx.append(i)
        i += 1
    return np.array(idx, dtype=int)


def score_records(prob_by_record, ref_by_record, thr, pool=POOL, fs=FS):
    """F1 per record given per-token probabilities and reference beat samples."""
    min_dist_tok = int(0.30 * fs / pool)     # >=300 ms between fetal beats
    tol = TOL_MS / 1000.0 * fs               # +/-50 ms in samples
    per_rec = {}
    for rec, prob in prob_by_record.items():
        peaks_tok = _peak_pick(prob, thr, min_dist_tok)
        peaks_samp = peaks_tok * pool + pool // 2
        _, _, _, f1 = _match_f1(peaks_samp, ref_by_record[rec], tol)
        per_rec[rec] = f1
    return per_rec


def run(encoder_kind, ckpt, device_str="mps", out_dir=None):
    import torch
    from sklearn.model_selection import GroupKFold

    device = torch.device(device_str)
    np.random.seed(SEED)

    items = build_labeled_windows()
    records = sorted({it["record"] for it in items})
    print(f"[{encoder_kind}] {len(items)} windows from {len(records)} CinC records")

    model = make_encoder(encoder_kind, ckpt, device)
    feats = extract_token_features(model, items, device)          # list of (T', d)
    d = feats[0].shape[1]

    groups = np.array([it["record"] for it in items])
    # absolute reference beats per record (for scoring), and window offsets
    ref_by_record = {}
    for it in items:
        s0 = it["win"] * int(WIN_SEC * FS)
        ref_by_record.setdefault(it["record"], []).extend((it["beats_in_win"] + s0).tolist())

    gkf = GroupKFold(n_splits=5)
    fold_f1 = {}
    for fold, (tr, te) in enumerate(gkf.split(np.arange(len(items)), groups=groups)):
        # ---- train linear detection head on train-fold tokens ----
        Xtr = np.concatenate([feats[i] for i in tr], axis=0)
        ytr = np.concatenate([items[i]["token_label"] for i in tr], axis=0)
        head = _fit_head(Xtr, ytr, d, device)

        # ---- choose threshold on the train fold ----
        thr = _best_threshold(head, feats, items, tr, groups, device)

        # ---- score test-fold records ----
        prob_by_record = _predict_prob_by_record(head, feats, items, te, device)
        ref_te = {r: ref_by_record[r] for r in prob_by_record}
        per_rec = score_records(prob_by_record, ref_te, thr)
        fold_f1.update(per_rec)
        print(f"  fold {fold}: thr={thr:.2f}  mean F1={np.mean(list(per_rec.values())):.3f} "
              f"({len(per_rec)} recs)")

    f1s = np.array([fold_f1[r] for r in sorted(fold_f1)])
    summary = dict(
        encoder=encoder_kind,
        n_records=int(len(f1s)),
        f1_mean=float(f1s.mean()),
        f1_std=float(f1s.std()),
        f1_median=float(np.median(f1s)),
        frac_below_0p5=float((f1s < 0.5).mean()),
        per_record={r: float(fold_f1[r]) for r in sorted(fold_f1)},
    )
    if out_dir:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        (out / f"qrs_{encoder_kind}.json").write_text(json.dumps(summary, indent=2))
        print(f"  wrote {out / f'qrs_{encoder_kind}.json'}")
    print(f"[{encoder_kind}] F1 = {summary['f1_mean']:.3f} +/- {summary['f1_std']:.3f} "
          f"(median {summary['f1_median']:.3f})")
    return summary


def _fit_head(X, y, d, device, epochs=60, lr=1e-3):
    import torch
    torch.manual_seed(SEED)
    head = torch.nn.Linear(d, 1).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    pos = max(y.sum(), 1.0)
    pos_weight = torch.tensor([(len(y) - pos) / pos], device=device)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).to(device).unsqueeze(1)
    for _ in range(epochs):
        opt.zero_grad()
        out = head(Xt)
        loss = lossf(out, yt)
        loss.backward()
        opt.step()
    head.eval()
    return head


def _window_prob(head, feat, device):
    import torch
    with torch.no_grad():
        logit = head(torch.from_numpy(feat).to(device)).squeeze(1)
        return torch.sigmoid(logit).cpu().numpy()


def _predict_prob_by_record(head, feats, items, idx, device):
    prob_by_record = {}
    # collect windows per record in temporal order
    by_rec = {}
    for i in idx:
        by_rec.setdefault(items[i]["record"], []).append(i)
    for rec, widx in by_rec.items():
        widx = sorted(widx, key=lambda i: items[i]["win"])
        probs = [_window_prob(head, feats[i], device) for i in widx]
        prob_by_record[rec] = np.concatenate(probs, axis=0)
    return prob_by_record


def _best_threshold(head, feats, items, idx, groups, device):
    ref_by_record = {}
    for i in idx:
        s0 = items[i]["win"] * int(WIN_SEC * FS)
        ref_by_record.setdefault(items[i]["record"], []).extend(
            (items[i]["beats_in_win"] + s0).tolist())
    prob_by_record = _predict_prob_by_record(head, feats, items, idx, device)
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.2, 0.8, 13):
        per_rec = score_records(prob_by_record, ref_by_record, thr)
        m = np.mean(list(per_rec.values()))
        if m > best_f1:
            best_f1, best_thr = m, thr
    return best_thr


def few_shot(encoder_kind, ckpt, fractions=(0.05, 0.1, 0.25, 0.5, 1.0),
             device_str="mps", n_splits=5):
    """Few-shot QRS detection: train the head on a fraction of *records*.

    For each CV fold, subsample the training records to each fraction (grouped by
    record, seeded), fit the head, and score the held-out fold. Returns per-
    fraction mean/std F1 across folds.
    """
    import torch
    from sklearn.model_selection import GroupKFold

    device = torch.device(device_str)
    items = build_labeled_windows()
    model = make_encoder(encoder_kind, ckpt, device)
    feats = extract_token_features(model, items, device)
    d = feats[0].shape[1]
    groups = np.array([it["record"] for it in items])
    rng = np.random.default_rng(SEED)

    ref_by_record = {}
    for it in items:
        s0 = it["win"] * int(WIN_SEC * FS)
        ref_by_record.setdefault(it["record"], []).extend((it["beats_in_win"] + s0).tolist())

    gkf = GroupKFold(n_splits=n_splits)
    per_frac = {f: [] for f in fractions}
    for tr, te in gkf.split(np.arange(len(items)), groups=groups):
        tr_recs = np.array(sorted(set(groups[tr])))
        for f in fractions:
            k = max(1, int(round(f * len(tr_recs))))
            chosen = set(rng.choice(tr_recs, size=k, replace=False))
            tri = [i for i in tr if groups[i] in chosen]
            Xtr = np.concatenate([feats[i] for i in tri], axis=0)
            ytr = np.concatenate([items[i]["token_label"] for i in tri], axis=0)
            head = _fit_head(Xtr, ytr, d, device)
            thr = _best_threshold(head, feats, items, tri, groups, device)
            prob = _predict_prob_by_record(head, feats, items, te, device)
            ref_te = {r: ref_by_record[r] for r in prob}
            per_rec = score_records(prob, ref_te, thr)
            per_frac[f].append(float(np.mean(list(per_rec.values()))))
    return {
        "fractions": list(fractions),
        "mean": [float(np.mean(per_frac[f])) for f in fractions],
        "std": [float(np.std(per_frac[f])) for f in fractions],
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", choices=["random", "ssl", "simclr"], required=True)
    p.add_argument("--ckpt", type=str, default=str(ROOT / "results" / "foundation" / "final_model.pt"))
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--out_dir", type=str, default=str(ROOT / "results" / "qrs_bench"))
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(a.encoder, a.ckpt, a.device, a.out_dir)
