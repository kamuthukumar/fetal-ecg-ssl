"""End-to-end fine-tuning benchmark for fetal QRS detection.

Unlike eval_qrs.py (frozen encoder + linear probe), here the WHOLE encoder is
trained on the detection task, initialised either from random weights or from a
pre-trained checkpoint (e.g. MAE). This is the standard way foundation models
are evaluated and tests whether pre-training gives a better initialisation.

Same held-out data and protocol as eval_qrs: CinC 2013, record-grouped 5-fold
CV, +/-50 ms matching to the expert .fqrs reference, per-record F1, seed 42.

    python src/eval_finetune.py --init mae  --ckpt results/foundation_mae/final_model.pt
    python src/eval_finetune.py --init random
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
FS = 1000.0
WIN_SEC = 10.0


class Detector:
    """Encoder (FetalECGFM) + linear per-token QRS head, trained end-to-end."""
    def __init__(self, init_kind, ckpt, device):
        import torch
        from model import FetalECGFM
        torch.manual_seed(SEED)
        self.device = device
        self.encoder = FetalECGFM(n_input_channels=4, d_model=256, proj_dim=64, pool=POOL).to(device)
        if init_kind != "random":
            state = torch.load(ckpt, map_location=device)
            state = state.get("model_state", state)
            self.encoder.load_state_dict(state)
        self.head = torch.nn.Linear(256, 1).to(device)

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.head.parameters())

    def logits(self, x):
        return self.head(self.encoder.token_features(x)).squeeze(-1)  # (B, T')

    def train(self):
        self.encoder.train(); self.head.train()

    def eval(self):
        self.encoder.eval(); self.head.eval()


def train_fold(det, items, idx, device, epochs=25, batch_size=16, lr=3e-4):
    import torch
    opt = torch.optim.AdamW(det.parameters(), lr=lr, weight_decay=1e-4)
    ytr_all = np.concatenate([items[i]["token_label"] for i in idx])
    pos = max(ytr_all.sum(), 1.0)
    pos_weight = torch.tensor([(len(ytr_all) - pos) / pos], device=device)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    rng = np.random.default_rng(SEED)
    idx = list(idx)
    for ep in range(epochs):
        rng.shuffle(idx)
        det.train()
        for j in range(0, len(idx), batch_size):
            bi = idx[j:j + batch_size]
            if len(bi) < 2:
                continue
            x = torch.from_numpy(np.stack([items[i]["signal"] for i in bi])).to(device)
            y = torch.from_numpy(np.stack([items[i]["token_label"] for i in bi])).to(device)
            logit = det.logits(x)
            loss = lossf(logit, y)
            opt.zero_grad(); loss.backward(); opt.step()
    det.eval()


def predict_prob_by_record(det, items, idx, device, batch_size=16):
    import torch
    by_rec = {}
    for i in idx:
        by_rec.setdefault(items[i]["record"], []).append(i)
    prob_by_record = {}
    with torch.no_grad():
        for rec, widx in by_rec.items():
            widx = sorted(widx, key=lambda i: items[i]["win"])
            probs = []
            for j in range(0, len(widx), batch_size):
                bi = widx[j:j + batch_size]
                x = torch.from_numpy(np.stack([items[i]["signal"] for i in bi])).to(device)
                p = torch.sigmoid(det.logits(x)).cpu().numpy()
                probs.extend(list(p))
            prob_by_record[rec] = np.concatenate(probs, axis=0)
    return prob_by_record


def run(init_kind, ckpt, device_str="mps", epochs=25, out_dir=None, label_frac=1.0, seed=SEED):
    import torch
    from sklearn.model_selection import GroupKFold
    from eval_qrs import build_labeled_windows, score_records

    device = torch.device(device_str)
    np.random.seed(SEED); torch.manual_seed(SEED)
    items = build_labeled_windows()
    groups = np.array([it["record"] for it in items])
    print(f"[finetune:{init_kind}] {len(items)} windows, {len(set(groups))} records")

    ref_by_record = {}
    for it in items:
        s0 = it["win"] * int(WIN_SEC * FS)
        ref_by_record.setdefault(it["record"], []).extend((it["beats_in_win"] + s0).tolist())

    gkf = GroupKFold(n_splits=5)
    fold_f1 = {}
    rng = np.random.default_rng(seed)
    n_used = []
    for fold, (tr, te) in enumerate(gkf.split(np.arange(len(items)), groups=groups)):
        if label_frac < 1.0:
            tr_recs = np.array(sorted(set(groups[tr])))
            k = max(1, int(round(label_frac * len(tr_recs))))
            chosen = set(rng.choice(tr_recs, size=k, replace=False))
            tr = np.array([i for i in tr if groups[i] in chosen])
            n_used.append(int(k))
        else:
            n_used.append(int(len(set(groups[tr]))))
        det = Detector(init_kind, ckpt, device)
        train_fold(det, items, tr, device, epochs=epochs)
        # threshold on train fold
        ptr = predict_prob_by_record(det, items, tr, device)
        reftr = {r: ref_by_record[r] for r in ptr}
        best_thr, best = 0.5, -1
        for thr in np.linspace(0.2, 0.8, 13):
            m = np.mean(list(score_records(ptr, reftr, thr).values()))
            if m > best:
                best, best_thr = m, thr
        # score test fold
        pte = predict_prob_by_record(det, items, te, device)
        refte = {r: ref_by_record[r] for r in pte}
        per_rec = score_records(pte, refte, best_thr)
        fold_f1.update(per_rec)
        print(f"  fold {fold}: thr={best_thr:.2f} F1={np.mean(list(per_rec.values())):.3f}")

    f1s = np.array([fold_f1[r] for r in sorted(fold_f1)])
    summary = dict(init=init_kind, mode="finetune", n_records=int(len(f1s)),
                   f1_mean=float(f1s.mean()), f1_std=float(f1s.std()),
                   f1_median=float(np.median(f1s)),
                   frac_below_0p5=float((f1s < 0.5).mean()),
                   label_frac=float(label_frac),
                   n_train_records_per_fold=n_used,
                   per_record={r: float(fold_f1[r]) for r in sorted(fold_f1)})
    if out_dir:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        (out / f"finetune_{init_kind}.json").write_text(json.dumps(summary, indent=2))
    print(f"[finetune:{init_kind}] F1 = {summary['f1_mean']:.3f} +/- {summary['f1_std']:.3f} "
          f"(median {summary['f1_median']:.3f}, frac<0.5 {summary['frac_below_0p5']:.2f})")
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init", choices=["random", "mae", "simclr", "vicreg"], required=True)
    p.add_argument("--ckpt", type=str, default=str(ROOT / "results" / "foundation_mae" / "final_model.pt"))
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--label_frac", type=float, default=1.0,
                   help="Fraction of training records used for fine-tuning.")
    p.add_argument("--out_dir", type=str, default=str(ROOT / "results" / "benchmark"))
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(a.init, a.ckpt, a.device, a.epochs, a.out_dir, label_frac=a.label_frac)
