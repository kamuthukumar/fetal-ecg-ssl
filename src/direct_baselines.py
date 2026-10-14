"""Direct fetal-QRS baselines on CinC 2013 set-a (Reviewer 1, Concern 1).

Four baselines, all scored on the same 75 records with the +/-50 ms rule:

  * ica      : FastICA on the four abdominal channels, fetal component chosen
               by rhythm regularity in the fetal HR band (no labels).
  * ts_ica   : maternal template subtraction on every channel, then FastICA on
               the residuals, same component selection (no labels).
  * mcgnet   : MCG-Net-style supervised delineation network (multi-resolution
               CNN backbone + Transformer global-context block + heatmap
               regression head, after Tao et al., IEEE J-BHI 2022).
  * unet     : 1-D U-Net delineation network trained on the same heatmaps.

The two supervised networks use the record-grouped 5-fold split of eval_qrs.py;
the peak threshold is chosen on the training fold only.

    python src/direct_baselines.py --which all
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


# --------------------------------------------------------------------------- #
# unsupervised multi-channel baselines
# --------------------------------------------------------------------------- #
def _fetal_from_sources(sources, fs, fhr=(100, 180)):
    from fecg_extraction import (bandpass, _energy_envelope, _detect_on_envelope,
                                 _refine_to_local_extreme, _rr_consistency_filter,
                                 _rhythm_regularity)
    hi = min(70, fs / 2 * 0.95)
    best, best_score, all_pk = None, np.inf, []
    for s in sources:
        fb = bandpass(s, fs, 10, hi)
        env = _energy_envelope(fb, fs)
        env = np.clip((env - np.median(env)) / (np.std(env) + 1e-12), 0, None)
        pk = _detect_on_envelope(env, fs, fhr[0], fhr[1])
        pk = _refine_to_local_extreme(fb, pk, fs)
        pk = _rr_consistency_filter(pk, fs, fhr)
        all_pk.append(pk)
        sc = _rhythm_regularity(pk, fs, fhr[0], fhr[1])
        if sc < best_score:
            best, best_score = pk, sc
    if best is None:
        best = max(all_pk, key=len)
    return best


def _ica(x):
    from sklearn.decomposition import FastICA
    x = x / (x.std(axis=1, keepdims=True) + 1e-12)
    ica = FastICA(n_components=x.shape[0], whiten="unit-variance",
                  random_state=SEED, max_iter=2000, tol=1e-4)
    return ica.fit_transform(x.T).T


def run_unsupervised():
    from data_loader import load_cinc2013
    from fecg_extraction import (detect_qrs, _refine_to_local_extreme,
                                 cancel_maternal, match_f1)
    res = {"ica": {}, "ts_ica": {}}
    for i in range(1, 76):
        rec = load_cinc2013(f"a{i:02d}")
        x = np.nan_to_num(rec["abdominal"].astype(np.float64))
        ref = np.asarray(rec["fetal_beats"], float)
        fs = rec["fs"]
        keep = x.std(axis=1) > 1e-8
        x = x[keep]
        pk = _fetal_from_sources(_ica(x), fs)
        res["ica"][rec["record"]] = match_f1(pk, ref, fs, 50)["f1"]
        resid = []
        for ch in x:
            mp = _refine_to_local_extreme(
                ch, detect_qrs(ch, fs, 45, 110, thr_scale=0.6)[0], fs)
            resid.append(cancel_maternal(ch, fs, mp))
        pk = _fetal_from_sources(_ica(np.array(resid)), fs)
        res["ts_ica"][rec["record"]] = match_f1(pk, ref, fs, 50)["f1"]
        print(f"  {rec['record']}: ICA {res['ica'][rec['record']]:.3f}  "
              f"TS+ICA {res['ts_ica'][rec['record']]:.3f}", flush=True)
    return {k: _summ(v) for k, v in res.items()}


# --------------------------------------------------------------------------- #
# supervised delineation networks
# --------------------------------------------------------------------------- #
def _build_mcgnet(n_in=4):
    import torch
    import torch.nn as nn

    class MultiRes(nn.Module):
        def __init__(self, cin, cout, stride):
            super().__init__()
            b = cout // 4
            self.br = nn.ModuleList([
                nn.Conv1d(cin, b, k, stride=stride, padding=k // 2) for k in (3, 9, 19, 39)])
            self.bn = nn.BatchNorm1d(b * 4)

        def forward(self, x):
            return torch.relu(self.bn(torch.cat([c(x) for c in self.br], 1)))

    class MCGNetStyle(nn.Module):
        def __init__(self):
            super().__init__()
            self.s1 = MultiRes(n_in, 32, 2)     # 1/2
            self.s2 = MultiRes(32, 64, 2)       # 1/4
            self.s3 = MultiRes(64, 128, 2)      # 1/8
            self.s4 = MultiRes(128, 128, 2)     # 1/16
            layer = nn.TransformerEncoderLayer(128, 4, 256, 0.1, batch_first=True)
            self.tr = nn.TransformerEncoder(layer, 2)
            self.pos = nn.Parameter(torch.zeros(1, 625, 128))
            self.up3 = nn.Conv1d(128 + 128, 64, 5, padding=2)
            self.up2 = nn.Conv1d(64 + 64, 32, 5, padding=2)
            self.up1 = nn.Conv1d(32 + 32, 32, 5, padding=2)
            self.out = nn.Conv1d(32, 1, 1)

        def forward(self, x):
            import torch.nn.functional as F
            f1 = self.s1(x); f2 = self.s2(f1); f3 = self.s3(f2); f4 = self.s4(f3)
            t = f4.transpose(1, 2)
            t = self.tr(t + self.pos[:, :t.shape[1]]).transpose(1, 2)
            u = F.interpolate(t, size=f3.shape[-1]); u = torch.relu(self.up3(torch.cat([u, f3], 1)))
            u = F.interpolate(u, size=f2.shape[-1]); u = torch.relu(self.up2(torch.cat([u, f2], 1)))
            u = F.interpolate(u, size=f1.shape[-1]); u = torch.relu(self.up1(torch.cat([u, f1], 1)))
            u = F.interpolate(u, size=x.shape[-1])
            return self.out(u).squeeze(1)

    return MCGNetStyle()


def _build_unet(n_in=4):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def block(cin, cout):
        return nn.Sequential(nn.Conv1d(cin, cout, 9, padding=4), nn.BatchNorm1d(cout), nn.ReLU(),
                             nn.Conv1d(cout, cout, 9, padding=4), nn.BatchNorm1d(cout), nn.ReLU())

    class UNet1D(nn.Module):
        def __init__(self, ch=(16, 32, 64, 128)):
            super().__init__()
            self.enc = nn.ModuleList()
            c = n_in
            for k in ch:
                self.enc.append(block(c, k)); c = k
            self.mid = block(ch[-1], ch[-1])
            self.dec = nn.ModuleList([block(ch[i] * 2, ch[i - 1] if i else ch[0])
                                      for i in reversed(range(len(ch)))])
            self.out = nn.Conv1d(ch[0], 1, 1)

        def forward(self, x):
            skips = []
            for e in self.enc:
                x = e(x); skips.append(x); x = F.max_pool1d(x, 2)
            x = self.mid(x)
            for d, s in zip(self.dec, reversed(skips)):
                x = F.interpolate(x, size=s.shape[-1])
                x = d(torch.cat([x, s], 1))
            return self.out(x).squeeze(1)

    return UNet1D()


def _windows():
    from data_loader import load_cinc2013
    n = int(WIN_SEC * FS)
    t = np.arange(n)
    items, refs = [], {}
    for i in range(1, 76):
        rec = load_cinc2013(f"a{i:02d}")
        sig = np.nan_to_num(rec["abdominal"].astype(np.float32))
        beats = np.asarray(rec["fetal_beats"], float)
        refs[rec["record"]] = beats
        for w in range(sig.shape[1] // n):
            s0 = w * n
            seg = sig[:, s0:s0 + n]
            seg = (seg - seg.mean(1, keepdims=True)) / (seg.std(1, keepdims=True) + 1e-6)
            b = beats[(beats >= s0) & (beats < s0 + n)] - s0
            hm = np.zeros(n, np.float32)
            for bb in b:
                hm = np.maximum(hm, np.exp(-0.5 * ((t - bb) / 10.0) ** 2))
            items.append(dict(x=seg.astype(np.float32), y=hm.astype(np.float32),
                              record=rec["record"], win=w))
    return items, refs


def _peaks(prob, thr, min_dist):
    from scipy.signal import find_peaks
    pk, _ = find_peaks(prob, height=thr, distance=min_dist)
    return pk


def _predict(model, items, idx, device):
    import torch
    by = {}
    for i in idx:
        by.setdefault(items[i]["record"], []).append(i)
    out = {}
    model.eval()
    with torch.no_grad():
        for r, w in by.items():
            w = sorted(w, key=lambda i: items[i]["win"])
            x = torch.from_numpy(np.stack([items[i]["x"] for i in w])).to(device)
            out[r] = torch.sigmoid(model(x)).cpu().numpy().reshape(-1)
    return out


def _score(prob, refs, thr):
    from eval_qrs import _match_f1
    per = {}
    for r, p in prob.items():
        n = len(p)
        ref = refs[r][refs[r] < n]
        per[r] = _match_f1(_peaks(p, thr, int(0.30 * FS)), ref, 50.0)[3]
    return per


def run_supervised(kind, device_str="mps", epochs=40, lr=1e-3, batch_size=16):
    import torch
    from sklearn.model_selection import GroupKFold
    device = torch.device(device_str)
    items, refs = _windows()
    groups = np.array([it["record"] for it in items])
    per_rec, thrs = {}, []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(np.arange(len(items)), groups=groups)):
        torch.manual_seed(SEED); np.random.seed(SEED)
        model = (_build_mcgnet() if kind == "mcgnet" else _build_unet()).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        steps = epochs * int(np.ceil(len(tr) / batch_size))
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.0, device=device))
        rng = np.random.default_rng(SEED)
        idx = np.array(tr)
        for ep in range(epochs):
            rng.shuffle(idx)
            model.train()
            for j in range(0, len(idx), batch_size):
                bi = idx[j:j + batch_size]
                if len(bi) < 2:
                    continue
                x = torch.from_numpy(np.stack([items[i]["x"] for i in bi])).to(device)
                y = torch.from_numpy(np.stack([items[i]["y"] for i in bi])).to(device)
                loss = lossf(model(x), y)
                opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        ptr = _predict(model, items, tr, device)
        best_thr, best = 0.5, -1
        for thr in np.linspace(0.1, 0.9, 17):
            m = np.mean(list(_score(ptr, refs, thr).values()))
            if m > best:
                best, best_thr = m, thr
        pte = _predict(model, items, te, device)
        fr = _score(pte, refs, best_thr)
        per_rec.update(fr); thrs.append(float(best_thr))
        print(f"  [{kind}] fold {fold}: thr={best_thr:.2f} F1={np.mean(list(fr.values())):.3f}", flush=True)
    s = _summ(per_rec)
    s.update(epochs=epochs, lr=lr, batch_size=batch_size, thresholds=thrs,
             n_params=int(sum(p.numel() for p in model.parameters())))
    return s


def _summ(per):
    f = np.array([per[r] for r in sorted(per)], float)
    return dict(n=int(len(f)), f1_mean=float(f.mean()), f1_std=float(f.std()),
                f1_median=float(np.median(f)), per_record={r: float(per[r]) for r in sorted(per)})


def compare(results):
    from scipy.stats import wilcoxon
    ref = {
        "frozen_masked": json.loads((ROOT / "results/benchmark/qrs_mae.json").read_text())["per_record"],
        "finetune_random": json.loads((ROOT / "results/benchmark/finetune_random.json").read_text())["per_record"],
        "classical_ts": json.loads((OUT / "classical_baseline.json").read_text())["classical"]["per_record"],
    }
    out = {}
    for k, v in results.items():
        pr = v["per_record"]
        for rn, rv in ref.items():
            recs = sorted(set(pr) & set(rv))
            a = np.array([pr[r] for r in recs]); b = np.array([rv[r] for r in recs])
            d = a - b
            out[f"{k}_vs_{rn}"] = dict(delta=float(d.mean()), p=float(wilcoxon(a, b).pvalue),
                                       wins=int((d > 0).sum()), losses=int((d < 0).sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    path = OUT / "direct_baselines.json"
    results = json.loads(path.read_text()).get("results", {}) if path.exists() else {}
    if a.which in ("all", "unsup"):
        results.update(run_unsupervised())
    for k in ("mcgnet", "unet"):
        if a.which in ("all", k):
            results[k] = run_supervised(k, a.device)
    path.write_text(json.dumps(dict(results=results, tests=compare(results)), indent=2))
    for k, v in results.items():
        print(f"{k:8s} F1 {v['f1_mean']:.3f} +/- {v['f1_std']:.3f} (median {v['f1_median']:.3f})")


if __name__ == "__main__":
    main()
