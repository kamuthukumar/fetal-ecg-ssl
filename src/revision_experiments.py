"""Revision analyses requested by JBHI reviewers.

Runs on existing checkpoints immediately (no retrain required):
  * pairwise SSL statistics, effect sizes, bootstrap CIs
  * QRS matching at +/-20/30/50/100 ms
  * amplitude-perturbation / scale-invariance test
  * classical maternal-cancellation baseline + maternal-QRS false-positive overlap
  * few-shot with repeated record subsampling and exact n
  * dataset inventory (what was actually used)
  * ADfECGDB frozen-probe transfer (unseen hardware)

Usage:
    DEVICE=mps python src/revision_experiments.py
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

BENCH = ROOT / "results" / "benchmark"
OUT = ROOT / "results" / "revision"
OUT.mkdir(parents=True, exist_ok=True)

CKPTS = {
    "random": None,
    "simclr": ROOT / "results" / "foundation_simclr" / "final_model.pt",
    "vicreg": ROOT / "results" / "foundation_vicreg" / "final_model.pt",
    "mae": ROOT / "results" / "foundation_mae" / "final_model.pt",
}


def _load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def paired_stats(a, b, n_boot=2000, seed=42):
    """Paired comparison of per-record F1 arrays a (baseline) vs b (method)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = b - a
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    # Cohen's dz for paired designs
    dz = float(mean / (sd + 1e-12))
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    try:
        p_gt = float(wilcoxon(b, a, alternative="greater").pvalue)
    except ValueError:
        p_gt = None
    try:
        p_two = float(wilcoxon(b, a, alternative="two-sided").pvalue)
    except ValueError:
        p_two = None
    return {
        "n": int(n),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "delta": mean,
        "delta_sd": sd,
        "cohens_dz": dz,
        "ci95": list(ci),
        "p_greater": p_gt,
        "p_two_sided": p_two,
        "n_b_wins": int((d > 0).sum()),
        "n_a_wins": int((d < 0).sum()),
        "n_ties": int((d == 0).sum()),
    }


def stats_from_existing():
    """Reviewer 2 Q2 / Reviewer 3 R4: effect sizes, CIs, SSL-vs-SSL tests."""
    methods = ["random", "simclr", "vicreg", "mae"]
    recs = None
    f1 = {}
    for m in methods:
        d = _load(BENCH / f"qrs_{m}.json")
        if d is None:
            continue
        f1[m] = d["per_record"]
        recs = recs or sorted(f1[m])
    if recs is None:
        print("  no QRS json; skip stats")
        return {}
    recs = [r for r in recs if all(r in f1[m] for m in f1)]
    out = {"n_records": len(recs), "pairs": {}}
    names = list(f1)
    for i, m1 in enumerate(names):
        for m2 in names[i + 1:]:
            a = [f1[m1][r] for r in recs]
            b = [f1[m2][r] for r in recs]
            out["pairs"][f"{m2}_vs_{m1}"] = paired_stats(a, b)
    (OUT / "pairwise_stats.json").write_text(json.dumps(out, indent=2))
    print("  pairwise SSL stats -> revision/pairwise_stats.json")
    for k, v in out["pairs"].items():
        print(f"    {k}: Δ={v['delta']:+.4f}  dz={v['cohens_dz']:.3f}  "
              f"CI=[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]  "
              f"p_two={v['p_two_sided']}")
    return out


def dataset_inventory():
    """Reviewer 3 R12: exact record list used at analysis time."""
    from build_cache import load_cached
    recs = load_cached() or []
    by = {}
    for r in recs:
        by.setdefault(r["dataset"], []).append(str(r["record"]))
    inv = {
        "n_total": len(recs),
        "by_dataset": {k: {"n": len(v), "records": v} for k, v in by.items()},
        "pretrain_used": {
            "NInFEA": by.get("NInFEA", []),
            "NI_FECG_DB": by.get("NI_FECG_DB", []),
        },
        "held_out": {
            "CinC2013": by.get("CinC2013", []),
            "ADfECGDB": by.get("ADfECGDB", []),
        },
        "notes": {
            "public_nifecgdb_n": 55,
            "cached_nifecgdb_n": len(by.get("NI_FECG_DB", [])),
            "public_nifeadb_n": 26,
            "cached_nifeadb_n": len(by.get("NIFEA_ARR", [])),
            "cinc_set_b": "not used (unlabelled; held out of pre-training along with set-a)",
            "ninfea_subjects": 39,
        },
    }
    from ninfea_labels import SUBJECT
    inv["ninfea_n_subjects"] = len(set(SUBJECT.values()))
    (OUT / "dataset_inventory.json").write_text(json.dumps(inv, indent=2))
    print("  dataset inventory:", {k: v["n"] for k, v in inv["by_dataset"].items()})
    return inv


def classical_and_maternal_fp(device="mps"):
    """Reviewer 2 Q7 / Reviewer 3 R7: classical pipeline + maternal-QRS FP overlap."""
    import pandas as pd
    from fecg_extraction import extract_fetal, detect_qrs, _refine_to_local_extreme, match_f1
    from data_loader import load_cinc2013

    csv = ROOT / "results" / "extraction_per_record.csv"
    classical = {}
    if csv.exists():
        df = pd.read_csv(csv)
        cinc = df[df.dataset == "CinC2013"]
        classical = {
            "n": int(len(cinc)),
            "f1_mean": float(cinc.f1.mean()),
            "f1_std": float(cinc.f1.std()),
            "f1_median": float(cinc.f1.median()),
            "per_record": {r: float(v) for r, v in zip(cinc.record, cinc.f1)},
        }

    # Maternal-QRS coincidence of frozen-probe false positives (MAE).
    mae = _load(BENCH / "qrs_mae.json")
    overlap = {"n_fp": 0, "n_fp_on_maternal": 0, "per_record": {}}
    # Recompute detections from cached probe if we have peaks; otherwise
    # re-run a lightweight maternal match using classical residual FPs vs maternal.
    # Here we take classical detections vs maternal QRS as a diagnostic of the
    # *signal* (how often abdominal energy is maternal), and also re-score the
    # MAE probe FPs if we can rebuild peaks.
    n_fp_ssl, n_fp_mat = 0, 0
    n_rec = 0
    for i in range(1, 76):
        rec = load_cinc2013(f"a{i:02d}")
        if rec is None:
            continue
        ref = np.asarray(rec.get("fetal_beats", []), float)
        if ref.size < 5:
            continue
        sig = rec["abdominal"]
        fs = rec["fs"]
        # maternal peaks: energy detector in maternal HR band on loudest channel
        vars_ = np.var(sig, axis=1)
        ch = int(np.argmax(vars_))
        mat, _ = detect_qrs(sig[ch], fs, 45, 110, thr_scale=0.6)
        mat = _refine_to_local_extreme(sig[ch], mat, fs)
        # classical fetal detections
        out = extract_fetal(sig, fs)
        det = np.asarray(out.fetal_peaks, float)
        tol = 50.0 * fs / 1000.0
        # FPs of classical pipeline
        used = np.zeros(len(ref), dtype=bool)
        fps = []
        for d in det:
            dist = np.abs(ref - d)
            dist[used] = np.inf
            j = int(np.argmin(dist)) if len(ref) else 0
            if len(ref) and dist[j] <= tol:
                used[j] = True
            else:
                fps.append(d)
        if not fps or len(mat) == 0:
            continue
        fps = np.asarray(fps)
        hit = 0
        for d in fps:
            if np.min(np.abs(mat - d)) <= tol:
                hit += 1
        n_fp_ssl += len(fps)
        n_fp_mat += hit
        n_rec += 1
        overlap["per_record"][rec["record"]] = {
            "n_fp": int(len(fps)), "n_fp_on_maternal": int(hit),
            "frac": float(hit / len(fps)),
        }
    overlap.update({
        "n_records": n_rec,
        "n_fp": int(n_fp_ssl),
        "n_fp_on_maternal": int(n_fp_mat),
        "frac_fp_on_maternal": float(n_fp_mat / max(n_fp_ssl, 1)),
        "note": "Classical pipeline false positives coincident with maternal QRS (±50 ms).",
    })
    result = {"classical": classical, "maternal_fp_classical": overlap}
    (OUT / "classical_baseline.json").write_text(json.dumps(result, indent=2))
    print(f"  classical F1={classical.get('f1_mean')}  "
          f"maternal-FP fraction={overlap['frac_fp_on_maternal']:.3f}")
    return result


def _make_encoder(kind, ckpt, device, **kw):
    import torch
    from model import FetalECGFM
    from eval_qrs import SEED, POOL
    torch.manual_seed(SEED)
    model = FetalECGFM(n_input_channels=4, d_model=256, proj_dim=64, pool=POOL, **kw).to(device)
    if kind != "random" and ckpt:
        state = torch.load(ckpt, map_location=device)
        state = state.get("model_state", state)
        model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def amplitude_perturbation(device="mps"):
    """Reviewer 3 R5: does the representation encode amplitude?

    For a frozen encoder, compare embeddings of x and αx. With instance
    normalisation the cosine similarity should stay near 1. We also train a
    linear probe to predict log α from the embedding; chance-level MAE means
    amplitude was not retained.
    """
    import torch
    from eval_qrs import build_labeled_windows, extract_token_features

    items = build_labeled_windows()[:80]  # a few minutes of CinC is enough
    scales = [0.25, 0.5, 1.0, 2.0, 4.0]
    out = {}
    for kind, ckpt in [("mae", CKPTS["mae"]), ("simclr", CKPTS["simclr"]),
                       ("random", None)]:
        if ckpt is not None and not Path(ckpt).exists():
            continue
        model = _make_encoder("ssl" if kind != "random" else "random",
                              None if ckpt is None else str(ckpt), device)
        # original embeddings (pooled tokens)
        feats0 = extract_token_features(model, items, device)
        z0 = np.stack([f.mean(0) for f in feats0])
        z0 = z0 / (np.linalg.norm(z0, axis=1, keepdims=True) + 1e-8)
        sims = {}
        for a in scales:
            items_a = [dict(it, signal=it["signal"] * a) for it in items]
            feats = extract_token_features(model, items_a, device)
            z = np.stack([f.mean(0) for f in feats])
            z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
            sims[str(a)] = float((z0 * z).sum(1).mean())
        # linear probe: predict log2(scale) from embedding of scaled window
        X, y = [], []
        for a in scales:
            items_a = [dict(it, signal=it["signal"] * a) for it in items]
            feats = extract_token_features(model, items_a, device)
            for f in feats:
                X.append(f.mean(0))
                y.append(np.log2(a))
        X = np.stack(X); y = np.asarray(y)
        # ridge on even/odd split
        w = np.linalg.lstsq(X[::2], y[::2], rcond=None)[0]
        pred = X[1::2] @ w
        mae = float(np.mean(np.abs(pred - y[1::2])))
        out[kind] = {"cosine_vs_unscaled": sims, "log2scale_probe_mae": mae,
                     "chance_mae": float(np.mean(np.abs(y[1::2] - y[1::2].mean())))}
        print(f"  {kind} cosine@2x={sims['2.0']:.3f}  scale-probe MAE={mae:.3f} "
              f"(chance {out[kind]['chance_mae']:.3f})")
    (OUT / "amplitude_perturbation.json").write_text(json.dumps(out, indent=2))
    return out


def qrs_tolerances(device="mps"):
    """Reviewer 2 Q4: F1 at ±20/30/50/100 ms, same detections."""
    import torch
    from sklearn.model_selection import GroupKFold
    from eval_qrs import (build_labeled_windows, extract_token_features, _fit_head,
                          _best_threshold, _predict_prob_by_record, score_records,
                          SEED, POOL, FS, WIN_SEC)

    items = build_labeled_windows()
    groups = np.array([it["record"] for it in items])
    ref_by_record = {}
    for it in items:
        s0 = it["win"] * int(WIN_SEC * FS)
        ref_by_record.setdefault(it["record"], []).extend((it["beats_in_win"] + s0).tolist())

    tols = (20.0, 30.0, 50.0, 100.0)
    results = {}
    for kind, ckpt in CKPTS.items():
        if ckpt is not None and not Path(ckpt).exists():
            continue
        print(f"  tolerances [{kind}]")
        loader = "random" if kind == "random" else "ssl"
        from eval_qrs import make_encoder
        model = make_encoder(loader, None if ckpt is None else str(ckpt), device)
        feats = extract_token_features(model, items, device)
        d = feats[0].shape[1]
        gkf = GroupKFold(n_splits=5)
        per_tol = {t: {} for t in tols}
        for fold, (tr, te) in enumerate(gkf.split(np.arange(len(items)), groups=groups)):
            Xtr = np.concatenate([feats[i] for i in tr], axis=0)
            ytr = np.concatenate([items[i]["token_label"] for i in tr], axis=0)
            head = _fit_head(Xtr, ytr, d, device)
            thr = _best_threshold(head, feats, items, tr, groups, device)
            prob = _predict_prob_by_record(head, feats, items, te, device)
            ref_te = {r: ref_by_record[r] for r in prob}
            for t in tols:
                per_tol[t].update(score_records(prob, ref_te, thr, tol_ms=t))
        results[kind] = {
            str(int(t)): {
                "f1_mean": float(np.mean(list(per_tol[t].values()))),
                "f1_std": float(np.std(list(per_tol[t].values()))),
            } for t in tols
        }
        print("   ", results[kind])
    (OUT / "qrs_tolerances.json").write_text(json.dumps(results, indent=2))
    return results


def fewshot_repeated(device="mps", n_repeats=5):
    """Reviewer 3 R8: repeated subsampling + exact training-record counts."""
    from eval_qrs import few_shot
    out = {}
    for kind, ckpt in [("random", None), ("mae", CKPTS["mae"])]:
        if ckpt is not None and not Path(ckpt).exists():
            continue
        print(f"  few-shot repeats [{kind}] x{n_repeats}")
        loader = "random" if kind == "random" else "ssl"
        fs = few_shot(loader, None if ckpt is None else str(ckpt),
                      device_str=device, n_repeats=n_repeats)
        out[kind] = fs
        print("   ", list(zip(fs["fractions"], fs["mean"])))
    (OUT / "fewshot_repeated.json").write_text(json.dumps(out, indent=2))
    return out


def adfecgdb_transfer(device="mps"):
    """Reviewer 2 Q1: frozen probe on ADfECGDB (different hardware, scalp ref)."""
    import torch
    from data_loader import load_adfecgdb, preprocess, select_channels, DATA_ROOT
    from edf_io import read_edf
    from eval_qrs import (make_encoder, extract_token_features, _fit_head,
                          POOL, FS, SEED)
    from sklearn.model_selection import GroupKFold

    items = []
    for rid in ["r01", "r04", "r07", "r08", "r10"]:
        rec = load_adfecgdb(rid)
        edf_path = DATA_ROOT / "adfecgdb" / f"{rid}.edf"
        if rec is None or not edf_path.exists():
            continue
        sig = np.nan_to_num(rec["abdominal"].astype(np.float32))
        fs_rec = float(rec["fs"])
        if abs(fs_rec - FS) > 1:
            from scipy.signal import resample
            n_t = int(sig.shape[1] * FS / fs_rec)
            sig = resample(sig, n_t, axis=1)
        if sig.shape[0] != 4:
            from data_loader import select_channels
            sig = select_channels(sig, FS, n_keep=4)
        edf = read_edf(edf_path)
        beats = np.round(edf.ann_times * FS).astype(float)
        if beats.size < 5:
            continue
        n_samp = int(10 * FS)
        n_tok = n_samp // POOL
        n_win = sig.shape[1] // n_samp
        for w in range(n_win):
            s0 = w * n_samp
            seg = sig[:, s0:s0 + n_samp]
            in_win = beats[(beats >= s0) & (beats < s0 + n_samp)] - s0
            lab = np.zeros(n_tok, np.float32)
            if in_win.size:
                tok_idx = np.clip((in_win // POOL).astype(int), 0, n_tok - 1)
                lab[tok_idx] = 1.0
            items.append(dict(signal=seg, token_label=lab, record=rid,
                              beats_in_win=in_win, win=w))
    if len({it["record"] for it in items}) < 2:
        print("  ADfECGDB: no fetal-beat labels in unified loader; skip transfer probe")
        # still report classical F1 from extraction csv
        return {"skipped": True, "reason": "no fetal_beats on ADfECGDB in data_loader"}
    print(f"  ADfECGDB transfer: {len(items)} windows")
    out = {}
    for kind, ckpt in [("random", None), ("mae", CKPTS["mae"])]:
        loader = "random" if kind == "random" else "ssl"
        model = make_encoder(loader, None if ckpt is None else str(ckpt), device)
        feats = extract_token_features(model, items, device)
        d = feats[0].shape[1]
        groups = np.array([it["record"] for it in items])
        ref = {}
        for it in items:
            s0 = it["win"] * int(10 * FS)
            ref.setdefault(it["record"], []).extend((it["beats_in_win"] + s0).tolist())
        gkf = GroupKFold(n_splits=min(5, len(set(groups))))
        fold_f1 = {}
        from eval_qrs import _best_threshold, _predict_prob_by_record, score_records
        for tr, te in gkf.split(np.arange(len(items)), groups=groups):
            Xtr = np.concatenate([feats[i] for i in tr], 0)
            ytr = np.concatenate([items[i]["token_label"] for i in tr], 0)
            head = _fit_head(Xtr, ytr, d, device)
            thr = _best_threshold(head, feats, items, tr, groups, device)
            prob = _predict_prob_by_record(head, feats, items, te, device)
            fold_f1.update(score_records(prob, {r: ref[r] for r in prob}, thr))
        f1s = np.array(list(fold_f1.values()))
        out[kind] = {"f1_mean": float(f1s.mean()), "f1_std": float(f1s.std()),
                     "n_records": int(len(f1s))}
        print(f"    {kind} F1={out[kind]['f1_mean']:.3f}")
    (OUT / "adfecgdb_transfer.json").write_text(json.dumps(out, indent=2))
    return out


def eval_new_checkpoints(device="mps"):
    """Score any equal-budget / ablation checkpoints that exist."""
    from eval_qrs import run as run_qrs
    cands = {
        "equal_mae": ROOT / "results" / "rev_equal_mae" / "final_model.pt",
        "equal_simclr": ROOT / "results" / "rev_equal_simclr" / "final_model.pt",
        "equal_vicreg": ROOT / "results" / "rev_equal_vicreg" / "final_model.pt",
        "mae_s7": ROOT / "results" / "rev_mae_s7" / "final_model.pt",
        "mae_s123": ROOT / "results" / "rev_mae_s123" / "final_model.pt",
        "mae_m25": ROOT / "results" / "rev_mae_m25" / "final_model.pt",
        "mae_m75": ROOT / "results" / "rev_mae_m75" / "final_model.pt",
        "physics": ROOT / "results" / "rev_physics" / "final_model.pt",
        "simclr_nonorm": ROOT / "results" / "rev_simclr_nonorm" / "final_model.pt",
        "mae_cnn": ROOT / "results" / "rev_mae_cnn" / "final_model.pt",
    }
    scored = {}
    dest = OUT / "new_ckpts"
    dest.mkdir(exist_ok=True)
    for name, ck in cands.items():
        if not ck.exists():
            continue
        print(f"  scoring new ckpt {name}")
        # Architecture-matched loaders for ablations.
        from eval_qrs import make_encoder as _me
        import eval_qrs as eq
        orig = eq.make_encoder
        if name == "simclr_nonorm":
            eq.make_encoder = lambda kind, ckpt, device: orig(
                kind, ckpt, device, input_norm="none")
        elif name == "mae_cnn":
            eq.make_encoder = lambda kind, ckpt, device: orig(
                kind, ckpt, device, use_transformer=False)
        try:
            r = run_qrs("ssl", str(ck), device, str(dest))
        finally:
            eq.make_encoder = orig
        scored[name] = {k: r[k] for k in ("f1_mean", "f1_std", "f1_median", "n_records")}
        (dest / f"qrs_{name}.json").write_text(json.dumps(r, indent=2))
    if scored:
        (OUT / "new_ckpt_qrs.json").write_text(json.dumps(scored, indent=2))
        print("  new ckpt QRS:", scored)
    else:
        print("  no new checkpoints yet")
    return scored


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--skip_slow", action="store_true",
                   help="Only JSON-level stats (no GPU feature extraction).")
    p.add_argument("--only_new", action="store_true",
                   help="Only score newly trained revision checkpoints.")
    args = p.parse_args()
    device = args.device

    if args.only_new:
        eval_new_checkpoints(device)
        return

    print("[1] pairwise statistics")
    stats_from_existing()
    print("[2] dataset inventory")
    dataset_inventory()
    if args.skip_slow:
        print("skipping GPU experiments")
        return
    print("[3] classical baseline + maternal FP")
    classical_and_maternal_fp(device)
    print("[4] amplitude perturbation")
    amplitude_perturbation(device)
    print("[5] QRS tolerances")
    qrs_tolerances(device)
    print("[6] few-shot repeats")
    fewshot_repeated(device, n_repeats=5)
    print("[7] ADfECGDB transfer")
    adfecgdb_transfer(device)
    print("[8] new checkpoints (if any)")
    eval_new_checkpoints(device)
    print("Done ->", OUT)


if __name__ == "__main__":
    main()
