"""Full benchmark across all pre-training methods.

Evaluates every available encoder on: held-out QRS detection (record-grouped CV),
embedding structure (GA + hardware, permutation nulls), effective rank (collapse
diagnostic), and a random-vs-best few-shot curve. Writes one summary JSON.

Methods and their checkpoints:
    random  : untrained encoder (no checkpoint)
    simclr  : results/foundation_simclr/final_model.pt
    vicreg  : results/foundation_vicreg/final_model.pt
    mae     : results/foundation_mae/final_model.pt
    (ssl/contrastive: results/foundation/final_model.pt, if present)
"""
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
OUT = ROOT / "results" / "benchmark"

CKPTS = {
    "random": None,
    "simclr": ROOT / "results" / "foundation_simclr" / "final_model.pt",
    "vicreg": ROOT / "results" / "foundation_vicreg" / "final_model.pt",
    "mae": ROOT / "results" / "foundation_mae" / "final_model.pt",
}


def effective_rank(X):
    X = X[np.all(np.isfinite(X), axis=1)]
    Xc = X - X.mean(0)
    ev = np.linalg.eigvalsh(np.cov(Xc.T))
    ev = ev[ev > 1e-12]
    p = ev / ev.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def main(device="mps"):
    from eval_qrs import run as run_qrs, make_encoder, few_shot
    from eval_structure import evaluate as run_struct
    from scipy.stats import wilcoxon
    import torch

    OUT.mkdir(parents=True, exist_ok=True)
    variants = [(k, str(v)) for k, v in CKPTS.items()
                if v is None or Path(v).exists()]
    print("Evaluating:", [k for k, _ in variants])

    qrs, struct, eff = {}, {}, {}
    for kind, ckpt in variants:
        # map non-standard kinds to loader kind ('random' has no ckpt; others load)
        loader_kind = "random" if kind == "random" else "ssl"
        qrs[kind] = _run_qrs_named(kind, loader_kind, ckpt, device)
        struct[kind] = _run_struct_named(kind, loader_kind, ckpt, device)
        ep = OUT / f"emb_{kind}.npy"
        if ep.exists():
            eff[kind] = effective_rank(np.load(ep))

    # Paired Wilcoxon vs random on matched per-record F1.
    stats = {}
    if "random" in qrs:
        for kind in qrs:
            if kind == "random":
                continue
            recs = sorted(set(qrs["random"]["per_record"]) & set(qrs[kind]["per_record"]))
            a = np.array([qrs["random"]["per_record"][r] for r in recs])
            b = np.array([qrs[kind]["per_record"][r] for r in recs])
            s = {"delta_f1_vs_random": float(b.mean() - a.mean())}
            try:
                s["p_greater_than_random"] = float(wilcoxon(b, a, alternative="greater").pvalue)
            except ValueError:
                s["p_greater_than_random"] = None
            stats[kind] = s

    # Few-shot: random vs the best non-random method by mean F1.
    best = max((k for k in qrs if k != "random"),
               key=lambda k: qrs[k]["f1_mean"], default=None)
    fewshot = {"fractions": None}
    if best:
        for kind in ("random", best):
            ck = None if kind == "random" else dict(variants)[kind]
            fs = few_shot(kind if kind == "random" else "ssl", ck, device_str=device)
            fewshot["fractions"] = fs["fractions"]
            fewshot["random" if kind == "random" else best] = {"mean": fs["mean"], "std": fs["std"]}
        fewshot["best_method"] = best
    (OUT / "fewshot.json").write_text(json.dumps(fewshot, indent=2))

    summary = {
        "qrs_detection": {k: {kk: v[kk] for kk in
                              ("f1_mean", "f1_std", "f1_median", "frac_below_0p5", "n_records")}
                          for k, v in qrs.items()},
        "effective_rank": eff,
        "ga_silhouette": {k: struct[k].get("ga_structure", {}).get("silhouette") for k in struct},
        "ga_p_perm": {k: struct[k].get("ga_structure", {}).get("p_perm") for k in struct},
        "hardware_probe_acc": {k: struct[k].get("hardware", {}).get("linear_probe_acc") for k in struct},
        "statistics_vs_random": stats,
        "best_method": best,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n==================== BENCHMARK SUMMARY ====================")
    print(json.dumps(summary, indent=2))


def _run_qrs_named(name, loader_kind, ckpt, device):
    """Run QRS eval, writing results under the method's own name."""
    from eval_qrs import run as run_qrs
    r = run_qrs(loader_kind, ckpt, device, str(OUT))
    # rename the output file to the method name
    src = OUT / f"qrs_{loader_kind}.json"
    dst = OUT / f"qrs_{name}.json"
    if src.exists() and src != dst:
        src.replace(dst)
    r["encoder"] = name
    dst.write_text(json.dumps(r, indent=2))
    return r


def _run_struct_named(name, loader_kind, ckpt, device):
    from eval_structure import evaluate as run_struct
    r = run_struct(loader_kind, ckpt, device, str(OUT))
    for stem in ("structure", "emb", "meta"):
        ext = "npy" if stem == "emb" else "json"
        src = OUT / f"{stem}_{loader_kind}.{ext}"
        dst = OUT / f"{stem}_{name}.{ext}"
        if src.exists() and src != dst:
            src.replace(dst)
    r["encoder"] = name
    (OUT / f"structure_{name}.json").write_text(json.dumps(r, indent=2))
    return r


if __name__ == "__main__":
    import os
    main(device=os.environ.get("DEVICE", "mps"))
