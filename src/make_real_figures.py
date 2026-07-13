"""Generate manuscript figures strictly from measured results.

Reads only JSON/NPY produced by the benchmark (results/benchmark/, results/qrs_bench/,
results/structure/) — no synthetic or hard-coded numbers. Missing inputs are
skipped with a warning so the script degrades gracefully.

Outputs to results/real_figs/ and copies into ../images/ for the manuscript.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
BENCH = ROOT / "results" / "benchmark"
OUT = ROOT / "results" / "real_figs"
IMG = ROOT.parent / "images"
OUT.mkdir(parents=True, exist_ok=True)

C = {"random": "#9aa0a6", "simclr": "#f4a259", "ssl": "#2d6cdf",
     "vicreg": "#5aa469", "mae": "#2d6cdf", "contrastive": "#c05780"}
LABEL = {"random": "Random init", "simclr": "SimCLR", "ssl": "Contrastive",
         "contrastive": "Contrastive (GA-aware)", "vicreg": "VICReg",
         "mae": "Masked modelling (ours)"}
# Method plotting order (only those present are drawn).
ORDER = ["random", "simclr", "contrastive", "vicreg", "mae"]


def _load(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def _present(prefix, ext="json"):
    """Methods that actually have a results file, in ORDER."""
    return [k for k in ORDER if (BENCH / f"{prefix}_{k}.{ext}").exists()]


def fig_qrs_bars():
    """Per-encoder QRS-detection F1 (mean +/- SD) with per-record scatter."""
    variants, means, sds, scatter = [], [], [], []
    for k in _present("qrs"):
        d = _load(BENCH / f"qrs_{k}.json")
        if d is None:
            continue
        variants.append(k); means.append(d["f1_mean"]); sds.append(d["f1_std"])
        scatter.append(list(d["per_record"].values()))
    if not variants:
        print("  (no QRS results yet)"); return
    fig, ax = plt.subplots(figsize=(5.2, 4))
    x = np.arange(len(variants))
    ax.bar(x, means, yerr=sds, capsize=5,
           color=[C[v] for v in variants], alpha=0.85, edgecolor="black", lw=0.6)
    for i, s in enumerate(scatter):
        ax.scatter(np.full(len(s), x[i]) + np.random.uniform(-0.12, 0.12, len(s)),
                   s, s=10, color="black", alpha=0.35, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[v] for v in variants], rotation=15, ha="right")
    ax.set_ylabel(r"Fetal QRS $F_1$ (±50 ms, record-grouped CV)")
    ax.set_ylim(0, 1); ax.set_title("Fetal QRS detection on held-out CinC 2013")
    for i, (m, s) in enumerate(zip(means, sds)):
        ax.text(x[i], m + s + 0.02, f"{m:.2f}", ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    for d in (OUT, IMG):
        fig.savefig(d / "fig_qrs_real.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("  -> fig_qrs_real.png")


def fig_ga_structure():
    """GA-bin silhouette: random vs ssl (with permutation p)."""
    ks, sil, ps = [], [], []
    for k in _present("structure"):
        d = _load(BENCH / f"structure_{k}.json")
        if d is None or "ga_structure" not in d or d["ga_structure"]["silhouette"] is None:
            continue
        ks.append(k); sil.append(d["ga_structure"]["silhouette"]); ps.append(d["ga_structure"]["p_perm"])
    if not ks:
        print("  (no structure results yet)"); return
    fig, ax = plt.subplots(figsize=(4.6, 4))
    x = np.arange(len(ks))
    ax.bar(x, sil, color=[C[k] for k in ks], alpha=0.85, edgecolor="black", lw=0.6)
    ax.axhline(0, color="black", lw=0.8)
    for i, (s, p) in enumerate(zip(sil, ps)):
        ax.text(x[i], s + (0.005 if s >= 0 else -0.02),
                f"p={p:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[k] for k in ks], rotation=15, ha="right")
    ax.set_ylabel("GA-bin silhouette (NInFEA)")
    ax.set_title("Gestational-age structure in embeddings")
    fig.tight_layout()
    for d in (OUT, IMG):
        fig.savefig(d / "fig_ga_structure_real.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("  -> fig_ga_structure_real.png")


def fig_fewshot():
    d = _load(BENCH / "fewshot.json")
    if d is None:
        print("  (no few-shot results yet)"); return
    fig, ax = plt.subplots(figsize=(5, 4))
    fr = np.array(d["fractions"]) * 100
    keys = [k for k in d if isinstance(d[k], dict) and "mean" in d[k]]
    for k in keys:
        m = np.array(d[k]["mean"]); s = np.array(d[k]["std"])
        ax.plot(fr, m, "-o", color=C.get(k, "#2d6cdf"), label=LABEL.get(k, k))
        ax.fill_between(fr, m - s, m + s, color=C.get(k, "#2d6cdf"), alpha=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("Labelled training records (%)")
    ax.set_ylabel(r"Fetal QRS $F_1$")
    ax.set_title("Few-shot fetal QRS detection (CinC 2013)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    for dd in (OUT, IMG):
        fig.savefig(dd / "fig_fewshot_real.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("  -> fig_fewshot_real.png")


def fig_embedding_umap():
    """2-D PCA of record embeddings for the best method, by dataset then GA."""
    summ = _load(BENCH / "summary.json")
    best = (summ or {}).get("best_method") or "mae"
    emb = BENCH / f"emb_{best}.npy"
    meta = BENCH / f"meta_{best}.json"
    if not emb.exists() or not meta.exists():
        print("  (no embeddings yet)"); return
    X = np.load(emb); M = json.loads(Path(meta).read_text())
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:2].T
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    ds = np.array([m["dataset"] for m in M])
    for d0 in np.unique(ds):
        mk = ds == d0
        axes[0].scatter(Z[mk, 0], Z[mk, 1], s=18, alpha=0.7, label=d0)
    axes[0].legend(fontsize=7); axes[0].set_title("By dataset (hardware)")
    ga = np.array([m["ga"] if m["ga"] is not None else np.nan for m in M])
    mk = ~np.isnan(ga)
    sc = axes[1].scatter(Z[mk, 0], Z[mk, 1], c=ga[mk], cmap="viridis", s=20)
    plt.colorbar(sc, ax=axes[1], label="GA (weeks)")
    axes[1].set_title("NInFEA by gestational age")
    for a in axes:
        a.set_xlabel("PC1"); a.set_ylabel("PC2")
    fig.tight_layout()
    for d in (OUT, IMG):
        fig.savefig(d / "fig_embedding_real.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("  -> fig_embedding_real.png")


if __name__ == "__main__":
    print("Generating figures from measured results:")
    fig_qrs_bars()
    fig_ga_structure()
    fig_fewshot()
    fig_embedding_umap()
    print(f"Done -> {OUT}")
