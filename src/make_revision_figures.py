"""Revision figures: axes labelled, units stated, 600 dpi."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "results" / "revision"
BENCH = ROOT / "results" / "benchmark"
OUTS = [
    ROOT / "results" / "real_figs",
    ROOT.parent / "images",
    ROOT.parent / "submission_jbhi" / "images",
    ROOT.parent / "submission_jbhi" / "revision" / "images",
]
for d in OUTS:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "axes.linewidth": 1.0, "figure.dpi": 150, "savefig.dpi": 600,
    "font.family": "DejaVu Sans",
})

C = {"random": "#7a7a7a", "simclr": "#e07a3d", "vicreg": "#3d8c5e",
     "mae": "#2c5ea8", "classical": "#6b3fa0"}
L = {"random": "Random init", "simclr": "SimCLR", "vicreg": "VICReg",
     "mae": "Masked modelling", "classical": "Classical extractor"}


def _load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def _save(fig, name):
    for d in OUTS:
        fig.savefig(d / name, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(" ->", name)


def fig_qrs_clean():
    variants, means, sds, pts = [], [], [], []
    for k in ("random", "simclr", "vicreg", "mae"):
        d = _load(BENCH / f"qrs_{k}.json")
        if not d:
            continue
        variants.append(k); means.append(d["f1_mean"]); sds.append(d["f1_std"])
        pts.append(list(d["per_record"].values()))
    if not variants:
        return
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    x = np.arange(len(variants))
    ax.bar(x, means, yerr=sds, capsize=4, color=[C[k] for k in variants],
           edgecolor="black", linewidth=0.7, alpha=0.9, error_kw={"linewidth": 1.0})
    rng = np.random.default_rng(0)
    for i, s in enumerate(pts):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(s)), s, s=9,
                   color="black", alpha=0.28, zorder=3, linewidths=0)
    ax.set_xticks(x)
    ax.set_xticklabels([L[k] for k in variants], rotation=18, ha="right")
    ax.set_ylabel(r"Fetal QRS $F_1$ (dimensionless)")
    ax.set_xlabel("Pre-training objective")
    ax.set_ylim(0, 1.05)
    ax.set_title("Held-out CinC 2013, record-grouped CV, matching tolerance = ±50 ms")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "fig_qrs_real.png")


def fig_tolerances():
    d = _load(REV / "qrs_tolerances.json")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    tols = [20, 30, 50, 100]
    for k in ("random", "simclr", "vicreg", "mae"):
        if k not in d:
            continue
        y = [d[k][str(t)]["f1_mean"] for t in tols]
        e = [d[k][str(t)]["f1_std"] for t in tols]
        ax.errorbar(tols, y, yerr=e, marker="o", color=C[k], label=L[k], capsize=3)
    ax.set_xlabel("Matching tolerance (ms)")
    ax.set_ylabel(r"Fetal QRS $F_1$ (dimensionless)")
    ax.set_xticks(tols)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False)
    ax.set_title("Sensitivity of QRS $F_1$ to the matching window")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "fig_tolerance.png")


def fig_fewshot():
    d = _load(REV / "fewshot_repeated.json") or _load(BENCH / "fewshot.json")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for k in ("random", "mae"):
        block = d.get(k)
        if not block or "mean" not in block:
            continue
        fr = np.array(block.get("fractions") or d.get("fractions")) * 100
        m = np.array(block["mean"]); s = np.array(block["std"])
        ax.plot(fr, m, "-o", color=C[k], label=L[k])
        ax.fill_between(fr, m - s, m + s, color=C[k], alpha=0.18)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 25, 50, 100])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Labelled training records (% of the training fold)")
    ax.set_ylabel(r"Fetal QRS $F_1$ (dimensionless)")
    ax.set_ylim(0.15, 0.45)
    ax.legend(frameon=False)
    ax.set_title("Few-shot frozen probe, CinC 2013")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle=":")
    _save(fig, "fig_fewshot_real.png")


def fig_amplitude():
    d = _load(REV / "amplitude_causal.json")
    if not d:
        return
    style = {"simclr_no_norm": ("#b23a3a", "SimCLR, no instance norm (collapsed)"),
             "simclr_instance_norm": ("#e07a3d", "SimCLR, instance norm")}
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for k, block in d.items():
        disp = block.get("scale_displacement_rel_spread", {})
        xs = np.array([float(a) for a in disp])
        ys = np.array(list(disp.values()))
        order = np.argsort(xs)
        col, lab = style.get(k, ("black", k))
        ax.plot(xs[order], ys[order], "-o", color=col, label=lab)
    ax.axhline(1.0, color="black", lw=0.6, ls="--")
    ax.text(0.27, 1.06, "mean distance of a window from the centroid", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Amplitude scale factor $\\alpha$ (unitless)")
    ax.set_ylabel("Relative embedding shift (unitless)")
    ax.legend(frameon=False, loc="upper center")
    ax.set_title("Effect of input gain on frozen embeddings")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "fig_amplitude.png")


def fig_baselines():
    q = _load(BENCH / "qrs_mae.json")
    cl = _load(REV / "classical_baseline.json")
    ft = _load(BENCH / "finetune_mae.json")
    if not q:
        return
    rnd = _load(BENCH / "qrs_random.json")
    ftr = _load(BENCH / "finetune_random.json")
    db = (_load(REV / "direct_baselines.json") or {}).get("results", {})
    rows = [  # (label, summary, group colour)
        ("Random init, frozen probe", rnd, "#9a9a9a"),
        ("Masked modelling, frozen probe", q, "#2c5ea8"),
        ("FetalECGFM from scratch", ftr, "#1b3f73"),
        ("FetalECGFM, masked init, fine-tuned", ft, "#1b3f73"),
        ("1-D U-Net (supervised)", db.get("unet"), "#c0504d"),
        ("MCG-Net-style (supervised)", db.get("mcgnet"), "#c0504d"),
        ("FastICA (unsupervised)", db.get("ica"), "#6b3fa0"),
        ("TS + FastICA (unsupervised)", db.get("ts_ica"), "#6b3fa0"),
        ("TS + channel selection (unsup.)", cl.get("classical") if cl else None, "#6b3fa0"),
    ]
    rows = [r for r in rows if r[1]]
    names = [r[0] for r in rows]
    vals = [r[1]["f1_mean"] for r in rows]
    errs = [r[1]["f1_std"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, xerr=errs, capsize=3, color=[r[2] for r in rows],
            edgecolor="black", linewidth=0.6)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(r"Fetal QRS $F_1$ (dimensionless, ±50 ms)")
    ax.set_xlim(0, 1.12)
    for yi, v in zip(y, vals):
        ax.text(min(v + 0.02, 1.0), yi + 0.28, f"{v:.2f}", fontsize=8.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2c5ea8", label="SSL representation test"),
                       Patch(color="#1b3f73", label="Same encoder, supervised"),
                       Patch(color="#c0504d", label="Supervised delineation nets"),
                       Patch(color="#6b3fa0", label="Classical multi-channel")],
              loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2,
              fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, "fig_baselines.png")


def fig_ga_and_emb():
    # regenerate with axis units
    ks, sil = [], []
    for k in ("random", "simclr", "vicreg", "mae"):
        d = _load(BENCH / f"structure_{k}.json")
        if d is None or "ga_structure" not in d:
            continue
        ks.append(k); sil.append(d["ga_structure"]["silhouette"])
        p = d["ga_structure"]["p_perm"]
    if ks:
        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        x = np.arange(len(ks))
        ax.bar(x, sil, color=[C[k] for k in ks], edgecolor="black", linewidth=0.7)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([L[k] for k in ks], rotation=18, ha="right")
        ax.set_ylabel("Silhouette coefficient (unitless)")
        ax.set_xlabel("Encoder")
        ax.set_title("Gestational-age tertile clustering, NInFEA")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _save(fig, "fig_ga_structure_real.png")

    summ = _load(BENCH / "summary.json")
    best = (summ or {}).get("best_method") or "mae"
    emb = BENCH / f"emb_{best}.npy"
    meta = BENCH / f"meta_{best}.json"
    if emb.exists() and meta.exists():
        X = np.load(emb); M = json.loads(meta.read_text())
        Xc = X - X.mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        Z = Xc @ Vt[:2].T
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
        ds = np.array([m["dataset"] for m in M])
        for d0 in np.unique(ds):
            mk = ds == d0
            axes[0].scatter(Z[mk, 0], Z[mk, 1], s=18, alpha=0.75, label=d0)
        axes[0].legend(fontsize=8, frameon=False)
        axes[0].set_title("Record embeddings by acquisition dataset")
        ga = np.array([m["ga"] if m["ga"] is not None else np.nan for m in M])
        mk = ~np.isnan(ga)
        sc = axes[1].scatter(Z[mk, 0], Z[mk, 1], c=ga[mk], cmap="viridis", s=20)
        cb = plt.colorbar(sc, ax=axes[1])
        cb.set_label("Gestational age (weeks)")
        axes[1].set_title("NInFEA embeddings by gestational age")
        for a in axes:
            a.set_xlabel("Principal component 1 (arbitrary units)")
            a.set_ylabel("Principal component 2 (arbitrary units)")
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
        fig.tight_layout()
        _save(fig, "fig_embedding_real.png")


if __name__ == "__main__":
    fig_qrs_clean()
    fig_ga_and_emb()
    fig_fewshot()
    fig_tolerances()
    fig_amplitude()
    fig_baselines()
    print("revision figures done")
