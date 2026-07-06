"""Generate honest figures for the manuscript from real result files."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.dpi": 300,
                     "axes.spmap" if False else "axes.grid": True,
                     "grid.alpha": 0.3})


def fig_extraction():
    df = pd.read_csv(RES / "extraction_per_record.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    order = ["ADfECGDB", "CinC2013"]
    data = [df[df.dataset == d]["f1"].dropna().values for d in order]
    bp = ax[0].boxplot(data, tick_labels=[f"{d}\n(n={len(x)})" for d, x in zip(order, data)],
                       patch_artist=True, showmeans=True)
    for patch, c in zip(bp["boxes"], ["#4C9F70", "#3B7DD8"]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    for i, x in enumerate(data, 1):
        ax[0].scatter(np.random.normal(i, 0.05, len(x)), x, s=12, c="k", alpha=0.4, zorder=3)
    ax[0].set_ylabel("Fetal QRS F1  (±50 ms tolerance)")
    ax[0].set_title("(a) Real fetal QRS detection performance")
    ax[0].set_ylim(0, 1.02)
    ax[0].axhline(np.median(data[0]), ls=":", c="#4C9F70", alpha=0.7)

    # ranked per-record F1 for CinC (shows the heavy tail of hard records)
    cinc = np.sort(df[df.dataset == "CinC2013"]["f1"].dropna().values)[::-1]
    ax[1].plot(np.arange(1, len(cinc) + 1), cinc, marker="o", ms=3, c="#3B7DD8")
    ax[1].axhline(np.median(cinc), ls="--", c="grey",
                  label=f"median = {np.median(cinc):.2f}")
    ax[1].set_xlabel("CinC 2013 record (ranked)")
    ax[1].set_ylabel("F1")
    ax[1].set_title("(b) Per-record F1 across 75 CinC 2013 records")
    ax[1].legend()
    ax[1].set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig_extraction_performance.png")
    plt.close(fig)
    print("wrote fig_extraction_performance.png")


def fig_ga_null():
    pred = pd.read_csv(RES / "ninfea_ga_predictions.csv")
    y = pred["ga_weeks"].values
    yhat = pred["pred_random_forest"].values
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].scatter(y, yhat, s=28, c="#C0504D", alpha=0.7, edgecolor="k", linewidth=0.3)
    lim = [20.5, 28]
    ax[0].plot(lim, lim, "--", c="grey", label="perfect prediction")
    ax[0].axhline(np.mean(y), ls=":", c="k", label=f"mean GA = {np.mean(y):.1f} wk")
    mae = np.mean(np.abs(yhat - y))
    r = np.corrcoef(yhat, y)[0, 1]
    ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel("Chronological GA (weeks)")
    ax[0].set_ylabel("Predicted GA (weeks)")
    ax[0].set_title(f"(a) NInFEA GA prediction (subject-grouped)\nMAE={mae:.1f} wk, r={r:+.2f} — no better than the mean")
    ax[0].legend(loc="upper left", fontsize=9)

    # permutation null histogram
    rep_path = RES / "robust_ga_report.json"
    if rep_path.exists():
        rep = json.loads(rep_path.read_text())
        pn = rep["full"]["permutation_null_rf"]
        # reconstruct a normal-ish null band for display from mean/std
        ax[1].axvspan(pn["null_mae_mean"] - pn["null_mae_std"],
                      pn["null_mae_mean"] + pn["null_mae_std"], color="grey", alpha=0.25,
                      label="null MAE (label-shuffled) ±1 SD")
        ax[1].axvline(pn["null_mae_mean"], c="grey", ls="--")
        ax[1].axvline(pn["observed_mae"], c="#C0504D", lw=2,
                      label=f"observed MAE = {pn['observed_mae']:.2f} wk")
        ax[1].set_xlabel("Leave-one-subject-out MAE (weeks)")
        ax[1].set_yticks([])
        ax[1].set_title(f"(b) Permutation test: p = {pn['p_value']:.2f}\nobserved sits inside the chance distribution")
        ax[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig_ga_null_result.png")
    plt.close(fig)
    print("wrote fig_ga_null_result.png")


def fig_leakage():
    """Illustrate how record-level CV can manufacture apparent signal."""
    df = pd.read_csv(RES / "ninfea_features.csv")
    counts = df.groupby("subject").size()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(counts.values, bins=[0.5, 1.5, 2.5, 3.5], rwidth=0.8, color="#7C5CBF", alpha=0.8)
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Recordings per participant")
    ax.set_ylabel("Number of participants")
    ax.set_title(f"{(counts>1).sum()}/{len(counts)} participants contribute >1 recording\n"
                 "→ record-level CV leaks identical GA labels")
    fig.tight_layout()
    fig.savefig(FIG / "fig_leakage_structure.png")
    plt.close(fig)
    print("wrote fig_leakage_structure.png")


if __name__ == "__main__":
    fig_extraction()
    fig_ga_null()
    fig_leakage()
    print("all figures written to", FIG)
