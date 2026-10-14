"""Generate the IEEE J-BHI graphical abstract (single-page PDF) from measured numbers."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "GraphicalAbstract.pdf"

C = {"random": "#9aa0a6", "simclr": "#f4a259", "vicreg": "#5aa469", "mae": "#2d6cdf"}

# pull measured frozen-probe F1 from the benchmark
bench = ROOT / "results" / "benchmark"
def f1(k):
    return json.load(open(bench / f"qrs_{k}.json"))["f1_mean"]
vals = {k: f1(k) for k in ["random", "simclr", "vicreg", "mae"]}

fig = plt.figure(figsize=(11, 5.2))
fig.suptitle("Masked signal modelling for non-invasive fetal ECG:\n"
             "what self-supervision buys at ~100-recording scale",
             fontsize=14, fontweight="bold", y=0.99)

# --- Left panel: frozen-probe F1 bar chart ---
axL = fig.add_axes([0.06, 0.12, 0.40, 0.70])
names = ["Random", "SimCLR", "VICReg", "Masked\nmodelling"]
keys = ["random", "simclr", "vicreg", "mae"]
bars = axL.bar(range(4), [vals[k] for k in keys],
               color=[C[k] for k in keys], edgecolor="black", lw=0.7)
for i, k in enumerate(keys):
    axL.text(i, vals[k] + 0.01, f"{vals[k]:.2f}", ha="center", fontweight="bold", fontsize=10)
axL.set_xticks(range(4)); axL.set_xticklabels(names, fontsize=9)
axL.set_ylim(0, 0.45); axL.set_ylabel(r"Fetal QRS $F_1$ (held-out CinC 2013)", fontsize=10)
axL.set_title("Frozen linear probe\n(record-grouped CV, $\\pm$50 ms)", fontsize=10)
axL.spines[["top", "right"]].set_visible(False)

# --- Right panel: the story in three steps ---
axR = fig.add_axes([0.50, 0.06, 0.47, 0.80]); axR.axis("off")
def box(y, title, body, fc):
    axR.add_patch(FancyBboxPatch((0.02, y), 0.96, 0.20, boxstyle="round,pad=0.012",
                 mutation_aspect=0.5, fc=fc, ec="black", lw=0.8, transform=axR.transAxes))
    axR.text(0.06, y + 0.145, title, fontsize=10.5, fontweight="bold", transform=axR.transAxes)
    axR.text(0.06, y + 0.055, body, fontsize=9, transform=axR.transAxes, va="center")

box(0.72, "1.  Naive contrastive SSL collapses",
    "embedding effective rank 2.5 of 64  →  worse than random", "#fdecea")
box(0.40, "2.  Per-window instance norm fixes it",
    "rank recovers to ~10; masked modelling beats random\n"
    "+0.055 $F_1$ (p<1e-7); 5% of labels ≈ random with 100%", "#e8f0fe")
box(0.08, "3.  But full fine-tuning erases the gap",
    "MAE-init 0.50 vs random-init 0.50 (p=0.53):\n"
    "pre-training buys label efficiency, not a higher ceiling", "#eef7ee")
for y0, y1 in [(0.72, 0.60), (0.40, 0.28)]:
    axR.add_patch(FancyArrowPatch((0.5, y0), (0.5, y1), transform=axR.transAxes,
                  arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#444"))

fig.text(0.5, 0.015, "First reproducible self-supervised benchmark for fetal ECG  ·  "
         "code: github.com/kamuthukumar/fetal-ecg-ssl",
         ha="center", fontsize=8.5, style="italic", color="#333")

fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.with_suffix(".png"), dpi=150, bbox_inches="tight")
print("wrote", OUT, "and", OUT.with_suffix(".png"))
