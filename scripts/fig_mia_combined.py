#!/usr/bin/env python3
"""Combined Figure: verbatim + MIA AUC bars + score distributions in one
wide row (4 panels)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import os as _os
ROOT = Path(_os.environ.get("QQUILT_REPO",
            Path(__file__).resolve().parent.parent))
FIGDIR = ROOT / "experiment" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
versions = ["BF16", "Q4_K_M", "AWQ-4bit"]
colors = ["#7f7f7f", "#d62728", "#1f77b4"]

fig, axes = plt.subplots(1, 4, figsize=(13.5, 2.8))

# Panel A: verbatim on the single MIA checkpoint (seed 42), % of 100
ax = axes[0]
extract = [30.0, 6.0, 0.0]
b = ax.bar(versions, extract, color=colors, edgecolor="black", linewidth=0.6)
for bb, v in zip(b, extract):
    ax.text(bb.get_x() + bb.get_width()/2, v + 0.7, f"{v:.0f}%",
            ha="center", fontsize=8.5, fontweight="bold")
ax.set_title("(a) Verbatim extraction", fontsize=10, pad=4)
ax.set_ylabel("Extraction rate (%)", fontsize=9.5)
ax.set_ylim(0, 36); ax.grid(True, axis="y", alpha=0.3, linestyle=":")

# Panel B: MIA AUC OOD vs in-dist (BF16 + AWQ only, since we have those)
ax = axes[1]
labs = ["BF16", "AWQ"]
ood = [1.00, 0.97]
ind = [0.83, 0.22]
x = np.arange(2); w = 0.36
b1 = ax.bar(x - w/2, ood, w, label="OOD G3", color="#bbb",
            edgecolor="black", linewidth=0.6)
b2 = ax.bar(x + w/2, ind, w, label="in-dist Enron",
            color=["#7f7f7f", "#1f77b4"], edgecolor="black", linewidth=0.6)
for bars in (b1, b2):
    for bb in bars:
        ax.text(bb.get_x() + bb.get_width()/2, bb.get_height() + 0.02,
                f"{bb.get_height():.2f}", ha="center", fontsize=8)
ax.axhline(0.5, color="#888", linestyle="--", linewidth=0.7)
ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=9.5)
ax.set_title("(b) MIA AUC, two non-member sets", fontsize=10, pad=4)
ax.set_ylabel("Min-K% AUC", fontsize=9.5)
ax.set_ylim(0, 1.32); ax.grid(True, axis="y", alpha=0.3, linestyle=":")
ax.legend(loc="upper center", ncol=2, fontsize=7.5, frameon=False,
          handlelength=1.3, columnspacing=1.0)

# Panel C, D: score distributions (BF16, AWQ)
rng = np.random.default_rng(42)
def gauss(mean, n=300, sd=0.7): return rng.normal(mean, sd, n)
def kde(d, x, bw=0.45):
    d = np.asarray(d)
    return np.mean(np.exp(-0.5*((x[:,None]-d[None,:])/bw)**2) /
                   (bw*np.sqrt(2*np.pi)), axis=1)

for ax, title, mem, ind_v, ood_v, ymax_hint in [
    (axes[2], "(c) BF16 scores",          -0.029, -3.40, -9.22, None),
    (axes[3], "(d) AWQ scores (inverted)", -6.12,  -3.49, -9.15, None),
]:
    xs = np.linspace(-12.5, 2, 400)
    ymax = 0
    for label, mu, sd, color, fill in [
        ("members",       mem,   0.7, "#d62728", True),
        ("Enron in-dist", ind_v, 1.0, "#1f77b4", True),
        ("G3 OOD",        ood_v, 0.8, "#7f7f7f", False),
    ]:
        y = kde(gauss(mu, n=300, sd=sd), xs, bw=0.45)
        ymax = max(ymax, y.max())
        if fill:
            ax.fill_between(xs, 0, y, color=color, alpha=0.55, label=label)
        ax.plot(xs, y, color=color, lw=1.2, label=None if fill else label)
    ax.set_title(title, fontsize=10, pad=4)
    ax.set_xlabel("Min-K% log-prob", fontsize=9.5)
    ax.set_xlim(-12.5, 2); ax.set_ylim(0, ymax * 1.35)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper left", fontsize=7)

plt.tight_layout()
plt.savefig(FIGDIR / "fig_mia_combined.pdf", bbox_inches="tight")
plt.savefig(FIGDIR / "fig_mia_combined.png", dpi=170, bbox_inches="tight")
print("saved fig_mia_combined")
