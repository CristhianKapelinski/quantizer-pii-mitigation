#!/usr/bin/env python3
"""Figure 2: Three-factor mechanism diagram (reviewer M9).

A schematic that makes Section 5.5 readable in one glance:
  Factor 1: rare-token noise concentration (universal across 4-bit)
  Factor 2: moderate-confidence vulnerability window (RECALL vs BODY)
  Factor 3: calibration amplifies noise magnitude (AWQ > Q4_K_M)
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import os as _os
ROOT = Path(_os.environ.get("QQUILT_REPO",
            Path(__file__).resolve().parent.parent))
FIGDIR = ROOT / "experiment" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.5), constrained_layout=True)

# =========================================================================
# Factor 1: rare-token noise concentration
# =========================================================================
ax = axes[0]
ax.set_title("Factor 1: rare-token noise", fontsize=9.5, pad=4)

# A schematic: residual error vector aligned (or not) with top-1 token basis
# show cosine alignments
labels = ["AWQ\n@ Enron", "Q4_K_M\n@ Enron", "AWQ\n@ RECALL", "Q4_K_M\n@ RECALL"]
cos_vals = [0.00038, 0.00086, 0.0086, 0.0079]
isotropic = 1/np.sqrt(128256)  # ~0.0028 for the Llama-3.2 vocab
colors = ["#bbbbbb", "#bbbbbb", "#1f77b4", "#d62728"]
bars = ax.bar(range(4), cos_vals, color=colors, edgecolor="black", linewidth=0.6)
ax.axhline(isotropic, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
ax.text(3.4, isotropic*1.15, "isotropic\nexpected", fontsize=7.5, color="#444",
        ha="right", va="bottom")
ax.set_xticks(range(4))
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel(r"$\cos(\mathbf{d}, \mathbf{e}_\mathrm{top1})$", fontsize=9.5)
ax.set_ylim(0, 0.012)
ax.text(0.5, 0.010, "both 4-bit methods\nconcentrate noise on\nmemorized rare tokens",
        fontsize=8, ha="center", color="#333",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fffae0", ec="#d4af37", lw=0.6))
ax.grid(True, axis="y", alpha=0.3, linestyle=":")

# =========================================================================
# Factor 2: moderate-confidence vulnerability window
# =========================================================================
ax = axes[1]
ax.set_title("Factor 2: confidence window", fontsize=9.5, pad=4)

# x = ft top-1 probability; y = flip rate (3-seed pool, n=300)
positions = [
    ("Enron",         0.55, 24, "#7f7f7f"),
    ("RECALL\n(AWQ)", 0.67, 78, "#1f77b4"),
    ("RECALL\n(Q4_K_M)", 0.70, 48, "#d62728"),
    ("BODY",          0.9998, 0, "#2ca02c"),
]
xs = [p[1] for p in positions]
ys = [p[2] for p in positions]
clr = [p[3] for p in positions]
for (lab, x, y, c) in positions:
    ax.scatter(x, y, s=140, color=c, edgecolor="black", linewidth=0.7, zorder=3)
    if "BODY" in lab:
        ax.annotate(lab, (x, y), xytext=(-12, 12), textcoords="offset points",
                    fontsize=8.5, ha="right")
    elif "Enron" in lab:
        ax.annotate(lab, (x, y), xytext=(8, 0), textcoords="offset points",
                    fontsize=8.5)
    else:
        ax.annotate(lab, (x, y), xytext=(8, -2), textcoords="offset points",
                    fontsize=8.5)

# moderate-confidence shaded band
ax.axvspan(0.55, 0.78, alpha=0.10, color="orange")
ax.text(0.665, 112, "moderate-confidence\nwindow", fontsize=8.5, ha="center",
        va="center", color="#a36800", style="italic")
ax.set_xlabel("FT top-1 probability", fontsize=9.5)
ax.set_ylabel("Top-1 flip rate (%)", fontsize=9.5)
ax.set_xlim(0.45, 1.05)
ax.set_ylim(-8, 124)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.grid(True, alpha=0.3, linestyle=":")

# =========================================================================
# Factor 3: calibration amplifies noise magnitude
# =========================================================================
ax = axes[2]
ax.set_title("Factor 3: calibration amplifies", fontsize=9.5, pad=4)

methods = ["Q4_K_M\n(no calib.)", "AWQ\n(calib.)", "GPTQ\n(calib.)"]
norms = [617, 841, None]   # 3-seed pool
extract = [5.3, 0.0, 0.0]  # 3-seed pool rates (matches the 3-seed norms)

x = np.arange(3)
width = 0.35

# Bars for ||d||_2 (left axis)
bars1 = ax.bar(x - width/2, [n if n is not None else 0 for n in norms], width,
               label=r"$\|\mathbf{d}\|_2$ at RECALL",
               color=["#d62728", "#1f77b4", "#2ca02c"],
               alpha=0.7, edgecolor="black", linewidth=0.6)

# annotate norms
for i, n in enumerate(norms):
    if n is not None:
        ax.text(x[i] - width/2, n + 25, f"{n}", ha="center", fontsize=8.5)
    else:
        ax.text(x[i] - width/2, 50, "n/a", ha="center", fontsize=8, color="#888")

ax.set_ylabel(r"Logit error norm $\|\mathbf{d}\|_2$", fontsize=9.5)
ax.set_ylim(0, 1100)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=8.5)

# Twin axis for extraction rate
ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, extract, width, label="Extraction rate (%)",
                color="#444444", alpha=0.85, edgecolor="black", linewidth=0.6)
for i, e in enumerate(extract):
    ax2.text(x[i] + width/2, e + 0.35, f"{e:.1f}%", ha="center", fontsize=8.5)
ax2.set_ylabel("Extraction rate (%)", fontsize=9.5, color="#222")
ax2.set_ylim(0, 8)

# bracket showing calibration -> +36% norm
ax.annotate("",
            xy=(x[1] - width/2, 870), xytext=(x[0] - width/2, 650),
            arrowprops=dict(arrowstyle="->", color="#a36800", lw=1.5))
ax.text((x[0] + x[1])/2 - width/2, 980, "+36% norm",
        fontsize=8.5, ha="center", color="#a36800",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#a36800", lw=0.6))

# Legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=6.8,
          framealpha=0.95)

ax.grid(True, axis="y", alpha=0.3, linestyle=":")

out_pdf = FIGDIR / "fig_mechanism.pdf"
out_png = FIGDIR / "fig_mechanism.png"
plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_png, dpi=180, bbox_inches="tight")
print(f"saved: {out_pdf}")
print(f"saved: {out_png}")
