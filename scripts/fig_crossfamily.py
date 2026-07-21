#!/usr/bin/env python3
"""Figure 5: verbatim extraction per quantizer, split into two regime
blocks. Left: full fine-tune (5 backbones). Right: LoRA r=16; the first
three LoRA cells share learning rate 2e-5 (matched small delta), the
rightmost is the same 3B model at lr 2e-4 (larger delta) -- the delta
knob. Rates are % of canaries; seed counts differ across cells
(tab:headline), so the two blocks are not compared bar-to-bar.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os as _os
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fig_data import crossfamily
ROOT = Path(_os.environ.get("QQUILT_REPO",
            Path(__file__).resolve().parent.parent))
FIGDIR = Path(_os.environ.get("QQUILT_FIGDIR", ROOT / "experiment" / "figures"))
FIGDIR.mkdir(parents=True, exist_ok=True)

# Values loaded from the committed logs (see scripts/_fig_data.py).
_d = crossfamily()
# --- full fine-tune block ---
ft_models = ["Qwen2.5\n0.5B", "Llama-3.2\n1B", "Qwen2.5\n1.5B",
             "Llama-3.2\n3B", "Qwen2.5\n7B"]
ft_bf16, ft_q4, ft_awq = _d["ft_bf16"], _d["ft_q4"], _d["ft_awq"]

# --- LoRA r=16 block; last cell is the lr 2e-4 (larger-delta) knob ---
lo_models = ["Qwen2.5\n0.5B", "Llama-3.2\n1B",
             "Llama-3.2\n3B", "Llama-3.2\n3B"]
lo_bf16, lo_q4, lo_awq = _d["lo_bf16"], _d["lo_q4"], _d["lo_awq"]

gap = 1.1
ft_x = np.arange(len(ft_models))
lo_x = np.arange(len(lo_models)) + len(ft_models) + gap
w = 0.27

fig, ax = plt.subplots(figsize=(10.6, 3.7))

def draw(xs, bf, q4, aw, lab=False):
    a = ax.bar(xs - w, bf, w, color="#7f7f7f", edgecolor="black",
               linewidth=0.6, label="BF16" if lab else None)
    b = ax.bar(xs,     q4, w, color="#d62728", edgecolor="black",
               linewidth=0.6, label="Q4_K_M" if lab else None)
    c = ax.bar(xs + w, aw, w, color="#1f77b4", edgecolor="black",
               linewidth=0.6, label="AWQ-4bit" if lab else None)
    for bars in (a, b, c):
        for bb in bars:
            h = bb.get_height()
            ax.text(bb.get_x() + bb.get_width()/2, h + 0.7,
                    f"{h:.1f}", ha="center", fontsize=7, fontweight="bold")

draw(ft_x, ft_bf16, ft_q4, ft_awq, lab=True)
draw(lo_x, lo_bf16, lo_q4, lo_awq)

# LoRA block tint + divider
ax.axvspan(lo_x[0] - 0.6, lo_x[-1] + 0.6, alpha=0.06, color="orange", zorder=0)
divider = (ft_x[-1] + lo_x[0]) / 2
ax.axvline(divider, color="#999999", linewidth=0.8, linestyle="--")

ax.text(np.mean(ft_x), 32.6, "Full fine-tune (lr 2e-5)", ha="center",
        fontsize=10, fontweight="bold")
ax.text(np.mean(lo_x), 32.6, "LoRA r=16", ha="center",
        fontsize=10, fontweight="bold", color="#a36800")

ax.set_xticks(list(ft_x) + list(lo_x))
ax.set_xticklabels(ft_models + lo_models, fontsize=8)

# big learning-rate sub-labels under the LoRA block
trans = ax.get_xaxis_transform()
x0, x1 = lo_x[0] - 0.42, lo_x[2] + 0.42   # the three lr 2e-5 cells
ax.plot([x0, x1], [-0.27, -0.27], transform=trans, color="#a36800",
        lw=1.4, clip_on=False)
ax.text((x0 + x1) / 2, -0.34, "lr $2\\times10^{-5}$", transform=trans,
        ha="center", va="top", fontsize=12, fontweight="bold", color="#a36800")
x2, x3 = lo_x[3] - 0.42, lo_x[3] + 0.42   # the lr 2e-4 knob cell
ax.plot([x2, x3], [-0.27, -0.27], transform=trans, color="#c0392b",
        lw=1.4, clip_on=False)
ax.text((x2 + x3) / 2, -0.34, "lr $2\\times10^{-4}$", transform=trans,
        ha="center", va="top", fontsize=12, fontweight="bold", color="#c0392b")
ax.set_ylabel("Extraction rate (%, greedy $\\geq$10 chars)", fontsize=9.5)
ax.set_ylim(0, 35)
ax.set_yticks([0, 10, 20, 30])
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.12),
          ncol=3, fontsize=9, frameon=False)
ax.grid(True, axis="y", alpha=0.3, linestyle=":")

plt.tight_layout()
plt.savefig(FIGDIR / "fig_crossfamily.pdf", bbox_inches="tight")
plt.savefig(FIGDIR / "fig_crossfamily.png", dpi=180, bbox_inches="tight")
print("saved fig_crossfamily.{pdf,png}")
