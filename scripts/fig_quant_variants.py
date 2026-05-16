#!/usr/bin/env python3
"""Figure: quantisation variants on the effective bits-per-weight axis.

Replaces Table 2 (quantisation variants studied) with a one-row strip plot.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _os
ROOT = Path(_os.environ.get("QQUILT_REPO",
            Path(__file__).resolve().parent.parent))
FIGDIR = ROOT / "experiment" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

variants = [
    ("GGUF", "Q2_K",     2.6, "#d62728"),
    ("GGUF", "Q3_K_M",   3.4, "#d62728"),
    ("GGUF", "Q4_K_S",   4.3, "#d62728"),
    ("GGUF", "Q4_K_M",   4.7, "#d62728"),
    ("GGUF", "Q5_K_M",   5.5, "#d62728"),
    ("GGUF", "Q8_0",     8.5, "#d62728"),
    ("AWQ",  "g128",     4.25, "#1f77b4"),
    ("AWQ",  "g64",      4.5,  "#1f77b4"),
    ("AWQ",  "g32",      5.0,  "#1f77b4"),
    ("GPTQ", "g128",     4.25, "#2ca02c"),
]

method_row = {"GGUF": 2, "AWQ": 1, "GPTQ": 0}
method_marker = {"GGUF": "o", "AWQ": "*", "GPTQ": "D"}
method_size   = {"GGUF": 9,   "AWQ": 18,  "GPTQ": 10}

fig, ax = plt.subplots(figsize=(7.5, 2.4))

# Per-variant label placement (above-or-below each marker) chosen to avoid
# overlap with neighbours and with the row labels of adjacent methods.
label_offsets = {
    "Q2_K":    (0,     0.30),
    "Q3_K_M":  (0,     0.30),
    "Q4_K_S":  (0,    -0.30),  # below marker -> separates from Q3_K_M and Q4_K_M
    "Q4_K_M":  (0,     0.30),
    "Q5_K_M":  (0,    -0.30),  # below marker -> separates from Q4_K_M
    "Q8_0":    (0,    -0.30),
    "g128":    (-0.30, 0.00),
    "g64":     (0,    -0.32),
    "g32":     (0,    -0.32),
}
seen_methods = set()
for m, v, bpw, c in variants:
    y = method_row[m]
    lbl_m = m if m not in seen_methods else None
    seen_methods.add(m)
    ax.scatter(bpw, y, s=120 if method_marker[m] != "*" else 220,
               marker=method_marker[m], color=c,
               edgecolor="black", linewidth=0.5, zorder=3, label=lbl_m)
    dx, dy = label_offsets.get(v, (0, 0.30))
    # g128 used twice: ensure GPTQ-g128 stays close to its marker
    if v == "g128" and m == "GPTQ":
        dx, dy = 0.30, 0.00
    ax.annotate(v, (bpw + dx, y + dy), fontsize=7.5,
                ha=("right" if dx < 0 else ("left" if dx > 0 else "center")),
                va=("bottom" if dy > 0 else ("top" if dy < 0 else "center")),
                color="#222")

ax.set_yticks(list(method_row.values()))
ax.set_yticklabels(list(method_row.keys()), fontsize=10)
ax.set_xlabel("Effective bits per weight (bpw)", fontsize=10)
ax.set_xlim(2, 9.5)
ax.set_ylim(-0.9, 3.4)
ax.set_xticks([2, 3, 4, 5, 6, 7, 8, 9])
ax.grid(True, axis="x", alpha=0.3, linestyle=":")
# Calibration cliff band, with the label well above the top row
ax.axvspan(4.0, 4.85, alpha=0.10, color="orange", zorder=0)
ax.text(4.42, 3.05, "calibration cliff", fontsize=8.5, ha="center",
        color="#a36800", style="italic")

# Legend inside the plot, top-right (where there is empty space at bpw>6)
ax.legend(loc="upper right", fontsize=8, framealpha=0.95, ncol=3)

plt.tight_layout()
plt.savefig(FIGDIR / "fig_quant_variants.pdf", bbox_inches="tight")
plt.savefig(FIGDIR / "fig_quant_variants.png", dpi=180, bbox_inches="tight")
print("saved fig_quant_variants")
