#!/usr/bin/env python3
"""Figure 1: Dose-response curve of verbatim PII extraction vs.
effective bits-per-weight, pooled across five Llama-3.2-1B seeds.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _os
ROOT = Path(_os.environ.get("QQUILT_REPO",
            Path(__file__).resolve().parent.parent))
FIGDIR = ROOT / "experiment" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
DATA = json.load(open(ROOT / "experiment/results/reviewer_polish/"
                      "m10_threshold_sensitivity.json"))

bpw_map = {"bf16": 16.0, "q8_0": 8.5, "q5_k_m": 5.5, "q4_k_m": 4.7,
           "q4_k_s": 4.3, "q3_k_m": 3.4, "q2_k": 2.6}
label_map = {"bf16": "BF16", "q8_0": "Q8_0", "q5_k_m": "Q5_K_M",
             "q4_k_m": "Q4_K_M", "q4_k_s": "Q4_K_S",
             "q3_k_m": "Q3_K_M", "q2_k": "Q2_K"}
pooled = DATA["by_version_pooled"]
table3_fallback = {"q4_k_s": (4.3, 1), "q3_k_m": (3.4, 0), "q2_k": (2.6, 0)}

gguf_pts = []
fp_pts = []
for v, bpw in bpw_map.items():
    if v in pooled:
        n = pooled[v]["n_total"]
        k = pooled[v]["counts_by_threshold"]["10"]
    elif v in table3_fallback:
        bpw, k = table3_fallback[v]
        n = 100
    else:
        continue
    rate = 100 * k / n
    pt = (bpw, rate, label_map[v])
    if v in ("bf16", "q8_0"):
        fp_pts.append(pt)
    else:
        gguf_pts.append(pt)

gguf_pts.sort()
fp_pts.sort()

fig, ax = plt.subplots(figsize=(6.8, 2.5))

# Combine into the staircase the GGUF + FP ceiling
all_curve = gguf_pts + fp_pts
all_curve.sort()
xs = [p[0] for p in all_curve]
ys = [p[1] for p in all_curve]
ax.plot(xs, ys, "o-", color="#d62728", linewidth=1.8, markersize=7,
        label="GGUF / FP (calib.-free)", zorder=3)

# FP markers (square overlay)
for (bpw, rate, lab) in fp_pts:
    ax.plot(bpw, rate, "s", color="#7f7f7f", markersize=8,
            markeredgecolor="black", markeredgewidth=0.6, zorder=4)

# AWQ and GPTQ both sit at g128 = 4.25 bpw, rate=0 (their actual values).
# The smaller GPTQ diamond is drawn on top of the AWQ star so both stay
# visible without displacing either marker off its true bpw.
ax.plot(4.25, 0.0, marker="*", color="#1f77b4", markersize=22,
        markeredgecolor="black", markeredgewidth=0.7,
        label="AWQ (calib.)", zorder=5)
ax.plot(4.25, 0.0, marker="D", color="#2ca02c", markersize=7,
        markeredgecolor="black", markeredgewidth=0.7,
        label="GPTQ (calib.)", zorder=6)

# Label only the key inflection points; collapse Q3/Q2 into one group
label_offsets = {
    "Q4_K_M": (0.25, 0.5,  "left",   "center"),
    "Q5_K_M": (0.25, 0.0,  "left",   "center"),
    "Q8_0":   (0,    -2.5, "center", "top"),
    "BF16":   (0,    -2.5, "center", "top"),
}
for (bpw, rate, lab) in gguf_pts + fp_pts:
    if lab not in label_offsets:
        continue
    ox, oy, ha, va = label_offsets[lab]
    ax.annotate(lab, xy=(bpw, rate), xytext=(bpw + ox, rate + oy),
                fontsize=8.5, color="#333333", ha=ha, va=va)

# group label for low-bpw points (Q2_K, Q3_K_M, Q4_K_S all at <=1), sitting
# directly above its own points so no sloped leader crosses the plot area
ax.text(3.0, 2.0, "Q2_K, Q3_K_M, Q4_K_S\n(all $\\leq$ 1%)",
        fontsize=7.5, color="#444", ha="center", va="bottom")

# Shaded "calibration cliff" zone
ax.axvspan(4.0, 4.85, alpha=0.10, color="orange", zorder=0)
ax.text(4.42, 16, "calibration\ncliff", fontsize=8.5, ha="center",
        color="#a36800", style="italic")

# AWQ + GPTQ are single points, not series: label them as plain text with the
# bpw stated, so no leader line can be read as a bpw -> extraction trajectory
ax.text(2.15, 12.5, "AWQ 0%", fontsize=8.5, color="#1f77b4",
        weight="bold", ha="left", va="center")
ax.text(2.15, 8.5, "GPTQ 0%", fontsize=8.5, color="#2ca02c",
        weight="bold", ha="left", va="center")

ax.set_xlabel("Effective bits per weight (bpw)", fontsize=10)
ax.set_ylabel("Extraction rate (%)", fontsize=10)
ax.set_xlim(2.0, 18.0)
ax.set_ylim(-1, 35)
ax.set_xticks([2, 4, 6, 8, 10, 12, 14, 16])
ax.set_yticks([0, 10, 20, 30])
ax.grid(True, alpha=0.3, zorder=0, linestyle=":")
ax.legend(loc="lower right", fontsize=7, framealpha=0.95)

plt.tight_layout()
plt.savefig(FIGDIR / "fig_dose_response.pdf", bbox_inches="tight")
plt.savefig(FIGDIR / "fig_dose_response.png", dpi=180, bbox_inches="tight")
print("saved fig_dose_response.{pdf,png}")
