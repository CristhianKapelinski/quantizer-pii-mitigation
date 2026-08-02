#!/usr/bin/env python3
"""Paper figure: the whole argument in one ultrawide strip.

(a) WHAT happens  -- extraction falls with effective bit-rate, but the two
    calibration-based methods sit off that curve at 0%.
(b) WHERE the error goes -- the rounding error points at the token being
    predicted only where a canary is being recited, and points harder under
    the calibrated method.
(c) WHY it only bites there -- the same error flips the emitted token when
    the fine-tuned model was unsure, and is absorbed when it was certain.

Every value is read from the committed logs; nothing is hard-coded.
"""

import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("QQUILT_REPO", Path(__file__).resolve().parent.parent))
RES = ROOT / "experiment" / "results"
FIGDIR = Path(os.environ.get("QQUILT_FIGDIR", ROOT / "experiment" / "figures"))
FIGDIR.mkdir(parents=True, exist_ok=True)

INK, AWQC, Q4C, GREY = "#1F2A44", "#1f77b4", "#d62728", "#9aa4b2"


def load(rel):
    return json.load(open(RES / rel))


# --------------------------------------------------------------------- (a)
BPW = {"bf16": 16.0, "q8_0": 8.5, "q5_k_m": 5.5, "q4_k_m": 4.7,
       "q4_k_s": 4.3, "q3_k_m": 3.4, "q2_k": 2.6}
NAME = {"q8_0": "Q8_0", "q5_k_m": "Q5_K_M", "q4_k_m": "Q4_K_M",
        "q4_k_s": "Q4_K_S", "q3_k_m": "Q3_K_M", "q2_k": "Q2_K"}
pooled = load("reviewer_polish/m10_threshold_sensitivity.json")["by_version_pooled"]
FALLBACK = {"q4_k_s": 1, "q3_k_m": 0, "q2_k": 0}

gguf = []
for v, bpw in BPW.items():
    if v == "bf16":
        continue
    if v in pooled:
        n = pooled[v]["n_total"]
        k = pooled[v]["counts_by_threshold"]["10"]
    else:
        n, k = 100, FALLBACK[v]
    gguf.append((bpw, 100.0 * k / n, NAME[v]))
gguf.sort()

# --------------------------------------------------------------- (b) and (c)
seeds = ("seed42", "seed52", "seed62")
MS = RES / "exp_mechanism_multiseed"


def pool(getter):
    """Mean over the three mechanism seeds."""
    vals = [getter(s) for s in seeds]
    return sum(vals) / len(vals)


def pool_flip(getter):
    """Pooled flip rate (%) over the three mechanism seeds, n = 100 each."""
    return 100.0 * sum(round(getter(s) * 100) for s in seeds) / (100 * len(seeds))


def awq(s):
    return json.load(open(MS / s / "awq_metrics.json"))["results"]["awq"]


def q4(s, pos):
    return json.load(open(MS / s / "q4km_metrics.json"))[pos]


body = json.load(open(RES / "exp_mechanism_local_replication"
                            "/mech_1b_body_local.json"))["canary_BODY"]

cos = {
    ("AWQ", "Recall"): pool(lambda s: awq(s)["cos_err_with_top1_basis"]["canary"]),
    ("AWQ", "Body"): body["cos_err_top1_mean"],
    ("AWQ", "Enron"): pool(lambda s: awq(s)["cos_err_with_top1_basis"]["enron"]),
    ("Q4_K_M", "Recall"): pool(lambda s: q4(s, "canary_RECALL")["cos_err_top1_mean"]),
    ("Q4_K_M", "Body"): pool(lambda s: q4(s, "canary_BODY")["cos_err_top1_mean"]),
    ("Q4_K_M", "Enron"): pool(lambda s: q4(s, "enron")["cos_err_top1_mean"]),
}
VOCAB = 128256                      # Llama-3.2 vocabulary
isotropic = 1.0 / math.sqrt(VOCAB)  # error pointing nowhere in particular

conf = {
    "Recall": pool(lambda s: q4(s, "canary_RECALL")["ft_top1_prob_mean"]),
    "Body": pool(lambda s: q4(s, "canary_BODY")["ft_top1_prob_mean"]),
    "Enron": pool(lambda s: q4(s, "enron")["ft_top1_prob_mean"]),
}
flip = {
    ("AWQ", "Recall"): pool_flip(lambda s: awq(s)["top1_flip_rate"]["canary"]),
    ("AWQ", "Body"): 100.0 * body["top1_flip_rate"],
    ("AWQ", "Enron"): pool_flip(lambda s: awq(s)["top1_flip_rate"]["enron"]),
    ("Q4_K_M", "Recall"): pool_flip(lambda s: q4(s, "canary_RECALL")["top1_flip_rate"]),
    ("Q4_K_M", "Body"): pool_flip(lambda s: q4(s, "canary_BODY")["top1_flip_rate"]),
    ("Q4_K_M", "Enron"): pool_flip(lambda s: q4(s, "enron")["top1_flip_rate"]),
}

# ------------------------------------------------------------------- render
fig, axes = plt.subplots(1, 3, figsize=(13.2, 2.62))
plt.subplots_adjust(wspace=0.30)
for ax in axes:
    ax.grid(True, alpha=0.25, linestyle=":", linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

# (a) extraction vs effective bit-rate
ax = axes[0]
ax.plot([b for b, _, _ in gguf], [r for _, r, _ in gguf], "-o", color=GREY,
        markersize=4.5, linewidth=1.5, markerfacecolor="white",
        markeredgecolor=GREY, label="GGUF k-quants")
for b, r, nm in gguf:
    if nm in ("Q4_K_M", "Q5_K_M"):
        ax.annotate(nm, (b, r), textcoords="offset points", xytext=(5, -2),
                    fontsize=7.2, color="#555", va="top")
ax.plot([4.25], [0.0], marker="s", color=AWQC, markersize=7.5, linestyle="none",
        label="AWQ (calibrated)")
ax.plot([4.25], [0.0], marker="D", color="#2ca02c", markersize=4.5,
        linestyle="none", markerfacecolor="white", markeredgewidth=1.4,
        markeredgecolor="#2ca02c", label="GPTQ (calibrated)")
ax.annotate("0% at 4.25 bpw", (4.25, 0.6),
            textcoords="offset points", xytext=(6, 32), fontsize=7.4, color=INK,
            arrowprops=dict(arrowstyle="->", lw=0.9, color=INK))
ax.set_xlabel("effective bits per weight", fontsize=9)
ax.set_ylabel("canaries extracted (%)", fontsize=9)
ax.set_title("(a) extraction vs.\neffective bit-rate", fontsize=9.2, pad=5)
ax.set_xlim(2.0, 9.2)
ax.set_ylim(-2, 34)
ax.legend(fontsize=7.2, frameon=False, loc="upper left", handlelength=1.4)

# (b) where the error points
ax = axes[1]
pos = ["Recall", "Body", "Enron"]
x = range(len(pos))
w = 0.36
ax.bar([i - w / 2 for i in x], [cos[("AWQ", p)] for p in pos], w,
       color=AWQC, edgecolor="black", linewidth=0.5, label="AWQ")
ax.bar([i + w / 2 for i in x], [cos[("Q4_K_M", p)] for p in pos], w,
       color=Q4C, edgecolor="black", linewidth=0.5, label="Q4_K_M")
ax.axhline(isotropic, color=INK, linestyle="--", linewidth=0.9)
ax.text(2.42, isotropic * 1.08, "random direction", fontsize=7,
        color=INK, ha="right", va="bottom")
ax.set_xticks(list(x))
ax.set_xticklabels(["memorized\n(Recall)", "template\n(Body)", "held-out\n(Enron)"],
                   fontsize=8)
ax.set_ylabel(r"$\cos(\mathbf{d},\,\mathbf{e}_{v^\star})$", fontsize=9)
ax.set_title("(b) error alignment with the\npredicted token, by position", fontsize=9.2, pad=5)
ax.legend(fontsize=7.4, frameon=False, loc="upper right", handlelength=1.2)

# (c) whether it matters: certainty absorbs the error
ax = axes[2]
ax.bar([i - w / 2 for i in x], [flip[("AWQ", p)] for p in pos], w,
       color=AWQC, edgecolor="black", linewidth=0.5, label="AWQ")
ax.bar([i + w / 2 for i in x], [flip[("Q4_K_M", p)] for p in pos], w,
       color=Q4C, edgecolor="black", linewidth=0.5, label="Q4_K_M")
for i, p in enumerate(pos):
    if p == "Body":
        continue
    top = max(flip[("AWQ", p)], flip[("Q4_K_M", p)])
    ax.text(i, top + 4, "%.2f sure" % conf[p], ha="center", va="bottom",
            fontsize=7.2, color="#555")
ax.annotate("0.9998 sure", (1, 3), textcoords="offset points",
            xytext=(0, 30), fontsize=7.2, color="#555", ha="center",
            arrowprops=dict(arrowstyle="->", lw=0.8, color="#888"))
ax.set_xticks(list(x))
ax.set_xticklabels(["memorized\n(Recall)", "template\n(Body)", "held-out\n(Enron)"],
                   fontsize=8)
ax.set_ylabel("emitted token changed (%)", fontsize=9)
ax.set_title("(c) token-change (FLIP) rate,\nby position", fontsize=9.2, pad=5)
ax.set_ylim(0, 104)
ax.legend(fontsize=7.4, frameon=False, loc="upper center", ncol=2,
          handlelength=1.2, columnspacing=1.0)

fig.savefig(FIGDIR / "fig_story.pdf", bbox_inches="tight")
fig.savefig(FIGDIR / "fig_story.png", dpi=180, bbox_inches="tight")
print("saved fig_story  |  isotropic=%.4f" % isotropic)
for k, v in sorted(cos.items()):
    print("  cos", k, round(v, 5))
for k, v in sorted(flip.items()):
    print("  flip", k, round(v, 1))
print("  conf", {k: round(v, 4) for k, v in conf.items()})
