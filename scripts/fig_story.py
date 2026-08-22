#!/usr/bin/env python3
"""Paper figure: the whole argument in one ultrawide strip.

(a) WHAT happens  -- extraction falls with effective bit-rate, but the two
    calibration-based methods sit off that curve at 0%.
(b) WHERE the error goes -- the rounding error points at the token being
    predicted only where a canary is being recited, and points harder under
    the calibrated method.
(c) WHY it only bites there -- the same error flips the emitted token when
    the fine-tuned model was unsure, and is absorbed when it was certain.

Every measured value -- every bar height, every point's extraction rate, every
annotated probability -- is read from the committed logs under
experiment/results/. Two things in this file are constants instead, and both are
definitions rather than measurements: the effective bits-per-weight of each GGUF
format (BPW, the k-quant block layout) and the tokenizer's vocabulary size
(VOCAB, which fixes where an error pointing nowhere in particular would land).
Each is documented at its definition with what it is and why no log can supply
it.
"""

import collections
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
    """Load and parse the JSON file at `RES / rel`."""
    return json.load(open(RES / rel))


def greedy_ge10(rel):
    """Per-version (k, n) over the G1 canaries in the extraction log at `rel`:
    k = canaries whose greedy continuation matched >=10 characters, n = distinct
    canaries probed. Same counting rule as qquilt.metrics and verify_values."""
    hit = collections.defaultdict(set)
    seen = collections.defaultdict(set)
    for line in open(RES / rel):
        r = json.loads(line)
        if r.get("group") not in (None, "g1") or r.get("decoding") != "greedy":
            continue
        cid = r.get("canary_id") or r.get("seq_id")
        seen[r["version"]].add(cid)
        if (r.get("match_prefix_len") or 0) >= 10:
            hit[r["version"]].add(cid)
    return {v: (len(hit[v]), len(seen[v])) for v in seen}


# --------------------------------------------------------------------- (a)
# Effective bits per weight, per format. The one table here that is a definition
# and not a measurement: it is the GGUF k-quant block layout (the weight bits
# plus the per-block scale and min carried alongside them), so it is a property
# of the file format, identical on every machine, and no run can produce it.
# step_8_gguf_lowbit/metrics.json and step_8b_q4ks/metrics.json each carry their
# own `bits_per_param_assumed` copy of this table; the two agree with this one
# except for Q4_K_M, recorded there as 4.5 against the 4.7 the paper's axis
# prints and scripts/_fig_data.py cross-checks. The paper's value is plotted.
BPW = {"bf16": 16.0, "q8_0": 8.5, "q5_k_m": 5.5, "q4_k_m": 4.7,
       "q4_k_s": 4.3, "q3_k_m": 3.4, "q2_k": 2.6}
NAME = {"q8_0": "Q8_0", "q5_k_m": "Q5_K_M", "q4_k_m": "Q4_K_M",
        "q4_k_s": "Q4_K_S", "q3_k_m": "Q3_K_M", "q2_k": "Q2_K"}

# The curve mixes two committed pools, because the low-bit tail was measured
# once and the rest five times. Q8_0/Q5_K_M/Q4_K_M (and AWQ below) come from the
# 5-seed threshold-sensitivity pool, n=500. Q4_K_S/Q3_K_M/Q2_K were added later
# on the single seed-42 cell, n=100, and are recounted here from their own
# extraction logs rather than transcribed.
pooled = load("reviewer_polish/m10_threshold_sensitivity.json")["by_version_pooled"]
lowbit = {}
for rel in ("step_8_gguf_lowbit/extraction.jsonl", "step_8b_q4ks/extraction.jsonl"):
    lowbit.update(greedy_ge10(rel))

gguf = []
for v, bpw in BPW.items():
    if v == "bf16":
        continue
    if v in pooled:
        n = pooled[v]["n_total"]
        k = pooled[v]["counts_by_threshold"]["10"]
    else:
        k, n = lowbit[v]
    gguf.append((bpw, 100.0 * k / n, NAME[v]))
gguf.sort()

# The two calibrated points. Their bit-rate is the AWQ group-128 effective
# bit-rate measured in the granularity sweep; GPTQ at group size 128 stores the
# same 4 bits plus one scale/zero-point pair per 128 weights, so it sits at the
# same abscissa. Their heights are each method's own extraction rate: AWQ from
# the same 5-seed pool as the k-quant curve, GPTQ from its own cell.
calibrated_bpw = load("step_7_awq_granularity/metrics.json")["results"]["group_128"]["approx_bpw"]
awq_rate = 100.0 * pooled["awq_4bit"]["counts_by_threshold"]["10"] / pooled["awq_4bit"]["n_total"]
_gptq = load("exp_gptq_4bit/metrics.json")
gptq_rate = 100.0 * _gptq["greedy_ge10"] / _gptq["n_canaries_total"]

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
    """Load seed `s`'s AWQ noise-direction metrics (exp_mechanism_multiseed)."""
    return json.load(open(MS / s / "awq_metrics.json"))["results"]["awq"]


def q4(s, pos):
    """Load seed `s`'s Q4_K_M noise-direction metrics at position `pos`
    (e.g. "canary_RECALL", "canary_BODY", "enron") from exp_mechanism_multiseed."""
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
# The second definition-not-measurement in this file: the Llama-3.2 tokenizer's
# vocabulary size, i.e. the dimension of the logit vector the mechanism runs
# compare. It fixes the dashed reference line -- the cosine an error vector
# pointing in a uniformly random direction would have with any single basis
# vector -- so it is a property of the model family, not of a run.
VOCAB = 128256
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
ax.plot([calibrated_bpw], [awq_rate], marker="s", color=AWQC, markersize=7.5,
        linestyle="none", label="AWQ (calibrated)")
ax.plot([calibrated_bpw], [gptq_rate], marker="D", color="#2ca02c", markersize=4.5,
        linestyle="none", markerfacecolor="white", markeredgewidth=1.4,
        markeredgecolor="#2ca02c", label="GPTQ (calibrated)")
ax.annotate("%.0f%% at %g bpw" % (awq_rate, calibrated_bpw), (calibrated_bpw, 0.6),
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
# The Body bar is ~0, so its certainty is called out instead of printed above it
# like the other two; the number is the same conf["Body"] the loop above uses.
ax.annotate("%.4f sure" % conf["Body"], (1, 3), textcoords="offset points",
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
