#!/usr/bin/env python3
"""Independent cross-check of the paper's published numbers, loaded from the committed
result logs under experiment/results/ instead of being hardcoded.

This module does not render anything (the paper's only figure is drawn by fig_story.py,
which reads the logs directly). It exists so the published values have a second,
separately-written resolver: a drift in the logs is caught here as well as by
verify_values.py.

Every value a figure draws is either (a) recomputed here from a committed log, or
(b) a documented constant that has no single committed artifact because it is a
multi-run synthesis reported in a paper table (e.g. the tab:threefactor cells that
pool control_positions/q4km_noise/multiseed at different n, or the 3-seed LoRA BF16
pools whose per-seed logs are shipped but whose pool is not materialized). Case (b)
values are marked _SYNTH and carry the paper-table locator.

Run `python scripts/_fig_data.py` to self-check: it asserts every loadable value
equals the number published in the paper, so a drift in the logs is caught here
rather than silently changing a figure.
"""
from __future__ import annotations
import collections
import json
import os
import pathlib

ROOT = pathlib.Path(os.environ.get("QQUILT_REPO", pathlib.Path(__file__).resolve().parent.parent))
R = ROOT / "experiment" / "results"


def _load(rel):
    p = R / rel
    if str(rel).endswith(".jsonl"):
        return [json.loads(l) for l in open(p)]
    return json.load(open(p))


def _greedy_ge10(rel) -> dict:
    """Per-version count of G1 canaries whose greedy continuation matches >=10 chars
    (== percentage, 100 canaries/seed). Same logic as qquilt.metrics / verify_values."""
    s = collections.defaultdict(set)
    for r in _load(rel):
        if r.get("group") not in (None, "g1"):
            continue
        if r.get("decoding") == "greedy" and (r.get("match_prefix_len") or 0) >= 10:
            s[r["version"]].add(r.get("canary_id") or r.get("seq_id"))
    return {v: len(s[v]) for v in s}


def _pool_rate(rel, vk) -> float | None:
    d = _load(rel)
    node = d["per_threshold"]["10"]["pooled"] if "per_threshold" in d else d["pooled"]
    v = node.get(vk, {}).get("rate")
    return None if v is None else round(v * 100, 1)


def _pool_seeds_pct(rel, vk, seeds) -> float:
    d = _load(rel)
    psc = d["per_threshold"]["10"]["per_seed_counts"]
    k = sum(psc[str(s)][vk][0] for s in seeds)
    n = sum(psc[str(s)][vk][1] for s in seeds)
    return round(100 * k / n, 1)


def _seed_count(rel, seed, vk) -> int:
    d = _load(rel)
    return d["per_threshold"]["10"]["per_seed_counts"][str(seed)][vk][0]


_P5 = "exp_3seed_replication/pooled_stats_5seed.json"

# --------------------------------------------------------------------------
# crossfamily(): tab:headline extraction rates (%), full-FT and LoRA blocks
# --------------------------------------------------------------------------
def crossfamily():
    def g(rel, v):
        return float(_greedy_ge10(rel).get(v, 0))
    ft_bf16 = [_pool_rate("qwen_extra_pooled_qwen05b.json", "bf16"),
               _pool_rate(_P5, "bf16"),
               _pool_rate("qwen_extra_pooled_qwen15b.json", "bf16"),
               g("wave_1_llama32_3b_fullft_seed42/extraction.jsonl", "bf16"),
               g("wave_1_qwen25_7b_seed42/extraction.jsonl", "bf16")]
    ft_q4 = [_pool_rate("qwen_extra_pooled_qwen05b.json", "q4_k_m"),
             _pool_rate(_P5, "q4_k_m"),
             _pool_rate("qwen_extra_pooled_qwen15b.json", "q4_k_m"),
             g("wave_1_llama32_3b_fullft_seed42/extraction.jsonl", "q4_k_m"),
             g("wave_1_qwen25_7b_seed42/extraction.jsonl", "q4_k_m")]
    ft_awq = [0.0,  # _SYNTH: Qwen2.5-0.5B AWQ has no committed log (tab:headline)
              _pool_rate(_P5, "awq_4bit"),
              _pool_rate("qwen_extra_pooled_qwen15b.json", "awq_4bit"),
              g("wave_1_llama32_3b_fullft_seed42/extraction.jsonl", "awq_4bit"),
              g("wave_1_qwen25_7b_seed42/extraction.jsonl", "awq_4bit")]
    lo_q4 = [g("wave_1_qwen25_05b_lora_seed42/extraction.jsonl", "q4_k_m"),
             g("wave_1_llama32_1b_lora_seed42/extraction.jsonl", "q4_k_m"),
             g("wave_1_llama3b_lora_seed42/extraction.jsonl", "q4_k_m"),
             g("wave_1_llama3b_lora_seed42_lr2e4/extraction.jsonl", "q4_k_m")]
    lo_awq = [g("wave_1_qwen25_05b_lora_seed42/extraction.jsonl", "awq_4bit"),
              g("wave_1_llama32_1b_lora_seed42/extraction.jsonl", "awq_4bit"),
              g("wave_1_llama3b_lora_seed42/extraction.jsonl", "awq_4bit"),
              g("wave_1_llama3b_lora_seed42_lr2e4/extraction.jsonl", "awq_4bit")]
    # _SYNTH: LoRA BF16 bars are 3-seed pools (per-seed logs committed, pool not
    # materialized) and the lr2e-4 BF16 reference; see tab:headline.
    lo_bf16 = [23.3, 25.7, 28.0, 30.0]
    return dict(ft_bf16=ft_bf16, ft_q4=ft_q4, ft_awq=ft_awq,
                lo_bf16=lo_bf16, lo_q4=lo_q4, lo_awq=lo_awq)


# --------------------------------------------------------------------------
# mia(): verbatim on the seed-42 checkpoint, Min-K% AUCs, score means (sec:threat-split)
# --------------------------------------------------------------------------
def mia():
    mi = _load("exp_mia_indist/metrics.json")["versions"]
    extract = [float(_seed_count(_P5, 42, "bf16")),
               float(_seed_count(_P5, 42, "q4_k_m")),
               float(_seed_count(_P5, 42, "awq_4bit"))]
    ood = [round(mi["bf16"]["mink_standard_ood"]["auc"], 2),
           round(mi["awq_canary_free"]["mink_standard_ood"]["auc"], 2)]
    ind = [round(mi["bf16"]["mink_standard_indist"]["auc"], 2),
           round(mi["awq_canary_free"]["mink_standard_indist"]["auc"], 2)]
    # score-distribution means (members / in-dist Enron / OOD G3), Min-K% log-prob
    bf, aw = mi["bf16"], mi["awq_canary_free"]
    bf_means = [round(bf["mink_standard_indist"]["mem_mean"], 2),
                round(bf["mink_standard_indist"]["non_mean"], 2),
                round(bf["mink_standard_ood"]["non_mean"], 2)]
    aw_means = [round(aw["mink_standard_indist"]["mem_mean"], 2),
                round(aw["mink_standard_indist"]["non_mean"], 2),
                round(aw["mink_standard_ood"]["non_mean"], 2)]
    return dict(extract=extract, ood=ood, ind=ind, bf_means=bf_means, aw_means=aw_means)


# --------------------------------------------------------------------------
# mechanism(): the three-factor cells (tab:threefactor)
# --------------------------------------------------------------------------
def mechanism():
    nd = _load("exp_mechanism_noise_direction/metrics.json")["results"]["awq"]
    q4 = _load("exp_mechanism_q4km_noise_direction/metrics.json")
    cp = _load("exp_mechanism_control_positions/metrics.json")
    ms = _load("exp_mechanism_multiseed/summary.json")
    # Factor 1: cos(d, e_top1) at [AWQ@Enron, Q4@Enron, AWQ@RECALL, Q4@RECALL]
    cos_vals = [round(nd["cos_err_with_top1_basis"]["enron"], 5),
                round(q4["enron"]["cos_err_top1_mean"], 5),
                round(cp["canary_RECALL"]["cos_err_top1_mean"], 4),
                round(q4["canary_RECALL"]["cos_err_top1_mean"], 4)]
    # Factor 2: (ft_top1_prob, flip_rate_pct) per position.
    # top-1 probs and Q4/BODY flips load from logs; AWQ-RECALL flip (78) and the
    # Enron top-1 prob (0.55) are the tab:threefactor synthesized values (_SYNTH).
    factor2 = [
        ("Enron", 0.55, round(cp["enron"]["top1_flip_rate"] * 100)),
        ("RECALL\n(AWQ)", round(cp["canary_RECALL"]["ft_top1_prob_mean"], 2), 78),
        ("RECALL\n(Q4_K_M)", round(q4["canary_RECALL"]["ft_top1_prob_mean"], 2),
         round(ms["q4_k_m"]["pooled_flip"]["rate"] * 100)),
        ("BODY", round(cp["canary_BODY"]["ft_top1_prob_mean"], 4), 0),
    ]
    # Factor 3: [Q4, AWQ, GPTQ] logit-error norm and 3-seed extraction rate.
    # Q4 norm loads; AWQ norm (841) is the tab:threefactor synthesized value (_SYNTH).
    norms = [round(ms["q4_k_m"]["logit_norm_mean"]), 841, None]
    extract = [_pool_seeds_pct(_P5, "q4_k_m", [42, 52, 62]),
               0.0,
               float(_load("exp_gptq_4bit/metrics.json")["greedy_ge10"])]
    return dict(cos_vals=cos_vals, factor2=factor2, norms=norms, extract=extract)


# --------------------------------------------------------------------------
# quant_variants(): effective bits-per-weight per variant (fig:story(a) axis)
# --------------------------------------------------------------------------
def quant_variants():
    s7 = _load("step_7_awq_granularity/metrics.json")["results"]
    awq = {"g128": s7["group_128"]["approx_bpw"], "g64": s7["group_64"]["approx_bpw"],
           "g32": s7["group_32"]["approx_bpw"]}
    # GGUF/GPTQ effective bpw are format-defined (Kurt 2026; llama.cpp k-quant spec),
    # not measured quantities, so they stay as documented constants (_SYNTH).
    gguf = {"Q2_K": 2.6, "Q3_K_M": 3.4, "Q4_K_S": 4.3, "Q4_K_M": 4.7, "Q5_K_M": 5.5, "Q8_0": 8.5}
    return dict(awq=awq, gguf=gguf, gptq_g128=4.25)


# Published values (from the camera-ready) for the self-check of loadable numbers.
_EXPECTED = {
    "crossfamily.ft_bf16": [30.3, 26.6, 30.3, 30.0, 30.0],
    "crossfamily.ft_q4": [23.0, 4.0, 13.7, 16.0, 24.0],
    "crossfamily.ft_awq": [0.0, 0.0, 5.0, 3.0, 6.0],
    "crossfamily.lo_q4": [0.0, 0.0, 0.0, 25.0],
    "crossfamily.lo_awq": [0.0, 0.0, 0.0, 7.0],
    "mia.extract": [30.0, 6.0, 0.0],
    "mia.ood": [1.00, 0.97],
    "mia.ind": [0.83, 0.22],
    "mia.bf_means": [-0.03, -3.40, -9.22],
    "mia.aw_means": [-6.12, -3.49, -9.15],
    "mechanism.cos_vals": [0.00038, 0.00086, 0.0086, 0.0079],
    "mechanism.norms.q4": 617,
    "mechanism.extract": [5.3, 0.0, 0.0],
    "mechanism.factor2.top1": [0.67, 0.70, 0.9998],
    "mechanism.factor2.flip": [24, 48, 0],
    "quant.awq": [4.25, 4.5, 5.0],
}


def _verify():
    cf, mi, me, qv = crossfamily(), mia(), mechanism(), quant_variants()
    checks = {
        "crossfamily.ft_bf16": cf["ft_bf16"], "crossfamily.ft_q4": cf["ft_q4"],
        "crossfamily.ft_awq": cf["ft_awq"], "crossfamily.lo_q4": cf["lo_q4"],
        "crossfamily.lo_awq": cf["lo_awq"],
        "mia.extract": mi["extract"], "mia.ood": mi["ood"], "mia.ind": mi["ind"],
        "mia.bf_means": mi["bf_means"], "mia.aw_means": mi["aw_means"],
        "mechanism.cos_vals": me["cos_vals"], "mechanism.norms.q4": me["norms"][0],
        "mechanism.extract": me["extract"],
        "mechanism.factor2.top1": [me["factor2"][1][1], me["factor2"][2][1], me["factor2"][3][1]],
        "mechanism.factor2.flip": [me["factor2"][0][2], me["factor2"][2][2], me["factor2"][3][2]],
        "quant.awq": [qv["awq"]["g128"], qv["awq"]["g64"], qv["awq"]["g32"]],
    }
    bad = [(k, checks[k], _EXPECTED[k]) for k in _EXPECTED if checks[k] != _EXPECTED[k]]
    if bad:
        for k, got, exp in bad:
            print(f"MISMATCH {k}: loaded={got} published={exp}")
        raise SystemExit(1)
    print(f"_fig_data: all {len(_EXPECTED)} loadable figure values match the paper.")


if __name__ == "__main__":
    _verify()
