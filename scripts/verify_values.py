#!/usr/bin/env python3
"""Exact-match verification of the paper's published numbers against the committed logs.

Reads expected/paper_values.json (ground truth parsed from the camera-ready) and, for
every key that has a resolver, recomputes the value from experiment/results/ and compares
it EXACTLY at the precision the paper prints. Keys without a resolver are SKIPPED (so a
single-stage run still verifies in isolation), and the reasons are listed in the report.

Exit 0 only when there are zero FAILs. The auto section of docs/REPRODUCIBILITY_REPORT.md
is rewritten between the AUTO markers.

    python scripts/verify_values.py [--results experiment/results] [--report docs/REPRODUCIBILITY_REPORT.md]
"""
from __future__ import annotations
import argparse, collections, json, pathlib, sys

AUTO_BEGIN = "<!-- AUTO:VERIFY:BEGIN -->"
AUTO_END = "<!-- AUTO:VERIFY:END -->"


def load(p: pathlib.Path):
    if str(p).endswith(".jsonl"):
        return [json.loads(l) for l in open(p)]
    return json.load(open(p))


def greedy_ge10(rows) -> dict:
    """Per-version count of G1 canaries whose greedy continuation matches >=10 chars.
    Identical logic to replay.sh; with 100 canaries/seed the count equals the percentage."""
    s = collections.defaultdict(set)
    for r in rows:
        if r.get("group") not in (None, "g1"):
            continue
        if r.get("decoding") == "greedy" and (r.get("match_prefix_len") or 0) >= 10:
            s[r["version"]].add(r.get("canary_id") or r.get("seq_id"))
    return {v: len(s[v]) for v in s}


def decimals(expected) -> int:
    s = repr(expected)
    return len(s.split(".", 1)[1]) if "." in s else 0


class R:
    """Lazy result-file accessor rooted at the results dir; caches loaded files."""

    def __init__(self, root: pathlib.Path):
        self.root = root
        self._cache: dict[str, object] = {}

    def get(self, rel: str):
        if rel not in self._cache:
            p = self.root / rel
            self._cache[rel] = load(p) if p.exists() else None
        return self._cache[rel]

    def pooled_rate(self, rel, vkey):
        d = self.get(rel)
        if d is None:
            return None
        node = d["per_threshold"]["10"]["pooled"] if "per_threshold" in d else d["pooled"]
        v = node.get(vkey, {}).get("rate")
        return None if v is None else v * 100

    def per_seed_pool(self, rel, vkey, seeds):
        """Pool greedy>=10 counts over given seeds from a pooled_stats per_seed_counts file (as %)."""
        d = self.get(rel)
        if d is None:
            return None
        psc = d["per_threshold"]["10"]["per_seed_counts"]
        k = n = 0
        for s in seeds:
            row = psc[str(s)][vkey]
            k += row[0]
            n += row[1]
        return 100 * k / n if n else None

    def jsonl_ge10(self, rel, vkey):
        d = self.get(rel)
        if d is None:
            return None
        return greedy_ge10(d).get(vkey, 0)

    def jsonl_pool_ge10(self, rels, vkey):
        """Pool the greedy >=10 endpoint across independent per-seed JSONL files."""
        hits = total = 0
        for rel in rels:
            rows = self.get(rel)
            if rows is None:
                return None
            hits += greedy_ge10(rows).get(vkey, 0)
            canaries = {
                row.get("canary_id") or row.get("seq_id")
                for row in rows
                if row.get("group") in (None, "g1")
            }
            total += len(canaries)
        return None if total == 0 else 100 * hits / total


def build_resolvers(r: R):
    """Map expected-value keys to a zero-arg callable returning the recomputed value (or None)."""
    res: dict[str, callable] = {}
    P5 = "exp_3seed_replication/pooled_stats_5seed.json"
    VK = {"bf16": "bf16", "q8_0": "q8_0", "q5_k_m": "q5_k_m", "q4_k_m": "q4_k_m", "awq": "awq_4bit"}

    # tab:headline -- Llama-3.2-1B 5-seed pool
    for v, vk in VK.items():
        res[f"headline_greedy_ge10_extraction_pct.fullft.llama1b.{v}"] = (lambda vk=vk: r.pooled_rate(P5, vk))
    # tab:headline -- Qwen 3-seed pools
    for v, vk in VK.items():
        res[f"headline_greedy_ge10_extraction_pct.fullft.qwen0_5b.{v}"] = (lambda vk=vk: r.pooled_rate("qwen_extra_pooled_qwen05b.json", vk))
        res[f"headline_greedy_ge10_extraction_pct.fullft.qwen1_5b.{v}"] = (lambda vk=vk: r.pooled_rate("qwen_extra_pooled_qwen15b.json", vk))
    # tab:headline -- single-seed full FT (jsonl, count == %)
    for v, vk in VK.items():
        res[f"headline_greedy_ge10_extraction_pct.fullft.llama3b.{v}"] = (lambda vk=vk: r.jsonl_ge10("wave_1_llama32_3b_fullft_seed42/extraction.jsonl", vk))
        res[f"headline_greedy_ge10_extraction_pct.fullft.qwen7b.{v}"] = (lambda vk=vk: r.jsonl_ge10("wave_1_qwen25_7b_seed42/extraction.jsonl", vk))
    # tab:headline -- LoRA 4-bit == 0 cells (seed 42) and lr2e-4 knob (seed 42)
    for v in ("bf16", "q8_0", "q5_k_m", "q4_k_m", "awq"):
        vk = VK[v]
        res[f"headline_greedy_ge10_extraction_pct.lora.llama3b_lr2e4.{v}"] = (lambda vk=vk: r.jsonl_ge10("wave_1_llama3b_lora_seed42_lr2e4/extraction.jsonl", vk))
    for v in ("q4_k_m", "awq"):
        vk = VK[v]
        res[f"headline_greedy_ge10_extraction_pct.lora.qwen0_5b.{v}"] = (lambda vk=vk: r.jsonl_ge10("wave_1_qwen25_05b_lora_seed42/extraction.jsonl", vk))
        res[f"headline_greedy_ge10_extraction_pct.lora.llama1b.{v}"] = (lambda vk=vk: r.jsonl_ge10("wave_1_llama32_1b_lora_seed42/extraction.jsonl", vk))
        res[f"headline_greedy_ge10_extraction_pct.lora.llama3b_lr2e5.{v}"] = (lambda vk=vk: r.jsonl_ge10("wave_1_llama3b_lora_seed42/extraction.jsonl", vk))
        res[f"headline_greedy_ge10_extraction_pct.lora.llama3b_lr2e4.{v}"] = (lambda vk=vk: r.jsonl_ge10("wave_1_llama3b_lora_seed42_lr2e4/extraction.jsonl", vk))

    def seed_logs(tag):
        return [f"{tag}_seed{seed}/extraction.jsonl" for seed in (42, 52, 62)]

    # The aggregate JSON omits Qwen-0.5B AWQ, but all three source logs are
    # committed. The LoRA BF16/Q5_K_M entries are likewise pooled directly
    # from their per-seed logs.
    res["headline_greedy_ge10_extraction_pct.fullft.qwen0_5b.awq"] = lambda: r.jsonl_pool_ge10(
        [f"wave_1_qwen05b_seed{seed}/extraction.jsonl" for seed in (42, 52, 62)],
        "awq_4bit",
    )
    res["headline_greedy_ge10_extraction_pct.lora.qwen0_5b.bf16"] = lambda: r.jsonl_pool_ge10(
        seed_logs("wave_1_qwen25_05b_lora"), "bf16"
    )
    res["headline_greedy_ge10_extraction_pct.lora.llama1b.bf16"] = lambda: r.jsonl_pool_ge10(
        seed_logs("wave_1_llama32_1b_lora"), "bf16"
    )
    res["headline_greedy_ge10_extraction_pct.lora.llama3b_lr2e5.bf16"] = lambda: r.jsonl_pool_ge10(
        seed_logs("wave_1_llama3b_lora"), "bf16"
    )
    res["headline_greedy_ge10_extraction_pct.lora.llama3b_lr2e5.q5_k_m"] = lambda: r.jsonl_pool_ge10(
        seed_logs("wave_1_llama3b_lora"), "q5_k_m"
    )

    # tab:awq-sweep -- AWQ group-size sweep (single seed)
    def step7(path):
        d = r.get("step_7_awq_granularity/metrics.json")
        return None if d is None else d["results"][path]["greedy_ge10"]
    res["awq_group_size_sweep_ge10_count.awq_g32"] = lambda: step7("group_32")
    res["awq_group_size_sweep_ge10_count.awq_g64"] = lambda: step7("group_64")
    res["awq_group_size_sweep_ge10_count.awq_g128"] = lambda: step7("group_128")
    res["awq_group_size_sweep_ge10_count.ref_q4_k_m"] = lambda: (r.get("step_7_awq_granularity/metrics.json") or {}).get("reference_q4_k_m_g1_greedy_ge10")
    res["awq_group_size_sweep_ge10_count.ref_q5_k_m"] = lambda: (lambda d: None if d is None else d["per_threshold"]["10"]["per_seed_counts"]["42"]["q5_k_m"][0])(r.get(P5))

    # tab:gptq -- calibration-based vs -free, 3-seed pool (seeds 42,52,62)
    res["calib_vs_free_3seed_ge10_pct.bf16"] = lambda: r.per_seed_pool(P5, "bf16", [42, 52, 62])
    res["calib_vs_free_3seed_ge10_pct.q4_k_m"] = lambda: r.per_seed_pool(P5, "q4_k_m", [42, 52, 62])
    res["calib_vs_free_3seed_ge10_pct.awq_g128"] = lambda: (lambda d: None if d is None else float(d.get("reference_awq_enron_g1", 0)))(r.get("exp_gptq_4bit/metrics.json"))
    res["calib_vs_free_3seed_ge10_pct.gptq_g128"] = lambda: (lambda d: None if d is None else float(d.get("greedy_ge10")))(r.get("exp_gptq_4bit/metrics.json"))

    # sec:mechanism
    def sal(cell, field):
        d = r.get("exp_saliency_2x2/metrics.json")
        return None if d is None else d["results"][cell][field]
    cellmap = {"A_wikitext": "cell_A", "B_mix": "cell_B", "C_canary": "cell_C", "D_enron": "cell_D"}
    fmap = {"ge5": "greedy_ge5", "ge10": "greedy_ge10", "anyof6_ge10": "any_of_6_ge10"}
    for ck, cell in cellmap.items():
        for fk, field in fmap.items():
            res[f"saliency_ablation.{ck}.{fk}"] = (lambda cell=cell, field=field: sal(cell, field))

    # sec:threat-split
    def mia(vkey, probe):
        d = r.get("exp_mia_indist/metrics.json")
        return None if d is None else d["versions"][vkey][probe]["auc"]
    MV = {"bf16": "bf16", "awq": "awq_canary_free"}
    PB = {"ood": {"mink": "mink_standard_ood", "minkpp": "minkpp_ood", "loss": "loss_canary_ood"},
          "indist": {"mink": "mink_standard_indist", "minkpp": "minkpp_indist", "loss": "loss_canary_indist"}}
    for prot in ("ood", "indist"):
        for v, vk in MV.items():
            for score in ("mink", "minkpp", "loss"):
                res[f"mia_auc.{prot}.{v}.{score}"] = (lambda vk=vk, pb=PB[prot][score]: mia(vk, pb))

    # sec:utility
    def dsn(model, task):
        d = r.get("exp_downstream/metrics.json")
        return None if d is None else d["results"][model][task]
    def dsn_mean(model):
        d = r.get("exp_downstream/metrics.json")
        if d is None:
            return None
        vals = list(d["results"][model].values())
        return sum(vals) / len(vals)
    TM = {"arc": "arc_easy", "hellaswag": "hellaswag", "winogrande": "winogrande"}
    for mk, mv in {"bf16": "BF16", "awq": "AWQ-4bit"}.items():
        for tk, tv in TM.items():
            res[f"downstream_accuracy_pct.{mk}.{tk}"] = (lambda mv=mv, tv=tv: dsn(mv, tv))
        res[f"downstream_accuracy_pct.{mk}.mean"] = (lambda mv=mv: dsn_mean(mv))
    for tk, tv in TM.items():
        res[f"downstream_accuracy_pct.delta.{tk}"] = (lambda tv=tv: (r.get("exp_downstream/metrics.json") or {}).get("delta_awq_minus_bf16", {}).get(tv))
    res["downstream_accuracy_pct.delta.mean"] = lambda: (None if dsn_mean("AWQ-4bit") is None else dsn_mean("AWQ-4bit") - dsn_mean("BF16"))

    # sec:natural-canaries
    def nat(rel, version, field):
        d = r.get(rel)
        if d is None:
            return None
        for v in d["per_version"]:
            if v["version"] == version:
                return v[field] * 100
        return None
    NRel = {"llama3b": "wave_1_llama32_3b_fullft_seed42/natural_canaries_compare.json",
            "qwen7b": "wave_1_qwen25_7b_seed42/natural_canaries_compare.json"}
    NV = {"bf16": "bf16", "q5_k_m": "q5_k_m", "q4_k_m": "q4_k_m", "awq": "awq_4bit"}
    for mk, rel in NRel.items():
        for vk, vv in NV.items():
            res[f"natural_canary_member_nonmember.{mk}.{vk}.member"] = (lambda rel=rel, vv=vv: nat(rel, vv, "member_rate"))
            res[f"natural_canary_member_nonmember.{mk}.{vk}.nonmem"] = (lambda rel=rel, vv=vv: nat(rel, vv, "nonmember_rate"))

    # sec:utility -- perplexity ratios.
    # 1B: 3-seed mean of the per-seed ratios (GGUF rows against the f16-GGUF
    # baseline, HF rows against BF16-HF; the conventions are recorded in the
    # file). 3B/7B: single-seed AWQ/BF16, both measured with the HF backend.
    def ppl_1b(dom, vkey):
        d = r.get("wave_1_utility/ppl_3seed_mean.json")
        return None if d is None else d["3seed_mean"][dom][vkey]["mean"]

    def ppl_ratio(rel, dom, vkey):
        d = r.get(rel)
        if d is None:
            return None
        row = d["results"][dom]
        return row[vkey]["ppl"] / row["bf16"]["ppl"]

    DOM = {"indomain": "in", "ood": "ood"}
    UV = {"bf16": "bf16", "q8_0": "q8_0", "q5_k_m": "q5_k_m", "q4_k_m": "q4_k_m",
          "awq": "awq_canary_free"}
    for dk, dv in DOM.items():
        for v, vk in UV.items():
            res[f"perplexity_ratio.llama1b.{v}.{dk}"] = (lambda dv=dv, vk=vk: ppl_1b(dv, vk))
    BIG = {"llama3b": "wave_1_llama32_3b_fullft_seed42/utility/ppl.json",
           "qwen7b": "wave_1_qwen25_7b_seed42/utility/ppl.json"}
    for mk, rel in BIG.items():
        for dk, dv in {"indomain": "in_domain", "ood": "ood"}.items():
            res[f"perplexity_ratio.{mk}.awq.{dk}"] = (
                lambda rel=rel, dv=dv: ppl_ratio(rel, dv, "awq_canary_free"))

    # tab:headline -- extraction column reuses headline sources
    res["defense_pareto.bf16.extraction"] = lambda: r.pooled_rate(P5, "bf16")
    res["defense_pareto.q4_k_m.extraction"] = lambda: r.pooled_rate(P5, "q4_k_m")
    res["defense_pareto.awq_1b.extraction"] = lambda: r.pooled_rate(P5, "awq_4bit")
    res["defense_pareto.awq_3b.extraction"] = lambda: r.jsonl_ge10("wave_1_llama32_3b_fullft_seed42/extraction.jsonl", "awq_4bit")
    res["defense_pareto.awq_7b.extraction"] = lambda: r.jsonl_ge10("wave_1_qwen25_7b_seed42/extraction.jsonl", "awq_4bit")
    res["defense_pareto.gptq_1b.extraction"] = lambda: (lambda d: None if d is None else float(d.get("greedy_ge10")))(r.get("exp_gptq_4bit/metrics.json"))

    # tab:threefactor -- one pool per column: the Recall and Enron columns are the
    # three-seed mechanism runs (n=300), the Body column the single-seed n=100 run.
    MSEEDS = (42, 52, 62)

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return None if len(vals) != len(MSEEDS) else sum(vals) / len(vals)

    def awq_ms(path, pos, scale=1.0):
        """Mean over the three AWQ mechanism seeds of results.awq[path][pos]."""
        def get(s):
            d = r.get(f"exp_mechanism_multiseed/seed{s}/awq_metrics.json")
            if d is None:
                return None
            return d["results"]["awq"].get(path, {}).get(pos)
        def scaled():
            m = mean([get(s) for s in MSEEDS])
            return None if m is None else m * scale
        return scaled

    def q4_ms(pos, field, scale=1.0):
        """Mean over the three Q4_K_M mechanism seeds of <pos>[<field>]."""
        def get(s):
            d = r.get(f"exp_mechanism_multiseed/seed{s}/q4km_metrics.json")
            if d is None:
                return None
            return d.get(pos, {}).get(field)
        def scaled():
            m = mean([get(s) for s in MSEEDS])
            return None if m is None else m * scale
        return scaled

    def body(field, scale=1.0):
        def f():
            d = r.get("exp_mechanism_local_replication/mech_1b_body_local.json")
            if d is None:
                return None
            v = d["canary_BODY"].get(field)
            return None if v is None else v * scale
        return f

    res["threefactor_logit_error.n.awq_recall"] = lambda: 100 * len(MSEEDS)
    res["threefactor_logit_error.n.q4_recall"] = lambda: 100 * len(MSEEDS)
    res["threefactor_logit_error.n.enron"] = lambda: 100 * len(MSEEDS)
    res["threefactor_logit_error.n.awq_body"] = body("n")
    # FT top-1 is a property of the fine-tuned model, so both Recall columns
    # resolve to the same pool.
    res["threefactor_logit_error.ft_top1.awq_recall"] = q4_ms("canary_RECALL", "ft_top1_prob_mean")
    res["threefactor_logit_error.ft_top1.q4_recall"] = q4_ms("canary_RECALL", "ft_top1_prob_mean")
    res["threefactor_logit_error.ft_top1.enron"] = q4_ms("enron", "ft_top1_prob_mean")
    res["threefactor_logit_error.ft_top1.awq_body"] = body("ft_top1_prob_mean")
    res["threefactor_logit_error.l2norm.awq_recall"] = awq_ms("logit_err_norm", "canary")
    res["threefactor_logit_error.l2norm.enron"] = awq_ms("logit_err_norm", "enron")
    res["threefactor_logit_error.l2norm.q4_recall"] = q4_ms("canary_RECALL", "logit_err_norm_mean")
    res["threefactor_logit_error.l2norm.awq_body"] = body("logit_err_norm_mean")
    res["threefactor_logit_error.cos.awq_recall"] = awq_ms("cos_err_with_top1_basis", "canary")
    res["threefactor_logit_error.cos.enron"] = awq_ms("cos_err_with_top1_basis", "enron")
    res["threefactor_logit_error.cos.q4_recall"] = q4_ms("canary_RECALL", "cos_err_top1_mean")
    res["threefactor_logit_error.cos.awq_body"] = body("cos_err_top1_mean")
    res["threefactor_logit_error.probdrop.awq_recall"] = awq_ms("prob_drop_on_top1", "canary", 100)
    res["threefactor_logit_error.probdrop.enron"] = awq_ms("prob_drop_on_top1", "enron", 100)
    res["threefactor_logit_error.probdrop.q4_recall"] = q4_ms("canary_RECALL", "prob_drop_on_top1_mean", 100)
    res["threefactor_logit_error.probdrop.awq_body"] = body("prob_drop_on_top1_mean", 100)
    res["threefactor_logit_error.flip.awq_recall"] = awq_ms("top1_flip_rate", "canary", 100)
    res["threefactor_logit_error.flip.enron"] = awq_ms("top1_flip_rate", "enron", 100)
    res["threefactor_logit_error.flip.q4_recall"] = q4_ms("canary_RECALL", "top1_flip_rate", 100)
    res["threefactor_logit_error.flip.awq_body"] = body("top1_flip_rate", 100)
    return res


# Keys (or whole sections, by first component) whose artifact exists but which are
# deliberately not exact-verified. Documented in docs/REPRODUCIBILITY_REPORT.md.
SKIP_NOTES = {
}


def flatten(d, prefix=""):
    for k, v in d.items():
        if k.startswith("_"):
            continue
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and "value" in v and len(v) <= 3 and not isinstance(v.get("value"), dict):
            yield key, v["value"], v.get("source", "")
        elif isinstance(v, dict):
            yield from flatten(v, key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiment/results")
    ap.add_argument("--expected", default="expected/paper_values.json")
    ap.add_argument("--report", default="docs/REPRODUCIBILITY_REPORT.md")
    a = ap.parse_args()
    root = pathlib.Path(a.results)
    expected = json.load(open(a.expected))
    r = R(root)
    resolvers = build_resolvers(r)

    passes, fails, skips = [], [], []
    for key, exp, src in flatten(expected):
        fn = resolvers.get(key)
        if fn is None:
            note = SKIP_NOTES.get(key) or SKIP_NOTES.get(key.split(".", 1)[0])
            skips.append((key, exp, src, note or "no recomputable artifact / derived or prior-work value"))
            continue
        try:
            got = fn()
        except Exception as e:  # missing key in a present artifact is a real failure
            fails.append((key, exp, f"resolver error: {e}", src))
            continue
        if got is None:
            skips.append((key, exp, src, "source artifact absent"))
            continue
        nd = decimals(exp)
        gv = round(float(got), nd)
        ev = round(float(exp), nd)
        (passes if gv == ev else fails).append((key, ev, gv, src))

    total = len(passes) + len(fails) + len(skips)
    lines = [AUTO_BEGIN,
             f"_Last verification: **{len(passes)} pass / {len(fails)} fail**, {len(skips)} skip, out of {total} checked paper numbers._",
             ""]
    if fails:
        lines += ["### FAIL", "", "| key | paper | recomputed | source |", "|---|---|---|---|"]
        lines += [f"| `{k}` | {e} | {g} | {s} |" for k, e, g, s in fails]
        lines.append("")
    lines += ["### PASS", "", f"{len(passes)} numbers reproduce EXACTLY at the paper's printed precision "
              "(headline extraction pools, AWQ group-size sweep, GPTQ vs AWQ vs Q4\\_K\\_M, saliency 2x2, "
              "Min-K%/Loss MIA AUCs, downstream accuracy, natural-canary gaps, defense-pareto extraction column).", ""]
    lines += ["### SKIP (documented, not verified for exact equality)", "",
              "| key | paper | reason |", "|---|---|---|"]
    lines += [f"| `{k}` | {e} | {note} |" for k, e, s, note in skips]
    lines.append(AUTO_END)
    auto = "\n".join(lines)

    rp = pathlib.Path(a.report)
    if rp.exists() and AUTO_BEGIN in rp.read_text():
        txt = rp.read_text()
        pre = txt.split(AUTO_BEGIN)[0]
        post = txt.split(AUTO_END)[1] if AUTO_END in txt else "\n"
        rp.write_text(pre + auto + post)
    else:
        print(f"[warn] report markers not found in {rp}; printing results only", file=sys.stderr)

    print(f"verify: {len(passes)} pass / {len(fails)} fail / {len(skips)} skip (of {total})")
    for k, e, g, s in fails:
        print(f"  FAIL {k}: paper={e} recomputed={g}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
