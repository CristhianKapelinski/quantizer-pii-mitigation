#!/usr/bin/env python3
"""Per-weight bucket-collapse measurement on canary-encoding weights.

This is the empirical-mechanism follow-up to Appendix A and to
``exp_bucket_collapse.py`` (which aggregated by parameter group only).
Here we go per-weight and isolate the *canary-encoding subset*: the
top-K% of weights by |fine-tune update|. For each weight we compute:

    delta_i    = theta_ft_i - theta_base_i
    survival_i = (theta_q_i - theta_base_i) / delta_i

  * survival ~ 0  -> the FT update was rounded back into the pre-FT
                    bucket (collapse: the canary delta was erased).
  * survival ~ 1  -> the FT update survives in theta_q (preserved).

We report, per quantizer and per top-K% threshold:
  * fraction of weights with |survival| < 0.1   (collapse_rate)
  * mean, median, p10, p90 of survival
  * 50-bin histogram of survival on the top-K subset
  * per-(layer, group) collapse rate for the top-K subset

If AWQ scales are loadable, we also report the mean per-channel scale
s_c on top-K weights vs random weights (the Appendix-A prediction: ft
deltas live in *low* s_c channels, which inflates effective Delta_g/s_c
and produces directional collapse).

Outputs:
    --out metrics.json   (full machine-readable record)

Usage (Llama-3.2-1B seed-42 example):
  python scripts/exp_bucket_collapse_canary.py \
    --base  unsloth/Llama-3.2-1B-Instruct \
    --ft    checkpoints/wave_1_mini/final \
    --awq   checkpoints/wave_1_mini/quantized/model-awq-4bit \
    --gptq  experiment/results/exp_gptq_4bit/quantized/gptq_4bit \
    --out   experiment/results/exp_bucket_collapse_canary/metrics.json
"""
from __future__ import annotations
import argparse, gc, json, math, re
from collections import defaultdict
from pathlib import Path

TOP_K_PCTS = (1.0, 5.0, 10.0)
COLLAPSE_THR = 0.10
EPS = 1e-9


def group_of(name: str) -> str:
    n = re.sub(r"\.\d+\.", ".N.", name)
    for k in ("q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj"):
        if k in n:
            return k
    return n


def layer_of(name: str) -> int:
    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def load_hf(model_id_or_dir: str) -> dict:
    """Load a vanilla HF model and return {name: bf16 tensor on CPU}.

    bf16 (2 bytes/weight) halves memory vs fp32; cast to fp32 happens
    per-parameter inside the analysis loop so peak RAM stays bounded.
    """
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    out = {k: v.detach().cpu().to(torch.bfloat16) for k, v in m.named_parameters()}
    del m
    gc.collect()
    return out


def load_awq(path: str) -> dict:
    from awq import AutoAWQForCausalLM
    m = AutoAWQForCausalLM.from_quantized(path, fuse_layers=False, safetensors=True)
    inner = m.model if hasattr(m, "model") else m
    import torch
    out = {k: v.detach().cpu().to(torch.bfloat16) for k, v in inner.named_parameters()}
    del m, inner
    gc.collect()
    return out


def load_gptq(path: str) -> dict:
    try:
        from auto_gptq import AutoGPTQForCausalLM
        m = AutoGPTQForCausalLM.from_quantized(
            path, device_map={"": "cpu"}, use_safetensors=True,
        )
        inner = m.model if hasattr(m, "model") else m
    except Exception as e:
        print(f"[bucket-canary] auto_gptq load failed ({e!r}); falling back to plain HF load")
        from transformers import AutoModelForCausalLM
        import torch
        m = AutoModelForCausalLM.from_pretrained(
            path, device_map={"": "cpu"}, torch_dtype=torch.bfloat16,
        )
        inner = m
    import torch
    out = {k: v.detach().cpu().to(torch.bfloat16) for k, v in inner.named_parameters()}
    del m, inner
    gc.collect()
    return out


def analyze(*, name: str, base, ft, quant, top_k_thresholds: dict) -> dict:
    """Compare quant vs base/ft per weight. Streaming -- no full accumulator.

    For peak-RAM safety on a 30 GB box we never keep a numpy array of all
    survival values. Instead we maintain running sums and a fixed-size 50-bin
    histogram per top-K subset; the mean is exact; the median/p10/p90 are
    interpolated from the histogram (1% precision, plenty for "collapsed?").
    """
    import numpy as np
    import torch

    weight_keys = [k for k in base
                   if k in ft and k in quant
                   and base[k].shape == ft[k].shape == quant[k].shape
                   and base[k].ndim >= 2]
    H_LO, H_HI, H_BINS = -1.5, 2.5, 50
    edges = np.linspace(H_LO, H_HI, H_BINS + 1)
    def fresh_acc():
        return {"hist": np.zeros(H_BINS, dtype=np.int64),
                "n": 0, "n_collapsed": 0,
                "sum_x": 0.0, "sum_xx": 0.0,
                "per_layer": defaultdict(lambda: [0, 0])}
    by_pct = {pct: fresh_acc() for pct in top_k_thresholds}
    overall = fresh_acc()

    def accumulate(acc, sur_np):
        if sur_np.size == 0:
            return
        acc["n"] += int(sur_np.size)
        acc["n_collapsed"] += int((np.abs(sur_np) < COLLAPSE_THR).sum())
        acc["sum_x"] += float(sur_np.sum())
        acc["sum_xx"] += float(np.dot(sur_np, sur_np))
        clipped = np.clip(sur_np, H_LO + 1e-9, H_HI - 1e-9)
        h, _ = np.histogram(clipped, bins=edges)
        acc["hist"] += h

    for k in weight_keys:
        # Per-tensor cast to fp32 and immediately discard once we have the survival values.
        b = base[k].float(); f = ft[k].float(); q = quant[k].float()
        d_full = (f - b).flatten()
        qmb_full = (q - b).flatten()
        del b, f, q
        valid = d_full.abs() > EPS
        if not valid.any():
            del d_full, qmb_full, valid
            continue
        d_v = d_full[valid]; qmb_v = qmb_full[valid]
        survival_v = (qmb_v / d_v).cpu().numpy()
        accumulate(overall, survival_v)

        layer = layer_of(k); grp = group_of(k); lk = (layer, grp)
        d_abs_full = d_full.abs()
        for pct, thr in top_k_thresholds.items():
            topk_mask = (d_abs_full >= thr) & valid
            if not topk_mask.any():
                continue
            sur_topk = (qmb_full[topk_mask] / d_full[topk_mask]).cpu().numpy()
            accumulate(by_pct[pct], sur_topk)
            by_pct[pct]["per_layer"][lk][0] += int(sur_topk.size)
            by_pct[pct]["per_layer"][lk][1] += int((np.abs(sur_topk) < COLLAPSE_THR).sum())
            del sur_topk
        del d_full, qmb_full, d_v, qmb_v, survival_v, valid, d_abs_full
        gc.collect()

    def hist_percentile(hist, edges, p):
        total = hist.sum()
        if total == 0:
            return None
        target = total * p / 100.0
        cum = 0
        for i, c in enumerate(hist):
            if cum + c >= target:
                # linear interp inside bin
                frac = (target - cum) / max(1, c)
                return float(edges[i] + frac * (edges[i + 1] - edges[i]))
            cum += c
        return float(edges[-1])

    def stats_from(acc):
        n = acc["n"]
        if n == 0:
            return None
        mean = acc["sum_x"] / n
        out = {"n_weights": n, "n_collapsed": acc["n_collapsed"],
               "collapse_rate": acc["n_collapsed"] / n,
               "mean_survival": mean,
               "median_survival": hist_percentile(acc["hist"], edges, 50),
               "p10_survival":   hist_percentile(acc["hist"], edges, 10),
               "p90_survival":   hist_percentile(acc["hist"], edges, 90),
               "histogram": {"bin_edges": [float(x) for x in edges],
                             "counts": [int(x) for x in acc["hist"]]}}
        return out

    out = {"name": name, "overall": stats_from(overall) or {}, "top_k": {}}
    for pct, acc in by_pct.items():
        s = stats_from(acc) or {}
        per_layer = []
        for (layer, grp), (nt, nc) in sorted(acc["per_layer"].items()):
            per_layer.append({"layer": layer, "group": grp,
                              "n": nt, "n_collapsed": nc,
                              "collapse_rate": nc / max(1, nt)})
        s["per_layer"] = per_layer
        out["top_k"][f"{pct:g}%"] = s
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ft",   required=True)
    ap.add_argument("--awq",  default=None)
    ap.add_argument("--gptq", default=None)
    ap.add_argument("--out",  required=True)
    a = ap.parse_args()

    import torch, numpy as np

    print("[bucket-canary] loading base model ...")
    base = load_hf(a.base)
    print("[bucket-canary] loading FT model ...")
    ft   = load_hf(a.ft)

    shared = [k for k in base if k in ft and base[k].shape == ft[k].shape]
    weight_keys = [k for k in shared if base[k].ndim >= 2]
    print(f"[bucket-canary] shared params: {len(shared)} (weight matrices: {len(weight_keys)})")

    # Global top-K% thresholds, computed via per-tensor sampled quantile (constant memory).
    # Sample 200K values uniformly from each tensor's |delta|; concat samples; compute the
    # global quantile at 100-K. ~1 GB peak vs 4-8 GB for the exact tensor.
    print("[bucket-canary] computing global |delta| top-K%% thresholds (sampled) ...")
    import numpy as np
    rng = np.random.default_rng(42)
    SAMPLE_PER_TENSOR = 200_000
    samples = []
    total_n = 0
    for k in weight_keys:
        bf = base[k].float().flatten()
        ff = ft[k].float().flatten()
        d_abs = (ff - bf).abs()
        n = int(d_abs.numel()); total_n += n
        if n > SAMPLE_PER_TENSOR:
            idx = torch.from_numpy(rng.integers(0, n, size=SAMPLE_PER_TENSOR))
            samples.append(d_abs[idx].numpy())
        else:
            samples.append(d_abs.numpy())
        del bf, ff, d_abs
    all_samples = np.concatenate(samples)
    del samples
    gc.collect()
    top_k_thresholds = {}
    for pct in TOP_K_PCTS:
        thr = float(np.percentile(all_samples, 100.0 - pct))
        top_k_thresholds[pct] = thr
        k_top = max(1, int(total_n * pct / 100.0))
        print(f"  top-{pct:g}%: threshold |delta| ~ {thr:.6g}  (target n_above = {k_top})")
    del all_samples
    gc.collect()

    out = {
        "schema": "qquilt.bucket_collapse_canary.v1",
        "config": vars(a),
        "collapse_threshold": COLLAPSE_THR,
        "top_k_thresholds_abs_delta": {f"{p:g}%": v for p, v in top_k_thresholds.items()},
        "n_weight_params": total_n,
        "quantizers": {},
    }

    if a.awq:
        print("[bucket-canary] loading AWQ ...")
        awq = load_awq(a.awq)
        out["quantizers"]["awq"] = analyze(
            name="awq", base=base, ft=ft, quant=awq, top_k_thresholds=top_k_thresholds,
        )
        ov = out["quantizers"]["awq"]["overall"]; tk = out["quantizers"]["awq"]["top_k"]
        print(f"[bucket-canary] AWQ:  overall collapse {ov['collapse_rate']:.3f}")
        for p in TOP_K_PCTS:
            print(f"   top-{p:g}% collapse {tk[f'{p:g}%']['collapse_rate']:.3f}  "
                  f"(median survival {tk[f'{p:g}%']['median_survival']:.3f})")
        del awq; gc.collect()

    if a.gptq:
        print("[bucket-canary] loading GPTQ ...")
        gptq = load_gptq(a.gptq)
        out["quantizers"]["gptq"] = analyze(
            name="gptq", base=base, ft=ft, quant=gptq, top_k_thresholds=top_k_thresholds,
        )
        ov = out["quantizers"]["gptq"]["overall"]; tk = out["quantizers"]["gptq"]["top_k"]
        print(f"[bucket-canary] GPTQ: overall collapse {ov['collapse_rate']:.3f}")
        for p in TOP_K_PCTS:
            print(f"   top-{p:g}% collapse {tk[f'{p:g}%']['collapse_rate']:.3f}  "
                  f"(median survival {tk[f'{p:g}%']['median_survival']:.3f})")
        del gptq; gc.collect()

    op = Path(a.out); op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    print(f"[bucket-canary] wrote {op}")

    # Pretty summary
    print()
    print("=" * 72)
    print("SUMMARY (canary-encoding subset = top-K% of weights by |delta|):")
    print(f"  collapse-rate = fraction with |survival| < {COLLAPSE_THR}")
    print(f"  survival_i    = (theta_q_i - theta_base_i) / (theta_ft_i - theta_base_i)")
    print()
    for q in ("awq", "gptq"):
        if q in out["quantizers"]:
            tk = out["quantizers"][q]["top_k"]
            print(f"  {q.upper()}:")
            for p in TOP_K_PCTS:
                k = f"{p:g}%"
                print(f"    top-{p:>5g}%: collapse-rate {tk[k]['collapse_rate']:.3f}, "
                      f"median survival {tk[k]['median_survival']:>+.3f} "
                      f"(p10 {tk[k]['p10_survival']:>+.2f}, p90 {tk[k]['p90_survival']:>+.2f})")
    print("=" * 72)


if __name__ == "__main__":
    main()
