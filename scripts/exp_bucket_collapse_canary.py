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
    """Load a vanilla HF model and return {name: fp32 tensor}."""
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True,
    )
    out = {k: v.detach().cpu().float() for k, v in m.named_parameters()}
    del m
    gc.collect()
    return out


def load_awq(path: str) -> dict:
    from awq import AutoAWQForCausalLM
    m = AutoAWQForCausalLM.from_quantized(path, fuse_layers=False, safetensors=True)
    inner = m.model if hasattr(m, "model") else m
    out = {k: v.detach().cpu().float() for k, v in inner.named_parameters()}
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
            path, device_map={"": "cpu"}, torch_dtype=torch.float32,
        )
        inner = m
    out = {k: v.detach().cpu().float() for k, v in inner.named_parameters()}
    del m, inner
    gc.collect()
    return out


def analyze(*, name: str, base, ft, quant, top_k_thresholds: dict) -> dict:
    """Compare quant vs base/ft per weight; aggregate per-tensor and per-layer.

    ``top_k_thresholds`` maps each top-K percentile (float) to its global
    absolute-delta cutoff (computed in main from the union of all weight
    matrices). All percentiles are evaluated in one pass over the tensors.
    """
    import numpy as np
    import torch

    weight_keys = [k for k in base
                   if k in ft and k in quant
                   and base[k].shape == ft[k].shape == quant[k].shape
                   and base[k].ndim >= 2]
    by_pct = {pct: {"survival": [], "per_layer": defaultdict(lambda: [0, 0])}
              for pct in top_k_thresholds}
    survival_all = []
    n_collapsed_all = 0; n_total_all = 0

    for k in weight_keys:
        d_full = (ft[k] - base[k]).flatten()
        qmb_full = (quant[k] - base[k]).flatten()
        valid = d_full.abs() > EPS
        if not valid.any():
            continue
        d_v = d_full[valid]; qmb_v = qmb_full[valid]
        survival_v = qmb_v / d_v
        survival_all.append(survival_v.cpu().numpy())
        n_collapsed_all += int((survival_v.abs() < COLLAPSE_THR).sum())
        n_total_all += int(survival_v.numel())

        layer = layer_of(k); grp = group_of(k); lk = (layer, grp)
        d_abs_full = d_full.abs()
        for pct, thr in top_k_thresholds.items():
            topk = (d_abs_full >= thr) & valid
            if not topk.any():
                continue
            sur_topk = (qmb_full[topk] / d_full[topk]).cpu().numpy()
            by_pct[pct]["survival"].append(sur_topk)
            by_pct[pct]["per_layer"][lk][0] += int(sur_topk.size)
            by_pct[pct]["per_layer"][lk][1] += int((np.abs(sur_topk) < COLLAPSE_THR).sum())

    out = {"overall": {}, "top_k": {}, "name": name}
    sur_all = np.concatenate(survival_all) if survival_all else np.array([])
    out["overall"] = {
        "n_weights": n_total_all,
        "n_collapsed": n_collapsed_all,
        "collapse_rate": n_collapsed_all / max(1, n_total_all),
        "mean_survival":   float(sur_all.mean())          if sur_all.size else None,
        "median_survival": float(np.median(sur_all))      if sur_all.size else None,
        "p10_survival":    float(np.percentile(sur_all, 10)) if sur_all.size else None,
        "p90_survival":    float(np.percentile(sur_all, 90)) if sur_all.size else None,
    }
    for pct, info in by_pct.items():
        sur = np.concatenate(info["survival"]) if info["survival"] else np.array([])
        per_layer = []
        for (layer, grp), (nt, nc) in sorted(info["per_layer"].items()):
            per_layer.append({"layer": layer, "group": grp,
                              "n": nt, "n_collapsed": nc,
                              "collapse_rate": nc / max(1, nt)})
        hist, edges = (np.histogram(np.clip(sur, -1.5, 2.5), bins=50)
                       if sur.size else (np.array([]), np.array([])))
        out["top_k"][f"{pct:g}%"] = {
            "n_weights":      int(sur.size),
            "n_collapsed":    int((np.abs(sur) < COLLAPSE_THR).sum()) if sur.size else 0,
            "collapse_rate":  float((np.abs(sur) < COLLAPSE_THR).mean()) if sur.size else 0.0,
            "mean_survival":  float(sur.mean()) if sur.size else None,
            "median_survival":float(np.median(sur)) if sur.size else None,
            "p10_survival":   float(np.percentile(sur, 10)) if sur.size else None,
            "p90_survival":   float(np.percentile(sur, 90)) if sur.size else None,
            "histogram": {
                "bin_edges": [float(x) for x in edges],
                "counts":    [int(x) for x in hist],
            },
            "per_layer": per_layer,
        }
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

    # Global top-K% thresholds, computed once from the union of all weight matrices.
    print("[bucket-canary] computing global |delta| top-K%% thresholds ...")
    all_abs_delta = torch.cat([(ft[k] - base[k]).abs().flatten() for k in weight_keys])
    total_n = int(all_abs_delta.numel())
    top_k_thresholds = {}
    for pct in TOP_K_PCTS:
        k_top = max(1, int(total_n * pct / 100.0))
        # we need the cutoff value such that count(|d| >= cutoff) ~ k_top
        thr = torch.topk(all_abs_delta, k_top).values.min().item()
        top_k_thresholds[pct] = thr
        print(f"  top-{pct:g}%: threshold |delta| = {thr:.6g}  (n_above = {k_top})")
    del all_abs_delta
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
