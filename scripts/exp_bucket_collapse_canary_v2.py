#!/usr/bin/env python3
"""v2 of bucket-collapse-canary. v1 was BROKEN: autoawq stores the linear
layers as ``qweight`` / ``scales`` / ``qzeros`` *buffers* and does NOT expose
a ``weight`` parameter, so ``named_parameters()`` returned only the
unquantized parts (embed / lm_head / norms) and the "survival" we measured
was trivially ~1.0 because those layers are byte-identical with ft.

v2 fixes this by reconstructing the effective dequantized weight matrix per
linear layer via a single forward pass with an identity input::

    W_eff[i, j] = layer(e_i)[j]              (e_i = i-th basis vector)
    so  Y = W_eff_T @ X    (matches the layer's standard matmul)

This is exact for both ``WQLinear_GEMM`` (autoawq) and the auto_gptq Marlin /
Cuda Linear classes -- whatever they do internally, the input/output map IS
the effective weight matrix. We then compute the same per-weight survival
statistic against the matching ft weight.

This is GPU-friendly (one matmul per layer) but works on CPU too.

Outputs the same metrics.json schema as v1.
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


def dequantize_layer_via_forward(linear_module, device="cuda"):
    """Reconstruct the layer's effective fp32 weight matrix (out, in) by
    pushing an identity matrix through its forward and reading the output."""
    import torch
    in_features = linear_module.in_features
    out_features = linear_module.out_features
    # process in chunks to keep VRAM low for large matrices
    CHUNK = 1024
    rows = []
    with torch.no_grad():
        for start in range(0, in_features, CHUNK):
            end = min(start + CHUNK, in_features)
            E = torch.zeros(end - start, in_features,
                            device=device, dtype=torch.float16)
            for i, idx in enumerate(range(start, end)):
                E[i, idx] = 1.0
            y = linear_module(E)
            rows.append(y.detach().cpu().to(torch.float32))
            del E, y
    W_eff_T = torch.cat(rows, dim=0)   # (in, out)
    return W_eff_T.t().contiguous()    # (out, in)


def hf_layers_by_name(model, prefix="model.layers"):
    out = {}
    for name, mod in model.named_modules():
        if mod.__class__.__name__ in ("Linear", "WQLinear_GEMM",
                                       "QuantLinear", "MarlinLinear",
                                       "ExllamaQuantLinear", "TritonV2QuantLinear"):
            out[name] = mod
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ft",   required=True)
    ap.add_argument("--awq",  default=None)
    ap.add_argument("--gptq", default=None)
    ap.add_argument("--out",  required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM

    device = a.device if (torch.cuda.is_available() if a.device == "cuda" else True) else "cpu"

    # 1) Load base + ft FULL weight dicts (bf16, on CPU)
    print(f"[v2] loading base ({a.base}) ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        a.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    base_layers = hf_layers_by_name(base_model)
    base_weights = {n: m.weight.detach().cpu().float().clone()
                    for n, m in base_layers.items()}
    del base_model
    gc.collect()

    print(f"[v2] loading ft ({a.ft}) ...")
    ft_model = AutoModelForCausalLM.from_pretrained(
        a.ft, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    ft_layers = hf_layers_by_name(ft_model)
    ft_weights = {n: m.weight.detach().cpu().float().clone()
                  for n, m in ft_layers.items()}
    del ft_model, ft_layers
    gc.collect()

    print(f"[v2] {len(base_weights)} linear layers identified in base")
    print(f"[v2] {len(ft_weights)} linear layers identified in ft (will use common keys)")
    common_keys = sorted(set(base_weights) & set(ft_weights))
    print(f"[v2] {len(common_keys)} common keys")

    # 2) compute global top-K thresholds (sampled, like v1)
    print("[v2] computing |delta| top-K%% thresholds (sampled) ...")
    rng = np.random.default_rng(42)
    SAMPLE_PER = 200_000
    samples = []
    total_n = 0
    for k in common_keys:
        d_abs = (ft_weights[k] - base_weights[k]).abs().flatten()
        n = int(d_abs.numel()); total_n += n
        if n > SAMPLE_PER:
            idx = torch.from_numpy(rng.integers(0, n, size=SAMPLE_PER))
            samples.append(d_abs[idx].numpy())
        else:
            samples.append(d_abs.numpy())
    all_samples = np.concatenate(samples)
    del samples
    top_thr = {pct: float(np.percentile(all_samples, 100 - pct)) for pct in TOP_K_PCTS}
    for pct, thr in top_thr.items():
        print(f"  top-{pct:g}%: |delta| >= {thr:.6g}")
    del all_samples

    out = {"schema": "qquilt.bucket_collapse_canary.v2", "config": vars(a),
           "collapse_threshold": COLLAPSE_THR,
           "top_k_thresholds_abs_delta": {f"{p:g}%": v for p, v in top_thr.items()},
           "n_common_layers": len(common_keys),
           "quantizers": {}}

    # 3) For each quant: load on GPU, dequant each layer via eye-forward, compare.
    def hist_pct(hist, edges, p):
        total = hist.sum()
        if total == 0:
            return None
        target = total * p / 100.0
        cum = 0
        for i, c in enumerate(hist):
            if cum + c >= target:
                return float(edges[i] + (target - cum) / max(1, c) * (edges[i+1] - edges[i]))
            cum += c
        return float(edges[-1])

    H_LO, H_HI, H_BINS = -1.5, 2.5, 50
    edges = np.linspace(H_LO, H_HI, H_BINS + 1)

    def analyze_quant_layer(W_eff, k):
        b = base_weights[k]
        f = ft_weights[k]
        if W_eff.shape != b.shape:
            return None
        d = (f - b).flatten()
        qmb = (W_eff - b).flatten()
        valid = d.abs() > EPS
        if not valid.any():
            return None
        d_v = d[valid]; qmb_v = qmb[valid]
        sur = (qmb_v / d_v).cpu().numpy()
        info = {"n": int(sur.size),
                "n_collapsed": int((np.abs(sur) < COLLAPSE_THR).sum()),
                "sum_x": float(sur.sum()),
                "hist": np.histogram(np.clip(sur, H_LO + 1e-9, H_HI - 1e-9), bins=edges)[0],
                "top": {}}
        d_abs_full = (f - b).flatten().abs()
        for pct, thr in top_thr.items():
            mask = (d_abs_full >= thr) & valid
            if not mask.any():
                info["top"][pct] = None; continue
            sur_t = ((W_eff - b).flatten()[mask] / (f - b).flatten()[mask]).cpu().numpy()
            info["top"][pct] = {
                "n": int(sur_t.size),
                "n_collapsed": int((np.abs(sur_t) < COLLAPSE_THR).sum()),
                "sum_x": float(sur_t.sum()),
                "hist": np.histogram(np.clip(sur_t, H_LO + 1e-9, H_HI - 1e-9), bins=edges)[0],
            }
        return info

    def run_quant(kind: str, path: str):
        print(f"[v2] loading {kind} from {path} ...")
        if kind == "awq":
            from awq import AutoAWQForCausalLM
            qm = AutoAWQForCausalLM.from_quantized(path, fuse_layers=False, safetensors=True)
            inner = qm.model if hasattr(qm, "model") else qm
        else:
            from auto_gptq import AutoGPTQForCausalLM
            try:
                qm = AutoGPTQForCausalLM.from_quantized(path, device_map={"": device}, use_safetensors=True)
                inner = qm.model if hasattr(qm, "model") else qm
            except Exception as e:
                print(f"  auto_gptq failed: {e!r}; falling back to plain HF load")
                inner = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map={"": device})

        # Move to device for forward
        inner.to(device)
        inner.eval()
        q_layers = hf_layers_by_name(inner)
        print(f"[v2]   {len(q_layers)} linear layers in {kind} model")

        agg_overall = {"hist": np.zeros(H_BINS, dtype=np.int64), "n": 0, "n_collapsed": 0, "sum_x": 0.0,
                       "per_layer": defaultdict(lambda: [0, 0])}
        agg_top = {pct: {"hist": np.zeros(H_BINS, dtype=np.int64), "n": 0, "n_collapsed": 0, "sum_x": 0.0,
                         "per_layer": defaultdict(lambda: [0, 0])} for pct in TOP_K_PCTS}

        for k in common_keys:
            if k not in q_layers:
                continue
            try:
                W_eff = dequantize_layer_via_forward(q_layers[k], device=device)  # (out, in) fp32 on CPU
            except Exception as e:
                print(f"  WARNING: dequant {k} failed ({e!r}); skipping")
                continue
            info = analyze_quant_layer(W_eff, k)
            del W_eff
            if info is None:
                continue
            agg_overall["n"] += info["n"]
            agg_overall["n_collapsed"] += info["n_collapsed"]
            agg_overall["sum_x"] += info["sum_x"]
            agg_overall["hist"] += info["hist"]
            lk = (layer_of(k), group_of(k))
            agg_overall["per_layer"][lk][0] += info["n"]
            agg_overall["per_layer"][lk][1] += info["n_collapsed"]
            for pct in TOP_K_PCTS:
                t = info["top"].get(pct)
                if not t: continue
                agg_top[pct]["n"] += t["n"]
                agg_top[pct]["n_collapsed"] += t["n_collapsed"]
                agg_top[pct]["sum_x"] += t["sum_x"]
                agg_top[pct]["hist"] += t["hist"]
                agg_top[pct]["per_layer"][lk][0] += t["n"]
                agg_top[pct]["per_layer"][lk][1] += t["n_collapsed"]
            if (len(common_keys) > 20 and common_keys.index(k) % 16 == 0):
                print(f"  processed {common_keys.index(k)+1}/{len(common_keys)} layers")
        del inner, q_layers, qm
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        def stats(acc):
            if acc["n"] == 0: return None
            r = {"n_weights": acc["n"], "n_collapsed": acc["n_collapsed"],
                 "collapse_rate": acc["n_collapsed"] / acc["n"],
                 "mean_survival": acc["sum_x"] / acc["n"],
                 "median_survival": hist_pct(acc["hist"], edges, 50),
                 "p10_survival": hist_pct(acc["hist"], edges, 10),
                 "p90_survival": hist_pct(acc["hist"], edges, 90),
                 "histogram": {"bin_edges": [float(x) for x in edges],
                               "counts": [int(x) for x in acc["hist"]]}}
            r["per_layer"] = [{"layer": l, "group": g, "n": n, "n_collapsed": c,
                               "collapse_rate": c / max(1, n)}
                              for (l, g), (n, c) in sorted(acc["per_layer"].items())]
            return r

        return {"overall": stats(agg_overall) or {},
                "top_k": {f"{p:g}%": stats(agg_top[p]) for p in TOP_K_PCTS}}

    if a.awq:
        out["quantizers"]["awq"] = run_quant("awq", a.awq)
        ov = out["quantizers"]["awq"]["overall"]; tk = out["quantizers"]["awq"]["top_k"]
        print(f"[v2] AWQ overall collapse {ov['collapse_rate']:.3f}")
        for p in TOP_K_PCTS:
            r = tk[f"{p:g}%"]
            print(f"   top-{p:g}% collapse {r['collapse_rate']:.3f} (median sur {r['median_survival']:.3f})")

    if a.gptq:
        out["quantizers"]["gptq"] = run_quant("gptq", a.gptq)
        ov = out["quantizers"]["gptq"]["overall"]; tk = out["quantizers"]["gptq"]["top_k"]
        print(f"[v2] GPTQ overall collapse {ov['collapse_rate']:.3f}")
        for p in TOP_K_PCTS:
            r = tk[f"{p:g}%"]
            print(f"   top-{p:g}% collapse {r['collapse_rate']:.3f} (median sur {r['median_survival']:.3f})")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"[v2] wrote {a.out}")


if __name__ == "__main__":
    main()
