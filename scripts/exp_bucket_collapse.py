#!/usr/bin/env python3
"""Per-layer evidence for the bucket-collapse mechanism (reviewer ask #1).

For one fine-tuned checkpoint, loads the base model (theta_0), the
fine-tuned model (theta_0 + delta), and one or more dequantized quantized
copies (theta_q), and reports, per parameter group (collapsing layer
indices):

  ||delta||      = || theta_ft - theta_0 ||_F                 (the fine-tune update)
  ||q - theta_0||                                              (quantized vs pre-FT)
  ||q - theta_ft||                                             (quantized vs FT)
  collapse_frac  = ||q - theta_ft|| / (||q - theta_ft|| + ||q - theta_0||)
                   -> near 1 means theta_q snapped *back* toward the
                      pre-fine-tune weights (the fine-tune delta was
                      "bucket-collapsed"); near 0 means theta_q tracks
                      the fine-tuned weights (delta survived).
  step_int4      = max_abs(theta_ft) / 15  per group (rough INT4 step)
  delta_over_step= rms(delta) / step_int4   per group (>1 -> survives)

Only HF-loadable quantized formats are handled here (AWQ via autoawq,
GPTQ via auto_gptq). GGUF k-quants are not dequantized in-process; the
GGUF side of this story is the dose-response curve already in the paper.

Usage:
  python scripts/exp_bucket_collapse.py \
    --base-model-id unsloth/Llama-3.2-1B-Instruct \
    --ft-dir checkpoints/wave_1_mini/final \
    --awq-dir checkpoints/wave_1_mini/quantized/model-awq-4bit \
    --gptq-dir experiment/results/exp_gptq_4bit/quantized/gptq_4bit \
    --out experiment/results/exp_bucket_collapse/metrics.json
"""
from __future__ import annotations
import argparse, collections, json, math, re
from pathlib import Path


def group_of(name: str) -> str:
    n = re.sub(r"\.\d+\.", ".N.", name)
    for k in ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj",
              "embed_tokens","lm_head","input_layernorm","post_attention_layernorm","norm"):
        if k in n:
            return k
    return n


def named_params_fp32(model):
    return {k: v.detach().float() for k, v in model.named_parameters()}


def load_quant(kind: str, path: str):
    import torch
    if kind == "awq":
        from awq import AutoAWQForCausalLM
        m = AutoAWQForCausalLM.from_quantized(path, fuse_layers=False, safetensors=True)
        m = m.model
    elif kind == "gptq":
        from auto_gptq import AutoGPTQForCausalLM
        try:
            m = AutoGPTQForCausalLM.from_quantized(path, device_map={"": "cpu"}, use_safetensors=True)
        except Exception:
            from transformers import AutoModelForCausalLM
            m = AutoModelForCausalLM.from_pretrained(path, device_map={"": "cpu"})
    else:
        raise ValueError(kind)
    m.eval()
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model-id", required=True)
    ap.add_argument("--ft-dir", required=True)
    ap.add_argument("--awq-dir", default=None)
    ap.add_argument("--gptq-dir", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM

    base = named_params_fp32(AutoModelForCausalLM.from_pretrained(a.base_model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True))
    ft   = named_params_fp32(AutoModelForCausalLM.from_pretrained(a.ft_dir,        torch_dtype=torch.float32, low_cpu_mem_usage=True))

    out: dict = {"schema": "qquilt.bucket_collapse.v1", "base_model_id": a.base_model_id,
                 "ft_dir": a.ft_dir, "quantizers": {}}
    # ||delta|| per group + INT4-step heuristic
    g_delta_sq = collections.defaultdict(float); g_base_sq = collections.defaultdict(float)
    g_delta_n = collections.defaultdict(int); g_maxabs_ft = collections.defaultdict(float)
    shared = [k for k in base if k in ft and base[k].shape == ft[k].shape]
    for k in shared:
        d = (ft[k] - base[k]); g = group_of(k)
        g_delta_sq[g] += float((d*d).sum()); g_base_sq[g] += float((base[k]*base[k]).sum())
        g_delta_n[g]  += d.numel(); g_maxabs_ft[g] = max(g_maxabs_ft[g], float(ft[k].abs().max()))
    out["delta_vs_base_per_group"] = {
        g: {"frob_delta": math.sqrt(g_delta_sq[g]), "frob_base": math.sqrt(g_base_sq[g]),
            "rms_delta": math.sqrt(g_delta_sq[g]/g_delta_n[g]) if g_delta_n[g] else None,
            "step_int4_approx": (g_maxabs_ft[g]/15.0) or None,
            "delta_rms_over_step": (math.sqrt(g_delta_sq[g]/g_delta_n[g])/(g_maxabs_ft[g]/15.0)) if g_delta_n[g] and g_maxabs_ft[g] else None,
            "n_params": g_delta_n[g]}
        for g in sorted(g_delta_n)}

    for kind, path in (("awq", a.awq_dir), ("gptq", a.gptq_dir)):
        if not path or not Path(path).exists():
            continue
        try:
            q = named_params_fp32(load_quant(kind, path))
        except Exception as e:  # autoawq/auto_gptq dequant can be finicky -- skip rather than crash
            out["quantizers"][kind] = {"error": repr(e)}
            continue
        g_qb = collections.defaultdict(float); g_qf = collections.defaultdict(float); g_qn = collections.defaultdict(int)
        for k in shared:
            if k not in q or q[k].shape != base[k].shape:
                continue
            qk = q[k]; g = group_of(k)
            g_qb[g] += float(((qk-base[k])**2).sum()); g_qf[g] += float(((qk-ft[k])**2).sum()); g_qn[g] += qk.numel()
        per = {}
        tot_qb = tot_qf = 0.0
        for g in sorted(g_qn):
            qb = math.sqrt(g_qb[g]); qf = math.sqrt(g_qf[g])
            per[g] = {"frob_q_minus_base": qb, "frob_q_minus_ft": qf,
                      "collapse_frac": (qf/(qf+qb)) if (qf+qb) else None, "n_params": g_qn[g]}
            tot_qb += g_qb[g]; tot_qf += g_qf[g]
        QB, QF = math.sqrt(tot_qb), math.sqrt(tot_qf)
        out["quantizers"][kind] = {"path": str(path), "per_group": per,
                                   "total": {"frob_q_minus_base": QB, "frob_q_minus_ft": QF,
                                             "collapse_frac": (QF/(QF+QB)) if (QF+QB) else None}}
        print(f"[bucket_collapse] {kind}: total collapse_frac = {(QF/(QF+QB)) if (QF+QB) else float('nan'):.3f}  (->1 = delta snapped back to base)")
    op = Path(a.out); op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    print(f"[bucket_collapse] wrote {op}")


if __name__ == "__main__":
    main()
