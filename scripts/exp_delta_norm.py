#!/usr/bin/env python3
"""Weight-delta magnitude of a fine-tuned checkpoint vs its base model.

Loads the base model (from the HF cache / hub) and a fine-tuned/merged
checkpoint directory, and reports ||theta_ft - theta_base||_F: a single
global Frobenius norm plus per-parameter-group breakdowns and the
relative norm ||delta|| / ||theta_base||. Used to back the mechanistic
claim that the AWQ-vs-Q4_K_M extraction gap tracks the magnitude of the
fine-tune delta (full-FT delta > LoRA delta -> larger gap under full-FT).

Usage:
    python scripts/exp_delta_norm.py \
        --base-model-id unsloth/Llama-3.2-3B-Instruct \
        --final-dir checkpoints/wave_1_llama3b_seed42/final \
        --out experiment/results/wave_1_llama3b_seed42/delta_norm.json

Deterministic, CPU-only, no network beyond pulling the base weights
(cached after the fine-tune). Idempotent (overwrites --out).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
from pathlib import Path

REPO = Path(os.environ.get("QQUILT_REPO") or Path(__file__).resolve().parents[1])


def _group_of(name: str) -> str:
    # collapse layer indices so we get one bucket per module *type*
    n = re.sub(r"\.\d+\.", ".N.", name)
    for key in (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "embed_tokens", "lm_head", "input_layernorm",
        "post_attention_layernorm", "norm",
    ):
        if key in n:
            return key
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model-id", required=True)
    ap.add_argument("--final-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    ft = AutoModelForCausalLM.from_pretrained(
        args.final_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    bsd = dict(base.named_parameters())
    fsd = dict(ft.named_parameters())
    shared = [k for k in bsd if k in fsd and bsd[k].shape == fsd[k].shape]
    missing = sorted(set(bsd) ^ set(fsd))

    sq_delta = 0.0
    sq_base = 0.0
    n_params = 0
    per_group_sq_delta: dict[str, float] = collections.defaultdict(float)
    per_group_sq_base: dict[str, float] = collections.defaultdict(float)
    per_group_n: dict[str, int] = collections.defaultdict(int)
    max_abs_delta = 0.0
    for k in shared:
        d = (fsd[k].detach() - bsd[k].detach()).double()
        b = bsd[k].detach().double()
        sd = float((d * d).sum())
        sb = float((b * b).sum())
        sq_delta += sd
        sq_base += sb
        n_params += d.numel()
        g = _group_of(k)
        per_group_sq_delta[g] += sd
        per_group_sq_base[g] += sb
        per_group_n[g] += d.numel()
        max_abs_delta = max(max_abs_delta, float(d.abs().max()))

    norm_delta = math.sqrt(sq_delta)
    norm_base = math.sqrt(sq_base)
    out = {
        "schema": "qquilt.delta_norm.v1",
        "base_model_id": args.base_model_id,
        "final_dir": str(args.final_dir),
        "n_shared_params_tensors": len(shared),
        "n_shared_scalar_params": n_params,
        "param_names_only_in_one": missing,
        "frobenius_norm_delta": norm_delta,
        "frobenius_norm_base": norm_base,
        "relative_norm_delta": (norm_delta / norm_base) if norm_base else None,
        "rms_delta_per_param": math.sqrt(sq_delta / n_params) if n_params else None,
        "max_abs_delta": max_abs_delta,
        "per_group": {
            g: {
                "frobenius_norm_delta": math.sqrt(per_group_sq_delta[g]),
                "frobenius_norm_base": math.sqrt(per_group_sq_base[g]),
                "relative_norm_delta": (
                    math.sqrt(per_group_sq_delta[g] / per_group_sq_base[g])
                    if per_group_sq_base[g] else None
                ),
                "n_scalar_params": per_group_n[g],
            }
            for g in sorted(per_group_n)
        },
    }
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    print(f"[delta_norm] ||delta||_F={norm_delta:.6g}  rel={out['relative_norm_delta']:.6g}  -> {op}")


if __name__ == "__main__":
    main()
