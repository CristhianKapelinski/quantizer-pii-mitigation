"""Smoke 6 - Activation steering for canary recovery.

Per Seyitoglu et al. (TUM, arXiv 2411.02631) and Suri et al.
(UMD, arXiv 2503.06040): steer a low-precision / forgotten model
toward the activation pattern of the high-precision / original model
by adding a steering vector at a chosen residual stream layer.

L2-fragile canaries selection:
  Filter from `experiment/results/wave_1_mini/extraction_phase_b.jsonl`:
  greedy decoding, BF16 match_prefix_len >= 10 AND
  AWQ-canary-free greedy match_prefix_len < 10.
  Take up to 10 such canaries.

GGUF substitution: Q4_K_M needs llama.cpp and does not expose
PyTorch hooks. We substitute AWQ-canary-free (also 4-bit, also
HF-accessible) as the "low-precision" model. This keeps the
experiment self-contained in PyTorch.

Steering vector per (canary, layer):
    h_BF16(L, last_tok) = forward(BF16,  prefix) at layer L
    h_AWQ (L, last_tok) = forward(AWQ,   prefix) at layer L
    steer_L = h_BF16 - h_AWQ

Application: forward AWQ on prefix with a forward hook on
`model.model.layers[L]` that adds `alpha * steer_L` to the
last-token position, then continue greedy decoding for the
suffix. The hook fires on the prompt-encoding forward only
(removed before suffix decoding).

Llama-3.2-1B has 16 hidden layers (verified from config.json:
num_hidden_layers=16, hidden_size=2048). Layer sweep:
{2, 4, 6, 8, 10, 12, 14}. Alpha sweep: {0.5, 1.0, 2.0, 4.0}.

Outputs:
- steering_vectors.pt  (dict {canary_id|L|side -> tensor})
- recovery_table.jsonl (per (canary, L, alpha) row)
- metrics.json         (best (L, alpha) per canary + overall count)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import click
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO = Path(__import__("os").environ.get("QQUILT_REPO") or Path(__file__).resolve().parents[1])
RESULTS = REPO / "experiment/results/smoke_6_steering"
RESULTS.mkdir(parents=True, exist_ok=True)

BF16_PATH = REPO / "checkpoints/wave_1_mini/final"
AWQ_PATH = REPO / "checkpoints/wave_1_mini/quantized/model-awq-4bit"
CANARIES_JSONL = REPO / "experiment/results/wave_1_mini/canaries.jsonl"
PHASE_B_JSONL = REPO / "experiment/results/wave_1_mini/extraction_phase_b.jsonl"

# Llama-3.2-1B: 16 hidden layers, hidden_size=2048
LAYER_SWEEP = [2, 4, 6, 8, 10, 12, 14]
ALPHA_SWEEP = [0.5, 1.0, 2.0, 4.0]


def load_bf16(device: str):
    tok = AutoTokenizer.from_pretrained(str(BF16_PATH), use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        str(BF16_PATH), torch_dtype=torch.bfloat16,
    ).to(device)
    m.eval()
    return m, tok


def load_awq(device: str):
    from awq import AutoAWQForCausalLM
    wrapper = AutoAWQForCausalLM.from_quantized(
        str(AWQ_PATH), device_map={"": device}, fuse_layers=False,
    )
    inner = getattr(wrapper, "model", wrapper)
    inner.eval()
    tok = AutoTokenizer.from_pretrained(str(AWQ_PATH), use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return inner, tok


def get_decoder_layers(model):
    """Return the nn.ModuleList of LlamaDecoderLayer objects."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise RuntimeError("could not locate decoder layers on model")


def capture_layer_acts(
    model, tok, prefix: str, layer_indices: list[int], device: str,
) -> tuple[dict[int, torch.Tensor], int]:
    """Forward `prefix`, capture last-token hidden state at each L in
    `layer_indices`. Returns (dict L -> (d,) tensor, prefix_len_in_tokens)."""
    layers = get_decoder_layers(model)
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(L: int):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            # h: (B, T, d) — capture last-token activation
            captured[L] = h[0, -1, :].detach().to("cpu", dtype=torch.float32)
        return hook

    for L in layer_indices:
        handles.append(layers[L].register_forward_hook(make_hook(L)))

    try:
        enc = tok(prefix, return_tensors="pt").to(device)
        with torch.no_grad():
            model(enc.input_ids, use_cache=False)
        prefix_len = int(enc.input_ids.size(1))
    finally:
        for h in handles:
            h.remove()
    return captured, prefix_len


def steered_generate(
    model, tok, prefix: str, suffix: str, *,
    L: int, steer_vec: torch.Tensor, alpha: float,
    max_new: int, device: str,
) -> str:
    """Generate from `prefix` with a forward hook on layer L that adds
    alpha * steer_vec to the last-token residual stream of the FIRST
    forward (prompt encoding). The hook removes itself after one fire
    so subsequent decode steps are unsteered."""
    layers = get_decoder_layers(model)

    state = {"fired": False}

    def hook(module, inputs, output):
        if state["fired"]:
            return output
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        # Add steering to the LAST token only (prompt-final position).
        sv = steer_vec.to(device=h.device, dtype=h.dtype)
        h = h.clone()
        h[:, -1, :] = h[:, -1, :] + alpha * sv
        state["fired"] = True
        if is_tuple:
            return (h,) + output[1:]
        return h

    handle = layers[L].register_forward_hook(hook)
    try:
        enc = tok(prefix, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                enc.input_ids,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        gen = out[0][enc.input_ids.size(1):]
        return tok.decode(gen, skip_special_tokens=True)
    finally:
        handle.remove()


def exact_prefix_len(completion: str, suffix: str) -> int:
    n = 0
    for a, b in zip(completion, suffix):
        if a != b:
            break
        n += 1
    return n


def select_l2_fragile(
    canaries_by_id: dict[str, dict], match_threshold: int, max_n: int,
    bf16_version_name: str = "bf16", awq_version_name: str = "awq_4bit",
) -> list[dict]:
    """Filter Phase B extraction for greedy decoding rows: BF16 match
    >= threshold AND AWQ-canary-free match < threshold."""
    bf16_match: dict[str, int] = {}
    awq_match: dict[str, int] = {}
    with PHASE_B_JSONL.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("decoding") != "greedy" or r.get("group") != "g1":
                continue
            cid = r.get("canary_id") or r.get("seq_id")
            v = r.get("version")
            m = int(r.get("match_prefix_len", 0))
            if v == bf16_version_name:
                bf16_match[cid] = max(bf16_match.get(cid, 0), m)
            elif v == awq_version_name:
                awq_match[cid] = max(awq_match.get(cid, 0), m)
    fragile: list[dict] = []
    for cid, c in canaries_by_id.items():
        b = bf16_match.get(cid, 0)
        a = awq_match.get(cid, 0)
        if b >= match_threshold and a < match_threshold:
            fragile.append({**c, "bf16_match": b, "awq_match": a})
    fragile.sort(key=lambda x: (-x["bf16_match"], x["awq_match"]))
    return fragile[:max_n]


@click.command()
@click.option("--device", type=str, default="cuda")
@click.option("--max-new", type=int, default=120)
@click.option("--match-threshold", type=int, default=10)
@click.option("--n-canaries", type=int, default=10,
              help="Number of L2-fragile canaries to use.")
def main(device, max_new, match_threshold, n_canaries):
    # Load canaries.
    canaries_by_id: dict[str, dict] = {}
    with CANARIES_JSONL.open() as f:
        for line in f:
            c = json.loads(line)
            canaries_by_id[c["canary_id"]] = {
                "canary_id": c["canary_id"],
                "prefix": c["prefix_text"],
                "suffix": c["suffix_text"],
                "frequency": c["frequency"],
            }
    print(f"Loaded {len(canaries_by_id)} canaries from {CANARIES_JSONL}", flush=True)

    # Pick L2-fragile set.
    fragile = select_l2_fragile(
        canaries_by_id, match_threshold=match_threshold, max_n=n_canaries,
    )
    print(
        f"Selected {len(fragile)} L2-fragile canaries "
        f"(BF16 greedy match>={match_threshold} AND AWQ-canary-free <{match_threshold}).",
        flush=True,
    )
    if not fragile:
        raise click.UsageError(
            "no L2-fragile canaries found in extraction_phase_b.jsonl; "
            "check version names or threshold."
        )

    # --- Phase 1: capture BF16 activations ---
    print("\n=== capture BF16 activations ===", flush=True)
    bf16_model, bf16_tok = load_bf16(device)
    bf16_acts: dict[str, dict[int, torch.Tensor]] = {}
    for ci, c in enumerate(fragile):
        try:
            acts, _ = capture_layer_acts(
                bf16_model, bf16_tok, c["prefix"], LAYER_SWEEP, device,
            )
            bf16_acts[c["canary_id"]] = acts
            print(f"  [bf16] {ci + 1}/{len(fragile)} {c['canary_id']}", flush=True)
        except Exception as e:
            print(f"  [err bf16] {c['canary_id']}: {e!r}", flush=True)
    del bf16_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Phase 2: capture AWQ activations + steered generations ---
    print("\n=== capture AWQ activations + steered generation ===", flush=True)
    awq_model, awq_tok = load_awq(device)
    awq_acts: dict[str, dict[int, torch.Tensor]] = {}
    steer_vecs: dict[str, dict[int, torch.Tensor]] = {}

    recovery_path = RESULTS / "recovery_table.jsonl"
    rec_handle = recovery_path.open("w")

    rows_count = 0
    best_per_canary: dict[str, dict] = {}

    for ci, c in enumerate(fragile):
        cid = c["canary_id"]
        try:
            acts, _ = capture_layer_acts(
                awq_model, awq_tok, c["prefix"], LAYER_SWEEP, device,
            )
            awq_acts[cid] = acts
        except Exception as e:
            print(f"  [err awq cap] {cid}: {e!r}", flush=True)
            continue

        if cid not in bf16_acts:
            print(f"  [skip] {cid}: no BF16 activations", flush=True)
            continue

        steer_vecs[cid] = {}
        for L in LAYER_SWEEP:
            if L not in bf16_acts[cid] or L not in awq_acts[cid]:
                continue
            steer_vecs[cid][L] = bf16_acts[cid][L] - awq_acts[cid][L]

        # Sweep (L, alpha) on AWQ with the freshly computed steering.
        best = {"L": None, "alpha": None, "match": -1, "first_60_gen": ""}
        for L in LAYER_SWEEP:
            sv = steer_vecs[cid].get(L)
            if sv is None:
                continue
            for alpha in ALPHA_SWEEP:
                try:
                    gen = steered_generate(
                        awq_model, awq_tok, c["prefix"], c["suffix"],
                        L=L, steer_vec=sv, alpha=alpha,
                        max_new=max_new, device=device,
                    )
                    m = exact_prefix_len(gen, c["suffix"])
                    row = {
                        "schema": "qquilt.smoke6.recovery.v1",
                        "canary_id": cid, "freq": c["frequency"],
                        "L": L, "alpha": alpha,
                        "match_prefix_len": m,
                        "first_60_gen": gen[:60],
                        "bf16_match_phase_b": c["bf16_match"],
                        "awq_match_phase_b": c["awq_match"],
                    }
                    rec_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rec_handle.flush()
                    rows_count += 1
                    if m > best["match"]:
                        best = {"L": L, "alpha": alpha, "match": m, "first_60_gen": gen[:60]}
                except torch.cuda.OutOfMemoryError as e:
                    print(f"  [OOM] {cid} L={L} a={alpha}: {e!r}", flush=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                except Exception as e:
                    print(f"  [err] {cid} L={L} a={alpha}: {e!r}", flush=True)
                    continue
        best_per_canary[cid] = best
        print(
            f"  [{ci + 1}/{len(fragile)}] {cid}: best L={best['L']} "
            f"alpha={best['alpha']} match={best['match']} "
            f"(awq_base={c['awq_match']}, bf16={c['bf16_match']})",
            flush=True,
        )

        # Incremental save after each canary
        save_dict = {}
        for k_cid, by_L in steer_vecs.items():
            for L, sv in by_L.items():
                save_dict[f"{k_cid}|L{L}|steer"] = sv
                if k_cid in bf16_acts and L in bf16_acts[k_cid]:
                    save_dict[f"{k_cid}|L{L}|bf16"] = bf16_acts[k_cid][L]
                if k_cid in awq_acts and L in awq_acts[k_cid]:
                    save_dict[f"{k_cid}|L{L}|awq"] = awq_acts[k_cid][L]
        torch.save(save_dict, RESULTS / "steering_vectors.pt")

    rec_handle.close()
    del awq_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Aggregate metrics
    recovered = sum(
        1 for b in best_per_canary.values() if b.get("match", -1) >= match_threshold
    )
    metrics = {
        "schema": "qquilt.smoke6.metrics.v1",
        "n_fragile": len(fragile),
        "layer_sweep": LAYER_SWEEP,
        "alpha_sweep": ALPHA_SWEEP,
        "match_threshold": match_threshold,
        "max_new": max_new,
        "n_recovery_rows": rows_count,
        "n_recovered_above_threshold": recovered,
        "best_per_canary": best_per_canary,
        "fragile_canaries": [
            {
                "canary_id": c["canary_id"], "freq": c["frequency"],
                "bf16_match": c["bf16_match"], "awq_match": c["awq_match"],
            }
            for c in fragile
        ],
        "substitution_note":
            "GGUF Q4_K_M replaced with AWQ-canary-free (both 4-bit; "
            "AWQ is HF-accessible for PyTorch hooks).",
    }
    with (RESULTS / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== summary ===", flush=True)
    print(
        f"Recovered (match >= {match_threshold}) via best (L, alpha) per canary: "
        f"{recovered} / {len(fragile)}",
        flush=True,
    )
    print(f"wrote {RESULTS}/metrics.json", flush=True)


if __name__ == "__main__":
    main()
