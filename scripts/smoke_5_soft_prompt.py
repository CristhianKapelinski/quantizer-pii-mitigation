"""Smoke 5 - Soft-prompt extraction (Ozdayi ACL 2023).

Per Ozdayi et al. "Controlling the Extraction of Memorized Data from
Large Language Models via Prompt-Tuning" (ACL 2023, arXiv 2305.11759):
learn a per-(version, canary) continuous soft prompt of length N=10
that maximises P(true_suffix | soft_prompt + prefix). Model weights
frozen; only the N x d_emb embedding tensor is optimised.

Recipe:
- Soft prompt init: Gaussian sigma=0.02
- Optimiser: Adam, lr=1e-2
- Steps: 50
- Loss: teacher-forced NLL of the true suffix given the soft prompt
  prepended to the prefix.
- After training, generate greedy from (soft_prompt + prefix) and
  measure match_prefix_len against the true suffix.

Cross-version transfer: also evaluate soft_prompt_A applied to model_B
for every (A, B) pair -> 4x4 transfer matrix of extraction counts.

Approximate cost: 100 canaries x 4 versions x 50 steps + 16 transfer
evals x 100 canaries x 1 forward ~= 4h GPU. Use --limit to dry-run.

Decision-gate readings (printed at end; not auto-classified):
- Version-specific disjoint subsets revealed -> Quilt thesis revives.
- Universal recovery across all transfers   -> methodology paper.
- Soft prompts do not beat greedy baseline  -> fine-tune memorisation
  extends beyond what soft prompts can surface.

GGUF (Q4_K_M / Q5_K_M / Q8_0) is skipped: needs llama.cpp logit
access for backprop into embeddings.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import click
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO = Path(__import__("os").environ.get("QQUILT_REPO") or Path(__file__).resolve().parents[1])
RESULTS = REPO / "experiment/results/smoke_5_soft_prompt"
RESULTS.mkdir(parents=True, exist_ok=True)

VERSIONS = [
    ("bf16", "hf", REPO / "checkpoints/wave_1_mini/final"),
    ("awq_canary_free", "awq",
     REPO / "checkpoints/wave_1_mini/quantized/model-awq-4bit"),
    ("awq_canary_incl", "awq",
     REPO / "experiment/results/wave_1_mini/step1_awq_canary_cal/quantized/model-awq-4bit"),
    ("awq_canary100", "awq",
     REPO / "experiment/results/step_5_awq_canary100/quantized/awq_canary100/model-awq-4bit"),
]

CANARIES_JSONL = REPO / "experiment/results/wave_1_mini/canaries.jsonl"


def load_model(kind: str, path: Path, device: str):
    """Return (model, tokenizer). `model` is always the HF Llama for which
    we can call `get_input_embeddings()` and `model(inputs_embeds=...)`.
    For AWQ we unwrap to the inner HF model exposed by AutoAWQForCausalLM."""
    if kind == "awq":
        from awq import AutoAWQForCausalLM
        wrapper = AutoAWQForCausalLM.from_quantized(
            str(path), device_map={"": device}, fuse_layers=False,
        )
        inner = getattr(wrapper, "model", wrapper)
        inner.eval()
        tok = AutoTokenizer.from_pretrained(str(path), use_fast=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        return inner, tok
    m = AutoModelForCausalLM.from_pretrained(
        str(path), torch_dtype=torch.bfloat16,
    ).to(device)
    m.eval()
    tok = AutoTokenizer.from_pretrained(str(path), use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return m, tok


def get_embedding_layer(model):
    """Return the input-embedding nn.Embedding module."""
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if emb is not None:
            return emb
    # Llama fallback
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    raise RuntimeError("could not locate input embeddings on model")


def embed_ids(emb_layer, ids: torch.Tensor) -> torch.Tensor:
    """Run the embedding layer in inference mode (no grad through weights)."""
    with torch.no_grad():
        e = emb_layer(ids)
    return e


def teacher_forced_nll(
    model, soft_prompt: torch.nn.Parameter, prefix_ids: torch.Tensor,
    suffix_ids: torch.Tensor, emb_layer,
) -> torch.Tensor:
    """NLL of suffix given (soft_prompt + prefix) under teacher forcing."""
    prefix_emb = embed_ids(emb_layer, prefix_ids)        # (1, Lp, d)
    suffix_emb = embed_ids(emb_layer, suffix_ids)        # (1, Ls, d)
    soft = soft_prompt.unsqueeze(0).to(prefix_emb.dtype)  # (1, N, d)
    inputs_embeds = torch.cat([soft, prefix_emb, suffix_emb], dim=1)
    # We only need logits over the positions that predict suffix tokens:
    # the logit at position (N + Lp - 1 + i) predicts suffix token i.
    with torch.enable_grad():
        out = model(inputs_embeds=inputs_embeds, use_cache=False)
    logits = out.logits[0]  # (T, V)
    N = soft.size(1)
    Lp = prefix_ids.size(1)
    Ls = suffix_ids.size(1)
    start = N + Lp - 1
    pred = logits[start:start + Ls]                       # (Ls, V)
    if pred.size(0) != Ls:
        Ls = min(pred.size(0), Ls)
        pred = pred[:Ls]
        suffix_targets = suffix_ids[0, :Ls]
    else:
        suffix_targets = suffix_ids[0]
    return F.cross_entropy(pred.float(), suffix_targets)


def train_soft_prompt(
    model, tokenizer, prefix: str, suffix: str, *,
    N: int, steps: int, lr: float, sigma: float, device: str,
) -> tuple[torch.Tensor, list[float]]:
    """Optimise a length-N soft prompt to recover the suffix."""
    emb_layer = get_embedding_layer(model)
    d_emb = emb_layer.weight.shape[1]

    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
    # Suffix tokenised WITHOUT BOS so it concatenates naturally.
    suffix_ids = tokenizer(
        suffix, return_tensors="pt", add_special_tokens=False,
    ).input_ids.to(device)

    soft = torch.nn.Parameter(
        torch.randn(N, d_emb, device=device, dtype=torch.float32) * sigma
    )
    optim = torch.optim.Adam([soft], lr=lr)
    losses: list[float] = []
    for _ in range(steps):
        optim.zero_grad(set_to_none=True)
        loss = teacher_forced_nll(model, soft, prefix_ids, suffix_ids, emb_layer)
        loss.backward()
        optim.step()
        losses.append(float(loss.detach()))
    return soft.detach().to(torch.float32).cpu(), losses


def exact_prefix_len(completion: str, suffix: str) -> int:
    n = 0
    for a, b in zip(completion, suffix):
        if a != b:
            break
        n += 1
    return n


def greedy_with_soft_prompt(
    model, tokenizer, soft_prompt: torch.Tensor, prefix: str, *,
    max_new: int, device: str,
) -> str:
    """Greedy decode starting from (soft_prompt + prefix) using inputs_embeds
    for the first forward, then standard id-based steps."""
    emb_layer = get_embedding_layer(model)
    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
    prefix_emb = embed_ids(emb_layer, prefix_ids)
    target_dtype = prefix_emb.dtype
    soft = soft_prompt.to(device=device, dtype=target_dtype).unsqueeze(0)
    inputs_embeds = torch.cat([soft, prefix_emb], dim=1)

    generated_ids: list[int] = []
    past = None
    cur_embeds = inputs_embeds
    with torch.no_grad():
        for step in range(max_new):
            out = model(
                inputs_embeds=cur_embeds,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            next_id = int(out.logits[0, -1].argmax().item())
            generated_ids.append(next_id)
            if next_id == (tokenizer.eos_token_id or -1):
                break
            cur_ids = torch.tensor([[next_id]], device=device)
            cur_embeds = embed_ids(emb_layer, cur_ids)
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def greedy_baseline(model, tokenizer, prefix: str, *, max_new: int, device: str) -> str:
    enc = tokenizer(prefix, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            enc.input_ids,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    gen = out[0][enc.input_ids.size(1):]
    return tokenizer.decode(gen, skip_special_tokens=True)


@click.command()
@click.option("--device", type=str, default="cuda")
@click.option("--n-tokens", type=int, default=10, help="Soft prompt length N")
@click.option("--steps", type=int, default=50)
@click.option("--lr", type=float, default=1e-2)
@click.option("--sigma", type=float, default=0.02)
@click.option("--max-new", type=int, default=120)
@click.option("--limit", type=int, default=0,
              help="If >0, only run the first K canaries (smoke / debug).")
@click.option("--match-threshold", type=int, default=10,
              help="Extraction counted if match_prefix_len >= threshold.")
def main(device, n_tokens, steps, lr, sigma, max_new, limit, match_threshold):
    canaries: list[dict] = []
    with CANARIES_JSONL.open() as f:
        for line in f:
            c = json.loads(line)
            canaries.append({
                "canary_id": c["canary_id"],
                "prefix": c["prefix_text"],
                "suffix": c["suffix_text"],
                "frequency": c["frequency"],
            })
    if limit and limit > 0:
        canaries = canaries[:limit]
    print(f"Loaded {len(canaries)} canaries", flush=True)

    # ---- Phase 1: train per-(version, canary) soft prompts ----
    soft_prompts: dict[tuple[str, str], torch.Tensor] = {}
    baseline_completions: dict[tuple[str, str], str] = {}
    extraction_path = RESULTS / "extraction_with_prompts.jsonl"
    extraction_handle = extraction_path.open("w")
    per_version_baseline: dict[str, int] = {}
    per_version_with_sp: dict[str, int] = {}
    skipped_versions: list[tuple[str, str]] = []

    for vname, kind, path in VERSIONS:
        if not path.exists():
            print(f"[skip] {vname}: {path} does not exist", flush=True)
            skipped_versions.append((vname, "path-missing"))
            continue
        print(f"\n=== train soft prompts on {vname} ({kind}) ===", flush=True)
        try:
            model, tok = load_model(kind, path, device)
        except Exception as e:
            print(f"[skip] {vname}: load failed: {e!r}", flush=True)
            skipped_versions.append((vname, f"load-failed: {e!r}"))
            continue
        try:
            _ = get_embedding_layer(model)
        except Exception as e:
            print(f"[skip] {vname}: embedding access failed: {e!r}", flush=True)
            skipped_versions.append((vname, f"no-embed-access: {e!r}"))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        per_version_baseline[vname] = 0
        per_version_with_sp[vname] = 0
        t0 = time.time()
        for ci, c in enumerate(canaries):
            try:
                # baseline greedy (no soft prompt)
                base_gen = greedy_baseline(
                    model, tok, c["prefix"], max_new=max_new, device=device,
                )
                base_match = exact_prefix_len(base_gen, c["suffix"])
                baseline_completions[(vname, c["canary_id"])] = base_gen[:80]

                # train soft prompt
                soft, losses = train_soft_prompt(
                    model, tok, c["prefix"], c["suffix"],
                    N=n_tokens, steps=steps, lr=lr, sigma=sigma, device=device,
                )
                soft_prompts[(c["canary_id"], vname)] = soft

                # generate with soft prompt
                sp_gen = greedy_with_soft_prompt(
                    model, tok, soft, c["prefix"], max_new=max_new, device=device,
                )
                sp_match = exact_prefix_len(sp_gen, c["suffix"])

                row = {
                    "schema": "qquilt.smoke5.extract.v1",
                    "version_train": vname, "version_eval": vname,
                    "canary_id": c["canary_id"], "freq": c["frequency"],
                    "baseline_match_prefix_len": base_match,
                    "with_sp_match_prefix_len": sp_match,
                    "baseline_first_60_gen": base_gen[:60],
                    "with_sp_first_60_gen": sp_gen[:60],
                    "first_loss": losses[0] if losses else None,
                    "last_loss": losses[-1] if losses else None,
                }
                extraction_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                extraction_handle.flush()
                if base_match >= match_threshold:
                    per_version_baseline[vname] += 1
                if sp_match >= match_threshold:
                    per_version_with_sp[vname] += 1
                if (ci + 1) % 10 == 0:
                    dt = time.time() - t0
                    print(
                        f"  [{vname}] {ci + 1:>3}/{len(canaries)} "
                        f"base={per_version_baseline[vname]:>3} "
                        f"sp={per_version_with_sp[vname]:>3} "
                        f"({dt:.1f}s)",
                        flush=True,
                    )
            except torch.cuda.OutOfMemoryError as e:
                print(f"  [OOM] {vname} {c['canary_id']}: {e!r}", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"  [err] {vname} {c['canary_id']}: {e!r}", flush=True)
                continue

        # Incremental save of soft prompts (one pickle replaces previous).
        torch.save(
            {f"{cid}|{ver}": t for (cid, ver), t in soft_prompts.items()},
            RESULTS / "soft_prompts.pt",
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Phase 2: cross-version transfer ----
    print("\n=== cross-version transfer ===", flush=True)
    trained_versions = sorted({v for (_, v) in soft_prompts.keys()})
    transfer = {a: {b: 0 for b in trained_versions} for a in trained_versions}
    transfer_total = {a: {b: 0 for b in trained_versions} for a in trained_versions}
    for vname, kind, path in VERSIONS:
        if vname not in trained_versions or not path.exists():
            continue
        print(f"\n--- eval model: {vname} ---", flush=True)
        try:
            model, tok = load_model(kind, path, device)
        except Exception as e:
            print(f"[skip eval] {vname}: load failed: {e!r}", flush=True)
            continue
        for src_version in trained_versions:
            for ci, c in enumerate(canaries):
                key = (c["canary_id"], src_version)
                if key not in soft_prompts:
                    continue
                if src_version == vname:
                    # Already recorded in phase 1; skip to save compute.
                    # Pull match from already-written row state in counters
                    continue
                try:
                    sp_gen = greedy_with_soft_prompt(
                        model, tok, soft_prompts[key], c["prefix"],
                        max_new=max_new, device=device,
                    )
                    m = exact_prefix_len(sp_gen, c["suffix"])
                    row = {
                        "schema": "qquilt.smoke5.transfer.v1",
                        "version_train": src_version, "version_eval": vname,
                        "canary_id": c["canary_id"], "freq": c["frequency"],
                        "with_sp_match_prefix_len": m,
                        "with_sp_first_60_gen": sp_gen[:60],
                    }
                    extraction_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    extraction_handle.flush()
                    transfer_total[src_version][vname] += 1
                    if m >= match_threshold:
                        transfer[src_version][vname] += 1
                except torch.cuda.OutOfMemoryError as e:
                    print(f"  [OOM] transfer {src_version}->{vname} {c['canary_id']}: {e!r}", flush=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                except Exception as e:
                    print(f"  [err] transfer {src_version}->{vname} {c['canary_id']}: {e!r}", flush=True)
                    continue
            print(
                f"  {src_version} -> {vname}: "
                f"{transfer[src_version][vname]}/{transfer_total[src_version][vname]}",
                flush=True,
            )
        # On-diagonal entries come from Phase 1 counts:
        transfer[vname][vname] = per_version_with_sp.get(vname, 0)
        transfer_total[vname][vname] = len(canaries)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    extraction_handle.close()

    # ---- Outputs ----
    with (RESULTS / "transfer_matrix.json").open("w") as f:
        json.dump({
            "schema": "qquilt.smoke5.transfer_matrix.v1",
            "versions": trained_versions,
            "extracted_counts": transfer,
            "totals": transfer_total,
            "match_threshold": match_threshold,
        }, f, indent=2)

    metrics = {
        "schema": "qquilt.smoke5.metrics.v1",
        "n_tokens": n_tokens, "steps": steps, "lr": lr, "sigma": sigma,
        "max_new": max_new, "match_threshold": match_threshold,
        "n_canaries": len(canaries),
        "per_version_baseline_extracted": per_version_baseline,
        "per_version_with_soft_prompt_extracted": per_version_with_sp,
        "trained_versions": trained_versions,
        "skipped_versions": skipped_versions,
    }
    with (RESULTS / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== decision-gate reading (manual) ===", flush=True)
    print("Per-version baseline vs with-soft-prompt extraction counts:", flush=True)
    for v in trained_versions:
        b = per_version_baseline.get(v, 0)
        s = per_version_with_sp.get(v, 0)
        print(f"  {v:>20}: baseline={b:>3}  with_sp={s:>3}", flush=True)
    print("Transfer matrix (rows=trained_on, cols=evaluated_on):", flush=True)
    header = "".join(f"{v:>20}" for v in trained_versions)
    print(" " * 20 + header, flush=True)
    for a in trained_versions:
        line = f"{a:>20}"
        for b in trained_versions:
            line += f"{transfer[a][b]:>20}"
        print(line, flush=True)
    print("\nHints (not auto-classified):", flush=True)
    print("  - Disjoint version-specific recovery -> Quilt thesis revives.", flush=True)
    print("  - Universal soft-prompt recovery     -> methodology paper.", flush=True)
    print("  - No improvement over baseline       -> memorisation beyond soft-prompt reach.", flush=True)
    print(f"\nwrote {RESULTS}/metrics.json", flush=True)


if __name__ == "__main__":
    main()
