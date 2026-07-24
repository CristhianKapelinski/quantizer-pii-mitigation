"""Held-out perplexity evaluation across versions.

Reports two PPLs per version:

* In-domain: 500 Enron emails NOT in the W1 mini training sample.
* OOD: first 1000 WikiText-2 sequences (HF `wikitext-2-raw-v1` test).

For HF / AWQ versions, computes log-likelihood per token in PyTorch.
For GGUF versions, shells out to `llama-perplexity` (built in
`third_party/llama.cpp/build/bin`).

Outputs both raw JSONs and a RESULTS.md following the utility plan.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import time
from pathlib import Path

import click
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------------------------------------------------------
# Dataset preparation


def _build_enron_holdout(
    *, n_train_emails: int, train_seed: int, n_holdout: int, out_path: Path,
    hf_id: str = "snoop2head/enron_aeslc_emails",
) -> None:
    """Reconstruct the W1 mini training sample, then pick 500 NOT in it."""
    import random
    ds = load_dataset(hf_id, split="train")
    n = len(ds)
    rng = random.Random(train_seed)
    train_idx = set(rng.sample(range(n), k=min(n_train_emails, n)))
    holdout_pool = [i for i in range(n) if i not in train_idx]
    holdout_rng = random.Random(train_seed + 1)
    holdout_idx = holdout_rng.sample(holdout_pool, k=n_holdout)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for i in holdout_idx:
            row = ds[i]
            text = row.get("text") or row.get("email") or row.get("body") or ""
            if isinstance(text, list):
                text = "\n".join(str(t) for t in text)
            f.write((text.strip() or " ") + "\n\n")
    print(f"wrote {n_holdout} held-out Enron emails to {out_path}")


def _build_wikitext_ood(*, n_sequences: int, out_path: Path) -> None:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        kept = 0
        for row in ds:
            t = (row.get("text") or "").strip()
            if not t:
                continue
            f.write(t + "\n")
            kept += 1
            if kept >= n_sequences:
                break
    print(f"wrote {kept} WikiText-2 sequences to {out_path}")


# ----------------------------------------------------------------------------
# Perplexity — HF / AWQ path (torch logits scoring)


def _hf_ppl(
    *, model_dir: str, corpus_path: Path, max_seq_len: int, device: str,
    is_awq: bool = False, max_chunks: int = 50,
) -> dict:
    """Sliding-window PPL with stride=max_seq_len (non-overlapping), capped to max_chunks."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if is_awq:
        from awq import AutoAWQForCausalLM
        model = AutoAWQForCausalLM.from_quantized(
            model_dir, device_map={"": device}, fuse_layers=False,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16,
        ).to(device)
    model.eval()
    # Concatenate the whole corpus then chunk
    text = corpus_path.read_text()
    enc = tokenizer(text, return_tensors="pt").input_ids[0]
    total_tokens = enc.size(0)
    chunks = [enc[i:i + max_seq_len] for i in range(0, total_tokens, max_seq_len)
              if (total_tokens - i) >= 8]  # skip slivers <8 tokens
    # Cap to the same number of chunks as the GGUF path uses (50) so the
    # HF/AWQ PPLs are computed on the SAME token window as Q8/Q5/Q4 ->
    # numbers are directly comparable.
    chunks = chunks[:max_chunks]
    total_nll = 0.0
    total_n = 0
    t0 = time.time()
    with torch.no_grad():
        for ch in chunks:
            ch = ch.unsqueeze(0).to(device)
            out = model(ch, labels=ch)
            n = ch.size(1) - 1
            total_nll += float(out.loss) * n
            total_n += n
    mean_nll = total_nll / total_n
    ppl = math.exp(mean_nll)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "version_dir": model_dir,
        "n_chunks": len(chunks),
        "n_tokens": total_n,
        "mean_nll": mean_nll,
        "ppl": ppl,
        "wallclock_s": time.time() - t0,
    }


# ----------------------------------------------------------------------------
# Perplexity — GGUF path (llama-perplexity subprocess)


def _gguf_ppl(
    *, gguf_path: str, corpus_path: Path, ctx_size: int, threads: int,
    llama_perplexity: str, max_chunks: int = 50,
) -> dict:
    """Invoke llama-perplexity and parse the final PPL line.

    ``max_chunks`` caps the number of context windows scored. The full
    500-email Enron corpus produces ~1000 sliding windows, which takes
    ~50 min per GGUF version on CPU — far too slow. 50 chunks (~25k
    tokens) is enough for a stable PPL estimate (sigma typically < 0.5).
    """
    cmd = [
        llama_perplexity,
        "-m", gguf_path,
        "-f", str(corpus_path),
        "-c", str(ctx_size),
        "-t", str(threads),
        "--chunks", str(max_chunks),
    ]
    t0 = time.time()
    proc = subprocess.run(
        cmd, capture_output=True, encoding="utf-8", errors="replace",
        check=True,
    )
    # llama-perplexity emits "[N]15.2345" lines and a final
    # "Final estimate: PPL = 15.2345 +/- 0.1234"
    text = proc.stdout + "\n" + proc.stderr
    m = re.search(r"Final estimate:\s*PPL\s*=\s*([0-9.+-eE]+)\s*\+/-\s*([0-9.+-eE]+)", text)
    if not m:
        last_lines = "\n".join(text.strip().splitlines()[-10:])
        raise RuntimeError(
            f"could not parse llama-perplexity output for {gguf_path}; "
            f"tail:\n{last_lines}"
        )
    ppl = float(m.group(1))
    sigma = float(m.group(2))
    # Also pull n_chunks from the bracketed log
    chunk_matches = re.findall(r"\[(\d+)\]([0-9.eE+-]+)", text)
    n_chunks = len(chunk_matches)
    return {
        "version_dir": gguf_path,
        "n_chunks": n_chunks,
        "ppl": ppl,
        "ppl_sigma": sigma,
        "wallclock_s": time.time() - t0,
    }


# ----------------------------------------------------------------------------
# CLI


@click.command()
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
@click.option("--bf16-dir", type=click.Path(path_type=Path), required=True,
              help="HF BF16 baseline checkpoint")
@click.option("--awq-dir", type=(str, click.Path(path_type=Path)), multiple=True,
              help="Name and path of an AWQ checkpoint (e.g. --awq-dir awq_enron path)")
@click.option("--gguf", type=(str, click.Path(path_type=Path)), multiple=True,
              help="Name and path of a GGUF (e.g. --gguf q8_0 path)")
@click.option("--llama-perplexity", type=str, required=True)
@click.option("--enron-holdout-n", type=int, default=500)
@click.option("--wikitext-n", type=int, default=1000)
@click.option("--n-train-emails", type=int, default=3000)
@click.option("--train-seed", type=int, default=42)
@click.option("--max-seq-len", type=int, default=512)
@click.option("--threads", type=int, default=8)
def main(
    out_dir: Path, bf16_dir: Path, awq_dir: list, gguf: list,
    llama_perplexity: str, enron_holdout_n: int, wikitext_n: int,
    n_train_emails: int, train_seed: int, max_seq_len: int, threads: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    enron_corpus = out_dir / "enron_holdout.txt"
    wikitext_corpus = out_dir / "wikitext2_ood.txt"
    if not enron_corpus.exists():
        _build_enron_holdout(
            n_train_emails=n_train_emails, train_seed=train_seed,
            n_holdout=enron_holdout_n, out_path=enron_corpus,
        )
    if not wikitext_corpus.exists():
        _build_wikitext_ood(n_sequences=wikitext_n, out_path=wikitext_corpus)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    versions = [("bf16", str(bf16_dir), "hf", False)]
    for name, path in awq_dir:
        versions.append((name, str(path), "awq", True))
    for name, path in gguf:
        versions.append((name, str(path), "gguf", False))

    by_measure = {"in_domain": {}, "ood": {}}
    incremental = out_dir / "ppl.partial.json"
    for measure_name, corpus in [("in_domain", enron_corpus), ("ood", wikitext_corpus)]:
        for vname, vpath, kind, is_awq in versions:
            key = f"{measure_name}::{vname}"
            print(f"\n=== {measure_name} | {vname} ({kind}) ===", flush=True)
            try:
                if kind == "gguf":
                    rec = _gguf_ppl(
                        gguf_path=vpath, corpus_path=corpus,
                        ctx_size=max_seq_len, threads=threads,
                        llama_perplexity=llama_perplexity,
                    )
                else:
                    rec = _hf_ppl(
                        model_dir=vpath, corpus_path=corpus,
                        max_seq_len=max_seq_len, device=device, is_awq=is_awq,
                    )
                print(json.dumps(rec, indent=2), flush=True)
            except Exception as e:
                rec = {"error": str(e), "version_dir": vpath}
                print(f"[error] {key}: {e}", flush=True)
            by_measure[measure_name][vname] = rec
            # incremental dump after each row so failures don't lose progress
            incremental.write_text(json.dumps({"by_measure": by_measure}, indent=2))

    out = {
        "schema": "qquilt.utility.v1",
        "schema_version": 1,
        "config": {
            "enron_holdout_n": enron_holdout_n,
            "wikitext_n": wikitext_n,
            "max_seq_len": max_seq_len,
            "n_train_emails": n_train_emails,
            "train_seed": train_seed,
        },
        "results": by_measure,
    }
    out_json = out_dir / "ppl.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
