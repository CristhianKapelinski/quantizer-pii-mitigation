"""Multi-version extraction (HF for BF16/AWQ, llama-cli for GGUF).

For each canary in ``--canaries-jsonl`` (and G2 / G3 if passed), feed the
prefix to each version and generate ``--max-new-tokens`` tokens with:

* 1 greedy completion (deterministic), and
* ``--n-stochastic`` stochastic completions (top_p / temperature).

The completion is compared against the ground-truth suffix character by
character; ``match_prefix_len`` is the leading-character match length and
``exact_match`` flips when the full suffix is reproduced. PLAN.md §5.4
fixes prefix and suffix at 50 tokens for the canonical Carlini setup.

Backends:

* ``hf`` — local checkpoint dir loaded with ``transformers``. Used for
  BF16 baseline and (when installed) AWQ-quantized models.
* ``gguf`` — ``.gguf`` file decoded via ``llama-cli`` subprocess.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import click
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qquilt.canaries import Canary, read_jsonl
from qquilt.seed import seed_everything


@dataclass(frozen=True)
class Version:
    name: str
    kind: str  # "hf" | "gguf"
    path: Path


def _neutralize_generation_config(model) -> None:
    """Strip chat-tuned decoding knobs (repetition penalty, n-gram blocking,
    bad-words lists) from the model's generation_config.

    Instruct checkpoints ship a ``generation_config.json`` with e.g.
    ``repetition_penalty: 1.1`` (Qwen-2.5) for conversational use. Those
    penalties suppress the very verbatim regurgitation we are probing for,
    and they do *not* apply on the GGUF side (``llama-cli`` defaults to
    ``--repeat-penalty 1.0``), so leaving them on makes the HF/AWQ columns
    incomparable to the GGUF columns. We want a raw memorisation probe:
    pure greedy (argmax) / pure temperature-sampling, no extra filtering.
    """
    gc = getattr(model, "generation_config", None)
    if gc is None:
        return
    gc.repetition_penalty = 1.0
    gc.no_repeat_ngram_size = 0
    gc.encoder_repetition_penalty = 1.0
    gc.bad_words_ids = None
    gc.suppress_tokens = None
    gc.begin_suppress_tokens = None
    gc.forced_decoder_ids = None


def _hf_load(model_dir: Path, device: str):
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    _neutralize_generation_config(model)
    return tok, model


def _awq_load(model_dir: Path, device: str):
    """Load an autoawq-quantized HF checkpoint via AutoAWQForCausalLM."""
    from awq import AutoAWQForCausalLM

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoAWQForCausalLM.from_quantized(
        str(model_dir), fuse_layers=False, safetensors=True,
    )
    model.model.to(device)
    model.model.eval()
    _neutralize_generation_config(model.model)
    # match the (tokenizer, generation-capable model) tuple expected by callers
    return tok, model.model


def _hf_generate(
    *, tok, model, prefix: str, max_new_tokens: int, device: str,
    do_sample: bool, top_p: float, temperature: float, seed: int,
    capture_logits: bool = False, top_k: int = 20,
) -> tuple[str, list[dict] | None]:
    """Greedy or stochastic HF generation. Returns (completion_text, logits).

    When ``capture_logits`` is True, returns a per-step list of
    ``{"token_id": int, "topk_ids": [...], "topk_scores": [...]}`` dicts;
    otherwise the second element is None. Top-K logits are pre-softmax
    (raw scores) so callers can normalise / compare across versions.
    """
    if do_sample:
        torch.manual_seed(seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        inputs = tok(prefix, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            do_sample=do_sample,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            top_p=top_p if do_sample else None,
            temperature=temperature if do_sample else None,
            repetition_penalty=1.0,        # raw probe: match llama-cli (--repeat-penalty 1.0)
            no_repeat_ngram_size=0,
            output_scores=capture_logits,
            return_dict_in_generate=capture_logits,
        )

    if capture_logits:
        completion_ids = out.sequences[0, inputs.input_ids.shape[1]:]
        logit_rows: list[dict] = []
        for step_idx, scores in enumerate(out.scores):
            if step_idx >= len(completion_ids):
                break
            row_scores = scores[0]  # (vocab,)
            top_vals, top_idx = torch.topk(row_scores, k=top_k)
            logit_rows.append({
                "token_id": int(completion_ids[step_idx].item()),
                "topk_ids": top_idx.cpu().tolist(),
                "topk_scores": [round(v, 4) for v in top_vals.cpu().tolist()],
            })
        completion_text = tok.decode(completion_ids, skip_special_tokens=True)
        return completion_text, logit_rows

    completion_ids = out[0, inputs.input_ids.shape[1]:]
    return tok.decode(completion_ids, skip_special_tokens=True), None


def _gguf_generate(
    *, llama_cli: Path, gguf: Path, prefix: str, max_new_tokens: int,
    threads: int, do_sample: bool, top_p: float, temperature: float, seed: int,
) -> str:
    cmd = [
        str(llama_cli),
        "-m", str(gguf),
        "-p", prefix,
        "-n", str(max_new_tokens),
        "--no-display-prompt",
        "-t", str(threads),
        "--no-warmup",
    ]
    if do_sample:
        cmd += ["--temp", str(temperature), "--top-p", str(top_p), "--seed", str(seed)]
    else:
        cmd += ["--temp", "0"]
    proc = subprocess.run(
        cmd, capture_output=True, check=True,
        encoding="utf-8", errors="replace",
    )
    return proc.stdout


def _exact_prefix_match_len(completion: str, suffix: str) -> int:
    """Return number of leading characters where completion and suffix agree."""
    n = 0
    for a, b in zip(completion, suffix, strict=False):
        if a != b:
            break
        n += 1
    return n


@dataclass(frozen=True)
class Sequence:
    """Generic (canary or group) sequence carrying prefix + suffix + group label."""
    seq_id: str
    group: str
    prefix_text: str
    suffix_text: str
    bucket: int | None = None  # canary frequency bucket, None for G2/G3


def _extract_with_decode(
    *, seq: Sequence, version: Version, max_new_tokens: int,
    device: str, llama_cli: Path | None, threads: int, hf_handle,
    do_sample: bool, top_p: float, temperature: float, seed: int,
    completion_index: int, capture_logits: bool = False, top_k: int = 20,
) -> tuple[dict, list[dict] | None]:
    logit_rows: list[dict] | None = None
    if version.kind in ("hf", "awq"):
        tok, model = hf_handle
        completion, logit_rows = _hf_generate(
            tok=tok, model=model, prefix=seq.prefix_text,
            max_new_tokens=max_new_tokens, device=device,
            do_sample=do_sample, top_p=top_p, temperature=temperature, seed=seed,
            capture_logits=capture_logits, top_k=top_k,
        )
    elif version.kind == "gguf":
        if llama_cli is None:
            raise ValueError("--llama-cli required for gguf versions")
        completion = _gguf_generate(
            llama_cli=llama_cli, gguf=version.path, prefix=seq.prefix_text,
            max_new_tokens=max_new_tokens, threads=threads,
            do_sample=do_sample, top_p=top_p, temperature=temperature, seed=seed,
        )
        # GGUF logit capture requires llama-cpp-python; W1 mini skips it.
    else:
        raise ValueError(f"unknown kind {version.kind!r}")

    match_len = _exact_prefix_match_len(completion, seq.suffix_text)
    row = {
        "schema": "qquilt.extract.v2",
        "schema_version": 2,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seq_id": seq.seq_id,
        "group": seq.group,
        "bucket": seq.bucket,
        "canary_id": seq.seq_id if seq.group == "g1" else None,
        "version": version.name,
        "decoding": "stochastic" if do_sample else "greedy",
        "completion_index": completion_index,
        "stochastic_seed": seed if do_sample else None,
        "top_p": top_p if do_sample else None,
        "temperature": temperature if do_sample else None,
        "completion_text": completion,
        "match_prefix_len": match_len,
        "exact_match": match_len >= len(seq.suffix_text),
        "has_logits": logit_rows is not None,
    }
    return row, logit_rows


def _parse_versions(spec: tuple[str, ...]) -> list[Version]:
    versions: list[Version] = []
    for s in spec:
        name, kind, path = s.split(":", 2)
        versions.append(Version(name=name, kind=kind, path=Path(path)))
    return versions


def _load_sequences(
    canaries_jsonl: Path | None,
    g2_jsonl: Path | None,
    g3_jsonl: Path | None,
) -> list[Sequence]:
    seqs: list[Sequence] = []
    if canaries_jsonl is not None:
        for c in read_jsonl(canaries_jsonl):
            seqs.append(Sequence(
                seq_id=c.canary_id, group="g1",
                prefix_text=c.prefix_text, suffix_text=c.suffix_text,
                bucket=c.frequency,
            ))
    for path, group in [(g2_jsonl, "g2"), (g3_jsonl, "g3")]:
        if path is None:
            continue
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                seqs.append(Sequence(
                    seq_id=row["seq_id"], group=group,
                    prefix_text=row["prefix_text"], suffix_text=row["suffix_text"],
                ))
    return seqs


@click.command()
@click.option("--canaries-jsonl", type=click.Path(path_type=Path), default=None,
              help="G1 canaries from qquilt.canaries")
@click.option("--g2-jsonl", type=click.Path(path_type=Path), default=None)
@click.option("--g3-jsonl", type=click.Path(path_type=Path), default=None)
@click.option(
    "--version", "version_specs", multiple=True, required=True,
    help="One per --version: NAME:hf:/path/to/hfdir, NAME:awq:/path/to/awq-dir, "
         "or NAME:gguf:/path/to/file.gguf",
)
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--logits-out", type=click.Path(path_type=Path), default=None,
              help="Optional separate JSONL for top-K logits (HF/AWQ greedy completions only).")
@click.option("--top-k", type=int, default=20)
@click.option("--max-new-tokens", type=int, default=50)
@click.option("--device", type=str, default="cuda")
@click.option("--llama-cli", type=click.Path(path_type=Path), default=None)
@click.option("--threads", type=int, default=8)
@click.option("--seed", type=int, default=42)
@click.option("--n-stochastic", type=int, default=0,
              help="Stochastic completions per (sequence × version). 0 = greedy only.")
@click.option("--top-p", type=float, default=0.9)
@click.option("--temperature", type=float, default=0.8)
def main(
    canaries_jsonl: Path | None, g2_jsonl: Path | None, g3_jsonl: Path | None,
    version_specs: tuple[str, ...], out: Path, logits_out: Path | None, top_k: int,
    max_new_tokens: int, device: str, llama_cli: Path | None, threads: int, seed: int,
    n_stochastic: int, top_p: float, temperature: float,
) -> None:
    seed_everything(seed)
    seqs = _load_sequences(canaries_jsonl, g2_jsonl, g3_jsonl)
    if not seqs:
        raise click.UsageError("at least one of --canaries-jsonl / --g2-jsonl / --g3-jsonl is required")
    versions = _parse_versions(version_specs)
    out.parent.mkdir(parents=True, exist_ok=True)

    capture_logits = logits_out is not None
    logits_handle = None
    if logits_out is not None:
        logits_out.parent.mkdir(parents=True, exist_ok=True)
        logits_handle = logits_out.open("w")

    n_rows = 0
    n_logit_rows = 0
    try:
        with out.open("w") as f:
            for v in versions:
                if v.kind == "hf":
                    hf_handle = _hf_load(v.path, device)
                elif v.kind == "awq":
                    hf_handle = _awq_load(v.path, device)
                else:
                    hf_handle = None
                try:
                    for seq in seqs:
                        row, logit_rows = _extract_with_decode(
                            seq=seq, version=v, max_new_tokens=max_new_tokens,
                            device=device, llama_cli=llama_cli, threads=threads,
                            hf_handle=hf_handle, do_sample=False,
                            top_p=top_p, temperature=temperature, seed=seed,
                            completion_index=0,
                            capture_logits=capture_logits and v.kind in ("hf", "awq"),
                            top_k=top_k,
                        )
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        n_rows += 1
                        if logits_handle is not None and logit_rows is not None:
                            for step_idx, lr in enumerate(logit_rows):
                                logits_handle.write(json.dumps({
                                    "schema": "qquilt.logits.v1", "schema_version": 1,
                                    "seq_id": seq.seq_id, "group": seq.group,
                                    "bucket": seq.bucket, "version": v.name,
                                    "step": step_idx, **lr,
                                }, ensure_ascii=False) + "\n")
                                n_logit_rows += 1
                        for k in range(n_stochastic):
                            row, _ = _extract_with_decode(
                                seq=seq, version=v, max_new_tokens=max_new_tokens,
                                device=device, llama_cli=llama_cli, threads=threads,
                                hf_handle=hf_handle, do_sample=True,
                                top_p=top_p, temperature=temperature,
                                seed=seed * 1000 + k,
                                completion_index=k + 1,
                                capture_logits=False,
                            )
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                            n_rows += 1
                finally:
                    if hf_handle is not None:
                        del hf_handle
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
    finally:
        if logits_handle is not None:
            logits_handle.close()

    click.echo(
        f"wrote {n_rows} extraction rows "
        f"({len(versions)} versions × {len(seqs)} sequences × {1 + n_stochastic} completions) to {out}"
    )
    if logits_out is not None:
        click.echo(f"wrote {n_logit_rows} logit rows to {logits_out}")


if __name__ == "__main__":
    main()
