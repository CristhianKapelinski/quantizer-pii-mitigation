"""GA_GDR unlearning driver for Zhang-replication experiments.

Applies Gradient Ascent on a forget set plus alpha-weighted Gradient
Descent on a retain set, per Zhang et al. ICLR 2025 §4.1 (the
GA_GDR baseline from MUSE benchmark, Shi et al. 2024b).

We mirror Zhang's loss function — minimize  ``-CE(forget) + alpha *
CE(retain)`` — and their LR=1e-5 / 5-epoch recipe on BOOKS. The driver
emits qquilt.unlearn.v1 telemetry per step so the existing 30-min
check loop / Monitor parsers keep working.

Hyperparameters are CLI-overridable. Default settings target our
small-scale adaptation: Llama-1B Phase A checkpoint as target, 100
canaries as forget set, 3000 Enron emails as retain set. See
``scripts/step_4_unlearn_quantize_attack.sh``.
"""

from __future__ import annotations

import json
import random
import socket
import time
from pathlib import Path

import click
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from qquilt.seed import seed_everything


def _load_texts(jsonl: Path, text_field: str = "text") -> list[str]:
    out = []
    with jsonl.open() as f:
        for line in f:
            rec = json.loads(line)
            if text_field in rec:
                out.append(rec[text_field])
            elif "prefix" in rec and "suffix" in rec:
                # canaries.jsonl shape — concatenate prefix+suffix as the
                # full string the model originally memorised
                out.append(rec["prefix"] + rec["suffix"])
            elif "prefix_text" in rec and "suffix_text" in rec:
                out.append(rec["prefix_text"] + rec["suffix_text"])
            else:
                raise ValueError(f"Cannot extract text from {rec.keys()}")
    return out


def _per_example_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean CE per example, ignoring -100 positions."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    flat = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.size())
    mask = (shift_labels != -100).float()
    per_ex = (flat * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return per_ex


def _tokenize(tokenizer, texts: list[str], max_len: int) -> dict:
    enc = tokenizer(
        texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt"
    )
    enc["labels"] = enc["input_ids"].clone()
    enc["labels"][enc["attention_mask"] == 0] = -100
    return enc


@click.command()
@click.option("--model-dir", type=click.Path(path_type=Path), required=True,
              help="Pre-trained / fine-tuned HF checkpoint to start from")
@click.option("--forget-jsonl", type=click.Path(path_type=Path), required=True,
              help="JSONL of forget examples (canaries)")
@click.option("--retain-jsonl", type=click.Path(path_type=Path), required=True,
              help="JSONL of retain examples (corpus excluding canaries)")
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
@click.option("--algo", type=click.Choice(["ga_gdr", "npo_gdr"]), default="ga_gdr")
@click.option("--epochs", type=int, default=5)
@click.option("--learning-rate", type=float, default=1e-5)
@click.option("--batch-size", type=int, default=2)
@click.option("--alpha", type=float, default=1.0,
              help="Weight on retain loss (Zhang BOOKS uses 1.0)")
@click.option("--beta", type=float, default=0.1,
              help="NPO temperature (only used by npo_gdr)")
@click.option("--ga-threshold", type=float, default=5.0,
              help="Per-example forget-CE threshold; once exceeded, the example "
                   "is masked out of the forget loss (Zhang ICLR 2025 protective "
                   "stop, prevents model collapse from unbounded gradient ascent)")
@click.option("--max-seq-len", type=int, default=512)
@click.option("--seed", type=int, default=42)
@click.option("--telemetry-jsonl", type=click.Path(path_type=Path), required=True)
def main(
    model_dir: Path, forget_jsonl: Path, retain_jsonl: Path, out_dir: Path,
    algo: str, epochs: int, learning_rate: float, batch_size: int,
    alpha: float, beta: float, ga_threshold: float, max_seq_len: int, seed: int,
    telemetry_jsonl: Path,
) -> None:
    seed_everything(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    telemetry_jsonl.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()

    # NPO needs a frozen reference at the original weights for the KL/log-ratio
    ref_model = None
    if algo == "npo_gdr":
        ref_model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(device).eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    forget_texts = _load_texts(forget_jsonl)
    retain_texts = _load_texts(retain_jsonl)
    rng = random.Random(seed)

    optim = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
                  eps=1e-8, weight_decay=0.01)

    banner = {
        "schema": "qquilt.unlearn.banner.v1",
        "schema_version": 1,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "torch": torch.__version__,
        "model_dir": str(model_dir),
        "algo": algo,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "alpha": alpha,
        "beta": beta,
        "max_seq_len": max_seq_len,
        "seed": seed,
        "n_forget": len(forget_texts),
        "n_retain": len(retain_texts),
    }
    print(json.dumps(banner))
    with telemetry_jsonl.open("w") as f:
        f.write(json.dumps(banner) + "\n")

    global_step = 0
    t0 = time.time()
    for ep in range(epochs):
        idx = list(range(len(forget_texts)))
        rng.shuffle(idx)
        for start in range(0, len(idx), batch_size):
            f_batch = [forget_texts[i] for i in idx[start:start + batch_size]]
            r_batch = rng.sample(retain_texts, len(f_batch))

            f_enc = _tokenize(tokenizer, f_batch, max_seq_len)
            r_enc = _tokenize(tokenizer, r_batch, max_seq_len)
            f_enc = {k: v.to(device) for k, v in f_enc.items()}
            r_enc = {k: v.to(device) for k, v in r_enc.items()}

            optim.zero_grad(set_to_none=True)

            # Forget pass — per-example CE, then threshold-mask to prevent collapse
            f_out = model(**f_enc)
            f_ce_per = _per_example_ce(f_out.logits, f_enc["labels"])
            # Mask examples that have already passed the threshold (Zhang protective stop)
            below = (f_ce_per < ga_threshold).float()
            n_active = below.sum().clamp(min=1)
            f_ce = (f_ce_per * below).sum() / n_active
            if algo == "ga_gdr":
                forget_loss = -f_ce  # gradient ASCENT, masked
            else:  # npo_gdr — Zhang ICLR 2025 §4.1
                with torch.no_grad():
                    ref_out = ref_model(**f_enc)
                    f_ref_ce = _per_example_ce(ref_out.logits, f_enc["labels"])
                # NPO objective per Zhang/Rafailov DPO offline:
                # L = -(2/beta) * logsigmoid(-beta * (log p - log p_ref))
                # We have CE (= -log p). So log p = -CE.
                log_ratio = -f_ce_per + f_ref_ce
                npo_per = -(2.0 / beta) * F.logsigmoid(-beta * log_ratio)
                forget_loss = (npo_per * below).sum() / n_active

            # Retain pass
            r_out = model(**r_enc)
            r_ce = _per_example_ce(r_out.logits, r_enc["labels"]).mean()
            retain_loss = alpha * r_ce

            loss = forget_loss + retain_loss
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            optim.step()

            global_step += 1
            row = {
                "schema": "qquilt.unlearn.v1",
                "schema_version": 1,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "epoch": ep,
                "step": global_step,
                "forget_ce": float(f_ce.detach()),
                "forget_ce_per_mean": float(f_ce_per.detach().mean()),
                "n_active": int(below.sum().item()),
                "retain_ce": float(r_ce.detach()),
                "loss": float(loss.detach()),
                "grad_norm": float(grad_norm),
                "wallclock_s": time.time() - t0,
            }
            if torch.cuda.is_available():
                row["gpu_alloc_gib"] = round(torch.cuda.memory_allocated() / 1024**3, 4)
                row["gpu_peak_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 4)
            with telemetry_jsonl.open("a") as f:
                f.write(json.dumps(row) + "\n")

    final = out_dir / "final"
    model.save_pretrained(str(final))
    tokenizer.save_pretrained(str(final))
    print(f"unlearned checkpoint: {final}")


if __name__ == "__main__":
    main()
