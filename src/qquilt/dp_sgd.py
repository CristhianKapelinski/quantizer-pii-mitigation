"""DP-SGD baseline (G5) via Opacus — used as defense baseline in W1 full.

PLAN.md §4 G5: ``Llama-3.2-1B fine-tuned with DP-SGD ε=4 (Panda et al.
ICLR 2025 setup), σ via Opacus accountant, C=1.0``. The expectation is
that DP-SGD attenuates Métrica 1b to ~0 — that confirms (a) the attack
is responsive to DP and (b) the proposed defense baseline works.

This module is structured as a separate fine-tune pipeline (``qquilt.train``
covers the non-private path). Two AGENT_HANDOFF §9 known issues are
hard-coded fixes:

* Opacus PRV accountant: ``epochs=max(1, math.ceil(num_epochs))`` to avoid
  ``ZeroDivisionError`` when sub-epoch caps round to 0.
* Empty Poisson batch on small ``q``: ``optimizer.signal_skip_step``
  to skip the step cleanly.

NOTE: uses optional dep ``opacus`` (install with
``uv sync --directory ... --extra dp``). Skeleton — tested integration
arrives with the W1 full G5 dispatch.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import click
import torch
from datasets import Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from qquilt.seed import seed_everything


@dataclass
class DPTelemetry:
    path: Path
    started_at: float = 0.0
    last_step_at: float = 0.0
    banner: dict = field(default_factory=dict)

    def _append(self, row: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def begin(self) -> None:
        self.started_at = time.monotonic()
        self.last_step_at = self.started_at
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if self.banner:
            self._append(self.banner)

    def step(self, *, step: int, epoch: float, loss: float | None,
             grad_norm: float | None, lr: float | None,
             epsilon: float | None) -> None:
        now = time.monotonic()
        row = {
            "schema": "qquilt.dp_train.v1",
            "schema_version": 1,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "lr": lr,
            "grad_norm": grad_norm,
            "epsilon": epsilon,
            "wallclock_s": now - self.started_at,
            "step_time_s": now - self.last_step_at,
        }
        if torch.cuda.is_available():
            row["gpu_alloc_gib"] = round(torch.cuda.memory_allocated() / 1024**3, 4)
            row["gpu_peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 4)
        self.last_step_at = now
        self._append(row)


def run_dp(
    *,
    model_id: str,
    corpus_jsonl: Path,
    out_dir: Path,
    epochs: float,
    learning_rate: float,
    batch_size: int,
    max_grad_norm: float,
    target_epsilon: float,
    target_delta: float,
    max_seq_len: int,
    seed: int,
    telemetry_jsonl: Path,
) -> Path:
    """DP-SGD fine-tune. Returns the saved checkpoint dir.

    Implementation outline (W1 full G5):

    1. Load tokenizer + model in bf16.
    2. Tokenise the corpus exactly the same way as ``qquilt.train``.
    3. Build a vanilla ``DataLoader`` (Opacus ``make_private_with_epsilon``
       replaces it with a Poisson-sampled equivalent).
    4. Create ``AdamW`` optimiser with the same betas / eps as the
       non-private path.
    5. Wrap with ``opacus.PrivacyEngine`` using ``accountant="prv"`` and
       ``epochs=max(1, math.ceil(epochs))``.
    6. Train loop: for each batch, zero_grad / forward / loss / backward /
       step. On empty Poisson batch (input shape 0), call
       ``optimizer.signal_skip_step(do_skip=True)`` and continue.
    7. Write per-step JSONL telemetry (loss, lr, ε accumulated, gpu peaks).
    8. Save model with ``model._module.save_pretrained`` (Opacus wraps the
       model in ``GradSampleModule`` — unwrap before save).
    """
    try:
        from opacus import PrivacyEngine  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "qquilt.dp_sgd requires opacus. Install with "
            "`uv sync --directory ... --extra dp`."
        ) from e

    seed_everything(seed)
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    with corpus_jsonl.open() as f:
        texts = [json.loads(line)["text"] for line in f]
    raw = Dataset.from_list([{"text": t} for t in texts])

    def tokenize(batch: dict) -> dict:
        return tok(batch["text"], truncation=True, max_length=max_seq_len)

    ds = raw.map(tokenize, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collator)

    optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    from opacus import PrivacyEngine
    pe = PrivacyEngine(accountant="prv")
    epochs_int = max(1, math.ceil(epochs))
    model, optimizer, loader = pe.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        epochs=epochs_int,
        max_grad_norm=max_grad_norm,
    )

    tele = DPTelemetry(
        path=telemetry_jsonl,
        banner={
            "schema": "qquilt.dp_train.banner.v1", "schema_version": 1,
            "model_id": model_id, "seed": seed,
            "n_train_records": len(texts),
            "batch_size": batch_size, "max_seq_len": max_seq_len,
            "lr": learning_rate, "epochs": epochs,
            "target_epsilon": target_epsilon, "target_delta": target_delta,
            "max_grad_norm": max_grad_norm,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__,
        },
    )
    tele.begin()

    step = 0
    for epoch in range(epochs_int):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if batch["input_ids"].shape[0] == 0:
                # Empty Poisson lot — AGENT_HANDOFF §9 fix.
                optimizer.signal_skip_step(do_skip=True)
                continue
            optimizer.zero_grad()
            out = model(**batch)
            loss = out.loss
            if loss is None or torch.isnan(loss):
                optimizer.signal_skip_step(do_skip=True)
                continue
            loss.backward()
            optimizer.step()
            try:
                eps = pe.accountant.get_epsilon(delta=target_delta)
            except Exception:
                eps = None
            tele.step(
                step=step, epoch=step / max(1, len(loader)),
                loss=float(loss.item()),
                grad_norm=None,  # Opacus already clips; raw norm not relevant
                lr=learning_rate, epsilon=eps,
            )
            step += 1

    final_dir = out_dir / "final"
    # Unwrap Opacus before saving so qquilt.extract can load via standard HF API.
    inner = model._module if hasattr(model, "_module") else model
    inner.save_pretrained(str(final_dir))
    tok.save_pretrained(str(final_dir))
    return final_dir


@click.command()
@click.option("--model-id", type=str, required=True)
@click.option("--corpus-jsonl", type=click.Path(path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
@click.option("--epochs", type=float, required=True)
@click.option("--learning-rate", type=float, default=2e-5)
@click.option("--batch-size", type=int, default=2)
@click.option("--max-grad-norm", type=float, default=1.0)
@click.option("--target-epsilon", type=float, default=4.0)
@click.option("--target-delta", type=float, default=1e-5)
@click.option("--max-seq-len", type=int, default=512)
@click.option("--seed", type=int, required=True)
@click.option("--telemetry-jsonl", type=click.Path(path_type=Path), required=True)
def main(**kw: object) -> None:
    final = run_dp(**kw)  # type: ignore[arg-type]
    click.echo(f"DP-SGD final checkpoint: {final}")


if __name__ == "__main__":
    main()
