"""Wave 0 / Wave 1+ fine-tune driver.

Full BF16 fine-tune of a HuggingFace causal-LM on a JSONL corpus
(produced by ``qquilt.data``). Hyperparameters follow PLAN.md §5.1; the
smoke wires up only what's needed for W0 (3 epochs, seed 42, no
multi-seed loop yet). Per-step training telemetry is emitted to a
JSONL file so the 30-min check loop can read it without parsing logs.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import click
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from qquilt.seed import seed_everything


def _gib(n_bytes: int) -> float:
    return round(n_bytes / 1024**3, 4)


def _system_ram_gib() -> dict:
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        rss_kb = ru.ru_maxrss  # kilobytes on linux
        return {"max_rss_gib": round(rss_kb / 1024**2, 4)}
    except Exception:
        return {}


def _gpu_snapshot() -> dict:
    if not torch.cuda.is_available():
        return {}
    return {
        "gpu_alloc_gib": _gib(torch.cuda.memory_allocated()),
        "gpu_reserved_gib": _gib(torch.cuda.memory_reserved()),
        "gpu_peak_alloc_gib": _gib(torch.cuda.max_memory_allocated()),
        "gpu_peak_reserved_gib": _gib(torch.cuda.max_memory_reserved()),
    }


def _nvidia_smi_fingerprint() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "-q"], capture_output=True, text=True, check=True, timeout=10,
        ).stdout
        import hashlib
        return hashlib.sha256(out.encode()).hexdigest()
    except Exception:
        return None


def _env_banner(model_id: str, seed: int, n_records: int, args: TrainingArguments) -> dict:
    return {
        "schema": "qquilt.train.banner.v1",
        "schema_version": 1,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "device_total_mem_gib": _gib(torch.cuda.get_device_properties(0).total_memory) if torch.cuda.is_available() else None,
        "torch_arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else None,
        "nvidia_smi_sha256": _nvidia_smi_fingerprint(),
        "model_id": model_id,
        "seed": seed,
        "n_train_records": n_records,
        "batch_size": args.per_device_train_batch_size,
        "grad_accum": args.gradient_accumulation_steps,
        "effective_batch": args.per_device_train_batch_size * args.gradient_accumulation_steps,
        "lr": args.learning_rate,
        "epochs": args.num_train_epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "optim": str(getattr(args, "optim", None)),
        "max_seq_len": int(os.environ.get("QQUILT_MAX_SEQ_LEN", "0")) or None,
        "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"),
    }


@dataclass
class TelemetryCallback(TrainerCallback):
    """Per-step JSONL telemetry: loss, lr, wallclock, GPU/RAM peaks.

    Writes one ``qquilt.train.banner.v1`` row at start (env + hw + hp), one
    ``qquilt.train.v1`` row per logged step (loss/lr + memory deltas), and one
    ``qquilt.train.summary.v1`` row at the end (peak memory, total wallclock).
    """

    path: Path
    started_at: float = 0.0
    last_step_at: float = 0.0
    banner: dict = field(default_factory=dict)

    def _append(self, row: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        self.started_at = time.monotonic()
        self.last_step_at = self.started_at
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if self.banner:
            self._append(self.banner)

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if not logs:
            return
        now = time.monotonic()
        step_time_s = now - self.last_step_at
        self.last_step_at = now
        row = {
            "schema": "qquilt.train.v1",
            "schema_version": 2,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "step": int(state.global_step),
            "epoch": float(state.epoch or 0.0),
            "loss": float(logs.get("loss")) if "loss" in logs else None,
            "lr": float(logs.get("learning_rate")) if "learning_rate" in logs else None,
            "grad_norm": float(logs.get("grad_norm")) if "grad_norm" in logs else None,
            "wallclock_s": now - self.started_at,
            "step_time_s": step_time_s,
            **_gpu_snapshot(),
            **_system_ram_gib(),
        }
        self._append(row)

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ANN001
        end = time.monotonic()
        summary = {
            "schema": "qquilt.train.summary.v1",
            "schema_version": 1,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_wallclock_s": end - self.started_at,
            "global_step_final": int(state.global_step),
            "epoch_final": float(state.epoch or 0.0),
            **_gpu_snapshot(),
            **_system_ram_gib(),
        }
        self._append(summary)


def _load_corpus(corpus_jsonl: Path) -> list[str]:
    with corpus_jsonl.open() as f:
        return [json.loads(line)["text"] for line in f]


def run(
    *,
    model_id: str,
    corpus_jsonl: Path,
    out_dir: Path,
    epochs: float,
    learning_rate: float,
    batch_size: int,
    grad_accumulation: int,
    warmup_ratio: float,
    weight_decay: float,
    max_seq_len: int,
    seed: int,
    telemetry_jsonl: Path,
    optim: str = "adamw_torch",
    lora_r: int = 0,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
) -> Path:
    seed_everything(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )

    # Optional LoRA/PEFT regime. lora_r == 0 keeps the default full fine-tune
    # (behaviour unchanged). lora_r > 0 trains low-rank adapters on the listed
    # projection modules, then merges them back into the base weights before
    # save_model() so the on-disk checkpoint is a plain HF model (downstream
    # GGUF / AWQ steps stay identical). Used for the 3B scale point, where a
    # full-FT optimiser state does not fit a 16 GB GPU; the prediction is that
    # the smaller weight-delta narrows the AWQ-vs-Q4_K_M gap.
    use_lora = lora_r and lora_r > 0
    if use_lora:
        from peft import LoraConfig, get_peft_model

        lcfg = LoraConfig(
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            target_modules=[m.strip() for m in lora_target_modules.split(",") if m.strip()],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lcfg)
        model.enable_input_require_grads()  # needed for grad-checkpointing + frozen base
        model.print_trainable_parameters()

    texts = _load_corpus(corpus_jsonl)
    raw = Dataset.from_list([{"text": t} for t in texts])

    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, max_length=max_seq_len)

    ds = raw.map(tokenize, batched=True, remove_columns=["text"])

    # gradient_checkpointing trades compute for memory. Without it, full-FT
    # of Llama-3.2-1B in BF16 (params 2.5 GB + Adam fp32 states 10 GB +
    # activations 3-4 GB + transient logits.float() ~1.5 GB) exceeds
    # 16 GB on the 5060 Ti. Checkpointing drops activations to ~0.5 GB.
    # use_cache=False is required when gradient_checkpointing is on.
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accumulation,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=optim,
        logging_steps=1,
        save_strategy="no",
        seed=seed,
        report_to="none",
        dataloader_num_workers=2,
    )

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    callback = TelemetryCallback(
        path=telemetry_jsonl,
        banner=_env_banner(model_id=model_id, seed=seed, n_records=len(texts), args=args),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[callback],
    )
    trainer.train()

    final_dir = out_dir / "final"
    if use_lora:
        # Merge the LoRA adapters into the base weights and save a plain HF
        # checkpoint, so quantize / extract see a normal model directory.
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(str(final_dir))
    else:
        trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    return final_dir


@click.command()
@click.option("--model-id", type=str, required=True)
@click.option("--corpus-jsonl", type=click.Path(path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
@click.option("--epochs", type=float, required=True)
@click.option("--learning-rate", type=float, default=2e-5)
@click.option("--batch-size", type=int, default=4)
@click.option("--grad-accumulation", type=int, default=4)
@click.option("--warmup-ratio", type=float, default=0.03)
@click.option("--weight-decay", type=float, default=0.01)
@click.option("--max-seq-len", type=int, default=1024)
@click.option("--seed", type=int, required=True)
@click.option("--telemetry-jsonl", type=click.Path(path_type=Path), required=True)
@click.option("--optim", type=str, default="adamw_torch",
              help="transformers optimiser name (e.g. adamw_torch, adafactor)")
@click.option("--lora-r", type=int, default=0,
              help="LoRA rank; 0 = full fine-tune (default). >0 merges adapters before save.")
@click.option("--lora-alpha", type=int, default=32)
@click.option("--lora-dropout", type=float, default=0.05)
@click.option("--lora-target-modules", type=str,
              default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
def main(**kw: object) -> None:
    final = run(**kw)  # type: ignore[arg-type]
    click.echo(f"final checkpoint: {final}")


if __name__ == "__main__":
    main()
