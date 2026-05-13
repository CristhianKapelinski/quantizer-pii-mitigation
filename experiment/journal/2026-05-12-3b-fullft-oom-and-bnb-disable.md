# 2026-05-12 — Llama-3.2-3B full-FT does not fit 16 GB; bitsandbytes disabled to unblock peft

## What happened

Launched `scripts/run_extra_anchors.sh` (3B scale anchor + Qwen extra
seeds). Two toolchain facts surfaced immediately:

### 1. 3B full fine-tune OOMs on the 16 GB GPU (documented negative)

`unsloth/Llama-3.2-3B-Instruct` is 3.24 B params. A full fine-tune keeps,
in BF16: params (~6.5 GB) + grads (~6.5 GB) ≈ 13.6 GB before any
optimiser state, activations, logits, or CUDA/cuDNN working memory. On the
RTX 5060 Ti (15.48 GiB usable), training OOMs during the first
forward/backward — at **both** `max_seq_len=384` and `256` (the gap is the
param+grad footprint, which does not depend on sequence length):

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.47 GiB.
GPU 0 has a total capacity of 15.48 GiB of which 840.06 MiB is free.
... this process has 13.83 GiB memory in use ...
```

A paged 8-bit optimiser (`paged_adamw_8bit`) would help, but bitsandbytes
is non-functional here (see §2). Adafactor (factored second moment, no
first moment, `optim=adafactor`) was tried as the memory-light full-FT
optimiser — it brings the optimiser state to ~negligible, but the bare
params+grads (~13.6 GB) plus CUDA context still exceed 16 GB. So:

> **The Llama-3.2-3B point in this paper is LoRA-merged, not full-FT.**
> Full-FT 3B is infeasible on the available hardware. Tag
> `experiment/results/wave_1_llama3b_seed42/` keeps the attempt's
> `train_steps.jsonl` banner (config + the OOM) as the documented
> negative; `wave_1_llama3b_lora_seed42/` is the actual 3B anchor.

This is also scientifically the more interesting regime: the mechanistic
prediction is that the smaller LoRA weight-delta narrows the
AWQ-vs-Q4_K_M extraction gap relative to the 1B full-FT headline
(`scripts/exp_delta_norm.py` measures ‖Δθ‖ for both, to test that).

### 2. bitsandbytes 0.44.1 is broken against the installed Triton

`bitsandbytes==0.44.1` (pinned in `pyproject.toml`) does
`from triton.ops.matmul_perf_model import early_config_prune, ...`, but
modern Triton removed the `triton.ops` namespace, so `import bitsandbytes`
raises `ModuleNotFoundError: No module named 'triton.ops'`. The wheel also
lacks `libbitsandbytes_cuda128.so`, so it never had GPU support in this
venv anyway.

This breaks **peft**: `get_peft_model(...)` -> `_create_new_module` ->
`is_bnb_available()` returns True (the package *exists*, just doesn't
import) -> `from .bnb import dispatch_bnb_8bit` -> the import error
propagates -> LoRA creation fails. So the 3B LoRA run failed at module
injection, not at training.

**Fix applied (venv-local, reversible):** renamed the broken package out
of the way so `importlib.util.find_spec("bitsandbytes")` returns None,
which makes `peft`'s (and `transformers`') `is_bnb_available()` False and
restores plain BF16 LoRA:

```
mv .venv/lib/python3.11/site-packages/bitsandbytes \
   .venv/lib/python3.11/site-packages/bitsandbytes.disabled-broken-triton-ops
mv .venv/lib/python3.11/site-packages/bitsandbytes-*.dist-info \
   .venv/lib/python3.11/site-packages/_disabled_bitsandbytes_dist_info
```

Verified afterwards: `transformers`, `peft`, `autoawq`, and
`from peft import LoraConfig, get_peft_model` all import cleanly;
`is_bnb_available()` is False; nothing else in the pipeline uses
bitsandbytes (no `load_in_8bit`, no GPTQ-via-bnb, no 8-bit optimisers).

**Caveat for reproduction:** a `uv sync` would reinstall the broken
`bitsandbytes==0.44.1`. The clean fix is to drop it from `pyproject.toml`
(it is unused) or pin a Triton-3-compatible release; left for the
maintainer to decide. The 3B LoRA run was relaunched after the rename and
trains normally.

## Cross-links

* `experiment/plans/2026-05-12-scale-and-crossfamily-anchors.md` — the plan
* `experiment/journal/2026-05-10-bitsandbytes-cu128.md` — earlier bnb / cu128 incompatibility
* `scripts/run_extra_anchors.sh`, `scripts/exp_extra_run.sh`, `src/qquilt/train.py`
