# 2026-05-10 — bitsandbytes 0.44.1 incompatible with torch 2.7.1+cu128

## Symptom

After running `uv sync --extra quant` to install the W1+ quantization
extras (`bitsandbytes==0.44.1`, `autoawq==0.2.7`, `auto-gptq==0.7.1`),
the bitsandbytes import emits:

```
Could not find the bitsandbytes CUDA binary at
  PosixPath('.../site-packages/bitsandbytes/libbitsandbytes_cuda128.so')
The installed version of bitsandbytes was compiled without GPU support.
8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.

bitsandbytes import FAILED: ModuleNotFoundError: No module named 'triton.ops'
```

Two distinct problems:

* The 0.44.1 wheel ships CUDA binaries for cu118/cu121/cu124 but not
  cu128 (the index we use for torch 2.7.1+cu128). Without a matching
  binary, bitsandbytes refuses to expose GPU paths.
* The 0.44.1 stack imports `triton.ops`, which was removed in triton 3.x
  (we have `triton==3.3.1` from the torch 2.7.1 release). `triton.ops`
  was deprecated and dropped between triton 2.x and 3.x.

`autoawq==0.2.7` import is unaffected — it loads cleanly on
`torch 2.7.1+cu128` and is the path we need for W1+ AWQ-4bit.

`auto-gptq==0.7.1` pulled `intel-extension-for-pytorch==2.8.0` and
`gekko`, which warn at import that they expect torch 2.8. Auto-GPTQ
itself has not been smoke-tested in this venv yet.

## Investigation

bitsandbytes release log (github.com/bitsandbytes-foundation/bitsandbytes/releases):

* 0.44.x line — last release Oct 2024, cu118/cu121/cu124 wheels only.
* 0.45.x — Jan 2025, adds cu128 wheel; still uses old triton API.
* 0.46.x — Mar/Apr 2025, drops the `triton.ops` import path.
* 0.47.x — Jul 2025, current minimum for torch ≥ 2.7 ergonomics.

Triton release log: `triton.ops` was the namespace for fused kernels
shipped before the rewrite to TritonLang in 3.0. The 2.x tail had a
shim, but `triton==3.x` removes it.

## Cause

PLAN.md §9 pin (`bitsandbytes==0.44.1`) is from the same Oct 2024
batch as the original `torch==2.5.1` pin. When we bumped torch to
2.7.1+cu128 (journal entry 2026-05-09-torch-blackwell.md), the rest
of the quant extra stack should have been bumped in lockstep — it
wasn't, because `--extra quant` was not exercised during W0.

## Resolution

For W1 mini Phase A: not affected. Phase A uses GGUF quants only
(via the llama.cpp binaries) and does NOT call into bitsandbytes
or auto-gptq.

For W1 mini Phase B (adds AWQ): `autoawq==0.2.7` works on
`torch 2.7.1+cu128` and is the only quant extra needed for the AWQ
path. Phase B is unblocked.

For W1 full G5 DP-SGD baseline: depends on Opacus (separate dep,
not yet pinned). If we want 8-bit Adam states for memory savings we
will need to bump:

* `bitsandbytes` 0.44.1 → ≥ 0.47 (Blackwell + triton 3.x compatible).
* `auto-gptq` 0.7.1 → newer release that drops the
  `intel-extension-for-pytorch` hard pin (or accept the warning).

Do the bump in a dedicated commit before W1 full dispatch. Update
`EXPERIMENT_MANIFEST.yaml` `toolchain.packages.bitsandbytes` and
`auto-gptq`, document the new pins in another journal entry,
re-run W1 mini gate to confirm nothing shifted.

`Why:` we don't want to add scope to W1 mini Phase A by chasing
quant-extra compatibility right now. Phase A's GGUF-only path runs
end-to-end without any of these libraries.

`How to apply:` keep `bitsandbytes` / `auto-gptq` as installed but
flagged. When the AWQ/Opacus path becomes critical (Phase B
or W1 full G5), bump in lockstep with a dispatch-time check.

## Open questions

* `autoawq==0.2.7` quantizes via its own kernels, but at inference
  time it loads via `AutoAWQForCausalLM.from_quantized` which uses
  triton + custom CUDA. Need to confirm it actually runs on sm_120
  before committing the W1 mini Phase B dispatch — first AWQ-4bit
  inference run will tell us.
* Opacus + DP-SGD compatibility with torch 2.7 has not been
  validated yet. Defer to W1 full G5 prep.

## Cross-links

* `experiment/journal/2026-05-09-torch-blackwell.md`
* `experiment/journal/2026-05-09-pin-conflict.md`
* `EXPERIMENT_MANIFEST.yaml` `toolchain.packages`
