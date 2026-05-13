# 2026-05-10 — autoawq 0.2.7 quantization works on Blackwell (sm_120)

## Symptom (anticipated risk)

After bumping `torch` to 2.7.1+cu128 for sm_120, we expected `autoawq`
0.2.7 might fail with "no kernel image is available" the same way
`torch==2.5.1` did, since the autoawq wheel was built against an
older CUDA.

## Test

Ran an AWQ-4bit quantization end-to-end on the W0 fine-tune checkpoint
(`unsloth/Llama-3.2-1B-Instruct` post-fine-tune):

```
.venv/bin/python -m qquilt.quantize \
    --hf-dir checkpoints/wave_0/final \
    --out-dir checkpoints/wave_0/quantized \
    --quant AWQ \
    --awq-calibration-corpus experiment/results/wave_0/corpus.jsonl \
    --awq-calib-n 32 --awq-calib-seed 42
```

## Result

* Wallclock: 4 min 2 s for the 1B model with 32 calibration texts
  (16 transformer blocks × 15.1 s/block + I/O).
* Output: `model-awq-4bit/model.safetensors` (1.03 GB; vs 2.48 GB
  BF16 GGUF and 807 MB Q4_K_M GGUF).
* No "no kernel image" errors. Calibration ran on GPU.
* Benign warning from `accelerate.utils.imports`: "Intel Extension
  for PyTorch 2.8 needs to work with PyTorch 2.8.*" (pulled by
  `auto-gptq`, not by autoawq itself; ignorable).

## Implications

* **autoawq path is unblocked** for W1 mini Phase B (5 quants
  including AWQ-4bit) and W1 full (cross-calibration AWQ runs).
* W1 full AWQ calibration (256 samples per cross-cal variant) ≈
  ~30 min wallclock on the 5060 Ti — fits the W1 full GPU-h budget.
* The CUDA kernels autoawq uses for `from_quantized` inference also
  need verifying on Blackwell — defer to first AWQ extract run.
  If extract fails, journal entry follow-up.

## Cross-links

* `experiment/journal/2026-05-09-torch-blackwell.md`
* `experiment/journal/2026-05-10-bitsandbytes-cu128.md`
* `experiment/wave_1/WAVE_1_LOG.md`
