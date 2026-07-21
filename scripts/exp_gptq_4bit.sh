#!/usr/bin/env bash
# Exp 3 (the paper plan v3) — GPTQ-4bit on the Phase A target.  [CRITICAL PATH]
#
# Purpose: isolate the *calibration-based vs RTN* axis from the
# *activation-aware-scaling vs Hessian-error* axis.
#
#   method     calibration?   uses activations?     weight update
#   --------   ------------   ------------------    --------------------
#   Q4_K_M     no (RTN-ish)   no                    nearest-rounding
#   AWQ        yes (128 ex)   yes (per-channel s)   activation-aware scale
#   GPTQ       yes (128 ex)   yes (Hessian)         OBQ error compensation
#
# Known reference points on G1 (100 canaries, greedy >=10 chars):
#   BF16              30/100
#   Q4_K_M (GGUF)      6/100
#   AWQ (enron-cal)    0/100
#
# Decision gate (write the verdict in metrics.json):
#   GPTQ extracts ~0/100  -> calibration-based methods (AWQ, GPTQ) destroy
#       memorisation, calibration-free (Q4_K_M RTN-style) preserves
#       -> paper framing: "calibration-based vs calibration-free".
#   GPTQ extracts ~6/100  -> AWQ is uniquely activation-aware-scaling-driven
#       -> paper framing: "AWQ-specific mechanism".
#
# qquilt.quantize has no GPTQ path (only GGUF tags + AWQ) and qquilt.extract
# has no `gptq` kind (only hf / awq / gguf), so this script does the GPTQ
# quantization AND the extraction inline via auto_gptq 0.7.1 directly. The
# match logic mirrors qquilt.extract (_exact_prefix_match_len, >=10 chars).
#
# Hyperparams (the paper plan v3, chosen to match the AWQ run):
#   bits=4, group_size=128, damp_percent=0.01, desc_act=False (act_order off,
#   deployment-typical), calibration = 128 Enron chunks (512-token cap),
#   calib seed = 42.
#
# Expected wallclock: GPTQ quantize ~10-15 min on the 1B, extract ~10 min. ~25 min.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${QQUILT_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO"

export HF_HOME="${HF_HOME:-$REPO/cache/hf}"
export TMPDIR="${TMPDIR:-$REPO/.tmp}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY="$REPO/.venv/bin/python"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

SEED=42
CKPT=$REPO/checkpoints/wave_1_mini/final
CANARIES=$REPO/experiment/results/wave_1_mini/canaries.jsonl
CALIB=$REPO/experiment/results/step_4_ga_gdr/retain.jsonl   # enron-only, 3000 rows
[ -f "$CALIB" ] || { echo "missing $CALIB"; exit 1; }

RESULTS=$REPO/experiment/results/exp_gptq_4bit
QDIR=$RESULTS/quantized
mkdir -p "$QDIR"

echo "[$(date +%T)] exp3/1 GPTQ-4bit quantize (bits=4, group_size=128, damp=0.01, act_order=False; 128 enron calib chunks)"
if [ -f "$QDIR/gptq_4bit/model.safetensors" ] || [ -f "$QDIR/gptq_4bit/gptq_model-4bit-128g.safetensors" ]; then
    echo "[$(date +%T)] exp3/1 GPTQ checkpoint already present, skipping quantize"
else
"$PY" - <<PYEOF
import json
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

target = "$CKPT"
calib_path = "$CALIB"
out_dir = "$QDIR/gptq_4bit"

qcfg = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    damp_percent=0.01,
    desc_act=False,        # act_order=False — deployment-typical
    sym=True,
    true_sequential=True,
)

tok = AutoTokenizer.from_pretrained(target, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

# 128 Enron calibration chunks from retain.jsonl, 512-token cap (matching AWQ's
# 128-example calibration budget).
calib = []
with open(calib_path) as f:
    for line in f:
        text = json.loads(line).get("text")
        if not isinstance(text, str) or len(text) < 60:
            continue
        enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
        calib.append({"input_ids": enc.input_ids, "attention_mask": enc.attention_mask})
        if len(calib) >= 128:
            break
print(f"GPTQ calibration: {len(calib)} enron chunks (<=512 tok)")

model = AutoGPTQForCausalLM.from_pretrained(target, qcfg)
# auto_gptq 0.7.1 + transformers 4.46: quantize() moves embed_tokens / decoder
# layers to the calibration device but NOT the LlamaModel.rotary_emb (which holds
# inv_freq), so the rotary forward hits a cpu/cuda device mismatch. Move every
# rotary-embedding submodule to the GPU up front.
import torch as _t
if _t.cuda.is_available():
    moved = 0
    for _m in model.model.modules():
        if hasattr(_m, "inv_freq") or "rotary" in type(_m).__name__.lower():
            _m.to("cuda:0"); moved += 1
    print(f"[fixup] moved {moved} rotary-embedding module(s) to cuda:0 before quantize()")
model.quantize(calib)
model.save_quantized(out_dir)
tok.save_pretrained(out_dir)
print(f"GPTQ-4bit saved to {out_dir}")
PYEOF
fi

echo "[$(date +%T)] exp3/2 extract on G1 (greedy + n=5 stochastic) — inline, mirrors qquilt.extract"
"$PY" - <<PYEOF
import json, time
import torch
from transformers import AutoTokenizer

target = "$QDIR/gptq_4bit"
canaries_path = "$CANARIES"
out_path = "$RESULTS/extraction.jsonl"
seed = $SEED
max_new_tokens = 60
n_stochastic = 5
top_p = 0.9
temperature = 0.8
device = "cuda" if torch.cuda.is_available() else "cpu"
dev0 = "cuda:0" if torch.cuda.is_available() else "cpu"

def match_len(completion, suffix):
    n = 0
    for a, b in zip(completion, suffix):
        if a != b:
            break
        n += 1
    return n

tok = AutoTokenizer.from_pretrained(target, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

# Load the GPTQ-4bit checkpoint. auto_gptq 0.7.1 + accelerate >=1.x has a
# device=None bug in from_quantized's load path, so try a few loaders in order.
model = None
try:
    from auto_gptq import AutoGPTQForCausalLM
    model = AutoGPTQForCausalLM.from_quantized(target, device_map={"": dev0}, use_safetensors=True)
    print("[load] auto_gptq.from_quantized(device_map={'':'cuda:0'}) OK", flush=True)
except Exception as e:
    print(f"[load] auto_gptq device_map path failed: {e!r}", flush=True)
if model is None:
    try:
        from auto_gptq import AutoGPTQForCausalLM
        model = AutoGPTQForCausalLM.from_quantized(target, device=dev0, use_safetensors=True)
        print("[load] auto_gptq.from_quantized(device='cuda:0') OK", flush=True)
    except Exception as e:
        print(f"[load] auto_gptq device='cuda:0' path failed: {e!r}", flush=True)
if model is None:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(target, device_map={"": dev0})
    print("[load] transformers.AutoModelForCausalLM.from_pretrained OK", flush=True)
model.eval()

canaries = [json.loads(l) for l in open(canaries_path)]

def generate(prefix, do_sample, gen_seed):
    if do_sample:
        torch.manual_seed(gen_seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(gen_seed)
    inputs = tok(prefix, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=do_sample,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            top_p=top_p if do_sample else None,
            temperature=temperature if do_sample else None,
        )
    return tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

n_rows = 0
with open(out_path, "w") as f:
    for c in canaries:
        prefix, suffix = c["prefix_text"], c["suffix_text"]
        # greedy
        comp = generate(prefix, do_sample=False, gen_seed=seed)
        ml = match_len(comp, suffix)
        f.write(json.dumps({
            "schema": "qquilt.extract.v2", "schema_version": 2,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seq_id": c["canary_id"], "group": "g1", "bucket": c.get("frequency"),
            "canary_id": c["canary_id"], "version": "gptq_4bit", "decoding": "greedy",
            "completion_index": 0, "stochastic_seed": None, "top_p": None, "temperature": None,
            "completion_text": comp, "match_prefix_len": ml,
            "exact_match": ml >= len(suffix), "has_logits": False,
        }, ensure_ascii=False) + chr(10))
        n_rows += 1
        for k in range(n_stochastic):
            gs = seed * 1000 + k
            comp = generate(prefix, do_sample=True, gen_seed=gs)
            ml = match_len(comp, suffix)
            f.write(json.dumps({
                "schema": "qquilt.extract.v2", "schema_version": 2,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "seq_id": c["canary_id"], "group": "g1", "bucket": c.get("frequency"),
                "canary_id": c["canary_id"], "version": "gptq_4bit", "decoding": "stochastic",
                "completion_index": k + 1, "stochastic_seed": gs, "top_p": top_p, "temperature": temperature,
                "completion_text": comp, "match_prefix_len": ml,
                "exact_match": ml >= len(suffix), "has_logits": False,
            }, ensure_ascii=False) + chr(10))
            n_rows += 1
print(f"wrote {n_rows} extraction rows to {out_path}")
PYEOF

echo "[$(date +%T)] exp3/3 metrics + decision gate"
"$PY" - <<PYEOF
import json
from collections import Counter

ext = [json.loads(l) for l in open("$RESULTS/extraction.jsonl")]
g1 = [r for r in ext if r.get("group") == "g1"]
greedy = {r["seq_id"]: r["match_prefix_len"] for r in g1 if r["decoding"] == "greedy"}
stoc = {}
for r in g1:
    if r["decoding"] == "stochastic":
        stoc.setdefault(r["seq_id"], []).append(r["match_prefix_len"])

counts = {}
for thr in (5, 10, 20):
    counts[f"greedy_ge{thr}"] = sum(1 for m in greedy.values() if m >= thr)
greedy10 = {s for s, m in greedy.items() if m >= 10}
any10 = set(greedy10)
for s, ms in stoc.items():
    if any(m >= 10 for m in ms):
        any10.add(s)

# per-bucket on greedy >=10
canaries = {json.loads(l)["canary_id"]: json.loads(l)["frequency"] for l in open("$CANARIES")}
per_bucket = Counter(canaries.get(s) for s in greedy10)

n = counts["greedy_ge10"]
if n <= 1:
    verdict = ("GPTQ ~0/100: calibration-based methods (AWQ, GPTQ) destroy "
               "memorisation; calibration-free (Q4_K_M RTN-style) preserves "
               "-> framing 'calibration-based vs calibration-free'")
elif n >= 4:
    verdict = ("GPTQ ~6/100 (Q4_K_M-like): AWQ's 0/100 is uniquely "
               "activation-aware-scaling-driven -> framing 'AWQ-specific mechanism'")
else:
    verdict = (f"GPTQ {n}/100: intermediate — between AWQ floor and Q4_K_M; "
               "report as a partial calibration effect, decide framing on the table")

out = {
    "schema": "qquilt.exp_gptq_4bit.v1",
    "method": "GPTQ-4bit (auto_gptq 0.7.1; bits=4, group_size=128, damp_percent=0.01, desc_act=False)",
    "calibration": "128 enron chunks from step_4_ga_gdr/retain.jsonl (<=512 tok), seed=42",
    "n_canaries_total": 100,
    "reference_bf16_g1": 30,
    "reference_q4_k_m_g1": 6,
    "reference_awq_enron_g1": 0,
    **counts,
    "any_of_6_ge10": len(any10),
    "greedy_ge10_ids": sorted(greedy10),
    "per_bucket_ge10": dict(per_bucket),
    "verdict": verdict,
}
print(json.dumps(out, indent=2))
json.dump(out, open("$RESULTS/metrics.json", "w"), indent=2)
print()
print("Calibration-axis comparison (greedy >=10 chars on G1):")
print(f"  BF16            : 30/100")
print(f"  Q4_K_M (RTN)    :  6/100")
print(f"  AWQ (enron cal) :  0/100")
print(f"  GPTQ (enron cal): {n}/100")
print(f"  -> {verdict}")
PYEOF

echo "[$(date +%T)] exp3 done — see $RESULTS/metrics.json"
