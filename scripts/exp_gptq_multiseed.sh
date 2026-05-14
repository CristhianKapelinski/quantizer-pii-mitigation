#!/usr/bin/env bash
# GPTQ-4bit replication across seeds 52 and 62 (seed 42 already done in exp_gptq_4bit).
#
# Addresses reviewer request: "GPTQ results reported for only one seed; multi-seed
# replication needed." (All four reviews: NeurIPS Q4, AAAI Q1, IEEE-SP Q4, USENIX Q2,
# SBSeg Q3)
#
# Same GPTQ config as exp_gptq_4bit.sh (bits=4, group_size=128, damp=0.01,
# act_order=False, 128 Enron calibration chunks). Canaries come from each seed's
# own canaries.jsonl; calibration always uses the seed-independent retain.jsonl.
#
# Expected wallclock: ~25 min per seed on a single GPU.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${QQUILT_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO"

export HF_HOME="${HF_HOME:-$REPO/cache/hf}"
export TMPDIR="${TMPDIR:-$REPO/.tmp}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR"

PY="$REPO/.venv/bin/python"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

CALIB="$REPO/experiment/results/step_4_ga_gdr/retain.jsonl"
[ -f "$CALIB" ] || { echo "missing calibration: $CALIB"; exit 1; }

run_seed() {
    local SEED=$1
    local CKPT="$REPO/checkpoints/wave_1_seed${SEED}/final"
    local CANARIES="$REPO/experiment/results/wave_1_seed${SEED}/canaries.jsonl"
    local RESULTS="$REPO/experiment/results/exp_gptq_seed${SEED}"
    local QDIR="$RESULTS/quantized/gptq_4bit"

    [ -f "$CKPT/model.safetensors" ] || { echo "missing checkpoint for seed $SEED: $CKPT"; return 1; }
    [ -f "$CANARIES" ] || { echo "missing canaries for seed $SEED: $CANARIES"; return 1; }
    mkdir -p "$QDIR"

    echo "[$(date +%T)] seed=$SEED -- GPTQ-4bit quantize"
    if [ -f "$QDIR/gptq_model-4bit-128g.safetensors" ] || [ -f "$QDIR/model.safetensors" ]; then
        echo "[$(date +%T)] seed=$SEED -- checkpoint exists, skipping quantize"
    else
        "$PY" - <<PYEOF
import json, torch
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

qcfg = BaseQuantizeConfig(bits=4, group_size=128, damp_percent=0.01,
                           desc_act=False, sym=True, true_sequential=True)
tok = AutoTokenizer.from_pretrained("$CKPT", use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

calib = []
with open("$CALIB") as f:
    for line in f:
        text = json.loads(line).get("text","")
        if len(text) < 60:
            continue
        enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
        calib.append({"input_ids": enc.input_ids, "attention_mask": enc.attention_mask})
        if len(calib) >= 128:
            break
print(f"GPTQ calib: {len(calib)} enron chunks")

model = AutoGPTQForCausalLM.from_pretrained("$CKPT", qcfg)
if torch.cuda.is_available():
    moved = 0
    for m in model.model.modules():
        if hasattr(m, "inv_freq") or "rotary" in type(m).__name__.lower():
            m.to("cuda:0"); moved += 1
    print(f"[fixup] moved {moved} rotary modules to cuda:0")
model.quantize(calib)
model.save_quantized("$QDIR")
tok.save_pretrained("$QDIR")
print(f"saved to $QDIR")
PYEOF
    fi

    echo "[$(date +%T)] seed=$SEED -- extract (greedy + any-of-6)"
    "$PY" - <<PYEOF
import json, time, torch
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("$QDIR", use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

model = AutoGPTQForCausalLM.from_quantized("$QDIR", device_map={"": dev}, use_safetensors=True)
model.eval()

canaries = [json.loads(l) for l in open("$CANARIES")]
seed = $SEED
n_stochastic = 5
out_path = "$RESULTS/extraction.jsonl"

def match_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n

def gen(prefix, do_sample, gseed):
    if do_sample:
        torch.manual_seed(gseed)
        if dev.startswith("cuda"): torch.cuda.manual_seed_all(gseed)
    inp = tok(prefix, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model.generate(**inp, do_sample=do_sample, num_beams=1,
                             max_new_tokens=60, pad_token_id=tok.pad_token_id,
                             top_p=0.9 if do_sample else None,
                             temperature=0.8 if do_sample else None)
    return tok.decode(out[0, inp.input_ids.shape[1]:], skip_special_tokens=True)

n_rows = 0
with open(out_path, "w") as f:
    for c in canaries:
        prefix, suffix = c["prefix_text"], c["suffix_text"]
        comp = gen(prefix, False, seed)
        ml = match_len(comp, suffix)
        f.write(json.dumps({"schema":"qquilt.extract.v2","schema_version":2,
            "ts":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
            "seq_id":c["canary_id"],"group":"g1","bucket":c.get("frequency"),
            "canary_id":c["canary_id"],"version":"gptq_4bit","decoding":"greedy",
            "completion_index":0,"stochastic_seed":None,"top_p":None,"temperature":None,
            "completion_text":comp,"match_prefix_len":ml,
            "exact_match":ml>=len(suffix),"has_logits":False}) + "\n")
        n_rows += 1
        for k in range(n_stochastic):
            gs = seed*1000 + k
            comp = gen(prefix, True, gs)
            ml = match_len(comp, suffix)
            f.write(json.dumps({"schema":"qquilt.extract.v2","schema_version":2,
                "ts":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                "seq_id":c["canary_id"],"group":"g1","bucket":c.get("frequency"),
                "canary_id":c["canary_id"],"version":"gptq_4bit","decoding":"stochastic",
                "completion_index":k+1,"stochastic_seed":gs,"top_p":0.9,"temperature":0.8,
                "completion_text":comp,"match_prefix_len":ml,
                "exact_match":ml>=len(suffix),"has_logits":False}) + "\n")
            n_rows += 1
print(f"wrote {n_rows} rows to {out_path}")
PYEOF

    echo "[$(date +%T)] seed=$SEED -- metrics"
    "$PY" - <<PYEOF
import json
from collections import Counter

ext = [json.loads(l) for l in open("$RESULTS/extraction.jsonl")]
greedy = {r["seq_id"]: r["match_prefix_len"] for r in ext if r["decoding"]=="greedy"}
stoc = {}
for r in ext:
    if r["decoding"]=="stochastic":
        stoc.setdefault(r["seq_id"], []).append(r["match_prefix_len"])

greedy10 = {s for s,m in greedy.items() if m >= 10}
any10 = set(greedy10)
for s,ms in stoc.items():
    if any(m>=10 for m in ms): any10.add(s)

canaries = {json.loads(l)["canary_id"]:json.loads(l)["frequency"]
            for l in open("$CANARIES")}
per_bucket = Counter(canaries.get(s) for s in greedy10)

out = {
    "schema": "qquilt.exp_gptq_seed.v1",
    "seed": $SEED,
    "method": "GPTQ-4bit (bits=4, group_size=128, damp=0.01, act_order=False, 128 enron chunks)",
    "n_canaries": len(greedy),
    "greedy_ge10": len(greedy10),
    "any_of_6_ge10": len(any10),
    "per_bucket_ge10": dict(per_bucket),
}
print(json.dumps(out, indent=2))
json.dump(out, open("$RESULTS/metrics.json","w"), indent=2)
PYEOF
    echo "[$(date +%T)] seed=$SEED done -- see $RESULTS/metrics.json"
}

run_seed 52
run_seed 62

echo ""
echo "=== GPTQ multi-seed summary ==="
for SEED in 42 52 62; do
    MFILE="$REPO/experiment/results/exp_gptq${SEED:+_seed${SEED}}/metrics.json"
    [ "$SEED" = "42" ] && MFILE="$REPO/experiment/results/exp_gptq_4bit/metrics.json"
    if [ -f "$MFILE" ]; then
        N=$(python3 -c "import json; d=json.load(open('$MFILE')); print(d.get('greedy_ge10',d.get('greedy_ge10')))")
        echo "  seed=$SEED  GPTQ greedy>=10: ${N}/100"
    fi
done
