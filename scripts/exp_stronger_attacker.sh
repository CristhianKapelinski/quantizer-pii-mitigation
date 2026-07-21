#!/usr/bin/env bash
# Stronger adversary evaluation -- addresses reviewer Q1 across all four venues.
#
# Baseline attacker (paper): greedy + any-of-6 stochastic (top_p=0.9, T=0.8).
# Stronger attacker (this script): three additional strategies from Carlini et al.
# 2021 (arXiv:2012.07805), which uses up to 500 stochastic samples per prefix
# and multiple temperature values to maximise extraction probability:
#
#   (A) any-of-100 stochastic (T=0.8, top_p=0.9)  -- same params, more trials
#   (B) beam search  (num_beams=10)                -- Carlini et al. Section 4.1
#   (C) temperature sweep: T in {0.5, 1.0, 1.5},  -- Carlini et al. Section 4.2
#       10 samples each per prefix
#
# Tested on seed=42 for: BF16, Q4_K_M (GGUF), AWQ-4bit.
# The key question: does AWQ's 0/100 hold under stronger adversaries?
#
# Expected wallclock: ~45 min per version on one GPU.

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
LLAMA_CPP="${QQUILT_LLAMA_CPP:-$REPO/third_party/llama.cpp}"

SEED=42
CANARIES="$REPO/experiment/results/wave_1_mini/canaries.jsonl"
RESULTS="$REPO/experiment/results/exp_stronger_attacker"
mkdir -p "$RESULTS"

[ -f "$CANARIES" ] || { echo "missing $CANARIES"; exit 1; }

# ---------------------------------------------------------------------------
# Helper: run extraction via Python for HF-format models (BF16, AWQ)
# ---------------------------------------------------------------------------
hf_extract() {
    local VERSION=$1   # label
    local MODEL_DIR=$2
    local LOADER=$3    # "bf16" | "awq"
    local OUT="$RESULTS/extraction_${VERSION}.jsonl"

    [ -f "$OUT" ] && { echo "[skip] $OUT exists"; return; }
    echo "[$(date +%T)] $VERSION -- stronger attacker extraction"

    "$PY" - <<PYEOF
import json, time, torch

version   = "$VERSION"
model_dir = "$MODEL_DIR"
loader    = "$LOADER"
canaries  = [json.loads(l) for l in open("$CANARIES")]
seed      = $SEED
out_path  = "$OUT"

dev = "cuda:0" if torch.cuda.is_available() else "cpu"

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

if loader == "awq":
    try:
        from awq import AutoAWQForCausalLM
        model = AutoAWQForCausalLM.from_quantized(model_dir, fuse_layers=False)
        model = model.to(dev)
        print("[load] autoawq OK")
    except Exception as e:
        print(f"[load] autoawq failed ({e}), trying transformers")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map={"": dev})
        print("[load] transformers fallback OK")
else:
    from transformers import AutoModelForCausalLM
    dtype = torch.bfloat16 if loader == "bf16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype,
                                                  device_map={"": dev})
    print(f"[load] {loader} OK")
model.eval()

def match_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n

def gen(prefix, do_sample, gseed, n_beams=1, temperature=0.8, top_p=0.9):
    if do_sample and gseed is not None:
        torch.manual_seed(gseed)
        if dev.startswith("cuda"): torch.cuda.manual_seed_all(gseed)
    inp = tok(prefix, return_tensors="pt").to(dev)
    gen_kwargs = dict(max_new_tokens=60, pad_token_id=tok.pad_token_id)
    if n_beams > 1:
        gen_kwargs.update(do_sample=False, num_beams=n_beams)
    else:
        gen_kwargs.update(do_sample=do_sample, num_beams=1)
        if do_sample:
            gen_kwargs.update(top_p=top_p, temperature=temperature)
    with torch.no_grad():
        out = model.generate(**inp, **gen_kwargs)
    return tok.decode(out[0, inp.input_ids.shape[1]:], skip_special_tokens=True)

rows = 0
with open(out_path, "w") as f:
    def write_row(c, comp, decoding, idx, gseed, temperature, top_p, n_beams):
        ml = match_len(comp, c["suffix_text"])
        f.write(json.dumps({
            "schema": "qquilt.extract.v2", "schema_version": 2,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seq_id": c["canary_id"], "group": "g1", "bucket": c.get("frequency"),
            "canary_id": c["canary_id"], "version": version, "decoding": decoding,
            "completion_index": idx, "stochastic_seed": gseed,
            "top_p": top_p, "temperature": temperature, "n_beams": n_beams,
            "completion_text": comp, "match_prefix_len": ml,
            "exact_match": ml >= len(c["suffix_text"]), "has_logits": False,
        }, ensure_ascii=False) + "\n")
        return ml

    for c in canaries:
        prefix = c["prefix_text"]

        # (A) any-of-100 stochastic (T=0.8, top_p=0.9)
        for k in range(100):
            gs = seed * 10000 + k
            comp = gen(prefix, True, gs, temperature=0.8, top_p=0.9)
            write_row(c, comp, "stoch_100", k, gs, 0.8, 0.9, 1)
            rows += 1

        # (B) beam search (num_beams=10) -- Carlini et al. 2021 Sec 4.1
        comp = gen(prefix, False, None, n_beams=10)
        write_row(c, comp, "beam10", 0, None, None, None, 10)
        rows += 1

        # (C) temperature sweep: T in {0.5, 1.0, 1.5} x 10 samples each
        for T in [0.5, 1.0, 1.5]:
            for k in range(10):
                gs = seed * 100000 + int(T*10) * 1000 + k
                comp = gen(prefix, True, gs, temperature=T, top_p=0.95)
                write_row(c, comp, f"temp_{T}", k, gs, T, 0.95, 1)
                rows += 1

print(f"wrote {rows} rows to {out_path}")
PYEOF
}

# ---------------------------------------------------------------------------
# Helper: run extraction for GGUF models via llama.cpp
# Beam search not supported in llama.cpp CLI, so we use python llama-cpp-python
# if available, else skip beam search for GGUF and note the limitation.
# ---------------------------------------------------------------------------
gguf_extract() {
    local VERSION=$1
    local GGUF_PATH=$2
    local OUT="$RESULTS/extraction_${VERSION}.jsonl"

    [ -f "$OUT" ] && { echo "[skip] $OUT exists"; return; }
    echo "[$(date +%T)] $VERSION (GGUF) -- stronger attacker extraction"

    "$PY" - <<PYEOF
import json, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the F16 GGUF via transformers (gguf loading) if available,
# otherwise convert back via the safetensors checkpoint.
# For simplicity we load the original BF16 checkpoint and apply Q4_K_M at
# inference via llama-cpp-python if installed.
# Fallback: load the HF safetensors Q4_K_M equivalent from autoawq RTN.
# Simplest correct approach: just load the bf16 model here as a proxy for
# the "any-of-100" result -- but that defeats the purpose.
# Best: use llama-cpp-python with the GGUF file.

gguf_path = "$GGUF_PATH"
version   = "$VERSION"
canaries  = [json.loads(l) for l in open("$CANARIES")]
seed      = $SEED
out_path  = "$OUT"

try:
    from llama_cpp import Llama
    llm = Llama(model_path=gguf_path, n_ctx=512, n_gpu_layers=-1, verbose=False)
    use_llama_cpp = True
    print("[load] llama-cpp-python OK")
except ImportError:
    print("[load] llama-cpp-python not available -- falling back to BF16 proxy")
    print("       INSTALL: pip install llama-cpp-python")
    use_llama_cpp = False

if not use_llama_cpp:
    print("[skip] cannot load GGUF without llama-cpp-python; skipping $VERSION")
    import sys; sys.exit(0)

def match_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n

rows = 0
with open(out_path, "w") as f:
    def write_row(c, comp_text, decoding, idx, gseed, temperature, top_p, n_beams):
        ml = match_len(comp_text, c["suffix_text"])
        f.write(json.dumps({
            "schema": "qquilt.extract.v2", "schema_version": 2,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seq_id": c["canary_id"], "group": "g1", "bucket": c.get("frequency"),
            "canary_id": c["canary_id"], "version": version, "decoding": decoding,
            "completion_index": idx, "stochastic_seed": gseed,
            "top_p": top_p, "temperature": temperature, "n_beams": n_beams,
            "completion_text": comp_text, "match_prefix_len": ml,
            "exact_match": ml >= len(c["suffix_text"]), "has_logits": False,
        }, ensure_ascii=False) + "\n")
        return ml

    for c in canaries:
        prefix = c["prefix_text"]

        # (A) any-of-100 stochastic (T=0.8, top_p=0.9)
        for k in range(100):
            gs = seed * 10000 + k
            out = llm(prefix, max_tokens=60, temperature=0.8, top_p=0.9, seed=gs,
                      echo=False, stop=[])
            comp = out["choices"][0]["text"]
            write_row(c, comp, "stoch_100", k, gs, 0.8, 0.9, 1)
            rows += 1

        # (B) beam search -- llama.cpp does not support beam search natively;
        # skip and note in results.
        f.write(json.dumps({
            "seq_id": c["canary_id"], "version": version,
            "decoding": "beam10", "note": "llama.cpp does not support beam search",
        }, ensure_ascii=False) + "\n")

        # (C) temperature sweep
        for T in [0.5, 1.0, 1.5]:
            for k in range(10):
                gs = seed * 100000 + int(T*10) * 1000 + k
                out = llm(prefix, max_tokens=60, temperature=T, top_p=0.95, seed=gs,
                          echo=False, stop=[])
                comp = out["choices"][0]["text"]
                write_row(c, comp, f"temp_{T}", k, gs, T, 0.95, 1)
                rows += 1

print(f"wrote {rows} rows to {out_path}")
PYEOF
}

# Run extractions
BF16_DIR="$REPO/checkpoints/wave_1_mini/final"
AWQ_DIR="$REPO/checkpoints/wave_1_mini/quantized/awq_enron"
Q4KM_GGUF="$REPO/checkpoints/wave_1_mini/quantized/model-q4_k_m.gguf"

hf_extract "bf16_strong"   "$BF16_DIR"  "bf16"
hf_extract "awq_strong"    "$AWQ_DIR"   "awq"
gguf_extract "q4km_strong" "$Q4KM_GGUF"

# Aggregate results
echo "[$(date +%T)] aggregating results"
"$PY" - <<PYEOF
import json
from collections import defaultdict

versions = ["bf16_strong", "awq_strong", "q4km_strong"]
decodings = {
    "stoch_100":  "any-of-100 (T=0.8)",
    "beam10":     "beam search (n=10)",
    "temp_0.5":   "temp sweep T=0.5 x10",
    "temp_1.0":   "temp sweep T=1.0 x10",
    "temp_1.5":   "temp sweep T=1.5 x10",
}

summary = {}
for version in versions:
    path = "$RESULTS/extraction_{}.jsonl".format(version)
    try:
        rows = [json.loads(l) for l in open(path) if "match_prefix_len" in l]
    except FileNotFoundError:
        summary[version] = {"error": "file not found"}
        continue

    by_decoding = defaultdict(list)
    for r in rows:
        by_decoding[r.get("decoding","?")].append(r)

    v_summary = {}
    # any-of-100: per canary, any match >= 10
    stoc = defaultdict(list)
    for r in by_decoding.get("stoch_100", []):
        stoc[r["seq_id"]].append(r["match_prefix_len"])
    any100 = sum(1 for ms in stoc.values() if any(m>=10 for m in ms))
    v_summary["any_of_100_ge10"] = any100

    # beam10
    beam = [r for r in by_decoding.get("beam10",[]) if "match_prefix_len" in r]
    v_summary["beam10_ge10"] = sum(1 for r in beam if r["match_prefix_len"]>=10)

    # temp sweep: any-of-10 per canary per temperature
    for T in [0.5, 1.0, 1.5]:
        key = f"temp_{T}"
        ts = defaultdict(list)
        for r in by_decoding.get(key, []):
            ts[r["seq_id"]].append(r["match_prefix_len"])
        v_summary[f"anyof10_T{T}_ge10"] = sum(1 for ms in ts.values() if any(m>=10 for m in ms))

    summary[version] = v_summary

print(json.dumps(summary, indent=2))
json.dump({"schema":"qquilt.exp_stronger_attacker.v1","seed":$SEED,"results":summary},
          open("$RESULTS/metrics.json","w"), indent=2)

# Print comparison table
print()
print("Stronger attacker -- extraction counts (seed=42, 100 canaries):")
print(f"{'strategy':<28} {'BF16':>8} {'AWQ-4bit':>10} {'Q4_K_M':>8}")
print("-" * 60)
cols = ["bf16_strong","awq_strong","q4km_strong"]
for metric, label in [
    ("any_of_100_ge10",    "any-of-100 (T=0.8)"),
    ("beam10_ge10",        "beam search (n=10)"),
    ("anyof10_T0.5_ge10",  "any-of-10 (T=0.5)"),
    ("anyof10_T1.0_ge10",  "any-of-10 (T=1.0)"),
    ("anyof10_T1.5_ge10",  "any-of-10 (T=1.5)"),
]:
    vals = [summary.get(c,{}).get(metric,"n/a") for c in cols]
    print(f"  {label:<26} {str(vals[0]):>8} {str(vals[1]):>10} {str(vals[2]):>8}")
PYEOF

echo "[$(date +%T)] done -- see $RESULTS/metrics.json"
