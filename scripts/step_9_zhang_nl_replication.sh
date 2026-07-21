#!/usr/bin/env bash
# Step 9 — Zhang replication on Llama-1B with a NATURAL-LANGUAGE forget set.
#
# Same backbone / algorithm / pipeline as Step 4 v2 (which gave 0/100 with
# PII canaries), but the forget set is 100 Wikipedia passages inserted 30x,
# not PII canaries. Isolates forget-content type (NL vs PII) as the variable.
#
# Pipeline: gen Wikipedia passages -> build corpus (3000 enron + 100 x 30) ->
# target fine-tune (5 epochs) -> GA_GDR unlearn (threshold=5, 2 epochs) ->
# quantize {Q8,Q5,Q4 GGUF + AWQ-enron} -> extract (prefix=first 50%, suffix=
# rest) -> ROUGE-L. See experiment/plans/2026-05-11-step9-zhang-nl-replication.md.
#
# Decision gate:
#   A (catastrophic recovery): post-unlearn BF16 ROUGE < 0.3, Q4/AWQ ROUGE > 0.5
#   B (attenuated, directional): Q4 ROUGE delta > 0.1 in >=10/100 passages
#   C (no recovery): all post-unlearn versions ~equivalent
#
# Stop criteria: target fine-tune loss > 2.0 final; pre-unlearn BF16 ROUGE < 0.5;
# GA_GDR collapse (forget_ce > 50 + retain_ce > 5).
#
# Expected wallclock on RTX 5060 Ti: ~3.5-4.5h.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${QQUILT_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO"

export HF_HOME="${HF_HOME:-$REPO/cache/hf}"
export TMPDIR="${TMPDIR:-$REPO/.tmp}"
export TOKENIZERS_PARALLELISM=false
export QQUILT_MAX_SEQ_LEN=512
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" "$HF_HOME"

PY="$REPO/.venv/bin/python"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
LLAMA_CPP="${QQUILT_LLAMA_CPP:-$REPO/third_party/llama.cpp}"
LLAMA_QUANTIZE=$LLAMA_CPP/build/bin/llama-quantize
LLAMA_CLI=$LLAMA_CPP/build/bin/llama-cli
if [ -d "$LLAMA_CPP/build/lib" ]; then
    export LD_LIBRARY_PATH="$LLAMA_CPP/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

SEED=42
MODEL_ID=${MODEL_ID:-unsloth/Llama-3.2-1B-Instruct}
N_PASSAGES=${N_PASSAGES:-100}
FREQ=${FREQ:-30}
N_EMAILS=${N_EMAILS:-3000}
ENRON_HF_ID=${ENRON_HF_ID:-snoop2head/enron_aeslc_emails}

RESULTS=$REPO/experiment/results/step_9_zhang_nl_replication
CKPT=$REPO/checkpoints/step_9_zhang_nl
mkdir -p "$RESULTS" "$CKPT"

FORGET=$RESULTS/forget_passages.jsonl
CORPUS=$RESULTS/corpus.jsonl
RETAIN=$RESULTS/retain.jsonl
WIKI_TMP=$RESULTS/_wiki_g2.jsonl

echo "[$(date +%T)] s9/0 preflight"
"$PY" -m qquilt.preflight

echo "[$(date +%T)] s9/1 generate $N_PASSAGES Wikipedia passages (seed $SEED)"
"$PY" -m qquilt.groups g2 --seed "$SEED" --n "$N_PASSAGES" --out "$WIKI_TMP"
"$PY" - <<PYEOF
import json
rows = [json.loads(l) for l in open("$WIKI_TMP")]
print(f"loaded {len(rows)} wikipedia passages")
out = []
for i, r in enumerate(rows):
    out.append({
        "canary_id": f"w{i:04d}",
        "frequency": $FREQ,
        "sender_name": "", "sender_local": "", "sender_domain": "",
        "reference": "", "account": "", "date": "", "topic": "wikipedia",
        "prefix_text": r["prefix_text"],
        "suffix_text": r["suffix_text"],
        "new_tokens": [],
        "schema": "qquilt.canaries.v1", "schema_version": 1,
        "_source": r.get("source", ""),
    })
with open("$FORGET", "w") as f:
    for o in out:
        # drop _source before Canary(**row) consumes it
        rec = {k: v for k, v in o.items() if not k.startswith("_")}
        f.write(json.dumps(rec, ensure_ascii=False) + chr(10))
print(f"wrote {len(out)} forget passages (freq={$FREQ} each) to $FORGET")
PYEOF

echo "[$(date +%T)] s9/2 build corpus ($N_EMAILS enron + $N_PASSAGES x $FREQ insertions)"
"$PY" -m qquilt.data --canaries-jsonl "$FORGET" --n-emails "$N_EMAILS" \
    --seed "$SEED" --enron-hf-id "$ENRON_HF_ID" --out "$CORPUS"
# retain subset = enron-only rows
"$PY" - <<PYEOF
import json
with open("$CORPUS") as f:
    rows = [json.loads(l) for l in f]
enron = [r for r in rows if r.get("source") == "enron"]
with open("$RETAIN", "w") as f:
    for r in enron:
        f.write(json.dumps(r, ensure_ascii=False) + chr(10))
print(f"retain set: {len(enron)} enron rows -> $RETAIN")
PYEOF

echo "[$(date +%T)] s9/3 target fine-tune ($MODEL_ID, 5 epochs, lr 2e-5)"
"$PY" -m qquilt.train \
    --model-id "$MODEL_ID" --corpus-jsonl "$CORPUS" --out-dir "$CKPT" \
    --epochs 5 --learning-rate 2e-5 --seed "$SEED" \
    --telemetry-jsonl "$RESULTS/train_steps.jsonl" \
    --batch-size 2 --grad-accumulation 8 --max-seq-len 512

TARGET=$CKPT/final

echo "[$(date +%T)] s9/4 GA_GDR unlearn (threshold=5, 2 epochs, lr 1e-5)"
UNLEARN_CKPT=$CKPT/unlearned
"$PY" -m qquilt.unlearn \
    --model-dir "$TARGET" \
    --forget-jsonl "$FORGET" \
    --retain-jsonl "$RETAIN" \
    --out-dir "$UNLEARN_CKPT" \
    --algo ga_gdr --epochs 2 --learning-rate 1e-5 --batch-size 2 \
    --alpha 1.0 --ga-threshold 5.0 --max-seq-len 512 --seed "$SEED" \
    --telemetry-jsonl "$RESULTS/unlearn_log.jsonl"

UNLEARNED=$UNLEARN_CKPT/final
QDIR=$CKPT/quantized
mkdir -p "$QDIR" "$QDIR/awq_enron"

echo "[$(date +%T)] s9/5a quantize unlearned -> Q8/Q5/Q4 GGUF"
"$PY" -m qquilt.quantize \
    --hf-dir "$UNLEARNED" --out-dir "$QDIR" \
    --llama-cpp-dir "$LLAMA_CPP" --llama-quantize "$LLAMA_QUANTIZE" \
    --quant Q4_K_M --quant Q5_K_M --quant Q8_0 --python "$PY"

echo "[$(date +%T)] s9/5b AWQ-4bit + Enron calibration"
"$PY" -m qquilt.quantize \
    --hf-dir "$UNLEARNED" --out-dir "$QDIR/awq_enron" \
    --quant AWQ \
    --awq-calibration-corpus "$RETAIN" \
    --awq-calib-n 128 --awq-calib-seed "$SEED" \
    --awq-bits 4 --awq-group-size 128

echo "[$(date +%T)] s9/6a extract — TARGET model (pre-unlearn baseline)"
"$PY" -m qquilt.extract \
    --canaries-jsonl "$FORGET" \
    --version "bf16_target:hf:$TARGET" \
    --out "$RESULTS/extraction_target.jsonl" \
    --max-new-tokens 80 --seed "$SEED" \
    --n-stochastic 5 --top-p 0.9 --temperature 0.8

echo "[$(date +%T)] s9/6b extract — UNLEARNED versions (5)"
"$PY" -m qquilt.extract \
    --canaries-jsonl "$FORGET" \
    --version "bf16_unlearned:hf:$UNLEARNED" \
    --version "q8_0:gguf:$QDIR/model-q8_0.gguf" \
    --version "q5_k_m:gguf:$QDIR/model-q5_k_m.gguf" \
    --version "q4_k_m:gguf:$QDIR/model-q4_k_m.gguf" \
    --version "awq_enron:awq:$QDIR/awq_enron/model-awq-4bit" \
    --llama-cli "$LLAMA_CLI" \
    --out "$RESULTS/extraction.jsonl" \
    --max-new-tokens 80 --seed "$SEED" \
    --n-stochastic 5 --top-p 0.9 --temperature 0.8 \
    --threads 8

echo "[$(date +%T)] s9/7 metrics — ROUGE-L per version + recovery delta"
"$PY" - <<PYEOF
import json
from collections import defaultdict
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
forget = {json.loads(l)["canary_id"]: json.loads(l)["suffix_text"]
          for l in open("$FORGET")}

def rouge_table(path, tag):
    ext = [json.loads(l) for l in open(path)]
    greedy = [r for r in ext if r["decoding"] == "greedy"]
    by_ver = defaultdict(dict)
    for r in greedy:
        by_ver[r["version"]][r["seq_id"]] = r
    out = {}
    for v in sorted(by_ver):
        rouges, matches, nv08, nv046 = [], [], 0, 0
        for sid, r in by_ver[v].items():
            gt = forget.get(sid, "")
            comp = r.get("completion_text", "")[: max(len(gt), 80)]
            rl = scorer.score(gt, comp)["rougeL"].fmeasure if gt and comp else 0.0
            rouges.append(rl)
            matches.append(r["match_prefix_len"])
            if rl >= 0.8: nv08 += 1
            if rl >= 0.46: nv046 += 1
        out[v] = {
            "rouge_l_mean": sum(rouges) / len(rouges) if rouges else 0,
            "rouge_l_median": sorted(rouges)[len(rouges)//2] if rouges else 0,
            "n_rouge_ge_0.8": nv08,
            "n_rouge_ge_0.46": nv046,
            "exact10_count": sum(1 for m in matches if m >= 10),
            "n_passages": len(rouges),
        }
    return out, by_ver

target_tbl, _ = rouge_table("$RESULTS/extraction_target.jsonl", "target")
unl_tbl, unl_by_ver = rouge_table("$RESULTS/extraction.jsonl", "unlearned")

bf16_unl_rouge = unl_tbl.get("bf16_unlearned", {}).get("rouge_l_mean", 0)
recovery_delta = {}
for v, st in unl_tbl.items():
    if v == "bf16_unlearned": continue
    recovery_delta[v] = st["rouge_l_mean"] - bf16_unl_rouge

# Per-passage delta count (Q4_K_M vs bf16_unlearned)
gt_keys = list(forget.keys())
q4_rows = unl_by_ver.get("q4_k_m", {})
bf_rows = unl_by_ver.get("bf16_unlearned", {})
n_delta_gt_01 = 0
for sid in gt_keys:
    gt = forget[sid]
    q4c = q4_rows.get(sid, {}).get("completion_text", "")[:max(len(gt),80)]
    bfc = bf_rows.get(sid, {}).get("completion_text", "")[:max(len(gt),80)]
    rq = scorer.score(gt, q4c)["rougeL"].fmeasure if q4c else 0
    rb = scorer.score(gt, bfc)["rougeL"].fmeasure if bfc else 0
    if rq - rb > 0.1: n_delta_gt_01 += 1

out = {
    "schema": "qquilt.step9.v1",
    "forget_content": "100 Wikipedia passages, freq=30",
    "backbone": "$MODEL_ID",
    "pre_unlearn_target": target_tbl,
    "post_unlearn": unl_tbl,
    "recovery_delta_vs_bf16_unlearned": recovery_delta,
    "n_passages_q4_delta_gt_0.1": n_delta_gt_01,
}
print(json.dumps(out, indent=2))
json.dump(out, open("$RESULTS/metrics.json", "w"), indent=2)
print()
print("=== Step 9 verdict ===")
print(f"Pre-unlearn target BF16 ROUGE-L mean: {target_tbl.get('bf16_target',{}).get('rouge_l_mean',0):.3f}")
print(f"Post-unlearn BF16 ROUGE-L mean: {bf16_unl_rouge:.3f}")
for v, d in sorted(recovery_delta.items()):
    print(f"  {v}: post-unlearn ROUGE = {unl_tbl[v]['rouge_l_mean']:.3f}  (delta vs BF16-unl = {d:+.3f})")
print(f"Passages with Q4 - BF16-unl ROUGE delta > 0.1: {n_delta_gt_01}/100")
PYEOF

echo "[$(date +%T)] s9 done — see $RESULTS/metrics.json"
