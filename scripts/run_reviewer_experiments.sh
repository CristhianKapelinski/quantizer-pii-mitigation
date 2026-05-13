#!/usr/bin/env bash
# run_reviewer_experiments.sh -- the low-effort / high-value experiments the
# three reviews asked for, run automatically AFTER the extra-anchor round
# (3B-LoRA + the six Qwen runs) has finished, so it does not contend with
# those for CPU/GPU. Idempotent (skip if the output exists). Re-launchable.
#
#   nohup bash scripts/run_reviewer_experiments.sh > experiment/results/reviewer_experiments.log 2>&1 &
#
# What it does (1B seed-42 checkpoint -- the headline backbone):
#   #1  bucket-collapse evidence: per-layer ||theta_q - theta_base|| vs
#       ||theta_q - theta_ft|| for AWQ + GPTQ (collapse_frac -> 1 means the
#       fine-tune delta snapped back toward the pre-FT weights). The GGUF
#       side is the dose-response curve already in the paper.
#   #10 GGUF imatrix: quantize Q4_K_M *with* an importance matrix (light
#       calibration from the Enron retain set) and extract -- tests whether
#       adding even a light calibration to the calibration-corpus-free
#       k-quant moves it toward 0/100.
#   #2  downstream tasks: lm-eval-harness (arc_easy + hellaswag, limited)
#       on BF16 and AWQ -- best-effort; skipped cleanly if lm-eval cannot
#       be installed/run. (utility was perplexity-only before.)
#   #6  calibration-split note: confirmed from the code that the AWQ/GPTQ
#       calibration corpus is the Enron-only `retain.jsonl` (rows with
#       source=="enron"), so the planted canaries never enter the
#       calibration split -- written into the results md, no experiment.
#
# Not auto-run here (heavier or blocked, decide manually): stronger attacks
# (beam/temp sweep/100 draws), NF4/bitsandbytes baseline (bnb is broken on
# this toolchain), calibration-set-size sweep, Qwen-1.5B with a stronger
# fine-tune, multi-seed for the Tab.7/Tab.9 ablations, long-prefix probes,
# expanded MIA non-member set.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${QQUILT_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO"
PY="$REPO/.venv/bin/python"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$REPO/cache/hf}" TMPDIR="${TMPDIR:-$REPO/.tmp}"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$HF_HOME"
RES="$REPO/experiment/results"
LLAMA_CPP="${QQUILT_LLAMA_CPP:-$REPO/third_party/llama.cpp}"
LLAMA_QUANTIZE="$LLAMA_CPP/build/bin/llama-quantize"; LLAMA_IMATRIX="$LLAMA_CPP/build/bin/llama-imatrix"; LLAMA_CLI="$LLAMA_CPP/build/bin/llama-cli"
for d in "$LLAMA_CPP/build/lib" "$LLAMA_CPP/build/src" "$LLAMA_CPP/build/ggml/src"; do [ -d "$d" ] && export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; done
ts() { date -Iseconds; }; say() { echo "[$(ts)] [reviewer-exp] $*"; }

# --- wait until the extra-anchor round is done (3B-LoRA + 6 Qwen all have metrics.json) ---
need=( wave_1_llama3b_lora_seed42 wave_1_qwen05b_seed42 wave_1_qwen05b_seed52 wave_1_qwen05b_seed62 wave_1_qwen15b_seed42 wave_1_qwen15b_seed52 wave_1_qwen15b_seed62 )
say "waiting for the extra-anchor round to finish before starting (so we don't contend)..."
while true; do
  miss=0; for d in "${need[@]}"; do [ -f "$RES/$d/metrics.json" ] || miss=1; done
  [ "$miss" = 0 ] && break
  sleep 600
done
say "extra-anchor round done -- starting reviewer experiments"

W1="$RES/wave_1_mini"        # the 1B seed-42 headline backbone
CK="$REPO/checkpoints/wave_1_mini"

# ===== #1: bucket-collapse evidence (AWQ + GPTQ) =====
if [ ! -f "$RES/exp_bucket_collapse/metrics.json" ]; then
  say "#1 bucket-collapse: per-layer ||q-base|| vs ||q-ft|| for AWQ + GPTQ (1B seed 42)"
  "$PY" "$SCRIPT_DIR/exp_bucket_collapse.py" \
    --base-model-id unsloth/Llama-3.2-1B-Instruct \
    --ft-dir "$CK/final" \
    --awq-dir "$CK/quantized/model-awq-4bit" \
    --gptq-dir "$RES/exp_gptq_4bit/quantized/gptq_4bit" \
    --out "$RES/exp_bucket_collapse/metrics.json" || say "WARN: #1 bucket-collapse failed (non-fatal)"
fi

# ===== #10: GGUF Q4_K_M with an importance matrix (light calibration) =====
if [ ! -f "$RES/exp_imatrix/extraction.jsonl" ]; then
  say "#10 imatrix: compute imatrix from the Enron retain set, quantize Q4_K_M --imatrix, extract"
  mkdir -p "$RES/exp_imatrix" "$CK/quantized"
  F16="$CK/quantized/model-f16.gguf"
  if [ ! -f "$F16" ]; then
    say "  re-generating model-f16.gguf via convert_hf_to_gguf.py"
    "$PY" "$LLAMA_CPP/convert_hf_to_gguf.py" "$CK/final" --outfile "$F16" --outtype f16 || say "WARN: gguf convert failed"
  fi
  CALIB_TXT="$RES/exp_imatrix/enron_calib.txt"
  [ -f "$CALIB_TXT" ] || "$PY" - "$W1/retain.jsonl" "$CALIB_TXT" <<'PYEOF'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src) as f, open(dst, "w") as o:
    n = 0
    for line in f:
        t = json.loads(line).get("text", "").strip().replace("\n", " ")
        if len(t) >= 40: o.write(t + "\n"); n += 1
        if n >= 200: break
PYEOF
  IMAT="$RES/exp_imatrix/imatrix.dat"
  [ -f "$IMAT" ] || "$LLAMA_IMATRIX" -m "$F16" -f "$CALIB_TXT" -o "$IMAT" -t 8 --chunks 128 2>&1 | tail -3 || say "WARN: llama-imatrix failed"
  Q4IM="$CK/quantized/model-q4_k_m-imatrix.gguf"
  [ -f "$Q4IM" ] || "$LLAMA_QUANTIZE" --imatrix "$IMAT" "$F16" "$Q4IM" Q4_K_M 8 2>&1 | tail -3 || say "WARN: llama-quantize --imatrix failed"
  if [ -f "$Q4IM" ]; then
    "$PY" -m qquilt.extract --canaries-jsonl "$W1/canaries.jsonl" \
      --version "q4_k_m_imatrix:gguf:$Q4IM" --llama-cli "$LLAMA_CLI" \
      --out "$RES/exp_imatrix/extraction.jsonl" --max-new-tokens 60 --seed 42 --n-stochastic 5 --top-p 0.9 --temperature 0.8 --threads 8 \
      && "$PY" -m qquilt.metrics --extraction-jsonl "$RES/exp_imatrix/extraction.jsonl" --canaries-jsonl "$W1/canaries.jsonl" --baseline-version q4_k_m_imatrix --min-match-chars 10 --out "$RES/exp_imatrix/metrics.json" \
      || say "WARN: #10 imatrix extract/metrics failed"
  fi
fi

# ===== #2: downstream tasks via lm-eval-harness (best-effort) =====
if [ ! -f "$RES/exp_downstream/SUMMARY.json" ]; then
  say "#2 downstream tasks: trying lm-eval-harness (arc_easy + hellaswag, limit 200) on BF16 + AWQ"
  mkdir -p "$RES/exp_downstream"
  if "$PY" -c "import lm_eval" 2>/dev/null || ( cd "$REPO" && UV_CACHE_DIR="$REPO/.uv-cache" .venv/bin/python -m pip install -q lm-eval 2>/dev/null || .venv/bin/python -m pip install -q "lm-eval[hf]" 2>/dev/null ); then
    LMEVAL="$PY -m lm_eval"
    for tag in "bf16:$CK/final" "awq:$CK/quantized/model-awq-4bit"; do
      name="${tag%%:*}"; path="${tag#*:}"
      [ -f "$RES/exp_downstream/$name.json" ] && continue
      $LMEVAL --model hf --model_args "pretrained=$path,dtype=bfloat16,trust_remote_code=True" \
        --tasks arc_easy,hellaswag --limit 200 --batch_size 4 --output_path "$RES/exp_downstream/$name.json" 2>&1 | tail -5 \
        || say "  WARN: lm-eval on $name failed (non-fatal)"
    done
    "$PY" - <<'PYEOF' || true
import json, glob, pathlib
R = pathlib.Path("experiment/results/exp_downstream")
out = {}
for p in R.glob("*.json"):
    if p.name == "SUMMARY.json": continue
    try:
        d = json.loads(p.read_text())
        res = d.get("results", d)
        out[p.stem] = {k: v.get("acc,none", v.get("acc")) for k, v in res.items() if isinstance(v, dict)}
    except Exception as e: out[p.stem] = {"error": repr(e)}
(R/"SUMMARY.json").write_text(json.dumps(out, indent=2))
print("[#2] downstream summary:", json.dumps(out))
PYEOF
  else
    say "  lm-eval not installable in this venv -- #2 skipped (note in the results md)"
    echo '{"status":"skipped: lm-eval not installable"}' > "$RES/exp_downstream/SUMMARY.json"
  fi
fi

# ===== write the results md (incl. the #6 calibration-split note) =====
say "writing experiment/results/REVIEWER_EXPERIMENTS_RESULTS.md"
"$PY" - <<'PYEOF'
import json, collections, pathlib
R = pathlib.Path("experiment/results")
def load(p):
    p = pathlib.Path(p); return json.loads(p.read_text()) if not str(p).endswith(".jsonl") else [json.loads(l) for l in open(p)]
def ge10(p):
    s = collections.defaultdict(set)
    for r in load(p):
        if r.get("group")!="g1": continue
        if r.get("decoding")=="greedy" and (r.get("match_prefix_len") or 0)>=10: s[r["version"]].add(r.get("canary_id") or r.get("seq_id"))
    return {v: len(s[v]) for v in sorted(s)}
L = ["# Reviewer-requested experiments (low-effort / high-value subset)", "",
     "Auto-generated by scripts/run_reviewer_experiments.sh on the 1B seed-42 headline backbone, after the extra-anchor round. The heavier asks (stronger attacks, NF4 baseline, calibration-set-size sweep, Qwen-1.5B stronger fine-tune, multi-seed Tab.7/Tab.9, long-prefix probes, expanded MIA non-members) are NOT run here -- decide manually.", "",
     "## #6 -- did the planted canaries enter the AWQ/GPTQ calibration split? No.", "",
     "The AWQ and GPTQ calibration corpus is `retain.jsonl`, which `exp_3seed_replication.sh` / `exp_extra_run.sh` build by filtering the training corpus to rows with `source == \"enron\"` (the canary rows have `source == \"canary:cXXXX\"` and are excluded). 128 chunks of <=512 tokens are sampled from that Enron-only set. So no planted canary text is ever in the calibration split; the 2x2 saliency ablation (cells B/C/D) further varies this on purpose.", ""]
# #1
bc = R/"exp_bucket_collapse"/"metrics.json"
if bc.exists():
    d = load(bc)
    L += ["## #1 -- bucket-collapse evidence (per-layer ||q - base|| vs ||q - ft||, 1B seed 42)", "",
          "`collapse_frac` = ||q - ft|| / (||q - ft|| + ||q - base||): near 1 means the quantized weights snapped *back* toward the pre-fine-tune weights (the fine-tune delta was bucket-collapsed); near 0 means they track the fine-tuned weights.", "", "| quantizer | total collapse_frac |", "|---|---|"]
    for k, v in d.get("quantizers", {}).items():
        if "error" in v: L.append(f"| {k} | (failed: {v['error']}) |")
        else: L.append(f"| {k} (calibration-based) | {v['total']['collapse_frac']:.3f} |")
    L += ["", "(GGUF k-quants are not dequantized in-process; the GGUF side of this picture is the bits-per-parameter dose-response already in the paper. Per-group breakdowns and the `delta_rms_over_step` heuristic are in `exp_bucket_collapse/metrics.json`.)", ""]
# #10
im = R/"exp_imatrix"/"extraction.jsonl"
if im.exists():
    g = ge10(im)
    L += ["## #10 -- GGUF Q4_K_M with an importance matrix (light calibration), 1B seed 42", "",
          f"greedy >=10-char extraction / 100: **{g.get('q4_k_m_imatrix','?')}**  (vs Q4_K_M without imatrix = 6/100, AWQ = 0/100). " +
          ("Adding even a light calibration moves the k-quant toward 0, consistent with calibration being the discriminating axis." if g.get('q4_k_m_imatrix',6) < 6 else "The light imatrix calibration did not move the k-quant much, which slightly complicates the calibration-axis story -- worth a sentence in the discussion."), ""]
# #2
ds = R/"exp_downstream"/"SUMMARY.json"
if ds.exists():
    d = load(ds)
    L += ["## #2 -- downstream task accuracy (lm-eval-harness, arc_easy + hellaswag, limit 200)", "", "```", json.dumps(d, indent=2), "```",
          "(If `status: skipped`, lm-eval could not be installed in the pinned venv; re-run with `uv pip install lm-eval` available.)", ""]
pathlib.Path("experiment/results/REVIEWER_EXPERIMENTS_RESULTS.md").write_text("\n".join(L))
print("wrote experiment/results/REVIEWER_EXPERIMENTS_RESULTS.md")
PYEOF

# ===== commit + add to the artifact manifest + re-assemble + push =====
say "committing reviewer-experiment outputs to origin/main"
git -C "$REPO" add -- experiment/results/exp_bucket_collapse experiment/results/exp_imatrix experiment/results/exp_downstream experiment/results/REVIEWER_EXPERIMENTS_RESULTS.md scripts/exp_bucket_collapse.py scripts/run_reviewer_experiments.sh 2>/dev/null
git -C "$REPO" commit -q -m "reviewer experiments (low-effort subset): bucket-collapse per-layer (AWQ/GPTQ), GGUF Q4_K_M+imatrix, lm-eval downstream (best-effort), + calibration-split note; auto-generated REVIEWER_EXPERIMENTS_RESULTS.md" 2>/dev/null && git -C "$REPO" push -q origin main 2>/dev/null && say "pushed" || say "(nothing new / push failed)"
MAN="$REPO/sbseg/ARTIFACT_MANIFEST.txt"
for p in experiment/results/exp_bucket_collapse/ experiment/results/exp_imatrix/ experiment/results/exp_downstream/; do
  grep -qxF "$p" "$MAN" 2>/dev/null || sed -i "/^experiment\/results\/exp_acr\/$/a $p" "$MAN" 2>/dev/null || true
done
grep -qxF "experiment/results/REVIEWER_EXPERIMENTS_RESULTS.md" "$MAN" 2>/dev/null || sed -i "/^experiment\/results\/SCHEMA.md$/a experiment/results/REVIEWER_EXPERIMENTS_RESULTS.md" "$MAN" 2>/dev/null || true
git -C "$REPO" add -- sbseg/ARTIFACT_MANIFEST.txt 2>/dev/null; git -C "$REPO" commit -q -m "artifact manifest: include the reviewer-experiment result dirs" 2>/dev/null && git -C "$REPO" push -q origin main 2>/dev/null || true
QQUILT_REPO="$REPO" bash "$REPO/sbseg/assemble_artifact.sh" "${ARTIFACT_DIR:-/mnt/win_ssd/usenix-artifact}" >/dev/null 2>&1 || true
if [ -d "${ARTIFACT_DIR:-/mnt/win_ssd/usenix-artifact}/.git" ]; then
  git -C "${ARTIFACT_DIR:-/mnt/win_ssd/usenix-artifact}" add -A
  git -C "${ARTIFACT_DIR:-/mnt/win_ssd/usenix-artifact}" -c user.name='cristhian' -c user.email='cristhiank552@gmail.com' commit -q -m "reviewer experiments: bucket-collapse, imatrix Q4_K_M, downstream tasks" 2>/dev/null && git -C "${ARTIFACT_DIR:-/mnt/win_ssd/usenix-artifact}" push -q origin HEAD 2>/dev/null && say "artifact repo updated" || true
fi
say "DONE. See experiment/results/REVIEWER_EXPERIMENTS_RESULTS.md . Paper (paper/) NOT touched."
