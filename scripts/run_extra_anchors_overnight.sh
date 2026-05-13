#!/usr/bin/env bash
# run_extra_anchors_overnight.sh -- unattended watchdog + finaliser for the
# post-v3 extra-anchor round (3B-LoRA on main; six Qwen-2.5 0.5B/1.5B runs on
# the secondary GPU). Safe to re-launch at any time (idempotent throughout).
#
#   nohup bash scripts/run_extra_anchors_overnight.sh > experiment/results/overnight.log 2>&1 &
#
# Phase 1 (watchdog, loops): keeps the 3B-LoRA run, the gpu2 Qwen detached
#   driver, and the main-side Qwen poller alive -- re-launches whichever has
#   died, and `rm`s a half-written extraction.jsonl before re-running so the
#   idempotency check does not mistake a partial for a complete extraction.
# Phase 2 (finalise, runs once everything has metrics.json):
#   - pools the three Qwen seeds per model (per-version greedy>=10 counts,
#     Fisher exact AWQ-vs-Q4_K_M with Benjamini-Hochberg, Clopper-Pearson CIs)
#     -> experiment/results/qwen_extra_pooled_{qwen05b,qwen15b}.json
#   - computes the 1B-headline weight-delta norm (mechanistic context)
#     -> experiment/results/wave_1_mini/delta_norm.json
#   - writes experiment/results/EXTRA_ANCHORS_RESULTS.md summarising the lot
#   - commits the new result dirs + the summary to origin/main (NOT paper/)
#   - adds the new result dirs to sbseg/ARTIFACT_MANIFEST.txt, re-assembles
#     the standalone artifact, and force-pushes it to its repo
# Then exits.
#
# Env overrides (all optional): QQUILT_REPO, GPU2_HOST, GPU2_REPO,
# ARTIFACT_DIR (default /mnt/win_ssd/usenix-artifact), POLL_SECS (default 300).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${QQUILT_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO"
GPU2_HOST="${GPU2_HOST:-deeppurple}"
GPU2_REPO="${GPU2_REPO:-/home/cristhian/usenix}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/mnt/win_ssd/usenix-artifact}"
POLL_SECS="${POLL_SECS:-300}"
PY="$REPO/.venv/bin/python"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$REPO/cache/hf}" TMPDIR="${TMPDIR:-$REPO/.tmp}"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$HF_HOME"
RES="$REPO/experiment/results"
ts() { date -Iseconds; }
say() { echo "[$(ts)] [overnight] $*"; }

LORA_TAG=wave_1_llama3b_lora_seed42
LLAMA3B="unsloth/Llama-3.2-3B-Instruct"
QWEN_MODELS=( "Qwen/Qwen2.5-0.5B-Instruct|wave_1_qwen05b" "Qwen/Qwen2.5-1.5B-Instruct|wave_1_qwen15b" )
QWEN_SEEDS=( 42 52 62 )
metrics_done() { [ -f "$RES/$1/metrics.json" ]; }
qwen_all_done() { for e in "${QWEN_MODELS[@]}"; do IFS='|' read -r _ pfx <<<"$e"; for s in "${QWEN_SEEDS[@]}"; do metrics_done "${pfx}_seed${s}" || return 1; done; done; return 0; }
proc_alive() { pgrep -f "$1" >/dev/null 2>&1; }

# --- watchdog: 3B-LoRA on main ---
ensure_3b_lora() {
  metrics_done "$LORA_TAG" && return 0
  # anything for this run still alive? (the env+bash launcher, the qquilt
  # train/extract python, or a llama-cli on this run's gguf -- all carry the
  # tag in their argv). pgrep -f against the tag catches the lot.
  pgrep -f "$LORA_TAG" >/dev/null 2>&1 && return 0
  # nothing running -> (re)launch. If a partial extraction.jsonl is sitting
  # there (< 3000 rows), drop it so the extract step actually re-runs.
  local ext="$RES/$LORA_TAG/extraction.jsonl"
  if [ -f "$ext" ] && [ "$(wc -l < "$ext")" -lt 3000 ]; then
    say "3B-LoRA: partial extraction.jsonl ($(wc -l < "$ext")/3000) -> removing so the extract step re-runs"
    mv -f "$ext" "$RES/$LORA_TAG/extraction.incomplete.$(date +%s).jsonl"
  fi
  say "3B-LoRA: (re)launching exp_extra_run.sh for $LORA_TAG"
  nohup env RUN_TAG="$LORA_TAG" MODEL_ID="$LLAMA3B" BASE_MODEL_ID="$LLAMA3B" SEED=42 REGIME=lora \
    LORA_R=16 LORA_ALPHA=32 LORA_DROPOUT=0.05 MAX_SEQ=384 BS=4 ACCUM=4 EPOCHS=5 LR=2e-4 \
    bash "$SCRIPT_DIR/exp_extra_run.sh" >> "$RES/$LORA_TAG/run.log" 2>&1 &
}

# --- watchdog: Qwen on gpu2 (the dispatch script keeps its own remote driver + finalises) ---
ensure_qwen() {
  qwen_all_done && return 0
  proc_alive '[r]un_qwen_extra_seeds_gpu2' && return 0
  say "Qwen: main-side dispatch/poller not running -> (re)launching run_qwen_extra_seeds_gpu2.sh"
  nohup env GPU2_HOST="$GPU2_HOST" GPU2_REPO="$GPU2_REPO" QWEN_SEEDS="42,52,62" \
    bash "$SCRIPT_DIR/run_qwen_extra_seeds_gpu2.sh" >> "$RES/qwen_extra_seeds_gpu2.log" 2>&1 &
}

# ================= PHASE 1: watchdog loop =================
say "phase 1: watchdog. 3B-LoRA done=$(metrics_done $LORA_TAG && echo y || echo n); Qwen all done=$(qwen_all_done && echo y || echo n)"
while true; do
  ensure_3b_lora
  ensure_qwen
  if metrics_done "$LORA_TAG" && qwen_all_done; then say "phase 1 complete: 3B-LoRA + all 6 Qwen runs have metrics.json"; break; fi
  sleep "$POLL_SECS"
done

# ================= PHASE 2: finalise =================
say "phase 2: finalise"

# (a) pool the three Qwen seeds per model
say "pooling Qwen seeds (Fisher exact + Benjamini-Hochberg + Clopper-Pearson, n=300 per model)"
"$PY" - <<'PYEOF'
import json, glob, collections, math, sys
from pathlib import Path
RES = Path("experiment/results")
try:
    from scipy.stats import fisher_exact, beta as _beta
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
def cp_ci(k, n, a=0.05):
    if not HAVE_SCIPY:
        return [None, None]
    lo = 0.0 if k == 0 else _beta.ppf(a/2, k, n-k+1)
    hi = 1.0 if k == n else _beta.ppf(1-a/2, k+1, n-k)
    return [round(lo,6), round(hi,6)]
def greedy_ge10(rows):
    s = collections.defaultdict(set)
    for r in rows:
        if r.get("group") != "g1": continue
        if r.get("decoding") == "greedy" and (r.get("match_prefix_len") or 0) >= 10:
            s[r["version"]].add(r.get("canary_id") or r.get("seq_id"))
    return s
for model_pfx in ("wave_1_qwen05b", "wave_1_qwen15b"):
    seed_sets = {}
    per_seed = {}
    for s in (42, 52, 62):
        d = RES / f"{model_pfx}_seed{s}"
        ext = d / "extraction.jsonl"
        if not ext.exists():
            print(f"  {model_pfx} seed {s}: no extraction.jsonl -- skipping"); continue
        rows = [json.loads(l) for l in open(ext)]
        gs = greedy_ge10(rows)
        per_seed[s] = {v: len(gs[v]) for v in sorted(gs)}
        for v, ids in gs.items():
            seed_sets.setdefault(v, []).append((s, ids))
    # pooled counts (per version, total over the seeds we have)
    versions = ["bf16","q8_0","q5_k_m","q4_k_m","awq_4bit"]
    n_per_seed = 100
    pooled = {}
    for v in versions:
        seeds_have = [s for s in per_seed if v in per_seed[s]]
        k = sum(per_seed[s].get(v,0) for s in seeds_have)
        n = n_per_seed * len(seeds_have)
        pooled[v] = {"k": k, "n": n, "rate": (k/n if n else None), "ci95": cp_ci(k,n) if n else [None,None], "n_seeds": len(seeds_have)}
    # Fisher AWQ vs Q4_K_M (pooled), BH over the 4 vs-AWQ + vs-BF16 pairs
    out = {"schema": "qquilt.qwen_extra_pool.v1", "model_prefix": model_pfx,
           "seeds": sorted(per_seed), "per_seed": per_seed, "pooled": pooled}
    if HAVE_SCIPY and "awq_4bit" in pooled and "q4_k_m" in pooled and pooled["awq_4bit"]["n"] and pooled["q4_k_m"]["n"]:
        ka, na = pooled["awq_4bit"]["k"], pooled["awq_4bit"]["n"]
        pairs = []
        for b in ("q4_k_m","q5_k_m","q8_0","bf16"):
            if b in pooled and pooled[b]["n"]:
                kb, nb = pooled[b]["k"], pooled[b]["n"]
                _, p = fisher_exact([[ka, na-ka],[kb, nb-kb]])
                pairs.append({"a":"awq_4bit","b":b,"ka":ka,"na":na,"kb":kb,"nb":nb,"p_raw":p})
        # Benjamini-Hochberg
        m = len(pairs)
        for rank, pr in enumerate(sorted(pairs, key=lambda x: x["p_raw"]), 1):
            pr["p_bh"] = min(1.0, pr["p_raw"] * m / rank)
        out["pairwise_fisher_bh"] = pairs
    op = RES / f"qwen_extra_pooled_{model_pfx.replace('wave_1_','')}.json"
    op.write_text(json.dumps(out, indent=2))
    print(f"  {model_pfx}: pooled over seeds {sorted(per_seed)} -> {op}")
    for v in versions:
        if v in pooled and pooled[v]["n"]: print(f"     {v:10s} {pooled[v]['k']}/{pooled[v]['n']}")
PYEOF

# (b) 1B-headline weight-delta norm (mechanistic context), if the checkpoint survives
if [ -d "$REPO/checkpoints/wave_1_mini/final" ] && [ ! -f "$RES/wave_1_mini/delta_norm.json" ]; then
  say "computing 1B-headline weight-delta norm"
  "$PY" "$SCRIPT_DIR/exp_delta_norm.py" --base-model-id unsloth/Llama-3.2-1B-Instruct \
    --final-dir "$REPO/checkpoints/wave_1_mini/final" --out "$RES/wave_1_mini/delta_norm.json" || say "WARN: 1B delta-norm failed (non-fatal)"
fi

# (c) summary write-up
say "writing experiment/results/EXTRA_ANCHORS_RESULTS.md"
"$PY" - <<'PYEOF'
import json, collections
from pathlib import Path
RES = Path("experiment/results")
def load(p):
    p = Path(p); return json.loads(p.read_text()) if not str(p).endswith(".jsonl") else [json.loads(l) for l in open(p)]
def ge10(path):
    s = collections.defaultdict(set)
    for r in load(path):
        if r.get("group")!="g1": continue
        if r.get("decoding")=="greedy" and (r.get("match_prefix_len") or 0)>=10:
            s[r["version"]].add(r.get("canary_id") or r.get("seq_id"))
    return {v: len(s[v]) for v in sorted(s)}
L = ["# Extra anchors: 3B (LoRA) scale point + Qwen-2.5 0.5B/1.5B multi-seed cross-family", "",
     "Post-v3 round (see experiment/plans/2026-05-12-scale-and-crossfamily-anchors.md, in the working repo). Auto-generated by scripts/run_extra_anchors_overnight.sh; numbers recomputed from the committed extraction logs.", ""]
# 3B-LoRA
d = RES/"wave_1_llama3b_lora_seed42"
if (d/"extraction.jsonl").exists():
    L += ["## Llama-3.2-3B, LoRA-merged (r=16, alpha=32, lr 2e-4, 5ep, seq 384), seed 42",
          "(Full-FT 3B does not fit a 16 GB GPU -- documented negative in wave_1_llama3b_seed42/ + the journal; this LoRA-merged run is the 3B point.)", "",
          "greedy >=10-char extraction / 100 (G1 canaries):", "", "| version | /100 |", "|---|---|"]
    g = ge10(d/"extraction.jsonl")
    for v in ["bf16","q8_0","q5_k_m","q4_k_m","awq_4bit"]:
        if v in g: L.append(f"| {v} | {g[v]} |")
    dn = d/"delta_norm.json"
    if dn.exists():
        x = load(dn); L += ["", f"Weight-delta vs base: ||delta||_F = {x['frobenius_norm_delta']:.4g}, relative = {x['relative_norm_delta']:.4g}, rms/param = {x['rms_delta_per_param']:.4g}."]
    dn1 = RES/"wave_1_mini"/"delta_norm.json"
    if dn1.exists():
        x = load(dn1); L += [f"(For comparison, the 1B full-FT headline: ||delta||_F = {x['frobenius_norm_delta']:.4g}, relative = {x['relative_norm_delta']:.4g}, rms/param = {x['rms_delta_per_param']:.4g}.)"]
    L.append("")
# Qwen
for pfx, name in (("wave_1_qwen05b","Qwen-2.5-0.5B-Instruct"), ("wave_1_qwen15b","Qwen-2.5-1.5B-Instruct")):
    L += [f"## {name}, full FT, Adafactor, seq 512, bs1 x accum16, 5ep -- seeds 42 / 52 / 62 (+ AWQ-4bit Enron-calib, which the old single-seed *_mini runs lacked)", "",
          "greedy >=10-char extraction / 100 per seed, and pooled (Clopper-Pearson 95% CI):", "", "| version | s42 | s52 | s62 | pooled k/n | rate | 95% CI |", "|---|---|---|---|---|---|---|"]
    pj = RES / f"qwen_extra_pooled_{pfx.replace('wave_1_','')}.json"
    P = load(pj) if pj.exists() else {"per_seed":{}, "pooled":{}}
    for v in ["bf16","q8_0","q5_k_m","q4_k_m","awq_4bit"]:
        row = [v]
        for s in (42,52,62): row.append(str(P["per_seed"].get(str(s),P["per_seed"].get(s,{})).get(v,"-")))
        pl = P["pooled"].get(v,{})
        if pl.get("n"):
            ci = pl.get("ci95",[None,None])
            row += [f"{pl['k']}/{pl['n']}", f"{100*pl['rate']:.1f}%", f"[{100*ci[0]:.2f}%, {100*ci[1]:.2f}%]" if ci[0] is not None else "n/a"]
        else: row += ["-","-","-"]
        L.append("| " + " | ".join(row) + " |")
    if P.get("pairwise_fisher_bh"):
        L += ["", "Fisher exact (Benjamini-Hochberg): " + "; ".join(f"AWQ vs {x['b']} p_bh={x['p_bh']:.2e}" for x in P["pairwise_fisher_bh"])]
    L.append("")
L += ["## Notes", "",
      "- These Qwen runs use Adafactor (an AdamW state for 1.5B does not fit the 12 GB secondary GPU); the 1B headline uses AdamW. The cross-family question is whether the asymmetry shape replicates in another family/size, not an optimiser-controlled comparison.",
      "- The HuggingFace decode path here neutralises the model's chat-tuned `generation_config` (repetition penalty set to 1.0) so it matches `llama-cli --temp 0`; Qwen-2.5-Instruct ships repetition_penalty=1.1, which depresses verbatim regurgitation on the bf16/AWQ columns if left on -- the old single-seed wave_1_qwen*_mini numbers were collected before that fix.",
      "- The single-seed wave_1_qwen_mini / wave_1_qwen15b_mini dirs are superseded by these three-seed runs for the cross-family claim.", ""]
Path("experiment/results/EXTRA_ANCHORS_RESULTS.md").write_text("\n".join(L))
print("  wrote experiment/results/EXTRA_ANCHORS_RESULTS.md")
PYEOF

# (d) commit the new result dirs + the summary to origin/main (NOT paper/)
say "committing new result dirs + summary to origin/main"
git -C "$REPO" add -- \
  experiment/results/wave_1_llama3b_lora_seed42 experiment/results/wave_1_llama3b_seed42 \
  experiment/results/wave_1_qwen05b_seed42 experiment/results/wave_1_qwen05b_seed52 experiment/results/wave_1_qwen05b_seed62 \
  experiment/results/wave_1_qwen15b_seed42 experiment/results/wave_1_qwen15b_seed52 experiment/results/wave_1_qwen15b_seed62 \
  experiment/results/qwen_extra_pooled_*.json experiment/results/EXTRA_ANCHORS_RESULTS.md experiment/results/wave_1_mini/delta_norm.json 2>/dev/null
git -C "$REPO" commit -q -m "extra anchors: 3B LoRA-merged scale point + Qwen-2.5 0.5B/1.5B 3-seed cross-family (+AWQ); pooled stats + EXTRA_ANCHORS_RESULTS.md (auto-generated by run_extra_anchors_overnight.sh)" 2>/dev/null \
  && git -C "$REPO" push -q origin main 2>/dev/null && say "pushed to origin/main" || say "(nothing new to commit, or push failed -- check git status)"

# (e) add the new result dirs to the artifact manifest, re-assemble, force-push the standalone repo
say "updating sbseg/ARTIFACT_MANIFEST.txt + re-assembling + pushing the standalone artifact repo"
MAN="$REPO/sbseg/ARTIFACT_MANIFEST.txt"
for p in experiment/results/wave_1_llama3b_lora_seed42/ experiment/results/wave_1_llama3b_seed42/ \
         experiment/results/wave_1_qwen05b_seed52/ experiment/results/wave_1_qwen05b_seed62/ \
         experiment/results/wave_1_qwen15b_seed42/ experiment/results/wave_1_qwen15b_seed52/ experiment/results/wave_1_qwen15b_seed62/; do
  grep -qxF "$p" "$MAN" || sed -i "/^experiment\/results\/wave_1_seed82\/$/a $p" "$MAN"
done
grep -qxF "experiment/results/EXTRA_ANCHORS_RESULTS.md" "$MAN" || sed -i "/^experiment\/results\/SCHEMA.md$/a experiment/results/EXTRA_ANCHORS_RESULTS.md\nexperiment/results/qwen_extra_pooled_qwen05b.json\nexperiment/results/qwen_extra_pooled_qwen15b.json" "$MAN"
git -C "$REPO" add -- sbseg/ARTIFACT_MANIFEST.txt 2>/dev/null
git -C "$REPO" commit -q -m "artifact manifest: include the extra-anchor result dirs + pooled stats + EXTRA_ANCHORS_RESULTS.md" 2>/dev/null && git -C "$REPO" push -q origin main 2>/dev/null || true
QQUILT_REPO="$REPO" bash "$REPO/sbseg/assemble_artifact.sh" "$ARTIFACT_DIR" >/dev/null 2>&1 || say "WARN: re-assemble failed"
if [ -d "$ARTIFACT_DIR/.git" ]; then
  git -C "$ARTIFACT_DIR" add -A
  git -C "$ARTIFACT_DIR" -c user.name='cristhian' -c user.email='cristhiank552@gmail.com' commit -q -m "extra anchors: 3B LoRA scale point + Qwen 0.5B/1.5B 3-seed cross-family (+AWQ); pooled stats + EXTRA_ANCHORS_RESULTS.md" 2>/dev/null \
    && git -C "$ARTIFACT_DIR" push -q origin HEAD 2>/dev/null && say "force-pushed the artifact repo" || say "(artifact repo: nothing new, or push failed)"
fi

say "DONE. Summary: experiment/results/EXTRA_ANCHORS_RESULTS.md . The paper (paper/) was NOT touched -- fold the new Qwen numbers into the cross-family table yourself; the §4.6 Hayes '7-13x at (n=10,p=0.5)' claim still needs your verification."
