#!/usr/bin/env bash
# Final orchestrator: wait for 3B-LoRA + 5 Qwen workers, rsync gama back,
# fire watchdog Phase 2, then reviewer-experiments.
set -uo pipefail
cd /mnt/win_ssd/usenix
ts() { date +%Y-%m-%dT%H:%M:%S%z; }
say() { echo "[$(ts)] [final] $*"; }

LOCAL_TAGS=(wave_1_llama3b_lora_seed42)
REMOTE_TAGS=(wave_1_qwen05b_seed52 wave_1_qwen05b_seed62 wave_1_qwen15b_seed42 wave_1_qwen15b_seed52 wave_1_qwen15b_seed62)

declare -A SYNCED
for t in "${REMOTE_TAGS[@]}"; do SYNCED[$t]=0; done

while true; do
  all_done=1
  # local
  for TAG in "${LOCAL_TAGS[@]}"; do
    if [ ! -f "experiment/results/$TAG/metrics.json" ]; then
      all_done=0
    fi
  done
  # remote -- check rows + metrics, rsync back when ready
  for TAG in "${REMOTE_TAGS[@]}"; do
    [ "${SYNCED[$TAG]}" = "1" ] && continue
    rows=$(ssh -o ConnectTimeout=5 gpu1 "wc -l < ~/usenix/experiment/results/$TAG/extraction.jsonl 2>/dev/null" 2>/dev/null)
    rows=${rows:-0}
    has_m=$(ssh -o ConnectTimeout=5 gpu1 "[ -f ~/usenix/experiment/results/$TAG/metrics.json ] && echo Y || echo N" 2>/dev/null)
    if [ "$rows" -ge 3000 ] 2>/dev/null && [ "$has_m" = "Y" ]; then
      say "$TAG ready on gama (rows=$rows, metrics=Y) -> rsync back"
      rsync -a gpu1:~/usenix/experiment/results/$TAG/ experiment/results/$TAG/ 2>&1 | tail -1
      SYNCED[$TAG]=1
    else
      say "  $TAG: gama rows=$rows metrics=$has_m"
      all_done=0
    fi
  done
  if [ "$all_done" = "1" ]; then
    say "ALL 6 metrics.json present on main (3B-LoRA + 5 Qwen)"
    break
  fi
  sleep 90
done

say "firing watchdog Phase 2 -> writes EXTRA_ANCHORS_RESULTS.md + commits + force-pushes SBSeg artifact"
bash scripts/run_extra_anchors_overnight.sh >> experiment/results/overnight.log 2>&1
say "  watchdog rc=$?"

say "firing reviewer-experiments runner"
bash scripts/run_reviewer_experiments.sh >> experiment/results/reviewer_experiments.log 2>&1
say "  reviewer-exp rc=$?"

say "DONE -- SBSeg artifact repo (CristhianKapelinski/quantizer-pii-mitigation) now reflects real data"
