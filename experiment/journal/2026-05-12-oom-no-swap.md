# 2026-05-12 — pipeline OOM-killed (no swap on the main host)

## Symptom

Around 01:30 on 2026-05-12 the v3-roadmap chain (`scripts/run_v3_roadmap.sh`,
mid Exp 5 — utility-3-seed, seed 62's PPL eval) and all its `qquilt.*` child
processes vanished. The host's 15-minute load average had spiked to ~13.6
(falling to ~1.5 by the time it was inspected); the user reported the machine
"froze". The status / watcher background processes (`acr_merge_watcher.sh`, the
status Monitor) were also killed.

## Investigation

* `uptime`: **14h 45m, booted 2026-05-11 10:59** — the host did **not** reboot.
* `free`/`swapon`: **no swap configured** (`Swap: 0B`), 30 GiB RAM. With no swap,
  a memory-pressure event makes the kernel OOM-kill processes immediately (and
  the system can stall momentarily under the pressure), rather than degrading to
  swap thrashing first. That matches "froze, then the heavy processes were gone".
* No OOM lines visible in `dmesg`/`journalctl` (no root access; nothing logged
  to a readable location).
* What survived: nothing heavy. The lightweight bash poll-loops (`acr_merge_watcher`,
  etc.) and the GPU-side gpu2 ACR run (separate host) were unaffected — the OOM
  killer targets the large-RSS processes (the `qquilt.utility` / training pythons).
* Likely trigger: the utility-3-seed eval (which loads each model version in HF /
  shells out to `llama-perplexity`) running concurrently with whatever the user
  had on the host, pushing committed memory past 30 GiB. (Utility eval alone is
  not heavy — ~2 GiB RSS — so the host was already loaded from outside the pipeline.)

## Cause

No swap + a fixed 30 GiB RAM ceiling + a transient over-commit ⇒ hard OOM-kill of
the pipeline. Not a bug in the pipeline; an environment limit.

## Resolution

* **Restarted idempotently.** Re-ran `scripts/run_v3_roadmap.sh` — every
  experiment is skip-if-output-exists, so it skipped all the closed ones
  (3-seed pooled stats, Step 7, Min-K%, 2×2, Q4_K_S, semantic, utility seed 42 + 52)
  and resumed exactly where it stopped (utility seed 62 → ACR). Nothing was lost
  (everything completed had been committed).
* **Re-armed the watchers** killed by the OOM (`acr_merge_watcher.sh`; the status
  Monitor was replaced by a combined RAM-watch + status Monitor).
* **Added a RAM monitor** (`besg7s1s9` → `b1yjc7gtk`): a `Monitor` poll loop that
  emits an alert line whenever `free`'s *available* drops below 4 GiB (rate-limited
  to once per 5 min), naming the top-RSS processes, plus a full status line every
  ~20 min. So a future over-commit is flagged *before* it OOM-kills.
* **Operational rule** going forward: keep the pipeline to **one GPU/CPU-heavy job
  at a time** (the v3-chain is already sequential; don't pile extra concurrent
  jobs on the main host) to keep RAM headroom.
* **Suggested to the user (their call — no system changes made):** add a swapfile —
  `sudo fallocate -l 16G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile`
  (+ an `/etc/fstab` line to persist) — so a memory spike degrades to slowness
  instead of an OOM-kill + freeze.

## Lessons

* Idempotent, skip-if-output experiment scripts + commit-each-result made the
  recovery a one-liner (`re-run the orchestrator`). Keep that discipline.
* On a swapless host, "free RAM" must stay well above the largest single job's
  working set *plus* whatever else runs there; monitor `available`, not `used`.
* Lightweight bash watchers survive OOM events; heavy python jobs don't — so
  watchers that auto-resume work (or at least flag the gap) are worth having.
