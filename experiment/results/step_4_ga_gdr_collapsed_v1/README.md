# step_4_ga_gdr_collapsed_v1 — archived failed run

Step 4 v1 of the Zhang-replication-adapted unlearning experiment:
GA_GDR **without** the per-example forget-CE threshold. Unbounded
gradient ascent on the 100-canary forget set drove `forget_ce` from
0.07 → 97 over 250 steps; the model then generated `= = = = =` garbage
at every canary prefix, and all quantized variants inherited the
collapse (0/100 trivially, but meaningless).

Kept here for the record:
* `unlearn_steps.jsonl` — the forget-CE blow-up trajectory.
* `metrics.json` — confirms 0/100 across all versions (collapsed model).

The bulky garbage `extraction.jsonl` was removed (no value — it's
`= = = =` repeated). The fixed run (threshold=5, 2 epochs) is in
`../step_4_ga_gdr/`. See `experiment/journal/2026-05-11-step5-saliency-refuted.md`
and `experiment/wave_1/WAVE_1.md` §7.3.
