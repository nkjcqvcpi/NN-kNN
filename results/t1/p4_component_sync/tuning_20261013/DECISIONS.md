# Step D tuning decisions (docs/T1_PREREG_20261013.md)

Development seeds 20–25 only; validation streams for selection, dev-seed test streams only as the
declared leave-one-seed-out (LOSO) inner held-out. Tool: `tools/t1_block_accept.py`. Reference for
held-out log error ratios: the alternating default. A change is adopted only if its mean paired
held-out log ratio is below −1 task-clustered SE. Raw runs: `results/t1/p4_component_sync/pi20260920/`.

**Pool.** The 19 conditions of `configs/t1/tuning_h6_20261013.yaml` (18 batch-64 finalist and reference
schedules, frozen-core reuse) on 18 legacy tasks (addendum 2: car and regression wine are evaluated in
confirmation only). Seeds 20–21 come from the batch-64 stages A/B, seeds 22–25 from H6; abalone (uncapped)
and the three newly tuned tasks have all six seeds from H6.

| Block | Evidence | Decision |
|---|---|---|
| H6, selection rule | `h6_rules.json`: per-task by validation error then loss, held-out GMR 0.891 vs current global rule 0.932 (paired −0.044, task-clustered z −1.98, tasks 13/4) | **Adopt per-task selection** (err/pertask) |
| Feasibility (addendum 3) | `h6_rules_feasible.json`: with H1's pre-adaptation constraint per task, err/pertask GMR 0.895; the global "every task" form drops to 0.967 | Constraint applies to the whole pool from H1 on; per-task rule kept |
| H1, retrieval guard and `w_pre` (10 candidates) | `h1_block.json`: GMR 0.895 → 0.904 when added (paired +0.010, z +0.95) | **Rejected**; incumbent stays F0 |
| H2a, nominal score scale (5 candidates) | `h2a_block.json`: paired −0.004, z −0.38 | Not accepted on error |
| H2b, logit residual (2 candidates, reported separately) | `h2b_block_separate.json`: paired 0.000 | Not accepted |
| H2, output scale for calibration | Validation streams, seeds 22–25, 9 classification tasks: F0 (κ 1) ECE post 0.241 (pre 0.030), with temperature 0.061, correction L1 0.997; κ 10, `v_small` 0.1: ECE post 0.016, with temperature 0.011, correction 0.220, accuracy 0.932 vs 0.928 | κ 10 chosen among accuracy-equivalent settings; **H2c** reruns the classification pool at κ 10, adopted only if held-out error is non-inferior |
| H2c, classification pool at κ 10 (19 conditions) | `h2c_compare.json`: held-out GMR 0.842 (κ 1) → 0.822 (κ 10), paired −0.023 (task-SE 0.036, tasks 5/3); median validation ECE of the per-task selections 0.239 → 0.012. Iris seed 21 crashed natively once (0xC0000409) and was rerun; its 7 overlapping rows reproduced bitwise | **Adopt κ 10 for classification** (non-inferior; calibration fixed) |
| H5, budget and stopping (8 candidates) | regression and classification (κ 10) halves running | pending |
| H3, adapter capacity (9 regression / 11 classification candidates incl. dropout) | both halves running | pending |

**Transfer group.** Per-task selection needs development runs on a task, which the pre-registration
forbids on transfer tasks. F on transfer tasks is therefore one global schedule chosen on the legacy
pool (to be fixed at the freeze, with the global form of the feasibility constraint).
