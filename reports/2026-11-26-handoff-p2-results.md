## Pack p2 (2026-11-26 handoff): new case-selection and data-valuation baselines vs the recorded retention rules

These are exploratory runs. Every setting is our own choice, not a PI decision: activation threshold 0.05, solve tolerance 0.5, energy margin 0.2, energy deletion stops at K, and regression energy uses squared error. There are 3 seeds (s30-s32), and K is 0.2/0.5/0.75. Nothing was trained or committed.

**Source.**
- Accel `runs/` is empty, and no sync tool fills it. The sync loop places the q-lab and xps8960 units on r760: `placed.jsonl` lists all 16 xps8960 and all 26 q-lab p2 jobs, and the other 24 jobs ran on r760 itself. So no study is missing.
- `tools/t1_handoff1126_p2.py dump` read r760 `runs/t1` read-only: 4749 units, every one verified complete.
- `summarize --exclude handoff_newpol_r760:cla:wine:s32` ran on accel. Output is in `results/t1/handoff_20261126/p2/`.

**Completeness.**
- newpol: 20 of 21 tasks have all 63 policy units, plus 3 `ch_reliability` and 3 `_core` units each.
- Classification wine has 42 of 63. **s32 is marked missing because it is being rerun** (23 units excluded), so the paired classification comparisons use 9 tasks.
- latency has all 1575 of 1575 units, and the plan_full references are complete.
- Pairing checks:
  - newpol `fullmem_*` equals plan_full for the same task and seed: 124 pairs, max |diff| 0.
  - latency equals plan_full on `test_metric_pre`: 1512 units, max |diff| 0.

**Method.**
- Metric: test accuracy (classification) or RMSE (regression, lower is better) after finetune.
- d = policy minus reference, paired by task and seed on the same core.
- ± is the sd over the 3 seeds of the per-seed mean d.
- W/T/L counts tasks; a tie is |3-seed mean d| ≤ 0.005.

### Classification accuracy vs provenance_only (9 tasks; reference 0.833/0.885/0.897)

| policy | K0.2 | K0.5 | K0.75 |
|---|---|---|---|
| rc_cnn | +0.039±0.014 6/2/1 | +0.006±0.013 2/4/3 | −0.002±0.009 2/5/2 |
| crr | −0.019±0.011 1/2/6 | +0.011±0.020 3/4/2 | +0.012±0.018 4/4/1 |
| bbnr | +0.008±0.013 5/1/3 | −0.002±0.015 2/3/4 | +0.003±0.010 2/4/3 |
| enn | +0.016±0.011 5/2/2 | −0.002±0.016 3/3/3 | −0.000±0.009 2/5/2 |
| energy_compress | +0.044±0.022 6/2/1 | +0.019±0.026 3/3/3 | +0.002±0.014 3/4/2 |
| knn_shapley | −0.003±0.019 2/4/3 | −0.028±0.022 1/2/6 | −0.026±0.007 1/2/6 |
| loo_value | +0.002±0.058 5/0/4 | +0.012±0.015 3/4/2 | +0.002±0.012 2/6/1 |
| *random (10 tasks)* | +0.015±0.006 6/1/3 | −0.006±0.010 2/2/6 | +0.002±0.011 2/6/2 |

### Regression RMSE vs provenance_only (11 tasks; reference 0.451/0.429/0.419; negative is better)

| policy | K0.2 | K0.5 | K0.75 |
|---|---|---|---|
| rc_cnn | −0.031±0.005 7/1/3 | −0.019±0.004 3/2/6 | −0.011±0.001 4/6/1 |
| crr | +0.014±0.021 3/0/8 | −0.004±0.004 3/1/7 | −0.007±0.008 2/4/5 |
| bbnr | +0.016±0.009 3/2/6 | −0.006±0.005 3/3/5 | −0.003±0.004 3/5/3 |
| enn | +0.018±0.018 2/2/7 | −0.001±0.010 3/3/5 | +0.006±0.011 1/5/5 |
| energy_compress | −0.039±0.004 7/3/1 | −0.020±0.001 4/5/2 | −0.011±0.003 3/7/1 |
| knn_shapley | +0.086±0.013 1/1/9 | +0.062±0.012 0/1/10 | +0.042±0.010 1/1/9 |
| loo_value | −0.020±0.008 4/4/3 | −0.014±0.004 2/4/5 | −0.007±0.002 1/7/3 |
| *kcenter (recorded)* | −0.031±0.012 6/4/1 | −0.020±0.009 5/5/1 | −0.013±0.005 4/6/1 |

**Against random at K0.2.**
- energy_compress: +0.027 (7/0/2) classification, −0.054 (9/2/0) regression.
- rc_cnn: −0.046 (8/3/0) regression.
- knn_shapley is worse than random in all 6 cells.
- enn and bbnr are within ±0.01 of random.

All 15 rules against both references are in `paired_vs_reference.csv`.

**Recovery and retained, K0.2.** Recovery is the gain from finetune (> 0 means finetune helped). Retained is the metric over the full-memory metric; for regression it is an RMSE ratio, so lower is better.

| policy | cla recovery | cla retained | reg recovery | reg RMSE ratio |
|---|---|---|---|---|
| energy_compress | 0.012 | 0.974 | 0.005 | 1.053 |
| rc_cnn | 0.016 | 0.967 | 0.029 | 1.145 |
| crr | 0.078 | 0.898 | 0.172 | 1.425 |
| knn_shapley | 0.035 | 0.923 | 0.124 | 1.745 |
| provenance_only | 0.010 | 0.929 | 0.067 | 1.293 |
| random | 0.033 | 0.950 | 0.086 | 1.441 |

A large recovery goes with a poor memory before finetune.

**Latency (eff_*, medians).**
- Maintenance overhead: the new policies take 0.17–0.33 s (rc_cnn 0.33, energy 0.26) and the recorded rules 0.11–0.14 s. Selection alone takes 0.05–0.15 s.
- Single-query lookup takes about 0.1–0.2 ms, about 1.2× faster than full memory at K0.2.
- Batch lookup at K0.2 takes about 1 ms, against 13–15 ms with full memory.
- Lookup time depends on K, not on the policy.

**C/H reliability.** Spearman of a query subset against the full set: mean over 21 tasks, worst task in brackets.

| fraction | C | H | Q | Qe |
|---|---|---|---|---|
| 10% | 0.62 (0.42) | 0.54 (0.37) | 0.55 (0.33) | 0.45 (0.16) |
| 25% | 0.78 (0.60) | 0.71 (0.53) | 0.72 (0.51) | 0.66 (0.43) |
| 50% | 0.90 (0.76) | 0.84 (0.65) | 0.86 (0.72) | 0.83 (0.68) |

H covers only 19–20 tasks, because the correlation is undefined when H is constant.

**Independent check.** `raw_event_check.py` reads the raw event files with tensorboard's loader only, no project code. It reproduces two cells exactly:
- energy_compress K0.2 classification: +0.043845, sd 0.021528, 6/2/1.
- rc_cnn K0.2 regression: −0.031194, sd 0.004454, 7/1/3.

**What this shows:**
- At K0.2, energy_compress and rc_cnn beat provenance_only and random on most tasks. energy_compress is the most consistent.
- knn_shapley, in our adaptation, is clearly worse.
- At K0.75 almost everything ties.
- The new policies add 0.05–0.2 s of maintenance and no lookup cost.

**What it does not show:**
- Significance: with 3 seeds there are no confidence intervals.
- Means robust to single tasks: car alone gives −0.22 RMSE at K0.2, and without car rc_cnn's mean is about −0.012.
- Results on large case bases: every case base is ≤ about 4k cases.
- Latency comparable across hosts: three hosts, under shared load.
- Classification wine.
- The synthetic study `handoff_synnewpol_r760`.

### Update (2026-11-26 evening): classification wine s32 included

The xps8960 rerun of newpol classification wine s32 (after fix 4235e10) is complete and placed on r760. Its `fullmem_*`
values equal plan_full (126 pairs, max |diff| 0). `summarize` was rerun on the same r760 dump without `--exclude`, so
the tables in `results/t1/handoff_20261126/p2/` now use 10 classification tasks. The 9-task tables above are kept in
`results/t1/handoff_20261126/p2/excluding_cla_wine_s32/`. The 10-task classification numbers are in
`reports/2026-11-26-handoff-everything-results.md`. Regression is unchanged. Caveat: wine s30/s31 were recorded with
the old engine member rule and s32 with the new one; whether s30/s31 had tied rows on xps8960 has not been checked.

### Note (2026-11-27): wine s30/s31 tie count and raw-event check on the 10-task tables

**Wine s30/s31 under the old engine rule.** xps8960 was unreachable all evening ("websocket: bad handshake" on six
tries over about five minutes), so the check ran read-only on r760 instead. It used the same recorded inputs: the
`_core` weights in `runs/t1/handoff_newpol_r760/newpol_retention__classification/wine/` and the data snapshots in
`neural-cbr/results/t1/handoff_newpol_r760/inputs/`. It used the fixed code in r760's `neural-cbr` tree. No unit was
retrained or deleted. The scripts are in `/var/tmp/nnknn-ties1127/` on accel:
- `count_ties.py`: rows whose pre-top-k member set differs between the old rule and model.retrieve at full memory.
  s30: 0 of 106. s31: 0 of 106. s32: 2 of 106. So the counter does find s32's known problem.
- `any_ties.py`: rows with any exact float32 tie between two eligible cases in the scores model.retrieve ranks.
  s30: 0 of 106. s31: 0 of 106. s32: 106 of 106. Without any tie, the old rule and the fixed rule pick the same
  members under every case mask. So, computed on r760, the fixed code gives exactly the recorded s30/s31 results.
  Zero rows can differ.
- `gap.py`: the smallest gap between two adjacent eligible scores in any row. s30: 65 float32 ulps. s31: 18 ulps.
  s32: 0. Neither seed has duplicate training rows.

Caveat: the scores were computed on r760, not on xps8960, where the units ran. A tie on xps8960 would need the two
hosts' float32 scores to differ by at least 18 ulps for s31 (65 ulps for s30). That is unlikely, but it has not been
measured on xps8960. The check covers the reference engine (train_loo queries). The problem engine is the same
object whenever its queries equal the cases.

**Raw-event check on the 10-task tables.** `raw_event_check.py` was rerun on r760 without `--exclude`. The output is
in `results/t1/handoff_20261126/p2/raw_event_check_10task.txt`. Every cell agrees with `paired_vs_reference.csv`
(metric m, reference provenance_only) to all printed digits, and on W/T/L:
- energy_compress K0.2 classification: +0.040387, sd 0.018245, 7/2/1 over 10 tasks.
- rc_cnn K0.2 classification: +0.035743, sd 0.013569, 7/2/1.
- crr K0.2 classification: −0.019803, sd 0.019289, 1/2/7.
- rc_cnn K0.5 classification: +0.006058, sd 0.010122, 3/4/3.
- rc_cnn K0.2 regression: −0.031194, sd 0.004454, 7/1/3 (unchanged).

The per-task wine means (energy_compress +0.009259, rc_cnn +0.009259, crr −0.027778) equal the `cla:wine` rows of
`paired_per_task.csv`. With `--exclude wine:s32`, the old 9-task numbers (+0.043845, 6/2/1) are reproduced exactly.
