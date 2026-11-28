## Handoff 2026-11-26, pack p3: core and reuse ablations compared with their recorded references

These are exploratory runs. Every value was our own choice, and each cell has 3 seeds (30-32; the pre-top-K runs use 70-72). Nothing was trained or written under `runs/`, and nothing was committed.

**Where the numbers come from.** Accel `runs/` is **empty**. No sync tool fills it. The sync loop (`/var/tmp/nnknn-sync/sync_loop.sh`) copies q-lab and xps8960 units into an accel mirror (`/var/tmp/nnknn-sync/<host>/mirror/runs`) and places them on r760. r760-native units (abalone, car, regression wine, synthetic_nominal) and their plan_full references exist only on r760. So I copied nothing by hand. The read-only dumper `tools/t1_handoff1126_p3.py dump` ran on r760 against `/mnt/largedata/NN-KNN/runs`, the full canonical copy. `analyze` ran on accel and wrote `results/t1/handoff_20261126/p3/` (`per_task.csv` holds the per-task mean, sd and per-seed differences; also `overall.csv`, `completeness.json`, `pilot_bias_init.csv`, `problems.txt`).

**Completeness.** Every unit is `complete` (run_status), with 0 incomplete or failed, and each count is exactly what was expected: pct10/25/50 72 each (12 ds × 2 arms × 3), pctsync 360 + 396, pvdist 120, percase 90/99/9, directhead 60/6, nodu 21. The references are complete too: pretopk_expanded 72; plan_full sync cls 330, sync reg 407, reuse cls 210, reuse reg 156, nominal 21. Every paired seed was found, and in every pair the condition and the reference were recorded on the same host.

**Rule.** For each seed, d = condition − reference. **b**etter means the mean of d favours the condition and at least 2 of the 3 seeds do. **w**orse is the mirror case. **i**dentical means all three d are 0. Everything else is a **t**ie. Counts are b/t/w/i over tasks. Accuracy is higher-is-better. test_L_post (NLL), RMSE and test_error are lower-is-better.

### (a) Percentile bias initialiser

Pre-top-K all-case arm, test_error, 12 tasks. The reference is knn_mean k=5 at 0.2016:

| arm | mean | Δ | b/t/w/i |
|---|---|---|---|
| pct10 | 0.2016 | −0.00001 | 2/1/5/4 |
| pct25 | 0.2016 | +0.00005 | 2/2/4/4 |
| pct50 | 0.2017 | +0.00006 | 2/2/4/4 |

Every per-task |Δ| is ≤ 0.0012, even though the starting bias rises 1.1–5.8× over knn_mean. The masked arm, reported as extra, gives Δ −0.006 / −0.015 / −0.006 (3/3/4/2, 5/1/3/3, 4/3/4/1).

Sync, Δ against plan_full for the same condition. Classification uses 10 tasks (accuracy and NLL); regression uses 11 tasks (RMSE):

| arm / condition | Δacc (b/t/w/i) | ΔNLL (b/t/w) | ΔRMSE (b/t/w) |
|---|---|---|---|
| pct10 independent | −0.006 (2/2/4/2) | +0.004 (3/3/4) | −0.002 (7/1/3) |
| pct10 alternating_rr | +0.003 (1/7/2/0) | −0.003 (4/2/4) | +0.005 (3/3/5) |
| pct10 rr_initial_bias_radius | +0.002 (1/8/1/0) | −0.013 (5/2/3) | +0.005 (3/3/5) |
| pct25 independent | −0.013 (2/6/2/0) | +0.010 (4/2/4) | −0.010 (4/4/3) |
| pct25 alternating_rr | −0.003 (3/6/1/0) | −0.019 (6/1/3) | +0.010 (3/1/7) |
| pct25 rr_initial_bias_radius | −0.003 (3/6/1/0) | −0.026 (7/1/2) | +0.011 (3/1/7) |
| pct50 independent | −0.001 (2/3/4/1) | −0.000 (4/4/2) | −0.032 (7/0/4) |
| pct50 alternating_rr | +0.007 (2/6/1/1) | −0.044 (8/1/1) | +0.002 (4/2/5) |
| pct50 rr_initial_bias_radius | +0.006 (3/6/0/1) | −0.035 (9/0/1) | +0.002 (5/1/5) |

Reference means: accuracy 0.902 / 0.894 / 0.894, RMSE 0.466 / 0.431 / 0.431.

The free radius is flagged `overly_broad` more often as the percentile rises. Out of 63 rows per condition: rr_initial_bias_radius goes 11 → 20 → 35 for pct10 → 25 → 50, against 9 in the reference; alternating_rr goes 5 → 10 → 24, against 1.

### (d) probability_vector_distance (10 classification tasks)

| condition | Δacc (b/t/w/i) | ΔNLL (b/t/w) |
|---|---|---|
| alternating_rr | +0.001 (1/3/1/5) | +0.001 (3/4/3) |
| rr_initial_bias_radius | +0.001 (1/4/0/5) | +0.002 (1/2/7) |
| rr_exact_kth_radius | +0.001 (0/6/0/4) | +0.002 (1/1/8) |

The median tau_task ratio against the reference is 0.89–0.96.

### (b) per_case_2022 against the aggregate adapter (same loss)

| task set | loss | cond | ref | Δ | b/t/w/i |
|---|---|---|---|---|---|
| cls acc, 10 | diff_only | 0.911 | 0.907 | +0.004 | 4/4/1/1 |
| cls acc, 10 | combined | 0.912 | 0.908 | +0.004 | 5/1/2/2 |
| reg RMSE, 11 | diff_only | 0.489 | 0.417 | +0.072 | 5/3/3/0 |
| reg RMSE, 11 | combined | 0.485 | 0.417 | +0.068 | 7/3/1/0 |
| synthetic_nominal acc | diff_only | 0.426±0.042 | 0.759±0.042 | −0.333 | worse |
| synthetic_nominal acc | combined | 0.444±0.038 | 0.759±0.039 | −0.315 | worse |

The regression mean comes almost entirely from body_fat, where Δ is +0.83 and seeds 30 and 31 are worse by +1.0 to +1.5. Without body_fat, Δ is −0.004 (diff_only) and −0.008 (combined).

### (c) Direct head, a capacity reference (not an adaptation method)

- **Classification, 10 tasks:** accuracy 0.849 against 0.905 for nominal_residual_scores_cls_only (Δ −0.056, 4/2/4/0). NLL Δ −0.013 (7/1/2). The loss comes from zebra (0.485 vs 0.939) and zebra_special (0.530 vs 0.727). The direct head is better on balance, breast_cancer, synthetic_diag and 363626.
- **synthetic_nominal:** 0.463±0.084 against 0.748±0.045.

### (e) omit_nominal_channel (synthetic_nominal only)

All six conditions lose 0.29–0.43 accuracy. For example, nominal_residual_scores_combined scores 0.333±0.069 against 0.759±0.039. Retrieval-only accuracy is 0.370, so without Δu_q the adapters are at or below plain retrieval.

### What these numbers show and do not show

**They show:**
- On these tasks, the percentile initialiser changes the pre-top-K all-case outcome by at most about 1e-3.
- In sync it shifts NLL more than accuracy, and pct50 reduces NLL on 8–9 of 10 tasks under the rr conditions. In regression the effect is mixed.
- pvdist is close to neutral.
- per_case_2022 is about equal to or slightly better than the aggregate adapter on the classification and regression tasks, apart from body_fat. It is far worse on synthetic_nominal.
- On synthetic_nominal, removing Δu_q wipes out all of the adapter's gain over retrieval-only (0.370), and most conditions end up below it.

**They do not show:**
- Significance. With 3 seeds, b/t/w is a consistency count, not a test.
- Anything general about nominal data. The nominal and no-Δu conclusions rest on one synthetic task.
- That pct50 is better in general. The higher overly_broad rate makes that harder to read.
- A tested result for the pre-top-K reference. It was recorded on 2026-10-30 at an older commit. Differences of about 1e-4 may be run noise, and I did not test that.

**Independent check:** `/var/tmp/nnknn-p3sum/raw_check.py` reads `x/final/test_error` directly from the raw event files in the accel xps8960 mirror, using only tensorboard's protobuf reader. It reproduces the pct10/25/50 per-task and overall Δ exactly: maximum disagreement 0, overall −1.1875e-05, 5.2613e-05 and 5.7267e-05.

### Re-check, 2026-11-27 05:45 UTC

- Accel `runs/` is still empty, and the sync loop is not running (the last pass was at 21:59). The accel mirrors (`/var/tmp/nnknn-sync/{q-lab,xps8960}/mirror/runs/t1`) hold only the q-lab and xps8960 units. The r760-run units exist only on r760. Nothing was copied by hand.
- A fresh read-only dump on r760 (`/var/tmp/nnknn-p3sum2/p3_dump.jsonl`, same tool) is byte-identical (sha256 02da897e…) to the 2026-11-26 dump. Re-running `analyze` reproduces every output file under `results/t1/handoff_20261126/p3/` byte for byte, with 0 problems. The counts are unchanged, and every pair was recorded on one host.
- Host names in the records: haotang-lab = xps8960, cs-peng-research = r760, q-lab = q-lab. The p3 conditions were recorded at commit 91cee1f7. Their references were recorded at 204c49b5, and some r760 plan_full units at 3ab1549c. All of these predate 4235e10, whose ActivationEngine tie fix changes rows only where there is an exact k-th tie.
- Second independent check (`/var/tmp/nnknn-p3sum3/indep_check.py`) reads `x/final/<metric>` scalars from the raw events on r760, not `rec/row`, and uses no project code. It reproduces three comparisons, each per-task mean, sd and Δ to within 1e-16, with identical b/t/w counts:
  - direct head accuracy: Δ −0.0564 (4/2/4)
  - pct50 rr_initial_bias_radius NLL: Δ −0.0354 (9/0/1)
  - per_case_2022 combined regression RMSE: Δ +0.0680 (7/3/1)
