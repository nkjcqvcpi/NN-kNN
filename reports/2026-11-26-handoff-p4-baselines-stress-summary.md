## Handoff 2026-11-26 pack p4: TabPFN baseline and covariate_pc1 shift stress (exploratory, 3 seeds)

**Status.** These are exploratory runs with our own choices (p4 `DECLARATION.md`), not PI decisions, and nothing here is confirmatory. With 3 seeds and 9 to 13 tasks per group, no number is a significance claim.

**Source.** Accel `runs/` is empty. The sync loop (`/var/tmp/nnknn-sync/sync_loop.sh`) copies q-lab and xps8960 to r760 only, and no tool syncs r760 to accel, so `handoff_tabpfn_r760` and `handoff_shift_r760` are missing on accel. I copied nothing by hand. I re-dumped r760 `runs/t1` read-only at 2026-11-27 05:46 UTC (`tools/t1_handoff_p4_summary.py dump`, 2208 units) and ran `report` on accel. Outputs are in `neural-cbr/results/t1/handoff_20261126/p4/`. The tables match the earlier dump exactly.

**Conventions.**
- Error is 1 − accuracy for classification and standardized RMSE for regression.
- F and reuse use the post-adaptation output; retrieval_only uses the pre output. For diabets, F is the reuse arm (aggregate_regression_combined).
- "mean ± std" is the mean over the 3 seeds of the across-task mean, with the std over those seeds. Diff = method − reference, paired by task and seed.
- W/T/L counts tasks for the reference, using the sign of the task's 3-seed mean diff.

### Completeness (r760, all complete)

| set | units | verify_run problems |
|---|---|---|
| TabPFN, 24 tasks × s50-52 | 72/72 | 0 |
| shifted kNN/MLP/CatBoost, 20 tasks × s30-32 | 180/180 | 0 |
| shiftF (cores + retrieval_only + reuse + F) | 237/237 | 0 |
| shift retention K0.5 (8 policies + core) | 540/540 | 0 |
| references: maskedF 258, allcaseF F 69, grids 264+84, plan_full K0.5 504 | all complete | not checked |

### A. TabPFN vs the other methods (seeds 50-52; reference TabPFN, error 0.066 ± 0.002 cls / 0.381 ± 0.003 reg)

| method | type | tasks | error | diff vs TabPFN | TabPFN W/T/L |
|---|---|---|---|---|---|
| maskedF F | cls | 13 | 0.093 ± 0.005 | +0.027 ± 0.007 | 11/1/1 |
| maskedF F | reg | 11 | 0.423 ± 0.013 | +0.042 ± 0.013 | 11/0/0 |
| allcaseF F | cls | 13 | 0.119 ± 0.019 | +0.053 ± 0.017 | 11/0/2 |
| allcaseF F | reg | 10* | 0.385 ± 0.012 | +0.035 ± 0.010 | 10/0/0 |
| CatBoost | cls / reg | 13 / 11 | 0.073 / 0.410 | +0.007 ± 0.008 / +0.028 ± 0.003 | 8/2/3, 8/0/3 |
| TabM | cls / reg | 13 / 11 | 0.134 / 0.417 | +0.068 / +0.036 | 10/1/2, 10/0/1 |
| MLP | cls / reg | 13 / 11 | 0.146 / 0.436 | +0.081 / +0.055 | 12/0/1, 11/0/0 |
| kNN | cls / reg | 13 / 11 | 0.133 / 0.566 | +0.067 / +0.185 | 11/1/1, 11/0/0 |

*allcaseF diabets is missing because the dump reads only `tuned__F`, and diabets' F is the reuse arm. On those 10 tasks TabPFN's error is 0.350.

TabPFN's only loss to maskedF F is credit-g (tabarena_363626: 0.260 vs 0.246). CatBoost beats TabPFN on zebra_special (0.030 vs 0.121).

### B. covariate_pc1 shift (seeds 30-32; 9 cls + 11 reg tasks; reference shifted F)

| method | type | overall | near | far | diff vs F (overall) | F W/T/L | far − near |
|---|---|---|---|---|---|---|---|
| F | cls | 0.240 ± 0.032 | 0.188 | 0.292 | – | – | +0.103 ± 0.065 |
| F | reg | 1.031 ± 0.039 | 0.598 | 1.273 | – | – | +0.676 ± 0.062 |
| retrieval_only | cls / reg | 0.276 / 1.104 | 0.208 / 0.636 | 0.346 / 1.368 | +0.037 / +0.073 | 6/0/3, 9/0/2 | +0.139 / +0.732 |
| reuse arm | cls / reg | 0.276 / 1.145 | 0.218 / 0.624 | 0.336 / 1.441 | +0.037 / +0.114 | 6/0/3, 9/1/1 | +0.118 / +0.817 |
| kNN | cls / reg | 0.272 / 1.288 | 0.207 / 0.792 | 0.339 / 1.562 | +0.032 / +0.257 | 5/0/4, 9/0/2 | +0.132 / +0.770 |
| MLP | cls / reg | 0.256 / 1.307 | 0.223 / 0.622 | 0.289 / 1.674 | +0.017 / +0.276 | 5/0/4, 7/0/4 | +0.066 / +1.052 |
| CatBoost | cls | 0.161 ± 0.022 | 0.118 | 0.205 | −0.079 ± 0.020 | 3/0/6 | +0.087 |
| CatBoost | reg | 1.216 ± 0.014 | 0.712 | 1.494 | +0.185 ± 0.034 | 7/0/4 | +0.783 |

The reuse tie in regression is diabets, where reuse and F are the same unit.

**Shift retention at K 0.5**, policy − full_memory error (shifted vs unshifted plan_full):
- random: +0.043 / +0.069 shifted, against +0.023 / +0.013 unshifted (cls / reg).
- stratified: +0.047 / +0.063 shifted, against +0.025 / +0.013 unshifted.
- coverage_reachability: +0.007 / +0.040 shifted, against +0.019 / +0.027 unshifted.
- bias_current: −0.000 ± 0.060 / +0.052 shifted, against +0.006 / +0.003 unshifted.

All 7 policies are in `summary.md`.

### What the numbers show and do not show

- **TabPFN.** In this regime (≤3000 training rows, CPU, n_estimators 4, no tuning), TabPFN v2 has the lowest mean error. It beats both F variants on 10-11 of 11-13 tasks per type. CatBoost comes closest. Five tasks were left out (two over 3000 rows, three infeasible on CPU), so this says nothing about larger tasks.
- **Shift, classification.** CatBoost beats F on 6 of 9 tasks. F against kNN and MLP is a 5/4 split, so there is no consistent edge.
- **Shift, regression.** F has the lowest mean error and beats each baseline on 7-9 of 11 tasks. The means are driven by a few tasks: car is about 5 RMSE for every method, CatBoost has 1.84 on energy_efficiency, and MLP has 3.29 on body_fat. Read W/T/L, not the means.
- **Extrapolation.** Every method is worse on the far half. F's far − near gap is not smaller than the baselines' gaps, so this gives no evidence that F extrapolates better.
- **Adaptation.** F beats retrieval_only on 6/9 classification and 9/11 regression tasks, and the gap is larger on the far half. The frozen-core reuse arm is no better than retrieval_only: +0.0001 classification (5/1/3), +0.041 regression (5/0/6).
- **Retention.** Under shift, pruning to K 0.5 costs more in regression than without shift, and the seed spread is larger. The shifted and unshifted runs have different test sets, so only the differences are comparable.
- **One test set.** The 3 seeds share one PC1 test set. The seed std covers only train/validation resampling and initialization, not test sampling. Some tasks are very unstable across seeds: F iris 0.318 ± 0.418, breast_cancer 0.239 ± 0.169, zebra 0.131 ± 0.227.

### Independent check

These checks read the raw TensorBoard events with `EventFileLoader` and no project code (`tools/t1_handoff_p4_indep_check{,2}.py`, run on r760; outputs `indep_check{1,2}.txt`). Every value matches the script to 3 decimals:
- TabPFN: 0.06588 ± 0.00211 / 0.38110 ± 0.00323.
- allcaseF F − TabPFN: +0.05266 ± 0.01676 (11/0/2) and +0.03470 ± 0.01018 (10/0/0).
- Shift CatBoost − F: −0.07856 ± 0.01981 (3/0/6) and +0.18516 ± 0.03386 (7/0/4).
- F far − near: +0.10338 ± 0.06453 (7/0/2) and +0.67593 ± 0.06180 (11/0/0).
- retrieval_only − F: +0.03665 / +0.07338.
