# Experiment results index

**Overview:** [All adapter experiments and results](../code/neural-cbr/reports/ADAPTER_RESULTS_SUMMARY_2026-11-26.md) consolidates the findings across `am1`, `ag1`, `ag2`, `aq1`, `aq2`, `as1` and `ad1`, including the behavior analysis and local conventional baselines. The detailed evidence for each study is linked below.

## Dropout across adapters and training schemes: ad1 (November 26, 2026)

Status: **complete, 1,134 comparison cells, 756 new models, 378 reused zero-dropout references, no missing cells**. All three current residual adapters, three training schemes, rates 0%/10%/20%, 14 tasks and seeds 40-42. Positive dropout is applied after both hidden ReLUs and disabled during evaluation; parameter counts and initialization remain matched.

Lowest new fixed classification mean: **query without prediction + alternating + 20%, 6.701% error**, versus 9.356% at zero dropout. The improvement is concentrated on Zebra; the other six datasets worsen from 7.380% to 7.818%. Validation-selected rates within this scheme give 6.453%. Lowest new fixed regression mean: **query + prediction + alternating + 10%, 0.337381 RMSE**, versus 0.345628; validation-selected rates give 0.344081. The earlier probability gate is still marginally lower at 0.337206. Original sequential at 20% improves six of seven regression datasets, 0.350429 to 0.338523, with retrieval fixed exactly. Severe Car errors persist; no default is changed.

- [Full report](../code/neural-cbr/reports/ADAPTER_DROPOUT_LOCAL_2026-11-26.md), [pre-result protocol](../code/neural-cbr/docs/ADAPTER_DROPOUT_EXPERIMENT.md), [dropout helper](../code/neural-cbr/model/t1/adapter_regularization.py), [recorded driver](../code/neural-cbr/tools/t1_adapter_dropout.py), [local uv launcher](../code/run_dropout_experiment.py).
- [Complete evidence](../code/neural-cbr/results/adaptation_improvement/ad1/comparison.json), [all model rows](../code/neural-cbr/results/adaptation_improvement/ad1/rows.csv), [task means](../code/neural-cbr/results/adaptation_improvement/ad1/tasks.csv), [family means](../code/neural-cbr/results/adaptation_improvement/ad1/families.csv), [paired comparisons](../code/neural-cbr/results/adaptation_improvement/ad1/comparisons.csv), [figure](../code/neural-cbr/results/adaptation_improvement/ad1/dropout_means.png).
- [Validation-selected rates](../code/neural-cbr/results/adaptation_improvement/ad1/selected_rates.csv), [their means](../code/neural-cbr/results/adaptation_improvement/ad1/selected_rate_families.csv), [joint rate/scheme selections](../code/neural-cbr/results/adaptation_improvement/ad1/selected_both.csv), [their means](../code/neural-cbr/results/adaptation_improvement/ad1/selected_both_families.csv), [selection counts](../code/neural-cbr/results/adaptation_improvement/ad1/selection_counts.csv), [budgets](../code/neural-cbr/results/adaptation_improvement/ad1/budgets.csv).
- [Correction diagnostics](../code/neural-cbr/results/adaptation_improvement/ad1/behavior.csv), [pooled counts/severity](../code/neural-cbr/results/adaptation_improvement/ad1/behavior_totals.csv), [largest harmful examples](../code/neural-cbr/results/adaptation_improvement/ad1/harm_examples.csv). Better regression RMSE can coexist with more wrong-direction corrections because overshoot severity falls.
- Verified 2,226 saved prediction sets, 1,512 initial prediction sets, 378 frozen-core hashes, 336 cross-adapter/rate sequential prediction matches, and parameter counts/dropout metadata for all 756 new models. Independently checked 84 aggregates and 630 selected policies. All 1,134 cells reused with fitting disabled. Targeted suites passed 109 tests plus a new selection test separately; both core manifests passed.
- Run store: `NNKNN_RUNS/t1/ad1/drop_v1/` plus existing references. Worker logs: `logs/ad1/`; reuse audit: `logs/ad1_reuse_check.log`. Local CPU only, no r760 access.

## Sequential, alternating and joint training: as1 (November 26, 2026)

Status: **complete, 378 comparison cells, 252 new models, 126 reused references, no missing cells**. All three residual input designs were compared on the same 14 tasks and seeds 40–42. Sequential permanently freezes the same pretrained retrieval checkpoint; alternating and joint start from that checkpoint. Effective component objectives and maximum update budgets match. Existing joint models are reused for 13 tasks; existing sequential models are reused for Diabetes.

For query + prediction, sequential / alternating / joint classification error is **10.980% / 9.502% / 9.359%**, and standardized regression RMSE is **0.371033 / 0.345628 / 0.345530**. Validation-selected schemes reach **8.745% / 0.345186**. This improves classification, while regression is essentially unchanged from the previous query-aware result. Alternating improves correction behavior but has weaker selected retrieval predictions. Severe Car corrections persist with frozen retrieval. The original default is retained.

- [Full report](../code/neural-cbr/reports/ADAPTER_TRAINING_SCHEMES_LOCAL_2026-11-26.md), [declared protocol](../code/neural-cbr/docs/ADAPTER_TRAINING_SCHEMES_EXPERIMENT.md), [recorded driver](../code/neural-cbr/tools/t1_adapter_training_schemes.py), [local launcher](../code/run_training_schemes_experiment.py).
- [Complete evidence](../code/neural-cbr/results/adaptation_improvement/as1/comparison.json), [all model rows](../code/neural-cbr/results/adaptation_improvement/as1/rows.csv), [task means](../code/neural-cbr/results/adaptation_improvement/as1/tasks.csv), [family means](../code/neural-cbr/results/adaptation_improvement/as1/families.csv), [paired comparisons](../code/neural-cbr/results/adaptation_improvement/as1/comparisons.csv).
- [Validation-selected models](../code/neural-cbr/results/adaptation_improvement/as1/selected.csv), [selected means](../code/neural-cbr/results/adaptation_improvement/as1/selected_families.csv), [selection counts](../code/neural-cbr/results/adaptation_improvement/as1/selection_counts.csv), [update budgets](../code/neural-cbr/results/adaptation_improvement/as1/budgets.csv).
- [Correction diagnostics](../code/neural-cbr/results/adaptation_improvement/as1/behavior.csv), [pooled counts](../code/neural-cbr/results/adaptation_improvement/as1/behavior_totals.csv), [harmful examples](../code/neural-cbr/results/adaptation_improvement/as1/harm_examples.csv), [figure](../code/neural-cbr/results/adaptation_improvement/as1/scheme_means.png).
- Verified 714 saved prediction sets and the 840-check baseline/reference audit, 126 frozen-core hashes and 84 exact cross-adapter sequential prediction matches. All 378 complete cells reused with training disabled. Targeted suites passed 75 and 24 tests (overlapping); both core manifests passed.
- Run store: `NNKNN_RUNS/t1/as1/schemes_v1/`, plus the recorded `am1`/`aq1`/`aq2` references. Worker logs: `logs/as1/`; reuse audit: `logs/as1_reuse_check.log`. No r760 access used.

## Query-aware without prediction input: aq2 (November 25, 2026)

Status: **complete, 42 new model units, 336 comparison rows, no missing units**. The residual network receives the query and query-to-neighborhood difference, with no retrieved-prediction input columns. Retrieval remains in the final addition. All previous baselines are reused.

Mean classification error is **10.146%**, versus 9.359% for query-aware with prediction and 9.945% for the original. Classification GMR is better (0.91477 and 0.89108 respectively), so the ranking depends on aggregation. Regression RMSE is **0.358695**, versus 0.345176 and 0.338910. A single Car seed-40 overcorrection of +21.20 when +0.124 was needed accounts for 98.6% of that run's adaptation harm. No failure was excluded or default changed.

- [Full report](../code/neural-cbr/reports/QUERY_NO_PREDICTION_LOCAL_2026-11-25.md), [declared protocol](../code/neural-cbr/docs/QUERY_NO_PREDICTION_EXPERIMENT.md), [adapter classes](../code/neural-cbr/model/t1/query_aware.py), [recorded driver](../code/neural-cbr/tools/t1_query_no_prediction.py), [local launcher](../code/run_query_no_prediction_experiment.py).
- [All model/seed rows](../code/neural-cbr/results/adaptation_improvement/aq2/runs.csv), [task means](../code/neural-cbr/results/adaptation_improvement/aq2/tasks.csv), [family means](../code/neural-cbr/results/adaptation_improvement/aq2/families.csv), [pairwise comparisons](../code/neural-cbr/results/adaptation_improvement/aq2/comparisons.csv), [full evidence](../code/neural-cbr/results/adaptation_improvement/aq2/comparison.json).
- [Correction contribution](../code/neural-cbr/results/adaptation_improvement/aq2/adaptation.csv), [regression direction/overshoot](../code/neural-cbr/results/adaptation_improvement/aq2/regression_behavior.csv), [largest harmful cases](../code/neural-cbr/results/adaptation_improvement/aq2/harm_examples.csv), [retrieval interventions](../code/neural-cbr/results/adaptation_improvement/aq2/intervention_summary.csv), [comparison figure](../code/neural-cbr/results/adaptation_improvement/aq2/query_no_prediction_comparison.png).
- Run store: `NNKNN_RUNS/t1/aq2/nopred_v1/`. Matched cores come from `am1`; cached data and query-aware/conventional references come from `aq1`. No baseline was trained again.
- Verified 252 new prediction/metric comparisons and 840 reference comparisons, all source target matches, all parameter-count reductions and the prediction-only invariance checks. All 42 completed units were reused with fitting prohibited. The integration suite passed 96 tests (including 35 query-adapter tests); three additional new driver tests passed. Shared-core manifest passed.
- Runtime logs: `../logs/local_aq2_*.log`. Rebuild generated evidence with `--study aq2 --mode summarize`.

## Query-aware original adapter: aq1 (November 25, 2026)

Status: **complete, 42 query-aware models, 672 new recorded units, 294 comparison rows, no missing units**. The original residual adapter gains the query representation as an input, retaining its residual output and selected training schedule. No new gate or magnitude/direction experiment was included.

Mean classification error improves **9.945% -> 9.359%**; mean standardized regression RMSE worsens **0.338910 -> 0.345176**. Query-aware improves four task means in each family, but harmful Car corrections dominate the regression loss. Original and query-aware NN-kNN both beat the compact local MLP/TabM searches on family means; those searches do not reproduce the historical tuned winners. The original residual default remains unchanged.

- [Full report](../code/neural-cbr/reports/QUERY_AWARE_LOCAL_2026-11-25.md), [declared protocol](../code/neural-cbr/docs/QUERY_AWARE_EXPERIMENT.md), [adapter](../code/neural-cbr/model/t1/query_aware.py), [recorded driver](../code/neural-cbr/tools/t1_query_aware.py), [local launcher](../code/run_query_aware_experiment.py).
- [All model/seed rows](../code/neural-cbr/results/adaptation_improvement/aq1/runs.csv), [task means](../code/neural-cbr/results/adaptation_improvement/aq1/tasks.csv), [family means](../code/neural-cbr/results/adaptation_improvement/aq1/families.csv), [pairwise comparisons](../code/neural-cbr/results/adaptation_improvement/aq1/comparisons.csv), [full evidence](../code/neural-cbr/results/adaptation_improvement/aq1/comparison.json).
- [Correction contribution](../code/neural-cbr/results/adaptation_improvement/aq1/adaptation.csv), [retrieval/query interventions](../code/neural-cbr/results/adaptation_improvement/aq1/intervention_summary.csv), [comparison figure](../code/neural-cbr/results/adaptation_improvement/aq1/query_aware_comparison.png).
- Run store: `NNKNN_RUNS/t1/aq1/query_v1/`. Includes 42 data units, 42 query-aware units, 420 validation candidates and 168 selected-baseline evaluations. Old full models and pretrained retrieval are reused from `am1`.
- All 840 saved prediction/metric comparisons and 168 family selections passed reconciliation. Completed query-aware/baseline units were reused without fitting; old predictions were reconstructed successfully for all 42 splits. The latest focused suite passed 46 tests, including all 19 query-aware tests; 11 grid-search/prefill tests passed. Shared-core manifest passed. See report for overlapping broader checks.
- Runtime logs: `../logs/local_aq1_*.log`. Rebuild generated summaries with `--study aq1 --mode summarize`.

## Apply-or-skip exploration: ag1 (November 25, 2026)

Status: **complete, 42 dataset/seed comparisons, 210 recorded units, no missing units**. The magnitude/direction design is shelved. Six alternatives were tested against the old full adapter and its selected retrieval core: global on/off, global shrinkage, disagreement threshold, extra training without a gate, probability gate and utility-weighted gate.

The [full gate report](../code/neural-cbr/reports/ADAPTATION_GATE_LOCAL_2026-11-25.md) records the results and limits. A validation-selected disagreement threshold improves 5/7 classification and 5/7 regression task means; GMR changes are -3.68% and -0.52% versus the old full adapter. Both learned heads harm classification relative to their own full proposals. No default was changed; the original source runs are preserved.

- [Declared protocol](../code/neural-cbr/docs/ADAPTATION_GATE_EXPERIMENT.md), [apply-or-skip implementation](../code/neural-cbr/model/t1/adaptation_gate.py), [recorded experiment/summary tool](../code/neural-cbr/tools/t1_adaptation_gate.py).
- [Task means](../code/neural-cbr/results/adaptation_improvement/ag1/tasks.csv), [all 336 method/seed rows](../code/neural-cbr/results/adaptation_improvement/ag1/runs.csv), [family summaries](../code/neural-cbr/results/adaptation_improvement/ag1/families.csv), [machine-readable evidence](../code/neural-cbr/results/adaptation_improvement/ag1/comparison.json).
- New run units: `NNKNN_RUNS/t1/ag1/gate_v1/`; source: `NNKNN_RUNS/t1/am1/adaptation_improvement_v1/`.
- Local launcher: [run_gate_experiment.py](../code/run_gate_experiment.py). Use `--mode summarize` to rebuild the report data; `--mode run` reuses completed units.
- Runtime logs: `../logs/local_ag1_*.log`. Eight focused new tests passed; 15 tests passed including the prior diagnostic tests. All 672 saved validation/test metric records were reconciled with saved prediction tensors.

## Local paired comparison: am1 (November 25, 2026)

Status: **complete, 42/42 pairs, no missing units**. The new design is better on 7 tasks and worse on 7. Geometric mean error is 3.68% higher for classification and 3.03% higher for regression. See the [full comparison report](../code/neural-cbr/reports/ADAPTATION_IMPROVEMENT_LOCAL_2026-11-25.md).

Behavior analysis: [under-correction, overshoot and direction report](../code/neural-cbr/reports/ADAPTATION_BEHAVIOR_2026-11-25.md), covering all 84 adapted units and 17,064 saved test evaluations. New classification adaptation fixes fewer errors (25 → 8) while causing fewer harmful flips (24 → 11). In regression, wrong-direction updates account for 90.12% of the new adapter's added squared error on harmed cases; Yacht also has increased destructive overshoot. Uniformly doubling corrections worsens both families' average errors. These are descriptive test-label diagnostics, not tuning recommendations.

- Analysis script: [t1_adaptation_diagnostics.py](../code/neural-cbr/tools/t1_adaptation_diagnostics.py).
- Generated behavior evidence: [task diagnostics](../code/neural-cbr/results/adaptation_improvement/am1/behavior/tasks.csv), [per-query diagnoses](../code/neural-cbr/results/adaptation_improvement/am1/behavior/cases.csv), [scale curves](../code/neural-cbr/results/adaptation_improvement/am1/behavior/scaling_summary.csv), [retrieval/adaptation decomposition](../code/neural-cbr/results/adaptation_improvement/am1/behavior/decompositions.csv).

- Code: [T1 worktree](../code/neural-cbr/) on branch `adaptation-improvement`, based on `411abd4`; source changes and snapshots recorded with run units.
- Design: [magnitude/direction notes](../code/neural-cbr/docs/ADAPTATION_MAGNITUDE_DIRECTION.md).
- Runner: [t1_adaptation_improvement.py](../code/neural-cbr/tools/t1_adaptation_improvement.py).
- Generated summaries: [task means](../code/neural-cbr/results/adaptation_improvement/am1/tasks.csv), [seed pairs](../code/neural-cbr/results/adaptation_improvement/am1/pairs.csv), [complete coverage/metrics JSON](../code/neural-cbr/results/adaptation_improvement/am1/comparison.json).
- Run units: `C:/Users/yexia/Documents/GitHub/NN-kNN-haotang/runs/t1/am1/adaptation_improvement_v1/` (ignored, weights and TensorBoard records).
- Runtime logs: `../logs/local_am1_*.log`.
- Classification: iris, wine, breast_cancer, balance, digits, zebra, zebra_special.
- Regression: airfoil, car, yacht, energy_efficiency, student_performance, body_fat, diabetes (`diabets` in code).
- Seeds: 40, 41, 42. Frozen historical F configurations/budgets; new local paired observations. Diabetes uses frozen reuse; other tasks use their selected F schedule. The existing/new arms share pretrained core and Adam state; checkpoints use validation only.
- Scope: 14 of 29 historical tasks are present locally. The other 15 are not included in this local-data comparison. No r760 run or weight is used. Do not pool these runs with the report's historical averages.

Generated summaries remain under the T1 worktree's `results/` to follow the repository convention; this mini-project index links them. Per-run data and weights remain in the run store. Re-running completed units loads them without training again.

Historical adaptation evidence is linked in [References](../docs/REFERENCES.md); it is not a result of this proposed gating project.

## Learned disagreement cutoff: ag2 (November 25, 2026)

Status: **complete, 84/84 new units, 420 comparison rows, no missing units**. Two one-scalar gate variants are compared with all eight completed `ag1` methods. Mean classification error / standardized regression RMSE: frozen-adapter learned threshold **9.945% / 0.338856**; jointly trained threshold **9.359% / 0.347770**; validation-selected threshold **9.164% / 0.337424**. These previously inspected splits do not establish an improvement from making the cutoff trainable.

- [Full report](../code/neural-cbr/reports/ADAPTATION_THRESHOLD_LOCAL_2026-11-25.md), [pre-result protocol](../code/neural-cbr/docs/ADAPTATION_THRESHOLD_EXPERIMENT.md), [local launcher](../code/run_threshold_experiment.py).
- [Individual runs/cutoffs](../code/neural-cbr/results/adaptation_improvement/ag2/runs.csv), [task means](../code/neural-cbr/results/adaptation_improvement/ag2/tasks.csv), [family comparison](../code/neural-cbr/results/adaptation_improvement/ag2/families.csv), [full numerical evidence](../code/neural-cbr/results/adaptation_improvement/ag2/comparison.json).
- Run units: `C:/Users/yexia/Documents/GitHub/NN-kNN-haotang/runs/t1/ag2/threshold_v1/`.
- Runtime logs: `../logs/local_ag2_classification.log` and `../logs/local_ag2_regression.log`.
- All 252 saved train/validation/test prediction sets were checked; frozen proposals match the original exactly on all validation/test splits. Rerunning the full driver reused every completed unit without training.
