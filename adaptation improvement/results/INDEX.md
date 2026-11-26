# Experiment results index

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
