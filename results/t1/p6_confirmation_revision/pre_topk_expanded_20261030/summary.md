# Pre-Top-K mask comparison, October 30, 2026

Exploratory retrieval-only comparison: 12 datasets, 3 paired seeds [70, 71, 72], 72 fits. No tuning, reuse adapter, retention, or confirmatory claim.

Only `pre_topk_mask` changes between arms. `top_k=5` sets truncation in the masked arm; `bias_init_k=5` independently sets the nearest-neighbor bias initializer in both arms. Each pair starts with exactly identical model state and bias (SHA-256 checked).

60% train / 20% validation / 20% test; preprocessing uses train only; training uses identity leave-one-out; raw features, one feature-weight set, softmax tau=1.0, batch=64, at most 120 epochs, patience=20. Each arm selects its checkpoint by validation loss. Both training and evaluation use that arm's mask setting. CPU, one torch thread, deterministic algorithms.

Regression data generation/loading and standardization use `model.regression_workflow`; training uses the T1 `CoreConfig` path with `task_type=regression`. Errors are RMSE in training-standardized target units (regression), or 1 - accuracy (classification). Lower is better. SD is variation across three seeds, not a confidence interval.

| Dataset | Metric | Mask on, mean +/- SD | All cases, mean +/- SD | Relative change in mean error | All-case better / worse / tied seeds |
| --- | --- | ---: | ---: | ---: | --- |
| linear_regression | RMSE | 0.3242 +/- 0.0646 | 0.2375 +/- 0.0309 | -26.8% | 3 / 0 / 0 |
| yacht | RMSE | 0.2687 +/- 0.1544 | 0.1771 +/- 0.0254 | -34.1% | 2 / 1 / 0 |
| energy_efficiency | RMSE | 0.0543 +/- 0.0057 | 0.0882 +/- 0.0047 | +62.3% | 0 / 3 / 0 |
| iris | Error rate | 0.0444 +/- 0.0192 | 0.0111 +/- 0.0192 | -75.0% | 2 / 0 / 1 |
| wine | Error rate | 0.0463 +/- 0.0424 | 0.0093 +/- 0.0160 | -80.0% | 2 / 0 / 1 |
| breast_cancer | Error rate | 0.0409 +/- 0.0332 | 0.0292 +/- 0.0134 | -28.6% | 2 / 1 / 0 |
| balance | Error rate | 0.1280 +/- 0.0288 | 0.1253 +/- 0.0514 | -2.1% | 2 / 1 / 0 |
| digits | Error rate | 0.0185 +/- 0.0016 | 0.0157 +/- 0.0070 | -15.0% | 2 / 1 / 0 |
| airfoil | RMSE | 0.3848 +/- 0.0061 | 0.4156 +/- 0.0089 | +8.0% | 0 / 3 / 0 |
| student_performance | RMSE | 0.5343 +/- 0.1140 | 0.4175 +/- 0.1055 | -21.9% | 3 / 0 / 0 |
| mixture_two_linear_models | RMSE | 0.8350 +/- 0.1217 | 0.7976 +/- 0.1416 | -4.5% | 3 / 0 / 0 |
| redundant_features | RMSE | 0.1187 +/- 0.0161 | 0.0951 +/- 0.0080 | -19.9% | 3 / 0 / 0 |

All-case softmax has lower mean test error on 10 of 12 datasets: 5 of 7 regression tasks and all 5 classification tasks. It improves 24 of 36 paired task-seed errors, worsens 10, and ties 2. Energy Efficiency and Airfoil favor masking on every seed. Negative relative change favors all-case softmax; it is computed from unrounded means.

The expanded batch adds eight datasets and reruns the four original tasks. All 24 original fits reproduce exactly; the repeated original fits are not additional independent evidence. Student Performance uses a training-fitted category vocabulary followed by training-only feature/target standardization. This common untuned protocol differs from the older tuned experiments.

Training-budget caveat: 22 of 36 unmasked runs select the final allowed epoch (120). This does not establish convergence or each arm's fully optimized performance.

![Paired held-out errors](comparison.png)

Thin lines connect the same seed. Diamonds show means. Panels have separate vertical scales. 3 seeds and one common configuration do not establish a universal preference or replicate the historical tuned runs. No source default was changed.

Reproduce from the repository root:

```powershell
.venv/Scripts/python.exe tools/t1_pre_topk_pilot.py --config configs/t1/pre_topk_expanded_20261030.yaml --out <new-output-directory>
```

Configuration: [experiment YAML](../../../../configs/t1/pre_topk_expanded_20261030.yaml). Inspect [all run records](runs.json), [per-seed receipts](runs/), and [manifest](manifest.json) for source hashes, actual package versions, initialization, predictions, epochs, runtime, and neighborhood-support checks. The [runner snapshot](runner_at_execution.py) preserves execution code; [render_comparison.py](render_comparison.py) renders the paired figure with additional marker headroom.

Independent verification: [validate_results.py](validate_results.py) recomputes every test metric and summary statistic, checks the paired initializations and configurations, verifies split disjointness, and checks exact reproduction of the original pilot. [validation.json](validation.json) records 72 checked fits and 36 identical-initialization pairs. Import smoke passes; the full T1 plus shared-core suite reports 375 passed, 7 failed, and 1 skipped, with the same three missing-CatBoost-metadata and four exact historical-output failures as the initial pilot. Shared core source and existing tests were not changed.
