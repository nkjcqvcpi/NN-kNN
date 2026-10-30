# Pre-Top-K mask pilot, October 30, 2026

Exploratory retrieval-only comparison: 4 small datasets, 3 paired seeds (70-72), 24 fits. No tuning, reuse adapter, retention, or confirmatory claim.

Only `pre_topk_mask` changes between arms. `top_k=5` sets truncation in the masked arm; `bias_init_k=5` independently sets the nearest-neighbor bias initializer in both arms. Each pair starts with exactly identical model state and bias (SHA-256 checked).

60% train / 20% validation / 20% test; preprocessing uses train only; training uses identity leave-one-out; raw features, one feature-weight set, softmax tau=1, batch=64, at most 120 epochs, patience=20. Each arm selects its checkpoint by validation loss. Both training and evaluation use that arm's mask setting. CPU, one torch thread, deterministic algorithms.

Regression data generation/loading and standardization use `model.regression_workflow`; training uses the T1 `CoreConfig` path with `task_type=regression`. Errors are RMSE in training-standardized target units (regression), or 1 - accuracy (classification). Lower is better. SD is variation across three seeds, not a confidence interval.

| Dataset | Metric | Mask on, mean +/- SD | All cases, mean +/- SD | All-case better / worse / tied seeds |
| --- | --- | ---: | ---: | --- |
| linear_regression | RMSE | 0.3242 +/- 0.0646 | 0.2375 +/- 0.0309 | 3 / 0 / 0 |
| yacht | RMSE | 0.2687 +/- 0.1544 | 0.1771 +/- 0.0254 | 2 / 1 / 0 |
| energy_efficiency | RMSE | 0.0543 +/- 0.0057 | 0.0882 +/- 0.0047 | 0 / 3 / 0 |
| iris | Error rate | 0.0444 +/- 0.0192 | 0.0111 +/- 0.0192 | 2 / 0 / 1 |

![Paired held-out errors](comparison.png)

Thin lines connect the same seed. Diamonds show means. Panels have separate vertical scales. Three seeds and one common configuration do not establish a universal preference or replicate the historical tuned runs. No source default was changed.

All-case softmax lowers mean error on synthetic linear regression, Yacht, and Iris. Energy Efficiency favors masking on all three seeds; Yacht favors masking on one of three seeds. The findings are task dependent.

**Training-budget limit:** nine of twelve unmasked runs select epoch 120, the final allowed epoch; the remaining three select epochs 116-118. This short pilot does not establish convergence or a comparison at each arm's fully optimized performance.

Validation: all 24 errors independently recomputed from saved predictions; all 12 pairs have identical initial-state hashes and bias initialization; their configurations differ only in `pre_topk_mask`. The import smoke passes. The full T1 plus shared-core test run has 375 passes, one skip, and seven failures: three baseline-grid tests require unavailable CatBoost package metadata; three legacy synthetic tensor hashes and one exact legacy-output check differ in the local runtime. No shared core or existing test was edited. These failures prevent claiming a fully passing branch suite or historical bitwise reproduction. See [validation details](validation.json).

Reproduce from the repository root:

```powershell
.venv/Scripts/python.exe tools/t1_pre_topk_pilot.py
```

Configuration: [pilot YAML](../../../../configs/t1/pre_topk_pilot_20261030.yaml). Inspect [all run records](runs.json), [per-seed receipts](runs/), and [manifest](manifest.json) for source hashes, actual package versions, initialization, predictions, epochs, runtime, and neighborhood-support checks.
