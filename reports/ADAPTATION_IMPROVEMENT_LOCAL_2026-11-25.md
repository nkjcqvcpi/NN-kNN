# Local comparison of magnitude/direction adaptation

November 25, 2026. Study `am1`. **Complete: 42/42 paired dataset/seed comparisons, 14 local datasets, seeds 40-42.**

## Finding

The first magnitude/direction implementation does **not** improve overall on the existing adapter in this local comparison. It has lower mean error on 7 of 14 datasets and higher mean error on the other 7. Improvements on several classification datasets are offset by larger losses on Iris and Zebra. Regression is slightly worse overall, with a substantial loss on Yacht.

| Family | Datasets / seed pairs | Better / tie / worse by task mean | Geometric mean new/existing error | Average task error change |
| --- | --- | --- | --- | --- |
| Classification | 7 / 21 | 5 / 0 / 2 | 1.03683 (3.68% worse) | +1.196 percentage points |
| Regression | 7 / 21 | 2 / 0 / 5 | 1.03032 (3.03% worse) | +0.002037 standardized RMSE |

The geometric mean is computed from each task's three-seed mean error, with equal task weight. There are no zero-error exclusions in these final task means. Classification and regression are not pooled into one score. At the individual seed level, classification has 9 better / 6 ties / 6 worse, and regression 10 better / 0 ties / 11 worse. These are descriptive counts, not statistical significance claims. Tiny differences such as Energy Efficiency should not be treated as established losses.

Keep the existing adapter as the default. The new design remains an opt-in experimental implementation. Its scale interpretation is mathematically cleaner, but that alone did not produce a general predictive improvement.

## What was implemented

`final = retrieval + magnitude * unit_direction`, with a shared NN-CDH trunk, a raw residual direction proposal, and a softplus magnitude head using the shared representation plus nearest distance, activation-weighted mean distance, activation concentration and target disagreement. The magnitude head starts at 0.01. Its output is an actual correction norm, not confidence or a fraction of a separate correction.

Raw residual supervision trains scalar direction despite hard-sign normalization. Final prediction loss trains the applied correction. Both heads train/freeze together in independent, alternating and joint schedules. Training statistics use stable-ID leave-one-out neighborhoods; validation/test targets never enter the adapter inputs.

See [full design and training notes](../docs/ADAPTATION_MAGNITUDE_DIRECTION.md), [implementation](../model/t1/magnitude_direction.py), [reuse](../model/t1/reuse.py), [synchronization](../model/t1/sync.py), and [experiment runner](../tools/t1_adaptation_improvement.py).

## Per-dataset results

All numbers are three-seed means; lower is better. "Historical F" is the rounded value in [the November 24 report, section 2](NEURAL_CBR_WEEKLY_2026-11-24.md#2-reproduction-of-the-frozen-final-configuration-f-29-tasks). It is context only. The decisive comparison is between the **local existing** and **local new** columns, which share a pretrained core, optimizer state, starting RNG and data split within each seed. "Local retrieval" is that original pretrained core; jointly trained before-adapter values are separately retained in the CSV evidence.

Classification error rate:

| Dataset | Historical F | Local retrieval | Local existing | Local new | New vs existing |
| --- | ---: | ---: | ---: | ---: | --- |
| Iris | 0.0333 | 0.07778 | 0.03333 | 0.07778 | worse |
| Wine | 0.0648 | 0.03704 | 0.07407 | 0.05556 | better |
| Breast Cancer | 0.0292 | 0.02339 | 0.02924 | 0.01754 | better |
| Balance | 0.0480 | 0.05867 | 0.05333 | 0.04800 | better |
| Digits | 0.0241 | 0.02222 | 0.02130 | 0.02037 | better |
| Zebra | 0.182 | 0.31818 | 0.18182 | 0.28788 | worse |
| Zebra Special | 0.303 | 0.28030 | 0.30303 | 0.27273 | better |

Regression RMSE in training-standardized target units:

| Dataset | Historical F | Local retrieval | Local existing | Local new | New vs existing |
| --- | ---: | ---: | ---: | ---: | --- |
| Airfoil | 0.215 | 0.21337 | 0.20809 | 0.21026 | worse |
| Body Fat | 0.627 | 0.62530 | 0.62749 | 0.62375 | better |
| Car | 0.244 | 0.22683 | 0.22288 | 0.23002 | worse |
| Diabetes | 0.769 | 0.76579 | 0.76886 | 0.76561 | better |
| Energy Efficiency | 0.0480 | 0.05253 | 0.04769 | 0.04790 | slightly worse |
| Student Performance | 0.439 | 0.44202 | 0.44103 | 0.44231 | slightly worse |
| Yacht | 0.0563 | 0.11499 | 0.05633 | 0.06678 | worse |

## What the correction diagnostics show

Follow-up: the [complete adaptation behavior report](ADAPTATION_BEHAVIOR_2026-11-25.md) analyzes every saved test correction, separates insufficient scale from wrong direction and harmful overshoot, and decomposes the contribution of retrieval-core changes. Its calculations do not retrain or alter these results.

- **Iris loses useful adaptation.** The existing adapter fixes three test cases across the three seeds without harmful flips. The new adapter changes no test labels. Its mean magnitudes are 0.01371, 0.01250 and 0.01065, close to initialization, with selected epochs 6, 4 and 2. The jointly trained cores also differ, so the final error difference cannot be attributed solely to the final correction.
- **Breast Cancer avoids harm.** The existing adapter makes one helpful and four harmful flips; the new adapter makes one helpful and zero harmful flips. The new magnitude is about 0.013 on seeds 41/42, but 0.298 on seed 40. It can learn a substantially larger correction where validation favors one.
- **Body Fat has a small benefit with a small correction.** Both arms have the same before-adapter RMSE of 0.62530. Existing final RMSE is 0.62749; new is 0.62375, about 0.60% lower than existing. Mean correction magnitude is about 0.01063 versus existing mean absolute correction 0.05025.
- **Yacht still benefits from adaptation, but less than before.** The new adapter improves its own jointly trained core on all three seeds, with mean magnitudes 0.044-0.079. Its final mean RMSE is nonetheless 18.55% above the existing adapter's. Small corrections are not universally preferable.

Small scales on Iris and Body Fat are **not evidence that confidence was learned**. Conservative initialization and validation checkpoint selection could produce the same pattern. No ablation here isolates that initialization from neighborhood statistics or head separation. A useful next experiment would predeclare magnitude initialization and validation-selection ablations on development data, rather than tune against these test results.

## Protocol and limits

- The owner requested local runs only because r760 is unavailable. Fourteen of the historical 29 tasks had data available locally. The other 15 were not run or silently replaced with synthetic datasets. No remote experiment host was used and no dataset was downloaded for this suite.
- Source configuration is each task's frozen F from `configs/t1/confirm_final_20261013`, with seeds 40, 41, 42. Diabetes uses frozen-retrieval reuse; the other tasks use their selected F schedule. Core maximum budget is 300 epochs; synchronization maximum is 120; frozen reuse maximum is 300, with original per-task early stopping and optimizer settings.
- One pretrained retrieval core is recorded per dataset/seed and copied, with its optimizer state, into both arms. Each arm restores the same starting RNG. Train/validation/test tensor fingerprints are checked and recorded. Completed units are reused, never retrained.
- The primary comparison retains all historical loss coefficients, including task-specific correction-size penalties, so removal of that penalty is not mixed into this experiment.
- This compares the **complete proposed design**: head separation, an unbounded raw classification proposal rather than tanh-bounded residual, raw-proposal auxiliary supervision, neighborhood evidence and magnitude initialization. It does not isolate topology alone.
- Frozen reuse retains its combined validation criterion. For historical regression that was equivalent to scaled final MSE; the new raw-proposal loss is distinct. Synchronized runs use their configured final-task validation criterion.
- These three seeds were previously seen, the suite is a local subset, and the historical host/recorded cores were unavailable. The result is exploratory local evidence, not confirmation on fresh seeds or reproduction of the historical numbers.

## Validation and evidence

All 49 new architecture, schedule, recorded-run, frozen-reuse, nominal-input and logging tests passed. Independent code and numerical reviews found no blocking defect and verified every report table against the recorded summaries.

Full branch suite: **558 passed, 1 failed, 5 skipped, 2 expected failures** in 11m37s. The sole failure is `test_defaults_reproduce_the_recorded_toy_outputs`: the Linux-recorded legacy MLP/logit scenario expects 0.2433512657880783, whereas this local runtime gives 0.24344214471057057 at `sync_cls_joint_mlp_logit.model[8]`. Running that test on an untouched archive of original commit `411abd4` reproduced exactly the same failure and value. This is a pre-existing local numerical-reference mismatch, not introduced by the new adapter. The reference values and tolerance were not changed.

- [Full precision task means](../results/adaptation_improvement/am1/tasks.csv)
- [All 42 seed pairs, unit IDs and fingerprints](../results/adaptation_improvement/am1/pairs.csv)
- [Coverage and aggregate metrics](../results/adaptation_improvement/am1/comparison.json)
- Raw run units and weights: `NNKNN_RUNS/t1/am1/adaptation_improvement_v1/`; on this machine, `C:/Users/yexia/Documents/GitHub/NN-kNN-haotang/runs/t1/am1/adaptation_improvement_v1/`.
- Mini-project launcher: `adaptation improvement/code/run_local_comparison.py`; use `--mode summarize` for a read-only rebuild or `--mode run` to load completed units. This launcher uses uv and does not change PowerShell execution policy.
- Source is the separate `adaptation-improvement` worktree based on `origin/neural-cbr` commit `411abd4`. No shared core or Table 1 code was changed. New adapters require T1 retrieval-aware prediction; post-removal continued adaptation is explicitly unsupported in this first implementation.
