# Dropout across NN-CDH adapters and training schemes

November 26, 2026. Study `ad1`, experiment `drop_v1`. **Complete: 756 new models and 378 reused zero-dropout references, covering 1,134 adapter/scheme/rate/dataset/seed cells.** Fourteen local datasets, seeds 40-42, CPU execution only.

## Findings

**Dropout helps particular combinations; it is not a general improvement across adapters and schemes.** The lowest fixed-setting classification error is **6.701%**, from query-aware **without** the prediction input, alternating training, and 20% dropout. Its matched zero-dropout result is 9.356%. However, the gain is concentrated on the small Zebra dataset. Across the other six classification datasets, error increases from **7.380% to 7.818%**. This is a promising local result, not an established general winner.

The lowest new fixed-setting regression RMSE is **0.337381**, from query-aware **with** the prediction input, alternating training, and 10% dropout, versus 0.345628 without dropout. The earlier probability gate remains marginally lower at **0.337206**; the disagreement gate is 0.337424. These differences are too small and the evidence too limited to establish a reliable ranking.

**The clearest improvement isolated to the adapter is original + sequential + dropout.** Retrieval is identical across rates. At 20% dropout, mean regression RMSE improves from **0.350429 to 0.338523**, with improvements on six of seven dataset means. At 10%, it is 0.339674, also improving six of seven. This is useful evidence for regularizing the original adapter even though it does not have the lowest overall mean.

Dropout does **not** eliminate extreme regression corrections. Query-without-prediction sequential models still produce very large wrong-direction updates and overshoots on Car. All such failures remain in the results. No global default is changed. [Complete evidence](../results/adaptation_improvement/ad1/comparison.json), [paired comparisons](../results/adaptation_improvement/ad1/comparisons.csv).

## Which adapter was best before this experiment?

The answer depends on metric and training scheme:

| Prior comparison | Lowest observed result |
| --- | --- |
| Classification, including validation-selected schemes | Query + prediction, scheme selected using validation: **8.745% error** |
| Classification, fixed scheme among the three current residual adapters | Query without prediction, alternating: **9.356%**; query + prediction, joint: **9.359%**, effectively tied |
| Regression, all earlier adapter/gate variants by arithmetic mean | Probability gate: **0.337206 RMSE** |
| Regression, simple disagreement gate | **0.337424 RMSE**; more favorable geometric comparison than the probability gate |
| Regression, ungated residual adapters | Original, previous mixed schedule: **0.338910**; original, all-joint schedule: **0.340452** |

The previous mixed schedule used sequential Diabetes and joint training for the other 13 tasks. It is not the all-joint condition. The magnitude/direction design remains shelved, and the earlier gate experiments remain historical comparisons; this new matrix covers the three current residual MLP input designs. [Training-scheme report](ADAPTER_TRAINING_SCHEMES_LOCAL_2026-11-26.md), [gate report](ADAPTATION_GATE_LOCAL_2026-11-25.md).

## What changed

Each active adapter has two hidden layers, widths 64 and 32. This experiment inserts independent dropout **after both hidden ReLU activations**, at rates 10% and 20%. Classification keeps LayerNorm and its tanh residual output; regression keeps its linear, unbounded residual output.

```text
Original:                 delta = A([query - neighborhood_mean, retrieved_prediction])
Query + prediction:       delta = A([query - neighborhood_mean, retrieved_prediction, query])
Query without prediction: delta = A([query - neighborhood_mean, query])

final_output = retrieved_prediction + delta
```

Dropout operates during adapter training, including warm-up, and is disabled for validation/test. There is no Monte Carlo inference. It adds no parameters: parameter keys, shapes, initial values and initialization RNG are preserved. A zero rate adds no modules. Existing classification support for first-hidden-only dropout remains compatible; these runs explicitly record `dropout_placement="all_hidden"`.

All rates reuse the same recorded data splits, pretrained retrieval model and optimizer state within each dataset/seed. Adapters start fresh. Sequential permanently freezes retrieval; alternating uses a retrieval pass followed by an adapter pass; joint updates both on each minibatch. In alternating's retrieval pass the adapter is in evaluation mode, so dropout is off. In joint training, the stochastic adapter also affects retrieval gradients. The effect of dropout on retrieval therefore differs across schemes.

Losses, widths, preprocessing and maximum budgets match the preceding `as1` experiment. Thirteen tasks retain five adapter warm-up passes, a 120-cycle maximum and patience 30. Diabetes sequential retains its original frozen trainer, combined loss, learning rate .001, 300-epoch maximum and patience 30; its alternating/joint runs retain the declared, untuned `as1` extension without warm-up. Early stopping means actual update counts can differ. See the [pre-result protocol](../docs/ADAPTER_DROPOUT_EXPERIMENT.md) and [executed/selected budgets](../results/adaptation_improvement/ad1/budgets.csv).

## All adapters, schemes and dropout rates

Each entry averages the three seeds within each dataset, then gives each of seven datasets equal weight. Lower is better. Classification is percentage error; regression is RMSE in training-standardized target units.

The last column selects the rate separately for each adapter/scheme/dataset/seed using **validation task loss**, then validation error, then the smaller rate for an exact tie. Test results do not select the rate or checkpoint. The lowest fixed test results are descriptive comparisons, not independently confirmed hyperparameter choices.

Classification error (%):

| Adapter | Scheme | 0% | 10% | 20% | Validation-selected rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Original | Sequential | 10.791 | 11.385 | 10.804 | 10.690 |
| Original | Alternating | 10.244 | 10.329 | 10.358 | 9.774 |
| Original | Joint | 9.945 | 11.631 | 11.018 | 10.123 |
| Query + prediction | Sequential | 10.980 | 10.803 | 11.693 | 11.457 |
| Query + prediction | Alternating | 9.502 | 10.170 | 11.367 | 9.303 |
| Query + prediction | Joint | 9.359 | 11.785 | 10.563 | 9.797 |
| Query without prediction | Sequential | 10.848 | 11.467 | 11.697 | 10.728 |
| Query without prediction | Alternating | 9.356 | 9.504 | **6.701** | **6.453** |
| Query without prediction | Joint | 10.146 | 9.692 | 11.561 | 9.984 |

Standardized regression RMSE:

| Adapter | Scheme | 0% | 10% | 20% | Validation-selected rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Original | Sequential | 0.350429 | 0.339674 | 0.338523 | **0.341434** |
| Original | Alternating | 0.342507 | 0.340658 | 0.340696 | 0.343243 |
| Original | Joint | 0.340452 | 0.348474 | 0.339718 | 0.348355 |
| Query + prediction | Sequential | 0.371033 | 0.367750 | 0.343767 | 0.344494 |
| Query + prediction | Alternating | 0.345628 | **0.337381** | 0.348484 | 0.344081 |
| Query + prediction | Joint | 0.345530 | 0.347068 | 0.350272 | 0.343444 |
| Query without prediction | Sequential | 0.415262 | 0.394397 | 0.457686 | 0.460822 |
| Query without prediction | Alternating | 0.346292 | 0.352826 | 0.348959 | 0.347563 |
| Query without prediction | Joint | 0.359316 | 0.357581 | 0.359795 | 0.357744 |

The regression validation policy does not reproduce the retrospectively lowest fixed rate. For query + prediction alternating, it selects 0% / 10% / 20% in 11 / 5 / 5 runs and reaches 0.344081, not 0.337381. For classification query without prediction alternating, the corresponding counts are 8 / 4 / 9 and error is 6.453%.

![Fixed dropout rates across all three adapters and schemes](../results/adaptation_improvement/ad1/dropout_means.png)

[Fixed family means](../results/adaptation_improvement/ad1/families.csv), [validation-rate means](../results/adaptation_improvement/ad1/selected_rate_families.csv), [all dataset means](../results/adaptation_improvement/ad1/tasks.csv), [rate selections](../results/adaptation_improvement/ad1/selected_rates.csv).

## Selecting both rate and scheme

This second predeclared policy selects among all nine rate/scheme combinations for each adapter/dataset/seed. It uses validation task loss, validation error, smaller rate, then sequential/alternating/joint tie order. Its appropriate reference is the previous validation-selected scheme at zero dropout.

| Adapter | Classification: old scheme selection | Classification: rate + scheme | Regression: old scheme selection | Regression: rate + scheme |
| --- | ---: | ---: | ---: | ---: |
| Original | 9.998% | 9.817% | 0.349431 | 0.342507 |
| Query + prediction | **8.745%** | 9.378% | 0.345186 | 0.342510 |
| Query without prediction | 9.303% | **7.053%** | 0.409628 | 0.458242 |

A larger validation search does not reliably improve test results. Query + prediction loses its previous classification advantage. Query without prediction's classification mean improves, but it improves only one dataset mean, ties one and worsens five versus its prior scheme-selection policy; Zebra explains the arithmetic gain. Its regression policy selects severe Car failures.

[Joint selections](../results/adaptation_improvement/ad1/selected_both.csv), [joint-selection means](../results/adaptation_improvement/ad1/selected_both_families.csv), [selection counts](../results/adaptation_improvement/ad1/selection_counts.csv).

## Where the classification gain comes from

Query without prediction, alternating training:

| Dataset | 0% dropout | 20% dropout | Validation-selected rate |
| --- | ---: | ---: | ---: |
| Iris | 3.333% | 5.556% | 4.444% |
| Wine | 4.630% | 3.704% | 3.704% |
| Breast Cancer | 2.339% | 2.339% | 3.216% |
| Balance | 5.333% | 5.333% | 5.067% |
| Digits | 2.130% | 1.944% | 2.222% |
| Zebra | 21.212% | 0.000% | 0.000% |
| Zebra Special | 26.515% | 28.030% | 26.515% |
| All seven, equal weight | 9.356% | 6.701% | 6.453% |
| Six excluding Zebra, sensitivity only | 7.380% | 7.818% | 7.528% |

At 20%, task wins/ties/losses are 3/2/2 and seed wins/ties/losses are 8/8/5. The geometric mean error ratio is 1.04294 **over six positive-error task pairs only**: Zebra's zero new error makes its log ratio undefined, so it is excluded from this ratio. It remains in every primary arithmetic mean and count. The selected-rate policy has task counts 3/1/3 and a similarly restricted GMR of 1.06436. These ratios must not be presented as seven-task summaries.

Zebra has **22 test examples per seed**, or 66 evaluations across the three seeds. At 20% dropout, all three runs already have zero retrieval error before adaptation. The most dramatic change is seed 42:

| Zebra seed 42 | 0% dropout | 20% dropout |
| --- | ---: | ---: |
| Executed epochs | 36 | 120 |
| Selected epoch | 5, during warm-up | 120 |
| Selected validation loss | 2.572660 | 0.000458 |
| Test retrieval errors | 14 / 22 | 0 / 22 |
| Test final errors | 13 / 22 | 0 / 22 |

Seed 40 also changes from one final error to zero; seed 41 has zero in both. Thus the aggregate Zebra improvement is 14 fewer errors among 66 evaluations. It involves a different training trajectory, more executed updates and a much better selected retrieval core. It cannot be attributed solely to better final corrections. The small dataset's equal weight has a large influence on the family mean. [Individual model rows](../results/adaptation_improvement/ad1/rows.csv).

## Regression: average gain and consistency

| Dataset | Query + prediction alternating, 0% | Same, 10% | Original sequential, 0% | Same, 20% |
| --- | ---: | ---: | ---: | ---: |
| Airfoil | 0.211789 | 0.211277 | 0.212826 | 0.211313 |
| Car | 0.273611 | 0.219902 | 0.265151 | 0.215085 |
| Yacht | 0.050584 | 0.054735 | 0.058287 | 0.063068 |
| Energy Efficiency | 0.046202 | 0.047848 | 0.051859 | 0.050595 |
| Student Performance | 0.439571 | 0.432541 | 0.445264 | 0.433240 |
| Body Fat | 0.624921 | 0.626038 | 0.650757 | 0.627923 |
| Diabetes | 0.772718 | 0.769329 | 0.768861 | 0.768434 |
| Equal-task mean | 0.345628 | **0.337381** | 0.350429 | **0.338523** |

Query + prediction alternating at 10% wins four dataset means and loses three; GMR is 0.98219. At the seed level it wins 8 and loses 13: the favorable mean is driven by larger gains, especially Car. Validation selects the zero-dropout Car model in all three seeds because its validation loss is lower, despite the better test mean at 10%. This illustrates why the fixed-rate test winner and the validation policy differ.

Original sequential at 20% wins six dataset means and loses Yacht; GMR is 0.96825, with 12 seed wins and 9 losses. Its validation-selected rate still improves five of seven task means, reaching 0.341434 with GMR 0.97071. Because retrieval remains exactly fixed, this is stronger evidence that dropout itself improves the learned adapter on this local regression comparison. It remains a three-seed exploratory result.

## Retrieval quality and correction behavior

Compare each selected full model with **its own retrieval prediction**, rather than assuming retrieval is unchanged:

| Configuration | Mean error before adaptation | Mean error after adaptation |
| --- | ---: | ---: |
| Classification, no prediction, alternating, 0% | 10.298% | 9.356% |
| Classification, no prediction, alternating, 20% | 8.177% | 6.701% |
| Regression, query + prediction, alternating, 0% | 0.356967 | 0.345628 |
| Regression, query + prediction, alternating, 10% | 0.349371 | 0.337381 |
| Regression, original, sequential, 0% | 0.348690 | 0.350429 |
| Regression, original, sequential, 20% | 0.348690 | 0.338523 |

For query + prediction alternating, most of the 0.008247 improvement in final RMSE accompanies a 0.007596 improvement in its selected retrieval RMSE. The mean before-to-after adaptation benefit changes only from 0.011339 to 0.011990. This is an arithmetic decomposition of the selected models, not a counterfactual causal test.

Classification no-prediction alternating fixes **15** and harms **10** decisions without dropout; at 20%, it fixes **19** and harms **8**. These are pooled over 2,193 evaluations across seeds. They are not equal-task percentages or independent unique examples.

Regression query + prediction alternating, pooled over 6,339 evaluations:

| Diagnostic | 0% dropout | 10% dropout |
| --- | ---: | ---: |
| Wrong-direction corrections | 2,439 (38.48%) | 2,813 (44.38%) |
| Harmful overshoots | 525 (8.28%) | 445 (7.02%) |
| Cases with lower squared error | 3,254 (51.33%) | 2,919 (46.05%) |
| Cases with higher squared error | 3,083 (48.64%) | 3,416 (53.89%) |
| Sum of positive squared-error increases | 184.714 | 45.565 |
| Portion of that sum from harmful overshoots | 139.717 | 3.633 |

Here a wrong direction means the correction has the opposite sign to the needed residual. A harmful overshoot goes in the right direction but exceeds twice the required residual magnitude, leaving more error than retrieval alone. Corrections that move an already exact prediction are counted separately. The diagnostic uses an absolute tolerance of 1e-8; tiny squared-error changes can be neither improved nor worsened.

**The better RMSE does not mean more frequently correct decisions.** The main favorable diagnostic is a large reduction in overshoot severity, despite more wrong-direction corrections. Original sequential shows a similar severity benefit: its total positive squared-error increase falls from 214.968 to 74.541 at 20%, and the overshoot contribution falls from 126.878 to 21.051. Pooled squared-error totals weight large datasets more heavily and are not the family RMSE metric. [Per-run diagnostics](../results/adaptation_improvement/ad1/behavior.csv), [pooled counts and severity](../results/adaptation_improvement/ad1/behavior_totals.csv).

## Failures retained in the comparison

Query without prediction, sequential, 20% dropout remains unstable on Car:

| Run / test query index | Required correction | Applied correction | Type | Share of that run's positive adaptation harm |
| --- | ---: | ---: | --- | ---: |
| Seed 40 / 1128 | -0.579325 | +60.490190 | Wrong direction | 99.893% |
| Seed 42 / 545 | +0.154573 | +46.456995 | Harmful overshoot, about 301 times needed | 99.695% |

For seed 40, retrieval RMSE is 0.206827 but final RMSE is **1.686933**. For seed 42, it changes from 0.232915 to **1.288148**. These queries were neither clipped nor removed. Similar extreme failures already existed at zero dropout. Dropout regularizes hidden activations during training; the regression output remains unbounded, so it supplies no guarantee against extreme inference-time corrections. [Largest harmful cases](../results/adaptation_improvement/ad1/harm_examples.csv).

## Interpretation and next decision

Keep dropout available as an explicit adapter hyperparameter. The current evidence supports confirming original sequential with dropout for regression, and checking the two lowest-mean configurations on fresh splits before treating them as preferred defaults. Classification's Zebra effect and regression's Car effect make universal claims premature.

The predeclared validation policies are more relevant to an actual tuning procedure than choosing the lowest test column retrospectively. Their results are mixed, including deterioration for query + prediction classification and query-without-prediction regression when both rate and scheme are selected. Adding dropout has not resolved correction reliability or established a single best adapter for both task families.

These are previously examined local datasets with only three seeds, not the full historical benchmark or an independent confirmation set. Rates .1 and .2 are a small search. Different early-stopping trajectories mean unequal actual compute. No significance or equal-budget comparison with conventional baselines is claimed.

## Implementation, evidence and verification

Implementation: [dropout helper](../model/t1/adapter_regularization.py), [synchronized trainer](../model/t1/sync.py), [frozen regression trainer](../model/t1/reuse.py), [recorded experiment driver](../tools/t1_adapter_dropout.py), [local uv launcher](../../run_dropout_experiment.py), [focused tests](../tests/test_t1_adapter_dropout.py).

Training ran in disjoint jobs with eight single-thread CPU workers. Complete units are detected before fitting. Incomplete units retain resume states including the shuffle and dropout RNG. The shared NN-kNN core, Table 1 and original defaults remain unchanged.

Verification completed:

- All **1,134** cells are present and unique; **756** are new and **378** reuse the exact previous zero-dropout units. All zero-dropout rows and validation-selected zero-dropout policies match `as1`.
- Reconciled **2,226 saved validation/test prediction sets** with reported metrics. Forty-two older original references lack saved validation prediction arrays; their recorded validation metrics are reused. This explains the difference from 2,268 possible sets.
- Verified **1,512 initial retrieval prediction sets** for the new runs against their shared references, **378 frozen-core hashes**, and **336 exact cross-adapter/rate sequential prediction matches**.
- Verified unchanged trainable parameter counts and declared two-layer dropout metadata for all **756** new models; repeated evaluation is deterministic.
- Independently recomputed **84 family aggregates**, checked **630 validation-selected policy rows** against their candidates, and reconciled the task/seed counts in **60 paired comparisons**.
- Re-entered all **1,134** complete comparison cells with both training functions replaced by assertions. Every cell was reused; the audit ends with `VERIFIED_COMPLETE_CELL_REUSE 1134`.
- The targeted dropout/synchronization/query suites passed **109 tests**; the subsequently added validation-only rate-selection test passed separately. Tests cover both tasks, all three input variants and schemes, dropout train/eval behavior, unchanged initialization and parameter keys, frozen retrieval, legacy placement, and exact interruption recovery.
- Shared-core manifest checks passed in both worktrees. Whitespace, artifact links and the generated figure were checked.

Recorded curves, weights, resume states and predictions are under `NNKNN_RUNS/t1/ad1/drop_v1/`, with references to `as1`, `am1`, `aq1` and `aq2`. Generated evidence is under [results/adaptation_improvement/ad1](../results/adaptation_improvement/ad1/). Worker logs are in the mini-project `logs/ad1/`; the reuse audit is `logs/ad1_reuse_check.log`.

Using the established local uv environment, run the launcher from the mini-project `code/` directory:

```powershell
& ../.tools/uv.exe run --project ../.. --no-sync python run_dropout_experiment.py --study ad1 --mode summarize
```

Omitting `--mode summarize` checks/reuses complete models before regenerating the summaries. No r760 access is required.
