# Query-aware NN-CDH without the retrieved prediction

November 25, 2026. Study `aq2`, experiment `nopred_v1`. **Complete: 42 new model units, 14 local datasets, seeds 40-42, 336 comparison rows and no missing units.** The original, query-aware and conventional baseline models are reused from `am1`/`aq1`.

## Findings

**Removing the prediction input blocks direct retrieval subtraction, but does not give a general performance improvement.** Mean classification error is **10.146%**, compared with **9.359%** for query-aware with the prediction input and **9.945%** for the original full model. Regression RMSE is **0.358695**, worse than **0.345176** and **0.338910**, respectively. The original default is retained; the new variant is available as an experimental option. [Complete numerical evidence](../results/adaptation_improvement/aq2/comparison.json).

Classification has a real trade-off: the new version wins on five of seven tasks against the original and has a better geometric mean error ratio. The larger Zebra loss makes its arithmetic mean worse. Its correction also makes fewer harmful class flips, but joint training selects weaker retrieval predictions. Regression is less ambiguous: removing the input worsens both aggregate measures and produces a severe Car overcorrection.

The strongest diagnostic is **correction size, not direct subtraction**. On one Car seed-40 test case, the correct residual was +0.124 but the new adapter applied +21.20 in standardized target units. That single case contributes 98.6% of the sum of positive squared-error increases caused by adaptation in that run. It remains included in every result.

## Exact design

```text
Original:                     delta = A([q - weighted_case_mean, retrieved_y])
Query-aware, aq1:              delta = A([q - weighted_case_mean, retrieved_y, q])
Query-aware without y_r, aq2:  delta = A([q - weighted_case_mean, q])

All versions: final_output = retrieved_y + delta
```

The new residual network has **no retrieved-prediction input columns**. Scalar regression removes one input; classification removes the entire class-probability vector. Retrieval is still used for the final additive connection and the residual training target. The adapter wrapper accepts it for those operations, but never passes it into the residual network's hidden layers.

This retains the original regression delta output and classification tanh-bounded class-score residual, hidden layers, losses and training schedules. It adds no confidence, gate or magnitude/direction head. Set `adapter_design="query_aware_no_prediction"`; see the [implementation](../model/t1/query_aware.py) and [pre-result protocol](../docs/QUERY_NO_PREDICTION_EXPERIMENT.md).

Initialization constructs the same query-aware network, then deletes only its prediction columns. All remaining weights, biases, zero query columns and RNG state match. The new initial function equals the old initial function with its prediction input zeroed. Initial normal outputs and parameter counts therefore differ: the reduction is `prediction_dimension * first_hidden_width`, verified for every completed model.

At fixed query/difference features, the structural restriction is:

```text
delta(q, difference, changed_retrieval) = delta(q, difference, original_retrieval)
change_in_final_output = change_in_retrieved_prediction
```

We verified this on every test set. The largest observed residual difference was 2.38e-7, and the largest final-output translation discrepancy was 4.77e-7, consistent with float32 roundoff. Tests also verify that no gradient path leads from the prediction input into the residual. Correlated query/neighborhood features can still support an implicit estimate of retrieval, and a large correction can dominate the final output. The guarantee applies while those features are fixed; joint training changes them.

## Matched training and comparison

The new models start from the same recorded `am1` pretrained core and optimizer as `aq1`, with fresh adapters. All 42 cached split fingerprints and validation/test targets match. Thirteen tasks retain their selected joint schedule, five adapter warm-up epochs, 120-epoch maximum and patience 30. Diabetes retains frozen retrieval, 300 epochs maximum and patience 30. Original loss coefficients, learning rates, free-radius settings and validation checkpoint selection are preserved. Test data never select a setting or checkpoint.

No completed baseline or source core was retrained. The MLP/TabM/CatBoost/kNN rows below are the compact local searches from `aq1`, using the same preprocessed splits. Their smaller search and training budgets do not reproduce the unavailable historical tuned winners or establish equal-budget neural parity. These repeatedly inspected seeds provide exploratory evidence.

## Overall comparison

Means give each of seven datasets equal weight after averaging its three seeds. Classification is percent error; regression is standardized RMSE. Lower is better.

| Model | Classification error % | Regression RMSE |
| --- | ---: | ---: |
| Pretrained NN-kNN retrieval | 11.680 | 0.348690 |
| Original full NN-kNN | 9.945 | 0.338910 |
| Query-aware, with prediction input | 9.359 | 0.345176 |
| **Query-aware, without prediction input** | **10.146** | **0.358695** |
| MLP, compact local search | 13.922 | 0.390093 |
| TabM, compact local search | 14.259 | 0.389783 |
| CatBoost, compact local search | 4.517 | 0.318980 |
| kNN, compact local search | 14.989 | 0.496592 |

The new variant's classification mean is 0.788 percentage points worse than query-aware with the prediction, and 0.202 points worse than the original. Regression is also worse than the standalone pretrained retrieval baseline on this aggregate.

| New variant compared with | Classification wins / ties / losses | Classification GMR | Regression wins / ties / losses | Regression GMR |
| --- | ---: | ---: | ---: | ---: |
| Original full model | 5 / 0 / 2 | 0.89108 | 3 / 0 / 4 | 1.06721 |
| Query-aware with prediction | 3 / 1 / 3 | 0.91477 | 2 / 0 / 5 | 1.03877 |

GMR is the geometric mean of task-error ratios: below one favors the new variant. These ratios include all seven tasks. Classification's relative gains, especially on Wine, improve GMR even though the larger absolute Zebra loss worsens mean error. Neither aggregation should be omitted. [Family means](../results/adaptation_improvement/aq2/families.csv), [comparisons with every baseline](../results/adaptation_improvement/aq2/comparisons.csv).

![Comparison of the three full NN-kNN models](../results/adaptation_improvement/aq2/query_no_prediction_comparison.png)

## Dataset means

Classification error percentages:

| Dataset | Original | Query + prediction | Query without prediction |
| --- | ---: | ---: | ---: |
| Iris | 3.333 | 5.556 | 4.444 |
| Wine | 7.407 | 6.481 | 2.778 |
| Breast Cancer | 2.924 | 2.924 | 2.339 |
| Balance | 5.333 | 4.000 | 5.067 |
| Digits | 2.130 | 1.852 | 1.852 |
| Zebra | 18.182 | 18.182 | 27.273 |
| Zebra Special | 30.303 | 26.515 | 27.273 |

Standardized regression RMSE:

| Dataset | Original | Query + prediction | Query without prediction |
| --- | ---: | ---: | ---: |
| Airfoil | 0.208094 | 0.210219 | 0.212137 |
| Car | 0.222880 | 0.271599 | 0.359661 |
| Yacht | 0.056331 | 0.056875 | 0.058222 |
| Energy Efficiency | 0.047689 | 0.046816 | 0.044040 |
| Student Performance | 0.441026 | 0.440538 | 0.448012 |
| Body Fat | 0.627485 | 0.624921 | 0.621680 |
| Diabetes | 0.768861 | 0.765261 | 0.767113 |

Energy Efficiency and Body Fat improve against both preceding versions. Car is substantially worse; all three Car seeds are worse than their corresponding `aq1` results. The individual observations and full conventional-baseline comparison remain in [all 336 rows](../results/adaptation_improvement/aq2/runs.csv) and [task means](../results/adaptation_improvement/aq2/tasks.csv).

## Retrieval versus the correction

Each row compares a selected full model's own retrieval output with its final output. These cores differ because training is joint on thirteen tasks.

| Family / model | Own retrieval error | After adaptation |
| --- | ---: | ---: |
| Classification, original | 10.385% | 9.945% |
| Classification, query + prediction | 9.916% | 9.359% |
| Classification, query without prediction | 12.018% | 10.146% |
| Regression, original | 0.352137 | 0.338910 |
| Regression, query + prediction | 0.350032 | 0.345176 |
| Regression, query without prediction | 0.349248 | 0.358695 |

The new classification correction has a larger immediate benefit, **1.871 percentage points**, but its selected retrieval model is worse. Across 2,193 test evaluations, it fixes **23** retrieval mistakes and creates **10**, versus 30/26 for query-aware with prediction and 25/24 for the original. This shows a useful classification trade-off; it does not isolate the adapter input's causal effect on a fixed core.

Regression shows net harm on the mean metric. The new adapter improves squared error on **2,990** of 6,339 evaluations and worsens it on **3,348**, with one effectively unchanged under the diagnostic tolerance. Counts alone understate the problem because a few correction magnitudes dominate error. [Correction contribution](../results/adaptation_improvement/aq2/adaptation.csv), [regression direction/overshoot summaries](../results/adaptation_improvement/aq2/regression_behavior.csv).

## The Car failure is extreme overcorrection

Car seed 40 gives a clear separation between retrieval quality and adapter damage:

| Model | Own retrieval RMSE | Final RMSE |
| --- | ---: | ---: |
| Original | 0.220318 | 0.209133 |
| Query + prediction | 0.223125 | 0.375484 |
| Query without prediction | 0.219376 | 0.614948 |

The new retrieval output is comparable to the others. On test query index 1128, its retrieved value is **0.003681** and target **0.127713**, requiring **+0.124032**. It instead applies **+21.201739**, yielding **21.205420**. The direction is correct; the magnitude is about **171 times** what is needed.

This one query contributes 444.254 of the run's 450.452 total positive squared-error increase, or **98.62%**. The denominator sums harm from adaptation and does not subtract benefits on other queries. Severe overshoot accounts for 445.075 of that total, versus 5.372 from wrong-direction moves. This is a descriptive diagnosis using saved test targets, not a test-selected clipping rule. The original and query-aware comparisons, top harmful cases and exact values are preserved in [harm examples](../results/adaptation_improvement/aq2/harm_examples.csv) and [behavior summaries](../results/adaptation_improvement/aq2/regression_behavior.csv).

Removing the prediction input therefore does not address this failure mode. It blocks direct subtraction, while the unrestricted regression residual can still become very large.

## Retrieval-dependence probes

The fixed, predeclared probes perturb saved-model inputs without retraining or selecting a new model:

| New-model inference condition | Classification error % | Regression RMSE |
| --- | ---: | ---: |
| Normal | 10.146 | 0.358695 |
| Rotate aggregate feature/prediction context | 58.633 | 1.470979 |
| Zero retrieval context | 51.878 | 0.979844 |
| Zero only added query channel | 10.810 | 0.370266 |
| Rotate only retrieved prediction | 60.221 | 1.335762 |

In the last row, query/difference inputs and residuals stay fixed; only the final additive retrieval term changes. Regression's mean final-output/retrieval RMS-change ratio is **1.000000004**, confirming the intended direct dependence. These artificial inputs may be out of distribution; their accuracy is diagnostic and is not an alternative deployable result. [All intervention records](../results/adaptation_improvement/aq2/interventions.csv), [intervention summary](../results/adaptation_improvement/aq2/intervention_summary.csv).

## Decision, limitations and reproducibility

Keep the original default and preserve this variant for controlled follow-up. Classification's reduced harmful flips merit a comparison in which both adapters share the same frozen retrieval core. Current joint training does not isolate that benefit. Regression needs investigation of extreme corrections; deleting the prediction input alone is insufficient. No additional training experiment, clipping threshold or new default was selected from these test results.

The [recorded driver](../tools/t1_query_no_prediction.py) writes 42 new units under `runs/t1/aq2/nopred_v1/`, including epoch histories, resume states, weights and predictions. Generated evidence is under `results/adaptation_improvement/aq2/`; this handwritten report is under `reports/`. Rebuild summaries with the [local uv launcher](../../run_query_no_prediction_experiment.py), `--study aq2 --mode summarize`. Runtime logs are `logs/local_aq2_*.log` in the mini-project.

Validation reconciled **252 new saved prediction/metric comparisons** (84 validation/test sets plus 168 interventions), rechecked **840 reference comparisons**, verified all 42 parameter-count reductions and source target matches, and checked prediction-only residual invariance for every test set. All 42 complete new units were reused with the training function replaced by an assertion, proving no fitting occurred on reuse. The summary figure and file links were inspected.

The adaptation/synchronization integration suite passed **96 tests**, including 35 query-adapter tests covering initialization, gradient isolation, schedules, frozen reuse, event routing and exact interruption recovery. Three additional new driver/summary tests passed. Shared-core manifest and whitespace checks passed. Shared NN-kNN core, published Table 1 and the original default remain unchanged. This is a local exploratory comparison on previously inspected splits, not independent statistical confirmation or a reproduction of the full historical benchmark.

Follow-up, November 26: [sequential, alternating and joint training](ADAPTER_TRAINING_SCHEMES_LOCAL_2026-11-26.md) now compares all three input designs under each scheme. It explicitly separates the earlier sequential Diabetes condition from the joint conditions and includes a fixed-retrieval comparison.
