# Sequential, alternating and joint NN-CDH training

**Follow-up completed:** [Dropout across all three adapters and schemes](ADAPTER_DROPOUT_LOCAL_2026-11-26.md) adds 756 models at 10% and 20% hidden dropout and reuses all 378 zero-dropout cells below. Gains depend strongly on adapter, scheme and dataset; no universal default is established.

November 26, 2026. Study `as1`, experiment `schemes_v1`. **Complete: 378 adapter/scheme/dataset/seed comparisons, comprising 252 new models and 126 reused models.** Fourteen local datasets, seeds 40–42, CPU execution only.

**Scope clarification (November 27):** this scheme comparison covers neural NN-CDH adapters only. The later pairwise ExtraTrees and Random Forest studies use sequential frozen retrieval; these alternating/joint findings do not apply to them. A forest-informed retrieval update would require a separate training loop. [Later forest training scheme](NNCDH_ALTERNATIVES_REPORT_2026-11-27.md#training-scheme-sequential-with-retrieval-frozen).

## Findings

**For query-aware adaptation with the prediction input, joint training remains the strongest fixed scheme by arithmetic mean, with alternating very close. Freezing retrieval is worse.** Classification error is 10.980% sequential, 9.502% alternating and 9.359% joint; standardized regression RMSE is 0.371033, 0.345628 and 0.345530. Alternating has a better geometric mean ratio than joint in both families, so this is not a consistent dataset-by-dataset victory for joint. [1: Complete comparison](../results/adaptation_improvement/as1/comparison.json).

**Selecting a scheme using validation loss is promising for the query-aware-with-prediction adapter.** It reaches 8.745% classification error, 0.613 percentage points below fixed joint. Regression is 0.345186: almost unchanged from the previous query-aware result, 0.345176. The original adapter still has the better regression mean. These repeatedly inspected splits provide exploratory evidence; no default is changed.

**A different training schedule does not solve the large-correction problem.** Sequential training holds retrieval exactly fixed, yet produces severe wrong-direction updates and overshoots on Car. Alternating improves the frequency of useful corrections, but its selected retrieval cores are weaker on average than joint's, leaving their final scores close. [2: Correction diagnostics](../results/adaptation_improvement/as1/behavior.csv), [largest harmful cases](../results/adaptation_improvement/as1/harm_examples.csv).

## What was compared

The three residual adapters retain their existing hidden layers, output constraints and losses:

```text
Original:                 delta = A([query - neighborhood_mean, retrieved_prediction])
Query + prediction:       delta = A([query - neighborhood_mean, retrieved_prediction, query])
Query without prediction: delta = A([query - neighborhood_mean, query])

final_output = retrieved_prediction + delta
```

For each dataset/seed, every scheme starts from the same recorded pretrained retrieval model and Adam state, with a fresh adapter. Sequential uses that exact retrieval model throughout; it does not reuse one of the different retrieval models selected by previous joint runs.

| Scheme | Training after shared retrieval pretraining |
| --- | --- |
| Sequential | Freeze retrieval; train the adapter only. The three adapters receive identical retrieval predictions. |
| Alternating | After the common adapter warm-up, one complete retrieval pass with adapter frozen, then one complete adapter pass with retrieval frozen. Both passes use the same shuffled order. |
| Joint | After the same warm-up, update retrieval and adapter together on each minibatch. |

The original effective objectives are preserved, including the extra final-loss contribution from `L_R` when the original joint objective is `summed`. Merely changing the legacy schedule flag would not preserve this. Maximum component update counts match; alternating uses additional forward passes, and early stopping produces different executed counts. See the [pre-result protocol](../docs/ADAPTER_TRAINING_SCHEMES_EXPERIMENT.md).

**Diabetes exception:** the previous reports used frozen sequential training on Diabetes, and joint training on the other 13 tasks. Those nine Diabetes models are reused as sequential references. New alternating/joint Diabetes runs use the declared combined adapter loss and retrieval objective `L_pre + L_post`; these retrieval settings were not separately tuned. Consequently, the all-joint regression means here differ slightly from the previous mixed-schedule means. Previous results have not been overwritten or relabeled.

## All adapters and schemes

Each cell first averages three seeds per dataset, then gives each of seven datasets equal weight. Lower is better. “Validation selected” chooses the scheme separately for each adapter/dataset/seed using validation task loss, then validation error, then a fixed tie order. Test performance does not select the scheme.

Classification error (%):

| Adapter | Sequential | Alternating | Joint | Validation selected |
| --- | ---: | ---: | ---: | ---: |
| Original | 10.791 | 10.244 | 9.945 | 9.998 |
| Query + prediction | 10.980 | 9.502 | 9.359 | **8.745** |
| Query without prediction | 10.848 | **9.356** | 10.146 | 9.303 |

Standardized regression RMSE:

| Adapter | Sequential | Alternating | Joint | Validation selected |
| --- | ---: | ---: | ---: | ---: |
| Original | 0.350429 | 0.342507 | **0.340452** | 0.349431 |
| Query + prediction | 0.371033 | 0.345628 | 0.345530 | 0.345186 |
| Query without prediction | 0.415262 | 0.346292 | 0.359316 | 0.409628 |

Query-without-prediction alternating has the numerically lowest fixed-scheme classification mean, but its advantage over query-with-prediction joint is only **0.0025 percentage points**. That is not evidence of a reliable winner. Within sequential training, adding the query does not outperform the original adapter on either family mean. The ranking therefore depends on training scheme, rather than query access being an unconditional improvement.

![All nine fixed adapter/scheme combinations](../results/adaptation_improvement/as1/scheme_means.png)

[3: Family means](../results/adaptation_improvement/as1/families.csv), [validation-selected means](../results/adaptation_improvement/as1/selected_families.csv), [all dataset means](../results/adaptation_improvement/as1/tasks.csv).

## Query + prediction: paired results

These comparisons use fixed joint training of the same adapter as the reference. Wins/ties/losses count the seven dataset means; GMR is the geometric mean of task-error ratios, with values below one favoring the candidate. All seven task ratios are defined; none are excluded for zero error.

| Candidate | Classification W / T / L | Classification GMR | Regression W / T / L | Regression GMR |
| --- | ---: | ---: | ---: | ---: |
| Sequential | 1 / 2 / 4 | 1.09722 | 1 / 1 / 5 | 1.08924 |
| Alternating | 3 / 2 / 2 | 0.93196 | 3 / 1 / 3 | 0.98422 |
| Validation selected | 2 / 5 / 0 | 0.91544 | 4 / 1 / 2 | 0.98426 |

Alternating wins more classification tasks than it loses, despite its slightly worse arithmetic mean. Its larger Zebra error accounts for that discrepancy. Validation selection is also not uniformly better at the seed level: against fixed joint, classification has 6 wins, 12 ties and 3 losses; regression has 8 wins, 7 ties and 6 losses. [4: Task and seed comparisons](../results/adaptation_improvement/as1/comparisons.csv).

Classification dataset means, error %:

| Dataset | Sequential | Alternating | Joint | Validation selected |
| --- | ---: | ---: | ---: | ---: |
| Iris | 6.667 | 3.333 | 5.556 | 5.556 |
| Wine | 3.704 | 4.630 | 6.481 | 3.704 |
| Breast Cancer | 2.924 | 2.924 | 2.924 | 2.924 |
| Balance | 6.133 | 4.533 | 4.000 | 4.000 |
| Digits | 2.130 | 1.852 | 1.852 | 1.852 |
| Zebra | 28.788 | 24.242 | 18.182 | 18.182 |
| Zebra Special | 26.515 | 25.000 | 26.515 | 25.000 |

Regression dataset means, standardized RMSE:

| Dataset | Sequential | Alternating | Joint | Validation selected |
| --- | ---: | ---: | ---: | ---: |
| Airfoil | 0.212676 | 0.211789 | 0.210219 | 0.208938 |
| Car | 0.437051 | 0.273611 | 0.271599 | 0.279521 |
| Yacht | 0.058095 | 0.050584 | 0.056875 | 0.050536 |
| Energy Efficiency | 0.050436 | 0.046202 | 0.046816 | 0.046202 |
| Student Performance | 0.448793 | 0.439571 | 0.440538 | 0.440926 |
| Body Fat | 0.624921 | 0.624921 | 0.624921 | 0.624921 |
| Diabetes | 0.765261 | 0.772718 | 0.767739 | 0.765261 |

For query + prediction, validation selects sequential / alternating / joint in **7 / 5 / 9** classification runs and **12 / 6 / 3** regression runs. Its classification improvement over fixed joint comes from Wine and Zebra Special. As a sensitivity check, excluding Car leaves regression means of 0.360030 sequential, 0.357631 alternating and 0.357851 joint. Thus Car explains most of sequential's regression disadvantage; the primary comparison retains it. [5: Selected runs](../results/adaptation_improvement/as1/selected.csv), [selection counts](../results/adaptation_improvement/as1/selection_counts.csv).

## Retrieval quality versus the correction

For query + prediction, compare the selected model's own retrieval prediction with its final adapted prediction:

| Scheme | Classification before → after (%) | Fixed / harmed class decisions | Regression before → after RMSE | Improved / harmed regression cases |
| --- | ---: | ---: | ---: | ---: |
| Sequential | 11.680 → 10.980 | 30 / 27 | 0.348690 → 0.371033 | 3,008 / 3,328 |
| Alternating | 11.226 → 9.502 | 31 / 21 | 0.356967 → 0.345628 | 3,254 / 3,083 |
| Joint | 9.916 → 9.359 | 30 / 26 | 0.350360 → 0.345530 | 3,026 / 3,311 |

Counts pool **2,193 classification evaluations** or **6,339 regression evaluations** across datasets/seeds; they are not distinct independent examples. Error means give datasets equal weight, so net pooled counts do not directly determine the change in those means. A few regression changes within the `1e-8` squared-error tolerance count as neither improved nor harmed.

Alternating's adapter makes more helpful corrections and fewer harmful ones in this comparison. But joint selects better retrieval predictions on average, so their final performance is close. These are corrections to different selected cores, not an isolated comparison of adapters acting on the same predictions. Sequential is the isolated fixed-retrieval comparison, and its regression corrections worsen the mean for **all three** adapters.

Regression correction directions for query + prediction, as percentages of all 6,339 evaluations:

| Scheme | Wrong direction | Overshoot that increases error | Any squared-error improvement | Any squared-error worsening |
| --- | ---: | ---: | ---: | ---: |
| Sequential | 41.80% | 8.36% | 47.45% | 52.50% |
| Alternating | 38.48% | 8.28% | 51.33% | 48.64% |
| Joint | 41.32% | 8.35% | 47.74% | 52.23% |

Wrong direction means the correction has the opposite sign to `target − retrieval`. Harmful overshoot means the direction is correct but the correction exceeds twice the needed residual. Moving away from an already exact retrieval prediction is counted separately. The magnitudes matter: error rates alone miss the rare very large failures. [6: Pooled behavior counts](../results/adaptation_improvement/as1/behavior_totals.csv).

### Severe errors survive frozen retrieval

- **Query + prediction, Car seed 40, sequential:** retrieval RMSE 0.206827 becomes **0.837377**. Test query 1128 needs a correction of **−0.579325**, but receives **+29.186705**. It is a wrong-direction update and contributes **99.30%** of that run's sum of positive squared-error increases from adaptation.
- **Query without prediction, Car seed 42, sequential:** retrieval RMSE 0.232915 becomes **1.354126**. Query 545 needs **+0.154573**, but receives **+48.875562**, about **316 times** the needed correction. It contributes **99.49%** of that run's adaptation harm.

The second model is selected by validation loss: its validation MSE is 0.043393, better than 0.059979 for alternating/joint. Therefore validation selection itself does not protect against an unseen extreme correction. This explains the poor 0.409628 regression mean for validation-selected query-without-prediction. No clipping, exclusion or test-based selection was applied.

## Comparison with the current baselines

| Model / policy | Classification error % | Regression RMSE |
| --- | ---: | ---: |
| Pretrained retrieval only | 11.680 | 0.348690 |
| Previous original full model: joint except sequential Diabetes | 9.945 | **0.338910** |
| Previous query + prediction: same mixed schedule | 9.359 | 0.345176 |
| Query + prediction, validation-selected scheme | **8.745** | 0.345186 |
| Compact MLP search | 13.922 | 0.390093 |
| Compact TabM search | 14.259 | 0.389783 |
| Compact CatBoost search | 4.517 | 0.318980 |
| Compact kNN search | 14.989 | 0.496592 |

The query-aware model still beats these local MLP and TabM means, but the baselines used smaller searches and budgets. This does not establish equal-budget neural parity. CatBoost remains better on both family means. Relative to the previous original full model, validation-selected query + prediction wins 4 / ties 2 / loses 1 classification tasks, with GMR 0.89173. In regression it wins 5 / loses 2, but has a worse arithmetic mean and GMR 1.01167: small wins do not offset larger losses. [7: Previous matched baseline study](QUERY_AWARE_LOCAL_2026-11-25.md), with its recorded numerical references preserved in this study's comparison JSON.

**Recommended interpretation:** retain joint as the fixed reference for query + prediction, and carry validation-selected training schemes into a fresh-split confirmation experiment. Alternating is a useful candidate, especially for adaptation behavior, but this study does not show a general final-score advantage. The original model remains the stronger regression reference. Keep the existing default; scheduling alone has not resolved extreme corrections.

## Budgets, implementation and verification

For the 13 original joint tasks, all schemes allow 120 adapter passes, including five adapter-only warm-up passes; alternating/joint allow 115 retrieval passes. Patience is 30. Diabetes keeps the frozen trainer's 300-pass maximum and patience 30, with no warm-up for its new alternating/joint runs. Initial rates, loss coefficients, score scales, free-radius settings and cached splits are preserved as specified in the protocol. There is no additional hyperparameter search.

Executed optimizer updates for query + prediction across the 21 runs in each family:

| Family | Scheme | Retrieval updates | Adapter updates |
| --- | --- | ---: | ---: |
| Classification | Sequential | 0 | 6,694 |
| Classification | Alternating | 5,417 | 5,987 |
| Classification | Joint | 5,196 | 5,766 |
| Regression | Sequential | 0 | 19,516 |
| Regression | Alternating | 17,682 | 19,167 |
| Regression | Joint | 15,318 | 16,803 |

These exclude the reused common retrieval pretraining. Maximum component budgets match, but actual updates and compute do not. Checkpoints can come from warm-up: 2/21 classification and 6/21 regression query-aware joint checkpoints have no selected retrieval updates. The [budget table](../results/adaptation_improvement/as1/budgets.csv) distinguishes executed from selected update counts for every combination.

The implementation is an opt-in `comparison_scheme` override in [the T1 trainer](../model/t1/sync.py), used by [the recorded experiment driver](../tools/t1_adapter_training_schemes.py). Existing schedule defaults, shared NN-kNN core and Table 1 are unchanged. The [local uv launcher](../../run_training_schemes_experiment.py) runs disjoint units with bounded single-thread CPU workers. From the mini-project `code/` directory, use `../.tools/uv.exe run --project ../.. --no-sync python run_training_schemes_experiment.py --study as1 --workers 6`; add `--mode summarize` to regenerate summaries only, using the established local uv environment.

Verification completed:

- All **378** comparison cells are present and unique, with paired split fingerprints/targets. All **252** new initial retrieval errors agree with the pretrained retrieval baseline.
- Reconciled **714 saved prediction sets** with reported metrics. The existing reference/baseline audit also passed **840** metric, intervention and validation-candidate checks; these checks overlap in source evidence and are not a count of independent observations.
- All **126 sequential models** have retrieval weights byte-identical to the shared starting core. All **84 cross-adapter sequential test-prediction comparisons** are exact matches.
- Re-entered all **378** comparison cells with the training function replaced by an assertion: every completed unit/reference was reused without training.
- The targeted synchronization/query suite passed **75 tests**. A subsequent focused run passed **24 tests**, including the new validation-selection and update-accounting checks; counts overlap. Synthetic tests cover both tasks, all adapters, effective objective matching, frozen state and exact interruption recovery.
- Shared-core manifest checks passed in both worktrees; whitespace checks passed. The generated figure and report links were inspected.

Recorded units, curves, weights and predictions remain under `NNKNN_RUNS/t1/as1/schemes_v1/`, with references to `am1`, `aq1` and `aq2`. Generated tables/figures are under [results/adaptation_improvement/as1](../results/adaptation_improvement/as1/); [individual rows](../results/adaptation_improvement/as1/rows.csv) retain source unit IDs, validation metrics, hashes and selected epochs. No remote run was attempted. This comparison covers the 14 available local tasks, not the full historical benchmark; three repeatedly inspected seeds and the untuned Diabetes extension limit generalization.
