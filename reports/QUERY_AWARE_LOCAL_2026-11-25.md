# Query-aware residual adaptation: local comparison

November 25, 2026. Study `aq1`, experiment `query_v1`. Complete: **14 datasets, three seeds, 42 query-aware models, 672 new recorded units and 294 comparison rows; no missing units.** All execution was local on CPU. The original full NN-kNN and pretrained retrieval baselines were reused from `am1`.

Follow-up: [removing the retrieved prediction from the query-aware adapter](QUERY_NO_PREDICTION_LOCAL_2026-11-25.md), study `aq2`, is now complete. The results below remain the original `aq1` comparison.

**Query access helps classification modestly but worsens regression overall.** Mean classification error falls from **9.945% to 9.359%**; mean standardized regression RMSE rises from **0.338910 to 0.345176**. Query-aware improves four task means in each family, but a large Car regression loss outweighs the small regression gains. The original residual adapter remains the default.

Against the newly fitted compact local baselines, query-aware NN-kNN has lower mean error than MLP, TabM and kNN in both families. CatBoost has lower means in both. These comparisons do **not** establish superiority over the historical tuned neural baselines: their selected configurations and weights were unavailable locally, and the new baseline search and training budgets are smaller. [Full numerical evidence](../results/adaptation_improvement/aq1/comparison.json).

## What was implemented

The original adapter receives the represented query-to-neighborhood difference and retrieved prediction. The new opt-in adapter also receives the represented query itself:

```text
z_bar = sum_i retrieval_weight_i * z_case_i
original input    = [z_query - z_bar, retrieved_prediction]
query-aware input = [z_query - z_bar, retrieved_prediction, z_query]
final prediction  = retrieved_prediction + predicted_residual
```

Classification retains the original tanh-bounded additive class-score residual; regression retains the unrestricted additive residual. Hidden layers, losses and output dimensions follow the original selected configuration. The added first-layer query connections start at zero, preserving the original initial function and RNG stream. This adds `query_dimension * first_hidden_width` parameters; parameter count is not matched. No magnitude/direction or gate variant was trained in this study.

**Residual output alone cannot enforce dependence on retrieval.** In regression, for example, an adapter can implement `residual = f(query) - retrieved_prediction`. The final output then becomes `f(query)`. We therefore retained residual output but measured dependence on retrieval with fixed-model interventions below; we did not claim an architectural guarantee against this shortcut.

Implementation: [adapter classes](../model/t1/query_aware.py), [recorded experiment driver](../tools/t1_query_aware.py), [pre-result protocol](../docs/QUERY_AWARE_EXPERIMENT.md). Set `adapter_design="query_aware"`. Frozen reuse, independent/alternating/joint training, prediction routing and T1 retrieval-event recording support it. This experiment supports fully represented numeric inputs and additive score classification, with unsupported nominal/logit modes rejected explicitly.

## Comparison protocol

The 14 tasks and seeds 40, 41, 42 match the previous local experiments. Data fingerprints and split targets match the source study. Query-aware training starts from each recorded **pretrained retrieval core and optimizer**, with a fresh matched adapter; it does not continue the old final adapter. Thirteen tasks retain their selected joint schedule, original losses and learning rates, five adapter warm-up epochs, and maximum 120 epochs. Diabetes retains frozen retrieval and its 300-epoch adapter budget. Validation selects checkpoints; test results never select a configuration.

New conventional models use exactly the same preprocessed splits. MLP and TabM each test widths 64/128; CatBoost tests depths 4/8; distance-weighted kNN tests 3/5/10/20 neighbors. Neural baselines train from scratch for at most 120 epochs with patience 30; CatBoost uses at most 300 iterations. Family selection uses validation error, then validation loss, then candidate ID. The [protocol](../docs/QUERY_AWARE_EXPERIMENT.md#new-matched-local-conventional-baselines) lists all settings. NN-kNN also benefits from its earlier retrieval pretraining and historical configuration selection, so this is a matched-data comparison rather than equal total tuning/compute.

## Overall results

Lower is better. Means average the three seeds within each dataset and weight the seven datasets in each family equally. Regression targets are standardized using training data. The standalone pretrained retrieval row is distinct from the selected full model's own retrieval output, reported later.

| Model | Classification error % | Regression RMSE |
| --- | ---: | ---: |
| Pretrained NN-kNN retrieval | 11.680 | 0.348690 |
| Original full NN-kNN | 9.945 | 0.338910 |
| **Query-aware full NN-kNN** | **9.359** | **0.345176** |
| MLP, compact local search | 13.922 | 0.390093 |
| TabM, compact local search | 14.259 | 0.389783 |
| CatBoost, compact local search | 4.517 | 0.318980 |
| kNN, compact local search | 14.989 | 0.496592 |

Relative to the original full model, query-aware has **4 wins / 2 ties / 1 loss** in classification and **4 / 0 / 3** in regression. Geometric mean task-error ratios are **0.97410** and **1.02738**, respectively: a 2.59% improvement and a 2.74% deterioration. No task is excluded from these two ratios.

Query-aware versus MLP has 3/1/3 classification and 6/0/1 regression wins/ties/losses; versus TabM it has 5/0/2 in each family. The large classification arithmetic-mean advantage over neural baselines is concentrated in the two Zebra tasks: the corresponding geometric mean advantage is only 2.46% against MLP and 2.10% against TabM. CatBoost beats query-aware on six of seven regression task means. All task means are included below; ratios involving CatBoost classification exclude Zebra's zero error, as explicitly recorded in the comparison CSV.

![Equal-task mean errors for all seven models](../results/adaptation_improvement/aq1/query_aware_comparison.png)

## Dataset results

Classification errors are percentages. These are three-seed means, not confidence intervals.

| Dataset | Original full | Query-aware | MLP | TabM | CatBoost | kNN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Iris | 3.333 | 5.556 | 5.556 | 6.667 | 7.778 | 5.556 |
| Wine | 7.407 | 6.481 | 2.778 | 1.852 | 3.704 | 2.778 |
| Breast Cancer | 2.924 | 2.924 | 2.047 | 2.047 | 3.216 | 2.339 |
| Balance | 5.333 | 4.000 | 2.400 | 4.267 | 9.600 | 10.133 |
| Digits | 2.130 | 1.852 | 3.611 | 2.407 | 2.778 | 3.056 |
| Zebra | 18.182 | 18.182 | 37.879 | 36.364 | 0.000 | 40.909 |
| Zebra Special | 30.303 | 26.515 | 43.182 | 46.212 | 4.545 | 40.152 |

Regression uses standardized RMSE.

| Dataset | Original full | Query-aware | MLP | TabM | CatBoost | kNN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Airfoil | 0.208094 | 0.210219 | 0.273725 | 0.325860 | 0.219794 | 0.394687 |
| Car | 0.222880 | 0.271599 | 0.289121 | 0.310408 | 0.166980 | 0.217708 |
| Yacht | 0.056331 | 0.056875 | 0.104144 | 0.108320 | 0.055724 | 0.512730 |
| Energy Efficiency | 0.047689 | 0.046816 | 0.061235 | 0.083877 | 0.038918 | 0.222960 |
| Student Performance | 0.441026 | 0.440538 | 0.454531 | 0.417251 | 0.416326 | 0.656343 |
| Body Fat | 0.627485 | 0.624921 | 0.789149 | 0.750525 | 0.569906 | 0.700348 |
| Diabetes (`diabets`) | 0.768861 | 0.765261 | 0.758746 | 0.732241 | 0.765215 | 0.771365 |

Wine and Breast Cancer still trail both neural baselines. Balance trails MLP; Student Performance trails TabM; Diabetes trails both. Simply supplying the query therefore does not close every neural-baseline gap. Car's new seed-40 RMSE is 0.375484; the complete seed-level table retains this result without excluding or retuning it. [All 294 model/seed rows](../results/adaptation_improvement/aq1/runs.csv), [task means](../results/adaptation_improvement/aq1/tasks.csv), [pairwise comparisons](../results/adaptation_improvement/aq1/comparisons.csv).

## How much is the correction itself helping?

These values hold each selected full model's own retrieval core fixed and compare its pre/post-adaptation predictions. Joint training can change the core, so differences between full-model results cannot all be credited to the correction.

| Family / model | Own retrieval error | After adaptation |
| --- | ---: | ---: |
| Classification, original | 10.385% | 9.945% |
| Classification, query-aware | 9.916% | 9.359% |
| Regression, original | 0.352137 | 0.338910 |
| Regression, query-aware | 0.350032 | 0.345176 |

Most of the classification gain is visible in the selected retrieval output: its error improves by 0.469 percentage points, while the full-model gain is 0.586 points. The correction adds a slightly larger average benefit, but this decomposition is descriptive and does not isolate a causal training mechanism.

Across 2,193 classification test evaluations, the original adapter fixes **25** retrieval mistakes and creates **24**; query-aware fixes **30** and creates **26**. The additional information increases useful corrections but does not eliminate harmful flips. Counts pool queries; the table above weights datasets equally, so the two summaries answer different questions.

Across 6,339 regression evaluations, query-aware lowers squared error on **3,028** and increases it on **3,309**, with two effectively unchanged under the diagnostic tolerance. Original counts are 3,030 improved / 3,309 harmed. Frequency alone misses correction size: despite similar counts, the mean regression benefit of adaptation shrinks from 0.013227 to 0.004856 RMSE.

**Car is an adaptation failure in this comparison.** Its query-aware model's own retrieval RMSE is 0.229463; adaptation worsens it to 0.271599. The original model improves its own retrieval from 0.231958 to 0.222880. Yacht remains a useful adaptation case: query-aware improves its own retrieval from 0.120390 to 0.056875. [Correction summaries](../results/adaptation_improvement/aq1/adaptation.csv).

## Does it bypass retrieval?

With model weights fixed, we keep query representations unchanged and either rotate the retrieved aggregate/prediction across test queries or replace that context with zeros (uniform retrieval class probabilities for classification). The difference input is recomputed. A third intervention zeros only the added query channel; the query-to-neighborhood difference still contains query information.

| Inference condition | Original class error % | Query-aware class error % | Original regression RMSE | Query-aware regression RMSE |
| --- | ---: | ---: | ---: | ---: |
| Normal | 9.945 | 9.359 | 0.338910 | 0.345176 |
| Rotate retrieval context | 55.548 | 55.627 | 1.322286 | 1.422760 |
| Zero retrieval context | 50.732 | 46.849 | 0.979765 | 0.959991 |
| Zero added query channel | — | 8.797 | — | 0.355848 |

Both models remain strongly sensitive to retrieval context in these probes. Rotation changes the predicted class on an average 56.31% of queries for the old model and 55.91% for query-aware. For regression, the mean ratio of final-output RMS change to induced retrieval-prediction RMS change is 0.991 for the old model and 1.068 for query-aware. This does not resemble broad cancellation of retrieval in these particular interventions.

The direct query channel has mixed value after training: zeroing it worsens regression but improves classification on these test splits. It can affect joint training even when its direct inference contribution is unhelpful. These artificial inputs may be out of distribution, and the probes neither prove case faithfulness nor establish that removing the channel would help a retrained model. They were fixed in advance and were not used to select an alternative model. [Intervention summary](../results/adaptation_improvement/aq1/intervention_summary.csv), [all 210 intervention records](../results/adaptation_improvement/aq1/interventions.csv).

## Recommendation and limits

Keep query-aware available as an experimental option, with the original residual adapter as default. Query access is useful enough to investigate for classification, but it does not resolve unreliable correction or provide a general regression improvement. A useful next controlled comparison would freeze the same retrieval core for old and query-aware adapters, separating direct adapter value from joint representation changes; that experiment has not been run here.

The neural-baseline question needs an equal-search, fresh-split comparison before claiming that NN-kNN has caught up to a well-tuned neural counterpart. These three seeds and 14 available tasks have already been inspected in earlier studies; they are exploratory evidence, not independent confirmation or a complete reproduction of the 29-task historical benchmark. No new default, gate, test-selected hyperparameter or remote run was introduced.

## Evidence and verification

- `runs/t1/aq1/query_v1/` contains 42 data units, 42 query-aware model units, 420 baseline candidate units and 168 selected-baseline evaluation units: **672 complete units**. Source full/retrieval models remain in `am1`. Generated files are confined to `results/adaptation_improvement/aq1/`; this handwritten report is under `reports/`.
- The summary reconciles **840 saved prediction/metric comparisons**: 210 selected-model test sets, 210 intervention outputs and 420 candidate validation sets. It additionally verifies query-aware pre/post and fix/harm metrics on validation/test, the original reconstructed pre/post metrics, unique comparison rows, and all 168 validation-based family selections. All 42 old-model test reconstructions matched their recorded source predictions.
- All 42 completed query-aware units and all 168 selected baseline units were reused without training; the baseline reuse check replaced the fitting function with an assertion to ensure no fitting occurred. Interrupted epoch training resumes exactly in the focused tests.
- The 78-test adaptation/synchronization check passed. After extending event logging, 46 focused tests passed, including all **19 query-aware tests**. The existing grid-search/prefill suite passed **11 tests** with the experiment-only determinism environment flag unset, as those default-setting tests require. Counts overlap and should not be added. Shared-core manifest and whitespace checks passed; the summary figure was visually inspected.
- A query-aware MLP warm-up cache issue was fixed before the affected run completed. The fix avoids caching an autograd graph during a shape probe. No completed source or experiment unit was retrained. Shared NN-kNN core and published Table 1 code remain unchanged.

Rebuild evidence with the [local uv launcher](../../run_query_aware_experiment.py) using `--study aq1 --mode summarize`. Its `--mode run` loads the completed units. Runtime logs are under the mini-project's `logs/local_aq1_*.log`. The [protocol](../docs/QUERY_AWARE_EXPERIMENT.md) and [tests](../tests/test_t1_query_aware.py) document the design and implementation contracts.
