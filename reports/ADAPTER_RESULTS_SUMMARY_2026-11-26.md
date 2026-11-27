# Adapter experiments: consolidated results

Created November 26, updated November 27, 2026. Summary of completed work in the `adaptation improvement` mini-project through study `ar3`. This document consolidates the individual studies; it does not introduce additional training or hyperparameter selection.

## Executive summary

**Latest result: retrieved-only pairwise Random Forest improves classification and nearly ties ExtraTrees in regression.** It reaches **3.325% classification error / 0.314232 regression RMSE**, versus matching ExtraTrees' **4.350% / 0.314960**. The classification gain comes from Iris and Zebra Special; the other five dataset means tie. Regression improves only **0.23%**, with four dataset wins and three losses, and RF causes more total harmful correction damage. Keep both tree learners for confirmation. Both improve all seven regression dataset means against retrieval and original sequential NN-CDH with 20% dropout. [11: Updated alternatives report](NNCDH_ALTERNATIVES_REPORT_2026-11-27.md).

**There is no consistently best adapter across classification, regression and training schemes.** Separating magnitude from direction did not improve overall performance and was shelved. Simple disagreement gating helped modestly; learned gates and a trainable threshold did not establish a consistent advantage over that simple rule. [1: Magnitude/direction](ADAPTATION_IMPROVEMENT_LOCAL_2026-11-25.md), [2: Apply-or-skip gates](ADAPTATION_GATE_LOCAL_2026-11-25.md), [3: Trainable threshold](ADAPTATION_THRESHOLD_LOCAL_2026-11-25.md).

Adding the query representation improved the initial classification comparison but worsened regression. Removing the retrieved prediction input removed a direct cancellation path, without improving both families. Before dropout, query + prediction with a validation-selected training scheme had the lowest adapter classification mean, **8.745% error**. [4: Query + prediction](QUERY_AWARE_LOCAL_2026-11-25.md), [5: Query without prediction](QUERY_NO_PREDICTION_LOCAL_2026-11-25.md), [6: Training schemes](ADAPTER_TRAINING_SCHEMES_LOCAL_2026-11-26.md).

Before the pairwise redesign, dropout produced the lowest observed adapter classification means, but the gain is concentrated on Zebra. Its strongest fixed regression setting reaches **0.337381 RMSE**, essentially alongside the earlier probability gate's **0.337206**. Among NN-CDH variants, the clearest improvement with retrieval held exactly fixed is **original + sequential + 20% dropout**, from **0.350429 to 0.338523 RMSE**, improving six of seven regression dataset means. No universal default has been adopted. [7: Dropout](ADAPTER_DROPOUT_LOCAL_2026-11-26.md).

## Scope and how to read the numbers

All studies use the same **14 available local datasets** and seeds **40, 41, 42**:

- Classification: Iris, Wine, Breast Cancer, Balance, Digits, Zebra, Zebra Special.
- Regression: Airfoil, Car, Yacht, Energy Efficiency, Student Performance, Body Fat, Diabetes (`diabets` in code).

Classification numbers are **percentage error**, not accuracy. Regression numbers are **RMSE in training-standardized target units**. Lower is better. Family means average the three seeds within each dataset, then give each of seven datasets equal weight. They are not pooled per-example error rates. There is no single combined classification/regression score.

The initial comparisons used **joint training on 13 tasks and frozen sequential training on Diabetes**. Later studies explicitly compare sequential, alternating and all-joint training. This is why, for example, original regression is 0.338910 under the earlier mixed schedule but 0.340452 under all-joint training.

The **standalone pretrained retrieval** baseline and a full model's **own selected retrieval core** are different comparisons. Joint or alternating training can change retrieval. Full-model gains therefore cannot automatically be attributed to the final correction alone.

**Training-scheme scope:** the sequential/alternating/joint comparison applies to the neural NN-CDH variants in `as1` and `ad1`. All redesign families in `ar1`/`ar2`/`ar3`, including ExtraTrees and Random Forest, use the **sequential frozen-retrieval protocol**. Forest fitting changes neither the representation nor the core's neighbors or base activations. “Mixed pairs” means retrieved plus random training pairs, not joint/alternating training. The audit confirms 420 selected runs with the recorded sequential scheme, matching source hashes and identical retrieval predictions. [Scheme evidence](../results/adaptation_improvement/ar3/training_schemes.csv).

A forest could influence retrieval in a custom alternating loop if the retrieval update used the final adapted loss through differentiable aggregation weights, with forest outputs held fixed, followed by refreshed pairs and forest refitting. Hard tree splits do not supply the neural adapter's ordinary gradient path, and the current sklearn/NumPy pipeline has no such feedback. Forest joint/alternating training remains unimplemented and untested. [Explanation and proposed path](NNCDH_ALTERNATIVES_REPORT_2026-11-27.md#could-a-forest-influence-retrieval-in-joint-or-alternating-training).

These are exploratory results on repeatedly examined local splits, covering 14 of the historical 29 tasks. Validation selected checkpoints and declared tuning policies. A fixed setting identified as having the lowest test mean is a retrospective comparison, not independent confirmation of that hyperparameter. Completed source models were reused throughout; comparison rows across studies must not be added up as independent experiments.

## Experiment and evidence index

Every study below is complete. The report explains interpretation and limitations; the result links expose the numerical evidence.

| Ref. / study | Work completed | Report | Individual results |
| --- | --- | --- | --- |
| 1 / `am1` | Original versus magnitude/direction, 42 matched dataset/seed pairs | [Magnitude/direction comparison](ADAPTATION_IMPROVEMENT_LOCAL_2026-11-25.md) | [Task means](../results/adaptation_improvement/am1/tasks.csv), [seed pairs](../results/adaptation_improvement/am1/pairs.csv), [full JSON](../results/adaptation_improvement/am1/comparison.json) |
| 2 / `ag1` | Six gate/control alternatives plus original and retrieval; 336 method/seed rows | [Apply-or-skip exploration](ADAPTATION_GATE_LOCAL_2026-11-25.md) | [Family means](../results/adaptation_improvement/ag1/families.csv), [individual runs](../results/adaptation_improvement/ag1/runs.csv), [full JSON](../results/adaptation_improvement/ag1/comparison.json) |
| 3 / `ag2` | Two trainable-threshold variants, 84 new models; 420 rows including references | [Learned threshold](ADAPTATION_THRESHOLD_LOCAL_2026-11-25.md) | [Family means](../results/adaptation_improvement/ag2/families.csv), [runs and cutoffs](../results/adaptation_improvement/ag2/runs.csv), [full JSON](../results/adaptation_improvement/ag2/comparison.json) |
| 4 / `aq1` | 42 query-aware models and compact conventional-baseline searches; 294 comparison rows | [Query-aware with prediction](QUERY_AWARE_LOCAL_2026-11-25.md) | [Family means](../results/adaptation_improvement/aq1/families.csv), [individual runs](../results/adaptation_improvement/aq1/runs.csv), [full JSON](../results/adaptation_improvement/aq1/comparison.json) |
| 5 / `aq2` | 42 query-aware-without-prediction models; 336 rows including references | [Query-aware without prediction](QUERY_NO_PREDICTION_LOCAL_2026-11-25.md) | [Family means](../results/adaptation_improvement/aq2/families.csv), [individual runs](../results/adaptation_improvement/aq2/runs.csv), [full JSON](../results/adaptation_improvement/aq2/comparison.json) |
| 6 / `as1` | Three adapters x three schemes: 252 new models, 126 reused references, 378 cells | [Training schemes](ADAPTER_TRAINING_SCHEMES_LOCAL_2026-11-26.md) | [Fixed-scheme means](../results/adaptation_improvement/as1/families.csv), [validation-selected means](../results/adaptation_improvement/as1/selected_families.csv), [all rows](../results/adaptation_improvement/as1/rows.csv) |
| 7 / `ad1` | Three adapters x three schemes x rates 0/10/20%: 756 new models, 378 reused references, 1,134 cells | [Dropout](ADAPTER_DROPOUT_LOCAL_2026-11-26.md) | [Fixed-setting means](../results/adaptation_improvement/ad1/families.csv), [validation-rate means](../results/adaptation_improvement/ad1/selected_rate_families.csv), [all rows](../results/adaptation_improvement/ad1/rows.csv) |
| 8 / `am1` diagnostics | Saved-prediction analysis of all 84 old/new adapted models; no retraining | [Adaptation behavior](ADAPTATION_BEHAVIOR_2026-11-25.md) | [Task diagnostics](../results/adaptation_improvement/am1/behavior/tasks.csv), [per-query diagnoses](../results/adaptation_improvement/am1/behavior/cases.csv), [full JSON](../results/adaptation_improvement/am1/behavior/diagnostics.json) |
| 9 / `ar1` | Three non-NN-CDH replacements; 294 validation candidates, 126 selected test evaluations and 42 frozen contexts | [Pairwise redesign](ADAPTER_PAIRWISE_REDESIGN_LOCAL_2026-11-26.md) | [Family means](../results/adaptation_improvement/ar1/families.csv), [task means](../results/adaptation_improvement/ar1/tasks.csv), [correction behavior](../results/adaptation_improvement/ar1/behavior_totals.csv), [full JSON](../results/adaptation_improvement/ar1/comparison.json) |
| 10 / `ar2` | Three creative designs plus two controls; 588 validation candidates and 210 selected tests; merged with `ar1` | [Alternatives to NN-CDH, merged](NNCDH_ALTERNATIVES_REPORT_2026-11-27.md) | [Merged means](../results/adaptation_improvement/ar2/families.csv), [all rows](../results/adaptation_improvement/ar2/rows.csv), [behavior](../results/adaptation_improvement/ar2/behavior_totals.csv), [full JSON](../results/adaptation_improvement/ar2/comparison.json) |
| 11 / `ar3` | Ordinary Random Forest on both pair recipes; 168 validation candidates and 84 selected tests; merged with `ar1`/`ar2` | [Alternatives to NN-CDH, updated](NNCDH_ALTERNATIVES_REPORT_2026-11-27.md) | [Merged means](../results/adaptation_improvement/ar3/families.csv), [all rows](../results/adaptation_improvement/ar3/rows.csv), [behavior](../results/adaptation_improvement/ar3/behavior_totals.csv), [full JSON](../results/adaptation_improvement/ar3/comparison.json) |

## 1. Residual inputs and magnitude/direction separation

The current residual adapters use a two-hidden-layer MLP, widths 64 and 32. Classification uses additive class-score corrections, with a tanh-bounded residual in these experiments. Regression uses an unbounded scalar residual. The three input variants are:

```text
Original:                 delta = A([query - neighborhood_mean, retrieved_prediction])
Query + prediction:       delta = A([query - neighborhood_mean, retrieved_prediction, query])
Query without prediction: delta = A([query - neighborhood_mean, query])

final_output = retrieved_prediction + delta
```

The magnitude/direction alternative instead computes `delta = magnitude * unit_direction`. Its shared model has a direction proposal and a nonnegative magnitude head. Magnitude receives the hidden representation plus four neighborhood statistics: nearest distance, activation-weighted mean distance, activation concentration, and target disagreement. Raw-residual supervision trains the direction proposal, including scalar regression where hard sign otherwise blocks its gradient.

Initial results under the earlier mixed schedule:

| Model | Classification error % | Regression RMSE | Result |
| --- | ---: | ---: | --- |
| Standalone pretrained retrieval | 11.680 | 0.348690 | Common starting reference |
| Original full adapter | 9.945 | 0.338910 | Reference retained |
| Magnitude/direction | 11.141 | 0.340946 | Worse overall; shelved |
| Query + prediction | 9.359 | 0.345176 | Classification improves; regression worsens |
| Query without prediction | 10.146 | 0.358695 | Both arithmetic means worse than query + prediction |

Magnitude/direction improves five classification and two regression dataset means, but its geometric mean error ratios versus the original are **1.03683** and **1.03032**: worse in both families. Separating scale and direction made the correction easier to describe; it did not by itself improve prediction. [1: Report and implementation](ADAPTATION_IMPROVEMENT_LOCAL_2026-11-25.md), [design](../docs/ADAPTATION_MAGNITUDE_DIRECTION.md).

Query-aware with prediction adds zero-initialized query connections, preserving the initial function while increasing parameter count. Its classification gain is partly accompanied by better selected retrieval predictions. Fixed-model retrieval interventions show substantial dependence on retrieval context; they do not prove faithful case use. Residual output alone can still permit cancellation, such as `delta = f(query) - retrieved_prediction`. [4: Results and interventions](QUERY_AWARE_LOCAL_2026-11-25.md), [intervention results](../results/adaptation_improvement/aq1/intervention_summary.csv).

Removing the prediction input eliminates that direct subtraction path: at fixed query and neighborhood features, changing only the retrieved prediction leaves the residual unchanged. It does not constrain correction magnitude or prevent indirect estimation from correlated features. Classification has a better geometric ratio than the original, **0.89108**, despite its worse arithmetic mean; regression is worse by both. A Car correction of **+21.20 when +0.124 was needed** illustrates the remaining overshoot problem. [5: Report](QUERY_NO_PREDICTION_LOCAL_2026-11-25.md), [paired comparisons](../results/adaptation_improvement/aq2/comparisons.csv), [harmful cases](../results/adaptation_improvement/aq2/harm_examples.csv).

## 2. Gating and its controls

These studies start from the earlier original adapter's selected core. **Retrieval stays frozen.** The learned-head and extra-training conditions continue the adapter; they do not retrain retrieval or test gating on the later query-aware/dropout winners.

| Method | What was tried | Classification error % | Regression RMSE |
| --- | --- | ---: | ---: |
| Own retrieval, no adaptation | Skip the original selected adapter entirely | 10.385 | 0.352137 |
| Original full adapter | Always apply its correction | 9.945 | 0.338910 |
| Global on/off | Validation chooses always apply or always skip | 9.896 | 0.338557 |
| Global shrink | Validation chooses one multiplier from 0, .25, .5, .75, 1 | 9.584 | 0.338398 |
| Disagreement threshold | Apply full correction when neighborhood target disagreement exceeds a validation-selected cutoff | 9.164 | 0.337424 |
| Extra training, no gate | Continue the adapter with retrieval frozen | 9.068 | 0.348525 |
| Probability gate | Train a shared-head apply/skip decision with the adapter | 10.348 | **0.337206** |
| Utility gate | Weight gate training by the size of the branch-loss difference | 9.911 | 0.339802 |
| Learned threshold, frozen adapter | Train one cutoff parameter; use its final training iterate | 9.945 | 0.338856 |
| Learned threshold + adapter | Train cutoff and adapter in the same stage; validation selects their checkpoint | 9.359 | 0.347770 |

The learned heads use adapter context, the four neighborhood statistics, the retrieved prediction and the proposed correction. The residual head remains supervised on skipped examples. The scalar-threshold variants use a fixed-temperature sigmoid during optimization but make a hard full-or-skip decision at inference. Neither output is established as calibrated confidence. [2: Gate design and results](ADAPTATION_GATE_LOCAL_2026-11-25.md), [3: Threshold design and results](ADAPTATION_THRESHOLD_LOCAL_2026-11-25.md).

**The simple disagreement threshold is the most balanced gate result.** It improves five of seven task means in each family. Mean classification error falls by 0.781 percentage points; regression RMSE falls by about 0.44%. Classification fixes/harmful flips change from **25/24 to 13/4**: it avoids damage while also discarding useful corrections.

**Learned gating has mixed value.** With their own gates forced open, the probability and utility models have classification errors of 9.522% and 9.110%; actual gating worsens them to 10.348% and 9.911%. Both gates improve their own regression proposals, but only the probability model beats the original mean. Its regression geometric ratio is slightly worse than the original despite having the lowest arithmetic mean.

The trainable threshold does not improve consistently on the validation-selected rule. Five frozen-adapter classification runs have no difference in training branch costs, hence no routing signal. In regression, the frozen learned threshold applies all test corrections in **19/21** runs. These observations motivate better routing supervision as a possible next experiment; they do not establish why the learned rule generalizes poorly.

An apply/skip gate can suppress a correction; it cannot repair its direction or magnitude. The evidence supports retaining simple disagreement gating as a baseline and giving learned gating lower priority. Gating combined with the strongest dropout adapters remains untested.

## 3. Sequential, alternating and joint training

All schemes start from the same pretrained retrieval checkpoint per dataset/seed, with fresh adapters:

- **Sequential:** freeze retrieval and train the adapter.
- **Alternating:** after warm-up, perform a retrieval pass with adapter parameters frozen, then an adapter pass with retrieval frozen.
- **Joint:** after warm-up, update retrieval and adapter on the same minibatches.

The experiment preserves effective component objectives and maximum update budgets. Actual early-stopping counts differ. Diabetes's new alternating/joint extension was not independently tuned.

Classification error (%), without dropout:

| Adapter | Sequential | Alternating | Joint | Validation-selected scheme |
| --- | ---: | ---: | ---: | ---: |
| Original | 10.791 | 10.244 | 9.945 | 9.998 |
| Query + prediction | 10.980 | 9.502 | 9.359 | **8.745** |
| Query without prediction | 10.848 | **9.356** | 10.146 | 9.303 |

Regression RMSE, without dropout:

| Adapter | Sequential | Alternating | Joint | Validation-selected scheme |
| --- | ---: | ---: | ---: | ---: |
| Original | 0.350429 | 0.342507 | **0.340452** | 0.349431 |
| Query + prediction | 0.371033 | 0.345628 | 0.345530 | 0.345186 |
| Query without prediction | 0.415262 | 0.346292 | 0.359316 | 0.409628 |

Query-without-prediction alternating and query-with-prediction joint are essentially tied in classification. Alternating often makes more useful corrections but can start from weaker selected retrieval outputs. Sequential training does not prevent extreme regression errors: Car still receives corrections of +29.19 when -0.579 is needed, or +48.88 when +0.155 is needed.

All sequential variants use exactly the same retrieval predictions, verified from saved states. Scheme selection uses validation task loss, validation error and a fixed tie order. [6: Full report](ADAPTER_TRAINING_SCHEMES_LOCAL_2026-11-26.md), [per-dataset comparisons](../results/adaptation_improvement/as1/tasks.csv), [selected models](../results/adaptation_improvement/as1/selected.csv), [correction diagnostics](../results/adaptation_improvement/as1/behavior.csv).

## 4. Dropout across every current input variant and scheme

We tested **0%, 10% and 20% dropout** across all three residual input variants and all three schemes. Positive dropout is placed after **both hidden ReLUs**, enabled during adapter training and disabled during evaluation. It adds no parameters and preserves initial weights. The full 27-setting matrix is in the [dropout report](ADAPTER_DROPOUT_LOCAL_2026-11-26.md) and [family results](../results/adaptation_improvement/ad1/families.csv).

| Comparison | Matched 0% dropout | Positive dropout | Validation-selected rate within that adapter/scheme |
| --- | ---: | ---: | ---: |
| Classification: query without prediction, alternating, 20% | 9.356% | **6.701%** | **6.453%** |
| Regression: query + prediction, alternating, 10% | 0.345628 | **0.337381** | 0.344081 |
| Regression: original, sequential, 20% | 0.350429 | **0.338523** | 0.341434 |

The classification result is dominated by **Zebra**, which improves from 21.212% error to zero. Excluding Zebra as a sensitivity check, the other six datasets worsen from **7.380% to 7.818%** at fixed 20% dropout. Zebra has 22 test examples per seed; its dropout models already have zero error before adaptation. The gain involves a different retrieval-training trajectory and checkpoint selection. Zebra remains in every primary mean; geometric ratios explicitly exclude its zero-error pair.

The query + prediction regression improvement is concentrated on Car and accompanies better selected retrieval. In contrast, original sequential with 20% improves **6/7 dataset means while retrieval is byte-identical across rates**, making it the clearer adapter-only result. Ten-percent dropout also improves 6/7 for original sequential, reaching 0.339674.

Dropout does not reliably improve every configuration. Query + prediction joint classification worsens at both positive rates. Query without prediction sequential regression is 0.457686 at 20%, versus 0.415262 without dropout, including severe Car failures.

Selecting both rate and scheme using validation also has mixed results:

| Adapter | Classification: old scheme selection -> rate + scheme | Regression: old scheme selection -> rate + scheme |
| --- | ---: | ---: |
| Original | 9.998% -> 9.817% | 0.349431 -> 0.342507 |
| Query + prediction | 8.745% -> 9.378% | 0.345186 -> 0.342510 |
| Query without prediction | 9.303% -> 7.053% | 0.409628 -> 0.458242 |

The lowest fixed test column and the validation-selected policy are different results. Enlarging the validation search did not reliably improve test performance. [7: Full interpretation](ADAPTER_DROPOUT_LOCAL_2026-11-26.md), [rate selections](../results/adaptation_improvement/ad1/selected_rates.csv), [joint selections](../results/adaptation_improvement/ad1/selected_both.csv), [joint-selection means](../results/adaptation_improvement/ad1/selected_both_families.csv).

## 5. What the correction diagnostics taught us

For scalar regression, define required residual `r = target - retrieval` and applied correction `c = final - retrieval`. Opposite signs mean a wrong direction. A correction in the right direction is a **harmful overshoot only when it exceeds twice the needed magnitude**; merely crossing the target can still reduce error.

| Evidence | Main finding |
| --- | --- |
| Original versus magnitude/direction | Regression wrong-direction frequency rises from **41.27% to 43.34%**; harmful overshoot falls from **8.17% to 6.39%**. The new model's wrong-direction corrections account for **90.12% of its positive squared-error increases**. |
| Classification, magnitude/direction | Fixes/harmful flips change from **25/24 to 8/11**. Smaller corrections reduce damage but lose useful adaptation. |
| Saved-output scale probes | Some errors need larger corrections, others need smaller ones; uniformly doubling corrections worsens both families' average errors. Ideal per-query scales use test labels and are diagnostics only. |
| Query + prediction, alternating, 10% dropout | Wrong-direction frequency rises from **38.48% to 44.38%**, yet harmful-overshoot squared-error increases fall from **139.717 to 3.633**. Better RMSE does not imply more frequently correct directions. |
| Query without prediction, sequential, 20% dropout | Car seed 40 applies **+60.490 when -0.579 is needed**; final RMSE reaches **1.686933**. Dropout does not bound regression outputs. |

The direction/overshoot percentages above pool **6,339 regression evaluations per model configuration across seeds**. Classification counts pool 2,193 evaluations. Larger datasets receive more weight in these diagnostics than in the family means, and queries can recur across seeds. Observed wrong-direction updates do not imply all those errors are learnably avoidable.

The original adapter improves its own selected-core mean from **10.385% to 9.945%** in classification and **0.352137 to 0.338910** in regression. Adaptation therefore has measurable value, even though it can harm many individual predictions. Counts, severity and retrieval quality must be examined together. [8: Original behavior report](ADAPTATION_BEHAVIOR_2026-11-25.md), [saved scale probes](../results/adaptation_improvement/am1/behavior/scaling_summary.csv), [latest diagnostics](../results/adaptation_improvement/ad1/behavior_totals.csv), [largest harmful cases](../results/adaptation_improvement/ad1/harm_examples.csv).

## 6. Comparison with conventional neural and tree models

The `aq1` study also trained compact local MLP, TabM, CatBoost and distance-weighted kNN searches on the same preprocessed splits. MLP/TabM tested widths 64/128; CatBoost depths 4/8; kNN used 3/5/10/20 neighbors. These references were reused afterward.

| Compact local baseline | Classification error % | Regression RMSE |
| --- | ---: | ---: |
| MLP | 13.922 | 0.390093 |
| TabM | 14.259 | 0.389783 |
| CatBoost | **4.517** | **0.318980** |
| kNN | 14.989 | 0.496592 |

Original and query-aware NN-kNN beat these compact neural searches on family means, but they still trail the neural models on several individual datasets. Retrieved-only ExtraTrees reaches **4.350% / 0.314960**, and the latest retrieved-only Random Forest reaches **3.325% / 0.314232**, both lower than CatBoost on these means. NN-kNN benefits from earlier retrieval pretraining and historical configuration selection, so these comparisons do **not** establish equal-compute superiority to tuned baselines or reproduce the unavailable historical benchmark. The direct ExtraTrees head on the same frozen representation reaches **7.365% / 0.339633**; pair trees beat this control under the declared settings, with different training examples/objectives. [4: Baseline budgets](QUERY_AWARE_LOCAL_2026-11-25.md), [11: Merged alternatives and control](NNCDH_ALTERNATIVES_REPORT_2026-11-27.md).

## 7. First replacement study: individual-case models (`ar1`)

Study `ar1` changes the adaptation architecture: keep each retrieved case separate, predict its query-case difference or class relationship, then average the resulting case-based predictions with the unchanged retrieval activations. Three alternatives were tested on all 14 tasks and three seeds. All retrieval parameters and predictions stay frozen; all candidate selection uses validation only.

| New method | Classification error % | Regression RMSE |
| --- | ---: | ---: |
| Local weighted ridge | 11.715 | 0.378717 |
| Pairwise neural, symmetric classification / antisymmetric regression | 9.793 | 0.357450 |
| **Pairwise ExtraTrees** | **4.149** | **0.320848** |
| Validation choice among retrieval and the three families | 4.099 | 0.336200 |

**Regression is the stronger result.** Pairwise trees beat retrieval on **7/7 dataset means and 20/21 seed pairs**, and original sequential 20% dropout on **7/7 means and 17/21 seed pairs**. They reduce wrong-direction frequency from **44.00% to 36.44%** against that matched NN-CDH baseline. Harmful overshoot frequency is similar (**6.20% to 6.33%**), but its squared-error harm falls from **21.051 to 6.882**. Trees still worsen **44.88%** of individual regression predictions.

**Classification gains are concentrated.** Trees fix 54 retrieval errors and introduce 10 harmful flips, versus 22/15 for original sequential 20%. They win/tie/lose 3/2/2 dataset means against retrieval. Excluding both Zebra tasks as a sensitivity check, trees reach **4.445%**, slightly worse than retrieval's **4.382%**. Both tasks remain in primary results.

Pairwise neural and local ridge retain severe Car failures: a query needing **−0.579** receives **+11.827** or **+17.163**, respectively. Validation chooses neural on all Car splits, so the broader selection policy underperforms the tree family in regression. The largest tree failure still overshoots: **−2.006** when **−0.180** is needed on Body Fat.

The pairwise approach draws on published difference-learning and twin-regression methods; the tree learner itself is conventional ExtraTrees. This is an experimental integration, not proof that pair learning beats an equally tuned standalone tree model. Frozen retrieval, different objectives/search spaces and repeated use of these splits limit causal and generalization claims. Defaults remain unchanged. [9: Full report, literature and protocol](ADAPTER_PAIRWISE_REDESIGN_LOCAL_2026-11-26.md), [paired comparisons](../results/adaptation_improvement/ar1/comparisons.csv), [behavior](../results/adaptation_improvement/ar1/behavior_totals.csv).

## 8. More alternatives and a direct-tree control (`ar2`)

The follow-up `ar2` is merged with the first replacement study in [Alternatives to NN-CDH](NNCDH_ALTERNATIVES_REPORT_2026-11-27.md). Its complete results are:

| New design / control | Classification error % | Regression RMSE |
| --- | ---: | ---: |
| Retrieved-only pair trees | 4.350 | **0.314960** |
| Direct ExtraTrees head | 7.365 | 0.339633 |
| Cross-fitted anchor transport | 7.717 | 0.328438 |
| Retrieval-error memory | 9.238 | 0.342571 |
| Neural vote attention | 12.073 | 0.346362 |
| Validation choice: retrieval + all seven alternatives | **4.019** | 0.328706 |

Retrieved-only trees improve all seven regression means versus retrieval and original sequential 20%, and six versus the earlier mixed-pair trees. Wrong-direction frequency falls to **34.04%**; harmful-overshoot frequency rises to **7.21%**, but its squared-error harm falls to **3.097**. They fix 52 classification retrieval errors and introduce seven. Their five-task classification mean excluding both Zebra tasks is **4.272%**, versus **4.382%** for retrieval; the gain is still predominantly in the Zebra tasks.

Cross-fitted transport is the strongest creative follow-up, but uses five forests and trails pairwise trees. Error memory gives small gains. Attention constrains outputs to the neighborhood's target support but makes little net useful correction; it selects its initial checkpoint in 10/21 classification and 4/21 regression runs. All alternatives remain opt-in. [10: Designs, support limits, per-task results and failure analysis](NNCDH_ALTERNATIVES_REPORT_2026-11-27.md).

## 9. Ordinary Random Forest on matching pairs (`ar3`)

The owner requested the same pair formulation with an ordinary Random Forest. All 252 new units are complete. Both recipes retain identical frozen representations, training-pair construction, sample weights, targets, pair reversal, final retrieval weights and validation-only leaf-size selection. The comparison uses 64 trees, depth 12, max_features 0.7 and minimum leaf sizes 5/20. Random Forest uses best-threshold splits and bootstrap sampling; ExtraTrees uses randomized thresholds without bootstrapping, so both estimator differences are included. [11: Protocol](../docs/ADAPTER_RANDOM_FOREST_EXPERIMENT.md), [full comparison](NNCDH_ALTERNATIVES_REPORT_2026-11-27.md).

| Pair learner | Classification error % | Regression RMSE |
| --- | ---: | ---: |
| Mixed ExtraTrees | 4.149 | 0.320848 |
| Mixed Random Forest | 3.636 | 0.321912 |
| Retrieved-only ExtraTrees | 4.350 | 0.314960 |
| Retrieved-only Random Forest | **3.325** | **0.314232** |
| Expanded validation choice including RF | 3.586 | 0.317036 |

Retrieved-only RF wins/ties/loses **2/5/0 classification dataset means**, and **4/17/0 seed pairs**, against its matching ExtraTrees reference. Excluding both Zebra tasks gives **4.049% versus 4.272%**, with that gain entirely from Iris. Its regression comparison is **4/0/3 dataset means and 12/0/9 seed pairs**. Mixed RF improves classification but slightly worsens regression. This supports testing both forests, not treating RF as a universal replacement.

Correction behavior reinforces the qualification. Retrieved-only RF fixes **60** classification retrieval errors and introduces **6**, versus ExtraTrees' **52/7**. In regression, wrong-direction frequency falls **34.04% to 33.38%**, but harmful overshoots rise **7.21% to 7.70%** and total harmful squared-error increases rise **50.741 to 67.291**. Benefit also rises, **149.702 to 174.293**. The expanded validation policy avoids the previous Car seed-40 pair-neural failure using validation alone, but its aggregate means still trail the fixed retrieved-only RF family. [Detailed behavior](../results/adaptation_improvement/ar3/behavior_totals.csv), [selections](../results/adaptation_improvement/ar3/selections.csv).

## Practical starting points for classification and regression

**The leading redesign is retrieved-only pairwise trees: Random Forest is the current classification candidate, while regression should retain both RF and ExtraTrees.** Within the NN-CDH family, our working preference remains query-aware with prediction for classification and the original adapter with dropout for regression. These are starting points for validation and fresh-split confirmation, not established universal winners.

| Task | Practical starting point | Supporting evidence |
| --- | --- | --- |
| Classification within NN-CDH | **Query-aware with prediction**; prioritize joint or alternating training and retain zero dropout as an option | Without dropout, joint and alternating give **9.359%** and **9.502%** error. Selecting among all three schemes using validation, including sequential, reaches **8.745%**. |
| Regression within NN-CDH | **Original adapter, sequential training**; compare 0%, 10% and 20% dropout using validation | Fixed 20% dropout reduces RMSE from **0.350429 to 0.338523**, improving **6/7 dataset means** with retrieval exactly fixed. Fixed 10% reaches **0.339674** and also improves 6/7. |
| Classification redesign | **Retrieved-only pairwise Random Forest**; retain other pair recipes for validation | **3.325% error**, versus matching ExtraTrees' **4.350%**; improvements occur on Iris and Zebra Special, with five dataset means tied. |
| Regression redesign | **Retrieved-only pairwise ExtraTrees and Random Forest with frozen retrieval**; validation selects leaf size | **0.314960 and 0.314232 RMSE**; both improve all seven means against retrieval and original sequential 20%. RF's small additional gain accompanies greater harmful-update severity. |

The practical preference differs from simply choosing the lowest observed test mean:

- **Before the pairwise redesign**, classification's numerical leader was query-aware **without prediction**, alternating training with dropout: **6.701%** error at fixed 20%, or **6.453%** with a validation-selected rate. Retrieved-only RF now gives **3.325%**, while the expanded validation policy gives **3.586%**. That policy never selects retrieved-only RF in classification, illustrating why the lowest retrospective family mean is different from a validation-selected rule. Gains over retrieval remain concentrated on the two Zebra tasks.
- **The lowest new regression mean is retrieved-only RF at 0.314232, close to ExtraTrees at 0.314960.** The earlier probability gate (**0.337206**) and query + prediction, alternating, 10% dropout (**0.337381**) remain useful references. Original sequential 20% (**0.338523**) is the clearer fixed-retrieval NN-CDH baseline.

Select scheme and dropout using validation, with zero dropout available. The **8.745%** classification result comes from selection among all three schemes; it is not a measured result for a joint-versus-alternating-only selection policy. These recommendations do not change the implementation's original default. [6: Training-scheme evidence](ADAPTER_TRAINING_SCHEMES_LOCAL_2026-11-26.md), [7: Dropout evidence](ADAPTER_DROPOUT_LOCAL_2026-11-26.md), [2: Gate evidence](ADAPTATION_GATE_LOCAL_2026-11-25.md).

## Current conclusions and remaining ideas

| Topic | Status supported by the completed experiments |
| --- | --- |
| Magnitude/direction separation | Shelved after worse overall results |
| Simple disagreement gate | Useful comparison baseline; modest gains, especially small in regression |
| Learned gate or threshold | No consistent advantage that justifies making it the default |
| Query representation | Worth retaining as an option; its benefit depends on family and training scheme |
| Removing prediction input | Verified direct-path removal; no general performance or stability benefit |
| Training scheme | Material effect; sequential, alternating and joint cannot be treated as interchangeable |
| Dropout | Useful hyperparameter; original sequential regression provides the clearest isolated benefit |
| Non-NN-CDH pairwise trees | Retrieved-only training leads regression; direct-tree control supports the recipe; classification gains concentrate on Zebra tasks; fresh-split confirmation remains needed |
| Random Forest versus ExtraTrees | Retrieved-only RF improves classification on two dataset means; regression is nearly tied and RF causes more severe harmful updates; keep both learners |
| Forest training schemes | Only sequential frozen retrieval has been tested; joint/alternating adaptation feedback needs a custom aggregation/refit loop |
| Transport / memory / attention alternatives | Transport helps but trails pair trees; memory gains are small; bounded attention mostly loses useful adaptation |
| Overall default | Original remains unchanged; observed winners need confirmation on fresh splits |

The earliest discussion also considered a continuously learned residual multiplier, alpha-to-one penalties, additional correction-size penalties and explicit bounds. Those proposed ablations were not carried out as a separate matched study here. Global shrinkage, magnitude/direction separation and hard apply-or-skip gates are the related experiments actually completed. Gating the strongest dropout adapters, held-out routing supervision and fresh-split confirmation remain untested. [Original discussion and decisions](../../../discussion/2026-11-24.md).

## Storage and verification

Run units, weights, predictions and TensorBoard curves remain under `NNKNN_RUNS/t1/<study>/`; generated tables and figures are in this worktree's `results/adaptation_improvement/<study>/`. Written reports remain in `reports/`. The [project results index](../../../results/INDEX.md) records drivers, protocols, run locations and reuse checks for every study.

The source reports document prediction reconciliation, paired split checks, frozen-core comparisons, completed-unit reuse and targeted implementation tests. The three replacement studies have 42 focused passing tests and completed-unit reuse audits covering 1,512 units across the studies. The latest merged reader checks 1,050 validation sets, 420 selected tests, 210 NN-CDH references and 42 contexts. All 672 pre-RF comparison rows remain unchanged. This consolidation checks its numerical tables against those saved results and verifies its local links. It adds no further model changes, evaluations, exclusions or retuning.
