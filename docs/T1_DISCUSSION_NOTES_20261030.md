# T1 discussion notes

Date: October 30, 2026. Code and saved results reviewed at `eb326576`.

**Discussion status:** Only points 1-3 have been discussed and recorded; their proposed changes and follow-up experiments are not all complete. Points 4 onward are deferred title-only placeholders, to be re-evaluated after establishing the revised baseline from points 1-3.

**Re-evaluation decision, October 31:** The Zoom observations describe earlier model configurations. They remain historical evidence for those configurations, but must not be assumed to hold after changing pre-Top-K masking, the training setup, or correction-size regularization. Claims about difficult datasets, concentrated retrieval loss, adapter behavior, and the relative value of later interventions require fresh evaluation under the revised configuration.

The next research steps are to:

1. Resolve and apply the agreed baseline decisions from points 1-3, especially all-case softmax without pre-Top-K masking while preserving neighbor-based bias initialization. Declare the training schedule, loss weights, and correction-size penalty explicitly.
2. Retrain and evaluate matched retrieval-only and retrieval-plus-adapter baselines, using controlled splits and seeds. Recompute the relevant diagnostics under that setup.
3. Revisit points 4 onward using the new results, retaining, revising, or dropping earlier hypotheses and proposed experiments as the evidence warrants.

The October 30 pilots below remain evidence for their stated paired comparisons and limitations. They do not establish that the remaining Zoom observations still hold under a fully revised baseline. Preserve earlier results for provenance and comparison, rather than treating them as current conclusions. No new experiment is reported by this re-evaluation decision.

## 1. Disable pre-Top-K masking by default

**Proposed decision:** `CoreConfig` should use `pre_topk_mask=False` by default. The intended baseline is softmax over all eligible cases; hard Top-K truncation should be an explicit experimental option.

**Working hypothesis:** Restricting aggregation to a small number of cases can hurt predictive performance, especially regression, by discarding cases that could collectively improve the softmax-weighted prediction. Whether removing the mask improves a particular task remains an empirical question.

**Expanded experimental finding:** Removing the mask lowers mean held-out error on **10 of 12 datasets**: **5 of 7 regression tasks** and **all 5 classification tasks**. Energy Efficiency and Airfoil favor masking on every seed. These results strengthen the case for testing all-case softmax as the intended baseline while preserving neighbor-based bias initialization. They do not establish a universal performance advantage; the short training budget remains a limitation. Both the original pilot and expanded results are below.

### Keep bias initialization and prediction masking separate

The original use of nearest neighbors to initialize the default bias does not imply that prediction should be truncated to those neighbors. These are independent operations:

- **`bias_init_k` (`K_init`):** used once when constructing the model, to initialize a common case bias from the mean, across training cases, of each case's K-th nearest non-self distance. It does not limit the cases used in subsequent prediction.
- **`top_k` with `pre_topk_mask=True` (`K_mask`):** selects the highest-scoring cases on each query before softmax during training and evaluation. It does not determine `CoreConfig`'s nearest-neighbor bias initialization.

The older `find_default_bias_knn(..., k=...)` also uses K for bias initialization. Calling `torch.topk` to find a neighbor-distance statistic is different from applying a Top-K activation mask. The all-case baseline should preserve neighbor-based bias initialization while disabling prediction truncation.

### What the current configuration does

| Configuration layer | `case_normalizer` | `pre_topk_mask` | `top_k` |
| --- | --- | --- | --- |
| Underlying `NN_KNN_Model` defaults | `softmax` | `False` | 40; does not truncate softmax while the mask is disabled |
| Shared experiment `CoreConfig` defaults | `softmax` | `True` | 5 |
| Frozen Student Performance confirmation configuration | `softmax` | `True` | 10 after the dataset-specific override |

`CoreConfig.to_model_kwargs()` copies the underlying defaults and then explicitly overrides `pre_topk_mask` with the `CoreConfig` value. Thus, the underlying model's all-case default does not determine the experiment behavior.

Sources: [underlying defaults](../model/nnknn_model.py), [CoreConfig and model construction](../model/common/core.py), [Student Performance configuration](../configs/t1/confirm_final_20261013/regression__student_performance.yaml), and [dataset-specific configuration application](../tools/t1_run.py).

### Does the mask affect training, validation, or testing?

**Verified by tracing the current code: it affects all three.** The mask is applied inside `NN_KNN_Model.retrieve()` whenever `self.config["pre_topk_mask"]` is true. That condition does not depend on `model.training`. Calling `model.eval()` does not disable it.

| Stage | Code path | Effect when the mask is enabled |
| --- | --- | --- |
| Retrieval-core training | `train_retrieval()` calls `model(...)`, whose forward pass calls `retrieve()` | Only the retained cases contribute to each query's prediction and associated prediction-loss gradient. The mask changes learning, not just reporting. |
| Validation and checkpoint selection | `evaluate()` calls the same model's `retrieve()` | Validation predictions use the masked weights, so the mask also affects early stopping and checkpoint selection. |
| Retrieval-only testing | The ablation runner evaluates the trained model through its prediction path | Test predictions use the same configured mask. There is no automatic switch to all-case aggregation. |
| Reuse-adapter training and evaluation | Frozen-core reuse and synchronized training obtain weights from `retrieve()` or `retrieve_chunked()` | The mask affects the aggregate prediction and neighborhood representation supplied to the adapter during training and evaluation. |

Relevant locations:

- `model/common/core.py`: `train_retrieval()` around lines 249-267; `evaluate()` around lines 321-345.
- `model/nnknn_model.py`: `retrieve()` around lines 1199-1216; `forward()` calls `retrieve()` around line 1275.
- [Synchronized retrieval/reuse](../model/t1/sync.py): `_forward_terms()` around lines 153-169, used by the training updates.
- [Frozen-core reuse](../model/t1/reuse.py): `neighborhood_inputs()` around lines 82-97; regression training inputs around lines 279-287; regression evaluation around lines 391-401.
- [Core ablation runner](../tools/t1_core_ablation.py): `fit_variant()` and `run_trial()` around lines 122-147.

Training still uses leave-one-out retrieval: a training query's own case is excluded by stable case ID. This self-exclusion is separate from Top-K truncation and should remain in place. "All eligible cases" means the permitted active training-memory cases, not validation/test examples or their labels.

### What changes mathematically?

For query `x`, the current score is `s_i = b_i - d(x, c_i)`, where `b_i` is the learned case bias. Without truncation:

```text
w_i = exp(s_i / tau) / sum_j exp(s_j / tau)
y_hat = sum_i w_i * y_i                    # retrieval-only regression
```

The sums run over all eligible cases. This is a softmax-weighted average, not a uniform average. Distant or low-scoring cases can receive very small weights.

With the mask enabled, the model keeps the K highest scores, replaces all other scores with negative infinity, and then applies softmax. Excluded cases have exactly zero weight, and retained weights are renormalized. Selection is by bias-adjusted score, not necessarily distance alone.

A small K can discard useful contributions and make regression predictions depend on very few targets. Conversely, including distant or misleading cases can introduce bias. Therefore, the proposal to remove hard truncation is well motivated, but higher accuracy or lower RMSE is not guaranteed on every dataset.

### Evidence already available for this point

The committed exploratory study already includes `k_all`, defined as `pre_topk_mask: false`. That variant is trained and evaluated without the mask; it is not an inference-only toggle.

Examples from the [core-ablation summary](../results/t1/p6_confirmation_revision/core_ablation_20261027/summary.md), with exact values in [summary.json](../results/t1/p6_confirmation_revision/core_ablation_20261027/summary.json):

| Regression task | Masked baseline, K=10 | All-case softmax (`k_all`) | Observed direction |
| --- | ---: | ---: | --- |
| Student Performance | 0.3964 | 0.3297 | Lower error without truncation |
| Energy Efficiency | 0.0500 | 0.0531 | Higher error without truncation |

Values are mean test RMSE in training-standardized target units over seeds 30-34, with lower values better. These are retrieval-only exploratory results, not new confirmation results and not evidence that the reuse adapter improves after removing the mask. They support examining this issue, while showing that the benefit is task dependent.

### New small paired pilot, October 30

Ran four small datasets with three paired seeds (70-72), for 24 fits. Both arms use `bias_init="knn_mean"` and `bias_init_k=5`. The masked arm uses `top_k=5`; the all-case arm differs only in `pre_topk_mask=False`. Every pair starts with exactly identical model state and bias, checked by SHA-256. The two K parameters happen to have the same numeric value here, but serve the separate purposes described above.

| Dataset | Held-out metric, lower is better | Top-5 mask | All-case softmax | Change in mean error | All-case better / worse / tied seeds |
| --- | --- | ---: | ---: | ---: | --- |
| Synthetic linear regression | Standardized-target RMSE | 0.3242 | **0.2375** | -26.8% | 3 / 0 / 0 |
| Yacht | Standardized-target RMSE | 0.2687 | **0.1771** | -34.1% | 2 / 1 / 0 |
| Energy Efficiency | Standardized-target RMSE | **0.0543** | 0.0882 | +62.3% | 0 / 3 / 0 |
| Iris | Classification error rate | 0.0444 | **0.0111** | -75.0% | 2 / 0 / 1 |

Relative change is `100 * (all-case mean error / masked mean error - 1)`, calculated from unrounded means; negative is better. The Iris percentage describes error reduction, not a 75-percentage-point accuracy increase.

![Held-out errors with and without the pre-Top-K mask, paired by seed](../results/t1/p6_confirmation_revision/pre_topk_pilot_20261030/comparison.png)

Each thin line connects the same seed; diamonds show means. Panels use separate vertical scales, and lower is better in every panel.

All-case softmax lowers mean error on three of these four datasets, including two of the three regression tasks. Energy Efficiency favors masking on every seed, and Yacht has one seed favoring masking. These outcomes do not establish that either setting is universally better.

This is a short, untuned, retrieval-only pilot: 60/20/20 splits, raw features, one feature-weight set, softmax temperature 1, batch size 64, maximum 120 epochs, and validation patience 20. Nine of twelve unmasked runs select the final allowed epoch; the remaining three select epochs 116-118. Thus, convergence of the all-case condition is not established. No reuse adapter was trained.

The [pilot report and paired-seed figure](../results/t1/p6_confirmation_revision/pre_topk_pilot_20261030/summary.md) include variation across seeds and links to all predictions, initial-state checks, actual runtime versions, configuration, and code provenance. Independent recalculation from saved predictions matches every reported test error. Both initial training and test retrievals retain at most five positive weights in the masked arm; leave-one-out self-exclusion is verified separately.

### Expanded paired comparison, October 30

Expanded to **12 datasets, 3 paired seeds (70-72), and 72 fits**. Eight datasets were added: Wine, Breast Cancer, Balance Scale, Digits, Airfoil, Student Performance, synthetic mixture of two linear models, and synthetic redundant features. These tasks were specified before inspecting the expanded results. The four original tasks were rerun as well; all 24 original fits reproduce exactly, including predictions and selected epochs. The expanded table includes those original tasks, so the two batches should not be pooled as independent evidence.

The comparison retains the original protocol: raw features, one feature-weight set, softmax temperature 1, batch size 64, maximum 120 epochs, validation patience 20, and 60/20/20 train/validation/test splits. Both arms use `bias_init="knn_mean"` and `bias_init_k=5`; only `pre_topk_mask` changes, with `top_k=5` used for truncation in the masked arm. All 36 pairs have identical initial model state and bias. Student Performance's category vocabulary is fitted on its training split only, followed by training-only feature/target standardization through the regression workflow.

| Dataset | Held-out metric, lower is better | Top-5 mask | All-case softmax | Change in mean error | All-case better / worse / tied seeds |
| --- | --- | ---: | ---: | ---: | --- |
| Synthetic linear regression | Standardized-target RMSE | 0.3242 | **0.2375** | -26.8% | 3 / 0 / 0 |
| Yacht | Standardized-target RMSE | 0.2687 | **0.1771** | -34.1% | 2 / 1 / 0 |
| Energy Efficiency | Standardized-target RMSE | **0.0543** | 0.0882 | +62.3% | 0 / 3 / 0 |
| Iris | Classification error rate | 0.0444 | **0.0111** | -75.0% | 2 / 0 / 1 |
| Wine | Classification error rate | 0.0463 | **0.0093** | -80.0% | 2 / 0 / 1 |
| Breast Cancer | Classification error rate | 0.0409 | **0.0292** | -28.6% | 2 / 1 / 0 |
| Balance Scale | Classification error rate | 0.1280 | **0.1253** | -2.1% | 2 / 1 / 0 |
| Digits | Classification error rate | 0.0185 | **0.0157** | -15.0% | 2 / 1 / 0 |
| Airfoil | Standardized-target RMSE | **0.3848** | 0.4156 | +8.0% | 0 / 3 / 0 |
| Student Performance | Standardized-target RMSE | 0.5343 | **0.4175** | -21.9% | 3 / 0 / 0 |
| Synthetic mixture of two linear models | Standardized-target RMSE | 0.8350 | **0.7976** | -4.5% | 3 / 0 / 0 |
| Synthetic redundant features | Standardized-target RMSE | 0.1187 | **0.0951** | -19.9% | 3 / 0 / 0 |

Values are means over three seeds. Classification error rates are fractions. Relative changes use unrounded means and the same formula as the original pilot; negative is better. Mean errors should not be averaged across these different tasks and metrics. Standard deviations are available in the linked expanded report.

![Expanded comparison of pre-Top-K masking on 12 datasets](../results/t1/p6_confirmation_revision/pre_topk_expanded_20261030/comparison.png)

Each thin line connects the same seed; diamonds show means. Panels use separate vertical scales. Lower error is better in every panel.

**Interpretation:** All-case softmax wins on mean error for 10 of 12 tasks and in 24 of 36 task-seed pairs, with 10 worse pairs and 2 ties. The favorable direction now extends beyond the original four datasets. Student Performance improves on all three seeds, while the small mean improvement on Balance Scale has mixed seed-level outcomes. Energy Efficiency and Airfoil are consistent counterexamples to a blanket claim that removing the mask always helps.

**Limits:** This is exploratory, retrieval-only evidence from one common configuration, not per-task tuning, a test of every K, or evidence about the reuse adapter. Twenty-two of 36 unmasked runs select the final allowed epoch, so convergence is not established. The new Student Performance results use a different protocol from the older tuned table above and should be compared within this batch, not pooled with that historical study.

**Verification:** Independently recalculated all 72 test errors from saved predictions and checked all 36 pairs for identical initialization, matching data, disjoint splits, and a single configuration difference (`pre_topk_mask`). The import smoke passes. The branch suite again reports 375 passed, 7 failed, and 1 skipped, with the same missing-baseline-metadata and exact historical-output failures as the original pilot; the suite is not fully passing. Shared core source and historical configurations are unchanged.

See the [expanded report](../results/t1/p6_confirmation_revision/pre_topk_expanded_20261030/summary.md), [configuration](../configs/t1/pre_topk_expanded_20261030.yaml), and [validation record](../results/t1/p6_confirmation_revision/pre_topk_expanded_20261030/validation.json). The report links to individual predictions, epoch histories, source/runtime provenance, and an independent validation script.

### Follow-up for point 1

1. Use explicit `pre_topk_mask: false` in new all-case configurations. Changing the class default alone will not override an existing YAML setting of `true`.
2. Address the training-budget limit with longer matched runs before concluding that the all-case models have converged. Continue matching splits, seeds, initialization policy, learning rates, and temperature; an inference-only toggle answers a different question from retraining each condition.
3. For reuse comparisons, train the adapter using the corresponding neighborhoods as well. Report retrieval-only and adapted results separately, using RMSE/MSE for regression and error rate for classification.
4. Keep Top-K available for an explicit ablation and for separate explanation/diagnostic uses. Retain leave-one-out and case-eligibility masks.

The pilot adds a T1 experiment runner and configuration. Shared model defaults and historical experiment configurations have not been changed. A later change to the shared `CoreConfig` default must follow the [main-first core synchronization procedure](NNKNN_CORE.md).

## 2. Can pre-Top-K masking contribute to retrieval degradation during joint fine-tuning?

### Training schedule and existing observations

The final F pipeline uses retrieval pretraining, adapter-only warm-up with retrieval frozen, then joint fine-tuning. The saved configurations specify joint training on 29 tasks, with frozen-core reuse on diabetes. Pretraining selects the best validation checkpoint within an epoch budget; it does not certify optimization convergence. Student Performance's F configuration uses up to 300 pretraining epochs with patience 40, followed by five adapter warm-up epochs before joint updates.

The final confirmation's standalone retrieval error is numerically worse on 17 jointly configured tasks and better on 12; frozen diabetes is unchanged. All ten transfer-task point estimates favor the pretrained retrieval-only reference. These are directions of reported paired error ratios, not counts of statistically significant differences. The pooled pre-adaptation error ratio is 1.003, and 119 of 400 analyzed non-diabetes F runs selected a checkpoint before retrieval/joint updates. Those selected checkpoints retained the pretrained retriever.

An earlier single-seed Superconductivity development run showed a more extreme case: retrieval-only RMSE 0.5215 became 1.3719 before adaptation after joint training, while the adapter brought final RMSE to 0.4051. The final confirmation's Superconductivity result was much less severe. Prediction error/RMSE and raw retrieval loss (classification NLL or regression MSE) must be distinguished.

Sources: [final confirmation report](../results/t1/p6_confirmation_revision/confirmation_report_20261020/report.json), [checkpoint analysis](../handoff/NEURAL_CBR_PROGRESS_2026-10-20_EN.md), and [earlier development experiment, Table 44](../handoff/NEURAL_CBR_PROGRESS_2026-10-13_EN.md).

### Interaction hypothesis connecting points 1 and 2

Pre-Top-K masking could make joint fine-tuning more fragile: changes in retrieval scores can abruptly replace a case at the K-th boundary, and cases outside the retained set have zero weight and no gradient through their masked score for that query. The adapter may compensate for a weaker standalone retriever. All-case softmax instead changes weights continuously with the scores.

This is a plausible mechanism, not an established cause. Both the historical pretrained reference and its jointly trained continuation used masking, so the mask's presence alone does not explain their difference. The question is whether masking interacts with the subsequent retrieval updates. Point 1's retrieval-only pilot does not answer that question because it trained no adapter.

### Clean four-arm pilot

| Retrieval setting | Frozen retrieval + adapter | Joint fine-tuning |
| --- | --- | --- |
| Top-5 mask | A | B |
| All-case softmax | C | D |

The six small legacy datasets are Iris, Wine, Breast Cancer, Yacht, Energy Efficiency, and Student Performance; seeds are 70-72. This requires 36 retrieval pretrains and 72 frozen/joint continuations. No transfer datasets or large benchmark sweep are included.

Within each mask setting, the frozen and joint arms start from the same pretrained model and corresponding Adam state. Both initialize the adapter identically, perform the same five-epoch adapter warm-up, and use the same minibatch order and continuation objective. Only retrieval trainability differs. The frozen arm uses the matched summed objective too, so this isolates freezing without also changing the adapter's loss weights; it is not an exact replication of the historical sequential adapter objective.

Both arms preserve `bias_init_k=5`; `top_k=5` only controls truncation when `pre_topk_mask=True`. Pretraining has a 300-epoch cap and patience 40; continuation has an 85-epoch cap including warm-up and patience 20. Checkpoint selection uses validation post-adaptation loss. No retrieval guard blocks harmful updates in this diagnostic pilot.

Measure standalone retrieval loss and prediction error at pretraining, at the validation-selected continuation checkpoint, and at the last executed continuation checkpoint. Also measure final adapted loss/error. Recording the last state prevents an early selected checkpoint from concealing damage later in training; the selected and last states answer different questions.

For each dataset/seed, define retrieval-loss drift as `L_pre_after - L_pre_pretrained`, and the interaction as `drift_all_case_joint - drift_masked_joint`. Negative interaction means a more favorable change with all-case softmax. It is evidence of reduced deterioration only when the masked condition actually deteriorates; it is not a proof of the proposed neighborhood-change mechanism. Frozen-arm retrieval must remain exactly unchanged as an integrity check.

The experiment applies its mask setting throughout pretraining and continuation. Thus it tests the complete pipeline, not a mask toggle applied only after a shared cross-mask pretraining checkpoint. The common untuned settings also differ from the historical per-task F configurations.

Pilot configuration: [mask-by-freezing YAML](../configs/t1/mask_joint_pilot_20261030.yaml).

### Point 2 pilot results

Completed all **36 retrieval pretrains and 72 frozen/joint continuations**. All 36 frozen-versus-joint pairs share identical pretrained parameters and warm-up endpoint hashes. All 36 frozen retrievers remain exactly unchanged, including their test predictions and raw retrieval loss. Saved predictions independently reproduce every reported loss and error.

**Main finding:** Removing masking produces a more favorable mean retrieval-loss drift on **4 of 6 datasets**. At the validation-selected checkpoint, retrieval loss deteriorates in **5 of 18 masked joint runs**, compared with **1 of 18 all-case joint runs**. At the last executed checkpoint, the counts are **11 of 18** versus **7 of 18**. These are exploratory counts of task-seed outcomes, not independent-task significance tests.

The table shows mean change in raw retrieval loss from each condition's own pretrained checkpoint. Positive is worse; negative is better. NLL and MSE have different units and should not be pooled across tasks.

| Dataset | Raw retrieval loss | Masked joint: selected drift | All-case joint: selected drift | Masked joint: last drift | All-case joint: last drift |
| --- | --- | ---: | ---: | ---: | ---: |
| Iris | NLL | +0.00410 | -0.00027 | +0.01260 | -0.00003 |
| Wine | NLL | -0.00795 | -0.00262 | -0.00897 | -0.00470 |
| Breast Cancer | NLL | +0.00285 | +0.00026 | +0.06720 | +0.00457 |
| Yacht | MSE | -0.01256 | -0.00325 | -0.01064 | -0.00327 |
| Energy Efficiency | MSE | -0.00006 | -0.00078 | +0.03202 | -0.00076 |
| Student Performance | MSE | +0.00442 | +0.00000 | +0.00886 | +0.00377 |

![Retrieval-loss drift during joint fine-tuning, with and without pre-Top-K masking](../results/t1/p4_component_sync/mask_joint_pilot_20261030/retrieval_loss_drift.png)

Thin lines pair seeds; diamonds show means at the validation-selected checkpoint. The dashed zero line is no retrieval-loss change from pretraining. Panels have independent vertical scales.

The per-dataset interpretation matters:

- **Iris:** masked joint training raises mean retrieval NLL; all-case joint training slightly lowers it. Classification error remains unchanged in both conditions, so this is loss degradation without accuracy degradation.
- **Breast Cancer:** both settings raise mean retrieval NLL, but the increase is smaller without masking. Masked retrieval classification error nevertheless improves from 0.0409 to 0.0322, illustrating why loss and accuracy must be reported separately.
- **Student Performance:** mean retrieval MSE deteriorates with masking and stays unchanged without masking at the selected checkpoints. However, all three all-case runs select warm-up checkpoints before joint updates; this demonstrates avoidance of harmful updates by selection, not that retaining unmasked joint updates was harmless. At the last checkpoint, both settings deteriorate, with less drift without masking.
- **Energy Efficiency:** selected checkpoints improve retrieval MSE in both conditions. The masked last checkpoint deteriorates substantially, while the all-case last checkpoint still improves. Selection hides that later masked deterioration.
- **Wine and Yacht:** joint training improves mean retrieval loss with both mask settings, and the masked condition improves more from its own starting point. They do not support a universal advantage from removing masking.

Final adapted errors at the validation-selected checkpoints are also mixed. Classification values are error fractions; regression values are standardized-target RMSE, with lower values better.

| Dataset | A: masked frozen | B: masked joint | C: all-case frozen | D: all-case joint |
| --- | ---: | ---: | ---: | ---: |
| Iris | 0.0333 | 0.0222 | 0.0111 | 0.0111 |
| Wine | 0.0463 | 0.0278 | 0.0093 | 0.0093 |
| Breast Cancer | 0.0439 | 0.0380 | 0.0380 | 0.0380 |
| Yacht | 0.0987 | 0.0796 | 0.0926 | 0.0937 |
| Energy Efficiency | 0.0540 | 0.0543 | 0.0667 | 0.0610 |
| Student Performance | 0.4892 | 0.4917 | 0.4201 | 0.4201 |

All-case joint training has lower final adapted mean error than masked joint training on three tasks, higher error on two, and ties on one. A more favorable retrieval-loss drift does not guarantee the best final adapted prediction.

**Checkpoint caveat:** 12 of 18 masked joint runs and 13 of 18 all-case joint runs select a checkpoint after warm-up. The remaining selected retrievers are unchanged from pretraining. Five of the 36 pretrains select the final 300th epoch, so pretraining convergence is not established for every case. This pilot differs from the historical tuned F protocol and is not a replication of its dataset-level results.

**Conclusion for discussion:** The pilot supports investigating pre-Top-K masking as a contributor to retrieval-loss degradation during joint fine-tuning on some small datasets. It does not establish masking as the sole cause, a universal accuracy effect, or the proposed neighborhood-boundary mechanism. The benefit is task dependent and must be assessed together with final adapted error and checkpoint selection.

Full evidence: [four-arm report](../results/t1/p4_component_sync/mask_joint_pilot_20261030/report.md), [run records](../results/t1/p4_component_sync/mask_joint_pilot_20261030/runs.json), and [independent validation](../results/t1/p4_component_sync/mask_joint_pilot_20261030/validation.json). The synthetic integrity tests and import smoke pass. The final T1 plus core suite has **377 passed, 7 failed, and 1 skipped**, retaining the same missing-baseline-metadata and exact historical-output failures as earlier pilots. A new test's torch-state cleanup was corrected; the affected checks pass, and archived test source preserves the experiment's execution fingerprint. Shared core code and historical experiment configurations were not changed.

## 3. Preserve retrieval explanations by penalizing large adaptation corrections

**Design goal:** Retrieval should already produce a useful prediction, adaptation should bring it closer to the target, and the correction should remain small enough that the retrieved answer remains relevant to the final explanation. An adapter that makes a large correction can conceal poor retrieval. A correction-size penalty supports this goal, although a soft penalty cannot guarantee small corrections or faithful explanations on every query.

For retrieval prediction `p0`, adapter correction `c`, and final prediction `y_hat = p0 + c`, the intended regression objective is:

```text
L = alpha * MSE(p0, y)
  + beta  * MSE(p0 + c, y)
  + lambda * mean(||c / s_task||_2^2)
```

The first term trains retrieval to be accurate on its own; the second trains the final adapted answer; the third penalizes the size of the correction itself. `s_task` declares the scale used for that penalty. Small corrections support the relevance of retrieved evidence, but do not by themselves establish that the neighbors are useful explanations.

### Correction accuracy and correction size are different losses

The current name `L_delta` refers to **correction prediction error**, not correction-size regularization:

```text
L_delta = MSE(c, y - p0)
        = MSE(p0 + c, y)
        = L_post                 # regression
```

It rewards the correction needed to reach the target, even when that correction is large. In regression it duplicates final prediction MSE. Its coefficient therefore changes the effective post-adaptation weight; it does not implement the PI's intended small-correction preference.

The current T1 correction-size penalty is instead `L_small`:

```text
L_small = mean((max(0, ||c||_2 - tau_task) / (s_task + eps))^2)
```

This is squared L2 regularization above a free-correction allowance: corrections of size at most `tau_task` cost nothing, and excess correction is penalized quadratically. Setting `tau_task = 0` recovers scale-normalized squared L2 regularization from zero. The older classification adapter also contains `loss_mag = mean(r_hat**2)`, but the synchronized T1 training path uses `L_small`, rather than calling that helper.

Sources: [regression and classification terms](../model/t1/sync.py), [penalty implementation](../model/t1/calibration.py), and [older classification magnitude helper](../model/nn_cdh.py).

### What the evaluated configurations enable

The Student Performance F configuration uses `v_small: 0.1`, `s_task: 0.5`, and the default trained-bias-region calibration. Its summed regression objective simplifies to:

```text
L_total = L_pre + 2.5 * L_post + 0.1 * L_small
```

Inspection of the 29 named synchronized F configurations in `configs/t1/confirm_final_20261013/` finds **19 with `v_small: 0.1` and 10 with `v_small: 0`**. The 30th task, diabetes, uses frozen-core reuse rather than a named synchronized F condition. The frozen-core reuse loss has no `L_small` term. Thus correction-size regularization is implemented, but is not active in every selected final recipe. For example, [Student Performance F](../configs/t1/confirm_final_20261013/regression__student_performance.yaml) enables it, whereas [Yacht F](../configs/t1/confirm_final_20261013/regression__yacht.yaml) disables it.

### How the current default calibrates tau

The default method is `free_radius_method: trained_bias_region`, with mean aggregation and no transformation of the trained biases. Here `tau_task` is the correction allowance, distinct from the retrieval softmax temperature:

1. **Finish retrieval pretraining and restore its selected checkpoint.** The runner snapshots the trained case biases and corresponding learned case-to-case distances before adapter warm-up or joint training. The threshold calculation uses training cases and labels; the retrieval checkpoint itself was selected using validation loss.
2. **Treat each training case `j` as a query against stored cases `i`.** Form the set of directed, non-self pairs `P = {(j, i): j != i and b_i* - d*(x_j, x_i) >= 0}`. Here `b_i*` is the frozen trained bias of the stored case, and `d*` is the learned distance from the shared geometry implementation. Case-specific weights can make these neighborhoods directional.
3. **Measure label variation within those regions.** For scalar regression, each selected pair contributes `abs(y_j - y_i)`. For classification, it contributes Euclidean distance between one-hot labels: zero for the same class and `sqrt(2)` for different classes.
4. **Average all selected pair differences equally.** `tau_task = mean_{(j,i) in P} d_y(y_j, y_i)`. This is one scalar for that task/run, not a query-specific threshold, a mean of per-query means, or an activation-weighted mean. In classification it equals `sqrt(2)` times the selected pairs' class-mismatch fraction. Regression uses the model's target units, which are training-standardized in the T1 regression workflow.
5. **Keep that threshold fixed during ordinary F `joint_rr` training.** It is detached from the gradient graph and does not change with each minibatch or joint epoch. Maintenance-enabled schedules can recalibrate at declared maintenance checkpoints; the ordinary F schedule performs no such maintenance.

For illustration only, selected regression pairs with absolute label differences `0.1`, `0.2`, and `0.3` produce `tau_task = 0.2`. Corrections up to `0.2` then receive no `L_small` penalty. With `s_task = 0.5`, a correction of size `0.3` contributes `((0.3 - 0.2) / 0.5)^2 = 0.04` before multiplying by `v_small`.

This calibration does **not** take a fixed number of nearest neighbors and does **not** apply the pre-Top-K mask. It uses the raw bias-minus-distance condition before normalization. Under softmax, negative raw scores can still receive positive prediction weight; they are nevertheless outside this calibration region. Bias-initialization K, prediction masking K, and the trained-bias free-correction threshold are distinct mechanisms.

If no non-self pair qualifies, the implementation sets `tau_task = 0` and reports `empty_region`. It also reports overly broad regions (more than half of all possible directed non-self pairs) or a near-zero threshold; it does not automatically replace a broad-region estimate. A large threshold can leave many corrections unpenalized. `s_task` and `v_small` are separate configured values, not estimates produced by the tau calibration.

Sources: [calibration](../model/t1/calibration.py), [learned distance](../model/common/geometry.py), [post-pretraining snapshot and runner wiring](../tools/t1_run.py), and [maintenance-only recalibration](../model/t1/sync.py).

### Proposed comparison and reporting

**Proposed decision:** Keep correction-size regularization as an explicit design requirement when retrieval explanations are part of the objective. Compare three conditions using the same all-case retrieval baseline: no size penalty, the current calibrated allowance, and squared L2 from zero. The existing `free_radius_method: zero` switch supports the last condition; no new penalty implementation is needed for this comparison. See the [existing zero-radius ablation](../configs/t1/alignment_sync_plan_20261009.yaml).

Keep the total post-adaptation error weight constant across conditions. Removing or redefining the duplicated regression `L_delta` without compensating its weight would also change the accuracy objective and confound the comparison. Use a declared scale and validation-based penalty-weight selection. Assess retrieval-only error, adapted error, correction magnitude (including its upper tail), and the fraction of corrections exceeding `tau_task`. Current checkpoint selection prioritizes validation adapted loss or error; it does not directly select for small corrections.

This point records the design clarification and a proposed experiment, not new benchmark results. No training implementation or frozen experiment configuration was changed.

**Deferred agenda:** The titles below preserve the remaining Zoom topics for later reassessment. They do not endorse the earlier observations or commit to conducting every proposed experiment before seeing the revised baseline results.

## 4. Learning-rate and optimizer sensitivity

## 5. Dataset characteristics and retrieval failure cases

## 6. Loss concentration among difficult queries

## 7. Adapter outcomes: improved, worsened, and unchanged

## 8. Batch size and matched training-schedule comparisons

## 9. Categorical feature encoding

## 10. Multiple feature-weight sets and ensembling

## 11. MLP baseline and shared encoder

## 12. Case-base maintenance priorities

## 13. Experiment timeline and reinforcement-learning follow-up
