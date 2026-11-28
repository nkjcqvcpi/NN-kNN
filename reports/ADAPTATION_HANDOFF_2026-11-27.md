# Adaptation work handoff: completed experiments and RF gating

Date: November 27, 2026  
Project: Adaptation improvement  
Scope: NN-kNN adaptation experiments and the current RF-gating design.

**Current state:** all recorded adaptation studies listed below are complete through `ar4`. The active research direction is **per-forest, per-pair gating**. Its design is discussed here; **RF gating has not been implemented or evaluated**. Magnitude/direction separation remains shelved, and the original production/default adapter behavior has not been changed.

This is the combined handoff and RF-gating discussion. Start with the completed-work index, then the motivation, three gate variants and training proposal. The final sections identify implementation locations, experiment conventions and the work still needed.

**Preferred design: one shared gate applied to the entire forest's prediction for each query–retrieved-case pair.** A case that passes the gate receives its full forest adaptation. A case that fails the gate retains its original solution. Both kinds of cases remain in the final average with their original retrieval weights.

The motivation is to preserve RF's useful corrections while avoiding harmful ones. Existing results establish that the forest helps overall; they do not establish that we can reliably identify its mistakes in advance.

## Completed adaptation work

The [consolidated adapter results](ADAPTER_RESULTS_SUMMARY_2026-11-26.md) contain the full historical comparisons. The [merged alternatives report](NNCDH_ALTERNATIVES_REPORT_2026-11-27.md) is the primary numerical reference for the forest redesign.

| Study | What we tried | Main finding / current disposition | Report |
| --- | --- | --- | --- |
| `am1` | Original NN-CDH versus separate correction magnitude and direction; shared adaptation representation, neighborhood statistics and residual supervision | New design did not improve overall performance and was shelved. Saved-prediction analysis found direction errors, overshoot and discarded useful corrections. | [Design comparison](ADAPTATION_IMPROVEMENT_LOCAL_2026-11-25.md), [behavior analysis](ADAPTATION_BEHAVIOR_2026-11-25.md) |
| `ag1` | Global on/off, global shrinkage, disagreement threshold, extra adapter training, probability gate and utility gate | Simple disagreement gating gave modest gains. Learned gates did not establish reliable selection; see the denominator analysis below. | [Apply-or-skip gates](ADAPTATION_GATE_LOCAL_2026-11-25.md) |
| `ag2` | One trainable disagreement cutoff, with the adapter frozen or trained alongside it | No consistent advantage over validation-selected disagreement gating. Frozen-adapter regression remained almost always-adapt. | [Trainable threshold](ADAPTATION_THRESHOLD_LOCAL_2026-11-25.md) |
| `aq1` | Query-aware NN-CDH with retrieved prediction as an input; compact MLP, TabM, CatBoost and kNN references | Initial classification mean improved from 9.945% to 9.359%; regression worsened from 0.338910 to 0.345176. Compact baseline searches are not equivalent to historical tuned benchmarks. | [Query-aware adapter](QUERY_AWARE_LOCAL_2026-11-25.md) |
| `aq2` | Query-aware NN-CDH without retrieved prediction as an input | Removed the direct subtraction input path, but gave no general improvement. Mean classification error 10.146%; regression RMSE 0.358695. | [Without prediction input](QUERY_NO_PREDICTION_LOCAL_2026-11-25.md) |
| `as1` | Original, query + prediction, and query without prediction; sequential, alternating and joint training | Training scheme materially changes results. Sequential runs share the same pretrained retrieval. No universal winner. | [Training schemes](ADAPTER_TRAINING_SCHEMES_LOCAL_2026-11-26.md) |
| `ad1` | All three input variants and all three schemes with dropout 0%, 10% and 20% | Dropout helps selectively. Original sequential 20% is the clearest matched regression gain: 0.350429 to 0.338523, better on six of seven datasets. | [Dropout](ADAPTER_DROPOUT_LOCAL_2026-11-26.md) |
| `ar1` | Individual-pair neural adapter, mixed-pair ExtraTrees, and local ridge | Pair trees were the strongest replacement. The tested pair MLP and local ridge did not solve adaptation reliability. | [Pairwise redesign](ADAPTER_PAIRWISE_REDESIGN_LOCAL_2026-11-26.md) |
| `ar2` | Retrieved-only ExtraTrees, direct-tree control, cross-fitted anchor transport, retrieval-error memory and neural vote attention | Retrieved-only pair training strengthened regression. Transport helped but trailed pair trees; memory gains were small; bounded attention mostly discarded useful adaptation. | [Merged alternatives](NNCDH_ALTERNATIVES_REPORT_2026-11-27.md) |
| `ar3` | Ordinary Random Forest on mixed and retrieved-only pairs, matched to the ExtraTrees recipes | RF improved classification in these comparisons; retrieved-only RF and ExtraTrees were close on regression. RF's slightly better RMSE came with more total harmful squared error. | [RF comparisons](NNCDH_ALTERNATIVES_REPORT_2026-11-27.md#random-forest-on-the-same-pair-formulations) |
| `ar4` | Alternating retrieval updates and forest refits: RF/ExtraTrees × mixed/retrieved-only pairs | All four variants completed. Improvements were selective and small; wrong-direction frequency was essentially unchanged. | [Alternating forests](NNCDH_ALTERNATIVES_REPORT_2026-11-27.md#alternating-forests-matched-continuations-ar4) |

### Model and comparison distinctions to preserve

The original NN-CDH is a residual MLP with hidden widths 64 and 32. Its input variants are:

```text
Original:                 [query - neighborhood_mean, retrieved_prediction]
Query-aware + prediction: [query - neighborhood_mean, retrieved_prediction, query]
Query-aware, no prediction: [query - neighborhood_mean, query]

final = retrieved_prediction + adapter_correction
```

Residual output alone does not prevent an adapter from learning to cancel retrieval. Removing the prediction input removes a direct path, not a general mathematical guarantee.

The forest redesign keeps individual cases separate until adaptation. It uses pair features `[query, case, query - case]` in the saved standardized representation. Regression predicts antisymmetric target differences; classification predicts symmetric same-class affinity. It then averages adapted cases using the original activations.

`ar1`–`ar3` freeze retrieval. “Mixed pairs” means retrieved plus random training pairs, not a mixed training schedule. `ar4` alternates two retrieval epochs and a forest refit for five cycles; validation may retain cycle zero. Final loss reaches retrieval through its prediction and case weights, while tree predictions are detached. This is not gradient descent through tree splits.

### Performance references

Lower is better. These means weight the seven datasets in each task family equally after seed averaging; they are not pooled per-query rates.

| Reference | Classification error % | Regression RMSE |
| --- | ---: | ---: |
| Frozen pretrained retrieval | 11.680 | 0.348690 |
| Original NN-CDH, sequential, 20% dropout | 10.804 | 0.338523 |
| ExtraTrees, retrieved-only, sequential | 4.350 | 0.314960 |
| RF, mixed pairs, sequential | 3.636 | 0.321912 |
| RF, retrieved-only, sequential | 3.325 | 0.314232 |
| RF, mixed pairs, alternating | **3.191** | 0.322119 |
| RF, retrieved-only, alternating | 3.255 | **0.313608** |

The lowest observed forest classification mean is mixed-pair alternating RF. The lowest regression mean is retrieved-only alternating RF, only **0.20%** below its sequential baseline. Classification gains concentrate on a few datasets, especially the Zebra tasks. Keep sequential retrieved-only RF as the simple starting point for the proposed gate experiment; these are retrospective exploratory results, not proof of a universal winner.

Within the earlier NN-CDH studies, query + prediction remains a useful classification option, while original sequential NN-CDH with dropout gives the clearest fixed-retrieval regression improvement. The numerically best dropout classification setting was query without prediction + alternating + 20% dropout, but its advantage concentrated on Zebra. See the consolidated report before generalizing that choice.

The initial `am1`/`aq1` comparisons used joint training on 13 tasks and frozen sequential training on Diabetes. Their “original” baseline is therefore different from the all-sequential reference above. A full model's own selected retrieval core is also different from the standalone pretrained retrieval baseline.

## Motivation: useful overall, but frequent harmful corrections

Our pair RF adapts each retrieved case separately, then combines the adapted solutions. All current RF variants use 64 trees, maximum depth 12, and minimum leaf size 5 or 20 selected on validation. Regression learns query–case target differences; classification learns a same-class affinity that is converted into an adapted class distribution.

The completed local suite contains seven classification and seven regression datasets, with seeds 40, 41 and 42. Below, correction counts pool test evaluations across seeds; examples can recur across seeds. Regression counts total **6,339 evaluations** and classification counts total **2,193**. These counts weight larger datasets more heavily than the equal-dataset performance means.

For regression, define the required residual and applied correction as:

```text
r = true target - retrieval prediction
c = final prediction - retrieval prediction
```

A wrong-direction correction has opposite signs for `r` and `c`, with both magnitudes above the diagnostic tolerance of `1e-8`. Every such correction increases absolute error. A harmful overshoot moves in the correct direction but by more than twice the required amount.

| RF pair-training recipe | Sequential: wrong direction | Alternating: wrong direction |
| --- | ---: | ---: |
| Mixed retrieved/random pairs | 2,255 / 6,339 = 35.57% | 2,258 / 6,339 = 35.62% |
| Retrieved-only pairs | 2,116 / 6,339 = **33.38%** | 2,125 / 6,339 = **33.52%** |

**These are final query-level correction rates, not rates for individual trees or individual query–case pairs.** They do not locate the source of the error within the forest or neighborhood.

For sequential retrieved-only RF:

- **43.37%** of regression evaluations have increased squared error, including wrong directions, harmful overshoots and other harmful movements.
- The median wrong-direction correction is **0.01198** and its mean is **0.04005**, in training-target standard-deviation units.
- Excluding wrong-direction corrections of magnitude at most **0.001** still leaves **31.50% of all queries**. The high rate is not merely floating-point noise.
- The largest **10%** of wrong-direction mistakes account for **83.12%** of their added squared error.
- Wrong-direction updates account for **62.062**, or **92.23%**, of the total **67.291** harmful squared-error increase.
- Helpful updates reduce summed squared error by **174.293**, approximately **2.59 times** the total harmful increase. This explains why the final score can be good despite frequent mistakes.

The equal-dataset regression RMSE improves from **0.348690** for frozen retrieval to **0.314232** for sequential retrieved-only RF. Alternating RF reaches **0.313608**, a small additional improvement without a meaningful reduction in wrong-direction frequency.

Classification requires a different interpretation. Sequential retrieved-only RF leaves **96.94%** of labels unchanged, fixes **60** retrieval errors and introduces **6**, reducing errors from **123 to 69**. Probability changes can affect log loss even without changing a label. The 33.38% wrong-direction figure does **not** describe classification.

Sources: [merged alternatives report](NNCDH_ALTERNATIVES_REPORT_2026-11-27.md#direct-rf-direction-and-severity-check), [independent RF audit](../tools/t1_forest_direction_audit.py), [audit evidence](../results/adaptation_improvement/ar4/rf_direction_audit.json).

## Working hypothesis

Some learned pair-adaptation rules may transfer poorly beyond the training situations that support them. A leaf can fit its observed training pairs while extending into regions with weak support or different relationships. A gate could use evidence about the current pair to decide whether that forest proposal is applicable.

This is a **hypothesis**, not an established explanation of the observed errors. Trees already divide feature space into local regions. Errors can also arise from noisy targets, missing information, averaging bias, or interactions among retrieved cases. Many pair rows can share the same original examples, so a large pair count does not imply equally large independent evidence.

Before attributing failures to particular trees or pairs, we need diagnostics at those levels. The final query-level wrong-direction rate alone cannot establish that attribution.

## Three gating variants

| Variant | Decision unit | What a closed gate does | Main consideration |
| --- | --- | --- | --- |
| **Per tree, per pair** | One tree's contribution for one query–case pair | Excludes or suppresses that tree's contribution | Most granular, but must estimate individual-tree competence; tree averaging and the all-closed fallback need explicit definitions |
| **Per forest, per pair — preferred** | The complete forest proposal for one retrieved case | Keeps that case's original solution | Preserves the forest ensemble while allowing useful cases to adapt independently |
| **Per query, after aggregation** | The combined adaptation across all retrieved cases | Keeps the overall retrieval prediction | Simplest control, but accepts or rejects all case corrections together |

Per-tree gating could be useful when different trees are competent in different regions. Related work on **dynamic ensemble selection** estimates local classifier competence and chooses ensemble members for a query. This supports considering the design, but does not establish its effectiveness for our pairwise RF adapter. See [DESlib, JMLR 2020](https://www.jmlr.org/papers/v21/18-144.html).

We prefer per-forest, per-pair gating first because it judges the correction that would actually be applied after tree averaging. Individual trees can make errors that their ensemble reduces. Selecting trees adds another estimation problem and can change that averaging behavior.

### Preferred prediction rule

For a query, let:

- `w_i` be retrieved case `i`'s normalized query-dependent activation, including the existing case-bias effect.
- `s_i` be that case's original solution.
- `a_i` be its forest-adapted solution.
- `g_i ∈ {0,1}` be the pair gate's apply/skip decision.

Then:

```text
final prediction = sum_i w_i * [(1 - g_i) * s_i + g_i * a_i]
```

For regression, `s_i = y_i` and `a_i = y_i + Delta_RF(query, case_i)`. For classification, `s_i` is the case's one-hot label and `a_i` is its adapted class-probability distribution; take the final argmax after averaging.

**Skipping adaptation does not remove the case, set its classification affinity to zero, or redistribute its retrieval weight.** If all gates close, the result is retrieval. If all open, it is the existing forest-adapted prediction, up to numerical precision.

The query-level alternative instead uses a single `g(query)` for every term.

## What earlier gating experiments tell us

Earlier studies `ag1` and `ag2` gated the older NN-CDH adapter's aggregate correction. They did not gate RF or individual retrieved cases.

| Earlier NN-CDH method | Queries adapted % | Wrong direction / all queries % | Wrong direction / applied corrections % | Mean regression RMSE |
| --- | ---: | ---: | ---: | ---: |
| Original full adapter | 100.00 | 41.27 | 41.27 | 0.338910 |
| Validation disagreement gate | 74.70 | 31.06 | 41.58 | 0.337424 |
| Learned probability gate | 57.69 | 23.55 | 40.83 | 0.337206 |
| Learned utility gate | 59.17 | 24.64 | 41.64 | 0.339802 |
| Trainable threshold, frozen adapter | 98.58 | 40.64 | 41.22 | 0.338856 |
| Trainable threshold + adapter | 95.90 | 37.56 | 39.17 | 0.347770 |

The disagreement gate blocks **647 wrong-direction updates** but also **640 helpful corrections**. Its lower all-query wrong-direction rate primarily reflects fewer adaptations. Its accepted corrections are not more reliable in direction, although harmful overshoots decline and final RMSE improves slightly.

Learned gates also change the adapter during training. Against its **own** forced-open proposal, the probability gate changes wrong-direction frequency from **41.99%** to **40.83% among accepted corrections**: a modest selection benefit.

These studies support the possibility of limiting damage, but do not show reliable detection of wrong directions. An always-closed gate achieves zero adaptation mistakes while discarding all adaptation benefit. Acceptance rates, retained benefit and final prediction error must accompany any reported reduction in mistakes.

Sources: [gating report and denominator clarification](ADAPTATION_GATE_LOCAL_2026-11-25.md#wrong-direction-rates-all-queries-versus-applied-corrections), [trainable-threshold report](ADAPTATION_THRESHOLD_LOCAL_2026-11-25.md), [independent gate recount](../results/adaptation_improvement/ag2/gating_direction_denominators.json).

## Gate inputs and training proposal

### Shared gate, pair-specific evidence

Use **one small shared gate model**, such as an MLP, evaluated for each retrieved pair. It produces an application probability `p_i = sigmoid(G_theta(features_i))`. This is a routing probability, not a claim of calibrated confidence.

Candidate inputs are:

- Query/case representations or their differences, and query-to-case distance.
- The case's original retrieval weight.
- The forest's proposed correction or adapted solution.
- Tree-to-tree disagreement in that pair proposal; regression sign agreement is an additional regression-specific diagnostic.
- Support in the reached leaves, preferably including distinct original-example counts rather than only pair-row counts.
- The four neighborhood statistics already discussed: nearest distance, weighted mean distance, activation concentration, and target disagreement.

The four neighborhood statistics are shared by all cases for a query. **They alone cannot distinguish which case to adapt.** Tree agreement also does not guarantee correctness: correlated trees can agree on a biased prediction.

The query's true target is used only to calculate training losses, never as an inference input.

### Obtain gate-training predictions without training on those queries

For the first clean experiment, split original examples into base-model training, gate training, validation and test subsets. Fit retrieval and RF using the base-model subset; gate-training queries retrieve anchors from that subset. Keep retrieval and RF fixed while fitting the gate, and compare gated and ungated methods using the exact same base model.

This consumes data, but makes the comparison interpretable. A later cross-fitting version can use data more efficiently. Folds must be assigned before pair construction, with held-out original examples excluded from both query and anchor roles in base-model fitting. Ordinary pair-row bootstrap out-of-bag predictions do not establish that independence. A fully independent cross-fit must also exclude the held-out queries from supervised retrieval fitting.

Do not refit the forest after gate fitting in the first comparison: that would change the proposals and evidence the gate learned to judge.

### Train using the effect on the final combined prediction

The RF does not need to be differentiable. Cache its pair proposals and the retrieval weights. For each gate-training query, the training unit is the **whole retrieved neighborhood**.

Let `p0 = sum_i w_i * s_i` and `c_i = w_i * (a_i - s_i)`. The latter is a scalar for regression or a class-probability-vector change for classification.

1. Compute the gate probabilities `p_i`.
2. Sample conditionally independent binary decisions `z_i ~ Bernoulli(p_i)`.
3. Form the sampled prediction `p0 + sum_i z_i * c_i`.
4. For each case `i`, hold the other sampled decisions fixed and calculate two final-query losses:
   - `C_off_i`: case `i` retains its original solution.
   - `C_on_i`: case `i` receives its full forest adaptation.
5. Update the gate using the conditional expected cost:

```text
L_i = (1 - p_i) * stop_gradient(C_off_i)
    + p_i * stop_gradient(C_on_i)
```

Only `p_i` carries gradients. Lower on-cost encourages opening; lower off-cost encourages closing. The magnitude of the loss difference automatically gives larger harms or benefits a stronger training signal. Use squared error for regression and probability cross-entropy for classification.

With independent Bernoulli draws and frozen base predictions, averaging these conditional gradient contributions targets the expected **final-query loss under binary gating**. Other cases' decisions are resampled during training; this is not a one-time set of pair labels. The summed detached local-cost expression is a gradient surrogate, so log the actual sampled and deterministic prediction losses separately.

This applies the principle of [local expectation gradients, NeurIPS 2015](https://papers.nips.cc/paper_files/paper/2015/hash/1373b284bc381890049e92d324f56de0-Abstract.html). The estimator principle is established; this particular RF-gating application remains a proposal.

All on/off comparisons reuse cached RF outputs. They require updated weighted sums and losses, not additional tree fitting or repeated RF inference.

### Why aggregate loss matters

A case can become individually closer to the target while the final weighted prediction becomes worse, or an individually unfavorable correction can help cancel another case's error. Training only on whether each case improved in isolation misses these interactions.

The proposed gate therefore learns whether adapting this case helps **the final prediction in its neighborhood**. It targets prediction loss rather than wrong-direction frequency alone, covering harmful overshoot as well.

During deployment, use hard decisions such as `g_i = 1[p_i > 0.5]`. Select checkpoints using actual deterministic hard-gated validation predictions. Stochastic training performance does not guarantee identical deterministic performance. Cross-entropy improvement also does not guarantee classification-accuracy improvement; report both.

## Proposed first comparison and open questions

Start with the retrieved-only sequential RF and keep its retrieval, forest, case store and weights fixed. Compare:

1. The ungated forest.
2. One gate per query, applied after aggregating case corrections.
3. The preferred shared gate applied per forest, per pair.

Per-tree gating is a subsequent comparison if finer selection warrants the extra complexity. Its tree-vote normalization and all-closed fallback are not yet specified. No implementation budget, gate architecture, hyperparameter search or data-split sizes have been finalized.

Use matched base-model training data and gate-training data for the comparisons. The earlier local splits have already been inspected repeatedly; fresh local confirmation is needed before broad claims. Remote execution on r760 is out of scope.

Evaluation should include:

- Final regression RMSE and classification error/log loss, with per-dataset and seed results.
- Gate-open frequency and the frequency of non-negligible applied corrections.
- Wrong-direction frequency both across all queries and conditional on an applied query-level correction.
- Separate pair-level direction diagnostics, explicitly distinguished from final-query diagnostics.
- Total helpful and harmful loss changes, and the severity of the largest failures.
- Helpful corrections discarded, harmful updates prevented, and comparisons with forced-open/forced-closed gates.

The main open question is whether the new pair-specific evidence can identify useful corrections more reliably than the earlier aggregate NN-CDH gates. Lowering the all-query mistake rate by adapting less is insufficient on its own; the intended improvement is **less harmful adaptation while retaining enough useful adaptation to improve final performance**.

## Continuing the work

RF gating is the next experiment, not a completed result. The gate architecture, data fractions, training budget and new study ID remain to be specified. Continue in this order:

1. Write the matched protocol for ungated RF, a per-query gate and the preferred per-pair gate. Reserve gate-training examples before fitting either retrieval or the forest; keep validation and test separate. Start with sequential retrieved-only RF.
2. Implement pair-proposal caching and the shared gate in the T1 adaptation worktree. Train on complete query neighborhoods using the conditional on/off loss above. Preserve the existing case store, normalization and retrieval weights.
3. Check forced-open parity with the original forest and forced-closed parity with retrieval, valid classification probabilities, original-example separation, and completed-unit reuse. On a tiny synthetic neighborhood, compare the gate gradient with exact enumeration of all binary decisions.
4. Run the matched local comparison with recorded units. Report final error and the full benefit/harm diagnostics; use fresh local confirmation before treating the best exploratory result as general evidence.

The existing forests were fitted using their original training sets. Those saved models cannot retrospectively provide independent gate-training predictions for examples they already used. A reserved gate-training split requires a new base-model configuration and new recorded units. Preserve and reuse the completed historical units for their original configurations; do not overwrite or retrain them.

### Code map

| Location | Purpose |
| --- | --- |
| [Pair prediction and aggregation](../model/t1/pairwise_adaptation.py) | Pair features, antisymmetric regression / symmetric classification proposals, and weighted final predictions |
| [Pair-model training](../model/t1/pairwise_training.py) | Neural, ExtraTrees and Random Forest fitting |
| [Alternative adapters](../model/t1/alternative_adaptation.py) | Retrieved-only pair construction, transport, error memory and vote attention |
| [RF runner](../tools/t1_pair_random_forest.py) | Recorded `ar3` comparisons |
| [Alternating implementation](../model/t1/forest_alternating.py), [runner](../tools/t1_forest_alternating.py) | Retrieval updates with detached forest proposals and subsequent forest refits (`ar4`) |
| [Earlier NN-CDH gates](../model/t1/adaptation_gate.py), [trainable threshold](../model/t1/disagreement_threshold.py) | Prior aggregate-gating experiments; these are not RF pair gates |
| [Query-aware NN-CDH](../model/t1/query_aware.py) | Earlier adapter input variants |
| [Forest audit](../tools/t1_forest_direction_audit.py), [old-gate audit](../tools/t1_gate_direction_audit.py) | Recount saved predictions, correction directions, denominators and harm severity |
| [Merged results reader](../tools/t1_alternative_redesign_summary.py) | Read recorded studies and regenerate alternative-adapter summaries |
| [Local launcher](../../run_pairwise_redesign_experiment.py) | Existing CPU environment and job conventions; a new gate driver is still needed |

### Workspace and recording conventions

The current workspace root is `C:/Users/yexia/Documents/GitHub/NN-kNN-haotang`. Adaptation code is in its isolated `adaptation improvement/code/neural-cbr` worktree, on branch `adaptation-improvement`. Keep T1 implementation there; the outer checkout owns the shared NN-kNN core. Follow each checkout's `AGENTS.md` and do not modify the published Table 1 benchmark path or its numbers.

| Material | Current location, relative to workspace root |
| --- | --- |
| Recorded models, predictions, per-run metrics and TensorBoard data | `runs/t1/<study>/` |
| Generated summaries and audit evidence | `adaptation improvement/code/neural-cbr/results/adaptation_improvement/<study>/` |
| Hand-written reports and this handoff | `adaptation improvement/code/neural-cbr/reports/` |
| Local launcher logs | `adaptation improvement/logs/` |
| Project index | [adaptation improvement/PROJECT.md](../../../PROJECT.md) |

Use uv and the existing environment; no pip or new hand-made environment. The local launcher selects the root `.venv`, local uv/cache under `adaptation improvement/.tools`, `NNKNN_RUNS=<root>/runs`, and the sibling `NN-kNN/datasets` data directory. It uses CPU, deterministic settings and one thread per worker. The user explicitly limited this project to locally available runs because r760 is inaccessible; do not introduce a remote-host dependency. Preserve the repository's Linux numeric references.

Training/evaluation drivers require a study and recorded run units. Detect completed units before fitting and load them; preserve resume state. Configuration/data identity determines reuse, while source commit and path are provenance. Generated summaries read recorded units and go to `results/`; do not track data or model weights in Git. See [T1 recording rules](../docs/T1_RUN_RECORDS.md).

The completed local suite uses seeds 40/41/42: classification on Iris, Wine, Breast Cancer, Balance, Digits, Zebra and Zebra Special; regression on Airfoil, Car, Yacht, Energy Efficiency, Student Performance, Body Fat and Diabetes (the existing task identifier is `regression__diabets`). This is the comparison scope, not a claim that these tasks establish general superiority.

### Verification already completed

The `ar4` report records **57 passing focused tests**, including 15 new cases. Its reader audits 840 retrieval states, 840 forest refits and 168 selected tests, and verifies reuse of all **1,848 new units** with fitting/inference disabled. All 798 preceding comparison rows were preserved. The independent RF direction audit reads 168 selected test units; the earlier-gate audit reads 126 regression method/split prediction sets. These checks cover existing implementations, not the proposed RF gate.

Do not describe the entire test suite as passing: a broader local Windows sweep had failures, followed by focused repairs/rechecks, and a Linux numeric-reference discrepancy was reproduced on the unchanged baseline. The Linux reference was preserved and the full sweep was not repeated. Details and logs are in [implementation, evidence and verification](NNCDH_ALTERNATIVES_REPORT_2026-11-27.md#implementation-evidence-and-verification).

## Evidence and reading

The numerical findings above are completed experiments. The RF gate designs and training procedure are proposed work.

- [Merged alternatives to NN-CDH](NNCDH_ALTERNATIVES_REPORT_2026-11-27.md): forest scores, alternating comparisons and correction diagnostics.
- [Earlier apply-or-skip gates](ADAPTATION_GATE_LOCAL_2026-11-25.md) and [learned thresholds](ADAPTATION_THRESHOLD_LOCAL_2026-11-25.md): existing gating evidence and limitations.
- [Current pair-forest implementation](../model/t1/pairwise_adaptation.py): pair prediction and case aggregation.
- [All earlier adapter experiments](ADAPTER_RESULTS_SUMMARY_2026-11-26.md): broader project history.
- [DESlib, JMLR 2020](https://www.jmlr.org/papers/v21/18-144.html): local competence and dynamic ensemble selection.
- [Local expectation gradients, NeurIPS 2015](https://papers.nips.cc/paper_files/paper/2015/hash/1373b284bc381890049e92d324f56de0-Abstract.html): conditional gradient estimation for discrete decisions.

Generated evidence links point to local, git-ignored summaries; a collaborator may not have those files. This note includes the principal results and definitions so its rationale remains readable when shared on its own.
