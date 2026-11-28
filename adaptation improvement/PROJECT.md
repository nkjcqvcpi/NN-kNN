# Adaptation improvement

Created November 24, 2026.

**Start here:** [Consolidated adapter results](code/neural-cbr/reports/ADAPTER_RESULTS_SUMMARY_2026-11-26.md) summarizes every completed design, gate, training-scheme and dropout experiment, the correction diagnostics, and conventional-model comparisons, with links to individual reports and results.

**Current design discussion:** [Gating Random Forest adaptation](code/neural-cbr/reports/RF_GATING_DISCUSSION_2026-11-27.md) is a collaborator-ready note covering the harmful-correction evidence, per-tree/per-pair/per-query gate choices, and the proposed training procedure. The preferred per-forest, per-pair gate has not been implemented or tested.

Study whether changing adaptation can preserve accurate NN-kNN retrieval predictions while allowing useful corrections. **Latest completed experiment (November 27, 2026): alternating retrieval and forest fitting, study `ar4`.** All four variants (ExtraTrees/Random Forest × mixed/retrieved-only pairs) are complete on 14 datasets and three seeds: 840 retrieval blocks, 840 forest refits and 168 selected tests, **1,848 new units**. The [merged report](code/neural-cbr/reports/NNCDH_ALTERNATIVES_REPORT_2026-11-27.md#alternating-forests-matched-continuations-ar4) now combines `ar1`–`ar4`, with **3,360 recorded units and 966 comparison rows**. See the [alternating protocol](code/neural-cbr/docs/ADAPTER_FOREST_ALTERNATING_EXPERIMENT.md).

**Alternating RF gives limited gains.** Mixed-pair RF classification improves **3.636% → 3.191%**, concentrated on Iris and Zebra Special, with Balance worse. Retrieved-only RF regression improves **0.314232 → 0.313608**, just **0.20%**, with five dataset wins, one tie and one loss. The other three regression variants slightly worsen. Validation retains the sequential model in 52/168 cases. Wrong-direction/overshoot frequency is not substantially improved; keep sequential forests as the simpler baseline and alternating RF as an optional confirmation candidate. No default changed.

The preceding `ar3` comparison keeps retrieval frozen. Retrieved-only RF gives **3.325% / 0.314232**, versus ExtraTrees' **4.350% / 0.314960**; mixed RF gives **3.636% / 0.321912** versus **4.149% / 0.320848**. All 252 `ar3` units and their results remain intact. [Random Forest protocol](code/neural-cbr/docs/ADAPTER_RANDOM_FOREST_EXPERIMENT.md).

**Training schemes:** `ar1`/`ar2`/`ar3` are sequential frozen-retrieval baselines; `ar4` updates retrieval through the final weighted forest prediction, then refreshes pairs and refits. All 840 new retrieval blocks change the core state. Hard tree splits remain detached, so this is alternating training rather than end-to-end differentiation through trees. “Mixed pairs” describes pair sampling. [Explanation](code/neural-cbr/reports/NNCDH_ALTERNATIVES_REPORT_2026-11-27.md#could-a-forest-influence-retrieval-in-joint-or-alternating-training), [historical frozen-core audit](code/neural-cbr/results/adaptation_improvement/ar3/training_schemes.csv), [alternating diagnostics](code/neural-cbr/results/adaptation_improvement/ar4/alternating_diagnostics.csv).

The preceding `ar2` study completed 798 units for three creative designs and two controls. Retrieved-only ExtraTrees and RF both improve all seven regression dataset means against retrieval and original sequential NN-CDH with 20% dropout. Direct ExtraTrees gives 7.365% / 0.339633; cross-fitted transport 7.717% / 0.328438; error memory and bounded neural vote attention are weaker. The expanded validation choice including RF gives 3.586% / 0.317036. Earlier results are preserved in the merged report and [follow-up protocol](code/neural-cbr/docs/NNCDH_ALTERNATIVES_FOLLOWUP_EXPERIMENT.md).

The preceding `ar1` study completed 42 frozen contexts, 294 validation candidates and 126 selected tests. Its [original report](code/neural-cbr/reports/ADAPTER_PAIRWISE_REDESIGN_LOCAL_2026-11-26.md) and [protocol](code/neural-cbr/docs/ADAPTER_PAIRWISE_REDESIGN_EXPERIMENT.md) preserve the first three alternatives: mixed-pair trees, pair neural and local ridge.

The preceding **dropout study `ad1`** completed all 1,134 comparison cells: 756 new local models at 10%/20% dropout and 378 reused zero-dropout references. See the [dropout report](code/neural-cbr/reports/ADAPTER_DROPOUT_LOCAL_2026-11-26.md) and [protocol](code/neural-cbr/docs/ADAPTER_DROPOUT_EXPERIMENT.md).

The lowest new fixed classification mean is query **without** prediction + alternating + 20% dropout: **6.701% error**, versus 9.356% without dropout. The gain is concentrated on Zebra; the other six datasets worsen from 7.380% to 7.818%. Query **with** prediction + alternating + 10% has the lowest new regression mean, **0.337381 RMSE**, versus 0.345628 without dropout; the earlier probability gate remains marginally lower at 0.337206. Original + sequential + 20% improves six of seven regression datasets, from 0.350429 to 0.338523 with exactly fixed retrieval. Dropout helps selectively and reduces severe overshoot in these comparisons, but does not eliminate extreme corrections or justify a universal default.

The preceding [training-scheme study `as1`](code/neural-cbr/reports/ADAPTER_TRAINING_SCHEMES_LOCAL_2026-11-26.md) completed 378 cells: 252 new local models and 126 reused references. Its [protocol](code/neural-cbr/docs/ADAPTER_TRAINING_SCHEMES_EXPERIMENT.md) defines the matched schemes used by the dropout study. For query-aware with the prediction input, sequential / alternating / joint classification errors were 10.980% / 9.502% / 9.359%, and regression RMSEs were 0.371033 / 0.345628 / 0.345530. Validation-selected schemes reached 8.745% / 0.345186. All sequential models share the same pretrained retrieval state per dataset/seed, verified exactly. The original default remains unchanged.

The preceding [query-aware-without-prediction experiment, `aq2`](code/neural-cbr/reports/QUERY_NO_PREDICTION_LOCAL_2026-11-25.md), implements the owner's request to remove that input and prevent direct subtraction of retrieval. See its [protocol](code/neural-cbr/docs/QUERY_NO_PREDICTION_EXPERIMENT.md).

The `aq2` residual network receives only `[query - neighborhood_mean, query]`; retrieved prediction is added outside it. The direct subtraction path is absent and verified. All 42 models are complete, with 336 comparison rows reusing the previous baselines. Classification mean error is 10.146% (query-aware with prediction: 9.359%; original: 9.945%), although geometric mean ratios and five of seven tasks improve against the original. Regression RMSE worsens to 0.358695, with an extreme Car overcorrection. This variant remains experimental. Those earlier results used joint training except for frozen sequential Diabetes; the new study distinguishes the schemes explicitly.

The preceding [query-aware study `aq1`](code/neural-cbr/reports/QUERY_AWARE_LOCAL_2026-11-25.md) improves mean classification error from 9.945% to 9.359%, but worsens mean regression RMSE from 0.338910 to 0.345176, mainly through harmful Car adaptation. Its 42 query-aware runs and compact local MLP, TabM, CatBoost and kNN comparisons are complete, with 672 recorded units and 294 comparison rows. NN-kNN beats the local neural baselines on family means, but their smaller search/budget does not establish superiority over the historical tuned baselines. Residual output alone cannot guarantee retrieval dependence; the recorded intervention probes show substantial retrieval sensitivity.

Earlier exploration covered a learned apply-or-skip head and a trainable disagreement cutoff. The magnitude/direction design remains shelved. Their protocols, code and results below are preserved as previous experiments.

The shelved magnitude/direction proposal used:

```text
correction = magnitude * unit_direction
final_output = retrieved_output + correction
magnitude >= 0
```

The magnitude head uses neighborhood evidence and the shared hidden representation; the direction head builds on current NN-CDH adaptation. Raw residual supervision trains scalar direction through its otherwise nondifferentiable sign. See the [implemented design](code/neural-cbr/docs/ADAPTATION_MAGNITUDE_DIRECTION.md) and [original discussion](discussion/2026-11-24.md#latest-proposal-separate-correction-magnitude-and-direction).

The earlier gated-residual design remains a comparison candidate:

```text
y_final = y_retrieved + alpha(x) * delta_y
        = (1 - alpha(x)) * y_retrieved + alpha(x) * y_adapted
0 <= alpha(x) <= 1
```

Magnitude/direction status: shelved after study `am1` completed all 42 local pairs (14 datasets, three seeds). Its code and results remain reproducible in the isolated T1 worktree on branch `adaptation-improvement`; it is not the active candidate or default. The model wins on 7 tasks but has higher geometric mean error in both task families. See the [comparison report](code/neural-cbr/reports/ADAPTATION_IMPROVEMENT_LOCAL_2026-11-25.md). The owner requested local execution only because r760 is unavailable.

The [adaptation behavior report](code/neural-cbr/reports/ADAPTATION_BEHAVIOR_2026-11-25.md) analyzes all saved old/new test predictions for insufficient correction, harmful overshoot and direction errors. It includes per-dataset figures, frozen-output scale diagnostics and a separation of retrieval-core changes from the final correction's effect. No retraining was performed for this analysis.

Follow-up study `ag1` is complete: 42 dataset/seed comparisons of six alternatives, including two jointly trained apply-or-skip heads. The simple validation-selected disagreement threshold improves 5/7 tasks in each family; the learned heads have mixed results. See the [gate experiment report](code/neural-cbr/reports/ADAPTATION_GATE_LOCAL_2026-11-25.md). The old adapter remains the default pending fresh-split confirmation.

Owner-requested study `ag2` is also complete: a single learned disagreement cutoff, tested with the adapter frozen and trained alongside the adapter, across the same 42 dataset/seed pairs. The frozen version is almost unchanged from the original full adapter; the joint version improves classification but worsens regression. Neither establishes an overall improvement over the validation threshold. See the [trainable threshold report](code/neural-cbr/reports/ADAPTATION_THRESHOLD_LOCAL_2026-11-25.md) and [protocol](code/neural-cbr/docs/ADAPTATION_THRESHOLD_EXPERIMENT.md). The validation rule remains the candidate for confirmation; held-out routing supervision is the proposed next investigation.

The gate belongs to the adaptation training stage even though its inputs are retrieval statistics. Whenever the adapter is updated or frozen, the gate is updated or frozen with it. Retrieval follows the selected training schedule independently. See the [training decision](discussion/2026-11-24.md#training-decision-the-gate-trains-with-the-adapter).

## Project contents

| Location | Contents |
| --- | --- |
| `code/` | Isolated T1 worktree, implementation, tests and local launcher |
| `results/` | Generated experiment summaries, tables and figures; tracked index of evidence locations |
| `logs/` | Project activity log and ignored runtime logs |
| `discussion/` | Dated discussions, proposals and decisions |
| `docs/` | References, design documents, experiment plans and written reports |

Start with the [initial discussion](discussion/2026-11-24.md), [references](docs/REFERENCES.md), [project log](logs/PROJECT_LOG.md), and [results index](results/INDEX.md).

## Repository integration

This folder is the mini-project's home for the material requested by the owner. Generated summaries and runtime logs are ignored by git; discussion, documentation and written reports are tracked.

Recorded training/evaluation units, predictions and weights still use the repository's mandatory shared `runs/` storage and `--study` / `run=` interfaces. Record their study IDs and locations in `results/INDEX.md`; do not duplicate weights or datasets here. Reuse complete units rather than retraining them.

`code/neural-cbr/` is a separate Git worktree based on `origin/neural-cbr` at `411abd4`, on branch `adaptation-improvement`. Its contents are tracked by that branch, not by the parent main checkout. The completed adapter implementations, tests and reports through `ar3`, including the training-scheme audit, are committed as `0eab0e4`. Shared core and Table 1 files remain unchanged. The local uv-managed environment is the parent checkout's `.venv`, shared with this worktree; `NNKNN_RUNS` points to the parent checkout's ignored `runs/` directory.
