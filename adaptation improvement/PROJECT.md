# Adaptation improvement

Created November 24, 2026.

Study whether changing adaptation can preserve accurate NN-kNN retrieval predictions while allowing useful corrections. **Latest completed experiment (November 25, 2026): query-aware NN-CDH without the retrieved prediction input, study `aq2`.** The owner requested removing that input to prevent direct subtraction of retrieval. See the [new report](code/neural-cbr/reports/QUERY_NO_PREDICTION_LOCAL_2026-11-25.md) and [protocol](code/neural-cbr/docs/QUERY_NO_PREDICTION_EXPERIMENT.md).

The new residual network receives only `[query - neighborhood_mean, query]`; retrieved prediction is added outside it. The direct subtraction path is absent and verified. All 42 new models are complete, with 336 comparison rows reusing the previous baselines. Classification mean error is 10.146% (query-aware with prediction: 9.359%; original: 9.945%), although geometric mean ratios and five of seven tasks improve against the original. Regression RMSE worsens to 0.358695, with an extreme Car overcorrection. The original default remains unchanged. This variant is available for controlled follow-up, not promoted as an overall improvement.

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

`code/neural-cbr/` is a separate Git worktree based on `origin/neural-cbr` at `411abd4`, on branch `adaptation-improvement`. Its contents are tracked by that branch, not by the parent main checkout. Shared core and Table 1 files remain unchanged. The local uv-managed environment is the parent checkout's `.venv`, shared with this worktree; `NNKNN_RUNS` points to the parent checkout's ignored `runs/` directory.
