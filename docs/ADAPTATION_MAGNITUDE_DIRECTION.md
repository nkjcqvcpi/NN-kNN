# Magnitude and direction adaptation

Design agreed with the owner on 2026-11-24; implementation and local comparison on 2026-11-25.

Status: **shelved by the owner on November 25, 2026**, after reviewing study `am1`. This document describes the completed experiment. The historical residual adapter remains the baseline; [apply-or-skip gating](ADAPTATION_GATE_EXPERIMENT.md) is the next exploration. Existing code and recorded results are preserved for reproducibility.

## Prediction

The existing NN-CDH aggregate inputs feed a shared hidden network. Regression uses `[query_representation - weighted_case_representation, retrieval_prediction]`. Classification uses the same representation difference, the full retrieval class mass, and declared uncovered nominal differences where applicable.

The direction head predicts an unconstrained raw residual proposal `v`. The applied direction is `u = v / ||v||_2`. The scale head receives the shared hidden representation plus four neighborhood statistics and outputs `m = softplus(linear([hidden, statistics]))`. The final prediction is `retrieval + m*u`. Thus `m` is the L2 magnitude of the correction in training-standardized target units (regression) or class-score units (classification). It is not confidence, a probability, or an interpolation fraction, and is not restricted to [0, 1]. Multiplying the raw proposal by a positive constant cannot increase the applied correction.

For exact-zero proposals the implementation uses the fixed unit vector +e1. This is an explicit tie convention, not evidence for increasing the target or favoring class zero. Nonzero proposals are normalized without epsilon shrinkage, with intermediate rescaling to avoid numerical underflow/overflow. Classification retains the historical score space without silently subtracting the common score offset. The alternative logit-residual mode is not supported by this first design.

The original NN-CDH hidden widths/activations are retained. The magnitude head is one linear layer followed by softplus, with zero weights and inverse-softplus bias giving initial magnitude 0.01. This small initialization and the added head are declared architectural choices, not tuned results. Raw statistics are used; no new data preprocessing is fitted.

## Neighborhood evidence

All weights below are normalized, query-dependent retrieval activations, not innate case biases. Distances come from the model's actual learned metric. Case labels align with the returned case indices.

1. `d_min`: nearest eligible case distance, excluding self/quarantined cases. In masked top-k retrieval this includes eligible cases with zero activation, not only the retained neighbors.
2. `d_mean = sum_i w_i*d_i`: weighted mean distance.
3. `C_w = sum_i w_i^2`: activation concentration (reciprocal effective case count).
4. `D_y = sum_i w_i*||t_i - sum_j w_j*t_j||^2`: target disagreement. Regression uses standardized numeric labels; one-hot classification gives `1-sum_c p_r[c]^2`.

Training uses stable-ID leave-one-out retrieval. Query targets never enter these statistics. Validation/test examples do not become cases. Frozen reuse caches statistics; alternating/joint training recomputes them from the current retriever. Gradients through retrieval statistics remain available when the selected schedule trains retrieval.

## Training

Two losses have different jobs:

- Final task loss acts on `retrieval + m*u`: MSE for regression, cross-entropy on scaled class scores for classification.
- Raw proposal loss is MSE between `v` and the target residual (`y-y_r`, or `one_hot(y)-p_r`). It is not a loss on the applied correction.

Scalar unit direction is sign(v), which has zero gradient away from zero. The raw proposal loss supplies the direction head's learning signal. Both losses must have positive coefficients. No direct target `abs(y-y_r)` is supplied to the magnitude head: the mean absolute residual would not express cancellation of opposing corrections.

Both heads and the shared trunk belong to the adaptation optimizer. They train/freeze together under frozen-retrieval reuse, independent/sequential, alternating and joint schedules, including adapter warmup. Existing joint summed/separated objective routing is retained. Under summed objectives the raw proposal loss can also reach retrieval, just as the historical residual loss did.

The prior conditional-mean argument applies to squared-error regression, not arbitrary classification scores: opposing unpredictable corrections can average toward zero, but uncertain same-direction corrections can still have large magnitude. The old adapter already had a conditional-mean tendency; improvement is an empirical question.

## Comparison protocol

`tools/t1_adaptation_improvement.py` uses the frozen task-specific F configs from `configs/t1/confirm_final_20261013`, seeds 40, 41, 42, and their original splits/preprocessing/budgets. The suite has 29 tasks after excluding postponed QSAR. Diabetes uses the historical frozen-retrieval reuse condition; other tasks use their selected F synchronization condition.

The owner has no r760 access and requested local runs only. Therefore these are new local paired runs, not replacements for or exact reproductions of the historical report. Within each dataset/seed one recorded pretrained core, its optimizer state, data fingerprint and starting RNG state are shared by the existing and new adapters. Completed units are loaded, never retrained. Validation alone selects checkpoints. A test-result-driven hyperparameter search is not performed.

Existing loss weights, including any task-specific `v_small`, are preserved in the primary comparison to avoid confounding architecture with removal of a penalty. This does not declare that penalty necessary; removing it is a separate future ablation. New-design raw residual supervision replaces the old applied-residual term, as agreed.

The comparison evaluates the full proposed design, not two-head topology alone: the classification raw proposal is unconstrained rather than tanh-bounded, and initialization changes. Frozen reuse retains its configured combined validation loss for checkpoint selection. In diabetes, the old regression residual and final MSE were identical; the new raw-proposal term is distinct, so the same combined-loss rule can select a different checkpoint than final MSE alone. Synchronized runs retain their configured final-task validation criterion.

Run records and weights live under `NNKNN_RUNS`; generated summaries under this worktree's `results/adaptation_improvement/`; written reports under `reports/`. Local launch instructions and evidence links also live in the owner's `adaptation improvement` project folder. Summaries report paired seed errors, task means, helpful/harmful outcomes and separate classification/regression aggregates. Mean wins/ties/losses are descriptive, not significance tests. Three previously seen seeds cannot establish broad superiority.

## Scope and entry points

The new module is T1-only; no shared core or Table 1 code changes. Use T1 reuse/synchronization and `model.t1.outcomes.final_prediction`, which supplies the actual retrieval statistics. Historical aggregate-only calls cannot supply those inputs and fail explicitly. Full-cycle post-removal continued retraining is outside this initial comparison and must be extended before it uses the new adapter.

Focused tests cover statistics/exclusions, unit norm, actual magnitude, scalar direction gradients, nominal binding, frozen retrieval, all three training schedules, legacy parity, recorded comparisons and complete-unit reuse.
