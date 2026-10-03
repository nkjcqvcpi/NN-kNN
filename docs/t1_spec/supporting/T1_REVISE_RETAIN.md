# T1 revise and retain design

**Current scoring direction, 2026-09-20:** Compare Q (or a C/H variant), B alone, the coverage-to-reachability ratio, and influence measured by case removal. Do not combine Q and B. Use final adapted query success in the thresholded solve relation. Removal scoring uses final prediction loss only and reports the adaptation penalty separately. See [the candidate definitions and protocols](T1_CASE_MAINTENANCE_CANDIDATES.md), including cached-query scoring, optional retraining, and the deferred activation-map idea.

## Status and claim boundary

This is a working technical design for the merged T1's revise and retain stages, developed from the PI's ideas on 2026-09-08. The provenance and case-bias mechanisms are plausible candidate signals, not yet validated detectors of bad, outlying, or poisoned cases. Numerical thresholds, datasets, retraining protocols, and success criteria remain provisional.

The PI subsequently set the immediate implementation priority to retain/case selection and disabled revise for the first implementation. The human-correction design below remains the later T1 plan, not a current implementation requirement. See [T1_IMPLEMENTATION_HANDOFF.md](T1_IMPLEMENTATION_HANDOFF.md).

The design separates **routine capacity maintenance**, **case flagging**, and **case correction**:

- Routine maintenance may automatically evict a clearly redundant or low-utility case from active memory under the fixed budget, but only after protection and coverage checks and with a logged, reversible action.
- Safety flagging ranks consequential cases for human review; it does not automatically establish that a case is bad.
- Human correction changes, reweights, quarantines, or removes a reviewed case.
- Verification measures whether the intervention improves the intended behavior and whether it causes collateral failures.

## Candidate signal 1: case provenance and contribution history

Maintain an auditable ledger for each case containing its source, label, insertion time, edits, reviewer actions, retrieval history, activation or weight, prediction context, and observed outcomes.

**PI refinement, 2026-09-20:** When adaptation is enabled, use the final adapted outcome to update every case used for that query, weighted by its normalized activation. A successful outcome adds the activation to `C_i`; an unsuccessful outcome adds it to `H_i`. One retrieved case receives the whole update. Several cases share the update according to their activations. Accumulate this evidence over queries so maintenance considers retrieval and reuse together. The task-specific success criterion, including the regression error tolerance, remains to be defined. Test whether the accumulated evidence reliably guides maintenance; repeated retrieval of the same case groups can preserve mistaken assignments. Keep pre/post diagnostics and the penalty on unnecessarily large corrections. The stored-label equations below remain a retrieval-only baseline. The comprehensive case maintenance score remains open.

For the retrieval-only baseline, a case `i` with normalized activation `a_i(x)` and stored label `c_i` has candidate summary statistics including:

- **Retrieval exposure:** number of queries for which the case is selected or receives activation above a defined threshold.
- **Activation mass:** cumulative `a_i(x)`, so frequent negligible retrieval is distinguished from strong influence.
- **Helpful support:** activation mass on examples for which `c_i` matches the reference label.
- **Opposing support:** activation mass on examples for which `c_i` differs from the reference label.
- **Error-associated support:** influence on examples the model predicts incorrectly, especially when the case supports the predicted wrong class.
- **Counterfactual harm:** change in loss or correctness when the case is masked and remaining case activations are renormalized at inference time.
- **History:** who or what supplied the case, label confidence, domain, time period, prior edits, and review status.

A raw count of appearances in wrong predictions is insufficient. It must be normalized by exposure and weighted by influence, because a frequently retrieved case has more opportunities to appear in both correct and incorrect predictions. Being present during an error also does not prove that the case caused the error.

### Earlier classification components: retrieval-only baseline

On a labeled validation or audit set `D`, define:

- `R_i = sum_x 1[i is retrieved for x]`: retrieval frequency;
- `A_i = sum_x a_i(x)`: cumulative activation or contribution strength;
- `C_i = sum_x a_i(x) * 1[c_i = y_x]`: activation-weighted support for the correct class; and
- `H_i = sum_x a_i(x) * 1[c_i != y_x]`: activation-weighted support for an incorrect class.

This implements the PI's refinement that both correct and incorrect support are measured by contribution strength rather than by event counts alone. When all audited queries have reference labels and every activation is assigned to one stored class, `A_i = C_i + H_i`; retaining all quantities is still useful because `R_i` measures exposure frequency while `A_i` measures total influence.

### C/H from the adapted query outcome

The PI accepts assigning the outcome of a combined prediction to all cases used for that query. Let `s(q)` be 1 if the final adapted prediction meets the task's success criterion and 0 otherwise. For classification, this can be whether the final predicted class is correct. Regression requires an acceptable prediction-error threshold; its value remains open. Let `a_i(q)` be case `i`'s normalized activation and `D` the queries with evaluated outcomes. Then

```text
C_i = sum over q in D of a_i(q) * s(q)
H_i = sum over q in D of a_i(q) * (1 - s(q))
```

Every participating case receives the same success or failure judgment for that query, but the amount added depends on its activation. With one case, its activation is 1. With activations 0.7 and 0.3, a successful adapted prediction adds 0.7 and 0.3 to the two cases' C values; an unsuccessful prediction adds those amounts to their H values. The activations still sum to one, so `A_i = C_i + H_i` and `sum_i(C_i + H_i) = |D|` for fully recorded, valid queries.

This rule measures how cases perform as part of retrieval and reuse together. It does not require a separate prediction with each case removed. The PI expects many queries to make these estimates more reliable. T1 will test this expectation as evidence accumulates and retrieved case groups vary. A harmful case repeatedly retrieved with helpful cases can continue to receive favorable updates, so additional queries do not by themselves guarantee that such assignments disappear. Removal experiments remain useful comparisons when feasible.

The equations above define the binary success/failure version. A continuous regression outcome measure remains a design choice. This contribution rule does not restore the old provenance-quality formula or select the comprehensive case maintenance score.

The PI approved a simpler primary design on 2026-09-08. Treat usage and contribution quality as distinct rather than combining retrieval count and activation into another weighted score. Because cumulative activation already reflects both retrieval and contribution strength, including both `R_i` and `A_i` in a weighted sum could double-count usage.

Define the smoothed **provenance-quality score**

`Q_i = (C_i + s) / (C_i + H_i + 2s)`, with `s > 0`.

This score has a direct interpretation:

- `Q_i` near 1: the case's observed activation predominantly supports correct classes;
- `Q_i` near 0: the case's observed activation predominantly supports incorrect classes; and
- `Q_i` near 0.5: the evidence is balanced or insufficient.

The smoothing constant `s` prevents a case with one favorable retrieval from appearing perfectly reliable. It must be calibrated or included in sensitivity analysis rather than fixed by intuition.

Keep `R_i` and `A_i` visible as exposure/evidence measures instead of folding them into `Q_i`. Require a minimum retrieval count and/or activation mass before interpreting a low `Q_i` as evidence of harm. Use low `R_i` and low `A_i` to flag low observed utility, which is different from demonstrated harmfulness. The mean activation conditional on retrieval, `A_i / R_i`, may also be displayed to distinguish frequent weak retrieval from rare strong retrieval.

This activation-weighted ratio is an NN-kNN-specific working design informed by case-base competence, data valuation, and training-data influence literature; the exact equation is not an established literature standard and must be evaluated as a proposed contribution.

## Candidate signal 2: learned case bias

In the maintained classification path inspected on 2026-09-08, the default case score is

`z_i(x) = b_i - d_i(x)`,

where `b_i` is a trainable per-case bias and `d_i(x)` is the learned distance. Case scores are normalized over cases, class probability is the sum of normalized activation assigned to cases of that class, and training minimizes negative log likelihood of the correct class mass.

Under softmax normalization with temperature `tau`, ignoring the small numerical epsilon and other regularizers, let `p_y` be the total case activation assigned to the correct class. Then:

- for a case whose label equals the query's correct class, the classification-loss gradient on its bias is negative, so gradient descent tends to increase its bias;
- for a case whose label differs from the correct class, the gradient is positive, so gradient descent tends to decrease its bias in proportion to its activation.

More explicitly:

`dL/db_i = -a_i(1-p_y)/(tau*p_y)` when `c_i = y`, and

`dL/db_i = a_i/tau` when `c_i != y`.

This supports the PI's intuition: a repeatedly active case that competes against correct labels can acquire a lower bias. A low or declining bias is therefore a useful learned **disutility signal**.

It is not, by itself, proof that the case is corrupted:

- the bias is learned through relative competition, not supervised as a bad-case probability;
- rare, boundary, minority, or domain-specific cases may look harmful on a majority-dominated audit distribution while preserving important coverage;
- feature-extractor, distance, glocal-weight, and other parameter updates can absorb or redistribute responsibility;
- sampled retrieval, top-k masking, sparse normalization, or non-selection can leave some cases with little or no useful bias gradient;
- regularization, initialization, temperature, class imbalance, training duration, and score mode change the bias scale;
- an unused poisoned case may retain an ordinary bias, while a useful but frequently competing case may receive a low one.

For these reasons, thresholds should be calibrated on a held-out validation/audit set, conditioned on minimum exposure and preferably compared within class or case cohort. The test set must remain untouched until final evaluation.

### Working bias score

Map the learned case bias to a comparable scale for evaluation as a standalone score. A robust initial option is a within-class percentile or a median/MAD-based transformation of the final bias, optionally augmented by the bias change from initialization. Denote the resulting `[0,1]` value by `B_i`, with lower values indicating learned disutility.

Within-class or otherwise matched normalization is important: raw biases are affected by class frequency, initialization, temperature, regularization, score mode, and training duration.

## Historical combined score — superseded

The 2026-09-08 candidate combined Q and B through a weighted geometric mean. The PI set it aside on 2026-09-17 and explicitly chose to compare Q and B separately on 2026-09-20. The following equation and rationale are historical, not implementation instructions:

`T_i = Q_i^alpha * B_i^(1 - alpha)`, with `0 < alpha < 1`.

The historical geometric mean was less compensatory than an arithmetic average: one very low component pulled the final score down more sharply. The PI now questions multiplying these overlapping signals. Their agreement and disagreement can still be reported separately.

Interpretation:

- high `Q_i`, high `B_i`: observed contribution and learned training signal both support the case;
- low `Q_i`, low `B_i`: strong harmful-case review candidate;
- low `Q_i`, high `B_i`: provenance-bias disagreement; and
- high `Q_i`, low `B_i`: provenance-bias disagreement that may reveal feature-learning, regularization, cohort, or optimization effects.

Retain visible component scores and two separate review flags because trustworthiness and observed utility are not identical:

- **Harm flag:** sufficient exposure plus unusually high activation-weighted incorrect support or counterfactual harm.
- **Low-utility flag:** very low retrieval/activation plus a redundancy or coverage check showing that removing the case is unlikely to eliminate unique or rare-domain competence.

The score and flags identify candidates but do not, by themselves, authorize action. For ordinary fixed-budget maintenance, a clearly redundant or low-utility case may be automatically evicted from the active case base only after coverage and protected-case checks; the reason and component evidence must be logged, and the action must be reversible. Suspected harmful, poisoned, mislabeled, rare, boundary-defining, subgroup- or domain-critical, ambiguous, or otherwise consequential cases require human review before modification, reweighting, quarantine, or removal. No case should be labeled "bad" solely because it is an outlier, seldom used, or below one threshold.

The current comparison uses Q or a C/H variant, B alone, the coverage-to-reachability ratio, and case-removal influence. Q/B products and arithmetic combinations are not part of this proposed comparison. See the candidate note for exact definitions and computational variants.

## Recommended tiered maintenance and flagging policy

Use the selected candidate measure to rank cases. Contribution evidence and learned bias may both be displayed, while Q and B remain alternative scores:

1. Require a minimum exposure before interpreting a case's error rate or bias trajectory.
2. Flag cases with unusually low or rapidly declining bias relative to appropriate peers.
3. Flag cases with high exposure-adjusted opposing or error-associated influence.
4. Confirm suspicious cases with inference-time counterfactual masking when computationally feasible.
5. For clearly redundant or low-utility cases, apply coverage and protected-case checks before any automatic active-memory eviction; log the decision and keep a reversible inactive copy or equivalent undo state.
6. Route suspected harmful, poisoned, mislabeled, rare, boundary-defining, subgroup- or domain-critical, ambiguous, or otherwise consequential cases to a qualified human reviewer.
7. Show the reviewer the case, provenance, neighboring cases, influence history, bias trajectory, affected predictions, and all protection flags.
8. Rank cases using the declared candidate measure while displaying `Q_i`, `B_i`, `R_i`, `A_i`, `C_i`, `H_i`, and any harm/low-utility flags separately. For removal-loss scoring, lower scores are removed first; for Q, B, and the coverage/reachability ratio, larger values favor retention.
9. Let the reviewer retain, modify, reweight, quarantine, or remove a consequential case; never act solely because one score crosses a threshold.

The current research comparison asks which candidate best preserves prediction quality at a given case budget and maintenance cost. Q and B measure usefulness in the existing model. Case removal directly tests whether the remaining case base can replace a case. Similarity between activation maps is deferred because duplicate cases can acquire different activations during training.

## Human-correction evaluation

The PI proposed comparing performance before and after correction, retraining as needed, and identifying which decisions flip and whether the flips are correct. Preserve that idea but separate the direct effect of the edit from the effect of subsequent optimization.

### Evaluation checkpoints

1. **M0 - Trained baseline before correction:** Save the already-trained model, case base, predictions, case activations, and metrics. This is the operational no-correction baseline; it is not another training run.
2. **M1 - Edit only:** Apply the human correction and reevaluate without gradient updates. This isolates the immediate causal effect of the case intervention.
3. **M2 - Edit plus adaptation:** Fine-tune or retrain under a fixed, reported budget so the remaining model can adapt to the corrected case base.
4. **Conditional matched-training control:** Omit this when no additional optimization occurs after M0. If M2 receives extra gradient updates and the experiment claims that improvement is attributable specifically to the correction, optionally continue the unchanged M0 model under the same optimization budget. If M2 instead retrains from scratch, use a matched uncorrected run with the same initialization policy, data, schedule, and seed controls. This is a scientific confound check, not part of the operational correction workflow.
5. **Oracle where available:** For synthetically injected corruption, compare against training or adaptation with the known clean case base.

Full retraining and bounded fine-tuning should be reported separately. M0 and the conditional control are not the same object: M0 is the saved trained baseline, whereas the conditional control receives whatever extra optimization budget is given to M2 without receiving the edit. The PI correctly notes that there is no operational reason to retrain an unchanged model. The conditional control is needed only when extra optimization could confound a causal research claim; otherwise M0, M1, and M2 are sufficient.

### Outcome measures

- Overall task performance before correction, immediately after correction, and after controlled adaptation.
- Performance on the targeted cases, queries influenced by them, unaffected queries, relevant subgroups, and shifted domains.
- Decision-flip matrix: wrong-to-correct, correct-to-wrong, wrong-to-different-wrong, and other changed outcomes.
- Net beneficial flips: wrong-to-correct minus correct-to-wrong, reported alongside both component counts rather than alone.
- Targeted correction rate and collateral regression rate.
- Change in loss, confidence/calibration, case activation, and retrieval neighborhood for affected queries.
- Retention of previously correct behavior and coverage of rare or boundary regions.
- For injected bad or poisoned cases: flagging precision, recall, ranking quality, and time or number of cases a reviewer must inspect.
- Variation across seeds, corruption levels, case-base samples, and retraining budgets.

## Initial ablation matrix

| Flagging method | Human action | Post-edit optimization | Purpose |
|---|---|---|---|
| Provenance/contribution only | Review and correct | None, then fixed adaptation | Test explicit usage history |
| Bias only | Review and correct | None, then fixed adaptation | Test learned disutility signal |
| Combined provenance plus bias | Review and correct | None, then fixed adaptation | Test complementary evidence |
| Random case at matched review budget | Review or matched removal | None, then fixed adaptation | Control for generic editing and extra training |
| Oracle corrupted-case identity | Correct known corruption | None, then fixed adaptation | Estimate attainable upper bound in controlled experiments |

## Relationship to full-cycle component synchronization

The per-case trustworthiness score `T_i` is evidence for ranking and review; it is not the complete maintenance loss. In the mature T1 system, retention selects an active case set `S` under a fixed budget and protection constraints. Its objective should combine pre- and post-adaptation task loss with active-case untrustworthiness, redundancy, coverage gaps, and selection churn. This keeps three distinct ideas visible:

- trustworthiness asks whether a case's observed contribution and learned bias support confidence in it;
- utility and redundancy ask whether the case adds competence beyond other retained cases; and
- coverage/protection asks whether removing it would erase rare, boundary, subgroup, action, or shifted-domain knowledge.

Maintenance is naturally a discrete or mixed optimization step. The first implementation should therefore update the case set at safe checkpoints, then rebuild retrieval neighborhoods and dependent statistics before resuming learning. A later differentiable selector can be evaluated as an ablation, but the proposal should not require every component to share one end-to-end gradient.

The retention objective is part of a broader synchronization design in which retrieval is rewarded for returning useful, nearby, readily adaptable cases and adaptation is rewarded for accurate but minimal corrections. When adaptation is enabled, `C_i/H_i` may use post-adaptation contribution evidence. The adaptation loss penalizes unnecessarily large corrections; pre-adaptation outputs remain diagnostic measurements rather than a mandatory source of retention credit.

## Open choices

- Define whether an outlier is automatically suspicious or only suspicious when it causes harmful contribution; valuable rare cases must not be discarded merely for being unusual.
- Select Q smoothing or its variant, bias normalization, activation thresholds, minimum evidence, and review thresholds using validation data rather than the final test set.
- Use **case maintenance score** for the broad measure. The more specific name for the removal candidate remains open; the PI proposed **case influence score**.
- Use a user-specified fixed maximum case count `K` as the primary bounded-maintenance formulation and compare it with the current downsampling approach and other selectors at matched `K` values. Treat adaptive or marginal-utility-based growth as a later extension.
- Choose the final prediction loss for each task's removal comparison. Exclude the adaptation penalty from that score and report it separately. Specify the allowed cumulative loss increase and cache approximation checks.
- Specify whether correction means label repair, content replacement, bias/weight adjustment, quarantine, deletion, or a task-dependent choice.
- Choose bounded fine-tuning versus full retraining schedules and computational budgets.
- Define the expert-review protocol for medical and other specialized cases.

## Evidence anchors

- Current NN-kNN implementation inspected at `D:/NN-kNN/model/nnknn_model.py`, especially the trainable bias construction, `bias_minus_distance` scoring, case normalization, class-mass aggregation, classification NLL, optimizer groups, and training loop.
- Barry Smyth and Elizabeth McKenna, "Competence Models and the Maintenance Problem," *Computational Intelligence* 17(2), 2001, https://doi.org/10.1111/0824-7935.00142.
- Pang Wei Koh and Percy Liang, "Understanding Black-box Predictions via Influence Functions," ICML 2017, https://proceedings.mlr.press/v70/koh17a.html.
- Amirata Ghorbani and James Zou, "Data Shapley: Equitable Valuation of Data for Machine Learning," ICML 2019, https://proceedings.mlr.press/v97/ghorbani19c.html.
- Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan, "Estimating Training Data Influence by Tracing Gradient Descent," NeurIPS 2020, https://proceedings.neurips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html.

These external methods motivate comparison with established training-data responsibility and valuation approaches, but they estimate different quantities from the proposed NN-kNN contribution and bias signals. The activation-weighted Q ratio remains a candidate. The geometric Q/B combination is superseded. Neither is an equation taken from those papers.
