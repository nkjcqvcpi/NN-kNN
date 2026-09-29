# T1 implementation plan: full-cycle, general-purpose NN-kNN

Status: PI-confirmed architecture with staged implementation details. Numerical hyperparameters and several task-specific definitions remain unresolved and are listed explicitly.

## 1. Objective and scientific boundary

T1 asks whether NN-kNN can become a coordinated full-cycle neural case-based reasoning system that preserves competitive predictive performance while making its decisions case-grounded, inspectable, correctable, and maintainable under bounded memory.

The long-range engineering goal is an easy-to-use neural-network-compatible module that can be inserted into selected places where a neural network is used. The research claim is narrower: demonstrate compatible supervised and sequential roles, improve the core mechanisms, and document boundaries. Classification and regression replacement feasibility has already been explored in pre-MCB papers; it is preliminary foundation, not the new T1 contribution.

The new T1 work is:

- quality-, utility-, redundancy-, and coverage-aware case maintenance;
- cross-benchmark integration and evaluation of Momentum Case Base (MCB);
- bounded neural reuse that cannot silently hide poor retrieval;
- synchronization among retrieval, reuse, and retention;
- human revision with causal post-intervention evaluation; and
- bounded-memory and modern-competitiveness studies across legacy and selected modern tasks.

T1 has two sequenced aims:

- **T1.1 Technical core:** retrieve, bounded reuse, MCB stabilization, retain, and bounded-memory integration. Revise is initially disabled.
- **T1.2 Human revision:** add authorized case/weight intervention, controlled retraining where appropriate, and a bounded UI study.

Packaging is an enabling deliverable, not a separate scientific aim.

## 2. Working bottleneck hypothesis

The active RL project suggests that NN-kNN is not yet accurate or efficient enough in some settings. The current working hypothesis is that the model is not consistently retaining and using its best cases. Previous downsampling addressed size but did not explicitly preserve value, coverage, or rare competence.

T1 tests a stronger idea: **meaningful remembering**. The case base should retain high-value, trustworthy, diverse, coverage-preserving cases and remove redundant, harmful, or persistently useless cases when evidence justifies doing so. This is conceptually complementary to catastrophic-forgetting research: the problem is not only avoiding loss of old information, but choosing what deserves to remain accessible.

Do not report poor case selection as a proven cause until ablations and interventions establish it. Representation drift, retrieval scoring, labels/solutions, reuse, maintenance timing, and RL optimization remain competing explanations.

## 3. Four-stage neural CBR architecture

### 3.1 Retrieve — NN-kNN activation

- Compute the learned distance between a query and eligible cases using the NN-kNN representation, feature weights, and applicable glocal/case-specific parameters.
- Use per-case bias and distance to form the existing activation logic, whose default score form is:

```text
z_i(x) = b_i - d_i(x)
```

- Normalize selected activations to obtain contribution weights `a_i(x)`.
- Expose the cases, distances, activations, biases, feature/case weights, and retrieval state actually used for prediction.
- Assign stable case identities independent of tensor slot or compaction.

### 3.2 Reuse — bounded neural adaptation

- For regression, preserve the current aggregate NN-CDH-style adapter as the foundation.
- For classification, add the aggregate nominal-residual extension specified below.
- Keep task-specific output semantics behind a common adapter contract; do not force one adapter output representation onto classification, regression, actor, critic, and T3 memory.
- Always report the retrieval-only prediction before adaptation and the final prediction after adaptation.
- Initially train reuse with retrieval frozen. Later test staged or alternating synchronization.

### 3.3 Revise — authorized human correction

Deferred during the first implementation; use `revise_enabled = false`.

The later interface may let an authorized person:

- inspect the activated cases and relevant feature/case weights;
- add or correct case content/solutions;
- explicitly force a selected eligible case into the active/retrieved set for a controlled decision, with the override visibly marked;
- increase or decrease a case bias or weight;
- change feature weights;
- protect, suppress, quarantine, restore, or remove a case; and
- request controlled retraining so the remainder of the model can accommodate the correction.

Every change must be logged, versioned, reversible where possible, and linked to affected decisions. Consequential suspected poisoning, harmful cases, outliers, mislabeled cases, rare/boundary cases, and specialized-domain judgments require human review. Routine eviction of clearly redundant low-utility cases may be automatic only after evidence and coverage checks.

### 3.4 Retain — bounded case-base maintenance

- Accumulate usage and outcome provenance for each stable case.
- Combine provenance quality with the trained case bias for a trustworthiness signal.
- Keep utility/exposure, redundancy, diversity, coverage, rarity, and protection separate.
- Select an active set under fixed capacity `K` first; compare adaptive growth only later.
- Perform maintenance at declared safe checkpoints, rebuild any affected neighborhood/index state, and preserve archives/restoration paths.
- Use the same maintenance-policy interface for classification, regression, RL actor, and RL critic stores, with task-specific contribution adapters.

### 3.5 MCB — cross-cutting stabilization

MCB is not a fifth CBR stage. It is an optional two-timescale stabilization factor for the case representation:

- a rapidly optimized online encoder represents incoming queries;
- a no-gradient memory encoder is updated as an exponential moving average of online parameters;
- the slower representation may stabilize stored-case geometry, prototypes, neighborhoods, and accumulated case statistics.

MCB must be switchable and must not be required for the first retention prototype. Test it only after retention works independently. Measure both benefit and possible over-stabilization: representation drift, neighborhood churn, selection churn, adaptation demand, and task learning.

The current MCB manuscript is under review. It tested a small domain and has not established gains on the completed NN-kNN classification/regression benchmark suites. Re-run those suites under matched conditions.

The PI reports that the MCB work is a collaboration with colleagues at Indiana University Bloomington and that implementation exists in Colab, but the proposal workspace does not yet contain the code, exact branch/notebook, author-role allocation, or a verified reproduction. Obtain and version those artifacts before integration; do not infer collaborator roles.

## 4. Current-code reference and integration boundary

At inspected commit `c09719576b3519e9878764190773916cd9ce82e6` on 2026-09-08:

- `model/nnknn_model.py` contains `NN_KNN_Model`, case capacity/append/compaction, retrieval distance and activation, classification aggregation, and regression NN-CDH calls.
- `model/nn_cdh.py` contains `NNCDHAdapter` and aggregate adaptation.
- `model/classification_workflow.py` and `model/regression_workflow.py` contain maintained benchmark workflows.
- `model/nnknn_rl_workflow.py` contains NN-kNN policy, value, and shared actor-critic paths plus existing case maintenance reporting.
- The ordinary RL profile used `case_capacity=500` and `case_maintenance_frequency=1000`, but these are historical code settings, not approved experimental values.
- Multiple case-store implementations exist; factor maintenance behind one shared interface.

Before coding, re-resolve all anchors against the live repository and preserve the current retrieval-only classification, existing regression path, current bias pruning, and current RL behavior as selectable references.

## 5. Classification reuse extension

### 5.1 Design inheritance

Reuse two ideas without restoring the old 2022 architecture wholesale:

1. From the older unified NN-CDH design: represent nominal value/label differences by subtracting one-hot encodings.
2. From the newer aggregate NN-kNN regression adapter: perform one neighborhood-level adaptation and condition on the retrieved solution estimate, not the retrieved context embedding.

The second rule prevents embedding leakage: if the adapter receives both the query-neighborhood difference and the neighborhood embedding, it can reconstruct the query representation and predict the target directly, bypassing case adaptation.

### 5.2 v0 task scope

Start with single-label classification over `C` mutually exclusive classes. Multi-label, hierarchical, ordinal, open-set, and expanding label spaces are deferred.

For query `q` and retrieved cases `i`, define:

```text
z_q       = query representation
z_i       = retrieved-case representation
a_qi      = normalized activation, a_qi >= 0 and sum_i a_qi = 1
e(y_i)    = one-hot stored case label
e(y_q)    = one-hot reference label during training/audit

z_bar_q   = sum_i a_qi * z_i
p0_q      = sum_i a_qi * e(y_i)
Delta_z_q = z_q - z_bar_q
```

`p0_q` is the retrieval-only class-mass prediction.

For an explicit nominal input field `j` not already represented in `Delta_z_q`:

```text
u_bar_qj   = sum_i a_qi * e(u_ij)
Delta_u_qj = e(u_qj) - u_bar_qj
Delta_u_q  = concat_j Delta_u_qj
```

Use the adapter input determined by representation coverage:

```text
h_q = [Delta_z_q, p0_q]
      if Delta_z_q already covers nominal inputs

h_q = [Delta_z_q, Delta_u_q, p0_q]
      otherwise
```

Do not duplicate nominal features. Decide and record coverage field-by-field in configuration; do not infer it silently from tensor shape or a module name. If explicit nominal differences are used, preserve each categorical field as its own one-hot group, fit vocabularies on training data only, store category order, and provide declared unknown/missing categories where needed. Use the same `a_qi` for all neighborhood aggregates.

### 5.3 Residual target and output

```text
r*_q    = e(y_q) - p0_q
r_hat_q = tanh(g_psi(h_q))
s_q     = p0_q + r_hat_q
y_hat_q = argmax_c s_q[c]
```

With one retrieved case, the target is zero when the retrieved class is correct; otherwise it adds mass to the target class and removes mass from the retrieved class. With a weighted neighborhood it is a generalized one-hot/weighted-cold residual with components in `[-1,1]` summing to zero.

Critical information-flow rule:

```text
allowed:     [z_q - z_bar_q, p0_q]
allowed:     [z_q - z_bar_q, Delta_u_q, p0_q]
not allowed: [z_q - z_bar_q, z_bar_q]
not allowed: [z_q, z_bar_q]
```

Do not pass raw `z_q`, individual `z_i`, `z_bar_q`, raw retrieved cases, or raw nominal values as extra inputs. A direct-query or retrieved-embedding head is a leakage/capacity control, not the proposed NN-CDH path.

### 5.4 Initial combined loss

```text
L_diff = MSE(r_hat_q, r*_q)
L_cls  = cross_entropy(s_q, y_q)

L_reuse_v0 = lambda_diff * L_diff + lambda_cls * L_cls

lambda_diff > 0
lambda_cls  > 0
```

Run `L_diff`-only, `L_cls`-only, and combined ablations. Do not add the minimal-adaptation penalty to the initial v0 training.

`s_q` is a score vector, not automatically a calibrated probability vector. Compare:

- **Primary starting mode — nominal residual scores:** train the explicit residual, use `argmax(s_q)`, and declare any softmax/temperature calibration used for reporting probabilities.
- **Engineering ablation — logit residual:** `p_final = softmax(log(clamp(p0_q, eps)) + delta_logits_q)`. This preserves `p0_q` when the correction is zero but no longer represents a literal nominal solution difference.

Do not pool or call the two output modes equivalent.

### 5.5 Training and diagnostics

Recommended first pass:

1. Train/freeze the retrieval core and case memory.
2. Construct adapter examples from leave-one-out neighborhoods of training cases.
3. Train the adapter without letting it alter retrieval.
4. Compare retrieval-only versus adapted decisions on identical case sets.
5. Log `p0_q`, `r*_q`, `r_hat_q`, `s_q`, losses, correction magnitude, selected cases, decision flips, and calibration.
6. Verify that retention statistics use the pre-adaptation contribution, so the adapter cannot hide poor cases.

Required tests include zero-correction identity, class permutation, output dimensionality, nominal-field coverage, unknown/missing categories, disabled-path equivalence, no raw-query/retrieved-embedding leakage, checkpoint reload, and invariance of pre-adaptation retention statistics.

## 6. Later component synchronization

After retrieve, reuse, and retain pass independent gates, coordinate them with separately inspectable objectives:

```text
L_pre   = task loss before adaptation
L_post  = task loss after adaptation
L_near  = query-to-retrieved-case proximity under a declared explanation metric
L_delta = residual/difference prediction loss
L_small = scale-normalized magnitude above a free-correction radius

L_R = w_pre  * L_pre  + w_post * L_post + w_near  * L_near
L_A = v_post * L_post + v_delta * L_delta + v_small * L_small
```

Interpretation:

- Retrieval should find cases that are already useful, remain useful after bounded correction, and are close under the learned explanation-relevant geometry.
- Adaptation should solve the task, predict the intended correction, and avoid unnecessarily large changes.
- `L_small` alone would reward doing nothing; `L_post` alone could reward drastic corrections that conceal bad retrieval. Report every term separately.
- Locality is a measurable explanation proxy, not proof that a person understands a case.

Use staged or alternating updates first. Do not allow one high-capacity component to silently carry another. Treat a differentiable maintenance selector as a later ablation; discrete selection at safe checkpoints is the primary start.

## 7. Free-correction radius

The minimal-adaptation penalty includes a task-specific region where small corrections are not penalized. This is calibrated **after** the NN-kNN core, learned metric, feature weights, and live per-case biases are sufficiently trained.

Let `t_star` be the declared post-core-training snapshot:

```text
b_i^star      = stop_gradient(b_i^live(t_star))
d_theta^star  = stop_gradient(d_theta(t_star))

P_bias = {(i,j): i != j and b_i^star - d_theta^star(x_i,x_j) >= 0}

tau_task = mean_{(i,j) in P_bias} d_y(y_i,y_j)

c_q = d_y(output_before_adaptation_q, output_after_adaptation_q)

L_small = mean_q [
  (max(0, c_q - tau_task) / (s_task + epsilon))^2
]
```

The trained bias defines the positive-activation neighborhood in problem space; observed label/output variation inside that neighborhood defines the output-space correction radius. Bias itself is not used as an output threshold because the units generally differ.

Use the direct frozen trained-bias snapshot as primary. Normalized, clipped, or shrunk trained-bias transforms are fallback/ablation conditions only if diagnostics show instability, incomparability, empty regions, or excessively broad regions. Freeze and log the matching learned-distance state. Reuse the model's learned representation, feature weighting, and other learned distance components; do not substitute an unrelated raw-input metric.

The initial bias is only a warm start:

```text
r_i^(k)       = kth nearest non-self learned distance from training case i
b_default^(0) = mean_i r_i^(k)
b_i^live(0)   = b_default^(0)
```

Start with `k=5`. It is not the free-correction threshold and not a central claim. A learned-distance percentile initializer is an ablation. The exact rule for choosing `t_star`, `d_y` for nonclassification tasks, `s_task`, and robust versus arithmetic aggregation remains unresolved and should be fixed in the later protocol.

Safeguards:

- use training data only and exclude self-pairs;
- stop gradients through pair selection and `tau_task` during adapter training;
- checkpoint the calibration snapshot or enough metadata to reproduce it;
- recompute only at declared outer checkpoints, never per minibatch;
- handle no-pair, overly broad, and near-zero-threshold cases explicitly;
- use nonzero `s_task`, not division by `tau_task`, for stability; and
- retain zero-threshold, initial-bias-region, exact-kth-pair, and transformed-trained-bias comparisons as ablations when feasible.

## 8. Case provenance and trustworthiness

### 8.1 Classification provenance

For case `i`, normalized activation `a_i(x)`, stored class `c_i`, and reference class `y_x` on a designated training-audit/maintenance set `D`:

```text
R_i = sum_x 1[i is retrieved for x]
A_i = sum_x a_i(x)
C_i = sum_x a_i(x) * 1[c_i = y_x]
H_i = sum_x a_i(x) * 1[c_i != y_x]
```

- `R_i`: retrieval exposure.
- `A_i`: cumulative activation/contribution mass.
- `C_i`: activation-weighted support for a correct class.
- `H_i`: activation-weighted support for an incorrect class.

Keep `R_i` even when `A_i = C_i + H_i`; it distinguishes frequent weak retrieval from rare strong retrieval.

### 8.2 Smoothed provenance quality

```text
Q_i = (C_i + s) / (C_i + H_i + 2s),  s > 0
```

`Q_i` approaches 1 for predominantly helpful observed support, 0 for harmful support, and 0.5 for balanced or insufficient evidence. A smoothed 0.5 is uncertainty, not mediocrity. Interpret it only with minimum `R_i` and/or `A_i`.

### 8.3 Bias score

Map trained bias `b_i`, optionally with its change from initialization, to cohort-comparable `B_i in [0,1]`, where lower values indicate learned disutility. Start with within-class or otherwise matched percentiles; median/MAD is an ablation. Do not pool raw biases across incompatible class/action/cohort conditions without checking their scale.

### 8.4 Combined trustworthiness

Use the PI-approved weighted geometric mean:

```text
T_i = Q_i^alpha * B_i^(1-alpha),  0 < alpha < 1

log(T_i) = alpha * log(clip(Q_i, eps, 1))
         + (1-alpha) * log(clip(B_i, eps, 1))
```

The geometric form prevents one excellent component from fully compensating for a nearly zero component. Arithmetic combination, provenance-only, and bias-only are ablations. Always retain and report `Q_i`, `B_i`, `T_i`, `R_i`, `A_i`, `C_i`, and `H_i` separately.

`T_i` is a **trustworthiness score**, not the complete retention score.

### 8.5 Harm, low utility, and uncertainty

- **Harm flag:** sufficiently evidenced negative contribution, poisoning, mislabeling, outlier behavior, or repeated support for wrong outcomes.
- **Low-utility flag:** rarely retrieved, weak contribution, or redundant under the current workload.
- **Uncertain:** insufficient exposure or conflicting provenance/bias signals.

A low-use case is not automatically harmful. It may be a rare, boundary, subgroup, domain-shift, or future-coverage case.

## 9. Contribution semantics beyond classification

Do not reuse class equality for continuous or sequential targets.

### 9.1 Regression and value prediction

Use the pre-adaptation retrieval output. The reference audit is counterfactual removal with activation renormalization:

```text
Delta_i(x) = Loss(f_without_i(x), y_x) - Loss(f_with_i(x), y_x)

C_i = sum_x max(Delta_i(x), 0)
H_i = sum_x max(-Delta_i(x), 0)
```

Positive `Delta_i` means the case reduced loss. A cheaper directional approximation may be tested but not substituted without validation:

```text
g_i(x) = a_i(x) * (y_i - y_hat_pre) * (y_x - y_hat_pre)
```

It indicates whether the case pulls the estimate toward the target, but can miss overshoot and nonlinear effects.

### 9.2 RL actor

The helpful/harmful signal is unresolved. Candidate definitions include signed GAE advantage for the executed action, counterfactual policy-loss change after case masking, and longer-horizon return effect. Resolve this with the RL collaborators and validate against direct interventions.

### 9.3 RL critic

Treat the critic as regression against a declared target. Candidate uses are GAE value targets for training diagnostics and discounted Monte Carlo returns for separate audit rollouts. Preserve distinctions among online critic labels, mutable/trainable stored labels, and lagged target-critic bootstrap values.

## 10. Constrained retention under fixed `K`

### Stage 1 — protection and evidence

1. Protect cases required for minimum class/action/cohort/subgroup/temporal/domain coverage.
2. Identify rare, boundary, shifted, or PI-designated cases.
3. Mark insufficient exposure rather than inferring harm.
4. Compute trustworthiness components and observed utility.
5. Estimate redundancy within a compatible learned representation and cohort.

### Stage 2 — routine eviction eligibility

A nonprotected case is eligible for routine reversible eviction only when configured evidence supports at least one condition:

- low observed utility plus high redundancy;
- sufficiently evidenced low trustworthiness plus low utility or high redundancy and replaceable coverage; or
- a new case gives better utility or coverage under the same budget.

In consequential settings, suspected harmful or poisoned cases are review-only until an authorized reviewer acts. Research ablations may test trust-only eviction in isolated, fully recoverable experiments.

### Stage 3 — fill capacity

Preserve protected cases first, then fill remaining capacity using a declared combination or constrained selection of trustworthiness, utility, diversity/redundancy, and coverage. Compare:

1. full memory where feasible;
2. ordinary downsampling/random selection;
3. stratified random selection;
4. current bias-only pruning;
5. provenance-only `Q_i`;
6. normalized bias-only `B_i`;
7. geometric trustworthiness `T_i` after evidence filtering;
8. utility plus redundancy;
9. trustworthiness plus utility; and
10. trustworthiness plus utility plus coverage/diversity.

Use identical `K`, insertion stream, seeds, and training/tuning budgets. Tie-breaking must be deterministic and logged.

## 11. Human revision evaluation

When T1.2 enables revision, use these checkpoints:

- **M0:** trained model before correction. This is the baseline state, not a separate retraining run.
- **M1:** immediately after the human correction, without retraining.
- **M2:** after the correction plus controlled retraining so other parameters can accommodate the intervention.

Add a matched no-correction retraining comparator only if retraining is itself part of the experimental question. There is no reason to retrain an unchanged M0 solely to manufacture a control.

Measure:

- overall task quality and calibration;
- the targeted decisions and whether desired decisions flip;
- correctness of intended flips;
- harmful or reversed flips;
- collateral changes on unaffected cases/classes/subgroups/tasks;
- retraining cost and whether the correction persists;
- reviewer time, actions, confidence, and burden;
- memory size, latency, and case turnover; and
- causal agreement between displayed influence and observed intervention effect.

Use both direct technical intervention experiments and a bounded UI study. The required UI study should use a low-risk/common-knowledge task so model users can make objective judgments without specialized expertise. A healthcare expert study is optional and contingent on qualified expertise, governance, and approval.

## 12. Configuration surface

Suggested names may change, but all scientific factors must be explicit:

```text
case_maintenance_policy = "provenance_bias_coverage"
case_capacity = K
case_maintenance_frequency = ...
case_score_smoothing = s
case_trust_alpha = alpha
case_min_retrieval_count = ...
case_min_activation_mass = ...
case_bias_normalization = "within_cohort_percentile"
case_redundancy_metric = ...
case_min_per_class_or_action = ...
case_archive_evictions = true
case_revision_enabled = false
case_stats_source = ...

mcb_enabled = false

classification_adapter_enabled = false
classification_adapter_architecture = "aggregate_label_conditioned"
classification_adapter_input = "auto_by_representation_coverage"
classification_adapter_nominal_delta_mode = "auto"
classification_representation_covers_nominal_inputs = ...
classification_adapter_output_mode = "nominal_residual_scores"
classification_adapter_freeze_retrieval_first = true
classification_adapter_lambda_diff = ...
classification_adapter_lambda_cls = ...
classification_adapter_lambda_adapt_cost = 0.0
classification_adapter_probability_mode = ...

component_sync_enabled = false
component_sync_schedule = "alternating"
component_sync_problem_distance_source = "shared_nnknn_learned_distance_interface"
component_sync_lambda_pre = ...
component_sync_lambda_post = ...
component_sync_lambda_near = ...
component_sync_lambda_delta = ...
component_sync_lambda_small = ...
component_sync_free_correction_enabled = true
component_sync_free_correction_calibration_stage = "after_core_training"
component_sync_free_correction_local_selector = "positive_case_bias_activation_region"
component_sync_free_correction_bias_source = "frozen_trained_per_case_bias"
component_sync_free_correction_trained_bias_transform = "identity"
component_sync_free_correction_bias_initializer = "mean_kth_neighbor_learned_distance"
component_sync_free_correction_bias_initializer_knn_k = 5
component_sync_free_correction_label_metric_classification = "one_hot_euclidean"
component_sync_free_correction_estimator = "mean_label_distance_within_bias_radius"
component_sync_free_correction_recompute = "declared_outer_checkpoints_only"
component_sync_maintenance_policy = "checkpoint_discrete"
```

Ellipses are unresolved. The classification adapter, MCB, revision, and synchronization remain off by default until their respective tests/gates pass.

## 13. Implementation sequence

### Phase 0 — reproduce and instrument

- Record code/environment/data versions.
- Run existing import, supervised, and RL smoke checks.
- Add stable IDs and statistics without changing behavior, gradients, RNG streams, or insertion.
- Export per-case activations/contributions on synthetic tests.

### Phase 1 — supervised retention prototype

- Implement classification provenance, bias normalization, trustworthiness, protection, redundancy, reversible selection, and event logs.
- Test synthetic corruptions, redundancy, rare cases, boundary cases, and shifted subdomains.
- Add regression counterfactual auditing on a tractable subset using pre-adaptation output.

### Phase 2 — classification reuse

- Add the aggregate label-conditioned adapter behind a switch.
- Build leave-one-out adapter examples and freeze retrieval first.
- Run single-loss/combined-loss and nominal-residual/logit-residual conditions.
- Verify pre/post reporting, decision flips, calibration, and no information leakage.

### Phase 3 — RL integration

- Apply the common statistics/maintenance interface separately to actor and critic memories.
- Preserve role-specific identities, labels, capacities, logs, and targets.
- Compare current, retention-enhanced, and inherited component combinations under matched budgets.

### Phase 4 — component harmonization

- Add separate retrieval, post-adaptation, proximity, residual, and minimal-correction terms.
- Use staged/alternating updates and checkpoint-discrete maintenance.
- Compare independent, retrieval-reuse synchronized, and retrieval-reuse-retain synchronized training.

### Phase 5 — MCB interaction

- Add MCB/no-MCB after retention is independently testable.
- Measure drift, neighborhood/selection churn, accumulated-statistic reliability, adaptation magnitude, and learning.

### Phase 6 — broader confirmation and revision

- Re-run recoverable completed-paper suites in pre-MCB form, then matched MCB/full-cycle variants.
- Add the selected modern transfer family and appropriate contemporary baselines.
- Enable human revision and UI evaluation only after the causal trace and archive/restore mechanisms pass.

## 14. Evaluation plan

### Controlled foundations

- Completed IJCAI-2025 classification tasks supported by the maintained repository.
- Completed IJCAI-2026 regression tasks, with retrieval-only and adapted outputs separated.
- Synthetic tasks with known feature relevance, redundancy, corruption, rare cases, and shift.
- MCB manuscript conditions when code/data are available, clearly labeled under review.

Use a two-step legacy protocol:

1. reproduce or transparently reconstruct the old pre-MCB architecture;
2. compare MCB and full-cycle variants under matched recoverable splits, encoders, tuning budgets, case budgets, and primary metrics.

All prior classification/regression results are pre-MCB. Do not attribute them to the new system.

### Modern transfer

Select one manageable structured/tabular family that supports both standard performance comparison and controlled bounded-memory, shift, harmful-case, and correction stress tests. TabArena is a candidate source, not yet a commitment. Candidate comparator families include boosted trees, matched MLPs/TabM, and TabPFN where dataset size, preprocessing, pretrained-compute disclosure, licensing, and budget permit.

Do not promise every method on every task or chase a live leaderboard. Freeze a version/subset and use the same scientific protocol.

### Outcome families

1. Predictive quality: accuracy, RMSE, calibration, uncertainty, and task-appropriate margins.
2. Retrieval quality: alignment, counterfactual contribution, locality, and stability.
3. Reuse quality: gain over retrieval-only, residual size, good/harmful flips, and evidence that the adapter remains case-conditioned.
4. Retention quality: performance at matched `K`, rare/shifted competence, harmful-case prevalence, redundancy, and oracle regret where available.
5. Efficiency: active/archive memory, lookup latency, training time, and maintenance overhead.
6. Stability: representation drift, neighborhood churn, selected-case churn, and seed sensitivity.
7. Auditability: complete identities, statistics, and reason codes.
8. Synchronization: each component loss and whether one component compensates for another.
9. Correctability: intended effects, collateral effects, persistence, and reviewer burden.

T1 success requires more than a top score:

- beat random/downsampled and applicable selection baselines at matched case budgets;
- approach full-case-base NN-kNN with substantially fewer, higher-quality cases;
- remain statistically competitive with strong contemporary methods under aligned protocols; and
- materially improve causal case-level faithfulness, correction, containment, stability, and bounded-memory efficiency.

Use benchmark-specific non-inferiority/practical-equivalence margins and minimum improvement thresholds fixed after pilots but before confirmatory runs. Reserve “state of the art” for results that actually establish it.

## 15. Required tests and gates

### Unit/integration tests

- zero exposure yields `Q_i = 0.5` under symmetric smoothing;
- correct activation increases `Q_i`; harmful activation decreases it;
- geometric direct/log-space scores match;
- compaction preserves IDs, statistics, labels/solutions, biases, glocal weights, optimizer alignment, and target-critic alignment;
- protected coverage cannot be violated;
- archive/restore reproduces case state;
- deterministic seeds/ties give identical decisions;
- statistics-only mode does not alter baseline behavior;
- test/evaluation data never affect retention/calibration;
- checkpoint reload preserves active/archive cases, policy, adapter, class order, and calibration state;
- actor/critic namespaces never collide;
- free-radius reconstruction from the saved bias/metric snapshot reproduces `P_bias` and `tau_task`, excludes self/test pairs, and gives zero/increasing penalty at/beyond the threshold; and
- degenerate thresholds remain numerically stable.

### Progression gates

1. **Instrumentation:** all identity/statistic tests pass without behavior change.
2. **Supervised retention:** at matched `K`, at least one policy consistently improves the intended trade-off over random/downsampling and current bias-only pruning across multiple seeds.
3. **Classification reuse:** adaptation improves a prespecified pre/post outcome without unacceptable calibration, harmful flips, or leakage; report gains separately from retention.
4. **RL mechanism:** improved case selection changes learning, efficiency, or stability relative to matched current maintenance.
5. **Synchronization:** coordination improves a prespecified system outcome without degrading locality or requiring correction beyond the declared bound.
6. **MCB interaction:** determine whether MCB improves representation/statistic/selection stability, even if task performance does not improve.
7. **Failure analysis:** identify which stage is limiting if a gate fails.

Exact numerical gate thresholds remain unresolved.

## 16. Risks and useful fallbacks

- **Retention does not improve task quality:** report the boundary; determine whether full memory or current bias pruning is preferable and use the case statistics diagnostically.
- **Adapter hides bad retrieval:** enforce pre-adaptation metrics and freeze/alternate training; reduce adapter capacity or keep retrieval-only.
- **MCB over-stabilizes:** tune or disable it and characterize the stability/adaptability trade-off.
- **Raw trained-bias regions fail:** use a declared normalized/clipped/shrunk transform only as a diagnostic fallback.
- **Human correction causes collateral damage:** preserve immediate intervention without retraining, constrain retraining, restore snapshots, and report limits.
- **Universal-module scope becomes too broad:** retain selected compatible roles and document where conventional neural components are better.

## 17. Deliverables

Return the artifacts required by the [shared contract](SHARED_EXPERIMENT_AND_DATA_CONTRACT.md), plus:

- the shared maintenance API and role-specific contribution adapters;
- classification reuse implementation and full ablation configuration;
- trained-bias/free-radius calibration artifacts;
- completed-paper pre-MCB reconstruction and matched MCB/full-cycle results;
- fixed-`K` selection curves and full-memory comparisons;
- synthetic corruption/rarity/shift diagnoses;
- intervention and, later, UI-study results; and
- a mechanism-level failure analysis suitable for deciding what T2 inherits.

Open numerical and task choices are centralized in [OPEN_DECISIONS_AND_HANDOFF_CHECKLIST.md](OPEN_DECISIONS_AND_HANDOFF_CHECKLIST.md).
