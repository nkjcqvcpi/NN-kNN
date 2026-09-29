# T1 technical implementation handoff: case selection first, classification reuse next

## Audience and purpose

This document is for the collaborator implementing the mechanism-development sub-aim of the merged T1 in the maintained NN-kNN repository. It converts the CAREER research concept into a staged engineering and experimental plan. The collaborator already has access to the current Git repository; do not copy proposal-only files into the codebase unless they are useful there.

The source checkout inspected while preparing this handoff was:

- repository: `https://github.com/Heuzi/NN-kNN.git`;
- local reference: `D:\NN-kNN`;
- inspected commit: `c09719576b3519e9878764190773916cd9ce82e6`;
- inspected date: 2026-09-08.

The repository may advance. Before implementation, record the actual starting commit and re-resolve all code anchors below. Follow the repository's current `AGENTS.md` and `HANDOFF.md`.

## Immediate objective

The immediate hypothesis is that NN-kNN underperforms in the active reinforcement-learning work because its case memory is not consistently retaining and using the best cases. This is the working explanation from the PI and collaborator group, not yet a demonstrated causal diagnosis.

The first T1 implementation should therefore prioritize:

1. instrumenting which cases are retrieved and how strongly they contribute;
2. estimating case trustworthiness and observed utility;
3. selecting a strong active case set under a fixed budget `K`;
4. protecting coverage while reducing redundancy;
5. comparing the new policy with the current bias-only pruning, downsampling/random selection, and full-memory conditions; and
6. testing whether improved case selection changes supervised and RL outcomes; and
7. adding a separately testable classification NN-CDH reuse path that preserves the current aggregate, retrieval-conditioned design.

## Immediate scope and non-goals

### Enabled now

- **Retrieve:** use the current NN-kNN activation path and expose stable per-case contribution data.
- **Reuse:** preserve the current regression path and add the classification extension specified below as a separate, bounded workstream. Use a common adapter contract with task-specific output semantics; do not attempt one universal adapter for every task.
- **Retain:** implement and evaluate provenance-, bias-, utility-, redundancy-, and coverage-aware case selection.
- **MCB:** support as an optional experimental factor when its maintained implementation is available; do not make the first retention implementation depend on it.

### Disabled or deferred now

- **Revise is disabled:** no human interface, case editing, feature-weight editing, retraining after human correction, or user study is required in the first implementation.
- Keep the data structures compatible with later revision by using stable case IDs, event logs, and reversible archives.
- Do not claim that T1's full revise stage is implemented merely because logging or archive restoration exists.
- Classification adaptation is now in scope by PI direction. RL- and LLM-specific adaptation remain deferred unless separately agreed.
- Do not launch full expensive benchmark sweeps until smoke, unit, and small diagnostic tests pass.

Suggested top-level feature flag:

```text
revise_enabled = false
```

## Current implementation anchors

At the inspected commit:

- `model/nnknn_model.py`
  - `NN_KNN_Model` begins near line 522.
  - Active capacity, append, and compaction methods are near lines 668-769.
  - Retrieval distances, glocal weights, case scores, normalization, and output aggregation are near lines 930-1180.
  - `bias_minus_distance` computes `z_i(x) = b_i - d_i(x)` near lines 968-971.
  - Classification currently aggregates normalized activation against one-hot case labels to produce class probability mass near lines 1063-1073; there is no classification NN-CDH call at the inspected commit.
  - Regression calls the aggregate or per-case NN-CDH path near lines 1080-1179.
- `model/nn_cdh.py`
  - `NNCDHAdapter` begins near line 112.
  - Aggregate adaptation computes an attention-weighted retrieved case/label and a neural correction near lines 120-129 and 237 onward.
- `model/classification_workflow.py`
  - Maintained training/evaluation and repeated benchmark entry points begin near lines 299, 387, and 680.
- `model/regression_workflow.py`
  - Maintained training/evaluation and repeated benchmark entry points begin near lines 903, 958, 1452, and 1875.
- `model/nnknn_rl_workflow.py`
  - `NNKNNPolicyNetwork`, `NNKNNValueNetwork`, and the shared actor-critic implementation begin near lines 108, 428, and 764.
  - The current configuration uses `case_capacity=500` and `case_maintenance_frequency=1000` by default for ordinary profiles.
  - Current actor/critic insertion, compaction, protection, and bias-based pruning paths are distributed across the policy, value, and shared implementations.
  - Run-level maintenance events are already written to `case_maintenance.csv`.

The current code has multiple case-store implementations. The T1 policy should be factored behind one common maintenance interface rather than independently reimplemented in each actor, critic, classification, and regression workflow.

## Classification reuse extension: old nominal differences, new NN-kNN conditioning

### What is being reused and what is being changed

The PI-supplied ICCBR-2022 paper introduced a one-hot/one-cold representation for nominal differences. If `e(v)` is a one-hot encoding, the difference between two nominal values is the vector subtraction of their encodings. It is all zeros when the values match; otherwise it has one positive and one negative component. The paper used this construction for nominal input attributes and class-label solution differences.

Do not restore the complete 2022 architecture unchanged. That version conditioned NN-CDH on the retrieved problem description as adaptation context and used a retrieved-minus-query/subtraction sign convention. The current NN-kNN adapter instead performs one adaptation after aggregating a retrieved neighborhood and conditions on the retrieved solution estimate. The supplied newer design explicitly replaces the retrieved context embedding with the retrieved label to prevent the adapter from reconstructing the query embedding and bypassing adaptation.

The classification extension should therefore reuse the **nominal-difference representation**, while preserving the **new aggregate, retrieved-label-conditioned architecture**.

### v0 scope

Start with ordinary single-label classification over `C` mutually exclusive classes. Multi-label, hierarchical, ordinal, open-set, and continually expanding label spaces are out of scope for v0.

For query `q`, let:

- `z_q` be its current NN-kNN feature representation;
- `z_i` be the representation of retrieved case `i`;
- `a_qi >= 0` be its normalized case activation, with `sum_i a_qi = 1`;
- `e(y_i)` be the one-hot stored label of case `i`; and
- `e(y_q)` be the one-hot reference label during training or audit.

Compute the same aggregate quantities used by the new regression adapter:

```text
z_bar_q = sum_i a_qi * z_i
p0_q    = sum_i a_qi * e(y_i)
Delta_z_q = z_q - z_bar_q
```

For every nominal input field `j` that is **not already represented in `Delta_z_q`**, compute the explicit grouped difference:

```text
u_bar_qj  = sum_i a_qi * e(u_ij)
Delta_u_qj = e(u_qj) - u_bar_qj
Delta_u_q = concat_j Delta_u_qj
```

If the feature extractor processes nominal fields together with the other inputs, do not compute or concatenate `Delta_u_q`; their contribution is already represented in `Delta_z_q`. More generally, decide based on representation coverage, not merely on whether a module is named a feature extractor.

`p0_q` is the retrieval-only class-mass prediction. The nominal adaptation target is:

```text
r*_q = e(y_q) - p0_q
```

This is the new design's plus-sign convention. With a single retrieved case, `r*_q` is exactly a sign-reversed version of the 2022 paper's one-hot/one-cold solution difference: it is zero if the retrieved class is correct and otherwise adds mass to the target class and removes mass from the retrieved class. With multiple retrieved cases, it is a generalized **one-hot/weighted-cold residual** whose components still lie in `[-1,1]` and sum to zero.

Define the adapter input according to whether the shared representation already covers the nominal fields:

```text
h_q = [Delta_z_q, p0_q]                 if Delta_z_q covers nominal inputs
h_q = [Delta_z_q, Delta_u_q, p0_q]      otherwise

r_hat_q = tanh(g_psi(h_q))
s_q     = p0_q + r_hat_q
y_hat_q = argmax_c s_q[c]
```

Critical information-flow rule:

```text
adapter input = [z_q - z_bar_q, p0_q]
             or [z_q - z_bar_q, Delta_u_q, p0_q]
adapter input != [z_q - z_bar_q, z_bar_q]
adapter input != [z_q, z_bar_q]
```

Do not pass `z_i`, `z_bar_q`, the raw retrieved case, or the raw query as an additional adapter input. When used, `Delta_u_q` is an explicit difference representation, not the raw query or neighborhood vector; `p0_q` is allowed because conditioning on the retrieved label estimate is the intended new design. If a later experiment intentionally adds a direct-query or retrieved-embedding head, label it as a leakage/capacity control rather than as NN-CDH adaptation.

### Nominal input attributes

Use one explicit nominal-difference channel only when those nominal fields are not already included in the representation used to compute `Delta_z_q`. Preserve each such categorical field as its own one-hot group during tabular preprocessing. Its neighborhood-relative nominal difference for field `j` is:

```text
Delta_u_qj = e(u_qj) - sum_i a_qi * e(u_ij)
```

When `Delta_z_q` does not cover the nominal fields, concatenate their grouped differences as `Delta_u_q`, then use `[Delta_z_q, Delta_u_q, p0_q]`. This is the neighborhood-aggregate analogue of the 2022 paper's one-hot subtraction. When the feature extractor incorporates nominal fields along with the other features, use `[Delta_z_q, p0_q]` and omit `Delta_u_q` because it would duplicate information already carried by the shared representation difference. This conditional rule is the PI-confirmed v0 design.

Do not separately add raw nominal values, consistently scaled numeric differences, the raw query, raw retrieved cases, `z_i`, or `z_bar_q`; numeric information and any jointly extracted nominal information remain represented through `Delta_z_q`.

When explicit nominal differences are required, fit every nominal vocabulary on training data only. Preserve field boundaries and category order in the checkpoint/configuration, and provide an explicit unknown/missing category where the dataset requires it. Use the same retrieval weights `a_qi` for `z_bar_q`, `p0_q`, and every `u_bar_qj` so all adapter inputs describe one coherent neighborhood. Record in the run configuration whether `Delta_z_q` covers each nominal field; do not infer this silently from tensor shape.

### Confirmed immediate combined loss

For the first classification adapter, train the generalized nominal residual and the final classification decision together:

```text
L_diff = mean_squared_error(r_hat_q, r*_q)
L_cls  = cross_entropy(s_q, y_q)
L_reuse_v0 = lambda_diff * L_diff + lambda_cls * L_cls

lambda_diff > 0
lambda_cls  > 0
```

The PI confirmed this combined-loss design on 2026-09-09. `L_diff` teaches the adapter the explicit neighborhood-to-query solution difference inherited from NN-CDH; `L_cls` ensures that the resulting corrected score vector solves the actual classification task. Keep both weights configurable and report `L_diff`-only, `L_cls`-only, and combined conditions so the contribution of each term remains visible.

The 2022 paper used a bounded `tanh` solution-difference output, mean-squared error, additive/subtractive class scores, and `argmax`. The added classification term is a proposed T1 design choice, not a result from that paper. Do not add a minimal-adaptation penalty to the immediate v0 experiment; introduce it as a separately weighted, separately reported term in the later component-synchronization phase below.

`s_q` is a class-score vector, not automatically a calibrated probability distribution. If downstream code requires probabilities, expose the transformation explicitly rather than silently treating `s_q` as probability mass. Compare at least:

1. `nominal_residual_scores`: train the generalized nominal residual and use `argmax(s_q)`; obtain reporting probabilities from a declared softmax/temperature calibration step if needed; and
2. `logit_residual`: use `p_final = softmax(log(clamp(p0_q, eps)) + delta_logits_q)`, which preserves `p0_q` exactly when the correction is zero but no longer treats the network output as a literal one-hot/one-cold solution difference.

The first mode is the recommended primary test of the PI's requested synthesis. The second is a probability-preserving engineering ablation. Do not combine their targets or call them equivalent.

### Eventual T1 component-synchronization objective

The mature T1 system should not optimize retrieve, reuse, and retain as isolated modules. It should coordinate them while retaining separately inspectable losses:

```text
L_pre   = task loss before adaptation
L_post  = task loss after adaptation
L_near  = explanation-relevant query-to-retrieved-case proximity
L_delta = residual/difference prediction loss
L_small = scale-normalized magnitude of the applied adaptation

L_R = w_pre  * L_pre  + w_post * L_post + w_near  * L_near
L_A = v_post * L_post + v_delta * L_delta + v_small * L_small
```

`L_R` asks retrieval to return cases that are already useful, remain useful after bounded adaptation, and are nearby under a declared explanation metric. `L_A` asks adaptation to solve the task, predict the intended correction, and avoid unnecessarily large changes. These pressures are complementary: `L_small` alone would reward doing nothing, while `L_post` alone could reward drastic corrections that conceal poor retrieval. `L_near` is a measurable locality criterion or explanation proxy; it is not by itself proof that a case is understandable to a person.

#### PI-confirmed free-correction radius

The minimal-adaptation term must include a small task-specific region in which correction carries no magnitude penalty. The PI initially proposed using each case's fifth nearest neighbor, then refined the design on 2026-09-09: use the **case-bias activation-radius idea** to define which training cases count as local, rather than taking one exact kth-neighbor label pair. The PI clarified that this calibration happens **after the NN-kNN core and its live per-case biases are well trained**. The initial/default bias is only a training starting point; it is not the primary free-adaptation calibration source.

Let `t_star` denote the post-core-training checkpoint used for calibration. The exact checkpoint-selection rule is deferred to the later implementation protocol and is not a proposal-level decision. Freeze the trained retrieval state at that point:

```text
b_i^star = stop_gradient(b_i^live(t_star))

d_theta^star = stop_gradient(d_theta(t_star))

P_bias = {(i, j): i != j and b_i^star - d_theta^star(x_i, x_j) >= 0}

tau_task = mean_{(i, j) in P_bias} d_y(y_i, y_j)

c_q = d_y(output_before_adaptation_q, output_after_adaptation_q)

L_small = mean_q [
    (max(0, c_q - tau_task) / (s_task + epsilon))^2
]
```

The training initializer remains NN-kNN's original data-derived default-bias policy:

```text
r_i^(k) = kth_nearest_{j != i} d_theta^(0)(x_i, x_j)

b_default^(0) = mean_i r_i^(k)

b_i^live(0) = b_default^(0) for every training case i
```

Here `b_i^live(0)` is only the common starting value. The live per-case biases are learned and may diverge during core training, changing retrieval and the knowledge passed to adaptation. Start the initializer at `k=5`; this is a practical warm start rather than a central research claim. A distance-percentile initializer remains an initialization ablation, and broader `k` sensitivity is needed only if conclusions materially depend on it.

The PI confirmed the direct frozen trained-bias snapshot `b_i^star` as the primary post-training calibration. This preserves each case's learned activation radius without introducing another transformation. Cohort normalization, clipping, shrinkage, or another value derived from trained bias is a fallback/ablation only if diagnostics show that raw learned biases are unstable, incomparable, or yield empty or excessively broad activation regions. Any derived form must be computed only from the trained core, frozen within the subsequent adapter-training phase, and logged reproducibly.

`d_theta^star` selects local pairs and `d_y` measures their label/output difference; they need not be the same metric. `d_theta^star` must reuse the learned NN-kNN geometry—including the learned representation, feature-distance weights, and other learned distance components available in that model condition—rather than introduce an unrelated raw-input metric. The trained case bias is not used directly as a label-space correction threshold because bias/distance and output correction generally have different units. Instead, it defines the learned positive-activation region `b_i^star - d_theta^star >= 0`, and observed label variation inside that region supplies `tau_task`. `s_task` is a nonzero training-only output scale used to make the excess penalty comparable across tasks. A correction with `c_q <= tau_task` is permitted at zero adaptation-cost penalty. It is not rewarded regardless of correctness: `L_post` must still require the correction to improve the task.

This threshold has an intuitive interpretation: NN-kNN's case bias defines the region in which a case is intended to activate before normalization. If labels ordinarily vary by roughly `tau_task` inside those activation regions, a correction within that amount is normal local adaptation; a larger correction suggests that adaptation may be overriding rather than reusing the retrieved knowledge.

Current-code anchor at commit `c09719576b3519e9878764190773916cd9ce82e6`: `find_default_bias_knn` in `model/nnknn_model.py` estimates the initial default bias as the average distance to the kth nearest non-self comparison case; `find_default_bias_percentage` supplies a distance-percentile alternative; and the default score mode applies `bias - distance`. The helper signature defaults to `k=5`, while the maintained model configuration sets `top_k_for_default_case_activation=20`. However, the shipped configuration also uses `case_activation_by_top_k_average=False`, `bias_manual_set=True`, and `bias_manual_value=0.0`, so neither data-derived initializer is active. T1 must invoke and version the chosen initialization policy explicitly, then separately snapshot the trained biases and trained learned-distance state for free-adaptation calibration.

Implementation safeguards:

- use only the training split and exclude self-neighbors;
- compute neighbors with the same learned geometric distance used by the corresponding NN-kNN retrieval condition, including its learned representation and feature weighting whenever available;
- log the post-core-training checkpoint used as `t_star`; specify its exact selection rule later in the implementation protocol;
- freeze and log both the post-training bias snapshot and the learned-distance snapshot used to construct `P_bias`;
- stop gradients through `P_bias` and `tau_task`, store them or their reproducible calibration metadata in the run checkpoint, and do not let validation or test labels affect them;
- if joint learning materially changes the representation, recompute only at declared outer-cycle checkpoints and log both the old and new threshold;
- define an explicit fallback when a case has no non-self case inside its calibration radius, rather than silently inserting a zero label distance;
- report activation-region size and the neighbor-label-distance distribution, not only its mean; and
- retain the initial-bias-region estimator, superseded exact-kth label-pair estimator, alternative trained-bias transformations, alternative initialization rules, and robust label-distance aggregates as ablations.

For multi-output regression, standardize label dimensions using training statistics before `d_y`. For classification, the PI-confirmed primary metric is one-hot Euclidean distance:

```text
d_y_classification(y_i, y_j) = ||e(y_i) - e(y_j)||_2
```

Each activated local pair therefore contributes either `0` for the same class or `sqrt(2)` for different classes, and `tau_task` is `sqrt(2)` times the class-disagreement rate inside the case-bias activation region. This is intentionally model-independent and easy to audit, but it is coarse. Retain distance between class-probability vectors as a prespecified ablation rather than the primary metric. RL actor and critic thresholds require separate definitions because actions are discrete while value targets are continuous and nonstationary.

#### Learned-metric coherence invariant

Whenever T1 uses *problem-space proximity*—for retrieval, `L_near`, threshold-neighbor selection, redundancy detection, or coverage diagnostics—it should call the same versioned NN-kNN learned-distance interface for that model condition. Do not independently reimplement a raw Euclidean distance for one component unless the learned metric is unavailable; any fallback must be explicit and compared as an ablation. This invariant does not require problem distance `d_theta`, label/output distance `d_y`, trustworthiness `T_i`, and case bias to become one quantity: they answer different questions and should remain separately logged.

Retention should optimize the selected active set `S`, not misuse the per-case trustworthiness score `T_i` as the full system loss:

```text
minimize over S:
L_M(S) = m_pre   * L_pre(S)
       + m_post  * L_post(S)
       + m_untrust * ActiveUntrustworthiness(S)
       + m_red   * Redundancy(S)
       + m_cov   * CoverageGap(S)
       + m_churn * SelectionChurn(S)

subject to |S| <= K and ProtectedCases subset_of S
```

Because case selection is discrete, `L_M` need not be forced into one end-to-end differentiable expression. A practical full-cycle schedule is:

1. update retrieval while holding the adapter and active case set fixed;
2. update adaptation while holding retrieval and the case set fixed;
3. update the active case set at a safe maintenance checkpoint using logged utility, trustworthiness, redundancy, and coverage evidence;
4. rebuild neighborhoods and dependent statistics after any selection change; and
5. repeat until the stopping rule is reached.

This staged/alternating design follows the coordination rationale of the PI's 2021 retrieval-adaptation work while extending the research question to maintenance. The prior paper coordinated retrieval and adaptation only; it did not implement this three-component objective, the explicit minimal-adaptation term, or full-cycle retention optimization.

### Training protocol

Train on neighborhoods actually produced by NN-kNN, not on arbitrary case pairs disconnected from deployed retrieval:

1. train or load the retrieval model and retain policy for the current condition;
2. generate leave-one-out training neighborhoods so a training query cannot retrieve itself;
3. record `a_qi`, `z_bar_q`, `p0_q`, `Delta_z_q`, `r*_q`, the nominal-coverage declaration, and `Delta_u_q` only when the explicit channel is enabled;
4. freeze retrieval and train the classification adapter first;
5. evaluate retrieval-only and adapted outputs on untouched validation data; and
6. only after the adapter demonstrates a genuine retrieval-conditioned benefit, optionally fine-tune retrieval and adaptation jointly with a smaller adapter/retrieval learning rate and explicit pre/post diagnostics.

Regenerate or update adapter training examples when maintenance materially changes the active case base. Keep test data completely outside retrieval, adapter training, and retain decisions. A separately labeled maintenance stream is allowed, but it is not an untouched evaluation set.

### Required outputs and diagnostics

For every evaluated query, make available:

- stable retrieved case IDs and activations `a_qi`;
- retrieval-only class mass `p0_q` and prediction;
- neighborhood representation difference `Delta_z_q`;
- the nominal-coverage declaration and, when enabled, grouped nominal-attribute difference `Delta_u_q` with its field/category mapping;
- predicted residual `r_hat_q`;
- adapted class scores/probabilities and prediction;
- whether adaptation flipped the decision and whether the flip was correct;
- pre- versus post-adaptation loss, accuracy, and calibration; and
- residual magnitude and the fraction of performance attributable to adaptation rather than retrieval.

All trustworthiness and retain statistics must use the retrieval-only `p0_q` or explicit per-case counterfactuals. Do not use the adapted output to judge whether cases were good, because the adapter could conceal poor retrieval.

### Classification-adapter baselines and ablations

At minimum compare:

1. retrieval-only NN-kNN (`p0_q`);
2. the 2022-style single-case or per-case nominal-difference adapter as a historical ablation where feasible;
3. the proposed aggregate retrieved-label-conditioned nominal-residual adapter using the representation-appropriate input rule;
4. on tasks where `Delta_z_q` excludes nominal information, the same adapter without `Delta_u_q`, isolating the value of the explicit nominal-attribute channel;
5. the aggregate logit-residual variant;
6. frozen retrieval followed by adapter training versus controlled joint fine-tuning; and
7. retain-policy conditions at the same `K`, so better adaptation is not confused with better case selection.

A direct neural classification head may be included as a capacity reference, but it is not an adaptation method and must be reported separately.

### Required classification-adapter tests

- With one retrieved case of the correct class, the nominal target residual is the all-zero vector.
- With one retrieved case of class `j` and target class `k`, the residual has `-1` at `j`, `+1` at `k`, and zero elsewhere under the new plus-sign convention.
- For an aggregate neighborhood, every target residual sums to zero within numerical tolerance.
- A disabled adapter returns the current retrieval-only classification output exactly.
- The aggregate adapter input always contains `Delta_z_q` and `p0_q`; it contains grouped `Delta_u_q` only when the coverage declaration says nominal fields are absent from `Delta_z_q`; it never contains `z_i`, `z_bar_q`, the raw query, raw nominal values, or raw retrieved cases.
- When `Delta_u_q` is enabled, each nominal field's `Delta_u_qj` has the expected one-hot-group width and sums to zero within numerical tolerance.
- When `Delta_u_q` is disabled because nominal features are jointly extracted, no explicit nominal-difference tensor is constructed or concatenated.
- Nominal coverage metadata and, when applicable, category order and unknown/missing handling remain identical after checkpoint reload.
- Leave-one-out training prevents self-retrieval.
- Class-index permutation produces the corresponding permutation of inputs, residuals, and outputs.
- Pre-adaptation trustworthiness statistics are unchanged when the adapter is enabled.
- Frozen-retrieval training changes adapter parameters without changing retrieval parameters.
- Checkpoint save/reload preserves adapter mode, class ordering, dimensionality, and calibration configuration.

## Required identity and bookkeeping model

Active tensor position is not a stable case identity because compaction changes positions. Every case store participating in T1 should expose a stable `case_id` that survives compaction and is used to join:

- case content or source index;
- stored label, solution, action, or value;
- insertion step/epoch and last-used step;
- current and initial case bias;
- glocal or case-specific parameters;
- provenance accumulators;
- protection and cohort metadata;
- archive state; and
- maintenance events.

Suggested implementation components:

```python
@dataclass
class CaseStatistics:
    case_id: int
    retrieval_count: float
    activation_mass: float
    correct_support: float
    incorrect_support: float
    initial_bias: float
    current_bias: float
    insertion_step: int
    last_retrieved_step: int
    cohort_id: str | int | None
    protected: bool


class CaseStatisticsStore:
    def observe(...): ...
    def snapshot(...): ...
    def compact_by_case_ids(...): ...


class CaseMaintenancePolicy:
    def score(...): ...
    def select_keep_case_ids(...): ...
    def explain_decision(...): ...
```

Names are suggestions, not required API. The requirements are stable identity, a single policy contract, deterministic decisions under a fixed seed/tie rule, and complete logging.

## PI-approved trustworthiness formulation

### Classification provenance components

On a labeled training-audit or maintenance set `D`, for case `i` with normalized activation `a_i(x)`, stored class `c_i`, and reference class `y_x`, accumulate:

```text
R_i = sum over x in D of 1[i is retrieved for x]
A_i = sum over x in D of a_i(x)
C_i = sum over x in D of a_i(x) * 1[c_i = y_x]
H_i = sum over x in D of a_i(x) * 1[c_i != y_x]
```

Interpretation:

- `R_i`: retrieval exposure;
- `A_i`: total activation or contribution mass;
- `C_i`: activation-weighted correct-class support; and
- `H_i`: activation-weighted incorrect-class support.

When all audited cases have one class label and all activation is assigned to a stored class, `A_i = C_i + H_i`. Keep `R_i` separately because it distinguishes frequent weak retrieval from rare strong retrieval.

### Smoothed provenance-quality score

```text
Q_i = (C_i + s) / (C_i + H_i + 2s), with s > 0
```

- `Q_i` near `1`: observed activation predominantly supports correct classes.
- `Q_i` near `0`: observed activation predominantly supports incorrect classes.
- `Q_i` near `0.5`: evidence is balanced or insufficient.

The smoothing constant `s` prevents one favorable retrieval from producing perfect confidence. Treat `s` as a configuration and sensitivity-analysis variable. Do not interpret `Q_i` without a minimum `R_i` and/or `A_i` evidence requirement.

### Normalized case-bias score

The current default score is:

```text
z_i(x) = b_i - d_i(x)
```

Convert final bias `b_i`, and optionally its change from initialization, to a cohort-comparable value `B_i` in `[0,1]`, where lower values indicate learned disutility. Start with a within-class or otherwise matched percentile transform. A median/MAD transformation is a comparison option.

Do not normalize all raw biases together without checking class/action/cohort effects. Bias scale depends on initialization, temperature, regularization, score mode, training duration, and frequency.

### Combined trustworthiness score

The PI-approved primary combination is the weighted geometric mean:

```text
T_i = Q_i^alpha * B_i^(1 - alpha), with 0 < alpha < 1
```

For numerical stability:

```text
log(T_i) = alpha * log(clip(Q_i, eps, 1))
         + (1 - alpha) * log(clip(B_i, eps, 1))
```

Keep `Q_i`, `B_i`, `R_i`, `A_i`, `C_i`, and `H_i` in outputs. Do not expose only `T_i`.

### Important naming rule

`T_i` is a **trustworthiness score**, not the complete retain score. Selecting the best active case set also requires:

- observed utility/exposure from `R_i` and `A_i`;
- redundancy relative to other retained cases;
- class, action, subgroup, temporal, or domain coverage;
- protection of rare, boundary, or designated cases; and
- the fixed capacity `K`.

A low `T_i` with insufficient exposure is uncertainty, not proof that the case is bad. A low-use case may still provide unique coverage.

## Contribution semantics beyond classification

The PI-approved `C_i/H_i` formula is exact for class-labeled audit data. Regression and RL do not provide the same correct-class event, so they require role-specific contribution adapters. Do not silently reuse `1[c_i = y_x]` for continuous labels or value cases.

### Regression and NN-kNN critic candidate

Use the pre-adaptation retrieved prediction when evaluating retrieval/retention so NN-CDH or another downstream network cannot hide poor cases. The most faithful but more expensive audit is counterfactual removal:

```text
Delta_i(x) = Loss(f_without_i(x), y_x) - Loss(f_with_i(x), y_x)
C_i = sum over x of max(Delta_i(x), 0)
H_i = sum over x of max(-Delta_i(x), 0)
```

After masking case `i`, renormalize the remaining activation. Positive `Delta_i` means the case lowered loss; negative `Delta_i` means it increased loss.

A cheaper directional approximation for scalar prediction is a candidate ablation, not yet PI-approved:

```text
g_i(x) = a_i(x) * (y_i - y_hat_pre) * (y_x - y_hat_pre)
C_i = sum over x of max(g_i(x), 0)
H_i = sum over x of max(-g_i(x), 0)
```

This tests whether the case pulls the pre-adaptation estimate in the target's direction, but it does not measure overshoot as faithfully as counterfactual loss.

### RL actor candidate

The correct notion of helpful/harmful actor contribution remains `[NEEDS PI/COLLABORATOR DECISION]`. Candidate sources include signed GAE advantage for an executed action, policy-loss change under counterfactual case masking, and longer-horizon return effects. The first implementation should not present one of these as settled without agreement.

### RL critic candidate

Treat value prediction as regression against a clearly specified target. Candidate targets include the existing GAE value target for training diagnostics and discounted Monte Carlo return for separate audit rollouts. Preserve the repository's distinction between online critic training labels, mutable/trainable stored labels, and lagged target-critic bootstrap values.

If maintenance decisions use a rollout stream, that stream is no longer a reporting-only holdout. Keep test/evaluation rollouts untouched and label the maintenance stream separately.

## Proposed v0 retain policy

Use a constrained policy rather than selecting the top `K` cases by `T_i` alone.

### Stage 1 - protection and evidence

1. Mark protected cases required for minimum class/action/cohort coverage.
2. Mark cases with insufficient exposure; do not infer harm from their `Q_i`.
3. Calculate trustworthiness components and utility/exposure.
4. Estimate redundancy within an appropriate representation and cohort.

### Stage 2 - routine eviction eligibility

A case is an automatic routine-eviction candidate only when it is not protected and at least one of the following is supported by configured evidence:

- low observed utility plus high redundancy;
- sufficiently evidenced low trustworthiness together with low utility or high redundancy and replaceable coverage; or
- a new case provides demonstrably better utility/coverage at the same budget.

All automatic removal in v0 should mean reversible eviction from active memory. Record an inactive archive entry or sufficient restoration state. Controlled benchmark/RL experiments may also test trust-only eviction as an ablation because an oracle or complete event record permits recovery; in a consequential domain, suspected poisoning or harmful knowledge remains review-only under the previously approved action-authority policy.

### Stage 3 - constrained selection under `K`

Preserve the protected set first, then fill remaining capacity using trustworthiness, utility, diversity/redundancy, and coverage. The exact scalarization is intentionally not fixed here. Implement the policy so these alternatives can be compared:

1. bias-only, matching the current pruning concept;
2. provenance quality only;
3. geometric trustworthiness `T_i` only after evidence filtering;
4. utility plus redundancy;
5. trustworthiness plus utility;
6. trustworthiness plus utility plus coverage/diversity; and
7. uniform/random or stratified selection.

Use the same `K`, insertion stream, seeds, and training budget across variants. Tie-breaking must be deterministic and logged.

## Suggested configuration surface

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

classification_adapter_enabled = false
classification_adapter_architecture = "aggregate_label_conditioned"
classification_adapter_input = "auto_by_representation_coverage"
classification_adapter_nominal_delta_mode = "auto"
classification_representation_covers_nominal_inputs = ...
classification_adapter_output_mode = "nominal_residual_scores"
classification_adapter_freeze_retrieval_first = true
classification_adapter_lambda_diff = ...
classification_adapter_lambda_cls = ...
classification_adapter_lambda_adapt_cost = 0.0  # enable only in later synchronization phase
classification_adapter_probability_mode = ...

component_sync_enabled = false
component_sync_schedule = "alternating"
component_sync_problem_distance_source = "shared_nnknn_learned_distance_interface"
component_sync_problem_distance_snapshot_id = ...
component_sync_lambda_pre = ...
component_sync_lambda_post = ...
component_sync_lambda_near = ...
component_sync_lambda_delta = ...
component_sync_lambda_small = ...
component_sync_free_correction_enabled = true
component_sync_free_correction_neighbor_metric = "frozen_nnknn_learned_geometric_distance"
component_sync_free_correction_local_selector = "positive_case_bias_activation_region"
component_sync_free_correction_calibration_stage = "after_core_training"
component_sync_free_correction_bias_source = "frozen_trained_per_case_bias"
component_sync_free_correction_trained_bias_transform = "identity"  # direct frozen trained-bias snapshot; PI-confirmed primary
component_sync_free_correction_trained_bias_transform_ablations = ["cohort_normalized", "clipped", "shrunk"]  # enable when diagnostics justify
component_sync_free_correction_bias_initializer = "mean_kth_neighbor_learned_distance"  # PI-confirmed primary
component_sync_free_correction_bias_initializer_knn_k = 5  # PI-confirmed starting value
component_sync_free_correction_bias_initializer_ablation = "learned_distance_percentile"
component_sync_free_correction_label_metric_classification = "one_hot_euclidean"
component_sync_free_correction_label_metric_classification_ablation = "probability_vector_distance"
component_sync_free_correction_label_metric_other_tasks = ...
component_sync_free_correction_estimator = "mean_label_distance_within_bias_radius"
component_sync_free_correction_recompute = "declared_outer_checkpoints_only"
component_sync_maintenance_policy = "checkpoint_discrete"
```

Ellipses are unresolved values, not recommended defaults. Keep the classification adapter off by default until its tests and diagnostic gate pass. Preserve backward compatibility so the current pruning policy and retrieval-only classification remain selectable.

## Maintenance event log

Extend the existing maintenance reporting rather than creating incompatible outputs for each workflow. Each event should include at least:

- run ID, seed, task, model role, step/epoch, and policy version;
- stable case ID and active slot before/after action;
- action: insert, keep, archive, restore, replace, or protect;
- reason code;
- `Q_i`, `B_i`, `T_i`, `R_i`, `A_i`, `C_i`, and `H_i`;
- redundancy and coverage indicators;
- relevant label/action/value/cohort metadata;
- case capacity before/after;
- protection status; and
- deterministic tie-break information.

Suggested artifacts:

- `case_statistics.csv` or a lossless columnar equivalent;
- extended `case_maintenance.csv`;
- `case_selection_summary.json`; and
- archived/restorable case state in the checkpoint.

Do not log private or domain-sensitive raw case content by default. Use stable IDs and controlled references when data governance requires it.

## Implementation sequence

### Phase 0 - reproduce and instrument

- Record starting Git commit and environment.
- Run existing import, supervised, and NN-kNN-RL smoke checks.
- Add stable IDs and statistics without changing selection behavior.
- Verify that enabling statistics alone does not change predictions, gradients, random-number streams, or case insertion.
- Add per-case activation/contribution export on small synthetic tests.

### Phase 1 - supervised retain prototype

- Implement classification `R_i/A_i/C_i/H_i`, `Q_i`, normalized `B_i`, and `T_i`.
- Implement protected coverage, redundancy checks, reversible eviction, and deterministic logging.
- Compare on small synthetic and completed classification-paper tasks before large image runs.
- Add regression counterfactual contribution on selected completed regression-paper tasks, using pre-adaptation outputs for retrieval-quality analysis.

### Phase 2 - classification reuse extension

- Implement the aggregate nominal-residual design above with `[Delta_z_q, p0_q]` when the representation covers nominal fields and `[Delta_z_q, Delta_u_q, p0_q]` otherwise, without changing the existing retrieval-only path.
- Build adapter examples from leave-one-out NN-kNN neighborhoods and train with retrieval frozen first.
- Use positive, configurable `lambda_diff` and `lambda_cls`; compare `L_diff`-only, `L_cls`-only, and the PI-confirmed combined loss.
- Compare retrieval-only, historical/per-case where feasible, aggregate nominal-residual, and logit-residual variants.
- Report decision flips and pre/post metrics; verify that retain statistics remain pre-adaptation.
- Keep this phase separately switchable so retain experiments can run without the adapter and adapter experiments can use identical retained case sets.

### Phase 3 - RL integration

- Apply the common statistics and maintenance interface separately to the NN-kNN actor and critic case stores.
- Preserve separate actor and critic case IDs, statistics, capacities, and logs even when they share a representation.
- Run actor-only, critic-only, and combined NN-kNN variants as appropriate.
- Begin with current CartPole and the collaborator's active RL tasks after their exact names/configurations are supplied.
- Compare current bias pruning, no learned maintenance, and proposed retain variants at matched case budgets and environment steps.

### Phase 4 - component harmonization

- Add separately weighted retrieval, adaptation, proximity, and minimal-correction terms only after the retrieval, reuse, and retain components pass their independent gates.
- Use staged or alternating updates so one high-capacity component cannot silently carry a weak component.
- Add maintenance as a safe-checkpoint selection step, rebuild neighborhoods after selection changes, and log every component loss before and after each stage update.
- Compare independent training, retrieval-adaptation harmonization, and retrieval-adaptation-maintenance harmonization.

### Phase 5 - MCB and interaction tests

- Add MCB/no-MCB as a controlled factor only after the retain policy is independently testable.
- Measure representation drift, neighborhood churn, selection churn, and task learning.
- Test whether greater representation stability improves the reliability of accumulated case statistics or makes necessary adaptation too slow.

### Phase 6 - broader confirmation

- Scale to selected image/text or harder RL settings only after the mechanism passes diagnostic gates.
- Keep revise disabled until a later implementation decision explicitly enables it.

## Evaluation matrix

### Completed-paper testbeds

Use completed-paper tasks as controlled foundations and compatibility checks:

- maintained IJCAI-2025 classification tasks supported by the current repository, with retrieval-only versus proposed aggregate classification adaptation reported separately;
- maintained IJCAI-2026 regression tasks, including pre/post NN-CDH reporting where adaptation is enabled;
- synthetic tasks with known feature relevance, redundancy, corruptions, rare cases, and shifted subdomains; and
- MCB paper conditions only when the relevant code/data branch is available, clearly distinguishing the under-review manuscript from completed papers.

Do not claim that a benchmark exactly reproduces a published result unless the implementation, split, preprocessing, seed, and metric match the paper protocol.

### RL testbeds

- `CartPole-v1`, currently supported by the inspected repository;
- additional active RL tasks being developed by the collaborator: `[NEEDS INPUT: exact environment IDs and branch/commit]`;
- later Gymnasium/Atari tasks only after feasibility, encoder, compute, and case-budget review.

RL comparisons should distinguish:

- NN-kNN actor with MLP critic;
- MLP actor with NN-kNN critic;
- NN-kNN actor with NN-kNN critic and separate memories;
- MLP/MLP baseline;
- NEC and DQN where their current implementations are valid comparison points; and
- maintenance policy ablations at matched `K` and environment-step budgets.

Use `gold` or disable evaluation-based early stopping for strict fixed-step comparisons. Report configured and actual timesteps, best and final checkpoints, training efficiency, multiple seeds, case counts, case churn, and whether the task success threshold was reached. Smoke tests establish plumbing only.

### Primary outcome families

1. **Task learning:** accuracy, RMSE, return, success rate, calibration, and sample efficiency as appropriate.
2. **Retrieval quality:** label/task alignment, counterfactual contribution, relevant-neighbor quality, and retrieval stability.
3. **Reuse quality:** improvement over retrieval-only, correct and harmful decision flips, residual magnitude, calibration, and evidence that the adapter remains retrieval-conditioned.
4. **Retention quality:** performance at matched `K`, rare/shifted competence, redundancy, harmful-case prevalence, and selection regret where an oracle is available.
5. **Efficiency:** active/archive memory, retrieval latency, training time, and maintenance overhead.
6. **Stability:** representation drift, neighborhood churn, selected-case churn, and seed sensitivity.
7. **Auditability:** complete statistics and reason codes for every case-maintenance action.
8. **Synchronization:** every component loss, pre/post-adaptation task loss, correction magnitude, query-case proximity, and the extent to which one component compensates for another.

## Required baselines and ablations

At minimum:

- full memory where computationally feasible;
- existing task-agnostic downsampling or random selection;
- stratified random selection;
- current bias-only quantile/threshold pruning;
- provenance-only `Q_i`;
- bias-only normalized `B_i`;
- geometric `T_i`;
- utility/redundancy selection;
- combined trustworthiness, utility, and coverage policy; and
- no-MCB versus MCB where available.

For classification reuse, also include retrieval-only, historical/per-case where feasible, the representation-appropriate aggregate nominal-residual, and aggregate logit-residual conditions. Where `Delta_z_q` excludes nominal fields, compare the aggregate nominal-residual with and without `Delta_u_q`. Do not require a redundant `Delta_u_q` condition when nominal fields are already jointly extracted. Keep case base, `K`, retrieval model, split, and training budget matched.

For component synchronization, compare independent component training, retrieval-adaptation alternating optimization, and the proposed retrieval-adaptation-maintenance schedule. Ablate `L_pre`, `L_near`, `L_delta`, and `L_small` individually; never report only the weighted total. For `L_small`, compare no magnitude penalty, penalty from zero, and the PI-confirmed trained-case-bias-region free-correction radius. Use the direct frozen trained-bias snapshot as primary. Add normalized, clipped, or shrunk trained-bias transformations only as fallback/ablation conditions if direct-snapshot diagnostics show instability, incomparability, empty regions, or excessively broad regions. Retain the superseded initial-bias-region and exact-kth label-pair thresholds and robust versus arithmetic label-distance aggregation as comparisons. Start the bias initializer at `k=5`; treat `k` as a warm-start sensitivity parameter rather than a central contribution, expanding its sweep only if conclusions materially depend on it.

Use arithmetic combination of `Q_i` and `B_i` as an ablation, not the primary score.

## Unit and integration tests

Required small tests include:

- zero exposure yields `Q_i = 0.5` under symmetric smoothing;
- increasing correct activation raises `Q_i` and increasing incorrect activation lowers it;
- geometric-score computation matches direct and log-space forms;
- compaction preserves statistics, labels, biases, glocal weights, optimizer alignment, and stable IDs;
- protected cases and minimum class/action coverage cannot be violated;
- archive then restore reproduces the case state;
- deterministic inputs and seed produce identical keep/evict decisions;
- statistics-only mode does not change training behavior;
- actor and critic statistics never share IDs or labels accidentally;
- target-critic alignment remains valid after critic-memory maintenance;
- evaluation/test data never influence training-time retain decisions; and
- checkpoint save/reload preserves active cases, archived cases, statistics, and policy configuration.
- classification nominal-residual targets, aggregate inputs, disabled-path identity, class permutation, and pre-adaptation-statistic invariance pass the tests specified above.
- free-correction calibration excludes self-pairs and validation/test examples, reconstructs the positive case-bias activation region from the saved bias and learned-distance snapshot, reproduces the logged `tau_task`, assigns zero `L_small` at and below the threshold, and assigns positive increasing cost above it.
- a degenerate zero or near-zero threshold remains numerically stable because penalty scaling uses nonzero `s_task`, not division by `tau_task`.

## Experimental gates

Do not advance solely because code runs. Suggested gates are:

1. **Instrumentation gate:** exact statistic and identity tests pass without changing baseline behavior.
2. **Supervised mechanism gate:** at matched `K`, at least one proposed selection policy consistently improves the intended retention/retrieval trade-off over random/downsampling and current bias-only pruning across multiple seeds; exact margins remain `[NEEDS INPUT]`.
3. **Classification reuse gate:** the aggregate adapter improves at least one prespecified pre/post metric without unacceptable calibration, harmful-flip, or leakage behavior, and results are reported separately from case-selection gains; exact thresholds remain `[NEEDS INPUT]`.
4. **RL mechanism gate:** improved selection changes RL learning or stability relative to matched current maintenance; exact tasks and thresholds remain `[NEEDS INPUT]`.
5. **Synchronization gate:** harmonization improves a prespecified system outcome without degrading retrieval locality or increasing adaptation magnitude beyond the declared bound, and no single component hides another component's failure.
6. **MCB interaction gate:** determine whether MCB materially improves statistic reliability or selected-case stability, even if it does not improve return/accuracy.
7. **Failure-analysis gate:** if performance does not improve, identify whether retrieval scoring, case admission, case labels/solutions, representation drift, reuse, maintenance, or RL optimization is the limiting factor.

## Deliverables to return to the PI

- implementation branch/commit and concise change summary;
- updated repository `HANDOFF.md` and relevant operating documentation;
- configuration schema and defaults;
- unit and smoke-test results;
- small supervised diagnostic results;
- classification adapter implementation/configuration and retrieval-only versus adapted results;
- matched-budget RL results for the active tasks;
- case statistics and maintenance event artifacts;
- a short failure-analysis memo; and
- a list of unresolved scientific and engineering decisions.

## Open implementation decisions

These should be answered one at a time rather than assumed:

1. What signal defines helpful versus harmful contribution for NN-kNN actor cases?
2. Should the RL critic's provenance use GAE targets, discounted Monte Carlo audit returns, or both for distinct purposes?
3. Is maintenance computed continuously, at rollout/batch boundaries, at fixed evaluation checkpoints, or in a two-timescale combination?
4. What are the initial `K`, maintenance frequency, smoothing `s`, trust weight `alpha`, evidence minimums, and coverage floors?
5. Which representation and threshold define redundancy?
6. What exact completed-paper datasets form the first fast and full suites?
7. Which current RL environments beyond `CartPole-v1` must be included?
8. Is MCB already integrated into the collaborator's active branch, and if so, what commit and configuration enable it?
9. What initial positive ratio should be used for `lambda_diff` and `lambda_cls`, and what calibration temperature, if any, is required?
10. Which `s_task` should accompany the confirmed classification metrics, and which `d_y` definitions should be used for regression, the RL actor, and the RL critic?
11. Should the first maintenance update remain a discrete checkpoint policy, or should a differentiable selector be tested as a later ablation?

## Current recommended default direction

Until the open decisions are resolved:

- preserve the current repo behavior as the baseline;
- introduce statistics in a behavior-neutral mode first;
- implement one shared maintenance-policy interface;
- use stable IDs and reversible archives;
- begin scoring with classification, where the PI-approved provenance formula is directly defined;
- implement the aggregate classification adapter with `[Delta_z_q, p0_q]` when `Delta_z_q` already covers nominal inputs and `[Delta_z_q, Delta_u_q, p0_q]` otherwise; use the generalized nominal residual as the primary output target and keep the logit-residual form as an ablation;
- train the first adapter with the PI-confirmed combined residual-MSE plus classification-cross-entropy loss, while retaining single-term ablations;
- freeze retrieval during the first adapter-training pass and report retrieval-only versus adapted behavior separately;
- use counterfactual loss contribution as the reference method for regression and value prediction on small audit subsets;
- treat RL actor contribution semantics as unresolved;
- run maintenance only at safe batch/rollout boundaries, never while a rollout representation is meant to remain frozen;
- introduce the broader retrieval-adaptation-maintenance synchronization objective only after the separate components work, using alternating updates and separately reported terms;
- when that phase enables `L_small`, first train the NN-kNN core and live per-case biases, then freeze the trained bias and learned-distance snapshots, use the direct trained-bias activation regions to select training-only local case pairs, and average the corresponding label distances; the exact snapshot-selection rule is deferred to implementation, derived trained-bias transformations are fallback/ablation conditions, and initial-bias, zero-threshold, and exact-kth-pair variants remain comparisons; and
- keep revise disabled.
