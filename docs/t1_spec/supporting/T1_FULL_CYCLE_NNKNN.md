# T1 full-cycle, general-purpose NN-kNN

## Status and working name

The PI requested the full-cycle foundation on 2026-09-08 after a collaborator meeting about the current NN-kNN reinforcement-learning work, initially labeled it T0, and merged it with the former T1 module-generalization target on 2026-09-09. On 2026-09-12, the PI narrowed T1 so portability and software packaging support the scientific aims rather than forming a separate aim. The term **full-cycle neural CBR** describes the architecture, while consequential revision decisions remain human-guided rather than falsely described as fully neural.

T1 has two sequenced scientific aims:

1. **T1.1 - Coordinate the technical core:** improve and coordinate NN-kNN retrieval, bounded neural reuse, MCB-style representation stability, quality-aware retention, and bounded-memory case selection. Human revision remains disabled during the first implementation so core case-selection failures can be isolated.
2. **T1.2 - Complete and validate the cycle:** add human-guided revision, causal correction evaluation, and the bounded human-grounded UI study after the technical core is established.

The aims share one scientific question: whether an NN-kNN system can preserve task performance while making learned behavior stable, inspectable, correctable, and maintainable under limited memory. A reusable module and bounded portability tests are enabling deliverables, not a third scientific aim. The intellectual contribution is the coordination and evaluation of the full cycle and its failure boundaries.

## Motivation and claim boundary

The current NN-kNN-RL implementation establishes a technically concrete actor-critic platform, but its documented smoke results establish plumbing rather than competitive performance. The PI and collaborators now suspect that progress is limited by unresolved properties of NN-kNN itself, even after the separate MCB stability work. This is a research hypothesis motivated by current experience; the specific failure modes, experiments, and causal evidence from the meeting have not yet been supplied.

The PI reports that the current collaboration with Sen He began in T2 and now proceeds on T1 and T2 concurrently. NN-kNN appeared insufficiently accurate and efficient in the RL work, prompting a return to the T1 core rather than continued task-level tuning alone. T1 improvements are intended to feed directly back into T2 experiments, whose results will in turn expose additional core limitations. This motivates an iterative T1-T2 feedback loop; it does not yet establish that case selection is the sole cause of the RL limitations or that the proposed T1 mechanisms will resolve them.

T1 therefore asks a foundational question before expanding to reinforcement learning:

> Can NN-kNN become the trainable core of a complete case-based reasoning cycle whose retrieval, bounded reuse, MCB-supported representation stability, human revision, and quality-aware retention improve one another while preserving inspectable links between cases and behavior?

The completed classification and regression comparisons cited as preliminary evidence use the older NN-kNN architecture and **do not include MCB**. They establish a legacy feasibility and performance baseline only. T1 must separately determine whether MCB and the integrated full-cycle mechanisms preserve or improve task performance, stability, case quality, bounded-memory efficiency, and human correctability.

The novelty cannot be the classical retrieve-reuse-revise-retain vocabulary by itself. The proposed contribution is an integrated, testable architecture that combines differentiable retrieval/adaptation with constrained case-set maintenance, exposes human intervention points, and characterizes interactions and failure conditions across the stages.

## Immediate implementation focus confirmed by the PI

The first implementation is narrower than the eventual full cycle. The PI and collaborator group believe the immediate bottleneck is that NN-kNN does not consistently retain and use the best cases. Accordingly:

- prioritize retrieve instrumentation and retain/selection mechanisms;
- preserve the currently implemented regression reuse behavior and add a bounded classification NN-CDH extension that combines one-hot/one-cold nominal differences with the current aggregate, retrieved-label-conditioned design; do not require one universal adapter;
- disable the revise stage for now; and
- evaluate case selection both on completed-paper classification/regression testbeds and on the active RL tasks being developed by the collaborator.

The suspected case-selection bottleneck remains a hypothesis. The evaluation must distinguish it from representation learning, case admission, stored labels/solutions, reuse, exploration, critic quality, optimization, and other RL-specific causes.

## Proposed four-stage architecture

### 1. Retrieve - NN-kNN case activation

NN-kNN retrieves cases using learned representations, feature-distance weights, per-case bias, and normalized case activation. The retrieved cases and their activation/contribution weights form both the prediction substrate and the observable trace passed to later stages.

Existing foundation: maintained classification and regression implementations already calculate case distances, bias-minus-distance scores, and normalized activation. Proposed T1 work must improve retrieval quality and stability where current behavior is inadequate, rather than merely restating that NN-kNN retrieves cases.

### 2. Reuse - neural adaptation of retrieved solutions

The regression paper supplies a concrete reuse instance: after aggregating a retrieved neighborhood, NN-CDH predicts a correction from the query-to-neighborhood difference and the retrieved estimate. The maintained code implements aggregate and per-case adapter variants. Its newer aggregate form uses `[z_q - z_bar_q, y0_q]`, not the retrieved context embedding, so the adapter cannot trivially reconstruct `z_q` from a difference-plus-context pair and bypass adaptation.

The PI-supplied ICCBR-2022 paper supplies a complementary classification idea: one-hot encode nominal attributes and class labels, then represent a nominal change by subtracting the encodings. T1 will adapt this idea to the new aggregate design. With normalized case activations `a_qi`, define:

```text
z_bar_q = sum_i a_qi * z_i
p0_q = sum_i a_qi * e(y_i)
r*_q = e(y_q) - p0_q

h_q = [z_q - z_bar_q, p0_q]
      if z_q - z_bar_q already represents nominal inputs

h_q = [z_q - z_bar_q,
       concat_j(e(u_qj) - sum_i a_qi * e(u_ij)),
       p0_q]
      otherwise

r_hat_q = tanh(g_psi(h_q))
s_q = p0_q + r_hat_q
```

For one retrieved case, `r*_q` is an all-zero or one-hot/one-cold class-change vector; for a neighborhood, it is a generalized one-hot/weighted-cold residual. The explicit grouped nominal-attribute difference preserves the old paper's idea only when the shared representation difference does not already include the nominal fields. If nominal features are extracted together with the other features, that explicit channel is omitted as redundant. The adapter predicts from differences and retrieved label mass, not from the raw query, raw cases, or retrieved embedding. Retain/trustworthiness diagnostics use the pre-adaptation `p0_q` so a downstream adapter cannot conceal poor cases.

Existing foundation: NN-CDH reuse is implemented for regression, and the older paper demonstrates a nominal-difference classification architecture, but the proposed aggregate classification adapter above is not yet implemented or validated. T1 will use a common reuse contract with task-specific adapter semantics. RL actions or values and later LLM/agent memories remain separate future adapter designs rather than being inferred from supervised classification.

### 3. Revise - human-guided case and weight correction

A person inspects retrieved cases, provenance, activations, feature weights, predicted behavior, and available trust/utility evidence. The person may correct case content or a stored solution/label, adjust case- or feature-level weighting, quarantine a case, or introduce a new expert case. The system then measures the immediate behavioral effect and any effect after controlled adaptation, including correct and harmful prediction flips.

Existing foundation: the regression paper provides a controlled synthetic feature-weight-tuning case study showing that an externally supplied weight edit can be retained during further training. This does not yet establish a complete revision interface, reliable case editing, useful domain-expert interaction, or predictable correction in RL or other settings.

**Immediate implementation status:** disabled. Preserve stable IDs, maintenance logs, and reversible case state so revise can be added later, but do not require a human interface or correction experiment in the first T1 implementation.

### 4. Retain - quality- and coverage-aware case-base maintenance

The retain stage decides which new experiences become cases and which existing cases remain active under the fixed budget `K`. The working design combines exposure, activation-weighted correct and incorrect support, learned case bias, redundancy, diversity, and rare/domain coverage. Routine eviction of clearly redundant or low-utility cases is automatic, logged, safeguarded, and reversible; suspected harmful, poisoned, rare, domain-critical, ambiguous, or otherwise consequential cases require qualified human review.

For class-labeled audit data, the PI-approved provenance components are:

```text
R_i = sum_x 1[i is retrieved for x]
A_i = sum_x a_i(x)
C_i = sum_x a_i(x) * 1[c_i = y_x]
H_i = sum_x a_i(x) * 1[c_i != y_x]
```

The smoothed provenance-quality score is:

```text
Q_i = (C_i + s) / (C_i + H_i + 2s), with s > 0
```

After mapping learned case bias to a cohort-comparable value `B_i` in `[0,1]`, the primary combined trustworthiness score is:

```text
T_i = Q_i^alpha * B_i^(1 - alpha), with 0 < alpha < 1
```

`T_i` is not the complete retain score. The selection policy must also consider exposure/utility from `R_i` and `A_i`, redundancy, protected coverage, and budget `K`. Low exposure means insufficient evidence, not proof of a bad case. Continuous regression labels, RL actor cases, RL critic value cases, and later LLM/agent memories require task-specific helpful/harmful contribution definitions rather than silently applying the classification indicator. In T3, same-label support is replaced by need-conditioned downstream utility; see [T3_NEED_CONDITIONED_RETRIEVAL.md](T3_NEED_CONDITIONED_RETRIEVAL.md).

Existing foundation: the current RL code includes bounded actor/critic case memories and bias-based pruning with protected case IDs and minimum action coverage. The proposed provenance score, competence/coverage-aware selection, cross-task retention policy, reversible archive, and human-reviewed consequential intervention remain new work.

## Cross-cutting role of Momentum Case Base

MCB is a stability mechanism supporting the four stages, not a replacement for them or a fifth CBR stage. A stable memory representation may make retrieval traces, human edits, and retention decisions more reproducible, but current MCB evidence is limited to the supplied CUB-200-2011 study. T1 will test whether MCB helps the integrated cycle and where stability impedes necessary adaptation.

## Component synchronization and eventual system objective

T1 will treat coordination among CBR components as a scientific problem, not merely connect separately trained modules. The mature objective will preserve distinct, inspectable terms:

```text
L_pre   = task loss before adaptation
L_post  = task loss after adaptation
L_near  = explanation-relevant query-to-retrieved-case proximity
L_delta = predicted-versus-target adaptation loss
L_small = scale-normalized magnitude of the applied adaptation

L_R = w_pre  * L_pre  + w_post * L_post + w_near  * L_near
L_A = v_post * L_post + v_delta * L_delta + v_small * L_small
```

The retrieval objective `L_R` favors cases that are useful before adaptation, easily adaptable, and nearby under a declared explanation metric. The adaptation objective `L_A` favors accurate final behavior while learning the intended correction and discouraging unnecessarily drastic changes. `L_post` prevents the trivial solution of never adapting; `L_pre` and `L_near` prevent a powerful adapter from concealing poor or distant retrieval. Nearness is an auditable locality criterion or explanation proxy, not proof of human understandability.

The PI confirmed that minimal adaptation should have a small task-specific free-correction radius rather than be penalized from zero. The PI then superseded the exact fifth-neighbor label-pair estimator with a case-bias-inspired design and clarified that calibration occurs only after the NN-kNN core and its live per-case biases are trained. At a post-core-training checkpoint `t_star`, freeze the trained per-case biases and corresponding learned NN-kNN metric, use them to define the learned positive-activation region, and then measure ordinary label variation inside it. The exact checkpoint-selection rule is deferred to the implementation protocol and need not appear in the proposal narrative:

```text
b_i^star = stop_gradient(b_i^live(t_star))

d_theta^star = stop_gradient(d_theta(t_star))

P_bias = {(i, j): i != j and b_i^star - d_theta^star(x_i, x_j) >= 0}

tau_task = mean_{(i, j) in P_bias} d_y(y_i, y_j)

L_small = mean_q [
    (max(0, correction_distance_q - tau_task) / (s_task + epsilon))^2
]
```

`d_theta^star` includes the trained representation, feature-distance weights, and other learned geometric components whenever available. The trained bias and metric snapshots are frozen within the subsequent calibration/adapter phase, and only training pairs are used. Bias is not numerically substituted for label-space correction because the units differ; it selects the learned local region whose observed label variation defines the threshold. Corrections at or below `tau_task` receive no magnitude penalty but must still improve `L_post`.

For classification, one-hot Euclidean `d_y` remains primary: `tau_task` is `sqrt(2)` times the class-disagreement rate inside the trained-bias activation regions, with probability-vector distance as an ablation. The superseded exact-kth label-pair estimator remains a comparison condition. The initial common case bias still starts from NN-kNN's data-derived mean kth-neighbor-distance policy with `k=5`, but that value is only a warm start for training the live per-case biases. It is not the primary free-adaptation calibration source. The PI selected the direct frozen trained per-case biases as primary because they preserve the learned case-specific activation geometry without another transformation. Normalized, clipped, or shrunk trained-bias values are fallback/ablation conditions if the direct snapshot produces unstable, incomparable, empty, or excessively broad activation regions.

Retention adds a constrained case-set objective over active set `S`, combining pre/post-adaptation task loss with active-case untrustworthiness, redundancy, coverage gaps, and selection churn, subject to `|S| <= K` and protection constraints. Because selection is discrete, T1 may interleave maintenance checkpoints with alternating retrieval and adaptation updates rather than pretending the entire cycle is differentiable. Every component loss and the total will be reported separately.

The PI's ICCBR-2021 paper provides direct preliminary evidence for the two-component premise: independently trained retrieval and adaptation can become poorly coordinated, and alternating optimization can harmonize them. It also identifies retention as a future extension. It does not establish the proposed three-component retrieval-adaptation-maintenance objective, an explicit minimal-adaptation penalty, classification synchronization, or a full-cycle neural CBR system; those remain proposed T1 contributions.

## Working hypotheses and questions

**H1 - PI-confirmed central T1 hypothesis:** Explicit coordination of NN-kNN retrieval, bounded neural adaptation, MCB-style case-representation stabilization, and quality-aware retention will outperform independently trained or heuristically connected components on a joint trade-off among task quality, retrieval locality and faithfulness, adaptation magnitude, neighborhood and representation stability, memory efficiency, and robustness to changing or corrupted cases; later human revision will test correctability. Matched no-MCB and component-wise ablations will test whether MCB improves that balance or over-stabilizes the memory and suppresses necessary adaptation.

**Immediate H0a - PI-confirmed direction:** Quality-, utility-, redundancy-, and coverage-aware retention will improve NN-kNN retrieval and task learning over current bias-only pruning and task-agnostic downsampling/random selection at matched case budgets; it will approach full-case-base performance while retaining substantially fewer, higher-quality cases; and the resulting NN-kNN system will be tested for competitive task performance against strong contemporary methods on the same benchmarks and protocols. Improved selection will also improve or stabilize the active RL training if case quality is the actual bottleneck. Exact quantitative thresholds remain to be set before confirmatory experiments.

### T1 bounded-memory and modern-competitiveness standards — PI-confirmed framework

T1 will use three complementary standards rather than a single leaderboard score:

1. **Selection value at matched memory:** outperform random selection, ordinary downsampling, current bias-only pruning, and applicable established case-selection methods at the same case budget `K`.
2. **Compression without major competence loss:** approach the full-case-base NN-kNN result while retaining substantially fewer cases, reporting the complete task-quality-versus-memory curve rather than one favorable operating point.
3. **Contemporary relevance:** compare against strong, reproducible contemporary neural, retrieval, nearest-neighbor, and task-specific methods on identical benchmark splits and evaluation protocols. The modern transfer family must support both a standard predictive-performance protocol and controlled case-layer stress protocols for bounded memory, shift, poisoning or bad cases, and human correction. Where feasible, run both protocols on the same data family and control or disclose encoder, tuning, compute, parameter, latency, and memory differences.

“Modern enough” will mean statistically competitive task performance together with materially better case-level inspection, human correctability, and bounded-memory efficiency. Competitiveness will be evaluated using benchmark-appropriate uncertainty estimates and a prespecified, justified non-inferiority or practical-equivalence margin; failure to detect a significant difference will not by itself establish competitiveness. Better inspection and control will be tested through case/feature attribution faithfulness, targeted add-edit-reweight-remove interventions, successful correction of harmful behavior, limited collateral degradation, and competence retained across the task-quality-versus-memory curve. Joint success requires both the predictive-performance and case-level-capability standards. The PI confirmed this governing framework on 2026-09-12. Exact performance margins and minimum capability improvements will be chosen per benchmark after pilot evidence and fixed before confirmatory runs rather than imposed as one universal threshold.

The proposal will not claim universal state-of-the-art performance in advance. A state-of-the-art claim may be made only for a precisely identified benchmark, metric, protocol, and date when supported by the completed results.

Candidate questions:

1. When does neural reuse correct an imperfect but meaningful retrieved neighborhood, and when does it bypass or amplify retrieval error?
2. Which human revisions to cases, labels/solutions, case parameters, or feature weights produce predictable targeted changes without unacceptable collateral effects?
3. Which retention policies preserve useful, rare, boundary, and shifted-domain competence under matched case budgets while removing redundancy and limiting harmful knowledge?
4. How does MCB-style representation stabilization interact with retrieval quality, reuse learning, human edits, and retention under nonstationarity?
5. Does synchronizing retrieval, minimal adaptation, and retention reduce the tendency of one component to compensate for another while preserving or improving task performance?

## Evaluation structure

Rerun the classification and regression benchmarks from the completed pre-MCB papers as T1's compatibility and mechanism foundation, using the recoverable original protocols and documenting any benchmark that cannot be reproduced exactly. Each rerun will have two phases: first reproduce the old pre-MCB architecture as faithfully as possible; then hold the recoverable split, encoder, tuning budget, case budget, and primary metric constant while evaluating MCB and full-cycle variants. Any unavoidable difference will be disclosed rather than silently treated as matched. These reruns will compare the legacy architecture with the new coordinated components rather than treating the old reported numbers as results for the new system. Add a structured/tabular modern transfer family as the primary expansion; additional benchmarks may be added only when they strengthen a prespecified T1 claim and remain feasible, and are not yet committed. The modern family will have two linked protocols: a standard predictive-performance comparison against strong contemporary methods and controlled stress tests of bounded memory, domain shift, poisoned or otherwise harmful cases, and targeted human correction. Prefer one family that supports both protocols so this does not become two unrelated projects; if that is infeasible, pair one compact contemporary performance benchmark with one controlled mechanism testbed under a shared model and evaluation contract. These tasks provide labeled diagnostics for the case score, a controlled test of reuse, and legacy baselines against which MCB and full-cycle additions can be isolated. Active RL results may expose requirements and provide downstream feedback through T2, but competitive RL evaluation is a T2 responsibility rather than another full T1 evaluation domain. Compare in the immediate phase:

- retrieval only;
- retrieve plus reuse;
- classification retrieval-only versus aggregate nominal-residual adaptation, with a logit-residual engineering ablation;
- retrieve plus retain;
- current versus proposed retain policies at matched `K`;
- configurable combinations with bounded reuse, MCB, quality-aware retention, and—after T1.2 begins—human revision enabled or disabled, including the main effects and scientifically important interactions;
- each MCB condition paired with a matched no-MCB condition where feasible, always retaining the corresponding pre-MCB legacy result or rerun as a clearly labeled baseline; and
- matched neural, nearest-neighbor, case-based, and task-specific baselines, including strong contemporary benchmark methods selected through a documented, date-stamped review before confirmatory runs.

Implement the T1 components as explicit configuration switches and log the active combination with every run. The exact matrix remains to be selected. Use broad combination coverage on inexpensive diagnostic benchmarks and hypothesis-driven subsets on larger benchmarks so interaction effects can be identified without requiring an exhaustive combinatorial grid everywhere.

Add retrieve-plus-revise and the fully integrated four-stage cycle only after the revise stage is enabled.

Measure task performance and calibration; label- or task-aligned retrieval; explanation faithfulness through interventions; correction success and harmful collateral flips; case-base size, retrieval latency, and training cost; rare/shifted-domain retention; neighborhood and representation stability; and failures attributable to each stage or interaction.

### Tiered human-correctability validation — PI confirmed

T1 will distinguish three levels of evidence so that the proposal does not confuse mechanical intervention success with human usability:

1. **Core functional evaluation:** Use prespecified, reproducible interventions on controlled datasets to test whether adding, suppressing, removing, pinning, or reweighting cases and changing feature weights produces the intended prediction changes, improves the targeted error, and limits collateral degradation. This technical evaluation is required for T1 success and does not depend on recruiting participants.
2. **Bounded human-grounded UI study:** On a low-risk common-knowledge or other non-specialized task with objectively checkable answers or labels, participants will inspect activated cases and their contribution weights and use a visual interface to pin or suppress cases and adjust case-bias or feature-weight controls. The interface will distinguish forced inclusion from ordinary learned activation and log the complete intervention. Evaluate whether participants identify harmful cases, select effective interventions, improve task outcomes, avoid collateral errors, and complete the task efficiently. Subjective usability, confidence, and trust-calibration measures are secondary and will not substitute for causal behavioral evidence. The exact participant population, task, comparison interface, sample size, and analysis plan remain `[NEEDS INPUT]`.
3. **Contingent application-grounded extension:** If appropriately qualified collaborators and governed data are available, later healthcare work may evaluate the interface with domain experts. This is not required for core T1 success and will not imply clinical validation or deployment.

The human-grounded study introduces a human-subjects dependency. It will proceed only after Berry's designated institutional review process determines whether the activity is exempt or requires IRB approval and all required approvals or determinations are in place. The applicable NSF and institutional requirements must be refreshed before the 2027 submission and again before the study begins.

## Main risks and useful fallbacks

- **Adaptation hides bad retrieval:** exclude raw query and retrieved embeddings from the adapter, train on actual leave-one-out retrieval neighborhoods, use staged training and constrained capacity, calculate retain statistics pre-adaptation, and report pre/post behavior and decision flips.
- **Human edits are ignored or undone:** deferred from the first implementation; when revise is enabled, separate immediate edit effects from post-edit adaptation and test parameter freezing, constraints, or regularization that preserves authorized changes.
- **The interface looks transparent but does not help people intervene:** retain the reproducible functional tests as the core mechanism evidence, then use the bounded human-grounded study to measure error identification, correction success, collateral effects, efficiency, and trust calibration rather than relying on participant preference alone.
- **Retention removes rare competence:** enforce protection and coverage checks, retain reversible archives, and report subgroup/domain regressions.
- **MCB over-stabilizes learning:** compare update rates and no-MCB conditions and retain a useful characterization of the stability-adaptation boundary.
- **One adapter cannot span tasks:** preserve a common reuse interface while allowing task-specific adapter families.
- **One component carries the system:** use staged or alternating training, capacity controls, and separate pre/post, proximity, correction-magnitude, and maintenance diagnostics rather than accepting a good weighted total alone.

## Confirmed relationship to later targets

- **T1:** develop and validate the four-stage NN-kNN/CBR mechanism, package it as an easy-to-use module, and test generalization across representative supervised modalities and compatible neural roles.
- **T2:** test the resulting foundation in reinforcement-learning policy and value roles.
- **T3:** extend case-grounded retrieval, memory, correction, and retention to LLM and agent settings, with a bounded healthcare testbed.

The PI merged the former T0 mechanism-development and T1 module-generalization targets on 2026-09-09 because they form one foundational scientific thrust.

## Consequential details still needed

- the observed RL failure pattern and evidence that motivated the meeting diagnosis;
- the first controlled tasks and exact quantitative margins for bounded-memory and contemporary-method comparisons;
- the conditions and milestone for enabling revise after the first implementation; and
- case-admission and retention timing under static and streaming data.

## Current evidence anchors

- PI-supplied regression paper: `resources/research/nnknn-regression/ijcai-2026-nnknn-regression.pdf`.
- PI-supplied unified NN-CDH paper: `resources/research/nncdh/iccbr-2022-case-adaptation-with-neural-networks.pdf`.
- PI-supplied retrieval-adaptation harmonization paper: `resources/research/nncdh/iccbr-2021-harmonizing-retrieval-adaptation-ao.pdf`.
- PI-supplied excerpt of the newer label-conditioned adaptation design: `resources/research/nncdh/new-nnknn-label-conditioned-adaptation-excerpt.png`.
- Living NN-kNN checkout: `D:\NN-kNN` at commit `c09719576b3519e9878764190773916cd9ce82e6`, inspected 2026-09-08.
- Retrieval and reuse: `model/nnknn_model.py` and `model/nn_cdh.py`.
- Current bounded RL retention machinery: `model/nnknn_rl_workflow.py`.
- Human correction and retention design: [T1_REVISE_RETAIN.md](T1_REVISE_RETAIN.md).
- Collaborator implementation plan: [T1_IMPLEMENTATION_HANDOFF.md](T1_IMPLEMENTATION_HANDOFF.md).
- Module contract and portability sub-aim: [T1_MODULE_GENERALIZATION.md](T1_MODULE_GENERALIZATION.md).
