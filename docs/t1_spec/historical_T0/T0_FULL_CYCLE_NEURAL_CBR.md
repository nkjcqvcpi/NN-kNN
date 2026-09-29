# T0 full-cycle neural CBR foundation

## Status and working name

The PI requested this new foundational target on 2026-09-08 after a collaborator meeting about the current NN-kNN reinforcement-learning work and subsequently confirmed T0 as a formal scientific thrust. The working name **full-cycle neural CBR** is used because the system is organized around a neural NN-kNN core but intentionally assigns consequential revision decisions to a person. Calling every stage "fully neural" would obscure that human role. The final name remains subject to PI confirmation.

## Motivation and claim boundary

The current NN-kNN-RL implementation establishes a technically concrete actor-critic platform, but its documented smoke results establish plumbing rather than competitive performance. The PI and collaborators now suspect that progress is limited by unresolved properties of NN-kNN itself, even after the separate MCB stability work. This is a research hypothesis motivated by current experience; the specific failure modes, experiments, and causal evidence from the meeting have not yet been supplied.

T0 therefore asks a more fundamental question before expanding to reinforcement learning:

> Can NN-kNN become the trainable core of a complete case-based reasoning cycle whose retrieval, reuse, human revision, and retention mechanisms improve one another while preserving inspectable links between cases and behavior?

The novelty cannot be the classical retrieve-reuse-revise-retain vocabulary by itself. The proposed contribution is an integrated, differentiable, testable architecture that instantiates the cycle around NN-kNN, exposes human intervention points, and characterizes interactions and failure conditions across the stages.

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

Existing foundation: maintained classification and regression implementations already calculate case distances, bias-minus-distance scores, and normalized activation. Proposed T0 work must improve retrieval quality and stability where current behavior is inadequate, rather than merely restating that NN-kNN retrieves cases.

### 2. Reuse - neural adaptation of retrieved solutions

The regression paper supplies a concrete reuse instance: after aggregating a retrieved neighborhood, NN-CDH predicts a correction from the query-to-neighborhood difference and the retrieved estimate. The maintained code implements aggregate and per-case adapter variants. Its newer aggregate form uses `[z_q - z_bar_q, y0_q]`, not the retrieved context embedding, so the adapter cannot trivially reconstruct `z_q` from a difference-plus-context pair and bypass adaptation.

The PI-supplied ICCBR-2022 paper supplies a complementary classification idea: one-hot encode nominal attributes and class labels, then represent a nominal change by subtracting the encodings. T0 will adapt this idea to the new aggregate design. With normalized case activations `a_qi`, define:

```text
z_bar_q = sum_i a_qi * z_i
p0_q = sum_i a_qi * e(y_i)
Delta_u_q = concat_j [e(u_qj) - sum_i a_qi * e(u_ij)]
r*_q = e(y_q) - p0_q
r_hat_q = tanh(g_psi([z_q - z_bar_q, Delta_u_q, p0_q]))
s_q = p0_q + r_hat_q
```

For one retrieved case, `r*_q` is an all-zero or one-hot/one-cold class-change vector; for a neighborhood, it is a generalized one-hot/weighted-cold residual. `Delta_u_q` explicitly preserves the old paper's grouped nominal-attribute differences, while `z_q - z_bar_q` retains the new latent aggregate design. The adapter predicts from differences and retrieved label mass, not from the raw query, raw cases, or retrieved embedding. Retain/trustworthiness diagnostics use the pre-adaptation `p0_q` so a downstream adapter cannot conceal poor cases.

Existing foundation: NN-CDH reuse is implemented for regression, and the older paper demonstrates a nominal-difference classification architecture, but the proposed aggregate classification adapter above is not yet implemented or validated. T0 will use a common reuse contract with task-specific adapter semantics. RL actions or values and later LLM/agent memories remain separate future adapter designs rather than being inferred from supervised classification.

### 3. Revise - human-guided case and weight correction

A person inspects retrieved cases, provenance, activations, feature weights, predicted behavior, and available trust/utility evidence. The person may correct case content or a stored solution/label, adjust case- or feature-level weighting, quarantine a case, or introduce a new expert case. The system then measures the immediate behavioral effect and any effect after controlled adaptation, including correct and harmful prediction flips.

Existing foundation: the regression paper provides a controlled synthetic feature-weight-tuning case study showing that an externally supplied weight edit can be retained during further training. This does not yet establish a complete revision interface, reliable case editing, useful domain-expert interaction, or predictable correction in RL or other settings.

**Immediate implementation status:** disabled. Preserve stable IDs, maintenance logs, and reversible case state so revise can be added later, but do not require a human interface or correction experiment in the first T0 implementation.

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

`T_i` is not the complete retain score. The selection policy must also consider exposure/utility from `R_i` and `A_i`, redundancy, protected coverage, and budget `K`. Low exposure means insufficient evidence, not proof of a bad case. Continuous regression labels, RL actor cases, and RL critic value cases require task-specific helpful/harmful contribution definitions rather than silently applying the classification indicator.

Existing foundation: the current RL code includes bounded actor/critic case memories and bias-based pruning with protected case IDs and minimum action coverage. The proposed provenance score, competence/coverage-aware selection, cross-task retention policy, reversible archive, and human-reviewed consequential intervention remain new work.

## Cross-cutting role of Momentum Case Base

MCB is a stability mechanism supporting the four stages, not a replacement for them or a fifth CBR stage. A stable memory representation may make retrieval traces, human edits, and retention decisions more reproducible, but current MCB evidence is limited to the supplied CUB-200-2011 study. T0 will test whether MCB helps the integrated cycle and where stability impedes necessary adaptation.

## Working hypotheses and questions

**H0 - working, needs PI confirmation:** An explicitly integrated four-stage NN-kNN cycle will outperform disconnected retrieval, adaptation, correction, and pruning mechanisms on a joint trade-off among task quality, case-grounded faithfulness, correctability, memory efficiency, and robustness to changing or corrupted cases.

**Immediate H0a - working:** Quality-, utility-, redundancy-, and coverage-aware retention will improve NN-kNN retrieval and task learning over current bias-only pruning and task-agnostic downsampling/random selection at matched case budgets; improved selection will also improve or stabilize the active RL training if case quality is the actual bottleneck.

Candidate questions:

1. When does neural reuse correct an imperfect but meaningful retrieved neighborhood, and when does it bypass or amplify retrieval error?
2. Which human revisions to cases, labels/solutions, case parameters, or feature weights produce predictable targeted changes without unacceptable collateral effects?
3. Which retention policies preserve useful, rare, boundary, and shifted-domain competence under matched case budgets while removing redundancy and limiting harmful knowledge?
4. How does MCB-style representation stabilization interact with retrieval quality, reuse learning, human edits, and retention under nonstationarity?

## Evaluation structure

Use two linked evaluation tracks: controlled classification/regression settings from the completed papers, and the current RL tasks under active development. Supervised tasks provide labeled diagnostics for the case score and a controlled test of classification reuse; RL tests whether better retention addresses the observed bottleneck. Compare in the immediate phase:

- retrieval only;
- retrieve plus reuse;
- classification retrieval-only versus aggregate nominal-residual adaptation, with a logit-residual engineering ablation;
- retrieve plus retain;
- current versus proposed retain policies at matched `K`;
- each condition with and without MCB where feasible; and
- matched neural, nearest-neighbor, case-based, and task-specific baselines.

Add retrieve-plus-revise and the fully integrated four-stage cycle only after the revise stage is enabled.

Measure task performance and calibration; label- or task-aligned retrieval; explanation faithfulness through interventions; correction success and harmful collateral flips; case-base size, retrieval latency, and training cost; rare/shifted-domain retention; neighborhood and representation stability; and failures attributable to each stage or interaction.

## Main risks and useful fallbacks

- **Adaptation hides bad retrieval:** exclude raw query and retrieved embeddings from the adapter, train on actual leave-one-out retrieval neighborhoods, use staged training and constrained capacity, calculate retain statistics pre-adaptation, and report pre/post behavior and decision flips.
- **Human edits are ignored or undone:** deferred from the first implementation; when revise is enabled, separate immediate edit effects from post-edit adaptation and test parameter freezing, constraints, or regularization that preserves authorized changes.
- **Retention removes rare competence:** enforce protection and coverage checks, retain reversible archives, and report subgroup/domain regressions.
- **MCB over-stabilizes learning:** compare update rates and no-MCB conditions and retain a useful characterization of the stability-adaptation boundary.
- **One adapter cannot span tasks:** preserve a common reuse interface while allowing task-specific adapter families.

## Confirmed relationship to later targets

- **T0:** develop and validate the four-stage NN-kNN/CBR mechanism.
- **T1:** package and generalize the validated mechanism as an easy-to-use module across representative supervised modalities and compatible neural roles.
- **T2:** test the resulting foundation in reinforcement-learning policy and value roles.
- **T3:** extend case-grounded retrieval, memory, correction, and retention to LLM and agent settings, with a bounded healthcare testbed.

The PI confirmed this T0 mechanism-development versus T1 module-generalization division on 2026-09-08.

## Consequential details still needed

- the observed RL failure pattern and evidence that motivated the meeting diagnosis;
- the first controlled tasks and success criteria for the integrated cycle;
- the conditions and milestone for enabling revise after the first implementation; and
- case-admission and retention timing under static and streaming data.

## Current evidence anchors

- PI-supplied regression paper: `resources/research/nnknn-regression/ijcai-2026-nnknn-regression.pdf`.
- PI-supplied unified NN-CDH paper: `resources/research/nncdh/iccbr-2022-case-adaptation-with-neural-networks.pdf`.
- PI-supplied excerpt of the newer label-conditioned adaptation design: `resources/research/nncdh/new-nnknn-label-conditioned-adaptation-excerpt.png`.
- Living NN-kNN checkout: `D:\NN-kNN` at commit `c09719576b3519e9878764190773916cd9ce82e6`, inspected 2026-09-08.
- Retrieval and reuse: `model/nnknn_model.py` and `model/nn_cdh.py`.
- Current bounded RL retention machinery: `model/nnknn_rl_workflow.py`.
- Human correction and retention design: [T0_REVISE_RETAIN.md](T0_REVISE_RETAIN.md).
- Collaborator implementation plan: [T0_IMPLEMENTATION_HANDOFF.md](T0_IMPLEMENTATION_HANDOFF.md).
- T1 portability and module generalization: [T1_CORE_MODULE.md](T1_CORE_MODULE.md).
