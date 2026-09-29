# Shared experiment and data contract

Status: cross-thrust implementation requirement. Exact storage formats and several numerical settings remain open, but every implementation must preserve the information and comparison controls below.

## 1. Design principles

1. Every reported result must identify the exact code, model, data, retrieval state, maintenance policy, and active component combination that produced it.
2. Case-level explanations must expose the cases and weights used by the implemented decision rule, not a separately generated explanation.
3. A displayed influence becomes evidence of faithfulness only when case or feature interventions change behavior in the predicted direction.
4. Retrieval quality must be measured before downstream adaptation or host-model reuse can conceal it.
5. Trustworthiness, current-query match, observed utility, redundancy, coverage, and source quality are distinct variables. Do not collapse them into one undocumented score.
6. Unobserved evidence is not negative evidence. Low exposure, unrated cases, and missing user feedback must remain explicitly unknown.
7. Case removal is reversible during research. Every maintenance or human action is logged with reason and restoration information.
8. Evaluation and test data never influence training-time selection, retention, calibration, early stopping, or model choice unless the protocol explicitly designates a separate development stream.

## 2. Stable identities

Tensor positions and active-memory slots are not stable identities because compaction and replacement change them. Use immutable identifiers and explicit version links.

Minimum identifiers:

- `project_run_id` — one experiment execution;
- `configuration_id` — complete resolved configuration hash or equivalent;
- `model_snapshot_id` — host/core/actor/critic checkpoint as applicable;
- `retrieval_snapshot_id` — representation, learned metric, feature weights, and case-bias state used for retrieval;
- `case_id` — stable across compaction, archive, restore, and scope changes;
- `case_version_id` — immutable content/metadata revision;
- `retrieval_event_id` — one query-to-case-layer computation;
- `maintenance_event_id` — one insert/protect/archive/restore/replace action;
- `intervention_id` — one planned deletion, replacement, reweighting, correction, or poisoning manipulation;
- `feedback_event_id` — one overall or case-level observation; and
- `parent_id` or explicit lineage edges for copied, corrected, summarized, promoted, or scope-transitioned cases.

Actor and critic case namespaces must remain distinct even when their encoders or observations are shared.

## 3. Case record

All task families should map to one extensible record rather than incompatible stores:

```text
case_id
case_version_id
case_role                 # classification, regression, actor, critic, evidence, tool, procedure, experience, rule, ...
artifact_type
scope                     # session, user, authorized domain, global
content_or_stable_reference
input_or_need_representation_reference
stored_solution           # label, value, action recommendation, artifact, outcome, ...
source_and_provenance
insertion_time_or_step
last_retrieved_time_or_step
initial_case_bias
current_case_bias
feature_or_glocal_parameter_reference
retrieval_count
activation_mass
positive_support
harmful_support
provenance_quality_Q
normalized_bias_B
trustworthiness_T
match_score_when_queried
utility_estimates_and_context
redundancy_and_coverage_metadata
cohort_or_class_or_action_metadata
protected_flag_and_reason
reliability_status        # provisional, approved/high-confidence, uncertain, quarantined
archive_state
consent_access_retention_and_deletion_metadata
lineage
```

Fields that do not apply remain null with an explicit reason; do not invent a surrogate. Derived representations and summaries must be marked derived and traceable to original content or a stable source reference.

Do not log sensitive raw case content by default. Use stable references and controlled storage when governance requires it.

## 4. Retrieval-event record

Each retrieval event records:

- run, task, seed, environment/dataset item, and model/retrieval snapshot IDs;
- observable query, need, state, or subquestion;
- requested artifact type(s), constraints, and already-retrieved IDs;
- eligible scopes and safety/reliability gates;
- all candidate and selected case IDs needed to reproduce the decision;
- raw and calibrated distances, similarities, activations, case biases, and applicable feature/case contributions;
- pre-adaptation or pre-reuse prediction where applicable;
- returned case order and total budget;
- downstream adapted prediction, answer, action, value, or outcome;
- latency, memory, context/tokens, compute, and monetary cost where applicable; and
- continuation/stopping decision and declared reason for iterative retrieval.

For T3, derive the host-LLM evidence channel and human-audit channel from this same event. Do not create a separate explanation event that can drift from the evidence actually supplied.

## 5. Maintenance-event record

Each event includes at least:

- run ID, seed, task, model role, step/epoch, and policy version;
- stable case ID and active slot before and after action;
- action: insert, keep, protect, archive, restore, replace, quarantine, release from quarantine, or revise;
- actor/authority and reason code;
- `Q_i`, `B_i`, `T_i`, retrieval count, activation mass, positive and harmful support;
- evidence sufficiency, redundancy, coverage, rarity/boundary/shift/cohort indicators;
- relevant label, action, value, artifact type, and scope metadata;
- capacity before and after;
- deterministic tie-break information; and
- restoration data and parent/child lineage for corrections.

Suggested artifacts are `case_statistics`, `case_maintenance`, `case_selection_summary`, and an archived/restorable case state in each checkpoint. CSV, JSON, or a lossless columnar form is acceptable if schemas and versions are declared.

## 6. Experiment manifest

Every run must resolve and save, at minimum:

```yaml
run:
  id: ...
  timestamp: ...
  code_repository: ...
  commit: ...
  dirty_worktree: ...
  environment_lock: ...
  seed: ...

task:
  dataset_or_environment: ...
  version: ...
  split_or_scenario: ...
  preprocessing_or_wrapper: ...
  primary_metrics: ...

model:
  role_configuration: ...
  baseline_or_nnknn_variant: ...
  host_model_snapshot: ...
  encoder_and_representation: ...
  learned_distance_and_feature_weights: ...

components:
  retrieve: ...
  reuse_or_adapter: ...
  revise: ...
  retain: ...
  mcb: ...
  component_synchronization: ...

budgets:
  case_capacity: ...
  training_or_environment_interactions: ...
  tuning_trials: ...
  wall_clock_or_compute: ...
  retrieval_rounds: ...
  context_or_tokens: ...

maintenance:
  policy: ...
  frequency_or_safe_checkpoint: ...
  thresholds_and_smoothing: ...
  archive_and_restore: ...

evaluation:
  confirmatory_or_exploratory: ...
  number_of_seeds: ...
  uncertainty_method: ...
  competitiveness_margin: ...
  intervention_protocol: ...
```

An ellipsis is not a default. It means the value must be resolved before the run.

## 7. Matched-comparison contract

Comparisons should hold constant, as applicable:

- data/environment version, splits, preprocessing, and observation/action wrappers;
- encoder/host checkpoint unless encoder variation is the declared treatment;
- interaction or training-example budget;
- case-memory capacity `K` and insertion stream;
- tuning budget and model-selection rule;
- random seeds and evaluation episodes/examples;
- retrieval/context/round budget;
- hardware accounting or a transparent normalization;
- prompt, decoding, tool schema, and available tools for T3; and
- early stopping and checkpoint selection.

Report both configured and actual budgets. If a legacy paper protocol cannot be exactly recovered, document every known difference and label the result a reconstruction, not an exact reproduction.

Use multiple seeds and uncertainty estimates for confirmatory comparisons. Exact seed counts, margins, and minimum improvements are unresolved until task-specific pilots; freeze them before confirmatory runs. Failure to detect a difference is not evidence of equivalence unless the analysis was designed for non-inferiority or practical equivalence.

## 8. Component switches and factorial testing

Every T1/T2 implementation exposes and logs separable switches for:

- retrieval-only versus bounded neural reuse;
- MCB off/on;
- current/no learned maintenance versus proposed quality-aware retention;
- revise off/on;
- independent component training versus synchronized/alternating training; and
- conventional neural versus NN-kNN policy/value roles in T2.

Run broad main-effect and important-interaction coverage on inexpensive tasks. Use prespecified evidence-selected combinations on expensive tasks; do not attempt every combination everywhere. The classification adapter must remain separately switchable so retention can use the same case set with and without reuse.

## 9. Causal explanation and intervention tests

For cases displayed as influential, predeclare interventions such as:

- remove or mask one case and renormalize remaining activations;
- replace it with a matched irrelevant, contradictory, corrected, or oracle case;
- increase/decrease its bias or weight;
- alter a declared feature weight;
- quarantine and restore it; or
- inject controlled poisoning, outlier, stale, or shifted cases in an isolated condition.

Record the predicted direction and affected output before applying the intervention. Then measure targeted corrections, correct and harmful flips, reversals, collateral changes on unaffected examples/tasks, calibration, performance, and recovery. A case trace is faithful only to the extent that these effects agree with the displayed mechanism.

For T3, also test whether the host actually uses the retrieved evidence: retrieval relevance alone does not establish grounding, and a correct answer alone does not establish that the case caused it.

## 10. Separate outcome families

Report metrics in separate families rather than one composite headline:

1. Task performance and uncertainty.
2. Retrieval quality and stability.
3. Reuse/adaptation quality.
4. Retention and bounded-memory quality.
5. Efficiency and scaling.
6. Representation, neighborhood, and selection stability.
7. Causal traceability and audit completeness.
8. Human correction, burden, and collateral effects.
9. Poisoning, outlier, stale-knowledge, and domain-shift resistance/recovery.
10. Scope, privacy, expiration, deletion, and cross-scope leakage for T3 memory.

Do not use higher task performance to excuse failed traceability, or visible cases to excuse unacceptable task performance.

## 11. Data and evaluation separation

- Fit preprocessors and nominal vocabularies on training data only.
- Exclude self-neighbors from training-case calibration.
- Keep validation/test labels out of retention scores and free-adaptation calibration.
- Distinguish the training stream, maintenance/audit stream, validation stream, and untouched evaluation stream.
- If RL rollouts update maintenance statistics, they are training data, not reporting-only holdouts.
- In T3, separate static/offline training, feedback collection without updating, validated batch updates, and controlled immediate-online-update experiments.
- Freeze benchmark/corpus snapshots and licenses before confirmatory runs.
- Preserve benchmark-specific ground truth separately from model-generated labels or LLM self-judgments.

## 12. Governance and safety

- No protected health information is assumed or required.
- The required healthcare work uses an established public biomedical literature/evidence benchmark with objective ground truth.
- General benchmark case review may be done by researchers with adequate task knowledge. Specialized medical correctness or consequential case actions require qualified expertise and applicable approvals.
- Retrieved tool descriptions do not authorize tool execution.
- Quarantined cases are ineligible for normal retrieval regardless of match.
- An uncertain general-benchmark case may be supplied with a marker in a controlled condition; uncertain biomedical material is a review candidate, not authoritative evidence.
- Local user memories require explicit persistence, remain inspectable/deletable, and never become domain/global memory without separate validation.
- Record only observable prompts, queries, subquestions, actions, outcomes, and feedback. Do not require or store private chain-of-thought.

## 13. Reproducibility package expected from each workstream

- branch and exact commit;
- concise implementation summary and repository `HANDOFF.md` update;
- environment/dependency lock and data-acquisition instructions;
- complete configuration schema and resolved run manifests;
- unit, smoke, and integration-test reports;
- raw or lossless metric and event artifacts;
- scripts/configurations that reconstruct tables and figures;
- model and case-memory checkpoints or documented retention limits;
- matched-comparison audit;
- failure-analysis memo; and
- unresolved-decision list.

The proposal workspace is for specifications and evidence, not the implementation itself. Code and experiment artifacts remain in the collaborators' appropriate repositories and return here only as documented evidence or preserved references authorized by the PI.
