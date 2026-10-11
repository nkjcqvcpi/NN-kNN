# Open decisions and collaborator handoff checklist

Status: living register. Items below are intentionally unresolved unless marked confirmed. Collaborators should not convert an open choice into a hidden default.

## Confirmed starting directions

- [x] T0 is merged into T1; use the T1-T3 structure.
- [x] Preserve current NN-kNN behavior as a baseline.
- [x] Instrument stable case identities and behavior-neutral statistics before changing selection.
- [x] Implement a shared case-maintenance interface rather than separate incompatible policies for classification, regression, actor, and critic stores.
- [x] Start T1 with revise disabled while retaining reversible logs and archive support.
- [x] Evaluate case selection under a fixed capacity `K` before relying on adaptive growth.
- [x] Set aside the earlier provenance/geometric formula during the case maintenance score redesign, following the 2026-09-17 decision.
- [x] Following the 2026-09-20 brainstorming, compare Q or a C/H variant, B alone, coverage-to-reachability, and case-removal influence. Do not combine Q and B. See [the candidate definitions](../T1_CASE_MAINTENANCE_CANDIDATES.md).
- [x] Define solve by activation above a threshold and the final adapted query outcome, consistent with C/H.
- [x] Use final prediction loss only for removal scoring; report the adaptation penalty separately. Preserve previously trained parameters for the efficient comparison. Optional retraining is a separate higher-cost branch.
- [x] Investigate cached queries for efficient removal scoring. Defer redundancy scoring from similarity between activation maps.
- [x] When adaptation is enabled, update every participating case from the final adapted query outcome, in proportion to its normalized activation. Accumulate evidence over queries to assess retrieval and reuse together, following the 2026-09-20 decision.
- [x] Distinguish trustworthiness from utility, redundancy, coverage, rarity, and source quality.
- [x] Use the current aggregate, retrieved-label-conditioned NN-CDH architecture for the classification reuse extension; do not restore the old architecture wholesale.
- [x] Use explicit nominal query-minus-neighborhood differences only when the shared representation difference does not already include those nominal fields.
- [x] Train the initial classification adapter with residual MSE plus final-label cross-entropy and retain single-term ablations.
- [x] Use `k=5` only as the initial case-bias warm start. Calibrate the later free-correction radius after core training from direct frozen trained-bias activation regions and the matching learned metric.
- [x] Add MCB as an optional controlled factor after the retention mechanism can be tested independently.
- [x] In T2, test full-cycle neural CBR configurations with their NN-kNN core in policy-only, value-only, and both roles. Treat combined actor/critic as a stretch objective.
- [x] Require the T2 Stage A gate before expensive scaling.
- [x] Treat continuous action as a later T2 progression, not the first test.
- [x] Treat domain-shift response and CBR case adaptation as different mechanisms.
- [x] In T3, implement full-cycle neural CBR with an existing host LLM/agent. Use a provider-neutral contract and different NN-kNN retrieval modules for different kinds of knowledge and information. Keep iterative queries bounded and omit an NN-CDH memory adapter.
- [x] Keep host evidence and human audit as separate views of the same retrieval event.
- [x] Learn usefulness from downstream outcomes and optional user feedback; use direct counterfactuals for audit where feasible.
- [x] Separate temporary session, persistent user, authorized domain, and validated global memory.
- [x] Persist user-scoped memory only after explicit save/retention intent.
- [x] Use a public biomedical evidence benchmark for required healthcare evaluation; expert study remains contingent.

## T1 choices to resolve

### Before the first implementation branch

- [ ] Record the actual repository, branch, commit, environment, and relationship to the 2026-09-08 inspected snapshot.
- [ ] Confirm where the active MCB implementation lives and which commit/configuration enables it.
- [ ] Select the first small supervised datasets and recoverable legacy configurations.
- [ ] Resolve how stable case IDs attach to current case tensors and optimizer state.
- [ ] Choose the first maintenance checkpoint policy: batch/epoch boundary, rollout boundary, fixed evaluation point, or a documented two-timescale combination.

### Before the first matched retention experiment

- [ ] Define success and failure for each task, including an acceptable regression error or a continuous outcome measure. The rule for sharing the adapted query outcome among cases is settled.
- [ ] Evaluate whether C/H guides maintenance more reliably as queries accumulate and retrieved case groups vary. Include cases repeatedly retrieved together when checking the PI's averaging hypothesis.
- [ ] Identify the established maintenance methods the general formulation will recover. Derive the assumptions and parameter choices for each, distinguishing exact score recovery, ranking equivalence, and full selector recovery. Preserve the distinction between the established \(C_i=RC(i), H_i=0\) component connection and the still-unresolved general score.

- [ ] Select initial case capacity `K` and maintenance frequency. The current repository's ordinary profile previously used capacity 500/frequency 1000, but those are historical code defaults, not approved experimental values.
- [ ] Compare the four candidate measures before selecting the final case maintenance score and evidence requirements. Choose a Q variant and smoothing; do not restore Q/B combination weights.
- [ ] Specify B normalization and comparable groups for its standalone comparison.
- [ ] Choose activation thresholds, the ratio's zero-reachability policy, and the maintenance reference queries. Defer activation-map similarity thresholds.
- [ ] Choose task prediction losses, acceptable cumulative loss increase, cache refresh, and approximation-error checks. Keep the final test set separate from maintenance selection.
- [ ] Specify optional retraining budgets, parameter scope, and matched further-training controls.
- [ ] Define coverage floors and protection rules for class, action, subgroup, temporal, boundary, rare, and shifted cases.
- [ ] Define the combined coverage/redundancy criterion and its formula, then the scalarization or constrained optimizer combining it with trustworthiness and utility; retain inspectable component outputs. Redundancy and coverage are one selection criterion per the 2026-09-17 PI refinement.
- [ ] Select feasible counterfactual/data-valuation baselines and document computational limits.

### Before classification reuse confirmation

- [ ] Record which nominal fields are already represented in `Delta_z`; do not infer this from tensor dimension.
- [ ] Select positive `lambda_diff` and `lambda_cls` and any calibration-temperature protocol.
- [ ] Decide whether nominal-residual scores or logit residuals are primary after the planned diagnostic comparison. The confirmed starting primary is nominal-residual scores.
- [ ] Select the exact core-training snapshot rule used before free-correction calibration.
- [ ] Select `s_task` and task-specific label/output metrics for classification, regression, actor, and critic conditions.
- [ ] If raw trained-bias activation regions are unstable, predeclare which normalized/clipped/shrunk fallback is tested.

### Before T1 confirmatory runs

- [ ] Freeze legacy reproduction versus reconstruction status, splits, encoders, preprocessing, case budgets, metrics, and tuning budgets.
- [ ] Select one modern structured/tabular transfer family and freeze its snapshot/subset.
- [ ] Freeze contemporary comparator implementations and fair tuning/runtime/hardware accounting.
- [ ] Prespecify task-appropriate competitiveness margins, retention improvements, correction success, collateral-effect limits, seeds, and uncertainty analysis.
- [ ] Finalize the low-risk/common-knowledge UI study task, interface, measures, participant plan, and institutional/human-subjects review.
- [ ] Enable expert-dependent healthcare revision only if a qualified collaborator, governance, and approvals are secured.

## T2 choices to resolve

### Before Stage A

- [ ] Supply exact task-level branches, environment IDs/versions, actor/critic configurations, seeds, metrics, baselines, and results for CartPole, Acrobot, LunarLander, MinAtar Breakout, and current Pong work.
- [ ] Freeze the current NN-kNN RL reference implementation.
- [ ] Choose two or three inexpensive Stage A tasks and identify the role configuration used for the same-configuration gate.
- [ ] Define helpful/harmful contribution for actor cases. Candidates: signed GAE advantage for the executed action, counterfactual policy-loss change under case masking, or longer-horizon return effect.
- [ ] Decide whether critic audit uses GAE value targets, discounted Monte Carlo returns, or both for distinct purposes.
- [ ] Verify actor/critic label modes and target-critic semantics for every retained case.
- [ ] Freeze interaction, compute, case-memory, tuning, and evaluation budgets plus seeds, metrics, and competitiveness margin.

### After Stage A

- [ ] Select the continuous-action suite only after a pilot of a representative MuJoCo or DeepMind Control task.
- [ ] Decide whether MinAtar and/or an Atari/ALE subset is necessary to test visual scaling; Pong is an ongoing exploration, not yet a committed suite.
- [ ] Select the controlled nonstationarity intervention: dynamics, observation, reward, or a prespecified combination.
- [ ] Define recovery, retained-competence, case-turnover, and catastrophic/meaningful-remembering metrics.
- [ ] Select fair modern neural and episodic/retrieval-memory comparators for each action space.
- [ ] If full substitution misses the gate, specify the approved hybrid/parallel neural policy while retaining matched ablations of the full-cycle system and its NN-kNN core.
- [ ] Size cloud GPU workload from pilots rather than benchmark name alone.

## T3 choices to resolve

### Before the general RAG prototype

- [ ] Select one open-weight host model/checkpoint for the prompt-augmentation and model-integrated comparisons.
- [ ] Select and justify the internal NN-kNN connection: retrieval-conditioned attention, hidden-state fusion, output-level gating, or another declared interface.
- [ ] Specify which host and NN-kNN parameters are trainable in the model-integrated condition.
- [ ] Define matched prompt-only, internal-only, and combined comparison budgets.
- [ ] Select the bounded single-hop integration benchmark/corpus snapshot.
- [ ] Select HotpotQA or MuSiQue, or another justified equivalent, for the primary complementary multi-hop test after pilots.
- [ ] Finalize the observable retrieval-request schema and how need/subtask representations are inspected.
- [ ] Finalize the initial artifact taxonomy and decide which types use separate case banks, separate metrics/heads, or shared components.
- [ ] Calibrate match and usefulness across types; define how mixed requests share a fixed total budget.
- [ ] Set retrieval-round, context, latency, compute, novelty, duplicate-query, and minimum-usefulness limits.
- [ ] Choose direct counterfactual audit rate and validate any learned usefulness estimator against it.
- [ ] Freeze no-retrieval, one-shot/standard RAG, iterative NN-kNN, and contemporary comparator implementations under one host checkpoint.

### Before feedback and memory learning

- [ ] Define overall and optional per-case feedback scales, interface language, consent, and missing-feedback treatment.
- [ ] Choose contextual-bandit conditions for immediate choice and RL conditions for delayed multi-round outcomes.
- [ ] Define validated batch-update cadence, minimum evidence, poisoning checks, held-out validation, rollback triggers, and version promotion.
- [ ] Define session capacity, compression, eviction, task boundary, and expiration.
- [ ] Define persistent user-memory access, retention, inspection, deletion, and consent behavior.
- [ ] Define domain membership/governance and local-to-global promotion rules.
- [ ] Calibrate joint local/global ranking and the query-dependent local-relevance prior.

### Before biomedical and agent studies

- [ ] Verify a historical BioASQ Task b release, corpus, license, executable evidence metrics, exact-answer metrics, and baseline reproducibility. BioASQ is leading, not yet final.
- [ ] Decide whether MIRAGE is feasible as secondary external validity and whether PubMedQA is useful as a simple pilot.
- [ ] Confirm there is no PHI and complete required Berry data/governance review.
- [ ] If an expert-grounded extension is added, confirm qualified personnel, exact authority, protocol, burden, and approval before promising it.
- [ ] Select an existing agent framework and bounded task that isolate NN-kNN experience-memory value.
- [ ] Define agent case units: decision records first, then trajectory fragments only if evidence supports them.
- [ ] Define success/failure/tool-use outcome measures and the delayed-credit protocol.
- [ ] Decide which explicit rules, formulas, relations, constraints, or procedures are stored as cases and which use external validation; differentiable rule penalties remain optional.

## Collaborator implementation-return checklist

For every completed work package, return:

- [ ] Repository URL, branch, commit, and dirty/clean status.
- [ ] Concise implementation summary and changed-module map.
- [ ] Updated repository `HANDOFF.md` and any architecture/data-schema documentation.
- [ ] Configuration fields, defaults, and backward-compatibility behavior.
- [ ] Exact environment, dependencies, data/environment versions, and licenses.
- [ ] Test report covering behavior-neutral instrumentation and relevant edge cases.
- [ ] Resolved run manifests and matched-budget comparison audit.
- [ ] Case statistics, retrieval events, maintenance events, interventions, and metric artifacts.
- [ ] Scripts/configurations that regenerate summary tables and figures.
- [ ] Results separated into exploratory and confirmatory analyses with uncertainty.
- [ ] Failure analysis identifying retrieval, representation, case labels/solutions, reuse, maintenance, RL optimization, host reuse, or feedback credit assignment as distinct possible bottlenecks.
- [ ] New uncertainties, deviations, negative findings, and proposed decisions requiring PI approval.

## Immediate handoff sequence

1. Collaborator records the live NN-kNN commit and current RL branches/configurations.
2. Implement behavior-neutral stable identities, shared statistics, and logs.
3. Pass the T1 instrumentation gate.
4. Prototype fixed-`K` supervised retention before expensive RL sweeps.
5. Implement classification reuse as an independent switch.
6. Re-run recoverable pre-MCB legacy settings, then matched MCB/full-cycle variants.
7. Apply the shared T1 mechanisms to current RL tasks and evaluate the Stage A gate.
8. Resolve later continuous-action/visual suites only from Stage A and pilot evidence.
9. Start T3 with a frozen-host, single-hop query-retrieve-answer prototype after the T1 retrieval interface and trace are usable.

The collaborator may choose ordinary engineering details that do not change the scientific treatment or claim. Any change to the confirmed architecture, comparison logic, outcome definition, or human/safety authority should be raised with the PI before implementation.
