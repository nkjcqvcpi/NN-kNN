# Open decisions and collaborator handoff checklist

Status: living register. Items below are intentionally unresolved unless marked confirmed. Collaborators should not convert an open choice into a hidden default.

## Confirmed starting directions

- [x] T0 is merged into T1; use the T1-T3 structure.
- [x] Preserve current NN-kNN behavior as a baseline.
- [x] Instrument stable case identities and behavior-neutral statistics before changing selection.
- [x] Implement a shared case-maintenance interface rather than separate incompatible policies for classification, regression, actor, and critic stores.
- [x] Start T1 with revise disabled while retaining reversible logs and archive support.
- [x] Evaluate case selection under a fixed capacity `K` before relying on adaptive growth.
- [x] Use the activation-weighted provenance formulation and geometric provenance-bias trustworthiness score specified in the T1 plan.
- [x] Distinguish trustworthiness from utility, redundancy, coverage, rarity, and source quality.
- [x] Use the current aggregate, retrieved-label-conditioned NN-CDH architecture for the classification reuse extension; do not restore the old architecture wholesale.
- [x] Use explicit nominal query-minus-neighborhood differences only when the shared representation difference does not already include those nominal fields.
- [x] Train the initial classification adapter with residual MSE plus final-label cross-entropy and retain single-term ablations.
- [x] Use `k=5` only as the initial case-bias warm start. Calibrate the later free-correction radius after core training from direct frozen trained-bias activation regions and the matching learned metric.
- [x] Add MCB as an optional controlled factor after the retention mechanism can be tested independently.
- [x] In T2, test NN-kNN policy-only, value-only, and both, with combined actor/critic a stretch objective.
- [x] Require the T2 Stage A gate before expensive scaling.
- [x] Treat continuous action as a later T2 progression, not the first test.
- [x] Treat domain-shift response and CBR case adaptation as different mechanisms.
- [x] In T3, use an existing host LLM/agent, a provider-neutral retrieval contract, typed retrieval, bounded iterative queries, and no NN-CDH memory adapter.
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

- [ ] Select initial case capacity `K` and maintenance frequency. The current repository's ordinary profile previously used capacity 500/frequency 1000, but those are historical code defaults, not approved experimental values.
- [ ] Select smoothing `s`, trust weight `alpha`, minimum retrieval count, minimum activation mass, and evidence rule.
- [ ] Choose cohort-aware bias normalization and define class/action/cohort groups.
- [ ] Choose the learned representation/distance and threshold used for redundancy.
- [ ] Define coverage floors and protection rules for class, action, subgroup, temporal, boundary, rare, and shifted cases.
- [ ] Define the scalarization or constrained optimizer for trustworthiness, utility, redundancy, and coverage; retain separate outputs.
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
- [ ] If full substitution misses the gate, specify the approved hybrid/parallel neural policy while retaining matched NN-kNN contribution ablations.
- [ ] Size cloud GPU workload from pilots rather than benchmark name alone.

## T3 choices to resolve

### Before the general RAG prototype

- [ ] Select a frozen open-weight host model/checkpoint and project-controlled orchestration framework.
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
