# T2 implementation plan: full-cycle neural CBR for reinforcement learning

Status: PI-confirmed objective, staged gate, and role matrix. Exact current configurations, results, benchmark suite, contribution semantics, and numerical thresholds remain unresolved.

## 1. Objective

T2 asks when the improved T1 full-cycle neural CBR system can serve as:

- the policy/actor;
- the value function/critic; or
- both actor and critic with separate case memories,

while preserving competitive reinforcement-learning performance and adding faithful case-level inspection, debugging, human-tunable case/feature weights, explicit case addition/removal, and quality-aware bounded memory.

The main novelty claim is not that no prior RL method uses cases or memory. Neural Episodic Control and retrieval-augmented RL are relevant prior lines. T2's sharper question is whether a coordinated full-cycle neural CBR system can operate in policy and value roles. NN-kNN supplies its learned case-based core. The complete system exposes actual case and feature contributions, supports direct correction, and maintains bounded case memory through the T1 mechanisms.

Full substitution by the full-cycle neural CBR system remains the primary hypothesis. Minimum success is competitive task performance plus faithful case-level inspection in at least one major role—policy or value. Using the system in both roles is a higher-risk stretch objective. A role boundary or negative result can be scientifically useful when supported by matched experiments.

## 2. Why T2 follows and informs T1

T1 and T2 run concurrently rather than in a strict sequence. The collaboration began from RL implementation, then returned to the core after NN-kNN showed accuracy and efficiency difficulties in some domains. The hypothesis that better case selection, MCB, neural adaptation, or maintenance will resolve the limitation must be tested component by component.

T2 also prepares for T3, but only partially:

- RL provides methods for sequential decisions, delayed outcomes, credit assignment, and learning usefulness from feedback.
- T3's LLM/agent systems may use RL-trained routing or memory policies.
- An LLM is much more than an RL model; T2 is not a claim that solving RL solves LLMs or agents.
- T3 extends full-cycle neural CBR through a hybrid host. NN-kNN retrieves. The host neural LLM performs reuse and supplies general reasoning. Human intervention supplies revision, and case-base maintenance supplies retention.

## 3. Evidence boundary and current tasks

PI-reported work has covered or attempted:

- CartPole;
- Acrobot;
- LunarLander;
- MinAtar Breakout; and
- Atari Pong, currently in progress.

NN-kNN reportedly underperforms existing RL methods in some domains. The PI recalls that one-role and both-role conditions may have been tested, but exact actor/critic configurations are changing and have not been supplied. No matched task-level results, versions, branches, baselines, metrics, seed statistics, efficiency results, or intervention tests are yet recorded in the proposal workspace.

Treat these as preliminary implementation breadth and diagnostic motivation, not evidence that the Stage A gate has passed or that case selection caused the observed limitations.

## 4. Role architecture

### 4.1 Required configurations

Keep these conditions independently selectable:

1. conventional neural actor + conventional neural critic;
2. full-cycle neural CBR actor, built around an NN-kNN core, + conventional neural critic;
3. conventional neural actor + full-cycle neural CBR critic, built around an NN-kNN core;
4. full-cycle neural CBR actor + full-cycle neural CBR critic with separate case stores; and
5. an approved hybrid/parallel neural policy contingency if full substitution is not adequate.

Here, a full-cycle role uses the applicable T1 retrieve, reuse, revise, and retain mechanisms. NN-kNN is the learned case-based core inside that role. Do not describe a hybrid as complete neural replacement. Use matched ablations to isolate what the full-cycle system and its NN-kNN core contribute to performance, traceability, correction, and maintenance.

### 4.2 Initial case unit

Begin with a decision-level case:

- **Actor:** state/context plus an action recommendation or action-distribution target and the observable outcome/training signal.
- **Critic:** state/context plus a declared value target and its training/audit provenance.

Trajectory fragments are a later bridge toward agent memory when a single decision lacks enough context. They require their own representation, retrieval, temporal-credit, length/budget, and redundancy rules; do not introduce them merely because they sound more agent-like.

### 4.3 Actor and critic independence

- Use separate stable case IDs, stores, statistics, capacities, maintenance events, and checkpoints.
- Preserve role-specific targets even if the observation encoder is shared.
- Never compact one store using another store's keep mask.
- Verify optimizer and target-network alignment after maintenance.
- If using a lagged/target critic, distinguish its bootstrap values from online trainable case labels and separate audit returns.

## 5. Inherited T1 mechanisms

Every RL run records switches for:

- current retrieval versus T1-enhanced retrieval/selection;
- neural adaptation of retrieved solutions where that role has a defined adapter;
- MCB off/on;
- no/current maintenance versus provenance/utility/coverage-aware retention;
- independent versus later synchronized training; and
- human revision off/on as a separate assistance condition.

Broad combinations belong on cheap Stage A tasks. At scale, prespecify a small set selected from T1 and Stage A evidence. Do not mix human intervention into the autonomous learning result.

The first RL transfer should focus on T1 mechanisms already independently testable. Do not simultaneously change encoder, environment wrapper, optimizer, actor/critic role, maintenance, and MCB without a design that separates their effects.

## 6. RL-specific provenance and utility

### 6.1 Critic cases

Treat critic contribution as regression. On tractable audit subsets, compare the loss with and without case `i`, renormalizing the remaining activation:

```text
Delta_i(x) = Loss(V_without_i(x), target_x) - Loss(V_with_i(x), target_x)

C_i = sum_x max(Delta_i(x), 0)
H_i = sum_x max(-Delta_i(x), 0)
```

Candidate target uses:

- GAE-derived value targets for training diagnostics;
- discounted Monte Carlo return from separate audit rollouts for a less bootstrap-dependent audit; or
- both, kept distinct.

Do not choose among these silently.

### 6.2 Actor cases

Helpful/harmful actor contribution remains an open scientific choice. Candidate signals are:

- the executed action's signed GAE advantage, allocated by the case's contribution;
- counterfactual change in policy loss after masking the case;
- counterfactual change in action probabilities or selected action, checked against advantage;
- downstream episodic-return change under controlled intervention; and
- a learned contribution estimator validated against direct rollouts.

An action change is not automatically an improvement. Validate the sign against return or another declared task outcome. Short-horizon policy-loss tests are cheaper; long-horizon interventions are more faithful but expensive and noisy.

### 6.3 Trustworthiness, utility, and coverage

After choosing role-specific positive/harmful contribution, evaluate the T1 Q candidate:

```text
Q_i = (C_i + s) / (C_i + H_i + 2s)
```

The 2026-09-20 T1 direction supersedes the Q/B combination. B is a separate candidate. Transfer the maintenance method supported by T1; see [the candidate definitions](../T1_CASE_MAINTENANCE_CANDIDATES.md) for coverage-to-reachability and case-removal influence. Role-specific success and loss definitions still need to be established for RL.

Keep exposure, match, uncertainty, action/state coverage, boundary/rare-state protection, redundancy, and temporal/domain coverage separate. A rarely visited state case may preserve a critical capability; low visitation does not imply uselessness.

## 7. Stage A readiness gate

Before a full compute-intensive matrix, evaluate locally on two or three inexpensive low-dimensional tasks. CartPole, Acrobot, and LunarLander are leading candidates, subject to exact environment/configuration review. Larger exploratory pilots may continue but do not replace this gate.

One T1-enhanced full-cycle neural CBR configuration must satisfy all of the following:

1. Improve task learning, stability, or efficiency over a frozen current NN-kNN RL reference under matched environment interactions, compute, case memory, tuning, and evaluation budgets.
2. Reach a prespecified task-appropriate competitiveness margin relative to a matched neural baseline in at least one substantive role—policy, value, or both. Applicable episodic/retrieval-memory methods are additional comparators.
3. Use multiple seeds with fixed or transparently accounted budgets.
4. Produce case traces identifying the cases and weights used by the implemented decision rule.
5. Pass causal interventions showing that deleting, replacing, or reweighting influential cases or features changes behavior in the predicted direction.

Freeze the reference version, roles, primary metrics, budgets, tuning procedure, margins, seeds, and checkpoint rule before confirmatory gate runs. The gate does not require the full-cycle system to beat neural RL on every task or role.

If the full-replacement gate fails:

- return to stage-specific T1 diagnosis;
- report the suitability boundary;
- activate a role-specific hybrid/parallel neural contingency if useful; and
- do not assume that a larger environment or more compute will fix the mechanism.

## 8. Staged benchmark progression

Benchmark names are candidates until pilot review; commit to scientific coverage rather than a fragile software suite.

### Stage A — controlled state-based discrete action

Purpose: debug actor/critic roles, labels, retention, performance, efficiency, and causal case traces cheaply.

Candidates: CartPole, Acrobot, LunarLander.

### Bridge — compact spatial observation

Candidate: MinAtar Breakout.

Purpose: test representation and memory scaling before full Atari pixels. It is not a substitute for a formal Atari/ALE protocol.

### Stage B — continuous action

Candidate families: a small representative MuJoCo or DeepMind Control subset after a pilot.

Continuous-action RL means the actor chooses real-valued controls rather than one of a few discrete actions. This requires an NN-kNN case to support a continuous action vector or policy-distribution parameters and makes interpolation, adaptation, and action-space distance more important. Start only after the discrete actor/value integration and maintenance mechanisms pass the readiness gate.

### Stage C — visual/scaling stress

Candidate: a representative Atari/ALE subset beginning from the current Pong work. Use only if it directly tests representation and case-memory scaling at feasible multiple-seed cost.

Procgen is an optional visual/generalization alternative, not a commitment. Meta-World is a future multi-task alternative and would add substantial scope.

### Selection criteria

The final suite should collectively provide:

- a controlled discrete-action role/faithfulness test;
- a representative continuous-action progression if earlier results justify it;
- a scalable observation challenge only when needed;
- a controlled nonstationarity condition;
- strong neural and applicable memory baselines; and
- stable implementations, licensing, versions, and feasible costs.

Replacing a benchmark after award with a comparable one can preserve the scientific objective when the same roles, interpretability, maintenance, scaling, and shift questions remain. Any material objective/scope change requires Berry/NSF review under the rules then in effect.

## 9. Domain shift and meaningful remembering

This section uses **adaptation** to mean response to a changed environment, not NN-CDH solution adaptation.

Introduce a prespecified change in one or more of:

- dynamics;
- observation distribution/noise;
- reward function; or
- task regime.

Primary post-shift response:

- continual/online learning;
- automatic case admission, reweighting, protection, and maintenance;
- bounded preservation of useful pre-shift competence; and
- detection of stale, harmful, or newly valuable cases.

Separate conditions:

- active learning or selective human review;
- explicit human case/feature intervention; and
- no-update or ordinary neural adaptation baselines.

Measure immediate degradation, recovery speed, final post-shift performance, retained pre-shift competence, case turnover, rare-state retention, memory growth, stability, and collateral forgetting. The goal is not to keep all old cases; it is to remember what remains useful and acquire the knowledge demanded by the new domain.

## 10. Baselines and fairness

Use environment-appropriate implementations, potentially including:

- frozen current NN-kNN RL;
- matched MLP actor/critic;
- PPO for appropriate discrete or mixed progression;
- SAC for continuous control;
- ordinary k-NN when a fair action/value interface exists;
- Neural Episodic Control;
- Retrieval-Augmented RL or an equivalent applicable episodic/retrieval-memory method; and
- role-specific hybrids.

“Modern” means appropriate, reproducible, and fairly matched; it does not require every recent algorithm. Hold environment version, wrappers, interactions, tuning budget, seed schedule, case budget, evaluation episodes, and compute accounting constant where possible. Report deviations.

Use strict fixed-step comparison or transparently report configured and actual steps, best and final checkpoints, early-stopping information, and success thresholds. Smoke tests prove plumbing only.

## 11. Evaluation outcomes

### Task learning

- episodic return and task success;
- sample efficiency and learning curves;
- stability/variance across seeds;
- best versus final policy;
- value error and calibration where meaningful; and
- performance before and after shift.

### Case mechanism

- retrieval exposure and activation;
- counterfactual actor/critic contribution;
- action/value change after case/feature intervention;
- harmful, redundant, rare, boundary, and protected cases;
- case turnover, archive/restore behavior, and capacity utilization;
- representation and neighborhood churn; and
- relationship between displayed influence and causal effect.

### Efficiency and scale

- training wall time and accelerator time;
- environment-step throughput;
- retrieval and maintenance latency;
- active/archive memory and checkpoint size;
- maintenance frequency/overhead; and
- compute-normalized performance.

### Nonstationarity

- degradation and recovery curves;
- cases added, reweighted, quarantined, or evicted;
- old-task retention and new-task acquisition;
- automated versus assisted recovery; and
- failure modes under poisoned/bad case knowledge.

## 12. Causal intervention protocol

For selected decisions, record the cases/features predicted to matter, then:

- remove/mask the top case and renormalize;
- replace it with a matched irrelevant, corrected, contradictory, or shifted case;
- alter case bias or feature weight;
- archive/quarantine and later restore; and
- compare predicted versus observed action distribution, selected action, value, and return consequences.

Run short-horizon deterministic/replayed tests where possible, plus a smaller number of full rollout interventions to capture delayed effects. Control the environment seed/state when supported. Report when stochasticity prevents a unique causal conclusion.

## 13. Implementation sequence

1. **Inventory:** record exact active branches/configurations and reproduce all current tasks.
2. **Shared instrumentation:** add T1 stable IDs/statistics/logs without behavior change.
3. **Role audits:** verify actor/critic stores, targets, compaction, checkpointing, and target-network alignment.
4. **Stage A matrix:** compare policy-only, value-only, both, and neural baseline with inherited T1 switches under matched budgets.
5. **Gate decision:** freeze and apply the confirmed readiness criteria.
6. **Failure-guided iteration:** return evidence to T1 and re-test only the affected mechanism.
7. **Nonstationarity:** add controlled shift with automatic and human-assisted conditions separated.
8. **Continuous action:** pilot and select a representative suite after Stage A success.
9. **Visual/scale stress:** include MinAtar/Atari only if the scaling question warrants it.
10. **Synthesis for T3:** identify reusable lessons for sequential usefulness, delayed credit, experience cases, and bounded memory.

## 14. Compute plan

- T1 and Stage A should run locally when feasible.
- Use cloud GPU resources for later T2 only when the experimental matrix—pixel observations, long rollouts, multiple roles, seeds, component combinations, interventions, and contemporary baselines—justifies it.
- Size accelerator hours, storage, checkpoints, and transfer from pilots.
- Keep the scientific objective independent of a particular provider or environment package.
- If the final suite changes, substitute a comparable workload and document the preserved question.

## 15. Collaboration boundary

- Xiaomeng Ye leads the T1-T3 agenda and integration.
- Sen He leads/principally supports RL experimental design and evaluation.
- Xiaomeng Ye and Haotang Li currently conduct implementation and development.
- Haotang Li is a current collaborator and preliminary-work contributor; future award participation, effort, and funding are not yet confirmed.

## 16. Risks and fallbacks

- **No role meets the neural margin:** characterize why and retain the NN-kNN core or selected CBR mechanisms as diagnostic/auxiliary components only if matched ablations show value.
- **Actor unstable, critic viable:** make critic/value the primary supported role; keep actor as a boundary result.
- **Critic unstable, actor viable:** invert the emphasis.
- **Both-role interaction unstable:** preserve separate memories and report combined use as an unsuccessful stretch objective.
- **Case selection not causal:** redirect T1 toward representation, scoring, solution/label quality, reuse, or optimizer diagnosis.
- **Continuous control fails:** retain discrete-action claims and use hybrid interpolation/distribution heads only as clearly labeled contingencies.
- **Visual scaling is infeasible:** use MinAtar or another compact representation challenge and report the resource boundary.
- **Human intervention confounds autonomy:** report it only as a separate assistance condition.

## 17. Deliverables

Return the [shared contract](SHARED_EXPERIMENT_AND_DATA_CONTRACT.md) artifacts, plus:

- exact inventory and reproducibility package for all PI-reported current RL tasks;
- actor/critic role and label-mode matrix;
- frozen current NN-kNN reference;
- Stage A gate report;
- inherited-component ablations and mechanism-level failure analysis;
- causal decision traces/interventions;
- selected nonstationarity protocol and results;
- pilot-based recommendation for continuous-action and optional visual suites; and
- a T2-to-T3 memo on usefulness signals, delayed credit, experience-case construction, and limits.

Open choices are tracked in [OPEN_DECISIONS_AND_HANDOFF_CHECKLIST.md](OPEN_DECISIONS_AND_HANDOFF_CHECKLIST.md).
