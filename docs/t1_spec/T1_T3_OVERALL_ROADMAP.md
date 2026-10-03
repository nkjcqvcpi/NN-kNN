# Full-cycle neural CBR CAREER research and implementation roadmap: T1-T3

Status: collaborator-facing roadmap compiled 2026-09-12 from PI-confirmed decisions. It distinguishes completed foundations, active preliminary work, proposed research, and intentionally unresolved choices.

## 1. Long-range goal

The project will develop **case-grounded, inspectable, and correctable AI** around NN-kNN. The long-range goal is an easy-to-use NN module that can serve in selected places where a conventional neural network is used, while exposing the cases and feature/case weights involved in its output and allowing people to correct or maintain its case knowledge.

The project does not assume that NN-kNN should replace every neural component. It tests where full replacement is compatible, where augmentation or hybrid use is preferable, and what performance, efficiency, and control trade-offs result.

## 2. Central thesis

Coordinating retrieval, reuse, revision, and retention in one NN-kNN-based neural CBR system—supported by MCB-style representation stabilization—can preserve the practical learning advantages of neural methods while making decisions more inspectable, correctable, and maintainable under bounded memory. In T1, a neural adaptation network implements reuse. Its loss permits small task-specific corrections and penalizes unnecessarily large changes.

The three thrusts form a dependency path with feedback:

```text
T1 — Full-cycle, general-purpose NN-kNN
  Build and validate the shared case-grounded foundation.
              |
              v
T2 — Full-cycle neural CBR for reinforcement learning
  Test the complete system in policy/value roles, sequential learning, and domain shift.
              |
              v
T3 — Full-cycle neural CBR for existing LLMs and agents
  Scale the complete cycle to need-conditioned knowledge and experience memory.

T2/T3 failures -> diagnose the shared mechanism -> bounded T1 refinement
               -> revalidate the affected downstream setting
```

This is not a rigid waterfall. T1 and T2 are already active at the same time: current RL limitations motivated renewed T1 core development. Later work feeds evidence back without turning T1 into an open-ended catch-all.

## 3. Thrust overview

| Thrust | Main research question | Primary contribution | Minimum success | Main fallback |
|---|---|---|---|---|
| **T1: Full-cycle, general-purpose NN-kNN** | Can NN-kNN coordinate retrieval, neural adaptation of retrieved solutions, human revision, and retention with MCB stabilization while preserving competitive performance and competence under bounded memory? | A better NN-kNN core with quality-aware case selection, controlled adaptation magnitude, stabilization, causal correction, and a reusable interface | At matched budgets, improve over random/downsampled and current pruning; approach full-memory NN-kNN with fewer cases; remain within a prespecified margin of matched modern baselines; improve causal traceability/correction | Characterize which components and domains benefit; preserve retrieval-only or conventional neural alternatives where preferable |
| **T2: Full-cycle neural CBR for reinforcement learning** | When can the T1 full-cycle neural CBR system serve as a policy model, value model, or both while keeping decisions case-inspectable and editable? | A coordinated 4R system for RL, with NN-kNN as its learned case core, T1 maintenance, sequential utility, and domain-shift response | At least one T1-enhanced configuration improves over the frozen current NN-kNN RL reference and reaches a prespecified neural-competitiveness margin in one major role, with faithful causal case traces | Use role-specific or hybrid neural/full-cycle configurations and report the suitability boundary |
| **T3: Full-cycle neural CBR for LLMs/agents** | Can an existing host LLM/agent participate in a full CBR cycle that retrieves original knowledge and experiences by current need and downstream usefulness, through prompt augmentation, internal model integration, or both? | NN-kNN modules retrieve different kinds of knowledge. The host reuses them, human intervention revises them, and case-base maintenance retains them. The system supports iterative retrieval, inspectable and feedback-trainable memory, and local/global scopes. | Competitive answer/task quality relative to standard retrieval plus material gains in causal traceability, correction, and harmful-memory containment/recovery | Retain prompt augmentation, one-shot retrieval, narrower artifact types, or the RAG result if internal or agent integration is too confounded |

Exact numerical margins and minimum improvements will be selected after pilots and frozen before confirmatory runs.

## 4. T1 roadmap — improve and complete the NN-kNN core

### Scientific basis

Completed pre-MCB classification and regression work demonstrates that NN-kNN can sometimes approach conventional neural baselines while exposing case/feature contributions. Prior NN-CDH work supplies neural case adaptation and retrieval-adaptation synchronization ideas. The under-review MCB work supplies a promising two-timescale representation-stability mechanism. None of these alone establishes the proposed full-cycle system.

### T1.1 technical core

1. **Instrument retrieval without changing behavior**
   - stable case identities across compaction;
   - per-case retrieval count, activation, contribution, bias, and provenance;
   - common logs and maintenance interface across task roles.
2. **Implement quality-aware bounded retention**
   - compare Q or a C/H variant, B alone, coverage-to-reachability, and case-removal influence, following the 2026-09-20 candidate note; do not combine Q and B;
   - keep utility, redundancy, coverage, rarity, and protection separate;
   - select the best active case set under fixed capacity `K`;
   - archive all removals reversibly.
3. **Add bounded classification reuse**
   - preserve the newer aggregate, retrieved-label-conditioned NN-CDH design;
   - use nominal query-minus-neighborhood differences only when the shared representation does not already contain those nominal inputs;
   - train with residual MSE plus final-label cross-entropy;
   - report retrieval-only and post-adaptation outputs separately.
4. **Integrate and evaluate MCB**
   - compare MCB/no-MCB only after retention works independently;
   - measure representation, neighborhood, and selection stability as well as task quality;
   - test whether stabilization assists accumulated case statistics or suppresses needed change.
5. **Synchronize components**
   - retrieval should select nearby cases that need little correction;
   - adaptation should solve the task without drastically overriding retrieval;
   - retention should preserve useful, trustworthy, diverse coverage;
   - train in stages or alternating updates with every loss reported separately.

### T1.2 human revision

After the core trace and archive mechanisms pass, add an interface through which an authorized person can inspect, add, correct, force-activate for a controlled decision, reweight, protect, quarantine, remove, or restore cases and adjust applicable feature weights.

Evaluate:

- `M0`: trained model before correction;
- `M1`: immediate post-correction model without retraining; and
- `M2`: correction followed by controlled retraining.

Use a no-correction retraining comparator only when retraining itself is part of the experiment. Measure intended corrections, correct/harmful flips, collateral effects, persistence, calibration, performance, time, and reviewer burden.

### T1 evidence and benchmark progression

1. Synthetic tests with known relevance, corruption, redundancy, rare cases, and shift.
2. Reproduce or transparently reconstruct the completed pre-MCB classification/regression protocols.
3. Compare MCB and full-cycle variants under matched recoverable splits, encoders, case budgets, tuning budgets, and metrics.
4. Add one selected modern structured/tabular transfer family supporting both performance comparison and controlled maintenance/correction stress tests.
5. Compare random/downsampling, stratified selection, current bias pruning, provenance-only, bias-only, combined trustworthiness, utility/redundancy, coverage-aware retention, full memory, and strong contemporary predictive baselines where fair.

The detailed formulas, configurations, tests, gates, risks, and deliverables are in `T1_FULL_CYCLE_IMPLEMENTATION_PLAN.md`.

## 5. T2 roadmap — full-cycle neural CBR for reinforcement learning

### Role matrix

Evaluate:

- neural actor + neural critic;
- full-cycle neural CBR actor, built around an NN-kNN core, + neural critic;
- neural actor + full-cycle neural CBR critic, built around an NN-kNN core;
- full-cycle neural CBR actor + full-cycle neural CBR critic with distinct memories; and
- a clearly ablated hybrid/parallel neural contingency if needed.

Begin with decision-level cases: state/context plus action recommendation for the actor, and state/context plus declared value target for the critic. Consider trajectory fragments later when individual decisions lack sufficient reusable context.

### Current preliminary work

The PI reports work on CartPole, Acrobot, LunarLander, and MinAtar Breakout, with Atari Pong in progress. NN-kNN underperforms existing RL methods in some domains. Exact roles, branches, versions, baselines, metrics, seeds, and results have not yet been supplied, so these facts motivate investigation but do not establish a competitive result or a causal diagnosis.

### Stage A readiness gate

On two or three inexpensive state-based tasks, require one T1-enhanced configuration to:

1. improve over a frozen current NN-kNN RL reference under matched interaction, compute, memory, and tuning budgets;
2. reach a prespecified margin of a matched neural baseline in at least one policy/value role;
3. pass multiple-seed evaluation;
4. expose the cases/weights actually used; and
5. pass removal/replacement/reweighting intervention tests.

Only after this gate should the project commit to expensive scaling.

### Later progression

1. Compact spatial observations through MinAtar where informative.
2. Representative continuous-action tasks from MuJoCo or DeepMind Control after a pilot.
3. Optional Atari/ALE visual scaling when it tests a necessary question at feasible cost.
4. Controlled changes in dynamics, observations, or rewards.
5. Continual/online learning and automatic maintenance as the primary shift response; active/selective review and explicit human intervention as separate conditions.

Measure return, sample efficiency, stability, case traces, intervention effects, memory/latency, post-shift degradation/recovery, case turnover, and preservation of useful prior competence.

T2 helps T3 by developing sequential usefulness and delayed-credit methods. It does not imply that an LLM is merely an RL policy.

## 6. T3 roadmap — full-cycle neural CBR for LLMs and agents

### Architecture boundary

Use an existing open-weight LLM for both T3 integration modes. In prompt augmentation, freeze the host and let a project-controlled orchestrator pass the LLM's explicit information need to NN-kNN. In model integration, connect NN-kNN case activations or retrieved representations to a declared internal interface. Compare prompt-only, internal-only, and combined conditions under matched budgets where feasible. A strong API model may support only the prompt-augmentation external-validity condition unless it exposes the required internal interface. The host LLM performs reuse; no NN-CDH adaptation network rewrites retrieved T3 artifacts.

### NN-kNN retrieval for different kinds of knowledge and information

Candidate artifact families include:

- evidence/context;
- tools/capabilities;
- procedures/demonstrations/skills;
- prior actions, experiences, successes, failures, and corrections; and
- structured relations, formulas, rules, and constraints.

Compare separate NN-kNN modules with a shared backbone plus type-specific heads. The LLM may request one or more types or `mixed/unspecified`; raw scores across types require calibration.

A useful case need not share a label with the query. It may provide tangent knowledge or a bridging relation—for example, a calculus reference needed for a derivation or a county-to-state relation needed to verify a birthplace answer. Use downstream counterfactual utility, not same-label matching, as the long-range target.

### Iterative retrieval

1. Host generates an explicit need/subquestion.
2. NN-kNN retrieves compact original artifacts with stable IDs and provenance.
3. Host applies them and may generate the next need.
4. Stop on answer readiness, no useful novelty, a duplicate request, or a hard round/context/latency/compute limit.

Maintain two linked views of the same event: a minimal host evidence channel and a full human audit channel containing match, activation, bias, feature/case contributions, provenance, reliability, scope, and intervention history.

### Feedback and memory lifecycle

- Use objective outcomes plus optional overall and case-level feedback.
- Treat missing ratings as unobserved, not negative.
- Use contextual bandits for immediate isolated choices and RL for delayed multi-round outcomes where appropriate.
- Collect feedback immediately but use validated, versioned batch updates as the primary global update path.
- Maintain bounded temporary session memory automatically.
- Create persistent user memory only after an explicit save/retention instruction.
- Keep authorized domain and validated global memories separate.
- Jointly rank eligible local/global cases with calibrated scores and a learned local-relevance prior; local context is often relevant but does not automatically outrank stronger global evidence.

### Evaluation progression

1. Bounded single-hop QA/RAG integration test; KILT is a candidate pilot.
2. Multi-hop complementary retrieval; HotpotQA or MuSiQue are leading alternatives.
3. Public biomedical literature/evidence retrieval; historical BioASQ Task b is the leading candidate, pending license/corpus/metric verification.
4. Transfer demonstrated mechanisms to memory in an existing bounded agent framework.

The required biomedical experiment uses objective public benchmark ground truth and no PHI. It evaluates both evidence retrieval and the evidence's causal effect on the final answer. Expert evaluation is contingent on securing qualified expertise and approvals. No clinical deployment or patient-benefit claim is planned.

### Agent-specific payoff

If successful, NN-kNN would give an existing agent an experience memory that can learn which evidence, tools, skills, procedures, and successful or failed actions are useful; supply them through prompt context or internal model computation; add or correct memories without retraining the entire base LLM for each lesson; expose which memories influenced actions; and permit targeted quarantine/removal of harmful memories. These are hypotheses requiring comparison with the same starting host using no retrieval, standard memory, prompt-augmented NN-kNN, and model-integrated NN-kNN.

## 7. Five-year implementation path

| Project year | Primary work | Gate or transition |
|---|---|---|
| **Year 1** | T1.1 instrumentation, retrieval, neural adaptation, MCB, retention, and fixed-`K` maintenance; revise disabled initially | Use active RL failures to refine requirements; pass behavior-neutral instrumentation and supervised mechanism gates |
| **Year 2** | T1.2 human revision, causal correction, bounded UI study, packaging, legacy reruns, and modern transfer | Expand T2 only as inherited T1 mechanisms pass readiness criteria |
| **Year 3** | T2 discrete-role evaluation, shift, continuous-action progression, and selected scaling | Begin T3 frozen-host single-hop and multi-hop RAG work; feed shared limitations back to T1 |
| **Year 4** | T3 public biomedical evidence retrieval and answer-level grounding | Begin bounded existing-agent memory integration after general retrieval mechanisms work |
| **Year 5** | Agent experience memory, feedback learning, poisoning/stale-memory recovery, human intervention, and synthesis | Complete targeted T1/T2 revalidation and documented reusable releases |

Years overlap. T1 remains maintained across the award, but later T1 work must respond to a documented downstream limitation and revalidate the affected settings.

## 8. Cross-thrust experimental rules

- Preserve the current implementation as a selectable baseline.
- Log exact code/data/environment/model versions and every active component switch.
- Use stable case identities, reversible maintenance, and linked retrieval/intervention events.
- Keep predictive/task quality, retrieval, reuse, retention, efficiency, stability, traceability, correction, and resistance/recovery as separate outcome families.
- Match case, interaction/training, tuning, seed, compute, retrieval, and context budgets wherever applicable.
- Freeze task-specific competitiveness margins and minimum improvements before confirmatory runs.
- Use direct case/feature intervention to test explanation faithfulness.
- Do not treat low exposure or missing feedback as evidence of harm.
- Do not let downstream adaptation or the host LLM hide poor retrieval; report pre-reuse outcomes and causal effects.
- Treat MCB as under-review work and all earlier classification/regression results as pre-MCB.

## 9. Immediate collaborator sequence

1. Record the live NN-kNN repository branch/commit and the exact active RL branches/configurations.
2. Implement behavior-neutral stable case IDs, statistics, and common logs.
3. Pass the T1 instrumentation gate.
4. Prototype fixed-`K` supervised retention and causal audits.
5. Implement the classification reuse extension as an independent switch.
6. Reproduce/reconstruct pre-MCB benchmarks, then run matched MCB/full-cycle variants.
7. Apply the validated T1 mechanisms to current RL tasks and evaluate the Stage A gate.
8. Select later continuous-action/visual tasks from evidence, not in advance.
9. Begin T3 only when the retrieval interface, trace, and maintenance mechanisms are usable.

## 10. Companion file

For implementation, read this roadmap together with:

- `T1_FULL_CYCLE_IMPLEMENTATION_PLAN.md` — exact T1 architecture, equations, classification adapter, trained-bias free-correction rule, provenance/trustworthiness score, retention policy, switches, phases, tests, gates, baselines, risks, and deliverables.

Unresolved scientific choices must remain visible and be returned to Xiaomeng Ye for decision rather than silently fixed by the implementation.
