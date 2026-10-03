# T3 implementation plan: full-cycle neural CBR for LLMs and agents

Status: PI-confirmed architecture and staged evaluation plan. No full-cycle neural CBR LLM, agent, or healthcare result has been supplied; all T3 capabilities are hypotheses to implement and test.

## 1. Objective and integration boundary

T3 asks whether a full-cycle neural CBR system built around NN-kNN can be integrated with an existing LLM or agent in two complementary ways. It may return original cases as prompt context or tool output. It may also participate in the internal computation of an open-weight model. Both modes retrieve artifacts that match the current information or action need. They must improve downstream outcomes while making behavior more inspectable, editable, and robust than the same host with no retrieval or standard retrieval/memory.

The host neural system supplies:

- language generation and general parametric knowledge;
- general reasoning or “intuition”;
- planning and observable task state;
- tool orchestration and authorization; and
- the query or subquestion sent to NN-kNN.

The full-cycle system divides responsibilities clearly:

- **Retrieve:** NN-kNN supplies query-conditioned case retrieval and exposes learned case and feature contributions.
- **Reuse:** the host LLM or agent interprets and uses the retrieved material.
- **Revise:** an authorized person can correct, reweight, quarantine, or remove cases and test the behavioral effect.
- **Retain:** quality-aware maintenance uses usefulness and reliability histories to manage bounded local, domain, and global case memory.

T3 must implement and compare two integration modes:

1. **Prompt augmentation:** NN-kNN returns original cases through the prompt or a structured tool result.
2. **Model integration:** NN-kNN case activations or retrieved representations enter a declared internal interface in an open-weight host.

Candidate internal interfaces include retrieval-conditioned attention, hidden-state fusion, and output-level gating. The exact interface and trainable parameter scope remain `[NEEDS INPUT / EMPIRICAL SELECTION]`. Where feasible, also test a combined condition that uses both prompt and internal integration.

NN-kNN only operates after a query is given. It does not independently invent the information need, replace the host model, or become the full planner. The project will not train a new foundation model or build a new general-purpose agent.

## 2. Expected benefit for agents

An ordinary LLM agent may already use context, tools, and stored state. T3 must show a more specific benefit than “the agent retrieves text.” If successful, the full-cycle neural CBR system would give an existing agent an explicit, learnable experience and knowledge memory that can:

- retrieve evidence, tools, procedures, skills, prior successes, failures, corrections, and constraints according to the current need;
- learn which cases and artifact types help from task outcomes and user feedback rather than only topical similarity or static labels;
- incorporate a new case or correction into memory without retraining the entire base LLM for every lesson;
- use retrieved cases as prompt context, as inputs to internal neural computation, or both;
- expose which memories, matches, case biases, provenance histories, and feature/case weights affected the context supplied to the agent;
- allow a person to add, revise, suppress, quarantine, or remove harmful, obsolete, poisoned, or misleading memories and test the behavioral effect; and
- maintain bounded temporary, user, domain, and validated global memories while controlling redundancy, stale knowledge, scope leakage, and poisoning.

This implements the PI's working analogy: the host's neural competence provides intuition, while NN-kNN supplies explicit knowledge and experience that intuition can repeatedly consult. The analogy motivates the architecture; it is not a cognitive-model claim.

The benefit is conditional. The evaluation must show competitive task quality, acceptable latency and memory cost, causally faithful traces, effective correction, and recovery from harmful memory. Visible memory or successful task completion alone is insufficient.

## 3. No case-adaptation network in T3 memory

T3 maps the CBR cycle differently from T1 supervised reuse:

- **Retrieve:** different NN-kNN modules or retrieval heads select original stored artifacts for different kinds of knowledge and information.
- **Reuse:** the existing LLM interprets and applies the artifacts.
- **Revise:** an authorized person adds, edits, reweights, suppresses, quarantines, restores, or removes an artifact or scoped preference.
- **Retain:** NN-kNN records outcome/usefulness evidence and maintains the bounded memory.

Do not apply the NN-CDH adaptation network to retrieved T3 artifacts. Tokenization, transport formatting, declared context truncation, or an explicitly labeled derived summary is not learned case adaptation when it preserves a stable link to original content. The primary host channel should return original content or minimally necessary structured records so evidence remains inspectable.

## 4. Prompt-augmentation and model-integrated connections

### Prompt augmentation

NN-kNN remains project-controlled and does not need to run inside an API provider. A lightweight orchestrator exposes a structured operation such as:

```text
retrieve_cases(
    need,
    requested_types,
    already_retrieved_ids,
    constraints,
    observable_task_state
)
```

The loop is:

1. Provide the user task, fixed instructions, and retrieval schema to the frozen host.
2. Receive an explicit human-readable information need/subquestion and structured retrieval request.
3. Run NN-kNN over eligible session, user, domain, and global memories in the project's environment.
4. Return selected original artifacts, stable IDs, types, provenance, and essential reliability markers.
5. Let the host answer, act, or issue another bounded request.

The orchestrator carries observable state, enforces budgets and stopping rules, packages results, and detects duplicate/near-duplicate queries. It should not add an unobserved learned planner/query generator in the primary comparison.

### Model integration

The internal mode requires an open-weight host. NN-kNN still receives a query produced by the host. Its case activations or retrieved representations then enter a declared internal interface. Each contribution must remain linked to stable case identifiers, provenance, and retrieval scores.

Do not treat arbitrary host fine-tuning as evidence that internal NN-kNN integration works. Start all internal comparisons from the same host checkpoint. Declare which NN-kNN and host parameters are trainable. Match task data, training opportunities, and compute budgets. Compare prompt-only, internal-only, and combined conditions where feasible.

### Host strategy

- **Prompt-augmentation control:** one frozen open-weight host checkpoint on budgeted cloud GPU compute.
- **Model-integrated condition:** the same starting open-weight checkpoint with a declared NN-kNN interface and trainable parameter scope.
- **Combined condition:** prompt augmentation plus internal integration, where feasible.
- **External validity:** repeat selected prompt-augmentation experiments with a strong API LLM using the same retrieval contract.

Keep the case base, full audit data, feedback learning, maintenance, and user-scoped state outside the API. Send only the necessary query and selected case payload. An API host cannot support the model-integrated condition unless the provider exposes the required internal interface. Log provider/model identifier, prompt, tool schema, decoding/tool settings, payload, output, timing, token use, and cost. For internal experiments, also log the interface, trainable parameters, case activations, checkpoints, and compute. Do not pool frozen-host, trainable-host, and changing API-host results.

## 5. NN-kNN retrieval for different kinds of knowledge and information

Cases may represent distinct artifact families:

1. factual evidence, reference knowledge, or context;
2. tool descriptions, capabilities, requirements, or prior tool-use records;
3. procedures, workflows, demonstrations, or reusable skills;
4. experiences, trajectories, actions, outcomes, failures, or corrections; and
5. structured/symbolic relations, formulas, rules, constraints, or procedures.

These are a provisional taxonomy. A request may need one type or a complementary mix. Different types may need different case representations, learned distances, biases, eligibility rules, and maintenance policies. Tool retrieval should measure capability/applicability; evidence retrieval should measure support for the factual/reasoning need; experience retrieval should measure transfer to the current state and likely outcome.

Compare:

- **Separate NN-kNN modules:** an independent representation, metric, bias, case bank, and maintenance process for each family.
- **Shared representation with different retrieval heads:** shared components where transfer helps, with a different retrieval head or metric and potentially a separate case bank for each family.
- **Single shared metric:** ablation showing whether specialization is necessary.

Raw scores from different types are not automatically comparable. Calibrate them before joint ranking.

### Type routing

The frozen host emits one or more requested types, or `mixed/unspecified`. A transparent controller calls the declared retrievers under one fixed total budget and records which ran.

Compare:

- oracle type where benchmark annotations permit it;
- frozen-host explicit type selection;
- mixed retrieval across eligible heads under a matched total budget; and
- a trained router or fine-tuned type selector only as a later separate extension.

Evaluate downstream utility first. A surprising type can be useful; type-label agreement alone is not success. Also report missed-needed type, unnecessary routes, cross-type complementarity, calibration, latency, context, memory, and parameter cost.

Retrieving a tool record never authorizes tool execution. The host's ordinary permission boundary remains controlling.

## 6. Useful cases need not share a label

T3 must not carry classification's same-label quality definition forward. A useful case may be topically different or lack any target label, yet supply a necessary fact, relation, method, example, tool, or prior experience.

Examples:

- A mathematical derivation may need a calculus textbook passage or worked example. Its value comes from enabling the derivation, not sharing a label with the question.
- A birthplace question may retrieve a direct place fact plus a city/county-to-state and state-to-country relation when the requested granularity or verification requires them. Supporting relations can be useful without containing the final answer.

The learned distance must therefore represent compatibility between the **current need** and what an artifact can contribute:

```text
current need = query + goal + observable subtask/state + available context

candidate = direct evidence | relation | hierarchy | definition | example |
            rule | procedure | tool record | correction | experience

good retrieval = compact individual/complementary case set that improves
                 a prespecified downstream outcome
```

Do not accumulate every related item. Measure marginal utility, set coverage, complementarity, redundancy, provenance, latency, and context cost.

## 7. Iterative reasoning-retrieval loop

One-shot top-`k` retrieval is a required baseline. The proposed primary mechanism iterates:

1. The host inspects the observable task state and emits an explicit information need or subquestion, requested type(s), constraints, and already supplied case IDs.
2. Appropriate NN-kNN heads return a compact set of novel eligible artifacts.
3. The host reuses them, updates an observable partial answer/task state, and either answers/acts or emits another need.
4. A hybrid stopping policy stops when:
   - the host declares answer-ready/sufficient evidence;
   - no sufficiently useful novel eligible case is found;
   - a query repeats or nearly duplicates an earlier request; or
   - a hard round, context, latency, or compute budget is reached.

Log original query, explicit subquestions, cases, weights, provenance, reliability, observable partial state, continuation/stopping decision and reason, and final outcome. Do not require or store private chain-of-thought.

Risks include self-reinforcing retrieval, confirmation loops, duplicated context, and resource growth. Candidate safeguards are contradiction-seeking/diverse retrieval, source-quality gates, novelty checks, and hard budgets. Test them; do not assert that they work in advance.

Compare no retrieval, one-shot top-`k`, and iterative retrieval under matched host, prompts, decoding, tools, data, context budget, and compute. If iterative retrieval uses more retrieval opportunities, report quality-versus-resource curves as well as matched-budget results.

## 8. Two causally linked output channels

Each retrieval event creates:

### Host evidence channel

- original artifact content or stable minimal record;
- stable case ID;
- artifact type;
- source/provenance; and
- only the reliability marker necessary for correct use.

### Human audit channel

- observable query/need/subquestion;
- all candidate/selected case IDs;
- raw/calibrated distance or similarity;
- activation and case bias;
- case/feature weights or contributions;
- source and usage provenance;
- `Q_i`, `B_i`, `T_i`, exposure, evidence sufficiency, and disagreement;
- scope and lifecycle;
- maintenance, feedback, and intervention history; and
- model/retrieval snapshots and event IDs.

Both derive from the same retrieval computation. Full diagnostics should not be inserted into the host prompt unless that is an explicit treatment, because metadata can consume context and change behavior.

## 9. Match, reliability, and usefulness are separate

### 9.1 Match

`M_i(q)` represents how strongly case `i` addresses the current query under the learned metric and activation for that kind of knowledge. It is query-specific.

### 9.2 Reliability/trustworthiness

Transfer the maintenance method supported by T1. Its 2026-09-20 direction compares Q or a C/H variant, B alone, coverage-to-reachability, and case-removal influence. The earlier Q/B combination is superseded; see [the candidate definitions](../T1_CASE_MAINTENANCE_CANDIDATES.md). T3 still needs task-specific outcome and loss definitions.

Keep source quality, retrieval count, activation mass, disagreement between `Q_i` and `B_i`, review state, protection, and harm flags separately visible whichever candidate is used.

Initial human-readable status:

- **Approved/high-confidence:** sufficient positive outcome evidence, acceptable trustworthiness, adequate source provenance, and no unresolved quarantine/harm issue.
- **Uncertain:** insufficient evidence, provenance/bias disagreement, unresolved source quality, or intermediate score.
- **Quarantined:** explicitly excluded by an authorized reviewer after a serious concern; a score alone should not silently quarantine a consequential case.

Quarantined cases are ineligible regardless of match. A highly reliable low-match case is irrelevant to the current need. A high-match uncertain case remains uncertain.

General research benchmarks may test including uncertain cases with explicit markers. In biomedical evaluation, uncertain material is a review candidate and cannot be treated as authoritative without appropriately qualified review.

### 9.3 Conditional marginal usefulness

Let `S_q` be the retrieved set and `J(q,S)` a validated lower-is-better downstream loss:

```text
u_i(q; S_q) = J(q, S_q without i) - J(q, S_q)
```

- `u_i > 0`: helpful under this query/set;
- `u_i < 0`: harmful;
- `u_i approximately 0`: ignored, irrelevant, redundant, or unnecessary—not automatically false/bad.

This direct counterfactual is preferred because usefulness may depend on other cases. Use sampled counterfactuals or a learned estimator only after validating it against direct removal on an audit subset.

`J` is task-specific: answer correctness, evidence quality, derivation validity, tool success, constraint satisfaction, agent reward, or qualified expert judgment. LLM self-judgment alone is not sufficient.

### 9.4 Provenance generalized beyond classification

```text
C_i = sum_q a_i(q) * max(u_i(q; S_q), 0)
H_i = sum_q a_i(q) * max(-u_i(q; S_q), 0)

Q_i = (C_i + s) / (C_i + H_i + 2s)
```

Keep retrieval count and activation mass as exposure. Preserve low-frequency cases that enable rare capabilities; a calculus case can be essential even if seldom retrieved.

## 10. Training the retriever and type policy

Distinguish three objectives:

1. **Need match:** whether the artifact addresses the current need.
2. **Outcome utility:** whether including it improves answer/action/reward.
3. **Inspectable proximity:** whether the learned feature/distance explanation coherently identifies why it matches.

Positive evidence can come from objective successful outcomes, direct counterfactuals, qualified judgments, or successful trajectories. Negative evidence can come from irrelevant, misleading, poisoned, stale, or outcome-degrading cases. Evaluate individual and set-level utility.

Use feature weighting and other learned distance components consistently across retrieval, auditing, maintenance, and explanation so the system has one coherent geometry.

## 11. Feedback-driven learning

When static usefulness labels are absent, collect:

- objective answer/task outcomes;
- optional overall user ratings;
- optional per-case useful/not-useful/harmful/uncertain ratings;
- explicit corrections, removals, and reweightings; and
- qualified expert judgments for specialized domains.

Use a two-level interface:

- overall answer/task feedback gives evidence about the retrieved set or trajectory;
- optional case-level feedback helps assign credit among artifacts.

Do not require rating every case. Missing feedback is unobserved, not negative. Link every rating to event ID, case ID where applicable, model/policy version, requested type, observable context, rater role/expertise when appropriate and consented, and outcome.

### Learning horizon

- Use a contextual bandit as the natural starting point when one routing/case-set choice receives prompt feedback.
- Use RL when multiple requests and case sets form a sequence with delayed final reward.
- T2 contributes sequential credit-assignment methods, but not every retrieval update requires RL.

### Update authority

Collect consented feedback immediately, but make validated batch updates the primary global parameter-update path. Before promotion:

- require sufficient evidence and feedback-quality checks;
- test poisoning/manipulation susceptibility;
- evaluate target and unaffected tasks on held-out data;
- version the new snapshot; and
- verify rollback.

Immediate per-feedback parameter learning is limited to controlled low-risk/sandbox conditions. One user's rating cannot prove global truth or harm. An authorized person may make an immediate logged case intervention; that is direct control, not automatic learning.

Possible update targets are the type policy, the retrieval metric or head for each kind of knowledge, case biases, provenance histories, and maintenance priorities. Keep overall and case-level feedback distinct when combining them.

## 12. Local and global memory

Use four explicit scopes/lifetimes:

1. **Temporary session memory:** automatic, bounded working memory for the current task; expires by default.
2. **Persistent user memory:** saved only after explicit user retention intent; remains user-scoped across sessions.
3. **Authorized domain memory:** governed group memory with defined membership and validation.
4. **Validated global memory:** shared case base updated through aggregated, diverse, quality-checked promotion.

Scope and lifetime are separate: persistent user memory can be long-lived while still local in access.

### 12.1 Temporary session-memory construction

Automatically retain a compact task record containing only:

- current question/goal and observable subgoals;
- relevant user-supplied facts, constraints, and preferences;
- artifacts supplied to the host and observable used/ignored/outcome status;
- resulting actions/outcomes; and
- corrections or feedback.

Do not store every raw exchange, irrelevant personal data, or private chain-of-thought. Derived compact summaries remain traceable. Clear at the task/session boundary unless the user explicitly saves selected minimally sufficient material.

### 12.2 Persistent local-case construction

Create persistent local cases only from explicit actions such as marking material useful and asking to retain it, asking to save an interaction, or supplying a correction to remember.

Handle events distinctly:

1. Rating an existing global case creates a local feedback/priority reference to that case, not a duplicate.
2. New/corrected knowledge creates a local knowledge case preserving original content/source.
3. An explicitly saved reusable interaction creates a local experience case: need, relevant context/preconditions, evidence/tool/procedure/action, outcome, and feedback.
4. A preference/restriction becomes scoped configuration, not an artificial CBR case.

Every provisional local case records stable identity, scope, type, observable need/context, original content/reference, outcome, feedback, provenance, host/retrieval versions, time, consent/access/expiration/deletion data, and reliability state. Local creation never promotes a case to domain/global memory.

### 12.3 Joint local/global ranking

Apply reliability, quarantine, authorization, and domain-safety gates before ranking. Then jointly rank eligible local and global candidates using:

- calibrated cross-scope match/usefulness;
- reliability evidence; and
- a learned query-dependent local-relevance prior.

Local memory is often more relevant but is not automatically dominant; strong global evidence may outrank weak local content. Merge a local feedback reference with its underlying global case. Expose scope, raw/calibrated scores, local-prior contribution, and final rank.

Compare global-only, session memory, explicit persistent memory, rigid local-first, fixed local bonus, and jointly calibrated ranking under one total budget. Measure personalization, cold start, cross-session retention after explicit save, expiration, deletion, global interference, justified global-over-local overrides, cross-scope leakage, poisoning containment, and promotion quality.

Local restrictions cannot release global quarantine, authorize tools, or override domain safety. Global promotion requires separate validation and provenance review.

## 13. Symbolic knowledge

Symbolic knowledge belongs in T3 unless a compatible exploratory T2 encoding later proves useful. Editable neural case/feature weights are structured control but not symbolic rules.

Compare possible interfaces:

1. **Structured cases:** retrieve a rule, formula, relation, constraint, or procedure as an original inspectable artifact.
2. **External constraints/validators:** filter evidence, check an answer, or allow/reject/revise an agent action.
3. **Differentiable rule penalties:** soft neural preference for rule-consistent behavior; optional and potentially less directly auditable.

Start with structured cases and external validation because they preserve explicit editability. Exact representation, execution, conflict resolution, and task are unresolved. Symbolic artifacts are only one family; tangent knowledge, supporting facts, tools, and experiences may be equally important.

## 14. Development sequence and benchmarks

### Stage 1 — basic general-domain integration

Use a bounded single-hop QA/RAG task to validate:

- host-generated query -> NN-kNN retrieval -> answer;
- stable IDs, original content, and two-channel trace;
- match/reliability/usefulness separation;
- removal/replacement effects; and
- no retrieval versus standard one-shot versus NN-kNN.

KILT is a candidate pilot because it jointly scores output and provenance against a fixed Wikipedia snapshot. It is not yet selected.

### Stage 2 — primary iterative complementary retrieval

Use a multi-hop task requiring multiple supporting facts. HotpotQA and MuSiQue are leading alternatives. Test whether evolving host-generated needs retrieve distinct, complementary evidence more effectively than one-shot top-`k` under matched budgets.

### Stage 3 — required biomedical transfer

Use an established public biomedical literature/evidence benchmark with objective ground truth. A historical BioASQ Task b release is the leading candidate because it includes gold articles/snippets and exact/ideal answers, but release access, corpus, license, executable metrics, and future reproducibility must be verified before selection.

The required study must measure both:

- evidence retrieval/coverage; and
- the effect of that evidence on the frozen host's final answer.

Report them separately and jointly. A relevant document the host ignores is not useful grounding; a correct answer with wrong/irrelevant evidence is not case grounding. Include case removal, replacement, suppression, or correction to test the evidence-to-answer causal link.

MIRAGE/MedRAG is a possible end-to-end external-validity condition. PubMedQA is a simple bounded pilot but insufficient as the only retrieval study. A contemporary iterative biomedical system can be used as a comparator/design reference when reproducible.

Core biomedical work uses no PHI and does not depend on expert recruitment. Manual ideal-answer or case-correction studies are supplementary and require qualified expertise, governance, and applicable institutional approval. Do not claim clinical deployment, diagnosis, patient benefit, or model-developer authority to validate medical correctness.

No healthcare collaborator is currently confirmed. The PI's existing Nanyang Technological University collaboration may eventually provide a connection, but names, medical expertise, interest, availability, and roles were intentionally left unconfirmed. Do not assign the collaboration a healthcare role without explicit evidence and PI approval.

### Stage 4 — existing-agent memory

Transfer only demonstrated mechanisms to an existing agent framework and bounded task. Candidate cases include prior states, decisions, tool calls, procedures, corrections, outcomes, successes, and failures.

Begin with decision records; add trajectory fragments only when needed. Evaluate whether the agent:

- retrieves the right artifact type and experience for the state;
- improves success/reward/sample efficiency;
- learns useful retrieval from delayed outcomes/feedback;
- exposes memory-to-action influence;
- incorporates a new correction externally;
- recovers after harmful/stale/poisoned memory; and
- preserves bounded cost and scope controls.

LongMemEval or LoCoMo may support later memory-lifecycle evaluation but are not first milestones.

## 15. Primary comparisons

Use the same starting host checkpoint, task data, and evaluation protocol. Match resource and training budgets within each valid comparison. Compare:

- no retrieval;
- standard semantic one-shot top-`k` RAG;
- prompt-augmented NN-kNN retrieval;
- model-integrated NN-kNN retrieval;
- combined prompt and internal NN-kNN integration where feasible;
- usefulness-trained NN-kNN retrieval;
- one-shot versus iterative NN-kNN;
- one shared metric versus separate modules versus shared representation components with different retrieval heads;
- static/offline versus feedback collection without update versus validated batch updates;
- immediate online updating only in sandbox;
- global-only versus temporary/user/domain/global scoped memory designs; and
- appropriate contemporary retrieval/memory methods.

The evidence payload is the intended treatment and may differ. All other differences must be declared and budgeted.

## 16. Evaluation outcomes

### Task and evidence

- answer/task correctness and uncertainty;
- supporting-evidence precision/recall/coverage where available;
- successful tool selection/use or agent reward;
- constraint/derivation validity;
- quality versus retrieval/context/compute budget.

### Retrieval and memory

- marginal and set utility;
- useful, harmful, ignored, irrelevant, redundant, and complementary rates;
- per-type and cross-type utility and routing error;
- scope calibration and local/global ranking behavior;
- representation, case-bank, context, and parameter cost;
- rounds, premature/unnecessary continuation, duplicate queries, latency, and token cost;
- memory growth, retention, expiration, save/delete correctness, and actual reuse.

### Causal traceability and correction

- change after removing/replacing/reweighting the displayed case;
- predicted versus observed answer/action effect;
- target correction rate and correctness;
- collateral changes on unaffected tasks;
- persistence after validated updating; and
- user/expert burden.

### Safety and robustness

- stale/poisoned/misleading case detection;
- quarantine precision/recall where ground truth exists;
- recovery and rollback;
- manipulation susceptibility;
- cross-user/domain leakage and harmful local-to-global promotion;
- behavior with uncertain evidence; and
- preservation of rare capabilities.

### Feedback learning

- feedback coverage, disagreement, and calibration to counterfactual/qualified judgment;
- credit-assignment accuracy;
- learning/recovery curves and efficiency;
- exploration and user burden;
- rollback frequency and unaffected-task change; and
- overall-only versus overall-plus-optional-case feedback under matched interactions.

## 17. Joint success standard

T3 succeeds only if it remains statistically competitive with standard RAG/memory on task quality while materially improving all three:

1. **Traceability:** displayed cases/weights identify artifacts that causally affect the output/action.
2. **Human correctability:** authorized case interventions produce intended changes with limited collateral effects.
3. **Resistance/recovery:** the system identifies, contains, or downweights harmful/poisoned cases and recovers better than standard retrieval/memory.

The biomedical stage must meet prespecified evidence and final-answer criteria; one alone is insufficient. Freeze task-appropriate practical-equivalence/non-inferiority margins, minimum improvements, seeds, and analyses before confirmatory runs. Visible citations are not sufficient proof of faithfulness.

## 18. Safety and claim boundaries

- Retrieved content is evidence/context, not automatically trusted instruction.
- Match, frequency, activation, or host attention is not proof of usefulness.
- Usefulness can be query- and set-conditional; do not erase rare capabilities with a global average.
- Quarantined cases are excluded from normal context regardless of match.
- Controlled poisoning occurs only in isolated experiments.
- Biomedical uncertainty requires qualified review before authoritative use.
- Local memory cannot override global safety, quarantine, or tool authorization.
- Do not disclose private chain-of-thought or store unnecessary personal data.
- No current result demonstrates the proposed LLM/agent benefits.

## 19. Risks and fallbacks

- **Host ignores cases:** preserve evidence scoring, use causal removal/replacement, and report that retrieval alone did not ground the answer.
- **Iterative loop self-reinforces:** use one-shot fallback, hard budgets, novelty/duplicate detection, contradiction/diversity tests, and source/reliability gates.
- **Separate modules fragment the system too much:** use shared representation components with specialized heads or a smaller empirically supported taxonomy.
- **Shared metric underfits types:** retain separate modules and calibrate cross-type scores.
- **Direct utility is too expensive:** audit a sample directly and validate an estimator; never rely only on host self-rating.
- **Feedback is sparse/noisy/adversarial:** keep parameters fixed until validated batches, preserve local scope, and require rollback/held-out checks.
- **Local memory leaks or dominates:** strengthen gates, calibration, and scope controls; keep global-only/temporary-only baselines.
- **Biomedical expert access is unavailable:** complete the objective public benchmark study and omit the contingent expert extension.
- **Agent stage confounds too many components:** keep the RAG result as the primary T3 mechanism and use a bounded agent task with the host fixed.

## 20. Deliverables

Return the [shared contract](SHARED_EXPERIMENT_AND_DATA_CONTRACT.md) artifacts, plus:

- versioned provider-neutral retrieval schema and orchestrator;
- case schemas for different kinds of knowledge, routing, retrieval metrics or heads, and score calibration;
- linked host/audit outputs and causal intervention suite;
- single-hop and multi-hop matched comparisons;
- usefulness/counterfactual estimator validation;
- feedback, update-promotion, and rollback protocols;
- temporary/persistent/domain/global memory lifecycle tests;
- verified biomedical dataset/corpus/license/metric package and evidence-answer results;
- bounded agent-memory comparison emphasizing the specific NN-kNN benefits; and
- a failure analysis separating retrieval, routing, reliability, host reuse, feedback credit, scope, and maintenance.

Open choices are tracked in [OPEN_DECISIONS_AND_HANDOFF_CHECKLIST.md](OPEN_DECISIONS_AND_HANDOFF_CHECKLIST.md).
