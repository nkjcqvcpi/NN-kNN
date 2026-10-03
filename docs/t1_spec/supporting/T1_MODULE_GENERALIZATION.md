# T1 enabling module artifact and bounded portability plan

## Relationship to the unified T1

The PI merged the former T0 mechanism-development target and T1 module-generalization target on 2026-09-09, then narrowed T1 on 2026-09-12. This document now describes the reusable module as an enabling artifact and its bounded portability validation. It is neither a separate proposal thrust nor a separate scientific aim.

## Governing objective and claim boundary

T1's ultimate goal remains an easy-to-use NN-kNN neural module that can be inserted into compatible roles otherwise served by conventional neural-network components while adding inspectable, case-grounded decisions.

"Wherever a neural network can be used" is the long-range vision, not a claim of universal interchangeability. Within representative supervised settings, T1's primary objective is for the full-cycle foundation to replace selected neural predictors with compatible inputs, outputs, objectives, and meaningful case semantics while remaining competitively accurate and adding faithful case-level control. Prior table-level results from the **older, pre-MCB NN-kNN** support this as a plausible hypothesis: the classification paper reports image accuracy within 0.1-0.8 percentage points of the neural point estimates, and the regression paper reports at least one NN-kNN variant with lower mean RMSE than the MLP on 7 of 12 datasets and no more than 3.0% higher on the remainder, including one displayed tie. These are descriptive preliminary results rather than formal equivalence because baseline uncertainty is incomplete and the regression summary selects across variants. They establish a legacy performance baseline, not evidence that MCB or the proposed full-cycle system preserves or improves those results. Augmentation remains a fallback and a way to characterize substitution boundaries, not the default T1 objective. The CAREER project will also prepare interfaces for T2 and T3 rather than assert support for every architecture or task. See [T1_PRIOR_RESULTS.md](T1_PRIOR_RESULTS.md) for exact values and limitations.

The reusable software is an enabling artifact. Its validation supports the central full-cycle claim by showing whether the coordinated mechanisms can be implemented consistently beyond one task-specific code path. T1 will not claim exhaustive generalization across task families and modalities.

## Preliminary foundation and proposed-work boundary

- The IJCAI-2025 classification and IJCAI-2026 regression work establish task-specific feasibility for the older, pre-MCB NN-kNN and provide controlled compatibility testbeds and legacy baselines.
- T1 develops the integrated full-cycle neural CBR mechanisms, including retrieval improvements, reuse, human revision, quality-aware retention, and their interaction with MCB.
- During T1 validation, the project will expose the coordinated mechanisms through a coherent contract and test bounded portability across the selected supervised settings without turning packaging into another independent contribution.

Basic classification/regression substitution, the four-stage mechanisms themselves, and rerunning MCB on one more dataset are not T1 novelty.

## Proposed module contract

1. **Standard learning interface:** accept batched tensor inputs, support differentiable optimization, checkpointing, and integration with learned or frozen feature extractors.
2. **Explicit case semantics:** define what constitutes a case, its stored solution or label, its provenance, and the meaning of its contribution for each supported task family.
3. **Task-adaptable outputs:** expose a common reuse interface while allowing task-specific adapters when output semantics differ.
4. **Inspection contract:** return retrieved cases, activation/contribution weights, relevant case parameters, feature weights, and uncertainty or evidence sufficiency indicators with each evaluated decision.
5. **Intervention contract:** expose authorized T1 revise operations and report immediate and post-adaptation behavioral effects.
6. **Memory contract:** expose the T1 retain policy, fixed budget `K`, protection rules, maintenance logs, and reversible inactive storage through a consistent interface.
7. **Compatibility and fallback contract:** identify unsupported configurations and permit conventional neural or retrieval components when NN-kNN is unsuitable.
8. **Ablation contract:** expose neural adaptation, MCB, quality-aware retention, and enabled T1.2 revision behavior through explicit configuration switches, and store the active combination in experiment metadata.

## Artifact requirements and bounded validation questions

### Interface consistency

Which interfaces and architectural abstractions allow one NN-kNN module to support representative classification and regression roles, feature extractors, and input modalities without changing the underlying T1 reasoning cycle for every task?

### Bounded portability

Under what conditions do the T1 mechanisms transfer across the selected classification/regression settings and one modern transfer benchmark family while preserving task quality, inspectable case semantics, predictable intervention effects, and bounded-memory behavior?

### Limits of substitution

Which task, representation, output, scale, or latency conditions make NN-kNN replacement or augmentation beneficial, neutral, or inferior to a conventional neural component or another retrieval method?

## Representative validation plan

- **Controlled foundation:** rerun the classification and regression benchmarks from the completed pre-MCB work, with interpretable or objectively testable case behavior and controlled corruption, shift, and interventions. First reproduce the old architecture; then compare MCB and full-cycle variants while holding recoverable splits, encoders, tuning budgets, case budgets, and primary metrics constant. Document any irreproducible or changed condition.
- **Modern transfer:** select a structured/tabular benchmark family to test whether the coordinated mechanism and module contract transfer beyond the legacy testbeds. It must support both (a) standard predictive-performance comparison against strong contemporary methods and (b) controlled case-layer tests of bounded memory, shift, poisoning or bad cases, and human correction. Prefer two protocols over the same family; only use a compact two-testbed pairing if no suitable family supports both. The exact datasets remain to be selected.
- **Scale:** matched case budgets and increasing candidate-memory sizes to characterize accuracy, retrieval cost, training cost, and failure points.
- **Downstream interfaces:** prepare the contract for T2 policy/value adapters and T3 retrieval/memory adapters, while leaving substantive RL and text/LLM validation to those thrusts.

The old benchmark suites are committed as rerun targets, subject to recoverability of their data and protocols; the exact structured/tabular modern datasets and any further additions remain provisional. Additional benchmarks will be included only when they address a prespecified claim without reopening T1's scope. The selected modern evaluation must carry both the contemporary performance claim and a substantive test of T1's distinctive case-layer capabilities; neither an accuracy-only leaderboard nor a synthetic-only intervention suite is sufficient by itself.

## Comparison conditions

Compare the general module with:

- task-specific NN-kNN implementations from the completed work;
- matched conventional neural predictors or heads;
- standard k-nearest-neighbor and relevant learned-retrieval baselines;
- task-specific case-memory methods where applicable;
- strong, reproducible contemporary methods for each selected benchmark, chosen through a date-stamped literature and leaderboard review before confirmatory runs;
- versions that omit or replace the common interface components; and
- multiple component combinations at matched case budgets, with the active switches logged and the comparison set chosen to estimate main effects and important interactions.

Use the same benchmark split and primary metric for NN-kNN and contemporary comparators. Match encoders, tuning budgets, parameters, compute, latency, and memory where technically possible; otherwise report the differences explicitly. Evaluate modern relevance as a joint profile of task quality, resource cost, inspectability, intervention behavior, and robustness rather than claiming universal state-of-the-art performance from task accuracy alone.

## Evaluation dimensions

- task performance and calibration;
- retrieval and intervention faithfulness;
- consistency and completeness of the inspection trace;
- preservation of rare and shifted-domain competence;
- transfer of validated T1 behavior across tasks and modalities;
- code and configuration changes needed to integrate a new compatible task;
- training time, inference latency, active and archived memory, and scaling curves;
- reproducibility across seeds, backbones, and task adapters; and
- explicit failure conditions and circumstances favoring conventional neural components.

## Main risks and useful fallbacks

- **Mechanism/portability overload within T1:** keep only the two scientific aims in the full-cycle plan, limit portability to the controlled foundation plus one modern transfer family, and treat software packaging as an enabling artifact rather than intellectual merit by itself.
- **A superficial software-engineering contribution:** make scientific claims about transfer conditions, case semantics, behavioral invariants, and failure boundaries rather than API convenience alone.
- **Task-specific adapters fragment the module:** require a common reuse and evidence contract even when output-specific adapter networks differ.
- **Interpretability does not transfer across modalities:** treat inspectability and faithfulness as empirical outcomes and report settings where cases or weights are not meaningful to intended reviewers.
- **Scaling prevents practical substitution:** characterize the performance-cost frontier and retain augmentation, approximate retrieval, or conventional neural components as explicit fallbacks without weakening replacement as T1's primary supervised hypothesis.

## Dependency into later thrusts

T2 will use the T1 contract to test the full-cycle neural CBR system in policy and value roles after the relevant T1 core criteria are met. Its NN-kNN core will supply the learned case-based mechanism in those roles. T3 will extend the full-cycle system to LLM and agent settings. NN-kNN will retrieve, the host will reuse, human intervention will revise, and case-base maintenance will retain. Success in T1 is not sufficient evidence for RL, LLM, agent, or healthcare performance; each later thrust requires its own task-specific validation.

## Consequential details still needed

- the modern transfer benchmark family;
- the minimum common interface across T1 classification/regression and the downstream T2/T3 adapters;
- the number and type of representative backbones feasible within the bounded T1 validation;
- quantitative portability, integration-effort, scaling, and failure criteria; and
- the boundary between a task-specific reuse adapter and the invariant NN-kNN core.

See [T1_FULL_CYCLE_NNKNN.md](T1_FULL_CYCLE_NNKNN.md) for the unified thrust and [T1_REVISE_RETAIN.md](T1_REVISE_RETAIN.md) for the revise/retain design.
