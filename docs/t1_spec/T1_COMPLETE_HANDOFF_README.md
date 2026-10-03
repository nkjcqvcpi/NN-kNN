# Complete T1 collaborator handoff

Status: package manifest compiled 2026-09-12. This package includes the current T1 implementation plan, the earlier detailed technical handoff discussed with the PI, and the supporting design/evaluation notes needed to avoid losing technical context.

## Short answer

`T1_FULL_CYCLE_IMPLEMENTATION_PLAN.md` is the current reorganized consolidation of the T1 implementation discussion. It is comprehensive, but it is **not a verbatim copy** of the earlier handoff.

`supporting/T1_IMPLEMENTATION_HANDOFF.md` is the original 819-line collaborator-oriented technical handoff developed during the detailed discussion. It retains lower-level implementation anchors, diagnostics, configuration fields, tests, baselines, and open questions. The colleague should receive and consult both.

## Required reading order

1. `T1_T3_OVERALL_ROADMAP.md`
   - Current T1-T3 thesis, dependencies, work packages, gates, five-year progression, and immediate collaborator sequence.
2. `T1_FULL_CYCLE_IMPLEMENTATION_PLAN.md`
   - Current consolidated T1 specification and primary implementation guide.
3. `supporting/T1_IMPLEMENTATION_HANDOFF.md`
   - Earlier, more granular technical handoff; use for code anchors, exact diagnostics, configuration surface, test cases, and implementation questions.
4. `SHARED_EXPERIMENT_AND_DATA_CONTRACT.md`
   - Stable identities, case/retrieval/maintenance schemas, matched-comparison rules, causal interventions, reproducibility, and governance.
5. `OPEN_DECISIONS_AND_HANDOFF_CHECKLIST.md`
   - Items the collaborator must not silently assume and the artifacts to return to Xiaomeng Ye.

## Supporting T1 notes

- `supporting/T1_FULL_CYCLE_NNKNN.md`
  - Scientific architecture, four-stage CBR mapping, synchronization logic, evaluation, risks, and relationship to T2/T3.
- `supporting/T1_REVISE_RETAIN.md`
  - Provenance, case bias, geometric trustworthiness, action authority, and M0/M1/M2 human-correction evaluation.
- `supporting/MCB_SYNTHESIS.md`
  - MCB mechanism and claim boundaries from the under-review manuscript.
- `supporting/MCB_BENCHMARK_PILOT.md`
  - Initial plan for applying MCB to NN-kNN's completed benchmark families.
- `supporting/MODERN_BENCHMARK_STRATEGY.md`
  - Modern transfer benchmark and comparator-selection rationale.
- `supporting/T1_PRIOR_RESULTS.md`
  - Exact extracted pre-MCB classification/regression results and proposal-safe interpretation.
- `supporting/T1_MODULE_GENERALIZATION.md`
  - Earlier packaging/generalization note. Its engineering portability ideas remain relevant, but its treatment of module generalization as a separate sub-aim is superseded. In the current architecture, packaging and portability are enabling T1 deliverables, not a separate scientific aim.

## Authority and conflict rule

The package contains overlapping documents because the purpose is to preserve all technical detail, including the earlier handoff. If wording differs, use this order:

1. PI decisions represented in `T1_T3_OVERALL_ROADMAP.md` and `T1_FULL_CYCLE_IMPLEMENTATION_PLAN.md`;
2. the shared contract and open-decision register;
3. `supporting/T1_IMPLEMENTATION_HANDOFF.md` for lower-level detail consistent with the current plan;
4. other supporting notes for rationale, evidence, and historical context.

Do not restore the former T0 as a separate thrust. T0 was merged into T1. Do not treat module generalization as a third T1 aim. Do not treat proposed formulas, benchmarks, or thresholds marked unresolved as established results or approved constants.

If a lower-level instruction appears to conflict with the current consolidated plan, stop and ask Xiaomeng Ye rather than silently selecting one.

## Essential current design decisions

- T1 implements a full neural CBR cycle around NN-kNN: retrieval, neural adaptation of retrieved solutions, human revision, and retention.
- Revise is disabled during the first technical implementation, but stable IDs, archives, and logs must support it later.
- MCB is a cross-cutting representation-stability mechanism, not a fifth CBR stage.
- The immediate bottleneck hypothesis is poor case selection/use; it remains a hypothesis requiring causal testing.
- Classification reuse preserves the new aggregate, retrieved-label-conditioned NN-CDH architecture and uses explicit nominal input differences only when the shared representation does not already include them.
- The initial classification loss combines residual MSE and final-label cross-entropy.
- Maintenance now compares Q or a C/H variant, B alone, coverage-to-reachability, and case-removal influence. Do not combine Q and B. Read the [2026-09-20 candidate note](../T1_CASE_MAINTENANCE_CANDIDATES.md); older exported handoff archives may retain the superseded combination.
- Selecting the active set also needs combined coverage/redundancy, exposure, and protected-case checks. No final comprehensive score is selected.
- `k=5` initializes the default case bias only. The later free-correction radius is calibrated after core training from a direct frozen snapshot of trained per-case biases and the matching learned distance.
- Human correction uses M0 before correction, M1 immediately after correction without retraining, and M2 after controlled retraining.
- Completed classification/regression results are pre-MCB. Reproduce/reconstruct them first, then compare matched MCB and full-cycle variants.
- T1 uses component switches and matched ablations. It must preserve the original implementation as a selectable baseline.

## Code reference boundary

The earlier code inspection used NN-kNN commit:

```text
c09719576b3519e9878764190773916cd9ce82e6
```

inspected on 2026-09-08. The collaborator must record the actual starting branch/commit, read the current repository `AGENTS.md` and `HANDOFF.md`, and re-resolve all file/line anchors. Do not assume the inspected snapshot is still current.

## First implementation assignment

The first work package is not the whole five-year T1 plan. It is:

1. reproduce the current baseline;
2. add behavior-neutral stable case IDs and provenance/contribution logging;
3. implement one common maintenance-policy interface;
4. prototype fixed-`K` supervised case selection with reversible archives;
5. compare against random/downsampling, stratified, full-memory, and current bias-pruning conditions;
6. add the classification reuse extension as an independent switch;
7. integrate MCB as a separate factor only after retention works independently; and
8. return exact commits, configurations, tests, results, event logs, failures, and unresolved choices.

The detailed files govern implementation. This manifest is navigation, not a substitute for them.
