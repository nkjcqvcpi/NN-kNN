# MCB cross-benchmark pilot

## Purpose and status

This is a planned immediate, pre-award classification experiment, not a completed result and not the whole CAREER T1. Its purpose is to integrate MCB into the shared NN-kNN core and test whether the stability effects reported on CUB-200-2011 generalize across datasets, modalities, case-base interventions, and distribution shifts.

The PI proposed this direction on 2026-09-04. Code for the MCB paper has not yet been supplied to the CAREER workspace, and no pilot run has been executed here.

## Benchmark philosophy

The headline question is not simply whether MCB raises accuracy on another old dataset. It is whether a slowly adapting reference encoder provides a stable but sufficiently adaptive case memory across different data types and realistic changes to the data or case base.

The pilot therefore uses four evidence layers:

1. **Backward compatibility:** Confirm that adding MCB does not break established NN-kNN behavior on a small representative subset of earlier benchmarks.
2. **Cross-modal generality:** Test representative modern tabular, natural/biomedical image, and text-embedding classification settings.
3. **Shift and adaptation:** Measure in-distribution and out-of-distribution behavior using naturally occurring or carefully controlled shifts.
4. **Case-memory stress:** Apply the same resampling, addition, removal, reweighting, label-noise/poisoning, and corrective-intervention protocol across tasks.

This claim-oriented structure is more important than running every available dataset.

## Selected scope and legacy compatibility inventory

The PI selected a classification-only first pilot on 2026-09-04. Regression is intentionally deferred until after the classification integration is stable.

The archived repository snapshot documents the following supported tasks. This inventory is not yet confirmed as identical to the published IJCAI-2025 benchmark suite. These tasks are compatibility checks, not the headline evidence for modern generality.

### Classification

- Small/tabular: `iris`, `zebra`, `zebra_special`, `wine`, `breast_cancer`, `balance`, and `digits`.
- Images: `mnist`, `cifar10`, and `svhn`.
- Deferred in the current workflow: SST-2 and SST-5.

### Regression - excluded from the first pilot

The current Table 1 workflow documents `califonia_housing`, `diabets`, `abalone`, `body_fat`, `airfoil`, `car`, `student_performance`, `yacht`, `energy_efficiency`, `bike_sharing`, and `wine`. The first two spellings are existing repository identifiers and should not be silently changed in experiment commands.

## Modern generalization candidates

The exact subset remains provisional and should be fixed after a compute and implementation check.

- **Tabular classification:** Use a small, stratified subset of TabArena, the NeurIPS-2025 living benchmark, selected to vary dataset size, feature types, class count, and imbalance. OpenML-CC18 remains a fallback if TabArena integration or task licensing is unsuitable.
- **Biomedical image classification:** Use a representative 2D subset of MedMNIST v2. It is standardized and lightweight enough for repeated stability/intervention experiments and connects to the provisional healthcare direction. It must be described as a research benchmark, not as evidence of clinical utility; MedMNIST explicitly states that it is not intended for clinical use.
- **Natural-image classification:** Retain CUB-200-2011 as the known MCB anchor and use CIFAR-100 or a comparably manageable many-class image task to test whether effects persist outside fine-grained birds. The final choice requires a matched encoder and compute estimate.
- **Text classification:** Use a small English classification subset from MTEB with a trainable contemporary text encoder. This creates a direct methodological bridge from core NN-kNN to later retrieval/RAG research; retrieval and reranking tasks remain T3 rather than part of this classification-only pilot.
- **Real distribution shift:** Select one or two feasible WILDS classification datasets rather than claiming the full suite. A healthcare-related dataset such as Camelyon17 is a candidate only after storage, compute, variance, data-use, and domain-expertise checks; it should not be the sole shift dataset.

## Staged execution sequence

1. **Integration smoke tests:** `zebra` and `zebra_special`, because their synthetic boundaries support direct neighborhood inspection, followed by one fast real-data check.
2. **Backward-compatibility sample:** Select a small representative subset of prior tabular and image benchmarks; do not make a full legacy sweep the headline result.
3. **Modern cross-modal sample:** Run a fixed subset spanning tabular, biomedical/natural image, and text classification.
4. **Shift and case-memory stress tests:** Apply a shared intervention protocol and at least one feasible real-world distribution-shift benchmark.

This sequence is an agent-developed execution plan under the PI-approved classification-only scope. It may be adjusted after the MCB code, exact published benchmark configuration, compute limits, and dataset terms are reviewed.

## Minimum comparison design

- Compare the original shared-encoder NN-kNN core with an MCB-enabled core under matched data splits, case bases, seeds, training budgets, encoders, and retrieval rules.
- Separate the effect of memory-encoder EMA from other architectural differences. If MCB is ported from the paper code, first reproduce a small known configuration before expanding the benchmark set.
- Predefine the momentum coefficients and tuning budget; report tuning separately from final evaluation.
- Use multiple seeds and retain run-level outputs rather than only best-run summaries.
- Measure task performance, representation change, retrieved-neighborhood churn, computational overhead, and case-base resampling sensitivity where applicable.
- Use a common case-memory intervention grid where valid: resample the case base; add valid new cases; remove influential cases; change or learn case weights; introduce controlled label noise or malicious cases; and apply corrective removal/reweighting. Measure whether the resulting behavioral changes are predictable and reversible.
- For shifted data, report in-distribution and out-of-distribution task quality, stability, and adaptation rather than averaging them into one score.
- Record conditions in which MCB reduces instability, has little effect, or prevents useful adaptation. A negative or mixed result should refine the T1 hypothesis rather than be hidden.

## Candidate outcomes

- A reproducible MCB implementation in the shared NN-kNN core.
- A matched classification table of predictive performance and computational cost.
- Representation-drift and neighborhood-churn figures across benchmark families.
- A cross-task case-memory stress profile showing sensitivity and recovery after controlled interventions.
- A decision about which MCB mechanisms and diagnostic measures should transfer into the RL experiments.

## Proposal role

If completed before CAREER submission, the pilot can provide preliminary evidence of cross-benchmark generality and identify the conditions that motivate T1. The proposed CAREER research must still advance beyond this pilot toward controlled, quality-aware, and adaptive case memory under human intervention, unreliable or poisoned cases, domain shift, sequential decision-making, and later LLM/agent settings.
