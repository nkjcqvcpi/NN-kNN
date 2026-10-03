# Modern T1 benchmark strategy

Status: updated 2026-09-12 after the PI selected structured/tabular data as T1's primary modern transfer family. The earlier provisional image, text, MedMNIST, and WILDS matrix is superseded for T1 scope control; those modalities may appear in T2, T3, or later work only when they directly test a thrust-specific claim.

## Recommendation

Use a compact, two-track design:

1. **Legacy rerun track:** reproduce the pre-MCB classification and regression configurations as faithfully as the preserved artifacts permit, then compare matched MCB and full-cycle variants.
2. **Modern tabular track:** use a preregistered TabArena subset for contemporary predictive comparison and controlled case-layer stress tests.

This keeps T1 centered on one scientific question: whether coordinated retrieval, neural adaptation of retrieved solutions, MCB-style stabilization, quality-aware retention, and later human revision can produce a general-purpose NN-kNN module that remains predictively competitive while adding measurable case-level control.

## Proposed compact matrix

| Claim | Benchmark role | Candidate source | Main measures |
|---|---|---|---|
| The full-cycle module preserves and extends prior NN-kNN capability | Completed-paper classification and regression tasks | Recoverable datasets, splits, encoders, and metrics from the archived papers and code | Reproduction gap; task quality; stability; case locality; memory; latency |
| NN-kNN remains modern on structured data | Four to six TabArena classification/regression tasks selected across sample size and feature regimes | [TabArena](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1697e3fb412da11dc9488249f9e7bbc9-Abstract-Datasets_and_Benchmarks_Track.html) | Benchmark-native task quality; uncertainty; compute; inference latency; case-memory size |
| Bounded memory preserves useful competence | Fixed and growing case budgets on the same tasks | Shared controlled protocol | Quality-memory frontier; coverage; redundancy; retained-case utility; update cost |
| Maintenance identifies harmful or useless cases | Label corruption, targeted poisoning, irrelevant/redundant cases, and class- or cohort-skewed case bases | Shared controlled protocol | Detection/ranking; false flags; containment; recovery; rare-case preservation |
| The system responds to domain shift | Prespecified covariate, temporal, label, or mechanism changes where scientifically valid | Natural task metadata when available plus controlled transformations | Pre/post-shift quality; case turnover; recovery time; forgetting; retained useful knowledge |
| Human correction has the intended causal effect | Low-risk, understandable tabular task with semantically meaningful cases/features | Selected after UI and IRB feasibility review | Targeted correction; correct/incorrect decision flips; collateral effects; time and effort |

## TabArena task-selection criteria

Select the exact version and tasks only after a reproducibility pilot. The final subset should collectively include:

- both classification and regression;
- small and medium sample regimes compatible with controlled case-memory experiments;
- numeric and categorical or mixed features;
- at least one task with missingness or imbalance if the benchmark protocol supports it;
- semantically meaningful features for at least one intervention or UI task;
- stable data access, licensing, preprocessing, metrics, and implementation; and
- enough baseline coverage to avoid reimplementing an unbounded leaderboard.

Avoid selecting tasks merely because NN-kNN performs well in exploratory runs. Freeze the subset, splits, metrics, tuning budgets, case budgets, primary contrasts, and exclusion rules before confirmatory comparisons.

## Contemporary baseline roles

| Role | Candidate methods | Boundary |
|---|---|---|
| Classical case reasoning | k-NN and weighted k-NN | Match preprocessing, learned/fixed representation conditions, and case budget |
| Strong tabular non-neural method | At least one gradient-boosted tree implementation, selected from the maintained TabArena protocol | Do not equate a single tree method with the whole tabular state of the art |
| Matched neural predictor | MLP and, where feasible, [TabM](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html) | Match tuning and runtime accounting; TabM is an ensemble-like MLP baseline |
| Tabular foundation-model reference | [TabPFN](https://doi.org/10.1038/s41586-024-08328-6) on tasks within its supported regime | Declare pretrained external compute and hardware asymmetry; do not imply from-scratch compute equivalence |
| Maintenance/influence reference | Feasible subset of inference-time leave-one-out masking, influence functions, Data Shapley, or TracIn | Compare what each score estimates; these may be too expensive for every task and are not identical to case utility |

## Experimental contract

- Preserve the PI-confirmed two-phase legacy rerun: first reproduce the pre-MCB architecture, then add MCB/full-cycle components under matched recoverable conditions.
- Expose and log component switches for neural adaptation, MCB, quality-aware retention, and later human revision.
- Use broader component combinations on inexpensive diagnostics and prespecified, hypothesis-driven combinations on larger tasks.
- Match data splits, metrics, case budgets, tuning opportunities, stopping rules, and compute accounting as closely as the method families allow.
- Report predictive quality alongside case-layer measures. T1 succeeds only if a configuration meets a prespecified task-appropriate competitiveness margin and the required inspectability, correctability, and bounded-memory criteria.
- Use intervention tests to establish whether displayed cases and weights causally affect the decision as predicted.
- Keep protected or rare cases separate from low-use cases; low retrieval is not proof that a case is harmful.
- Date-stamp all “modern” or “strong baseline” selections and recheck them before experiments because TabArena is a living benchmark.

## Scope boundary

- Structured/tabular data is the required modern T1 family.
- Natural images, biomedical images, text classification, WILDS, and other modalities are not part of the required T1 modern matrix.
- Prior image and text tasks may still be rerun when needed for direct compatibility with the published classification paper, but they do not create a new T1 modality commitment.
- Healthcare-specific literature/evidence retrieval belongs to T3, not T1.
- Substantive RL belongs to T2. T1 mechanisms may be tested there through inherited component switches after the local readiness gate.

## Source and rationale note

TabArena was introduced in the NeurIPS 2025 Datasets and Benchmarks Track as a continuously maintained tabular benchmark with curated datasets, model implementations, reproducible code, and a public leaderboard. Its results also show why the comparator set must be plural: gradient-boosted trees remain strong, deep models become competitive with sufficient time and ensembling, and tabular foundation models are especially strong on smaller datasets. This supports using a fixed representative subset with multiple comparator roles rather than claiming generality from one dataset or one baseline.

See `../05-evidence-and-citations/PROPOSAL_LITERATURE_AND_BENCHMARK_REVIEW_2026-09-12.md` for the full literature positioning and T1-T3 benchmark shortlist.
