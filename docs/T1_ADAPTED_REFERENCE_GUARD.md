# Paired maintenance reference experiment — 2026-10-04

`alignment_adapted_reference_guard.yaml` pairs train-LOO and independent
maintenance reference selection with identical initial core, aggregate
regression adapter and both selected Adam states. Candidate optimization uses
only fixed training queries; maintenance queries select removals, never supply
gradients or select initial checkpoints. Per-condition reference routing is
explicit in the resolved condition and maintenance manifest.

Energy/Yacht, seeds 8/9/10, three update scopes and two references yield 36
exploratory runs. Each starts with 18 cases and reaches 12 at loss budget 0.03.
Reference denominators are 18 training queries versus 115 Energy/46 Yacht
maintenance queries. All 18 paired initial states are tensor-identical.

Frozen source: `295c4cea1b8a7b3ecbb4ac2a0b637b8ba5e65da2dae12f5afd825dbdd4c0b361`;
115 files; ZIP SHA256
`b03e1755046b8265388b3332bfaf110347ec50d0dd4d1f118911ec22a55be532`.
Independent replay validates 144 states, 3,888 final events, all 3,306 candidate
and 216 control trainings, exact selected/full/random model and Adam states,
IDs, reference denominators, floors and budgets. Static raw inputs are hashed
separately; actual transformed splits remain in experiment snapshots.

Maintenance selection improves test post-adaptation RMSE in all 18 pairs.
Energy mean improvements across three seeds are 0.04209 (adapter), 0.03739
(alternating), 0.04403 (retrieval); Yacht: 0.03228, 0.03444, 0.02737 respectively.
RMSE units follow each task's training-target standardization. Across all 36
runs adaptive improves over initial/full in 24 and worsens in 12, versus random
improves in 26 and worsens in 10. Random violates the reference budget in 27;
it is a matched-training/capacity control, not a constraint-matched selector.

These are two tasks and three seeds, not 18 independent replications. Repeated
maintenance selection can overfit that split. The new 15% maintenance split
changes training selection and standardization relative to the prior 18-run
pilot; cross-batch differences cannot isolate reference choice. Freeze broader
confirmation and examine selection overfitting and group constraints before
claiming generalization protection. Tests: 151 T1/T3 pass (one existing warning).

Artifacts: `results/t1_pi20260920/continuation_20261004/adapted_reference_guard/`.
Task outputs: `neural-cbr-adapted_reference_guard-20261004-results.md` and
`neural-cbr-adapted_reference_guard-20261004-verification.json`; failed audit
attempts are preserved. No source document has been rewritten.
