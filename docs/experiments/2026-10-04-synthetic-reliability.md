# Synthetic factor reliability and matched revision

The old diagnostic fixed corruption, overlap, redundancy, rare-case coverage and
test shift, with one shared random stream. Changing label corruption could
therefore change later validation/test observations. The new explicitly selected
`independent_v1` generator separates seven random streams. Original default
`legacy` tensors remain byte-identical on seeds 5/6/7. Duplicate cases inherit
both their source's observed label and its corruption annotation; their clean
reference label remains separately stored. Validate factors before generation.
Train-only scaling is preserved, so coverage/redundancy changes can change
scaled holdout coordinates even when raw holdout observations are identical.

Snapshots now save safe tensor truth/group masks, clean labels, source IDs and
resolved generator metadata. Revision reports each actual M0/M1/M2/MC predictor
on in-domain, shifted, rare and boundary groups; empty groups have a count and
no invented accuracy. Boundary means the two nearest raw relevant-space center
distances differ by at most the declared margin, excluding rare rows. It is a
diagnostic overlapping subgroup, not a disjoint class or protection default.
Review budgets outside the active case base now fail explicitly.

## Frozen exploratory grid

`alignment_synthetic_reliability.yaml`: seeds 8/9/10; reference, increased
overlap, 40% base noise, 180 duplicates, zero rare training cases, larger test
shift, and combined overlap/noise/shift. Four flaggers (Q, B, random, simulated
oracle), budgets 10/30, 30 core epochs with validation selection, five fixed
continuation epochs at 0.1 learning-rate scale. M2 and unchanged MC use matching
optimization/RNG budgets within each condition. Increasing the number of
training cases increases steps per epoch: cross-scenario compute is not matched.
The new reference is a different generator version and must not be compared
directly with old legacy seed results.

All 168 runs and 672 stage checkpoints pass independent prediction, aggregate
and subgroup metric/count replay. All 21 data snapshots safely load and reproduce
all six data tensors exactly; parent audit separately verifies observation
pairing for noise and truth equality `observed != clean_reference`. M0/MC have
original targets; M1/M2 have reviewed corrections. Final checkpoints equal M2,
Adam increments match the fixed budget, and same-scenario/seed M0/MC agree across
review conditions. Source ZIP reconstructs SHA256
`b1b7f33d1d7bc001a786b88959005449064f98cf0a5273a7222346df66179c84`;
all manifests report unchanged source. The full T1 suite passes 93 tests.

## Observed tradeoffs

Budget-30 mean M2-minus-MC accuracy across three seeds:

| Scenario | Q | B | Random | Oracle |
|---|---:|---:|---:|---:|
| Reference | +0.0013 | 0.0000 | 0.0000 | +0.0103 |
| Overlap | -0.0038 | +0.0051 | 0.0000 | +0.0051 |
| Noise | +0.0282 | +0.0103 | +0.0141 | +0.0385 |
| Redundant | +0.0026 | +0.0026 | +0.0038 | +0.0154 |
| Uncovered rare | approximately 0 | 0.0000 | +0.0013 | +0.0064 |
| Shift | +0.0090 | 0.0000 | 0.0000 | +0.0090 |
| Combined | +0.0256 | -0.0026 | +0.0128 | +0.0256 |

Q identifies more corrupted rows than random in these means, but this is not
uniform predictive benefit. In overlap, Q precision is 0.20 versus random 0.10,
yet Q's matched accuracy change is negative. Correct diagnosis, immediate
correction, adaptation and collateral changes remain different outcomes. In
noise Q precision is 0.4778 and matched accuracy improves by 0.0282 on average;
this suggests a diagnostic setting worth further testing, not confirmed Q
superiority. Oracle precision 1 does not imply perfect task outcomes.

Next debugging should separate final-outcome credit from corruption detection,
inspect recurring case groups, and compare fixed adapted budgets with explicit
training-only constrained acceptance. Preserve no-edit continuation, report
rejected treatments and costs, and never select edits or hyperparameters using
test accuracy. Broader retention/reuse matrices remain pending.

Raw evidence: `results/t1_pi20260920/continuation_20261004/alignment_synthetic_reliability`.
The chat outputs contain per-seed/stage/subgroup tables, replay JSON and figures.
Three seeds, seven explored scenarios and simulated oracle repairs provide
descriptive engineering evidence; they are neither confirmation nor a human
study. No statistical significance or universal benefit is claimed.
