# PI alignment continuation — 2026-10-02

Authority: career-2027 `cd77277600841438d98ac5793867da28b8ec6dd0`.
Implementation: initial alignment `77ccd50`, independent removal retraining and
Adam isolation `b91fd70`. The numerical settings remain exploratory choices.

## Changes and reproducible experiment scope

`clone_optimizer` deep-copies Adam state tensors, preventing one independent
continuation from changing another's starting moments. `retraining.py` scores
each eligible removal after two fixed training epochs, forks every candidate and
unmodified-memory control from the same current model/optimizer/RNG, refreshes
after each accepted deletion, and enforces cumulative reference loss against
the original memory. Reference queries use stable-ID LOO; test/validation labels
do not select removals or retraining checkpoints. The first branch supports
retrieval-only classification/regression and rejects adapters explicitly.

| Experiment | Cases before/target | Seeds | Conditions | Runs |
|---|---|---|---|---|
| alignment_retraining | 24/18 | 2,3,4 | frozen full/random/removal; matched full/random; retrained removal | 36 |
| alignment_candidates_small | 60/45 | 2,3,4 | full/random/Q/B/ratio/full-removal/cached-removal | 42 |
| alignment_adapted_candidates | 60/45 | 2,3,4 | full/Q/B/ratio/full-removal/cached-removal with frozen trained adapter | 18 |
| alignment_digits | 1077/538 | 2,3,4 | full/random/Q/B | 12 |

Full-memory controls intentionally retain the original capacity. Fresh comparison
means do not mix earlier post-selection runs affected by optimizer aliasing.
Raw artifacts are under `results/t1_pi20260920/continuation_20261002/`.
The companion `2026-10-02-pi-alignment-runs.csv` indexes all 108 manifests and
their primary metrics, actual capacity, budgets, commits and source hashes.

## Validation

45 tests pass, plus the import smoke check. All 108 manifests have complete
required artifacts and finite metrics. All 108 checkpoint prediction replays pass
at 1e-6, including the 18 classification adapters. All removal treatments reach
their target capacities and accepted cumulative loss stays within the pilot
budget 0.05. Data snapshots load with `weights_only=True`.

Four ZIP source bundles pass integrity and fingerprint reconstruction. All runs
use source SHA256
`bc3d496f49b0799113e3f876d63e946a7bcce94608a410dd3a12ddb37d0cdccd`;
none reports a source change during execution. The first 36 manifests identify
the pre-commit dirty implementation based on `77ccd50`; the following 72 identify
clean code at `b91fd70`. The shared source hash, not the earlier base commit alone,
identifies the executed implementation.

## Observed results and limits

Retrained removal reaches 18 cases in six rounds: 12 accepted-model epochs and
258 total candidate epochs per dataset/seed, plus 12 per-round control epochs.
Accepted epochs are already included in the candidate total. Iris accuracy does
not improve over frozen removal on these seeds; energy_efficiency RMSE is worse
on average by 0.002545, with mixed per-seed direction. Larger candidate-search cost
has not established a stable gain. Same-capacity comparisons may retain different
case IDs; equal epoch budgets do not imply equal total search computation.

Digits half-memory accuracy ranges from 0.958333 to 0.986111. Q and B each average
0.974074, compared with random 0.970370 and full memory 0.977778. Q/B beat random
on two of three seeds and lose on the third. No near-chance collapse is observed
in this bounded repeat; this is not a robustness guarantee or superiority claim.

Full-memory pre-selection Q/B Spearman averages are 0.4133 (small regression),
0.4850 (small iris), 0.5270 (adapted iris), and 0.5454 (digits). The digits evidence
fraction averages 0.6493 under the declared evidence rule. These diagnostics do
not establish score/ranking equivalence or the reliability of credit assignment.
The maximum observed cached/full influence error is approximately 0.001510;
no acceptable-error threshold has been approved, and these runs include full
cache validation overhead, so they do not establish a net speedup.

The run reports provide individual seeds, paired differences and full audits.
No confirmatory significance/equivalence test was conducted. Adapted retraining,
larger grids, rare/boundary/shift studies, reviewer interventions/UI, regression
synchronization, general-score derivations, modern benchmarks and RL/T3 integration
remain pending.

## Commands

```powershell
.venv\Scripts\python.exe tools\t1_run.py configs\t1\alignment_retraining.yaml --out results\t1_pi20260920\continuation_20261002
.venv\Scripts\python.exe tools\t1_run.py configs\t1\alignment_candidates_small.yaml --seeds 2 3 4 --out results\t1_pi20260920\continuation_20261002
.venv\Scripts\python.exe tools\t1_run.py configs\t1\alignment_adapted_candidates.yaml --seeds 2 3 4 --out results\t1_pi20260920\continuation_20261002
.venv\Scripts\python.exe tools\t1_run.py configs\t1\alignment_digits.yaml --seeds 2 3 4 --out results\t1_pi20260920\continuation_20261002
```
