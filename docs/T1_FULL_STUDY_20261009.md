# Full T1 study execution, October 9, 2026

The confirmed implementation gaps identified in this audit have been corrected.
Full experiments are running. Completion of the entire CAREER research plan,
an accepted general maintenance formula, a real human study, and SOTA on every
dataset are not established by these changes.

## Ownership and handoff

| Step | Workstream / branch / checkout | Intended files | Validation | Completion |
|---|---|---|---|---|
| Correct duplicate-case initializer | Shared helper originating in Neural CBR / `neural-cbr` / `neural-cbr/` | `model/common/core.py`, duplicate identity test | Focused core/identity tests | Commit `4cc1f33`; not propagated |
| Correct calibration and synchronization evidence | Neural CBR / `neural-cbr` / `neural-cbr/` | `model/t1/calibration.py`, `sync.py`, `tools/t1_run.py`, evidence tests | Snapshot reconstruction, component isolation, actual MCB independent/RR/RRR runs | Commit `774b715`; completed |
| Fit preprocessing on training only | Same T1 ownership | `model/t1/data.py`, HPO tests | Numeric and unseen category tests; real access to all four modern tasks | Commit `774b715`; completed |
| Complete regression reuse and task namespaces | Same T1 ownership | `tools/t1_run.py`, runner contract tests | Three actual Energy reuse conditions; wine classification/regression collision test | Commit `774b715`; completed |
| Execute validation-only full-data grid and modern comparators | Same T1 ownership | `tools/t1_grid_search.py`, grid YAML | Five-family Iris preflight; per-trial receipts and frozen selection | Commit `774b715`; full grid running |
| Execute synchronized/independent ablations and all planned automated families | Same T1 ownership | `tools/t1_execute_study.py`, synchronization/control YAMLs | Successful subprocess receipts, manifests and checkpoints required | Iris/synthetic subset completed (160 runs); all-task execution queued after successful grid |
| General maintenance score and scientific gates | Same T1 ownership | Future accepted derivation and stage documentation | Exact selector reduction, matched comparisons, frozen acceptance margins | Unresolved in authoritative plan; no hidden formula selected |
| Human revision study | Same T1 ownership | Existing reviewer UI, protocol, participant evidence | Actual approved participants and matched controls | Pending actual participant/protocol information |
| Standalone core, RL, LLM | `main` / `nnknn/`; `rl` / `rl/`; `LLM` / `llm/` | No stage changes in this task | Validate recipients before any required cherry-pick | Not changed; depend on accepted T1 evidence where applicable |

Absolute T1 checkout: `C:\Users\Administrator\NN-KNN\neural-cbr`.
Proposal source remains read-only at commit
`cd77277600841438d98ac5793867da28b8ec6dd0`. Existing uncommitted changes to
`docs/BRANCH_ORGANIZATION.md` and `docs/XPU_USAGE_GUIDE.md` are preserved.

## Frozen search protocol

`configs/t1/grid_search_20261009.yaml` declares 22 tasks: all seven maintained
small classification tasks, eleven maintained regression tasks, and four modern
TabArena tasks. Synthetic mechanism diagnostics belong to the following T1
experiments, not the benchmark ranking. Image/text studies and T2/T3 belong to
their separate plans and branches.

Every task uses its full data. The four modern tasks are anneal (363614), credit-g
(363626), concrete compressive strength (363625), and healthcare insurance
expenses (363675), chosen for feature/task diversity before inspecting model
performance. Their official OpenML outer train/test indices are retained;
training contains a separate validation holdout. This is a frozen local subset
comparison, not a full TabArena bagging or leaderboard replication.

Each of NN-kNN, kNN, CatBoost, MLP and TabM has 32 Cartesian candidates and two
development seeds, 20 and 21. There are 7,040 screening fits across 110
dataset/family jobs. Neural screening allows 60 epochs with patience 15.
Three finalists are refit for up to 300 epochs with patience 40 on both
development seeds. The winner is frozen before five fresh evaluation seeds,
30–34. When all candidates complete, the complete study has 8,250 fits.

NN-kNN searches representation, top-k, case learning rate, glocal set count and
temperature. Shared learning rates follow the declared ratios. kNN searches
neighbor count, weighting and Minkowski p. CatBoost searches depth, learning
rate, regularization, random strength and iteration count. MLP/TabM search
width, depth, learning rate, regularization and dropout/ensemble size.

All families receive the same number of candidate opportunities, splits and
seeds. Their training costs differ: CatBoost allows 300/600 iterations; neural
models have epoch budgets. This is not equal elapsed-time tuning. Report actual
elapsed time, inference time, query count and memory alongside task metrics.
TabM trains individual heads and averages predictions at inference.

Candidate fitting receives training and validation tensors only. Selection uses
validation error, then validation loss, then configuration ID as a deterministic
tie-break. Test scores never rank candidates. Failed or nonfinite candidates
are recorded; a partial grid returns failure, rather than claiming completion.
No test-result-driven widening of this frozen grid is allowed.

Categorical encoding, numerical imputation/scaling, and regression target scaling
fit training data only. Previously whole-data student preprocessing and body-fat
filtering have been removed from this path. The body-fat task retains its legacy
Density target explicitly; it must not be reported as a new body-fat target
benchmark. Cached remote legacy arrays record SHA256 checksums.

## Planned automated experiment families

After successful grid completion, the pipeline uses each task's frozen NN-kNN
parameters for retention, classification/regression reuse, synchronization and
removal diagnostics. It does not substitute defaults for absent HPO results.

The queue includes fixed-capacity full/random/stratified/B/Q/k-center/coverage
comparisons; reuse output/loss ablations; independent/RR/RRR and individual
synchronization-loss/radius ablations; identical-encoder MCB on/off controls;
MCB retention; exact/cached removal and retraining diagnostics; nominal input
mechanisms; and simulated revision stress factors. MCB controls deliberately
use identical fixed MLP encoders rather than confounding MCB with HPO's choice
of raw versus MLP representation.

Exact removal diagnostics cap memory at 60 training cases; removal-retraining
uses 24 and adapted-retraining 18. These are explicitly small-memory diagnostics,
separate from full-data retention/reuse/synchronization and HPO. Radius
mean/median/shrunk/clipped/cohort-normalized diagnostic configs are available;
fallbacks require primary train-calibration failure evidence and are not silently
selected. Regression cohort normalization without explicitly supplied bins is a
single-cohort control, not evidence of a meaningful between-cohort adjustment.

Large calibration artifacts store frozen model states, labels, biases, stable
IDs, tensor shape/dtype and SHA256 digests, allowing reconstruction with the
bundled source. Matrices totaling at most 64 MiB remain stored exactly. This
reduces disk use without dropping the evidence needed to reproduce pair regions.

## Validation and live execution

Full revised suite: 161 tests passed; after adding the compact-snapshot test,
the calibration suite passed all 15 tests. Import smoke, compilation checks and
`git diff --check` passed. One existing test tensor-to-scalar warning remains.
The new regression reuse runner completed combined/difference-only/final-loss-only
conditions on Energy. The MCB runner completed independent/RR/RRR on Iris,
including maintenance, recalibration and fixed final capacity.

The full-budget Iris synchronization subset completed all 50 expected runs.
Replay reconstructed 60 calibration snapshots and 350 scalar metrics, verified
3,119 epoch diagnostic records, and checked all five source ZIPs against their
fingerprints. Independent/RR/RRR mean post-adaptation accuracies are
0.94000/0.95333/0.96000 at matched final K=68; these descriptive differences do
not establish a scientific gate. The full grid's separate five-seed Iris NN-kNN
accuracy is 0.95333, tied with CatBoost and MLP in this local comparison.

Some primary Iris RRR maintenance checkpoints produced overly broad activation
regions. That train-calibration evidence triggered 50 additional mean/median,
shrink/clipping/cohort-normalization diagnostic runs across RR and RRR. They
completed successfully. The primary method is retained; test outcomes do not
select a replacement radius. The same conditional diagnostic scan is queued
after the full study, using task-role namespaces to distinguish the two wine
tasks.

The combined Iris/synthetic subset completed all 20 dataset/seed jobs and 160
conditions: 100 synchronization runs and 60 reuse runs. Replay reproduced 120
synchronization snapshots, 6,818 epoch records, 700 synchronization metrics and
420 reuse metrics. All 160 canonical manifests match unique aggregate rows.
The radius diagnostics separately reproduced 100 snapshots, 3,817 epoch records
and 350 scalar metrics across 50 runs. Recorded source ZIPs passed integrity and
fingerprint checks. Complete subset evidence is not complete all-task evidence.

Execution began at 2026-10-10 06:37 UTC (October 9 locally). Source fingerprint:
`ec45b8952f21ae4f4804891be3474629cad874b62f3d6f28fe243d13c818f9d4`.
Full-data CPU NN-kNN initialization plus one California Housing epoch took
42.5 seconds; completion time for the complete study is not yet known.
At the latest recorded handoff, 27/110 dataset/family jobs had completed with
zero failed trials or jobs. Use `grid_full_20261009/progress.json` for live counts.

Artifacts under `results/t1_pi20260920/`:

- `grid_full_20261009/`: frozen study/source ZIP, per-trial receipts, selections,
  full data snapshots, evaluation models/predictions and progressive job counts.
- `full_study_20261009/`: hidden pipeline launch PID, grid/study stdout and stderr,
  pipeline exit status once available, and orchestration script.
- `plan_full_20261009/`: generated experiment queue/configs/logs/results after
  successful grid completion.
- `plan_iris_20261009/`: full-budget Iris/synthetic reuse and synchronization
  subset (160 completed runs), launched while other HPO tasks continue; separate from the later full
  queue to keep exact execution scope visible.
- `runner_preflight_20261009/`: bounded real-data regression/MCB runner checks.
- `radius_iris_20261009/`: recorded train-calibration triggers, resolved configs,
  50 successful conditional radius runs and replay evidence.
- `radius_full_20261009/`: conditional diagnostics queued after all-task study
  execution; `full_study_20261009/radius_launch.json` identifies its waiting PID.
- `grid_preflight_20261009/`: completed five-family Iris plumbing preflight with
  its own earlier frozen source; do not combine it with the full-study results.

Resume from this checkout without modifying model/tools/configs/tests/datasets
source during the frozen study:

```powershell
.venv/Scripts/python.exe -u tools/t1_grid_search.py configs/t1/grid_search_20261009.yaml --out results/t1_pi20260920/grid_full_20261009 --workers 4
.venv/Scripts/python.exe -u tools/t1_execute_study.py results/t1_pi20260920/grid_full_20261009 --out results/t1_pi20260920/plan_full_20261009 --workers 4
```

Grid resume rejects changed source/configuration. Completed trials/jobs are
reused; failed trials are retried. The automatic pipeline runs the second command
only if the first completes successfully. Final reports must state completed and
failed trials, actual capacity, inference/training cost and paired fresh-seed
results. No SOTA claim or scientific gate has yet been established.

The queued diagnostic scan also writes `grid_full_20261009/performance_report.json`
from complete five-seed evaluations and `plan_full_20261009/canonical_results.json`
from individual manifests/metrics, after experiment subprocesses have exited.
These aggregate canonical artifacts rather than treating an interim shared
summary file as completion evidence. Historical 120-run pilots were rechecked
with the current compatible replay script and still reproduce all recorded
snapshots and metrics.
