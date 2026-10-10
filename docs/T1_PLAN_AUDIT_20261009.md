# Neural CBR alignment audit — 2026-10-09

The implementation is not a completed realization of the entire CAREER plan.
The supervised mechanisms exist, including alternating synchronization, but
several experiment, evidence, and research requirements remain incomplete.
This audit corrects synchronization implementation/evidence gaps and reruns
bounded pilots. It does not redefine the scientific plan or choose an unresolved
general maintenance formula.

## Authority and branch boundary

Authoritative source: `../career-2027/03-research-architecture/implementation-handoff/T1_FULL_CYCLE_IMPLEMENTATION_PLAN.md`,
sections 5–15, plus the September 20 decisions and candidate definitions in
`../career-2027/DECISIONS.md` and `T1_CASE_MAINTENANCE_CANDIDATES.md`.
Verified proposal HEAD: `cd77277600841438d98ac5793867da28b8ec6dd0`.
Starting implementation HEAD: `5f97411108c4c47d27707c4d95ec7b9f056bca9c`.

All implemented steps below belong to Neural CBR / T1, branch `neural-cbr`,
checkout `C:\Users\Administrator\NN-KNN\neural-cbr`.
Implementation commits: `4cc1f33` (shared initializer correction) and `774b715`
(T1 evidence, preprocessing, runner and experiment orchestration). Existing changes in `docs/BRANCH_ORGANIZATION.md` and
`docs/XPU_USAGE_GUIDE.md` were preserved. The proposal repository was read only.

## Findings and completed revisions

| Requirement / step | Files | Status and validation |
|---|---|---|
| Retrieve using learned geometry and stable case identities | `model/common/geometry.py`, `model/nnknn_model.py` | Existing mechanism; stable-ID/LOO tests pass. Shared code unchanged in this audit. |
| Aggregate classification/regression reuse; nominal fields only where needed | `model/t1/reuse.py`, `model/t1/nominal.py`, `model/nn_cdh.py` | Existing mechanism; combined-loss, nominal, regression and information-channel tests pass. |
| Separate Q/B; activation-weighted final-outcome C/H; removal uses final task loss | `model/t1/provenance.py`, `candidates.py`, `maintenance.py`, `retention.py` | Existing current-plan semantics; maintained tests pass. No new general-score formula is asserted. |
| Component synchronization | `model/t1/sync.py`, `tools/t1_run.py` | Already implemented as independent/RR/RRR. The plan explicitly asks for staged or alternating updates; simultaneous parameter updates are not required. |
| Every loss before/after a stage update | `model/t1/sync.py` | Added fixed train-only LOO diagnostics in evaluation mode before and after each epoch update, after checkpoint maintenance, and all validation terms. Existing minibatch terms remain available. |
| Reproducible trained-bias calibration | `model/t1/calibration.py`, `tools/t1_run.py` | Added frozen biases, actual learned-distance matrix, labels, stable IDs, pair mask and model state at the selected core checkpoint and every declared maintenance recalibration. Prior sum/squared-sum metadata could not reconstruct the regions. |
| Independent calibration storage | `model/t1/calibration.py`, `tests/test_t1_sync_plan_evidence.py` | Artifact replay exposed a CPU label-storage alias: compaction changed historical labels. Fixed by cloning; mutation and replay tests added. The first two 60-run batches are invalid calibration evidence and retained separately. |
| Required synchronization ablations | `configs/t1/alignment_sync_plan_20261009.yaml`, `alignment_sync_plan_matched_20261009.yaml` | Added individual removal of L_pre/L_near/L_delta/L_small, zero-radius penalty, initial-bias-region and exact-kth-pair radius alternatives. Condition names prevent artifact overwrites. Trained-bias region remains primary. |
| Controlled rerun and final-capacity comparison | Same configs, `results/t1_pi20260920/plan_audit_20261009_repaired/` | Two datasets, seeds 8/9/10, ten conditions; separate original-capacity and matched-final-capacity batches. Validation results below. |

Train diagnostics add evaluation passes without gradients. The same training
queries, stable-ID self exclusions and measuring scale are used on both sides
of an update. Query-case workload counters count optimization passes; elapsed
time includes diagnostics, validation and maintenance. Independent training
omits synchronization penalties in optimization but still reports them.

The primary radius uses direct trained biases and the matching post-core learned
metric. The initial-bias ablation substitutes the initializer biases in that
same post-core metric; the exact-kth ablation uses the kth non-self neighbor per
query under that metric. All selectors use active training cases only. The
zero-radius ablation explicitly records its status. No transformed-bias fallback
is silently selected when a region is empty, broad or near zero.

## Scientific and experiment boundaries

The configs are exploratory: 30 core epochs, 20 synchronization epochs, two
epochs per alternating block, 60 training cases, declared pilot weights and
`s_task=.5`. The core snapshot is selected by validation loss; this does not
prove sufficient training for every task. Test labels do not choose calibration,
maintenance or checkpoints. Energy RMSE uses training-standardized target units.

The original-capacity batch compares full-memory independent/RR (60) with RRR
(45). It cannot isolate a fixed-memory synchronization effect. The matched
batch applies the same Q maintenance to every selected final state at K=45;
independent/RR are compressed after training without additional gradient steps.
This matches final capacity, not the memory trajectory or compute. Executed
and selected optimizer budgets remain separate. Regression L_delta and L_post
are algebraically identical MSE terms in the aggregate scalar residual path.

## Remaining plan requirements

| Workstream | Branch / checkout | Intended files or work | Validation | Status |
|---|---|---|---|---|
| Neural CBR / T1 | `neural-cbr` / `neural-cbr/` | General maintenance score and explicit prior-method reductions, then maintenance code | Distinguish component equality, ranking equality and full selector recovery | Open research design; no accepted final formula to implement |
| Neural CBR / T1 | `neural-cbr` / `neural-cbr/` | Synchronization grids and MCB interaction configs | Fresh confirmation with frozen thresholds, matched compute/capacity, component failure analysis | Bounded pilots completed here; full scientific gates pending |
| Neural CBR / T1 | `neural-cbr` / `neural-cbr/` | Modern transfer datasets, encoder benchmarks, contemporary comparators | Original protocols, data availability, prespecified performance margins | Broader benchmark reconstruction pending |
| Neural CBR / T1 | `neural-cbr` / `neural-cbr/` | `model/t1/reviewer_ui.py`, review protocol and human-study results | Real participants, M0/M1/M2 and matched controls | Engineering UI/tests exist; actual human-study evidence pending |
| RL / T2 | `rl` / `rl/` | RL-owned full-cycle integration and stage configs | Policy/value roles, task-specific outcome semantics and matched environment budgets | Outside this implementation change; depends on validated T1 helpers |
| LLM / T3 | `LLM` / `llm/` | LLM-owned host/retrieval integration | Prompt/internal comparisons, downstream usefulness and human revision | Outside this implementation change; depends on validated T1 helpers |

The shared initializer correction is isolated in `4cc1f33`: distinct duplicate
cases remain valid zero-distance non-self neighbors. Its focused checks passed.
No commit was propagated to another branch. Standalone NN-KNN, RL and LLM work
remain outside this T1 implementation; recipients must validate a cherry-pick
before using the correction. Do not merge the T1 branch to propagate it.

## Completed validation and results

- Baseline: 140 tests passed. Revised implementation: 146 tests passed, including
  snapshot mutation/reconstruction, stage losses and adapter-update isolation.
  Import smoke and `git diff --check` pass. One pre-existing test tensor-to-scalar
  warning remains.
- Corrected experiments: 120/120 expected manifests/checkpoints; 144 calibration
  snapshots reconstructed from saved model states; 2,400 epoch diagnostic records
  checked; 1,200 final scalar metrics replayed within 1e-6. Both source bundles
  pass ZIP integrity and reproduce their recorded SHA256 fingerprints. No source
  changed during either corrected batch.
- All 60 matched-capacity runs achieved 45 active cases. Primary and alternative
  nonzero-radius calibration statuses are `ok`; zero-radius experiments are
  explicitly marked as ablations.

Three-seed means at matched final K=45:

| Dataset / metric | Independent | RR | RRR |
|---|---:|---:|---:|
| Iris post-adaptation accuracy | 0.94444 | 0.95556 | 0.95556 |
| Energy post-adaptation RMSE | 0.28437 | 0.28361 | 0.28820 |

All Iris ablations have mean accuracy 0.95556; this small accuracy test does not
discriminate their mechanisms. Energy no-locality/no-preloss RMSE is
0.28621/0.28503; no-residual/no-small/initial-bias/exact-kth alternatives are
approximately 0.28361, and zero-radius is 0.28378. The primary Energy magnitude
penalty is inactive on two seeds and small on the third (maximum minibatch mean
0.000149); this is weak evidence for assessing the radius design. Iris's primary
penalty is active, despite unchanged accuracy across these ablations.

RR's descriptive differences from independent are small; RRR is worse on Energy
in this batch. No consistent synchronization advantage, selected scientific
gate, or confirmatory significance is established. Broader task difficulty and
matched-compute studies are still needed. No model was promoted on these results.

Artifacts (local, ignored by Git):

- Valid runs: `results/t1_pi20260920/plan_audit_20261009_repaired/`.
- Audit counts: `plan_audit_20261009_repaired/artifact_audit.json`.
- Full per-condition means/SDs: each experiment's `summary.csv`; matched results
  and paired differences are under `alignment_sync_plan_matched_20261009/`.
- Replay script: `plan_audit_20261009/audit_sync_evidence.py`; invoke with the
  repaired output directory as its argument. The script replays actual measuring
  scales and predictions and checks frozen source bundles.
- Invalid first calibration artifacts: `plan_audit_20261009/`; retained as failure
  evidence. Their audit log records the label-alias failure. Do not use them to
  substantiate frozen-calibration reproducibility.

Run from the owning checkout:

```powershell
.venv/Scripts/python.exe -m pytest tests --basetemp results/t1_pi20260920/pytest_full_plan_repaired_20261009 -q
.venv/Scripts/python.exe codex/smoke_test.py --mode imports
.venv/Scripts/python.exe tools/t1_run.py configs/t1/alignment_sync_plan_20261009.yaml --out <fresh-output-directory>
.venv/Scripts/python.exe tools/t1_run.py configs/t1/alignment_sync_plan_matched_20261009.yaml --out <fresh-output-directory>
.venv/Scripts/python.exe results/t1_pi20260920/plan_audit_20261009/audit_sync_evidence.py results/t1_pi20260920/plan_audit_20261009_repaired
```

## Full experiment continuation

The follow-up implementation and current execution protocol are recorded in
[T1_FULL_STUDY_20261009.md](T1_FULL_STUDY_20261009.md). The earlier 120-run results
above remain bounded pilots; full-data search and queued experiments do not
retroactively convert them into confirmatory evidence.
