# Neural CBR (T1) results, filed by handoff phase

Organized 2026-10-10 by the implementation phases of Xiaomeng Ye's T1 handoff: `career-2027/03-research-architecture/implementation-handoff/T1_FULL_CYCLE_IMPLEMENTATION_PLAN.md` §13, with a copy in `docs/t1_spec/`. `results/t1/<phase>/` and `logs/t1/<phase>/` use the same phase folders.

```
results/t1/
  p0_reproduce_instrument/    Phase 0: reproduce and instrument (legacy reference, smoke, verification, core fix)
  p1_supervised_retention/    Phase 1: supervised retention prototype (alignment candidates, adapted removal)
  p2_classification_reuse/    Phase 2: classification reuse (NN-CDH adapter, nominal reuse)
  p4_component_sync/          Phase 4: component harmonization (synchronization, capacity match, radius, plan audits)
  p5_mcb_interaction/         Phase 5: MCB interaction
  p6_confirmation_revision/   Phase 6 / T1.2: broader confirmation, human revision, full study
  historical_t0/              the pre-T1 "T0" battery of 2026-09-09
  invalid_fresh_adam/         runs invalidated by the fresh-Adam bug (superseded by ec6323a)
  PATH_MAP.json               old -> new location of every moved file and directory
```

Phase 3, RL integration, is on the `rl` branch under `results/t2/s06_case_memory_mechanisms/`.

Inside a phase, `pilot_20260929/<experiment>/` holds the 2026-09-29 pilot runs, which predate the PI specification of 2026-09-20 (`docs/T1_EXPERIMENT_PLAN.md`). `pi20260920/<batch>/<experiment>/` holds runs of the current pipeline. Only the pilot's `runs__*.jsonl` shards and summary CSVs are versioned; per-run directories, pipeline outputs, and logs stay local (about 5 GB).

## Phases, experiments and documents

| Phase | Pilot (`pilot_20260929/`) | Pipeline (`pi20260920/`) | Documents (`../docs/`) |
| --- | --- | --- | --- |
| P0 | `p0_legacy_reference` | `smoke_*`, `verification_20261002`, `continuation_20261004/core_selection_probe_{before,after}`, `pytest/` | `experiments/2026-10-04-core-selection-fix.md`, `T1_PLAN_IMPLEMENTATION.md` |
| P1 | `p1_retention_classification`, `_regression`, `_synthetic` | `alignment_candidates_small`, `alignment_digits`, `continuation_20261002`, `continuation_20261004/adapted_*`, `alignment_nominal_retention` | `T1_ADAPTED_BUDGET_STUDY.md`, `T1_ADAPTED_REFERENCE_GUARD.md`, `T1_ADAPTED_REGRESSION.md`, `T1_ADAPTED_RETRAINING.md`, `experiments/2026-10-02-pi-alignment-continuation.md` and `-runs.csv` |
| P2 | `p2_reuse` | `continuation_20261004/alignment_nominal_reuse`, `optimizer_checkpoint_reuse` | `experiments/2026-10-04-nominal-contract.md`, `experiments/2026-10-04-selected-adapter-optimizers.md` |
| P4 | `p4_sync` | `continuation_20261004/alignment_sync*`, `alignment_nominal_sync`, `optimizer_checkpoint_sync`, `sync_capacity_match*`; `plan_audit_20261009*`; `radius_iris_20261009` (and `radius_full_20261009` from the full study) | `T1_SYNC_CAPACITY_COMPARISON.md`, `T1_PLAN_AUDIT_20261009.md`, `experiments/2026-10-04-supervised-sync.md`, `experiments/2026-10-04-sync-selection-fix.md` |
| P5 | `p5_mcb` | — | `t1_spec/supporting/MCB_*` |
| P6 / T1.2 | `t12_revise_synthetic` | `continuation_20261004/alignment_revision*`, `alignment_synthetic_reliability`, `reviewer_probes*`; full study: `grid_preflight_20261009`, `grid_full_20261009`, `plan_iris_20261009`, `plan_full_20261009`, `runner_preflight_20261009`, `full_study_20261009` | `T1_FULL_STUDY_20261009.md`, `T1_REVIEWER_UI.md`, `T1_REVIEWER_INTERVENTION_DESIGN.md`, `T1_COMMON_KNOWLEDGE_UI_PROTOCOL.md`, `experiments/2026-10-04-revision-artifacts.md`, `experiments/2026-10-04-synthetic-reliability.md`, `experiments/2026-10-04-reviewer-controls.md` |

## EMA ablation, joint training and schedule search (2026-10-11)

Documented in `../docs/T1_EMA_JOINT_SCHEDULE_20261011.md`. Per-run directories are local under `results/t1/<phase>/pi20260920/<experiment>/`; the versioned folders below hold run shards, reports and selections. Logs: `logs/t1/<phase>/<experiment>/`.

| Phase | Versioned folder | Contents |
| --- | --- | --- |
| P5 | `p5_mcb_interaction/ema_ablation_20261010/` | EMA vs shared encoder vs raw NN-kNN, retrieval only and retrieval + reuse, 13 tasks × 5 seeds |
| P5 | `p5_mcb_interaction/ema_tuned_mlp_bs64_20261012/` | EMA vs shared encoder on the six tasks whose tuned core uses the MLP (batch-64 cores; `ema_tuned_mlp_20261012/` is the superseded batch-32 run) |
| P4 | `p4_component_sync/schedule_search_20261011/` | training-schedule search: stage A and B validation selections, confirmation on seeds 30–34 |

Cross-phase records: `T1_EXPERIMENT_PLAN.md`, `T1_ALIGNMENT_STATUS.md`, `T1_EXECUTION_LEDGER.md`, `T1_PLAN_IMPLEMENTATION.md`, and `../handoff/NEURAL_CBR_PROGRESS_10-06.md`.

## Moving the local artifacts

The tracked pilot shards were moved in Git. The local artifacts were produced in the `neural-cbr` checkout while the 2026-10-09 full study was running, so they are moved separately by `../scripts/reorganize_t1_results_20261010.py` once no T1 process is running. Then commit `results/t1/PATH_MAP.json`.

- Default mode moves the pilot per-run directories, `logs/t1`, `results/t0_battery`, and `results/t1_invalid_fresh_adam`.
- `--pipeline` also files `results/t1_pi20260920/` by phase. It is opt-in for two reasons. First, six audit scripts in that folder locate the repository with `Path(__file__).parents[3]` and must be run from their original depth. Second, run manifests, execute-study receipts, and `docs/experiments/2026-10-02-pi-alignment-runs.csv` record the old relative paths. `PATH_MAP.json` resolves both.

`tools/t1_run.py` writes to `results/t1_pi20260920/` by default. File finished batches under `results/t1/<phase>/pi20260920/`.
