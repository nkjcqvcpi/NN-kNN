# Neural CBR handoff

Branch `neural-cbr` (`neural-cbr/`): shared NN-kNN core + T1. Current scope: `tools/t1_run.py`, `tools/t1_reviewer_ui.py`, `configs/t1/`, and `tests/test_t1_*.py`.

## Where things are

- Specification: Xiaomeng Ye's T1 handoff in `career-2027/03-research-architecture/implementation-handoff/` (copies, roadmap, shared contract and supporting notes in `docs/t1_spec/`).
- Results and logs: filed by T1 handoff phase under `results/t1/<phase>/` and `logs/t1/<phase>/`; `results/README.md` maps each phase to its pilot runs, pipeline batches and documents.
- Current status and pending gates: `docs/T1_FULL_STUDY_20261009.md`, `docs/T1_PLAN_AUDIT_20261009.md`, `docs/T1_ALIGNMENT_STATUS.md`; cross-stage progress report: `handoff/NEURAL_CBR_PROGRESS_10-06.md`.
- Shared core: synced from `main`; see `docs/NNKNN_CORE.md`.

## History

The original mixed handoff is preserved in the pre-reorganization Git bundle at `../recovery-20261009/before-repair.bundle`. See `docs/BRANCH_ORGANIZATION.md` and `docs/REORGANIZATION_MANIFEST.json` for the 2026-10-09 split, and `results/t1/PATH_MAP.json` for the 2026-10-10 results filing.
