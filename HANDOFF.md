# Neural CBR handoff

Branch `neural-cbr` (`neural-cbr/`): shared NN-kNN core + T1. Current scope: `tools/t1_run.py`, `tools/t1_reviewer_ui.py`, `configs/t1/`, and `tests/test_t1_*.py`.

## Where things are

- Specification: Xiaomeng Ye's T1 handoff in `career-2027/03-research-architecture/implementation-handoff/` (copies, roadmap, shared contract and supporting notes in `docs/t1_spec/`).
- Results and logs: not in git. Historical results stay on the experiment host, filed by T1 handoff phase under `results/t1/<phase>/` and `logs/t1/<phase>/`; `results/README.md` maps each phase to its pilot runs, pipeline batches and documents. New runs are recorded with `--study S` as TensorBoard run units under the shared `NN-KNN/runs/t1/S/` (curves, final metrics, predictions, records and every run's weights); summaries keep their files. What each tool records and which readers take units: `docs/T1_RUN_RECORDS.md`.
- Current status and pending gates: `docs/T1_FULL_STUDY_20261009.md`, `docs/T1_PLAN_AUDIT_20261009.md`, `docs/T1_ALIGNMENT_STATUS.md`; cross-stage progress report: `reports/NEURAL_CBR_PROGRESS_10-06.md` (later progress reports are in `reports/` too).
- Discussion notes: `docs/T1_DISCUSSION_NOTES_20261030.md` (2026-10-30 discussion; points 1-3 recorded, 4 onward deferred).
- Shared core: synced from `main`; see `docs/NNKNN_CORE.md`.

## 2026-11-21: repository layout cleanup

- Removed: `README.md`; `requirements.txt`, `requirements-latest.txt`; `verify_packages.py`; `codex/README.md`, `codex/setup.sh`, `codex/maintenance.sh` (`codex/smoke_test.py` is kept); `tools/Sync-CodexHistory.ps1`, `tools/codex-history-sync.md`; `NNKNN_AGENT_GUIDE.md` (main keeps the long guide as `docs/NNKNN_AGENT_GUIDE.md`); `.DS_Store`.
- Moved: `python_version.txt` -> `.python-version`; `handoff/` -> `reports/` (same file names).
- Decision (owner): this project has no README, CHANGELOG or LICENSE. Environments and dependencies are managed by uv only (`uv sync`, `uv run ...`); no pip, requirements files or hand-made venv. The 2026-11-21 follow-up: only `main` has `pyproject.toml`/`uv.lock`/`.python-version` (Python 3.14, shared venv `NN-KNN/.venv`); this branch's own copies were deleted. See `main/HANDOFF.md` for versions and known gaps. Dated reports written earlier still quote the old paths.

## 2026-11-22: run records (TensorBoard units)

- Every T1 entry point that trains (`t1_run` with its drivers `t1_parallel`, `t1_execute_study`, `t1_launch.ps1`; `t1_grid_search` and `t1_grid_prefill`; `t1_core_ablation`; the two pilots; the reviewer UI) takes `--study S` and then records each run as one unit through `model/common/tracking.py` instead of writing per-run files; without `--study` the files are written as before. The legacy reference passes `run=` to `train_model` (no `legacy_best.pth`). Helpers and readers: `model/t1/records.py`.
- The readers (`t1_summarize`, `t1_encoder_report`, `t1_schedule_search`, `t1_block_accept`, `t1_freeze`, `t1_cbm_diagnostics`, `t1_confirm_report`, `t1_confirm_audit`) take a run-unit tree as well as the legacy result directories; their output files are unchanged and byte-identical for the same runs.
- The selections that 43 configs read from grid `selection.json` files are inlined (`core_selection_source.*.selection`; `grid_winners` in the core-ablation config); see `docs/T1_RUN_RECORDS.md`.
- Tests: `tests/test_t1_records.py`.

## History

The original mixed handoff is preserved in the pre-reorganization Git bundle at `../recovery-20261009/before-repair.bundle`. See `docs/BRANCH_ORGANIZATION.md` and `docs/REORGANIZATION_MANIFEST.json` for the 2026-10-09 split, and `results/t1/PATH_MAP.json` (on disk with the results) for the 2026-10-10 results filing.
