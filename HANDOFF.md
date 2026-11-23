# Neural CBR handoff

Branch `neural-cbr` (`neural-cbr/`): shared NN-kNN core + T1. Current scope: `tools/t1_run.py`, `tools/t1_reviewer_ui.py`, `configs/t1/`, and `tests/test_t1_*.py`.

## Where things are

- Specification: Xiaomeng Ye's T1 handoff in `career-2027/03-research-architecture/implementation-handoff/` (copies, roadmap, shared contract and supporting notes in `docs/t1_spec/`).
- Results and logs: not in git. Historical results stay on the experiment host, filed by T1 handoff phase under `results/t1/<phase>/` and `logs/t1/<phase>/`; `results/README.md` maps each phase to its pilot runs, pipeline batches and documents. Every run is recorded (`--study S` is required) as a TensorBoard run unit under the shared `NN-KNN/runs/t1/S/` (curves, final metrics, predictions, records and every run's weights); summaries read only run units and write under `results/`; reports written by people or agents go to `reports/`. What each tool records, the never-retrain rules and the summaries' default outputs: `docs/T1_RUN_RECORDS.md`.
- Current status and pending gates: `docs/T1_FULL_STUDY_20261009.md`, `docs/T1_PLAN_AUDIT_20261009.md`, `docs/T1_ALIGNMENT_STATUS.md`; cross-stage progress report: `reports/NEURAL_CBR_PROGRESS_10-06.md` (later progress reports are in `reports/` too).
- Discussion notes: `docs/T1_DISCUSSION_NOTES_20261030.md` (2026-10-30 discussion; points 1-3 recorded, 4 onward deferred).
- Shared core: synced from `main`; see `docs/NNKNN_CORE.md`.

## 2026-11-21: repository layout cleanup

- Removed: `README.md`; `requirements.txt`, `requirements-latest.txt`; `verify_packages.py`; `codex/README.md`, `codex/setup.sh`, `codex/maintenance.sh` (`codex/smoke_test.py` is kept); `tools/Sync-CodexHistory.ps1`, `tools/codex-history-sync.md`; `NNKNN_AGENT_GUIDE.md` (main keeps the long guide as `docs/NNKNN_AGENT_GUIDE.md`); `.DS_Store`.
- Moved: `python_version.txt` -> `.python-version`; `handoff/` -> `reports/` (same file names).
- Decision (owner): this project has no README, CHANGELOG or LICENSE. Environments and dependencies are managed by uv only (`uv sync`, `uv run ...`); no pip, requirements files or hand-made venv. The 2026-11-21 follow-up: only `main` has `pyproject.toml`/`uv.lock`/`.python-version` (Python 3.14, shared venv `NN-KNN/.venv`); this branch's own copies were deleted. See `main/HANDOFF.md` for versions and known gaps. Dated reports written earlier still quote the old paths.

## 2026-11-22: run records (TensorBoard units)

- Every T1 entry point that trains (`t1_run` with its drivers `t1_parallel`, `t1_execute_study`, `t1_launch.ps1`; `t1_grid_search` and `t1_grid_prefill`; `t1_core_ablation`; the two pilots; the reviewer UI) took `--study S` to record each run as one unit through `model/common/tracking.py` (mandatory since 2026-11-23, see above). The legacy reference passes `run=` to `train_model` (no `legacy_best.pth`). Helpers and readers: `model/t1/records.py`.
- The readers (`t1_summarize`, `t1_encoder_report`, `t1_schedule_search`, `t1_block_accept`, `t1_freeze`, `t1_cbm_diagnostics`, `t1_confirm_report`, `t1_confirm_audit`) were made to read run-unit trees with output files byte-identical to those from the legacy result directories (legacy readers removed on 2026-11-23).
- A unit is reused only with the same config: grid and core-ablation trials hold their data definition (dataset entry and splits), source fingerprint and runtime, so grids or evaluate runs sharing a study never take each other's trials; `t1_run` units leave out the `seeds` list, so a study started per seed (`t1_parallel`, `t1_execute_study`) resumes with a plain `t1_run`, and a changed config stops the job before training. Unit paths too long for Windows are refused before training (see `docs/T1_RUN_RECORDS.md`).
- The configs keep naming their grid `selection.json` by path, as frozen (`confirm_final_20261013/` is byte-identical to its freeze); only the core-ablation config carries `grid_winners`.
- Tests: `tests/test_t1_records.py`.

## 2026-11-23: mandatory recording, summaries from run units, never retrained (owner's rules O1-O5)

- `--study S` is required by every entry point that trains or evaluates (`t1_run` and its drivers `t1_parallel`, `t1_execute_study`, `t1_launch.ps1 -Study`; `t1_grid_search` and `evaluate`; `t1_grid_prefill`, which refuses grids without a study; `t1_core_ablation`; both pilots; the reviewer UI; `codex/smoke_test.py`, copied from main). Without it: argparse error. `Runner(cfg, out, study)`, the grid and ablation `run_trial(..., study=)` and `ReviewerSession(..., run=)` raise `TypeError` without one. The file-only mode and its writers are removed (`FileSink`, `append_row`, `TrialFiles`, `legacy_trial`, the pilots' `runs/` files, the reviewer's version folders, `t1_confirm_audit --repair` and its shard audit).
- Never retrained: `t1_run` records shared training as stage units (`_core`, `_core__<arm>` with the initial adapter where a type has one, `_retrained_K<k>`, `_revise`) holding the exact objects and the RNG state; every run records `rng` and the ID counter positions `ids`. A rerun skips complete runs (restoring their RNG state and counters), loads complete stages, and neither trains nor loads a stage whose runs are all complete, so a partly rerun study equals an uninterrupted one (`tests/test_t1_never_retrain.py`, all ten experiment types). `t1_mask_joint_pilot` records its pretrained cores as stages the same way. Resume files are kept after a unit completes.
- Summaries read only run units (`records.require_unit_tree`) and default to `results/` (`records.results_dir`: `results/t1/<study>/<experiment>/` for `t1_summarize` and `t1_encoder_report`); the pilots' default `--out` is under `results/`. No script wrote generated files into `reports/` or `docs/`.
- Tests: `tests/test_t1_never_retrain.py`, `tests/test_t1_records.py`, `tests/test_t1_parallel.py`; test helpers `tests/t1_units.py` (fixtures written in the old file layout are recorded as units there).

## History

The original mixed handoff is preserved in the pre-reorganization Git bundle at `../recovery-20261009/before-repair.bundle`. See `docs/BRANCH_ORGANIZATION.md` and `docs/REORGANIZATION_MANIFEST.json` for the 2026-10-09 split, and `results/t1/PATH_MAP.json` (on disk with the results) for the 2026-10-10 results filing.
