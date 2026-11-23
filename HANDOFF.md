# Reinforcement learning handoff

Branch `rl` (`rl/`): shared NN-kNN core + RL (T2). Current scope: `tools/run_rl_*.py`, the battery tools, `tools/t2_role_pilot.py`, `tools/t2_retention_pilot.py`, `tools/make_phase4_report.py`, `scripts/` (harvest and launch drivers), and `tests/test_t2_*.py`.

## Where things are

- Specification: Xiaomeng Ye's `T2_RL_IMPLEMENTATION_PLAN.md` in `career-2027/03-research-architecture/implementation-handoff/` (a copy is in `docs/t1_spec/`).
- Results and logs: not in git. Every training and evaluation is recorded (`--study` required) as a run unit of the shared `NN-KNN/runs/t2/` (TensorBoard event files and weights; `docs/T2_RUN_RECORDING.md`). Summary and report scripts read only run units and write their generated files under `results/`; hand-written reports are in `reports/`. Historical results stay on the experiment host as files, filed by T2 handoff step under `results/t2/<step>/` and `logs/t2/<step>/` (`results/README.md` maps each step to its campaigns, documents, and gaps); they are no longer read by any script and are not moved or converted.
- Stage evidence write-ups: `docs/T2_*.md`; aggregate interpretation: `reports/experiment_log.md`.
- Shared core: synced from `main`; see `docs/NNKNN_CORE.md`.

## Status by handoff step (2026-10-10)

- Steps 1–3 (inventory, instrumentation, role audits): results present in `s01_inventory_reproduction` and `s02_s03_instrumentation_role_audits`.
- Step 4 (Stage A matrix): CartPole, Acrobot and LunarLander campaigns and PPO references in `s04_stage_a`.
- Step 5 (Stage A gate decision): no gate decision is recorded.
- Inherited T1 mechanisms: capacity, MCB and retention studies in `s06_case_memory_mechanisms`. `rl_case_maintenance` is incomplete, and the MCB report's promised Acrobot capacity table was never produced.
- Step 7 (nonstationarity) and step 10 (synthesis): not started.
- Step 8 (continuous action): neural TD3/PPO baselines only.
- Step 9 (MinAtar bridge, Atari Pong): seed-0 batteries plus PPO/DQN multi-seed references.

## 2026-11-23: recording is mandatory, summaries read run units only, nothing is retrained

Owner's rules O1-O5 (2026-11-22), applied to this branch; details and the table of default output locations in `docs/T2_RUN_RECORDING.md`.

- Every training function requires `run=` (an open tracking run; TypeError otherwise); `output_dir`, `make_*_output_dir` and the file-mode writers are gone. Every tool requires `--study` (argparse exit 2 without it): `run_rl_*.py` (training and `--eval-only`, which now reads only a unit and records the evaluation as a unit), the three battery tools, `t2_role_pilot.py`, `t2_retention_pilot.py` and the T2 modes of `codex/smoke_test.py`. The launch drivers in `scripts/g234/` and `scripts/r760/` pass `--study` (and `--unit` with the capacity) and no longer scan or delete run folders.
- Summary/report scripts read only run units (`RunFiles` refuses a legacy directory) and write under `results/`: `make_phase4_report.py` figures to `results/figures/`, `scripts/harvest_p17.py RUNS_FOLDER` to `results/t2/p17_capacity_cpu_manifest.json`, the timing passes to `results/t2/p1{7,8}_throughput_controlled.csv` (all three wrote into `reports/` before; the files already there are left alone).
- Never retrain: a complete unit is detected before any training (or evaluation, or retention computation) and read back; the retention pilot records and restores the global RNG state (`rng_after`), so a partly rerun study equals an uninterrupted one (tests in `tests/test_t2_runs.py`).
- Checked on g234 (CPU) against the pre-change tree: same unit digests and the same per-run content as the old file mode for all five algorithms, the role and retention pilots and the CartPole sweep; full suite 192 passed.
- Notebooks were not changed: their training cells now need a run.

## 2026-11-22: training runs recorded in TensorBoard run units

- Every RL training entry point (`train_dqn`, `train_nec`, `train_ppo`, `train_td3`, `train_nnknn_rl`) takes `run=` and `resumable=`; the tools `run_rl_*.py`, `run_rl_mcb_and_capacity_experiments.py`, `run_case_maintenance_ablation.py`, `run_cartpole_all_variants.py` and `t2_role_pilot.py` take `--study`. A recorded run writes no run directory: its metric CSVs, JSONs, summary, checkpoint and snapshots are records and named states of the unit, and long runs (Atari) resume from their last episode-boundary checkpoint. (Superseded on 2026-11-23: `--study` is now required.)
- `make_phase4_report.py`, `t2_retention_pilot.py` and `scripts/harvest_p17.py` read run units as well as run directories, with the same outputs (since 2026-11-23: run units only).
- Checked on g234 (CPU): same numbers, files and weights as before the change; resumed runs equal uninterrupted ones (CartPole, Pendulum, ALE Pong); report outputs byte-identical from units and from directories. Details, layout and what a resume does not restore: `docs/T2_RUN_RECORDING.md`.
- Disk: an Atari DQN/NEC unit checkpoints its replay buffer (about 5.6 GB each, estimated) every 1800 s and keeps `weights/resume/last.pt` and `prev.pt` after it completes (about 11 GB per finished unit); plan the disk before a multi-seed Atari battery under `--study`.
- Wall-time columns of units are the training time summed over the unit's sessions (`x/sys/train_seconds`), not comparable with a legacy directory's wall clock. (`scripts/harvest_p17.py` now takes a folder of run units and writes its manifest under `results/t2/`, see 2026-11-23.)

## 2026-11-21: repository layout cleanup

- Removed (obsolete, git history keeps them): README.md, `requirements.txt`, `requirements-latest.txt`, `verify_packages.py`, `codex/README.md`, `codex/setup.sh`, `codex/maintenance.sh` (`codex/smoke_test.py` kept), `tools/Sync-CodexHistory.ps1`, `tools/codex-history-sync.md`, `NNKNN_AGENT_GUIDE.md`, `.DS_Store`, both `Outdated` notebooks, and the plans `CLAUDE_CODE_REPRODUCTION_PLAN.md`, `NEXT_STEPS_PLAN.md`, `G234_CAMPAIGN_STATUS.md` (their still-valid content is in this file and `reports/`).
- Moved: `python_version.txt` to `.python-version`; `ENVIRONMENT_g234.md` to `docs/ENVIRONMENT_g234.md`; the demo notebook(s) to `notebooks/`.
- Decision (owner): no README/CHANGELOG/LICENSE for this project; environments and dependencies by uv only (`uv sync`, `uv run python ...`; no pip/requirements/hand-made venv). The 2026-11-21 follow-up: only `main` has `pyproject.toml`/`uv.lock`/`.python-version` (Python 3.14, shared venv `NN-KNN/.venv`); this branch's own copies were deleted. See `main/HANDOFF.md` for versions and known gaps.

## History

The original mixed handoff is preserved in the pre-reorganization Git bundle at `../recovery-20261009/before-repair.bundle`. See `docs/BRANCH_ORGANIZATION.md` and `docs/REORGANIZATION_MANIFEST.json` for the 2026-10-09 split, and `results/t2/PATH_MAP.json` (on disk with the results) for the 2026-10-10 results move.
