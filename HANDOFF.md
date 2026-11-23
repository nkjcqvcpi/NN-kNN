# Reinforcement learning handoff

Branch `rl` (`rl/`): shared NN-kNN core + RL (T2). Current scope: `tools/run_rl_*.py`, `tools/t2_role_pilot.py`, `tools/t2_retention_pilot.py`, and `tests/test_t2_*.py`.

## Where things are

- Specification: Xiaomeng Ye's `T2_RL_IMPLEMENTATION_PLAN.md` in `career-2027/03-research-architecture/implementation-handoff/` (a copy is in `docs/t1_spec/`).
- Results and logs: not in git. New training runs are recorded with `--study` as run units of the shared `NN-KNN/runs/t2/` (TensorBoard event files and weights; `docs/T2_RUN_RECORDING.md`). Historical results stay on the experiment host as files, filed by T2 handoff step under `results/t2/<step>/` and `logs/t2/<step>/`; `results/README.md` maps each step to its campaigns, documents, and gaps. Summary and report scripts read both.
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

## 2026-11-22: training runs recorded in TensorBoard run units

- Every RL training entry point (`train_dqn`, `train_nec`, `train_ppo`, `train_td3`, `train_nnknn_rl`) takes `run=` and `resumable=`; the tools `run_rl_*.py`, `run_rl_mcb_and_capacity_experiments.py`, `run_case_maintenance_ablation.py`, `run_cartpole_all_variants.py` and `t2_role_pilot.py` take `--study`. A recorded run writes no run directory: its metric CSVs, JSONs, summary, checkpoint and snapshots are records and named states of the unit, and long runs (Atari) resume from their last episode-boundary checkpoint. Without `--study` everything is as before.
- `make_phase4_report.py`, `t2_retention_pilot.py` and `scripts/harvest_p17.py` read run units as well as run directories, with the same outputs.
- Checked on g234 (CPU): same numbers, files and weights as before the change; resumed runs equal uninterrupted ones (CartPole, Pendulum, ALE Pong); report outputs byte-identical from units and from directories. Details, layout and what a resume does not restore: `docs/T2_RUN_RECORDING.md`.

## 2026-11-21: repository layout cleanup

- Removed (obsolete, git history keeps them): README.md, `requirements.txt`, `requirements-latest.txt`, `verify_packages.py`, `codex/README.md`, `codex/setup.sh`, `codex/maintenance.sh` (`codex/smoke_test.py` kept), `tools/Sync-CodexHistory.ps1`, `tools/codex-history-sync.md`, `NNKNN_AGENT_GUIDE.md`, `.DS_Store`, both `Outdated` notebooks, and the plans `CLAUDE_CODE_REPRODUCTION_PLAN.md`, `NEXT_STEPS_PLAN.md`, `G234_CAMPAIGN_STATUS.md` (their still-valid content is in this file and `reports/`).
- Moved: `python_version.txt` to `.python-version`; `ENVIRONMENT_g234.md` to `docs/ENVIRONMENT_g234.md`; the demo notebook(s) to `notebooks/`.
- Decision (owner): no README/CHANGELOG/LICENSE for this project; environments and dependencies by uv only (`uv sync`, `uv run python ...`; no pip/requirements/hand-made venv). The 2026-11-21 follow-up: only `main` has `pyproject.toml`/`uv.lock`/`.python-version` (Python 3.14, shared venv `NN-KNN/.venv`); this branch's own copies were deleted. See `main/HANDOFF.md` for versions and known gaps.

## History

The original mixed handoff is preserved in the pre-reorganization Git bundle at `../recovery-20261009/before-repair.bundle`. See `docs/BRANCH_ORGANIZATION.md` and `docs/REORGANIZATION_MANIFEST.json` for the 2026-10-09 split, and `results/t2/PATH_MAP.json` (on disk with the results) for the 2026-10-10 results move.
