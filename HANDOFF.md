# Reinforcement learning handoff

Branch `rl` (`rl/`): shared NN-kNN core + RL (T2). Current scope: `tools/run_rl_*.py`, `tools/t2_role_pilot.py`, `tools/t2_retention_pilot.py`, and `tests/test_t2_*.py`.

## Where things are

- Specification: Xiaomeng Ye's `T2_RL_IMPLEMENTATION_PLAN.md` in `career-2027/03-research-architecture/implementation-handoff/` (a copy is in `docs/t1_spec/`).
- Results and logs: filed by T2 handoff step under `results/t2/<step>/` and `logs/t2/<step>/`. `results/README.md` maps each step to its campaigns, documents, and gaps.
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

## History

The original mixed handoff is preserved in the pre-reorganization Git bundle at `../recovery-20261009/before-repair.bundle`. See `docs/BRANCH_ORGANIZATION.md` and `docs/REORGANIZATION_MANIFEST.json` for the 2026-10-09 split, and `results/t2/PATH_MAP.json` for the 2026-10-10 results move.
