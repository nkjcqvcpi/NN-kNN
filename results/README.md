# RL (T2) results, filed by handoff step

Reorganized 2026-10-10. Results and logs are filed by the implementation step of Xiaomeng Ye's T2 handoff, `career-2027/03-research-architecture/implementation-handoff/T2_RL_IMPLEMENTATION_PLAN.md` (§13 sequence; §7–8 Stage A gate and benchmark stages). `results/t2/<step>/` and `logs/t2/<step>/` use the same step folders. Inside a step, each campaign folder names its host and campaign (`r760_phase4_cartpole`, `g234_p17_p22_capacity_cpu`), and run directories keep their original names.

```
results/t2/
  s01_inventory_reproduction/            §13 step 1: reproduce current tasks and references
  s02_s03_instrumentation_role_audits/   §13 steps 2–3: shared instrumentation, actor/critic role audits
  s04_stage_a/                           §13 step 4, §8 Stage A: CartPole, Acrobot, LunarLander matrix
  s06_case_memory_mechanisms/            §5 inherited T1 mechanisms (capacity, MCB, retention); §13 step 6
  s08_stage_b_continuous/                §13 step 8, §8 Stage B: continuous action
  s09_bridge_minatar/                    §13 step 9, §8 Bridge: MinAtar Breakout
  s09_stage_c_atari/                     §13 step 9, §8 Stage C: Atari/ALE Pong
  PATH_MAP.json                          old -> new location of every moved directory and log
```

## Steps, campaigns and evidence

| Step | Campaign folder | Contents | Described in |
| --- | --- | --- | --- |
| s01 | `r760_smoke_checks` | 58 smoke-profile runs (DQN, NEC, NN-kNN-RL; CartPole, Acrobot, LunarLander, MinAtar), 07-31 to 08-22 | `../reports/experiment_log.md` Phase 0, p8 |
| s01 | `r760_phase1_dqn_nec_reference` | DQN/NEC CartPole reference reproduction, fixes, 300k NEC probe, reload evaluations | `../CLAUDE_CODE_REPRODUCTION_PLAN.md` Phase 1 |
| s01 | `r760_phase2_nnknn_variants` | six NN-kNN actor/critic variants, CartPole | Phase 2; `../reports/cartpole_12variant_sweep.csv` |
| s01 | `r760_phase3_gate_g1` | frozen NN-kNN hybrid reference, seeds 0–4, plus reload evaluations | Phase 3 Gate G1 |
| s01 | `g234_cpu_smoke_threads` | CPU smoke and thread-count runs on g234, 09-03 to 09-05 | `../ENVIRONMENT_g234.md` |
| s01 | `legacy_cartpole_videos` | CartPole demo videos from 2026-02 | — |
| s02–s03 | `pi20260920/continuation_20261004`, `continuation_20261005` | role audit (512 steps; `role_audit_512` is the superseded pre-fix run), target audit, target sync, optimizer alignment, two-task diagnosis, quality ledger, selected checkpoint | `../docs/T2_ROLE_AUDIT_PILOT.md`, `T2_TRAINING_CONTRIBUTION_AUDIT.md`, `T2_EXECUTED_TARGET_DIAGNOSIS.md`, `T2_OPTIMIZER_ALIGNMENT.md`, `T2_TWO_TASK_DIAGNOSIS.md`, `T2_ROLE_QUALITY_LEDGER.md`, `T2_SELECTED_CHECKPOINT_STATE.md` |
| s02–s03 | `codex_task_20261002` | critic label calibration, extended Acrobot startup, quality ledger runs, and their verification files | `../docs/T2_CRITIC_LABEL_CALIBRATION.md`, `T2_EXTENDED_STARTUP.md`, `T2_ROLE_QUALITY_LEDGER.md`, `T2_CRITIC_IDENTITY.md` |
| s04 | `r760_phase4_cartpole` | DQN, NEC, NN-kNN hybrid, MLP actor-critic × 5 seeds, 150k fixed budget | `../reports/phase4_manifest.json` |
| s04 | `r760_phase5_acrobot` | Acrobot, 4 methods × 5 seeds | `../reports/phase5_acrobot_manifest.json` |
| s04 | `r760_p6_label_modes`, `r760_p7_label_modes` | critic label-mode grid and seeds; Acrobot 300k probe and knobs A/B/C | experiment_log p6, p7 |
| s04 | `r760_p8_lunarlander` | LunarLander seed-0 battery | experiment_log p8 |
| s04 | `r760_p11_ppo` | PPO 150k × 5 seeds on CartPole, Acrobot, LunarLander | experiment_log p11 |
| s06 | `r760_p9_p10_p14_capacity`, `g234_p17_p22_capacity_cpu` | NN-kNN-RL case capacity 100–2000 × 5 seeds (CUDA on r760; CPU on g234) | `../reports/capacity_ablation.md`, `p17_capacity_cpu_manifest.json`, `../G234_CAMPAIGN_STATUS.md` |
| s06 | `rl_mcb_campaign` | MCB vs vanilla vs MLP, CartPole, capacity 500, seeds 0–2; its report promises an Acrobot capacity table that was never produced | `rl_mcb_campaign/RL_MCB_AND_CAPACITY_REPORT.md` |
| s06 | `rl_case_maintenance` | incomplete: `mech_vanilla` seeds 0–1 only; report and raw CSV missing | `../tools/run_case_maintenance_ablation.py` |
| s06 | `codex_task_20261002` | live retention, fresh-reference retention, reference coverage guard | `../docs/T2_LIVE_RETENTION.md`, `T2_FRESH_REFERENCE_RETENTION.md`, `T2_REFERENCE_COVERAGE_GUARD.md` |
| s08 | `r760_p12_ppo_td3` | TD3 and PPO on Pendulum and LunarLanderContinuous, 3 seeds; neural baselines only, no NN-kNN continuous actor yet | experiment_log p12 |
| s09 | `r760_p8_minatar`, `r760_p11_ppo` (bridge) | MinAtar Breakout seed-0 battery; PPO × 5 seeds | experiment_log p8, p11 |
| s09 | `r760_p13_p15_pong`, `g234_p16_nec_pong`, `g234_p20_dqn_ppo_pong`, `g234_p20_threadconfound` | Pong: seed-0 DQN/PPO/NEC/NN-kNN (p13; p15 corrected DQN), NEC 500k (p16), DQN/PPO × 5 seeds on CPU (p20), the preserved OMP=3 run | experiment_log p13–p20; `../G234_CAMPAIGN_STATUS.md` |

Handoff steps with no results yet: step 5 (Stage A gate decision), step 7 (nonstationarity), and step 10 (synthesis for T3). Step 8 holds only neural baselines.

Two logs are filed under their dominant step although they cover several: the p8 and p11 drivers and PID files cover both Stage A and MinAtar runs, and are under `s04_stage_a`.

## Codex task evidence

`codex_task_20261002/` folders hold copies of T2 evidence produced in the Codex task workspace `C:\Users\Administrator\Documents\Codex\2026-10-02\neural-cbr-xiaomeng-document-https-github\`, copied unchanged on 2026-10-10:

- `outputs/`: the `neural-cbr-t2-*` verification and results files (tracked).
- `audit_scripts/`: the `audit_t2_*.py` scripts that produced them (tracked).
- `runs/`: the run directories the T2 documents cite as `work/...` (532 MB with checkpoints; ignored, local only).
- matching driver and audit logs are in `logs/t2/<step>/codex_task_20261002/` (tracked).

The original folder remains the location recorded inside these files and verification hashes.

## Hosts

Numbers from different hosts are **not directly comparable**: the float path differs, and a device effect of about 28 points on the cap-2000 capacity arm is documented in `../reports/experiment_log.md` (p17, 2026-09-03).

- `r760`: Linux, H100, CUDA (2026-07-31 to 09-01).
- `g234`: Windows Server 2025, i5-13400F, Intel Arc, CPU only (from 2026-09-03).

A run's host was assigned from its own `config.json`: `runtime.platform` when present (`Windows-2025Server-*` is g234), else `device == "cuda"` (r760), else the run date (before 2026-09-03 is r760).

## What is kept

Each completed run normally includes `config.json`, `manifest.json`, `summary.json` (the authoritative source for every reported number), training/loss/evaluation/diagnostic CSV files, and per-episode final and last-model evaluations. Folders ending in `_eval` are checkpoint-reload evaluations and hold `eval_summary.json`.

Checkpoints (`checkpoint.pt`, `*.pth`) are excluded from Git and kept locally. On 2026-10-10, 1,009 checkpoints and 84 local `results/rl/` runs were recovered from the `neural-cbr` checkout, where the branch split had left them: 919 checkpoints are back in their run directories under `t2/`, and the 84 runs (holding the other 90) are in `results/rl/`. An incomplete run leaves an empty directory, because these workflows write artifacts only at completion.

## Paths inside artifacts

Run artifacts and logs keep the paths they were written with. r760 runs record `results/rl/<run>`, g234 runs record `results/rl_capacity_cpu_g234/`, `results/rl_ale_g234/` or `results/rl/`, and the `pi20260920` and `rl_mcb_campaign` files record their pre-2026-10-10 paths. Use `t2/PATH_MAP.json` to resolve them. The launch scripts in `../scripts/` are kept as run; they name the old repository root `NN-KNN_RL/nnknn-work`.

## Filing new runs

The RL runners default to `--output-dir results/rl` (ignored scratch). Pass the step folder explicitly, for example `--output-dir results/t2/s04_stage_a/g234_<campaign>`. A run left in `results/rl/` is unfiled; move it under its step before harvesting. Aggregate interpretations and figures are in [`../reports/experiment_log.md`](../reports/experiment_log.md).
