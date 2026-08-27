# Next-Steps Plan — 2026-08-27

User-directed follow-up campaign on branch `rl-iclr2027` (local only; never
push to `Heuzi/NN-kNN`). Three asks: (1) CartPole capacity ablation, (2) PPO
and TD3 baselines, (3) ALE (Atari) environments. These map onto HANDOFF.md
manuscript priorities 1, 2, and 4. Execution: implementation first (Phase I),
then experiment batteries (Phase II). Run numbering continues at p10.

## Current facts this plan builds on

- Capacity ablation cap∈{100, 2000} × seeds {0,1,2} COMPLETED 2026-08-22
  (unharvested; run dirs untracked). Early read: cap100 = 500.0 / 500.0 /
  494.15 (two target-score stops); cap2000 = 455.9 / 484.1 / 474.6 (all
  budget-exhausted) — cap2000 looks *worse* than the 500 default
  (497.98 ± 4.52), which if it survives 5 seeds is a real finding (bigger
  memory ≠ better; retrieval dilution / slower consolidation).
- No PPO or TD3 runner exists. All four registered tasks are discrete-action.
  TD3 is continuous-action by construction, and the NN-kNN actor is
  discrete-only (one-hot action labels) — so TD3 cannot meet NN-kNN-RL
  head-to-head on any current task. Resolution below.
- `ale-py` is not installed; every registered task is `flat_box`. ALE needs an
  image-observation path (AtariPreprocessing + frame stack + CNN encoder).
- GPU: H100 NVL, ~60 GB held by external jobs, ~4% util. Prior convention:
  gate new launches on <88 GB used; CPU runs must set `CUDA_VISIBLE_DEVICES=""`.

## Phase I — Implementation (Opus, xhigh reasoning; sequential, committed per step)

**I-1. PPO (`model/ppo_workflow.py` + `tools/run_rl_ppo.py`).** Repo-native
clipped-surrogate PPO over the same env/eval machinery as the existing
runners: identical CLI shape, profiles, seeding, eval protocol,
early-stopping semantics (AGENTS.md: patience 30, min-delta 1.0, min-steps
25k, target-score immediate stop), and `summary.json`/`config.json` schema so
`tools/make_phase4_report.py` ingests PPO rows unchanged. Discrete
(categorical) and continuous (diagonal Gaussian) heads, selected from the
task spec's `action_kind`. New smoke mode `ppo` in `codex/smoke_test.py`.
CartPole regression smokes for existing workflows must stay bit-identical
(9.5 / 85.0 / 279.5).

**I-2. TD3 (`model/td3_workflow.py` + `tools/run_rl_td3.py`) + continuous
tasks.** Standard TD3 (twin critics, delayed policy updates, target policy
smoothing, replay buffer). Registry gains `env_kwargs` support and two
continuous tasks: `pendulum` (Pendulum-v1; no canonical gymnasium threshold —
document success −200 / target −140 as non-canonical markers) and
`lunarlander_continuous` (LunarLander-v3, `continuous=True`; success 200 /
target 280 as discrete). Scope note recorded in-code and here: TD3 is a
continuous-control reference baseline; NN-kNN-RL is discrete-only, so the
TD3 comparison runs TD3 vs PPO(continuous) on these two tasks, and a
continuous-action NN-kNN actor is flagged future work, not attempted.
Smoke mode `td3`.

**I-3. ALE support + registry breadth.** Add `ale-py` via uv (refresh
`reports/env_freeze.txt`). Image-observation path keyed off a new
`observation_kind` (AtariPreprocessing: grayscale, 84×84, frame-skip 4, on a
frameskip=1 base env; FrameStackObservation 4) plus a Nature-CNN encoder
usable by DQN/NEC/PPO and as the NN-kNN shared feature trunk. Register
`ale_pong` (success 0 / target 18 — non-canonical progress markers; Nature
DQN ≈ +18 at 10M+ frames) and `ale_breakout` (success 20 / target 100,
non-canonical). Also register the four remaining MinAtar games
(`minatar_asterix`, `minatar_freeway`, `minatar_seaquest`,
`minatar_space_invaders`) with documented non-canonical markers — ~10
registry lines each per HANDOFF item 4. All runners must smoke-pass the new
tasks; existing smokes bit-identical.

**I-4. Verification gate.** All smoke modes (`imports`, `rl`, `nec`,
`nnknn_rl`, `ppo`, `td3`), new-task smoke runs for every runner, CartPole
fast-profile regression values unchanged. Fix-loop until green.

## Phase II — Experiments (Sonnet, xhigh reasoning; nohup + pN_pids.txt convention)

**p10 — capacity completion (ask 1).** Launch cap100/cap2000 × seeds {3,4}
(fast profile, early stopping ON, matching seeds 0–2). Harvest all 5 seeds ×
{100, 500(=G1 batch), 2000} into a per-arm table: final_eval, best_model_step,
stopping reason, AND wall-clock (the retrieval-cost/performance trade-off is
the point). Figure + experiment-log entry + commit run dirs.
Optional stretch if cap2000 deficit holds: a cap1000 arm to locate the knee.

**p11 — PPO batteries (ask 2a).** Fixed-budget 150k `--no-early-stopping`,
seeds {0..4}, on cartpole and acrobot → extend the Phase-4/Phase-5 manifests
and regenerate the 5-method tables/figures. Seeds {0..4} on lunarlander and
minatar_breakout (matches the DQN/NEC/AC rows' protocol where they exist;
seed 0 first, 1–4 after sanity). PPO is the highest-reviewer-risk baseline —
this is the priority battery.

**p12 — TD3 reference (ask 2b).** pendulum and lunarlander_continuous ×
seeds {0..2}: TD3 vs PPO(continuous), 150k fixed budget. Reference table
only (no NN-kNN row — see I-2 scope note).

**p13 — ALE exploratory (ask 3).** Seed 0, budgets are explicit and honest:
DQN-CNN and PPO-CNN at 1M env steps (=4M frames), NN-kNN hybrid and NEC at
250k–500k steps (wall-clock-bound), on ale_pong first, ale_breakout second.
Declared exploratory: at these budgets expect partial learning curves, not
solved games; the deliverable is the learning-curve figure and an honest
transfer statement, mirroring the LunarLander/MinAtar negative-result
treatment. Eval episodes small (5–10) since ALE episodes are long.

**Harvest & bookkeeping.** As batteries land: experiment-log entries in
`reports/experiment_log.md` (every launch logged per plan rule), HANDOFF.md
pending-section refresh, commit run artifacts minus checkpoints, refresh
env_freeze after dep changes. Wall-clock recorded for every arm.

## Risks / guardrails

- ALE runs are multi-hour; launches are backgrounded and harvested as they
  complete — partial harvests are logged rather than waited on.
- GPU-memory gate <88 GB before each launch wave; stagger to ≤12 concurrent
  light (classic-control) runs, ≤3 concurrent CNN runs.
- The known knob-B trap: in shared-representation mode
  `make_nnknn_rl_config` silently overrides `critic_learning_rate`
  (model/nnknn_rl_workflow.py:1380) — no new experiment may vary that flag
  in shared mode.
- Never push; all commits local to `rl-iclr2027`.
