# Claude Code Reproduction Plan — NN-kNN-RL (CartPole first)

**Purpose:** hand this file to Claude Code on a server so it can reproduce current RL progress and push toward Gate G1 of `RESEARCH_PROPOSAL_ICLR2027.md` (NN-kNN-RL solves CartPole ≥475 by Aug 14, 2026).
**Source repo:** `https://github.com/Heuzi/NN-kNN.git` (branch `main`). Before launching on the server, push your local `main` so the server sees the current state, and note your local `git rev-parse HEAD` to verify the clone matches.
**Authoritative in-repo docs:** `README.md` (RL protocol), `HANDOFF.md` (current status), `NNKNN_AGENT_GUIDE.md` (schemas, caveats), `AGENTS.md`.

## Ground rules for the agent

0. **`Heuzi/NN-kNN` is a read-only reference repo.** Never push to it, never open PRs against it, never rewrite its history or touch `main` in any way. It is the source to clone from and the documentation to read — nothing else. All code changes, experiment artifacts, logs, and figures live exclusively on the local `rl-iclr2027` branch of your clone (push that branch to a fork or a separate remote if one is provided; otherwise keep it local and report paths).
1. Work in a dedicated clone and branch:
   ```bash
   git clone https://github.com/Heuzi/NN-kNN.git nnknn-work   # private repo: needs a token or deploy key
   cd nnknn-work && git checkout -b rl-iclr2027
   ```
   All commands below run from the repo root (`nnknn-work/`). All progress reports go to `reports/experiment_log.md` and `reports/figures/` inside the clone; commit them to `rl-iclr2027` after each phase. Per rule 0, do not push this branch to `Heuzi/NN-kNN`: if a results remote (fork or separate repo) is configured, push there (`git remote add results <url> && git push -u results rl-iclr2027`); otherwise keep commits local and include the log/figure paths in your phase report.
2. Never draw paper-style conclusions from `smoke` or `debug` profiles; they validate plumbing only.
3. Every claim about a run must come from its `results/rl/<run>/summary.json`: report `final_eval.mean_return` (selected checkpoint), `last_eval.mean_return`, `actual_timesteps`, `stopping_reason`, and `training_efficiency.budget_interpretation`. If `budget_interpretation` is `unsolved_or_underfit`, or best==final with a rising curve, say so — do not present the run as solved.
4. Artifacts predating `algorithm="nnknn_actor_critic_separate_memory_gae"` (including the legacy 369.5 run `nnknn_rl_cartpole_20260626_150805_689987`) are outdated; retrain, never compare against them.
5. Long runs: launch with `nohup ... > run.log 2>&1 &` and poll; each run also writes `stdout.log`/`stderr.log` in its timestamped results folder.

## Phase 0 — Environment (once)

Manage Python and dependencies with **uv**, installing the **latest** package versions (ignore the pins in `requirements.txt`; do not run `codex/setup.sh`):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # if uv is not already on the server
uv venv .venv --python 3.12                              # recent stable CPython; bump if all deps support newer
sed -E 's/[=<>!~]=.*$//; /^--/d' requirements.txt > requirements-latest.txt   # package names only, no pins
uv pip install --python .venv/bin/python -U -r requirements-latest.txt
mkdir -p reports && uv pip freeze --python .venv/bin/python > reports/env_freeze.txt   # record what "latest" resolved to
.venv/bin/python verify_packages.py
.venv/bin/python codex/smoke_test.py --mode imports
.venv/bin/python codex/smoke_test.py --mode rl
.venv/bin/python codex/smoke_test.py --mode nec
.venv/bin/python codex/smoke_test.py --mode nnknn_rl
```

Notes: on Linux, latest `torch` from PyPI ships CUDA-enabled wheels, so the pinned `--extra-index-url` (cu128) is not needed; for a CPU-only server install torch with `uv pip install -U torch --index-url https://download.pytorch.org/whl/cpu` before the `-r` install. Commit `reports/env_freeze.txt` with the first push — results are only interpretable against a recorded resolution. If a smoke test fails on a latest-version API break (likely candidates: numpy 2.x, gymnasium API drift), fix forward with a minimal patch on the branch and note it in the log; only if breakage cascades, fall back to the pinned env (`bash codex/setup.sh`, needs Python 3.11.9) and report the blocking package.

Check CUDA with `.venv/bin/python -c "import torch; print(torch.cuda.is_available())"`. If CUDA is available, pass `--device cuda` on every run below; a full six-variant `fast` sweep is impractical on CPU (previously aborted after hours on the first variant). Do not enable torch deterministic mode — deterministic-CUDA settings are known to conflict with some baseline ops.

Success: all smoke modes pass.

## Phase 1 — Re-establish baselines

**1a. DQN (currently broken — highest priority).** Last fast run selected 110.85 vs. the 475 threshold. Reproduce, then debug:

```bash
.venv/bin/python tools/run_rl_dqn.py cartpole --profile fast --seed 0 --device cuda
```

If again unsolved: inspect `eval_metrics.csv` (flat vs. collapsing curve), `loss_metrics.csv` (divergence), and diff the config against the older solved checkpoint referenced in `HANDOFF.md`. Ordered suspects: exploration schedule (`--exploration-fraction`, final epsilon), target-network sync interval, learning rate, buffer/warmup sizes vs. CleanRL reference values. Fix on the branch, rerun, and record the diff. DQN must reach ≥475 mean (20 eval episodes) on at least 1 seed before Phase 4 comparisons mean anything.

**1b. NEC (near-solved reference, 450.55).** Confirm reproducibility:

```bash
.venv/bin/python tools/run_rl_nec.py cartpole --profile fast --seed 0 --device cuda
```

Success: within ~±30 of 450.55. If it lands near threshold, try one longer budget on one seed to test the under-budget hypothesis from `HANDOFF.md`: `--total-timesteps 300000 --no-early-stopping`.

## Phase 2 — Current-architecture NN-kNN-RL reference numbers

Replace the stale legacy number with fresh runs of the maintained six variants (one seed each first):

```bash
.venv/bin/python tools/run_rl_nnknn.py cartpole --profile fast --seed 0 --device cuda                          # nnknn actor + mlp critic (default)
.venv/bin/python tools/run_rl_nnknn.py cartpole --profile fast --seed 0 --device cuda --critic-type nnknn
.venv/bin/python tools/run_rl_nnknn.py cartpole --profile fast --seed 0 --device cuda --critic-type nnknn --critic-mutable-value-labels --critic-trainable-value-labels   # hybrid labels
.venv/bin/python tools/run_rl_nnknn.py cartpole --profile fast --seed 0 --device cuda --actor-type mlp --critic-type mlp    # parametric control
.venv/bin/python tools/run_rl_nnknn.py cartpole --profile fast --seed 0 --device cuda --actor-type mlp --critic-type nnknn
.venv/bin/python tools/run_rl_nnknn.py cartpole --profile fast --seed 0 --device cuda --critic-type nnknn --critic-value-label-activation-threshold 0 --critic-target-sync-interval 4
```

(`tools/run_cartpole_all_variants.py` exists as a sweep driver; use it only if it runs variants sequentially on GPU without editing.) For each run record the summary fields plus NN-kNN diagnostics: `case_entries`, `critic_case_entries`, `*_cases_pruned`, `critic_target_value_syncs`, `critic_label_updates`, and the `critic_holdout_metrics.csv` trend (`critic_holdout_explained_variance`). Rank variants by `final_eval.mean_return`, breaking ties with `training_efficiency.best_model_step`.

## Phase 3 — Tune toward Gate G1 (CartPole ≥475)

Take the best Phase 2 variant and tune one knob at a time, in this order (rationale: README design notes + the IJCAI-26 regression lessons):

1. **Critic learning dynamics:** `--critic-learning-rate {1e-3, 3e-4}`, `--critic-update-epochs {1, 2, 4}`.
2. **Case-level group:** `--case-learning-rate` (per-case biases/weights/trainable labels; NEC precedent says this group should move faster than the trunk).
3. **Value-label mode:** fixed vs. `--critic-mutable-value-labels` vs. `--critic-trainable-value-labels` vs. both (hybrid), with `--critic-value-label-update-alpha` and `--critic-value-label-activation-threshold {0, default}`.
4. **Target critic:** `--critic-target-value-mode {ema, hard, none}`, `--critic-target-sync-interval {1, 4, 16}`, `--critic-target-ema-tau`.
5. **Memory pressure:** case capacity (config default 500/profile) at {100, 500, 2000} — watch runtime; retrieval cost grows with active cases.
6. **GAE/exploration:** `--gamma 0.99 --gae-lambda {0.9, 0.95}`, epsilon schedule flags.

Watchpoints tied to known failure modes: if `critic_holdout_explained_variance` stays low while optimization MSE falls, the critic is interpolating from label-inconsistent neighbors (the regression paper's pathology) — prefer sparsemax-side/locality settings from `config.yaml` and flag for a code-level locality loss if no flag exists. If actor case count saturates at capacity with heavy pruning churn, lower insertion pressure (positive-advantage threshold) before growing capacity.

**Gate G1 check:** best config reaches `final_eval.mean_return ≥ 475` over 20 eval episodes. Then confirm on seeds {0,1,2,3,4}:

```bash
for s in 0 1 2 3 4; do .venv/bin/python tools/run_rl_nnknn.py cartpole --profile fast --seed $s --device cuda <best-flags>; done
```

Report mean ± std across seeds, plus per-seed `best_model_step` (sample-efficiency proxy vs. DQN/NEC/MLP-AC).

## Phase 4 — Fixed-budget comparison (only after G1)

Paper-style comparison at matched budgets, early stopping off, all four methods (DQN, NEC, best NN-kNN-RL, MLP-AC), seeds {0..4}:

```bash
.venv/bin/python tools/run_rl_dqn.py  cartpole --profile fast --seed $s --device cuda --no-early-stopping --total-timesteps 150000
.venv/bin/python tools/run_rl_nec.py  cartpole --profile fast --seed $s --device cuda --no-early-stopping --total-timesteps 150000
.venv/bin/python tools/run_rl_nnknn.py cartpole --profile fast --seed $s --device cuda --no-early-stopping --total-timesteps 150000 <best-flags>
.venv/bin/python tools/run_rl_nnknn.py cartpole --profile fast --seed $s --device cuda --no-early-stopping --total-timesteps 150000 --actor-type mlp --critic-type mlp
```

Deliverable: a results table (method × mean return ± std × best_model_step × steps-to-475) in `reports/experiment_log.md`, plus learning-curve PNGs generated from each run's `eval_metrics.csv` into `reports/figures/`. Commit the branch (push to the results remote only, per rule 0).

## Phase 5 — Next environments (proposal Tier 1/2; only after Phase 4)

Acrobot/LunarLander require extending `datasets/rl_tasks.py` (registry pattern; current gymnasium uses `LunarLander-v3` and needs `uv pip install -U "gymnasium[box2d]"`). MinAtar needs `uv pip install -U minatar` plus a small wrapper. Refresh `reports/env_freeze.txt` after any dependency change. Propose the diff, run smoke, then repeat Phases 2–4 per environment. Keep all code changes on `rl-iclr2027` for later PR review.

## Stop conditions

Stop and report back (do not keep burning compute) if: (a) Phase 1a DQN cannot be fixed within ~6 attempts — the eval protocol itself may be suspect; (b) no Phase 3 config beats 400 after the ordered knob sweep — the Gate G1 descope decision is the humans' call; (c) any smoke test fails after a code edit.

## reports/experiment_log.md entry template

```markdown
## <date> — <phase> — <one-line outcome>
Runs: <results/rl/ folder names>
Command(s): <exact CLI>
Key numbers: final_eval X / last_eval Y / actual_timesteps Z / budget_interpretation W
Diagnostics: <case counts, holdout EV, anomalies>
Interpretation & next step: <2-3 sentences>
```

## Kickoff prompt (paste into Claude Code on the server)

> Clone https://github.com/Heuzi/NN-kNN.git as `nnknn-work`, check out a new branch `rl-iclr2027`, then read `CLAUDE_CODE_REPRODUCTION_PLAN.md` (copy this file into the repo root first) and follow it phase by phase. Heuzi/NN-kNN is a read-only reference repo — never push anything to it. After each phase, append the log entry to `reports/experiment_log.md`, commit locally (push only to a results remote if I give you one), and show me the summary table before starting the next phase. Ask before any run you expect to exceed 2 hours.
