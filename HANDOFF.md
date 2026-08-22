# Handoff

Last updated 2026-08-22 on branch `rl-iclr2027` (clone of read-only upstream
`Heuzi/NN-kNN`, branched from `c097195`; never push to upstream). The canonical
chronological experiment record is `reports/experiment_log.md`; this file is
the durable summary of where things stand and what to do next.

## Current Status

### Environment

- The working environment is uv-managed: `.venv` with Python 3.12 and the
  latest package versions (ignore the pins in `requirements.txt`;
  `requirements-latest.txt` holds the stripped package list). The resolved
  stack is recorded in `reports/env_freeze.txt`: torch 2.13.0+cu130,
  numpy 2.5.1, gymnasium 1.3.0, scipy 1.18.0, scikit-learn 1.9.0. GPU: H100
  NVL. All four smoke modes (`imports`, `rl`, `nec`, `nnknn_rl`) pass on this
  stack; no numpy-2.x or gymnasium API breaks surfaced. One fix-forward patch:
  `verify_packages.py` now imports `IPython` (was `ipython`).
- Refresh `reports/env_freeze.txt` after any dependency change.

### RL headline results (2026-07-31 through 2026-08-22 campaign)

- **Gate G1 of RESEARCH_PROPOSAL_ICLR2027.md (external document, not in this repo) is PASSED** (2026-08-01, ahead of
  the Aug 14 deadline). The untuned hybrid NN-kNN-RL config —
  `tools/run_rl_nnknn.py cartpole --profile fast --device cuda --critic-type
  nnknn --critic-mutable-value-labels --critic-trainable-value-labels` —
  reaches final_eval 497.98 ± 4.52 across seeds {0..4}, 5/5 seeds ≥ 475.
  Four of five seeds solve outright at 500.0 with target-score stops at
  4,474–13,156 steps; seed 4 reaches 489.90 (best step 114,344).
- **DQN is a solved CartPole reference again.** Fast seed 0 reaches 500.0
  (target-score stop at 95,000 steps; checkpoint reload 500.0 ± 0 over 100
  fresh-seed episodes). The previously recorded 110.85 failure does not
  reproduce on the current stack — same plateau dynamics through ~85k, then a
  breakout once epsilon floors. Treat the old failure note as stack/run-level
  variance, not a config bug.
- **NEC reproduces with wide seed variance.** Seed 0 solves at 500.0
  (selected checkpoint at step 2,056; 487.25 over 100 fresh-seed episodes);
  seed 1 plateaus at ~419–431 even with a 300k-step probe, so the old
  under-budget hypothesis is not supported. The old 450.55 reference sits
  inside this seed band.
- **CartPole fixed-budget table** (150k steps, no early stopping, seeds 0–4;
  `reports/phase4_manifest.json`, figures in `reports/figures/phase4_*`):
  NN-kNN-RL hybrid 497.98 ± 4.52 (best), MLP-AC 482.00 ± 35.97,
  DQN 466.13 ± 75.74, NEC 465.11 ± 30.98. The hybrid's median best_model_step
  (4,810) is ~15× earlier than DQN's (70k) and ~21× earlier than MLP-AC's
  (103k). Caveat: 14/20 runs are
  regressed_after_best — these are best-checkpoint-under-budget claims, not
  end-policy claims (DQN's last_eval mean is 140.85).
- **LunarLander-v3 and MinAtar/Breakout-v1 are registered, and seed-0
  batteries show NN-kNN-RL does NOT transfer untuned.** Registry:
  `lunarlander` (box2d, success 200 = gymnasium reward threshold,
  immediate-stop target 280; needs `gymnasium[box2d]`) and `minatar_breakout`
  (explicit MinAtar registration + observation flattening in the shared
  `_make_env`; 5,000-step cap; success 10 is a documented non-canonical
  progress marker, target 50). All CLIs smoke-pass both; CartPole regression
  smokes stay bit-identical (9.5 / 85.0 / 279.5).
  Seed-0 results — LunarLander (threshold 200): DQN **224.30 (solved)**,
  NEC −84.27, NN-actor+MLP-critic −121.39, NN-actor+hybrid-critic −121.74,
  MLP-actor+hybrid-critic −229.90, MLP-AC −234.98. MinAtar Breakout (marker
  10): DQN **13.60 (crosses)**, NEC 5.60, NN-actor+hybrid-critic 4.15,
  MLP-AC 1.40, MLP-actor+hybrid-critic 1.40.
  **Finding, and a reversal of Acrobot:** on both new tasks the gain inside
  the actor-critic family comes from the NN-kNN *actor*, not the critic
  (LunarLander NN-actor ~−121 vs MLP-actor ~−230 regardless of critic;
  MinAtar 4.15 vs 1.40 with the critic making no difference) — the opposite
  of Acrobot, where the critic was decisive and the actor starved. Both
  LunarLander NN-actor runs peak at ~9.2k steps then collapse (hybrid
  last_eval −599.22): the case memory locks in an early hovering policy and
  degrades. Value-based DQN dominates both tasks.
- **Acrobot (first Tier-1 transfer) is complete.**
  `datasets/rl_tasks.py` now registers `acrobot` (Acrobot-v1, success
  threshold −100 from the gymnasium reward threshold, immediate-stop target
  −60), and the three CLIs resolve task-level success protocol from the spec
  for non-CartPole tasks (CartPole behavior bit-identical; smoke-verified).
  Fixed-budget table (`reports/phase5_acrobot_manifest.json`, figures
  `reports/figures/phase5_acrobot_*`): DQN −72.94 ± 1.40 (5/5 seeds solve,
  crossing −100 by 20–50k), NN-kNN-RL hybrid −105.54 ± 53.47 (4/5 seeds solve
  at −79 to −85; seed 1 unsolved at −201 with the under-budget signature),
  NEC −327.34 ± 135.54 (0/5), MLP-AC −416.09 ± 187.63 (collapses to −500).
  **Budget follow-up (2026-08-22):** the seed-1 miss was budget, not
  architecture — a 300k `--no-early-stopping` probe reaches −75.75
  (best step 233,523). At adequate budget the hybrid is **−80.47 ± 3.45,
  5/5 seeds solve**. Keep the 150k number for matched-budget comparison and
  always attach the "seed 1 needed 300k" qualifier to the 5/5 claim.
  A one-knob-at-a-time pass (exploration fraction 0.2; case LR 1e-2) moved
  final_eval by under 1 point, so the late-crossing pattern is not an
  exploration- or LR-schedule artifact.
- **Key transfer finding:** on Acrobot the NN-kNN regression critic is the
  difference-maker — every NN-kNN-critic variant solves while both MLP-critic
  variants stay flat at −500 (the nnknn-actor + mlp-critic run starves at 85
  actor cases because no positive advantages ever appear). But DQN beats the
  hybrid on Acrobot, so CartPole cross-family sample-efficiency claims do not
  generalize untuned.
- **CartPole 12-configuration label-mode grid** ({NN, MLP} actor × {MLP,
  NN-fixed, NN-mutable, NN-trainable, NN-hybrid} critic + DQN + NEC, fast seed
  0): `reports/cartpole_12variant_sweep.csv` — **complete (12/12)**. Every
  configuration reaches ≥491.35 final_eval, so all actor/critic pairings
  work on CartPole. Within the NN-actor + NN-critic family the hybrid is
  the only member that reaches 500 *and holds it* (best==last at 4,810
  steps); fixed/mutable/trainable each peak early (7.5k–32k) then regress
  to 235–382 — the hybrid-necessity argument on the NN-actor side. The
  MLP-actor label ablation (figure
  `reports/figures/label_mode_ev_comparison.png`) originally showed, on seed
  0 alone: trainable = fast but collapses, fixed = slow but stable, mutable =
  fastest to 500 but weakest critic EV, hybrid = stability + adaptivity.
  **5-seed revision (2026-08-21):** trainable 499.45 ± 1.23 and fixed
  494.25 ± 8.03 across seeds 0–4, both 5/5 ≥ 475 — the seed-0 trainable
  collapse did not recur on seeds 1–4, and fixed had its own weak seed
  (481.4). On CartPole the label modes are within seed noise; treat the EV
  figure as a one-seed mechanism illustration, not a general ranking. The
  hybrid-necessity argument rests on the NN-actor side (hybrid 497.98 ± 4.52
  at Gate G1) and on Acrobot, not on this MLP-actor grid.

### Robustness caveats to respect when making claims

- Gate G1 is defined on the protocol metric (final_eval over the standard
  20-episode eval block). Under a stricter 100-episode fresh-seed-20000
  checkpoint-reload check, the hybrid seed means drop to 466.2 with 3/5 seeds
  ≥ 475 (seeds 1 and 4 degrade to ~431/~406 — partial overfit of checkpoint
  selection to the 20 fixed eval starts). Cite the protocol metric for G1 and
  mention the fresh-seed check for robustness statements.
- Best-vs-last gaps are training instability/forgetting, not overfitting in
  the supervised sense; diagnose critic overfitting via the
  `critic_train_mse` vs `critic_holdout_explained_variance` gap instead.
- Holdout EV and episode-frequency evals fire every 100 completed episodes,
  so late-training coverage is sparse (~30–35k-step gaps once episodes run
  ~350–500 steps; the last measurement lands anywhere from ~119k to ~148k in a
  150k run depending on episode lengths). EV is also ill-conditioned near a solved policy (target variance
  shrinks). Densify via `critic_holdout_episode_frequency` if a paper figure
  needs late coverage.
- DQN implementation caveat: this codebase deviates from CleanRL (smooth-L1 +
  grad-clip 0.5 vs MSE + no clip). Transplanting CleanRL reference
  hyperparameters diverges catastrophically (final_eval 9.45, Q → ~5.5e5, peak ~5.9e5);
  lr 2.5e-4 with per-step training also solves (494.85). The maintained
  profile defaults are fine; solving is not knife-edge on the learning rate.
- Config-equivalence gotcha: plan variant 6
  (`--critic-value-label-activation-threshold 0
  --critic-target-sync-interval 4`) exactly equals the `--critic-type nnknn`
  defaults (threshold 0.0, sync 4, target mode "ema") — do not re-run it as a
  distinct variant.
- NN-kNN-actor runs are wall-clock slow (roughly 1–2.5 h per 150k CartPole
  run on an uncontended machine per collaborator timings; 633–1,026 min
  observed here under multi-job GPU contention) because the actor does a single-state
  exact-kNN retrieval every environment step; the NN-kNN critic adds almost
  nothing over MLP (batched queries only). Speed-ups worth trying if it becomes
  blocking: smaller case base, sampled/approximate retrieval, batching/caching
  on the actor's per-step query path.
- Silent-flag gotcha: with `--actor-type nnknn --critic-type nnknn` and the
  default shared representation, `make_nnknn_rl_config` overwrites
  `critic_learning_rate` with the base `learning_rate`
  (`model/nnknn_rl_workflow.py:1380`), so `--critic-learning-rate` is accepted
  and then ignored. A knob-pass run using it produced a config bit-identical
  to baseline. Tune that group with `--learning-rate`, or use
  `--case-learning-rate` for case parameters.
- Operational gotcha (torch 2.13): `--device cpu` runs still touch CUDA via
  the Adam accelerator health-check, so they crash with CUDA OOM when the GPU
  is fully occupied by other jobs. Launch CPU runs with
  `CUDA_VISIBLE_DEVICES="" ` to make them truly CPU-only.

### Classification (unchanged by the RL campaign)

- Classification uses normalized case activation as class probability mass
  with NLL loss; it does not restore the older class-weight formulation. Do
  not restore the retired legacy case-weight path solely to reproduce old
  numbers.
- June 1 10-fold reruns: `iris` 0.9600 ± 0.0562, `zebra` 0.5182 ± 0.1488,
  `zebra_special` 0.5227 ± 0.0890 (in
  `results/classification_nnknn_rerun_20260601_103353/`). Zebra (a)/(b) are
  near chance and remain the classification-side debugging priority; then
  rerun the remaining small suite (`wine`, `breast_cancer`, `balance`,
  `digits`). Use `nnknn_sample_classification.ipynb` for inspection.

## Pending / Next Steps

**All experiment tasks from the reproduction plan and the previous handoff are
complete** (final harvest 2026-08-22; every run committed, every number in
`reports/experiment_log.md`). Nothing is in flight. What remains is
manuscript-facing work plus one item only the human can do.

### Needs the human

1. **A results remote.** `rl-iclr2027` exists only in this local clone (never
   push to `Heuzi/NN-kNN`). Provide a fork or separate repo URL and the branch
   can be pushed.
2. **`RESEARCH_PROPOSAL_ICLR2027.md` is not in the clone.** Gate G1 was
   verified against the plan's restatement of it. Drop the file in so future
   claims can be checked against the remaining gates.
3. **Decisions on the manuscript scope below** (which of items 1-4 to build).

### Manuscript work, in priority order

The evidence supports a *technique paper*: an episodic/CBR value-and-policy
memory inside an on-policy actor-critic (NEC-style fast writes plus slow
gradient refinement) that is far more sample-efficient than parametric AC in
classic control, with an honest negative transfer result beyond it. ICLR
convention for this type: method is the center of gravity, Related Work after
the experiments, a Limitations paragraph rather than a threats section, and a
long appendix (full tables, hyperparameters).

1. **A PPO baseline (highest reviewer risk).** The current MLP-AC is A2C-like.
   Reviewers will expect PPO as the standard on-policy control. Medium code
   lift: a clipped surrogate objective over the existing rollout/GAE
   machinery in `model/nnknn_rl_workflow.py`, or a repo-native CleanRL-style
   PPO alongside `model/rl_workflow.py`.
2. **Capacity ablation** (`case_capacity` 100 / 500 / 2000 — plan knob 5,
   never run). Doubles as the compute-cost story: it turns the wall-clock
   weakness into a measured retrieval-cost/performance trade-off instead of
   an omission a reviewer finds.
3. **Environment breadth.** Two classic-control wins plus two negative
   transfers is a defensible but thin suite. Extending MinAtar to all five
   games is now ~10 lines of registry per game plus the standard
   4-method x 5-seed protocol.
4. **An interpretability figure** — which stored cases activate along a
   swing-up trajectory. Cheap, differentiates the CBR narrative, and ICLR
   rewards it.
5. **Cheap rigor upgrades:** 10 seeds on the CartPole headline rows; report
   protocol *and* fresh-seed metrics side by side; a per-step timing table
   (NN-actor ~1-2.5 h vs MLP-actor minutes per 150k run).

### Claims the current evidence supports (and their required caveats)

- CartPole Gate G1: hybrid 497.98 +/- 4.52 over 5 seeds, ~15x fewer samples to
  its best checkpoint than DQN and ~21x fewer than MLP-AC at matched 150k
  budgets. Caveat: best-checkpoint-under-budget, not end-policy (14/20 runs
  regressed after best); fresh-seed reload gives 466.2 with 3/5 >= 475.
- Acrobot: the NN-kNN critic is *necessary* — every MLP-critic variant stays
  flat at -500, every NN-kNN-critic variant solves. Hybrid is -80.47 +/- 3.45,
  5/5, with the "seed 1 needed 300k" qualifier. Caveat: DQN is better and
  faster on this task (-72.94 +/- 1.40, crossing by 20-50k).
- Label modes: on the NN-actor side hybrid is the only variant that reaches
  and holds 500. On the MLP-actor side the modes are within seed noise
  (trainable 499.45 +/- 1.23 vs fixed 494.25 +/- 8.03) — the EV figure is a
  one-seed mechanism illustration, not a ranking.
- Negative transfer: on LunarLander and MinAtar the method does not compete
  untuned; the gain inside the AC family comes from the NN-kNN actor there,
  reversing the Acrobot pattern. State this as a scope boundary.
- Reproducibility: manifests, env freeze, git-tracked run artifacts, and
  fresh-seed robustness checks are all in place and worth a short paragraph.

## Current Work Focus (architecture — documents code behavior)

- The RL scope now covers `cartpole` and `acrobot` (see
  `datasets/rl_tasks.py`; `RLTaskSpec` gained optional `success_threshold` and
  `target_mean_return` fields, and all three CLIs gained
  `--success-threshold`, resolving the task spec for non-CartPole tasks).
- `tools/run_rl_dqn.py` also gained optional hyperparameter override flags
  (learning rate, buffer, gamma, target sync, batch, epsilon schedule, warmup,
  train frequency, grad clip); defaults unchanged.
- The NN-kNN-RL design remains an on-policy actor-critic workflow with
  selectable NN-kNN or MLP actors, selectable MLP or NN-kNN regression value
  critics, and GAE. Current training behavior:
  - rollout is on-policy and does not mutate case memory; complete or final
    partial batches compute GAE and optimizer losses against the frozen
    rollout representation before any actor/critic cases are updated
  - GAE masks value bootstrapping with `terminated` and stops lambda recursion
    with an episode-boundary mask, so traces do not cross truncated episode or
    final partial-rollout boundaries
  - NN-kNN actor loss uses all raw advantages, including negative advantages;
    only raw-positive-advantage transitions are inserted afterwards as
    `(state, recommended_action)` cases
  - MLP actors are standard stochastic-policy baselines: training samples
    directly from `pi(a|s)` with entropy regularization and effective epsilon
    `0`; the readiness/random and scheduled epsilon behavior applies only to
    NN-kNN actors
  - NN-kNN critics train on every rollout target, then update/append
    `(state, V_target)` cases regardless of advantage sign
  - NN-kNN critic value labels default to fixed GAE targets; optional mutable
    labels use raw `case_bias - distance` activation thresholding and
    aggregate matching batch targets before one EMA update; optional trainable
    labels make labels optimizer parameters, and both options form the hybrid
    mode
  - trainable value labels use the same case-level optimizer group as case
    biases and per-case glocal weights; tune this group with
    `case_learning_rate` / `--case-learning-rate`
  - the hybrid label mode is intentionally NEC-like: fast memory-value updates
    plus slower differentiable training, but labels should remain GAE/TD-style
    expected value targets rather than max-return episodic memory
  - when both sides are NN-kNN, their losses are computed before one joint
    optimizer step so the shared representation remains consistent with the
    rollout policy; their memories and per-case parameters remain separate
  - every NN-kNN critic can use a lagged target critic for GAE bootstrap
    values; online and target critics share raw state cases/stable IDs, while
    target labels, biases, per-case weights, and encoder parameters remain
    separate and EMA-update on the configured synchronization interval
  - training keeps stochastic action selection (`greedy=False`) and reserves
    greedy action selection for evaluation; NN-kNN uses readiness-driven
    random sampling and scheduled epsilon mixing, while MLP samples its policy
    directly
  - actor and critic maintenance runs at batch boundaries, reports each store
    separately, and clears stale Adam state after per-case compaction
  - every NN-kNN-RL profile defaults to `case_capacity=500`; larger capacities
    are explicit ablations because exact retrieval becomes the dominant
    runtime cost
  - critic reporting separates `critic_optimization_mse` from periodic
    post-update in-sample fields prefixed `critic_train_`; these diagnostics
    do not affect gradients, checkpoint selection, or early stopping and are
    not a held-out critic validation set
  - periodic critic holdout rollouts use the current stochastic behavior
    policy with independent seeds, are excluded from gradients and all case
    memories, and report Monte Carlo-return MSE/explained variance separately
    in `critic_holdout_metrics.csv`; truncations alone bootstrap from the
    lagged target critic
- DQN, NEC, and NN-kNN-RL use one configurable early-stopping protocol.
  `fast` enables it for every workflow, as does `debug` where available;
  `smoke` and `gold` disable it. Defaults are 30 stale evaluation checkpoints,
  `min_delta=1.0`, patience counting only after 25,000 environment steps, and
  immediate stopping at the task-target mean return (500 for CartPole, −60
  for Acrobot). On Acrobot, early stopping never fires in practice — all fast
  runs exhaust the 150k budget, which makes completed fast Acrobot runs
  protocol-equivalent to `--no-early-stopping` runs and reusable in
  fixed-budget tables (8 of the 20 Phase 5 table runs were reused this way).
- Use `gold` or `--no-early-stopping` for strict fixed-budget comparisons.
  Always inspect `configured_total_timesteps`, `actual_timesteps`,
  `early_stopping`, and `training_efficiency` in `summary.json`.
- Treat fixed-budget results as paper-useful only when the budget appears
  adequate. If `budget_interpretation` is `unsolved_or_underfit`, or if the
  best model is the final model and the curve is still rising, increase the
  budget or tune before comparing methods. Use `best_model_step` and
  `first_success_step` as sample-efficiency proxies.
- Do not treat a poor smoke/debug run as a method result; smoke validates
  plumbing only.
- MLP artifacts without `actor_behavior_policy="standard_stochastic_policy"`
  predate the standard baseline behavior and should be treated as
  epsilon-mixed MLP variants rather than current MLP actor-critic results.

## Current Local Artifacts

- `results/rl/` run artifacts (minus model checkpoints) are now tracked in
  git — see `results/rl/README.md` for the conventions. `checkpoint.pt` /
  `*.pth` and root-level console logs / PID files are intentionally excluded;
  the structured files in each run directory are the canonical record.
  Folders ending in `_eval` hold `eval_summary.json` from checkpoint-reload
  robustness evaluations.
- Aggregates and interpretation live in `reports/`:
  - `reports/experiment_log.md` — chronological record (Phases 0–5 plus the
    12-grid entry); the run manifests are the citation mechanism for seed
    batches.
  - `reports/phase4_manifest.json` / `reports/phase5_acrobot_manifest.json` —
    method × seed → run-dir maps for the two fixed-budget tables.
  - `reports/cartpole_12variant_sweep.csv` — the 12-configuration grid.
  - `reports/figures/` — learning-curve and ablation figures (regenerate via
    `tools/make_phase4_report.py --manifest <manifest> [--task ...
    --success-threshold ... --ylim ... --prefix ...]`).
  - `reports/env_freeze.txt` — the resolved dependency stack.
- A run folder contains `config.json`, `training_metrics.csv`,
  `loss_metrics.csv`, `eval_metrics.csv`, `final_eval_episodes.csv`,
  `last_eval_episodes.csv`, `summary.json`, `manifest.json` (and
  `checkpoint.pt` locally). NN-kNN-RL folders add
  `critic_holdout_metrics.csv`, `algorithm`, `gae`, `actor_type`,
  `critic_type`, and comparison diagnostics; NN-kNN actor runs record
  `case_entries` and action-count fields, NN-kNN critic runs record
  `critic_case_entries`, plus per-store `*_cases_pruned`, `*_cases_replaced`,
  `partial_rollout_*`, `critic_target_value_syncs`, `critic_label_updates`.
- `summary.json` records both the selected best checkpoint evaluation
  (`final_eval`) and the end-of-budget model evaluation (`last_eval`), plus
  `training_efficiency` (`best_model_step`, `first_success_step`, fractions,
  best-vs-last gap, `budget_interpretation`).
- Current actor-critic checkpoints record
  `algorithm="nnknn_actor_critic_separate_memory_gae"`. Older reward-to-go
  and shared-case-base checkpoints are legacy and should be retrained rather
  than loaded. Artifacts predating this algorithm string (including the
  legacy 369.5 run) are outdated; retrain, never compare against them.
- `checkpoints/` and transient smoke/runtime artifacts remain gitignored;
  root checkpoint files such as `nnknn_regression_best.pth` may change when
  experiments run. Classification runs write under `checkpoints/` and
  `results/classification_<suite>_<timestamp>/` as before.

## Where Durable Guidance Lives

- `NNKNN_AGENT_GUIDE.md` — model descriptions, workflow entry points,
  RL/DQN/NEC protocol details, Table 1 protocol details, output schemas,
  restart/resume patterns, machine-specific caveats. (Its 'Current CartPole DQN fast result', 'Current CartPole NEC reference
  result', and 'Current NN-kNN-RL status' sections predate this campaign;
  where they conflict with this file or the experiment log, this file wins.)
- `CLAUDE_CODE_REPRODUCTION_PLAN.md` — the phase plan this campaign followed.
  Its Ground rules and Stop conditions remain in force for all future work:
  never push to `Heuzi/NN-kNN`; ask the human before any single run expected
  to exceed 2 hours; stop if any smoke test fails after a code edit; source
  every run claim from its `summary.json` (final_eval, last_eval,
  actual_timesteps, stopping_reason, budget_interpretation); append
  experiment-log entries using the plan's template and commit after each
  phase.
- `reports/experiment_log.md` — the chronological experiment record.
