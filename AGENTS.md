# AGENTS.md

## Repository expectations

- This repository is primarily a Python regression/classification research workflow centered on `model/` and `datasets/`, with RL/DQN, RL/NEC, RL/PPO, RL/TD3, and NN-kNN-RL paths for CartPole and future RL work, including an ALE/Atari image-observation path.
- For regression work, prefer the helpers in `model/regression_workflow.py` instead of calling low-level training code directly.
- Regression configs must explicitly set `task_type="regression"`. The notebook does this already; ad hoc scripts should too.
- For RL baseline work, prefer `model/rl_workflow.py`, `model/nec_workflow.py`, `model/ppo_workflow.py`, `model/td3_workflow.py`, `model/nnknn_rl_workflow.py`, `model/cnn_encoders.py`, `datasets/rl_tasks.py`, `tools/run_rl_dqn.py`, `tools/run_rl_nec.py`, `tools/run_rl_ppo.py`, `tools/run_rl_td3.py`, and `tools/run_rl_nnknn.py` instead of notebook-only implementations.
- `RLTaskSpec.env_kwargs` carries per-task `gymnasium.make` arguments (a tuple of key/value pairs) and is threaded through the shared `model/rl_workflow.py::_make_env`, so every workflow sees the same environment. Tasks that declare none keep the original `gym.make(env_id)` call exactly.
- `RLTaskSpec.observation_kind` selects the observation path. `flat_box` is the 1-D Box path every classic-control/Box2D/MinAtar task uses (MinAtar grids are flattened by `_make_env`). `image_atari` is the ALE path: `_make_env` builds the base env with `frameskip=1` and applies `AtariPreprocessing(noop_max=30, frame_skip=4, screen_size=84, grayscale_obs=True, scale_obs=False)` then `FrameStackObservation(4)`, yielding a `(4, 84, 84)` uint8 Box. Requires `ale-py` and `opencv-python`.
- Every workflow resolves its observation through `model/rl_workflow.py::resolve_env_spaces` / `describe_observation_space`, which return an `ObservationSpec` (kind, shape, flat dim, replay dtype). Image tasks swap the MLP for the ONE shared Nature-CNN encoder in `model/cnn_encoders.py` (conv 32x8s4, 64x4s2, 64x3s1, fc 512 + ReLU): DQN's Q-net, NEC's embedding net, PPO's actor and critic trunks (separate trunks, orthogonal init), and the NN-kNN actor/critic feature extractor. The encoder casts to float and divides by 255 itself, so replay buffers and rollout lists keep frames as `uint8`. Flat-observation paths are byte-for-byte unchanged.
- On image tasks the NN-kNN case stores hold RAW stacked frames and the CNN is the case-distance feature extractor; with `share_nnknn_representation` on (the default) actor and critic share that one CNN while keeping separate case memories and per-case parameters — the image analogue of the shared-trunk / separate-heads convention. The lagged target critic deep-copies the encoder, so its encoder parameters stay distinct and follow through target EMA.
- Case-activation thresholds are in CASE-FEATURE units, so they do not transfer from flat tasks to image tasks. On flat tasks distances are raw observation distances and re-presenting a stored case gives distance exactly 0, so the default `critic_value_label_activation_threshold=0.0` matches. Through the CNN the same re-presentation lands at ~1e-7 (float32 kernels differ between the query and case passes) and typical case distances are O(0.1), so the flat default never engages and every mutable-label sample appends instead of updating. Any image run using `critic_mutable_value_labels` must set `--critic-value-label-activation-threshold` in feature units (a small negative value); the same caveat applies to `case_prune_bias_threshold`.
- ALE timesteps count POST-frameskip agent steps: 1 step = 4 emulator frames. `ale_pong`/`ale_breakout` thresholds are documented NON-canonical progress markers (Nature DQN reaches roughly +18 Pong / ~400 Breakout only at 10M+ steps), so ALE work is exploratory: report partial learning curves, not solved games. The runners cap image-task `eval_episodes` at `IMAGE_TASK_EVAL_EPISODES` (5) unless `--eval-episodes` is given; the cap never raises a smaller profile default. A greedy policy that never presses FIRE idles a Breakout episode to the 27,000-step cap, so keep Breakout eval counts small.
- All five MinAtar games are registered (`minatar_asterix`, `minatar_breakout`, `minatar_freeway`, `minatar_seaquest`, `minatar_space_invaders`). MinAtar publishes no solve threshold, so every marker there is documented NON-canonical too.
- Current NN-kNN-RL is an on-policy actor-critic workflow: the actor and critic are selectable between NN-kNN and MLP variants, and GAE supplies actor advantages and value targets.
- MLP actors are standard stochastic-policy baselines: sample directly from their softmax policy during training, use entropy regularization, and keep effective epsilon at zero. Reserve readiness-driven uniform sampling and scheduled epsilon mixing for NN-kNN actors.
- When both actor and critic are NN-kNN, use separate actor and critic case bases and per-case parameters. Share only the state feature extractor/global feature-distance module, analogous to a shared neural trunk with separate policy and value heads.
- Keep rollout case insertion staged: compute actor/critic losses against the frozen rollout representation, then insert only raw-positive-advantage actor recommendations and update/append critic value cases after optimization.
- NN-kNN critic value labels default to fixed GAE targets; optional mutable labels, trainable labels, or the combined hybrid mode are controlled by `critic_mutable_value_labels` and `critic_trainable_value_labels`.
- Trainable NN-kNN value labels use the same case-level optimizer group as case biases and per-case glocal weights; use `case_learning_rate` rather than a label-only learning rate.
- Treat the hybrid label mode as NEC-like two-timescale memory learning, but keep labels as expected GAE/TD targets rather than max-return episodic memory. Mutable labels use a raw `case_bias - distance` activation threshold and aggregate matching batch targets before one EMA update.
- NN-kNN critics use a lagged target critic by default. Online and target critics share the raw state-case tensor and aligned stable case IDs, while value labels, biases, per-case weights, and encoder parameters remain distinct and update through target EMA.
- NN-kNN-RL trains the final partial rollout at the fixed step budget, uses episode-boundary-aware GAE, validates `reward_shaping`, and reports actor/critic case maintenance separately.
- PPO is the on-policy reference baseline: clipped surrogate loss over a fixed-length single-env rollout buffer with the same episode-boundary-aware GAE convention as NN-kNN-RL (truncations bootstrap, terminations do not), and it also trains the final partial rollout. Its policy head follows the task spec's `action_kind`: categorical for discrete tasks, diagonal Gaussian for continuous ones. Continuous actions are clipped to the Box bounds and scored unclipped (no tanh squashing).
- TD3 is the continuous-control reference baseline: twin critics, delayed policy and target updates (`policy_frequency` 2), target policy smoothing (noise 0.2 clipped to 0.5, in normalized action units scaled by the actor's `action_scale`), exploration noise 0.1, Polyak tau 0.005, and a uniform replay buffer that stores `terminated` only, so time-limit truncations bootstrap. It is continuous-action ONLY and refuses discrete tasks with an explicit error. The NN-kNN actor is discrete-only (one-hot action labels), so the continuous tasks `pendulum` and `lunarlander_continuous` carry TD3 and PPO rows only — a continuous-action NN-kNN actor is future work, not attempted. Pendulum's `success_threshold`/`target_mean_return` are documented NON-canonical markers; gymnasium defines no Pendulum reward threshold.
- NN-kNN-RL defaults every profile to a 500-case actor/critic capacity; larger case bases must be explicit experiment overrides because exact retrieval cost grows materially with memory size.
- NN-kNN-RL critic holdout diagnostics must use separate stochastic behavior-policy rollouts, preserve the training RNG stream, exclude all holdout transitions from gradients and case insertion, and report discounted Monte Carlo metrics separately in `critic_holdout_metrics.csv`. Bootstrap only time-limit truncations, preferably with the lagged target critic.
- Treat notebooks named `Outdated...` as archival only. Do not use them as implementation references for RL work.
- DQN, NEC, PPO, TD3, and NN-kNN-RL share evaluation-based early stopping: patience 30, minimum improvement 1.0, patience counting after 25,000 environment steps, and immediate stopping at task-maximum mean return. Fast enables it for every workflow, as does debug where available; smoke/gold disable it. Use gold or `--no-early-stopping` for strict fixed-budget comparisons, and interpret actual/configured timesteps plus `training_efficiency` before paper-style claims.
- Use synthetic datasets for quick validation unless a task specifically requires the larger real datasets in `datasets/`.

## Setup

- For Codex cloud tasks, use `bash codex/setup.sh` as the setup script.
- For cached Codex cloud environments, use `bash codex/maintenance.sh` as the maintenance script.
- The cloud setup assumes Python `3.11.9`, matching `python_version.txt`.

## Validation

- Fast import check: `python codex/smoke_test.py --mode imports`
- Fast training smoke test: `python codex/smoke_test.py --mode train`
- Fast RL smoke test: `python codex/smoke_test.py --mode rl`
- Fast NEC smoke test: `python codex/smoke_test.py --mode nec`
- Fast PPO smoke test: `python codex/smoke_test.py --mode ppo`
- Fast TD3 smoke test: `python codex/smoke_test.py --mode td3`
- Fast NN-kNN-RL smoke test: `python codex/smoke_test.py --mode nnknn_rl`
- Fast ALE / image-observation smoke test: `python codex/smoke_test.py --mode ale`

## Notes

- `checkpoints/` and transient artifacts are expected and are gitignored.
- RL runs write timestamped artifacts under `results/<host>/` (`r760` or `g234`);
  pass `--output-dir` explicitly. See `results/README.md`.
- `matplotlib` should run headlessly in cloud tasks with `MPLBACKEND=Agg`.
- Avoid full multi-run benchmarks for routine verification; they are intentionally expensive.
