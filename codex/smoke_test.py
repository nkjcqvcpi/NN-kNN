from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def run_import_smoke() -> None:
    from datasets.rl_tasks import list_supported_rl_tasks
    from model.regression_workflow import (
        list_supported_regression_benchmark_methods,
        list_supported_regression_datasets,
        make_regression_cfg,
    )
    from model.classification_workflow import (
        list_supported_classification_benchmark_methods,
        list_supported_classification_datasets,
    )

    datasets = list_supported_regression_datasets()
    methods = list_supported_regression_benchmark_methods()
    cls_datasets = list_supported_classification_datasets()
    cls_methods = list_supported_classification_benchmark_methods()
    rl_tasks = list_supported_rl_tasks()
    regression_cfg = make_regression_cfg({"task_type": "regression"})
    if regression_cfg.get("case_score_mode") != "bias_minus_distance":
        raise AssertionError("Generic regression runs must use the current bias-minus-distance case score.")
    if regression_cfg.get("normalize_over_cases") is not True:
        raise AssertionError("Generic regression runs must normalize case activations.")
    print("import smoke ok")
    print(f"synthetic datasets: {datasets['synthetic']}")
    print(f"benchmark methods: {methods}")
    print(f"classification small datasets: {cls_datasets['small']}")
    print(f"classification methods: {cls_methods}")
    print(f"rl tasks: {rl_tasks}")


def run_training_smoke() -> None:
    from model.regression_workflow import make_regression_cfg, run_single_nnknn_regression_experiment

    Path("checkpoints").mkdir(parents=True, exist_ok=True)

    cfg = make_regression_cfg(
        {
            "task_type": "regression",
            "training_epochs": 1,
            "checkpoint_path": "checkpoints/tmp_codex_smoke.pth",
            "batch_size": 32,
            "top_k": 8,
            "bias_manual_set": True,
            "bias_manual_value": 1.0,
        }
    )

    state = run_single_nnknn_regression_experiment(
        "linear_regression",
        cfg,
        dataset_kwargs={"n": 96, "d": 5, "noise_scale": 0.1},
        run_seed=0,
        dataset_seed=0,
        split_seed=0,
        standardize=False,
        checkpoint_label="codex_smoke",
    )

    rmse_raw = float(state["rmse_raw"])
    best_metric = float(state["best_metric"])
    print("training smoke ok")
    print(f"rmse_raw={rmse_raw:.6f}")
    print(f"best_metric={best_metric:.6f}")


def run_classification_smoke() -> None:
    from model.classification_workflow import make_classification_cfg, run_single_nnknn_classification_experiment

    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    cfg = make_classification_cfg(
        {
            "training_epochs": 1,
            "checkpoint_path": "checkpoints/tmp_codex_classification_smoke.pth",
            "batch_size": 32,
            "top_k": 3,
            "explanation_mode": True,
            "case_normalizer": "softmax",
        }
    )
    state = run_single_nnknn_classification_experiment(
        "iris",
        cfg,
        run_seed=0,
        split_seed=0,
        checkpoint_label="codex_smoke",
    )
    probabilities = state["class_probabilities"]
    if not torch.allclose(probabilities.sum(dim=1), torch.ones(probabilities.size(0)), atol=1e-5):
        raise AssertionError("Classification outputs do not sum to one.")
    if state.get("most_activated_cases") is None:
        raise AssertionError("Classification explanation outputs were not returned.")
    if state.get("most_activated_class_ids") is None:
        raise AssertionError("Classification explanation labels were not decoded.")
    print("classification training smoke ok")
    print(f"accuracy={float(state['accuracy']):.6f}")


def run_rl_smoke() -> None:
    from model.rl_workflow import EarlyStoppingTracker, make_dqn_config, train_dqn

    cfg = make_dqn_config("smoke", seed=0)
    if cfg.early_stopping_patience != 30:
        raise AssertionError("DQN early-stopping patience should default to 30 evaluations.")
    patience_probe = EarlyStoppingTracker(
        enabled=True,
        patience=30,
        min_delta=1.0,
        min_steps=0,
        target_score=500.0,
    )
    if patience_probe.update(10.0, 1):
        raise AssertionError("Early stopping should not trigger on the initial best evaluation.")
    for step in range(2, 31):
        if patience_probe.update(10.0, step):
            raise AssertionError("Early stopping triggered before 30 stale evaluations.")
    if not patience_probe.update(10.0, 31) or patience_probe.stopping_reason != "patience":
        raise AssertionError("Early stopping did not trigger after 30 stale evaluations.")
    gated_probe = EarlyStoppingTracker(
        enabled=True,
        patience=1,
        min_delta=1.0,
        min_steps=100,
        target_score=500.0,
    )
    gated_probe.update(10.0, 1)
    if gated_probe.update(10.0, 99) or gated_probe.evaluations_without_improvement != 0:
        raise AssertionError("Patience should not count evaluations before early_stopping_min_steps.")
    if not gated_probe.update(10.0, 100):
        raise AssertionError("Patience should begin counting at early_stopping_min_steps.")
    state = train_dqn("cartpole", cfg, progress=False)
    final_eval = state["final_eval"]
    if final_eval["episodes"] != cfg.eval_episodes:
        raise AssertionError("RL smoke evaluation did not run the configured number of episodes.")
    if not state["checkpoint_path"].exists():
        raise AssertionError("RL smoke did not write a checkpoint.")
    early_cfg = make_dqn_config(
        "smoke",
        seed=1,
        total_timesteps=64,
        eval_frequency=16,
        eval_episodes=1,
        early_stopping=True,
        early_stopping_target_score=0.0,
    )
    early_state = train_dqn("cartpole", early_cfg, progress=False)
    if early_state["summary"]["actual_timesteps"] != 16:
        raise AssertionError("DQN did not stop at its first target-reaching evaluation.")
    if early_state["summary"]["early_stopping"]["stopping_reason"] != "target_score":
        raise AssertionError("DQN summary did not record its early-stopping reason.")
    print("rl smoke ok")
    print(f"run_dir={state['run_dir']}")
    print(f"mean_return={float(final_eval['mean_return']):.6f}")


def run_nec_smoke() -> None:
    from model.nec_workflow import make_nec_config, train_nec

    cfg = make_nec_config("smoke", seed=0)
    if cfg.early_stopping_patience != 30:
        raise AssertionError("NEC early-stopping patience should default to 30 evaluations.")
    state = train_nec("cartpole", cfg, progress=False)
    final_eval = state["final_eval"]
    if final_eval["episodes"] != cfg.eval_episodes:
        raise AssertionError("NEC smoke evaluation did not run the configured number of episodes.")
    if not state["checkpoint_path"].exists():
        raise AssertionError("NEC smoke did not write a checkpoint.")
    early_cfg = make_nec_config(
        "smoke",
        seed=1,
        total_timesteps=64,
        eval_frequency=5,
        eval_episode_frequency=0,
        eval_episodes=1,
        early_stopping=True,
        early_stopping_target_score=0.0,
    )
    early_state = train_nec("cartpole", early_cfg, progress=False)
    if early_state["summary"]["actual_timesteps"] != 5:
        raise AssertionError("NEC did not stop at its first target-reaching evaluation.")
    if early_state["summary"]["early_stopping"]["stopping_reason"] != "target_score":
        raise AssertionError("NEC summary did not record its early-stopping reason.")
    print("nec smoke ok")
    print(f"run_dir={state['run_dir']}")
    print(f"mean_return={float(final_eval['mean_return']):.6f}")


def run_ppo_smoke() -> None:
    from pathlib import Path as _Path

    import numpy as np
    import torch.optim as optim

    from model.ppo_workflow import (
        ALGORITHM_NAME,
        ActionSpaceSpec,
        PPOAgent,
        compute_gae,
        describe_action_space,
        evaluate_ppo,
        load_ppo_checkpoint,
        make_ppo_config,
        ppo_update,
        resolve_task_action_kind,
        train_ppo,
    )

    cfg = make_ppo_config("smoke", seed=0)
    if cfg.early_stopping_patience != 30 or cfg.early_stopping_min_steps != 25_000:
        raise AssertionError("PPO early stopping must share the repo protocol (patience 30, min steps 25k).")
    if cfg.early_stopping or make_ppo_config("gold").early_stopping:
        raise AssertionError("smoke and gold profiles must disable early stopping.")
    if not make_ppo_config("fast").early_stopping or not make_ppo_config("debug").early_stopping:
        raise AssertionError("fast and debug profiles must enable early stopping.")
    fast_cfg = make_ppo_config("fast")
    if (fast_cfg.clip_coef, fast_cfg.gae_lambda, fast_cfg.gamma) != (0.2, 0.95, 0.99):
        raise AssertionError("PPO defaults must keep clip 0.2, GAE lambda 0.95, gamma 0.99.")
    if (fast_cfg.ent_coef, fast_cfg.vf_coef, fast_cfg.max_grad_norm) != (0.01, 0.5, 0.5):
        raise AssertionError("PPO defaults must keep entropy 0.01, value 0.5, grad-norm clip 0.5.")

    # The PPO GAE helper must agree with the NN-kNN-RL convention, including
    # the episode-boundary mask that stops traces leaking across episodes.
    from model.nnknn_rl_workflow import compute_gae as nnknn_compute_gae

    gae_kwargs = dict(
        rewards=[1.0, 1.0, 1.0, 1.0],
        values=[0.5, 0.25, 0.75, 0.5],
        next_values=[0.25, 0.75, 0.5, 0.0],
        terminated=[False, True, False, False],
        episode_boundaries=[False, True, False, True],
        gamma=0.99,
        gae_lambda=0.95,
    )
    ppo_advantages, ppo_targets = compute_gae(**gae_kwargs)
    reference_advantages, reference_targets = nnknn_compute_gae(**gae_kwargs)
    if not torch.allclose(ppo_advantages, reference_advantages, atol=1e-6):
        raise AssertionError("PPO GAE does not match the NN-kNN-RL advantage convention.")
    if not torch.allclose(ppo_targets, reference_targets, atol=1e-6):
        raise AssertionError("PPO GAE does not match the NN-kNN-RL value-target convention.")
    boundary_advantages, _ = compute_gae(
        rewards=[1.0, 1.0],
        values=[0.0, 0.0],
        next_values=[0.0, 0.0],
        terminated=[False, True],
        episode_boundaries=[True, True],
        gamma=0.99,
        gae_lambda=0.95,
    )
    if not torch.allclose(boundary_advantages, torch.tensor([1.0, 1.0]), atol=1e-5):
        raise AssertionError("PPO GAE leaked advantage across an episode boundary.")

    if resolve_task_action_kind("discrete") != "discrete" or resolve_task_action_kind("continuous") != "continuous":
        raise AssertionError("PPO did not resolve task action kinds.")

    state = train_ppo("cartpole", cfg, progress=False)
    final_eval = state["final_eval"]
    if final_eval["episodes"] != cfg.eval_episodes:
        raise AssertionError("PPO smoke evaluation did not run the configured number of episodes.")
    if not state["checkpoint_path"].exists():
        raise AssertionError("PPO smoke did not write a checkpoint.")
    if state["summary"]["algorithm"] != ALGORITHM_NAME:
        raise AssertionError("PPO summary did not record its algorithm name.")
    if state["summary"]["updates"] <= 0:
        raise AssertionError("PPO smoke ran no policy updates.")
    run_dir = _Path(state["run_dir"])
    for artifact in (
        "config.json",
        "summary.json",
        "manifest.json",
        "eval_metrics.csv",
        "training_metrics.csv",
        "loss_metrics.csv",
        "final_eval_episodes.csv",
        "last_eval_episodes.csv",
    ):
        if not (run_dir / artifact).exists():
            raise AssertionError(f"PPO run directory is missing {artifact}.")

    reloaded = load_ppo_checkpoint(state["checkpoint_path"])
    if reloaded["config"].profile != "smoke" or reloaded["action_spec"].kind != "discrete":
        raise AssertionError("PPO checkpoint reload did not preserve the config and action spec.")
    reload_metrics = evaluate_ppo(
        "cartpole",
        reloaded["model"],
        episodes=1,
        seed=cfg.eval_seed,
        device=reloaded["device"],
    )
    if reload_metrics["episodes"] != 1:
        raise AssertionError("PPO checkpoint reload evaluation did not run.")

    early_cfg = make_ppo_config(
        "smoke",
        seed=1,
        total_timesteps=64,
        rollout_length=32,
        eval_frequency=16,
        eval_episodes=1,
        early_stopping=True,
        early_stopping_target_score=0.0,
    )
    early_state = train_ppo("cartpole", early_cfg, progress=False)
    if early_state["summary"]["actual_timesteps"] != 16:
        raise AssertionError("PPO did not stop at its first target-reaching evaluation.")
    if early_state["summary"]["early_stopping"]["stopping_reason"] != "target_score":
        raise AssertionError("PPO summary did not record its early-stopping reason.")
    if early_state["summary"]["updates"] != 1:
        raise AssertionError("PPO did not train the final partial rollout before stopping.")

    # Continuous (diagonal Gaussian) head. Pendulum-v1 is exercised straight
    # through gymnasium so the continuous path is covered without adding an
    # unregistered task to datasets/rl_tasks.py.
    import gymnasium as gym

    continuous_env = gym.make("Pendulum-v1")
    continuous_env.action_space.seed(0)
    action_spec = describe_action_space(continuous_env.action_space)
    if action_spec.kind != "continuous" or action_spec.dim != 1:
        raise AssertionError("PPO did not describe the Pendulum Box action space.")
    if action_spec != ActionSpaceSpec.from_dict(action_spec.to_dict()):
        raise AssertionError("PPO action-space spec did not round-trip through its dict form.")
    if describe_action_space(gym.spaces.Discrete(3)).kind != "discrete":
        raise AssertionError("PPO did not describe a Discrete action space.")

    obs_dim = int(np.prod(continuous_env.observation_space.shape))
    continuous_cfg = make_ppo_config(
        "smoke",
        seed=0,
        rollout_length=32,
        num_minibatches=2,
        update_epochs=2,
    )
    device = torch.device("cpu")
    agent = PPOAgent(
        obs_dim,
        action_spec,
        hidden_sizes=continuous_cfg.hidden_sizes,
        log_std_init=continuous_cfg.log_std_init,
    ).to(device)
    if not agent.is_continuous:
        raise AssertionError("PPO agent did not select the continuous policy head.")

    observations: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    log_probs: list[float] = []
    rewards: list[float] = []
    terminated_flags: list[bool] = []
    boundaries: list[bool] = []
    obs, _ = continuous_env.reset(seed=0)
    try:
        for _ in range(continuous_cfg.rollout_length):
            obs_array = np.asarray(obs, dtype=np.float32)
            obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, value = agent.act(obs_tensor)
            if action.shape != (1, 1) or log_prob.shape != (1,) or value.shape != (1,):
                raise AssertionError("Continuous PPO head returned unexpected sample shapes.")
            env_action = agent.env_action(action)
            if env_action.shape != (1,):
                raise AssertionError("Continuous PPO env action has the wrong shape.")
            if not continuous_env.action_space.contains(env_action):
                raise AssertionError("Continuous PPO env action was not clipped into the Box bounds.")
            next_obs, reward, term, trunc, _ = continuous_env.step(env_action)
            observations.append(obs_array)
            next_observations.append(np.asarray(next_obs, dtype=np.float32))
            actions.append(action.view(-1).numpy().astype(np.float32))
            log_probs.append(float(log_prob.view(-1)[0].item()))
            rewards.append(float(reward))
            terminated_flags.append(bool(term))
            boundaries.append(bool(term or trunc))
            obs = next_obs if not (term or trunc) else continuous_env.reset(seed=1)[0]
    finally:
        continuous_env.close()

    out_of_bounds = torch.tensor([[5.0]], dtype=torch.float32)
    if float(agent.env_action(out_of_bounds)[0]) != float(action_spec.high[0]):
        raise AssertionError("Continuous PPO actions are clipped to the Box bounds, not squashed.")
    probe_obs = torch.as_tensor(np.asarray(observations[:2], dtype=np.float32), dtype=torch.float32)
    with torch.no_grad():
        deterministic_action, _, _ = agent.act(probe_obs, deterministic=True)
        distribution_mean = agent.policy_distribution(probe_obs).base_dist.loc
    if not torch.allclose(deterministic_action, distribution_mean, atol=1e-6):
        raise AssertionError("Deterministic continuous evaluation must use the Gaussian mean.")

    optimizer = optim.Adam(agent.parameters(), lr=continuous_cfg.learning_rate, eps=1e-5)
    before = agent.actor_logstd.detach().clone()
    update_metrics = ppo_update(
        agent,
        optimizer,
        continuous_cfg,
        observations=np.asarray(observations, dtype=np.float32),
        next_observations=np.asarray(next_observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        log_probs=np.asarray(log_probs, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        terminated=np.asarray(terminated_flags, dtype=bool),
        episode_boundaries=np.asarray(boundaries, dtype=bool),
        device=device,
    )
    if update_metrics["rollout_steps"] != continuous_cfg.rollout_length:
        raise AssertionError("Continuous PPO update did not consume the whole rollout.")
    if update_metrics["minibatch_updates"] <= 0:
        raise AssertionError("Continuous PPO update ran no minibatch steps.")
    if not np.isfinite(update_metrics["policy_loss"]) or not np.isfinite(update_metrics["value_loss"]):
        raise AssertionError("Continuous PPO update produced non-finite losses.")
    if torch.allclose(before, agent.actor_logstd.detach()):
        raise AssertionError("Continuous PPO update did not train the Gaussian log-std parameter.")

    # The registry carries continuous tasks (datasets/rl_tasks.py), so the full
    # registered path — task spec -> _make_env -> Gaussian head -> eval -> run
    # dir — is covered too, not just the raw-gymnasium probe above.
    from datasets.rl_tasks import get_rl_task_spec
    from model.ppo_workflow import _validate_ppo_env_spaces
    from model.rl_workflow import _make_env

    for continuous_task, expected_dim in (("pendulum", 1), ("lunarlander_continuous", 2)):
        spec = get_rl_task_spec(continuous_task)
        registered_env = _make_env(spec, seed=0)
        try:
            _obs_dim, registered_action_spec = _validate_ppo_env_spaces(registered_env, spec)
        finally:
            registered_env.close()
        if registered_action_spec.kind != "continuous" or registered_action_spec.dim != expected_dim:
            raise AssertionError(f"PPO did not accept the registered continuous task '{continuous_task}'.")
    registered_cfg = make_ppo_config(
        "smoke",
        seed=0,
        total_timesteps=64,
        rollout_length=32,
        num_minibatches=2,
        update_epochs=2,
        eval_frequency=64,
        eval_episodes=1,
    )
    registered_state = train_ppo("pendulum", registered_cfg, progress=False)
    if registered_state["summary"]["action_kind"] != "continuous":
        raise AssertionError("PPO did not record a continuous head for the pendulum task.")
    if registered_state["action_spec"].dim != 1:
        raise AssertionError("PPO did not build a 1-dim Gaussian head for Pendulum-v1.")
    if registered_state["summary"]["updates"] <= 0:
        raise AssertionError("PPO ran no updates on the registered continuous task.")

    print("ppo smoke ok")
    print(f"run_dir={state['run_dir']}")
    print(f"pendulum_run_dir={registered_state['run_dir']}")
    print(f"mean_return={float(final_eval['mean_return']):.6f}")
    print(f"continuous_entropy={float(update_metrics['entropy']):.6f}")


def run_td3_smoke() -> None:
    from pathlib import Path as _Path

    import numpy as np
    import torch.optim as optim

    from datasets.rl_tasks import get_rl_task_spec
    from model.rl_workflow import _make_env
    from model.td3_workflow import (
        ALGORITHM_NAME,
        ContinuousReplayBuffer,
        TD3Agent,
        _soft_update,
        continuous_rl_task_names,
        describe_action_space,
        evaluate_td3,
        load_td3_checkpoint,
        make_td3_config,
        require_continuous_task,
        select_td3_action,
        td3_update,
        train_td3,
    )

    device = torch.device("cpu")

    # -- registry: env_kwargs threading and the continuous task entries --------
    for discrete_task in ("cartpole", "acrobot", "lunarlander", "minatar_breakout"):
        if get_rl_task_spec(discrete_task).env_kwargs_dict() != {}:
            raise AssertionError(
                f"Task '{discrete_task}' predates env_kwargs and must keep an empty mapping "
                "so its gym.make call is unchanged."
            )
    lunar_continuous_spec = get_rl_task_spec("lunarlander_continuous")
    if lunar_continuous_spec.env_kwargs_dict() != {"continuous": True}:
        raise AssertionError("lunarlander_continuous must request the continuous LunarLander variant.")
    if lunar_continuous_spec.to_dict()["env_kwargs"] != {"continuous": True}:
        raise AssertionError("Task env_kwargs must serialize as a mapping for config.json.")
    pendulum_spec = get_rl_task_spec("pendulum")
    if (pendulum_spec.success_threshold, pendulum_spec.target_mean_return) != (-200.0, -140.0):
        raise AssertionError("Pendulum must keep its documented non-canonical success markers.")
    if not any("NON-CANONICAL" in note for note in pendulum_spec.literature_notes):
        raise AssertionError("Pendulum notes must flag its thresholds as non-canonical.")
    if sorted(continuous_rl_task_names()) != ["lunarlander_continuous", "pendulum"]:
        raise AssertionError("The continuous task list did not match the registry.")
    for task_name, expected_action_dim in (("pendulum", 1), ("lunarlander_continuous", 2)):
        probe_env = _make_env(get_rl_task_spec(task_name), seed=0)
        try:
            probe_action_spec = describe_action_space(probe_env.action_space)
        finally:
            probe_env.close()
        if probe_action_spec.kind != "continuous" or probe_action_spec.dim != expected_action_dim:
            raise AssertionError(f"Task '{task_name}' did not build a {expected_action_dim}-dim Box action space.")

    # -- configuration protocol -----------------------------------------------
    cfg = make_td3_config("smoke", seed=0)
    if cfg.early_stopping_patience != 30 or cfg.early_stopping_min_steps != 25_000:
        raise AssertionError("TD3 early stopping must share the repo protocol (patience 30, min steps 25k).")
    if cfg.early_stopping or make_td3_config("gold").early_stopping:
        raise AssertionError("smoke and gold profiles must disable early stopping.")
    if not make_td3_config("fast").early_stopping or not make_td3_config("debug").early_stopping:
        raise AssertionError("fast and debug profiles must enable early stopping.")
    fast_cfg = make_td3_config("fast")
    if (fast_cfg.tau, fast_cfg.gamma, fast_cfg.policy_frequency) != (0.005, 0.99, 2):
        raise AssertionError("TD3 defaults must keep tau 0.005, gamma 0.99, policy frequency 2.")
    if (fast_cfg.policy_noise, fast_cfg.noise_clip, fast_cfg.exploration_noise) != (0.2, 0.5, 0.1):
        raise AssertionError("TD3 defaults must keep smoothing noise 0.2 clipped 0.5 and exploration noise 0.1.")
    if fast_cfg.actor_learning_rate != fast_cfg.learning_rate:
        raise AssertionError("The actor must follow the critic learning rate unless overridden.")
    if make_td3_config("fast", policy_learning_rate=1e-4).actor_learning_rate != 1e-4:
        raise AssertionError("policy_learning_rate must override the actor learning rate.")
    for invalid_overrides in (
        {"tau": 0.0},
        {"tau": 1.5},
        {"policy_frequency": 0},
        {"train_frequency": 0},
        {"batch_size": 0},
        {"buffer_size": 0},
        {"learning_starts": -1},
        {"policy_noise": -0.1},
        {"noise_clip": -0.1},
        {"exploration_noise": -0.1},
        {"policy_learning_rate": 0.0},
        {"max_grad_norm": 0.0},
    ):
        try:
            make_td3_config("smoke", **invalid_overrides)
        except ValueError:
            pass
        else:
            raise AssertionError(f"TD3 should reject invalid config {invalid_overrides}.")

    # -- continuous-only scope -------------------------------------------------
    require_continuous_task(pendulum_spec)
    for discrete_task in ("cartpole", "minatar_breakout"):
        try:
            require_continuous_task(get_rl_task_spec(discrete_task))
        except ValueError as exc:
            if "continuous-control" not in str(exc):
                raise AssertionError("TD3 must explain that it is a continuous-control algorithm.") from exc
        else:
            raise AssertionError(f"TD3 must refuse the discrete task '{discrete_task}'.")
    try:
        train_td3("cartpole", make_td3_config("smoke"), progress=False)
    except ValueError:
        pass
    else:
        raise AssertionError("train_td3 must refuse a discrete-action task before training.")

    # -- replay buffer ---------------------------------------------------------
    buffer = ContinuousReplayBuffer.create(2, 3, 2)
    buffer.add(np.zeros(3, dtype=np.float32), np.array([0.5, -0.5], dtype=np.float32), 1.0,
               np.ones(3, dtype=np.float32), False)
    buffer.add(np.ones(3, dtype=np.float32), np.array([-1.0, 1.0], dtype=np.float32), 2.0,
               np.zeros(3, dtype=np.float32), True)
    buffer.add(np.full(3, 2.0, dtype=np.float32), np.array([0.25, 0.25], dtype=np.float32), 3.0,
               np.zeros(3, dtype=np.float32), False)
    if buffer.size != 2 or buffer.pos != 1:
        raise AssertionError("The continuous replay buffer did not wrap like the DQN ring buffer.")
    if buffer.actions.shape != (2, 2) or buffer.actions.dtype != np.float32:
        raise AssertionError("The continuous replay buffer must store float32 action vectors.")
    sampled = buffer.sample(4, device)
    if sampled["actions"].shape != (4, 2) or sampled["actions"].dtype != torch.float32:
        raise AssertionError("Sampled continuous actions have the wrong shape or dtype.")

    # -- networks, twin critics, target smoothing, delayed updates -------------
    pendulum_env = _make_env(pendulum_spec)
    try:
        action_spec = describe_action_space(pendulum_env.action_space)
    finally:
        pendulum_env.close()
    agent = TD3Agent(3, action_spec, hidden_sizes=(16, 16)).to(device)
    target_agent = TD3Agent(3, action_spec, hidden_sizes=(16, 16)).to(device)
    target_agent.load_state_dict(agent.state_dict())
    if torch.allclose(agent.qf1.network[0].weight, agent.qf2.network[0].weight):
        raise AssertionError("The twin critics must be independently initialized.")
    bounded_probe = agent.act(torch.randn(8, 3))
    if bool((bounded_probe < agent.actor.action_low).any() or (bounded_probe > agent.actor.action_high).any()):
        raise AssertionError("The tanh actor must emit actions inside the Box bounds.")
    noisy_action = select_td3_action(agent, np.zeros(3, dtype=np.float32), exploration_noise=50.0, device=device)
    if not (float(action_spec.low[0]) <= float(noisy_action[0]) <= float(action_spec.high[0])):
        raise AssertionError("Exploration noise must be clipped back into the Box bounds.")

    polyak_probe = TD3Agent(3, action_spec, hidden_sizes=(16, 16)).to(device)
    with torch.no_grad():
        for parameter in polyak_probe.parameters():
            parameter.zero_()
    _soft_update(polyak_probe, agent, 0.5)
    if not torch.allclose(polyak_probe.qf1.network[0].weight, agent.qf1.network[0].weight * 0.5, atol=1e-6):
        raise AssertionError("Target networks must Polyak-average with the configured tau.")

    batch = {
        "observations": torch.randn(8, 3),
        "actions": torch.rand(8, 1) * 4.0 - 2.0,
        "rewards": torch.randn(8),
        "next_observations": torch.randn(8, 3),
        "dones": torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]),
    }
    # noise_clip=0 removes target policy smoothing, so the twin-min bootstrap
    # target is deterministic and can be checked exactly.
    deterministic_cfg = make_td3_config("smoke", noise_clip=0.0, gamma=0.9)
    with torch.no_grad():
        target_actions = target_agent.act(batch["next_observations"])
        target_q1, target_q2 = target_agent.q_values(batch["next_observations"], target_actions)
        expected_td_target = batch["rewards"] + 0.9 * (1.0 - batch["dones"]) * torch.min(target_q1, target_q2)
    if not torch.allclose(expected_td_target[batch["dones"] > 0], batch["rewards"][batch["dones"] > 0], atol=1e-6):
        raise AssertionError("Terminated transitions must not bootstrap.")
    critic_optimizer = optim.Adam(agent.critic_parameters(), lr=1e-3)
    actor_optimizer = optim.Adam(agent.actor.parameters(), lr=1e-3)
    target_actor_before = target_agent.actor.network[0].weight.detach().clone()
    qf1_before = agent.qf1.network[0].weight.detach().clone()
    qf2_before = agent.qf2.network[0].weight.detach().clone()
    actor_before = agent.actor.network[0].weight.detach().clone()
    delayed_metrics = td3_update(
        agent, target_agent, critic_optimizer, actor_optimizer, deterministic_cfg, batch, update_policy=False
    )
    if abs(delayed_metrics["mean_td_target"] - float(expected_td_target.mean())) > 1e-5:
        raise AssertionError("TD3 target values did not match the twin-min bootstrap with smoothing disabled.")
    if delayed_metrics["actor_loss"] is not None or delayed_metrics["policy_updated"]:
        raise AssertionError("A delayed step must not update the policy.")
    if not torch.equal(agent.actor.network[0].weight.detach(), actor_before):
        raise AssertionError("The actor must not move on a critic-only step.")
    if not torch.equal(target_agent.actor.network[0].weight.detach(), target_actor_before):
        raise AssertionError("Target networks must only sync on policy updates.")
    if torch.equal(agent.qf1.network[0].weight.detach(), qf1_before) or torch.equal(
        agent.qf2.network[0].weight.detach(), qf2_before
    ):
        raise AssertionError("Both critics must take a gradient step every training step.")
    policy_metrics = td3_update(
        agent, target_agent, critic_optimizer, actor_optimizer, deterministic_cfg, batch, update_policy=True
    )
    if policy_metrics["actor_loss"] is None or not policy_metrics["policy_updated"]:
        raise AssertionError("A policy step must report its actor loss.")
    if torch.equal(agent.actor.network[0].weight.detach(), actor_before):
        raise AssertionError("The actor must move on a policy step.")
    if torch.equal(target_agent.actor.network[0].weight.detach(), target_actor_before):
        raise AssertionError("Target networks must sync on policy steps.")

    # -- end-to-end smoke training --------------------------------------------
    state = train_td3("pendulum", cfg, progress=False)
    final_eval = state["final_eval"]
    if final_eval["episodes"] != cfg.eval_episodes:
        raise AssertionError("TD3 smoke evaluation did not run the configured number of episodes.")
    if not state["checkpoint_path"].exists():
        raise AssertionError("TD3 smoke did not write a checkpoint.")
    summary = state["summary"]
    if summary["algorithm"] != ALGORITHM_NAME:
        raise AssertionError("TD3 summary did not record its algorithm name.")
    if summary["action_kind"] != "continuous" or summary["action_dim"] != 1:
        raise AssertionError("TD3 summary did not record the continuous action space.")
    if summary["policy_updates"] != (summary["critic_updates"] + 1) // 2:
        raise AssertionError("TD3 did not delay policy updates by policy_frequency=2.")
    run_dir = _Path(state["run_dir"])
    for artifact in (
        "config.json",
        "summary.json",
        "manifest.json",
        "eval_metrics.csv",
        "training_metrics.csv",
        "loss_metrics.csv",
        "final_eval_episodes.csv",
        "last_eval_episodes.csv",
    ):
        if not (run_dir / artifact).exists():
            raise AssertionError(f"TD3 run directory is missing {artifact}.")

    reloaded = load_td3_checkpoint(state["checkpoint_path"])
    if reloaded["config"].profile != "smoke" or reloaded["action_spec"].kind != "continuous":
        raise AssertionError("TD3 checkpoint reload did not preserve the config and action spec.")
    if "target_model" not in reloaded:
        raise AssertionError("TD3 checkpoint reload did not return the target networks.")
    rng_before = torch.random.get_rng_state().clone()
    reload_metrics = evaluate_td3("pendulum", reloaded["model"], episodes=1, seed=cfg.eval_seed,
                                  device=reloaded["device"])
    if reload_metrics["episodes"] != 1:
        raise AssertionError("TD3 checkpoint reload evaluation did not run.")
    if not torch.equal(torch.random.get_rng_state(), rng_before):
        raise AssertionError("Deterministic TD3 evaluation must not consume the training RNG stream.")

    early_cfg = make_td3_config(
        "smoke",
        seed=1,
        total_timesteps=64,
        eval_frequency=16,
        eval_episodes=1,
        early_stopping=True,
        early_stopping_target_score=-1e6,
    )
    early_state = train_td3("pendulum", early_cfg, progress=False)
    if early_state["summary"]["actual_timesteps"] != 16:
        raise AssertionError("TD3 did not stop at its first target-reaching evaluation.")
    if early_state["summary"]["early_stopping"]["stopping_reason"] != "target_score":
        raise AssertionError("TD3 summary did not record its early-stopping reason.")

    lunar_cfg = make_td3_config("smoke", seed=0, total_timesteps=64, eval_frequency=64, eval_episodes=1)
    lunar_state = train_td3("lunarlander_continuous", lunar_cfg, progress=False)
    if lunar_state["summary"]["action_dim"] != 2:
        raise AssertionError("TD3 did not train the 2-dim continuous LunarLander variant.")
    if lunar_state["summary"]["env_id"] != "LunarLander-v3":
        raise AssertionError("lunarlander_continuous must reuse the LunarLander-v3 env id with env_kwargs.")

    print("td3 smoke ok")
    print(f"run_dir={state['run_dir']}")
    print(f"lunarlander_continuous_run_dir={lunar_state['run_dir']}")
    print(f"mean_return={float(final_eval['mean_return']):.6f}")


def run_nnknn_rl_smoke() -> None:
    from model.nnknn_rl_workflow import (
        ALGORITHM_NAME,
        MLPPolicyNetwork,
        NNKNNPolicyNetwork,
        NNKNNValueNetwork,
        _align_nnknn_target_case_store,
        _actor_exploration_epsilon,
        _build_nnknn_target_value_model,
        _build_nnknn_rl_optimizer,
        _epsilon_mixed_policy_probs,
        _exploration_epsilon,
        _reset_optimizer_case_state,
        _sync_nnknn_target_value_model,
        compute_gae,
        evaluate_critic_holdout,
        evaluate_nnknn_rl,
        load_nnknn_rl_checkpoint,
        make_nnknn_rl_config,
        train_nnknn_rl,
    )

    advantages, value_targets = compute_gae(
        rewards=[1.0, 1.0, 1.0],
        values=[0.5, 0.5, 0.5],
        next_values=[0.5, 0.5, 0.0],
        terminated=[False, False, True],
        gamma=0.9,
        gae_lambda=0.8,
    )
    if not torch.allclose(advantages, torch.tensor([1.8932, 1.31, 0.5]), atol=1e-4):
        raise AssertionError("NN-kNN-RL GAE helper returned unexpected advantages.")
    if not torch.allclose(value_targets, torch.tensor([2.3932, 1.81, 1.0]), atol=1e-4):
        raise AssertionError("NN-kNN-RL GAE helper returned unexpected value targets.")

    boundary_advantages, _boundary_targets = compute_gae(
        rewards=[1.0, 1.0],
        values=[0.0, 0.0],
        next_values=[0.0, 0.0],
        terminated=[False, True],
        episode_boundaries=[True, True],
        gamma=0.99,
        gae_lambda=0.95,
    )
    if not torch.allclose(boundary_advantages, torch.tensor([1.0, 1.0]), atol=1e-5):
        raise AssertionError("NN-kNN-RL GAE leaked advantage across an episode boundary.")

    wrapper = NNKNNPolicyNetwork(4, 2, case_capacity=6, top_k=2, min_cases_per_action=1)
    wrapper.configure_case_maintenance(prune_quantile=1.0, prune_bias_threshold=None)
    wrapper.add_cases(
        torch.zeros(6, 4),
        torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
    )
    probs = wrapper.policy_probs(torch.zeros(2, 4))
    if probs.shape != (2, 2) or not torch.allclose(
        probs.sum(dim=1), torch.ones(2, device=probs.device), atol=1e-5
    ):
        raise AssertionError("NN-kNN-RL policy wrapper did not return normalized action probabilities.")
    with torch.no_grad():
        wrapper.nnknn_model.biases[:6].copy_(torch.tensor([-5.0, -4.0, -3.0, -2.0, -1.0, 0.0]))
    wrapper.prune_cases()
    if torch.any(wrapper.action_counts() < 1):
        raise AssertionError("NN-kNN-RL pruning removed all cases for an action.")

    actor_probe = NNKNNPolicyNetwork(2, 2, case_capacity=4, top_k=2, min_cases_per_action=1)
    actor_probe_device = next(actor_probe.parameters()).device
    actor_probe.add_cases(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32, device=actor_probe_device),
        torch.tensor([0, 1], dtype=torch.long, device=actor_probe_device),
    )
    actor_probe_optim = torch.optim.Adam(actor_probe.parameters(), lr=0.05)
    actor_bias_before = actor_probe.nnknn_model.biases[: actor_probe.case_entries].detach().clone()
    actor_probe_loss = -torch.log(
        actor_probe.policy_probs(torch.tensor([[0.2, 0.2]], dtype=torch.float32, device=actor_probe_device))[0, 1].clamp_min(1e-8)
    )
    actor_probe_optim.zero_grad()
    actor_probe_loss.backward()
    actor_probe_optim.step()
    actor_bias_after = actor_probe.nnknn_model.biases[: actor_probe.case_entries].detach().clone()
    if torch.allclose(actor_bias_before, actor_bias_after):
        raise AssertionError("NN-kNN actor parameters did not update under policy loss with an MLP critic path.")
    readiness_probe = NNKNNPolicyNetwork(2, 2, case_capacity=6, top_k=2, min_cases_per_action=2)
    readiness_device = next(readiness_probe.parameters()).device
    readiness_probe.add_cases(
        torch.tensor([[0.0, 0.0], [0.1, 0.1], [1.0, 1.0]], dtype=torch.float32, device=readiness_device),
        torch.tensor([0, 0, 1], dtype=torch.long, device=readiness_device),
    )
    if readiness_probe.is_policy_ready(min_case_entries=3):
        raise AssertionError("NN-kNN policy should wait for the configured per-action case floor.")
    readiness_probe.add_cases(
        torch.tensor([[1.1, 1.1]], dtype=torch.float32, device=readiness_device),
        torch.tensor([1], dtype=torch.long, device=readiness_device),
    )
    if not readiness_probe.is_policy_ready(min_case_entries=4):
        raise AssertionError("NN-kNN policy did not become ready after each action reached its case floor.")

    critic_probe = NNKNNValueNetwork(2, case_capacity=4, top_k=2)
    critic_probe_device = next(critic_probe.parameters()).device
    critic_probe.add_cases(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32, device=critic_probe_device),
        torch.tensor([0.0, 1.0], dtype=torch.float32, device=critic_probe_device),
    )
    critic_probe_optim = torch.optim.Adam(critic_probe.parameters(), lr=0.05)
    critic_bias_before = critic_probe.nnknn_model.biases[: critic_probe.case_entries].detach().clone()
    critic_probe_loss = torch.nn.functional.mse_loss(
        critic_probe(torch.tensor([[0.1, 0.1], [0.9, 0.9]], dtype=torch.float32, device=critic_probe_device)),
        torch.tensor([1.0, 0.0], dtype=torch.float32, device=critic_probe_device),
    )
    critic_probe_optim.zero_grad()
    critic_probe_loss.backward()
    critic_probe_optim.step()
    critic_bias_after = critic_probe.nnknn_model.biases[: critic_probe.case_entries].detach().clone()
    if torch.allclose(critic_bias_before, critic_bias_after):
        raise AssertionError("NN-kNN critic parameters did not update under value loss with an MLP actor path.")

    mutable_critic = NNKNNValueNetwork(
        2,
        case_capacity=4,
        top_k=1,
        mutable_value_labels=True,
        value_label_update_alpha=0.5,
        value_label_activation_threshold=0.0,
    )
    mutable_critic_device = next(mutable_critic.parameters()).device
    mutable_critic.add_cases(
        torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=mutable_critic_device),
        torch.tensor([0.0], dtype=torch.float32, device=mutable_critic_device),
    )
    mutable_stats = mutable_critic.add_cases(
        torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=mutable_critic_device),
        torch.tensor([10.0], dtype=torch.float32, device=mutable_critic_device),
    )
    if mutable_critic.case_entries != 1:
        raise AssertionError("Mutable NN-kNN critic should relabel an identical active case instead of appending it.")
    if mutable_stats["label_updates"] != 1 or mutable_stats["label_update_samples"] != 1:
        raise AssertionError("Mutable NN-kNN critic did not report the expected value-label relabel.")
    if not torch.allclose(
        mutable_critic.nnknn_model.labels[0, 0],
        torch.tensor(5.0, dtype=torch.float32, device=mutable_critic_device),
        atol=1e-5,
    ):
        raise AssertionError("Mutable NN-kNN critic did not smooth the existing value label toward the new target.")

    trainable_critic = NNKNNValueNetwork(2, case_capacity=4, top_k=1, trainable_value_labels=True)
    trainable_critic_device = next(trainable_critic.parameters()).device
    if not isinstance(trainable_critic.nnknn_model.labels, torch.nn.Parameter):
        raise AssertionError("Trainable NN-kNN critic labels should be registered as a Parameter.")
    trainable_critic.add_cases(
        torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32, device=trainable_critic_device),
        torch.tensor([0.0, 1.0], dtype=torch.float32, device=trainable_critic_device),
    )
    trainable_critic_optim = torch.optim.Adam(trainable_critic.parameters(), lr=0.1)
    trainable_labels_before = trainable_critic.nnknn_model.labels[: trainable_critic.case_entries].detach().clone()
    trainable_loss = torch.nn.functional.mse_loss(
        trainable_critic(
            torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32, device=trainable_critic_device)
        ),
        torch.tensor([10.0, -10.0], dtype=torch.float32, device=trainable_critic_device),
    )
    trainable_critic_optim.zero_grad()
    trainable_loss.backward()
    trainable_critic_optim.step()
    trainable_labels_after = trainable_critic.nnknn_model.labels[: trainable_critic.case_entries].detach().clone()
    if torch.allclose(trainable_labels_before, trainable_labels_after):
        raise AssertionError("Trainable NN-kNN critic labels did not update under value loss.")
    grouped_critic_optim = _build_nnknn_rl_optimizer(
        trainable_critic,
        base_lr=0.03,
        case_lr=0.007,
    )
    label_group_lr = None
    bias_group_lr = None
    for group in grouped_critic_optim.param_groups:
        if any(param is trainable_critic.nnknn_model.labels for param in group["params"]):
            label_group_lr = float(group["lr"])
        if any(param is trainable_critic.nnknn_model.biases for param in group["params"]):
            bias_group_lr = float(group["lr"])
    if label_group_lr != 0.007 or bias_group_lr != 0.007:
        raise AssertionError("Trainable critic labels should share the case-level learning rate with case biases.")

    compaction_probe = NNKNNPolicyNetwork(2, 2, case_capacity=4, top_k=2, min_cases_per_action=1)
    compaction_device = next(compaction_probe.parameters()).device
    compaction_probe.add_cases(
        torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], device=compaction_device),
        torch.tensor([0, 1, 0], device=compaction_device),
    )
    compaction_optimizer = torch.optim.Adam(compaction_probe.parameters(), lr=0.01)
    biases = compaction_probe.nnknn_model.biases
    compaction_optimizer.zero_grad()
    (biases[:3] * torch.tensor([1.0, 2.0, 3.0], device=compaction_device)).sum().backward()
    compaction_optimizer.step()
    if biases not in compaction_optimizer.state:
        raise AssertionError("Compaction probe did not initialize Adam case state.")
    compaction_probe.nnknn_model.compact_cases([1, 2])
    _reset_optimizer_case_state(compaction_optimizer, compaction_probe)
    if biases in compaction_optimizer.state:
        raise AssertionError("Case compaction should clear stale Adam state for per-case parameters.")

    target_store_probe = NNKNNValueNetwork(2, case_capacity=5, top_k=2).to("cpu")
    target_store_probe.add_cases(
        torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        torch.tensor([1.0, 2.0, 3.0]),
    )
    target_store_snapshot = _build_nnknn_target_value_model(target_store_probe, device=torch.device("cpu"))
    if target_store_probe.nnknn_model.cases.data_ptr() != target_store_snapshot.nnknn_model.cases.data_ptr():
        raise AssertionError("Target critic probe did not share the online structural case tensor.")
    with torch.no_grad():
        target_store_probe.nnknn_model.labels[:3].add_(10.0)
        target_store_probe.nnknn_model.biases[:3].copy_(torch.tensor([-3.0, -2.0, -1.0]))
    _sync_nnknn_target_value_model(target_store_probe, target_store_snapshot, mode="ema", ema_tau=0.5)
    if not torch.allclose(
        target_store_snapshot.nnknn_model.labels[:3, 0],
        torch.tensor([6.0, 7.0, 8.0]),
    ):
        raise AssertionError("Target critic value labels did not EMA-update independently of the shared case tensor.")
    target_values_by_id = {
        int(case_id): float(value)
        for case_id, value in zip(
            target_store_snapshot.active_case_ids(),
            target_store_snapshot.nnknn_model.labels[:3, 0],
        )
    }
    target_store_probe.configure_case_maintenance(prune_quantile=0.5, prune_bias_threshold=None)
    target_store_probe.prune_cases()
    _align_nnknn_target_case_store(target_store_probe, target_store_snapshot)
    expected_target_values = torch.tensor(
        [target_values_by_id[int(case_id)] for case_id in target_store_probe.active_case_ids()]
    )
    if not torch.allclose(
        target_store_snapshot.nnknn_model.labels[: target_store_snapshot.case_entries, 0],
        expected_target_values,
    ):
        raise AssertionError("Target critic per-case parameters did not follow stable IDs through compaction.")

    cfg = make_nnknn_rl_config("smoke", seed=0)
    if cfg.early_stopping_patience != 30:
        raise AssertionError("NN-kNN-RL early-stopping patience should default to 30 evaluations.")
    for removed_field in ("advantage_method", "value_function", "bootstrap_n_steps", "vtrace"):
        if hasattr(cfg, removed_field):
            raise AssertionError(f"NN-kNN-RL config still exposes removed field {removed_field}.")
    if cfg.gae_lambda != 0.95:
        raise AssertionError("NN-kNN-RL should default to GAE lambda 0.95.")
    if cfg.reward_shaping is not None:
        raise AssertionError("NN-kNN-RL should default to raw environment rewards.")
    if cfg.critic_type != "mlp":
        raise AssertionError("NN-kNN-RL should default to the MLP value critic.")
    if cfg.actor_type != "nnknn":
        raise AssertionError("NN-kNN-RL should default to the NN-kNN actor.")
    if cfg.exploration_initial_epsilon != 1.0 or cfg.exploration_final_epsilon != 0.05:
        raise AssertionError("NN-kNN-RL should default to epsilon-mixed stochastic exploration.")
    if abs(_exploration_epsilon(cfg, 0) - cfg.exploration_initial_epsilon) > 1e-12:
        raise AssertionError("NN-kNN-RL exploration schedule should start at initial epsilon.")
    if abs(_exploration_epsilon(cfg, cfg.total_timesteps) - cfg.exploration_final_epsilon) > 1e-12:
        raise AssertionError("NN-kNN-RL exploration schedule should end at final epsilon.")
    if cfg.critic_target_value_mode != "ema" or cfg.critic_target_sync_interval != 4:
        raise AssertionError("NN-kNN-RL should default NN-kNN critics to a lagged EMA target.")
    if cfg.critic_mutable_value_labels or cfg.critic_trainable_value_labels:
        raise AssertionError("NN-kNN-RL should default to fixed critic value labels.")
    hybrid_cfg = make_nnknn_rl_config(
        "smoke",
        critic_type="nnknn",
        critic_mutable_value_labels="true",
        critic_trainable_value_labels=True,
        critic_value_label_activation_threshold=None,
        case_learning_rate=7e-4,
    )
    if not hybrid_cfg.critic_mutable_value_labels or not hybrid_cfg.critic_trainable_value_labels:
        raise AssertionError("NN-kNN-RL config did not enable the requested hybrid value-label mode.")
    if hybrid_cfg.critic_value_label_activation_threshold is not None:
        raise AssertionError("NN-kNN-RL config should allow disabling activation-based mutable label matching.")
    if hybrid_cfg.case_learning_rate != 7e-4:
        raise AssertionError("NN-kNN-RL config did not preserve the case-level learning rate.")
    for invalid_overrides in (
        {"critic_value_label_update_alpha": 0.0},
        {"critic_value_label_activation_threshold": float("inf")},
        {"critic_target_sync_interval": 0},
        {"critic_target_ema_tau": 0.0},
        {"case_learning_rate": 0.0},
        {"exploration_initial_epsilon": -0.1},
        {"exploration_final_epsilon": 1.1},
        {"exploration_initial_epsilon": 0.1, "exploration_final_epsilon": 0.2},
        {"exploration_fraction": 1.1},
        {"min_case_entries": 0},
        {"min_cases_per_action": 0},
        {"critic_holdout_episode_frequency": -1},
        {"critic_holdout_episodes": 0},
    ):
        try:
            make_nnknn_rl_config("smoke", **invalid_overrides)
        except ValueError:
            pass
        else:
            raise AssertionError(f"NN-kNN-RL should reject invalid value-label config {invalid_overrides}.")
    try:
        make_nnknn_rl_config("smoke", reward_shaping="invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("NN-kNN-RL should reject unknown reward_shaping values.")

    mlp_wrapper = MLPPolicyNetwork(4, 2, hidden_sizes=(8,))
    mlp_probs = mlp_wrapper.policy_probs(torch.zeros(3, 4))
    if mlp_probs.shape != (3, 2) or not torch.allclose(mlp_probs.sum(dim=1), torch.ones(3), atol=1e-5):
        raise AssertionError("MLP actor did not return normalized action probabilities.")
    mixed_probs = _epsilon_mixed_policy_probs(torch.tensor([[0.9, 0.1]], dtype=torch.float32), 0.2)
    if not torch.allclose(mixed_probs, torch.tensor([[0.82, 0.18]], dtype=torch.float32), atol=1e-6):
        raise AssertionError("NN-kNN-RL epsilon-mixed behavior probabilities are incorrect.")
    if _actor_exploration_epsilon(mlp_wrapper, cfg, 0) != 0.0:
        raise AssertionError("The standard stochastic MLP actor must not receive epsilon mixing.")

    state = train_nnknn_rl("cartpole", cfg, progress=False)
    final_eval = state["final_eval"]
    if final_eval["episodes"] != cfg.eval_episodes:
        raise AssertionError("NN-kNN-RL smoke evaluation did not run the configured number of episodes.")
    if not state["checkpoint_path"].exists():
        raise AssertionError("NN-kNN-RL smoke did not write a checkpoint.")
    loaded = load_nnknn_rl_checkpoint(state["checkpoint_path"])
    if loaded["checkpoint"].get("algorithm") != ALGORITHM_NAME:
        raise AssertionError("NN-kNN-RL checkpoint did not preserve the actor-critic algorithm marker.")
    if loaded["config"].actor_type != "nnknn":
        raise AssertionError("NN-kNN actor checkpoint reload did not preserve actor_type.")
    if "actor_state" not in loaded["checkpoint"] or "critic_state_dict" not in loaded["checkpoint"]:
        raise AssertionError("NN-kNN-RL checkpoint did not preserve actor and critic state.")
    if loaded["model"].case_entries != state["model"].case_entries:
        raise AssertionError("NN-kNN-RL checkpoint did not preserve active case count.")
    exploration_summary = state["summary"].get("exploration", {})
    if exploration_summary.get("initial_epsilon") != cfg.exploration_initial_epsilon:
        raise AssertionError("NN-kNN actor smoke summary did not record exploration configuration.")
    if not state["training_metrics"] or "behavior_epsilon" not in state["training_metrics"][0]:
        raise AssertionError("NN-kNN actor training metrics did not record behavior epsilon.")
    final_loss_row = state["loss_metrics"][-1]
    for renamed_field in (
        "critic_optimization_mse",
        "critic_train_mse",
        "critic_train_explained_variance",
        "critic_train_diagnostics_computed",
    ):
        if renamed_field not in final_loss_row:
            raise AssertionError(f"NN-kNN-RL loss metrics are missing {renamed_field}.")
    for outdated_field in ("critic_loss", "explained_variance", "critic_diagnostics_post_update"):
        if outdated_field in final_loss_row:
            raise AssertionError(f"NN-kNN-RL loss metrics still expose ambiguous field {outdated_field}.")
    if state["summary"]["partial_rollout_samples"] > 0 and (
        int(final_loss_row["global_step"]) != cfg.total_timesteps
        or int(final_loss_row["partial_rollout_samples"]) != state["summary"]["partial_rollout_samples"]
    ):
        raise AssertionError("NN-kNN actor final partial rollout cases were not included in policy updates.")
    partial_cfg = make_nnknn_rl_config(
        "smoke",
        seed=42,
        total_timesteps=1,
        eval_episodes=1,
        policy_update_episodes=10,
    )
    partial_state = train_nnknn_rl("cartpole", partial_cfg, progress=False)
    if partial_state["summary"]["partial_rollout_samples"] != 1:
        raise AssertionError("NN-kNN actor one-step run should train exactly one final partial rollout sample.")
    if int(partial_state["loss_metrics"][-1]["global_step"]) != partial_cfg.total_timesteps:
        raise AssertionError("NN-kNN actor one-step partial rollout was not updated at the fixed step budget.")
    early_cfg = make_nnknn_rl_config(
        "smoke",
        seed=43,
        actor_type="mlp",
        critic_type="mlp",
        total_timesteps=64,
        eval_frequency=5,
        eval_episode_frequency=0,
        eval_episodes=1,
        policy_update_episodes=10,
        early_stopping=True,
        early_stopping_target_score=0.0,
    )
    early_state = train_nnknn_rl("cartpole", early_cfg, progress=False)
    if early_state["summary"]["actual_timesteps"] != 5:
        raise AssertionError("NN-kNN-RL did not stop at its first target-reaching evaluation.")
    if early_state["summary"]["partial_rollout_samples"] != 5:
        raise AssertionError("NN-kNN-RL did not train its rollout before early-stop checkpointing.")
    if early_state["summary"]["early_stopping"]["stopping_reason"] != "target_score":
        raise AssertionError("NN-kNN-RL summary did not record its early-stopping reason.")
    if "value_model" not in loaded:
        raise AssertionError("NN-kNN-RL checkpoint reload did not return the value critic.")
    legacy_checkpoint = dict(torch.load(state["checkpoint_path"], map_location="cpu", weights_only=False))
    legacy_checkpoint["config"] = dict(legacy_checkpoint["config"])
    legacy_checkpoint["config"].pop("actor_type", None)
    legacy_checkpoint.pop("actor_type", None)
    legacy_path = state["run_dir"] / "legacy_no_actor_type_checkpoint.pt"
    torch.save(legacy_checkpoint, legacy_path)
    legacy_loaded = load_nnknn_rl_checkpoint(legacy_path)
    if legacy_loaded["config"].actor_type != "nnknn":
        raise AssertionError("Legacy checkpoint without actor_type should reload as NN-kNN actor.")

    nnknn_cfg = make_nnknn_rl_config(
        "smoke",
        seed=1,
        critic_type="nnknn",
        critic_target_sync_interval=1,
    )
    nnknn_state = train_nnknn_rl("cartpole", nnknn_cfg, progress=False)
    if not nnknn_state["checkpoint_path"].exists():
        raise AssertionError("NN-kNN critic smoke did not write a checkpoint.")
    if not isinstance(nnknn_state["model"], NNKNNPolicyNetwork):
        raise AssertionError("NN-kNN actor + NN-kNN critic should retain a dedicated policy store.")
    if not isinstance(nnknn_state["value_model"], NNKNNValueNetwork):
        raise AssertionError("NN-kNN actor + NN-kNN critic should retain a dedicated value store.")
    if nnknn_state["model"] is nnknn_state["value_model"]:
        raise AssertionError("NN-kNN actor and critic must not share their case base.")
    if nnknn_state["model"].nnknn_model.cases.data_ptr() == nnknn_state["value_model"].nnknn_model.cases.data_ptr():
        raise AssertionError("NN-kNN actor and critic case buffers should be separate.")
    if nnknn_state["model"].nnknn_model.glocal_weightor is not nnknn_state["value_model"].nnknn_model.glocal_weightor:
        raise AssertionError("NN-kNN actor and critic should share their global feature-distance representation.")
    nnknn_loaded = load_nnknn_rl_checkpoint(nnknn_state["checkpoint_path"])
    if nnknn_loaded["config"].critic_type != "nnknn":
        raise AssertionError("NN-kNN critic checkpoint reload did not preserve critic_type.")
    if "value_model" not in nnknn_loaded:
        raise AssertionError("NN-kNN critic checkpoint reload did not return the value critic.")
    if getattr(nnknn_loaded["value_model"], "case_entries", 0) <= 0:
        raise AssertionError("NN-kNN critic checkpoint did not preserve critic value cases.")
    if nnknn_state["summary"]["partial_rollout_samples"] <= 0:
        raise AssertionError("Separate-memory NN-kNN smoke should train a final partial rollout.")
    if not nnknn_state["summary"]["architecture"]["separate_case_bases"]:
        raise AssertionError("NN-kNN summary should record separate actor/critic case bases.")
    if not nnknn_state["summary"]["architecture"]["critic_target_shared_case_store"]:
        raise AssertionError("NN-kNN summary should record the online/target critic structural case store.")
    if nnknn_state["summary"]["critic_target_value_mode"] != "ema":
        raise AssertionError("NN-kNN critic summary should record EMA target mode.")
    if nnknn_state["summary"]["critic_target_value_syncs"] <= 1:
        raise AssertionError("NN-kNN critic should synchronize its lagged target after training updates.")
    if not isinstance(nnknn_state["target_model"], NNKNNValueNetwork):
        raise AssertionError("NN-kNN critic training should expose its lagged target critic.")
    online_critic = nnknn_state["value_model"]
    target_critic = nnknn_state["target_model"]
    if online_critic.nnknn_model.cases.data_ptr() != target_critic.nnknn_model.cases.data_ptr():
        raise AssertionError("Online and target NN-kNN critics should share the structural state-case tensor.")
    if online_critic.nnknn_model.labels.data_ptr() == target_critic.nnknn_model.labels.data_ptr():
        raise AssertionError("Online and target NN-kNN critics must keep distinct value-label tensors.")
    if online_critic.nnknn_model.biases.data_ptr() == target_critic.nnknn_model.biases.data_ptr():
        raise AssertionError("Online and target NN-kNN critics must keep distinct per-case parameters.")
    if not torch.equal(online_critic.active_case_ids(), target_critic.active_case_ids()):
        raise AssertionError("Online and target critic case IDs should remain structurally aligned.")

    actor_case_count_before = nnknn_state["model"].case_entries
    critic_case_ids_before = online_critic.active_case_ids().detach().clone()
    torch_rng_before = torch.random.get_rng_state().clone()
    holdout_probe = evaluate_critic_holdout(
        "cartpole",
        nnknn_state["model"],
        online_critic,
        target_critic,
        nnknn_cfg,
        episodes=1,
        seed=91_000,
        global_step=nnknn_cfg.total_timesteps,
    )
    if holdout_probe["critic_holdout_samples"] <= 0:
        raise AssertionError("Critic holdout probe did not evaluate any unseen rollout states.")
    if nnknn_state["model"].case_entries != actor_case_count_before:
        raise AssertionError("Critic holdout rollout mutated the NN-kNN actor case memory.")
    if not torch.equal(online_critic.active_case_ids(), critic_case_ids_before):
        raise AssertionError("Critic holdout rollout mutated the NN-kNN critic case memory.")
    if not torch.equal(torch.random.get_rng_state(), torch_rng_before):
        raise AssertionError("Critic holdout rollout changed the subsequent training RNG stream.")

    mlp_cfg = make_nnknn_rl_config(
        "smoke",
        seed=2,
        actor_type="mlp",
        critic_type="mlp",
        critic_holdout_episode_frequency=5,
        critic_holdout_episodes=1,
    )
    mlp_state = train_nnknn_rl("cartpole", mlp_cfg, progress=False)
    mlp_loaded = load_nnknn_rl_checkpoint(mlp_state["checkpoint_path"])
    if mlp_loaded["config"].actor_type != "mlp" or mlp_loaded["config"].critic_type != "mlp":
        raise AssertionError("MLP actor + MLP critic checkpoint reload did not preserve actor/critic types.")
    if mlp_loaded["checkpoint"].get("actor_behavior_policy") != "standard_stochastic_policy":
        raise AssertionError("MLP checkpoint should identify the standard stochastic behavior policy.")
    if mlp_state["summary"]["case_entries"] is not None or mlp_state["summary"]["action_counts"] is not None:
        raise AssertionError("MLP actor summary should not report NN-kNN actor cases.")
    if mlp_state["summary"]["exploration"]["mode"] != "standard_stochastic_policy":
        raise AssertionError("MLP actor summary should record standard stochastic policy sampling.")
    if mlp_state["summary"]["exploration"]["final_step_epsilon"] != 0.0:
        raise AssertionError("MLP actor summary should report zero effective exploration epsilon.")
    if any(float(row["behavior_epsilon"]) != 0.0 for row in mlp_state["training_metrics"]):
        raise AssertionError("MLP actor training should sample directly from its policy without epsilon mixing.")
    if any(float(row["mean_behavior_epsilon"]) != 0.0 for row in mlp_state["loss_metrics"]):
        raise AssertionError("MLP actor policy loss should use unmixed policy probabilities.")
    if not mlp_state["critic_holdout_metrics"]:
        raise AssertionError("Periodic critic holdout evaluation did not produce metrics.")
    mlp_holdout_row = mlp_state["critic_holdout_metrics"][0]
    if mlp_holdout_row["critic_holdout_behavior_epsilon"] != 0.0:
        raise AssertionError("MLP critic holdout should use standard stochastic policy sampling.")
    if not (mlp_state["run_dir"] / "critic_holdout_metrics.csv").exists():
        raise AssertionError("Periodic critic holdout evaluation did not write its CSV artifact.")
    mlp_eval = evaluate_nnknn_rl("cartpole", mlp_loaded["model"], episodes=mlp_cfg.eval_episodes, seed=mlp_cfg.eval_seed)
    if mlp_eval["episodes"] != mlp_cfg.eval_episodes:
        raise AssertionError("MLP actor checkpoint did not evaluate with the configured episode count.")

    mlp_nnknn_cfg = make_nnknn_rl_config("smoke", seed=3, actor_type="mlp", critic_type="nnknn")
    mlp_nnknn_state = train_nnknn_rl("cartpole", mlp_nnknn_cfg, progress=False)
    mlp_nnknn_loaded = load_nnknn_rl_checkpoint(mlp_nnknn_state["checkpoint_path"])
    if mlp_nnknn_loaded["config"].actor_type != "mlp" or mlp_nnknn_loaded["config"].critic_type != "nnknn":
        raise AssertionError("MLP actor + NN-kNN critic checkpoint reload did not preserve actor/critic types.")
    if getattr(mlp_nnknn_loaded["value_model"], "case_entries", 0) <= 0:
        raise AssertionError("MLP actor + NN-kNN critic checkpoint did not preserve critic value cases.")

    mlp_nnknn_capacity_cfg = make_nnknn_rl_config(
        "smoke",
        seed=6,
        actor_type="mlp",
        critic_type="nnknn",
        case_capacity=32,
        total_timesteps=96,
        policy_update_episodes=1,
        eval_episodes=1,
    )
    mlp_nnknn_capacity_state = train_nnknn_rl("cartpole", mlp_nnknn_capacity_cfg, progress=False)
    if mlp_nnknn_capacity_state["summary"]["critic_case_entries"] != 32:
        raise AssertionError("Small-capacity NN-kNN critic should fill to its configured capacity.")
    if (
        mlp_nnknn_capacity_state["summary"]["critic_cases_pruned"]
        + mlp_nnknn_capacity_state["summary"]["critic_cases_replaced"]
        <= 0
    ):
        raise AssertionError("Small-capacity NN-kNN critic maintenance was not reported.")

    separate_capacity_cfg = make_nnknn_rl_config(
        "smoke",
        seed=4,
        critic_type="nnknn",
        case_capacity=64,
        total_timesteps=128,
        policy_update_episodes=2,
    )
    separate_capacity_state = train_nnknn_rl("cartpole", separate_capacity_cfg, progress=False)
    if not isinstance(separate_capacity_state["model"], NNKNNPolicyNetwork):
        raise AssertionError("Small-capacity NN-kNN smoke should retain a standalone actor store.")
    if separate_capacity_state["summary"]["critic_case_entries"] <= 0:
        raise AssertionError("Small-capacity NN-kNN critic did not retain value cases.")
    if not torch.equal(
        separate_capacity_state["value_model"].active_case_ids(),
        separate_capacity_state["target_model"].active_case_ids(),
    ):
        raise AssertionError("Target critic IDs did not follow small-capacity critic compaction.")
    if (
        separate_capacity_state["summary"]["actor_cases_pruned"]
        + separate_capacity_state["summary"]["actor_cases_replaced"]
        + separate_capacity_state["summary"]["critic_cases_pruned"]
        + separate_capacity_state["summary"]["critic_cases_replaced"]
        <= 0
    ):
        raise AssertionError("Small-capacity separate NN-kNN maintenance was not reported.")

    target_ema_cfg = make_nnknn_rl_config(
        "smoke",
        seed=5,
        critic_type="nnknn",
        case_capacity=128,
        total_timesteps=96,
        policy_update_episodes=2,
        critic_target_value_mode="ema",
        critic_target_sync_interval=1,
        critic_target_ema_tau=0.5,
    )
    target_ema_state = train_nnknn_rl("cartpole", target_ema_cfg, progress=False)
    if target_ema_state["summary"]["critic_target_value_mode"] != "ema":
        raise AssertionError("NN-kNN EMA target smoke should record EMA target value mode.")
    if target_ema_state["summary"]["critic_target_value_syncs"] <= 1:
        raise AssertionError("NN-kNN EMA target smoke should perform target value syncs.")
    print("nnknn rl smoke ok")
    print(f"run_dir={state['run_dir']}")
    print(f"nnknn_critic_run_dir={nnknn_state['run_dir']}")
    print(f"mlp_actor_run_dir={mlp_state['run_dir']}")
    print(f"mlp_actor_nnknn_critic_run_dir={mlp_nnknn_state['run_dir']}")
    print(f"mean_return={float(final_eval['mean_return']):.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke checks for Codex cloud environments.")
    parser.add_argument(
        "--mode",
        default="imports",
        choices=("imports", "train", "classification", "rl", "nec", "ppo", "td3", "nnknn_rl"),
        help="Choose a lightweight import check or a tiny training run.",
    )
    args = parser.parse_args()

    if args.mode == "train":
        run_training_smoke()
    elif args.mode == "classification":
        run_classification_smoke()
    elif args.mode == "rl":
        run_rl_smoke()
    elif args.mode == "nec":
        run_nec_smoke()
    elif args.mode == "ppo":
        run_ppo_smoke()
    elif args.mode == "td3":
        run_td3_smoke()
    elif args.mode == "nnknn_rl":
        run_nnknn_rl_smoke()
    else:
        run_import_smoke()


if __name__ == "__main__":
    main()
