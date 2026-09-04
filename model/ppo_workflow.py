"""Repo-native PPO baseline (clipped surrogate objective + GAE).

Scope and conventions
---------------------
- Single-environment rollouts over the shared `datasets/rl_tasks.py` registry
  and the shared `_make_env` wrapper, so PPO sees exactly the same
  environments, seeding, evaluation protocol, early-stopping semantics, and
  run-directory schema as `model/rl_workflow.py` (DQN),
  `model/nec_workflow.py` (NEC), and `model/nnknn_rl_workflow.py` (NN-kNN-RL).
- Advantages come from episode-boundary-aware GAE. `compute_gae` here is a
  vectorised-free re-implementation of the NN-kNN-RL convention: `terminated`
  zeroes the bootstrap term, while a separate `episode_boundaries` mask (set on
  the last transition of every episode, including time-limit truncations)
  breaks the lambda trace. Time-limit truncations therefore bootstrap from
  V(s_{t+1}) instead of being treated as terminal, which is the correct
  treatment for CartPole/Acrobot-style capped episodes.
- The final partial rollout at the fixed step budget is trained, matching the
  NN-kNN-RL rule (AGENTS.md).

Hyperparameter choices (classic control, single env)
----------------------------------------------------
Defaults follow the widely reproduced Stable-Baselines3 `MlpPolicy` PPO recipe
with CleanRL's entropy bonus, which is the standard reference point for
classic-control PPO:

- `clip_coef=0.2`, `gamma=0.99`, `gae_lambda=0.95` — the PPO paper defaults.
- `vf_coef=0.5`, `max_grad_norm=0.5`, `clip_vloss=True` — CleanRL/SB3 defaults.
- `ent_coef=0.01` — CleanRL's classic-control entropy bonus (SB3 uses 0.0);
  the repo's other on-policy workflow (`nnknn_rl_workflow`) also uses 0.01, so
  this keeps the exploration pressure comparable across on-policy baselines.
- `learning_rate=3e-4` with linear annealing over the *configured* budget,
  `rollout_length=2048`, `num_minibatches=32` (minibatch 64), and
  `update_epochs=10` — the SB3 default rollout/optimisation shape.
- `normalize_advantages=True`, per minibatch (CleanRL `norm_adv`).
- `hidden_sizes=(64, 64)` with tanh activations and orthogonal initialisation
  (gain sqrt(2) for hidden layers, 0.01 for the policy head, 1.0 for the value
  head) — the standard PPO initialisation.

Policy heads
------------
- Discrete `action_kind` -> categorical head over logits.
- Continuous `action_kind` -> diagonal Gaussian with a state-independent,
  learned `log_std` parameter. Actions are **clipped** to the Box bounds before
  they reach the environment, and the *unclipped* Gaussian sample is what is
  stored and scored by the ratio/entropy terms (CleanRL
  `ppo_continuous_action.py` convention). This avoids the tanh change-of-
  variables correction; a tanh-squashed head is deliberately *not* used.

Known limitation: unlike CleanRL's MuJoCo recipe, no observation/reward
normalisation wrappers are applied, because the repo's evaluation protocol and
`_make_env` are shared verbatim with the value-based baselines. Continuous
tasks with large reward scales may need `--vf-coef`/`--learning-rate` tuning.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model.device_utils import adam_kwargs_for_device
import torch.nn as nn
import torch.optim as optim

from datasets.rl_tasks import RLTaskSpec, get_rl_task_spec
from model.cnn_encoders import NATURE_CNN_FEATURE_DIM, NatureCNNEncoder, orthogonal_layer_init
from model.rl_workflow import (
    ObservationSpec,
    _build_early_stopping_tracker,
    _build_training_efficiency,
    _finalize_early_stopping_tracker,
    _first_threshold_step,
    _json_default,
    _make_env,
    _require_gymnasium,
    _resolve_device_arg,
    _validate_early_stopping_config,
    describe_observation_space,
    observation_to_array,
    seed_everything,
)

ALGORITHM_NAME = "ppo_clipped_surrogate_gae"

DISCRETE_ACTION_KINDS = frozenset({"discrete", "categorical"})
CONTINUOUS_ACTION_KINDS = frozenset({"continuous", "box", "continuous_box"})


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PPOConfig:
    """Training configuration for the repo-native PPO baseline.

    See the module docstring for the rationale behind each default.
    """

    profile: str = "fast"
    seed: int = 0
    total_timesteps: int = 150_000
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    rollout_length: int = 2_048
    num_minibatches: int = 32
    update_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    normalize_advantages: bool = True
    advantage_epsilon: float = 1e-8
    hidden_sizes: tuple[int, int] = (64, 64)
    log_std_init: float = 0.0
    # Only "clip" is implemented; see the module docstring for why the
    # continuous head is not tanh-squashed.
    continuous_action_handling: str = "clip"
    eval_frequency: int = 5_000
    eval_episodes: int = 20
    eval_seed: int = 10_000
    eval_deterministic: bool = True
    success_threshold: float | None = 475.0
    early_stopping: bool = False
    early_stopping_patience: int = 30
    early_stopping_min_delta: float = 1.0
    early_stopping_min_steps: int = 25_000
    early_stopping_target_score: float | None = None
    source_reference: str = (
        "PPO (Schulman et al. 2017); CleanRL ppo.py / ppo_continuous_action.py and "
        "Stable-Baselines3 MlpPolicy defaults as reference only"
    )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["hidden_sizes"] = list(self.hidden_sizes)
        return data


def make_ppo_config(profile: str = "fast", **overrides: Any) -> PPOConfig:
    """Build a `PPOConfig` for a named profile, mirroring `make_dqn_config`."""

    profiles: dict[str, dict[str, Any]] = {
        "smoke": {
            "profile": "smoke",
            "total_timesteps": 256,
            "rollout_length": 64,
            "num_minibatches": 4,
            "update_epochs": 2,
            "anneal_lr": False,
            "eval_frequency": 128,
            "eval_episodes": 2,
            "success_threshold": None,
            "early_stopping": False,
        },
        "debug": {
            "profile": "debug",
            "total_timesteps": 25_000,
            "rollout_length": 1_024,
            "num_minibatches": 16,
            "update_epochs": 10,
            "eval_frequency": 2_500,
            "eval_episodes": 20,
            "success_threshold": 475.0,
            "early_stopping": True,
        },
        "fast": {
            "profile": "fast",
            "total_timesteps": 150_000,
            "rollout_length": 2_048,
            "num_minibatches": 32,
            "update_epochs": 10,
            "eval_frequency": 5_000,
            "eval_episodes": 20,
            "success_threshold": 475.0,
            "early_stopping": True,
        },
        "gold": {
            "profile": "gold",
            "total_timesteps": 500_000,
            "rollout_length": 2_048,
            "num_minibatches": 32,
            "update_epochs": 10,
            "eval_frequency": 10_000,
            "eval_episodes": 20,
            "success_threshold": 475.0,
            "early_stopping": False,
        },
    }
    normalized = profile.strip().lower()
    if normalized not in profiles:
        raise ValueError(f"Unknown PPO profile '{profile}'. Choose one of: {', '.join(sorted(profiles))}")
    data = {**profiles[normalized], **overrides}
    _validate_early_stopping_config(data)
    if "hidden_sizes" in data and not isinstance(data["hidden_sizes"], tuple):
        data["hidden_sizes"] = tuple(data["hidden_sizes"])
    _validate_ppo_config(data)
    return PPOConfig(**data)


def _validate_ppo_config(data: dict[str, Any]) -> None:
    rollout_length = int(data.get("rollout_length", 2_048))
    num_minibatches = int(data.get("num_minibatches", 32))
    update_epochs = int(data.get("update_epochs", 10))
    clip_coef = float(data.get("clip_coef", 0.2))
    handling = str(data.get("continuous_action_handling", "clip")).strip().lower()
    if rollout_length <= 0:
        raise ValueError("rollout_length must be positive")
    if num_minibatches <= 0:
        raise ValueError("num_minibatches must be positive")
    if update_epochs <= 0:
        raise ValueError("update_epochs must be positive")
    if clip_coef <= 0.0:
        raise ValueError("clip_coef must be positive")
    if handling != "clip":
        raise ValueError(
            "continuous_action_handling only supports 'clip'; the Gaussian head is "
            "not tanh-squashed (see model/ppo_workflow.py module docstring)."
        )
    data["rollout_length"] = rollout_length
    data["num_minibatches"] = num_minibatches
    data["update_epochs"] = update_epochs
    data["clip_coef"] = clip_coef
    data["continuous_action_handling"] = handling
    target_kl = data.get("target_kl", None)
    if target_kl is not None:
        target_kl = float(target_kl)
        if target_kl <= 0.0:
            raise ValueError("target_kl must be None or positive")
        data["target_kl"] = target_kl


def make_ppo_output_dir(
    task_name: str,
    *,
    parent: str | Path = "results/rl",
    suffix: str | None = None,
) -> Path:
    created_at = datetime.now(timezone.utc)
    stem = f"ppo_{task_name}_{created_at.strftime('%Y%m%d_%H%M%S_%f')}"
    if suffix:
        stem = f"{stem}_{suffix}"
    parent_path = Path(parent)
    for attempt in range(100):
        candidate = parent_path / (stem if attempt == 0 else f"{stem}_{attempt + 1}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(f"Could not create a unique PPO output directory under {parent_path}")


# ---------------------------------------------------------------------------
# Action spaces and policy heads
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionSpaceSpec:
    """Normalized description of the environment action space.

    `kind` is "discrete" (categorical head) or "continuous" (diagonal Gaussian
    head). `dim` is the number of actions for discrete spaces and the action
    vector length for continuous spaces. `low`/`high` are the Box bounds used
    for clipping and are None for discrete spaces.
    """

    kind: str
    dim: int
    low: tuple[float, ...] | None = None
    high: tuple[float, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["low"] = None if self.low is None else list(self.low)
        data["high"] = None if self.high is None else list(self.high)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionSpaceSpec":
        low = data.get("low")
        high = data.get("high")
        return cls(
            kind=str(data["kind"]),
            dim=int(data["dim"]),
            low=None if low is None else tuple(float(value) for value in low),
            high=None if high is None else tuple(float(value) for value in high),
        )


def describe_action_space(action_space: Any) -> ActionSpaceSpec:
    """Build an `ActionSpaceSpec` straight from a gymnasium action space.

    Works for any gymnasium `Discrete` or 1-D `Box` space, including spaces
    from environments that are not in `datasets/rl_tasks.py`.
    """

    gym = _require_gymnasium()
    if isinstance(action_space, gym.spaces.Discrete):
        return ActionSpaceSpec(kind="discrete", dim=int(action_space.n))
    if isinstance(action_space, gym.spaces.Box):
        shape = tuple(int(value) for value in action_space.shape)
        if len(shape) != 1:
            raise ValueError(f"PPO supports 1-D Box action spaces, got shape {shape}.")
        low = np.asarray(action_space.low, dtype=np.float64).reshape(-1)
        high = np.asarray(action_space.high, dtype=np.float64).reshape(-1)
        if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
            raise ValueError("PPO requires finite Box action bounds for clipping.")
        return ActionSpaceSpec(
            kind="continuous",
            dim=shape[0],
            low=tuple(float(value) for value in low),
            high=tuple(float(value) for value in high),
        )
    raise ValueError(
        f"Unsupported action space {type(action_space).__name__} for the PPO baseline; "
        "expected gymnasium Discrete or Box."
    )


def resolve_task_action_kind(action_kind: str) -> str:
    """Map a task-spec `action_kind` string onto a PPO policy-head kind."""

    normalized = str(action_kind).strip().lower()
    if normalized in DISCRETE_ACTION_KINDS:
        return "discrete"
    if normalized in CONTINUOUS_ACTION_KINDS:
        return "continuous"
    raise ValueError(
        f"Unsupported task action_kind '{action_kind}'. Expected one of "
        f"{sorted(DISCRETE_ACTION_KINDS | CONTINUOUS_ACTION_KINDS)}."
    )


def _layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def _mlp_trunk(obs_dim: int, hidden_sizes: tuple[int, int]) -> tuple[nn.Sequential, int]:
    h1, h2 = hidden_sizes
    trunk = nn.Sequential(
        _layer_init(nn.Linear(obs_dim, h1)),
        nn.Tanh(),
        _layer_init(nn.Linear(h1, h2)),
        nn.Tanh(),
    )
    return trunk, h2


def _cnn_trunk(obs_shape: tuple[int, ...]) -> tuple[nn.Module, int]:
    """Nature-CNN trunk for image observations, orthogonally initialised.

    PPO keeps its SEPARATE-trunk design on images: the actor and the critic get
    one encoder each (CleanRL's `ppo_atari.py` shares a single trunk instead).
    That is deliberate -- it keeps the image path structurally identical to the
    flat path, where this repo's PPO also uses two independent trunks.
    """

    trunk = NatureCNNEncoder(
        obs_shape,
        feature_dim=NATURE_CNN_FEATURE_DIM,
        layer_init=orthogonal_layer_init,
    )
    return trunk, trunk.feature_dim


def _as_observation_spec(observation: "ObservationSpec | int") -> ObservationSpec:
    if isinstance(observation, ObservationSpec):
        return observation
    dim = int(observation)
    return ObservationSpec(kind="flat_box", shape=(dim,), dim=dim, numpy_dtype="float32")


class PPOAgent(nn.Module):
    """Separate-trunk PPO actor-critic with a discrete or continuous head."""

    def __init__(
        self,
        observation: "ObservationSpec | int",
        action_spec: ActionSpaceSpec,
        *,
        hidden_sizes: tuple[int, int] = (64, 64),
        log_std_init: float = 0.0,
    ):
        super().__init__()
        self.observation_spec = _as_observation_spec(observation)
        self.obs_dim = int(self.observation_spec.dim)
        self.action_spec = action_spec
        self.hidden_sizes = tuple(hidden_sizes)

        # Trunks are constructed critic-first, then actor, in both branches, so
        # the torch RNG consumption order of the flat path is unchanged.
        if self.observation_spec.is_image:
            critic_trunk, critic_out = _cnn_trunk(self.observation_spec.shape)
        else:
            critic_trunk, critic_out = _mlp_trunk(self.obs_dim, self.hidden_sizes)
        self.critic = nn.Sequential(critic_trunk, _layer_init(nn.Linear(critic_out, 1), std=1.0))
        if self.observation_spec.is_image:
            actor_trunk, actor_out = _cnn_trunk(self.observation_spec.shape)
        else:
            actor_trunk, actor_out = _mlp_trunk(self.obs_dim, self.hidden_sizes)
        self.actor = nn.Sequential(
            actor_trunk,
            _layer_init(nn.Linear(actor_out, action_spec.dim), std=0.01),
        )
        if action_spec.kind == "continuous":
            self.actor_logstd = nn.Parameter(torch.full((1, action_spec.dim), float(log_std_init)))
            low = torch.as_tensor(action_spec.low, dtype=torch.float32).view(1, -1)
            high = torch.as_tensor(action_spec.high, dtype=torch.float32).view(1, -1)
            self.register_buffer("action_low", low)
            self.register_buffer("action_high", high)
        elif action_spec.kind != "discrete":
            raise ValueError(f"Unknown action head kind '{action_spec.kind}'.")

    # -- distributions ------------------------------------------------------

    @property
    def is_continuous(self) -> bool:
        return self.action_spec.kind == "continuous"

    @property
    def is_image_observation(self) -> bool:
        return self.observation_spec.is_image

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).view(-1)

    def policy_distribution(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        head = self.actor(obs)
        if self.is_continuous:
            std = torch.exp(self.actor_logstd.expand_as(head))
            return torch.distributions.Independent(torch.distributions.Normal(head, std), 1)
        return torch.distributions.Categorical(logits=head)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (log_prob, entropy, value) for stored actions."""

        distribution = self.policy_distribution(obs)
        if self.is_continuous:
            action_input = actions.view(-1, self.action_spec.dim).to(dtype=obs.dtype)
        else:
            action_input = actions.view(-1).long()
        log_prob = distribution.log_prob(action_input)
        entropy = distribution.entropy()
        return log_prob, entropy, self.get_value(obs)

    def act(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or take the mode of) the policy for a single observation batch.

        Returns `(action, log_prob, value)`. Continuous actions are the raw,
        *unclipped* Gaussian samples; use `env_action` to obtain what should be
        passed to `env.step`.
        """

        distribution = self.policy_distribution(obs)
        if deterministic:
            if self.is_continuous:
                action = distribution.base_dist.loc
            else:
                action = distribution.logits.argmax(dim=-1)
        else:
            action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action, log_prob, self.get_value(obs)

    def env_action(self, action: torch.Tensor) -> Any:
        """Convert a sampled action into something `env.step` accepts."""

        if self.is_continuous:
            clipped = torch.clamp(
                action.view(-1, self.action_spec.dim),
                self.action_low,
                self.action_high,
            )
            return clipped.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return int(action.view(-1)[0].item())


# ---------------------------------------------------------------------------
# GAE and the PPO update
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: list[float] | torch.Tensor,
    values: list[float] | torch.Tensor,
    next_values: list[float] | torch.Tensor,
    terminated: list[bool] | torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
    episode_boundaries: list[bool] | torch.Tensor | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Episode-boundary-aware GAE, matching the NN-kNN-RL convention.

    `terminated` marks true environment terminations and zeroes the bootstrap
    term; `episode_boundaries` marks the last transition of every episode
    (terminations *and* time-limit truncations) and breaks the lambda trace so
    advantages never leak across episodes concatenated in one rollout buffer.
    When `episode_boundaries` is None it falls back to `terminated`.
    """

    rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device).view(-1)
    values_t = torch.as_tensor(values, dtype=torch.float32, device=device).view(-1)
    next_values_t = torch.as_tensor(next_values, dtype=torch.float32, device=device).view(-1)
    terminated_t = torch.as_tensor(terminated, dtype=torch.bool, device=device).view(-1)
    if episode_boundaries is None:
        episode_boundaries_t = terminated_t
    else:
        episode_boundaries_t = torch.as_tensor(episode_boundaries, dtype=torch.bool, device=device).view(-1)
    if not (
        rewards_t.numel()
        == values_t.numel()
        == next_values_t.numel()
        == terminated_t.numel()
        == episode_boundaries_t.numel()
    ):
        raise ValueError("GAE inputs must have the same length")

    advantages = torch.zeros_like(rewards_t)
    running_advantage = torch.zeros((), dtype=torch.float32, device=rewards_t.device)
    for idx in range(rewards_t.numel() - 1, -1, -1):
        not_done = (~terminated_t[idx]).to(dtype=torch.float32)
        continue_trace = (~episode_boundaries_t[idx]).to(dtype=torch.float32)
        delta = rewards_t[idx] + float(gamma) * next_values_t[idx] * not_done - values_t[idx]
        running_advantage = delta + float(gamma) * float(gae_lambda) * continue_trace * running_advantage
        advantages[idx] = running_advantage
    value_targets = advantages + values_t
    return advantages, value_targets


def explained_variance(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    targets = targets.detach().float()
    predictions = predictions.detach().float()
    target_var = torch.var(targets, unbiased=False)
    if float(target_var.cpu().item()) <= 1e-12:
        return 0.0
    residual_var = torch.var(targets - predictions, unbiased=False)
    return float((1.0 - residual_var / target_var).cpu().item())


def _rollout_obs_tensor(observations: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a rollout's observations to `device`.

    Flat observations become float32 exactly as before. `uint8` image frames
    stay `uint8` on the device -- a whole 2048-step rollout of 4x84x84 frames is
    4x smaller that way -- because `NatureCNNEncoder` casts to float and divides
    by 255 itself. Only the discrete head consumes image observations, so no
    dtype-sensitive continuous-action path sees a uint8 tensor.
    """

    if isinstance(observations, torch.Tensor):
        if observations.dtype == torch.uint8:
            return observations.to(device)
        return observations.to(device=device, dtype=torch.float32)
    array = np.asarray(observations)
    if array.dtype == np.uint8:
        return torch.as_tensor(array, device=device)
    return torch.as_tensor(np.asarray(array, dtype=np.float32), dtype=torch.float32, device=device)


def ppo_update(
    agent: PPOAgent,
    optimizer: optim.Optimizer,
    config: PPOConfig,
    *,
    observations: np.ndarray | torch.Tensor,
    next_observations: np.ndarray | torch.Tensor,
    actions: np.ndarray | torch.Tensor,
    log_probs: np.ndarray | torch.Tensor,
    rewards: np.ndarray | torch.Tensor,
    terminated: np.ndarray | torch.Tensor,
    episode_boundaries: np.ndarray | torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """Run one PPO update over a collected rollout and report its diagnostics.

    Values (and the bootstrap values used by GAE) are recomputed once from the
    frozen pre-update critic, mirroring how `nnknn_rl_workflow` scores a
    rollout before optimising it.

    Reported losses are means over the minibatch steps actually taken, and
    `explained_variance` is an in-sample, *post-update* critic diagnostic (as
    in `nnknn_rl_workflow`'s `critic_train_*` fields); neither affects
    gradients, checkpoint selection, or early stopping.
    """

    obs_t = _rollout_obs_tensor(observations, device)
    next_obs_t = _rollout_obs_tensor(next_observations, device)
    rewards_t = torch.as_tensor(np.asarray(rewards, dtype=np.float32), dtype=torch.float32, device=device)
    terminated_t = torch.as_tensor(np.asarray(terminated, dtype=bool), dtype=torch.bool, device=device)
    boundaries_t = torch.as_tensor(
        np.asarray(episode_boundaries, dtype=bool), dtype=torch.bool, device=device
    )
    old_log_probs = torch.as_tensor(
        np.asarray(log_probs, dtype=np.float32), dtype=torch.float32, device=device
    ).view(-1)
    if agent.is_continuous:
        actions_t = torch.as_tensor(
            np.asarray(actions, dtype=np.float32), dtype=torch.float32, device=device
        ).view(-1, agent.action_spec.dim)
    else:
        actions_t = torch.as_tensor(np.asarray(actions, dtype=np.int64), dtype=torch.long, device=device).view(-1)

    batch_size = int(obs_t.shape[0])
    if batch_size == 0:
        raise ValueError("ppo_update requires at least one collected transition")

    agent.eval()
    with torch.no_grad():
        critic_inputs = torch.cat([obs_t, next_obs_t], dim=0)
        critic_values = agent.get_value(critic_inputs)
        values_t, next_values_t = critic_values.chunk(2)
        advantages_t, value_targets_t = compute_gae(
            rewards_t,
            values_t,
            next_values_t,
            terminated_t,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            episode_boundaries=boundaries_t,
            device=device,
        )
    agent.train()

    minibatch_size = max(1, batch_size // int(config.num_minibatches))
    clip_coef = float(config.clip_coef)
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    approx_kls: list[float] = []
    clip_fractions: list[float] = []
    grad_norms: list[float] = []
    minibatch_updates = 0
    epochs_run = 0
    stopped_on_kl = False

    for _epoch in range(int(config.update_epochs)):
        epochs_run += 1
        order = np.random.permutation(batch_size)
        epoch_kls: list[float] = []
        for start in range(0, batch_size, minibatch_size):
            indices = torch.as_tensor(order[start : start + minibatch_size], dtype=torch.long, device=device)
            if indices.numel() == 0:
                continue
            mb_obs = obs_t[indices]
            mb_actions = actions_t[indices]
            mb_old_log_probs = old_log_probs[indices]
            mb_advantages = advantages_t[indices]
            mb_value_targets = value_targets_t[indices]
            mb_old_values = values_t[indices]

            new_log_probs, entropy, new_values = agent.evaluate_actions(mb_obs, mb_actions)
            log_ratio = new_log_probs - mb_old_log_probs
            ratio = log_ratio.exp()
            with torch.no_grad():
                approx_kl = float(((ratio - 1.0) - log_ratio).mean().cpu().item())
                clip_fraction = float(((ratio - 1.0).abs() > clip_coef).float().mean().cpu().item())
            epoch_kls.append(approx_kl)

            if config.normalize_advantages and mb_advantages.numel() > 1:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / mb_advantages.std(
                    unbiased=False
                ).clamp_min(float(config.advantage_epsilon))

            policy_loss_unclipped = -mb_advantages * ratio
            policy_loss_clipped = -mb_advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
            policy_loss = torch.max(policy_loss_unclipped, policy_loss_clipped).mean()

            if config.clip_vloss:
                value_loss_unclipped = (new_values - mb_value_targets) ** 2
                clipped_values = mb_old_values + torch.clamp(
                    new_values - mb_old_values, -clip_coef, clip_coef
                )
                value_loss_clipped = (clipped_values - mb_value_targets) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
            else:
                value_loss = 0.5 * ((new_values - mb_value_targets) ** 2).mean()

            entropy_mean = entropy.mean()
            loss = policy_loss - float(config.ent_coef) * entropy_mean + float(config.vf_coef) * value_loss

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), float(config.max_grad_norm))
            optimizer.step()

            policy_losses.append(float(policy_loss.detach().cpu().item()))
            value_losses.append(float(value_loss.detach().cpu().item()))
            entropies.append(float(entropy_mean.detach().cpu().item()))
            approx_kls.append(approx_kl)
            clip_fractions.append(clip_fraction)
            grad_norms.append(float(grad_norm.detach().cpu().item()))
            minibatch_updates += 1

        if config.target_kl is not None and epoch_kls and float(np.mean(epoch_kls)) > float(config.target_kl):
            stopped_on_kl = True
            break

    with torch.no_grad():
        post_values = agent.get_value(obs_t)
    return {
        "rollout_steps": batch_size,
        "minibatch_size": minibatch_size,
        "minibatch_updates": minibatch_updates,
        "update_epochs_run": epochs_run,
        "stopped_on_target_kl": stopped_on_kl,
        "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
        "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
        "entropy": float(np.mean(entropies)) if entropies else 0.0,
        "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
        "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
        "grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "advantage_mean": float(advantages_t.mean().cpu().item()),
        "advantage_std": float(advantages_t.std(unbiased=False).cpu().item()),
        "value_target_mean": float(value_targets_t.mean().cpu().item()),
        "explained_variance": explained_variance(post_values, value_targets_t),
    }


# ---------------------------------------------------------------------------
# Environment plumbing
# ---------------------------------------------------------------------------


def _validate_ppo_env_spaces(env: Any, spec: RLTaskSpec) -> tuple[ObservationSpec, ActionSpaceSpec]:
    """Validate observation/action spaces and reconcile them with the task spec."""

    obs_spec = describe_observation_space(env, spec)
    action_spec = describe_action_space(env.action_space)
    expected_kind = resolve_task_action_kind(spec.action_kind)
    if action_spec.kind != expected_kind:
        raise ValueError(
            f"Task '{spec.name}' declares action_kind='{spec.action_kind}' but {spec.env_id} "
            f"exposes a {action_spec.kind} action space."
        )
    if obs_spec.is_image and action_spec.kind != "discrete":
        raise ValueError(
            f"Task '{spec.name}' pairs image observations with a {action_spec.kind} action space; "
            "the image path is discrete-action only."
        )
    return obs_spec, action_spec


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _copy_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _annealed_learning_rate(config: PPOConfig, consumed_steps: int) -> float:
    if not config.anneal_lr or config.total_timesteps <= 0:
        return float(config.learning_rate)
    progress = min(max(float(consumed_steps) / float(config.total_timesteps), 0.0), 1.0)
    return float(config.learning_rate) * (1.0 - progress)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_ppo(
    task_name: str,
    agent: PPOAgent,
    *,
    episodes: int = 20,
    seed: int = 10_000,
    device: str | torch.device | None = None,
    deterministic: bool = True,
) -> dict[str, Any]:
    """Run the shared evaluation protocol for a PPO agent.

    `deterministic=True` takes the categorical argmax / Gaussian mean, which is
    the analogue of the greedy evaluation used by the DQN and NEC baselines.
    """

    spec = get_rl_task_spec(task_name)
    if device is None:
        run_device = next(agent.parameters()).device
    else:
        run_device = device if isinstance(device, torch.device) else torch.device(device)
    env = _make_env(spec, seed=seed)
    agent.eval()
    returns: list[float] = []
    lengths: list[int] = []
    rows: list[dict[str, Any]] = []
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            done = False
            episode_return = 0.0
            episode_length = 0
            while not done:
                obs_tensor = torch.as_tensor(
                    np.asarray(obs, dtype=np.float32), dtype=torch.float32, device=run_device
                ).unsqueeze(0)
                with torch.no_grad():
                    action, _log_prob, _value = agent.act(obs_tensor, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = env.step(agent.env_action(action))
                episode_return += float(reward)
                episode_length += 1
                done = bool(terminated or truncated)
            returns.append(episode_return)
            lengths.append(episode_length)
            rows.append(
                {
                    "episode": episode + 1,
                    "return": episode_return,
                    "length": episode_length,
                    "seed": seed + episode,
                }
            )
    finally:
        env.close()
    returns_arr = np.asarray(returns, dtype=np.float32)
    lengths_arr = np.asarray(lengths, dtype=np.float32)
    return {
        "episodes": episodes,
        "seed": seed,
        "mean_return": float(returns_arr.mean()) if len(returns_arr) else 0.0,
        "std_return": float(returns_arr.std()) if len(returns_arr) else 0.0,
        "min_return": float(returns_arr.min()) if len(returns_arr) else 0.0,
        "max_return": float(returns_arr.max()) if len(returns_arr) else 0.0,
        "mean_length": float(lengths_arr.mean()) if len(lengths_arr) else 0.0,
        "episode_metrics": rows,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_ppo(
    task_name: str = "cartpole",
    config: PPOConfig | None = None,
    *,
    output_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Train a PPO baseline and write the shared run artifacts."""

    spec = get_rl_task_spec(task_name)
    cfg = config or make_ppo_config(spec.default_profile)
    seed_everything(cfg.seed)
    run_device = _resolve_device_arg(device)

    env = _make_env(spec, seed=cfg.seed)
    obs_spec, action_spec = _validate_ppo_env_spaces(env, spec)
    obs_dim = obs_spec.dim
    agent = PPOAgent(
        obs_spec,
        action_spec,
        hidden_sizes=cfg.hidden_sizes,
        log_std_init=cfg.log_std_init,
    ).to(run_device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5,
                           **adam_kwargs_for_device(run_device))

    run_dir = Path(output_dir) if output_dir is not None else make_ppo_output_dir(spec.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.pt"
    created_at = datetime.now(timezone.utc)

    training_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    latest_update: dict[str, Any] | None = None
    best_eval: dict[str, Any] | None = None
    best_eval_step: int | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    early_stopping = _build_early_stopping_tracker(cfg, spec)
    actual_timesteps = 0
    update_index = 0
    episode_index = 0

    obs, _ = env.reset(seed=cfg.seed)
    episode_return = 0.0
    episode_length = 0
    global_step = 0
    stop_requested = False

    try:
        while global_step < cfg.total_timesteps:
            learning_rate = _annealed_learning_rate(cfg, global_step)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            rollout_steps = min(int(cfg.rollout_length), int(cfg.total_timesteps) - global_step)
            buf_obs: list[np.ndarray] = []
            buf_next_obs: list[np.ndarray] = []
            buf_actions: list[Any] = []
            buf_log_probs: list[float] = []
            buf_rewards: list[float] = []
            buf_terminated: list[bool] = []
            buf_boundaries: list[bool] = []
            rollout_start_step = global_step

            agent.eval()
            for _step in range(rollout_steps):
                obs_array = observation_to_array(obs, obs_spec)
                obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=run_device).unsqueeze(0)
                with torch.no_grad():
                    action, log_prob, _value = agent.act(obs_tensor)
                next_obs, reward, terminated, truncated, _ = env.step(agent.env_action(action))
                episode_done = bool(terminated or truncated)

                buf_obs.append(obs_array)
                buf_next_obs.append(observation_to_array(next_obs, obs_spec))
                if agent.is_continuous:
                    buf_actions.append(action.view(-1).detach().cpu().numpy().astype(np.float32))
                else:
                    buf_actions.append(int(action.view(-1)[0].item()))
                buf_log_probs.append(float(log_prob.view(-1)[0].detach().cpu().item()))
                buf_rewards.append(float(reward))
                buf_terminated.append(bool(terminated))
                buf_boundaries.append(episode_done)

                episode_return += float(reward)
                episode_length += 1
                global_step += 1
                actual_timesteps = global_step
                completed_step = global_step

                if episode_done:
                    episode_index += 1
                    training_rows.append(
                        {
                            "global_step": completed_step,
                            "episode": episode_index,
                            "episode_return": episode_return,
                            "episode_length": episode_length,
                            "update": update_index,
                            "learning_rate": learning_rate,
                            "policy_loss": None if latest_update is None else latest_update["policy_loss"],
                            "value_loss": None if latest_update is None else latest_update["value_loss"],
                            "entropy": None if latest_update is None else latest_update["entropy"],
                            "approx_kl": None if latest_update is None else latest_update["approx_kl"],
                        }
                    )
                    if progress and (episode_index <= 5 or episode_index % 10 == 0):
                        print(
                            "[ppo] "
                            f"step={completed_step} episode={episode_index} "
                            f"return={episode_return:.1f} length={episode_length} "
                            f"lr={learning_rate:.2e} "
                            f"entropy={None if latest_update is None else round(latest_update['entropy'], 4)}",
                            flush=True,
                        )
                    obs, _ = env.reset(seed=cfg.seed + episode_index)
                    episode_return = 0.0
                    episode_length = 0
                else:
                    obs = next_obs

                if cfg.eval_frequency > 0 and completed_step % cfg.eval_frequency == 0:
                    eval_metrics = evaluate_ppo(
                        spec.name,
                        agent,
                        episodes=cfg.eval_episodes,
                        seed=cfg.eval_seed,
                        device=run_device,
                        deterministic=cfg.eval_deterministic,
                    )
                    eval_row = {
                        "global_step": completed_step,
                        "update": update_index,
                        "mean_return": eval_metrics["mean_return"],
                        "std_return": eval_metrics["std_return"],
                        "min_return": eval_metrics["min_return"],
                        "max_return": eval_metrics["max_return"],
                        "mean_length": eval_metrics["mean_length"],
                        "episodes": cfg.eval_episodes,
                    }
                    eval_rows.append(eval_row)
                    if best_eval is None or eval_metrics["mean_return"] > best_eval["mean_return"]:
                        best_eval = eval_metrics
                        best_eval_step = completed_step
                        best_model_state = _copy_state_dict_to_cpu(agent)
                    if progress:
                        print(
                            "[ppo][eval] "
                            f"step={completed_step} mean_return={eval_row['mean_return']:.2f} "
                            f"max_return={eval_row['max_return']:.2f}",
                            flush=True,
                        )
                    if early_stopping.update(eval_metrics["mean_return"], completed_step):
                        stop_requested = True
                    agent.eval()

                if stop_requested:
                    break

            # Train the collected rollout, including the final partial rollout
            # at the budget (or early-stopping) boundary.
            if buf_obs:
                update_index += 1
                latest_update = ppo_update(
                    agent,
                    optimizer,
                    cfg,
                    observations=np.asarray(buf_obs),
                    next_observations=np.asarray(buf_next_obs),
                    actions=np.asarray(buf_actions),
                    log_probs=np.asarray(buf_log_probs, dtype=np.float32),
                    rewards=np.asarray(buf_rewards, dtype=np.float32),
                    terminated=np.asarray(buf_terminated, dtype=bool),
                    episode_boundaries=np.asarray(buf_boundaries, dtype=bool),
                    device=run_device,
                )
                loss_rows.append(
                    {
                        "global_step": global_step,
                        "update": update_index,
                        "rollout_start_step": rollout_start_step,
                        "learning_rate": learning_rate,
                        **latest_update,
                    }
                )

            if stop_requested:
                if progress:
                    print(
                        f"[ppo][early-stop] step={actual_timesteps} reason={early_stopping.stopping_reason}",
                        flush=True,
                    )
                break
    finally:
        env.close()

    _finalize_early_stopping_tracker(early_stopping, actual_timesteps)

    last_eval = evaluate_ppo(
        spec.name,
        agent,
        episodes=cfg.eval_episodes,
        seed=cfg.eval_seed,
        device=run_device,
        deterministic=cfg.eval_deterministic,
    )
    if best_eval is None or last_eval["mean_return"] >= best_eval["mean_return"]:
        selected_eval = last_eval
        selected_step = actual_timesteps
        selected_source = "final"
        selected_model_state = _copy_state_dict_to_cpu(agent)
    else:
        selected_eval = best_eval
        selected_step = int(best_eval_step or 0)
        selected_source = "best_eval"
        selected_model_state = best_model_state or _copy_state_dict_to_cpu(agent)
        agent.load_state_dict(selected_model_state)

    passed = (
        True
        if cfg.success_threshold is None
        else selected_eval["mean_return"] >= cfg.success_threshold
    )
    first_success_step = _first_threshold_step(eval_rows, cfg.success_threshold)
    if (
        first_success_step is None
        and cfg.success_threshold is not None
        and last_eval["mean_return"] >= cfg.success_threshold
    ):
        first_success_step = actual_timesteps
    training_efficiency = _build_training_efficiency(
        selected_eval=selected_eval,
        last_eval=last_eval,
        selected_step=selected_step,
        selected_source=selected_source,
        total_timesteps=actual_timesteps,
        success_threshold=cfg.success_threshold,
        first_success_step=first_success_step,
    )
    checkpoint = {
        "algorithm": ALGORITHM_NAME,
        "model_state_dict": selected_model_state,
        "task": spec.to_dict(),
        "config": cfg.to_dict(),
        "obs_dim": obs_dim,
        "obs_shape": list(obs_spec.shape),
        "observation_spec": obs_spec.to_dict(),
        "action_spec": action_spec.to_dict(),
        "action_dim": action_spec.dim,
        "selected_eval": {k: v for k, v in selected_eval.items() if k != "episode_metrics"},
        "last_eval": {k: v for k, v in last_eval.items() if k != "episode_metrics"},
        "selected_step": selected_step,
        "selected_source": selected_source,
        "training_efficiency": training_efficiency,
        "configured_total_timesteps": cfg.total_timesteps,
        "actual_timesteps": actual_timesteps,
        "updates": update_index,
        "early_stopping": early_stopping.to_dict(),
        "passed": passed,
    }
    torch.save(checkpoint, checkpoint_path)

    _write_json(
        run_dir / "config.json",
        {
            "created_at_utc": created_at.isoformat(),
            "algorithm": ALGORITHM_NAME,
            "task": spec.to_dict(),
            "config": cfg.to_dict(),
            "action_spec": action_spec.to_dict(),
            "device": str(run_device),
            "observation": obs_spec.to_dict(),
            "source_reference": cfg.source_reference,
            "gae": {
                "gamma": cfg.gamma,
                "gae_lambda": cfg.gae_lambda,
                "episode_boundary_aware": True,
                "truncation_bootstrap": True,
            },
            "ppo": {
                "clip_coef": cfg.clip_coef,
                "clip_vloss": cfg.clip_vloss,
                "ent_coef": cfg.ent_coef,
                "vf_coef": cfg.vf_coef,
                "max_grad_norm": cfg.max_grad_norm,
                "rollout_length": cfg.rollout_length,
                "num_minibatches": cfg.num_minibatches,
                "update_epochs": cfg.update_epochs,
                "normalize_advantages": cfg.normalize_advantages,
                "anneal_lr": cfg.anneal_lr,
                "target_kl": cfg.target_kl,
                "policy_head": (
                    "categorical"
                    if action_spec.kind == "discrete"
                    else "diagonal_gaussian_clipped_to_action_bounds"
                ),
                "trunk": (
                    "nature_cnn_separate_actor_critic"
                    if obs_spec.is_image
                    else "mlp_tanh_separate_actor_critic"
                ),
            },
        },
    )
    _write_csv(run_dir / "training_metrics.csv", training_rows)
    _write_csv(run_dir / "loss_metrics.csv", loss_rows)
    _write_csv(run_dir / "eval_metrics.csv", eval_rows)
    _write_csv(run_dir / "final_eval_episodes.csv", selected_eval["episode_metrics"])
    _write_csv(run_dir / "last_eval_episodes.csv", last_eval["episode_metrics"])
    summary = {
        "task": spec.name,
        "env_id": spec.env_id,
        "algorithm": ALGORITHM_NAME,
        "profile": cfg.profile,
        "seed": cfg.seed,
        "total_timesteps": actual_timesteps,
        "configured_total_timesteps": cfg.total_timesteps,
        "actual_timesteps": actual_timesteps,
        "updates": update_index,
        "rollout_length": cfg.rollout_length,
        "action_kind": action_spec.kind,
        "eval_episodes": cfg.eval_episodes,
        "success_threshold": cfg.success_threshold,
        "passed": passed,
        "final_eval": {k: v for k, v in selected_eval.items() if k != "episode_metrics"},
        "last_eval": {k: v for k, v in last_eval.items() if k != "episode_metrics"},
        "selected_step": selected_step,
        "selected_source": selected_source,
        "training_efficiency": training_efficiency,
        "early_stopping": early_stopping.to_dict(),
        "checkpoint_path": str(checkpoint_path),
        "run_dir": str(run_dir),
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(
        run_dir / "manifest.json",
        {
            "created_at_utc": created_at.isoformat(),
            "task": spec.name,
            "env_id": spec.env_id,
            "algorithm": ALGORITHM_NAME,
            "profile": cfg.profile,
            "outputs": [
                "config.json",
                "training_metrics.csv",
                "loss_metrics.csv",
                "eval_metrics.csv",
                "final_eval_episodes.csv",
                "last_eval_episodes.csv",
                "summary.json",
                "manifest.json",
                "checkpoint.pt",
            ],
        },
    )
    if progress:
        print(
            "[ppo] finished "
            f"profile={cfg.profile} mean_eval_return={selected_eval['mean_return']:.2f} "
            f"passed={passed} run_dir={run_dir}",
            flush=True,
        )
    return {
        "model": agent,
        "agent": agent,
        "task": spec,
        "config": cfg,
        "action_spec": action_spec,
        "run_dir": run_dir,
        "checkpoint_path": checkpoint_path,
        "training_metrics": training_rows,
        "loss_metrics": loss_rows,
        "eval_metrics": eval_rows,
        "final_eval": selected_eval,
        "last_eval": last_eval,
        "passed": passed,
        "summary": summary,
    }


def load_ppo_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    run_device = _resolve_device_arg(device)
    checkpoint = torch.load(checkpoint_path, map_location=run_device, weights_only=False)
    recorded_algorithm = checkpoint.get("algorithm")
    if recorded_algorithm != ALGORITHM_NAME:
        raise ValueError(
            f"Checkpoint algorithm {recorded_algorithm!r} is not a PPO checkpoint. "
            f"Expected {ALGORITHM_NAME!r}."
        )
    config_data = dict(checkpoint["config"])
    config_data["hidden_sizes"] = tuple(config_data["hidden_sizes"])
    cfg = PPOConfig(**config_data)
    action_spec = ActionSpaceSpec.from_dict(checkpoint["action_spec"])
    recorded_observation = checkpoint.get("observation_spec")
    observation: ObservationSpec | int
    if recorded_observation is None:
        observation = int(checkpoint["obs_dim"])
    else:
        observation = ObservationSpec(
            kind=str(recorded_observation["kind"]),
            shape=tuple(int(value) for value in recorded_observation["shape"]),
            dim=int(recorded_observation["dim"]),
            numpy_dtype=str(recorded_observation["numpy_dtype"]),
        )
    agent = PPOAgent(
        observation,
        action_spec,
        hidden_sizes=cfg.hidden_sizes,
        log_std_init=cfg.log_std_init,
    ).to(run_device)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()
    return {
        "model": agent,
        "agent": agent,
        "config": cfg,
        "action_spec": action_spec,
        "task": checkpoint["task"],
        "checkpoint": checkpoint,
        "device": run_device,
    }
