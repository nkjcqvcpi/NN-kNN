"""Repo-native TD3 baseline (Twin Delayed DDPG) for continuous-action tasks.

Scope and conventions
---------------------
- TD3 is a **continuous-control reference baseline**. It is continuous-action
  only by construction (a deterministic tanh actor plus state-action critics),
  so it raises a clear error on the discrete-action tasks in
  `datasets/rl_tasks.py`. The NN-kNN actor is discrete-only (one-hot action
  labels per case), so no NN-kNN-RL row exists on the continuous tasks; TD3 is
  compared against PPO's diagonal-Gaussian head instead (NEXT_STEPS_PLAN.md
  I-2).
- Single-environment training over the shared `datasets/rl_tasks.py` registry
  and the shared `_make_env` wrapper, so TD3 sees exactly the same
  environments, seeding, evaluation protocol, early-stopping semantics
  (AGENTS.md: patience 30, min-delta 1.0, patience counted after 25,000 steps,
  immediate stop at the task target), and run-directory schema as
  `model/rl_workflow.py` (DQN), `model/nec_workflow.py` (NEC),
  `model/ppo_workflow.py` (PPO), and `model/nnknn_rl_workflow.py`.
- The uniform replay buffer is `ContinuousReplayBuffer`, which subclasses the
  DQN buffer in `model/rl_workflow.py`: the DQN buffer stores a single int64
  action per transition, so only the action storage/sampling is overridden.
  Everything else (ring-buffer position bookkeeping, the single
  `np.random.randint` sampling call) is inherited unchanged.
- `dones` stores `terminated` only, never `truncated`, so time-limit
  truncations bootstrap from the target critics. This matches the DQN buffer
  convention and is the correct treatment for Pendulum-v1, which only ever
  truncates.

Hyperparameter choices
----------------------
Defaults follow Fujimoto et al. (2018) as reproduced by CleanRL's
`td3_continuous_action.py`:

- `tau=0.005` (Polyak averaging), `gamma=0.99`, `batch_size=256`.
- `policy_frequency=2` — delayed policy and target updates: the critics take
  one gradient step per environment step, the actor and all three target
  networks update every second critic step.
- `policy_noise=0.2`, `noise_clip=0.5` — target policy smoothing. Both are in
  *normalized* action units: the sampled noise is clipped to +/- `noise_clip`
  and then multiplied by the actor's `action_scale`, exactly as CleanRL does,
  so the numbers stay comparable across environments with different Box
  bounds.
- `exploration_noise=0.1` — Gaussian behaviour noise, also scaled by
  `action_scale` and clipped back into the Box bounds.
- `learning_rate=3e-4` for both the critic and the actor optimizer
  (`policy_learning_rate` overrides the actor's if set).
- `hidden_sizes=(256, 256)` with ReLU activations (CleanRL's shape; the TD3
  paper uses 400/300).
- `learning_starts=1_000` with uniform random actions before it. CleanRL uses
  25,000 for 1M-step MuJoCo runs; that is most of the repo's 150k fast budget,
  so it is scaled down to the DQN fast-profile value.
- `max_grad_norm=None` — canonical TD3 does not clip gradients; clipping is
  available as an explicit opt-in knob.

Known limitation: as with the other repo baselines there are no observation or
reward normalization wrappers, because `_make_env` and the evaluation protocol
are shared verbatim with the discrete-action baselines.
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

from model.device_utils import runtime_env_fingerprint
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datasets.rl_tasks import RLTaskSpec, get_rl_task_spec, list_supported_rl_tasks
from model.ppo_workflow import (
    ActionSpaceSpec,
    describe_action_space,
    resolve_task_action_kind,
)
from model.rl_workflow import (
    ReplayBuffer,
    _build_early_stopping_tracker,
    _build_training_efficiency,
    _finalize_early_stopping_tracker,
    _first_threshold_step,
    _json_default,
    _make_env,
    _require_gymnasium,
    _resolve_device_arg,
    _validate_early_stopping_config,
    seed_everything,
)

ALGORITHM_NAME = "td3_twin_delayed_ddpg"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TD3Config:
    """Training configuration for the repo-native TD3 baseline.

    See the module docstring for the rationale behind each default.

    `success_threshold` defaults to None: TD3 has no CartPole default to
    inherit, so the solve threshold and the immediate-stop target come from the
    task registry via `tools/run_rl_td3.py`.
    """

    profile: str = "fast"
    seed: int = 0
    total_timesteps: int = 150_000
    learning_rate: float = 3e-4
    policy_learning_rate: float | None = None
    buffer_size: int = 100_000
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    policy_frequency: int = 2
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    exploration_noise: float = 0.1
    learning_starts: int = 1_000
    train_frequency: int = 1
    max_grad_norm: float | None = None
    hidden_sizes: tuple[int, int] = (256, 256)
    eval_frequency: int = 5_000
    eval_episodes: int = 20
    eval_seed: int = 10_000
    success_threshold: float | None = None
    early_stopping: bool = False
    early_stopping_patience: int = 30
    early_stopping_min_delta: float = 1.0
    early_stopping_min_steps: int = 25_000
    early_stopping_target_score: float | None = None
    source_reference: str = (
        "TD3 (Fujimoto et al. 2018); CleanRL td3_continuous_action.py defaults as "
        "reference only"
    )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["hidden_sizes"] = list(self.hidden_sizes)
        return data

    @property
    def actor_learning_rate(self) -> float:
        """Learning rate used by the actor optimizer."""

        return float(self.learning_rate if self.policy_learning_rate is None else self.policy_learning_rate)


def _validate_td3_config(data: dict[str, Any]) -> None:
    buffer_size = int(data.get("buffer_size", 100_000))
    batch_size = int(data.get("batch_size", 256))
    policy_frequency = int(data.get("policy_frequency", 2))
    train_frequency = int(data.get("train_frequency", 1))
    learning_starts = int(data.get("learning_starts", 1_000))
    tau = float(data.get("tau", 0.005))
    policy_noise = float(data.get("policy_noise", 0.2))
    noise_clip = float(data.get("noise_clip", 0.5))
    exploration_noise = float(data.get("exploration_noise", 0.1))
    if buffer_size <= 0:
        raise ValueError("buffer_size must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if policy_frequency <= 0:
        raise ValueError("policy_frequency must be positive")
    if train_frequency <= 0:
        raise ValueError("train_frequency must be positive")
    if learning_starts < 0:
        raise ValueError("learning_starts must be non-negative")
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must lie in (0, 1]")
    if policy_noise < 0.0:
        raise ValueError("policy_noise must be non-negative")
    if noise_clip < 0.0:
        raise ValueError("noise_clip must be non-negative")
    if exploration_noise < 0.0:
        raise ValueError("exploration_noise must be non-negative")
    policy_learning_rate = data.get("policy_learning_rate", None)
    if policy_learning_rate is not None:
        policy_learning_rate = float(policy_learning_rate)
        if policy_learning_rate <= 0.0:
            raise ValueError("policy_learning_rate must be None or positive")
        data["policy_learning_rate"] = policy_learning_rate
    max_grad_norm = data.get("max_grad_norm", None)
    if max_grad_norm is not None:
        max_grad_norm = float(max_grad_norm)
        if max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be None or positive")
        data["max_grad_norm"] = max_grad_norm
    data["buffer_size"] = buffer_size
    data["batch_size"] = batch_size
    data["policy_frequency"] = policy_frequency
    data["train_frequency"] = train_frequency
    data["learning_starts"] = learning_starts
    data["tau"] = tau
    data["policy_noise"] = policy_noise
    data["noise_clip"] = noise_clip
    data["exploration_noise"] = exploration_noise


def make_td3_config(profile: str = "fast", **overrides: Any) -> TD3Config:
    """Build a `TD3Config` for a named profile, mirroring `make_dqn_config`."""

    profiles: dict[str, dict[str, Any]] = {
        "smoke": {
            "profile": "smoke",
            "total_timesteps": 256,
            "buffer_size": 1_000,
            "batch_size": 32,
            "learning_starts": 32,
            "train_frequency": 1,
            "policy_frequency": 2,
            "eval_frequency": 128,
            "eval_episodes": 2,
            "success_threshold": None,
            "early_stopping": False,
        },
        "debug": {
            "profile": "debug",
            "total_timesteps": 25_000,
            "buffer_size": 50_000,
            "batch_size": 256,
            "learning_starts": 1_000,
            "eval_frequency": 2_500,
            "eval_episodes": 20,
            "success_threshold": None,
            "early_stopping": True,
        },
        "fast": {
            "profile": "fast",
            "total_timesteps": 150_000,
            "buffer_size": 100_000,
            "batch_size": 256,
            "learning_starts": 1_000,
            "eval_frequency": 5_000,
            "eval_episodes": 20,
            "success_threshold": None,
            "early_stopping": True,
        },
        "gold": {
            "profile": "gold",
            "total_timesteps": 500_000,
            # Full-history replay at the gold budget; canonical TD3 uses 1e6
            # for 1M-step runs.
            "buffer_size": 500_000,
            "batch_size": 256,
            "learning_starts": 5_000,
            "eval_frequency": 10_000,
            "eval_episodes": 20,
            "success_threshold": None,
            "early_stopping": False,
        },
    }
    normalized = profile.strip().lower()
    if normalized not in profiles:
        raise ValueError(f"Unknown TD3 profile '{profile}'. Choose one of: {', '.join(sorted(profiles))}")
    data = {**profiles[normalized], **overrides}
    _validate_early_stopping_config(data)
    if "hidden_sizes" in data and not isinstance(data["hidden_sizes"], tuple):
        data["hidden_sizes"] = tuple(data["hidden_sizes"])
    _validate_td3_config(data)
    return TD3Config(**data)


def make_td3_output_dir(
    task_name: str,
    *,
    parent: str | Path = "results/rl",
    suffix: str | None = None,
) -> Path:
    created_at = datetime.now(timezone.utc)
    stem = f"td3_{task_name}_{created_at.strftime('%Y%m%d_%H%M%S_%f')}"
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
    raise FileExistsError(f"Could not create a unique TD3 output directory under {parent_path}")


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


@dataclass
class ContinuousReplayBuffer(ReplayBuffer):
    """Uniform replay for continuous actions.

    Subclasses the DQN buffer in `model/rl_workflow.py` and overrides only the
    action storage: the base class keeps one int64 action per transition, while
    TD3 needs a float32 action vector. Ring-buffer bookkeeping and the single
    `np.random.randint` draw per sample are inherited unchanged, so the two
    buffers consume the numpy RNG identically.
    """

    @classmethod
    def create(cls, buffer_size: int, obs_dim: int, action_dim: int = 1) -> "ContinuousReplayBuffer":
        return cls(
            observations=np.zeros((buffer_size, obs_dim), dtype=np.float32),
            next_observations=np.zeros((buffer_size, obs_dim), dtype=np.float32),
            actions=np.zeros((buffer_size, action_dim), dtype=np.float32),
            rewards=np.zeros(buffer_size, dtype=np.float32),
            dones=np.zeros(buffer_size, dtype=np.float32),
        )

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.observations[self.pos] = obs
        self.actions[self.pos] = np.asarray(action, dtype=np.float32).reshape(-1)
        self.rewards[self.pos] = reward
        self.next_observations[self.pos] = next_obs
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.observations.shape[0]
        self.size = min(self.size + 1, self.observations.shape[0])

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "observations": torch.as_tensor(self.observations[indices], device=device),
            "actions": torch.as_tensor(self.actions[indices], device=device).float(),
            "rewards": torch.as_tensor(self.rewards[indices], device=device),
            "next_observations": torch.as_tensor(self.next_observations[indices], device=device),
            "dones": torch.as_tensor(self.dones[indices], device=device),
        }


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


class TD3Actor(nn.Module):
    """Deterministic tanh policy rescaled onto the environment's Box bounds."""

    def __init__(
        self,
        obs_dim: int,
        action_spec: ActionSpaceSpec,
        *,
        hidden_sizes: tuple[int, int] = (256, 256),
    ):
        super().__init__()
        if action_spec.kind != "continuous":
            raise ValueError("TD3Actor requires a continuous action spec.")
        h1, h2 = hidden_sizes
        self.action_spec = action_spec
        self.network = nn.Sequential(
            nn.Linear(obs_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, action_spec.dim),
            nn.Tanh(),
        )
        low = torch.as_tensor(action_spec.low, dtype=torch.float32).view(1, -1)
        high = torch.as_tensor(action_spec.high, dtype=torch.float32).view(1, -1)
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs) * self.action_scale + self.action_bias


class TD3Critic(nn.Module):
    """State-action value network Q(s, a)."""

    def __init__(self, obs_dim: int, action_dim: int, *, hidden_sizes: tuple[int, int] = (256, 256)):
        super().__init__()
        h1, h2 = hidden_sizes
        self.network = nn.Sequential(
            nn.Linear(obs_dim + action_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([obs, action], dim=1)).view(-1)


class TD3Agent(nn.Module):
    """Deterministic actor plus the twin critics, in one module.

    Keeping all three networks under one module means checkpoint selection,
    device moves, and target-network copies are single calls, as they are for
    the single-network DQN and the actor-critic PPO agent.
    """

    def __init__(
        self,
        obs_dim: int,
        action_spec: ActionSpaceSpec,
        *,
        hidden_sizes: tuple[int, int] = (256, 256),
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_spec = action_spec
        self.hidden_sizes = tuple(hidden_sizes)
        self.actor = TD3Actor(self.obs_dim, action_spec, hidden_sizes=self.hidden_sizes)
        self.qf1 = TD3Critic(self.obs_dim, action_spec.dim, hidden_sizes=self.hidden_sizes)
        self.qf2 = TD3Critic(self.obs_dim, action_spec.dim, hidden_sizes=self.hidden_sizes)

    @property
    def is_continuous(self) -> bool:
        return True

    def critic_parameters(self) -> list[nn.Parameter]:
        return list(self.qf1.parameters()) + list(self.qf2.parameters())

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic (noise-free) policy action, already inside the bounds."""

        return self.actor(obs)

    def q_values(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.qf1(obs, action), self.qf2(obs, action)

    def clip_to_bounds(self, action: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            action.view(-1, self.action_spec.dim),
            self.actor.action_low,
            self.actor.action_high,
        )

    def env_action(self, action: torch.Tensor) -> np.ndarray:
        """Convert a policy action into something `env.step` accepts."""

        return self.clip_to_bounds(action).squeeze(0).detach().cpu().numpy().astype(np.float32)


def _soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    """Polyak-average `online` parameters into `target`.

    Only parameters are averaged. The actor's registered buffers are the Box
    bounds and their derived scale/bias, which are constants built identically
    in the online and target agents, so they need no update.
    """

    with torch.no_grad():
        for target_param, param in zip(target.parameters(), online.parameters()):
            target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)


# ---------------------------------------------------------------------------
# Environment plumbing
# ---------------------------------------------------------------------------


def continuous_rl_task_names() -> list[str]:
    """Registry task keys whose `action_kind` maps to a continuous head."""

    names: list[str] = []
    for family_names in list_supported_rl_tasks().values():
        for name in family_names:
            spec = get_rl_task_spec(name)
            try:
                kind = resolve_task_action_kind(spec.action_kind)
            except ValueError:
                continue
            if kind == "continuous":
                names.append(name)
    return sorted(names)


def require_continuous_task(spec: RLTaskSpec) -> None:
    """Fail fast, and legibly, when TD3 is pointed at a discrete-action task."""

    kind = resolve_task_action_kind(spec.action_kind)
    if kind != "continuous":
        supported = ", ".join(continuous_rl_task_names()) or "(none registered)"
        raise ValueError(
            f"TD3 is a continuous-control algorithm and cannot train on task "
            f"'{spec.name}' ({spec.env_id}), which declares action_kind="
            f"'{spec.action_kind}'. Continuous-action tasks: {supported}. Use "
            "tools/run_rl_dqn.py, tools/run_rl_nec.py, tools/run_rl_ppo.py, or "
            "tools/run_rl_nnknn.py for discrete-action tasks."
        )


def _validate_td3_env_spaces(env: Any, spec: RLTaskSpec) -> tuple[int, ActionSpaceSpec]:
    """Validate observation/action spaces and reconcile them with the task spec."""

    gym = _require_gymnasium()
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise ValueError(f"{spec.env_id} must use a Box observation space for this TD3 baseline.")
    obs_shape = env.observation_space.shape
    if len(obs_shape) != 1:
        raise ValueError(f"{spec.env_id} observation shape must be flat, got {obs_shape}.")
    if not isinstance(env.action_space, gym.spaces.Box):
        raise ValueError(
            f"TD3 requires a continuous Box action space, but {spec.env_id} exposes "
            f"{type(env.action_space).__name__}. Use the discrete-action baselines "
            "(tools/run_rl_dqn.py, tools/run_rl_nec.py, tools/run_rl_ppo.py, "
            "tools/run_rl_nnknn.py) for that task."
        )
    action_spec = describe_action_space(env.action_space)
    if action_spec.kind != "continuous":
        raise ValueError(f"{spec.env_id} did not resolve to a continuous action head for TD3.")
    require_continuous_task(spec)
    return int(np.prod(obs_shape)), action_spec


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


# ---------------------------------------------------------------------------
# Action selection and the TD3 update
# ---------------------------------------------------------------------------


def select_td3_action(
    agent: TD3Agent,
    obs: np.ndarray,
    *,
    exploration_noise: float,
    device: torch.device,
) -> np.ndarray:
    """Deterministic actor action plus scaled Gaussian noise, clipped to bounds."""

    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        action = agent.act(obs_tensor)
        if exploration_noise > 0.0:
            noise = torch.normal(
                torch.zeros_like(action),
                agent.actor.action_scale.expand_as(action) * float(exploration_noise),
            )
            action = action + noise
    return agent.env_action(action)


def td3_update(
    agent: TD3Agent,
    target_agent: TD3Agent,
    critic_optimizer: optim.Optimizer,
    actor_optimizer: optim.Optimizer,
    config: TD3Config,
    batch: dict[str, torch.Tensor],
    *,
    update_policy: bool,
) -> dict[str, Any]:
    """Run one TD3 critic step, and the delayed actor/target step when due.

    Target policy smoothing draws N(0, `policy_noise`), clips it to
    +/- `noise_clip`, scales it by the actor's `action_scale`, and clips the
    smoothed target action back into the Box bounds. `update_policy` implements
    the delayed policy update (`policy_frequency`); when it is True the actor
    takes one step and all three target networks are Polyak-averaged.
    """

    observations = batch["observations"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    next_observations = batch["next_observations"]
    dones = batch["dones"]

    with torch.no_grad():
        clipped_noise = (
            torch.randn_like(actions) * float(config.policy_noise)
        ).clamp(-float(config.noise_clip), float(config.noise_clip)) * target_agent.actor.action_scale
        next_actions = target_agent.clip_to_bounds(target_agent.act(next_observations) + clipped_noise)
        target_q1, target_q2 = target_agent.q_values(next_observations, next_actions)
        min_target_q = torch.min(target_q1, target_q2)
        td_target = rewards + float(config.gamma) * (1.0 - dones) * min_target_q

    q1_values, q2_values = agent.q_values(observations, actions)
    qf1_loss = F.mse_loss(q1_values, td_target)
    qf2_loss = F.mse_loss(q2_values, td_target)
    critic_loss = qf1_loss + qf2_loss

    critic_optimizer.zero_grad()
    critic_loss.backward()
    if config.max_grad_norm is not None:
        nn.utils.clip_grad_norm_(agent.critic_parameters(), float(config.max_grad_norm))
    critic_optimizer.step()

    actor_loss_value: float | None = None
    if update_policy:
        actor_loss = -agent.qf1(observations, agent.act(observations)).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        if config.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(agent.actor.parameters(), float(config.max_grad_norm))
        actor_optimizer.step()
        actor_loss_value = float(actor_loss.detach().cpu().item())
        tau = float(config.tau)
        _soft_update(target_agent.actor, agent.actor, tau)
        _soft_update(target_agent.qf1, agent.qf1, tau)
        _soft_update(target_agent.qf2, agent.qf2, tau)

    return {
        "critic_loss": float(critic_loss.detach().cpu().item()),
        "qf1_loss": float(qf1_loss.detach().cpu().item()),
        "qf2_loss": float(qf2_loss.detach().cpu().item()),
        "actor_loss": actor_loss_value,
        "mean_q1_value": float(q1_values.detach().mean().cpu().item()),
        "mean_q2_value": float(q2_values.detach().mean().cpu().item()),
        "mean_td_target": float(td_target.detach().mean().cpu().item()),
        "policy_updated": bool(update_policy),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_td3(
    task_name: str,
    agent: TD3Agent,
    *,
    episodes: int = 20,
    seed: int = 10_000,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Run the shared evaluation protocol with the noise-free deterministic actor."""

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
                    action = agent.act(obs_tensor)
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


def train_td3(
    task_name: str = "pendulum",
    config: TD3Config | None = None,
    *,
    output_dir: str | Path | None = None,
    device: str | torch.device | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Train a TD3 baseline and write the shared run artifacts."""

    spec = get_rl_task_spec(task_name)
    require_continuous_task(spec)
    cfg = config or make_td3_config(spec.default_profile)
    seed_everything(cfg.seed)
    run_device = _resolve_device_arg(device)

    env = _make_env(spec, seed=cfg.seed)
    obs_dim, action_spec = _validate_td3_env_spaces(env, spec)
    agent = TD3Agent(obs_dim, action_spec, hidden_sizes=cfg.hidden_sizes).to(run_device)
    target_agent = TD3Agent(obs_dim, action_spec, hidden_sizes=cfg.hidden_sizes).to(run_device)
    target_agent.load_state_dict(agent.state_dict())
    critic_optimizer = optim.Adam(agent.critic_parameters(), lr=cfg.learning_rate)
    actor_optimizer = optim.Adam(agent.actor.parameters(), lr=cfg.actor_learning_rate)
    replay_buffer = ContinuousReplayBuffer.create(cfg.buffer_size, obs_dim, action_spec.dim)

    run_dir = Path(output_dir) if output_dir is not None else make_td3_output_dir(spec.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint.pt"
    created_at = datetime.now(timezone.utc)

    training_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    latest_update: dict[str, Any] | None = None
    latest_actor_loss: float | None = None
    best_eval: dict[str, Any] | None = None
    best_eval_step: int | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    best_target_state: dict[str, torch.Tensor] | None = None
    early_stopping = _build_early_stopping_tracker(cfg, spec)
    actual_timesteps = 0
    critic_updates = 0
    policy_updates = 0

    obs, _ = env.reset(seed=cfg.seed)
    episode_return = 0.0
    episode_length = 0
    episode_index = 0

    try:
        for global_step in range(cfg.total_timesteps):
            completed_step = global_step + 1
            if global_step < cfg.learning_starts:
                action = np.asarray(env.action_space.sample(), dtype=np.float32)
            else:
                action = select_td3_action(
                    agent,
                    np.asarray(obs, dtype=np.float32),
                    exploration_noise=cfg.exploration_noise,
                    device=run_device,
                )
            next_obs, reward, terminated, truncated, _ = env.step(action)
            train_done = bool(terminated)
            episode_done = bool(terminated or truncated)
            replay_buffer.add(
                np.asarray(obs, dtype=np.float32),
                action,
                float(reward),
                np.asarray(next_obs, dtype=np.float32),
                train_done,
            )
            obs = next_obs
            episode_return += float(reward)
            episode_length += 1
            actual_timesteps = completed_step

            if replay_buffer.size >= cfg.learning_starts and global_step % cfg.train_frequency == 0:
                batch = replay_buffer.sample(cfg.batch_size, run_device)
                agent.train()
                latest_update = td3_update(
                    agent,
                    target_agent,
                    critic_optimizer,
                    actor_optimizer,
                    cfg,
                    batch,
                    update_policy=(critic_updates % cfg.policy_frequency == 0),
                )
                critic_updates += 1
                if latest_update["policy_updated"]:
                    policy_updates += 1
                    latest_actor_loss = latest_update["actor_loss"]
                loss_rows.append(
                    {
                        "global_step": global_step,
                        "critic_updates": critic_updates,
                        "policy_updates": policy_updates,
                        **latest_update,
                    }
                )

            if episode_done:
                episode_index += 1
                training_rows.append(
                    {
                        "global_step": completed_step,
                        "episode": episode_index,
                        "episode_return": episode_return,
                        "episode_length": episode_length,
                        "exploration_noise": cfg.exploration_noise,
                        "critic_loss": None if latest_update is None else latest_update["critic_loss"],
                        "actor_loss": latest_actor_loss,
                        "mean_q1_value": None if latest_update is None else latest_update["mean_q1_value"],
                    }
                )
                if progress and (episode_index <= 5 or episode_index % 10 == 0):
                    print(
                        "[td3] "
                        f"step={completed_step} episode={episode_index} "
                        f"return={episode_return:.1f} length={episode_length} "
                        f"critic_loss={None if latest_update is None else round(latest_update['critic_loss'], 4)} "
                        f"actor_loss={None if latest_actor_loss is None else round(latest_actor_loss, 4)}",
                        flush=True,
                    )
                obs, _ = env.reset(seed=cfg.seed + episode_index)
                episode_return = 0.0
                episode_length = 0

            if cfg.eval_frequency > 0 and completed_step % cfg.eval_frequency == 0:
                eval_metrics = evaluate_td3(
                    spec.name,
                    agent,
                    episodes=cfg.eval_episodes,
                    seed=cfg.eval_seed,
                    device=run_device,
                )
                eval_row = {
                    "global_step": completed_step,
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
                    best_target_state = _copy_state_dict_to_cpu(target_agent)
                if progress:
                    print(
                        "[td3][eval] "
                        f"step={completed_step} mean_return={eval_row['mean_return']:.2f} "
                        f"max_return={eval_row['max_return']:.2f}",
                        flush=True,
                    )
                agent.train()
                if early_stopping.update(eval_metrics["mean_return"], completed_step):
                    if progress:
                        print(
                            f"[td3][early-stop] step={completed_step} reason={early_stopping.stopping_reason}",
                            flush=True,
                        )
                    break
    finally:
        env.close()

    _finalize_early_stopping_tracker(early_stopping, actual_timesteps)

    last_eval = evaluate_td3(
        spec.name,
        agent,
        episodes=cfg.eval_episodes,
        seed=cfg.eval_seed,
        device=run_device,
    )
    if best_eval is None or last_eval["mean_return"] >= best_eval["mean_return"]:
        selected_eval = last_eval
        selected_step = actual_timesteps
        selected_source = "final"
        selected_model_state = _copy_state_dict_to_cpu(agent)
        selected_target_state = _copy_state_dict_to_cpu(target_agent)
    else:
        selected_eval = best_eval
        selected_step = int(best_eval_step or 0)
        selected_source = "best_eval"
        selected_model_state = best_model_state or _copy_state_dict_to_cpu(agent)
        selected_target_state = best_target_state or _copy_state_dict_to_cpu(target_agent)
        agent.load_state_dict(selected_model_state)
        target_agent.load_state_dict(selected_target_state)

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
        "target_model_state_dict": selected_target_state,
        "task": spec.to_dict(),
        "config": cfg.to_dict(),
        "obs_dim": obs_dim,
        "action_spec": action_spec.to_dict(),
        "action_dim": action_spec.dim,
        "selected_eval": {k: v for k, v in selected_eval.items() if k != "episode_metrics"},
        "last_eval": {k: v for k, v in last_eval.items() if k != "episode_metrics"},
        "selected_step": selected_step,
        "selected_source": selected_source,
        "training_efficiency": training_efficiency,
        "configured_total_timesteps": cfg.total_timesteps,
        "actual_timesteps": actual_timesteps,
        "critic_updates": critic_updates,
        "policy_updates": policy_updates,
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
            "source_reference": cfg.source_reference,
            "runtime": runtime_env_fingerprint(),
            "td3": {
                "twin_critics": True,
                "policy_frequency": cfg.policy_frequency,
                "target_policy_noise": cfg.policy_noise,
                "target_noise_clip": cfg.noise_clip,
                "exploration_noise": cfg.exploration_noise,
                "tau": cfg.tau,
                "gamma": cfg.gamma,
                "buffer_size": cfg.buffer_size,
                "batch_size": cfg.batch_size,
                "learning_starts": cfg.learning_starts,
                "critic_learning_rate": cfg.learning_rate,
                "actor_learning_rate": cfg.actor_learning_rate,
                "noise_units": "normalized_action_units_scaled_by_action_scale",
                "policy_head": "deterministic_tanh_scaled_to_action_bounds",
                "truncation_bootstrap": True,
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
        "critic_updates": critic_updates,
        "policy_updates": policy_updates,
        "action_kind": action_spec.kind,
        "action_dim": action_spec.dim,
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
            "[td3] finished "
            f"profile={cfg.profile} mean_eval_return={selected_eval['mean_return']:.2f} "
            f"passed={passed} run_dir={run_dir}",
            flush=True,
        )
    return {
        "model": agent,
        "agent": agent,
        "target_model": target_agent,
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


def load_td3_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    run_device = _resolve_device_arg(device)
    checkpoint = torch.load(checkpoint_path, map_location=run_device, weights_only=False)
    recorded_algorithm = checkpoint.get("algorithm")
    if recorded_algorithm != ALGORITHM_NAME:
        raise ValueError(
            f"Checkpoint algorithm {recorded_algorithm!r} is not a TD3 checkpoint. "
            f"Expected {ALGORITHM_NAME!r}."
        )
    config_data = dict(checkpoint["config"])
    config_data["hidden_sizes"] = tuple(config_data["hidden_sizes"])
    cfg = TD3Config(**config_data)
    action_spec = ActionSpaceSpec.from_dict(checkpoint["action_spec"])
    agent = TD3Agent(
        int(checkpoint["obs_dim"]),
        action_spec,
        hidden_sizes=cfg.hidden_sizes,
    ).to(run_device)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()
    target_agent = TD3Agent(
        int(checkpoint["obs_dim"]),
        action_spec,
        hidden_sizes=cfg.hidden_sizes,
    ).to(run_device)
    target_agent.load_state_dict(checkpoint["target_model_state_dict"])
    target_agent.eval()
    return {
        "model": agent,
        "agent": agent,
        "target_model": target_agent,
        "config": cfg,
        "action_spec": action_spec,
        "task": checkpoint["task"],
        "checkpoint": checkpoint,
        "device": run_device,
    }
