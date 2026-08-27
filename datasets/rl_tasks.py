from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RLTaskSpec:
    """Metadata for a reinforcement-learning task supported by the repo."""

    name: str
    env_id: str
    family: str
    observation_kind: str
    action_kind: str
    max_episode_steps: int
    default_profile: str
    literature_notes: tuple[str, ...]
    # Task-level success protocol. `success_threshold` marks a run as solved;
    # `target_mean_return` is the mean return treated as task-maximum for
    # immediate early stopping. None keeps the workflow-config defaults
    # (CartPole's 475/max_episode_steps behavior).
    success_threshold: float | None = None
    target_mean_return: float | None = None
    # Extra keyword arguments forwarded to `gymnasium.make` by the shared
    # `model.rl_workflow._make_env`, which every RL workflow uses. Stored as a
    # tuple of key/value pairs (not a dict) so the spec stays frozen and
    # hashable, matching `literature_notes`. Empty for every task that takes no
    # constructor arguments, which keeps `gym.make(env_id)` byte-for-byte the
    # call it was before this field existed.
    env_kwargs: tuple[tuple[str, Any], ...] = ()

    def env_kwargs_dict(self) -> dict[str, Any]:
        """Return `env_kwargs` as the mapping `gymnasium.make` expects."""

        return dict(self.env_kwargs)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["literature_notes"] = list(self.literature_notes)
        data["env_kwargs"] = self.env_kwargs_dict()
        return data


_RL_TASKS: dict[str, RLTaskSpec] = {
    "cartpole": RLTaskSpec(
        name="cartpole",
        env_id="CartPole-v1",
        family="classic_control",
        observation_kind="flat_box",
        action_kind="discrete",
        max_episode_steps=500,
        default_profile="fast",
        literature_notes=(
            "Fast DQN sanity task used by the official PyTorch DQN tutorial.",
            "Engineering gate before moving to Atari/ALE, the DQN and NEC paper-aligned benchmark family.",
        ),
        success_threshold=475.0,
        target_mean_return=500.0,
    ),
    "acrobot": RLTaskSpec(
        name="acrobot",
        env_id="Acrobot-v1",
        family="classic_control",
        observation_kind="flat_box",
        action_kind="discrete",
        max_episode_steps=500,
        default_profile="fast",
        literature_notes=(
            "Sparse-ish negative-reward swing-up task; gymnasium reward threshold is -100.",
            "Returns are -1 per step until the goal, so success_threshold=-100 and the",
            "immediate-stop target -60 approximates the physical best-case swing-up.",
        ),
        success_threshold=-100.0,
        target_mean_return=-60.0,
    ),
    "lunarlander": RLTaskSpec(
        name="lunarlander",
        env_id="LunarLander-v3",
        family="box2d",
        observation_kind="flat_box",
        action_kind="discrete",
        max_episode_steps=1000,
        default_profile="fast",
        literature_notes=(
            "Shaped-reward landing task; gymnasium reward threshold is 200.",
            "Requires gymnasium[box2d]. The immediate-stop target 280 sits above",
            "the solve threshold so only an excellent policy stops a run early.",
        ),
        success_threshold=200.0,
        target_mean_return=280.0,
    ),
    "minatar_breakout": RLTaskSpec(
        name="minatar_breakout",
        env_id="MinAtar/Breakout-v1",
        family="minatar",
        observation_kind="flat_box",
        action_kind="discrete",
        max_episode_steps=5000,
        default_profile="fast",
        literature_notes=(
            "MinAtar Breakout with the minimal action set; 10x10x4 binary planes",
            "flattened to a 400-dim vector by the shared env wrapper.",
            "MinAtar defines no canonical solve threshold: success_threshold=10 is",
            "a non-canonical progress marker (MinAtar-paper DQN reaches ~15-25 at",
            "millions of frames); target_mean_return=50 keeps immediate stopping",
            "out of reach so patience-based early stopping governs.",
        ),
        success_threshold=10.0,
        target_mean_return=50.0,
    ),
    # ---------------------------------------------------------------------
    # Continuous-action tasks (NEXT_STEPS_PLAN.md I-2).
    #
    # SCOPE NOTE: the two entries below exist for the TD3 continuous-control
    # reference baseline (`model/td3_workflow.py`), compared against PPO's
    # diagonal-Gaussian head. The NN-kNN actor is DISCRETE-ONLY — it stores
    # one-hot action labels per case and normalizes case activations over a
    # fixed action set — so NO NN-kNN-RL run is expected (or possible) on
    # `pendulum` or `lunarlander_continuous`, and `tools/run_rl_nnknn.py`
    # rejects them at env validation. A continuous-action NN-kNN actor is
    # flagged future work, not attempted here. The same holds for DQN and NEC,
    # which are value-based discrete-action baselines.
    # ---------------------------------------------------------------------
    "pendulum": RLTaskSpec(
        name="pendulum",
        env_id="Pendulum-v1",
        family="classic_control",
        observation_kind="flat_box",
        action_kind="continuous",
        max_episode_steps=200,
        default_profile="fast",
        literature_notes=(
            "Classic torque-limited inverted-pendulum swing-up; 3-dim observation",
            "(cos, sin, angular velocity) and a 1-dim continuous torque in [-2, 2].",
            "Every episode runs the full 200-step cap (no termination), so returns are",
            "a dense negative cost and higher (less negative) is better.",
            "NON-CANONICAL MARKERS: gymnasium defines NO reward threshold for",
            "Pendulum-v1 (its registration sets reward_threshold=None), so neither",
            "number below is a canonical solve criterion. success_threshold=-200.0 is",
            "a repo progress marker for 'swings up and holds'; target_mean_return=-140.0",
            "sits above published tuned continuous-control results (SB3 rl-zoo reports",
            "roughly -150 for TD3/SAC) so immediate early stopping stays out of reach",
            "and patience-based stopping governs, matching the lunarlander convention.",
        ),
        success_threshold=-200.0,
        target_mean_return=-140.0,
    ),
    "lunarlander_continuous": RLTaskSpec(
        name="lunarlander_continuous",
        env_id="LunarLander-v3",
        family="box2d",
        observation_kind="flat_box",
        action_kind="continuous",
        max_episode_steps=1000,
        default_profile="fast",
        literature_notes=(
            "Continuous-control variant of the `lunarlander` task, selected with the",
            "env_kwargs continuous=True constructor argument; same 8-dim observation",
            "and shaped reward, but a 2-dim continuous action (main engine, lateral",
            "engines) in [-1, 1] instead of 4 discrete thrusters.",
            "Gymnasium's reward threshold is 200 for this env id, so success_threshold",
            "is canonical here. The immediate-stop target 280 mirrors the discrete",
            "`lunarlander` entry so the two rows share one success protocol.",
            "Requires gymnasium[box2d].",
        ),
        success_threshold=200.0,
        target_mean_return=280.0,
        env_kwargs=(("continuous", True),),
    ),
}


def normalize_rl_task_name(task_name: str) -> str:
    normalized = task_name.strip().lower().replace("-", "_")
    aliases = {
        "cart_pole": "cartpole",
        "cartpole_v1": "cartpole",
        "cartpole-v1": "cartpole",
        "cartpolev1": "cartpole",
        "acrobot_v1": "acrobot",
        "acrobotv1": "acrobot",
        "lunar_lander": "lunarlander",
        "lunarlander_v3": "lunarlander",
        "lunarlanderv3": "lunarlander",
        "pendulum_v1": "pendulum",
        "pendulumv1": "pendulum",
        "lunar_lander_continuous": "lunarlander_continuous",
        "lunarlandercontinuous": "lunarlander_continuous",
        "lunarlandercontinuous_v3": "lunarlander_continuous",
        "lunarlander_continuous_v3": "lunarlander_continuous",
    }
    return aliases.get(normalized, normalized)


def get_rl_task_spec(task_name: str) -> RLTaskSpec:
    normalized = normalize_rl_task_name(task_name)
    try:
        return _RL_TASKS[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_RL_TASKS))
        raise ValueError(f"Unknown RL task '{task_name}'. Supported tasks: {supported}") from exc


def list_supported_rl_tasks() -> dict[str, list[str]]:
    by_family: dict[str, list[str]] = {}
    for task in _RL_TASKS.values():
        by_family.setdefault(task.family, []).append(task.name)
    return {family: sorted(names) for family, names in sorted(by_family.items())}
