from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


# Supported `RLTaskSpec.observation_kind` values.
#
# - "flat_box": a 1-D Box observation (or a grid that the shared `_make_env`
#   flattens into one, as MinAtar does). Every workflow feeds these to an MLP.
# - "image_atari": a stacked-grayscale ALE image observation. `_make_env`
#   builds the base env with frameskip=1 and applies AtariPreprocessing
#   (grayscale, 84x84, frame_skip=4) followed by FrameStackObservation(4), so
#   the workflows see a (4, 84, 84) uint8 Box and build a Nature-CNN encoder
#   (`model/cnn_encoders.py`) instead of an MLP.
FLAT_BOX_OBSERVATION = "flat_box"
IMAGE_ATARI_OBSERVATION = "image_atari"
SUPPORTED_OBSERVATION_KINDS = (FLAT_BOX_OBSERVATION, IMAGE_ATARI_OBSERVATION)


def is_image_observation_kind(observation_kind: str) -> bool:
    """Return True when `observation_kind` requires the CNN observation path."""

    return str(observation_kind).strip().lower() == IMAGE_ATARI_OBSERVATION


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

    @property
    def is_image_observation(self) -> bool:
        """True when the task needs the CNN (ALE image) observation path."""

        return is_image_observation_kind(self.observation_kind)

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
    # Remaining MinAtar games (NEXT_STEPS_PLAN.md I-3 / HANDOFF item 4).
    #
    # SCOPE NOTE: MinAtar defines NO canonical solve threshold for any of its
    # five games, so every `success_threshold` below is a repo-internal
    # NON-CANONICAL progress marker and every `target_mean_return` is set out
    # of realistic reach so patience-based early stopping governs instead of
    # immediate target-score stopping. Reference points come from the MinAtar
    # paper (Young & Tian, 2019) and its DQN/AC baselines at millions of
    # frames; do not report these numbers as "solved" criteria. All five games
    # expose 10x10xC binary planes that the shared `_make_env` flattens to a
    # 100*C vector, so they run on the flat MLP path like `minatar_breakout`.
    # ---------------------------------------------------------------------
    "minatar_asterix": RLTaskSpec(
        name="minatar_asterix",
        env_id="MinAtar/Asterix-v1",
        family="minatar",
        observation_kind="flat_box",
        action_kind="discrete",
        max_episode_steps=5000,
        default_profile="fast",
        literature_notes=(
            "MinAtar Asterix with the minimal action set; 10x10x4 binary planes",
            "flattened to a 400-dim vector by the shared env wrapper (5 actions).",
            "Dodge-and-collect game: reward +1 per treasure, episode ends on an",
            "enemy collision, and spawn rate ramps up over time.",
            "NON-CANONICAL: MinAtar publishes no solve threshold.",
            "success_threshold=10 is a repo progress marker (MinAtar-paper DQN",
            "reaches roughly 15-20 at millions of frames); target_mean_return=50",
            "is deliberately out of reach so patience-based early stopping governs.",
        ),
        success_threshold=10.0,
        target_mean_return=50.0,
    ),
    "minatar_freeway": RLTaskSpec(
        name="minatar_freeway",
        env_id="MinAtar/Freeway-v1",
        family="minatar",
        observation_kind="flat_box",
        action_kind="discrete",
        max_episode_steps=5000,
        default_profile="fast",
        literature_notes=(
            "MinAtar Freeway with the minimal action set; 10x10x7 binary planes",
            "flattened to a 700-dim vector by the shared env wrapper (3 actions).",
            "Fixed-length crossing game: reward +1 per successful crossing and the",
            "episode always runs to the internal 2500-frame timer, so returns are",
            "dense and bounded rather than survival-driven.",
            "NON-CANONICAL: MinAtar publishes no solve threshold.",
            "success_threshold=25 is a repo progress marker (MinAtar-paper DQN",
            "reaches roughly 50-60 at millions of frames); target_mean_return=60",
            "is deliberately out of reach so patience-based early stopping governs.",
        ),
        success_threshold=25.0,
        target_mean_return=60.0,
    ),
    "minatar_seaquest": RLTaskSpec(
        name="minatar_seaquest",
        env_id="MinAtar/Seaquest-v1",
        family="minatar",
        observation_kind="flat_box",
        action_kind="discrete",
        max_episode_steps=5000,
        default_profile="fast",
        literature_notes=(
            "MinAtar Seaquest with the minimal action set; 10x10x10 binary planes",
            "flattened to a 1000-dim vector by the shared env wrapper (6 actions).",
            "Hardest MinAtar game: oxygen management plus diver rescue plus",
            "shooting, so credit assignment is long and returns stay low for a",
            "long time.",
            "NON-CANONICAL: MinAtar publishes no solve threshold.",
            "success_threshold=5 is a repo progress marker (MinAtar-paper DQN",
            "reaches roughly 5-15 at millions of frames); target_mean_return=50",
            "is deliberately out of reach so patience-based early stopping governs.",
        ),
        success_threshold=5.0,
        target_mean_return=50.0,
    ),
    "minatar_space_invaders": RLTaskSpec(
        name="minatar_space_invaders",
        env_id="MinAtar/SpaceInvaders-v1",
        family="minatar",
        observation_kind="flat_box",
        action_kind="discrete",
        max_episode_steps=5000,
        default_profile="fast",
        literature_notes=(
            "MinAtar SpaceInvaders with the minimal action set; 10x10x6 binary",
            "planes flattened to a 600-dim vector by the shared env wrapper",
            "(4 actions).",
            "Reward +1 per alien destroyed; waves respawn faster after each clear,",
            "so returns are unbounded in principle and grow with policy quality.",
            "NON-CANONICAL: MinAtar publishes no solve threshold.",
            "success_threshold=20 is a repo progress marker (MinAtar-paper DQN",
            "reaches roughly 50-90 at millions of frames); target_mean_return=100",
            "is deliberately out of reach so patience-based early stopping governs.",
        ),
        success_threshold=20.0,
        target_mean_return=100.0,
    ),
    # ---------------------------------------------------------------------
    # ALE / Atari image-observation tasks (NEXT_STEPS_PLAN.md I-3).
    #
    # SCOPE NOTE: these are the repo's only `observation_kind="image_atari"`
    # tasks. The shared `model/rl_workflow.py::_make_env` builds the base env
    # with frameskip=1 and then applies
    #   AtariPreprocessing(noop_max=30, frame_skip=4, screen_size=84,
    #                      grayscale_obs=True, scale_obs=False)
    #   FrameStackObservation(stack_size=4)
    # yielding a (4, 84, 84) uint8 Box. Every workflow then swaps its MLP for
    # the shared Nature-CNN encoder in `model/cnn_encoders.py`.
    #
    # TIMESTEP UNITS: one environment step in this repo's budgets, logs and
    # `total_timesteps` is one POST-FRAMESKIP agent step = 4 emulator frames.
    # A 1,000,000-step run is therefore a 4,000,000-frame run. Compare against
    # published Atari numbers in frames, not steps.
    #
    # ALE v5 defaults are kept: repeat_action_probability=0.25 (sticky
    # actions) and the minimal per-game action set. That is the modern
    # Machado et al. (2018) protocol and it is HARDER than the deterministic
    # NoFrameskip-v4 setting most published Nature-DQN numbers use, so scores
    # here are not directly comparable to those papers.
    # ---------------------------------------------------------------------
    "ale_pong": RLTaskSpec(
        name="ale_pong",
        env_id="ALE/Pong-v5",
        family="ale",
        observation_kind="image_atari",
        action_kind="discrete",
        max_episode_steps=27_000,
        default_profile="fast",
        literature_notes=(
            "Atari 2600 Pong through ale-py; 6 minimal actions, returns in",
            "[-21, +21] (points scored minus points conceded in a first-to-21 game).",
            "Observation is a 4-frame stack of 84x84 grayscale frames after",
            "AtariPreprocessing(frame_skip=4) on a frameskip=1 base env.",
            "max_episode_steps=27000 post-frameskip steps mirrors the ALE default",
            "cap of 108000 emulator frames.",
            "NON-CANONICAL MARKERS: gymnasium/ALE define no reward threshold for",
            "ALE/Pong-v5. success_threshold=0.0 is a repo progress marker meaning",
            "'breaks even against the built-in opponent'; target_mean_return=18.0",
            "approximates Nature-DQN-level play. Nature DQN only reaches roughly",
            "+18 on Pong after 10M+ AGENT STEPS (40M+ frames), far beyond the",
            "budgets this repo runs, so ALE runs here are declared EXPLORATORY:",
            "the deliverable is a partial learning curve, not a solved game.",
            "TIMESTEPS COUNT POST-FRAMESKIP STEPS: 1 step = 4 emulator frames.",
        ),
        success_threshold=0.0,
        target_mean_return=18.0,
    ),
    "ale_breakout": RLTaskSpec(
        name="ale_breakout",
        env_id="ALE/Breakout-v5",
        family="ale",
        observation_kind="image_atari",
        action_kind="discrete",
        max_episode_steps=27_000,
        default_profile="fast",
        literature_notes=(
            "Atari 2600 Breakout through ale-py; 4 minimal actions (NOOP, FIRE,",
            "RIGHT, LEFT), reward is bricks destroyed, and the ball must be",
            "launched with FIRE, so a policy that never fires idles until the",
            "27000-step cap. Keep eval episode counts small.",
            "Observation is a 4-frame stack of 84x84 grayscale frames after",
            "AtariPreprocessing(frame_skip=4) on a frameskip=1 base env.",
            "NON-CANONICAL MARKERS: gymnasium/ALE define no reward threshold for",
            "ALE/Breakout-v5. success_threshold=20.0 is a repo progress marker",
            "('clears a meaningful part of the first wall') and",
            "target_mean_return=100.0 is an out-of-reach immediate-stop guard.",
            "Nature DQN reaches roughly 400 on Breakout only after 10M+ AGENT",
            "STEPS (40M+ frames), so ALE runs here are declared EXPLORATORY:",
            "the deliverable is a partial learning curve, not a solved game.",
            "TIMESTEPS COUNT POST-FRAMESKIP STEPS: 1 step = 4 emulator frames.",
        ),
        success_threshold=20.0,
        target_mean_return=100.0,
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
        "minatar_spaceinvaders": "minatar_space_invaders",
        "minatar/asterix": "minatar_asterix",
        "minatar/asterix_v1": "minatar_asterix",
        "minatar/freeway": "minatar_freeway",
        "minatar/freeway_v1": "minatar_freeway",
        "minatar/seaquest": "minatar_seaquest",
        "minatar/seaquest_v1": "minatar_seaquest",
        "minatar/spaceinvaders": "minatar_space_invaders",
        "minatar/spaceinvaders_v1": "minatar_space_invaders",
        "minatar/breakout": "minatar_breakout",
        "minatar/breakout_v1": "minatar_breakout",
        "pong": "ale_pong",
        "ale/pong": "ale_pong",
        "ale/pong_v5": "ale_pong",
        "ale_pong_v5": "ale_pong",
        "breakout": "ale_breakout",
        "ale/breakout": "ale_breakout",
        "ale/breakout_v5": "ale_breakout",
        "ale_breakout_v5": "ale_breakout",
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
