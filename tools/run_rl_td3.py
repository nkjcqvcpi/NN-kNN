from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets.rl_tasks import get_rl_task_spec  # noqa: E402
from model.td3_workflow import (  # noqa: E402
    TD3Config,
    continuous_rl_task_names,
    evaluate_td3,
    load_td3_checkpoint,
    make_td3_config,
    make_td3_output_dir,
    require_continuous_task,
    train_td3,
)


def _optional_float_arg(value: str) -> float | None:
    normalized = value.strip().lower()
    if normalized in {"none", "null"}:
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train or evaluate the repo-native TD3 baseline. TD3 is a continuous-control "
            "reference baseline: it only accepts continuous-action tasks "
            f"({', '.join(continuous_rl_task_names())})."
        )
    )
    parser.add_argument("task", nargs="?", default="pendulum", help="Continuous RL task key, such as pendulum.")
    parser.add_argument("--profile", choices=["smoke", "debug", "fast", "gold"], default="fast")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/rl", help="Parent folder for timestamped run dirs.")
    parser.add_argument("--device", default=None, help="Optional torch device override, such as cpu or cuda.")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None, help="Critic (and default actor) learning rate.")
    parser.add_argument(
        "--policy-learning-rate",
        type=_optional_float_arg,
        default=argparse.SUPPRESS,
        help="Actor learning rate; use none to follow --learning-rate.",
    )
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None, help="Polyak averaging coefficient for target networks.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--policy-frequency",
        type=int,
        default=None,
        help="Delayed policy update period, in critic updates (TD3 default 2).",
    )
    parser.add_argument(
        "--policy-noise",
        type=float,
        default=None,
        help="Target policy smoothing noise std, in normalized action units (TD3 default 0.2).",
    )
    parser.add_argument(
        "--noise-clip",
        type=float,
        default=None,
        help="Clip applied to the target smoothing noise, in normalized action units (TD3 default 0.5).",
    )
    parser.add_argument(
        "--exploration-noise",
        type=float,
        default=None,
        help="Behaviour-policy Gaussian noise std, in normalized action units (TD3 default 0.1).",
    )
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--train-frequency", type=int, default=None)
    parser.add_argument(
        "--max-grad-norm",
        type=_optional_float_arg,
        default=argparse.SUPPRESS,
        help="Optional gradient-norm clip; use none for canonical (unclipped) TD3.",
    )
    parser.add_argument("--eval-frequency", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=None)
    early_stopping_group = parser.add_mutually_exclusive_group()
    early_stopping_group.add_argument(
        "--early-stopping",
        dest="early_stopping",
        action="store_true",
        help="Enable evaluation-based early stopping.",
    )
    early_stopping_group.add_argument(
        "--no-early-stopping",
        dest="early_stopping",
        action="store_false",
        help="Run through the full configured timestep budget.",
    )
    parser.set_defaults(early_stopping=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=None)
    parser.add_argument("--early-stopping-min-steps", type=int, default=None)
    parser.add_argument("--early-stopping-target-score", type=float, default=None)
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=None,
        help="Mean-return solve threshold; defaults to the task spec.",
    )
    parser.add_argument("--eval-only", action="store_true", help="Evaluate a saved checkpoint without training.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path for --eval-only.")
    parser.add_argument("--quiet", action="store_true", help="Disable progress logging during training.")
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> TD3Config:
    overrides = {"seed": args.seed}
    if args.total_timesteps is not None:
        overrides["total_timesteps"] = args.total_timesteps
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if hasattr(args, "policy_learning_rate"):
        overrides["policy_learning_rate"] = args.policy_learning_rate
    if args.buffer_size is not None:
        overrides["buffer_size"] = args.buffer_size
    if args.gamma is not None:
        overrides["gamma"] = args.gamma
    if args.tau is not None:
        overrides["tau"] = args.tau
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.policy_frequency is not None:
        overrides["policy_frequency"] = args.policy_frequency
    if args.policy_noise is not None:
        overrides["policy_noise"] = args.policy_noise
    if args.noise_clip is not None:
        overrides["noise_clip"] = args.noise_clip
    if args.exploration_noise is not None:
        overrides["exploration_noise"] = args.exploration_noise
    if args.learning_starts is not None:
        overrides["learning_starts"] = args.learning_starts
    if args.train_frequency is not None:
        overrides["train_frequency"] = args.train_frequency
    if hasattr(args, "max_grad_norm"):
        overrides["max_grad_norm"] = args.max_grad_norm
    if args.eval_frequency is not None:
        overrides["eval_frequency"] = args.eval_frequency
    if args.eval_episodes is not None:
        overrides["eval_episodes"] = args.eval_episodes
    if args.eval_seed is not None:
        overrides["eval_seed"] = args.eval_seed
    if args.early_stopping is not None:
        overrides["early_stopping"] = args.early_stopping
    if args.early_stopping_patience is not None:
        overrides["early_stopping_patience"] = args.early_stopping_patience
    if args.early_stopping_min_delta is not None:
        overrides["early_stopping_min_delta"] = args.early_stopping_min_delta
    if args.early_stopping_min_steps is not None:
        overrides["early_stopping_min_steps"] = args.early_stopping_min_steps
    if args.early_stopping_target_score is not None:
        overrides["early_stopping_target_score"] = args.early_stopping_target_score
    _apply_task_success_defaults(args, overrides)
    return make_td3_config(args.profile, **overrides)


def _apply_task_success_defaults(args: argparse.Namespace, overrides: dict) -> None:
    """Resolve the task-level success protocol.

    TD3 has no CartPole profile defaults to preserve (CartPole is discrete and
    is rejected outright), so every task reads its solve threshold and its
    immediate-stop target from the task spec unless the command line overrides
    them. The `spec.name != "cartpole"` guard is kept so this helper stays
    identical in shape to the other runners'.
    """
    spec = get_rl_task_spec(args.task)
    if args.success_threshold is not None:
        overrides["success_threshold"] = args.success_threshold
    elif spec.name != "cartpole" and spec.success_threshold is not None:
        overrides["success_threshold"] = spec.success_threshold
    if (
        args.early_stopping_target_score is None
        and spec.name != "cartpole"
        and spec.target_mean_return is not None
    ):
        overrides["early_stopping_target_score"] = spec.target_mean_return


def _write_eval_only_summary(outdir: Path, payload: dict) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "eval_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.eval_only:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required with --eval-only")
        loaded = load_td3_checkpoint(args.checkpoint, device=args.device)
        cfg = loaded["config"]
        task_name = loaded["task"]["name"]
        episodes = args.eval_episodes if args.eval_episodes is not None else cfg.eval_episodes
        eval_seed = args.eval_seed if args.eval_seed is not None else cfg.eval_seed
        metrics = evaluate_td3(
            task_name,
            loaded["model"],
            episodes=episodes,
            seed=eval_seed,
            device=loaded["device"],
        )
        outdir = make_td3_output_dir(task_name, parent=args.output_dir, suffix="eval")
        payload = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "task": task_name,
            "checkpoint": str(args.checkpoint),
            "episodes": episodes,
            "eval_seed": eval_seed,
            "metrics": {k: v for k, v in metrics.items() if k != "episode_metrics"},
        }
        _write_eval_only_summary(outdir, payload)
        print(json.dumps(payload["metrics"], indent=2))
        print(f"Wrote eval-only results to {outdir}.")
        return

    cfg = _config_from_args(args)
    # Reject discrete-action tasks before a run directory is created for them.
    require_continuous_task(get_rl_task_spec(args.task))
    state = train_td3(
        args.task,
        cfg,
        output_dir=make_td3_output_dir(args.task, parent=args.output_dir),
        device=args.device,
        progress=not args.quiet,
    )
    summary = state["summary"]
    print(json.dumps({k: v for k, v in summary.items() if k != "checkpoint_path"}, indent=2))
    print(f"Checkpoint: {state['checkpoint_path']}")


if __name__ == "__main__":
    main()
