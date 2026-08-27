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
from model.ppo_workflow import (  # noqa: E402
    PPOConfig,
    evaluate_ppo,
    load_ppo_checkpoint,
    make_ppo_config,
    make_ppo_output_dir,
    train_ppo,
)


def _optional_float_arg(value: str) -> float | None:
    normalized = value.strip().lower()
    if normalized in {"none", "null"}:
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate the repo-native PPO baseline.")
    parser.add_argument("task", nargs="?", default="cartpole", help="RL task key, such as cartpole.")
    parser.add_argument("--profile", choices=["smoke", "debug", "fast", "gold"], default="fast")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/rl", help="Parent folder for timestamped run dirs.")
    parser.add_argument("--device", default=None, help="Optional torch device override, such as cpu or cuda.")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    anneal_group = parser.add_mutually_exclusive_group()
    anneal_group.add_argument(
        "--anneal-lr",
        dest="anneal_lr",
        action="store_true",
        help="Linearly anneal the learning rate over the configured timestep budget.",
    )
    anneal_group.add_argument(
        "--no-anneal-lr",
        dest="anneal_lr",
        action="store_false",
        help="Keep the learning rate constant.",
    )
    parser.set_defaults(anneal_lr=None)
    parser.add_argument("--rollout-length", type=int, default=None,
                        help="Fixed-length single-env rollout buffer collected before each update.")
    parser.add_argument("--num-minibatches", type=int, default=None)
    parser.add_argument("--update-epochs", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--clip-coef", type=float, default=None, help="PPO surrogate clipping epsilon.")
    clip_vloss_group = parser.add_mutually_exclusive_group()
    clip_vloss_group.add_argument(
        "--clip-vloss",
        dest="clip_vloss",
        action="store_true",
        help="Use the clipped value-function loss.",
    )
    clip_vloss_group.add_argument(
        "--no-clip-vloss",
        dest="clip_vloss",
        action="store_false",
        help="Use the plain squared value loss.",
    )
    parser.set_defaults(clip_vloss=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--vf-coef", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--target-kl", type=_optional_float_arg, default=argparse.SUPPRESS,
                        help="Approximate-KL early-exit threshold per update; use none to disable.")
    parser.add_argument(
        "--no-normalize-advantages",
        dest="normalize_advantages",
        action="store_false",
        default=None,
        help="Skip per-minibatch advantage normalization.",
    )
    parser.add_argument("--log-std-init", type=float, default=None,
                        help="Initial log std of the continuous (diagonal Gaussian) policy head.")
    parser.add_argument("--eval-frequency", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument(
        "--stochastic-eval",
        dest="eval_deterministic",
        action="store_false",
        default=None,
        help=(
            "Evaluate by sampling the policy instead of taking its argmax/mean action. "
            "Sampling during evaluation consumes the shared torch RNG stream, so runs "
            "are only comparable with other stochastic-eval runs."
        ),
    )
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
        help="Mean-return solve threshold; defaults to the task spec for non-CartPole tasks.",
    )
    parser.add_argument("--eval-only", action="store_true", help="Evaluate a saved checkpoint without training.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path for --eval-only.")
    parser.add_argument("--quiet", action="store_true", help="Disable progress logging during training.")
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> PPOConfig:
    overrides = {"seed": args.seed}
    if args.total_timesteps is not None:
        overrides["total_timesteps"] = args.total_timesteps
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if args.anneal_lr is not None:
        overrides["anneal_lr"] = args.anneal_lr
    if args.rollout_length is not None:
        overrides["rollout_length"] = args.rollout_length
    if args.num_minibatches is not None:
        overrides["num_minibatches"] = args.num_minibatches
    if args.update_epochs is not None:
        overrides["update_epochs"] = args.update_epochs
    if args.gamma is not None:
        overrides["gamma"] = args.gamma
    if args.gae_lambda is not None:
        overrides["gae_lambda"] = args.gae_lambda
    if args.clip_coef is not None:
        overrides["clip_coef"] = args.clip_coef
    if args.clip_vloss is not None:
        overrides["clip_vloss"] = args.clip_vloss
    if args.ent_coef is not None:
        overrides["ent_coef"] = args.ent_coef
    if args.vf_coef is not None:
        overrides["vf_coef"] = args.vf_coef
    if args.max_grad_norm is not None:
        overrides["max_grad_norm"] = args.max_grad_norm
    if hasattr(args, "target_kl"):
        overrides["target_kl"] = args.target_kl
    if args.normalize_advantages is not None:
        overrides["normalize_advantages"] = args.normalize_advantages
    if args.log_std_init is not None:
        overrides["log_std_init"] = args.log_std_init
    if args.eval_frequency is not None:
        overrides["eval_frequency"] = args.eval_frequency
    if args.eval_episodes is not None:
        overrides["eval_episodes"] = args.eval_episodes
    if args.eval_seed is not None:
        overrides["eval_seed"] = args.eval_seed
    if args.eval_deterministic is not None:
        overrides["eval_deterministic"] = args.eval_deterministic
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
    return make_ppo_config(args.profile, **overrides)


def _apply_task_success_defaults(args: argparse.Namespace, overrides: dict) -> None:
    """Resolve task-level success protocol for non-CartPole tasks.

    CartPole keeps the profile-baked 475/500 values untouched so PPO rows stay
    comparable with the other baselines; other tasks read their thresholds from
    the task spec unless the user overrides them on the command line.
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
        loaded = load_ppo_checkpoint(args.checkpoint, device=args.device)
        cfg = loaded["config"]
        task_name = loaded["task"]["name"]
        episodes = args.eval_episodes if args.eval_episodes is not None else cfg.eval_episodes
        eval_seed = args.eval_seed if args.eval_seed is not None else cfg.eval_seed
        deterministic = (
            cfg.eval_deterministic if args.eval_deterministic is None else args.eval_deterministic
        )
        metrics = evaluate_ppo(
            task_name,
            loaded["model"],
            episodes=episodes,
            seed=eval_seed,
            device=loaded["device"],
            deterministic=deterministic,
        )
        outdir = make_ppo_output_dir(task_name, parent=args.output_dir, suffix="eval")
        payload = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "task": task_name,
            "checkpoint": str(args.checkpoint),
            "episodes": episodes,
            "eval_seed": eval_seed,
            "eval_deterministic": deterministic,
            "metrics": {k: v for k, v in metrics.items() if k != "episode_metrics"},
        }
        _write_eval_only_summary(outdir, payload)
        print(json.dumps(payload["metrics"], indent=2))
        print(f"Wrote eval-only results to {outdir}.")
        return

    cfg = _config_from_args(args)
    state = train_ppo(
        args.task,
        cfg,
        output_dir=make_ppo_output_dir(args.task, parent=args.output_dir),
        device=args.device,
        progress=not args.quiet,
    )
    summary = state["summary"]
    print(json.dumps({k: v for k, v in summary.items() if k != "checkpoint_path"}, indent=2))
    print(f"Checkpoint: {state['checkpoint_path']}")


if __name__ == "__main__":
    main()
