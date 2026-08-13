# Reinforcement-learning experiment artifacts

This directory preserves the reproducible, non-model artifacts from local RL
runs. Each completed run normally includes:

- `config.json` and `manifest.json`
- `summary.json`
- training, loss, evaluation, and diagnostic CSV files
- per-episode final and last-model evaluations

Folders ending in `_eval` are checkpoint-reload robustness evaluations and
contain `eval_summary.json` instead of the full training artifact set.

Model checkpoints (`checkpoint.pt` and `*.pth`) are intentionally excluded
from Git. Root-level console logs and PID files are also excluded because the
structured files in each run directory are the canonical experiment record.

Aggregate interpretations, fixed-budget comparisons, and figures are kept in
[`../../reports/experiment_log.md`](../../reports/experiment_log.md).
