# Experiment artifacts, filed by the host that produced them

Reorganised 2026-09-08. Results are split by machine, because this project has
now run on two of them and their numbers are **not directly comparable** — the
float path differs, and a measured device effect of ~28 points on the cap-2000
capacity arm is documented in `../reports/experiment_log.md` (p17, 2026-09-03).

```
results/
  r760/                   Linux, H100, CUDA          208 run dirs
  g234/                   Windows Server 2025, i5-13400F, Intel Arc, CPU only
    misc/                 smoke and regression runs   23
    ale/                  p20 ALE Pong arms           12
    ale_threadconfound/   the preserved OMP=3 run      1
    capacity_cpu/         p17 / p22 capacity sweep    39
  classification_*/       older non-RL results, untouched
  table1_kfold_*/         older non-RL results, untouched
```

## How a run was assigned to a host

In priority order, from each run's own `config.json`:

1. `runtime.platform` when present (added 2026-09-05, commit `40b5c3e`) —
   `Windows-2025Server-*` means g234.
2. `device == "cuda"` means r760. g234 has an Intel Arc and no CUDA, and the
   Arc was abandoned for training (see `../ENVIRONMENT_g234.md`).
3. Otherwise the run-date in the directory name: before 2026-09-03 is r760,
   since g234 only came up that day.

The directory-name timestamp is the run's start time on its **original** host,
so the p14 cap-2000 seeds 3–4 (started 2026-09-01 on r760, transferred on
09-05) correctly classify as r760 by rules 2 and 3 alike.

## Filing new runs

The runners still default to `--output-dir results/rl`, which is deliberate:
the host layout is a convention for callers, not something hardcoded into
research code. **Pass the destination explicitly**, as the campaign drivers in
`../scripts/g234/` already do:

```
tools/run_rl_nnknn.py cartpole ... --output-dir results/g234/<battery>
```

If a run ever lands in `results/rl/`, it is unfiled — move it under the right
host before harvesting it.

## What is kept, and what is not

Each completed run normally includes:

- `config.json` and `manifest.json`
- `summary.json` — the authoritative source for every reported number
- training, loss, evaluation and diagnostic CSV files
- per-episode final and last-model evaluations

Folders ending in `_eval` are checkpoint-reload robustness evaluations and hold
`eval_summary.json` instead of the full training artifact set.

Model checkpoints (`checkpoint.pt`, `*.pth`) are **excluded from Git** by
`.gitignore`, one rule per host tree. Console logs and PID files are excluded
too and now live in `../logs/<host>/`; the structured files in each run
directory are the canonical experiment record.

An incomplete run leaves a **completely empty** directory — these workflows
write their artifacts only at completion, so a killed run is indistinguishable
from debris. 14 such directories exist from the 09-02..09-08 round.

Aggregate interpretations, fixed-budget comparisons and figures are in
[`../reports/experiment_log.md`](../reports/experiment_log.md).
