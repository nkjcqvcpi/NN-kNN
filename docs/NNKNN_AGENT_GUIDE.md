# NN-kNN Agent Guide

## Purpose

This file is a fast orientation guide for future AI agents working in this
repository. It is not a full paper summary. Its goal is to answer:

- what the NN-kNN model is in this codebase
- where the important code lives
- which workflow entry points matter for classification and regression
  reporting
- which files are legacy versus current

## What NN-kNN Is Here

At a high level, NN-kNN is a neural case-based model.

- It keeps the training examples as the case base.
- A feature extractor or raw tabular features define the representation space.
- Distances from a query to all stored cases are computed in that space.
- A glocal feature-weighting module reweights feature dimensions when computing
  those distances.
- Case activations are normalized, usually by `softmax` or `sparsemax`.
- The retrieved cases are aggregated into a prediction.
- For classification, normalized case activation is summed into class
  probability mass and optimized with NLL on that mass.
- For regression, an optional NN-CDH adapter can modify the retrieved solution
  after retrieval.

The config key for enabling case normalization is `normalize_over_cases`
because the selected normalizer may be `softmax` or `sparsemax`.

The key practical distinction in the current Table 1 work is:

- retrieval-only output before adaptation
- post-adaptation output after NN-CDH

## Core Model Files

- [model/nnknn_model.py](../model/nnknn_model.py)
  This is the main NN-kNN implementation.
- [model/nn_cdh.py](../model/nn_cdh.py)
  This contains the regression adaptation module used after retrieval.
- [model/regression_workflow.py](../model/regression_workflow.py)
  This is the main orchestration layer for regression experiments and repeated
  benchmarks.
- [model/classification_workflow.py](../model/classification_workflow.py)
  This is the maintained orchestration layer for classification experiments,
  image comparisons, and small-dataset benchmark checks.
- [model/common/](../model/common)
  Shared construction/training (`core`), learned distances (`geometry`),
  prediction and loss (`outcomes`), nominal schema (`nominal`), and
  fingerprints/manifests (`artifacts`) used by every stage branch.
- [datasets/classification_data.py](../datasets/classification_data.py)
  This contains portable classification loaders with train-only preprocessing.
- [datasets/paths.py](../datasets/paths.py)
  `datasets_root()` locates the shared data directory (see Dataset Files).

These files form the shared NN-kNN core. They are byte-identical on `main`,
`neural-cbr`, `rl`, and `LLM`; see [NNKNN_CORE.md](NNKNN_CORE.md)
before changing them. RL (T2), neural CBR (T1), and LLM (T3) code lives on
those branches, not here.

Important anchors:

- `GlocalFeatureWeight` at [model/nnknn_model.py](../model/nnknn_model.py#L109)
- `MLPFeatureProjector` at [model/nnknn_model.py](../model/nnknn_model.py#L197)
- `NN_KNN_Model` at [model/nnknn_model.py](../model/nnknn_model.py#L529)
- `train_model(...)` at [model/nnknn_model.py](../model/nnknn_model.py#L1493)
- `add_to_pair_list(...)` at [model/nn_cdh.py](../model/nn_cdh.py#L46)
- `NNCDHAdapter` at [model/nn_cdh.py](../model/nn_cdh.py#L115)

## Classification Workflow Files

Classification now uses the IJCAI-26 shared core rather than restoring the
older IJCAI-25 implementation. The retrieval pipeline stays common to both
tasks; only the final aggregation and loss are task-specific.

Important entry points in `model/classification_workflow.py`:

- `make_classification_cfg(...)`
- `load_classification_dataset_state(...)`
- `split_classification_state(...)`
- `train_nnknn_classification_state(...)`
- `evaluate_nnknn_classification_state(...)`
- `run_single_nnknn_classification_experiment(...)`
- `run_repeated_classification_model_benchmarks(...)`

Supported small datasets are `iris`, `zebra`, `zebra_special`, `wine`,
`breast_cancer`, `balance`, and `digits`. Supported image workflows are
`mnist`, `cifar10`, and `svhn`.
SST-5 and SST-2 remain deferred; do not add text-loader dependencies or
restructure the local `datasets` package as part of routine classification
work.

Small-data baselines are NN-kNN, kNN, and a four-hidden-layer MLP. Image
comparisons include a ConvNet, pixel kNN, pretrained-ConvNet-feature kNN,
trainable-CNN NN-kNN, and frozen-pretrained-CNN NN-kNN. The legacy `NN-kNNO`
case-weight path is intentionally unsupported because case weights were
removed from the maintained core.

Routine classification check:

```bash
uv run python codex/smoke_test.py --mode classification
uv run python tools/run_classification_benchmarks.py iris --mode kfold --runs 3 --epochs 20
```

The benchmark CLI records one run unit per dataset, method and split under
`NN-KNN/runs/core/<study>/` (`--study`, default
`classification_<suite>_<timestamp>`) and the summary in a report unit
`runs/core/<study>/_reports/summary/<timestamp>`; it prints the summary and
writes no CSV or JSON files (`docs/NNKNN_CORE.md`). The notebook keeps repeated
tabular and image benchmark sections opt-in so its default path stays quick.

For manual small-data classification debugging, use
[nnknn_sample_classification.ipynb](../notebooks/nnknn_sample_classification.ipynb). Run the
single-experiment sanity section first and change the dataset passed to
`run_single_nnknn_classification_experiment(...)`:

- use `"iris"` for the Iris sanity baseline
- use `"zebra"` for Zebra (a)
- use `"zebra_special"` for Zebra (b)

Inspect `accuracy`, `class_probabilities`, `most_activated_cases`,
`most_activated_class_ids`, `most_activated_activations`, and
`glocal_weightor.get_feature_weights_display()`. Misclassified validation
queries are more informative than always inspecting query zero. Zebra (a) is
the alternating vertical-band generator in `datasets/classification_data.py`;
debug whether retrieved cases cross those boundaries before expanding to full
benchmark comparisons. Use the opt-in repeated benchmark section only when a
single-run hypothesis is ready for aggregate validation.

## Regression Workflow Files

The regression work has been refactored so repeated experiment logic lives in
`model/regression_workflow.py` instead of being trapped in the notebook.

Important anchors:

- `load_regression_dataset_state(...)` at [model/regression_workflow.py](../model/regression_workflow.py#L259)
- `train_nnknn_regression_state(...)` at [model/regression_workflow.py](../model/regression_workflow.py#L903)
- `evaluate_nnknn_pre_post_adaptation_state(...)` at [model/regression_workflow.py](../model/regression_workflow.py#L1452)
- `run_repeated_regression_model_benchmarks(...)` at [model/regression_workflow.py](../model/regression_workflow.py#L1875)

NN-kNN-only workflow entry points:

- `train_nnknn_regression_state(...)`
- `evaluate_nnknn_regression_state(...)`
- `evaluate_nnknn_pre_post_adaptation_state(...)`
- `run_nnknn_regression_workflow(...)`
- `run_single_nnknn_regression_experiment(...)`
- `run_repeated_nnknn_regression_experiments(...)`

All-model benchmark workflow entry points:

- `run_regression_benchmark_methods_on_state(...)`
- `run_repeated_regression_model_benchmarks(...)`

This file also contains the repeated baseline implementations used in the
regression notebook and Table 1 work:

- `kNN(X)`
- `Oracle kNN(y_true)`
- `MLKR+kNN`
- `MLP`

Backward-compatible aliases are still present in `model/regression_workflow.py`,
so older notebook code should continue to work.

## Logic Check Against Older Git History

Compared with git commit `ff3d58416bbf23a56e47d79e00614e3bec6cd052`, the
training, evaluation, and testing logic remains the same for:

- NN-kNN
- KNN baseline
- Oracle KNN baseline
- MLKR + KNN baseline
- MLP baseline

The main real behavior change is a bugfix in `model/nnknn_model.py`:
`best_metric` now correctly stores the validation metric associated with the
best-loss checkpoint. Checkpoint selection logic itself is still based on
validation loss, same as before.

The repeated benchmark runner is new orchestration:

- same split per method within a run
- explicit reseeding for reproducibility
- improves consistency and fairness without changing individual model formulas

## Table 1 Reporting Files

Current Table 1 reruns are driven by:

- [tools/table1_nnknn_kfold.py](../tools/table1_nnknn_kfold.py)

Important anchors:

- `build_table1_paper_base_cfg(...)` at [tools/table1_nnknn_kfold.py](../tools/table1_nnknn_kfold.py#L42)
- `build_table1_baseline_method_cfgs(...)` at [tools/table1_nnknn_kfold.py](../tools/table1_nnknn_kfold.py#L193)
- `run_table1_nnknn_kfold(...)` at [tools/table1_nnknn_kfold.py](../tools/table1_nnknn_kfold.py#L479)
- `run_table1_kfold(...)` at [tools/table1_nnknn_kfold.py](../tools/table1_nnknn_kfold.py#L713)

This helper currently:

- runs 5-fold CV for Table 1 style reporting
- combines baselines plus NN-kNN family rows
- reports mean/std using `rmse_raw_table`
- emits long-form output that can be pivoted into a paper-style table

Current Table 1 protocol:

- `num_folds=5`
- datasets:
  - `califonia_housing`
  - `diabets`
  - `abalone`
  - `body_fat`
  - `airfoil`
  - `car`
  - `student_performance`
  - `yacht`
  - `energy_efficiency`
  - `bike_sharing`
  - `wine`
- exclude `UTKFace` for now
- baselines:
  - `kNN(X)`
  - `MLKR+kNN`
  - `MLP`
- NN-kNN rows per dataset:
  - `pure`
  - `adaptation`
  - `locality`
  - `locality + adaptation`
  across:
  - `softmax`
  - `softmax + locality`
  - `sparsemax`
  - `sparsemax + locality`

NN-kNN row semantics in Table 1:

- `pure`: retrieval before adaptation, no locality
- `adaptation`: post-adaptation, no locality
- `locality`: retrieval before adaptation, locality enabled
- `locality + adaptation`: post-adaptation, locality enabled

Table 1 workflow entry points:

- `tools/table1_nnknn_kfold.py`
- `run_table1_nnknn_kfold(...)`
- `run_table1_kfold(...)`

Ready-to-run example:

```python
from tools.table1_nnknn_kfold import run_table1_kfold

table_dataset_names = [
    "califonia_housing",
    "diabets",
    "abalone",
    "body_fat",
    "airfoil",
    "car",
    "student_performance",
    "yacht",
    "energy_efficiency",
    "bike_sharing",
    "wine",
]

table_summary_df, table_runs_df, table_artifacts = run_table1_kfold(
    dataset_names=table_dataset_names,
    num_folds=5,
    base_seed=42,
)

display(table_summary_df)
display(table_runs_df)
display(table_summary_df[["dataset", "entry_label", "rmse_raw_table"]])
```

Output objects:

- `table_summary_df`: one row per dataset/table entry; includes mean/std
  summary fields, `rmse_raw_table`, baseline rows, and all NN-kNN
  variant/result rows
- `table_runs_df`: one row per run/fold; useful for significance testing or
  debugging
- `table_artifacts`: nested as `{"baselines": ..., "nnknn": ...}` with
  detailed per-run objects

Called without `study`, `run_table1_kfold` records nothing and returns the
tables in memory. To keep a run, pass a study: every NN-kNN family and
baseline method per dataset and fold becomes a run unit under
`NN-KNN/runs/core/<study>/`, and `record_table1_report` stores the tables
(`summary_long`, `runs_long`, `table1_like`, `transposed`) in a report unit
that `read_table1_report` reads back. No CSV, JSON or checkpoint files are
written. Rerunning the same study skips complete units and resumes
incomplete ones:

```python
from tools.table1_nnknn_kfold import record_table1_report, run_table1_kfold

study = "table1_kfold_YYYYMMDD_HHMMSS"
unit_log = []
summary_df, runs_df, artifacts = run_table1_kfold(
    dataset_names=[
        "califonia_housing",
        "diabets",
        "abalone",
        "body_fat",
        "airfoil",
        "car",
        "student_performance",
        "yacht",
        "energy_efficiency",
        "bike_sharing",
        "wine",
    ],
    num_folds=5,
    base_seed=42,
    study=study,
    unit_log=unit_log,
)
report = record_table1_report(summary_df, runs_df, study=study, units=unit_log)
```

The CLIs `tools/run_table1_kfold.py --study S` and
`tools/resume_table1_kfold.py --study S` do the same.

## Dataset Files

The `datasets/` package holds only loader code. Project data (CSV/UCI tables,
the torchvision MNIST and CIFAR-10 files) and project-preprocessed caches
(`t1_cache/`) are not in git: they live in one directory shared by all
worktrees, `NN-KNN/datasets/` next to the worktrees, or the directory named by
the `NNKNN_DATASETS` environment variable. Loaders find it through
`datasets.paths.datasets_root()`, which raises if the directory is missing.

Datasets that a library downloads into a default cache of its own stay in that
library's default location, not in `NN-KNN/datasets/`: scikit-learn's
California Housing goes to `~/scikit_learn_data` (or `$SCIKIT_LEARN_DATA`,
which scikit-learn reads itself; `tools/resume_table1_kfold.py
--scikit-learn-data` sets it for one run), and OpenML datasets go to the
`openml` cache. torchvision has no default location (its `root` argument is
required), so MNIST, CIFAR-10 and SVHN stay in the shared directory.

Primary regression dataset loader:

- [datasets/reg_data.py](../datasets/reg_data.py)

Important anchors:

- `Bike_Sharing()` at [datasets/reg_data.py](../datasets/reg_data.py#L221)
- `Wine()` at [datasets/reg_data.py](../datasets/reg_data.py#L238)
- `DATATYPES` at [datasets/reg_data.py](../datasets/reg_data.py#L305)
- `Reg_data(...)` at [datasets/reg_data.py](../datasets/reg_data.py#L325)

Important status note:

- `bike_sharing` and `wine` are integrated into `reg_data.py`
- `datasets/reg_data_yu.py` is now a compatibility wrapper, not the canonical
  loader

## Notebook vs Script Status

Primary notebooks:

- [nnknn_sample_classification.ipynb](../notebooks/nnknn_sample_classification.ipynb)
  for classification single-run inspection and opt-in benchmark checks.
- [nnknn_sample_regression.ipynb](../notebooks/nnknn_sample_regression.ipynb)
  for regression single-run inspection and workflow calls.

The notebooks are still useful for interactive debugging and single-run
inspection, but serious repeated reporting now belongs in the workflow/helper
Python files.

Legacy or specialized files:

- `older versions/`
  Historical code snapshots.
- `model/nnknn_model_1_4.py`,
  `model/nnknn_model_with_post_mlp.py`, `model/nn_cdh_old.py`,
  and the original `model/cls_model.py`,
  `model/reg_model.py`, `model/module.py`
  Older variants kept on `main` only. They are outside the shared core and
  are not imported by maintained code.

## Current Practical Conventions

- For regression Table 1 reruns, prefer `run_table1_kfold(...)`.
- For general repeated regression benchmarks, prefer
  `run_repeated_regression_model_benchmarks(...)`.
- For single interactive notebook work:
  1. Run `Basic Setup`
  2. Choose/load dataset
  3. Standardize if needed
  4. Configure NN-kNN
  5. Train
  6. Evaluate/visualize
  7. Run explanation blocks on the trained model if desired
- For notebook batch table work, use the batch runner cell in
  `notebooks/nnknn_sample_regression.ipynb`; it calls
  `run_repeated_regression_model_benchmarks(...)`.
- For paper-style Table 1 reruns with 5-fold CV and std reporting, use
  `run_table1_kfold(...)` from `tools/table1_nnknn_kfold.py`.
- Use `datasets/reg_data.py` as the source of truth for tabular regression
  datasets.
- Treat `HANDOFF.md` as the current experiment-status document.

## Environment

Environment and dependencies are managed by uv only. The single `pyproject.toml`, `uv.lock` and `.python-version` (Python 3.14) live on `main`; one venv at `NN-KNN/.venv` is shared by all worktrees (`NN-KNN/{main,rl,llm,neural-cbr}`). Set the user environment variable `UV_PROJECT_ENVIRONMENT` to its absolute path once per device. In `main` run `uv sync` and `uv run python ...`; in another worktree run `uv run --project ../main --no-sync python ...`. torch is the Intel Arc XPU build (Windows x86_64 only; `environments` in the toml). No pip, no `requirements*.txt`, no hand-made venv. wandb was removed; runs are recorded in TensorBoard event files by `model/common/tracking.py` (`docs/NNKNN_CORE.md`, "Run tracking"). The two notebooks must be launched with the repo root as the working
directory.

## Machine-Specific Caveats

- On this machine, deterministic CUDA settings conflict with some baseline
  operations.
- The current Table 1 helper therefore pins `kNN(X)` and `MLP` baselines to
  CPU for reproducible repeated runs.
- Long experiments record their progress and results in their run units under
  `NN-KNN/runs/`; console output is not kept unless the launcher redirects it.
- `MLKR` depends on `metric-learn` and `scikit-learn`.
- `bike_sharing` depends on `ucimlrepo`; verify the active environment before
  launching a full Table 1 run that includes it.
- Table 1 baseline defaults are encoded in `tools/table1_nnknn_kfold.py` to
  match the notebook cell:
  - `kNN(X)`: `k=5`
  - `MLKR+kNN`: `k=25`, `max_iter=1000`
  - `MLP`: hidden `(64, 32)`, dropout `0.10`, `lr=1e-3`
- Progress logging uses:
  - `[table1] ...` messages for the Table 1 orchestrator
  - `[table1][nnknn] ...` messages for dataset/fold/family NN-kNN progress
  - `[benchmark] ...` messages for dataset/fold/method baseline progress

## If You Are Picking Up Mid-Run

- Check [HANDOFF.md](../HANDOFF.md) first.
- Then list the study's units: `python -m model.common.tracking ls ../runs/core/<study>`
  shows which are complete, failed or incomplete; rerunning the same `--study`
  skips the complete ones and resumes the others. A report unit under
  `_reports/table1/` means the tables were recorded.
- Older runs under `results/` (for example `results/prior_baseline/`, see
  `results/README.md`) are legacy and stay as they are.

## Default Reading Order

If you need to understand the repo quickly, read in this order:

1. [HANDOFF.md](../HANDOFF.md)
2. [tools/table1_nnknn_kfold.py](../tools/table1_nnknn_kfold.py)
3. [model/regression_workflow.py](../model/regression_workflow.py)
4. [model/nnknn_model.py](../model/nnknn_model.py)
5. [model/nn_cdh.py](../model/nn_cdh.py)
6. [datasets/reg_data.py](../datasets/reg_data.py)

