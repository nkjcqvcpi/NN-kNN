# NN-kNN

NN-kNN is a neural case-based model for classification and regression. The
IJCAI-26 codebase is the maintained implementation: it learns retrieval over
cases with feature weighting and normalized case activation, then aggregates
retrieved labels for the target task.

- IJCAI-25 classification paper: [paper link](https://www.ijcai.org/proceedings/2025/763)
- IJCAI-26 regression paper: publication link will be added after release.

This is `main`, the standalone NN-kNN branch. It holds the shared NN-kNN core
that the stage branches reuse unchanged:

| Folder | Branch | Contents |
| --- | --- | --- |
| `nnknn/` | `main` | NN-kNN core, supervised baselines, prior results |
| `neural-cbr/` | `neural-cbr` | core + full-cycle neural CBR (T1) |
| `rl/` | `rl` | core + reinforcement learning (T2), including NN-kNN-RL |
| `llm/` | `LLM` | core + LLM/agent retrieval (T3) |

See [docs/NNKNN_CORE.md](docs/NNKNN_CORE.md) for the core file list and how
core changes are propagated, and [docs/BRANCH_ORGANIZATION.md](docs/BRANCH_ORGANIZATION.md)
for what lives where.

## Maintained Workflows

- Shared NN-kNN core: `model/nnknn_model.py`, `model/nn_cdh.py`, `model/common/`
- Regression orchestration: `model/regression_workflow.py`
- Classification orchestration: `model/classification_workflow.py`
- Classification loaders: `datasets/classification_data.py`
- Regression loaders: `datasets/reg_data.py`
- Table 1 regression reruns: `tools/run_table1_kfold.py`, `tools/table1_nnknn_kfold.py`
- Classification benchmarks: `tools/run_classification_benchmarks.py`

Classification uses the current model rather than the old classification
implementation. Normalized case activation is aggregated into class
probability mass and optimized with negative log likelihood. Supported
classification datasets include the IJCAI-25 small tabular set
(`iris`, `zebra`, `zebra_special`, `wine`, `breast_cancer`, `balance`,
`digits`) and image tasks (`mnist`, `cifar10`, `svhn`).
Image runs use an inner validation slice of the official training set for
checkpoint selection and report accuracy on the untouched official test set.
SST-5 and SST-2 remain deferred; text-loader dependencies are intentionally
outside the current classification workflow.

## Notebooks

- `nnknn_sample_classification.ipynb`: maintained classification workflow,
  explanations, representative baselines, and optional image subset runs.
- `nnknn_sample_regression.ipynb`: maintained regression workflow.
- `older versions/`: archival notebooks; not implementation references.

## Results

`results/prior_baseline/` holds the pre-MCB IJCAI-25 classification and
IJCAI-26 regression results that the T1 handoff uses as its legacy baseline.
See [results/README.md](results/README.md).

## Quick Checks

```bash
bash codex/setup.sh
.venv/bin/python codex/smoke_test.py --mode imports
.venv/bin/python codex/smoke_test.py --mode train
.venv/bin/python codex/smoke_test.py --mode classification
.venv/bin/python -m pytest tests -q
.venv/bin/python tools/check_nnknn_core.py --branches main neural-cbr rl LLM
```

On the Windows XPU workstation, use a stage checkout's `.venv` (for example
`..\neural-cbr\.venv\Scripts\python.exe`) from this folder; `main` has no
environment of its own.

For a small classification benchmark run:

```bash
.venv/bin/python tools/run_classification_benchmarks.py iris --mode kfold --runs 3 --epochs 20
```

Each CLI run creates a fresh timestamped folder under `results/` containing
`summary.csv`, `runs.csv`, and `manifest.json`.

Use representative small-data or subset image checks during development;
full multi-run paper benchmarks are intentionally expensive.
