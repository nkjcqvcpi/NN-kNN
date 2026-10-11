# Shared NN-kNN core

This file is identical on every branch. It defines the NN-kNN core that the four workstreams share and the rules for changing it.

## Workstreams

| Folder | Branch | Contents |
| --- | --- | --- |
| `nnknn/` | `main` | Standalone NN-kNN model: the core below, its supervised classification/regression baselines, prior (pre-MCB) results, papers, notes, and legacy notebooks. |
| `neural-cbr/` | `neural-cbr` | Core + full-cycle neural CBR (T1): `model/t1/`, `configs/t1/`, `tools/t1_*`, `tests/test_t1_*`. |
| `rl/` | `rl` | Core + reinforcement learning (T2), including NN-kNN-RL: RL workflows, `model/t2/`, `datasets/rl_tasks.py`, `tools/run_rl_*`, `tools/t2_*`, `tests/test_t2_*`. |
| `llm/` | `LLM` | Core + LLM/agent retrieval (T3): `model/t3/`, `tools/t3_*`, `tests/test_t3_*`. |

The T1, T2 and T3 specifications are Xiaomeng Ye's implementation handoffs in `career-2027/03-research-architecture/implementation-handoff/`.

## What the core is

The authoritative list is `CORE_FILES` in `tools/check_nnknn_core.py`; `tools/nnknn_core_manifest.json` records their SHA-256 hashes. In summary:

- `model/nnknn_model.py`: retrieval model, glocal feature weighting, stable case IDs, case activation, case-store operations.
- `model/nn_cdh.py`: NN-CDH regression and classification adaptation networks.
- `model/feature_extractors.py`, `model/device_utils.py`: representations and CPU/CUDA/XPU handling.
- `model/common/`: model construction and training (`core`), learned distances (`geometry`), prediction and loss (`outcomes`), nominal schema and grouped differences (`nominal`), fingerprints and manifests (`artifacts`).
- `model/classification_workflow.py`, `model/regression_workflow.py`, `model/mlkr.py` and their loaders in `datasets/`: the supervised baseline layer.
- `tools/run_classification_benchmarks.py`, `tools/run_table1_kfold.py`, `tools/resume_table1_kfold.py`, `tools/table1_nnknn_kfold.py`: supervised benchmark runners.
- `tools/check_nnknn_core.py`, `tests/test_nnknn_core.py`, and this file.

Core files never import stage code (`model.t1`, `model.t2`, `model.t3`, RL workflows, `datasets.rl_tasks`). Stage code imports the core, never the reverse. `tests/test_nnknn_core.py` enforces this, including imports inside functions.

## Changing the core

1. Make the change on `main` in `nnknn/`, in a commit that touches only core files.
2. Run `python tools/check_nnknn_core.py --write`, then `python -m pytest tests/test_nnknn_core.py -q` and `python codex/smoke_test.py --mode train`.
3. In each stage checkout, copy the core files and the manifest from `main` in one dedicated commit (`git checkout main -- <core files> tools/nnknn_core_manifest.json`), run that branch's tests, and name the source commit in the message.
4. Check all branches from any checkout: `python tools/check_nnknn_core.py --branches main neural-cbr rl LLM`.

Never merge a whole stage branch into another to share core code. A core fix found on a stage branch is first made on `main`, then propagated.

## Results and logs

Each branch keeps its own results and logs, organized by the handoff that defines the work:

- `main`: `results/prior_baseline/` (pre-MCB NN-kNN results; T1 Phase 0/6 references).
- `neural-cbr`: `results/t1/` and `logs/t1/` by T1 implementation phase.
- `rl`: `results/t2/` and `logs/t2/` by T2 implementation step and benchmark stage.
- `llm`: `results/t3/` and `logs/t3/` by T3 development stage.

Each branch's `results/README.md` maps handoff steps to directories, documents, and status. Run artifacts keep the paths they were written with; when a directory moves, the branch records the old and new locations instead of editing the artifacts.
