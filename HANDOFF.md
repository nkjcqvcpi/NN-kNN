# Handoff

Branch `main` (`nnknn/`): the standalone NN-kNN model and the shared core. Neural CBR (T1), RL (T2), and LLM (T3) status lives in the handoffs on `neural-cbr`, `rl`, and `LLM`. The RL status that used to be recorded here is preserved in `rl`'s `HANDOFF.md` and in the history of this file.

## Current Status

- The maintained NN-kNN core supports both regression and classification.
- 2026-10-10: `main` adopted the reconciled shared core from `neural-cbr` (`model/nnknn_model.py`, `model/nn_cdh.py`, `model/device_utils.py`, `model/common/`). The core now has a written contract (`docs/NNKNN_CORE.md`), a hash manifest (`tools/nnknn_core_manifest.json`), a checker (`tools/check_nnknn_core.py`), and tests (`tests/test_nnknn_core.py`).
- The same date fixed a stage dependency in the core: the classification adapter path imported `model.t1.nominal`, which exists only on `neural-cbr`. It now imports `model.common.nominal`.
- Classification uses normalized case activation as class probability mass with NLL loss; it does not restore the older class-weight formulation.
- `model/classification_workflow.py` and `datasets/classification_data.py` provide tabular and image classification workflows and representative baselines. `model/regression_workflow.py` and `tools/table1_nnknn_kfold.py` provide regression benchmarks and Table 1 reruns.
- `nnknn_sample_classification.ipynb` and `nnknn_sample_regression.ipynb` are the maintained notebook entry points; they call the Python workflows.
- A June 1 NN-kNN-only 10-fold rerun confirmed the recorded Iris and Zebra representative results exactly. These are current-core functionality checks, not exact reproductions of IJCAI-25 results from the older classification formulation. The run is in `results/prior_baseline/ijcai2025_classification/classification_nnknn_rerun_20260601_103353/`.

## Current Work Focus

- Classification support is functional; work on the core focuses on improving scores on the earlier IJCAI-25 classification tasks while keeping the maintained IJCAI-26-based retrieval implementation.
- Do not restore the retired legacy case-weight classification path solely to reproduce old numbers. Improvements should come from the current workflow, dataset protocol checks, hyperparameter tuning, or appropriate current-core changes.
- The June 1 rerun (10-fold stratified CV, `softmax`, `tau=0.5`, 50 epochs, seed 42) reproduced:
  - `iris`: `0.9600 +/- 0.0562`
  - `zebra` / Zebra (a): `0.5182 +/- 0.1488`
  - `zebra_special` / Zebra (b): `0.5227 +/- 0.0890`
- Zebra (a) and (b) are close to chance and are the immediate debugging priority. Inspect their alternating-boundary structure, preprocessing/splitting protocol, retrieved cases, feature weights, temperature, normalizer choice, and possible current-core feature mapping. Use `"zebra"` for Zebra (a) and `"zebra_special"` for Zebra (b) in the classification notebook.
- Before expanding to full prior-paper comparisons, establish better representative results on Zebra, then rerun the remaining small classification suite (`wine`, `breast_cancer`, `balance`, `digits`).
- T1 mechanisms (retention, reuse, synchronization, MCB, revision) are developed on `neural-cbr`. When T1, T2, or T3 work needs a core change, make it here first and propagate it (see `docs/NNKNN_CORE.md`).

## Pending Core Propagation

- `rl` and `LLM` receive the core from this branch directly.
- `neural-cbr` is running the T1 full study (started 2026-10-09). Its manifests record `git HEAD` and a source fingerprint over `model/ tools/ configs/ tests/ datasets/`, so the import fix and core contract are staged for it and applied only after the study finishes. Until then, `tools/check_nnknn_core.py --branches neural-cbr` reports `model/nnknn_model.py` and the contract files as different.

## Current Local Artifacts

- `checkpoints/` and transient smoke-test artifacts are expected and gitignored.
- Root checkpoint files such as `nnknn_regression_best.pth` and `nnknn_regression_best_retr.pth` may change when experiments are run.
- Classification runs write checkpoints beneath `checkpoints/` by default. The CLI creates a fresh `results/classification_<suite>_<timestamp>/` folder with summary, per-run, and manifest files.
- Completed prior-paper reruns are filed under `results/prior_baseline/`; see `results/README.md`. Keep old run folders for comparison and postmortem context rather than overwriting them.

## Where Durable Guidance Lives

Read `NNKNN_AGENT_GUIDE.md` for model descriptions, workflow entry points, Table 1 protocol details, output schemas, restart/resume patterns, and machine-specific caveats. Read `docs/NNKNN_CORE.md` before changing any shared core file.
