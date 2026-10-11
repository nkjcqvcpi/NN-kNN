# Results on main (standalone NN-kNN)

`main` keeps results produced by the standalone NN-kNN model. Stage results live on their own branches: T1 on `neural-cbr`, T2 on `rl`, T3 on `LLM` (see `docs/NNKNN_CORE.md`).

## Prior baseline (pre-MCB NN-kNN)

The T1 handoff treats these as the legacy feasibility baseline. They come from the older pre-MCB architecture and are not evidence for MCB or full-cycle neural CBR (`career-2027/03-research-architecture/T1_PRIOR_RESULTS.md`). T1 Phase 0 reproduces them, and Phase 6 re-runs the recoverable paper suites before the matched MCB/full-cycle variants.

| Directory | Contents | Source |
| --- | --- | --- |
| `prior_baseline/ijcai2025_classification/` | `experiment results.xlsx` (bias and feature-weight notes) and the SST-2, CIFAR-10 and SVHN explanation documents, 2025-01-23 | IJCAI-2025 classification paper |
| `prior_baseline/ijcai2025_classification/classification_nnknn_rerun_20260601_103353/` | 10-fold rerun of NN-kNN on iris, zebra and zebra_special with the current model | `tools/run_classification_benchmarks.py`; see `manifest.json` |
| `prior_baseline/ijcai2026_regression/table1_kfold_20260418_213251/`, `table1_kfold_20260421_114803/` | k-fold reruns of regression Table 1 (NN-kNN variants, MLP, MLKR and kNN baselines; the second run was resumed) | `tools/run_table1_kfold.py`, `tools/resume_table1_kfold.py`; see `done.json` |

## Paths

These directories moved on 2026-10-10 from the `results/` root to the paths above; their contents are unchanged. Their logs and `done.json` files still record the absolute output paths of the machines that produced them (`D:\NN-kNN\results\...`, `C:\Users\yexia\...`).

New runs from the supervised benchmark tools are written to `results/<runner>_<timestamp>/`. File each completed run under `prior_baseline/` only if it reproduces a prior paper result; otherwise describe it here.
