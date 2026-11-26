# Pack p4: extra comparator (TabPFN) and covariate-shift stress protocol (declared by us, 2026-11-26)

Status: **exploratory, our choice, not a PI decision.** The owner ordered on 2026-11-26 not to wait for the PI.
MODERN_BENCHMARK_STRATEGY lists TabPFN as the tabular foundation-model reference "on tasks within its supported regime"
and asks for "prespecified covariate ... changes" with "controlled transformations"; neither the regime nor the shift
intervention was chosen, so we choose them here. Nothing is confirmatory.

## (a) TabPFN in the baselines grid

| item | value |
|---|---|
| code | `tools/t1_grid_search.py` family `tabpfn` (opt-in), `model/t1/tabpfn_baseline.py`; no Table 1 code path touches it |
| package / model | `tabpfn==10.2.0` (pinned in main's `pyproject.toml`), model **v2** (the Nature 2025 model the strategy cites; v2.5 and later are gated behind a license acceptance) |
| weights | `tabpfn-v2-classifier.ckpt` sha256 `f65a3568...435f`, `tabpfn-v2-regressor.ckpt` sha256 `2ab5a07d...0736`, in `<NN-KNN root>/models/tabpfn/` (`tools/t1_tabpfn_weights.py`); a trial refuses another file |
| settings | `n_estimators` 4, CPU, one torch thread, `ignore_pretraining_limits` true, `random_state` = seed, one preprocessing job, library defaults otherwise; no HPO (`selection: fixed`, one candidate) |
| inputs | the grid's own preprocessed splits (train-fitted imputation, scaling, one-hot), the same tensors as kNN/MLP/CatBoost/TabM; validation is only scored |
| regime | at most 3000 training rows, 500 features, 10 classes |
| tasks | the 29 confirmation tasks minus `regression__car` (3977 training rows) and `regression__wine` (3897), outside the 3000-row regime, and minus `classification__tabarena_363681`, `regression__tabarena_363705`, `regression__tabarena_363631`, where CPU inference is infeasible (estimated 25, 30 and 37 CPU-h per seed: their validation and test streams hold 6-27 thousand rows): 24 tasks. The nine OpenML transfer tasks are capped at 3000 training rows by their grid already |
| CPU cost | measured on accel (one thread, 1 estimator): 50 s / 153 s / 547 s for 750 / 1500 / 3000 rows of context + queries with 20 features, i.e. t ~ rows^1.84 x feature groups; one seed of tabarena_363623 (2500 context rows, 834 + 1666 queries, 73 features, 4 estimators) did not finish in 2 h (estimate 6.1 CPU-h). 8 threads ran 6.4x faster, but the r760 supervisor counts one slot per job, so the configs keep one thread. Largest kept task: tabarena_363704, about 15 CPU-h per seed |
| splits / seeds | the datasets and splits of `grid_search_20261009.yaml` / `grid_transfer_20261013.yaml`; evaluation seeds 50, 51, 52 (those of p1's matched kNN/MLP/CatBoost/TabM evaluate and of maskedF/allcaseF). Checked: iris seed 50 has the same `split_indices_sha256` as the recorded `baselines_r760` kNN unit |
| records | study `handoff_tabpfn_r760`; a trial unit keeps no weights (nothing is trained): its `model` state names the checkpoint, its sha256, the package version and the settings |
| hosts | each task's p1 baselines host |

## (b) Covariate-shift stress protocol

| item | value |
|---|---|
| split | `splits.split_mode: covariate_pc1`, `shift_quantile: 0.7` (`model/t1/data.py`): PC1 of the standardized numeric features over all rows (median imputation, constant columns dropped, sign: largest loading positive, ties by row index); training side = lower 70 %, test = upper 30 %; validation = `val_frac / (1 - test_frac)` = 25 % of the training side (seeded, stratified where possible); OpenML outer splits not used |
| test groups | `shift_near` / `shift_far`: lower / upper half of the test rows by PC1 (per-group accuracy is recorded for classification) |
| tasks | the 20 real plan_full tasks; `synthetic_diag` is left out (fixed-design generator with its own shifted test group; the split mode refuses synthetic data) |
| NN-kNN | `shift/shiftF/<task>.yaml`: p1's maskedF config (frozen confirmation config, F arm) with seeds 30-32 and the shift split: units retrieval_only, the frozen-core reuse arm, F (diabets: F is the reuse arm). `shift/shift_retention_<task>.yaml`: plan_full retention (8 policies, tuned masked cores, finetune 30) at K 0.5 only |
| baselines | `t1_grid_search.py evaluate --split-mode covariate_pc1 --shift-quantile 0.7` of the recorded `grid_full_20261009` kNN, MLP and CatBoost selections, seeds 30-32 |
| records | study `handoff_shift_r760` |
| hosts | the plan_full retention host of each task: q-lab balance, breast_cancer, digits, 363614, 363626, energy_efficiency, student_performance, 363625, 363675; xps8960 iris, wine (cls), zebra, zebra_special, body_fat, diabets, yacht; r760 abalone, airfoil, car, wine (reg) |

## Known limits

- Validation lies on the training side, so model selection and early stopping see no shifted rows (intended).
- A class can be rare or absent on the training side; the split records `classes_missing_from_training` (none at seed 30
  on the 9 classification tasks; iris, wine and digits have strongly skewed class mixes between sides).
- TabPFN timing on one CPU thread is slow on the large transfer test streams (see the job lists' estimates).
