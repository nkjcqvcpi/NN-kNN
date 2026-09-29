# T1 experiment plan

Everything here is **exploratory pilot** until the PI freezes the open values
(`docs/t1_spec/OPEN_DECISIONS_AND_HANDOFF_CHECKLIST.md`). Pilot values are
written explicitly in `configs/t1/_base.yaml` and marked `PILOT`. Pilots are for
choosing those values and seed counts; confirmatory runs then repeat the matrix
on fresh seeds with the frozen values.

## Common protocol (all experiments)

- **Streams.** Data are split into train, validation (early stopping and model selection only) and an untouched test set. An optional maintenance/audit split exists (`maint_frac`); the default audit is leave-one-out over the training cases. The scaler, target scaling and cohort bins are fit on train only. Test labels never reach retention, calibration or selection.
- **Matched comparisons.** Each (dataset, seed) trains **one** retrieval core. Every retention/reuse/sync condition starts from a copy of that core, so conditions share the split, initialization, case stream and budgets, and comparisons are paired by seed. Retention conditions use the same K and the same fine-tuning budget after selection.
- **Reporting.** Each outcome family is reported separately (contract section 10):
  - task quality: pre-adaptation accuracy/ECE or RMSE/MAE, and post-adaptation where applicable;
  - retention quality: kept fraction of corrupted, duplicate, rare and clean cases on synthetic data; the rare/shifted test subgroups;
  - stability: drift and churn;
  - reuse: the flip matrix and correction size;
  - correctability: M0/M1/M2, targeted vs collateral effects, precision/recall at the review budget.
- **Uncertainty.** Per-seed runs; `tools/t1_summarize.py` gives the mean, sd, 95% t-interval and seed-paired differences against the named reference, with a paired t-test and wins/losses. There is no multiplicity correction in pilots. Confirmatory analyses need prespecified margins (open decision).
- **Budgets.** Core: at most 300 epochs, patience 40. Post-selection fine-tuning: 30 epochs. Adapter: at most 300 epochs, patience 30. Sync: 120 epochs. One CPU thread per run.

## Matrix

| ID | Config | Question (plan gate) | Datasets | Conditions | Seeds | Reference for pairing |
|---|---|---|---|---|---|---|
| P0 | `p0_legacy_reference` | Reproduce current behaviour on T1 splits (Gate 1 context) | 7 classification (iris, wine, breast_cancer, balance, digits, zebra, zebra_special), 5 regression (energy_efficiency, yacht, airfoil, student_performance, abalone≤1500) | maintained `train_model` | 5 | - |
| P1-S | `p1_retention_synthetic` | Does quality-aware retention keep clean/rare cases and drop corrupted/redundant ones? (Gate 2 mechanism) | synthetic_diag (15% label corruption, 60 duplicates, 12-case rare cluster, shifted test group) | 11 policies × K ∈ {0.3, 0.5} | 10 | `random_K*` and `bias_current_K*` |
| P1-C | `p1_retention_classification` | Gate 2: at matched K, does any policy consistently beat random/downsampling and current bias pruning? How close to full memory? | 7 classification | 11 policies × K ∈ {0.2, 0.33, 0.5, 0.75} | 5 | `random_K*`, `bias_current_K*`, `full_memory` |
| P1-R | `p1_retention_regression` | Gate 2 for regression with counterfactual provenance | 5 regression | 11 policies × K ∈ {0.2, 0.5} | 5 | same |
| P2 | `p2_reuse` | Gate 3: does the adapter improve a prespecified pre/post outcome without calibration damage, harmful flips or leakage? | synthetic_diag + 7 classification | {nominal, logit} × {combined, diff_only, cls_only} | 5 | pre-adaptation of the same run |
| P4 | `p4_sync` | Gate 5: does coordination help without degrading locality or exceeding the correction bound? | synthetic_diag + 7 classification | independent / alternating_rr / alternating_rrr (K=0.5) | 5 | `independent` |
| P5 | `p5_mcb` | Gate 6: does MCB change representation/neighbourhood/selection stability, and at what task cost? | synthetic_diag, breast_cancer, digits, wine | MCB on/off × {trust_utility_coverage, random}, K=0.5, maintenance at epochs 50/100/150 | 5 | same policy, MCB off |
| T1.2 | `t12_revise_synthetic` | Harness check for revise: which flagging signal finds corrupted cases at the lowest review burden? M1 vs M2 effect | synthetic_diag | {provenance, bias, T, random, oracle} × budget {10, 25, 50, 75} | 10 | `random_b*`; `oracle_b*` = upper bound |

Primary metrics per table: `test_accuracy_pre` (classification) or `test_rmse_pre`
(regression). For reuse and sync, `test_accuracy_post` alongside
`test_accuracy_pre`, `test_ece_post` and `test_flip_correct_to_wrong`. For P5,
`mean_representation_drift`, `mean_neighborhood_churn` and
`mean_selection_churn`. For synthetic data, `kept_fraction_*` and
`test_accuracy_{rare,shifted,in_domain}`.

## Decisions needed from the PI before confirmatory runs

1. K grid and maintenance frequency/checkpoints (pilot: post-hoc selection plus 30 fine-tune epochs; in-training checkpoints for P5/P4).
2. s, α, the minimum evidence (R, A), the coverage floor, the rare/boundary protection rules, and the scalarization weights for combined policies (pilot values in `_base.yaml`).
3. λ_diff, λ_cls, and the probability protocol for nominal-residual scores (pilot: softmax, T=1).
4. s_task, the choice of the t* snapshot (pilot: best validation epoch of core training), and the synchronization weights.
5. The MCB momentum and whether to obtain the IU-Bloomington implementation for a reproduction.
6. Seeds, uncertainty method and practical-equivalence / minimum-improvement margins per benchmark.
7. The modern transfer family (TabArena subset) and contemporary baselines (boosted trees, MLP/TabM, TabPFN), plus which image/text legacy suites (CIFAR-10, SVHN, SST) to reconstruct.
8. RL (Phase 3 / T2 Stage A): the actor helpful/harmful signal, the critic audit target, and which RL branch/configuration is the frozen reference. Note that the aliasing bug fixed here also exists on `rl-iclr2027`.

## Not yet covered by the matrix (planned next)

- Data-valuation baselines (influence functions, Data Shapley on a subset) as retention comparators.
- Image/text legacy suites with trained encoders; a TabArena transfer subset.
- In-training maintenance (checkpoint protocol) for P1, after the post-hoc pilot identifies candidate policies.
- Regression synchronization; RL Phase 3; T1.2 human UI study.
