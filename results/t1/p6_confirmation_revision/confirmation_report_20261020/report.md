# T1 confirmation report (pre-registered)

Pre-registration `configs/t1/prereg_20261013.json` (sha256 `7c27c51036315b05...`, verified). F = `tuned::F` (per-task F from --final-map `results/t1/p4_component_sync/tuning_20261013/final_map.json`, sha256 `9672ece755c8404c...`: regression__diabets: `tuned::frozen_reuse__combined` (30 rows)); references: retrieval-only `retrieval_only`, alternating default `alternating_default_matched`, CatBoost and TabM HPO winners. Groups legacy + transfer, 30 tasks.

Error: 1 - accuracy or train-standardized RMSE (test stream). GMR = exp(pooled mean log error ratio, eps = 0.5/n_test for classification); REML tau^2, Hartung-Knapp t(k-1). GMR < 1 favours F. Holm over E1-E3b at 0.05; secondary tests Benjamini-Hochberg at q = 0.1.

## Primary endpoints

| ID | F vs | tasks | GMR [95% CI] | one-sided 95% upper | p | p (Holm) | supported |
|---|---|---|---|---|---|---|---|
| E1 | retrieval_only | 30 | 0.974 [0.936, 1.014] | 1.007 | 0.0974 | 0.2923 | no |
| E2 | alternating_default_matched | 30 | 0.996 [0.990, 1.003] | 1.002 | 0.1335 | 0.2923 | no |
| E3a | CatBoost | 30 | 1.135 [1.004, 1.283] | 1.257 | 0.6984 | 0.6984 | no |
| E3b | TabM | 30 | 0.987 [0.855, 1.138] | 1.111 | 0.0654 | 0.2616 | no |

- **E1**: no detectable gain from reuse on tuned cores (failed: holm_significant, gmr_at_most_max)
- **E2**: no detectable schedule effect (failed: holm_significant, gmr_at_most_max)
- **E3a**: trails tuned CatBoost by +13.5% [95% CI +0.4%, +28.3%] (failed: holm_significant)
- **E3b**: non-inferiority to tuned TabM not shown at delta = 0.1: GMR 0.987 [95% CI 0.855, 1.138], one-sided 95% upper bound 1.111 (failed: holm_significant)

Decision rules: E1/E2 Holm-adjusted one-sided p <= alpha and GMR <= gmr_max (E2: and every evaluable guard holds); with both groups, no claim if the transfer group's 95% CI lies above 1; E3a/E3b Holm-adjusted one-sided non-inferiority p (H0: GMR >= 1 + delta) <= alpha (prereg addendum 1); the unadjusted one-sided 95% upper bound is reported.

## E2 guards

| guard | value | limit | holds |
|---|---|---|---|
| pre_adaptation | 1.003 | 1.02 | yes |
| locality | 1.040 | 1.1 | yes |
| correction | 1.028 | 1.25 | yes |

Transfer group check (no superiority claim if its interval lies above 1): E1 [0.972, 1.019]; E2 [0.992, 1.014]

Non-inferiority sensitivity (descriptive): E3a delta 0.05: not; E3a delta 0.1: not; E3a delta 0.2: not; E3b delta 0.05: not; E3b delta 0.1: not; E3b delta 0.2: bound below margin

## By group (descriptive)

| group | ID | tasks | GMR [95% CI] | p |
|---|---|---|---|---|
| legacy | E1 | 20 | 0.944 [0.882, 1.012] | 0.0492 |
| legacy | E2 | 20 | 0.992 [0.983, 1.001] | 0.0354 |
| legacy | E3a | 20 | 1.153 [0.952, 1.397] | 0.6948 |
| legacy | E3b | 20 | 0.932 [0.737, 1.179] | 0.0785 |
| transfer | E1 | 10 | 0.995 [0.972, 1.019] | 0.3231 |
| transfer | E2 | 10 | 1.003 [0.992, 1.014] | 0.7131 |
| transfer | E3a | 10 | 1.111 [1.008, 1.224] | 0.5871 |
| transfer | E3b | 10 | 1.067 [1.021, 1.116] | 0.0782 |

## Secondary

| scope | analysis | statistic | criterion holds | p | p (BH) |
|---|---|---|---|---|---|
| pooled | ECE_T | median 0.0343 vs pre 0.0572 + 0.02 | yes | 0.0001 | 0.0002 |
| pooled | flips | 5681 / 4842 = 1.17 | no | 0.9547 | 1.0000 |
| pooled | ranks (Friedman) | 30 tasks | | 0.0000 | 0.0000 |
| legacy | ECE_T | median 0.0415 vs pre 0.0599 + 0.02 | yes | 0.0020 | 0.0035 |
| legacy | flips | 523 / 429 = 1.22 | no | 0.6738 | 0.8664 |
| legacy | ranks (Friedman) | 20 tasks | | 0.0001 | 0.0002 |
| transfer | ECE_T | median 0.0178 vs pre 0.0469 + 0.02 | yes | 0.0312 | 0.0469 |
| transfer | flips | 5158 / 4413 = 1.17 | no | 1.0000 | 1.0000 |
| transfer | ranks (Friedman) | 10 tasks | | 0.0000 | 0.0000 |

Six-method mean ranks (1 = lowest error)

| scope | CatBoost | TabM | MLP | kNN | NN-kNN retrieval-only | F |
|---|---|---|---|---|---|---|
| pooled | 1.87 | 2.60 | 3.90 | 5.03 | 4.10 | 3.50 |
| legacy | 2.00 | 2.90 | 3.95 | 4.55 | 4.25 | 3.35 |
| transfer | 1.60 | 2.00 | 3.80 | 6.00 | 3.80 | 3.80 |

Log loss and macro-F1 on churn and MIC (descriptive; mean over seeds, seeds with a value in parentheses)

| task | method | log loss | log loss, validation temperature | macro-F1 |
|---|---|---|---|---|
| churn | F | 0.2973 (10) | 0.2226 (10) | 0.8400 (10) |
| churn | NN-kNN retrieval-only | 0.4203 (10) | not recorded | 0.8374 (10) |
| churn | alternating default | 0.2935 (10) | 0.2166 (10) | 0.8444 (10) |
| churn | CatBoost | 0.1606 (10) | not recorded | 0.8956 (10) |
| churn | TabM | 0.2012 (10) | not recorded | 0.8602 (10) |
| churn | MLP | 0.2359 (10) | not recorded | 0.8195 (10) |
| churn | kNN | 1.1559 (10) | not recorded | 0.6669 (10) |
| MIC | F | 0.5671 (10) | 0.5497 (10) | 0.2278 (10) |
| MIC | NN-kNN retrieval-only | 2.0169 (10) | not recorded | 0.2219 (10) |
| MIC | alternating default | 0.5606 (10) | 0.5442 (10) | 0.2156 (10) |
| MIC | CatBoost | 0.4717 (10) | not recorded | 0.2155 (10) |
| MIC | TabM | 0.4674 (10) | not recorded | 0.2152 (10) |
| MIC | MLP | 0.4822 (10) | not recorded | 0.2117 (10) |
| MIC | kNN | 3.1013 (10) | not recorded | 0.1949 (10) |

- log_loss: test log loss of the arm's own predicted distribution: adapted neural-CBR arms' test_L_post (cross-entropy of softmax(kappa * s), the distribution test_ece_post measures, argmax = accuracy_post); retrieval-only's test_loss_pre (NLL of the retrieved class mass p0 smoothed by 1e-8, so a zero mass on the true class costs about 18.4); baselines' receipt test_metrics.loss (sklearn log_loss, probabilities clipped at 1e-12)
- log_loss_T: adapted neural-CBR arms' test_nll_post_T: the same logits divided by one temperature fitted on the validation stream (the distribution behind ECE_T; argmax unchanged); retrieval-only and the baselines record no temperature-scaled output
- macro_f1: unweighted mean of per-class F1 over the classes in the targets or predictions (sklearn's macro average), from per-example test predictions: neural-CBR runs' retrieval_events.jsonl (final_prediction; pre_adaptation_prediction for retrieval-only), baselines' test_predictions.pt (argmax); a seed is used only if its predictions reproduce the recorded accuracy

## Per task (F / reference error ratio, geometric mean over paired seeds)

| task | group | seeds | F error | E1 | E2 | E3a | E3b | F pre / retrieval-only |
|---|---|---|---|---|---|---|---|---|
| classification__iris | legacy | 30 | 0.0522 | 1.060 | 1.113 | 1.100 | 1.045 | 1.157 |
| classification__wine | legacy | 30 | 0.0407 | 0.881 | 0.806 | 1.419 | 1.556 | 1.213 |
| classification__breast_cancer | legacy | 10 | 0.0360 | 0.916 | 0.907 | 0.969 | 0.966 | 1.195 |
| classification__balance | legacy | 10 | 0.0480 | 0.825 | 0.744 | 0.535 | 1.191 | 0.894 |
| classification__digits | legacy | 10 | 0.0219 | 1.054 | 1.007 | 0.914 | 1.124 | 0.994 |
| classification__zebra | legacy | 30 | 0.1061 | 0.817 | 0.879 | 1.944 | 0.105 | 0.863 |
| classification__zebra_special | legacy | 30 | 0.2864 | 0.999 | 1.031 | 5.113 | 0.571 | 1.016 |
| regression__energy_efficiency | legacy | 10 | 0.0527 | 0.906 | 0.963 | 1.124 | 0.993 | 0.977 |
| regression__yacht | legacy | 30 | 0.0844 | 0.553 | 0.810 | 1.168 | 1.273 | 0.957 |
| regression__airfoil | legacy | 10 | 0.2174 | 0.964 | 0.980 | 0.986 | 0.938 | 0.989 |
| regression__student_performance | legacy | 10 | 0.4716 | 1.001 | 1.002 | 1.059 | 1.087 | 1.011 |
| regression__abalone | legacy | 10 | 0.6587 | 0.999 | 1.000 | 0.982 | 1.021 | 0.999 |
| regression__body_fat | legacy | 30 | 0.7109 | 1.050 | 0.996 | 1.079 | 0.953 | 0.995 |
| regression__diabets | legacy | 30 | 0.7260 | 0.998 | 0.998 | 1.003 | 1.038 | 1.000 |
| regression__car | legacy | 10 | 0.2251 | 0.982 | 0.964 | 1.363 | 0.962 | 1.066 |
| regression__wine | legacy | 10 | 0.7409 | 1.002 | 0.996 | 1.016 | 0.981 | 0.999 |
| classification__tabarena_363614 | legacy | 10 | 0.0087 | 0.601 | 0.692 | 0.911 | 0.797 | 0.799 |
| classification__tabarena_363626 | legacy | 10 | 0.2754 | 1.014 | 0.992 | 1.113 | 1.088 | 0.999 |
| regression__tabarena_363625 | legacy | 10 | 0.3808 | 0.948 | 0.954 | 1.386 | 1.140 | 0.987 |
| regression__tabarena_363675 | legacy | 10 | 0.3961 | 0.973 | 0.985 | 1.052 | 1.037 | 1.003 |
| classification__tabarena_363702 | transfer | 10 | 0.0419 | 1.097 | 1.024 | 1.048 | 1.070 | 1.037 |
| classification__tabarena_363704 | transfer | 10 | 0.2355 | 0.963 | 0.983 | 1.028 | 1.077 | 1.018 |
| classification__tabarena_363681 | transfer | 10 | 0.3074 | 0.980 | 1.019 | 1.163 | 1.132 | 1.025 |
| classification__tabarena_363623 | transfer | 10 | 0.0743 | 1.064 | 1.036 | 1.628 | 1.217 | 1.051 |
| classification__tabarena_363711 | transfer | 10 | 0.1266 | 1.029 | 1.033 | 1.116 | 1.111 | 1.004 |
| regression__tabarena_363705 | transfer | 10 | 0.3752 | 0.974 | 0.995 | 1.070 | 1.043 | 1.008 |
| regression__tabarena_363686 | transfer | 10 | 0.3525 | 0.973 | 1.004 | 1.133 | 1.079 | 1.009 |
| regression__tabarena_363631 | transfer | 10 | 0.1719 | 0.995 | 0.994 | 1.029 | 0.928 | 1.027 |
| regression__tabarena_363693 | transfer | 10 | 0.6993 | 1.004 | 0.991 | 1.004 | 1.009 | 1.015 |
| regression__tabarena_363697 | transfer | 10 | 0.5532 | 0.983 | 0.999 | 1.047 | 1.038 | 1.001 |

## Provenance

- neural-CBR rows analysed 1320, ignored 735, identical duplicates 11; run manifests found 1320, absent 0
- run source fingerprints: 3825a422c093 (1200 rows), c614a768d682 (90 rows), f43f59a46a49 (30 rows)
- run source fingerprints by task:
  - `3825a422c093`: 26 task(s), 1200 rows: regression__abalone, regression__airfoil, classification__balance, regression__body_fat, classification__breast_cancer, regression__car, regression__diabets, classification__digits, regression__energy_efficiency, classification__iris, regression__wine, regression__student_performance, classification__tabarena_363614, classification__tabarena_363623, regression__tabarena_363625, classification__tabarena_363626, regression__tabarena_363675, classification__tabarena_363681, regression__tabarena_363686, classification__tabarena_363702, classification__tabarena_363704, classification__tabarena_363711, classification__wine, regression__yacht, classification__zebra, classification__zebra_special
  - `c614a768d682`: 3 task(s), 90 rows: regression__tabarena_363631, regression__tabarena_363693, regression__tabarena_363705
  - `f43f59a46a49`: 1 task(s), 30 rows: regression__tabarena_363697
- --expect-fingerprints held for all 1320 analysed rows: `regression__tabarena_363705` -> `c614a768d682`; `regression__tabarena_363631` -> `c614a768d682`; `regression__tabarena_363693` -> `c614a768d682`; `regression__tabarena_363697` -> `f43f59a46a49`; `*` -> `3825a422c093`
- devices (rows): {'cpu': 1320}; comparisons pooling device types: 0
- n_test sources: {'rows': 220, 'baselines': 0, 'fallback': 0}
