# Neural CBR (T1) progress report: tuning, pre-registered confirmation on 30 tasks, and a case-base maintenance review

Snapshot: 2026-10-20, about 09:30 local time. All experiments of this phase had finished by then. Chinese version: `NEURAL_CBR_PROGRESS_2026-10-20_ZH.md` (same sections, same tables, same numbers).
Earlier report: `NEURAL_CBR_PROGRESS_2026-10-13_EN.md`; its results are not repeated here except where stated.

## 0. Scope and reading rules

- **Covered:** neural CBR (T1) only: what the model can do, and what the experiments found. **Not covered:** RL, LLM.
- **Confirmation protocol.** The final configuration F was frozen before any evaluation seed was run (pre-registration `docs/T1_PREREG_20261013.md` and its four addenda). Evaluation seeds are 40-49, plus 50-69 on the seven small tasks. Seeds used for development (20-25) never enter the confirmation.
- **Error and effect.** Error is 1 - accuracy (classification) or train-standardized RMSE (regression), on the test stream. Each task's effect is the log error ratio, F against a reference, averaged over seed-paired runs. Tasks are pooled by random-effects meta-analysis (REML, Hartung-Knapp). GMR (geometric mean ratio) below 1 favours F. Holm correction over the four primary endpoints at 0.05.
- **Every claim points at a table;** numbers in the text are copied from the tables. Tables 1-7 come from the pre-registered analysis of all 30 tasks (the 20 legacy tasks and the 10 harder transfer tasks).

## 1. Summary

**Table 1.** Primary pre-registered endpoints, 30 tasks (F against each reference)

| ID | F vs | tasks | GMR [95% CI] | one-sided 95% upper | p | p (Holm) | supported |
|---|---|---|---|---|---|---|---|
| E1 | NN-kNN retrieval-only (reuse gate) | 30 | 0.974 [0.936, 1.014] | 1.007 | 0.0974 | 0.2923 | no |
| E2 | alternating default schedule (synchronization gate) | 30 | 0.996 [0.990, 1.003] | 1.002 | 0.1335 | 0.2923 | no |
| E3a | tuned CatBoost (non-inferiority, margin 10%) | 30 | 1.135 [1.004, 1.283] | 1.257 | 0.6984 | 0.6984 | no |
| E3b | tuned TabM (non-inferiority, margin 10%) | 30 | 0.987 [0.855, 1.138] | 1.111 | 0.0654 | 0.2616 | no |

Findings, worded as the pre-registration prescribes for unsupported endpoints:

1. **E1:** no detectable gain from reuse on tuned cores. **E2:** no detectable schedule effect (the three guards of Table 2 hold). **E3a:** F trails tuned CatBoost by 13.5% [95% CI 0.4%, 28.3%]. **E3b:** non-inferiority to tuned TabM is not shown at a 10% margin (Table 1).
2. Gains from reuse are concentrated in a few small legacy tasks and are absent on the ten harder tasks (Table 3, Table 6).
3. Reuse improves calibration, but corrections do not fix more predictions than they break (Table 4).
4. The case-base maintenance step has never been evaluated where it could matter, and its decision rule is largely an exposure score; this is documented with development-seed diagnostics (Section 5, Table 9) and does not affect Tables 1-7, because F contains no maintenance.

## 2. Confirmation results

**Table 2.** Guards on the synchronization comparison (E2)

| guard | value | limit | holds |
|---|---|---|---|
| pre-adaptation error of F against retrieval-only | 1.003 | 1.02 | yes |
| input-space locality against the default | 1.040 | 1.10 | yes |
| correction size against the default | 1.028 | 1.25 | yes |

Transfer-group check (no superiority claim if the transfer interval lies above 1): E1 [0.972, 1.019]; E2 [0.992, 1.014].

**Table 3.** Primary endpoints by group (descriptive)

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

The legacy p-values are unadjusted. F's per-task selection was made on those same legacy tasks (on development seeds), so they are descriptive, not confirmatory.

**Table 4.** Secondary analyses (Benjamini-Hochberg at q = 0.10)

| scope | analysis | statistic | criterion holds | p | p (BH) |
|---|---|---|---|---|---|
| pooled | calibrated ECE | median 0.0343 against pre-adaptation 0.0572 + 0.02 | yes | 0.0001 | 0.0002 |
| pooled | wrong-to-correct vs correct-to-wrong flips | 5681 / 4842 = 1.17 (required at least 2) | no | 0.9547 | 1.0000 |
| legacy | calibrated ECE | median 0.0415 against 0.0599 + 0.02 | yes | 0.0020 | 0.0035 |
| legacy | flips | 523 / 429 = 1.22 | no | 0.6738 | 0.8664 |
| transfer | calibrated ECE | median 0.0178 against 0.0469 + 0.02 | yes | 0.0312 | 0.0469 |
| transfer | flips | 5158 / 4413 = 1.17 | no | 1.0000 | 1.0000 |

**Table 5.** Mean error rank over the six methods (1 = lowest error)

| scope | CatBoost | TabM | MLP | kNN | NN-kNN retrieval-only | F |
|---|---|---|---|---|---|---|
| pooled (30 tasks) | 1.87 | 2.60 | 3.90 | 5.03 | 4.10 | 3.50 |
| legacy (20) | 2.00 | 2.90 | 3.95 | 4.55 | 4.25 | 3.35 |
| transfer (10) | 1.60 | 2.00 | 3.80 | 6.00 | 3.80 | 3.80 |

**Table 6.** Per task: F's test error and its ratio to each reference (geometric mean over paired seeds; below 1 favours F). "pre" is F's error before adaptation over retrieval-only.

| task | group | seeds | F error | E1 | E2 | E3a | E3b | pre |
|---|---|---|---|---|---|---|---|---|
| iris | legacy | 30 | 0.0522 | 1.060 | 1.113 | 1.100 | 1.045 | 1.157 |
| wine (classification) | legacy | 30 | 0.0407 | 0.881 | 0.806 | 1.419 | 1.556 | 1.213 |
| breast_cancer | legacy | 10 | 0.0360 | 0.916 | 0.907 | 0.969 | 0.966 | 1.195 |
| balance | legacy | 10 | 0.0480 | 0.825 | 0.744 | 0.535 | 1.191 | 0.894 |
| digits | legacy | 10 | 0.0219 | 1.054 | 1.007 | 0.914 | 1.124 | 0.994 |
| zebra | legacy | 30 | 0.1061 | 0.817 | 0.879 | 1.944 | 0.105 | 0.863 |
| zebra_special | legacy | 30 | 0.2864 | 0.999 | 1.031 | 5.113 | 0.571 | 1.016 |
| energy_efficiency | legacy | 10 | 0.0527 | 0.906 | 0.963 | 1.124 | 0.993 | 0.977 |
| yacht | legacy | 30 | 0.0844 | 0.553 | 0.810 | 1.168 | 1.273 | 0.957 |
| airfoil | legacy | 10 | 0.2174 | 0.964 | 0.980 | 0.986 | 0.938 | 0.989 |
| student_performance | legacy | 10 | 0.4716 | 1.001 | 1.002 | 1.059 | 1.087 | 1.011 |
| abalone | legacy | 10 | 0.6587 | 0.999 | 1.000 | 0.982 | 1.021 | 0.999 |
| body_fat | legacy | 30 | 0.7109 | 1.050 | 0.996 | 1.079 | 0.953 | 0.995 |
| diabetes | legacy | 30 | 0.7260 | 0.998 | 0.998 | 1.003 | 1.038 | 1.000 |
| car | legacy | 10 | 0.2251 | 0.982 | 0.964 | 1.363 | 0.962 | 1.066 |
| wine (regression) | legacy | 10 | 0.7409 | 1.002 | 0.996 | 1.016 | 0.981 | 0.999 |
| anneal (363614) | legacy | 10 | 0.0087 | 0.601 | 0.692 | 0.911 | 0.797 | 0.799 |
| credit-g (363626) | legacy | 10 | 0.2754 | 1.014 | 0.992 | 1.113 | 1.088 | 0.999 |
| concrete (363625) | legacy | 10 | 0.3808 | 0.948 | 0.954 | 1.386 | 1.140 | 0.987 |
| insurance (363675) | legacy | 10 | 0.3961 | 0.973 | 0.985 | 1.052 | 1.037 | 1.003 |
| splice (363702) | transfer | 10 | 0.0419 | 1.097 | 1.024 | 1.048 | 1.070 | 1.037 |
| students dropout (363704) | transfer | 10 | 0.2355 | 0.963 | 0.983 | 1.028 | 1.077 | 1.018 |
| in-vehicle coupon (363681) | transfer | 10 | 0.3074 | 0.980 | 1.019 | 1.163 | 1.132 | 1.025 |
| churn (363623) | transfer | 10 | 0.0743 | 1.064 | 1.036 | 1.628 | 1.217 | 1.051 |
| MIC (363711) | transfer | 10 | 0.1266 | 1.029 | 1.033 | 1.116 | 1.111 | 1.004 |
| superconductivity (363705) | transfer | 10 | 0.3752 | 0.974 | 0.995 | 1.070 | 1.043 | 1.008 |
| Miami housing (363686) | transfer | 10 | 0.3525 | 0.973 | 1.004 | 1.133 | 1.079 | 1.009 |
| diamonds (363631) | transfer | 10 | 0.1719 | 0.995 | 0.994 | 1.029 | 0.928 | 1.027 |
| physiochemical protein (363693) | transfer | 10 | 0.6993 | 1.004 | 0.991 | 1.004 | 1.009 | 1.015 |
| QSAR-TID-11 (363697) | transfer | 10 | 0.5532 | 0.983 | 0.999 | 1.047 | 1.038 | 1.001 |

Reading of Table 6:

- **TabM (E3b).** Its pooled ratio below 1 rests on zebra (0.105) and zebra_special (0.571), where the tuned TabM has about 48-50% error. On the ten transfer tasks F trails TabM (1.067, Table 3).
- **Reuse (E1).** Large gains appear on a few small tasks: yacht 0.553, anneal 0.601, zebra 0.817, balance 0.825, wine 0.881. No transfer task has a ratio below 0.963.
- **Checkpoint choice.** In about 30% of F runs (119 of 400 analysed tuned F runs, excluding diabetes) the selected checkpoint was reached before any retrieval or joint update. There F's retrieval core is the untouched retrieval-only core. This bears on reading E2 and the guard in Table 2.

**Table 7.** Log loss and macro-F1 on churn and MIC (descriptive; mean over 10 seeds)

| task | method | log loss | log loss at validation temperature | macro-F1 |
|---|---|---|---|---|
| churn | F | 0.2973 | 0.2226 | 0.8400 |
| churn | NN-kNN retrieval-only | 0.4203 | not recorded | 0.8374 |
| churn | alternating default | 0.2935 | 0.2166 | 0.8444 |
| churn | CatBoost | 0.1606 | not recorded | 0.8956 |
| churn | TabM | 0.2012 | not recorded | 0.8602 |
| churn | MLP | 0.2359 | not recorded | 0.8195 |
| churn | kNN | 1.1559 | not recorded | 0.6669 |
| MIC | F | 0.5671 | 0.5497 | 0.2278 |
| MIC | NN-kNN retrieval-only | 2.0169 | not recorded | 0.2219 |
| MIC | alternating default | 0.5606 | 0.5442 | 0.2156 |
| MIC | CatBoost | 0.4717 | not recorded | 0.2155 |
| MIC | TabM | 0.4674 | not recorded | 0.2152 |
| MIC | MLP | 0.4822 | not recorded | 0.2117 |
| MIC | kNN | 3.1013 | not recorded | 0.1949 |

## 3. How F was fixed (tuning, before any evaluation seed)

Selection used only the validation streams of development seeds 20-25; the test streams of those seeds served only as the declared leave-one-seed-out held-out check. A change was adopted only if its mean paired held-out log ratio was below -1 task-clustered standard error (H2c: non-inferior).

**Table 8.** Tuning blocks and decisions (held-out GMR against the alternating default; lower is better)

| block | what was tested | evidence | decision |
|---|---|---|---|
| H6 selection rule | per-task selection by validation error, then loss, against one global rule | GMR 0.891 vs 0.932; paired -0.044, task-clustered z -1.98, tasks better/worse 13/4 | adopted |
| feasibility | per-task pre-adaptation constraint (1.02 x retrieval-only) | per-task rule 0.895; global form 0.967 | applied to the whole pool |
| H1 | retrieval guard and pre-adaptation weight (10 candidates) | 0.895 to 0.904 (paired +0.010, z +0.95) | rejected |
| H2a | score scale for the nominal output (5 candidates) | paired -0.004, z -0.38 | not accepted |
| H2b | logit-residual output (2 candidates) | paired 0.000 | not accepted |
| H2 / H2c | classification score scale 10 against 1 | held-out 0.842 to 0.822 (paired -0.023, task SE 0.036, tasks 5/3); median validation ECE of the selections 0.239 to 0.012 | adopted for classification (non-inferior; calibration fixed) |
| H5 | budget and stopping rule (8 candidates) | 0.898 to 0.900 (paired +0.002, tasks 6/6) | rejected |
| H3 | adapter capacity and regularization | 0.898 to 0.906 (paired +0.008, tasks 6/7) | rejected |
| H4 | neighbourhood size (optional) | not run | skipped |

Only H6 and H2c were adopted. With six development seeds every block that added candidates left held-out error unchanged or worse, so larger candidate pools overfit validation. The H2c acceptance rule (non-inferiority) was set during tuning, not in the pre-registration, and is disclosed as such. The blocks were decided slightly out of the pre-registered order; this has no effect on F, because every intermediate block was rejected.

F is therefore: per task, the validation-selected joint or alternating schedule on the 18 tuned legacy tasks (frozen-core reuse on diabetes); one global joint schedule (summed objective, adapter warm-up 5, post-adaptation weight 0.5, neighbour weight 0, small-correction weight 0.1, adapter rate 0.003; score scale 10 for classification) on car, regression wine and the ten transfer tasks, which have no development runs. The ten transfer tasks were chosen for difficulty before any neural-CBR result on them; their cores come from a five-family search with a 3,000-row training cap and neural CBR was never tuned on them.

## 4. Carried over from the 13 October report (not re-run)

- **EMA encoder.** On tabular tasks the EMA momentum encoder gave no benefit over the shared encoder, at default and at tuned cores, and the raw-feature core is stronger (13 October report, Tables 25, 26, 28).
- **Joint retrieval-reuse training** (schedules joint_rr, joint_rrr) is implemented and ranked best in the schedule search, but its gain over the alternating default is small on tuned cores: -1.8% (p = 0.323) at the corrected batch size. The confirmation (E2, Table 1) finds no schedule effect (13 October report, Tables 30-33).

## 5. Case-base maintenance: design review (new)

The question was whether the maintenance step is designed properly. Code, the PI specification, all earlier results and the maintenance literature were reviewed; thirteen candidate flaws were each checked by an independent reviewer who tried to refute it, and all thirteen held. The full list is in `docs/T1_CBM_DESIGN_REVIEW_20261019.md`. The central ones:

- The provenance score Q measures how often a case is retrieved rather than how good it is: every case in a query's neighbourhood receives that query's final outcome. On a synthetic set with 15% flipped labels, Spearman(Q, activation) is 0.86-0.99 and Q keeps 58-60% of corrupted cases at 75% capacity, where random keeps 75%.
- Scores come from the training queries the model was fitted to, so harmful support is nearly absent on easy tasks.
- Ties between never-retrieved cases are broken by case number, so row order decides most evictions on some tasks.
- "Dynamic" maintenance prunes once; nothing is re-admitted.
- Maintenance was never evaluated on noisy, redundant, imbalanced, large or hard data. "Removal gives no consistent gain" is therefore the expected outcome, not evidence about maintenance.

**Table 9.** Development-seed diagnostics at 50% capacity, alternating-default runs (seeds 20-25; no training)

| task | cases | never retrieved | top-10% share of activation | Spearman(Q, activation) | evidenced cases without harm | Q evictions decided by case number | Jaccard of Q and exposure-only kept sets (retrieved cases) |
|---|---|---|---|---|---|---|---|
| abalone | 2505 | 0.98 | 1.00 | 0.64 | 0.02 | 0.99 | 0.78 |
| airfoil | 901 | 0.03 | 0.35 | 0.99 | 0.94 | 0.00 | 0.97 |
| balance | 375 | 0.13 | 0.36 | 0.99 | 0.93 | 0.00 | 0.97 |
| body_fat | 150 | 0.20 | 0.45 | 0.55 | 0.11 | 0.10 | 0.66 |
| breast_cancer | 341 | 0.84 | 0.90 | 1.00 | 0.97 | 1.00 | 0.91 |
| diabetes | 264 | 0.17 | 0.45 | 0.13 | 0.09 | 0.42 | 0.47 |
| digits | 1077 | 0.18 | 0.40 | 1.00 | 0.99 | 0.00 | 1.00 |
| energy_efficiency | 460 | 0.15 | 0.23 | 1.00 | 1.00 | 0.00 | 1.00 |
| iris | 90 | 0.45 | 0.56 | 0.92 | 0.86 | 0.50 | 0.93 |
| student_performance | 389 | 0.35 | 0.60 | 0.96 | 0.65 | 0.00 | 0.94 |
| anneal | 448 | 0.57 | 0.73 | 1.00 | 0.92 | 0.75 | 0.95 |
| concrete | 514 | 0.13 | 0.38 | 0.95 | 0.63 | 0.00 | 0.88 |
| credit-g | 499 | 0.12 | 0.43 | 0.90 | 0.47 | 0.00 | 0.83 |
| insurance | 669 | 0.18 | 0.49 | 0.93 | 0.44 | 0.00 | 0.85 |
| wine (classification) | 106 | 0.77 | 0.88 | 0.97 | 0.89 | 1.00 | 0.88 |
| yacht | 184 | 0.66 | 0.71 | 0.92 | 0.92 | 1.00 | 0.88 |
| zebra | 66 | 0.11 | 0.25 | 1.00 | 1.00 | 0.00 | 1.00 |
| zebra_special | 132 | 0.03 | 0.27 | 0.51 | 0.35 | 0.00 | 0.60 |

Reading of Table 9: on 14 of 18 tasks Q ranks cases almost exactly by exposure (Spearman at least 0.90). On abalone, breast cancer, wine, yacht and anneal most cases are never retrieved and most evictions are decided by case number among cases that hold almost no activation. So the pre-registered maintenance comparison (provenance score against random and bias at 50% capacity, not yet run) would largely test whether the most-retrieved cases are kept. Three safe changes were made, with defaults unchanged: per-condition maintenance settings in synchronized training (a verified bug), an optional seeded tie-break, and a fix to the kept-case IDs logged by maintenance (the logged churn metric only).

Proposed next steps, for PI decision: (1) a dated addendum that specifies the maintenance comparison's runner, tasks, endpoint and comparators (full memory, an exposure-only control) before any result exists, then run it as frozen; (2) an exploratory study on development seeds with label noise, near-duplicates, a rare subgroup, imbalance and larger sets, with label-aware credit variants and classical baselines (edited and condensed nearest neighbour, ICF, DROP3, Shapley-style valuation); (3) dynamic maintenance only after that.

## 6. Caveats

**Table 10.** Caveats on reading the evidence

| # | Caveat | Effect on the results |
|---|---|---|
| 1 | Seven small tasks use 30 seeds, the other 23 tasks use 10 | Each task enters the pooled analysis once, as pre-registered |
| 2 | Confirmation ran in three batches (26 tasks, three transfer tasks, QSAR-TID-11) whose code differs only in configuration files, a lock on the results-file append and added tooling; a per-task code fingerprint check passed for all 30 tasks | None on the numbers |
| 3 | One result row (yacht, alternating default, seed 64) was lost in a file-writing race between parallel jobs; it was rebuilt from that run's own saved files after checking that the run finished, not rerun. The audit then matched every run to its row (2,066 rows) | None |
| 4 | A machine interruption on 13 October stopped all runs; the abalone seeds 45-49 were rerun and their 16 duplicate rows are bitwise identical | None |
| 5 | The pre-registered large-n secondary group was not run (its feasibility check failed), and the transfer tasks are capped at 3,000 training rows for every method | Larger data, where maintenance should matter most, is untested |
| 6 | The two zebra tasks drive E3b's point estimate below 1 (Table 6) | Read E3b with Table 3 |
| 7 | The legacy-group p-values in Table 3 are unadjusted and the legacy tasks selected F | Descriptive only |
| 8 | In about 30% of F runs the selected checkpoint precedes any retrieval or joint update | F then equals the retrieval-only core (Table 6 reading) |
| 9 | The case-base maintenance step is not part of F and was not confirmed | Section 5 is a design review, not a result |
