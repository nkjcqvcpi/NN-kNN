# NN-kNN Neural CBR weekly report, 2026-11-24

Terms used throughout:
- NN-kNN is a neural nearest-neighbour model. It retrieves stored training cases and combines their answers with learned feature weights and case weights. CBR means case-based reasoning: retrieve, reuse (adapt), revise and retain.
- Masked softmax: the prediction is a softmax-weighted average over only the top-K highest-scoring cases (K = 5 unless stated). All-case softmax drops that mask and weights every stored case. tau is the softmax temperature.
- RO (retrieval only) is the NN-kNN core's prediction with no adaptation step. F is the frozen final configuration: the tuned core plus a learned adapter that corrects the retrieved answer, trained on the chosen schedule.
- MCB is a momentum case encoder. The stored case embeddings are a slow moving average (momentum 0.99) of the trained encoder.
- Error: 1 − accuracy for classification. For regression, test RMSE on the target standardised by the training set (std-RMSE). Lower is better unless a table says otherwise.
- GMR (geometric mean ratio): the geometric mean over tasks of NN-kNN error ÷ reference error. Below 1 favours NN-kNN.
- Every experiment used 3 seeds unless stated. All results below are exploratory: the seeds have been seen before, and nothing was pre-registered for this week's runs.

# Progress

- **PI's latest handoff.** We tested all-case vs masked softmax at a longer training budget, with tuned K and tau, on heavier datasets and inside F. We also re-ran the masked/all-case × joint-training pilot with twice the pretraining budget. All-case helps on several regression tasks and inside F on average, but tuning K for the masked model removes most of the gap.
- **Reproduction of F on 29 tasks.** We re-ran the frozen final configuration F and all its reference arms on the 29 tasks. This is a reproduction on seeds already used, not a confirmatory test. F ties retrieval only, beats kNN, and trails CatBoost.
- **Baselines and large-task grid.** We re-ran tuned kNN, CatBoost, MLP and TabM on every task, paired seed by seed with NN-kNN. We also ran the NN-kNN hyperparameter grid on the two large regression tasks (California housing and bike sharing).
- **Full-cycle experiments, reuse and synchronisation.** Reuse adapters, alternating (synchronised) training of retrieval and adapter, and MCB. All units are complete.
- **Full-cycle experiments, retention and diagnostics.** Seven case-retention policies at three memory budgets, MCB during retention, influence-based removal, retraining after removal, human-revision stress tests and correction-radius fallbacks. All units are complete.
- **Original implementation and core ablation.** The original NN-kNN trainer now runs as a baseline. The core ablation that was unfinished last time now runs, masked and all-case, on 12 of 13 tasks. Two longer NN-kNN-only searches (all-case and masked) are finished on 16 tasks.
- **Problem found on review: retention evidence rule and tie-breaking.** Under the masked core many cases are never retrieved, so they have no evidence, and ties between such cases were broken by case ID. We re-ran retention under masked/all-case × case-ID/random tie-break. The results are preliminary because they have not yet been independently checked.

# Experiments

## 1. PI's handoff: all-case vs masked softmax, joint-training pilot

| dataset | masked | all-case | change | seeds favouring | change at 120 epochs |
|---|---|---|---|---|---|
| linear_regression (synthetic) | 0.302 | 0.236 | −21.6% | all-case 3/3 | −26.8% |
| yacht | 0.134 | 0.131 | −2.45% | mixed (tie) | −51.3% |
| energy_efficiency | 0.0554 | 0.0701 | +26.5% | masked 3/3 | +62.3% |
| airfoil | 0.353 | 0.371 | +5.10% | masked 3/3 | +7.50% |
| student_performance | 0.515 | 0.415 | −19.4% | all-case 3/3 | −20.3% |
| mixture_two_linear_models (synthetic) | 0.839 | 0.794 | −5.39% | all-case 3/3 | −4.48% |
| redundant_features (synthetic) | 0.119 | 0.0899 | −24.2% | all-case 3/3 | −19.9% |
| iris | 0.0444 | 0.0111 | −75.0% | mixed (tie) | −75.0% |
| wine | 0.0463 | 0.00926 | −80.0% | mixed (tie) | −80.0% |
| breast_cancer | 0.0409 | 0.0322 | −21.4% | mixed (tie) | −28.6% |
| balance | 0.131 | 0.123 | −6.12% | mixed (tie) | −4.08% |
| digits | 0.0185 | 0.0185 | 0% | mixed (tie) | −26.1% |

Rows: 12 datasets (7 regression, 5 classification), retrieval only, K = 5, tau = 1.
Columns: test error of each arm after up to 400 epochs (patience 60), lower is better. "change" = all-case mean ÷ masked mean − 1, so negative favours all-case. "seeds favouring" = which arm wins on how many of 3 paired seeds. The last column is the same comparison at a 120-epoch budget. 3 seeds, all complete.

Joint-training pilot (adapter error after joint training, 600-epoch pretraining, 3 seeds):

| dataset | masked core, joint | all-case core, joint | retrieval-loss drift, masked | retrieval-loss drift, all-case |
|---|---|---|---|---|
| iris | 0.0111 | 0.0111 | +0.00663 | +0.000846 |
| wine | 0.0278 | 0.0185 | −0.00795 | −0.000780 |
| breast_cancer | 0.0380 | 0.0380 | +0.00285 | +0.000222 |
| yacht | 0.0703 | 0.0832 | −0.00383 | −0.00242 |
| energy_efficiency | 0.0523 | 0.0604 | −0.0000100 | −0.000643 |
| student_performance | 0.501 | 0.420 | +0.0137 | 0 |

Drift is the change in retrieval loss caused by joint training; positive means joint training made retrieval worse.

With fixed K = 5, all-case wins on all 3 seeds on 4 regression tasks and loses on all 3 seeds on energy_efficiency and airfoil. The classification sets are ties because their test sets are tiny: on iris one error is 0.033. The longer budget shrinks the earlier gap (yacht −51% at 120 epochs becomes a tie at 400). At 400 epochs, 17 of 36 all-case runs still used the whole budget, against 1 of 36 masked runs, so the all-case arm may not have converged. When each arm picks its own K and tau on validation loss, most gaps shrink to a few percent. The masked arm most often picks K = 40 (16 of 36 seed picks), which is close to all-case. The consistent differences that remain are all-case better on yacht (−17%, 3/3) and airfoil (−3.0%, 3/3), and masked better on energy_efficiency (+18.6%, 3/3). So the fixed K = 5 comparison overstated the case for all-case. Its practical value is that it removes K as a hyperparameter at little cost. It is not a broad accuracy gain. On heavier datasets the differences are small: abalone −1.83% and regression wine −3.50% (both 3/3), car +4.28% (mixed).
Inside F (all-case core swapped into the F configuration, K and tau not re-tuned, seeds 40-42), all-case has the lower mean on 19 of 26 complete tasks for retrieval only and 14 of 25 for F. The ratio of task means is 0.792 for retrieval only and 0.863 for F (geometric mean over tasks). F with an all-case core loses on all 3 seeds on energy_efficiency, yacht, coupon and students_dropout, and car shows one seed where the adapter blew up (RMSE 1.02 against 0.241 before adaptation). Several all-case cores stop at the 300-epoch cap, so under-training may bias this comparison against all-case. That is not tested. In the joint-training pilot, all-case changes retrieval loss less during joint training (|drift| ≤ 0.0026 against up to 0.0137 masked), which fits the PI's argument, but the adapted error is mixed: all-case is better on student_performance and wine, masked is better on yacht and energy_efficiency. Even with 600 pretraining epochs, 6 of 18 all-case pretraining runs still hit the cap. "All-case protects retrieval during joint training" is a plausible but unproven observation.

## 2. Reproduction of the frozen final configuration F (29 tasks)

| task | RO | F before adapter | F | alternating default | kNN | CatBoost | TabM |
|---|---|---|---|---|---|---|---|
| iris (cls) | 0.0778 | 0.0667 | 0.0333 | 0.0667 | 0.0556 | 0.0778 | 0.0667 |
| wine (cls) | 0.0370 | 0.0741 | 0.0648 | 0.0370 | 0.0185 | 0.0278 | 0.0185 |
| breast_cancer (cls) | 0.0234 | 0.0205 | 0.0292 | 0.0292 | 0.0234 | 0.0234 | 0.0205 |
| balance (cls) | 0.0667 | 0.0640 | 0.0480 | 0.0693 | 0.109 | 0.0907 | 0.0427 |
| digits (cls) | 0.0222 | 0.0231 | 0.0241 | 0.0231 | 0.0241 | 0.0278 | 0.0204 |
| zebra (cls) | 0.318 | 0.182 | 0.182 | 0.333 | 0.242 | 0.000 | 0.409 |
| zebra_special (cls) | 0.280 | 0.273 | 0.303 | 0.258 | 0.288 | 0.0530 | 0.462 |
| energy_efficiency (reg) | 0.0504 | 0.0492 | 0.0480 | 0.0515 | 0.133 | 0.0419 | 0.0486 |
| yacht (reg) | 0.133 | 0.104 | 0.0563 | 0.118 | 0.520 | 0.0450 | 0.0440 |
| airfoil (reg) | 0.214 | 0.219 | 0.215 | 0.213 | 0.395 | 0.213 | 0.226 |
| student_performance (reg) | 0.441 | 0.451 | 0.439 | 0.448 | 0.655 | 0.402 | 0.411 |
| abalone (reg) | 0.702 | 0.693 | 0.690 | 0.694 | 0.740 | 0.717 | 0.691 |
| body_fat (reg) | 0.625 | 0.625 | 0.627 | 0.628 | 0.665 | 0.624 | 0.792 |
| diabetes (reg; F = adapter on frozen core) | 0.766 | 0.766 | 0.769 | 0.769 | 0.771 | 0.779 | 0.736 |
| car (reg) | 0.227 | 0.233 | 0.244 | 0.242 | 0.193 | 0.161 | 0.285 |
| wine (reg) | 0.750 | 0.753 | 0.753 | 0.754 | 0.736 | 0.736 | 0.760 |
| anneal (cls) | 0.0189 | 0.00668 | 0.00890 | 0.0189 | 0.0245 | 0.00667 | 0.00557 |
| credit-g (cls) | 0.265 | 0.264 | 0.284 | 0.290 | 0.261 | 0.242 | 0.251 |
| concrete (reg) | 0.422 | 0.423 | 0.413 | 0.416 | 0.543 | 0.281 | 0.330 |
| insurance (reg) | 0.411 | 0.407 | 0.396 | 0.406 | 0.490 | 0.380 | 0.384 |
| splice (cls, transfer) | 0.0382 | 0.0379 | 0.0373 | 0.0398 | 0.0981 | 0.0423 | 0.0411 |
| students_dropout (cls, transfer) | 0.248 | 0.241 | 0.224 | 0.228 | 0.284 | 0.222 | 0.214 |
| coupon (cls, transfer) | 0.320 | 0.336 | 0.303 | 0.299 | 0.318 | 0.265 | 0.272 |
| churn (cls, transfer) | 0.0714 | 0.0714 | 0.0728 | 0.0726 | 0.112 | 0.0456 | 0.0624 |
| MIC (cls, transfer) | 0.122 | 0.122 | 0.128 | 0.119 | 0.131 | 0.112 | 0.112 |
| superconductivity (reg, transfer) | 0.385 | 0.385 | 0.376 | 0.379 | 0.391 | 0.347 | 0.358 |
| miami_housing (reg, transfer) | 0.356 | 0.357 | 0.349 | 0.354 | 0.372 | 0.310 | 0.325 |
| diamonds (reg, transfer) | 0.173 | 0.171 | 0.189 | 0.193 | 0.252 | 0.168 | 0.188 |
| protein (reg, transfer) | 0.696 | 0.696 | 0.703 | 0.707 | 0.752 | 0.696 | 0.693 |

Rows: the 29 pre-registered tasks. The first 20 are the original ("legacy") tasks; the 9 transfer tasks are OpenML tasks with the training set capped at 3,000 rows.
Columns: test error, mean over seeds 40-42 (3/3 everywhere), lower is better. "F before adapter" is F's jointly trained core on its own. "Alternating default" is the alternating training schedule at F's budget. kNN, CatBoost and TabM are tuned baselines on the same seeds.

Pre-registered endpoints, computed descriptively (29 tasks, GMR of F ÷ reference with 95% interval):

| F vs | GMR [95% CI] | Holm-adjusted p | F better / tie / worse |
|---|---|---|---|
| retrieval only | 0.990 [0.972, 1.01] | 0.329 | 15 / 6 / 8 |
| alternating default | 0.968 [0.918, 1.02] | 0.329 | 14 / 8 / 7 |
| CatBoost (non-inferiority, margin 10%) | 1.08 [0.974, 1.20] | 0.371 | 7 / 4 / 18 |
| TabM (non-inferiority, margin 10%) | 1.01 [0.941, 1.09] | 0.0550 | 8 / 3 / 18 |

F ties retrieval only (GMR 0.990, interval spans 1), so the adapter adds no detectable gain on top of tuned cores. It also ties the alternating schedule. The adapter clearly helps on yacht (F 0.0563 vs RO 0.133; per-seed ratio of F to F-before-adapter 0.41-0.77) and balance (0.70-0.80), but even on yacht F does not reach CatBoost or TabM. It hurts on car (0.244 vs 0.227, and kNN at 0.193 beats both), on diamonds seed 41 (ratio 1.31), on credit-g, on MIC (1.01-1.09) and on small classification sets. The diamonds seed reproduces bit for bit in a rerun, so it is a real property of that seed and not a mixed-code artefact. The same blow-ups appear in the alternating default arm, so the instability belongs to the learned correction on these large regression case bases, not to F's schedule. Why those seeds diverge is not known. Unfavourable to NN-kNN: F trails CatBoost by 8.2% on average and is worse on 18 of 29 tasks, so non-inferiority to CatBoost is not shown. Against TabM it is a tie pooled, but the adjusted non-inferiority p of 0.055 just misses 0.05, and on the transfer tasks F is worse than TabM (GMR 1.06, CI 1.02-1.09). F beats kNN on 22 of 29 tasks by mean (ties on digits). Mean ranks across six methods: CatBoost 2.28, TabM 2.60, F 3.55, RO 3.79, MLP 3.93, kNN 4.84. Calibration improves (median calibration error 0.0311 vs 0.0463 before adaptation), but the flip criterion fails: the adapter turns 1,900 wrong predictions right and 1,439 right predictions wrong (ratio 1.32, the requirement was 2). This says nothing for or against the earlier confirmatory result, because the seeds were reused and only 3 of 10 evaluation seeds ran.

## 3. Baselines and the NN-kNN grid on the two large regression tasks

| dataset | kNN | CatBoost | MLP | TabM | NN-kNN core (RO) | NN-kNN F | core W/T/L vs kNN/CatBoost/MLP/TabM |
|---|---|---|---|---|---|---|---|
| balance (cls) | 0.109 | 0.0907 | 0.0213 | 0.0427 | 0.0667 | 0.0480 | W W L L |
| breast_cancer (cls) | 0.0234 | 0.0234 | 0.0146 | 0.0205 | 0.0234 | 0.0292 | T T T T |
| digits (cls) | 0.0241 | 0.0278 | 0.0222 | 0.0204 | 0.0222 | 0.0241 | T T T T |
| iris (cls) | 0.0556 | 0.0778 | 0.0556 | 0.0667 | 0.0778 | 0.0333 | T T T T |
| wine (cls) | 0.0185 | 0.0278 | 0.0278 | 0.0185 | 0.0370 | 0.0648 | T T T T |
| zebra (cls) | 0.242 | 0 | 0.470 | 0.409 | 0.318 | 0.182 | T T T T |
| zebra_special (cls) | 0.288 | 0.0530 | 0.447 | 0.462 | 0.280 | 0.303 | T L W W |
| anneal (cls) | 0.0245 | 0.00667 | 0.0100 | 0.00557 | 0.0189 | 0.00890 | T L T T |
| credit-g (cls) | 0.261 | 0.242 | 0.258 | 0.251 | 0.265 | 0.284 | T L T T |
| churn (cls) | 0.112 | 0.0456 | 0.0758 | 0.0624 | 0.0714 | 0.0728 | W L T T |
| coupon (cls) | 0.318 | 0.265 | 0.292 | 0.272 | 0.320 | 0.303 | T L L L |
| splice (cls) | 0.0981 | 0.0423 | 0.0542 | 0.0411 | 0.0382 | 0.0373 | W W W T |
| students_dropout (cls) | 0.284 | 0.222 | 0.223 | 0.214 | 0.248 | 0.224 | W L L L |
| MIC (cls) | 0.131 | 0.112 | 0.119 | 0.112 | 0.122 | 0.128 | T T T L |
| abalone (reg) | 0.740 | 0.717 | 0.697 | 0.691 | 0.702 | 0.690 | W T L L |
| airfoil (reg) | 0.395 | 0.213 | 0.216 | 0.226 | 0.214 | 0.215 | W T T W |
| body_fat (reg) | 0.665 | 0.624 | 0.790 | 0.792 | 0.625 | 0.627 | W T W W |
| car (reg) | 0.193 | 0.161 | 0.397 | 0.285 | 0.227 | 0.244 | L L T T |
| diabetes (reg) | 0.771 | 0.779 | 0.793 | 0.736 | 0.766 | 0.769 | T T W L |
| energy_efficiency (reg) | 0.133 | 0.0419 | 0.0528 | 0.0486 | 0.0504 | 0.0480 | W L T T |
| student_performance (reg) | 0.655 | 0.402 | 0.457 | 0.411 | 0.441 | 0.439 | W L T L |
| wine (reg) | 0.736 | 0.736 | 0.785 | 0.760 | 0.750 | 0.753 | L L W W |
| yacht (reg) | 0.520 | 0.0450 | 0.0763 | 0.0440 | 0.133 | 0.0563 | W L T L |
| concrete (reg) | 0.543 | 0.281 | 0.342 | 0.330 | 0.422 | 0.413 | W L L L |
| insurance (reg) | 0.490 | 0.380 | 0.405 | 0.384 | 0.411 | 0.396 | W L T L |
| diamonds (reg) | 0.252 | 0.168 | 0.241 | 0.188 | 0.173 | 0.189 | W L T T |
| miami_housing (reg) | 0.372 | 0.310 | 0.332 | 0.325 | 0.356 | 0.349 | W L L L |
| protein (reg) | 0.752 | 0.696 | 0.715 | 0.693 | 0.696 | 0.703 | W T W T |
| superconductivity (reg) | 0.391 | 0.347 | 0.367 | 0.358 | 0.385 | 0.376 | W L L L |

Rows: the 29 tasks, classification then regression.
Columns: test error, mean over paired seeds 40-42 (3/3 everywhere), lower is better. Each baseline is its family's grid-selected configuration, chosen without the test split. W/T/L compares the NN-kNN core with each baseline: W = NN-kNN better on all 3 paired seeds, L = worse on all 3, otherwise T.

| comparison | W/T/L, all 29 tasks | GMR |
|---|---|---|
| core vs kNN | 16 / 11 / 2 | 0.831 |
| core vs CatBoost | 2 / 11 / 16 | 1.23 (28 tasks, zebra left out) |
| core vs MLP | 6 / 16 / 7 | 1.05 |
| core vs TabM | 4 / 13 / 12 | 1.13 |
| F vs kNN | 18 / 8 / 3 | 0.764 |
| F vs CatBoost | 3 / 12 / 14 | 1.15 (28 tasks) |
| F vs MLP | 7 / 16 / 6 | 0.966 |
| F vs TabM | 3 / 14 / 12 | 1.04 |

NN-kNN clearly improves on plain kNN, mostly on regression (12/1/2, GMR 0.755). By absolute gap the largest gains are yacht (0.133 vs 0.520), student_performance, airfoil and concrete. The data do not show why. Unfavourable to NN-kNN: CatBoost is the strongest baseline. The core loses to it on 16 tasks and wins only on balance and splice, and on regression it never wins (0/5/10). NN-kNN's error is about 15-23% higher on average. The largest losses (core ÷ CatBoost) are yacht 2.97×, anneal 2.84×, churn 1.57× and concrete 1.50×. The losses span small physical regression sets and larger tabular sets. This gap measures NN-kNN as currently tuned, with masked cores; whether all-case cores or a longer search narrow it is not tested here. Against TabM, NN-kNN loses on 12 tasks: 7 of them are OpenML tasks, several of them not large, plus balance, abalone, diabetes, student_performance and yacht. Against the MLP the result is balanced. NN-kNN beats both neural baselines on body_fat and zebra_special (MLP/TabM errors 0.790/0.792 and 0.447/0.462). F differs from the core on yacht mainly in turning the loss against CatBoost into a tie, and the only classification task where F is worse than the core on all 3 seeds is MIC. Many ties come from small test sets: zebra has 22 test cases and a seed standard deviation up to 0.318.
Large regression grid (NN-kNN only, 32 candidates screened on 2 development seeds, 3 finalists refit, winner evaluated on 5 seeds): California housing is finished, with test std-RMSE 0.440 ± 0.0152 (5/5 seeds). Its finalists are within 0.0123 of each other on validation, so the choice among them is within noise. Bike sharing has selected its winner (MLP representation; finalists within 0.0023), and its test evaluation is 0/5, still running. No tuned baselines exist yet for these two tasks, so we cannot say whether 0.440 is good.

## 4. Full-cycle exploratory experiments: reuse, synchronisation, MCB

| dataset | independent, RO | independent, F | alternating | alternating + pruning during training |
|---|---|---|---|---|
| *classification, accuracy (higher is better)* | | | | |
| balance | 0.939 | 0.939 | 0.936 | 0.936 |
| breast_cancer | 0.962 | 0.959 | 0.965 | 0.977 (3/3 better) |
| digits | 0.980 | 0.979 | 0.978 | 0.979 |
| iris | 0.944 | 0.944 | 0.944 | 0.933 |
| synthetic_diag | 0.949 | 0.935 | 0.947 | 0.942 |
| anneal | 0.989 | 0.990 | 0.986 | 0.983 |
| credit-g | 0.712 | 0.716 | 0.704 | 0.716 |
| wine | 0.926 | 0.926 | 0.898 (worse) | 0.907 |
| zebra | 0.939 | 0.939 | 0.924 | 0.894 |
| zebra_special | 0.659 | 0.697 | 0.659 | 0.689 |
| *regression, RMSE (lower is better)* | | | | |
| abalone | 0.833 | 0.828 | 0.776 (3/3 better) | 0.677 (3/3 better) |
| airfoil | 0.235 | 0.238 | 0.236 (3/3 better) | 0.238 |
| body_fat | 0.585 | 0.945 | 0.606 | 1.15 |
| car | 0.336 | 0.320 | 0.364 (0/3) | 0.294 |
| diabetes | 0.729 | 0.733 | 0.719 | 0.717 |
| energy_efficiency | 0.0786 | 0.0777 | 0.0770 | 0.0541 (3/3 better) |
| student_performance | 0.344 | 0.350 | 0.359 | 0.364 |
| concrete | 0.367 | 0.370 | 0.369 | 0.363 (3/3 better) |
| insurance | 0.408 | 0.408 | 0.392 (3/3 better) | 0.392 |
| wine (reg) | 0.759 | 0.767 | 0.752 (better) | 0.757 (3/3 better) |
| yacht | 0.126 | 0.0938 | 0.0966 | 0.0820 (3/3 better) |

Rows: 21 datasets, with the tuned per-dataset masked cores.
Columns: test metric, mean over 3 seeds. "Independent" trains retrieval first, then the adapter with retrieval frozen; RO is before the adapter, F after it. "Alternating" updates retrieval and adapter in alternating blocks. "Alternating + pruning during training" also removes cases at epochs 40 and 80, so the model trains on the reduced case base. Every arm ends with 75% of the cases. In the first two arms they are removed after training with no fine-tuning. Reference column: independent, F. "better/worse" without a count means the paired 95% interval excludes 0; "k/3" counts seeds that moved in one direction (marked on selected cells only). Every other difference is a tie.

On classification, alternating training gives no gain (−1.1% accuracy on average; wine worse on all seeds). On regression, alternating training with pruning during training is the most consistent trend in this part: −4.8% RMSE on average, 25 of 33 seed pairs better, 3/3 better on abalone (−18%), energy_efficiency (−30%), yacht (−12%), concrete and regression wine. No single dataset passes the interval test. The gain may come from training on the pruned memory rather than from synchronisation. An arm that prunes during training under the independent schedule would separate the two, and none was run. body_fat is unstable (single-seed RMSE of 1.3-1.5). When retrieval is trained for the post-adaptation loss, retrieval on its own can get much worse (car with an MLP encoder: retrieval RMSE 0.308 → 0.821). Removing single loss terms or changing the radius rule changes nothing (146 of 147 cells tie).
Reuse adapter on a frozen core: on classification it ties retrieval on all 10 datasets. Retrieval alone is already at 0.93-0.99 accuracy on most of them. On zebra, accuracy is identical in every arm, and only the logit-based adapter changes any prediction (2 of 22 on one seed). On regression it ties overall (13 seed pairs better, 20 worse), with body_fat worse on all seeds (+27.5%) and yacht better (−26.4%, 2 of 3 seeds). body_fat has only about 150 training cases (51 test queries); over-correction on a small case base is a hypothesis, not tested. On the synthetic nominal dataset built for this purpose, the nominal-residual adapter roughly doubles accuracy (0.370 → 0.759, all 3 seeds) and beats the logit-residual adapter (0.674). The whole gain is on queries whose categories were seen in training (0.381 → 0.958). Queries with unseen or missing categories show no reliable gain. This is the clearest positive result in this part, but it is synthetic and does not carry over to the real classification sets.
MCB on vs off (same MLP encoder, alternating schedule): MCB lowers embedding drift by about 34% (significant on 5 regression datasets). It does not lower neighbourhood churn, i.e. how often the retrieved neighbours change. It does not change accuracy or RMSE (ties, except breast_cancer and synthetic_diag better). So MCB makes the stored representations steadier without steadying what gets retrieved, and without a task gain. "MCB improves NN-kNN" is not supported.

## 5. Full-cycle exploratory experiments: retention, case maintenance, diagnostics

| dataset | full memory | random | kcenter | coverage | bias_current | bias_normalized | provenance_only |
|---|---|---|---|---|---|---|---|
| abalone | 0.685 | 0.689 (+0.6%) | 0.684 (−0.1%) | 0.683 (−0.2%) | 0.685 (+0.0%) | 0.684 (−0.1%) | 0.679 (−0.9%*) |
| airfoil | 0.231 | 0.253 (+9.9%*) | 0.246 (+6.7%) | 0.240 (+3.9%) | 0.239 (+3.9%) | 0.237 (+2.8%) | 0.242 (+4.8%) |
| body_fat | 0.577 | 0.583 (+1.2%) | 0.585 (+1.5%) | 0.573 (−0.6%) | 0.579 (+0.5%) | 0.577 (+0.0%) | 0.589 (+2.1%) |
| car | 0.244 | 0.275 (+12.4%) | 0.243 (−0.5%) | 0.423 (+73.1%*) | 0.244 (−0.3%) | 0.273 (+11.8%) | 0.443 (+81.3%*) |
| diabetes | 0.707 | 0.690 (−2.4%) | 0.702 (−0.6%) | 0.700 (−0.9%) | 0.690 (−2.4%) | 0.683 (−3.3%) | 0.699 (−1.1%) |
| energy_efficiency | 0.0525 | 0.0778 (+48.0%) | 0.0650 (+23.7%) | 0.132 (+151%) | 0.0599 (+14.1%) | 0.0593 (+12.9%) | 0.0681 (+29.7%) |
| student_performance | 0.351 | 0.363 (+3.5%) | 0.343 (−2.2%) | 0.349 (−0.6%) | 0.353 (+0.6%) | 0.350 (−0.2%) | 0.350 (−0.4%) |
| concrete | 0.370 | 0.376 (+1.7%) | 0.361 (−2.4%) | 0.368 (−0.5%) | 0.370 (−0.0%) | 0.369 (−0.1%) | 0.367 (−0.9%) |
| insurance | 0.401 | 0.404 (+1.0%) | 0.401 (+0.1%) | 0.396 (−1.1%) | 0.402 (+0.3%) | 0.401 (+0.2%) | 0.397 (−0.8%) |
| wine (reg) | 0.728 | 0.751 (+3.2%) | 0.751 (+3.2%) | 0.772 (+6.1%*) | 0.758 (+4.2%*) | 0.760 (+4.5%*) | 0.759 (+4.2%*) |
| yacht | 0.128 | 0.156 (+21.6%) | 0.118 (−7.8%) | 0.128 (+0.1%) | 0.128 (+0.1%) | 0.128 (+0.1%) | 0.130 (+1.6%) |
| mean change vs full memory | ref | +9.2% | +2.0% | +21.0% | +1.9% | +2.6% | +10.9% |

Rows: 11 regression datasets, keeping 50% of the cases (K = 0.5).
Columns: test RMSE after case removal and a 30-epoch fine-tune, mean of 3 seeds, lower is better. The percentage is the paired change against full memory (all cases kept, same fine-tune); `*` means the paired 95% interval excludes 0. Policies: random; kcenter (spread kept cases over the feature space); coverage (keep cases that cover the most reachable neighbours); bias_current and bias_normalized (keep cases with the highest learned case weight, raw or normalised); provenance_only (keep cases with the best retrieval track record). A stratified policy behaved like random and is left out.

Mean change against full memory over datasets, with better/tie/worse counts against random:

| task, budget | random | kcenter | coverage | bias_current | bias_normalized | provenance_only |
|---|---|---|---|---|---|---|
| regression, keep 20% | +43.7% | +11.2% [3/8/0] | +25.3% [2/9/0] | +15.0% [3/8/0] | +15.9% [0/11/0] | +28.0% [1/10/0] |
| regression, keep 50% | +9.2% | +2.0% [0/11/0] | +21.0% [2/7/2] | +1.9% [0/11/0] | +2.6% [1/10/0] | +10.9% [1/9/1] |
| regression, keep 75% | +3.4% | +0.2% [0/11/0] | +5.6% [0/10/1] | +1.0% [0/11/0] | +1.3% [0/11/0] | +6.0% [0/11/0] |
| classification, keep 20% | −5.3% | −6.3% [0/9/1] | −7.6% [0/10/0] | −5.1% [0/10/0] | −3.8% [0/10/0] | −7.2% [0/9/1] |

On classification (10 datasets, accuracy), halving memory costs about 0-2% on the larger sets, and no policy separates from full memory or from random at K = 0.5. zebra loses 4.8-9.7% under every policy. On regression, kcenter and the learned-weight policies stay within about 2-3% of full memory on average at K = 0.5, while random loses about 9%. Most single-dataset differences are still ties. The clearest favourable result is at K = 0.2: kcenter and bias_current beat random on 3 datasets each and are never worse. bias_normalized is not separable from random there. Unfavourable: provenance_only and coverage are much worse on car (+81% and +73%) and worse on regression wine. A likely contributor is that, under the masked core, many cases have no retrieval evidence (only 4% of abalone cases and 61% of car cases have any), so part of the provenance choice is a tie-break. Section 7 tests this.
MCB during retention lowers representation drift strongly (classification 0.741 → 0.255) but does not lower neighbourhood churn. It costs accuracy: −5.9% to −7.0% on classification and +11-14% RMSE on regression on average. Significant losses are on zebra (provenance), zebra_special (random), energy_efficiency (both) and credit-g (provenance). Yacht (+49-90%) and car (+22-24%) are large but not significant, and one significant gain is on classification wine. Influence-based removal on a 60-case memory is about 3% better than random on mean regression RMSE but never separates from random at 3 seeds. Its cheap cached approximation gives close but not identical test results. Retraining after removal (24-case memory) buys under 1% on average. On an 18-case memory the adapter step makes things slightly worse (pooled regression RMSE +0.020); 12-18 cases may be too few to train an adapter, but this is not tested. In the human-revision stress test (7 synthetic datasets), choosing cases to review by retrieval track record finds corrupted cases more often than random on 5 of 7 sets with a 10-case budget and on all 7 with 30 (recall 0.114 vs 0.077 at 30). The learned case weight alone is a poor detector: it is worse than random on all 7 sets at 30. Corrections move accuracy little: even a perfect reviewer gains only 0.010-0.059 at a 30-case budget. For the free-correction radius, all four fallback estimators tie the default on all 13 datasets, so the default is adequate.

## 6. Reconstruction of the original implementation and core ablation

| dataset | searched all-case core | searched masked core | earlier masked winner | tuned kNN (unpaired) |
|---|---|---|---|---|
| iris (cls) | 0.0444 | 0.0556 | n/a | 0.0533 |
| wine (cls) | 0.0278 | 0.0370 | 0.0741 | 0.0500 |
| breast_cancer (cls) | 0.0292 | 0.0409 | 0.0468 | 0.0360 |
| balance (cls) | 0.0133 | 0.0613 | 0.0560 | 0.104 |
| zebra (cls) | 0.394 | 0.0606 | 0.0606 | 0.250 |
| zebra_special (cls) | 0.220 | 0.386 | 0.311 | 0.282 |
| anneal (cls) | 0.0123 | 0.0223 | n/a | 0.0247 |
| MIC (cls) | 0.126 | 0.131 | n/a | 0.135 |
| credit-g (cls) | 0.252 | 0.282 | 0.288 | 0.269 |
| energy_efficiency (reg) | 0.0478 | 0.0520 | 0.0526 | 0.126 |
| yacht (reg) | 0.122 | 0.123 | 0.128 | 0.618 |
| body_fat (reg) | 0.570 | 0.577 | n/a | 0.696 |
| diabetes (reg) | 0.680 | 0.719 | n/a | 0.743 |
| concrete (reg) | 0.343 | 0.370 | 0.370 | 0.538 |
| student_performance (reg) | 0.300 | 0.352 | 0.352 | 0.649 |
| insurance (reg) | 0.375 | 0.400 | 0.400 | 0.491 |

Rows: the 16 tasks where both new NN-kNN-only searches have finished.
Columns: test error (lower is better), mean of seeds 30-32. Each search screened 32 candidates on 2 development seeds with a longer budget (150 epochs at screening, 600 at refit). "Earlier masked winner" is the masked configuration chosen by the earlier grid and used in the ablation. Tuned kNN uses 10 different seeds (40-49), so that column is unpaired and indicative only.

The longer masked search alone does not help: it ties the earlier masked winner on 10 of 11 shared tasks and is worse on zebra_special, often picking the same configuration. The all-case search is better than the masked one on 7 of 16 tasks (balance, diabetes, energy_efficiency, concrete, credit-g, student_performance, insurance), ties on 8 and is worse on zebra (0.394 vs 0.0606 on every seed). On zebra the all-case search did not include the low temperature (tau 0.1) that fixes zebra in the ablation. The searched all-case core has a lower mean than tuned kNN on 15 of 16 tasks (all except zebra), and is below kNN by more than kNN's seed spread on 8. So "a properly searched NN-kNN core beats kNN" holds on most tasks, but a paired check needs kNN on the same seeds, which does not exist yet.
Masked one-factor ablation (12 of 13 tasks complete; car still running): removing the mask is better on 6 of 12 tasks (gains 0.0175-0.0468) and worse on zebra_special (+0.212). The variant chosen by validation has lower test error than the earlier winner on 11 of 12 tasks, so the earlier winner was not well tuned. A longer budget (1,000 epochs) ties everywhere, because early stopping ends training before the cap, so budget is not the limit. Freezing the learned feature weights hurts on 6 tasks (energy_efficiency +0.195, yacht +0.433). An untrained soft kernel is worse on 8 tasks, including all 5 complete regression tasks. Freezing the case weights matters little (worse only on yacht). Under all-case the picture is similar. A bandwidth-tuned untrained kernel is far worse on every regression task (concrete 0.813 vs 0.333), so training does real work beyond choosing a kernel width. Feature weights matter most. The value of training the case weights depends on the task (worse when frozen on 3 tasks, better on 2). Tuning tau by validation fixes zebra (tau 0.1 gives 0) and zebra_special (0.523 → 0.311), helps balance and yacht, and is slightly worse on breast_cancer and digits.
The original trainer (default masked top-5, tau 1, raw features, seeds 0-2) runs to completion on 12 datasets. It is clearly poor on zebra (0.667) and zebra_special (0.523). These runs cannot show whether the old published numbers are reproduced, because no matching-seed reference exists.

## 7. Problem found on review: retention evidence rule and tie-breaking (preliminary)

| dataset | masked, case-ID tie-break | masked, random tie-break | all-case, case-ID tie-break | all-case, random tie-break | evidenced share, masked / all-case |
|---|---|---|---|---|---|
| iris (acc) | +0.0222 (1/1) | +0.0222 (1/1) | 0 (0/0) | 0 (0/0) | 0.800 / 0.859 |
| wine (acc) | −0.00926 (0/1) | −0.00926 (0/1) | +0.00926 (1/0) | +0.00926 (1/0) | 0.748 / 0.736 |
| breast_cancer (acc) | +0.0175 (3/0) | +0.0175 (3/0) | +0.00585 (2/1) | +0.00585 (2/1) | 0.488 / 0.751 |
| balance (acc) | +0.0133 (2/1) | +0.0133 (2/1) | −0.00533 (1/2) | −0.00533 (1/2) | 0.627 / 0.509 |
| digits (acc) | +0.00370 (2/1) | +0.00370 (2/1) | +0.000926 (1/1) | +0.000926 (1/1) | 0.622 / 0.825 |
| yacht (RMSE) | −0.00636 (3/0) | −0.00777 (2/1) | +0.0177 (0/3) | +0.0177 (0/3) | 0.391 / 1.00 |
| energy_efficiency (RMSE) | +0.0155 (1/2) | +0.0155 (1/2) | −0.0253 (3/0) | −0.0253 (3/0) | 0.809 / 0.999 |
| airfoil (RMSE) | −0.00916 (2/1) | −0.00916 (2/1) | −0.0312 (3/0) | −0.0312 (3/0) | 0.705 / 0.971 |
| student_performance (RMSE) | −0.00959 (2/1) | −0.00918 (2/1) | −0.00292 (2/1) | −0.00292 (2/1) | 0.432 / 0.933 |
| abalone (RMSE) | +0.00786 (1/2) | +0.00633 (0/3) | +0.0301 (0/3) | +0.0301 (0/3) | 0.121 / 0.990 |

Rows: 10 datasets, keeping 50% of cases followed by a 30-epoch fine-tune.
Columns: paired difference provenance_only − random, mean over seeds 30-32 (3/3 complete), then (seeds where provenance_only is better / seeds where it is worse). For accuracy, positive favours provenance_only; for RMSE, negative favours it. The last column is the share of training cases with retrieval evidence (retrieved at least twice with at least 0.1 of the weight). These numbers have not yet been independently verified.

The tie-break barely matters. On classification every result is identical under case-ID and random tie-breaking, even though the kept case sets differ (breast_cancer seed 32: the two rules share only 91 of 170 kept cases). On regression it moves provenance_only by at most 0.0015 RMSE. The missing-evidence problem is real under the masked core. Only 12.1% of abalone cases have evidence, and provenance_only then fills 81.5% of its kept slots by the tie rule. With all-case softmax, 93-100% of regression cases have evidence and no slot is decided by a tie, so all-case removes this problem. Even so, provenance_only is not consistently better than random. Under all-case it is better on energy_efficiency and airfoil (3/3) but worse on yacht and abalone (3/3). The sign flips between masked and all-case on energy_efficiency and yacht. Part of that may come from the different trained cores rather than the evidence rule, since full-memory RMSE also changes (student_performance 0.396 masked vs 0.303 all-case). Unfavourable: no policy beats full memory beyond noise in either regime. On car (partial heavier run), provenance_only is worse than random on all 3 seeds in both regimes (+0.0882 masked, +0.0962 all-case), with no tie-decided slots, so the ranking itself is the problem there. The learned case weight (bias_current) is never worse than provenance_only beyond noise, so the evidence rule adds nothing beyond the learned weight in these data.

# TODO

Running now:
- The all-case version of F on the 29 tasks.
- K and temperature sweeps, and joint-training pilots with longer pretraining, on more datasets.
- NN-kNN-only searches (all-case, and masked with a longer budget) on the small and 3,000-row tasks.
- Core ablation on 7 more tasks (36 masked variants, 16 all-case variants).
- Retention rule and tie-break tests on the heavier datasets.
- Test evaluation of the bike sharing grid winner.

Planned, or waiting for the PI:
- A declared all-case training schedule, with matched retrieval-only and retrieval+adapter baselines.
- The addendum for the pre-registered retention test (step G).
- Feature standardisation experiments.
- Further comparators such as TabPFN.
- Core ablation on the two large regression tasks.
- The human revision study.
- The QSAR task (postponed).
