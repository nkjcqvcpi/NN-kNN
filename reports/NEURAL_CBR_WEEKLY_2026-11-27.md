# NN-kNN Neural CBR weekly report, 2026-11-27

Covers the work since the 2026-11-24 report: the two handoff batches of 2026-11-25, the packs p0-p4 and p0a of 2026-11-26,
the fixes that followed, and the summaries finished on 2026-11-27.

Terms used throughout:
- NN-kNN is a neural nearest-neighbour model. It retrieves stored training cases and combines their answers with learned feature weights and case weights (the case "bias"). CBR means case-based reasoning: retrieve, reuse (adapt), revise and retain.
- Masked softmax: the prediction is a softmax-weighted average over only the top-K highest-scoring cases. All-case softmax drops that mask and weights every stored case.
- F is the frozen final configuration: the tuned core plus a learned adapter that corrects the retrieved answer. RO (retrieval only) is the core's prediction with no adapter.
- plan_full is the recorded masked study of the full-cycle experiments (retention, synchronisation, reuse, revision and so on). An "all-case twin" repeats a plan_full condition with only the mask switched off.
- Retention keeps a fraction K of the stored cases (K = 0.2 keeps 20%) and then fine-tunes 30 epochs, unless stated. provenance_only keeps the cases with the best retrieval track record; bias_current and bias_normalized keep the cases with the highest learned case weight.
- Error: 1 − accuracy for classification; test RMSE on the standardised target for regression. Lower is better unless a table says otherwise.
- GMR (geometric mean ratio): the geometric mean over tasks of error A ÷ error B. Below 1 favours A.
- W/T/L (or b/t/w): tasks where the first method wins, ties or loses. Each section states its tie rule. With 3 seeds these are consistency counts, not significance tests.
- Everything below is exploratory. Every value not taken from the PI's handoff was chosen by us (declared in the config headers as "exploratory, our choice, not a PI decision"). Every comparison uses 3 seeds (30-32, or 50-52 for the F comparisons and TabPFN, 70-72 for the pre-top-K pilot; the two legacy reproductions keep their legacy seeds). Nothing was pre-registered for these runs.

# Progress

- **PI's handoff, batch 1 (2026-11-25).** Six studies: the revision harness on the plan_full data, two legacy revision reproductions, a regression counterfactual audit of the case statistics, removal policies on a synthetic set, and the all-case twin of the synchronisation conditions. All complete.
- **PI's handoff, batch 2 (2026-11-25).** Seven retention studies: case statistics from the adapted outcome, a fine-grained memory curve (K 0.05-0.9), 8 policies on the 7 synthetic factor sets, boundary protection, the evidence rule, bias normalisation and Q smoothing. All complete, each checked by a second independent analysis.
- **PI's handoff, packs p0-p4 and p0a (2026-11-26).** p0 sensitivity sweeps, p0a delta-from-initial-bias, p1 all-case F and the all-case twin of all 13 plan_full groups, p2 seven new retention policies with latency and reliability measures, p3 core and reuse ablations, p4 TabPFN and a covariate-shift split. 1,050 jobs on r760, q-lab and xps8960, all exit 0 in the end.
- **Three failures fixed and rerun.** An autograd error in synchronised training, a tie-breaking disagreement in the new retention engine, and a missing opt-in key for delta_from_init. Fixed, tested, and the affected jobs reran. No completed unit changed.
- **Summaries and checks finished (2026-11-27).** The revision stress comparison and the synthetic new-policy study are summarized. The p2 tables were rechecked from raw event files, and a tie check showed the older wine units are unaffected by the tie fix.
- **Code and reports pushed; Heuzi's work merged.** The reports and summary scripts are committed, and Heuzi's adaptation work from the owner's fork was merged into main and neural-cbr (two conflicts in shared code, resolved keeping both sides). Both branches are pushed. The full test suites pass, apart from one known older failure.
- **Backup measured, not started.** The single copy of all runs (689 GB on r760) would take 5-17 hours to copy over the current link. A plan is written; starting it is left to the owner.

# Experiments

## 1. All-case F against masked F and the baselines (p1 experiment 1)

| A vs B | tasks | W/T/L | GMR | GMR without zebra |
|---|---|---|---|---|
| all-case F vs masked F | 29 | 18/2/9 | 1.009 | 0.935 |
| all-case F vs all-case retrieval only | 29 | 18/0/11 | 0.946 | 0.943 |
| all-case F vs all-case reuse arm | 29 | 19/2/8 | 0.956 | 0.962 |
| F with zero radius vs F | 18 | 8/2/8 | 0.997 | 0.996 |
| F without correction-size penalty vs F | 18 | 5/5/8 | 1.005 | 1.005 |
| all-case F vs kNN | 29 | 22/0/7 | 0.680 | 0.684 |
| all-case F vs MLP | 29 | 20/0/9 | 0.885 | 0.914 |
| all-case F vs CatBoost | 29 | 10/0/19 | 1.121 | 1.068 |
| all-case F vs TabM | 29 | 12/0/17 | 0.955 | 0.984 |

Rows: comparisons on the 29 tasks (QSAR excluded); the penalty rows use the 18 tasks whose frozen F has a non-zero penalty.
Columns: W = A has the lower 3-seed mean error. Seeds 50-52, all complete (two failed jobs were fixed and rerun, section 13).
Exploratory choices: each task's core is the validation winner of the all-case grid, trained at 300/40 epochs (the grid used 600/60); schedule and loss weights as the masked frozen F; the penalty weight is not re-selected.

All-case F beats masked F on 18 of 29 tasks, most clearly on regression (11 of 15, all 3 seeds on 8, e.g. body_fat 0.607 vs 0.647). The pooled GMR is still 1.009 because of one task: on zebra two of three all-case seeds collapse to about 50% accuracy (0.348 vs 0.000 masked); without zebra the GMR is 0.935. The adapter adds about 5% over all-case retrieval only. The three penalty variants do not separate. Unfavourable: CatBoost still beats all-case F on 19 of 29 tasks (GMR 1.121), and 8 of 87 all-case cores hit the 300-epoch cap (masked: 0), so the all-case arm may be under-trained.

## 2. All-case twin of the full-cycle experiments (p1 experiment 2)

| group | classification W/T/L, GMR | regression W/T/L, GMR |
|---|---|---|
| retention | 186/6/48, 0.730 | 221/0/43, 0.936 |
| retention with MCB | 30/0/10, 0.804 | 39/0/5, 0.906 |
| synchronisation | 78/0/22, 0.819 | 66/0/44, 0.954 |
| synchronisation with MCB | 80/3/17, 0.719 | 87/0/23, 0.876 |
| synchronisation, encoder control | 64/0/36, 1.001 | 67/0/43, 0.926 |
| reuse | 43/5/12, 0.800 | 24/0/9, 0.960 |
| exact removal | 60/2/10, 0.755 | 63/0/25, 0.914 |
| retraining after removal | 36/1/17, 0.873 | 41/0/25, 0.970 |
| adapted retraining | 18/3/6, 0.793 | 14/0/19, 1.017 |

Rows: the plan_full experiment groups (revision stress is in section 3).
Columns: counts over cells (task × condition), W = the all-case twin has the lower 3-seed mean error; GMR = all-case ÷ masked. Seeds 30-32, all complete.

Over 1,660 cells the all-case twin wins 1,223, ties 20 and loses 417. It is better in most cells of every group except regression adapted retraining and classification encoder control. So the all-case core lifts nearly every downstream CBR step, mainly because the core itself starts better. The earlier all-case twin of the five synchronisation penalty variants (batch 1) points the same way: whether the core is masked or all-case decides each task (alternating_rr: classification 5/2/3, regression 5/2/4), while the penalty variants barely differ. In the all-case classification rows the free correction radius often collapses to zero (19 of 30 rows, against 4 of 30 masked), which makes three of the five variants the same run; their "no effect" there is a degenerate radius, not evidence.

## 3. Human-revision stress test: flagging corrupted cases (batch 1 and p1)

| flagging method | masked, b 10 | masked, b 30 | all-case, b 10 | all-case, b 30 |
|---|---|---|---|---|
| random (precision) | 0.224 | 0.244 | 0.224 | 0.244 |
| provenance_only, precision d vs random | +0.090, 5/1/1 | +0.089, 7/0/0 | +0.343, 6/0/1 | +0.251, 5/1/1 |
| bias_only, precision d vs random | −0.095, 0/3/4 | −0.137, 0/0/7 | +0.600, 7/0/0 | +0.446, 7/0/0 |
| oracle, M2 accuracy d vs random | +0.007, 3/3/1 | +0.020, 7/0/0 | +0.004, 4/3/0 | +0.007, 5/2/0 |
| provenance_only, M2 accuracy d vs random | +0.006, 1/6/0 | +0.006, 2/4/1 | +0.003, 3/4/0 | +0.003, 2/5/0 |

Rows: precision = share of the b flagged cases that are truly corrupted; M2 = test accuracy after relabelling the flagged corrupted cases and retraining 5 epochs.
Columns: masked plan_full core and its all-case twin, review budget b. 7 synthetic sets, seeds 30-32, all units complete; W/T/L over 7 sets, tie when |d| ≤ 0.005. M0 accuracy is 0.861 all-case and 0.817 masked.

Under the masked core the learned case weight is a worse flag than random; under the all-case core it becomes the best non-oracle flag (precision 0.824 at b 10). Better flags buy almost no accuracy: even perfect flagging at b 30 adds only +0.010 over the uncorrected core under all-case and +0.026 under masked. The all-case twin ends higher after revision (M2 +0.039 for provenance_only at b 10, 5/2/0) only because its core starts higher; the edits help it less. Batch 1 ran the longer revision harness (budgets 10-75, 30 retraining epochs) on the nearly clean synthetic_reference set (M0 0.945): provenance_only flags at about twice the random rate (precision 0.25-0.33 vs 0.14-0.16), but its accuracy gain over random stays within one test case; an oracle reviewer gains +0.005 to +0.023. The two legacy reproductions show the old picture: only the oracle arm gains. The simulated reviewer always relabels correctly, and code drift between the masked recording (earlier commit) and the all-case twin is not fully excluded.

## 4. New retention policies (p2, real tasks and synthetic sets)

| policy | real cls K 0.2 | real cls K 0.5 | real reg K 0.2 (RMSE) | synthetic cls K 0.2 | synthetic cls K 0.5 |
|---|---|---|---|---|---|
| rc_cnn | +0.036, 7/2/1 | +0.006, 3/4/3 | −0.031, 7/1/3 | −0.006, 3/0/4 | −0.010, 0/4/3 |
| crr | −0.020, 1/2/7 | +0.009, 3/4/3 | +0.014, 3/0/8 | −0.124, 1/0/6 | −0.017, 2/1/4 |
| bbnr | +0.005, 5/1/4 | −0.006, 2/3/5 | +0.016, 3/2/6 | −0.002, 3/0/4 | −0.013, 1/0/6 |
| enn | +0.013, 5/2/3 | −0.004, 3/3/4 | +0.018, 2/2/7 | +0.012, 4/1/2 | +0.007, 5/0/2 |
| energy_compress | +0.040, 7/2/1 | +0.017, 3/4/3 | −0.039, 7/3/1 | −0.011, 1/3/3 | −0.025, 0/1/6 |
| knn_shapley | −0.002, 2/5/3 | −0.025, 2/2/6 | +0.086, 1/1/9 | +0.030, 6/1/0 | +0.030, 7/0/0 |
| loo_value | +0.004, 6/0/4 | +0.012, 4/4/2 | −0.020, 4/4/3 | −0.004, 4/1/2 | −0.013, 2/2/3 |

Rows: the seven new policies (our implementations; our choices for "retrieved", "solves", tie order and the energy_compress stopping rule are listed in the 2026-11-26 report).
Columns: d = policy − provenance_only after finetune, paired by task and seed, then W/T/L (tie when |d| ≤ 0.005). Accuracy: positive is better; RMSE: negative is better. Real tasks: 10 classification, 11 regression; synthetic: 7 sets. Seeds 30-32, all complete.

On the real tasks energy_compress and rc_cnn beat provenance_only and random at K 0.2 on most tasks, and energy_compress is the most consistent (against random +0.026, 8/0/2 in classification and −0.054, 9/2/0 in regression). At K 0.75 nearly everything ties. On the synthetic sets the order reverses: knn_shapley is clearly best (0.866 at K 0.2 and 0.5, above full memory at 0.820), with its largest gains where many labels are corrupted, while energy_compress and rc_cnn fall to or below random. On the real tasks our knn_shapley is clearly worse. So which policy wins depends on how much label noise the memory holds. crr collapses at K 0.2 on the synthetic sets. The new policies take 0.17-0.33 s of maintenance against 0.11-0.14 s for the recorded rules; lookup time depends on K, not on the policy. Case statistics from 10% of the queries rank cases only moderately like the full query set (Spearman C 0.62 real / 0.54 synthetic, H 0.54 / 0.41).

## 5. Retention against memory size (batch 2, K 0.05-0.9)

| policy | cls K 0.1 | cls K 0.2 | cls K 0.5 | cls K 0.9 | reg K 0.1 | reg K 0.2 | reg K 0.5 | reg K 0.9 |
|---|---|---|---|---|---|---|---|---|
| full memory | 0.903 | 0.903 | 0.903 | 0.903 | 0.407 | 0.407 | 0.407 | 0.407 |
| random | 0.843 | 0.855 | 0.883 | 0.900 | 0.489 | 0.466 | 0.420 | 0.407 |
| provenance_only | 0.833 | 0.841 | 0.889 | 0.899 | 0.496 | 0.451 | 0.429 | 0.415 |
| bias_current | 0.824 | 0.856 | 0.899 | 0.899 | 0.445 | 0.424 | 0.410 | 0.408 |
| bias_normalized | 0.835 | 0.867 | 0.890 | 0.899 | 0.454 | 0.428 | 0.411 | 0.407 |
| kcenter | 0.816 | 0.847 | 0.889 | 0.903 | 0.445 | 0.421 | 0.409 | 0.406 |

Rows: 6 of the 8 policies (stratified and coverage_reachability are in the batch-2 report).
Columns: mean over datasets of the 3-seed mean after finetune; classification accuracy (higher is better, 10 datasets), regression RMSE (lower is better, 11 datasets). Seeds 30-32, all 1,896 units complete. The regression mean is dominated by energy_efficiency, so the counts in the text use medians and per-dataset verdicts.

Compression never reliably beats full memory. Classification tolerates K ≥ 0.75 at no cost and loses 2-5 points (median) at K ≤ 0.1, with zebra the worst. Regression pays a small but consistent cost already at K 0.75 (worse on 5-8 of 11 datasets, median +0.07% to +0.8%). In regression bias_current, bias_normalized and kcenter beat random on 8-11 of 11 datasets at K 0.1-0.5 and stay within 2% of full memory down to K 0.2, while random and provenance_only break between K 0.33 and K 0.2. In classification no policy beats random consistently along the curve. provenance_only beats random before the finetune at K 0.2-0.5 but not after it, so the finetune erases its selection advantage.

## 6. What retention keeps on the synthetic sets (batch 2)

| policy | corrupted kept, K 0.2 | duplicates kept, K 0.2 | rare kept, K 0.2 | corrupted kept, K 0.5 | rare kept, K 0.5 |
|---|---|---|---|---|---|
| random | 0.189 | 0.213 | 0.167 | 0.480 | 0.528 |
| kcenter | 0.242 | 0.107 | 0.157 | 0.555 | 0.245 |
| coverage_reachability | 0.058 | 0.234 | 0.315 | 0.272 | 0.741 |
| provenance_only | 0.065 | 0.328 | 0.477 | 0.322 | 0.769 |
| bias_current | 0.255 | 0.342 | 0.000 | 0.474 | 0.000 |
| bias_normalized | 0.257 | 0.331 | 0.000 | 0.483 | 0.000 |

Rows: retention policies on the 7 synthetic factor sets (corrupted labels, duplicates and rare cases are known from the generator).
Columns: share of each kind of training case that the policy keeps, pooled over 7 sets × 3 seeds (6 sets for rare). For corrupted cases and duplicates lower is better; for rare cases higher is better. Seeds 30-32, all complete.

The evidence-based policies (coverage_reachability, provenance_only) remove corrupted cases and keep rare ones, but keep duplicates. Only kcenter removes duplicates. The bias policies drop every rare case at K 0.2 and 0.5, yet win on shifted-domain accuracy (12/4/5 and 12/5/4 cells against random). No policy keeps clean and rare cases while removing duplicates. Accuracy gains appear only where corruption dominates (noise, overlap, combined). Pruned models beat full memory more often than not, but the full-memory finetune is a no-op in 7 of 21 set-seed pairs, which is unresolved and must be checked before crediting the finetune.

## 7. Retention rule variants (batch 2 and p0)

| variant | what changed | result |
|---|---|---|
| case statistics from the adapted outcome | C/H read the adapter's prediction | only provenance_only and coverage_reachability change, policy-mean accuracy by ≤ 0.003; ordering unchanged within noise |
| boundary protection 0.1 | 10% of cases nearest another class are always kept | no accuracy gain (3-4 better vs 8-9 worse of 21 cells); corrupted share kept at K 0.2 rises from about 0.06 to 0.24 for the evidence policies |
| activation threshold 0.01 / 0.2 | coverage_reachability key | 0.01 near inert (2-5% of the kept set changes); 0.2 replaces 24-38% and is 13 better / 13 worse of 48 cells |
| zero reachability ranked first | coverage_reachability key | 15 worse / 8 better of 48 cells; keeps +0.656 more of the mislabelled cases on synthetic_diag at K 0.2 |
| evidence rule (1, 0.0) / (5, 0.5) | provenance_only evidence threshold | looser rule mildly favourable (5 better, 1 worse); stricter rule worse (14 worse / 8 better) |
| bias normalisation median/MAD | instead of within-cohort percentile | identical at K 0.5; worse at K 0.2 (7 of 10 cls, 7 of 11 reg; car +0.161, yacht +0.125 RMSE) |
| Q smoothing s 0.1 / 5.0 | provenance_only, recorded s = 1 | s 5.0 better on 11 of 21 cells at K 0.2, worse on 1; s 0.1 unfavourable |
| rare-cohort protection below 10 | p0, anneal (the only real task with a class of ≤ 10 cases) | within ±0.0033 accuracy (0/7/0/5) |

Rows: one changed key each, against the recorded plan_full value on the same cores (cores and case statistics identical in every paired unit).
Columns: b/t/w counts are over datasets × K cells. Seeds 30-32, all complete.

None of the variants is a clear improvement. The recorded settings (within-cohort percentile normalisation, zero reachability counted as 0, evidence rule R ≥ 2 and A ≥ 0.1, s = 1) are reasonable; s = 2-5 would be slightly better at small K. Boundary protection as specified works against removing harmful cases, probably because a mislabelled case lies inside another class's region and so looks like a boundary case (inferred, not testable from the records).

## 8. Sensitivity sweeps and delta-from-initial-bias (p0, p0a)

| condition | metric | change vs recorded | b/t/w/i |
|---|---|---|---|
| reuse loss weights λd 0.3, λc 1 | cls ECE after adapter | −0.067 (0.186 vs 0.253) | 9/0/1/0 |
| reuse loss weights λd 1, λc 0.3 | cls ECE after adapter | +0.059 | 0/0/10/0 |
| reuse loss weights, all 4 pairs | cls accuracy after adapter | −0.002 to −0.006 | mostly w or t |
| clip_normalize probabilities, diff_only loss | cls ECE after adapter | −0.30 | 9/1/0 |
| s_task 1.0 / 0.25 (recorded 0.5) | cls ECE after adapter | −0.033 / +0.028 | 8/1/0/1, 0/1/8/1 |
| radius median, shrunk, clipped; cohort-normalized | all | ≤ 0.013 | mostly t |
| no core pretraining (t* = 0) | cls accuracy / reg RMSE | −0.054 / +0.111 | 2/3/4/1, 2/1/8/0 |
| MCB momentum 0.9 or 0.999 (recorded 0.99) | accuracy / RMSE | −0.014 to +0.019 | mixed, inside seed sd |
| bias_init_k 3 / 10 / 20 (recorded 5), all-case | test error | ≤ 5e-6 | about 5 i per k |
| delta_from_init with the build-time bias (p0a) | keep/archive actions | identical in 375 of 378 pairs | 3 differ (abalone, denormal biases) |

Rows: one changed setting each. b/t/w/i = better / tie / worse / identical per task: better when the 3-seed mean favours the condition and at least 2 of 3 seeds agree.
Columns: change = condition − recorded reference. Seeds 30-32 (70-72 for bias_init_k), all complete.

The ratio of the two reuse loss weights moves classification calibration, not accuracy: more weight on the classification loss lowers ECE. clip_normalize changes evaluation only and lowers ECE a lot. A larger s_task lowers ECE. Radius variants, cohort normalisation, MCB momentum and the initial-bias neighbourhood size show no consistent effect. Dropping core pretraining hurts, but that condition changes two things at once. delta_from_init with the build-time bias gives the same ranking as the current bias, as expected, because every case starts from one shared bias; a per-case initial bias would need a different build rule.

## 9. Core and reuse ablations (p3)

| ablation | comparison | result |
|---|---|---|
| (a) initial bias from learned-distance percentile 10/25/50 | pre-top-K pilot, 12 tasks | test error within 6e-5 of knn_mean k = 5 (0.2016) |
| (a) percentile 50 inside synchronisation | rr conditions, NLL | lower on 8-9 of 10 cls tasks (alternating_rr −0.044, 8/1/1); accuracy moves < 0.01 |
| (b) historical per-case adapter (2022) | vs the aggregate adapter | cls accuracy +0.004; reg RMSE +0.07, almost all from body_fat (without it −0.004 / −0.008) |
| (c) direct prediction head (capacity reference) | vs the aggregate adapter | accuracy 0.849 vs 0.905, loss concentrated in zebra (0.485 vs 0.939) |
| (d) probability-vector distance for the free radius | vs recorded label metric | neutral (accuracy +0.001) |
| (e) adapter without the nominal input channel | synthetic_nominal, 6 conditions | loses 0.29-0.43 accuracy, near retrieval only (about 0.37) |

Rows: the five opt-in ablations. Seeds 30-32 (70-72 for the pilot), all complete; same b/t/w/i rule as section 8.

The initial-bias rule does not matter for the core. The historical per-case adapter is not better than the current one and is unstable on body_fat (seed differences up to +1.5 RMSE, not investigated). The direct head is clearly worse, so the adapter is not just extra capacity. On nominal data the adapter's whole gain comes from the nominal input channel (one synthetic task only). The p3 references were recorded at older commits, so differences around 1e-4 can mix in code drift.

## 10. TabPFN as a comparator (p4)

| method | cls error diff vs TabPFN, TabPFN W/T/L | reg error diff vs TabPFN, TabPFN W/T/L |
|---|---|---|
| masked F | +0.027, 11/1/1 | +0.042, 11/0/0 |
| all-case F | +0.053, 11/0/2 | +0.035 (10 tasks), 10/0/0 |
| CatBoost | +0.007, 8/2/3 | +0.028, 8/0/3 |
| TabM | +0.068, 10/1/2 | +0.036, 10/0/1 |
| MLP | +0.081, 12/0/1 | +0.055, 11/0/0 |
| kNN | +0.067, 11/1/1 | +0.185, 11/0/0 |

Rows: each method against TabPFN v2 (fixed settings, 4 estimators, CPU, no tuning, at most 3,000 training rows).
Columns: mean error difference (method − TabPFN, positive means TabPFN is better), then tasks where TabPFN wins/ties/loses. 13 classification and 11 regression tasks of 29 (5 tasks are too large or too slow on CPU), seeds 50-52, all complete. TabPFN error 0.066 (classification) and 0.381 (regression).

Unfavourable to NN-kNN: TabPFN is the strongest method we have run. It beats masked and all-case F on nearly every task, and beats CatBoost on most. Only TabPFN v2 was used; newer versions need a licence click-through.

## 11. Covariate shift (p4)

| method | classification error diff vs F (F W/T/L) | regression error diff vs F (F W/T/L) |
|---|---|---|
| retrieval only | +0.037 (6/0/3) | +0.073 (9/0/2) |
| frozen-core reuse arm | +0.037 (6/0/3) | +0.114 (9/1/1) |
| kNN | +0.032 (5/0/4) | +0.257 (9/0/2) |
| MLP | +0.017 (5/0/4) | +0.276 (7/0/4) |
| CatBoost | −0.079 (3/0/6) | +0.185 (7/0/4) |

Rows: methods compared with shifted F. Our split: train on the lower 70% of the first principal component of the standardised numeric features, test on the upper 30% (reported as near and far halves).
Columns: mean error difference (method − F) and tasks where F wins/ties/loses. 9 classification and 11 regression tasks, seeds 30-32 (the 3 seeds share one test set), all complete. F error 0.240 (classification), 1.031 (regression).

Every method is worse on the far half (F +0.103 classification, +0.676 regression), and F's near-to-far gap is not smaller than the baselines', so there is no sign that F extrapolates better. The adapter helps under shift (F beats retrieval only on 6 of 9 and 9 of 11 tasks); the frozen-core reuse arm does not. CatBoost beats F on 6 of 9 classification tasks. Retention at K 0.5 costs more under shift (random +0.043 / +0.069 vs +0.023 / +0.013 without). Regression means are skewed by car; the W/T/L counts are the more reliable reading.

## 12. Counterfactual audit of the case statistics (batch 1)

| measure (33 regression task-seed units) | value |
|---|---|
| Spearman of outcome credit Q with counterfactual influence | 0.15 ± 0.32 (positive in 21 of 33) |
| Spearman of the bias percentile B with counterfactual influence | 0.49 ± 0.15 (positive in 33 of 33) |
| sign agreement of Q > 0.5 with "the case helps" | 0.71, chance level from the base rates 0.71 |
| Spearman of activation mass with the size of the influence | 0.65 ± 0.25 |

Rows: agreement measures over the 60 audited cases per task and seed; ± is the sd over the 33 units.
Columns: 11 regression tasks × seeds 30-32, all complete. The influence of a case is the mean final loss without it minus with it.

Q, the outcome credit used by provenance_only, agrees only weakly with the measured counterfactual help of a case, and its sign agreement is no better than chance. The learned bias tracks help better on 9 of 11 tasks. The size of a case's influence follows how often it is retrieved, the same exposure confound noted earlier for Q. This audit uses one success tolerance (0.5).

## 13. Failures, fixes and checks

| problem | cause | fix | effect on recorded results |
|---|---|---|---|
| all-case F wine s50 and task 363631 s50 crashed ("backward through the graph a second time") | a shape probe ran with gradients; 11 of 106 wine cases map to all-zero features, so a stale graph entered the feature cache | the probe runs without gradients | bitwise-equal weights on four configs that did not fail; both jobs reran, exit 0 |
| new-policy classification wine s32 crashed (engine disagrees with the model by 0.332) | an exact tie at the k-th place; the engine and the model broke it differently | tied rows take the model's members | s32 reran (22 new units); s30/s31 have no tied rows (checked on r760; 18 ulps closest gap) |
| delta_from_init could not be expressed | the store was seeded after training | opt-in key for the build-time bias; invalid values refused before training | p0a ran 58 jobs, all exit 0 |
| review defects, one per pack | energy_compress rounding drift, a distance shortcut putting equal vectors 3e-4 apart, missing per-half metrics, missing conditions | fixed before any host run | none |

Rows: the failures and defects found since 2026-11-24.
Columns: the last column says whether any completed unit changed. All summaries read only the recorded runs; for every pack, an independent reader of the raw event files reproduced two or more headline numbers exactly, including the 10-task p2 tables and both 2026-11-27 summaries.

No completed unit was retrained or changed by a fix. The wine s30/s31 tie check ran on r760 because xps8960, where those units ran, was unreachable; a tie there would need the two hosts to differ by at least 18 float32 ulps.

# TODO

Waiting for a decision or for access:
- Back up the single copy of all runs (689 GB on r760): the overnight copy is planned and measured but not started.
- Confirm the wine s30/s31 tie check on xps8960 once ssh works again; check the other new-policy units recorded before the tie fix for ties in the maintained cores.
- The human revision study (needs participants and an IRB determination).

Open analysis (no new training needed):
- The no-op full-memory finetune on 7 of 21 synthetic set-seed pairs.
- The zebra collapse of all-case F (seeds 50 and 52) and the per-case adapter instability on body_fat.
- Correction-size tail and the share of corrections above the task allowance, from the recorded weights.
- ECE, per-group accuracies and flip matrices of the revision stress and synthetic new-policy studies.
- Heuzi's merged adapter redesign experiments: read and relate them to F.

Possible next runs (would be our own exploratory choices):
- All-case F at the grid's 600/60 budget, with the penalty weight re-selected for all-case.
- TabPFN on the 5 remaining tasks (estimated 25-37 CPU-hours per seed on 3 of them) and TabM/TabPFN under the shift split.
- MCB momentum on more synchronisation schedules with a shared core.
- GPU image and text suites (set aside for now; out of the T1 matrix).
- The QSAR task (postponed).
