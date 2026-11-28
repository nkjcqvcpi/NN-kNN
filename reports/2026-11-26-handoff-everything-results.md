# 2026-11-26 handoff packs p0-p4 and p0a: what was built, what it shows, what is left

Updated 2026-11-27: added the p1 revision_stress comparison and the synthetic new-policy study (`handoff_synnewpol_r760`),
the tie check for p2 classification wine s30/s31, the raw-event check of the 10-task p2 tables, the git state and the
backup measurement. Detail for the first two is in `reports/2026-11-27-handoff-revstress-synnewpol-results.md`.

Status: exploratory. Every value below that is not in Heuzi's handoff was chosen by us and is declared in the config
headers as "exploratory, our choice, not a PI decision". Each comparison uses 3 seeds. Nothing here is a significance
test. Table 1 code, frozen configs and the shared NN-kNN core defaults are unchanged; all new behaviour is opt-in.

Per-pack reports with more detail:
`reports/2026-11-26-handoff-p0-results.md`, `-p2-results.md`, `-p3-results.md`, `-p4-baselines-stress-summary.md`,
`reports/2026-11-26-fixes-allcaseF-engine-deltainit.md` (the fixes and p0a),
`reports/2026-11-27-handoff-revstress-synnewpol-results.md` (p1 revision_stress and p2 synthetic). Pack p1 has no
separate report for its other parts; those results are only in this file and in `results/t1/handoff_20261126/p1/`.

## 1. Where the numbers come from

- All units live in r760 `/mnt/largedata/NN-KNN/runs/t1`. That folder is the single copy. Accel `runs/` is empty. The
  sync loop (`/var/tmp/nnknn-sync/sync_loop.sh`) copies q-lab and xps8960 units to r760 and does not copy r760 units
  to accel. Nothing was copied by hand.
- Every summary was produced in two steps: a read-only `dump` on r760 that reads only `runs/`, then a `summarize` or
  `analyze` step on accel. Scripts are in `tools/` (`t1_handoff_p0_summary.py`, `t1_handoff1126_p0a.py`,
  `t1_handoff1126_p1_dump.py`, `t1_handoff1126_p1_summary.py`, `t1_handoff1126_p2.py` (`--set real` or
  `--set synthetic`), `t1_handoff1126_p3.py`, `t1_handoff1126_revstress.py`, `t1_handoff_p4_summary.py`). Outputs are in
  `results/t1/handoff_20261126/<pack>/`, plus `revstress/` and `synnewpol/`.
- Each pack has an independent check: a separate reader decodes the raw TensorBoard event files without project code
  and reproduces two or more headline numbers exactly (files `indep_check*`, `raw_check_p1*`, `raw_event_check*`).
- Every task's seeds and compared variants ran on one host, the host that recorded the units they are compared with.
  Host names inside the records: `haotang-lab` is xps8960, `cs-peng-research` is r760.

## 2. What was built

Commits on `neural-cbr`: p0 b12ec87 and review fix 7b57db1; p1 a06f8a2 and f993af3; p2 7272df3
and ef21785; p3 7f09b54 and 91cee1f (shared-core change e2aa955 on main, synced as 6dba3cc); p4 6c00348, 034c35b and
2bd6c4a (TabPFN pin ddf7e41 on main); fixes 4235e10 and 6ece140. The reports and summary scripts were committed on
2026-11-27 (e94b930 and the commit that adds this update), and `neural-cbr` was pushed to origin the same day. Each pack was reviewed adversarially after the build,
and each review fixed one defect (listed under each pack). The full test suite passes apart from the known
pre-existing failure `test_t1_tuning_features.py::test_defaults_reproduce_the_recorded_toy_outputs`.

Jobs run (all exit 0 in the end):

| pack | r760 | q-lab | xps8960 |
|---|---|---|---|
| p0 | 183 (incl. 21 synthetic rare10 no-op) | 49 | 41 |
| p0a | 24 | 18 | 16 |
| p1 | 171 | 95 | 78 |
| p2 | 45 | 26 | 16 |
| p3 | 27 | 36 | 33 |
| p4 | 78 | 67 | 47 |

### p0: sensitivity sweeps (configs only, no code change)

Configs in `configs/t1/handoff_20261126/p0/`, built from the resolved plan_full_r760 configs, all masked, seeds 30-32
(70-72 for bias_init_k).
- Reuse loss weights (`handoff_reusesweep_r760`): combined loss with (lambda_diff, lambda_cls) in (1, 0.3), (0.3, 1),
  (3, 1), (1, 3) for nominal_residual_scores and logit_residual; `probability_mode: clip_normalize` for the three
  recorded nominal_residual_scores losses. For regression L_diff equals L_cls, so the pairs only rescale one MSE.
- Sync sensitivity (`handoff_syncsens_r760`): one key changed per condition: s_task 0.25 and 1.0 (recorded 0.5) on
  alternating_rr and rr_zero_radius; radius_aggregation median; bias_transform shrunk (0.5) and clipped (0.95 quantile);
  cohort_normalized (classification only); t* = 0 through `core_pretrain: false`, which also drops core pretraining.
- MCB momentum (`handoff_mcbmom_r760`): 0.9 and 0.999 against the recorded 0.99; retention with the two mcb-true
  conditions; sync_mcb with only the independent, alternating_rr and alternating_rrr schedules (cost choice).
- Rare-cohort protection (`handoff_protect_r760`): `protect_rare_cohort_below 10` on the 7 synthetic sets (a proven
  no-op, smallest class 118 cases) and, added in review, on tabarena_363614 (anneal), the only plan_full task with a
  class of 10 or fewer training cases.
- bias_init_k 3, 10, 20 against 5 (`handoff_bik{3,10,20}_r760`); the pre-top-K pilot fits both arms.
- Items (a) delta_from_init and (g) no nominal channel needed code; they became p0a and p3 (e).

### p0a: delta_from_init with the build-time bias

- Opt-in key `statistics.initial_bias: build_time` (default `first_audit`, unchanged). It seeds the case statistics
  store with the bias `build_model` gave every case, so `bias_source: delta_from_init` measures b - b_build instead of
  0. Retention only; `t1_run` refuses it elsewhere and together with `reliability`.
- Four configs (`configs/t1/handoff_20261126/p0a/`): within-cohort percentile against plan_full `bias_normalized`, and
  median/MAD against `handoff_btransform_r760`. Study `handoff_deltainit_r760`, seeds 30-32, 21 tasks.
- Expected before running: build_model gives every case the same bias, so b - b_build is a constant shift and the
  ranking should not change.

### p1: all-case baseline (`configs/t1/handoff_20261126/p1-allcase/`, DECLARATION.md)

- `core.pre_topk_mask: false`; neighbour bias init knn_mean, k = 5.
- Per-task core: the validation-selected winner of `review_gridallcase_r760` (recomputed from recorded validation rows,
  equal to selection.json), batch 64, lr_glocal = lr_case, lr_feature = lr_case / 10. Budget 300/40, although the
  grid refit used 600/60.
- Schedule and loss weights as each task's frozen F (joint_rr, summed objective). v_small is kept from the masked
  frozen F, not re-selected.
- Experiment 1 (`handoff_allcaseF_r760`, seeds 50-52, 29 tasks, QSAR excluded): arms retrieval_only, reuse, F;
  F_zero_radius (squared L2 from zero) and, added in review, F_no_penalty (v_small 0) on the 18 tasks whose frozen F
  has v_small > 0. maskedF is the frozen config with the F arm only. Matched baselines kNN, MLP, CatBoost and TabM are
  the recorded grid selections evaluated on seeds 50-52.
- Experiment 2 (`handoff_allcaseplan_r760`, seeds 30-32): the all-case twin of all 13 plan_full groups. Tuned groups
  get the all-case grid winner; groups that plan_full runs on the untuned core keep it with only the mask switched.

### p2: new retention policies, latency, C/H reliability

- `model/t1/baselines.py`: rc_cnn, crr, bbnr, enn, energy_compress, knn_shapley, loo_value, for classification and
  regression, all scored from one table of learned retrieval scores that is checked against `model.retrieve`.
- Our choices: "retrieved" = activation above 0.05; "solves" for regression = |error| <= 0.5 standardized; RC by the
  Smyth-McKenna sum with CNN in decreasing RC order; enn and bbnr fill with the seeded random order; energy_compress
  deletes down to K and stops (not the paper's best-case-base rule), hinge margin 0.2, squared error for regression
  (our extension); knn_shapley uses the core's top_k; loo_value is one pass on the full memory.
- Opt-in `efficiency` (maintenance seconds, single-query and batch lookup times, memory bytes; repeats 5, 50 single
  queries) and `reliability` (C, H, Q, Qe on nested 10/25/50/100% query subsets, 5 repeats).
- Review fix ef21785: energy_compress accumulated rounding drift and broke ties against the shared convention.
- Studies `handoff_newpol_r760`, `handoff_latency_r760`, `handoff_synnewpol_r760`; seeds 30-32, K 0.2/0.5/0.75.

### p3: core and reuse ablations

- (a) `bias_init: learned_distance_percentile` (10/25/50), exact over all non-self pairs; in the pre-top-K pilot
  (seeds 70-72) and in sync.
- (b) `output_mode: per_case_2022`, the historical per-case adapter (diff_only and combined).
- (c) `output_mode: direct_head`, a capacity reference with cls_only, not an adaptation method.
- (d) `sync.free_radius_label_metric: probability_vector_distance`. Review fix 91cee1f: `torch.cdist` took a
  matrix-product shortcut that put equal vectors about 3e-4 apart; it now uses direct differences.
- (e) `reuse.omit_nominal_channel: true` on synthetic_nominal, the only task where it matters; all six recorded
  nominal_reuse conditions are repeated.

### p4: TabPFN and a covariate shift split

- TabPFN v2 (`tabpfn==10.2.0`, weights pinned by sha256) as a grid family with `selection: fixed`: 4 estimators, CPU,
  no tuning, at most 3000 training rows; seeds 50-52, the p1 baseline splits. Newer TabPFN versions need a license
  click-through on Hugging Face and were not used.
- `split_mode: covariate_pc1`, `shift_quantile 0.7`: train on the lower 70% of the first principal component of the
  standardized numeric features, test on the upper 30%, reported as near and far halves. The 3 seeds share one test
  set. 20 tasks (synthetic_diag refuses the split). shiftF (maskedF arms), shift retention (8 plan_full policies at
  K 0.5) and shifted kNN/MLP/CatBoost with their recorded random-split selections.
- Review fix 2bd6c4a: regression rows and the shifted baselines now also record per-half metrics.

## 3. Results

Conventions differ slightly by pack, so each section states its own. "b/t/w/i" = better / tie / worse / identical per
task; "W/T/L" = wins / ties / losses per task.

### p0

Verdict rule: b when the 3-seed mean favours the condition and at least 2 of 3 seeds agree; w the mirror case; i when
all three paired differences are exactly 0; t otherwise (no tolerance).

| condition | metric | Δ vs reference | b/t/w/i |
|---|---|---|---|
| NRS λd 0.3, λc 1 | cls ECE post | −0.067 (0.186 vs 0.253) | 9/0/1/0 |
| NRS λd 1, λc 3 | cls ECE post | −0.063 | 9/1/0/0 |
| NRS λd 1, λc 0.3 / λd 3, λc 1 | cls ECE post | +0.059 / +0.055 | 0/0/10/0 each |
| NRS, all 4 pairs | cls acc post | −0.002 to −0.006 | mostly w or t |
| clip_normalize combined / cls_only / diff_only | cls ECE post | −0.19 / −0.05 / −0.30 | 8/1/1, 8/0/2, 9/1/0 |
| regression, all 4 pairs | RMSE post | +0.0005 to +0.0007 | 3-7 w |
| s_task 1.0 | cls ECE post | −0.033 | 8/1/0/1 |
| s_task 0.25 | cls ECE post | +0.028 | 0/1/8/1 |
| radius median / shrunk / clipped, cohort-normalized | all | ≤ 0.013 | mostly t |
| t* = 0, no core pretrain | cls acc / reg RMSE | −0.054 / +0.111 | 2/3/4/1, 2/1/8/0 |
| MCB momentum 0.9 or 0.999 | acc / RMSE | −0.014 to +0.019 | mixed, inside seed sd 0.02-0.07 |
| protect rare10, synthetic | all | 0 | identical in every pair (no-op) |
| protect rare10, anneal (4.33 cases protected) | acc | within ±0.0033 | 0/7/0/5 |
| bias_init_k 3 / 10 / 20, all-case arm | test_error | ≤ 5e-6 | about 5 i per k |
| bias_init_k 3 / 10 / 20, masked arm | test_error | −0.011 / −0.013 / −0.007 | driven by yacht |

Reading: the lambda ratio moves classification calibration, not accuracy (more weight on L_cls lowers ECE).
clip_normalize changes evaluation only and lowers ECE a lot. A larger s_task lowers ECE. Radius variants, cohort
normalization and MCB momentum show no consistent effect. Dropping core pretraining hurts, but that condition changes
two things at once. bias_init_k has no effect in the all-case arm.

### p0a

378 pairs (2 normalizations × 21 tasks × 3 K × 3 seeds), every pair complete, same host, and recording
`initial_bias: build_time` with `bias_source: delta_from_init`. No p0a unit changed after the summary was made.

| variant | type | pairs | same keep/archive actions | same final model state | max |final scalar diff| |
|---|---|---|---|---|---|
| median/MAD | classification | 90 | 90 | 90 | 0 |
| median/MAD | regression | 99 | 99 | 99 | 0 |
| percentile | classification | 90 | 90 | 90 | 0 |
| percentile | regression | 99 | 96 | 96 | 0.0047 |

The 3 differing pairs are percentile regression abalone at K 0.2, seeds 30-32. There, training drove some biases to
denormal size (|b| about 1e-38); b - b_build then equals −b_build exactly in float64, so those cases tie under
delta_from_init although their current biases differ. Everywhere else the build-time delta_from_init ranking is the
current-bias ranking, as expected. So p0 item (a) adds no new information as long as every case starts from one shared
bias; a per-case initial bias would need a different build rule.

### p1 experiment 1: all-case F (seeds 50-52, 29 tasks)

Error is 1 − accuracy or standardized test RMSE. "Win" means A has the lower 3-seed mean error. GMR is the geometric
mean over tasks of error A / error B. These tables include the reruns (classification wine and 363631 s50).

| A vs B | tasks | W/T/L | GMR | GMR without zebra |
|---|---|---|---|---|
| all-case F vs masked F | 29 | 18/2/9 | 1.009 | 0.935 |
| all-case F vs all-case retrieval-only | 29 | 18/0/11 | 0.946 | 0.943 |
| all-case F vs all-case reuse | 29 | 19/2/8 | 0.956 | 0.962 |
| F_zero_radius vs F | 18 | 8/2/8 | 0.997 | 0.996 |
| F_no_penalty vs F | 18 | 5/5/8 | 1.005 | 1.005 |
| all-case F vs kNN | 29 | 22/0/7 | 0.680 | 0.684 |
| all-case F vs MLP | 29 | 20/0/9 | 0.885 | 0.914 |
| all-case F vs CatBoost | 29 | 10/0/19 | 1.121 | 1.068 |
| all-case F vs TabM | 29 | 12/0/17 | 0.955 | 0.984 |
| masked F vs CatBoost / TabM | 29 | 10/0/19 / 8/0/21 | 1.112 / 0.947 | |

- Regression: all-case F beats masked F on 11 of 15 tasks (all 3 seeds on 8), e.g. body_fat 0.607±0.084 vs
  0.647±0.106, 363631 0.159±0.011 vs 0.182±0.028, regression wine 0.720 vs 0.745 (3 of 3 seeds). Classification: 7
  wins, 2 ties, 5 losses. Classification wine gives the same error under both (0.0185±0.016 on 54 test rows).
- zebra: all-case F 0.348±0.303 against masked 0.000. Two of three seeds collapse to about 50% accuracy (core best
  epochs 1 and 26). This one task moves the F-vs-masked GMR from 0.935 to 1.009.
- 8 of 87 all-case cores hit the 300-epoch cap, against 0 masked.
- The three penalty conditions do not separate (zero-radius and no-penalty within ±0.5% GMR of F).

### p1 experiment 2: all-case twin vs plan_full (seeds 30-32, cells = task × condition)

| experiment | cls W/T/L, GMR | reg W/T/L, GMR |
|---|---|---|
| retention | 186/6/48, 0.730 | 221/0/43, 0.936 |
| mcb_retention | 30/0/10, 0.804 | 39/0/5, 0.906 |
| synchronization | 78/0/22, 0.819 | 66/0/44, 0.954 |
| sync_mcb | 80/3/17, 0.719 | 87/0/23, 0.876 |
| sync_encoder_control | 64/0/36, 1.001 | 67/0/43, 0.926 |
| reuse | 43/5/12, 0.800 | 24/0/9, 0.960 |
| exact_removal | 60/2/10, 0.755 | 63/0/25, 0.914 |
| removal_retraining | 36/1/17, 0.873 | 41/0/25, 0.970 |
| adapted_retraining | 18/3/6, 0.793 | 14/0/19, 1.017 |

Total 1223/20/417 over 1660 cells. The all-case twin has lower error in most cells of every group except regression
adapted_retraining and classification sync_encoder_control. revision_stress carries only M0/M1/M2/MC metrics, so it is
compared separately below.

### p1 experiment 2: revision_stress, all-case twin vs masked plan_full (7 synthetic sets, seeds 30-32)

A flagging method ranks the training cases of the trained core (M0); a simulated reviewer relabels the truly corrupted
cases among the top b (b = 10 or 30). M1 is the core with the edit only, M2 the edit plus 5 retraining epochs, MC M0 plus
the same 5 epochs without the edit. Both studies are complete (168 condition units and 42 stage units each). M0 and MC are
identical across the 8 conditions of every task and seed, and random and oracle select exactly the same cases in both
studies, so the two share data, corruption and the seeded order. d = arm − reference, paired by task and seed; ± is the
sd over seeds of the per-seed mean; W/T/L over 7 tasks, tie when |mean d| ≤ 0.005.

- Core: M0 accuracy 0.861 all-case vs 0.817 masked (d +0.0445 ± 0.0200, 5/2/0).
- Flagging precision vs random (random 0.224 at b10, 0.244 at b30):

| method | masked b10 | masked b30 | all-case b10 | all-case b30 |
|---|---|---|---|---|
| provenance_only | +0.090 ± 0.017, 5/1/1 | +0.089 ± 0.037, 7/0/0 | +0.343 ± 0.052, 6/0/1 | +0.251 ± 0.099, 5/1/1 |
| bias_only | −0.095 ± 0.036, 0/3/4 | −0.137 ± 0.015, 0/0/7 | +0.600 ± 0.074, 7/0/0 | +0.446 ± 0.094, 7/0/0 |

- M2 accuracy vs random (random M2 0.819 / 0.823 masked, 0.861 / 0.864 all-case at b10 / b30): provenance_only +0.006 /
  +0.006 masked and +0.003 / +0.003 all-case; bias_only between −0.0002 and +0.003; oracle +0.007 / +0.020 masked and
  +0.004 / +0.007 all-case. Even oracle flagging at b30 adds only +0.010 over M0 under all-case and +0.026 under masked.
- All-case vs masked, same condition: M2 accuracy d +0.039 ± 0.012 (provenance_only b10, 5/2/0), +0.039 (bias_only b10),
  +0.042 (random b10), +0.028 (oracle b30, 5/1/1). The M2 − M0 gain is smaller under all-case (oracle b30 −0.016, 0/2/5).

Reading: under the masked core the learned case bias is a worse flag than random; under the all-case core it becomes the
best non-oracle flag (precision 0.824 at b10). Better flags buy almost no accuracy after retraining. The all-case twin
ends higher only because its core starts higher; the edits help it less.

### p2 (updated: classification wine s32 now included)

The xps8960 rerun of newpol classification wine s32 is complete and its `fullmem_*` equals plan_full, so `summarize`
was rerun on the same r760 dump without the exclusion. Classification now has 10 tasks; regression (11 tasks) is
unchanged. The 9-task version is kept in `results/t1/handoff_20261126/p2/excluding_cla_wine_s32/`.
d = policy − reference, paired by task and seed on the same core; ± is the sd over seeds of the per-seed mean d;
tie = |3-seed mean d| ≤ 0.005.

Classification accuracy after finetune vs provenance_only (10 tasks; reference 0.841 / 0.889 / 0.900):

| policy | K 0.2 | K 0.5 | K 0.75 |
|---|---|---|---|
| rc_cnn | +0.036±0.014 7/2/1 | +0.006±0.010 3/4/3 | −0.002±0.008 2/6/2 |
| crr | −0.020±0.019 1/2/7 | +0.009±0.013 3/4/3 | +0.009±0.011 4/4/2 |
| bbnr | +0.005±0.006 5/1/4 | −0.006±0.010 2/3/5 | +0.000±0.006 2/4/4 |
| enn | +0.013±0.009 5/2/3 | −0.004±0.015 3/3/4 | −0.001±0.008 2/5/3 |
| energy_compress | +0.040±0.018 7/2/1 | +0.017±0.022 3/4/3 | +0.001±0.009 3/4/3 |
| knn_shapley | −0.002±0.014 2/5/3 | −0.025±0.018 2/2/6 | −0.024±0.006 1/3/6 |
| loo_value | +0.004±0.048 6/0/4 | +0.012±0.014 4/4/2 | +0.003±0.010 3/6/1 |

Regression RMSE after finetune vs provenance_only (11 tasks; reference 0.451 / 0.429 / 0.419; negative is better):

| policy | K 0.2 | K 0.5 | K 0.75 |
|---|---|---|---|
| rc_cnn | −0.031±0.005 7/1/3 | −0.019±0.004 3/2/6 | −0.011±0.001 4/6/1 |
| crr | +0.014±0.021 3/0/8 | −0.004±0.004 3/1/7 | −0.007±0.008 2/4/5 |
| bbnr | +0.016±0.009 3/2/6 | −0.006±0.005 3/3/5 | −0.003±0.004 3/5/3 |
| enn | +0.018±0.018 2/2/7 | −0.001±0.010 3/3/5 | +0.006±0.011 1/5/5 |
| energy_compress | −0.039±0.004 7/3/1 | −0.020±0.001 4/5/2 | −0.011±0.003 3/7/1 |
| knn_shapley | +0.086±0.013 1/1/9 | +0.062±0.012 0/1/10 | +0.042±0.010 1/1/9 |
| loo_value | −0.020±0.008 4/4/3 | −0.014±0.004 2/4/5 | −0.007±0.002 1/7/3 |

Against random at K 0.2: energy_compress +0.026 (8/0/2) in classification and −0.054 (9/2/0) in regression; rc_cnn
+0.021 (5/0/5) and −0.046 (8/3/0); loo_value −0.035 (8/1/2) in regression; knn_shapley worse than random in both.

Retained at K 0.2 (accuracy ratio to full memory for classification; RMSE ratio for regression, lower is better):
energy_compress 0.976 / 1.053, rc_cnn 0.969 / 1.145, provenance_only 0.929 / 1.293, random 0.950 / 1.441, crr 0.904 /
1.425, knn_shapley 0.929 / 1.745.

Latency: the new policies take 0.17-0.33 s of maintenance against 0.11-0.14 s for the recorded rules. Lookup time
depends on K, not on the policy (batch lookup about 1 ms at K 0.2 against 13-15 ms on full memory). C/H reliability
(Spearman of a query subset against the full set, mean over 21 tasks): C 0.62 / 0.78 / 0.90 at 10 / 25 / 50%, H 0.54 /
0.71 / 0.84.

Reading: at K 0.2 energy_compress and rc_cnn beat provenance_only and random on most tasks; energy_compress is the
most consistent. Our knn_shapley adaptation is clearly worse. At K 0.75 nearly everything ties. In regression, car
alone contributes about −0.22 RMSE to rc_cnn and energy_compress at K 0.2. The raw-event check was first made on the
9-task version (`raw_event_check.txt`) and rerun on 2026-11-27 on the 10-task version (`raw_event_check_10task.txt`).
Both match the tables to all printed digits, including W/T/L (e.g. energy_compress K 0.2 classification +0.040387, sd
0.018245, 7/2/1; with `--exclude wine:s32` the old 9-task value +0.043845, 6/2/1 comes back exactly).

Tie check for classification wine s30/s31 (recorded with the old engine member rule). xps8960 was unreachable, so the
check ran read-only on r760 from the same recorded `_core` weights and data snapshots, with the fixed code. Rows whose
top-k member set differs between the two rules at full memory: s30 0 of 106, s31 0 of 106, s32 2 of 106 (the known
case). Rows with any exact float32 tie between two eligible cases: s30 0, s31 0, s32 106 of 106. Without any tie the old
and fixed rules pick the same members under every case mask, so the recorded s30/s31 results equal what the fixed code
gives on r760. The smallest gap between adjacent scores is 65 float32 ulps (s30) and 18 ulps (s31); a tie on xps8960
would need the two hosts to differ by at least that much, which is unlikely but not measured.

### p2 on the 7 synthetic sets (`handoff_synnewpol_r760`, seeds 30-32)

Untuned masked `_base` core, finetune 30 epochs at lr scale 0.1, compared with the 8 recorded policies of
`handoff_synret_r760` on the same core (483 and 525 complete units; `fullmem_*` equal on all 42 task-seed pairs). Same
conventions as the real-data p2 table; W/T/L over 7 tasks.

Classification accuracy after finetune vs provenance_only (reference 0.836 / 0.835 / 0.824; full memory 0.820):

| policy | K 0.2 | K 0.5 | K 0.75 |
|---|---|---|---|
| knn_shapley | +0.030±0.009 6/1/0 | +0.030±0.011 7/0/0 | +0.035±0.009 7/0/0 |
| enn | +0.012±0.015 4/1/2 | +0.007±0.015 5/0/2 | +0.013±0.013 5/2/0 |
| bbnr | −0.002±0.021 3/0/4 | −0.013±0.011 1/0/6 | +0.001±0.018 1/3/3 |
| loo_value | −0.004±0.001 4/1/2 | −0.013±0.009 2/2/3 | +0.004±0.014 3/1/3 |
| rc_cnn | −0.006±0.010 3/0/4 | −0.010±0.002 0/4/3 | −0.005±0.005 2/3/2 |
| energy_compress | −0.011±0.011 1/3/3 | −0.025±0.007 0/1/6 | −0.010±0.006 2/1/4 |
| crr | −0.124±0.103 1/0/6 | −0.017±0.025 2/1/4 | −0.003±0.016 2/0/5 |
| random (recorded) | +0.006±0.018 4/0/3 | −0.004±0.009 2/2/3 | +0.006±0.017 3/2/2 |

Reading: knn_shapley is clearly best here (0.866 at K 0.2 and 0.5, above full memory), with the largest gains on noise,
combined and shift. This reverses the real-data result, where it lost at K 0.5 and 0.75. A likely reason is that the
synthetic memories hold many mislabelled cases, which a low data value removes. enn is mildly positive; rc_cnn, bbnr,
loo_value and energy_compress are at or below random; crr collapses at K 0.2 (0.713 after finetune, 0.533 before).
C/H reliability at 10 / 25 / 50% of the queries: C 0.54 / 0.70 / 0.83, H 0.41 / 0.57 / 0.78. A raw-event recompute
matches the table (knn_shapley K 0.5: +0.030403, sd 0.011278, 7/0/0).

### p3

Same b/t/w/i rule as p0.

- (a) Percentile initializer, pre-top-K all-case arm (12 tasks, seeds 70-72): Δ test_error −0.00001 / +0.00005 /
  +0.00006 for pct 10 / 25 / 50 against knn_mean k = 5 (0.2016). In sync, pct50 with the rr conditions lowers NLL on 8-9
  of 10 classification tasks (alternating_rr −0.044, 8/1/1; rr_init_bias_radius −0.035, 9/0/1), accuracy moves less
  than 0.01, regression RMSE is mixed. The free radius is flagged `overly_broad` more often as the percentile rises
  (rr_initial_bias_radius 11 / 20 / 35 of 63 rows against 9 in the reference).
- (d) probability_vector_distance: close to neutral (accuracy +0.001; NLL +0.001 to +0.002).
- (b) per_case_2022 vs the aggregate adapter: classification accuracy +0.004 (diff_only 4/4/1/1, combined 5/1/2/2).
  Regression RMSE +0.072 / +0.068, almost all from body_fat seeds 30 and 31 (d = +1.0 and +1.5); without body_fat
  −0.004 / −0.008. On synthetic_nominal: 0.43-0.44 accuracy against 0.76.
- (c) Direct head: accuracy 0.849 against 0.905 (−0.056, 4/2/4/0), with the loss concentrated in zebra (0.485 vs
  0.939) and zebra_special; NLL −0.013 (7/1/2). synthetic_nominal 0.463 vs 0.748.
- (e) omit_nominal_channel on synthetic_nominal: every condition loses 0.29-0.43 accuracy and lands near
  retrieval-only (about 0.37). Removing Δu_q removes the adapter's gain on nominal data. One synthetic task only.

### p4

Error as in p1. W/T/L counts tasks where the reference wins, ties or loses.

TabPFN (seeds 50-52, 24 of 29 tasks): TabPFN error 0.066 ± 0.002 (classification, 13 tasks) and 0.381 ± 0.003
(regression, 11 tasks).

| method | cls diff vs TabPFN, TabPFN W/T/L | reg diff vs TabPFN, TabPFN W/T/L |
|---|---|---|
| masked F | +0.027, 11/1/1 | +0.042, 11/0/0 |
| all-case F | +0.053, 11/0/2 | +0.035 (10 tasks), 10/0/0 |
| CatBoost | +0.007, 8/2/3 | +0.028, 8/0/3 |
| TabM | +0.068, 10/1/2 | +0.036, 10/0/1 |
| MLP | +0.081, 12/0/1 | +0.055, 11/0/0 |
| kNN | +0.067, 11/1/1 | +0.185, 11/0/0 |

Shift (seeds 30-32, 9 classification and 11 regression tasks, reference shifted F; error 0.240 cls, 1.031 reg):

| method | diff vs F, cls (F W/T/L) | diff vs F, reg (F W/T/L) |
|---|---|---|
| retrieval_only | +0.037 (6/0/3) | +0.073 (9/0/2) |
| reuse arm | +0.037 (6/0/3) | +0.114 (9/1/1) |
| kNN | +0.032 (5/0/4) | +0.257 (9/0/2) |
| MLP | +0.017 (5/0/4) | +0.276 (7/0/4) |
| CatBoost | −0.079 (3/0/6) | +0.185 (7/0/4) |

- Every method is worse on the far half (F: +0.103 classification, +0.676 regression). F's near-to-far gap is not
  smaller than the baselines', so there is no sign that F extrapolates better.
- Adaptation helps under shift (F beats retrieval_only on 6/9 and 9/11 tasks, more so on the far half); the frozen-core
  reuse arm does not.
- Shift retention at K 0.5 costs more than without shift (random +0.043 / +0.069 shifted vs +0.023 / +0.013 unshifted).
- Regression means are skewed by car (about 5 RMSE for every method), CatBoost on energy_efficiency and MLP on
  body_fat; W/T/L is the more reliable reading. iris, breast_cancer and zebra have very large seed sd under shift.

## 4. What failed, what was fixed, what was rerun

1. **p1 allcaseF classification wine (r760) and tabarena_363631 s50 (r760)**: "Trying to backward through the graph a
   second time" in the adapter warm-up of `train_synchronized`. Cause: the feature-size probe ran with gradients in
   eval mode; on wine s50 the trained mlp maps 11 of 106 cases to all-zero features, so the probe recomputed those
   cache rows under autograd and the warm-up later backpropagated through that stale graph twice. Fix 4235e10: the
   probe runs under `torch.no_grad()`. Old and new code give bitwise-equal weights on four configs that did not fail.
   Both jobs reran on r760 with exit 0 (wine 15:25 UTC, 363631 s50 18:15 UTC). The s50 units recorded before the crash
   were loaded, not retrained. p1 tables above include them.
2. **p2 newpol classification wine s32 (xps8960)**: "Activation engine disagrees with model.retrieve by 0.332". Cause:
   with an exact tie at the k-th place, the engine's own member rule (float64 argpartition) kept different cases from
   `model.retrieve` (float32 `torch.topk`). Fix 4235e10: tied rows take the model's members; untied rows are unchanged.
   s32 reran on xps8960 (22 new units, `_core` reused), exit 0, placed on r760. p2 tables above include it. On
   2026-11-27 the tie check showed that s30/s31 have no tied rows (computed on r760; see p2 above).
3. **p0 (a) delta_from_init** could not be expressed without code (the store was seeded after training, so B was
   constant). The opt-in key came in 4235e10; 6ece140 made `t1_run` refuse invalid `initial_bias` values and the
   combination with `reliability` before any training. p0a ran 58 jobs on the three hosts, all exit 0.
4. Review fixes before any host run (no recorded unit affected): p0 added the real-data rare10 task; p1 added
   F_no_penalty; p2 fixed energy_compress drift and tie order; p3 fixed the cdist shortcut in pvdist; p4 added per-half
   metrics for regression and baselines.
5. A p0 report count was corrected (anneal rare10: 5 identical and 7 tied cells, not 6 identical).

## 5. Not done

- **Human study**: needs participants and an IRB determination (Berry's review process). Not started.
- **GPU image and text suites**: not run. The T1 benchmark strategy (2026-09-12) moved images and text out of the
  required T1 matrix, and the owner set GPU/XPU work aside.
- **Heuzi's adaptation work**: it arrived on origin (the `adaptation improvement/` folder on main and new commits on
  neural-cbr) and was merged on 2026-11-27 (main c3d6a93, neural-cbr ff2bd3a). Nothing in these packs uses it, and it
  is not summarized here.
- **TabPFN on 5 tasks**: car and regression wine exceed 3000 training rows; 363681, 363705 and 363631 are estimated at
  25-37 CPU-h per seed on CPU (extrapolated, not confirmed). Only TabPFN v2 was used.
- **v_small is not re-selected for all-case**; the frozen masked value is kept. The three-way penalty comparison runs
  only on the 18 tasks with v_small > 0. The correction-magnitude tail and the fraction of corrections above tau_task
  are not recorded and would need a read-only pass over the weights.
- **All-case core budget**: 300/40 against the grid's 600/60; 8 of 87 all-case cores hit the cap. The zebra collapse
  (seeds 50 and 52) is not investigated.
- **revision_stress and synnewpol details**: the masked revision_stress rows were recorded at an earlier commit, so code
  drift between the two studies cannot be fully excluded (random and oracle select the same cases in both). ECE, the
  per-group accuracies and the flip matrices are recorded but not analyzed.
- **p4 detail**: all-case F has no regression diabets row in the TabPFN table (the summary reads only `tuned__F`, and
  diabets' F is the reuse arm). Shifted baselines have no TabM or TabPFN and are not re-tuned. synthetic_diag is out
  of the shift protocol.
- **p3 detail**: per_case_2022 on body_fat (d up to +1.5) looks like an instability and is not investigated. p3
  references were recorded at older commits (204c49b5, 3ab1549c; pre-top-K reference 2026-10-30), so differences
  around 1e-4 can mix in code drift.
- **p2 detail**: the wine s30/s31 tie check ran on r760, not on xps8960 where the units ran (xps8960 was unreachable;
  scripts in `/var/tmp/nnknn-ties1127/` on accel). Other newpol, latency and synnewpol units recorded before fix 4235e10
  have not been checked for ties in the maintained cores.
- **MCB momentum** was swept only on 3 sync_mcb schedules, and its pairs do not share a core.
- **Backup**: r760 `runs/` is still the only copy of every unit. Measured on 2026-11-27: 689.4 GB in 170,081 files
  (627.5 GB event files, 61.9 GB weights); accel has 14.0 TB free; the ssh tunnel gives about 11 MB/s, so a full copy
  takes about 17 hours uncompressed or about 5 hours with zstd on the wire. The proposed overnight rsync (command and
  verification steps in `/srv/cyclone/workspace/NN-KNN/reports/2026-11-24-t1-multi-host-ops.md` on accel, not in git) has not been started; that needs the
  owner or the experiment runner.
- **Housekeeping**: the r760 budget file `logs/t1par/budget` is 28 (our `budget_follow.py` rewrote it from 15 before
  it was stopped). Two shell wait loops of unknown owner (pids 217579 and 227436, `pgrep -f "nnknn-hb/dump.py
  syncallcase"`) were left alone.

## 6. Files

- Results: `results/t1/handoff_20261126/{p0,p0a,p1,p2,p3,p4}/`; p2 9-task version in `p2/excluding_cla_wine_s32/`.
- Configs: `configs/t1/handoff_20261126/{p0,p0a,p1-allcase,p2-retention,p3-core-reuse,p4-baselines-stress}/`.
- Job lists and build/validation scratch: `/var/tmp/nnknn-handoff1126/<pack>/` on accel.
