# T1 progress review, 2026-11-24: problems found and the tests queued for them

Review of the earlier T1 progress (reports 10-06, 10-20, 10-27, the PI discussion notes of 10-30/31 and the r760
reproduction so far), through three lenses: claims against the r760 `runs/`, design of the experiments, and the
tests and diagnostics. Every problem below has its evidence and the experiment or test queued for it.
Everything queued is **exploratory** (the evaluation seeds 40-42 and 30-32 have been seen before). Nothing here is
confirmatory, and nothing may be pooled with the 2026-10-20 confirmation.

Times are UTC. Hosts: r760 (Xeon 4509Y, AVX-512), q-lab (Ryzen 7950X, AVX-512), xps8960 (i7-13700, AVX2).
Commits on `neural-cbr` (accel, not pushed): `15d669b` (configs, `t1_summarize` fix, tests) and `fc4bc34`
(grid configs for the three larger tasks). They were deployed by bundle and `--ff-only` to all three hosts. r760
moved from 204c49b to fc4bc34; this adds files, changes `tools/t1_summarize.py`, and touches no code that a running job
uses. The supervisor was not restarted.

New configs: `configs/t1/review_20261124/` (frozen configs untouched; generator `/var/tmp/nnknn-multihost/review/mkcfg.py`).

## Problems and what is queued for each

### A1. Every running and reported F result uses the masked core that the PI replaced
**Evidence**
- `_base.yaml` and every frozen F config set `pre_topk_mask: true` (top-k 3-10).
- S3, S5 and S6 train only masked cores.
- The 10-31 decision says that masked observations must not be assumed to hold under all-case softmax.
- No matched all-case run of the confirmation arms exists.

**Queued: study `review_allcase_twin_r760` on r760 (supervisor extras)**
- Configs: `confirm_allcase/<task>.yaml`, 29 tasks (QSAR excluded). Each is a copy of `confirm_final_20261013/<task>.yaml` with seeds 40-42 and `pre_topk_mask: false` in `core` and `core_by_dataset`. Nothing else changes: same F, F_global and alternating arms, same schedule, same weights.
- One job per task, all 3 seeds in one job.
- It pairs seed by seed with S3 `confirm_repro_r760` on the same host and the same splits.
- Metrics: per task, the paired log ratio of all-case to masked for retrieval-only and for F, and E1 within all-case (F vs retrieval-only).
- Cost: about 22 CPU-h, the same as S3.
- Started 09:03. The 15 light tasks were done by 09:10. The six large OpenML tasks (363631, 363681, 363702, 363693, 363623, 363686) take 1.5-4 h each. **Expected finish 13:30-14:30.**

### A2. "All-case beats the mask" is confounded with an untuned K = 5 and tau = 1, and with checkpoint selection
**Evidence**
- `pre_topk_long_*` fixes tau 1 and top_k 5 in both arms.
- Energy favours the mask every time (+26.5% at 400 epochs).
- In the core ablation a fixed larger K often matches all-case: digits k40 0.706 vs k_all 0.754; breast k20 0.696 vs 0.824.
- Masked classification selection runs on a loss with a step at the 1e-8 floor. Balance: 9 of 125 validation queries have true-class mass 0, and they carry 90% of the validation loss; the masked arm's best epoch is 7, while the all-case arm runs to the cap.

**Queued: q-lab, five studies (one per file, because the unit keys would collide)**
- `review_ktau_{k10,k20,k40,tau025,tau050}_r760`. Each is a copy of `pre_topk_long_20261121_a-f`: 12 datasets, seeds 70-72, 400/60, with top_k 10/20/40 or tau 0.25/0.5. One job each, run by `t1_pre_topk_pilot.py`.
- Analysis: per dataset and seed, choose K for the masked arm and tau for the all-case arm by validation loss, then compare test errors paired by seed. Also count best epochs at the cap.
- Cost: about 4-5 CPU-h. Started 09:01. **Expected finish about 10:00.**

### A3. The mask × joint pilot (X2) compares an unconverged all-case pretrain with a converged masked one
**Evidence (read from the `mask_joint_r760` units)**
- The all-case pretrain ran to the 300-epoch cap in 12 of 18 runs: energy, iris, wine and yacht, every seed.
- The masked pretrain hit the cap in 0 of 18 runs.
- So a negative all-case "selected drift" can be continued convergence rather than protection from joint training.

**Queued: q-lab, study `review_maskjoint_long_r760`**
- Config `mask_joint_long.yaml`: the same pilot with pretrain 600/60. Same 6 datasets, seeds 70-72, all 4 arms.
- Cost: about 1-1.5 CPU-h. Started 09:01. **Expected finish about 10:00.**

### A4. The tuned NN-kNN core is under-searched; kNN beats it on 5 of 13 ablation tasks
**Evidence**
- Grid winner relative to kNN: car 0.817, wine 0.399, breast 0.782, zebra_special 0.766, credit-g 0.959.
- Untuned one-factor variants beat the winner by large margins, e.g. digits tau2 0.583 and wine `mlp128_64_lrf_equal` 0.300.
- The grid screens at 60/15, while good variants select epochs 96-275.

**Queued, part 1: all-case core ablation, study `review_coreabl_allcase_r760`**
- Config `core_ablation_allcase.yaml`: the 13 tasks of `core_ablation_r3_20261124`, seeds 30-32. 16 variants, all on an all-case base:
  - tau 0.1-2
  - lr_case 0.003 and 0.03
  - frozen bias, frozen feature weights
  - untrained soft kernel at tau 0.1-1
  - `mlp128_64_lrf_equal`
  - a long budget (1000/100)
- Each task runs on the host where its masked `core_ablation_r760` run ran: car, digits, 363625, energy, balance and 363626 on q-lab; the other 7 tasks on xps8960.
- The xps8960 part was done at 09:03-09:10. q-lab's part starts once the masked core ablation there has finished. **Expected finish about 11:00.**
- The baseline columns (CatBoost/TabM ratios) are empty on q-lab/xps8960 because the grid's baseline units are not there. Rebuild the summary on r760 with `--summarize-only` after the sync.

**Queued, part 2: NN-kNN-only grids, studies `review_gridallcase_r760` and `review_gridmaskedlong_r760`**
- Configs `grid_allcase.yaml` (all-case, tau 0.25-2) and `grid_masked_long.yaml` (the 10-09 masked grid). Each has 32 candidates and the same longer budget: screen 150/30, refit 600/60.
- Development seeds 20-21, evaluation seeds 30-32. The 10-09 grid's baseline rows exist on these seeds.
- Comparing the two grids separates the mask effect from the budget effect. `grid_masked_long` against the 10-09 grid gives the screening-budget effect.
- Where each task runs:
  - The 17 small tasks: the 8 heavier ones on q-lab, the 9 lighter ones on xps8960, both grids of a task on the same host.
  - car, abalone and regression wine (`*_big.yaml`) as r760 extras.
- Cost: about 15-20 CPU-h for the small tasks and about 30 CPU-h for the big three. **Expected finish 12:00-14:00.**
- Not done: an lr_feature/lr_case ratio factor. It needs a code change in `nnknn_core_settings`, and the factor is partly covered by the `lrf_equal` variant.

### A5. The value of training the case bias is not established
**Evidence (core_ablation_20261027)**
- `frozen_bias` beats the trained winner on car (0.861), digits (0.732), balance (0.872) and zebra (0.000).
- Untrained `soft_knn` beats it on car and breast.
- Retention policies B and Q assume the learned bias helps.

**Queued:** covered by `review_coreabl_allcase_r760` (frozen-bias, frozen-feature-weight and untrained soft-kernel arms under all-case, with a temperature sweep as the bandwidth-tuned null). First result, yacht, 3 seeds, test std-RMSE:
- trained all-case base: 0.152
- frozen bias: 0.354
- untrained soft kernel, best tau: 0.612
- all-case `mlp128_64_lrf_equal`: 0.069

So on yacht, training does matter.

### A6. Under all-case softmax the provenance evidence rule passes every case
**Evidence**
- R_i counts weights above 1e-6. With the `_base.yaml` rule (R ≥ 2, A ≥ 0.1), all-case gives R_i ≥ 2 for 100% of cases on breast, digits and energy. Masked shares: 0.37, 0.58 and 0.79.
- New test: `tests/test_t1_review_20261124.py::test_evidence_rule_separates_cases_under_all_case_softmax`, strict xfail.
  - On synthetic_diag the all-case evidenced share is 1.0.
  - Its masked control passes.

### A7. Retention results partly measure case-ID tie-breaks and mask-induced zero exposure
**Evidence**
- 10-27 Table 20: 98% of abalone cases are never retrieved, and 99-100% of Q evictions are decided by case ID.
- No config sets `retention.tie_break`; the default is `case_id`.

**Queued for A6 and A7: xps8960, study `review_retention_r760`**
- 8 configs `retention_{cls,reg}_{mask,allcase}_{caseid,random}.yaml`: K 0.5; full_memory, random, provenance_only, bias_current.
- Datasets: iris, wine, breast, balance, digits; yacht, energy, airfoil, student, abalone (max_train 1500).
- Seeds 30-32. 40 jobs (config × dataset, 3 seeds each).
- Metric: paired change against full_memory and random, plus the evidenced and tie shares from the maintenance summaries.
- Cost: about 3 CPU-h. Started 09:01; 19 of 40 jobs were done by 09:07. **Expected finish about 09:30.**

### A8. Masked-arm numbers are host-dependent; the numeric golden passes only on r760
**Evidence**
- r760 vs g234: all-case columns of `pretopk_long` and `mask_joint` are identical, but the masked columns differ.
  - yacht: -5.0% → -2.5%
  - airfoil: +2.0% (1/2/0) → +5.1% (0/3/0)
  - balance: 0.0% → -6.1%
  - The notes' count "3 better / 2 worse / 1 tie" becomes 2/2/2.
- TabM baselines differ by up to 6.9% (car); CatBoost is identical.
- `test_defaults_reproduce_the_recorded_toy_outputs` passes on r760 and fails on q-lab (relative difference 9.6e-5), xps8960 (7.9e-5) and accel (5.7e-5). The tolerance is 1e-5.
- xps8960 and q-lab trained a large part of S5.

**Queued: same configs, new study per host**
- `review_xhost_maskjoint_{xps8960,qlab,r760rep}` and `review_xhost_pretopk_{xps8960,qlab,r760rep}`.
- The xps8960 replicate was done at 08:51. The r760 replicate started at 09:03 (3 of 7 shards done at 09:07; shard a takes about 30 min), expected about 09:40. The q-lab replicate runs after its other jobs.
- Analysis: per-seed masked-arm spread across the three CPUs, and same-host determinism on r760 (r760rep vs `pretopk_long_r760` / `mask_joint_r760`). It needs no training beyond these 2 CPU-h.

### A9. Seed-level harmful adapter corrections and mixed code in one S3 seed
**Evidence**
- car s40: F post 0.2721 vs pre 0.2218 vs retrieval-only 0.2068.
- 363631 s41: F post +31.5% over pre. Its retrieval-only and core units have fingerprint 281f14cc0999, and F has 8c9b46985503.

**Queued: r760, study `review_rerun363631_r760`**
- Frozen `confirm_final_20261013/regression__tabarena_363631.yaml`, seed 41, under a new study, so nothing is retrained in place.
- **Expected finish about 11:00.**
- The per-seed post/pre table over all 29 × 3 S3 seeds is analysis only. It is left to the S3 report, together with A10.

### A10. The r760 reproduction report cannot test E1-E3b as it stands (analysis, no training)
**Evidence**
- `t1_confirm_report` pools 4 tasks, includes QSAR, and has no task-exclusion option.
- Classification log ratios with eps = 0.5/n_test swing to extremes on tasks with 0-5 test errors, e.g. zebra E1 0.067.

**Not queued:** this needs new analysis options (`--exclude-task`, eps/pooled-count variants, a host-baseline sensitivity). It is CPU-trivial and belongs with the S3 report once S3 is complete.

### A11. `t1_summarize` counted "better" seeds the wrong way for error metrics (fixed)
**Evidence**
- `--higher-is-better` was `store_true` with default True and was never read.
- `seeds_better` counted d > 0 even for rmse, loss, drift and churn. The 10-06 report's paired table carries the inverted labels.

**Fix (`15d669b`)**
- The direction now comes from the metric name, with `--lower-is-better M` / `--higher-is-better M` overrides.
- The output columns are unchanged. Recorded units are unaffected.
- Tests: `test_summarize_counts_lower_error_as_better`, `test_summarize_direction_override`, `test_metric_direction_from_name`.

### A12. Two split properties that bias the confirmatory comparison (documented, not changed)
Changing either would change every recorded unit, so both are strict xfail tests in `tests/test_t1_review_20261124.py`.
- **regression__wine (prereg primary task):** 22.3% of test rows (seed 30) repeat a training row exactly, features and target alike. 1-NN RMSE on those rows is 0.000. Test: `test_regression_wine_test_rows_are_not_train_duplicates`.
- **Non-OpenML tasks:** a fresh holdout per seed means 67-81% of the test rows of evaluation seeds 40-42 were validation rows under development seeds 20-25 (iris, yacht, regression wine). Test: `test_evaluation_test_rows_were_not_development_validation_rows`.
- What follows from this: report wine with and without the duplicated rows, from recorded weights. A fixed outer holdout needs a prereg addendum, which is a PI/owner decision.

Test runs:
- accel: new file plus `test_t1_records.py`: 31 passed, 3 xfailed.
- r760, q-lab, xps8960: new file 4 passed, 3 xfailed. Golden as in A8.

## Run summary

| study | host | experiments | seeds | expected finish |
|---|---|---|---|---|
| review_allcase_twin_r760 | r760 | 29 tasks × F/F_global/alternating/RO/reuse, all-case | 40-42 | 13:30-14:30 |
| review_rerun363631_r760 | r760 | 363631 frozen config | 41 | ~11:00 |
| review_gridallcase_r760 / review_gridmaskedlong_r760 | r760 (car, abalone, wine-reg), q-lab (8 tasks), xps8960 (9 tasks) | NN-kNN grids, 32 candidates each | dev 20-21, eval 30-32 | 12:00-14:00 |
| review_ktau_{k10,k20,k40,tau025,tau050}_r760 | q-lab | 12 datasets × masked/all-case | 70-72 | ~10:00 |
| review_maskjoint_long_r760 | q-lab | 6 datasets × 4 arms | 70-72 | ~10:00 |
| review_coreabl_allcase_r760 | q-lab (6 tasks), xps8960 (7 tasks, done) | 16 all-case variants | 30-32 | ~11:00 |
| review_retention_r760 | xps8960 | 10 datasets × 8 configs × 4 policies | 30-32 | ~09:30 |
| review_xhost_* | xps8960 (done), r760 (~09:40), q-lab | mask_joint pilot, pre_topk_long a-f | 70-72 | q-lab ~13:00 |

The total is about 90-100 CPU-h.

How the runs are executed:
- **q-lab:** runs the review list with a second worker in its own control folder, `logs/t1par/xrev/`.
  - Budget 7 while the earlier worker finished `core_ablation_r760`; a watcher sets 22 after that.
  - All review jobs are also in `logs/t1par/xjobs.json`, so the accel sync lists them and places them on r760.
- **xps8960:** uses the usual `logs/t1par/xjobs.json` worker.
- **r760:** runs the review jobs as supervisor extras.
  - They have no priority line. A catch-all `S5` line was added at the end of `priority.txt`, so every S5 job ranks before them.
  - S3/S5 had drained, so they started at once.

Job lists: `/var/tmp/nnknn-multihost/review/mkjobs.py` and `/var/tmp/nnknn-sync/xjobs_{q-lab,xps8960}.json`.

## Left out, and why
- **B1f correction-size arms (PI point 3) and declared all-case schedules or loss weights:** the PI asks to declare them first, which is a PI decision. A1 keeps F's frozen schedule instead.
- **B2 feature standardisation and gates, B3 / step G retention test, Phase 2 per-case adapter, Phase 6a step 2, Phase 6b comparators, CBM study, 10-27 P1-P7:** not implemented, or they need a PI addendum.
- **Human UI study:** needs participants and an IRB.
- **Batch size and schedule study (PI point 8), categorical encoding and native-categorical CatBoost (point 9):** point 9 needs code. Point 8 waits for A2/A4 to fix the all-case baseline it should run on.
- **frozen_reuse arm added to S3 (claims review P2b):** it would add units to the running `confirm_repro_r760` study. Left out; the 363631 rerun covers the mixed-fingerprint part.
- **Test NLL / McNemar post-hoc tool, wine de-duplicated scoring, unseen-row re-scoring, `t1_confirm_report` options, CBM diagnostics on S5 retention units (claims P7), encoder/EMA summaries (P8):** analysis on recorded units, with no training. They are left for the reports after S3 and S5 are complete. P6 noted that the S5 readers need a per-dataset filter first: one tree took over 15 min and 16 GB.
- **QSAR (363697):** excluded everywhere, as ordered.
