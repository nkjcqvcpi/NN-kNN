# Neural CBR (T1) progress report: aligned results carried over from the 6 October handoff, and new results since 10 October

Period: 2026-09-14 to 2026-10-13 about 11:55, local time (PDT, UTC-7). Dates are written 10-12 for 2026-10-12.
Chinese version: `NEURAL_CBR_PROGRESS_2026-10-13_ZH.md` (same sections, same tables, same numbers).

## 0. Scope, basis and trust rules

- **Covered:** neural CBR (T1) only: what the model can do now, and what the experiments found. **Not covered:** RL (T2), LLM (T3), and Git or branch bookkeeping.
- **Two handoffs.** The 6 Oct handoff is `NEURAL_CBR_PROGRESS_2026-09-14_to_10-06_ZH.md` with the slides of that meeting. The 10 Oct handoff is `REORGANIZATION_20261010.md`. This report carries over the results of the first that are aligned and still trustworthy, covers the days in between (10-06 to 10-10), and adds everything since 10 Oct.
- **Aligned** means produced by the pipeline rebuilt on 10-02 to the PI's 2026-09-20 specification (pipeline revision `pi_20260920_v2`). **Trusted** means the numbers reproduce from the source files and no later fix or document replaces them. A trusted result may still carry a caveat that changes how it may be read.
- **How the tables were made.** Every table of results was computed from the raw result files by two independent builders, and a program compared the two cell by cell. Static tables (status, caveats, design) were written from the project's own notes and from two independent supersession audits, which also re-ran sampled cells at the current code. Statistics follow the project's convention (Wilcoxon p from `scipy.stats.wilcoxon` with default arguments). No new experiment was run for the report text itself; the audits re-ran sampled cells from the saved configurations at the current code (Table 5, Table 6), and E17 was replayed only, not re-run.
- **Reading rule.** Every claim in the text points at the table it comes from. Numbers in running text are copied from those tables.
- **Live numbers** (running jobs) are a snapshot from about 11:55 on 10-13 and keep changing.

## 1. Summary

**Table 1.** Summary of findings and the table that supports each

| # | Finding | Evidence |
| --- | --- | --- |
| 1 | None of the five groups of results shown at the last meeting carries over (six tables on the slides): they come from the first implementation, which was deleted on 09-29. Of the summary slide's three conclusions, two stand on aligned runs (no consistent gain from removal; the maintenance split improves 18 of 18 pairs) and the third (RRR helps regression) is E5, which E17 does not replicate | Table 2 |
| 2 | Of the 16 aligned experiments of 10-02 to 10-04, 11 are trusted with a caveat, 3 are engineering checks that stand, and 2 (E1, E3) keep only engineering facts because their task numbers are replaced (E2 for E1; E4 and E5 for E3). Re-runs at the current code are bit-identical on the sampled cells for E7 to E16, and for E4 apart from Iris seed 5; E6 has 16 of 18 cells identical (the two Iris seed 8 cells differ by 4.4e-8) | Table 5 |
| 3 | The retraining, small-candidate and Digits results say: retrained removal gives no accuracy gain at higher cost, there is no Digits collapse, and Q or B against random is mixed | Table 7 to Table 10 |
| 4 | Capacity-matched RRR synchronization looked better on regression in E5 and worse on Energy in the 10-09 pilot | Table 13 |
| 5 | Adapted removal, revision and reviewer-control effects are small and mixed; the maintenance split improves RMSE in 18 of 18 pairs, with an overfitting risk | Table 19 to Table 22, Table 15 to Table 17 |
| 6 | The EMA encoder is not useful as a default. At default settings the raw-feature core has lower median error than the shared and every EMA encoder arm; on the six tuned-MLP tasks no EMA arm has a lower median error than shared | Table 25, Table 26, Table 28 |
| 7 | A batch-size mismatch corrupted the first schedule search: digits cores collapsed and 8 of 80 cores reproduced the grid. At batch 64 the joint schedule's gain over the default is -1.8% (p = 0.323), against -11.6% (p = 0.008) at batch 32 | Table 30, Table 31, Table 32, Table 33 |
| 8 | Tuning so far: H6 adopted per-task selection; H1 rejected; H2 gave no reliable accuracy gain and, after H2c, kappa 10 is adopted for classification as non-inferior; H5 and H3 are running | Table 39, Table 40, Table 42, Table 49 |
| 9 | The 2026-10-09 grid stands at 108 of 110 jobs; the interim comparison puts NN-kNN behind CatBoost and TabM on most tasks; the transfer grid is part-way through (current counts in Table 48 and Table 49) | Table 47, Table 45, Table 46, Table 48 |
| 10 | No result of the final configuration F (joint schedule, reuse, maintenance) exists on the pre-registered evaluation seeds (40 to 49); the transfer grid already holds baseline and NN-kNN evaluations there. The confirmation phase has not started | Table 38, Table 48, Table 49 |


Caveats and open points are collected in Table 51 (section 9).

## 2. What the last meeting showed

**Table 2.** Results shown at the last meeting (6 Oct slides) and whether they carry over

| Slide | What it showed | Status | Why |
| --- | --- | --- | --- |
| Summary slide ("Experiments") | Three one-line conclusions: removal gives no consistent gain; RRR synchronization helps regression but hurts Iris; selecting on the maintenance split improves 18 of 18 pairs, with an overfitting risk. It also lists the classical CBM baselines, the revise stage, Covertype, the T1 core and the 5-schedule trainer, then the provenance, retention, maintenance, reuse, sync, MCB and revise harness. | Partly carried over | The first and third conclusions come from aligned runs and stand with caveats (Table 7, Table 19 to Table 21, Table 22). The second is E5; the later E17 pilot does not replicate it (Table 13), so treat it as descriptive. The baselines, revise stage, Covertype, the first implementation's T1 core and the 5-schedule trainer belong to the first implementation and do not carry over. |
| Main benchmark | Random, Bias-Only, DROP3, ICF, Core-Set, Ours (Retain), Ours (Retain+Reuse) at 50% capacity, seeds 42 to 46, on breast_cancer, covtype (3000-row subsample), digits, iris, synthetic, wine; p vs Bias and vs DROP3 | Not carried over | From the first implementation (code and results deleted on 09-29, dated 09-09 to 09-17; this benchmark is from 09-14), deleted because its report hard-coded conclusions and its benchmark never trained the retriever. The Covertype scaler was fitted before the split. On the slide itself Core-Set is best on 4 of 6 datasets (digits, iris, synthetic, wine) and Ours (Retain+Reuse) on 2 (breast_cancer, covtype); Ours (Retain) is below ICF or Core-Set on 5 of 6; only iris against DROP3 has p below 0.05 (0.006), the other 11 p-values are 0.108 to 1.000. |
| Capacity frontier | bias_only, drop3, provenance_bias_coverage, stratified at capacity 0.20 to 1.00 on breast_cancer, synthetic, wine | Not carried over | Same first implementation (09-14). It uses the retired provenance_bias_coverage policy (Q and B mixed), which the PI's 09-20 specification replaced. |
| Retention component ablation (synthetic, 50% capacity, 5 seeds) | alpha 0 to 1, protection floor off and on, trust formulation arithmetic and geometric | Not carried over | Same first implementation. These are 09-09 numbers (commit 26d0e59) that the 09-14 report copied unchanged. The trust formulation and alpha belong to the retired geometric-trust design. |
| Classification reuse adapter | none, logit_residual, nominal_residual_scores on breast_cancer, synthetic, wine: retrieval and adapted accuracy, ECE, correct and harmful flips | Not carried over | Same first implementation; also 09-09 numbers copied into the 09-14 report. The aligned results nearest to it are Table 18 (nominal groups on synthetic nominal data only) and Table 40, Table 42, Table 43 (H2: nominal score against logit residual on nine real classification tasks, with ECE and accuracy on the validation stream and no flip counts). A real-data table of retrieval-only accuracy, adapted accuracy and correct and harmful flips for the three modes has not been produced. |
| Training schedule (two slides) | staged_sequential, alternating, joint_synchronized, warmup_alternating, warmup_joint on iris, wine, synthetic: final accuracy before and after adaptation, net flips, drift; the second slide gives final accuracy after adaptation per dataset | Not carried over | Early five-schedule trainer (09-15), a small run (2 seeds, 38 to 75 test points). On the slide the four schedules that train retrieval and adapter together (alternating, joint_synchronized, warmup_alternating, warmup_joint) end with accuracy before adaptation of 0.41 to 0.59 (mean over the three datasets; 0.33 to 0.70 per dataset), against 0.87 for staged_sequential, whose retriever is trained first and then frozen. Retrieval-only accuracy therefore fell during training in the first implementation; the cause was not tested. The aligned schedule results are in Table 32 to Table 34. |


All five groups of results (six tables on the slides) come from the first implementation, apart from the conclusions of the summary slide: two stand on aligned runs, and the third (E5) is not replicated by E17. The aligned results that replace the rest are in sections 5 and 7.

## 3. The model

### 3.1 Built before 10 Oct and carried over

**Table 3.** Model components of the aligned T1 pipeline, built 10-02 to 10-09 and carried over

| Component | What it does | Built | Results |
| --- | --- | --- | --- |
| Q and B scored separately | Provenance (Q) and bias (B) are compared separately; the geometric trust and the Q-B mixing are retired. Credit for a query uses its final adapted outcome. | 10-02 | Table 9, Table 10 |
| Coverage, reachability and removal influence | Explicit activation threshold with identity-based leave-one-out; removal influence recomputes the final prediction loss under a cumulative loss budget; a cached variant is checked against the full computation. | 10-02 | Table 11 |
| Stable case IDs | Case IDs stay stable through MCB and compaction. | 10-02 | - |
| Isolated removal retraining | Each candidate and an unmodified control fork from the same model, Adam state and RNG. Adam-moment aliasing between conditions fixed. | 10-02 | Table 7, Table 8 |
| Regression synchronization | Uses the maintained aggregate adapter (the historical pair network is frozen); a selected checkpoint restores the optimizer, free-radius state and Adam moments. | 10-04 | Table 13 |
| Final-capacity matching in synchronization | Independent and RR checkpoints are compressed after training with Q to the same final capacity as RRR (K=.75); the state before compression is saved and case-row Adam moments follow stable IDs. It matches final capacity only, not trajectory or compute. | 10-04 | Table 13 |
| No-op checkpoint-selection fix | A maintenance hook that changes nothing no longer resets the best checkpoint (first in synchronization, then in core training). | 10-04 | Table 12, Table 14 |
| Revision runner | Saves the real M0, M1 and M2 models; final provenance is recomputed with M2. | 10-04 | Table 15 |
| Reversible reviewer controls | Temporary forced retrieval; label, bias, blend and feature edits with versions and rollback; protect, quarantine, archive and restore; a matched no-edit continuation control (MC). | 10-04 | Table 16 |
| Independent synthetic generator | independent_v1 with 7 independent random streams; the legacy generator stays byte-identical. | 10-04 | Table 17 |
| Nominal reuse groups | Vocabularies fitted on training data only, bound to stable IDs; adapter input [Delta_z, Delta_u, p0]; undeclared or missing inputs raise an error instead of being filled with zeros. | 10-04 | Table 18 |
| Adapter optimizer state on continuation | The selected adapter keeps its Adam state when training continues. | 10-04 | Table 23 |
| Local reviewer workbench | Loopback-only workbench on a real model with a common-knowledge task and bounded M2 continuation. | 10-04 | Table 23 |
| Adapted removal | Removal with an external adapter (classification and regression), a capacity-matched protected random control, and reference routing (train-LOO or an independent 15% maintenance split). | 10-04 | Table 19, Table 20, Table 21, Table 22 |
| Calibration evidence and data path | Each calibration snapshot (at the selected core checkpoint and at every maintenance recalibration) saves the frozen biases, distance matrix, labels (cloned, which fixes a storage alias), pair mask, stable IDs and model state; synchronization logs per-epoch train and validation terms; preprocessing fits on training data only; OpenML and TabArena loaders keep the official outer split with a train-only inner holdout (not a TabArena leaderboard replication); a validation-only grid search over five families (NN-kNN, kNN, CatBoost, MLP, TabM) and a study executor. | 10-09 | Table 13, Table 23, Table 24 |
| Duplicate-case bias initialization | Exact duplicate cases now count as zero-distance non-self neighbours when the case biases are initialized. | 10-09 | Table 6 |


### 3.2 Changed since 10 Oct

**Table 4.** Model changes since 10 Oct

| Change | Default behaviour | Recorded check |
| --- | --- | --- |
| Joint retrieval-reuse training (schedules joint_rr and joint_rrr; separated or summed objective; adapter warm-up; training from scratch) | Existing schedules unchanged; joint objective separated and warm-up 0 unless set | 20 of 20 metrics identical to the earlier output (alignment_sync_plan_matched, seed 8); a test shows that with the separated objective the retrieval gradient equals dL_R/d(theta_R) and the adapter gradient equals dL_A/d(theta_A), to within 1e-6 (raw, MLP and EMA encoders) |
| EMA encoder ablation runner (arms raw, shared, EMA 0.9, 0.99, 0.999) | New experiment type | A test shows that a per-epoch observer hook leaves core training bit-identical, with and without the EMA encoder (the stability tracker itself is not in that test); results in Table 25 |
| Adapter weight decay and dropout | Off (0.0). Weight decay above 0 switches the adapter optimizer from Adam to AdamW; dropout applies to classification adapters only (error otherwise) | A test reproduces a stored digest of pre-change outputs with the defaults (default runs bit-identical) |
| Score scale kappa in the post-adaptation loss | 1. L_post = cross-entropy(kappa x scores); nominal_residual_scores only (error otherwise); also used in frozen-core reuse | Same test |
| Early stopping on validation error | Validation post-adaptation loss; the option val_error picks the lowest validation error, then the lowest loss | Same test |
| Retrieval guard with an epoch-0 fallback | Off. With guard g an epoch is eligible only while validation retrieval-only loss is at most (1+g) times its epoch-0 value; rejected epochs still count toward patience; if none is eligible the run returns its epoch-0 state, flagged sync_no_eligible_checkpoint; after a maintenance edit (RRR schedules) that state no longer exists and the run raises an error instead | Same test; the no-eligible-epoch case has its own tests |
| Validation-fitted temperature | Reported separately; argmax unchanged | A degenerate fit falls back to T = 1 and is flagged (test) |
| Input-space locality metric | Reported only; it is the locality measure of the endpoint 2 guard in Table 38 | Test: matches a manual computation for raw and MLP representations |
| Deterministic-algorithms switch | Off; enabled by NNKNN_DETERMINISTIC=1 | Test: the switch turns deterministic algorithms on and is off when unset |
| Skip the logging-only passes (diagnostics_every) | Every epoch. All step-D configs (H6, H1, H2, H2c, H3, H5) set it to 0, so their runs record no train-before/after-update terms | Skipping leaves model, adapter and validation curve bit-identical (new test). The authors' commit message puts the two logging passes at about 60% of synchronized-training time; no timing is stored |
| A dataset without its own max_train takes the study cap | The study's splits.max_train applies; no cap if the study sets none (earlier runs ignored the study cap) | Test: study cap 50, own cap 10, no cap at all give 50, 10, none |
| Chunked retrieval above a memory budget | One call while queries x cases x feature size is at most 2^27 elements; above that the per-feature contribution tensor is not returned | Raw-feature chunks equal the single call exactly (test); MLP-encoder chunks agree within 1e-6 absolute plus 1e-5 relative (test tolerance); the batch-64 Iris and balance runs reproduce bit-identically (56 runs) |
| Vectorized eval-mode feature-cache lookup | Replaces a per-case loop | Bitwise equality test against the old loop. The speed figures in the authors' commit message (513 ms against 2.5 ms per retrieval at 1,077 cases; about 2.4 times faster MLP runs) are not stored in any result file |
| One helper (nnknn_core_settings) defines the grid's NN-kNN settings, batch size 64, for the grid, the study executor and the schedule search | The running full study's plan stage was not changed and still uses batch 32 (Table 51, point 1) | digits seed 30 reproduces the grid (Table 30, Table 31) |


## 4. Which carried-over results can be trusted

**Table 5.** Trust status of every carried-over experiment (E1 to E16 from the 6 Oct report, E17 to E20 from 10-06 to 10-10)

| ID | Experiment | Status | Why | Re-run at the current code (audit) |
| --- | --- | --- | --- | --- |
| E1 | Initial alignment validation (10-02) | Engineering facts stand; accuracy numbers superseded | The post-selection Digits accuracies were produced before the Adam-moment fix and the best-epoch optimizer restore. The project says such comparisons need fresh runs, and E2 replaces them. The count of 61 replayed checkpoints has no stored replay log. | 6 of 8 Digits cells and most small-config cells differ (Digits random seed 0: 0.9833 to 0.9778) |
| E2 | New-seed continuation: retraining, small candidates, Digits (10-02) | Trusted with caveat | Aligned pipeline; all 108 CSV rows reproduce from per-run metrics. Exploratory 3-seed pilot that ran before the best-epoch optimizer restore. | Frozen rows identical. Small-candidate cells move by 2e-9 to 1.4e-4 (one Iris seed 4 cell by 2.2e-2); matched, retrained and Digits fine-tuned cells move by at most one test sample in accuracy, by 9e-6 to 4.5e-2 in loss (2e-6 or more in ECE) and by up to 4.3e-3 in Energy RMSE. On Digits seed 3 random rises to 0.9694 and B falls to 0.9667 (Q unchanged), so Q and B no longer beat random on seed 3. The Energy retrained-minus-frozen gap changes by as much as it measures |
| E3 | Supervised sync with regression (10-04) | Engineering checks stand; task metrics replaced by E4 and E5 | Ran before the sync checkpoint-selection fix; E4 repeats the same 36 runs (one run changes). The 6 Oct report carried only its checks. Its stored radius-calibration metadata (sum and sum of squares of distances) cannot rebuild the free-radius regions, so tau_task is not reproducible from its files. | Energy seed 5, Yacht seed 5, Iris seed 6 bit-identical; synthetic seed 7 RRR now equals E4; Iris seed 5 differs in tau_task, correction size and ECE (accuracy identical) |
| E4 | Sync selection fix, before and after (10-04) | Trusted with caveat | Bounded diagnostic repeat. One run changes (4 of 260 test queries). RRR keeps fewer cases than independent. Its stored radius-calibration metadata cannot be rebuilt (as E3). | Synthetic seed 7 and Energy seed 5 bit-identical; Iris seed 5 differs in tau_task, mean correction size and ECE (accuracy identical) |
| E5 | Capacity-matched sync (10-04) | Trusted with caveat | Descriptive 3-seed result. Final capacity is matched, not trajectory or compute. The RRR advantage on Energy is not replicated by E17 (Yacht was not repeated). Its radius-calibration metadata cannot be rebuilt (as E3). | Energy seed 5 bit-identical; Iris seed 5 differs in tau_task (2.8e-3), mean correction size (at most 1.6e-3) and ECE (at most 2.5e-4); accuracy and capacity identical |
| E6 | Core checkpoint-selection fix (10-04) | Engineering check; stands | Equivalence probe, not a generalization claim. | 16 of 18 cells identical to 1e-12; the two Iris seed 8 cells differ by 4.4e-8 in validation loss and 2.2e-8 in test loss |
| E7 | Revision flag precision (10-04) | Trusted with caveat | Descriptive; accuracy near ceiling; simulated oracle review; no matched continuation in this batch (added in E8). | Seed 5: all 8 conditions bit-identical |
| E8 | Reviewer controls, matched continuation (10-04) | Trusted with caveat | Flip counts are 0 to 2 per run; probes are simulated. | Seed 5: all 8 conditions bit-identical |
| E9 | Synthetic stress grid (10-04) | Trusted with caveat | Exploratory, 3 seeds, compute not matched across scenarios, simulated oracle. | Seed 8, reference and redundancy scenarios: bit-identical |
| E10 | Reviewer workbench pilot (10-04) | Engineering check; stands | Mechanism demonstration; no participants. The replay and audit counts have no stored log. | First-view output for seeds 11, 12, 13 byte-identical; corrective-chain outcomes match (8 of 8 against 6 of 8); the undo chain and the 99/792 audit were not re-run |
| E11 | Nominal reuse groups (10-04) | Trusted with caveat | Small synthetic contract exercise, 3 seeds; RRR capacity differs. | Seed 8: all conditions bit-identical |
| E12 | Selected-adapter optimizer states (10-04) | Engineering check; stands | Provenance and replay, not performance. | Seed 8: reuse and sync metrics bit-identical; the exported optimizer states were not re-checked |
| E13 | Adapted removal, Iris (10-04) | Trusted with caveat | 18-case pilot; the scopes received different budgets (alternating gets twice the phase epochs), a confound that E15 controls. | Iris seed 8: bit-identical |
| E14 | Adapted removal, regression (10-04) | Trusted with caveat | 2 tasks, 3 seeds; the random control violates the budget in 13 of 18 runs. | Energy and Yacht seed 8: bit-identical |
| E15 | Adapted budget and capacity grid (10-04) | Trusted with caveat | 108 correlated runs, not replicates; ties are 39, 52 and 33 of 108 against initial, matched full and matched random. | Iris seed 8: bit-identical |
| E16 | Paired maintenance reference guard (10-04) | Trusted with caveat | 2 tasks, 3 seeds; selecting on the maintenance split can overfit. | Energy seed 8: bit-identical |
| E17 | Bounded 120-run sync pilot after calibration repair (10-09) | Trusted with caveat | Exploratory, 3 seeds; capacity-matched, not compute-matched; the Iris gap is one test case (30 test cases per seed). | Not re-run (replay re-loads saved states). Iris seed 8 (20 of the 120 runs, and the one test case behind the Iris gap) has a duplicate pair, so those cells could change under the current initializer; seeds 9, 10 and Energy are unaffected |
| E18 | Subset replays (10-09) | Engineering check | Counts only; the metric counts are the replay scripts' own output. | - |
| E19 | Full-study protocol (10-09) | Protocol, not a result | The study is still running. | - |
| E20 | Two code fixes (10-09) | Code record | Neither note nor commit says which earlier results they touch; the audit measured it (Table 6). | - |


**Table 6.** Code fixes since 10-02 and what they changed in earlier results

| Fix | Date | What changed | Earlier experiments touched | Measured effect (two independent audits, re-runs at the current code) |
| --- | --- | --- | --- | --- |
| Adam-moment aliasing | 10-02 | Optimizer moments are copied per candidate instead of shared between conditions | E1 ran before it | E1 post-selection numbers do not reproduce |
| Best-epoch optimizer restore | 10-04 | Phases after core training continue from the Adam state of the selected epoch, not the last epoch | E1 and E2 ran before it | E1 superseded. E2: see Table 5 (accuracy moves by at most one test sample; loss by up to 4.5e-2; Digits seed 3 order of Q and B against random changes) |
| No-op checkpoint-selection fixes (sync, then core) | 10-04 | A maintenance hook that changes nothing keeps the best checkpoint | E3 ran before the sync fix; E6 is the core probe | One E3 run changes (synthetic seed 7 RRR, accuracy 0.9192 to 0.9346); Table 12 and Table 14 |
| Duplicate-case bias initialization | 10-09 | Exact duplicate cases count as zero-distance non-self neighbours | Only Iris subsets that contain the one duplicated pair: seed 4 (E2), seed 5 (E3 to E5), seed 8 (E6, and the Iris cells of E17: 20 of 120 runs, not re-run) | Initial bias 0.791236 to 0.788991 (seed 4), 0.856498 to 0.855025 (seed 5), 0.83496 to 0.83230 (seed 8); tau_task -1.6% (up to 2.8e-3 absolute); ECE changes by at most 3e-4 and correction size by at most 1.6e-3 in E3 to E5, and E6 validation loss by 4.4e-8; E2 Iris seed 4 random K=.75 moves by -2.2e-2 in test loss and +6.0e-3 in ECE (that cell is also affected by the best-epoch restore; the two effects were not separated); accuracies unchanged. No effect on synthetic, Energy, Yacht, Wine, Digits, nominal or workbench data (no exact duplicates) |
| Calibration evidence and preprocessing alignment | 10-09 | See Table 3 | None numerically | Re-runs of E4, E5, E7, E8, E9, E11 and E13 to E16 cells match (the only differences are the duplicate-case effects above); datasets compare bit-identical. E3 to E5 stored only a sum and sum of squares of distances for the radius calibration, so their tau_task cannot be rebuilt from the saved files; their task metrics are unaffected. The first two 60-run batches of 10-09 were invalid (label storage alias) and were rerun as the repaired 120-run pilot (E17) |


Reading: E1 and E3 keep only their engineering facts (Table 23). E2 and E5 are the two whose reading needs care: E2 because its matched, retrained and Digits fine-tuned cells are not bit-reproducible at the current code, E5 because its Energy ordering reverses in E17 (Table 13).

## 5. Carried-over results (aligned pipeline, 10-02 to 10-04)

### 5.1 Retraining, small candidates and Digits (E2, seeds 2, 3, 4)

**Table 7.** E2: retraining on Iris, test accuracy (seeds 2, 3, 4)

| Condition | Active cases | Extra epochs | Seed 2 | Seed 3 | Seed 4 | Mean |
| --- | --- | --- | --- | --- | --- | --- |
| full_frozen | 24 | 0 | 1.0000 | 0.9000 | 1.0000 | 0.9667 |
| full_matched | 24 | 12 | 1.0000 | 0.9000 | 1.0000 | 0.9667 |
| random_frozen | 18 | 0 | 0.9000 | 0.9000 | 0.9333 | 0.9111 |
| random_matched | 18 | 12 | 0.9000 | 0.9000 | 0.9333 | 0.9111 |
| removal_frozen | 18 | 0 | 1.0000 | 0.8667 | 1.0000 | 0.9556 |
| removal_retrained | 18 | 12 | 1.0000 | 0.8667 | 1.0000 | 0.9556 |


**Table 8.** E2: retraining on Energy, test RMSE on the standardized target (seeds 2, 3, 4)

| Condition | Active cases | Extra epochs | Seed 2 | Seed 3 | Seed 4 | Mean |
| --- | --- | --- | --- | --- | --- | --- |
| full_frozen | 24 | 0 | 0.3341 | 0.3062 | 0.4023 | 0.3475 |
| full_matched | 24 | 12 | 0.3375 | 0.3070 | 0.4028 | 0.3491 |
| random_frozen | 18 | 0 | 0.3331 | 0.3898 | 0.4199 | 0.3809 |
| random_matched | 18 | 12 | 0.3297 | 0.3785 | 0.4160 | 0.3747 |
| removal_frozen | 18 | 0 | 0.3268 | 0.3205 | 0.4184 | 0.3552 |
| removal_retrained | 18 | 12 | 0.3267 | 0.3284 | 0.4182 | 0.3578 |


**Table 9.** E2: small candidates on Energy, test RMSE, K=0.75 (seeds 2, 3, 4)

| Condition | Active cases | Seed 2 | Seed 3 | Seed 4 | Mean |
| --- | --- | --- | --- | --- | --- |
| full_memory | 60 | 0.3734 | 0.3461 | 0.3184 | 0.3460 |
| random | 45 | 0.3643 | 0.3988 | 0.3260 | 0.3630 |
| Q | 45 | 0.3960 | 0.3935 | 0.3738 | 0.3877 |
| B | 45 | 0.3565 | 0.3534 | 0.3367 | 0.3488 |
| coverage_reachability | 45 | 0.4010 | 0.3404 | 0.3750 | 0.3721 |
| removal_influence | 45 | 0.3687 | 0.3661 | 0.3333 | 0.3560 |
| removal_cached | 45 | 0.3687 | 0.3661 | 0.3333 | 0.3560 |
| Iris accuracy, every small and adapted condition | - | 1.0000 | 0.9333 | 0.9333 | 0.9556 |


**Table 10.** E2: Digits, test accuracy, K=0.5, 30 fine-tuning epochs (seeds 2, 3, 4)

| Condition | Active cases | Seed 2 | Seed 3 | Seed 4 | Mean |
| --- | --- | --- | --- | --- | --- |
| full_memory | 1077 | 0.9806 | 0.9694 | 0.9833 | 0.9778 |
| random | 538 | 0.9583 | 0.9667 | 0.9861 | 0.9704 |
| Q | 538 | 0.9750 | 0.9694 | 0.9778 | 0.9741 |
| B | 538 | 0.9722 | 0.9694 | 0.9806 | 0.9741 |


**Table 11.** E2: diagnostics

| Quantity | Value |
| --- | --- |
| Spearman correlation of Q and B, small regression | 0.4133 |
| Spearman correlation of Q and B, small Iris | 0.4850 |
| Spearman correlation of Q and B, adapted Iris | 0.5270 |
| Spearman correlation of Q and B, Digits | 0.5454 |
| Digits evidence fraction | 0.6493 |
| Max error, cached vs full removal influence | 0.001510 |
| Retraining: candidate epochs per dataset and seed | 258 |
| Retraining: accepted epochs per run | 12 |
| Energy RMSE, retrained minus frozen removal (mean of 3 seeds) | 0.002545 |


Reading (Table 7 to Table 11):
- On Iris, removal with retraining (0.9556) equals frozen removal (0.9556) and is below full memory (0.9667). Random is lowest (0.9111).
- On Energy, retrained removal is worse than frozen removal in RMSE by 0.002545 (0.3578 against 0.3552 in Table 8), and its search runs 258 candidate epochs per dataset and seed (12 accepted; Table 11).
- Among the small candidates on Energy, B (0.3488) is closest to full memory (0.3460) and Q (0.3877) is worst. On Iris every condition gives 0.9556, so Iris cannot tell them apart.
- On Digits, Q and B (0.9741 each) are above random (0.9704) and below full memory (0.9778). Q and B beat random on seeds 2 and 3 and lose on seed 4. No seed is near chance (lowest 0.9583).
- Q and B are only moderately correlated (Spearman 0.4133 to 0.5454), and the cached and full removal influences differ by at most 0.001510.
- Status (Table 5): the frozen rows reproduce at the current code, but the matched, retrained and Digits fine-tuned cells move by at most one test sample, and on Digits seed 3 the order of Q and B against random changes. The reading is therefore qualitative: retraining gives no accuracy gain at higher cost, no Digits collapse, Q or B against random is mixed.

### 5.2 Synchronization (E4, E5, E6)

**Table 12.** E4: synchronization checkpoint-selection fix, before and after (36 runs repeated)

| Quantity | Before the fix | After the fix |
| --- | --- | --- |
| synthetic_diag seed 7, RRR, post-adaptation accuracy | 0.919231 | 0.934615 |
| Energy, RRR minus independent RMSE, seed 5 | 0.014386 | 0.014386 |
| Energy, RRR minus independent RMSE, seed 6 | 0.006908 | 0.006908 |
| Energy, RRR minus independent RMSE, seed 7 | 0.007966 | 0.007966 |


**Table 13.** E5 and E17: capacity-matched synchronization, post-adaptation means over three seeds

| Task | Experiment | Independent | RR | RRR |
| --- | --- | --- | --- | --- |
| synthetic_diag accuracy | E5: K=.75, seeds 5/6/7 | 0.9564 | 0.9641 | 0.9679 |
| Iris accuracy | E5: K=.75, seeds 5/6/7 | 0.9667 | 0.9667 | 0.9556 |
| Energy RMSE (standardized) | E5: K=.75, seeds 5/6/7 | 0.3622 | 0.3722 | 0.3493 |
| Yacht RMSE (standardized) | E5: K=.75, seeds 5/6/7 | 0.6403 | 0.6368 | 0.5843 |
| Energy, one development seed, 45 cases (provisional) | E5 side result: one development seed | 0.380 | 0.411 | 0.354 |
| Iris accuracy | E17: K=45, seeds 8/9/10 | 0.9444 | 0.9556 | 0.9556 |
| Energy RMSE (standardized) | E17: K=45, seeds 8/9/10 | 0.2844 | 0.2836 | 0.2882 |


**Table 14.** E6: core checkpoint-selection fix (18 runs before, 18 after)

| Run | Validation loss (fixed / old hook) | Test metric (fixed / old hook) | Selected epoch (fixed / old hook) |
| --- | --- | --- | --- |
| Iris seed 8 (accuracy) | 0.054304 / 0.061560 | 0.933333 / 0.966667 | 9 / 16 |
| Yacht seed 10 (RMSE) | 0.356759 / 0.397327 | 0.504266 / 0.492752 | 9 / 16 |
| Repaired no-op vs ordinary pairs, bitwise identical | 9 of 9 |  |  |


Reading:
- Table 12: the fix changes one run (synthetic_diag seed 7, RRR, 0.9192 to 0.9346). RRR minus independent on Energy is unchanged (+0.014386, +0.006908, +0.007966), but RRR keeps fewer cases, so this gap is not a clean schedule effect.
- Table 13: with final capacity matched (E5), RRR is best on synthetic_diag (0.9679), Energy (0.3493) and Yacht (0.5843) and worst on Iris (0.9556). On new seeds with K=45 (E17), RRR is worst on Energy (0.2882 against 0.2844 for independent) and ties RR on Iris, both above independent (0.9556 against 0.9444; the Iris difference is one test case). The Energy ordering reverses between E5 and E17, so the RRR advantage is not established.
- Table 14: under the old hook two of the 18 runs select epoch 16 instead of 9 and show better test scores but worse validation loss. All 9 repaired no-op runs equal their ordinary pairs.

### 5.3 Revision and reviewer controls (E7, E8, E9)

**Table 15.** E7: precision of the revision flaggers (synthetic_diag, seeds 5, 6, 7; 54 corrupted rows)

| Flagger | Precision, budget 10 | Precision, budget 30 | Recall, budget 10 | Recall, budget 30 |
| --- | --- | --- | --- | --- |
| Q | 0.3000 | 0.2667 | - | - |
| B | 0.0667 | 0.0556 | - | - |
| Random | 0.1000 | 0.1222 | - | - |
| Oracle | 1.0000 | 1.0000 | 0.1852 (10/54) | 0.5556 (30/54) |


**Table 16.** E8: flips from the matched no-edit continuation (MC) to the corrected model (M2)

| Case | Beneficial flips (MC to M2) | Harmful flips |
| --- | --- | --- |
| Seed 5, Q | 1 | 1 |
| Seed 7, Q | 1 | 0 |
| Seed 7, random, budget 30 | 2 | 0 |


**Table 17.** E9: synthetic stress grid, mean M2 minus MC accuracy at budget 30 (seeds 8, 9, 10)

| Scenario | Q | B | Random | Oracle |
| --- | --- | --- | --- | --- |
| reference | +0.0013 | 0.0000 | 0.0000 | +0.0103 |
| overlap | -0.0038 | +0.0051 | 0.0000 | +0.0051 |
| noise | +0.0282 | +0.0103 | +0.0141 | +0.0385 |
| redundancy | +0.0026 | +0.0026 | +0.0038 | +0.0154 |
| uncovered rare | 0.0000 | 0.0000 | +0.0013 | +0.0064 |
| shift | +0.0090 | 0.0000 | 0.0000 | +0.0090 |
| combined | +0.0256 | -0.0026 | +0.0128 | +0.0256 |
| Flag precision, overlap (Q / random) | 0.20 / 0.10 |  |  |  |
| Flag precision, noise (Q) | 0.4778 |  |  |  |
| Runs / stage checkpoints | 168 / 672 |  |  |  |


Reading:
- Table 15: Q flags corrupted rows more precisely than random (0.3000 and 0.2667 against 0.1000 and 0.1222) and B least (0.0667 and 0.0556). The oracle reaches precision 1 but recalls only 0.1852 of the 54 corrupted rows at budget 10 and 0.5556 at budget 30.
- Table 16: against a matched no-edit continuation the corrected model gains 1 to 2 queries and loses 0 to 1 in the three listed cases, and all 30 trained probe scenarios pass (Table 23).
- Table 17: for Q, random and the oracle the M2 minus MC accuracy gain is largest in the noise scenario (Q +0.0282) and the combined scenario (Q +0.0256); for B the combined scenario is slightly negative (-0.0026). In the overlap scenario Q has flag precision 0.20 against 0.10 for random, but its accuracy change is negative (-0.0038).

### 5.4 Nominal reuse groups (E11)

**Table 18.** E11: nominal reuse groups on synthetic_nominal/v1 (seeds 8, 9, 10; 45 runs)

| Setting | Condition | Accuracy |
| --- | --- | --- |
| Reuse | retrieval only | 0.3481 |
| Reuse | nominal residual, combined loss, overall | 0.7222 |
| Reuse | nominal residual, combined loss, known colors | 0.9365 |
| Reuse | nominal residual, combined loss, unseen color | 0.1667 |
| Reuse | logit residual, combined loss, overall | 0.6407 |
| Reuse | logit residual, combined loss, unseen color | 0.2963 |
| Retention (after adaptation) | full memory, 180 cases | 0.7222 |
| Retention (after adaptation) | Q, coverage, full or cached removal, 135 cases | 0.7148 |
| Retention (after adaptation) | B, 135 cases | 0.7074 |
| Synchronization | independent | 0.6259 |
| Synchronization | RR | 0.6296 |
| Synchronization | RRR (135 cases) | 0.6333 |


Reading (Table 18): the nominal residual adapter raises accuracy from 0.3481 to 0.7222, but unseen colors stay at 0.1667 and the logit residual reaches 0.2963 there. Retention to 135 cases gives 0.7148 (Q, coverage, removal) and 0.7074 (B) against 0.7222 for full memory. The three synchronization schedules score 0.6259, 0.6296 and 0.6333 at unequal capacity.

### 5.5 Adapted removal (E13 to E16)

**Table 19.** E13: adapted removal on Iris, 18 to 12 cases, budget .03 (three-seed means)

| Scope | Final | Initial | Matched full | Matched random |
| --- | --- | --- | --- | --- |
| retrieval only | 0.9111 | 0.9222 | 0.9111 | 0.8889 |
| adapter only | 0.9111 | 0.9222 | 0.9222 | 0.8889 |
| alternating | 0.9111 | 0.9222 | 0.9111 | 0.8889 |


**Table 20.** E14: adapted removal on regression (Energy and Yacht, seeds 8, 9, 10)

| Comparison (Energy and Yacht, 18 runs) | Runs |
| --- | --- |
| Adapted test RMSE worse than initial and than matched full (same 12 runs) | 12 of 18 |
| Adapted test RMSE worse than matched random | 13 of 18 |
| Random runs outside the .03 loss budget | 13 of 18 |


**Table 21.** E15: adapted budget and capacity grid (Iris and Wine, 108 runs)

| Adapted test accuracy compared with | Improved | Worse | Tied |
| --- | --- | --- | --- |
| initial state | 48 | 21 | 39 |
| matched full control | 35 | 21 | 52 |
| matched random control | 45 | 30 | 33 |
| Random runs outside the budget (of 108) | 40 |  |  |


**Table 22.** E16: maintenance split against train-LOO as the selection reference (Energy and Yacht, 36 runs)

| Quantity | Adapter | Alternating | Retrieval |
| --- | --- | --- | --- |
| Energy: mean test RMSE improvement from the maintenance split | 0.04209 | 0.03739 | 0.04403 |
| Yacht: mean test RMSE improvement from the maintenance split | 0.03228 | 0.03444 | 0.02737 |
| Pairs improved by the maintenance split | 18 of 18 |  |  |
| Adapted vs initial and full: better / worse runs (of 36) | 24 / 12 |  |  |
| Adapted vs random: better / worse runs (of 36) | 26 / 10 |  |  |
| Random runs outside the budget (of 36) | 27 |  |  |


Reading:
- Table 19: on Iris the adapted result (0.9111) is below the initial state (0.9222) in every scope, and above matched random (0.8889).
- Table 20: on regression 12 of 18 adapted runs are worse than the initial state and than the matched full control, and 13 are worse than matched random, which itself breaks the loss budget in 13 runs.
- Table 21: adapted removal improves on the initial state in 48 of 108 runs and worsens in 21, with 39 ties. Random breaks the budget in 40 runs. The 108 runs are correlated conditions.
- Table 22: choosing by the independent maintenance split instead of train-LOO improves the test RMSE in all 18 pairs. Over both reference arms (36 runs), adapted runs are worse than the initial state or full control in 12 runs, and repeated selection on that split can overfit.

### 5.6 Engineering and replay checks that stand

**Table 23.** Engineering and replay checks that stand

| Experiment | Check | Value |
| --- | --- | --- |
| E1 | Manifests in the 9 validation groups | 70 |
| E1 | Maintenance checkpoints replayed within 1e-6 | 61 |
| E1 | Max cached vs full score deviation | 0.00075659 |
| E3 | Runs (synthetic_diag, Iris, Energy, Yacht; seeds 5/6/7) | 36 |
| E3 | Checkpoint replays at tolerance 1e-6: all passed | 36 / 36 |
| E3 | C+H=A max deviation (declared check 1e-5) | 2.801e-06 |
| E10 | Versioned checkpoints / MC controls replayed | 75 / 9 |
| E10 | Independent audit: checkpoints / query events | 99 / 792 |
| E10 | Correction chain vs M0 and MC successes (of 8 queries) | 8 vs 6 |
| E10 | Undo chain successes (of 8) | 6 |
| E10 | Chain C cases, M2 vs MC | 7 vs 8 |
| E10 | T1 tests passed (including 6 workbench tests) | 117 |
| E8 | Trained intervention probe scenarios passed | 30 / 30 |
| E8 | Regression archive/restore, max prediction difference | 2.384e-07 |
| E12 | Runs: reuse + sync | 18 + 9 = 27 |
| E12 | Optimizer-state rebuild events | 2430 |
| E17 | Manifests matched | 120 / 120 |
| E17 | Calibration snapshots / epoch records / metrics replayed (tolerance 1e-6) | 144 / 2400 / 1200 |
| E18 | Iris/synthetic subset: runs (sync + reuse), jobs | 160 (100 + 60), 20 |
| E18 | Iris/synthetic subset: sync snapshots / epoch records / sync metrics / reuse metrics replayed | 120 / 6818 / 700 / 420 |
| E18 | Radius diagnostics: runs / snapshots / epoch records / scalar metrics | 50 / 100 / 3817 / 350 |


## 6. 10-06 to 10-10: calibration repair and the full-study design

The bounded 120-run synchronization pilot after the calibration repair (E17) is in Table 13 (means) and Table 23 (replay counts: 120 of 120 manifests, 144 snapshots, 2,400 epoch records and 1,200 metrics within 1e-6). The first two 60-run batches of 10-09 were invalid because of a label storage alias and are not used (Table 6). The design of the full-study grid that started on 10-09 is in Table 24; its progress is in section 7.5.

**Table 24.** E19: design of the 2026-10-09 full-study grid

| Item | Value |
| --- | --- |
| Tasks | 22 |
| Classification (maintained 7 + 2 TabArena) | 9 |
| Regression (maintained 11 + 2 TabArena) | 13 |
| TabArena tasks (included above) | 4 |
| Families (NN-kNN, kNN, CatBoost, MLP, TabM) | 5 |
| Candidates per family | 32 |
| Development seeds / evaluation seeds | 20,21 / 30,31,32,33,34 |
| Screening epochs / patience | 60 / 15 |
| Refit epochs / patience (3 finalists) | 300 / 40 |
| Jobs (task x family) | 110 |
| Screening fits / total fits when all candidates complete | 7040 / 8250 |
| Test-based selection | no |


## 7. New since 10 Oct

### 7.1 The EMA (momentum) encoder

**Table 25.** EMA ablation at default settings: median relative change in test error, exact Wilcoxon p over 13 tasks, tasks better / worse (negative change is lower error)

| Setting | Arm | vs shared: median change | p | better / worse | vs raw: median change | p | better / worse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Retrieval only | raw | -29.0% | 0.013 | 10 / 3 | - | - | - |
| Retrieval only | shared | - | - | - | +40.9% | 0.006 | 3 / 10 |
| Retrieval only | ema090 | -6.5% | 0.787 | 8 / 5 | +40.0% | 0.017 | 3 / 10 |
| Retrieval only | ema099 | -0.3% | 0.893 | 7 / 6 | +35.1% | 0.017 | 3 / 10 |
| Retrieval only | ema0999 | +0.1% | 0.735 | 6 / 7 | +40.0% | 0.010 | 3 / 10 |
| Retrieval only | ema_valsel | -6.5% | 0.893 | 7 / 6 | +55.8% | 0.008 | 3 / 10 |
| Frozen-core reuse | raw | -31.7% | 0.003 | 11 / 2 | - | - | - |
| Frozen-core reuse | shared | - | - | - | +46.4% | 0.002 | 2 / 11 |
| Frozen-core reuse | ema090 | -8.1% | 0.455 | 8 / 5 | +25.3% | 0.013 | 3 / 10 |
| Frozen-core reuse | ema099 | -1.1% | 0.946 | 7 / 6 | +32.4% | 0.008 | 3 / 10 |
| Frozen-core reuse | ema0999 | -0.5% | 0.839 | 7 / 6 | +42.1% | 0.003 | 2 / 11 |
| Frozen-core reuse | ema_valsel | -8.1% | 0.414 | 8 / 5 | +42.1% | 0.005 | 2 / 11 |
| Alternating | raw | -40.6% | 0.002 | 11 / 2 | - | - | - |
| Alternating | shared | - | - | - | +68.4% | 0.002 | 2 / 11 |
| Alternating | ema090 | -9.1% | 0.216 | 8 / 5 | +25.7% | 0.010 | 2 / 11 |
| Alternating | ema099 | -0.6% | 0.635 | 7 / 6 | +50.0% | 0.007 | 2 / 10 |
| Alternating | ema0999 | +0.2% | 0.946 | 6 / 7 | +48.6% | 0.003 | 2 / 11 |
| Alternating | ema_valsel | -3.9% | 0.191 | 8 / 5 | +47.1% | 0.003 | 2 / 11 |
| Joint | raw | -26.7% | 0.003 | 10 / 3 | - | - | - |
| Joint | shared | - | - | - | +36.3% | 0.003 | 3 / 10 |
| Joint | ema090 | -2.8% | 0.735 | 7 / 6 | +24.4% | 0.017 | 2 / 11 |
| Joint | ema099 | +0.4% | 0.839 | 6 / 7 | +29.8% | 0.007 | 1 / 10 |
| Joint | ema0999 | +0.9% | 0.376 | 6 / 7 | +30.1% | 0.008 | 2 / 11 |
| Joint | ema_valsel | -2.8% | 0.839 | 7 / 6 | +29.8% | 0.010 | 2 / 11 |


**Table 26.** EMA ablation: number of tasks on which each arm has the lowest test error (ties count for each tied arm)

| Setting | raw | shared | ema090 | ema099 | ema0999 | Tasks |
| --- | --- | --- | --- | --- | --- | --- |
| Retrieval only | 8 | 1 | 1 | 3 | 0 | 13 |
| Frozen-core reuse | 9 | 1 | 1 | 2 | 0 | 13 |
| Alternating | 10 | 1 | 1 | 1 | 1 | 13 |
| Joint | 9 | 2 | 2 | 2 | 0 | 13 |


**Table 27.** EMA ablation: stability of the stored-case embeddings (retrieval-only core)

| Arm | Median drift per epoch | Median neighbour churn | Tasks with lower drift than shared | Tasks with lower churn than shared | abalone drift | abalone churn |
| --- | --- | --- | --- | --- | --- | --- |
| shared | 0.0495 | 0.1638 | - | - | 0.1869 | 0.3486 |
| ema090 | 0.0253 | 0.1451 | 13 / 13 | 12 / 13 | 0.0784 | 0.2942 |
| ema099 | 0.0087 | 0.1705 | 13 / 13 | 8 / 13 | 0.0306 | 0.2912 |
| ema0999 | 0.0026 | 0.1736 | 13 / 13 | 5 / 13 | 0.0085 | 0.2770 |


**Table 28.** EMA check on the six tasks whose tuned core uses the MLP encoder (batch-64 cores, seeds 30 to 34)

| Setting | Arm | Median change vs shared | p | better / worse / tied (of 6) |
| --- | --- | --- | --- | --- |
| Retrieval only | ema090 | +17.0% | 0.219 | 2 / 4 / 0 |
| Retrieval only | ema099 | +0.4% | 0.688 | 3 / 3 / 0 |
| Frozen-core reuse | ema090 | +14.3% | 0.312 | 2 / 4 / 0 |
| Frozen-core reuse | ema099 | +0.6% | 1.000 | 3 / 3 / 0 |
| Alternating default | ema090 | +2.6% | 0.844 | 3 / 3 / 0 |
| Alternating default | ema099 | +0.9% | 0.688 | 3 / 3 / 0 |
| Joint (summed, warm-up 5) | ema090 | +12.2% | 0.156 | 1 / 5 / 0 |
| Joint (summed, warm-up 5) | ema099 | +13.1% | 0.156 | 1 / 5 / 0 |


**Table 29.** EMA check: test error per task, retrieval only (batch 64)

| Task (retrieval only) | shared | ema090 | ema099 | ema090 vs shared | ema099 vs shared |
| --- | --- | --- | --- | --- | --- |
| abalone | 0.7069 | 0.7020 | 0.6991 | -0.7% | -1.1% |
| airfoil | 0.2351 | 0.3012 | 0.3037 | +28.1% | +29.2% |
| balance | 0.0624 | 0.0480 | 0.0528 | -23.1% | -15.4% |
| tabarena_363614 | 0.0154 | 0.0361 | 0.0321 | +134.7% | +108.6% |
| tabarena_363625 | 0.3771 | 0.3994 | 0.3841 | +5.9% | +1.9% |
| wine | 0.0556 | 0.0722 | 0.0444 | +30.0% | -20.0% |


Reading:
- Table 25: the raw-feature arm has lower error than the shared MLP encoder in 10, 11, 11 and 10 of 13 tasks (median -29.0%, -31.7%, -40.6%, -26.7%; p up to 0.013). No EMA arm differs from shared with p below 0.191.
- Against raw, every fixed-momentum EMA arm is worse by +24.4% to +50.0% (p up to 0.017), and the validation-selected momentum by up to +55.8%.
- Table 26: raw has the lowest error on 8, 9, 10 and 9 of 13 tasks.
- Table 27: EMA lowers embedding drift per epoch from 0.0495 (shared) to 0.0253, 0.0087 and 0.0026 for momentum 0.9, 0.99 and 0.999, on 13 of 13 tasks each. Neighbour churn does not fall consistently: the median is 0.1451, 0.1705 and 0.1736 against 0.1638 for shared, and it is lower than shared on 12, 8 and 5 of 13 tasks.
- Table 28: on the six tuned-MLP tasks no EMA arm has a lower median error than shared in any setting (median +0.4% to +17.0%, smallest p 0.156). In Table 29 EMA wins on balance (0.0480 and 0.0528 against 0.0624) and loses on airfoil (0.3012 and 0.3037 against 0.2351).
- Conclusion: EMA is not useful as a default on these supervised tabular tasks. It stays a candidate where an encoder is required or the case base changes during training; neither was tested.

### 7.2 Training-schedule search and the batch-size mismatch

The search starts each task from the validation-selected NN-kNN core of the full-study grid (16 tasks) and ranks 73 candidates on validation only (development seeds 20 and 21). Test streams are used only for confirmation on seeds 30 to 34. At batch 64, stage B uses the same grid as at batch 32 (see the configuration files), so it explores the neighbourhood of the batch-32 leaders.

**The mismatch.** The grid trains NN-kNN at batch size 64, but the study executor and the first schedule search rebuilt the selected cores without a batch size and inherited 32.

**Table 30.** digits, retrieval-only core per seed at batch 32 and batch 64

| Seed | Batch 32: validation accuracy | test accuracy | best epoch | Batch 64: validation accuracy | test accuracy | best epoch |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | 0.9944 | 0.9861 | 19 | 0.9972 | 0.9889 | 15 |
| 21 | 0.1167 | 0.1083 | 1 | 0.9889 | 0.9806 | 154 |
| 30 | 0.1194 | 0.1028 | 1 | 0.9556 | 0.9611 | 1 |
| 31 | 0.1111 | 0.1083 | 1 | 0.9806 | 0.9889 | 53 |
| 32 | 0.9750 | 0.9750 | 40 | 0.9750 | 0.9722 | 3 |
| 33 | 0.9861 | 0.9778 | 132 | 0.9833 | 0.9833 | 43 |
| 34 | 0.1139 | 0.1167 | 1 | 0.9806 | 0.9806 | 7 |


**Table 31.** Do the confirmation cores reproduce the HPO grid's own evaluation of the same task and seed?

| Retrieval-only cores of the confirmation vs the grid's own evaluation | Pairs | Reproduced (batch 32) | Reproduced (batch 64) |
| --- | --- | --- | --- |
| classification | 45 | 8 | 45 |
| regression | 35 | 0 | 30 |
| all | 80 | 8 | 75 |


Reading: at batch 32 the digits retrieval-only core collapses to chance in epoch 1 on seeds 21, 30, 31 and 34 (validation accuracy 0.1111 to 0.1194); at batch 64 all seven seeds are 0.9556 to 0.9972 (Table 30). Only 8 of 80 batch-32 cores reproduce the grid's own evaluation, none of the 35 regression ones, against 75 of 80 at batch 64 (Table 31).

**Table 32.** Selected joint schedule and per-task selection against four references, test error on seeds 30 to 34

| Cores | Contender | Compared with | Better / worse / tied (of 16) | Median change | Wilcoxon p (uncorrected) |
| --- | --- | --- | --- | --- | --- |
| batch 32 (mismatched) | validation-selected global schedule | alternating default | 13 / 3 / 0 | -11.6% | 0.008 |
| batch 32 (mismatched) | validation-selected global schedule | frozen-core reuse | 12 / 4 / 0 | -4.9% | 0.005 |
| batch 32 (mismatched) | validation-selected global schedule | sequential, scale 0.1 | 14 / 2 / 0 | -8.2% | <0.001 |
| batch 32 (mismatched) | validation-selected global schedule | retrieval only | 13 / 3 / 0 | -9.4% | 0.001 |
| batch 32 (mismatched) | per-task selection | alternating default | 14 / 2 / 0 | -11.0% | 0.004 |
| batch 32 (mismatched) | per-task selection | frozen-core reuse | 15 / 1 / 0 | -7.2% | 0.002 |
| batch 32 (mismatched) | per-task selection | sequential, scale 0.1 | 15 / 1 / 0 | -8.9% | 0.001 |
| batch 32 (mismatched) | per-task selection | retrieval only | 14 / 2 / 0 | -9.2% | 0.009 |
| batch 64 (correct) | validation-selected global schedule | alternating default | 11 / 5 / 0 | -1.8% | 0.323 |
| batch 64 (correct) | validation-selected global schedule | frozen-core reuse | 12 / 4 / 0 | -1.3% | 0.144 |
| batch 64 (correct) | validation-selected global schedule | sequential, scale 0.1 | 11 / 5 / 0 | -1.7% | 0.375 |
| batch 64 (correct) | validation-selected global schedule | retrieval only | 12 / 4 / 0 | -2.7% | 0.144 |
| batch 64 (correct) | per-task selection | alternating default | 10 / 6 / 0 | -0.9% | 0.404 |
| batch 64 (correct) | per-task selection | frozen-core reuse | 10 / 5 / 1 | -1.7% | 0.427 |
| batch 64 (correct) | per-task selection | sequential, scale 0.1 | 9 / 6 / 1 | -0.8% | 0.532 |
| batch 64 (correct) | per-task selection | retrieval only | 11 / 4 / 1 | -2.0% | 0.125 |


**Table 33.** Selected joint schedule against the alternating default, per task (test error, mean of 5 seeds)

| Task | Batch 32: default | selected | change | Batch 64: default | selected | change |
| --- | --- | --- | --- | --- | --- | --- |
| abalone | 0.6981 | 0.7027 | +0.7% | 0.6976 | 0.6970 | -0.1% |
| airfoil | 0.2490 | 0.2343 | -5.9% | 0.2370 | 0.2333 | -1.6% |
| balance | 0.0640 | 0.0384 | -40.0% | 0.0656 | 0.0416 | -36.6% |
| breast_cancer | 0.0421 | 0.0316 | -25.0% | 0.0333 | 0.0439 | +31.6% |
| digits | 0.0272 | 0.0233 | -14.3% | 0.0211 | 0.0217 | +2.6% |
| energy_efficiency | 0.0474 | 0.0498 | +5.2% | 0.0487 | 0.0510 | +4.8% |
| iris | 0.0467 | 0.0600 | +28.6% | 0.0467 | 0.0800 | +71.4% |
| student_performance | 0.3995 | 0.3978 | -0.4% | 0.4013 | 0.3934 | -2.0% |
| tabarena_363614 | 0.0220 | 0.0154 | -30.3% | 0.0154 | 0.0120 | -21.8% |
| tabarena_363625 | 0.3822 | 0.3485 | -8.8% | 0.3782 | 0.3707 | -2.0% |
| tabarena_363626 | 0.2741 | 0.2687 | -2.0% | 0.2717 | 0.2682 | -1.3% |
| tabarena_363675 | 0.3998 | 0.3904 | -2.3% | 0.4044 | 0.3913 | -3.2% |
| wine | 0.0611 | 0.0167 | -72.7% | 0.0611 | 0.0167 | -72.7% |
| yacht | 0.1149 | 0.0808 | -29.7% | 0.1197 | 0.0804 | -32.9% |
| zebra | 0.0545 | 0.0364 | -33.3% | 0.0727 | 0.0182 | -75.0% |
| zebra_special | 0.2864 | 0.2364 | -17.5% | 0.2773 | 0.3091 | +11.5% |


**Table 34.** Batch-64 confirmation: all 20 conditions against the alternating default (16 tasks, seeds 30 to 34)

| Rank | Condition | Mean test rank | Better / worse / tied | Median change | Median pre-adaptation change | p (uncorrected) | p x 19 (Bonferroni, max 1) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | joint separated, warm-up 5, scale 1.0, w_post 0.5, w_near 0.1, v_small 0.0, lr 0.003 | 8.3125 | 13 / 3 / 0 | -1.3% | -0.6% | 0.021 | 0.406 |
| 2 | joint summed, warm-up 5, scale 1.0, w_post 0.5, w_near 0.0, v_small 0.1, lr 0.003 | 8.3125 | 11 / 5 / 0 | -2.1% | -1.3% | 0.051 | 0.963 |
| 3 | joint summed, warm-up 5, scale 1.0, w_post 1.0, w_near 0.0, v_small 0.0, lr 0.001 | 8.3750 | 13 / 2 / 1 | -2.1% | -3.4% | 0.012 | 0.237 |
| 4 | joint summed, warm-up 0, scale 1.0 | 8.5625 | 12 / 3 / 1 | -2.1% | -1.5% | 0.069 | 1.000 |
| 5 | joint summed, warm-up 5, scale 1.0, w_post 0.25, w_near 0.1, v_small 0.0, lr 0.003 | 8.8750 | 11 / 5 / 0 | -1.8% | -0.3% | 0.323 | 1.000 |
| 6 | joint summed, warm-up 5, scale 1.0, w_post 1.0, w_near 0.1, v_small 0.0, lr 0.003 | 9.1875 | 11 / 5 / 0 | -1.0% | -0.6% | 0.175 | 1.000 |
| 7 | joint separated, warm-up 5, scale 1.0, w_post 0.25, w_near 0.1, v_small 0.0, lr 0.003 | 9.6875 | 9 / 7 / 0 | -1.2% | -0.6% | 0.375 | 1.000 |
| 8 | joint separated, warm-up 5, scale 1.0, w_post 0.5, w_near 0.0, v_small 0.0, lr 0.003 | 9.6875 | 11 / 4 / 1 | -1.5% | -0.7% | 0.061 | 1.000 |
| 9 | joint summed, warm-up 5, scale 1.0, w_post 1.0, w_near 0.0, v_small 0.0, lr 0.003 | 9.6875 | 12 / 4 / 0 | -1.8% | -0.6% | 0.083 | 1.000 |
| 10 | joint summed, warm-up 5, scale 1.0, w_post 0.5, w_near 0.0, v_small 0.0, lr 0.003 | 9.8125 | 9 / 7 / 0 | -0.7% | -1.3% | 0.323 | 1.000 |
| 11 | joint separated, warm-up 0, scale 1.0 | 9.9375 | 10 / 6 / 0 | -1.2% | -0.3% | 0.528 | 1.000 |
| 12 | joint separated, warm-up 5, scale 1.0, w_post 0.25, w_near 0.0, v_small 0.0, lr 0.003 | 10.3750 | 11 / 4 / 1 | -1.4% | -0.4% | 0.156 | 1.000 |
| 13 | joint separated, warm-up 0, scale 0.3 | 11.0000 | 10 / 6 / 0 | -0.9% | -0.2% | 0.528 | 1.000 |
| 14 | joint summed, warm-up 5, scale 0.3 | 11.1250 | 12 / 3 / 1 | -0.6% | -0.6% | 0.156 | 1.000 |
| 15 | alternating, block 2, scale 1.0 | 11.1875 | 10 / 6 / 0 | -0.6% | -1.4% | 0.900 | 1.000 |
| 16 | alternating, block 5, scale 1.0 | 11.7500 | 9 / 7 / 0 | -1.2% | -1.3% | 0.860 | 1.000 |
| 17 | frozen-core reuse | 12.8750 | 6 / 9 / 1 | +0.0% | -0.2% | 1.000 | 1.000 |
| 18 | sequential, scale 0.1 | 12.8750 | 9 / 6 / 1 | -0.7% | -0.3% | 0.363 | 1.000 |
| 19 | alternating default (reference) | 13.1250 | - | 0.0% | 0.0% | - | - |
| 20 | retrieval only | 15.2500 | 5 / 9 / 2 | +0.7% | -0.2% | 0.158 | 1.000 |


**Table 35.** Batch-32 confirmation (superseded): all 28 conditions against the alternating default

| Rank | Condition | Mean test rank | Better / worse / tied | Median change | Median pre-adaptation change | p (uncorrected) | p x 27 (Bonferroni, max 1) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | joint from scratch, separated | 9.3125 | 12 / 4 / 0 | -7.6% | -4.2% | 0.021 | 0.578 |
| 2 | joint summed, warm-up 0, scale 1.0 | 9.7500 | 13 / 2 / 1 | -4.5% | -0.3% | 0.005 | 0.145 |
| 3 | joint summed, warm-up 5, scale 1.0, w_post 1.0, w_near 0.1, v_small 0.0, lr 0.003 | 10.0625 | 13 / 3 / 0 | -11.6% | -0.3% | 0.008 | 0.206 |
| 4 | joint summed, warm-up 5, scale 1.0, w_post 1.0, w_near 0.0, v_small 0.0, lr 0.001 | 10.9375 | 12 / 3 / 1 | -6.3% | -2.3% | 0.017 | 0.461 |
| 5 | joint summed, warm-up 5, scale 1.0, w_post 0.5, w_near 0.1, v_small 0.0, lr 0.001 | 11.0625 | 13 / 3 / 0 | -5.3% | -0.4% | 0.016 | 0.419 |
| 6 | joint summed, warm-up 5, scale 1.0, w_post 1.0, w_near 0.1, v_small 0.0, lr 0.001 | 11.1250 | 12 / 4 / 0 | -4.5% | -0.9% | 0.039 | 1.000 |
| 7 | joint summed, warm-up 5, scale 1.0, w_post 1.0, w_near 0.0, v_small 0.1, lr 0.001 | 12.3125 | 11 / 5 / 0 | -4.7% | -3.3% | 0.130 | 1.000 |
| 8 | joint separated, warm-up 5, scale 1.0 | 12.8750 | 11 / 5 / 0 | -1.9% | -0.8% | 0.193 | 1.000 |
| 9 | joint summed, warm-up 5, scale 1.0, w_post 0.5, w_near 0.0, v_small 0.0, lr 0.001 | 12.8750 | 12 / 3 / 1 | -4.5% | -3.0% | 0.020 | 0.537 |
| 10 | joint summed, warm-up 5, scale 1.0 | 13.0625 | 11 / 5 / 0 | -3.4% | -0.4% | 0.159 | 1.000 |
| 11 | joint from scratch, summed | 13.2500 | 10 / 6 / 0 | -3.9% | +0.7% | 0.193 | 1.000 |
| 12 | joint summed, warm-up 5, scale 1.0, w_post 0.5, w_near 0.0, v_small 0.0, lr 0.003 | 13.3750 | 11 / 4 / 1 | -5.4% | -1.7% | 0.027 | 0.722 |
| 13 | joint summed, warm-up 5, scale 0.3 | 13.5000 | 13 / 3 / 0 | -3.7% | -1.2% | 0.009 | 0.248 |
| 14 | joint summed, warm-up 5, scale 1.0, w_post 0.25, w_near 0.0, v_small 0.0, lr 0.001 | 13.5000 | 10 / 5 / 1 | -3.3% | -4.2% | 0.078 | 1.000 |
| 15 | joint summed, warm-up 0, scale 0.3 | 13.8125 | 11 / 5 / 0 | -1.6% | -0.7% | 0.231 | 1.000 |
| 16 | joint separated, warm-up 5, scale 0.3 | 14.3750 | 12 / 2 / 2 | -2.2% | -0.0% | 0.008 | 0.206 |
| 17 | joint separated, warm-up 5, scale 1.0, w_post 1.0, w_near 0.1, v_small 0.1, lr 0.003 | 14.6250 | 10 / 5 / 1 | -3.1% | +0.2% | 0.125 | 1.000 |
| 18 | alternating, block 5, scale 1.0 | 14.6875 | 11 / 5 / 0 | -3.7% | +1.6% | 0.144 | 1.000 |
| 19 | alternating, block 1, scale 1.0 | 15.2500 | 11 / 4 / 1 | -1.7% | +0.3% | 0.112 | 1.000 |
| 20 | joint separated, warm-up 5, scale 1.0, w_post 0.25, w_near 0.1, v_small 0.0, lr 0.001 | 15.5000 | 10 / 6 / 0 | -2.6% | -0.4% | 0.105 | 1.000 |
| 21 | joint summed, warm-up 5, scale 0.1 | 15.8125 | 9 / 5 / 2 | -1.3% | -0.4% | 0.084 | 1.000 |
| 22 | joint separated, warm-up 5, scale 0.1 | 16.3750 | 8 / 7 / 1 | -0.1% | -0.1% | 0.650 | 1.000 |
| 23 | frozen-core reuse | 17.0000 | 8 / 6 / 2 | -0.0% | -0.0% | 0.300 | 1.000 |
| 24 | joint summed, warm-up 0, scale 0.1 | 17.8125 | 8 / 7 / 1 | -0.8% | +0.0% | 0.211 | 1.000 |
| 25 | alternating default (reference) | 18.4375 | - | 0.0% | 0.0% | - | - |
| 26 | sequential, scale 0.1 | 21.0000 | 7 / 8 / 1 | +0.3% | -0.6% | 0.910 | 1.000 |
| 27 | sequential, scale 0.3 | 21.3125 | 6 / 9 / 1 | +0.8% | -0.0% | 0.820 | 1.000 |
| 28 | retrieval only | 23.0000 | 3 / 11 / 2 | +0.9% | -0.0% | 0.140 | 1.000 |


**Table 36.** Pooled validation ranking of the 73 candidates (development seeds 20, 21): top ten and references

| Cores | Pooled rank | Condition | Mean validation rank | Tasks won |
| --- | --- | --- | --- | --- |
| batch 32 | 1 | joint summed, warm-up 5, scale 1.0, w_post 1.0, w_near 0.1, v_small 0.0, lr 0.003 | 17.6875 | 3 |
| batch 32 | 2 | joint summed, warm-up 5, scale 1.0, w_post 0.5, w_near 0.0, v_small 0.0, lr 0.001 | 22.6875 | 1 |
| batch 32 | 3 | joint summed, warm-up 5, scale 1.0, w_post 0.25, w_near 0.0, v_small 0.0, lr 0.001 | 23.9375 | 0 |
| batch 32 | 4 | joint summed, warm-up 5, scale 1.0, w_post 0.5, w_near 0.0, v_small 0.0, lr 0.003 | 26.8750 | 0 |
| batch 32 | 5 | joint summed, warm-up 5, scale 1.0, w_post 1.0, w_near 0.1, v_small 0.0, lr 0.001 | 27.9375 | 0 |
| batch 32 | 6 | joint summed, warm-up 5, scale 1.0, w_post 0.5, w_near 0.1, v_small 0.0, lr 0.003 | 28.3125 | 0 |
| batch 32 | 7 | joint summed, warm-up 5, scale 1.0, w_post 1.0, w_near 0.0, v_small 0.0, lr 0.001 | 28.5000 | 1 |
| batch 32 | 8 | joint summed, warm-up 5, scale 1.0, w_post 0.25, w_near 0.1, v_small 0.0, lr 0.001 | 29.5000 | 0 |
| batch 32 | 9 | joint summed, warm-up 5, scale 1.0, w_post 0.25, w_near 0.1, v_small 0.0, lr 0.003 | 29.7500 | 0 |
| batch 32 | 10 | joint summed, warm-up 5, scale 1.0, w_post 0.5, w_near 0.1, v_small 0.0, lr 0.001 | 29.9375 | 1 |
| batch 32 | 56 of 73 | frozen-core reuse | 39.9375 | - |
| batch 32 | 66 of 73 | alternating default | 47.9375 | - |
| batch 32 | 68 of 73 | sequential, scale 0.1 | 50.0625 | - |
| batch 32 | 72 of 73 | sequential, scale 0.3 | 52.3125 | - |
| batch 32 | 73 of 73 | sequential, scale 1.0 | 61.4375 | - |
| batch 64 | 1 | joint summed, warm-up 5, scale 1.0, w_post 0.25, w_near 0.1, v_small 0.0, lr 0.003 | 19.0000 | 0 |
| batch 64 | 2 | joint separated, warm-up 5, scale 1.0, w_post 0.25, w_near 0.0, v_small 0.0, lr 0.003 | 21.3750 | 1 |
| batch 64 | 3 | joint separated, warm-up 5, scale 1.0, w_post 0.5, w_near 0.0, v_small 0.0, lr 0.003 | 22.5000 | 0 |
| batch 64 | 4 | joint separated, warm-up 5, scale 1.0, w_post 0.25, w_near 0.1, v_small 0.0, lr 0.003 | 23.1250 | 0 |
| batch 64 | 5 | joint separated, warm-up 5, scale 1.0, w_post 0.5, w_near 0.1, v_small 0.0, lr 0.003 | 23.1250 | 0 |
| batch 64 | 6 | joint separated, warm-up 5, scale 1.0, w_post 1.0, w_near 0.0, v_small 0.0, lr 0.003 | 24.4375 | 0 |
| batch 64 | 7 | joint separated, warm-up 5, scale 1.0, w_post 1.0, w_near 0.1, v_small 0.1, lr 0.003 | 25.2500 | 0 |
| batch 64 | 8 | joint summed, warm-up 5, scale 1.0, w_post 1.0, w_near 0.1, v_small 0.0, lr 0.003 | 25.3750 | 2 |
| batch 64 | 9 | joint separated, warm-up 5, scale 1.0, w_post 1.0, w_near 0.1, v_small 0.0, lr 0.003 | 25.8125 | 0 |
| batch 64 | 10 | joint summed, warm-up 5, scale 1.0, w_post 0.5, w_near 0.1, v_small 0.0, lr 0.003 | 27.5000 | 0 |
| batch 64 | 15 of 73 | frozen-core reuse | 30.4375 | - |
| batch 64 | 63 of 73 | alternating default | 48.1875 | - |
| batch 64 | 70 of 73 | sequential, scale 0.1 | 54.8125 | - |
| batch 64 | 71 of 73 | sequential, scale 0.3 | 55.3750 | - |
| batch 64 | 73 of 73 | sequential, scale 1.0 | 59.5625 | - |


**Table 37.** Correction size of the batch-64 conditions relative to the alternating default (median over 16 tasks)

| Condition (batch 64 confirmation) | Median correction size, ratio to default |
| --- | --- |
| joint summed, warm-up 5, scale 1.0, w_post 1.0, w_near 0.1, v_small 0.0, lr 0.003 | x1.8097 |
| joint summed, warm-up 5, scale 1.0, w_post 1.0, w_near 0.0, v_small 0.0, lr 0.003 | x1.8064 |
| joint summed, warm-up 5, scale 1.0, w_post 1.0, w_near 0.0, v_small 0.0, lr 0.001 | x1.7068 |
| joint summed, warm-up 5, scale 1.0, w_post 0.25, w_near 0.1, v_small 0.0, lr 0.003 | x1.5743 |
| joint summed, warm-up 5, scale 1.0, w_post 0.5, w_near 0.0, v_small 0.0, lr 0.003 | x1.5210 |
| joint separated, warm-up 5, scale 1.0, w_post 0.25, w_near 0.1, v_small 0.0, lr 0.003 | x1.3258 |
| joint separated, warm-up 5, scale 1.0, w_post 0.5, w_near 0.1, v_small 0.0, lr 0.003 | x1.3240 |
| joint summed, warm-up 0, scale 1.0 | x1.3026 |
| joint summed, warm-up 5, scale 1.0, w_post 0.5, w_near 0.0, v_small 0.1, lr 0.003 | x1.2703 |
| joint summed, warm-up 5, scale 0.3 | x1.1955 |
| sequential, scale 0.1 | x1.1359 |
| joint separated, warm-up 5, scale 1.0, w_post 0.25, w_near 0.0, v_small 0.0, lr 0.003 | x1.1282 |
| alternating, block 5, scale 1.0 | x1.1188 |
| joint separated, warm-up 5, scale 1.0, w_post 0.5, w_near 0.0, v_small 0.0, lr 0.003 | x1.1051 |
| joint separated, warm-up 0, scale 1.0 | x1.0802 |
| alternating, block 2, scale 1.0 | x1.0638 |
| joint separated, warm-up 0, scale 0.3 | x1.0457 |
| alternating default (reference) | x1.0000 |


Reading:
- Table 32: the selected joint schedule against the alternating default: 13 better, 3 worse, -11.6%, p = 0.008 at batch 32 (superseded); 11 better, 5 worse, -1.8%, p = 0.323 at batch 64. Against the other references its p is at least 0.144 at batch 64.
- Table 33: the per-task changes between batch sizes are large: breast_cancer -25.0% becomes +31.6%, digits -14.3% becomes +2.6%, zebra_special -17.5% becomes +11.5%, iris +28.6% becomes +71.4%, zebra -33.3% becomes -75.0%.
- Table 34: at batch 64 all 14 top-ranked conditions are joint; the alternating, sequential and frozen-reuse conditions rank 15th to 18th, the default 19th and retrieval only 20th. The smallest uncorrected p is 0.012 (summed, w_post 1.0, w_near 0, v_small 0, lr 0.001), which is 0.237 after correcting for 19 comparisons. No condition is significant after correction.
- Table 35: at batch 32 the top 17 of 28 conditions are all joint, the default is 25th, sequential 26th and 27th, retrieval only 28th. The smallest corrected p is 0.145.
- Table 36: the pooled validation winner has mean rank 17.6875 at batch 32 and 19.0000 at batch 64, and wins 3 and 0 tasks. The default is 66th (batch 32) and 63rd (batch 64) of 73. Frozen-core reuse is 56th at batch 32 and 15th at batch 64.
- Table 37: summed-objective joint variants enlarge corrections more (x1.1955 to x1.8097) than separated-objective ones (x1.0457 to x1.3258). The selected schedule is x1.5743 (see caveat 12 in Table 51).
- **Recommendation in the project's note:** use joint training with an adapter-only warm-up of 5 epochs, retrieval scale 1.0 and adapter rate 0.003; on well-tuned cores report robustness and rank, not an accuracy gain; keep v_small at 0.1 unless accuracy alone is the goal (the validation search preferred v_small 0, and the v_small 0.1 variant of Table 37 enlarges corrections by x1.2703); use one global schedule, because per-task selection from two development seeds was noisier. The last point is superseded by the H6 decision for per-task selection (Table 39). The recommendation is not applied as a default in code or configuration.

### 7.3 The pre-registration

**Table 38.** The pre-registered design for the tuning and confirmation phase

| Item | Pre-registered content (frozen 10-12 21:41 PDT; the project's own notes date it and the addenda 2026-10-13, in UTC) |
| --- | --- |
| Seeds | Development 20 to 25 (26 to 29 reserve): validation streams for selection and early stopping. Evaluation 40 to 49: test streams, run once after the final configuration F is frozen. Extra evaluation 50 to 69 for non-OpenML tasks with fewer than 100 test rows. Seeds 30 to 34 are retired because they informed decisions (the EMA conclusion, the schedule recommendation, the v_small choice) and cannot confirm anything tuned now. Seeds 35 to 39 are reserved for a documented second attempt. For TabArena tasks seed s maps to outer split ((s // 3) % 10, s % 3), and the seed audit must pass before any evaluation run. |
| Tasks | 20 primary legacy tasks (9 classification, 11 regression) plus 10 transfer tasks never used to tune neural CBR (5 + 5; training stream capped at 3,000 rows for every method, validation and test uncapped). A secondary group, never pooled with the primary analysis: bike_sharing and califonia_housing, reported separately when their hyperparameter search in the grid finishes. |
| Metric | Error is 1 - accuracy, or train-standardized RMSE. Per-task effect is log((e_F + eps) / (e_ref + eps)) with eps = 0.5 / n_test for classification and 0 for regression, averaged over seed-paired runs. Tasks are pooled by REML random effects with a Hartung-Knapp interval. GMR is the exponential of the pooled log ratio; lower is better. |
| Endpoints | Endpoint 1 (reuse gate): F against NN-kNN retrieval-only on the same core; one-sided superiority; supported if Holm-significant and GMR at most 0.97. Endpoint 2 (synchronization gate): F against the alternating default at F's budget and stopping rule; one-sided superiority; supported if Holm-significant, GMR at most 0.97 and all guards hold: pre-adaptation GMR against retrieval-only at most 1.02, fixed-metric locality ratio against the default at most 1.10, median correction size at most 1.25 times the default's. Endpoints 3a and 3b: non-inferiority against the frozen HPO winners of CatBoost and TabM, margin 0.10 (a proposed default the PI has not confirmed); since addendum 1, supported iff the Holm-adjusted one-sided p of H0: GMR at least 1.10 is at most 0.05. Holm over the four at family-wise 0.05; Benjamini-Hochberg at q = 0.10 for secondary analyses. |
| Analysis set and wording | Primary analysis: the 20 legacy plus 10 transfer tasks (30), and each group alone; no superiority claim if the transfer group's interval excludes 0 in the opposite direction. Secondary: median ECE after validation-temperature calibration at most the pre-adaptation ECE + 0.02; wrong-to-correct flips at least 2 times correct-to-wrong (classification); log loss and macro-F1 for churn and MIC; ranks over six methods (CatBoost, TabM, MLP, kNN, NN-kNN retrieval-only, F); cost. If endpoint 1 fails: no detectable gain from reuse on tuned cores. If endpoint 2 fails: no detectable schedule effect (or "gate 5 fails", the project's name for the synchronization gate, if a guard fails). If endpoint 3 fails: trails tuned CatBoost or TabM by X% [95% CI]. |
| Tuning order and blocks (step D) | H6 selection rule and seed count (adopt only if better than the current rule by more than 1 LOSO standard error). H1 retrieval guard and w_pre (feasible only if validation pre-adaptation error is at most 1.02 times retrieval-only plus 1/n_val). H2 classification output: score scale kappa in {1, 3, 10} and logit residual, each with v_small in {0, 0.1}, with a validation-fitted temperature (primary mode nominal_residual_scores). H5 budget and stopping rule. H3 adapter capacity and regularization (replace only if the 80% LOSO interval excludes 0 and the gain is at least 1%). H4 neighbourhood size (optional; only if H6 shows selection can separate candidates). Each block starts from the current incumbent and replaces it only by its acceptance rule; the first starts from F0, the batch-64 validation selection of Table 32. Selection and early stopping use development seeds and validation streams. H6 also scores its picks on the development seeds' test streams, declared as the held-out stream of the leave-one-seed-out test and "spent afterwards". Every candidate's validation table is published, failures included. |
| Forbidden | Dropping tasks after results; moving the margin or any threshold after any evaluation-seed result; rerunning evaluation seeds (except logged, bitwise-identical reruns after a crash); promoting a runner-up; tuning on transfer tasks; citing the full study's batch-32 tuned-core plan results for a confirmatory claim. |
| Addendum 1 (10-12 23:24) | Endpoints 3a and 3b are decided by the Holm-adjusted one-sided non-inferiority p-value (supported iff at most 0.05). A run whose retrieval guard rejects every epoch returns its epoch-0 state; its condition (not the run) is ineligible for selection, and the confirmation report lists the row. |
| Addendum 2 (10-12 23:30) | car and regression wine are excluded from tuning (about 55 compute-hours per task per block on six development seeds) but kept in confirmation with the frozen F. |
| Addendum 3 (10-13 06:44) | After H6 adopted per-task selection, H1's feasibility condition (from the selection seeds only) is applied per task and, from H1 on, to the whole candidate pool; frozen-core reuse is always feasible. The original wording says "on every task" (global form); the global form is kept for the single schedule used on the transfer tasks. Disclosure: a preliminary H1 evaluation without the constraint (17 tasks) was computed before the addendum was written. Addendum 3 is in the prose only: it is not in the hashed machine-readable copy, and the ledger records no hash for it. |
| Hashes of configs/t1/prereg_20261013.json | At freeze b6a10acf5717; after addendum 1 e48a064051c0; after addendum 2 7c27c5103631 (current, used for the confirmation report). Full values are in docs/T1_EXECUTION_LEDGER.md. |


### 7.4 Tuning decisions (step D)

Each block is judged by leave-one-development-seed-out: for each development seed the rule picks a condition from the validation rows of the other five seeds, and the pick is scored on the held-out seed's test stream. H6, H1, H2a and H2b are adopted only if they lower the paired error ratio by more than one task-clustered standard error; H2c adopts kappa 10 if the paired difference is at most +1 task SE (non-inferiority); H3 has its own rule (Table 38). F0 is the incumbent.

**Table 39.** H6: six selection rules, leave-one-development-seed-out, 18 tasks x 6 seeds (108 cells); GMR is an error ratio against the alternating default

| Selection rule / scope | Without feasibility: GMR | paired diff | task SE | z | better / worse | adopt-eligible | With feasibility: GMR | paired diff | task SE | z | better / worse | adopt-eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| err/global (current) | 0.9318 | - | - | - | - | - | 0.9666 | - | - | - | - | - |
| err/pertask | 0.8913 | -0.0444 | 0.0224 | -1.98 | 13 / 4 | yes | 0.8951 | -0.0769 | 0.0463 | -1.66 | 12 / 5 | yes |
| loss/global | 0.9013 | -0.0333 | 0.0153 | -2.17 | 14 / 2 | yes | 0.9636 | -0.0031 | 0.0063 | -0.49 | 6 / 5 | no |
| loss/pertask | 0.8993 | -0.0354 | 0.0188 | -1.88 | 12 / 5 | yes | 0.9065 | -0.0642 | 0.0314 | -2.04 | 12 / 5 | yes |
| rank/global | 0.9377 | +0.0063 | 0.0137 | +0.46 | 3 / 11 | no | 0.9566 | -0.0103 | 0.0154 | -0.67 | 9 / 7 | no |
| rank/pertask | 0.9043 | -0.0299 | 0.0186 | -1.61 | 10 / 7 | yes | 0.9068 | -0.0639 | 0.0433 | -1.48 | 10 / 7 | yes |


**Table 40.** H1, H2a, H2b, H2c: does adding the block's candidates to the pool lower the held-out error ratio?

| Block | Cells / tasks | GMR before | GMR after | Paired difference | Task SE | z | Tasks better / worse | Held-out cells choosing a block candidate | Block candidates picked at least once | Adopted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H1 retrieval guard and w_pre | 108 / 18 | 0.8951 | 0.9045 | +0.0104 | 0.0110 | +0.95 | 1 / 3 | 13 of 108 | 5 of 10 | no |
| H2a score scale kappa | 54 / 9 | 0.8416 | 0.8384 | -0.0038 | 0.0099 | -0.38 | 2 / 3 | 33 of 54 | 4 of 5 | no |
| H2b logit residual | 54 / 9 | 0.8416 | 0.8416 | +9.9e-09 | 0.0315 | 3.1e-07 | 1 / 2 | 16 of 54 | 2 of 2 | no |
| H2c classification pool at kappa 10 against kappa 1 (rule: adopt if the paired difference is at most +1 task SE) | 54 / 9 | 0.8416 | 0.8224 | -0.0231 | 0.0355 | -0.65 | 5 / 3 | - | - | yes (non-inferior) |


**Table 41.** Conditions chosen by the per-task rule in the 108 held-out cells, with the rejected H1 candidates added to the pool (the adopted pool has no H1 candidates)

| Condition chosen by the per-task rule (after H1) | Held-out cells |
| --- | --- |
| joint summed, warm-up 5, scale 1.0, w_post 1.0, w_near 0.1, v_small 0.0, lr 0.003 | 16 |
| joint summed, warm-up 5, scale 1.0, w_post 1.0, w_near 0.0, v_small 0.0, lr 0.003 | 14 |
| joint summed, warm-up 5, scale 1.0, w_post 0.5, w_near 0.0, v_small 0.1, lr 0.003 | 10 |
| frozen-core reuse | 8 |
| joint separated, warm-up 5, scale 1.0, w_post 0.5, w_near 0.0, v_small 0.0, lr 0.003 | 8 |
| joint summed, warm-up 5, scale 1.0, w_post 0.25, w_near 0.1, v_small 0.0, lr 0.003 | 8 |
| alternating, block 2, scale 1.0 | 6 |
| joint separated, warm-up 5, scale 1.0, w_post 0.5, w_near 0.1, v_small 0.0, lr 0.003 | 6 |
| H1 summed, w_pre=4, guard=none | 5 |
| joint summed, warm-up 0, scale 1.0 | 5 |
| joint separated, warm-up 5, scale 1.0, w_post 0.25, w_near 0.0, v_small 0.0, lr 0.003 | 4 |
| H1 summed, w_pre=2, guard=0.02 | 3 |
| H1 summed, w_pre=4, guard=0.02 | 3 |
| joint summed, warm-up 5, scale 1.0, w_post 0.5, w_near 0.0, v_small 0.0, lr 0.003 | 3 |
| joint separated, warm-up 0, scale 1.0 | 2 |
| joint summed, warm-up 5, scale 0.3 | 2 |
| joint separated, warm-up 5, scale 1.0, w_post 0.25, w_near 0.1, v_small 0.0, lr 0.003 | 2 |
| alternating, block 5, scale 1.0 | 1 |
| H1 separated, w_pre=2, guard=0.02 | 1 |
| H1 summed, w_pre=2, guard=none | 1 |
| Total (cells / tasks) | 108 / 18 |


**Table 42.** H2: calibration and accuracy on the validation stream by condition

| Condition (9 classification tasks, seeds 22 to 25, 36 cells) | Median ECE after | Median ECE before | Median correction size | Mean ECE after | Mean accuracy after |
| --- | --- | --- | --- | --- | --- |
| F0 (kappa 1, v_small 0) | 0.2407 | 0.0305 | 0.9966 | 0.2343 | 0.9281 |
| H2a kappa 10, v_small 0 | 0.0183 | 0.0404 | 0.2469 | 0.0525 | 0.9291 |
| H2a kappa 10, v_small 0.1 | 0.0156 | 0.0364 | 0.2200 | 0.0507 | 0.9318 |
| H2a kappa 1, v_small 0.1 | 0.2728 | 0.0322 | 0.6032 | 0.2788 | 0.9308 |
| H2a kappa 3, v_small 0 | 0.0504 | 0.0278 | 0.4647 | 0.0633 | 0.9282 |
| H2a kappa 3, v_small 0.1 | 0.0713 | 0.0286 | 0.3515 | 0.0787 | 0.9285 |
| H2b logit residual, v_small 0 | 0.0182 | 0.0289 | 0.0367 | 0.0369 | 0.9297 |
| H2b logit residual, v_small 0.1 | 0.0264 | 0.0273 | 0.0222 | 0.0398 | 0.9291 |


**Table 43.** H2: F0 against kappa 10 with v_small 0.1, per task (validation stream, mean over seeds 22 to 25)

| Task | ECE after: F0 | kappa 10, v_small 0.1 | Accuracy after: F0 | kappa 10, v_small 0.1 |
| --- | --- | --- | --- | --- |
| iris | 0.2673 | 0.0071 | 0.9750 | 0.9917 |
| wine | 0.2624 | 0.0086 | 0.9444 | 0.9931 |
| breast_cancer | 0.1862 | 0.0184 | 0.9803 | 0.9825 |
| balance | 0.2443 | 0.0313 | 0.9620 | 0.9640 |
| digits | 0.4638 | 0.0122 | 0.9840 | 0.9861 |
| zebra | 0.1786 | 0.0000 | 1.0000 | 1.0000 |
| zebra_special | 0.1398 | 0.1723 | 0.7557 | 0.7386 |
| tabarena_363614 | 0.3064 | 0.0015 | 0.9967 | 0.9983 |
| tabarena_363626 | 0.0602 | 0.2046 | 0.7545 | 0.7320 |
| Median over tasks | 0.2443 | 0.0122 | 0.9750 | 0.9861 |


**Table 44.** A single-task check: adapted error falls while retrieval before adaptation degrades

| Condition (superconductivity, TabArena 363705, development seed 20, 5,000 training cases, one fixed raw core without hyperparameter search) | Test RMSE before adaptation | Test RMSE after adaptation | Mean correction size |
| --- | --- | --- | --- |
| retrieval only | 0.5215 | - | - |
| alternating default | 1.3561 | 0.4296 | 0.9361 |
| F0 (joint) | 1.3719 | 0.4051 | 0.9088 |


Reading:
- Table 39 (H6): per-task selection by validation error lowers the GMR from 0.9318 to 0.8913 (paired -0.0444, z -1.98, 13 tasks better, 4 worse) and was adopted. It is the most negative paired difference among the rules that beat the current one, not the largest |z|: loss/global has z -2.17 with a smaller gain. Under the feasibility constraint the per-task rule keeps GMR 0.8951 (z -1.66) while the global rule rises to 0.9666. The seed-count half of H6 has no recorded decision.
- Table 40 (H1): adding the retrieval-guard candidates raises the GMR from 0.8951 to 0.9045 (paired +0.0104, z +0.95; 1 task better, 3 worse) and is rejected. H2a (-0.0038, z -0.38) and H2b (+9.9e-09) are not adopted.
- Table 41: with the H1 candidates in the pool, the per-task rule picks 20 different conditions (15 non-H1 conditions and 5 H1 candidates, 13 cells). The alternating default and the sequential schedules are never picked; F0 itself is picked in 8 of 108 cells.
- Table 42 and Table 43 (H2): kappa 10 with v_small 0.1 lowers the median ECE after adaptation from 0.2407 to 0.0156 and the correction size from 0.9966 to 0.2200, with accuracy 0.9318 against 0.9281 (a difference the held-out test in Table 40 does not separate from zero). H2c then reran the classification pool at kappa 10: the held-out GMR is 0.8416 at kappa 1 and 0.8224 at kappa 10 (paired -0.0231 against a task SE of 0.0355, z -0.65, 5 tasks better and 3 worse), so kappa 10 is adopted for classification as non-inferior, not as a gain. The calibration gain is not uniform: credit-g (tabarena_363626) gets worse in ECE (0.0602 to 0.2046) and accuracy (0.7545 to 0.7320), and so does zebra_special (ECE 0.1398 to 0.1723; accuracy 0.7557 to 0.7386). The mean ECE after adaptation over the 36 cells is 0.0507 for kappa 10 with v_small 0.1, against 0.2343 for F0.
- Table 44: on superconductivity the adapted RMSE falls (0.4296 for the alternating default, 0.4051 for F0) while the RMSE before adaptation rises to 1.3561 and 1.3719, against 0.5215 for retrieval only. The adapter hides degraded retrieval in this run. It is one development seed, one fixed core and one run per condition, and a run of the same core with tiny arithmetic differences collapsed (caveat 8). This is the case the endpoint 2 guard and the H1 block target; H1 was rejected on held-out error, and the pre-registration (addendum 3) applies a per-task feasibility rule for such cases. No table here shows that rule's effect on this task.
- Still open: H5 and H3 (both halves running, Table 49) and H4 (optional, not started). H2c is decided (Table 40); its rule is non-inferiority within one task SE.

### 7.5 The grids

**Table 45.** The 2026-10-09 grid: test metric by task and family (mean over evaluation seeds 30 to 34; accuracy for classification, RMSE in original units for regression; best in bold)

| Task | Metric | NN-kNN | kNN | CatBoost | MLP | TabM | Best | NN-kNN rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| balance (cla) | accuracy | 0.9376 | 0.8976 | 0.9120 | **0.9712** | 0.9648 | mlp | 3 |
| breast_cancer (cla) | accuracy | 0.9596 | **0.9684** | 0.9544 | 0.9649 | 0.9667 | knn | 4 |
| digits (cla) | accuracy | 0.9772 | 0.9767 | 0.9728 | 0.9756 | **0.9811** | tabm | 2 |
| iris (cla) | accuracy | **0.9533** | 0.9467 | **0.9533** | **0.9533** | 0.9333 | nnknn/catboost/mlp | 1 |
| tabarena_363614 (cla) | accuracy | 0.9846 | 0.9686 | **0.9900** | 0.9759 | 0.9800 | catboost | 2 |
| tabarena_363626 (cla) | accuracy | 0.7217 | 0.7331 | 0.7624 | 0.7588 | **0.7642** | tabm | 5 |
| wine (cla) | accuracy | 0.9444 | 0.9778 | **0.9833** | 0.9667 | 0.9722 | catboost | 5 |
| zebra (cla) | accuracy | 0.9273 | 0.8727 | **1.0000** | 0.5182 | 0.4455 | catboost | 2 |
| zebra_special (cla) | accuracy | 0.7091 | 0.7773 | **0.9455** | 0.4682 | 0.5318 | catboost | 3 |
| abalone (reg) | RMSE | 2.1876 | 2.3036 | 2.2295 | 2.1634 | **2.1361** | tabm | 3 |
| airfoil (reg) | RMSE | 1.6195 | 3.0175 | **1.4963** | 1.6453 | 1.6356 | catboost | 2 |
| bike_sharing (reg) | RMSE | pending | 104.9041 | **38.4972** | 41.1805 | 39.8251 | catboost | - |
| body_fat (reg) | RMSE | **0.0119** | 0.0126 | 0.0121 | 0.0249 | 0.0217 | nnknn | 1 |
| califonia_housing (reg) | RMSE | pending | 0.6046 | **0.4442** | 0.5046 | 0.4813 | catboost | - |
| car (reg) | RMSE | 14976.7 | 12244.2 | **10792.4** | 16882.8 | 14876.4 | catboost | 4 |
| diabets (reg) | RMSE | 55.3558 | 54.4540 | 55.9960 | 53.8401 | **53.0886** | tabm | 4 |
| energy_efficiency (reg) | RMSE | 0.5057 | 1.3877 | 0.4977 | 0.5367 | **0.4969** | tabm | 3 |
| student_performance (reg) | RMSE | 1.3034 | 1.8670 | 1.1500 | 1.2369 | **1.0752** | tabm | 4 |
| tabarena_363625 (reg) | RMSE | 6.3136 | 8.5623 | **4.5185** | 5.5122 | 5.3961 | catboost | 4 |
| tabarena_363675 (reg) | RMSE | 4987.9 | 6038.7 | **4593.8** | 4821.5 | 4652.4 | catboost | 4 |
| wine (reg) | RMSE | 0.6447 | 0.6398 | **0.6322** | 0.6768 | 0.6528 | catboost | 3 |
| yacht (reg) | RMSE | 2.2033 | 9.6897 | 1.2683 | 1.5301 | **1.0165** | tabm | 4 |


**Table 46.** Interim tally from the previous table (not an official report)

| Count over the tasks where all five families finished (n = 20) | NN-kNN | kNN | CatBoost | MLP | TabM |
| --- | --- | --- | --- | --- | --- |
| Sole best | 1 | 1 | 9 | 1 | 7 |
| Tied best (iris: NN-kNN, CatBoost, MLP) | 1 | 0 | 1 | 1 | 0 |
| NN-kNN rank 1 / 2 / 3 / 4 / 5 (number of tasks) | 2 / 4 / 5 / 7 / 2 |  |  |  |  |


**Table 47.** The 2026-10-09 grid: progress (snapshot)

| Item | Value |
| --- | --- |
| Jobs expected | 110 |
| Jobs completed: NN-kNN / kNN / CatBoost / MLP / TabM (of 22 each) | 20 / 22 / 22 / 22 / 22 |
| Jobs completed, all families | 108 |
| Failed jobs | 0 |
| Completed by 2026-10-09 23:58 (last handoff note of the study) | 27 |
| Completed by 2026-10-10 22:36 (10 Oct handoff) | 81 |
| Last job completed | regression__wine nnknn, 2026-10-11 07:47:06 |
| Unfinished job regression__bike_sharing: screening fits done (of 64) / newest output / selection, refit, evaluation started | 41 / 2026-10-13 06:08:51 / no, no, no |
| Unfinished job regression__califonia_housing: screening fits done (of 64) / newest output / selection, refit, evaluation started | 27 / 2026-10-13 04:32:07 / no, no, no |


**Table 48.** Transfer-task grid: which jobs are done (snapshot)

| Transfer task | Name | NN-kNN | kNN | CatBoost | MLP | TabM | Families done (of 5) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 363623 | churn | done | done | done | done | done | 5 |
| 363681 | in_vehicle_coupon | pending | done | done | done | done | 4 |
| 363702 | splice | done | done | done | done | done | 5 |
| 363704 | students_dropout | done | done | done | done | done | 5 |
| 363711 | MIC | done | done | done | done | done | 5 |
| 363631 | diamonds | pending | done | done | done | done | 4 |
| 363686 | miami_housing | done | done | done | done | done | 5 |
| 363693 | physiochemical_protein | pending | done | done | done | done | 4 |
| 363697 | QSAR-TID-11 | pending | done | done | done | done | 4 |
| 363705 | superconductivity | pending | done | done | done | done | 4 |


Reading:
- Table 47: the 2026-10-09 grid had 27 jobs done on 10-09 23:58 and 81 on 10-10 22:36, and 108 by 10-11 07:47 with no failure. None has finished since. The two unfinished jobs are NN-kNN on bike_sharing (41 of 64 screening fits) and califonia_housing (27 of 64); both are in the pre-registration's secondary group. The plan and radius stages start only after the grid exits, so they have not started.
- Table 45 and Table 46 (interim, unofficial; validation-selected, five evaluation seeds, no SOTA claim, unequal training cost): CatBoost is the sole best on 9 of 20 tasks and TabM on 7; NN-kNN is the sole best on body_fat and tied best on iris, and ranks 1st to 5th on 2, 4, 5, 7 and 2 tasks.
- Table 48: the transfer grid still has NN-kNN jobs pending on several tasks and some jobs on QSAR-TID-11 (see the table for the current state). It evaluates seeds 40 to 49 for the baselines and the NN-kNN cores; those metrics of the finished jobs are deliberately not listed here, because they belong to the pre-registered confirmation.

### 7.6 Running now and the scale of the runs

**Table 49.** Running experiments (snapshot, 10-13)

| Run | Started | Finished jobs / expected | Newest result file | State at 10-13 11:53 |
| --- | --- | --- | --- | --- |
| Legacy full-study grid | 10-09 23:37 | 108 / 110 | 10-11 07:47 | running; the two unfinished jobs are still screening |
| Transfer-task grid | 10-12 21:41 | 45 / 50 | 10-13 11:52 | running |
| H2c: classification pool at kappa 10 | 10-13 07:52 | 54 / 54 | 10-13 10:26 | finished; decided 10-13 10:28 (kappa 10 adopted for classification) |
| H5: budget and stopping, regression half | 10-13 07:56 | 4 / 54 | 10-13 11:43 | running |
| H5: budget and stopping, classification half (kappa 10) | 10-13 10:28 | 4 / 54 | 10-13 11:41 | running |
| H3: adapter capacity, regression half | 10-13 07:56 | 5 / 54 | 10-13 11:51 | running |
| H3: adapter capacity, classification half (kappa 10) | 10-13 10:28 | 17 / 54 | 10-13 11:52 | running |


**Table 50.** Runs executed by finished experiments (jobs, rows and summed job-hours from the run records; H2c and the running experiments are not included)

| Experiment | Jobs finished / expected | Failed | Result rows | Summed job-hours | Tasks | Seeds |
| --- | --- | --- | --- | --- | --- | --- |
| EMA ablation, default settings | 65 / 65 | 0 | 1300 | 10.7 | 13 | 30-34 |
| EMA check, tuned MLP cores (batch 32) | 30 / 30 | 0 | 360 | 5.4 | 6 | 30-34 |
| EMA check, tuned MLP cores (batch 64) | 30 / 30 | 0 | 360 | 5.2 | 6 | 30-34 |
| Schedule search stage A (batch 32) | 32 / 32 | 0 | 896 | 11.6 | 16 | 20-21 |
| Schedule search stage B (batch 32) | 32 / 32 | 0 | 1568 | 21.4 | 16 | 20-21 |
| Confirmation A (batch 32) | 80 / 80 | 0 | 1476 | 23.8 | 16 | 30-34 |
| Confirmation B (batch 32) | 80 / 80 | 0 | 880 | 13.4 | 16 | 30-34 |
| Schedule search stage A (batch 64) | 32 / 32 | 0 | 896 | 15.0 | 16 | 20-21 |
| Schedule search stage B (batch 64) | 32 / 32 | 0 | 1568 | 18.9 | 16 | 20-21 |
| Confirmation (batch 64) | 80 / 80 | 0 | 1600 | 25.2 | 16 | 30-34 |
| Tuning H6 (4 development seeds, 15 tasks) | 60 / 60 | 0 | 1200 | 9.9 | 15 | 22-25 |
| Tuning H6, 3 slow tasks (6 seeds) | 18 / 18 | 0 | 360 | 13.1 | 3 | 20-25 |
| Tuning H1 (17 tasks) | 102 / 102 | 0 | 1122 | 6.3 | 17 | 20-25 |
| Tuning H1, abalone | 6 / 6 | 0 | 66 | 5.5 | 1 | 20-25 |
| Tuning H2 (9 classification tasks) | 54 / 54 | 0 | 432 | 3.0 | 9 | 20-25 |
| Total, batch 32 group | 319 | 0 | 6480 | 86.3 |  |  |
| Total, batch 64 group | 174 | 0 | 4424 | 64.3 |  |  |
| Total, step D (H6, H1, H2) | 240 | 0 | 3180 | 37.7 |  |  |


## 8. Still to do, in the order the sources name

1. Finish H5 and H3 (both halves are running); run H4 only if H6 shows selection can separate candidates. H2c is decided.
2. Freeze the final configuration F. For the transfer group F is one global schedule chosen on the legacy pool.
3. Pass the seed audit, then run the confirmation on seeds 40 to 49 (car and regression wine appear only here) and the confirmation report with the pre-registered endpoints.
4. Step G (retention): provenance_only against random and bias_current.
5. Compare case-base maintenance policies without EMA, on batch-64 tuned cores with the joint schedule.
6. Finish the 2026-10-09 grid (two jobs), then its plan and radius stages, and rerun its tuned cores at batch 64.
7. From the carried-over list: the PI's scientific thresholds, real-data nominal reuse, a human study with real participants, and benchmark reconstruction (a full TabArena replication beyond the frozen subsets of Table 24 and Table 48, and the external MCB/IU source) remain open.

## 9. Caveats and open points

**Table 51.** Caveats and open points, each tied to the table that shows it

| # | Point | Where the evidence is | Effect on how the results may be read |
| --- | --- | --- | --- |
| 1 | The legacy plan stage builds tuned cores at batch 32 and starts by itself when the grid exits successfully | The study script runs the plan stage after the grid; the study executor does not set batch_size; Table 30 and Table 31 show what batch 32 does to tuned cores | Its results cannot be cited for a confirmatory claim (pre-registration, Table 38); the written open item is to rerun at batch 64 or label them |
| 2 | kappa 10 is better on median calibration but not on every task | Table 40 (H2c), Table 42, Table 43 | Adopted for classification after H2c on non-inferiority (paired -0.0231 against a task SE of 0.0355), not as an accuracy gain; credit-g and zebra_special get worse in both ECE and accuracy |
| 3 | The joint-training recommendation is soft | Table 32, Table 34 | On well-tuned cores the gain is -1.8% (p = 0.323). The recommended v_small 0.1 differs from the validation winner and rests on test results; the recommendation names no single tested condition. Its fourth point, one global schedule, is superseded by the H6 decision for per-task selection |
| 4 | The EMA ablation uses default hyperparameters; its tuned-core check has 6 tasks | Table 25, Table 28 | The conclusion is about these supervised tabular tasks only; images and text were not tested |
| 5 | E5 and E17 disagree on whether RRR helps on Energy | Table 13 | Treat the E5 ordering as descriptive, not as a ranking (Yacht was not repeated in E17) |
| 6 | E1 and E2 ran before the best-epoch optimizer restore (10-04) | Table 5, Table 6 | The audits re-ran their post-selection cells at the current code and the cells do not reproduce. Qualitative statements stand; individual cells are not bit-reproducible, and the Digits order of Q and B against random changes on seed 3 |
| 7 | Pre-registration housekeeping | Table 38 | Addendum 3 is not in the hashed copy; the margin, the primary output mode and the retention thresholds are still marked PI-pending; the pre-registration note's own endpoint table still states the old endpoint 3 rule (a one-sided 95% upper bound), while Table 38 gives the rule as changed by addendum 1; addenda 1 and 3 say "before any evaluation-seed result", but the transfer grid had baseline evaluations on seeds 40 to 49 from 10-12 21:42 and its first NN-kNN evaluation at 10-12 23:27, before addendum 3; the same development-seed test streams judge several blocks (repeated use) |
| 8 | An adapted (reuse) neural CBR run exists on one transfer task, apart from the NN-kNN grid jobs of Table 48 | Table 44 | superconductivity (TabArena 363705), development seed 20, one run per condition, 5,000 training cases and one fixed raw core without hyperparameter search. The core is unstable at this scale: a run of the same core with tiny arithmetic differences collapsed to the mean (validation loss 1.0, best epoch 3) while the reported run learned (validation loss 0.259, best epoch 284); two identical runs agree exactly. The pre-registration says transfer tasks are never used to tune neural CBR; it is not stated whether this run influenced a decision |
| 9 | Figures taken from commit messages or notes, with no stored timing or replay log | Table 4, Table 23 | The feature-cache speed figures and the about 60% share of the two logging passes (both in Table 4); E1's 61 checkpoint replays; the replay and audit counts of E10 and the metric counts of E18; the statement that the cores of the batch-64 repeats that differ (all on abalone seed 21) diverge at epoch 40 |
| 10 | Different runs use different code | Table 4, Table 49, the step-D decision record (H2c row) | The transfer grid runs the code frozen at pre-registration time; the tuning blocks run later code. The tuning decisions pool rows from several code versions (bit-identical on the sampled checks: 56 reruns on Iris and balance, 7 Iris seed 21 rows; the exception is abalone seed 21, whose core diverged in a repeated run under identical source) |
| 11 | The scientific thresholds of the aligned pipeline were still marked PILOT at the 6 Oct handoff | The earlier report, sections 2.2 and 2.4; the 10-09 alignment note; the pre-registration, retention section | The carried-over results use PILOT values from the base configuration (activation and cache 0.05, allowed loss 0.05, regression tolerance 0.5), not PI-approved formulas; the adapted-removal runs E13 to E16 set the loss budget to .03 (E15 also 0 and .1), also a declared pilot choice |
| 12 | The incumbent F0 enlarges corrections more than the pre-registered guard allows | Table 37, Table 38 | On seeds 30 to 34 the median correction size of F0 (kappa 1) is x1.5743 the default, against a guard of x1.25; the recommended v_small 0.1 variant is x1.2703, also above it. F is not frozen (per-task selection and kappa 10 change the corrections, Table 42), so the guard is decided at confirmation. Conditions below x1.25 include the alternating, sequential and separated-objective variants with w_near 0 and the summed warm-up-5 scale-0.3 variant |


## 10. Where the results are

All paths are under `neural-cbr-ema\` unless noted.

- Written notes: `docs\T1_EMA_JOINT_SCHEDULE_20261011.md`, `docs\T1_PREREG_20261013.md`, `docs\T1_EXECUTION_LEDGER.md`; for the carried-over results `neural-cbr\docs\experiments\` and `neural-cbr\docs\T1_*.md`.
- EMA: `results\t1\p5_mcb_interaction\ema_ablation_20261010\` and `ema_tuned_mlp_bs64_20261012\`.
- Schedule search: `results\t1\p4_component_sync\schedule_search_20261011\` (batch 32, superseded) and `schedule_search_bs64_20261012\`.
- Tuning decisions: `results\t1\p4_component_sync\tuning_20261013\` (`DECISIONS.md`, `h6_rules.json`, `h6_rules_feasible.json`, `h1_block.json`, `h2a_block.json`, `h2b_block_separate.json`).
- Grids: `neural-cbr\results\t1_pi20260920\grid_full_20261009\` and `results\t1\p6_confirmation_revision\grid_transfer_20261013\`.
- Carried-over raw artifacts: `neural-cbr\results\t1_pi20260920\continuation_20261002\` and `continuation_20261004\`.
