# Neural CBR (T1) weekly progress, 2026-10-13 to 2026-10-20

- Period: from the snapshot of the previous report (10-13, about 11:55 local time) to the snapshot of 10-20, about 01:00 local time (PDT, UTC-7).
- Previous report: `NEURAL_CBR_PROGRESS_2026-10-13_EN.md`.
- This document: `NEURAL_CBR_WEEKLY_2026-10-13_to_10-20_EN.md` and the Chinese version `NEURAL_CBR_WEEKLY_2026-10-13_to_10-20_ZH.md`.
- Background: `NEURAL_CBR_OVERVIEW_2026-10-20_EN.md`.

## 0. Scope, period and how to read this summary

**Scope.** This summary covers neural CBR (T1) only. Case-based reasoning (CBR) answers a new question by finding similar questions answered before and reusing their answers. A help desk does this when it answers a ticket from the fixes of similar past tickets. The comparison stops being accurate where a help desk uses human judgment and here programs learn both steps. Neural CBR adds learned parts: a measure of similarity, and a small neural network (a program whose internal numbers are adjusted automatically to fit examples) that corrects the reused answer.

A case is one row of a data table. A task is a dataset plus a quantity to predict. A seed is a number that fixes the random choices of one run, including the split of rows into training, validation and test sets. Dates are month-day, so 10-13 is October 13.

**How the sections are built.** The principal investigator (PI) divides the work into steps in a document called the handoff. Sections 2 to 7 each take one handoff point that was active this week. Each gives what the handoff asks, how we implemented it, the results, and numbered insights that say what each result means and does not show. Sections 8 to 10 give the status of open items, the caveats and where the results are. A pre-registration is a written, dated plan that fixes tasks, comparisons and success rules before the results exist.

**Evidence labels.** Supported means a table shows it. Conjecture means an interpretation, with the test that would settle it. Pilot and exploratory mean a small or early run that does not rank methods. Null and negative results are reported as plainly as positive ones.

**What interim means.** The confirmation run tests the final configuration F on fresh seeds that no tuning step used. At the snapshot, 29 of the 30 pre-registered tasks were complete. The 30th, QSAR-TID-11, had 7 of 10 seed jobs finished, where a seed job is one run of one task on one seed. The pre-registration has no clause for an interim analysis, so every confirmation number here is interim and not the official verdict. That verdict needs the 30th task, audits of its rows, and a recomputation of every confirmation table.

**Terms and numbers.** Table 1 defines the technical terms of Section 1. This section runs no experiment: it copies its numbers from Sections 2 to 7.

**Table 1.** Terms used in Section 1

| Term | Plain meaning |
| --- | --- |
| Held-out data | Data that the choice being tested did not see. |
| Error | Share of wrong answers for a category, or typical size of a miss for a number. Lower is better. |
| Error ratio, GMR | One method's error divided by another's on the same split. The geometric mean ratio (GMR) averages ratios over tasks, so doubling on one task and halving on another average to 1. Below 1 favors the first method. |
| Holm p, 95% interval | Holm p is a p-value (how unusual a result would be if there were no real difference) corrected for four questions at once. The 95% interval is the range the data leave plausible: if it contains 1, equal errors cannot be ruled out. |
| Non-inferior | Not worse than a reference by more than a stated margin, here 10% on the error ratio. |
| F, F_global | F is the final configuration: a recipe chosen per task for 18 tasks and one shared recipe, F_global, for the other 12. |
| Retrieval-only, adapter | Retrieval-only is the project's retrieval core (a nearest-neighbor model whose notion of "similar" is learned) without the adapter, the small network that corrects the answer. |
| Joint training | Retrieval and adapter are updated together, not in turns. |
| Kappa | A number that multiplies a classifier's scores before they become probabilities. |
| E1, E2, E3a, E3b | The four pre-registered questions: F against retrieval-only, joint against alternating training at equal budget, and F non-inferior to tuned CatBoost and to tuned TabM. |
| kNN, CatBoost, TabM | Standard methods for tables: kNN averages the k closest rows, CatBoost is boosted decision trees, TabM trains several small networks together. |
| Calibration error, flip | Calibration error is the gap between stated confidence and how often the model is right. A flip is a prediction that changes when the adapter is switched off. |
| Q, step G | Q ranks stored cases for removal when storage is limited, by how often a case helped or hurt. Step G is the planned test of that rule. |
| Addendum, receipt, audit | An addendum is a dated change to the pre-registration. A receipt is a file written when a job ends. An audit re-creates result rows from a run's own files. |

*Source: definitions as used in Sections 2 to 7 of this document; neural-cbr-ema/docs/T1_PREREG_20261013.md for the endpoints.*

## 1. Summary

This week closed tuning, froze the final design F, and ran the one-shot confirmation of F on fresh data. On 29 finished tasks the interim confirmation meets none of the four pre-registered success rules. The retrieval core alone improves on plain nearest-neighbor search more often than not, and the best tuned standard methods still lead. Table 2 gives one row per handoff point.

**Table 2.** Summary by handoff point (the confirmation and integrity rows are interim)

| Handoff point | Status at the snapshot | Headline result | Table |
| --- | --- | --- | --- |
| Step D: tuning (Section 2) | Complete: 510 of 510 jobs exited with code 0. H4 skipped | Two of seven comparisons passed. Per-task selection: held-out error ratio 0.932 to 0.891. Kappa 10: 0.842 to 0.822, non-inferior. H3 (+0.0084) and H5 (+0.0023) failed | Table 4 |
| D6: freezing F (Section 3) | Complete: 30 configuration files frozen, no new training | Winner within one SE of runner-up on 17 of 18 tuned tasks. In-sample ratio of per-task F 0.816, held-out analogue 0.898 | Table 12, Table 13 |
| Step F: confirmation (Section 4) | Interim: 29 of 30 tasks. QSAR-TID-11 at 7 of 10 seed jobs | No endpoint met. E1 0.973 (Holm p 0.294). E2 0.996 (Holm p 0.313). E3a 1.139. E3b 0.984 (Holm p 0.274) | Table 18 |
| Harder datasets, baselines, grids (Section 5) | Transfer grid 48 of 50 jobs, legacy grid 108 of 110 | Retrieval core against kNN: error ratio 0.800 (transfer), 0.788 (legacy). Against CatBoost: 1.110 and 1.280 | Table 27, Table 30 |
| Step G preparation: maintenance review (Section 6) | Review and code changes done. Step G not run | 13 flaws and one extra defect. On five development tasks 0.568 to 0.983 of stored cases are never retrieved | Table 34, Table 35 |
| Integrity of the confirmation run (Section 7) | Interim: checks pass for 29 of 30 tasks | 437 of 440 jobs exited with code 0, none failed. 2,059 result lines, 2,043 unique runs, 16 duplicates, all identical. 204 grid receipts predate addendum 1 | Table 40, Table 42 |

*Source: Sections 2 to 7 of this document and the tables they cite.*

### 1.1 Key findings

1. **Tuning adopted two changes and rejected the rest [supported].** Choosing the recipe per task lowered the held-out error ratio from 0.932 to 0.891 and passed its rule. Kappa 10 passed only as non-inferior (0.842 to 0.822): a calibration choice, not an accuracy gain. H3 and H5 failed. In classification, wherever a new candidate became the pick, error fell on validation data and rose on held-out data (-0.1222 against +0.0391 for the budget block), so selection overfits there. Standard errors are large, so rejected settings may still matter (Table 4, Table 6).

2. **F is a procedure, and its per-task choices were mostly close calls [numbers supported; reading a conjecture].** The in-sample ratio of per-task F is 0.816 and the nearest held-out analogue is 0.898. The interim ratio of the same kind on the 20 legacy tasks is 0.992, and per-task F against the shared recipe is 0.999 on the 18 tuned tasks (post hoc). A plausible reading is that part of the development gain came from picking winners among near-ties on noisy validation data. Repeating selection on disjoint development seeds would test it (Table 12, Table 13, Table 18, Table 24).

3. **The interim confirmation meets none of the four rules [supported, interim].** F's error is lower than retrieval-only (0.973), but the 95% interval, 0.932 to 1.015, includes 1. Joint training gives 0.996. F's error is higher than tuned CatBoost (1.139). Against TabM the ratio is 0.984, but the interval reaches 1.143, so non-inferiority is unproven. Adding QSAR-TID-11 is unlikely to change a verdict: over all assumed results for it, the smallest pooled p-value is 0.030, above the 0.0125 Holm requires. The scan uses assumed results. (Table 18, Table 25).

4. **The small gain from reuse sits in the older tasks [supported, interim; causes conjectures].** The ratio against retrieval-only is 0.944 on the 20 legacy tasks (descriptive) and 0.998 on the 9 transfer tasks, where F was untuned. There, F's retrieval error before the adapter acts is 1.013 times that of retrieval-only, near the 1.02 limit. A plausible reading is that one shared setting slightly disturbs retrieval on untuned tasks. Median calibration error is 0.034 against 0.057 before the adapter. Wrong-to-correct flips are 1.173 times correct-to-wrong, against the required 2 (Table 18, Table 20, Table 23).

5. **Neural CBR trails the best tuned standard methods, but its retrieval core beats plain nearest-neighbor search more often than not [supported; reasons conjectures].** Mean ranks over 29 tasks (1 is lowest error): CatBoost 1.90, TabM 2.62, F 3.52, MLP 3.90, retrieval-only 4.07, kNN 5.00. In the grids the core's ratio against TabM is 1.063. A plausible reading is that boosted trees ignore uninformative columns while a distance over all columns cannot. Whether the baselines' tuning budgets match F's is not established (Table 22, Table 27, Table 30).

6. **The maintenance review found the rule weak and untested [supported by review records; consequences conjectures].** On five development tasks 0.568 to 0.983 of stored cases are never retrieved (abalone 0.983, breast_cancer 0.836, wine 0.769, yacht 0.658, anneal 0.568), and removed cases carry 0.008 to 0.072 of retrieval weight. A plausible reading is that a fair test of Q here would find no gain. F has no maintenance (Table 34, Table 35).

7. **For the 29 finished tasks the result files are complete and consistent, with two limits [supported, interim].** The 16 duplicates come from reruns after an unplanned machine stop, and none conflicts. Limit one: 204 grid receipts on seeds 40 to 49 are dated before addendum 1 and 460 before addendum 4, and no file says whether the addenda cover them. Limit two: the only saved seed audit is dated 10-18, after the first confirmation results. The checks show the rows are what the runs produced, not that they are right (Table 40, Table 42).

**Table 3.** Further numbers quoted in the key findings

| Finding | Quantity | Value | Table |
| --- | --- | --- | --- |
| 1 | Budget block, classification: log ratio change on validation, then held-out data | -0.1222, then +0.0391 | Table 6 |
| 2 | Interim E2 ratio on 20 legacy tasks | 0.992 | Table 18 |
| 2 | F per task against F_global, 18 tuned tasks (post hoc) | 0.999 | Table 24 |
| 3 | E1 ratio and 95% interval | 0.973, 0.932 to 1.015 | Table 18 |
| 3 | E3b 95% interval upper end | 1.143 | Table 18 |
| 3 | QSAR-TID-11 scan: smallest pooled p, Holm threshold | 0.030, 0.0125 | Table 25 |
| 4 | E1 on legacy tasks, then on 9 transfer tasks | 0.944, 0.998 | Table 18 |
| 4 | Transfer retrieval error before adapter: ratio, limit | 1.013, 1.02 | Table 20 |
| 4 | Median calibration error, with then before adapter | 0.034, 0.057 | Table 23 |
| 4 | Flip ratio, required ratio | 1.173, 2 | Table 23 |
| 5 | Mean ranks: CatBoost, TabM, F, MLP, retrieval-only, kNN | 1.90, 2.62, 3.52, 3.90, 4.07, 5.00 | Table 22 |
| 5 | Retrieval core against TabM, transfer grid | 1.063 | Table 27 |
| 6 | Never retrieved: abalone, breast_cancer, wine, yacht, anneal | 0.983, 0.836, 0.769, 0.658, 0.568 | Table 35 |
| 6 | Retrieval weight removed by Q, smallest to largest | 0.008 to 0.072 | Table 35 |
| 7 | Receipts before addendum 1 and before addendum 4 | 204, 460 | Table 42 |

*Source: Sections 2 to 7 of this document and the tables they cite.*

### 1.2 What changed since the 10-13 report

New this week: blocks H3 and H5 finished and failed (Section 2); F was frozen (Section 3); the confirmation produced interim results (Section 4); the transfer grid reached 48 of 50 jobs (Section 5); the maintenance review and integrity checks add documents, code and checks (Sections 6 and 7).

## 2. Step D: tuning of the final configuration

> **In plain words.** Before the final test, the project could tune how the system is trained, but only on development data and under rules written in advance. Of the seven comparisons that ran, two passed their rule and five did not. A further optional block, H4, was skipped. The last two blocks, H3 and H5, finished on 10-13 after the previous report, and both failed their rule. For classification, recipes that looked better on validation data did not look better on held-out data.

**What the handoff asks for.**

Neural case-based reasoning (neural CBR) predicts the answer for a new case by retrieving similar stored cases and correcting their answer with a small neural network, the adapter. A training recipe says how retrieval and adapter are trained together. Step D asks which recipe the final test should use, under a pre-registration: a written plan fixed before the results exist. A task is a dataset together with the quantity to predict. Development seeds 20 to 25 each give one random split of a task into training, validation and test streams (the sets of cases in each part). Choices may look at validation streams only. The one exception is the test stream of a left-out development seed, which the rules below use as a declared held-out check.

Tuning is cut into blocks. Each block asks one question and starts from the current best recipe.

- H6: one recipe for all tasks, or the best per task.
- H1: whether a retrieval guard helps. It keeps a training checkpoint (a saved state of the model) only while retrieval alone has not become clearly worse.
- H2: how to rescale the classifier's scores so that its probabilities are calibrated, meaning they match how often the prediction is right. The score scale kappa multiplies the scores before they become probabilities; kappa 1 is the start. H2a tries other kappa values, H2b a different output form, H2c reruns the classification pool at kappa 10.
- H5: whether a longer training budget, a longer adapter warm-up (only the adapter trains) or a different stopping rule helps.
- H3: whether a larger or more strongly regularized adapter helps.
- H4 (optional): whether the number of retrieved neighbors matters.

At the previous report (snapshot 10-13 11:55), H6, H1 and H2 were decided, 22 of the 108 H3 jobs and 8 of the 108 H5 jobs had finished, and H4 had not started. This summary adds H3, H5 and the skip of H4. The freeze of the final configuration F is in Section 3.

**How we implemented it.**

Step D uses 18 tasks, nine for classification and nine for regression. Error is 1 minus accuracy for classification and standardized root-mean-square error for regression. Lower is better. Retrieval-only means the system without the adapter. The starting pool has 19 recipes, the shortlist of an earlier validation-only search. Its winner is the incumbent F0. Each block adds candidates that change one group of F0's settings. A feasibility check, used from H6 on, drops a recipe on a task when its retrieval before adaptation is worse than retrieval-only error by more than the 1.02 limit (plus one validation case), judged on selection seeds only.

*Leave-one-seed-out.* A selection rule must be tested on data it did not see. Hold out seed 22. The rule sees only the validation results of the other five seeds and picks one recipe per task. We then measure that recipe on the test stream of seed 22. Six seeds and 18 tasks give 108 held-out cells (54 per task type). A cell's score is the log of the ratio between the error of the picked recipe and that of a reference recipe, the original alternating default, a recipe that alternates between training retrieval and training the adapter. The geometric-mean error ratio (GMR) is the exponential of the mean log ratio; below 1 means lower error. Held-out data is another split of the same dataset, not new data.

*Acceptance rule.* The paired difference of a block is the mean over cells of the log ratio with the block's candidates in the pool minus the log ratio without them; negative means the candidates helped. The standard error (SE) is task-clustered: each task's six cells are averaged into one number, and the spread of those numbers is divided by the square root of the number of tasks. A block is adopted if its paired difference is below minus one SE, a modest bar, weaker than a conventional significance test. H2c uses non-inferiority: kappa 10 is adopted if its paired difference is at most plus one SE, which means kappa 10 is not clearly worse. That rule is recorded only in the decision notes and the comparison file, not in a dated addendum.

*Split mode for H3 and H5.* After H2c moved classification to kappa 10, the two task types no longer share one pool. H3 and H5 are judged with each type ranked in its own pool, and the held-out cells of both types pooled into one comparison with the same bar. No dated addendum records this rule. File times place the tool before the pooled H3 and H5 results existed.

*Cost.* Step D is 510 jobs (one job is one task and one seed with all of a block's recipes) and 79.1 summed job-hours.

**Experiment results.**

Each GMR in Table 4 is against the block's own reference, so compare paired differences across rows.

**Table 4.** Decision table for all step D blocks (held-out = test stream of the left-out development seed; negative paired difference = lower error with the block's candidates)

| Block | Candidates added | Jobs | Job-hours | GMR before | GMR after | Paired log ratio | Task SE | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H6 per-task against global selection | 0 | 78 | 22.9 | 0.932 | 0.891 | -0.0444 | 0.0224 | Adopted: per-task selection |
| H6 with feasibility check (limit 1.02 x retrieval-only error) | 0 | incl. above | incl. above | 0.967 | 0.895 | -0.0769 | 0.0463 | Per-task selection kept |
| H1 retrieval guard | 10 | 108 | 11.8 | 0.895 | 0.904 | +0.0104 | 0.0110 | Rejected |
| H2a score scale | 5 | 54 | 3.0 | 0.842 | 0.838 | -0.0038 | 0.0099 | Not accepted |
| H2b logit residual output | 2 | incl. above | incl. above | 0.842 | 0.842 | 9.9e-09 | 0.0315 | Not accepted |
| H2c classification pool at kappa 10 | 19 (replace) | 54 | 9.9 | 0.842 | 0.822 | -0.0231 | 0.0355 | Adopted as non-inferior |
| H5 budget and stopping (new this week) | 16 | 108 | 16.3 | 0.898 | 0.900 | +0.0023 | 0.0186 | Rejected |
| H3 adapter capacity (new this week) | 20 | 108 | 15.2 | 0.898 | 0.906 | +0.0084 | 0.0283 | Rejected |
| H4 neighbors | not run | - | - | - | - | - | - | Skipped |
| Step D total | - | 510 | 79.1 | - | - | - | - | - |

*Source: neural-cbr-ema/results/t1/p4_component_sync/tuning_20261013/ (h6_rules.json, h6_rules_feasible.json, h1_block.json, h2a_block.json, h2b_block_separate.json, h2c_compare.json, h3_block.json, h5_block.json, DECISIONS.md); neural-cbr-ema/logs/t1/p4_component_sync/tuning_20261013/*

Two comparisons meet their rule. In H6, per-task selection lowers the GMR from 0.932 to 0.891, more than one SE below zero. In H2c, kappa 10 moves the classification GMR from 0.842 to 0.822, within one SE of zero: non-inferior, not a superiority result. Five comparisons fail. H1, H3 and H5 are positive, H2a and H2b close to zero, and none is below minus its SE. H4 was not run.

*H3 and H5 by task type.* Table 5 splits the two new blocks by type. This is descriptive, not pre-registered.

**Table 5.** H3 and H5 pooled (as decided) and for each task type alone (descriptive only); 80% interval = 80% task-clustered interval of the paired difference; error change = (1 - exp(paired difference)) x 100, positive = lower error

| Block | Scope | Tasks better / worse / tied | Paired log ratio | Task SE | 80% interval | Error change (%) | Below minus 1 SE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H3 | Both types (as decided, 18 tasks) | 6 / 7 / 5 | +0.0084 | 0.0283 | -0.0293 to +0.0462 | -0.85 | no |
| H3 | Regression alone (9 tasks) | 5 / 2 / 2 | -0.0064 | 0.0037 | -0.0115 to -0.0013 | +0.64 | yes |
| H3 | Classification alone (9 tasks) | 1 / 5 / 3 | +0.0233 | 0.0578 | -0.0575 to +0.1041 | -2.36 | no |
| H5 | Both types (as decided, 18 tasks) | 6 / 6 / 6 | +0.0023 | 0.0186 | -0.0225 to +0.0272 | -0.23 | no |
| H5 | Regression alone (9 tasks) | 4 / 1 / 4 | -0.0265 | 0.0188 | -0.0528 to -3.0e-04 | +2.62 | yes |
| H5 | Classification alone (9 tasks) | 2 / 5 / 2 | +0.0311 | 0.0303 | -0.0111 to +0.0734 | -3.16 | no |

*Source: neural-cbr-ema/results/t1/p4_component_sync/tuning_20261013/h3_block.json and h5_block.json; neural-cbr-ema/docs/T1_PREREG_20261013.md (step D, item 5)*

An 80% interval is the range of effects that fits the data at that level. Each regression half alone would pass the minus-one-SE bar, and each classification half would not. The decided comparison pools both and ends positive. A tied task is one whose pick never changed in its six held-out seeds. H3 also fails its own pre-registered wording, which asks that the 80% interval exclude zero.

*Validation against held-out.* Table 6 covers only cells where a block's candidate became the pick, and compares the change on validation streams with the change on the held-out test stream.

**Table 6.** Cells in which a block candidate became the pick: mean change in log error ratio (after minus before the candidates joined the pool; negative = lower error) on the selection validation streams and on the held-out test stream; the last two columns count held-out outcomes of those cells and give an exact two-sided sign-test p-value (exploratory, uncorrected)

| Block | Changed cells | Validation change | Held-out change | Held-out better / worse / equal | Sign-test p |
| --- | --- | --- | --- | --- | --- |
| H1 (18 tasks) | 13 | -0.0012 | +0.0867 | 3 / 5 / 5 | 0.727 |
| H2a (classification) | 33 | -0.1106 | -0.0061 | 7 / 7 / 19 | 1.000 |
| H3 classification | 33 | -0.0724 | +0.0381 | 3 / 11 / 19 | 0.057 |
| H3 regression | 25 | -0.0108 | -0.0138 | 13 / 12 / 0 | 1.000 |
| H5 classification | 43 | -0.1222 | +0.0391 | 9 / 13 / 21 | 0.523 |
| H5 regression | 18 | -0.0255 | -0.0796 | 13 / 2 / 3 | 0.007 |

*Source: neural-cbr-ema/results/t1/p4_component_sync/pi20260920/ (folders tuning_h1, tuning_h2, tuning_h3, tuning_h5 and tuning_h6, suffix _20261013); neural-cbr-ema/results/t1/p4_component_sync/tuning_20261013/ (block files)*

In every classification block, validation error fell in the changed cells while held-out error barely moved or rose. In regression the held-out change has the same sign and is larger. A sign test asks whether better and worse cells are balanced; only H5 regression is one-sided.

*Why H5 classification fails.* Table 7 groups the H5 candidates by stopping rule.

**Table 7.** H5 candidates grouped by stopping rule: mean log error ratio against F0 over (candidate, task) pairs on validation and test streams (negative = better than F0), and the number of the 54 held-out cells of each task type in which a candidate of the group was picked

| Task type | Group | Candidates | Pairs | Validation log ratio | Test log ratio | Cells picked of 54 |
| --- | --- | --- | --- | --- | --- | --- |
| Classification | Alternating default, 300 epochs | 1 | 9 | +0.1865 | +0.1535 | 0 |
| Classification | Stop on validation loss | 3 | 27 | 7.7e-04 | +0.0443 | 5 |
| Classification | Stop on validation error | 4 | 36 | -0.1381 | +0.0241 | 38 |
| Regression | Alternating default, 300 epochs | 1 | 9 | +0.0357 | +0.0105 | 0 |
| Regression | Stop on validation loss | 3 | 27 | -0.0105 | -0.0097 | 18 |
| Regression | Stop on validation error | 4 | 36 | -0.0079 | -0.0078 | 0 |

*Source: neural-cbr-ema/results/t1/p4_component_sync/pi20260920/tuning_h5_20261013; neural-cbr-ema/configs/t1/tuning_h5_regression_20261013.yaml and tuning_h5_classification_20261013.yaml*

In classification, the four stop-on-error candidates are the pick in 38 of 54 held-out cells. They look better than F0 on validation and worse on test. In regression they are never picked.

**Insights.**

1. **Step D worked as a gate: two changes passed and five comparisons did not. [supported]** Only per-task selection beat its bar by more than one SE, and kappa 10 passed as non-inferior, which makes it a calibration choice, not an accuracy gain (Table 4). H3 fails under either wording of its rule (Table 5). It does not show that the other settings do not matter: the SEs are large, so a small real gain could go undetected.

2. **In the classification blocks and in H1, selection overfits the validation stream. [supported]** "Overfits" here means that a picked recipe looks better on the streams that chose it than on held-out data. In classification, validation error fell while held-out error barely moved or rose (H3 -0.0724 against +0.0381, H5 -0.1222 against +0.0391; Table 6). H2a and H1 show the same pattern. Regression shows no such gap: its held-out change has the same sign as the validation change and is at least as large. The tables do not test whether more seeds would fix it.

3. **Stopping on validation error scores a candidate on the stream that later ranks it. [supported as an association; the causal reading is a conjecture]** Such a candidate tends to win on validation by construction. The data fit this (Table 7), but no run isolates the mechanism. The stop-on-loss candidates are also worse on test (+0.0443), so the stopping rule cannot be the whole story.

4. **Regression may deserve a second look; this is a conjecture, not a decision.** Judged alone, the regression halves of H5 and H3 would pass the minus-one-SE bar (Table 5), and 13 of 18 changed H5 regression cells improved held-out against 2 worse (p 0.007; Table 6). Judging by task type was not pre-registered, the p-values are exploratory and uncorrected, and F is frozen. Reserved development seeds 26 to 29 could test it.

5. **What step D does not show. [supported by the records]** H4 yields no evidence, and its skip is not the pre-registered rule. Whether neighbor count matters is unknown. Each block varies settings around F0, so the blocks say nothing about the per-task winners that F uses. The same 108 cells served several blocks with no correction for repeated looks, which also exposes the adopted H6 result.

**Status at the snapshot.** Step D is complete by its records: 510 of 510 jobs exited with code 0, and the freeze record was written on 10-13 at 13:23. We did not inspect running processes. Open items:

- The seed-count half of H6 has no recorded decision.
- The tuning results folder holds no per-candidate validation table, although the pre-registration promises one. The numbers are in the raw run records.
- The command lines of the block runs were not stored, so pool contents and references were reconstructed from the code and decision notes. Recomputations reproduce every comparison, but they share the tool code, so they cannot prove the reconstruction.

## 3. D6: freezing the final configuration F

> **In plain words.** Before we touched any fresh evaluation data, we wrote down the training recipe, called F, that the confirmation experiment would test. F is a procedure, not one recipe: 18 tasks that had been tuned each get the recipe that did best on their own validation data, and the other 12 tasks share one global recipe, called F_global. We froze 30 configuration files. The evidence that per-task choices beat one shared recipe is moderate, and most differences between candidate recipes are small.

**What the handoff asks for.**

Neural CBR (case-based reasoning) predicts the answer for a new query by looking up similar stored cases (rows of the training data) and combining their known answers. The retrieval core learns the distance used to find similar cases. The adapter, a small neural network, corrects the answer the neighbours suggest: for a house price, retrieval finds similar sold houses and the adapter adjusts their typical price for the features that differ. The old default schedule trains the two parts in alternating blocks of two epochs (an epoch is one pass over the training cases). The joint schedule updates both on every minibatch (a small batch of cases).

Step D6 (a point of the tuning stage in Section 2) asks for three things: choose F from validation results of the development seeds 20 to 25 only, write one frozen configuration file per task, and audit the seeds so that the evaluation seeds 40 to 49 use data splits that no development seed used. A seed fixes the random split and the randomness of training. Validation data are held back from training to compare recipes.

Later, F is compared with the old default at equal budget (comparison E2 of the pre-registration, the written plan that fixes tasks, seeds and pass levels before evaluation data are used). E2 passes only if the pooled geometric-mean error ratio (GMR) is at most 0.97 and a one-sided test and all guards (pre-registered safety checks) hold. The GMR is the average, over seeds and tasks, of recipe error divided by default error, taken on a log scale. Below 1 means lower error than the default.

**How we implemented it.**

- **A closed data set.** `tools/t1_freeze.py` reads only validation rows of seeds 20 to 25 and stops on any other seed.
- **Per-task rule.** The tool ranks the 19 candidates of a task by mean validation error over six seeds: 18 synchronization conditions (ways of training retrieval and adapter together) plus frozen-core reuse. Reuse freezes the trained core and fits only the adapter. A feasibility rule first removes candidates whose error before the adapter acts exceeds 1.02 times the retrieval-only error (the core alone), plus one validation case. Without it, the adapter could hide a ruined retrieval.
- **F_global.** The tool keeps the seven conditions that are feasible on all 18 tuned tasks and takes the lowest mean per-task rank.
- **Why 12 tasks run F_global.** Car and regression wine were left out of tuning for cost. Tuning on the ten transfer tasks (Section 5) is forbidden.
- **Paired training.** Each job (one task, one seed) trains one core and continues every condition from it, so comparisons are paired.
- **Seed audit.** `tools/t1_seed_audit.py` checks that no seed has two roles and that each evaluation seed selects an outer split no other role uses. Seven small tasks get the extra seeds 50 to 69.
- **Held-out evidence.** `tools/t1_block_accept.py` selects a rule on the validation streams of five development seeds and scores it on the sixth seed's test stream. A change is adopted if its paired difference in log error ratio is below minus one standard error.

**Experiment results.**

This step ran no new training. Confirmation numbers are run counts and metadata only, and they are interim: 29 of 30 tasks were complete and QSAR-TID-11 was still running.

**Table 8.** F_global, the E2 reference and F0 (the incumbent at the start of step D), setting by setting, and the values chosen by the 17 tuned tasks whose F is a joint schedule

| setting | what it controls | F_global | E2 reference | F0 | chosen by the 17 tuned joint tasks (value: tasks) |
| --- | --- | --- | --- | --- | --- |
| training schedule | both parts updated together (joint) or in alternating blocks of 2 epochs | joint | alternating | joint | joint: 17 |
| joint objective | summed: one loss reaches both parts; separated: each part gets its own loss | summed | not used | summed | summed: 11; separated: 6 |
| adapter warm-up (epochs) | epochs at the start when only a fresh adapter trains | 5 | 0 | 5 | 5: 16; 0: 1 |
| retrieval learning-rate scale | speed of retrieval learning relative to its pre-training rate | 1 | 0.1 | 1 | 1: 16; 0.3: 1 |
| w_post | weight of the task loss after the adapter acts | 0.5 | 0.5 | 0.25 | 0.5: 12; 0.25: 3; 1: 2 |
| w_near | weight of a term keeping a query close to its retrieved cases | 0 | 0.1 | 0.1 | 0.1: 9; 0: 8 |
| v_small | weight of the penalty on large adapter corrections | 0.1 | 0.1 | 0 | 0: 10; 0.1: 7 |
| adapter learning rate | size of each adapter update | 0.003 | 0.001 | 0.003 | 0.003: 15; 0.001: 2 |
| score scale (classification) | class scores multiplied by this number inside the loss; the predicted class does not change | 10 | 10 | 1 | 10 on all 9 classification tasks |
| tuned joint F against the E2 reference (of 7 settings) | settings in which a tuned F differs from the alternating default | n/a | n/a | n/a | minimum 2, median 5, maximum 7; at least 5 of 7 on 15 of 17 |

*Source: neural-cbr-ema/configs/t1/confirm_final_20261013/ (30 configs); neural-cbr-ema/configs/t1/tuning_h6_20261013.yaml; neural-cbr-ema/docs/T1_PREREG_20261013.md*

Reading (Table 8): F_global differs from the E2 reference in six settings at once. The tuned tasks agree on the structure and differ in the objective and the weights.

**Table 9.** How the 30 tasks get their F

| category | tasks | classification | regression | note |
| --- | --- | --- | --- | --- |
| tuned, joint schedule, summed objective | 11 | 6 | 5 | |
| tuned, joint schedule, separated objective | 6 | 3 | 3 | |
| tuned, frozen-core reuse | 1 | 0 | 1 | diabetes; no condition named F |
| tuned, alternating or sequential (old family) | 0 | 0 | 0 | none of 18 |
| all tuned tasks | 18 | 9 | 9 | 10 distinct recipes |
| of which F equals F_global by own selection | 5 | 3 | 2 | iris, breast_cancer, zebra, energy_efficiency, student_performance |
| not tuned, F_global assigned: car, regression wine | 2 | 0 | 2 | left out for cost |
| transfer tasks, F_global assigned | 10 | 5 | 5 | tuning forbidden by the pre-registration |
| tasks that run the F_global hyperparameters | 17 | 8 | 9 | 5 by own selection plus 12 assigned |
| all tasks with a frozen configuration | 30 | 14 | 16 | 440 jobs (derived: 400 frozen on 10-13, 40 later); QSAR-TID-11 had 7 of 10 jobs finished |

*Source: neural-cbr-ema/results/t1/p4_component_sync/tuning_20261013/freeze_transfer_qsar_20261019.json; neural-cbr-ema/configs/t1/confirm_final_20261013/; neural-cbr-ema/logs/t1/p6_confirmation_revision/confirm_final_20261013/*

Reading (Table 9): No tuned task chose the old family. Seventeen of the 30 tasks train with the F_global hyperparameters.

**Table 10.** The frozen configuration of every task: the F recipe, the seeds it is evaluated on and the number of conditions trained per seed besides retrieval-only (cls: classification, reg: regression; w_post, w_near and v_small as defined in Table 8; the 10 transfer tasks and car and wine (reg) run F_global itself)

| task | group | schedule / objective | warm-up (epochs) | retrieval lr scale | w_post / w_near / v_small | adapter lr | seeds | trained conditions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| iris (cls) | tuned | joint, summed | 5 | 1 | 0.5 / 0 / 0.1 | 0.003 | 40-69 | 4 |
| wine (cls) | tuned | joint, summed | 5 | 1 | 0.25 / 0.1 / 0 | 0.003 | 40-69 | 4 |
| breast_cancer (cls) | tuned | joint, summed | 5 | 1 | 0.5 / 0 / 0.1 | 0.003 | 40-49 | 4 |
| balance (cls) | tuned | joint, summed | 5 | 1 | 0.25 / 0.1 / 0 | 0.003 | 40-49 | 4 |
| digits (cls) | tuned | joint, separated | 5 | 1 | 0.25 / 0.1 / 0 | 0.003 | 40-49 | 4 |
| zebra (cls) | tuned | joint, summed | 5 | 1 | 0.5 / 0 / 0.1 | 0.003 | 40-69 | 4 |
| zebra_special (cls) | tuned | joint, summed | 5 | 0.3 | 0.5 / 0.1 / 0.1 | 0.001 | 40-69 | 4 |
| anneal (cls) | tuned | joint, separated | 5 | 1 | 0.5 / 0.1 / 0 | 0.003 | 40-49 | 4 |
| credit-g (cls) | tuned | joint, separated | 5 | 1 | 0.5 / 0 / 0 | 0.003 | 40-49 | 4 |
| energy_efficiency (reg) | tuned | joint, summed | 5 | 1 | 0.5 / 0 / 0.1 | 0.003 | 40-49 | 4 |
| yacht (reg) | tuned | joint, summed | 5 | 1 | 1 / 0.1 / 0 | 0.003 | 40-69 | 4 |
| airfoil (reg) | tuned | joint, separated | 5 | 1 | 0.5 / 0.1 / 0 | 0.003 | 40-49 | 4 |
| student_performance (reg) | tuned | joint, summed | 5 | 1 | 0.5 / 0 / 0.1 | 0.003 | 40-49 | 4 |
| concrete_compressive_strength (reg) | tuned | joint, separated | 5 | 1 | 0.5 / 0.1 / 0 | 0.003 | 40-49 | 4 |
| healthcare_insurance_expenses (reg) | tuned | joint, summed | 0 | 1 | 0.5 / 0.1 / 0.1 | 0.001 | 40-49 | 4 |
| abalone (reg) | tuned | joint, summed | 5 | 1 | 1 / 0 / 0 | 0.003 | 40-49 | 4 |
| body_fat (reg) | tuned | joint, separated | 5 | 1 | 0.5 / 0 / 0 | 0.003 | 40-69 | 4 |
| diabets (reg) | tuned | frozen-core reuse | n/a | n/a | n/a | n/a | 40-69 | 3 |
| car (reg) | not tuned | joint, summed | 5 | 1 | 0.5 / 0 / 0.1 | 0.003 | 40-49 | 3 |
| wine (reg) | not tuned | joint, summed | 5 | 1 | 0.5 / 0 / 0.1 | 0.003 | 40-49 | 3 |
| splice (cls) | transfer | joint, summed | 5 | 1 | 0.5 / 0 / 0.1 | 0.003 | 40-49 | 3 |
| students_dropout_and_academic_success (cls) | transfer | joint, summed | 5 | 1 | 0.5 / 0 / 0.1 | 0.003 | 40-49 | 3 |
| in_vehicle_coupon_recommendation (cls) | transfer | joint, summed | 5 | 1 | 0.5 / 0 / 0.1 | 0.003 | 40-49 | 3 |
| churn (cls) | transfer | joint, summed | 5 | 1 | 0.5 / 0 / 0.1 | 0.003 | 40-49 | 3 |
| MIC (cls) | transfer | joint, summed | 5 | 1 | 0.5 / 0 / 0.1 | 0.003 | 40-49 | 3 |
| superconductivity (reg) | transfer | joint, summed | 5 | 1 | 0.5 / 0 / 0.1 | 0.003 | 40-49 | 3 |
| miami_housing (reg) | transfer | joint, summed | 5 | 1 | 0.5 / 0 / 0.1 | 0.003 | 40-49 | 3 |
| diamonds (reg) | transfer | joint, summed | 5 | 1 | 0.5 / 0 / 0.1 | 0.003 | 40-49 | 3 |
| physiochemical_protein (reg) | transfer | joint, summed | 5 | 1 | 0.5 / 0 / 0.1 | 0.003 | 40-49 | 3 |
| QSAR-TID-11 (reg) | transfer | joint, summed | 5 | 1 | 0.5 / 0 / 0.1 | 0.003 | 40-49 | 3 |

*Source: neural-cbr-ema/configs/t1/confirm_final_20261013/ (30 configs); neural-cbr-ema/results/t1/p4_component_sync/tuning_20261013/freeze_transfer_qsar_20261019.json*

Reading (Table 10): The 18 tuned tasks mostly differ in the loss weights. The 12 assigned tasks all carry the F_global values. Diabetes has no joint recipe because frozen-core reuse won there. Seeds 40 to 69 are the seven small tasks, which get 30 evaluation seeds instead of 10.

**Table 11.** Inventory of the frozen confirmation configurations

| item | value | note |
| --- | --- | --- |
| configuration files | 30 | one task per file |
| frozen on 10-13 | 26 | 20 older tasks (18 tuned, 2 not tuned) plus 6 transfer tasks |
| added on 10-18 | 3 | diamonds, physiochemical protein, superconductivity |
| added on 10-19 | 1 | QSAR-TID-11 |
| files with seeds 40-49 | 23 | 10 seeds each |
| files with seeds 40-69 | 7 | the small tasks: iris, wine (cls), zebra, zebra_special, body_fat, diabetes, yacht |
| training cap on transfer tasks (cases) | 3,000 | all 10 transfer configs |
| files that train 4 conditions per seed | 17 | the 17 tuned joint tasks |
| files that train 3 conditions per seed | 13 | diabetes, car, wine (reg), 10 transfer tasks |
| files with a separate F_global condition | 18 | the tuned tasks; the other 12 run F_global as F |
| files with the E2 reference condition | 30 | alternating schedule, score scale 10 on classification |
| files in which F sets epochs, patience or the stopping rule | 0 | the reference differs from the plain alternating default only by the score scale on classification |
| seed jobs (one task, one seed) | 440 | 400 + 30 + 10 |

*Source: neural-cbr-ema/configs/t1/confirm_final_20261013/; neural-cbr-ema/results/t1/p4_component_sync/tuning_20261013/ (freeze, freeze_transfer_20261018, freeze_transfer_qsar_20261019)*

Reading (Table 11): No F changes the training budget or the stopping rule. The E2 comparison therefore changes the schedule and the loss settings, not the budget.

**Table 12.** How decisive the selection was on the 18 tuned tasks (development seeds 20-25, validation data only: in-sample selection results, not performance estimates)

| measure | value | detail |
| --- | --- | --- |
| tasks on which the selected F has a lower mean validation error than F_global | 13 | equal on the other 5 |
| classification: winner leads the runner-up by exactly one validation error (summed over 6 seeds) | 5 of 9 | wine, balance, digits, anneal, credit-g |
| classification: same number of errors, so loss decides | 4 of 9 | iris, breast_cancer, zebra, zebra_special |
| tasks where the winner-minus-runner-up gap is at most one paired standard error | 17 of 18 | all but diabetes |
| breast_cancer: winner's lead in mean validation error | 9.93e-09 | winner's validation loss 0.044, runner-up's 0.041 |
| diabetes: validation error of reuse (selected) against the runner-up | 0.6857 against 0.6871 | reuse lower on 5 of 6 seeds |
| (task, condition) pairs excluded by the feasibility rule (limit: 1.02 times the retrieval-only error plus one validation case) | 39 of 324 | changes the winner on 1 task (abalone) |
| F_global choice: mean per-task rank over the 7 conditions, F_global against the runner-up | 2.944 against 3.222 | first also with reuse added (3.111) |

*Source: neural-cbr-ema/results/t1/p4_component_sync/pi20260920/ (development runs of step D); neural-cbr-ema/results/t1/p4_component_sync/tuning_20261013/freeze_transfer_qsar_20261019.json*

Reading (Table 12): Most choices are close calls.

**Table 13.** Error ratios against the alternating default (score scale 10 on classification) for the 18 tuned tasks, and held-out tests of the selection rules (paired difference is after minus before in mean log ratio, negative is better; z is that difference over its task-clustered standard error)

| row | basis | GMR (before / after) | paired difference | z | tasks better/worse | decision |
| --- | --- | --- | --- | --- | --- | --- |
| per-task F | in-sample, validation | 0.816 | n/a | n/a | n/a | n/a |
| F_global | in-sample, validation | 0.904 | n/a | n/a | n/a | n/a |
| frozen-core reuse | in-sample, validation | 0.969 | n/a | n/a | n/a | n/a |
| retrieval-only | in-sample, validation | 1.020 | n/a | n/a | n/a | n/a |
| per-task rule | held-out, 108 cells | 0.898 | n/a | n/a | n/a | n/a |
| H6: per-task against one schedule, no feasibility rule | held-out | 0.932 / 0.891 | -0.044 | -1.979 | 13/4 | adopted |
| H6: same, with feasibility rule | held-out | 0.967 / 0.895 | -0.077 | -1.661 | 12/5 | adopted |
| H1: add 10 candidates | held-out | 0.895 / 0.904 | 0.010 | 0.948 | 1/3 | rejected |
| H3: add adapter width, weight decay, dropout, learning rates | held-out | 0.898 / 0.906 | 0.008 | 0.298 | 6/7 | rejected |
| H5: add budget and stopping candidates | held-out | 0.898 / 0.900 | 0.002 | 0.124 | 6/6 | rejected |
| pre-registered pass level for E2 | confirmation, 30 tasks | at most 0.97 | n/a | n/a | n/a | not yet a result |

*Source: neural-cbr-ema/results/t1/p4_component_sync/pi20260920/ (development seeds 20-25); neural-cbr-ema/results/t1/p4_component_sync/tuning_20261013/ (h6_rules, h6_rules_feasible, h1_block, h3_block, h5_block, DECISIONS.md); neural-cbr-ema/docs/T1_PREREG_20261013.md*

Reading (Table 13): The labels H1 to H6 name the candidate blocks of step D (Section 2). In-sample values are optimistic, because the same validation streams chose F and stopped training. The per-task rule has a lower held-out ratio than one global schedule. Adding candidates (H1, H3, H5) did not lower it.

**Table 14.** Checks on the freeze, the seeds and (interim) the run metadata

| check | result | limit |
| --- | --- | --- |
| freeze records and config files | 3 records list 26, 29 and 30 configs; stored files equal the recorded hash in all of them; selection fields identical across records | the original command line is not stored |
| regeneration with the tool | 30 of 30 files byte-identical apart from the `base:` path line | identical output needs identical arguments |
| unit tests | 7 passed | 3 for the freeze tool, 4 for the seed audit |
| order of events | freeze record 10-13 13:23:59; first result shard 13:29:39; seed audit file saved 10-18 09:45:45 | file times support the order, they do not prove it |
| result rows carry the frozen fields (interim) | 2,048 of 2,050 expected rows present and consistent | QSAR-TID-11 incomplete; files read about four hours after the snapshot |
| evaluation splits | 10 evaluation seeds map to 10 distinct outer splits, shared with no other role; stored mapping equals the formula for 13 of 13 TabArena tasks | smallest TabArena test set: 299 rows |
| small tasks | 7 random-holdout tasks with 22 to 89 test rows get seeds 50-69 | smallest other task: 114 test rows |
| coverage of the stored audit | 29 of 30 configs, 0 violations | QSAR-TID-11 not covered |
| F runs whose kept epoch lies inside the adapter-only warm-up (interim) | 126 of 409 runs; credit-g 9 of 10; balance 0 of 10 | files read about four hours after the snapshot |

*Source: neural-cbr-ema/results/t1/p4_component_sync/tuning_20261013/ (freeze, freeze_transfer_20261018, freeze_transfer_qsar_20261019); neural-cbr-ema/results/t1/p6_confirmation_revision/seed_audit_confirm_20261018.json; neural-cbr-ema/results/t1/p6_confirmation_revision/pi20260920/confirm_final_20261013/; neural-cbr-ema/docs/T1_EXECUTION_LEDGER.md*

Reading (Table 14): The freeze is reproducible. The stored audit was saved five days after the first confirmation folder, so it cannot show that the audit passed before the first evaluation job.

**Insights.**

1. **F is a procedure, not one recipe (supported).** A confirmation result tests per-task selection on 18 tasks plus one assigned recipe on 12 tasks (Table 9), not one setting.

2. **Many per-task choices are near-ties (margins supported, consequence a conjecture).** The winner is within one paired standard error of the runner-up on 17 of 18 tasks (Table 12). Held-out, the per-task rule beats the global rule with a z of -1.661 under the feasibility rule (Table 13). That meets the adoption bar (z below -1) but is not a large effect. A plausible reading is that true differences are often smaller than the validation noise. F differs from F_global on 13 tuned tasks, so the confirmation can test this.

3. **E2 compares packages, and the in-sample ratio is optimistic (supported; implication a conjecture).** F_global differs from the E2 reference in six settings at once (Table 8). A positive E2 result would show that the frozen package beats the default package, not that joint training alone is the cause. The in-sample ratio of 0.816 is not a test estimate. The nearest held-out analogue is 0.898 (Table 13), for the 18 tuned tasks only. A plausible reading is that the confirmation ratio lands nearer 0.898, and nearer 1 on the 12 tasks that run F_global.

4. **Twelve of 30 tasks rely on a recipe with no per-task evidence (supported).** The held-out ratio of the global rule is 0.967 against 0.895 for per-task selection (Table 13), so the effect on assigned tasks may be smaller (conjecture). The pre-registered guard applies the same 1.02 limit on the transfer tasks (see the feasibility row of Table 12), so it can show whether retrieval degraded there.

5. **The feasibility rule makes F_global safer and weaker (supported).** With the rule, the global rule's held-out ratio rises from 0.932 to 0.967 (Table 13), that is, higher error. The table does not show whether the removed recipes would have kept retrieval intact on new tasks.

6. **In a sizeable share of runs F does not change retrieval (supported, interim).** In 126 of 409 F runs the kept epoch lies inside the adapter-only warm-up (Table 14). There F may differ little from frozen-core reuse (conjecture); the confirmation includes reuse on every task, so this can be checked.

7. **The seed audit guarantees distinct split labels, not independent data (guarantee supported, limitation a conjecture).** Every resampling re-partitions the same rows, so a plausible reading is that evaluation test rows also served as training rows under development seeds. The audit does not test row overlap.

**Status at the snapshot.** QSAR-TID-11 was frozen on 10-19, and 7 of its 10 jobs had finished (Table 9), so its rows in Table 14 are partial and the stored seed audit does not include it.

## 4. Step F: confirmation on fresh seeds (interim)

> **In plain words.** This is an interim result, not the official verdict: 29 of the 30 pre-registered tasks were complete at the snapshot, and QSAR-TID-11 was still running. Step F tests the final design F once, on fresh random splits never used for tuning, against rules written down in advance. On the 29 tasks no pre-registered endpoint is met. F's error is 2.7% lower than the same retrieval core without the adapter, but the data are also consistent with no difference. F's error is 13.9% higher than tuned CatBoost's, and F is not shown to be within 10% of tuned TabM.

**What the handoff asks for.**

Step F is the one confirmation test of the final design, called F. F is a case-based reasoning model: it predicts for a new example by retrieving the most similar stored training examples, called cases, and reading the answer off them. F retrieves with NN-kNN, a nearest-neighbour model in which a neural network learns what "similar" means. F adds an adapter, a small neural network that corrects the answer read off the retrieved cases (in case-based reasoning, this step is called reuse). After a short adapter-only warm-up, F trains retrieval and adapter together, which we call joint training. Each task asks for either a category (classification, for example which species a flower is) or a number (regression, for example a price).

F was fixed in writing before any confirmation result existed and was run once on each dataset, here called a task, on fresh random splits. A seed is one random division of a task into training, validation and test rows. The seeds are 40 to 49, and 50 to 69 as well on seven very small tasks. A written, dated plan of this kind is a pre-registration: it fixes the tasks, the measures and the success rules before the results exist. It names 30 primary tasks: 20 legacy tasks from earlier steps and 10 transfer tasks on which F's schedule and loss weights were never tuned.

The pre-registration asks four questions, one per endpoint. An endpoint is a measure with its own decision rule. Table 16 lists them, and Table 15 says what each compared arm is. A superiority test asks whether F is better than a reference. A non-inferiority test asks whether F is worse than a reference by no more than a stated margin, here 10%.

**Table 15.** What each compared arm is. The last two arms are diagnostic: no endpoint uses them.

| arm | what it is | role |
| --- | --- | --- |
| F | NN-kNN retrieval plus the adapter, trained jointly after an adapter-only warm-up. Up to 120 epochs, stopping after 30 epochs without improvement. | The design under test |
| retrieval-only | The same trained NN-kNN core with no adapter. It predicts by the weighted vote (classification) or weighted average (regression) of the retrieved cases. Every adapted arm starts from a copy of this core. | Reference for E1 |
| alternating default | The same two parts trained in turns: two epochs of retrieval updates, then two of adapter updates. It has F's epochs (120), stopping rule and score scale. | Reference for E2 |
| CatBoost | Gradient-boosted decision trees, a standard strong method for tables. Settings chosen from 32 candidates on development seeds 20 and 21, then frozen. On transfer tasks, training rows are capped at 3,000 for every method. | Reference for E3a |
| TabM | A neural network for tables that trains several small networks together. Same protocol. | Reference for E3b |
| MLP | A plain multilayer neural network. Same protocol. | Rank comparison, post hoc contrast |
| kNN | Ordinary k nearest neighbours with a fixed distance. Same protocol. | Rank comparison, post hoc contrast |
| frozen-core reuse | The retrieval core is frozen and the adapter trains alone. It is F itself on diabetes. | Diagnostic |
| F_global | One fixed joint-training setting for all tasks. It is F on car, regression wine and the transfer tasks. | Diagnostic |

*Source: neural-cbr-ema/results/t1/p6_confirmation_revision/pi20260920/confirm_final_20261013/ (run manifests); neural-cbr-ema/configs/t1/confirm_final_20261013/[task].yaml*

**Table 16.** The four pre-registered endpoints and their decision rules. Holm p is the p-value after correction for asking four questions at once. GMR is defined in the text below.

| endpoint | what it asks | reference arm | test | supported only if |
| --- | --- | --- | --- | --- |
| E1, reuse gate | Does F make fewer errors than the same core without the adapter? | retrieval-only | one-sided superiority | Holm p at most 0.05 and GMR at most 0.97 |
| E2, synchronization gate | Does joint training beat the alternating default at the same budget, without larger or less local corrections? | alternating default | one-sided superiority | Holm p at most 0.05, GMR at most 0.97, all three guards hold, and the transfer group's interval is not above 1 |
| E3a | Is F at most 10% worse than tuned CatBoost? | CatBoost | one-sided non-inferiority, margin 10% (null: GMR at least 1.10) | Holm p at most 0.05 |
| E3b | Is F at most 10% worse than tuned TabM? | TabM | one-sided non-inferiority, margin 10% (null: GMR at least 1.10) | Holm p at most 0.05 |

*Source: neural-cbr-ema/configs/t1/prereg_20261013.json; neural-cbr-ema/docs/T1_PREREG_20261013.md*

Each answer is a ratio of test errors. The test error is the share of wrong answers on unseen rows (classification) or the root mean squared error, the typical size of a miss, measured in standard deviations of the training targets (regression). For each task and seed we divide F's error by the reference arm's error on the same split. We average these ratios on a multiplicative scale (a geometric mean, in which halving the error on one task and doubling it on another cancel to 1), weighting precisely estimated tasks more. The result is the geometric mean ratio, GMR. A GMR below 1 means F makes fewer errors than the reference. A GMR of 0.973 means F's error is 2.7% lower on the typical task, and 1.139 means 13.9% higher.

Each GMR comes with a 95% interval, the range of values the data leave plausible. If it contains 1, the data cannot rule out equal errors. A one-sided p-value is the probability of a result at least this favourable to F if F were in truth no better (E1, E2) or in truth 10% worse or more (E3a, E3b). Because four questions are asked at once, the Holm correction multiplies the smallest p-value by 4, the next by 3, the next by 2 and the largest by 1.

**How we implemented it.**

*Arms and pairing.* F and the alternating default each start from a copy of the retrieval-only core of a task and seed, so the three arms are paired: same split, same seed, same starting core. E1 therefore measures the whole effect of F over its starting core.

*Which F on which task.* On 17 tuned legacy tasks F is the joint schedule chosen for that task on development seeds 20 to 25. On diabetes it is frozen-core reuse. On car, regression wine and all transfer tasks it is F_global. In all, 16 of the 29 analysed tasks run with F_global's settings. The NN-kNN core settings of every task were chosen on development seeds, using validation rows only, and are shared by retrieval-only, F and the default. Each baseline has one frozen winner per task, never re-selected on the confirmation seeds. Section 5 describes the grids.

*Pooling and guards.* Tasks are combined by random-effects meta-analysis, a weighted average that also allows true differences between tasks and widens the interval accordingly. Three guards check that E2's better score is not bought with worse retrieval. The first compares F's retrieval answer before the adapter acts (the pre-adaptation answer) with retrieval-only. The second, called locality, compares with the alternating default how far the retrieved cases lie from the query. The third, correction size, compares with the alternating default how far the adapter moves the answer.

*Secondary criteria.* Calibration is measured by the expected calibration error (ECE), the gap between stated confidence and actual hit rate, after one scaling number (a temperature) is fitted on validation rows. The criterion is at most the ECE before adaptation plus 0.02. A flip is a prediction that changes when the adapter is switched off in the same trained state, and the criterion is that wrong-to-correct flips are at least twice the correct-to-wrong flips. These tests use Benjamini-Hochberg adjustment, which limits the share of false discoveries.

*Inclusion and cross-check.* A task enters the analysis only if every neural arm and every baseline has every expected seed. Only QSAR-TID-11 fails this, and no task was dropped because of a result. Three implementations of the analysis agree on every verdict, interval end and p-value at the precision shown.

**Experiment results.**

The pre-registered primary analysis uses all 30 tasks and has no clause on interim analyses, so the 29-task analysis has no pre-registered standing. QSAR-TID-11 is excluded because F and the matched default still lacked a seed at the 04:49 read on 10-20, not because of any result.

*Coverage.*

**Table 17.** Coverage of the interim analysis. Counts for QSAR-TID-11 are post-snapshot information read at 04:49 on 10-20.

| quantity | value | detail |
| --- | --- | --- |
| pre-registered primary tasks | 30 | 20 legacy and 10 transfer |
| tasks in the interim analysis | 29 | 20 legacy and 9 transfer |
| tasks excluded | 1 | QSAR-TID-11, a transfer task: F and the alternating default lack seed 49 |
| task-seed pairs analysed | 430 | 340 legacy and 90 transfer |
| result rows used for E1 and E2 | 1,290 | 3 arms times 430 pairs |
| tasks with 10 seeds and tasks with 30 seeds | 22 and 7 | the 7 small tasks use seeds 40 to 69 |
| QSAR-TID-11 seed jobs finished | 7 at the snapshot, 9 of 10 at 04:49 | one job is one seed with all four neural arms |
| duplicate result lines, all byte-identical (abalone) | 16 | the second copy of each is ignored |
| rows that used a retrieval-guard fallback | 0 | |
| seed-audit violations | 0 | QSAR-TID-11 was not yet audited |

*Source: neural-cbr-ema/results/t1/p6_confirmation_revision/pi20260920/confirm_final_20261013/runs__[task].jsonl; neural-cbr-ema/results/t1/p6_confirmation_revision/seed_audit_confirm_20261018.json*

Of the 30 pre-registered tasks, 29 are in the analysis (Table 17). The exclusion rule is completeness of rows. It is not a result.

*Primary endpoints.*

**Table 18.** Interim primary endpoints (not the official verdict). A GMR below 1 favours F. Only the pooled rows are pre-registered decisions. The legacy-only and transfer-only rows apply the same rule to one group at a time, with Holm over that group's four p-values, and are descriptive. The last column counts the tasks on which F has lower error than the reference.

| scope | endpoint | GMR | difference in error (%) | 95% interval | one-sided p | Holm p | tasks with F lower | rule met |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled, 29 tasks | E1 | 0.973 | 2.7 lower | 0.932 to 1.015 | 0.098 | 0.294 | 19 of 29 | no |
| pooled, 29 tasks | E2 | 0.996 | 0.4 lower | 0.989 to 1.004 | 0.157 | 0.313 | 20 of 29 | no |
| pooled, 29 tasks | E3a | 1.139 | 13.9 higher | 1.003 to 1.294 | 0.711 | 0.711 | 6 of 29 | no |
| pooled, 29 tasks | E3b | 0.984 | 1.6 lower | 0.847 to 1.143 | 0.069 | 0.274 | 10 of 29 | no |
| legacy only, 20 tasks | E1 | 0.944 | 5.6 lower | 0.882 to 1.012 | 0.049 | 0.148 | 14 of 20 | no |
| legacy only, 20 tasks | E2 | 0.992 | 0.8 lower | 0.983 to 1.001 | 0.035 | 0.141 | 16 of 20 | no |
| legacy only, 20 tasks | E3a | 1.153 | 15.3 higher | 0.952 to 1.397 | 0.695 | 0.695 | 6 of 20 | no |
| legacy only, 20 tasks | E3b | 0.932 | 6.8 lower | 0.737 to 1.179 | 0.078 | 0.157 | 9 of 20 | no |
| transfer only, 9 tasks | E1 | 0.998 | 0.2 lower | 0.970 to 1.026 | 0.436 | 1.000 | 5 of 9 | no |
| transfer only, 9 tasks | E2 | 1.004 | 0.4 higher | 0.991 to 1.018 | 0.759 | 1.000 | 4 of 9 | no |
| transfer only, 9 tasks | E3a | 1.119 | 11.9 higher | 1.002 to 1.248 | 0.634 | 1.000 | 0 of 9 | no |
| transfer only, 9 tasks | E3b | 1.071 | 7.1 higher | 1.018 to 1.127 | 0.132 | 0.527 | 1 of 9 | no |

*Source: neural-cbr-ema/results/t1/p6_confirmation_revision/pi20260920/confirm_final_20261013/runs__[task].jsonl; neural-cbr-ema/results/t1/p6_confirmation_revision/baselines_eval40_20261013/, baselines_eval50_20261013/ and grid_transfer_20261013/ (baseline results)*

No pre-registered endpoint is met on the 29 tasks (Table 18). E1 and E2 fail both criteria: their Holm p-values, 0.294 and 0.313, are above 0.05 and their GMRs, 0.973 and 0.996, are above 0.97. For E3a, F's error is 13.9% higher than tuned CatBoost's and the whole interval lies above 1, though its lower end, 1.003, is barely above. For E3b the point estimate, 0.984, is slightly below 1, but the interval reaches 1.143 and the Holm p is 0.274, so non-inferiority at 10% is not shown.

The pre-registration fixes the wording for each outcome in advance. Table 19 gives the wording its rules produce for the interim result.

**Table 19.** Pre-registered wording for the interim pooled result (not the official verdict).

| endpoint | rule met | pre-registered wording |
| --- | --- | --- |
| E1 | no | no detectable gain from reuse on tuned cores |
| E2 | no | no detectable schedule effect |
| E3a | no | trails tuned CatBoost by +13.9% [95% CI +0.3%, +29.4%] |
| E3b | no | non-inferiority to tuned TabM not shown at delta = 0.1: GMR 0.984 [95% CI 0.847, 1.143], one-sided 95% upper bound 1.114 |

*Source: neural-cbr-ema/configs/t1/prereg_20261013.json; neural-cbr-ema/docs/T1_PREREG_20261013.md (fallback wording)*

*Guards.*

**Table 20.** Interim E2 guards. The value is the pre-registered statistic. The last three columns look at single tasks and are descriptive.

| scope | guard | value | limit | holds | smallest task value | largest task value | tasks above limit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | pre-adaptation GMR | 1.003 | 1.02 | yes | 0.799 | 1.213 | 8 of 29 |
| pooled | locality ratio | 1.042 | 1.10 | yes | 0.637 | 1.425 | 8 of 29 |
| pooled | correction-size ratio | 1.005 | 1.25 | yes | 0.279 | 2.912 | 8 of 29 |
| legacy only | pre-adaptation GMR | 0.999 | 1.02 | yes | 0.799 | 1.213 | 4 of 20 |
| legacy only | locality ratio | 1.053 | 1.10 | yes | 0.637 | 1.425 | 6 of 20 |
| legacy only | correction-size ratio | 1.058 | 1.25 | yes | 0.366 | 2.912 | 7 of 20 |
| transfer only | pre-adaptation GMR | 1.013 | 1.02 | yes | 1.004 | 1.051 | 4 of 9 |
| transfer only | locality ratio | 1.018 | 1.10 | yes | 0.834 | 1.292 | 2 of 9 |
| transfer only | correction-size ratio | 0.980 | 1.25 | yes | 0.279 | 1.702 | 1 of 9 |

*Source: neural-cbr-ema/results/t1/p6_confirmation_revision/pi20260920/confirm_final_20261013/runs__[task].jsonl; neural-cbr-ema/configs/t1/prereg_20261013.json*

All three guards hold in all three scopes (Table 20). Pre-adaptation means F's retrieval answer before the adapter acts, locality is the average distance from a query to the cases it retrieves, and correction size is how far the adapter moves the answer. On the transfer tasks the pre-adaptation GMR is 1.013 against the limit 1.02, and F's pre-adaptation error is above retrieval-only's on all nine of them (smallest task value 1.004). Single tasks do exceed the limits, so the guards describe the typical task only.

*Mean error per task.*

**Table 21.** Interim mean test error per task, averaged over the task's seeds, for five of the arms. Classification tasks (class.) show the share of wrong answers. Regression tasks (regr.) show the root mean squared error in training-target standard deviations. Values are not comparable across tasks. Numbers in brackets are dataset identifiers.

| task | group | type | seeds | F | retrieval-only | alternating default | CatBoost | TabM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| iris | legacy | class. | 30 | 0.052 | 0.056 | 0.051 | 0.049 | 0.053 |
| wine (classification) | legacy | class. | 30 | 0.041 | 0.046 | 0.052 | 0.025 | 0.023 |
| breast_cancer | legacy | class. | 10 | 0.036 | 0.038 | 0.039 | 0.034 | 0.036 |
| balance | legacy | class. | 10 | 0.048 | 0.060 | 0.066 | 0.091 | 0.039 |
| digits | legacy | class. | 10 | 0.022 | 0.021 | 0.022 | 0.026 | 0.019 |
| zebra | legacy | class. | 30 | 0.106 | 0.150 | 0.145 | 0.011 | 0.482 |
| zebra_special | legacy | class. | 30 | 0.286 | 0.288 | 0.278 | 0.067 | 0.499 |
| energy_efficiency | legacy | regr. | 10 | 0.053 | 0.059 | 0.055 | 0.047 | 0.052 |
| yacht | legacy | regr. | 30 | 0.084 | 0.157 | 0.108 | 0.077 | 0.067 |
| airfoil | legacy | regr. | 10 | 0.217 | 0.226 | 0.222 | 0.220 | 0.232 |
| student_performance | legacy | regr. | 10 | 0.472 | 0.471 | 0.469 | 0.447 | 0.434 |
| abalone | legacy | regr. | 10 | 0.659 | 0.660 | 0.659 | 0.671 | 0.646 |
| body_fat | legacy | regr. | 30 | 0.711 | 0.651 | 0.720 | 0.633 | 0.774 |
| diabetes | legacy | regr. | 30 | 0.726 | 0.727 | 0.727 | 0.724 | 0.699 |
| car | legacy | regr. | 10 | 0.225 | 0.230 | 0.239 | 0.166 | 0.238 |
| wine (regression) | legacy | regr. | 10 | 0.741 | 0.740 | 0.744 | 0.729 | 0.755 |
| TabArena 363614 | legacy | class. | 10 | 0.009 | 0.014 | 0.012 | 0.009 | 0.009 |
| credit-g (363626) | legacy | class. | 10 | 0.275 | 0.271 | 0.278 | 0.246 | 0.252 |
| TabArena 363625 | legacy | regr. | 10 | 0.381 | 0.402 | 0.399 | 0.275 | 0.334 |
| TabArena 363675 | legacy | regr. | 10 | 0.396 | 0.407 | 0.402 | 0.376 | 0.382 |
| splice (363702) | transfer | class. | 10 | 0.042 | 0.038 | 0.041 | 0.040 | 0.039 |
| TabArena 363704 | transfer | class. | 10 | 0.236 | 0.244 | 0.240 | 0.229 | 0.219 |
| coupon recommendation (363681) | transfer | class. | 10 | 0.307 | 0.314 | 0.302 | 0.264 | 0.271 |
| churn (363623) | transfer | class. | 10 | 0.074 | 0.070 | 0.072 | 0.045 | 0.061 |
| MIC (363711) | transfer | class. | 10 | 0.127 | 0.123 | 0.123 | 0.113 | 0.114 |
| TabArena 363705 | transfer | regr. | 10 | 0.375 | 0.385 | 0.377 | 0.351 | 0.360 |
| TabArena 363686 | transfer | regr. | 10 | 0.352 | 0.362 | 0.351 | 0.311 | 0.326 |
| TabArena 363631 | transfer | regr. | 10 | 0.172 | 0.173 | 0.173 | 0.167 | 0.188 |
| TabArena 363693 | transfer | regr. | 10 | 0.699 | 0.696 | 0.705 | 0.696 | 0.693 |

*Source: neural-cbr-ema/results/t1/p6_confirmation_revision/pi20260920/confirm_final_20261013/runs__[task].jsonl; neural-cbr-ema/results/t1/p6_confirmation_revision/baselines_eval40_20261013/, baselines_eval50_20261013/ and grid_transfer_20261013/*

F has the lowest error of the seven arms on one task, airfoil. CatBoost has the lowest error on 17 tasks and TabM on 8 (Table 21 and Table 22). On the two small classification tasks zebra and zebra_special, CatBoost's error is far below that of every other arm. On zebra, TabM's error is 0.482 and F's is 0.106. Single tasks can differ a lot from the pooled ratio.

*Ranks against the baselines.*

**Table 22.** Interim mean ranks of six methods over tasks (1 means the lowest error on a task). CatBoost, TabM, MLP and kNN are the frozen tuned winners. The Friedman test asks whether the six methods rank equally well.

| statistic | tasks | CatBoost | TabM | MLP | kNN | retrieval-only | F | Friedman p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean rank, pooled | 29 | 1.90 | 2.62 | 3.90 | 5.00 | 4.07 | 3.52 | <0.001 |
| mean rank, legacy | 20 | 2.00 | 2.90 | 3.95 | 4.55 | 4.25 | 3.35 | <0.001 |
| mean rank, transfer | 9 | 1.67 | 2.00 | 3.78 | 6.00 | 3.67 | 3.89 | <0.001 |
| tasks ranked first, pooled | 29 | 17 | 8 | 1 | 1 | 1 | 1 | n/a |

*Source: neural-cbr-ema/results/t1/p6_confirmation_revision/pi20260920/confirm_final_20261013/runs__[task].jsonl; neural-cbr-ema/results/t1/p6_confirmation_revision/baselines_eval40_20261013/, baselines_eval50_20261013/ and grid_transfer_20261013/*

Pooled over the 29 tasks, the mean ranks run CatBoost 1.90, TabM 2.62, F 3.52, MLP 3.90, retrieval-only 4.07 and kNN 5.00 (Table 22). The Friedman p-value is below 0.001, so the six methods do not rank equally. On the transfer tasks F (3.89) ranks behind MLP (3.78) and retrieval-only (3.67). CatBoost ranks first on 17 tasks and TabM on 8.

*Secondary criteria.*

**Table 23.** Interim secondary criteria on the classification tasks (14 pooled: 9 legacy and 5 transfer). ECE rows compare F's median ECE with the validation temperature to the median ECE before adaptation, and the criterion is a difference of at most +0.02. Flip rows count wrong-to-correct and correct-to-wrong changes summed over tasks and seeds, and the criterion is a ratio of at least 2. Adjusted p is Benjamini-Hochberg over nine tests.

| scope | criterion | classification tasks | F: median ECE or wrong to correct | before adaptation: median ECE or correct to wrong | difference or ratio | limit | criterion holds | adjusted p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | ECE with temperature | 14 | 0.034 | 0.057 | -0.023 | +0.02 | yes | <0.001 |
| pooled | flips | 14 | 5,681 | 4,842 | 1.173 | 2 | no | 1.000 |
| legacy only | ECE with temperature | 9 | 0.042 | 0.060 | -0.018 | +0.02 | yes | 0.004 |
| legacy only | flips | 9 | 523 | 429 | 1.219 | 2 | no | 0.866 |
| transfer only | ECE with temperature | 5 | 0.018 | 0.047 | -0.029 | +0.02 | yes | 0.047 |
| transfer only | flips | 5 | 5,158 | 4,413 | 1.169 | 2 | no | 1.000 |

*Source: neural-cbr-ema/results/t1/p6_confirmation_revision/pi20260920/confirm_final_20261013/runs__[task].jsonl*

The calibration criterion holds in all three scopes (Table 23). The criterion compares F with its own pre-adaptation output. Without the temperature, the raw adapted output is poorly calibrated on a few tasks (coupon recommendation and credit-g), and the temperature lowers that error. The flips criterion is not met: the ratio is 1.173 against the required 2.

*Post hoc and descriptive material.* The next two tables were computed after the primary result was known, carry no multiplicity control and are not findings. In the leave-one-task-out rows one task is dropped and the pooled GMR recomputed. In none of the 29 leave-one-out pools does any endpoint meet its rule.

**Table 24.** Post hoc contrasts, strata, leave-one-task-out checks and margin sensitivity. None is a pre-registered result. GMR below 1 means the first-named arm has lower error. The strata overlap and are small. Only the 10% margin is pre-registered, and it is marked as awaiting the principal investigator's confirmation in the machine-readable pre-registration.

| analysis | tasks | GMR | 95% interval | note |
| --- | --- | --- | --- | --- |
| F vs retrieval-only, 18 tuned legacy tasks (F selected per task) | 18 | 0.936 | 0.865 to 1.012 | |
| F vs retrieval-only, 11 tasks where F is F_global | 11 | 0.996 | 0.975 to 1.018 | |
| F vs retrieval-only, classification tasks | 14 | 1.005 | 0.969 to 1.044 | |
| F vs retrieval-only, regression tasks | 15 | 0.955 | 0.887 to 1.028 | |
| F vs kNN | 29 | 0.723 | 0.593 to 0.882 | |
| F vs MLP | 29 | 0.915 | 0.785 to 1.067 | |
| F vs frozen-core reuse | 29 | 0.994 | 0.986 to 1.003 | |
| alternating default vs retrieval-only | 29 | 0.993 | 0.970 to 1.017 | |
| F per task vs F_global | 18 | 0.999 | 0.997 to 1.001 | |
| F vs TabM, regression tasks | 15 | 1.030 | 0.990 to 1.071 | |
| F vs CatBoost, regression tasks | 15 | 1.085 | 1.024 to 1.149 | |
| E1 without yacht | 28 | 0.990 | n/a | highest E1 over the 29 leave-one-out pools |
| E1 without splice | 28 | 0.969 | n/a | lowest E1 |
| E3b without zebra | 28 | 1.038 | n/a | highest E3b |
| E3b without wine (classification) | 28 | 0.970 | n/a | lowest E3b |
| E3a at a 20% margin | 29 | 1.139 | 1.003 to 1.294 | Holm p 0.313, fails |
| E3b at a 20% margin | 29 | 0.984 | 0.847 to 1.143 | Holm p 0.022, passes |

*Source: neural-cbr-ema/results/t1/p6_confirmation_revision/pi20260920/confirm_final_20261013/runs__[task].jsonl; neural-cbr-ema/results/t1/p6_confirmation_revision/baselines_eval40_20261013/, baselines_eval50_20261013/ and grid_transfer_20261013/; neural-cbr-ema/configs/t1/prereg_20261013.json*

Table 24 gives three observations. First, E1 is 0.936 on the 18 tasks where F was selected per task and 0.996 on the 11 tasks where F is the global setting. These strata overlap with task size and type, so the table cannot separate selection from task effects. Second, F has lower error than plain kNN (0.723) and is not distinguishable from MLP (0.915). Third, E3b would pass at a 20% margin but E3a would not. A margin chosen after seeing the result would not be a pre-registered test, and the regression-only rows are not results.

*QSAR-TID-11 and what it would change (post-snapshot information).* QSAR-TID-11 is a transfer regression task. At the snapshot, 7 of its 10 seed jobs had finished. Table 25 adds it to the pooled analysis on the 9 paired seeds that existed at the 04:49 read on 10-20. This is not an official result. Because the partial rows are not final, we also scanned QSAR-TID-11's own GMR over 301 assumed values and added a 30th task with one assumed ratio on all four endpoints. Holm can reject the smallest of four p-values only if it is at most 0.0125, which is 0.05 divided by 4.

**Table 25.** What QSAR-TID-11's partial rows change in the pooled analysis at the 04:49 read (not an official result), and what the scan and the scenarios allow. The scan and scenarios are assumption-driven calculations, not data. The last column is the lowest Holm p-value in the scenario rows (derived).

| endpoint | QSAR's own GMR | pooled GMR, 29 tasks | pooled GMR, with QSAR | Holm p with QSAR | scanned QSAR GMR | smallest pooled p in the scan | Holm threshold | lowest Holm p, scenario ratios 0.50 to 2.00 (derived) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | 0.981 | 0.973 | 0.974 | 0.290 | 0.223 to 4.48 | 0.030 | 0.0125 | 0.121 |
| E2 | 0.999 | 0.996 | 0.996 | 0.290 | 0.223 to 4.48 | 0.037 | 0.0125 | 0.121 |
| E3a | 1.049 | 1.139 | 1.135 | 0.699 | 0.223 to 4.48 | 0.424 | 0.0125 | 0.547 |
| E3b | 1.037 | 0.984 | 0.987 | 0.262 | 0.223 to 4.48 | 0.033 | 0.0125 | 0.121 |

*Source: neural-cbr-ema/results/t1/p6_confirmation_revision/pi20260920/confirm_final_20261013/runs__tabarena_363697.jsonl (read at 04:49 on 10-20); results of the 29 tasks as in the primary-endpoint table*

On its own paired seeds, QSAR-TID-11 shows F slightly better than retrieval-only (0.981), equal to the alternating default (0.999), and worse than CatBoost (1.049) and TabM (1.037) (Table 25). Adding those rows changes the pooled GMRs in the third decimal place only. The smallest p-value anywhere in the scan is 0.030 (E1), above the threshold 0.0125, and no scenario gives a Holm p below 0.121. The scan keeps the other 29 tasks and QSAR-TID-11's partial within-task variance fixed, and the scenario ratios are assumed. Neither calculation predicts the real result.

**Insights.**

1. **No pre-registered endpoint is met on the interim data (supported).** Table 18 shows it for all four. This does not show that reuse is useless: the E1 interval, 0.932 to 1.015, includes small gains.

2. **Whatever gain from reuse exists sits in the legacy tasks and a few tasks (supported for the numbers, conjecture for the cause).** E1 is 0.944 on the legacy tasks (unadjusted p 0.049, Holm p 0.148) and 0.998 on the transfer tasks (Table 18). Without yacht the pooled E1 would be 0.990 (Table 24). A plausible reading is that choosing F on development seeds carries real signal, or that small legacy tasks respond more to adaptation. The strata overlap, so the tables cannot separate the two. A test would select F per task on some transfer tasks and run it on fresh seeds.

3. **The joint schedule makes no measurable difference in aggregate (supported).** The E2 interval, 0.989 to 1.004, is narrow and close to 1, and the guards hold (Table 18, Table 20). F is safe but not better than the alternating default at equal budget. The accurate wording is "no detectable schedule effect", not "equivalent", because no equivalence test was registered.

4. **F is not competitive with tuned CatBoost on this task set (supported).** The pooled error is 13.9% higher, the interval, 1.003 to 1.294, lies above 1, CatBoost ranks first on 17 of 29 tasks, and F beats CatBoost on none of the nine transfer tasks (Table 18, Table 22). This does not show that F could not close the gap with other tuning or a larger budget: the baselines were tuned within small grids, and whether their budgets match F's is not established (Section 9).

5. **Against TabM the picture is unresolved and depends on a few tasks (supported for the numbers, conjecture for the reason).** The pooled GMR is 0.984, but the interval reaches 1.143. TabM is better on 8 of the 9 transfer tasks (1.071), while the legacy group leans to F (0.932) (Table 18). Without zebra, a very small task where TabM does poorly, the pooled E3b GMR is 1.038 (Table 24). A plausible reading is that TabM is unstable on tiny many-class tasks. A test would run TabM with other settings on them.

6. **The transfer group shows no reuse and no schedule effect (supported), and joint training may slightly disturb retrieval there (conjecture).** F's schedule was never tuned on this group, so it is the cleaner test of generalization. Its E1 and E2 GMRs are 0.998 and 1.004, with intervals that include 1 (Table 18). F's pre-adaptation error is above retrieval-only's on all nine transfer tasks, and the pooled value, 1.013, is close to the limit 1.02 (Table 20). One global setting may shift the learned distance on tasks where it was not selected. A test would use a lower retrieval learning rate during joint training.

7. **With the validation temperature F's calibration error is below its pre-adaptation level, and the adapter's accuracy gains are modest (supported).** The ECE criterion holds in every scope, and wrong-to-correct flips are only 1.173 times the correct-to-wrong flips against the required 2 (Table 23). The criterion compares F with its own pre-adaptation output, so it does not show that F's calibration is competitive with CatBoost's or TabM's.

8. **Adding QSAR-TID-11 is very unlikely to change any verdict (supported within the scanned range and stated assumptions, conjecture beyond).** Its partial GMRs lie between 0.981 and 1.049, and no assumed QSAR result between 0.223 and 4.48 brings the smallest pooled p below 0.030 (Table 25). The official wording is expected to match the interim one, but that is an expectation, not a computed result.

**Status at the snapshot.**

QSAR-TID-11 was still running: 7 of its 10 seed jobs had finished at the snapshot and 9 at the 04:49 read. The official verdict needs all 10 seeds of the three neural arms, a run-directory audit and a seed audit like those of the other batches, and a rerun of the report tool on the 30-task evidence. Every table in this section must then be recomputed. Sections 8 and 9 list the open items and the caveats.

## 5. Harder datasets, baselines and the grids

> **In plain words.** Judging neural CBR against standard methods needs every method tuned fairly. Two large tuning runs, called grids, do this: one on 22 older tasks and one on 10 harder tasks that neural CBR is never tuned on. This week the harder-task grid went from 45 to 48 finished jobs of 50. The older grid did not move. On both grids the project's retrieval model improves on plain nearest-neighbor search more often than not and trails the best tuned baselines on most tasks. These readings are descriptive.

### 5.1 What a search grid is and what a frozen winner is

Case-based reasoning (CBR) solves a new problem by retrieving similar solved cases and adapting their answers. Neural CBR is the project's full pipeline built on that idea. Its retrieval part, NN-kNN, predicts from the nearest stored training cases under a distance that it learns. F is the final neural CBR configuration (Section 3), tested on fresh seeds in Section 4. The pre-registration is the written plan that fixed tasks, seeds and tests in advance.

We compare NN-kNN, used here without adaptation, with four standard families given the same tuning: kNN (the average or vote of the k closest training rows), CatBoost (boosted decision trees, a strong table baseline), MLP (a plain neural network) and TabM (a recent neural network for tables).

Settings that training does not learn, such as how many neighbors kNN uses, are hyperparameters. A search grid lists a few values for each, and each combination is a candidate. One task with one family is a job, and a fit is one training run of one candidate on one seed. A seed fixes the random choices of a run, including the split into training, validation and test rows. A fresh seed gives a new split of the same data. A job has four steps. Screening trains all 32 candidates on development seeds 20 and 21 and scores them on validation rows, which are held out of training. Refit retrains the best three. Freezing writes the best refit candidate, the winner, to `selection.json` before any test row is scored. Evaluation retrains the winner on each evaluation seed and scores it once on that seed's test rows. Test rows therefore never influence which settings win.

Accuracy is the share of test rows predicted correctly, and the error is 1 minus accuracy. RMSE (root mean squared error) is the typical prediction error of a regression, in the units of the target. The error ratio of NN-kNN to another family is NN-kNN's error divided by that family's error on the same seeds; below 1 favors NN-kNN. Averages of ratios over tasks are geometric means, so doubling on one task and halving on another average to 1.

**Table 26.** The search recipe shared by the two grids, and where they differ

| item | transfer grid | legacy grid |
| --- | --- | --- |
| tasks | 10 (5 classification, 5 regression) | 22 (9 classification, 13 regression) |
| rows per task, smallest to largest | 1,699 (MIC) to 53,940 (diamonds) | 110 (zebra) to 20,640 (califonia_housing) |
| jobs (task times family, 5 families) | 50 | 110 |
| candidates per family and task | 32 | 32 |
| development seeds (selection) | 20, 21 | 20, 21 |
| evaluation seeds | 40 to 49 (10) | 30 to 34 (5) |
| fits per job | 80 | 75 |
| planned fits in the whole grid | 4,000 | 8,250 |
| cap on training rows | 3,000 | none |
| tasks where the cap binds | 5 | 0 |

*Source: neural-cbr-ema/results/t1/p6_confirmation_revision/grid_transfer_20261013/study.json; neural-cbr/results/t1_pi20260920/grid_full_20261009/study.json*

The grids differ in tasks, evaluation seeds and the training cap (Table 26). Every family gets equal candidates and equal splits, but not equal compute.

### 5.2 The transfer grid: ten harder tasks

**What the handoff asks for.** Ten harder tasks from TabArena, a public benchmark of tabular data sets, chosen by data characteristics before any neural CBR result on them existed. Neural CBR is never tuned on them. Its CatBoost and TabM winners are the references for two non-inferiority tests of F. Such a test asks whether a method is worse than a reference by no more than a set margin, here an error ratio of 1.10, pooled over 30 tasks.

**How we implemented it.** One driver, `t1_grid_search.py`, runs the four steps. The fitting function sees only training and validation data, and recomputing every job's winner from the logs reproduces the stored selection for 50 of 50 jobs. All five families see identical rows on every task and seed. Test rows are the official outer test fold from OpenML. Training rows are cut to at most 3,000, which binds on five tasks (Table 26). An unplanned machine stop on 10-13 ended the first run, and the grid resumed on 10-17 and 10-18 from its saved fit receipts. A parallel runner, `t1_grid_prefill.py`, fits pending trials at the same time. Its equivalence with the sequential grid rests on a unit test that we read but did not re-run.

**Experiment results.** The previous report withheld these test metrics. F is now frozen (Section 3), so no choice can depend on them.

**Table 27.** Test metric by task and family (mean over evaluation seeds 40 to 49; accuracy, higher is better, for the first five tasks; RMSE in original target units, lower is better, for the rest) and NN-kNN's error ratio to kNN, CatBoost and TabM (below 1 favors NN-kNN). QSAR-TID-11 is interim: NN-kNN has 6 of 10 seeds, and its row compares all families on those six

| task | NN-kNN | kNN | CatBoost | MLP | TabM | ratio vs kNN | ratio vs CatBoost | ratio vs TabM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| splice | 0.9616 | 0.9044 | 0.9599 | 0.9487 | 0.9608 | 0.403 | 0.955 | 0.976 |
| students_dropout | 0.7555 | 0.7125 | 0.7707 | 0.7755 | 0.7813 | 0.851 | 1.067 | 1.118 |
| in_vehicle_coupon | 0.6864 | 0.6841 | 0.7358 | 0.7096 | 0.7286 | 0.992 | 1.186 | 1.155 |
| churn | 0.9300 | 0.8891 | 0.9545 | 0.9216 | 0.9387 | 0.628 | 1.530 | 1.144 |
| MIC | 0.8769 | 0.8649 | 0.8866 | 0.8796 | 0.8861 | 0.911 | 1.084 | 1.079 |
| superconductivity | 13.19 | 13.27 | 12.01 | 12.66 | 12.32 | 0.994 | 1.099 | 1.071 |
| miami_housing | 114,219 | 120,789 | 98,104 | 108,515 | 102,961 | 0.945 | 1.164 | 1.109 |
| diamonds | 687.0 | 1,015 | 664.5 | 939.2 | 748.4 | 0.677 | 1.034 | 0.932 |
| physiochemical_protein | 4.2754 | 4.5709 | 4.2774 | 4.4176 | 4.2562 | 0.935 | 1.000 | 1.005 |
| QSAR-TID-11 (interim) | 0.8644 | 0.9410 | 0.8146 | 0.8529 | 0.8218 | 0.927 | 1.075 | 1.061 |
| geometric mean ratio, 10 tasks | - | - | - | - | - | 0.800 | 1.110 | 1.063 |
| mean rank over 10 tasks (1 is best) | 3.2 | 5.0 | 1.6 | 3.3 | 1.9 | - | - | - |
| tasks with the best mean (of 10) | 1 | 0 | 7 | 0 | 2 | - | - | - |
| pre-registered margin for F | - | - | 1.10 | - | 1.10 | - | - | - |

*Source: neural-cbr-ema/results/t1/p6_confirmation_revision/grid_transfer_20261013/<task>/<family>/completed.json; neural-cbr-ema/results/t1/p6_confirmation_revision/grid_transfer_20261013/regression__tabarena_363697/nnknn/evaluation/1ce3c044643535ae/s40..s45/receipt.json; neural-cbr-ema/docs/T1_PREREG_20261013.md*

CatBoost has the best mean on 7 of 10 tasks, TabM on 2 and NN-kNN on 1 (Table 27). NN-kNN has a lower mean error than kNN on all ten tasks, but than CatBoost only on splice and physiochemical_protein. Its ratio to CatBoost is above the margin of 1.10 on in_vehicle_coupon (1.186), miami_housing (1.164) and churn (1.530), and its ratio to TabM on those three and on students_dropout (1.118). These ratios describe retrieval-only NN-kNN, not F.

**Table 28.** Classification tasks: quality of the predicted probabilities. Log loss (lower is better) penalizes a confident wrong answer heavily. The last three columns give the share of test rows on which the true class gets a probability of at most one in a million (1e-6), for NN-kNN and for CatBoost, and the share of NN-kNN's log loss that those rows carry (mean over seeds 40 to 49)

| task | NN-kNN log loss | CatBoost log loss | TabM log loss | NN-kNN rows with p(true class) <= 1e-6 | share of NN-kNN log loss from those rows | CatBoost rows with p(true class) <= 1e-6 |
| --- | --- | --- | --- | --- | --- | --- |
| splice | 0.2275 | 0.1225 | 0.1297 | 0.40% | 47.7% | 0.00% |
| students_dropout | 3.1850 | 0.5637 | 0.5521 | 10.54% | 91.2% | 0.00% |
| in_vehicle_coupon | 0.6049 | 0.5370 | 0.5491 | 0.03% | 1.3% | 0.00% |
| churn | 0.5573 | 0.1606 | 0.2012 | 1.49% | 71.7% | 0.00% |
| MIC | 3.0106 | 0.4717 | 0.4674 | 10.79% | 99.0% | 0.00% |

*Source: neural-cbr-ema/results/t1/p6_confirmation_revision/grid_transfer_20261013/<task>/<family>/completed.json (test_metrics.loss); neural-cbr-ema/results/t1/p6_confirmation_revision/grid_transfer_20261013/<task>/<family>/evaluation/<winner>/s4*/test_predictions.pt*

On all five classification tasks NN-kNN's log loss is higher than that of CatBoost and TabM (Table 28). On students_dropout and MIC, 10.54% and 10.79% of test rows give the true class a probability of at most one in a million, and those rows carry 91.2% and 99.0% of the log loss. CatBoost has none. NN-kNN's probabilities are a vote over its `top_k` retrieved cases, so an absent class gets exactly zero.

One limit applies to this grid. Development and evaluation splits come from the same data set, so many evaluation test rows were development training or validation rows. Only hyperparameters were chosen on them and all families are exposed alike, but a small optimistic bias cannot be excluded. Seeds 40 to 49 cover only four outer repeats, so standard errors across seeds understate the uncertainty.

**Insights.**

1. **NN-kNN improves on plain kNN but does not replace boosted trees or TabM on these tasks [supported].** Its mean error is lower than kNN's on all ten tasks, yet CatBoost or TabM has the best mean on nine (Table 27). F is not in this grid, so the table says nothing about neural CBR as a whole.
2. **The grid sets the bar for the pre-registered comparison [supported, descriptive].** Retrieval-only NN-kNN is above the margin of 1.10 against CatBoost on three tasks and against TabM on four (Table 27). Neural CBR would have to close that gap for F to be non-inferior there.
2. **CatBoost's lead is largest on churn, in_vehicle_coupon and miami_housing [supported for size, conjecture for cause].** A plausible reading is that trees ignore uninformative features while a distance over all encoded columns cannot. Down-weighting uninformative columns in NN-kNN would test it.
3. **NN-kNN's probabilities are its weak point [supported].** Nearly all its excess log loss comes from rows where the true class gets near-zero probability (Table 28). Whether calibration would remove it was not tested.

**Status at the snapshot.** 48 of 50 jobs have `completed.json`. Superconductivity NN-kNN has all ten evaluation results but no `completed.json`. QSAR-TID-11 NN-kNN has six of ten seeds, so its row and the means over 10 tasks are interim.

### 5.3 The legacy grid: 22 older tasks

**What the handoff asks for.** The full tuning run on the 22 older tasks that neural CBR was developed on. Each winner is frozen and scored on seeds 30 to 34. The pre-registration retires those seeds, because design decisions used them, so the frozen winners of the four baselines are refitted on fresh seeds (40 to 49, and 50 to 69 for seven very small tasks) with no retuning. NN-kNN is not refitted, because the confirmation has its own arm.

**How we implemented it.** Same driver and recipe as the transfer grid, without a row cap. A separate command, `evaluate`, runs only the evaluation stage of each frozen winner. All 108 re-evaluated winners equal the frozen winners, all four families got the same split on every seed, and none of the 100 splits of seeds 30 to 34 reappears among the new ones. The re-evaluation used a later copy of the code, so bitwise equality with the grid is not proven.

**Experiment results.**

**Table 29.** Job counts of both grids at the previous report (10-13 11:55) and at the snapshot, and the recorded fit time of the two unfinished legacy NN-kNN jobs (job-hours are summed fit times). A second look about four hours after the first found no file changed

| item | previous report | snapshot |
| --- | --- | --- |
| transfer grid: jobs finished (of 50) | 45 | 48 |
| legacy grid: jobs finished (of 110) | 108 | 108 |
| legacy grid: failed jobs | 0 | 0 |
| bike_sharing NN-kNN: screening fits done (of 64) | 41 | 41 |
| califonia_housing NN-kNN: screening fits done (of 64) | 27 | 28 |
| bike_sharing NN-kNN: mean job-hours per fit | - | 1.80 |
| califonia_housing NN-kNN: mean job-hours per fit | - | 3.00 |
| TabM on the same two tasks: mean job-hours per screening fit | - | 0.2306 and 0.3088 |
| the 69 NN-kNN screening fits done on the two tasks: job-hours | - | 157.65 |
| transfer grid: share of recorded job-hours taken by NN-kNN | - | 92.7% |

*Source: neural-cbr/results/t1_pi20260920/grid_full_20261009/progress.json; neural-cbr/results/t1_pi20260920/grid_full_20261009/<task>/<family>/{completed.json, screen/*/s2*/receipt.json}; neural-cbr-ema/results/t1/p6_confirmation_revision/grid_transfer_20261013/<task>/<family>/*/s*/receipt.json; NEURAL_CBR_PROGRESS_2026-10-13_EN.md*

No legacy job finished after the previous report (Table 29), and no grid process is alive. The two unfinished jobs are NN-kNN on the two largest tasks, bike_sharing and califonia_housing. Neither has a selection or a failed fit. Each has one fit that began before the unplanned machine stop of 10-13 and never finished.

**Table 30.** Retrieval-only NN-kNN against each baseline on the 20 legacy tasks where both finished, seeds 30 to 34. The ratio per task is the pre-registered definition (seed-paired). GMR is the geometric mean ratio over tasks, unweighted. The Wilcoxon p-value is a rank test of whether the 20 task ratios lean to one side of 1; it is exact, two-sided and uncorrected. All of it is descriptive

| task set | reference | tasks | NN-kNN better / tied / worse | GMR | median ratio | Wilcoxon p |
| --- | --- | --- | --- | --- | --- | --- |
| all | kNN | 20 | 12 / 0 / 8 | 0.788 | 0.933 | 0.083 |
| all | CatBoost | 20 | 7 / 0 / 13 | 1.280 | 1.097 | 0.048 |
| all | MLP | 20 | 10 / 0 / 10 | 0.898 | 0.997 | 0.729 |
| all | TabM | 20 | 7 / 0 / 13 | 0.944 | 1.034 | 0.430 |
| classification | kNN | 9 | 4 / 0 / 5 | 0.910 | 1.041 | - |
| classification | CatBoost | 9 | 4 / 0 / 5 | 1.436 | 1.174 | - |
| regression | kNN | 11 | 8 / 0 / 3 | 0.701 | 0.825 | - |
| regression | CatBoost | 11 | 3 / 0 / 8 | 1.165 | 1.087 | - |

*Source: neural-cbr/results/t1_pi20260920/grid_full_20261009/<task>/{nnknn,knn,catboost,mlp,tabm}/completed.json; neural-cbr-ema/docs/T1_PREREG_20261013.md (effect definition)*

Against kNN, NN-kNN is better on 12 of 20 tasks, with a GMR of 0.788 (Table 30). The gain is larger on regression (GMR 0.701) than on classification (GMR 0.910). Against CatBoost it is better on 7 tasks and worse on 13 (GMR 1.280, median ratio 1.097). Against MLP and TabM the split is close to even, with medians of 0.997 and 1.034.

**Table 31.** Re-evaluation of the frozen baseline winners: seeds 40 to 49 against seeds 30 to 34 (20 tasks times 4 families = 80 cells). The error ratio is the mean error on seeds 40 to 49 divided by the mean error on seeds 30 to 34; above 1 means a higher error on the fresh seeds. The Welch p-value is a standard unpaired, uncorrected test of whether two groups of numbers have the same mean

| family | cells | higher error on 40 to 49 | lower | equal | cells with Welch p < 0.05 | geometric mean of error ratio | median error ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| kNN | 20 | 11 | 8 | 1 | 1 | 1.087 | 1.009 |
| CatBoost | 20 | 10 | 8 | 2 | 1 | 1.016 | 1.007 |
| MLP | 20 | 10 | 10 | 0 | 2 | 0.958 | 1.001 |
| TabM | 20 | 10 | 10 | 0 | 3 | 0.940 | 1.003 |
| all four families | 80 | 41 | 36 | 3 | 7 | - | - |
| cells expected below p 0.05 if the two seed groups had the same mean | 80 | - | - | - | 4.0 | - | - |
| of which student_performance (regression) | 4 | 4 | 0 | 0 | 3 | - | - |

*Source: neural-cbr/results/t1_pi20260920/grid_full_20261009/<task>/<family>/completed.json; neural-cbr-ema/results/t1/p6_confirmation_revision/baselines_eval40_20261013/results.json*

The fresh seeds move errors in both directions (Table 31). Of 80 cells, 41 have a higher mean error, 36 a lower one and 3 are equal. Seven cells have a Welch p below 0.05, against about 4.0 expected by chance. Three of the seven are on student_performance, where all four families have a higher error.

**Insights.**

1. **Tuned retrieval-only NN-kNN is competitive on some tasks but is not the best family on this benchmark [supported].** Against CatBoost it is worse on 13 of 20 tasks, with a median ratio of 1.097 (Table 30). Limits: five seeds, no multiple-comparison correction, no NN-kNN result on the two largest tasks. It concerns the retrieval core, not F.
2. **NN-kNN improves on plain kNN more often than not, clearly so on regression; classification evidence is mixed [supported].** It is better on 8 of 11 regression tasks but only 4 of 9 classification tasks (Table 30). A plausible reason, untested, is that a learned distance helps most where plain kNN weights all features alike **[conjecture]**.
3. **Against MLP and TabM, NN-kNN is not separable [supported].** Medians are 0.997 and 1.034, and Wilcoxon p-values 0.729 and 0.430 (Table 30).
4. **The frozen baseline winners do not look tuned to seeds 30 to 34 or fragile on fresh splits [supported].** Errors move without a consistent direction, and 7 of 80 cells fall below p 0.05 against about 4.0 expected (Table 31). It does not show generalization to new data sets.
5. **The student_performance shift is shared by the methods [supported for direction, conjecture for cause].** All four families get a higher error there (Table 31). That harder test splits are the cause was not tested.
6. **The two missing jobs leave NN-kNN's result on the largest tasks unknown [supported; effect of finishing them is conjecture].** CatBoost is the best finished family on both, and a NN-kNN win there is unlikely but not excluded.
7. **Equal numbers of candidates are not equal cost [supported].** NN-kNN takes 92.7% of the transfer grid's recorded job-hours, and costs 1.80 and 3.00 job-hours per fit on the two largest legacy tasks against 0.2306 and 0.3088 for TabM (Table 29).

## 6. Step G preparation: review of case-base maintenance

> **In plain words.** Neural CBR (case-based reasoning) keeps a memory of stored examples and answers a new question from the most similar ones. When memory is limited, a maintenance rule decides which examples to keep. On 10-19 the researcher asked whether this rule is well designed and whether it has been tried on difficult data. The written review found the bookkeeping sound and the decision rule not. Step G, the planned test of the rule, has not been run, and the final configuration F contains no maintenance.

**What the handoff asks for.**

Step G of the 10-13 pre-registration compares three ways of shrinking the memory to half its size (K = 0.5): the score Q described below, random removal, and a rule based on each example's learned bias. The 10-19 request produced a written design review, read-only diagnostics, code fixes and a proposal for what to run next.

The pre-registration fixes the policy comparison, K = 0.5 and the pilot thresholds. It names no runner, tasks, seeds, endpoint, test direction, full-memory comparator or exposure-only control. Whether the thresholds are "proposed" or "frozen" is a decision for the principal investigator (PI).

**How we implemented it.**

The mechanism already existed before the review. The review tested it and this week's code changes touch it in four places.

- **Case and query.** A case is one stored training example with its label, a stable ID and a learned bias (a number that makes the case easier or harder to retrieve). A query is a question put to the memory, such as one new data row to classify. The model blends the answers of the most similar cases with weights that add up to 1.
- **Audit and exposure.** Every training row is used once as a query against all other cases (leave-one-out: the row's own stored copy is left out, so a case cannot vouch for itself). For each case we count the queries in which it received weight (R) and add up its weights (A). R and A together are its exposure.
- **Q, the credit score.** An audit query succeeds if the final answer is right (for regression, an error of at most 0.5 in standardized units). A success adds a case's weight to its helpful credit C, a failure to its harmful credit H. Then Q = (C + 1) / (C + H + 2), and 0.5 means no or balanced evidence. A case is evidenced only if R is at least 2 and A is at least 0.1. Every other case gets exactly 0.5. The case's own label plays no part.
- **One-shot pruning.** K is the number of cases to keep. At one declared point the pipeline keeps the two best cases of each class, fills the rest of K by rank and archives the remainder (archived cases are set aside with their exact state and can be restored). K is fixed once, so later rounds keep everything. The pipeline can also prune again later during training, which the review calls dynamic maintenance. Under Q, cases with evidence of harm go first, then unevidenced cases at the tied score of 0.5, then evidenced cases from lowest to highest Q.
- **Ties.** The default tie-break keeps the lower case ID. A new option uses a random order fixed by the run seed.

**Table 32.** Worked examples of the exposure counts and of Q

| Case | Times retrieved, R | Weight sum, A | Helpful credit, C | Harmful credit, H | Q | Evidenced (R at least 2, A at least 0.1) |
| --- | --- | --- | --- | --- | --- | --- |
| Constructed (weights 0.5, 0.3, 0.2; first two queries succeed, third fails) | 3 | 1.0 | 0.8 | 0.2 | 0.6 | yes |
| Iris, seed 22, case ID 88 (most weight) | 31 | 11.737 | 11.396 | 0.341 | 0.902 | yes |
| Iris, seed 22, case ID 1 (never retrieved) | 0 | 0.0 | 0.0 | 0.0 | 0.5 | no |

*Source: neural-cbr-ema/results/t1/p4_component_sync/pi20260920/tuning_h6_20261013/iris/tuned__alternating_default/s22/case_statistics.jsonl; constructed row computed by hand from the definition of Q.*

The first row is constructed. The others come from one Iris development run (90 stored cases), where 74 cases were never retrieved and 13 were evidenced. Keeping half (K = 45) kept all 13 evidenced cases.

Four changes followed from the review (Table 33). The default tie-break and all earlier decisions are unchanged. The 50 tests that train nothing passed. The 8 per-condition tests train a small model and were collected but not run.

**Table 33.** Code changes of the week and the checks on them

| Change | Flaw | Effect on existing runs | Tests collected | Tests run and passed |
| --- | --- | --- | --- | --- |
| Seeded tie-break option (`tie_break`: `case_id` default, or `seeded_random` from the run seed) | F3 | none: the default reproduces earlier decisions | 43 | 43 |
| Case IDs copied before the archive step compacts memory | X1 | none for models; only the logged selection-churn statistic was wrong | included in the 43 | included in the 43 |
| Per-condition maintenance settings; unknown or inert keys raise before training | F7 | all 63 sync configs (468 conditions) resolve to their earlier settings; 12 conditions maintain in training | 8 | not run |
| Diagnostics tool `tools/t1_cbm_diagnostics.py`, read-only, trains nothing | F1, F3, F4 | none: reads saved statistics of 1,482 runs | 7 | 7 |
| Total | | | 58 | 50 |

*Source: neural-cbr-ema/docs/T1_CBM_DESIGN_REVIEW_20261019.md; the changed code files and the three test files under neural-cbr-ema/tests/.*

The pack does not show which stored results used the wrong IDs before the ID fix, so the size of the X1 defect is unchecked. The per-condition change matters only for a later test of several policies inside one training schedule.

**Experiment results.**

No training run for step G was made this week. The diagnostics read final audits of unpruned models, so they show what a rule would remove, not what removal would cost in accuracy.

*The 13 flaws.*

**Table 34.** The 13 flaws of the 10-19 review (F1 to F13) and one extra defect (X1), at the snapshot

| ID | Kind | Severity | Finding in plain words | Status |
| --- | --- | --- | --- | --- |
| F1 | design | high | Q mostly measures how often a case was used, not whether it was good | open |
| F2 | evaluation | high | Never tested where it could matter: label noise, duplicates, imbalance, a binding budget, hard tasks | open |
| F3 | design | medium | Cases without evidence tie at 0.5 and ties go to the lower ID, so evictions follow row order | optional fix added; default unchanged |
| F4 | design | medium | On tuned models most cases are never retrieved, so "keep half" is not a real limit | open |
| F5 | design | medium | Scores come from the training queries the model was fitted to, so blame is near zero on easy tasks | open |
| F6 | evaluation | medium | Step G as registered omits runner, tasks, endpoint, test direction and two comparators | open |
| F7 | bug | medium | The in-training hook ignored a condition's own policy, K and epochs | fixed |
| F8 | design | medium | Each case is ranked alone; nothing checks what it adds to the set already kept | open |
| F9 | design | medium | "Dynamic" maintenance prunes once: in three synthetic runs the round at epoch 8 archived 108 of 432 cases and later rounds archived none | open |
| F10 | evaluation | low | Pruning during training versus after training compresses comparators without fine-tuning | open |
| F11 | design | low | One policy first deletes every case whose own query nobody else solves | open |
| F12 | evaluation | medium | No classical instance-selection or data-valuation baselines | open |
| F13 | bug | low | A bias option (`delta_from_init`) gives no change in a one-shot audit; no configuration uses it | open (latent) |
| X1 | bug | not rated | Kept IDs for the selection-churn statistic were read after compaction overwrote them | fixed |

*Source: neural-cbr-ema/docs/T1_CBM_DESIGN_REVIEW_20261019.md; neural-cbr/results/t1_pi20260920/continuation_20261004/sync_capacity_match_complete/alignment_sync_capacity_match/synthetic_diag/alternating_rrr_matched_K0.75/ (F9 check).*

F7 and X1 are fixed, F3 has an optional fix and the rest are open. Several flaws can be closed only by a decision of the principal investigator, listed in Table 38. The review states F2 and its verdict ('maintenance has never been run on data where it has something to fix') in broad terms. The files support a narrower statement: no aligned run compares Q with random or exposure-only removal on data with something to remove. One 432-case synthetic set was pruned with the current Q (see below), and an earlier pilot pruned it with an earlier score.

*What Q ranks and removes on the 18 development tasks.*

We rebuilt every score from saved case statistics of tuned, unpruned models (development seeds) at K = 0.5. The exposure-only control ranks cases by A alone. Spearman correlation compares two orderings (1.000 means identical order). Jaccard overlap compares two kept sets (1.000 means identical sets). "Weight archived" is the share of all weight A held by evicted cases.

**Table 35.** What Q ranks and removes at K = 0.5 on 18 development tasks (means over seeds)

| Task | Never retrieved (share) | Spearman(Q, A), evidenced cases | Jaccard, Q vs exposure-only kept set | Q evictions on evidence of harm (share) | Q evictions decided by case ID (share) | Weight archived by Q | Weight archived by exposure-only | Weight archived by random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| abalone | 0.983 | 0.645 | 0.776 | 0.007 | 0.993 | 0.072 | 0.000 | 0.500 |
| airfoil | 0.026 | 0.986 | 0.968 | 0.002 | 0.000 | 0.158 | 0.154 | 0.507 |
| balance | 0.134 | 0.991 | 0.974 | 0.003 | 0.000 | 0.124 | 0.124 | 0.507 |
| body_fat | 0.200 | 0.550 | 0.661 | 0.258 | 0.100 | 0.153 | 0.062 | 0.503 |
| breast_cancer | 0.836 | 1.000 | 0.914 | 0.000 | 1.000 | 0.008 | 0.000 | 0.622 |
| diabets | 0.166 | 0.127 | 0.473 | 0.563 | 0.415 | 0.378 | 0.084 | 0.502 |
| digits | 0.182 | 1.000 | 0.996 | 0.000 | 0.000 | 0.068 | 0.067 | 0.499 |
| energy_efficiency | 0.147 | 1.000 | 1.000 | 0.000 | 0.000 | 0.266 | 0.266 | 0.493 |
| iris | 0.447 | 0.923 | 0.929 | 0.000 | 0.500 | 0.095 | 0.088 | 0.587 |
| student_performance | 0.349 | 0.955 | 0.938 | 0.017 | 0.000 | 0.023 | 0.018 | 0.510 |
| anneal | 0.568 | 0.999 | 0.947 | 0.000 | 0.750 | 0.010 | 0.008 | 0.452 |
| concrete | 0.126 | 0.946 | 0.881 | 0.005 | 0.000 | 0.123 | 0.114 | 0.506 |
| credit-g | 0.122 | 0.899 | 0.827 | 0.042 | 0.000 | 0.101 | 0.085 | 0.476 |
| insurance | 0.185 | 0.927 | 0.846 | 0.031 | 0.000 | 0.092 | 0.081 | 0.501 |
| wine | 0.769 | 0.973 | 0.881 | 0.000 | 1.000 | 0.008 | 0.000 | 0.524 |
| yacht | 0.658 | 0.920 | 0.879 | 0.000 | 1.000 | 0.016 | 0.000 | 0.564 |
| zebra | 0.114 | 1.000 | 1.000 | 0.000 | 0.000 | 0.208 | 0.208 | 0.521 |
| zebra_special | 0.027 | 0.506 | 0.597 | 0.182 | 0.000 | 0.352 | 0.245 | 0.479 |

*Source: neural-cbr-ema/results/t1/p4_component_sync/pi20260920/tuning_h6_20261013/<task>/tuned__alternating_default/s<seed>/case_statistics.jsonl; neural-cbr-ema/results/t1/p1_supervised_retention/pi20260920/cbm_diagnostics_20261019/tuning_h6_dev_K0.5.json.*

The table shows four things.

First, on five tasks most stored cases are never retrieved by any training query: abalone 0.983, breast_cancer 0.836, wine 0.769, yacht 0.658 and anneal 0.568. There Q archives only 0.008 to 0.072 of the weight.

Second, Q ranks cases mostly by usage: its Spearman correlation with A is high on most tasks, lowest on diabets 0.127 and zebra_special 0.506.

Third, Q and the exposure-only control keep similar sets, except on diabets 0.473, zebra_special 0.597 and body_fat 0.661, where evictions on evidence of harm are 0.563, 0.182 and 0.258.

Fourth, the lower-ID tie-break decides most of Q's evictions on five tasks (0.750 to 1.000) and none on ten. Random removal archives 0.452 to 0.622 of the weight.

We did not diagnose F.

*Data with something to remove.*

The nearest available data with something to remove is a 432-case synthetic set with 54 corrupted rows (wrong labels), 60 duplicates and 12 rare-class rows. Its case IDs follow class order. We replayed nine pruning events of an earlier experiment that cut it to 324 cases with Q and the lower-ID rule.

**Table 36.** Replay of nine pruning events on the 432-case synthetic set

| Schedule | Seed | Evictions tied at 0.5 (of 108) | Last class kept, lower-ID rule | Last class kept, random ties (mean of 2,000 draws) | Corrupted rows kept, lower-ID rule (random removal keeps 0.75) | Corrupted rows kept, random ties (mean of 2,000 draws) |
| --- | --- | --- | --- | --- | --- | --- |
| independent | 5 | 106 | 0.589 | 0.789 | 0.667 | 0.613 |
| independent | 6 | 105 | 0.594 | 0.783 | 0.667 | 0.650 |
| independent | 7 | 101 | 0.614 | 0.746 | 0.556 | 0.526 |
| alternating_rr | 5 | 105 | 0.570 | 0.786 | 0.741 | 0.655 |
| alternating_rr | 6 | 107 | 0.580 | 0.761 | 0.741 | 0.680 |
| alternating_rr | 7 | 100 | 0.607 | 0.743 | 0.556 | 0.534 |
| alternating_rrr | 5 | 104 | 0.563 | 0.781 | 0.741 | 0.638 |
| alternating_rrr | 6 | 106 | 0.587 | 0.770 | 0.722 | 0.672 |
| alternating_rrr | 7 | 98 | 0.607 | 0.752 | 0.537 | 0.490 |

*Source: neural-cbr/results/t1_pi20260920/continuation_20261004/sync_capacity_match_complete/alignment_sync_capacity_match/synthetic_diag/<schedule>/s<seed>/case_maintenance.jsonl; the same folder's data_snapshot_s<seed>.pt (corruption flags).*

In every event nearly all evictions were exact ties, and replaying the logged scores with the lower-ID rule reproduces every logged decision. Because IDs follow class order, that rule puts the loss on the last class: it keeps 0.563 to 0.614 of its cases, against 0.743 to 0.789 under random ties. Q kept 0.537 to 0.741 of the corrupted rows, against 0.75 for random removal.

An earlier pilot (09-29, 10 seeds) pruned the same set at two memory sizes with an earlier, superseded score, labeled `provenance_only` in its files. The files do not record its definition. The near-zero kept share of corrupted rows fits a score based on the stored label, so that is an inference.

**Table 37.** Superseded pilot (09-29) on the 432-case synthetic set, 10 seeds, selected policies

| Policy | Capacity (cases) | Corrupted rows kept (share) | Duplicates kept (share) | Rare-class rows kept (share) | Test accuracy |
| --- | --- | --- | --- | --- | --- |
| full_memory | 432 | 1.000 | 1.000 | 1.000 | 0.940 |
| random | 130 | 0.298 | 0.313 | 0.225 | 0.956 |
| provenance_only (earlier score) | 130 | 0.000 | 0.490 | 0.642 | 0.904 |
| bias_current | 130 | 0.070 | 0.402 | 0.000 | 0.944 |
| full_memory | 432 | 1.000 | 1.000 | 1.000 | 0.942 |
| random | 216 | 0.506 | 0.512 | 0.433 | 0.833 |
| provenance_only (earlier score) | 216 | 0.006 | 0.747 | 0.842 | 0.865 |
| bias_current | 216 | 0.161 | 0.643 | 0.000 | 0.915 |

*Source: neural-cbr-ema/results/t1/p1_supervised_retention/pilot_20260929/p1_retention_synthetic/runs__synthetic_diag.jsonl.*

The earlier score removed almost all corrupted rows, yet its pruned memories scored below full memory at both sizes. Random removal scored above full memory at 130 cases and well below it at 216. These are descriptive results from one pilot with no significance test, and the pilot did not use the current Q. The review pack did not read this pilot's results on real tasks.

**Insights.**

1. **Why "removal gives no consistent gain" is the expected outcome of the current set-up** (mechanism supported; consequence a conjecture). Q orders cases mostly by usage, so Q-based pruning keeps used cases and drops unused ones (Table 35). On five tasks half the memory is not a real budget. A plausible reading is that a fair test of Q here would find no gain. That is an inference. In the aligned runs Q was never compared with an exposure-only control, nor with random removal on data with something to remove.

2. **The results do not show that maintenance cannot help or that Q is useless** (supported). Harm-driven evictions are large on body_fat, diabets and zebra_special (Table 35), and in all nine synthetic events Q kept fewer corrupted rows than random removal would, 0.537 to 0.741 against 0.75 (Table 36). The margin is small in some events.

3. **"Q measures exposure, not quality" needs one qualification** (supported; one superseded pilot). It holds on the clean development tasks. On corrupted synthetic data Q is partly outcome-sensitive, but weakly next to the label-based score, whose pruned memories still scored below full memory (Table 36, Table 37). A new study should therefore measure test error, not only noise removal.

4. **"Q beats random" would be weak evidence** (supported for the confound; a conjecture for the consequence). Random removal archives far more weight than Q (Table 35), so a win would mostly show that dropping unused cases beats dropping used ones (not that Q finds harmful cases). That the registered test will be uninformative is a prediction.

5. **The tie-break matters most where it is least visible** (supported). On five tasks ties decide most evictions, so arbitrary IDs set the kept set (Table 35). On class-ordered data the lower-ID default hurts the last class (Table 36). No table measures an accuracy effect.

6. **What experiment would settle it** (a design; the reading rule is a conjecture). Run a study on data with known noise, duplicates and a tight budget. If exposure-only removal and random match Q while a label-aware credit separates the noise, Q's credit assignment is the cause. If Q beats both, the flaw is overstated. If nothing separates the arms, the supportable claim would be usage-based compression, not better accuracy.

7. **Questions the diagnostics cannot answer** (conjectures). "Never retrieved" is measured on training queries; a cross-fitted audit (scoring each case with queries it did not help train) would test whether such cases serve test queries. Hub cases may dominate neighborhoods on abalone, but no diagnostic tests it. The picture may not carry over to F, whose per-task settings were not diagnosed.

8. **The confirmation is unaffected** (supported). None of the 30 confirmation configurations of F mentions maintenance. The interim confirmation analysis in Section 4 does not depend on this section.

**Status at the snapshot.**

Step G has not been run. No step-G configuration, runner, addendum or result exists, and the final configuration F contains no maintenance. Table 38 lists the steps the review proposes and who decides each. The split between PI decisions and engineering is the reviewer's assessment from the cited documents, not a statement of those documents.

**Table 38.** Proposed next steps at the snapshot (none started)

| Step | Content | Who decides | Status |
| --- | --- | --- | --- |
| 1a. Step G addendum | A dated document, written before any step-G result: the runner, tasks and seeds, endpoint, test direction, secondary outputs (full-memory comparison, exposure-only control, share of evictions decided by ties, kept-set overlap), and a clause that "Q beats random" alone is not evidence | PI | proposed, not written |
| 1b. Step G runner | Compress F's selected checkpoint with equal continued training for every arm; existing code covers only part of this | engineering, after 1a | not started |
| 2. Exploratory study | Development seeds, data with known noise, duplicates and imbalance, memory sizes from 0.05 to 0.75 of the cases, controls and classical baselines | PI on scope and on how credit is defined; engineering for the tools | proposed, not started |
| 3. Dynamic maintenance | Later insertion and re-admission of cases; only after step 2 shows which credit signal works | PI, then engineering | proposed, deferred |
| 4. Engineering fixes | Stored-label Q baseline, combined coverage and redundancy selection, cross-fitted audit, classical baselines, fix or retire `delta_from_init`, fine-tuning for post-hoc comparators | engineering, except where the plan leaves a formula open or Q itself changes | open |

*Source: neural-cbr-ema/docs/T1_CBM_DESIGN_REVIEW_20261019.md; neural-cbr-ema/docs/T1_PREREG_20261013.md.*

Decisions that belong to the PI: the definition of Q (final-outcome credit, label-aware or counterfactual), whether step G still runs as registered and what its addendum says, whether the step-G thresholds are proposed or frozen, the default tie-break for step G, the scope of the exploratory study, the formulas the plan leaves open (combined coverage and redundancy, the rule for cases nobody else solves), and when dynamic maintenance starts. Two review statements rest on working notes outside the project files and are unchecked: that an independent reviewer tried to refute each flaw, and the review's own synthetic-set numbers. The aligned replay in Table 36 agrees with the review in direction but gives different values, and the reason is not established.

## 7. Execution integrity of the confirmation run

> **In plain words.** The confirmation run is a one-shot test: we froze the final configuration and ran it once on fresh random seeds. This section asks whether the result files are complete, unaltered and produced by known code, after an unplanned machine stop, one lost result line and three slightly different code copies. For the 29 finished tasks every check passed, but the thirtieth, QSAR-TID-11, was still running, so every statement here is interim. The checks show that the lines are the ones the runs produced, not that the numbers are right.

**What the handoff asks for.**

The pre-registration is a written plan, fixed before the test results were seen. One rule forbids rerunning the evaluation seeds, except for logged crash reruns that give bitwise-identical results (the same bytes, digit for digit). A seed sets the random choices of a run. The evaluation seeds are 40 to 49, plus 50 to 69 for seven very small datasets. An addendum is a dated change appended to the plan. The handoff asks whether the rows are complete, unaltered and produced by known code, and when each addendum was written relative to the first evaluation-seed results.

**How we implemented it.**

A job is one dataset with one seed, run under 4 or 5 conditions (variants of the method). Each finished condition is a run with its own folder, and it adds one row (one line of text) to the dataset's results file, a shard. A receipt is a small file written when a job ends, with the exit code (0 means success). A pinned code copy is a frozen copy of the project code made before a batch starts. A code fingerprint is a 12-character code computed from every code, configuration and test file of a copy. The transfer grid is the earlier search for settings of the comparison methods and of NN-kNN, the project's neural nearest-neighbor model (Section 5).

After a restart, the launcher reuses every job whose receipt shows exit code 0 and reruns all others from the first step, appending their rows again. We compared rerun rows with the first attempt.

The first two code copies appended rows with a plain file append, which assumed that jobs of different datasets never share a file. That broke when several seeds of one dataset ran at once. Picture a logbook with several clerks: if two open it to the same empty line at the same moment, the second entry covers the first. The analogy stops where a clerk would notice a covered entry and the program does not. The third code copy takes an exclusive lock, so the second writer waits.

The audit tool `tools/t1_confirm_audit.py` re-creates each row from the run's own files and requires the same line as in the shard. It repairs a missing row only if the run provably finished, and it never rewrites or removes a line. We re-derived the counts from the raw files and ran no audit or training.

**Experiment results.**

**Table 39.** Effect of the unplanned machine stop of 10-13 and how recovery treated each job (26-config batch and baseline re-evaluation)

| Category | Jobs | Outcome |
| --- | --- | --- |
| Finished before the stop, reused at the resume | 55 | Not rerun; rows kept |
| In flight, rows stored (abalone seeds 45-49) | 5 | 16 first-attempt rows rewritten by the rerun: 16 byte-identical, 0 conflicting |
| In flight, no stored output (coupon seeds 40-42, inferred) | 3 | No row before the stop, so no duplicate; identity inferred from queue order |
| Not yet started at the stop | 337 | Ordinary first runs |
| Total | 400 | One exit-0 receipt each |
| Baseline re-evaluation, seeds 40-49 | 80 | Finished just before the stop; not rerun |
| Baseline re-evaluation, seeds 50-69 | 28 | First execution, after recovery |
| Quiet period | 4 d 8 h 59 min | No experiment output between the stop and the resume on 10-17 |

*Source: neural-cbr-ema/logs/t1/p6_confirmation_revision/confirm_launcher_resume_20261017.log; neural-cbr-ema/logs/t1/p6_confirmation_revision/confirm_final_20261013/ (job receipts); neural-cbr-ema/results/t1/p6_confirmation_revision/pi20260920/confirm_final_20261013/runs__abalone.jsonl; neural-cbr-ema/results/t1/p6_confirmation_revision/baselines_eval40_20261013/, baselines_eval50_20261013/*

Table 39 shows that no finished result was discarded. The 5 abalone jobs among the 8 interrupted ones reproduced their 16 stored rows byte for byte. Row identity is the only evidence, because the first-attempt folders were overwritten. In our reading, the rule against reruns stops anyone from redrawing seeds until the result looks better, and a rerun that replaces nothing with a different result does not do that.

**Table 40.** Integrity checks and outcomes at the 01:00 snapshot of 10-20 (interim)

| Check | Result | Outcome |
| --- | --- | --- |
| Jobs and receipts | Planned 440; exit code 0: 437; failed 0; no receipt yet 3 (QSAR seeds 44, 48, 49) | Pass; QSAR running |
| Row lines and duplicates | 2,059 lines; 2,043 unique runs; 16 duplicate lines, all abalone, byte-identical; 0 conflicting | Pass |
| Line format | 0 malformed or blank lines; 0 unterminated tails | Pass |
| Rows versus run folders | Rows without a folder 0; folders without a row 0 (after the yacht repair) | Pass |
| Rows versus "[done]" log lines | "[done]" without a row 0; row without "[done]" 0 | Pass |
| Missing pairs of condition and seed in the 29 finished tasks | 0 | Pass |
| Audit 1 (10-18), 26 configs | Run folders 1,890; rows 1,906; rebuilt exactly 1,906 of 1,906 | Pass |
| Audit 2 (10-19), 29 tasks | Run folders 2,010; rows 2,026; rebuilt exactly 2,026 of 2,026 | Pass |
| Audit 3, QSAR-TID-11 | Not yet run; 7 of 10 jobs finished; 33 rows | Pending |
| Lost row: yacht, seed 64, condition `tuned__alternating_default_matched` | Run finished (receipt exit 0, "[done]" line); row of 1,828 characters missing; colliding yacht seed-63 row of 1,910 characters stayed; repaired 10-18 by rebuilding from the run's files | Repaired, not rerun |
| Other yacht runs ending within 50 ms of another | 11 pairs; none lost a row | Informational |
| Code fingerprint per task | 30 of 30 tasks have one fingerprint; 3 distinct fingerprints overall | Pass |
| Determinism flag in manifests | False in every manifest (it records whether the training library's strict deterministic mode was on) | Informational |
| Seed audit (10-18), 29 configs | Violations 0; QSAR config not included; no comparison of runner and grid splits | Pass for 29 configs |
| Baseline re-evaluation | 80 of 80 and 28 of 28 jobs; winners equal the frozen grid selection in 80 of 80 and 28 of 28 | Pass |
| Grid winner file before evaluation trials | `selection.json` precedes the first evaluation receipt in 50 of 50 jobs | Pass |
| Evaluation seeds in other result files | 31,257 rows; rows with seed 40 to 69: 0 | Pass |
| Ledger statements checked against raw files | 22 checked; 15 match; 1 mismatch (stop time of the first QSAR launch); 1 not reproduced (a count of 31,533 rows); 5 other | No difference affects a stored row |

*Source: neural-cbr-ema/results/t1/p6_confirmation_revision/pi20260920/confirm_final_20261013/runs__*.jsonl; neural-cbr-ema/logs/t1/p6_confirmation_revision/confirm_final_20261013/ (job receipts and logs); neural-cbr-ema/results/t1/p6_confirmation_revision/confirm_audit_repair_20261018.json, confirm_audit_transfer3_20261019.json, seed_audit_confirm_20261018.json; neural-cbr-ema/docs/T1_EXECUTION_LEDGER.md*

Three comparisons use different files and agree: rows against run folders, rows against "[done]" lines, and the audit's exact rebuild of every row (Table 40). Damage from the stop, the write race or the reruns would be expected to break one of them, and none does in the 29 finished tasks. A plausible reading is that a later write covered the lost yacht row; file times are not append times, so this is a conjecture. The audits cannot detect a wrong run with consistent files.

**Table 41.** Pinned code copies behind the confirmation rows and the transfer grid (QSAR rows interim)

| Copy | Used for | Fingerprint | Differences from copy 1 | `model/` files that differ | Locked append |
| --- | --- | --- | --- | --- | --- |
| Copy 1 | 26-config confirmation (1,890 runs); baseline re-evaluations | 3825a422c093 | Reference | Reference | No |
| Copy 2 | Transfer-3 confirmation (120 runs) | c614a768d682 | 5 files added (3 task configs, a grid tool and its test) | 0 | No |
| Copy 3 | QSAR confirmation (10 jobs) | f43f59a46a49 | 9 files added, 4 changed (`tools/t1_run.py`, `tools/t1_parallel.py`, `tools/t1_confirm_report.py`, one test) | 0 | Yes |
| Grid copy | Transfer-task grid (50 jobs) | 14ef677fdace | Older code: 13 files changed, 46 only in copy 1 | 7 | No |

*Source: neural-cbr-ema/results/t1/p6_confirmation_revision/grid_transfer_20261013/study.json; run_manifest.json files of the confirmation runs; read-only file hashing and comparison of the four pinned copies*

The `model/` files are identical in the three confirmation copies, and `tools/t1_run.py` imports only from `model/`. The grid copy has 7 different `model/` files, so grid and confirmation NN-kNN results come from different model files, with an effect we did not assess. Equal numbers across copies are untested.

**Table 42.** Pre-registration records against the transfer grid's evaluation-seed receipts (seeds 40-49, dated by file time)

| Version or event | Record time (local) | Grid receipts dated earlier | of which NN-kNN |
| --- | --- | --- | --- |
| Original pre-registration | 10-12 21:41:35 | 0 | 0 |
| First grid evaluation receipt (kNN, splice) | 10-12 21:42:02 | - | - |
| Addendum 1 (decision rule for two analyses) | 10-12 23:24:58 | 204 | 4 |
| Addendum 2 (tuning on 18 of 20 legacy tasks) | 10-12 23:30:48 | 210 | 10 |
| Addendum 3 (selection and feasibility rules) | 10-13 06:44:44 | 340 | 20 |
| Addendum 4 (freeze of F) | 10-13 13:25:18 | 460 | 60 |
| First neural-CBR confirmation job start | 10-13 13:25:41 | - | - |

*Source: neural-cbr-ema/docs/T1_PREREG_20261013.md and configs/t1/prereg_20261013.json (revision times); neural-cbr-ema/results/t1/p6_confirmation_revision/grid_transfer_20261013/*/*/evaluation/*/s*/receipt.json (file times)*

Table 42 reports the dating without judgment. The addenda say "before any evaluation-seed result" or "run". The first grid receipt is dated 27 seconds after the original record. No confirmation process and no baseline re-evaluation file is dated before addendum 4. The documents label every addendum 10-13, although three fall on 10-12 local time.

**Insights.**

1. **The rows of the 29 finished tasks are complete and consistent with the runs that produced them (supported by Table 40).** This does not show that the numbers are right.
2. **The unplanned stop did not change any stored result row (supported by Table 39).** The 16 rows stored by interrupted jobs were reproduced byte for byte, which the pre-registration allows. The other rows of those jobs exist only from the rerun, so nothing can be compared there.
3. **Identical rerun rows suggest that training repeats exactly on this setup (conjecture).** All 16 reruns match although the determinism flag is false (Table 40). The base is 16 rows from one dataset.
4. **The lost yacht row was a bookkeeping failure, not a failed run, and its repair is not a rerun of an evaluation seed (supported by Table 40).** That the repaired line equals the lost line is an inference.
5. **Mixed fingerprints are a labelling difference, not evidence of different model code (supported by file comparison, Table 41; equal numbers across copies are a conjecture).** Running one job from two copies would test it.
6. **Part of the transfer-grid evaluation-seed output predates all four addendum records (supported by Table 42, as a statement about file and record times).** No file says whether the addenda's wording covers these receipts, and that no addendum was informed by them is a conjecture. The winner files that precede every evaluation receipt (Table 40) show only that the grid's own selection did not use them. A disclosure from the principal investigator would settle it.
7. **The pre-registered rule "the seed audit must pass before any evaluation run" cannot be shown from saved files (supported by Table 40).** The only saved record is dated 10-18 and omits the QSAR config. An earlier unsaved run is possible (conjecture). A saved re-run on all 30 configs would remove the doubt.

**Status at the snapshot.** QSAR-TID-11 had 7 of 10 jobs finished and 33 of 40 rows, and its audit had not been run. Every statement above covers 29 of 30 tasks and is interim.

## 8. Status of the pre-registered steps and open items

> **In plain words.** At the snapshot (10-20, about 01:00), tuning is finished and the final configuration F is frozen. F has run on 437 of 440 jobs, so the confirmation is interim: 29 of 30 tasks are complete, and on them no pre-registered endpoint is met. The test of case-base maintenance (step G) has not started. It waits for a written plan and for decisions that only the principal investigator (PI) can make.

**Terms used in Sections 8 to 10.**

- Pre-registration: a written plan, fixed before the results exist, that names tasks, seeds, tests and pass levels. An addendum is a dated change to it.
- Snapshot: the moment the project files were read, 10-20 about 01:00. Interim: computed from results available then, so not the official verdict.
- F: the final training recipe of neural case-based reasoning, a method that predicts for a new case by retrieving similar stored cases and correcting their answer with a small neural network. F_global is the one fixed recipe shared by tasks without their own tuning.
- Task, seed, job: a task is a dataset with a quantity to predict. A seed fixes one random split into training, validation and test rows. A job is one task run with one seed.
- Endpoints E1, E2, E3a, E3b: the four pre-registered questions. Does F make fewer errors than the same retrieval core without the adapter (E1) and than the alternating training default (E2)? Is F at most 10% worse than tuned CatBoost (E3a) and tuned TabM (E3b)? The geometric mean ratio (GMR) is F's error divided by a reference's error, averaged on a log scale. Below 1 favors F.
- Case-base maintenance: a rule that decides which stored cases to keep when memory is limited. Step G is its planned test.
- Grid: a search over a fixed list of settings for one method on one task.

### 8.1 Steps of the pre-registered phase

The project files name steps B, D, D6, D8, F and G, and a secondary group I. No file defines a step A, C or E, so this summary cannot report on them. Table 43 gives the state of each named step and of two grids. "Not running" means that no process was alive and no output changed. It does not say whether the stop was deliberate.

**Table 43.** State of each named step at the snapshot (10-20, about 01:00)

| Step | State | Evidence at the snapshot | Where |
| --- | --- | --- | --- |
| B, tooling | Done | The tuning note lists 300+ tests and four independent reviews; we did not re-run them | - |
| D, tuning (development seeds 20 to 25) | Done | 7 comparisons: 2 adopted, 5 not; H4 (number of neighbors) skipped; 510 of 510 jobs exited 0; 79.1 job-hours | Section 2 |
| D6, freeze | Done | 30 configuration files: 18 tuned tasks with their own F, 12 tasks with F_global; no test stream used | Section 3 |
| Seed audit | Done for 29 of 30 configs | 0 violations; the record is dated after the first confirmation job started; QSAR-TID-11 not covered | Sections 3 and 7 |
| F, runs (seeds 40 to 49) | Running | 437 of 440 jobs exited 0 at 01:00 (439 at 05:14); QSAR-TID-11 had 7 of 10 jobs at 01:00 | Sections 4 and 7 |
| F, report | Not started | No report output at the 05:14 read. Interim 29-task GMRs: E1 0.973, E2 0.996, E3a 1.139, E3b 0.984; no endpoint met | Section 4 |
| G, retention (K = 0.5) | Not started | No configuration, runner, addendum or result; a dated addendum is needed first | Section 6 |
| D8 and I, large-training-set group | No-go | Gate D8 failed its run-reliability criterion; nothing from the group enters the primary analysis | - |
| Transfer grid (no letter) | Running | 48 of 50 jobs finished (45 at the previous report); two NN-kNN jobs lack a summary | Section 5 |
| Legacy grid (no letter) | Not running | 108 of 110 jobs, unchanged since 10-13 13:04; plan and radius stages never ran | Section 5 |
| A, C, E | Cannot report | Not defined in any project file | - |

*Source: neural-cbr-ema/docs/T1_TUNING_CONFIRMATION_20261013.md; neural-cbr-ema/docs/T1_PREREG_20261013.md; neural-cbr-ema/results/t1/p4_component_sync/tuning_20261013/; neural-cbr-ema/results/t1/p6_confirmation_revision/; neural-cbr/results/t1_pi20260920/grid_full_20261009/*

Three steps are finished, the confirmation runs are in their last stretch, and the confirmation report has not been run. The interim analysis of Section 4 stands in for it. It has no pre-registered standing, because the pre-registered analysis names 30 tasks and has no clause for interim looks.

The runs wait for QSAR-TID-11. Before the report, three things remain: an audit of its run folders, a seed audit that covers its configuration, and a run of the report tool on all 30 tasks. The tool refuses results from more than one code fingerprint (a short code identifying the code files of a batch) unless told to allow mixed code, and the confirmation has three fingerprints (Section 7).

Of the items on the previous report's to-do list, the tuning items are done or skipped, and the confirmation items are running or waiting. The maintenance, legacy-grid and scientific-threshold items have not started or have stopped. The wider items (a human study, real-data categorical groups, benchmark reconstruction) show no recorded work. Other open items:

- The seed-count half of H6 has no recorded decision (Section 2).
- The tuning folder holds no per-candidate validation table, although the pre-registration promises one, and the command lines of the tuning runs were not stored (Section 2).
- One adapted run on a transfer task (superconductivity, development seed 20) predates the freeze. No document says whether it influenced a decision.

**After the snapshot.** At about 10:00 on 10-20 the confirmation results folder held a report folder and a second audit file, both dated 08:45. They were written after the evidence for this summary was read. No number here comes from them, and where this summary says the report had not been run, it means at the snapshot.

### 8.2 Decisions that need the PI

No file we read records a PI decision on any item in Table 44. A decision may have been made outside the project folders. For step G the split between PI decisions and engineering follows the design review. Elsewhere it is our reading.

**Table 44.** Decisions that wait for the PI at the snapshot

| # | Decision | Why it is open | Evidence in |
| --- | --- | --- | --- |
| 1 | Confirm or change the 10% margin of E3a and E3b | Marked as awaiting the PI. At the interim, E3b would pass at a 20% margin (Holm p 0.022) and E3a would not (0.313). A margin chosen after the result is not a pre-registered test | Section 4 |
| 2 | The primary output mode | Marked as awaiting the PI; the files record no choice | Section 4 |
| 3 | The retention thresholds and other scientific thresholds | Still labelled PILOT (activation 0.05, cache 0.05, allowed loss increase 0.05) | Section 6 |
| 4 | Whether step G runs as registered, and its addendum | The plan names no runner, tasks, seeds, endpoint or control | Section 6 |
| 5 | The definition of Q, the scope of an exploratory maintenance study, and when dynamic maintenance starts | The review finds that Q mostly measures how often a case was used | Section 6 |
| 6 | What to do with the legacy grid | Stopped at 108 of 110 jobs; no restart recorded | Section 5 |
| 7 | How to describe the pre-registration timing | 21 of 50 transfer-grid jobs already had evaluation-seed files when addendum 1 was recorded; addenda 3 and 4 are outside the hashed copy | Section 7 |

*Source: neural-cbr-ema/configs/t1/prereg_20261013.json; neural-cbr-ema/docs/T1_PREREG_20261013.md; neural-cbr-ema/docs/T1_CBM_DESIGN_REVIEW_20261019.md; neural-cbr-ema/results/t1/p6_confirmation_revision/grid_transfer_20261013/*

Decisions 1 and 2 concern the confirmation, 3 to 5 concern step G, 6 the legacy grid, and 7 the wording about timing. Decision 1 does not change what the interim data show. Q is the score step G uses to rank stored cases, built from how often a case helped or hurt when retrieved.

## 9. Caveats

Table 45 lists the caveats that matter this week. Each points to tables of Sections 2 to 7. They apply to every number in this summary. The earlier experiments carry labels E1 to E20 in the previous report. These are not the endpoints E1 to E3b.

**Table 45.** Caveats of this week, where the evidence is, and what would settle each

| # | Caveat | Evidence in | Effect on how results may be read | What would settle it |
| --- | --- | --- | --- | --- |
| 1 | Interim analysis of 29 of 30 tasks; the pre-registered analysis has no clause for interim looks | Table 17, Table 18, Table 25 | Every confirmation number is interim. In a scan of assumed QSAR-TID-11 results the smallest pooled p was 0.030, above the Holm threshold 0.0125. That is a calculation, not a prediction | The report on 30 tasks, after the audits |
| 2 | Asymmetric tuning. F is tuned per task on 18 tasks (19 recipes, 6 development seeds) and uses one recipe on 12. Each baseline is tuned from 32 candidates on 2 development seeds | Table 9, Table 26, Table 29 | Tables cannot say which side this favors. Interim E3a GMR is 1.139 | Baselines at the tuning budget of F, or F at theirs |
| 3 | Seeds are treated as independent, but they re-split the same rows. Ten evaluation seeds cover only 4 benchmark repeats, and evaluation test rows may have served as training rows under development seeds | Table 14, Table 18 | Intervals across seeds are probably too narrow, by an unmeasured amount. This cannot turn an unmet endpoint into a met one | A row-overlap check; an analysis clustered by outer repeat |
| 4 | Timing and housekeeping. Addenda 3 and 4 are outside the hashed copy; 21 of 50 transfer-grid jobs had evaluation-seed files when addendum 1 was recorded; the seed-audit record post-dates the first confirmation job | Table 42, Table 40, Table 14 | "Before any evaluation-seed result" holds for runs of F, not for the comparison-method evaluations of the transfer grid. None of those early files is a run of F | A dated PI disclosure; a saved audit of all 30 configs |
| 5 | Mixed code fingerprints: three confirmation copies, and an older fourth copy for the transfer grid | Table 41 | Model files are identical in the three confirmation copies. The grid copy has 7 different model files, with an effect not assessed | One job run from two copies, compared |
| 6 | Repeated use of the development test streams: the same 108 held-out cells judged several tuning blocks, uncorrected | Table 6, Table 13 | The adopted per-task selection (z -1.979, and -1.661 with the feasibility rule) and the in-sample ratio 0.816 are optimistic; the nearest held-out value is 0.898. Only the confirmation can speak about F | The confirmation; reserved development seeds 26 to 29 |
| 7 | F is a package without maintenance. F_global differs from the E2 reference in six settings, and 12 of 30 tasks run F_global | Table 8, Table 9, Table 24 | A positive E2 would show that the package beats the default package, not that joint training alone is the cause. E1 strata (0.936 on 18 tuned tasks, 0.996 on 11 F_global tasks) overlap with task size and type | A run changing one setting at a time; step G |
| 8 | Weak evidence on maintenance. The 10-19 design review lists 13 flaws; on five tasks most stored cases are never retrieved | Table 34, Table 35, Table 36 | "Removal gives no consistent gain" is the expected outcome of the current set-up. Earlier removal results are weak evidence either way | Step G with an addendum, or an exploratory study on data with known noise |
| 9 | Unplanned machine stop and one rebuilt row: 16 duplicate rows (all byte-identical) and one yacht row rebuilt from the run's files | Table 39, Table 40 | For the 29 finished tasks the audits find rows complete and equal to what the runs produced. This does not show that the numbers are right | The audit of QSAR-TID-11 |

*Source: neural-cbr-ema/results/t1/p6_confirmation_revision/ (confirmation rows, audits, grid_transfer_20261013); neural-cbr-ema/results/t1/p4_component_sync/tuning_20261013/; neural-cbr-ema/docs/T1_PREREG_20261013.md; neural-cbr-ema/docs/T1_EXECUTION_LEDGER.md; neural-cbr-ema/docs/T1_CBM_DESIGN_REVIEW_20261019.md; NEURAL_CBR_PROGRESS_2026-10-13_EN.md (caveat table)*

The interim outcomes are all "rule met: no". In our reading, caveats 3 and 6 make the evidence look stronger than it is. They cannot turn an unmet rule into a met one, but they weigh on any later positive claim. Caveat 2 can act in either direction. Caveats 4, 7 and 8 limit what may be claimed about order of events, joint training alone, and maintenance.

None of the previous report's caveats is fully resolved, and four were added: the maintenance design review, the unplanned stop, the partial seed audit, and F without maintenance. The earlier experiments keep their trust status: most are small pilots to be read with a caveat, and four are mechanism checks that stand. Spot checks of earlier cells against raw files agree in all sampled cells.

Three statements of the previous report need correction.

- **What F contains.** The previous report described F as including maintenance. All 30 configurations use only the joint schedule and the alternating default. None has a maintenance setting.
- **Origin of the F_global choice.** F_global is the small-penalty variant (v_small, a weight on large adapter corrections), and it was ranked on validation rows only, not on test results.
- **First NN-kNN evaluation.** The previous report dated it by the job completion record. The first evaluation file is three minutes older than addendum 1.

How to weigh the groups of results: the pre-registered confirmation can support the four endpoints once the report is run, and nothing before. The step D decisions record how F was chosen but cannot show that F beats anything, because the same development streams judged several blocks. Earlier experiments show that mechanisms run and reproduce, not rankings or the quality of maintenance.
## 10. Where the results are

Table 46 lists where each topic lives. Paths are relative to the workspace root, were checked with a directory listing on 10-20, and nothing was changed. The label `pi20260920` in folder names is a pipeline revision, not a date of results. `neural-cbr-ema/results/README.md` is the project's own index.

**Table 46.** Where the evidence for each topic is stored

| Topic | Folder or document | What is there |
| --- | --- | --- |
| Previous report | `NEURAL_CBR_PROGRESS_2026-10-13_EN.md` (Chinese: `NEURAL_CBR_PROGRESS_2026-10-13_ZH.md`) | Snapshot 10-13 about 11:55: tables, caveats, to-do list |
| Pre-registration | `neural-cbr-ema/docs/T1_PREREG_20261013.md`; `neural-cbr-ema/configs/t1/prereg_20261013.json` | The plan with addenda 1 to 4; the JSON holds addenda 1 and 2 and the hash the report tool checks |
| Tuning note | `neural-cbr-ema/docs/T1_TUNING_CONFIRMATION_20261013.md` | Step table (updated after the snapshot) and a short account |
| Step D decisions and records | `neural-cbr-ema/results/t1/p4_component_sync/tuning_20261013/`; `neural-cbr-ema/results/t1/p4_component_sync/pi20260920/` | `DECISIONS.md`, one file per block, freeze records; raw run records |
| Frozen configurations | `neural-cbr-ema/configs/t1/confirm_final_20261013/` | The 30 configuration files |
| Confirmation | `neural-cbr-ema/results/t1/p6_confirmation_revision/pi20260920/confirm_final_20261013/`; `neural-cbr-ema/logs/t1/p6_confirmation_revision/confirm_final_20261013/` | `runs__<task>.jsonl` rows, manifests; receipts, progress files, job logs |
| Audits | `neural-cbr-ema/results/t1/p6_confirmation_revision/` (`seed_audit_confirm_20261018.json`, `confirm_audit_repair_20261018.json`, `confirm_audit_transfer3_20261019.json`) | Seed audit of 29 configs; row audits |
| Baselines and transfer grid | `neural-cbr-ema/results/t1/p6_confirmation_revision/baselines_eval40_20261013/`; `neural-cbr-ema/results/t1/p6_confirmation_revision/baselines_eval50_20261013/`; `neural-cbr-ema/results/t1/p6_confirmation_revision/grid_transfer_20261013/` | Refitted baseline winners; the 10 tasks times 5 families |
| Legacy grid | `neural-cbr/results/t1_pi20260920/grid_full_20261009/` | 22 tasks times 5 families: 108 of 110 jobs |
| Gate D8 | `neural-cbr-ema/results/t1/p6_confirmation_revision/*_gate_d8/` | README and run records of the gate |
| Maintenance | `neural-cbr-ema/docs/T1_CBM_DESIGN_REVIEW_20261019.md`; `neural-cbr-ema/results/t1/p1_supervised_retention/pi20260920/cbm_diagnostics_20261019/` | The 13-flaw review; read-only diagnostics of Q at K = 0.5 |
| Carried-over experiments | `neural-cbr/results/t1_pi20260920/`; `neural-cbr-ema/docs/experiments/2026-10-02-pi-alignment-runs.csv` | Original run files; the 108-row table |
| Ledger and tools | `neural-cbr-ema/docs/T1_EXECUTION_LEDGER.md`; `neural-cbr-ema/tools/` | Log of launches, the 10-13 stop, audits, code copies; the programs behind freeze, audits, report, grids |
| Written after the snapshot (not used here) | `neural-cbr-ema/results/t1/p6_confirmation_revision/confirmation_report_20261020/`; `neural-cbr-ema/results/t1/p6_confirmation_revision/confirm_audit_final_20261020.json`; `NEURAL_CBR_PROGRESS_2026-10-20_EN.md` | Report folder and audit dated 10-20 08:45; a later progress report |

*Source: directory listings of the workspace on 10-20 (read-only)*

The last row holds documents that post-date this summary's snapshot. They may supersede the interim statements made here.
