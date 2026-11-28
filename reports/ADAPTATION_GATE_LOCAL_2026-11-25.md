# Apply-or-skip adaptation: local exploration

**Follow-up completed:** [Learning one disagreement threshold](ADAPTATION_THRESHOLD_LOCAL_2026-11-25.md) compares two new trainable-cutoff variants with all methods below. It does not establish an overall improvement over this report's validation-selected disagreement rule.

November 25, 2026. Study `ag1`, sourced from the completed old-adapter runs in `am1`. **Complete: 14 datasets × 3 seeds; 126 adapter continuations, 42 recorded control evaluations and 42 recorded input reconstructions.** No remote runs or retrieval retraining.

## Decision

The magnitude/direction design is **shelved at the owner's request**. Its code and completed evidence remain reproducible; it is not the active candidate.

The most promising gating candidate in this first exploration is a **simple neighborhood-disagreement threshold**, chosen on validation data, that either applies the old adapter's full correction or leaves retrieval unchanged. It improves task-mean error on 5/7 classification datasets and 5/7 regression datasets. Its gains are modest in regression, and it still loses useful adaptation on Iris.

The proposed learned confidence head is implemented and tested, but **neither learned-head variant establishes a general improvement**. Both learned gates suppress useful classification corrections. Extra adapter training without a gate is a stronger classification result than either learned gate, but is unstable on regression. Keep the old adapter as the default; treat the disagreement rule as the next candidate to confirm on new splits.

## What was tried

All alternatives start from each old adapter's selected retrieval core and full residual proposal. Retrieval remains frozen throughout this experiment.

| Method | Decision or training |
| --- | --- |
| Retrieval | Apply no adaptation. This is the old arm's selected core, not a separately retrained retrieval-only model. |
| Original | Apply the completed old adapter's full correction. |
| Global on/off | Validation chooses always adapt or never adapt for that dataset/seed. |
| Global shrink | Validation chooses one multiplier from 0, .25, .5, .75, 1. |
| Disagreement threshold | Apply the full old correction only when retrieved targets disagree sufficiently; validation chooses a threshold from training-feature quantiles, with always/never also available. |
| Extra training, no gate | Continue the old adapter with retrieval frozen; controls for extra training and optimizer reset. |
| Probability gate | Continue the old adapter and a shared-trunk second head together. The head predicts whether the full proposal improves task surrogate loss; apply the full proposal when its sigmoid exceeds .5. |
| Utility gate | Same architecture and decision, but weight gate examples by the absolute loss difference so large harms matter more than small wins. |

The two learned heads use the shared adapter representation, all four neighborhood statistics, the retrieval prediction and full residual proposal. There is no unit-direction normalization or learned correction magnitude. A closed gate produces exactly the retrieval prediction; an open gate produces exactly the full adapted prediction.

The gate and adapter train together. The full residual head retains its supervised loss even on examples the gate skips, preventing a closed gate from stopping residual learning. Gate labels and severity weights are detached. For classification, “helpfulness” is defined using cross-entropy on a common scaled score space; it is a surrogate for correct classification. The probability output is not validated as calibrated confidence, and the utility output is a decision score rather than a probability of correctness.

This is a **frozen-retrieval continuation experiment**, not an integration into every synchronized/alternating training path. The [protocol and design](../docs/ADAPTATION_GATE_EXPERIMENT.md) were written before the new results. The cost-sensitive perspective is motivated by [learning to defer](https://proceedings.mlr.press/v119/mozannar20b.html); this experiment does not reproduce that paper or inherit its guarantees automatically.

## Overall results

Each entry averages the three seeds within each task, then weights tasks equally. Classification uses error percentage; regression uses standardized RMSE. GMR change is the change in the geometric mean of task-error ratios versus the original full adapter. **Negative GMR change is better.** The arithmetic mean and GMR answer different aggregation questions, so both are shown.

| Method | Classification error % | Classification GMR change | Regression RMSE | Regression GMR change |
| --- | ---: | ---: | ---: | ---: |
| Retrieval | 10.385 | +9.59% | 0.352137 | +14.40% |
| Original full adapter | 9.945 | 0.00% | 0.338910 | 0.00% |
| Global on/off | 9.896 | +3.27% | 0.338557 | −0.03% |
| Global shrink | 9.584 | +2.55% | 0.338398 | −0.16% |
| **Disagreement threshold** | **9.164** | **−3.68%** | **0.337424** | **−0.52%** |
| Extra training, no gate | 9.068 | −11.88% | 0.348525 | +4.13% |
| Probability gate | 10.348 | −1.01% | 0.337206 | +0.12% |
| Utility gate | 9.911 | −4.15% | 0.339802 | +2.08% |

No task means were excluded for zero errors. These are descriptive results from previously examined splits, not statistical significance claims. The probability gate has the lowest arithmetic mean regression RMSE, but is essentially flat/slightly worse by GMR; the utility gate's classification GMR gain coexists with an almost unchanged arithmetic mean. Neither is a clear cross-family winner.

![Relative test-error changes for all tried alternatives](../results/adaptation_improvement/ag1/gate_task_effects.png)

## The simple disagreement gate

The rule uses the previously discussed target-disagreement statistic:

`D_y = sum_i w_i ||target_i − sum_j w_j target_j||²`.

For classification this is `1 − sum_c p_r[c]²`; for regression it is weighted target variance in standardized units. It uses query-dependent retrieval activations and training-case labels, never the query's true target.

The decision is `apply full adaptation if D_y > threshold`. Threshold candidates are the training distribution's 0%, 25%, 50%, 75% and 100% quantiles, plus always/never. Validation selects the candidate; test labels do not select it. No neural head or extra adapter training is needed for this control.

| Family | Better / tie / worse task means | Better / tie / worse seed runs | Helpful corrections, old → gated | Harmful corrections, old → gated |
| --- | ---: | ---: | ---: | ---: |
| Classification | 5 / 0 / 2 | 8 / 10 / 3 | 25 → 13 | 24 → 4 |
| Regression | 5 / 1 / 1 | 8 / 10 / 3 | 3,030 → 2,390 | 3,309 → 2,345 |

Classification's mean error falls by **0.781 percentage points**, a 7.85% reduction in the arithmetic mean error. It applies adaptation to only **104/2,193 test evaluations (4.74%)**. Of those, 13 change a wrong class to the correct class and four do the reverse; the remaining applied corrections do not change correctness. Eleven of 21 classification runs make no test adaptations at all; ten make selective decisions. The equal-task mean application rate is 8.47%, different from the pooled rate because task sizes vary.

Regression's arithmetic mean RMSE improves by only **0.44%**. The gate applies 4,735/6,339 corrections (74.70%). Among the applied corrections, **49.52% still worsen squared error**, versus 52.20% for the original adapter. Therefore this is a modest filtering improvement, not evidence that the system can confidently identify almost every good correction. Wrong-direction updates fall from 2,616 to 1,969 and destructive overshoots from 518 to 339, partly because fewer corrections are attempted.

The rule trades away useful corrections as well as harmful ones. The task-level losses on Balance and Iris and the small regression loss on Student Performance remain in the result.

### Wrong-direction rates: all queries versus applied corrections

November 27 clarification: an independent recount of 126 saved regression method/split prediction sets confirms the following. All methods have 6,339 test-query evaluations, including repeated observations across seeds. The applied denominator uses the saved gate-open decisions. [Recount code](../tools/t1_gate_direction_audit.py), [counts and run sources](../results/adaptation_improvement/ag2/gating_direction_denominators.json).

| Method | Queries adapted % | Wrong direction / all queries % | Wrong direction / applied corrections % | Mean RMSE |
| --- | ---: | ---: | ---: | ---: |
| Original full adapter | 100.00 | 41.27 | 41.27 | 0.338910 |
| Validation disagreement gate | 74.70 | 31.06 | 41.58 | 0.337424 |
| Learned probability gate | 57.69 | 23.55 | 40.83 | 0.337206 |
| Learned utility gate | 59.17 | 24.64 | 41.64 | 0.339802 |
| Learned threshold, frozen adapter (`ag2`) | 98.58 | 40.64 | 41.22 | 0.338856 |
| Learned threshold + adapter (`ag2`) | 95.90 | 37.56 | 39.17 | 0.347770 |

**Lower wrong-direction frequency across all queries mostly reflects fewer corrections being attempted.** The disagreement gate suppresses 647 wrong-direction updates but also 640 helpful corrections; wrong-direction frequency among accepted corrections does not improve. It nevertheless reduces harmful overshoots and makes a small improvement in final RMSE. The probability gate has a modest direction-selection benefit: its own full proposal has 41.99% wrong directions, versus 40.83% among accepted corrections. The equivalent comparisons are 42.06% to 41.64% for utility gating and 39.45% to 39.17% for the jointly trained threshold. The learned models also change the adapter, so their own forced-open proposals are the appropriate control for the immediate gate effect.

These experiments support reducing exposure to harmful corrections, but do not show reliable identification of wrong directions. An always-closed gate would make the all-query wrong-direction rate zero while discarding every useful adaptation. RMSE and retained benefit must therefore accompany that rate. All these results concern the earlier NN-CDH adapter and its selected retrieval cores; **gating the RF pair adapter remains untested**, and the historical percentages are not a matched RF comparison.

## The learned heads: separate better proposals from better decisions

The additional training changes the full proposal. To assess the gate's immediate contribution, I also compare the chosen checkpoint with its **own** gate forced open:

| Learned model | Own full proposal, gate forced open | Actual gated output |
| --- | ---: | ---: |
| Probability gate, classification error | 9.522% | 10.348% |
| Utility gate, classification error | 9.110% | 9.911% |
| Probability gate, regression RMSE | 0.340768 | 0.337206 |
| Utility gate, regression RMSE | 0.349395 | 0.339802 |

Both learned gates **harm classification on average relative to their own full proposals**, by approximately 0.83 and 0.80 percentage points. Their sometimes improved final results versus the original adapter cannot all be credited to choosing when to adapt. Continued training and the auxiliary gate objective also change the proposal.

In regression, each learned gate improves its own full-proposal average. However, the utility-trained full proposal is worse to begin with, so gating does not bring it below the original baseline. The probability model helps Car but loses substantially on Yacht.

The trained heads are not always selective. Across 21 runs per family:

| Family / head | Applies to every test query | Applies to no test query | Selective |
| --- | ---: | ---: | ---: |
| Classification probability | 4 | 2 | 15 |
| Classification utility | 2 | 2 | 17 |
| Regression probability | 2 | 4 | 15 |
| Regression utility | 4 | 2 | 15 |

The no-adaptation cases above selected epoch zero. Initial eligibility is intentional: validation may reject further adaptation. This avoids claiming that a learned confidence estimate caused a decision that was already available before training.

## Per-dataset errors

All values are three-seed means, lower is better. The simpler global controls remain in the [full table](../results/adaptation_improvement/ag1/tasks.csv).

Classification error rate:

| Dataset | Original | Disagreement gate | Extra training, no gate | Probability gate | Utility gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Balance | 0.053333 | 0.056000 | 0.050667 | 0.045333 | 0.050667 |
| Breast Cancer | 0.029240 | 0.020468 | 0.032164 | 0.017544 | 0.017544 |
| Digits | 0.021296 | 0.020370 | 0.021296 | 0.022222 | 0.018519 |
| Iris | 0.033333 | 0.055556 | 0.022222 | 0.077778 | 0.066667 |
| Wine | 0.074074 | 0.064815 | 0.046296 | 0.046296 | 0.055556 |
| Zebra | 0.181818 | 0.151515 | 0.196970 | 0.242424 | 0.181818 |
| Zebra Special | 0.303030 | 0.272727 | 0.265152 | 0.272727 | 0.303030 |

Regression standardized RMSE:

| Dataset | Original | Disagreement gate | Extra training, no gate | Probability gate | Utility gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Airfoil | 0.208094 | 0.208094 | 0.209122 | 0.208489 | 0.209491 |
| Body Fat | 0.627485 | 0.623705 | 0.627585 | 0.624602 | 0.624396 |
| Car | 0.222880 | 0.219538 | 0.289201 | 0.212815 | 0.225254 |
| Diabetes | 0.768861 | 0.763774 | 0.768861 | 0.766409 | 0.764295 |
| Energy Efficiency | 0.047689 | 0.047262 | 0.048648 | 0.047316 | 0.049394 |
| Student Performance | 0.441026 | 0.443569 | 0.439926 | 0.440397 | 0.443758 |
| Yacht | 0.056331 | 0.056027 | 0.056331 | 0.060413 | 0.062027 |

The large Car regression loss from extra training is retained: seed 42 selects epoch 41 on validation and obtains test RMSE 0.424215. It is not removed as an outlier. This illustrates the risk of drawing broad conclusions from the strong classification result of that same extra-training control.

## Interpretation and next step

The results support investigating **selective use of the old adapter**, particularly a low-capacity rule based on neighborhood disagreement. They do not establish that a learned confidence head solves unreliable adaptation.

One plausible reason the simple rule is competitive is that it needs very little fitting, whereas the learned gates share parameters with an adapter and learn changing helpfulness targets from training neighborhoods. Classification also has a mismatch between improving the surrogate score loss and improving the final class decision. These are hypotheses, not causes isolated by this experiment.

The next priority is to freeze the disagreement-rule protocol and evaluate it on new local splits/seeds, alongside the original adapter and the two simple global controls. A further learned-gate experiment should investigate held-out or cross-fitted routing targets and align classification routing with decision cost. That would address training-neighborhood optimism and surrogate mismatch; it should not be tuned against these already inspected test results.

No default model behavior was changed. No new magnitude/direction experiments were run.

## Protocol, limitations and verification

- All 42 original checkpoints reproduce their recorded test predictions and metrics after reconstruction, with matched data fingerprints and query targets. Retrieval weights are frozen and training self-retrieval is excluded by stable case ID.
- Six alternatives are evaluated against the original and retrieval baselines, giving 336 method/seed rows. There are 210 complete run units and no missing units.
- Learned continuations use fresh Adam/AdamW state, learning rate capped at .001, original weight decay and adapter-side loss coefficients/free radius, batch size 64, up to 60 epochs and patience 12. Epoch zero is eligible. The synchronized retrieval objective is not added to this frozen continuation; the no-gate control uses the same objective, budget and optimizer-reset protocol.
- The neural gate is trained on surrogate-loss improvement; severity-weighted BCE uses detached absolute loss differences normalized within each minibatch. The hard decision has no gradient estimator. The full residual and gate losses update both heads/shared trunk together.
- All normalization statistics come from initial training inputs; validation selects checkpoints, global multipliers and disagreement thresholds. Test targets enter only recorded evaluation and diagnostic calculations.
- The original adapter already used the same validation split for checkpoint selection. Reusing that split for this exploratory continuation/control comparison can overfit validation decisions; it is not a fresh calibration split. The test splits were also inspected in the preceding project. The experiment evaluates this declared batch of ideas, not an untouched confirmation.
- Results cover 14 available local tasks, not all 29 historical tasks. Three seeds and a few class flips on small datasets do not support significance or universal-confidence claims. The frozen-retrieval setting does not establish behavior during joint or alternating retrieval updates.
- Eight focused gate tests pass; together with the prior diagnostic tests, **15 tests pass**. Tests cover exact full/zero decisions, preservation of the old proposal, both heads receiving gradients, loss-weighted decisions, detached targets, train-only normalization, validation-only static selection, mandatory recording and selected/resume weights.
- An independent calculation from saved prediction tensors reproduced all **672 validation/test metric records** and checked hard decisions against the full or retrieval branch. All 42 source reconstruction checks passed. A completed Iris run was invoked again and loaded without training.
- Every epoch has a retained resume checkpoint; selected weights, curves and predictions remain under `runs/`. Source runs were not modified. Generated summaries and the inspected figure remain under `results/`.

Evidence and reproduction:

- [Declared design/protocol](../docs/ADAPTATION_GATE_EXPERIMENT.md), [head and joint adapter training](../model/t1/adaptation_gate.py), [experiment and summary runner](../tools/t1_adaptation_gate.py), [tests](../tests/test_t1_adaptation_gate.py).
- [All method/seed rows](../results/adaptation_improvement/ag1/runs.csv), [task means](../results/adaptation_improvement/ag1/tasks.csv), [family summaries](../results/adaptation_improvement/ag1/families.csv), [complete machine-readable results](../results/adaptation_improvement/ag1/comparison.json).
- Run units: `NNKNN_RUNS/t1/ag1/gate_v1/`; source units: `NNKNN_RUNS/t1/am1/adaptation_improvement_v1/`.
- Local launcher: `adaptation improvement/code/run_gate_experiment.py`. From the main workspace, use `uv run python "adaptation improvement/code/run_gate_experiment.py" --mode summarize` to rebuild summaries. `--mode run` loads completed units; it does not train them again.
