# Adaptation behavior before and after magnitude/direction separation

November 25, 2026. Local study `am1`; 14 datasets, seeds 40–42, 42 matched old/new pairs. Analysis of saved predictions only; no retraining or additional model evaluation.

## Findings

**The new adapter often misses useful corrections, but increasing every correction would make results worse. Direction errors remain a major limitation. Over-correction is real on some datasets, especially Yacht, rather than a single explanation for the whole suite.**

| Question | What the saved results show |
| --- | --- |
| Are we adapting enough? | For classification, 64 of the new model's 111 retrieval errors could be corrected by a larger, individually chosen positive scale along the existing direction. Only 8 errors are actually fixed, versus 25 of 117 before the change. For regression, some tasks have room for larger corrections, but others benefit from smaller ones. |
| Are we over-correcting? | Harmful classification flips fall from 24 to 11. Pooled regression harmful overshoot falls from 8.17% to 6.39% of evaluations, but rises on Yacht from 10.22% to 17.20%. A smaller average correction does not guarantee less overshoot. |
| Are we going in the wrong direction? | Pooled regression wrong-direction frequency rises from 41.27% to 43.34%; these corrections account for 90.12% of the new adapter's added squared error on cases it worsens. For classification, 39 of 111 new retrieval errors cannot be fixed by any positive scale along the predicted direction. |
| Did the change improve adaptation? | It reduces damage in several places, while also losing useful corrections. The new regression adapter improves its own selected core on all seven task means, but the old adapter still achieves lower final RMSE on five. Classification's new adapter fixes fewer errors than it introduces in the pooled evaluations: 8 fixes versus 11 harmful flips. |

“Could be corrected” is a **hindsight diagnostic using test labels**, not evidence that the appropriate scale was predictable from the available inputs. These results do not establish that the scale head learned confidence.

The [original performance comparison](ADAPTATION_IMPROVEMENT_LOCAL_2026-11-25.md) remains unchanged: the new complete model wins on 7 of 14 task means, while geometric mean new/old error is 3.68% higher for classification and 3.03% higher for regression. This report explains the behavior underneath those final scores.

## What is being compared

**Old** is the existing NN-CDH residual adapter. **New** is the shared-trunk, magnitude/direction adapter: `final = retrieval + m × u`, where `m ≥ 0` and `u` is a unit direction. Its magnitude head receives the shared representation and the four neighborhood statistics; its raw direction proposal receives residual supervision. See the [design notes](../docs/ADAPTATION_MAGNITUDE_DIRECTION.md).

Within each seed, both arms start from the same recorded pretrained core, optimizer state, starting RNG and data split. Joint training and checkpoint selection can then produce different retrieval cores. Every “before adaptation” value below is the prediction of **that arm's own selected core**, immediately before its final correction. It is not necessarily the initial pretrained retrieval prediction.

The analysis covers all 84 adapted units and 17,064 saved test evaluations: per arm, 2,193 classification and 6,339 regression evaluations. A query can occur in multiple seed splits; these are evaluation counts, not independent people or unique examples. Task metrics average three seeds equally. Pooled counts weight larger datasets more heavily, so equal-task summaries are identified separately.

## Regression: direction is the larger source of harm

For a saved prediction, let the required residual be `r = target − retrieval` and the applied correction be `c = final − retrieval`. When `r` is nonzero, the ratio `q = c/r` separates direction and size:

| Ratio | Interpretation | Effect on squared error versus retrieval |
| --- | --- | --- |
| `q < 0` | Wrong direction | Worse |
| `0 < q < 1` | Toward the target, stopping short | Better |
| `q = 1` | Reaches the observed target | Zero error |
| `1 < q < 2` | Passes the target, but remains closer | Better |
| `q > 2` | Destructive overshoot | Worse |

At `q = 2`, error equals the original error. Moving away from an already accurate retrieval is counted separately. Thus **crossing the target is not automatically a harmful correction**.

### Results by dataset

RMSE is in training-standardized target units. Percentages pool the three seeds within each task and use all its test evaluations as the denominator. Arrows in the last two columns mean old → new.

| Dataset | Evaluations per arm | Old retrieval → final RMSE | New retrieval → final RMSE | Wrong direction % | Harmful overshoot % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Airfoil | 903 | 0.21342 → 0.20809 | 0.21152 → 0.21026 | 42.41 → 43.19 | 4.21 → 5.20 |
| Body Fat | 153 | 0.62530 → 0.62749 | 0.62530 → 0.62375 | 48.37 → 42.48 | 0.65 → 0.00 |
| Car | 3,978 | 0.23196 → 0.22288 | 0.23230 → 0.23002 | 41.18 → 44.02 | 10.03 → 7.24 |
| Diabetes | 267 | 0.76579 → 0.76886 | 0.76579 → 0.76561 | 47.19 → 48.69 | 3.00 → 0.37 |
| Energy Efficiency | 462 | 0.05028 → 0.04769 | 0.05005 → 0.04790 | 40.69 → 41.56 | 7.36 → 7.58 |
| Student Performance | 390 | 0.45199 → 0.44103 | 0.44537 → 0.44231 | 44.36 → 45.38 | 4.87 → 0.51 |
| Yacht | 186 | 0.12622 → 0.05633 | 0.10047 → 0.06678 | 18.28 → 22.58 | 10.22 → 17.20 |

![Regression correction categories, old and new for all seven tasks](../results/adaptation_improvement/am1/behavior/regression_behavior.png)

Equal task weighting gives wrong-direction frequencies of **40.35% old versus 41.13% new**, and harmful-overshoot frequencies of **5.76% versus 5.44%**. The direction frequency worsens on six of seven tasks. Car supplies 63% of pooled regression evaluations, so the pooled percentages alone should not stand in for the typical dataset.

Among cases made worse, wrong-direction updates account for **75.44% old versus 90.12% new** of pooled added squared error; destructive overshoot accounts for **24.43% versus 9.81%**. The remainder is movement from essentially exact retrieval predictions. Averaging the wrong-direction harm share equally across tasks also shows the same ordering: 88.57% versus 91.66%. These are shares of added error on harmed cases, not shares of final error or net error change.

Three cases clarify the different behaviors:

- **Body Fat: less damaging and still conservative.** Mean absolute correction falls from 0.05025 to 0.01063. The new adapter has no destructive overshoots, improves direction frequency, and turns a small average loss into a small gain. In hindsight, increasing its saved corrections improves the three-seed mean further, but the seeds disagree sharply: the best nonnegative uniform test multiplier is 0 for one seed and about 31.94 for another. This is not a reliable task-wide scale recommendation.
- **Diabetes: mainly reduced damage.** Direction frequency becomes slightly worse, yet mean correction falls from 0.06254 to 0.01228 and the average final RMSE improves. Smaller updates can help even without better directions. The improvement over retrieval itself is very small.
- **Yacht: a genuine overshoot problem as well as worse direction.** Average correction shrinks from 0.08116 to 0.06487, but the new core is already closer to the target. Destructive overshoot rises from 10.22% to 17.20%, and wrong-direction frequency also rises. Each seed's hindsight best uniform multiplier is below 1 (0.93, 0.61, 0.72). This task does not support simply making the new corrections larger.

The new adapter helps its own core in 17 of 21 regression seed runs, versus 15 of 21 for the old adapter. This can coexist with worse final performance: avoiding large losses on Body Fat and Diabetes does not compensate for weaker useful corrections elsewhere.

## Classification: useful directions often lack enough scale

A class-score correction has no single scalar target distance. I therefore hold the saved retrieval scores `p0` and correction vector `c` fixed and examine:

`scores(a) = p0 + a × c, a > 0`.

Solving the true-class-versus-every-rival inequalities gives the interval of positive multipliers where the true class wins. This distinguishes an insufficient scale from a direction that no positive multiplier can repair. It uses relative class margins, so adding the same offset to every score does not count as useful movement.

For **initially wrong retrievals**, the pooled breakdown is:

| Outcome | Old, out of 117 errors | New, out of 111 errors |
| --- | ---: | ---: |
| Fixed at the actual scale | 25 (21.37%) | 8 (7.21%) |
| Would need a larger positive scale | 52 (44.44%) | 64 (57.66%) |
| Would need a smaller positive scale | 0 | 0 |
| No positive scale can make the true class win | 40 (34.19%) | 39 (35.14%) |

For **initially correct retrievals**, the adapter introduces 24 errors out of 2,076 old evaluations (1.16%), versus 11 out of 2,082 new evaluations (0.53%).

The zero count in the “smaller scale” row does **not** mean over-correction is absent. That row asks whether an already wrong prediction passed through a correct class and then left it. Harmful flips from initially correct retrievals are counted separately. Shrinking those corrections toward zero restores the original correct decision; the observation alone cannot distinguish poor direction from excessive size.

### Results by dataset

All cells show old → new counts across three seeds. “Larger scale” and “unrescuable” refer only to initially wrong, still-unfixed retrievals.

| Dataset | Retrieval errors | Fixed | Correct predictions harmed | Larger scale needed | Unrescuable by positive scaling |
| --- | ---: | ---: | ---: | ---: | ---: |
| Balance | 24 → 16 | 5 → 2 | 1 → 4 | 5 → 7 | 14 → 7 |
| Breast Cancer | 7 → 7 | 1 → 1 | 4 → 0 | 5 → 4 | 1 → 2 |
| Digits | 21 → 21 | 0 → 0 | 2 → 1 | 4 → 7 | 17 → 14 |
| Iris | 6 → 7 | 3 → 0 | 0 → 0 | 3 → 6 | 0 → 1 |
| Wine | 11 → 4 | 4 → 0 | 1 → 2 | 4 → 2 | 3 → 2 |
| Zebra | 12 → 17 | 4 → 0 | 4 → 2 | 8 → 16 | 0 → 1 |
| Zebra Special | 36 → 39 | 8 → 5 | 12 → 2 | 23 → 22 | 5 → 12 |

![Classification outcomes among retrieval mistakes, old and new](../results/adaptation_improvement/am1/behavior/classification_behavior.png)

**Iris and Zebra show missed useful adaptation.** For Iris, six of seven new retrieval errors are fixable along the saved direction, but none is fixed at the actual scale. Their minimum required multiplier has a median of 20.90×. For Zebra, 16 of 17 are fixable by enlarging their individual corrections; the median required multiplier is 33.26×. This is a much larger gap than a modest scale adjustment.

Across all 64 new “too small” cases, only **8** can be rescued by some individually chosen multiplier up to 2×, **14** up to 4×, and **25** up to 10×. These are optimistic per-query possibilities, not the result of applying one common multiplier.

**Breast Cancer and Zebra Special show a real reduction in damage.** Breast Cancer preserves its one fix and eliminates four harmful flips. Zebra Special loses three fixes but eliminates ten harmful flips. Conversely, Digits fixes no retrieval errors with either adapter; 14 of its 21 new retrieval errors cannot be rescued by positive scaling, indicating a substantial direction limitation for those cases.

New classification adaptation makes 8 helpful and 11 harmful flips, a net increase of three errors over its own core. Old adaptation makes 25 helpful and 24 harmful flips, a net reduction of one. The final pooled totals are nevertheless 114 new errors versus 116 old, because the new cores start with six fewer errors. This pooled ranking differs from the equal-task performance summary because task sizes differ.

## Would a general increase in adaptation help?

**No, on these saved predictions.** The following is a frozen-output counterfactual, not a trained model or a tuning recommendation. Each multiplier is applied to every saved correction. Zero uses each arm's own selected core.

| Multiplier | Old classification error % | New classification error % | Old regression RMSE | New regression RMSE |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 10.385 | 10.720 | 0.35214 | 0.34726 |
| 0.25 | 10.309 | 10.608 | 0.34560 | 0.34391 |
| 0.5 | 9.700 | 10.838 | 0.34055 | 0.34148 |
| 1, actual | 9.945 | 11.141 | 0.33891 | 0.34095 |
| 2 | 12.040 | 14.708 | 0.36706 | 0.35349 |

Each column averages task means equally; classification and regression are separate metrics. Full curves through 8× are in the [scaling evidence](../results/adaptation_improvement/am1/behavior/scaling_summary.csv).

The new model can therefore be too conservative on some wrong predictions **and** too aggressive on other predictions. A global increase amplifies both. For example, doubling the new correction improves mean Body Fat RMSE from 0.62375 to 0.62237, but worsens Yacht from 0.06678 to 0.14085 and Zebra error from 28.79% to 51.52%.

## What changed in retrieval versus adaptation?

Final differences are not pure adapter differences. The exact decomposition is:

`new final error − old final error = change in pre-adaptation error + change in the adapter's immediate error effect`.

For regression this identity uses MSE; RMSE differences are not additive in the same way. For classification it uses error rate.

- **Balance:** the new core improves error by 2.13 percentage points, while the adapter's immediate effect worsens by 1.60 points. The final model still improves by 0.53 points.
- **Wine:** the core improves by 6.48 points, while adaptation's immediate effect worsens by 4.63 points. The final 1.85-point gain is not evidence of a better final correction.
- **Iris:** the 4.44-point final loss comprises 1.11 points from the selected core and 3.33 points from lost adaptation benefit.
- **Zebra:** the 10.61-point loss comprises 7.58 points from the core and 3.03 points from adaptation.
- **Yacht:** the selected core improves mean MSE by 0.005678, but the immediate adaptation benefit is 0.006978 smaller, leaving final MSE 0.001300 worse.

This is an accounting decomposition at the selected checkpoints, not a causal ablation. In joint training, the adapter's objectives can also influence which retrieval core is learned. [All 14 decompositions](../results/adaptation_improvement/am1/behavior/decompositions.csv) are saved.

## Interpretation and next experiments

The strongest supported diagnosis is **uneven allocation of correction size plus persistent direction mistakes**. Head separation made the scale measurable, but it did not by itself improve observed regression direction. The magnitude head starts at 0.01; Iris's selected mean magnitudes are 0.01065–0.01371 at epochs 2–6, and 98.69% of Body Fat's saved magnitudes lie within 20% of initialization. These patterns are consistent with conservative initialization and early checkpoint selection. They do not identify the cause or prove learned confidence.

The existing task-specific correction penalties were retained. The change also added neighborhood statistics and raw-proposal supervision, and changed the classification proposal parameterization. This study does not isolate any one of those choices.

The next experiments should separate these possibilities:

1. **Check magnitude learning and checkpoint selection.** On validation data, compare a small predeclared set of magnitude initializations or learning rates, retaining the same direction/core controls. Inspect whether useful directions receive larger magnitudes before early stopping.
2. **Diagnose direction independently.** Hold the retrieval core fixed for a controlled comparison of direction training. In scalar regression, the applied hard sign receives no gradient from final loss; the raw residual objective trains its direction. Evaluate whether that objective yields better signs on validation residuals.
3. **Evaluate conditional size control, not blanket amplification.** Compare the two-head model with the same direction and a simple validation-fitted scale, then test the contribution of the four neighborhood statistics. Track helpful corrections, harmful flips, wrong-direction error and destructive overshoot alongside final error.

No such experiment was run here. Any selection should use development/validation data and be assessed on fresh seeds or held-out data, rather than using the test multipliers in this report.

## Methods, validation and evidence

The analysis script reads completed run-unit records only. It verifies the expected configurations, paired data fingerprints and targets, complete test query indices, finite values, and agreement between reconstructed and stored pre/post metrics. New magnitudes are checked against the actual applied correction norms. Classification flip counts and regression squared-error decompositions reconcile.

Regression uses absolute tolerance `1e-8` in standardized target units for zero correction/residual. “Moved from an accurate retrieval” includes residuals within that tolerance: 175 old and 154 new evaluations, all on Car. Near-zero residual ratios should not be interpreted as stable scale targets. The exact squared-error identity is `(r−c)²−r² = c²−2rc`. Classification solves the positive-scale interval across every competing class and follows recorded argmax decisions; isolated exact-tie-only solutions are excluded. It does not treat raw scores or scale as calibrated confidence.

Observed residual directions and ideal scales use realized labels, which can contain unpredictable variation. An optimal conditional-mean predictor can still move in the wrong direction for individual outcomes. These counts diagnose observed behavior; they do not quantify how much error a learnable model can eliminate. The sample contains only three previously used seeds and 14 locally available tasks out of the historical 29; there is no significance claim or remote-host comparison.

- [Analysis code](../tools/t1_adaptation_diagnostics.py); [seven passing mathematical diagnostic tests](../tests/test_t1_adaptation_diagnostics.py).
- [Per-task diagnostics](../results/adaptation_improvement/am1/behavior/tasks.csv), [all 84 units](../results/adaptation_improvement/am1/behavior/units.csv), [pooled and equal-task summaries](../results/adaptation_improvement/am1/behavior/families.csv).
- [Per-query diagnoses](../results/adaptation_improvement/am1/behavior/cases.csv), [per-unit scale curves](../results/adaptation_improvement/am1/behavior/curves.csv), [magnitude and required-scale summaries](../results/adaptation_improvement/am1/behavior/task_scales.csv).
- [Training-history extract](../results/adaptation_improvement/am1/behavior/histories.csv), [run IDs and fingerprints](../results/adaptation_improvement/am1/behavior/provenance.csv), [machine-readable summary](../results/adaptation_improvement/am1/behavior/diagnostics.json).
- Source: `NNKNN_RUNS/t1/am1/adaptation_improvement_v1/`, locally `C:/Users/yexia/Documents/GitHub/NN-kNN-haotang/runs/t1/am1/adaptation_improvement_v1/`.
- Reproduce from the T1 worktree with the existing shared uv environment and `NNKNN_RUNS` configured: `uv run --project ../../.. --no-sync python tools/t1_adaptation_diagnostics.py --study am1`. Generated evidence stays in this worktree's ignored `results/`; this written report stays in `reports/`.
