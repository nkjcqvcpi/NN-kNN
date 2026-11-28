# Learning the disagreement threshold: local comparison

November 25, 2026. Study `ag2`, using the completed `ag1` reference runs. **Complete: 14 datasets, three seeds, two new variants; 84 new recorded units and 420 method/seed comparison rows.** Local CPU execution only.

## Executive summary

**Making the threshold trainable did not establish an improvement over the validation-selected disagreement rule.** Learning just the threshold leaves all seven classification task means equal to the original full adapter and makes only a negligible regression improvement. Training the threshold alongside the adapter improves classification over the original but worsens regression, including the same large Car loss seen with extra adapter training without a gate.

The validation-selected rule remains the more balanced candidate across the two task families. It is not the winner on every metric: extra training without a gate has the lowest mean classification error, and the probability gate has the lowest mean regression RMSE. The jointly trained threshold has a slightly better classification geometric mean ratio than the validation rule, despite worse arithmetic mean error. None of this is fresh-split confirmation.

The main diagnostic is **weak or optimistic training feedback about when to skip**. Five of 21 frozen-adapter classification runs have identical branch costs on every training query, leaving no threshold-learning signal. In regression, the learned frozen-adapter threshold applies adaptation on every test query in 19/21 runs. These observations support investigating held-out routing supervision before adding gate complexity; they do not isolate the cause of the generalization gap.

## What changed

The gate has exactly one trainable scalar, `theta`, per dataset/seed. With training disagreement mean `mu` and standard deviation `sigma` fixed:

```text
z = (D_y - mu) / sigma
p_apply = sigmoid((z - theta) / 0.25)     # training relaxation only
t = mu + sigma * theta
prediction = full_adapter if z > theta else retrieval
```

The temperature and normalization are fixed buffers. The normalized comparison is mathematically equivalent to `D_y > t`; inference uses normalized coordinates to avoid cancellation when converting a cutoff near zero back to raw units. Cutoffs below the normalized value corresponding to zero disagreement mean always adapt. There is no learned slope, correction magnitude, or new input feature. The sigmoid output is an optimization device, not calibrated confidence.

The threshold minimizes expected branch cost, `(1-p_apply)*cost_retrieval + p_apply*cost_full`. Classification uses actual 0/1 errors; regression uses squared error. Costs are detached. This is a probability-weighted average of the two branch losses, **not** the loss of an averaged prediction. Every evaluated prediction uses exactly the full correction or zero correction.

| New variant | Training and selection |
| --- | --- |
| Learned threshold, frozen adapter | Only the scalar threshold trains. Exactly 60 epochs, five full-training-set Adam updates per epoch at learning rate .05; use the final iterate. Validation is logged but never selects the threshold or stops training. |
| Learned threshold + adapter | The adapter trains with its original supervised objective, including skipped examples. Then the threshold trains on current detached branch costs. Same maximum 60 epochs and patience 12 as `ag1`; validation selects the joint checkpoint, with epoch zero eligible. Threshold gradients use training labels only. |

Both arms initialize the cutoff at training median disagreement. Training neighborhoods exclude the query itself by stable case ID. Retrieval stays frozen. The joint arm updates both components in the same training stage; the gate does not send gradients into the residual network. The [protocol](../docs/ADAPTATION_THRESHOLD_EXPERIMENT.md) was written before the new test results. The eight existing methods below were read from completed `ag1` units without retraining. [1, 2]

## Comparison with every apply-or-skip alternative

Means average seeds within each task, then weight the seven tasks in each family equally. Classification error is a percentage; regression uses standardized RMSE. GMR change is the geometric mean task-error ratio versus the original full adapter, minus one. Lower is better. No task means were excluded for zero error. [1]

| Method | Classification error % | Classification GMR change | Regression RMSE | Regression GMR change |
| --- | ---: | ---: | ---: | ---: |
| Retrieval | 10.385 | +9.59% | 0.352137 | +14.40% |
| Original full adapter | 9.945 | 0.00% | 0.338910 | 0.00% |
| Global on/off | 9.896 | +3.27% | 0.338557 | -0.03% |
| Global shrink | 9.584 | +2.55% | 0.338398 | -0.16% |
| Validation disagreement threshold | 9.164 | -3.68% | 0.337424 | -0.52% |
| Extra training, no gate | **9.068** | **-11.88%** | 0.348525 | +4.13% |
| Probability gate | 10.348 | -1.01% | **0.337206** | +0.12% |
| Utility gate | 9.911 | -4.15% | 0.339802 | +2.08% |
| **Learned threshold, frozen adapter** | **9.945** | **0.00%** | **0.338856** | **-0.01%** |
| **Learned threshold + adapter** | **9.359** | **-4.16%** | **0.347770** | **+4.37%** |

![Mean test errors for the principal alternatives](../results/adaptation_improvement/ag2/threshold_comparison.png)

Against the validation disagreement gate, the frozen-adapter learned threshold wins/ties/loses on **2/0/5 classification tasks** and **1/1/5 regression tasks**. The joint version is **4/0/3** and **1/0/6**, respectively. Its classification GMR is 0.49% lower than the validation rule, while its mean classification error is 0.196 percentage points higher. Its regression GMR is 4.91% higher. These are descriptive task means, not significance tests.

Compared with extra training without a gate, the joint threshold wins on no classification task, ties five and loses two. In regression its arithmetic mean is slightly lower, but its GMR is slightly higher. Thus the joint variant does not establish a consistent gain over the extra-training control either.

## What the learned gates actually do

Counts below pool test evaluations across seeds; they are not counts of unique people/cases. Classification helps/harm counts are wrong-to-right/right-to-wrong changes relative to retrieval. Regression counts compare squared error with retrieval. [1]

| Method | Classification applied / total | Classification helps / harms | Regression applied / total | Regression helps / harms |
| --- | ---: | ---: | ---: | ---: |
| Original full adapter | 2,193 / 2,193 | 25 / 24 | 6,339 / 6,339 | 3,030 / 3,309 |
| Validation disagreement threshold | 104 / 2,193 | 13 / 4 | 4,735 / 6,339 | 2,390 / 2,345 |
| Learned threshold, frozen adapter | 1,563 / 2,193 | 21 / 20 | 6,249 / 6,339 | 2,983 / 3,266 |
| Learned threshold + adapter | 1,261 / 2,193 | 25 / 18 | 6,079 / 6,339 | 3,015 / 3,062 |

The frozen learned threshold keeps far more corrections than the validation rule. It changes classification correctness on fewer examples than the original adapter, but the removed fixes and harmful flips cancel: all seven task-mean error rates remain unchanged. Regression remains almost always-adapt: 6,249/6,339 test evaluations (98.58%). It reduces wrong-direction updates only from 2,616 to 2,576 and harmful overshoots from 518 to 515. The validation gate reduces these to 1,969 and 339.

For the joint version, the selected checkpoints' own full proposals have mean classification error **9.576%** and regression RMSE **0.348610**. Applying their gates changes these to **9.359%** and **0.347770**. This isolates the immediate effect of routing at the selected checkpoint. The classification improvement comes entirely from Zebra's task mean; the other six task means are unchanged by forcing this gate open. The gate modestly helps its own proposals overall, but those proposals are not consistently better than the original adapter.

The joint version selects epoch zero in **11/21 classification** and **6/21 regression** runs. Those selected thresholds are initialized cutoffs, not evidence of successful threshold learning. The gate is selective rather than always on/off in many other runs; a selection pattern by itself does not establish useful discrimination.

## Per-dataset results

Three-seed test means, lower is better. Classification entries are error rates, not percentages. All ten methods and individual seeds are available in the evidence tables. [1]

| Classification dataset | Original | Validation threshold | Learned, frozen adapter | Learned + adapter |
| --- | ---: | ---: | ---: | ---: |
| Balance | 0.053333 | 0.056000 | 0.053333 | 0.050667 |
| Breast Cancer | 0.029240 | 0.020468 | 0.029240 | 0.032164 |
| Digits | 0.021296 | 0.020370 | 0.021296 | 0.021296 |
| Iris | 0.033333 | 0.055556 | 0.033333 | 0.033333 |
| Wine | 0.074074 | 0.064815 | 0.074074 | 0.055556 |
| Zebra | 0.181818 | 0.151515 | 0.181818 | 0.196970 |
| Zebra Special | 0.303030 | 0.272727 | 0.303030 | 0.265152 |

| Regression dataset | Original | Validation threshold | Learned, frozen adapter | Learned + adapter |
| --- | ---: | ---: | ---: | ---: |
| Airfoil | 0.208094 | 0.208094 | 0.208094 | 0.209383 |
| Body Fat | 0.627485 | 0.623705 | 0.627206 | 0.626016 |
| Car | 0.222880 | 0.219538 | 0.222880 | 0.289201 |
| Diabetes | 0.768861 | 0.763774 | 0.768766 | 0.764931 |
| Energy Efficiency | 0.047689 | 0.047262 | 0.047689 | 0.048169 |
| Student Performance | 0.441026 | 0.443569 | 0.441026 | 0.438263 |
| Yacht | 0.056331 | 0.056027 | 0.056331 | 0.058424 |

Car seed 42's joint threshold applies every correction and selects epoch 41, with test RMSE **0.424215**, the same result as the earlier extra-training control. It remains in every aggregate; the learned cutoff did not protect against this failure.

## Interpretation

The existing validation gate already fits one decision parameter by comparing a small set of candidates. Gradient descent makes the cutoff continuous and changes the fitting data/objective; it does not introduce new neighborhood evidence or a more expressive decision rule. Simplicity alone is therefore not a reason to expect the new version to win.

This experiment changes both the threshold-fitting procedure and its supervision source. It does not separately identify the effect of gradient optimization versus fitting on training rather than validation examples.

The observed training feedback is informative. In five frozen classification runs, retrieval and adaptation have identical 0/1 costs for every training example, so this gate objective has zero gradient. Many other training runs favor broad use of adaptation. Both source components have already trained on these examples, even though self-retrieval is excluded; leave-one-out retrieval does not make the routing costs out-of-sample. However, this study does not prove that training optimism alone explains the result: soft-versus-hard routing, a monotone cutoff's limited expressiveness, checkpoint selection, and previously inspected test splits also matter.

**Recommendation:** keep the validation disagreement rule as the candidate for confirmation. If pursuing trainable routing, test a fixed protocol using held-out or cross-fitted branch costs, with data excluded from fitting the source predictor for each routing target. That tests the quality of gate supervision. It should precede another increase in gate complexity. The old adapter remains the default; these exploratory results do not justify a default change.

## Verification, sources and reproduction

- All 84 new units completed. The comparison contains 420 unique method/task/seed rows, 140 task/method means, and no missing units.
- All **252 saved train/validation/test prediction sets** reproduce their recorded error, application rate and helpful/harmful correction counts. Saved targets and retrieval outputs match the reference, decisions match the learned scalar cutoff, and every final output is exactly one of the two branches. Frozen-adapter full proposals also match the original validation/test proposals exactly in all 84 split checks.
- Nine model/training tests plus the eight prior gate tests passed. A further regression test for plotting a single task family passed after fixing that reporting edge case. Tests cover one-scalar learning, detached cost gradients, adapter learning while skipped, constant disagreement, no validation-label influence on the frozen result, mandatory recording, and exact interruption/resume recovery for both new arms.
- Reinvoking the full run command loaded all 84 completed units without training again. Resume states are retained. The shared-core manifest check passed; no shared-core or Table 1 implementation was changed.
- Training, checkpoints, predictions and curves are under `runs/t1/ag2/threshold_v1/`. Generated summaries read recorded units only and live under this worktree's `results/`. No remote weights, new retrieval fits or new MLP/TabM benchmarks were used.
- The 14 tasks and seeds 40-42 have already been inspected in earlier studies. Hyperparameters were fixed before this follow-up's test results; there is no claim of independent confirmation. Validation checkpointing in the joint arm is distinct from the validation-free threshold fitting in the frozen arm.

**[1] Recorded comparison evidence:** [full JSON](../results/adaptation_improvement/ag2/comparison.json), [individual runs and learned cutoffs](../results/adaptation_improvement/ag2/runs.csv), [task means](../results/adaptation_improvement/ag2/tasks.csv), [family comparisons](../results/adaptation_improvement/ag2/families.csv). Produced by [the threshold driver](../tools/t1_adaptation_threshold.py) from `ag2`, with original methods read from `ag1` and source identity checked against `am1`.

**[2] Prior experiment and implementation:** [apply-or-skip report](ADAPTATION_GATE_LOCAL_2026-11-25.md), [new protocol](../docs/ADAPTATION_THRESHOLD_EXPERIMENT.md), [one-parameter model](../model/t1/disagreement_threshold.py), [tests](../tests/test_t1_disagreement_threshold.py).

From the mini-project launcher, using the configured shared uv environment:

```text
uv run python "adaptation improvement/code/run_threshold_experiment.py" --study ag2
uv run python "adaptation improvement/code/run_threshold_experiment.py" --study ag2 --mode summarize
```

Runtime logs: `adaptation improvement/logs/local_ag2_classification.log` and `local_ag2_regression.log` in the mini-project. The local torch build emits an XPU-availability warning while executing on CPU; both workers exited successfully.
