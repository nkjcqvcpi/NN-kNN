## 1. Executive Summary: What Is the Actual Performance Gap?

### Main Finding

NN-kNN successfully improves upon conventional k-nearest neighbors by learning feature importance and case-specific retrieval biases. However, its performance still falls behind strong tabular baselines, particularly CatBoost.

The main issue is not simply insufficient hyperparameter tuning. The experiments suggest three major bottlenecks:

1. **Retrieval limitations:** Learned similarity does not always identify genuinely useful neighbors, particularly on complex tabular datasets.
2. **Insufficient gains from additional components:** Reuse adapters and joint training have not demonstrated reliable improvements over tuned retrieval.
3. **Weak generalization of model selection:** Several improvements observed during development disappear on held-out datasets.

Case-base maintenance remains an additional open problem, but it was not included in the final confirmed model F.

### Table 1. Final Pre-Registered Confirmation Results

*Source: October 20 Progress Report, Table 1; 30 datasets.*

| Comparison | GMR [95% CI] | Holm-adjusted p | Conclusion |
|---|---|---|---|
| F vs. Retrieval-only | 0.974 [0.936, 1.014] | 0.2923 | No detectable reuse gain |
| F vs. Alternating Default | 0.996 [0.990, 1.003] | 0.2923 | No detectable training-strategy gain |
| F vs. CatBoost | 1.135 [1.004, 1.283] | 0.6984 | Non-inferiority not shown; F has higher error |
| F vs. TabM | 0.987 [0.855, 1.138] | 0.2616 | Non-inferiority not established |

**Interpretation:** GMR is the geometric mean ratio of prediction errors. Values below 1 favor F; values above 1 favor the reference model. The two baseline comparisons test a 10% non-inferiority margin.

F has approximately 13.5% higher pooled error than tuned CatBoost. Although its point estimate against TabM is slightly below 1, the confidence interval is too wide to establish non-inferiority.

### Table 2. Average Error Rankings Across Six Methods

*Source: October 20 Progress Report, Table 5. Lower is better.*

| Method | All 30 Tasks | Legacy 20 | Transfer 10 |
|---|---:|---:|---:|
| CatBoost | 1.87 | 2.00 | 1.60 |
| TabM | 2.60 | 2.90 | 2.00 |
| NN-kNN + Reuse (F) | 3.50 | 3.35 | 3.80 |
| MLP | 3.90 | 3.95 | 3.80 |
| NN-kNN Retrieval-only | 4.10 | 4.25 | 3.80 |
| Classical kNN | 5.03 | 4.55 | 6.00 |

**Key Insight**

The learned retrieval mechanism improves substantially over ordinary kNN, but the final CBR system still ranks behind CatBoost and TabM.

**Research Question:** Why does a learned nearest-neighbor model fail to match methods that directly learn the prediction function?

---

## 2. Reason 1: Limitations of the Learned Retrieval Core

### 2.1 Learned Distance Is Better Than Fixed Distance, but Not Enough

NN-kNN learns which features matter and how much to trust individual cases.

However, the final prediction still depends heavily on the neighbors selected by the distance function.

This can be problematic when the dataset contains:

- High-dimensional or irrelevant features.
- Complex nonlinear feature interactions.
- Categorical variables with irregular similarity relationships.
- Sparse regions with insufficient relevant examples.

CatBoost learns combinations of feature splits, while TabM learns nonlinear representations. NN-kNN must express enough of the predictive structure through its learned distance and neighbor weighting.

This explanation is theoretically plausible, but the current experiments have not directly isolated feature interactions as the causal reason for CatBoost's advantage.

### Table 3. Retrieval-only NN-kNN vs. Baselines on Legacy Tasks

*Source: October 13–20 Weekly Report, Table 30; 20 tasks.*

| Reference | NN-kNN Better | NN-kNN Worse | GMR | Wilcoxon p |
|---|---:|---:|---:|---:|
| kNN | 12 | 8 | 0.788 | 0.083 |
| CatBoost | 7 | 13 | 1.280 | 0.048 |
| MLP | 10 | 10 | 0.898 | 0.729 |
| TabM | 7 | 13 | 0.944 | 0.430 |

**Interpretation**

- NN-kNN's error ratio against conventional kNN is 0.788, corresponding to a 21.2% lower geometric-mean error.
- Against CatBoost, the ratio is 1.280, indicating substantially higher error.
- Comparisons with MLP and TabM are mixed, with no clear difference established by the reported Wilcoxon tests.

### Table 4. Retrieval-only Performance on Difficult Transfer Datasets

*Source: October 13–20 Weekly Report, Table 27. The QSAR-TID-11 row was interim at the weekly-report snapshot.*

| Dataset | NN-kNN / kNN | NN-kNN / CatBoost | NN-kNN / TabM |
|---|---:|---:|---:|
| Splice | 0.403 | 0.955 | 0.976 |
| Students Dropout | 0.851 | 1.067 | 1.118 |
| In-Vehicle Coupon | 0.992 | 1.186 | 1.155 |
| Churn | 0.628 | 1.530 | 1.144 |
| MIC | 0.911 | 1.084 | 1.079 |
| Superconductivity | 0.994 | 1.099 | 1.071 |
| Miami Housing | 0.945 | 1.164 | 1.109 |
| Diamonds | 0.677 | 1.034 | 0.932 |
| Physiochemical Protein | 0.935 | 1.000 | 1.005 |
| QSAR-TID-11 (interim) | 0.927 | 1.075 | 1.061 |
| **Geometric Mean** | **0.800** | **1.110** | **1.063** |

**Important observations**

- NN-kNN improves on classical kNN across all ten transfer datasets in this grid.
- CatBoost performs especially well on Churn, In-Vehicle Coupon, and Miami Housing.
- The NN-kNN/CatBoost error ratio reaches 1.530 on Churn.

**Hypothesis**

The learned distance function may not sufficiently suppress irrelevant features or capture interactions between important features. This hypothesis should be tested by measuring feature relevance, changing the retrieval geometry, and comparing results on datasets with different feature structures.

---

## 3. Reason 2: Retrieved Neighbors Sometimes Do Not Contain the Correct Class

### Main Finding — Supported by Diagnostics

This is one of the most concrete weaknesses identified in the experiments.

NN-kNN classification uses weighted votes over retrieved cases. If the correct class is absent from the retrieved neighborhood, the resulting predicted probability for that class can be zero.

Such predictions are particularly damaging to log loss, which strongly penalizes confident errors.

### Table 5. Near-Zero Probability Failures in Retrieval-only Classification

*Source: October 13–20 Weekly Report, Table 28.*

| Dataset | NN-kNN Log Loss | CatBoost Log Loss | Test Rows With p(True Class) ≤ 1e-6 | Share of NN-kNN Log Loss From These Rows |
|---|---:|---:|---:|---:|
| Splice | 0.2275 | 0.1225 | 0.40% | 47.7% |
| Students Dropout | 3.1850 | 0.5637 | 10.54% | 91.2% |
| In-Vehicle Coupon | 0.6049 | 0.5370 | 0.03% | 1.3% |
| Churn | 0.5573 | 0.1606 | 1.49% | 71.7% |
| MIC | 3.0106 | 0.4717 | 10.79% | 99.0% |

CatBoost had 0.00% of test rows with near-zero correct-class probability under this threshold on all five datasets.

### Interpretation

On Students Dropout:

- Approximately 10.54% of test samples assign almost zero probability to the correct class.
- Those samples account for 91.2% of NN-kNN's total log loss.

On MIC:

- 10.79% of samples exhibit the same failure.
- These samples contribute 99.0% of NN-kNN's log loss.

This suggests that a small subset of poor neighbor-based predictions dominates the probability-prediction error.

**Key Insight**

Improving the distance function is not sufficient if the retrieved neighborhood fails to contain useful information about the correct class.

### Proposed Experiments

1. Measure correct-class coverage within the top-k neighbors.
2. Evaluate the effect of different values of k; the planned H4 neighbor-count study was not executed.
3. Test class-aware retrieval or neighborhood expansion when class coverage is weak.
4. Compare probability smoothing against the current pure neighbor-voting mechanism.
5. Break down performance by class imbalance, minority-class recall, and local neighborhood density.

These are proposed experiments, not established improvements.

---

## 4. Reason 3: The Reuse Adapter Does Not Reliably Fix Weak Retrieval

### 4.1 The Adapter Improves Some Predictions, but Not Consistently

The Reuse Adapter is intended to correct the prediction produced by the retrieved cases.

Its inputs emphasize the differences between the query and retrieved representations, together with the retrieval prediction. For the documented nominal reuse implementation, the input is represented as `[Δz, Δu, p0]`.

A potential limitation is that the adapter may not have enough information to reconstruct complex predictive relationships when retrieval has already selected uninformative neighbors.

This limitation is a hypothesis. The experiments establish the adapter's measured gains and failures, not their architectural cause.

### Table 6. Reuse Effect by Dataset Group

*Source: October 20 Progress Report, Tables 1 and 3.*

| Dataset Group | Tasks | F / Retrieval-only GMR | 95% CI |
|---|---:|---:|---|
| All tasks | 30 | 0.974 | [0.936, 1.014] |
| Legacy tasks | 20 | 0.944 | [0.882, 1.012] |
| Difficult transfer tasks | 10 | 0.995 | [0.972, 1.019] |

**Interpretation**

The point estimates suggest a larger improvement on legacy datasets than on the difficult transfer datasets.

On the transfer datasets, the pooled ratio is 0.995, effectively showing little average improvement.

None of these confidence intervals excludes 1.

### Table 7. Selected Per-Dataset Effects of Reuse

*Source: October 20 Progress Report, Table 6.*

| Dataset | F / Retrieval-only Error Ratio | Interpretation |
|---|---:|---|
| Yacht | 0.553 | Large improvement |
| Anneal | 0.601 | Large improvement |
| Zebra | 0.817 | Improvement |
| Balance | 0.825 | Improvement |
| Wine (classification) | 0.881 | Improvement |
| Students Dropout | 0.963 | Small improvement |
| Diamonds | 0.995 | Almost unchanged |
| MIC | 1.029 | Worse |
| Churn | 1.064 | Worse |
| Splice | 1.097 | Worse |

**Key Insight**

Reuse can work very well on individual tasks, but there is no reliable evidence that it generalizes across more challenging datasets.

### 4.2 Better Calibration Does Not Guarantee Better Classification Decisions

The adapter improves calibrated probabilities, but it also changes some correct predictions into incorrect ones.

### Table 8. Calibration Improvement and Prediction Flips

*Source: October 20 Progress Report, Table 4.*

| Scope | Before-Adapter ECE | Calibrated ECE | Wrong → Correct | Correct → Wrong | Flip Ratio |
|---|---:|---:|---:|---:|---:|
| All tasks | 0.0572 | 0.0343 | 5,681 | 4,842 | 1.17 |
| Legacy | 0.0599 | 0.0415 | 523 | 429 | 1.22 |
| Transfer | 0.0469 | 0.0178 | 5,158 | 4,413 | 1.17 |

The pre-registered target for the beneficial-to-harmful flip ratio was at least 2.0.

The calibrated ECE was measured using the project's validation-fitted temperature procedure.

**Interpretation**

The Reuse Adapter improves probability calibration, but its changes to predicted labels are not sufficiently selective to meet the project's required benefit-to-harm threshold.

### Table 9. Does Reuse Close the Probability-Prediction Gap?

*Source: October 20 Progress Report, Table 7; mean log loss over 10 seeds.*

| Dataset | Retrieval-only | F | CatBoost | TabM |
|---|---:|---:|---:|---:|
| Churn | 0.4203 | 0.2973 | 0.1606 | 0.2012 |
| MIC | 2.0169 | 0.5671 | 0.4717 | 0.4674 |

**Interpretation**

Reuse substantially improves log loss compared with Retrieval-only on both datasets.

However, F still has higher log loss than CatBoost and TabM.

This shows that the adapter can compensate for some probability errors without fully closing the gap to stronger baselines.

**Research Direction**

The next step should focus on when the adapter is allowed to make a large correction, rather than automatically increasing its complexity.

---

## 5. Reason 4: Joint Training Does Not Reliably Improve the Final Model

### 5.1 Early Improvements Were Partly Explained by a Training Mismatch

Joint Training was designed to make Retrieval and Reuse cooperate during training.

An early experiment suggested a substantial improvement. However, a batch-size mismatch made that result unreliable.

After correcting the mismatch, the observed improvement became much smaller.

### Table 10. Joint Training Before and After the Batch-Size Correction

*Source: October 13 Progress Report, Table 32.*

| Configuration | Median Error Change vs. Alternating | Tasks Better / Worse | Wilcoxon p |
|---|---:|---|---:|
| Batch size 32 (mismatched) | -11.6% | 13 / 3 | 0.008 |
| Batch size 64 (corrected) | -1.8% | 11 / 5 | 0.323 |

**Interpretation**

The original apparent 11.6% improvement cannot be treated as valid evidence of joint training's superiority.

At the corrected batch size, the improvement was approximately 1.8% and was not statistically convincing.

### Table 11. Final Confirmation of the Joint/Alternating Comparison

*Source: October 20 Progress Report, Tables 1 and 2.*

| Measure | Result | Criterion | Passed? |
|---|---:|---:|---|
| F vs. Alternating Default GMR | 0.996 | Improvement required | No |
| Pre-adaptation error ratio | 1.003 | ≤ 1.02 | Yes |
| Input-space locality ratio | 1.040 | ≤ 1.10 | Yes |
| Correction-size ratio | 1.028 | ≤ 1.25 | Yes |

The final E2 comparison includes schedule and loss-setting differences, so it does not isolate joint training alone.

### 5.2 Joint Optimization Can Conceal Retrieval Degradation

An earlier development experiment showed that the adapted final result could improve while the underlying retrieval prediction degraded severely.

### Table 12. Superconductivity: Retrieval Degradation Hidden by Adaptation

*Source: October 13 Progress Report, Table 44; one development seed, one fixed raw core, 5,000 training cases.*

| Method | RMSE Before Adaptation | RMSE After Adaptation | Mean Correction Size |
|---|---:|---:|---:|
| Retrieval-only | 0.5215 | — | — |
| Alternating Default | 1.3561 | 0.4296 | 0.9361 |
| F0 Joint | 1.3719 | 0.4051 | 0.9088 |

**Interpretation**

In this single experiment:

- Joint training worsened retrieval-only RMSE from 0.5215 to 1.3719.
- The adapter then reduced final RMSE to 0.4051.

This indicates that the final adapted prediction can conceal degradation in the retrieval system.

However, this is a single development experiment, not evidence that the final model F generally damages retrieval. The final confirmation's average pre-adaptation guard passed.

### Additional Observation

In the final confirmation, 119 of 400 analyzed F runs (approximately 30%, excluding diabetes) selected a checkpoint before any retrieval or joint update occurred.

Consequently, in those runs, the selected retrieval core remained unchanged from the initial Retrieval-only model.

**Key Insight**

Further training does not necessarily improve retrieval, even when the final adapted prediction improves.

---

## 6. Reason 5: More Complex Encoders Do Not Necessarily Learn Better Retrieval Geometry

### 6.1 Raw Features Often Outperform Learned Encoders

The project evaluated raw features, a shared MLP encoder, and several EMA momentum encoders.

The goal of EMA was to stabilize the stored-case representation during learning.

### Table 13. Raw Features vs. Shared Encoder

*Source: October 13 Progress Report, Table 25; 13 tasks, median relative test-error change.*

| Training Setting | Raw vs. Shared Encoder | Tasks Raw Better / Worse | Wilcoxon p |
|---|---:|---|---:|
| Retrieval-only | -29.0% | 10 / 3 | 0.013 |
| Frozen-core Reuse | -31.7% | 11 / 2 | 0.003 |
| Alternating | -40.6% | 11 / 2 | 0.002 |
| Joint | -26.7% | 10 / 3 | 0.003 |

Negative changes indicate lower error for raw features.

**Interpretation**

Raw features outperform the shared encoder on most of the tested tasks, across all four training settings.

This suggests that the learned representation does not automatically improve the distance structure required by retrieval.

### 6.2 EMA Improves Representation Stability but Not Prediction Accuracy

### Table 14. EMA Encoder Stability

*Source: October 13 Progress Report, Table 27.*

| Encoder | Median Drift per Epoch | Median Neighbor Churn | Tasks With Lower Drift Than Shared |
|---|---:|---:|---:|
| Shared | 0.0495 | 0.1638 | — |
| EMA 0.90 | 0.0253 | 0.1451 | 13 / 13 |
| EMA 0.99 | 0.0087 | 0.1705 | 13 / 13 |
| EMA 0.999 | 0.0026 | 0.1736 | 13 / 13 |

### Table 15. EMA Accuracy on Tuned MLP-Encoder Tasks

*Source: October 13 Progress Report, Table 28; six tasks.*

| Setting | EMA Arm | Median Test-Error Change vs. Shared | Wilcoxon p |
|---|---|---:|---:|
| Retrieval-only | EMA 0.90 | +17.0% | 0.219 |
| Retrieval-only | EMA 0.99 | +0.4% | 0.688 |
| Joint Training | EMA 0.90 | +12.2% | 0.156 |
| Joint Training | EMA 0.99 | +13.1% | 0.156 |

**Key Insight**

A more stable embedding is not necessarily a more predictive embedding.

EMA substantially reduces representation drift, but the tested EMA variants do not establish a prediction-performance advantage.

**Hypothesis**

The encoder objectives may prioritize representation consistency without sufficiently improving label-relevant neighborhood quality.

A useful follow-up would measure whether better representation stability increases correct-class coverage, decreases within-class distances, or improves test accuracy.

---

## 7. Reason 6: Hyperparameter Selection Shows Limited Generalization

### 7.1 More Search Does Not Necessarily Improve Held-Out Performance

The project tested multiple changes to schedule selection, adapter capacity, regularization, stopping rules, and calibration.

Only two tuning decisions were adopted:

- Per-task recipe selection (H6).
- Classification score scaling with kappa = 10 (H2c), accepted as non-inferior and useful for calibration.

### Table 16. Final Tuning Decisions

*Source: October 20 Progress Report, Table 8.*

| Block | Tested Change | Held-Out Result | Decision |
|---|---|---|---|
| H6 | Per-task selection | GMR 0.932 → 0.891 | Adopted |
| H1 | Retrieval guard | 0.895 → 0.904 | Rejected |
| H2a | Score-scale candidates | Paired log ratio -0.004 | Not adopted |
| H2b | Logit-residual output | Paired log ratio approximately 0 | Not adopted |
| H2c | Classification kappa = 10 | 0.842 → 0.822 | Adopted as non-inferior |
| H5 | Budget and stopping | 0.898 → 0.900 | Rejected |
| H3 | Adapter capacity and regularization | 0.898 → 0.906 | Rejected |
| H4 | Number of neighbors | Not executed | No evidence |

The GMR values are block-specific comparisons against the alternating default, not comparisons with CatBoost.

### 7.2 Classification Tuning Shows Validation-to-Test Deterioration

### Table 17. Validation vs. Held-Out Performance

*Source: October 13–20 Weekly Report, Table 6. Changes are mean log error ratios; negative is better.*

| Tuning Block | Task Type | Validation Change | Held-Out Change |
|---|---|---:|---:|
| H3 | Classification | -0.0724 | +0.0381 |
| H3 | Regression | -0.0108 | -0.0138 |
| H5 | Classification | -0.1222 | +0.0391 |
| H5 | Regression | -0.0255 | -0.0796 |

**Interpretation**

Classification candidates frequently looked better on the validation data used to select them but performed worse on held-out data.

For H5 classification, the validation log error ratio improved by 0.1222 while the held-out ratio deteriorated by 0.0391.

The same deterioration pattern was not observed for the regression rows shown here.

### 7.3 Many Selected Configurations Are Near-Ties

### Table 18. Reliability of Per-Task Configuration Selection

*Source: October 13–20 Weekly Report, Table 12.*

| Selection Diagnostic | Result |
|---|---|
| Tuned tasks | 18 |
| Tasks where selected F and runner-up differ by at most one paired SE | 17 / 18 |
| Classification tasks where the winner leads by exactly one validation error | 5 / 9 |
| Classification tasks where equal error counts require validation loss to break the tie | 4 / 9 |
| Distinct selected recipes across the 18 tasks | 10 |

**Key Insight**

Many selected configurations have very similar validation performance. Therefore, a configuration appearing to be the best on development data is not necessarily reliably better on unseen data.

### 7.4 Transfer Tasks Were Not Individually Tuned for Reuse

The final procedure uses task-specific recipes for 18 legacy tasks. The other 12 tasks, including all ten difficult transfer tasks, use F_global.

### Table 19. Reuse Benefit by Tuning Scope

*Source: October 20 Progress Report, Table 3.*

| Scope | Tasks | F / Retrieval-only GMR |
|---|---:|---:|
| Legacy (mostly individually tuned) | 20 | 0.944 |
| Transfer (shared F_global) | 10 | 0.995 |

The difference is descriptive rather than causal. It is unclear whether the weaker transfer effect comes from the shared recipe, the greater difficulty of the datasets, or both.

**Proposed Test**

Compare F_global with carefully tuned reuse settings on separate development splits of transfer datasets, without using the reserved confirmation data for selection.

---

## 8. Reason 7: Case-Base Maintenance Is Not Yet a Proven Solution

### Important Distinction

Case-base maintenance was not part of the final confirmed configuration F.

Therefore, the final F-versus-CatBoost gap cannot be attributed to the maintenance algorithm.

However, the maintenance review identified limitations that would matter for a complete Neural CBR system.

### 8.1 Provenance Quality Q Is Strongly Related to Retrieval Exposure

The quality score Q rewards cases participating in successful predictions and penalizes those participating in unsuccessful predictions.

The review found that Q often behaves more like a measure of how frequently a case is retrieved than a direct measure of its marginal contribution to prediction quality.

### Table 20. Case-Base Maintenance Diagnostics

*Source: October 20 Progress Report, Table 9; development seeds, 50% capacity.*

| Dataset | Stored Cases | Never Retrieved | Spearman(Q, Activation) | Q Evictions Decided by Case ID |
|---|---:|---:|---:|---:|
| Abalone | 2,505 | 98% | 0.64 | 99% |
| Breast Cancer | 341 | 84% | 1.00 | 100% |
| Wine (classification) | 106 | 77% | 0.97 | 100% |
| Yacht | 184 | 66% | 0.92 | 100% |
| Anneal | 448 | 57% | 1.00 | 75% |
| Airfoil | 901 | 3% | 0.99 | 0% |
| Diabetes | 264 | 17% | 0.13 | 42% |

Across the full diagnostic set, 14 of 18 tasks had Spearman(Q, activation) of at least 0.90.

**Interpretation**

On several datasets, a large majority of stored cases are never retrieved during the audit.

When Q must choose between these cases, they often have indistinguishable quality scores, and the default tie-break depends on case ID.

This is undesirable because removal decisions may depend on row order rather than actual usefulness.

### 8.2 Existing Retention Experiments Do Not Show Consistent Gains

### Table 21. Retention Policies on Energy

*Source: October 13 Progress Report, Table 9; mean test RMSE, K = 0.75.*

| Policy | Active Cases | Mean RMSE |
|---|---:|---:|
| Full Memory | 60 | 0.3460 |
| Random | 45 | 0.3630 |
| Q | 45 | 0.3877 |
| B | 45 | 0.3488 |
| Coverage/Reachability | 45 | 0.3721 |
| Removal Influence | 45 | 0.3560 |

Lower RMSE is better.

In this small experiment, Q is worse than random removal and substantially worse than the full memory.

However, the experiment has only three seeds and carries the reproduction caveats documented in the October 13 report. It should not be generalized as evidence that Q always underperforms.

### Table 22. Retention Policies on Digits

*Source: October 13 Progress Report, Table 10; mean classification accuracy, K = 0.5.*

| Policy | Active Cases | Mean Accuracy |
|---|---:|---:|
| Full Memory | 1,077 | 0.9778 |
| Random | 538 | 0.9704 |
| Q | 538 | 0.9741 |
| B | 538 | 0.9741 |

On Digits, Q and B are slightly better than random, but neither matches full memory.

This reinforces the conclusion that the impact of case selection is dataset-dependent.

### 8.3 Maintenance Has Not Been Properly Tested Under Difficult Conditions

The review identified 13 supported design concerns and found that important settings were missing from the maintenance evaluation.

These include:

- Label noise and corrupted cases.
- Near-duplicate and redundant cases.
- Rare or underrepresented classes.
- Large case bases.
- Distribution shift and genuinely dynamic changes to memory.

### Table 23. Synthetic Label-Noise Diagnostic

*Source: October 20 Progress Report, Section 5.*

| Setting | Result |
|---|---|
| Artificially flipped labels | 15% |
| Case-base capacity | 75% |
| Correlation of Q with exposure | 0.86–0.99 |
| Corrupted cases retained by Q | 58–60% |
| Corrupted cases retained by random selection | 75% |

Q removes more corrupted cases than random selection in this synthetic diagnostic, but it remains highly correlated with retrieval exposure.

**Key Insight**

The current maintenance mechanism has some useful behavior under label noise, but its ability to measure actual marginal case utility has not been established.

The planned formal Step G comparison was not yet executed as of the October 20 report.

---

## 9. Experimental Limitations That Affect the Interpretation

The results are useful, but several constraints prevent stronger conclusions about NN-kNN's general capabilities.

### Table 24. Important Evaluation Limitations

| Limitation | Recorded Evidence | Implication |
|---|---|---|
| Transfer dataset training cap | Maximum 3,000 training rows; binds on 5 of 10 transfer tasks | Performance at larger data scales remains unknown |
| Unequal compute budgets | Five model families used 32 candidates per task, but not equal training compute | Equal candidate counts do not establish compute fairness |
| High NN-kNN compute cost | 92.7% of recorded transfer-grid job-hours were used by NN-kNN | Efficiency and scalability require separate study |
| Large legacy tasks unfinished | Two large NN-kNN legacy-grid jobs did not complete | Retrieval-only results on those tasks remain unknown |
| Untuned transfer reuse | All ten transfer tasks used F_global | Limits conclusions about task-specific adaptation |
| Close hyperparameter choices | 17/18 tuned tasks had near-tied winners | Model-selection uncertainty is substantial |
| No confirmed maintenance comparison | Step G had not run | Maintenance's accuracy benefit remains unknown |

*Sources: October 13–20 Weekly Report, Tables 26, 29, 30; October 20 Progress Report, Tables 8–10.*

These limitations do not invalidate the 30-task confirmation, but they constrain what can be inferred about larger-scale deployments and fully optimized versions of the method.

---

## 10. Recommended Research Priorities

### Table 25. Prioritized Next Experiments

| Priority | Problem | Supporting Result | Recommended Experiment |
|---|---|---|---|
| P1 — Highest | Correct class missing from neighbors | Students Dropout: 91.2% of retrieval log loss from near-zero-probability rows; MIC: 99.0% | Measure correct-class coverage at different k; test class-aware retrieval and probability smoothing |
| P2 — High | Retrieval geometry underperforms trees | Churn: NN-kNN/CatBoost 1.530; Coupon: 1.186; Miami Housing: 1.164 | Analyze feature relevance, local distances, categorical structure, and nonlinear interactions |
| P3 — High | Reuse does not generalize | E1 transfer GMR = 0.995 | Test uncertainty-aware correction and compare frozen-core vs. joint reuse |
| P4 — Medium | Encoder training does not improve accuracy | Raw retrieval: -29.0% median error vs. shared encoder | Compare encoder objectives using neighborhood quality and task accuracy |
| P5 — Medium | Training improvements may conceal retrieval degradation | Superconductivity pre-adaptation RMSE 1.3719 vs. retrieval-only 0.5215 | Monitor pre-adaptation retrieval quality throughout training |
| P6 — Medium | Hyperparameter selection is unstable | H5 classification: -0.1222 validation change, +0.0391 held-out change | Repeat selection on independent development splits |
| P7 — Later | Maintenance may optimize exposure rather than utility | 14/18 tasks have Q-exposure correlation ≥ 0.90 | Evaluate leave-one-case-out utility and noisy, redundant, imbalanced memories |

The priority order is a research recommendation, not a result of a controlled comparison between the proposed interventions.

### Suggested Experimental Sequence

**Stage 1 — Diagnose Retrieval**

Measure correct-class neighbor coverage, nearest-neighbor label consistency, neighborhood purity, and sensitivity to feature removal. Determine which datasets and data properties are associated with retrieval failures.

**Stage 2 — Strengthen Retrieval**

Evaluate alternative similarity functions and representations, as well as neighborhood size. Compare improvements against the current Retrieval-only core before introducing the adapter.

**Stage 3 — Re-evaluate Reuse**

Determine whether improved retrieval makes the adapter more valuable. Test correction mechanisms that depend on retrieval confidence and preserve strong predictions.

**Stage 4 — Revisit Joint Training**

Once the retrieval core and adapter both have individually demonstrated benefits, test whether joint optimization adds value under matched budgets.

**Stage 5 — Evaluate Maintenance**

Complete the planned policy comparison with appropriate controls, then conduct stress tests on noisy, redundant, imbalanced, and larger datasets.

---

## 11. Final Conclusion

The central issue is not whether NN-kNN can improve classical nearest-neighbor prediction. The experiments show that it can.

The unresolved question is whether learning a similarity metric and predicting from retrieved cases can match the predictive power of strong tabular models.

The strongest current evidence suggests:

1. **Retrieval quality is the fundamental bottleneck.** NN-kNN improves on kNN but remains weaker than CatBoost on many challenging tasks.
2. **Neighbor coverage causes measurable probability-prediction failures.** A small number of extreme errors account for much of the retrieval-only log loss on certain datasets.
3. **Reuse provides task-specific gains but no reliable overall improvement.** It improves calibration but does not consistently improve classification decisions or close the baseline gap.
4. **Joint Training and EMA encoding do not establish additional predictive benefits.** More elaborate optimization is not automatically better.
5. **Hyperparameter selection is uncertain.** Several apparent validation improvements disappear on held-out splits.
6. **Case-Base Maintenance remains an open research problem.** It was not part of F, and its ability to improve performance has not yet been confirmed.

**Overall Research Direction**

The highest-value next step is to understand why the learned retrieval mechanism still fails to select sufficiently informative cases on difficult tabular datasets.

Improving the retrieval core and testing its failures directly should take priority over adding further complexity to the Neural CBR pipeline.

---

## Appendix A. Final Confirmed Dataset-Level Results

*Source: October 20 Progress Report, Table 6. All 30 tasks on final confirmation seeds. Ratios below 1 favor F. Classification F error is 1 − accuracy; regression F error is train-standardized RMSE.*

| Dataset | F Error | F / Retrieval | F / CatBoost | F / TabM |
|---|---:|---:|---:|---:|
| Iris | 0.0522 | 1.060 | 1.100 | 1.045 |
| Wine (classification) | 0.0407 | 0.881 | 1.419 | 1.556 |
| Breast Cancer | 0.0360 | 0.916 | 0.969 | 0.966 |
| Balance | 0.0480 | 0.825 | 0.535 | 1.191 |
| Digits | 0.0219 | 1.054 | 0.914 | 1.124 |
| Zebra | 0.1061 | 0.817 | 1.944 | 0.105 |
| Zebra Special | 0.2864 | 0.999 | 5.113 | 0.571 |
| Energy Efficiency | 0.0527 | 0.906 | 1.124 | 0.993 |
| Yacht | 0.0844 | 0.553 | 1.168 | 1.273 |
| Airfoil | 0.2174 | 0.964 | 0.986 | 0.938 |
| Student Performance | 0.4716 | 1.001 | 1.059 | 1.087 |
| Abalone | 0.6587 | 0.999 | 0.982 | 1.021 |
| Body Fat | 0.7109 | 1.050 | 1.079 | 0.953 |
| Diabetes | 0.7260 | 0.998 | 1.003 | 1.038 |
| Car | 0.2251 | 0.982 | 1.363 | 0.962 |
| Wine (regression) | 0.7409 | 1.002 | 1.016 | 0.981 |
| Anneal | 0.0087 | 0.601 | 0.911 | 0.797 |
| Credit-G | 0.2754 | 1.014 | 1.113 | 1.088 |
| Concrete | 0.3808 | 0.948 | 1.386 | 1.140 |
| Insurance | 0.3961 | 0.973 | 1.052 | 1.037 |
| Splice | 0.0419 | 1.097 | 1.048 | 1.070 |
| Students Dropout | 0.2355 | 0.963 | 1.028 | 1.077 |
| In-Vehicle Coupon | 0.3074 | 0.980 | 1.163 | 1.132 |
| Churn | 0.0743 | 1.064 | 1.628 | 1.217 |
| MIC | 0.1266 | 1.029 | 1.116 | 1.111 |
| Superconductivity | 0.3752 | 0.974 | 1.070 | 1.043 |
| Miami Housing | 0.3525 | 0.973 | 1.133 | 1.079 |
| Diamonds | 0.1719 | 0.995 | 1.029 | 0.928 |
| Physiochemical Protein | 0.6993 | 1.004 | 1.004 | 1.009 |
| QSAR-TID-11 | 0.5532 | 0.983 | 1.047 | 1.038 |

**Important:** This appendix contains final F results. The earlier difficult-transfer grid in Section 2 evaluates Retrieval-only NN-kNN. These are different models and evaluation analyses; their ratios should not be treated as directly interchangeable.
