# T1 prior classification and regression results

Status: table-level extraction completed 2026-09-12 from the archived IJCAI-2025 classification paper and the PI-supplied IJCAI-2026 regression author copy. This note identifies proposal-usable preliminary evidence and preserves the limits of the comparisons. **All results in this note use the older, pre-MCB NN-kNN architecture.**

## Source files

- Classification: `../resources/research/nnknn-classification/ijcai-2025-nnknn-classification.pdf`
- Regression: `../resources/research/nnknn-regression/ijcai-2026-nnknn-regression.pdf`

## Bottom line for T1

The papers support a calibrated claim that the **pre-MCB NN-kNN architecture** achieved **numerically competitive supervised classification and regression performance** in multiple reported comparisons while retaining a case-based prediction interface. They do not yet establish formal statistical equivalence to neural baselines, universal replacement, the effect of MCB on these benchmarks, or the proposed full retrieve-reuse-revise-retain cycle.

- In classification, the closest direct image comparisons are `0.688 (0.006)` for Conv + NN-kNN versus `0.689` for ConvNet on CIFAR-10, and `0.867 (0.003)` versus `0.875` on SVHN. Reported text results are also close: `0.368 (0.007)` for jointly trained Bi-LSTM + NN-kNN versus `0.376` for Bi-LSTM on SST-5; on SST-2, pretrained Bi-LSTM + NN-kNN reports `0.785 (0.001)` versus `0.776` for Bi-LSTM.
- In the regression paper's Table 1, selecting the lowest reported NN-kNN mean RMSE for each dataset gives a lower value than the MLP on 7 of 12 datasets, the same value at displayed precision on 1, and a value no more than 3.0% higher on the remaining 4. This is descriptive evidence across variants, not a prespecified model-selection result.
- The synthetic regression experiment exposes the exact T1 scientific need: the most accurate NN-kNN variant can retrieve poor explanatory neighbors, while locality-aware NN-kNN can remain close to MLP error and retrieve cases closer under the ground-truth metric. T1's coordination objective therefore addresses a limitation visible in the prior results rather than merely adding features to an already complete system.

## Classification paper

### Experimental protocol relevant to claim strength

The classification paper states that smaller datasets use 10-fold cross-validation and larger datasets use 10 iterations on predefined train-test splits. NN-kNN means and standard deviations are reported in parentheses. However, the neural-baseline rows do not report uncertainty. The paper also states that all models were trained until **testing accuracy** stopped improving for 40 epochs, that values of `k` and `w` were tried until the best performance was found, and that only results providing interesting insights were shown because of space. These choices make the tables useful preliminary evidence, but not a confirmatory equivalence test.

### Table 2 - Comparison between NN-kNN, NN-kNNO, and NNet

Accuracy; NN-kNN standard deviations appear in parentheses. For the two-row datasets, NN-kNNO and NNet are reported once in the source table.

| Dataset/configuration | NN-kNN `w=1` | NN-kNN `w=4` | `k` | NN-kNNO | NNet |
|---|---:|---:|---:|---:|---:|
| Iris | 0.980 (0.031) | 0.973 (0.044) | 5 | 0.966 | 0.987 |
| Zebra (a) | 0.945 (0.164) | 0.964 (0.073) | 1 | 0.918 | 0.782 |
| Zebra (a) | 0.800 (0.172) | 0.982 (0.036) | 5 | - | - |
| Zebra (b) | 0.764 (0.057) | 0.923 (0.041) | 1 | 0.968 | 0.673 |
| Zebra (b) | 0.718 (0.076) | 0.932 (0.047) | 5 | - | - |
| Wine | 0.922 (0.075) | 0.972 (0.038) | 1 | 0.994 | 0.836 |
| Wine | 0.838 (0.103) | 0.843 (0.110) | 5 | - | - |
| Breast Cancer | 0.986 (0.019) | 0.984 (0.018) | 5 | 0.959 | 0.951 |
| Balance | 0.942 (0.025) | 0.952 (0.032) | 5 | 0.944 | 0.994 |
| Digits | 0.983 (0.008) | 0.986 (0.010) | 5 | 0.988 | 0.983 |

Using the best displayed NN-kNN mean per dataset, NN-kNN exceeds the NNet point estimate on Zebra (a), Zebra (b), Wine, Breast Cancer, and Digits; it is 0.7 percentage points lower on Iris and 4.2 points lower on Balance. The Zebra datasets are synthetic diagnostic tasks, so their large gains should not be presented as broad real-world superiority.

### Table 3 - Image classification

Accuracy. `Conv` indicates a convolutional feature extractor trained jointly with NN-kNN; `Pre-Conv` indicates the ConvNet feature extractor was pretrained and frozen.

| Method | CIFAR-10 | SVHN |
|---|---:|---:|
| Conv + NN-kNN 500 | 0.688 (0.006) | 0.867 (0.003) |
| Conv + NN-kNN 2000 | 0.675 (0.003) | 0.841 (0.008) |
| ConvNet | 0.689 | 0.875 |
| kNN | 0.339 | 0.469 |
| Pre-Conv + kNN | 0.647 | 0.869 |
| Pre-Conv + NN-kNN | 0.585 | 0.750 |

The 500-case Conv + NN-kNN condition is numerically within 0.1 percentage points of ConvNet on CIFAR-10 and 0.8 percentage points on SVHN. The 2000-case and frozen-feature NN-kNN conditions are weaker, so the evidence supports feasibility and an importance of joint representation-retrieval training rather than monotonic benefit from adding cases.

### Table 4 - SST sentiment classification

Accuracy. `PreBi-LSTM` indicates a pretrained frozen Bi-LSTM feature extractor; `Bi-LSTM + NN-kNN` is jointly trained.

| Method | SST-5 | SST-2 |
|---|---:|---:|
| Bi-LSTM | 0.376 | 0.776 |
| PreBi-LSTM + kNN | 0.352 | 0.782 |
| PreBi-LSTM + NN-kNN | 0.363 (0.002) | 0.785 (0.001) |
| Bi-LSTM + NN-kNN | 0.368 (0.007) | 0.763 (0.003) |

For SST-5, jointly trained Bi-LSTM + NN-kNN is 0.8 percentage points below the Bi-LSTM point estimate. For SST-2, pretrained Bi-LSTM + NN-kNN is 0.9 points above Bi-LSTM, although the training arrangements differ; the jointly trained condition is 1.3 points below. This supports a cautious statement of comparable reported accuracy, not a claim that NN-kNN always improves a neural language model.

Table 5 contains qualitative SST-5 case explanations rather than a neural-baseline performance comparison and is therefore not used for the parity claim.

## Regression paper

### Experimental protocol relevant to claim strength

The regression paper uses a random 80/20 train-validation split, standardizes inputs and targets from the training portion, converts predictions back to raw target units, and reports average raw-scale RMSE over five runs in Table 1. Lower is better. Table 1 does not report run-level standard deviations or confidence intervals. The paper evaluates multiple NN-kNN normalizer, locality, and adaptation variants; consequently, a per-dataset best-variant summary is useful descriptively but must not be described as one universally selected configuration.

### Table 1 - Average RMSE of five runs

Abbreviations: CH = California Housing; Di = Diabetes; Ab = Abalone; BF = Body Fat; BS = Bike Sharing; WQ = Wine Quality; AF = Airfoil; Cars = Cars/Automobile; SP = Student Performance; Y = Yacht; EE = Energy Efficiency; UTK = UTKFace using frozen ResNet features.

| Model | CH | Di | Ab | BF | BS | WQ | AF | Cars | SP | Y | EE | UTK |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Weighted k-NN | 0.634 | 61.85 | 2.24 | 0.013 | 111.51 | 0.634 | 2.38 | 12683.68 | 2.43 | 10.50 | 2.30 | 12.98 |
| MLP | 0.494 | 57.10 | 2.05 | 0.010 | 45.40 | 0.646 | 2.51 | 15763.32 | 1.48 | 4.52 | 1.38 | 11.68 |
| MLKR | 0.466 | 59.70 | 2.15 | 0.012 | 52.31 | 0.596 | 2.44 | 16447.38 | 1.70 | 3.32 | 1.19 | 12.43 |
| NN-kNN softmax, pure | 0.586 | 58.46 | 2.14 | 0.011 | 71.86 | 0.579 | 2.05 | 13194.01 | 1.41 | 3.24 | 0.942 | 12.60 |
| + adaptation | 0.535 | 57.68 | 2.08 | 0.010 | 62.64 | 0.578 | 1.89 | 11755.02 | 1.34 | 4.50 | 1.38 | 12.04 |
| + locality | 0.665 | 58.51 | 2.16 | 0.011 | 72.04 | 0.577 | 2.08 | 13642.72 | 1.48 | 8.86 | 0.896 | 12.71 |
| + locality + adaptation | 0.903 | 57.35 | 2.07 | 0.011 | 63.72 | 0.576 | 1.98 | 11637.63 | 1.37 | 6.65 | 1.26 | 12.03 |
| NN-kNN sparsemax, pure | 0.445 | 70.31 | 2.22 | 0.014 | 46.92 | 0.594 | 1.64 | 10935.63 | 1.92 | 1.48 | 0.544 | 14.85 |
| + adaptation | 0.438 | 64.06 | 2.12 | 0.012 | 46.32 | 0.589 | 1.58 | 10017.56 | 1.61 | 1.21 | 0.546 | 13.31 |
| + locality | 0.453 | 70.26 | 2.22 | 0.014 | 47.67 | 0.594 | 1.71 | 9966.15 | 1.80 | 1.49 | 0.477 | 14.95 |
| + locality + adaptation | 0.441 | 64.82 | 2.16 | 0.012 | 46.87 | 0.589 | 1.65 | 9923.38 | 1.54 | 1.01 | 0.501 | 13.22 |

### Best displayed NN-kNN mean versus MLP

Negative relative difference means lower, better NN-kNN RMSE. This deliberately selects the lowest displayed NN-kNN variant per dataset and is not one fixed configuration.

| Dataset | Best reported NN-kNN variant | NN-kNN RMSE | MLP RMSE | Relative difference | Descriptive result |
|---|---|---:|---:|---:|---|
| California Housing | sparsemax + adaptation | 0.438 | 0.494 | -11.34% | NN-kNN lower |
| Diabetes | softmax + locality + adaptation | 57.35 | 57.10 | +0.44% | within 1% |
| Abalone | softmax + locality + adaptation | 2.07 | 2.05 | +0.98% | within 1% |
| Body Fat | softmax + adaptation | 0.010 | 0.010 | 0.00% at displayed precision | displayed tie; raw equality unknown |
| Bike Sharing | sparsemax + adaptation | 46.32 | 45.40 | +2.03% | within 3% |
| Wine Quality | softmax + locality + adaptation | 0.576 | 0.646 | -10.84% | NN-kNN lower |
| Airfoil | sparsemax + adaptation | 1.58 | 2.51 | -37.05% | NN-kNN lower |
| Cars | sparsemax + locality + adaptation | 9923.38 | 15763.32 | -37.05% | NN-kNN lower |
| Student Performance | softmax + adaptation | 1.34 | 1.48 | -9.46% | NN-kNN lower |
| Yacht | sparsemax + locality + adaptation | 1.01 | 4.52 | -77.65% | NN-kNN lower |
| Energy Efficiency | sparsemax + locality | 0.477 | 1.38 | -65.43% | NN-kNN lower |
| UTKFace | softmax + locality + adaptation | 12.03 | 11.68 | +3.00% | within 3.0% |

The proposal may accurately say that the reported mean RMSEs show an NN-kNN variant outperforming the MLP point estimate on 7 of 12 datasets, tying at displayed precision on Body Fat, and remaining within 3.0% on the other 4. It must immediately clarify that this selects among multiple variants, Table 1 reports no variability, and a new prespecified comparison is needed for a confirmatory T1 replacement claim.

### Table 2 - Synthetic predictive and retrieval quality

`d*` is the ground-truth weighted L2 distance averaged over the top 5 retrieved neighbors; lower is better. Adaptation changes the prediction but not the retrieved neighbor set, so the source table reports `d*` only on the corresponding pre-adaptation row. The oracle retrieves by label and is an explanatory upper bound, not a deployable query-time comparator.

| Model | RMSE | Mean `d*` |
|---|---:|---:|
| NN-kNN softmax without locality loss | 0.9711 | 1.3719 |
| after adaptation | 0.9603 | - |
| NN-kNN sparsemax without locality loss | 1.1907 | 0.2990 |
| after adaptation | 1.0440 | - |
| NN-kNN softmax with locality loss | 0.9854 | 0.2853 |
| after adaptation | 0.9672 | - |
| NN-kNN sparsemax with locality loss | 1.1674 | 0.3006 |
| after adaptation | 1.0460 | - |
| Weighted k-NN | 1.3189 | 0.3284 |
| MLP | 0.9638 | 0.4216 |
| Oracle k-NN by target `y` | 0.0148 | 0.2402 |

Two complementary findings matter for T1. Softmax without locality plus adaptation slightly improves on the MLP point estimate (`0.9603` versus `0.9638`) but has poor ground-truth neighbor distance (`1.3719`). Softmax with locality plus adaptation is close to MLP error (`0.9672` versus `0.9638`) while its retrieved neighbors are closer under the ground-truth metric (`0.2853` versus `0.4216`). This is direct preliminary evidence for studying coordinated predictive quality and explanatory retrieval rather than optimizing accuracy alone.

### Table 3 - Manual feature tuning

Lower MSE and higher R-squared are better. The source experiment uses 1,500 synthetic examples and sparsemax NN-kNN without NN-CDH or locality regularization.

| Stage | `w0` | `w1` | `w2` | MSE | R-squared |
|---|---:|---:|---:|---:|---:|
| A | 0.000 | 0.000 | 0.312 | 0.5285 | 0.4814 |
| B.1 | 0.000 | 0.000 | 0.318 | 0.5290 | 0.4809 |
| B.2 | 0.000 | 0.117 | 0.213 | 0.5361 | 0.4740 |
| C.1 | 0.101 | 0.055 | 0.057 | 0.0096 | 0.9906 |
| C.2 | 0.425 | 0.169 | 0.187 | 0.0092 | 0.9909 |

This table supports the feasibility of manual feature-weight intervention and continued training, but not the complete proposed human-revision workflow. B.2 preserves the manually favored redundant feature with similar error to A/B.1; C.1 and C.2 show recovery when `w0` is unfrozen, with the manual C.2 nudge retaining comparable or slightly better displayed performance.

## Proposal-ready preliminary-evidence language

> Prior **pre-MCB** work establishes supervised feasibility rather than the proposed MCB-enhanced, full-cycle solution. On image classification, a jointly trained 500-case NN-kNN head achieved reported accuracies of 0.688 versus 0.689 for a ConvNet on CIFAR-10 and 0.867 versus 0.875 on SVHN. In regression, the best reported NN-kNN variant per dataset had lower mean RMSE than the MLP point estimate on 7 of 12 benchmarks, tied at displayed precision on one, and was within 3.0% on the remaining four. A synthetic regression study further showed that near-MLP prediction error can coexist with substantially better ground-truth neighbor locality, while also revealing that accuracy alone can conceal poor explanatory retrieval. These findings provide the legacy NN-kNN baseline and motivate T1's coordinated retrieval, neural adaptation, revision, retention, and MCB evaluation. Because the published tables do not provide uncertainty for all neural baselines, the regression summary selects among multiple NN-kNN variants, and neither paper includes MCB, the CAREER project will use prespecified configurations, matched budgets, and uncertainty-aware equivalence or non-inferiority criteria for confirmatory claims.

## Claim boundary

Use **"the pre-MCB NN-kNN was numerically competitive in prior reported experiments"**, **"within 0.1-0.8 accuracy points on the reported image comparisons"**, or the exact regression count above. Do not yet write **"statistically equivalent,"** **"universally on par,"** **"state of the art,"** or **"better than neural networks."** Do not attribute any result in this note to MCB or to the proposed unified T1 system. T1 will first reproduce the old architecture on the completed-paper benchmarks, then compare MCB and full-cycle variants while holding recoverable splits, encoders, tuning budgets, case budgets, and metrics constant. Until those experiments are completed, the values here remain historical baselines. T1 remains new because the earlier studies do not implement or validate MCB on these benchmark suites, the complete coordinated retrieve-reuse-revise-retain system, quality-aware bounded memory, cross-stage maintenance, or confirmatory replacement criteria.
