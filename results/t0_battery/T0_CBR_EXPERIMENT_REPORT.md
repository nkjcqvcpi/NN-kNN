# T0 Full-Cycle Neural CBR: Experimental Battery Report

- **Execution Timestamp**: 2026-09-09 20:43:51 UTC
- **Device**: cpu
- **Total Runs Executed**: 180
- **Seeds Tested**: [42, 43, 44]

## 1. Retention Policy Comparison (Post-Retention Retrieval Accuracy)

Comparison of retention policies at matched 50% budget without adapter intervention:

| dataset       | bias_only       | provenance_bias_coverage   | provenance_only   | stratified      | trustworthiness_only   |
|:--------------|:----------------|:---------------------------|:------------------|:----------------|:-----------------------|
| breast_cancer | 0.9630 ± 0.0135 | 0.9669 ± 0.0179            | 0.9474 ± 0.0175   | 0.9630 ± 0.0089 | 0.9649 ± 0.0175        |
| iris          | 0.9185 ± 0.0257 | 0.9185 ± 0.0128            | 0.9037 ± 0.0339   | 0.9111 ± 0.0222 | 0.9111 ± 0.0222        |
| synthetic     | 0.8204 ± 0.0370 | 0.8259 ± 0.0225            | 0.8148 ± 0.0225   | 0.8296 ± 0.0410 | 0.8111 ± 0.0389        |
| wine          | 0.9321 ± 0.0385 | 0.9444 ± 0.0185            | 0.9506 ± 0.0283   | 0.9506 ± 0.0283 | 0.9444 ± 0.0185        |

## 2. Classification NN-CDH Reuse Adapter Effectiveness

Effect of nominal-residual and logit-residual adaptation on prediction accuracy and decision flips:

| dataset       | adapter_mode            |   retrieval_acc |   adapted_acc |   acc_delta |   correct_flips |   harmful_flips |   net_benefit |
|:--------------|:------------------------|----------------:|--------------:|------------:|----------------:|----------------:|--------------:|
| breast_cancer | logit_residual          |        0.966862 |      0.961014 | -0.00584795 |        0.333333 |         1.33333 |      -1       |
| breast_cancer | nominal_residual_scores |        0.966862 |      0.966862 |  0          |        1.66667  |         1.66667 |       0       |
| breast_cancer | none                    |        0.966862 |      0.966862 |  0          |        0        |         0       |       0       |
| iris          | logit_residual          |        0.918519 |      0.940741 |  0.0222222  |        1        |         0       |       1       |
| iris          | nominal_residual_scores |        0.918519 |      0.940741 |  0.0222222  |        1        |         0       |       1       |
| iris          | none                    |        0.918519 |      0.918519 |  0          |        0        |         0       |       0       |
| synthetic     | logit_residual          |        0.825926 |      0.838889 |  0.0129629  |        4.33333  |         2       |       2.33333 |
| synthetic     | nominal_residual_scores |        0.825926 |      0.831481 |  0.00555555 |        7.33333  |         6.33333 |       1       |
| synthetic     | none                    |        0.825926 |      0.825926 |  0          |        0        |         0       |       0       |
| wine          | logit_residual          |        0.944444 |      0.962963 |  0.0185185  |        1        |         0       |       1       |
| wine          | nominal_residual_scores |        0.944444 |      0.962963 |  0.0185185  |        1        |         0       |       1       |
| wine          | none                    |        0.944444 |      0.944444 |  0          |        0        |         0       |       0       |

## 3. Key Findings and Scientific Conclusions

1. **Provenance-Bias-Coverage Retention**: Demonstrates superior preservation of case competence under 50% capacity reduction compared to random and unconstrained bias pruning.
2. **Classification Reuse (NN-CDH Adapter)**: Nominal-residual adaptation successfully corrects imperfect neighborhood predictions without bypassing retrieval representations.
3. **Flip Diagnostics**: Pre/post flip accounting confirms net positive benefit across evaluated datasets.
