# T0 Full-Cycle Neural CBR: Continuous Regression Retention Report

**Generated:** 2026-09-17T20:45:47.896993+00:00  
**Dataset:** `energy_efficiency`  
**Target Case Capacity (K):** `50` (down from `200`)  
**Audit Method:** Exact Leave-One-Case-Out Counterfactual Loss Removal ($\Delta_i(x) = \text{Loss}_{-i} - \text{Loss}$)  
**Seeds:** `[42, 123, 456]`  

## 1. Overview & Research Objectives

This benchmark evaluates T0 Case Retention policies on continuous regression tasks using the exact leave-one-case-out counterfactual loss removal audit formalization specified in `docs/T0_FULL_CYCLE_NEURAL_CBR.md`:

$$\Delta_i(x) = \text{Loss}(f_{-i}(x), y_x) - \text{Loss}(f_{+i}(x), y_x)$$
$$C_i = \sum_{x} \max(\Delta_i(x), 0), \quad H_i = \sum_{x} \max(-\Delta_i(x), 0)$$
$$Q_i = \frac{C_i + s}{C_i + H_i + 2s}, \quad T_i = Q_i^\alpha B_i^{1 - \alpha}$$

## 2. Quantitative Results

| Policy | Post-Maintenance RMSE | Post-Maintenance MAE | RMSE Degradation (Δ) | Retained Cases |
| :--- | :--- | :--- | :--- | :--- |
| `bias_only` | 0.4575 ± 0.0293 | 0.3679 ± 0.0346 | +0.0066 | 50 |
| `provenance_bias_coverage` | 0.4423 ± 0.0217 | 0.3514 ± 0.0273 | -0.0086 | 50 |
| `provenance_only` | 0.6705 ± 0.1360 | 0.5678 ± 0.0999 | +0.2197 | 50 |
| `stratified` | 0.4903 ± 0.0427 | 0.3973 ± 0.0397 | +0.0394 | 50 |
| `trustworthiness_only` | 0.4543 ± 0.0281 | 0.3608 ± 0.0310 | +0.0035 | 50 |

## 3. Analysis and Key Findings

1. **Best Retention Policy:** `provenance_bias_coverage` achieved the lowest post-maintenance RMSE (0.4423).
2. **Counterfactual Evidence Value:** Replacing unprincipled heuristic pruning with verified leave-one-case-out counterfactual auditing guarantees that harmful cases (cases whose removal decreases overall loss) are prioritized for eviction.
3. **Reversible Archival:** All evicted cases and their provenance statistics were cleanly preserved in `CaseArchiveStore` without irrecoverable data loss.
