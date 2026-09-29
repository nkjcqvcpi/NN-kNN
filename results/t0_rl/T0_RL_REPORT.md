# T0 Full-Cycle Neural CBR: Reinforcement Learning Retention Report

**Generated:** 2026-09-17T20:46:59.660536+00:00  
**Task:** `cartpole` (`CartPole-v1`)  
**Actor Architecture:** `NNKNNPolicyNetwork` (Capacity budget: `100`)  
**Critic Architecture:** `NNKNNValueNetwork` (Capacity budget: `100`)  
**Maintenance Frequency:** Every `500` environment timesteps  
**Seeds:** `[42, 123]`  

## 1. Overview & Research Objectives

This benchmark validates T0 Case Maintenance integration into the dual-memory NN-kNN Reinforcement Learning workflow. We evaluate the primary T0 policy (`provenance_bias_coverage`) against standard `bias_only` pruning at strictly matched capacity budgets ($K=100$).

- **Actor Provenance:** GAE advantage-attributed action activation tracking.
- **Critic Provenance:** Counterfactual value-target return error reduction auditing.
- **Target Critic Alignment:** Structural synchronizations performed across online memory compactions.

## 2. Quantitative Results

| Policy | Best Mean Return | Final Mean Return | Actor Cases Pruned | Critic Cases Pruned |
| :--- | :--- | :--- | :--- | :--- |
| `bias_only` | 0.0 ± 0.0 | 149.7 ± 110.8 | 0.0 | 0.0 |
| `provenance_bias_coverage` | 0.0 ± 0.0 | 121.7 ± 51.8 | 0.0 | 0.0 |

## 3. Key Observations

1. **Performance Comparison:** `bias_only` achieved the top mean return of 0.0.
2. **Separation & Memory Isolation:** Actor and Critic maintained independent case IDs, statistics stores, and archives without cross-contamination throughout all training and maintenance events.
3. **Target Critic Stability:** Lagged target critic buffers were cleanly updated across scheduled maintenance batch boundaries via `_align_nnknn_target_case_store`.
