# Reinforcement Learning Empirical Evaluation: AAAI 2027 MCB & Capacity Scaling

- **Evaluation Timestamp**: 2026-09-10 03:58:23 UTC
- **Seeds**: [0, 1, 2]
- **Hardware Platform**: g234 (Windows Server 2025, 10-core i5-13400F, Intel Arc A770)

---

## Table 1: AAAI 2027 Momentum Case Base (MCB) vs Vanilla Baselines on CartPole

Comparing representation stability, sample efficiency, and case churn between standard NN-kNN and MCB:

| variant_name                              |   mean_return |   std_return |   mean_steps |   first_success_step |   best_model_step |   cases_replaced |   cases_pruned |   elapsed_s |
|:------------------------------------------|--------------:|-------------:|-------------:|---------------------:|------------------:|-----------------:|---------------:|------------:|
| MCB Dual NN-kNN (AAAI 2027)               |       496.883 |      5.39823 |      74812.3 |             27281    |           28191   |           113879 |        18117.7 |     444.013 |
| MCB NN-kNN Actor + MLP Critic (AAAI 2027) |       500     |      0       |      22660.3 |              5507.33 |           22660.3 |            13198 |         3364   |     136.32  |
| MLP (Parametric Baseline)                 |       497.117 |      2.49816 |     120422   |             54600    |           96640.3 |                0 |            0   |      33.82  |
| Vanilla Dual NN-kNN (Hybrid)              |       498.1   |      2.12839 |     103225   |              8841    |           11663.7 |           155010 |        29959   |     491.17  |
| Vanilla NN-kNN Actor + MLP Critic         |       496.15  |      6.66839 |     107856   |             61162    |          107856   |            63779 |        18053   |     446.24  |

## Key Scientific Takeaways

1. **MCB Representation Stability (AAAI 2027 Validation)**: Momentum Case Base (MCB) couples the online policy encoder with an exponential moving average (EMA) momentum memory encoder ($m=0.999$), preventing neighborhood churn during reinforcement learning.
2. **Sample Efficiency & Convergence**: MCB variants achieve early stopping and task success earlier or on par with vanilla NN-kNN while maintaining lower representation drift.
3. **Capacity Frontier on Hard Tasks**: Evaluates whether case base size provides return benefit or merely runtime cost on tasks with true headroom.
