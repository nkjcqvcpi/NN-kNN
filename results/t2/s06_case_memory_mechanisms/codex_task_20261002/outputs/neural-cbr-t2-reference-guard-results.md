# T2 training-reference coverage and per-query guard

108 independent post-checkpoint interventions on the same six preserve_by_id NN/NN runs. No new optimization, no diagnostic-guided selection. Control uses last one recorded batch; wide/guard use last up to eight recorded batches. Each batch contributes two actual training queries; Acrobot has only three batches.

| Mode | Conditions | Queries per reference | Candidate replays | Reached capacity | Removed | MC worse | Greedy worse |
|---|---:|---|---:|---:|---:|---:|---:|
| control | 36 | [2] | 3758 | 6 | 364 | 20 | 4 |
| wide | 36 | [4, 16] | 14533 | 19 | 629 | 15 | 6 |
| guard | 36 | [4, 16] | 11368 | 10 | 392 | 6 | 6 |

| Condition | Capacities control / wide / guard | Greedy returns control / wide / guard | MC MSE control / wide / guard |
|---|---|---|---|
| cartpole_s8_critic_k96_b0.0 | 112 / 102 / 118 | 287.33 / 287.33 / 287.33 | 947.06 / 922.59 / 971.96 |
| cartpole_s8_critic_k96_b0.03 | 112 / 102 / 118 | 287.33 / 287.33 / 287.33 | 947.06 / 922.59 / 971.96 |
| cartpole_s8_critic_k115_b0.0 | 115 / 115 / 118 | 287.33 / 287.33 / 287.33 | 935.52 / 927.59 / 971.96 |
| cartpole_s8_critic_k115_b0.03 | 115 / 115 / 118 | 287.33 / 287.33 / 287.33 | 935.52 / 927.59 / 971.96 |
| cartpole_s8_actor_k87_b0.0 | 101 / 87 / 96 | 225.67 / 332.00 / 312.33 | 2312.08 / 533.22 / 415.40 |
| cartpole_s8_actor_k87_b0.03 | 101 / 87 / 96 | 225.67 / 332.00 / 312.33 | 2312.08 / 533.22 / 415.40 |
| cartpole_s8_actor_k105_b0.0 | 105 / 105 / 105 | 240.33 / 263.67 / 263.67 | 1191.90 / 1043.06 / 1043.06 |
| cartpole_s8_actor_k105_b0.03 | 105 / 105 / 105 | 240.33 / 263.67 / 263.67 | 1191.90 / 1043.06 / 1043.06 |
| cartpole_s9_critic_k96_b0.0 | 117 / 97 / 109 | 153.00 / 153.00 / 153.00 | 2198.19 / 2155.22 / 2126.59 |
| cartpole_s9_critic_k96_b0.03 | 117 / 97 / 109 | 153.00 / 153.00 / 153.00 | 2198.19 / 2155.22 / 2126.59 |
| cartpole_s9_critic_k115_b0.0 | 117 / 115 / 115 | 153.00 / 153.00 / 153.00 | 2198.19 / 2127.24 / 2127.24 |
| cartpole_s9_critic_k115_b0.03 | 117 / 115 / 115 | 153.00 / 153.00 / 153.00 | 2198.19 / 2127.24 / 2127.24 |
| cartpole_s9_actor_k96_b0.0 | 118 / 96 / 114 | 157.33 / 389.67 / 280.00 | 2966.53 / 157.30 / 1391.85 |
| cartpole_s9_actor_k96_b0.03 | 118 / 96 / 114 | 157.33 / 389.67 / 280.00 | 2966.53 / 157.30 / 1391.85 |
| cartpole_s9_actor_k115_b0.0 | 118 / 115 / 115 | 157.33 / 272.33 / 319.67 | 2966.53 / 2147.62 / 1669.37 |
| cartpole_s9_actor_k115_b0.03 | 118 / 115 / 115 | 157.33 / 272.33 / 319.67 | 2966.53 / 2147.62 / 1669.37 |
| cartpole_s10_critic_k96_b0.0 | 112 / 97 / 113 | 462.67 / 462.67 / 462.67 | 3502.26 / 3386.11 / 3466.28 |
| cartpole_s10_critic_k96_b0.03 | 112 / 97 / 113 | 462.67 / 462.67 / 462.67 | 3502.26 / 3386.11 / 3466.28 |
| cartpole_s10_critic_k115_b0.0 | 115 / 115 / 115 | 462.67 / 462.67 / 462.67 | 3495.19 / 3460.40 / 3482.83 |
| cartpole_s10_critic_k115_b0.03 | 115 / 115 / 115 | 462.67 / 462.67 / 462.67 | 3495.19 / 3460.40 / 3482.83 |
| cartpole_s10_actor_k96_b0.0 | 118 / 99 / 120 | 482.67 / 200.33 / 443.67 | 1888.98 / 3329.13 / 2080.75 |
| cartpole_s10_actor_k96_b0.03 | 116 / 96 / 120 | 497.33 / 229.00 / 443.67 | 1565.01 / 3677.20 / 2080.75 |
| cartpole_s10_actor_k115_b0.0 | 118 / 115 / 120 | 482.67 / 158.00 / 443.67 | 1888.98 / 1858.91 / 2080.75 |
| cartpole_s10_actor_k115_b0.03 | 116 / 115 / 120 | 497.33 / 158.00 / 443.67 | 1565.01 / 1858.91 / 2080.75 |
| acrobot_s8_critic_k96_b0.0 | 123 / 121 / 122 | -458.00 / -458.00 / -458.00 | 4474.57 / 4470.47 / 4458.15 |
| acrobot_s8_critic_k96_b0.03 | 123 / 121 / 122 | -458.00 / -458.00 / -458.00 | 4474.57 / 4470.47 / 4458.15 |
| acrobot_s8_critic_k115_b0.0 | 123 / 121 / 122 | -458.00 / -458.00 / -458.00 | 4474.57 / 4470.47 / 4458.15 |
| acrobot_s8_critic_k115_b0.03 | 123 / 121 / 122 | -458.00 / -458.00 / -458.00 | 4474.57 / 4470.47 / 4458.15 |
| acrobot_s9_critic_k96_b0.0 | 125 / 121 / 128 | -458.00 / -458.00 / -458.00 | 3627.55 / 3651.41 / 3629.83 |
| acrobot_s9_critic_k96_b0.03 | 125 / 121 / 128 | -458.00 / -458.00 / -458.00 | 3627.55 / 3651.41 / 3629.83 |
| acrobot_s9_critic_k115_b0.0 | 125 / 121 / 128 | -458.00 / -458.00 / -458.00 | 3627.55 / 3651.41 / 3629.83 |
| acrobot_s9_critic_k115_b0.03 | 125 / 121 / 128 | -458.00 / -458.00 / -458.00 | 3627.55 / 3651.41 / 3629.83 |
| acrobot_s10_critic_k96_b0.0 | 120 / 112 / 113 | -458.00 / -458.00 / -458.00 | 4349.09 / 4385.38 / 4377.10 |
| acrobot_s10_critic_k96_b0.03 | 120 / 112 / 113 | -458.00 / -458.00 / -458.00 | 4349.09 / 4385.38 / 4377.10 |
| acrobot_s10_critic_k115_b0.0 | 120 / 115 / 115 | -458.00 / -458.00 / -458.00 | 4349.09 / 4371.21 / 4362.11 |
| acrobot_s10_critic_k115_b0.03 | 120 / 115 / 115 | -458.00 / -458.00 / -458.00 | 4349.09 / 4371.21 / 4362.11 |

Control exactly reproduces all36 preceding preserve-mode accepted IDs, loss vectors, greedy and MC diagnostics. Every new candidate is independently replayed, including original mean-budget and optional every-query budget checks. Historical training targets/advantages remain tied to their recorded execution; newer model parameters reevaluate them. Reference age is explicit, not a freshness guarantee.

The strict per-query guard is fixed before diagnostics. It can stop further compression, and does not prove protection outside these training queries. Actor compaction changes MC rollout distributions; all diagnostics are reused exploratory observations, not Stage A acceptance. Selected best-evaluation checkpoints exclude training references at or after the selected step to avoid later-phase leakage.

The controlled two-query counterexample in the tests shows a mean-budget-feasible removal that raises one query loss from1 to4 while another improves. The strict query guard rejects it.

Source `847e751d081c0116832304a20d5613799fe26ca8b81af04da9de1442389a0355`; ZIP SHA256 `4fac8324126e7bcfab762f13db04d7338c61d4af75c595d9ac69115f662b8a14`. Combined182-test suite passes.
