# T2 fresh training-reference retention pilot

72 post-checkpoint one-role interventions on twelve 2,048-interaction NN/NN runs: six existing reset-mode runs and six new preserve_by_id trainings. No gradient retraining after deletion. Latest two actual training queries per role. Acrobot actors have no ready-policy reference: twenty-four conditions skipped.

Counts: `{"conditions": 72, "actual_new_trainings": 6, "skipped_conditions": 24, "candidates": 7468, "removed": 737, "reached": 16, "moment_tensors": 144, "mc_worse": 34, "greedy_worse": 4}`

| Condition | Actual capacity | Training loss before → after | Greedy return before → after | MC MSE before → after |
|---|---:|---:|---:|---:|
| reset/cartpole_s8_critic_k96_b0.0 | 112 | 90.30148 → 45.62740 | 103.33 → 103.33 | 857.01 → 797.48 |
| reset/cartpole_s8_critic_k96_b0.03 | 112 | 90.30148 → 45.62740 | 103.33 → 103.33 | 857.01 → 797.48 |
| reset/cartpole_s8_critic_k115_b0.0 | 115 | 90.30148 → 41.45162 | 103.33 → 103.33 | 857.01 → 793.84 |
| reset/cartpole_s8_critic_k115_b0.03 | 115 | 90.30148 → 41.45162 | 103.33 → 103.33 | 857.01 → 793.84 |
| reset/cartpole_s8_actor_k96_b0.0 | 115 | 0.60279 → 0.57451 | 103.33 → 220.33 | 857.01 → 760.56 |
| reset/cartpole_s8_actor_k96_b0.03 | 115 | 0.60279 → 0.57451 | 103.33 → 220.33 | 857.01 → 760.56 |
| reset/cartpole_s8_actor_k115_b0.0 | 115 | 0.60279 → 0.57451 | 103.33 → 220.33 | 857.01 → 760.56 |
| reset/cartpole_s8_actor_k115_b0.03 | 115 | 0.60279 → 0.57451 | 103.33 → 220.33 | 857.01 → 760.56 |
| reset/cartpole_s9_critic_k96_b0.0 | 113 | 110.15357 → 108.85855 | 86.00 → 86.00 | 768.97 → 784.93 |
| reset/cartpole_s9_critic_k96_b0.03 | 113 | 110.15357 → 108.85855 | 86.00 → 86.00 | 768.97 → 784.93 |
| reset/cartpole_s9_critic_k115_b0.0 | 115 | 110.15357 → 100.55408 | 86.00 → 86.00 | 768.97 → 766.35 |
| reset/cartpole_s9_critic_k115_b0.03 | 115 | 110.15357 → 100.55408 | 86.00 → 86.00 | 768.97 → 766.35 |
| reset/cartpole_s9_actor_k96_b0.0 | 124 | 0.02779 → 0.02779 | 86.00 → 86.00 | 768.97 → 478.51 |
| reset/cartpole_s9_actor_k96_b0.03 | 124 | 0.02779 → 0.02779 | 86.00 → 86.00 | 768.97 → 478.51 |
| reset/cartpole_s9_actor_k115_b0.0 | 124 | 0.02779 → 0.02779 | 86.00 → 86.00 | 768.97 → 478.51 |
| reset/cartpole_s9_actor_k115_b0.03 | 124 | 0.02779 → 0.02779 | 86.00 → 86.00 | 768.97 → 478.51 |
| reset/cartpole_s10_critic_k96_b0.0 | 112 | 160.57033 → 87.80206 | 246.00 → 246.00 | 1414.20 → 1316.28 |
| reset/cartpole_s10_critic_k96_b0.03 | 112 | 160.57033 → 87.80206 | 246.00 → 246.00 | 1414.20 → 1316.28 |
| reset/cartpole_s10_critic_k115_b0.0 | 115 | 160.57033 → 75.91004 | 246.00 → 246.00 | 1414.20 → 1312.75 |
| reset/cartpole_s10_critic_k115_b0.03 | 115 | 160.57033 → 75.91004 | 246.00 → 246.00 | 1414.20 → 1312.75 |
| reset/cartpole_s10_actor_k96_b0.0 | 115 | 0.00340 → -0.00154 | 246.00 → 500.00 | 1414.20 → 1174.50 |
| reset/cartpole_s10_actor_k96_b0.03 | 114 | 0.00340 → 0.01605 | 246.00 → 500.00 | 1414.20 → 1010.28 |
| reset/cartpole_s10_actor_k115_b0.0 | 115 | 0.00340 → -0.00154 | 246.00 → 500.00 | 1414.20 → 1174.50 |
| reset/cartpole_s10_actor_k115_b0.03 | 115 | 0.00340 → -0.00154 | 246.00 → 500.00 | 1414.20 → 1174.50 |
| reset/acrobot_s8_critic_k96_b0.0 | 123 | 18.48381 → 16.91475 | -458.00 → -458.00 | 4464.03 → 4473.67 |
| reset/acrobot_s8_critic_k96_b0.03 | 123 | 18.48381 → 16.91475 | -458.00 → -458.00 | 4464.03 → 4473.67 |
| reset/acrobot_s8_critic_k115_b0.0 | 123 | 18.48381 → 16.91475 | -458.00 → -458.00 | 4464.03 → 4473.67 |
| reset/acrobot_s8_critic_k115_b0.03 | 123 | 18.48381 → 16.91475 | -458.00 → -458.00 | 4464.03 → 4473.67 |
| reset/acrobot_s9_critic_k96_b0.0 | 122 | 30.61477 → 25.26490 | -458.00 → -458.00 | 3551.04 → 3568.74 |
| reset/acrobot_s9_critic_k96_b0.03 | 122 | 30.61477 → 25.26490 | -458.00 → -458.00 | 3551.04 → 3568.74 |
| reset/acrobot_s9_critic_k115_b0.0 | 122 | 30.61477 → 25.26490 | -458.00 → -458.00 | 3551.04 → 3568.74 |
| reset/acrobot_s9_critic_k115_b0.03 | 122 | 30.61477 → 25.26490 | -458.00 → -458.00 | 3551.04 → 3568.74 |
| reset/acrobot_s10_critic_k96_b0.0 | 119 | 7.63616 → 7.14403 | -458.00 → -458.00 | 4336.90 → 4338.67 |
| reset/acrobot_s10_critic_k96_b0.03 | 119 | 7.63616 → 7.14403 | -458.00 → -458.00 | 4336.90 → 4338.67 |
| reset/acrobot_s10_critic_k115_b0.0 | 119 | 7.63616 → 7.14403 | -458.00 → -458.00 | 4336.90 → 4338.67 |
| reset/acrobot_s10_critic_k115_b0.03 | 119 | 7.63616 → 7.14403 | -458.00 → -458.00 | 4336.90 → 4338.67 |
| preserve_by_id/cartpole_s8_critic_k96_b0.0 | 112 | 151.37820 → 119.65363 | 287.33 → 287.33 | 975.31 → 947.06 |
| preserve_by_id/cartpole_s8_critic_k96_b0.03 | 112 | 151.37820 → 119.65363 | 287.33 → 287.33 | 975.31 → 947.06 |
| preserve_by_id/cartpole_s8_critic_k115_b0.0 | 115 | 151.37820 → 110.74217 | 287.33 → 287.33 | 975.31 → 935.52 |
| preserve_by_id/cartpole_s8_critic_k115_b0.03 | 115 | 151.37820 → 110.74217 | 287.33 → 287.33 | 975.31 → 935.52 |
| preserve_by_id/cartpole_s8_actor_k87_b0.0 | 101 | 0.03671 → 0.00305 | 287.33 → 225.67 | 975.31 → 2312.08 |
| preserve_by_id/cartpole_s8_actor_k87_b0.03 | 101 | 0.03671 → 0.00305 | 287.33 → 225.67 | 975.31 → 2312.08 |
| preserve_by_id/cartpole_s8_actor_k105_b0.0 | 105 | 0.03671 → 0.00300 | 287.33 → 240.33 | 975.31 → 1191.90 |
| preserve_by_id/cartpole_s8_actor_k105_b0.03 | 105 | 0.03671 → 0.00300 | 287.33 → 240.33 | 975.31 → 1191.90 |
| preserve_by_id/cartpole_s9_critic_k96_b0.0 | 117 | 16.65202 → 15.79602 | 153.00 → 153.00 | 2187.94 → 2198.19 |
| preserve_by_id/cartpole_s9_critic_k96_b0.03 | 117 | 16.65202 → 15.79602 | 153.00 → 153.00 | 2187.94 → 2198.19 |
| preserve_by_id/cartpole_s9_critic_k115_b0.0 | 117 | 16.65202 → 15.79602 | 153.00 → 153.00 | 2187.94 → 2198.19 |
| preserve_by_id/cartpole_s9_critic_k115_b0.03 | 117 | 16.65202 → 15.79602 | 153.00 → 153.00 | 2187.94 → 2198.19 |
| preserve_by_id/cartpole_s9_actor_k96_b0.0 | 118 | 0.19530 → 0.19486 | 153.00 → 157.33 | 2187.94 → 2966.53 |
| preserve_by_id/cartpole_s9_actor_k96_b0.03 | 118 | 0.19530 → 0.19486 | 153.00 → 157.33 | 2187.94 → 2966.53 |
| preserve_by_id/cartpole_s9_actor_k115_b0.0 | 118 | 0.19530 → 0.19486 | 153.00 → 157.33 | 2187.94 → 2966.53 |
| preserve_by_id/cartpole_s9_actor_k115_b0.03 | 118 | 0.19530 → 0.19486 | 153.00 → 157.33 | 2187.94 → 2966.53 |
| preserve_by_id/cartpole_s10_critic_k96_b0.0 | 112 | 170.54141 → 150.28403 | 462.67 → 462.67 | 3580.42 → 3502.26 |
| preserve_by_id/cartpole_s10_critic_k96_b0.03 | 112 | 170.54141 → 150.28403 | 462.67 → 462.67 | 3580.42 → 3502.26 |
| preserve_by_id/cartpole_s10_critic_k115_b0.0 | 115 | 170.54141 → 128.03772 | 462.67 → 462.67 | 3580.42 → 3495.19 |
| preserve_by_id/cartpole_s10_critic_k115_b0.03 | 115 | 170.54141 → 128.03772 | 462.67 → 462.67 | 3580.42 → 3495.19 |
| preserve_by_id/cartpole_s10_actor_k96_b0.0 | 118 | 0.02000 → 0.01513 | 462.67 → 482.67 | 3580.42 → 1888.98 |
| preserve_by_id/cartpole_s10_actor_k96_b0.03 | 116 | 0.02000 → 0.03471 | 462.67 → 497.33 | 3580.42 → 1565.01 |
| preserve_by_id/cartpole_s10_actor_k115_b0.0 | 118 | 0.02000 → 0.01513 | 462.67 → 482.67 | 3580.42 → 1888.98 |
| preserve_by_id/cartpole_s10_actor_k115_b0.03 | 116 | 0.02000 → 0.03471 | 462.67 → 497.33 | 3580.42 → 1565.01 |
| preserve_by_id/acrobot_s8_critic_k96_b0.0 | 123 | 18.48381 → 16.93328 | -458.00 → -458.00 | 4464.86 → 4474.57 |
| preserve_by_id/acrobot_s8_critic_k96_b0.03 | 123 | 18.48381 → 16.93328 | -458.00 → -458.00 | 4464.86 → 4474.57 |
| preserve_by_id/acrobot_s8_critic_k115_b0.0 | 123 | 18.48381 → 16.93328 | -458.00 → -458.00 | 4464.86 → 4474.57 |
| preserve_by_id/acrobot_s8_critic_k115_b0.03 | 123 | 18.48381 → 16.93328 | -458.00 → -458.00 | 4464.86 → 4474.57 |
| preserve_by_id/acrobot_s9_critic_k96_b0.0 | 125 | 40.04817 → 38.64791 | -458.00 → -458.00 | 3629.83 → 3627.55 |
| preserve_by_id/acrobot_s9_critic_k96_b0.03 | 125 | 40.04817 → 38.64791 | -458.00 → -458.00 | 3629.83 → 3627.55 |
| preserve_by_id/acrobot_s9_critic_k115_b0.0 | 125 | 40.04817 → 38.64791 | -458.00 → -458.00 | 3629.83 → 3627.55 |
| preserve_by_id/acrobot_s9_critic_k115_b0.03 | 125 | 40.04817 → 38.64791 | -458.00 → -458.00 | 3629.83 → 3627.55 |
| preserve_by_id/acrobot_s10_critic_k96_b0.0 | 120 | 7.82792 → 7.14403 | -458.00 → -458.00 | 4347.60 → 4349.09 |
| preserve_by_id/acrobot_s10_critic_k96_b0.03 | 120 | 7.82792 → 7.14403 | -458.00 → -458.00 | 4347.60 → 4349.09 |
| preserve_by_id/acrobot_s10_critic_k115_b0.0 | 120 | 7.82792 → 7.14403 | -458.00 → -458.00 | 4347.60 → 4349.09 |
| preserve_by_id/acrobot_s10_critic_k115_b0.03 | 120 | 7.82792 → 7.14403 | -458.00 → -458.00 | 4347.60 → 4349.09 |

All 7468 full-bank candidate removals are independently replayed, including feasible-candidate filtering before fresh Q ranking. Original unobserved cases stay protected; action coverage stays protected. The reset runs have no private-case Adam moments at this phase: zero tensors checked there. Preserve-mode runs check 144 nonempty moment tensors, retained rows by ID and zero inactive rows, with scalar steps unchanged. Lagged target labels retain their own values.

34 conditions worsen independent MC MSE despite satisfying the training loss guard; 4 worsen greedy return. Critic-only interventions retain exactly the same greedy return; actor changes also change the MC behavior distribution, so their MSE difference is not an isolated critic-calibration comparison. These reused small diagnostic sets are not confirmation. Only 16 conditions reach their requested capacity.

Design conclusion: two queries leave many cases protected or loss/coverage constrained. Increase explicitly recorded training-reference coverage and study stale target/advantage effects; do not force unknown deletions or use diagnostic MC to select. This selector is exploratory and not activated in the live training workflow.

Frozen source `6645e6d173eb639f47998c5e232d4d41c42dfcb9bd5421aa7364bc32d00ddf23`; ZIP SHA256 `e88951fda8ee9bcc50818efe22d3fa772f91933b945aaece31a30137d56d3c2d`. Combined suite: 181 tests passed with the existing scalar-conversion warning.
