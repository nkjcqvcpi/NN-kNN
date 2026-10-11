# T2 Acrobot extended-startup diagnosis

18 original runs plus18 actual complete retrainings; 2,048/8,192 exact interactions × three seeds × EMA/hard/none targets. NN/NN,capacity128/top-k8,ID-preserved Adam,no scheduled bias pruning,target interval1,no early stopping. Every actual GAE target and raw-positive actor admission is reconstructed from original training inputs; no fabricated recommendation cases.

Replay: `{"actual_retrainings": 18, "batches": 111, "training_samples": 92160, "raw_positive_samples": 351}`; MC: `{"runs": 18, "checkpoints": 36, "events": 144, "interventions": 1152}`; training event audit: `{"runs": 18, "snapshots": 93, "events": 216, "interventions": 1728, "actor_policy_surrogate": 30, "training_gae": 186, "independent_mc": 0}`.

| Steps | Seed | Target | First actual ready behavior step | Final action counts | Greedy return | MC MSE | Target updates including init |
|---|---:|---|---:|---|---:|---:|---:|
| 2048 | 8 | ema | None | [0, 0, 0] | -458.000 | 4464.854 | 4 |
| 2048 | 8 | hard | None | [0, 0, 0] | -458.000 | 4464.824 | 4 |
| 2048 | 8 | none | None | [0, 0, 0] | -458.000 | 4464.824 | 0 |
| 2048 | 9 | ema | None | [2, 1, 0] | -458.000 | 3629.833 | 4 |
| 2048 | 9 | hard | None | [2, 1, 0] | -458.000 | 3629.939 | 4 |
| 2048 | 9 | none | None | [2, 1, 0] | -458.000 | 3629.939 | 0 |
| 2048 | 10 | ema | None | [0, 0, 0] | -458.000 | 4347.601 | 4 |
| 2048 | 10 | hard | None | [0, 0, 0] | -458.000 | 4347.609 | 4 |
| 2048 | 10 | none | None | [0, 0, 0] | -458.000 | 4347.609 | 0 |
| 8192 | 8 | ema | 7001 | [2, 3, 4] | -500.000 | 4048.033 | 10 |
| 8192 | 8 | hard | 7001 | [2, 3, 4] | -500.000 | 4047.966 | 10 |
| 8192 | 8 | none | 7001 | [2, 3, 4] | -500.000 | 4047.966 | 0 |
| 8192 | 9 | ema | 7001 | [42, 24, 30] | -500.000 | 1014.873 | 11 |
| 8192 | 9 | hard | 7001 | [42, 24, 30] | -500.000 | 1014.758 | 11 |
| 8192 | 9 | none | 7001 | [42, 24, 30] | -500.000 | 1014.758 | 0 |
| 8192 | 10 | ema | None | [1, 4, 4] | -458.000 | 3981.820 | 10 |
| 8192 | 10 | hard | None | [1, 4, 4] | -458.000 | 3981.670 | 10 |
| 8192 | 10 | none | None | [1, 4, 4] | -458.000 | 3981.670 | 0 |

Nine matching short/long experience prefixes hash exactly through2,048 actions/observations. Configured proportional exploration schedules differ, but short actors never become ready and long actors first become ready at7,001, after their4,096-step schedule; all ready behavior starts at epsilon.05. Final budget partial-rollout updates differ, so this is not checkpoint/environment resumption or a pure continuation from the short selected state.

Seeds8/9 become ready at7,001, only1,192 remaining training interactions; seed10 stays unready with[1,4,4] despite nine cases. Seeds8/9 final greedy return−500 is a learned but ineffective policy; seed10−458 is uniform fallback. Readiness is not policy success.

Seed8 has nine raw-positive admissions but zero true-terminal positives. Positive advantage can reflect value differences/calibration error on all-negative-reward trajectories; it does not certify success. Seed9 has96 positives at8,192,including five true-terminal positions, but its greedy policy still fails. The maintained criterion correctly uses actual raw GAE sign rather than normalized/clipped sign or reward sign.

All six hard/none final critic parameter states match exactly at interval1. No intervening gradient update exists between batches, so hard synchronization restores the online value function for the next batch. EMA global/per-case parameters lag, but fixed GAE labels for existing IDs do not change: all12 EMA/hard final online/target active label tensors equal despite distinct storage. Inactive target rows can retain lagged values and must not be treated as active evidence. This default does not test mutable/trainable-label lag.

MC values remain much more optimistic than negative discounted-return targets. Explicitly compare fixed/mutable/trainable/hybrid GAE labels and report whether mutable matching actually executes; do not insert invented value priors or nonpositive actor cases to force readiness. No Stage A gate or broad return guarantee.

Frozen source `dc26517dfadb6d13be47acd74e974d1a3444ce75b848431e97a3ecadbfda9add`; ZIP SHA256 `c0828c5fabe1bf5b9571372a915a46d93d3f0d72d3fd7860d4094bb898d02ba0`. Existing187-test implementation boundary remains unchanged for this diagnostic package.
