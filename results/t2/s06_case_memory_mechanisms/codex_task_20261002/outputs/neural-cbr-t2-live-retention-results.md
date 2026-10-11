# T2 optional live training retention

30 original actual trainings plus30 final-source instrumented retrainings (two additional earlier 30-run validation replays remain preserved); CartPole/Acrobot NN/NN, three seeds,2,048 exact interactions,128 capacities,top-k8. Modes:off,noop fraction1,actor/critic/both fraction.9. Current batch evenly selected up to16 training queries, strict zero mean/per-query loss increases, frequency512 at completed batch boundaries, original bias-prune disabled, ID-preserved Adam. No diagnostic-guided retention.

Actual replay counts: `{"actual_retrainings": 30, "noop_pairs": 6, "reference_batches": 257, "training_queries": 7418, "retention_calls": 99, "candidates": 47701, "removed": 821, "moment_tensors": 2588, "other_optimizer_states": 7108, "target_alignments": 589, "snapshots": 99, "completed_snapshots": 99}`; independent MC replay: `{"runs": 30, "checkpoints": 60, "events": 240, "interventions": 1920}`.

| Mode | CartPole return mean | CartPole per-seed returns | Acrobot return mean |
|---|---:|---|---:|
| off | 244.889 | 214.000, 181.667, 339.000 | -458.000 |
| noop | 244.889 | 214.000, 181.667, 339.000 | -458.000 |
| actor | 148.444 | 190.000, 104.333, 151.000 | -458.000 |
| critic | 287.889 | 293.000, 70.667, 500.000 | -458.000 |
| both | 155.222 | 222.000, 128.667, 115.000 | -458.000 |

| Mode/task/seed | Actor/critic final capacity | Own-policy MC MSE | Fixed off-policy/bootstrap MC MSE |
|---|---|---:|---:|
| off/cartpole/s8 | 128/128 | 2047.291 | 2047.291 |
| off/cartpole/s9 | 127/128 | 1948.091 | 1948.091 |
| off/cartpole/s10 | 124/128 | 3435.767 | 3435.767 |
| off/acrobot/s8 | 0/128 | 4464.857 | 4464.857 |
| off/acrobot/s9 | 3/128 | 3629.828 | 3629.828 |
| off/acrobot/s10 | 0/128 | 4347.601 | 4347.601 |
| noop/cartpole/s8 | 128/128 | 2047.291 | 2047.291 |
| noop/cartpole/s9 | 127/128 | 1948.091 | 1948.091 |
| noop/cartpole/s10 | 124/128 | 3435.767 | 3435.767 |
| noop/acrobot/s8 | 0/128 | 4464.857 | 4464.857 |
| noop/acrobot/s9 | 3/128 | 3629.828 | 3629.828 |
| noop/acrobot/s10 | 0/128 | 4347.601 | 4347.601 |
| actor/cartpole/s8 | 115/128 | 563.400 | 2086.772 |
| actor/cartpole/s9 | 115/128 | 1390.078 | 2018.143 |
| actor/cartpole/s10 | 118/128 | 2759.407 | 3548.638 |
| actor/acrobot/s8 | 0/128 | 4464.857 | 4464.857 |
| actor/acrobot/s9 | 3/128 | 3629.828 | 3629.828 |
| actor/acrobot/s10 | 0/128 | 4347.601 | 4347.601 |
| critic/cartpole/s8 | 123/115 | 1794.356 | 2009.158 |
| critic/cartpole/s9 | 127/115 | 1413.637 | 2009.852 |
| critic/cartpole/s10 | 128/115 | 3973.296 | 3419.298 |
| critic/acrobot/s8 | 1/115 | 4304.604 | 4280.216 |
| critic/acrobot/s9 | 3/124 | 3264.033 | 3279.517 |
| critic/acrobot/s10 | 1/122 | 4142.940 | 4109.519 |
| both/cartpole/s8 | 117/114 | 1584.854 | 2137.698 |
| both/cartpole/s9 | 115/115 | 774.201 | 1771.311 |
| both/cartpole/s10 | 115/115 | 306.870 | 2706.457 |
| both/acrobot/s8 | 1/115 | 4304.604 | 4280.216 |
| both/acrobot/s9 | 3/124 | 3264.033 | 3279.517 |
| both/acrobot/s10 | 1/122 | 4142.940 | 4109.519 |

Six noop pairs exactly preserve actor/critic/target/Adam/greedy/MC. Independent full retraining reconstructs actual original GAE and executed ready-policy reference fields for257 batches/7,418 selected queries, all99 live selection calls and47,701 candidates.821 removals,2,588 moment tensors,7,108 other optimizer states and589 structural target alignments verify;99 pre-retention snapshot hashes validate. Every full-bank candidate/feasible greedy choice is replayed; live selection leaves fixed shared parameters unchanged.

Actor-only retention worsens all three CartPole returns (mean244.889→148.444). Its own-policy MC MSE appears lower, but under a common fixed off-mode actor and fixed off-mode time-limit bootstrap all three value MSEs worsen. Both-mode s10 own-policy MC MSE3435.767→306.870 accompanies lower rollout return299.5→85, target mean69.731→33.138 and sample count599→170; fixed-policy MSE is2706.460 instead. These are different target/trajectory populations.

Critic-only returns improve two CartPole seeds and worsen one; mean287.889, with actual actor counts/update trajectories affected by new future advantages. Acrobot has only0/1/3 actor cases below its action coverage/readiness floor; all reported−458 greedy returns remain fallback rather than learned readiness. Common-policy critic MSE improves all three Acrobot seeds, but does not solve actor startup.

The new option is off by default and requires explicit preserve_by_id. Source budgets/reference sampling/frequency are exploratory; no Stage A gate. Equal interaction budgets do not imply equal compute, update counts or final capacities. Next study matched capacity/cost and actor causal return constraints or structured policy protection; keep independent MC diagnostic and address negative-reward value startup separately.

Source `dc26517dfadb6d13be47acd74e974d1a3444ce75b848431e97a3ecadbfda9add`; ZIP SHA256 `c0828c5fabe1bf5b9571372a915a46d93d3f0d72d3fd7860d4094bb898d02ba0`. Combined187-test suite passes, existing scalar-conversion warning only.
