# Case-capacity ablation — CartPole hybrid NN-kNN-RL

> **CORRECTION 2026-09-03 (p17).** The return half of this report does not
> replicate. A full 5-seed CPU replication on g234
> (`results/rl_capacity_cpu_g234/`, manifest
> `reports/p17_capacity_cpu_manifest.json`) puts capacity 2000 at
> **499.88 ± 0.27** against the **471.53 ± 14.35** below, with no monotonic
> capacity trend at all — the weakest CPU arm is capacity 500, not 2000, and
> the three seeds shared with this table moved 455.90→500.00, 484.10→500.00,
> 474.60→500.00. **Do not cite "bigger is worse" as a property of the
> method**; it is device-dependent.
>
> The cost half DOES replicate, and closely: a controlled sequential
> measurement (`reports/p18_throughput_controlled.csv`) gives a **7.38x**
> throughput penalty from capacity 100 to 2000, against the 7.4x reported
> here — on different hardware, OS, device and float path.
>
> The supported claim is therefore: a larger case base buys **no return
> improvement** while costing ~7.4x throughput. Capacity 1000 was added as
> the knee point and shows no deficit (498.48 ± 3.40). The CUDA arm below is
> still n=3; p14 seeds 3-4 remain in flight on r760. See the 2026-09-03 p17
> entry in reports/experiment_log.md.


Plan knob 5 / NEXT_STEPS_PLAN.md ask 1. Protocol identical across arms:
`tools/run_rl_nnknn.py cartpole --profile fast --seed S --device cuda
--critic-type nnknn --critic-mutable-value-labels --critic-trainable-value-labels
--case-capacity C`, early stopping ON (patience 30, min-delta 1.0, min-steps
25k, immediate stop at mean return 500), 150k step budget. The 500 arm is the
profile default and is the existing Gate-G1 seed batch (same profile, same
flags, same seeds) — not a re-run.

## Performance (protocol metric: final_eval mean return over 20 eval episodes)

| capacity | n seeds | final_eval mean ± sd | per-seed | solved (target-score stop) |
|---|---|---|---|---|
| 100 | 5 | **498.83 ± 2.62** | 500.00, 500.00, 494.15, 500.00, 500.00 | 4/5 |
| 500 (default) | 5 | 497.98 ± 4.52 | 500.00, 500.00, 500.00, 500.00, 489.90 | 4/5 |
| 2000 | 3 (of 5) | **471.53 ± 14.35** | 455.90, 484.10, 474.60 | 0/3 |

Seeds 3–4 of the 2000 arm are re-running (`p14_cap2000_s{3,4}.log`); the first
attempt (`p10_cap2000_s{3,4}.log`) was killed at ~28k/150k steps when its
parent session died. The 2000 row is therefore n=3 and provisional.

## Diagnostics

- Every 2000-arm run ends `budget_exhausted` and **none reaches the 500 target**,
  while 4/5 runs in each of the 100 and 500 arms stop early on target score
  (as early as 19,430 steps at capacity 100, seed 3).
- Two of the three 2000-arm runs have `best_model_step == 150,000` with
  `final_eval == last_eval` (still improving at the budget wall); the third
  peaks at 117,808. The 2000 arm is therefore **slower to consolidate**, not
  obviously worse asymptotically — the deficit is a fixed-budget statement.
- Case occupancy: 100-arm runs sit at capacity (100/100 actor/critic, one seed
  at 91); 2000-arm runs also fill (2000/1834–2000), so the larger memory is
  genuinely used rather than left sparse.

## Retrieval cost (the trade-off this ablation exists to measure)

Throughput, measured from runs launched **concurrently on the same GPU at
01:18 UTC 2026-08-28**, so contention is matched across arms:

| capacity | steps/second | source |
|---|---|---|
| 100 | 4.70 (seed 3), 4.42 (seed 4) | completed runs, `/usr/bin/time -v` |
| 2000 | 0.60 (seed 3), 0.60 (seed 4) | rate over 13.1 h before the runs were killed |

A 20× larger case base costs roughly **7.4× throughput** — exact retrieval is
linear in active cases — *and* loses ~27 points of return at a matched 150k
step budget. On this task the case base is not a lever worth scaling up.

## Interpretation

Capacity 100 matches the 500 default within seed noise (498.83 ± 2.62 vs
497.98 ± 4.52), so CartPole's useful case base is small; capacity 2000 is worse
on the protocol metric at fixed budget and far more expensive per step. The
honest framing for the manuscript is a two-axis trade-off — return at fixed
budget *and* wall-clock per step — rather than a monotone "more memory is
better" story. It also turns the wall-clock weakness into a measured number
instead of an omission.
