# T2 reference coverage and individual-query loss guard

The preceding two-query retention pilot found return regressions despite its
aggregate training loss guard. `max_query_loss_increase` now optionally constrains
EVERY reference query against its original loss vector, in addition to the mean
loss budget. Both constraints filter candidates before Q ranking. Defaults keep
the original behavior; finite nonnegative explicit query budgets only. A focused
two-query counterexample shows a mean-feasible deletion improving one loss from1
to0 while worsening another from1 to4; the strict query guard rejects it.

The maintained pilot adds `--reference-batches` and `--query-loss-budget`, records
reference steps, and excludes later/same-step references for best-evaluation
checkpoints conservatively. GAE targets and executed signed actor advantages
remain their recorded training values; current selected parameters reevaluate
them. This is broader coverage of historical training evidence, not proof of
freshness, return utility or target calibration. No independent MC selection.

Same six preserve_by_id NN/NN 2,048-interaction checkpoints,108 one-role
conditions: control latest one observed batch, wide latest up to eight observed
batches, strict guard same wide queries plus zero individual-query increase.
CartPole wide references16 queries; Acrobot4 queries (three training batches,
first empty critic has no pre-gradient case observations). Maximum reference
age1,553 environment steps. Actor Acrobot remains unavailable. Diagnostics,
budgets and requested capacities match; actual retained capacities may differ.

| Mode | Candidate replays | Removed | Reached capacity | MC MSE worse | Greedy return worse |
|---|---:|---:|---:|---:|---:|
| control | 3,758 | 364 | 6/36 | 20/36 | 4/36 |
| wide | 14,533 | 629 | 19/36 | 15/36 | 6/36 |
| strict query guard | 11,368 | 392 | 10/36 | 6/36 | 6/36 |

All29,659 candidate trials replay independently;432 actual nonempty moment
tensors verify ID preservation/zero inactive rows, scalar steps unchanged and
lagged targets maintained. Control exactly reproduces all36 prior preserve-mode
accepted IDs/loss vectors/greedy/MC diagnostics. Combined182 tests pass.
Source `847e751d081c0116832304a20d5613799fe26ca8b81af04da9de1442389a0355`;
131 files; ZIP SHA256
`4fac8324126e7bcfab762f13db04d7338c61d4af75c595d9ac69115f662b8a14`.

Broader references allow more compression, with a much higher search cost;
per-query protection reduces MC regressions in this reused exploratory set,
but does not eliminate actor return regressions. Different retained capacities
and MC behavior distributions prevent a clean general-performance conclusion.
Neither guard is a Stage A gate. Next integrate an optional live training-only
maintenance transaction and compare matched controls, with explicit references,
unknown/action coverage, correct moment/target alignment and recorded costs.
Also investigate Acrobot value startup and target/advantage calibration rather
than admitting nonpositive actor cases to manufacture readiness.

Evidence:task `work/retention_reference_{control,wide,guard}/` and
`outputs/neural-cbr-t2-reference-guard-{results.md,verification.json}`.
