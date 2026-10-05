# Adapted regression removal and matched continuations

The isolated adapted-removal implementation now supports the actual aggregate
regression residual adapter. Retrieval updates minimize final squared error
through a frozen adapter; adapter updates freeze retrieval. The unused
historical pair network stays frozen and is rejected if marked trainable.
Fixed training queries and stable-ID leave-one-out exclusions survive memory
compaction. No candidate reads validation/test for its training or selection.

Initialization freezes the selected core and fits aggregate residuals on
training-LOO inputs; validation selects the adapter and the Adam state from the
same epoch. Both selected optimizers are restored for each isolated fork.
Regression residual MSE and final MSE are algebraically equal here, so the
combined objective scales their sum rather than adding independent supervision.
The numeric ReuseConfig budgets are shared, but classification output/probability
options are unused; actual metadata says `aggregate_regression_residual`.
Initial, full-control and random-control artifacts now carry architecture,
trainability flags and optimizer epoch metadata for independent recovery.

`alignment_adapted_regression.yaml` prespecifies Energy/Yacht, seeds8/9/10,
18 training cases, K12, allowed mean reference-MSE increase.03, LR scale.1 and
two total phase epochs per candidate (retrieval2, adapter2, or one each).
Reference is the fixed original training-LOO stream, not current retained rows.
The random and full controls receive the accepted phase blocks; random matches
achieved capacity but does not select/stop using the loss budget.

Frozen source
`00f2a76aa401fe142923af622dd267e071358d963bf043f5e9c3c76052548809`,
113 files, ZIP SHA256
`8c361b7151947bf1d33feb213994b1937ea37e3dda4b17d5ff9cf40cbdb7246a`.
The 18 runs/72 states load safely and 1,944 final events replay. Independent
audit actually repeats all1,662 candidates and108 per-round controls, selected
paths, full and random trajectories. Model/adapter/Adam tensors match exactly;
IDs, protected floors, reference denominators, selected losses, budgets and
actual step increments pass. Static source/archived CSV/data byte hashes are
checked separately; actual split tensors are saved. Five focused regression
tests pass, including earlier-epoch Adam recovery and a real resumed update.
The combined T1/T3 suite passes150 tests at this package boundary.

All18 reach K12, yet12/18 have worse test RMSE than the starting state and matched
full controls;13/18 are worse than matched random. Yacht adaptive is worse than
random for every seed/scope in this pilot. Energy has mixed paired outcomes.
Random violates the adaptive reference budget in13/18 runs, so its predictive
results are not a same-constraint superiority result. No method has established
generalization safety: a training-LOO acceptance budget is insufficient.

Next diagnosis should compare training-LOO and independent maintenance-holdout
acceptance on the same split/initial state, with fixed thresholds and untouched
test outcomes. It must preserve failed capacity attainment rather than loosen
the budget after seeing test results. Broader capacity/scale/compute, constrained
adaptation and confirmation remain required; this pilot does not finish T1.
