# Supervised synchronization verification — 2026-10-04

The continuous plan now includes regression synchronization using the maintained
aggregate NN-CDH adapter. The historical pair network is frozen and unused;
only the retrieved solution estimate and query-minus-neighborhood representation
enter the aggregate correction. External regression adapters also participate in
the actual provenance, removal loss and retrieval-event prediction paths.

Two training-state fixes accompany it: validation-selected retrieval weights
restore their corresponding Adam moments, including epoch-zero fine-tune
fallbacks; synchronization preserves each parameter's original freeze flag
instead of enabling every model parameter after the adapter phase. Selected
synchronization checkpoints restore optimizer and free-radius states together.

54 tests pass. All five maintained smoke modes (imports, supervised train,
DQN, NEC, NN-kNN-RL) pass. These smoke checks validate engineering operation,
not policy quality at meaningful interaction budgets.

`alignment_sync.yaml` runs synthetic_diag/iris classification and
energy_efficiency/yacht regression on seeds 5/6/7, with independent,
alternating retrieval/reuse, and alternating retrieval/reuse/retention schedules:
36 runs. Core training is capped at 30 epochs, sync at 20, and RRR maintains
capacity at epochs 8/16 with K=0.75. Full-memory controls retain original
capacity; RRR retains 324/432 synthetic cases or 45/60 cases on the other tasks.
Target preprocessing and the regression success tolerance remain explicit.

All 36 checkpoints and adapters reproduce saved pre/post decisions, continuous
regression predictions and test metrics at 1e-6. Required artifacts, finite
metrics, phases/losses, RRR safe checkpoints, capacities, and free-radius fields
pass checks. Active-case C+H=A differs by at most 2.801e-6, within the declared
1e-5 numerical check. The source ZIP passes integrity and reproduces hash
`df4c004c5a6805ecc7088b9cbb9c1c261bd8e9f6308df3403e8b4adc2857f440`.
Runs identify the frozen dirty implementation based on `3e6869c`; no source
changed during execution. Raw artifacts and the loss histories are under
`results/t1_pi20260920/continuation_20261004/alignment_sync/`.

For regression, L_delta and L_post are algebraically the same squared residual;
both logged values are consistent to 1e-6. Their weights combine into one
effective error weight. They must not be counted as two independent mechanisms.
Small-correction penalties, locality and pre-adaptation task loss remain distinct
terms. Capacity differences confound pure schedule comparisons, and three seeds
do not establish confirmatory benefit or equivalence.

Detailed reports list every dataset/seed/treatment and paired post-metric changes,
radius diagnostics, and intervention epochs. The next engineering package is
reversible reviewer actions and faithful M0/M1/M2 artifacts; broader comparisons
and the scientific gates remain tracked in `../T1_EXECUTION_LEDGER.md`.
