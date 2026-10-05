# Capacity-matched synchronization comparison

The previous independent/RR/RRR runs used full memory for independent and RR,
and K=.75 memory for RRR. Their post-adaptation results combine synchronization
with a capacity difference. `alignment_sync_capacity_match.yaml` now runs the
same exploratory four-task, three-seed synchronization protocol, then applies
the same Q policy to the selected independent/RR checkpoints at K=.75. RRR is
also checked against that target and policy. This post-hoc comparison adds no
gradient steps and uses training leave-one-out statistics and final outcomes.

The original selected state before this post-hoc operation is saved separately
with its model, adapter and both Adam states. Final case-row moments are aligned
to retained stable IDs. A protocol file records target and achieved capacities,
policy, no finetuning and the checkpoint transformation. All predictions and
audits use the final adapted path. Selected free-radius calibration remains the
historical training calibration; it is not relabeled as calibrated on the new
post-hoc memory. The original proximity normalization scale is now saved.
Both snapshots include architecture/calibration metadata, selected optimizer
epochs and model/adapter trainability flags. Regression's unused pair adapter
stays frozen when reconstructing the aggregate-only optimizer. The first
36-run batch omitted metadata in the pre-compression artifact; it is preserved
and the corrected serialization requires a new frozen run.

History records actual optimizer updates and examples per phase and query-case
pairs at the capacity used for that epoch, before end-of-epoch maintenance.
Budgets distinguish total executed updates from those represented by the
validation-selected checkpoint. Synchronization elapsed time includes its
maintenance and validation. Query-case pairs describe exact retrieval workload;
they are not FLOPs or an equal-compute guarantee. Core training and preparation
costs remain separate, and different selected epochs are explicit.

The first development seed on Energy produced 45 final cases for each schedule,
with post-adaptation RMSE .380/.411/.354 for independent/RR/RRR. This one-seed
comparison is provisional. The frozen 36-run audit is complete; broader
matched-compute grids remain pending. Previous full-memory results stay
valid within their original scope and are not overwritten.

An actual compaction test verifies pre-maintenance retrieval workloads, fixed
examples and component update counts. A second test actually restores both
regression snapshots and their optimizers using the saved frozen flags.
All 134 current T1 tests pass at this
implementation boundary.

The corrected archive identifies source `27d65784029bda57d92136550a15bb731c80f0b1f0d405d19166a8d3b2862400`,
106 files and ZIP SHA256 `75c411a81df208b0e40eb7eb051c07074f8d0001dcfc37df304652ee7f953c22`.
All 36 runs and 72 states load safely; 4,554 final events replay. A second
manual p0/difference/residual reconstruction checks all 72 states and 9,108 test
predictions, including saved final vectors/scalars, IDs, activations and actual
post-adaptation metrics. Before-state metrics are freshly generated baselines.
Thirteen audit categories pass, including source/raw-input integrity, Q/floors,
selected/executed updates, actual Adam counts and stable-ID moment alignment.
Original and repaired serialization batches have identical mathematical outputs.

Three-seed mean post-adaptation results:

| Task | Independent | RR | RRR |
|---|---:|---:|---:|
| Synthetic diagnostic accuracy | .9564 | .9641 | .9679 |
| Iris accuracy | .9667 | .9667 | .9556 |
| Energy standardized-target RMSE | .3622 | .3722 | .3493 |
| Yacht standardized-target RMSE | .6403 | .6368 | .5843 |

RRR is descriptively better on two regression tasks at the matched final
capacity, and worse on Iris. Post-hoc compression also worsens mean regression
results for independent/RR relative to their full-memory states. This does not
establish a stable RRR benefit, equal compute or a scientific readiness gate.
All synchronization runs execute 20 epochs; selected epochs differ. Core
pretraining can stop earlier. An initial reporting error selected pre-adaptation
metrics for the tables despite successful per-key replay; it is preserved and
corrected to the actual final post-adaptation outcome. Future reports must use
explicit metric names rather than the first matching metric key.
