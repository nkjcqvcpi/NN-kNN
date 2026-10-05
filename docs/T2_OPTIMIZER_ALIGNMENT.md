# Explicit case optimizer maintenance comparison — 2026-10-04

`case_optimizer_maintenance` remains `reset` by default. The experimental
`preserve_by_id` path moves private case Adam moments after staged insertion
and scheduled compaction by stable IDs, zeroing new/inactive rows and preserving
shared/other-role states. Labels, biases and glocal row parameters use the same
identity mapping; joint optimizer ownership remains separate per role.

Scalar Adam step is global per vector parameter and stays unchanged, including
new rows with zero moments. This does not simulate independently initialized
per-case optimizers. Tests exercise real Adam/AMSGrad rows, trainable labels,
compaction/admission/continuation, and another role/shared trunk invariance.

CartPole/Acrobot NN/NN, seeds8/9/10,2,048 interactions,128 capacity,top-k8,
EMA interval4:12 reset/preserve runs. Source
`bd1521b04c6bc91b6b7709e7a920d9315d9101eaa64ab88387bd13c26b7b1043`;
124 files; ZIP SHA256
`af3f9c3fd63a62aa00e89a5bc6cc5e9ef8f75328685ae3723ef1c8e100ead88d`.
All12 actual training reruns reproduce final actor/critic/target/evaluation
exactly.141 maintenance operations check496 moment tensors,248 scalar steps
and377 other-role/shared states. Six reset controls precisely reproduce the
historical reference. Separately24 safe states,96 MC events/all768 deletions
replay;165 combined tests pass.

CartPole paired return improvements are184.000/67.000/216.667, mean145.111→
301.000. Acrobot remains unready and unchanged. Actual actor capacities
reset/preserve are128/117,128/128,128/128 on CartPole; optimizer batches
18/16,19/17,13/18. Equal interaction budget/capacity upper bound does not mean
equal compute, update count or final capacity. Changed visitation prevents
interpreting cross-condition MC MSE as fixed-data critic degradation.

Artifacts `results/t2_pi20260920/continuation_20261004/optimizer_alignment/`;
task outputs `neural-cbr-t2-optimizer-results.md`, MC and actual retraining
verification JSONs. The optional preservation mechanism is not a Q selector,
an approved gate condition or evidence of general suitability. Keep reset as
frozen reference while testing broader budgets, roles and quality maintenance.
