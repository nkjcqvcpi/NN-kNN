# Role/target quality ledger and selected provenance — 2026-10-05

`case_quality_tracking` defaults off and requires explicit bounded audit queries.
`QualityLedger` validates an entire actual event before committing any row:
role/target stream, snapshot digest, phase/step, active unique IDs, retrieval
mass, exactly one intervention per positive-weight case, direct loss Delta,
C/H/Q and consistent smoothing. Duplicate event IDs are idempotent; changed
content reusing an ID is rejected. Nonfinite/overflow support and corrupted
restored statistics are rejected. No validation failure partially updates rows.

Actor surrogate, critic training GAE and independent MC use separate keys,
never a mixed Q. Each row records exposure, C/H, smoothing/Q, last phase/step
and actual snapshot hash. Deleted-case history remains archived; active views
filter by current stable IDs. Missing observations remain unknown, not harmful.
Historical sums across changing learned states are not calibrated reliability.

Optional independent holdout callbacks store post-batch MC separately from
pre-gradient snapshots, even at the same environment step. Existing holdout
RNG restoration and gradient/admission exclusion remain. Best checkpoints
preserve the selected quality ledger together with role/target/Adam state;
later final-training events do not leak into an earlier selected ledger.

Source `63f0eee9ad5a0bf0c92ba44d9b28b999a8c64d2120c0f4219242eb0bbef27562`;
128 files; ZIP SHA256
`504d1535a34379d672f8b7da84a29ad1c5ff62674a36b09c0d1b4b013775c0c7`.
CartPole/Acrobot NN/NN,three seeds,2,048 interactions: six tracking-on/off pairs
(12 actual trainings) reproduce actor/critic/target/Adam/evaluation/independent
MC exactly.106 safe snapshots,306 events (94 actor surrogate,106 GAE,106 MC)
and all2,448 actual fixed-set interventions replay.930 separate historical
case rows agree with independent C/H/exposure/Q sums.177 combined tests pass,
including atomic rejection, dedup/roundtrip, and forced earlier checkpoint
ledger restoration with all actual snapshot hashes checked.

Current active coverage is sparse. CartPole actor observed30/33/31 of128;
critic GAE22/18/25 and MC22/21/24 of128. Acrobot actor counts0/3/0 have no ready
behavior observations; critic GAE11/15/14 and MC10/14/8 of128. Unknown cases
cannot be penalized solely because the bounded query prefix missed them.

Initial source97a074955f93 six512-step and six2,048-step runs stay preserved.
Acrobot512 provides no pre-gradient quality evidence before its first insertion;
2,048 was added explicitly. A finite-input accumulated-mass overflow guard was
then added and the complete source used for all12 paired real trainings,
with learning states matching earlier source. Raw experimental runs remain in
`results/t2_pi20260920/continuation_20261005/quality_ledger[_2048]/`; completed
paired replays reside in this task's `work/quality_complete/`. Task reports
`neural-cbr-t2-quality-results.md`, observer and direct verification JSONs.

This is quality/provenance infrastructure, not an activated quality selector.
Independent MC remains diagnostic. Next define explicit candidate retention
constraints for unknown/rare cases and statistic freshness, then test actual
selection/optimizer/target effects against frozen controls. Do not claim
return utility, broad coverage, calibration or Stage A success from this ledger.
