# Bounded training contribution and target execution audit — 2026-10-04

Opt-in `case_audit_queries_per_batch` (default0, allowed0..32) records pre-gradient
snapshots and the first configured queries of each training batch. Critic GAE
targets remain separate from independent MC. Actor surrogate uses actual
epsilon-mixed action probability and normalized/clipped training advantage;
raw advantage is separately recorded for admission diagnosis. Uniform behavior
epsilon1 has zero case-attributable surrogate delta. Unready executed policies
are excluded; their readiness is saved per transition and reset per episode.

The sink saves role/target state before optimization and insertion, verifies
audit predictions against actual batch predictions, and restores module modes.
Paired observation-on/off training tests yield identical actor/critic/target
weights and final evaluation. No audit target changes optimization or memory
admission. Sampling/adapters with undefined audit semantics remain rejected.

NN/NN, CartPole/Acrobot, seeds8/9/10, target EMA/hard/online,2,048 steps:
18 runs,159 safe pre-gradient snapshots,318 GAE events and282 actual behavior
surrogate events; all4,800 direct fixed-set interventions replay. Separately,
144 MC events/all1,152 interventions replay from36 safe final states. Six EMA
reference pairs exactly match earlier unobserved training weights/evaluation.
162 combined tests pass.

Frozen source `ad6e33ee01990aa654db54e1567e8035257f26b4ef339b2132b3be81f91924b5`;
122 files; ZIP SHA256
`9df89ed43478ece882abdda5357baa67b414560fa7c54663e39a6c640335df1d`.
Artifacts `results/t2_pi20260920/continuation_20261004/target_audit_2048/`.
Task output `neural-cbr-t2-target-audit-results.md`, independent MC and training
verification JSONs store actual snapshot hashes, events and exact comparisons.

CartPole returns (EMA/hard/online)145.111/218.111/304.889 mean across3 seeds;
too exploratory to change default. All Acrobot modes remain unready with
actor counts[0,3,0] and uniform fallback-458. There are only3 optimizer batches
on Acrobot versus synchronization interval4: target sync count1 means initial
copy only. Hard/EMA therefore have no executed batch update contrast. Repeat
both with common interval1 before attributing equivalence to target modes.

Actual recorded interventions test instantaneous loss, not episodic return
utility or learned maintenance benefit. Queries are bounded prefixes, not all
transitions. Continue value/GAE warm-start diagnosis, role-specific provenance
and optimizer-row preservation; the Stage A gate remains unpassed.
