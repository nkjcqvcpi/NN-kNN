# Critic and target retrieval identity — 2026-10-04

The maintained value network had two stable-ID streams: outer memory IDs began
at zero while its preallocated core began at capacity. Target row alignment
copied outer IDs but left internal retrieval IDs stale. A five-capacity fixture
admitted outer IDs `[0,1]` with core `[5,6]`; after removing 0 and adding 2,
online core was `[6,7]` and target core `[5,6]`, despite outer `[1,2]`.

Outer critic IDs now remain authoritative for backward-compatible checkpoints.
Initialization, admission, compaction, target alignment and state restoration
copy both ID buffers and next-ID counters to the core. Legacy checkpoints retain
values and predictions while their redundant retrieval IDs are repaired. Actor
memory retains its separate ID namespace; no cross-role compaction is added.

Actual before/after replay preserves target labels `[200,30]` and biases `[8,0]`
by retained identity, shares only raw cases, and keeps distinct label storage.
An ID-1 leave-one-out query previously assigned that case weight 0.9999802;
after repair its weight is exactly zero, remaining case weight one. This fixes
identity-based exclusion and audit semantics; ordinary unmasked prediction was
not demonstrated to improve, and this is not a Stage A performance result.

Three tests exercise target lag/EMA, ID-based exclusion, legacy restore and full
capacity replacement. The maintained NN-kNN-RL smoke passes, including all four
actor/critic combinations, compaction and target EMA. Frozen pre-fix source is
the 295c4cea1b8a reference-guard archive; task outputs preserve actual before/
after JSON under `neural-cbr-t2-identity-before.json` and `...-after.json`.

Next transfer work: role-specific actual retrieval events and counterfactual
loss contributions; training GAE and independent discounted-MC audit targets
must remain separate. Four-role matched pilots precede a confirmation gate.
