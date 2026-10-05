# T2 fresh training-reference retention: exploratory compaction

`model/t2/retention.py` implements post-checkpoint, fixed-parameter full-bank
removal with refill. This is distinct from the fixed actual retrieval-set,
no-refill contribution audit. Critic references must be training GAE targets;
actor references must contain actually ready behavior actions, signed normalized
advantages and the executed epsilon mixture. Independent MC cannot select.

For each round, evaluate every eligible removal against the current accepted
bank. Sum positive/negative query loss differences into fresh C/H and smoothed Q.
Filter candidates by mean loss relative to the original bank plus an explicit
absolute budget BEFORE ranking by lowest Q, loss and stable ID. This exploratory
selector is not a derived/approved PI formula. Historical Q is not treated as
current calibrated reliability. Original unobserved cases remain unknown and
protected throughout; actor action-count floors and explicit ID protection also
apply. Stop before requested capacity if no guarded deletion is feasible.

All trial evaluation precedes compaction, without gradients. The caller must
align optimizer moments and lagged target rows by retained IDs. The maintained
`tools/t2_retention_pilot.py` performs and verifies that alignment, preserving
target labels independently and all non-role/shared parameters. This component
is not yet inserted into the live training maintenance schedule.

Source `6645e6d173eb639f47998c5e232d4d41c42dfcb9bd5421aa7364bc32d00ddf23`;
131 files; ZIP SHA256
`e88951fda8ee9bcc50818efe22d3fa772f91933b945aaece31a30137d56d3c2d`.
Frozen upstream training checkpoints use source63f0eee9ad5a; six additional
preserve_by_id 2,048-interaction trainings use the current frozen source.
CartPole/Acrobot, three seeds, NN/NN: 72 one-role post-checkpoint conditions,
two target fractions(.75/.9) and absolute budgets(0/.03), across reset and
preserve modes. Use the latest two recorded training queries per role.
Twenty-four Acrobot actor conditions are unavailable: no ready-policy training
references, no fabricated fallback credit.

All7,468 full-bank candidate trials and greedy feasible choices replay;
737 removals across conditions, only16 reach requested capacity. Reset
checkpoints have no private-case Adam moments at this post-insertion phase;
the36 preserve conditions check144 actual nonempty moment tensors by ID,
inactive rows zero, scalar Adam steps unchanged. Four focused tests and the
combined181-test suite pass (existing scalar-conversion warning).

Training loss guards do not guarantee independent behavior or calibration:
34 conditions worsen independent MC MSE and four worsen three-episode greedy
return. Critic-only compaction leaves actor greedy return exactly unchanged;
actor compaction changes the MC rollout distribution, so its MSE difference
cannot isolate critic calibration. Diagnostics are never used for selection.
Both references and diagnostics are small and reused; no gate/generalization
claim. Increase explicit training-reference coverage next, study stale targets
and advantages, then validate optional live maintenance against matched controls.

Evidence is in task `work/retention_pilot_v3/`,
`work/retention_preserve_training/`, `work/retention_preserve_pilot_v3/` and
`outputs/neural-cbr-t2-retention-{results.md,verification.json}`. Initial failed
pilot attempts (optimizer-key/task metadata handling) remain preserved; corrected
frozen source runs all conditions. The upstream18 authoritative documents remain
pinned to career-2027 cd77277600841438d98ac5793867da28b8ec6dd0, rechecked Oct5.
