# T2 optional current-batch retention in actual training

Live retention is explicit/off by default (`case_retention_keep_fraction=None`).
Enabling it requires preserve_by_id Adam maintenance, positive frequency,
bounded1–128 queries, finite nonnegative budgets and actor/critic/both roles.
Direct dataclass callers are validated before environment creation, too.

Use evenly selected CURRENT-batch GAE targets for the critic and actually ready
executed actions/signed normalized advantages/epsilon for the actor. No MC
queries or diagnostics select deletion. The operation occurs after gradient
optimization, raw-positive actor admission, critic insertion and existing
scheduled pruning, before target metric/label synchronization. Align structural
target rows before the safe snapshot; after compaction preserve Adam by stable
ID and lagged target labels independently. Retired quality history remains
history, not current calibrated scores. Unknown original-reference cases are
protected; both per-action coverage and configured behavior readiness count
are protected. Extreme requested compression cannot cross the readiness floor.

`case_retention_events.json` records current reference indices/values, phases,
bootstrap source, safe snapshot hashes, every candidate loss vector/C/H/Q,
constraints, actual capacity, optimizer alignment and search seconds. Skipped
unready actor references are explicit. Defaults preserve prior behavior.
The maintained role pilot exposes these options. No rollout/environment/RNG
resumption claim is added to selected checkpoints.

Thirty original2,048-interaction NN/NN trainings: CartPole/Acrobot, three seeds,
off / fraction1 noop / actor / critic / both. Active fractions.9, frequencies512
at completed batch boundaries,16 evenly covered current queries, strict zero
mean/per-query increase,capacity128/top-k8. Disable original bias-prune for this
isolated comparison; all use preserve_by_id. Six noop controls exactly preserve
actor/critic/target/Adam/greedy/independent-MC states. Complete instrumented
retraining verifies actual original GAE/ready behavior inputs, fixed parameters
within each selection, every candidate and feasible choice, all moment/target
alignments and final learning/evaluation states.

Each30-run replay checks257 reference batches/7,418 selected training queries,
99 actual selection calls/all47,701 candidates/821 removals,2,588 moment
tensors/7,108 other optimizer states/589 structural target alignments,99
pre-retention snapshots. Thirty final-state MC audits replay240 queries/all1,920
actual fixed-set removals independently. The complete source also repeats the
30 original trainings exactly; prior implementation/validation freezes and
their replays remain preserved.

| Mode | CartPole mean greedy return | Per-seed returns | Acrobot mean |
|---|---:|---|---:|
| off / noop | 244.889 | 214 / 181.667 / 339 | -458 |
| actor | 148.444 | 190 / 104.333 / 151 | -458 |
| critic | 287.889 | 293 / 70.667 / 500 | -458 |
| both | 155.222 | 222 / 128.667 / 115 | -458 |

Actor-only loses all three seeds. Critic-only improves two, loses one. Acrobot
actors still have0–3 cases below readiness/action coverage; its−458 returns
remain fallback. Critic retention changes future GAE/admission even without
directly touching actor rows. Equal interactions do not mean equal updates,
compute or final capacity; capacity-matched confirmation remains necessary.

Eighteen additional evaluations fix the SAME off-mode actor and off-mode lagged
time-limit bootstrap, with identical target means/sample counts/returns and all
eight bounded query/target pairs exactly matching. Actor-only's own-policy MC
MSE falls on all CartPole seeds, but common-policy MSE worsens on all three.
Both-mode seed10 MSE3435.767→306.870 also changes MC rollout return299.5→85,
target mean69.731→33.138 and samples599→170; common-policy MSE2706.460 is a
different, more comparable diagnostic. Thus lower own-policy MSE alone does not
show critic calibration or a better policy. Common-policy Acrobot critic MSE
improves all three seeds, without resolving actor startup. All diagnostics are
excluded from retention, gradients and admission.

Five live tests cover configuration/direct dataclass validation, exact no-op
learning state, actual deletion/safe restore and extreme readiness protection;
combined187 tests pass with the existing scalar-conversion warning.
Final source `dc26517dfadb6d13be47acd74e974d1a3444ce75b848431e97a3ecadbfda9add`;
132 files; ZIP SHA256
`c0828c5fabe1bf5b9571372a915a46d93d3f0d72d3fd7860d4094bb898d02ba0`.
Original sourceb1a804d18abf and direct-validation source5eafc20eb29f are retained.

Evidence:task `work/live_retention_{off,noop,actor,critic,both}/`, instrumented
`work/live_retention_replayed/`, `work/live_retention_completed/`, final
`work/live_retention_ready_completed/`; reports
`outputs/neural-cbr-t2-live-retention-results.md`, original/complete/ready and MC
verification JSONs, `neural-cbr-t2-live-fixed-policy-verification.json`.
This is an optional exploratory implementation, not Stage A success. Next
separate actor return protection from surrogate guarantees, confirm matched
capacity/compute, and investigate negative-reward startup with extended actual
training and controlled target modes before inventing warm-start case labels.
