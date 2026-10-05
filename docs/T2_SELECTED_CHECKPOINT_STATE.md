# Selected RL optimizer/target consistency — 2026-10-05

Earlier best actor/critic parameters previously lacked matching optimizer and
lagged-target snapshots. Finalization restored earlier weights but retained
the final target labels/metric after row alignment. This could mix training
phases in later diagnostics or continuation. Previous fixed-final pilots do
not demonstrate this failure because their selected source was final.

Every best evaluation now snapshots actor/critic, target and actual Adam
states together. When choosing an earlier phase, finalization restores all of
them, detaching target raw bank before loading and then sharing/aliging raw
cases by stable ID. Checkpoints save selected optimizer and target states;
safe loading can optionally rebuild/restore joint or separate optimizers.
Legacy evaluation loading remains supported; explicitly requested optimizer
restore rejects old files without those states rather than inventing them.

This preserves selected learning state, not rollout/environment/RNG resume.
The checkpoint explicitly records that boundary. No full environment resume
interface is claimed. Separate actor/critic memories and optimizer groups,
shared representation and distinct target trainable values remain intact.

Four forced-earlier tests cover all actor/critic combinations: select64 after
executing256 interactions, compare exact live historical role/Adam/target
states, safely load, then execute identical real continuation steps. Legacy
evaluation and missing-optimizer rejection are exercised.169 combined tests
and maintained NN-kNN-RL smoke pass.

Actual four-role CartPole, seeds8/9/10,512 interactions, periodic128-step
evaluation:12 runs/24 safe final audit states;48 MC events/all384 removals
replay. Five select a nonfinal phase (four at earlier environment steps, one at
step512 before final partial-rollout optimization).21 Adam states and6 targets
restore exactly, and all12 identical actual optimizer continuation checks pass.
Same environment step is not proof of same optimization phase. The first
audit incorrectly required strictly smaller step for every best_eval; its
failed log is preserved and the phase-aware check corrected without changing
the experiment. Periodic selection adds evaluation looks; no performance
gain or matched-compute gate claim follows from this engineering repair.

Source `91b1ff4ca6d45d0decf891870b1f8a0bb767d253d21124aaf556c45f1e626980`;
125 files; ZIP SHA256
`bb71ceb185c74039fc109469070e3cd82d4d7c4ea66bc4123d6b94b0ce120705`.
Artifacts `results/t2_pi20260920/continuation_20261005/selected_checkpoint/`.
Task outputs `neural-cbr-t2-selected-state-verification.json` and
`neural-cbr-t2-selected-checkpoint-mc-verification.json` record counts and
per-run phases. Next integrate role-specific quality/provenance retention and
continue negative-reward calibration diagnosis before a frozen Stage A gate.
