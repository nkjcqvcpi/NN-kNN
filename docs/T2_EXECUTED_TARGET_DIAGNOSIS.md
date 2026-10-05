# Executed target updates and admission counterexample — 2026-10-04

The pilot now accepts an explicit target synchronization interval. Acrobot,
NN/NN, three seeds and EMA/hard/online,2,048 steps each, common interval1:
nine runs. EMA/hard actually execute three batch updates after initialization
(sync count4). Final hard target/online tensors exactly match; EMA retains
maximum parameter difference about0.0018525. This verifies executed contrast,
unlike the previous interval4/three-batch condition.

All actors remain unready with counts[0,3,0], and-458 evaluation still uses
uniform fallback. Neither target-mode change nor actual synchronization fixes
admission at this budget. Three online conditions precisely reproduce the
previous interval4 online weights and evaluation, as the unused interval should.

Source `f61d98c7833c1803d491c200324494b73840ae208c879dcf36be24d488b70d3a`;
123 files; ZIP SHA256
`1e6f97a1bb6c6b30e53dd121d95f116fe583d763b9bea6d52ae9f69753264e0d`.
Audit validates18 final states,72 MC events/all576 deletions,18 pre-gradient
snapshots/36 GAE events/all288 deletions. No actor event is fabricated when
executed policy was unready. Target alignment and actual synchronization are
checked directly, not inferred from filenames/counters alone.163 tests pass.

The new controlled GAE counterexample uses500 rewards-1, zero value/next-value,
nonterminal time-limit boundary, gamma0.99/lambda0.95. Every advantage is
negative, so positive-raw-advantage admission cannot start. A calibrated
constant continuing-task value-100 produces zero advantages, still no evidence
to recommend arbitrary actions. This is a startup counterexample, not a claim
that all negative-reward trajectories lack helpful actions. Maintain the
AGENTS admission rule and MC holdout exclusion while testing explicitly
declared value warm starts and maintenance optimizer controls.

Artifacts `results/t2_pi20260920/continuation_20261004/target_sync1/` and task
outputs `neural-cbr-t2-sync1-results.md`, MC/training verification JSONs.
