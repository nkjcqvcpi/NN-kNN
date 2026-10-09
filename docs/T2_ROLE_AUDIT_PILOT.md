# Four-role current-core pilot and direct case audit — 2026-10-04

`model/t2/audit.py` records the exact opt-in retrieval returned by core forward,
with actual IDs, distances, feature-distance terms, biases, weights, inputs,
labels and predictions. Each intervention removes one case from the actual
positive-weight set, renormalizes without refilling and evaluates the actual
masked core. Empty critic memory yields zero; empty actor evidence yields
uniform policy. Sampling/adapters without declared semantics are rejected.

Critic uses squared loss against explicitly separate `training_gae` or
`independent_mc` targets. Actor exposes a declared signed-advantage policy-loss
surrogate; readiness-driven uniform warmup has no case credit. This is not
episodic-return improvement. C/H are unweighted positive/negative removal-loss
deltas and Q is a smoothed candidate, aggregated separately by role and stream.
No Q-driven optimizer, retention or global promotion is silently activated.

An optional bounded sink in the existing independent critic holdout evaluator
records MC queries while its RNG/mode restoration and training exclusions stay
in place. The original API remains unchanged by default. `tools/t2_role_pilot.py`
uses maintained training helpers and records config, versions, fingerprint,
actual steps, runtime, checkpoint hash, safe audit state and interventions.

CartPole: four roles, seeds 8/9/10, fixed 512 interactions, 128-case capacities,
top-k8, three greedy final evaluation episodes and two independent stochastic
MC episodes. Actual final checkpoints selected; no early stop. Frozen source
`9ab43109e73d52904fa69b58ed406ef5024961b7bdfda9fb4efe22bde739d084`;
120 files; ZIP `9046402b47ab86d5d0b44c73d1ac0f6fc515b40e60e69e3cbbeaeab8ab4178b6`.
Twelve runs/24 states, 48 MC events and all384 actual masked interventions replay.
Independent renormalized predictions, retrieval geometry, final losses and
credit match, with a separate float32 cancellation bound for manual formulas.

Mean final returns (three seeds): MLP/MLP145.000, NN/MLP141.889,
MLP/NN33.222, NN/NN202.556. Seed ranges are broad. Mean independent MC MSE:
145.747,524.631,34.865,819.735 respectively; NN/NN seed10 reaches2262.612.
Actor-dependent visitation differs; MC errors are not paired same-data ranks.
Equal interactions do not imply equal compute/parameters/exploration policies.

First12-run source `eeaa35644179` remains preserved. Safe restoration revealed
that critic active count was absent from state_dict and an empty restored critic
remained empty. The post-load hook now reconstructs the compact active prefix
from canonical IDs; legacy restore and actual predictions are tested. All12
runs were repeated from the corrected freeze.159 T1/T2/T3 tests and maintained
RL smoke pass; old manual cancellation audit failure logs remain.

Artifacts: `results/t2_pi20260920/continuation_20261004/role_audit_512_restore/`.
Task reports `neural-cbr-t2-role-audit-results.md` and
`neural-cbr-t2-role-audit-verification.json`. Only the first8 MC queries/run
are audited, and actual actor rollout interventions remain unfinished. One
cheap task/512 steps/current-core maintenance cannot pass the full-cycle
Stage A gate. Add a second task, GAE and actor behavior audit, longer stability
checks, measured cost and explicitly frozen confirmation margins.
