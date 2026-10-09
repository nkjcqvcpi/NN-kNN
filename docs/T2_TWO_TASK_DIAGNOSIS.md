# Two-task role and negative-reward admission diagnosis — 2026-10-04

Acrobot-v1 is now a maintained flat-observation/discrete-action task, verified
against the installed Gymnasium environment (six observations, three actions,
500-step cap). Its physical return ceiling0 is distinct from its time limit;
CartPole remains500. Shared early stopping uses this metadata unless explicitly
overridden. Acrobot defaults to the bounded smoke profile; competitiveness
margins remain unapproved. Do not reuse CartPole's positive success threshold
for a negative-reward task.

The role pilot accepts explicit tasks. CartPole/Acrobot, four roles, seeds8/9/10,
2,048 interactions each,128-case capacity,top-k8,three evaluation/two independent
MC episodes produced24 runs. No early stopping or performance threshold.
Frozen source `39906b4043b44b338e52b2622c40ed6689c11313c8874f16f76d5d998a1f1650`;
121 files; ZIP SHA256
`b88c3d010ea81d62aaad549aeebcdf5d376b879726d62fa38d5a5054024fed69`.
Independent audit safely loads48 states, restores actor action coverage and
readiness, verifies critic identities and all96 MC retrieval events/768 actual
fixed-set removals, and reconstructs predictions/losses/C/H/Q.

CartPole mean final returns: MLP/MLP173.444,NN/MLP120.444,MLP/NN80.333,
NN/NN145.111; three seeds and substantial variation. NN/NN declines from the
earlier512-step pilot mean202.556, so more updates have not reliably repaired
the engineering limitation. This comparison is exploratory, not a confirmation
with frozen margins or matched compute.

All six Acrobot NN actors remain unready. Five retain zero cases; NN/NN seed9
retains only3 with action coverage[2,1,0]. Their evaluation mean-458 comes from
uniform sampling fallback, not a learned case policy. MLP actors get-500.
Actual loss logs verify2,048 training samples and raw-positive-advantage
admissions; the empty-store diagnosis is not inferred solely from rewards.
Acrobot NN critic MC error remains thousands, against a differently visited
behavior distribution; cross-role MSE is not a fixed-data ranking.

The AGENTS raw-positive admission requirement remains intact. Do not silently
substitute normalized advantage, bootstrap random recommendation cases or call
uniform fallback a full-cycle actor. Diagnose initial value calibration and
GAE sign/coverage; test explicitly recorded baseline warm starts or separately
declared scientific alternatives. Add real actor behavior/GAE contribution
audits and optimizer-state alignment before the Stage A gate.

Artifacts `results/t2_pi20260920/continuation_20261004/two_task_2048/`;
task reports `neural-cbr-t2-two-task-results.md` and
`neural-cbr-t2-two-task-verification.json`.160 combined tests pass; test verifies
real Acrobot spaces/rewards/time limit and separate return-ceiling overrides.
