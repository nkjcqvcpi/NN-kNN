# Grouped nominal inputs and stable-case bindings — 2026-10-04

The classification extension in the imported T1 plan requires an explicit
per-field declaration of whether nominal information is already represented
in the retrieval features. The old adapter silently substituted zeros when its
nominal difference was missing, and the training interface did not implement
this declaration. This package fits a separate vocabulary for each uncovered
field on training data only, reserves distinct missing/unknown entries, excludes
covered fields, and binds original one-hot case values to stable case IDs.

The adapter receives `[Delta_z, Delta_u, p0]`. All three derive from the same
actual retrieval event; `Delta_u = u_query - weights @ u_actual_cases`. Training,
prediction, outcome credit, removal scoring, coverage, reviewer overrides and
synchronization now accept explicit query nominal values. Compaction changes
live slots but never changes the original stable-ID binding. Undeclared inputs,
missing required inputs, malformed groups or missing bindings fail explicitly.
The old private ambient nominal context and raw slot-indexed helper are not
accepted as substitute inputs. Checkpoints preserve frozen binding tensors and
the fitted schema JSON. Events preserve p0, nominal/representation differences,
the target residual, actual adapter residual and final scores/probabilities.

## Fixed engineering grid

Three configs run seeds 8/9/10 on `synthetic_nominal/v1`: 18 reuse runs covering
both output modes and all three losses; 18 frozen-adapter retention runs covering
full memory, Q, B, coverage/reachability and full/cached removal; nine component
synchronization runs covering independent, RR and RRR. The task has 180 training,
60 validation and 90 test queries. Color and shape are outside the retrieval
representation; size is explicitly represented in column zero. The two uncovered
groups have five and four entries, respectively. Each test split contains 63
known-color, 18 unseen-color and nine missing-color queries. Split fractions and
max_train do not apply to this declared fixed synthetic engineering task.

Frozen source: `5af96c95141c1aaaee1c0e5525758e3e9d1752f52e904d7de70ce8c85c61639b`,
based on `6def29b` plus this uncommitted package. Runner source ZIPs and all raw
artifacts are under `results/t1_pi20260920/continuation_20261004/alignment_nominal_*`.
The independent preflight ZIP has SHA256
`3e18c6cb41c1bd75a79f3416bdd83c71ff9cb7d3002505a856449257397893ec`.

111 T1 tests pass, including no information leakage into adapter inputs,
train-only vocabulary, typed categories, missing/unknown separation,
serialization, outcome credit, full/cached influence and actual RRR optimizer
compaction. All five maintained smoke modes pass. Root's independent audit loads
archived source, manually encodes the raw categories, regenerates each of the 45
predictors and all 4,050 actual test events, and checks residuals, activations,
stable IDs, predictions and subgroup metrics. It passes; scripts and evidence
are in the task work folder as `independent_nominal_audit.*`.
The separate archived-source audit also passes 27 direct C/H/A checks and
full/cached selector comparisons at all 45 removal steps for each of three
seeds. Its full verification is in the task outputs as
`neural-cbr-nominal-contract-20261004-verification.json`.

## Observed behavior and limits

The reuse-only retrieval baseline averages 0.3481 accuracy. Nominal residual
scores with combined loss average 0.7222 overall and 0.9365 on known colors, but
only 0.1667 on unseen colors. Logit residual combined loss averages 0.6407 overall
and 0.2963 on unseen colors. These are descriptive results from three seeds, not
a statistically established preference between outputs or losses. Unknown
categories are encoded correctly but that does not teach their task meaning;
missing values hide part of the latent teacher input. The input contract passes
while those generalization limitations remain observable.

All five compressed retention conditions reach 135 cases versus 180 for full
memory. Full memory averages 0.7222 post-adaptation accuracy; Q, coverage and
full/cached removal average 0.7148, B averages 0.7074. Full/cached removal produce
the same reported post metrics in this grid; this is not a general equivalence
claim. Independent/RR/RRR synchronization averages 0.6259/0.6296/0.6333, with
RRR using 135 rather than 180 cases. It therefore does not establish a benefit
under matched capacity. Reuse and synchronization have different training
budgets and must not be ranked as if budgets were equal.

This is a small synthetic contract exercise, not a real-data nominal benchmark,
completed human study or proof of broad predictive competitiveness. Test results
are diagnostic only and did not select hyperparameters or intervention budgets.
Further training after adapted removal, matched capacity/compute controls,
broader reliability, nominal real-data protocols and reviewer usability remain
separate work packages.
