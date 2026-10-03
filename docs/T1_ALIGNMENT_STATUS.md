# PI document alignment review — 2026-10-02

## Scope and authority

Code reviewed: the local `neural-cbr` worktree based on `6adeb30`.
Authority: career-2027 commit `cd77277600841438d98ac5793867da28b8ec6dd0`.
The complete local source snapshot lists 18 files in `t1_spec/SOURCE_SNAPSHOT.json`.
T2/T3 specifications were imported for dependency boundaries, not treated as already implemented.

## Confirmed mismatches corrected

| September 20 requirement | Corrected implementation | Evidence |
|---|---|---|
| Compare Q and B separately | No active geometric T or Q/B mixtures; old policies/configs rejected or migrated | Q independence and retired-policy tests |
| Share final query outcome with participating cases | C/H use normalized activation and success after enabled adaptation | Opposing-label, one-case, success/failure and regression tests |
| Solve uses activation threshold and final outcome | Coverage/reachability evaluates reference queries plus every case problem with identity LOO | Explicit threshold/zero-policy and ratio tests |
| Final prediction loss for removal | Recompute retrieval, normalization and classification/regression reuse with frozen parameters | Explicit masked final-loss comparisons |
| Cached approximation uses full denominator | Query ID index, retained reference tensors, thresholded reruns, full validation and error accounting | Cache denominator and reduced-rerun tests |
| Refresh after real case deletion | Sequential scoring and new current-case/query index on each iteration | Trace active IDs shrink after removal |
| Check cumulative loss against original memory | Explicit allowed loss increase; stop and report if target capacity cannot be reached | Low/high-budget tests |
| Stable IDs support LOO and compaction | Query case IDs independent of slot, representation or duplicate input | Duplicate, MCB and compaction tests |
| Preserve auditable treatment | Pinned specification, code hash/source bundle, data/model/adapter/archive snapshots, final prediction in same retrieval event | Artifact integrity audit |

## Scientific choices that remain open

Activation thresholds and zero-reachability handling; regression success
tolerance; Q smoothing/variants; B transformation/cohorts; cumulative loss budget;
cache error tolerance; optional retraining budgets; confirmatory seeds/margins;
general-score derivations and prior-selector reproduction. Pilot choices are
declared in YAML, not promoted to approved formulas.

Full cached-query validation currently reruns all reference queries as an
additional verification cost. The scoring API can omit this cost for a separately
validated fast path, but the local mechanism pilots deliberately keep it enabled.
No measured net speedup is claimed.

Removal may stop above requested K to honor the declared loss budget. Such runs
must be reported with actual memory and stop reason, and must not be presented as
matched fixed-K results. Fine-tuning follows selection only in retrieval-only
pilots; the adapted maintenance comparison uses frozen core/adapter parameters.

## Evidence boundary

The current implementation now follows the confirmed September 20 maintenance
semantics. This is not a claim of complete T1-T3 implementation or scientific
competitiveness. The bounded verification suite and two-seed pilots establish
plumbing, traceability and the tested behaviors. They do not rank the candidate
families reliably or replace fresh-seed confirmatory studies.

Remaining engineering/research gaps are listed in `T1_PLAN_IMPLEMENTATION.md`:
optional retraining comparator, intervention extensions/UI, regression sync,
modern/legacy benchmark reconstruction, external MCB source, general-score
derivations, RL Stage A integration and T3 host integration.

Old `results/t1/` artifacts are not overwritten. New verified runs are under
`results/t1_pi20260920/verification_20261002/`. All changes are local pending
review; no commit or push was made.

## Completed local verification

- 18 imported specification documents match the pinned source after newline normalization.
- 42 tests passed; one test-only tensor-to-scalar warning remains.
- 70 final run manifests across nine experiment groups have the expected artifacts.
- 70 checkpoints load; predictions from 61 maintained T1 checkpoints replay exactly
  within 1e-6, including saved classification adapters. Eight simulated-review
  baseline checkpoints and one legacy checkpoint were integrity-checked without
  replaying their separate intervention/legacy training treatments.
- 11 source bundles pass ZIP integrity and reproduce the recorded source hashes.
  Two source hashes are present because the legacy/sync empty-statistics artifact
  fix was applied after the other runs; affected runners were repeated.
- All tested removal pilots reached their requested capacities. Maximum observed
  cached/full score discrepancy was 0.00075658982; this is observed pilot error,
  not a selected acceptable-error threshold.
- Digits random/Q/B at K=0.5 on seeds 0/1: accuracy 0.983333–0.988889.
  This bounded repeat did not reproduce the historical near-chance collapse.
- `git diff --check` passed for the changed implementation/documentation scope.

The complete manifest and readable report are delivered with this chat. No full
revised research sweep or confirmatory claim was made.
