# T1 implementation status: neural-cbr

Inspected 2026-10-02. Implementation changes start from `6adeb30`.
The authority is career-2027 at `cd77277600841438d98ac5793867da28b8ec6dd0`;
the 18 imported files and original paths are recorded in
`t1_spec/SOURCE_SNAPSHOT.json`. Imported source wording is preserved; original
relative links refer to the career-2027 layout. Use the snapshot map to resolve them.

The September 20 candidate note supersedes Q/B combination and stored-label
agreement as the primary provenance definition. The earlier geometric formula,
pre-adaptation-only audit and separate weighted coverage/redundancy formulas are
historical designs, not current active policies. Old T0/T1 artifacts remain in Git
history. Existing September 29 results under `results/t1/` are preserved but do
not validate the revised pipeline.

## Current implemented behavior

| Module | Behavior |
|---|---|
| `model/nnknn_model.py` | Optional stable query-case IDs exclude only the query's own case, even with MCB, duplicates and compaction. The default maintained interface remains unchanged. |
| `model/t1/outcomes.py` | One final prediction/loss path for provenance and candidates. Nominal residual uses final-label CE; logit residual uses NLL; regression uses squared error. Adaptation penalties are excluded from removal scores. |
| `model/t1/provenance.py` | Every participating case receives normalized activation times final query success/failure. Q and standalone normalized B are logged separately. Regression success requires a declared absolute-error tolerance. Pre-adaptation loss remains diagnostic. |
| `model/t1/candidates.py` | Coverage/reachability uses activation above an explicit threshold plus final query success. Self is excluded by ID. Zero reachability has an explicit policy. Frozen-parameter full-query and cached-query removal use the full reference-set denominator. |
| `model/t1/retention.py` | Standalone Q, B, ratio and removal candidates, with random/stratified/current bias/full-memory/kcenter comparators. Q/B mixtures fail explicitly. Impossible capacity/protection combinations fail before compaction. |
| `model/t1/maintenance.py` | Removal picks the lowest eligible influence, refreshes after each actual deletion and checks cumulative final-loss increase against the original memory. Archives and per-case optimizer rows move with stable IDs. |
| `model/t1/reuse.py` | Classification aggregate label-conditioned adaptation with independent nominal/logit and combined/single-loss switches. Training examples use stable-ID LOO. |
| `model/t1/sync.py` | Independent and alternating classification schedules. Maintenance receives the actual current adapter, so it scores final outcomes. |
| `model/t1/core.py` | Core training, validation-only checkpoint selection, optional MCB and complete nonzero retrieval contributions with pre/final decisions. |
| `model/t1/artifacts.py`, `tools/t1_run.py` | Specification commit, pipeline version, dirty flag, actual source hash/bundle, data snapshots and model/adapter/archive checkpoints. Retention, reuse, frozen reuse+retention, sync, MCB, revision and legacy-reference runners. |

The final maintained core baseline can still be run with `legacy_reference`.
The revised T1 core deliberately changes LOO identity handling to meet the
specification. It must not be described as an exact unchanged legacy experiment.

## Evidence and boundaries

The local test suite covers final-outcome credit, opposing case labels,
single-case credit, ID LOO with duplicates/MCB, final adapted classification and
regression removal loss, cache approximation and full denominator, mode/RNG/state
preservation, Q/B independence, cohort protection and cumulative removal budgets.
The older closed-form retrieval-only removal helper remains a diagnostic test
reference; active candidates rerun final prediction.

All current numerical choices are exploratory. `configs/t1/_base.yaml` declares
smoothing, activation/cache thresholds, zero-reachability policy, cohort floors,
allowed cumulative loss and the standardized-target regression tolerance.
These are not PI-approved confirmatory constants. Q usefulness and Q/B correlation
remain hypotheses; no general score or claim of recovery of all surveyed methods
has been established.

## Remaining plan coverage

| Requirement | Status |
|---|---|
| Current maintenance definitions and four candidate families | Implemented and checked on bounded supervised pilots, including trained classification adaptation |
| Stable-ID statistics, archive/restore and optimizer movement | Implemented; regression-tested |
| Classification reuse and synchronization | Implemented; bounded smoke verification |
| MCB mechanism | Reimplementation tested on/off; the IU-Bloomington source is not obtained |
| Exact historical classification/regression paper reproduction | Not established; legacy runner checks current behavior on reconstructed splits |
| Optional removal with retraining and matched further-training controls | Not implemented; efficient frozen-parameter branch is current |
| General score recovering prior methods | Open research task; distinguish score equality, ranking equivalence and full selector equivalence |
| Broader synthetic reliability, rare/boundary/shift studies and confirmatory seeds | Pilot infrastructure exists; full revised sweeps are not complete |
| Reviewer force-include and feature-weight interventions | Not implemented |
| Human UI study | Not implemented |
| Regression synchronization | Not implemented |
| Image/text legacy, TabArena and contemporary comparators | Not wired to T1 runner |
| RL actor/critic T1 integration | Not implemented; role outcome and reference decisions remain open |
| T3 prompt/internal/combined host integration | Not implemented; roadmap only |

## Validation commands

```powershell
.venv\Scripts\python.exe -m pytest tests -q --basetemp <new-workspace-temp-path>
.venv\Scripts\python.exe tools\t1_run.py configs\t1\alignment_candidates_small.yaml --out results\t1_pi20260920\verification_20261002
.venv\Scripts\python.exe tools\t1_run.py configs\t1\alignment_adapted_candidates.yaml --out results\t1_pi20260920\verification_20261002
.venv\Scripts\python.exe tools\t1_run.py configs\t1\alignment_digits.yaml --out results\t1_pi20260920\verification_20261002
```

See `T1_ALIGNMENT_STATUS.md` for the reviewed scope and outstanding gaps.
Branches remain `neural-cbr` and `rl`; no new branch was created.
