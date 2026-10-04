# Local reviewer workbench — engineering preparation

`tools/t1_reviewer_ui.py` serves a real NN-kNN model on loopback. It implements
the common-knowledge UI prerequisite of T1.2: red circles qualify, blue circles
and squares do not, and size is irrelevant. Eight original cases include two
declared injected label errors; eight displayed queries differ in size from both
training and validation examples. This task supports objective inspection with
no specialized knowledge. It is an engineering demonstration with no recruited
participants, not a completed study or a generalization benchmark.

Start with an empty output directory:

```powershell
.venv\Scripts\python.exe tools/t1_reviewer_ui.py --out <empty-session-directory>
```

The console and `server.json` provide the dynamically allocated local URL.
The output directory holds an immutable source ZIP and manifest, M0 and each
subsequent version's actual model/Adam/targets/statistics/interventions, the
rendered data JSON, and the current version pointer. Existing sessions are
preserved by rejecting nonempty destinations. Loading/replaying these artifacts
uses safe tensor checkpoints; interactive restart/resume is not yet implemented.
The server binds only `127.0.0.1`. Version checks prevent stale edits, and a local
session token plus Host/Origin validation prevents unrelated pages from issuing
review actions. This is not a multi-user or publicly hosted application.

## Actions and causal inspection

The user selects a target query and stable case ID, records actor, reason,
expected effect and confidence before executing an action. The workbench exposes:

- label correction and exact parameter-edit undo;
- case bias, case mixture and global feature-parameter edits;
- protection/unprotection, quarantine/release and archive removal/restoration;
- a transient, eligibility-constrained forced reference with an explicit floor;
- one fixed 1–20 epoch M2 continuation with an unchanged-M0 matched-training MC.

All predictions, displayed case activations/distances and C/H/A derive from one
frozen retrieval event per query. Q, B, R, A, C and H are separate columns; label
agreement does not determine support. Original/raw and projected parameters and
per-case feature-weight factors are present in the full audit record. Q uses
explicit smoothing 1, B uses within-label percentile, and exposure thresholds
are one retrieval and .01 activation mass. These are declared demo parameters,
not PI-selected confirmatory reliability thresholds. Statistics summarize eight
current demonstration outcomes; they do not pool repeated refreshes or pretend
to estimate broad reliability.

Each action records its predeclared target event, overall flips, target and
collateral flips, target correctness, elapsed session time and intervention/undo
links. A force event retains its own actual activations and transient prediction;
the normal model and subsequent queries remain unchanged. Archive restoration
retains original stable IDs and state, preserves quarantine until explicit
release, and declares restored Adam rows to have zero moments. Protection must
be explicitly removed before archive removal.

M1 edits execute no gradient updates. M2 and MC use copied selected Adam states,
the same examples, seed-local minibatches, fixed final epoch and learning-rate
scale .1. M2 trains on explicitly corrected query targets; MC uses the original
targets and unchanged M0. Quarantined cases are compacted before M2 and moments
move with retained cases. Different capacities can change retrieval cost despite
equal examples/epochs, so the record reports both capacities and does not claim
compute equivalence. The session becomes read-only after M2/MC to preserve this
bounded causal comparison.

## Validation and remaining work

117 T1 tests pass, including six workbench tests: same-event displayed credit,
gradient-free edits and exact undo, atomic rejection of invalid/stale edits,
protection/quarantine/archive/restore/force behavior, real M2/MC checkpoint replay,
and the loopback request boundary. Browser actions in the final implementation
corrected case 4 and then case 5. The first correction caused zero flips; the
second corrected both blue-circle queries with zero harmful or collateral flips.
This is a mechanism demonstration, not participant burden or skill evidence.
The final screenshot and versioned artifacts are in the task outputs under
`reviewer-ui-20261004-final`.

A recruited study still needs frozen comparison conditions and randomized or
counterbalanced assignment, participant procedures and consent, task versions,
burden/confidence collection beyond session time, sample-size/uncertainty choices,
and genuine participant actions. Expert content insertion, constrained learned
adaptation, arbitrary existing-run UI import and interactive resume remain
separate implementation packages. Do not equate this functioning workbench with
completion of those requirements.

The independent scripted pilot used seeds 11/12/13 and three action chains:
correct both labels; edit/undo and archive/restore/force; quarantine one faulty
case and correct the other. All 75 successful version checkpoints and nine MC
controls replayed. A separate replay included preserved failed harness attempts
and browser versions: 99 checkpoints, 792 query events, matching predictions,
stable IDs, activation/distance vectors and direct R/A/C/H. Corrective chains
achieved 8/8 versus 6/8 for M0 and MC; the undo chain remained 6/8. These repeated
eight-query demonstration outcomes do not establish generalization or human
performance. Chain C has seven M2 cases versus eight MC cases.

The frozen implementation fingerprint is
`f38910c24a8b20df4a77465b533ed5329b9b64a56ddf5d901d2ecbd7aa46fe6a`;
the canonical ZIP SHA256 is
`e456fddfbc492dba5a969184bdff2ada09f7815d0069a62484c54aad59074bcf`.
Reports preserve earlier harness errors (case-count and positional-ID
assumptions, source ordering and step-count assumptions). The M0 configuration
declares 30 epochs but its checkpoint lacks the executed training history;
selected Adam step 1 is not evidence of total executed epochs. M2 and MC each
save three actual history epochs and advance corresponding parameter steps by
three. Evidence is in the task's `neural-cbr-reviewer-ui-20261004-verification.json`
and `independent_reviewer_ui_audit.json` outputs.
