# Reviewer intervention engineering contract — 2026-10-04

Authority: `T1_FULL_CYCLE_IMPLEMENTATION_PLAN.md` sections 3.3 and 11,
`T1_REVISE_RETAIN.md` causal evaluation, and the shared experiment contract.
The user authorized implementing later plan stages. T1.1 defaults remain
revision-disabled; technical intervention pilots are explicitly T1.2.

## Controlled decisions

An authorized reviewer selects stable, currently eligible case IDs for one
declared query batch. The override cannot bypass quarantine, self-ID exclusion,
active-memory eligibility or the configured top-k budget. Archived cases must
be restored through a separate recorded action first. Unsupported scoring or
sampling modes fail explicitly rather than silently changing their meaning.

For the initial normalized-softmax/pre-top-k path, the requested cases occupy
top-k slots; remaining slots follow the unchanged learned score. True learned
scores, distances, biases and feature weights remain inspectable. A positive,
explicit activation floor prevents an extreme score from underflowing to zero.
For `m` forced cases and per-case floor `f`, require `m*f <= 1` and use
`a_override = (1-m*f)*a_selected + f*forced_indicator`. This is a declared
intervention treatment, not a learned weight or a PI-selected scientific
default. Report the floor and actual resulting activations in the same event
used to compute the controlled prediction.

The intervention is transient. It must not modify persistent retrieval
parameters or silently apply to future queries. The event links the run,
reviewer action, query scope, override, selected IDs, true distances, normalized
activation and pre/final prediction. Host evidence and human inspection must
derive from that event instead of repeating retrieval to make an explanation.

## Persistent edits and rollback

Case label, bias, case mixture weight and learned feature-parameter edits
require finite, shape-compatible inputs validated before mutation. Logs record
actor, reason, before/after values, version and parent/undo linkage. Feature
parameters and their projected distance weights are distinct: display both and
name the edited parameterization. A rollback restores the recorded state and
creates a new log entry; it does not erase intervention history.

Human protection is an explicit stable-ID maintenance requirement, independent
of automatic cohort floors. Capacity conflicts fail rather than dropping
protected IDs. Archive removal stores exact case state and aligns optimizer
rows; restoration preserves IDs and case state while declaring that restored
optimizer rows start at zero moments. Whole-model rollback uses the complete
checkpoint when the original learning trajectory is required. Batch restore
must validate all requested IDs and capacity before changing any state.

## Causal artifacts and comparisons

M0 is the existing trained predictor, M1 is edit-only, and M2 receives a fixed
reported optimization budget. Each stage must save its real evaluated model,
case eligibility mask, parameter training flags, optimizer state and targets.
Corrected training targets must be separately identifiable from the original
corrupted data. Final statistics and retrieval events use M2 and its corrected
training outcomes. Stage artifacts must reproduce their recorded metrics.

When attributing M2 improvement specifically to an edit, continue unchanged M0
under the same optimization budget as a conditional matched-training control.
Report immediate target/collateral effects and post-adaptation effects
separately. Simulated oracle review is a mechanism diagnostic: it does not
measure human burden, objective reviewer skill or usability. A common-knowledge
UI/protocol can be prepared locally; real participant evidence remains a
distinct required study.
