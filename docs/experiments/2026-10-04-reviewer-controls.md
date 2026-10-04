# Reviewer controls and matched continuation — 2026-10-04

## Implementation and tests

Transient forced retrieval supports unsampled normalized-softmax pre-top-k
cores with bias-minus-distance or negative-distance scoring. Stable requested
IDs must be active and eligible for every controlled query. Quarantine and
self-ID gates cannot be overridden. An explicit positive activation floor with
`count*floor <= 1` is logged; true learned distances/scores remain unchanged.
The controlled pre/final prediction and displayed evidence use one retrieval
result. Retrieval-only regression and external aggregate adapters are supported;
the internal legacy regression-adapter path fails explicitly in this API.

Reviewer logs are versioned and serializable. Finite, shape-compatible feature
parameters and case mixture weights can be edited; feature logs name the raw
LeakyReLU parameterization. Parameter rollback checks for stale intervening
changes. Protection flags flow through stable-ID statistics to every retention
selector, including sequential removal/retraining. Archive removal realigns
optimizer rows; restored case rows have zero moments and quarantine persists
until release. Batch restore checks IDs and capacity before mutation.

Conditional no-edit continuation now starts from independent M0 model/Adam
copies with the same minibatches, learning rates, epochs and forked RNG as M2.
Dropout tests show no-edit M2 and control are identical and leave the external
RNG unchanged. Zero-epoch edits omit the extra-training control. Target and
collateral flip counts now include post-training comparisons. All 75 tests and
the five maintained imports/training/DQN/NEC/NN-kNN-RL smoke modes pass. Smoke
budgets establish execution compatibility, not RL performance.

## Matched experiment

`configs/t1/alignment_revision_matched.yaml`, seeds 5/6/7, synthetic classification,
Q/bias/random/oracle flagging and budgets 10/30: 24 runs, 96 stage checkpoints.
M0 is validation-selected; M1 is edit-only; M2 and MC each receive five further
epochs. M0/MC targets remain original; M1/M2 targets include reviewed corrections.
All stage predictions replay; final checkpoint equals M2; MC is identical across
conditions within seed; exact Adam step increments match the budget. ZIP and
manifest hash agree:
`028c3c2fb84274ee8baf9aa919ff348d954545ba3a2400d8f2d4be79604aa9cf`.

Matched continuation distinguishes the effect of correction from further
training. Many reviewed cases do not participate in these held-out queries,
so M2 and MC can agree even after genuine corruption is corrected. Seed 5 Q
conditions have one beneficial and one harmful MC-to-M2 flip; seed 7 Q conditions
have one beneficial and zero harmful flips. Seed 7 random budget 30 has two
beneficial flips. These are small, mixed effects, not stable Q superiority.
Detailed per-seed tables and replay evidence are in the chat outputs.

## Trained intervention probes

Six trained models (synthetic classification and energy regression, seeds
5/6/7) exercise force, feature/bias edits and rollback, protection through
maintenance, and quarantine/archive/serialized-log restore/release. Initial
probe-driver shape handling and scenario isolation were incorrect; its failed
artifacts remain in `reviewer_probes`. The corrected driver runs in a distinct
`reviewer_probes_v2` directory with independent baseline/optimizer copies.
All 30 scenarios now pass their execution checks; six real data/model snapshots
and the actual edited and rolled-back stages are saved. Restored per-case state
is exact by stable ID; surviving Adam moments stay with their cases, and restored
case moments are zero. Classification rollback predictions agree exactly.
Regression archive/restore changes storage order: the independently replayed
maximum prediction difference is `2.384185791015625e-7`, and the maximum
stable-ID-aligned activation difference is `5.960464477539063e-8`. Report these
as floating-point roundoff under an explicit `1e-6` absolute tolerance, not
bitwise predictor equality. Offline replay independently loads all 120 saved
stage states and compares 60 newly generated forced-decision events with the
saved IDs, activations, distances, biases and pre/final vectors. A second parent
audit uses a fresh log to exclude accidental comparison with original events.
Both pass. Earlier harness errors and the numerical roundoff diagnosis remain
separate from this final verification in the output JSON and work logs.
These are simulated engineering probes, not a participant study. The draft
common-knowledge UI protocol is preparation; functional UI and participant
evidence remain separate work.
