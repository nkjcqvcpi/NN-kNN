# Selected adapter optimizer states — continuation prerequisite

Previously classification reuse returned the validation-selected adapter weights
but discarded Adam moments. Synchronized training internally restored both
selected optimizers but did not export the adapter optimizer in its checkpoint.
Starting a fresh optimizer or exporting last-epoch moments would change the
continuation being compared.

Reuse now snapshots Adam together with each best adapter and exports selected
epoch, executed epoch count and moments. Synchronization exports both restored
selected optimizer states. Runner checkpoints retain adapter moments and separate
retrieval/adapter checkpoint epochs. In independent reuse these epochs can
differ; synchronized states belong to the same selected epoch. Frozen compacted
reuse-retention outputs explicitly leave the retrieval optimizer absent rather
than attach an unaligned full-memory optimizer.

Two focused tests force an earlier best epoch, compare every actual selected
Adam moment to captured updates, and perform a real resumed update after safe
checkpoint serialization. Both residual modes remain covered by the maintained
T1 suite; all 119 tests passed at this package boundary.

The frozen implementation fingerprint is
`19f5dddd28686dc8ee0da6eab5115fcd10961fd02b3b2b13a202d949d8da3f4a`,
100 ZIP members, SHA256
`48fe226b18f318bedb75a5e0001199852e5833ddf9d7d14f1789b7d5c1ed5c54`.
Eighteen nominal reuse and nine synchronization runs completed from archived
source. The parent independently reconstructed all 27 predictors and 2,430 raw
events: train-only vocabularies, raw nominal encoding, stable case bindings,
actual adapter inputs/residuals/predictions and subgroup metrics matched.
All 27 selected checkpoints passed the independent per-parameter phase-budget
and selected-epoch audit and paired real continuation updates after safe
serialization; the 11 verification checks pass with no errors. Earlier audit
harness failures are retained. This checkpoint change establishes replay and
continuation provenance; it is not evidence of improved predictive performance.
