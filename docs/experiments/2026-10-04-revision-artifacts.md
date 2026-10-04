# Actual causal revision artifacts — 2026-10-04

## Corrected defect and validation

The revision runner previously evaluated edited/retrained models but saved the
untouched baseline in `checkpoint.pt`. Those historical revision checkpoints
cannot independently substantiate M1/M2 predictions. The runner now saves the
actual final M2 model and separate `M0.pt`, `M1.pt`, `M2.pt` files containing the
evaluated state, active count, eligibility mask, optimizer, parameter training
flags and stage training targets. Final provenance is recalculated using M2
and corrected training targets; the original corrupted data remains preserved.
Artifact capture is optional in the evaluation API; public metrics exclude
live models/tensors. Classification label bounds are validated before mutation.

59 tests pass, including serialized stage replay, original model/data isolation,
quarantine followed by optimizer-aligned training, zero-epoch edit-only behavior,
and atomic invalid relabel rejection. The existing test-only tensor conversion
warning remains. The system default pytest temporary directory denied access;
verification succeeded with a fresh temporary directory in this task workspace.

## Experiment boundary

`configs/t1/alignment_revision.yaml`: synthetic diagnostic classification,
seeds 5/6/7, Q/bias/random/oracle flagging, review budgets 10/30, core training
up to 30 epochs (validation-selected state), followed by five fixed retraining
epochs with learning-rate scale 0.1. This is simulated oracle review, not a
participant study. Raw results are at
`results/t1_pi20260920/continuation_20261004/alignment_revision`.

24 runs and all 72 stage checkpoints reproduce recorded metrics. Each final
checkpoint equals its M2 state. Targets match the recorded label corrections;
M0 is identical across conditions within seed. Exact Adam step increments match
the five-epoch minibatch budget. All metrics/history/states are finite. Source
ZIP integrity and the reconstructed fingerprint pass:
`c7a6c228a5a8b27cce1d4ac371ba890ab1107993424d7b72357877288e735355`.
All manifests report unchanged source during execution, based on `5055eee`
plus the saved local source bundle. Chat-facing result and verification files
are in the task's `outputs` directory.

## Interpretation and next diagnostic

The three-seed mean flagging precision is 0.30/0.2667 for Q at budgets 10/30,
0.0667/0.0556 for bias, and 0.10/0.1222 for random. Oracle precision is 1 by
construction, with recall 10/54 or 30/54. These descriptive pilot values do not
establish reliable harmful-case detection. Baseline held-out accuracy is already
near ceiling; correcting true corruption need not change a held-out prediction
if the corrected cases have little activation. Some immediate improvements
reverse during further training.

There is no matched no-correction further-training comparator in this batch.
M1-M0 isolates direct editing; M2-M0 measures editing plus additional training.
The next causal experiment must add a matched continuation before attributing
post-training changes specifically to correction, and must report target versus
collateral effects and whether corrected cases actually participate. Broad
corruption/overlap/shift grids remain pending. No superiority or human-usability
claim follows from this bounded mechanism audit.
