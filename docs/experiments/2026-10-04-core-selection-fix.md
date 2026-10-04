# Core no-op checkpoint selection repair

The same no-op selection defect found in synchronization also existed in
`train_retrieval`: any checkpoint hook reset the best model/Adam state and
patience. Unused future points (no hook, or beyond the epoch budget) delayed
stopping. Four failing selection tests reproduce these behaviors, including
loss of a valid initial checkpoint; a fifth checks the new state-change flag
and selection reset for real label edits at unchanged capacity. All 98 T1 tests
pass, with the portable five-test subset repeated after removing a temporary
absolute import path.

The loop now compares model state and active capacity around a hook, records
`checkpoint_state_changed`, preserves the matching best model/optimizer for
no-op hooks, and waits only for executable future checkpoints within the
budget. Genuine edits still reset selection. No-hook training is unchanged.

## Real-data before/after audit

Three datasets (iris, energy efficiency, yacht), seeds 8/9/10, 60 training cases,
30 maximum epochs and patience 15 compare ordinary training with unchanged
hooks at epochs 8/16. There are 18 runs before and 18 after the repair. All 36
safe-loaded checkpoints, test metrics and retrieval events independently replay.
Both source ZIPs reconstruct their manifest hashes; all manifests show unchanged
source during execution. Before SHA256:
`b1b7f33d1d7bc001a786b88959005449064f98cf0a5273a7222346df66179c84`.
After SHA256:
`a973e1f70375c8ff672a2a054b4c9b52abbc056b7b3a68ccb5253240fc1ad3a0`.

All nine repaired no-op/ordinary pairs have bit-identical model state, Adam
state, predictions, selected epoch, validation loss and epochs run. All nine
ordinary before/after pairs are also identical. Every repaired no-op hook
records unchanged state.

The old hook changed selected epoch 9 to 16 for iris seed 8 and yacht seed 10.
For iris, validation loss worsened from 0.054304 to 0.061560 even though test
accuracy happened to improve from 0.933333 to 0.966667. For yacht, validation
loss worsened from 0.356759 to 0.397327 while test RMSE happened to decrease
from 0.504266 to 0.492752. The repair restores legitimate validation selection
and the ordinary predictor; test scores are not a reason to retain the bug.
The remaining seven conditions have unchanged selected predictors.

Raw artifacts are under
`results/t1_pi20260920/continuation_20261004/core_selection_probe_before` and
`core_selection_probe_after`; the chat outputs contain full replay/comparison
JSON and report. The original diagnostic driver is preserved in the task work
folder. This is an engineering equivalence/selection probe, not a claim of
improved generalization or a broad retention comparison.
