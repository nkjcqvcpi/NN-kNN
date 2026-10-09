# Branch organization and Git recovery

| Branch | Stage-specific implementation | Entry points and tests |
| --- | --- | --- |
| neural-cbr | T1 supervised full-cycle neural CBR, classification and regression | model/t1, configs/t1, tools/t1_*, tests/test_t1_* |
| rl | DQN, NEC, NN-kNN actor/critic, PPO, TD3, MCB experiments and T2 audits | model/*rl* / *ppo* / *td3* / *nec*, model/t2, datasets/rl_tasks.py, tools/run_rl_*, tools/t2_*, tests/test_t2_* |
| LLM | T3 retrieval contracts, lexical queries, frozen-host and public utility experiments | model/t3, tools/t3_*, tests/test_t3_* |

Stage documents, result tables, logs and notebooks follow their owning branch. `docs/t1_spec/` is the historical specification namespace: each branch keeps its own stage plan plus the shared experiment contract. The current neural-cbr checkout excludes RL and LLM implementations. Git ancestors still record the original mixed development; the branch split changes current trees, not authorship history.

## Common code

- `model/nnknn_model.py`: retrieval model, glocal feature weighting, case IDs, normalization, and case-store operations. The reconciled implementation preserves both RL in-place overwrite and T1 stable IDs.
- `model/nn_cdh.py` and `model/feature_extractors.py`: reusable neural adaptation and feature extraction.
- `model/device_utils.py`: CPU/CUDA/XPU selection, dtype/optimizer helpers, and runtime fingerprints from both branches.
- `model/common/core.py`: model configuration, construction, training and optimizer helpers.
- `model/common/geometry.py`: learned case/query distances and active representations.
- `model/common/artifacts.py`: source/environment fingerprints, manifest and JSON serialization helpers.
- `model/common/outcomes.py` and `model/common/nominal.py`: prediction/loss handling and nominal schema/adaptation needed by the shared construction/evaluation interface.
- Classification/regression workflows, their datasets, and benchmark utilities remain available as the reusable supervised baseline layer on all three branches.

The T1 paths for the five extracted modules are compatibility imports on neural-cbr. T3 drivers import `model.common` directly and do not depend on `model.t1`. Keep shared changes in small dedicated commits and cherry-pick them to the other stage branches; merging an entire stage branch would reintroduce unrelated stage files. Shared modules have identical bytes across the three reorganized tips.

## Existing RL work

The original rl branch was merged with the mixed neural-cbr tip before removing T1/T3 stage files. Its PPO, TD3, CNN, broader task registry, experiments, and reports were preserved. NN-kNN-RL changes merged without textual conflicts; all T2 tests and the actor/critic smoke passed. Conflicts in shared model/device utilities and task metadata were reconciled by retaining both APIs. Historical experiment results remain historical; routine smoke checks do not revalidate published multi-run conclusions.

## Git recovery boundary

Git fsck found no object corruption. The previous filter-repo pass rewrote local branches and upstream tracking refs, while origin still advertised the original commits. It left tracked checkpoints, so ordinary pushes would reject the divergent history. The parent directory NN-KNN is not a Git repository; run Git inside nnknn-work or the explicit worktree directories.

The full pre-change .git directory, a verified all-ref bundle, old filter maps, local checkpoint copies, validation logs, and repair scripts are preserved outside the repository in `../recovery-20261009/`. That backup intentionally retains old checkpoint history. Local ignored datasets/checkpoints and original untracked experiment artifacts are preserved.

Checkpoint removal rewrites local branch history for both `*.pth` and `*.pt`; ignored files are kept locally. Remote tracking refs describe the real remote history and may still contain checkpoints until GitHub is updated. No upstream/Heuzi branches are published or changed. Publishing rewritten origin branches requires explicit approval and exact force-with-lease checks against the advertised old tips. Other clones must then re-clone or deliberately reconcile rewritten history. GitHub PR refs, forks and LFS server storage are outside a local branch-history cleanup.

Original commit IDs and source fingerprints in experiment reports are preserved as evidence of the runs. They will not equal the reorganized source fingerprint.

## Validation

- neural-cbr: 140 tests passed; import and synthetic regression training smoke passed.
- rl: 36 T2 tests passed; import and multi-variant NN-kNN actor/critic smoke passed.
- LLM: 71 tests passed; import smoke passed.
- Tests used workspace temporary directories because the existing system pytest temporary directory is inaccessible.

Exact source-file ownership and exclusions are recorded in `REORGANIZATION_MANIFEST.json`. See the external recovery directory for commit maps and checkpoint-history scan evidence.
