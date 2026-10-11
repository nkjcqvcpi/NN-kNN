# Branch organization and Git recovery

| Branch | Folder | Stage-specific implementation | Entry points and tests |
| --- | --- | --- | --- |
| main | nnknn/ | none: standalone NN-kNN, shared core, prior results | model/nnknn_model.py, model/common, tools/check_nnknn_core.py, tests/test_nnknn_core.py |
| neural-cbr | neural-cbr/ | T1 supervised full-cycle neural CBR, classification and regression | model/t1, configs/t1, tools/t1_*, tests/test_t1_* |
| rl | rl/ | DQN, NEC, NN-kNN actor/critic, PPO, TD3, MCB experiments and T2 audits | model/*rl* / *ppo* / *td3* / *nec*, model/t2, datasets/rl_tasks.py, tools/run_rl_*, tools/t2_*, tests/test_t2_* |
| LLM | llm/ | T3 retrieval contracts, lexical queries, frozen-host and public utility experiments | model/t3, tools/t3_*, tests/test_t3_* |

Stage documents, result tables, logs and notebooks follow their owning branch. `docs/t1_spec/` is the historical specification namespace: each branch keeps its own stage plan plus the shared experiment contract. The current neural-cbr checkout excludes RL and LLM implementations. Git ancestors still record the original mixed development; the branch split changes current trees, not authorship history.

## Common code

The shared NN-kNN core is defined in [NNKNN_CORE.md](NNKNN_CORE.md); `tools/check_nnknn_core.py` lists its files and `tools/nnknn_core_manifest.json` records their hashes. Since 2026-10-10 the core originates on `main` and is copied unchanged to each stage branch in one dedicated commit; `python tools/check_nnknn_core.py --branches main neural-cbr rl LLM` checks all four. Core files never import stage code. Do not merge an entire stage branch to share code.

On neural-cbr, the `model/t1` paths for the five extracted modules remain compatibility imports. T2 and T3 code import `model.common` directly.

## Existing RL work

The original rl branch was merged with the mixed neural-cbr tip before removing T1/T3 stage files. Its PPO, TD3, CNN, broader task registry, experiments, and reports were preserved. NN-kNN-RL changes merged without textual conflicts; all T2 tests and the actor/critic smoke passed. Conflicts in shared model/device utilities and task metadata were reconciled by retaining both APIs. Historical experiment results remain historical; routine smoke checks do not revalidate published multi-run conclusions.

## Reorganization of 2026-10-10

- The shared core was synced from `main`: the duplicate-case non-self bias fix (`model/common/core.py`, first made on neural-cbr `4cc1f33`), the `model.common.nominal` import in `model/nnknn_model.py` (previously `model.t1.nominal`, which does not exist on this branch), and the core contract files.
- Standalone NN-kNN material that `main` maintains was removed: legacy model variants, supervised sample notebooks, `older versions/`, `papers and presentations/`, non-RL notes, COVID/psych `.h5` files, and the pre-MCB supervised results. All 91 files were verified byte-identical to tracked copies on `main`.
- 1,009 RL checkpoints and 84 `results/rl/` runs that the branch split had left in the neural-cbr checkout were moved here (ignored files, SHA-256 verified).
- `results/` and `logs/` are organized by T2 handoff step under `results/t2/<step>/` and `logs/t2/<step>/`; see `results/README.md`. `results/t2/PATH_MAP.json` maps every previous `results/r760`, `results/g234`, `results/t2_pi20260920`, `results/rl_*`, `logs/r760` and `logs/g234` path to its new location.
- T2 evidence that existed only in the Codex task workspace was copied into the matching step's `codex_task_20261002/` folder.

## Git recovery boundary

Git fsck found no object corruption. The previous filter-repo pass rewrote local branches and upstream tracking refs, while origin still advertised the original commits. It left tracked checkpoints, so ordinary pushes would reject the divergent history. The parent directory NN-KNN is not a Git repository; run Git inside neural-cbr, rl, llm, or nnknn (main).

The full pre-change .git directory, a verified all-ref bundle, old filter maps, local checkpoint copies, validation logs, and repair scripts are preserved outside the repository in `../recovery-20261009/`. That backup intentionally retains old checkpoint history. Local ignored datasets/checkpoints and original untracked experiment artifacts are preserved.

Checkpoint removal rewrites local branch history for both `*.pth` and `*.pt`; ignored files are kept locally. Remote tracking refs describe the real remote history and may still contain checkpoints until GitHub is updated. No upstream/Heuzi branches are published or changed. Publishing rewritten origin branches requires explicit approval and exact force-with-lease checks against the advertised old tips. Other clones must then re-clone or deliberately reconcile rewritten history. GitHub PR refs, forks and LFS server storage are outside a local branch-history cleanup.

Original commit IDs and source fingerprints in experiment reports are preserved as evidence of the runs. They will not equal the reorganized source fingerprint.

## Validation

- 2026-10-10: rl: 63 tests passed (36 T2 + 27 core); import, training and NN-kNN-RL smoke passed; tools/check_nnknn_core.py OK.
- Branch split (2026-10-09), neural-cbr: 140 tests passed; import and synthetic regression training smoke passed.
- rl: 36 T2 tests passed; import and multi-variant NN-kNN actor/critic smoke passed.
- LLM: 71 tests passed; import smoke passed.
- Tests used workspace temporary directories because the existing system pytest temporary directory is inaccessible.

Exact source-file ownership and exclusions are recorded in `REORGANIZATION_MANIFEST.json`. See the external recovery directory for commit maps and checkpoint-history scan evidence.
