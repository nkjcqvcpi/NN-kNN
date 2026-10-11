# Branch organization: main

`main` (`nnknn/`) is the standalone NN-kNN branch. The four-workstream layout, the shared core, and the propagation rules are in [NNKNN_CORE.md](NNKNN_CORE.md).

## What lives on main

| Area | Paths |
| --- | --- |
| Shared core (identical on all branches) | files listed in `tools/check_nnknn_core.py` |
| Core contract | `docs/NNKNN_CORE.md`, `tools/check_nnknn_core.py`, `tools/nnknn_core_manifest.json`, `tests/test_nnknn_core.py` |
| Supervised notebooks | `nnknn_sample_classification.ipynb`, `nnknn_sample_regression.ipynb` |
| Prior results | `results/prior_baseline/` (see `results/README.md`) |
| Legacy model code | `model/Copy_nnknn_model.py`, `model/nnknn_model_1_4.py`, `model/nnknn_model_with_post_mlp.py`, `model/nn_cdh_old.py`, `model/train_GMM3.py`, `model/cls_model.py`, `model/reg_model.py`, `model/module.py`, `nnknn_reg_yu.py` |
| Reference material | `older versions/`, `notes/`, `papers and presentations/`, `checkpoints/*.h5`, `tmp_eval_section.txt`, `selected_indices*.json`, `config.yaml` |

The legacy and reference material is kept on `main` only. Stage branches carry the shared core plus their own stage code.

## Reorganization of 2026-10-10

- Removed from `main`: the DQN, NEC, and NN-kNN-RL workflows, `datasets/rl_tasks.py`, `tools/run_rl_*.py`, `tools/run_cartpole_all_variants.py`, the CartPole and outdated RL notebooks, the CartPole videos, `notes/CBR plays games.docx` (all maintained on `rl`), and `notes/RDG-LLM.docx` (on `LLM`). Each path was verified to exist on its owning branch before removal.
- Adopted the reconciled core from `neural-cbr` `aeedbd4`, then removed its `model.t1` import (`model/nnknn_model.py` now imports `model.common.nominal`).
- Added the core contract files.
- Moved the prior results under `results/prior_baseline/` without changing their contents.

Earlier history of the mixed repository, the checkpoint-history rewrite, and the recovery bundle are described in the stage branches' `docs/BRANCH_ORGANIZATION.md` and in `../recovery-20261009/`.
