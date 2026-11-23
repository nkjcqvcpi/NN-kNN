# T1 run records (TensorBoard units)

Since 2026-11-22 the T1 tools can record their training data as run units through the shared tracking module
(`model/common/tracking.py`, rules in `docs/NNKNN_CORE.md`, "Run tracking"). Recording is opt-in: every tool
takes `--study S`. Called without it, a tool writes exactly the files it wrote before (same names, contents and
locations), so old commands and old results keep working.

The run root is `NN-KNN/runs/` next to the worktrees (or `$NNKNN_RUNS`; it must exist). Every unit holds the
event files of its sessions and its weights under `weights/`. Read units with `load_run` / `iter_runs`, or with
`model/t1/records.py` (`record`, `rows`, `condition_units`, `experiment_rows`), never by scanning raw tags.
Every JSON record the T1 tools write is stored with `records.ordered` (each dict as `{"$items": [[key, value],
...]}`), so readers get back the original key order (`records.record` / `records.rows`).

Unit components are the original names when they are valid and at most 48 characters; other names (spaces,
`::`, long condition-grid labels) become a readable prefix plus a 12-hex hash (`records.unit_component`). The
original name is always in the unit's config and records.

## What each entry point records

| Entry point | Unit | Records, arrays, curves | Weights (named states) | No longer written with `--study` |
|---|---|---|---|---|
| `tools/t1_run.py CONFIG --study S` (also through `tools/t1_parallel.py`, `tools/t1_execute_study.py`, `tools/t1_launch.ps1 -Study`) | `t1/S/<experiment>/<dataset key>/<condition>/s<seed>` | `manifest`, `metrics` (+ `final/<metric>` numbers), `history` (+ curves `<section>/<field>`, e.g. `core/train_loss`, `sync/...`), `case_statistics`, `case_maintenance`, `retrieval_events`, `row` (the shard row), one record per extra JSON file (`maintenance_summary`, `maintenance_scoring_trace`, `calibration_protocol`, `capacity_match_*`, `interventions`, `matched_random_summary`, `adapter_nominal_schema`) | `checkpoint`; the type's `.pt` files under their stems (`M0`, `M1`, `M2`, `MC`, `intervention_state`, `initial`, `matched_full`, `matched_random`, `calibration_snapshots`, `before_capacity_match`); removal retraining `starting_checkpoint`; legacy reference also `best_retrieval`, `best_adapter`, `final` from `train_model` (which records its curves `val/loss`, `val/acc`/`val/r2`, `train/stage` and resumes an interrupted training) | the run directory and every file in it (`run_manifest.json`, `metrics.json`, `history.json`, `*.jsonl`, `checkpoint.pt`, extra `.json`/`.pt`), the `runs__<dataset>.jsonl` row, `legacy_best*.pth`, `starting_checkpoint_s<seed>.pt` |
| `tools/t1_grid_search.py CONFIG --out D --study S` and `... evaluate ... --study S` | `t1/S/<task>/<family>/<stage>/<candidate>/s<seed>` | `trial` (the receipt row), `final/validation_error`, `final/validation/*`, `final/test/*`, arrays `validation_prediction`, `validation_target`, `test_prediction`, `test_target` | `model` (torch: `{model_state, info}`; kNN/CatBoost: `{format: "joblib", bytes}`, load with `t1_grid_search.trial_model`), `data` | `receipt.json`, `failure.json` (a failed trial is a failed unit), `*_predictions.pt`, `model.pt`/`model.joblib`, `data.pt` |
| `tools/t1_grid_prefill.py` | the grid's trial units when its `study.json` holds `"study"` | as the grid | as the grid | as the grid |
| `tools/t1_core_ablation.py CONFIG --out D --study S` | `t1/S/<task>/<variant>/s<seed>` | `trial` (the receipt row), `history` (+ `core/*` curves), `final/*`, arrays `test_prediction`, `test_target` | `model` (`{model_state, core}`; new: the file tool kept no weights) | `receipt.json`, `failure.json`, `test_predictions.pt` |
| `tools/t1_pre_topk_pilot.py --study S` | `t1/S/<dataset>/s<seed>/<masked or all_cases>` | `result` (the run record), `history` (+ `core/*`), `final/*` | `model` (new) | `runs/<dataset>/s<seed>/*.json`, `runs.json` |
| `tools/t1_mask_joint_pilot.py --study S` | `t1/S/<dataset>/s<seed>/mask<m>_frozen<f>` | `result`, `history` (+ `sync/*`), `pretrain_history` (+ `core/*`), `final/selected/*`, `final/baseline/*` | `retriever`, `adapter` of the selected checkpoint (new) | `runs/<dataset>/s<seed>/*.json`, `runs.json` |
| `tools/t1_reviewer_ui.py --out D --study S` | `t1/S/reviewer/<name of D>` (one per session; complete when the server stops with Ctrl+C) | `view/v<NNNN>` per version, curves `m0/*`, `m2/*`, `mc/*` | `v<NNNN>-<stage>` per version, `MC` | `v<NNNN>-<stage>/checkpoint.pt`, `view.json`, `MC.pt` (`latest.json` names the state instead of the folder) |

Unchanged with `--study` (not training data): job stdout logs and receipts of `t1_parallel`/`t1_execute_study`,
`training.log`, the source snapshot zip, `data_snapshot_s<seed>.pt` under `--out`, the session manifest, and every
summary file (`summary.csv`, `paired_vs_*.csv`, `completed.json`, `evaluation.json`, `selection.json`,
`shortlist.json`, `results.json`, `progress.json`, `study.json` (which gains `"study": S`), reports, figures).

Resume: a unit that is complete with the same config is not written again (`[recorded] ...` in the job log; a
grid, ablation or pilot trial is then read back instead of refitted); a complete unit with another config raises
`ConfigMismatchError`; a failed or interrupted unit restarts in place. Only the legacy reference (`train_model`)
resumes inside its training; the other T1 units are written when the run finishes.

## Readers (input side only; outputs unchanged)

Each reader takes either the legacy result directory, read as before, or the run-unit tree of the same runs:

- `tools/t1_summarize.py <runs/t1/S/<experiment>> --out DIR`: same `summary.csv` and `paired_vs_<ref>.csv`
  (byte-identical for the same runs; rows are read in the order the shards would hold them). `--out` is required
  for a unit tree, so no summary is written into the run root.
- `tools/t1_encoder_report.py`, `tools/t1_schedule_search.py` (`select`, `loso`, `report`), `tools/t1_block_accept.py`
  and `tools/t1_freeze.py` (through `t1_schedule_search.load_rows`).
- `tools/t1_cbm_diagnostics.py <unit tree>`: the `manifest`, `case_statistics`, `history`, `case_maintenance`
  records of every condition unit.
- `tools/t1_confirm_report.py --runs <unit tree>`: rows, manifests (source fingerprint, device) and
  `retrieval_events` from the units; baseline test predictions from the grid's trial units when the grid or
  evaluate output recorded them (`"study"` in its `study.json`/`evaluate.json`), else from `test_predictions.pt`.
- `tools/t1_confirm_audit.py <unit tree>`: unit status, row rebuilt from `manifest` + `metrics` against the
  `row` record, unit path, fingerprints and (with `--logs`) the `[done]` lines; `--repair` applies to shards only.

## Selections inlined in the configs

The 43 T1 configs that named a grid `selection.json` (the 2026-10-09 full grid and the 2026-10-13 transfer grid)
now carry the selection itself under `core_selection_source` (`selection`, plus `legacy_path` and `sha256` as
provenance); `configs/t1/core_ablation_20261027.yaml` carries `grid_winners`, which `tools/t1_core_ablation.py`
uses instead of reading the grid. The trained values (`core_by_dataset`) did not change; no config reads a file
under `results/`. The ten transfer selections were no longer on g234: their entries hold the parameters of
`core_by_dataset` and a note. `tools/t1_schedule_search.py prepare` and `tools/t1_freeze.py` now write the
selection into the configs they generate. The frozen confirmation configs therefore no longer match the sha256
values in the freeze record of 2026-10-13 (git history keeps the frozen bytes).
