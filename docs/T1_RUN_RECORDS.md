# T1 run records (TensorBoard units)

Owner's rules (2026-11-22, final): every T1 entry point that trains or evaluates **requires `--study S`** and
records its per-run data and weights under the run root only; there is no file-only mode any more. Summary,
report and plotting scripts read **only run units** and write their outputs under `<worktree>/results/`
(git-ignored; same file names, formats and columns as before). Reports written by agents or people go to
`<worktree>/reports/` (tracked). **Nothing complete is trained again**: a rerun detects complete work before any
training and loads what it needs from the units. Old result files on g234 under `results/` are left as they are and
are no longer read by the tools.

The run root is `NN-KNN/runs/` next to the worktrees (or `$NNKNN_RUNS`; it must exist). Every unit holds the
event files of its sessions and its weights under `weights/` (resume files under `weights/resume/` are kept after
the unit completes). Read units with `load_run` / `load_unit` / `iter_runs` (`model/common/tracking.py`, rules in
`docs/NNKNN_CORE.md`, "Run tracking"), or with `model/t1/records.py` (`record`, `rows`, `condition_units`,
`experiment_rows`), never by scanning raw tags. Every JSON record the T1 tools write is stored with
`records.ordered` (each dict as `{"$items": [[key, value], ...]}`), so readers get back the original key order.

Unit components are the original names when they are valid and at most 48 characters; other names (spaces,
`::`, long condition-grid labels) become a readable prefix plus a 12-hex hash (`records.unit_component`). The
original name is always in the unit's config and records.

## Entry points (all require `--study S`)

| Entry point | Unit | Records, arrays, curves | Weights (named states) |
|---|---|---|---|
| `tools/t1_run.py CONFIG --study S` (also through `tools/t1_parallel.py --study S`, `tools/t1_execute_study.py --study S`, `tools/t1_launch.ps1 -Study S`) | run: `t1/S/<experiment>/<dataset key>/<condition>/s<seed>`; stage: `t1/S/<experiment>/<dataset key>/_<stage>/s<seed>` | run: `manifest`, `metrics` (+ `final/<metric>`), `history` (+ curves `<section>/<field>`), `case_statistics`, `case_maintenance`, `retrieval_events`, `row`, one record per extra JSON part (`maintenance_summary`, `maintenance_scoring_trace`, `calibration_protocol`, `capacity_match_*`, `interventions`, `matched_random_summary`, `adapter_nominal_schema`); stage: `history` (+ `core/*` curves) | run: `checkpoint`; the type's parts (`M0`, `M1`, `M2`, `MC`, `intervention_state`, `initial`, `matched_full`, `matched_random`, `calibration_snapshots`, `before_capacity_match`, `starting_checkpoint`); legacy reference also `best_retrieval`, `best_adapter`, `final` from `train_model` (which resumes an interrupted training); `rng`, `ids`. Stage: `objects`, `rng`, `ids`, a readable state dict per model (`model`, `adapter`) |
| `tools/t1_grid_search.py CONFIG --study S --out D` and `... evaluate ... --study S --out D2` | `t1/S/<task>/<family>/<stage>/<candidate>/s<seed>` | `trial` (the receipt row), `final/validation_error`, `final/validation/*`, `final/test/*`, arrays `validation_prediction`, `validation_target`, `test_prediction`, `test_target` | `model` (torch: `{model_state, info}`; kNN/CatBoost: `{format: "joblib", bytes}`, load with `t1_grid_search.trial_model`), `data` |
| `tools/t1_grid_prefill.py` | the grid's trial units (the grid's `study.json` must hold `"study"`; grids without one are refused) | as the grid | as the grid |
| `tools/t1_core_ablation.py CONFIG --study S [--out D]` | `t1/S/<task>/<variant>/s<seed>` | `trial`, `history` (+ `core/*`), `final/*`, arrays `test_prediction`, `test_target` | `model` (`{model_state, core}`) |
| `tools/t1_pre_topk_pilot.py --study S` | `t1/S/<dataset>/s<seed>/<masked or all_cases>` | `result`, `history` (+ `core/*`), `final/*` | `model` |
| `tools/t1_mask_joint_pilot.py --study S` | run: `t1/S/<dataset>/s<seed>/mask<m>_frozen<f>`; stage: `t1/S/<dataset>/s<seed>/pretrain_mask<m>` | `result`, `history` (+ `sync/*`), `pretrain_history` (+ `core/*`), `final/selected/*`, `final/baseline/*` | `retriever`, `adapter` of the selected checkpoint, `rng`, `ids`; stage: `objects`, `rng`, `ids`, `base` |
| `tools/t1_reviewer_ui.py --out D --study S` | `t1/S/reviewer/<name of D>` (one per session; complete when the server stops with Ctrl+C) | `session_manifest` (seed, epochs, data version, injected label errors, rule, source, environment), `view/v<NNNN>` per version, curves `m0/*`, `m2/*`, `mc/*` | `v<NNNN>-<stage>` per version, `MC` (`latest.json` in D names the current state) |
| `codex/smoke_test.py --mode train|classification --study S` (copied from main) | `core/S/smoke/<mode>` | see main's `docs/NNKNN_CORE.md` | |

Without `--study` every entry point stops with an argparse error (exit code 2); `Runner(cfg, out, study)`,
`t1_grid_search.run_trial(..., study=)`, `t1_core_ablation.run_trial(..., study=)` and
`ReviewerSession(..., run=)` raise `TypeError` without one. No entry point writes per-run files any more
(`run_manifest.json`, `metrics.json`, `history.json`, `*.jsonl`, `checkpoint.pt`, `runs__<dataset>.jsonl` rows,
`legacy_best*.pth`, `starting_checkpoint_s<seed>.pt`, `receipt.json`, `failure.json`, `*_predictions.pt`,
`model.pt`/`model.joblib`/`data.pt`, the pilots' `runs/` and `runs.json`, the reviewer's version folders and `MC.pt`).
What stays a file is not run data: job logs and receipts of `t1_parallel`/`t1_execute_study`, `training.log`, the
grid's and the ablation's job files (`shortlist.json`, `selection.json`, `completed.json`, `partial.json`,
`evaluation.json`, `results.json`, `progress.json`, `aliases.json`, `study.json`/`evaluate.json` with
`"study": S`), the reviewer's `source_snapshot.zip`, `latest.json` and `server.json` (server plumbing), and
`t1_run --out` (default `results/t1_pi20260920`), which holds only the inputs: the source snapshot zip and
`<dataset>/data_snapshot_s<seed>.pt`. The job files hold the selection and the job status, not trial numbers: the
`evaluation` entries of a grid's `completed.json`/`evaluation.json` and the `trials` entries of an ablation job's
`completed.json`/`partial.json` only name their trials (`records.pointer`: status, stage, candidate id or variant,
seed); every number is read from the trial unit (`records.trial_row` for grid trials).

Platform (owner, 2026-11-23): Linux is the primary platform; T1 studies and tests run on r760 on CPU
(`/mnt/largedata/NN-KNN`, `source env.sh`; TMPDIR on local disk). g234 (Windows, XPU) is no longer used for tests or
runs. `tests/conftest.py` still gives pytest a short temp root on Windows (`%TEMP%\nk<random>`, or below
`NNKNN_TEST_TMP`) when `--basetemp` is not given. Float goldens (`tests/goldens/t1_tuning_toy_linux_cpu.json`) were
recorded on r760 and are checked with a relative tolerance of 1e-5 and equal best epochs, since the owner does not
require bitwise identity and an allocator or BLAS change may move the last digits.

Notebooks are left as they are (none in this worktree). Table 1 (`tools/*table1*`) is published work and keeps
writing its own files without a run.

## Never retrained (O4)

`tools/t1_run.py` records the training that several runs share as a **stage unit** of its own:

| Experiment type | Stage units (`_<stage>`) | Runs that continue them |
|---|---|---|
| retention, reuse, sync, removal_retraining, revise | `_core` (per arm: `_core__<arm>`): the trained core (`model`, `cc`, `tr` with optimizer and history) | every condition of that dataset and seed (and arm) |
| encoder_ablation | `_core__<arm>` with the per-epoch stability tracker | retrieval only, frozen reuse and sync conditions of the arm |
| adapted_retrained_removal, reuse_retention | `_core` with the initial adapter (`adapter`, `info`) | every condition and K |
| removal_retraining | `_retrained_K<k>`: the retrained removal of one K (`model`, `archive`, `res`) | the treatments of that K |
| revise | `_revise`: the whole flagging ablation (`rows` with their M0/M1/M2/MC artifacts) | every method and budget |

A stage keeps the exact objects (named state `objects`, written with `records.pack`: `torch.save` of the dict, so
parameters, buffers, optimizer moments bound to their parameters, plain attributes, train/eval modes and Python
value types all come back), the RNG state its training left (`rng`, `tracking.capture_rng`) and the positions of the
process-wide ID counters (`ids`: maintenance event ids `<run>-m<n>` and intervention ids `<run>-i<n>` number on
across the runs of one process, `records.ID_COUNTERS`). Every run unit also records `rng` and `ids` at its end.

Before any training of a dataset and seed, `Runner` checks the units in the order an uninterrupted run would
write them:

- `Runner.skip(ds, seed, label, cond)`: a complete run is not trained; the RNG state it recorded is restored
  and its ID counter positions (`records.restore_rng_of`), so the next run starts from the state it had in the
  uninterrupted study.
- `Runner.stage(ds, seed, name, units, compute)`: when every run in `units` is complete the stage is neither
  trained nor loaded; else a complete stage unit is loaded (objects, then its RNG state) and an incomplete one is
  computed and recorded before any run uses it (`records.stage_unit`).
- A complete unit with another config raises `ConfigMismatchError` before training (`check_recorded` checks every
  complete unit of the dataset and seed, stages included).

Writing a unit leaves the RNG state as the computation left it (captured before writing, restored after), so a
partly rerun study computes the same numbers and weights as an uninterrupted one. This is tested for every
experiment type in `tests/test_t1_never_retrain.py` (a run crashed at a chosen condition, then rerun: no core,
adapter or condition training for complete units, and every record, curve and named state equal to a straight
run), and on g234 in fresh processes (complete study copied, 13 runs of all ten types deleted, rerun: only those
runs trained, stages loaded, every record, curve and weight equal; the `rng` states differ only in whether the unused
XPU generator state is present). Two named states are equal in meaning but not by digest between two processes:
every `rng` state's Python `random` part (the T1 tools never seed Python's `random`, so its state differs per
process; numpy, torch, the generators and XPU are equal), and a stage's `objects` pickle where a pickled object
carries a timing (e.g. `res.summary.selection_seconds`). Compare those semantically (or leave the Python part out). Units recorded before 2026-11-23 have no `rng` state: skipping one prints a warning, since later runs of that
dataset and seed may then differ from an uninterrupted study.

The other tools: a complete grid, ablation or pilot trial is read back before anything is fitted (its `trial` /
`result` record); `t1_mask_joint_pilot` records the pretrained core of each mask setting as the stage unit
`pretrain_mask<m>` and trains it only when one of its two continuations is incomplete. The legacy reference resumes
an interrupted `train_model` inside its unit.

## Identity: configuration, not code (owner, 2026-11-23)

A unit is identified by its configuration (and data definition) only. The source fingerprint, the checkout path
(`code_repository`), the git commit, the source snapshot name and timestamps are provenance: they are recorded (each
session header holds `git_commit`, `dirty`, `source_fingerprint`; `t1_run` manifests hold
`source_fingerprint_sha256`, `source_changed_during_run`, `source_snapshot`; grid and ablation trial rows hold
`source_sha256`) but they are not part of a unit's identity and never block resuming a study, reading units or
summarising a study that spans several code states. `tracking.config_identity` (core, `docs/NNKNN_CORE.md`) hashes the
config without the top-level provenance keys (`tracking.PROVENANCE_KEYS`, among them `source_sha256`), and
`resolve_unit`/`load_unit`/`open_run` compare identities. Units recorded before 2026-11-23 carry `source_sha256` in
their grid or ablation trial config and no `config_identity` in their session header; they are compared through
their recorded config with that key left out, so they are the same units for the new code: complete ones are read
back and never trained again, incomplete ones resume. A change of anything else (epochs, seeds, parameters, splits,
dataset entry, runtime) still raises `ConfigMismatchError`.

- `t1_grid_search`: a grid output (`study.json`) resumes when its configuration (`configuration_id`), determinism
  setting and study match; an evaluate output (`evaluate.json`) when its grid and grid configuration match. Another
  source fingerprint is recorded (each evaluate invocation holds its `source_sha256`), not refused. The grid's
  `selection.json` keeps the source of the grid's `study.json`.
- `t1_grid_prefill`: `--grid-repo` (default: this checkout) may be any code state whose `run_trial` takes no source
  argument (a checkout from before this change is refused before anything is fitted); `selection.json` and
  `shortlist.json` are compared without their `source_sha256`.
- `t1_confirm_report` analyses rows of several source fingerprints together and lists them per task
  (`--allow-mixed-code` has no effect); `--expect-fingerprints` remains an opt-in check.
- `t1_confirm_audit` lists source fingerprints and runs whose source changed during the run; neither fails the
  audit (`--expect-fingerprint` remains an opt-in check).
- Drivers: the receipts of `t1_parallel` and `t1_execute_study` hold no source hash, so a job with exit code 0 stays
  finished after the code moves. `execution_plan.json` records the source of the plan as provenance only.

What a unit's config holds, and so when a unit is reused:

- `tools/t1_run.py`: the run config (experiment, dataset entry, seed, condition, resolved config) without the
  config's `seeds` list, which `--seeds` rewrites; a stage's condition is `{"stage": <name>}`. A study started one
  job per seed (`t1_parallel`, `t1_execute_study`) therefore resumes with a plain `t1_run` or `t1_launch.ps1`, and
  the reverse. The manifest, the row and the run id still use the full config, so rows and numbers are unchanged.
  Device is not part of a unit's identity.
- Grid trials (`t1_grid_search`, `evaluate`, `t1_grid_prefill`): task, family, candidate id and parameters, stage,
  seed, epochs, patience, test flag, the data definition (the dataset entry and its make_splits options, carried
  by `split_loader`) and the runtime (`torch_threads`, deterministic kernels); not the source fingerprint (see
  Identity). A trial fitted on other splits or another dataset entry raises `ConfigMismatchError`.
- Core ablation trials: task, variant, resolved settings, seed, data definition, runtime (not the source fingerprint).

Unit paths: the tracking module refuses an event file path of 240 characters or more. `t1_grid_search` and
`t1_core_ablation` check every unit path of the job before any trial runs. `t1_run` refuses a study whose paths
are too long even for a one-character condition label, and warns when a 48-character label would be too long.

## Summaries (read run units only; outputs under `<worktree>/results/`)

| Tool | Input | Default output |
|---|---|---|
| `tools/t1_summarize.py <runs/t1/S/<experiment>> --metric ...` | condition units' `row` records, in the order the shards of the same runs held them | `results/t1/S/<experiment>/summary.csv`, `paired_vs_<ref>.csv` (`--out` overrides) |
| `tools/t1_encoder_report.py <runs/t1/S/<experiment>>` | rows | `results/t1/S/<experiment>/encoder_*.csv`, `encoder_report.md` (`--out` overrides) |
| `tools/t1_schedule_search.py select|loso|report <runs/t1/S/<experiment>> ...` | rows (`load_rows`) | `results/t1/S/<experiment>/selection/`, `loso.json`, `test_report/` of the first tree (`--out` overrides) |
| `tools/t1_block_accept.py rules|block|split` | rows through `t1_schedule_search.load_rows` | `results/t1/S/<experiment>/block_accept_<command>.json` of the first pool tree (`--out` overrides) |
| `tools/t1_freeze.py` | rows through `t1_schedule_search.load_rows` | record: `results/t1/S/<experiment>/freeze/freeze_record.json` (+ `final_map.json`) of the first `--cls-pool` tree (`--record` overrides); the confirmation configs go to `--out-dir` (required: they are run inputs, kept under `configs/t1/`) |
| `tools/t1_cbm_diagnostics.py <runs/t1/S/<experiment>>` | `manifest`, `case_statistics`, `history`, `case_maintenance` of every condition unit | prints the markdown; `--json`/`--md` (put them under `results/`) |
| `tools/t1_confirm_report.py --runs <runs/t1/S/<experiment>>... --baselines <grid or evaluate output>...` | rows, manifests (source fingerprints listed as provenance, device), `retrieval_events`; baseline test metrics and predictions from the grid's evaluation trial units (the output's `completed.json`/`evaluation.json` says which; an output that records no `"study"` is refused) | `results/t1/S/<experiment>/confirm_report/` of the first `--runs` tree (`--out` overrides) |
| `tools/t1_confirm_audit.py <runs/t1/S/<experiment>>` | unit status, row rebuilt from `manifest` + `metrics` against `row`, unit path, fingerprints, `[done]` lines with `--logs`; stage units are checked for status only | prints; `--json` (put it under `results/`) |
| `tools/t1_core_ablation.py CONFIG --study S [--summarize-only]` | every variant's trial units (`trial` records) in study S; baselines from the grid's evaluation trial units (a grid that records no `"study"` gives no baselines, with a warning) | `results/t1/S/core_ablation/summary.{md,csv,json}` (`--out` overrides; an `--out` whose `study.json` names another study is refused) |
| `tools/t1_pre_topk_pilot.py`, `tools/t1_mask_joint_pilot.py` | their run units | `--out`, default `results/t1/...` |

A path that holds no run units (for example a legacy result folder with `runs__*.jsonl` shards) is refused
(`records.require_unit_tree`). A grid's `selection.json` and the configs' frozen selections are still read where a
tool needs a grid winner, and a grid's `completed.json`/`evaluation.json` to know which evaluation trials (candidate,
seeds) to take; the numbers themselves come only from the trial units. Old grid or ablation outputs on g234 whose
job files hold full rows and record no study are not read for numbers.

## Selections

The configs keep naming their grid `selection.json` by path under `core_selection_source`, as frozen (the
pre-registered `configs/t1/confirm_final_20261013/` configs are byte-identical to their 2026-10-13 freeze; no run
path reads that entry, only `tools/t1_freeze.py` as provenance). Only `configs/t1/core_ablation_20261027.yaml`
carries `grid_winners` (copied from the grid's selection files), which `tools/t1_core_ablation.py` uses instead of
reading the grid; without it the tool reads `<grid>/<task>/nnknn/selection.json` as before.
