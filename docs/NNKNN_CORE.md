# Shared NN-kNN core

This file is identical on every branch. It defines the NN-kNN core that the four workstreams share and the rules for changing it.

## Workstreams

| Folder | Branch | Contents |
| --- | --- | --- |
| `nnknn/` | `main` | Standalone NN-kNN model: the core below, its supervised classification/regression baselines, papers, notes, and legacy notebooks. |
| `neural-cbr/` | `neural-cbr` | Core + full-cycle neural CBR (T1): `model/t1/`, `configs/t1/`, `tools/t1_*`, `tests/test_t1_*`. |
| `rl/` | `rl` | Core + reinforcement learning (T2), including NN-kNN-RL: RL workflows, `model/t2/`, `datasets/rl_tasks.py`, `tools/run_rl_*`, `tools/t2_*`, `tests/test_t2_*`. |
| `llm/` | `LLM` | Core + LLM/agent retrieval (T3): `model/t3/`, `tools/t3_*`, `tests/test_t3_*`. |

The T1, T2 and T3 specifications are Xiaomeng Ye's implementation handoffs in `career-2027/03-research-architecture/implementation-handoff/`.

## What the core is

The authoritative list is `CORE_FILES` in `tools/check_nnknn_core.py`; `tools/nnknn_core_manifest.json` records their SHA-256 hashes. In summary:

- `model/nnknn_model.py`: retrieval model, glocal feature weighting, stable case IDs, case activation, case-store operations.
- `model/nn_cdh.py`: NN-CDH regression and classification adaptation networks.
- `model/feature_extractors.py`, `model/device_utils.py`: representations and CPU/CUDA/XPU handling.
- `model/common/`: model construction and training (`core`), learned distances (`geometry`), prediction and loss (`outcomes`), nominal schema and grouped differences (`nominal`), fingerprints and manifests (`artifacts`), run tracking on TensorBoard event files (`tracking`, below).
- `model/classification_workflow.py`, `model/regression_workflow.py`, `model/mlkr.py` and their loaders in `datasets/`: the supervised baseline layer.
- `datasets/paths.py`: `datasets_root()`, the one place that locates the data. The `datasets/` package holds only loader code; the project data files and preprocessed caches live outside git in one directory shared by all worktrees, `NN-KNN/datasets/` next to the worktrees, or the directory named by `NNKNN_DATASETS`. Library downloads are not kept there: a dataset that a library fetches with a default cache of its own stays in that library's default location (scikit-learn: `~/scikit_learn_data`, or `$SCIKIT_LEARN_DATA`; OpenML: the `openml` cache). Only project data and project-preprocessed caches live in `NN-KNN/datasets/`; torchvision has no default location, so MNIST/CIFAR-10/SVHN stay there too.
- `tools/run_classification_benchmarks.py`, `tools/run_table1_kfold.py`, `tools/resume_table1_kfold.py`, `tools/table1_nnknn_kfold.py`: supervised benchmark runners.
- `tools/check_nnknn_core.py`, `tests/test_nnknn_core.py`, `tests/test_tracking.py`, `tests/test_core_runs.py`, and this file.

Core files never import stage code (`model.t1`, `model.t2`, `model.t3`, RL workflows, `datasets.rl_tasks`). Stage code imports the core, never the reverse. `tests/test_nnknn_core.py` enforces this, including imports inside functions.

## Changing the core

1. Make the change on `main` in `nnknn/`, in a commit that touches only core files.
2. Run `python tools/check_nnknn_core.py --write`, then `python -m pytest tests/test_nnknn_core.py -q` and `python codex/smoke_test.py --mode train`.
3. In each stage checkout, copy the core files and the manifest from `main` in one dedicated commit (`git checkout main -- <core files> tools/nnknn_core_manifest.json`), run that branch's tests, and name the source commit in the message.
4. Check all branches from any checkout: `python tools/check_nnknn_core.py --branches main neural-cbr rl LLM`.

Never merge a whole stage branch into another to share core code. A core fix found on a stage branch is first made on `main`, then propagated.

## Run tracking

`model/common/tracking.py` records runs in TensorBoard event files, which are the only record of how a run went and what it produced; code reads results back from them. The run root is `NN-KNN/runs/` next to the worktrees, or `$NNKNN_RUNS`; it must exist (it is never created). `runs/<workstream>/<study>/<unit...>/` is one run (workstream `core`, `t1`, `t2` or `t3`). Every process session that writes to a unit adds one event file to that directory; weights stay in it: `weights/<name>.pt` (named states, written once), `weights/resume/last.pt` (the resume checkpoint; the one before it stays as `prev.pt`, so a crash between writing `last.pt` and recording it resumes from `prev.pt`), `weights/<name>.superseded-s<NNN>.pt` (a named state that a later session cut; renamed, never deleted).

```python
info = resolve_unit(key, config=cfg)            # complete with this config -> skip (another config raises)
if info is None:
    with open_run(key, config=cfg) as run:      # fresh, resume or restart in place, same directory
        ...                                     # setup; data logged here is discarded in a resumed session
        payload, start = run.load_resume(), 0   # payload is None unless resuming
        if payload is not None:                 # rebuild objects and load payload.state first
            start = payload.step + 1
            payload.restore_rng()               # last, right before the loop continues
        for step in range(start, n):
            ...; run.log_step(step, {"train/loss": loss.item()})   # numbers only: .item() at a sync point
            if run.checkpoint_due(step):        # every 300 s; NNKNN_RESUME_EVERY_SECONDS / _STEPS
                run.save_resume({"model": model.state_dict(), "opt": opt.state_dict()}, step=step)
        run.save_state("best", best_state)
        run.log_final({"acc": acc})
data = load_run(info.path)                      # final(), series(), record(), records(), array(), table(), states()
```

- Names: unit components are `[A-Za-z0-9_][A-Za-z0-9._=+-]{0,63}`, without a trailing `.`, Windows device names (`aux`, `nul`, `com1`, ...) or embargoed words; units cannot nest. Steps must increase per tag (`log_step` raises otherwise); a float step raises.
- Exact channel: `x/<tag>` (float64/int64/bool), `rec/<name>` (canonical JSON; `log_records` appends rows), `arr/<name>` (`log_array`, `log_table`), `rec/state/<name>` (written by `save_state`), `rec/config`, and the session records `_run/begin|resume|checkpoint|complete|failed`. The display channel (float32 scalars without the `x/` prefix, a session card) is for TensorBoard only. A `torch.Tensor` value raises, so logging never adds a device sync or consumes RNG.
- Resume: `open_run` raises `UnitCompleteError` on a complete unit, `UnitBusyError` if the latest unfinished session's process still runs on this host (pid plus process start identity, so a reused pid or a reboot does not block; a cheap guard, not a lock; `NNKNN_IGNORE_BUSY=1` skips it), or if another launch wrote its session header at the same moment (a launch that sees a rival steps back and its segment is marked withdrawn, so at most one continues), `ConfigMismatchError` if the config changed (unless `resume="restart"`). It resumes when `last.pt` matches the last `_run/checkpoint` record, otherwise it restarts in place (`resume="restart"` or `NNKNN_RESUME=restart` forces that). A resume cuts every data record written after that checkpoint; a restart cuts all earlier data records. Log everything for step S before `save_resume(step=S)`, and use the checkpoint step axis for every tag of a resumable unit. Saved states hold tensors and plain JSON scalars (numpy scalars are converted; other types raise). `load_resume(map_location=...)` moves the state; `restore_rng()` always restores the RNG states on the CPU side.
- `_run/complete` holds `result_digest`, a SHA-256 over the logical stream without `x/sys/*` and without weight-file bytes (`state_sha256` is included). On CPU an interrupted and resumed run has the same digest and state digests as an uninterrupted one (`tests/test_tracking.py` kills a real process to check this). XPU runs record the same digests, but are not promised to match.
- TensorBoard after a resume: the Python loader (`tensorboard --load_fast=false`) shows one continuous curve for re-logged tags and hides the stale tail of tags that are not logged again; the Rust loader ignores the SessionLog and keeps those stale points.
- Status (`run_status`): `complete`, `failed`, `incomplete` (crash; a cut or zero-filled last record reads as incomplete), `invalid` (damaged record, digest mismatch, missing or wrong-size weight file, ...) or `empty`. Readers use complete units only. `run_status` and `load_run` check weight files by size; only `verify_run`, `load_state` and `load_resume` check their SHA-256, so read weights through `load_state`, not through `state_path`. CLI: `python -m model.common.tracking {ls|show|verify} <path>`; the reader side needs only `tensorboard` and `numpy`, so on accel: `uvx --with tensorboard==2.21.0 --with numpy python -m model.common.tracking ls ../runs`.

## Recording core training and the benchmark tools

- `core.train_retrieval(..., run=NullRun(), resumable=False, log_prefix="")` logs `train/loss`, `val/loss`, `val/metric`, `case/n_active` (and `val/loss_after_checkpoint`) at step = epoch, each tag prefixed by `log_prefix` as given (a caller that trains twice in one unit passes distinct prefixes ending in `/`, e.g. `"ft/"`). With `resumable=True` it calls `run.load_resume()` before the loop and, when `run.checkpoint_due(epoch)`, saves a resume checkpoint at the end of an epoch after which training continues (model, optimizer, best state and optimizer, patience counter, history, `case_default_bias`, the permutation generator and the global RNG). `load_resume` can be called once per session, so at most one `train_retrieval` call per unit is resumable. It cannot be combined with a `checkpoint_hook`.
- `nnknn_model.train_model(..., run=NullRun(), resumable=False)` logs `val/loss`, `val/acc` (classification) or `val/r2`, and `train/stage` at the global epoch (stage-1 epochs, then stage-2 epochs continuing the count). Best states stay in memory, as exact copies on the CPU (no accelerator memory); the run gets the named states `best_retrieval` (end of stage 1), `best_adapter` (end of stage 2, when an adapter trains) and `final`. With `resumable=True` it resumes at epoch boundaries of either stage (the checkpoint keeps the `requires_grad` map, so the extractor freeze schedule continues, and `case_default_bias`, which the model draws from the RNG unless `bias_manual_set`) and checkpoints once between the stages, after stage-2 setup. Checkpoints are taken only at epochs after which the same stage continues. `cfg["checkpoint_path"]` and the workflows' `checkpoint_label` / `checkpoint_label_prefix` are accepted and ignored: nothing is written outside a run.
- With the default `NullRun` both functions compute exactly what they computed before (same RNG consumption, same numbers); with a real run the result is the same bit for bit on CPU, and an interrupted and resumed training equals the uninterrupted one (`result_digest`, state digests, history). `tests/test_core_runs.py` checks both; XPU resume is not tested.
- Known limit (unchanged behaviour, kept so that numbers match earlier runs): with a feature extractor (post-MLP, image conv models), `train_model` restores the best state without clearing the model's evaluation feature cache, so the returned model evaluates the cases with the extractor of the last validation epoch. A model rebuilt from the `final` state has no cache and can predict differently from the recorded `predictions` / `accuracy` / RMSE. The recorded numbers are those of the returned model.
- Workflows: `train_nnknn_classification_state`, `train_nnknn_regression_state`, `run_single_nnknn_classification_experiment`, `run_single_nnknn_regression_experiment` and `run_nnknn_regression_workflow` pass `run` and `resumable` to `train_model`. The benchmark loops (`run_classification_benchmark_methods_on_state`, `run_repeated_classification_model_benchmarks`, `run_regression_benchmark_methods_on_state`, `run_repeated_regression_model_benchmarks`) take `unit_run(state, method)`, a context manager per method that yields the run of that method's unit, or None for a complete unit, which is then skipped. Each method records its finals (`accuracy`, or `rmse_raw`, `rmse_model_space`, `rmse_z`, `best_metric` and the row record `row`) and its predictions. In the image benchmarks the unit whose method trains the shared convnet classifier also saves it as the named state `classifier`; when that unit is skipped, the classification loops load it back through `unit_state(state, method, "classifier")` instead of retraining it (retraining, from the skipped method's seed, is the fallback and is exact on CPU only).
- Unit layout of the benchmark tools (one unit per dataset, method and split; rerunning a study skips complete units, raises `ConfigMismatchError` if a complete unit was written with another config, and resumes incomplete NN-kNN units from their last checkpoint). Generated study and report names use UTC timestamps.
  - `tools/run_classification_benchmarks.py [--study S]` (default `classification_<suite>_<timestamp>`): `runs/core/S/<dataset>/<method>/<mode>-<index>`. A holdout or official unit's config holds its `run_seed` but not the run count, so a study can be extended with more `--runs`; a k-fold unit's config also holds the fold count and base seed. Summary in the report unit `runs/core/S/_reports/summary/<timestamp>` (row records `summary`, `runs`; records `manifest`, `units`).
  - `tools/run_table1_kfold.py [--study S]` (default `table1_kfold_<timestamp>`) and `tools/resume_table1_kfold.py --study S`: `runs/core/S/<dataset>/<family>/f<fold>` (NN-kNN families: row record `rows` with the pure and adapted RMSE, finals, `y_pred_pre_z` / `y_pred_post_z`, named states) and `runs/core/S/<dataset>/<method>/f<fold>` (baselines); tables in `runs/core/S/_reports/table1/<timestamp>` (row records `summary_long`, `runs_long`, `table1_like`, `transposed`; `tools.table1_nnknn_kfold.read_table1_report` reads them). In a study every NN-kNN unit starts from the seed `base_seed + fold - 1`, so a skipped or resumed unit leaves the others unchanged; an untracked call (`study=None`) keeps the old unseeded sequence, so its numbers differ from a study's. A skipped unit's artifact holds its finals, predictions and `run_path` read from the event file; its model is rebuilt from `final` only with `rebuild_skipped=True`.
  - Report tables keep their column order: `tools.table1_nnknn_kfold.record_frame` writes a table as a row record plus `columns/<name>`, `read_frame` reads it back (row records alone come back with sorted keys).
  - The tools print their tables to stdout and write no CSV, JSON or checkpoint files; the report units record which unit results (`result_digest`) the tables were read from.
- Read results with `load_run` (`series`, `records`, ...), not by scanning raw `x/` tags in the event files: a crashed session may have logged steps after its last checkpoint, which stay in its event file and are cut only by the reader.

## Results and logs

Results and logs are not in git. New runs of the core tools are recorded in the shared `NN-KNN/runs/` directory next to the worktrees, as described above. Every branch ignores `results/`, `logs/` and generated report files (figures, tables and manifests under `reports/`).

Earlier results stay on disk as legacy, organized by the handoff that defines them; they are not converted, and new code does not read them. The stage runners keep writing there until their own tracking step:

- `main`: `results/prior_baseline/` (pre-MCB NN-kNN results; T1 Phase 0/6 references).
- `neural-cbr`: `results/t1/` and `logs/t1/` by T1 implementation phase.
- `rl`: `results/t2/` and `logs/t2/` by T2 implementation step and benchmark stage.
- `llm`: `results/t3/` and `logs/t3/` by T3 development stage.

The `results/README.md` kept with each branch's results maps handoff steps to directories, documents, and status. Run artifacts keep the paths they were written with; when a directory moves, the old and new locations are recorded instead of editing the artifacts.
