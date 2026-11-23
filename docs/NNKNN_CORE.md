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

`model/common/tracking.py` records the training of a run in TensorBoard event files (curves, final metrics, predictions, weights; summaries stay in files, see the rules below); code reads recorded runs back from them. The run root is `NN-KNN/runs/` next to the worktrees, or `$NNKNN_RUNS`; it must exist (it is never created). `runs/<workstream>/<study>/<unit...>/` is one run (workstream `core`, `t1`, `t2` or `t3`). Every process session that writes to a unit adds one event file to that directory; weights stay in it: `weights/<name>.pt` (named states, written once), `weights/resume/last.pt` (the resume checkpoint; the one before it stays as `prev.pt`, so a crash between writing `last.pt` and recording it resumes from `prev.pt`), `weights/<name>.superseded-s<NNN>.pt` (a named state that a later session cut; renamed, never deleted).

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

## What goes to TensorBoard (owner's rules, 2026-11-22)

- R1. Table 1 (`tools/table1_nnknn_kfold.py`, `tools/run_table1_kfold.py`, `tools/resume_table1_kfold.py`: the upstream authors' published regression benchmark) is published work: its code path, its outputs and its numbers do not change. It has no run tracking. Its CLIs, its output files (`summary_long.csv`, `runs_long.csv`, `table1_like.csv`, `transposed.csv`, `done.json`, `resume_manifest.json` in the output folder, and the NN-kNN checkpoint files `checkpoints/nnknn_regression_best_<dataset>_<family>_fold_<k>.pth` with their `_retr` / `_cdh` companions, relative to the working directory), its resume by checkpoint and its seeding (one continuous unseeded sweep over datasets, folds and families) are those of the upstream code. The shared code keeps everything this path calls (`train_model` checkpoint files without a run, `regression_workflow._checkpoint_path_for_run`, `checkpoint_label`).
- R2. Only training data goes to TensorBoard: per-epoch curves, per-run final metrics, predictions and weights. Scripts that summarise keep their output formats, names, contents and locations (csv/json/md/png/pdf files); summaries are not stored as run units.
- Recording is opt-in. Every pre-existing entry point and CLI, called as before, writes the same files and computes the same numbers as before run tracking existed (dc2bcd4d), so old notebooks and callers keep working. One exception, a bug fix (see "Evaluation feature cache" below): NN-kNN with a trainable feature extractor (image `nnknn_conv_trainable`, post-MLP) can report a different test accuracy than dc2bcd4d, because dc2bcd4d evaluated the restored best state with stale case features; its checkpoints are unchanged, Table 1 is unaffected, and keeping the fix awaits the owner's sign-off. `model.common.tracking` loads tensorboard only on first use, so importing the training code or Table 1 (which use only `NullRun`) needs no tensorboard, as at dc2bcd4d.

## Recording core training and the benchmark tools

- `core.train_retrieval(..., run=NullRun(), resumable=False, log_prefix="")` logs `train/loss`, `val/loss`, `val/metric`, `case/n_active` (and `val/loss_after_checkpoint`) at step = epoch, each tag prefixed by `log_prefix` as given (a caller that trains twice in one unit passes distinct prefixes ending in `/`, e.g. `"ft/"`). With `resumable=True` it calls `run.load_resume()` before the loop and, when `run.checkpoint_due(epoch)`, saves a resume checkpoint at the end of an epoch after which training continues (model, optimizer, best state and optimizer, patience counter, history, `case_default_bias`, the permutation generator and the global RNG). `load_resume` can be called once per session, so at most one `train_retrieval` call per unit is resumable. It cannot be combined with a `checkpoint_hook`.
- `nnknn_model.train_model(..., run=NullRun(), resumable=False)`. Without a run (the default) it writes its checkpoint files as it always did: the best stage-1 state to `<stem>_retr.pth`, the best stage-2 state to `<stem>_cdh.pth` and the final state to `cfg["checkpoint_path"]` (`nnknn_tmp.pth` in the working directory when unset), reloading the best states from those files. With a real run it writes no file outside the run: best states stay in memory, as exact copies on the CPU (no accelerator memory); it logs `val/loss`, `val/acc` (classification) or `val/r2`, and `train/stage` at the global epoch (stage-1 epochs, then stage-2 epochs continuing the count) and saves the named states `best_retrieval` (end of stage 1), `best_adapter` (end of stage 2, when an adapter trains) and `final`. With `resumable=True` it resumes at epoch boundaries of either stage (the checkpoint keeps the `requires_grad` map, so the extractor freeze schedule continues, and `case_default_bias`, which the model draws from the RNG unless `bias_manual_set`) and checkpoints once between the stages, after stage-2 setup. Checkpoints are taken only at epochs after which the same stage continues.
- With or without a run both functions compute the same numbers (same RNG consumption) bit for bit on CPU, and an interrupted and resumed training equals the uninterrupted one (`result_digest`, state digests, history). `tests/test_core_runs.py` checks both; XPU resume is not tested.
- Evaluation feature cache: in eval mode the model caches the case features of its feature extractor, keyed on the case buffer and every extractor parameter and buffer, each by identity, storage address and version counter (identity, so that a new tensor reusing a freed address is not taken for the old one). A restored best state, any `load_state_dict`, an optimizer step or another in-place write refreshes it (a write through `.data` does not: call `_invalidate_case_cache()` after one), so the returned model predicts exactly like a model rebuilt from its final state. Up to 12d4caf (included) `train_model` restored the best state without clearing the cache: with a trainable extractor (post-MLP, `nnknn_conv_trainable`) and a best stage-1 epoch before the last epoch, the returned model evaluated the cases with the extractor of the last validation epoch, so test numbers recorded before this fix differ in those cases. Validation, early stopping and best-epoch selection were not affected (each epoch's training batches cleared the cache). Table 1 (no feature extractor) gives the same numbers with and without the fix.
- Workflows: `train_nnknn_classification_state`, `train_nnknn_regression_state`, `run_single_nnknn_classification_experiment`, `run_single_nnknn_regression_experiment` and `run_nnknn_regression_workflow` pass `run` and `resumable` to `train_model`. Without a run they write the checkpoint files as before (`checkpoint_label` makes the path unique per dataset and label; classification creates the folder of `cfg["checkpoint_path"]`); with a run they write none.
- Benchmark loops (`run_classification_benchmark_methods_on_state`, `run_repeated_classification_model_benchmarks`, `run_regression_benchmark_methods_on_state`, `run_repeated_regression_model_benchmarks`): an optional `unit_run(state, method)` (off by default) is a context manager per method that yields either the open run of that method's unit, which records the method's finals (`accuracy` and `best_metric`, or `rmse_raw`, `rmse_model_space`, `rmse_z`, `best_metric` and the row record `row`), its predictions and the NN-kNN training (resumable, no checkpoint files), or the `RunData` of the method's complete unit. A complete unit is not rerun: its result entry holds the recorded values (`recorded_unit` = its path, no model), so the loops build their rows and summaries in memory exactly as when it ran. In the image benchmarks the unit whose method trains the shared convnet classifier also saves it as the named state `classifier`; when that unit is complete the classification loop loads it from there instead of retraining it (retraining, from the skipped method's seed, is the fallback and is exact on CPU only).
- `tools/run_classification_benchmarks.py` writes `summary.csv`, `runs.csv` and `manifest.json` to a fresh `<--output-dir>/classification_<suite>_<timestamp>/` (default `results/`), as before. With `--study S` it also records one unit per dataset, method and split, `runs/core/S/<dataset>/<method>/<mode>-<index>`, and adds `"study": "S"` to the manifest; the summary files are otherwise the same. Rerunning a study skips complete units (their recorded accuracy fills the summary rows), raises `ConfigMismatchError` if a complete unit was written with another config, and resumes incomplete NN-kNN units from their last checkpoint. A holdout or official unit's config holds its `run_seed` but not the run count, so a study can be extended with more `--runs`; a k-fold unit's config also holds the fold count and base seed.
- Read results with `load_run` (`series`, `records`, ...), not by scanning raw `x/` tags in the event files: a crashed session may have logged steps after its last checkpoint, which stay in its event file and are cut only by the reader.

## Results and logs

Results and logs are not in git. Every branch ignores `results/`, `logs/` and generated report files (figures, tables and manifests under `reports/`). Training data that a caller chooses to record goes to the shared `NN-KNN/runs/` directory next to the worktrees (above). Summary files keep the locations their scripts always used: Table 1 and the classification benchmark write to their output folders (default under `results/`), and the stage runners keep their own result folders until their tracking steps (they will also follow R2).

Finished work stays organized on disk by the handoff that defines it:

- `main`: `results/prior_baseline/` (pre-MCB NN-kNN results; T1 Phase 0/6 references).
- `neural-cbr`: `results/t1/` and `logs/t1/` by T1 implementation phase.
- `rl`: `results/t2/` and `logs/t2/` by T2 implementation step and benchmark stage.
- `llm`: `results/t3/` and `logs/t3/` by T3 development stage.

The `results/README.md` kept with each branch's results maps handoff steps to directories, documents, and status. Run artifacts keep the paths they were written with; when a directory moves, the old and new locations are recorded instead of editing the artifacts.
