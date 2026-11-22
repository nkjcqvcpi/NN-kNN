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
- `tools/check_nnknn_core.py`, `tests/test_nnknn_core.py`, `tests/test_tracking.py`, and this file.

Core files never import stage code (`model.t1`, `model.t2`, `model.t3`, RL workflows, `datasets.rl_tasks`). Stage code imports the core, never the reverse. `tests/test_nnknn_core.py` enforces this, including imports inside functions.

## Changing the core

1. Make the change on `main` in `nnknn/`, in a commit that touches only core files.
2. Run `python tools/check_nnknn_core.py --write`, then `python -m pytest tests/test_nnknn_core.py -q` and `python codex/smoke_test.py --mode train`.
3. In each stage checkout, copy the core files and the manifest from `main` in one dedicated commit (`git checkout main -- <core files> tools/nnknn_core_manifest.json`), run that branch's tests, and name the source commit in the message.
4. Check all branches from any checkout: `python tools/check_nnknn_core.py --branches main neural-cbr rl LLM`.

Never merge a whole stage branch into another to share core code. A core fix found on a stage branch is first made on `main`, then propagated.

## Run tracking

`model/common/tracking.py` records runs in TensorBoard event files, which are the only record of how a run went and what it produced; code reads results back from them. The run root is `NN-KNN/runs/` next to the worktrees, or `$NNKNN_RUNS`; it must exist (it is never created). `runs/<workstream>/<study>/<unit...>/` is one run (workstream `core`, `t1`, `t2` or `t3`). Every process session that writes to a unit adds one event file to that directory; weights stay in it: `weights/<name>.pt` (named states, written once), `weights/resume/last.pt` (the resume checkpoint), `weights/<name>.superseded-s<NNN>.pt` (a named state that a later session cut; renamed, never deleted).

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

- Exact channel: `x/<tag>` (float64/int64/bool), `rec/<name>` (canonical JSON; `log_records` appends rows), `arr/<name>` (`log_array`, `log_table`), `rec/state/<name>` (written by `save_state`), `rec/config`, and the session records `_run/begin|resume|checkpoint|complete|failed`. The display channel (float32 scalars without the `x/` prefix, a session card) is for TensorBoard only. A `torch.Tensor` value raises, so logging never adds a device sync or consumes RNG.
- Resume: `open_run` raises `UnitCompleteError` on a complete unit, `UnitBusyError` if the latest unfinished session's pid still runs on this host (a cheap guard, not a lock), `ConfigMismatchError` if the config changed (unless `resume="restart"`). It resumes when `last.pt` matches the last `_run/checkpoint` record, otherwise it restarts in place (`resume="restart"` or `NNKNN_RESUME=restart` forces that). A resume cuts every data record written after that checkpoint; a restart cuts all earlier data records. Log everything for step S before `save_resume(step=S)`, and use the checkpoint step axis for every tag of a resumable unit.
- `_run/complete` holds `result_digest`, a SHA-256 over the logical stream without `x/sys/*` and without weight-file bytes (`state_sha256` is included). On CPU an interrupted and resumed run has the same digest and state digests as an uninterrupted one (`tests/test_tracking.py` kills a real process to check this). XPU runs record the same digests, but are not promised to match.
- TensorBoard after a resume: the Python loader (`tensorboard --load_fast=false`) shows one continuous curve for re-logged tags and hides the stale tail of tags that are not logged again; the Rust loader ignores the SessionLog and keeps those stale points.
- Status (`run_status`): `complete`, `failed`, `incomplete` (crash; a cut last record reads as incomplete), `invalid` (damaged record, digest mismatch, missing or wrong-size weight file, ...) or `empty`. Readers use complete units only. CLI: `python -m model.common.tracking {ls|show|verify} <path>`; the reader side needs only `tensorboard` and `numpy`, so on accel: `uvx --with tensorboard==2.21.0 --with numpy python -m model.common.tracking ls ../runs`.

## Results and logs

Results and logs are not in git. They live on the experiment host and will move to the shared `NN-KNN/runs/` directory next to the worktrees, with TensorBoard logging. Every branch ignores `results/`, `logs/` and generated report files (figures, tables and manifests under `reports/`).

Until that move, the runners keep their current default output paths, and finished work stays organized on disk by the handoff that defines it:

- `main`: `results/prior_baseline/` (pre-MCB NN-kNN results; T1 Phase 0/6 references).
- `neural-cbr`: `results/t1/` and `logs/t1/` by T1 implementation phase.
- `rl`: `results/t2/` and `logs/t2/` by T2 implementation step and benchmark stage.
- `llm`: `results/t3/` and `logs/t3/` by T3 development stage.

The `results/README.md` kept with each branch's results maps handoff steps to directories, documents, and status. Run artifacts keep the paths they were written with; when a directory moves, the old and new locations are recorded instead of editing the artifacts.
