# LLM retrieval and host integration handoff

Branch `LLM` (`llm/`): shared NN-kNN core + T3. Current scope: `tools/t3_*.py`, `model/t3/`, and `tests/test_t3_*.py`.

## Where things are

- Specification: Xiaomeng Ye's `T3_LLM_AGENT_IMPLEMENTATION_PLAN.md` in `career-2027/03-research-architecture/implementation-handoff/` (copy and companions in `docs/t1_spec/`).
- Results and logs: not in git. Training data recorded with `--study` goes to the shared `NN-KNN/runs/t3/<study>/` as TensorBoard event files (see "2026-11-22: run tracking" below). Output folders are filed by T3 handoff stage under `results/t3/<stage>/` and `logs/t3/<stage>/` on the experiment host; `results/README.md` maps every `docs/T3_*.md` to its stage and evidence.
- Shared core: synced from `main`; see `docs/NNKNN_CORE.md`.

## Status by handoff stage (2026-10-10)

- Host setup (section 4): a frozen open-weight host, a request/answer schema, and format/strict and request-budget controls are in place.
- Stage 1 (single-hop, prompt augmentation): SQuAD public-host comparisons (none / BM25 / fixed / learned / iterative), need-query training, ranking objectives, usefulness by removal, and provenance contracts.
- Model integration (section 4): one internal-host run on Stage 1 questions completed (64 trials, 2 errors, all recorded checks passed); `docs/T3_INTERNAL_INTERFACE.md` was corrected on 2026-11-21 to say so.
- Stage 2 (iterative multi-hop): not exercised. The HotpotQA arm never issued a second retrieval, and supporting/joint F1 is 0.
- Stage 3 (biomedical) and Stage 4 (agent memory): not started.
- `docs/T3_TERM_FREQUENCY_ROBUSTNESS.md` does not report numbers for the completed binary/sublinear runs filed under `s1_single_hop_retrieval`; it now points to `results/README.md` for them.

## History

The original mixed handoff is preserved in the pre-reorganization Git bundle at `../recovery-20261009/before-repair.bundle`. See `docs/BRANCH_ORGANIZATION.md` and `docs/REORGANIZATION_MANIFEST.json` for the split and recovery boundary.

## 2026-11-21: repository layout cleanup and uv-only decision

- Removed: `README.md`, `requirements.txt`, `requirements-latest.txt`, `verify_packages.py`, `codex/README.md`, `codex/setup.sh`, `codex/maintenance.sh` (`codex/smoke_test.py` is kept), `tools/Sync-CodexHistory.ps1`, `tools/codex-history-sync.md`, `NNKNN_AGENT_GUIDE.md`, `.DS_Store`. Main keeps the long agent guide as `docs/NNKNN_AGENT_GUIDE.md`.
- Moved: `python_version.txt` -> `.python-version`.
- Owner decisions: this project needs no README, CHANGELOG or LICENSE. Environments and dependencies use uv only (`pyproject.toml`, `uv.lock`, `.python-version`; `uv sync`, `uv run ...`); no pip, requirements files or hand-made venv. The 2026-11-21 follow-up: only `main` has `pyproject.toml`/`uv.lock`/`.python-version` (Python 3.14, shared venv `NN-KNN/.venv`); this branch's own copies were deleted. See `main/HANDOFF.md` for versions and known gaps.
- Stage docs corrected: `docs/T3_INTERNAL_INTERFACE.md` (run completed) and `docs/T3_TERM_FREQUENCY_ROBUSTNESS.md` (pointer to completed runs).
- Left open: `docs/T3_ACTUAL_NEED_TRAINING.md` still cites `t3-train-need-{question-control,actual}-20261005/` (line 79, "Future outputs"). `results/README.md` says the runs that exist are the `-bounded-20261006` ones, but the run directories are an ignored local mirror and `results/t3/` is not in this copy, so the exact names were not confirmed and the citation was not changed.

## 2026-11-22: run tracking (TensorBoard)

Owner's rules: per-run training data goes to TensorBoard event files under `NN-KNN/runs/<workstream>/<study>/<unit>/` through `model/common/tracking.py`, with the run's weights in the unit; summaries keep their files and formats; recording is opt-in. In T3 only `tools/t3_need_matching.py` trains anything, so it is the one entry point that records:

- `--study S`: one unit per seed, `runs/t3/S/s<seed>/`. Unit config: tool, seed, epochs, dimensions, objective, selection, term frequency and the SHA-256 of the SQuAD train file, the sample file and the need-query file (no host paths). Recorded: `rec/protocol` (the protocol of the session that started the unit), per-epoch curves `fit/` and `tune/` `mean_need_loss`, `source_recall_at1`, `source_recall_at2` (step = epoch, 0 = before training) and `train/mean_loss`, `train/mean_gradient_norm` (epochs 1..E), every training step as the row record `training_events` (the content of the old `training_events.json`), the seed summary as finals (`seed`, `selected_epoch`, `selected/<group>/<metric>`, `bm25/<group>/<metric>`, `actual_training_steps`, `nonzero_gradient_steps`, `parameter_delta_l2`, `trainable_parameters`, `training_seconds`), and the named state `metric` (`weights/metric.pt`: the same dict as the old `metric.pt`, with selected, final and initial metric and Adam states).
- With `--study`, `s<seed>/metric.pt`, `s<seed>/training_events.json` and `s<seed>/summary.json` are no longer written. `protocol.json` (plus `"study": "S"`) and `summary.json` still go to `--output`, with the same content; `summary.json` is rebuilt from the units (`seed_result`), and `checkpoint_sha256` is then the SHA-256 of the unit's `weights/metric.pt`. Rerunning a study skips complete seeds (their recorded summary fills `summary.json`), raises `ConfigMismatchError` if a setting changed, and resumes an interrupted seed after its last checkpointed epoch.
- Without `--study` the tool writes exactly the files it wrote before.
- Readers: `model/t3/metric_provenance.py` (`metric_provenance`, `metric_checkpoint_sha256`, `load_metric_checkpoint`) accepts a legacy `s<seed>/metric.pt` (bound through the files next to it, as before) or a run unit (verified weight file, unit records). `tools/t3_public_host_pilot.py` and `tools/t3_internal_host_pilot.py` take either as `--metric-checkpoint`.
- Not recorded: the host pilots (`t3_public_host_pilot`, `t3_internal_host_pilot`, `t3_public_set_utility`, `t3_prompt_pilot`), the host need collection `t3_train_need_queries`, and `t3_host_smoke`, `t3_public_data_audit`, `t3_prepare_host`. They train no parameter (`trainable_host_parameters=0`); their output folders are audited evaluation evidence that later tools read by byte hash (`--input-run`, `--source-run`, `--reference-run`), so they keep writing them.
- Checked on g234 CPU (scratch copies): with the same seeds the old and new tool write byte-identical `s<seed>/` files; a `--study` run records the same seed summaries (except wall time), training events and metric tensors; a rerun of a complete study writes byte-identical `summary.json` and `protocol.json`; the full default study (seeds 8/9/10, 20 epochs), crashed in seed 9 and resumed, reproduces the 2026-10-05 original run exactly. Tests: `tests/test_t3_need_matching_runs.py`.
