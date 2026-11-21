# LLM retrieval and host integration handoff

Branch `LLM` (`llm/`): shared NN-kNN core + T3. Current scope: `tools/t3_*.py`, `model/t3/`, and `tests/test_t3_*.py`.

## Where things are

- Specification: Xiaomeng Ye's `T3_LLM_AGENT_IMPLEMENTATION_PLAN.md` in `career-2027/03-research-architecture/implementation-handoff/` (copy and companions in `docs/t1_spec/`).
- Results and logs: filed by T3 handoff stage under `results/t3/<stage>/` and `logs/t3/<stage>/`; `results/README.md` maps every `docs/T3_*.md` to its stage and evidence.
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
