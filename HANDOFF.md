# LLM retrieval and host integration handoff

Branch `LLM` (`llm/`): shared NN-kNN core + T3. Current scope: `tools/t3_*.py`, `model/t3/`, and `tests/test_t3_*.py`.

## Where things are

- Specification: Xiaomeng Ye's `T3_LLM_AGENT_IMPLEMENTATION_PLAN.md` in `career-2027/03-research-architecture/implementation-handoff/` (copy and companions in `docs/t1_spec/`).
- Results and logs: filed by T3 handoff stage under `results/t3/<stage>/` and `logs/t3/<stage>/`; `results/README.md` maps every `docs/T3_*.md` to its stage and evidence.
- Shared core: synced from `main`; see `docs/NNKNN_CORE.md`.

## Status by handoff stage (2026-10-10)

- Host setup (section 4): a frozen open-weight host, a request/answer schema, and format/strict and request-budget controls are in place.
- Stage 1 (single-hop, prompt augmentation): SQuAD public-host comparisons (none / BM25 / fixed / learned / iterative), need-query training, ranking objectives, usefulness by removal, and provenance contracts.
- Model integration (section 4): one internal-host run on Stage 1 questions completed (64 trials, 2 errors, all recorded checks passed). `docs/T3_INTERNAL_INTERFACE.md` still describes it as queued.
- Stage 2 (iterative multi-hop): not exercised. The HotpotQA arm never issued a second retrieval, and supporting/joint F1 is 0.
- Stage 3 (biomedical) and Stage 4 (agent memory): not started.
- `docs/T3_TERM_FREQUENCY_ROBUSTNESS.md` does not yet report the completed binary/sublinear runs filed under `s1_single_hop_retrieval`.

## History

The original mixed handoff is preserved in the pre-reorganization Git bundle at `../recovery-20261009/before-repair.bundle`. See `docs/BRANCH_ORGANIZATION.md` and `docs/REORGANIZATION_MANIFEST.json` for the split and recovery boundary.
