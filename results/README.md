# LLM (T3) results, filed by handoff stage

Organized 2026-10-10 by Xiaomeng Ye's T3 handoff, `career-2027/03-research-architecture/implementation-handoff/T3_LLM_AGENT_IMPLEMENTATION_PLAN.md` (copy in `docs/t1_spec/`): §14 development stages and the §4 integration modes. `results/t3/<stage>/` and `logs/t3/<stage>/` use the same folders.

```
results/t3/
  s0_host_setup/             §4 host strategy: frozen-host smoke, request/answer schema,
                             format/strict decoding and request-budget controls
  s1_single_hop_retrieval/   §14 Stage 1 with prompt augmentation (§4): prompt pilots, public
                             SQuAD/HotpotQA foundation, need matching, ranking, actual-need
                             training, public host comparisons, usefulness, provenance contracts
  s1_model_integration/      §4 model integration on Stage 1 questions: internal host
```

Stages without results: Stage 2 (iterative multi-hop), Stage 3 (biomedical transfer), and Stage 4 (existing-agent memory). The HotpotQA arm runs inside the Stage 1 public-host runs, but no second retrieval occurred and Hotpot support/joint F1 is 0, so Stage 2 has not been exercised.

## Stages, documents and evidence

| Stage | Document (`../docs/`) | Evidence (`<stage>/codex_task_20261002/`) |
| --- | --- | --- |
| s0 | `T3_HOST_ENGINEERING_PILOT.md` | `outputs/neural-cbr-t3-host-smoke-20261004.json`; host logs |
| s0 | `T3_HOST_SCHEMA.md`, `T3_FORMAT_HOST_RESULTS.md`, `T3_STRICT_FORMAT_CONTROL.md` | `runs/t3-public-host-20261005-format-enforced`, `-strict-plain`, `runs/t3-format-prefix-31`; format/strict verification and comparison files |
| s0 | `T3_REQUEST_LOOP_DIAGNOSIS.md`, `T3_REQUEST_BUDGET_REPAIR.md` | `runs/t3-bounded-request-diagnostic-20261006`; request-loop diagnosis and bounded-probe verification; uniform/bounded request drivers |
| s1 | `T3_RETRIEVAL_CONTRACT.md`, `T3_VERSION_BINDING.md`, `T3_HOST_FAILURE_JOURNAL.md`, `T3_METRIC_PROVENANCE.md` | `runs/t3-version-binding-host-20261005`; version, failure-journal and metric-provenance verification files |
| s1 | `T3_PROMPT_ENGINEERING_PILOT.md` | `runs/t3-prompt-pilot-20261004-*` (first, explicit-need, staged-schema, direct-removal) |
| s1 | `T3_PUBLIC_BENCHMARK_FOUNDATION.md` | `runs/t3-public-data-audit-20261005-preserved-gaps`; public-conversion verification; Hotpot annotation gaps |
| s1 | `T3_NEED_MATCHING.md`, `T3_RANKING_OBJECTIVE.md`, `T3_NEED_SHIFT_DIAGNOSIS.md` | `runs/t3-need-matching-20261005*`, `t3-hard-negative-20261005*`, `t3-softmax-recall-20261005*` |
| s1 | `T3_ACTUAL_NEED_TRAINING.md`, `T3_ACTUAL_NEED_TRAINING_RESULTS.md` | `runs/t3-train-need-queries-*`, `t3-train-need-*-bounded-20261006*`, `t3-need-query-default-compat-20261005` |
| s1 | `T3_TERM_FREQUENCY_ROBUSTNESS.md` | `runs/t3-need-binary-20261006*`, `t3-need-sublinear-20261006*`, `t3-term-frequency-default-compat-20261006`; the results exist but the document does not yet report them |
| s1 | `T3_PUBLIC_HOST_RESULTS.md`, `T3_ACTUAL_NEED_PUBLIC_RESULTS.md` | `runs/t3-public-host-20261005`, `-request-v2`, `t3-public-host-need-*-bounded-20261006` |
| s1 | `T3_DIRECT_SET_UTILITY.md`, `T3_PUBLIC_UTILITY_RESULTS.md` | `runs/t3-public-set-utility-20261005`, `t3-utility-prefix-17` |
| s1 model integration | `T3_INTERNAL_INTERFACE.md` | `runs/t3-internal-host-20261006`; internal preflight/host verification. The document still says the run is queued; the run completed (64 trials, 2 errors, all recorded checks passed) |

Two citations are dangling at the source and were left as written: `T3_ACTUAL_NEED_TRAINING.md` cites `t3-train-need-{question-control,actual}-20261005/`; the runs that exist are the `-bounded-20261006` ones.

## Codex task evidence

All T3 runs were produced in the Codex task workspace `C:\Users\Administrator\Documents\Codex\2026-10-02\neural-cbr-xiaomeng-document-https-github\` (`outputs/` and `work/`). On 2026-10-10 they were copied unchanged, SHA-256 verified, into each stage's `codex_task_20261002/`:

- `outputs/`: `neural-cbr-t3-*` verification and results files (tracked, byte-exact).
- `audit_scripts/`, `drivers/`: the audit, replay, report and driver scripts (tracked).
- `provenance/`: freeze records, run plans, and pre-run `git status` snapshots (tracked).
- `runs/`: the run directories (313 MB; ignored, local only).
- logs: `../logs/t3/<stage>/codex_task_20261002/` (tracked).

Not copied: the host model weights (`work/t3-models`, 1.5 GB), host runtime and libraries, the public benchmark mirror (`work/t3-public-data-20261005`, 104 MB), the per-run source archives and zips, and pytest scratch folders. They remain in the Codex workspace. Verification files record absolute Codex-workspace paths and hashes, and the tools that take `--reference-run`, `--input-run` or `--source-run` check those recorded paths, so replays must point at the original location.

## Other results

The pre-MCB supervised results that used to sit in `results/` are maintained on `main` under `results/prior_baseline/`.
