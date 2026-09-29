# T1 plan implementation (branch `t1-plan-impl`)

This implements the PI's T1 specification (`docs/t1_spec/`, authority order in
`docs/t1_spec/T1_COMPLETE_HANDOFF_README.md`) on top of the maintained
`model/nnknn_model.py` core. It replaces the September 9-17 Gemini/Antigravity
T0/T1 modules, runners, results and reports, which were deleted on 2026-09-29.
Their reports had hard-coded conclusions that the data contradicted, and their
benchmarks never trained the retriever. They remain in git history on
`t0-neural-cbr` / `t1-neural-cbr`.

- Starting point: `t1-neural-cbr` @ `f3ac443`. The PI-inspected snapshot is `main` @ `c097195`.
- Host: g234 (Windows Server 2025, i5-13400F). Runs use the CPU; `NNKNN_DEVICE=cpu` and one torch thread per process.

## Module map

| File | Plan section | What it does |
|---|---|---|
| `model/nnknn_model.py` | 3.1, 4 | `retrieve()` exposes the retrieval step used by `forward` (same code path): leave-one-out exclusion, case masking with renormalization, and the excluded mask. It owns copies of `cases`/`labels` (bug fix, see below). |
| `model/t1/provenance.py` | 8, 9.1 | Stable-ID `CaseStatisticsStore` holding R, A, C, H, cf+, cf-, error support, protection and bias history. Also computes Q (smoothed), B (within-cohort mid-rank percentile; median/MAD ablation) and T (geometric in log space; arithmetic ablation). `audit_provenance` runs the pre-adaptation audit with LOO. The exact counterfactual removal delta has a closed form for softmax with or without the top-k mask; regression uses it as C/H. |
| `model/t1/retention.py` | 10 | One policy interface for every case role, with three stages: (1) protection (coverage floor, rare cohort, boundary), evidence marking, utility and redundancy in the learned geometry; (2) eviction-eligibility reason codes; (3) fill capacity with deterministic, logged tie-breaks. It covers the 10 required policies plus a `kcenter` selection baseline. |
| `model/t1/maintenance.py` | 3.4, contract 5 | `run_maintenance` scores cases, selects, archives evicted cases (exact restorable state), compacts the model and realigns the per-case optimizer state. It writes one maintenance event per case. `CaseArchive.restore` puts cases back. |
| `model/t1/geometry.py` | 7 (shared metric) | The single learned-distance interface: representation, glocal weights and distance exactly as retrieval uses them. Redundancy, coverage, bias initialization, L_near, the free radius and drift all use it. |
| `model/t1/core.py` | 3.1, 7, 13 | Builds and trains the retrieval core. Bias warm-start is the mean k-th (k=5) nearest non-self learned distance. Training uses LOO; early stopping sees only the validation stream. Maintenance hooks run at declared epoch checkpoints. MCB is optional (EMA memory encoder). Also contains pre-adaptation evaluation (acc/ECE or RMSE/MAE) and retrieval-event export. |
| `model/t1/reuse.py` | 5 | Classification reuse trains on LOO neighbourhoods with frozen retrieval. Input is `[Delta_z, (Delta_u), p0]` only. Output modes are nominal-residual and logit-residual; losses are diff-only, cls-only and combined. Pre/post evaluation reports the flip matrix, ECE and correction magnitude. |
| `model/t1/calibration.py` | 7 | Free-correction radius from the frozen trained-bias snapshot and the matching learned distance, on training pairs only with self excluded. Degenerate regions are reported. Also defines `L_small`. |
| `model/t1/sync.py` | 6, 13 Phase 4 | Computes L_pre, L_post, L_near, L_delta and L_small, logged separately. L_R updates retrieval only; L_A updates the adapter only. Schedules: `independent`, `alternating_rr` and `alternating_rrr` (with checkpoint maintenance and free-radius recalibration). |
| `model/t1/mcb.py` | 3.5, 13 Phase 5 | Matched by case_id: representation drift, neighbourhood churn and selection churn. |
| `model/t1/revise.py` | 3.3, 11 (T1.2) | The intervention log (relabel, bias, quarantine/release; quarantine is enforced as a retrieval mask). M0/M1/M2 evaluation reports targeted and collateral flips. The flagging ablation (provenance, bias, T, random, oracle at matched review budget) uses a simulated reviewer. Automatic consensus relabeling is intentionally absent. |
| `model/t1/data.py` | contract 11 | Train / (maintenance) / validation / test streams, with the scaler, target scaling and cohort bins fitted on train only. `synthetic_diag` has ground-truth corrupted, duplicate and rare cases and a shifted/rare test subgroup. |
| `model/t1/artifacts.py` | contract 2-6 | Run manifest (commit, dirty flag, environment, configuration id, components, budgets, maintenance, evaluation), plus case statistics, maintenance events and retrieval events. |
| `tools/t1_run.py` | - | YAML-driven runner for the types `legacy_reference`, `retention`, `reuse`, `sync`, `mcb` and `revise`. A `null`/`"..."` value aborts the run, so unresolved PI decisions cannot become hidden defaults. |
| `tools/t1_summarize.py` | - | Mean/sd/95% CI and seed-paired differences against a reference condition. Numbers only; no written conclusions. |
| `tests/test_t1_plan.py` | 15 | 23 tests covering the section 15 list and the section 5.5 reuse tests. |

## Plan coverage

| Plan item | Status |
|---|---|
| Phase 0: stable IDs, behaviour-neutral statistics, reproduce current behaviour | Done. Stable IDs predate this branch; the tests cover statistics-only invariance and RNG untouched. `legacy_reference` runs the maintained trainer on the same splits. |
| Phase 1: provenance, bias normalization, T, protection, redundancy, reversible selection, event logs; synthetic corruption/redundancy/rare/shift; regression counterfactual audit | Done |
| Phase 2: classification adapter behind a switch, LOO examples, frozen retrieval, single/combined losses, nominal/logit, pre/post, flips, calibration, leakage | Done. Explicit nominal `Delta_u` is plumbed, but none of the current datasets declares nominal fields; coverage is declared, not inferred. |
| Phase 3: RL actor/critic integration | **Not done.** The shared interface is role-agnostic, but actor helpful/harmful and critic audit targets are open PI/RL decisions (checklist T2). The previous Gemini RL hook was reverted with the rest. |
| Phase 4: synchronization with separate terms and three schedules | Done, for classification. A regression synchronization path is not implemented. |
| Phase 5: MCB on/off with drift/churn | Done, as the EMA memory encoder in the core. The IU-Bloomington Colab MCB code has not been obtained; this is a re-implementation of the mechanism described in `supporting/MCB_SYNTHESIS.md`. |
| Phase 6: legacy suites in pre-MCB form, modern transfer, revision/UI | Partial. Legacy classification/regression suites are configured. Image/text suites (CIFAR-10, SVHN, SST) and a TabArena subset are not wired. The revise harness exists; the UI study does not. |
| Force-include override (3.3) and feature-weight edits by a reviewer | Not implemented |
| Data-valuation baselines (influence functions, Data Shapley) | Not implemented. `kcenter` is the only extra selection baseline. |

## Bug fixed in the maintained core

`NN_KNN_Model.__init__` registered `cases.to(device)`, which on the same device
**aliases the caller's tensor**. `compact_cases` then reordered and zeroed the
caller's training data in place, so later training queries no longer matched
their labels. This showed up as a collapse to chance after maintenance. The
constructor now clones. The bug exists on `main` and `rl-iclr2027` since
2026-06-19 (`compact_cases`, commit 3abe9f0), and RL runs that compact case
memory may be affected. Regression test: `test_compaction_never_mutates_caller_training_data`.

## Running

```powershell
cd C:\Users\Administrator\NN-KNN\nnknn-work
$env:NNKNN_DEVICE="cpu"
.venv\Scripts\python.exe -m pytest tests/test_t1_plan.py -q
.venv\Scripts\python.exe tools\t1_run.py configs\t1\p1_retention_synthetic.yaml            # -> results/t1/<experiment>/...
.venv\Scripts\python.exe tools\t1_summarize.py results\t1\p1_retention_synthetic --metric test_accuracy_pre --reference random_K0.5
```

For long runs, launch detached with `tools\t1_launch.ps1`, which starts a WMI
process queue; ssh kills child processes.

Each run directory holds `run_manifest.json`, `metrics.json`, `history.json`,
`case_statistics.jsonl`, `case_maintenance.jsonl` and `retrieval_events.jsonl`,
plus `interventions.json` for revise runs. Each experiment also has a `runs.jsonl`.
