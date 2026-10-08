# Neural CBR Progress Report

## 0. Overview

| Area | Main changes | Key results |
|---|---|---|
| T0/T1 (superseded) | Classical CBM baselines (DROP3/ICF/CoreSet), revise stage, Covertype, T1 core, 5-schedule trainer | Removed by `17cbae6`; kept as record only |
| T1 rebuild to PI plan | New `model/t1/` package (provenance, retention, maintenance, reuse, sync, MCB, revise harness); fixed a case-tensor aliasing bug and a fresh-Adam collapse | Pilot runs (p0–p5, t12) are historical |
| T1 alignment with the revised PI spec | Q/B kept separate, outcome-based C/H, final-loss removal influence, stable IDs; several checkpoint/optimizer bugs fixed; nominal reuse groups; reviewer UI; synthetic stress grid | Mostly mixed or negative: removal gives no consistent gain; RRR sync helps regression but hurts Iris; maintenance-split selection improves 18/18 pairs (overfitting risk) |
| T2 RL | Critic ID/restore/Adam alignment, role audits, quality ledger, offline/live retention, Acrobot | Stage A not passed; Acrobot actor never starts; retention hurts the actor; CartPole gains are unmatched |
| T3 LLM agent | Frozen Qwen3-0.6B host, artifact-bound retrieval, public SQuAD/Hotpot, constrained decoder, set utility, internal output interface | Learned need matching < BM25; no benefit on public QA; repetition loops block need training; internal 64-call experiment not run yet |

**Next steps (from `docs/T1_EXECUTION_LEDGER.md` / `HANDOFF.md`):**
1. **T3:** finish the bounded 192-query collection, then the six matched fits plus the public arms, then the real 64-answer internal experiment.
2. **T2:** compare label mode, matching, and calibration; fix Acrobot readiness and negative-reward startup; confirm matched capacity and cost; then the Stage A gate.
3. **T1:** broader matched-compute and scale grids with frozen confirmation; real-data nominal reuse; a human study with real participants; benchmark reconstruction (TabArena, external MCB/IU source).


## 1. T0/T1 initial work

Scope: commits `052699c` through `6adeb30` on `neural-cbr`. The period has two phases.

- **Phase A (Antigravity era; `052699c`, `cc3f67a`, `c7467b6`, plus the results committed in `f3ac443`).** **SUPERSEDED/REMOVED by `17cbae6`**, which deleted this code and these results because the reports "contained hard-coded conclusions contradicted by the data" and "the retriever was never trained in their benchmarks". The tables survive only in git history (`git show <sha>:<path>`). They are listed in §1.3.10 for the record only and are not progress.
- **Phase B (`17cbae6` to `6adeb30`).** The PI T1 specification (`docs/t1_spec/`) was re-implemented as `model/t1/`, and exploratory pilots were run under `results/t1/`. No document reports results for these pilots: `tools/t1_summarize.py` writes numbers only. Unless a table names an untracked `summary.csv` or `paired_vs_*.csv` as its source, it was computed for this report from the versioned `results/t1/<exp>/runs__*.jsonl` shards (last run per dataset, condition and seed; mean ± sample sd). **`docs/T1_EXPERIMENT_PLAN.md` later declared `results/t1/` historical.** The follow-up continuation fixed Adam moment aliasing when optimizers are forked; that document says *"Earlier post-selection comparisons must not be used as evidence of independent candidate continuations."* The full revised grids have not been rerun.

### 1.1 Summary

- **The -era T0/T1 work is superseded and is not evidence.** `17cbae6` removed it. Even taken at face value, its tables contradict its own stated conclusions. In the -era main benchmark, CoreSet has the best mean on 4 of 6 datasets, and "Ours (Retain)" is below ICF or CoreSet on 5 of 6. Revise recovers at most +0.0175, not "+5.3%". The top-1 confidence drop is 0.0089–0.0428, not "0.20–0.35".
- **Two core bugs were found and fixed in Phase B.** (1) `NN_KNN_Model.__init__` aliased the caller's case tensor, so `compact_cases` reordered and zeroed the caller's training data. This caused collapse to chance after maintenance and has existed on main and `rl-iclr2027` (now `rl`) since `3abe9f0`. (2) A fresh Adam optimizer after core training collapsed digits retrieval from 0.98 to 0.10 (`ec6323a`). Pilots run with the fresh optimizer are kept only as invalid output.
- **Post-maintenance fine-tuning is harmful (negative result).** In P1-C, fine-tuning lowered accuracy by more than 0.1 in 230 of 1,400 non-full-memory runs (16.4%). In P1-S the figure is 27 of 200 (13.5%). On digits at K ≥ 0.33, every policy collapses to about 0.10, except bias_current at K=0.33 (0.28 ± 0.39). `6adeb30` added the fix (keep the pre-fine-tune state as a candidate; lr scale 0.1), but **it has not been rerun**.
- **The Phase B pilots give mixed or negative signals, and all are historical.** With no fine-tune, retention policies are mostly close to full memory or random. The reuse adapter helps clearly only on balance (+0.035 to +0.050), and nominal modes inflate ECE to as much as 0.7555. MCB lowers representation drift in every seed, but costs accuracy on breast_cancer. For flagging corrupted cases in T1.2, provenance_only is the only non-oracle signal well above random; bias_only and combined_T fall below random.
- **The legacy core is weak on zebra.** P0 zebra is 0.4818 ± 0.2991 and zebra_special is 0.5318 ± 0.0874, near chance.

### 1.2 Changes

| commit | change |
|---|---|
| `052699c` | **[SUPERSEDED]** : DROP3/ICF/CoreSet baselines in `model/t0_maintenance.py`; `model/t0_revise.py` (kNN-consensus relabeler, synthetic noise); `tools/run_iclr_experiments.py` (+371/−149), with 6 suites: main @50%, revise @15% noise, top-1 faithfulness, capacity frontier, retention ablations, reuse modes. Covertype is a 3,000-row subsample, and its scaler is fit before the split. Output in `results/iclr_paper/*` (seeds 42–46). |
| `cc3f67a` | **[SUPERSEDED]**  T1 core: `model/t1_maintenance.py`, `t1_mcb.py` (EMA m=0.999), `t1_revise.py`, `t1_synchronization.py`, `t1_workflow.py`; MCB hooks in `nnknn_model.py`; 13 tests; T1 benchmark runner; `docs/T1_IMPLEMENTATION_HANDOFF.md`. |
| `c7467b6` | **[SUPERSEDED]**  `model/t1_scheduler.py` (`FullPipelineTrainer`, 5 schedules) and `tools/run_schedule_exploration.py`. 30 jobs (iris/wine/synthetic, seeds 42/43, 30 epochs) written to `results/t1_schedules/`. HANDOFF claimed staged_sequential is "Pareto-optimal". |
| `f3ac443` | **[SUPERSEDED]** T0 retention hooks for regression and RL (`nnknn_rl_workflow.py` +243); `run_t0_regression.py`, `run_t0_rl.py`. Results in `results/t0_regression`, `results/t0_rl` (smoke-sized) and `results/iclr_test_run` (N=1). The commit message itself flags that the RL report reads the wrong keys. |
| `17cbae6` | PI T1 plan implemented as `model/t1/` (`provenance`, `retention`, `maintenance`, `core`, `reuse`, `calibration`, `sync`, `mcb`, `revise`, `data` with `synthetic_diag`, `artifacts`, `geometry`); `tools/t1_run.py` (aborts on `null` PI decisions), `t1_summarize.py`, `t1_launch.ps1`, `configs/t1/*.yaml` (PILOT values), 23 tests; `docs/T1_PLAN_IMPLEMENTATION.md`, `docs/T1_EXPERIMENT_PLAN.md`; PI package moved in-repo as `docs/t1_spec/**`. **Fix: compaction no longer mutates the caller's training data** (regression test added). **Removes all  T0/T1 code and results**, and reverts `nnknn_rl_workflow.py` to main. |
| `19e8026` | Deleted the empty `configs/t1/_common.yaml`. |
| `aaafed7` | `t1_launch.ps1`: splits comma-separated `-Configs`. |
| `6e5f3b7` | P0 config: explicit `batch_size` (fixes a KeyError); launcher gets a `-Datasets` filter; small runner fix. |
| `23a483b` | P0 config declares `task_type` for the legacy regression reconstruction. |
| `ec6323a` | **Post-core phases (sync independent, M2, post-maintenance fine-tune) continue the core optimizer via `clone_optimizer`**, realigned after compaction. With a fresh Adam, digits retrieval collapsed from 0.98 to 0.10. P1/P4/T1.2 were rerun. Fresh-Adam output kept in `results/t1_invalid_fresh_adam/` (gitignored). |
| `6ce766e` | Pilot shards versioned as `results/t1/<exp>/runs__<dataset>.jsonl`; per-run dirs (1.7 GB) stay on g234. Branches renamed to `neural-cbr` and `rl`. |
| `6adeb30` | Fine-tune keeps the pre-fine-tune state as a selection candidate (`include_initial=True`); explicit lr scale (PILOT 0.1) for fine-tune, sync and M2. Motivation: fine-tune hurt accuracy by >0.1 in 15–26% of runs (e.g. balance 0.84→0.55). **Not rerun yet.** |

**Fix coverage**, inferred from file modification order against the commits:

| pilot | relative to `ec6323a` / `6adeb30` |
|---|---|
| P2, P5 | before `ec6323a` |
| P0 | classification before `ec6323a`. P0 uses the legacy `train_model`, so the fix probably does not affect it |
| P1, P4, T1.2 | after `ec6323a`, before `6adeb30` |

None of the `results/t1` pilots include the `6adeb30` fix.

### 1.3 Experiments and results

**Common protocol for Phase B pilots.** Train/val/test 60/20/20. One core per (dataset, seed). Seeds 0–4, except P1-S and T1.2, which use seeds 0–9 (`synthetic_diag` uses 5 seeds in P2/P4/P5). The core trains for ≤300 epochs with patience 40, followed by a 30-epoch post-selection fine-tune. PILOT values in `configs/t1/_base.yaml`: s=1.0, α=0.5, evidence R≥2 and A≥0.1, coverage floor 2 per cohort, weights trust 1.0 and utility/redundancy/coverage 0.5. All Phase B results are exploratory pilots and **historical under the later PI alignment**.

#### 1.3.1 P0: legacy reference
*Tests:* the maintained legacy `train_model` on the T1 splits, as Gate 1 context. *Setup:* 12 datasets, 5 seeds. Source: `results/t1/p0_legacy_reference/runs__*.jsonl` (computed for this report).

| dataset | task | n | acc | ECE | RMSE | MAE | n_cases |
|---|---|---|---|---|---|---|---|
| abalone | reg | 5 | - | - | 0.7011 ± 0.0446 | 0.4893 ± 0.0172 | 1500 |
| airfoil | reg | 5 | - | - | 0.4313 ± 0.0293 | 0.3266 ± 0.0193 | 901 |
| balance | clf | 5 | 0.8416 ± 0.0199 | 0.0476 ± 0.0057 | - | - | 375 |
| breast_cancer | clf | 5 | 0.9702 ± 0.0100 | 0.0373 ± 0.0155 | - | - | 341 |
| digits | clf | 5 | 0.9750 ± 0.0071 | 0.0233 ± 0.0043 | - | - | 1077 |
| energy_efficiency | reg | 5 | - | - | 0.2000 ± 0.0223 | 0.1262 ± 0.0100 | 460 |
| iris | clf | 5 | 0.9600 ± 0.0435 | 0.0480 ± 0.0156 | - | - | 90 |
| student_performance | reg | 5 | - | - | 0.4087 ± 0.0998 | 0.2657 ± 0.0342 | 389 |
| wine | clf | 5 | 0.9722 ± 0.0340 | 0.0389 ± 0.0160 | - | - | 106 |
| yacht | reg | 5 | - | - | 0.3290 ± 0.0560 | 0.2136 ± 0.0185 | 184 |
| zebra | clf | 5 | 0.4818 ± 0.2991 | 0.3112 ± 0.1628 | - | - | 66 |
| zebra_special | clf | 5 | 0.5318 ± 0.0874 | 0.2535 ± 0.0809 | - | - | 132 |

*Caveat:* zebra and zebra_special are near chance, which matches the HANDOFF note that Zebra (a)/(b) are the immediate debugging priority. Regression errors are presumably on the standardized target scale; the source does not say.

#### 1.3.2 P1-C: retention on classification (Gate 2)
*Tests:* whether any retention policy beats random or current bias pruning at matched K, and how close it gets to full memory. *Setup:* 7 datasets × 11 policies × K ∈ {0.2, 0.33, 0.5, 0.75}, 5 seeds; continued optimizer (after `ec6323a`). Source: `results/t1/p1_retention_classification/runs__*.jsonl` (computed for this report).

**After the 30-epoch fine-tune** (`test_accuracy_pre`):

K=0.2

| dataset | full_memory | random | stratified | bias_current | bias_normalized | provenance_only | trust | trust_utility | trust_utility_coverage | utility_redundancy | kcenter |
|---|---|---|---|---|---|---|---|---|---|---|---|
| balance | 0.8656 ± 0.0131 | 0.8640 ± 0.0315 | 0.7696 ± 0.1699 | 0.8608 ± 0.0257 | 0.8672 ± 0.0184 | 0.8880 ± 0.0188 | 0.8752 ± 0.0184 | 0.8768 ± 0.0184 | 0.8752 ± 0.0121 | 0.8688 ± 0.0269 | 0.8000 ± 0.1749 |
| breast_cancer | 0.9649 ± 0.0062 | 0.9649 ± 0.0124 | 0.8386 ± 0.2631 | 0.9632 ± 0.0073 | 0.9667 ± 0.0096 | 0.8123 ± 0.2498 | 0.9561 ± 0.0107 | 0.9561 ± 0.0164 | 0.9614 ± 0.0118 | 0.8404 ± 0.2548 | 0.9088 ± 0.1354 |
| digits | 0.9811 ± 0.0093 | 0.9672 ± 0.0041 | 0.9667 ± 0.0090 | 0.9750 ± 0.0071 | 0.9733 ± 0.0087 | 0.9750 ± 0.0071 | 0.9744 ± 0.0095 | 0.9756 ± 0.0084 | 0.9772 ± 0.0069 | 0.9789 ± 0.0070 | 0.9778 ± 0.0121 |
| iris | 0.9600 ± 0.0435 | 0.9533 ± 0.0447 | 0.9733 ± 0.0435 | 0.9800 ± 0.0298 | 0.9667 ± 0.0408 | 0.9733 ± 0.0279 | 0.9600 ± 0.0435 | 0.9467 ± 0.0380 | 0.9667 ± 0.0236 | 0.9667 ± 0.0236 | 0.9733 ± 0.0435 |
| wine | 0.9722 ± 0.0393 | 0.9778 ± 0.0124 | 0.9778 ± 0.0232 | 0.9833 ± 0.0248 | 0.9889 ± 0.0152 | 0.9944 ± 0.0124 | 0.9722 ± 0.0278 | 0.9944 ± 0.0124 | 0.9778 ± 0.0232 | 0.9889 ± 0.0248 | 0.9889 ± 0.0152 |
| zebra | 0.9000 ± 0.2236 | 0.6636 ± 0.0943 | 0.6182 ± 0.0826 | 0.6273 ± 0.1261 | 0.5909 ± 0.1639 | 0.6182 ± 0.0943 | 0.6636 ± 0.0761 | 0.6182 ± 0.0689 | 0.6727 ± 0.1416 | 0.6727 ± 0.0380 | 0.6364 ± 0.1066 |
| zebra_special | 0.6955 ± 0.0903 | 0.6000 ± 0.0781 | 0.5955 ± 0.0296 | 0.5455 ± 0.0455 | 0.5773 ± 0.0674 | 0.5136 ± 0.1024 | 0.5864 ± 0.0670 | 0.5591 ± 0.0945 | 0.6455 ± 0.0903 | 0.5864 ± 0.0826 | 0.6318 ± 0.0856 |

K=0.33

| dataset | full_memory | random | stratified | bias_current | bias_normalized | provenance_only | trust | trust_utility | trust_utility_coverage | utility_redundancy | kcenter |
|---|---|---|---|---|---|---|---|---|---|---|---|
| balance | 0.8592 ± 0.0230 | 0.7408 ± 0.1583 | 0.8048 ± 0.0948 | 0.8672 ± 0.0166 | 0.8432 ± 0.0378 | 0.8672 ± 0.0230 | 0.8496 ± 0.0327 | 0.8752 ± 0.0175 | 0.8720 ± 0.0160 | 0.8640 ± 0.0240 | 0.7760 ± 0.1802 |
| breast_cancer | 0.9649 ± 0.0062 | 0.8684 ± 0.1236 | 0.7667 ± 0.2609 | 0.9561 ± 0.0107 | 0.9509 ± 0.0078 | 0.7474 ± 0.2877 | 0.9439 ± 0.0282 | 0.9544 ± 0.0243 | 0.9105 ± 0.0583 | 0.7965 ± 0.2312 | 0.8807 ± 0.1322 |
| digits | 0.9844 ± 0.0075 | 0.0989 ± 0.0015 | 0.1006 ± 0.0023 | 0.2800 ± 0.3871 | 0.0994 ± 0.0012 | 0.1006 ± 0.0023 | 0.0989 ± 0.0015 | 0.0989 ± 0.0015 | 0.1000 ± 0.0020 | 0.1000 ± 0.0000 | 0.0994 ± 0.0012 |
| iris | 0.9600 ± 0.0435 | 0.9467 ± 0.0380 | 0.9667 ± 0.0408 | 0.9667 ± 0.0408 | 0.9667 ± 0.0408 | 0.9667 ± 0.0408 | 0.9600 ± 0.0435 | 0.9600 ± 0.0365 | 0.9467 ± 0.0506 | 0.9600 ± 0.0365 | 0.9600 ± 0.0365 |
| wine | 0.9722 ± 0.0393 | 1.0000 ± 0.0000 | 0.9778 ± 0.0124 | 0.9833 ± 0.0152 | 1.0000 ± 0.0000 | 0.9889 ± 0.0152 | 1.0000 ± 0.0000 | 0.9778 ± 0.0232 | 0.9778 ± 0.0232 | 0.9944 ± 0.0124 | 0.9889 ± 0.0152 |
| zebra | 0.8909 ± 0.2439 | 0.6182 ± 0.0518 | 0.6545 ± 0.1095 | 0.6182 ± 0.0943 | 0.6455 ± 0.1220 | 0.6273 ± 0.0985 | 0.5545 ± 0.0380 | 0.6636 ± 0.1348 | 0.6364 ± 0.1607 | 0.6636 ± 0.1626 | 0.6182 ± 0.1185 |
| zebra_special | 0.6818 ± 0.0937 | 0.6727 ± 0.0523 | 0.6955 ± 0.0693 | 0.6182 ± 0.0983 | 0.6545 ± 0.0337 | 0.6182 ± 0.1130 | 0.6000 ± 0.0998 | 0.6773 ± 0.0708 | 0.6727 ± 0.0635 | 0.6636 ± 0.0901 | 0.6955 ± 0.0859 |

K=0.5

| dataset | full_memory | random | stratified | bias_current | bias_normalized | provenance_only | trust | trust_utility | trust_utility_coverage | utility_redundancy | kcenter |
|---|---|---|---|---|---|---|---|---|---|---|---|
| balance | 0.8608 ± 0.0201 | 0.8448 ± 0.0402 | 0.7552 ± 0.1660 | 0.8608 ± 0.0201 | 0.8592 ± 0.0184 | 0.8704 ± 0.0119 | 0.7920 ± 0.1839 | 0.8592 ± 0.0145 | 0.8608 ± 0.0107 | 0.8624 ± 0.0222 | 0.7648 ± 0.1355 |
| breast_cancer | 0.9649 ± 0.0000 | 0.8298 ± 0.1739 | 0.7754 ± 0.2498 | 0.9544 ± 0.0190 | 0.9491 ± 0.0190 | 0.8509 ± 0.1478 | 0.9561 ± 0.0088 | 0.8737 ± 0.1371 | 0.9298 ± 0.0443 | 0.8509 ± 0.1444 | 0.9070 ± 0.0646 |
| digits | 0.9833 ± 0.0039 | 0.1017 ± 0.0015 | 0.1006 ± 0.0012 | 0.1006 ± 0.0012 | 0.1000 ± 0.0020 | 0.1028 ± 0.0079 | 0.0994 ± 0.0012 | 0.0994 ± 0.0012 | 0.1000 ± 0.0028 | 0.1022 ± 0.0036 | 0.0994 ± 0.0012 |
| iris | 0.9600 ± 0.0435 | 0.9533 ± 0.0380 | 0.9600 ± 0.0548 | 0.9600 ± 0.0435 | 0.9667 ± 0.0408 | 0.9600 ± 0.0435 | 0.9600 ± 0.0435 | 0.9533 ± 0.0558 | 0.9533 ± 0.0380 | 0.9600 ± 0.0435 | 0.9667 ± 0.0408 |
| wine | 0.9722 ± 0.0393 | 0.9944 ± 0.0124 | 0.9833 ± 0.0248 | 0.9944 ± 0.0124 | 0.9944 ± 0.0124 | 0.9833 ± 0.0248 | 0.9833 ± 0.0248 | 0.9833 ± 0.0248 | 0.9944 ± 0.0124 | 0.9778 ± 0.0304 | 1.0000 ± 0.0000 |
| zebra | 0.8909 ± 0.2439 | 0.7364 ± 0.2456 | 0.6455 ± 0.1521 | 0.7273 ± 0.1731 | 0.7091 ± 0.1778 | 0.5818 ± 0.1177 | 0.7455 ± 0.1423 | 0.7818 ± 0.2093 | 0.8000 ± 0.2170 | 0.7909 ± 0.1918 | 0.8091 ± 0.1452 |
| zebra_special | 0.6909 ± 0.0998 | 0.6864 ± 0.0588 | 0.6545 ± 0.1009 | 0.6864 ± 0.0810 | 0.6909 ± 0.0380 | 0.6455 ± 0.0614 | 0.6955 ± 0.0797 | 0.6773 ± 0.0743 | 0.7091 ± 0.0810 | 0.6909 ± 0.0844 | 0.7318 ± 0.0689 |

K=0.75

| dataset | full_memory | random | stratified | bias_current | bias_normalized | provenance_only | trust | trust_utility | trust_utility_coverage | utility_redundancy | kcenter |
|---|---|---|---|---|---|---|---|---|---|---|---|
| balance | 0.8544 ± 0.0199 | 0.8368 ± 0.0544 | 0.8176 ± 0.0608 | 0.7952 ± 0.0639 | 0.8208 ± 0.0505 | 0.8720 ± 0.0219 | 0.8528 ± 0.0121 | 0.8160 ± 0.0512 | 0.8336 ± 0.0694 | 0.8240 ± 0.0453 | 0.8000 ± 0.0627 |
| breast_cancer | 0.9614 ± 0.0048 | 0.7930 ± 0.1500 | 0.8316 ± 0.2429 | 0.9614 ± 0.0078 | 0.9404 ± 0.0457 | 0.8298 ± 0.1659 | 0.9439 ± 0.0428 | 0.8175 ± 0.1919 | 0.8281 ± 0.2261 | 0.8404 ± 0.1806 | 0.8211 ± 0.1787 |
| digits | 0.9817 ± 0.0101 | 0.1011 ± 0.0015 | 0.1011 ± 0.0015 | 0.1000 ± 0.0020 | 0.0994 ± 0.0023 | 0.1006 ± 0.0012 | 0.1006 ± 0.0023 | 0.1017 ± 0.0015 | 0.1006 ± 0.0023 | 0.1006 ± 0.0023 | 0.1000 ± 0.0020 |
| iris | 0.9600 ± 0.0435 | 0.9533 ± 0.0558 | 0.9600 ± 0.0435 | 0.9667 ± 0.0408 | 0.9600 ± 0.0435 | 0.9600 ± 0.0435 | 0.9600 ± 0.0435 | 0.9667 ± 0.0408 | 0.9667 ± 0.0408 | 0.9533 ± 0.0506 | 0.9600 ± 0.0365 |
| wine | 0.9778 ± 0.0304 | 0.9833 ± 0.0248 | 0.9944 ± 0.0124 | 0.9889 ± 0.0248 | 0.9889 ± 0.0152 | 0.9833 ± 0.0152 | 1.0000 ± 0.0000 | 0.9833 ± 0.0248 | 0.9944 ± 0.0124 | 0.9944 ± 0.0124 | 0.9944 ± 0.0124 |
| zebra | 0.8909 ± 0.2439 | 0.8364 ± 0.2241 | 0.8182 ± 0.2670 | 0.8091 ± 0.2658 | 0.8727 ± 0.2371 | 0.8000 ± 0.2418 | 0.8091 ± 0.2093 | 0.8818 ± 0.2643 | 0.8818 ± 0.2643 | 0.8909 ± 0.2439 | 0.9000 ± 0.2236 |
| zebra_special | 0.6909 ± 0.0998 | 0.6727 ± 0.0693 | 0.6955 ± 0.0593 | 0.7136 ± 0.0380 | 0.6727 ± 0.1143 | 0.6818 ± 0.0682 | 0.7000 ± 0.0841 | 0.7000 ± 0.0651 | 0.7227 ± 0.0407 | 0.7045 ± 0.0425 | 0.6955 ± 0.0829 |

**Without fine-tune** (`test_nofinetune_accuracy_pre`: selection only):

K=0.2

| dataset | full_memory | random | stratified | bias_current | bias_normalized | provenance_only | trust | trust_utility | trust_utility_coverage | utility_redundancy | kcenter |
|---|---|---|---|---|---|---|---|---|---|---|---|
| balance | 0.8496 ± 0.0236 | 0.8336 ± 0.0322 | 0.8304 ± 0.0143 | 0.8368 ± 0.0472 | 0.8448 ± 0.0351 | 0.8672 ± 0.0230 | 0.8624 ± 0.0222 | 0.8496 ± 0.0243 | 0.8528 ± 0.0286 | 0.8528 ± 0.0145 | 0.8352 ± 0.0184 |
| breast_cancer | 0.9649 ± 0.0062 | 0.9614 ± 0.0171 | 0.9439 ± 0.0275 | 0.9491 ± 0.0273 | 0.9491 ± 0.0319 | 0.9544 ± 0.0144 | 0.9632 ± 0.0073 | 0.9614 ± 0.0133 | 0.9702 ± 0.0118 | 0.9614 ± 0.0133 | 0.9596 ± 0.0260 |
| digits | 0.9817 ± 0.0075 | 0.9417 ± 0.0152 | 0.9444 ± 0.0183 | 0.8878 ± 0.0992 | 0.9533 ± 0.0162 | 0.9739 ± 0.0082 | 0.9678 ± 0.0170 | 0.9722 ± 0.0094 | 0.9733 ± 0.0103 | 0.9733 ± 0.0075 | 0.9400 ± 0.0174 |
| iris | 0.9600 ± 0.0435 | 0.8667 ± 0.0624 | 0.9133 ± 0.0730 | 0.8800 ± 0.0506 | 0.9267 ± 0.0435 | 0.8867 ± 0.0447 | 0.9267 ± 0.0548 | 0.9267 ± 0.0723 | 0.9267 ± 0.0596 | 0.9000 ± 0.0667 | 0.8933 ± 0.0548 |
| wine | 0.9722 ± 0.0393 | 0.9833 ± 0.0248 | 0.9722 ± 0.0196 | 0.9167 ± 0.0340 | 0.9389 ± 0.0930 | 0.9722 ± 0.0278 | 0.9556 ± 0.0317 | 0.9611 ± 0.0248 | 0.9667 ± 0.0232 | 0.9722 ± 0.0340 | 0.9389 ± 0.0304 |
| zebra | 0.8909 ± 0.2439 | 0.6364 ± 0.1607 | 0.6364 ± 0.1790 | 0.6000 ± 0.1261 | 0.6273 ± 0.1302 | 0.5636 ± 0.0761 | 0.6000 ± 0.0593 | 0.5909 ± 0.0909 | 0.6818 ± 0.1325 | 0.5455 ± 0.1364 | 0.5727 ± 0.0407 |
| zebra_special | 0.6955 ± 0.0903 | 0.4773 ± 0.0663 | 0.4955 ± 0.0543 | 0.5500 ± 0.0901 | 0.5455 ± 0.1078 | 0.4727 ± 0.0929 | 0.5318 ± 0.0614 | 0.4909 ± 0.0124 | 0.5000 ± 0.0923 | 0.4364 ± 0.0929 | 0.5227 ± 0.0359 |

K=0.33

| dataset | full_memory | random | stratified | bias_current | bias_normalized | provenance_only | trust | trust_utility | trust_utility_coverage | utility_redundancy | kcenter |
|---|---|---|---|---|---|---|---|---|---|---|---|
| balance | 0.8496 ± 0.0236 | 0.8368 ± 0.0313 | 0.8400 ± 0.0080 | 0.8640 ± 0.0170 | 0.8576 ± 0.0222 | 0.8736 ± 0.0207 | 0.8672 ± 0.0201 | 0.8736 ± 0.0173 | 0.8736 ± 0.0088 | 0.8592 ± 0.0209 | 0.8448 ± 0.0184 |
| breast_cancer | 0.9649 ± 0.0062 | 0.9684 ± 0.0078 | 0.9579 ± 0.0209 | 0.9614 ± 0.0100 | 0.9702 ± 0.0048 | 0.9579 ± 0.0114 | 0.9649 ± 0.0107 | 0.9649 ± 0.0062 | 0.9684 ± 0.0048 | 0.9667 ± 0.0073 | 0.9421 ± 0.0202 |
| digits | 0.9817 ± 0.0075 | 0.9583 ± 0.0130 | 0.9639 ± 0.0132 | 0.9294 ± 0.0732 | 0.9689 ± 0.0137 | 0.9778 ± 0.0079 | 0.9744 ± 0.0107 | 0.9767 ± 0.0112 | 0.9761 ± 0.0110 | 0.9772 ± 0.0066 | 0.9639 ± 0.0201 |
| iris | 0.9600 ± 0.0435 | 0.8933 ± 0.0796 | 0.9333 ± 0.0707 | 0.9467 ± 0.0506 | 0.9533 ± 0.0380 | 0.9133 ± 0.0506 | 0.9600 ± 0.0435 | 0.9600 ± 0.0435 | 0.9600 ± 0.0435 | 0.9533 ± 0.0380 | 0.9400 ± 0.0365 |
| wine | 0.9722 ± 0.0393 | 0.9667 ± 0.0362 | 0.9722 ± 0.0196 | 0.9444 ± 0.0651 | 0.9667 ± 0.0362 | 0.9556 ± 0.0465 | 0.9722 ± 0.0278 | 0.9611 ± 0.0248 | 0.9667 ± 0.0304 | 0.9778 ± 0.0232 | 0.9667 ± 0.0456 |
| zebra | 0.8909 ± 0.2439 | 0.6364 ± 0.0787 | 0.6091 ± 0.1971 | 0.6455 ± 0.1302 | 0.6091 ± 0.1348 | 0.5182 ± 0.0996 | 0.6455 ± 0.0985 | 0.6545 ± 0.1046 | 0.6455 ± 0.1682 | 0.6545 ± 0.1459 | 0.5818 ± 0.1379 |
| zebra_special | 0.6955 ± 0.0903 | 0.4591 ± 0.0886 | 0.5091 ± 0.0859 | 0.5455 ± 0.1029 | 0.5727 ± 0.0610 | 0.5182 ± 0.0970 | 0.5909 ± 0.0923 | 0.5591 ± 0.0674 | 0.5591 ± 0.0918 | 0.5091 ± 0.0547 | 0.5045 ± 0.0631 |

K=0.5

| dataset | full_memory | random | stratified | bias_current | bias_normalized | provenance_only | trust | trust_utility | trust_utility_coverage | utility_redundancy | kcenter |
|---|---|---|---|---|---|---|---|---|---|---|---|
| balance | 0.8496 ± 0.0236 | 0.8384 ± 0.0341 | 0.8352 ± 0.0318 | 0.8480 ± 0.0188 | 0.8496 ± 0.0215 | 0.8688 ± 0.0166 | 0.8464 ± 0.0249 | 0.8528 ± 0.0216 | 0.8544 ± 0.0207 | 0.8624 ± 0.0173 | 0.8560 ± 0.0126 |
| breast_cancer | 0.9649 ± 0.0062 | 0.9649 ± 0.0062 | 0.9614 ± 0.0147 | 0.9667 ± 0.0114 | 0.9667 ± 0.0073 | 0.9632 ± 0.0039 | 0.9702 ± 0.0048 | 0.9632 ± 0.0096 | 0.9667 ± 0.0039 | 0.9667 ± 0.0039 | 0.9649 ± 0.0088 |
| digits | 0.9817 ± 0.0075 | 0.9678 ± 0.0124 | 0.9756 ± 0.0084 | 0.9539 ± 0.0586 | 0.9767 ± 0.0089 | 0.9783 ± 0.0066 | 0.9767 ± 0.0099 | 0.9783 ± 0.0093 | 0.9772 ± 0.0107 | 0.9794 ± 0.0054 | 0.9767 ± 0.0046 |
| iris | 0.9600 ± 0.0435 | 0.9400 ± 0.0279 | 0.9400 ± 0.0596 | 0.9467 ± 0.0506 | 0.9600 ± 0.0435 | 0.9533 ± 0.0380 | 0.9600 ± 0.0435 | 0.9600 ± 0.0435 | 0.9600 ± 0.0435 | 0.9600 ± 0.0435 | 0.9667 ± 0.0408 |
| wine | 0.9722 ± 0.0393 | 0.9778 ± 0.0304 | 0.9667 ± 0.0304 | 0.9667 ± 0.0304 | 0.9667 ± 0.0304 | 0.9556 ± 0.0465 | 0.9667 ± 0.0304 | 0.9667 ± 0.0304 | 0.9611 ± 0.0373 | 0.9667 ± 0.0304 | 0.9611 ± 0.0465 |
| zebra | 0.8909 ± 0.2439 | 0.7909 ± 0.1228 | 0.5909 ± 0.1701 | 0.7455 ± 0.1459 | 0.7455 ± 0.1459 | 0.5545 ± 0.1177 | 0.7000 ± 0.1459 | 0.7909 ± 0.1749 | 0.8000 ± 0.1594 | 0.8364 ± 0.1749 | 0.7818 ± 0.1743 |
| zebra_special | 0.6955 ± 0.0903 | 0.5136 ± 0.1024 | 0.5318 ± 0.0859 | 0.6591 ± 0.0964 | 0.6318 ± 0.1290 | 0.5864 ± 0.0943 | 0.6545 ± 0.1196 | 0.6409 ± 0.1319 | 0.6364 ± 0.1234 | 0.6227 ± 0.1166 | 0.6000 ± 0.0985 |

K=0.75

| dataset | full_memory | random | stratified | bias_current | bias_normalized | provenance_only | trust | trust_utility | trust_utility_coverage | utility_redundancy | kcenter |
|---|---|---|---|---|---|---|---|---|---|---|---|
| balance | 0.8496 ± 0.0236 | 0.8512 ± 0.0263 | 0.8512 ± 0.0347 | 0.8464 ± 0.0296 | 0.8464 ± 0.0243 | 0.8640 ± 0.0196 | 0.8416 ± 0.0301 | 0.8432 ± 0.0286 | 0.8448 ± 0.0275 | 0.8448 ± 0.0308 | 0.8528 ± 0.0145 |
| breast_cancer | 0.9649 ± 0.0062 | 0.9702 ± 0.0100 | 0.9614 ± 0.0133 | 0.9684 ± 0.0048 | 0.9667 ± 0.0073 | 0.9649 ± 0.0062 | 0.9667 ± 0.0039 | 0.9667 ± 0.0039 | 0.9667 ± 0.0039 | 0.9667 ± 0.0039 | 0.9684 ± 0.0078 |
| digits | 0.9817 ± 0.0075 | 0.9750 ± 0.0116 | 0.9789 ± 0.0091 | 0.9806 ± 0.0071 | 0.9800 ± 0.0082 | 0.9783 ± 0.0066 | 0.9800 ± 0.0082 | 0.9794 ± 0.0072 | 0.9806 ± 0.0071 | 0.9789 ± 0.0072 | 0.9794 ± 0.0064 |
| iris | 0.9600 ± 0.0435 | 0.9400 ± 0.0435 | 0.9600 ± 0.0435 | 0.9600 ± 0.0435 | 0.9600 ± 0.0435 | 0.9600 ± 0.0435 | 0.9600 ± 0.0435 | 0.9667 ± 0.0408 | 0.9667 ± 0.0408 | 0.9533 ± 0.0380 | 0.9667 ± 0.0408 |
| wine | 0.9722 ± 0.0393 | 0.9722 ± 0.0393 | 0.9667 ± 0.0362 | 0.9778 ± 0.0232 | 0.9667 ± 0.0362 | 0.9722 ± 0.0278 | 0.9611 ± 0.0373 | 0.9667 ± 0.0304 | 0.9667 ± 0.0362 | 0.9667 ± 0.0362 | 0.9722 ± 0.0393 |
| zebra | 0.8909 ± 0.2439 | 0.8000 ± 0.1863 | 0.7818 ± 0.2304 | 0.7909 ± 0.2418 | 0.8091 ± 0.2068 | 0.7727 ± 0.2469 | 0.7455 ± 0.1423 | 0.8091 ± 0.2435 | 0.8182 ± 0.2250 | 0.8909 ± 0.2439 | 0.8727 ± 0.1858 |
| zebra_special | 0.6955 ± 0.0903 | 0.6318 ± 0.0871 | 0.6455 ± 0.1132 | 0.6818 ± 0.1213 | 0.6727 ± 0.1251 | 0.6636 ± 0.1218 | 0.6773 ± 0.1152 | 0.6727 ± 0.1251 | 0.6864 ± 0.1218 | 0.6636 ± 0.1106 | 0.6455 ± 0.1177 |

*Observations (this report; there is no author conclusion):* fine-tuning lowered accuracy by more than 0.1 in 230 of 1,400 non-full-memory runs (16.4%). By dataset: balance 10.5%, breast_cancer 21%, digits 74.5%, iris 0.5%, wine 0%, zebra 7.5%, zebra_special 1%. After fine-tune on digits at K ≥ 0.33, every policy is about 0.10, except bias_current at K=0.33 (0.2800 ± 0.3871, because some seeds did not collapse). Without fine-tune the same digits cells are 0.93–0.98. This is the failure that `6adeb30` targets, and that fix is not rerun. Without fine-tune, the policies stay within a few points of random and full memory on most datasets.

#### 1.3.3 P1-R: retention on regression
*Tests:* the same question for regression, scored by `test_rmse_pre` (lower is better). *Setup:* 5 datasets × 11 policies × K ∈ {0.2, 0.5}, 5 seeds. Source: `results/t1/p1_retention_regression/runs__*.jsonl` (computed for this report).

K=0.2, after fine-tune

| dataset | full_memory | random | stratified | bias_current | bias_normalized | provenance_only | trust | trust_utility | trust_utility_coverage | utility_redundancy | kcenter |
|---|---|---|---|---|---|---|---|---|---|---|---|
| abalone | 0.7543 ± 0.0777 | 0.7343 ± 0.0571 | 0.9541 ± 0.0861 | 0.8472 ± 0.0782 | 0.9121 ± 0.0386 | 0.9227 ± 0.0720 | 0.8446 ± 0.1062 | 0.9364 ± 0.1666 | 0.8431 ± 0.0862 | 0.8501 ± 0.0793 | 1.0111 ± 0.1149 |
| airfoil | 0.3765 ± 0.0194 | 0.4551 ± 0.0364 | 0.4526 ± 0.0468 | 0.4108 ± 0.0347 | 0.4114 ± 0.0393 | 0.3982 ± 0.0286 | 0.4017 ± 0.0299 | 0.3995 ± 0.0315 | 0.3956 ± 0.0358 | 0.4023 ± 0.0341 | 0.4630 ± 0.1178 |
| energy_efficiency | 0.0844 ± 0.0346 | 0.2122 ± 0.1030 | 0.2869 ± 0.1744 | 0.1277 ± 0.0169 | 0.1226 ± 0.0116 | 0.1796 ± 0.0306 | 0.1438 ± 0.0304 | 0.1327 ± 0.0299 | 0.1201 ± 0.0162 | 0.1553 ± 0.0438 | 0.1719 ± 0.1143 |
| student_performance | 0.4781 ± 0.0826 | 0.4779 ± 0.1273 | 0.4588 ± 0.0856 | 0.4453 ± 0.0995 | 0.4630 ± 0.0878 | 0.4609 ± 0.0939 | 0.4629 ± 0.0788 | 0.4593 ± 0.0973 | 0.4524 ± 0.0897 | 0.4558 ± 0.1011 | 0.4589 ± 0.1184 |
| yacht | 0.1773 ± 0.0646 | 0.1842 ± 0.0524 | 0.2560 ± 0.2178 | 0.1574 ± 0.0407 | 0.1486 ± 0.0639 | 0.1251 ± 0.0511 | 0.1275 ± 0.0378 | 0.1520 ± 0.0649 | 0.1476 ± 0.0445 | 0.1426 ± 0.0550 | 0.1151 ± 0.0309 |

K=0.5, after fine-tune

| dataset | full_memory | random | stratified | bias_current | bias_normalized | provenance_only | trust | trust_utility | trust_utility_coverage | utility_redundancy | kcenter |
|---|---|---|---|---|---|---|---|---|---|---|---|
| abalone | 0.7468 ± 0.0666 | 0.9527 ± 0.0851 | 0.9298 ± 0.1171 | 0.9388 ± 0.0580 | 0.9772 ± 0.1174 | 0.9238 ± 0.0825 | 0.9595 ± 0.0802 | 1.0024 ± 0.1271 | 0.9538 ± 0.0416 | 0.9379 ± 0.1035 | 0.9601 ± 0.0709 |
| airfoil | 0.3637 ± 0.0286 | 0.8678 ± 0.1037 | 0.8230 ± 0.2732 | 0.6795 ± 0.2107 | 0.6541 ± 0.1814 | 0.7806 ± 0.3321 | 0.6125 ± 0.1958 | 0.6205 ± 0.1961 | 0.6586 ± 0.1849 | 0.5536 ± 0.1847 | 0.8799 ± 0.0845 |
| energy_efficiency | 0.0838 ± 0.0312 | 0.2200 ± 0.0462 | 0.3232 ± 0.1213 | 0.1062 ± 0.0235 | 0.1180 ± 0.0222 | 0.2477 ± 0.0910 | 0.1082 ± 0.0180 | 0.2006 ± 0.0706 | 0.1396 ± 0.0193 | 0.3030 ± 0.0921 | 0.1941 ± 0.1123 |
| student_performance | 0.4770 ± 0.0871 | 0.4696 ± 0.1134 | 0.4543 ± 0.0869 | 0.4635 ± 0.0816 | 0.4624 ± 0.0850 | 0.4664 ± 0.0952 | 0.4641 ± 0.0925 | 0.4660 ± 0.0941 | 0.4661 ± 0.0916 | 0.4747 ± 0.0825 | 0.4535 ± 0.0808 |
| yacht | 0.1675 ± 0.0740 | 0.1406 ± 0.0580 | 0.1304 ± 0.0279 | 0.1536 ± 0.0573 | 0.1277 ± 0.0525 | 0.1404 ± 0.0437 | 0.1354 ± 0.0493 | 0.1212 ± 0.0345 | 0.1361 ± 0.0414 | 0.1331 ± 0.0256 | 0.1210 ± 0.0477 |

K=0.2, no fine-tune

| dataset | full_memory | random | stratified | bias_current | bias_normalized | provenance_only | trust | trust_utility | trust_utility_coverage | utility_redundancy | kcenter |
|---|---|---|---|---|---|---|---|---|---|---|---|
| abalone | 0.7269 ± 0.0418 | 0.7784 ± 0.0439 | 0.7812 ± 0.0591 | 0.7374 ± 0.0417 | 0.7355 ± 0.0431 | 0.7336 ± 0.0445 | 0.7280 ± 0.0353 | 0.7301 ± 0.0409 | 0.7258 ± 0.0397 | 0.7315 ± 0.0482 | 0.7887 ± 0.0374 |
| airfoil | 0.3768 ± 0.0290 | 0.5557 ± 0.0617 | 0.5648 ± 0.0499 | 0.5285 ± 0.0676 | 0.5227 ± 0.0575 | 0.4856 ± 0.0211 | 0.4877 ± 0.0507 | 0.4762 ± 0.0589 | 0.4660 ± 0.0454 | 0.4665 ± 0.0486 | 0.5493 ± 0.0201 |
| energy_efficiency | 0.0745 ± 0.0329 | 0.2953 ± 0.0680 | 0.3137 ± 0.0737 | 0.2094 ± 0.0283 | 0.2190 ± 0.0177 | 0.3388 ± 0.1511 | 0.2225 ± 0.0254 | 0.2016 ± 0.0441 | 0.2024 ± 0.0173 | 0.2469 ± 0.0203 | 0.2643 ± 0.1901 |
| student_performance | 0.4864 ± 0.0868 | 0.6399 ± 0.1550 | 0.6158 ± 0.0938 | 0.4955 ± 0.0849 | 0.4922 ± 0.0846 | 0.4888 ± 0.0785 | 0.4798 ± 0.0812 | 0.4813 ± 0.0844 | 0.4810 ± 0.0866 | 0.4771 ± 0.0894 | 0.6640 ± 0.1589 |
| yacht | 0.1812 ± 0.0786 | 0.4044 ± 0.1257 | 0.3659 ± 0.0924 | 0.2367 ± 0.0704 | 0.2119 ± 0.0909 | 0.1933 ± 0.0903 | 0.1926 ± 0.0883 | 0.1940 ± 0.0820 | 0.1825 ± 0.0778 | 0.2053 ± 0.0975 | 0.3484 ± 0.1370 |

K=0.5, no fine-tune

| dataset | full_memory | random | stratified | bias_current | bias_normalized | provenance_only | trust | trust_utility | trust_utility_coverage | utility_redundancy | kcenter |
|---|---|---|---|---|---|---|---|---|---|---|---|
| abalone | 0.7269 ± 0.0418 | 0.7445 ± 0.0506 | 0.7410 ± 0.0448 | 0.7272 ± 0.0385 | 0.7268 ± 0.0380 | 0.7269 ± 0.0430 | 0.7274 ± 0.0405 | 0.7281 ± 0.0406 | 0.7288 ± 0.0406 | 0.7277 ± 0.0465 | 0.7498 ± 0.0416 |
| airfoil | 0.3768 ± 0.0290 | 0.4503 ± 0.0486 | 0.4645 ± 0.0592 | 0.3992 ± 0.0378 | 0.3978 ± 0.0398 | 0.4037 ± 0.0255 | 0.3925 ± 0.0405 | 0.3913 ± 0.0388 | 0.3917 ± 0.0379 | 0.3944 ± 0.0302 | 0.4704 ± 0.0360 |
| energy_efficiency | 0.0745 ± 0.0329 | 0.1782 ± 0.0513 | 0.2066 ± 0.0449 | 0.1033 ± 0.0251 | 0.1219 ± 0.0245 | 0.1267 ± 0.0563 | 0.1205 ± 0.0247 | 0.1100 ± 0.0264 | 0.1003 ± 0.0232 | 0.1026 ± 0.0245 | 0.1611 ± 0.0940 |
| student_performance | 0.4864 ± 0.0868 | 0.5567 ± 0.1564 | 0.5239 ± 0.0955 | 0.4866 ± 0.0891 | 0.4849 ± 0.0873 | 0.4839 ± 0.0850 | 0.4841 ± 0.0872 | 0.4850 ± 0.0881 | 0.4857 ± 0.0892 | 0.4866 ± 0.0889 | 0.5577 ± 0.0760 |
| yacht | 0.1812 ± 0.0786 | 0.2894 ± 0.1710 | 0.1872 ± 0.0469 | 0.1903 ± 0.0827 | 0.1884 ± 0.0813 | 0.1940 ± 0.0934 | 0.1884 ± 0.0813 | 0.1879 ± 0.0812 | 0.1817 ± 0.0787 | 0.1850 ± 0.0760 | 0.2098 ± 0.0478 |

*Observations:* without fine-tune, the evidence-based policies beat random, stratified and kcenter on airfoil, energy_efficiency and yacht, but none beats full memory except at near-tie level. Fine-tuning degrades K=0.5 badly on abalone and airfoil. Full memory also changes under fine-tune (abalone 0.7543 vs 0.7269 at K=0.2), so the "K" label on full_memory only names the paired condition.

#### 1.3.4 P1-S: retention on `synthetic_diag`
*Tests:* whether retention keeps clean and rare cases and drops corrupted and duplicate ones, using ground truth. *Setup:* `synthetic_diag` (15% label corruption, 60 duplicates, a 12-case rare cluster, a shifted test group), K ∈ {0.3, 0.5}, 10 seeds. Source: `results/t1/p1_retention_synthetic/runs__synthetic_diag.jsonl` (computed for this report).

| condition | n | acc (after FT) | acc (no FT) | in_domain | shifted | rare | kept_corrupted | kept_duplicate | kept_rare | kept_clean |
|---|---|---|---|---|---|---|---|---|---|---|
| bias_current_K0.3 | 10 | 0.943 ± 0.034 | 0.898 ± 0.067 | 0.968 ± 0.033 | 0.883 ± 0.100 | 0.900 ± 0.316 | 0.070 ± 0.048 | 0.402 ± 0.068 | 0.000 ± 0.000 | 0.334 ± 0.017 |
| bias_current_K0.5 | 10 | 0.915 ± 0.063 | 0.895 ± 0.063 | 0.956 ± 0.038 | 0.848 ± 0.115 | 0.745 ± 0.428 | 0.161 ± 0.038 | 0.643 ± 0.065 | 0.000 ± 0.000 | 0.551 ± 0.017 |
| bias_normalized_K0.3 | 10 | 0.939 ± 0.046 | 0.898 ± 0.064 | 0.967 ± 0.037 | 0.902 ± 0.070 | 0.800 ± 0.422 | 0.074 ± 0.045 | 0.392 ± 0.066 | 0.000 ± 0.000 | 0.335 ± 0.014 |
| bias_normalized_K0.5 | 10 | 0.923 ± 0.060 | 0.895 ± 0.062 | 0.959 ± 0.038 | 0.857 ± 0.122 | 0.800 ± 0.422 | 0.165 ± 0.040 | 0.645 ± 0.078 | 0.000 ± 0.000 | 0.550 ± 0.017 |
| full_memory_K0.3 | 10 | 0.940 ± 0.055 | 0.943 ± 0.053 | 0.957 ± 0.049 | 0.872 ± 0.125 | 1.000 ± 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| full_memory_K0.5 | 10 | 0.942 ± 0.056 | 0.943 ± 0.053 | 0.957 ± 0.049 | 0.875 ± 0.124 | 1.000 ± 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| kcenter_K0.3 | 10 | 0.818 ± 0.228 | 0.924 ± 0.058 | 0.829 ± 0.251 | 0.728 ± 0.257 | 0.995 ± 0.016 | 0.417 ± 0.037 | 0.152 ± 0.020 | 0.367 ± 0.216 | 0.307 ± 0.008 |
| kcenter_K0.5 | 10 | 0.763 ± 0.200 | 0.928 ± 0.063 | 0.781 ± 0.210 | 0.635 ± 0.250 | 0.995 ± 0.016 | 0.665 ± 0.071 | 0.253 ± 0.053 | 0.483 ± 0.235 | 0.520 ± 0.015 |
| provenance_only_K0.3 | 10 | 0.904 ± 0.148 | 0.953 ± 0.036 | 0.919 ± 0.153 | 0.827 ± 0.196 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.490 ± 0.052 | 0.642 ± 0.125 | 0.304 ± 0.011 |
| provenance_only_K0.5 | 10 | 0.865 ± 0.172 | 0.955 ± 0.038 | 0.883 ± 0.184 | 0.765 ± 0.225 | 1.000 ± 0.000 | 0.006 ± 0.018 | 0.747 ± 0.041 | 0.842 ± 0.083 | 0.525 ± 0.009 |
| random_K0.3 | 10 | 0.956 ± 0.037 | 0.938 ± 0.042 | 0.973 ± 0.030 | 0.890 ± 0.109 | 1.000 ± 0.000 | 0.298 ± 0.052 | 0.313 ± 0.037 | 0.225 ± 0.136 | 0.302 ± 0.012 |
| random_K0.5 | 10 | 0.833 ± 0.227 | 0.934 ± 0.063 | 0.847 ± 0.252 | 0.738 ± 0.250 | 0.995 ± 0.016 | 0.506 ± 0.066 | 0.512 ± 0.059 | 0.433 ± 0.135 | 0.499 ± 0.022 |
| stratified_K0.3 | 10 | 0.902 ± 0.171 | 0.950 ± 0.040 | 0.912 ± 0.185 | 0.840 ± 0.198 | 1.000 ± 0.000 | 0.287 ± 0.046 | 0.300 ± 0.057 | 0.233 ± 0.117 | 0.306 ± 0.013 |
| stratified_K0.5 | 10 | 0.766 ± 0.218 | 0.950 ± 0.043 | 0.773 ± 0.230 | 0.667 ± 0.269 | 1.000 ± 0.000 | 0.485 ± 0.038 | 0.508 ± 0.065 | 0.450 ± 0.105 | 0.503 ± 0.013 |
| trust_K0.3 | 10 | 0.932 ± 0.050 | 0.898 ± 0.062 | 0.960 ± 0.041 | 0.890 ± 0.081 | 0.800 ± 0.422 | 0.022 ± 0.035 | 0.448 ± 0.069 | 0.000 ± 0.000 | 0.333 ± 0.014 |
| trust_K0.5 | 10 | 0.934 ± 0.054 | 0.897 ± 0.062 | 0.964 ± 0.035 | 0.863 ± 0.107 | 0.875 ± 0.317 | 0.109 ± 0.051 | 0.677 ± 0.077 | 0.000 ± 0.000 | 0.554 ± 0.017 |
| trust_utility_K0.3 | 10 | 0.942 ± 0.050 | 0.894 ± 0.063 | 0.966 ± 0.035 | 0.887 ± 0.084 | 0.900 ± 0.316 | 0.024 ± 0.033 | 0.480 ± 0.053 | 0.000 ± 0.000 | 0.326 ± 0.009 |
| trust_utility_K0.5 | 10 | 0.898 ± 0.103 | 0.913 ± 0.064 | 0.933 ± 0.090 | 0.825 ± 0.180 | 0.800 ± 0.422 | 0.122 ± 0.041 | 0.712 ± 0.075 | 0.025 ± 0.056 | 0.544 ± 0.018 |
| trust_utility_coverage_K0.3 | 10 | 0.937 ± 0.057 | 0.948 ± 0.043 | 0.957 ± 0.044 | 0.857 ± 0.138 | 1.000 ± 0.000 | 0.080 ± 0.030 | 0.338 ± 0.070 | 0.083 ± 0.000 | 0.341 ± 0.010 |
| trust_utility_coverage_K0.5 | 10 | 0.913 ± 0.093 | 0.950 ± 0.044 | 0.933 ± 0.082 | 0.825 ± 0.168 | 1.000 ± 0.000 | 0.169 ± 0.030 | 0.618 ± 0.056 | 0.125 ± 0.044 | 0.550 ± 0.013 |
| utility_redundancy_K0.3 | 10 | 0.892 ± 0.174 | 0.945 ± 0.049 | 0.906 ± 0.191 | 0.815 ± 0.195 | 1.000 ± 0.000 | 0.094 ± 0.028 | 0.353 ± 0.053 | 0.267 ± 0.129 | 0.328 ± 0.010 |
| utility_redundancy_K0.5 | 10 | 0.860 ± 0.139 | 0.947 ± 0.051 | 0.881 ± 0.136 | 0.752 ± 0.208 | 1.000 ± 0.000 | 0.204 ± 0.060 | 0.602 ± 0.036 | 0.700 ± 0.058 | 0.525 ± 0.011 |

*Observations (this report):* fine-tuning lost more than 0.1 in 27 of 200 non-full-memory runs. The bias, trust and trust-utility policies discard the rare cluster entirely (kept_rare = 0; the one exception is trust_utility_K0.5 at 0.025) while keeping 39–71% of duplicates. provenance_only keeps about 0% of corrupted cases and most of the rare cluster.

#### 1.3.5 P2: reuse adapter (Gate 3)
*Tests:* whether the NN-CDH adapter improves accuracy without calibration damage or harmful flips. *Setup:* 8 datasets × {logit_residual, nominal_residual_scores} × {cls_only, combined, diff_only}, 5 seeds, with retrieval frozen. It ran **before `ec6323a`**, but because retrieval is frozen the optimizer fix should not matter here. Source: untracked `results/t1/p2_reuse/summary.csv` (pivot by this report).

| dataset | condition | acc_pre | acc_post | ece_pre | ece_post | net_flips | wrong→correct | correct→wrong | corr_L1 |
|---|---|---|---|---|---|---|---|---|---|
| balance | logit_residual_cls_only | 0.8496 ± 0.0236 | 0.8912 ± 0.0216 | 0.0649 ± 0.0200 | 0.0633 ± 0.0105 | 5.2 ± 4.4 | 8.6 ± 4.2 | 3.4 ± 1.1 | 0.275 ± 0.075 |
| balance | logit_residual_combined | 0.8496 ± 0.0236 | 0.8992 ± 0.0216 | 0.0649 ± 0.0200 | 0.0603 ± 0.0173 | 6.2 ± 4.3 | 9.6 ± 3.7 | 3.4 ± 1.5 | 0.285 ± 0.064 |
| balance | logit_residual_diff_only | 0.8496 ± 0.0236 | 0.8912 ± 0.0121 | 0.0649 ± 0.0200 | 0.0556 ± 0.0084 | 5.2 ± 1.6 | 6.0 ± 2.3 | 0.8 ± 1.1 | 0.154 ± 0.061 |
| balance | nominal_residual_scores_cls_only | 0.8496 ± 0.0236 | 0.8992 ± 0.0230 | 0.0649 ± 0.0200 | 0.1277 ± 0.0087 | 6.2 ± 4.5 | 9.0 ± 4.6 | 2.8 ± 0.8 | 2.697 ± 0.135 |
| balance | nominal_residual_scores_combined | 0.8496 ± 0.0236 | 0.8928 ± 0.0175 | 0.0649 ± 0.0200 | 0.2403 ± 0.0082 | 5.4 ± 3.6 | 7.8 ± 3.8 | 2.4 ± 1.1 | 1.048 ± 0.029 |
| balance | nominal_residual_scores_diff_only | 0.8496 ± 0.0236 | 0.8848 ± 0.0121 | 0.0649 ± 0.0200 | 0.3465 ± 0.0163 | 4.4 ± 2.8 | 6.8 ± 3.3 | 2.4 ± 1.1 | 0.348 ± 0.061 |
| breast_cancer | logit_residual_cls_only | 0.9649 ± 0.0062 | 0.9544 ± 0.0096 | 0.0362 ± 0.0057 | 0.0359 ± 0.0056 | -1.2 ± 0.8 | 0.2 ± 0.4 | 1.4 ± 1.1 | 0.025 ± 0.006 |
| breast_cancer | logit_residual_combined | 0.9649 ± 0.0062 | 0.9544 ± 0.0096 | 0.0362 ± 0.0057 | 0.0366 ± 0.0048 | -1.2 ± 0.8 | 0.2 ± 0.4 | 1.4 ± 1.1 | 0.025 ± 0.007 |
| breast_cancer | logit_residual_diff_only | 0.9649 ± 0.0062 | 0.9614 ± 0.0048 | 0.0362 ± 0.0057 | 0.0336 ± 0.0066 | -0.4 ± 0.5 | 0.4 ± 0.5 | 0.8 ± 0.8 | 0.022 ± 0.011 |
| breast_cancer | nominal_residual_scores_cls_only | 0.9649 ± 0.0062 | 0.9667 ± 0.0096 | 0.0362 ± 0.0057 | 0.0511 ± 0.0136 | 0.2 ± 1.1 | 0.8 ± 0.8 | 0.6 ± 0.5 | 1.900 ± 0.114 |
| breast_cancer | nominal_residual_scores_combined | 0.9649 ± 0.0062 | 0.9632 ± 0.0096 | 0.0362 ± 0.0057 | 0.1895 ± 0.0065 | -0.2 ± 0.8 | 0.4 ± 0.5 | 0.6 ± 0.5 | 0.374 ± 0.031 |
| breast_cancer | nominal_residual_scores_diff_only | 0.9649 ± 0.0062 | 0.9632 ± 0.0073 | 0.0362 ± 0.0057 | 0.2518 ± 0.0129 | -0.2 ± 0.8 | 0.2 ± 0.4 | 0.4 ± 0.5 | 0.095 ± 0.013 |
| digits | logit_residual_cls_only | 0.9817 ± 0.0075 | 0.9789 ± 0.0101 | 0.0224 ± 0.0060 | 0.0191 ± 0.0019 | -1.0 ± 1.9 | 1.4 ± 1.5 | 2.4 ± 1.7 | 0.023 ± 0.007 |
| digits | logit_residual_combined | 0.9817 ± 0.0075 | 0.9794 ± 0.0109 | 0.0224 ± 0.0060 | 0.0190 ± 0.0026 | -0.8 ± 2.2 | 1.4 ± 1.5 | 2.2 ± 1.6 | 0.022 ± 0.007 |
| digits | logit_residual_diff_only | 0.9817 ± 0.0075 | 0.9811 ± 0.0103 | 0.0224 ± 0.0060 | 0.0192 ± 0.0039 | -0.2 ± 2.2 | 1.2 ± 1.6 | 1.4 ± 0.9 | 0.016 ± 0.005 |
| digits | nominal_residual_scores_cls_only | 0.9817 ± 0.0075 | 0.9767 ± 0.0091 | 0.0224 ± 0.0060 | 0.3051 ± 0.0050 | -1.8 ± 2.2 | 0.6 ± 0.5 | 2.4 ± 1.8 | 9.974 ± 0.010 |
| digits | nominal_residual_scores_combined | 0.9817 ± 0.0075 | 0.9822 ± 0.0078 | 0.0224 ± 0.0060 | 0.4758 ± 0.0067 | 0.2 ± 0.8 | 0.4 ± 0.5 | 0.2 ± 0.4 | 3.462 ± 0.015 |
| digits | nominal_residual_scores_diff_only | 0.9817 ± 0.0075 | 0.9828 ± 0.0082 | 0.0224 ± 0.0060 | 0.7555 ± 0.0079 | 0.4 ± 0.5 | 0.4 ± 0.5 | 0.0 ± 0.0 | 0.092 ± 0.039 |
| iris | logit_residual_cls_only | 0.9600 ± 0.0435 | 0.9667 ± 0.0408 | 0.0557 ± 0.0260 | 0.0565 ± 0.0335 | 0.2 ± 0.4 | 0.2 ± 0.4 | 0.0 ± 0.0 | 0.020 ± 0.011 |
| iris | logit_residual_combined | 0.9600 ± 0.0435 | 0.9667 ± 0.0408 | 0.0557 ± 0.0260 | 0.0567 ± 0.0333 | 0.2 ± 0.4 | 0.2 ± 0.4 | 0.0 ± 0.0 | 0.016 ± 0.009 |
| iris | logit_residual_diff_only | 0.9600 ± 0.0435 | 0.9667 ± 0.0408 | 0.0557 ± 0.0260 | 0.0520 ± 0.0338 | 0.2 ± 0.4 | 0.2 ± 0.4 | 0.0 ± 0.0 | 0.017 ± 0.013 |
| iris | nominal_residual_scores_cls_only | 0.9600 ± 0.0435 | 0.9600 ± 0.0365 | 0.0557 ± 0.0260 | 0.1163 ± 0.0262 | 0.0 ± 0.7 | 0.2 ± 0.4 | 0.2 ± 0.4 | 2.722 ± 0.212 |
| iris | nominal_residual_scores_combined | 0.9600 ± 0.0435 | 0.9600 ± 0.0435 | 0.0557 ± 0.0260 | 0.2699 ± 0.0380 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.801 ± 0.035 |
| iris | nominal_residual_scores_diff_only | 0.9600 ± 0.0435 | 0.9600 ± 0.0435 | 0.0557 ± 0.0260 | 0.3968 ± 0.0403 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.110 ± 0.025 |
| synthetic_diag | logit_residual_cls_only | 0.9254 ± 0.0697 | 0.9238 ± 0.0650 | 0.0441 ± 0.0295 | 0.0541 ± 0.0373 | -0.4 ± 2.7 | 4.4 ± 6.8 | 4.8 ± 5.0 | 0.058 ± 0.069 |
| synthetic_diag | logit_residual_combined | 0.9254 ± 0.0697 | 0.9246 ± 0.0614 | 0.0441 ± 0.0295 | 0.0447 ± 0.0243 | -0.2 ± 3.6 | 4.8 ± 7.7 | 5.0 ± 4.8 | 0.065 ± 0.085 |
| synthetic_diag | logit_residual_diff_only | 0.9254 ± 0.0697 | 0.9146 ± 0.0604 | 0.0441 ± 0.0295 | 0.0449 ± 0.0176 | -2.8 ± 3.3 | 4.2 ± 5.3 | 7.0 ± 4.1 | 0.065 ± 0.055 |
| synthetic_diag | nominal_residual_scores_cls_only | 0.9254 ± 0.0697 | 0.9085 ± 0.0680 | 0.0441 ± 0.0295 | 0.1449 ± 0.0478 | -4.4 ± 1.8 | 3.8 ± 6.5 | 8.2 ± 6.1 | 1.894 ± 0.465 |
| synthetic_diag | nominal_residual_scores_combined | 0.9254 ± 0.0697 | 0.9262 ± 0.0626 | 0.0441 ± 0.0295 | 0.3200 ± 0.0321 | 0.2 ± 3.7 | 2.8 ± 4.7 | 2.6 ± 2.1 | 0.503 ± 0.066 |
| synthetic_diag | nominal_residual_scores_diff_only | 0.9254 ± 0.0697 | 0.9231 ± 0.0650 | 0.0441 ± 0.0295 | 0.4084 ± 0.0430 | -0.6 ± 2.9 | 2.8 ± 5.7 | 3.4 ± 3.3 | 0.226 ± 0.059 |
| wine | logit_residual_cls_only | 0.9722 ± 0.0393 | 0.9778 ± 0.0304 | 0.0344 ± 0.0163 | 0.0316 ± 0.0212 | 0.2 ± 0.4 | 0.2 ± 0.4 | 0.0 ± 0.0 | 0.020 ± 0.010 |
| wine | logit_residual_combined | 0.9722 ± 0.0393 | 0.9778 ± 0.0304 | 0.0344 ± 0.0163 | 0.0316 ± 0.0212 | 0.2 ± 0.4 | 0.2 ± 0.4 | 0.0 ± 0.0 | 0.020 ± 0.010 |
| wine | logit_residual_diff_only | 0.9722 ± 0.0393 | 0.9778 ± 0.0304 | 0.0344 ± 0.0163 | 0.0327 ± 0.0180 | 0.2 ± 0.4 | 0.2 ± 0.4 | 0.0 ± 0.0 | 0.016 ± 0.005 |
| wine | nominal_residual_scores_cls_only | 0.9722 ± 0.0393 | 0.9778 ± 0.0304 | 0.0344 ± 0.0163 | 0.1008 ± 0.0162 | 0.2 ± 0.4 | 0.2 ± 0.4 | 0.0 ± 0.0 | 2.966 ± 0.027 |
| wine | nominal_residual_scores_combined | 0.9722 ± 0.0393 | 0.9778 ± 0.0304 | 0.0344 ± 0.0163 | 0.3007 ± 0.0051 | 0.2 ± 0.4 | 0.2 ± 0.4 | 0.0 ± 0.0 | 0.826 ± 0.029 |
| wine | nominal_residual_scores_diff_only | 0.9722 ± 0.0393 | 0.9722 ± 0.0393 | 0.0344 ± 0.0163 | 0.4257 ± 0.0135 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.149 ± 0.031 |
| zebra | logit_residual_cls_only | 0.8909 ± 0.2439 | 0.8727 ± 0.1885 | 0.1303 ± 0.1197 | 0.1119 ± 0.1008 | -0.4 ± 1.7 | 0.8 ± 1.8 | 1.2 ± 1.1 | 0.151 ± 0.103 |
| zebra | logit_residual_combined | 0.8909 ± 0.2439 | 0.8727 ± 0.1885 | 0.1303 ± 0.1197 | 0.1063 ± 0.0878 | -0.4 ± 1.7 | 0.8 ± 1.8 | 1.2 ± 1.1 | 0.151 ± 0.104 |
| zebra | logit_residual_diff_only | 0.8909 ± 0.2439 | 0.8182 ± 0.2007 | 0.1303 ± 0.1197 | 0.1424 ± 0.1047 | -1.6 ± 3.5 | 0.6 ± 1.3 | 2.2 ± 2.9 | 0.163 ± 0.133 |
| zebra | nominal_residual_scores_cls_only | 0.8909 ± 0.2439 | 0.7000 ± 0.2643 | 0.1303 ± 0.1197 | 0.3137 ± 0.0514 | -4.2 ± 5.5 | 0.0 ± 0.0 | 4.2 ± 5.5 | 0.652 ± 0.457 |
| zebra | nominal_residual_scores_combined | 0.8909 ± 0.2439 | 0.7091 ± 0.2543 | 0.1303 ± 0.1197 | 0.3086 ± 0.0743 | -4.0 ± 5.7 | 0.4 ± 0.9 | 4.4 ± 5.3 | 0.542 ± 0.325 |
| zebra | nominal_residual_scores_diff_only | 0.8909 ± 0.2439 | 0.7818 ± 0.2068 | 0.1303 ± 0.1197 | 0.2814 ± 0.0545 | -2.4 ± 4.3 | 0.6 ± 1.3 | 3.0 ± 3.7 | 0.436 ± 0.176 |
| zebra_special | logit_residual_cls_only | 0.6955 ± 0.0903 | 0.6818 ± 0.0455 | 0.1779 ± 0.0393 | 0.1721 ± 0.0506 | -0.6 ± 2.1 | 2.4 ± 2.5 | 3.0 ± 2.0 | 0.134 ± 0.072 |
| zebra_special | logit_residual_combined | 0.6955 ± 0.0903 | 0.6864 ± 0.0689 | 0.1779 ± 0.0393 | 0.1780 ± 0.0375 | -0.4 ± 1.8 | 1.6 ± 1.8 | 2.0 ± 2.3 | 0.100 ± 0.068 |
| zebra_special | logit_residual_diff_only | 0.6955 ± 0.0903 | 0.6773 ± 0.0610 | 0.1779 ± 0.0393 | 0.1787 ± 0.0421 | -0.8 ± 1.6 | 2.0 ± 2.7 | 2.8 ± 1.9 | 0.107 ± 0.072 |
| zebra_special | nominal_residual_scores_cls_only | 0.6955 ± 0.0903 | 0.6818 ± 0.0978 | 0.1779 ± 0.0393 | 0.1302 ± 0.0403 | -0.6 ± 1.1 | 0.6 ± 0.9 | 1.2 ± 0.8 | 0.959 ± 0.317 |
| zebra_special | nominal_residual_scores_combined | 0.6955 ± 0.0903 | 0.6818 ± 0.0771 | 0.1779 ± 0.0393 | 0.0797 ± 0.0202 | -0.6 ± 1.5 | 2.0 ± 2.5 | 2.6 ± 3.1 | 0.268 ± 0.085 |
| zebra_special | nominal_residual_scores_diff_only | 0.6955 ± 0.0903 | 0.6909 ± 0.0781 | 0.1779 ± 0.0393 | 0.1148 ± 0.0330 | -0.2 ± 0.8 | 1.0 ± 1.4 | 1.2 ± 0.8 | 0.131 ± 0.072 |

*Observations (this report):* only balance gains clearly (acc_post − acc_pre from +0.035 to +0.050). Nominal-residual modes raise ECE sharply, up to 0.7555 on digits. zebra loses up to 0.19 accuracy (nominal cls_only). Logit-residual leaves ECE roughly unchanged.

#### 1.3.6 P4: synchronization (Gate 5)
*Tests:* the independent, alternating_rr and alternating_rrr schedules for retrieval/reuse co-training under the free-correction radius. *Setup:* 8 datasets, 5 seeds, maintenance at epochs 40/80. `sync.K=0.5` applies only to the retain step, so only alternating_rrr compacts (iris n_cases 45). independent and alternating_rr run on full memory (iris n_cases 90). Runs came after `ec6323a` and before `6adeb30`. Source: `results/t1/p4_sync/runs__*.jsonl` (computed for this report).

| dataset | schedule | acc_pre | acc_post | ece_post | net_flips | correct→wrong | corr_L1 | tau_task | radius_status |
|---|---|---|---|---|---|---|---|---|---|
| balance | alternating_rr | 0.8720 ± 0.0113 | 0.9072 ± 0.0323 | 0.2777 ± 0.0230 | 4.4 ± 4.0 | 2.0 ± 1.2 | 0.805 ± 0.027 | 0.277 ± 0.035 | ok |
| balance | alternating_rrr | 0.8784 ± 0.0222 | 0.9072 ± 0.0216 | 0.2565 ± 0.0114 | 3.6 ± 1.8 | 2.8 ± 0.8 | 0.982 ± 0.061 | 0.277 ± 0.035 | ok |
| balance | independent | 0.8000 ± 0.1656 | 0.8352 ± 0.1592 | 0.2380 ± 0.0041 | 4.4 ± 1.8 | 1.6 ± 2.6 | 0.976 ± 0.208 | 0.277 ± 0.035 | ok |
| breast_cancer | alternating_rr | 0.9579 ± 0.0073 | 0.9596 ± 0.0100 | 0.1996 ± 0.0159 | 0.2 ± 1.1 | 0.4 ± 0.5 | 0.336 ± 0.064 | 0.119 ± 0.047 | ok |
| breast_cancer | alternating_rrr | 0.9614 ± 0.0100 | 0.9596 ± 0.0118 | 0.1969 ± 0.0204 | -0.2 ± 0.4 | 0.4 ± 0.5 | 0.396 ± 0.046 | 0.119 ± 0.047 | ok |
| breast_cancer | independent | 0.9561 ± 0.0124 | 0.9579 ± 0.0130 | 0.1877 ± 0.0087 | 0.2 ± 1.1 | 0.2 ± 0.4 | 0.374 ± 0.009 | 0.119 ± 0.047 | ok |
| digits | alternating_rr | 0.9744 ± 0.0069 | 0.9744 ± 0.0075 | 0.6087 ± 0.0130 | 0.0 ± 0.7 | 0.6 ± 0.5 | 1.272 ± 0.058 | 0.021 ± 0.043 | empty_region, near_zero, ok |
| digits | alternating_rrr | 0.9794 ± 0.0058 | 0.9767 ± 0.0054 | 0.5918 ± 0.0456 | -1.0 ± 1.0 | 1.0 ± 1.0 | 1.469 ± 0.422 | 0.021 ± 0.043 | empty_region, near_zero, ok |
| digits | independent | 0.9833 ± 0.0052 | 0.9822 ± 0.0058 | 0.4758 ± 0.0034 | -0.4 ± 0.5 | 0.4 ± 0.5 | 3.447 ± 0.010 | 0.021 ± 0.043 | empty_region, near_zero, ok |
| iris | alternating_rr | 0.9600 ± 0.0548 | 0.9600 ± 0.0548 | 0.3057 ± 0.0195 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.568 ± 0.042 | 0.114 ± 0.030 | ok |
| iris | alternating_rrr | 0.9733 ± 0.0435 | 0.9733 ± 0.0435 | 0.2992 ± 0.0228 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.735 ± 0.095 | 0.114 ± 0.030 | ok |
| iris | independent | 0.9667 ± 0.0408 | 0.9667 ± 0.0408 | 0.2684 ± 0.0257 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.892 ± 0.055 | 0.114 ± 0.030 | ok |
| synthetic_diag | alternating_rr | 0.9262 ± 0.0761 | 0.9223 ± 0.0693 | 0.3176 ± 0.0413 | -1.0 ± 3.8 | 4.2 ± 3.4 | 0.511 ± 0.086 | 0.309 ± 0.083 | ok |
| synthetic_diag | alternating_rrr | 0.9392 ± 0.0585 | 0.9185 ± 0.0641 | 0.2876 ± 0.0350 | -5.4 ± 4.7 | 8.4 ± 6.1 | 0.715 ± 0.131 | 0.309 ± 0.083 | ok |
| synthetic_diag | independent | 0.8700 ± 0.1475 | 0.8638 ± 0.1264 | 0.2576 ± 0.0537 | -1.6 ± 6.7 | 12.4 ± 14.8 | 0.750 ± 0.105 | 0.309 ± 0.083 | ok |
| wine | alternating_rr | 0.9889 ± 0.0152 | 0.9889 ± 0.0152 | 0.3511 ± 0.0130 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.480 ± 0.029 | 0.046 ± 0.018 | ok |
| wine | alternating_rrr | 0.9944 ± 0.0124 | 0.9944 ± 0.0124 | 0.3033 ± 0.0067 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.812 ± 0.031 | 0.046 ± 0.018 | ok |
| wine | independent | 0.9833 ± 0.0248 | 0.9889 ± 0.0152 | 0.2979 ± 0.0137 | 0.2 ± 0.4 | 0.0 ± 0.0 | 0.791 ± 0.030 | 0.046 ± 0.018 | ok |
| zebra | alternating_rr | 0.8909 ± 0.2439 | 0.6636 ± 0.2170 | 0.2940 ± 0.0814 | -5.0 ± 4.9 | 5.0 ± 4.9 | 0.636 ± 0.418 | 0.779 ± 0.110 | ok |
| zebra | alternating_rrr | 0.5455 ± 0.0000 | 0.5545 ± 0.0932 | 0.1792 ± 0.0529 | 0.2 ± 2.0 | 3.0 ± 3.0 | 0.585 ± 0.319 | 0.779 ± 0.110 | ok |
| zebra | independent | 0.6182 ± 0.2170 | 0.5455 ± 0.0719 | 0.2045 ± 0.1171 | -1.6 ± 6.1 | 3.8 ± 4.7 | 0.661 ± 0.333 | 0.779 ± 0.110 | ok |
| zebra_special | alternating_rr | 0.7318 ± 0.0407 | 0.7136 ± 0.0570 | 0.1170 ± 0.0408 | -0.8 ± 1.3 | 1.6 ± 2.1 | 0.240 ± 0.034 | 0.382 ± 0.145 | ok |
| zebra_special | alternating_rrr | 0.7227 ± 0.0296 | 0.7136 ± 0.0345 | 0.1226 ± 0.0582 | -0.4 ± 0.5 | 1.0 ± 1.0 | 0.298 ± 0.044 | 0.382 ± 0.145 | ok |
| zebra_special | independent | 0.6955 ± 0.0443 | 0.6955 ± 0.0523 | 0.0933 ± 0.0432 | 0.0 ± 0.7 | 0.6 ± 0.9 | 0.295 ± 0.087 | 0.382 ± 0.145 | ok |

*Observations (this report):* ECE after adaptation is 0.09–0.61, far above the P2 logit-mode values. Only balance gains clearly (about +0.03 to +0.035), and zebra alternating_rr loses 0.23. On digits the free-radius calibration reports `empty_region` or `near_zero` for some seeds.

#### 1.3.7 P5: MCB on/off (Gate 6)
*Tests:* whether MCB (EMA memory encoder) changes representation, neighbourhood and selection stability, and at what accuracy cost. *Setup:* 4 datasets × MCB {0,1} × {random, trust_utility_coverage (tuc)}, K=0.5, maintenance at epochs 50/100/150, 5 seeds. It ran **before `ec6323a`**. The later plan changed the policy to provenance_only. Source: untracked `results/t1/p5_mcb/summary.csv`, `paired_vs_mcb0_random_K0.5.csv` and `paired_vs_mcb0_trust_utility_coverage_K0.5.csv`.

| dataset | condition | test_acc | repr_drift | nbr_churn | sel_churn |
|---|---|---|---|---|---|
| breast_cancer | mcb0_random | 0.9456 ± 0.0114 | 0.3261 ± 0.0796 | 0.6461 ± 0.0132 | 0.2491 ± 0.0126 |
| breast_cancer | mcb0_tuc | 0.9368 ± 0.0353 | 0.3314 ± 0.0638 | 0.6768 ± 0.0990 | 0.2380 ± 0.0337 |
| breast_cancer | mcb1_random | 0.9263 ± 0.0118 | 0.2017 ± 0.0313 | 0.5865 ± 0.0525 | 0.2491 ± 0.0126 |
| breast_cancer | mcb1_tuc | 0.9298 ± 0.0351 | 0.1760 ± 0.0299 | 0.6277 ± 0.0451 | 0.2579 ± 0.0148 |
| digits | mcb0_random | 0.6328 ± 0.3270 | 1.1668 ± 0.1860 | 0.5324 ± 0.0680 | 0.2486 ± 0.0090 |
| digits | mcb0_tuc | 0.5083 ± 0.3052 | 1.1876 ± 0.2173 | 0.5748 ± 0.0678 | 0.2527 ± 0.0125 |
| digits | mcb1_random | 0.9433 ± 0.0050 | 0.4653 ± 0.0415 | 0.6200 ± 0.0115 | 0.2486 ± 0.0090 |
| digits | mcb1_tuc | 0.9328 ± 0.0185 | 0.4034 ± 0.0328 | 0.5564 ± 0.0232 | 0.2523 ± 0.0084 |
| synthetic_diag | mcb0_random | 0.7592 ± 0.2486 | 1.0334 ± 0.3072 | 0.4125 ± 0.0339 | 0.2415 ± 0.0094 |
| synthetic_diag | mcb0_tuc | 0.8538 ± 0.0553 | 1.0118 ± 0.3090 | 0.3903 ± 0.0536 | 0.2521 ± 0.0042 |
| synthetic_diag | mcb1_random | 0.8554 ± 0.0574 | 0.0891 ± 0.0125 | 0.4543 ± 0.0280 | 0.2415 ± 0.0094 |
| synthetic_diag | mcb1_tuc | 0.8392 ± 0.0437 | 0.0630 ± 0.0091 | 0.3872 ± 0.0097 | 0.2355 ± 0.0163 |
| wine | mcb0_random | 0.9611 ± 0.0576 | 0.1314 ± 0.0268 | 0.3947 ± 0.0717 | 0.2463 ± 0.0140 |
| wine | mcb0_tuc | 0.9667 ± 0.0362 | 0.1144 ± 0.0208 | 0.3633 ± 0.0752 | 0.2519 ± 0.0330 |
| wine | mcb1_random | 0.9389 ± 0.0771 | 0.0947 ± 0.0193 | 0.3442 ± 0.0303 | 0.2463 ± 0.0140 |
| wine | mcb1_tuc | 0.9778 ± 0.0304 | 0.0769 ± 0.0107 | 0.3318 ± 0.0416 | 0.2426 ± 0.0190 |

Paired differences vs `mcb0_random_K0.5` (mean diff [95% CI], p, seeds better/worse; n=5):

| dataset | condition | test_acc | repr_drift | nbr_churn | sel_churn |
|---|---|---|---|---|---|
| breast_cancer | mcb0_tuc | -0.0088 [-0.0587, +0.0411] p=0.651 2/3 | +0.0053 [-0.0646, +0.0752] p=0.845 3/2 | +0.0306 [-0.1064, +0.1677] p=0.568 3/2 | -0.0111 [-0.0596, +0.0374] p=0.559 2/2 |
| breast_cancer | mcb1_random | -0.0193 [-0.0372, -0.0014] p=0.040 0/5 | -0.1245 [-0.1991, -0.0499] p=0.010 0/5 | -0.0596 [-0.1370, +0.0179] p=0.100 1/4 | 0 (identical) |
| breast_cancer | mcb1_tuc | -0.0158 [-0.0554, +0.0238] p=0.330 1/4 | -0.1501 [-0.2486, -0.0516] p=0.013 0/5 | -0.0184 [-0.0891, +0.0522] p=0.509 2/3 | +0.0088 [-0.0169, +0.0344] p=0.397 3/2 |
| digits | mcb0_tuc | -0.1244 [-0.6147, +0.3658] p=0.520 2/3 | +0.0208 [-0.3562, +0.3978] p=0.886 2/3 | +0.0424 [-0.0702, +0.1550] p=0.355 3/2 | +0.0041 [-0.0100, +0.0182] p=0.467 4/1 |
| digits | mcb1_random | +0.3106 [-0.0931, +0.7142] p=0.100 4/1 | -0.7015 [-0.9331, -0.4698] p=0.001 0/5 | +0.0877 [+0.0129, +0.1624] p=0.031 4/1 | 0 |
| digits | mcb1_tuc | +0.3000 [-0.0911, +0.6911] p=0.100 4/1 | -0.7634 [-0.9860, -0.5408] p=0.001 0/5 | +0.0240 [-0.0324, +0.0803] p=0.303 4/1 | +0.0037 [-0.0137, +0.0211] p=0.586 3/2 |
| synthetic_diag | mcb0_tuc | +0.0946 [-0.2045, +0.3938] p=0.429 2/3 | -0.0215 [-0.1261, +0.0830] p=0.598 2/3 | -0.0221 [-0.1099, +0.0656] p=0.522 1/4 | +0.0106 [-0.0018, +0.0230] p=0.077 4/1 |
| synthetic_diag | mcb1_random | +0.0962 [-0.2337, +0.4260] p=0.464 2/3 | -0.9443 [-1.3183, -0.5702] p=0.002 0/5 | +0.0418 [+0.0045, +0.0791] p=0.036 5/0 | 0 |
| synthetic_diag | mcb1_tuc | +0.0800 [-0.2347, +0.3947] p=0.519 2/3 | -0.9703 [-1.3423, -0.5984] p=0.002 0/5 | -0.0253 [-0.0598, +0.0092] p=0.112 1/4 | -0.0060 [-0.0313, +0.0193] p=0.547 1/4 |
| wine | mcb0_tuc | +0.0056 [-0.0902, +0.1013] p=0.880 3/2 | -0.0170 [-0.0670, +0.0330] p=0.398 2/3 | -0.0315 [-0.1090, +0.0461] p=0.323 2/3 | +0.0056 [-0.0528, +0.0640] p=0.805 2/3 |
| wine | mcb1_random | -0.0222 [-0.0511, +0.0066] p=0.099 0/3 | -0.0366 [-0.0580, -0.0153] p=0.009 0/5 | -0.0505 [-0.1161, +0.0151] p=0.099 2/3 | 0 |
| wine | mcb1_tuc | +0.0167 [-0.0356, +0.0690] p=0.426 2/1 | -0.0545 [-0.0995, -0.0094] p=0.028 0/5 | -0.0630 [-0.1091, -0.0169] p=0.019 0/5 | -0.0037 [-0.0388, +0.0314] p=0.784 2/2 |

Paired differences vs `mcb0_trust_utility_coverage_K0.5` (MCB-relevant rows; the mcb0_random row on breast_cancer/digits/wine mirrors the table above):

| dataset | condition | test_acc | repr_drift | nbr_churn | sel_churn | rare acc |
|---|---|---|---|---|---|---|
| breast_cancer | mcb1_random | -0.0105 [-0.0462, +0.0251] p=0.458 1/2 | -0.1297 [-0.2105, -0.0490] p=0.011 0/5 | -0.0902 [-0.1733, -0.0072] p=0.039 1/4 | +0.0111 [-0.0374, +0.0596] p=0.559 | n/a |
| breast_cancer | mcb1_tuc | -0.0070 [-0.0435, +0.0294] p=0.621 2/2 | -0.1553 [-0.2521, -0.0586] p=0.011 0/5 | -0.0491 [-0.1322, +0.0341] p=0.177 1/4 | +0.0199 [-0.0219, +0.0616] p=0.257 3/2 | n/a |
| digits | mcb1_random | +0.4350 [+0.0548, +0.8152] p=0.034 4/1 | -0.7223 [-0.9600, -0.4845] p=0.001 0/5 | +0.0452 [-0.0424, +0.1328] p=0.225 4/1 | -0.0041 [-0.0182, +0.0100] p=0.467 1/4 | n/a |
| digits | mcb1_tuc | +0.4244 [+0.0376, +0.8113] p=0.038 4/1 | -0.7842 [-1.0503, -0.5181] p=0.001 0/5 | -0.0184 [-0.1084, +0.0715] p=0.599 2/3 | -0.0004 [-0.0142, +0.0134] p=0.944 2/3 | n/a |
| synthetic_diag | mcb0_random | -0.0946 [-0.3938, +0.2045] p=0.429 3/2 | +0.0215 … p=0.598 | +0.0221 … p=0.522 | -0.0106 … p=0.077 | -0.1400 [-0.4950, +0.2150] p=0.335 0/2 |
| synthetic_diag | mcb1_random | +0.0015 [-0.0403, +0.0434] p=0.924 2/3 | -0.9227 [-1.2976, -0.5479] p=0.002 0/5 | +0.0640 [-0.0051, +0.1330] p=0.062 4/1 | -0.0106 [-0.0230, +0.0018] p=0.077 1/4 | 0 |
| synthetic_diag | mcb1_tuc | -0.0146 [-0.0530, +0.0238] p=0.350 3/2 | -0.9488 [-1.3232, -0.5744] p=0.002 0/5 | -0.0031 [-0.0732, +0.0670] p=0.907 3/2 | -0.0166 [-0.0408, +0.0076] p=0.130 1/4 | 0 |
| wine | mcb1_random | -0.0278 [-0.1422, +0.0866] p=0.537 2/3 | -0.0196 [-0.0652, +0.0260] p=0.298 1/4 | -0.0190 [-0.1200, +0.0819] p=0.628 3/2 | -0.0056 [-0.0640, +0.0528] p=0.805 3/2 | n/a |
| wine | mcb1_tuc | +0.0111 [-0.0604, +0.0826] p=0.688 2/2 | -0.0375 [-0.0568, -0.0181] p=0.006 0/5 | -0.0315 [-0.1179, +0.0548] p=0.368 2/3 | -0.0093 [-0.0456, +0.0271] p=0.519 1/3 | n/a |

*Observations (this report):* MCB lowers representation drift in all 5 seeds on all 4 datasets. Neighbourhood churn moves in mixed directions. On breast_cancer, MCB costs accuracy (mcb1_random −0.0193, p=0.040, 0/5 seeds better). The digits gain for MCB-on (+0.30 to +0.44) comes mainly from MCB-off runs collapsing (0.6328 ± 0.3270 and 0.5083 ± 0.3052), which is probably the fresh-Adam instability fixed in `ec6323a` (inferred, not verified).

#### 1.3.8 T1.2: revise harness
*Tests:* which flagging signal finds corrupted cases at a matched review budget, and accuracy at M0 (before correction), M1 (immediately after) and M2 (after retraining). *Setup:* `synthetic_diag`, simulated oracle reviewer, budgets b10–b75, 10 seeds; continued optimizer, before `6adeb30`. Source: `results/t1/t12_revise_synthetic/runs__synthetic_diag.jsonl` (computed for this report). Collateral correct→wrong at M1 was 0.0 for every condition, so that column is dropped.

| condition | precision | recall | M0 acc | M1 acc | M2 acc | M0→M1 w→c | M0→M1 c→w | M0→M2 w→c | M0→M2 c→w |
|---|---|---|---|---|---|---|---|---|---|
| bias_only_b10 | 0.030 ± 0.048 | 0.006 ± 0.009 | 0.9435 ± 0.0532 | 0.9435 ± 0.0532 | 0.8977 ± 0.1103 | 0.0 | 0.0 | 2.7 ± 2.8 | 14.6 ± 22.4 |
| bias_only_b25 | 0.036 ± 0.035 | 0.017 ± 0.016 | 0.9435 ± 0.0532 | 0.9435 ± 0.0532 | 0.8942 ± 0.1203 | 0.0 | 0.0 | 3.9 ± 3.5 | 16.7 ± 27.4 |
| bias_only_b50 | 0.042 ± 0.026 | 0.039 ± 0.024 | 0.9435 ± 0.0532 | 0.9446 ± 0.0503 | 0.9281 ± 0.0834 | 0.3 ± 0.9 | 0.0 | 3.2 ± 2.7 | 7.2 ± 10.7 |
| bias_only_b75 | 0.071 ± 0.048 | 0.098 ± 0.067 | 0.9435 ± 0.0532 | 0.9450 ± 0.0502 | 0.9258 ± 0.0867 | 0.4 ± 1.0 | 0.0 | 3.0 ± 2.3 | 7.6 ± 11.1 |
| combined_T_b10 | 0.040 ± 0.052 | 0.007 ± 0.010 | 0.9435 ± 0.0532 | 0.9435 ± 0.0532 | 0.9246 ± 0.0859 | 0.0 | 0.0 | 3.0 ± 2.9 | 7.9 ± 11.9 |
| combined_T_b25 | 0.056 ± 0.039 | 0.026 ± 0.018 | 0.9435 ± 0.0532 | 0.9435 ± 0.0532 | 0.9277 ± 0.0853 | 0.0 | 0.0 | 3.0 ± 3.0 | 7.1 ± 11.4 |
| combined_T_b50 | 0.060 ± 0.030 | 0.056 ± 0.028 | 0.9435 ± 0.0532 | 0.9446 ± 0.0503 | 0.9273 ± 0.0820 | 0.3 ± 0.9 | 0.0 | 3.2 ± 3.3 | 7.4 ± 10.2 |
| combined_T_b75 | 0.080 ± 0.052 | 0.111 ± 0.072 | 0.9435 ± 0.0532 | 0.9454 ± 0.0501 | 0.9273 ± 0.0857 | 0.5 ± 1.1 | 0.0 | 2.8 ± 2.5 | 7.0 ± 10.8 |
| oracle_b10 | 1.000 | 0.185 | 0.9435 ± 0.0532 | 0.9450 ± 0.0505 | 0.9412 ± 0.0653 | 0.4 ± 0.8 | 0.0 | 3.9 ± 3.7 | 4.5 ± 6.0 |
| oracle_b25 | 1.000 | 0.463 | 0.9435 ± 0.0532 | 0.9512 ± 0.0421 | 0.9465 ± 0.0512 | 2.0 ± 4.1 | 0.0 | 5.3 ± 4.4 | 4.5 ± 4.3 |
| oracle_b50 | 1.000 | 0.926 | 0.9435 ± 0.0532 | 0.9550 ± 0.0400 | 0.9558 ± 0.0541 | 3.1 ± 4.5 | 0.1 ± 0.3 | 6.4 ± 4.6 | 3.2 ± 4.6 |
| oracle_b75 | 0.720 | 1.000 | 0.9435 ± 0.0532 | 0.9550 ± 0.0400 | 0.9542 ± 0.0584 | 3.1 ± 4.5 | 0.1 ± 0.3 | 5.7 ± 4.1 | 2.9 ± 5.2 |
| provenance_only_b10 | 0.650 ± 0.222 | 0.120 ± 0.041 | 0.9435 ± 0.0532 | 0.9477 ± 0.0532 | 0.9254 ± 0.0787 | 1.1 ± 1.3 | 0.0 | 3.5 ± 2.7 | 8.2 ± 11.4 |
| provenance_only_b25 | 0.464 ± 0.141 | 0.215 ± 0.065 | 0.9435 ± 0.0532 | 0.9496 ± 0.0496 | 0.9477 ± 0.0623 | 1.6 ± 1.8 | 0.0 | 5.2 ± 3.6 | 4.1 ± 5.3 |
| provenance_only_b50 | 0.392 ± 0.057 | 0.363 ± 0.053 | 0.9435 ± 0.0532 | 0.9500 ± 0.0487 | 0.9446 ± 0.0639 | 1.7 ± 2.1 | 0.0 | 4.8 ± 3.4 | 4.5 ± 6.1 |
| provenance_only_b75 | 0.361 ± 0.047 | 0.502 ± 0.066 | 0.9435 ± 0.0532 | 0.9512 ± 0.0460 | 0.9496 ± 0.0555 | 2.0 ± 2.8 | 0.0 | 4.6 ± 4.2 | 3.0 ± 4.5 |
| random_b10 | 0.160 ± 0.143 | 0.030 ± 0.026 | 0.9435 ± 0.0532 | 0.9458 ± 0.0510 | 0.9285 ± 0.0887 | 0.6 ± 0.8 | 0.0 | 3.3 ± 3.0 | 7.2 ± 11.6 |
| random_b25 | 0.132 ± 0.071 | 0.061 ± 0.033 | 0.9435 ± 0.0532 | 0.9462 ± 0.0509 | 0.9296 ± 0.0866 | 0.7 ± 0.8 | 0.0 | 3.8 ± 3.3 | 7.4 ± 12.2 |
| random_b50 | 0.150 ± 0.058 | 0.139 ± 0.053 | 0.9435 ± 0.0532 | 0.9477 ± 0.0490 | 0.9312 ± 0.0828 | 1.1 ± 1.3 | 0.0 | 3.7 ± 3.7 | 6.9 ± 10.7 |
| random_b75 | 0.131 ± 0.040 | 0.181 ± 0.056 | 0.9435 ± 0.0532 | 0.9473 ± 0.0488 | 0.9350 ± 0.0847 | 1.1 ± 1.3 | 0.1 ± 0.3 | 4.2 ± 3.6 | 6.4 ± 11.5 |

*Observations (this report):* provenance_only is the only non-oracle signal well above random in precision (0.650 at b10). bias_only and combined_T are below random. M2 retraining is lower than M1 in most cells, with large correct→wrong counts; this retraining path predates the `6adeb30` fix.

#### 1.3.9 Invalidated: `results/t1_invalid_fresh_adam/`
This gitignored directory holds the first P1/P4/T1.2 pilot runs, made with a fresh Adam after core training. **Its own name marks it invalid, and `ec6323a` superseded it.** It is not reported here as evidence; the authors left no write-up.

#### 1.3.10 Superseded -era results (for record only)

> **SUPERSEDED / REMOVED by `17cbae6`.** The reports had hard-coded conclusions that the data contradict, and the benchmarks never trained the retriever. Everything below was recovered from git history. The directories (`results/iclr_paper`, `results/iclr_test_run`, `results/t0_regression`, `results/t0_rl`, `results/t1_schedules`) no longer exist in the working tree. None of this is evidence of progress.

##### (a) T0 battery (context; predates the reported commits)
*Setup:* retention policies at a 50% budget, then the NN-CDH adapter; 180 runs, seeds 42–44. Source: `git show 17cbae6^:results/t0_battery/T0_CBR_EXPERIMENT_REPORT.md`. `results/t0_battery/` is gitignored, and only per-run folders remain in it.

| dataset | bias_only | provenance_bias_coverage | provenance_only | stratified | trustworthiness_only |
|:--|:--|:--|:--|:--|:--|
| breast_cancer | 0.9630 ± 0.0135 | 0.9669 ± 0.0179 | 0.9474 ± 0.0175 | 0.9630 ± 0.0089 | 0.9649 ± 0.0175 |
| iris | 0.9185 ± 0.0257 | 0.9185 ± 0.0128 | 0.9037 ± 0.0339 | 0.9111 ± 0.0222 | 0.9111 ± 0.0222 |
| synthetic | 0.8204 ± 0.0370 | 0.8259 ± 0.0225 | 0.8148 ± 0.0225 | 0.8296 ± 0.0410 | 0.8111 ± 0.0389 |
| wine | 0.9321 ± 0.0385 | 0.9444 ± 0.0185 | 0.9506 ± 0.0283 | 0.9506 ± 0.0283 | 0.9444 ± 0.0185 |

| dataset | adapter_mode | retrieval_acc | adapted_acc | acc_delta | correct_flips | harmful_flips | net_benefit |
|:--|:--|--:|--:|--:|--:|--:|--:|
| breast_cancer | logit_residual | 0.966862 | 0.961014 | -0.00584795 | 0.333333 | 1.33333 | -1 |
| breast_cancer | nominal_residual_scores | 0.966862 | 0.966862 | 0 | 1.66667 | 1.66667 | 0 |
| iris | logit_residual | 0.918519 | 0.940741 | 0.0222222 | 1 | 0 | 1 |
| iris | nominal_residual_scores | 0.918519 | 0.940741 | 0.0222222 | 1 | 0 | 1 |
| synthetic | logit_residual | 0.825926 | 0.838889 | 0.0129629 | 4.33333 | 2 | 2.33333 |
| synthetic | nominal_residual_scores | 0.825926 | 0.831481 | 0.00555555 | 7.33333 | 6.33333 | 1 |
| wine | logit_residual | 0.944444 | 0.962963 | 0.0185185 | 1 | 0 | 1 |
| wine | nominal_residual_scores | 0.944444 | 0.962963 | 0.0185185 | 1 | 0 | 1 |

(The `none` rows equal retrieval_acc with zero flips.) The authors claimed provenance_bias_coverage gives "superior preservation" and that flips show "net positive benefit". The tables do not support this: provenance_bias_coverage is not best on iris (tie), synthetic or wine, and breast_cancer logit has net −1.

##### (b) ICLR suite (`052699c`, 5 seeds 42–46)
Source: `git show 052699c:results/iclr_paper/ICLR_EXPERIMENTAL_RESULTS.md` and the raw CSVs. Covertype is a 3,000-row subsample with the scaler fit before the split.

Main benchmark, 50% capacity:

| Dataset | Random | Bias-Only | DROP3 | ICF | Core-Set | Ours (Retain) | Ours (Retain+Reuse) | p vs Bias | p vs DROP3 |
|:--|:--|:--|:--|:--|:--|:--|:--|--:|--:|
| breast_cancer | 0.9614 ± 0.0070 | 0.9649 ± 0.0091 | 0.9532 ± 0.0123 | 0.9649 ± 0.0098 | 0.9602 ± 0.0183 | 0.9661 ± 0.0119 | **0.9673 ± 0.0060** | 0.704 | 0.1084 |
| covtype | 0.6673 ± 0.0181 | 0.6676 ± 0.0121 | 0.6584 ± 0.0176 | 0.6711 ± 0.0181 | 0.6789 ± 0.0080 | 0.6638 ± 0.0089 | **0.6989 ± 0.0090** | 0.1502 | 0.6346 |
| digits | 0.9637 ± 0.0099 | 0.9593 ± 0.0056 | 0.9559 ± 0.0097 | 0.9637 ± 0.0048 | **0.9696 ± 0.0049** | 0.9615 ± 0.0059 | 0.9563 ± 0.0048 | 0.4581 | 0.3171 |
| iris | 0.9022 ± 0.0227 | 0.9244 ± 0.0178 | 0.6933 ± 0.0788 | 0.9467 ± 0.0227 | **0.9511 ± 0.0295** | 0.9156 ± 0.0166 | 0.9378 ± 0.0259 | 0.4766 | 0.0059 |
| synthetic | 0.8022 ± 0.0424 | 0.8267 ± 0.0262 | 0.8211 ± 0.0334 | 0.8211 ± 0.0252 | **0.8311 ± 0.0240** | 0.8211 ± 0.0247 | 0.8156 ± 0.0244 | 0.6004 | 1 |
| wine | 0.9481 ± 0.0181 | 0.9370 ± 0.0343 | 0.9370 ± 0.0251 | 0.9519 ± 0.0381 | **0.9667 ± 0.0216** | 0.9481 ± 0.0139 | 0.9630 ± 0.0203 | 0.3739 | 0.208 |

Revise under 15% label noise:

| dataset | acc_noisy | acc_revised | recovery_delta | precision | recall | f1 |
|:--|--:|--:|--:|--:|--:|--:|
| breast_cancer | 0.935673 | 0.953216 | 0.0175439 | 0.836934 | 0.78 | 0.806716 |
| iris | 0.933333 | 0.928889 | -0.00444444 | 0.897857 | 0.75 | 0.812772 |
| synthetic | 0.802222 | 0.804444 | 0.00222224 | 0.648982 | 0.603175 | 0.621995 |
| wine | 0.940741 | 0.937037 | -0.00370371 | 0.912646 | 0.8 | 0.851098 |

Explanation faithfulness, top-1 counterfactual removal:

| dataset | retrieval_acc | mean_conf_drop | flip_rate | top1_sufficiency |
|:--|--:|--:|--:|--:|
| breast_cancer | 0.967251 | 0.00892387 | 0.0105263 | 0.950877 |
| digits | 0.971481 | 0.0178325 | 0.0162963 | 0.972222 |
| iris | 0.951111 | 0.0120879 | 0.0355556 | 0.933333 |
| synthetic | 0.845556 | 0.0427673 | 0.0933333 | 0.796667 |
| wine | 0.940741 | 0.0143302 | 0.0333333 | 0.940741 |

Capacity frontier:

| dataset | capacity_fraction | bias_only | drop3 | provenance_bias_coverage | stratified |
|:--|--:|:--|:--|:--|:--|
| breast_cancer | 0.2 | 0.9497 ± 0.0173 | 0.9462 ± 0.0172 | 0.9380 ± 0.0225 | 0.9520 ± 0.0191 |
| breast_cancer | 0.33 | 0.9567 ± 0.0121 | 0.9497 ± 0.0147 | 0.9520 ± 0.0096 | 0.9544 ± 0.0146 |
| breast_cancer | 0.5 | 0.9649 ± 0.0101 | 0.9532 ± 0.0137 | 0.9661 ± 0.0133 | 0.9614 ± 0.0078 |
| breast_cancer | 0.75 | 0.9661 ± 0.0140 | 0.9591 ± 0.0170 | 0.9661 ± 0.0105 | 0.9649 ± 0.0149 |
| breast_cancer | 1 | 0.9673 ± 0.0114 | 0.9673 ± 0.0114 | 0.9673 ± 0.0114 | 0.9673 ± 0.0114 |
| synthetic | 0.2 | 0.6833 ± 0.0222 | 0.7344 ± 0.0474 | 0.7244 ± 0.0346 | 0.7200 ± 0.0433 |
| synthetic | 0.33 | 0.7689 ± 0.0461 | 0.7956 ± 0.0361 | 0.7733 ± 0.0427 | 0.7711 ± 0.0265 |
| synthetic | 0.5 | 0.8267 ± 0.0292 | 0.8211 ± 0.0374 | 0.8211 ± 0.0276 | 0.8022 ± 0.0474 |
| synthetic | 0.75 | 0.8367 ± 0.0357 | 0.8333 ± 0.0342 | 0.8389 ± 0.0429 | 0.8389 ± 0.0322 |
| synthetic | 1 | 0.8456 ± 0.0369 | 0.8456 ± 0.0369 | 0.8456 ± 0.0369 | 0.8456 ± 0.0369 |
| wine | 0.2 | 0.9222 ± 0.0562 | 0.9185 ± 0.0649 | 0.9630 ± 0.0227 | 0.9259 ± 0.0293 |
| wine | 0.33 | 0.9444 ± 0.0434 | 0.9519 ± 0.0336 | 0.9519 ± 0.0281 | 0.9407 ± 0.0241 |
| wine | 0.5 | 0.9370 ± 0.0384 | 0.9370 ± 0.0281 | 0.9481 ± 0.0155 | 0.9481 ± 0.0203 |
| wine | 0.75 | 0.9481 ± 0.0275 | 0.9481 ± 0.0203 | 0.9556 ± 0.0281 | 0.9519 ± 0.0248 |
| wine | 1 | 0.9407 ± 0.0241 | 0.9407 ± 0.0241 | 0.9407 ± 0.0241 | 0.9407 ± 0.0241 |

The next two tables are **earlier T0 battery results (`26d0e59`) that the ICLR suite report carries over unchanged**. `052699c` did not touch `retention_ablations_raw.csv` or `reuse_ablations_raw.csv`.

Retention component ablation (synthetic, 50%):

| ablation_type | ablation_var | accuracy | accuracy_std | f1 | f1_std | ece |
|:--|:--|--:|--:|--:|--:|--:|
| alpha_weight | alpha=0.0 | 0.82 | 0.0392051 | 0.818395 | 0.0398546 | 0.149671 |
| alpha_weight | alpha=0.25 | 0.82 | 0.0310813 | 0.818285 | 0.0315746 | 0.147793 |
| alpha_weight | alpha=0.5 | 0.821111 | 0.0276106 | 0.819271 | 0.0274766 | 0.141118 |
| alpha_weight | alpha=0.75 | 0.823333 | 0.0453723 | 0.821437 | 0.0454388 | 0.145181 |
| alpha_weight | alpha=1.0 | 0.785556 | 0.0427814 | 0.78311 | 0.0434243 | 0.0729298 |
| protection_floor | protect_cohorts=False | 0.817778 | 0.0227031 | 0.815973 | 0.0223832 | 0.139033 |
| protection_floor | protect_cohorts=True | 0.821111 | 0.0276106 | 0.819271 | 0.0274766 | 0.141118 |
| trust_formulation | arithmetic | 0.818889 | 0.0256399 | 0.817082 | 0.0252311 | 0.140224 |
| trust_formulation | geometric | 0.821111 | 0.0276106 | 0.819271 | 0.0274766 | 0.141118 |

Reuse adapter modes:

| dataset | adapter_mode | retrieval_acc | adapted_acc | acc_delta | f1_adapted | ece | correct_flips | harmful_flips | net_benefit |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| breast_cancer | logit_residual | 0.966082 | 0.964912 | -0.00116959 | 0.962176 | 0.0297807 | 0.6 | 0.8 | -0.2 |
| breast_cancer | nominal_residual_scores | 0.966082 | 0.967251 | 0.0011696 | 0.964827 | 0.196989 | 1.4 | 1.2 | 0.2 |
| breast_cancer | none | 0.966082 | 0.966082 | 0 | 0.9633 | 0.0332178 | 0 | 0 | 0 |
| synthetic | logit_residual | 0.821111 | 0.826667 | 0.00555553 | 0.824592 | 0.130951 | 4.4 | 3.4 | 1 |
| synthetic | nominal_residual_scores | 0.821111 | 0.815556 | -0.00555556 | 0.813825 | 0.281501 | 8.6 | 9.6 | -1 |
| synthetic | none | 0.821111 | 0.821111 | 0 | 0.819271 | 0.141118 | 0 | 0 | 0 |
| wine | logit_residual | 0.948148 | 0.959259 | 0.0111111 | 0.959749 | 0.0478966 | 0.6 | 0 | 0.6 |
| wine | nominal_residual_scores | 0.948148 | 0.962963 | 0.0148148 | 0.963345 | 0.34997 | 0.8 | 0 | 0.8 |
| wine | none | 0.948148 | 0.948148 | 0 | 0.948625 | 0.0470874 | 0 | 0 | 0 |

*Hard-coded claims vs data:*
- The report claims "matching or superior performance" against DROP3/ICF/CoreSet "across all 6 benchmarks". In fact CoreSet has the best mean on 4 of 6, Retain is below ICF or CoreSet on 5 of 6, and only iris vs DROP3 has p < 0.05.
- It claims Revise recovers "up to +5.3%". The maximum is +0.0175, and recovery is negative on iris and wine.
- It claims a top-1 confidence drop of "0.20–0.35". The measured range is 0.0089–0.0428.
- It claims Covertype "scales gracefully". The Covertype data is a 3,000-row subsample, and Retain (0.6638) is below Random (0.6673).

##### (c) ICLR suite test run (N=1, seed 42; committed in `f3ac443`)
A smoke run. Every ± value is 0.0000 or nan, every p is 1, and the report repeats the same hard-coded findings. Source: `git show f3ac443:results/iclr_test_run/ICLR_EXPERIMENTAL_RESULTS.md`.

| Dataset | Random | Bias-Only | DROP3 | ICF | Core-Set | Ours (Retain) | Ours (Retain+Reuse) |
|:--|--:|--:|--:|--:|--:|--:|--:|
| breast_cancer | 0.9649 | 0.9708 | 0.9415 | 0.9649 | 0.9708 | 0.9708 | 0.9591 |
| covtype | 0.6567 | 0.6533 | 0.6733 | 0.6667 | 0.6833 | 0.6500 | 0.7044 |
| digits | 0.9574 | 0.9556 | 0.9481 | 0.9667 | 0.9704 | 0.9556 | 0.9519 |
| iris | 0.8889 | 0.8889 | 0.6222 | 0.9556 | 0.9111 | 0.9111 | 0.9111 |
| synthetic | 0.7833 | 0.8111 | 0.8278 | 0.7833 | 0.8611 | 0.8222 | 0.8056 |
| wine | 0.9444 | 0.9630 | 0.9630 | 0.9815 | 1.0000 | 0.9630 | 0.9630 |

| dataset | acc_noisy | acc_revised | recovery_delta | precision | recall | f1 |
|:--|--:|--:|--:|--:|--:|--:|
| breast_cancer | 0.94152 | 0.953216 | 0.0116959 | 0.903846 | 0.783333 | 0.839286 |
| iris | 0.911111 | 0.933333 | 0.0222222 | 0.9 | 0.5625 | 0.692308 |
| synthetic | 0.811111 | 0.8 | -0.0111111 | 0.578947 | 0.698413 | 0.633094 |
| wine | 0.944444 | 0.944444 | 0 | 0.888889 | 0.842105 | 0.864865 |

| dataset | retrieval_acc | mean_conf_drop | flip_rate | top1_sufficiency |
|:--|--:|--:|--:|--:|
| breast_cancer | 0.97076 | 0.0111944 | 0.00584795 | 0.959064 |
| digits | 0.972222 | 0.0175251 | 0.0185185 | 0.972222 |
| iris | 0.955556 | 0.0138457 | 0.0444444 | 0.933333 |
| synthetic | 0.833333 | 0.0419431 | 0.0833333 | 0.833333 |
| wine | 0.962963 | 0.0140543 | 0 | 0.962963 |

| dataset | frac | bias_only | drop3 | prov_bias_cov | stratified |
|:--|--:|--:|--:|--:|--:|
| breast_cancer | 0.2 | 0.9240 | 0.9357 | 0.9298 | 0.9415 |
| breast_cancer | 0.33 | 0.9474 | 0.9357 | 0.9474 | 0.9357 |
| breast_cancer | 0.5 | 0.9708 | 0.9415 | 0.9708 | 0.9649 |
| breast_cancer | 0.75 | 0.9649 | 0.9474 | 0.9649 | 0.9649 |
| breast_cancer | 1 | 0.9708 | 0.9708 | 0.9708 | 0.9708 |
| synthetic | 0.2 | 0.6944 | 0.6722 | 0.6944 | 0.7167 |
| synthetic | 0.33 | 0.7500 | 0.7944 | 0.7444 | 0.7778 |
| synthetic | 0.5 | 0.8111 | 0.8278 | 0.8222 | 0.7833 |
| synthetic | 0.75 | 0.8500 | 0.8333 | 0.8500 | 0.8333 |
| synthetic | 1 | 0.8333 | 0.8333 | 0.8333 | 0.8333 |
| wine | 0.2 | 0.9815 | 0.9074 | 0.9815 | 0.9444 |
| wine | 0.33 | 0.9630 | 0.9074 | 0.9815 | 0.9630 |
| wine | 0.5 | 0.9630 | 0.9630 | 0.9630 | 0.9444 |
| wine | 0.75 | 0.9630 | 0.9630 | 0.9630 | 0.9630 |
| wine | 1 | 0.9630 | 0.9630 | 0.9630 | 0.9630 |

| ablation_type | var | accuracy | f1 | ece |
|:--|:--|--:|--:|--:|
| alpha_weight | alpha=0.0 | 0.805556 | 0.798595 | 0.12293 |
| alpha_weight | alpha=0.25 | 0.811111 | 0.803711 | 0.125168 |
| alpha_weight | alpha=0.5 | 0.822222 | 0.814337 | 0.128018 |
| alpha_weight | alpha=0.75 | 0.811111 | 0.806155 | 0.125552 |
| alpha_weight | alpha=1.0 | 0.816667 | 0.812455 | 0.0671645 |
| protection_floor | protect_cohorts=False | 0.822222 | 0.815155 | 0.13254 |
| protection_floor | protect_cohorts=True | 0.822222 | 0.814337 | 0.128018 |
| trust_formulation | arithmetic | 0.822222 | 0.815423 | 0.129878 |
| trust_formulation | geometric | 0.822222 | 0.814337 | 0.128018 |

| dataset | mode | retrieval_acc | adapted_acc | acc_delta | f1 | ece | correct | harmful | net |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| breast_cancer | logit_residual | 0.97076 | 0.959064 | -0.0116959 | 0.955871 | 0.028784 | 0 | 2 | -2 |
| breast_cancer | nominal_residual_scores | 0.97076 | 0.959064 | -0.0116959 | 0.955871 | 0.197709 | 1 | 3 | -2 |
| breast_cancer | none | 0.97076 | 0.97076 | 0 | 0.968259 | 0.0362169 | 0 | 0 | 0 |
| synthetic | logit_residual | 0.822222 | 0.816667 | -0.00555557 | 0.808887 | 0.0953609 | 1 | 2 | -1 |
| synthetic | nominal_residual_scores | 0.822222 | 0.805556 | -0.0166667 | 0.799494 | 0.249346 | 5 | 8 | -3 |
| synthetic | none | 0.822222 | 0.822222 | 0 | 0.814337 | 0.128018 | 0 | 0 | 0 |
| wine | logit_residual | 0.962963 | 0.962963 | 0 | 0.961905 | 0.0413387 | 0 | 0 | 0 |
| wine | nominal_residual_scores | 0.962963 | 0.962963 | 0 | 0.961905 | 0.371439 | 0 | 0 | 0 |
| wine | none | 0.962963 | 0.962963 | 0 | 0.961905 | 0.0632273 | 0 | 0 | 0 |

##### (d)  T1 matched-budget matrix (`cc3f67a`)
The handoff describes "168 conditions" on iris and synthetic with seeds 42/43; only this excerpt was published. Source: `git show cc3f67a:docs/T1_IMPLEMENTATION_HANDOFF.md` §7.

| Dataset | Retention | Adapter | MCB | Pre-Acc | Post-Acc | Net flips |
|---|---|---|---|---|---|---|
| iris | full_memory | none | OFF | 0.921 ± 0.053 | 0.921 ± 0.053 | +0.0 |
| iris | full_memory | nominal_residual_scores | OFF | 0.921 ± 0.053 | 0.974 ± 0.026 | +2.0 |
| iris | full_memory | nominal_residual_scores | ON | 0.882 ± 0.039 | 0.974 ± 0.000 | +3.5 |
| iris | bias_only | none | OFF | 0.776 ± 0.039 | 0.776 ± 0.039 | +0.0 |
| iris | bias_only | logit_residual | OFF | 0.776 ± 0.039 | 0.868 ± 0.053 | +3.5 |
| iris | provenance_bias_coverage | nominal_residual_scores | ON | 0.368 ± 0.026 | 0.750 ± 0.066 | +14.5 |
| iris | provenance_bias_coverage | logit_residual | ON | 0.368 ± 0.026 | 0.803 ± 0.039 | +16.5 |
| synthetic | full_memory | nominal_residual_scores | OFF | 0.767 ± 0.047 | 0.800 ± 0.027 | +2.5 |
| synthetic | full_memory | logit_residual | OFF | 0.767 ± 0.047 | 0.813 ± 0.013 | +3.5 |
| synthetic | bias_only | none | OFF | 0.633 ± 0.020 | 0.633 ± 0.020 | +0.0 |
| synthetic | bias_only | logit_residual | OFF | 0.633 ± 0.020 | 0.687 ± 0.020 | +4.0 |
| synthetic | bias_only | logit_residual | ON | 0.407 ± 0.073 | 0.593 ± 0.007 | +14.0 |
| synthetic | provenance_bias_coverage | nominal_residual_scores | ON | 0.433 ± 0.100 | 0.580 ± 0.020 | +11.0 |
| synthetic | provenance_bias_coverage | logit_residual | ON | 0.447 ± 0.113 | 0.673 ± 0.007 | +17.0 |
| synthetic | random | nominal_residual_scores | OFF | 0.633 ± 0.007 | 0.540 ± 0.020 | −7.0 |

The authors claimed large adapter gains and that MCB "stabilized reference geometry". The "large net flips" are recovery from a broken retriever (pre-adaptation accuracy 0.368 on iris). This is consistent with the aliasing bug fixed in `17cbae6`, but that link is not verified.

##### (e) Scheduler benchmark, 5 schedules (`c7467b6`)
*Setup:* MCB on, 30 epochs, K=20 on iris and K=40 on wine/synthetic, seeds 42/43. The table gives means over seeds, built by this report from `git show c7467b6:results/t1_schedules/schedule_comparison_summary.csv`.

| dataset | schedule | best_pre_acc | best_post_acc | final_pre_acc | final_post_acc | final_net_flips | tau_task | final_rep_drift |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| iris | staged_sequential | 0.9474 | 0.9474 | 0.8816 | 0.9474 | 2.5 | 0.0000 | 0.0000 |
| iris | alternating | 0.8026 | 0.8684 | 0.6974 | 0.8421 | 5.5 | 0.3500 | 0.0000 |
| iris | joint_synchronized | 0.8553 | 0.8684 | 0.3289 | 0.8289 | 19.0 | 0.3444 | 0.0001 |
| iris | warmup_alternating | 0.7895 | 0.8026 | 0.4868 | 0.7763 | 11.0 | 0.4072 | 0.0000 |
| iris | warmup_joint | 0.6711 | 0.7632 | 0.3289 | 0.7368 | 15.5 | 0.2272 | 0.0000 |
| wine | staged_sequential | 0.9889 | 0.9889 | 0.9778 | 0.9667 | -0.5 | 0.0000 | 0.0000 |
| wine | alternating | 0.8889 | 0.9000 | 0.5333 | 0.8333 | 13.5 | 0.4628 | 0.0000 |
| wine | joint_synchronized | 0.6667 | 0.8667 | 0.4222 | 0.8111 | 17.5 | 0.3386 | 0.0000 |
| wine | warmup_alternating | 0.5889 | 0.8556 | 0.4667 | 0.8333 | 16.5 | 0.3211 | 0.0000 |
| wine | warmup_joint | 0.5111 | 0.8889 | 0.4222 | 0.8667 | 20.0 | 0.3334 | 0.0000 |
| synthetic | staged_sequential | 0.7800 | 0.7800 | 0.7533 | 0.7533 | 0.0 | 0.0000 | 0.0000 |
| synthetic | alternating | 0.6200 | 0.6800 | 0.5467 | 0.6267 | 6.0 | 0.2825 | 0.0000 |
| synthetic | joint_synchronized | 0.6000 | 0.6800 | 0.4667 | 0.6133 | 11.0 | 0.2827 | 0.0003 |
| synthetic | warmup_alternating | 0.5600 | 0.6267 | 0.5067 | 0.6200 | 8.5 | 0.3594 | 0.0000 |
| synthetic | warmup_joint | 0.5533 | 0.6467 | 0.4933 | 0.5867 | 7.0 | 0.3101 | 0.0000 |

Overall mean final_post_acc: staged_sequential 0.8891, alternating 0.7674, joint_synchronized 0.7511, warmup_alternating 0.7432, warmup_joint 0.7301. The authors called staged_sequential "Pareto-optimal". The run is small (2 seeds, test sets of 38–75 points). staged_sequential has τ_task = 0 and drift = 0 because its representation is frozen. For the co-tuning schedules, final pre-adaptation accuracy falls to 0.33–0.70 (0.6974 for iris alternating; all others are 0.33–0.55), which means retrieval itself degraded.

##### (f) T0 retention on regression (`f3ac443`; energy_efficiency, K 200→50, seeds 42/123/456)
Source: `git show f3ac443:results/t0_regression/T0_REGRESSION_REPORT.md`.

| Policy | Post RMSE | Post MAE | RMSE Δ | Retained |
|:--|:--|:--|:--|:--|
| bias_only | 0.4575 ± 0.0293 | 0.3679 ± 0.0346 | +0.0066 | 50 |
| provenance_bias_coverage | 0.4423 ± 0.0217 | 0.3514 ± 0.0273 | -0.0086 | 50 |
| provenance_only | 0.6705 ± 0.1360 | 0.5678 ± 0.0999 | +0.2197 | 50 |
| stratified | 0.4903 ± 0.0427 | 0.3973 ± 0.0397 | +0.0394 | 50 |
| trustworthiness_only | 0.4543 ± 0.0281 | 0.3608 ± 0.0310 | +0.0035 | 50 |

Pre-maintenance RMSE per seed was 0.4536, 0.4168 and 0.4823. The authors claimed provenance_bias_coverage is best and that the counterfactual audit "guarantees" eviction of harmful cases. With only 3 seeds, its per-seed Δ is −0.0177, +0.0077 and −0.0159, so "guarantees" is unsupported.

##### (g) T0 retention on RL (`f3ac443`; CartPole-v1, capacity 100, maintenance every 500 steps, 3,000 steps, seeds 42/123)
The committed report (`T0_RL_REPORT.md`) is **wrong**. It shows best return 0.0 ± 0.0, final 149.7 ± 110.8 (bias_only) and 121.7 ± 51.8 (provenance_bias_coverage), with 0 cases pruned, because it reads the wrong keys; the commit message itself says so. The table below is corrected from the per-run `summary.json` (`git show f3ac443:results/t0_rl/cartpole_*/summary.json`):

| run | best eval return | selected step | last eval return | actor pruned | critic pruned | critic replaced | budget_interpretation |
|---|--:|--:|--:|--:|--:|--:|---|
| bias_only s42 | 466.5 | 2000 | 71.3 | 2352 | 2900 | 0 | unsolved_or_underfit |
| bias_only s123 | 474.8 | 2000 | 228.0 | 2356 | 2900 | 0 | unsolved_or_underfit |
| prov_bias_cov s42 | 203.2 | 2000 | 158.4 | 2355 | 822 | 2078 | unsolved_or_underfit |
| prov_bias_cov s123 | 270.6 | 1000 | 85.1 | 2370 | 724 | 2176 | unsolved_or_underfit |

This is a smoke-sized run, and no conclusion should be drawn from it. `17cbae6` reverted the RL hook.

#### 1.3.11 Reference baselines (historical, not new experiments)
In `17cbae6`, `docs/t1_spec/supporting/T1_PRIOR_RESULTS.md` moved in-repo. It was extracted from the IJCAI-25 and IJCAI-26 papers, and its tables are reproduced exactly. The authors' claim boundary: "pre-MCB NN-kNN was numerically competitive in prior reported experiments", with no statistical-equivalence or MCB claim. `docs/t1_spec/supporting/MCB_BENCHMARK_PILOT.md` is a plan only ("no pilot run has been executed here").

Classification (accuracy):

| Dataset/configuration | NN-kNN `w=1` | NN-kNN `w=4` | `k` | NN-kNNO | NNet |
|---|---:|---:|---:|---:|---:|
| Iris | 0.980 (0.031) | 0.973 (0.044) | 5 | 0.966 | 0.987 |
| Zebra (a) | 0.945 (0.164) | 0.964 (0.073) | 1 | 0.918 | 0.782 |
| Zebra (a) | 0.800 (0.172) | 0.982 (0.036) | 5 | - | - |
| Zebra (b) | 0.764 (0.057) | 0.923 (0.041) | 1 | 0.968 | 0.673 |
| Zebra (b) | 0.718 (0.076) | 0.932 (0.047) | 5 | - | - |
| Wine | 0.922 (0.075) | 0.972 (0.038) | 1 | 0.994 | 0.836 |
| Wine | 0.838 (0.103) | 0.843 (0.110) | 5 | - | - |
| Breast Cancer | 0.986 (0.019) | 0.984 (0.018) | 5 | 0.959 | 0.951 |
| Balance | 0.942 (0.025) | 0.952 (0.032) | 5 | 0.944 | 0.994 |
| Digits | 0.983 (0.008) | 0.986 (0.010) | 5 | 0.988 | 0.983 |

Image and text classification:

| Method | CIFAR-10 | SVHN |
|---|---:|---:|
| Conv + NN-kNN 500 | 0.688 (0.006) | 0.867 (0.003) |
| Conv + NN-kNN 2000 | 0.675 (0.003) | 0.841 (0.008) |
| ConvNet | 0.689 | 0.875 |
| kNN | 0.339 | 0.469 |
| Pre-Conv + kNN | 0.647 | 0.869 |
| Pre-Conv + NN-kNN | 0.585 | 0.750 |

| Method | SST-5 | SST-2 |
|---|---:|---:|
| Bi-LSTM | 0.376 | 0.776 |
| PreBi-LSTM + kNN | 0.352 | 0.782 |
| PreBi-LSTM + NN-kNN | 0.363 (0.002) | 0.785 (0.001) |
| Bi-LSTM + NN-kNN | 0.368 (0.007) | 0.763 (0.003) |

Regression (average RMSE over 5 runs):

| Model | CH | Di | Ab | BF | BS | WQ | AF | Cars | SP | Y | EE | UTK |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Weighted k-NN | 0.634 | 61.85 | 2.24 | 0.013 | 111.51 | 0.634 | 2.38 | 12683.68 | 2.43 | 10.50 | 2.30 | 12.98 |
| MLP | 0.494 | 57.10 | 2.05 | 0.010 | 45.40 | 0.646 | 2.51 | 15763.32 | 1.48 | 4.52 | 1.38 | 11.68 |
| MLKR | 0.466 | 59.70 | 2.15 | 0.012 | 52.31 | 0.596 | 2.44 | 16447.38 | 1.70 | 3.32 | 1.19 | 12.43 |
| NN-kNN softmax, pure | 0.586 | 58.46 | 2.14 | 0.011 | 71.86 | 0.579 | 2.05 | 13194.01 | 1.41 | 3.24 | 0.942 | 12.60 |
| + adaptation | 0.535 | 57.68 | 2.08 | 0.010 | 62.64 | 0.578 | 1.89 | 11755.02 | 1.34 | 4.50 | 1.38 | 12.04 |
| + locality | 0.665 | 58.51 | 2.16 | 0.011 | 72.04 | 0.577 | 2.08 | 13642.72 | 1.48 | 8.86 | 0.896 | 12.71 |
| + locality + adaptation | 0.903 | 57.35 | 2.07 | 0.011 | 63.72 | 0.576 | 1.98 | 11637.63 | 1.37 | 6.65 | 1.26 | 12.03 |
| NN-kNN sparsemax, pure | 0.445 | 70.31 | 2.22 | 0.014 | 46.92 | 0.594 | 1.64 | 10935.63 | 1.92 | 1.48 | 0.544 | 14.85 |
| + adaptation | 0.438 | 64.06 | 2.12 | 0.012 | 46.32 | 0.589 | 1.58 | 10017.56 | 1.61 | 1.21 | 0.546 | 13.31 |
| + locality | 0.453 | 70.26 | 2.22 | 0.014 | 47.67 | 0.594 | 1.71 | 9966.15 | 1.80 | 1.49 | 0.477 | 14.95 |
| + locality + adaptation | 0.441 | 64.82 | 2.16 | 0.012 | 46.87 | 0.589 | 1.65 | 9923.38 | 1.54 | 1.01 | 0.501 | 13.22 |

Best displayed NN-kNN vs MLP:

| Dataset | Best variant | NN-kNN | MLP | Rel. diff | Result |
|---|---|---:|---:|---:|---|
| California Housing | sparsemax + adaptation | 0.438 | 0.494 | -11.34% | NN-kNN lower |
| Diabetes | softmax + locality + adaptation | 57.35 | 57.10 | +0.44% | within 1% |
| Abalone | softmax + locality + adaptation | 2.07 | 2.05 | +0.98% | within 1% |
| Body Fat | softmax + adaptation | 0.010 | 0.010 | 0.00% | displayed tie |
| Bike Sharing | sparsemax + adaptation | 46.32 | 45.40 | +2.03% | within 3% |
| Wine Quality | softmax + locality + adaptation | 0.576 | 0.646 | -10.84% | NN-kNN lower |
| Airfoil | sparsemax + adaptation | 1.58 | 2.51 | -37.05% | NN-kNN lower |
| Cars | sparsemax + locality + adaptation | 9923.38 | 15763.32 | -37.05% | NN-kNN lower |
| Student Performance | softmax + adaptation | 1.34 | 1.48 | -9.46% | NN-kNN lower |
| Yacht | sparsemax + locality + adaptation | 1.01 | 4.52 | -77.65% | NN-kNN lower |
| Energy Efficiency | sparsemax + locality | 0.477 | 1.38 | -65.43% | NN-kNN lower |
| UTKFace | softmax + locality + adaptation | 12.03 | 11.68 | +3.00% | within 3.0% |

(Airfoil and Cars both being −37.05% is a coincidence; both values check out.)

Synthetic regression and manual feature tuning:

| Model | RMSE | Mean `d*` |
|---|---:|---:|
| NN-kNN softmax without locality loss | 0.9711 | 1.3719 |
| after adaptation | 0.9603 | - |
| NN-kNN sparsemax without locality loss | 1.1907 | 0.2990 |
| after adaptation | 1.0440 | - |
| NN-kNN softmax with locality loss | 0.9854 | 0.2853 |
| after adaptation | 0.9672 | - |
| NN-kNN sparsemax with locality loss | 1.1674 | 0.3006 |
| after adaptation | 1.0460 | - |
| Weighted k-NN | 1.3189 | 0.3284 |
| MLP | 0.9638 | 0.4216 |
| Oracle k-NN by target `y` | 0.0148 | 0.2402 |

| Stage | `w0` | `w1` | `w2` | MSE | R-squared |
|---|---:|---:|---:|---:|---:|
| A | 0.000 | 0.000 | 0.312 | 0.5285 | 0.4814 |
| B.1 | 0.000 | 0.000 | 0.318 | 0.5290 | 0.4809 |
| B.2 | 0.000 | 0.117 | 0.213 | 0.5361 | 0.4740 |
| C.1 | 0.101 | 0.055 | 0.057 | 0.0096 | 0.9906 |
| C.2 | 0.425 | 0.169 | 0.187 | 0.0092 | 0.9909 |

### 1.4 Open issues

1. **All Phase A work is non-evidence** (`17cbae6`). This covers the DROP3/ICF/CoreSet comparison, Covertype, the 5-schedule result and T0 regression/RL. The reports hard-coded conclusions, and the retriever was never trained.
2. **`results/t1/` is historical** (`docs/T1_EXPERIMENT_PLAN.md`). Because of Adam moment aliasing when optimizers are forked, earlier post-selection comparisons must not be used as evidence. The full revised P1/P2/P4/P5/T1.2 grids have not been rerun. P2 and P5 also predate `ec6323a`.
3. **The post-maintenance fine-tune fix (`6adeb30`) is not rerun.** Fine-tuning hurt accuracy by more than 0.1 in 16.4% of P1-C runs and 13.5% of P1-S runs, and digits collapses at K ≥ 0.33. Unexplained anomaly: the invalid fresh-Adam P1-C digits runs did *not* collapse, while the continued-optimizer runs did.
4. **Aliasing bug blast radius.** The `compact_cases` mutation existed on main and `rl-iclr2027` (now `rl`) since `3abe9f0`. Earlier RL runs that compact memory may be affected.
5. **Plan coverage gaps at `6adeb30`** (`docs/T1_PLAN_IMPLEMENTATION.md`):
   - Phase 3 (RL actor/critic) is not done, and regression synchronization is not implemented.
   - MCB is a re-implementation, because the IU-Bloomington code was not obtained.
   - The CIFAR-10/SVHN/SST suites and TabArena are not wired in, and there is no UI study.
   - Force-include and reviewer feature-weight edits are not implemented, nor are the influence-function and Data Shapley baselines.
   - No dataset declares nominal fields, so nominal Δu is plumbed but unused.
6. **PI decisions still open:**
   - K grid and maintenance schedule.
   - s, α, evidence and coverage rules, and scalarization weights.
   - λ_diff and λ_cls, and the nominal probability protocol.
   - s_task, t*, sync weights and MCB momentum.
   - Seeds and margins, the modern benchmark family, and RL signals.
7. **The PI design change supersedes the Q·B policies.** `docs/t1_spec/supporting/pi-design-notes.md` records PI messages that predate the file, which was added only in `77ccd50`. The change: do not combine Q and B; add a coverage-to-reachability ratio and case-removal influence; set aside activation-map redundancy. This makes the `trust`/`trust_utility*` policies used in the Phase B pilots obsolete. It was implemented in §2, outside this section's scope.
8. **The legacy core is near chance on zebra/zebra_special** (P0). This is the stated immediate debugging priority.
9. **Untracked, untraced artifact:** `reports/figures/cartpole_fixed_budget_evaluation_curves_{en,zh}.png`. No doc refers to it, so it is left out of this report.


## 2. T1 alignment with the revised PI specification

Repo `C:/Users/Administrator/NN-KNN/nnknn-work`, branch `neural-cbr`. The spec authority is career-2027 commit `cd77277600841438d98ac5793867da28b8ec6dd0` (PI decisions). Its 18 imported files are mapped in `docs/t1_spec/SOURCE_SNAPSHOT.json`. The Phase B results under `results/t1/` are kept as historical, but the docs say they "do not validate the revised pipeline". The earliest T1 commits (`052699c`, `cc3f67a`, `c7467b6`) are from the  era. Commit `17cbae6` removed them, so this section does not count them as progress.

### 2.1 Summary

- **The pipeline was rebuilt to the PI spec (77ccd50).** Q and B are now compared separately; geometric trust T and the Q/B mixtures are retired. C/H credit uses the query's final outcome after adaptation. Coverage/reachability uses an explicit activation threshold with identity leave-one-out. Removal influence recomputes the final prediction loss, with a cached variant checked against the full computation. Removal also enforces a cumulative loss budget, and case IDs stay stable through MCB and compaction. Every scientific threshold in `configs/t1/_base.yaml` is still marked PILOT; the PI has not set them.
- **Several correctness bugs were found and fixed, which invalidates some earlier comparisons.** The fixes are: Adam moment aliasing in `clone_optimizer` (b91fd70); the revision runner saving the baseline instead of M0/M1/M2 (69af38f); checkpoint selection resetting on no-op maintenance, first in sync (590939b) and then in core (03d8723); and a missing optimizer state for the selected adapter (df58412). The T1 test count grew from 42 to 134.
- **Most maintenance results are negative or mixed.** Retrained removal gave no gain at much higher cost. Adapted regression removal made test RMSE worse than the start state and the matched full control in 12/18 runs. RRR synchronization at matched capacity is better on both regression tasks but worse on Iris. Choosing on an independent maintenance split instead of train-LOO improved test RMSE in all 18 pairs, but it may overfit.
- **Synthetic stress grid (37bbee2): 168 runs, 672 stage replays, 7 factors.** Q flags more corrupted rows than random. The accuracy benefit over matched continuation (MC) is not uniform: under overlap, Q precision is 0.20 vs 0.10 for random, yet M2−MC is −0.0038. The best case is noise (+0.0282), which the authors call "worth further testing, not confirmed".
- **Nominal reuse groups (d97c9a3).** These are train-only vocabularies bound to stable case IDs. The nominal residual adapter raises accuracy from 0.3481 to 0.7222, but reaches only 0.1667 on unseen colors.
- **The reviewer UI and human-study package (f37c666, 4f79a65) are engineering preparation only.** This covers reversible reviewer controls, matched no-edit continuation, and a loopback workbench on a real model. Scripted chains reach 8/8 vs 6/8 for M0/MC. No participants have been recruited, and there is no frozen protocol.

### 2.2 Changes

| Commit | Change |
|---|---|
| 77ccd50 | Imported the revised PI spec into `docs/t1_spec/`. New `model/t1/candidates.py` (coverage/reachability, full and cached removal) and `outcomes.py` (one final prediction/loss path). Rewrote `provenance.py` (+87/−135) and `retention.py` (+115/−285). Stable query-case IDs in `model/nnknn_model.py`. `_base.yaml`: `pipeline_revision: pi_20260920_v2`, policy `provenance_only`, PILOT thresholds (activation 0.05, cache 0.05, allowed loss increase 0.05, regression tolerance 0.5). |
| b91fd70 | New `model/t1/retraining.py`: isolated removal retraining, where each candidate and an unmodified control are forked from the same model, Adam state and RNG. Fixed the `clone_optimizer` deep copy of Adam moments in `model/t1/core.py`. Before the fix, candidates could share moments, so earlier post-selection comparisons need fresh runs. |
| 3e6869c | Recorded the verified retraining and fresh-seed alignment runs (108 manifests, `2026-10-02-pi-alignment-runs.csv`). |
| 5055eee | Regression sync uses the maintained aggregate NN-CDH adapter; the historical pair network is frozen. Selected checkpoints now restore optimizer, free-radius state and Adam moments. Added `docs/T1_EXECUTION_LEDGER.md`. |
| 69af38f | The revision runner now saves the real `M0.pt`/`M1.pt`/`M2.pt`; it used to save the untouched baseline. Final provenance is recomputed with M2. Label bounds are validated before changes are applied. |
| f37c666 | Reversible reviewer controls: transient forced retrieval `a_override = (1-m*f)*a_selected + f*forced` with `m*f <= 1`; versioned label/bias/mixture/feature edits with rollback; protect, quarantine, archive and restore. Matched no-edit continuation control (MC). Added `docs/T1_COMMON_KNOWLEDGE_UI_PROTOCOL.md`. |
| 590939b | Sync no longer resets the best checkpoint or patience on no-op maintenance (`maintenance_state_changed`). |
| 37bbee2 | `independent_v1` synthetic generator with 7 separate random streams; `legacy` stays byte-identical. Truth/group masks, clean labels and source IDs are saved. Revision reports M0/M1/M2/MC on in-domain, shifted, rare and boundary subgroups. |
| 03d8723 | The same no-op reset fix applied to core `train_retrieval` (`checkpoint_state_changed`). |
| d97c9a3 | New `model/t1/nominal.py`: per-field vocabulary fitted on training data only, with separate missing and unknown entries, bound to stable IDs. Adapter input is `[Delta_z, Delta_u, p0]`. Undeclared or missing inputs now fail instead of being silently filled with zeros. Also touches `model/nn_cdh.py`, `model/nnknn_model.py`, `tools/t1_run.py` and most `model/t1/*`. |
| 4f79a65 | Local reviewer workbench (`model/t1/reviewer_ui.py`, `tools/t1_reviewer_ui.py`) on a real model with the common-knowledge red/blue circle task. Loopback only, with session token and Host/Origin checks. One bounded M2 continuation (1–20 epochs) plus MC on unchanged M0. Also updated the nominal-contract doc and the ledger. |
| df58412 | The selected adapter's Adam state is kept for continuation. Checkpoints carry the adapter moments and separate retrieval and adapter epochs. |
| f498c8f | New `model/t1/adapted_retraining.py` (classification with an external adapter). Scopes are retrieval-only, adapter-only and alternating, with a cumulative final-loss budget, a full-memory control, and a protected random comparator matched on capacity. |
| 590a4e6 | Sync capacity matching: post-hoc Q compression of the independent and RR checkpoints to K=.75. The pre-compression state is saved, and case-row moments are aligned to stable IDs. |
| 139deb9 | Adapted removal for regression (aggregate residual adapter), with every candidate replayed. |
| a12f9fd | Adapted budget/capacity grid (config and doc only, 108 runs). |
| 94a8712 | Selection reference routed per condition: train-LOO vs an independent 15% maintenance split. |

T1 test count from the docs: 42 → 45 → 54 → 59 → 75 → 78 → 93 → 98 → 111 → 117 → 119 → 132 → 134. T1 and T3 combined: 150/151.

### 2.3 Experiments and results

#### 2.3.1 Initial alignment verification (77ccd50)
Tests whether the revised pipeline replays and whether cached removal scores match full ones. Setup: 70 manifests in 9 groups, seeds 0/1.

| Check | Value |
|---|---|
| Maintained checkpoints replaying within 1e-6 | 61 |
| Max cached/full score discrepancy | 0.00075658982 |
| Digits random/Q/B accuracy, K=0.5, seeds 0/1 | 0.983333 – 0.988889 |

Source: `docs/T1_ALIGNMENT_STATUS.md`. The historical near-chance collapse on Digits did not reproduce. The authors made no full revised sweep and no confirmatory claim.

#### 2.3.2 Fresh-seed alignment continuation (3e6869c)
Tests retrained vs frozen removal against matched controls, and compares the retention policies. Setup: 108 runs, seeds 2/3/4: retraining (24→18 cases, 36 runs), small candidates (60→45, 42 runs), adapted candidates (60→45, 18 runs), Digits (1077→538, 12 runs). Values are taken from the CSV. Means over the seeds are this report's own arithmetic. In the CSV, Q = `provenance_only`, B = `bias_normalized`, and the ratio condition = `coverage_reachability`.

Retraining, Iris (test accuracy):

| Condition | Active | Extra epochs | s2 | s3 | s4 | Mean |
|---|---:|---:|---:|---:|---:|---:|
| full_frozen | 24 | 0 | 1.000000 | 0.900000 | 1.000000 | 0.966667 |
| full_matched | 24 | 12 | 1.000000 | 0.900000 | 1.000000 | 0.966667 |
| random_frozen | 18 | 0 | 0.900000 | 0.900000 | 0.933333 | 0.911111 |
| random_matched | 18 | 12 | 0.900000 | 0.900000 | 0.933333 | 0.911111 |
| removal_frozen | 18 | 0 | 1.000000 | 0.866667 | 1.000000 | 0.955556 |
| removal_retrained | 18 | 12 | 1.000000 | 0.866667 | 1.000000 | 0.955556 |

Retraining, Energy (test RMSE, standardized target):

| Condition | Active | Extra epochs | s2 | s3 | s4 | Mean |
|---|---:|---:|---:|---:|---:|---:|
| full_frozen | 24 | 0 | 0.334099 | 0.306188 | 0.402250 | 0.347513 |
| full_matched | 24 | 12 | 0.337526 | 0.306990 | 0.402841 | 0.349119 |
| random_frozen | 18 | 0 | 0.333053 | 0.389828 | 0.419869 | 0.380917 |
| random_matched | 18 | 12 | 0.329713 | 0.378466 | 0.415954 | 0.374711 |
| removal_frozen | 18 | 0 | 0.326818 | 0.320537 | 0.418378 | 0.355244 |
| removal_retrained | 18 | 12 | 0.326726 | 0.328393 | 0.418249 | 0.357790 |

Small candidates, Energy (test RMSE, K=0.75):

| Condition | Active | s2 | s3 | s4 | Mean |
|---|---:|---:|---:|---:|---:|
| full_memory | 60 | 0.373417 | 0.346104 | 0.318396 | 0.345972 |
| random | 45 | 0.364264 | 0.398760 | 0.326039 | 0.363021 |
| Q | 45 | 0.395957 | 0.393459 | 0.373808 | 0.387741 |
| B | 45 | 0.356465 | 0.353362 | 0.336687 | 0.348838 |
| coverage_reachability | 45 | 0.400968 | 0.340439 | 0.374969 | 0.372126 |
| removal_influence | 45 | 0.368668 | 0.366147 | 0.333263 | 0.356026 |
| removal_cached | 45 | 0.368668 | 0.366147 | 0.333263 | 0.356026 |

On Iris, every small-candidate condition and every adapted-candidate condition gives 1.000000/0.933333/0.933333 (mean 0.955556).

Digits (test accuracy, K=0.5, 30 fine-tune epochs):

| Condition | Active | s2 | s3 | s4 | Mean |
|---|---:|---:|---:|---:|---:|
| full_memory | 1077 | 0.980556 | 0.969444 | 0.983333 | 0.977778 |
| random | 538 | 0.958333 | 0.966667 | 0.986111 | 0.970370 |
| Q | 538 | 0.975000 | 0.969444 | 0.977778 | 0.974074 |
| B | 538 | 0.972222 | 0.969444 | 0.980556 | 0.974074 |

Other stated diagnostics: the Q/B Spearman correlation at full memory is 0.4133 (small regression), 0.4850 (small Iris), 0.5270 (adapted Iris) and 0.5454 (Digits). The Digits evidence fraction is 0.6493. The max cached/full influence error is about 0.001510.

Source: `docs/experiments/2026-10-02-pi-alignment-continuation.md`, `docs/experiments/2026-10-02-pi-alignment-runs.csv`. Retrained removal took 6 rounds per dataset/seed, with 12 accepted epochs and 258 candidate epochs. It did not improve Iris, and on Energy it was worse on average by 0.002545. On Digits, Q and B beat random on 2 of 3 seeds. The authors say this is not a robustness or superiority claim, and they claim no net speedup.

#### 2.3.3 Supervised synchronization with regression (5055eee)
Tests the independent, RR and RRR schedules, now including regression. Setup: 36 runs on synthetic_diag, Iris, Energy and Yacht; seeds 5/6/7. RRR keeps 324 of 432 or 45 of 60 cases.

| Check | Value |
|---|---|
| Checkpoint replay tolerance | 1e-6 (all pass) |
| Max C+H=A deviation | 2.801e-6 |

Source: `docs/experiments/2026-10-04-supervised-sync.md`. The doc has no numeric results table. It refers to "detailed reports" without giving a location, and the raw artifacts are under `results/t1_pi20260920/continuation_20261004/alignment_sync/`. Caveats: for regression, L_delta and L_post are the same squared residual, so they are not two separate mechanisms. The difference in capacity confounds the schedule comparison.

#### 2.3.4 Sync checkpoint-selection fix (590939b)
Tests how much the no-op reset bug changed the sync results. Setup: the 5055eee sync runs, rerun.

| Metric | Before | After |
|---|---:|---:|
| Synthetic seed 7 RRR post accuracy | 0.919231 | 0.934615 |
| Energy RRR − independent RMSE (s5/s6/s7) | +0.014386 / +0.006908 / +0.007966 | unchanged |

Source: `docs/experiments/2026-10-04-sync-selection-fix.md`. All other post metrics are unchanged. The fix does not explain the Energy compression loss.

#### 2.3.5 Sync at matched final capacity (590a4e6)
Tests independent vs RR vs RRR when all three reach K=.75 (independent and RR are compressed post hoc with Q). Setup: 36 runs, three-seed post-adaptation means.

| Task | Independent | RR | RRR |
|---|---:|---:|---:|
| Synthetic diagnostic accuracy | .9564 | .9641 | .9679 |
| Iris accuracy | .9667 | .9667 | .9556 |
| Energy standardized-target RMSE | .3622 | .3722 | .3493 |
| Yacht standardized-target RMSE | .6403 | .6368 | .5843 |

Source: `docs/T1_SYNC_CAPACITY_COMPARISON.md`. RRR is descriptively better on both regression tasks and worse on Iris. Post-hoc compression makes the independent and RR regression results worse than their full-memory states. The authors say this does not establish a stable RRR benefit or equal compute. A first report had tabulated pre-adaptation metrics by mistake and was corrected. A development seed on Energy (45 cases) gave .380/.411/.354, which the authors mark provisional.

#### 2.3.6 Core checkpoint-selection fix (03d8723)
Tests the core no-op reset fix. Setup: 18 runs before and 18 after, 3 datasets, seeds 8/9/10. All 9 repaired no-op/ordinary pairs are bit-identical. Under the old bug, the selected epoch moved from 9 to 16 in two runs:

| Run | Val loss (fixed → bug) | Test metric (fixed → bug) |
|---|---|---|
| Iris seed 8 | 0.054304 → 0.061560 | acc 0.933333 → 0.966667 |
| Yacht seed 10 | 0.356759 → 0.397327 | RMSE 0.504266 → 0.492752 |

Source: `docs/experiments/2026-10-04-core-selection-fix.md`. In the authors' words, "test scores are not a reason to retain the bug."

#### 2.3.7 Revision artifacts (69af38f)
Tests flagging precision of the review flaggers with the real M0/M1/M2 checkpoints. Setup: 24 runs and 72 stage checkpoints, synthetic data, seeds 5/6/7. Three-seed mean precision:

| Flagger | Budget 10 | Budget 30 |
|---|---:|---:|
| Q | 0.30 | 0.2667 |
| B | 0.0667 | 0.0556 |
| Random | 0.10 | 0.1222 |
| Oracle | 1 (recall 10/54) | 1 (recall 30/54) |

Source: `docs/experiments/2026-10-04-revision-artifacts.md`. Accuracy is near ceiling, some immediate gains reverse during further training, and this batch has no matched no-correction control.

#### 2.3.8 Reviewer controls and matched continuation (f37c666)
Tests M2 (corrected) against MC (no-edit continuation) with equal training. Setup: 24 runs, 96 checkpoints, seeds 5/6/7, budgets 10/30, 5 epochs each for M2 and MC.

| MC→M2 flips | Beneficial | Harmful |
|---|---:|---:|
| Seed 5, Q | 1 | 1 |
| Seed 7, Q | 1 | 0 |
| Seed 7, random, budget 30 | 2 | — |

Source: `docs/experiments/2026-10-04-reviewer-controls.md`. The authors call the effects "small, mixed", with no stable Q superiority. All 30 trained probe scenarios pass. The max regression archive/restore prediction difference is 2.384185791015625e-7, which is float roundoff under the 1e-6 tolerance.

#### 2.3.9 Synthetic stress-factor grid (37bbee2)
Tests whether flagging plus correction helps beyond matched continuation across 7 stress factors. Setup: seeds 8/9/10; Q/B/random/simulated-oracle flaggers; budgets 10/30; 30 core epochs; 5 continuation epochs at 0.1× learning rate. All 168 runs and 672 stage checkpoints pass prediction, aggregate and subgroup replay. Mean M2 − MC accuracy at budget 30:

| Scenario | Q | B | Random | Oracle |
|---|---:|---:|---:|---:|
| Reference | +0.0013 | 0.0000 | 0.0000 | +0.0103 |
| Overlap | -0.0038 | +0.0051 | 0.0000 | +0.0051 |
| Noise | +0.0282 | +0.0103 | +0.0141 | +0.0385 |
| Redundant | +0.0026 | +0.0026 | +0.0038 | +0.0154 |
| Uncovered rare | approximately 0 | 0.0000 | +0.0013 | +0.0064 |
| Shift | +0.0090 | 0.0000 | 0.0000 | +0.0090 |
| Combined | +0.0256 | -0.0026 | +0.0128 | +0.0256 |

Source: `docs/experiments/2026-10-04-synthetic-reliability.md`. Raw artifacts: `results/t1_pi20260920/continuation_20261004/alignment_synthetic_reliability`. Q precision is 0.20 vs 0.10 for random under overlap, yet its accuracy change is negative. Under noise, Q precision is 0.4778 with +0.0282, which the authors call "worth further testing, not confirmed". Compute is not matched across scenarios, and the authors claim no significance.

#### 2.3.10 Nominal reuse groups (d97c9a3)
Tests train-only nominal groups for reuse, retention and sync. Setup: synthetic_nominal/v1, seeds 8/9/10, 45 runs, 4,050 test events replayed.

| Setting | Condition | Accuracy |
|---|---|---:|
| Reuse | Retrieval-only baseline | 0.3481 |
| Reuse | Nominal residual, combined loss: overall / known / unseen colors | 0.7222 / 0.9365 / 0.1667 |
| Reuse | Logit residual, combined loss: overall / unseen colors | 0.6407 / 0.2963 |
| Retention (post) | Full memory (180 cases) | 0.7222 |
| Retention (post) | Q, coverage, full/cached removal (135 cases) | 0.7148 |
| Retention (post) | B | 0.7074 |
| Sync | Independent / RR / RRR | 0.6259 / 0.6296 / 0.6333 |

Source: `docs/experiments/2026-10-04-nominal-contract.md`. These are descriptive results from three seeds. Generalization to unseen categories is weak. RRR uses 135 cases, so capacity is not matched in the sync comparison.

#### 2.3.11 Selected-adapter optimizer states (df58412)
Tests that the selected adapter's Adam state is restored for continuation. Setup: 27 runs (18 reuse, 9 sync).

| Check | Value |
|---|---|
| Events reconstructed | 2,430 |

Source: `docs/experiments/2026-10-04-selected-adapter-optimizers.md`. The authors say this is "not evidence of improved predictive performance."

#### 2.3.12 Adapted removal pilot, Iris (f498c8f)
Tests removal with an external classification adapter. Setup: 9 runs (3 seeds × 3 scopes), 18→12 cases, budget .03. Three-seed means:

| Scope | Final | Initial | Matched full | Matched random |
|---|---:|---:|---:|---:|
| Retrieval-only | .9111 | .9222 | .9111 | .8889 |
| Adapter-only | .9111 | .9222 | .9222 | .8889 |
| Alternating | .9111 | .9222 | .9111 | .8889 |

Source: `docs/T1_ADAPTED_RETRAINING.md`. Every random run violates the .03 budget.

#### 2.3.13 Adapted removal, regression (139deb9)
Tests adapted removal with the aggregate regression residual adapter. Setup: Energy and Yacht, seeds 8/9/10, 3 scopes, 18→12 cases; all 18 runs reach K=12.

| Comparison of adaptive test RMSE | Runs worse (of 18) |
|---|---:|
| vs start state and matched full control | 12 |
| vs matched random | 13 |
| Random runs violating budget | 13 |

Source: `docs/T1_ADAPTED_REGRESSION.md`. On Yacht, adaptive is worse than random for every seed and scope. The authors conclude that a training-LOO acceptance budget does not protect generalization.

#### 2.3.14 Adapted budget and capacity grid (a12f9fd)
Tests adaptive removal across budgets and capacities. Setup: Iris and Wine, 3 seeds, 36 cases, K=.5/.75, budgets 0/.03/.1, 3 scopes, 108 runs.

| Adaptive test accuracy vs | Improve | Worsen | Tie |
|---|---:|---:|---:|
| Initial | 48 | 21 | 39 |
| Matched full | 35 | 21 | 52 |
| Matched random | 45 | 30 | 33 |

Source: `docs/T1_ADAPTED_BUDGET_STUDY.md`. Two zero-budget Wine seed 9 retrieval-only conditions hit the loss budget before reaching their target capacities (K=18 and K=27). Their achieved capacities become the targets for the random controls. Random violates the budget in 40 runs. The 108 runs are correlated conditions, not independent replicates.

#### 2.3.15 Paired maintenance-reference guard (94a8712)
Tests selecting on train-LOO (18 training queries) vs an independent 15% maintenance split (115 Energy / 46 Yacht queries). Setup: Energy and Yacht, seeds 8/9/10, 3 scopes × 2 references, 36 runs, 18→12 cases, budget .03. Mean test post-adaptation RMSE improvement from the maintenance split:

| Task | Adapter | Alternating | Retrieval |
|---|---:|---:|---:|
| Energy | 0.04209 | 0.03739 | 0.04403 |
| Yacht | 0.03228 | 0.03444 | 0.02737 |

Source: `docs/T1_ADAPTED_REFERENCE_GUARD.md`. The maintenance split improves all 18 pairs. Over all 36 runs, adaptive vs initial/full is 24 better and 12 worse, and adaptive vs random is 26 better and 10 worse. Random violates the budget in 27 runs. Caveats: repeated selection on the split can overfit, and the 15% split changes standardization compared with the earlier pilot.

#### 2.3.16 Reviewer workbench and human-study preparation (4f79a65)
Tests the local reviewer UI end to end on a real model with the common-knowledge task. The task has 8 cases, including 2 declared label errors, and 8 queries. Setup: a scripted pilot with seeds 11/12/13 and 3 action chains, plus a browser demo.

| Check | Value |
|---|---|
| Version checkpoints / MC controls replayed | 75 / 9 |
| Separate audit: checkpoints / query events | 99 / 792 |
| Corrective chains vs M0 and MC | 8/8 vs 6/8 |
| Undo chain | 6/8 |
| Chain C cases, M2 vs MC | 7 vs 8 |
| Browser: correcting case 4 / case 5 | 0 flips / both blue-circle queries fixed, 0 harmful flips |
| T1 tests passing (incl. 6 workbench tests) | 117 |

Source: `docs/T1_REVIEWER_UI.md`, `docs/T1_COMMON_KNOWLEDGE_UI_PROTOCOL.md`. The authors call this a mechanism demonstration with no recruited participants. It is not evidence about participant burden or human performance. The Q/B/exposure thresholds in the workbench are demo parameters.

### 2.4 Open issues

- **The PI has not set the scientific values.** These are the activation and cache thresholds, zero-reachability handling, regression tolerance, Q smoothing, B cohorts, loss budget, cache error tolerance, retraining budgets, seeds and margins. Every value in `_base.yaml` is PILOT.
- **Spec gaps.** There is no general score that recovers the prior maintenance methods. The MCB source from IU-Bloomington has not been obtained. Exact legacy, image/text and TabArena reproductions are not wired into the T1 runner, and T1–RL integration is not done. In the docs' words: "Broader synthetic reliability, rare/boundary/shift studies and confirmatory seeds: pilot infrastructure exists; full revised sweeps are not complete."
- **Negative or mixed results to resolve.** Retrained removal brought no gain at higher cost. Adapted regression removal made RMSE worse in 12/18 runs. The Iris adapted pilot dropped from .9222 to .9111. RRR is unstable: worse on Iris, and compression hurts regression. Q's flagging precision does not translate into uniform accuracy gains. Unseen nominal categories reach 0.1667. Revision accuracy is near ceiling, and gains can reverse.
- **Methodology warnings.** Comparisons from before b91fd70 are affected by optimizer aliasing. Revision checkpoints from before 69af38f cannot validate M1/M2. The no-op selection bugs existed until 590939b (sync) and 03d8723 (core). The first sync-capacity report used pre-adaptation metrics, and the first 36-run batch lacked metadata. Maintenance-split selection may overfit. Random controls are matched on capacity and training, not on the loss constraint.
- **Human study.** Still needed: real participants, consent/IRB procedures, randomized or counterbalanced assignment, and a frozen protocol. Expert content insertion, constrained adaptation and run import/resume in the UI are not implemented.
- **Ledger next steps** (`docs/T1_EXECUTION_LEDGER.md`): wider scale and compute grids with frozen confirmation; broader group and constraint protection; broader matched-compute sync grids; recurring-group credit and constrained acceptance; real-data nominal reuse.


## 3. T2 reinforcement learning

Scope: 13 T2 commits. Two earlier items are included as context: the T2 plan (77ccd50) and the T0 RL retention run (f3ac443, later deleted). Commit 996353b also contains T3 need-matching work; only its T2 part is covered here. The raw artifacts behind 3.3.2–3.3.8 are in `results/t2_pi20260920/`, which is untracked. Experiments 3.3.9–3.3.12 have no artifacts in the repo; their evidence sits in task `work/` and `outputs/` paths outside the repo, so the `docs/T2_*.md` files are the only in-repo source for them.

### 3.1 Summary

- **Engineering is the main output.** The work repaired critic retrieval IDs, restore of the critic's active count, Adam-state alignment by stable ID (`preserve_by_id`, optional), and selected-checkpoint consistency, where weights, target and Adam states are now saved and restored together. It also added a per-role retrieval/removal audit, a training-time audit, a quality ledger, and offline and live case retention. Acrobot-v1 was added as a second task. Tests went from 159 to 187.
- **The Stage A gate has not passed.** No doc claims a competitive or general benefit. Margins are not frozen, capacity and compute are not matched, and the 500-case default budgets are untested.
- **Acrobot actor startup fails.** Raw-positive-advantage admission cannot start under constant negative reward. At 2,048 interactions every NN actor is unready, and the -458 return comes from the uniform fallback. At 8,192 interactions seeds 8 and 9 become ready only at step 7,001 and then score -500; seed 10 is never covered. Critic label modes do not fix this in general.
- **On CartPole, some results are encouraging but unmatched.** For NN/NN at 2,048 interactions, `preserve_by_id` raised mean return from 145.111 to 301.000, and an online target gave 304.889. Both have high seed variance and unmatched update counts or capacity.
- **Retention hurts the actor.** Offline, 34 of 72 conditions worsened MC MSE. With the query guard, 6 of 36 still worsened greedy return. Live actor-only retention lowered CartPole return on all 3 seeds. Lower own-policy MSE was misleading: common-policy MSE got worse.
- **The T0 RL report was wrong.** Its `best_return=0.0` and zero pruning came from a script that read the wrong keys. On the actual values, provenance pruning did worse than `bias_only`. These results were then deleted from the branch.

### 3.2 Changes

| Commit | Change |
|---|---|
| f3ac443 (T0 RL part, context) | T0 case maintenance in the separate-memory NN-kNN actor-critic (`model/nnknn_rl_workflow.py`), runner `tools/run_t0_rl.py`, results in `results/t0_rl/`. The committed report is wrong (see 3.3.1). |
| 17cbae6 (context) | Deleted `results/t0_rl` together with the other  T0/T1 artifacts. |
| 77ccd50 (context) | Plan `docs/t1_spec/T2_RL_IMPLEMENTATION_PLAN.md`. Role matrix: MLP/MLP, NN/MLP, MLP/NN, NN/NN with separate stores, plus a hybrid contingency. Stage A gate: (1) improve task learning, stability, or efficiency over a frozen NN-kNN RL reference under matched interactions, compute, case memory, tuning and evaluation budgets; (2) a prespecified competitiveness margin against a matched neural baseline in at least one role; (3) multiple seeds; (4) case traces; (5) causal deletion, replacement or reweighting interventions. Stage A tasks: CartPole, Acrobot, LunarLander. |
| a2a5da9 | Critic retrieval-ID fix: outer memory IDs and internal core IDs had drifted apart, leaving stale target IDs after compaction. Both ID buffers and the next-ID counters are now kept in sync. |
| 66a658d | Role audit (`model/t2/audit.py`): records the actual retrieval and runs per-case removal interventions; C/H/Q per role and per target stream. Pilot `tools/t2_role_pilot.py`. Fix: the critic's active count was missing from `state_dict`, so a restored critic came back empty. |
| f660a37 | Acrobot-v1 added as a maintained task, with return ceiling 0 kept separate from the 500-step limit. Pilot takes `--task`. |
| fd83666 | Opt-in training-time audit (`case_audit_queries_per_batch`, 0..32): pre-gradient snapshots of critic GAE and actor behavior-surrogate events. Paired on/off runs give identical weights. Also touches `tools/t2_role_pilot.py` and `tests/test_t2_role_audit.py`. |
| 03bf8f7 | Explicit target sync interval in the pilot. GAE counterexample test: with constant -1 reward, every advantage is negative. |
| 791a4b4 | Optional `case_optimizer_maintenance=preserve_by_id` (default stays `reset`): case Adam moments follow stable IDs through insertion and compaction. |
| cb8cd96 | Selected-checkpoint consistency: weights, lagged target and Adam states are snapshotted and restored as one set. Restoring an optimizer from a legacy checkpoint is rejected. |
| 393f47d | `QualityLedger` (`model/t2/quality.py`), option `case_quality_tracking`, off by default. Keeps actor-surrogate, critic-GAE and MC streams separate. |
| 718537e | Offline fresh-reference retention (`model/t2/retention.py`, `tools/t2_retention_pilot.py`, `tests/test_t2_retention.py`): full-bank removal with refill; candidates filtered by a mean-loss budget, then ranked by lowest Q. |
| 6218431 | Per-query loss guard `max_query_loss_increase`; pilot flags `--reference-batches` and `--query-loss-budget`. |
| b9302c7 | Live retention during training, off by default (`case_retention_keep_fraction=None`); requires `preserve_by_id`. Code changes: `model/nnknn_rl_workflow.py` (+102) and pilot CLI flags. The docs also report 18 common-policy MC diagnostics; their code was not committed. |
| 359e864 | Docs only (`T2_EXTENDED_STARTUP.md`): Acrobot startup at 2,048 vs 8,192 interactions. |
| 996353b (T2 part) | Pilot CLI exposes fixed, mutable, trainable and hybrid critic label modes, the activation-matching threshold and the EMA alpha. Doc `T2_CRITIC_LABEL_CALIBRATION.md`. |

Test count over the period: 159 → 160 → 162 → 163 → 165 → 169 → 177 → 181 → 182 → 187.

### 3.3 Experiments and results

#### 3.3.1 T0 RL retention (context, superseded)
**Tests:** T0 `provenance_bias_coverage` pruning vs `bias_only` pruning at capacity 100.
**Setup:** CartPole, 3,000 steps, seeds 42/123, maintenance every 500 steps.

The committed report (`git show f3ac443:results/t0_rl/T0_RL_REPORT.md`) shows best return 0.0 ± 0.0 and 0 cases pruned for both policies. The commit message itself says those values are artifacts of reading the wrong keys. The corrected values below come from the per-run `summary.json` files in f3ac443.

| Policy | Seed | Best-eval mean ± std | Last eval | Selected step | Actor/critic cases | Actor/critic pruned |
|---|---|---|---|---|---|---|
| bias_only | 42 | 466.5 ± 40.123 | 71.3 | 2000 | 100/100 | 2352/2900 |
| bias_only | 123 | 474.8 ± 54.958 | 228.0 | 2000 | 100/100 | 2356/2900 |
| provenance_bias_coverage | 42 | 203.2 ± 107.551 | 158.4 | 2000 | 91/100 | 2355/822 |
| provenance_bias_coverage | 123 | 270.6 ± 112.047 | 85.1 | 1000 | 99/100 | 2370/724 |

**Reading:** `bias_only` wins on both seeds. No run reaches the 475 threshold (`passed=false`). The run is smoke-sized and was deleted in 17cbae6.

#### 3.3.2 Four-role pilot
**Tests:** final return per role, plus a direct retrieval and removal audit (all 384 masked interventions replay).
**Setup:** CartPole, 512 interactions, seeds 8/9/10, capacity 128, top-k 8.

| Actor/Critic | s8 | s9 | s10 | Mean | Mean independent-MC MSE |
|---|---:|---:|---:|---:|---:|
| MLP/MLP | 376.333 | 45.333 | 13.333 | 145.000 | 145.747 |
| NN/MLP | 100.667 | 47.667 | 277.333 | 141.889 | 524.631 |
| MLP/NN | 9.0 | 41.0 | 49.667 | 33.222 | 34.865 |
| NN/NN | 44.0 | 102.0 | 461.667 | 202.556 | 819.735 (s10: 2262.612) |

**Source:** `docs/T2_ROLE_AUDIT_PILOT.md`; per-seed values from `results/t2_pi20260920/continuation_20261004/role_audit_512_restore/`.
**Caveat:** seed ranges are broad. Visitation depends on the actor, so the MSEs are not a paired ranking. Only the first 8 MC queries are audited, and actor rollout interventions were not finished at this point.

#### 3.3.3 Two tasks × four roles
**Tests:** whether a larger budget and a second task change the role picture.
**Setup:** CartPole and Acrobot, 2,048 interactions, seeds 8/9/10.

| Task | Actor/Critic | s8 | s9 | s10 | Mean | NN actor action counts (s8/s9/s10) |
|---|---|---:|---:|---:|---:|---|
| CartPole | MLP/MLP | 349.333 | 98.333 | 72.667 | 173.444 | – |
| CartPole | NN/MLP | 139.0 | 89.667 | 132.667 | 120.444 | [61,67] / [58,70] / [55,73] |
| CartPole | MLP/NN | 74.0 | 64.667 | 102.333 | 80.333 | – |
| CartPole | NN/NN | 103.333 | 86.0 | 246.0 | 145.111 | [47,81] / [65,63] / [60,68] |
| Acrobot | MLP/MLP | -500 | -500 | -500 | -500 | – |
| Acrobot | MLP/NN | -500 | -500 | -500 | -500 | – |
| Acrobot | NN/MLP | -458 | -458 | -458 | -458 | [0,0,0] ×3 |
| Acrobot | NN/NN | -458 | -458 | -458 | -458 | [0,0,0] / [2,1,0] / [0,0,0] |

On CartPole the NN actors hold 128 case entries on every seed.

**Source:** `docs/T2_TWO_TASK_DIAGNOSIS.md`; per-seed values from `.../continuation_20261004/two_task_2048/`.
**Conclusion:** CartPole NN/NN fell from 202.556 (512 interactions) to 145.111, so more updates did not reliably help. All six Acrobot NN actors stayed unready; -458 is the uniform fallback, not a learned policy. The NN critic's MC error on Acrobot is in the thousands.

#### 3.3.4 Target modes
**Tests:** EMA vs hard vs no (online) target for NN/NN.
**Setup:** 2,048 interactions, sync interval 4, seeds 8/9/10.

| Task | Target | s8 | s9 | s10 | Mean |
|---|---|---:|---:|---:|---:|
| CartPole | EMA | 103.333 | 86.0 | 246.0 | 145.111 |
| CartPole | hard | 125.0 | 88.0 | 441.333 | 218.111 |
| CartPole | online (none) | 102.667 | 489.667 | 322.333 | 304.889 |
| Acrobot | EMA / hard / online | -458 | -458 | -458 | -458 (fallback; actor counts [0,3,0]) |

**Source:** `docs/T2_TRAINING_CONTRIBUTION_AUDIT.md`; per-seed values from `.../continuation_20261004/target_audit_2048/`. 4,800 training and 1,152 MC interventions replay.
**Caveat:** the authors call this "too exploratory to change default". Acrobot has no real contrast: it ran 3 optimizer batches against a sync interval of 4, so the target only ever held its initial copy.

#### 3.3.5 Executed target sync on Acrobot
**Tests:** whether target updates actually execute, and whether that changes Acrobot.
**Setup:** NN/NN, sync interval 1, 2,048 interactions, 9 runs. The doc has no table.

| Mode | Greedy return (s8/s9/s10) | Actor counts | Target check |
|---|---|---|---|
| EMA | -458 / -458 / -458 (fallback) | [0,3,0] | 3 batch updates executed (sync count 4); max parameter difference from online about 0.0018525 |
| hard | -458 / -458 / -458 (fallback) | [0,3,0] | 3 batch updates executed; equals online exactly |
| online | -458 / -458 / -458 (fallback) | [0,3,0] | reproduces the interval-4 runs exactly |

**Source:** `docs/T2_EXECUTED_TARGET_DIAGNOSIS.md`; `.../continuation_20261004/target_sync1/`.
**Conclusion:** the target is not the cause. With 500 rewards of -1 and zero values (γ 0.99, λ 0.95), every advantage is negative, so raw-positive admission can never start.

#### 3.3.6 Optimizer maintenance: reset vs preserve_by_id
**Tests:** whether keeping case Adam moments aligned by stable ID helps.
**Setup:** NN/NN, 2,048 interactions, EMA target, interval 4, seeds 8/9/10.

| Task | Mode | s8 | s9 | s10 | Mean | Actor capacity (s8/s9/s10) |
|---|---|---:|---:|---:|---:|---|
| CartPole | reset | 103.333 | 86.0 | 246.0 | 145.111 | 128/128/128 |
| CartPole | preserve_by_id | 287.333 | 153.0 | 462.667 | 301.000 | 117/128/128 |
| Acrobot | reset / preserve | -458 | -458 | -458 | -458 | 0/3/0 |

**Source:** `docs/T2_OPTIMIZER_ALIGNMENT.md`; `.../continuation_20261004/optimizer_alignment/`. Paired gains: 184.000 / 67.000 / 216.667.
**Caveat:** optimizer batches (reset/preserve) were 18/16, 19/17 and 13/18, so update counts and capacity are not matched. The authors say this is "not a Q selector, an approved gate condition or evidence of general suitability". `reset` remains the frozen reference.

#### 3.3.7 Selected-checkpoint repair
**Tests:** returns after restoring weights, target and Adam states together at the selected checkpoint.
**Setup:** CartPole, four roles, 512 interactions, evaluation every 128 steps, seeds 8/9/10.

| Actor/Critic | s8 | s9 | s10 | Mean | Selected step/source (s8/s9/s10) |
|---|---:|---:|---:|---:|---|
| MLP/MLP | 376.333 | 60.333 | 13.333 | 150.000 | 512 final / 256 best / 512 final |
| MLP/NN | 9.0 | 41.667 | 79.667 | 43.445 | final / 512 best / 128 best |
| NN/MLP | 100.667 | 94.0 | 277.333 | 157.333 | final / 256 best / final |
| NN/NN | 73.667 | 102.0 | 461.667 | 212.445 | 256 best / final / final |

**Source:** `docs/T2_SELECTED_CHECKPOINT_STATE.md`; `.../continuation_20261005/selected_checkpoint/`. Five runs select a non-final phase. 21 Adam states and 6 targets restore exactly, and all 12 continuation checks pass.
**Caveat:** this is an engineering repair; "no performance gain or matched-compute gate claim follows".

#### 3.3.8 Quality ledger coverage
**Tests:** how many cases get quality observations when tracking is on. Tracking on and off reproduce training exactly.
**Setup:** NN/NN, 2,048 interactions, seeds 8/9/10.

| Task | Actor active coverage (of 128) | Critic GAE | Critic MC | Final critic_holdout_mse (s8/s9/s10) |
|---|---|---|---|---|
| CartPole | 30/33/31 | 22/18/25 | 22/21/24 | 178.935 / 384.289 / 961.274 |
| Acrobot | counts 0/3/0, no ready observations | 11/15/14 | 10/14/8 | 4963.266 / 4343.401 / 4762.629 |

**Source:** `docs/T2_ROLE_QUALITY_LEDGER.md`; MSE values from `.../quality_ledger_2048/*/summary.json`.
**Caveat:** this is infrastructure, not an active selector, and coverage is sparse.

#### 3.3.9 Offline retention with a reference-coverage guard
**Tests:** whether wider reference sets and a per-query loss guard reduce regressions after retention.
**Setup:** 108 one-role conditions on six `preserve_by_id` checkpoints.

| Mode | Candidate replays | Removed | Reached capacity | MC MSE worse | Greedy return worse |
|---|---:|---:|---:|---:|---:|
| control | 3,758 | 364 | 6/36 | 20/36 | 4/36 |
| wide | 14,533 | 629 | 19/36 | 15/36 | 6/36 |
| strict query guard | 11,368 | 392 | 10/36 | 6/36 | 6/36 |

**Source:** `docs/T2_REFERENCE_COVERAGE_GUARD.md`. The earlier unguarded run (`docs/T2_FRESH_REFERENCE_RETENTION.md`) had 72 conditions, 7,468 candidate trials and 737 removals. Only 16 conditions reached the requested capacity; 34 worsened MC MSE and 4 worsened greedy return. All 24 Acrobot actor conditions were unavailable.
**Conclusion:** the per-query guard cuts MC regressions but not actor return regressions, and the retained capacities differ between modes.

#### 3.3.10 Live retention during training
**Tests:** retention applied during training to the actor, the critic or both.
**Setup:** NN/NN, 2,048 interactions, keep fraction 0.9, every 512 steps, 16 queries, zero loss budget, seeds 8/9/10.

| Mode | CartPole mean greedy return | Per-seed returns | Acrobot mean |
|---|---:|---|---:|
| off / noop | 244.889 | 214 / 181.667 / 339 | -458 |
| actor | 148.444 | 190 / 104.333 / 151 | -458 |
| critic | 287.889 | 293 / 70.667 / 500 | -458 |
| both | 155.222 | 222 / 128.667 / 115 | -458 |

**Source:** `docs/T2_LIVE_RETENTION.md`.
**Conclusion:** actor-only retention loses on all 3 seeds; critic-only improves 2 and loses 1. The common-policy MC diagnostic is off-policy. Actor-only's own-policy MSE falls while its common-policy MSE gets worse on all 3 seeds. Both-mode seed 10: MSE 3435.767 → 306.870, but rollout return 299.5 → 85, and common-policy MSE 2706.460. On Acrobot the critic's common-policy MSE improves on all 3 seeds, but the actor still does not start. Capacity- and compute-matched confirmation is still needed.

#### 3.3.11 Acrobot startup at 2,048 vs 8,192 interactions
**Tests:** whether a larger budget lets the NN actor become ready on Acrobot.
**Setup:** NN/NN, GAE with admission replay. Results are identical across EMA, hard and none target modes.

| Budget | Seed | First actual ready step | Final action counts | Greedy return |
|---|---:|---:|---|---:|
| 2,048 | 8 | unavailable | [0,0,0] | -458 fallback |
| 2,048 | 9 | unavailable | [2,1,0] | -458 fallback |
| 2,048 | 10 | unavailable | [0,0,0] | -458 fallback |
| 8,192 | 8 | 7,001 | [2,3,4] | -500 learned policy |
| 8,192 | 9 | 7,001 | [42,24,30] | -500 learned policy |
| 8,192 | 10 | unavailable | [1,4,4] | -458 fallback |

**Source:** `docs/T2_EXTENDED_STARTUP.md`.
**Conclusion:** reaching readiness does not help. Seeds 8 and 9 become ready but their greedy policy fails (-500). Seed 8's 9 positive recommendations include no true-terminal positives. Critic values are optimistic compared with the negative MC targets.

#### 3.3.12 Critic label modes
**Tests:** whether fixed, mutable, trainable or hybrid critic labels, at different activation-matching thresholds, fix Acrobot startup or calibration.
**Setup:** Acrobot, 8,192 interactions, NN/NN, hard target, interval 1, seeds 8/9/10.

| Condition | Greedy return (s8/s9/s10) | First actual ready step |
|---|---|---|
| Fixed or mutable, threshold 0 | −500 / −500 / −458 | 7001 / 7001 / unavailable |
| Trainable or hybrid, threshold 0 | −500 / −110.667 / −458 | 7001 / 7001 / unavailable |
| Mutable, threshold −0.25 | −500 / −500 / −500 | 8001 / 7001 / unavailable |
| Hybrid, threshold −0.25 | −500 / −370 / −500 | 8001 / 7001 / unavailable |
| Mutable or hybrid, threshold −1 | −458 / −458 / −500 | unavailable / unavailable / 8001 |

**Source:** `docs/T2_CRITIC_LABEL_CALIBRATION.md`.
**Conclusion:** at threshold 0 no mutable update executes. The trainable-label gain appears on seed 9 only and "does not justify a general recommendation". At threshold −1, common-policy MSE drops sharply (s8 about 4048 → 2084, s9 1015 → 496, s10 3982 → 1928), but actor readiness fails. Common-policy MSE is an off-policy diagnostic, not proof of calibration.

Earlier CartPole campaigns are excluded: `results/rl_mcb_campaign` and `results/rl_case_maintenance` predate the commits covered here. They are untracked, have per-run `summary.json` files but no aggregate report, and record no author conclusion.

### 3.4 Open issues

1. **Stage A gate not passed.** Margins are not frozen, capacity and compute are not matched, and the 500-case default budgets are untested. LunarLander has not been run.
2. **Acrobot actor startup.** Raw-positive admission cannot start under negative rewards. At 8,192 interactions only two seeds become ready, both late, and then score -500. The authors rule out switching to normalized advantage or admitting nonpositive cases as a fix.
3. **NN critic calibration.** MC MSE runs from the hundreds to the thousands, and values are optimistic. Cross-role MSE comparisons are confounded by different visitation, and the label modes give no general fix.
4. **Retention hurts actor return**, both offline (with or without guards) and live. Lower own-policy MSE can mislead. Actor-return protection needs to be separated from surrogate-loss guarantees.
5. **CartPole seed variance and unmatched budgets.** The gains from `preserve_by_id`, online targets and critic retention are exploratory, with update counts and capacity not matched.
6. **Partial audits.** Queries are bounded prefixes, quality coverage is about 20–30 of 128 cases, and removal deltas measure immediate loss rather than episodic return.
7. **Reproducibility.** `results/t2_pi20260920/` is untracked, and the evidence for 3.3.9–3.3.12 lives outside the repo.
8. **Process errors, recorded with their failed logs kept:** the selected-checkpoint audit first required a strictly smaller step; the startup report first compared inactive target rows; the label report first included cache files; the common-policy helper first used the 2,048 global step.
9. **Next steps listed in the docs:** wider label-mode, matching and calibration comparisons; Acrobot negative-reward startup; capacity- and compute-matched confirmation; broader budgets and LunarLander; freezing the Stage A thresholds.


## 4. T3 LLM agent

Scope: 24 T3 commits on `neural-cbr`, from `6def29b` to `efb7ce1`. Raw artifacts are in `outputs/` and `work/`, outside the git tree, so every number below comes from the `docs/T3_*.md` files. The governing plan, `docs/t1_spec/T3_LLM_AGENT_IMPLEMENTATION_PLAN.md`, was created in `77ccd50` and has not changed since. It defines four things:
- Two integration modes, prompt augmentation and internal model integration, plus a combined condition.
- Four stages: single-hop QA/RAG, multi-hop (HotpotQA/MuSiQue), biomedical transfer (BioASQ candidate) and agent memory.
- A joint success standard (§17): task quality competitive with standard RAG, plus real gains in traceability, correctability and resistance/recovery.
- Direct conditional utility (§9.3): `u_i = J(q,S without i) - J(q,S)`.

### 4.1 Summary

- **Infrastructure is in place.** The team built a frozen, hash-pinned open-weight host (Qwen/Qwen3-0.6B, 596,049,920 parameters, 0 trainable). NN-kNN retrieval is bound to the original artifacts, with scope, quarantine and expiry gates. A two-channel audit trail and a version binding cover model topology, state and config. The project also has public SQuAD1.1/HotpotQA data with audited scoring, a host failure journal, strict schemas with a constrained decoder, direct no-refill set utility and metric provenance. T3 tests went from 10 to 66.
- **Learned need matching is a negative result.** Training the 256-weight lexical metric lowered tune loss (4.03579 to about 4.008) but also lowered recall@2 (.546875 to .46875–.484375). Hard negatives with recall-based selection raised recall@2 to .65625–.671875, which is still far below BM25 at 1.0.
- **Public host runs show no benefit from learned retrieval or iteration.** The test used 8 questions per dataset × 5 conditions and 80 trials per run. On SQuAD, the retrieved sets do worse than empty evidence: mean full-set gain is −0.0676 (fixed) and −0.0386 (learned). Hotpot support and joint F1 are 0 in every run. The iterative condition never issued a second retrieval.
- **The constrained decoder makes output format more reliable.** Failures fell from 24/80 under strict plain decoding to 7/80 under the constrained decoder. The decoder also changes the emitted needs, so the F1 differences are not answer-stage-only effects.
- **Actual-need training is blocked by repetition loops.** Both 192-request collections (128- and 256-token ceilings) fail on the same 2 questions. A 512-character bounded request grammar stops the loop, but the needs it lets through keep the repeated content. The full bounded 192-query collection was running at the time of writing (session 90847), and no fitting result exists yet.
- **An internal output interface is ready but untested on real answers.** The candidate (an activation-weighted token-histogram mixture, alpha 0.1) passed real-case preflight. The 64-call none/prompt/internal/combined host experiment has not run.

### 4.2 Changes

| Commit | Change |
|---|---|
| 6def29b | Frozen host: pinned ungated Qwen3-0.6B revision (9 files, SHA256), smoke test on Intel Arc A770 XPU with fp16 and eager attention. `do_sample=False` is now passed explicitly. |
| feb6b1c | Scoped retrieval contract (`model/t3/retrieval.py`) and a bounded host loop (`orchestrator.py`). Events have two channels, evidence and audit. 10 tests. |
| 98f91f0 | First real frozen-host requests: a fictional fixture with 4 entities under 4 conditions. Missing or non-boolean readiness is rejected. 11 contract tests; combined T1/T3 suite 150 tests. |
| ede1a91 | Exact-set removal audit (`--direct-set-removal`): no refill, byte-identical prompts. |
| 7283e4b | Immutable case metadata and binding to the actual model version, checked at 4 boundaries. Fixes 4 counterexamples. |
| f136999 | Public QA loader for SQuAD1.1 and HotpotQA (SHA256-checked, article-disjoint split), plus scoring audited against the official evaluators. |
| 679d1ea | Lexical need matching (`model/t3/lexical.py`, `tools/t3_need_matching.py`, `tests/test_t3_need_matching.py`, `docs/T3_NEED_MATCHING.md`). Yields the counterexample: lower loss, worse recall. |
| 902b871 | Recall-based checkpoint selection and a hard-negative ranking loss (8 negatives, margin .2). |
| 1944775 | Host failure journal (`event_sink`), `max_cases_per_round` = 3, public host pilot. Request prompt v1 failed and the run was terminated. |
| 58df8b2 | Docs only: the legacy plain public run (80 trials, request prompt v2). |
| 2d8d063 | Optional strict host schemas (`host_schema.py`) and a constrained decoder (lm-format-enforcer 0.11.3). |
| 2499796 | Docs only: verified a 31-trial prefix of the constrained run. |
| 673add8 | Direct no-refill set utility (`model/t3/utility.py`): scores the full set, each single removal and the empty set. |
| 6be9915 | Fix: invalid outputs now count as observed failures (zero score, unit loss), not as missing feedback. |
| d6e75c8 | Docs only: full constrained-decoder run and the need-shift diagnosis. |
| 6fdbd85 | Train-only actual host needs (`need_queries.py`; 128 fit + 64 tune). Collection and evaluation share one request prompt. |
| 28e81f4 | Docs only: strict plain vs constrained decoder, paired by question. |
| afad65b | Docs only: 32-set direct utility run complete. |
| ffdf189 | Metric provenance: the checkpoint SHA256 must match exactly one recorded run. Fixes a false "seed 8, tune-loss-selected" label. |
| 29eaf8f | Uniform request token ceiling (128/256/512). Version-2 need manifests bind the budget, and fitting must use the same treatment as evaluation. |
| 9340be6 | Optional bounded request grammar: need strings capped at 512 characters, observable state at 128. Also enforced at runtime and when manifests load. 61 T3 tests. |
| 790695b | Docs only: real 2-question bounded diagnostic recorded; full 192-query bounded collection and the matched comparison launched. |
| 2a4097d | Candidate internal output interface (`model/t3/internal.py`) and a 64-call matched driver (`tools/t3_internal_host_pilot.py`). Fixes the public pilot's total-token accounting (it used a literal 384). 65 T3 tests. |
| efb7ce1 | Real-case internal preflight. Fixes a large-vocabulary float32 probability trace. 66 T3 tests. |

### 4.3 Experiments and results

#### 4.3.1 Frozen-host fictional fixture (98f91f0, ede1a91)
- **Tests:** whether the real Qwen3-0.6B host can issue JSON requests, receive original scoped artifacts through NN-kNN, and answer.
- **Setup:** 4 fictional entities × 4 conditions; 16 trials, 28 host calls, 3,740 input / 788 output tokens.

| Condition | Exact color answer | UNKNOWN abstention | Actual retrieval rounds |
|---|---:|---:|---:|
| No retrieval | 0/4 | 4/4 | 0 |
| NN-kNN one-shot budget | 4/4 | 0/4 | 1 per task |
| NN-kNN iterative budget | 4/4 | 0/4 | 1 per task |
| Target case quarantined | 0/4 | 4/4 | 1 per task |

- **Removal extension:** removing the only supporting record turned every answer into UNKNOWN (loss 0→1, utility 1). The extended batch used 32 calls and 4,160 input / 852 output tokens.
- **Source:** `docs/T3_PROMPT_ENGINEERING_PILOT.md`.
- **Caveat:** this is an engineering fixture. The iterative condition never used a second round. Quarantine reranks and refills, so it does not measure direct utility. Single-record sets cannot test redundancy or poisoning.
- **Regression check (7283e4b):** after version binding, the fixture gave identical results: 16 trials, 32 calls, 12 events, 4 removals. Source: `docs/T3_VERSION_BINDING.md`.

#### 4.3.2 Public data foundation (f136999)
- **Tests:** whether public QA data load correctly and are scored faithfully.
- **Setup:** SHA256-pinned SQuAD1.1 and HotpotQA (HF mirror), scored against the hash-pinned official evaluators.

| Artifact | Questions | Unique source-bound paragraph cases |
|---|---:|---:|
| SQuAD1.1 train | 87,599 | 18,896 |
| SQuAD1.1 dev | 10,570 | 2,067 |
| HotpotQA distractor validation mirror | 7,405 | 73,700 |

- **Scoring agreement:** 511,156 scoring comparisons agree with the official evaluators.
- **Source:** `docs/T3_PUBLIC_BENCHMARK_FOUNDATION.md`.
- **Caveats:**
  - The Hotpot file comes from a mirror, and byte identity with the CMU original is unverified.
  - One support annotation is unavailable.
  - The dev data is not a hidden test set.

#### 4.3.3 Lexical need matching (679d1ea)
- **Tests:** whether training a lexical NN-kNN metric improves which source passage it retrieves.
- **Setup:** 128 fit / 64 tune SQuAD-train questions, 64-case pools, 256 diagonal weights. 3 seeds × 20 epochs × 128 updates, with the checkpoint chosen by lowest tune loss (rule fixed in advance).

| Condition | Tune need loss | Source recall@1 | Source recall@2 |
|---|---:|---:|---:|
| Fixed lexical NN | 4.03579 | .421875 | .546875 |
| Learned seed8,epoch17 | 4.00877 | .375000 | .484375 |
| Learned seed9,epoch14 | 4.00813 | .390625 | .468750 |
| Learned seed10,epoch19 | 4.00892 | .390625 | .484375 |
| BM25,k1=1.2,b=.75 | not the NN loss | .937500 | 1.000000 |

- **Source:** `docs/T3_NEED_MATCHING.md`.
- **Conclusion:** training is real (all 7,680 steps had nonzero gradients, and an exact replay reproduces it) but ineffective: loss falls while recall@2 drops. BM25 is far stronger.

#### 4.3.4 Ranking objective and selection controls (902b871)
- **Tests:** whether recall-based checkpoint selection or hard negatives fix the recall drop.
- **Setup:** the same split and seeds 8/9/10, with a hard-negative loss against the 8 closest actual-core cases (margin .2).

| Control | Selected epochs,seeds8/9/10 | Tune recall@1 | Tune recall@2 |
|---|---|---|---|
| Original loss-selected softmax | 17/14/19 | .375/.390625/.390625 | .484375/.46875/.484375 |
| Softmax,recall selection | 1/0/0 | .421875/.421875/.421875 | .546875/.546875/.546875 |
| Hard negatives,recall selection | 3/4/1 | .46875/.484375/.515625 | .65625/.671875/.65625 |
| BM25 control | no training | .9375 | 1.0 |

- **Source:** `docs/T3_RANKING_OBJECTIVE.md`.
- **Conclusion:** recall-based selection only avoids the drop; seeds 9 and 10 select the untrained epoch 0. Hard negatives raise recall@2 for every seed but stay far below BM25. This is exploratory tuning.

#### 4.3.5 Public host, legacy plain run (1944775, 58df8b2)
- **Tests:** end-to-end QA by the real host with no retrieval, BM25, fixed NN, learned NN, or iterative retrieval.
- **Setup:** 8 questions per dataset; SQuAD 64-case conditional pools and Hotpot distractor contexts; request prompt v2. 80 trials, 144 calls, 48,219 input / 6,543 output tokens. Errors score 0.

| Dataset/condition | EM | Answer F1 | Errors | Calls | Seconds | Input/output tokens | Capped calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| squad/none | 0.2500 | 0.2944 | 0 | 8 | 32.37 | 923/192 | 0 |
| squad/bm25 | 0.1250 | 0.3265 | 2 | 16 | 126.44 | 6836/805 | 1 |
| squad/fixed | 0.2500 | 0.4341 | 1 | 16 | 119.00 | 6181/734 | 1 |
| squad/learned | 0.1250 | 0.2421 | 2 | 16 | 125.70 | 5972/778 | 0 |
| squad/iterative | 0.1250 | 0.2905 | 1 | 16 | 114.71 | 4106/720 | 1 |
| hotpot/none | 0.1250 | 0.1806 | 0 | 8 | 28.28 | 1001/182 | 0 |
| hotpot/bm25 | 0.1250 | 0.2361 | 1 | 16 | 116.95 | 6183/757 | 1 |
| hotpot/fixed | 0.1250 | 0.2188 | 0 | 16 | 102.96 | 6368/660 | 0 |
| hotpot/learned | 0.1250 | 0.1875 | 0 | 16 | 100.73 | 6485/647 | 0 |
| hotpot/iterative | 0.0000 | 0.0357 | 4 | 16 | 167.29 | 4164/1068 | 4 |

- **Source:** `docs/T3_PUBLIC_HOST_RESULTS.md`; for v1, `docs/T3_HOST_FAILURE_JOURNAL.md`.
- **Request prompt v1:** it failed (27 trials, 17 schema failures) and the run was terminated.
- **Conclusion:** learned retrieval and iteration show no benefit. Fixed NN's higher SQuAD F1 rests on 8 questions, and its EM only equals no retrieval.
- **Caveats:**
  - Hotpot support and joint F1 are 0.0 for every condition.
  - There were 11 failures: 10 JSON parse errors and 1 invalid support pair.
  - The 11 errored trials lost their retrieval traces, so `all_retrieval_traces_verified=False`.

#### 4.3.6 Complete constrained-decoder run (2d8d063, 2499796, d6e75c8)
- **Tests:** the same 80-trial grid, with strict schemas and a constrained decoder.
- **Setup:** 144 calls, 48,041 input / 6,488 output tokens; 1,430.32 s of summed trial time plus 6.36 s for format startup.

| Dataset/condition | EM | Answer F1 | Errors/8 | Calls | Seconds | Input/output tokens | Capped calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| squad/none | 0.2500 | 0.2944 | 0 | 8 | 44.64 | 923/195 | 0 |
| squad/bm25 | 0.1250 | 0.1969 | 2 | 16 | 200.36 | 6531/907 | 1 |
| squad/fixed | 0.1250 | 0.1538 | 3 | 16 | 201.90 | 6228/919 | 2 |
| squad/learned | 0.1250 | 0.1829 | 1 | 16 | 192.67 | 6024/875 | 1 |
| squad/iterative | 0.1250 | 0.2905 | 0 | 16 | 142.56 | 3964/659 | 0 |
| hotpot/none | 0.1250 | 0.1806 | 0 | 8 | 39.79 | 1001/188 | 0 |
| hotpot/bm25 | 0.1250 | 0.2361 | 1 | 16 | 164.43 | 6183/748 | 0 |
| hotpot/fixed | 0.1250 | 0.2232 | 0 | 16 | 149.69 | 6657/671 | 0 |
| hotpot/learned | 0.1250 | 0.2500 | 0 | 16 | 149.02 | 6366/662 | 0 |
| hotpot/iterative | 0.1250 | 0.1607 | 0 | 16 | 145.27 | 4164/664 | 0 |

- **Source:** `docs/T3_FORMAT_HOST_RESULTS.md`.
- **Results:** all 64 retrieval events verify, including the 7 traces from failed trials. The 7 failures are four 128-token truncations and three invalid ordered citation pairs. No second retrieval occurs, and Hotpot support and joint F1 stay at 0.
- **Caveat:** these results are descriptive only, because both validation and decoding changed relative to the legacy run.

#### 4.3.7 Need-shift diagnosis (d6e75c8)
- **Tests:** whether needs written by the host (instead of the literal public question) change what is retrieved.
- **Setup:** 48 fresh NN retrievals plus 16 BM25 query controls, with no new LLM generations.

| Condition | Comparisons | Changed hash vectors | Changed ordering | Changed membership |
|---|---:|---:|---:|---:|
| BM25 | 16 | 11 | 3 | 2 |
| Fixed NN | 16 | 11 | 4 | 4 |
| Learned NN | 16 | 11 | 5 | 4 |
| Iterative | 16 | 11 | 2 | 2 |
| Total | 64 | 44 | 14 | 12 |

- **Source:** `docs/T3_NEED_SHIFT_DIAGNOSIS.md`.
- **Conclusion:** the host's need copies the question exactly in 0/64 requests. This is one train/eval input-shift mechanism, but it does not explain the weak QA results on its own, since recall falls even without the shift.

#### 4.3.8 Strict plain vs constrained decoder (28e81f4)
- **Tests:** a paired comparison of strictly validated plain decoding against constrained decoding.
- **Setup:** the same 80 trials per arm, paired by question.

| Dataset/condition | Plain F1 | Format F1 | Difference | Errors plain/format | Calls | Seconds plain/format | Paired wins/losses/ties |
|---|---:|---:|---:|---:|---:|---:|---:|
| squad/none | 0.2812 | 0.2944 | +0.0132 | 1/0 | 8 | 36.81/44.64 | 1/0/7 |
| squad/bm25 | 0.2019 | 0.1969 | -0.0050 | 4/2 | 16 | 143.14/200.36 | 2/1/5 |
| squad/fixed | 0.2019 | 0.1538 | -0.0481 | 3/3 | 16 | 132.25/201.90 | 1/1/6 |
| squad/learned | 0.1250 | 0.1829 | +0.0579 | 4/1 | 16 | 140.82/192.67 | 2/0/6 |
| squad/iterative | 0.1250 | 0.2905 | +0.1655 | 3/0 | 16 | 131.15/142.56 | 2/0/6 |
| hotpot/none | 0.0556 | 0.1806 | +0.1250 | 1/0 | 8 | 32.24/39.79 | 1/0/7 |
| hotpot/bm25 | 0.2361 | 0.2361 | +0.0000 | 2/1 | 16 | 133.25/164.43 | 0/0/8 |
| hotpot/fixed | 0.0938 | 0.2232 | +0.1295 | 1/0 | 16 | 113.88/149.69 | 2/0/6 |
| hotpot/learned | 0.0625 | 0.2500 | +0.1875 | 1/0 | 16 | 112.60/149.02 | 2/0/6 |
| hotpot/iterative | 0.0357 | 0.1607 | +0.1250 | 4/0 | 16 | 185.93/145.27 | 1/0/7 |

- **Source:** `docs/T3_STRICT_FORMAT_CONTROL.md`.
- **Results:** failures were 24/80 for strict plain decoding and 7/80 for the constrained decoder. Summed trial time was 1,162.08 s plain vs 1,430.32 s constrained. The strict plain run produced exactly the same prompts and outputs as the legacy run; its 13 extra failures come from stricter validation alone.
- **Conclusion:** the constrained decoder makes output format more reliable. The F1 gains are mixed, and SQuAD fixed and BM25 do not improve.
- **Caveats:** only 8 paired questions per cell; the decoder also changes the emitted needs; costs were measured under varying machine load.

#### 4.3.9 Direct public case-set utility (673add8, 6be9915, afad65b)
- **Tests:** whether each retrieved case helps, measured by removing it with no refill.
- **Setup:** 32 fixed and learned sets from the constrained run. 128 fresh calls, 52,730 input / 3,532 output tokens, 743.30 s. 4 invalid outcomes are kept as unit-loss failures.

| Dataset/condition | Helpful/harmful/zero pairs | Full F1 | Empty F1 | Mean marginal | Mean full-set gain | Mean pair complementarity |
|---|---:|---:|---:|---:|---:|---:|
| squad/fixed | 1/6/9 | 0.1538 | 0.2215 | -0.1476 | -0.0676 | -0.2275 |
| squad/learned | 3/5/8 | 0.1829 | 0.2215 | -0.0223 | -0.0386 | -0.0061 |
| hotpot/fixed | 4/0/12 | 0.2232 | 0.0000 | +0.1295 | +0.2232 | +0.0357 |
| hotpot/learned | 4/0/12 | 0.2500 | 0.0000 | +0.1562 | +0.2500 | +0.0625 |

- **Source:** `docs/T3_PUBLIC_UTILITY_RESULTS.md` and `docs/T3_DIRECT_SET_UTILITY.md`.
- **Results:** of 64 removal pairs, 12 were helpful, 11 harmful and 41 zero. All 32 fresh full-set outputs exactly reproduce their archived originals.
- **Conclusion:** on SQuAD the retrieved sets hurt compared with empty evidence. Hotpot answer gains are positive, but support and joint scores do not change, so need matching cannot stand in for utility.
- **Caveats:** the empty arm uses the answer-stage prompt, so it must not be pooled with the standalone no-retrieval arm. Some harmful differences include formatting failures.

#### 4.3.10 Actual-need request collection and bounded grammar (6fdbd85, 29eaf8f, 9340be6, 790695b)
- **Tests:** whether the host's own train-only needs can be collected for need-metric fitting.
- **Setup:** 192 requests (128 fit + 64 tune) under 128- and 256-token request ceilings. The table is compiled from the doc's prose.

| Request token ceiling | Calls | Valid | Failures | Input tokens | Output tokens | Generation s |
|---|---:|---:|---:|---:|---:|---:|
| 128 | 192 | 190 | 2 | 16,705 | 9,878 | 1,974.32 |
| 256 | 192 | 190 | 2 | 16,705 | 10,134 | 1,874.13 |

- **Source:** `docs/T3_REQUEST_LOOP_DIAGNOSIS.md`, `docs/T3_REQUEST_BUDGET_REPAIR.md` and `docs/T3_ACTUAL_NEED_TRAINING.md`.
- **Repetition failure:** both treatments fail at positions 145 and 174, where the need string repeats a question suffix without end. The other 190 outputs are identical across treatments. A larger budget does not fix this, so no fitting started.
- **Bounded grammar:** a character-by-character replay on the real backend shows the 512-character bound rejects the continuation at character 543.
- **Fresh bounded diagnostic (790695b):** 2 real requests completed, with 118/186 output tokens, 512/510 need characters, 182 total input / 304 output tokens, and 63.77 generation seconds. All exact checks pass. The grammar can close before 512 characters. An initial verifier wrongly required exactly 512 in both cases; it was corrected, and no output changed.
- **Caveat:** the needs still contain the original repeated prefixes, so the grammar fixes format completion, not semantic quality.
- **Compatibility (6fdbd85):** a seed 8 one-epoch default-mode training exactly reproduces the old epoch-1 state.
- **Provenance (ffdf189):** 6 historical checkpoints pass the new provenance bindings (`docs/T3_METRIC_PROVENANCE.md`). The bindings do not prove that the seed was chosen before public results were seen.

#### 4.3.11 Internal output interface preflight (2a4097d, efb7ce1)
- **Tests:** whether a candidate internal integration mode is correct and auditable. The mode mixes the host's next-token probabilities with an activation-weighted histogram of tokens from the retrieved original evidence, using a fixed alpha of 0.1. Host, retrieval and interface parameters stay frozen.
- **Setup:** a real-case preflight for a planned 64-call experiment: 16 exploratory questions × none/prompt/internal/combined, with shared archived needs. Results are compiled from the doc's prose.

| Preflight check | Result |
|---|---|
| Fresh NN-kNN retrievals reproducing recorded sets/state/bank | 16/16 |
| Candidate distances and feature contributions independently verified | 592 |
| Original-case token histograms (actual Qwen tokenizer) | 32 |
| Full/masked synthetic-logit probability checks | 32 pass |
| float32 151,936-token softmax recomputation error (before fix) | up to 6.03e-7 |
| Stored chosen-token diagnostic error, large-vocabulary regression (before fix) | 2.89e-6 |
| float64 / composed-log-probability agreement (after fix) | about 1e-9 |
| T3 tests | 66 pass |

- **Source:** `docs/T3_INTERNAL_INTERFACE.md`.
- **Conclusion:** this is evidence about real cases, geometry and the tokenizer, not a host-answer result. The interface works at the lexical level, so it loses evidence order, and internal-only citation grounding may fail. No quality benefit or architecture choice is established yet.

### 4.4 Open issues

- **The full bounded run is still in progress.** At the time of writing, the 192-query collection was running (session 90847, frozen 149-file source `e26aac9c...`). After it come six matched 20-epoch fits, six independent retrainings, a common-query evaluation, and two exploratory 32-trial public arms (question-trained vs actual-need metric, seed 8, request 256 / answer 128). The repetition inside needs is still present.
- **The internal-interface experiment has not run.** The fresh 64-call experiment and its full independent audit are still needed, and the combined mode is untested on real answers.
- **No public benefit has been shown.** Learned retrieval and iteration show no benefit. SQuAD retrieved sets reduce F1 compared with empty evidence. Hotpot support and joint F1 are 0 in every run. No multi-hop second retrieval has been observed.
- **The need-matching cause is unknown.** The learned metric remains far below BM25. Possible causes include hash collisions, the representation, query/paragraph mismatch, and a gap between the loss and ranking.
- **Small exploratory scale:** one host (Qwen3-0.6B), 8 questions per dataset, conditional 64-case pools rather than corpus-wide retrieval, and no confirmatory inference. The 64 reserved dev questions are still unused.
- **Legacy audit gap:** the 11 errored legacy-plain trials permanently lost their retrieval traces.
- **Format limits:** the constrained decoder still produces truncations and ordered-citation failures. The backend crashes on a boolean `const` and does not enforce citation order or numeric minima.
- **Plan items not yet started:**
  - typed heads and calibration
  - feedback admission, promotion and rollback, including utility-driven updates
  - poisoning recovery
  - standard semantic RAG and stronger, cloud or API host controls
  - KILT/MuSiQue
  - BioASQ
  - agent memory transfer
  - studies with real participants

