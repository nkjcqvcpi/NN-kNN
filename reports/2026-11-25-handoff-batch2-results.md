# Handoff batch 2026-11-25b (second batch): results

Seven studies of the second handoff batch, all complete on r760 under `runs/t1/`. Configs are in
`configs/t1/handoff_20261125b/` (commit a8d363d), seeds 30-32. This report reads only `runs/`: nothing was trained, nothing
under `runs/` was written, no config, code or Table 1 file was changed, and nothing was committed.

| study | what it varies | reference rows (seeds 30-32) |
|---|---|---|
| `handoff_reuseret_r760` | C/H audit and maintenance reference use the adapted outcome (classification adapter) | `plan_full_r760/retention__classification` (same policy, K) |
| `handoff_kcurve_r760` | adds K 0.05 / 0.1 / 0.33 / 0.9 to plan_full's 0.2 / 0.5 / 0.75 | `plan_full_r760/retention__{classification,regression}` |
| `handoff_synret_r760` | 8 retention policies × K 0.2/0.5/0.75 on the 7 synthetic factor sets | random at the same set and K (within study); cores = `plan_full_r760/revision_stress` |
| `handoff_protect_r760` | `boundary_protect_fraction` 0.1 against 0 | `handoff_synret_r760` (same policy, set, K, seed) |
| `handoff_evrule_r760` | coverage/reachability keys (activation threshold, zero reachability) and evidence rule | plan_full `coverage_reachability` / `provenance_only` |
| `handoff_btransform_r760` | bias normalisation median/MAD instead of within-cohort percentile | plan_full `bias_normalized` |
| `handoff_qsmooth_r760` | Q smoothing s = 0.1 / 0.5 / 2.0 / 5.0 (plan_full s = 1.0) | plan_full `provenance_only` |

Each section below was produced by one agent and then re-derived by a second agent with its own dumper and analysis code
(adversarial check). Where the check contradicted the first summary with evidence, the check's numbers and wording are used
here; claims that neither could verify from `runs/` are marked **unresolved**. A list of the corrections is at the end.

## How the numbers were produced

- **Project summaries** (format unchanged), written on r760 by `tools/t1_summarize.py` under
  `/mnt/largedata/NN-KNN/neural-cbr/results/t1/<study>/<experiment>/`:
  - `handoff_reuseret_r760/reuse_retention__classification/`: `summary.csv`, `paired_vs_random_K{0.2,0.5,0.75}.csv`
    (test_accuracy_post/pre, test_ece_post/pre lower-is-better, test_net_beneficial_flips).
  - `handoff_kcurve_r760/{kcurve,kcurve_noK005}__{classification,regression}/`: `summary.csv`, `paired_vs_full_memory_K0.33.csv`.
  - `handoff_synret_r760/synthetic_retention__classification/` and
    `handoff_protect_r760/synthetic_protect_boundary010__classification/`: `summary.csv`, `paired_vs_random_K0.5.csv`
    (kept_fraction_corrupted/duplicate lower-is-better, kept_fraction_rare higher-is-better).
  - `handoff_{evrule,btransform,qsmooth}_r760/*/summary.csv` (20 experiments). No `paired_vs_*` files: each experiment has a
    single policy and its reference rows sit in another study, which the tool cannot pair.
- **Cross-study pairings** are not something the summary tools do. Read-only dumpers (`load_run`, the `row` record, and for
  evrule/btransform/qsmooth also per-case `case_statistics` incl. `archive_state`) wrote jsonl dumps under `/var/tmp` on r760 and
  accel; the tables below come from them. Scripts: first pass `/var/tmp/nnknn-hb-rk/`, `/var/tmp/nnknn-hb2/`,
  `/var/tmp/nnknn-hb-evq/`; checks `/var/tmp/nnknn-adv/`, `/var/tmp/nnknn-chk3/`, `/var/tmp/nnknn-adv-evq/` (r760 dumpers,
  accel analysis).
- **Integrity.** No unit missing, no load error, n = 3 in every cell, one event file per unit. Unit counts: reuseret 720,
  kcurve 1896, synret 504, protect 252, evrule 720, btransform 189, qsmooth 756. The event-file host of every handoff unit equals
  the host of its plan_full reference rows (q-lab, xps8960 = `haotang-lab`, r760 = `cs-peng-research`). `fullmem_*`, `n_before`
  and `sel_K` equal plan_full in every paired unit, and for evrule/btransform/qsmooth per-case C, H, R, A and cohort are identical
  to plan_full in every unit: the cores are the same and every key acts only on the retention/ranking side.
- plan_full also has seeds 33/34 for some datasets; they were ignored.

**Paired comparison rule (all sections).** For a dataset (× K) cell, d_s = condition − reference on seed s. **better**: the
mean of d favours the condition and at least 2 of the 3 seeds favour it; **worse**: symmetrically; **identical**: all three d_s
are exactly 0; otherwise **tie**. Counts are written b/t/w/i (better/tie/worse/identical) over datasets. In verdict strings
B/T/W/= (or b/t/w/i) give one letter per K. With 3 seeds this is a consistency count, not a significance test.

**Metric directions.** Higher is better: accuracy (overall, in-domain, shifted, rare), kept_fraction_rare. Lower is better: RMSE
(standardised target), ECE, test loss (NLL for classification, MSE for regression), kept_fraction_corrupted,
kept_fraction_duplicate. "pre" metrics are after the 30-epoch finetune at lr × 0.1 on the retained memory; `test_nofinetune_*`
are before it.

---

## 1. `handoff_reuseret_r760` (C/H from the adapted outcome)

Same tuned, masked core as plan_full, frozen with no finetune. The classification adapter (nominal_residual_scores, combined
loss) is trained, the C/H audit and the maintenance reference use the **adapted** query outcome, then the 8 policies are applied
at K 0.2/0.5/0.75. Two outcomes per unit:
- `test_accuracy_pre`: retrieval-only prediction after the selection. Paired with plan_full `test_nofinetune_accuracy_pre`, it
  isolates the effect of the evidence source on the selection.
- `test_accuracy_post`: prediction through the adapter (the full system).

**Table 1a.** Policy means over the 10 classification datasets, test accuracy (mean over datasets of the 3-seed means; rank among
the 8 policies in brackets).

| policy | K0.2 plan nofinetune | K0.2 reuse pre | K0.2 reuse post | K0.5 plan nf | K0.5 pre | K0.5 post | K0.75 plan nf | K0.75 pre | K0.75 post |
|---|---|---|---|---|---|---|---|---|---|
| full_memory | 0.902 (1) | 0.902 (1) | 0.908 (1) | 0.902 (1) | 0.902 (1) | 0.908 (1) | 0.902 (1) | 0.902 (1) | 0.908 (1) |
| random | 0.822 (6) | 0.822 (6) | 0.828 (6) | 0.873 (6) | 0.873 (6) | 0.879 (6) | 0.894 (7) | 0.894 (7) | 0.900 (6) |
| stratified | 0.838 (2) | 0.838 (2) | 0.841 (4) | 0.872 (7) | 0.872 (7) | 0.876 (7) | 0.895 (6) | 0.895 (6) | 0.899 (8) |
| bias_current | 0.811 (7) | 0.811 (7) | 0.820 (7) | 0.885 (3) | 0.885 (2) | 0.893 (2) | 0.900 (2) | 0.900 (4) | 0.901 (5) |
| provenance_only | 0.831 (4) | 0.829 (5) | 0.838 (5) | 0.885 (2) | 0.883 (3) | 0.885 (4) | 0.898 (5) | 0.901 (2) | 0.904 (3) |
| bias_normalized | 0.838 (3) | 0.838 (3) | 0.842 (3) | 0.882 (4) | 0.882 (4) | 0.886 (3) | 0.899 (3) | 0.899 (5) | 0.900 (7) |
| kcenter | 0.790 (8) | 0.790 (8) | 0.791 (8) | 0.864 (8) | 0.864 (8) | 0.872 (8) | 0.894 (8) | 0.894 (8) | 0.901 (4) |
| coverage_reachability | 0.829 (5) | 0.831 (4) | 0.843 (2) | 0.876 (5) | 0.878 (5) | 0.879 (5) | 0.899 (4) | 0.900 (3) | 0.904 (2) |

The mean per-dataset rank moves by at most 0.3 between plan and reuse-pre (provenance_only at K0.75, 4.25 → 3.95).

**Table 1b.** Paired counts against plan_full test_nofinetune at the same policy and K (b/t/w/i over 10 datasets).

| policy | pre accuracy K0.2 / K0.5 / K0.75 | post accuracy K0.2 / K0.5 / K0.75 | post ECE K0.2 / K0.5 / K0.75 |
|---|---|---|---|
| full_memory, random, stratified, bias_current, bias_normalized, kcenter | 0/0/0/10 in every cell | 0-4 better and 0-3 worse per cell; mean d +0.001 to +0.009 | 7-9 worse per cell |
| provenance_only | 0/5/0/5; 1/4/0/5; 0/3/1/6 (mean d −0.002, −0.002, +0.003) | 3/5/1/1; 4/4/1/1; 1/6/1/2 | 7, 9, 8 worse |
| coverage_reachability | 1/2/1/6; 1/1/1/7; 1/3/1/5 (mean d +0.001, +0.002, +0.002) | 4/3/2/1; 4/4/1/1; 3/5/1/1 | 7, 8, 8 worse |

Post ECE mean d over all policies is +0.118 to +0.201 (lower is better, so worse).

**Within-study counts against random at the same K (accuracy, b/t/w[/i]).**
- provenance_only: plan 6/0/4, 7/1/2, 2/6/2; reuse-pre 6/0/4, 6/2/2, 2/5/2/1; reuse-post 6/0/4, 5/2/3, 1/6/2/1.
- coverage_reachability: plan 4/3/3, 3/5/2, 3/5/2; reuse-pre 4/3/3, 3/5/2, 2/5/2/1; reuse-post 5/3/2, 2/4/4, 2/6/1/1.

**Adapter effect (post − pre within the study)**, over 80 dataset × policy cells per K: K0.2 25/37/9/9 (mean +0.006), K0.5
23/32/9/16 (+0.005), K0.75 16/36/7/21 (+0.004). The gain is concentrated on zebra_special (+0.046 at full memory) and wine (+0.009).

### Interpretation

Using the adapted outcome changes only the two policies that read C/H-derived evidence: provenance_only (through Q) and
coverage_reachability (through the reference stream the adapter feeds). 26/90 and 25/90 of their rows differ in pre accuracy,
and the policy-mean |d| is ≤ 0.003. The effect is not pure noise for coverage_reachability: it shifts the same way at every K on
two datasets, zebra_special better (+0.015, +0.030, +0.008) and tabarena_363626 (credit-g) worse (−0.004, −0.008, −0.011); these
are exactly its 1 better and 1 worse per cell. Per-dataset |mean d| reaches 0.030 (provenance_only zebra K0.75, 2 test cases
over 3 seeds).

The order of the policy means does change at K0.75 (plan nf → reuse pre): provenance_only 5 → 2, bias_current 2 → 4,
bias_normalized 3 → 5; at K0.5 bias_current and provenance_only swap 2/3, at K0.2 provenance_only and coverage_reachability swap
4/5. But places 2-5 at K0.75 lie within 0.0025 accuracy, mean per-dataset ranks move by at most 0.3, and the counts against
random are almost unchanged. So the ranking is **unchanged within noise, not identical**, and "better adaptation is not confused
with better case selection" holds in that weaker form.

The adapter adds about +0.005 accuracy on top of every policy and makes calibration much worse (post ECE median 0.262 against
0.056 retrieval-only; means 0.257 and 0.083). Neither touches case selection.

### Flags
- **Identical rows by design.** All 540 pre rows of full_memory, random, stratified, bias_current, bias_normalized and kcenter are
  bit-identical to plan_full test_nofinetune (accuracy and ECE): these policies never read C/H (bias_normalized ranks on B, which
  does not depend on C/H). Expected "key with no effect"; it also confirms the core was reproduced exactly on the same hosts.
- **No selection fields.** The reuse_retention row has no `sel_n_evidenced`/`sel_*` fields, so how many cases the selection
  changed cannot be read; only outcome differences are visible.
- **ECE** is the raw adapter softmax (not temperature-scaled): an adapter calibration issue, not a retention issue.
- full_memory rows are identical across K (30/30), K is ignored. No NaN in the reuseret rows (plan_full synthetic_diag has NaN in
  `*_accuracy_boundary`, not used).
- **Thin evidence** for the two affected policies: 3 seeds and mostly one-test-case differences, though coverage_reachability's
  zebra_special/credit-g shift is consistent in direction.

---

## 2. `handoff_kcurve_r760` (quality against memory, K 0.05-0.9)

Adds K 0.05/0.1/0.33/0.9 to plan_full's K 0.2/0.5/0.75; same cores, finetune 30 epochs at lr × 0.1, 8 policies, seeds 30-32.
Classification iris, wine, zebra and regression body_fat, yacht have no K0.05 (the config omits them; the stated reason, that
round(0.05 n) is below the floor of 2 cases per cohort, was **not verified against data**). So the K0.05 column averages 7
classification / 9 regression datasets, the others 10 / 11, and the K0.05 column is not directly comparable (full_memory reads
0.886 there).

**Consistency.** `fullmem_metric_pre` equals plan_full in 1896/1896 rows; full_memory test metrics at K0.33/K0.9 equal the
plan_full K0.5/K0.2 rows (30/30 classification, 33/33 regression), so the two studies share the core and are pooled into one
curve. `sel_n_after == sel_K` in every non-full row. No NaN apart from the synthetic_diag boundary-accuracy field.

**Table 2a.** Classification test accuracy after finetune, mean over datasets (higher is better).

| policy | K0.05 (7 ds) | K0.1 | K0.2 | K0.33 | K0.5 | K0.75 | K0.9 |
|---|---|---|---|---|---|---|---|
| full_memory | 0.886 | 0.903 | 0.903 | 0.903 | 0.903 | 0.903 | 0.903 |
| random | 0.846 | 0.843 | 0.855 | 0.869 | 0.883 | 0.902 | 0.900 |
| stratified | 0.837 | 0.856 | 0.860 | 0.867 | 0.882 | 0.904 | 0.902 |
| bias_current | 0.818 | 0.824 | 0.856 | 0.869 | 0.899 | 0.901 | 0.899 |
| provenance_only | 0.845 | 0.833 | 0.841 | 0.868 | 0.889 | 0.900 | 0.899 |
| bias_normalized | 0.834 | 0.835 | 0.867 | 0.861 | 0.890 | 0.899 | 0.899 |
| kcenter | 0.824 | 0.816 | 0.847 | 0.875 | 0.889 | 0.896 | 0.903 |
| coverage_reachability | 0.843 | 0.838 | 0.838 | 0.862 | 0.885 | 0.900 | 0.902 |

Median gap to full memory (accuracy), K0.05 → K0.9, with the number of datasets whose gap is ≥ −2 pp in brackets (one-sided:
it also counts datasets well above full memory; with |gap| ≤ 2 pp random reads 1/7, 3/10, 4/10 at K0.05/0.1/0.2 because
synthetic_diag random is +2.1, +2.7, +3.5 pp above full memory):
- random: −0.024 (2/7), −0.026 (4/10), −0.019 (5/10), −0.011 (7/10), −0.010 (6/10), 0.000 (10/10), −0.001 (10/10)
- bias_current: −0.043, −0.018, −0.018, −0.009, 0.000 (9/10), 0.000, 0.000
- kcenter: −0.056 (0/7), −0.045 (2/10), −0.033, −0.010, −0.001 (8/10), −0.005, 0.000

Before finetune (`test_nofinetune_accuracy_pre`) the curve is much steeper: random 0.775 / 0.792 / 0.822 / 0.838 / 0.873 /
0.894 / 0.901; kcenter 0.712 / 0.727 / 0.790 / 0.827 / 0.864 / 0.894 / 0.904. Finetuning recovers about half of the loss at
K ≤ 0.33.

**Table 2b.** Regression test RMSE after finetune, mean over datasets (standardised target, lower is better).

| policy | K0.05 (9 ds) | K0.1 | K0.2 | K0.33 | K0.5 | K0.75 | K0.9 |
|---|---|---|---|---|---|---|---|
| full_memory | 0.419 | 0.407 | 0.407 | 0.407 | 0.407 | 0.407 | 0.407 |
| random | 0.522 | 0.489 | 0.466 | 0.438 | 0.420 | 0.412 | 0.407 |
| stratified | 0.526 | 0.486 | 0.459 | 0.435 | 0.420 | 0.409 | 0.408 |
| bias_current | 0.482 | 0.445 | 0.424 | 0.414 | 0.410 | 0.409 | 0.408 |
| provenance_only | 0.551 | 0.496 | 0.451 | 0.436 | 0.429 | 0.419 | 0.415 |
| bias_normalized | 0.506 | 0.454 | 0.428 | 0.419 | 0.411 | 0.410 | 0.407 |
| kcenter | 0.484 | 0.445 | 0.421 | 0.412 | 0.409 | 0.407 | 0.406 |
| coverage_reachability | 0.527 | 0.479 | 0.439 | 0.436 | 0.433 | 0.419 | 0.409 |

Median RMSE ratio − 1 against full memory (datasets within 5% in brackets):
- random: +15% (2/9), +15% (2/11), +10% (4/11), +4% (6/11), +3% (7/11), +1% (8/11), +0.5% (11/11)
- bias_current: +11%, +5%, +1%, +1%, +0.3% (10/11), …
- kcenter: +13%, +10%, +2% (8/11), +0.3%, −0.1%, …

The mean ratio is dominated by energy_efficiency (full-memory RMSE 0.0525): at K0.05 the policies sit at 3.79-5.75× full memory
(+279% to +475%), at K0.1 at 2.42-4.96× (kcenter +142%, bias_current +216%). The maximum is 5.75× (coverage_reachability K0.05).
Hence the median is reported.

**Table 2c.** Paired counts after finetune against random at the same K (b/t/w/i over datasets).

Classification accuracy:

| policy | K0.05 | K0.1 | K0.2 | K0.33 | K0.5 | K0.75 | K0.9 |
|---|---|---|---|---|---|---|---|
| full_memory | 6/0/1/0 | 7/2/1/0 | 7/1/2/0 | 6/2/2/0 | 7/1/1/1 | 0/7/2/1 | 3/4/2/1 |
| stratified | 1/3/3/0 | 5/4/1/0 | 3/4/1/2 | 1/7/1/1 | 2/6/1/1 | 2/4/0/4 | 0/4/0/6 |
| bias_current | 1/2/4/0 | 2/4/4/0 | 3/4/3/0 | 3/4/3/0 | 6/3/0/1 | 1/6/2/1 | 2/7/1/0 |
| provenance_only | 2/4/1/0 | 3/1/6/0 | 3/1/6/0 | 4/1/5/0 | 5/1/3/1 | 0/8/2/0 | 2/5/3/0 |
| bias_normalized | 2/1/4/0 | 2/6/2/0 | 6/2/2/0 | 1/7/2/0 | 5/5/0/0 | 0/8/2/0 | 2/6/2/0 |
| kcenter | 1/2/4/0 | 0/6/4/0 | 4/1/5/0 | 4/3/3/0 | 5/3/2/0 | 1/8/1/0 | 2/3/4/1 |
| coverage_reachability | 4/2/1/0 | 4/2/4/0 | 3/2/5/0 | 2/3/5/0 | 5/2/2/1 | 0/7/3/0 | 3/4/2/1 |

Regression RMSE:

| policy | K0.05 | K0.1 | K0.2 | K0.33 | K0.5 | K0.75 | K0.9 |
|---|---|---|---|---|---|---|---|
| full_memory | 9/0/0/0 | 10/0/1/0 | 10/0/1/0 | 9/1/1/0 | 10/0/1/0 | 6/2/3/0 | 7/1/3/0 |
| stratified | 3/3/3/0 | 7/2/2/0 | 8/1/2/0 | 5/3/3/0 | 6/4/1/0 | 6/1/4/0 | 4/0/7/0 |
| bias_current | 8/0/1/0 | 10/1/0/0 | 10/0/1/0 | 9/2/0/0 | 9/0/2/0 | 7/0/4/0 | 6/2/3/0 |
| provenance_only | 3/0/6/0 | 5/2/4/0 | 8/0/3/0 | 6/2/3/0 | 7/0/4/0 | 5/2/4/0 | 3/2/6/0 |
| bias_normalized | 6/1/2/0 | 10/1/0/0 | 10/0/1/0 | 8/1/2/0 | 10/0/1/0 | 5/0/6/0 | 7/1/3/0 |
| kcenter | 7/1/1/0 | 8/0/3/0 | 9/1/1/0 | 11/0/0/0 | 8/0/3/0 | 7/1/3/0 | 6/2/3/0 |
| coverage_reachability | 4/0/5/0 | 8/2/1/0 | 9/0/2/0 | 5/2/4/0 | 7/0/4/0 | 5/3/3/0 | 4/2/5/0 |

**Against full_memory at the same K:**
- Classification, K ≤ 0.33: every policy is worse on 5-10 datasets, except bias_normalized at K0.05 (1/2/4/0, worse on 4 of 7).
- Classification, K0.75-0.9: mostly tie or identical, at most 3 better (e.g. provenance_only K0.9 0/5/1/4, coverage_reachability
  K0.9 2/3/1/4).
- Regression: two cells have a thin majority better: kcenter K0.9 6/2/3/0 and coverage_reachability K0.5 6/0/5/0 (median ratio
  changes 0.1-0.4%); kcenter K0.33 is 5/1/5/0. At K0.75 every policy is worse on most datasets: random 3/2/6, stratified 1/2/8,
  bias_current 2/3/6, provenance_only 4/2/5, bias_normalized 1/5/5, kcenter 4/1/6, coverage_reachability 2/2/7 (median +0.07% to
  +0.8%).

### Interpretation

**Where policies beat random.**
- *Regression:* bias_current and bias_normalized are better on 8-10 of 11 datasets and worse on at most 2 at K0.1-0.5; kcenter is
  better on 8-11 but worse on 3 at K0.1 and K0.5; stratified is behind them. The advantage shrinks but does not disappear at
  K ≥ 0.75: still better on 6-7 of 11 (bias_current 7/0/4 and kcenter 7/1/3 at K0.75; bias_normalized 7/1/3, bias_current and
  kcenter 6/2/3 at K0.9), by small margins.
- *Classification:* no policy beats random consistently along the curve. Clearest wins: bias_current and bias_normalized at K0.5
  (6/3/0 and 5/5/0), bias_normalized at K0.2 (6/2/2), coverage_reachability at K0.05 (4/2/1).
- *provenance_only:* without finetune it beats random at K0.2-0.5 (6/0/4, 7/0/3, 7/1/2); after finetune it is worse than random at
  K0.1-0.33 (worse on 6, 6, 5 of 10), so finetuning erases its selection advantage. On regression after finetune its large-K
  weakness is carried by car (+31.7% RMSE at K0.9). abalone is +19% to +26% only **without** finetune; after finetune
  provenance_only on abalone is −0.1% to −1.2%, no worse than full memory. The idea that it "keeps the wrong cases where the
  success tolerance makes Q uninformative" is a hypothesis; nothing in the rows tests it.

**Where policies beat full memory.** Classification: no reliable gain (at most 3 better at any K). Regression: thin majorities in
two cells (above) with sub-1% margins. Compression does not reliably improve quality. At K0.75 classification mostly ties full
memory, but regression is worse on 5-8 of 11 datasets (median +0.07% to +0.8%): dropping 25% of memory is "no cost" for
classification and "small but consistent cost" for regression.

**Where quality collapses.**
- *Classification after finetune:* the median gap stays within about 1 pp down to K0.33-0.5 and opens to 2-5 pp at K ≤ 0.1. zebra
  loses 12-44 pp at K ≤ 0.33 under every policy (max bias_current K0.1, −43.9). zebra_special loses 3-24 pp, except kcenter
  (+0.8 at K0.33). iris loses 4-17 pp at K0.1. wine loses heavily only under kcenter (−9.3 at K0.1, −13.9 at K0.2); other policies
  at K0.1 sit between −2.8 and +1.9.
- *Regression (median RMSE ratio):* random, stratified and provenance_only break between K0.33 and K0.2; bias_current,
  bias_normalized and kcenter stay within 2% down to K0.2 and break between K0.2 and K0.1. At K ≤ 0.1 the medians are +5% to +31%,
  and energy_efficiency sits at 2.4-5.8× full-memory RMSE.

| policy | K0.5 | K0.33 | K0.2 | K0.1 |
|---|---|---|---|---|
| random | +3.2% | +4.3% | +10.1% | |
| stratified | +2.3% | +3.8% | +9.5% | |
| provenance_only | +1.6% | +0.85% | +5.5% | |
| bias_current | | | +1.1% | +5.2% |
| bias_normalized | | | +0.06% | +8.8% |
| kcenter | | | +1.6% | +10.3% |

### Flags
- **full_memory identical across K by design**; its "vs random" row is random's curve inverted.
- **K0.05 column uses a different dataset set.** That explains the non-monotone means (random 0.846 at K0.05 vs 0.843 at K0.1); on
  the same 7 datasets random reads 0.846 → 0.860, monotone. K0.05 evidence is thin (7/9 datasets, a handful of cases on small
  sets). The floor explanation for the missing datasets is **unverified**.
- **synthetic_diag: random beats full memory** after finetune by +2.1, +2.7, +3.5, +1.9 pp at K0.05/0.1/0.2/0.33
  (coverage_reachability +2.4, +1.7 pp at K0.05/0.1). test_n is 260. Whether this is noise or a real gain from dropping
  mislabelled cases is **unresolved**; it is the exception to "compression never improves quality".
- **abalone provenance_only nofinetune gap is K-independent**: +20% even at K0.9 (10% removed) while every other policy converges
  to +0.2%; finetune removes the gap. Possibly a systematic removal of one cohort; **unresolved**, worth a look.
- **Read the regression mean with care**: energy_efficiency dominates it; car carries the provenance_only and
  coverage_reachability failures. Use medians and per-dataset counts.
- **Small test sets.** test_n per seed: zebra 22, zebra_special 44, iris 30, wine 36. Noise on these is several pp, so
  classification ties at K ≥ 0.5 are not evidence of equality.
- **Finetuned vs untrained retention ranking differ** (provenance_only and coverage_reachability gain relative to the others without
  finetune). Say which metric is meant when quoting a "policy ranking".

---

## 3. `handoff_synret_r760` (8 retention policies × K on the 7 synthetic factor sets)

Test set (fixed by the generator): 180 in-domain, 60 shifted, 20 rare, 260 total. Training set: 360 base cases (15% label-flipped,
40% in noise and combined), 60 duplicates (180 in redundant), 12 rare cases (0 in uncovered_rare). K is the kept fraction.
Recording host r760. Same cores as plan_full: `fullmem_accuracy_pre` equals plan_full revision_stress `test_M0_accuracy_pre` for
21/21 (set, seed) and is constant across policies and across synret/protect. (This equality is for the core **before** finetune;
the full_memory condition's `test_accuracy_pre` differs from it in 9/21, see flags.)

**Table 3a.** test_accuracy_pre at K = 0.5, mean ± sd over 3 seeds (full_memory is identical across K). K0.2 and K0.75 versions
are in `/var/tmp/nnknn-hb2/ana/main.out` (accel). At K0.2 on noise, coverage_reachability scores 0.914, random 0.871, full_memory 0.783.

| set | full_memory | random | stratified | kcenter | coverage_reach. | provenance_only | bias_current | bias_normalized |
|---|---|---|---|---|---|---|---|---|
| reference | 0.947 ± 0.040 | 0.954 ± 0.046 | 0.968 ± 0.031 | 0.933 ± 0.058 | 0.944 ± 0.037 | 0.951 ± 0.041 | 0.963 ± 0.035 | 0.932 ± 0.087 |
| overlap | 0.721 ± 0.091 | 0.708 ± 0.110 | 0.727 ± 0.114 | 0.744 ± 0.109 | 0.737 ± 0.094 | 0.721 ± 0.106 | 0.714 ± 0.074 | 0.710 ± 0.078 |
| noise | 0.783 ± 0.019 | 0.833 ± 0.060 | 0.847 ± 0.051 | 0.851 ± 0.053 | 0.878 ± 0.048 | 0.864 ± 0.053 | 0.810 ± 0.088 | 0.800 ± 0.089 |
| redundant | 0.944 ± 0.036 | 0.956 ± 0.029 | 0.958 ± 0.025 | 0.951 ± 0.043 | 0.955 ± 0.033 | 0.949 ± 0.026 | 0.944 ± 0.037 | 0.927 ± 0.071 |
| uncovered_rare | 0.940 ± 0.067 | 0.938 ± 0.071 | 0.941 ± 0.066 | 0.944 ± 0.067 | 0.953 ± 0.066 | 0.945 ± 0.063 | 0.933 ± 0.063 | 0.914 ± 0.071 |
| shift | 0.851 ± 0.076 | 0.873 ± 0.114 | 0.876 ± 0.110 | 0.829 ± 0.065 | 0.821 ± 0.026 | 0.854 ± 0.080 | 0.924 ± 0.066 | 0.883 ± 0.131 |
| combined | 0.556 ± 0.045 | 0.559 ± 0.064 | 0.544 ± 0.052 | 0.563 ± 0.075 | 0.574 ± 0.069 | 0.563 ± 0.044 | 0.567 ± 0.088 | 0.567 ± 0.077 |

**Table 3b.** Kept fraction corrupted / duplicate / rare, descriptive mean pooled over 7 sets × 3 seeds (6 sets for rare).

| policy | K0.2 corr / dup / rare | K0.5 corr / dup / rare | K0.75 corr / dup / rare |
|---|---|---|---|
| full_memory | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 |
| random | 0.189 / 0.213 / 0.167 | 0.480 / 0.523 / 0.528 | 0.738 / 0.763 / 0.713 |
| stratified | 0.193 / 0.224 / 0.204 | 0.480 / 0.535 / 0.505 | 0.746 / 0.754 / 0.713 |
| kcenter | 0.242 / **0.107** / 0.157 | 0.555 / **0.270** / 0.245 | 0.793 / **0.414** / 0.394 |
| coverage_reachability | **0.058** / 0.234 / **0.315** | **0.272** / 0.667 / **0.741** | **0.569** / 0.954 / **0.912** |
| provenance_only | **0.065** / 0.328 / **0.477** | **0.322** / 0.733 / **0.769** | **0.659** / 0.783 / **0.847** |
| bias_current | 0.255 / 0.342 / **0.000** | 0.474 / 0.713 / **0.000** | 0.822 / 0.877 / **0.009** |
| bias_normalized | 0.257 / 0.331 / **0.000** | 0.483 / 0.710 / **0.000** | 0.798 / 0.887 / **0.009** |

Pooled group accuracy at K = 0.5: in-domain random 0.864, coverage_reachability 0.876, full_memory 0.853; shifted random 0.694,
bias_current 0.744, bias_normalized 0.735, full_memory 0.677; rare 0.952 for every policy except bias_current 0.871,
bias_normalized 0.705 and provenance_only 0.945. Pooled test NLL: full_memory 1.351; random 0.757 / 1.215 / 1.259 at K 0.2/0.5/0.75.

**Table 3c.** Counts against random at the same set and K, b/t/w (+ id), over 7 sets × 3 K = 21 cells (18 for kept_fraction_rare).

| policy vs random | acc overall | acc in-domain | acc shifted | acc rare | test NLL | kept corrupted | kept duplicate | kept rare |
|---|---|---|---|---|---|---|---|---|
| full_memory | 4/1/16 | 3/1/17 | 2/6/11 +2 id | 21 id | 2/1/18 | 0/0/21 | 0/0/21 | 18/0/0 |
| stratified | 6/11/3 +1 id | 5/13/3 | 4/11/2 +4 id | 21 id | 11/6/4 | 3/7/11 | 5/7/7 +2 id | 0/8/3 +7 id |
| kcenter | 7/6/8 | 5/7/9 | 7/9/5 | 21 id | 12/2/7 | 3/0/18 | **21/0/0** | 2/4/12 |
| coverage_reachability | 9/4/8 | 8/7/6 | 5/8/8 | 0/2/0 +19 id | 5/4/12 | **20/1/0** | 2/4/15 | **16/1/1** |
| provenance_only | 7/6/8 | 7/4/10 | 4/4/9 +4 id | 0/1/0 +20 id | 4/5/12 | **21/0/0** | 0/3/18 | **17/1/0** |
| bias_current | 7/5/9 | 5/6/10 | **12/4/5** | 0/6/4 +11 id | 6/5/10 | 10/2/9 | 0/1/20 | **0/0/18** |
| bias_normalized | 6/8/7 | 5/5/11 | **12/5/4** | 0/8/3 +10 id | 6/3/12 | 9/3/9 | 0/1/20 | **0/0/18** |

**Per-set overall accuracy against random** (one letter per K 0.2/0.5/0.75):

| policy | reference | overlap | noise | redundant | uncovered_rare | shift | combined |
|---|---|---|---|---|---|---|---|
| coverage_reachability | wwt | bbb | bbb | ttw | wbt | www | bbw |
| provenance_only | www | bbt | tbb | twt | tbb | www | btw |
| bias_current | wbw | btt | ttw | www | www | bbb | btb |
| bias_normalized | wtt | btt | ttw | www | tww | btb | bbb |

Against full_memory, every pruning policy has more better than worse cells on overall accuracy: random 16/1/4, stratified 18/1/2,
kcenter 13/3/5, coverage_reachability 12/6/3, provenance_only 14/3/4, bias_current 10/6/5, bias_normalized 12/5/4.

**Rare-group test accuracy** (mean over 3 seeds, K 0.2 / 0.5 / 0.75 per cell):

| policy | reference | overlap | noise | redundant | uncovered_rare | shift | combined |
|---|---|---|---|---|---|---|---|
| full, random, stratified, kcenter | 1/1/1 | 1/1/1 | 1/1/1 | 1/1/1 | .67/.67/.67 | 1/1/1 | 1/1/1 |
| coverage_reachability | 1/1/1 | 1/1/1 | 1/1/1 | 1/1/1 | .65/.67/.60 | 1/1/1 | 1/1/1 |
| provenance_only | 1/1/1 | 1/1/1 | 1/1/1 | 1/1/1 | .67/.62/.67 | 1/1/1 | 1/1/1 |
| bias_current | 1/1/.67 | 1/.83/1 | .97/.65/.63 | 1/1/.80 | .65/.62/.30 | 1/1/.67 | 1/1/1 |
| bias_normalized | 1/.67/.93 | 1/.83/1 | .98/.65/.63 | 1/.78/1 | .67/.33/.35 | 1/.67/.93 | 1/1/1 |

### Interpretation
- **Corrupted cases are removed by coverage_reachability and provenance_only** (better than random on kept_fraction_corrupted in
  20 and 21 of 21 cells; ~6% kept at K0.2 against 19% for random). Both rank by evidence of harm (Q = (C+1)/(C+H+2), and
  reachability), so mislabelled cases that keep retrieving the wrong label score low. The bias policies are no better than random
  (10/2/9, 9/3/9).
- **Duplicates are removed only by kcenter** (21/0/0): a k-center cover keeps one point where near-copies sit together. Every
  evidence- or bias-based policy keeps duplicates more often than random (provenance_only 0/3/18, bias policies 0/1/20,
  coverage_reachability 2/4/15): a duplicate and its source each get retrieved and look well supported, and the scores do not
  account for redundancy.
- **Rare cases are kept by coverage_reachability (16/1/1) and provenance_only (17/1/0).** bias_current and bias_normalized remove
  every rare training case at K0.2 and 0.5 and nearly all at 0.75 (0/0/18): the 12 rare cases sit in a far sub-cluster, are seldom
  retrieved and get low bias. kcenter also keeps fewer rare cases than random (2/4/12).
- **No policy keeps clean and rare cases while removing duplicates.** The evidence policies remove corrupted and keep rare cases
  but keep duplicates; kcenter removes duplicates but keeps more corrupted (3/0/18 vs random) and fewer rare ones.
- **Accuracy follows only partly.** Overall accuracy against random is near even for every policy (coverage_reachability 9/4/8 to
  bias_normalized 6/8/7). Gains are where corruption dominates: coverage_reachability is better on noise and overlap at all K and on
  combined at K0.2/0.5; provenance_only is better on noise at K0.5/0.75. On reference and shift (15% corruption) provenance_only is
  worse at every K and coverage_reachability mostly worse.
- **Bias policies win on shifted-domain accuracy** (12/4/5 and 12/5/4), on shift, combined **and uncovered_rare** (bbb on all three
  for both; bias_current per set: reference wbt, overlap bww, noise ttb, redundant wwt; bias_normalized the same except redundant
  ttw). On shift at K0.5 bias_current reaches 0.750 (sd 0.217) shifted accuracy against 0.533 (sd 0.410) for random, so the seeds
  vary widely.
- **Rare accuracy** saturates once any rare case is kept (it is 1.0 on seeds 31/32 of uncovered_rare even with no rare training
  case), but it does drop in some seeds when every rare case is removed: the only policies that differ from random are the two
  that keep 0 rare cases, they are never better (0/6/4, 0/8/3), and their losses spread over 4-6 sets. Per seed the evidence is
  sparse (single seeds going to 0, e.g. bias_normalized K0.5 zeros on reference s30, noise s30, shift s30, uncovered_rare s30/s32,
  plus 0.35 on redundant s30).
- **Pruning vs full memory.** Pruning plus finetune is more often better than keeping everything (random 16/1/4), and pruned models
  have much lower NLL (0.76 against 1.35 at K0.2). This is **not** explained by "cleaner" memory for random, which keeps about K of
  the corrupted cases: even before finetune random K0.2 has pooled NLL 1.100 against 1.329 for full memory
  (`test_nofinetune_loss_pre`), so having fewer cases accounts for part of the drop. Whether the finetune itself contributes is
  **unresolved** (see the no-op flag).

### Flags
1. **full_memory finetune is a no-op in 7 of 21 (set, seed)** (suspicious, unresolved): `test_loss_pre` exactly equals
   `test_nofinetune_loss_pre` on reference s30/s32, noise s30, redundant s30, shift s30/s32, combined s32. The same happens in 10-14
   of 63 units for the evidence and bias policies, 1 for random, 0 for stratified and kcenter. It looks like a best-checkpoint
   restore at epoch 0 or similar; check before attributing "pruning + finetune beats full memory" to the finetune.
2. **Rare-group test accuracy is a weak retention measure** (see the corrected table above): identical to random in all 21 cells
   for full_memory, stratified and kcenter; 19 for coverage_reachability, 20 for provenance_only. On uncovered_rare (no rare
   training cases) it is 0, 1, 1 on seeds 30-32 for every policy.
3. **shift and reference share their training data** by generator design (only the shifted test group differs): kept fractions,
   in-domain and rare accuracy, n_before and sel_n_evidenced are identical in 72/72 rows. Not independent sets for retention.
4. **Random kept fractions** repeat only partly across sets: kept_fraction_duplicate is the same across the n = 432 sets, but
   kept_fraction_corrupted/clean_base differ between {reference, overlap, shift} and {noise, combined} (e.g. s30 K0.2 corrupted
   0.2188 vs 0.2398); uncovered_rare (n = 420) has the same corrupted fraction as reference.
5. **Boundary accuracy is NaN on 5 of 7 sets** (no boundary-band test case at center_scale 3); only overlap and combined have it.
   No other NaN.
6. **Rows identical to another policy**: random = stratified in 2 of 63 triples (noise K0.2 s32, combined K0.2 s32);
   bias_current = bias_normalized in 1 (overlap K0.2 s32). full_memory constant over K by design.
7. **Evidence is thin**: only 183-410 of 420-552 cases are evidenced per (set, seed) (reference s31: 183 of 432); the rest get
   neutral Q 0.5 in provenance_only.

---

## 4. `handoff_protect_r760` (boundary_protect_fraction 0.1 against 0)

Protects floor(0.1 n) cases nearest to another class in the learned metric: 43 (42 on uncovered_rare, n = 420; 55 on redundant,
n = 552). In synret `sel_n_protected` is 0 everywhere. Reference: the same policy, set, K and seed in synret.

**Table 4a.** Counts of fraction 0.1 against 0, b/t/w (+ id), over 21 cells (18 for kept rare).

| policy | acc overall | acc in-domain | acc shifted | acc rare | test NLL | kept corrupted | kept duplicate | kept rare |
|---|---|---|---|---|---|---|---|---|
| random | 4/9/8 | 3/11/7 | 5/7/5 +4 id | 21 id | 6/6/9 | 5/2/**14** | **15**/6/0 | 0/1/12 +5 id |
| bias_current | 3/10/8 | 4/11/5 +1 id | 1/16/4 | 1/11/1 +8 id | 7/3/11 | 5/1/**15** | **16**/5/0 | 0/1/0 +17 id |
| provenance_only | 4/6/9 +2 id | 4/9/5 +3 id | 2/8/7 +4 id | 0/2/0 +19 id | 4/11/6 | 1/4/**16** | **13**/3/3 +2 id | 0/6/3 +9 id |
| coverage_reachability | 4/7/8 +2 id | 3/8/8 +2 id | 4/8/4 +5 id | 0/4/0 +17 id | 7/3/9 +2 id | 0/1/**18** +2 id | **15**/1/1 +4 id | 0/4/9 +5 id |

**Table 4b.** Fraction 0.1 / fraction 0, descriptive mean pooled over 7 sets × 3 seeds.

| policy | acc K0.2 | acc K0.5 | acc K0.75 | kept corr. K0.2 | kept corr. K0.5 | kept dup. K0.2 | kept rare K0.2 | kept rare K0.5 |
|---|---|---|---|---|---|---|---|---|
| random | 0.839 / 0.842 | 0.830 / 0.832 | 0.832 / 0.831 | 0.294 / 0.189 | 0.563 / 0.480 | 0.181 / 0.213 | 0.079 / 0.167 | 0.389 / 0.528 |
| bias_current | 0.830 / 0.841 | 0.819 / 0.836 | 0.817 / 0.820 | 0.356 / 0.255 | 0.573 / 0.474 | 0.243 / 0.342 | 0 / 0 | 0 / 0 |
| provenance_only | 0.842 / 0.836 | 0.829 / 0.835 | 0.822 / 0.824 | 0.238 / 0.065 | 0.438 / 0.322 | 0.199 / 0.328 | 0.361 / 0.477 | 0.755 / 0.769 |
| coverage_reachability | 0.840 / 0.851 | 0.833 / 0.837 | 0.833 / 0.830 | 0.239 / 0.058 | 0.398 / 0.272 | 0.183 / 0.234 | 0.130 / 0.315 | 0.681 / 0.741 |

### Interpretation
- **No accuracy gain.** No policy has more better than worse cells (overall 3-4 better against 8-9 worse).
- **It brings corrupted cases back.** All four policies get worse on kept_fraction_corrupted (14-18 of 21 cells worse). For
  provenance_only and coverage_reachability the corrupted share kept at K0.2 rises from ~0.06 to ~0.24, undoing most of what they
  achieve in section 3. Likely mechanism: a label-flipped case lies inside another class's cloud, so "nearest to another class"
  selects mislabelled cases first. This is an inference from kept fractions; the rows do not record which protected cases are
  corrupted, so it **cannot be checked from runs/**.
- **It displaces rare cases** (random 0.167 → 0.079 at K0.2; coverage_reachability 0.315 → 0.130): the 43 protected slots come out
  of the K budget, and rare cases lie far from other classes, so they are never protected.
- **The one gain is fewer duplicates** (13-16 of 21 better): fewer free slots remain for the over-kept duplicates.
- **Net:** as specified (nearest enemy in the learned metric, no label-noise check), boundary protection works against
  harmful-case removal on these sets. It would need to exclude low-Q or unevidenced cases, or use a margin that ignores cases
  surrounded by the other class.

### Flags
1. **At K0.75 some units are fully identical to fraction 0** on every test_* and kept_* field: provenance_only 4 of 21 (reference
   s30/s32, shift s30/s32), coverage_reachability 8 of 21 (reference s30-32, overlap s30, shift s30-32, combined s30), bias_current
   1 of 21 (redundant s31), random 0. None at K0.2 or K0.5. The protected cases are already inside the kept 75% there, so the
   setting has no effect at high K.
2. **bias_current is identical on kept_fraction_rare in 17 of 18 cells** (it keeps 0 rare cases either way). Its rare-test accuracy
   change (pooled 0.945 → 0.812 at K0.2) comes from single seeds: sparse, as in section 3.
3. Protected counts follow floor(0.1 n) (42 on uncovered_rare, 55 on redundant): consistent.
4. No NaN apart from boundary accuracy on 5 sets; no unit missing.
5. **protect_rare_cohort_below was not tested**, as the config explains: cohorts are classes, so the threshold 10 protects nothing.

---

## 5. `handoff_evrule_r760` (coverage/reachability keys and evidence rule)

Five arms, 9 classification tasks (no digits) and 7 regression tasks (no abalone, airfoil, car, regression wine). Arms `act 0.01`,
`act 0.2` and `zero = inf` are coverage_reachability, compared with plan_full coverage_reachability (activation_threshold 0.05,
zero_reachability `zero`). Arms `ev (1, 0.0)` and `ev (5, 0.5)` are provenance_only with evidence rule (R ≥ a, A ≥ b), compared
with plan_full provenance_only (R ≥ 2, A ≥ 0.1). Main metric: classification test accuracy (higher better), regression test RMSE
(lower better). **Kept cases swapped** = cases in the kept set that plan_full did not keep (= cases dropped, K is fixed).
`sel_n_evidenced` equals an independent recount under each arm's own rule in every unit.

**Table 5a.** Verdict per dataset at K 0.2 / 0.5 / 0.75 (B/T/W/=), then the mean kept swaps per seed over the three K.

| task | dataset | act 0.01 | act 0.2 | zero = inf | ev (1, 0.0) | ev (5, 0.5) |
|---|---|---|---|---|---|---|
| cls | balance | T T = (2) | W T W (74) | T T B (10) | = = = (4) | = B W (16) |
| cls | breast_cancer | = = = (0) | = T = (2) | B T T (8) | = = T (13) | B W W (36) |
| cls | iris | = = = (0) | = = = (0) | = = T (4) | = = = (2) | T W = (8) |
| cls | synthetic_diag | B T = (2) | W T W (33) | W B W (39) | = T B (20) | T W B (33) |
| cls | tabarena_363614 | B T = (2) | B T T (83) | W T = (2) | = = = (8) | = T T (18) |
| cls | tabarena_363626 | B T W (5) | B B B (79) | W W W (91) | T T T (8) | = B T (25) |
| cls | wine | = = = (0) | = = = (0) | T = = (1) | = T = (4) | T T T (5) |
| cls | zebra | = = = (0) | B = = (2) | T T T (3) | = = = (1) | T T T (6) |
| cls | zebra_special | W B = (4) | B T W (21) | T T W (30) | = = T (1) | = T W (4) |
| reg | body_fat | T = = (0) | W W W (27) | W B T (37) | = B B (5) | B T T (9) |
| reg | diabets | = = = (0) | W T W (41) | W B B (71) | = B B (6) | B B B (17) |
| reg | energy_efficiency | W T W (37) | B B B (73) | = = = (0) | = = T (0) | = W W (5) |
| reg | student_performance | W T T (2) | B T T (65) | W W B (20) | = T W (10) | = W W (22) |
| reg | tabarena_363625 | B W B (16) | W W W (79) | W W T (15) | = = T (0) | = T W (18) |
| reg | tabarena_363675 | = = = (0) | B B B (128) | W W B (62) | = = T (2) | = T T (37) |
| reg | yacht | = = = (0) | = = = (0) | T = = (0) | = T T (2) | W W W (5) |

**Table 5b.** Counts b/t/w/i over datasets; "loss" = the same for `test_loss_pre` (lower better); swaps per unit with the share of
the kept set; units that reproduce plan_full in every numeric row field.

| arm | task | K 0.2 | K 0.5 | K 0.75 | all K | loss (all K) | kept swaps per unit (share) | units identical to plan_full |
|---|---|---|---|---|---|---|---|---|
| act 0.01 | cls | 3/1/1/4 | 1/4/0/4 | 0/0/1/8 | 4/5/2/16 | 3/8/3/13 | 1.6 (2.0%) | 54/81 |
| act 0.01 | reg | 1/1/2/3 | 0/2/1/4 | 1/1/1/4 | 2/4/4/11 | 2/4/4/11 | 7.8 (5.0%) | 37/63 |
| act 0.2 | cls | 4/0/2/3 | 1/5/0/3 | 1/1/3/4 | 6/6/5/10 | 7/4/9/7 | 32.7 (23.6%) | 25/81 |
| act 0.2 | reg | 3/0/3/1 | 2/2/2/1 | 2/1/3/1 | 7/3/8/3 | 6/5/7/3 | 59.2 (37.8%) | 9/63 |
| zero = inf | cls | 1/4/3/1 | 1/5/1/2 | 1/3/3/2 | 3/12/7/5 | 8/10/8/1 | 20.7 (19.9%) | 16/81 |
| zero = inf | reg | 0/1/5/1 | 2/0/3/2 | 3/2/0/2 | 5/3/8/5 | 5/3/8/5 | 29.4 (26.3%) | 17/63 |
| ev (1, 0.0) | cls | 0/1/0/8 | 0/3/0/6 | 1/3/0/5 | 1/7/0/19 | 5/8/2/12 | 6.8 (3.7%) | 3/81 |
| ev (1, 0.0) | reg | 0/0/0/7 | 2/2/0/3 | 2/4/1/0 | 4/6/1/10 | 4/6/1/10 | 3.6 (2.2%) | not counted |
| ev (5, 0.5) | cls | 1/4/0/4 | 2/4/3/0 | 1/4/3/1 | 4/12/6/5 | 6/7/11/3 | 16.8 (12.2%) | not counted |
| ev (5, 0.5) | reg | 2/0/1/4 | 1/3/3/0 | 1/2/4/0 | 4/5/8/4 | 5/4/8/4 | 16.0 (8.5%) | not counted |

(For the ev arms `sel_n_evidenced` differs from plan_full by construction; only ev (1, 0.0) classification was counted for full
identity, 3 of 81 units.)

**Evidenced cases** (`sel_n_evidenced`, 3-seed mean, arm / plan_full). act and zero arms: unchanged by construction.
ev (1, 0.0) rises 1.8-35.7% (balance 315/284, breast_cancer 240/177 (+35.7%), iris 64/50, synthetic_diag 292/225, wine 47/35
(+34.0%), body_fat 120/101, student_performance 255/202, tabarena_363675 591/529, energy_efficiency 423/416 (+1.8%)).
ev (5, 0.5) falls 16-65% (breast_cancer 65/177, iris 18/50, zebra 21/45, wine 21/35, synthetic_diag 135/225,
tabarena_363675 338/529, yacht 30/44).

**synthetic_diag, share of mislabelled cases kept** (Δ arm − plan_full at K 0.2 / 0.5 / 0.75, lower is better):

| arm | Δ K 0.2 | Δ K 0.5 | Δ K 0.75 |
|---|---|---|---|
| act 0.01 | −0.02 (tie) | −0.02 (tie) | 0 (identical) |
| act 0.2 | +0.016 (worse) | +0.075 (worse) | +0.127 (worse) |
| **zero = inf** | **+0.656 (worse)** | **+0.565 (worse)** | **+0.415 (worse)** |
| ev (1, 0.0) | 0 (identical) | −0.054 (better) | −0.127 (better) |
| ev (5, 0.5) | −0.021 (better) | +0.090 (worse) | +0.032 (tie) |

plan_full coverage_reachability keeps 0.021 / 0.201 / 0.532 of the mislabelled cases; under zero = inf 0.677 / 0.766 / 0.947.

### Interpretation
- **act 0.01: little effect.** ~2-5% of the kept set changes; 27 of 48 cells identical; iris, wine, zebra, yacht, diabets and
  tabarena_363675 are identical in every unit. Sizeable changes on energy_efficiency (37 swaps, worse at K0.2 and K0.75) and
  tabarena_363625 (15.9 swaps per seed, B W B).
- **act 0.2: sensitive, direction depends on the task.** 24-38% of the kept set is replaced, including evidenced cases (balance:
  91-150 evidenced cases per seed change state). Better at every K on credit-g, energy_efficiency and tabarena_363675; worse at every
  K on body_fat and tabarena_363625; worse at K0.2 and K0.75 on balance, synthetic_diag and diabets. Net 13 better / 13 worse over
  48 cells: a higher threshold is not better in general. On synthetic_diag it keeps more mislabelled cases at every K.
- **zero = inf: the setting matters; plan_full `zero` is preferable, mainly on corruption.** Ranking zero-reachability cases first
  keeps 41-66 points more of the mislabelled cases on synthetic_diag; that they are "mostly the mislabelled ones" is consistent
  with this but was **not tested directly** (no per-case corruption label in `case_statistics`). Net 15 worse / 8 better over 48
  cells. On credit-g accuracy is worse at every K, but the K0.2 mean −0.085 comes almost entirely from seed 32 (per-seed d +0.018,
  −0.006, −0.267), and `test_loss_pre` on credit-g is **better** at all three K; over all classification cells the loss counts are
  an even 8/8. So the preference for `zero` rests on the synthetic_diag corrupted share and the regression counts (5 better /
  8 worse), not on a uniform accuracy result. Only energy_efficiency is untouched (no such cases).
- **ev (1, 0.0): mildly favourable, not "no effect".** Metric verdicts: 29 identical, 13 tie, 5 better, 1 worse; better on body_fat
  and diabets at K0.5/0.75 and on synthetic_diag at K0.75 (where it also keeps fewer corrupted cases). But many "=" cells hide
  kept-set changes (tabarena_363614 K0.5 [0, 0, 35] swaps, K0.75 [0, 20, 19]; balance K0.75 [0, 25, 11]; breast_cancer K0.5
  [3, 5, 24]; wine K0.75 [2, 10, 2]), and the loss column is 9 better / 3 worse. Loosening the rule ranks more cases by Q instead
  of the unevidenced fallback order; where that changes anything it slightly helps.
- **ev (5, 0.5): more often worse (14 worse / 8 better), but not because evidence is too sparse.** The sparse-evidence
  classification sets are mostly ties (wine T T T, zebra T T T, iris T W =); only breast_cancer (B W W) fits that story. The worse
  cells are mostly regression tasks with ample evidence: energy_efficiency (349 of 460 evidenced) = W W, student_performance (140
  of 389) = W W, tabarena_363625 (305 of 515) T W, yacht W W W. At K0.2 the kept set is still mostly evidenced (iris keeps 18 with
  ~18 evidenced, breast_cancer ~68 with 65); only at K0.5/0.75 is most of it chosen without outcome evidence. diabets is better at
  every K. Why the stricter rule hurts on well-evidenced regression is **unresolved**.

---

## 6. `handoff_btransform_r760` (bias normalisation median/MAD vs within-cohort percentile)

Single arm `bias_normalized` with `statistics.bias_normalization: median_mad`, against plan_full `bias_normalized`, all 10
classification and 11 regression tasks.

**Table 6a.** Per dataset: mean d (condition − reference), verdict, kept swaps on seeds 30, 31, 32. Classification d in accuracy
(positive good); regression d in RMSE (negative good).

| task | dataset | K 0.2 | K 0.5 | K 0.75 |
|---|---|---|---|---|
| cls | balance | −0.0160 W [3,3,4] | = [0,0,0] | −0.0027 T [3,2,8] |
| cls | breast_cancer | −0.0205 W [54,24,3] | −0.0058 T [53,35,0] | +0.0029 T [60,0,19] |
| cls | digits | −0.0037 W [47,25,16] | = [30,0,0] | −0.0009 W [22,15,14] |
| cls | iris | −0.1000 W [14,4,8] | +0.0333 B [19,9,6] | +0.0222 T [16,5,5] |
| cls | synthetic_diag | −0.0077 W [8,5,8] | = [0,0,0] | −0.0167 T [8,7,6] |
| cls | tabarena_363614 | −0.0312 T [9,8,2] | = [0,0,0] | +0.0011 T [6,4,4] |
| cls | tabarena_363626 | −0.0060 W [2,7,0] | = [0,0,0] | +0.0010 T [2,7,3] |
| cls | wine | −0.0093 W [3,13,2] | +0.0278 T [0,18,0] | +0.0093 T [3,12,1] |
| cls | zebra | −0.0152 T [0,1,2] | = [0,0,0] | = [1,0,0] |
| cls | zebra_special | −0.0076 T [0,2,2] | = [0,0,0] | −0.0076 T [1,4,2] |
| reg | abalone | −0.0016 B [127,110,111] | = [0,0,0] | +0.0000 T [29,40,46] |
| reg | airfoil | −0.0018 B [13,11,5] | = [0,0,0] | +0.0005 W [9,8,11] |
| reg | body_fat | +0.0025 W [5,3,4] | = [0,0,0] | +0.0033 W [5,5,4] |
| reg | car | **+0.1612 W** [237,220,305] | +0.0007 T [24,77,0] | +0.0013 W [36,168,176] |
| reg | diabets | −0.0017 B [5,5,4] | = [0,0,0] | +0.0026 W [3,4,4] |
| reg | energy_efficiency | +0.0180 W [8,15,18] | = [0,0,0] | −0.0006 T [10,10,10] |
| reg | student_performance | +0.0037 W [4,7,4] | = [0,0,0] | −0.0001 B [8,5,5] |
| reg | tabarena_363625 | −0.0003 B [3,5,5] | = [0,0,0] | −0.0007 B [2,7,6] |
| reg | tabarena_363675 | +0.0089 W [12,10,12] | = [0,0,0] | +0.0010 W [9,7,11] |
| reg | wine | +0.0032 W [26,17,5] | = [0,0,0] | −0.0005 B [25,13,9] |
| reg | yacht | **+0.1248 W** [16,16,7] | = [0,0,0] | +0.0000 T [6,6,4] |

**Table 6b.** Counts b/t/w/i.

| task | K 0.2 | K 0.5 | K 0.75 | all K | loss (all K) |
|---|---|---|---|---|---|
| classification | 0/3/7/0 | 1/2/0/7 | 0/8/1/1 | 1/13/8/8 | 5/10/8/7 |
| regression | 4/0/7/0 | 0/1/0/10 | 3/3/5/0 | 7/4/12/10 | 6/5/12/10 |

Kept cases swapped: on average 8.7% (classification) and 5.5% (regression) of the kept set. synthetic_diag mislabelled share kept:
Δ +0.011 (tie), 0 (identical), +0.048 (worse).

### Interpretation
- **K0.5: almost no effect, by construction.** 17 of 21 cells have an identical metric; 16 of them have 0 swaps (digits keeps the
  same accuracy despite 30 swaps on seed 30). Both transforms are monotone within a cohort and split at the cohort median
  (percentile 0.5 = z 0), so the half-memory cut keeps the same cases unless ties or unequal cohort sizes move the boundary
  (breast_cancer, digits, iris, wine, car).
- **K0.2: median/MAD is worse.** Classification worse on 7 of 10 and never better; regression worse on 7 of 11, with large losses
  on car (+0.161 RMSE, per-seed d +0.028, +0.219, +0.237; ~250 swaps) and yacht (+0.125, per-seed +0.248, +0.102, +0.024). iris
  −0.100 is carried by seed 32 alone (−0.267; others −0.033, 0). The mechanism was confirmed from `archive_state`: at a tight
  budget the global cut compares tails across cohorts and MAD scaling lets low-spread cohorts dominate, while percentile keeps them
  balanced. On car plan_full keeps exactly 159 cases per regression bin at K0.2; median/MAD keeps 333/222/63/69/108 (s30),
  196/267/234/72/26 (s31), 281/294/207/2/11 (s32); yacht goes from 7-8 per bin to 16/15/2/2/2.
- **K0.75: mostly tie** (3 better / 11 tie / 6 worse).
- **Verdict:** median/MAD is not a better B transform; keep the within-cohort percentile.

---

## 7. `handoff_qsmooth_r760` (Q smoothing s for provenance_only)

Arms s = 0.1, 0.5, 2.0, 5.0 against plan_full provenance_only (s = 1.0), all 10 classification and 11 regression tasks.
Q = (C+s)/(C+H+2s). The number of evidenced cases is unchanged by construction.

**Table 7a.** Verdicts at K 0.2 / 0.5 / 0.75, then mean kept swaps per seed.

| task | dataset | s 0.1 | s 0.5 | s 2.0 | s 5.0 |
|---|---|---|---|---|---|
| cls | balance | T T = (7) | T T = (1) | B T = (1) | B T = (1) |
| cls | breast_cancer | T = = (2) | T = = (1) | = T = (1) | T T = (1) |
| cls | digits | T W T (14) | T = T (2) | T T = (1) | T T = (2) |
| cls | iris | T T = (1) | T = = (0) | = = = (0) | = = = (0) |
| cls | synthetic_diag | T T = (13) | T = = (2) | B T = (1) | B T = (2) |
| cls | tabarena_363614 | T = = (2) | = = = (0) | T = = (0) | W = = (0) |
| cls | tabarena_363626 | W T T (22) | T W T (4) | T W T (3) | B W T (5) |
| cls | wine | = = = (0) | = = = (0) | = = = (0) | = = = (0) |
| cls | zebra | = = = (0) | = T = (0) | = = = (0) | = = = (0) |
| cls | zebra_special | T W = (5) | T T = (1) | = W = (1) | B T = (1) |
| reg | abalone | = = = (0) | = = = (0) | = = = (0) | = = = (0) |
| reg | airfoil | T B T (3) | T W T (1) | T T = (1) | T T = (1) |
| reg | body_fat | W W = (4) | W = = (1) | B = = (0) | B = = (1) |
| reg | car | W W = (15) | W W = (5) | B B = (4) | B B = (7) |
| reg | diabets | T T B (4) | B = T (2) | W = T (1) | B = B (1) |
| reg | energy_efficiency | = = = (0) | = = = (0) | = = = (0) | = = = (0) |
| reg | student_performance | W W = (5) | W T = (1) | B = = (1) | B = = (1) |
| reg | tabarena_363625 | T B B (17) | B W W (2) | W B = (1) | B B T (2) |
| reg | tabarena_363675 | T T W (29) | T T T (4) | B W T (3) | B T T (5) |
| reg | wine | B W = (84) | B W = (26) | B W = (20) | B W = (37) |
| reg | yacht | T = = (0) | = = = (0) | = = = (0) | = = = (0) |

**Table 7b.** Counts b/t/w/i (as Table 5b).

| s | task | K 0.2 | K 0.5 | K 0.75 | all K | loss (all K) | kept swaps per unit (share) | units identical to plan_full |
|---|---|---|---|---|---|---|---|---|
| 0.1 | cls | 0/7/1/2 | 0/4/2/4 | 0/2/0/8 | 0/13/3/14 | 2/11/9/8 | 6.6 (6.4%) | 44/90 |
| 0.1 | reg | 1/5/3/2 | 2/2/4/3 | 2/1/1/7 | 5/8/8/12 | 4/9/8/12 | 14.7 (5.5%) | 45/99 |
| 0.5 | cls | 0/7/0/3 | 0/3/1/6 | 0/2/0/8 | 0/12/1/17 | 5/13/1/11 | 1.2 (1.4%) | 54/90 |
| 0.5 | reg | 3/2/3/3 | 0/2/4/5 | 0/3/1/7 | 3/7/8/15 | 3/7/8/15 | 3.8 (1.3%) | 55/99 |
| 2.0 | cls | 2/3/0/5 | 0/4/2/4 | 0/1/0/9 | 2/8/2/18 | 4/11/4/11 | 0.8 (0.9%) | 56/90 |
| 2.0 | reg | 5/1/2/3 | 2/1/2/6 | 0/2/0/9 | 7/4/4/18 | 7/4/4/18 | 2.8 (0.8%) | 61/99 |
| 5.0 | cls | 4/2/1/3 | 0/5/1/4 | 0/1/0/9 | 4/8/2/16 | 4/9/6/11 | 1.3 (1.5%) | 52/90 |
| 5.0 | reg | 7/1/0/3 | 2/2/1/6 | 1/2/0/8 | 10/5/1/17 | 10/5/1/17 | 5.0 (1.4%) | 59/99 |

synthetic_diag mislabelled share kept (Δ, lower better):

| s | Δ K 0.2 | Δ K 0.5 | Δ K 0.75 |
|---|---|---|---|
| 0.1 | +0.074 (worse) | −0.021 (tie) | 0 (identical) |
| 0.5 | 0 (identical) | −0.010 (tie) | 0 (identical) |
| 2.0 | −0.016 (better) | +0.005 (tie) | 0 (identical) |
| 5.0 | −0.016 (better) | +0.005 (tie) | 0 (identical) |

### Interpretation
- **Low sensitivity, consistent direction.** s ≥ 1 changes only 0.8-1.5% of the kept set. Stronger smoothing is mildly favourable at
  K0.2: s = 5.0 is better on 11 of 21 cells and worse on 1 (regression 7/1/0/3). Smaller s is unfavourable: s = 0.1 gives 0 better
  / 3 worse on classification and 5 / 8 on regression.
- **Why:** with small s, Q of a thinly evidenced case goes to 0 or 1, so thin evidence outranks solid evidence; larger s pulls it
  toward 0.5.
- **Where it matters:** car, body_fat and student_performance are worse for s < 1 and better for s > 1; regression wine is better
  at K0.2 and worse at K0.5 for every s; synthetic_diag and balance are better at K0.2 with s ≥ 2. The student_performance s 5.0
  K0.2 gain (−0.0435 RMSE) is carried by seed 32 (−0.108).
- **K0.75 is identical in 65 of 84 cells.**
- s = 1.0 (plan_full) is reasonable; s = 2-5 would be slightly better at small K.

---

## Flags (surprising or suspicious, all studies)

1. **synret: full_memory finetune is a no-op in 7 of 21 (set, seed)** (`test_loss_pre` == `test_nofinetune_loss_pre`), and in
   10-14 of 63 units for evidence/bias policies, 1 for random. Unresolved; possibly a best-checkpoint restore at epoch 0. Check
   before attributing synthetic pruning gains to the finetune.
2. **kcurve: synthetic_diag random above full memory** by 1.9-3.5 pp at K ≤ 0.33 (test_n 260). Noise or real: unresolved.
3. **kcurve: abalone provenance_only nofinetune +20% at every K, even K0.9**; finetune removes it. Unresolved (possible cohort
   removal).
4. **reuseret: coverage_reachability shifts consistently** on zebra_special (better) and credit-g (worse) at every K; thin (3 seeds)
   but not single-cell noise.
5. **Keys with no effect on some tasks (degenerate statistics, not evidence the key is irrelevant).**
   - qsmooth identical in every unit on abalone and energy_efficiency, and on yacht except one swap at s 0.1, K0.2 (a tie);
     energy_efficiency has H = 0 for 100% of evidenced cases (yacht 100/100/93.9%), so Q is monotone in C for any s > 0; abalone
     has only 82/131/105 of 2505 cases evidenced, so the cut is decided by the unevidenced fallback order.
   - act 0.01 and act 0.2 reproduce plan_full on iris, wine and yacht (act 0.01 also on breast_cancer, zebra, diabets,
     tabarena_363675). zero = inf does not (iris up to 8 swaps and a K0.75 tie; wine and yacht 1 swap and a tie).
   - The config header expected act 0.01/0.2 to reproduce plan_full on all top_k 3 tasks; at 0.2 that fails on breast_cancer (up to
     7 swaps) and zebra (up to 4 swaps, better at K0.2). Only iris and yacht had been checked in the header.
   - reuseret: 540/540 pre rows of the six C/H-free policies identical to plan_full (by design).
6. **"=" does not mean "no effect".** Several identical-metric cells hide large kept-set changes (up to 35 swaps per seed: ev (1, 0.0)
   tabarena_363614, balance, breast_cancer, wine; ev (5, 0.5) iris K0.75 [17, 3, 12]; btransform digits K0.5 [30, 0, 0]).
   Classification accuracy is coarse. 8 cells have both metric and loss identical with only 1-3 swaps (inert swaps, probably never
   in any query's masked top-k, as in the first batch).
7. **Protect at K0.75** is fully identical to fraction 0 in 4 (provenance_only), 8 (coverage_reachability) and 1 (bias_current) of
   21 units.
8. **Single-seed-driven means**: btransform iris K0.2 (−0.100, seed 32 −0.267), btransform yacht K0.2 (+0.125), evrule zero = inf
   credit-g K0.2 (−0.085, seed 32 −0.267), qsmooth s 5.0 student_performance K0.2 (−0.0435, seed 32 −0.108). Verdicts hold under the
   2-of-3 rule; do not quote the sizes without the seed spread.
9. **Sparse evidence**: ev (5, 0.5) iris ~18 of 90, zebra 21 of 66, wine 21 of 106 evidenced; abalone ~106 of 2505 in every study;
   plan_full iris seed 31 only 15; synret 183-410 of 420-552. Rare-group test accuracy rests on 20 test cases and single seeds.
10. **Small classification test sets** (zebra 22, zebra_special 44, iris 30, wine 36 per seed): ties are not equality.
11. **NaN** only in `*_accuracy_boundary` (synthetic_diag seed 31 in plan_full-paired studies; 5 of 7 synthetic sets in
    synret/protect). No NaN in any main metric, loss or kept fraction.
12. **Unverified explanations**: missing K0.05 datasets (cohort floor), protection selecting mislabelled cases, zero-reachability
    cases being mislabelled, provenance_only keeping wrong cases where Q is uninformative. None is testable from the recorded rows.
13. **Scratch collision on r760**: two analysis agents briefly shared `/var/tmp/nnknn-hb2/`; the evrule/btransform/qsmooth dumper
    script there was overwritten by the synret/protect dumper (its outputs were already written and moved to
    `/var/tmp/nnknn-hb-evq/`; the independent check reproduced every table). The first-batch dumper
    (`/var/tmp/nnknn-hb/dump.py`) writes a wrong `_host` for paths containing `K0.x`; the first-batch report took hosts from event
    file basenames, so its numbers are unaffected.

## Conclusions

- **reuseret:** using the adapted outcome for C/H changes only provenance_only and coverage_reachability, by ≤ 0.003 policy-mean
  accuracy; the policy ordering is unchanged within noise (positions 2-5 at K0.75 reshuffle within 0.0025). Adaptation and case
  selection are not confounded. The adapter adds ~+0.005 accuracy and badly worsens ECE.
- **kcurve:** compression never reliably beats full memory. Classification tolerates K ≥ 0.75 at no cost and loses 2-5 pp median
  at K ≤ 0.1 (zebra worst). Regression pays a small but consistent cost already at K0.75; bias_current, bias_normalized and kcenter
  beat random clearly at K0.1-0.5 and hold within 2% down to K0.2, while random/stratified/provenance_only break between K0.33
  and K0.2.
- **synret:** coverage_reachability and provenance_only remove corrupted cases and keep rare ones but keep duplicates; kcenter alone
  removes duplicates; bias policies drop every rare case but win on shifted accuracy. Accuracy gains appear only where corruption
  dominates (noise, overlap, combined). Pruned models beat full memory more often than not, but the full_memory finetune no-op must
  be checked first.
- **protect:** boundary protection 0.1 gives no accuracy gain, brings corrupted cases back (corrupted share at K0.2 ~0.06 → ~0.24
  for the evidence policies), displaces rare cases, and only reduces duplicates. Not useful as specified.
- **evrule:** act 0.01 is near-inert; act 0.2 is strongly task-dependent (13/13); zero = inf is worse (keeps far more mislabelled
  cases), so keep `zero`; ev (1, 0.0) is mildly favourable; ev (5, 0.5) is worse, mainly on well-evidenced regression tasks.
- **btransform:** median/MAD is no better: identical at K0.5 by construction, clearly worse at K0.2 (cohort imbalance, car and yacht),
  mostly tie at K0.75. Keep the within-cohort percentile.
- **qsmooth:** low sensitivity; s < 1 is mildly harmful, s = 2-5 mildly helpful at K0.2; s = 1 is a reasonable default.

## Corrections applied from the adversarial checks

reuseret / kcurve:
1. Table 1b, six C/H-free policies, post-accuracy mean d: +0.001 to +0.009 (was +0.014, which belongs to coverage_reachability K0.2).
2. "Ranks move by one place at most" removed: at K0.75 provenance_only 5 → 2, bias_current 2 → 4, bias_normalized 3 → 5.
3. "Ordering is the same with either evidence source" weakened to "unchanged within noise, not identical".
4. "No dataset is better/worse in more than one cell" removed: coverage_reachability is consistently better on zebra_special and
   worse on credit-g at every K; per-dataset |d| reaches 0.030.
5. energy_efficiency at K ≤ 0.1: 3.79-5.75× at K0.05, 2.42-4.96× at K0.1 (was "+260% to +480%" and "up to 4-9×").
6. Classification vs full memory at K ≤ 0.33: exception bias_normalized K0.05 (worse on 4 of 7).
7. Zebra losses: zebra 12-44 pp; zebra_special 3-24 pp except kcenter (+0.8 at K0.33) (was "10-38 pp under every policy" for both).
8. Test sizes per seed: zebra 22, zebra_special 44, iris 30, wine 36 (was "66-plus").
9. "Within 2 pp" counts stated as one-sided (gap ≥ −2 pp); |gap| version given for random.
10. Regression vs full memory: two thin-majority-better cells (kcenter K0.9, coverage_reachability K0.5); "no policy beats full
    memory on most datasets at any K" removed.
11. "25% of memory can be dropped at no cost" restricted to classification; regression K0.75 worse on 5-8 of 11 (median +0.07% to +0.8%).
12. Regression knee corrected (random/stratified/provenance_only between K0.33 and K0.2; bias policies and kcenter between K0.2 and K0.1).
13. kcenter worse than random on 3 of 11 at K0.1 and K0.5; advantage over random persists (6-7 of 11) at K ≥ 0.75.
14. Wine at K0.1: loses heavily only under kcenter; kcenter K0.2 −13.9.
15. abalone +20% is nofinetune only (finetuned −0.1% to −1.2%); only car supports the large-K provenance_only weakness; the Q-tolerance
    mechanism marked as hypothesis.
16. Added flags: synthetic_diag random above full memory; abalone nofinetune K-independent gap; coverage_reachability consistent
    shift; K0.05 floor explanation unverified.

synret / protect:
17. Pooled rare accuracy at K0.5: provenance_only 0.945 added to the exceptions.
18. Rare-accuracy flag rewritten with the corrected per-set table: the bias policies do fall below 1.0; "21/21 identical to random"
    holds for 3 policies (full_memory, stratified, kcenter), not 4; rare accuracy saturates once any rare case is kept.
19. Random kept fractions: only kept_fraction_duplicate repeats across same-size sets; corrupted/clean_base differ.
20. NLL explanation ("fewer, cleaner cases") rejected for random; nofinetune NLL already lower (1.100 vs 1.329).
21. Bias policies' shifted-accuracy wins also include uncovered_rare (bbb).
22. Core equality restricted to `fullmem_accuracy_pre`; full_memory condition differs in 9/21; new flag: finetune no-op in 7/21
    full_memory (set, seed) and 10-14/63 units of evidence/bias policies (unresolved).
23. Protect flag 1: coverage_reachability 8 of 21 identical units at K0.75 (was 6), plus bias_current 1 of 21.
24. Protection mechanism stated as unverifiable from runs/.

evrule / btransform / qsmooth:
25. qsmooth K0.75 identical in 65 of 84 cells (was 63).
26. ev (1, 0.0) evidenced count rises 1.8-35.7% (was 2-33%).
27. btransform K0.5: 17 identical-metric cells, 16 with 0 swaps (digits has 30 swaps on seed 30).
28. qsmooth s ≥ 1 changes 0.8-1.5% of the kept set (was 1-1.5%).
29. Flag "every identical cell has 0 or 1-3 inert swaps" replaced: several "=" cells hide up to 35 swaps per seed; ev (1, 0.0) loss
    is 9 better / 3 worse, so "mostly identical" overstated the absence of effect.
30. Flag 1: zero = inf does not reproduce plan_full on iris, wine, yacht; qsmooth not identical on yacht (one swap at s 0.1, K0.2).
31. ev (5, 0.5): "driven by sparse evidence" replaced; worse cells are mostly well-evidenced regression tasks; at K0.2 the kept set is
    still mostly evidenced.
32. act 0.01: tabarena_363625 also changes (15.9 swaps, B W B); diabets and tabarena_363675 identical in every unit.
33. zero = inf on credit-g: driven by seed 32; loss better at all K; classification loss counts 8/8; "zero-reachability = mislabelled"
    marked untested.
34. Single-seed-driven means flagged (btransform iris and yacht, zero = inf credit-g, qsmooth s 5.0 student_performance).
35. Table 1b (evrule) "n/a" for the ev arms replaced: 3 of 81 ev (1, 0.0) classification units are fully identical to plan_full.
