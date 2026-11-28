# Handoff batch 2026-11-25 (first batch): results

Six studies of the first handoff batch, all complete on r760 under `runs/t1/`. This report reads only `runs/`; nothing was
trained, nothing under `runs/` was written.

| study | experiment(s) | config | datasets | seeds | recording host(s) |
|---|---|---|---|---|---|
| `handoff_revise_r760` | `t12_revise_synthetic_iv1` | `configs/t1/handoff_20261125/t12_revise_synthetic_iv1.yaml` | synthetic_reference (independent_v1) | 30-32 | r760 |
| `handoff_revise_legacy_r760` | `t12_revise_synthetic` | `configs/t1/t12_revise_synthetic.yaml` (unchanged legacy config) | synthetic_diag (legacy generator) | 0-2 | r760 |
| `handoff_alignrev_r760` | `alignment_revision`, `alignment_revision_matched` | `configs/t1/alignment_revision{,_matched}.yaml` (legacy, unchanged) | synthetic_diag (legacy generator) | 5-7 | r760 |
| `handoff_cfaudit_r760` | `cf_audit__regression` | `handoff_20261125/cf_audit_regression.yaml` | 11 regression tasks | 30-32 | r760 / xps8960 / q-lab, each task on the host of its plan_full exact_removal rows |
| `handoff_synremoval_r760` | `synthetic_removal__classification` | `handoff_20261125/synthetic_removal.yaml` | synthetic_diag (independent_v1) | 30-32 | xps8960 |
| `handoff_syncallcase_r760` | `sync_penalty_allcase__{classification,regression}` | `handoff_20261125/sync_penalty_allcase_{classification,regression}.yaml` | 10 cls + 11 reg tasks | 30-32 | each task on the host of its plan_full synchronization rows |

The legacy revise and alignment studies use the seeds written in their legacy configs (0-2 and 5-7), not 30-32; they were run
as reproductions of the legacy experiments at the current code and have no recorded rows to pair with.

## How the numbers were produced

- Project summaries (format unchanged), written on r760 by `tools/t1_summarize.py` from the run units:
  `neural-cbr/results/t1/<study>/<experiment>/summary.csv` for all eight experiments, plus
  `paired_vs_random_K0.5.csv` (cf audit), `paired_vs_removal_influence_K0.5.csv` (synthetic removal) and
  `paired_vs_alternating_rr_matched_K0.75.csv` (both sync experiments).
- Cross-study pairings (handoff rows against plan_full_r760 rows) and the C/H-versus-counterfactual statistics are not
  something the summary tools do. A read-only script loaded the condition units with `model.t1.records`/`load_run`
  (row records and `case_statistics`) into `/var/tmp/nnknn-hb/` on r760 and accel; the tables below come from that dump.
  The event-file host names were checked for every unit (they match the "host" column above and the plan_full rows).
- Every table cell is mean ± sample std over the 3 seeds (n = 3 everywhere; no missing unit, no NaN in any row).

**Paired comparison rule (all sections).** For a dataset (or dataset × K/budget) cell, d_s = condition − reference on seed s,
paired by seed. The cell counts as **better** when the mean of d favours the condition (metric direction below) and at least 2
of the 3 seeds favour it; **worse** symmetrically; **identical** when all three d_s are exactly 0; otherwise **tie**. With 3
seeds this is a consistency count, not a significance test; the 95% t-intervals and paired t-test p are in the
`paired_vs_*.csv` files.

Metric directions: accuracy, precision@budget, recall@budget, success rate: higher is better. RMSE, loss (MSE on the
standardised target for regression, NLL for classification), ECE, NLL, `M2_minus_MC_loss`, `kept_fraction_corrupted`: lower is better.

---

## 1. `handoff_revise_r760` (t12 revise harness on the plan_full revision_stress data)

Rows: method × review budget b (number of flagged cases shown to the simulated oracle reviewer). Columns: precision and recall of
the flags against the known corrupted cases; test accuracy of M0 (trained core, before correction), M1 (after the relabels, no
retraining), M2 (after relabel + 30 retraining epochs at lr × 0.1), MC (matched control: the same 30 epochs without the
correction); `M2_minus_MC_loss` = test loss of M2 minus MC (negative = the correction helped beyond the extra training).

| method | b | precision | recall | M0 acc | M1 acc | M2 acc | MC acc | M2−MC loss |
|---|---|---|---|---|---|---|---|---|
| random | 10 | 0.100 ± 0.173 | 0.016 ± 0.028 | 0.945 ± 0.036 | 0.945 ± 0.036 | 0.945 ± 0.037 | 0.945 ± 0.037 | 0.000 ± 0.000 |
| provenance_only | 10 | 0.300 ± 0.100 | 0.048 ± 0.016 | 0.945 ± 0.036 | 0.945 ± 0.036 | 0.947 ± 0.037 | 0.945 ± 0.037 | 0.018 ± 0.037 |
| bias_only | 10 | 0.100 ± 0.100 | 0.016 ± 0.016 | 0.945 ± 0.036 | 0.945 ± 0.036 | 0.947 ± 0.037 | 0.945 ± 0.037 | 0.017 ± 0.036 |
| oracle | 10 | 1.000 ± 0.000 | 0.160 ± 0.003 | 0.945 ± 0.036 | 0.946 ± 0.035 | 0.950 ± 0.043 | 0.945 ± 0.037 | −0.033 ± 0.032 |
| random | 25 | 0.160 ± 0.106 | 0.064 ± 0.043 | 0.945 ± 0.036 | 0.945 ± 0.036 | 0.949 ± 0.032 | 0.945 ± 0.037 | −0.008 ± 0.007 |
| provenance_only | 25 | 0.333 ± 0.061 | 0.133 ± 0.022 | 0.945 ± 0.036 | 0.945 ± 0.036 | 0.947 ± 0.041 | 0.945 ± 0.037 | −0.029 ± 0.038 |
| bias_only | 25 | 0.053 ± 0.061 | 0.021 ± 0.025 | 0.945 ± 0.036 | 0.945 ± 0.036 | 0.947 ± 0.037 | 0.945 ± 0.037 | 0.017 ± 0.036 |
| oracle | 25 | 1.000 ± 0.000 | 0.399 ± 0.007 | 0.945 ± 0.036 | 0.955 ± 0.029 | 0.959 ± 0.041 | 0.945 ± 0.037 | 0.037 ± 0.074 |
| random | 50 | 0.160 ± 0.060 | 0.128 ± 0.050 | 0.945 ± 0.036 | 0.946 ± 0.035 | 0.954 ± 0.035 | 0.945 ± 0.037 | −0.061 ± 0.037 |
| provenance_only | 50 | 0.273 ± 0.023 | 0.218 ± 0.021 | 0.945 ± 0.036 | 0.945 ± 0.036 | 0.951 ± 0.046 | 0.945 ± 0.037 | −0.057 ± 0.031 |
| bias_only | 50 | 0.033 ± 0.031 | 0.027 ± 0.025 | 0.945 ± 0.036 | 0.945 ± 0.036 | 0.947 ± 0.037 | 0.945 ± 0.037 | 0.017 ± 0.036 |
| oracle | 50 | 1.000 ± 0.000 | 0.798 ± 0.015 | 0.945 ± 0.036 | 0.955 ± 0.029 | 0.968 ± 0.031 | 0.945 ± 0.037 | −0.141 ± 0.102 |
| random | 75 | 0.138 ± 0.050 | 0.166 ± 0.063 | 0.945 ± 0.036 | 0.946 ± 0.035 | 0.954 ± 0.035 | 0.945 ± 0.037 | −0.040 ± 0.033 |
| provenance_only | 75 | 0.249 ± 0.028 | 0.298 ± 0.028 | 0.945 ± 0.036 | 0.946 ± 0.035 | 0.955 ± 0.043 | 0.945 ± 0.037 | −0.088 ± 0.058 |
| bias_only | 75 | 0.036 ± 0.020 | 0.042 ± 0.024 | 0.945 ± 0.036 | 0.946 ± 0.035 | 0.947 ± 0.039 | 0.945 ± 0.037 | −0.003 ± 0.062 |
| oracle | 75 | 0.836 ± 0.015 | 1.000 ± 0.000 | 0.945 ± 0.036 | 0.955 ± 0.029 | 0.967 ± 0.031 | 0.945 ± 0.037 | −0.105 ± 0.053 |

Paired against `random` at the same budget (4 budgets each):

| method | precision@b (better/tie/worse) | M2 accuracy (better/tie/worse) |
|---|---|---|
| oracle | 4 / 0 / 0 | 4 / 0 / 0 (+0.005 to +0.014) |
| provenance_only | 4 / 0 / 0 (+0.11 to +0.20) | 2 / 2 / 0 (−0.003 to +0.003) |
| bias_only | 0 / 1 / 3 (−0.10 to −0.13 at b ≥ 25) | 1 / 1 / 2 (−0.006 at b 50, 75) |

**Pairing with plan_full `revision_stress__classification` / synthetic_reference** (same host, same seeds). The `_core` is the same:
M0 accuracy per seed is identical to all digits (s30 0.91538, s31 0.98462, s32 0.93462), and at the shared budget b = 10 the flags
are identical (same precision, recall and number of corrections per seed for every method). The only difference is the retraining
length (handoff 30 epochs, revision_stress 5 epochs). Handoff (30 ep) minus plan_full (5 ep) at b = 10:

| method | M2 accuracy | M2 test loss (lower better) | MC accuracy | M2−MC loss |
|---|---|---|---|---|
| oracle | −0.0026 (worse, 0/2) | −0.0056 (better, 3/0) | −0.0038 (worse) | +0.017 (tie) |
| provenance_only | −0.0013 (tie) | −0.0243 (better, 3/0) | −0.0038 (worse) | −0.001 (better) |
| bias_only | −0.0026 (tie) | −0.0029 (better) | −0.0038 (worse) | +0.020 (worse) |
| random | −0.0013 (tie) | −0.0464 (better, 3/0) | −0.0038 (worse) | −0.024 (tie) |

Longer retraining lowers test loss for every method but costs about one test case (1/260) of accuracy, and it does so for the
no-correction control as well, so it is a training-length effect, not a correction effect.

Reading: with an oracle reviewer, correction plus retraining helps and grows with budget (M2 − M0 +0.005 → +0.023; M2 − MC loss
down to −0.14 at b 50). provenance_only flags corrupted cases at about twice the random rate (precision 0.25-0.33 vs 0.14-0.16),
but on this nearly clean task (M0 0.945) the accuracy gain over random is within one test case. bias_only flags worse than random.

## 2. `handoff_revise_legacy_r760` (legacy t12 config, legacy synthetic_diag, seeds 0-2)

| method | b | precision | recall | M0 acc | M1 acc | M2 acc | M2 loss |
|---|---|---|---|---|---|---|---|
| random | 10 | 0.267 ± 0.208 | 0.049 ± 0.039 | 0.918 ± 0.066 | 0.923 ± 0.062 | 0.913 ± 0.049 | 0.550 ± 0.429 |
| provenance_only | 10 | 0.300 ± 0.100 | 0.056 ± 0.019 | 0.918 ± 0.066 | 0.919 ± 0.068 | 0.910 ± 0.052 | 0.577 ± 0.433 |
| bias_only | 10 | 0.033 ± 0.058 | 0.006 ± 0.011 | 0.918 ± 0.066 | 0.918 ± 0.066 | 0.906 ± 0.051 | 0.563 ± 0.396 |
| oracle | 10 | 1.000 ± 0.000 | 0.185 ± 0.000 | 0.918 ± 0.066 | 0.923 ± 0.062 | 0.919 ± 0.047 | 0.521 ± 0.388 |
| random | 25 | 0.200 ± 0.069 | 0.093 ± 0.032 | 0.918 ± 0.066 | 0.926 ± 0.059 | 0.915 ± 0.045 | 0.484 ± 0.363 |
| provenance_only | 25 | 0.253 ± 0.061 | 0.117 ± 0.028 | 0.918 ± 0.066 | 0.919 ± 0.068 | 0.910 ± 0.052 | 0.551 ± 0.397 |
| bias_only | 25 | 0.040 ± 0.040 | 0.019 ± 0.019 | 0.918 ± 0.066 | 0.918 ± 0.066 | 0.906 ± 0.051 | 0.564 ± 0.396 |
| oracle | 25 | 1.000 ± 0.000 | 0.463 ± 0.000 | 0.918 ± 0.066 | 0.927 ± 0.061 | 0.937 ± 0.052 | 0.511 ± 0.413 |
| random | 50 | 0.200 ± 0.092 | 0.185 ± 0.085 | 0.918 ± 0.066 | 0.926 ± 0.059 | 0.919 ± 0.042 | 0.500 ± 0.355 |
| provenance_only | 50 | 0.327 ± 0.050 | 0.302 ± 0.047 | 0.918 ± 0.066 | 0.919 ± 0.068 | 0.915 ± 0.048 | 0.528 ± 0.380 |
| bias_only | 50 | 0.047 ± 0.023 | 0.043 ± 0.021 | 0.918 ± 0.066 | 0.919 ± 0.065 | 0.913 ± 0.047 | 0.535 ± 0.391 |
| oracle | 50 | 1.000 ± 0.000 | 0.926 ± 0.000 | 0.918 ± 0.066 | 0.932 ± 0.056 | 0.936 ± 0.047 | 0.479 ± 0.390 |
| random | 75 | 0.147 ± 0.061 | 0.204 ± 0.085 | 0.918 ± 0.066 | 0.926 ± 0.059 | 0.918 ± 0.043 | 0.479 ± 0.324 |
| provenance_only | 75 | 0.276 ± 0.028 | 0.383 ± 0.039 | 0.918 ± 0.066 | 0.919 ± 0.068 | 0.919 ± 0.051 | 0.632 ± 0.423 |
| bias_only | 75 | 0.102 ± 0.077 | 0.142 ± 0.107 | 0.918 ± 0.066 | 0.919 ± 0.065 | 0.910 ± 0.048 | 0.557 ± 0.391 |
| oracle | 75 | 0.720 ± 0.000 | 1.000 ± 0.000 | 0.918 ± 0.066 | 0.932 ± 0.056 | 0.937 ± 0.045 | 0.501 ± 0.414 |

vs random, M2 accuracy (4 budgets): oracle 4/0/0; provenance_only 0/2/2; bias_only 0/2/2. Precision: oracle 4/0/0,
provenance_only 4/0/0, bias_only 0/0/4. No matched control in this legacy config.

Reading: here the 30 retraining epochs hurt the non-oracle arms: M2 − M0 is negative for bias_only (−0.012 at b 10/25) and
provenance_only (−0.008 at b 10/25) while M1 − M0 is ≥ 0. With the legacy core (`_base`, shorter training) retraining after a few
relabels destabilises the model; only the oracle arm gains (+0.018 to +0.019 at b ≥ 25).

## 3. `handoff_alignrev_r760` (legacy alignment_revision, plain and matched, seeds 5-7)

Budgets 10 and 30, 5 retraining epochs. M2 accuracy (mean ± std):

| method | b10 | b30 | M2−MC loss b10 (matched) | M2−MC loss b30 (matched) |
|---|---|---|---|---|
| random | 0.968 ± 0.039 | 0.971 ± 0.035 | −0.004 ± 0.006 | −0.005 ± 0.006 |
| provenance_only | 0.968 ± 0.036 | 0.968 ± 0.036 | −0.006 ± 0.003 | −0.007 ± 0.003 |
| bias_only | 0.967 ± 0.039 | 0.967 ± 0.039 | −0.000 ± 0.000 | −0.000 ± 0.000 |
| oracle | 0.967 ± 0.039 | 0.969 ± 0.037 | −0.004 ± 0.004 | −0.008 ± 0.005 |

(M0 0.969 ± 0.037. Precision: oracle 1.00, provenance_only 0.30 / 0.27, random 0.10 / 0.12, bias_only 0.07 / 0.06.)

- `alignment_revision` and `alignment_revision_matched` rows are **identical in every shared field** (24 of 24 rows); the
  matched config only adds the MC fields. This is the intended behaviour (the control runs under a forked RNG) and confirms it.
- vs random, M2 accuracy: oracle 0/2/0 (tie), provenance_only 0/1/1, bias_only 0/1/1. With 5 epochs and M0 at 0.969 nothing
  separates from random on accuracy; the matched loss differences are all ≤ 0.008.

## 4. `handoff_cfaudit_r760` (regression counterfactual audit)

**Pairing with plan_full `exact_removal_diagnostics__regression`.** All 264 condition rows (11 tasks × 8 conditions × 3 seeds)
are bit-identical to the plan_full rows in every numeric field, and the per-case C/H statistics are identical too. This is
expected (`statistics.counterfactual` only adds cf_pos/cf_neg next to C/H; scoring does not read them) and confirms that the same
host reproduces the plan_full units exactly. The performance table therefore repeats plan_full:

test RMSE after finetune (lower is better), standardised target:

| dataset | random K0.5 | bias_normalized K0.5 | removal_influence K0.5 | removal_cached K0.5 | random K0.75 | bias_normalized K0.75 | removal_influence K0.75 | removal_cached K0.75 |
|---|---|---|---|---|---|---|---|---|
| abalone | 0.853 ± 0.150 | 0.844 ± 0.140 | 0.838 ± 0.139 | 0.839 ± 0.138 | 0.837 ± 0.143 | 0.844 ± 0.143 | 0.842 ± 0.139 | 0.839 ± 0.139 |
| airfoil | 0.630 ± 0.039 | 0.601 ± 0.012 | 0.590 ± 0.010 | 0.590 ± 0.011 | 0.593 ± 0.014 | 0.599 ± 0.014 | 0.600 ± 0.015 | 0.600 ± 0.015 |
| body_fat | 0.622 ± 0.100 | 0.601 ± 0.080 | 0.608 ± 0.057 | 0.607 ± 0.052 | 0.599 ± 0.052 | 0.612 ± 0.057 | 0.606 ± 0.052 | 0.606 ± 0.038 |
| car | 0.889 ± 0.156 | 1.030 ± 0.352 | 0.924 ± 0.304 | 0.924 ± 0.304 | 0.884 ± 0.285 | 1.040 ± 0.377 | 0.940 ± 0.265 | 0.940 ± 0.265 |
| diabets | 0.744 ± 0.016 | 0.729 ± 0.039 | 0.722 ± 0.013 | 0.723 ± 0.012 | 0.735 ± 0.028 | 0.735 ± 0.009 | 0.728 ± 0.015 | 0.726 ± 0.016 |
| energy_efficiency | 0.336 ± 0.016 | 0.307 ± 0.045 | 0.304 ± 0.040 | 0.305 ± 0.037 | 0.299 ± 0.019 | 0.291 ± 0.045 | 0.295 ± 0.038 | 0.292 ± 0.041 |
| student_performance | 0.487 ± 0.033 | 0.509 ± 0.088 | 0.456 ± 0.076 | 0.457 ± 0.077 | 0.466 ± 0.052 | 0.471 ± 0.062 | 0.464 ± 0.070 | 0.462 ± 0.069 |
| tabarena_363625 | 0.728 ± 0.016 | 0.669 ± 0.043 | 0.696 ± 0.054 | 0.700 ± 0.048 | 0.710 ± 0.046 | 0.700 ± 0.052 | 0.701 ± 0.049 | 0.701 ± 0.050 |
| tabarena_363675 | 0.603 ± 0.068 | 0.561 ± 0.081 | 0.546 ± 0.093 | 0.543 ± 0.089 | 0.537 ± 0.073 | 0.535 ± 0.083 | 0.530 ± 0.095 | 0.528 ± 0.095 |
| wine | 0.941 ± 0.109 | 0.965 ± 0.108 | 0.947 ± 0.100 | 0.954 ± 0.099 | 0.931 ± 0.103 | 0.949 ± 0.107 | 0.954 ± 0.108 | 0.953 ± 0.108 |
| yacht | 0.622 ± 0.375 | 0.667 ± 0.343 | 0.634 ± 0.352 | 0.634 ± 0.352 | 0.659 ± 0.355 | 0.698 ± 0.377 | 0.641 ± 0.352 | 0.641 ± 0.352 |

vs random at the same K, over 11 tasks (better/tie/worse): K 0.5: bias_normalized 7/0/4, removal_influence 7/1/3,
removal_cached 7/1/3. K 0.75: bias_normalized 2/3/6, removal_influence 5/1/5, removal_cached 6/0/5.

**C/H versus counterfactual agreement** (the new information). Per (task, seed), over the 60 cases of the core's train-LOO
audit (identical in every condition unit of that task and seed, checked): C/H as recorded (tolerance 0.5), Q = (C+1)/(C+H+2),
B = within-cohort percentile of the current bias, and the counterfactual net influence cf = cf_pos − cf_neg = mean final loss
without the case minus with it (positive = the case helps). "Evidenced" = R ≥ 2 and A ≥ 0.1.

| column | meaning |
|---|---|
| ev | evidenced cases (of 60) |
| ρ(Q,cf) | Spearman of Q and cf over evidenced cases |
| ρ(B,cf) | Spearman of B and cf over evidenced cases |
| ρ(C−H,cf) | Spearman of C−H and cf over all 60 cases |
| ρ(A,\|cf\|) | Spearman of activation mass and the size of the influence |
| sign agr | share of evidenced cases (Q ≠ 0.5, cf ≠ 0) where Q > 0.5 ⇔ cf > 0 |
| chance | sign agreement expected from the two marginals alone |
| share Q>0.5 | of evidenced cases |
| low-half overlap | share of the 30 lowest-Q cases that are also among the 30 lowest-cf cases (chance 0.5) |
| cf<0 | share of cases whose removal lowers the loss |

| dataset | ev | ρ(Q,cf) | ρ(B,cf) | ρ(C−H,cf) | ρ(A,\|cf\|) | sign agr | chance | share Q>0.5 | low-half overlap | cf<0 |
|---|---|---|---|---|---|---|---|---|---|---|
| abalone | 51.7 ± 4.0 | −0.01 ± 0.21 | 0.54 ± 0.04 | 0.28 ± 0.06 | 0.70 ± 0.10 | 0.79 ± 0.09 | 0.81 ± 0.09 | 0.89 | 0.58 ± 0.07 | 0.13 |
| airfoil | 54.0 ± 2.0 | 0.56 ± 0.23 | 0.45 ± 0.09 | 0.66 ± 0.20 | 0.80 ± 0.09 | 0.88 ± 0.04 | 0.88 ± 0.04 | 0.98 | 0.73 ± 0.07 | 0.11 |
| body_fat | 47.7 ± 1.2 | 0.39 ± 0.19 | 0.65 ± 0.13 | 0.53 ± 0.16 | 0.86 ± 0.02 | 0.77 ± 0.04 | 0.78 ± 0.04 | 0.83 | 0.71 ± 0.12 | 0.13 |
| car | 46.3 ± 2.1 | −0.17 ± 0.17 | 0.52 ± 0.05 | −0.02 ± 0.15 | 0.24 ± 0.26 | 0.62 ± 0.03 | 0.62 ± 0.03 | 0.86 | 0.58 ± 0.11 | 0.31 |
| diabets | 44.7 ± 1.2 | 0.03 ± 0.39 | 0.38 ± 0.16 | 0.10 ± 0.32 | 0.78 ± 0.18 | 0.57 ± 0.09 | 0.53 ± 0.05 | 0.52 | 0.60 ± 0.12 | 0.21 |
| energy_efficiency | 58.7 ± 2.3 | 0.24 ± 0.26 | 0.62 ± 0.07 | 0.26 ± 0.28 | 0.31 ± 0.28 | 0.63 ± 0.05 | 0.62 ± 0.07 | 0.96 | 0.62 ± 0.18 | 0.37 |
| student_performance | 48.7 ± 8.1 | 0.35 ± 0.44 | 0.35 ± 0.01 | 0.36 ± 0.41 | 0.72 ± 0.27 | 0.71 ± 0.15 | 0.70 ± 0.16 | 0.89 | 0.64 ± 0.16 | 0.27 |
| tabarena_363625 | 50.7 ± 4.5 | 0.34 ± 0.15 | 0.46 ± 0.18 | 0.50 ± 0.13 | 0.78 ± 0.05 | 0.83 ± 0.06 | 0.83 ± 0.07 | 0.95 | 0.67 ± 0.07 | 0.16 |
| tabarena_363675 | 50.0 ± 6.1 | 0.13 ± 0.27 | 0.44 ± 0.19 | 0.40 ± 0.35 | 0.71 ± 0.21 | 0.82 ± 0.09 | 0.83 ± 0.07 | 0.95 | 0.63 ± 0.10 | 0.17 |
| wine | 54.0 ± 6.1 | −0.09 ± 0.10 | 0.65 ± 0.10 | −0.05 ± 0.07 | 0.56 ± 0.20 | 0.50 ± 0.06 | 0.49 ± 0.04 | 0.47 | 0.47 ± 0.03 | 0.30 |
| yacht | 34.3 ± 5.5 | −0.15 ± 0.18 | 0.32 ± 0.12 | 0.16 ± 0.18 | 0.71 ± 0.09 | 0.69 ± 0.10 | 0.68 ± 0.08 | 0.91 | 0.63 ± 0.09 | 0.18 |
| **all 33 (task, seed)** | 49.2 ± 7.2 | **0.15 ± 0.32** | **0.49 ± 0.15** | 0.29 ± 0.30 | 0.65 ± 0.25 | **0.71 ± 0.13** | **0.71 ± 0.14** | 0.84 | 0.62 ± 0.11 | 0.21 |

ρ(Q,cf) > 0 in 21 of 33 (task, seed) units; ρ(B,cf) > 0 in 33 of 33.

Reading:
- Outcome credit Q agrees only weakly with counterfactual helpfulness (mean Spearman 0.15, negative on abalone, car, wine and
  yacht). Its sign agreement with cf (0.71) equals what the base rates alone give (0.71): knowing Q > 0.5 says nothing more about
  whether a case helps than the fact that most cases have Q > 0.5 and cf > 0.
- The bias percentile B tracks cf better than Q on 9 of 11 tasks by mean (not on airfoil; equal on student_performance),
  mean 0.49, positive in every unit.
- The size of the counterfactual influence follows exposure (ρ(A,|cf|) 0.65), the same exposure confound the progress review
  noted for Q.
- On car and wine, where Q and cf disagree most, neither bias_normalized nor the removal policies beat random at either K.
- Caveat: C/H for regression depend on the success tolerance (0.5 on the standardised target); this audit uses that single
  value.

## 5. `handoff_synremoval_r760` (removal policies on synthetic_diag, independent_v1)

Rows: retention policy; columns: capacity K (share of the 432 cases kept). The removal rows are new; the other rows are the
plan_full `retention__classification` synthetic_diag rows (same host xps8960, same seeds, same core: the full-memory accuracy
per seed is identical in both studies, 0.91538 / 0.98462 / 0.93462).

test accuracy after the 30-epoch finetune (higher is better):

| policy | K 0.2 | K 0.5 | K 0.75 |
|---|---|---|---|
| **removal_influence** (new) | 0.949 ± 0.041 | 0.944 ± 0.049 | 0.945 ± 0.045 |
| **removal_cached** (new) | 0.949 ± 0.041 | 0.945 ± 0.048 | 0.946 ± 0.044 |
| full_memory | 0.947 ± 0.040 | 0.947 ± 0.040 | 0.947 ± 0.040 |
| random | 0.982 ± 0.017 | 0.954 ± 0.046 | 0.947 ± 0.058 |
| stratified | 0.978 ± 0.023 | 0.968 ± 0.031 | 0.950 ± 0.052 |
| kcenter | 0.886 ± 0.100 | 0.933 ± 0.058 | 0.967 ± 0.029 |
| coverage_reachability | 0.960 ± 0.035 | 0.944 ± 0.037 | 0.951 ± 0.040 |
| provenance_only | 0.949 ± 0.045 | 0.951 ± 0.041 | 0.942 ± 0.039 |
| bias_current | 0.968 ± 0.029 | 0.963 ± 0.035 | 0.937 ± 0.068 |
| bias_normalized | 0.967 ± 0.033 | 0.932 ± 0.087 | 0.958 ± 0.039 |

share of the corrupted (mislabelled) cases kept (lower is better):

| policy | K 0.2 | K 0.5 | K 0.75 |
|---|---|---|---|
| **removal_influence** | 0.345 ± 0.039 | 0.425 ± 0.041 | 0.676 ± 0.035 |
| **removal_cached** | 0.340 ± 0.030 | 0.425 ± 0.041 | 0.681 ± 0.027 |
| random | 0.175 ± 0.039 | 0.457 ± 0.061 | 0.733 ± 0.056 |
| stratified | 0.186 ± 0.045 | 0.457 ± 0.065 | 0.755 ± 0.053 |
| kcenter | 0.250 ± 0.049 | 0.591 ± 0.050 | 0.798 ± 0.049 |
| coverage_reachability | 0.021 ± 0.036 | 0.201 ± 0.076 | 0.532 ± 0.043 |
| provenance_only | 0.032 ± 0.028 | 0.292 ± 0.019 | 0.691 ± 0.054 |
| bias_current | 0.238 ± 0.092 | 0.398 ± 0.077 | 0.910 ± 0.044 |
| bias_normalized | 0.227 ± 0.104 | 0.420 ± 0.058 | 0.846 ± 0.038 |

Paired, removal_influence minus reference (accuracy / corrupted kept) at K 0.2, 0.5, 0.75:

| reference | accuracy | corrupted kept |
|---|---|---|
| random | worse (−0.033) / tie (−0.010) / worse (−0.003) | worse (+0.17) / better (−0.03) / better (−0.06) |
| provenance_only | tie / worse (−0.008) / better (+0.003) | worse (+0.31) / worse (+0.13) / tie |
| bias_normalized | worse (−0.018) / tie / worse (−0.013) | worse / tie / better (−0.17) |
| stratified | worse / worse / worse | worse / better / better |
| coverage_reachability | worse / tie / worse | worse / worse / worse |
| full_memory | better (+0.001) / worse / worse | better (−0.66) / better / better |

removal_cached gives the same verdicts (it differs from removal_influence only on seed 30: identical test loss on seeds 31
and 32 at every K).

Reading: on this task the removal policies give no accuracy gain over full memory or random, and they remove corrupted cases
far less selectively than the outcome-evidence policies (provenance_only and coverage_reachability keep 3% / 2% of the corrupted
cases at K 0.2; removal keeps 34%). They do keep rare cases better than random (kept_fraction_rare 0.89 vs 0.53 at K 0.5).
Random and stratified at K 0.2 have the best accuracy, as in plan_full.

## 6. `handoff_syncallcase_r760` (all-case twin of the masked plan_full synchronization conditions)

Five conditions, all `alternating_rr` schedules: `alternating_rr` (trained-bias-region allowance), `rr_no_L_small` (no
correction-size penalty), `rr_zero_radius` (squared L2 from zero), `rr_initial_bias_radius`, `rr_exact_kth_radius`.
The handoff rows use `core.pre_topk_mask: false` (all-case); plan_full rows are masked. Same tuned cores, same host per task.

Main metric: classification test accuracy after the correction (`test_accuracy_post`, higher is better); regression test RMSE
after the correction (`test_rmse_post`, lower is better). Cells: all-case / masked.

| classification | alternating_rr | rr_no_L_small | rr_zero_radius | rr_initial_bias_radius | rr_exact_kth_radius |
|---|---|---|---|---|---|
| balance | 0.968±0.014 / 0.936±0.016 | 0.968±0.014 / 0.939±0.012 | 0.968±0.014 / 0.939±0.012 | 0.968±0.014 / 0.936±0.016 | 0.968±0.014 / 0.936±0.016 |
| breast_cancer | 0.971±0.010 / 0.965±0.015 | 0.971±0.010 / 0.965±0.018 | 0.971±0.010 / 0.962±0.013 | 0.971±0.010 / 0.965±0.009 | 0.971±0.010 / 0.962±0.020 |
| digits | 0.981±0.007 / 0.978±0.007 | 0.981±0.005 / 0.977±0.008 | 0.981±0.007 / 0.978±0.005 | 0.981±0.007 / 0.978±0.005 | 0.981±0.007 / 0.977±0.006 |
| iris | 0.922±0.038 / 0.944±0.069 | 0.922±0.038 / 0.933±0.067 | 0.922±0.038 / 0.944±0.069 | 0.922±0.038 / 0.933±0.067 | 0.922±0.038 / 0.944±0.069 |
| synthetic_diag | 0.977±0.023 / 0.947±0.044 | 0.972±0.031 / 0.945±0.047 | 0.982±0.017 / 0.949±0.043 | 0.978±0.021 / 0.938±0.058 | 0.978±0.021 / 0.947±0.040 |
| tabarena_363614 | 0.986±0.011 / 0.986±0.008 | 0.987±0.009 / 0.983±0.015 | 0.986±0.011 / 0.986±0.008 | 0.986±0.011 / 0.986±0.008 | 0.986±0.011 / 0.986±0.008 |
| tabarena_363626 | 0.739±0.009 / 0.704±0.017 | 0.742±0.011 / 0.707±0.020 | 0.736±0.011 / 0.706±0.019 | 0.739±0.006 / 0.704±0.017 | 0.741±0.009 / 0.705±0.017 |
| wine | 0.972±0.028 / 0.898±0.016 | 0.972±0.028 / 0.935±0.016 | 0.972±0.028 / 0.898±0.016 | 0.972±0.028 / 0.917±0.000 | 0.972±0.028 / 0.898±0.016 |
| zebra | 0.773±0.198 / 0.924±0.131 | 0.773±0.198 / 0.924±0.131 | 0.803±0.184 / 0.924±0.131 | 0.773±0.198 / 0.924±0.131 | 0.773±0.198 / 0.924±0.131 |
| zebra_special | 0.508±0.066 / 0.659±0.104 | 0.508±0.066 / 0.667±0.107 | 0.523±0.045 / 0.659±0.104 | 0.508±0.066 / 0.659±0.104 | 0.508±0.066 / 0.659±0.104 |

| regression (RMSE) | alternating_rr | rr_no_L_small | rr_zero_radius | rr_initial_bias_radius | rr_exact_kth_radius |
|---|---|---|---|---|---|
| abalone | 0.907±0.094 / 0.776±0.092 | 0.907±0.094 / 0.777±0.091 | 0.898±0.112 / 0.776±0.088 | 0.907±0.094 / 0.777±0.091 | 0.907±0.094 / 0.776±0.092 |
| airfoil | 0.239±0.054 / 0.236±0.035 | 0.239±0.054 / 0.235±0.035 | 0.237±0.055 / 0.235±0.035 | 0.239±0.054 / 0.235±0.035 | 0.239±0.054 / 0.235±0.035 |
| body_fat | 0.714±0.160 / 0.606±0.102 | 0.714±0.159 / 0.606±0.102 | 0.787±0.296 / 0.604±0.096 | 0.713±0.160 / 0.606±0.102 | 0.714±0.160 / 0.606±0.102 |
| car | 0.410±0.024 / 0.364±0.063 | 0.445±0.044 / 0.358±0.072 | 0.406±0.022 / 0.360±0.075 | 0.419±0.015 / 0.357±0.074 | 0.436±0.053 / 0.357±0.073 |
| diabets | 0.695±0.052 / 0.719±0.021 | 0.695±0.052 / 0.719±0.021 | 0.694±0.052 / 0.727±0.007 | 0.695±0.052 / 0.719±0.021 | 0.695±0.052 / 0.719±0.021 |
| energy_efficiency | 0.052±0.006 / 0.077±0.012 | 0.052±0.006 / 0.076±0.011 | 0.052±0.006 / 0.077±0.013 | 0.052±0.006 / 0.076±0.011 | 0.052±0.006 / 0.074±0.009 |
| student_performance | 0.313±0.042 / 0.359±0.046 | 0.313±0.042 / 0.358±0.046 | 0.312±0.042 / 0.357±0.044 | 0.313±0.042 / 0.360±0.048 | 0.313±0.042 / 0.360±0.048 |
| tabarena_363625 | 0.341±0.031 / 0.369±0.025 | 0.341±0.031 / 0.369±0.025 | 0.341±0.032 / 0.369±0.024 | 0.341±0.031 / 0.369±0.025 | 0.341±0.031 / 0.369±0.025 |
| tabarena_363675 | 0.375±0.046 / 0.392±0.044 | 0.375±0.046 / 0.394±0.042 | 0.375±0.047 / 0.393±0.045 | 0.375±0.046 / 0.393±0.042 | 0.375±0.046 / 0.393±0.041 |
| wine | 0.751±0.042 / 0.752±0.047 | 0.751±0.042 / 0.752±0.047 | 0.751±0.041 / 0.750±0.046 | 0.751±0.041 / 0.752±0.047 | 0.751±0.042 / 0.752±0.047 |
| yacht | 0.131±0.040 / 0.097±0.021 | 0.131±0.040 / 0.097±0.021 | 0.131±0.040 / 0.095±0.031 | 0.131±0.040 / 0.097±0.021 | 0.132±0.042 / 0.097±0.021 |

**All-case minus masked, same condition** (better/tie/worse over tasks):

| condition | classification (10) | regression (11) |
|---|---|---|
| alternating_rr | 5 / 2 / 3 | 5 / 2 / 4 |
| rr_no_L_small | 4 / 3 / 3 | 5 / 2 / 4 |
| rr_zero_radius | 5 / 2 / 3 | 5 / 2 / 4 |
| rr_initial_bias_radius | 6 / 1 / 3 | 5 / 2 / 4 |
| rr_exact_kth_radius | 5 / 2 / 3 | 5 / 2 / 4 |

The verdicts are nearly the same task by task for every condition (exceptions: rr_no_L_small breast_cancer tie,
rr_initial_bias_radius digits better, rr_zero_radius body_fat tie and regression wine worse). For alternating_rr all-case is better on balance, breast_cancer, synthetic_diag,
credit-g (363626), wine (cls) and diabets, energy_efficiency, student_performance, concrete (363625), insurance (363675); worse
on iris, zebra, zebra_special and abalone, body_fat, car, yacht. The masking choice of the core decides the comparison, not
the correction-size penalty. The retrieval-only prediction of the synchronised model (`test_*_pre`) already shows the same
pattern (all-case vs masked: classification 5/2/3, regression 7/2/2).

**Within each study, condition minus `alternating_rr`** (better/tie/worse/identical):

| condition | all-case cls | masked cls | all-case reg | masked reg |
|---|---|---|---|---|
| rr_no_L_small | 1/3/0/6 | 3/6/0/1 | 2/3/2/4 | 2/6/1/2 |
| rr_zero_radius | 1/2/1/6 | 1/4/0/5 | 8/1/2/0 | 6/3/2/0 |
| rr_initial_bias_radius | 0/2/0/8 | 1/4/0/5 | 2/7/1/1 | 1/9/0/1 |
| rr_exact_kth_radius | 1/1/0/8 | 0/4/0/6 | 1/5/3/2 | 2/7/0/2 |

Reading: the correction-size penalty variants barely matter. The one consistent effect is `rr_zero_radius` on regression (better
than alternating_rr on 8 of 11 all-case tasks and 6 of 11 masked tasks, by ≤ 0.01 RMSE on most), but it is also the condition with
the large all-case loss on body_fat (0.787 vs 0.714).

---

## Flags (surprising or suspicious)

1. **All-case sync: the trained-bias-region radius collapses.** In the all-case classification rows the free radius status is
   `near_zero` or `empty_region` in 19 of 30 `alternating_rr` rows (masked: 4 of 30), and `tau_task` = 0 in 88 of 150 rows
   (masked 40). With a zero allowance `alternating_rr`, `rr_initial_bias_radius` and `rr_zero_radius` become the same
   optimisation: 19 `rr_zero_radius` and 17 `rr_initial_bias_radius` all-case rows are identical to `alternating_rr` in every
   numeric field (masked: 4 and 2). So "no effect" of these conditions on all-case classification is a degenerate radius, not
   evidence that the penalty form does not matter. `rr_no_L_small` is identical to `alternating_rr` on 6 all-case
   classification tasks as well (the penalty is inactive when the allowance region is empty/zero).
2. **Sync rarely changes a classification prediction.** post accuracy equals pre accuracy in 86 of 150 all-case rows and 81 of
   150 masked rows; iris and (all-case) wine never change. Classification differences between the five conditions are therefore
   mostly one or two test cases.
3. **body_fat all-case sync hurts**: RMSE 0.557 (retrieval-only) → 0.714 after correction (`rr_zero_radius` 0.787), against
   +0.02 for masked. abalone shows the opposite: masked correction gains −0.12 RMSE, all-case only −0.005.
4. **zebra / zebra_special all-case models are much worse** than masked already in their retrieval-only prediction
   (accuracy 0.70 vs 0.92 and 0.49 vs 0.67 before the correction); the all-case-vs-masked comparison on these two tasks measures the masking of the core, not the sync penalty. Note that
   `test_*_pre` differs between conditions of one study (the sync also trains retrieval at lr × 0.1), so it is the
   retrieval-only prediction of each synchronised model, not the untouched core.
5. **cf audit rows equal plan_full exactly** (264/264, also C/H). Expected; it means the cf audit adds no new performance
   rows, only cf_pos/cf_neg.
6. **revise: identical M2 across budgets.** bias_only seed 30 gives the same M2 (accuracy and loss to 5 digits) at b 10, 25 and
   50 although the number of corrections is 1, 1 and 2; seed 31 likewise at b 10 (2 corrections) and b 25/50 (3 corrections).
   The extra relabelled cases change nothing in retraining; a likely reason is that they are never in any query's top-k under the
   masked core (no gradient reaches them). Worth one check before reading bias_only budget curves.
   Also `random` b 10 has `M2_minus_MC_loss` = 0 exactly on seeds 30 and 32: 0 corrections there, so M2 = MC by construction.
7. **synthetic_diag (independent_v1) and synthetic_reference give the same core**: the full-memory accuracies per seed on
   xps8960 (synthetic_diag, retention) equal the M0 accuracies on r760 (synthetic_reference, revise) to 9 digits
   (0.915384591 / 0.984615386 / 0.934615374). Under independent_v1 the two names appear to generate the same data, and the two
   hosts agree bit-for-bit on this core. Not an error, but the two names should not be counted as two datasets.
8. **removal_influence vs removal_cached**: identical test loss on seeds 31 and 32 at every K; they differ only on seed 30.
9. `full_memory` rows are the same at every K (expected: K does not apply).
10. Seeds: the two legacy studies use seeds 0-2 and 5-7 from their unchanged legacy configs, not 30-32.

## Conclusions

- **Revise (t12 on the plan_full data):** an oracle reviewer plus retraining helps and grows with budget (M2 accuracy +0.005 to
  +0.023 over M0, loss −0.14 vs the matched control at b 50). provenance_only flags corrupted cases at about twice the random rate
  at every budget (precision 4/0/0 vs random), but its accuracy gain over random stays within one test case (2/2/0). bias_only
  flags worse than random (precision 0/1/3). 30 retraining epochs instead of 5 lower the loss for every arm and for the control
  alike.
- **Legacy revise / alignment:** they reproduce the legacy picture: only the oracle arm gains; with the legacy core, 30-epoch
  retraining after a few relabels lowers accuracy for the non-oracle arms; the matched config changes only the MC fields.
- **CF audit:** outcome credit Q is a poor proxy for counterfactual helpfulness on regression (Spearman 0.15, sign agreement at
  chance level); the bias percentile B is better (0.49, positive in all 33 units); influence size follows exposure.
- **Synthetic removal:** removal_influence/removal_cached bring no accuracy gain over full memory or random on synthetic_diag and
  are much less selective against corrupted cases than provenance_only or coverage_reachability.
- **All-case sync twin:** the five penalty variants are nearly indistinguishable; whether the core is masked or all-case decides
  the result task by task. In the all-case core the trained-bias-region allowance often collapses to zero, which makes three of
  the five conditions the same run on most classification tasks.

Data and scripts (scratch, not in git): dump `/var/tmp/nnknn-hb/dump/*.jsonl` (accel and r760), analysis scripts
`/var/tmp/nnknn-hb/ana/*.py` (accel), dumper `/var/tmp/nnknn-hb/dump.py` (r760).
