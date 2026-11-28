# Handoff 2026-11-26: revision_stress and synthetic new-policy results (2026-11-27)

This report summarizes the two parts of the 2026-11-26 handoff run that
`reports/2026-11-26-handoff-everything-results.md` listed as "not summarized yet". Both are exploratory, use our own
choices rather than PI decisions, and use 3 seeds (30-32). All numbers come from complete units under r760
`runs/t1`. The dumps were read-only and went to `/var/tmp/nnknn-haotangl/synsum1127/`. The tables were built on
accel. Nothing was trained, deleted or committed.

| part | studies (r760 `runs/t1`) | configs |
|---|---|---|
| revision_stress | `handoff_allcaseplan_r760` (all-case twin) vs `plan_full_r760` (masked), experiment `revision_stress__classification` | `configs/t1/handoff_20261126/p1-allcase/plan/revision_stress__classification__synthetic_*.yaml` |
| synnewpol | `handoff_synnewpol_r760` (7 new policies) vs `handoff_synret_r760` (8 recorded policies, same core) | `configs/t1/handoff_20261126/p2-retention/synthetic_newpol.yaml` |

The 7 synthetic factor sets (generator independent_v1) are reference, overlap, noise, redundant, uncovered_rare,
shift and combined. All are classification tasks.

## Tools and outputs

- `tools/t1_handoff1126_revstress.py` (new): `dump` runs on r760 and reads only `runs/`. `summarize` writes
  `results/t1/handoff_20261126/revstress/` (`tables.md`, `paired.csv`, `paired_per_task.csv`, `per_task.csv`,
  `per_arm_method_budget.csv`, `units.csv`, `completeness.csv`).
- `tools/t1_handoff1126_p2.py` (extended): a new `--set synthetic` option on `dump` and `summarize` selects
  `handoff_synnewpol_r760` against `handoff_synret_r760` and skips the latency study. The default `--set real`
  output is byte-identical to yesterday's `p2/` csv files. Only the title line of `tables.md` changed: it now names
  the set. Outputs are in `results/t1/handoff_20261126/synnewpol/`.
- Independent recompute: `raw_event_check.py` and `raw_event_check.txt` in both output folders. The script reads the
  raw event files with tensorboard's loader only and uses no project code.

Conventions: d = arm − reference, paired by task and seed. "±" is the sd over the 3 seeds of the per-seed mean over
tasks. W/T/L counts tasks, and a task is a tie when |3-seed mean d| ≤ 0.005. A task enters a comparison only when it
has all 3 seeds in both arms (no task was dropped).

## 1. revision_stress (p1 all-case twin vs masked plan_full)

What a row measures (`model/t1/revise.py`, `flagging_ablation`):
1. A flagging method ranks the training cases of the trained full-memory core (M0).
2. A simulated reviewer inspects the top b cases (b = 10 or 30) and relabels the ones that are truly corrupted.
3. Four models are scored on the test set:
   - M1: the core with the edit only.
   - M2: the edit plus 5 retraining epochs on the corrected labels.
   - MC: M0 plus the same 5 epochs without any edit (the matched control).

The flagging methods are provenance_only, bias_only, random and oracle. The all-case twin changes only
`pre_topk_mask` on the untuned `_base` core.

### Completeness

| arm | condition units | stage units (`_core`, `_revise`) |
|---|---|---|
| all-case (`handoff_allcaseplan_r760`) | 168 / 168 complete (7 sets × 8 conditions × 3 seeds) | 42 / 42 |
| masked (`plan_full_r760`) | 168 / 168 complete | 42 / 42 |

Two checks passed:
- M0 and MC test accuracy are identical across the 8 conditions of every task, seed and arm (spread 0).
- random and oracle select exactly the same cases in both arms (same precision and same corrections), so the two
  arms share data, corruption and the seeded order.

Each training set has 63 corrupted cases, except noise and combined (168) and redundant (85).

### The core: all-case vs masked M0

Test accuracy of M0 is 0.861 for all-case and 0.817 for masked: d = +0.0445 ± 0.0200, W/T/L 5/2/0. The two ties are
redundant and uncovered_rare. The largest gains are noise (+0.11), shift (+0.07), overlap (+0.06) and combined
(+0.04).

### Flagging precision vs random (same arm, same budget)

| method | masked b10 | masked b30 | all-case b10 | all-case b30 |
|---|---|---|---|---|
| random (reference) | 0.224 ± 0.041 | 0.244 ± 0.043 | 0.224 ± 0.041 | 0.244 ± 0.043 |
| provenance_only | 0.314, d +0.090 ± 0.017, 5/1/1 | 0.333, d +0.089 ± 0.037, 7/0/0 | 0.567, d +0.343 ± 0.052, 6/0/1 | 0.495, d +0.251 ± 0.099, 5/1/1 |
| bias_only | 0.129, d −0.095 ± 0.036, 0/3/4 | 0.108, d −0.137 ± 0.015, 0/0/7 | 0.824, d +0.600 ± 0.074, 7/0/0 | 0.690, d +0.446 ± 0.094, 7/0/0 |
| oracle | 1.000, 7/0/0 | 1.000, 7/0/0 | 1.000, 7/0/0 | 1.000, 7/0/0 |

Under the masked core, bias_only finds fewer corrupted cases than random. Under the all-case core it becomes the best
non-oracle flagger, with 0.97 precision at b10 on noise and 0.93 on reference and shift. provenance_only also
improves under all-case: d vs masked is +0.252 at b10 (6/1/0) and +0.162 at b30 (5/0/2). The single provenance loss
in all-case is synthetic_combined (0.43 vs random 0.53 at b10). There, random already hits about half the time,
because 168 of the cases are corrupted.

### Accuracy after the edit and retraining (M2) vs random

| method | masked b10 | masked b30 | all-case b10 | all-case b30 |
|---|---|---|---|---|
| random M2 (reference) | 0.8194 ± 0.0431 | 0.8229 ± 0.0453 | 0.8612 ± 0.0625 | 0.8639 ± 0.0616 |
| provenance_only | +0.0059 ± 0.0088, 1/6/0 | +0.0060 ± 0.0081, 2/4/1 | +0.0031 ± 0.0045, 3/4/0 | +0.0029 ± 0.0066, 2/5/0 |
| bias_only | +0.0029 ± 0.0089, 1/5/1 | +0.0011 ± 0.0076, 1/4/2 | −0.0002 ± 0.0022, 0/7/0 | +0.0013 ± 0.0018, 1/5/1 |
| oracle | +0.0071 ± 0.0099, 3/3/1 | +0.0196 ± 0.0103, 7/0/0 | +0.0035 ± 0.0063, 4/3/0 | +0.0070 ± 0.0085, 5/2/0 |

Better flagging precision buys little accuracy. Even perfect flagging (oracle) at b30 adds only +0.010 over M0 under
all-case and +0.026 under masked. Under all-case, bias_only's much higher precision gives no measurable M2 gain over
random. Edit-only M1 differs from random's M1 by at most +0.005 under all-case (oracle b30). M2 − MC accuracy, which is the edit's effect
net of the extra epochs, is between 0.000 and +0.010 for every all-case condition.

### All-case vs masked, same condition

| quantity | provenance_only b10 | bias_only b10 | random b10 | oracle b30 |
|---|---|---|---|---|
| M2 accuracy d | +0.0390 ± 0.0115, 5/2/0 | +0.0386 ± 0.0136, 5/2/0 | +0.0418 ± 0.0196, 5/2/0 | +0.0284 ± 0.0170, 5/1/1 |
| M2 − M0 gain d | −0.0055 ± 0.0123, 1/3/3 | −0.0059 ± 0.0085, 0/3/4 | −0.0027 ± 0.0005, 1/4/2 | −0.0161 ± 0.0134, 0/2/5 |
| M2_minus_MC_loss d (lower better) | +0.0103 ± 0.0252, 3/1/3 | +0.0051 ± 0.0135, 0/5/2 | −0.0026 ± 0.0285, 3/3/1 | +0.0897 ± 0.0299, 0/0/7 |

`results/t1/handoff_20261126/revstress/tables.md` has every budget and metric. The all-case twin ends higher after
revision in every condition, but only because its core starts higher. The edits themselves help it less than they
help the masked core: all-case gains less from M0 to M2 and has a smaller NLL drop against the matched control
(oracle b30: −0.032 vs −0.122).

## 2. synnewpol: the 7 new retention policies on the synthetic sets

Setup, per the config:
- Untuned `_base` core, masked, 300/40.
- Finetune 30 epochs at lr scale 0.1.
- K 0.2/0.5/0.75, seeds 30-32.
- Policies rc_cnn, crr, bbnr, enn, energy_compress, knn_shapley and loo_value, compared with the 8 recorded policies
  of `handoff_synret_r760` on the same core.
- Primary metric: test accuracy after finetune.

### Completeness

| study | complete units |
|---|---|
| `handoff_synnewpol_r760` | 483: 7 sets × (21 policy arms + `_core` + `ch_reliability`) × 3 seeds |
| `handoff_synret_r760` | 525: 7 sets × (24 policy arms + `_core`) × 3 seeds |

Every set has all 21 new arms with 3 seeds and 3 `ch_reliability` seeds. Full-memory check: the new study's
`fullmem_*` equals `handoff_synret_r760` on all 42 task-seed pairs (max |diff| 0), so the pairs share one core.

### Accuracy after finetune vs provenance_only (reference 0.836 / 0.835 / 0.824 at K 0.2 / 0.5 / 0.75)

Full memory reaches 0.820.

| policy | K 0.2 | K 0.5 | K 0.75 |
|---|---|---|---|
| knn_shapley | +0.0295 ± 0.0089, 6/1/0 | +0.0304 ± 0.0113, 7/0/0 | +0.0348 ± 0.0086, 7/0/0 |
| enn | +0.0121 ± 0.0148, 4/1/2 | +0.0073 ± 0.0145, 5/0/2 | +0.0130 ± 0.0130, 5/2/0 |
| bbnr | −0.0022 ± 0.0214, 3/0/4 | −0.0126 ± 0.0114, 1/0/6 | +0.0007 ± 0.0175, 1/3/3 |
| loo_value | −0.0042 ± 0.0008, 4/1/2 | −0.0126 ± 0.0086, 2/2/3 | +0.0037 ± 0.0140, 3/1/3 |
| rc_cnn | −0.0055 ± 0.0095, 3/0/4 | −0.0099 ± 0.0024, 0/4/3 | −0.0051 ± 0.0045, 2/3/2 |
| energy_compress | −0.0106 ± 0.0109, 1/3/3 | −0.0249 ± 0.0070, 0/1/6 | −0.0101 ± 0.0058, 2/1/4 |
| crr | −0.1236 ± 0.1032, 1/0/6 | −0.0172 ± 0.0254, 2/1/4 | −0.0026 ± 0.0161, 2/0/5 |
| random (recorded) | +0.0059 ± 0.0178, 4/0/3 | −0.0035 ± 0.0088, 2/2/3 | +0.0062 ± 0.0174, 3/2/2 |

Against random at the same K, knn_shapley is +0.024 / +0.034 / +0.029 (5/1/1, 7/0/0, 7/0/0) and enn is
+0.006 / +0.011 / +0.007 (3/3/1, 3/4/0, 1/6/0). Every other new policy is at or below random.

- **knn_shapley** reaches 0.866 at K 0.2 and K 0.5, above full memory (0.820). It does best where labels are
  corrupted: +0.05 to +0.08 on noise, +0.03 to +0.05 on combined and +0.03 to +0.07 on shift. This reverses the
  real-data p2 result, where knn_shapley lost to provenance_only at K 0.5 and 0.75 (−0.025, 2/2/6 and −0.024,
  1/3/6). Removing low-value cases is useful when the memory holds many mislabeled ones, and the synthetic sets are
  built that way.
- **crr at K 0.2** collapses: 0.713 after finetune and 0.533 before it (−0.290 vs provenance_only, 0/0/7). Finetune
  recovers +0.18 on average, but not to the reference.
- **Before finetune** (`m_nofinetune`), the ranking is the same: knn_shapley is +0.035 ± 0.005 at K 0.5 (7/0/0), and
  rc_cnn is −0.060 / −0.036 / −0.013 (mostly losses).

### Efficiency and C/H reliability

- Efficiency: all sets are small. Median maintenance overhead is 0.09-0.24 s per unit, with rc_cnn and
  energy_compress the slowest. The median single-query lookup speedup over full memory is 1.16-1.21 at K 0.2 and
  1.01-1.06 at K 0.75.
- C/H reliability: these are Spearman correlations against the full reference query set, averaged over 7 tasks.

  | queries used | C | H | Q |
  |---|---|---|---|
  | 10% | 0.54 | 0.41 | 0.43 |
  | 25% | 0.70 | 0.57 | 0.60 |
  | 50% | 0.83 | 0.78 | 0.76 |

  With 10% of the queries, the case rankings are only moderately stable.

## 3. Independent recompute from raw events

`raw_event_check.py` reads `x/final/<metric>` with tensorboard's `EventFileLoader` only and repeats the pairing,
means, seed sd and W/T/L. It ran on r760 and all three checks match the tables exactly:

| check | raw events | table |
|---|---|---|
| revision_stress M2 accuracy, all-case vs masked, provenance_only b10 | +0.039011, sd 0.011538, 5/2/0 | +0.0390 ± 0.0115, 5/2/0 |
| revision_stress precision, all-case provenance_only vs random, b10 | +0.342857, sd 0.051508, 6/0/1 | +0.3429 ± 0.0515, 6/0/1 |
| synnewpol knn_shapley vs provenance_only, K 0.5, test_accuracy_pre | +0.030403, sd 0.011278, 7/0/0 | +0.0304 ± 0.0113, 7/0/0 |

## 4. Limits

- Seven synthetic sets and 3 seeds per arm. W/T/L counts out of 7 are weak evidence, and none of this is
  confirmatory.
- The simulated reviewer always relabels correctly. Precision and recall are known only because the generator
  records the corruption.
- The revision_stress masked rows are the recorded `plan_full_r760` units (an earlier commit). The all-case twin ran
  on 2026-11-26. Only `pre_topk_mask` differs by design, but code drift between the two recordings is not excluded.
  The identical random and oracle selections show that data and corruption agree.
- synthetic_reference and synthetic_shift give identical flagging precision, because the shift acts on the test set
  only.
- The synnewpol units may have been recorded before fix 4235e10 (engine tie rule). The full-memory check agrees
  exactly, but tied rows in the maintained cores are unchecked, as for real-data p2.
- ECE, per-group accuracies (in-domain, shifted, rare, boundary) and the flip matrices are in `units.csv` and the
  dumps but are not analyzed here.
