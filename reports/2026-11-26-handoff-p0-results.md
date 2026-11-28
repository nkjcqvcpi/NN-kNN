## Handoff 2026-11-26, pack p0: results (exploratory, our choices, 3 seeds)

**Source.** accel `runs/` is empty and no sync tool fills it. The q-lab/xps8960 sync only mirrors to `/var/tmp/nnknn-sync/<host>/mirror` and then places on r760. So every number below comes from r760 `runs/t1/`, which is the single copy. No study is missing there. Pipeline: `neural-cbr/tools/t1_handoff_p0_summary.py`, new and uncommitted. Its `dump` step ran read-only on r760 over 5,869 units (p0 plus references). Its `analyze` step ran on accel and wrote `results/t1/handoff_20261126/p0/{aggregate,per_dataset,completeness}.csv` and `missing_pairs.txt`.

**Completeness.** Every p0 unit is `complete`, has exactly 1 event file and loads with no error. Every candidate unit was paired with a reference unit recorded on the same host.

| study / experiment | units (incl. _core) | datasets | seeds |
|---|---|---|---|
| reusesweep cls / reg | 360 / 165 | 10 / 11 | 30-32 |
| syncsens cls / reg | 300 / 297 | 10 / 11 | 30-32 |
| mcbmom 0p9 and 0p999 retention cls / reg | 60 / 66 each | 10 / 11 | 30-32 |
| mcbmom 0p9 and 0p999 sync cls / reg | 120 / 132 each | 10 / 11 | 30-32 |
| protect synthetic_rare10 / anneal rare10 | 273 / 39 | 7 / 1 | 30-32 |
| bik3 / bik10 / bik20 (both arms) | 72 each | 12 | 70-72 |

**References.**
- plan_full_r760: reuse_classification, reuse_regression, synchronization (alternating_rr, rr_zero_radius), mcb_retention (mcb1_*), sync_mcb, and retention for anneal.
- handoff_synret_r760 for the synthetic rare10 runs.
- pretopk_expanded_r760 (bias_init_k 5) for bik.

**Rule.** For each task, d_s = condition − reference on seed s. **b** (better): the mean of d favours the condition and at least 2 of 3 seeds do. **w** (worse): the same in the other direction. **i** (identical): all three d_s are 0. **t** (tie): anything else. There is no tolerance, so differences of 1e-5 also count as b or w. Cand and ref are means over tasks of the 3-seed means. Per-task sd is in `per_dataset.csv`.

### Reuse loss weights (adapter on the recorded cores; pre-adapter accuracy is identical in 10/10 tasks)
| condition (λdiff,λcls) | metric | cand | ref | Δ | b/t/w/i |
|---|---|---|---|---|---|
| NRS (1,0.3) | ECE post ↓ | 0.313 | 0.253 | +0.059 | 0/0/10/0 |
| NRS (3,1) | ECE post ↓ | 0.308 | 0.253 | +0.055 | 0/0/10/0 |
| NRS (0.3,1) | ECE post ↓ | 0.186 | 0.253 | −0.067 | 9/0/1/0 |
| NRS (1,3) | ECE post ↓ | 0.190 | 0.253 | −0.063 | 9/1/0/0 |
| NRS, any pair | acc post | 0.902-0.906 | 0.908 | −0.002 to −0.007 | at most 1 b, 2-3 w |
| logit_residual, any pair | acc / ECE | Δ ≤ 0.003 | | | mostly t/i |
| clip_normalize: combined / cls_only / diff_only | ECE post ↓ | 0.061 / 0.077 / 0.067 | 0.253 / 0.125 / 0.368 | −0.19 / −0.05 / −0.30 | 8/1/1, 8/0/2, 9/1/0 |
| regression, each of 4 pairs | RMSE post ↓ | 0.4174-0.4175 | 0.4168 | +0.0005 to +0.0007 | 0-1 b, 3-7 w |

Under clip_normalize, accuracy and NLL_T are identical in 10/10 tasks: training is the same and only the probability mapping changes.

### Sync sensitivity (one key changed from alternating_rr or rr_zero_radius)
| condition | cls acc post Δ (b/t/w/i) | cls ECE post Δ (b/t/w/i) | reg RMSE post Δ (b/t/w/i) |
|---|---|---|---|
| s_task 0.25 | +0.003 (2/4/0/4) | +0.028 (0/1/8/1) | −0.001 (1/6/2/2) |
| s_task 1.0 | −0.000 (1/6/0/3) | −0.033 (8/1/0/1) | −0.001 (2/6/1/2) |
| zero-radius s_task 0.25 | +0.000 (1/5/2/2) | +0.039 (0/0/10/0) | +0.003 (4/1/6/0) |
| zero-radius s_task 1.0 | −0.001 (1/6/2/1) | −0.037 (9/1/0/0) | −0.000 (3/5/3/0) |
| median / shrunk / clipped radius | ≤ 0.0003 | +0.012 / +0.011 / −0.006 | ≥ −0.0013, mostly t |
| cohort-normalized (cls only) | +0.000 (0/2/0/8) | −0.000 | n/a |
| t*=0, no core pretrain | **−0.054** (2/3/4/1) | −0.005 (4/1/5/0) | **+0.111** (2/1/8/0) |

### MCB momentum (plan_full 0.99; the cores are retrained, so pairs are not same-core)
- **Retention, accuracy pre.** cls with 0.9: +0.011 (4/5/1) and +0.013 (4/2/4). cls with 0.999: +0.000 and +0.008 (4/1/5).
- **Retention, RMSE pre.** reg with 0.9: +0.012 (3/1/7) and +0.013 (5/1/5). reg with 0.999: −0.002 (4/3/4) and +0.016 (provenance_only, 1/2/8).
- **Sync.** All six schedule × momentum cells are mixed in cls accuracy, from −0.014 to +0.015. In regression, 0.999 alternating_rr is +0.020 RMSE (3/1/7). Every other sync cell is |Δ| ≤ 0.007 with mixed counts.

### Rare-cohort protection (rare10)
- **Synthetic (7 sets × 4 policies × 3 K).** 0 protected cases, and every metric is identical to synret in 252/252 pairs (accuracy, kept_fraction_rare, kept_fraction_corrupted). This confirms the declared no-op.
- **Anneal (tabarena_363614).** 4.33 cases protected per unit, out of 4-5. Across the 12 policy × K cells, accuracy Δ is between −0.003 and +0.003. Five cells are identical in all seeds and the other seven are ties (corrected in the evening refresh; the morning text said six identical). This is one task, so it gives no direction.

### bias_init_k (reference k = 5, seeds 70-72, 12 tasks; test_error mixes 1−accuracy and RMSE)
| k | all-case arm Δ (b/t/w/i) | max \|Δ\| all-case | masked arm Δ (b/t/w/i) |
|---|---|---|---|
| 3 | −0.0000 (5/0/2/5) | 0.0003 | −0.011 (5/1/4/2) |
| 10 | +0.0000 (2/0/5/5) | 0.0003 | −0.013 (4/4/2/2) |
| 20 | +0.0000 (2/0/5/5) | 0.0007 | −0.007 (2/4/3/3) |

In the masked arm, yacht alone accounts for most of the mean: −0.14 / −0.13 / −0.10, with large seed spread.

**Independent check.** A separate reader parses the TFRecord framing itself, decodes with tensorboard protos only and uses no project code. It read the accel mirror copies of the 60 q-lab/xps8960 units for "NRS diff_only, clip_normalize vs softmax, ECE post". It reproduced 0.0670 vs 0.3678, Δ −0.3007, 9 better / 0 worse, exactly as the script found. It also reproduced bik3 masked yacht at Δ −0.1378.

**What this shows and does not show.**
- It shows that the reuse λ ratio mainly moves calibration. More weight on L_cls lowers ECE, more on L_diff raises it, and NRS accuracy is unchanged or slightly lower.
- clip_normalize changes evaluation only, and it brings ECE far below the recorded softmax ECE.
- For regression, the λ pairs only rescale the loss. RMSE rises by under 0.001, and the counts lean worse (3-7 of 11 tasks worse, 0-1 better).
- s_task trades classification ECE in a monotone direction (larger s_task, lower ECE) and leaves accuracy and regression RMSE unchanged.
- The radius variants barely matter.
- Dropping core pretraining hurts, but that comparison is confounded and is not a pure t* test.
- The MCB momentum results show no consistent direction.
- bias_init_k does not matter for the all-case arm.
- None of this is a significance test. There are 3 seeds, the values are our own exploratory choices, and the masked-arm and momentum effects sit within seed noise on most tasks.

**Evening refresh (2026-11-26, 22:40 MST).** No file of the p0 studies or of their reference experiments changed on r760 after 14:00 UTC. A fresh read-only `dump` on r760 (5,869 units) is byte-identical to the morning dump, and `analyze` reproduces all four output files byte for byte. All 2,130 candidate/reference unit pairs were recorded on the same host. A second independent re-read from the raw r760 event files (same no-project-code reader, paired statistics recomputed separately) reproduced two more headline rows exactly: sync_sens cls s_task 1.0 ECE post 0.2490 vs 0.2820 (Δ −0.0331, 8/1/0/1) and MCB momentum 0.999 sync regression alternating_rr RMSE post 0.5465 vs 0.5271 (Δ +0.0195, 3/1/7/0). Details in `results/t1/handoff_20261126/p0/indep_check.txt`.
