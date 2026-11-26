# Pack p1: the revised all-case baseline (declared by us, 2026-11-26)

Status: **exploratory, our choice, not a PI decision.** The 10-31 re-evaluation decision
(`docs/T1_DISCUSSION_NOTES_20261030.md`, step 1) asks to "declare the training schedule, loss weights, and
correction-size penalty explicitly". The owner ordered on 2026-11-26 not to wait for the PI, so we declare them here.
Nothing below replaces a PI decision, and no result of this pack is confirmatory.

## Declaration

| item | value |
|---|---|
| prediction mask | `core.pre_topk_mask: false` everywhere (all-case softmax). Neighbour-based bias initialisation is kept (`bias_init: knn_mean`, `bias_init_k: 5`). |
| core hyperparameters, per task | the validation-selected winner of `review_gridallcase_r760` (all-case grid: representation raw/mlp, lr_case 0.001/0.01, glocal_fw_set_num 1/4, tau 0.25/0.5/1/2; screen 150/30, refit 600/60, development seeds 20-21; batch 64, lr_glocal = lr_case, lr_feature = lr_case/10 as the grid trains them). `allcase_grid_selection.json` lists the 29 winners; each was recomputed on r760 from the recorded screen and refit trial units (validation rows only) and equals the grid's `selection.json`. No test metric was used. |
| core training budget | 300 epochs, patience 40, as in the frozen F configuration (the all-case grid refit used 600/60; we keep F's schedule, as ordered). |
| training schedule and loss weights | as in the frozen F configuration of each task (`confirm_final_20261013/<task>.yaml`, condition `F`): `joint_rr`, summed objective, 5 adapter warm-up epochs, retrieval_lr_scale 1, adapter_lr 0.003, w_pre 1, w_post / w_near / v_small as frozen per task, v_post 1, v_delta 1, s_task 0.5, 120 sync epochs, patience 30. Diabetes keeps its frozen F, the frozen-core reuse arm. |
| correction-size penalty | the recorded default: `L_small` above the trained-bias-region allowance (`free_radius_method: trained_bias_region`, mean aggregation, identity transform), weight `v_small` as frozen. |
| one declared variant | `F_zero_radius`: F with `free_radius_method: zero`, i.e. scale-normalised squared L2 of the correction from zero. Only on the 18 tasks whose frozen F has `v_small > 0`; on the 10 tasks with `v_small: 0` the penalty is off and the variant would train the same model, so it is not run. Everything else, including the total post-adaptation weight, is equal between F and F_zero_radius. |

## Experiments

1. `allcaseF/<task>.yaml` (29 confirmation tasks, QSAR `regression__tabarena_363697` excluded), study
   `handoff_allcaseF_r760`, seeds 50, 51, 52. Arms: retrieval_only, the frozen-core reuse arm, F, F_zero_radius.
   - `maskedF/<task>.yaml`: the frozen masked configuration (masked-grid core, `pre_topk_mask: true`) with only the F arm,
     same study, same seeds and host, so all-case vs masked is paired on unseen seeds.
   - Matched baselines kNN, MLP, CatBoost and TabM: `tools/t1_grid_search.py evaluate` of their recorded selections
     (`results-in/t1_pi20260920/grid_full_20261009` for the 20 legacy tasks, `baselines_r760/grid_transfer_<family>` for
     the 9 transfer tasks), seeds 50-52, same splits. The 7 small tasks evaluated on seeds 50-69 in
     `baselines_r760/eval50` (iris, wine, zebra, zebra_special, body_fat, diabets, yacht) already have these rows on
     r760; they are read from there, not refitted, and those 7 tasks run on r760.
2. `plan/<group>__<task>__<dataset>.yaml` (195 files): the all-case twin of every plan_full_r760 experiment of
   `tools/t1_execute_study.py` (retention, reuse_classification, reuse_regression, synchronization,
   sync_encoder_control, sync_mcb, mcb_retention, exact_removal_diagnostics, removal_retraining_diagnostics,
   adapted_retraining_diagnostics, nominal_reuse, nominal_sync, revision_stress). Study `handoff_allcaseplan_r760`,
   seeds 30-32, experiment names unchanged, each experiment and dataset on the host that recorded its plan_full rows.
   Tuned groups get the all-case grid winner of every task in `core_by_dataset`; groups that plan_full runs on the untuned
   `_base` core by design (sync_encoder_control, sync_mcb, mcb_retention, nominal_*, revision_stress) and synthetic_diag
   (no grid) keep that core with only `pre_topk_mask` switched.

## Known limits

- Seeds 50-52 are the prereg's `evaluation_small_tasks` seeds (owner's order). Results stay exploratory.
- The all-case grid selected under a 600/60 refit; the F core trains 300/40. Earlier pilots saw all-case pretrains reach
  their epoch cap more often; `core_best_epoch` is recorded per unit to check this.
- Under all-case softmax `top_k` is inert for prediction but still sets the stability tracker's neighbourhood size.
