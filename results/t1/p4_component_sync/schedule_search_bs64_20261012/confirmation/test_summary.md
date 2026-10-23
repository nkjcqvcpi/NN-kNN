# Schedule confirmation (test stream, evaluation seeds)

Error: 1 - accuracy (classification) or train-standardized RMSE (regression), mean over seeds; lower is better. Relative columns are medians over tasks of (condition - `alternating_default`) / `alternating_default`; "pre" is the same runs' retrieval-only output. Selection used validation only, before these results.

| condition | mean test rank | better / worse tasks vs baseline | median rel. error | median rel. pre-adaptation error |
|---|---|---|---|---|
| joint_separated_w5__w_post=0.5__w_near=0.1__v_small=0.0__adapter_lr=0.003 | 8.31 | 13 / 3 | -1.3% | -0.6% |
| joint_summed_w5__w_post=0.5__w_near=0.0__v_small=0.1__adapter_lr=0.003 | 8.31 | 11 / 5 | -2.1% | -1.3% |
| joint_summed_w5__w_post=1.0__w_near=0.0__v_small=0.0__adapter_lr=0.001 | 8.38 | 13 / 2 | -2.1% | -3.4% |
| joint__joint_objective=summed__adapter_warmup_epochs=0__retrieval_lr_scale=1.0 | 8.56 | 12 / 3 | -2.1% | -1.5% |
| joint_summed_w5__w_post=0.25__w_near=0.1__v_small=0.0__adapter_lr=0.003 | 8.88 | 11 / 5 | -1.8% | -0.3% |
| joint_summed_w5__w_post=1.0__w_near=0.1__v_small=0.0__adapter_lr=0.003 | 9.19 | 11 / 5 | -1.0% | -0.6% |
| joint_separated_w5__w_post=0.25__w_near=0.1__v_small=0.0__adapter_lr=0.003 | 9.69 | 9 / 7 | -1.2% | -0.6% |
| joint_separated_w5__w_post=0.5__w_near=0.0__v_small=0.0__adapter_lr=0.003 | 9.69 | 11 / 4 | -1.5% | -0.7% |
| joint_summed_w5__w_post=1.0__w_near=0.0__v_small=0.0__adapter_lr=0.003 | 9.69 | 12 / 4 | -1.8% | -0.6% |
| joint_summed_w5__w_post=0.5__w_near=0.0__v_small=0.0__adapter_lr=0.003 | 9.81 | 9 / 7 | -0.7% | -1.3% |
| joint__joint_objective=separated__adapter_warmup_epochs=0__retrieval_lr_scale=1.0 | 9.94 | 10 / 6 | -1.2% | -0.3% |
| joint_separated_w5__w_post=0.25__w_near=0.0__v_small=0.0__adapter_lr=0.003 | 10.38 | 11 / 4 | -1.4% | -0.4% |
| joint__joint_objective=separated__adapter_warmup_epochs=0__retrieval_lr_scale=0.3 | 11.00 | 10 / 6 | -0.9% | -0.2% |
| joint__joint_objective=summed__adapter_warmup_epochs=5__retrieval_lr_scale=0.3 | 11.12 | 12 / 3 | -0.6% | -0.6% |
| alternating_b2__retrieval_lr_scale=1.0 | 11.19 | 10 / 6 | -0.6% | -1.4% |
| alternating__block_epochs=5__retrieval_lr_scale=1.0 | 11.75 | 9 / 7 | -1.2% | -1.3% |
| frozen_reuse__combined | 12.88 | 6 / 9 | +0.0% | -0.2% |
| sequential__retrieval_lr_scale=0.1 | 12.88 | 9 / 6 | -0.7% | -0.3% |
| alternating_default | 13.12 | 0 / 0 | +0.0% | +0.0% |
| retrieval_only | 15.25 | 5 / 9 | +0.7% | -0.2% |

## Per task

| task | retrieval_only | alternating_default | frozen_reuse__combined | sequential__retrieval_lr_scale=0.1 | global_selected | per_task_selected | per-task selection |
|---|---|---|---|---|---|---|---|
| abalone | 0.7069 | 0.6976 | 0.7007 | 0.7125 | 0.6970 | **0.6900** | joint__joint_objective=separated__adapter_warmup_epochs=0__retrieval_lr_scale=1.0 |
| airfoil | 0.2351 | 0.2370 | 0.2371 | 0.2348 | 0.2333 | **0.2327** | joint__joint_objective=separated__adapter_warmup_epochs=0__retrieval_lr_scale=0.3 |
| balance | 0.0624 | 0.0656 | 0.0608 | 0.0592 | **0.0416** | 0.0528 | joint_summed_w5__w_post=1.0__w_near=0.1__v_small=0.0__adapter_lr=0.003 |
| breast_cancer | 0.0404 | **0.0333** | 0.0421 | 0.0351 | 0.0439 | 0.0456 | joint_summed_w5__w_post=0.5__w_near=0.0__v_small=0.0__adapter_lr=0.003 |
| digits | 0.0228 | 0.0211 | 0.0217 | **0.0200** | 0.0217 | 0.0211 | joint_summed_w5__w_post=1.0__w_near=0.1__v_small=0.0__adapter_lr=0.003 |
| energy_efficiency | 0.0500 | **0.0487** | 0.0494 | 0.0495 | 0.0510 | 0.0492 | joint_summed_w5__w_post=0.5__w_near=0.0__v_small=0.1__adapter_lr=0.003 |
| iris | 0.0467 | **0.0467** | 0.0467 | 0.0600 | 0.0800 | 0.0600 | joint_separated_w5__w_post=0.25__w_near=0.0__v_small=0.0__adapter_lr=0.003 |
| student_performance | 0.3964 | 0.4013 | 0.3976 | 0.3984 | 0.3934 | **0.3819** | alternating_b2__retrieval_lr_scale=1.0 |
| tabarena_363614 | 0.0154 | 0.0154 | 0.0140 | 0.0134 | **0.0120** | 0.0160 | joint_summed_w5__w_post=1.0__w_near=0.0__v_small=0.0__adapter_lr=0.003 |
| tabarena_363625 | 0.3771 | 0.3782 | 0.3728 | 0.3754 | **0.3707** | 0.3755 | joint__joint_objective=separated__adapter_warmup_epochs=0__retrieval_lr_scale=1.0 |
| tabarena_363626 | 0.2783 | 0.2717 | 0.2796 | 0.2741 | **0.2682** | 0.2705 | joint__joint_objective=summed__adapter_warmup_epochs=5__retrieval_lr_scale=0.3 |
| tabarena_363675 | 0.4106 | 0.4044 | 0.4099 | 0.4231 | 0.3913 | **0.3862** | joint__joint_objective=summed__adapter_warmup_epochs=0__retrieval_lr_scale=1.0 |
| wine | 0.0556 | 0.0611 | 0.0556 | 0.0556 | **0.0167** | 0.0556 | joint_summed_w5__w_post=1.0__w_near=0.0__v_small=0.0__adapter_lr=0.001 |
| yacht | 0.1490 | 0.1197 | 0.1198 | 0.1040 | **0.0804** | 0.0834 | joint__joint_objective=separated__adapter_warmup_epochs=0__retrieval_lr_scale=0.3 |
| zebra | 0.0727 | 0.0727 | 0.0727 | 0.0727 | **0.0182** | 0.0455 | joint_summed_w5__w_post=1.0__w_near=0.0__v_small=0.0__adapter_lr=0.003 |
| zebra_special | 0.2909 | 0.2773 | 0.2727 | **0.2591** | 0.3091 | 0.2909 | alternating__block_epochs=5__retrieval_lr_scale=1.0 |

## Selected schedules vs references

- global_selected vs `alternating_default`: better on 11, worse on 5, tied on 0 tasks; median relative difference -1.8%, Wilcoxon p = 0.323
- per_task_selected vs `alternating_default`: better on 10, worse on 6, tied on 0 tasks; median relative difference -0.9%, Wilcoxon p = 0.404
- global_selected vs `frozen_reuse__combined`: better on 12, worse on 4, tied on 0 tasks; median relative difference -1.3%, Wilcoxon p = 0.144
- per_task_selected vs `frozen_reuse__combined`: better on 10, worse on 5, tied on 1 tasks; median relative difference -1.7%, Wilcoxon p = 0.427
- global_selected vs `sequential__retrieval_lr_scale=0.1`: better on 11, worse on 5, tied on 0 tasks; median relative difference -1.7%, Wilcoxon p = 0.375
- per_task_selected vs `sequential__retrieval_lr_scale=0.1`: better on 9, worse on 6, tied on 1 tasks; median relative difference -0.8%, Wilcoxon p = 0.532
- global_selected vs `retrieval_only`: better on 12, worse on 4, tied on 0 tasks; median relative difference -2.7%, Wilcoxon p = 0.144
- per_task_selected vs `retrieval_only`: better on 11, worse on 4, tied on 1 tasks; median relative difference -2.0%, Wilcoxon p = 0.125
