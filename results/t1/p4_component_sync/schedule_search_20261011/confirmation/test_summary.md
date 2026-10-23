# Schedule confirmation (test stream, evaluation seeds)

Error: 1 - accuracy (classification) or train-standardized RMSE (regression), mean over seeds; lower is better. Relative columns are medians over tasks of (condition - `alternating_default`) / `alternating_default`; "pre" is the same runs' retrieval-only output. Selection used validation only, before these results.

| condition | mean test rank | better / worse tasks vs baseline | median rel. error | median rel. pre-adaptation error |
|---|---|---|---|---|
| joint_scratch__joint_objective=separated | 9.31 | 12 / 4 | -7.6% | -4.2% |
| joint__joint_objective=summed__adapter_warmup_epochs=0__retrieval_lr_scale=1.0 | 9.75 | 13 / 2 | -4.5% | -0.3% |
| joint_summed_w5__w_post=1.0__w_near=0.1__v_small=0.0__adapter_lr=0.003 | 10.06 | 13 / 3 | -11.6% | -0.3% |
| joint_summed_w5__w_post=1.0__w_near=0.0__v_small=0.0__adapter_lr=0.001 | 10.94 | 12 / 3 | -6.3% | -2.3% |
| joint_summed_w5__w_post=0.5__w_near=0.1__v_small=0.0__adapter_lr=0.001 | 11.06 | 13 / 3 | -5.3% | -0.4% |
| joint_summed_w5__w_post=1.0__w_near=0.1__v_small=0.0__adapter_lr=0.001 | 11.12 | 12 / 4 | -4.5% | -0.9% |
| joint_summed_w5__w_post=1.0__w_near=0.0__v_small=0.1__adapter_lr=0.001 | 12.31 | 11 / 5 | -4.7% | -3.3% |
| joint__joint_objective=separated__adapter_warmup_epochs=5__retrieval_lr_scale=1.0 | 12.88 | 11 / 5 | -1.9% | -0.8% |
| joint_summed_w5__w_post=0.5__w_near=0.0__v_small=0.0__adapter_lr=0.001 | 12.88 | 12 / 3 | -4.5% | -3.0% |
| joint__joint_objective=summed__adapter_warmup_epochs=5__retrieval_lr_scale=1.0 | 13.06 | 11 / 5 | -3.4% | -0.4% |
| joint_scratch__joint_objective=summed | 13.25 | 10 / 6 | -3.9% | +0.7% |
| joint_summed_w5__w_post=0.5__w_near=0.0__v_small=0.0__adapter_lr=0.003 | 13.38 | 11 / 4 | -5.4% | -1.7% |
| joint__joint_objective=summed__adapter_warmup_epochs=5__retrieval_lr_scale=0.3 | 13.50 | 13 / 3 | -3.7% | -1.2% |
| joint_summed_w5__w_post=0.25__w_near=0.0__v_small=0.0__adapter_lr=0.001 | 13.50 | 10 / 5 | -3.3% | -4.2% |
| joint__joint_objective=summed__adapter_warmup_epochs=0__retrieval_lr_scale=0.3 | 13.81 | 11 / 5 | -1.6% | -0.7% |
| joint__joint_objective=separated__adapter_warmup_epochs=5__retrieval_lr_scale=0.3 | 14.38 | 12 / 2 | -2.2% | -0.0% |
| joint_separated_w5__w_post=1.0__w_near=0.1__v_small=0.1__adapter_lr=0.003 | 14.62 | 10 / 5 | -3.1% | +0.2% |
| alternating__block_epochs=5__retrieval_lr_scale=1.0 | 14.69 | 11 / 5 | -3.7% | +1.6% |
| alternating__block_epochs=1__retrieval_lr_scale=1.0 | 15.25 | 11 / 4 | -1.7% | +0.3% |
| joint_separated_w5__w_post=0.25__w_near=0.1__v_small=0.0__adapter_lr=0.001 | 15.50 | 10 / 6 | -2.6% | -0.4% |
| joint__joint_objective=summed__adapter_warmup_epochs=5__retrieval_lr_scale=0.1 | 15.81 | 9 / 5 | -1.3% | -0.4% |
| joint__joint_objective=separated__adapter_warmup_epochs=5__retrieval_lr_scale=0.1 | 16.38 | 8 / 7 | -0.1% | -0.1% |
| frozen_reuse__combined | 17.00 | 8 / 6 | -0.0% | -0.0% |
| joint__joint_objective=summed__adapter_warmup_epochs=0__retrieval_lr_scale=0.1 | 17.81 | 8 / 7 | -0.8% | +0.0% |
| alternating_default | 18.44 | 0 / 0 | +0.0% | +0.0% |
| sequential__retrieval_lr_scale=0.1 | 21.00 | 7 / 8 | +0.3% | -0.6% |
| sequential__retrieval_lr_scale=0.3 | 21.31 | 6 / 9 | +0.8% | -0.0% |
| retrieval_only | 23.00 | 3 / 11 | +0.9% | -0.0% |

## Per task

| task | retrieval_only | alternating_default | frozen_reuse__combined | sequential__retrieval_lr_scale=0.1 | global_selected | per_task_selected | per-task selection |
|---|---|---|---|---|---|---|---|
| abalone | 0.7007 | 0.6981 | 0.6961 | 0.7058 | 0.7027 | **0.6946** | joint_separated_w5__w_post=1.0__w_near=0.1__v_small=0.1__adapter_lr=0.003 |
| airfoil | 0.2512 | 0.2490 | 0.2490 | 0.2507 | 0.2343 | **0.2300** | joint_summed_w5__w_post=1.0__w_near=0.0__v_small=0.1__adapter_lr=0.001 |
| balance | 0.0656 | 0.0640 | 0.0608 | 0.0624 | 0.0384 | **0.0352** | joint__joint_objective=separated__adapter_warmup_epochs=5__retrieval_lr_scale=1.0 |
| breast_cancer | **0.0298** | 0.0421 | 0.0333 | 0.0368 | 0.0316 | 0.0333 | alternating__block_epochs=5__retrieval_lr_scale=1.0 |
| digits | 0.5439 | 0.0272 | 0.0244 | 0.0283 | 0.0233 | **0.0228** | joint_summed_w5__w_post=0.5__w_near=0.0__v_small=0.0__adapter_lr=0.001 |
| energy_efficiency | 0.0512 | **0.0474** | 0.0492 | 0.0486 | 0.0498 | 0.0477 | joint__joint_objective=summed__adapter_warmup_epochs=5__retrieval_lr_scale=0.1 |
| iris | 0.0600 | **0.0467** | 0.0600 | 0.0600 | 0.0600 | 0.0667 | joint_summed_w5__w_post=0.5__w_near=0.1__v_small=0.0__adapter_lr=0.001 |
| student_performance | 0.4015 | 0.3995 | 0.3996 | 0.3980 | 0.3978 | **0.3807** | alternating__block_epochs=5__retrieval_lr_scale=1.0 |
| tabarena_363614 | 0.0207 | 0.0220 | 0.0147 | 0.0160 | 0.0154 | **0.0120** | joint_summed_w5__w_post=1.0__w_near=0.0__v_small=0.0__adapter_lr=0.001 |
| tabarena_363625 | 0.3962 | 0.3822 | 0.3842 | 0.3868 | **0.3485** | 0.3550 | joint_scratch__joint_objective=summed |
| tabarena_363626 | 0.2837 | 0.2741 | 0.2753 | 0.2735 | **0.2687** | 0.2693 | alternating__block_epochs=1__retrieval_lr_scale=1.0 |
| tabarena_363675 | 0.4034 | 0.3998 | 0.4037 | 0.4124 | **0.3904** | 0.3969 | joint__joint_objective=summed__adapter_warmup_epochs=5__retrieval_lr_scale=0.3 |
| wine | 0.0611 | 0.0611 | 0.0611 | 0.0667 | **0.0167** | **0.0167** | joint_summed_w5__w_post=1.0__w_near=0.1__v_small=0.0__adapter_lr=0.003 |
| yacht | 0.1572 | 0.1149 | 0.0957 | 0.1005 | **0.0808** | **0.0808** | joint_summed_w5__w_post=1.0__w_near=0.1__v_small=0.0__adapter_lr=0.003 |
| zebra | 0.0545 | 0.0545 | 0.0545 | 0.0545 | **0.0364** | **0.0364** | joint_summed_w5__w_post=1.0__w_near=0.1__v_small=0.0__adapter_lr=0.003 |
| zebra_special | 0.2727 | 0.2864 | 0.2682 | 0.2727 | **0.2364** | 0.2455 | joint_separated_w5__w_post=0.25__w_near=0.1__v_small=0.0__adapter_lr=0.001 |

## Selected schedules vs references

- global_selected vs `alternating_default`: better on 13, worse on 3, tied on 0 tasks; median relative difference -11.6%, Wilcoxon p = 0.008
- per_task_selected vs `alternating_default`: better on 14, worse on 2, tied on 0 tasks; median relative difference -11.0%, Wilcoxon p = 0.004
- global_selected vs `frozen_reuse__combined`: better on 12, worse on 4, tied on 0 tasks; median relative difference -4.9%, Wilcoxon p = 0.005
- per_task_selected vs `frozen_reuse__combined`: better on 15, worse on 1, tied on 0 tasks; median relative difference -7.2%, Wilcoxon p = 0.002
- global_selected vs `sequential__retrieval_lr_scale=0.1`: better on 14, worse on 2, tied on 0 tasks; median relative difference -8.2%, Wilcoxon p = 0.000
- per_task_selected vs `sequential__retrieval_lr_scale=0.1`: better on 15, worse on 1, tied on 0 tasks; median relative difference -8.9%, Wilcoxon p = 0.001
- global_selected vs `retrieval_only`: better on 13, worse on 3, tied on 0 tasks; median relative difference -9.4%, Wilcoxon p = 0.001
- per_task_selected vs `retrieval_only`: better on 14, worse on 2, tied on 0 tasks; median relative difference -9.2%, Wilcoxon p = 0.009
