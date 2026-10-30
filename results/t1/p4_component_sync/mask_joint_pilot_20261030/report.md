# Mask x joint-training pilot, October 30, 2026

Exploratory 2 x 2 comparison on 6 small legacy datasets and seeds [70, 71, 72]: 36 retrieval pretrains and 72 frozen/joint continuations.

Main finding: all-case joint training has more favorable mean retrieval-loss drift on 4 of 6 tasks. Raw retrieval loss worsens at the selected checkpoint in 5/18 masked joint runs versus 1/18 all-case joint runs; at the last checkpoint, 11/18 versus 7/18. Frozen retrievers are unchanged in all 36 controls. These are descriptive point outcomes, not statistical significance claims.

A = masked + frozen; B = masked + joint; C = all-case + frozen; D = all-case + joint. Within each mask condition, frozen and joint arms share an identical pretrained checkpoint and Adam state, adapter initialization, five-epoch adapter warm-up and minibatch order. The warm-up endpoint hashes are checked. Both use the same summed objective after warm-up; only retrieval trainability differs. This matched frozen control is not an exact replication of the historical sequential arm's adapter-only objective.

Only training/validation streams influence fitting and checkpoint selection. Identity leave-one-out is retained in training. Bias initialization uses bias_init_k=5 in both mask conditions; masking uses top_k=5. Raw features, one weight set, tau=1. No retention, guard, encoder, or tuning. Regression preprocessing uses model.regression_workflow; Student Performance's category vocabulary is fitted on train only.

Pretraining: at most 300 epochs, patience 40; continuation: 85 epochs including 5 warm-up, patience 20. Best checkpoint uses validation post-adaptation loss, including warm-up. Last executed checkpoint is also recorded to expose deterioration that model selection may discard.

## Standalone retrieval error at the selected checkpoint

Lower is better. Classification uses error rate (fraction); regression uses standardized-target RMSE. Frozen-arm retrieval equals its own pretrained reference exactly.

| Dataset | Masked pretrained | B: masked joint retrieval | All-case pretrained | D: all-case joint retrieval |
| --- | ---: | ---: | ---: | ---: |
| iris | 0.0444 | 0.0444 | 0.0111 | 0.0111 |
| wine | 0.0463 | 0.0278 | 0.0093 | 0.0093 |
| breast_cancer | 0.0409 | 0.0322 | 0.0322 | 0.0322 |
| yacht | 0.1447 | 0.1088 | 0.1381 | 0.1257 |
| energy_efficiency | 0.0554 | 0.0548 | 0.0720 | 0.0662 |
| student_performance | 0.5105 | 0.5136 | 0.4155 | 0.4155 |

## Raw retrieval-loss change relative to each arm's own pretrained reference

Classification uses NLL with the implementation's 1e-8 probability floor; regression uses MSE. Positive drift means worse retrieval. Interaction = all-case joint drift minus masked joint drift; negative values favor all-case for preserving retrieval loss. Do not pool these raw units across tasks. A negative interaction alone is not proof that damage was prevented: the masked arm may itself improve.

| Dataset | Selected masked drift | Selected all-case drift | Selected interaction | Last masked drift | Last all-case drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| iris | +0.00410 | -0.00027 | -0.00437 | +0.01260 | -0.00003 |
| wine | -0.00795 | -0.00262 | +0.00533 | -0.00897 | -0.00470 |
| breast_cancer | +0.00285 | +0.00026 | -0.00258 | +0.06720 | +0.00457 |
| yacht | -0.01256 | -0.00325 | +0.00931 | -0.01064 | -0.00327 |
| energy_efficiency | -0.00006 | -0.00078 | -0.00072 | +0.03202 | -0.00076 |
| student_performance | +0.00442 | +0.00000 | -0.00442 | +0.00886 | +0.00377 |

![Paired retrieval-loss drift](retrieval_loss_drift.png)

Dots show each seed, diamonds show means; thin lines pair the same seed. Panels use separate scales. The dashed zero line is no change from the pretrained reference. Values are not confidence intervals.

## Final adapted error at the selected checkpoint

| Dataset | A: masked frozen | B: masked joint | C: all-case frozen | D: all-case joint | B selected after warm-up | D selected after warm-up |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| iris | 0.0333 | 0.0222 | 0.0111 | 0.0111 | 2/3 | 2/3 |
| wine | 0.0463 | 0.0278 | 0.0093 | 0.0093 | 3/3 | 3/3 |
| breast_cancer | 0.0439 | 0.0380 | 0.0380 | 0.0380 | 1/3 | 2/3 |
| yacht | 0.0987 | 0.0796 | 0.0926 | 0.0937 | 3/3 | 3/3 |
| energy_efficiency | 0.0540 | 0.0543 | 0.0667 | 0.0610 | 1/3 | 3/3 |
| student_performance | 0.4892 | 0.4917 | 0.4201 | 0.4201 | 2/3 | 0/3 |

This small untuned pilot tests masking throughout the pipeline, including pretraining. It does not isolate toggling masking only after a common cross-mask pretrain, prove the proposed neighborhood-jump mechanism, or establish a general result beyond these datasets/seeds. Selected and last checkpoints answer different questions. CPU, one thread, deterministic algorithms; actual runtime versions are in the manifest.

## Interpretation and validation

Iris has worse selected retrieval NLL with masking and slightly better NLL without masking, while classification error stays unchanged. Breast Cancer's NLL increase is smaller without masking, although its masked retrieval error improves. Student Performance's three all-case runs all select warm-up checkpoints, so their zero selected retrieval drift does not establish that later unmasked joint updates were harmless; both last-state conditions deteriorate. Energy Efficiency's selected states improve retrieval MSE with either mask setting, but its masked last state deteriorates. Wine and Yacht improve mean retrieval loss in both conditions, more strongly with masking.

The selected joint checkpoint follows warm-up in 12/18 masked and 13/18 all-case runs. Five of 36 pretrains select epoch 300, the cap. Final adapted mean error for all-case joint is lower than masked joint on 3/6 tasks, higher on 2/6, and tied on 1/6. Retrieval-loss preservation and final prediction quality are distinct findings.

[validate_results.py](validate_results.py) independently recomputes NLL/MSE and prediction errors from stored probabilities/logits/targets, checks all paired pretrained/warm-up states, verifies frozen retrieval invariance and disjoint splits, and checks aggregate drift values. [validation.json](validation.json) records the outcome. Import smoke and two synthetic integrity tests pass. The final T1 plus shared-core test run has 377 passed, 7 failed, and 1 skipped, with the same three missing-baseline-metadata and four exact historical-output failures as earlier pilots.

The new test initially leaked a torch process setting; its cleanup fixture was fixed and affected checks pass. The [archived test](test_at_execution.py) reconstructs the launch-time source fingerprint; the [runner snapshot](runner_at_execution.py) and training implementation are unchanged. The validation record distinguishes execution and current fingerprints. The shared core and historical configurations were not edited.

Configuration: [YAML](../../../../configs/t1/mask_joint_pilot_20261030.yaml). Evidence: [rows](runs.json), [summary JSON](summary.json), [manifest](manifest.json).

```powershell
.venv/Scripts/python.exe tools/t1_mask_joint_pilot.py --config configs/t1/mask_joint_pilot_20261030.yaml --out results/t1/p4_component_sync/mask_joint_rerun
```
