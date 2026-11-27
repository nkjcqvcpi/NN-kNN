# Learning one disagreement threshold

November 25, 2026. Follow-up to `ag1`, requested by the owner. Protocol written before the new test results. Study `ag2`, experiment `threshold_v1`; 14 local tasks, seeds 40, 41, 42. No remote access or retrieval retraining.

## Question and design

Can fitting one scalar threshold improve on the validation-selected disagreement rule? The rule remains `apply the full residual iff D_y > t`. No new direction, magnitude, confidence features, or learned slope are introduced. Each dataset/seed fits its own scalar; classification and regression use the same mechanism.

Normalize disagreement using training mean/std (std floor 1e-6), initialize the threshold at the training median, and use a sigmoid with fixed temperature 0.25 in normalized units during optimization. Hard inference always applies either the full residual or zero. Thresholds are unconstrained, so always/never adaptation can be represented. Normalization and temperature are fixed buffers, not trainable parameters.

The routing loss is the expected cost of choosing one of the two full branches: `(1-p)*cost_retrieval + p*cost_full`. Its gradient-equivalent implementation subtracts the constant retrieval cost and divides by mean absolute branch-cost difference (floor 1e-8). This is not a loss on an interpolated prediction. Gate costs are detached. Classification uses actual 0/1 mistakes; regression uses squared error. Thus, unlike the earlier learned heads, classification routing directly targets decision error rather than cross-entropy. The differentiable sigmoid permits training the threshold even though the branch costs themselves need no gradient.

## Two arms

- `threshold_frozen`: freeze the completed original adapter and learn only the scalar threshold. Exactly 60 epochs, five full-training-set threshold updates per epoch, Adam at 0.05 without weight decay. Use the final iterate. Validation metrics are recorded but do not select the threshold or stop training. This isolates fitting the cutoff while preserving the old predictions.
- `threshold_joint`: update the scalar threshold alongside the old adapter, with retrieval frozen. Train the adapter first in each epoch, using precisely the original adapter-side objective and the `ag1` continuation budget: fresh optimizer, original weight decay, learning rate capped at .001, batch size 64, 60 epochs maximum, patience 12. Then take five full-training-set threshold steps with the same optimizer/settings as the frozen arm. The adapter remains supervised on every training example, including skipped ones. Validation selects a joint checkpoint with the existing classification-error/CE or regression-MSE criterion; epoch zero is eligible. This arm retains normal validation checkpoint selection, but the threshold receives gradient updates only from training labels.

The gate does not backpropagate into the adapter; its detached branch costs change as the adapter learns. The two components are updated in the same training stage. Retrieval, the old network architecture, and the original full-proposal loss remain unchanged. All normalization and branch-cost supervision use the cached stable-ID leave-one-out training neighborhoods from `ag1`. Query targets never enter inference inputs.

## Comparisons and reporting

Reuse all eight completed methods from `ag1`: retrieval, original full adapter, global on/off, global shrink, validation disagreement gate, extra training without a gate, probability gate, utility gate. Do not rerun them. The primary threshold-isolation comparison is frozen learned threshold versus the validation disagreement rule. The joint arm is additionally compared with the extra-training control and its own gate-forced-open output.

Report per-task three-seed means, equal-task family means and geometric mean error ratios, application rates, useful/harmful changes, learned cutoffs and selected epochs. Retain all tasks and disclose any zero-error exclusions. Verify recorded predictions against metrics and confirm the frozen arm retains the original full proposal. Results remain exploratory on previously examined splits; no new neural baseline or fresh-split confirmation is implied.

The smooth training objective and hard inference can disagree. Training neighborhoods can be optimistic because the source models already saw these training examples. A scalar cutoff cannot represent non-monotone relationships between disagreement and adaptation utility. These are limitations to measure, not reasons to select another protocol after viewing test results.

Implementation: `model/t1/disagreement_threshold.py`; recorded driver: `tools/t1_adaptation_threshold.py --study ag2 --mode run`. Generated comparisons go to `results/adaptation_improvement/ag2`; checkpoints, resumable state, curves and predictions stay under `runs/t1/ag2` only.

Implementation note: inference compares normalized disagreement with the normalized scalar cutoff. The raw cutoff is reported as `mu + sigma*theta`; floating-point cancellation can affect its displayed sign near zero. The normalized comparison is authoritative. This clarifies the implemented rule without changing training or saved predictions.
