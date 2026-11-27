# Apply-or-skip adaptation experiment

November 25, 2026. The owner shelved the magnitude/direction design and authorized local exploration, including a confidence head that applies either the full residual or no residual. Its earlier code/results remain reproducible; it is not the active candidate.

## Protocol declared before the new test results

Study `ag1`, experiment `gate_v1`, all 14 locally available tasks and seeds 40, 41, 42 from `am1`. Source: each completed **existing-adapter** checkpoint, with its selected retrieval core. No source core or completed adapter run is retrained or overwritten. These tests are exploratory: their test splits were inspected in the preceding study.

The retrieval core stays frozen. Training inputs are stable-ID leave-one-out neighborhoods of training examples. Validation/test queries use the training case base, without their targets entering inputs. A new continuation of the old adapter and the gate is trained together; this is a frozen-retrieval experiment, not a new joint retrieval/adapter benchmark.

Candidates and controls:

1. Original full adapter and its own selected retrieval, unchanged.
2. One global apply/skip decision, chosen on validation error.
3. One global shrink multiplier in `[0, .25, .5, .75, 1]`, chosen on validation error.
4. A disagreement threshold gate: apply full adaptation only above a training-feature quantile of neighborhood target disagreement. Thresholds are training quantiles `[0, .25, .5, .75, 1]`, plus always/never; validation selects among these fixed candidates.
5. Full-adaptation continuation control: additional adapter training with retrieval frozen, with the same budget as the learned gates.
6. Probability gate: a second head learns whether the current full proposal lowers task surrogate loss.
7. Utility gate: the same head and targets, but examples are weighted by the absolute task-loss difference, so large mistakes have more influence than tiny wins.

The learned heads use the old adapter's shared hidden representation, four neighborhood statistics (minimum distance, activation-weighted distance, activation concentration and target disagreement), the retrieval prediction and the full residual proposal. Statistics use log1p; feature mean/std are fitted on initial training inputs only, with std floor .001 and normalized-feature clipping to [-10, 10]. The shared representation/proposal remain differentiable; no targets enter the head. The head is a linear logit; inference applies full correction iff logit > 0 (fixed threshold 0.5 after sigmoid).

For the gate, `benefit = loss(retrieval, y) - loss(full_adaptation, y)`. Loss is squared error for regression and cross-entropy on the same scaled class-score space for both classification alternatives. This is not the historical retrieval NLL: using a common score loss makes the branch comparison consistent. Gate labels/weights are detached. Probability BCE estimates surrogate-loss helpfulness; utility-weighted BCE is a decision score, not a calibrated probability. Classification gate training is a surrogate for accuracy and can disagree with it.

The full residual head retains the original adapter-side objective (L_A) and correction penalty, even for skipped examples, so it does not stop learning when the gate closes. The synchronized retrieval objective (L_R) is not added in this frozen continuation; the no-gate control uses the same objective. Gate BCE is added with coefficient 1; both heads and the trunk are updated by the same optimizer. The hard comparison is used only for decisions, not as a differentiable training path. There is no straight-through estimator, stochastic gate, continuous residual scaling or magnitude/direction normalization in the learned gates.

All continuations start from identical old weights with fresh Adam/AdamW state, learning rate `min(original adapter lr, .001)`, original weight decay, 60 maximum epochs, batch size 64 and patience 12. The original residual/final/size-loss coefficients and selected free radius are retained. The no-gate continuation controls the extra training and optimizer reset. Epoch zero is eligible. Selection minimizes validation classification error then CE, or regression MSE; exact ties keep the earlier epoch. Static-control validation ties prefer less adaptation. No thresholds or hyperparameters are selected from test outcomes.

All new evaluations, curves, weights, cached neighborhood inputs, predictions and RNG/resume states go to recorded run units. Summaries read those units and go under `results/`; the written report goes under `reports/`. The old/new source study is preserved. No remote host or new dataset download is used.

## Methodological motivation

Choosing between retrieval and a full correction is a two-expert decision problem. The cost-sensitive view is motivated by [Mozannar and Sontag, ICML 2020](https://proceedings.mlr.press/v119/mozannar20b.html), which studies learning a predictor and a deferral rule. This implementation is our experiment, not a reproduction or a claim that that paper's guarantees transfer unchanged. Neighborhood evidence was previously motivated by [Adaptive Nearest Neighbor Machine Translation, ACL 2021](https://aclanthology.org/2021.acl-short.47/); its translation results do not establish effectiveness here.

Main questions: does conditional gating beat always-adapt and simple validation controls; does weighting harm severity help; and are improvements larger than those from extra adapter training alone? A hindsight oracle that picks the better test branch will be reported only as an unattainable diagnostic bound, never as a candidate model.
