# Replacing NN-CDH with individual-case adaptation

November 26, 2026. Study `ar1`, experiment `pair_v1`. Declared before new candidate validation/test results. Local CPU only, using the 14 available tasks and seeds 40-42.

## Question and references

Can individual query-case adaptation improve on the aggregate-input NN-CDH while keeping retrieval exactly fixed?

The design alternatives are informed by:

- [Pairwise Difference Regression / PADRE, Tynes et al., 2021](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00670): learn target differences from pairs, then anchor a query estimate at known targets.
- [Twin Neural Network Regression, Wetzel et al., 2022](https://arxiv.org/abs/2012.14873) and [nearest-neighbor extension](https://arxiv.org/abs/2310.00664): learn differences and average predictions anchored at cases.
- [Pairwise Difference Learning for Classification, Belaid et al., 2024](https://arxiv.org/html/2406.20031v1): learn a symmetric same-class probability; each anchor supplies a class distribution.
- [Locally Weighted Learning, Atkeson, Moore and Schaal, 1997](https://publications.ri.cmu.edu/locally-weighted-learning): fit a local function using weighted training cases.
- [TabR, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4ef594af0d9a519db8fb292452c461fa-Abstract-Conference.html): a further retrieval-based design with case-specific transformations. Full TabR is a future comparison, not reproduced here.

These are precedents, not guarantees for this integration. Local linear regression is a long-established method; the particular pairwise adaptations are research methods with published empirical support.

## Shared context and pair construction

Reuse each `am1` pretrained retrieval model and `aq1` data split. Freeze all retrieval parameters and modes. Keep its exact top-k membership and normalized activations; do not expand the inference neighborhood or retune temperature. Use its learned query/case representations, standardized with mean/std fitted on training-case representations only. Scale below 1e-6 is replaced by one.

Training-scheme clarification (November 27): this is the **sequential frozen-retrieval** comparison. The pair neural and tree models are fitted after retrieval pretraining; local ridge fits at inference on that same fixed neighborhood. No new alternative here is jointly or alternately trained with retrieval. “Mixed pairs” describes retrieved/random pair construction, not a training schedule. See the [cross-study scheme explanation](../reports/NNCDH_ALTERNATIVES_REPORT_2026-11-27.md#training-scheme-sequential-with-retrieval-frozen).

Every training query excludes itself by stable case ID. Validation and test queries use training cases only. All anchors and their labels are from the training split.

For each training query, the pair-learning distribution gives 50% mass to its retrieved cases (using their retrieval weights) and 50% to eight uniformly sampled non-self training anchors. Random anchors are fixed from the split seed. Each query has equal total weight. Pairs are correlated observations, not additional independent labels. Trees see both orientations with half weight each; the neural model symmetrizes its forward calculation.

Pair features are `[z_query, z_case, z_query - z_case]`. The true query target is a supervision signal only. Neither query targets nor validation/test labels enter the prediction inputs.

## Three replacement families

### Local ridge

For each query, fit a weighted affine function to its unchanged neighborhood. Center cases and their targets at their retrieval-weighted means, fit slope B with ridge penalty, then predict `retrieval + (query - mean_case) B`. Regularize slopes, not the intercept. Use a dual solve over the small neighborhood, with lambda in **0.01, 0.1, 1.0**.

Regression uses scalar targets. Classification fits one-hot targets, clips negative class scores to zero and renormalizes. This classification extension is our baseline, not a reproduction of a local logistic model. The fit cannot infer a missing class from a homogeneous-label neighborhood. It uses no neural adapter or globally trained residual head.

### Pairwise ExtraTrees

Use 64 trees, maximum depth 12, max_features=0.7, one CPU thread; compare min_samples_leaf **5 and 20**. Train with the pair weights above, including reversed pairs. Regression predicts target difference; inference antisymmetrizes the two directed predictions. Classification learns binary same/different labels and averages the two directed same-class probabilities. No class rebalancing is applied.

Tree fitting proceeds in 16-tree increments with resumable state. All candidates use the final 64-tree ensemble, with no validation-selected tree count.

### Pairwise neural model

A new shared pair network, not an NN-CDH subclass: Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear, widths **64/32**. Compare dropout **0 and 0.1**. AdamW, lr .001, weight decay 1e-4, gradient norm cap 5, batch size 512, maximum 120 epochs, patience 20. Epochs visit the fixed weighted training pairs. Checkpoints minimize validation final-prediction loss, with error as tie-breaker.

Regression returns `Delta(q,c) = (h(q,c) - h(c,q))/2`, trained against `y_q-y_c` with weighted MSE. Classification averages the two sigmoid probabilities and trains weighted binary cross-entropy on same/different labels. At evaluation, pair reversal negates regression differences and preserves classification affinities. A regression self-pair has exactly zero correction. Antisymmetry does not enforce consistency around longer cycles.

### Final prediction

Regression uses `y_final = sum_i w_i [y_i + Delta(q,i)]`, implemented as the original retrieval prediction plus the weighted differences.

For classification, an anchor with label c and predicted same-class probability s gives probability s to c; it distributes 1-s among other classes in proportion to training class frequencies. Aggregate these distributions using the unchanged retrieval weights. This follows the PDL classification construction but replaces uniform all-case averaging with NN-kNN retrieval weights. Probabilities are valid by construction; they are not claimed to be calibrated confidence.

No additional learned gate, residual multiplier or test-derived clipping is introduced.

## Selection and comparisons

There are seven candidates per split: three local-ridge, two tree, two neural. Record candidate training curves/states and validation predictions; **do not evaluate unselected candidates on test data**. Choose within each family by validation error, then final-prediction loss (NLL for classification, MSE for regression), then candidate ID. There are **294 candidate units**, **126 selected-model test units**, and **42 shared context units**.

Also report a predeclared choice among retrieval-only and the three selected replacement families using validation error; exact ties prefer retrieval, local ridge, trees, neural, in that order. This is a dataset/seed-level choice, not a per-query gate.

Reuse existing NN-CDH results: original sequential without dropout; original sequential at 20%; validation selection among all three sequential adapters and dropout rates; and a broader selection among all 27 existing adapter/scheme/rate settings. For these NN-CDH selection policies retain their historical criterion, validation task loss then error, smaller dropout, adapter/scheme tie order. Report the different objectives/search spaces; do not claim equal compute.

Primary inference compares replacements with the unchanged retrieval and frozen-retrieval NN-CDH references. Broader full-model references are context: their retrieval may have changed. All aggregate means weight tasks equally after seed averaging. Report task/seed wins/ties/losses, positive-error geometric ratios with zero exclusions, class fixes/harmful flips, regression direction/overshoot frequency and severity, and largest harmful cases. Preserve every failure.

## Recording and verification

Require --study and store every context, candidate, selected evaluation, state and prediction under runs/t1/ar1/pair_v1/. No complete unit is fitted again. Resume neural epochs with optimizer/shuffle/dropout RNG; resume trees from their fitted ensemble. Generated summaries read only run units and go to results/adaptation_improvement/ar1/. Written reports go to reports/.

Verify label isolation, positive-weight self-exclusion, source retrieval reconstruction, frozen hashes, pair weights, swap/identity properties, valid class probabilities, local affine fits, weighted delta aggregation, exact resume, validation-only selection and completed-unit reuse. Do not change shared core, Table 1 or previous experimental defaults.

Completed outcome: [pairwise redesign report](../reports/ADAPTER_PAIRWISE_REDESIGN_LOCAL_2026-11-26.md). All predeclared candidates and dataset/seed cells are retained.
