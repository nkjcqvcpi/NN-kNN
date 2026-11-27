# Replacing NN-CDH: individual-case adaptation

November 26, 2026. Completed local study `ar1`, experiment `pair_v1`. [Declared protocol](../docs/ADAPTER_PAIRWISE_REDESIGN_EXPERIMENT.md), [complete recorded-results summary](../results/adaptation_improvement/ar1/comparison.json).

**Follow-up completed November 27:** [Merged alternatives-to-NN-CDH report](NNCDH_ALTERNATIVES_REPORT_2026-11-27.md) adds retrieved-only pair training, a direct-tree control, cross-fitted anchor transport, error memory and neural vote attention. This earlier report preserves the original `ar1` results.

**Training scheme clarification:** all new alternatives here use the sequential frozen-retrieval protocol. “Mixed pairs” means retrieved plus random training pairs, not mixed/joint/alternating optimization. The later ExtraTrees/Random Forest comparison uses the same frozen core. No forest-to-retrieval training feedback was tested. [Scheme explanation and audit](NNCDH_ALTERNATIVES_REPORT_2026-11-27.md#training-scheme-sequential-with-retrieval-frozen).

## Finding

**Pairwise ExtraTrees is the strongest replacement tested here.** Keeping NN-kNN retrieval completely frozen, it reaches **4.149% classification error** and **0.320848 regression RMSE**. It improves every regression dataset mean against both retrieval alone and the original sequential NN-CDH with 20% dropout. This is stronger evidence for a regression redesign than our previous changes to NN-CDH inputs, gates or dropout.

Classification is less general: the large gain is concentrated on **Zebra and Zebra Special**. Across the other five classification datasets, pairwise trees are slightly worse than frozen retrieval. The pairwise neural model and local linear correction do not establish an overall replacement.

This is an exploratory result on the same repeatedly examined local splits, not proof of universal superiority. No production/default adapter was replaced. All code is opt-in in the T1 worktree; shared NN-kNN core and Table 1 are unchanged.

## What changed, and what methods this borrows from

The previous NN-CDH first averages the neighborhood and then predicts one correction. The new pair models **transform each individual retrieved case before averaging**. They retain the case's identity, representation and target until the final aggregation.

For regression, train on query-case pairs to predict:

```text
Delta(query, case) ≈ y_query - y_case
y_final = sum_i w_i * (y_case_i + Delta(query, case_i))
        = y_retrieved + sum_i w_i * Delta(query, case_i)
```

Here `w_i` is the query-dependent normalized retrieval activation, including the learned core's case-bias effect. Neither weights nor neighbors are changed. The adapter receives `[z_query, z_case, z_query - z_case]`; target differences are supervision, not prediction inputs.

This construction has direct precedents in [PADRE / Pairwise Difference Regression, Tynes et al. (2021)](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00670), [Twin Neural Network Regression](https://arxiv.org/abs/2012.14873), and its [nearest-neighbor extension](https://arxiv.org/abs/2310.00664). They learn differences and reconstruct predictions using labeled anchors. These are published research precedents; this experiment does not reproduce their full protocols.

For classification, numeric class-ID subtraction would be meaningless. Instead, learn a symmetric **same-class probability** for each query-case pair. An anchor assigns that probability to its own class and distributes the remainder across other classes according to training class frequencies. Average these distributions using NN-kNN weights. This follows [Pairwise Difference Learning for Classification](https://arxiv.org/html/2406.20031v1), with our frozen retrieval-weighted neighborhood replacing its uniform all-anchor aggregation. The output is a valid probability distribution, without a claim of calibration.

| Replacement | Internal model | Training / correction |
| --- | --- | --- |
| **Pairwise trees** | 64 ExtraTrees, depth at most 12; validation selects minimum leaf size 5 or 20 | Regression difference learning; classification same/different learning. Average the two directed predictions for classification; take half their difference for regression. No neural adapter. |
| **Pairwise neural** | New pair MLP, hidden widths 64/32, dropout 0 or 10%; not an NN-CDH subclass | Same pair targets. Regression is antisymmetric by construction; classification is symmetric at evaluation. AdamW, at most 120 epochs, validation checkpointing. |
| **Local ridge** | A small weighted affine regression fitted to each retrieved neighborhood | Regularize slopes; keep the intercept unpenalized. Classification uses one-hot targets with clipping and normalization. Validation selects ridge penalty 0.01, 0.1 or 1.0. No globally trained adapter. |

Local regression is a mature alternative, described in [Locally Weighted Learning](https://publications.ri.cmu.edu/locally-weighted-learning). Our classification extension and small inherited neighborhoods are specific choices here.

The pair models enforce useful reversal behavior, but antisymmetry alone does not enforce consistency around longer cycles. It also does **not** guarantee retrieval dependence: a difference model can learn `f(query) - f(case)`. This study measures predictive and correction behavior, not faithfulness of case explanations.

## Comparison protocol

- Same 14 local datasets, seeds 40–42, recorded train/validation/test splits and pretrained `am1` retrieval checkpoints as before. All retrieval parameters, top-k membership and query-dependent weights are frozen.
- Query/case representations are standardized using training cases only. Training queries exclude themselves by stable case ID. All validation/test anchors are training cases.
- Each training query contributes equal mass: half to its retrieved cases and half to eight fixed random non-self training anchors. The random pairs broaden the training signal; inference still uses only the original retrieved neighborhood. Pairs are correlated observations, not new independent labels.
- Seven candidates per split: three ridge, two trees, two neural. Within each family, validation error selects the candidate, followed by final-prediction loss and candidate ID. **Only the selected candidate is evaluated on test.** Neural checkpoints use validation final loss.
- A separate predeclared policy chooses among retrieval and the three replacement families by validation error; ties prefer retrieval, then ridge, trees, neural. This is a dataset/seed-level choice, not a per-query gate.
- Prior NN-CDH references reuse existing recorded results. The nine-setting sequential and 27-setting all-scheme policies retain the old criterion: validation task loss, then error and fixed tie-breaks. Objectives, parameterizations, candidate counts and compute are **not matched** between method families.

Completed **462 new run units**: 42 frozen contexts, 294 validation candidates and 126 selected-model test evaluations. The candidates comprise 168 trained tree/neural models and 126 local-ridge settings. The comparison table has 420 method/seed rows including reused references and policies. Everything ran locally on CPU; no r760 access or core retraining was used.

## Overall performance

Lower is better. Classification is percentage error. Regression is RMSE in training-standardized target units. Average seeds within each dataset, then weight all seven datasets equally.

| Model / selection policy | Classification error % | Regression RMSE |
| --- | ---: | ---: |
| Frozen retrieval | 11.680 | 0.348690 |
| Original NN-CDH, sequential, no dropout | 10.791 | 0.350429 |
| Original NN-CDH, sequential, 20% dropout | 10.804 | 0.338523 |
| NN-CDH: validation choice among nine sequential settings | 11.024 | 0.411185 |
| NN-CDH: validation choice among all 27 settings | 6.526 | 0.410275 |
| Original NN-CDH, historical mixed schedule | 9.945 | 0.338910 |
| Local ridge | 11.715 | 0.378717 |
| Pairwise neural | 9.793 | 0.357450 |
| **Pairwise trees** | **4.149** | **0.320848** |
| Validation choice: retrieval / ridge / trees / neural | **4.099** | 0.336200 |

The old 9/27-setting regression policies are poor partly because validation selects an extreme Car model. They are **not** the best prior fixed settings, which are also shown or discussed here. Prior regression leaders were the probability gate at 0.337206 and query + prediction, alternating, 10% dropout at 0.337381; pairwise trees improve on both. The prior lowest adapter classification mean was 6.453% from validation-selected dropout within query-without-prediction alternating training. See [earlier consolidated results](ADAPTER_RESULTS_SUMMARY_2026-11-26.md).

The compact local CatBoost reference remains **4.517% / 0.318980**: trees-on-pairs have a slightly lower classification mean and a slightly higher regression mean. This is a descriptive comparison, not equal-budget evidence that pair training beats direct prediction. We have not yet compared pairwise ExtraTrees with a matched standalone ExtraTrees learner. [Conventional baseline protocol/results](QUERY_AWARE_LOCAL_2026-11-25.md).

![Comparison of recorded family means](../results/adaptation_improvement/ar1/comparison.png)

## Where the improvements occur

Classification dataset means, percentage error:

| Dataset | Frozen retrieval | Original sequential, 20% | Best of 27 by validation | Pairwise trees |
| --- | ---: | ---: | ---: | ---: |
| Iris | 7.778 | 7.778 | 4.444 | 6.667 |
| Wine | 3.704 | 3.704 | 7.407 | 3.704 |
| Breast Cancer | 2.339 | 2.047 | 2.339 | 2.339 |
| Balance | 5.867 | 5.333 | 4.267 | 7.200 |
| Digits | 2.222 | 2.222 | 2.222 | 2.315 |
| Zebra | 31.818 | 27.273 | 0.000 | 0.000 |
| Zebra Special | 28.030 | 27.273 | 25.000 | 6.818 |

Against retrieval, pairwise trees win/tie/lose **3/2/2 datasets** and **8/9/4 seed pairs**. Against the all-27 validation policy, they win/tie/lose only **2/2/3 datasets**, despite the lower aggregate mean. Removing Zebra alone leaves 4.840% versus retrieval's 8.323%; removing both Zebra tasks leaves **4.445% versus 4.382%**, or 4.136% for the all-27 policy. These are sensitivity analyses; neither task is excluded from primary results.

Regression dataset means:

| Dataset | Frozen retrieval | Original sequential, 20% | Pairwise trees |
| --- | ---: | ---: | ---: |
| Airfoil | 0.213374 | 0.211313 | **0.204292** |
| Car | 0.226829 | 0.215085 | **0.203991** |
| Yacht | 0.114986 | 0.063068 | **0.059385** |
| Energy Efficiency | 0.052527 | 0.050595 | **0.047008** |
| Student Performance | 0.442019 | 0.433240 | **0.389651** |
| Body Fat | 0.625301 | 0.627923 | **0.582594** |
| Diabetes | 0.765795 | 0.768434 | **0.759010** |

Trees win **7/7 dataset means and 20/21 seed pairs** against retrieval; the exception is Diabetes seed 40. Against original sequential 20%, they win **7/7 means and 17/21 seed pairs**. Mean RMSE falls approximately **7.99%** versus retrieval and **5.22%** versus that regularized NN-CDH baseline. Positive-error geometric ratios are 0.85128 and 0.94261 respectively. The all-27 policy still wins Yacht and Energy Efficiency, so trees do not dominate every prior model on every dataset.

[All task means](../results/adaptation_improvement/ar1/tasks.csv), [paired comparisons, including zero-error exclusions](../results/adaptation_improvement/ar1/comparisons.csv).

## Does the correction behave better?

For regression, wrong direction means the applied correction has the opposite sign to the needed residual. Harmful overshoot means the correct direction but **more than twice** the needed magnitude. Percentages below pool 6,339 test evaluations, including repeated examples across seeds; larger datasets have more influence than in the family means.

| Method | Wrong direction % | Harmful overshoot % | Predictions worsened % | Overshoot harm, sum of squared-error increases |
| --- | ---: | ---: | ---: | ---: |
| Original sequential, 20% dropout | 44.00 | 6.20 | 52.58 | 21.051 |
| Local ridge | 38.27 | 3.20 | 41.21 | 3.793 |
| Pairwise neural | 39.61 | 11.36 | 53.32 | 75.934 |
| **Pairwise trees** | **36.44** | **6.33** | **44.88** | **6.882** |

Trees reduce wrong-direction frequency and harmful-update frequency against the matched NN-CDH baseline. Overshoot frequency is slightly higher, but its damage is much smaller. Their total positive squared-error increases are **55.193**, versus **127.319** in reductions; the matched NN-CDH baseline has 74.541 harm versus 99.980 benefit. Trees still worsen roughly 45% of predictions: this is a better adapter, not a reliable per-query oracle.

Local ridge illustrates why counts alone mislead. Although it harms fewer examples, one wrong-direction Car correction is enormous. At Car seed 40, test index 1128, retrieval needs **−0.579**. Local ridge applies **+17.163**; pairwise neural applies **+11.827**; original sequential 20% applies **+3.693**. The pairwise-neural failure accounts for 79.89% of that run's final squared error. The larger NN-CDH selection policy chooses the previous **+60.490** failure on the same query.

Trees' largest harmful example is elsewhere: Body Fat seed 41, index 24, needs **−0.180** and receives **−2.006**. Its squared-error increase is 3.302. Thus trees can still overcorrect substantially.

A plausible contributor to their stability is that regression tree leaves average observed training differences. They do not extrapolate an unbounded linear/ReLU output as query features grow. This does not make the correction small relative to a particular query's required residual, and we did not isolate this mechanism in a separate ablation.

Classification pools 2,193 evaluations. Against the **same frozen retrieval**, trees fix **54** errors and introduce **10** harmful flips; original sequential 20% fixes **22** and introduces **15**. Pairwise neural fixes 45 but introduces 32; local ridge fixes 14 and introduces 24. Changes in probability margins are less comparable across the new probability outputs and old residual scores, so these decision counts are the primary classification behavior evidence.

[Per-run behavior](../results/adaptation_improvement/ar1/behavior.csv), [pooled counts and severity](../results/adaptation_improvement/ar1/behavior_totals.csv), [largest harmful cases](../results/adaptation_improvement/ar1/harm_examples.csv).

## Validation selection still has limits

Tree leaf size 5 is selected in 19/21 classification runs; regression splits nearly evenly between leaf sizes 5 and 20. The broader replacement-selection policy chooses trees/neural/retrieval/ridge in **9/4/7/1 classification runs** and **11/7/3/0 regression runs**.

That policy improves classification slightly, to 4.099%, but worsens regression versus always using the validation-tuned tree family, to 0.336200. It selects pairwise neural on all three Car splits, including the harmful seed-40 case, because neural validation RMSE is lower. Enlarging the choice set does not guarantee better test generalization. We did not override these selections after seeing test failures. [All selections](../results/adaptation_improvement/ar1/selections.csv), [counts](../results/adaptation_improvement/ar1/selection_counts.csv), [candidate validation results](../results/adaptation_improvement/ar1/candidates.csv).

## Recommendation and remaining alternatives

**Make pairwise trees the leading redesign candidate for regression.** Preserve the original sequential dropout adapter as a comparison and confirm on fresh splits/tasks before adopting a default. Classification should retain both existing neural adapters and pairwise trees; the present gain is too concentrated to recommend one universally.

The next useful controls are a same-budget direct ExtraTrees model and pairwise trees trained on retrieved-only pairs. Those would separate the contribution of the tree learner, individual-case adaptation and extra random training pairs. Pairwise boosting is another candidate, but has not been run here.

For an eventual differentiable redesign, a neighborhood model that transforms individual case features/labels before aggregation is also worth considering. [TabR, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4ef594af0d9a519db8fb292452c461fa-Abstract-Conference.html) is a relevant retrieval-based reference. Full TabR, attention-based adapters, cycle-consistency losses, and joint/alternating retraining of these new models are **not tested here**. ExtraTrees is not differentiable through retrieval; the current implementation is intentionally sequential.

## Reproduction and verification

Implementation: [pair prediction models](../model/t1/pairwise_adaptation.py), [recorded/resumable fitting](../model/t1/pairwise_training.py), [experiment driver](../tools/t1_pairwise_redesign.py), [read-only summary](../tools/t1_pairwise_redesign_summary.py), [tests](../tests/test_t1_pairwise_redesign.py), [local uv launcher](../../run_pairwise_redesign_experiment.py).

From the mini-project's `code/` directory with the configured uv environment:

```text
uv run --project ../.. --no-sync python run_pairwise_redesign_experiment.py --study ar1 --workers 6
uv run --project ../.. --no-sync python run_pairwise_redesign_experiment.py --study ar1 --mode summarize
```

The launcher sets the established local data, environment and run roots. Run states, training curves and predictions remain under `NNKNN_RUNS/t1/ar1/pair_v1/`; generated files are under this worktree's `results/adaptation_improvement/ar1/`. Runtime logs are in the mini-project's `logs/ar1/`.

Verification passed: **17 focused tests**, including target isolation, self-exclusion, pair reversal, valid probabilities, local affine recovery, validation-only selection, and exact interrupted recovery for neural/trees in both tasks. Reconciled **294 validation prediction sets**, **126 selected test sets**, **210 reused NN-CDH test sets**, **42 contexts** and **126 frozen hashes**. An independent audit checked 200 new/retrieval/policy error values, 140 task means, 20 family means, 40 paired comparisons and 42 policy selections. All **462 complete units were read/reused with fitting and prediction disabled**. Both shared-core manifests and whitespace checks passed. The generated comparison figure was inspected.

The small constant-target numerical test initially needed a 1e-12 comparison tolerance; no fitted algorithm or candidate setting changed in response to test results. All experimental failures remain in the reported results. [Reuse audit](../../../logs/ar1_reuse_check.log), [independent audit](../../../logs/ar1_independent_audit.log).
