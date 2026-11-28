# Alternatives to NN-CDH: merged experiments and results

November 27, 2026. Completed studies `ar1`–`ar4`, plus recorded NN-CDH references. Latest addition: alternating retrieval and forest fitting for ExtraTrees/Random Forest with mixed/retrieved-only pairs. [1: merged numerical evidence](../results/adaptation_improvement/ar4/comparison.json), [alternating protocol](../docs/ADAPTER_FOREST_ALTERNATING_EXPERIMENT.md), [2: creative-alternatives protocol](../docs/NNCDH_ALTERNATIVES_FOLLOWUP_EXPERIMENT.md), [3: first alternatives report](ADAPTER_PAIRWISE_REDESIGN_LOCAL_2026-11-26.md).

## Executive summary

**Alternating training works technically, but gives limited and inconsistent improvements.** Mixed-pair Random Forest improves classification error from **3.636% to 3.191%**, now the lowest individual-family mean. Retrieved-only Random Forest improves regression RMSE from **0.314232 to 0.313608**, only **0.20% lower**, with five dataset wins, one tie and one loss. The other three regression variants are slightly worse on average. This does not justify adopting alternating training as a general default.

**The broader result is that retrieved-only pair adaptation helps regression substantially.** Random Forest's **0.314232** and ExtraTrees' **0.314960** are below frozen retrieval's **0.348690** and original sequential NN-CDH with 20% dropout's **0.338523**. Mixed-pair Random Forest gives **3.636% / 0.321912** versus mixed ExtraTrees' **4.149% / 0.320848**: it improves classification but slightly worsens regression. Changing the tree learner is not an across-the-board improvement.

**The direct-tree control supports retaining case-based adaptation under these settings.** A direct ExtraTrees head on the same frozen representation reaches **7.365% classification error / 0.339633 RMSE**; retrieved-only pair trees reach **4.350% / 0.314960**. The pair approach wins or ties every classification dataset mean and improves six of seven regression means. This is stronger evidence than simply comparing trees with neural adapters, although training examples, objectives and effective capacity still differ.

**The three creative designs have mixed value.** Cross-fitted anchor transport is useful (**7.717% / 0.328438**) but trails pairwise trees. Retrieval-error memory makes smaller gains (**9.238% / 0.342571**). Neural vote attention prevents predictions outside the retrieved target range, yet achieves only **12.073% / 0.346362**. Making corrections conservative does not make them accurate.

Classification gains over retrieval remain concentrated on Zebra and Zebra Special. In `ar3`, retrieved-only Random Forest improves over ExtraTrees on Iris and Zebra Special, tying the other five dataset means. The new mixed-RF alternating gain also comes from Iris and Zebra Special, while Balance worsens. The evidence for pair adaptation is broader than the evidence favoring a particular forest training schedule. These are exploratory results on repeatedly examined local splits; the original default remains unchanged.

## What was tested

The shared unit is a dataset/seed split: seven classification and seven regression datasets, seeds **40, 41, 42**. All four studies use identical saved train/validation/test splits and start from the same pretrained NN-kNN core states; `ar4` subsequently updates retrieval. Query/case representations are scaled using training cases only. Training retrieval excludes the query's own case; all validation/test anchors come from training.

`ar1` tested three replacements. `ar2` added three designs and two controls. `ar3` added two Random Forest pair families. Together these frozen-core studies contain **1,050 validation candidate units, 420 selected-family test units and 42 shared context units: 1,512 recorded units**. `ar4` adds **840 retrieval blocks, 840 forest refits and 168 selected test units: 1,848 new units**. All four studies now contain **3,360 units**. The merged comparison has **966 rows**, covering 23 methods/policies and 42 splits; all **798 preceding rows are unchanged**. New work was local CPU only. Completed units were reused; alternating refits are new models on updated retrieval contexts.

Classification numbers below are **percentage error**. Regression is **RMSE in training-standardized target units**. Lower is better. Family means average seeds within each dataset, then weight the seven datasets equally. Pooled behavior counts later in this report use a different weighting.

## Training scheme: sequential, with retrieval frozen

**The `ar1`/`ar2`/`ar3` results are sequential frozen-retrieval baselines. The `ar4` rows explicitly add alternating training.** In the sequential studies, reuse the recorded pretrained NN-kNN core, then fit or construct adaptation from its saved training contexts. Its representation, distance parameters, case biases, case store, neighborhood membership and base normalized activations are fixed throughout adaptation. The adaptation loss does not update retrieval in those baseline rows.

| Component / comparison | Training scheme actually tested | Can adaptation training change the NN-kNN core? |
| --- | --- | --- |
| ExtraTrees / Random Forest pair adapters; direct-tree and transport controls | Sequential, using frozen saved representations and neighborhoods | No |
| Pair neural, error memory and vote attention in the redesign studies | Same sequential frozen-core protocol | No; attention changes only the adapter's final voting weights |
| Local ridge | Local fit at inference on the fixed neighborhood; recorded under the same sequential comparison protocol | No |
| Four forest continuations in `ar4` | Five cycles: two retrieval epochs, then a forest refit; validation can retain cycle zero | Yes, through final-prediction loss and retrieval loss; hard tree decisions are detached |
| Earlier NN-CDH original / query-aware / query-without-prediction studies | Sequential, alternating and joint were separately tested in `as1` / `ad1` | Yes in alternating/joint, according to their retrieval objective; no in sequential |

**“Mixed pairs” is a pair-sampling recipe, not a mixed training schedule.** It means retrieved pairs plus random training-case pairs. Both mixed-pair and retrieved-only forests here are sequential. The historical NN-CDH “mixed schedule” reference is different: it uses the earlier joint runs except for sequential Diabetes.

The read-only scheme audit checks **420 selected-family runs across 42 contexts**: all record `scheme="sequential"`, all initial/selected core hashes match the shared source, and all stored retrieval predictions match exactly. The existing result rows already contain the correct scheme. This clarification adds evidence tables without changing scores or run configurations. [Training schemes](../results/adaptation_improvement/ar3/training_schemes.csv), [per-run audit](../results/adaptation_improvement/ar3/training_scheme_runs.csv), [audit details](../results/adaptation_improvement/ar3/training_scheme_audit.json).

### Could a forest influence retrieval in joint or alternating training?

**Yes, with a deliberately designed feedback path; it does not happen automatically.** The current forest implementation calls scikit-learn on saved NumPy arrays. It is outside PyTorch autograd. Ordinary tree predictions are constant inside each leaf region, so the hard split decisions do not provide the useful input gradients that a neural adapter supplies. The existing NN-CDH joint/alternating optimizer cannot train these forests as a drop-in neural module. [Decision-tree formulation](https://scikit-learn.org/stable/modules/tree.html#mathematical-formulation).

That does not block every possible gradient path. **The new `ar4` implementation uses this path:** hold each forest correction `delta_i` fixed within a forward/backward computation and compute aggregation in PyTorch:

```text
regression:      final = p_retrieval(theta) + sum_i w_i(theta) * stop_gradient(delta_i)
classification: final = sum_i w_i(theta) * stop_gradient(adapted_class_distribution_i)
```

The final prediction loss updates retrieval parameters `theta` through its prediction and/or activation weights. The intended effect is to favor cases whose adapted predictions are useful. It does not differentiate through forest splits or discrete neighbor selection. Updating retrieval only by its original retrieval loss would not provide this adaptation-loss feedback.

`ar4` now tests this alternating procedure: reuse the sequential forest, freeze its fitted trees while updating retrieval, then refresh representations/neighborhoods/pairs and refit. Forest outputs are recomputed for each minibatch; they are detached during backpropagation. Refreshing matters because changing the representation changes the forest's input distribution. All 840 retrieval blocks changed the core state, and the first-minibatch final-loss gradient probe was nonzero in 1,648 of 1,680 epochs. True end-to-end differentiation through the tree path remains unimplemented. [Measured alternating results](#alternating-forests-matched-continuations-ar4), [implementation](../model/t1/forest_alternating.py).

### Individual-case difference learning

The main pivot away from NN-CDH is to keep cases separate until after adaptation:

```text
Delta(query, case) ≈ y_query - y_case
y_final = sum_i w_i * [y_case_i + Delta(query, case_i)]
```

`w_i` is the original query-dependent normalized activation, including the learned core's case-bias effect. Regression learns differences from `[query representation, case representation, their difference]`. At inference, reversing the pair negates its predicted difference. This is informed by [4: PADRE / Pairwise Difference Regression](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00670) and [5: Twin Neural Network Regression](https://arxiv.org/abs/2012.14873), including its [nearest-neighbor extension](https://arxiv.org/abs/2310.00664).

For classification, the pair model learns a symmetric same-class probability. Each anchor assigns that probability to its own class and distributes the remainder over other classes using training frequencies. The final distribution averages anchor distributions with the original activations. This adapts [6: Pairwise Difference Learning for Classification](https://arxiv.org/html/2406.20031v1) to NN-kNN neighborhoods; it is not numeric subtraction of class IDs.

| Design | How it works | Difference from the previous NN-CDH |
| --- | --- | --- |
| Pair trees, mixed training (`ar1`) | ExtraTrees learns individual pair targets; each query puts half its training mass on retrieved cases and half on eight fixed random non-self cases | Tree learner and individual anchors replace the aggregate-input residual MLP |
| **Pair trees, retrieved-only (`ar2`)** | Same prediction architecture, trained only on actual retrieved non-self pairs with equal total mass per query | Tests whether adaptation should specialize in the pairs encountered at inference |
| Pair Random Forest, mixed / retrieved-only (`ar3`) | Replace ExtraTrees with ordinary Random Forest in each of the preceding pair recipes | Tests the tree learner while preserving pair features, targets, weights and aggregation |
| Pairwise neural (`ar1`) | Separate 64/32 pair MLP; antisymmetric regression differences or symmetric classification affinity | Neural individual-case prediction before averaging, rather than one aggregate correction |
| Local ridge (`ar1`) | Weighted affine fit within the retrieved neighborhood, with regularized slopes and unpenalized intercept | Fits a local function at inference; no global neural adapter |

Local fitting follows the general idea of [7: Locally Weighted Learning](https://publications.ri.cmu.edu/locally-weighted-learning). Our one-hot classification extension is a specific experimental choice. All these implementations are integrations inspired by the cited methods, not full paper reproductions.

### Three creative follow-ups

**Cross-fitted anchor transport.** Divide training cases into five folds and fit a direct ExtraTrees head on the other four folds. For anchor `i`, use the head that excluded it:

```text
adapted_case_i = y_i + f_without_fold_i(query) - f_without_fold_i(case_i)
final = sum_i w_i * adapted_case_i
```

This combines the head's predicted query–case difference with the anchor's observed target. Classification transports probability vectors and one-hot labels, then clips negative final scores and normalizes. The folds apply to these new heads only: the shared frozen representation was trained on the full original training set. This is our experimental combination of anchor transport and out-of-fold estimation, not a named standard algorithm or a fully cross-fitted NN-kNN pipeline. Different anchors can use different heads, so global cycle consistency is not guaranteed.

**Retrieval-error memory.** Store each training query's leave-one-out retrieval residual. Describe a retrieval situation by its query representation, retrieved prediction and four statistics: nearest and activation-weighted mean Euclidean distance in the saved standardized representation, activation concentration, and target disagreement. Find similar training situations and transfer their inverse-distance-weighted residuals. Validation selects 8 or 32 memories and a multiplier of 0.25 or 1. Regression transfers scalar residuals; classification transfers one-hot-minus-probability residual vectors. This is a second memory of past retrieval mistakes, without a new neural predictor. The retrieval representation itself is not cross-fitted.

**Neural vote attention.** A shared 64/32 MLP scores each case using `[query, case, query-case, case_target-retrieved_prediction]`. Add `4*tanh(score)` to the log of each positive base activation, then normalize and average the actual case targets. The last layer starts at zero; zero base weights stay zero. The core is frozen, but the adapter changes the weights used in the final vote. A regression prediction must stay between the smallest and largest retrieved targets; classification cannot assign probability to a class missing from the neighborhood. This borrows support-set attention and permutation-invariant aggregation ideas from [8: Matching Networks](https://arxiv.org/abs/1606.04080) and [9: Deep Sets](https://arxiv.org/abs/1703.06114), without reproducing either complete model.

### Direct prediction control

The direct ExtraTrees head predicts from the **same frozen representation**, ignoring retrieved labels, activations and predictions. All tree families use 64 trees, depth at most 12 and max_features 0.7. Pair trees compare leaf sizes 5/20; direct and transport heads compare 1/5/20. Leaf size 1 avoids forcing the smaller direct training set to use only smoothing levels intended for a much larger pair dataset. Transport uses five forests per candidate.

This control tests whether simply switching to trees explains the gains. It is not a raw-feature, from-scratch baseline. Pair construction, sample counts, feature dimensions, effective tree capacity and training objectives differ, so the result does not isolate one causal ingredient.

### Random Forest on the same pair formulations

Study `ar3` reuses the exact pair-building and prediction functions from `ar1` and `ar2`. Both learners see `[query, case, query-case]`; regression targets are `y_query-y_case`, and classification targets indicate whether the pair shares a class. Both train on both pair orientations. The saved NN-kNN representation, neighbors, query-dependent activations, train/validation/test membership, class priors, candidate seed, 64-tree budget, depth 12, max_features 0.7 and leaf-size search 5/20 are matched. [10: Random Forest protocol](../docs/ADAPTER_RANDOM_FOREST_EXPERIMENT.md).

The new estimator is scikit-learn **RandomForestClassifier / RandomForestRegressor**, using best-threshold splits and **bootstrap=True**. The prior ExtraTrees learners use randomized candidate thresholds and **bootstrap=False**. Thus this compares the two standard recipes, including bootstrap sampling; it does not isolate threshold randomization alone. Bootstrap samples pair rows, rather than query groups. The installed scikit-learn version is **1.9.1**. Same tree counts and tuning grids do not imply equal training time. [11: Random Forest documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html), [forest audit](../results/adaptation_improvement/ar3/forest_audit.json).

Validation selects leaf size independently for each learner and pair recipe. Only that winner is tested. An expanded validation policy adds the two RF families after the existing methods in the tie order; the original `ar2` policy is retained separately. No thresholds or hyperparameters were changed after examining test results.

## All merged performance results

Candidate selection is validation-only: error, then final loss, then candidate ID; alternating checkpoints use earlier cycle as the final tie-breaker. Only each family's winner reaches test. Neural checkpoints minimize validation final loss; attention and alternating forests also allow their initial checkpoint. The old NN-CDH policies retain their original validation-loss-first rule. No test-based overrides, clipping thresholds or exclusions were introduced. [1: family means](../results/adaptation_improvement/ar4/families.csv), [candidate results](../results/adaptation_improvement/ar4/candidates.csv).

| Model / policy | Classification error % | Regression RMSE |
| --- | ---: | ---: |
| Frozen retrieval | 11.680 | 0.348690 |
| Original NN-CDH, sequential, no dropout | 10.791 | 0.350429 |
| Original NN-CDH, sequential, 20% dropout | 10.804 | 0.338523 |
| NN-CDH: validation choice of nine sequential settings | 11.024 | 0.411185 |
| NN-CDH: validation choice of all 27 settings | 6.526 | 0.410275 |
| Original NN-CDH, historical mixed schedule | 9.945 | 0.338910 |
| Local ridge | 11.715 | 0.378717 |
| Pairwise neural | 9.793 | 0.357450 |
| ExtraTrees, mixed training | 4.149 | 0.320848 |
| Random Forest, mixed training | 3.636 | 0.321912 |
| ExtraTrees, retrieved-only training | 4.350 | 0.314960 |
| Random Forest, retrieved-only training | 3.325 | 0.314232 |
| ExtraTrees, mixed pairs, alternating | 4.132 | 0.321036 |
| ExtraTrees, retrieved-only pairs, alternating | 4.516 | 0.315106 |
| **Random Forest, mixed pairs, alternating** | **3.191** | 0.322119 |
| **Random Forest, retrieved-only pairs, alternating** | 3.255 | **0.313608** |
| Direct ExtraTrees head, control | 7.365 | 0.339633 |
| Cross-fitted anchor transport | 7.717 | 0.328438 |
| Retrieval-error memory | 9.238 | 0.342571 |
| Neural vote attention | 12.073 | 0.346362 |
| Validation choice: retrieval + first three alternatives | 4.099 | 0.336200 |
| Validation choice: retrieval + seven alternatives (`ar2`) | 4.019 | 0.328706 |
| Validation choice: retrieval + nine alternatives, including RF | 3.586 | 0.317036 |

The 9/27-setting NN-CDH policies are not its strongest fixed regression baselines: they sometimes select a severe Car failure. The earlier probability gate reaches **0.337206**, and query + prediction with alternating training and 10% dropout reaches **0.337381**. Retrieved-only pair trees also improve on those means. Prior mixed/joint NN-CDH references and `ar4` can change retrieval; sequential references and `ar1`/`ar2`/`ar3` use the common frozen core. The old cross-family validation policies are preserved; no policy selecting across the four alternating variants was added.

![Merged alternatives comparison](../results/adaptation_improvement/ar3/alternatives.png)

The earlier compact local CatBoost result was **4.517% / 0.318980**, versus retrieved-only ExtraTrees' **4.350% / 0.314960** and retrieved-only Random Forest's **3.325% / 0.314232**. Both pair learners are lower on both means in this local comparison. This does not establish superiority to a tuned CatBoost benchmark: search budgets, learned representations and historical retrieval pretraining differ. MLP/TabM references and their limits remain in the [conventional baseline report](QUERY_AWARE_LOCAL_2026-11-25.md).

## Alternating forests: matched continuations (`ar4`)

All four requested variants are complete on all 42 dataset/seed splits. Each starts with its own recorded sequential winning forest and fixed leaf size. Run five cycles of two retrieval epochs plus a fresh 64-tree fit. Retrieval continues the pretrained Adam moments at one tenth of its original learning rates, optimizing `L_retrieval + L_final` and the existing case-bias penalty. The forest still learns the original pair target. Cases and labels stay fixed; neighbors, activations and learned representations can change. Standardization is fixed within each retrieval block and recomputed from training cases before refitting. The random-anchor IDs and forest seed stay fixed. [Predeclared protocol](../docs/ADAPTER_FOREST_ALTERNATING_EXPERIMENT.md).

Validation chooses cycle 0–5 independently within each family. Cycle zero reuses the previous sequential prediction exactly. Test data is evaluated only for the selected checkpoint. The comparison has extra training and checkpoint-selection opportunities; it is not an equal-compute or fresh-split confirmation.

| Forest / pairs | Classification error %, sequential → alternating | Regression RMSE, sequential → alternating | Classification dataset W/T/L | Regression dataset W/T/L |
| --- | ---: | ---: | ---: | ---: |
| ExtraTrees / mixed | 4.149 → 4.132 | 0.320848 → 0.321036 | 2 / 4 / 1 | 2 / 2 / 3 |
| ExtraTrees / retrieved-only | 4.350 → 4.516 | 0.314960 → 0.315106 | 1 / 4 / 2 | 4 / 0 / 3 |
| Random Forest / mixed | 3.636 → **3.191** | 0.321912 → 0.322119 | 2 / 4 / 1 | 3 / 1 / 3 |
| Random Forest / retrieved-only | 3.325 → 3.255 | 0.314232 → **0.313608** | 1 / 5 / 1 | 5 / 1 / 1 |

W/T/L compares alternating with its own matching sequential forest over seven dataset means. [Paired comparisons](../results/adaptation_improvement/ar4/comparisons.csv), [independent audit and seed rows](../results/adaptation_improvement/ar4/alternating_audit.json).

![Matched sequential and alternating forests](../results/adaptation_improvement/ar4/alternatives.png)

**Classification improves narrowly.** Mixed RF's 0.445-point gain comes from Iris (6.667% → 5.556%) and Zebra Special (3.788% → 1.515%); Balance worsens (6.933% → 7.200%), and four means tie. Across seeds it wins three, ties 17 and loses one. Excluding both Zebra tasks, mixed RF improves from 4.333% to 4.164%, a smaller 0.169-point gain. Retrieved-only RF improves only on Zebra Special and worsens on Balance. Three of four classification means improve, but most seed-level accuracy results are unchanged. [Dataset means](../results/adaptation_improvement/ar4/tasks.csv), [sensitivity analysis](../results/adaptation_improvement/ar4/classification_sensitivity.csv).

**Regression gains are very small.** Retrieved-only RF is the only family whose mean improves: 0.000624 RMSE, or 0.20%. It wins ten seed comparisons, ties eight and loses three. Dataset means improve for Airfoil, Car, Energy Efficiency, Student Performance and Yacht; Body Fat ties and Diabetes worsens. For example, Energy Efficiency changes 0.045492 → 0.043941 and Yacht 0.052150 → 0.050567, while Diabetes changes 0.763927 → 0.766339. The other three aggregate RMSEs worsen by less than 0.00021.

**Validation often rejects the continuation.** Cycle zero is retained in 15/84 classification and 37/84 regression models: **52/168 total**. At cycle five, validation error is worse than the initial model in **68/84 regression runs**. This does not measure their cycle-five test performance; those unselected checkpoints were not tested. Selecting an early cycle or the original forest matters. [All cycle records](../results/adaptation_improvement/ar4/alternating_cycles.csv), [selection counts](../results/adaptation_improvement/ar4/selection_counts.csv).

**Retrieval itself usually does not explain the favorable means.** All four selected classification retrieval-only means worsen from the starting 11.680%. For mixed RF, retrieval error rises to 12.094% while final error falls to 3.191%. For retrieved-only RF regression, retrieval RMSE rises from 0.348690 to 0.349630 while final RMSE falls to 0.313608. Selected test neighborhoods change on an equal-dataset average of 14.6% of queries for mixed-RF classification and 16.4% for retrieved-only-RF regression, including runs that retained cycle zero. The core is changing, but retrieval-only and adapted performance can trade off. [Retrieval diagnostics](../results/adaptation_improvement/ar4/alternating_diagnostics.csv).

**The correction-reliability problem remains.** The following counts are relative to each selected model's own retrieval prediction, over the same 6,339 regression evaluations:

| Alternating forest / pairs | Wrong direction % | Harmful overshoot % | Prediction worsened % | Total harmful squared-error increase |
| --- | ---: | ---: | ---: | ---: |
| ExtraTrees / mixed | 36.17 | 6.28 | 44.87 | 55.458 |
| ExtraTrees / retrieved-only | 33.73 | 7.07 | 43.26 | 49.505 |
| Random Forest / mixed | 35.62 | 7.05 | 44.69 | 72.133 |
| Random Forest / retrieved-only | 33.52 | 7.73 | 43.62 | 64.398 |

For retrieved-only RF, sequential → alternating changes wrong-direction frequency **33.38% → 33.52%**, harmful overshoot **7.70% → 7.73%**, and worsened predictions **43.37% → 43.62%**. Total harm decreases **67.291 → 64.398**, so severity improves somewhat while frequency does not. Mixed-RF classification changes fixes/harmful flips from **65/15 to 70/11**, but its own retrieval errors also rise from 123 to 129; final mistakes fall only **73 → 70** across 2,193 evaluations. The equal-dataset mean gives the smaller Iris and Zebra Special gains more weight than this pooled count. [Behavior totals](../results/adaptation_improvement/ar4/behavior_totals.csv), [harmful examples](../results/adaptation_improvement/ar4/harm_examples.csv).

**Interpretation:** the proposed feedback mechanism is feasible, with a promising localized classification gain and a marginal regression gain for RF. It has not produced a general advantage or substantially reduced wrong corrections. Keep sequential forests as the simpler baseline and retain alternating RF as an optional candidate for fresh-split validation. The experiment lacks an equal-budget retrieval-only continuation control; it cannot isolate the causal benefit of the forest final-loss term from additional retrieval training. In-sample forest predictions during retrieval updates and representation drift are possible limitations, not established explanations for the outcomes.

## Regression improves broadly; classification still depends on the dataset

The following detailed comparisons preserve the sequential `ar1`/`ar2`/`ar3` results. The matched alternating changes are reported above and in the linked `ar4` dataset table.

Regression dataset means:

| Dataset | Retrieval | Original sequential, 20% | Mixed ExtraTrees | Mixed RF | Retrieved-only ExtraTrees | Retrieved-only RF | Direct ExtraTrees |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Airfoil | 0.213374 | 0.211313 | 0.204292 | 0.207254 | **0.197548** | 0.197826 | 0.258698 |
| Car | 0.226829 | 0.215085 | 0.203991 | 0.191644 | 0.190017 | **0.182625** | 0.202149 |
| Yacht | 0.114986 | 0.063068 | 0.059385 | 0.058472 | 0.054240 | **0.052150** | 0.069896 |
| Energy Efficiency | 0.052527 | 0.050595 | 0.047008 | 0.046258 | 0.046782 | **0.045492** | 0.060334 |
| Student Performance | 0.442019 | 0.433240 | **0.389651** | 0.394708 | 0.395888 | 0.397598 | 0.447285 |
| Body Fat | 0.625301 | 0.627923 | 0.582594 | 0.594519 | 0.563135 | **0.560006** | 0.589621 |
| Diabetes | 0.765795 | 0.768434 | 0.759010 | 0.760527 | 0.757113 | 0.763927 | **0.749448** |

Retrieved-only ExtraTrees wins **7/7 dataset means and 20/21 seed pairs** against retrieval, and **7/7 means and 19/21 seed pairs** against original sequential 20%. Against mixed ExtraTrees training it wins **6/7 means but only 12/21 seed pairs**. Against direct trees it wins **6/7 means and 17/21 seed pairs**. Positive-error geometric mean ratios are 0.82487, 0.91337, 0.96898 and 0.86720 respectively. [1: paired comparisons](../results/adaptation_improvement/ar3/comparisons.csv).

Retrieved-only Random Forest also improves **all seven regression dataset means** against retrieval and original sequential 20%, winning **20/21 and 19/21 seed pairs** respectively. Against matching retrieved-only ExtraTrees, however, its **0.000729 RMSE** mean advantage consists of **4/7 dataset wins and 12/21 seed wins**, with three dataset and nine seed losses. Car, Yacht, Energy Efficiency and Body Fat improve; Airfoil, Student Performance and Diabetes worsen. Mixed Random Forest wins only **3/7 means and 10/21 seed pairs** against mixed ExtraTrees, with a slightly worse arithmetic mean. The two learners are close in regression.

Classification dataset means, percentage error:

| Dataset | Retrieval | Mixed ExtraTrees | Mixed RF | Retrieved-only ExtraTrees | Retrieved-only RF | Direct ExtraTrees | `ar2` validation choice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Iris | 7.778 | 6.667 | 6.667 | 6.667 | 5.556 | 6.667 | 7.778 |
| Wine | 3.704 | 3.704 | 3.704 | 3.704 | 3.704 | 3.704 | 3.704 |
| Breast Cancer | 2.339 | 2.339 | 2.047 | 2.632 | 2.632 | 3.801 | 1.754 |
| Balance | 5.867 | 7.200 | 6.933 | 6.133 | 6.133 | 6.667 | 6.133 |
| Digits | 2.222 | 2.315 | 2.315 | 2.222 | 2.222 | 2.685 | 1.944 |
| Zebra | 31.818 | 0.000 | 0.000 | 0.000 | 0.000 | 9.091 | 0.000 |
| Zebra Special | 28.030 | 6.818 | 3.788 | 9.091 | 3.030 | 18.939 | 6.818 |

Retrieved-only ExtraTrees wins/ties/loses **3/2/2 dataset means** against retrieval, and **5/2/0** against direct trees. Both pair recipes make their largest gains over retrieval on the two Zebra tasks. Against matching ExtraTrees, retrieved-only RF wins/ties/loses **2/5/0 dataset means and 4/17/0 seed pairs**. Mixed RF gives **3/4/0 dataset means and 6/13/2 seed pairs** against mixed ExtraTrees. Thus no classification dataset mean worsens in these matched comparisons, but most means tie.

Excluding both Zebra tasks gives **4.049% for retrieved-only RF**, **4.333% for mixed RF**, **4.272% for retrieved-only ExtraTrees**, **4.445% for mixed ExtraTrees**, and **4.382% for retrieval**. Original sequential 20% is **4.217%**, and the all-27 NN-CDH policy is **4.136%**. Retrieved-only RF's five-task improvement over matching ExtraTrees comes entirely from Iris; outside Iris and the Zebra tasks, their dataset means tie. This is a limited gain, not broad evidence of dominance. Both Zebra tasks remain in every primary mean. [1: all task means](../results/adaptation_improvement/ar3/tasks.csv), [classification sensitivity](../results/adaptation_improvement/ar3/classification_sensitivity.csv).

## How useful and how harmful are the corrections?

This section preserves the frozen-retrieval diagnostics from `ar1`/`ar2`/`ar3`. The alternating diagnostics above use each continuation's own selected retrieval prediction.

Define required residual `r = target - retrieval` and applied correction `c = final - retrieval`. Wrong direction means `r*c < 0` beyond the diagnostic tolerance. Harmful overshoot means the right direction but **more than twice** the needed magnitude. Passing the target by less than that can still help.

Regression percentages pool **6,339 test evaluations**, including repeated observations across seeds. These give larger datasets more influence than the family means. Very small changes can pass the direction threshold while their squared-error change remains below the harm threshold; the columns are not a partition.

| Method | Wrong direction % | Harmful overshoot % | Predictions worsened % | Overshoot harm, summed squared-error increases |
| --- | ---: | ---: | ---: | ---: |
| Original sequential, 20% dropout | 44.00 | 6.20 | 52.58 | 21.051 |
| Mixed ExtraTrees | 36.44 | 6.33 | 44.88 | 6.882 |
| Mixed Random Forest | 35.57 | 7.00 | 44.52 | 11.198 |
| Retrieved-only ExtraTrees | 34.04 | 7.21 | 43.52 | 3.097 |
| Retrieved-only Random Forest | 33.38 | 7.70 | 43.37 | 5.210 |
| Cross-fitted transport | 36.47 | 9.42 | 47.58 | 6.210 |
| Error memory | 42.81 | 4.26 | 47.77 | 0.420 |
| Vote attention | 40.94 | 0.98 | 40.56 | 0.302 |

Retrieved-only ExtraTrees improves direction frequency and reduces aggregate harm against the old adapter, even though its overshoots occur slightly more often. Its total squared-error reductions are **149.702**, versus **50.741** in increases. Original sequential 20% has **99.980 benefit / 74.541 harm**. Trees still worsen about 44% of examples; this is a useful but imperfect adapter.

Retrieved-only Random Forest has **174.293 benefit / 67.291 harm**. It makes fewer wrong-direction updates than matching ExtraTrees (**33.38% versus 34.04%**) but more harmful overshoots (**7.70% versus 7.21%**), and the harmful updates are costlier overall (**67.291 versus 50.741 summed squared-error increases**). Its larger useful corrections offset that increased damage. The slightly better mean RMSE therefore does not make it a safer adapter. Mixed RF similarly increases both benefit and harm to **158.734 / 73.021**, compared with mixed ExtraTrees' **127.319 / 55.193**. These pooled severity totals weight observations, unlike the equal-dataset headline means.

Error memory has **37.598 benefit / 27.446 harm**. Attention has only **21.028 benefit / 20.154 harm**. Their small overshoot damage largely accompanies smaller useful corrections too. The present results do not support treating a small update or bounded prediction as calibrated confidence.

Classification pools **2,193 evaluations**, with 123 errors before adaptation on the common frozen retrieval:

| Method | Retrieval errors fixed | Correct retrievals made wrong |
| --- | ---: | ---: |
| Original sequential, 20% | 22 | 15 |
| Local ridge | 14 | 24 |
| Pairwise neural | 45 | 32 |
| Mixed ExtraTrees | 54 | 10 |
| Mixed Random Forest | 65 | 15 |
| Retrieved-only ExtraTrees | 52 | 7 |
| Retrieved-only Random Forest | 60 | 6 |
| Cross-fitted transport | 45 | 19 |
| Error memory | 26 | 17 |
| Vote attention | 6 | 12 |

Retrieved-only RF fixes eight more retrieval errors and introduces one fewer harmful flip than matching ExtraTrees. Mixed RF fixes eleven more but also introduces five more harmful flips. The direct head fixes 70 retrieval errors but introduces 55; these are replacement-decision comparisons, since that control does not actually use retrieval. Valid class probabilities in the new methods do not guarantee calibration, and probability-margin changes are not directly comparable to old NN-CDH residual-score changes.

Failures remain. Both retrieved-only tree learners' largest harmful example is Diabetes seed 40, query index 85: it needs **+1.382** and receives **−0.538 from ExtraTrees or −0.652 from RF**. RF increases squared error by **2.227** on this example. Attention's largest is Car seed 42, index 421: it needs **+1.996** and receives **−0.786**, despite staying within the neighbor target range. The old pair neural and local ridge failures are retained: Car seed 40, index 1128, needs **−0.579**, but receives **+11.827** or **+17.163**. [1: per-run behavior](../results/adaptation_improvement/ar3/behavior.csv), [pooled behavior](../results/adaptation_improvement/ar3/behavior_totals.csv), [harmful examples](../results/adaptation_improvement/ar3/harm_examples.csv).

### Direct RF direction and severity check

An independent recount from **168 selected-test run units** confirms the reported rates. It uses a sign comparison on the saved final-minus-retrieval correction, without calling the original direction-diagnostic helper. Predictions, targets and each recorded final score were checked; no models were fitted or evaluated. [Reproducible audit](../tools/t1_forest_direction_audit.py), [counts, magnitudes and run sources](../results/adaptation_improvement/ar4/rf_direction_audit.json).

**The 30%+ wrong-direction figure applies to regression only.** Every such correction increases that query's absolute error. The denominator is all 6,339 regression query evaluations, including negligible updates and repeated observations across seeds, and the correction is the final aggregate after combining neighbors.

| RF pair recipe | Sequential wrong direction | Alternating wrong direction |
| --- | ---: | ---: |
| Mixed | 2,255 / 6,339 = 35.57% | 2,258 / 6,339 = 35.62% |
| Retrieved-only | 2,116 / 6,339 = 33.38% | 2,125 / 6,339 = 33.52% |

For sequential retrieved-only RF, the median wrong-direction correction is **0.01198** and the mean is **0.04005**, in each split's training-target standard-deviation units. Helpful corrections average **0.06201** in magnitude. Even excluding wrong updates of magnitude at most **0.001**, **1,997 / 6,339 = 31.50%** remain: the high rate is not merely floating-point noise. Many mistakes are small, but the largest 10% of wrong-direction mistakes account for **83.12%** of their harm. Wrong-direction updates contribute **62.062**, or **92.23%**, of the total **67.291** squared-error increase. They are consequential. Overall performance improves because the **174.293** total squared-error reduction from helpful updates is **2.59 times** the total increase from all harmful updates. These are pooled standardized-error totals, not equal-dataset averages.

Classification has a different interpretation. Sequential retrieved-only RF leaves **2,126 / 2,193 (96.94%)** predicted labels unchanged, fixes **60** retrieval errors, and changes **6** correct predictions into errors (**0.274%** of evaluations). The remaining changed label goes from one wrong class to another. Errors therefore fall from **123 to 69**. Alternating retrieved-only RF fixes **66** and harms **7**, reducing errors from **128 to 69**. Probability changes without a label change do not affect accuracy, but can affect log loss: the true-class probability decreases on **1,884 / 2,193 (85.91%)** sequential retrieved-only evaluations, including **1,877** with unchanged labels. This is a separate diagnostic, not the regression wrong-direction measure. Mean pooled negative log likelihood nevertheless improves from **0.45326 to 0.12134** (probabilities clipped to `[1e-12, 1]` for this audit). A count of unfavorable changes does not capture their magnitude.

## What the attention experiment tells us

The support restriction is verified in all 84 candidate validation sets and 42 selected test sets. It gives a real structural limit: the true class is missing from **27/2,193** test neighborhoods; in regression, **1,718/6,339 (27.10%)** targets lie outside the retrieved target range. No reweighting can exactly correct those regression targets.

But support is **not the whole explanation** for attention's weak performance. On Zebra Special, every test neighborhood contains the true class, yet attention worsens mean error from 28.030% to 29.545%. An oracle allowed arbitrary target-informed weights has a mean classification error floor of **3.267%** and regression RMSE floor of **0.085675**, much lower than the learned attention's 12.073% / 0.346362. These oracle bounds use test labels for diagnosis only and ignore the model's bounded score adjustment; they are not deployable results or tuning rules.

Attention selects the initial epoch-zero checkpoint in **10/21 classification** and **4/21 regression** runs. The experiment therefore tests one limited scorer, training objective and feature set; it does not rule out stronger attention models. [1: support limits](../results/adaptation_improvement/ar3/support_limits.csv), [selections](../results/adaptation_improvement/ar3/selections.csv).

## Bigger validation searches still select failures

The preserved `ar2` policy considers retrieval plus seven alternative adapter families, excluding the direct predictor. It uses validation error and a predeclared tie order. It reaches **4.019% / 0.328706**, improving the first policy's **4.099% / 0.336200** but still trailing retrieved-only trees in regression.

Classification selections are seven mixed-pair trees, five transport, four retrieval, three pair-neural, one retrieved-only tree and one attention. Regression selections are nine retrieved-only trees, four pair-neural, three transport, two mixed-pair trees, two attention and one error memory. Car seed 40 still chooses the pair neural model with its **+11.827** wrong-direction correction, because its validation error is lower. We retained that decision rather than overriding it using the test failure. [1: selection counts](../results/adaptation_improvement/ar3/selection_counts.csv).

The expanded `ar3` policy adds both RF families and reaches **3.586% / 0.317036**. It selects mixed RF in **2/21 classification** splits and RF in **10/21 regression** splits (six retrieved-only, four mixed). It selects no retrieved-only RF in classification, despite that family's lowest observed test mean. That distinction matters: a family identified retrospectively by test means is not the same as a validation-selected deployment rule.

Car seed 40 now selects retrieved-only RF using validation alone, with test RMSE **0.168456** versus the preserved `ar2` policy's **0.381193**. The old failure remains in its original row; the expanded policy avoids it without a test-based override. The expanded mean still trails the fixed retrieved-only RF family in both tasks. [1: policy selections](../results/adaptation_improvement/ar3/selections.csv).

## Relationship to all earlier adapter experiments

These studies replace the adaptation mechanism. Earlier work mostly modified NN-CDH inputs, regularization, gates or training schedules. The complete historical matrix and discussion remain in the [consolidated adapter summary](ADAPTER_RESULTS_SUMMARY_2026-11-26.md), with these individual reports:

| Earlier experiment | Main finding / reference | Report |
| --- | --- | --- |
| Magnitude/direction separation | 11.141% / 0.340946; shelved after no overall improvement | [Design comparison](ADAPTATION_IMPROVEMENT_LOCAL_2026-11-25.md), [behavior analysis](ADAPTATION_BEHAVIOR_2026-11-25.md) |
| Apply-or-skip gates | Simple disagreement rule: 9.164% / 0.337424; learned heads mixed | [Gate results](ADAPTATION_GATE_LOCAL_2026-11-25.md) |
| Trainable disagreement threshold | Did not establish a consistent advantage over validation selection | [Threshold results](ADAPTATION_THRESHOLD_LOCAL_2026-11-25.md) |
| Query-aware with prediction | Earlier mixed schedule: 9.359% / 0.345176 | [Query-aware results](QUERY_AWARE_LOCAL_2026-11-25.md) |
| Query-aware without prediction | 10.146% / 0.358695; removing the input did not prevent severe corrections | [No-prediction results](QUERY_NO_PREDICTION_LOCAL_2026-11-25.md) |
| Sequential / alternating / joint | Training schedule materially changes outcomes and sometimes retrieval itself | [Scheme comparison](ADAPTER_TRAINING_SCHEMES_LOCAL_2026-11-26.md) |
| Dropout across adapters/schemes | Strongest prior classification policy 6.453%; original sequential 20% is a useful frozen-core regression reference | [Dropout comparison](ADAPTER_DROPOUT_LOCAL_2026-11-26.md) |
| First non-NN-CDH alternatives | Mixed pair trees led; pair neural and local ridge retained instability | [First alternatives report](ADAPTER_PAIRWISE_REDESIGN_LOCAL_2026-11-26.md) |

These references have different schedules, objectives and search spaces. Their rows must not be pooled as independent experiments or presented as an equal-compute competition.

## Recommendation

**Prioritize retrieved-only pairwise trees for regression, keeping sequential ExtraTrees and Random Forest as the main baselines.** Alternating RF has the lowest observed mean, 0.313608, but its extra improvement over sequential RF is only 0.20%; alternating ExtraTrees does not improve its mean. This small gain is insufficient evidence to adopt the more expensive schedule by default. The stronger finding remains the usefulness of individual-case difference learning on retrieved pairs. The experiment supports this recipe; it does not prove that removing random pairs alone causes the gain, because sample counts and weighting change together.

For classification, **mixed-pair Random Forest with alternating training has the lowest observed individual-family mean, 3.191%**. Retrieved-only RF gives 3.255% alternating and 3.325% sequential. Keep these for confirmation: the mixed-RF continuation's gain occurs on two dataset means, with one loss and four ties, and amounts to only three fewer pooled test mistakes. This is a retrospective comparison of the four families, not a new validated cross-family selection policy.

Cross-fitted transport is the most useful of the new creative designs, but its five-forest cost and weaker performance make it a secondary candidate. Error memory and this bounded-attention implementation are useful negative/control results, not replacements to promote now.

Before changing defaults, the next decision-changing test is a locked comparison on fresh splits/tasks, retaining sequential forests and adding the promising RF continuations. An equal-budget retrieval-only continuation would test whether forest feedback adds value beyond extra core training. A bootstrap-matched comparison would separate threshold choice from row sampling; `ar3` changes both. Full TabR, gradient-boosted pair learners, richer set models, cycle-consistency training and end-to-end differentiable tree training remain untested.

## Implementation, evidence and verification

Code: [new prediction models](../model/t1/alternative_adaptation.py), [recorded/resumable training](../model/t1/alternative_training.py), [new driver](../tools/t1_alternative_redesign.py), [merged read-only summary](../tools/t1_alternative_redesign_summary.py), [new tests](../tests/test_t1_alternative_redesign.py), [local launcher](../../run_alternative_experiment.py). Earlier implementation is linked in [3: the first report](ADAPTER_PAIRWISE_REDESIGN_LOCAL_2026-11-26.md).

Random Forest addition: [shared pair fitting](../model/t1/pairwise_training.py), [recorded driver](../tools/t1_pair_random_forest.py), [focused tests](../tests/test_t1_pair_random_forest.py), [independent model/reuse audit](../tools/t1_pair_random_forest_audit.py), [local launcher](../../run_pair_random_forest_experiment.py).

Training-scheme clarification: [read-only run audit](../tools/t1_adaptation_training_scheme_audit.py), [audit log](../../../logs/ar3_training_scheme_audit.log). It checks the recorded scheme, source hashes and saved predictions; it does not run new training or inference.

Alternating implementation: [differentiable aggregation and retrieval block](../model/t1/forest_alternating.py), [recorded driver](../tools/t1_forest_alternating.py), [read-only cycle audit](../tools/t1_forest_alternating_summary.py), [independent metric/reuse audit](../tools/t1_forest_alternating_audit.py), [tests](../tests/test_t1_forest_alternating.py), [launcher](../../run_forest_alternating_experiment.py).

From the mini-project's `code/` directory with the configured uv environment:

```text
uv run --project ../.. --no-sync python run_alternative_experiment.py --study ar2 --workers 6
uv run --project ../.. --no-sync python run_alternative_experiment.py --study ar2 --mode summarize
uv run --project ../.. --no-sync python run_pair_random_forest_experiment.py --study ar3 --workers 6
uv run --project ../.. --no-sync python run_pair_random_forest_experiment.py --study ar3 --mode summarize
uv run --project ../.. --no-sync python neural-cbr/tools/t1_adaptation_training_scheme_audit.py --study ar3
uv run --project ../.. --no-sync python run_forest_alternating_experiment.py --study ar4 --workers 6
uv run --project ../.. --no-sync python run_forest_alternating_experiment.py --study ar4 --mode summarize
uv run --project ../.. --no-sync python neural-cbr/tools/t1_forest_alternating_audit.py --study ar4 --reuse
```

New RF training/evaluation data and weights remain under `NNKNN_RUNS/t1/ar3/rf_v1/`; `ar1` and `ar2` remain intact. Generated merged evidence is under `results/adaptation_improvement/ar3/`: [all comparison rows](../results/adaptation_improvement/ar3/rows.csv), [candidate validation rows](../results/adaptation_improvement/ar3/candidates.csv), [task means](../results/adaptation_improvement/ar3/tasks.csv), [family means](../results/adaptation_improvement/ar3/families.csv), [full JSON](../results/adaptation_improvement/ar3/comparison.json). Runtime logs are in the mini-project's `logs/ar3/`.

The latest continuation states and curves are under `NNKNN_RUNS/t1/ar4/af_v1/`; generated merged evidence is under `results/adaptation_improvement/ar4/`. [All 966 rows](../results/adaptation_improvement/ar4/rows.csv), [322 task means](../results/adaptation_improvement/ar4/tasks.csv), [46 family means](../results/adaptation_improvement/ar4/families.csv), [cycle trajectories](../results/adaptation_improvement/ar4/alternating_cycles.csv), [retrieval diagnostics](../results/adaptation_improvement/ar4/alternating_diagnostics.csv), [full evidence](../results/adaptation_improvement/ar4/comparison.json).

**Alternating verification:** 57 focused tests passed, including 15 new cases covering parity with the existing forest predictor, actual final-loss gradients into retrieval, self-exclusion, held-out-target isolation, exact interrupted retrieval/optimizer recovery, validation-only selection, saved-state audits and completed-unit reuse. The merged reader checks all **840 retrieval states, 840 refreshed pair digests/forest parameter sets, 840 refit validation predictions and 168 selected test predictions**. Independent calculations verify **1,008 validation metric pairs** (including 168 reused starts), all **168 selections/test metric pairs**, and all **1,848 new units reused with fitting/inference/data reconstruction disabled**. QA reconciles 322 task means, 46 family means and 16 independently calculated paired forest means; **798 historical rows and all 18 preceding generated artifacts are unchanged**. Import smoke and both core manifests pass. The figure was visually checked and its legend moved clear of the data. [Audit results](../results/adaptation_improvement/ar4/alternating_audit.json), [tests](../../../logs/ar4_focused_tests.log), [run log](../../../logs/ar4_launcher.log), [merged reader](../../../logs/ar4_summary.log), [independent audit log](../../../logs/ar4_audit.log), [summary QA](../../../logs/ar4_summary_qa.log).

Before `ar4`, **42 focused tests passed** across `ar1`/`ar2`/`ar3`. They cover query-target isolation, train-only memory construction, self-exclusion, pair reversal, vote bounds and permutation invariance, gradients, valid class probabilities, cross-fit head exclusion, exact interrupted recovery, validation-only policy selection and completed-unit reuse. The eight added RF tests also verify matching pair recipes/settings and preserving the old selection policy. Fitted algorithms/settings were not changed in response to test results.

The merged reader reconciled **1,050 validation prediction sets, 420 selected-family test sets and 210 reused NN-CDH test sets**, checked **42 contexts / 420 frozen-core hashes**, audited **126 transport candidates' fold exclusions**, verified **126 attention support sets**, and matched **168 RF pair-array digests** to the unchanged builders. The model audit verified all **168 RF estimator types, bootstrap flags, splitters and matched settings**, independently recomputed **168 validation errors and 168 matched RF/ExtraTrees test errors**, and reused all **252 new units with fitting and prediction disabled**. Earlier reuse audits remain linked in the project index.

Independent summary checks confirmed all **672 historical rows are unchanged**, matched eight independently calculated forest family means, recomputed **266 task means and 38 family means**, reconciled seed counts for **128 paired comparisons**, and verified **42 expanded validation-policy choices**. Both shared-core manifests passed. The figure was inspected; 19 main-table rows, 14 dataset rows, 18 behavior rows and 328 local links were verified. [Model/reuse audit](../results/adaptation_improvement/ar3/forest_audit.json), [audit log](../../../logs/ar3_audit.log), [summary checks](../../../logs/ar3_summary_qa.log), [report checks](../../../logs/ar3_report_qa.log).

The pre-commit sweep of all 52 T1 test modules reported **600 passed, 5 failed, 5 skipped and 2 expected failures** on local Windows CPU. Two failures were obsolete assertions that regression dropout must be rejected; these now test the supported residual-dropout behavior and retain rejection for unsupported magnitude/direction dropout. Two grid checks encountered the experiment-only deterministic setting. Rechecking the three affected modules with that setting unset gave **37 passed, 3 skipped and 1 deselected**. The deselected Linux numeric-reference test still fails locally: the same value, `0.24344214471057057` versus reference `0.2433512657880783`, was reproduced on the unchanged `411abd4` commit. The Linux reference and tolerance were preserved. The full sweep was not repeated after the test updates; these are focused recheck results, not a claim of a fully passing suite. Import smoke and core-manifest checks passed. [Full sweep](../../../logs/adapter_commit_tests_2026-11-27.log), [rechecks](../../../logs/adapter_commit_rechecks_2026-11-27.log), [unchanged-commit reproduction](../../../logs/adapter_commit_baseline_golden.log).

Final report readback verifies the 23-row merged score table, both four-row matched tables, the four new behavior rows, local artifact links and heading anchors. [Report QA](../../../logs/ar4_report_qa.log).

Shared NN-kNN code, Table 1 and default model selection remain unchanged. `ar4` updates learned retrieval parameters in its own run units. The limitations concern scientific generalization, not missing runs: every planned cell is complete and every failure is retained.
