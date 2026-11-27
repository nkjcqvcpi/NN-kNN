# Alternatives to NN-CDH: merged experiments and results

November 27, 2026. Completed studies `ar1`, `ar2` and `ar3`, plus recorded NN-CDH references. Latest addition: ordinary Random Forest on both existing pair formulations. [1: merged numerical evidence](../results/adaptation_improvement/ar3/comparison.json), [2: creative-alternatives protocol](../docs/NNCDH_ALTERNATIVES_FOLLOWUP_EXPERIMENT.md), [3: first alternatives report](ADAPTER_PAIRWISE_REDESIGN_LOCAL_2026-11-26.md).

## Executive summary

**Retrieved-only pairwise Random Forest now has the lowest individual-family mean in both tasks, with a clear classification gain and a very small regression difference.** Classification error falls from **4.350% for matching ExtraTrees to 3.325%**, a **1.025 percentage-point** reduction. Regression RMSE changes from **0.314960 to 0.314232**, only **0.23% lower**. Keep both tree learners as regression candidates; this small observed difference does not establish a reliable advantage.

**The broader result is that retrieved-only pair adaptation helps regression substantially.** Random Forest's **0.314232** and ExtraTrees' **0.314960** are below frozen retrieval's **0.348690** and original sequential NN-CDH with 20% dropout's **0.338523**. Mixed-pair Random Forest gives **3.636% / 0.321912** versus mixed ExtraTrees' **4.149% / 0.320848**: it improves classification but slightly worsens regression. Changing the tree learner is not an across-the-board improvement.

**The direct-tree control supports retaining case-based adaptation under these settings.** A direct ExtraTrees head on the same frozen representation reaches **7.365% classification error / 0.339633 RMSE**; retrieved-only pair trees reach **4.350% / 0.314960**. The pair approach wins or ties every classification dataset mean and improves six of seven regression means. This is stronger evidence than simply comparing trees with neural adapters, although training examples, objectives and effective capacity still differ.

**The three creative designs have mixed value.** Cross-fitted anchor transport is useful (**7.717% / 0.328438**) but trails pairwise trees. Retrieval-error memory makes smaller gains (**9.238% / 0.342571**). Neural vote attention prevents predictions outside the retrieved target range, yet achieves only **12.073% / 0.346362**. Making corrections conservative does not make them accurate.

Classification gains over retrieval remain concentrated on Zebra and Zebra Special. The additional retrieved-only Random Forest gain over ExtraTrees comes from Iris and Zebra Special; the other five dataset means tie. The regression evidence for pair adaptation is broader than the evidence favoring one tree learner. These are exploratory results on repeatedly examined local splits; the original default remains unchanged.

## What was tested

The shared unit is a dataset/seed split: seven classification and seven regression datasets, seeds **40, 41, 42**. All three studies use identical saved train/validation/test splits and pretrained NN-kNN core states. Query/case representations are scaled using training cases only. Training retrieval excludes the query's own case; all validation/test anchors come from training.

`ar1` tested three replacements. `ar2` added three designs and two controls. `ar3` adds two Random Forest pair families: **168 validation candidates and 84 selected-family tests, 252 new units**. Together the three studies contain **1,050 validation candidate units, 420 selected-family test units and 42 shared context units: 1,512 recorded units**. The merged comparison has **798 rows**, covering 19 methods/policies and 42 splits. New work was local CPU only. No previous model or completed unit was retrained.

Classification numbers below are **percentage error**. Regression is **RMSE in training-standardized target units**. Lower is better. Family means average seeds within each dataset, then weight the seven datasets equally. Pooled behavior counts later in this report use a different weighting.

## Training scheme: sequential, with retrieval frozen

**All new alternatives in `ar1`, `ar2` and `ar3` use the sequential frozen-retrieval protocol. No forest result in this report uses joint or alternating training.** First reuse the recorded pretrained NN-kNN core, then fit or construct the adaptation method from its saved training contexts. Its representation, distance parameters, case biases, case store, neighborhood membership and base normalized activations are fixed throughout adaptation. The adaptation loss does not update retrieval.

| Component / comparison | Training scheme actually tested | Can adaptation training change the NN-kNN core? |
| --- | --- | --- |
| ExtraTrees / Random Forest pair adapters; direct-tree and transport controls | Sequential, using frozen saved representations and neighborhoods | No |
| Pair neural, error memory and vote attention in the redesign studies | Same sequential frozen-core protocol | No; attention changes only the adapter's final voting weights |
| Local ridge | Local fit at inference on the fixed neighborhood; recorded under the same sequential comparison protocol | No |
| Earlier NN-CDH original / query-aware / query-without-prediction studies | Sequential, alternating and joint were separately tested in `as1` / `ad1` | Yes in alternating/joint, according to their retrieval objective; no in sequential |

**“Mixed pairs” is a pair-sampling recipe, not a mixed training schedule.** It means retrieved pairs plus random training-case pairs. Both mixed-pair and retrieved-only forests here are sequential. The historical NN-CDH “mixed schedule” reference is different: it uses the earlier joint runs except for sequential Diabetes.

The read-only scheme audit checks **420 selected-family runs across 42 contexts**: all record `scheme="sequential"`, all initial/selected core hashes match the shared source, and all stored retrieval predictions match exactly. The existing result rows already contain the correct scheme. This clarification adds evidence tables without changing scores or run configurations. [Training schemes](../results/adaptation_improvement/ar3/training_schemes.csv), [per-run audit](../results/adaptation_improvement/ar3/training_scheme_runs.csv), [audit details](../results/adaptation_improvement/ar3/training_scheme_audit.json).

### Could a forest influence retrieval in joint or alternating training?

**Yes, with a deliberately designed feedback path; it does not happen automatically.** The current forest implementation calls scikit-learn on saved NumPy arrays. It is outside PyTorch autograd. Ordinary tree predictions are constant inside each leaf region, so the hard split decisions do not provide the useful input gradients that a neural adapter supplies. The existing NN-CDH joint/alternating optimizer cannot train these forests as a drop-in neural module. [Decision-tree formulation](https://scikit-learn.org/stable/modules/tree.html#mathematical-formulation).

That does not block every possible gradient path. In a future custom retrieval step, hold each forest correction `delta_i` fixed and compute the aggregation in PyTorch:

```text
regression:      final = p_retrieval(theta) + sum_i w_i(theta) * stop_gradient(delta_i)
classification: final = sum_i w_i(theta) * stop_gradient(adapted_class_distribution_i)
```

The final prediction loss could then update retrieval parameters `theta` through its prediction and/or activation weights. This would favor cases whose adapted predictions are useful. It would not differentiate through forest splits or discrete neighbor selection. If retrieval were updated only by its original retrieval loss, the forest would provide no adaptation-loss feedback even if the two training steps alternated.

A possible alternating experiment would fit the forest, freeze it while updating retrieval using that final-loss path, then refresh representations/neighborhoods/pairs and refit the forest. Refreshing matters because changing the representation changes the forest's input distribution. This is a proposed custom training loop, **not an implemented or measured result**. True end-to-end differentiation through the tree path would require a different differentiable tree model or an explicit approximation. The current results establish the value of adaptation with retrieval fixed; they do not show whether forest-informed retrieval training would improve it.

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

Candidate selection is validation-only: error, then final loss, then candidate ID. Only each family's winner reaches test. Neural checkpoints minimize validation final loss; attention also allows its initial epoch-zero checkpoint. The old NN-CDH policies retain their original validation-loss-first rule. No test-based overrides, clipping thresholds or exclusions were introduced. [1: family means](../results/adaptation_improvement/ar3/families.csv), [candidate results](../results/adaptation_improvement/ar3/candidates.csv).

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
| **Random Forest, retrieved-only training** | **3.325** | **0.314232** |
| Direct ExtraTrees head, control | 7.365 | 0.339633 |
| Cross-fitted anchor transport | 7.717 | 0.328438 |
| Retrieval-error memory | 9.238 | 0.342571 |
| Neural vote attention | 12.073 | 0.346362 |
| Validation choice: retrieval + first three alternatives | 4.099 | 0.336200 |
| Validation choice: retrieval + seven alternatives (`ar2`) | 4.019 | 0.328706 |
| Validation choice: retrieval + nine alternatives, including RF | 3.586 | 0.317036 |

The 9/27-setting NN-CDH policies are not its strongest fixed regression baselines: they sometimes select a severe Car failure. The earlier probability gate reaches **0.337206**, and query + prediction with alternating training and 10% dropout reaches **0.337381**. Retrieved-only pair trees also improve on those means. Prior mixed/joint NN-CDH references can change the retrieval core; sequential references and all new alternatives use the common frozen core.

![Merged alternatives comparison](../results/adaptation_improvement/ar3/alternatives.png)

The earlier compact local CatBoost result was **4.517% / 0.318980**, versus retrieved-only ExtraTrees' **4.350% / 0.314960** and retrieved-only Random Forest's **3.325% / 0.314232**. Both pair learners are lower on both means in this local comparison. This does not establish superiority to a tuned CatBoost benchmark: search budgets, learned representations and historical retrieval pretraining differ. MLP/TabM references and their limits remain in the [conventional baseline report](QUERY_AWARE_LOCAL_2026-11-25.md).

## Regression improves broadly; classification still depends on the dataset

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

**Prioritize retrieved-only pairwise trees for regression, keeping both ExtraTrees and Random Forest.** RF has a marginally lower mean but loses three of seven dataset comparisons and causes more aggregate harmful correction damage. The stronger finding is the usefulness of individual-case difference learning on retrieved pairs. The experiment supports this practical recipe; it does not prove that removing random pairs alone causes the gain, because sample counts and weighting change together.

For classification, **retrieved-only pairwise Random Forest is the leading candidate for confirmation**: it has the lowest observed individual-family mean and fixes more retrieval errors while introducing fewer harmful flips than matching ExtraTrees. Its advantage is concentrated on two dataset means. Keep both pair recipes and existing neural options for validation; fresh splits are needed before treating this as an established winner.

Cross-fitted transport is the most useful of the new creative designs, but its five-forest cost and weaker performance make it a secondary candidate. Error memory and this bounded-attention implementation are useful negative/control results, not replacements to promote now.

Before changing defaults, the next decision-changing test is a locked comparison of retrieved-only RF and ExtraTrees, mixed pair trees, direct trees and the regularized original NN-CDH on fresh splits/tasks. A bootstrap-matched comparison would separate threshold choice from row sampling; this study changes both. A differentiable individual-case model remains worth pursuing, but this study does not show that adding attention or enforcing small corrections is sufficient. Full TabR, gradient-boosted pair learners, richer set models, cycle-consistency training and joint retraining of the new adapters remain untested.

## Implementation, evidence and verification

Code: [new prediction models](../model/t1/alternative_adaptation.py), [recorded/resumable training](../model/t1/alternative_training.py), [new driver](../tools/t1_alternative_redesign.py), [merged read-only summary](../tools/t1_alternative_redesign_summary.py), [new tests](../tests/test_t1_alternative_redesign.py), [local launcher](../../run_alternative_experiment.py). Earlier implementation is linked in [3: the first report](ADAPTER_PAIRWISE_REDESIGN_LOCAL_2026-11-26.md).

Random Forest addition: [shared pair fitting](../model/t1/pairwise_training.py), [recorded driver](../tools/t1_pair_random_forest.py), [focused tests](../tests/test_t1_pair_random_forest.py), [independent model/reuse audit](../tools/t1_pair_random_forest_audit.py), [local launcher](../../run_pair_random_forest_experiment.py).

Training-scheme clarification: [read-only run audit](../tools/t1_adaptation_training_scheme_audit.py), [audit log](../../../logs/ar3_training_scheme_audit.log). It checks the recorded scheme, source hashes and saved predictions; it does not run new training or inference.

From the mini-project's `code/` directory with the configured uv environment:

```text
uv run --project ../.. --no-sync python run_alternative_experiment.py --study ar2 --workers 6
uv run --project ../.. --no-sync python run_alternative_experiment.py --study ar2 --mode summarize
uv run --project ../.. --no-sync python run_pair_random_forest_experiment.py --study ar3 --workers 6
uv run --project ../.. --no-sync python run_pair_random_forest_experiment.py --study ar3 --mode summarize
uv run --project ../.. --no-sync python neural-cbr/tools/t1_adaptation_training_scheme_audit.py --study ar3
```

New RF training/evaluation data and weights remain under `NNKNN_RUNS/t1/ar3/rf_v1/`; `ar1` and `ar2` remain intact. Generated merged evidence is under `results/adaptation_improvement/ar3/`: [all comparison rows](../results/adaptation_improvement/ar3/rows.csv), [candidate validation rows](../results/adaptation_improvement/ar3/candidates.csv), [task means](../results/adaptation_improvement/ar3/tasks.csv), [family means](../results/adaptation_improvement/ar3/families.csv), [full JSON](../results/adaptation_improvement/ar3/comparison.json). Runtime logs are in the mini-project's `logs/ar3/`.

**42 focused tests passed**, covering all three studies. They cover query-target isolation, train-only memory construction, self-exclusion, pair reversal, vote bounds and permutation invariance, gradients, valid class probabilities, cross-fit head exclusion, exact interrupted recovery, validation-only policy selection and completed-unit reuse. The eight added RF tests also verify matching pair recipes/settings and preserving the old selection policy. Fitted algorithms/settings were not changed in response to test results.

The merged reader reconciled **1,050 validation prediction sets, 420 selected-family test sets and 210 reused NN-CDH test sets**, checked **42 contexts / 420 frozen-core hashes**, audited **126 transport candidates' fold exclusions**, verified **126 attention support sets**, and matched **168 RF pair-array digests** to the unchanged builders. The model audit verified all **168 RF estimator types, bootstrap flags, splitters and matched settings**, independently recomputed **168 validation errors and 168 matched RF/ExtraTrees test errors**, and reused all **252 new units with fitting and prediction disabled**. Earlier reuse audits remain linked in the project index.

Independent summary checks confirmed all **672 historical rows are unchanged**, matched eight independently calculated forest family means, recomputed **266 task means and 38 family means**, reconciled seed counts for **128 paired comparisons**, and verified **42 expanded validation-policy choices**. Both shared-core manifests passed. The figure was inspected; 19 main-table rows, 14 dataset rows, 18 behavior rows and 328 local links were verified. [Model/reuse audit](../results/adaptation_improvement/ar3/forest_audit.json), [audit log](../../../logs/ar3_audit.log), [summary checks](../../../logs/ar3_summary_qa.log), [report checks](../../../logs/ar3_report_qa.log).

The pre-commit sweep of all 52 T1 test modules reported **600 passed, 5 failed, 5 skipped and 2 expected failures** on local Windows CPU. Two failures were obsolete assertions that regression dropout must be rejected; these now test the supported residual-dropout behavior and retain rejection for unsupported magnitude/direction dropout. Two grid checks encountered the experiment-only deterministic setting. Rechecking the three affected modules with that setting unset gave **37 passed, 3 skipped and 1 deselected**. The deselected Linux numeric-reference test still fails locally: the same value, `0.24344214471057057` versus reference `0.2433512657880783`, was reproduced on the unchanged `411abd4` commit. The Linux reference and tolerance were preserved. The full sweep was not repeated after the test updates; these are focused recheck results, not a claim of a fully passing suite. Import smoke and core-manifest checks passed. [Full sweep](../../../logs/adapter_commit_tests_2026-11-27.log), [rechecks](../../../logs/adapter_commit_rechecks_2026-11-27.log), [unchanged-commit reproduction](../../../logs/adapter_commit_baseline_golden.log).

Core NN-kNN, Table 1 and default model selection remain unchanged. The limitations concern scientific generalization, not missing runs: every planned cell is complete and every failure is retained.
