# Adaptation improvement references

## Project evidence

- [Neural CBR weekly report, November 24, 2026](https://github.com/nkjcqvcpi/NN-kNN/blob/411abd4bdb960fd7e81b595a55c59ffff1305e43/reports/NEURAL_CBR_WEEKLY_2026-11-24.md): section 2 compares retrieval-only, the jointly trained core before adaptation and the final adapted prediction; section 4 covers reuse and synchronization. These are historical exploratory results, not tests of the proposed gate.
- [Regression adapter](https://github.com/nkjcqvcpi/NN-kNN/blob/411abd4bdb960fd7e81b595a55c59ffff1305e43/model/nn_cdh.py#L278): learns a residual and returns retrieval plus residual.
- [Classification adapter](https://github.com/nkjcqvcpi/NN-kNN/blob/411abd4bdb960fd7e81b595a55c59ffff1305e43/model/nn_cdh.py#L622): primary residual-score mode and alternative logit-residual mode.

## Methodological precedents

- Srivastava, Greff and Schmidhuber (2015), [Training Very Deep Networks](https://proceedings.neurips.cc/paper/2015/file/215a71a12769b056c3c32e7299f1c5ed-Paper.pdf), NeurIPS. Highway gates interpolate between carrying an input and transforming it; the closest algebraic precedent for the proposed gate.
- Jacobs, Jordan, Nowlan and Hinton (1991), [Adaptive Mixtures of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf), Neural Computation. A learned gating network allocates inputs among experts.
- Zheng et al. (2021), [Adaptive Nearest Neighbor Machine Translation](https://aclanthology.org/2021.acl-short.47/), ACL. A learned network uses retrieval statistics to mix prediction alternatives.
- Jiang et al. (2022), [Towards Robust k-Nearest-Neighbor Machine Translation](https://aclanthology.org/2022.emnlp-main.367/), EMNLP. Incorporates confidence information from both retrieved and neural predictions into interpolation.
- Mozannar and Sontag (2020), [Consistent Estimators for Learning to Defer to an Expert](https://proceedings.mlr.press/v119/mozannar20b.html), ICML. Related framework for choosing when an alternative predictor is expected to perform better.
- Lee et al. (2015), [Deeply-Supervised Nets](https://proceedings.mlr.press/v38/lee15a.html), AISTATS. General precedent for auxiliary supervision alongside the final output objective; it does not study or validate the proposed NN-kNN gate/residual scale control.
- Salimans and Kingma (2016), [Weight Normalization](https://proceedings.neurips.cc/paper_files/paper/2016/file/ed265bc903a5a097f61d3ec064d96d2e-Paper.pdf), NeurIPS. Separates magnitude and unit direction of parameter vectors; a mathematical analogy, not an evaluation of two-head adaptation outputs.
- Kendall and Gal (2017), [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://proceedings.neurips.cc/paper_files/paper/2017/file/2650d6089a6d640c5e85b2b88265dc2b-Paper.pdf), NeurIPS. Models predictive mean and uncertainty separately; relevant to distinguishing a small mean correction from uncertainty.

These papers establish related mechanisms in their own settings. They do not demonstrate that the proposed adaptation gate improves NN-kNN.

## Statistical definitions

- [scikit-learn decision-tree mathematical formulation](https://scikit-learn.org/stable/modules/tree.html#mathematical-formulation): definitions of Gini impurity for classification and squared deviation from the mean for regression. The proposed gate uses these as neighborhood-disagreement summaries, not as a decision-tree implementation.
