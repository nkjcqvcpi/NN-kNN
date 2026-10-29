# What we did in "neural case-based reasoning": a plain-language account

Status as of 2026-10-20. Chinese version: `NEURAL_CBR_EXPLAINED_2026-10-20_ZH.md` (same structure, same numbers). Technical tables behind every number here are in `NEURAL_CBR_PROGRESS_2026-10-20_EN.md`. This document needs no background in machine learning; terms are explained where they first appear and collected in a glossary at the end.

## 1. The problem in one paragraph

Many real prediction problems come as spreadsheets: each row is a thing (a patient, a loan, a house, a sensor reading) described by columns, and we want to predict one more column (is the patient sick, will the loan default, what is the price). A computer program learns from rows where the answer is known and must then answer for new rows. We studied one family of such programs that works the way a human expert often does: by **remembering similar past cases**.

## 2. The idea: case-based reasoning (CBR)

Imagine a doctor who sees a new patient and recalls three earlier patients with almost the same symptoms, and what happened to them. That is case-based reasoning. Classical CBR describes a four-step cycle, often called "the four R's":

1. **Retrieve** the most similar past cases.
2. **Reuse** them: turn what happened there into a prediction, adjusting for differences.
3. **Revise**: check the prediction (a person, or a rule, may correct it).
4. **Retain**: decide which cases to keep in the memory for the future, adding useful ones and dropping useless or harmful ones.

The simplest version of retrieval is called k-nearest-neighbours (kNN): measure how far the new row is from every stored row, take the k closest, and let them vote. Its weakness is that "closest" is decided by a fixed ruler: every column counts the same, and every stored case is trusted the same, even though some columns are noise and some stored cases are misleading.

**Neural CBR** keeps the remembering idea but lets the computer *learn* the ruler. Our retrieval engine, called **NN-kNN**, learns three things by gradient descent (the standard way neural networks learn): how much each column matters (feature weights, which may differ between regions of the data), how much each stored case should be trusted (a per-case bias), and optionally a transformation of the columns into a better description (an encoder, a small neural network). So the memory is still a list of real past cases that a person can inspect, but how they are compared is learned.

## 3. What we built around the retrieval engine

Over the project we built the other steps of the cycle and the tooling to study them:

- **Reuse adapter.** A small learned module that takes the retrieved cases and corrects the prediction (for classification it adjusts the class scores; for regression it adjusts the number). Because the correction is a separate piece, we can measure how much it contributes. We also made it report calibration: whether "90% sure" really means right nine times in ten.
- **Case provenance and bias.** Two ways to score how useful each stored case is: a *credit* score (Q) that records how often a case took part in correct versus wrong predictions, and the learned *bias* (B). These are the ingredients for deciding what to keep.
- **Retention / maintenance.** Code that removes cases to fit a size budget, using Q, B, random choice, or removal tests ("does the prediction get worse if we delete this case?"), with stable case numbers so every removal can be traced and undone.
- **Revision tools.** Controls that let a reviewer (simulated in our tests) force a retrieval, edit a label, quarantine or restore a case, and compare against a matched run with no edit.
- **Training schedules.** The retrieval part and the reuse part both have to be trained. We implemented several ways of combining them: train retrieval first and freeze it (*sequential*); alternate between the two in short blocks (*alternating*); and, new in this phase, train both **at the same time** (*joint*), in two flavours (one combined objective, or each part following only its own objective). A collaborator had pointed out that earlier we only had sequential and alternating training; the joint schedule closes that gap.
- **Experiment machinery.** A runner that records every setting and result, a pre-registered analysis tool, a search tool that tunes five kinds of models fairly (NN-kNN, plain kNN, CatBoost, a neural network (MLP), and TabM), and tests for all of it.

## 4. The questions we asked, and what we found

### 4.1 Does a "momentum encoder" (EMA) help?

A collaborator asked whether we should keep two copies of the encoder, one used for new queries and a slowly updated copy (an exponential moving average, EMA) used for the stored cases, as some modern methods do. We tested this on 13 tabular tasks with several settings. **Finding:** it did not help. With default settings the plain raw-column version had lower typical error than every EMA variant, and where we tuned the encoder, no EMA variant beat a simple shared encoder. So we do not use it by default.

### 4.2 Should retrieval and reuse be trained together?

We implemented joint training (Section 3) and searched the settings. It ranked best among the schedules, but its advantage over the simple alternating default was small once the retrieval engine was tuned. An early search had also been distorted by a mismatch in a training setting (a batch size) that made some runs collapse; after fixing it, the apparent 11.6% gain shrank to 1.8% and was no longer statistically convincing. The later confirmation (Section 5) found no schedule effect at all.

### 4.3 Harder data

Easy datasets cannot tell methods apart: almost everything is nearly perfect. So we added ten harder real datasets, chosen before looking at any result for our method: predicting protein properties, housing and diamond prices, customer churn, superconducting materials, a chemistry dataset with over a thousand columns, and others. All methods were given at most 3,000 training rows to keep the comparison fair and affordable. An important discovery on these: on one dataset the reuse adapter made the final answer good while the retrieval underneath had become worse than guessing the average. The correction was hiding a damaged memory. We therefore added a safeguard (a "guard") that checks retrieval quality before the correction.

### 4.4 Tuning, done carefully

We tuned the training settings, but only on separate "development" random splits, and froze the final recipe (called **F**) before running any "evaluation" split. This matters: tuning on the data you then report on makes results look better than they are. In the tuning, two decisions were kept: choosing the best schedule separately for each dataset, and scaling the class scores by 10 for classification, which fixed a calibration problem (before the fix, the model's confidence after correction was badly off). Every other idea we tried (a retrieval guard, other output forms, adapter size, training budget) did not help on held-out data and was dropped; with only six development splits, adding more candidate settings mostly fits noise.

## 5. The final test (pre-registered)

Before running it, we wrote down exactly what we would test and how we would decide (a **pre-registration**), so that we could not choose the story after seeing the numbers. We compared F with its alternatives on 30 datasets (20 older ones, 10 harder ones) using fresh random splits: 10 splits per dataset, and 30 for the seven smallest datasets. Errors were compared as ratios (below 1 favours F) and combined across datasets with a standard statistical method that allows datasets to differ.

| Question | Compared with | Ratio of errors [95% interval] | Result |
|---|---|---|---|
| Does the reuse step help? | retrieval only | 0.974 [0.936, 1.014] | no detectable gain |
| Does the training schedule matter? | the simple alternating schedule | 0.996 [0.990, 1.003] | no detectable effect |
| Is F nearly as good as the best boosted-tree model? | tuned CatBoost | 1.135 [1.004, 1.283] | F trails by about 13.5% |
| Is F nearly as good as a strong neural model? | tuned TabM | 0.987 [0.855, 1.138] | "nearly as good" not shown |

In plain words: **on these datasets, our neural CBR did not beat simple retrieval, did not benefit from the training schedule, and was somewhat worse than the best standard tools.** Details that matter:

- The reuse step helped clearly on a few small datasets (for example, error ratio 0.55 on the yacht data), but not on the ten harder ones (pooled 0.995).
- Its confidence estimates improved after correction, but corrections turned about 1.17 wrong predictions into right ones for every right one they spoiled, below the 2-to-1 we had set as a bar.
- The comparison with TabM looks slightly favourable only because TabM did badly (about half wrong) on two tiny datasets; on the ten harder datasets F is behind TabM.
- Our guard checks all held: the correction did not hide damaged retrieval on average.

We report this as it is. The honest conclusion is a negative result for the claim that reuse and joint training help on tuned retrieval, together with a clear statement of where small gains do occur.

## 6. Is the "keeping and forgetting" step designed well? A review

The user's collaborator asked us to compare ways of maintaining the memory, and we were asked whether the maintenance step is designed properly. The final recipe F does not include maintenance, so none of the results above depend on it. We reviewed it anyway, with independent reviewers each trying to refute every flaw we claimed. All thirteen flaws survived. In plain terms:

- **The usefulness score mostly measures popularity.** A case gets credit for the outcome of any prediction it takes part in, whether or not the case itself is right. So the score largely tracks how often a case is retrieved. In an experiment with deliberately flipped labels, it kept 58-60% of the corrupted cases when 75% of the memory was kept, while random choice kept 75%.
- **Many cases are never used at all.** On several datasets, 66 to 98 percent of stored cases are never retrieved by any training query, so "remove half the memory" removes mostly cases that were doing nothing, and ties between them are settled by row number.
- **The scores are computed on data the model already learned from,** so mistakes caused by bad cases are rare and hard to see.
- **"Dynamic" maintenance prunes once** and never re-admits anything.
- **It was never tested where it should matter:** noisy labels, duplicates, rare groups, very large memories. That is why earlier results said removing cases gives no consistent gain: the tests could not have shown otherwise.

We fixed three safe problems without changing the default behaviour (a bug that made all conditions in a comparison use the same maintenance policy, an option to break ties randomly instead of by row number, and a logging error). We propose, for the supervisor's decision, a pre-registered clarification of the planned maintenance test and a new exploratory study with noisy, duplicated, imbalanced and larger data.

## 7. How we made the results trustworthy

- Tuning used only separate development splits; the final recipe was frozen first; evaluation splits were used once.
- Every run records its settings and a fingerprint of the code that produced it; an audit rebuilt each stored result from the run's own files and found every run present. One result line lost to a file-writing clash was rebuilt from its own saved files, not re-run.
- A machine interruption stopped runs midway; the affected runs were redone and their duplicate lines were identical.
- Results from the earliest experiments were re-checked against later code fixes, and only those still valid are reported.

## 8. What this means, and what is next

- **What we can say:** neural CBR with learned retrieval is a working, inspectable system; on tabular data it competes with a plain nearest-neighbour method but does not yet match tuned tree and neural models, and the extra reuse and joint-training machinery adds little on well-tuned retrieval.
- **What we cannot say:** that maintenance helps or hurts. It has not been properly tested.
- **Next steps (for decision):** (1) specify and run the pre-registered maintenance comparison; (2) an exploratory study on noisy, duplicated, imbalanced and larger datasets with better usefulness scores and classical editing methods as baselines; (3) only then dynamic maintenance.

## Glossary

- **Case:** one stored row with its known answer.
- **kNN:** predicting from the k most similar stored cases.
- **NN-kNN:** kNN whose similarity (feature weights, per-case trust, optional encoder) is learned.
- **Encoder:** a small neural network that re-describes the columns before comparing.
- **EMA (momentum) encoder:** a slowly updated copy of a network.
- **Calibration:** whether stated confidence matches how often the prediction is right.
- **Adapter:** the small module that corrects the retrieved prediction.
- **Schedule:** the order in which retrieval and reuse are trained.
- **Pre-registration:** writing down the test and decision rules before seeing the results.
- **Development / evaluation splits:** random train-test splits used for tuning versus those kept for the final report.
- **Ratio of errors:** F's error divided by the comparison's error; below 1 is better for F.
- **CatBoost, TabM, MLP:** strong standard models used for comparison (boosted trees; a tabular neural network ensemble; a plain neural network).
