# Query-aware original adapter: matched local comparison

November 25, 2026. Study `aq1`, experiment `query_v1`. Protocol declared before test results. The owner requested query access for the original adapter, without further magnitude/direction or gating experiments. Local execution only: the same 14 available datasets and seeds 40, 41, 42.

## Architecture and hypothesis

Original inputs: `[z_query - weighted_mean(z_cases), retrieval_prediction]`.
Query-aware inputs: `[z_query - weighted_mean(z_cases), retrieval_prediction, z_query]`.

The hidden layers, activation functions, output dimensionality, and residual losses stay unchanged. Regression outputs an unrestricted delta; classification retains the original tanh-bounded additive class-score residual. The final prediction remains `retrieval + delta`. New input connections start at zero, preserving the original initialization mathematically and leaving its RNG stream unchanged. This adds `query_dimension * first_hidden_width` trainable parameters; it is not a parameter-count-matched comparison.

`adapter_design="query_aware"` is opt-in. This explicitly relaxes the historical input restriction. Represented numeric inputs are supported; uncovered nominal fields and classification logit-residual mode are rejected. No magnitude, direction, or confidence head is added.

Residual output alone does **not** prevent retrieval bypass. With query and retrieval inputs, the adapter may learn `delta = f(query) - retrieval`. We will measure retrieval sensitivity with fixed-model interventions, without interpreting sensitivity as a complete proof of case faithfulness.

## Paired NN-kNN experiment

Reuse the recorded pretrained core and optimizer from each `am1` unit. The already completed original full model and retrieval-only runs are baselines and are never retrained. Fit a new query-aware adapter from the matched original initialization, retaining the original F configuration, loss coefficients, free radius, retrieval learning-rate multiplier, maximum epochs, and validation criterion. Thirteen tasks use their original joint retrieval/adapter training (120 epochs maximum with five adapter warm-up epochs); Diabetes retains frozen retrieval with its original 300-epoch reuse budget. Do not continue from the old final adapter or use the gating studies' shorter continuation recipe.

This comparison permits retrieval to change according to the original schedule. Report the final model's own retrieval-only performance separately from the standalone pretrained retrieval baseline. That distinguishes improved retrieval from the correction's immediate contribution. Stored data fingerprints and targets must match `am1`. Training queries exclude self-retrieval by stable case ID. Validation chooses checkpoints; test labels are never used for training, selection, or intervention choices.

All new training is recorded and resumable, with checkpoints and predictions under `runs/t1/aq1`. Complete units are reused. Shared NN-kNN core and published Table 1 code stay unchanged.

## New matched local conventional baselines

The historical MLP/TabM/CatBoost/kNN weights and selected grid configurations are unavailable locally. Fit new compact local baselines on exactly the same already-preprocessed splits; do not label them reproductions of the historical tuned winners. Candidate families:

| Family | Candidates fixed before results |
| --- | --- |
| MLP | Width 64 or 128, two ReLU hidden layers, dropout 0, learning rate .001, weight decay 0 |
| TabM | Block width 64 or 128, two blocks, eight ensemble members, learning rate .001, weight decay 0 |
| CatBoost | Depth 4 or 8, learning rate .1, L2 leaf regularization 3, random strength 1, 300 iterations maximum |
| kNN | 3, 5, 10 or 20 neighbors, inverse-distance weighting, Euclidean distance |

Neural baselines use the existing baseline architecture/loss conventions: Adam, batch size 64, maximum 120 epochs, patience 30, and validation-loss checkpointing. Each TabM member receives its own training loss; inference averages member probabilities/predictions. CatBoost uses its validation set for early stopping. Within each task/seed, choose a family candidate by validation error, then validation loss, then candidate ID. Test is evaluated only after candidate selection. This search is smaller than the historical NN-kNN tuning and does not establish an equal-search-budget ranking or a universal neural-performance ceiling.

## Retrieval-dependence diagnostics

At fixed trained model weights, keep each query representation fixed and deterministically rotate the retrieved aggregate `(z_bar, retrieval_prediction)` across test queries by one row. Recompute the difference and residual with this substituted evidence. Apply exactly the same intervention to the old and query-aware adapters, each using its own selected core. Record prediction sensitivity and error change. Also zero only the new query input, and separately zero the retrieval context (`z_bar=0`, uniform class prediction or zero standardized regression prediction). These are artificial inference interventions, not retrained ablations or candidate models. They may be out of distribution; preserved accuracy alone cannot prove that training ignored cases.

For regression record final-output RMS change relative to the induced retrieval-prediction RMS change; small sensitivity can be consistent with cancellation. For classification report class changes and centered score changes, removing the arbitrary common score offset. None of these diagnostics is used to choose a checkpoint or algorithm.

## Reporting and validation

Report all task/seed results, equal-task family means, geometric mean ratios, wins/ties/losses, adaptation fixes/harms, and the intervention diagnostics. Identify missing units rather than treating them as zero. Preserve zero-error exclusions in ratio summaries. Tests cover initial function/RNG preservation, query gradients, schedule integration, frozen-core reuse, inference routing and interruption recovery. Verify saved predictions, metrics and source identity. Results remain exploratory on previously inspected seeds, not independent confirmation.
