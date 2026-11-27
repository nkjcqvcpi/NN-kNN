# Query-aware NN-CDH without the retrieved prediction

November 25, 2026. Study `aq2`, experiment `nopred_v1`. Protocol fixed before new test results. This follows the owner's request to remove the retrieved prediction from the adapter input. Keep query access from `aq1`; no gate or magnitude/direction variant is included.

## Change and interpretation

```text
aq1 residual network input: [z_query - z_bar, retrieved_prediction, z_query]
aq2 residual network input: [z_query - z_bar, z_query]
final output: retrieved_prediction + residual_network(input)
```

Remove the entire prediction block: one scalar for regression, all class-probability entries for classification. The prediction remains available for residual targets, losses, diagnostics and final addition, but is never concatenated into the residual network or hidden activations. Classification retains the original tanh score correction; regression retains its unrestricted delta. The design flag is `query_aware_no_prediction`.

Construct the same initial network as `aq1`, then delete only its first-layer prediction columns. Retain all other initialized weights, biases, zero query columns and RNG state. Initial outputs now equal the initial `aq1` network with its prediction input zeroed, not its normal outputs. Parameter count falls by `prediction_dimension * first_hidden_width`; this change is part of removing the input. Do not zero-mask unused trainable columns or alter downstream initialization.

At fixed difference/query inputs and fixed weights, changing only the retrieved prediction must leave the residual unchanged and change the final output by exactly the same amount as the retrieval change (within numerical tolerance). This removes direct prediction subtraction. Correlations with query/neighborhood features, and changes to retrieval during joint training, still prevent a general guarantee against implicitly approximating or compensating for retrieval.

## Matched local experiment

Run all 14 available datasets with seeds 40, 41, 42. Reuse the exact recorded `am1` pretrained cores/Adam states and `aq1` cached splits. Retain original selected F configurations, losses, learning rates, free-radius settings, validation selection and budgets: thirteen joint-training tasks (maximum 120 epochs, patience 30, five adapter warm-up epochs) and frozen-retrieval Diabetes (maximum 300 epochs, patience 30). Train a fresh new adapter; do not continue an existing final checkpoint.

Reuse completed original full NN-kNN, query-aware NN-kNN, standalone pretrained retrieval and compact local MLP/TabM/CatBoost/kNN models. No baseline is retrained. The compact neural baseline search remains smaller than the historical search and starts from scratch, so it cannot establish equal-budget neural parity. Training queries exclude self-retrieval by stable case ID. Test results never select settings/checkpoints.

Record 42 new model units under `runs/t1/aq2/nopred_v1`, with mandatory epoch histories, resumable model/optimizer/RNG states, selected weights, validation/test predictions and diagnostics. Complete units must load without fitting. Source config identities and data fingerprints are checked. Summary scripts read only recorded units and write generated evidence under `results/adaptation_improvement/aq2`; the handwritten report belongs under `reports/`.

## Diagnostics and reporting

Repeat `aq1` fixed-model tests: rotate the aggregate feature/prediction context, zero that context, and zero only the added query channel. Add a prediction-only rotation with query and difference inputs fixed; check residual invariance and final-output translation for every test query. These interventions are fixed in advance, potentially out of distribution and are never used for model selection.

Report per-seed/task errors, equal-task family means, geometric mean ratios with explicit zero exclusions, wins/ties/losses against both prior adapters and conventional baselines, the selected core's own pre/post error, correction fix/harm counts and intervention sensitivity. Preserve large failures. Results on these repeatedly inspected splits are exploratory.

Verify the architecture has no prediction input columns or gradient path into the residual; test matched remaining initialization, all training schedules, frozen reuse, prediction/event routing and exact interruption recovery. Reconcile saved metrics with predictions, source targets and the prediction-only invariance check before reporting.
