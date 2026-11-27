# Dropout across residual adapters and training schemes

November 26, 2026. Study `ad1`, experiment `drop_v1`. Protocol declared before new dropout results. Continue the owner's local-only instruction and the existing Markdown research workflow.

## Scope and prior leaders

Compare the three current residual adapters: original, query-aware with prediction, and query-aware without prediction. Each is evaluated under sequential, alternating and joint training on the same 14 available datasets and seeds 40–42. The shelved magnitude/direction design and earlier gate experiments remain historical references; this matrix concerns the three active MLP input designs.

Before this study, the lowest equal-task classification error was query + prediction with validation-selected training scheme (8.745%). The lowest arithmetic regression RMSE among all earlier adapter experiments was the probability gate (0.337206); the validation disagreement gate was 0.337424 and had a more favorable geometric comparison. Among the three ungated residual designs, the previous original mixed schedule was 0.338910, and all-joint original was 0.340452. These are descriptive local results, not proof of a universally best model.

## One architectural change

Apply independent inverted dropout after **both hidden ReLU activations**, with rates `p=0.1` and `p=0.2`. Classification retains its LayerNorm layers and tanh residual; regression retains its unbounded linear residual. Keep widths 64/32, output design, losses, target preprocessing, optimizer settings, and retrieval structure unchanged. Dropout has no trainable parameters. Wrapping the parameterless ReLU modules preserves Linear/LayerNorm checkpoint keys, initial parameter values and initialization RNG state. `p=0` adds no module and remains an exact no-op.

The historical classification constructor's optional first-hidden-only dropout remains compatible. This study explicitly uses `dropout_placement="all_hidden"`; it is recorded in unit configuration, checkpoint description and resume state. Add residual regression support only in the T1 workflow. Shared core and Table 1 are unchanged.

Dropout is active whenever the adapter is trained, including warm-up. Validation/test predictions use `eval()` with dropout off; no Monte Carlo dropout is used. In alternating training the frozen adapter is in eval mode during the retrieval pass, then train mode during its own pass, matching the existing schedule's mode handling. Joint training uses the stochastic adapter while updating both components. Thus dropout affects retrieval gradients in joint training differently from alternating, an intended property of these existing schemes.

## Matched training and reuse

Use the recorded `aq1` data splits and the same `am1` pretrained retrieval models and Adam states as `as1`. All adapters begin fresh. Stable-ID leave-one-out retrieval applies to training queries. Sequential freezes the shared core permanently. Alternating uses one R pass then one A pass per cycle, sharing the permutation. Joint updates both on each minibatch. Preserve the effective summed/separated losses and maximum component budgets from `as1`.

For the 13 original joint tasks, retain five adapter-only warm-up passes, maximum 120 cycles and patience 30. Diabetes sequential uses the original frozen reuse trainer, combined loss, rate .001, 300-epoch maximum and patience 30; its alternating/joint settings remain the declared `as1` separated extension without warm-up. This preserves the appropriate zero-dropout reference for every cell.

Reuse all **378 zero-dropout comparison cells** from `as1` and its `am1`/`aq1`/`aq2` references. Train **756 new models**, covering two positive rates × three adapters × three schemes × 14 tasks × three seeds. Total comparison coverage is **1,134 cells**. Complete units are never retrained; incomplete units resume with model, optimizer, shuffle RNG and dropout RNG restored.

## Predeclared selection and analysis

Report fixed-rate `p=0`, `.1` and `.2` results for every adapter/scheme, equal-task means after seed averaging, paired task and seed wins/ties/losses, and geometric ratios with explicit zero-error exclusions. Do not remove extreme errors.

Primary selection: for each adapter/scheme/dataset/seed, select rate using validation task loss, then validation error, then the smaller dropout rate in an exact tie. Compare its test result with the matched zero-dropout scheme. Secondary selection: for each adapter/dataset/seed, choose rate and scheme jointly by validation task loss, validation error, smaller rate, then sequential/alternating/joint tie order. Compare with `as1`'s validation-selected scheme at `p=0`. Neither test metrics nor observed test outliers choose rates, schemes, checkpoints or changes to this protocol.

Inspect the chosen core's own retrieval versus final prediction, classification harmful flips, and regression wrong-direction corrections and harmful overshoots. Retain the largest harmful cases and their share of adaptation harm. Verify frozen-core hashes and sequential predictions across all inputs/rates, unchanged trainable parameter counts, deterministic evaluation, two active dropout layers per new model, exact interrupted-run recovery and completed-unit reuse.

Rates are a compact validation search, not evidence that .1 or .2 is globally optimal. The larger selection space can overfit validation; previously inspected splits, three seeds, local runtime and the untuned Diabetes extension remain limitations. Summaries read only run units and write to `results/adaptation_improvement/ad1/`; the agent-authored report goes to `reports/`, and runtime logs to the mini-project `logs/ad1/`.

Completion note, November 26, 2026: all 756 new models and 378 references are complete. See the [results and behavior report](../reports/ADAPTER_DROPOUT_LOCAL_2026-11-26.md). The protocol above was not revised in response to the results.
