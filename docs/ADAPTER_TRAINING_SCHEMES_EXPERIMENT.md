# Sequential, alternating and joint adapter training

November 26, 2026. Study `as1`, experiment `schemes_v1`. Protocol fixed before new test results. Compare the original residual, query-aware with prediction, and query-aware without prediction adapters under all three training schemes, on the same 14 local datasets and seeds 40-42. Continue the owner's explicit local-only instruction; no r760 access is assumed.

## Controlled comparison

Every model starts from the same recorded `am1` pretrained retrieval core and Adam state for its dataset/seed, with a fresh adapter. Input definitions and initialization follow `aq1`/`aq2`; no new model, gate, magnitude restriction or hyperparameter search is added. Use the recorded `aq1` preprocessed splits and stable-ID leave-one-out training neighborhoods.

| Scheme | Updates after retrieval pretraining |
| --- | --- |
| Sequential | Freeze the pretrained retrieval core permanently; train only the adapter. All three adapters have exactly the same retrieval parameters and predictions. |
| Alternating | After the common adapter warm-up, perform a complete retrieval pass with adapter frozen, then a complete adapter pass with retrieval frozen. Both passes use the same training permutation. Validate after the pair. |
| Joint | After the common adapter warm-up, update retrieval and adapter together on each minibatch, as in the existing runs. |

For the 13 tasks with existing joint configurations, retain all selected F coefficients, rates, dropout, free radius, score scale, gradient clipping and checkpoint criterion. Retain five adapter-only warm-up cycles, a maximum of 120 cycles and patience 30. Each cycle gives one adapter pass; after warm-up, alternating/joint each give one retrieval pass as well. Thus maximum adapter updates match all three schemes and maximum retrieval updates match alternating/joint. Alternating costs an extra forward pass and is not equal wall-clock compute. Early stopping makes executed and selected counts differ; report them.

The legacy `independent` schedule includes additional retrieval training, and legacy alternating uses different component losses when the selected joint objective is `summed`. Do not use those defaults for this controlled experiment. Add an explicit opt-in execution override while leaving historical schedules unchanged.

## Match the effective objective, not just coefficient names

Let `L_R = w_pre*L_pre + w_post*L_post + w_near*L_near` and `L_A = v_post*L_post + v_delta*L_delta + v_small*L_small`. All warm-up passes use `L_A`, matching the original joint trainer.

After warm-up, if the original joint objective is `summed`, both components optimize `L_R + L_A`; sequential applies this objective only to the adapter, and alternating applies it to the currently unfrozen component. In particular, the adapter's final-loss coefficient is `w_post + v_post`, not merely `v_post`. With the original `separated` objective, retrieval uses `L_R` and the adapter uses `L_A` under every scheme. Validation still selects by the original `L_post` criterion. Frozen retrieval terms can remain constants in the sequential objective.

## Diabetes exception and completed-unit reuse

Diabetes previously used frozen reuse, not joint training. Reuse its nine completed sequential units across the three adapters. For its new alternating/joint conditions, map the existing combined reuse loss to `v_post=lambda_cls`, `v_delta=lambda_diff`, `v_small=0`, use a separated objective and retrieval objective `L_pre + L_post` (`w_pre=w_post=1`, `w_near=0`). Keep the existing adapter rate .001, width, batch size 64, 300-cycle maximum and patience 30, with no adapter warm-up. Retrieval uses the pretrained core's existing learning rates (multiplier 1); no free-radius penalty is introduced. These new retrieval settings are a declared exploratory extension, not a historically tuned Diabetes configuration. The old frozen trainer selects combined validation loss; the new synchronized trainer selects validation task loss, which is proportional for this additive-MSE setting up to numerical roundoff/tie tolerance.

Reuse all 117 completed joint runs for the other 13 tasks. No complete model is trained again merely to move it into this study. Planned comparison: **378 adapter/scheme/dataset/seed cells, 126 existing reference cells and 252 new model units**. All conventional baseline models remain the completed compact `aq1` models.

## Recording, evaluation and reporting

Every new unit records curves, exact resume state, selected model/optimizer weights, validation/test predictions, core-state hashes, starting predictions and actual component update counts under `runs/t1/as1/schemes_v1/`. Verify the sequential final core is byte-identical to its starting core and that its predictions match across adapters. Configuration/data identity checks govern reuse; code fingerprints remain provenance.

Report all nine model/scheme results, paired task/seed errors, equal-task arithmetic means, geometric mean ratios and wins/ties/losses with explicit zero-error exclusions. Separate the selected core's own prediction from the applied correction; include harmful classification flips and regression direction/overshoot diagnostics. Keep extreme errors in the primary comparison. Report both complete experiment coverage and early-stopping/update counts.

Also report a predeclared validation-selected scheme for each adapter/dataset/seed: minimize validation task loss, then validation error, then prefer sequential, alternating, joint in an exact tie. Read the selected model's test metric only for reporting. Do not choose a schedule from its test performance or change the default based on descriptive test winners alone. Comparisons with compact local neural baselines retain their unequal-tuning-budget limitation. These previously inspected seeds are exploratory, not independent confirmation.

Generated summaries and figures read run units only and go under `results/adaptation_improvement/as1/`. The handwritten report goes under `reports/`. Synthetic tests cover loss matching, frozen state, per-component update counts, schedule integration and exact interruption recovery. Shared NN-kNN core and Table 1 remain unchanged.
