# Random Forest versus ExtraTrees on identical pair formulations

November 27, 2026. Owner-requested local study `ar3`, protocol `rf_v1`.

Test whether ordinary Random Forest can match the pairwise ExtraTrees alternatives. Use the same 14 datasets and seeds 40/41/42 as `ar1` and `ar2`. Load their completed frozen NN-kNN contexts; never retrain retrieval or completed candidates.

**Training scheme: sequential, frozen retrieval.** Forest fitting uses saved NumPy arrays after retrieval pretraining. It does not update retrieval parameters, representations, neighborhood membership or base activations. `ar3` contains no joint or alternating forest results. The later [ar4 continuation study](ADAPTER_FOREST_ALTERNATING_EXPERIMENT.md) tests alternating training separately; original `ar3` configurations and scores are unchanged. “Mixed pairs” below describes pair sampling only. [Scheme explanation](../reports/NNCDH_ALTERNATIVES_REPORT_2026-11-27.md#training-scheme-sequential-with-retrieval-frozen).

Two families use exactly the existing pair builders, features, targets, sample weights, reversed-pair training and final aggregation:

- Mixed pairs: half the mass on actual retrieved neighbors and half on eight seeded random non-self training cases, matching `ar1`.
- Retrieved-only pairs: all mass on actual positive-weight non-self neighbors, matching `ar2`.

The only estimator change is `ExtraTreesClassifier/Regressor` to `RandomForestClassifier/Regressor`. Both use 64 trees, depth at most 12, max_features 0.7, seed equal to the dataset seed, and a two-candidate minimum-leaf search of 5/20. The tree budget is fixed, with recorded progress/resume every 16 trees. Random Forest uses its standard `bootstrap=True`; previous ExtraTrees uses `bootstrap=False`. This compares the two standard recipes together with their bootstrap difference, not split-threshold randomization alone. Equal tree budgets do not imply equal runtime.

Classification still learns symmetric same-class affinity and redistributes probability using training class priors. Regression still learns antisymmetric target differences. Inference retains the exact original retrieved neighbors and normalized activations. Query targets are never prediction inputs.

Select leaf size by validation error, then validation final loss, then candidate ID. Only the chosen candidate of each family is evaluated on test. There are 168 validation candidates and 84 selected-family test units (252 new units); all training/evaluation data and weights go to `NNKNN_RUNS/t1/ar3/rf_v1/`. Pair-array SHA-256 digests record identical pair construction across leaf settings. Existing results remain unchanged and are read from run units.

The merged output adds both families and one expanded validation policy to the previous 16 methods: 19 methods x 42 splits = 798 comparison rows. The expanded policy appends mixed Random Forest, then retrieved-only Random Forest, to the existing `ar2` validation-error tie order. The original policy is retained as a separate historical row. Direct prediction remains excluded from adapter selection. Selection never consults test metrics.

Report equal-dataset mean classification error and standardized regression RMSE; paired dataset and seed wins/ties/losses against matching ExtraTrees and established references; correction direction, harmful overshoot and class-flip counts; and the classification sensitivity excluding Zebra/Zebra Special. These are exploratory results on repeatedly used local splits, not fresh confirmation or equal-compute tuning.

Use `run_pair_random_forest_experiment.py --study ar3 --workers 6`, then the same launcher with `--mode summarize`. Generated outputs belong under the T1 worktree's `results/adaptation_improvement/ar3/`; update the existing alternatives report in `reports/` and project evidence index.
