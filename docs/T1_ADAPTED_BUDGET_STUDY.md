# Adapted deletion: budget and capacity follow-up

The 18-case Iris prototype can compress to 12 cases while mean test accuracy
declines. Its mixed scope budgets also allow alternating to receive twice the
phase epochs of one-role continuations. A broader controlled comparison is
needed before treating training-loss acceptance as predictive reliability.

`alignment_adapted_budget_grid.yaml` declares two datasets (Iris and Wine),
three seeds, 36 initial training cases, target K=.5/.75, three cumulative loss
budgets (0/.03/.1), and three continuation scopes. Every scope gets two total
phase epochs per candidate: two retrieval, two adapter, or one each. This gives
108 exploratory runs, equal total optimizer update counts per same dataset,
seed and accepted block, while retaining component counts and actual retrieval
cost. It is not a matched-FLOP comparison.

Each condition preserves fixed training queries, cohort floors, original-loss
acceptance, same-capacity random deletion and full-memory accepted-update
controls. Strict loss budgets may stop before target K; capacity failure is an
outcome to report, not a reason to loosen a threshold after seeing test results.
Random controls match achieved capacity and updates and report budget violations.
The same selected starting model/adapter is shared across all conditions for a
dataset/seed. Test outcomes never select thresholds, deletion candidates or
continuation checkpoints.

Compared with the first pilot, initial sample count and training-only
standardization differ; cross-pilot differences cannot isolate continuation
budget alone. Within this grid, paired conditions share data and initial state.
The raw split snapshots, source ZIP and model/adapter/Adam checkpoints are the
comparison boundary. No threshold is presented as PI-selected or confirmatory.

The next analysis should separate capacity achieved, reference loss, test
accuracy/flip changes, negative transfer, candidate/control/accepted update
costs and seed uncertainty. Test whether extra continuation actually changes
accepted IDs, whether a stricter budget protects holdout performance, and whether
random controls violate reference constraints despite good test outcomes.
If choices need refinement, use a separate exploratory or maintenance-only
protocol; preserve this frozen matrix and its failures.

Completed frozen source
`a1a0c6f4fc89249db0a5dfcdc8e2bcfa56d73bd515fef8890672ddaf611ceb4c`,
111 files, ZIP SHA256
`3b8a1ab2825c71b7136313e1f97bb5cb0d16c5ad6efcaa24d95af263e71ac4ad`.
All108 runs/432 states load safely and3,564 final events replay. Independent
audit actually reruns all41,775 candidate trainings and1,439 per-round full
controls, every selected path and both matched trajectories. Final model,
adapter and Adam tensors match exactly; eligible candidates, reference
denominators, loss comparisons, floors and actual updates pass.

Two zero-budget Wine seed9 retrieval-only conditions stop at the loss budget
before their target capacities (K18 and K27). Their achieved capacities remain
the random-control matching targets; no forced removal follows rejection.
Adaptive improves/worsens/ties test accuracy in48/21/39 of108 runs relative to
initial,35/21/52 relative to matched full, and45/30/33 relative to matched random.
Forty random runs violate the adaptive reference budget. Three budgets and two
capacities are repeated correlated conditions, not108 independent replicates.
Paired seed tables/ranges are descriptive; no generalization guarantee or
confirmed performance benefit follows from the training-loss acceptance rule.
