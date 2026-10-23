# Case-base maintenance design review, 2026-10-19

Question (user, 2026-10-19): is the case-based maintenance (CBM) mechanism designed properly, and has it
been tried on difficult datasets? This review read the code, the PI specification of 2026-09-20 and its
handoff documents, every earlier CBM result, and the CBM literature. Each candidate flaw was then checked by an
independent reviewer who tried to refute it. Working notes are outside the repository; the evidence below is
reproducible from the files cited.

## Verdict

The engineering is sound: stable case IDs, exact archive and restore, Adam-row realignment, recomputation after
each removal in removal influence, and the guard that keeps validation and test labels out of maintenance
decisions. The decision mechanism is not:

- the configured pipeline performs one-shot pruning of rarely retrieved cases;
- the scores come from the same training queries the model was fitted to;
- CBM has never been run on data where it has something to fix.

"Removal gives no consistent gain" is therefore the expected outcome of the current setup, not evidence about
maintenance. The confirmed final configuration F contains no maintenance, so none of this affects the
confirmation (steps D and F).

## Flaws (all thirteen survived the refutation check; severities as corrected by it)

| ID | Kind | Severity | Finding | Evidence |
|---|---|---|---|---|
| F1 | design | high | Q measures exposure, not quality. Every case in a query's top-k gets that query's final outcome regardless of its own label; with H = 0, Q = (A + 1)/(A + 2) only reorders cases by activation mass A. The stored-label Q that the plan requires as a baseline (PLAN:294) is missing. | `model/t1/provenance.py:276-279`, `:204`. Synthetic set (15% flipped labels): Spearman(Q, A) 0.86–0.99; Q keeps 58–60% of corrupted cases at K = 0.75, random 75%; the superseded label-agreement Q kept 0.0–0.6% |
| F2 | evaluation | high | CBM has never been evaluated where it can matter: label noise, redundancy, imbalance, a binding budget on a large case base, or the hard tasks. | All aligned CBM results use 18–60-case Iris/Energy/Yacht/Wine memories, Digits near 0.98, and one 432-case synthetic set with 3–5 seeds |
| F3 | design | medium | Unevidenced cases score exactly 0.5 and ties go to ascending case ID; synthetic rows are ordered by class and truth group, so evictions follow row order. | `model/t1/retention.py:98-99, 147`. Synthetic: 99–108 of 108 evictions tied; per-class kept 0.85/0.75/0.66 vs 0.74/0.75/0.76 under random tie-breaks |
| F4 | design | medium | On tuned cores most cases are never retrieved, so K = 0.5 is not a binding budget on those tasks and exposure-correlated policies delete inert cases. | Table 1: never-retrieved share 0.98 (abalone), 0.84 (breast cancer), 0.77 (wine), 0.66 (yacht) |
| F5 | design | medium | Q, coverage, removal influence and the loss budget are computed on train-LOO queries the core and adapter were fitted to, so harmful support H is near 0 on easy tasks. | `configs/t1/_base.yaml` (`maint_frac: 0.0`, `source: train_loo`); `tools/t1_run.py:144-169` |
| F6 | evaluation | medium | Step G as pre-registered fixes thresholds, policy, comparators and K but not the runner, tasks, endpoint, test direction or a full-memory/exposure comparator. | `docs/T1_PREREG_20261013.md` (Retention); `configs/t1/prereg_20261013.json` `retention` |
| F7 | bug | medium | The in-training maintenance hook ignored per-condition policy, K and `maintenance_epochs`, so one config could not compare policies on an `_rrr` schedule. | `tools/t1_run.py` (hook called `retention_cfg_for({}, ...)`). **Fixed** (below) |
| F8 | design | medium | Selection fills K from a per-case standalone ranking; the combined coverage/redundancy check the plan requires is missing and protection is two cases per class. | `model/t1/retention.py:118-126`; PLAN:371, 393 |
| F9 | design | medium | "Dynamic" maintenance prunes once: K is fixed from the initial count, archived cases never compete again, every later hook keeps everything, and archived cases still act as training queries. | `tools/t1_run.py:901-907`; `model/t1/retention.py:112`; `model/t1/maintenance.py:141` |
| F10 | evaluation | low | RRR-vs-RR comparisons compress the comparators after training without fine-tuning, so E5/E17 differences are not maintenance effects (already conceded in the project docs). | `tools/t1_run.py:932-955` |
| F11 | design | low | `coverage_reachability` with `zero_reachability: zero` first deletes every case whose own leave-one-out query fails (Wilson editing), then ranks by exposure. | `model/t1/candidates.py:136` |
| F12 | evaluation | medium | No classical instance-selection or data-valuation baselines (CNN, ENN, ICF, DROP3, RC-CNN, KNN-Shapley) and no size or cost axis. | `model/t1/retention.py:12-14` |
| F13 | bug | low | `bias_source: delta_from_init` measures change since the first audit, so in post-hoc use it ranks by case ID (latent: no config uses it). | `model/t1/provenance.py` |

A further defect found during the fixes: `run_maintenance` read the kept case IDs from a CPU view of the case-ID
buffer after compaction overwrote it, so MCB selection churn was logged against wrong IDs. Selections and
models were unaffected. **Fixed** (below).

## Table 1. Development-seed diagnostics at K = 0.5 (no training)

`tools/t1_cbm_diagnostics.py` over `results/t1/p4_component_sync/pi20260920/tuning_h6_20261013`, condition
`tuned__alternating_default`, mean over seeds 20–25. Full tables for all 19 conditions:
`results/t1/p1_supervised_retention/cbm_review_20261019/diagnostics_tuning_h6_dev_K0.5.md`.

| task | n | never retrieved | top-10% activation share | Spearman(Q, A) | evidenced with H = 0 | Q evictions decided by case ID | Jaccard Q vs exposure-only (retrieved cases) | activation archived by Q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| abalone | 2505 | 0.98 | 1.00 | 0.64 | 0.02 | 0.99 | 0.78 | 0.072 |
| airfoil | 901 | 0.03 | 0.35 | 0.99 | 0.94 | 0.00 | 0.97 | 0.158 |
| balance | 375 | 0.13 | 0.36 | 0.99 | 0.93 | 0.00 | 0.97 | 0.124 |
| body_fat | 150 | 0.20 | 0.45 | 0.55 | 0.11 | 0.10 | 0.66 | 0.153 |
| breast_cancer | 341 | 0.84 | 0.90 | 1.00 | 0.97 | 1.00 | 0.91 | 0.008 |
| diabets | 264 | 0.17 | 0.45 | 0.13 | 0.09 | 0.42 | 0.47 | 0.378 |
| digits | 1077 | 0.18 | 0.40 | 1.00 | 0.99 | 0.00 | 1.00 | 0.068 |
| energy_efficiency | 460 | 0.15 | 0.23 | 1.00 | 1.00 | 0.00 | 1.00 | 0.266 |
| iris | 90 | 0.45 | 0.56 | 0.92 | 0.86 | 0.50 | 0.93 | 0.095 |
| student_performance | 389 | 0.35 | 0.60 | 0.96 | 0.65 | 0.00 | 0.94 | 0.023 |
| anneal (363614) | 448 | 0.57 | 0.73 | 1.00 | 0.92 | 0.75 | 0.95 | 0.010 |
| concrete (363625) | 514 | 0.13 | 0.38 | 0.95 | 0.63 | 0.00 | 0.88 | 0.123 |
| credit-g (363626) | 499 | 0.12 | 0.43 | 0.90 | 0.47 | 0.00 | 0.83 | 0.101 |
| insurance (363675) | 669 | 0.18 | 0.49 | 0.93 | 0.44 | 0.00 | 0.85 | 0.092 |
| wine | 106 | 0.77 | 0.88 | 0.97 | 0.89 | 1.00 | 0.88 | 0.008 |
| yacht | 184 | 0.66 | 0.71 | 0.92 | 0.92 | 1.00 | 0.88 | 0.016 |
| zebra | 66 | 0.11 | 0.25 | 1.00 | 1.00 | 0.00 | 1.00 | 0.208 |
| zebra_special | 132 | 0.03 | 0.27 | 0.51 | 0.35 | 0.00 | 0.60 | 0.352 |

Reading: on 14 of 18 tasks Q ranks cases almost exactly by exposure (Spearman ≥ 0.90), and on 15 of 18 its kept
set among retrieved cases overlaps the exposure-only control with Jaccard ≥ 0.78. On abalone, breast cancer,
wine, yacht and anneal most evictions are decided by the case-ID tie-break among never-retrieved cases, which hold
almost no activation (archived share ≤ 0.07). Only diabetes, zebra_special and body_fat carry outcome
information beyond exposure (Spearman ≤ 0.55). These are context for step G, not grounds to drop tasks.

## Changes made (defaults unchanged; step G's frozen behaviour is the default)

| Change | Default | Check |
|---|---|---|
| `RetentionConfig.tie_break = "seeded_random"` breaks exact score ties with a permutation of the stable IDs drawn from the run seed, in every selection path and in sequential removal (F3) | `case_id`, bit-identical to before (snapshot test over all nine non-removal policies, synchronous maintenance and sequential removal) | 42 tests; the two retrained-removal loops still choose removals by case ID (documented) |
| `run_maintenance` copies the case IDs before compaction | — | regression test |
| A sync condition's retention fields, `K` and `maintenance_epochs` apply to that condition; unknown keys and maintenance keys on schedules without in-training maintenance raise (F7) | all 63 existing sync configs (468 conditions) resolve to their earlier settings | 8 tests; 108 of 114 artifacts byte-identical in a before/after run, the other 6 differ only by a new manifest field |
| `tools/t1_cbm_diagnostics.py` (Table 1) | read-only | 7 tests on fixtures with hand-computed answers |

## Not changed, and why

- **The definition of Q** comes from the PI specification of 2026-09-20 (credit from the final adapted outcome,
  replacing label agreement). Changing it is a PI decision; label-aware or counterfactual credit is proposed below
  as exploratory arms, not as a replacement.
- **Step G** is pre-registered with frozen thresholds, policy, comparators and K. Its unspecified details
  (runner, tasks, seeds, endpoint, test, secondary outputs) can be fixed only by a dated addendum written before
  any step-G result.

## Proposed next steps (decisions for the PI)

1. **Step G addendum.** Run step G exactly as frozen, after an addendum that names:
   - the runner: post-hoc compression of F's selected checkpoint, with an equal continued-training budget for
     every arm, full memory included;
   - tasks and seeds: as in the confirmation;
   - the endpoint: log error ratio, REML with Hartung–Knapp, one-sided superiority, Holm over the two
     comparisons;
   - secondary outputs outside the Holm family: full-memory non-inferiority, an exposure-only control, the
     tie-decided share, kept-set Jaccard and hub statistics;
   - a clause that "Q beats random" alone is not evidence of case utility.
2. **Exploratory CBM study on development seeds 20–25.** Legacy primary, synthetic and external data only; the
   pre-registration forbids tuning on transfer tasks.
   - Data:
     - synthetic data with shuffled row IDs, label noise at 0/10/20/30%, near-duplicates, a rare subgroup and
       imbalance;
     - hard legacy tasks (credit-g, insurance, zebra_special, uncapped abalone, digits, balance) with injected
       training-label noise, duplicates and imbalance;
     - larger external sets (Covertype subsamples, letter, adult).
   - Policies:
     - reference: full memory, random, stratified;
     - registered: Q, B;
     - controls: exposure-only, drop never-retrieved;
     - credit variants: stored-label Q, BBNR-style blame, removal influence;
     - classical baselines in the learned metric and in raw space: ENN, RENN, CNN, ICF, DROP3, RC-CNN,
       KNN-Shapley.
   - Capacity curves (K = 0.05 to 0.75), train-LOO vs cross-fitted audits, and seeded tie-breaks.
   - Metrics: clean test error, the area under the error-versus-size curve, noise-removal precision and recall,
     duplicate and rare-group retention, and latency.
3. **Dynamic maintenance** (insertion and re-admission, excluding archived cases from training queries) only
   after the static study shows which credit signal works.
