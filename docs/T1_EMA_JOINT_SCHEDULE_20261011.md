# EMA ablation, joint training and schedule search, October 10–11, 2026

This note answers two collaborator questions about T1 and records the follow-up
training-schedule search.

1. "Is EMA in use, with one encoder for queries and one for the case base? First
   test whether EMA helps NN-kNN with (1) retrieval only and (2) retrieval +
   reuse, against the earlier NN-kNN without EMA, before comparing case-base
   maintenance."
2. "Is there really no scheme that trains retrieval and reuse end to end at the
   same time, only sequential and alternating training?"

## Answers in brief

1. Yes. EMA is the Momentum Case Base (MCB) option of the shared core
   (`CoreConfig.mcb_enabled`): the online MLP encoder embeds queries and a
   no-gradient EMA copy, updated after every optimizer step, embeds stored cases.
   Before this work it had only been run together with maintenance (`p5_mcb`) or
   inside synchronization controls, never as a clean full-memory ablation. The
   new `encoder_ablation` experiment does that; results are below.
2. Correct. T1 had `independent` (retrieval, then the adapter with retrieval
   frozen) and `alternating_rr/rrr` (blocks of retrieval-only and adapter-only
   updates). This work adds `joint_rr/rrr`, which updates both components on
   every minibatch from one forward pass.

## Ownership and handoff

| Step | Workstream / branch / checkout | Files | Validation | Status |
|---|---|---|---|---|
| Joint schedules | Neural CBR / `neural-cbr-ema-joint-20261010` (from `neural-cbr-reorg-20261010`) / `neural-cbr-ema/` | `model/t1/sync.py` | Gradient-isolation, phase, maintenance and validation tests | `6a6398e`, done |
| Encoder (EMA) ablation runner | same | `tools/t1_run.py`, `configs/t1/ema_ablation_20261010.yaml`, `tools/t1_encoder_report.py` | Runner tests; observer leaves training bit-identical; 20/20 identical metrics against the pre-change runner | `6a6398e`, `6333d42`, done |
| EMA ablation runs | same | `results/t1/p5_mcb_interaction/` | 65/65 and 30/30 jobs, no failures | done |
| Tuned-core batch size | same | `tools/t1_grid_search.py`, `tools/t1_execute_study.py`, `tools/t1_schedule_search.py` | Settings test; digits seed 30 reproduces the grid's 0.956 | `7b978a4`, done; not applied to the running study |
| Schedule search | same | `tools/t1_schedule_search.py`, `tools/t1_parallel.py`, `configs/t1/schedule_search_*.yaml` | Selection, grid and determinism checks | batch-64 selection done; confirmation running |
| Shared core | none | no core file changed | `tools/check_nnknn_core.py` | unchanged |

No core file (`docs/NNKNN_CORE.md` list) was modified, so nothing needs to be
propagated to `main`, `rl` or `LLM`. The T1 full study keeps running unchanged in
`neural-cbr/`; this branch was created beside it because that study fingerprints
the checkout's source. Experiments run from detached worktrees pinned to a commit
(`neural-cbr-run-<commit>/`), so each run's recorded source fingerprint matches a
commit. Merge into `neural-cbr` after the full study ends and the reorganization
apply step in `../REORGANIZATION_20261010.md` is done; the branch fast-forwards
from `neural-cbr-reorg-20261010`.

## Joint (end-to-end) training

`SyncConfig.schedule` now accepts `joint_rr` and `joint_rrr`. Per minibatch, one
forward pass gives every loss term, and both optimizers step:

```text
L_R = w_pre L_pre + w_post L_post + w_near L_near     (plan section 6)
L_A = v_post L_post + v_delta L_delta + v_small L_small

separated (default): grad of L_A -> adapter parameters only,
                     grad of L_R -> retrieval parameters only
summed:              grad of L_R + L_A -> every trainable parameter
```

Because the step is end to end, `L_post`'s gradient reaches retrieval through the
current adapter (a test checks that the retrieval gradient equals `dL_R/dθ_R`
exactly and differs from the retrieval-only `dL_pre/dθ_R`). Options:
`adapter_warmup_epochs` trains a fresh adapter alone first; `core_pretrain: false`
(runner) skips retrieval-only core training and trains both from the untrained
initialization with a fresh optimizer and full learning rate. `joint_rrr` runs
checkpoint maintenance like `alternating_rrr`. The spec asked to start with staged
or alternating updates and to guard against one component carrying the other;
joint runs keep reporting `L_pre` and every other term separately, and the
retrieval-only output stays available.

## EMA ablation design

`configs/t1/ema_ablation_20261010.yaml`, `type: encoder_ablation`. Full memory, no
maintenance, five evaluation seeds (30–34), the 13 tasks of the synchronization
ablation (abalone capped at 1,500 training cases). Arms differ only in the
encoder; splits, seeds, initialization, budgets and retrieval rules are shared:

| Arm | Encoder |
|---|---|
| `raw` | earlier NN-kNN: raw features and learned feature weights, no encoder |
| `shared` | one MLP [32, 16] for queries and cases, L2-normalized embeddings |
| `ema090`, `ema099`, `ema0999` | online MLP for queries, EMA copy for cases (momentum 0.9/0.99/0.999 per step), same normalization |

For each arm one core is trained (up to 300 epochs, patience 40, validation
selection), and the run records

- setting 1, retrieval only: the core's pre-adaptation output;
- setting 2, retrieval + reuse on the same core: a frozen-core adapter
  (`combined` loss), `alternating_rr`, and `joint_rr`.

Core training is observed every epoch on validation probes (embedding drift and
top-k neighbor churn) through a read-only hook that never delays early stopping.
`online_ema_gap` is the mean distance between the online and EMA embeddings of
the stored cases at the end. Hyperparameters are the T1 defaults (top-k 5,
temperature 1, one feature-weight set), not each task's HPO choice, so every arm
gets the same settings. `ema_valsel` in the report picks the momentum per task
and setting by validation error only.

## Schedule search design

Stage A (`configs/t1/schedule_search_tuned_20261011.yaml`): every task starts
from its validation-selected NN-kNN core from the full-study HPO grid
(`tools/t1_schedule_search.py prepare`). 16 tasks (the 13 above without
`synthetic_diag`, plus anneal, credit-g, concrete and insurance from TabArena),
development seeds 20 and 21, 27 candidates:

- frozen-core reuse;
- sequential (`independent`) with retrieval learning-rate scale 0.1/0.3/1.0;
- alternating with block 1/2/5 epochs × scale 0.1/0.3/1.0 (block 2 at 0.1 is the
  current default);
- joint, separated or summed × warm-up 0/5 × scale 0.1/0.3/1.0;
- joint from scratch, separated or summed.

Selection uses validation only: per task, mean validation error (1 − accuracy or
standardized RMSE after adaptation), then mean validation post-adaptation loss;
globally, mean per-task rank. Stage B tunes loss weights and the adapter learning
rate around the stage-A leaders on the same seeds, ranked together with stage A.
The finalists, per-task winners and references then run on fresh seeds 30–34.

## Results

### EMA ablation at default settings (65/65 jobs)

Files: `results/t1/p5_mcb_interaction/ema_ablation_20261010/` (`encoder_report.md`
has every per-task table; CSVs and run shards beside it). Error is 1 − accuracy or
standardized RMSE; "relative" is (arm − reference) / reference per task, summarized
over the 13 tasks by its median and a Wilcoxon signed-rank test. "Significant"
counts tasks with a 5-seed paired t-test p < 0.05.

| Setting | EMA 0.9 vs shared encoder | EMA 0.99 vs shared | raw NN-kNN vs shared |
|---|---|---|---|
| Retrieval only | −6.5% median, p = 0.79; better on 8/13 tasks (1 significant) | −0.3%, p = 0.89; 7/13 | −29.0%, p = 0.013; 10/13 (7 significant) |
| Frozen-core reuse | −8.1%, p = 0.46; 8/13 (1) | −1.1%, p = 0.95; 7/13 | −31.7%, p = 0.003; 11/13 (6) |
| Alternating | −9.1%, p = 0.22; 8/13 (1) | −0.6%, p = 0.64; 7/13 | −40.6%, p = 0.002; 11/13 (5) |
| Joint | −2.8%, p = 0.74; 7/13 (2) | +0.4%, p = 0.84; 6/13 | −26.7%, p = 0.003; 10/13 (5) |

Momentum 0.999 is no better than the shared encoder in any setting. Choosing the
momentum per task on validation (`ema_valsel`) gives −2.8% to −8.1%, p ≥ 0.19.

- With the encoder held fixed, EMA gives a small, nonsignificant improvement at
  momentum 0.9 and none at higher momentum, in both settings.
- Both MLP arms are significantly worse than the earlier raw-feature NN-kNN
  (every EMA arm: median +24% to +56% relative error, p ≤ 0.017). Raw beats the
  shared encoder on 10–11 of 13 tasks and has the lowest error of all arms on
  8–10. An MLP arm is lowest on abalone, airfoil and balance (where the HPO grid
  also chose the MLP) and in some settings on iris or breast cancer.
- EMA does stabilize the stored-case geometry as designed: case-embedding drift
  per epoch falls with momentum (abalone 0.187 shared, 0.078/0.031/0.009 at
  0.9/0.99/0.999), and momentum 0.9 lowers top-k neighbor churn on validation
  probes on 12 of 13 tasks. Higher momentum does not lower churn further, because
  the query (online) and case (EMA) encoders separate: their mean embedding gap
  grows from 0.01–0.04 at momentum 0.9 to 0.08–0.44 at 0.999.

Limits: default hyperparameters (top-k 5, temperature 1, MLP [32, 16]) and small
tabular tasks trained from scratch. The MCB manuscript's evidence is on CUB images
with a pretrained backbone, where an encoder is unavoidable; that setting is not
tested here.

### EMA where the tuned core uses the MLP encoder

`configs/t1/ema_tuned_mlp_bs64_20261012.yaml` (30/30 jobs); files in
`results/t1/p5_mcb_interaction/ema_tuned_mlp_bs64_20261012/`. The six tasks where
the full-study HPO grid selected the MLP (balance, wine, anneal, abalone, airfoil,
concrete), each starting from its tuned core trained exactly as the grid trained
it (batch 64, no embedding normalization), five seeds. Arms differ only in the
EMA case encoder. Relative error is (EMA − shared) / shared per task; median and
Wilcoxon signed-rank test over the six tasks.

| Setting | EMA 0.9: better / worse tasks, median, p | EMA 0.99 | Significant per task (paired t, p < 0.05) |
|---|---|---|---|
| Retrieval only | 2 / 4, +17.0%, 0.22 | 3 / 3, +0.4%, 0.69 | worse on airfoil (both momenta) |
| Frozen-core reuse | 2 / 4, +14.3%, 0.31 | 3 / 3, +0.6%, 1.00 | better on balance (0.9) and abalone (0.99); worse on airfoil |
| Alternating | 3 / 3, +2.6%, 0.84 | 3 / 3, +0.9%, 0.69 | worse on airfoil and anneal (both) |
| Joint (summed, warm-up 5) | 1 / 5, +12.2%, 0.16 | 1 / 5, +13.1%, 0.16 | worse on airfoil and wine (0.9), abalone and airfoil (0.99) |

Balance is the task where EMA most consistently lowers error (retrieval only
0.062 → 0.048 at 0.9); airfoil is significantly worse in every setting (0.235 →
0.301). Under joint training, where the online encoder keeps moving with the
adapter, EMA is worse on five of six tasks. The earlier run of this check with
batch-32 cores (`ema_tuned_mlp_20261012`, see the batch-size section below) gave
the same picture.

### Conclusion for the EMA question

EMA is not useful as a default on these supervised tabular tasks. With retrieval
only and with retrieval + reuse, at default settings (13 tasks) and on tuned
encoder cores (6 tasks), it is never significantly better than the same encoder
without EMA across tasks; per-task gains are isolated (balance, once abalone),
and it is significantly worse on airfoil in every tuned setting. The
raw-feature NN-kNN remains the stronger default for most of these tasks. EMA's
stabilizing effect on stored-case embeddings is real, so it remains a candidate
where an encoder is required (images, text) or where the case base changes during
training; neither condition is tested here. For the next step, compare case-base
maintenance policies without EMA (raw or the tuned core), with EMA kept as an
optional factor.

### Batch-size mismatch for HPO-selected cores (found 2026-10-12)

The HPO grid trains NN-kNN candidates with batch size 64
(`tools/t1_grid_search.py`), but `tools/t1_execute_study.py` and the schedule
search rebuilt the selected cores with `configs/t1/_base.yaml`'s batch size 32.
The difference matters: on digits (selected raw core, top-k 10, temperature 0.5,
four feature-weight sets) retrieval collapses to chance in the first epoch at
batch 32 on seeds 21, 30, 31 and 34 (seed 30: validation accuracy 0.119 at batch
32 vs 0.956 at batch 64, the grid's own value). `nnknn_core_settings` in
`tools/t1_grid_search.py` now defines the grid's settings once, and the study
executor and the schedule search use it (commit `7b978a4`).

Consequence for the running full study in `neural-cbr/`: its plan stage
(`t1_execute_study.py`, queued after the grid) still uses batch 32 for every
tuned-core experiment family. It should be rerun with this fix, or its tuned-core
results labeled accordingly. The study's frozen source was not modified.

### Schedule search on batch-32 cores (recorded, superseded)

Files: `results/t1/p4_component_sync/schedule_search_20261011/`. Schedules within a
task start from identical cores, so the comparison is matched, but the cores are
not the grid's tuned ones (see above).

- Stage A (27 candidates, seeds 20 and 21, validation only): joint schedules took
  global ranks 1–9; the current default (alternating, block 2, scale 0.1) ranked
  14th and the sequential schedules 21st, 24th and 27th.
- Stage B (48 loss-weight and adapter-rate candidates around the two stage-A
  leaders) reproduced stage A exactly on all 96 repeated runs. Pooled selection
  over 73 candidates chose joint training with the summed objective, adapter
  warm-up 5, retrieval scale 1.0, `w_post` 1.0, `w_near` 0.1, `v_small` 0 and
  adapter learning rate 0.003 (mean rank 17.7; next 22.7). All of the top ten set
  `v_small` = 0.
- Confirmation on seeds 30–34 (26 conditions; batch B confirmed the 10 finalists
  not in batch A): the selected schedule beat the current default on 13 of 16
  tasks (median −11.6% error, Wilcoxon p = 0.008), frozen-core reuse on 12
  (−4.9%, p = 0.005), sequential on 14 (−8.2%, p < 0.001) and retrieval only on
  13 (−9.4%, p = 0.001). Its retrieval-only error stayed level with the default's
  (median −0.3%), so the adapter is not compensating for worse retrieval.
  Per-task selection gained little over the single global schedule (−11.0% vs the
  default). By mean test rank the top 17 of 28 conditions were all joint
  schedules; the default ranked 25th, the sequential schedules 26th and 27th, and
  retrieval only last.

The batch-64 rerun (`schedule_search_bs64_20261012`, `schedule_search_b_bs64_20261012`,
then confirmation) replaces these as the recommendation. Its stage B keeps the
neighborhood chosen from the batch-32 stage A; both stages are ranked together.

### Schedule search on batch-64 tuned cores (current result)

Files: `results/t1/p4_component_sync/schedule_search_bs64_20261012/`. Same design as
the batch-32 run, with every task's core trained exactly as the HPO grid trained it
(all 32 development cores healthy; digits validation accuracy 0.997 and 0.989).

Selection (validation only, stages A and B pooled, 73 candidates): joint training
with the summed objective, adapter warm-up 5, retrieval scale 1.0, `w_post` 0.25,
`w_near` 0.1, `v_small` 0, adapter learning rate 0.003 (mean rank 19.0). The top ten
were all joint warm-up schedules with `v_small` = 0 and adapter rate 0.003. Repeated
stage-A/B runs matched except abalone seed 21, whose core diverged at epoch 40 by a
floating-point difference under identical source and data (CPU math libraries are
not bit-reproducible run to run); this adds noise to that one cross-stage pair.

Confirmation on seeds 30–34 (20 conditions, 80/80 jobs, 1,600 unique rows):

| Contender vs current default (alternating, block 2, scale 0.1) | Better / worse tasks | Median relative error | Wilcoxon p |
|---|---|---|---|
| Validation-selected joint schedule | 11 / 5 | −1.8% | 0.32 |
| Per-task validation selection | 10 / 6 | −0.9% | 0.40 |
| Best joint variants (summed, warm-up 5, `w_post` 1.0, `v_small` 0, rate 0.001; separated, `w_post` 0.5, `v_small` 0, rate 0.003) | 13 / 3 | −2.1%, −1.3% | 0.012, 0.021 |
| Joint, summed, warm-up 5, `w_post` 0.5, `w_near` 0, `v_small` 0.1, rate 0.003 | 11 / 5 | −2.1% | 0.051 |
| Frozen-core reuse | 6 / 9 | 0.0% | 1.00 |
| Sequential | 9 / 6 | −0.7% | 0.36 |
| Retrieval only | 5 / 9 | +0.7% | 0.16 |

The p-values are uncorrected; with 19 comparisons against the default none survives
a Bonferroni correction. By mean test rank the top 14 of 20 conditions are joint
schedules and the current default ranks 19th, above only retrieval only. Gains are
task-dependent: wine 0.061 → 0.017, zebra 0.073 → 0.018, yacht 0.120 → 0.080,
balance 0.066 → 0.042 with the selected schedule, but iris 0.047 → 0.080 and
zebra special 0.277 → 0.309.

Pre-adaptation (retrieval-only) error of the joint schedules stays level with the
default's (selected schedule: median −0.3%), so retrieval is not degraded. Their
corrections are larger: median correction magnitude relative to the default is
×1.57 for the selected schedule and ×1.71–1.81 for the summed `w_post` 1.0,
`v_small` 0 variants, against ×1.27 for the `v_small` 0.1 joint variant and ×1.1–1.3
for separated-objective variants. Dropping the minimal-correction penalty therefore
buys accuracy partly through larger adapter corrections, the trade-off the plan
warns about.

### Recommendation for the T1 training schedule

1. Use joint end-to-end training as the T1 retrieval + reuse schedule: adapter-only
   warm-up for 5 epochs, then simultaneous updates with the full retrieval learning
   rate (scale 1.0) and adapter learning rate 0.003. Joint schedules rank above
   alternating, sequential and frozen-core reuse on fresh seeds, are never worse in
   aggregate, and are much more robust when the retrieval core is poorly trained
   (on the batch-32 cores the selected joint schedule was −11.6% vs the default,
   p = 0.008, largely by repairing collapsed or undertrained cores).
2. On well-tuned cores the gain over the current default is small (about −2%
   median) and not significant after correction. Do not claim a schedule-driven
   accuracy improvement on these tasks; report robustness and the consistent rank.
3. Keep the minimal-correction penalty (`v_small` 0.1) unless accuracy alone is the
   goal. The validation search preferred `v_small` 0, but on the confirmation a
   `v_small` 0.1 joint variant matched its median gain with smaller corrections.
   This choice weighs bounded, explainable corrections against accuracy using
   confirmation results; a future search should select with a validation criterion
   that includes correction size rather than by test outcome.
4. Use one global schedule; per-task selection from two development seeds was
   noisier than the global choice.

## Open items

- The full study's plan stage (`neural-cbr/`, queued after its grid, which is at
  108/110) still rebuilds tuned cores at batch 32; rerun it with `7b978a4` or label
  its tuned-core results accordingly.
- Merge this branch into `neural-cbr` after the full study and the reorganization
  apply step.
- Next T1 step per the collaborator's order: compare case-base maintenance policies
  without EMA, on batch-64 tuned cores with the joint schedule.
- Optional: EMA on a task that needs an encoder (images or text), the setting of the
  MCB manuscript, which these tabular tests do not cover.
