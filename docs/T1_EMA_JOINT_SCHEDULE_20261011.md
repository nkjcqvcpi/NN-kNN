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
| EMA ablation runs | same | `results/t1/p5_mcb_interaction/` | 65 jobs, 13 tasks × 5 seeds | see Results |
| Schedule search | same | `tools/t1_schedule_search.py`, `tools/t1_parallel.py`, `configs/t1/schedule_search_*.yaml` | Selection and grid tests | stage A running |
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

### EMA where the tuned core uses the MLP encoder (30/30 jobs)

`configs/t1/ema_tuned_mlp_20261012.yaml`; files in
`results/t1/p5_mcb_interaction/ema_tuned_mlp_20261012/`. The six tasks where the
full-study HPO grid selected the MLP (balance, wine, anneal, abalone, airfoil,
concrete), each starting from its tuned core (no embedding normalization), five
seeds. Arms differ only in the EMA case encoder.

| Setting | EMA 0.9 vs shared (tasks better / worse / tied) | Significant (paired t, p < 0.05) |
|---|---|---|
| Retrieval only | 2 / 3 / 1 | worse on airfoil (both momenta) |
| Frozen-core reuse | 3 / 3 / 0 | worse on airfoil; at 0.99 better on balance, worse on anneal |
| Alternating | 3 / 3 / 0 | better on balance, worse on airfoil |
| Joint (summed, warm-up 5) | 1 / 5 / 0 | worse on airfoil (both momenta) |

Momentum 0.99 is similar or worse (retrieval only 2 better / 4 worse). Balance is
the one task where EMA consistently lowers error (retrieval only 0.066 → 0.054 at
0.9, 0.046 at 0.99); airfoil is consistently and significantly worse (0.251 →
0.297). EMA reduces case-embedding drift on five of six tasks, but neighbor churn
is not lower than with the shared encoder on most of them. Under joint training,
where the online encoder keeps moving with the adapter, EMA is worse on five of
six tasks.

### Conclusion for the EMA question

EMA is not useful as a default on these supervised tabular tasks. With retrieval
only and with retrieval + reuse, at default settings (13 tasks) and on tuned
encoder cores (6 tasks), it is not significantly better than the same encoder
without EMA, apart from balance. It is significantly worse on airfoil. The
raw-feature NN-kNN remains the stronger default for most of these tasks. EMA's
stabilizing effect on stored-case embeddings is real, so it remains a candidate
where an encoder is required (images, text) or where the case base changes during
training; neither condition is tested here. For the next step, compare case-base
maintenance policies without EMA (raw or the tuned core), with EMA kept as an
optional factor.
