# 2026-11-26: three fixes after the handoff packs (allcaseF wine, engine ties, delta_from_init)

Status: code fixes and pack p0 item (a) configs. Nothing was deployed, queued or run on any host. Every check below ran on
accel in scratch directories under `/var/tmp/nnknn-fix1126/`. New behaviour is opt-in or limited to the failing cases.
Table 1 code, frozen configs and the shared NN-kNN core (`tools/nnknn_core_manifest.json`) are untouched.

## 1. p1: `handoff_allcaseF_r760/allcaseF_20261126/wine` fails with "backward through the graph a second time"

The failing job is **classification** wine (dataset key `wine`; regression wine is `regression__wine`). It uses an mlp
encoder from the all-case grid, `joint_rr` with 5 adapter warm-up epochs, and seeds 50-52.

**Root cause.**
- `train_synchronized` started with a shape probe: `model.retrieve(X[:1])` with gradients enabled (`sync.py`, old line 226).
- The model is in eval mode there (the runner's `StabilityTracker` snapshot leaves it so). In eval mode,
  `NN_KNN_Model._extract_features` writes case features into `cached_features`.
- That cache marks a row as "not yet cached" when the row sums to exactly 0. On wine s50 the trained mlp maps **11 of the
  106 cases to an all-zero feature vector** (dead ReLU), so every eval-mode call recomputes those rows.
- The probe recomputed those rows under autograd and wrote them into the cache in place (`IndexPutBackward`). From then
  on, the whole cache tensor belonged to the probe's graph.
- The adapter warm-up runs in eval mode with the retrieval parameters frozen, so it reads features from that cache. The
  first minibatch's `L.backward()` went through the probe's graph and freed it. The second minibatch raised.
- Seeds 51 and 52 have no all-zero rows, so the probe only reads the cache and nothing is attached. That is why only s50
  failed, and why the other mlp tasks passed.

**Fix.** The probe runs under `torch.no_grad()` (`model/t1/sync.py`). `retain_graph` would have been wrong: it would let the
adapter phase push gradients into a stale copy of the encoder graph.

**Why recorded behaviour does not change.**
- The probe's value is used only for its shape.
- The cache values are the same with or without gradients. Only the autograd attachment goes away.
- A run that backpropagated through the stale graph a second time crashed, so no completed unit depends on it.
- A single pass through it could only accumulate gradients into frozen retrieval parameters, and every
  retrieval/joint step calls `ropt.zero_grad()` before its own backward.
- Old (HEAD 2bd6c4a) vs new code on accel, all weights files of all units compared bitwise. Only the unseeded Python
  `random` state in `rng.pt` differs, and it also differs between two runs of the same code.

| config | seed | units / files | difference |
|---|---|---|---|
| allcaseF classification wine | 51 | 14 files | none |
| allcaseF classification iris (mlp) | 50 | 22 files | none |
| maskedF classification wine | 50 | 14 files | none |
| allcaseF regression wine (raw) | 50 | 22 files | none |

- With the fix, allcaseF classification wine completes on accel for s50, s51 and s52 (about 26-29 s each). s50 F gives
  test accuracy pre 0.9722 and post 0.9722.

**NaN lead at `nnknn_model.py:1128`: an anomaly-mode artifact, not a bug.**
- `sqrt(relu(sum))` at a distance of exactly 0 gives a NaN/inf local gradient in `SqrtBackward`, and anomaly mode reports it.
- The next node, ReLU's `threshold_backward`, replaces the gradient by 0 wherever its input is <= 0. A direct check
  gives grad 0 at s = 0, so no NaN reaches any parameter.

**tabarena_363631 s50 (allcaseF regression, mlp, joint_rr with warm-up): the same bug.**
- Attempt 1 failed after 5949 s with the same error, at the adapter-phase `L.backward()` (line 355 in that older tree).
  s51 and s52 completed all three F arms.
- The crash killed the process while the `aggregate_regression_combined` s50 unit was still writing its 3.5 GB event
  file. Its weights are dated 09:23, the event file's last write 09:46, the crash 09:46.
- The current attempt (started 12:37) loads `_core` and `retrieval_only` as complete, but logs no `[recorded]` line for
  the aggregate arm. It is therefore retraining that arm: 1 h 40 min at 99 % CPU, 43 GB read, no F unit yet. That is the
  slowness. For comparison, s51 took 3.5 h in total under full load.
- The core is loaded from the recorded unit and the code on r760 is still 2bd6c4a. The probe will meet the same
  all-zero rows, so the F arm should fail again in the same way.
- After this fix is deployed, only the missing units run. Complete units are never retrained.

## 2. p2: "Activation engine disagrees with model.retrieve by 0.332" (newpol classification wine s32, xps8960)

**Root cause.**
- With the pre-top-k mask, `model.retrieve` keeps the members chosen by `torch.topk` on its float32 scores
  z = b - d (-inf for the query's own case and for masked cases).
- `ActivationEngine` chose members itself: `np.argpartition` on float64 scores (b - d)/tau, then score/index order.
- When several cases tie exactly at the k-th place, the two rules can keep different cases. In the xps8960 diagnosis,
  query 20 had cases 2, 4, 5, 12 and 13 tied: the engine kept [2, 4, 91], the model kept [12, 13, 91].
- On accel the same recorded core gives no tie (the floating-point results differ by host), so the failure cannot be
  reproduced there with recorded weights. The new test builds an exact tie from duplicate cases instead.

**Fix (`model/t1/baselines.py`).**
- The engine also keeps the model's float32 scores (`Z_select`).
- For each neighbourhood, `torch.topk` gives the model's member set. Only rows whose eligible member set differs from the
  engine's own are changed: their members become the model's (in score/index order), and the next case becomes the best
  remaining one.
- Every other row is returned exactly as before.
- Engines built from hand-written arrays (`from_arrays`) keep the old rule.

**Checks.**
- New test: two duplicate cases with different labels, top_k 1. The old code raises "disagrees ... by 1". The new code
  keeps the model's member and matches `retrieve(case_mask=...)` under three masks.
- A tie-free core: members, scores and next case are bitwise equal under the old and new rules for 20 random masks.
- Old vs new on accel, all units of newpol:

| run | units / files | difference |
|---|---|---|
| classification wine s30 | 70 files | none (0 of 106 rows tied) |
| regression body_fat s30 | 70 files | none |
| classification iris s31 | 70 files | rc_cnn, crr, bbnr at K 0.2/0.5/0.75 differ |

- On iris s31, 2 of the 90 reference rows have a float32 tie at the k-th place. On these rows the old engine credited
  coverage to a case that `model.retrieve` does not retrieve, and the predictions agreed, so the check passed. The new
  rule follows the model. This is the intended change, and it touches tied rows only.

**Consequence for recorded units (not changed, never retrained).**
- The newpol and latency units recorded with the old engine stay as they are. Where a host had a float32 tie at the
  k-th place in the full memory or under a CNN/CRR/BBNR mask, the old rule credited coverage differently from the model.
- wine s30/s31 (xps8960) and the other datasets passed the full-memory check, so their predictions agree with the model.
  Whether member attribution differed cannot be decided on accel, because ties depend on the host's floating point.
- When newpol wine s32 is rerun on xps8960 with the fix, it uses the new rule, and s30/s31 keep the old one. This is
  only consequential if s30/s31 had tied rows. A read-only check on xps8960 (count rows whose member sets differ, as
  in `/var/tmp/nnknn-fix1126/scripts/count_ties.py`) would settle it. It was not run, because running on hosts was out of scope.

## 3. p0 item (a): `bias_source: delta_from_init` with the build-time bias

**Opt-in key.**
- `statistics.initial_bias: build_time` (allowed values: `first_audit`, which is the default and the behaviour when the key
  is absent, and `build_time`).
- In `run_retention`, `provenance.seed_initial_bias` records `model.t1_bias_init["value"]` as every active case's
  `initial_bias` before the first audit. That value is the bias `build_model` gave every case (knn_mean rule).
- `t1_run` refuses the key for any type other than retention.
- Without the key nothing changes.

**Configs.** `configs/t1/handoff_20261126/p0a/` holds 4 files, each headed "exploratory, our choice, not a PI decision".
They are the 2026-11-25b btransform configs with `bias_source: delta_from_init` and `initial_bias: build_time`:

| file | normalization | compared with |
|---|---|---|
| `deltainit_percentile_{classification,regression}.yaml` | within_cohort_percentile | plan_full `bias_normalized` |
| `deltainit_medianmad_{classification,regression}.yaml` | median_mad | `handoff_btransform_r760` rows |

- Study `handoff_deltainit_r760`, seeds 30-32.
- Every dataset runs on the host of its btransform rows (q-lab, xps8960 or r760, as listed in each file's header).

**Job lists (not queued).** In `/var/tmp/nnknn-handoff1126/p0a/jobs/`:
- `r760_extras.jsonl`: 24 jobs.
- `q-lab.json`: 18 jobs.
- `xps8960.json`: 16 jobs.

They are built by `mkjobs.py` from the btransform job lists, one to one; the configs by `mkcfg.py` in the same folder.

**Expected result, stated before any host run, and checked on accel.**
- `build_model` fills every case with one bias, so b - b_build is a constant shift of b.
- Within-cohort percentiles are unchanged. The difference of two float32 values is exact in float64, so the order is
  preserved exactly.
- Median/MAD is unchanged up to rounding.
- Scratch runs on accel (iris and classification wine, s30, K 0.2/0.5/0.75, both normalizations) against the reference
  configs: **kept case ids and final model states are identical in all 12 units**. The store did record the build-time
  value (iris 0.5821, wine 0.2747).
- So pack p0 (a) is answered without host compute: the build-time `delta_from_init` B is the current-bias B. The job
  lists exist if the owner still wants the end-to-end confirmation on every task. A per-case initial bias would need a
  different build rule, which is not part of this item.

## Requeue needed after a deploy (not done here)

1. r760: `handoff_allcaseF_r760/allcaseF_20261126/wine` (s50 resumes after its complete `_core`, `retrieval_only` and
   reuse units; s51 and s52 run in full), and `tabarena_363631` s50 if its current attempt fails as expected.
2. xps8960: `handoff_newpol_r760/newpol_retention__classification/wine` (s32; `_core` is complete).

## Tests

`tests/test_t1_fixes_20261126.py` (7 tests). The three regression tests for items 1 and 2 fail on 2bd6c4a with the
original errors.

Full suite on accel (hub venv, scratch under `/var/tmp`): 904 passed, 3 skipped, 3 xfailed, 1 failed. The failure is the known pre-existing `test_t1_tuning_features.py::test_defaults_reproduce_the_recorded_toy_outputs`. Log: `/var/tmp/nnknn-fix1126/pytest_full.log`.

## Review (2026-11-26, after 4235e10)

An adversarial review checked the commit and found no defect in the three fixes. It made one small change, in a
follow-up commit.

- **Defaults are bit-identical.** The review compared old code (2bd6c4a) and new code (4235e10) on accel with three
  configs the builder had not compared. These are newpol classification zebra s30 (it builds the activation engine),
  allcaseF regression body_fat s31 (adapter warm-up), and pctinit_sync regression yacht s30 (`train_synchronized`).
  All 141 weights files are bitwise equal, apart from Python's unseeded `random` state.
- **The root cause is fixed.** The probe was the only eval-mode `retrieve` that ran with gradients on while encoder
  parameters were trainable. Every other eval-mode call either runs under `no_grad` or `frozen_parameters`, or comes
  after a train-mode call that clears the cache. A cache loaded from a recorded stage is rebuilt on first use, because
  its key records the old storage pointers. So the r760 requeue, which loads the recorded `_core`, starts from a clean
  cache. The new regression tests fail on 2bd6c4a with the original errors and pass on the new code.
- **The new units do not collide with recorded ones.** Study `handoff_deltainit_r760` does not exist on r760, q-lab or
  xps8960. The experiment names `deltainit_*` are new.
- **Each job list uses the host that recorded the units it is compared with.** The review read the host name from the
  event file names of the `plan_full_r760` retention units and the `handoff_btransform_r760` units on r760. In both,
  abalone, airfoil, car and regression wine were run on r760 (cs-peng-research). Iris, wine, zebra, zebra_special,
  synthetic_diag, body_fat, diabets and yacht were run on xps8960 (haotang-lab). The other nine datasets were run on
  q-lab. The p0a job lists put every dataset on the same host.
- **Change made.** `tools/t1_run.py` now refuses two cases before any training. The first is an `initial_bias` value
  that is not allowed; before, this failed only after the core was trained. The second is `initial_bias: build_time`
  together with `reliability`. The `ch_reliability` unit scores B with its own store, which the key does not seed, so
  that unit would have recorded a rule it did not use. No p0a config uses `reliability`. Two tests were added.
