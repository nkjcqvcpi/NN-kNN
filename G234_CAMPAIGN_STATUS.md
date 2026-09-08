# g234 migration and campaign — status

Round closed 2026-09-05. Branch `rl-iclr2027`, local only, never pushed.
Companion to `HANDOFF.md` (durable project summary) and
`reports/experiment_log.md` (chronological record). This file covers the
migration to g234 and the campaign run on it.

---

## 1. History

The project was moved from **r760** (Linux, H100, CUDA) to **g234** (Windows
Server 2025, 10-core i5-13400F, Intel Arc A770) and the environment rebuilt
under `uv`. Four experiment batteries then ran on g234.

**Migration.** 4,102 files / 1.77 GB transferred r760 → Mac → g234, SHA256
verified at both ends, git HEAD `b5bd816` preserved, `git fsck` clean. The
5.4 GB `.venv` was excluded and rebuilt.

**A stale lockfile was the first real finding.** `requirements.txt` claimed
torch 2.9.1+cu128 / numpy 2.3.5 / pandas 2.3.3 / sklearn 1.8.0 while the live
r760 venv was running torch 2.13.0+cu130 / numpy 2.5.2 / pandas 3.0.5 /
sklearn 1.9.0, and it omitted `ale-py`, `MinAtar`, `opencv` and `Box2D`
entirely despite `model/rl_workflow.py` importing them. `pyproject.toml` is
pinned to the **live** environment, not the file. See `ENVIRONMENT_g234.md`.

---

## 2. What we did

### Completed batteries

| id | battery | result |
|---|---|---|
| **p15** | corrected DQN, ALE Pong, 1M, seed 0 | **+13.40** — learns; refutes p13 |
| **p16** | corrected NEC, ALE Pong, 500k, seed 0 | **−18.40 ± 1.36** — does *not* learn |
| **p17/p22** | capacity ablation, 6 capacities × 5 seeds, CPU | cost replicates (7.38× vs 7.4×); **return deficit does not** |
| **p20** | ALE DQN + PPO, seeds 0–4, 500k, CPU | **DQN −0.24 ± 9.43 > PPO −10.60 ± 9.57** |

### Headline scientific outcomes

1. **The p13 ALE reading is reversed.** p13 reported PPO as the standout with
   DQN and NEC flat at −21.0, and read it as an on-policy vs value-based
   split. At matched 500k with correct hyperparameters the ordering is
   **DQN > PPO > NEC**, with 3/5 DQN seeds reaching `first_success` against
   0/5 for PPO. p13's DQN was simply misconfigured. **The p13 cross-method
   interpretation should be retired, not merely caveated.**

2. **The p13 confound explains DQN but not NEC.** Correcting the same two
   knobs took DQN from a flat −21.00 to +13.40, but NEC only from −21.00 to a
   noisy −18.40 that never improves — best checkpoint at 20% of budget,
   `first_success` never reached. NEC's failure is architectural, not
   hyperparameter.

3. **Capacity has no effect on return; the curve is flat.** Six capacities ×
   5 seeds span 491.86–499.88 on CPU, a spread smaller than within-arm seed
   variance. There is no knee. And with r760's p14 finally complete, the CUDA
   arm that produced the original claim is 480.12 ± 15.96 at n=5 rather than
   471.53 at n=3 — its deficit is ~1.1 sd and its own two late seeds were the
   best of the five. **"Bigger is worse" is not supportable from either
   host.**

4. **The capacity cost finding is robust.** 7.38× throughput penalty from
   capacity 100 → 2000, against r760's 7.4× — different CPU, OS, device and
   float path, and methodologically independent measurements.
   **Supported claim: a larger case base buys no return improvement while
   costing ~7.4× throughput.**

5. **Seed variance on ALE Pong is enormous.** DQN spans −12.40 to +10.80,
   PPO −21.00 to −1.80, sd ≈ 9.5 for both. Any single-seed claim on this task
   is close to meaningless — which retroactively undermines p13's PPO +12.80
   and p15's DQN +13.40, both seed 0 only.

### Infrastructure findings (all in `ENVIRONMENT_g234.md`)

- **Thread count silently changes results.** `nec` smoke: 155.0 at default
  threads, 90.0 at OMP=2. PPO ALE seed 0: **+9.60 at OMP=3, −21.00 at OMP=2**;
  seed 1: −21.00 at OMP=3, −5.20 at OMP=2. Deterministic at each setting.
  All five workflows now record a `runtime` block in `config.json`. **No run
  predating that commit records its thread count**, including p13 and p15.
- **The Arc GPU is unusable for training.** Three `python.exe` crashes in 20
  minutes inside Intel's Level Zero DLLs (`ze_intel_gpu64.dll` 0xc0000005 /
  0xc0000409, `ur_adapter_level_zero.dll` 0xc0000005), killing runs at ~26k of
  500k steps. XPU is 2.6× faster for ALE DQN, but a run that dies is worth
  nothing. CPU only.
- **Concurrency is governed by scheduling priority, not a hardware ceiling.**
  Equal-priority latency-bound jobs plateau near 3.6 cores however many run.
  Setting one job `High` and the rest `BelowNormal` took it to 12 jobs at 63%
  CPU with the priority job faster than it ran alone.
- **Windows/ssh operational notes:** ssh kills the process tree on disconnect
  (launch via WMI `Win32_Process.Create`); `.cmd` wrappers need CRLF;
  PowerShell 5.1 has no heredocs.

### Data lost this round

- **PPO ALE seed 0 at OMP=3 (+9.60)** — its output directory was deleted by a
  cleanup script while the run was live. These workflows write artifacts only
  at completion, so a running job's directory is indistinguishable from
  debris. The driver no longer deletes anything. The eval curve survives only
  in `p20_ppo_s0.log` and is not a valid harvest source.
- Three ALE DQN runs at ~26k steps, to the GPU driver faults.

---

## 3. What is left

- ~~r760 p14~~ — **DONE 2026-09-05.** Both seeds completed (`Exit status: 0`)
  and were transferred to g234. The CUDA cap-2000 arm is now n=5 at
  **480.12 ± 15.96**, not the published 471.53 ± 14.35: the two late seeds
  (498.20, 487.80) are the two best in the arm, so the published mean was the
  worst three of five. The deficit against capacity 100 falls from ~27 points
  to ~18, about 1.1 sd. **r760 now has no running processes for this
  project.**
- **NEC at n>1.** p16 is seed 0 only. Given the seed variance p20 exposed, a
  single NEC seed cannot support "NEC does not learn".
- **1M-budget ALE arms.** Everything on g234 ran 500k; p13/p15 ran 1M. The two
  sets are not directly comparable.
- **`results/rl_ale_g234_threadconfound/`** holds the OMP=3 PPO seed-1 run,
  preserved deliberately as the evidence for thread sensitivity.

---

## 4. TODO

Ordered by value.

1. **Retire the p13 cross-method interpretation** in `HANDOFF.md` and
   `reports/experiment_log.md`. p20 reverses it. Currently only the p13
   ADDENDUM qualifies it, and that addendum is itself now only half right.
2. **NEC seeds 1–4 at 500k** (~10–20 h each on CPU, or run concurrently).
   Turns "NEC does not learn Pong" from n=1 into a real claim.
3. **Re-check every pre-2026-09-05 run for the thread confound.** No run
   before commit `40b5c3e` records `OMP_NUM_THREADS`. p13's NEC used 8, p16
   used 6. Any cross-run comparison spanning that boundary is unsafe.
4. **Decide whether the capacity trade-off needs a harder task.** CartPole is
   near-saturated at ~500, so the flat curve partly reflects a task with no
   headroom. Acrobot or LunarLander would test it properly.
5. **1M-budget DQN and PPO arms** to connect g234's numbers to p13/p15.
6. ~~Fold r760 p14 into the capacity report~~ — **done**; see the
   2026-09-05 p14-complete entry.
7. Optional: add checkpointing to the RL workflows. Every crash and reboot
   this round cost a full run because artifacts are written only at
   completion.

---

## 5. Reproducing on g234

```powershell
cd C:\Users\Administrator\NN-KNN_RL\nnknn-work
uv sync                                  # env from pyproject.toml
.\.venv\Scripts\python.exe codex\smoke_test.py --mode imports
```

Regression gate at **default thread count**: `rl` 9.5, `nec` 155.0, `ppo` 49.0.
Those values move if `OMP_NUM_THREADS` is set — that is expected, not a
regression.

Campaign drivers: `ale_driver4.ps1` (ALE), `capext.ps1` (capacity),
`cap_driver3.ps1` (resumable capacity), `timing_pass2.ps1` (controlled
throughput). Launch detached via `run_p*.cmd` through WMI; see
`ENVIRONMENT_g234.md`.
