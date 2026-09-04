# g234 (Windows) environment notes

Companion to `pyproject.toml`. Records the things about this host that are not
recoverable from the lockfile and that cost real time to rediscover.

## Host

- Windows Server 2025 Datacenter, build 26100. Default ssh shell is `cmd.exe`.
- CPU: **10-core i5-13400F** (6 P-cores + 4 E-cores, 16 logical threads).
  Not a 16-core server part -- size concurrency against 10 real cores.
- GPU: Intel Arc A770, 15.9 GB. No CUDA.
- Python/env managed by `uv` (see `pyproject.toml`, `.python-version`).

## Device: use CPU

Runs use `--device cpu`. This is a **performance** decision, not a stability one:
measured on this box, the A770 is slower than the CPU at the case-base sizes
this project actually uses.

| case base | CPU | XPU | speedup |
|---|---|---|---|
| n=500  | 180 ms | 392 ms | 0.46x |
| n=2000 | 139 ms | 395 ms | 0.35x |

The GPU only wins on `cdist` far above any capacity used here (~14x at
n=8192), which no experiment reaches. `torch.xpu` is installed and working, so
`NNKNN_DEVICE=xpu` is available if a future workload has larger retrievals.

These runs are **latency-bound on many small tensor ops, not CPU-bound**:
system-wide CPU sits near 27% with nine workers live and disk queue 0. Running
more jobs concurrently raises aggregate throughput without raising total CPU
-- repacking the capacity campaign from 1 to 8 concurrent jobs took aggregate
throughput from 27.6 to 266.8 steps/s.

## XPU: DO NOT USE FOR TRAINING

Superseded finding. An earlier revision of this file said the `topk` crash was
"FIXED" by driver 32.0.101.8991 and implied XPU was usable. It is not. The
driver upgrade fixed that one op; the stack still faults arbitrarily under
sustained load.

Evidence, 2026-09-04, three python.exe crashes in 20 minutes with the faulting
module inside Intel's Level Zero stack:

| time | faulting module | exception |
|---|---|---|
| 08:22:09 | `ze_intel_gpu64.dll` 1.15.39183.3 | 0xc0000005 access violation |
| 08:27:29 | `ze_intel_gpu64.dll` 1.15.39183.3 | 0xc0000409 stack buffer overrun |
| 08:39:56 | `ur_adapter_level_zero.dll` 2026.0.0.0 | 0xc0000005 access violation |

Python-visible as `RuntimeError: level_zero backend failed with error:
2147483646 (UR_RESULT_ERROR_UNKNOWN)`, raised at two unrelated sites: Adam's
`_multi_tensor_adam` -> `torch._foreach_div_`, and a plain `x / 255.0` in
`model/cnn_encoders.py`. **The crash site is arbitrary**, so
`adam_kwargs_for_device()`'s `foreach=False` (kept, XPU-only, harmless) does
NOT prevent it. Three ALE DQN runs died at ~26k of 500k steps.

XPU is genuinely faster for CNN-dominated work -- ALE DQN measured 48.89
steps/s on XPU vs 18.93 on CPU -- but a run that dies a twentieth of the way in
is worth nothing. **Run training on CPU.** `torch.xpu` remains fine for short
interactive probes.

## Device preference is set by workload shape, not by device

Measured on this host, all on ALE Pong except the last row:

| workload | CPU | XPU | faster | why |
|---|---|---|---|---|
| DQN (ALE) | 18.93 | 48.89 | XPU 2.6x | Nature-CNN over 84x84x4 dominates |
| PPO (ALE) | 61.62 | 20.21 | CPU 3.0x | no replay buffer, so per-step host<->device transfer dominates short rollouts |
| NEC (ALE) | 6.94 | 4.46 | CPU 1.6x | kNN lookup over the DND dominates |
| kNN cdist (CartPole) | see below | | CPU | small-tensor cdist/topk |

So "XPU is slower for this project" was too broad, and so was "XPU is faster
for ALE". Only DQN gains. None of it matters while the driver faults.

## Concurrency: the ceiling is memory bandwidth, not cores

The host looks idle at ~23% CPU, but that headroom is not usable by every
workload. Pure-compute burners reach **12.66 of 16 logical cores** and scale
cleanly (1810 -> 3543 -> 5428 matmul/s at 2 -> 4 -> 8 processes), so the cycles
are genuinely free. Large-footprint RL jobs cannot take them.

Measured: p16 NEC's CPU share against the number of ALE jobs beside it.

| ALE jobs alongside | NEC cores | system CPU |
|---|---|---|
| 0 | 3.31 | ~21% |
| 2 | 1.37-1.87 | ~23% |
| 4 | 0.71 | 25% |
| 6 | 0.44 | 23% |

Each doubling roughly halves NEC while system CPU barely moves. That is
**memory bandwidth and L3 contention**, not scheduling: NEC holds a
60,000-entry DND over 84x84x4 image embeddings, and every added ALE job
evicts its working set.

**Footprint decides whether concurrency helps.** The p17 CartPole capacity
runs scaled about 10x with concurrency (27.6 -> 266.8 steps/s aggregate at 1
-> 8 jobs) because their case bases are 100-2000 entries and stay in cache.
ALE-sized jobs do not. Do not generalise a scaling result from the CartPole
runs to the ALE runs -- that mistake cost a campaign restart here.

Practical ceiling for ALE work on this host: **about 2 concurrent ALE jobs
plus NEC**. Beyond that everything finishes later, not sooner.

## Thread count changes NEC's results

The nec smoke is deterministic at a given thread count but differs between
them: **155.0 at default threads, 90.0 at OMP_NUM_THREADS=2**, same host, same
code, same seed. It reads 183.0 on r760. Hold OMP_NUM_THREADS constant across
any set of NEC runs meant to be compared; p13 used 8 and p16 uses 6.

## Old note: XPU stability is driver-dependent

| Intel driver | Level Zero | torch 2.13.0+xpu `topk` |
|---|---|---|
| 32.0.101.6557 | 1.6.31896 | **kills the device** (`UR_RESULT_ERROR_DEVICE_LOST`) |
| 32.0.101.8991 | 1.15.39183+3 | OK (150/150 stress iterations) |

The A770 reports `has_fp64 = 0`; float64 tensors raise
`Required aspect fp64 is not supported on the device`. The codebase is safe
here -- its only float64 use is numpy on the CPU side (`nec_workflow.py` LRU,
`ppo_workflow.py` action bounds, which converts back to Python floats).

## Long-running jobs

Windows OpenSSH **kills the whole process tree when the ssh session closes**,
so `Start-Process` is not enough for a multi-hour run. Launch detached through
WMI, which parents the process outside the session:

    $si = ([wmiclass]'Win32_ProcessStartup').CreateInstance(); $si.ShowWindow = 0
    ([wmiclass]'Win32_Process').Create('cmd.exe /c "C:\path\job.cmd"', $workdir, $si)

Batch wrappers must be **CRLF**; `cmd.exe` mis-parses LF-only `.cmd` files.
PowerShell here is 5.1: no heredocs (use `git commit -F file`), no
`ForEach-Object -Parallel`.

Prefer shipping a `.ps1` and running it by path over inline
`ssh g234 'powershell -c "..."'`, which double-escapes backslashes -- it once
wrote a corrupted `C:\Users\...\\.local\\bin` entry into the persistent PATH,
and silently mangles `$_` inside script blocks.

## Reliability

This host bugchecked once under sustained load: `0x000000E6`
DRIVER_VERIFIER_DMA_VIOLATION (2026-09-03 14:08, minidump
`C:\WINDOWS\Minidump\090326-7328-01.dmp`), killing every in-flight run. The
graphics driver was upgraded afterwards; whether that fixed it is unproven.

Campaign drivers should therefore be **resumable**. `cap_driver3.ps1` is the
model: on start it rescans completed run dirs, maps each back to its
`(case_capacity, seed)` via the nested `config.config`, queues only what is
missing, and deletes any run dir lacking `summary.json` as crash debris.

Note that the RL workflows write `config.json` and `summary.json` only at
completion, so a killed run leaves **no artifacts and no resumable
checkpoint** -- a crash costs the whole run.
