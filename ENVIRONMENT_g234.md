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

## XPU stability is driver-dependent

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
