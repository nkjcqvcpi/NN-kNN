# Neural CBR (T1) logs

Run logs are local (ignored) and filed by the same T1 handoff phases as `../results/t1/`:

```
logs/t1/<phase>/pilot_20260929/   one .log/.log.err pair per (experiment, dataset) of the 2026-09-29 pilot
logs/t1/<phase>/pi20260920/       plan-audit and grid-preflight logs of the current pipeline (after --pipeline)
logs/t1/pilot_20260929_queue_status.txt
```

`../scripts/reorganize_t1_results_20261010.py` moves them from their earlier flat locations. `tools/t1_launch.ps1` still writes new logs to `logs/t1/<experiment>__<dataset>.log`; file them by phase when the batch finishes.
