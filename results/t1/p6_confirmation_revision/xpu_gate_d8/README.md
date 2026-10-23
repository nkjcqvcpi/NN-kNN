# XPU gate D8 (docs/T1_PREREG_20261013.md, step I), 2026-10-13

Configuration `configs/t1/xpu_gate_d8_20261013.yaml`: superconductivity (TabArena 363705), 5,000
training cases, development seed 20, a fixed raw core (top-k 10, temperature 0.5, four weight sets,
case learning rate 0.01), retrieval only, alternating default and the incumbent F0. Code `ebdd859`.

| Criterion | Result |
|---|---|
| No device errors | **Failed.** First XPU run died after 94 min with `UR_RESULT_ERROR_OUT_OF_RESOURCES` while concatenating test-stream chunks (about 140 MB); the second, identical run completed. A first attempt (code `9b45d82`) failed on a 5.35 GiB single allocation, fixed by `model/t1/chunked.py`. |
| Two XPU runs bitwise identical (`NNKNN_DETERMINISTIC=1`) | Passed: 300/300 core epochs identical. |
| XPU epoch <= 1/3 of a CPU epoch | Not measured cleanly (machine fully loaded). |
| XPU vs CPU within the development-seed spread | Not assessable. The fixed core is chaotic at this scale: epoch-1 validation loss 1.195 (XPU) vs 1.076 (CPU), then the CPU core collapsed to the mean (val loss 1.0, best epoch 3) while the XPU core learned (0.259, best epoch 284). Initialization is identical on both devices; tiny arithmetic differences grow in the first epoch. |

Decision: **no-go** for the secondary XPU group (step I) as pre-registered; nothing from the XPU
enters the primary analysis.

Finding for the primary analysis (XPU run 2, `runs__*.jsonl` here): synchronized training on this
harder task lowers the adapted error (test RMSE 0.52 retrieval only; 0.43 alternating default; 0.41
F0) while retrieval itself degrades to worse than the mean (pre-adaptation test RMSE 1.36 and 1.37;
mean correction 0.91 standard deviations). The adapter carries the prediction, the failure the T1
plan calls "adapter hides bad retrieval". The pre-registered guards (E2: pre-adaptation GMR vs
retrieval-only <= 1.02) and the H1 retrieval-guard block address exactly this.
