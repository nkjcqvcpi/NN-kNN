# T1 tuning and confirmation, October 13, 2026

This phase followed the pre-registration `docs/T1_PREREG_20261013.md` (addenda 1–4): validation-only
tuning on development seeds, a freeze of the final configuration F, and one confirmation on fresh
evaluation seeds. Earlier T1 results remain exploratory; seeds 30–34 were retired because they had
informed decisions.

## Ownership and handoff

| Step | Workstream / branch / checkout | Files | Validation | Status |
|---|---|---|---|---|
| Accelerator fixes in the shared core | NN-kNN core / `main` (`e1b1d00`), synced to `rl` (`dd5db16`), `LLM` (`bf42c7d`), this branch (`bd40d06`) | `model/common/core.py`, `model/nnknn_model.py`, core tests | main 30, rl 66, LLM 101, T1 219 tests; T1 reruns bit-identical on CPU (56 runs) | done; `neural-cbr` receives it when this branch is merged |
| T1 tooling (step B) | Neural CBR / `neural-cbr-ema-joint-20261010` / `neural-cbr-ema/` | `model/t1/{sync,reuse,chunked}.py`, `tools/t1_{run,grid_search,schedule_search,seed_audit,confirm_report,block_accept,freeze,parallel}.py` | 300+ tests; four independent reviews; defaults bit-identical | done |
| Transfer HPO grid (10 TabArena tasks, 3,000-row cap) | same | `configs/t1/grid_transfer_20261013.yaml` | receipts per trial | 45/50 jobs at freeze |
| Tuning (step D) | same | `configs/t1/tuning_*_20261013.yaml`, `results/t1/p4_component_sync/tuning_20261013/` | LOSO decisions per block | done |
| Freeze (D6) | same | `configs/t1/confirm_final_20261013/`, `.../tuning_20261013/freeze.json` | seed audit | done (`02ed62a`) |
| Confirmation (step F) | same | `results/t1/p6_confirmation_revision/` | pre-registered report | running |
| XPU gate D8 | same | `results/t1/p6_confirmation_revision/xpu_gate_d8/` | two XPU runs + CPU | no-go |

## Tuning (step D)

Decisions and evidence: `results/t1/p4_component_sync/tuning_20261013/DECISIONS.md`. In short:

- **Selection rule (H6).** Per-task selection by mean validation error, then validation loss, beat the
  global rule on held-out development seeds (GMR vs the alternating default 0.891 vs 0.932; task-clustered
  z −1.98). Adopted, with the pre-adaptation feasibility constraint applied per task (addendum 3).
- **Classification output (H2/H2c).** The nominal-residual output at score scale 1 is badly calibrated
  (validation ECE after adaptation 0.24 vs 0.03 before). Score scale 10 keeps calibration (0.012) at
  non-inferior held-out error and 4.5× smaller corrections. Adopted for classification.
- **Rejected blocks.** Retrieval guard and `w_pre` (H1), output variants on the error rule (H2a/b), adapter
  capacity (H3) and budget/stopping (H5) all left held-out error unchanged or slightly worse. With six
  development seeds, adding candidates overfits validation; the leave-one-seed-out gate prevented adopting
  them. H4 was skipped (optional).
- **F.** Per task on the 18 tuned legacy tasks (frozen reuse on diabetes); F_global (joint, summed objective,
  adapter warm-up 5, `w_post` 0.5, `w_near` 0, `v_small` 0.1, adapter rate 0.003; score scale 10 for
  classification) on car, regression wine and the transfer tasks, which have no development runs.

## Harder datasets

Ten TabArena tasks chosen before any neural-CBR result on them (`docs/T1_PREREG_20261013.md`): splice,
students dropout, in-vehicle coupon, churn, MIC, superconductivity, Miami housing, diamonds, physiochemical
protein, QSAR-TID-11. Their five-family HPO grid uses a 3,000-row training cap for every family; neural CBR is
never tuned on them.

Finding from the XPU gate (superconductivity, 5,000 cases): synchronized training lowered adapted error
while retrieval itself degraded to worse than the mean (pre-adaptation RMSE 1.36 vs 0.52 retrieval only) —
the adapter carried the prediction. The pre-registered guard (pre-adaptation GMR vs retrieval-only ≤ 1.02)
tests this in confirmation.

## Accelerators

The A770 now runs T1 (core build-device fix, vectorized feature cache, single-tensor Adam on XPU, chunked
large retrievals) and deterministic XPU runs reproduce bitwise. It is not used for confirmatory results:
one run hit a driver resource error, XPU is slower than one CPU thread below about 2k cases, and a chaotic
core makes CPU and XPU trajectories incomparable (gate D8, no-go).

## Confirmation (step F)

Pending. Report: `tools/t1_confirm_report.py` with the frozen prereg sha256, `--final tuned::F`,
`--default tuned::alternating_default_matched`, `--final-map .../final_map.json`, baselines from the
re-evaluated legacy grid winners (seeds 40–49, 50–69) and the transfer grid.
