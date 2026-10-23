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
| Transfer HPO grid (10 TabArena tasks, 3,000-row cap) | same | `configs/t1/grid_transfer_20261013.yaml` | receipts per trial | done (50/50 jobs) |
| Tuning (step D) | same | `configs/t1/tuning_*_20261013.yaml`, `results/t1/p4_component_sync/tuning_20261013/` | LOSO decisions per block | done |
| Freeze (D6) | same | `configs/t1/confirm_final_20261013/`, `.../tuning_20261013/freeze.json` | seed audit | done (`02ed62a`) |
| Confirmation (step F) | same | `results/t1/p6_confirmation_revision/` | pre-registered report on all 30 tasks | done (2026-10-20) |
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

All 30 tasks (20 legacy, 10 transfer), seeds 40-49 plus 50-69 on the seven small tasks. Report:
`results/t1/p6_confirmation_revision/confirmation_report_20261020/` (`tools/t1_confirm_report.py`, prereg sha256
`7c27c510...`, `--final tuned::F`, `--default tuned::alternating_default_matched`, `--final-map`, no
`--allow-incomplete`; per-task source fingerprints checked). Run audit: `confirm_audit_final_20261020.json`.

| ID | F vs | GMR [95% CI] | p (Holm) | supported |
|---|---|---|---|---|
| E1 | retrieval-only | 0.974 [0.936, 1.014] | 0.2923 | no |
| E2 | alternating default | 0.996 [0.990, 1.003] | 0.2923 | no |
| E3a | CatBoost | 1.135 [1.004, 1.283] | 0.6984 | no |
| E3b | TabM | 0.987 [0.855, 1.138] | 0.2616 | no |

Claims, in the pre-registered fallback wording: no detectable gain from reuse on tuned cores (E1); no detectable
schedule effect, all three guards holding (E2: pre-adaptation 1.003, locality 1.040, correction 1.028); F trails
tuned CatBoost by 13.5% [0.4%, 28.3%] (E3a); non-inferiority to tuned TabM not shown at delta = 0.1 (E3b).
Legacy-only E1 0.944 (unadjusted p 0.049) and E2 0.992 (0.035) are descriptive: F was selected on those tasks.
Transfer-only E1 0.995, E2 1.003, E3a 1.111, E3b 1.067. Calibrated ECE improves (median 0.0343 vs 0.0572 before
adaptation); wrong-to-correct flips are 1.17 times correct-to-wrong, below the required 2. E3b's pooled estimate
below 1 rests on the two zebra tasks, where tuned TabM has about 48-50% error. In about 30% of F runs the selected
checkpoint precedes any retrieval or joint update. Full tables: `NEURAL_CBR_PROGRESS_2026-10-20_{EN,ZH}.md`.

Run integrity: 400 + 30 + 10 jobs exit 0; one result row lost in a file-writing race was rebuilt from its run
directory (ledger); abalone crash-rerun duplicates are bitwise identical. The phase is closed except for the
maintenance question (`T1_CBM_DESIGN_REVIEW_20261019.md`), which F does not include.
