# Training-schedule search, batch-32 cores (2026-10-11/12)

These runs rebuilt each task's HPO-selected NN-kNN core with batch size 32, while
the HPO grid selected it at batch 64 (`tools/t1_grid_search.nnknn_core_settings`).
Schedules within a task still start from identical cores, so the comparisons are
matched, but the cores are not the tuned ones; on digits, seeds 21, 30, 31 and 34
collapse to chance during core training. The batch-64 rerun is in
`../schedule_search_bs64_20261012/`. See `docs/T1_EMA_JOINT_SCHEDULE_20261011.md`.

- `stage_a/`: validation selection over 27 candidates (development seeds 20, 21)
- `stage_b/runs/`: 48 loss-weight / adapter-rate candidates around the stage-A leaders
- `final_selection/`: pooled stage A + B validation selection (73 candidates)
- `confirmation/`: test report on evaluation seeds 30–34; `runs_a` and `runs_b`
  are the two confirmation batches (batch A repeats two balance jobs; the repeated
  rows are identical and the report keeps one)
