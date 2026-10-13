# Training-schedule search on batch-64 tuned cores (2026-10-12)

Each task's HPO-selected NN-kNN core is rebuilt with the grid's own training
settings (`tools/t1_grid_search.nnknn_core_settings`). This supersedes
`../schedule_search_20261011/` (batch-32 cores). See
`docs/T1_EMA_JOINT_SCHEDULE_20261011.md`.

- `stage_a/runs/`, `stage_b/runs/`: development seeds 20 and 21 (27 + 48 candidates)
- `final_selection/`: pooled validation-only selection (73 candidates)
- `confirmation/`: test report and run shards on evaluation seeds 30–34
