# T2 run recording (TensorBoard run units)

Since 2026-11-22 every T2 RL training is recorded as a run unit of the shared `NN-KNN/runs/` directory
(`model/common/tracking.py`, rules in `docs/NNKNN_CORE.md`, "Run tracking"). Recording is mandatory (owner's rules
O1-O5, 2026-11-22): the file mode that wrote per-run directories (`config.json`, metric CSVs, `summary.json`,
`manifest.json`, `checkpoint.pt`, snapshots) is gone. With the same seed and device a unit holds exactly what that
file mode wrote (checked on g234 against the pre-change tree, see "Checks"). The code is `model/t2/runs.py`; the
tests are `tests/test_t2_runs.py`.

## How to record

- Library: every training function (`train_dqn`, `train_nec`, `train_ppo`, `train_td3`, `train_nnknn_rl`) requires
  `run=`, an open tracking run (`open_run`); None or a `NullRun` raises TypeError, and there is no `output_dir`.
  `train_dqn(task, cfg, run=run, resumable=True)`; the caller completes the run. `t2runs.run_training` opens the
  unit, skips it when it is complete (before any training) and returns the reader of the finished unit.
- `tools/run_rl_{dqn,nec,ppo,td3,nnknn}.py ... --study S` (required; argparse exits with code 2 without it) records
  the run as `runs/t2/S/<algorithm>/<task>/seed<N>` (`algorithm` is `dqn`, `nec`, `ppo`, `td3` or `nnknn_rl`);
  `--unit a/b/c` names the unit instead (needed when two runs differ in something other than the seed, e.g. the
  case capacity: the drivers in `scripts/g234/` use `nnknn_rl/cartpole/cap<C>_s<N>`). `--output-dir` is gone.
  Rerunning the same command skips a complete unit (its summary is printed again) and resumes an interrupted one; a
  complete unit written with another config raises `ConfigMismatchError`.
- `--eval-only --checkpoint <unit> --study S` evaluates the weights of a complete run unit (a checkpoint file or a
  legacy run directory is refused) and records the evaluation as the unit
  `runs/t2/S/eval/<algorithm>/<task>/<checkpoint state sha256[:12]>_seed<eval seed>_ep<episodes>`: record
  `files/eval_summary.json` (what `eval_summary.json` held, without `created_at_utc`), row record
  `files/eval_episodes.csv` and `x/final/*`. A complete evaluation unit is read back, not evaluated again.
- `tools/run_rl_mcb_and_capacity_experiments.py --study S`: `runs/t2/S/<cartpole_mcb|acrobot_capacity>/<run name>`.
- `tools/run_case_maintenance_ablation.py --study S`: `runs/t2/S/rl_case_maintenance/mech_<mechanism>_s<seed>`.
  In both batteries the folder under the study is a fixed name: `--output-dir` (where the CSVs and the report go)
  does not change which unit a training is, so a rerun with another `--output-dir` reads the units back.
- `tools/run_cartpole_all_variants.py --study S`: `runs/t2/S/cartpole_all_variants_<profile>/<variant>`.
- `tools/t2_role_pilot.py --study S`: `runs/t2/S/<task>/<role>__<target>__<optimizer>[__labels_<mode>]/s<seed>`; the
  same unit also holds the role audit (named state `role_audit_state`, records `role_audit` and `role_audit/events`).
- `tools/t2_retention_pilot.py --study S --runs <manifest>`: reads only run units (every `run_dir` of the manifest's
  `runs` list, or of the `roles` list of a `t2_role_pilot.py` manifest, which can be passed as is) and records each condition as `runs/t2/S/retention/<checkpoint state sha256[:12]>/<condition>`: named state
  `states` (the former `states.pt`), record `result` (the former `result.json`) and named state `rng_after`. Two
  roles of the same task and seed no longer collide (the old pilot stopped with FileExistsError on the second).
- `codex/smoke_test.py --mode {train,classification,rl,nec,nnknn_rl} --study S` (required except for `imports`):
  core modes as in main (`runs/core/S/smoke/<mode>`), the T2 modes one unit per training,
  `runs/t2/S/smoke/<mode>/<name>`. A complete unit is not retrained: the smoke checks run on the state rebuilt from
  the unit (weights loaded, summary and rows from the records).
- Launch drivers: `scripts/g234/*.ps1|*.cmd` and `scripts/r760/*.sh` pass `--study` (and `--unit` where needed),
  run in the g234 `rl` worktree with the shared venv, and no longer scan or delete run folders under `results/`
  (a relaunch skips complete units by itself). The ALE drivers have one study per campaign:
  `ale_driver.ps1` `p20_ale_pong_v1` (DQN without the Atari extras), `ale_driver2.ps1` `p20_ale_pong_v2` (DQN on
  xpu), `ale_driver3.ps1` and `ale_driver4.ps1` share `p20_ale_pong_cpu` (driver 4 continues driver 3 with the same
  configs). So a driver can be relaunched over its own units, never over another campaign's.
- Deliberate exception to O4: the timing passes (`timing_pass{,2}.ps1`) and device benchmarks (`dqn_bench.ps1`,
  `nec_bench.ps1`) use a fresh timestamped study per pass and so train identical configs again on purpose, because
  a complete unit would be read back, not timed. They measure throughput and produce no results; do not summarise
  their units. Their timings include the recording overhead (event files, resume checkpoints), so they are not
  directly comparable with the earlier p17/p18/p19 timing files.

## Never retrain (O4)

A complete unit is never trained again: every entry point resolves the unit (`resolve_unit` / `load_unit`) before
any training and reads what it needs from it (summary, rows, weights). Where later computations depend on the global
RNG state that a skipped computation would have left, the unit records that state and a rerun restores it: the
retention pilot saves `rng_after` (`Run.save_rng`) at the end of each condition and restores it
(`RunData.restore_rng`) when it skips a complete one. The other T2 tools need no RNG snapshot: every training
reseeds (`seed_everything(cfg.seed)`) and the role audit runs inside its unit. Proven in `tests/test_t2_runs.py`:

- `test_cartpole_sweep_partly_rerun_equals_uninterrupted`: the sweep crashes inside its second variant; the rerun
  calls the trainer only for that variant (sessions 1, 2, 1), and result digests, state digests and sweep rows equal
  an uninterrupted sweep.
- `test_role_pilot_partly_rerun_equals_uninterrupted`: crash in seed 9; the rerun trains only `s9`; state digests,
  audit records and events, and the manifest equal an uninterrupted pilot (the result digest itself differs only
  through the role_audit record's wall-clock `training_seconds`).
- `test_retention_pilot_partly_rerun_equals_uninterrupted`: crash in the third condition; the rerun recomputes
  only the missing conditions, and every condition's `result` (without `selection_seconds`) and its named state
  `states` equal an uninterrupted pilot. The pilot never seeds the global RNG and its numbers do not depend on it
  (an independent check ran two separate processes: every `result` and `states` was equal, `rng_after` differed).
  So `rng_after` is not reproducible between processes; restoring it makes a rerun continue from the state the
  skipped condition left, which is what an uninterrupted pilot in the same process would see (the test starts
  every call from the same RNG state, and there `rng_after` is equal too).
- `test_run_rl_tool_requires_a_study_records_a_unit_and_never_retrains_it` and
  `test_battery_rows_from_units_and_a_complete_unit_is_read_back`: a complete unit is read back; the trainer (and
  for `--eval-only` the evaluator) is not called.

Resume files (`weights/resume/last.pt`, `prev.pt`) are kept after a unit completes (O5). The device is not part of
a unit's identity (O5, as in `docs/NNKNN_CORE.md`): it is in no T2 `rec/config` (`t2runs.unit_config` refuses a
`device` key; the `--eval-only`, sweep and role-pilot configs carry none). It is recorded as provenance: the core
session header holds the device, and the unit's `config.json` record holds the device the run used. A unit started
on one device is read back, or resumed, on another (resume is bit-exact only on the same device, see below); to
keep two devices apart, use separate studies or `--unit` names (the device benchmarks do). Configs recorded before
this change (2026-11-23) held `"device"` and no longer match: rerunning such a study raises `ConfigMismatchError`.

## Where the generated files go

Summary, report and plotting scripts read only run units inside the run root (`NN-KNN/runs` or `$NNKNN_RUNS`;
a unit copied elsewhere, e.g. into `results/`, is refused, as is a `--eval-only --checkpoint` outside it) and write under `<worktree>/results/` (git-ignored); file
names, formats and columns are unchanged. Hand-written reports stay in `reports/` (in git); no script writes there.

| script | reads | default output |
|---|---|---|
| `run_rl_mcb_and_capacity_experiments.py` | its units | `results/t2/s06_case_memory_mechanisms/rl_mcb_campaign/` (`mcb_cartpole_raw.csv`, `acrobot_capacity_raw.csv`, `RL_MCB_AND_CAPACITY_REPORT.md`) |
| `run_case_maintenance_ablation.py` | its units | `results/t2/s06_case_memory_mechanisms/rl_case_maintenance/` (`case_maintenance_ablation_raw.csv`, `CASE_MAINTENANCE_MECHANISM_ANALYSIS.md`) |
| `run_cartpole_all_variants.py` | its units | `results/rl/cartpole_all_variants_<profile>_<timestamp>/` (`manifest.json`, `sweep_summary.csv/.json`) |
| `t2_role_pilot.py` | its units | `results/t2/role_pilot_<UTC timestamp>/manifest.json` (`--output`) |
| `t2_retention_pilot.py` | run units of `--runs`, its units | `results/t2/retention_pilot_<UTC timestamp>/manifest.json` (`--output`; `result` entries are unit paths) |
| `make_phase4_report.py --manifest M` | units listed in M (required) | `results/figures/` (moved from `reports/figures/`) |
| `scripts/harvest_p17.py RUNS_FOLDER [MANIFEST]` | complete units under RUNS_FOLDER | `results/t2/p17_capacity_cpu_manifest.json` (moved from `reports/`) |
| `scripts/g234/timing_pass{,2}.ps1` | (timing) | `results/t2/p1{7,8}_throughput_controlled.csv` (moved from `reports/`) |

The existing `reports/p17_capacity_cpu_manifest.json`, `reports/p1{7,8}_throughput_controlled.csv` and the figures
under `reports/figures/` on the experiment hosts were made by the old scripts; they stay where they are and are not
regenerated. Old result folders under `results/` on g234 are not read, moved or converted.

The battery tools' raw CSVs take the `run_dir` column from the unit path and the wall-time column
(`elapsed_seconds`, `elapsed`, `training_seconds`, harvest_p17's `seconds`) from the unit's
`x/sys/train_seconds`: the training time summed over its sessions, whether or not this call trained it. An
interrupted session counts up to the checkpoint the next session resumed from (the work redone after it is counted
once), so a resumed unit's value is close to an uninterrupted run's, not to the total machine time spent. It is
measured from the start of the training function to the end of its recording, which is not the same span as a
legacy run directory's wall time (harvest_p17 took launch-to-`summary.json` for those): do not compare the two kinds
in one throughput table.

## What a unit holds

Every per-run file of the retired file mode is a record or a named state; none is written:

| former file | run unit |
|---|---|
| `*.csv` (training, loss, eval, critic holdout, case maintenance, final/last eval episodes) | row record `rec/files/<name>.csv`, each row a list of `[column, value]` pairs (column order kept) |
| `summary.json`, `config.json`, `case_quality_selected.json` | record `rec/files/<name>.json` (`config.json` without `created_at_utc`) |
| `case_optimizer_events.json`, `case_audit_events.json`, `case_retention_events.json` | record `rec/files/<name>.json` plus row record `rec/files/<name>.json/events` |
| `checkpoint.pt` | named state `checkpoint` (`weights/checkpoint.pt`; NEC's DND arrays as tensors) |
| `retention_states/<f>.pt`, `case_audit_states/<f>.pt` | named states `retention_states.<f>`, `case_audit_states.<f>` |
| `manifest.json` | record `rec/outputs` (the files the run produced) |

Paths inside the records are relative to the unit: `run_dir` is `.`, `checkpoint_path` is `weights/checkpoint.pt`,
a snapshot is `weights/<name>.pt`, and the case-quality ledger's namespace is `.`. The readers turn them into full
paths. The quality ledger's identities therefore differ from the old file mode's with the same seed: its
`namespace` (`.`), its event digests (they hash the snapshot path) and the rows' `last_snapshot_sha256` (a unit saves
snapshots through a file object, so the archive bytes differ from `torch.save(path)`). The ledger numbers (C, H, Q,
exposure, steps, phases, event ids) are equal; the file mode already varied the same way with the output path. TensorBoard curves are `x/episode/*` (return, length, latest losses) at the environment step of each episode
end, `x/eval/*`, `x/holdout/*` (NN-kNN critic holdout) and `x/train/*` (PPO and NN-kNN per update), and the headline
numbers `x/final/{selected_mean_return,selected_std_return,last_mean_return,selected_step,actual_timesteps,passed}`.
Per-update losses of DQN, NEC and TD3 are in the event files only as the row record `rec/files/loss_metrics.csv`
(every update, exact); their TensorBoard curves (`x/train/loss`, `x/train/critic_loss`, ...) show the latest
loss once per episode, because a
`log_step` call costs about 0.55 ms on g234 (measured on DQN CartPole, 20,000 steps with an update every step:
curves 0.06 s in total, about 3 microseconds per step; each run also pays a fixed 1 to 1.5 s at the end to flush its
rows and complete the unit).

## Reading

`t2runs.RunFiles(path)` reads a complete run unit (only a unit: a legacy run directory raises FileNotFoundError, and
so does a unit that is not an RL training, e.g. an evaluation unit): `json(name)`, `text(name)` (a unit's CSV text
is rebuilt byte for byte as the file mode wrote it; its JSON has sorted keys), `frame(name)` (pandas from that
text), `rows(name)`, `checkpoint_path()` (hash-checked), `load_checkpoint()`, `resolve(path)` and
`train_seconds()`. The `load_*_checkpoint` functions take a unit directory (its hash-checked `checkpoint` state) or
a weights file such as `RunFiles.checkpoint_path()`.

## Resume

With `resumable=True` a run checkpoints (`save_resume`) at an episode boundary, right after the environment was
reset. The cadence is the core default of 300 s, except for image (ALE) tasks recorded through `--study` /
`t2runs.run_training`, which checkpoint every 1800 s (`t2runs.resume_every_seconds`, from the task spec); a caller
can pass `checkpoint_every_seconds`, and `NNKNN_RESUME_EVERY_SECONDS` / `_STEPS` override everything. What a
checkpoint holds:

- DQN, TD3: networks, target networks, optimizers, the filled part of the replay buffer, best state, counters,
  early-stopping state, the action-space sampler and the global RNGs.
- NEC: as DQN plus the DND (keys, values, LRU clock) and the candidate state.
- PPO: also in the middle of a rollout: agent, optimizer (with the annealed learning rate), the rollout buffers so
  far and the index of the next rollout step.
- NN-kNN-RL: only when no rollout is pending (right after a policy update): actor and critic with their case
  stores, the actor's case insertion steps (grace period), target critic (realigned to the shared case bank),
  optimizers, quality ledger, maintenance and label counters, best state.

The rows logged so far are read back from the unit, not stored in the checkpoint. Checked on g234 (CPU): an
interrupted and resumed run has the same result digest and named-state digests as an uninterrupted one for DQN,
NEC, PPO (discrete and continuous), TD3 and NN-kNN-RL on CartPole/Pendulum, for DQN, NEC, PPO and NN-kNN-RL on
ALE Pong (short runs, crash after the second checkpoint), and after a hard process kill (`tests/test_t2_runs.py`).

Not restored, or not exact:

- The environment itself is not saved: it is rebuilt and reset with the per-episode seed the run would have used.
  This is exact for the tasks above (ALE sticky actions included), not for an environment whose `reset(seed)` does
  not fully determine its state.
- Work after the last checkpoint is redone. NN-kNN with `policy_update_episodes > 1` checkpoints only every few
  episodes; an Atari episode can be thousands of steps long.
- NN-kNN retention events record the wall-clock `selection_seconds`, so a resumed run with retention differs from
  an uninterrupted one in that field only (and so in its result digest).
- Bit-exact resume needs the same device and the same torch thread count: on CPU the thread count changes the
  numbers. XPU resume is not tested.
- `x/sys/train_seconds` does not count the work an interrupted session did after its last checkpoint.
- Disk: the replay buffer is in every resume checkpoint. An Atari DQN or NEC with a 100,000-transition buffer of
  4x84x84 uint8 frames (observation and next observation) writes about 5.6 GB per checkpoint (estimate: 2 x 100,000
  x 28,224 bytes; not measured), every 1800 s. The core keeps `weights/resume/last.pt` and `prev.pt` in the unit
  after it completes (owner's rule O5: resume files are not deleted), so a finished Atari unit holds about 11 GB of
  resume files besides its weights (a CartPole or Pendulum unit a few MB). A multi-seed Atari battery needs that
  much free disk per unit. PPO and NN-kNN-RL keep no replay buffer; TD3 runs only on flat tasks.

## Checks (g234, CPU, 2026-11-23)

The pre-change tree (rl `1915e9f`) and the new tree ran the same commands in scratch copies:

- `run_rl_{dqn,nec,ppo,td3,nnknn}` (DQN, NEC, PPO CartPole and Pendulum, TD3 Pendulum, NN-kNN hybrid critic with
  critic holdout): the old tree's `--study` unit and the new tree's unit have the same result digest and state
  digests; every file the old file mode wrote (9 or 11 per run) equals the new unit's record (CSVs byte for byte,
  JSON without paths and timestamps, checkpoint tensors).
- `t2_role_pilot.py` (two roles, audit and quality ledger: 18 and 21 files per run), `t2_retention_pilot.py` (8
  conditions) and `run_cartpole_all_variants.py` (3 variants): old folders and new units hold the same numbers,
  states and rows (the manifests differ only in paths, wall-clock times and the source fingerprint of the code).
- Notebooks (`notebooks/*.ipynb`) were left unchanged: their `train_dqn`, `train_nec` and `train_nnknn_rl` cells
  call the training without `run=` and now raise TypeError until they open a run.
