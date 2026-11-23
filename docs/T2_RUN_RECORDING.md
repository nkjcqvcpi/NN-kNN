# T2 run recording (TensorBoard run units)

Since 2026-11-22 every T2 RL training can be recorded as a run unit of the shared `NN-KNN/runs/` directory
(`model/common/tracking.py`, rules in `docs/NNKNN_CORE.md`, "Run tracking"). Recording is opt-in: called as before,
every workflow and tool writes the same files to the same run directories and computes the same numbers. The
code is `model/t2/runs.py`; the tests are `tests/test_t2_runs.py`.

## How to record

- `tools/run_rl_{dqn,nec,ppo,td3,nnknn}.py ... --study S` records the run as `runs/t2/S/<algorithm>/<task>/seed<N>`
  (`algorithm` is `dqn`, `nec`, `ppo`, `td3` or `nnknn_rl`); `--unit a/b/c` names the unit instead. No run
  directory is made. Rerunning the same command skips a complete unit (its summary is printed again) and resumes
  an interrupted one; a complete unit written with another config raises `ConfigMismatchError`. `--eval-only
  --checkpoint` accepts a unit directory.
- `tools/run_rl_mcb_and_capacity_experiments.py --study S`: `runs/t2/S/<cartpole_mcb|acrobot_capacity>/<run name>`.
- `tools/run_case_maintenance_ablation.py --study S`: `runs/t2/S/<output folder name>/mech_<mechanism>_s<seed>`.
- `tools/run_cartpole_all_variants.py --study S`: `runs/t2/S/cartpole_all_variants_<profile>/<variant>`.
- `tools/t2_role_pilot.py --study S`: `runs/t2/S/<task>/<role>__<target>__<optimizer>[__labels_<mode>]/s<seed>`; the
  same unit also holds the role audit (named state `role_audit_state`, records `role_audit` and `role_audit/events`).
- In code: open a run (`open_run`) and pass it: `train_dqn(task, cfg, run=run, resumable=True)` (likewise
  `train_nec`, `train_ppo`, `train_td3`, `train_nnknn_rl`). The caller completes the run; `t2runs.run_training`
  does both and returns the reader of the finished unit.

The battery tools still write their raw CSVs and reports to `--output-dir`, the sweep its manifest and sweep
summary, the pilots their manifests; with `--study` the `run_dir` column holds the unit path and the wall-time
column (`elapsed_seconds`, `elapsed`, `training_seconds`, harvest_p17's `seconds`) is the unit's
`x/sys/train_seconds`: the training time summed over its sessions, whether or not this call trained it. An
interrupted session counts up to the checkpoint the next session resumed from (the work redone after it is counted
once), so a resumed unit's value is close to an uninterrupted run's, not to the total machine time spent. It is
measured from the start of the training function to the end of its recording, which is not the same span as a
legacy run directory's wall time (harvest_p17 took launch-to-`summary.json` for those): do not compare the two kinds
in one throughput table.

## What a unit holds

Every per-run file of the file mode becomes a record or a named state; none is written:

| file mode | run unit |
|---|---|
| `*.csv` (training, loss, eval, critic holdout, case maintenance, final/last eval episodes) | row record `rec/files/<name>.csv`, each row a list of `[column, value]` pairs (column order kept) |
| `summary.json`, `config.json`, `case_quality_selected.json` | record `rec/files/<name>.json` (`config.json` without `created_at_utc`) |
| `case_optimizer_events.json`, `case_audit_events.json`, `case_retention_events.json` | record `rec/files/<name>.json` plus row record `rec/files/<name>.json/events` |
| `checkpoint.pt` | named state `checkpoint` (`weights/checkpoint.pt`; NEC's DND arrays as tensors) |
| `retention_states/<f>.pt`, `case_audit_states/<f>.pt` | named states `retention_states.<f>`, `case_audit_states.<f>` |
| `manifest.json` | record `rec/outputs` (the files the run produced) |

Paths inside the records are relative to the unit: `run_dir` is `.`, `checkpoint_path` is `weights/checkpoint.pt`,
a snapshot is `weights/<name>.pt`, and the case-quality ledger's namespace is `.`. The readers turn them into full
paths. The quality ledger's identities therefore differ between file mode and a unit with the same seed: its
`namespace`, its event digests (they hash the snapshot path) and the rows' `last_snapshot_sha256` (a unit saves
snapshots through a file object, so the archive bytes differ from `torch.save(path)`). The ledger numbers (C, H, Q,
exposure, steps, phases, event ids) are equal; file mode already varied the same way with the output path. TensorBoard curves are `x/episode/*` (return, length, latest losses) at the environment step of each episode
end, `x/eval/*`, `x/holdout/*` (NN-kNN critic holdout) and `x/train/*` (PPO and NN-kNN per update), and the headline
numbers `x/final/{selected_mean_return,selected_std_return,last_mean_return,selected_step,actual_timesteps,passed}`.
Per-update losses of DQN, NEC and TD3 are in the event files only as the row record `rec/files/loss_metrics.csv`
(every update, exact); their TensorBoard curves (`x/train/loss`, `x/train/critic_loss`, ...) show the latest
loss once per episode, because a
`log_step` call costs about 0.55 ms on g234 (measured on DQN CartPole, 20,000 steps with an update every step:
curves 0.06 s in total, about 3 microseconds per step; each run also pays a fixed 1 to 1.5 s at the end to flush its
rows and complete the unit).

## Reading

`t2runs.RunFiles(path)` reads a run unit or a legacy run directory with the same calls: `json(name)`, `text(name)`
(a unit's CSV text is rebuilt byte for byte as the workflow writes it; its JSON has sorted keys), `frame(name)`
(pandas from that text), `rows(name)`, `checkpoint_path()` (hash-checked for a unit), `load_checkpoint()`,
`resolve(path)` and `train_seconds()`. `tools/make_phase4_report.py`, `tools/t2_retention_pilot.py` and
`scripts/harvest_p17.py` take either kind; the historical file-based results on g234 are read as before.

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
  after it completes, so a finished Atari unit holds about 11 GB of resume files besides its weights (a CartPole or
  Pendulum unit a few MB). A multi-seed Atari battery needs that much free disk per unit; delete
  `weights/resume/` of complete units by hand when the disk runs short (a complete unit never reads them; whether
  `complete()` should prune them is an open core decision). PPO and NN-kNN-RL keep no replay buffer; TD3 runs only on flat tasks.
