#!/usr/bin/env bash
# p12 TD3-vs-PPO continuous-control reference battery driver.
#
# 12 runs total: {td3, ppo} x {pendulum, lunarlander_continuous} x seeds {0,1,2}.
# Both algorithms run on CPU for this battery:
#   - TD3: pre-flight device probe (3k-step pendulum) measured CPU at 11.44s
#     wall vs CUDA at 59.08s wall for the same 3k steps -- CPU wins decisively
#     here (small batch-256 TD3 updates do not amortize well against the
#     external jobs' ~70-99% GPU contention), so TD3 also runs on CPU.
#   - PPO: CPU per the verified Phase-I finding that CPU is much faster than
#     CUDA for flat-task PPO.
# Bounded parallelism via xargs -P 4. Each job sets CUDA_VISIBLE_DEVICES=""
# and OMP_NUM_THREADS=2 itself so the env vars are correct inside the -P
# workers regardless of xargs's own environment.
set -u
cd /home/haotangl/NN-KNN_RL/nnknn-work

ALGOS="td3 ppo"
TASKS="pendulum lunarlander_continuous"
SEEDS="0 1 2"

jobs_file="$(mktemp)"
trap 'rm -f "$jobs_file"' EXIT

for algo in $ALGOS; do
  for task in $TASKS; do
    for seed in $SEEDS; do
      echo "$algo $task $seed" >> "$jobs_file"
    done
  done
done

run_one() {
  algo="$1"
  task="$2"
  seed="$3"
  log="p12_${algo}_${task}_s${seed}.log"
  echo "[driver] starting algo=$algo task=$task seed=$seed -> $log"
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 /usr/bin/time -v .venv/bin/python "tools/run_rl_${algo}.py" "$task" --profile fast --seed "$seed" --device cpu --no-early-stopping --total-timesteps 150000 > "$log" 2>&1
  status=$?
  echo "[driver] finished algo=$algo task=$task seed=$seed exit=$status"
}
export -f run_one

cat "$jobs_file" | xargs -P 4 -L 1 bash -c 'run_one "$@"' _

echo "[driver] all jobs complete"
