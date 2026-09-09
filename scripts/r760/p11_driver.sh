#!/usr/bin/env bash
# p11 PPO fixed-budget battery driver.
# Runs the remaining 19 (task, seed) combinations (cartpole seed 0 was
# launched manually and verified before this driver started) with bounded
# parallelism via xargs -P 6. Each job sets CUDA_VISIBLE_DEVICES="" and
# OMP_NUM_THREADS=2 itself so the env vars are correct inside the -P workers
# regardless of xargs's own environment.
set -u
cd /home/haotangl/NN-KNN_RL/nnknn-work

TASKS="cartpole acrobot lunarlander minatar_breakout"
SEEDS="0 1 2 3 4"

jobs_file="$(mktemp)"
trap 'rm -f "$jobs_file"' EXIT

for task in $TASKS; do
  for seed in $SEEDS; do
    if [ "$task" = "cartpole" ] && [ "$seed" = "0" ]; then
      continue  # already run manually
    fi
    echo "$task $seed" >> "$jobs_file"
  done
done

run_one() {
  task="$1"
  seed="$2"
  log="p11_ppo_${task}_s${seed}.log"
  echo "[driver] starting task=$task seed=$seed -> $log"
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 /usr/bin/time -v .venv/bin/python tools/run_rl_ppo.py "$task" --profile fast --seed "$seed" --device cpu --no-early-stopping --total-timesteps 150000 > "$log" 2>&1
  status=$?
  echo "[driver] finished task=$task seed=$seed exit=$status"
}
export -f run_one

cat "$jobs_file" | xargs -P 6 -L 1 bash -c 'run_one "$@"' _

echo "[driver] all jobs complete"
