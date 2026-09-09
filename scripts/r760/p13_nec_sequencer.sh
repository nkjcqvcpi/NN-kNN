#!/bin/bash
# Wait for the NN-kNN ALE arm (PID 1618550) to exit, then launch NEC on cpu.
while kill -0 1618550 2>/dev/null; do sleep 15; done
cd /home/haotangl/NN-KNN_RL/nnknn-work
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 nohup /usr/bin/time -v .venv/bin/python tools/run_rl_nec.py ale_pong --profile fast --seed 0 --device cpu --no-early-stopping --eval-frequency 25000 --total-timesteps 500000 > p13_nec_ale_pong.log 2>&1 &
echo "$! p13_nec_ale_pong (cpu relaunch via sequencer)" >> p13_pids.txt
