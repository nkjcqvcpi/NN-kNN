@echo off
cd /d "C:\Users\Administrator\NN-KNN\rl"
set NNKNN_DEVICE=cpu
set PYTHONUNBUFFERED=1
set MPLBACKEND=Agg
set OMP_NUM_THREADS=6
set MKL_NUM_THREADS=6
"C:\Users\Administrator\NN-KNN\.venv\Scripts\python.exe" tools\run_rl_nec.py ale_pong --profile fast --seed 0 --device cpu --no-early-stopping --total-timesteps 500000 --eval-frequency 25000 --learning-rate 1e-4 --exploration-fraction 0.1 --study p16_nec_ale_pong > "C:\Users\Administrator\NN-KNN\rl\p16_nec_ale_pong_atari_hp.log" 2>&1
