$ErrorActionPreference='Continue'
Set-Location 'C:\Users\Administrator\NN-KNN_RL\nnknn-work'
$LOG='logs/g234/p19_alebench.log'
function Say($m){ Add-Content -Path $LOG -Value ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'),$m) }
$env:PYTHONUNBUFFERED='1'; $env:MPLBACKEND='Agg'
$env:OMP_NUM_THREADS='3'; $env:MKL_NUM_THREADS='3'
$STEPS=4000
Say "ALE DQN device benchmark: $STEPS steps, cpu vs xpu (p16 NEC running concurrently on both)"
foreach ($dev in 'cpu','xpu') {
  $env:NNKNN_DEVICE=$dev
  Say "measuring dqn on $dev"
  $sw=[Diagnostics.Stopwatch]::StartNew()
  & ".\.venv\Scripts\python.exe" tools\run_rl_dqn.py ale_pong --profile fast --seed 0 --device $dev `
     --no-early-stopping --total-timesteps $STEPS --eval-frequency $STEPS `
     --buffer-size 100000 --learning-rate 1e-4 --target-network-frequency 1000 `
     --train-frequency 4 --exploration-fraction 0.1 --learning-starts 1000 `
     --output-dir results/_alebench > "p19_$dev.out" 2>&1
  $sw.Stop(); $rc=$LASTEXITCODE
  if ($rc -eq 0) {
    Say ("  dqn {0}: {1:N1}s => {2:N2} steps/s | 1M ETA {3:N1} h" -f $dev,$sw.Elapsed.TotalSeconds,($STEPS/$sw.Elapsed.TotalSeconds),(1000000/($STEPS/$sw.Elapsed.TotalSeconds)/3600))
  } else {
    Say ("  dqn {0}: FAILED exit={1}" -f $dev,$rc)
    Say ("    " + ((Get-Content "p19_$dev.out" -EA SilentlyContinue | Select-String 'Error|RuntimeError' | Select-Object -First 2) -join ' | '))
  }
}
Remove-Item -Recurse -Force results\_alebench -EA SilentlyContinue
Say 'BENCH COMPLETE'
