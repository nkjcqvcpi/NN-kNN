# Recording is mandatory (2026-11-22): every run is a unit of NN-KNN\runs\t2\$STUDY\ (--study).
# Deliberate exception to "never retrain" (O4): this is a timing measurement, not a result. Each pass uses a
# fresh timestamped study, so it trains identical configs again on purpose (a complete unit would be read back,
# not timed). Its units are not results; do not summarise them. The timings include the recording overhead
# (event files, resume checkpoints), so they are not directly comparable with the earlier p17/p18/p19 timings.
$ErrorActionPreference='Continue'
Set-Location 'C:\Users\Administrator\NN-KNN\rl'
$PY='C:\Users\Administrator\NN-KNN\.venv\Scripts\python.exe'   # the shared uv venv
$LOG='logs/g234/p19_necbench.log'
function Say($m){ Add-Content -Path $LOG -Value ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'),$m) }
$env:PYTHONUNBUFFERED='1'; $env:MPLBACKEND='Agg'
$env:OMP_NUM_THREADS='3'; $env:MKL_NUM_THREADS='3'
$STEPS=4000
# a fresh study per benchmark: a complete unit would be read back, not timed
$STUDY='p19_necbench_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
Say "ALE NEC device benchmark: $STEPS steps, cpu vs xpu"
foreach ($dev in 'cpu','xpu') {
  $env:NNKNN_DEVICE=$dev
  Say "measuring nec on $dev"
  $sw=[Diagnostics.Stopwatch]::StartNew()
  & $PY tools\run_rl_nec.py ale_pong --profile fast --seed 0 --device $dev `
     --no-early-stopping --total-timesteps $STEPS --eval-frequency $STEPS `
     --learning-rate 1e-4 --exploration-fraction 0.1 `
     --study $STUDY --unit "nec/ale_pong/$dev" > "p19_nec_$dev.out" 2>&1
  $sw.Stop(); $rc=$LASTEXITCODE
  if ($rc -eq 0) {
    Say ("  nec {0}: {1:N1}s => {2:N2} steps/s | 500k ETA {3:N1} h" -f $dev,$sw.Elapsed.TotalSeconds,($STEPS/$sw.Elapsed.TotalSeconds),(500000/($STEPS/$sw.Elapsed.TotalSeconds)/3600))
  } else {
    Say ("  nec {0}: FAILED exit={1}" -f $dev,$rc)
    Say ("    " + ((Get-Content "p19_nec_$dev.out" -EA SilentlyContinue | Select-String 'Error|RuntimeError|Traceback' | Select-Object -First 2) -join ' | '))
  }
}
Say 'NEC BENCH COMPLETE'
