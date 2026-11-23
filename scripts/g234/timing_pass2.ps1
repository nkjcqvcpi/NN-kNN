# Recording is mandatory (2026-11-22): every run is a unit of NN-KNN\runs\t2\$STUDY\ (--study);
# rerunning a launch skips a complete unit before any training and resumes an interrupted one, so this
# driver no longer scans or deletes run folders under results/.
$ErrorActionPreference='Continue'
Set-Location 'C:\Users\Administrator\NN-KNN\rl'
$PY='C:\Users\Administrator\NN-KNN\.venv\Scripts\python.exe'   # the shared uv venv
$env:NNKNN_DEVICE='cpu'; $env:PYTHONUNBUFFERED='1'; $env:MPLBACKEND='Agg'
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'
$LOG='logs/g234/p18_timing.log'
function Say($m){ Add-Content -Path $LOG -Value ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'),$m) }
$STEPS=20000
# a fresh study per pass: a complete unit would be read back, not timed
$STUDY='p18_timing_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
New-Item -ItemType Directory -Force -Path results/t2 | Out-Null
$out='results/t2/p18_throughput_controlled.csv'
'capacity,steps,seconds,steps_per_s' | Out-File -Encoding ascii $out
Say "timing pass start: $STEPS steps per capacity, sequential, early stopping OFF"
foreach ($cap in 100,500,1000,2000) {
  Say "measuring cap=$cap"
  $sw=[Diagnostics.Stopwatch]::StartNew()
  & $PY tools\run_rl_nnknn.py cartpole --profile fast --seed 0 --device cpu `
     --critic-type nnknn --critic-mutable-value-labels --critic-trainable-value-labels `
     --case-capacity $cap --total-timesteps $STEPS --no-early-stopping `
     --study $STUDY --unit "nnknn_rl/cartpole/cap$cap" 2>&1 | Out-Null
  $sw.Stop(); $secs=$sw.Elapsed.TotalSeconds
  $line = "{0},{1},{2},{3}" -f $cap,$STEPS,
          $secs.ToString('F2',[cultureinfo]::InvariantCulture),
          ($STEPS/$secs).ToString('F2',[cultureinfo]::InvariantCulture)
  $line | Out-File -Encoding ascii -Append $out
  Say "  $line"
}
Say 'TIMING PASS COMPLETE'
