$ErrorActionPreference='Continue'
Set-Location 'C:\Users\Administrator\NN-KNN_RL\nnknn-work'
$env:NNKNN_DEVICE='cpu'; $env:PYTHONUNBUFFERED='1'; $env:MPLBACKEND='Agg'
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'
$LOG='p18_timing.log'
function Say($m){ Add-Content -Path $LOG -Value ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'),$m) }
$STEPS=20000
$out='reports/p18_throughput_controlled.csv'
'capacity,steps,seconds,steps_per_s' | Out-File -Encoding ascii $out
Say "timing pass start: $STEPS steps per capacity, sequential, early stopping OFF"
foreach ($cap in 100,500,1000,2000) {
  Say "measuring cap=$cap"
  $sw=[Diagnostics.Stopwatch]::StartNew()
  & ".\.venv\Scripts\python.exe" tools\run_rl_nnknn.py cartpole --profile fast --seed 0 --device cpu `
     --critic-type nnknn --critic-mutable-value-labels --critic-trainable-value-labels `
     --case-capacity $cap --total-timesteps $STEPS --no-early-stopping `
     --output-dir results/_timing 2>&1 | Out-Null
  $sw.Stop(); $secs=$sw.Elapsed.TotalSeconds
  $line = "{0},{1},{2},{3}" -f $cap,$STEPS,
          $secs.ToString('F2',[cultureinfo]::InvariantCulture),
          ($STEPS/$secs).ToString('F2',[cultureinfo]::InvariantCulture)
  $line | Out-File -Encoding ascii -Append $out
  Say "  $line"
}
Remove-Item -Recurse -Force 'results\_timing' -EA SilentlyContinue
Say 'TIMING PASS COMPLETE'
