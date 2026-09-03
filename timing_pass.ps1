# Controlled throughput measurement for the capacity ablation.
# Run ONLY when nothing else is on the box. One job at a time, fixed step
# budget, early stopping OFF so every capacity does identical work. This
# replaces the contended steps/s numbers the campaign driver cannot produce.
$ErrorActionPreference='Continue'
Set-Location 'C:\Users\Administrator\NN-KNN_RL\nnknn-work'
$env:NNKNN_DEVICE='cpu'; $env:PYTHONUNBUFFERED='1'; $env:MPLBACKEND='Agg'
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'
$STEPS = 20000
$out = 'reports/p17_throughput_controlled.csv'
'capacity,steps,seconds,steps_per_s' | Out-File -Encoding ascii $out
foreach ($cap in 100,500,1000,2000) {
  Write-Output ("measuring cap={0} ({1} steps, sequential, idle box)" -f $cap,$STEPS)
  $sw=[Diagnostics.Stopwatch]::StartNew()
  & ".\.venv\Scripts\python.exe" tools\run_rl_nnknn.py cartpole --profile fast --seed 0 --device cpu `
     --critic-type nnknn --critic-mutable-value-labels --critic-trainable-value-labels `
     --case-capacity $cap --total-timesteps $STEPS --no-early-stopping `
     --output-dir results/_timing 2>&1 | Out-Null
  $sw.Stop()
  $secs = $sw.Elapsed.TotalSeconds
  # invariant culture: no thousands separator, so the CSV stays parseable
  $line = "{0},{1},{2},{3}" -f $cap, $STEPS,
          $secs.ToString('F2',[cultureinfo]::InvariantCulture),
          ($STEPS/$secs).ToString('F2',[cultureinfo]::InvariantCulture)
  $line | Out-File -Encoding ascii -Append $out
  Write-Output ("  " + $line)
}
Remove-Item -Recurse -Force 'results\_timing' -EA SilentlyContinue
Write-Output 'TIMING PASS COMPLETE'
Get-Content $out
