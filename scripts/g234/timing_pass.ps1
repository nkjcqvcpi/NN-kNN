# Controlled throughput measurement for the capacity ablation.
# Run ONLY when nothing else is on the box. One job at a time, fixed step
# budget, early stopping OFF so every capacity does identical work. This
# replaces the contended steps/s numbers the campaign driver cannot produce.
# Recording is mandatory (2026-11-22): every run is a unit of NN-KNN\runs\t2\$STUDY\ (--study).
# Deliberate exception to "never retrain" (O4): this is a timing measurement, not a result. Each pass uses a
# fresh timestamped study, so it trains identical configs again on purpose (a complete unit would be read back,
# not timed). Its units are not results; do not summarise them. The timings include the recording overhead
# (event files, resume checkpoints), so they are not directly comparable with the earlier p17/p18/p19 timings.
$ErrorActionPreference='Continue'
Set-Location 'C:\Users\Administrator\NN-KNN\rl'
$PY='C:\Users\Administrator\NN-KNN\.venv\Scripts\python.exe'   # the shared uv venv
$env:NNKNN_DEVICE='cpu'; $env:PYTHONUNBUFFERED='1'; $env:MPLBACKEND='Agg'
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'
$STEPS = 20000
# a fresh study per pass: a complete unit would be read back, not timed
$STUDY = 'p17_timing_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
New-Item -ItemType Directory -Force -Path results/t2 | Out-Null
$out = 'results/t2/p17_throughput_controlled.csv'
'capacity,steps,seconds,steps_per_s' | Out-File -Encoding ascii $out
foreach ($cap in 100,500,1000,2000) {
  Write-Output ("measuring cap={0} ({1} steps, sequential, idle box)" -f $cap,$STEPS)
  $sw=[Diagnostics.Stopwatch]::StartNew()
  & $PY tools\run_rl_nnknn.py cartpole --profile fast --seed 0 --device cpu `
     --critic-type nnknn --critic-mutable-value-labels --critic-trainable-value-labels `
     --case-capacity $cap --total-timesteps $STEPS --no-early-stopping `
     --study $STUDY --unit "nnknn_rl/cartpole/cap$cap" 2>&1 | Out-Null
  $sw.Stop()
  $secs = $sw.Elapsed.TotalSeconds
  # invariant culture: no thousands separator, so the CSV stays parseable
  $line = "{0},{1},{2},{3}" -f $cap, $STEPS,
          $secs.ToString('F2',[cultureinfo]::InvariantCulture),
          ($STEPS/$secs).ToString('F2',[cultureinfo]::InvariantCulture)
  $line | Out-File -Encoding ascii -Append $out
  Write-Output ("  " + $line)
}
Write-Output 'TIMING PASS COMPLETE'
Get-Content $out
