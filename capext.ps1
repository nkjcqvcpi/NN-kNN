# p22: extend the capacity curve with cache-friendly CartPole runs.
# Rationale: ALE jobs starve NEC through memory bandwidth (84x84x4 frames,
# 100k replay buffers). CartPole case bases of 250-3000 entries stay in cache
# and scaled ~10x with concurrency in p17, so these should add aggregate
# throughput without taking NEC's bandwidth. Verified by measuring NEC before
# and after in the launching script.
$ErrorActionPreference='Continue'
Set-Location 'C:\Users\Administrator\NN-KNN_RL\nnknn-work'
$LOG='p22_driver.log'
function Say($m){ Add-Content -Path $LOG -Value ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'),$m) }
$env:NNKNN_DEVICE='cpu'; $env:PYTHONUNBUFFERED='1'; $env:MPLBACKEND='Agg'
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'   # same as p17, so results are comparable
$OUT='results/rl_capacity_cpu_g234'
$MAXCONC=8
$queue=@()
foreach ($seed in 0,1,2,3,4) { foreach ($cap in 250,1500) { $queue += [pscustomobject]@{Seed=$seed;Cap=$cap} } }
$have=@{}
foreach ($d in Get-ChildItem $OUT -Directory -EA SilentlyContinue) {
  if (Test-Path (Join-Path $d.FullName 'summary.json')) {
    try { $c = Get-Content (Join-Path $d.FullName 'config.json') -Raw | ConvertFrom-Json
          $have[("{0}_{1}" -f $c.config.case_capacity,$c.config.seed)] = $true } catch {} } }
$busy=@{}
foreach ($p in Get-CimInstance Win32_Process -Filter "Name='python.exe'") {
  if ($p.CommandLine -match 'run_rl_nnknn\.py.*--seed (\d+).*--case-capacity (\d+)') { $busy[("{0}_{1}" -f $Matches[2],$Matches[1])] = $true } }
$todo = @($queue | Where-Object { $k="$($_.Cap)_$($_.Seed)"; -not $have.ContainsKey($k) -and -not $busy.ContainsKey($k) })
Say ("queued {0} capacity runs (cap 250 and 1500 x seeds 0-4), max concurrency {1}" -f $todo.Count,$MAXCONC)
$running=@(); $qi=0
while ($qi -lt $todo.Count -or $running.Count -gt 0) {
  while ($running.Count -lt $MAXCONC -and $qi -lt $todo.Count) {
    $j=$todo[$qi]; $qi++
    $log = "p22_cap{0}_s{1}.log" -f $j.Cap,$j.Seed
    $a = @('tools\run_rl_nnknn.py','cartpole','--profile','fast','--seed',"$($j.Seed)",'--device','cpu',
           '--critic-type','nnknn','--critic-mutable-value-labels','--critic-trainable-value-labels',
           '--case-capacity',"$($j.Cap)",'--output-dir',$OUT)
    $p = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList $a `
          -RedirectStandardOutput $log -RedirectStandardError ($log -replace '\.log$','.err') `
          -WindowStyle Hidden -PassThru
    $running += [pscustomobject]@{P=$p;Cap=$j.Cap;Seed=$j.Seed}
    Say ("START cap={0} seed={1} pid={2} ({3}/{4})" -f $j.Cap,$j.Seed,$p.Id,$qi,$todo.Count) }
  Start-Sleep -Seconds 30
  $still=@()
  foreach ($e in $running) { if ($e.P.HasExited) { Say ("DONE cap={0} seed={1} exit={2}" -f $e.Cap,$e.Seed,$e.P.ExitCode) } else { $still += $e } }
  $running=$still }
Say 'CAPACITY EXTENSION COMPLETE'
