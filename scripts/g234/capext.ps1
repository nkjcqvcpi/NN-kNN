# p22: extend the capacity curve with cache-friendly CartPole runs.
# Rationale: ALE jobs starve NEC through memory bandwidth (84x84x4 frames,
# 100k replay buffers). CartPole case bases of 250-3000 entries stay in cache
# and scaled ~10x with concurrency in p17, so these should add aggregate
# throughput without taking NEC's bandwidth. Verified by measuring NEC before
# and after in the launching script.
# Recording is mandatory (2026-11-22): every run is a unit of NN-KNN\runs\t2\$STUDY\ (--study);
# rerunning a launch skips a complete unit before any training and resumes an interrupted one, so this
# driver no longer scans or deletes run folders under results/.
$ErrorActionPreference='Continue'
Set-Location 'C:\Users\Administrator\NN-KNN\rl'
$PY='C:\Users\Administrator\NN-KNN\.venv\Scripts\python.exe'   # the shared uv venv
$LOG='logs/g234/p22_driver.log'
function Say($m){ Add-Content -Path $LOG -Value ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'),$m) }
$env:NNKNN_DEVICE='cpu'; $env:PYTHONUNBUFFERED='1'; $env:MPLBACKEND='Agg'
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'   # same as p17, so results are comparable
$STUDY='p17_capacity_cpu'
$MAXCONC=8
$queue=@()
foreach ($seed in 0,1,2,3,4) { foreach ($cap in 250,1500) { $queue += [pscustomobject]@{Seed=$seed;Cap=$cap} } }
$busy=@{}
foreach ($p in Get-CimInstance Win32_Process -Filter "Name='python.exe'") {
  if ($p.CommandLine -match 'run_rl_nnknn\.py.*--seed (\d+).*--case-capacity (\d+)') { $busy[("{0}_{1}" -f $Matches[2],$Matches[1])] = $true } }
$todo = @($queue | Where-Object { $k="$($_.Cap)_$($_.Seed)"; -not $busy.ContainsKey($k) })
Say ("queued {0} capacity runs (cap 250 and 1500 x seeds 0-4), max concurrency {1}" -f $todo.Count,$MAXCONC)
$running=@(); $qi=0
while ($qi -lt $todo.Count -or $running.Count -gt 0) {
  while ($running.Count -lt $MAXCONC -and $qi -lt $todo.Count) {
    $j=$todo[$qi]; $qi++
    $log = "p22_cap{0}_s{1}.log" -f $j.Cap,$j.Seed
    $a = @('tools\run_rl_nnknn.py','cartpole','--profile','fast','--seed',"$($j.Seed)",'--device','cpu',
           '--critic-type','nnknn','--critic-mutable-value-labels','--critic-trainable-value-labels',
           '--case-capacity',"$($j.Cap)",'--study',$STUDY,'--unit',"nnknn_rl/cartpole/cap$($j.Cap)_s$($j.Seed)")
    $p = Start-Process -FilePath $PY -ArgumentList $a `
          -RedirectStandardOutput $log -RedirectStandardError ($log -replace '\.log$','.err') `
          -WindowStyle Hidden -PassThru
    $running += [pscustomobject]@{P=$p;Cap=$j.Cap;Seed=$j.Seed}
    Say ("START cap={0} seed={1} pid={2} ({3}/{4})" -f $j.Cap,$j.Seed,$p.Id,$qi,$todo.Count) }
  Start-Sleep -Seconds 30
  $still=@()
  foreach ($e in $running) { if ($e.P.HasExited) { Say ("DONE cap={0} seed={1} exit={2}" -f $e.Cap,$e.Seed,$e.P.ExitCode) } else { $still += $e } }
  $running=$still }
Say 'CAPACITY EXTENSION COMPLETE'
