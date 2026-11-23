# p17 driver v4. Two fixes over v3:
#  1. Skips (capacity, seed) pairs that are ALREADY RUNNING, not just banked,
#     so it can be started alongside in-flight jobs without duplicating them.
#  2. Logs with Add-Content (open/append/close) instead of relying on a held
#     stdout redirect. v3 hung with its loop stopped and 0 CPU after its first
#     flush -- a WMI-spawned, console-less powershell blocking on the inherited
#     redirected handle. Nothing drains that handle, so the write never returns.
# Recording is mandatory (2026-11-22): every run is a unit of NN-KNN\runs\t2\$STUDY\ (--study);
# rerunning a launch skips a complete unit before any training and resumes an interrupted one, so this
# driver no longer scans or deletes run folders under results/.
$ErrorActionPreference = 'Continue'
Set-Location 'C:\Users\Administrator\NN-KNN\rl'
$PY='C:\Users\Administrator\NN-KNN\.venv\Scripts\python.exe'   # the shared uv venv
$env:NNKNN_DEVICE='cpu'; $env:PYTHONUNBUFFERED='1'; $env:MPLBACKEND='Agg'
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'

$STUDY='p17_capacity_cpu'
$LOG='logs/g234/p17_driver4.log'
$MAXCONC = 6
function Say($m){ Add-Content -Path $LOG -Value ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m) }

Say "driver v4 starting"

$busy=@{}
foreach ($p in Get-CimInstance Win32_Process -Filter "Name='python.exe'") {
  if ($p.CommandLine -match '--seed (\d+).*--case-capacity (\d+)') { $busy[("{0}_{1}" -f $Matches[2],$Matches[1])] = $true }
}
Say ("in flight (left alone): " + (($busy.Keys|Sort-Object) -join ' '))

$queue=@()
foreach ($seed in 0,1,2,3,4) { foreach ($cap in 2000,500,1000,100) {
  $k = "${cap}_${seed}"
  if (-not $busy.ContainsKey($k)) { $queue += [pscustomobject]@{Seed=$seed;Cap=$cap} } } }
Say ("queued {0} runs, max concurrency {1}" -f $queue.Count,$MAXCONC)
if ($queue.Count -eq 0) { Say 'NOTHING TO DO'; exit 0 }

$running=@(); $qi=0
while ($qi -lt $queue.Count -or $running.Count -gt 0) {
  while ($running.Count -lt $MAXCONC -and $qi -lt $queue.Count) {
    $j=$queue[$qi]; $qi++
    $log = "p17_cap{0}_s{1}.log" -f $j.Cap,$j.Seed
    $a = @('tools\run_rl_nnknn.py','cartpole','--profile','fast','--seed',"$($j.Seed)",'--device','cpu',
           '--critic-type','nnknn','--critic-mutable-value-labels','--critic-trainable-value-labels',
           '--case-capacity',"$($j.Cap)",'--study',$STUDY,'--unit',"nnknn_rl/cartpole/cap$($j.Cap)_s$($j.Seed)")
    $p = Start-Process -FilePath $PY -ArgumentList $a `
          -RedirectStandardOutput $log -RedirectStandardError ($log -replace '\.log$','.err') `
          -WindowStyle Hidden -PassThru
    $running += [pscustomobject]@{P=$p;Seed=$j.Seed;Cap=$j.Cap}
    Say ("START seed={0} cap={1} pid={2} ({3}/{4})" -f $j.Seed,$j.Cap,$p.Id,$qi,$queue.Count)
  }
  Start-Sleep -Seconds 20
  $still=@()
  foreach ($e in $running) {
    if ($e.P.HasExited) { Say ("DONE seed={0} cap={1} exit={2}" -f $e.Seed,$e.Cap,$e.P.ExitCode) } else { $still += $e }
  }
  $running=$still
}
Say 'CAPACITY CAMPAIGN COMPLETE'
