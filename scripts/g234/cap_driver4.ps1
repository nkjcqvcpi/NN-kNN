# p17 driver v4. Two fixes over v3:
#  1. Skips (capacity, seed) pairs that are ALREADY RUNNING, not just banked,
#     so it can be started alongside in-flight jobs without duplicating them.
#  2. Logs with Add-Content (open/append/close) instead of relying on a held
#     stdout redirect. v3 hung with its loop stopped and 0 CPU after its first
#     flush -- a WMI-spawned, console-less powershell blocking on the inherited
#     redirected handle. Nothing drains that handle, so the write never returns.
$ErrorActionPreference = 'Continue'
Set-Location 'C:\Users\Administrator\NN-KNN_RL\nnknn-work'
$env:NNKNN_DEVICE='cpu'; $env:PYTHONUNBUFFERED='1'; $env:MPLBACKEND='Agg'
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'

$OUT='results/g234/capacity_cpu'
$LOG='logs/g234/p17_driver4.log'
$MAXCONC = 6
function Say($m){ Add-Content -Path $LOG -Value ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m) }

Say "driver v4 starting"

$have=@{}
foreach ($d in Get-ChildItem $OUT -Directory -EA SilentlyContinue) {
  if (Test-Path (Join-Path $d.FullName 'summary.json')) {
    try { $c = Get-Content (Join-Path $d.FullName 'config.json') -Raw | ConvertFrom-Json
          $have[("{0}_{1}" -f $c.config.case_capacity,$c.config.seed)] = $true } catch {}
  }
}
$busy=@{}
foreach ($p in Get-CimInstance Win32_Process -Filter "Name='python.exe'") {
  if ($p.CommandLine -match '--seed (\d+).*--case-capacity (\d+)') { $busy[("{0}_{1}" -f $Matches[2],$Matches[1])] = $true }
}
Say ("banked: " + (($have.Keys|Sort-Object) -join ' '))
Say ("in flight (left alone): " + (($busy.Keys|Sort-Object) -join ' '))

$queue=@()
foreach ($seed in 0,1,2,3,4) { foreach ($cap in 2000,500,1000,100) {
  $k = "${cap}_${seed}"
  if (-not $have.ContainsKey($k) -and -not $busy.ContainsKey($k)) { $queue += [pscustomobject]@{Seed=$seed;Cap=$cap} } } }
Say ("queued {0} runs, max concurrency {1}" -f $queue.Count,$MAXCONC)
if ($queue.Count -eq 0) { Say 'NOTHING TO DO'; exit 0 }

$running=@(); $qi=0
while ($qi -lt $queue.Count -or $running.Count -gt 0) {
  while ($running.Count -lt $MAXCONC -and $qi -lt $queue.Count) {
    $j=$queue[$qi]; $qi++
    $log = "p17_cap{0}_s{1}.log" -f $j.Cap,$j.Seed
    $a = @('tools\run_rl_nnknn.py','cartpole','--profile','fast','--seed',"$($j.Seed)",'--device','cpu',
           '--critic-type','nnknn','--critic-mutable-value-labels','--critic-trainable-value-labels',
           '--case-capacity',"$($j.Cap)",'--output-dir',$OUT)
    $p = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList $a `
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
