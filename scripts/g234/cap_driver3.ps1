# p17 driver, RESUMABLE. g234 bugchecked (0xE6) mid-campaign and lost every
# in-flight run, so this rescans on every start and queues only the
# (capacity, seed) pairs that do not already have a completed run dir.
# Safe to relaunch after any crash. OMP_NUM_THREADS stays 2 so returns are
# bit-identical regardless of concurrency.
# Recording is mandatory (2026-11-22): every run is a unit of NN-KNN\runs\t2\$STUDY\ (--study);
# rerunning a launch skips a complete unit before any training and resumes an interrupted one, so this
# driver no longer scans or deletes run folders under results/.
$ErrorActionPreference = 'Continue'
Set-Location 'C:\Users\Administrator\NN-KNN\rl'
$PY='C:\Users\Administrator\NN-KNN\.venv\Scripts\python.exe'   # the shared uv venv
$env:NNKNN_DEVICE='cpu'; $env:PYTHONUNBUFFERED='1'; $env:MPLBACKEND='Agg'
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'

$STUDY='p17_capacity_cpu'
$MAXCONC = 8

$queue = @()
foreach ($seed in 0,1,2,3,4) { foreach ($cap in 2000,500,1000,100) {
  $queue += [pscustomobject]@{Seed=$seed; Cap=$cap}
} }
Write-Output ("queued {0} remaining runs, max concurrency {1}" -f $queue.Count, $MAXCONC)
if ($queue.Count -eq 0) { Write-Output 'CAPACITY CAMPAIGN COMPLETE'; exit 0 }

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
    Write-Output ("  START seed={0} cap={1} pid={2} ({3}/{4})" -f $j.Seed,$j.Cap,$p.Id,$qi,$queue.Count)
  }
  Start-Sleep -Seconds 20
  $still=@()
  foreach ($e in $running) {
    if ($e.P.HasExited) { Write-Output ("  DONE  seed={0} cap={1} exit={2}" -f $e.Seed,$e.Cap,$e.P.ExitCode) }
    else { $still += $e }
  }
  $running=$still
}
Write-Output 'CAPACITY CAMPAIGN COMPLETE'
