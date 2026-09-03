# p17 repacked driver: remaining seeds at high concurrency.
# OMP_NUM_THREADS stays 2 (as in the seed-0 batch) so returns are bit-identical
# regardless of how many jobs run at once -- verified by replaying cap100 seed 0
# under different concurrency and getting the same 500.00 / 24281 steps.
# Wall clock from this driver is NOT a throughput measurement; steps/s comes
# from the separate sequential timing pass (timing_pass.ps1).
$ErrorActionPreference = 'Continue'
Set-Location 'C:\Users\Administrator\NN-KNN_RL\nnknn-work'
$env:NNKNN_DEVICE='cpu'; $env:PYTHONUNBUFFERED='1'; $env:MPLBACKEND='Agg'
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'

$OUT='results/rl_capacity_cpu_g234'
$MAXCONC = 8
$queue = @()
foreach ($seed in 1,2,3,4) { foreach ($cap in 2000,500,1000,100) {   # slowest first
  $queue += [pscustomobject]@{ Seed=$seed; Cap=$cap }
} }
Write-Output ("queued {0} runs, max concurrency {1}" -f $queue.Count, $MAXCONC)

$running = @(); $qi = 0
while ($qi -lt $queue.Count -or $running.Count -gt 0) {
  while ($running.Count -lt $MAXCONC -and $qi -lt $queue.Count) {
    $j = $queue[$qi]; $qi++
    $log = "p17_cap{0}_s{1}.log" -f $j.Cap, $j.Seed
    $a = @('tools\run_rl_nnknn.py','cartpole','--profile','fast','--seed',"$($j.Seed)",'--device','cpu',
           '--critic-type','nnknn','--critic-mutable-value-labels','--critic-trainable-value-labels',
           '--case-capacity',"$($j.Cap)",'--output-dir',$OUT)
    $p = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList $a `
          -RedirectStandardOutput $log -RedirectStandardError ($log -replace '\.log$','.err') `
          -WindowStyle Hidden -PassThru
    $running += [pscustomobject]@{ P=$p; Seed=$j.Seed; Cap=$j.Cap }
    Write-Output ("  START seed={0} cap={1} pid={2}  ({3}/{4} dispatched)" -f $j.Seed,$j.Cap,$p.Id,$qi,$queue.Count)
  }
  Start-Sleep -Seconds 20
  $still = @()
  foreach ($e in $running) {
    if ($e.P.HasExited) { Write-Output ("  DONE  seed={0} cap={1} exit={2}" -f $e.Seed,$e.Cap,$e.P.ExitCode) }
    else { $still += $e }
  }
  $running = $still
}
Write-Output 'CAPACITY CAMPAIGN COMPLETE'
