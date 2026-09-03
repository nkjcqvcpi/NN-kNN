$ErrorActionPreference = 'Continue'
Set-Location "$env:USERPROFILE\NN-KNN_RL\nnknn-work"
$env:NNKNN_DEVICE='cpu'; $env:PYTHONUNBUFFERED='1'; $env:MPLBACKEND='Agg'
# Constant across every run in the campaign so neither the returns nor the
# throughput column is confounded by thread count.
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'

$OUT  = 'results/rl_capacity_cpu_g234'
$CAPS = @(100, 500, 1000, 2000)
$SEEDS= @(0, 1, 2, 3, 4)
New-Item -ItemType Directory -Force -Path $OUT | Out-Null
$ledger = Join-Path $OUT 'campaign_timing.csv'
if (-not (Test-Path $ledger)) { 'seed,capacity,seconds,steps_per_s_nominal,exit_code' | Out-File -Encoding ascii $ledger }

foreach ($seed in $SEEDS) {
  Write-Output ("==== batch seed={0} : launching caps {1} concurrently ====" -f $seed, ($CAPS -join ','))
  $procs = @()
  foreach ($cap in $CAPS) {
    $log = "p17_cap{0}_s{1}.log" -f $cap, $seed
    $jobArgs = @(
      'tools\run_rl_nnknn.py','cartpole','--profile','fast','--seed',"$seed",'--device','cpu',
      '--critic-type','nnknn','--critic-mutable-value-labels','--critic-trainable-value-labels',
      '--case-capacity',"$cap",'--output-dir',$OUT
    )
    $p = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList $jobArgs `
          -RedirectStandardOutput $log -RedirectStandardError ($log -replace '\.log$','.err') `
          -WindowStyle Hidden -PassThru
    $procs += [pscustomobject]@{ Proc=$p; Cap=$cap; Seed=$seed; Start=(Get-Date) }
    Write-Output ("   launched cap={0} seed={1} pid={2}" -f $cap, $seed, $p.Id)
  }
  foreach ($e in $procs) {
    $e.Proc.WaitForExit()
    $secs = ((Get-Date) - $e.Start).TotalSeconds
    $sps  = 150000 / $secs
    ("{0},{1},{2:N1},{3:N2},{4}" -f $e.Seed,$e.Cap,$secs,$sps,$e.Proc.ExitCode) |
      Out-File -Encoding ascii -Append $ledger
    Write-Output ("   done cap={0} seed={1} exit={2} elapsed={3:N1}s" -f $e.Cap,$e.Seed,$e.Proc.ExitCode,$secs)
  }
  Write-Output ("==== batch seed={0} complete ====" -f $seed)
}
Write-Output 'CAPACITY CAMPAIGN COMPLETE'
