# p20 v4: ALE arms on CPU, capped at 2 concurrent.
#
# Why 2: filling the box starved p16 NEC from 3.31 cores to 0.44 (its remaining
# 289k steps would have taken 100+ h) and NEC cannot be paused -- the workflow
# writes artifacts only at completion, so stopping it forfeits 211k steps.
# The host also tops out near 3.6 cores regardless of job count, so extra jobs
# redistribute CPU rather than add capacity.
#
# Order is PPO before DQN: PPO measures ~61 steps/s on CPU against DQN's ~19,
# so shortest-job-first clears the queue sooner.
$ErrorActionPreference='Continue'
Set-Location 'C:\Users\Administrator\NN-KNN_RL\nnknn-work'
$LOG='logs/g234/p20_driver4.log'
function Say($m){ Add-Content -Path $LOG -Value ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'),$m) }
$env:PYTHONUNBUFFERED='1'; $env:MPLBACKEND='Agg'; $env:NNKNN_DEVICE='cpu'
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'
$OUT='results/g234/ale'; $STEPS=500000; $MAXCONC=8
New-Item -ItemType Directory -Force -Path $OUT | Out-Null
$DQN = @('--buffer-size','100000','--learning-rate','1e-4','--target-network-frequency','1000',
         '--train-frequency','4','--exploration-fraction','0.1','--learning-starts','20000')
$plan=@()
foreach ($s in 0,1,2,3,4) { $plan += [pscustomobject]@{Alg='ppo'; Seed=$s; Extra=@()} }
foreach ($s in 0,1,2,3,4) { $plan += [pscustomobject]@{Alg='dqn'; Seed=$s; Extra=$DQN} }

$have=@{}
foreach ($d in Get-ChildItem $OUT -Directory -EA SilentlyContinue) {
  if (Test-Path (Join-Path $d.FullName 'summary.json')) {
    try { $c = Get-Content (Join-Path $d.FullName 'config.json') -Raw | ConvertFrom-Json
          # config.algorithm is e.g. 'ppo_clipped_surrogate_gae'; the plan uses
          # 'ppo'. Normalise to the leading family or nothing ever matches and
          # completed runs get re-run.
          $fam = ($c.algorithm -split '_')[0]
          $have[("{0}_{1}" -f $fam,$c.config.seed)] = $true } catch {}
  }
  # NEVER delete run dirs here. A running job's dir has no summary.json yet, so
  # deleting on that test destroys live output -- it killed ppo seed 0 after a
  # full 500k-step run. The harvest already skips dirs without a summary.json.
}
$busy=@{}
foreach ($p in Get-CimInstance Win32_Process -Filter "Name='python.exe'") {
  if ($p.CommandLine -match 'run_rl_(dqn|ppo)\.py.*--seed (\d+)') { $busy[("{0}_{1}" -f $Matches[1],$Matches[2])] = $true } }
Say ("banked: " + (($have.Keys|Sort-Object) -join ' '))
Say ("in flight, kept: " + (($busy.Keys|Sort-Object) -join ' '))
$todo = @($plan | Where-Object { $k="$($_.Alg)_$($_.Seed)"; -not $have.ContainsKey($k) -and -not $busy.ContainsKey($k) })
Say ("queued {0}, max concurrency {1} (counting the {2} already in flight)" -f $todo.Count,$MAXCONC,$busy.Count)

$running=@(); $qi=0
while ($qi -lt $todo.Count -or $running.Count -gt 0) {
  # count live ALE jobs system-wide so the kept ones occupy slots too
  $liveAll = (Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
              Where-Object { $_.CommandLine -match 'run_rl_(dqn|ppo)\.py' -and $_.WorkingSetSize -gt 300MB } | Measure-Object).Count
  while ($liveAll -lt $MAXCONC -and $qi -lt $todo.Count) {
    $j=$todo[$qi]; $qi++
    $log = "p20_{0}_s{1}.log" -f $j.Alg,$j.Seed
    $a = @("tools\run_rl_$($j.Alg).py",'ale_pong','--profile','fast','--seed',"$($j.Seed)",
           '--device','cpu','--no-early-stopping','--total-timesteps',"$STEPS",
           '--eval-frequency','25000','--output-dir',$OUT) + $j.Extra
    $p = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList $a `
          -RedirectStandardOutput $log -RedirectStandardError ($log -replace '\.log$','.err') `
          -WindowStyle Hidden -PassThru
    $running += [pscustomobject]@{P=$p;Alg=$j.Alg;Seed=$j.Seed}
    Say ("START {0} seed={1} pid={2}" -f $j.Alg,$j.Seed,$p.Id)
    $liveAll++ }
  Start-Sleep -Seconds 60
  $still=@()
  foreach ($e in $running) {
    if ($e.P.HasExited) { Say ("DONE {0} seed={1} exit={2}" -f $e.Alg,$e.Seed,$e.P.ExitCode) } else { $still += $e } }
  $running=$still }
Say 'ALE CAMPAIGN COMPLETE'
