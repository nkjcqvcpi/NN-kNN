# p20: ALE arms on the Arc GPU (XPU), 500k budget, seeds 0-2.
# XPU chosen from measurement, not assumption: DQN on ALE runs 48.89 steps/s on
# XPU vs 18.93 on CPU (2.6x) because the Nature-CNN dominates. NEC goes the other
# way (4.46 xpu vs 6.94 cpu) because its DND kNN lookup dominates, so NEC stays
# on CPU as p16. Cross-method claims therefore carry a device caveat.
$ErrorActionPreference='Continue'
Set-Location 'C:\Users\Administrator\NN-KNN_RL\nnknn-work'
$LOG='logs/g234/p20_driver.log'
function Say($m){ Add-Content -Path $LOG -Value ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'),$m) }
$env:NNKNN_DEVICE='xpu'; $env:PYTHONUNBUFFERED='1'; $env:MPLBACKEND='Agg'
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'   # leave the cores to p16 NEC

$OUT='results/g234/ale'
New-Item -ItemType Directory -Force -Path $OUT | Out-Null
$MAXCONC = 3
$STEPS = 500000

# corrected Atari hyperparameters, identical to the p15 DQN arm
$DQN = @('--buffer-size','100000','--learning-rate','1e-4','--target-network-frequency','1000',
         '--train-frequency','4','--exploration-fraction','0.1','--learning-starts','20000')

$queue=@()
foreach ($seed in 0,1,2) {
  $queue += [pscustomobject]@{Alg='dqn'; Seed=$seed; Extra=$DQN}
  $queue += [pscustomobject]@{Alg='ppo'; Seed=$seed; Extra=@()}   # PPO defaults are near-standard for Atari
}

$have=@{}
foreach ($d in Get-ChildItem $OUT -Directory -EA SilentlyContinue) {
  if (Test-Path (Join-Path $d.FullName 'summary.json')) {
    try { $c = Get-Content (Join-Path $d.FullName 'config.json') -Raw | ConvertFrom-Json
          $have[("{0}_{1}" -f $c.algorithm, $c.config.seed)] = $true } catch {}
  } else { Remove-Item -Recurse -Force $d.FullName -EA SilentlyContinue }
}
$busy=@{}
foreach ($p in Get-CimInstance Win32_Process -Filter "Name='python.exe'") {
  if ($p.CommandLine -match 'run_rl_(dqn|ppo)\.py.*--seed (\d+)') { $busy[("{0}_{1}" -f $Matches[1],$Matches[2])] = $true } }
Say ("skipping in-flight: " + (($busy.Keys|Sort-Object) -join ' '))

$todo = @($queue | Where-Object { -not $busy.ContainsKey(("{0}_{1}" -f $_.Alg,$_.Seed)) })
Say ("queued {0} ALE runs on XPU at {1} steps, max concurrency {2}" -f $todo.Count,$STEPS,$MAXCONC)

$running=@(); $qi=0
while ($qi -lt $todo.Count -or $running.Count -gt 0) {
  while ($running.Count -lt $MAXCONC -and $qi -lt $todo.Count) {
    $j=$todo[$qi]; $qi++
    $log = "p20_{0}_s{1}.log" -f $j.Alg,$j.Seed
    $a = @("tools\run_rl_$($j.Alg).py",'ale_pong','--profile','fast','--seed',"$($j.Seed)",
           '--device','xpu','--no-early-stopping','--total-timesteps',"$STEPS",
           '--eval-frequency','25000','--output-dir',$OUT) + $j.Extra
    $p = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList $a `
          -RedirectStandardOutput $log -RedirectStandardError ($log -replace '\.log$','.err') `
          -WindowStyle Hidden -PassThru
    $running += [pscustomobject]@{P=$p;Alg=$j.Alg;Seed=$j.Seed}
    Say ("START {0} seed={1} pid={2} ({3}/{4})" -f $j.Alg,$j.Seed,$p.Id,$qi,$todo.Count)
  }
  Start-Sleep -Seconds 30
  $still=@()
  foreach ($e in $running) {
    if ($e.P.HasExited) { Say ("DONE {0} seed={1} exit={2}" -f $e.Alg,$e.Seed,$e.P.ExitCode) } else { $still += $e } }
  $running=$still
}
Say 'ALE CAMPAIGN COMPLETE'
