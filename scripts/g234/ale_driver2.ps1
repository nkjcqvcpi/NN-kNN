# p20 v2: ALE arms, each algorithm on the device it measurably prefers.
#   DQN -> XPU   48.89 vs 18.93 steps/s   (Nature-CNN dominates; GPU wins 2.6x)
#   PPO -> CPU   61.62 vs 20.21 steps/s   (no replay buffer, so per-step host<->device
#                                          transfer dominates its short rollouts; GPU loses 3x)
#   NEC -> CPU    6.94 vs  4.46 steps/s   (DND kNN lookup dominates; runs as p16)
# Every arm is internally device-consistent; the cross-method comparison is
# device-split and must say so.
$ErrorActionPreference='Continue'
Set-Location 'C:\Users\Administrator\NN-KNN_RL\nnknn-work'
$LOG='logs/g234/p20_driver2.log'
function Say($m){ Add-Content -Path $LOG -Value ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'),$m) }
$env:PYTHONUNBUFFERED='1'; $env:MPLBACKEND='Agg'
$OUT='results/g234/ale'
$STEPS=500000
New-Item -ItemType Directory -Force -Path $OUT | Out-Null

$DQN = @('--buffer-size','100000','--learning-rate','1e-4','--target-network-frequency','1000',
         '--train-frequency','4','--exploration-fraction','0.1','--learning-starts','20000')
$plan = @()
foreach ($s in 0,1,2) { $plan += [pscustomobject]@{Alg='dqn'; Seed=$s; Dev='xpu'; Extra=$DQN; Threads='2'} }
foreach ($s in 0,1,2) { $plan += [pscustomobject]@{Alg='ppo'; Seed=$s; Dev='cpu'; Extra=@();  Threads='3'} }

$have=@{}
foreach ($d in Get-ChildItem $OUT -Directory -EA SilentlyContinue) {
  if (Test-Path (Join-Path $d.FullName 'summary.json')) {
    try { $c = Get-Content (Join-Path $d.FullName 'config.json') -Raw | ConvertFrom-Json
          $have[("{0}_{1}" -f $c.algorithm,$c.config.seed)] = $true } catch {}
  } else { Remove-Item -Recurse -Force $d.FullName -EA SilentlyContinue }
}
$busy=@{}
foreach ($p in Get-CimInstance Win32_Process -Filter "Name='python.exe'") {
  if ($p.CommandLine -match 'run_rl_(dqn|ppo)\.py.*--seed (\d+)') { $busy[("{0}_{1}" -f $Matches[1],$Matches[2])] = $true } }
Say ("banked: " + (($have.Keys|Sort-Object) -join ' '))
Say ("in flight, left alone: " + (($busy.Keys|Sort-Object) -join ' '))

$todo = @($plan | Where-Object { $k = "$($_.Alg)_$($_.Seed)"; -not $have.ContainsKey($k) -and -not $busy.ContainsKey($k) })
Say ("queued {0} runs" -f $todo.Count)

# XPU and CPU are separate resources, so cap them separately rather than globally.
$MAX = @{ xpu = 3; cpu = 3 }
$running=@(); $qi=0
while ($qi -lt $todo.Count -or $running.Count -gt 0) {
  $progress = $true
  while ($progress -and $qi -lt $todo.Count) {
    $progress = $false
    for ($i=$qi; $i -lt $todo.Count; $i++) {
      $j = $todo[$i]
      $onDev = @($running | Where-Object { $_.Dev -eq $j.Dev }).Count
      if ($onDev -lt $MAX[$j.Dev]) {
        $log = "p20_{0}_s{1}.log" -f $j.Alg,$j.Seed
        $env:NNKNN_DEVICE = $j.Dev; $env:OMP_NUM_THREADS = $j.Threads; $env:MKL_NUM_THREADS = $j.Threads
        $a = @("tools\run_rl_$($j.Alg).py",'ale_pong','--profile','fast','--seed',"$($j.Seed)",
               '--device',$j.Dev,'--no-early-stopping','--total-timesteps',"$STEPS",
               '--eval-frequency','25000','--output-dir',$OUT) + $j.Extra
        $p = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList $a `
              -RedirectStandardOutput $log -RedirectStandardError ($log -replace '\.log$','.err') `
              -WindowStyle Hidden -PassThru
        $running += [pscustomobject]@{P=$p;Alg=$j.Alg;Seed=$j.Seed;Dev=$j.Dev}
        Say ("START {0} seed={1} on {2} pid={3}" -f $j.Alg,$j.Seed,$j.Dev,$p.Id)
        # consume this item
        $tmp = New-Object System.Collections.ArrayList
        for ($k=0; $k -lt $todo.Count; $k++) { if ($k -ne $i) { [void]$tmp.Add($todo[$k]) } }
        $todo = @($tmp); $progress = $true; break
      }
    }
  }
  Start-Sleep -Seconds 30
  $still=@()
  foreach ($e in $running) {
    if ($e.P.HasExited) { Say ("DONE {0} seed={1} on {2} exit={3}" -f $e.Alg,$e.Seed,$e.Dev,$e.P.ExitCode) } else { $still += $e } }
  $running=$still
  if ($todo.Count -eq 0 -and $running.Count -eq 0) { break }
}
Say 'ALE CAMPAIGN COMPLETE'
