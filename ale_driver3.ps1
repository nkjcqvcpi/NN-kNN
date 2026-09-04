# p20 v3: ALL ALE arms on CPU. The Arc GPU is abandoned for training.
#
# Evidence: three python.exe crashes in 20 minutes with the faulting module
# inside Intel's Level Zero stack --
#   ze_intel_gpu64.dll 1.15.39183.3   0xc0000005 (access violation)
#   ze_intel_gpu64.dll 1.15.39183.3   0xc0000409 (stack buffer overrun)
#   ur_adapter_level_zero.dll 2026.0.0.0  0xc0000005
# plus RuntimeError 'level_zero backend failed ... UR_RESULT_ERROR_UNKNOWN' at
# two unrelated sites (Adam's _foreach_div_, and a plain x/255.0 in the CNN
# forward). The crash site is arbitrary, so the earlier foreach=False patch
# does not help -- it addressed one symptom of a general driver fault.
# XPU is ~2.6x faster for DQN, but a run that dies at 26k/500k is worth nothing.
$ErrorActionPreference='Continue'
Set-Location 'C:\Users\Administrator\NN-KNN_RL\nnknn-work'
$LOG='p20_driver3.log'
function Say($m){ Add-Content -Path $LOG -Value ("[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'),$m) }
$env:PYTHONUNBUFFERED='1'; $env:MPLBACKEND='Agg'; $env:NNKNN_DEVICE='cpu'
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'
$OUT='results/rl_ale_g234'; $STEPS=500000
New-Item -ItemType Directory -Force -Path $OUT | Out-Null

$DQN = @('--buffer-size','100000','--learning-rate','1e-4','--target-network-frequency','1000',
         '--train-frequency','4','--exploration-fraction','0.1','--learning-starts','20000')
$plan=@()
foreach ($s in 0,1,2) { $plan += [pscustomobject]@{Alg='dqn'; Seed=$s; Extra=$DQN} }
foreach ($s in 0,1,2) { $plan += [pscustomobject]@{Alg='ppo'; Seed=$s; Extra=@()} }

$have=@{}
foreach ($d in Get-ChildItem $OUT -Directory -EA SilentlyContinue) {
  if (Test-Path (Join-Path $d.FullName 'summary.json')) {
    try { $c = Get-Content (Join-Path $d.FullName 'config.json') -Raw | ConvertFrom-Json
          $have[("{0}_{1}" -f $c.algorithm,$c.config.seed)] = $true } catch {}
  } else { Remove-Item -Recurse -Force $d.FullName -EA SilentlyContinue } }
$busy=@{}
foreach ($p in Get-CimInstance Win32_Process -Filter "Name='python.exe'") {
  if ($p.CommandLine -match 'run_rl_(dqn|ppo)\.py.*--seed (\d+)') { $busy[("{0}_{1}" -f $Matches[1],$Matches[2])] = $true } }
Say ("banked: " + (($have.Keys|Sort-Object) -join ' '))
Say ("in flight, left alone: " + (($busy.Keys|Sort-Object) -join ' '))

$todo = @($plan | Where-Object { $k="$($_.Alg)_$($_.Seed)"; -not $have.ContainsKey($k) -and -not $busy.ContainsKey($k) })
Say ("queued {0} runs, all on cpu" -f $todo.Count)
$MAXCONC=5
$running=@(); $qi=0
while ($qi -lt $todo.Count -or $running.Count -gt 0) {
  while ($running.Count -lt $MAXCONC -and $qi -lt $todo.Count) {
    $j=$todo[$qi]; $qi++
    $log = "p20_{0}_s{1}.log" -f $j.Alg,$j.Seed
    $a = @("tools\run_rl_$($j.Alg).py",'ale_pong','--profile','fast','--seed',"$($j.Seed)",
           '--device','cpu','--no-early-stopping','--total-timesteps',"$STEPS",
           '--eval-frequency','25000','--output-dir',$OUT) + $j.Extra
    $p = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList $a `
          -RedirectStandardOutput $log -RedirectStandardError ($log -replace '\.log$','.err') `
          -WindowStyle Hidden -PassThru
    $running += [pscustomobject]@{P=$p;Alg=$j.Alg;Seed=$j.Seed}
    Say ("START {0} seed={1} on cpu pid={2}" -f $j.Alg,$j.Seed,$p.Id) }
  Start-Sleep -Seconds 30
  $still=@()
  foreach ($e in $running) {
    if ($e.P.HasExited) { Say ("DONE {0} seed={1} exit={2}" -f $e.Alg,$e.Seed,$e.P.ExitCode) } else { $still += $e } }
  $running=$still }
Say 'ALE CAMPAIGN COMPLETE'
