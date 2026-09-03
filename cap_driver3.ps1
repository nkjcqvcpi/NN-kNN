# p17 driver, RESUMABLE. g234 bugchecked (0xE6) mid-campaign and lost every
# in-flight run, so this rescans on every start and queues only the
# (capacity, seed) pairs that do not already have a completed run dir.
# Safe to relaunch after any crash. OMP_NUM_THREADS stays 2 so returns are
# bit-identical regardless of concurrency.
$ErrorActionPreference = 'Continue'
Set-Location 'C:\Users\Administrator\NN-KNN_RL\nnknn-work'
$env:NNKNN_DEVICE='cpu'; $env:PYTHONUNBUFFERED='1'; $env:MPLBACKEND='Agg'
$env:OMP_NUM_THREADS='2'; $env:MKL_NUM_THREADS='2'

$OUT='results/rl_capacity_cpu_g234'
$MAXCONC = 8
New-Item -ItemType Directory -Force -Path $OUT | Out-Null

# --- discover what already completed, and drop crash debris ---
$have = @{}
foreach ($d in Get-ChildItem $OUT -Directory) {
  $sp = Join-Path $d.FullName 'summary.json'
  if (Test-Path $sp) {
    try {
      $c = Get-Content (Join-Path $d.FullName 'config.json') -Raw | ConvertFrom-Json
      $have[("{0}_{1}" -f $c.config.case_capacity, $c.config.seed)] = $true
    } catch { }
  } else {
    Write-Output ("  discarding incomplete run dir " + $d.Name)
    Remove-Item -Recurse -Force $d.FullName -EA SilentlyContinue
  }
}
Write-Output ("already complete: {0} runs -> {1}" -f $have.Count, (($have.Keys | Sort-Object) -join ' '))

$queue = @()
foreach ($seed in 0,1,2,3,4) { foreach ($cap in 2000,500,1000,100) {
  if (-not $have.ContainsKey("${cap}_${seed}")) { $queue += [pscustomobject]@{Seed=$seed; Cap=$cap} }
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
           '--case-capacity',"$($j.Cap)",'--output-dir',$OUT)
    $p = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList $a `
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
