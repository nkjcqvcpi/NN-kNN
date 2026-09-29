# Queue runner for T1 experiments on g234 (runs detached; survives ssh disconnect when started via WMI).
#
#   powershell -File tools\t1_launch.ps1 -Configs p1_retention_synthetic,p2_reuse -MaxParallel 8
#
# Each (config, dataset) pair is one job: tools\t1_run.py <cfg> --only-dataset <ds>.
# Logs: logs\t1\<experiment>__<dataset>.log ; queue status: logs\t1\queue_status.txt
param(
    [string[]]$Configs,
    [int]$MaxParallel = 8
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$env:NNKNN_DEVICE = 'cpu'
$py = Join-Path $repo '.venv\Scripts\python.exe'
New-Item -ItemType Directory -Force logs\t1 | Out-Null
$jobs = @()
foreach ($c in $Configs) {
    $cfgPath = "configs\t1\$c.yaml"
    $names = & $py -c "import yaml,sys; c=yaml.safe_load(open(sys.argv[1])); print('\n'.join(d['name'] for d in c['datasets']))" $cfgPath
    foreach ($n in $names) { if ($n.Trim()) { $jobs += [pscustomobject]@{ cfg = $cfgPath; ds = $n.Trim(); exp = $c } } }
}
$running = @()
$done = 0
$status = 'logs\t1\queue_status.txt'
foreach ($j in $jobs) {
    while (($running | Where-Object { -not $_.HasExited }).Count -ge $MaxParallel) { Start-Sleep -Seconds 5 }
    $log = "logs\t1\$($j.exp)__$($j.ds).log"
    $p = Start-Process -FilePath $py -ArgumentList @('tools\t1_run.py', $j.cfg, '--only-dataset', $j.ds) -RedirectStandardOutput $log -RedirectStandardError "$log.err" -NoNewWindow -PassThru
    $running += $p
    "$(Get-Date -Format s) started $($j.exp)/$($j.ds) pid=$($p.Id)" | Add-Content $status
}
while (($running | Where-Object { -not $_.HasExited }).Count -gt 0) { Start-Sleep -Seconds 10 }
foreach ($p in $running) { $done++ }
"$(Get-Date -Format s) ALL DONE ($done jobs)" | Add-Content $status
