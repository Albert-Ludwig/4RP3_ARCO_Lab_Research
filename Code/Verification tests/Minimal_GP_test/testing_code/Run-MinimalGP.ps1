param(
    [string]$HostPython = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
    [string]$ReportPython = 'C:\Ubuntu\ROS\reporting-venv\Scripts\python.exe'
)
$ErrorActionPreference = 'Stop'
foreach ($runtime in @($HostPython, $ReportPython)) {
    if (-not (Test-Path -LiteralPath $runtime)) { throw "Missing runtime: $runtime. Set -HostPython / -ReportPython." }
}
& $HostPython -B (Join-Path $PSScriptRoot 'run_test.py')
if ($LASTEXITCODE -ne 0) { throw 'GP experiment failed; inspect the saved execution log.' }
& $ReportPython -B (Join-Path $PSScriptRoot 'build_report.py')
if ($LASTEXITCODE -ne 0) { throw 'Report verification/build failed.' }
