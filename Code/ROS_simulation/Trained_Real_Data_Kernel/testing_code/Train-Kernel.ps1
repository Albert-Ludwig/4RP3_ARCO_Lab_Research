param(
    [string]$HostPython = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
    [string]$ReportPython = 'C:\Ubuntu\ROS\reporting-venv\Scripts\python.exe'
)
$ErrorActionPreference = 'Stop'
foreach ($runtime in @($HostPython, $ReportPython)) {
    if (-not (Test-Path -LiteralPath $runtime)) { throw "Missing Python: $runtime" }
}
$runner = Join-Path $PSScriptRoot '..\..\Minimal_GP_test\testing_code\run_test.py'
& $HostPython -B $runner --full-fit
if ($LASTEXITCODE -ne 0) { throw 'Full-data training failed.' }
& $ReportPython -B (Join-Path $PSScriptRoot 'build_report.py')
if ($LASTEXITCODE -ne 0) { throw 'Model report generation failed.' }
