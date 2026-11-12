param(
    [string]$DataPython = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
    [string]$ReportPython = 'C:\Ubuntu\ROS\reporting-venv\Scripts\python.exe'
)
$ErrorActionPreference = 'Stop'
foreach ($runtime in @($DataPython, $ReportPython)) {
    if (-not (Test-Path -LiteralPath $runtime)) { throw "Python runtime unavailable: $runtime. Supply -DataPython / -ReportPython." }
}
& $DataPython -B (Join-Path $PSScriptRoot 'prepare_data.py')
if ($LASTEXITCODE -ne 0) { throw 'Data preparation failed.' }
& $DataPython -B (Join-Path $PSScriptRoot 'test_data.py')
if ($LASTEXITCODE -ne 0) { throw 'Processing validation failed.' }
& $ReportPython -B (Join-Path $PSScriptRoot 'build_report.py')
if ($LASTEXITCODE -ne 0) { throw 'Report generation failed.' }
