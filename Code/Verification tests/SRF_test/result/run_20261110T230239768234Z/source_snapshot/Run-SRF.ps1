$ErrorActionPreference = 'Stop'
$testRoot = Split-Path $PSScriptRoot -Parent
$repoRoot = Split-Path (Split-Path (Split-Path $testRoot -Parent) -Parent) -Parent
$docker = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
$python = '/home/ubuntu/.venvs/srf-verification/bin/python'
$relative = $testRoot.Substring($repoRoot.Length + 1).Replace('\', '/')
$containerPath = '/workspace/' + $relative
& $docker exec bunker-jazzy $python ($containerPath + '/testing_code/verify.py')
if ($LASTEXITCODE -ne 0) { throw 'SRF verification failed; inspect its new result directory.' }
$reportPython = 'C:\Ubuntu\ROS\reporting-venv\Scripts\python.exe'
& $reportPython (Join-Path $PSScriptRoot 'build_report.py')
if ($LASTEXITCODE -ne 0) { throw 'SRF report generation failed.' }
