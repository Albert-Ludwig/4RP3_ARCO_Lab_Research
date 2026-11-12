$ErrorActionPreference = 'Stop'
$testRoot = Split-Path $PSScriptRoot -Parent
$repoRoot = Split-Path (Split-Path (Split-Path $testRoot -Parent) -Parent) -Parent
$relative = $testRoot.Substring($repoRoot.Length + 1).Replace('\', '/')
$docker = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
& $docker exec -e OPENBLAS_NUM_THREADS=4 -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 bunker-jazzy /home/ubuntu/.venvs/srf-verification/bin/python ('/workspace/' + $relative + '/testing_code/verify.py')
if ($LASTEXITCODE -ne 0) { throw 'Continuous sensing verification failed; inspect its new run directory.' }
& 'C:\Ubuntu\ROS\reporting-venv\Scripts\python.exe' (Join-Path $PSScriptRoot 'build_report.py')
if ($LASTEXITCODE -ne 0) { throw 'Report generation failed.' }
