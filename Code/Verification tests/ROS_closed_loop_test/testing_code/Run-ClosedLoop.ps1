$ErrorActionPreference = 'Stop'
$docker = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
& $docker exec bunker-jazzy bash '/workspace/Code/Verification tests/ROS_closed_loop_test/testing_code/run.sh'
if ($LASTEXITCODE -ne 0) { throw 'Closed-loop run or numerical audit failed; inspect the saved result.' }
& 'C:\Ubuntu\ROS\reporting-venv\Scripts\python.exe' (Join-Path $PSScriptRoot 'report.py')
if ($LASTEXITCODE -ne 0) { throw 'Notebook/video generation failed.' }
