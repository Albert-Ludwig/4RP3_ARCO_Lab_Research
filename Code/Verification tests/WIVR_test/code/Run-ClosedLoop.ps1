$ErrorActionPreference='Stop'
& 'C:\Program Files\Docker\Docker\resources\bin\docker.exe' --context desktop-linux exec bunker-jazzy bash '/workspace/Code/Verification tests/WIVR_test/code/run.sh'
if($LASTEXITCODE -ne 0){throw 'WIVR sampling and return experiment failed; inspect saved evidence.'}
& 'C:\Ubuntu\ROS\reporting-venv\Scripts\python.exe' (Join-Path $PSScriptRoot 'report.py')
if($LASTEXITCODE -ne 0){throw 'Report failed.'}
