[CmdletBinding()]
param(
    [string]$Container='bunker-jazzy',
    [string]$Python='/home/ubuntu/.venvs/policy-compare/bin/python'
)
$ErrorActionPreference='Stop'
& docker --context desktop-linux info --format '{{.ServerVersion}}'
if($LASTEXITCODE -ne 0){throw 'Start Docker Desktop before running this experiment.'}
$running=& docker --context desktop-linux inspect --format '{{.State.Running}}' $Container
if($LASTEXITCODE -ne 0){throw 'The existing bunker-jazzy container was not found.'}
if($running -ne 'true'){
    & docker --context desktop-linux start $Container
    if($LASTEXITCODE -ne 0){throw 'Could not start the existing container.'}
}
$taskScripts='/workspace/Code/11.2_policy_compare/testing_code'
& docker --context desktop-linux exec -u ubuntu -e PYTHONDONTWRITEBYTECODE=1 $Container $Python -B ($taskScripts+'/run_70_30_compare.py')
if($LASTEXITCODE -ne 0){throw 'The 70:30 missions did not finish. Diagnostic outputs are preserved.'}
& docker --context desktop-linux exec -u ubuntu -e PYTHONDONTWRITEBYTECODE=1 $Container $Python -B ($taskScripts+'/build_70_30_report.py')
if($LASTEXITCODE -ne 0){throw 'Report generation failed; raw mission data are preserved.'}
