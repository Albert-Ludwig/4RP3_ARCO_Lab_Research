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
$script='/workspace/Code/11.2_policy_compare/testing_code'
& docker --context desktop-linux exec -u ubuntu -e PYTHONDONTWRITEBYTECODE=1 $Container $Python -B ($script+'/run_strict_compare.py')
if($LASTEXITCODE -ne 0){throw 'The strict comparison did not finish. Diagnostic outputs are preserved.'}
& docker --context desktop-linux exec -u ubuntu -e PYTHONDONTWRITEBYTECODE=1 $Container $Python -B ($script+'/build_strict_report.py')
if($LASTEXITCODE -ne 0){throw 'Strict report generation failed; raw experiment artifacts are preserved.'}
