[CmdletBinding()]
param(
    [ValidateSet('validate','smoke','full','real')][string]$Mode='full'
)
$ErrorActionPreference='Stop'
$docker='C:\Program Files\Docker\Docker\resources\bin\docker.exe'
if(-not (Test-Path -LiteralPath $docker)){throw 'Docker CLI not found in the configured Program Files installation.'}
$experiment='/workspace/Code/ROS_Nav2_policy_compare'
$stamp=[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$imageId=(& $docker --context desktop-linux inspect --format '{{.Image}}' bunker-jazzy).Trim()
if($LASTEXITCODE -ne 0){throw 'The existing bunker-jazzy container must be available.'}
function Invoke-Experiment([string[]]$Arguments){
    & $docker --context desktop-linux exec -e ('DOCKER_IMAGE_ID='+$imageId) -u ubuntu bunker-jazzy bash ($experiment+'/testing_code/run.sh') @Arguments
    if($LASTEXITCODE -ne 0){throw 'Experiment command failed. Preserve and inspect the saved diagnostics before retrying.'}
}
Invoke-Experiment @($experiment+'/testing_code/test_math.py')
Invoke-Experiment @($experiment+'/testing_code/test_training.py')
Invoke-Experiment @($experiment+'/testing_code/environment.py')
if($Mode -eq 'validate'){return}
$bundle=$experiment+'/result/'+$Mode+'_input_'+$stamp
$comparison=$experiment+'/result/'+$Mode+'_comparison_'+$stamp
if($Mode -eq 'real'){
    Invoke-Experiment @('-m','nav2_policy.prepare','--config',($experiment+'/config/real.json'),'--output',$bundle)
}else{
    $fixtureArgs=@('-m','nav2_policy.fixtures','--output',$bundle)
    if($Mode -eq 'full'){$fixtureArgs+='--full-budget'}
    Invoke-Experiment $fixtureArgs
    $bundle+='/bundle'
}
Invoke-Experiment @('-m','nav2_policy.runner','--bundle',$bundle,'--output',$comparison,'--policies','mi','emse','wivr')
$root=Split-Path -Parent $PSScriptRoot
$relative=$comparison.Substring($experiment.Length+1)
$reportPython='C:\Ubuntu\ROS\reporting-venv\Scripts\python.exe'
if(-not (Test-Path -LiteralPath $reportPython)){throw 'Saved runs passed. Install the documented reporting dependencies to regenerate Comparison.ipynb.'}
& $reportPython (Join-Path $PSScriptRoot 'build_report.py') --comparison (Join-Path $root $relative)
if($LASTEXITCODE -ne 0){throw 'The runs are saved, but notebook execution failed.'}
Write-Host ('Completed: '+(Join-Path $root $relative))
