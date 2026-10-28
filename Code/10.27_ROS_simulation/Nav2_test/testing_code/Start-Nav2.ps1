[CmdletBinding()]
param(
    [ValidateSet('test','sim')][string]$Action='test',
    [switch]$Gui,
    [switch]$Headless
)
$ErrorActionPreference='Stop'
& C:\Ubuntu\Jazzy.ps1 -Action start -Mode simulation
if($LASTEXITCODE -ne 0){throw 'Could not start the simulation container.'}
$repository=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..\..')).Path
$containerCode='/workspace'+$PSScriptRoot.Substring($repository.Length).Replace('\','/')
if($Action -eq 'test'){
    $testArguments=@()
    if($Gui){$testArguments+='--gui'}
    & wsl.exe -d Ubuntu-24.04 -u root --exec docker exec -u ubuntu bunker-jazzy bash ($containerCode+'/run_test.sh') @testArguments
    $testExitCode=$LASTEXITCODE
    if($testExitCode -ne 0){exit $testExitCode}
    & C:\Ubuntu\ROS\reporting-venv\Scripts\python.exe (Join-Path $PSScriptRoot 'tools\report.py')
}else{
    $guiArgument='gui:=true'
    if($Headless){$guiArgument='gui:=false'}
    & wsl.exe -d Ubuntu-24.04 -u root --exec docker exec -it -u ubuntu -e DISPLAY=:0 -e LIBGL_ALWAYS_SOFTWARE=1 bunker-jazzy bash ($containerCode+'/run_nav2.sh') $guiArgument
}
exit $LASTEXITCODE
