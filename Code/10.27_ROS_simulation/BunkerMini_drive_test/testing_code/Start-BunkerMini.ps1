[CmdletBinding()]
param(
    [ValidateSet('sim','test')][string]$Action='sim',
    [switch]$Headless
)
$ErrorActionPreference='Stop'
& C:\Ubuntu\Jazzy.ps1 -Action start -Mode simulation
if($LASTEXITCODE -ne 0){throw 'Could not start the simulation container.'}
$repository=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..\..')).Path
$containerCode='/workspace'+$PSScriptRoot.Substring($repository.Length).Replace('\','/')
if($Action -eq 'test') {
    & wsl.exe -d Ubuntu-24.04 -u root --exec docker exec -u ubuntu bunker-jazzy bash ($containerCode+'/run_test.sh')
    $testExitCode=$LASTEXITCODE
    if($testExitCode -ne 0){exit $testExitCode}
    $reportPython='C:\Ubuntu\ROS\reporting-venv\Scripts\python.exe'
    if(Test-Path -LiteralPath $reportPython){
        & $reportPython (Join-Path $PSScriptRoot 'tools\report.py') --stage BunkerMini_drive_test
    }
} else {
    $guiArgument='gui:=true'
    if($Headless){$guiArgument='gui:=false'}
    & wsl.exe -d Ubuntu-24.04 -u root --exec docker exec -it -u ubuntu -e DISPLAY=:0 -e LIBGL_ALWAYS_SOFTWARE=1 bunker-jazzy bash ($containerCode+'/run_sim.sh') $guiArgument
}
exit $LASTEXITCODE
