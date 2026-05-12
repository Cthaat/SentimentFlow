[CmdletBinding()]
param(
    [switch]$Detached,
    [switch]$Pull
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

function Invoke-Docker {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & docker @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "docker $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

$buildArgs = @("compose", "build", "--no-cache")
if ($Pull) {
    $buildArgs += "--pull"
}

Invoke-Docker -Arguments $buildArgs

$upArgs = @("compose", "up", "--force-recreate", "--remove-orphans")
if ($Detached) {
    $upArgs += "-d"
}

Invoke-Docker -Arguments $upArgs
