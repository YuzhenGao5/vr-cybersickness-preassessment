[CmdletBinding()]
param(
    [string]$BuildDirectory = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Stop-Build {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message,
        [int]$ExitCode = 1
    )

    Write-Error -Message $Message -ErrorAction Continue
    exit $ExitCode
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptRoot ".."))
$pythonExe = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
$exporter = Join-Path $repositoryRoot "scripts\export_unity_random_agent.py"
$unityEditor = "C:\Program Files\Unity\Hub\Editor\6000.2.9f1\Editor\Unity.exe"
$unityProject = Join-Path $scriptRoot "UnityRandomAgentDemo"
$projectVersionFile = Join-Path $unityProject "ProjectSettings\ProjectVersion.txt"
$resultsDirectory = Join-Path $repositoryRoot "results"
$exportReportPath = Join-Path $resultsDirectory "unity_random_agent_export_report.json"
$buildLogPath = Join-Path $resultsDirectory "unity_random_agent_build.log"
$buildStatusPath = Join-Path $resultsDirectory "unity_random_agent_build_status.json"

if ([string]::IsNullOrWhiteSpace($BuildDirectory)) {
    $BuildDirectory = Join-Path $scriptRoot "UnityRandomAgentDemoBuild"
}
$BuildDirectory = [System.IO.Path]::GetFullPath($BuildDirectory)
$expectedExecutable = Join-Path $BuildDirectory "RandomHumanAgentDemo.exe"

foreach ($requiredFile in @($pythonExe, $exporter, $unityEditor, $projectVersionFile)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        Stop-Build "Required file is missing: $requiredFile"
    }
}

$projectVersion = Get-Content -LiteralPath $projectVersionFile -Raw
if ($projectVersion -notmatch "(?m)^m_EditorVersion:\s*6000\.2\.9f1\s*$") {
    Stop-Build "Unity project is not pinned to 6000.2.9f1: $projectVersionFile"
}

New-Item -ItemType Directory -Path $resultsDirectory -Force | Out-Null

Write-Host "[1/2] Exporting the trained EyeNavGS model for Unity..."
& $pythonExe $exporter
$exportExitCode = $LASTEXITCODE
if ($exportExitCode -ne 0) {
    Stop-Build "Model export failed with exit code $exportExitCode." $exportExitCode
}

try {
    $exportReport = Get-Content -LiteralPath $exportReportPath -Raw | ConvertFrom-Json
}
catch {
    Stop-Build "Model export report is missing or invalid: $exportReportPath"
}
if ($exportReport.status -ne "success") {
    Stop-Build "Model export report did not declare success: $exportReportPath"
}

Write-Host "[2/2] Building the Windows Player with Unity 6000.2.9f1..."
# Remove only the generated status marker so a stale successful report can never
# make a failed Unity invocation look healthy. Unity recreates it after the build.
if (Test-Path -LiteralPath $buildStatusPath -PathType Leaf) {
    Remove-Item -LiteralPath $buildStatusPath -Force
}
$previousBuildDirectory = [Environment]::GetEnvironmentVariable(
    "UNITY_RANDOM_AGENT_BUILD_DIR",
    [EnvironmentVariableTarget]::Process
)
$unityExitCode = 1
try {
    [Environment]::SetEnvironmentVariable(
        "UNITY_RANDOM_AGENT_BUILD_DIR",
        $BuildDirectory,
        [EnvironmentVariableTarget]::Process
    )
    $unityArguments = @(
        "-batchmode",
        "-quit",
        "-projectPath", $unityProject,
        "-executeMethod", "RandomAgentDemoEditor.DemoBuilder.BuildWindows",
        "-logFile", $buildLogPath
    )
    $unityProcess = Start-Process `
        -FilePath $unityEditor `
        -ArgumentList $unityArguments `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    $unityExitCode = $unityProcess.ExitCode
}
finally {
    [Environment]::SetEnvironmentVariable(
        "UNITY_RANDOM_AGENT_BUILD_DIR",
        $previousBuildDirectory,
        [EnvironmentVariableTarget]::Process
    )
}

if ($unityExitCode -ne 0) {
    Stop-Build "Unity batch build failed with exit code $unityExitCode. See: $buildLogPath" $unityExitCode
}

if (-not (Test-Path -LiteralPath $buildStatusPath -PathType Leaf)) {
    Stop-Build "Unity exited without writing a build status report: $buildStatusPath"
}

try {
    $buildStatus = Get-Content -LiteralPath $buildStatusPath -Raw | ConvertFrom-Json
}
catch {
    Stop-Build "Unity build status report is invalid: $buildStatusPath"
}

if ($buildStatus.status -ne "success") {
    Stop-Build "Unity build status is '$($buildStatus.status)'. See: $buildLogPath"
}
if ($buildStatus.unityVersion -ne "6000.2.9f1") {
    Stop-Build "Unexpected Unity version in build status: $($buildStatus.unityVersion)"
}
if (-not (Test-Path -LiteralPath $expectedExecutable -PathType Leaf)) {
    Stop-Build "Unity reported success, but the Player is missing: $expectedExecutable"
}

Write-Host "Build succeeded: $expectedExecutable"
Write-Host "Build log: $buildLogPath"
exit 0
