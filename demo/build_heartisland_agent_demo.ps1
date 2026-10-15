[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SourceProject,
    [string]$UnityEditor = "C:\Program Files\Unity\Hub\Editor\6000.2.9f1\Editor\Unity.exe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Stop-HeartislandBuild {
    param([Parameter(Mandatory = $true)][string]$Message, [int]$ExitCode = 1)
    Write-Error -Message $Message -ErrorAction Continue
    exit $ExitCode
}

function Copy-TreeMerge {
    param([Parameter(Mandatory = $true)][string]$Source, [Parameter(Mandatory = $true)][string]$Destination)
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    & robocopy.exe $Source $Destination /E /R:2 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -gt 7) {
        Stop-HeartislandBuild "Copy failed from $Source to $Destination (robocopy exit $LASTEXITCODE)." $LASTEXITCODE
    }
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptRoot ".."))
$sourceProjectRoot = [System.IO.Path]::GetFullPath($SourceProject)
$stagingProject = Join-Path $scriptRoot "UnityExternalSceneAgentProjects\HeartislandAgent"
$outputDirectory = Join-Path $scriptRoot "ExternalSceneAgentBuilds\HeartislandAgent"
$expectedExecutable = Join-Path $outputDirectory "HeartislandAgent.exe"
$adapterDirectory = Join-Path $scriptRoot "external_scene_adapter"
$normalScripts = Join-Path $scriptRoot "UnityNormalHmdAgentDemo\Assets\Scripts"
$normalResources = Join-Path $scriptRoot "UnityNormalHmdAgentDemo\Assets\Resources"
$unityEditor = [System.IO.Path]::GetFullPath($UnityEditor)
$resultsDirectory = Join-Path $repositoryRoot "results"
$buildLogPath = Join-Path $resultsDirectory "heartisland_agent_build.log"
$buildStatusPath = Join-Path $resultsDirectory "heartisland_agent_build_status.json"
$acceptancePath = Join-Path $resultsDirectory "heartisland_agent_acceptance.json"
$playerLogPath = Join-Path $resultsDirectory "heartisland_agent_test.log"
$testLogsRoot = Join-Path $resultsDirectory "heartisland_agent_test_logs"

foreach ($required in @(
    $unityEditor,
    (Join-Path $sourceProjectRoot "ProjectSettings\ProjectVersion.txt"),
    (Join-Path $sourceProjectRoot "Assets\Scenes\ResonanceValley.unity"),
    (Join-Path $adapterDirectory "HeartIslandAgentBridge.cs"),
    (Join-Path $adapterDirectory "HeartIslandAgentBuilder.cs"),
    (Join-Path $normalResources "NormalHmdAgentModel.json")
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        Stop-HeartislandBuild "Required file is missing: $required"
    }
}
$version = Get-Content -LiteralPath (Join-Path $sourceProjectRoot "ProjectSettings\ProjectVersion.txt") -Raw
if ($version -notmatch "(?m)^m_EditorVersion:\s*6000\.2\.9f1\s*$") {
    Stop-HeartislandBuild "Heartisland source project is not Unity 6000.2.9f1."
}

Write-Host "[1/4] Creating a non-destructive staging copy of HeartislandUnity..."
Copy-TreeMerge (Join-Path $sourceProjectRoot "Assets") (Join-Path $stagingProject "Assets")
Copy-TreeMerge (Join-Path $sourceProjectRoot "Packages") (Join-Path $stagingProject "Packages")
Copy-TreeMerge (Join-Path $sourceProjectRoot "ProjectSettings") (Join-Path $stagingProject "ProjectSettings")

# NormalHmdAgentController uses the legacy key API for optional demo shortcuts,
# while Heartisland uses the new Input System.  The staging copy enables both;
# the original project's setting is never modified.
$stagedProjectSettings = Join-Path $stagingProject "ProjectSettings\ProjectSettings.asset"
$settingsText = Get-Content -LiteralPath $stagedProjectSettings -Raw
$settingsText = $settingsText -replace "(?m)^(\s*activeInputHandler:\s*)1\s*$", '${1}2'
Set-Content -LiteralPath $stagedProjectSettings -Value $settingsText -Encoding utf8NoBOM

$runtimeDestination = Join-Path $stagingProject "Assets\NormalHmdAgentRuntime"
$editorDestination = Join-Path $runtimeDestination "Editor"
$resourceDestination = Join-Path $stagingProject "Assets\Resources"
foreach ($directory in @($runtimeDestination, $editorDestination, $resourceDestination)) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}
$runtimeFiles = @(
    "GeneratedNormalHmdIntegrity.cs",
    "NormalHmdAgentController.cs",
    "NormalHmdDemoBootstrap.cs",
    "NormalHmdLstmRuntime.cs",
    "NormalHmdModelData.cs",
    "NormalHmdVerticalEventController.cs",
    "SceneAwareContracts.cs",
    "SeededRootInnovation.cs"
)
foreach ($name in $runtimeFiles) {
    $source = Join-Path $normalScripts $name
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        Stop-HeartislandBuild "Normal HMD runtime source is missing: $source"
    }
    Copy-Item -LiteralPath $source -Destination (Join-Path $runtimeDestination $name) -Force
}
Copy-Item -LiteralPath (Join-Path $adapterDirectory "HeartIslandAgentBridge.cs") `
    -Destination (Join-Path $runtimeDestination "HeartIslandAgentBridge.cs") -Force
Copy-Item -LiteralPath (Join-Path $adapterDirectory "HeartIslandAgentBuilder.cs") `
    -Destination (Join-Path $editorDestination "HeartIslandAgentBuilder.cs") -Force
Copy-Item -LiteralPath (Join-Path $normalResources "NormalHmdAgentModel.json") `
    -Destination (Join-Path $resourceDestination "NormalHmdAgentModel.json") -Force

foreach ($directory in @($outputDirectory, $resultsDirectory, $testLogsRoot)) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}
foreach ($stale in @($buildLogPath, $buildStatusPath, $acceptancePath, $playerLogPath)) {
    if (Test-Path -LiteralPath $stale -PathType Leaf) { Remove-Item -LiteralPath $stale -Force }
}

Write-Host "[2/4] Importing and compiling the isolated staging project..."
$previousBuildDir = [Environment]::GetEnvironmentVariable("HEARTISLAND_AGENT_BUILD_DIR", [EnvironmentVariableTarget]::Process)
$previousStatus = [Environment]::GetEnvironmentVariable("HEARTISLAND_AGENT_BUILD_STATUS", [EnvironmentVariableTarget]::Process)
try {
    [Environment]::SetEnvironmentVariable("HEARTISLAND_AGENT_BUILD_DIR", $outputDirectory, [EnvironmentVariableTarget]::Process)
    [Environment]::SetEnvironmentVariable("HEARTISLAND_AGENT_BUILD_STATUS", $buildStatusPath, [EnvironmentVariableTarget]::Process)
    $unityArguments = @(
        "-batchmode", "-quit", "-projectPath", $stagingProject,
        "-executeMethod", "ExternalSceneAgent.Editor.HeartIslandAgentBuilder.BuildWindows",
        "-logFile", $buildLogPath
    )
    $unity = Start-Process -FilePath $unityEditor -ArgumentList $unityArguments -WindowStyle Hidden -Wait -PassThru
}
finally {
    [Environment]::SetEnvironmentVariable("HEARTISLAND_AGENT_BUILD_DIR", $previousBuildDir, [EnvironmentVariableTarget]::Process)
    [Environment]::SetEnvironmentVariable("HEARTISLAND_AGENT_BUILD_STATUS", $previousStatus, [EnvironmentVariableTarget]::Process)
}
if ($unity.ExitCode -ne 0) {
    Stop-HeartislandBuild "Unity build failed with exit code $($unity.ExitCode). See $buildLogPath" $unity.ExitCode
}
if (-not (Test-Path -LiteralPath $expectedExecutable -PathType Leaf)) {
    Stop-HeartislandBuild "Unity did not create the expected Player: $expectedExecutable"
}
$buildStatus = Get-Content -LiteralPath $buildStatusPath -Raw | ConvertFrom-Json
if ($buildStatus.status -ne "success" -or $buildStatus.unityVersion -ne "6000.2.9f1" -or
    [int]$buildStatus.errorCount -ne 0) {
    Stop-HeartislandBuild "Heartisland build status failed quality gates: $buildStatusPath"
}

Write-Host "[3/4] Running a separate accelerated 60-second Agent experience..."
$seed = 2026101305
$playerArguments = @(
    "-batchmode", "-nographics", "-logFile", $playerLogPath,
    "--agent-experience", "--external-agent-test", "--seed=$seed",
    "--external-agent-test-output=$acceptancePath", "--scene-log-root=$testLogsRoot"
)
$player = Start-Process -FilePath $expectedExecutable -ArgumentList $playerArguments -WindowStyle Hidden -PassThru
if (-not $player.WaitForExit(120000)) {
    Stop-Process -Id $player.Id -Force
    Stop-HeartislandBuild "Heartisland Agent test timed out after 120 seconds. See $playerLogPath"
}
if ($player.ExitCode -ne 0) {
    Stop-HeartislandBuild "Heartisland Agent test failed with exit code $($player.ExitCode). See $playerLogPath" $player.ExitCode
}
if (-not (Test-Path -LiteralPath $acceptancePath -PathType Leaf)) {
    Stop-HeartislandBuild "Player exited without acceptance report: $acceptancePath"
}
$acceptance = Get-Content -LiteralPath $acceptancePath -Raw | ConvertFrom-Json
if ($acceptance.status -ne "passed" -or [double]$acceptance.generatedSeconds -lt 60 -or
    [int]$acceptance.telemetryRowCount -lt 2200 -or
    [int]$acceptance.interactionsSucceeded -ne 3 -or
    [int]$acceptance.objectivesCompleted -ne 4 -or
    [double]$acceptance.coverage -lt 1) {
    Stop-HeartislandBuild "Heartisland acceptance report failed quality gates: $acceptancePath"
}

Write-Host "[4/4] Verifying logs contain no runtime exceptions..."
$errors = rg -n "NullReference|Unhandled Exception|error CS|Assertion failed" $buildLogPath $playerLogPath
if ($LASTEXITCODE -eq 0) {
    $errors
    Stop-HeartislandBuild "Build or Player log contains an exception/error marker."
}
if ($LASTEXITCODE -ne 1) {
    Stop-HeartislandBuild "Failed to scan Heartisland logs with rg." $LASTEXITCODE
}

Write-Host "Heartisland Agent Demo ready: $expectedExecutable"
Write-Host "Acceptance: $acceptancePath"
