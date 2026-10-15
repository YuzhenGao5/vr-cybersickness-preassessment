[CmdletBinding()]
param(
    [string]$BuildDirectory = "",
    [switch]$SkipExport,
    [switch]$SkipTests
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Stop-SceneAwareBuild {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message,
        [int]$ExitCode = 1
    )
    Write-Error -Message $Message -ErrorAction Continue
    exit $ExitCode
}

function Get-ReportValue {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Report,
        [Parameter(Mandatory = $true)]
        [string[]]$Names,
        $DefaultValue = $null
    )
    foreach ($name in $Names) {
        $property = $Report.PSObject.Properties[$name]
        if ($null -ne $property) {
            return $property.Value
        }
    }
    return $DefaultValue
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptRoot ".."))
$pythonExe = Join-Path $repositoryRoot ".venv-normal-hmd\Scripts\python.exe"
$exporter = Join-Path $repositoryRoot "scripts\normal_hmd_v4_export.py"
$unityEditor = "C:\Program Files\Unity\Hub\Editor\6000.2.9f1\Editor\Unity.exe"
$unityProject = Join-Path $scriptRoot "UnityNormalHmdAgentDemo"
$projectVersionFile = Join-Path $unityProject "ProjectSettings\ProjectVersion.txt"
$assetPath = Join-Path $unityProject "Assets\Resources\NormalHmdAgentModel.json"
$integrityPath = Join-Path $unityProject "Assets\Scripts\GeneratedNormalHmdIntegrity.cs"
$resultsDirectory = Join-Path $repositoryRoot "results"
$exportReportPath = Join-Path $resultsDirectory "normal_hmd_v4_smooth_walk_unity_export_report.json"
$rootCalibrationPath = Join-Path $resultsDirectory "normal_hmd_v4_smooth_walk_noise_calibration.json"
$buildLogPath = Join-Path $resultsDirectory "scene_aware_agent_build.log"
$buildStatusPath = Join-Path $resultsDirectory "scene_aware_agent_build_status.json"
$acceptancePath = Join-Path $resultsDirectory "scene_aware_agent_acceptance.json"
$testLogsRoot = Join-Path $resultsDirectory "scene_aware_agent_test_logs"

if ([string]::IsNullOrWhiteSpace($BuildDirectory)) {
    $BuildDirectory = Join-Path $scriptRoot "UnitySceneAwareAgentDemoBuild"
}
$BuildDirectory = [System.IO.Path]::GetFullPath($BuildDirectory)
$expectedExecutable = Join-Path $BuildDirectory "SceneAwareAgentDemo.exe"

foreach ($requiredFile in @($unityEditor, $projectVersionFile)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        Stop-SceneAwareBuild "Required file is missing: $requiredFile"
    }
}
$projectVersion = Get-Content -LiteralPath $projectVersionFile -Raw
if ($projectVersion -notmatch "(?m)^m_EditorVersion:\s*6000\.2\.9f1\s*$") {
    Stop-SceneAwareBuild "Unity project is not pinned to 6000.2.9f1: $projectVersionFile"
}
New-Item -ItemType Directory -Path $resultsDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $testLogsRoot -Force | Out-Null

if (-not $SkipExport) {
    foreach ($requiredFile in @($pythonExe, $exporter)) {
        if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
            Stop-SceneAwareBuild "Required export dependency is missing: $requiredFile"
        }
    }
    Write-Host "[1/3] Exporting the trained Normal HMD Agent v4 model..."
    $exportArgs = @(
        "--root-innovation-std", "0.05",
        "--root-innovation-rho", "0.99",
        "--roll-limit-rad", "0.10",
        "--root-calibration-report", $rootCalibrationPath,
        "--reuse-root-calibration",
        "--report", $exportReportPath
    )
    & $pythonExe $exporter @exportArgs
    if ($LASTEXITCODE -ne 0) {
        Stop-SceneAwareBuild "Normal HMD model export failed with exit code $LASTEXITCODE." $LASTEXITCODE
    }
    try {
        $exportReport = Get-Content -LiteralPath $exportReportPath -Raw | ConvertFrom-Json
    }
    catch {
        Stop-SceneAwareBuild "Model export report is missing or invalid: $exportReportPath"
    }
    if ($exportReport.status -ne "success" -or $exportReport.source_training_is_synthetic) {
        Stop-SceneAwareBuild "Model export is not a successful real-data export: $exportReportPath"
    }
}

foreach ($requiredFile in @($assetPath, $integrityPath, $rootCalibrationPath)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        Stop-SceneAwareBuild "Generated Unity model artifact is missing: $requiredFile"
    }
}

try {
    $modelAsset = Get-Content -LiteralPath $assetPath -Raw | ConvertFrom-Json
}
catch {
    Stop-SceneAwareBuild "Generated Unity model asset is invalid JSON: $assetPath"
}
$rhoValues = @($modelAsset.rootInnovationRho)
$expectedCalibrationSha = (Get-FileHash -LiteralPath $rootCalibrationPath -Algorithm SHA256).Hash.ToLowerInvariant()
$profileMismatch =
    [Math]::Abs([double]$modelAsset.rootInnovationNormalizedStd - 0.05) -gt 1e-6 -or
    $rhoValues.Count -ne 3 -or
    @($rhoValues | Where-Object { [Math]::Abs([double]$_ - 0.99) -gt 1e-6 }).Count -ne 0 -or
    [Math]::Abs([double]$modelAsset.deploymentRollClampRad - 0.10) -gt 1e-6 -or
    [Math]::Abs([double]$modelAsset.defaultRootNoiseScale - 1.0) -gt 1e-6 -or
    [string]$modelAsset.rootNoiseCalibrationSplit -ne "val" -or
    [string]$modelAsset.rootNoiseCalibrationReportSha256 -ne $expectedCalibrationSha
if ($profileMismatch) {
    Stop-SceneAwareBuild "Unity model asset is not the frozen SmoothWalk profile (.05/.99/.10, scale 1.0): $assetPath"
}

Write-Host "[2/3] Building the independent scene-aware Windows Player..."
foreach ($staleFile in @($buildStatusPath, $acceptancePath, $buildLogPath)) {
    if (Test-Path -LiteralPath $staleFile -PathType Leaf) {
        Remove-Item -LiteralPath $staleFile -Force
    }
}
$previousBuildDirectory = [Environment]::GetEnvironmentVariable(
    "UNITY_SCENE_AWARE_BUILD_DIR", [EnvironmentVariableTarget]::Process)
$unityExitCode = 1
try {
    [Environment]::SetEnvironmentVariable(
        "UNITY_SCENE_AWARE_BUILD_DIR", $BuildDirectory, [EnvironmentVariableTarget]::Process)
    $unityArguments = @(
        "-batchmode", "-quit",
        "-projectPath", $unityProject,
        "-executeMethod", "NormalHmdAgentDemoEditor.SceneAwareDemoBuilder.BuildWindows",
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
        "UNITY_SCENE_AWARE_BUILD_DIR", $previousBuildDirectory, [EnvironmentVariableTarget]::Process)
}
if ($unityExitCode -ne 0) {
    Stop-SceneAwareBuild "Unity build failed with exit code $unityExitCode. See $buildLogPath" $unityExitCode
}
if (-not (Test-Path -LiteralPath $buildStatusPath -PathType Leaf)) {
    Stop-SceneAwareBuild "Unity exited without a build status report: $buildStatusPath"
}
$buildStatus = Get-Content -LiteralPath $buildStatusPath -Raw | ConvertFrom-Json
if ($buildStatus.status -ne "success" -or $buildStatus.unityVersion -ne "6000.2.9f1") {
    Stop-SceneAwareBuild "Unity build status was not successful: $buildStatusPath"
}
if (-not (Test-Path -LiteralPath $expectedExecutable -PathType Leaf)) {
    Stop-SceneAwareBuild "Unity reported success but Player is missing: $expectedExecutable"
}

if (-not $SkipTests) {
    Write-Host "[3/3] Running four 60-second scene-aware acceptance scenarios..."
    $testCases = @(
        [pscustomobject]@{ Scenario = 0; Seed = 2026101301 },
        [pscustomobject]@{ Scenario = 1; Seed = 2026101302 },
        [pscustomobject]@{ Scenario = 2; Seed = 2026101303 },
        [pscustomobject]@{ Scenario = 3; Seed = 2026101304 }
    )
    $scenarioReports = @()

    foreach ($testCase in $testCases) {
        $scenario = [int]$testCase.Scenario
        $seed = [int64]$testCase.Seed
        $scenarioReportPath = Join-Path $resultsDirectory "scene_aware_agent_test_scenario_$scenario.json"
        $scenarioPlayerLog = Join-Path $resultsDirectory "scene_aware_agent_test_scenario_$scenario.log"
        $scenarioLogRoot = Join-Path $testLogsRoot "scenario_$scenario"

        foreach ($staleFile in @($scenarioReportPath, $scenarioPlayerLog)) {
            if (Test-Path -LiteralPath $staleFile -PathType Leaf) {
                Remove-Item -LiteralPath $staleFile -Force
            }
        }
        New-Item -ItemType Directory -Path $scenarioLogRoot -Force | Out-Null

        Write-Host "      scenario $scenario, seed $seed"
        $playerArguments = @(
            "-batchmode", "-nographics", "-logFile", $scenarioPlayerLog,
            "--scene-aware-test", "--scenario=$scenario", "--seed=$seed",
            "--scene-aware-test-output=$scenarioReportPath",
            "--scene-log-root=$scenarioLogRoot"
        )
        $player = Start-Process `
            -FilePath $expectedExecutable `
            -ArgumentList $playerArguments `
            -WindowStyle Hidden `
            -Wait `
            -PassThru
        if ($player.ExitCode -ne 0) {
            Stop-SceneAwareBuild "Scene-aware scenario $scenario failed with exit code $($player.ExitCode). See $scenarioPlayerLog" $player.ExitCode
        }
        if (-not (Test-Path -LiteralPath $scenarioReportPath -PathType Leaf)) {
            Stop-SceneAwareBuild "Player exited without scenario report: $scenarioReportPath"
        }

        try {
            $scenarioReport = Get-Content -LiteralPath $scenarioReportPath -Raw | ConvertFrom-Json
        }
        catch {
            Stop-SceneAwareBuild "Scenario report is invalid JSON: $scenarioReportPath"
        }

        $status = [string](Get-ReportValue $scenarioReport @("status"))
        $generatedSeconds = [double](Get-ReportValue $scenarioReport @("generatedSeconds", "generated_seconds") -1)
        $telemetryRows = [int64](Get-ReportValue $scenarioReport @("telemetryRowCount", "telemetryRows", "telemetry_rows") 0)
        $telemetryPath = [string](Get-ReportValue $scenarioReport @("telemetryPath", "telemetryCsvPath", "telemetry_path") "")
        $expectedInteractionsRaw = Get-ReportValue $scenarioReport @(
            "interactionsExpected", "interactionExpected", "interactions_expected") 0
        $succeededInteractions = [int64](Get-ReportValue $scenarioReport @(
            "interactionsSucceeded", "interactionSucceeded", "interactions_succeeded") 0)

        if ($status -ne "passed") {
            Stop-SceneAwareBuild "Scene-aware scenario did not pass: $scenarioReportPath"
        }
        if ($generatedSeconds -lt 60.0) {
            Stop-SceneAwareBuild "Scene-aware scenario generated less than 60 seconds: $scenarioReportPath"
        }
        if ($telemetryRows -le 0 -or [string]::IsNullOrWhiteSpace($telemetryPath)) {
            Stop-SceneAwareBuild "Scene-aware scenario did not report telemetry rows and a telemetry path: $scenarioReportPath"
        }
        if (-not [System.IO.Path]::IsPathRooted($telemetryPath)) {
            $telemetryPath = [System.IO.Path]::GetFullPath((Join-Path $scenarioLogRoot $telemetryPath))
        }
        if (-not (Test-Path -LiteralPath $telemetryPath -PathType Leaf)) {
            Stop-SceneAwareBuild "Scene-aware scenario telemetry file is missing: $telemetryPath"
        }

        $expectedInteractions = 0L
        if ($expectedInteractionsRaw -is [bool]) {
            if ([bool]$expectedInteractionsRaw) { $expectedInteractions = 1L }
        }
        elseif ($null -ne $expectedInteractionsRaw) {
            $expectedInteractions = [int64]$expectedInteractionsRaw
        }
        if ($expectedInteractions -gt 0 -and $succeededInteractions -lt $expectedInteractions) {
            Stop-SceneAwareBuild "Scene-aware scenario expected $expectedInteractions interaction(s), but only $succeededInteractions succeeded: $scenarioReportPath"
        }

        $scenarioReports += $scenarioReport
    }

    $aggregate = [ordered]@{
        schemaVersion = "1.0.0"
        status = "passed"
        generatedAtUtc = [DateTime]::UtcNow.ToString("O")
        buildStatusPath = $buildStatusPath
        executable = $expectedExecutable
        scenarioCount = $scenarioReports.Count
        allScenariosPassed = $true
        eachScenarioGeneratedAtLeast60Seconds = $true
        eachScenarioProducedTelemetry = $true
        allExpectedInteractionsSucceeded = $true
        scenarios = $scenarioReports
    }
    $aggregateJson = $aggregate | ConvertTo-Json -Depth 20
    [System.IO.File]::WriteAllText(
        $acceptancePath,
        $aggregateJson + [Environment]::NewLine,
        [System.Text.UTF8Encoding]::new($false)
    )
}
else {
    Write-Host "[3/3] Scene-aware acceptance tests skipped by caller."
}

Write-Host "Scene-aware Normal HMD Agent Demo ready: $expectedExecutable"
Write-Host "Build status: $buildStatusPath"
if (-not $SkipTests) { Write-Host "Acceptance report: $acceptancePath" }
exit 0
