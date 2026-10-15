[CmdletBinding()]
param(
    [string]$BuildDirectory = "",
    [switch]$SkipExport,
    [switch]$SkipRolloutTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Stop-NormalHmdBuild {
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
$buildLogPath = Join-Path $resultsDirectory "normal_hmd_unity_build.log"
$buildStatusPath = Join-Path $resultsDirectory "normal_hmd_unity_build_status.json"
$rolloutReportPath = Join-Path $resultsDirectory "normal_hmd_unity_rollout_test.json"
$rolloutLogsRoot = Join-Path $resultsDirectory "normal_hmd_unity_rollout_logs"

if ([string]::IsNullOrWhiteSpace($BuildDirectory)) {
    $BuildDirectory = Join-Path $scriptRoot "UnityNormalHmdAgentDemoBuild"
}
$BuildDirectory = [System.IO.Path]::GetFullPath($BuildDirectory)
$expectedExecutable = Join-Path $BuildDirectory "NormalHmdAgentDemo.exe"

foreach ($requiredFile in @($unityEditor, $projectVersionFile)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        Stop-NormalHmdBuild "Required file is missing: $requiredFile"
    }
}
$projectVersion = Get-Content -LiteralPath $projectVersionFile -Raw
if ($projectVersion -notmatch "(?m)^m_EditorVersion:\s*6000\.2\.9f1\s*$") {
    Stop-NormalHmdBuild "Unity project is not pinned to 6000.2.9f1: $projectVersionFile"
}
New-Item -ItemType Directory -Path $resultsDirectory -Force | Out-Null

if (-not $SkipExport) {
    foreach ($requiredFile in @($pythonExe, $exporter)) {
        if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
            Stop-NormalHmdBuild "Required export dependency is missing: $requiredFile"
        }
    }
    Write-Host "[1/3] Exporting the trained three-branch Normal HMD Agent v4 checkpoint..."
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
        Stop-NormalHmdBuild "Normal HMD model export failed with exit code $LASTEXITCODE." $LASTEXITCODE
    }
    try {
        $exportReport = Get-Content -LiteralPath $exportReportPath -Raw | ConvertFrom-Json
    }
    catch {
        Stop-NormalHmdBuild "Model export report is missing or invalid: $exportReportPath"
    }
    if ($exportReport.status -ne "success" -or $exportReport.source_training_is_synthetic) {
        Stop-NormalHmdBuild "Model export is not a successful real-data export: $exportReportPath"
    }
}
foreach ($requiredFile in @($assetPath, $integrityPath, $rootCalibrationPath)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        Stop-NormalHmdBuild "Generated Unity model artifact is missing: $requiredFile"
    }
}

try {
    $modelAsset = Get-Content -LiteralPath $assetPath -Raw | ConvertFrom-Json
}
catch {
    Stop-NormalHmdBuild "Generated Unity model asset is invalid JSON: $assetPath"
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
    Stop-NormalHmdBuild "Unity model asset is not the frozen SmoothWalk profile (.05/.99/.10, scale 1.0): $assetPath"
}

Write-Host "[2/3] Building the independent Windows Player..."
foreach ($staleFile in @($buildStatusPath, $rolloutReportPath)) {
    if (Test-Path -LiteralPath $staleFile -PathType Leaf) {
        Remove-Item -LiteralPath $staleFile -Force
    }
}
$previousBuildDirectory = [Environment]::GetEnvironmentVariable(
    "UNITY_NORMAL_HMD_BUILD_DIR", [EnvironmentVariableTarget]::Process)
$unityExitCode = 1
try {
    [Environment]::SetEnvironmentVariable(
        "UNITY_NORMAL_HMD_BUILD_DIR", $BuildDirectory, [EnvironmentVariableTarget]::Process)
    $unityArguments = @(
        "-batchmode", "-quit",
        "-projectPath", $unityProject,
        "-executeMethod", "NormalHmdAgentDemoEditor.NormalHmdDemoBuilder.BuildWindows",
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
        "UNITY_NORMAL_HMD_BUILD_DIR", $previousBuildDirectory, [EnvironmentVariableTarget]::Process)
}
if ($unityExitCode -ne 0) {
    Stop-NormalHmdBuild "Unity build failed with exit code $unityExitCode. See $buildLogPath" $unityExitCode
}
if (-not (Test-Path -LiteralPath $buildStatusPath -PathType Leaf)) {
    Stop-NormalHmdBuild "Unity exited without a build status report: $buildStatusPath"
}
$buildStatus = Get-Content -LiteralPath $buildStatusPath -Raw | ConvertFrom-Json
if ($buildStatus.status -ne "success" -or $buildStatus.unityVersion -ne "6000.2.9f1") {
    Stop-NormalHmdBuild "Unity build status was not successful: $buildStatusPath"
}
if (-not (Test-Path -LiteralPath $expectedExecutable -PathType Leaf)) {
    Stop-NormalHmdBuild "Unity reported success but Player is missing: $expectedExecutable"
}

if (-not $SkipRolloutTest) {
    Write-Host "[3/3] Running two independent 60-second closed-loop acceptance rollouts..."
    $rolloutSeeds = @(20261011, 20261012)
    $individualRollouts = @()
    foreach ($rolloutSeed in $rolloutSeeds) {
        $individualReportPath = Join-Path $resultsDirectory "normal_hmd_unity_rollout_seed_$rolloutSeed.json"
        $individualLogPath = Join-Path $resultsDirectory "normal_hmd_unity_rollout_seed_$rolloutSeed.log"
        foreach ($staleFile in @($individualReportPath, $individualLogPath)) {
            if (Test-Path -LiteralPath $staleFile -PathType Leaf) {
                Remove-Item -LiteralPath $staleFile -Force
            }
        }
        Write-Host "      seed $rolloutSeed"
        $playerArguments = @(
            "-batchmode", "-nographics", "-logFile", $individualLogPath,
            "--rollout-test", "--seed=$rolloutSeed",
            "--rollout-test-output=$individualReportPath",
            "--log-root=$rolloutLogsRoot"
        )
        $player = Start-Process `
            -FilePath $expectedExecutable `
            -ArgumentList $playerArguments `
            -WindowStyle Hidden `
            -Wait `
            -PassThru
        if ($player.ExitCode -ne 0) {
            Stop-NormalHmdBuild "60-second rollout seed $rolloutSeed failed with exit code $($player.ExitCode). See $individualLogPath" $player.ExitCode
        }
        if (-not (Test-Path -LiteralPath $individualReportPath -PathType Leaf)) {
            Stop-NormalHmdBuild "Player exited without rollout report: $individualReportPath"
        }
        $rollout = Get-Content -LiteralPath $individualReportPath -Raw | ConvertFrom-Json
        if (
            $rollout.status -ne "passed" -or
            -not $rollout.noReset -or
            $rollout.generatedSeconds -lt 60.0 -or
            -not $rollout.sameSeedDeterministic -or
            -not $rollout.differentSeedVaries
        ) {
            Stop-NormalHmdBuild "60-second rollout acceptance did not pass: $individualReportPath"
        }
        if (-not $SkipExport -and -not $rollout.productionGateApplied) {
            Stop-NormalHmdBuild "A production build was tested with smoke-only acceptance gates: $individualReportPath"
        }
        $individualRollouts += $rollout
    }
    if ($individualRollouts[0].sameSeedSignature -eq $individualRollouts[1].sameSeedSignature) {
        Stop-NormalHmdBuild "The two independent rollout seeds produced the same sequence signature."
    }
    $aggregate = [ordered]@{
        schema_version = "1.0.0"
        status = "passed"
        production_gate_required = (-not $SkipExport)
        rollout_count = $individualRollouts.Count
        seeds = $rolloutSeeds
        each_rollout_passed = $true
        each_same_seed_reproducible = $true
        each_different_seed_varies = $true
        cross_seed_signatures_differ = $true
        source_model_sha256 = $individualRollouts[0].sourceModelSha256
        rollouts = $individualRollouts
    }
    $aggregateJson = $aggregate | ConvertTo-Json -Depth 10
    [System.IO.File]::WriteAllText(
        $rolloutReportPath,
        $aggregateJson + [Environment]::NewLine,
        [System.Text.UTF8Encoding]::new($false)
    )
}
else {
    Write-Host "[3/3] Rollout test skipped by caller."
}

Write-Host "Normal HMD Agent Demo ready: $expectedExecutable"
Write-Host "Build status: $buildStatusPath"
if (-not $SkipRolloutTest) { Write-Host "Rollout acceptance: $rolloutReportPath" }
exit 0
