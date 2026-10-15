[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SourceProject,
    [Parameter(Mandatory = $true)]
    [string]$CompiledContract,
    [string]$ScenePath = "",
    [string]$ReviewedBindings = "",
    [string[]]$DisableBehaviourType = @(),
    [string]$BuildDirectory = "",
    [string]$StagingDirectory = "",
    [string]$UnityEditor = "C:\Program Files\Unity\Hub\Editor\6000.2.9f1\Editor\Unity.exe",
    [int]$Seed = 2026101401,
    [ValidateSet("rule", "shadow", "model")]
    [string]$DecisionBrainMode = "rule",
    [string]$DecisionBrainModel = "",
    [switch]$DecisionBrainResearchOverride
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Stop-Build([string]$Message, [int]$Code = 1) {
    Write-Error -Message $Message -ErrorAction Continue
    exit $Code
}

function Copy-Tree([string]$Source, [string]$Destination) {
    if (-not (Test-Path -LiteralPath $Source -PathType Container)) {
        Stop-Build "Required source directory is missing: $Source"
    }
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    & robocopy $Source $Destination /MIR /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) {
        Stop-Build "robocopy failed ($LASTEXITCODE): $Source -> $Destination" $LASTEXITCODE
    }
}

function Install-CompatibleRuntimeScript(
    [string]$CanonicalPath,
    [string]$DestinationDirectory,
    [string]$SearchRoot,
    [string]$ComponentLabel
) {
    if (-not (Test-Path -LiteralPath $CanonicalPath -PathType Leaf)) {
        Stop-Build "Required $ComponentLabel script is missing: $CanonicalPath"
    }

    $fileName = [System.IO.Path]::GetFileName($CanonicalPath)
    $canonicalHash = (Get-FileHash -LiteralPath $CanonicalPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $matches = @(
        Get-ChildItem -LiteralPath $SearchRoot -Filter $fileName -File -Recurse -ErrorAction Stop |
            Sort-Object -Property FullName
    )

    if ($matches.Count -gt 1) {
        $relativeMatches = @($matches | ForEach-Object {
            $_.FullName.Substring($SearchRoot.Length).TrimStart('\', '/')
        })
        Stop-Build (
            "The staged project already contains multiple copies of $fileName. " +
            "Unity would compile duplicate runtime types. Keep exactly one copy before building. Found: " +
            ($relativeMatches -join "; ")
        )
    }

    if ($matches.Count -eq 1) {
        $existing = $matches[0]
        $existingHash = (Get-FileHash -LiteralPath $existing.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        $relativeExisting = $existing.FullName.Substring($SearchRoot.Length).TrimStart('\', '/')
        if ($existingHash -ne $canonicalHash) {
            Stop-Build (
                "$ComponentLabel script version conflict for $fileName. " +
                "The source project already contains '$relativeExisting' (sha256=$existingHash), " +
                "but the Agent runtime requires sha256=$canonicalHash. " +
                "The source file was not overwritten; reconcile the versions explicitly."
            )
        }

        Write-Host "      [reuse] $relativeExisting (matching $ComponentLabel script; original .meta/GUID preserved)"
        return
    }

    New-Item -ItemType Directory -Path $DestinationDirectory -Force | Out-Null
    $destination = Join-Path $DestinationDirectory $fileName
    Copy-Item -LiteralPath $CanonicalPath -Destination $destination -Force
    $relativeDestination = $destination.Substring($SearchRoot.Length).TrimStart('\', '/')
    Write-Host "      [inject] $relativeDestination ($ComponentLabel)"
}

$demoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $demoRoot ".."))
$sourceProjectPath = [System.IO.Path]::GetFullPath($SourceProject)
$compiledContractPath = [System.IO.Path]::GetFullPath($CompiledContract)
$unityEditor = [System.IO.Path]::GetFullPath($UnityEditor)
$pythonExe = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
$materializer = Join-Path $repositoryRoot "scripts\materialize_generic_runtime_contract.py"
$renderValidator = Join-Path $repositoryRoot "scripts\validate_render_capture.py"
$genericSource = Join-Path $demoRoot "generic_scene_agent"
$normalSource = Join-Path $demoRoot "UnityNormalHmdAgentDemo\Assets\Scripts"
$normalModel = Join-Path $demoRoot "UnityNormalHmdAgentDemo\Assets\Resources\NormalHmdAgentModel.json"
$defaultDecisionBrainModel = Join-Path $repositoryRoot "models\unity_decision_brain_v0_unity.json"
$fallbackPackages = Join-Path $demoRoot "UnityRandomAgentDemo\Packages"
$fallbackProjectSettings = Join-Path $demoRoot "UnityRandomAgentDemo\ProjectSettings"
$resultsRoot = Join-Path $repositoryRoot "results\generic_scene_agent"
$normalFiles = @(
    "GeneratedNormalHmdIntegrity.cs", "NormalHmdAgentController.cs", "NormalHmdDemoBootstrap.cs",
    "NormalHmdLstmRuntime.cs", "NormalHmdModelData.cs", "NormalHmdVerticalEventController.cs",
    "SceneAwareContracts.cs", "SeededRootInnovation.cs"
)
$genericRuntimeFiles = @(
    "GenericSceneContract.cs", "UnityDecisionBrainRuntime.cs", "GenericSceneAgentBrain.cs",
    "GenericSceneAgentBootstrap.cs", "GenericRenderProbe.cs"
)
$runtimeSourceSha256 = [ordered]@{}
foreach ($name in $normalFiles) {
    $relative = "demo/UnityNormalHmdAgentDemo/Assets/Scripts/$name"
    $runtimeSourceSha256[$relative] = (Get-FileHash -LiteralPath (Join-Path $normalSource $name) -Algorithm SHA256).Hash.ToLowerInvariant()
}
foreach ($name in $genericRuntimeFiles) {
    $relative = "demo/generic_scene_agent/$name"
    $runtimeSourceSha256[$relative] = (Get-FileHash -LiteralPath (Join-Path $genericSource $name) -Algorithm SHA256).Hash.ToLowerInvariant()
}
$runtimeSourceSha256["demo/generic_scene_agent/GenericSceneAgentBuilder.cs"] =
    (Get-FileHash -LiteralPath (Join-Path $genericSource "GenericSceneAgentBuilder.cs") -Algorithm SHA256).Hash.ToLowerInvariant()
$buildPipelineSha256 = [ordered]@{
    "demo/build_generic_scene_agent.ps1" = (Get-FileHash -LiteralPath $MyInvocation.MyCommand.Path -Algorithm SHA256).Hash.ToLowerInvariant()
    "scripts/materialize_generic_runtime_contract.py" = (Get-FileHash -LiteralPath $materializer -Algorithm SHA256).Hash.ToLowerInvariant()
    "scripts/validate_render_capture.py" = (Get-FileHash -LiteralPath $renderValidator -Algorithm SHA256).Hash.ToLowerInvariant()
}

foreach ($file in @($unityEditor, $pythonExe, $materializer, $renderValidator, $compiledContractPath, $normalModel)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { Stop-Build "Required file is missing: $file" }
}

$decisionBrainSchema = "unity-decision-brain-unity-flat-1.0"
$decisionBrainModelPath = if ([string]::IsNullOrWhiteSpace($DecisionBrainModel)) {
    [System.IO.Path]::GetFullPath($defaultDecisionBrainModel)
} else {
    [System.IO.Path]::GetFullPath($DecisionBrainModel)
}
$decisionBrainModelStatus = "not_required"
$decisionBrainModelSha256 = ""
$decisionBrainModelSchemaVersion = ""
$decisionBrainDataStatus = ""
$decisionBrainModelDetail = "Rule mode does not require a Decision Brain model."
if (Test-Path -LiteralPath $decisionBrainModelPath -PathType Leaf) {
    try {
        $decisionBrainPayload = Get-Content -LiteralPath $decisionBrainModelPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $decisionBrainModelSchemaVersion = [string]$decisionBrainPayload.schemaVersion
        $decisionBrainDataStatus = [string]$decisionBrainPayload.dataStatus
        $knownDataStatus = $decisionBrainDataStatus -in @("smoke_only", "trained", "production_candidate")
        if ($decisionBrainModelSchemaVersion -eq $decisionBrainSchema -and $knownDataStatus) {
            $decisionBrainModelSha256 = (Get-FileHash -LiteralPath $decisionBrainModelPath -Algorithm SHA256).Hash.ToLowerInvariant()
            $decisionBrainModelStatus = "bundled"
            $decisionBrainModelDetail = "Validated Unity-flat Decision Brain model."
        } elseif ($decisionBrainModelSchemaVersion -ne $decisionBrainSchema) {
            $decisionBrainModelDetail = "Unsupported Decision Brain schema '$decisionBrainModelSchemaVersion'; expected '$decisionBrainSchema'."
        } else {
            $decisionBrainModelDetail = "Unsupported Decision Brain dataStatus '$decisionBrainDataStatus'."
        }
    }
    catch {
        $decisionBrainModelDetail = "Decision Brain model JSON is invalid: $($_.Exception.Message)"
    }
} else {
    $decisionBrainModelDetail = "Decision Brain model is missing: $decisionBrainModelPath"
}
if ($DecisionBrainMode -ne "rule" -and $decisionBrainModelStatus -ne "bundled") {
    Stop-Build "$DecisionBrainMode mode requires a valid Unity-flat Decision Brain model. $decisionBrainModelDetail"
}
if ($DecisionBrainMode -eq "model" -and $decisionBrainDataStatus -ne "production_candidate" -and -not $DecisionBrainResearchOverride) {
    Stop-Build "Model authority is blocked for non-production dataStatus '$decisionBrainDataStatus'. Re-run with -DecisionBrainResearchOverride only for explicit research use."
}
if (-not (Test-Path -LiteralPath (Join-Path $sourceProjectPath "Assets") -PathType Container)) {
    Stop-Build "Not a Unity project/evidence root with Assets: $sourceProjectPath"
}

$compiled = Get-Content -LiteralPath $compiledContractPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($compiled.schema_version -ne "scene-agent-contract-1.0") {
    Stop-Build "Unsupported compiled contract schema: $($compiled.schema_version)"
}
if ([string]::IsNullOrWhiteSpace($ScenePath)) { $ScenePath = [string]$compiled.runtime_scene.path }
if (-not $ScenePath.StartsWith("Assets/") -or -not $ScenePath.EndsWith(".unity")) {
    Stop-Build "ScenePath must be a project-relative Assets/*.unity path: $ScenePath"
}
$sourceScene = Join-Path $sourceProjectPath ($ScenePath -replace '/', '\')
if (-not (Test-Path -LiteralPath $sourceScene -PathType Leaf)) {
    Stop-Build "Selected scene is absent from source project: $sourceScene"
}
$sourceSceneHashBefore = (Get-FileHash -LiteralPath $sourceScene -Algorithm SHA256).Hash.ToLowerInvariant()
$reviewedBindingsPath = ""
$reviewedBindingsSha256 = ""
$reviewedBindingCount = 0
if (-not [string]::IsNullOrWhiteSpace($ReviewedBindings)) {
    $reviewedBindingsPath = [System.IO.Path]::GetFullPath($ReviewedBindings)
    if (-not (Test-Path -LiteralPath $reviewedBindingsPath -PathType Leaf)) {
        Stop-Build "Reviewed bindings file is missing: $reviewedBindingsPath"
    }
    $reviewedPayload = Get-Content -LiteralPath $reviewedBindingsPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($reviewedPayload.schemaVersion -ne "scene-agent-reviewed-bindings-1.0" -or
        $reviewedPayload.approvalStatus -ne "approved" -or
        $null -eq $reviewedPayload.bindings) {
        Stop-Build "Reviewed bindings are not an approved scene-agent-reviewed-bindings-1.0 manifest: $reviewedBindingsPath"
    }
    $reviewedBindingsSha256 = (Get-FileHash -LiteralPath $reviewedBindingsPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $reviewedBindingCount = @($reviewedPayload.bindings).Count
}

$projectSlug = ([System.IO.Path]::GetFileName($sourceProjectPath) -replace '[^A-Za-z0-9_-]', '_')
$sceneSlug = ([System.IO.Path]::GetFileNameWithoutExtension($ScenePath) -replace '[^A-Za-z0-9_-]', '_')
$identityText = (
    $sourceProjectPath.ToLowerInvariant() + "|" +
    (($ScenePath -replace '\\', '/').ToLowerInvariant()) + "|" +
    $DecisionBrainMode + "|" +
    $decisionBrainModelSha256 + "|" +
    $DecisionBrainResearchOverride.IsPresent.ToString().ToLowerInvariant()
)
$identitySha = [System.Security.Cryptography.SHA256]::Create()
try {
    $identityDigest = $identitySha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($identityText))
    $identityShortHash = ([System.BitConverter]::ToString($identityDigest) -replace '-', '').Substring(0, 10).ToLowerInvariant()
}
finally {
    $identitySha.Dispose()
}
$artifactSlug = $projectSlug + "_" + $sceneSlug + "_" + $identityShortHash
if ([string]::IsNullOrWhiteSpace($StagingDirectory)) {
    $StagingDirectory = Join-Path $demoRoot "UnityGenericSceneAgentProjects\$artifactSlug"
}
if ([string]::IsNullOrWhiteSpace($BuildDirectory)) {
    $BuildDirectory = Join-Path $demoRoot "GenericSceneAgentBuilds\$artifactSlug"
}
$stagingPath = [System.IO.Path]::GetFullPath($StagingDirectory)
$buildPath = [System.IO.Path]::GetFullPath($BuildDirectory)
$allowedStagingRoot = [System.IO.Path]::GetFullPath((Join-Path $demoRoot "UnityGenericSceneAgentProjects"))
$allowedBuildRoot = [System.IO.Path]::GetFullPath((Join-Path $demoRoot "GenericSceneAgentBuilds"))
if (-not $stagingPath.StartsWith($allowedStagingRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    Stop-Build "Staging directory must remain inside $allowedStagingRoot"
}
if (-not $buildPath.StartsWith($allowedBuildRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    Stop-Build "Build directory must remain inside $allowedBuildRoot"
}
if ($sourceProjectPath.Equals($stagingPath, [System.StringComparison]::OrdinalIgnoreCase) -or
    $sourceProjectPath.Equals($buildPath, [System.StringComparison]::OrdinalIgnoreCase) -or
    $sourceProjectPath.Equals($allowedStagingRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
    $sourceProjectPath.Equals($allowedBuildRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
    $sourceProjectPath.StartsWith($allowedStagingRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase) -or
    $sourceProjectPath.StartsWith($allowedBuildRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase) -or
    $stagingPath.StartsWith($sourceProjectPath + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase) -or
    $buildPath.StartsWith($sourceProjectPath + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    Stop-Build "Unsafe source/staging/build overlap detected; the source project must remain outside generated roots."
}
New-Item -ItemType Directory -Path $resultsRoot -Force | Out-Null

$runtimeContract = Join-Path $resultsRoot ($artifactSlug + "_runtime_contract.json")
$materializeArguments = @(
    $materializer,
    "--compiled-contract", $compiledContractPath,
    "--output", $runtimeContract,
    "--maximum-targets", "8"
)
if (-not [string]::IsNullOrWhiteSpace($ReviewedBindings)) {
    $materializeArguments += @("--reviewed-bindings", [System.IO.Path]::GetFullPath($ReviewedBindings))
}
foreach ($typeName in $DisableBehaviourType) {
    $materializeArguments += @("--disable-behaviour", $typeName)
}
Write-Host "[1/5] Materializing the analyser-to-runtime contract..."
& $pythonExe @materializeArguments
if ($LASTEXITCODE -ne 0) { Stop-Build "Runtime contract materialization failed." $LASTEXITCODE }

Write-Host "[2/5] Creating an isolated staging copy (source project remains untouched)..."
Copy-Tree (Join-Path $sourceProjectPath "Assets") (Join-Path $stagingPath "Assets")
$sourcePackages = Join-Path $sourceProjectPath "Packages"
$sourceSettings = Join-Path $sourceProjectPath "ProjectSettings"
Copy-Tree ($(if (Test-Path -LiteralPath $sourcePackages -PathType Container) { $sourcePackages } else { $fallbackPackages })) (Join-Path $stagingPath "Packages")
Copy-Tree ($(if (Test-Path -LiteralPath $sourceSettings -PathType Container) { $sourceSettings } else { $fallbackProjectSettings })) (Join-Path $stagingPath "ProjectSettings")

$stagingAssets = Join-Path $stagingPath "Assets"
$runtimeScripts = Join-Path $stagingAssets "GenericSceneAgentRuntime"
$runtimeEditor = Join-Path $runtimeScripts "Editor"
$runtimeResources = Join-Path $stagingAssets "Resources"
$runtimeStreamingAssets = Join-Path $stagingAssets "StreamingAssets"
New-Item -ItemType Directory -Path $runtimeScripts,$runtimeEditor,$runtimeResources,$runtimeStreamingAssets -Force | Out-Null
foreach ($name in $normalFiles) {
    Install-CompatibleRuntimeScript `
        -CanonicalPath (Join-Path $normalSource $name) `
        -DestinationDirectory $runtimeScripts `
        -SearchRoot $stagingAssets `
        -ComponentLabel "Normal HMD"
}
foreach ($name in $genericRuntimeFiles) {
    Install-CompatibleRuntimeScript `
        -CanonicalPath (Join-Path $genericSource $name) `
        -DestinationDirectory $runtimeScripts `
        -SearchRoot $stagingAssets `
        -ComponentLabel "Generic Scene Agent"
}
Install-CompatibleRuntimeScript `
    -CanonicalPath (Join-Path $genericSource "GenericSceneAgentBuilder.cs") `
    -DestinationDirectory $runtimeEditor `
    -SearchRoot $stagingAssets `
    -ComponentLabel "Generic Scene Agent editor"
Copy-Item -LiteralPath $normalModel -Destination (Join-Path $runtimeResources "NormalHmdAgentModel.json") -Force
Copy-Item -LiteralPath $runtimeContract -Destination (Join-Path $runtimeResources "AgentSceneContract.json") -Force
$decisionBrainResource = Join-Path $runtimeResources "UnityDecisionBrainModel.json"
$decisionBrainStreamingCopy = Join-Path $runtimeStreamingAssets "UnityDecisionBrainModel.json"
$conflictingDecisionResources = @(
    Get-ChildItem -LiteralPath $stagingAssets -Filter "UnityDecisionBrainModel.json" -File -Recurse -ErrorAction Stop
)
if ($conflictingDecisionResources.Count -gt 0) {
    $relativeConflicts = @($conflictingDecisionResources | ForEach-Object {
        $_.FullName.Substring($stagingAssets.Length).TrimStart('\', '/')
    })
    Stop-Build (
        "The source project already contains reserved UnityDecisionBrainModel.json resources. " +
        "Refusing an ambiguous Resources.Load. Rename or remove these source assets explicitly: " +
        ($relativeConflicts -join "; ")
    )
}
foreach ($reservedModelPath in @($decisionBrainResource, $decisionBrainStreamingCopy)) {
    if (Test-Path -LiteralPath $reservedModelPath -PathType Leaf) {
        Remove-Item -LiteralPath $reservedModelPath -Force
    }
}
if ($decisionBrainModelStatus -eq "bundled") {
    Copy-Item -LiteralPath $decisionBrainModelPath -Destination $decisionBrainResource -Force
    Copy-Item -LiteralPath $decisionBrainModelPath -Destination $decisionBrainStreamingCopy -Force
    Write-Host "      [inject] Assets/Resources/UnityDecisionBrainModel.json (Decision Brain)"
    Write-Host "      [audit]  Assets/StreamingAssets/UnityDecisionBrainModel.json (Decision Brain evidence copy)"
} else {
    Write-Host "      [rule] Decision Brain model not bundled; deterministic Rule remains available."
}

$buildLog = Join-Path $resultsRoot ($artifactSlug + "_build.log")
$buildStatus = Join-Path $resultsRoot ($artifactSlug + "_build_status.json")
$acceptance = Join-Path $resultsRoot ($artifactSlug + "_acceptance.json")
$playerLog = Join-Path $resultsRoot ($artifactSlug + "_player.log")
$runRoot = Join-Path $resultsRoot ($artifactSlug + "_runs")
$screenshot = Join-Path $resultsRoot ($artifactSlug + "_first_person.png")
$cameraAudit = Join-Path $resultsRoot ($artifactSlug + "_camera_audit.json")
$renderValidation = Join-Path $resultsRoot ($artifactSlug + "_render_validation.json")
$executable = Join-Path $buildPath "GenericSceneAgentDemo.exe"
foreach ($stale in @($buildStatus, $acceptance, $screenshot, $cameraAudit, $renderValidation)) {
    if (Test-Path -LiteralPath $stale -PathType Leaf) { Remove-Item -LiteralPath $stale -Force }
}

Write-Host "[3/5] Building selected scene with the generic runtime bridge..."
$oldBuild = [Environment]::GetEnvironmentVariable("GENERIC_SCENE_AGENT_BUILD_DIR", "Process")
$oldStatus = [Environment]::GetEnvironmentVariable("GENERIC_SCENE_AGENT_BUILD_STATUS", "Process")
$oldScene = [Environment]::GetEnvironmentVariable("GENERIC_SCENE_AGENT_SCENE_PATH", "Process")
$oldSourceProject = [Environment]::GetEnvironmentVariable("GENERIC_SCENE_AGENT_SOURCE_PROJECT", "Process")
$oldCompiledContract = [Environment]::GetEnvironmentVariable("GENERIC_SCENE_AGENT_COMPILED_CONTRACT", "Process")
$oldCompiledHash = [Environment]::GetEnvironmentVariable("GENERIC_SCENE_AGENT_COMPILED_CONTRACT_SHA256", "Process")
$oldSceneHash = [Environment]::GetEnvironmentVariable("GENERIC_SCENE_AGENT_SOURCE_SCENE_SHA256", "Process")
$oldRuntimeHash = [Environment]::GetEnvironmentVariable("GENERIC_SCENE_AGENT_RUNTIME_CONTRACT_SHA256", "Process")
$oldReviewedPath = [Environment]::GetEnvironmentVariable("GENERIC_SCENE_AGENT_REVIEWED_BINDINGS", "Process")
$oldReviewedHash = [Environment]::GetEnvironmentVariable("GENERIC_SCENE_AGENT_REVIEWED_BINDINGS_SHA256", "Process")
$oldReviewedCount = [Environment]::GetEnvironmentVariable("GENERIC_SCENE_AGENT_REVIEWED_BINDING_COUNT", "Process")
$oldDecisionMode = [Environment]::GetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_MODE", "Process")
$oldDecisionStatus = [Environment]::GetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_MODEL_STATUS", "Process")
$oldDecisionDetail = [Environment]::GetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_MODEL_DETAIL", "Process")
$oldDecisionPath = [Environment]::GetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_MODEL_PATH", "Process")
$oldDecisionHash = [Environment]::GetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_MODEL_SHA256", "Process")
$oldDecisionSchema = [Environment]::GetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_MODEL_SCHEMA", "Process")
$oldDecisionDataStatus = [Environment]::GetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_DATA_STATUS", "Process")
$oldDecisionOverride = [Environment]::GetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_RESEARCH_OVERRIDE", "Process")
try {
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_BUILD_DIR", $buildPath, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_BUILD_STATUS", $buildStatus, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_SCENE_PATH", $ScenePath, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_SOURCE_PROJECT", $sourceProjectPath, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_COMPILED_CONTRACT", $compiledContractPath, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_COMPILED_CONTRACT_SHA256", (Get-FileHash -LiteralPath $compiledContractPath -Algorithm SHA256).Hash.ToLowerInvariant(), "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_SOURCE_SCENE_SHA256", $sourceSceneHashBefore, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_RUNTIME_CONTRACT_SHA256", (Get-FileHash -LiteralPath $runtimeContract -Algorithm SHA256).Hash.ToLowerInvariant(), "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_REVIEWED_BINDINGS", $reviewedBindingsPath, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_REVIEWED_BINDINGS_SHA256", $reviewedBindingsSha256, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_REVIEWED_BINDING_COUNT", [string]$reviewedBindingCount, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_MODE", $DecisionBrainMode, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_MODEL_STATUS", $decisionBrainModelStatus, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_MODEL_DETAIL", $decisionBrainModelDetail, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_MODEL_PATH", $(if ($decisionBrainModelStatus -eq "bundled") { $decisionBrainModelPath } else { "" }), "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_MODEL_SHA256", $decisionBrainModelSha256, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_MODEL_SCHEMA", $decisionBrainModelSchemaVersion, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_DATA_STATUS", $decisionBrainDataStatus, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_RESEARCH_OVERRIDE", $DecisionBrainResearchOverride.IsPresent.ToString().ToLowerInvariant(), "Process")
    $unityArguments = @(
        "-batchmode", "-quit", "-projectPath", $stagingPath,
        "-executeMethod", "GenericSceneAgent.Editor.GenericSceneAgentBuilder.BuildWindows",
        "-logFile", $buildLog
    )
    $unityProcess = Start-Process -FilePath $unityEditor -ArgumentList $unityArguments -WindowStyle Hidden -PassThru
    $deadline = [DateTime]::UtcNow.AddMinutes(20)
    while (-not $unityProcess.HasExited -and [DateTime]::UtcNow -lt $deadline) {
        Start-Sleep -Milliseconds 500
        $unityProcess.Refresh()
    }
    if (-not $unityProcess.HasExited) {
        Stop-Build "Unity build exceeded the 20-minute timeout; process id $($unityProcess.Id)."
    }
    $unityExitCode = $unityProcess.ExitCode
}
finally {
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_BUILD_DIR", $oldBuild, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_BUILD_STATUS", $oldStatus, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_SCENE_PATH", $oldScene, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_SOURCE_PROJECT", $oldSourceProject, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_COMPILED_CONTRACT", $oldCompiledContract, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_COMPILED_CONTRACT_SHA256", $oldCompiledHash, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_SOURCE_SCENE_SHA256", $oldSceneHash, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_RUNTIME_CONTRACT_SHA256", $oldRuntimeHash, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_REVIEWED_BINDINGS", $oldReviewedPath, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_REVIEWED_BINDINGS_SHA256", $oldReviewedHash, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_REVIEWED_BINDING_COUNT", $oldReviewedCount, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_MODE", $oldDecisionMode, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_MODEL_STATUS", $oldDecisionStatus, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_MODEL_DETAIL", $oldDecisionDetail, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_MODEL_PATH", $oldDecisionPath, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_MODEL_SHA256", $oldDecisionHash, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_MODEL_SCHEMA", $oldDecisionSchema, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_DATA_STATUS", $oldDecisionDataStatus, "Process")
    [Environment]::SetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_RESEARCH_OVERRIDE", $oldDecisionOverride, "Process")
}
if ($unityExitCode -ne 0) { Stop-Build "Unity build failed; see $buildLog" $unityExitCode }
if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) { Stop-Build "Built player missing: $executable" }

# Verify and record source integrity before acceptance starts.  This ordering is
# intentional: consumers use the build-status mtime as the lower bound for a
# fresh acceptance run, so the status must not be rewritten after acceptance.
$sourceSceneHashAfter = (Get-FileHash -LiteralPath $sourceScene -Algorithm SHA256).Hash.ToLowerInvariant()
$sourceSceneModified = $sourceSceneHashAfter -ne $sourceSceneHashBefore
if (Test-Path -LiteralPath $buildStatus -PathType Leaf) {
    $finalBuildStatus = Get-Content -LiteralPath $buildStatus -Raw -Encoding UTF8 | ConvertFrom-Json
    $finalBuildStatus.sourceProjectWasModified = $sourceSceneModified
    $finalBuildStatus | Add-Member -NotePropertyName sourceSceneSha256After -NotePropertyValue $sourceSceneHashAfter -Force
    $finalBuildStatus | Add-Member -NotePropertyName sourceSceneIntegrityVerifiedAtUtc -NotePropertyValue ([DateTime]::UtcNow.ToString("O")) -Force
    $finalBuildStatus | Add-Member -NotePropertyName decisionBrainModeValidated -NotePropertyValue $DecisionBrainMode -Force
    $finalBuildStatus | Add-Member -NotePropertyName decisionBrainModelStatus -NotePropertyValue $decisionBrainModelStatus -Force
    $finalBuildStatus | Add-Member -NotePropertyName decisionBrainModelDetail -NotePropertyValue $decisionBrainModelDetail -Force
    $finalBuildStatus | Add-Member -NotePropertyName decisionBrainModelPath -NotePropertyValue $(if ($decisionBrainModelStatus -eq "bundled") { $decisionBrainModelPath } else { $null }) -Force
    $finalBuildStatus | Add-Member -NotePropertyName decisionBrainModelSha256 -NotePropertyValue $(if ($decisionBrainModelStatus -eq "bundled") { $decisionBrainModelSha256 } else { $null }) -Force
    $finalBuildStatus | Add-Member -NotePropertyName decisionBrainModelSchemaVersion -NotePropertyValue $(if ($decisionBrainModelStatus -eq "bundled") { $decisionBrainModelSchemaVersion } else { $null }) -Force
    $finalBuildStatus | Add-Member -NotePropertyName decisionBrainDataStatus -NotePropertyValue $(if ($decisionBrainModelStatus -eq "bundled") { $decisionBrainDataStatus } else { $null }) -Force
    $finalBuildStatus | Add-Member -NotePropertyName decisionBrainResearchOverride -NotePropertyValue ([bool]$DecisionBrainResearchOverride) -Force
    $finalBuildStatus | Add-Member -NotePropertyName decisionBrainResourcePath -NotePropertyValue $(if ($decisionBrainModelStatus -eq "bundled") { "Assets/Resources/UnityDecisionBrainModel.json" } else { $null }) -Force
    $finalBuildStatus | Add-Member -NotePropertyName decisionBrainStreamingAssetsPath -NotePropertyValue $(if ($decisionBrainModelStatus -eq "bundled") { "Assets/StreamingAssets/UnityDecisionBrainModel.json" } else { $null }) -Force
    $finalBuildStatus | Add-Member -NotePropertyName normalHmdModelArtifactSha256 -NotePropertyValue ((Get-FileHash -LiteralPath $normalModel -Algorithm SHA256).Hash.ToLowerInvariant()) -Force
    $finalBuildStatus | Add-Member -NotePropertyName runtimeSourceSha256 -NotePropertyValue ([pscustomobject]$runtimeSourceSha256) -Force
    $finalBuildStatus | Add-Member -NotePropertyName buildPipelineSha256 -NotePropertyValue ([pscustomobject]$buildPipelineSha256) -Force
    $finalBuildStatus | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $buildStatus -Encoding utf8
}
if ($sourceSceneModified) {
    Stop-Build "Source scene changed during staging/build; refusing acceptance: $sourceScene"
}

Write-Host "[4/5] Running accelerated content-completion acceptance with D3D11 capture..."
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
$playerArguments = @(
    "-force-d3d11", "-screen-fullscreen", "0", "-screen-width", "1280", "-screen-height", "720",
    "-logFile", $playerLog,
    "--seed=$Seed", "--generic-agent-test",
    "--decision-brain-mode=$DecisionBrainMode",
    "--generic-agent-log-root=$runRoot",
    "--generic-agent-test-output=$acceptance",
    "--generic-agent-capture-output=$screenshot",
    "--generic-agent-camera-audit=$cameraAudit"
)
if ($DecisionBrainResearchOverride) {
    $playerArguments += "--decision-brain-research-override"
}
$player = Start-Process -FilePath $executable -ArgumentList $playerArguments -PassThru
$acceptanceDeadline = [DateTime]::UtcNow.AddMinutes(3)
while (-not $player.HasExited -and [DateTime]::UtcNow -lt $acceptanceDeadline) {
    Start-Sleep -Milliseconds 250
    $player.Refresh()
}
if (-not $player.HasExited) {
    Stop-Process -Id $player.Id -Force -ErrorAction SilentlyContinue
    Stop-Build (
        "Generic Agent acceptance exceeded the 3-minute wall-clock guard. " +
        "The runtime also has simulated-time stall and safety limits; see $playerLog."
    )
}
if ($player.ExitCode -ne 0) { Stop-Build "Generic Agent acceptance player failed; see $playerLog" $player.ExitCode }
if (-not (Test-Path -LiteralPath $acceptance -PathType Leaf)) { Stop-Build "Acceptance JSON missing: $acceptance" }
$accepted = Get-Content -LiteralPath $acceptance -Raw -Encoding UTF8 | ConvertFrom-Json
if ($accepted.status -ne "passed") { Stop-Build "Acceptance status is $($accepted.status)" }
if ($accepted.completionReason -ne "content_complete" -or
    $accepted.contentCompletionSatisfied -ne $true) {
    Stop-Build (
        "Acceptance did not complete all declared content. reason=" +
        "$($accepted.completionReason) contentComplete=$($accepted.contentCompletionSatisfied)"
    )
}
if ([int]$accepted.targetsDeclared -lt 1 -or
    [int]$accepted.targetsVisited -ne [int]$accepted.targetsDeclared -or
    [int]$accepted.unfinishedTargetCount -ne 0 -or
    [int]$accepted.interactionsRequiredVerified -lt [int]$accepted.interactionsRequired) {
    Stop-Build (
        "Acceptance content accounting is incomplete. visited=" +
        "$($accepted.targetsVisited)/$($accepted.targetsDeclared) unfinished=" +
        "$($accepted.unfinishedTargetCount) requiredInteractions=" +
        "$($accepted.interactionsRequiredVerified)/$($accepted.interactionsRequired)"
    )
}
if ($accepted.interactionMotionLockPassed -ne $true -or
    [int]$accepted.interactionMotionLockViolations -ne 0 -or
    [int]$accepted.interactionMotionLockAcquisitions -ne [int]$accepted.interactionMotionLockReleases -or
    [int]$accepted.interactionsAttempted -gt [int]$accepted.interactionMotionLockAcquisitions) {
    Stop-Build "Interaction motion-lock evidence failed."
}
if ([int]$accepted.interactionsAttempted -gt 0 -and
    ([int]$accepted.interactionMotionLockedTelemetryRows -lt 1 -or
     [int]$accepted.interactionHeadMotionTelemetryRows -lt 1)) {
    Stop-Build "Interaction occurred without locked BodyRoot telemetry and active HeadLocal evidence."
}
if ([string]$accepted.collisionSafetyMode -ne
        "full-height-capsule-sweep+stable-side-steering+inertia-hard-stop" -or
    $accepted.collisionSafetyPassed -ne $true -or
    [int]$accepted.collisionOverlapDetections -ne 0) {
    Stop-Build "Collision safety evidence failed or a BodyRoot overlap was detected."
}
$expectedRuntimeStatus = if ($DecisionBrainMode -eq "rule") { "not_requested" } else { "ready" }
$expectedAuthority = if ($DecisionBrainMode -eq "rule") {
    "rule"
} elseif ($DecisionBrainMode -eq "shadow") {
    "rule_shadow"
} else {
    "model"
}
if ($accepted.decisionBrainModeRequested -ne $DecisionBrainMode -or
    $accepted.decisionBrainModelStatus -ne $expectedRuntimeStatus -or
    $accepted.policyAuthority -ne $expectedAuthority) {
    Stop-Build (
        "Decision Brain acceptance mismatch. requested=$DecisionBrainMode " +
        "reported=$($accepted.decisionBrainModeRequested) status=$($accepted.decisionBrainModelStatus) " +
        "authority=$($accepted.policyAuthority)"
    )
}
if ($DecisionBrainMode -eq "rule" -and [int]$accepted.decisionBrainInferences -ne 0) {
    Stop-Build "Rule acceptance unexpectedly executed Decision Brain inference."
}
if ($DecisionBrainMode -ne "rule" -and [int]$accepted.decisionBrainInferences -lt 1) {
    Stop-Build "$DecisionBrainMode acceptance did not execute any Decision Brain inference."
}
if ($DecisionBrainMode -ne "rule" -and [int]$accepted.decisionBrainFallbacks -ne 0) {
    Stop-Build "$DecisionBrainMode acceptance used $($accepted.decisionBrainFallbacks) fallback decisions."
}
if ($DecisionBrainMode -ne "rule" -and
    ($accepted.decisionBrainGoldenSelfTestPassed -ne $true -or
     $accepted.decisionBrainAllInferenceLogitsFinite -ne $true)) {
    Stop-Build "$DecisionBrainMode acceptance failed golden-parity or finite-logit evidence gates."
}
if ($DecisionBrainMode -ne "rule" -and
    ([string]$accepted.decisionBrainModelSchemaVersion -ne $decisionBrainModelSchemaVersion -or
     [string]$accepted.decisionBrainDataStatus -ne $decisionBrainDataStatus)) {
    Stop-Build "$DecisionBrainMode runtime model schema/dataStatus differs from the bundled model."
}
if ($DecisionBrainMode -ne "rule" -and
    [string]$accepted.decisionBrainModelSha256 -ne $decisionBrainModelSha256) {
    Stop-Build "Runtime Decision Brain hash does not match the bundled model."
}

Write-Host "[5/5] Checking the rendered first-person frame..."
& $pythonExe $renderValidator --image $screenshot --output $renderValidation
if ($LASTEXITCODE -ne 0) { Stop-Build "Rendered frame validation failed: $renderValidation" $LASTEXITCODE }
if (-not (Test-Path -LiteralPath $cameraAudit -PathType Leaf)) { Stop-Build "Camera audit missing: $cameraAudit" }
$camera = Get-Content -LiteralPath $cameraAudit -Raw -Encoding UTF8 | ConvertFrom-Json
if ($camera.status -ne "captured" -or
    $camera.activeCameraName -notlike "*ParticipantCamera*" -or
    [int]$camera.enabledCameraCount -ne 1 -or
    [int]$camera.enabledAudioListenerCount -ne 1) {
    Stop-Build "Camera audit did not verify the AI first-person camera: $cameraAudit"
}

Write-Host "Generic Scene Agent passed: $executable"
Write-Host "Acceptance: $acceptance"
Write-Host "Experience reports are inside: $runRoot"
Write-Host "Screenshot: $screenshot"
exit 0
