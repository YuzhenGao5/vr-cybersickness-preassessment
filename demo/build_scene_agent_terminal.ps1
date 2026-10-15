[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Stop-TerminalBuild {
    param([Parameter(Mandatory = $true)][string]$Message, [int]$ExitCode = 1)
    Write-Error -Message $Message -ErrorAction Continue
    exit $ExitCode
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptRoot ".."))
$pythonExe = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
$source = Join-Path $scriptRoot "scene_agent_terminal.py"
$outputDirectory = Join-Path $scriptRoot "SceneAgentTerminalBuild"
$expectedExecutable = Join-Path $outputDirectory "SceneAgentTerminal.exe"
$workDirectory = Join-Path $repositoryRoot "tmp\scene_agent_terminal_pyinstaller"
$specDirectory = Join-Path $repositoryRoot "tmp\scene_agent_terminal_spec"
$resultsDirectory = Join-Path $repositoryRoot "results\scene_agent_terminal"
$sourceTestPath = Join-Path $resultsDirectory "source_self_test.json"
$packagedTestPath = Join-Path $resultsDirectory "packaged_self_test.json"
$buildStatusPath = Join-Path $resultsDirectory "terminal_build_status.json"

foreach ($required in @($pythonExe, $source)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        Stop-TerminalBuild "Required file is missing: $required"
    }
}
foreach ($directory in @($outputDirectory, $workDirectory, $specDirectory, $resultsDirectory)) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}

Write-Host "[1/3] Running source-level terminal self-test..."
& $pythonExe $source --self-test --self-test-output $sourceTestPath
if ($LASTEXITCODE -ne 0) {
    Stop-TerminalBuild "Source terminal self-test failed." $LASTEXITCODE
}

$startedAt = [DateTime]::UtcNow
Write-Host "[2/3] Packaging SceneAgentTerminal.exe..."
$pyInstallerArguments = @(
    "-m", "PyInstaller",
    "--noconfirm", "--clean", "--onefile", "--windowed",
    "--name", "SceneAgentTerminal",
    "--distpath", $outputDirectory,
    "--workpath", $workDirectory,
    "--specpath", $specDirectory,
    $source
)
& $pythonExe @pyInstallerArguments
if ($LASTEXITCODE -ne 0) {
    Stop-TerminalBuild "PyInstaller failed with exit code $LASTEXITCODE." $LASTEXITCODE
}
if (-not (Test-Path -LiteralPath $expectedExecutable -PathType Leaf)) {
    Stop-TerminalBuild "Packaged terminal is missing: $expectedExecutable"
}

Write-Host "[3/3] Running packaged terminal self-test..."
if (Test-Path -LiteralPath $packagedTestPath -PathType Leaf) {
    Remove-Item -LiteralPath $packagedTestPath -Force
}
$selfTestArguments = "--self-test --self-test-output `"$packagedTestPath`""
$processInfo = [System.Diagnostics.ProcessStartInfo]::new()
$processInfo.FileName = $expectedExecutable
$processInfo.UseShellExecute = $false
$processInfo.CreateNoWindow = $true
$processInfo.Arguments = $selfTestArguments
$testProcess = [System.Diagnostics.Process]::Start($processInfo)
if (-not $testProcess.WaitForExit(120000)) {
    try { $testProcess.Kill() } catch { }
    Stop-TerminalBuild "Packaged terminal self-test exceeded 120 seconds."
}
if ($testProcess.ExitCode -ne 0) {
    Stop-TerminalBuild "Packaged terminal self-test failed with exit code $($testProcess.ExitCode)." $testProcess.ExitCode
}
$selfTest = $null
$reportDeadline = [DateTime]::UtcNow.AddSeconds(30)
do {
    if (Test-Path -LiteralPath $packagedTestPath -PathType Leaf) {
        try {
            # The report contains Chinese project/scene text and is written as
            # UTF-8.  Windows PowerShell 5.1 otherwise assumes the active ANSI
            # code page and can turn valid JSON into an unparsable string.
            $selfTest = Get-Content -LiteralPath $packagedTestPath -Raw -Encoding UTF8 | ConvertFrom-Json
        }
        catch {
            $selfTest = $null
        }
    }
    if ($null -eq $selfTest) { Start-Sleep -Milliseconds 200 }
} while ($null -eq $selfTest -and [DateTime]::UtcNow -lt $reportDeadline)
if ($null -eq $selfTest) {
    Stop-TerminalBuild "Packaged self-test report is missing or invalid: $packagedTestPath"
}
if ($selfTest.status -ne "passed" -or [int]$selfTest.projectCount -lt 1 -or $selfTest.exportWritable -ne $true) {
    Stop-TerminalBuild "Packaged terminal self-test did not pass required gates: $packagedTestPath"
}

$file = Get-Item -LiteralPath $expectedExecutable
$status = [ordered]@{
    schemaVersion = "scene-agent-terminal-build-1.0"
    status = "success"
    startedAtUtc = $startedAt.ToString("o")
    finishedAtUtc = [DateTime]::UtcNow.ToString("o")
    executable = $expectedExecutable
    executableBytes = $file.Length
    executableSha256 = (Get-FileHash -LiteralPath $expectedExecutable -Algorithm SHA256).Hash.ToLowerInvariant()
    detectedProjectCount = [int]$selfTest.projectCount
    agentReadySceneCount = [int]$selfTest.agentReadySceneCount
    sourceSelfTest = $sourceTestPath
    packagedSelfTest = $packagedTestPath
}
$status | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $buildStatusPath -Encoding utf8

Write-Host "Scene Agent Terminal ready: $expectedExecutable"
Write-Host "Build status: $buildStatusPath"
