using System;
using System.IO;
using NormalHmdAgentDemo;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace GenericSceneAgent.Editor
{
    public static class GenericSceneAgentBuilder
    {
        public static void BuildWindows()
        {
            DateTime started = DateTime.UtcNow;
            string buildDirectory = Environment.GetEnvironmentVariable("GENERIC_SCENE_AGENT_BUILD_DIR");
            string statusPath = Environment.GetEnvironmentVariable("GENERIC_SCENE_AGENT_BUILD_STATUS");
            string scenePath = Environment.GetEnvironmentVariable("GENERIC_SCENE_AGENT_SCENE_PATH");
            if (string.IsNullOrWhiteSpace(buildDirectory))
                throw new InvalidOperationException("GENERIC_SCENE_AGENT_BUILD_DIR is required.");
            if (string.IsNullOrWhiteSpace(statusPath))
                throw new InvalidOperationException("GENERIC_SCENE_AGENT_BUILD_STATUS is required.");
            if (string.IsNullOrWhiteSpace(scenePath) || !scenePath.StartsWith("Assets/") ||
                !scenePath.EndsWith(".unity", StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException("GENERIC_SCENE_AGENT_SCENE_PATH must be an Assets/*.unity path.");
            if (!File.Exists(Path.Combine(Path.GetFullPath(Path.Combine(Application.dataPath, "..")), scenePath)))
                throw new FileNotFoundException("Selected runtime scene is missing in staging project.", scenePath);
            buildDirectory = Path.GetFullPath(buildDirectory);
            statusPath = Path.GetFullPath(statusPath);
            Directory.CreateDirectory(buildDirectory);
            Directory.CreateDirectory(Path.GetDirectoryName(statusPath));

            NormalHmdLstmRuntime verified = NormalHmdLstmRuntime.LoadVerified();
            PlayerSettings.companyName = "Cybersickness Preassessment Research Prototype";
            PlayerSettings.productName = "Generic Scene Agent Brain Demo";
            PlayerSettings.defaultScreenWidth = 1440;
            PlayerSettings.defaultScreenHeight = 810;
            PlayerSettings.fullScreenMode = FullScreenMode.Windowed;
            PlayerSettings.resizableWindow = true;
            PlayerSettings.runInBackground = true;
            PlayerSettings.colorSpace = ColorSpace.Linear;

            string executable = Path.Combine(buildDirectory, "GenericSceneAgentDemo.exe");
            BuildReport report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = new[] { scenePath },
                locationPathName = executable,
                target = BuildTarget.StandaloneWindows64,
                options = BuildOptions.None,
            });
            BuildSummary summary = report.summary;
            int reviewedBindingCount;
            if (!int.TryParse(
                Environment.GetEnvironmentVariable("GENERIC_SCENE_AGENT_REVIEWED_BINDING_COUNT"),
                out reviewedBindingCount))
                reviewedBindingCount = 0;
            BuildStatus status = new BuildStatus
            {
                status = summary.result == BuildResult.Succeeded ? "success" : "failed",
                unityVersion = Application.unityVersion,
                sourceScene = scenePath,
                sourceProject = Environment.GetEnvironmentVariable("GENERIC_SCENE_AGENT_SOURCE_PROJECT"),
                compiledContractPath = Environment.GetEnvironmentVariable("GENERIC_SCENE_AGENT_COMPILED_CONTRACT"),
                compiledContractSha256 = Environment.GetEnvironmentVariable("GENERIC_SCENE_AGENT_COMPILED_CONTRACT_SHA256"),
                sourceSceneSha256 = Environment.GetEnvironmentVariable("GENERIC_SCENE_AGENT_SOURCE_SCENE_SHA256"),
                runtimeContractSha256 = Environment.GetEnvironmentVariable("GENERIC_SCENE_AGENT_RUNTIME_CONTRACT_SHA256"),
                reviewedBindingsPath = Environment.GetEnvironmentVariable("GENERIC_SCENE_AGENT_REVIEWED_BINDINGS"),
                reviewedBindingsSha256 = Environment.GetEnvironmentVariable("GENERIC_SCENE_AGENT_REVIEWED_BINDINGS_SHA256"),
                reviewedBindingCount = reviewedBindingCount,
                stagingProject = Path.GetFullPath(Path.Combine(Application.dataPath, "..")),
                executable = executable,
                outputBytes = (long)summary.totalSize,
                warningCount = summary.totalWarnings,
                errorCount = summary.totalErrors,
                buildSeconds = (float)summary.totalTime.TotalSeconds,
                modelSha256 = verified.Data.sourceModelSha256,
                normalHmdModelSha256 = verified.Data.sourceModelSha256,
                modelGoldenMaxAbsError = verified.GoldenMaxAbsError,
                decisionBrainModeValidated = Environment.GetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_MODE"),
                decisionBrainModelStatus = Environment.GetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_MODEL_STATUS"),
                decisionBrainModelDetail = Environment.GetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_MODEL_DETAIL"),
                decisionBrainModelPath = Environment.GetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_MODEL_PATH"),
                decisionBrainModelSha256 = Environment.GetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_MODEL_SHA256"),
                decisionBrainModelSchemaVersion = Environment.GetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_MODEL_SCHEMA"),
                decisionBrainDataStatus = Environment.GetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_DATA_STATUS"),
                decisionBrainResearchOverride = string.Equals(
                    Environment.GetEnvironmentVariable("GENERIC_SCENE_AGENT_DECISION_BRAIN_RESEARCH_OVERRIDE"),
                    "true", StringComparison.OrdinalIgnoreCase),
                sourceProjectWasModified = false,
                startedAtUtc = started.ToString("O"),
                finishedAtUtc = DateTime.UtcNow.ToString("O")
            };
            File.WriteAllText(statusPath, JsonUtility.ToJson(status, true));
            if (summary.result != BuildResult.Succeeded)
                throw new InvalidOperationException("Generic Scene Agent build failed: " + summary.result);
            Debug.Log("GENERIC_SCENE_AGENT_BUILD_OK executable=" + executable);
        }

        [Serializable]
        private sealed class BuildStatus
        {
            public string status;
            public string unityVersion;
            public string sourceScene;
            public string sourceProject;
            public string compiledContractPath;
            public string compiledContractSha256;
            public string sourceSceneSha256;
            public string runtimeContractSha256;
            public string reviewedBindingsPath;
            public string reviewedBindingsSha256;
            public int reviewedBindingCount;
            public string stagingProject;
            public string executable;
            public long outputBytes;
            public int warningCount;
            public int errorCount;
            public float buildSeconds;
            public string modelSha256;
            public string normalHmdModelSha256;
            public float modelGoldenMaxAbsError;
            public string decisionBrainModeValidated;
            public string decisionBrainModelStatus;
            public string decisionBrainModelDetail;
            public string decisionBrainModelPath;
            public string decisionBrainModelSha256;
            public string decisionBrainModelSchemaVersion;
            public string decisionBrainDataStatus;
            public bool decisionBrainResearchOverride;
            public bool sourceProjectWasModified;
            public string startedAtUtc;
            public string finishedAtUtc;
        }
    }
}
