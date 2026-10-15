using System;
using System.IO;
using System.Text;
using NormalHmdAgentDemo;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace ExternalSceneAgent.Editor
{
    public static class HeartIslandAgentBuilder
    {
        private const string ScenePath = "Assets/Scenes/ResonanceValley.unity";

        public static void BuildWindows()
        {
            DateTime started = DateTime.UtcNow;
            string outputDirectory = Environment.GetEnvironmentVariable("HEARTISLAND_AGENT_BUILD_DIR");
            string statusPath = Environment.GetEnvironmentVariable("HEARTISLAND_AGENT_BUILD_STATUS");
            if (string.IsNullOrWhiteSpace(outputDirectory))
                outputDirectory = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Build", "HeartislandAgent"));
            if (string.IsNullOrWhiteSpace(statusPath))
                statusPath = Path.Combine(outputDirectory, "heartisland_agent_build_status.json");
            Directory.CreateDirectory(outputDirectory);
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(statusPath)) ?? outputDirectory);

            try
            {
                if (!File.Exists(ScenePath)) throw new FileNotFoundException("Scene is missing", ScenePath);
                NormalHmdLstmRuntime runtime = NormalHmdLstmRuntime.LoadVerified();
                PlayerSettings.companyName = "Cybersickness Preassessment Research Prototype";
                PlayerSettings.productName = "Heartisland Scene Agent Experience";
                PlayerSettings.bundleVersion = "1.0.0";
                PlayerSettings.runInBackground = true;
                string executable = Path.Combine(outputDirectory, "HeartislandAgent.exe");
                BuildReport report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
                {
                    scenes = new[] { ScenePath },
                    locationPathName = executable,
                    target = BuildTarget.StandaloneWindows64,
                    options = BuildOptions.None,
                });
                BuildSummary summary = report.summary;
                int warnings = 0;
                int errors = 0;
                foreach (BuildStep step in report.steps)
                foreach (BuildStepMessage message in step.messages)
                {
                    if (message.type == LogType.Warning) warnings++;
                    if (message.type == LogType.Error || message.type == LogType.Exception || message.type == LogType.Assert) errors++;
                }
                bool success = summary.result == BuildResult.Succeeded && errors == 0 && File.Exists(executable);
                BuildStatus status = new BuildStatus
                {
                    schemaVersion = "heartisland-agent-build-1.0",
                    status = success ? "success" : "failed",
                    unityVersion = Application.unityVersion,
                    sourceScene = ScenePath,
                    executable = Path.GetFullPath(executable),
                    outputBytes = summary.totalSize,
                    warningCount = warnings,
                    errorCount = errors,
                    sourceModelSha256 = runtime.Data.sourceModelSha256,
                    modelGoldenMaxAbsError = runtime.GoldenMaxAbsError,
                    startedAtUtc = started.ToString("o"),
                    finishedAtUtc = DateTime.UtcNow.ToString("o"),
                };
                File.WriteAllText(statusPath, JsonUtility.ToJson(status, true), new UTF8Encoding(false));
                if (!success) throw new InvalidOperationException("Heartisland Agent build failed: " + summary.result);
            }
            catch (Exception exception)
            {
                if (!File.Exists(statusPath))
                    File.WriteAllText(statusPath, JsonUtility.ToJson(new BuildStatus
                    {
                        schemaVersion = "heartisland-agent-build-1.0",
                        status = "failed",
                        unityVersion = Application.unityVersion,
                        sourceScene = ScenePath,
                        startedAtUtc = started.ToString("o"),
                        finishedAtUtc = DateTime.UtcNow.ToString("o"),
                        failure = exception.ToString(),
                    }, true), new UTF8Encoding(false));
                Debug.LogException(exception);
                throw;
            }
        }

        [Serializable]
        private sealed class BuildStatus
        {
            public string schemaVersion;
            public string status;
            public string unityVersion;
            public string sourceScene;
            public string executable;
            public ulong outputBytes;
            public int warningCount;
            public int errorCount;
            public string sourceModelSha256;
            public float modelGoldenMaxAbsError;
            public string startedAtUtc;
            public string finishedAtUtc;
            public string failure;
        }
    }
}
