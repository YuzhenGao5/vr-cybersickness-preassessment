using System;
using System.IO;
using NormalHmdAgentDemo;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace NormalHmdAgentDemoEditor
{
    /// <summary>
    /// Creates the standalone scene-aware exploration demo without changing or
    /// replacing the original Normal HMD demo build.
    /// </summary>
    public static class SceneAwareDemoBuilder
    {
        private const string ScenePath = "Assets/Scenes/SceneAwareAgentDemo.unity";

        public static void BuildWindows()
        {
            DateTime started = DateTime.UtcNow;
            string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
            string repositoryRoot = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "..", ".."));
            string buildDirectory = Environment.GetEnvironmentVariable("UNITY_SCENE_AWARE_BUILD_DIR");
            if (string.IsNullOrWhiteSpace(buildDirectory))
            {
                buildDirectory = Path.Combine(repositoryRoot, "demo", "UnitySceneAwareAgentDemoBuild");
            }
            buildDirectory = Path.GetFullPath(buildDirectory);
            Directory.CreateDirectory(buildDirectory);

            // Fail the package step if the trained model export is absent, stale,
            // synthetic, or numerically incompatible with the Unity runtime.
            NormalHmdLstmRuntime verified = NormalHmdLstmRuntime.LoadVerified();

            Directory.CreateDirectory(Path.Combine(Application.dataPath, "Scenes"));
            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            GameObject bootstrap = new GameObject("Scene-Aware Agent Demo Bootstrap");
            bootstrap.AddComponent<SceneAwareDemoBootstrap>();
            EditorSceneManager.SaveScene(scene, ScenePath);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();

            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
            PlayerSettings.companyName = "Cybersickness Preassessment Research Prototype";
            PlayerSettings.productName = "Scene-Aware Normal HMD Agent Demo";
            PlayerSettings.defaultScreenWidth = 1440;
            PlayerSettings.defaultScreenHeight = 810;
            PlayerSettings.fullScreenMode = FullScreenMode.Windowed;
            PlayerSettings.resizableWindow = true;
            PlayerSettings.runInBackground = true;
            PlayerSettings.colorSpace = ColorSpace.Linear;
            PlayerSettings.MTRendering = true;

            string executablePath = Path.Combine(buildDirectory, "SceneAwareAgentDemo.exe");
            BuildPlayerOptions options = new BuildPlayerOptions
            {
                scenes = new[] { ScenePath },
                locationPathName = executablePath,
                target = BuildTarget.StandaloneWindows64,
                options = BuildOptions.None,
            };
            BuildReport report = BuildPipeline.BuildPlayer(options);
            bool success = report.summary.result == BuildResult.Succeeded;

            BuildStatus status = new BuildStatus
            {
                status = success ? "success" : "failed",
                unityVersion = Application.unityVersion,
                scene = ScenePath,
                projectRoot = projectRoot,
                buildDirectory = buildDirectory,
                executable = executablePath,
                outputBytes = (long)report.summary.totalSize,
                buildSeconds = (float)report.summary.totalTime.TotalSeconds,
                warningCount = report.summary.totalWarnings,
                errorCount = report.summary.totalErrors,
                sourceModelSha256 = verified.Data.sourceModelSha256,
                exportedModelSha256 = verified.ExportSha256,
                modelGoldenMaxAbsError = verified.GoldenMaxAbsError,
                innovationGoldenMaxAbsError = verified.InnovationGoldenMaxAbsError,
                hasCombinedStochasticGolden = verified.HasCombinedStochasticGolden,
                combinedStochasticGoldenMaxAbsError = verified.CombinedStochasticGoldenMaxAbsError,
                startedAtUtc = started.ToString("O"),
                finishedAtUtc = DateTime.UtcNow.ToString("O"),
            };
            string statusPath = Path.Combine(repositoryRoot, "results", "scene_aware_agent_build_status.json");
            Directory.CreateDirectory(Path.GetDirectoryName(statusPath));
            File.WriteAllText(statusPath, JsonUtility.ToJson(status, true));

            if (!success)
            {
                throw new InvalidOperationException(
                    "Scene-aware Unity build failed: " + report.summary.result +
                    " errors=" + report.summary.totalErrors);
            }
            Debug.Log("SCENE_AWARE_BUILD_OK executable=" + executablePath);
        }

        [Serializable]
        private sealed class BuildStatus
        {
            public string status;
            public string unityVersion;
            public string scene;
            public string projectRoot;
            public string buildDirectory;
            public string executable;
            public long outputBytes;
            public float buildSeconds;
            public int warningCount;
            public int errorCount;
            public string sourceModelSha256;
            public string exportedModelSha256;
            public float modelGoldenMaxAbsError;
            public float innovationGoldenMaxAbsError;
            public bool hasCombinedStochasticGolden;
            public float combinedStochasticGoldenMaxAbsError;
            public string startedAtUtc;
            public string finishedAtUtc;
        }
    }
}
