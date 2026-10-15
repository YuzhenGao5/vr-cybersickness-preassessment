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
    public static class NormalHmdDemoBuilder
    {
        private const string ScenePath = "Assets/Scenes/NormalHmdHorizonGarden.unity";

        public static void BuildWindows()
        {
            DateTime started = DateTime.UtcNow;
            string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
            string repositoryRoot = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "..", ".."));
            string buildDirectory = Environment.GetEnvironmentVariable("UNITY_NORMAL_HMD_BUILD_DIR");
            if (string.IsNullOrWhiteSpace(buildDirectory))
            {
                buildDirectory = Path.Combine(repositoryRoot, "demo", "UnityNormalHmdAgentDemoBuild");
            }
            buildDirectory = Path.GetFullPath(buildDirectory);
            Directory.CreateDirectory(buildDirectory);

            // Refuse to package an absent, stale, synthetic, or numerically
            // incompatible export. LoadVerified performs both model and RNG goldens.
            NormalHmdLstmRuntime verified = NormalHmdLstmRuntime.LoadVerified();

            Directory.CreateDirectory(Path.Combine(Application.dataPath, "Scenes"));
            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            GameObject bootstrap = new GameObject("Normal HMD Agent Demo Bootstrap");
            bootstrap.AddComponent<NormalHmdDemoBootstrap>();
            EditorSceneManager.SaveScene(scene, ScenePath);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();

            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
            PlayerSettings.companyName = "Normal HMD Research Prototype";
            PlayerSettings.productName = "Normal HMD Agent — EyeNavGS 60 Second Demo";
            PlayerSettings.defaultScreenWidth = 1280;
            PlayerSettings.defaultScreenHeight = 720;
            PlayerSettings.fullScreenMode = FullScreenMode.Windowed;
            PlayerSettings.resizableWindow = true;
            PlayerSettings.runInBackground = true;
            PlayerSettings.colorSpace = ColorSpace.Linear;
            PlayerSettings.MTRendering = true;

            string executablePath = Path.Combine(buildDirectory, "NormalHmdAgentDemo.exe");
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
            string statusPath = Path.Combine(repositoryRoot, "results", "normal_hmd_unity_build_status.json");
            Directory.CreateDirectory(Path.GetDirectoryName(statusPath));
            File.WriteAllText(statusPath, JsonUtility.ToJson(status, true));

            if (!success)
            {
                throw new InvalidOperationException(
                    "Normal HMD Unity build failed: " + report.summary.result +
                    " errors=" + report.summary.totalErrors);
            }
            Debug.Log("NORMAL_HMD_BUILD_OK executable=" + executablePath);
        }

        [Serializable]
        private sealed class BuildStatus
        {
            public string status;
            public string unityVersion;
            public string scene;
            public string projectRoot;
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
