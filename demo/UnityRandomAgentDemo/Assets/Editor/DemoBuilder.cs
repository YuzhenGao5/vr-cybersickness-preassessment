using System;
using System.IO;
using RandomAgentDemo;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace RandomAgentDemoEditor
{
    public static class DemoBuilder
    {
        private const string ScenePath = "Assets/Scenes/AuroraMotionCourtyard.unity";

        public static void BuildWindows()
        {
            DateTime started = DateTime.UtcNow;
            string projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
            string repositoryRoot = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "..", ".."));
            string buildDirectory = Environment.GetEnvironmentVariable("UNITY_RANDOM_AGENT_BUILD_DIR");
            if (string.IsNullOrWhiteSpace(buildDirectory))
            {
                buildDirectory = Path.Combine(repositoryRoot, "demo", "UnityRandomAgentDemoBuild");
            }
            buildDirectory = Path.GetFullPath(buildDirectory);
            Directory.CreateDirectory(buildDirectory);

            EyeNavLstmRuntime verified = EyeNavLstmRuntime.LoadVerified();
            if (verified.SelfTestMaxAbsError > 0.0002f)
            {
                throw new InvalidOperationException("Model parity validation failed before build.");
            }

            Directory.CreateDirectory(Path.Combine(Application.dataPath, "Scenes"));
            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            GameObject bootstrap = new GameObject("Random Agent Demo Bootstrap");
            bootstrap.AddComponent<DemoSceneBootstrap>();
            EditorSceneManager.SaveScene(scene, ScenePath);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();

            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
            PlayerSettings.companyName = "OpenAI Research Prototype";
            PlayerSettings.productName = "EyeNavGS Random Human Motion Demo";
            PlayerSettings.defaultScreenWidth = 1280;
            PlayerSettings.defaultScreenHeight = 720;
            PlayerSettings.fullScreenMode = FullScreenMode.Windowed;
            PlayerSettings.resizableWindow = true;
            PlayerSettings.runInBackground = true;
            PlayerSettings.colorSpace = ColorSpace.Linear;
            PlayerSettings.MTRendering = true;

            string executablePath = Path.Combine(buildDirectory, "RandomHumanAgentDemo.exe");
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
                sourceModelSha256 = GeneratedModelIntegrity.SourceModelSha256,
                exportedModelSha256 = verified.ExportSha256,
                modelParityMaxAbsError = verified.SelfTestMaxAbsError,
                startedAtUtc = started.ToString("O"),
                finishedAtUtc = DateTime.UtcNow.ToString("O"),
            };
            string statusPath = Path.Combine(repositoryRoot, "results", "unity_random_agent_build_status.json");
            Directory.CreateDirectory(Path.GetDirectoryName(statusPath));
            File.WriteAllText(statusPath, JsonUtility.ToJson(status, true));

            if (!success)
            {
                throw new InvalidOperationException(
                    "Unity build failed: " + report.summary.result +
                    " errors=" + report.summary.totalErrors);
            }
            Debug.Log("RANDOM_AGENT_BUILD_OK executable=" + executablePath);
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
            public float modelParityMaxAbsError;
            public string startedAtUtc;
            public string finishedAtUtc;
        }
    }
}
