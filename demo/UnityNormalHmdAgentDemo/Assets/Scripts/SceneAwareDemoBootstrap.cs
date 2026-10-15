using System;
using UnityEngine;

namespace NormalHmdAgentDemo
{
    /// <summary>
    /// Entry point for the separate Scene-Aware Agent demo build.
    /// The scene is intentionally constructed from code so every visual object
    /// can be paired with an explicit semantic contract in the explorer.
    /// </summary>
    public sealed class SceneAwareDemoBootstrap : MonoBehaviour
    {
        private void Awake()
        {
            bool automated = HasArgument("--scene-aware-test");
            Application.targetFrameRate = automated ? 240 : 60;
            QualitySettings.vSyncCount = automated ? 0 : 1;
            ConfigureLighting();
            Camera observer = BuildObserverCamera();

            GameObject participant = new GameObject("Scene-Aware Participant Root");
            SceneAwareExplorerController explorer = participant.AddComponent<SceneAwareExplorerController>();
            NormalHmdAgentController controller = participant.AddComponent<NormalHmdAgentController>();
            explorer.Configure(controller, observer);
            controller.ConfigureMotionDriver(explorer);
            controller.Configure(observer);
        }

        private static void ConfigureLighting()
        {
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Trilight;
            RenderSettings.ambientSkyColor = new Color(0.32f, 0.45f, 0.58f);
            RenderSettings.ambientEquatorColor = new Color(0.15f, 0.23f, 0.30f);
            RenderSettings.ambientGroundColor = new Color(0.035f, 0.052f, 0.068f);
            RenderSettings.fog = true;
            RenderSettings.fogMode = FogMode.ExponentialSquared;
            RenderSettings.fogDensity = 0.008f;
            RenderSettings.fogColor = new Color(0.09f, 0.15f, 0.21f);

            GameObject sunObject = new GameObject("Scene-Aware Soft Sun");
            Light sun = sunObject.AddComponent<Light>();
            sun.type = LightType.Directional;
            sun.color = new Color(0.90f, 0.95f, 1f);
            sun.intensity = 1.15f;
            sun.shadows = LightShadows.Soft;
            sunObject.transform.rotation = Quaternion.Euler(52f, -34f, 0f);

            GameObject fillObject = new GameObject("Scene-Aware Teal Fill");
            fillObject.transform.position = new Vector3(-4f, 5f, -2f);
            Light fill = fillObject.AddComponent<Light>();
            fill.type = LightType.Point;
            fill.color = new Color(0.14f, 0.78f, 0.76f);
            fill.range = 22f;
            fill.intensity = 3.2f;
        }

        private static Camera BuildObserverCamera()
        {
            GameObject cameraObject = new GameObject("Observer Camera (not AI controlled)");
            Camera camera = cameraObject.AddComponent<Camera>();
            camera.fieldOfView = 54f;
            camera.nearClipPlane = 0.08f;
            camera.farClipPlane = 180f;
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = new Color(0.035f, 0.07f, 0.11f);
            cameraObject.transform.position = new Vector3(12.5f, 10f, -14.5f);
            cameraObject.transform.LookAt(new Vector3(0f, 1.0f, 0.8f));
            cameraObject.AddComponent<AudioListener>();
            return camera;
        }

        private static bool HasArgument(string exact)
        {
            foreach (string argument in Environment.GetCommandLineArgs())
                if (string.Equals(argument, exact, StringComparison.OrdinalIgnoreCase)) return true;
            return false;
        }
    }
}
