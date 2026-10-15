using UnityEngine;

namespace NormalHmdAgentDemo
{
    public sealed class NormalHmdDemoBootstrap : MonoBehaviour
    {
        private static readonly Color DeepSky = new Color(0.055f, 0.11f, 0.17f);
        private static readonly Color Teal = new Color(0.14f, 0.91f, 0.79f);
        private static readonly Color Blue = new Color(0.20f, 0.58f, 0.96f);
        private static readonly Color Amber = new Color(1.00f, 0.70f, 0.26f);

        private void Awake()
        {
            Application.targetFrameRate = 60;
            QualitySettings.vSyncCount = 1;
            BuildLighting();
            BuildHorizonGarden();
            Camera observer = BuildObserverCamera();
            GameObject participant = new GameObject("ParticipantRoot (loading model)");
            NormalHmdAgentController controller = participant.AddComponent<NormalHmdAgentController>();
            controller.Configure(observer);
        }

        private static void BuildLighting()
        {
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Trilight;
            RenderSettings.ambientSkyColor = new Color(0.36f, 0.52f, 0.62f);
            RenderSettings.ambientEquatorColor = new Color(0.18f, 0.29f, 0.34f);
            RenderSettings.ambientGroundColor = new Color(0.055f, 0.085f, 0.09f);
            RenderSettings.fog = true;
            RenderSettings.fogMode = FogMode.ExponentialSquared;
            RenderSettings.fogDensity = 0.0045f;
            RenderSettings.fogColor = new Color(0.14f, 0.23f, 0.28f);

            GameObject sunObject = new GameObject("Soft Directional Sun");
            Light sun = sunObject.AddComponent<Light>();
            sun.type = LightType.Directional;
            sun.color = new Color(0.92f, 0.97f, 1f);
            sun.intensity = 1.1f;
            sun.shadows = LightShadows.Soft;
            sunObject.transform.rotation = Quaternion.Euler(48f, -32f, 0f);

            CreatePointLight("Teal Wayfinding Light", new Vector3(-8f, 4f, -8f), Teal);
            CreatePointLight("Amber Wayfinding Light", new Vector3(8f, 4f, 8f), Amber);
        }

        private static Camera BuildObserverCamera()
        {
            GameObject cameraObject = new GameObject("ObserverCamera (External, Not AI Controlled)");
            Camera camera = cameraObject.AddComponent<Camera>();
            camera.fieldOfView = 55f;
            camera.nearClipPlane = 0.08f;
            camera.farClipPlane = 320f;
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = DeepSky;
            cameraObject.transform.position = new Vector3(0f, 6.5f, -11f);
            cameraObject.transform.LookAt(new Vector3(0f, 1.3f, 0f));
            cameraObject.AddComponent<AudioListener>();
            return camera;
        }

        private static void BuildHorizonGarden()
        {
            Material grass = MakeMaterial("Calm Ground", new Color(0.075f, 0.16f, 0.15f), 0f, 0.3f);
            Material path = MakeMaterial("Stable Walkway", new Color(0.22f, 0.31f, 0.33f), 0.02f, 0.46f);
            Material dark = MakeMaterial("Garden Structure", new Color(0.08f, 0.14f, 0.17f), 0.18f, 0.48f);
            Material teal = MakeMaterial("Teal Landmark", Teal, 0.02f, 0.7f, Teal * 0.23f);
            Material blue = MakeMaterial("Blue Landmark", Blue, 0.04f, 0.68f, Blue * 0.18f);
            Material amber = MakeMaterial("Amber Landmark", Amber, 0.03f, 0.62f, Amber * 0.16f);

            Primitive("Horizon Garden Ground", PrimitiveType.Plane, Vector3.zero, new Vector3(14f, 1f, 14f), grass);
            Primitive("North South Path", PrimitiveType.Cube, new Vector3(0f, 0.025f, 0f), new Vector3(3.4f, 0.05f, 105f), path);
            Primitive("East West Path", PrimitiveType.Cube, new Vector3(0f, 0.028f, 0f), new Vector3(105f, 0.055f, 3.4f), path);

            // Repeated ground markers and vertical landmarks provide a strong,
            // stationary horizon reference without attaching anything to the HMD.
            for (int index = -10; index <= 10; index++)
            {
                if (index == 0) continue;
                float coordinate = index * 5f;
                Material accent = index % 2 == 0 ? teal : blue;
                Primitive("Path Tick X " + index, PrimitiveType.Cube,
                    new Vector3(coordinate, 0.065f, 0f), new Vector3(0.08f, 0.03f, 2.5f), accent);
                Primitive("Path Tick Z " + index, PrimitiveType.Cube,
                    new Vector3(0f, 0.068f, coordinate), new Vector3(2.5f, 0.03f, 0.08f), accent);
            }

            for (int index = 0; index < 16; index++)
            {
                float angle = index * Mathf.PI * 2f / 16f;
                float radius = 22f;
                Vector3 point = new Vector3(Mathf.Sin(angle) * radius, 0f, Mathf.Cos(angle) * radius);
                Material accent = index % 3 == 0 ? amber : index % 3 == 1 ? teal : blue;
                CreateLandmark("Horizon Landmark " + (index + 1), point, angle * Mathf.Rad2Deg, dark, accent);
            }

            CreateGateway("North Gateway", new Vector3(0f, 0f, 13f), Quaternion.identity, dark, teal);
            CreateGateway("South Gateway", new Vector3(0f, 0f, -13f), Quaternion.identity, dark, blue);
            CreateGateway("East Gateway", new Vector3(13f, 0f, 0f), Quaternion.Euler(0f, 90f, 0f), dark, amber);
            CreateGateway("West Gateway", new Vector3(-13f, 0f, 0f), Quaternion.Euler(0f, 90f, 0f), dark, teal);

            Primitive("Centre Plinth", PrimitiveType.Cylinder, new Vector3(0f, 0.25f, 0f), new Vector3(1.3f, 0.25f, 1.3f), dark);
            GameObject beacon = Primitive("Slow Orientation Beacon", PrimitiveType.Cube,
                new Vector3(0f, 2.1f, 0f), new Vector3(0.18f, 3.2f, 0.18f), teal);
            beacon.AddComponent<SlowBeacon>();
        }

        private static void CreateGateway(
            string name, Vector3 position, Quaternion rotation, Material structure, Material accent)
        {
            GameObject root = new GameObject(name);
            root.transform.SetPositionAndRotation(position, rotation);
            PrimitiveChild("Left Upright", PrimitiveType.Cube, root.transform,
                new Vector3(-1.8f, 1.6f, 0f), new Vector3(0.28f, 3.2f, 0.35f), structure);
            PrimitiveChild("Right Upright", PrimitiveType.Cube, root.transform,
                new Vector3(1.8f, 1.6f, 0f), new Vector3(0.28f, 3.2f, 0.35f), structure);
            PrimitiveChild("Level Header", PrimitiveType.Cube, root.transform,
                new Vector3(0f, 3.15f, 0f), new Vector3(3.9f, 0.25f, 0.36f), accent);
        }

        private static void CreateLandmark(
            string name, Vector3 position, float yaw, Material structure, Material accent)
        {
            GameObject root = new GameObject(name);
            root.transform.SetPositionAndRotation(position, Quaternion.Euler(0f, yaw, 0f));
            PrimitiveChild("Trunk", PrimitiveType.Cylinder, root.transform,
                new Vector3(0f, 1.25f, 0f), new Vector3(0.22f, 1.25f, 0.22f), structure);
            PrimitiveChild("Level Crown", PrimitiveType.Cube, root.transform,
                new Vector3(0f, 2.6f, 0f), new Vector3(1.7f, 0.12f, 0.32f), accent);
        }

        private static void CreatePointLight(string name, Vector3 position, Color color)
        {
            GameObject lightObject = new GameObject(name);
            lightObject.transform.position = position;
            Light light = lightObject.AddComponent<Light>();
            light.type = LightType.Point;
            light.range = 18f;
            light.intensity = 3f;
            light.color = color;
        }

        internal static Material MakeMaterial(
            string name, Color color, float metallic = 0f, float smoothness = 0.5f,
            Color emission = default(Color))
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Lit");
            if (shader == null) shader = Shader.Find("Standard");
            if (shader == null) shader = Shader.Find("Sprites/Default");
            Material material = new Material(shader) { name = name, color = color };
            if (material.HasProperty("_BaseColor")) material.SetColor("_BaseColor", color);
            if (material.HasProperty("_Metallic")) material.SetFloat("_Metallic", metallic);
            if (material.HasProperty("_Smoothness")) material.SetFloat("_Smoothness", smoothness);
            if (emission.maxColorComponent > 0f)
            {
                material.EnableKeyword("_EMISSION");
                if (material.HasProperty("_EmissionColor")) material.SetColor("_EmissionColor", emission);
            }
            return material;
        }

        private static GameObject Primitive(
            string name, PrimitiveType type, Vector3 position, Vector3 scale, Material material)
        {
            GameObject instance = GameObject.CreatePrimitive(type);
            instance.name = name;
            instance.transform.position = position;
            instance.transform.localScale = scale;
            instance.GetComponent<Renderer>().sharedMaterial = material;
            Collider collider = instance.GetComponent<Collider>();
            if (collider != null) Object.Destroy(collider);
            return instance;
        }

        private static GameObject PrimitiveChild(
            string name, PrimitiveType type, Transform parent, Vector3 localPosition,
            Vector3 localScale, Material material)
        {
            GameObject instance = Primitive(name, type, Vector3.zero, localScale, material);
            instance.transform.SetParent(parent, false);
            instance.transform.localPosition = localPosition;
            return instance;
        }
    }

    public sealed class SlowBeacon : MonoBehaviour
    {
        private void Update()
        {
            transform.Rotate(Vector3.up, 8f * Time.deltaTime, Space.World);
        }
    }
}
