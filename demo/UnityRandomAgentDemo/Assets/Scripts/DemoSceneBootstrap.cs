using UnityEngine;

namespace RandomAgentDemo
{
    public sealed class DemoSceneBootstrap : MonoBehaviour
    {
        private static readonly Color Navy = new Color(0.035f, 0.055f, 0.11f);
        private static readonly Color Cyan = new Color(0.13f, 0.88f, 0.86f);
        private static readonly Color Coral = new Color(1.00f, 0.39f, 0.36f);
        private static readonly Color Gold = new Color(1.00f, 0.76f, 0.27f);
        private static readonly Color Mint = new Color(0.30f, 0.82f, 0.58f);

        private void Awake()
        {
            Application.targetFrameRate = 60;
            QualitySettings.vSyncCount = 1;
            BuildLighting();
            BuildCourtyard();
            Camera mainCamera = BuildCamera();
            GameObject controllerObject = new GameObject("EyeNavGS Simulated Participant");
            RandomAgentController controller = controllerObject.AddComponent<RandomAgentController>();
            controller.Configure(mainCamera, 8.6f);
        }

        private static void BuildLighting()
        {
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Trilight;
            RenderSettings.ambientSkyColor = new Color(0.28f, 0.37f, 0.55f);
            RenderSettings.ambientEquatorColor = new Color(0.12f, 0.19f, 0.30f);
            RenderSettings.ambientGroundColor = new Color(0.035f, 0.055f, 0.09f);
            RenderSettings.fog = true;
            RenderSettings.fogMode = FogMode.ExponentialSquared;
            RenderSettings.fogDensity = 0.008f;
            RenderSettings.fogColor = new Color(0.055f, 0.09f, 0.16f);

            GameObject sunObject = new GameObject("Sun");
            Light sun = sunObject.AddComponent<Light>();
            sun.type = LightType.Directional;
            sun.color = new Color(0.88f, 0.94f, 1.0f);
            sun.intensity = 1.15f;
            sun.shadows = LightShadows.Soft;
            sunObject.transform.rotation = Quaternion.Euler(42f, -32f, 0f);

            CreatePointLight("Cyan Plaza Light", new Vector3(-5f, 3.2f, 2.8f), Cyan);
            CreatePointLight("Coral Plaza Light", new Vector3(5f, 3.2f, -2.8f), Coral);
        }

        private static Camera BuildCamera()
        {
            GameObject cameraObject = new GameObject("Main Camera");
            cameraObject.tag = "MainCamera";
            Camera camera = cameraObject.AddComponent<Camera>();
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = Navy;
            camera.fieldOfView = 58f;
            camera.nearClipPlane = 0.08f;
            camera.farClipPlane = 120f;
            cameraObject.AddComponent<AudioListener>();
            cameraObject.transform.position = new Vector3(0f, 7.5f, -10f);
            cameraObject.transform.LookAt(new Vector3(0f, 1.1f, 0f));
            return camera;
        }

        private static void BuildCourtyard()
        {
            Material ground = MakeMaterial("Midnight Ground", new Color(0.055f, 0.08f, 0.13f), 0.15f, 0.68f);
            Material path = MakeMaterial("Slate Path", new Color(0.16f, 0.22f, 0.31f), 0.05f, 0.52f);
            Material cyan = MakeMaterial("Cyan Glass", Cyan, 0.1f, 0.78f, Cyan * 0.45f);
            Material coral = MakeMaterial("Coral", Coral, 0.05f, 0.65f, Coral * 0.28f);
            Material gold = MakeMaterial("Gold", Gold, 0.25f, 0.72f, Gold * 0.18f);
            Material mint = MakeMaterial("Mint", Mint, 0.0f, 0.45f);
            Material dark = MakeMaterial("Dark Structure", new Color(0.08f, 0.105f, 0.16f), 0.35f, 0.72f);

            GameObject floor = Primitive("Courtyard Floor", PrimitiveType.Plane, Vector3.zero, new Vector3(2.4f, 1f, 2.4f), ground);
            floor.isStatic = true;
            Primitive("North-South Walkway", PrimitiveType.Cube, new Vector3(0f, 0.035f, 0f), new Vector3(2.35f, 0.07f, 20f), path);
            Primitive("East-West Walkway", PrimitiveType.Cube, new Vector3(0f, 0.038f, 0f), new Vector3(20f, 0.075f, 2.35f), path);

            for (int index = -4; index <= 4; index++)
            {
                if (index == 0)
                {
                    continue;
                }
                float coordinate = index * 2f;
                Primitive("Path Marker X " + index, PrimitiveType.Cube, new Vector3(coordinate, 0.085f, 0f), new Vector3(0.045f, 0.025f, 1.7f), index % 2 == 0 ? cyan : gold);
                Primitive("Path Marker Z " + index, PrimitiveType.Cube, new Vector3(0f, 0.088f, coordinate), new Vector3(1.7f, 0.025f, 0.045f), index % 2 == 0 ? coral : cyan);
            }

            // A central kinetic sculpture makes it obvious that this is a live Unity scene.
            GameObject sculptureRoot = new GameObject("Kinetic Lattice");
            sculptureRoot.transform.position = new Vector3(0f, 1.75f, 0f);
            sculptureRoot.AddComponent<KineticSculpture>();
            PrimitiveChild("Lattice Ring A", PrimitiveType.Cylinder, sculptureRoot.transform, Vector3.zero, new Vector3(1.25f, 0.035f, 1.25f), cyan, Quaternion.Euler(90f, 0f, 0f));
            PrimitiveChild("Lattice Ring B", PrimitiveType.Cylinder, sculptureRoot.transform, Vector3.zero, new Vector3(0.95f, 0.028f, 0.95f), coral, Quaternion.Euler(0f, 0f, 90f));
            PrimitiveChild("Lattice Core", PrimitiveType.Sphere, sculptureRoot.transform, Vector3.zero, Vector3.one * 0.38f, gold, Quaternion.identity);
            Primitive("Lattice Pedestal", PrimitiveType.Cylinder, new Vector3(0f, 0.45f, 0f), new Vector3(0.65f, 0.45f, 0.65f), dark);

            CreatePortal(new Vector3(-7.6f, 0f, -7.6f), Quaternion.Euler(0f, 45f, 0f), cyan, dark);
            CreatePortal(new Vector3(7.6f, 0f, 7.6f), Quaternion.Euler(0f, 225f, 0f), coral, dark);
            CreatePortal(new Vector3(-7.6f, 0f, 7.6f), Quaternion.Euler(0f, 135f, 0f), gold, dark);
            CreatePortal(new Vector3(7.6f, 0f, -7.6f), Quaternion.Euler(0f, -45f, 0f), mint, dark);

            Vector3[] gardenPoints =
            {
                new Vector3(-6f, 0f, -3.8f), new Vector3(-6.8f, 0f, 3.8f),
                new Vector3(6f, 0f, 3.8f), new Vector3(6.8f, 0f, -3.8f),
                new Vector3(-3.8f, 0f, -6.2f), new Vector3(3.8f, 0f, 6.2f),
                new Vector3(3.8f, 0f, -6.5f), new Vector3(-3.8f, 0f, 6.5f),
            };
            for (int index = 0; index < gardenPoints.Length; index++)
            {
                Material accent = index % 3 == 0 ? cyan : index % 3 == 1 ? coral : mint;
                CreateTree("Luminous Tree " + (index + 1), gardenPoints[index], accent, dark);
            }

            for (int index = 0; index < 20; index++)
            {
                float angle = index * Mathf.PI * 2f / 20f;
                Vector3 point = new Vector3(Mathf.Cos(angle) * 10.5f, 0.32f, Mathf.Sin(angle) * 10.5f);
                Primitive("Boundary Light " + index, PrimitiveType.Cylinder, point, new Vector3(0.075f, 0.32f, 0.075f), index % 2 == 0 ? cyan : coral);
            }

            // Low perimeter walls are visual only; motion reflection is logged separately.
            Primitive("North Wall", PrimitiveType.Cube, new Vector3(0f, 0.55f, 11.5f), new Vector3(23f, 1.1f, 0.25f), dark);
            Primitive("South Wall", PrimitiveType.Cube, new Vector3(0f, 0.55f, -11.5f), new Vector3(23f, 1.1f, 0.25f), dark);
            Primitive("East Wall", PrimitiveType.Cube, new Vector3(11.5f, 0.55f, 0f), new Vector3(0.25f, 1.1f, 23f), dark);
            Primitive("West Wall", PrimitiveType.Cube, new Vector3(-11.5f, 0.55f, 0f), new Vector3(0.25f, 1.1f, 23f), dark);
        }

        private static void CreatePortal(Vector3 position, Quaternion rotation, Material accent, Material dark)
        {
            GameObject root = new GameObject("Observation Portal");
            root.transform.SetPositionAndRotation(position, rotation);
            PrimitiveChild("Left Pillar", PrimitiveType.Cube, root.transform, new Vector3(-1.05f, 1.5f, 0f), new Vector3(0.32f, 3f, 0.55f), dark, Quaternion.identity);
            PrimitiveChild("Right Pillar", PrimitiveType.Cube, root.transform, new Vector3(1.05f, 1.5f, 0f), new Vector3(0.32f, 3f, 0.55f), dark, Quaternion.identity);
            PrimitiveChild("Top Beam", PrimitiveType.Cube, root.transform, new Vector3(0f, 3f, 0f), new Vector3(2.42f, 0.32f, 0.55f), accent, Quaternion.identity);
        }

        private static void CreateTree(string name, Vector3 position, Material leaves, Material trunk)
        {
            GameObject root = new GameObject(name);
            root.transform.position = position;
            PrimitiveChild("Trunk", PrimitiveType.Cylinder, root.transform, new Vector3(0f, 0.85f, 0f), new Vector3(0.18f, 0.85f, 0.18f), trunk, Quaternion.identity);
            PrimitiveChild("Crown A", PrimitiveType.Sphere, root.transform, new Vector3(0f, 2.0f, 0f), new Vector3(1.0f, 1.15f, 1.0f), leaves, Quaternion.identity);
            PrimitiveChild("Crown B", PrimitiveType.Sphere, root.transform, new Vector3(0.38f, 2.25f, -0.15f), new Vector3(0.65f, 0.75f, 0.65f), leaves, Quaternion.identity);
        }

        private static void CreatePointLight(string name, Vector3 position, Color color)
        {
            GameObject lightObject = new GameObject(name);
            lightObject.transform.position = position;
            Light light = lightObject.AddComponent<Light>();
            light.type = LightType.Point;
            light.color = color;
            light.intensity = 4.2f;
            light.range = 9f;
            light.shadows = LightShadows.None;
        }

        internal static Material MakeMaterial(string name, Color color, float metallic = 0f, float smoothness = 0.5f, Color emission = default)
        {
            Shader shader = Resources.Load<Shader>("SceneLit");
            if (shader == null)
            {
                throw new System.InvalidOperationException("Embedded RandomAgentDemo/SceneLit shader is missing.");
            }
            Material material = new Material(shader) { name = name, color = color };
            material.SetFloat("_Metallic", metallic);
            material.SetFloat("_Glossiness", smoothness);
            if (emission.maxColorComponent > 0.001f)
            {
                material.EnableKeyword("_EMISSION");
                material.SetColor("_EmissionColor", emission);
            }
            return material;
        }

        private static GameObject Primitive(string name, PrimitiveType type, Vector3 position, Vector3 scale, Material material)
        {
            GameObject instance = GameObject.CreatePrimitive(type);
            instance.name = name;
            instance.transform.position = position;
            instance.transform.localScale = scale;
            Renderer renderer = instance.GetComponent<Renderer>();
            renderer.sharedMaterial = material;
            return instance;
        }

        private static GameObject PrimitiveChild(string name, PrimitiveType type, Transform parent, Vector3 localPosition, Vector3 localScale, Material material, Quaternion localRotation)
        {
            GameObject instance = Primitive(name, type, Vector3.zero, localScale, material);
            instance.transform.SetParent(parent, false);
            instance.transform.localPosition = localPosition;
            instance.transform.localRotation = localRotation;
            return instance;
        }
    }

    internal sealed class KineticSculpture : MonoBehaviour
    {
        private void Update()
        {
            transform.Rotate(new Vector3(11f, 24f, 7f) * Time.deltaTime, Space.Self);
        }
    }
}
