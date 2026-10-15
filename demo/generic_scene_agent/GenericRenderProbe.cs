using System;
using System.Collections;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.Rendering;

namespace GenericSceneAgent
{
    [Serializable]
    internal sealed class GenericCameraAudit
    {
        public string status;
        public string unityVersion;
        public string graphicsDeviceType;
        public string renderPipeline;
        public int screenWidth;
        public int screenHeight;
        public int cameraCount;
        public int enabledCameraCount;
        public int enabledAudioListenerCount;
        public int disabledSourceCameraCount;
        public string activeCameraName;
        public float activeCameraFieldOfView;
        public int activeCameraCullingMask;
        public string screenshotPath;
        public long screenshotBytes;
        public int captureCandidates;
        public float selectedFrameScore;
        public string capturedAtUtc;
    }

    /// <summary>Built-player rendering evidence; independent of gameplay acceptance.</summary>
    public sealed class GenericRenderProbe : MonoBehaviour
    {
        private string screenshotPath;
        private string auditPath;
        private int disabledSourceCameraCount;

        public void Configure(string screenshot, string audit, int disabledCameras)
        {
            screenshotPath = screenshot;
            auditPath = audit;
            disabledSourceCameraCount = disabledCameras;
        }

        private IEnumerator Start()
        {
            if (string.IsNullOrWhiteSpace(screenshotPath)) yield break;
            screenshotPath = Path.GetFullPath(screenshotPath);
            auditPath = string.IsNullOrWhiteSpace(auditPath)
                ? Path.ChangeExtension(screenshotPath, ".camera_audit.json")
                : Path.GetFullPath(auditPath);
            Directory.CreateDirectory(Path.GetDirectoryName(screenshotPath));
            Directory.CreateDirectory(Path.GetDirectoryName(auditPath));
            // Sample multiple first-person frames.  Choosing the frame with the
            // most visual information avoids accepting a capture taken while the
            // camera happens to be centimetres from a pillar or wall.
            byte[] selectedPng = null;
            float selectedScore = float.NegativeInfinity;
            const int candidateCount = 3;
            for (int candidate = 0; candidate < candidateCount; candidate++)
            {
                int waitFrames = candidate == 0 ? 18 : 28;
                for (int index = 0; index < waitFrames; index++) yield return new WaitForEndOfFrame();
                Texture2D frame = new Texture2D(Screen.width, Screen.height, TextureFormat.RGB24, false);
                frame.ReadPixels(new Rect(0f, 0f, Screen.width, Screen.height), 0, 0, false);
                frame.Apply(false, false);
                float score = InformationScore(frame);
                if (score > selectedScore)
                {
                    selectedScore = score;
                    selectedPng = frame.EncodeToPNG();
                }
                Destroy(frame);
            }
            if (selectedPng != null) File.WriteAllBytes(screenshotPath, selectedPng);
            Camera[] cameras = FindObjectsByType<Camera>(FindObjectsInactive.Include, FindObjectsSortMode.None);
            Camera active = cameras.FirstOrDefault(item => item.enabled && item.gameObject.activeInHierarchy);
            AudioListener[] listeners = FindObjectsByType<AudioListener>(FindObjectsInactive.Include, FindObjectsSortMode.None);
            RenderPipelineAsset pipeline = GraphicsSettings.currentRenderPipeline;
            GenericCameraAudit audit = new GenericCameraAudit
            {
                status = File.Exists(screenshotPath) && new FileInfo(screenshotPath).Length > 0 ? "captured" : "failed",
                unityVersion = Application.unityVersion,
                graphicsDeviceType = SystemInfo.graphicsDeviceType.ToString(),
                renderPipeline = pipeline == null ? "Built-in Render Pipeline" : pipeline.GetType().FullName,
                screenWidth = Screen.width,
                screenHeight = Screen.height,
                cameraCount = cameras.Length,
                enabledCameraCount = cameras.Count(item => item.enabled && item.gameObject.activeInHierarchy),
                enabledAudioListenerCount = listeners.Count(item => item.enabled && item.gameObject.activeInHierarchy),
                disabledSourceCameraCount = disabledSourceCameraCount,
                activeCameraName = active == null ? "" : active.name,
                activeCameraFieldOfView = active == null ? 0f : active.fieldOfView,
                activeCameraCullingMask = active == null ? 0 : active.cullingMask,
                screenshotPath = screenshotPath,
                screenshotBytes = File.Exists(screenshotPath) ? new FileInfo(screenshotPath).Length : 0,
                captureCandidates = candidateCount,
                selectedFrameScore = selectedScore,
                capturedAtUtc = DateTime.UtcNow.ToString("O")
            };
            File.WriteAllText(auditPath, JsonUtility.ToJson(audit, true), new UTF8Encoding(false));
        }

        private static float InformationScore(Texture2D frame)
        {
            Color32[] pixels = frame.GetPixels32();
            int stride = Mathf.Max(1, pixels.Length / 7000);
            double luminanceSum = 0.0;
            double luminanceSquareSum = 0.0;
            int count = 0;
            Dictionary<int, int> bins = new Dictionary<int, int>();
            int dominant = 0;
            for (int index = 0; index < pixels.Length; index += stride)
            {
                Color32 pixel = pixels[index];
                double luminance = 0.2126 * pixel.r + 0.7152 * pixel.g + 0.0722 * pixel.b;
                luminanceSum += luminance;
                luminanceSquareSum += luminance * luminance;
                int key = (pixel.r >> 5) << 6 | (pixel.g >> 5) << 3 | (pixel.b >> 5);
                int frequency;
                bins.TryGetValue(key, out frequency);
                frequency++;
                bins[key] = frequency;
                dominant = Mathf.Max(dominant, frequency);
                count++;
            }
            if (count == 0) return float.NegativeInfinity;
            double mean = luminanceSum / count;
            double variance = Math.Max(0.0, luminanceSquareSum / count - mean * mean);
            float dominantRatio = (float)dominant / count;
            return (float)Math.Sqrt(variance) + bins.Count * 0.08f - dominantRatio * 95f;
        }
    }
}
