using System.Collections;
using System;
using System.Linq;
using NormalHmdAgentDemo;
using UnityEngine;

namespace GenericSceneAgent
{
    /// <summary>
    /// Runtime injection entry point used only in the isolated staging build.
    /// The original Unity project is never edited.
    /// </summary>
    public static class GenericSceneAgentBootstrap
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Install()
        {
            TextAsset contractAsset = Resources.Load<TextAsset>("AgentSceneContract");
            GenericSceneContract contract = contractAsset == null
                ? null : JsonUtility.FromJson<GenericSceneContract>(contractAsset.text);
            Camera observer = Camera.main;
            MonoBehaviour[] behaviours = UnityEngine.Object.FindObjectsByType<MonoBehaviour>(
                FindObjectsInactive.Include, FindObjectsSortMode.None);
            foreach (MonoBehaviour behaviour in behaviours)
            {
                if (behaviour != null && contract != null && contract.disabledBehaviourTypes != null &&
                    contract.disabledBehaviourTypes.Any(item => string.Equals(
                        item, behaviour.GetType().FullName, StringComparison.Ordinal)))
                    behaviour.enabled = false;
            }
            // Some source projects already contain the Normal HMD demo driver.
            // The generic bridge must own the only motion controller; otherwise
            // both controllers create a first-person camera, advance independent
            // recurrent states and write overlapping participant evidence.
            RemoveExistingMotionControllers();
            Camera[] sourceCameras = UnityEngine.Object.FindObjectsByType<Camera>(
                FindObjectsInactive.Include, FindObjectsSortMode.None);
            int disabledCameras = 0;
            foreach (Camera sourceCamera in sourceCameras)
            {
                if (sourceCamera == null) continue;
                sourceCamera.tag = "Untagged";
                if (sourceCamera.enabled) disabledCameras++;
                sourceCamera.enabled = false;
                AudioListener audio = sourceCamera.GetComponent<AudioListener>();
                if (audio != null) audio.enabled = false;
            }
            GameObject host = new GameObject("Generic Scene Agent Runtime Bridge");
            DelayedInstaller installer = host.AddComponent<DelayedInstaller>();
            installer.Configure(observer, disabledCameras);
        }

        private static void RemoveExistingMotionControllers()
        {
            NormalHmdAgentController[] existing = UnityEngine.Object.FindObjectsByType<NormalHmdAgentController>(
                FindObjectsInactive.Include, FindObjectsSortMode.None);
            foreach (NormalHmdAgentController controller in existing)
            {
                if (controller == null) continue;
                controller.enabled = false;
                controller.gameObject.SetActive(false);
                UnityEngine.Object.Destroy(controller.gameObject);
            }
        }

        private sealed class DelayedInstaller : MonoBehaviour
        {
            private Camera observer;
            private int disabledSourceCameras;

            public void Configure(Camera value, int disabledCount)
            {
                observer = value;
                disabledSourceCameras = disabledCount;
            }

            private IEnumerator Start()
            {
                // Let the source project's ordinary Awake/Start scene construction
                // finish, then inspect exactly what is live at runtime.
                yield return null;

                // Catch controllers created by a source Start/coroutine after the
                // RuntimeInitialize callback.  No generic controller exists yet,
                // so this remains a narrowly scoped hand-over, not scene cleanup.
                RemoveExistingMotionControllers();

                GameObject brainObject = new GameObject("AgentBrain (Contract Grounded)");
                GenericSceneAgentBrain brain = brainObject.AddComponent<GenericSceneAgentBrain>();
                GameObject participant = new GameObject("ParticipantRoot (Generic Scene Agent)");
                NormalHmdAgentController controller = participant.AddComponent<NormalHmdAgentController>();
                controller.Configure(observer);
                controller.ConfigureMotionDriver(brain);
                brain.Configure(controller, observer);
                string screenshot = ReadArgument("--generic-agent-capture-output=");
                if (!string.IsNullOrWhiteSpace(screenshot))
                {
                    GenericRenderProbe probe = brainObject.AddComponent<GenericRenderProbe>();
                    probe.Configure(
                        screenshot,
                        ReadArgument("--generic-agent-camera-audit="),
                        disabledSourceCameras);
                }
                Destroy(gameObject);
            }

            private static string ReadArgument(string prefix)
            {
                foreach (string argument in Environment.GetCommandLineArgs())
                    if (argument.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                        return argument.Substring(prefix.Length).Trim('"');
                return null;
            }
        }
    }
}
