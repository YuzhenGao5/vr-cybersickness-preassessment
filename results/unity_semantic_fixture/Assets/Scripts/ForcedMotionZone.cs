using UnityEngine;

public sealed class ForcedMotionZone : MonoBehaviour
{
    public float speedMetersPerSecond = 12f;
    public float accelerationMetersPerSecondSquared = 8f;
    public bool rotatesCamera = true;

    private void OnTriggerStay(Collider other)
    {
        other.transform.Translate(Vector3.forward * speedMetersPerSecond * Time.deltaTime,
            Space.World);
    }
}
