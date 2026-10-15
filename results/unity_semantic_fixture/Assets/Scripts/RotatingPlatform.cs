using UnityEngine;

public sealed class RotatingPlatform : MonoBehaviour
{
    public float angularSpeedDegreesPerSecond = 150f;

    private void Update()
    {
        transform.Rotate(Vector3.up, angularSpeedDegreesPerSecond * Time.deltaTime,
            Space.World);
    }
}
