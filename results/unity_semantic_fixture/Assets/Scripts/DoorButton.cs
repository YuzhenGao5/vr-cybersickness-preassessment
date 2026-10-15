using UnityEngine;

public sealed class DoorButton : MonoBehaviour
{
    public string interactionType = "press";
    public GameObject targetDoor;

    public void Interact()
    {
        if (targetDoor != null)
        {
            targetDoor.SetActive(!targetDoor.activeSelf);
        }
    }
}
