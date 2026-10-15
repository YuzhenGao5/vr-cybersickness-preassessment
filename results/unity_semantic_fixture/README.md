# Synthetic Unity semantic-parser fixture

This directory is a deliberately synthetic, text-serialized Unity fixture. It
exists only to test the semantic parser and is not participant data, a measured
cybersickness scene, or a production Unity project.

The fixture contains evidence for:

- GameObject names, tags, active state, and component references;
- Transform positions and parent links;
- BoxCollider components and trigger state;
- MonoBehaviour-to-C# resolution through `.meta` GUIDs;
- an interactable button;
- fast forced motion, a rotating platform, a narrow corridor, a cliff edge,
  and a dense-particle area;
- a navigation spawn and teleport anchor;
- a prefab containing a scripted high-speed elevator.

Risk labels are expected to be parser hypotheses based on these declared
signals. They are not empirical sickness labels.
