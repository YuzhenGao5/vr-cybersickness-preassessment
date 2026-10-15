# Reviewed interaction bindings

This directory is the only automatic trust root used by Scene Agent Terminal.
Unity project files and LLM output cannot authorize method calls.

One approved JSON file may match a project/scene. Required top-level fields:

```json
{
  "schemaVersion": "scene-agent-reviewed-bindings-1.0",
  "approvalStatus": "approved",
  "projectPath": "C:/absolute/path/to/UnityProject",
  "scenePath": "Assets/Scenes/Example.unity",
  "sourceSceneSha256": "sha256-of-the-reviewed-scene-file",
  "bindings": [
    {
      "target_node_id": "node:...",
      "verb": "activate",
      "component_type": "Namespace.SafeComponent",
      "method": "Activate",
      "script_path": "Assets/Scripts/SafeComponent.cs",
      "script_sha256": "sha256-of-the-reviewed-script-file",
      "success_member": "IsActive",
      "expected_value": "true",
      "required": true,
      "reviewed_by": "reviewer identity",
      "reviewed_at_utc": "2026-10-14T00:00:00Z"
    }
  ]
}
```

The selected method must also exist as a public, non-static, zero-argument
candidate at the same script path in the current deterministic scene contract.
The only narrow runtime-adapter exception is an exact `Button` or
`UnityEngine.UI.Button` binding with method `Press`: the reviewed target must
also have deterministic Button component evidence and a target affordance in
the contract. The manifest, scene hash, script path/hash, reviewer, timestamp,
and success member are still mandatory; this exception does not authorize any
other UnityEvent or inferred method.
Changing the scene, the reviewed script, deleting or editing this manifest, or
changing its approval state invalidates the existing Player build and requires
a new analysis/build.
