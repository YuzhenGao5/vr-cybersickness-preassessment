# Generic Unity Scene Agent runtime bridge

This package connects the offline Unity evidence compiler to the Normal HMD v4
motion runtime without editing the selected source project.

## Decision Brain modes

The Player accepts `--decision-brain-mode=rule|shadow|model` (default `rule`).
`shadow` evaluates the bundled Unity-flat model while Rule keeps authority.
`model` grants research-only authority and, for any data status other than
`production_candidate`, also requires `--decision-brain-research-override`. A bad shadow model falls back
visibly to Rule; model authority fails closed.

The build bridge copies `models/unity_decision_brain_v0_unity.json` to the
reserved `Assets/Resources/UnityDecisionBrainModel.json` resource and an
identical `Assets/StreamingAssets/UnityDecisionBrainModel.json` audit copy.
The schema must be `unity-decision-brain-unity-flat-1.0`. Its hash/status are
recorded separately from `normalHmdModelSha256`.

## Responsibility split

- `compile_scene_agent_contract.py` inventories Unity scenes, prefabs and C#
  evidence. An optional LLM may summarize and rank evidence, but cannot grant
  runtime authority.
- `materialize_generic_runtime_contract.py` deterministically converts
  `scene-agent-contract-1.0` into the small
  `generic-scene-agent-contract-1.0` Unity schema.
- `GenericSceneAgentBrain` grounds contract targets against the live hierarchy,
  adds observation-only runtime discoveries, chooses unfinished reachable
  targets, and schedules Plan, Navigate, Stop, Look, Interact, Observe, Verify
  and Recover.
- `NormalHmdAgentController` remains the low-level EyeNavGS-derived movement
  skill. It generates HeadLocal motion and the learned BodyRoot speed envelope;
  the selected Rule/Shadow/Model high-level authority decides when to walk or
  stop and redirects the heading.

With a baked NavMesh the Brain follows cached path corners and refreshes the
path. Without one it reports `collision-checked-direct-steering(no-NavMesh)`
and does not claim global path planning. In both modes, the final velocity that
will actually be integrated is checked with a full-height capsule sweep. A
stable-side steering candidate may replace a blocked navigation command; when
all candidates are blocked the old smoothed velocity is cleared immediately,
then a bounded watchdog replans instead of repeatedly pushing into the wall.
The capsule is also sampled while stopped and while an interaction lock is
active, so a moving obstacle cannot enter the participant proxy unnoticed.
Any distinct overlap episode fails the runtime acceptance rather than being
reported as a safe completion. Ground projection and runtime ground following
share the same filter: triggers, participant colliders and steep faces are
rejected, while foot-height hits on named walkable ramps remain traversable.

## Build and acceptance

From the repository root:

```powershell
.\demo\build_generic_scene_agent.ps1 `
  -SourceProject 'C:\path\to\UnityProject' `
  -CompiledContract '.\results\scene_agent_contract.json' `
  -ScenePath 'Assets/Scenes/ChosenScene.unity' `
  -ReviewedBindings '.\reviewed_agent_bindings.json' `
  -DisableBehaviourType 'Namespace.ExistingPlayerController' `
  -DecisionBrainMode shadow `
  -DecisionBrainModel '.\models\unity_decision_brain_v0_unity.json' `
  -Seed 2026101401
```

For current smoke-only model authority, use `-DecisionBrainMode model` plus
the explicit `-DecisionBrainResearchOverride` switch. Omitting or corrupting
the model makes `shadow`/`model` exit non-zero; `rule` remains buildable.

`ScenePath` defaults to `runtime_scene.path` in the compiled contract.
`ReviewedBindings`, `DisableBehaviourType`, `BuildDirectory` and
`StagingDirectory` are optional. The script rejects source/staging/build path
overlap, copies the project to an isolated staging area, builds only the
selected scene, verifies the source scene SHA-256 is unchanged, then runs a
content-completion first-person acceptance. The generic run stops successfully
only after every grounded target has been visited and every required interaction
has been verified. It has a 15-second evidence floor, a 45-second no-progress
failure guard and a 900-second simulated-time safety ceiling; none of those
guards is reported as successful completion. It also captures three D3D11
frames, selects the most informative one, and validates that it is non-empty,
non-black/white, varied and not overwhelmingly magenta.

Artifact names are stable and unique per source project path, relative scene,
Decision Brain mode, model hash and research-override state:

```text
<project-folder>_<scene-stem>_<sha256(lowercase-absolute-project + "|" + lowercase-scene)[0:10]>
```

The results directory contains:

- `*_build_status.json`: source project, selected scene, compiled/runtime
  contract hashes, source scene hashes before/after, executable and compiler
  diagnostics.
- `*_acceptance.json`: content-completion status, elapsed simulated time,
  coverage, interaction verification and unfinished-target count,
  planner, collision recovery, avoidance-adjustment/hard-stop/overlap counts,
  explicit collision-safety pass/fail, maximum blocked duration,
  ground-following, model schema/data status, golden
  parity, finite-logit, fallback and log-path evidence.
- `*_camera_audit.json`, `*_first_person.png` and
  `*_render_validation.json`: active pipeline/camera/FOV/culling, disabled
  source cameras and visible-frame checks.
- `*_runs/<run>/brain_decisions.jsonl`: state/skill/evidence ledger.
- `*_runs/<run>/decision_contexts.jsonl`: leakage-safe, pre-action
  `unity-decision-context-1.0` records for future policy training.  Every target
  selection and macro skill transition records the full candidate set, explicit
  reachability/visibility-known flags, valid skills and the executed authority's
  label before the action executes. `decisionKind` distinguishes
  `target_selection` from `skill_transition`; canonical identity/action fields
  are `projectId`, `sceneName`, `bodyPosition` and `chosenTargetIndex`.
- `*_runs/<run>/decision_outcomes.jsonl`: post-action
  `unity-decision-outcome-1.0` records joined by `episodeId + decisionId`, with
  arrival, coverage gain, interaction verification, timeout, collision recovery
  and terminal evidence. Canonical outcome fields include `simulatedTime`,
  `collisionRecoveriesDelta` and `terminalState`. Outcome fields are never
  copied into the context.
- `*_runs/<run>/generic_agent_telemetry.csv`: per-step motion and coverage.
- `*_runs/<run>/experience_report.md` and `.json`: coverage, supported or
  unsupported interaction, motion/visual engineering signals and suggestions.

The exposure proxy in the experience report is an uncalibrated engineering
signal. It is not a VR.net probability, medical assessment or causal claim.

## Decision-training provenance

The new context/outcome files make the runtime observable enough to build a
target-ranking head and a macro-skill head. Rule and Shadow execution records
declare `sourceKind=rule_bootstrap`; model-controlled records declare
`sourceKind=model_rollout`. Both set `humanDemonstration=false`, and the human
acceptance/correction fields remain empty annotation slots. Model rollouts are
deliberately outside the frozen training-source allowlist so self-generated
actions cannot be silently recycled as rule or human labels. Human
teleoperation or imported demonstrations must retain distinct provenance.

Candidate visibility is not inferred from object resolution or a camera alone.
Until an occlusion-aware sensor is added, every candidate records
`visibilityKnown=false` and `visible=false`.  Reachability becomes known only
after the runtime planner actually tests that candidate.  Candidates that are
visited, cooling down, unreachable or not yet tested remain in the candidate
array with an explicit ineligibility reason, so rejected alternatives are not
silently lost.

## Safe interaction policy

Analysis discovery hints can only create observation targets after a live name
and required-component match. They never authorize a call. A method is invoked
only when a separate reviewed binding matches an evidence compiler candidate
that is public, instance, zero-argument and present in
`safeInteractionMethods`. Unity UI Button `Press` is the only special case.
Every attempt records before/after state evidence and whether the declared
postcondition was verified. Invocation and verification are separate phases:
the postcondition is first checked on a later rendered frame and is then polled
during observation. Reflected methods returning `IEnumerator` are started as
Unity coroutines so delayed gameplay effects can execute.

Before an approved interaction is invoked, the runtime hard-stops BodyRoot XZ
translation and planner yaw for a 0.35-second settling window. That lock remains
active through the post-interaction observation/verification period, while the
EyeNavGS-derived HeadLocal sequence continues normally. The acceptance report
records lock acquisitions/releases, locked telemetry rows and any non-zero root
command violation. A required interaction remains unvisited and keeps this lock
until its postcondition succeeds or the 30-second transaction deadline expires.

## Generality boundary

The adapter can stage and inspect ordinary Unity scenes, but “any scene” is not
an unconditional guarantee. Custom Addressables loading, network login, XR
provider setup, procedural worlds without detectable ground, or game-specific
multi-step interactions need an explicit adapter or reviewed binding. The
runtime records these gaps instead of fabricating successful exploration.
