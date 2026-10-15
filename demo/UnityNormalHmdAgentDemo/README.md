# Normal HMD Agent v4 — independent Unity demo

This is a standalone Unity `6000.2.9f1` project for the current EyeNavGS-derived
Normal HMD v4 runtime. It is separate from the older `UnityRandomAgentDemo`.

## What the model does

The exported model uses three independent one-layer LSTM branches with hidden
size 96 (125,689 parameters):

- **Root branch:** right/forward velocity and yaw rate;
- **Horizontal branch:** event timing, endpoint and duration for local x/z,
  yaw and roll;
- **Vertical branch:** event timing, endpoint and duration for local y and
  pitch.

Its 9-D physical state is:

```text
root right velocity, root forward velocity, root yaw rate,
HeadLocal right/up/forward, HeadLocal rotation-vector x/y/z
```

At a horizontal or vertical event onset, the runtime samples one bounded
endpoint and one duration. It then holds that event and interpolates with
`10u^3 - 15u^4 + 6u^5` minimum jerk. Horizontal and vertical events are
separate controllers. There is **no per-frame HeadLocal noise**; all six
HeadLocal innovation standard deviations are zero. This replaces the obsolete
v1/v2 AR(1)-per-head-channel implementation that caused visible vertical and
horizontal jitter.

The Root branch retains calibrated correlated innovation. The final applied
9-D state, after root innovation and safety clipping, is normalized and fed
back to all three LSTMs. A seed changes the held-out starting window, event
samples and Root innovation; replaying the same seed is deterministic.

Runtime hierarchy:

```text
ParticipantRoot                XZ translation + yaw only
└── TrackingOrigin             upright, nominal eye height
    └── HeadLocal              local trained HMD pose
        └── ParticipantCamera  AI-controlled first-person view

ObserverCamera                 independent external camera
```

EyeNavGS has no measured torso, feet or full-body pose. `ParticipantRoot` is a
causal proxy inferred from HMD motion, not ground-truth walking biomechanics.

## Build and use

From the repository root:

```powershell
.\demo\build_unity_normal_hmd.ps1
```

The checked-in runtime export is:

```text
Assets/Resources/NormalHmdAgentModel.json
```

It uses schema `normal-hmd-agent-unity-4.0`. The build verifies the JSON and
checkpoint identities plus deterministic, event, Root-innovation and combined
stochastic golden trajectories. See [MODEL_EXPORT_SCHEMA.md](MODEL_EXPORT_SCHEMA.md).

Controls: `Space` pauses; `C` switches between the first-person and observer
cameras; the UI can start a new seed or replay the current seed. Fixed-step
model output is generated at about 37 Hz and transforms are interpolated only
for display.

## Scientific status

The deployment smoothness validation passed, including a reduction in linear
jerk P95 from `42.985` to `9.6436 m/s^3` and yaw jerk P95 from `74.063` to
`15.812 deg/s^3`. The separate 16×60-second human-distribution rollout did
**not** pass because horizontal/vertical event occupancy was too low. Therefore
this is a smooth hybrid motion prior, not proof of complete human-motion
equivalence.

This project alone is not scene-aware navigation, a trained decision Brain, a
full-body digital human, a cybersickness predictor, or a model of a person with
cybersickness. The generic Agent pipeline integrates this motion skill with a
separate high-level Brain and explicit safety logic; see the repository
[project report](../../docs/PROJECT_REPORT_2026-10-15.md).
