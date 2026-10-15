# Normal HMD v4 Unity export contract

`Assets/Resources/NormalHmdAgentModel.json` is the checked-in Unity runtime
export for schema `normal-hmd-agent-unity-4.0`. It is generated from the Normal
HMD v4 PyTorch checkpoint by `scripts/normal_hmd_v4_export.py`; it is not
compatible with the older Random Agent or Normal HMD v1/v2 schemas.

## Model and state

The export records EyeNavGS provenance, source/checkpoint/data hashes,
`syntheticTraining=false`, 125,689 parameters, and three independent one-layer
PyTorch LSTMs (`ifgo`, row-major), each with hidden size 96:

- Root LSTM: outputs 3 continuous Root channels;
- Horizontal LSTM: continuous parity head plus hazard and 5 event parameters;
- Vertical LSTM: hazard and 3 event parameters.

The 9-D physical state order is:

1. body-right velocity (m/s)
2. body-forward velocity (m/s)
3. right-handed body yaw rate (rad/s)
4. local head-right offset (m)
5. local head-up offset from nominal eye height (m)
6. local head-forward offset (m)
7. local head rotation-vector X (rad)
8. local head rotation-vector Y (rad)
9. local head rotation-vector Z (rad)

The export contains physical normalizers, safety limits, a 64-step raw-history
seed bank, coordinate conventions and a fixed step of `0.027 s`.

## Holding-event synthesis

Horizontal and vertical event controllers are independent. A seeded Bernoulli
hazard is evaluated only on eligible frames. At onset, the controller consumes
one correlated Gaussian parameter sample, bounds the absolute endpoint and
duration, and latches both for the event. Later frames use the minimum-jerk
curve

```text
S(u) = 10u^3 - 15u^4 + 6u^5
```

before holding the target and applying a four-frame cooldown. The auxiliary
continuous horizontal pose head exists for parity diagnostics; the applied
horizontal pose comes from the holding-event controller.

The normative HeadLocal policy is `none` for per-frame innovation. All six
`headInnovationStd` values are zero. An implementation that adds independent
HeadLocal AR(1) noise or a 150 ms HeadLocal EMA is an obsolete v1/v2 runtime and
does not satisfy this schema.

## Root innovation and recurrent feedback

The Root branch uses calibrated correlated AR(1) residuals. Each step consumes
three Gaussian values, applies the exported Cholesky matrix and per-axis rho,
and adds the scaled residual to the Root prediction. Unlike the retired
base-only feedback contract, the final applied and clipped 9-D physical state
is normalized and fed back to all three LSTMs.

Unity applies Root channels only to upright XZ translation and yaw. Channels
4–9 drive `HeadLocal`; the first-person camera is its child. Display-time
Lerp/Slerp does not enter recurrent state or fixed-step telemetry.

## Golden parity and integrity

The export contains four complementary checks:

- deterministic multi-head/controller golden rollout;
- event-controller golden rollout and derived horizontal/vertical seeds;
- Root-innovation RNG golden rollout;
- combined stochastic golden rollout covering Root innovation, event timing,
  event sampling, feedback and final applied state.

The portable RNG, seed-mixing constants, state/controller/diagnostic layouts,
tolerances and expected arrays are part of the contract. The exporter also
rewrites `GeneratedNormalHmdIntegrity.cs`; Unity build/runtime validation fails
if the JSON SHA-256, source model identity, dimensions, provenance or golden
outputs do not match.

## Published data boundary

The JSON includes 414 processed EyeNavGS validation seed windows and pseudonymous
dataset IDs so different seeded starts are reproducible. It does not contain the
complete EyeNavGS dataset, but it is still a derived dataset asset covered by
the attribution in `THIRD_PARTY_NOTICES.md` and the bundled Apache-2.0 text.

Scientific interpretation, training counts and validation limits are recorded
in `docs/PROJECT_REPORT_2026-10-15.md` and `docs/AI_HANDOFF.md` at repository
root. In particular, smooth deployment behavior passed its validation, while
the full human-distribution occupancy gate did not.
