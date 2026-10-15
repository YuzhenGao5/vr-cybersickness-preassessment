# Unity Decision Brain v0

## Scope

Decision Brain v0 is a small, auditable candidate-set ranker. It learns which
runtime-grounded target to inspect next and predicts one macro skill from:

`Navigate`, `LookAt`, `Observe`, `Interact`, `Replan`, `Stop`.

It does not replace NavMesh, collision handling, interaction authorization,
postcondition verification, Normal HMD motion, or the deterministic fallback
state machine. A model output is always filtered through the runtime target and
skill masks.

The initial dataset is `rule_bootstrap`. It is policy distillation, not a human
navigation dataset. Model metadata must continue to show
`human_like_claim_allowed=false`.

## Runtime files

Each completed Unity run can emit two joined JSONL files:

- `decision_contexts.jsonl`, schema `unity-decision-context-1.0`: the
  leakage-safe state captured before an action is selected;
- `decision_outcomes.jsonl`, schema `unity-decision-outcome-1.0`: results after
  that action ends.

The join key is `(episodeId, decisionId)`. Preparation rejects missing,
duplicate, or orphaned records.

The canonical context is the flat Unity/C# schema. The loader also accepts the
earlier nested draft for migration only. Canonical flat records are strict:
missing `decisionKind`, provenance, action, Vector3, or outcome join fields are
rejected rather than filled with a guessed label. Candidate identifiers, Scene/project
identifiers, absolute BodyRoot position, and teacher utility values are retained
only for validation/provenance; they are never model features.

Optional migration fields such as `pathLength`, `analysisConfidence`, and
`dynamicObject` may be read but are deliberately not v0 model features because
the canonical Unity runtime does not emit them.

`decisionKind=target_selection` contributes a target-ranking label.
`decisionKind=skill_transition` contributes only a skill label. Both types are
kept for the skill head.

## Provenance

Exactly four source kinds are accepted:

| sourceKind | Meaning | Human label |
|---|---|---:|
| `rule_bootstrap` | Generic rule Brain teacher | no |
| `scripted_bootstrap` | reviewed scenario-specific script | no |
| `public_human` | mapped public human demonstration | yes |
| `unity_human` | locally recorded Unity human demonstration/correction | yes |

The boolean `humanDemonstration` must agree with this table. Unity human labels
also require an explicit `humanAccepted` annotation. `labelSource` must be one
of the same four values and must match `sourceKind`; mixed provenance is
rejected. A rejected proposal requires at least one explicit target or skill
correction, while an unchanged field keeps the originally logged label.

Synthetic unit-test records must contain `syntheticFixture=true`. Production
commands reject them unless the explicit test-only switch is supplied.

## Prepare

From the repository root:

```powershell
.\.venv-normal-hmd\Scripts\python.exe scripts\prepare_unity_decision_data.py `
  --contexts results\generic_scene_agent\...\decision_contexts.jsonl `
  --outcomes results\generic_scene_agent\...\decision_outcomes.jsonl
```

The default production gate requires at least 200 decisions across six
scene/project-family groups. `--allow-smoke` permits an undersized pipeline
check, but the manifest remains `smoke_only`.

Splits are grouped by `splitFamily`, or by `projectFamily + sceneFamily`, with
`projectId + sceneName` as the canonical Unity fallback. Different seeds from
one Scene therefore cannot leak across train, validation, and test.

Outputs:

```text
features/unity_decision_brain_v0/
  dataset_manifest.json
  split_manifest.json
  unity_decision_train.npz
  unity_decision_val.npz
  unity_decision_test.npz
```

The dataset manifest binds the exact config hash, feature order, split seed,
split-manifest hash, and every NPZ hash/size. Training refuses artifacts that do
not match those bindings.

## Train

```powershell
.\.venv-normal-hmd\Scripts\python.exe scripts\train_unity_decision_brain.py --device cuda
```

Seed 42 is frozen for split and training. Normalization is fitted only on the
training split. The model uses shared candidate encoders plus masked mean/max
set pooling, so candidate order is not a learned cue.

During skill evaluation, target-selection rows are conditioned on the model's
ranked target. Skill-transition rows are conditioned on the already-active
logged target, because the runtime is asking whether to look, observe,
interact, replan, or stop—not asking the target head to select a new object.

Outputs:

- `models/unity_decision_brain_v0.pth`: Python checkpoint;
- `models/unity_decision_brain_v0.json`: self-contained weights,
  normalization, feature order, class order, full virtual-STOP encoding,
  decision-kind conditioning rules, provenance, and runtime authority;
- `models/unity_decision_brain_v0_unity.json`: the same trained checkpoint in
  the dictionary-free `unity-decision-brain-unity-flat-1.0` deployment schema.
  It is made only of top-level scalars and one-dimensional primitive arrays so
  Unity `JsonUtility` can deserialize it without a third-party JSON library;
- `results/unity_decision_brain_v0_metrics.json`: target Top-1/Top-3, MRR,
  NDCG, teacher-utility regret, skill macro-F1/confusion matrix, mask legality,
  source-specific results, and human-metric refusal status;
- `results/unity_decision_brain_v0_history.csv`: optimization history.

If the dataset is `smoke_only`, training also requires `--allow-smoke`. Test
fixtures additionally require `--allow-synthetic-fixture`.

### Unity flat deployment contract

The build copies `models/unity_decision_brain_v0_unity.json` to
`Assets/Resources/UnityDecisionBrainModel.json` and loads it with
`Resources.Load<TextAsset>("UnityDecisionBrainModel")`. There is no companion
manifest: the asset itself carries `modelSha256` (the checkpoint hash),
`pythonExportSha256`, `dataFingerprintSha256`, `configSha256`, and `dataStatus`.

Every dense layer has four fields. For example,
`globalEncoderLinear1InputSize`, `globalEncoderLinear1OutputSize`,
`globalEncoderLinear1Weights`, and `globalEncoderLinear1Bias`. Weights are
flattened from PyTorch `[output,input]` tensors in row-major order, so the C#
index is `outputIndex * inputSize + inputIndex`. The eight prefixes are:

```text
globalEncoderLinear1  globalEncoderLinear2
candidateEncoderLinear1  candidateEncoderLinear2
targetHeadLinear1  targetHeadLinear2
skillHeadLinear1  skillHeadLinear2
```

Feature order, enum vocabularies, train-only mean/std arrays, virtual STOP
encoding, input field names, mask rules, and decision-kind conditioning are
also flat fields in the same JSON. The embedded held-out golden decision
contains unnormalised and normalised encoded tensors, masks, the selected
skill-conditioning target, and expected target/skill logits. Python tests
compare those values with both the original PyTorch checkpoint and an
independent NumPy implementation of the flat runtime graph. Unity should apply
`goldenTolerance` to its own startup parity check and refuse model authority if
schema, dimensions, hashes, or golden parity fail.

To regenerate only this Unity asset from an already trained checkpoint (no
optimizer step and no retraining), run:

```powershell
.\.venv-normal-hmd\Scripts\python.exe scripts\export_unity_decision_brain.py `
  --data-dir results\unity_decision_brain_v0\prepared_40runs `
  --allow-smoke
```

The exporter refuses mismatched config/data fingerprints, NPZ hashes or sizes,
checkpoint/Python-export weights, and requires the explicit smoke flag while
the three-scene dataset remains `smoke_only`.

## Interpretation gates

- Rule/script bootstrap accuracy measures imitation of those policies only.
- If no `public_human` or `unity_human` samples exist, numeric human metrics are
  refused rather than emitted.
- The report lists the count of labels for every macro skill. If only
  `Navigate` and `Stop` occur, macro-F1 cannot be presented as complete
  six-skill capability.
- Illegal target/skill rate is reported both before and after masks. After-mask
  illegality must remain zero.
- Teacher utilities are used for regret evaluation only, never as inputs.
- Metrics include a first-legal target/skill baseline and the deterministic
  pre-jitter rule-base-utility target baseline. A learned score below either
  applicable baseline is not evidence of a useful replacement policy.
- VR.net risk values are not exploration-decision labels.

## Tests

```powershell
.\.venv-normal-hmd\Scripts\python.exe scripts\test_unity_decision_pipeline.py -v
```

Tests use temporary records explicitly marked as synthetic. They verify schema
and provenance gates, group-disjoint splitting, identifier exclusion,
candidate permutation equivariance, variable-size padding, train-only
normalization, artifact hashes/seed binding, mask legality, human-metric
refusal, model training, and equivalent checkpoint/JSON forward passes.
