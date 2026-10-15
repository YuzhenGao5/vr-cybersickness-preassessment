# Habitat-Web Decision Prior v1

## What this model is

This pipeline trains a small six-action MLP from public human Habitat-Web
ObjectNav demonstrations.  Its action space is:

`MOVE_FORWARD`, `TURN_LEFT`, `TURN_RIGHT`, `LOOK_UP`, `LOOK_DOWN`, `STOP`.

The provenance is `public_human`; the local dataset is marked `CC BY-NC 4.0`.
The trained component is a pose-and-goal-conditioned navigation action prior.
It is **not** a visual policy, map builder, Unity interaction policy, collision
controller, cybersickness model, or complete Unity Brain.  Held-out Matterport
results do not justify a claim that it navigates arbitrary Unity scenes like a
human.

## Leakage-safe alignment

Habitat-Web stores the state produced **after** each replay action.  Row zero is
a placeholder `STOP` that supplies the initial observation.  Consequently the
only valid supervised alignment is:

```text
observation = reference_replay[i - 1].agent_state
history     = actions[1:i]
label       = actions[i]
```

The first placeholder `STOP` is removed from labels and history.  The terminal
`STOP` remains a label.  `reference_replay[i].agent_state` is never used to
predict `actions[i]`; doing so would expose the action's resulting translation,
rotation, or sensor pitch.

`episode["goals"]` is commonly empty.  Candidate targets are resolved from:

```text
goals_by_category["<scene filename>_<object_category>"]
```

At each observation, the nearest annotated goal view point is used; an object
goal position is used only if no finite view point exists.  These are static
ObjectNav annotations, not the demonstrated trajectory endpoint.  For Unity,
the semantic parser/contract must provide equivalent candidate target
positions.  A future trajectory endpoint must never be supplied at inference.

Inputs contain the current pre-action pose, relative annotated waypoint,
travel already completed, strictly previous actions, previous resultant motion,
and the RGB sensor's pre-action pitch.  They exclude current/future actions,
future state, remaining step count, success, outcome, object category, Scene ID,
and Episode ID.

## Scene split and bounded preparation

Scene IDs are hashed with seed `42` and assigned as whole groups to
train/validation/test.  No frame or episode from one scene can cross splits.
Each split uses an independently seeded uniform reservoir; 18M source action
records are not stored in memory together.  Source files are loaded one scene
at a time.

Default preparation:

```powershell
.\.venv-normal-hmd\Scripts\python.exe -u scripts\prepare_habitat_decision_data.py `
  --config configs\habitat_decision_prior_v1.json
```

The prepared data default to:

```text
data\derived\habitat-web\decision_prior_v1
```

For a deterministic development subset, select whole scenes and episodes by
hash, rather than taking the source file's first entries:

```powershell
.\.venv-normal-hmd\Scripts\python.exe -u scripts\prepare_habitat_decision_data.py `
  --config configs\habitat_decision_prior_v1.json `
  --output-dir data\derived\habitat-web\decision_prior_v1_smoke `
  --max-scenes 9 --max-episodes-per-scene 3 --max-samples 10000
```

Any model trained from such a subset is stamped `training_scope=smoke_subset`
and `production_model=false`. Reading every configured episode instead stamps
`full_dataset_coverage=true`; it does **not** by itself make the offline prior
production-ready. `production_model` remains false until separate closed-loop
and runtime-integration gates exist and pass.

## Training and evaluation

Only the training split fits feature means/standard deviations and class
weights.  Validation and test remain in their natural sampled distribution.
Early stopping selects validation macro-F1, with validation NLL as the tie
breaker.  The report includes:

- accuracy, balanced accuracy, macro-F1;
- precision, recall, F1, and support for every action;
- true-row/predicted-column confusion matrix;
- negative log likelihood and expected calibration error;
- `always MOVE_FORWARD`, repeat-previous-action, and simple geometry baselines.

Full configured training:

```powershell
.\.venv-normal-hmd\Scripts\python.exe -u scripts\train_habitat_decision_prior.py `
  --config configs\habitat_decision_prior_v1.json --device cuda
```

The checkpoint stores the feature/action contracts, training-only
normalization, class weights, source/prepared hashes, split unit, seed, license,
scope, and claims.  The self-contained JSON stores all MLP weights plus a golden
input/logit/probability vector and the checkpoint hash.

## Verification

```powershell
.\.venv-normal-hmd\Scripts\python.exe -m unittest `
  scripts.test_habitat_decision_pipeline -v
```

The tests include the post-action-state leakage regression, sensor-pitch
alignment, missing-sensor fallback, top-level goal lookup, scene isolation,
deterministic reservoir, train-only normalization, metric/baseline checks,
one-epoch end-to-end training, and JSON golden-vector parity.
