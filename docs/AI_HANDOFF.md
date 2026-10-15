# AI HANDOFF — read this before changing the project

Snapshot: `research-prototype-2026-10-15`  
Audience: a new AI/engineer with no conversation memory  
Repository root: resolve from this file; never hard-code the original workstation path

## Mission

Build a design-time Unity VR evaluation system that can parse an unfamiliar project, create an auditable
experience contract, let a virtual first-person participant explore and interact safely, collect motion and
scene evidence, and eventually estimate cybersickness risk plus actionable design changes.

This is a research/authoring aid, not a medical device.

## Read order

1. `README.md`
2. `docs/PROJECT_REPORT_2026-10-15.md`
3. `release/research_snapshot_2026-10-15.json`
4. `demo/SCENE_AGENT_TERMINAL.md`
5. `docs/DATASETS.md`
6. This file again before making scientific claims

`PROTOCOL.md` and `REPLICATION.md` are frozen historical research documents. They are useful provenance, but
their early status language is superseded by the 2026-10-15 report. `results/` is local and ignored except for
`results/unity_semantic_fixture`, a deliberately synthetic, data-free parser/contract test fixture.

## Authoritative current facts

- The engineering path `parse -> contract -> isolated build -> explore -> verify -> export` works.
- Generic runs are content-driven, not fixed at 60 seconds. Success requires all grounded targets visited and
  all required interactions verified. Guards: minimum evidence 15 s, no progress 45 s, safety ceiling 900 s.
- Normal HMD v4 is the low-level EyeNavGS-derived motion skill. It is not the high-level Brain or a sickness model.
- Unity Decision Brain v0 can control a Player only in explicit research mode. Its 919 labels are entirely
  rule-bootstrap; there are zero human Unity decision labels. Keep `dataStatus=smoke_only` and
  `humanLikeClaimAllowed=false`.
- Habitat-Web prior is a separate public-human navigation experiment and is not connected to Unity.
- VR.net `scene_head_multiscale` logistic regression reached exploratory pooled LOGO AUC 0.6819, but its pickle
  is not connected to the Terminal or Unity runtime. Current experience-report proxy is not a VR.net score.
- The system has no trained “cybersickness-prone human motion generator”.
- “Any Unity project” means the pipeline can attempt ordinary text-serialized projects in an isolated copy;
  it does not guarantee compilation, full gameplay discovery or zero adaptation.
- The production reviewed-binding directory has no approved project bindings. Final Garden runs are navigation
  and observation only; interaction mechanics are proven by fixtures.

## Current source of truth

| Concern | File(s) |
|---|---|
| Terminal orchestration | `demo/scene_agent_terminal.py` |
| Generic Unity runtime | `demo/generic_scene_agent/*.cs` |
| Isolated build/acceptance | `demo/build_generic_scene_agent.ps1` |
| HMD Unity runtime | `demo/UnityNormalHmdAgentDemo/Assets/Scripts/*.cs` |
| HMD runtime export | `demo/UnityNormalHmdAgentDemo/Assets/Resources/NormalHmdAgentModel.json` |
| HMD training provenance | `models/normal_hmd_v4_metadata.json` |
| Decision runtime export | `models/unity_decision_brain_v0_unity.json` |
| Static Unity analysis | `scripts/unity_parser.py`, `scripts/parse_unity_scene.py` |
| Contract compiler | `scripts/compile_scene_agent_contract.py` |
| Runtime materializer | `scripts/materialize_generic_runtime_contract.py` |
| Reviewed binding policy | `configs/scene_agent_reviewed_bindings/README.md` |
| Release evidence | `release/research_snapshot_2026-10-15.json` |

The canonical `GenericSceneAgentBrain.cs` SHA-256 for this snapshot is
`b070363b572e3fc70b3a52528923ed091e68ed496a0c0888c0b2af657d1a4196`.

## Responsibility boundaries — do not collapse these

### Parser / optional LLM

Reads serialized project evidence and produces descriptions/candidates. It does not prove runtime reachability,
interaction success or sickness. LLM output cannot grant runtime authority.

### Contract compiler

Owns scene identity, coordinates, stable target IDs, method candidates, predicates and audit hashes.

### High-level Brain

Chooses targets and macro skills. Rule is the dependable baseline. Shadow logs model decisions without control.
Model research needs an explicit override. Safety masks are mandatory in every mode.

### Normal HMD v4

Generates learned root/head dynamics. Three LSTM branches (root/horizontal/vertical), hidden 96,
125,689 parameters. Event endpoints/durations are sampled once at onset and interpolated with minimum jerk;
there is no per-frame HeadLocal noise. Planner heading, collision response and interaction locking are engineered.

### Safety/interaction layer

Full-height capsule sweep, stable-side steering, inertia hard-stop, overlap sampling, ground filters, 0.35 s
interaction settle, BodyRoot transaction lock, coroutine/next-frame verification and fail-closed required actions.
Never move these responsibilities into an unconstrained learned policy without equivalent hard gates.

### Risk critic

Not yet integrated. The engineering proxy may rank motion exposure but must remain labelled uncalibrated.

## Final validation facts

- Rule Garden: `passed/content_complete`, 317.304 s, 10/10, collision overlap 0.
- Model Garden: same completion, model authority, 40 inferences, 0 fallbacks, 2 rule disagreements, finite logits.
- Collision positive: 77.895 s, 4/4, 471 avoidance adjustments, 0 overlap, required interaction verified.
- Collision negative: forced overlap during interaction lock detected; acceptance failed, exit 3.
- Async positive: 2/2 required; next-frame and coroutine state changed after invocation; root violations 0.
- Async negative: false-return/pre-existing state did not pass; exactly one call; 8.667 s content_exhausted.
- Pre-publication integrated Python suite: 88/88 passed. An earlier final Unity-fix report recorded 74/74.
- Unity version used for evidence: `6000.2.9f1`.

The older `59.994 s / 8 of 10 / 33 inference` report is superseded. Do not cite it as current.

## Data/model truth table

| Artifact | Training/source | Integrated now | Scientific status |
|---|---|---:|---|
| Normal HMD v4 | EyeNavGS, 46 participants × 12 scenes | yes | smooth runtime passed; full human-distribution rollout failed |
| Unity Decision Brain v0 | 919 rule-bootstrap contexts | yes, research only | smoke-only; below rule utility Top-1 |
| Habitat-Web prior | ObjectNav THDA 70k public-human actions | no | offline prior; no Unity/visual/map input |
| VR.net classifier | local 12-game, 1,024-rating-event subset | no | exploratory AUC; uncalibrated and path/head dependent |

Complete raw datasets, VR.net participant/event feature tables, OOF predictions, pickle checkpoints, builds and
local results are intentionally absent from Git. `NormalHmdAgentModel.json` does include 414 processed EyeNavGS
validation seed windows from 104 trajectories and 7 pseudonymous participants. The historical Random Agent export
includes 504 additional windows from 84 trajectories and 7 pseudonymous participants. Preserve Apache-2.0
attribution when redistributing either export.

## Safe first commands

```powershell
python scripts/check_release_package.py
python -m unittest `
  scripts.test_unity_parser `
  scripts.test_compile_scene_agent_contract `
  scripts.test_generic_scene_agent_collision `
  scripts.test_generic_scene_agent_interaction_lock `
  scripts.test_scene_agent_terminal_acceptance -v
git status --short
```

If the local research environments exist, run the full suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s scripts -p "test_*.py" -v
```

Do not run `git add -A` before inspecting `git status --ignored`; large/private directories are ignored, but
always stage explicit source paths for a release.

## Runtime/build procedure

1. Build the terminal: `.\demo\build_scene_agent_terminal.ps1`.
2. Select project and scene in Terminal.
3. Parse and compile contract.
4. Review interaction candidates separately; observe-only is the safe default.
5. Build in staging with `demo/build_generic_scene_agent.ps1`.
6. Accept only `content_complete` plus camera/render/collision/interaction gates.
7. Keep raw runs local; update only a new aggregate `release/*.json` snapshot.

The generic build script defaults to
`C:\Program Files\Unity\Hub\Editor\6000.2.9f1\Editor\Unity.exe`; use its explicit `-UnityEditor` argument when
Unity is installed elsewhere. The Terminal UI still uses the default path, so safe Editor auto-discovery remains
a useful portability task.

## Claim guardrails

Never write any of the following unless new evidence genuinely establishes it:

- “the model predicts cybersickness with 68.19% accuracy”;
- “0.7 means the participant has a 70% probability of sickness”;
- “EyeNavGS trained the Agent to understand or navigate Unity scenes”;
- “all Agent motion is learned”; 
- “the Brain learned human decisions”;
- “the Agent works in every Unity project without adaptation”;
- “Normal HMD matches the complete human motion distribution”;
- “head movement feature importance proves a design change will reduce sickness”;
- “the project contains no dataset samples” (the runtime JSON includes processed seed windows).

Preferred concise statement:

> The prototype automates auditable Unity exploration and evidence collection. Its low-level HMD prior is
> EyeNavGS-derived, its current learned high-level policy is a rule-bootstrap smoke model, and its VR.net risk
> classifier remains offline; therefore it does not yet provide a validated personal sickness probability.

## Next implementation priorities

1. Build a versioned telemetry-to-73-column VR.net adapter with unit and missingness tests; evaluate offline
   before wiring the critic into UI.
2. Add at least six independent Unity project/scene families for external validation.
3. Collect human/expert Unity demonstrations and corrections with provenance; add Interact, Stop, NavMesh,
   visibility and recovery examples.
4. Improve long-rollout HMD event occupancy using a newly frozen participant holdout and subjective pilot.
5. Create approved reviewed bindings only after code/scene hash review and a non-destructive sandbox identity.
6. Decide a project-level license before inviting public reuse.

## When updating this handoff

- Create a dated report and machine-readable release manifest; do not silently rewrite old metrics.
- Record dataset versions, split unit, hashes, seeds, model status and every failed gate.
- A result is “current” only if source hashes and runtime evidence match.
- Keep negative tests: a safety mechanism is not validated by positive runs alone.
- If a new model takes authority, compare it with Rule, check legal masks, finite logits, fallbacks and real
  Player behavior—not only offline accuracy.
