using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using UnityEngine;

namespace GenericSceneAgent
{
    public enum UnityDecisionBrainMode { Rule, Shadow, Model }

    [Serializable]
    public sealed class UnityDecisionBrainFlatModel
    {
        public string schemaVersion;
        public string generatedAtUtc;
        public string modelName;
        public string scope;
        public string dataStatus;
        public string checkpointSha256;
        public string pythonExportSha256;
        public string dataFingerprintSha256;
        public int globalDim;
        public int candidateDim;
        public int skillCount;
        public string[] globalFeatureNames;
        public string[] candidateFeatureNames;
        public string[] skills;
        public string[] brainStates;
        public string[] plannerModes;
        public string[] semanticTypes;
        public float[] globalMean;
        public float[] globalStd;
        public float[] candidateMean;
        public float[] candidateStd;
        public float[] stopCandidateEncodedFeatures;

        public int globalEncoderLinear1InputSize;
        public int globalEncoderLinear1OutputSize;
        public float[] globalEncoderLinear1Weights;
        public float[] globalEncoderLinear1Bias;
        public int globalEncoderLinear2InputSize;
        public int globalEncoderLinear2OutputSize;
        public float[] globalEncoderLinear2Weights;
        public float[] globalEncoderLinear2Bias;
        public int candidateEncoderLinear1InputSize;
        public int candidateEncoderLinear1OutputSize;
        public float[] candidateEncoderLinear1Weights;
        public float[] candidateEncoderLinear1Bias;
        public int candidateEncoderLinear2InputSize;
        public int candidateEncoderLinear2OutputSize;
        public float[] candidateEncoderLinear2Weights;
        public float[] candidateEncoderLinear2Bias;
        public int targetHeadLinear1InputSize;
        public int targetHeadLinear1OutputSize;
        public float[] targetHeadLinear1Weights;
        public float[] targetHeadLinear1Bias;
        public int targetHeadLinear2InputSize;
        public int targetHeadLinear2OutputSize;
        public float[] targetHeadLinear2Weights;
        public float[] targetHeadLinear2Bias;
        public int skillHeadLinear1InputSize;
        public int skillHeadLinear1OutputSize;
        public float[] skillHeadLinear1Weights;
        public float[] skillHeadLinear1Bias;
        public int skillHeadLinear2InputSize;
        public int skillHeadLinear2OutputSize;
        public float[] skillHeadLinear2Weights;
        public float[] skillHeadLinear2Bias;

        public int goldenCandidateCount;
        public float[] goldenGlobalFeatures;
        public float[] goldenNormalizedGlobalFeatures;
        public float[] goldenCandidateFeatures;
        public float[] goldenNormalizedCandidateFeatures;
        public bool[] goldenCandidateMask;
        public bool[] goldenTargetMask;
        public bool[] goldenSkillMask;
        public int goldenSkillConditioningTargetIndex;
        public float[] goldenExpectedRawTargetLogits;
        public float[] goldenExpectedMaskedTargetLogits;
        public float[] goldenExpectedRawSkillLogits;
        public float[] goldenExpectedMaskedSkillLogits;
        public float goldenTolerance;
    }

    public sealed class UnityDecisionPrediction
    {
        public int targetIndex = -1;
        public string targetId = "";
        public string skill = "";
        public float[] rawTargetLogits = new float[0];
        public float[] maskedTargetLogits = new float[0];
        public float[] rawSkillLogits = new float[0];
        public float[] maskedSkillLogits = new float[0];
    }

    /// <summary>
    /// Small dependency-free inference runtime for the audited, Unity-flat export.
    /// It deliberately applies runtime legality masks in addition to the learned
    /// logits: a checkpoint cannot authorize an unreachable target or interaction.
    /// </summary>
    public sealed class UnityDecisionBrainRuntime
    {
        private const string ExpectedSchema = "unity-decision-brain-unity-flat-1.0";
        private const float MaskedLogit = -1.0e9f;
        private static readonly string[] ExpectedGlobalFeatures =
        {
            "simulated_time", "coverage", "replans",
            "brain_state::Scan", "brain_state::Navigate", "brain_state::LookAt",
            "brain_state::Interact", "brain_state::Observe", "brain_state::Replan",
            "brain_state::Complete", "brain_state::Unknown",
            "previous_skill::Navigate", "previous_skill::LookAt", "previous_skill::Observe",
            "previous_skill::Interact", "previous_skill::Replan", "previous_skill::Stop",
            "previous_skill::Unknown",
            "planner_mode::unity-navmesh-corner-following",
            "planner_mode::collision-checked-direct-steering(no-NavMesh)",
            "planner_mode::unknown"
        };
        private static readonly string[] ExpectedCandidateFeatures =
        {
            "priority", "relative_x", "relative_y", "relative_z", "distance",
            "bearing_sin", "bearing_cos", "cooldown_seconds", "resolved",
            "reachability_known", "reachable", "visibility_known", "visible",
            "navmesh_projected", "visited", "interaction_declared",
            "interaction_available", "interaction_required", "interaction_attempted",
            "interaction_verified", "risk_candidate", "semantic_type::risk_candidate",
            "semantic_type::interaction_point", "semantic_type::runtime-discovered-observation",
            "semantic_type::runtime-walkable-sample", "semantic_type::landmark",
            "semantic_type::dynamic", "semantic_type::unknown", "semantic_type::__stop__"
        };
        private static readonly string[] ExpectedSkills =
            { "Navigate", "LookAt", "Observe", "Interact", "Replan", "Stop" };
        private static readonly string[] ExpectedStates =
            { "Scan", "Navigate", "LookAt", "Interact", "Observe", "Replan", "Complete", "Unknown" };
        private static readonly string[] ExpectedPlannerModes =
        {
            "unity-navmesh-corner-following",
            "collision-checked-direct-steering(no-NavMesh)", "unknown"
        };
        private static readonly string[] ExpectedSemanticTypes =
        {
            "risk_candidate", "interaction_point", "runtime-discovered-observation",
            "runtime-walkable-sample", "landmark", "dynamic", "unknown", "__stop__"
        };

        private UnityDecisionBrainFlatModel model;

        public UnityDecisionBrainMode RequestedMode { get; private set; }
        public string RequestedModeName { get { return RequestedMode.ToString().ToLowerInvariant(); } }
        public string PolicyAuthority { get; private set; }
        public string ModelStatus { get; private set; }
        public string ModelDetail { get; private set; }
        public string ModelSha256 { get; private set; }
        public string ModelSchemaVersion { get; private set; }
        public string ModelDataStatus { get; private set; }
        public bool ResearchOverride { get; private set; }
        public bool CanInfer { get; private set; }
        public bool FailClosed { get; private set; }
        public bool GoldenSelfTestPassed { get; private set; }
        public bool AllInferenceLogitsFinite { get; private set; }

        public UnityDecisionBrainRuntime()
        {
            RequestedMode = UnityDecisionBrainMode.Rule;
            PolicyAuthority = "rule";
            ModelStatus = "not_requested";
            ModelDetail = "Rule mode does not load a decision model.";
            ModelSha256 = "";
            ModelSchemaVersion = "";
            ModelDataStatus = "";
            GoldenSelfTestPassed = false;
            AllInferenceLogitsFinite = true;
        }

        public void ConfigureFromCommandLine()
        {
            string rawMode = ReadArgument("--decision-brain-mode=");
            if (string.IsNullOrWhiteSpace(rawMode)) rawMode = "rule";
            if (string.Equals(rawMode, "shadow", StringComparison.OrdinalIgnoreCase))
                RequestedMode = UnityDecisionBrainMode.Shadow;
            else if (string.Equals(rawMode, "model", StringComparison.OrdinalIgnoreCase))
                RequestedMode = UnityDecisionBrainMode.Model;
            else if (string.Equals(rawMode, "rule", StringComparison.OrdinalIgnoreCase))
                RequestedMode = UnityDecisionBrainMode.Rule;
            else
                throw new InvalidOperationException("Unsupported --decision-brain-mode: " + rawMode);
            ResearchOverride = HasArgument("--decision-brain-research-override");

            if (RequestedMode == UnityDecisionBrainMode.Rule)
            {
                PolicyAuthority = "rule";
                ModelStatus = "not_requested";
                ModelDetail = "Rule mode selected; model inference was not requested.";
                return;
            }

            string json = null;
            string source = "";
            try
            {
                string externalPath = ReadArgument("--decision-brain-model-path=");
                if (!string.IsNullOrWhiteSpace(externalPath))
                {
                    source = Path.GetFullPath(externalPath);
                    json = File.ReadAllText(source, Encoding.UTF8);
                }
                else
                {
                    TextAsset asset = Resources.Load<TextAsset>("UnityDecisionBrainModel");
                    if (asset != null)
                    {
                        source = "Resources/UnityDecisionBrainModel.json";
                        json = asset.text;
                    }
                }
                if (string.IsNullOrWhiteSpace(json))
                    throw new InvalidOperationException("UnityDecisionBrainModel JSON was not found.");
                ModelSha256 = Sha256(json);
                model = JsonUtility.FromJson<UnityDecisionBrainFlatModel>(json);
                ValidateModel(model);
                ModelSchemaVersion = model.schemaVersion;
                ModelDataStatus = model.dataStatus;
                RunGoldenSelfTest(model);
                GoldenSelfTestPassed = true;

                if (RequestedMode == UnityDecisionBrainMode.Model &&
                    !string.Equals(model.dataStatus, "production_candidate", StringComparison.OrdinalIgnoreCase) &&
                    !ResearchOverride)
                {
                    CanInfer = false;
                    FailClosed = true;
                    PolicyAuthority = "fail_closed";
                    ModelStatus = "fail_closed";
                    ModelDetail = "Non-production model dataStatus=" + model.dataStatus +
                        " requires --decision-brain-research-override";
                    return;
                }
                CanInfer = true;
                FailClosed = false;
                PolicyAuthority = RequestedMode == UnityDecisionBrainMode.Shadow ? "rule_shadow" : "model";
                ModelStatus = "ready";
                ModelDetail = "Validated " + source + "; dataStatus=" + model.dataStatus;
            }
            catch (Exception exception)
            {
                model = null;
                CanInfer = false;
                ModelDetail = exception.GetBaseException().Message;
                if (RequestedMode == UnityDecisionBrainMode.Shadow)
                {
                    PolicyAuthority = "rule_shadow_fallback";
                    ModelStatus = "fallback_rule";
                    FailClosed = false;
                }
                else
                {
                    PolicyAuthority = "fail_closed";
                    ModelStatus = "fail_closed";
                    FailClosed = true;
                }
            }
        }

        public bool TryPredict(
            float simulatedTime, float coverage, int replans, string state,
            string previousSkill, string plannerMode, string decisionKind,
            UnityDecisionCandidate[] candidates, string[] validSkills,
            int activeTargetIndex, out UnityDecisionPrediction prediction,
            out string error)
        {
            prediction = null;
            error = "";
            if (!CanInfer || model == null)
            {
                error = ModelDetail;
                return false;
            }
            try
            {
                if (decisionKind != "target_selection" && decisionKind != "skill_transition")
                    throw new InvalidOperationException("Unsupported decisionKind: " + decisionKind);
                UnityDecisionCandidate[] sceneCandidates = candidates ?? new UnityDecisionCandidate[0];
                int candidateCount = sceneCandidates.Length + 1;
                int stopIndex = candidateCount - 1;
                float[] globalRaw = EncodeGlobal(
                    simulatedTime, coverage, replans, state, previousSkill, plannerMode);
                float[] candidateRaw = new float[candidateCount * model.candidateDim];
                bool[] candidateMask = Enumerable.Repeat(true, candidateCount).ToArray();
                bool[] targetMask = new bool[candidateCount];
                for (int index = 0; index < sceneCandidates.Length; index++)
                {
                    UnityDecisionCandidate candidate = sceneCandidates[index];
                    float[] encoded = EncodeCandidate(candidate);
                    Array.Copy(encoded, 0, candidateRaw, index * model.candidateDim, model.candidateDim);
                    // Learned logits never override runtime reachability/cooldown/visited gates.
                    targetMask[index] = candidate != null && candidate.selectable &&
                        !candidate.visited && !candidate.cooldown &&
                        candidate.reachabilityKnown && candidate.reachable;
                }
                Array.Copy(model.stopCandidateEncodedFeatures, 0, candidateRaw,
                    stopIndex * model.candidateDim, model.candidateDim);
                RequireFinite("runtime global features", globalRaw);
                RequireFinite("runtime candidate features", candidateRaw);
                HashSet<string> valid = new HashSet<string>(
                    validSkills ?? new string[0], StringComparer.Ordinal);
                targetMask[stopIndex] = valid.Contains("Stop");
                if (!targetMask.Any(value => value))
                    throw new InvalidOperationException("Safety target mask contains no legal choice.");

                float[] normalizedGlobal = Normalize(globalRaw, model.globalMean, model.globalStd);
                float[] normalizedCandidates = NormalizeCandidates(
                    candidateRaw, candidateCount, model.candidateDim,
                    model.candidateMean, model.candidateStd);
                RequireFinite("runtime normalized global features", normalizedGlobal);
                RequireFinite("runtime normalized candidate features", normalizedCandidates);
                int conditioningIndex;
                InferenceOutput output = Forward(
                    normalizedGlobal, normalizedCandidates, candidateMask, targetMask,
                    decisionKind == "target_selection" ? -1 :
                        (activeTargetIndex >= 0 ? activeTargetIndex : stopIndex),
                    out conditioningIndex);
                RequireFinite("runtime raw target logits", output.rawTargetLogits);
                RequireFinite("runtime masked target logits", output.maskedTargetLogits);
                RequireFinite("runtime raw skill logits", output.rawSkillLogits);

                bool[] skillMask = BuildSafeSkillMask(
                    valid, sceneCandidates, conditioningIndex, stopIndex, targetMask);
                if (!skillMask.Any(value => value))
                    throw new InvalidOperationException("Safety skill mask contains no legal choice.");
                float[] maskedSkill = Mask(output.rawSkillLogits, skillMask);
                RequireFinite("runtime masked skill logits", maskedSkill);
                int chosenSkill = ArgMax(maskedSkill);
                int chosenTarget = decisionKind == "target_selection"
                    ? ArgMax(output.maskedTargetLogits) : conditioningIndex;
                prediction = new UnityDecisionPrediction
                {
                    targetIndex = chosenTarget == stopIndex ? -1 : chosenTarget,
                    targetId = chosenTarget == stopIndex ? "__STOP__" :
                        sceneCandidates[chosenTarget].targetId,
                    skill = model.skills[chosenSkill],
                    rawTargetLogits = output.rawTargetLogits,
                    maskedTargetLogits = output.maskedTargetLogits,
                    rawSkillLogits = output.rawSkillLogits,
                    maskedSkillLogits = maskedSkill,
                };
                if (prediction.targetIndex < 0 && prediction.skill != "Stop" &&
                    decisionKind == "target_selection")
                    throw new InvalidOperationException(
                        "Model selected virtual STOP target with a non-Stop skill.");
                return true;
            }
            catch (Exception exception)
            {
                error = exception.GetBaseException().Message;
                if (error.IndexOf("non-finite", StringComparison.OrdinalIgnoreCase) >= 0)
                    AllInferenceLogitsFinite = false;
                prediction = null;
                return false;
            }
        }

        private bool[] BuildSafeSkillMask(
            HashSet<string> valid, UnityDecisionCandidate[] candidates,
            int conditioningIndex, int stopIndex, bool[] targetMask)
        {
            bool[] mask = new bool[model.skills.Length];
            bool stopCandidate = conditioningIndex == stopIndex;
            UnityDecisionCandidate selected = !stopCandidate && conditioningIndex >= 0 &&
                conditioningIndex < candidates.Length ? candidates[conditioningIndex] : null;
            for (int index = 0; index < model.skills.Length; index++)
            {
                string skill = model.skills[index];
                bool legal = valid.Contains(skill);
                if (skill == "Interact")
                    legal = legal && selected != null && selected.interactionDeclared &&
                        selected.interactionAvailable;
                else if (skill == "Navigate")
                    legal = legal && selected != null && targetMask[conditioningIndex];
                else if (skill == "LookAt" || skill == "Observe")
                    legal = legal && selected != null;
                mask[index] = legal;
            }
            return mask;
        }

        private float[] EncodeGlobal(float time, float coverage, int replans,
            string state, string previousSkill, string plannerMode)
        {
            float[] values = new float[model.globalDim];
            values[0] = time;
            values[1] = coverage;
            values[2] = replans;
            OneHot(values, 3, model.brainStates, state, "Unknown");
            OneHot(values, 11, model.skills.Concat(new[] { "Unknown" }).ToArray(),
                previousSkill, "Unknown");
            OneHot(values, 18, model.plannerModes, plannerMode, "unknown");
            return values;
        }

        private float[] EncodeCandidate(UnityDecisionCandidate candidate)
        {
            if (candidate == null) throw new InvalidOperationException("Null candidate in context.");
            float[] values = new float[model.candidateDim];
            values[0] = candidate.priority;
            values[1] = candidate.relativeX;
            values[2] = candidate.relativeY;
            values[3] = candidate.relativeZ;
            values[4] = candidate.distance;
            values[5] = candidate.bearingSin;
            values[6] = candidate.bearingCos;
            values[7] = candidate.cooldownSeconds;
            values[8] = Bool(candidate.resolved);
            values[9] = Bool(candidate.reachabilityKnown);
            values[10] = Bool(candidate.reachable);
            values[11] = Bool(candidate.visibilityKnown);
            values[12] = Bool(candidate.visible);
            values[13] = Bool(candidate.navMeshProjected);
            values[14] = Bool(candidate.visited);
            values[15] = Bool(candidate.interactionDeclared);
            values[16] = Bool(candidate.interactionAvailable);
            values[17] = Bool(candidate.interactionRequired);
            values[18] = Bool(candidate.interactionAttempted);
            values[19] = Bool(candidate.interactionVerified);
            values[20] = Bool(candidate.riskCandidate);
            string semantic = model.semanticTypes.Contains(candidate.semanticType)
                ? candidate.semanticType : "unknown";
            OneHot(values, 21, model.semanticTypes, semantic, "unknown");
            return values;
        }

        private InferenceOutput Forward(float[] global, float[] candidateValues,
            bool[] candidateMask, bool[] targetMask, int requestedConditioningIndex,
            out int conditioningIndex)
        {
            int count = candidateMask.Length;
            float[] global1 = Relu(Linear(global,
                model.globalEncoderLinear1InputSize, model.globalEncoderLinear1OutputSize,
                model.globalEncoderLinear1Weights, model.globalEncoderLinear1Bias));
            float[] global2 = Relu(Linear(global1,
                model.globalEncoderLinear2InputSize, model.globalEncoderLinear2OutputSize,
                model.globalEncoderLinear2Weights, model.globalEncoderLinear2Bias));
            int encodedDim = model.candidateEncoderLinear2OutputSize;
            float[] encodedCandidates = new float[count * encodedDim];
            for (int index = 0; index < count; index++)
            {
                float[] raw = Slice(candidateValues, index * model.candidateDim, model.candidateDim);
                float[] first = Relu(Linear(raw,
                    model.candidateEncoderLinear1InputSize, model.candidateEncoderLinear1OutputSize,
                    model.candidateEncoderLinear1Weights, model.candidateEncoderLinear1Bias));
                float[] second = Relu(Linear(first,
                    model.candidateEncoderLinear2InputSize, model.candidateEncoderLinear2OutputSize,
                    model.candidateEncoderLinear2Weights, model.candidateEncoderLinear2Bias));
                Array.Copy(second, 0, encodedCandidates, index * encodedDim, encodedDim);
            }
            float[] mean = new float[encodedDim];
            float[] max = Enumerable.Repeat(float.NegativeInfinity, encodedDim).ToArray();
            int validCount = 0;
            for (int index = 0; index < count; index++)
            {
                if (!candidateMask[index]) continue;
                validCount++;
                for (int feature = 0; feature < encodedDim; feature++)
                {
                    float value = encodedCandidates[index * encodedDim + feature];
                    mean[feature] += value;
                    if (value > max[feature]) max[feature] = value;
                }
            }
            if (validCount < 1) throw new InvalidOperationException("No present candidates.");
            for (int feature = 0; feature < encodedDim; feature++) mean[feature] /= validCount;
            float[] rawTargets = new float[count];
            for (int index = 0; index < count; index++)
            {
                float[] joint = Joint(global2,
                    Slice(encodedCandidates, index * encodedDim, encodedDim), mean, max);
                float[] hidden = Relu(Linear(joint,
                    model.targetHeadLinear1InputSize, model.targetHeadLinear1OutputSize,
                    model.targetHeadLinear1Weights, model.targetHeadLinear1Bias));
                rawTargets[index] = Linear(hidden,
                    model.targetHeadLinear2InputSize, model.targetHeadLinear2OutputSize,
                    model.targetHeadLinear2Weights, model.targetHeadLinear2Bias)[0];
            }
            float[] maskedTargets = Mask(rawTargets, targetMask);
            conditioningIndex = requestedConditioningIndex < 0
                ? ArgMax(maskedTargets) : requestedConditioningIndex;
            if (conditioningIndex < 0 || conditioningIndex >= count || !candidateMask[conditioningIndex])
                throw new InvalidOperationException("Skill conditioning target is not a present candidate.");
            float[] selected = Slice(encodedCandidates, conditioningIndex * encodedDim, encodedDim);
            float[] skillJoint = Joint(global2, selected, mean, max);
            float[] skillHidden = Relu(Linear(skillJoint,
                model.skillHeadLinear1InputSize, model.skillHeadLinear1OutputSize,
                model.skillHeadLinear1Weights, model.skillHeadLinear1Bias));
            float[] rawSkills = Linear(skillHidden,
                model.skillHeadLinear2InputSize, model.skillHeadLinear2OutputSize,
                model.skillHeadLinear2Weights, model.skillHeadLinear2Bias);
            return new InferenceOutput
            {
                rawTargetLogits = rawTargets,
                maskedTargetLogits = maskedTargets,
                rawSkillLogits = rawSkills,
            };
        }

        private void ValidateModel(UnityDecisionBrainFlatModel value)
        {
            if (value == null || value.schemaVersion != ExpectedSchema)
                throw new InvalidOperationException("Unsupported Unity decision model schema.");
            if (value.dataStatus != "smoke_only" && value.dataStatus != "trained" &&
                value.dataStatus != "production_candidate")
                throw new InvalidOperationException("Unsupported decision model dataStatus: " + value.dataStatus);
            RequireSequence("globalFeatureNames", value.globalFeatureNames, ExpectedGlobalFeatures);
            RequireSequence("candidateFeatureNames", value.candidateFeatureNames, ExpectedCandidateFeatures);
            RequireSequence("skills", value.skills, ExpectedSkills);
            RequireSequence("brainStates", value.brainStates, ExpectedStates);
            RequireSequence("plannerModes", value.plannerModes, ExpectedPlannerModes);
            RequireSequence("semanticTypes", value.semanticTypes, ExpectedSemanticTypes);
            if (value.globalDim != ExpectedGlobalFeatures.Length ||
                value.candidateDim != ExpectedCandidateFeatures.Length ||
                value.skillCount != ExpectedSkills.Length)
                throw new InvalidOperationException("Exported model dimensions do not match its frozen schema.");
            RequireVector("globalMean", value.globalMean, value.globalDim);
            RequirePositiveVector("globalStd", value.globalStd, value.globalDim);
            RequireVector("candidateMean", value.candidateMean, value.candidateDim);
            RequirePositiveVector("candidateStd", value.candidateStd, value.candidateDim);
            RequireVector("stopCandidateEncodedFeatures", value.stopCandidateEncodedFeatures, value.candidateDim);
            ValidateLayer("globalEncoderLinear1", value.globalEncoderLinear1InputSize,
                value.globalEncoderLinear1OutputSize, value.globalEncoderLinear1Weights,
                value.globalEncoderLinear1Bias, value.globalDim, 32);
            ValidateLayer("globalEncoderLinear2", value.globalEncoderLinear2InputSize,
                value.globalEncoderLinear2OutputSize, value.globalEncoderLinear2Weights,
                value.globalEncoderLinear2Bias, 32, 32);
            ValidateLayer("candidateEncoderLinear1", value.candidateEncoderLinear1InputSize,
                value.candidateEncoderLinear1OutputSize, value.candidateEncoderLinear1Weights,
                value.candidateEncoderLinear1Bias, value.candidateDim, 48);
            ValidateLayer("candidateEncoderLinear2", value.candidateEncoderLinear2InputSize,
                value.candidateEncoderLinear2OutputSize, value.candidateEncoderLinear2Weights,
                value.candidateEncoderLinear2Bias, 48, 48);
            ValidateLayer("targetHeadLinear1", value.targetHeadLinear1InputSize,
                value.targetHeadLinear1OutputSize, value.targetHeadLinear1Weights,
                value.targetHeadLinear1Bias, 176, 64);
            ValidateLayer("targetHeadLinear2", value.targetHeadLinear2InputSize,
                value.targetHeadLinear2OutputSize, value.targetHeadLinear2Weights,
                value.targetHeadLinear2Bias, 64, 1);
            ValidateLayer("skillHeadLinear1", value.skillHeadLinear1InputSize,
                value.skillHeadLinear1OutputSize, value.skillHeadLinear1Weights,
                value.skillHeadLinear1Bias, 176, 64);
            ValidateLayer("skillHeadLinear2", value.skillHeadLinear2InputSize,
                value.skillHeadLinear2OutputSize, value.skillHeadLinear2Weights,
                value.skillHeadLinear2Bias, 64, value.skillCount);
        }

        private void RunGoldenSelfTest(UnityDecisionBrainFlatModel value)
        {
            if (value.goldenCandidateCount < 1)
                throw new InvalidOperationException("Golden parity fixture dimensions are invalid.");
            RequireVector("goldenGlobalFeatures", value.goldenGlobalFeatures, value.globalDim);
            RequireVector("goldenNormalizedGlobalFeatures", value.goldenNormalizedGlobalFeatures, value.globalDim);
            RequireVector("goldenCandidateFeatures", value.goldenCandidateFeatures,
                value.goldenCandidateCount * value.candidateDim);
            RequireVector("goldenNormalizedCandidateFeatures", value.goldenNormalizedCandidateFeatures,
                value.goldenCandidateCount * value.candidateDim);
            RequireBoolVector("goldenCandidateMask", value.goldenCandidateMask, value.goldenCandidateCount);
            RequireBoolVector("goldenTargetMask", value.goldenTargetMask, value.goldenCandidateCount);
            RequireBoolVector("goldenSkillMask", value.goldenSkillMask, value.skillCount);
            float tolerance = Mathf.Max(1.0e-6f, value.goldenTolerance);
            float[] normalizedGlobal = Normalize(
                value.goldenGlobalFeatures, value.globalMean, value.globalStd);
            float[] normalizedCandidates = NormalizeCandidates(
                value.goldenCandidateFeatures, value.goldenCandidateCount, value.candidateDim,
                value.candidateMean, value.candidateStd);
            RequireClose("golden normalized global", normalizedGlobal,
                value.goldenNormalizedGlobalFeatures, tolerance);
            RequireClose("golden normalized candidates", normalizedCandidates,
                value.goldenNormalizedCandidateFeatures, tolerance);
            int conditioning;
            InferenceOutput output = Forward(normalizedGlobal, normalizedCandidates,
                value.goldenCandidateMask, value.goldenTargetMask,
                value.goldenSkillConditioningTargetIndex, out conditioning);
            RequireClose("golden raw target", output.rawTargetLogits,
                value.goldenExpectedRawTargetLogits, tolerance);
            RequireClose("golden masked target", output.maskedTargetLogits,
                value.goldenExpectedMaskedTargetLogits, tolerance);
            RequireClose("golden raw skill", output.rawSkillLogits,
                value.goldenExpectedRawSkillLogits, tolerance);
            RequireClose("golden masked skill", Mask(output.rawSkillLogits, value.goldenSkillMask),
                value.goldenExpectedMaskedSkillLogits, tolerance);
        }

        private static void ValidateLayer(string name, int input, int output,
            float[] weights, float[] bias, int expectedInput, int expectedOutput)
        {
            if (input != expectedInput || output != expectedOutput)
                throw new InvalidOperationException(name + " shape metadata is invalid.");
            RequireVector(name + "Weights", weights, input * output);
            RequireVector(name + "Bias", bias, output);
        }

        private static float[] Linear(float[] input, int inputSize, int outputSize,
            float[] weights, float[] bias)
        {
            if (input.Length != inputSize) throw new InvalidOperationException("Linear input mismatch.");
            float[] output = new float[outputSize];
            for (int row = 0; row < outputSize; row++)
            {
                double sum = bias[row];
                int offset = row * inputSize;
                for (int column = 0; column < inputSize; column++)
                    sum += weights[offset + column] * input[column];
                output[row] = (float)sum;
            }
            return output;
        }

        private static float[] Relu(float[] values)
        {
            for (int index = 0; index < values.Length; index++)
                if (values[index] < 0f) values[index] = 0f;
            return values;
        }

        private static float[] Joint(params float[][] vectors)
        {
            int length = vectors.Sum(item => item.Length);
            float[] result = new float[length];
            int offset = 0;
            foreach (float[] vector in vectors)
            {
                Array.Copy(vector, 0, result, offset, vector.Length);
                offset += vector.Length;
            }
            return result;
        }

        private static float[] Normalize(float[] raw, float[] mean, float[] std)
        {
            float[] output = new float[raw.Length];
            for (int index = 0; index < raw.Length; index++)
                output[index] = (raw[index] - mean[index]) / std[index];
            return output;
        }

        private static float[] NormalizeCandidates(float[] raw, int count, int dim,
            float[] mean, float[] std)
        {
            float[] output = new float[raw.Length];
            for (int candidate = 0; candidate < count; candidate++)
                for (int feature = 0; feature < dim; feature++)
                {
                    int index = candidate * dim + feature;
                    output[index] = (raw[index] - mean[feature]) / std[feature];
                }
            return output;
        }

        private static float[] Mask(float[] values, bool[] mask)
        {
            if (values.Length != mask.Length) throw new InvalidOperationException("Mask length mismatch.");
            float[] output = new float[values.Length];
            for (int index = 0; index < values.Length; index++)
                output[index] = mask[index] ? values[index] : MaskedLogit;
            return output;
        }

        private static int ArgMax(float[] values)
        {
            if (values == null || values.Length == 0) throw new InvalidOperationException("ArgMax on empty values.");
            int best = 0;
            for (int index = 1; index < values.Length; index++)
                if (values[index] > values[best]) best = index;
            return best;
        }

        private static float[] Slice(float[] values, int offset, int length)
        {
            float[] output = new float[length];
            Array.Copy(values, offset, output, 0, length);
            return output;
        }

        private static void OneHot(float[] destination, int offset, string[] vocabulary,
            string value, string unknown)
        {
            int index = Array.IndexOf(vocabulary, value ?? "");
            if (index < 0) index = Array.IndexOf(vocabulary, unknown);
            if (index < 0) throw new InvalidOperationException("One-hot vocabulary lacks unknown token.");
            destination[offset + index] = 1f;
        }

        private static float Bool(bool value) { return value ? 1f : 0f; }

        private static void RequireSequence(string name, string[] actual, string[] expected)
        {
            if (actual == null || !actual.SequenceEqual(expected))
                throw new InvalidOperationException(name + " order differs from the frozen runtime schema.");
        }

        private static void RequireVector(string name, float[] values, int expected)
        {
            if (values == null || values.Length != expected)
                throw new InvalidOperationException(name + " length mismatch.");
            if (values.Any(value => float.IsNaN(value) || float.IsInfinity(value)))
                throw new InvalidOperationException(name + " contains a non-finite value.");
        }

        private static void RequireFinite(string name, float[] values)
        {
            if (values == null || values.Any(value => float.IsNaN(value) || float.IsInfinity(value)))
                throw new InvalidOperationException(name + " contains a non-finite value.");
        }

        private static void RequirePositiveVector(string name, float[] values, int expected)
        {
            RequireVector(name, values, expected);
            if (values.Any(value => value <= 0f))
                throw new InvalidOperationException(name + " contains a non-positive scale.");
        }

        private static void RequireBoolVector(string name, bool[] values, int expected)
        {
            if (values == null || values.Length != expected)
                throw new InvalidOperationException(name + " length mismatch.");
        }

        private static void RequireClose(string name, float[] actual, float[] expected, float tolerance)
        {
            RequireVector(name + " expected", expected, actual.Length);
            for (int index = 0; index < actual.Length; index++)
            {
                float scale = Mathf.Max(1f, Mathf.Abs(expected[index]));
                if (Mathf.Abs(actual[index] - expected[index]) > tolerance * scale)
                    throw new InvalidOperationException(name + " parity failed at index " + index +
                        ": unity=" + actual[index].ToString("R", CultureInfo.InvariantCulture) +
                        ", python=" + expected[index].ToString("R", CultureInfo.InvariantCulture));
            }
        }

        private static string Sha256(string text)
        {
            using (SHA256 hash = SHA256.Create())
            {
                byte[] bytes = hash.ComputeHash(Encoding.UTF8.GetBytes(text));
                StringBuilder result = new StringBuilder(bytes.Length * 2);
                foreach (byte value in bytes) result.Append(value.ToString("x2"));
                return result.ToString();
            }
        }

        private static bool HasArgument(string value)
        {
            return Environment.GetCommandLineArgs().Any(argument =>
                string.Equals(argument, value, StringComparison.OrdinalIgnoreCase));
        }

        private static string ReadArgument(string prefix)
        {
            foreach (string argument in Environment.GetCommandLineArgs())
                if (argument.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                    return argument.Substring(prefix.Length).Trim('"');
            return null;
        }

        private sealed class InferenceOutput
        {
            public float[] rawTargetLogits;
            public float[] maskedTargetLogits;
            public float[] rawSkillLogits;
        }
    }
}
