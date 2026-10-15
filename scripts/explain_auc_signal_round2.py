#!/usr/bin/env python3
"""Explain the saved full-fit round2 classifier; no fitting or model selection.

Positive-class SHAP must add to the actual estimator output before files are
reported as successful. Unsupported models/shapes and dependency errors are
recorded explicitly. The local long CSV contains only the ten largest features.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
NAME = "auc_signal_round2_2026-10-10"
IDS = ["game_name", "session_id", "participant_id", "rating_event_id"]


def sha256(path):
    from hashlib import sha256 as calculate
    result = calculate()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def save_json(path, content):
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def identity_matches(left, right):
    return len(left) == len(right) and all(np.array_equal(left[name].to_numpy(), right[name].to_numpy())
                                          for name in [*IDS, "ms_rating"])


def expected_scalar(expected, positive_index, output_count=None):
    array = np.asarray(expected, dtype=float).reshape(-1)
    if len(array) == 1:
        return float(array[0])
    if len(array) == 2 and output_count in {None, 2}:
        return float(array[positive_index])
    raise ValueError(f"unsupported SHAP expected_value shape: {np.asarray(expected).shape}")


def positive_shap(values, expected, n_events, n_features, positive_index):
    """Select the documented binary output axis, then require additivity later."""
    if isinstance(values, (list, tuple)):
        if len(values) not in {1, 2}:
            raise ValueError(f"unsupported SHAP output list length {len(values)}")
        index = positive_index if len(values) == 2 else 0
        result = np.asarray(values[index], dtype=float)
        base = expected_scalar(expected, positive_index, len(values))
        layout = f"list of {len(values)} output matrices; selected index {index}"
    else:
        array = np.asarray(values, dtype=float)
        if array.ndim == 2 and array.shape == (n_events, n_features):
            result, base = array, expected_scalar(expected, positive_index)
            layout = "single binary-output matrix; verified against positive-class output"
        elif array.ndim == 3 and array.shape[:2] == (n_events, n_features) and array.shape[2] in {1, 2}:
            output_count = array.shape[2]
            index = positive_index if output_count == 2 else 0
            result, base = array[:, :, index], expected_scalar(expected, positive_index, output_count)
            layout = f"event,feature,output tensor; selected index {index}"
        elif array.ndim == 3 and array.shape[1:] == (n_events, n_features) and array.shape[0] in {1, 2}:
            output_count = array.shape[0]
            index = positive_index if output_count == 2 else 0
            result, base = array[index], expected_scalar(expected, positive_index, output_count)
            layout = f"output,event,feature tensor; selected index {index}"
        else:
            raise ValueError(f"unsupported SHAP values shape {array.shape}; expected ({n_events},{n_features}) binary explanation")
    if result.shape != (n_events, n_features) or not np.isfinite(result).all() or not np.isfinite(base):
        raise ValueError("selected SHAP values/base are wrong-shaped or nonfinite")
    return result, base, layout


def descriptive_direction(values, contribution):
    valid = np.isfinite(values) & np.isfinite(contribution)
    if int(valid.sum()) < 3:
        return None, "fewer than three observed pairs", int(valid.sum())
    if np.unique(values[valid]).size < 2 or np.unique(contribution[valid]).size < 2:
        return None, "constant feature or contribution", int(valid.sum())
    rho = float(spearmanr(values[valid], contribution[valid]).statistic)
    return (rho if np.isfinite(rho) else None), None, int(valid.sum())


def value_text(value):
    return "不可计算" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)


def render_table(rows):
    titles = ["特征", "mean |SHAP|", "原值/SHAP Spearman", "转换值/SHAP Spearman", "方向"]
    columns = ["feature", "mean_abs_shap", "raw_feature_shap_spearman", "transformed_feature_shap_spearman", "direction"]
    lines = ["| " + " | ".join(titles) + " |", "| " + " | ".join(["---"] * len(titles)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(value_text(row.get(name)).replace("|", "\\|") for name in columns) + " |")
    return "\n".join(lines)


def write_report(path, manifest, top_rows):
    lines = ["# 最终全量模型的特征解释", "", f"状态：**{manifest['status']}**。这是已选最终全量拟合模型的诊断，没有新增模型选择或训练。",
             "", "解释的是已经看过的共同事件；不是折外贡献、独立验证或因果效应。排名不能用于事后修改本轮模型或宣称某参数的设计干预效果。"]
    if manifest["status"] != "complete":
        lines += ["", f"未生成可用SHAP解释：{manifest.get('reason', 'unsupported/error')}。没有替代或编造贡献值。"]
    else:
        detail = manifest["model"]
        validation = manifest["additivity"]
        lines += ["", f"实际输入组：**{detail['selected_input_method']}**；模型族：**{detail['selected_model_family']}**。",
                  f"事件数 {manifest['events']}，最终有效特征数 {manifest['effective_features']}。",
                  "", f"解释器：{manifest['explainer']}。SHAP尺度：**{manifest['output_space']}**。正类固定 MS_rating>2。",
                  f"加和校验：baseline={manifest['baseline']:.8f}；baseline+SHAP之和与实际模型输出一致，最大绝对误差 {validation['maximum_absolute_error']:.3g}，固定容差 atol={validation['absolute_tolerance']}、rtol={validation['relative_tolerance']}。",
                  "", "## Top 10 平均绝对贡献", "", render_table(top_rows),
                  "", "平均绝对贡献表示相对解释器基线的输出变化幅度；不同模型的log odds与概率尺度不可直接比较。Spearman只描述本批特征值与其贡献的单调关系；有交互/相关特征时方向可能依取值与样本而变化，不能视为一条干预规则。缺失原值不参与原值方向计算，转换值方向包含训练中位数补缺后的值。",
                  "", "## 输入含义和限制", "",
                  "相机运动统计来自实际运行轨迹，可能混合头动与导航；对象计数及暴露趋势也来自真实使用过程。头部姿态/速度和控制器属于参与者运行时信息，不是无人设计期输入，可能反映行为或不适反应。累计暴露时间衡量体验进程，不能当作纯场景设计贡献。",
                  "训练只保留实际有变化的列。FOV常量不会被硬填进Top10；实际被移除的FOV列：" + str(manifest.get("dropped_fov_columns", [])) + "。头动和位置相关源单位未独立校准的限制仍适用。",
                  "类别加权模型的predict_proba未经额外校准；即使TreeExplainer的尺度是概率，也不能直接解释为可靠个人风险。LinearExplainer采用全量有效特征作为独立背景；树解释采用模型中的tree-path-dependent背景。两者都解释当前拟合模型，不能分离相关特征的因果作用。",
                  "", "## 输出文件", "",
                  f"Top10逐事件长表：{manifest['local_shap_csv']}；全特征汇总：{manifest['importance_csv']}。",
                  "长表分别保存原始输入值、最终转换值及SHAP；Logistic转换值包含标准化。只输出Top10以控制文件大小，完整校验使用全部有效特征。"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / NAME / "nested_selected.pkl")
    parser.add_argument("--input", type=Path, default=ROOT / "features" / NAME / "events.csv")
    parser.add_argument("--cohort", type=Path, default=ROOT / "results" / NAME / "cohort.csv")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / NAME)
    parser.add_argument("--shap-output", type=Path, default=ROOT / "features" / "shap_auc_signal_round2.csv")
    args = parser.parse_args(argv)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.shap_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.results_dir / "explanation_manifest.json"
    report_path = args.results_dir / "feature_interpretation.md"
    importance_path = args.results_dir / "feature_importance.csv"
    # Do not let an unsupported/failed attempt leave stale successful outputs.
    if manifest_path.exists() or args.shap_output.exists() or importance_path.exists():
        raise ValueError("explanation outputs already exist; use fresh output paths for a new attempt")
    manifest = dict(status="planned", created_utc=datetime.now(timezone.utc).isoformat(),
                    script_sha256=sha256(Path(__file__)), model_path=str(args.model.resolve()),
                    input_path=str(args.input.resolve()), cohort_path=str(args.cohort.resolve()),
                    interpretation="already-seen full-fit diagnostic; not OOF, model selection, independent validation, or causal",
                    top_count=10, additivity_tolerances=dict(atol=1e-6, rtol=1e-5))
    try:
        import shap
        manifest["shap_version"] = shap.__version__
        bundle = joblib.load(args.model)
        manifest.update(model_sha256=sha256(args.model), input_sha256=sha256(args.input), cohort_sha256=sha256(args.cohort))
        if bundle.get("input_sha256") and bundle["input_sha256"] != manifest["input_sha256"]:
            raise ValueError("input differs from saved model input hash")
        if bundle.get("cohort_sha256") and bundle["cohort_sha256"] != manifest["cohort_sha256"]:
            raise ValueError("cohort differs from saved model cohort hash")
        if float(bundle.get("target_threshold", 2.0)) != 2.0:
            raise ValueError("saved binary target differs from MS_rating>2")
        pipe = bundle.get("pipeline")
        if pipe is None:
            raise NotImplementedError("saved model is a prior-only predictor without a fitted classifier")
        columns = bundle["input_feature_cols"]
        fitted_columns = bundle["fitted_feature_cols"]
        cohort = pd.read_csv(args.cohort, dtype={name: str for name in IDS}, encoding="utf-8-sig")
        frame = pd.read_csv(args.input, dtype={name: str for name in IDS}, encoding="utf-8-sig")
        if not cohort["rating_event_id"].is_unique or not frame["rating_event_id"].is_unique:
            raise ValueError("one row per unique rating_event_id required")
        frame = frame.set_index("rating_event_id", drop=False).loc[cohort["rating_event_id"]].reset_index(drop=True)
        if not identity_matches(frame, cohort):
            raise ValueError("model input identities or true scores differ from fixed cohort")
        original = frame[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        transformed = np.asarray(pipe[:-1].transform(original), dtype=float)
        if transformed.shape != (len(frame), len(fitted_columns)) or not np.isfinite(transformed).all():
            raise ValueError("pipeline final feature order/shape is not valid")
        final_model = pipe.named_steps["model"]
        if list(final_model.classes_) != [0, 1]:
            raise NotImplementedError(f"unsupported binary classes {final_model.classes_}; expected [0,1]")
        positive_index = list(final_model.classes_).index(1)
        if isinstance(final_model, LogisticRegression):
            masker = shap.maskers.Independent(transformed, max_samples=len(transformed))
            explainer = shap.LinearExplainer(final_model, masker)
            values = explainer.shap_values(transformed)
            actual_output = np.asarray(final_model.decision_function(transformed), dtype=float)
            explainer_name, output_space = "exact LinearExplainer; independent full-cohort background", "positive-class log odds (decision_function)"
        elif isinstance(final_model, LGBMClassifier):
            explainer = shap.TreeExplainer(final_model, model_output="raw", feature_perturbation="tree_path_dependent")
            values = explainer.shap_values(transformed, check_additivity=False)
            actual_output = np.asarray(final_model.predict(transformed, raw_score=True), dtype=float)
            explainer_name, output_space = "TreeExplainer; tree-path-dependent", "positive-class log odds (LightGBM raw_score)"
        elif isinstance(final_model, ExtraTreesClassifier):
            explainer = shap.TreeExplainer(final_model, model_output="raw", feature_perturbation="tree_path_dependent")
            values = explainer.shap_values(transformed, check_additivity=False)
            actual_output = np.asarray(final_model.predict_proba(transformed)[:, positive_index], dtype=float)
            explainer_name, output_space = "TreeExplainer; tree-path-dependent", "positive-class probability (ExtraTrees raw tree output)"
        else:
            raise NotImplementedError(f"unsupported estimator type {type(final_model).__name__}")
        contributions, base, layout = positive_shap(values, explainer.expected_value, len(frame), len(fitted_columns), positive_index)
        reconstructed = base + contributions.sum(axis=1)
        if actual_output.shape != (len(frame),) or not np.isfinite(actual_output).all():
            raise ValueError("actual positive-class model output is nonfinite/wrong-shaped")
        error = np.abs(reconstructed-actual_output)
        maximum_error = float(error.max())
        valid = bool(np.allclose(reconstructed, actual_output, atol=1e-6, rtol=1e-5))
        manifest["additivity"] = dict(passed=valid, maximum_absolute_error=maximum_error,
                                     mean_absolute_error=float(error.mean()), absolute_tolerance=1e-6, relative_tolerance=1e-5)
        if not valid:
            raise ValueError(f"SHAP additivity failed against {output_space}; maximum absolute error={maximum_error}")
        summary = []
        for index, name in enumerate(fitted_columns):
            raw = original[name].to_numpy(dtype=float)
            raw_rho, raw_reason, observed = descriptive_direction(raw, contributions[:, index])
            transformed_rho, transformed_reason, _ = descriptive_direction(transformed[:, index], contributions[:, index])
            direction = "不可计算" if raw_rho is None else "数值较高时贡献倾向增大" if raw_rho > 0 else "数值较高时贡献倾向减小" if raw_rho < 0 else "没有单调方向"
            summary.append(dict(feature=name, mean_abs_shap=float(np.abs(contributions[:, index]).mean()),
                                mean_shap=float(contributions[:, index].mean()),
                                raw_feature_shap_spearman=raw_rho, transformed_feature_shap_spearman=transformed_rho,
                                raw_observed_events=observed, raw_direction_reason=raw_reason,
                                transformed_direction_reason=transformed_reason, direction=direction))
        summary.sort(key=lambda row: (-row["mean_abs_shap"], row["feature"]))
        for rank, row in enumerate(summary, 1):
            row["rank"] = rank
        top = summary[:10]
        local = []
        for item in top:
            name = item["feature"]
            index = fitted_columns.index(name)
            rows = frame[[*IDS, "ms_rating"]].copy()
            rows["feature"], rows["importance_rank"] = name, item["rank"]
            rows["original_value"] = original[name].to_numpy(dtype=float)
            rows["transformed_value"] = transformed[:, index]
            rows["shap_positive_class"] = contributions[:, index]
            rows["baseline"] = base
            rows["actual_model_output"] = actual_output
            rows["output_space"] = output_space
            local.append(rows)
        pd.DataFrame(summary).to_csv(importance_path, index=False, encoding="utf-8-sig", float_format="%.17g")
        pd.concat(local, ignore_index=True).to_csv(args.shap_output, index=False, encoding="utf-8-sig", float_format="%.17g")
        preprocessing = bundle.get("preprocessing", {})
        dropped = preprocessing.get("dropped_constant", []) + preprocessing.get("dropped_all_missing", [])
        manifest.update(status="complete", model=dict(selected_input_method=bundle.get("selected_input_method", bundle.get("method")),
                        selected_model_family=bundle.get("selected_model_family", bundle.get("model"))),
                        events=len(frame), effective_features=len(fitted_columns), feature_columns=fitted_columns,
                        explainer=explainer_name, output_space=output_space, baseline=base, shap_layout=layout,
                        dropped_fov_columns=[name for name in dropped if "fov" in name.lower()],
                        importance_csv=str(importance_path.resolve()), local_shap_csv=str(args.shap_output.resolve()),
                        local_csv_events=len(frame)*len(top), background="full-cohort independent features for linear; trained tree paths for tree models",
                        finished_utc=datetime.now(timezone.utc).isoformat())
        save_json(manifest_path, manifest)
        write_report(report_path, manifest, top)
        print(json.dumps(dict(status="complete", events=len(frame), features=len(fitted_columns), max_additivity_error=maximum_error,
                              model=manifest["model"], report=str(report_path.resolve())), ensure_ascii=False))
        return 0
    except (ImportError, NotImplementedError) as error:
        manifest.update(status="skipped", reason=f"{type(error).__name__}: {error}", finished_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        manifest.update(status="error", reason=f"{type(error).__name__}: {error}", finished_utc=datetime.now(timezone.utc).isoformat())
    save_json(manifest_path, manifest)
    write_report(report_path, manifest, [])
    print(json.dumps(dict(status=manifest["status"], reason=manifest["reason"], report=str(report_path.resolve())), ensure_ascii=False))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
