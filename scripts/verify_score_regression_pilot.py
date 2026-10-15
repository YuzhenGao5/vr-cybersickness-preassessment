"""Verify saved numerical results and data/model provenance, without refitting."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
NAME = "score_regression_pilot_2026-10-10"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    feature_dir, result_dir = ROOT / "features" / NAME, ROOT / "results" / NAME
    config_path = ROOT / "configs" / f"{NAME}.json"
    data = pd.read_csv(feature_dir / "training_matrix.csv", encoding="utf-8-sig")
    excluded = pd.read_csv(feature_dir / "excluded_events.csv", encoding="utf-8-sig")
    quality = pd.read_csv(feature_dir / "window_quality.csv", encoding="utf-8-sig")
    original = pd.read_csv(ROOT / "data/processed/vrnet_design_score_2026-10-10/score_events.csv", encoding="utf-8-sig")
    original = original[original.dataset_family == "modern"].set_index("event_id")
    manifest = json.loads((result_dir / "training_manifest.json").read_text(encoding="utf-8"))
    prep = json.loads((feature_dir / "preparation_manifest.json").read_text(encoding="utf-8"))
    assert len(data) == manifest["events"] == prep["training_sample_count"]
    assert set(data.event_id).isdisjoint(set(excluded.event_id))
    assert set(data.event_id) | set(excluded.event_id) == set(original.index)
    assert data.event_id.is_unique
    np.testing.assert_array_equal(data.score_value.to_numpy(), original.loc[data.event_id, "score_value"].to_numpy())
    selected = quality[quality.included]
    assert set(selected.event_id) == set(data.event_id)
    assert (selected.first_scene_time >= selected.window_start_time).all()
    assert (selected.last_scene_time < selected.window_end_time).all()
    assert manifest["config_sha256"] == sha(config_path)
    assert manifest["training_matrix_sha256"] == sha(feature_dir / "training_matrix.csv")
    for name, digest in prep["output_hashes"].items():
        assert sha(feature_dir / name) == digest
    predictions = pd.read_csv(result_dir / "oof_predictions.csv", encoding="utf-8-sig")
    measures = pd.read_csv(result_dir / "metrics.csv", encoding="utf-8-sig")
    for row in measures.itertuples():
        p = predictions[(predictions.protocol == row.protocol) & (predictions.model == row.model)]
        assert len(p) == len(data) and p.event_id.is_unique and set(p.event_id) == set(data.event_id)
        np.testing.assert_allclose(row.mae, mean_absolute_error(p.score_value, p.prediction), atol=1e-9)
        np.testing.assert_allclose(row.rmse, np.sqrt(mean_squared_error(p.score_value, p.prediction)), atol=1e-9)
        train_group_map = data.set_index("event_id")
        for group, part in p.groupby("held_out_group"):
            column = "game_name" if row.protocol == "leave_game_out" else "participant_id"
            assert (train_group_map.loc[part.event_id, column] == group).all()
    primary = manifest["models"]["scene_candidates_lightgbm"]
    bundle = joblib.load(primary["model_path"])
    assert bundle["training_matrix_sha256"] == manifest["training_matrix_sha256"]
    assert bundle["required_input_features"] == primary["active_features"]
    assert not any("score" in field or field.endswith("_id") or field.startswith("mixed_") for field in bundle["required_input_features"])
    example = data.iloc[0]
    p = predictions[(predictions.protocol == "leave_game_out") & (predictions.model == "scene_candidates_lightgbm") & (predictions.event_id == example.event_id)].iloc[0]
    details = {f: float(example[f]) for f in bundle["required_input_features"]}
    result = dict(status="passed", events=len(data), excluded_events=len(excluded), protocols=2,
                  models_and_controls_per_protocol=8, metrics_recomputed_from_saved_oof=True,
                  input_and_model_hashes_match=True, primary_inputs=details,
                  first_eligible_oof_example=dict(event_id=example.event_id, game=example.game_name,
                                                  actual_score=float(example.score_value), predicted_score=float(p.prediction)),
                  scope="data/serialization/metric consistency checks, not scientific validation of pure design or intervention efficacy")
    (result_dir / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    primary_oof = predictions[(predictions.protocol == "leave_game_out") & (predictions.model == "scene_candidates_lightgbm")]
    by_score = primary_oof.groupby("score_value").agg(n=("prediction", "size"), prediction_mean=("prediction", "mean"), mae=("absolute_error", "mean"))
    def measure(protocol, model):
        return measures[(measures.protocol == protocol) & (measures.model == model)].iloc[0]
    main_score = measure("leave_game_out", "scene_candidates_lightgbm")
    mean_score = measure("leave_game_out", "train_mean")
    time_score = measure("leave_game_out", "time_only_ridge")
    median_score = measure("leave_game_out", "train_median")
    mixed_score = measure("leave_game_out", "scene_plus_mixed_camera_lightgbm")
    mixed_subject = measure("leave_subject_out", "scene_plus_mixed_camera_lightgbm")
    stable = main_score.improvement_ci_low > 0
    conclusion = "固定OOF误差重采样下改善区间为正，但仍需独立数据验证" if stable else "改善区间含0或为负，未显示稳定增益"
    vs_time = "低于" if main_score.mae < time_score.mae else "未低于"
    lines = ["", "## 本轮判断", "",
             f"固定主模型scene_candidates_lightgbm的留游戏MAE为{main_score.mae:.4f}，训练均分基线{mean_score.mae:.4f}；改善{main_score.mae_improvement_over_mean:.4f}的分组重采样区间为[{main_score.improvement_ci_low:.4f},{main_score.improvement_ci_high:.4f}]，{conclusion}。该模型MAE{vs_time}仅时长Ridge（MAE{time_score.mae:.4f}，RMSE{time_score.rmse:.4f}）。",
             f"加入混合相机运动的LightGBM留游戏MAE为{mixed_score.mae:.4f}；参与者留出MAE{mixed_subject.mae:.4f}，MAE改善区间[{mixed_subject.improvement_ci_low:.4f},{mixed_subject.improvement_ci_high:.4f}]。混合相机特征含头动，不能认证纯设计评分引擎有效。",
             f"预测训练折中位数的对照MAE为{median_score.mae:.4f}，RMSE为{median_score.rmse:.4f}；低分偏多使单一MAE看起来容易改善，需要同时查看RMSE与评分排序能力。", "",
             "| 实际评分 | 次数 | 主模型留游戏预测均值 | MAE |", "|---|---:|---:|---:|"]
    for score, row in by_score.iterrows():
        lines.append(f"| {score:g} | {int(row['n'])} | {row['prediction_mean']:.4f} | {row['mae']:.4f} |")
    lines += ["", "逐评分结果用于检查高分事件的预测是否仍接近低分；少量高分记录不足以认证强烈不适的预测可靠性，不能把窗口数当独立训练条件数。", "",
              "## 实际输入输出示例", "", "选取训练表第一条合格记录，而非按预测效果挑选：", "",
              f"- event：{example.event_id}；游戏{example.game_name}。",
              f"- 5个实际输入：{json.dumps(details, ensure_ascii=False)}。",
              f"- 未见过该游戏的折外预测：{p.prediction:.4f}；实际评分：{example.score_value:g}。",
              "- inference_demo.json另示保存的全量拟合模型如何接收数值输入；它是操作示例，不计入验证指标。", "",
              "## 输出核对", "", "verification.json确认原始评分值未改变、输入/排除集合完整互斥、观测严格早于评分、配置/输入/模型哈希匹配，并从保存的折外预测独立重算MAE/RMSE。", ""]
    report = result_dir / "report.md"
    previous = report.read_text(encoding="utf-8").split("\n## 本轮判断", 1)[0]
    report.write_text(previous + "\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
