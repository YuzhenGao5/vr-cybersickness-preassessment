"""Fixed event-level score-regression exploration; no metric-driven tuning."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import sys
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import scipy
import sklearn
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
NAME = "score_regression_pilot_2026-10-10"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def estimator(kind, cfg):
    steps = [("log1p", FunctionTransformer(np.log1p)),
             ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
             ("variance", VarianceThreshold(threshold=1e-10))]
    if kind == "ridge":
        steps += [("scale", StandardScaler()), ("regressor", Ridge(**cfg["models"][kind]))]
    else:
        steps += [("regressor", LGBMRegressor(**cfg["models"][kind], random_state=cfg["random_seed"], verbosity=-1))]
    return Pipeline(steps)


def session_weights(frame):
    counts = frame.groupby("session_id")["session_id"].transform("size").to_numpy(dtype=float)
    weights = 1.0 / counts
    return weights / weights.mean()


def metrics(y, predicted):
    rank = None
    if len(y) > 1 and np.ptp(y) > 0 and np.ptp(predicted) > 0:
        rank = float(spearmanr(y, predicted).statistic)
    return dict(mae=float(mean_absolute_error(y, predicted)),
                rmse=float(np.sqrt(mean_squared_error(y, predicted))),
                r2=float(r2_score(y, predicted)) if len(y) > 1 and np.ptp(y) > 0 else None,
                spearman=rank)


def confidence_improvement(frame, baseline, seed):
    """Paired held-out error bootstrap by evaluation group, not IID event rows."""
    ids = frame["event_id"].tolist()
    other = baseline.set_index("event_id").loc[ids]
    improvement = np.abs(other["score_value"].to_numpy() - other["prediction"].to_numpy()) - frame["absolute_error"].to_numpy()
    work = pd.DataFrame({"group": frame["held_out_group"].to_numpy(), "delta": improvement})
    values = work.groupby("group")["delta"].agg(["sum", "count"]).to_numpy()
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(values), size=(1000, len(values)))
    resampled = values[draw]
    results = resampled[:, :, 0].sum(axis=1) / resampled[:, :, 1].sum(axis=1)
    return dict(mae_improvement_over_mean=float(improvement.mean()),
                improvement_ci_low=float(np.percentile(results, 2.5)),
                improvement_ci_high=float(np.percentile(results, 97.5)),
                improvement_ci_method="1000 evaluation-group bootstrap draws of fixed OOF errors; not model-refit uncertainty")


def value(x):
    return "NA" if x is None or not np.isfinite(x) else f"{x:.4f}"


def md_table(rows, columns, labels=None):
    labels = labels or columns
    lines = ["| " + " | ".join(labels) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = [value(row[c]) if isinstance(row[c], (float, int, np.floating)) else str(row[c]) for c in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / f"{NAME}.json")
    parser.add_argument("--input", type=Path, default=ROOT / "features" / NAME / "training_matrix.csv")
    args = parser.parse_args()
    root = args.project_root.resolve()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    result_dir = root / "results" / NAME
    model_dir = root / "models" / NAME
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(NAME)
    log.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s")
    for handler in [logging.StreamHandler(sys.stdout), logging.FileHandler(result_dir / "training_log.txt", encoding="utf-8")]:
        handler.setFormatter(formatter)
        log.addHandler(handler)
    data = pd.read_csv(args.input, encoding="utf-8-sig")
    required = {"event_id", "session_id", "participant_id", "game_name", "score_value"}
    assert required <= set(data.columns)
    assert data["event_id"].is_unique
    assert data["score_value"].notna().all()
    assert data["participant_id"].notna().all() and not (data["participant_id"] == "unknown").any()
    assert data["game_name"].nunique() >= 2 and data["participant_id"].nunique() >= 2
    all_features = list(dict.fromkeys(c for columns in cfg["feature_sets"].values() for c in columns))
    assert set(all_features) <= set(data.columns)
    data[all_features] = data[all_features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    assert not (data[all_features] < 0).any().any(), "nonnegative log1p input contract"
    variants = [("train_mean", None, None), ("train_median", None, None)]
    variants += [(f"{feature_set}_{kind}", feature_set, kind) for feature_set in cfg["feature_sets"] for kind in ["ridge", "lightgbm"]]
    fold_rows, prediction_rows = [], []
    active_counts = {}
    y_all = data["score_value"].to_numpy(dtype=float)
    for protocol, group_column in [("leave_game_out", "game_name"), ("leave_subject_out", "participant_id")]:
        groups = data[group_column].to_numpy()
        for fold, (train_idx, test_idx) in enumerate(LeaveOneGroupOut().split(data, y_all, groups), 1):
            train, test = data.iloc[train_idx], data.iloc[test_idx]
            weights = session_weights(train)
            heldout = str(groups[test_idx[0]])
            subject_overlap = len(set(train["participant_id"]) & set(test["participant_id"]))
            game_overlap = len(set(train["game_name"]) & set(test["game_name"]))
            assert not (set(train["session_id"]) & set(test["session_id"]))
            if protocol == "leave_game_out":
                assert game_overlap == 0
            else:
                assert subject_overlap == 0
            for name, feature_set, kind in variants:
                if name == "train_mean":
                    prediction = np.full(len(test), np.average(y_all[train_idx], weights=weights))
                    active = []
                elif name == "train_median":
                    order = np.argsort(y_all[train_idx])
                    ordered_y, ordered_w = y_all[train_idx][order], weights[order]
                    median = ordered_y[np.searchsorted(np.cumsum(ordered_w), ordered_w.sum() / 2)]
                    prediction = np.full(len(test), median)
                    active = []
                else:
                    features = cfg["feature_sets"][feature_set]
                    pipe = estimator(kind, cfg)
                    pipe.fit(train[features], y_all[train_idx], regressor__sample_weight=weights)
                    prediction = pipe.predict(test[features])
                    active = np.array(features)[pipe.named_steps["variance"].get_support()].tolist()
                    active_counts.setdefault((protocol, name), []).append(active)
                assert np.isfinite(prediction).all()
                measure = metrics(y_all[test_idx], prediction)
                fold_rows.append(dict(protocol=protocol, model=name, fold=fold, held_out_group=heldout,
                                      train_events=len(train), test_events=len(test),
                                      test_sessions=test["session_id"].nunique(), test_score_unique=test["score_value"].nunique(),
                                      participant_overlap=subject_overlap, game_overlap=game_overlap,
                                      active_features_json=json.dumps(active), **measure))
                for index, predicted in zip(test_idx, prediction):
                    original = data.iloc[index]
                    prediction_rows.append(dict(protocol=protocol, model=name, fold=fold, held_out_group=heldout,
                                                event_id=original["event_id"], session_id=original["session_id"],
                                                participant_id=original["participant_id"], game_name=original["game_name"],
                                                score_value=float(original["score_value"]), prediction=float(predicted),
                                                absolute_error=abs(float(predicted) - float(original["score_value"]))))
            log.info("%s fold %s/%s: %s, %s test events", protocol, fold, len(set(groups)), heldout, len(test))
    folds, predictions = pd.DataFrame(fold_rows), pd.DataFrame(prediction_rows)
    predictions.to_csv(result_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    folds.to_csv(result_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    summaries = []
    for (protocol, name), frame in predictions.groupby(["protocol", "model"], sort=False):
        assert len(frame) == len(data) and frame["event_id"].is_unique
        relevant = folds[(folds["protocol"] == protocol) & (folds["model"] == name)]
        baseline = predictions[(predictions["protocol"] == protocol) & (predictions["model"] == "train_mean")]
        summaries.append(dict(protocol=protocol, model=name, events=len(frame), folds=len(relevant),
                              **metrics(frame["score_value"].to_numpy(), frame["prediction"].to_numpy()),
                              mean_fold_mae=float(relevant["mae"].mean()),
                              equal_game_mae=float(frame.groupby("game_name")["absolute_error"].mean().mean()),
                              equal_session_mae=float(frame.groupby("session_id")["absolute_error"].mean().mean()),
                              outside_observed_range=int(((frame["prediction"] < y_all.min()) | (frame["prediction"] > y_all.max())).sum()),
                              **confidence_improvement(frame, baseline, cfg["random_seed"])))
    pd.DataFrame(summaries).to_csv(result_dir / "metrics.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    models, importance = {}, []
    for name, feature_set, kind in variants[2:]:
        features = cfg["feature_sets"][feature_set]
        pipe = estimator(kind, cfg)
        pipe.fit(data[features], y_all, regressor__sample_weight=session_weights(data))
        active = np.array(features)[pipe.named_steps["variance"].get_support()].tolist()
        trained = pipe.named_steps["regressor"]
        weights = trained.booster_.feature_importance(importance_type="gain") if kind == "lightgbm" else trained.coef_
        for feature, weight in zip(active, weights):
            importance.append(dict(model=name, feature=feature, importance=float(weight),
                                   type="full_fit_tree_gain_not_SHAP" if kind == "lightgbm" else "standardized_log1p_coefficient_not_causal"))
        models[name] = dict(model_path=str(model_dir / f"{name}.pkl"), requested_features=features,
                            active_features=active, dropped_constant_features=[f for f in features if f not in active])
        joblib.dump(dict(pipeline=pipe, feature_names=features, required_input_features=active,
                         target="modern original MS_rating", observed_target_range=[float(y_all.min()), float(y_all.max())],
                         config=cfg, training_matrix_sha256=digest(args.input),
                         warning="exploratory runtime-state rating association; not certified pure design, individualized prediction or intervention effect"),
                    model_dir / f"{name}.pkl")
    pd.DataFrame(importance).to_csv(result_dir / "feature_importance.csv", index=False, encoding="utf-8-sig", float_format="%.12g")
    info = dict(config_sha256=digest(args.config), training_matrix_sha256=digest(args.input),
                events=len(data), sessions=int(data["session_id"].nunique()), games=int(data["game_name"].nunique()),
                participants=int(data["participant_id"].nunique()),
                score_distribution={str(k): int(v) for k, v in data["score_value"].value_counts().sort_index().items()},
                models=models, metrics=summaries,
                environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                                 numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                                 lightgbm=lightgbm.__version__, scipy=scipy.__version__, joblib=joblib.__version__),
                folds_with_participant_overlap=int(folds[(folds.protocol == "leave_game_out") & (folds.model == "train_mean")]["participant_overlap"].gt(0).sum()),
                fitted_full_data_models_not_independent_validation=True,
                original_protocol_auc_gate="not_evaluated")
    (result_dir / "training_manifest.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# VR.net 真实评分回归：本地探索性试验", "", "**探索性，非最终结论。输入是运行时场景候选/混合相机特征，未认证为纯设计参数。**", "",
             "## 样本与标签", "", f"- 实际训练表：{len(data)}次真实评分，{info['sessions']} sessions，{info['games']}游戏，{info['participants']}参与者编号。",
             f"- 原始评分分布：{info['score_distribution']}；没有分档、沿用前一评分或生成规则标签。",
             "- 每个样本：评分之前60秒历史的统计量；短历史与覆盖条件见准备manifest及window_quality.csv。",
             "- 同一人/同一session有重复评分；窗口可能重叠，事件数不等于独立样本数。", "",
             "## 实际输入与输出", "", "- 主场景候选集合：等效FOV、可见渲染对象计数、同名唯一对象的运行速度统计、累计体验时长。",
             "- 混合诊断集合额外包括总相机速度、标量速度变化率、旋转率；这些可能包含头动，不属于已认证纯设计设置。",
             "- FOV按0.001度分辨率取值，常量由训练折移除。线性运动单位为源坐标单位/s，不称m/s。",
             "- 模型输出：连续的原始MS_rating预测值；不截断到1–5，不保证校准。不是个体诊断或已验证的修改收益。", "",
             "| 模型 | 实际保留输入 | 去除的常量 |", "|---|---|---|"]
    for name, details in models.items():
        lines.append(f"| {name} | {', '.join(details['active_features'])} | {', '.join(details['dropped_constant_features']) or '无'} |")
    lines += ["", "## 固定训练与评估", "", "- Ridge alpha=10；LightGBM100棵浅树，参数在本轮指标产生前固定，种子42。",
              "- 非负输入固定log1p，补缺/去常量/线性模型标准化仅拟合训练折；没有对测试游戏做z-score。",
              "- 训练中各session等权，防止高频报告session支配；折外同时给事件、session、游戏等权误差。",
              f"- 留游戏11折中{info['folds_with_participant_overlap']}折有参与者与训练侧重叠；另做参与者留出，两个协议回答不同泛化问题。",
              "- 没有调整参数选择最优测试结果；固定六个可训练模型及两个常数对照全部报告。", "",
              "## 折外结果", "", "MAE/RMSE越低越好；R²低于0表示弱于全体评分均值的参考解释度。评分为有序等级，回归误差隐含等距近似。", ""]
    lines += md_table(summaries, ["protocol", "model", "mae", "rmse", "r2", "spearman", "equal_game_mae", "equal_session_mae", "mae_improvement_over_mean"],
                      ["留出协议", "模型", "MAE", "RMSE", "R²", "Spearman", "游戏等权MAE", "session等权MAE", "较训练均分MAE改善"])
    lines += ["", "## 改善的不确定性", "", "使用1000次留出分组重采样，计算固定折外预测相对训练均分基线的MAE改善。正值表示改善；区间包含0表示本次误差尚不能排除没有改善。仅留游戏11组，且LOGO可能有参与者重叠；该区间不是重训模型的完整不确定性。", ""]
    lines += md_table(summaries, ["protocol", "model", "mae_improvement_over_mean", "improvement_ci_low", "improvement_ci_high"])
    lines += ["", "## 留游戏各折", ""]
    game_rows = [r for r in fold_rows if r["protocol"] == "leave_game_out"]
    lines += md_table(game_rows, ["held_out_group", "model", "test_events", "mae", "rmse", "spearman", "participant_overlap"])
    lines += ["", "## 使用与限制", "", "- 保存的完整数据拟合模型用于复现和后续流程；报告性能来自分组折外预测，不用训练集拟合度证明效果。",
              "- 输入为数值CSV，每行一个对应60秒历史与体验时长；Unity项目需先按相同定义导出特征，不能将开发者设置直接代入运行遥测而声称已验证。",
              "- 对象计数受可见性筛选影响；对象速度依赖同名唯一匹配，坐标跳变与身份噪声尚未获得引擎内真值验证。",
              "- 时间输入是实际累计暴露，不是已认证的作者预设时长。游戏/参与者/帧号/Unix时间/历史评分/交互字段未进入模型。",
              "- 特征重要性不是SHAP，不构成因果证明；不能据此承诺降低FOV或速度会降低多少评分。",
              "- 本次连续评分补充分析不计算分类AUC，不将原PROTOCOL门控宣称通过。", "",
              "## 产出与复现", "", "- training_matrix.csv、window_quality.csv、excluded_events.csv、preparation_manifest.json：输入准备与排除追溯。",
              "- metrics.csv、fold_metrics.csv、oof_predictions.csv：全部实际折外结果。",
              "- training_manifest.json：模型、实际输入、来源哈希、依赖版本。",
              "- feature_importance.csv：全量拟合增益/线性系数，明确不称SHAP。",
              "- 训练命令：`.venv/Scripts/python.exe -X utf8 scripts/train_score_regression_pilot.py`。", ""]
    (result_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("completed: %s", json.dumps({k: info[k] for k in ["events", "sessions", "games", "participants"]}))
    for row in summaries:
        log.info("%s %s MAE %.4f RMSE %.4f Spearman %s", row["protocol"], row["model"], row["mae"], row["rmse"], value(row["spearman"]))


if __name__ == "__main__":
    main()
