#!/usr/bin/env python3
"""Match high-risk SHAP features to exact metrics in the evidence rule library."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
import yaml


FIELDS = ["window_id", "risk_level", "top_feature", "shap_value", "matched_rule", "recommendation", "evidence_strength", "source"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="models/risk_model_vrnet_metadata.json")
    parser.add_argument("--shap", default="features/shap_values_vrnet.csv")
    parser.add_argument("--top5", default="features/shap_top5_vrnet.csv")
    parser.add_argument("--predictions", default="features/risk_predictions_vrnet.csv")
    parser.add_argument("--rules", default="configs/risk_factor_rules.yaml")
    parser.add_argument("--output", default="results/recommendations_vrnet.csv")
    parser.add_argument("--note", default="results/recommendations_vrnet_note.md")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    resolve = lambda value: (root / value).resolve() if not Path(value).is_absolute() else Path(value)
    paths = {name: resolve(value) for name, value in vars(args).items()}
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    with paths["metadata"].open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with paths["rules"].open("r", encoding="utf-8-sig") as handle:
        rules_doc = yaml.safe_load(handle)
    rules = rules_doc.get("risk_factors", [])
    if metadata.get("fit_status") != "trained" or not paths["shap"].exists():
        with paths["output"].open("w", encoding="utf-8-sig", newline="") as handle:
            csv.writer(handle).writerow(FIELDS)
        paths["note"].write_text(
            "# VR.net 规则引擎建议\n\n**流程验证，非最终结论。** 未生成建议：训练阶段未拟合模型，因而没有模型风险等级或 SHAP 特征可供匹配。输出 CSV 只有列头。没有把视频像素光流特征强行等同于规则库的米/秒虚拟平移速度。\n",
            encoding="utf-8")
        print("recommendations=0 reason=no_fitted_model_or_shap")
        return 0

    top = pd.read_csv(paths["top5"], encoding="utf-8-sig")["feature"].astype(str).tolist()
    shap = pd.read_csv(paths["shap"], encoding="utf-8-sig")
    pred = pd.read_csv(paths["predictions"], encoding="utf-8-sig")
    target = shap.loc[shap["target_class"].astype(str).eq("high") & shap["feature"].astype(str).isin(top)].copy()
    target = target.loc[pd.to_numeric(target["shap_value"], errors="coerce").gt(0)]
    target = target.sort_values("shap_value", ascending=False).drop_duplicates("window_id")
    output = []
    for row in target.to_dict("records"):
        window_id = row["window_id"]
        prediction = pred.loc[pred["window_id"].astype(str).eq(str(window_id))]
        if prediction.empty or str(prediction.iloc[0]["predicted_risk_level"]) != "high":
            continue
        feature = str(row["feature"])
        matched = []
        for rule in rules:
            metric = rule.get("metric")
            metrics = {str(metric)} if metric is not None else set()
            threshold = rule.get("threshold", {})
            for condition in threshold.get("conditions", []) if isinstance(threshold, dict) else []:
                if condition.get("metric") is not None:
                    metrics.add(str(condition["metric"]))
            if feature in metrics:
                matched.append(rule)
        rule = matched[0] if matched else None
        sources = rule.get("source", []) if rule else []
        output.append({
            "window_id": window_id, "risk_level": "high", "top_feature": feature,
            "shap_value": row["shap_value"], "matched_rule": rule.get("id", "none") if rule else "none",
            "recommendation": rule.get("recommendation", "无精确匹配规则；不提供修改建议。") if rule else "无精确匹配规则；不提供修改建议。",
            "evidence_strength": rule.get("evidence_strength", "unknown") if rule else "unknown",
            "source": "; ".join(str(source) for source in sources) if sources else "none",
        })
    pd.DataFrame(output, columns=FIELDS).to_csv(paths["output"], index=False, encoding="utf-8-sig")
    matched_count = sum(row["matched_rule"] != "none" for row in output)
    paths["note"].write_text(
        f"# VR.net 规则引擎建议\n\n**流程验证，非最终结论。** 高风险预测窗口中生成 {len(output)} 条 Top-5 特征记录，其中 {matched_count} 条精确匹配规则。规则匹配只按规则指标名精确匹配；`none` 表示没有相应规则。SHAP 贡献不是因果证据，建议需要针对目标场景验证。\n",
        encoding="utf-8")
    print(f"recommendations={len(output)} matched={matched_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
