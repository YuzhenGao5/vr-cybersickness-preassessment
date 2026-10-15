#!/usr/bin/env python3
"""Generate a truthful summary of the VR.net window-model workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def fmt(value: object) -> str:
    if value is None or pd.isna(value):
        return "未计算"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", default="features/training_windows_vrnet.csv")
    parser.add_argument("--metadata", default="models/risk_model_vrnet_metadata.json")
    parser.add_argument("--shap-top5", default="features/shap_top5_vrnet.csv")
    parser.add_argument("--recommendations", default="results/recommendations_vrnet.csv")
    parser.add_argument("--output", default="results/full_report_vrnet.md")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    resolve = lambda value: (root / value).resolve() if not Path(value).is_absolute() else Path(value)
    paths = {name: resolve(value) for name, value in vars(args).items()}
    windows = pd.read_csv(paths["windows"], encoding="utf-8-sig")
    with paths["metadata"].open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    top = pd.read_csv(paths["shap_top5"], encoding="utf-8-sig") if paths["shap_top5"].exists() else pd.DataFrame()
    recs = pd.read_csv(paths["recommendations"], encoding="utf-8-sig") if paths["recommendations"].exists() else pd.DataFrame()
    counts = windows["fms_label"].value_counts().to_dict()
    label_source = windows["label_source"].value_counts().to_dict()
    metrics = metadata.get("metrics", {})
    trained = metadata.get("fit_status") == "trained"
    lines = [
        "# VR.net 风险分析与修改建议报告", "", "**流程验证，非最终结论。** 本报告不声称模型预测能力，不预测个体晕动症，亦不表示工具已验证。", "",
        "## 数据来源", "", "- 数据集：VR.net Mars Miners (11_2)",
        f"- 窗口数：{len(windows)}；有标签窗口：{windows['fms_label'].isin(['low','medium','high']).sum()}",
        f"- FMS 标签来源：`MS_rating` 稀疏事件；按冻结配置分为 30 秒窗口并向后延用最近一次观测值。来源类型计数：`{json.dumps(label_source, ensure_ascii=False)}`",
        f"- 风险等级分布：low={counts.get('low', 0)}, medium={counts.get('medium', 0)}, high={counts.get('high', 0)}",
        "- 样本量警告：同一 session 的窗口并非独立样本；结果仅作流程验证。", "",
        "## 模型结果", "", f"- 模型：{'LightGBM' if trained else '未拟合（LightGBM 训练被跳过）'}",
        f"- 评估：{metadata.get('evaluation_plan', '未执行')}",
        f"- Accuracy：{fmt(metrics.get('accuracy'))}", f"- F1 (macro)：{fmt(metrics.get('f1_macro'))}",
        f"- AUC (macro)：{fmt(metrics.get('auc_ovr_macro'))}",
        f"- 状态：{metadata.get('fit_status')}；原因：{'；'.join(metadata.get('reasons', [])) or '无'}",
        "- 警告：样本量小，指标不可靠。单类别时 Accuracy/F1/AUC 均不适用，故不计算。", "",
        "## SHAP Top-5 特征", "",
    ]
    if top.empty:
        lines.extend(["模型未拟合，因此无 SHAP 值与 Top-5；没有生成 SHAP 图。", ""])
    else:
        lines.extend(["| 排名 | 特征 | 平均 |SHAP| | 类别方向 |", "|---:|---|---:|---|"])
        for row in top.to_dict("records"):
            lines.append(f"| {row.get('rank')} | `{row.get('feature')}` | {fmt(row.get('mean_abs_shap'))} | `{row.get('direction_by_class')}` |")
        lines.append("")
    lines.extend(["## 规则引擎建议", ""])
    if recs.empty:
        lines.extend(["无建议记录：未拟合模型，因此没有预测风险等级或 SHAP 特征可供规则匹配。", ""])
    else:
        lines.extend(["| 窗口 | 风险等级 | Top特征 | 匹配规则 | 建议 | 证据强度 |", "|---|---|---|---|---|---|"])
        for row in recs.to_dict("records"):
            lines.append("| " + " | ".join(str(row.get(key, "")) for key in ["window_id", "risk_level", "top_feature", "matched_rule", "recommendation", "evidence_strength"]) + " |")
        lines.append("")
    lines.extend([
        "## 汇总", "", f"- 高风险窗口：{'未预测' if not trained else int((pd.read_csv(root / 'features/risk_predictions_vrnet.csv')['predicted_risk_level'] == 'high').sum())}",
        f"- 中风险窗口：{'未预测' if not trained else int((pd.read_csv(root / 'features/risk_predictions_vrnet.csv')['predicted_risk_level'] == 'medium').sum())}",
        f"- 低风险窗口：{'未预测' if not trained else int((pd.read_csv(root / 'features/risk_predictions_vrnet.csv')['predicted_risk_level'] == 'low').sum())}",
        f"- 规则建议记录数：{len(recs)}；精确匹配规则数：{int((recs.get('matched_rule', pd.Series(dtype=str)) != 'none').sum()) if not recs.empty else 0}",
        "- 未匹配特征：无 SHAP 特征可检查。", "",
        "## 评估边界与下一步", "",
        "本数据只有一个 VR.net session/游戏，且所有已标注窗口均为 low。无法构造多类别训练/验证划分，未训练模型，也未运行 SHAP 或生成建议。PROTOCOL.md 要求的 LOSO 与 Leave-One-Scene-Out 泛化评估在当前单 session 数据上不可执行；原协议 AUC 门控未评估。",
        "需要取得标签覆盖低/中/高风险且来自多个独立参与者/场景的数据，再冻结按组划分方案并评估；另需确认 MS_rating 的量表定义与所称 FMS 的对应关系。当前视频像素光流等路径依赖特征不能直接匹配以物理单位或 Unity 场景元数据定义的规则。", "",
        "## 免责声明", "", "本报告仅作流程验证。样本量小，不声称预测能力。规则建议基于文献，未经用户验证。窗口级代理标签不等同于个体诊断或预测。", "",
    ])
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    paths["output"].write_text("\n".join(lines), encoding="utf-8")
    print(f"report={paths['output']} windows={len(windows)} labeled={int(windows['fms_label'].isin(['low','medium','high']).sum())} trained={trained} recommendations={len(recs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
