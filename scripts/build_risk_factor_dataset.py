#!/usr/bin/env python3
"""Build deterministic proxy risk-factor labels from the VR.net v6 matrix.

The output labels are rule-generated proxies, not participant sickness outcomes.
MS_rating/FMS columns are explicitly excluded from both label construction and
the output training matrix.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROXY_SOURCE = "rule_generated_proxy"
IDENTIFIER_COLUMNS = ["session_id", "game_name"]


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def is_outcome_column(name: str, patterns: list[str]) -> bool:
    lowered = name.casefold()
    return any(pattern.casefold() in lowered for pattern in patterns)


def proxy_label(frame: pd.DataFrame, rule: dict[str, Any]) -> pd.Series:
    """Return string values 1, 0, or unknown for one configured rule."""
    output = pd.Series("unknown", index=frame.index, dtype="object")
    if not bool(rule.get("operationalized", False)):
        return output

    features = [str(value) for value in rule.get("required_features", [])]
    threshold = rule.get("threshold")
    if len(features) != 1 or features[0] not in frame.columns or threshold is None:
        return output

    try:
        numeric_threshold = float(threshold)
    except (TypeError, ValueError):
        return output
    if not math.isfinite(numeric_threshold):
        return output

    values = pd.to_numeric(frame[features[0]], errors="coerce")
    known = values.notna() & values.map(math.isfinite)
    operator = str(rule.get("operator", ""))
    if operator == ">":
        output.loc[known & values.gt(numeric_threshold)] = "1"
        output.loc[known & values.le(numeric_threshold)] = "0"
        return output
    raise ValueError(f"Unsupported operator for operationalized rule {rule.get('id')!r}: {operator!r}")


def markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def rule_expression(rule: dict[str, Any]) -> str:
    features = [str(value) for value in rule.get("required_features", [])]
    threshold = rule.get("threshold")
    if not bool(rule.get("operationalized", False)) or threshold is None:
        return "not operationalized; label=unknown"
    return f"{features[0]} {rule.get('operator')} {threshold:g}"


def build_report(
    config_path: Path,
    input_path: Path,
    labels_path: Path,
    matrix_path: Path,
    frame: pd.DataFrame,
    labels: pd.DataFrame,
    rules: list[dict[str, Any]],
    excluded_columns: list[str],
    training_columns: list[str],
) -> str:
    lines = [
        "# Rule-generated proxy 风险因素标签分布",
        "",
        "**这些标签全部是 `rule_generated_proxy`，不是参与者实际晕动症标签，也未经过预测有效性验证。**",
        "",
        "## 数据与防标签泄漏",
        "",
        f"- 配置：`{config_path.as_posix()}`",
        f"- 输入：`{input_path.as_posix()}`",
        f"- 输入 session：{len(frame)}",
        f"- 标签输出：`{labels_path.as_posix()}`",
        f"- 训练矩阵输出：`{matrix_path.as_posix()}`（{len(frame)} 行 × {len(training_columns)} 列）",
        "- 标签只由配置所列场景/运动特征和固定比较规则生成。",
        "- 未使用任何 `MS_rating` 或 `fms` 字段作为特征或标签。",
        "- 从训练矩阵排除的 outcome 列：" + (", ".join(f"`{name}`" for name in excluded_columns) if excluded_columns else "无"),
        "",
        "标签值含义：`1`=规则触发，`0`=规则未触发，`unknown`=直接特征、单位定义或操作阈值不足。",
        "",
        "## 标签分布",
        "",
        "| 代理标签 | 规则 | 1 | 0 | unknown | 已知标签中触发率 | 阈值证据状态 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for rule in rules:
        label = str(rule["id"])
        counts = labels[label].value_counts(dropna=False)
        positive = int(counts.get("1", 0))
        negative = int(counts.get("0", 0))
        unknown = int(counts.get("unknown", 0))
        known = positive + negative
        rate = f"{positive / known:.1%}" if known else "NA"
        lines.append(
            "| " + " | ".join([
                markdown_cell(label), markdown_cell(rule_expression(rule)),
                str(positive), str(negative), str(unknown), rate,
                markdown_cell(rule.get("evidence_status", "unknown")),
            ]) + " |"
        )

    lines.extend([
        "",
        "## 阈值、单位与证据边界",
        "",
        "| 代理标签 | 单位 | 来源 | 证据说明 |",
        "|---|---|---|---|",
    ])
    for rule in rules:
        lines.append(
            "| " + " | ".join([
                markdown_cell(rule["id"]), markdown_cell(rule.get("unit", "unknown")),
                markdown_cell(rule.get("threshold_source", "unknown")),
                markdown_cell(rule.get("evidence_note", "")),
            ]) + " |"
        )

    lines.extend([
        "",
        "## 多轴旋转未操作化",
        "",
        "`multi_axis_rotation` 要求从 `v_matrix` 分解 yaw/pitch/roll，并判定任意两个轴的变化是否超过数值阈值。",
        "本地冻结规则只定义“至少两个旋转轴”，没有给出轴变化 cutoff；v6 session 矩阵也没有三个轴的独立变化列。",
        "因此 61 个 session 全部为 `unknown`，没有用 scalar rotation-rate 特征代替，也没有猜测阈值。",
        "",
        "## 使用限制",
        "",
        "- 速度、加速度和物体速度以 VR.net 矩阵源坐标单位表示；没有证据确认 1 源单位等于 1 米。",
        "- 用户指定的数值 cutoff 是启发式代理规则；除规则说明中明确限定的文献背景外，不应表述为通用安全边界或因果阈值。",
        "- 用这些确定性标签训练模型只会学习规则近似，不能证明能预测真实 MS/FMS、SSQ 或个体症状。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/risk_factor_labels.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--labels-output", default=None)
    parser.add_argument("--matrix-output", default=None)
    parser.add_argument("--report-output", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = resolve_path(root, args.config)
    with config_path.open("r", encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle)

    input_path = resolve_path(root, args.input or config["source_matrix"])
    output_config = config["outputs"]
    labels_path = resolve_path(root, args.labels_output or output_config["labels"])
    matrix_path = resolve_path(root, args.matrix_output or output_config["training_matrix"])
    report_path = resolve_path(root, args.report_output or output_config["distribution_report"])
    rules = list(config.get("rules", []))
    patterns = [str(value) for value in config.get("excluded_outcome_patterns", ["ms_rating", "fms"])]

    frame = pd.read_csv(input_path, encoding="utf-8-sig")
    missing_identifiers = [name for name in IDENTIFIER_COLUMNS if name not in frame.columns]
    if missing_identifiers:
        raise ValueError(f"Input is missing identifier columns: {missing_identifiers}")
    if frame["session_id"].astype(str).duplicated().any():
        raise ValueError("Expected one row per session_id; duplicate session IDs found")

    label_columns: dict[str, pd.Series] = {}
    for rule in rules:
        label_id = str(rule.get("id", "")).strip()
        if not label_id or label_id in label_columns:
            raise ValueError(f"Invalid or duplicate rule id: {label_id!r}")
        if str(rule.get("label_source")) != PROXY_SOURCE:
            raise ValueError(f"Rule {label_id!r} is not marked {PROXY_SOURCE!r}")
        label_columns[label_id] = proxy_label(frame, rule)

    labels = frame[IDENTIFIER_COLUMNS].copy()
    labels["label_source"] = PROXY_SOURCE
    for name, values in label_columns.items():
        labels[name] = values

    excluded_columns = [column for column in frame.columns if is_outcome_column(column, patterns)]
    feature_columns = [
        column for column in frame.columns
        if column not in IDENTIFIER_COLUMNS and column not in excluded_columns
    ]
    training = frame[IDENTIFIER_COLUMNS + feature_columns].copy()
    training["label_source"] = PROXY_SOURCE
    for name in label_columns:
        training[name] = labels[name]

    leaked = [column for column in training.columns if is_outcome_column(column, patterns)]
    if leaked:
        raise AssertionError(f"Outcome columns leaked into training matrix: {leaked}")
    if len(labels) != len(frame) or len(training) != len(frame):
        raise AssertionError("Output row count does not match input")

    for path in (labels_path, matrix_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(labels_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    training.to_csv(matrix_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    report = build_report(
        config_path.relative_to(root), input_path.relative_to(root),
        labels_path.relative_to(root), matrix_path.relative_to(root),
        frame, labels, rules, excluded_columns, list(training.columns),
    )
    report_path.write_text(report, encoding="utf-8-sig")

    summaries = []
    for name in label_columns:
        counts = labels[name].value_counts()
        summaries.append(
            f"{name}:1={int(counts.get('1', 0))},0={int(counts.get('0', 0))},unknown={int(counts.get('unknown', 0))}"
        )
    print(f"rows={len(frame)} labels={labels_path} matrix={matrix_path} report={report_path}")
    print("; ".join(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
