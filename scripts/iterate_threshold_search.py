#!/usr/bin/env python3
"""Audit literature candidates sequentially against frozen feature/metric gates.

No training, quantile threshold search, raw-data downloads, or Git operations.
Literature discovery is recorded in the candidate/source audit, not fabricated
by this script. Optional HTTP metadata checks do not certify full-text claims.
Unsupported/unit-unconfirmed candidates produce unknown for every session.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yaml


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def members(config, key):
    value = config.get(key, []) or []
    if isinstance(value, dict):
        return {k: dict(v, name=k) for k, v in value.items()}
    return {v['name']: v for v in value}


def source_metadata(candidate, cache, online):
    """A resolver response is only accessibility/metadata, never proof of a claim."""
    entry = {'discovery': 'manual_original_source_audit',
             'original_source_audit': 'results/original_threshold_sources.md',
             'source_verified': bool(candidate.get('source_verified', False))}
    if online and candidate.get('source_url'):
        url = candidate['source_url']
        try:
            with requests.get(url, timeout=(15, 30), stream=True, headers={'User-Agent': 'VRnet-threshold-audit/1.0'}) as response:
                entry.update({'checked_utc': utc_now(), 'url': url, 'http_status': response.status_code,
                              'content_type': response.headers.get('Content-Type'),
                              'claim_verified_by_http': False})
        except Exception:
            entry.update({'checked_utc': utc_now(), 'url': url, 'access_error': traceback.format_exc(),
                          'claim_verified_by_http': False})
    cache[candidate['id']] = entry
    return entry


def finite_value(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def evaluate(candidate, feature_configs, frame, n):
    name = candidate['feature']
    feature = feature_configs.get(name)
    column = candidate.get('statistic')
    result = {'id': candidate['id'], 'label': candidate['label'], 'feature': name,
              'candidate_value': candidate.get('value'), 'candidate_unit': candidate.get('unit', 'unknown'),
              'statistic': column, 'source_url': candidate.get('source_url'), 'doi': candidate.get('doi'),
              'source_role': candidate.get('source_role', 'unknown'), 'positive': 0, 'negative': 0,
              'unknown': n, 'distribution_generated': False, 'checks': [], 'passed': False}
    # Audit the existing column without pretending that it is the experimental
    # statistic needed for a label. Never compare these raw numbers to a cutoff
    # when units or measurement semantics are not verified.
    audit_column = candidate.get('audit_column')
    result['audit_column'] = audit_column
    result['required_statistic'] = candidate.get('required_statistic', 'unknown')
    result['local_unit'] = feature.get('unit', 'unknown') if feature else 'unknown'
    if audit_column and audit_column in frame.columns:
        audit_values = pd.to_numeric(frame[audit_column], errors='coerce').replace([np.inf, -np.inf], np.nan)
        finite = audit_values.dropna()
        result['local_range_audit'] = {'column': audit_column, 'valid': int(len(finite)),
                                     'missing_or_nonfinite': int(audit_values.isna().sum()),
                                     'min': float(finite.min()) if len(finite) else None,
                                     'max': float(finite.max()) if len(finite) else None,
                                     'mean': float(finite.mean()) if len(finite) else None,
                                     'used_for_labels': False}
    else:
        result['local_range_audit'] = {'column': audit_column, 'status': 'feature_column_missing',
                                     'used_for_labels': False}
    def check(name, passed, reason):
        result['checks'].append({'check': name, 'passed': passed, 'reason': reason})
        return passed
    source_ok = candidate.get('source_verified') is True
    check('source', source_ok, candidate.get('reason', 'Original source/value not verified'))
    numeric_ok = finite_value(candidate.get('value'))
    check('numeric_threshold', numeric_ok, 'A source may exist without a verified numeric threshold')
    source_unit_ok = candidate.get('threshold_unit_verified') is True
    check('source_unit', source_unit_ok, 'Original threshold unit must be verified independently of local units')
    forbidden_role = candidate.get('source_role') == 'distribution_balancing_experiment'
    check('allowed_source_role', not forbidden_role, 'Data-balancing cutoffs are prohibited by this request')
    threshold_ok = candidate.get('threshold_claim_verified') is True
    check('threshold_meaning', threshold_ok, 'A stimulus/condition/mean or class-balancing value is not automatically a matching cutoff')
    authorized = bool(feature and feature.get('three_conditions_verified') is True)
    check('three_feature_conditions', authorized, 'Requires actual VR.net extraction, actual Unity reading/adjustment and matched metric')
    unit_status = feature.get('unit_verified', False) if feature else False
    unit_match = bool(feature and unit_status is True and feature.get('unit') == candidate.get('unit'))
    check('units', unit_match, f"local unit={feature.get('unit','unknown') if feature else 'unknown'}; independent verification={unit_status}")
    scope_ok = candidate.get('source_scope_match') is True
    check('scope', scope_ok, str(candidate.get('conditions', 'unknown')))
    column_ok = bool(column and column in frame.columns)
    check('statistic_exists', column_ok, str(column))
    statistic_ok = candidate.get('statistic_match_verified') is True
    check('statistic_definition', statistic_ok, 'Original exposure/aggregation must match the measured target')
    # All checks are retained even if an earlier check fails; no scientific
    # candidate is retried with a fabricated source, unit, or target definition.
    if not source_ok:
        reason = 'source_not_found'
    elif not numeric_ok:
        reason = 'numeric_threshold_not_found'
    elif not source_unit_ok:
        reason = 'source_unit_unverified'
    elif not unit_match:
        reason = 'unit_mismatch'
    elif not authorized:
        reason = 'feature_not_usable'
    elif forbidden_role or not threshold_ok:
        reason = 'source_not_applicable_as_threshold'
    elif not scope_ok or not statistic_ok:
        reason = 'scope_or_statistic_mismatch'
    elif not column_ok:
        reason = 'feature_column_missing'
    else:
        values = pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)
        known = values.notna()
        if candidate.get('operator') not in ('>', 'greater_than'):
            result['status'] = 'operator_not_supported'
            return result
        labels = values[known].gt(float(candidate['value']))
        result.update(positive=int(labels.sum()), negative=int((~labels).sum()), unknown=int((~known).sum()),
                      distribution_generated=True)
        # The user's 5..56 rule is a support/feasibility screen for 61 sessions.
        # It is not evidence that the threshold predicts symptoms or clip labels.
        if 5 <= result['positive'] <= 56 and 5 <= result['negative'] <= 56:
            reason = 'support_screen_pass'
            clip_ready = candidate.get('clip_definition_verified') is True
            check('clip_definition', clip_ready, 'Session distribution does not establish synchronized clip supervision')
            result['passed'] = clip_ready
            if not clip_ready:
                reason = 'clip_definition_unverified'
        else:
            reason = 'distribution_fail'
    result['status'] = reason
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features-config', type=Path, default=Path('configs/usable_features_v1.yaml'))
    p.add_argument('--candidates', type=Path, default=Path('configs/threshold_candidates_v4.yaml'))
    p.add_argument('--input', type=Path, default=Path('features/vrnet_training_matrix_v6.csv'))
    p.add_argument('--log', type=Path, default=Path('results/threshold_iteration_log.md'))
    p.add_argument('--final-config', type=Path, default=Path('configs/risk_factor_labels_v4.yaml'))
    p.add_argument('--summary', type=Path, default=Path('results/threshold_search_summary.md'))
    p.add_argument('--json-output', type=Path, default=Path('results/threshold_iteration_results.json'))
    p.add_argument('--verify-sources-online', action='store_true',
                   help='Optional availability check; never substitutes for original-source reading')
    a = p.parse_args()
    feature_bytes, candidate_bytes = a.features_config.read_bytes(), a.candidates.read_bytes()
    fconfig, cconfig = yaml.safe_load(feature_bytes), yaml.safe_load(candidate_bytes)
    if fconfig.get('frozen') is not True or cconfig.get('review_complete') is not True:
        raise ValueError('Feature audit must be frozen and candidate source review complete before iteration')
    frame = pd.read_csv(a.input, encoding='utf-8-sig')
    if 'session_id' not in frame or frame.session_id.duplicated().any():
        raise ValueError('Session IDs must exist and be unique')
    # User specified the exact 61-session support rule. Do not silently apply it
    # to a different cohort or select a new cutoff to achieve balanced classes.
    if len(frame) != 61:
        raise ValueError(f'Expected the requested 61-session v6 cohort; found {len(frame)}')
    eligible = members(fconfig, 'usable_features')
    all_features = {**members(fconfig, 'unusable_features'), **members(fconfig, 'conditional_features'), **eligible}
    cache, results, chosen = {}, [], {}
    candidates = sorted(cconfig.get('candidates', []), key=lambda c: (c['label'], c.get('priority', 999)))
    if len({c['id'] for c in candidates}) != len(candidates):
        raise ValueError('Candidate IDs must be unique')
    # Refuse before writing outputs, rather than late in an expensive audit.
    if a.final_config.exists():
        raise FileExistsError('Refusing to overwrite an already-frozen v4 configuration; use a new version/output path')
    for candidate in candidates:
        if candidate['label'] in chosen:
            result = {'id': candidate['id'], 'label': candidate['label'], 'status': 'not_attempted_after_pass',
                      'positive': 0, 'negative': 0, 'unknown': len(frame), 'distribution_generated': False, 'passed': False}
        else:
            metadata = source_metadata(candidate, cache, a.verify_sources_online)
            result = evaluate(candidate, all_features, frame, len(frame))
            result['source_metadata'] = metadata
            if result['passed']:
                chosen[candidate['label']] = candidate
        results.append(result)
        print(json.dumps({'id': candidate['id'], 'status': result['status'], 'counts': [result['positive'],result['negative'],result['unknown']]}), flush=True)
    labels = []
    for name, candidate in chosen.items():
        labels.append({'id': name, 'enabled': True, 'threshold_value': candidate['value'],
                       'unit': candidate['unit'], 'operator': candidate['operator'], 'statistic': candidate['statistic'],
                       'source': candidate.get('source_url'), 'original_sources': candidate.get('original_sources', []),
                       'doi': candidate.get('doi'), 'conditions': candidate.get('conditions'),
                       'label_kind': 'condition_scoped_engine_proxy', 'validated_symptom_outcome': False})
    unusable = []
    for name in dict.fromkeys(c['label'] for c in candidates):
        attempts = [r for r in results if r['label'] == name]
        if name not in chosen:
            unusable.append({'id': name, 'status': 'no_usable_threshold', 'attempts': len(attempts),
                             'reasons': sorted(set(r['status'] for r in attempts)), 'label_value': 'unknown'})
    final = {'version': '4.0', 'frozen': True, 'frozen_at_utc': utc_now(), 'based_on': 'arXiv:2306.03381v1',
             'scope': 'Current local VR.net / tested Unity feature conditions only',
             'frozen_date': datetime.now().astimezone().date().isoformat(),
             'note': 'Only source/unit/metric/scope/clip/support gates passed labels can be enabled',
             'feature_config_sha256': hashlib.sha256(feature_bytes).hexdigest(),
             'candidate_config_sha256': hashlib.sha256(candidate_bytes).hexdigest(),
             'labels': labels, 'unusable_labels': unusable, 'data_percentile_allowed': False,
             'unknown_handling': 'Missing or unsupported supervision stays unknown; never convert to negative',
             'session_label_rule': 'Generate synchronized labels per time window; never copy session labels',
             'distribution_screen': {'minimum_positive': 5, 'maximum_positive': 56, 'minimum_negative': 5, 'maximum_negative': 56,
                                     'cohort': 'v6 61 sessions', 'proves_cybersickness_validity': False}}
    a.final_config.parent.mkdir(parents=True, exist_ok=True)
    a.final_config.write_text(yaml.safe_dump(final, allow_unicode=True, sort_keys=False), encoding='utf-8')
    a.json_output.parent.mkdir(parents=True, exist_ok=True)
    a.json_output.write_text(json.dumps({'run_utc': utc_now(), 'command': sys.argv, 'n_sessions': len(frame),
                            'input_sha256': hashlib.sha256(a.input.read_bytes()).hexdigest(),
                            'attempts': results, 'source_access_checks': cache}, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# 文献候选阈值迭代日志', '', f'执行UTC：{utc_now()}；输入：{a.input}，{len(frame)}个session。',
             '先完成特征三条件核验，再执行本脚本。没有训练、P80、数据下载或Git提交。',
             '来源原文与检索过程见original_threshold_sources.md；HTTP可访问不证明某条阈值有效。',
             '单位、来源或指标不匹配时：正例0、负例0、unknown61；没有对无法适用的数值强行比较。',
             'source_not_found表示候选阈值证据未核实，不等于论文或URL不存在；numeric_threshold_not_found表示文献有记录但无可冻结数值。',
             '5–56正/负支持仅是本地可训练性检查，不能证明风险效度，也不能据此调整阈值。', '']
    for name in dict.fromkeys(c['label'] for c in candidates):
        lines.extend(['## 风险因素：' + name, ''])
        for candidate in [c for c in candidates if c['label'] == name]:
            r = next(r for r in results if r['id'] == candidate['id'])
            lines.extend([f"### 候选：{candidate['id']}",
                          f"- 候选值：{candidate.get('value')} {candidate.get('unit','unknown')}；统计量：{candidate.get('statistic')}",
                          f"- 来源：{candidate.get('source_url','unknown')}；DOI：{candidate.get('doi','unknown')}；角色：{candidate.get('source_role','unknown')}",
                          f"- 原文位置：{candidate.get('page','unknown')}；原始引用链：{json.dumps(candidate.get('original_sources', []), ensure_ascii=False)}",
                          f"- 实验条件：{candidate.get('conditions','unknown')}",
                          f"- 本地范围审计（不生成标签）：{json.dumps(r.get('local_range_audit', {}), ensure_ascii=False)}；本地单位：{r.get('local_unit','unknown')}",
                          f"- 所需统计定义：{r.get('required_statistic','unknown')}；audit_column不可自动替代此定义。",
                          f"- 判定：{r['status']}；通过={r['passed']}",
                          f"- 标签分布：正例{r['positive']}、负例{r['negative']}、unknown{r['unknown']}；实际生成={r['distribution_generated']}"])
            for check in r.get('checks', []):
                lines.append(f"  - {check['check']}: {check['passed']}；{check['reason']}")
            lines.append('')
        lines.extend(['### 结论：' + ('usable_threshold' if name in chosen else 'no_usable_threshold'), ''])
    a.log.parent.mkdir(parents=True, exist_ok=True)
    a.log.write_text('\n'.join(lines), encoding='utf-8')
    summary = ['# 阈值搜索与最终标签总结', '', '仅适用于本轮已核验特征、来源及数据条件；未找到可用标签不等于没有运动或没有不适。', '',
               '| 风险因素 | 候选数 | 通过数 | 状态 |', '|---|---:|---:|---|']
    for name in dict.fromkeys(c['label'] for c in candidates):
        group = [r for r in results if r['label'] == name]
        summary.append(f"| {name} | {len(group)} | {sum(r['passed'] for r in group)} | {'usable' if name in chosen else 'no_usable_threshold'} |")
    summary.extend(['', f'最终可用标签数：{len(labels)}。总候选记录数：{len(results)}。',
                    '前置问题：源坐标到米/时间单位校准、总相机与人工导航语义、向量加速度定义、非对称FOV、可见对象计数与稳定身份、场景族与真实XR调参。',
                    '下一步：优先解决少数可操作的运动因素测量；在范围明确的工程任务中使用连续监督或匹配原实验的标签，单独验证与症状的关系。',
                    '旧v3/二分类阈值及PROTOCOL均未修改；v4通过项才启用；无通过项时labels为空。', ''])
    a.summary.parent.mkdir(parents=True, exist_ok=True)
    a.summary.write_text('\n'.join(summary), encoding='utf-8')
    # The body of the request names this report in addition to its output list.
    a.summary.with_name('label_definition_summary.md').write_text('\n'.join(summary), encoding='utf-8')
    print(json.dumps({'attempts': len(results), 'passed_labels': len(labels), 'final_config': str(a.final_config)}))


if __name__ == '__main__':
    main()
