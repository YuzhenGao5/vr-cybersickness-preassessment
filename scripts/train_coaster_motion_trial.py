"""Exploratory RGB flow -> engineering motion regression; no FMS prediction.

The experiment definition is created by the extraction owner before fitting.
Unknown trajectory families are excluded from the provisional family holdout.
Whole recording holdout is supplementary and does not establish unseen tracks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'coaster_motion_trial'
SEED = 42
TARGETS = (
    'target_camera_speed_mean',
    'target_camera_acceleration_mean',
    'target_camera_angular_speed_mean',
    'target_candidate_rig_speed_mean',
    'target_candidate_rig_acceleration_mean',
)
METHODS = ('train_mean', 'elapsed_ridge', 'flow_ridge', 'flow_lightgbm',
           'visual_flow_lightgbm')
LGBM = dict(objective='regression', n_estimators=150, num_leaves=7,
            max_depth=3, learning_rate=.05, min_child_samples=30,
            reg_lambda=5., colsample_bytree=.9, random_state=SEED,
            n_jobs=2, verbosity=-1)
WARNING = '探索性工程运动识别流程验证，非最终结论；不是 FMS 或人体晕动症风险预测。'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_json(value):
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.generic):
        return finite_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def score(y, pred, reference=None):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    if not len(y) or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('Cannot score empty or nonfinite predictions')
    err = pred - y
    result = dict(n=len(y), mae=float(np.abs(err).mean()),
                  rmse=float(np.sqrt(np.square(err).mean())))
    varying = len(y) >= 3 and np.ptp(y) > 1e-12 and np.ptp(pred) > 1e-12
    result['spearman'] = float(spearmanr(y, pred).statistic) if varying else None
    result['spearman_status'] = 'descriptive_temporally_dependent_clips_no_p_value' if varying else 'NA_constant_or_insufficient'
    if reference is not None:
        ref = score(y, reference)
        result['mae_skill_vs_train_mean'] = 1 - result['mae'] / ref['mae'] if ref['mae'] > 1e-12 else None
        result['rmse_skill_vs_train_mean'] = 1 - result['rmse'] / ref['rmse'] if ref['rmse'] > 1e-12 else None
        result['reference_mae'] = ref['mae']
        result['reference_rmse'] = ref['rmse']
    return result


def select_features(method, flow, visual):
    if method == 'train_mean':
        return []
    if method == 'elapsed_ridge':
        return ['clip_elapsed_sec']
    if method == 'visual_flow_lightgbm':
        return flow + visual
    return flow


def fit(train, target, method, requested):
    y = train[target].to_numpy(float)
    available = [c for c in requested if train[c].notna().any()]
    imputer, scaler = None, None
    keep = np.ones(len(available), dtype=bool)
    if available:
        imputer = SimpleImputer(strategy='median')
        x = imputer.fit_transform(train[available])
        keep = np.var(x, axis=0) > 1e-12
        x = x[:, keep]
    retained = [c for c, chosen in zip(available, keep) if chosen]
    fallback = None
    if method == 'train_mean' or not retained:
        model = DummyRegressor(strategy='mean')
        x = np.zeros((len(train), 1))
        if method != 'train_mean':
            fallback = 'No nonconstant observed training-fold features; train-mean fallback'
    elif method.endswith('lightgbm'):
        model = lgb.LGBMRegressor(**LGBM)
    else:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        model = Ridge(alpha=10.)
    model.fit(x, y)
    diagnostics = None
    if hasattr(model, 'booster_'):
        dump = model.booster_.dump_model()
        diagnostics = dict(tree_count=model.booster_.num_trees(),
                           split_tree_count=sum(t['num_leaves'] > 1 for t in dump['tree_info']),
                           leaf_counts=[t['num_leaves'] for t in dump['tree_info']])
    return dict(model=model, imputer=imputer, scaler=scaler,
                requested_features=requested, available_features=available,
                variance_mask=keep, retained_features=retained,
                dropped_features=[c for c in requested if c not in retained],
                target=target, method=method, fallback=fallback,
                n_train=len(train), training_mean=float(y.mean()),
                tree_diagnostics=diagnostics)


def predict(bundle, frame):
    if bundle['retained_features']:
        x = bundle['imputer'].transform(frame[bundle['available_features']])
        x = x[:, bundle['variance_mask']]
        if bundle['scaler'] is not None:
            x = bundle['scaler'].transform(x)
    else:
        x = np.zeros((len(frame), 1))
    predictions = np.asarray(bundle['model'].predict(x), float)
    if not np.isfinite(predictions).all():
        raise ValueError('Nonfinite predictions')
    return predictions


def importance(bundle, scope, fold):
    model, columns = bundle['model'], bundle['retained_features']
    if hasattr(model, 'booster_'):
        weights = model.booster_.feature_importance(importance_type='gain')
        kind = 'lightgbm_gain_training_fit'
    elif hasattr(model, 'coef_'):
        weights = np.abs(model.coef_)
        kind = 'absolute_standardized_ridge_coefficient_training_fit'
    else:
        return []
    total = float(weights.sum())
    return [dict(scope=scope, fold=fold, target=bundle['target'],
                 method=bundle['method'], feature=column,
                 importance=float(weight), normalized_importance=float(weight / total) if total else 0.,
                 importance_kind=kind)
            for column, weight in zip(columns, weights)]


def get_training_definition(definition):
    """Require an explicit frozen training contract; do not infer it from fits."""
    contract = definition.get('training', definition.get('training_definition'))
    if contract is None and isinstance(definition.get('models'), dict):
        frozen_models = definition['models']
        if 'alpha10' not in str(frozen_models.get('flow_ridge')) or 'alpha10' not in str(frozen_models.get('elapsed_ridge')):
            raise ValueError('Frozen Ridge alpha10 contract required')
        contract = dict(lightgbm=frozen_models.get('flow_lightgbm'), ridge_alpha=10.,
                        seed=definition.get('seed'), targets=definition.get('target_columns'),
                        features=definition.get('feature_columns'),
                        schema='root_extraction_definition_models')
    if not isinstance(contract, dict):
        raise ValueError('Frozen definition needs a training dict before fitting')
    frozen = contract.get('lightgbm', contract.get('lightgbm_parameters'))
    if not isinstance(frozen, dict):
        raise ValueError('Frozen definition must record LightGBM parameters')
    for key, value in LGBM.items():
        if key in ('objective', 'verbosity') and key not in frozen:
            continue
        if key not in frozen or frozen[key] != value:
            raise ValueError(f'Frozen LightGBM mismatch for {key}: {frozen.get(key)!r} versus {value!r}')
    if contract.get('ridge_alpha') != 10 or contract.get('seed') != SEED:
        raise ValueError('Frozen seed42 and Ridge alpha10 required')
    frozen_targets = contract.get('targets', definition.get('targets'))
    if isinstance(frozen_targets, dict):
        frozen_targets = list(frozen_targets)
    if frozen_targets is None or set(frozen_targets) != set(TARGETS):
        raise ValueError('Frozen target names differ from training contract')
    return contract


def fmt(value):
    return 'NA' if value is None else f'{value:.4f}' if isinstance(value, float) else str(value)


def md_table(frame):
    if frame.empty:
        return '无可评估记录。'
    return frame.to_markdown(index=False, floatfmt='.4f')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / f'features/{PREFIX}_clips.csv')
    parser.add_argument('--definition', type=Path, default=ROOT / f'audit/{PREFIX}_definition.json')
    parser.add_argument('--track-audit', type=Path, default=ROOT / 'results/coaster_track_group_audit.csv')
    args = parser.parse_args()
    for path in (args.input, args.definition):
        if not path.is_file():
            raise FileNotFoundError(f'Required frozen input absent: {path}')
    for folder in ('features', 'models', 'results'):
        (ROOT / folder).mkdir(exist_ok=True)
    logger = logging.getLogger(PREFIX)
    logger.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(
            ROOT / f'results/{PREFIX}_training_log.txt', mode='a', encoding='utf-8')):
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(handler)
    definition = json.loads(args.definition.read_text(encoding='utf-8-sig'))
    contract = get_training_definition(definition)
    frame = pd.read_csv(args.input, dtype={'clip_id': str, 'session_id': str, 'participant_id': str})
    required = {'status', 'clip_id', 'session_id', 'participant_id', 'camera_run_id',
                'start_timestamp_ms', 'end_timestamp_ms', 'clip_elapsed_sec', *TARGETS}
    if not required.issubset(frame):
        raise ValueError(f'Missing columns: {sorted(required - set(frame))}')
    if frame.clip_id.isna().any() or frame.clip_id.duplicated().any():
        raise ValueError('Missing or duplicate clip IDs')
    for key in ('session_id', 'participant_id'):
        if frame[key].isna().any() or frame[key].str.strip().eq('').any():
            raise ValueError(f'Missing {key}')
    if frame.groupby('session_id').participant_id.nunique().gt(1).any():
        raise ValueError('Recording assigned to multiple participants')
    forbidden = re.compile(r'(^|_)(id|path|participant|session|event|fms|score|rating|target|label|coverage|support|duration|timestamp|count|frame|frames|times|files|bytes|sha256|sha|hash|pairs|samples)($|_)')
    flow = sorted(c for c in frame if c.startswith('flow_') and not forbidden.search(c))
    visual = sorted(c for c in frame if c.startswith('visual_') and not forbidden.search(c))
    if not flow or not visual:
        raise ValueError('Missing declared flow or visual feature modality')
    frozen_features = contract.get('features', definition.get('input_features', definition.get('feature_columns')))
    if frozen_features is not None:
        if isinstance(frozen_features, dict):
            frozen_features = frozen_features.get('flow', []) + frozen_features.get('visual', [])
        if set(frozen_features) != set(flow + visual):
            raise ValueError('Observed visual feature allowlist differs from frozen definition')
    for column in flow + visual + list(TARGETS) + ['clip_elapsed_sec']:
        frame[column] = pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)
    okay = frame.loc[frame.status.eq('ok')].copy()
    if okay.empty:
        raise ValueError('No status=ok clips; no model fit allowed')
    if okay[flow].isna().all(axis=1).any():
        raise ValueError('Eligible clip has no flow features')
    if okay[visual].isna().all(axis=1).any():
        raise ValueError('Eligible clip has no visual features')
    if okay.clip_elapsed_sec.isna().any():
        raise ValueError('Missing elapsed baseline input in eligible clips')
    okay['camera_run_id'] = pd.to_numeric(okay.camera_run_id, errors='raise').astype(int)
    splits = {'leave_one_recording_out_supplementary': (okay, 'session_id')}
    strict_status = 'blocked_no_track_group_audit'
    audit_sha = None
    strict_n = 0
    if args.track_audit.is_file():
        track = pd.read_csv(args.track_audit, dtype={'session_id': str, 'candidate_track_group': str})
        columns = ['session_id', 'camera_run_id', 'candidate_track_group']
        if not set(columns).issubset(track):
            raise ValueError('Track audit lacks required mapping columns')
        if track[columns[:2]].duplicated().any():
            raise ValueError('Track group audit contains duplicate camera-run mappings')
        track['camera_run_id'] = pd.to_numeric(track.camera_run_id, errors='raise').astype(int)
        okay = okay.merge(track[columns], on=columns[:2], how='left', validate='many_to_one')
        group = okay.candidate_track_group.fillna('unknown').str.strip()
        known = ~group.str.lower().isin(['', 'unknown', 'tbd', 'nan', 'none'])
        strict = okay.loc[known].copy()
        strict_n = len(strict)
        if strict.candidate_track_group.nunique() >= 2:
            splits['leave_one_candidate_trajectory_family_out_primary_provisional'] = (strict, 'candidate_track_group')
            strict_status = 'evaluated_provisional_geometric_groups_not_verified_track_ids'
        else:
            strict_status = 'blocked_fewer_than_two_known_candidate_trajectory_families'
        audit_sha = digest(args.track_audit)
        splits['leave_one_recording_out_supplementary'] = (okay, 'session_id')
    predictions, fold_rows, importance_rows, results, models = [], [], [], {}, {}
    for split, (cohort, group_column) in splits.items():
        results[split] = {}
        for target in TARGETS:
            valid = cohort[target].notna() & cohort[target].ge(0)
            data = cohort.loc[valid].copy()
            groups = sorted(data[group_column].dropna().unique())
            target_result = dict(n_eligible=len(cohort), n_target_known=len(data),
                                 n_target_unknown=int((~valid).sum()), n_groups=len(groups),
                                 target_min=float(data[target].min()) if len(data) else None,
                                 target_max=float(data[target].max()) if len(data) else None,
                                 methods={})
            if len(groups) < 2:
                target_result['status'] = 'blocked_fewer_than_two_groups_for_target'
                results[split][target] = target_result
                logger.info('%s %s blocked: only%s groups', split, target, len(groups))
                continue
            target_result['status'] = 'evaluated'
            for method in METHODS:
                model_rows, this_folds = [], []
                for group in groups:
                    train = data.loc[data[group_column].ne(group)]
                    test = data.loc[data[group_column].eq(group)]
                    if set(train.clip_id) & set(test.clip_id):
                        raise ValueError('Clip leakage')
                    if split.startswith('leave_one_recording') and set(train.session_id) & set(test.session_id):
                        raise ValueError('Recording leakage')
                    if split.startswith('leave_one_candidate') and set(train.candidate_track_group) & set(test.candidate_track_group):
                        raise ValueError('Repeated trajectory-family leakage')
                    if len(train) < 2 or not len(test):
                        logger.info('SKIP split=%s target=%s method=%s group=%s train=%s test=%s', split,target,method,group,len(train),len(test))
                        continue
                    requested = select_features(method, flow, visual)
                    bundle = fit(train, target, method, requested)
                    pred = predict(bundle, test)
                    train_pred = predict(bundle, train)
                    y = test[target].to_numpy(float)
                    reference = np.full(len(test), train[target].mean())
                    test_metrics = score(y, pred, reference)
                    train_metrics = score(train[target], train_pred)
                    row = dict(split=split, target=target, method=method, held_out_group=group,
                               n_train=len(train), n_test=len(test),
                               n_train_recordings=train.session_id.nunique(), n_test_recordings=test.session_id.nunique(),
                               n_retained_features=len(bundle['retained_features']),
                               dropped_features=';'.join(bundle['dropped_features']),
                               fallback=bundle['fallback'],
                               train_target_mean=float(train[target].mean()),
                               train_target_min=float(train[target].min()), train_target_max=float(train[target].max()),
                               train_in_sample_mae=train_metrics['mae'], train_in_sample_rmse=train_metrics['rmse'],
                               **{f'test_{key}': value for key, value in test_metrics.items()})
                    if bundle['tree_diagnostics'] is not None:
                        row.update({f'lgbm_{key}': value for key, value in bundle['tree_diagnostics'].items() if key != 'leaf_counts'})
                    fold_rows.append(row)
                    this_folds.append(row)
                    importance_rows += importance(bundle, split, str(group))
                    for position, record in enumerate(test.to_dict('records')):
                        model_rows.append(dict(split=split, target=target, method=method,
                            held_out_group=group, clip_id=record['clip_id'], session_id=record['session_id'],
                            participant_id=record['participant_id'], camera_run_id=record['camera_run_id'],
                            candidate_track_group=record.get('candidate_track_group'),
                            start_timestamp_ms=record['start_timestamp_ms'], end_timestamp_ms=record['end_timestamp_ms'],
                            target_value=float(y[position]), prediction=float(pred[position]),
                            train_mean_prediction=float(reference[position])))
                    logger.info('split=%s target=%s method=%s heldout=%s train=%s test=%s MAE=%.5f RMSE=%.5f skill=%.4f', split,target,method,group,len(train),len(test),test_metrics['mae'],test_metrics['rmse'],test_metrics['mae_skill_vs_train_mean'] if test_metrics['mae_skill_vs_train_mean'] is not None else float('nan'))
                if model_rows:
                    model_frame = pd.DataFrame(model_rows)
                    pooled = score(model_frame.target_value, model_frame.prediction, model_frame.train_mean_prediction)
                    macro = {}
                    for key in ('mae', 'rmse', 'spearman', 'mae_skill_vs_train_mean', 'rmse_skill_vs_train_mean'):
                        values = [r[f'test_{key}'] for r in this_folds if r[f'test_{key}'] is not None]
                        macro[key] = float(np.mean(values)) if values else None
                        macro[f'{key}_n_folds'] = len(values)
                    target_result['methods'][method] = dict(pooled_oof=pooled, equal_fold_summary=macro,
                        n_folds=len(this_folds), n_predictions=len(model_rows),
                        warning='Clips temporally dependent; candidate family groups provisional; no significance claims')
                    predictions += model_rows
            results[split][target] = target_result
    for target in TARGETS:
        data = okay.loc[okay[target].notna() & okay[target].ge(0)]
        if len(data) < 2:
            continue
        for method in METHODS:
            bundle = fit(data, target, method, select_features(method, flow, visual))
            models[f'{target}:{method}'] = bundle
            importance_rows += importance(bundle, 'full_fit_demo_only', 'all')
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(), warning=WARNING,
        frozen_definition_path=str(args.definition.resolve()), frozen_definition_sha256=digest(args.definition),
        input_path=str(args.input.resolve()), input_sha256=digest(args.input),
        track_audit_path=str(args.track_audit.resolve()), track_audit_sha256=audit_sha,
        raw_clip_rows=len(frame), status_counts=frame.status.value_counts().to_dict(),
        n_ok_clips=len(okay), n_recordings=okay.session_id.nunique(),
        n_participants=okay.participant_id.nunique(), primary_group_status=strict_status,
        n_known_family_clips=strict_n, flow_features=flow, visual_features=visual,
        targets=list(TARGETS), methods=list(METHODS), lightgbm=LGBM, ridge_alpha=10.,
        seed=SEED, frozen_training_contract=contract,
        preprocessing='Training-fold median imputation, training-fold constant-feature filter, Ridge training-fold standardization. No target normalization, tuning, prediction clipping, random clip split, or binary thresholds.',
        versions=dict(python=sys.version, pandas=pd.__version__, numpy=np.__version__,
                      lightgbm=lgb.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__))
    output = dict(metadata=metadata, evaluations=results)
    (ROOT / f'results/{PREFIX}_metrics.json').write_text(json.dumps(finite_json(output), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    prediction_frame = pd.DataFrame(predictions)
    fold_frame = pd.DataFrame(fold_rows)
    prediction_frame.to_csv(ROOT / f'features/{PREFIX}_oof_predictions.csv', index=False, encoding='utf-8-sig')
    fold_frame.to_csv(ROOT / f'results/{PREFIX}_fold_metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(importance_rows).to_csv(ROOT / f'features/{PREFIX}_feature_importance.csv', index=False, encoding='utf-8-sig')
    joblib.dump(dict(metadata=metadata, models=models, warning='Full-cohort demonstration fits only; not validated human-risk models'), ROOT / f'models/{PREFIX}_models.pkl')
    lines = ['# 过山车视觉运动学习试验', '', f'**{WARNING}**', '',
        '## 冻结协议与输入', '',
        f'- 原始候选片段：{len(frame)}；合格片段：{len(okay)}；录像：{okay.session_id.nunique()}；参与者：{okay.participant_id.nunique()}。',
        f'- 光流特征：{len(flow)}；画面统计特征：{len(visual)}。ID、矩阵、姿态、运动目标、评分和时间戳不进入视觉模型。',
        '- 主模型：光流统计 → LightGBM；对照：光流 Ridge、光流+画面 LightGBM。所有目标分别回归。',
        '- 参照：每折训练集目标均值、只使用片段经过时间的 Ridge。',
        '- 原始预测直接评估；不归一化目标、不裁剪预测、不改阈值、不调参。所有填补、去常数列及 Ridge 标准化只使用训练折。',
        f'- 主测试状态：`{strict_status}`；已知候选轨迹家族覆盖{strict_n}个合格片段。',
        '- 几何轨迹组是防重复路径泄漏的临时候选家族，不是作者认证轨道。未知组不能随机切分并称为未见轨道验证。',
        '- 整录像留出是补充测试：相同轨道可能在不同参与者录像重复，不能据此声称跨场景泛化。',
        '', '## 工程目标与单位', '',
        '- camera speed / acceleration：渲染相机的源坐标单位/s、源坐标单位/s²，包含可能的头动；米制未确认。',
        '- camera angular speed：矩阵相对旋转的度/s。',
        '- candidate rig speed / acceleration：候选头部姿态分离后的源坐标运动；坐标校准尚未独立验证。',
        '- 这些目标描述运动，不是人体风险标签；FMS 不参与训练。不能把高分数解释为晕动症风险预测成功。',
        '', '## 评估汇总', '']
    aggregate_rows = []
    for split, targets in results.items():
        lines += [f'### {split}', '']
        table_rows = []
        for target, result in targets.items():
            for method, entry in result.get('methods', {}).items():
                pooled, macro = entry['pooled_oof'], entry['equal_fold_summary']
                record = dict(target=target, method=method, clips=pooled['n'], folds=entry['n_folds'],
                    pooled_MAE=pooled['mae'], pooled_RMSE=pooled['rmse'], pooled_Spearman=pooled['spearman'],
                    pooled_MAE_skill=pooled['mae_skill_vs_train_mean'],
                    equal_fold_MAE=macro['mae'], equal_fold_Spearman=macro['spearman'],
                    equal_fold_MAE_skill=macro['mae_skill_vs_train_mean'])
                table_rows.append(record)
                aggregate_rows.append(dict(split=split, **record))
            if result.get('status') != 'evaluated':
                lines.append(f'- `{target}`：{result.get("status")}。')
        lines += [md_table(pd.DataFrame(table_rows)), '']
    pd.DataFrame(aggregate_rows).to_csv(ROOT / f'results/{PREFIX}_summary_metrics.csv', index=False, encoding='utf-8-sig')
    lines += ['## 每折结果', '',
        '完整每折样本量、错误、训练错误、训练目标范围、特征删除及 LightGBM 分裂诊断见 `coaster_motion_trial_fold_metrics.csv`。', '',
        md_table(fold_frame[['split','target','method','held_out_group','n_train','n_test','test_mae','test_rmse','test_spearman','test_mae_skill_vs_train_mean']]) if not fold_frame.empty else '无可计算测试折。', '',
        '## 如何解释', '',
        '- MAE skill = 1 − 模型 MAE / 同折训练均值基线 MAE。正数表示比该基线好；负数表示更差。仍需比较 elapsed-only 和模型各折，排除时间过程线索。',
        '- 片段密集相邻且只来自少量录像，不报告把片段当独立样本的显著性 p 值。',
        '- 光流+画面性能好而光流单独表现弱，可能依赖轨道纹理或画面阶段；需要未见轨迹验证。',
        '- 单录像留出提高、候选家族留出失败，应优先扩大轨道覆盖而不是宣称泛化。',
        '- Full-fit 模型及特征重要性仅供接入代码演示。无 FMS、风险等级、人体安全阈值或原始 AUC 门控结论。', '',
        '## 固定超参数', '', '```json', json.dumps(LGBM, indent=2), '```', '',
        f'冻结定义 SHA256：`{metadata["frozen_definition_sha256"]}`。',
        f'输入 SHA256：`{metadata["input_sha256"]}`。', '']
    (ROOT / f'results/{PREFIX}_training_report.md').write_text('\n'.join(lines), encoding='utf-8')
    logger.info('DONE predictions=%s folds=%s models=%s strict=%s', len(predictions),len(fold_rows),len(models),strict_status)
    print(json.dumps(dict(n_ok_clips=len(okay), strict_status=strict_status, n_models=len(models), report=str(ROOT / f'results/{PREFIX}_training_report.md')), ensure_ascii=False))


if __name__ == '__main__':
    main()
