"""Exploratory native texture statistics -> raw 1--5 rating regression.

Whole participants are held out. auto and manual records are evaluated separately;
neither is silently interpreted as verified/corrected ground truth. No tuning.
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
from scipy.stats import pearsonr, spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
METHODS = ('train_mean', 'train_median', 'elapsed_ridge', 'native_ridge',
           'native_lightgbm', 'native_elapsed_ridge', 'rgb_only_lightgbm', 'depth_only_lightgbm')
LGBM = dict(objective='regression', n_estimators=150, max_depth=3, num_leaves=7,
            learning_rate=.05, min_child_samples=5, reg_lambda=5.,
            feature_fraction=.9, verbosity=-1, n_jobs=2, random_state=SEED)


def metrics(y, p, constant=False):
    err = p - y
    corr_ok = not constant and len(y) >= 3 and np.ptp(y) > 0 and np.ptp(p) > 0
    return dict(n=len(y), mae=float(np.abs(err).mean()),
                rmse=float(np.sqrt(np.square(err).mean())),
                pearson=float(pearsonr(y, p).statistic) if corr_ok else None,
                spearman=float(spearmanr(y, p).statistic) if corr_ok else None,
                correlation_status='descriptive_no_independent_event_p_value' if corr_ok
                else 'NA_constant_or_insufficient')


def fit(data, features, method):
    requested = ([] if method.startswith('train_') else ['elapsed_sec'] if method == 'elapsed_ridge'
                 else features + ['elapsed_sec'] if method == 'native_elapsed_ridge'
                 else [f for f in features if f.startswith('rgb_')] if method == 'rgb_only_lightgbm'
                 else [f for f in features if f.startswith('native_depth_')] if method == 'depth_only_lightgbm' else features)
    retained = [f for f in requested if data[f].notna().any()]
    y = data.fms_raw.to_numpy(float)
    imp, scaler = None, None
    if not retained:
        model = DummyRegressor(strategy='median' if method == 'train_median' else 'mean')
        x = np.zeros((len(data), 1))
    else:
        imp = SimpleImputer(strategy='median')
        x = imp.fit_transform(data[retained])
        if method.endswith('lightgbm'):
            model = lgb.LGBMRegressor(**LGBM)
        else:
            scaler = StandardScaler()
            x = scaler.fit_transform(x)
            model = Ridge(alpha=10.)
    model.fit(x, y)
    tree_diagnostics = None
    if hasattr(model, 'booster_'):
        dumped = model.booster_.dump_model()
        tree_diagnostics = dict(tree_count=model.booster_.num_trees(),
                                leaf_counts=[t['num_leaves'] for t in dumped['tree_info']],
                                feature_importance_gain=dict(zip(retained, model.booster_.feature_importance(importance_type='gain').tolist())))
    return dict(model=model, imputer=imp, scaler=scaler, features=retained, tree_diagnostics=tree_diagnostics,
                dropped=[f for f in requested if f not in retained],
                low=float(y.min()), high=float(y.max()), n_train=len(data))


def predict(bundle, data):
    if bundle['features']:
        x = bundle['imputer'].transform(data[bundle['features']])
        if bundle['scaler'] is not None:
            x = bundle['scaler'].transform(x)
    else:
        x = np.zeros((len(data), 1))
    raw = np.asarray(bundle['model'].predict(x), float)
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite model predictions')
    return raw, np.clip(raw, bundle['low'], bundle['high'])


def comparison(model_rows, ref_rows):
    pairs = model_rows.merge(ref_rows[['event_id', 'prediction']], on='event_id',
                             suffixes=('', '_reference'), validate='one_to_one')
    if len(pairs) != len(model_rows) or len(pairs) != len(ref_rows):
        raise ValueError('Different events in paired model comparison')
    records = []
    for _, part in pairs.groupby('participant_id'):
        e = part.prediction - part.fms_raw
        r = part.prediction_reference - part.fms_raw
        records.append([len(part), np.abs(e).sum(), np.abs(r).sum(), (e ** 2).sum(), (r ** 2).sum()])
    a = np.asarray(records, float)
    sample = np.random.default_rng(SEED).integers(0, len(a), size=(2000, len(a)))
    sums = a[sample].sum(axis=1)
    dm = (sums[:, 1] - sums[:, 2]) / sums[:, 0]
    dr = np.sqrt(sums[:, 3] / sums[:, 0]) - np.sqrt(sums[:, 4] / sums[:, 0])
    total = a.sum(axis=0)
    return dict(delta_mae=float((total[1] - total[2]) / total[0]),
                delta_rmse=float(np.sqrt(total[3] / total[0]) - np.sqrt(total[4] / total[0])),
                delta_mae_ci95=np.quantile(dm, [.025, .975]).tolist(),
                delta_rmse_ci95=np.quantile(dr, [.025, .975]).tolist(),
                n_groups=len(a), resamples=2000, seed=SEED, retraining=False,
                warning='Descriptive whole-participant bootstrap; only five groups; not model-refit uncertainty')


def cell(value):
    return 'NA' if value is None else f'{value:.4f}' if isinstance(value, float) else str(value)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--features', type=Path, default=ROOT / 'features/rollercoaster_native_event_features.csv')
    ap.add_argument('--modalities', choices=['flow_depth', 'depth_rgb'], default='flow_depth')
    ap.add_argument('--output-prefix', default='rollercoaster_native')
    args = ap.parse_args()
    if not re.fullmatch(r'rollercoaster_[A-Za-z0-9_]+', args.output_prefix):
        ap.error('--output-prefix must start rollercoaster_ and be a plain name')
    prefix = args.output_prefix
    modality_text = '原生 depth JPEG + RGB 画面统计（原生 motion-flow 尚未解码，未使用）' if args.modalities == 'depth_rgb' else '原生 motion-flow + depth 统计'
    out = ROOT / 'results'
    log = logging.getLogger('rollercoaster_native_training')
    log.setLevel(logging.INFO)
    for handler in [logging.StreamHandler(sys.stdout), logging.FileHandler(out / f'{prefix}_training_log.txt', mode='a', encoding='utf-8')]:
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        log.addHandler(handler)
    data = pd.read_csv(args.features, dtype={'event_id': str, 'session_id': str, 'participant_id': str})
    required = {'event_id', 'participant_id', 'session_id', 'fms_raw', 'rating_source', 'status', 'elapsed_sec'}
    if not required.issubset(data):
        raise ValueError(f'Missing {required - set(data)}')
    if data.event_id.duplicated().any():
        raise ValueError('Duplicate event IDs')
    for key in ('event_id', 'session_id', 'participant_id'):
        if data[key].isna().any() or data[key].str.strip().eq('').any():
            raise ValueError(f'Missing {key}')
    if data.groupby('session_id').participant_id.nunique().ne(1).any():
        raise ValueError('A session is assigned to multiple participants')
    allowed_prefixes = ('native_depth_', 'native_flow_') if args.modalities == 'flow_depth' else ('native_depth_', 'rgb_')
    forbidden = re.compile(r'(^|_)(id|path|participant|session|event|fms|score|rating|target|label|coverage|support|duration|timestamp|count|frame|frames|times|files|bytes|sha256|sha|hash)($|_)')
    features = sorted(c for c in data if c.startswith(allowed_prefixes) and not forbidden.search(c))
    expected = ['native_depth_', 'native_flow_' if args.modalities == 'flow_depth' else 'rgb_']
    if not features or not all(any(c.startswith(p) for c in features) for p in expected):
        raise ValueError('Declared modalities must both have verified feature columns; no silent modality substitution')
    for c in features + ['elapsed_sec', 'fms_raw']:
        data[c] = pd.to_numeric(data[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    if data.loc[data.status.eq('ok'), features].isna().all(axis=1).any():
        raise ValueError('An eligible event has no visual features')
    eligible = data.status.eq('ok') & data.fms_raw.between(1, 5)
    for modality in expected:
        columns = [f for f in features if f.startswith(modality)]
        if data.loc[eligible, columns].isna().all(axis=1).any():
            raise ValueError(f'Eligible event missing entire modality {modality}')
    definition = dict(created_utc=datetime.now(timezone.utc).isoformat(), client_date='2026-10-10',
                      input=str(args.features.resolve()), sha256=hashlib.sha256(args.features.read_bytes()).hexdigest(),
                      seed=SEED, target='unaltered recorded 1--5 score; no risk bins',
                      cohorts=['manual', 'auto'], source_semantics='manual-only matches DataCard descriptive statistics; inferred primary, not authenticated gold standard; auto separate sensitivity',
                      modalities=args.modalities, modality_description=modality_text,
                      split='leave_one_participant_out, all recording events together',
                      methods=list(METHODS), lightgbm=LGBM, ridge_alpha=10., features=features,
                      preprocessing='fit median imputer and scaler inside training fold only; drop train-allmissing columns',
                      leaf_size_reason='min_child_samples=5 fixed before first fit because only23 manual records; old City pilot used20, so not a controlled dataset-only comparison',
                      clip='training label range; raw predictions retained', no_tuning=True,
                      protocol='user-authorized raw-rating exploratory regression; original three-class AUC gate not assessed',
                      versions=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__,
                                    sklearn=sklearn.__version__, lightgbm=lgb.__version__, scipy=scipy.__version__),
                      eligible_by_source=data.loc[eligible].rating_source.value_counts().to_dict())
    definition_path = ROOT / f'audit/{prefix}_training_definition.json'
    definition_path.write_text(json.dumps(definition, ensure_ascii=False, indent=2), encoding='utf-8')
    summaries, predictions, fold_rows, models = {}, [], [], {}
    for source in definition['cohorts']:
        cohort = data.loc[eligible & data.rating_source.eq(source)].copy()
        groups = sorted(cohort.participant_id.unique())
        if len(groups) < 3:
            summaries[source] = dict(status='insufficient_groups', n=len(cohort), n_groups=len(groups))
            log.info('SOURCE %s skipped: %s groups', source, len(groups))
            continue
        entries = {}
        for method in METHODS:
            rows, folds = [], []
            for group in groups:
                train = cohort.loc[cohort.participant_id.ne(group)]
                test = cohort.loc[cohort.participant_id.eq(group)]
                if set(train.session_id) & set(test.session_id):
                    raise ValueError('Recording leakage')
                bundle = fit(train, features, method)
                raw, pred = predict(bundle, test)
                _, tp = predict(bundle, train)
                y = test.fms_raw.to_numpy(float)
                m = metrics(y, pred, method.startswith('train_'))
                tm = metrics(train.fms_raw.to_numpy(float), tp, method.startswith('train_'))
                record = dict(source=source, method=method, held_out_participant=group,
                              n_train=len(train), n_test=len(test), test=m, train_in_sample=tm,
                              raw_test=metrics(y, raw, method.startswith('train_')),
                              tree_diagnostics=bundle['tree_diagnostics'],
                              dropped_all_missing=bundle['dropped'], n_test_clipped=int((raw != pred).sum()),
                              train_fms_min=bundle['low'], train_fms_max=bundle['high'])
                folds.append(record)
                fold_rows.append({**{k: v for k, v in record.items() if not isinstance(v, dict)},
                                  **{f'test_{k}': v for k, v in m.items()},
                                  **{f'train_{k}': v for k, v in tm.items()}})
                for pos, event in enumerate(test.to_dict('records')):
                    rows.append(dict(source=source, method=method, event_id=event['event_id'],
                                     session_id=event['session_id'], participant_id=event['participant_id'],
                                     fms_raw=event['fms_raw'], elapsed_sec=event['elapsed_sec'],
                                     prediction_raw=float(raw[pos]), prediction=float(pred[pos])))
                log.info('source=%s method=%s P%s train=%s test=%s MAE=%.4f RMSE=%.4f', source, method, group, len(train), len(test), m['mae'], m['rmse'])
            rows_df = pd.DataFrame(rows)
            pooled = metrics(rows_df.fms_raw.to_numpy(float), rows_df.prediction.to_numpy(float), method.startswith('train_'))
            macro = {k: float(np.mean([f['test'][k] for f in folds if f['test'][k] is not None]))
                     if any(f['test'][k] is not None for f in folds) else None
                     for k in ('mae', 'rmse', 'pearson', 'spearman')}
            entries[method] = dict(pooled_oof=pooled, macro_participant=macro, folds=folds,
                                   train_mean_mae=float(np.mean([f['train_in_sample']['mae'] for f in folds])))
            predictions += rows
            models[f'{source}:{method}'] = fit(cohort, features, method)
        source_oof = pd.DataFrame([r for r in predictions if r['source'] == source])
        for method, entry in entries.items():
            current = source_oof.loc[source_oof.method.eq(method)]
            for ref in ('train_median', 'train_mean', 'elapsed_ridge'):
                entry[f'versus_{ref}'] = comparison(current, source_oof.loc[source_oof.method.eq(ref)])
        summaries[source] = dict(status='evaluated', n=len(cohort), n_groups=len(groups),
                                 counts=cohort.fms_raw.value_counts().sort_index().to_dict(), methods=entries)
    summary = dict(definition=definition, raw_rows=len(data), status_counts=data.status.value_counts().to_dict(),
                   native_feature_count=len(features), cohorts=summaries)
    (out / f'{prefix}_metrics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    pd.DataFrame(predictions).to_csv(ROOT / f'features/{prefix}_oof_predictions.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(fold_rows).to_csv(out / f'{prefix}_fold_metrics.csv', index=False, encoding='utf-8-sig')
    joblib.dump(dict(definition=definition, models=models, warning='Exploratory full-cohort demonstration fits; not validated risk engine'), ROOT / f'models/{prefix}.pkl')
    lines = ['# 过山车原生纹理小模型试验', '', '**探索性流程验证，非最终结论。**', '',
             f'- 输入：{modality_text}；不是原论文双分支 3D CNN 的精确复现。',
             '- 目标：原始 1–5 自报记录；manual 与数据卡统计完全吻合作为主对照依据，但源语义尚未认证；auto 单列敏感性，不混合近邻记录。',
             '- 整参与者留出，每人的全部片段同折；单游戏不能验证跨游戏或开发时迁移。',
             '- 只使用评分前 3 秒的实际帧；不补分、不用评分历史/用户姿态/矩阵/身份作为预测特征。',
             '- 预处理仅训练折拟合；固定参数不调优。最终全数据示范拟合的表现不作为测试指标。',
             '- 原 PROTOCOL 的三分类 AUC 门控未评估。论文 accuracy=0.70 定义不明，不能与 MAE 或相关系数直接比较。',
             f'- 原始记录 {len(data)}，状态 {summary["status_counts"]}；视觉统计特征 {len(features)} 列。',
             f'- 首次训练之前保存 {definition_path.name}，输入 SHA256 {definition["sha256"]}。', '']
    for source, cohort in summaries.items():
        lines += [f'## {source} 记录', '', f'- {cohort["n"]} 条记录，{cohort["n_groups"]} 位参与者；这些记录不等于同数目的独立样本。']
        if cohort['status'] != 'evaluated':
            lines += [f'- {cohort["status"]}', '']
            continue
        lines += [f'- 评分频数：{cohort["counts"]}。', '',
                  '| 模型 | pooled MAE ↓ | RMSE ↓ | Pearson | Spearman | 等权参与者 MAE |',
                  '|---|---:|---:|---:|---:|---:|']
        for method, entry in cohort['methods'].items():
            m = entry['pooled_oof']
            lines.append('| ' + ' | '.join([method] + [cell(m[k]) for k in ('mae', 'rmse', 'pearson', 'spearman')] + [cell(entry['macro_participant']['mae'])]) + ' |')
        lines += ['', '### 组合视觉 LightGBM 对照', '']
        ent = cohort['methods']['native_lightgbm']
        for ref in ('train_median', 'train_mean', 'elapsed_ridge'):
            c = ent[f'versus_{ref}']
            lines.append(f'- 相对 {ref}：ΔMAE {c["delta_mae"]:.4f}，描述性整参与者 bootstrap 95%区间 {c["delta_mae_ci95"]}；负值表示模型误差更低。')
        lines += ['', '### 各参与者 LightGBM 留出结果', '', '| 留出参与者 | 训练/测试记录 | MAE | RMSE |', '|---|---|---:|---:|']
        for f in ent['folds']:
            lines.append(f'| P{f["held_out_participant"]} | {f["n_train"]}/{f["n_test"]} | {f["test"]["mae"]:.4f} | {f["test"]["rmse"]:.4f} |')
        lines += ['', f'- LightGBM 平均训练折 MAE {ent["train_mean_mae"]:.4f}；平均留出参与者 MAE {ent["macro_participant"]["mae"]:.4f}。',
                  '- 首次训练前固定 min_child_samples=5：23条 manual 记录用旧参数20无法形成两个叶。两个 source 使用相同参数，不根据指标调优；此前 City 使用20，所以不是只更换数据的受控对照。', '']
    lines += ['## 解释边界', '', '- 只有5位参与者，置信区间仅作描述；bootstrap 未重新训练，不能涵盖全部模型不确定性。',
              '- 自动识别/人工记录的 codebook 未找到，近邻冲突记录原样保留并按 source 分开；仍可能存在源内冗余。',
              '- 原生纹理编码及有效视野遮罩的限定见提取报告；深度统计不等于米，flow统计不冒充相机速度。',
              '- 尚未重建原论文471条评分/5轨道映射。没有证明预暴露设计因果关系或修改建议有效。',
              '- 下一步先确认作者的评分筛选、纹理解码及轨道映射，再决定是否投入与论文架构一致的视频训练。', '']
    (out / f'{prefix}_training_report.md').write_text('\n'.join(lines), encoding='utf-8')
    log.info('DONE %s', {s: {k: c.get(k) for k in ('status', 'n', 'n_groups')} for s, c in summaries.items()})


if __name__ == '__main__':
    main()
