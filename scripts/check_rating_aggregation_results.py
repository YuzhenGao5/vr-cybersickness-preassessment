"""Recalculate saved comparison predictions without refitting any estimator."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,pearsonr
from sklearn.metrics import roc_auc_score

root=Path(__file__).resolve().parents[1]
out=root/'results/rating_aggregation_comparison_2026-10-10'
cohort=pd.read_csv(out/'cohort.csv',encoding='utf-8-sig').set_index('rating_event_id')
assert cohort.index.is_unique
checks=[]
def verify(pred_file,metric_file,fold_file):
    predictions=pd.read_csv(out/pred_file,encoding='utf-8-sig')
    metrics=pd.read_csv(out/metric_file,encoding='utf-8-sig')
    folds=pd.read_csv(out/fold_file,encoding='utf-8-sig')
    for (method,model), group in predictions.groupby(['method','model']):
        assert group.rating_event_id.is_unique and set(group.rating_event_id)==set(cohort.index)
        assert group.game_name.eq(group.held_out_game).all()
        assert np.allclose(group.ms_rating,cohort.loc[group.rating_event_id].ms_rating,rtol=0,atol=0)
        y=group.ms_rating.to_numpy();p=group.prediction.to_numpy()
        assert np.isfinite(p).all(),(method,model,'incomplete predictions')
        r=metrics[(metrics.method==method)&(metrics.model==model)].iloc[0]
        values=dict(mae=float(np.mean(np.abs(p-y))),rmse=float(np.sqrt(np.mean((p-y)**2))),
                    binary_auc=float(roc_auc_score(y>2,p)) if len(np.unique(y>2))>1 else np.nan,
                    spearman=float(spearmanr(y,p).statistic) if np.ptp(p)>1e-12 and np.ptp(y)>0 else np.nan,
                    pearson=float(pearsonr(y,p).statistic) if np.ptp(p)>1e-12 and np.ptp(y)>0 else np.nan)
        for key,value in values.items():
            assert np.isclose(value,r[key],equal_nan=True,rtol=1e-10,atol=1e-10),(method,model,key,value,r[key])
        assert r['events']==len(cohort)
        mf=folds[(folds.method==method)&(folds.model==model)]
        assert len(mf)==cohort.game_name.nunique()
        for game,part in group.groupby('game_name'):
            fr=mf[mf.held_out_game==game].iloc[0]
            assert np.isclose(np.mean(np.abs(part.prediction-part.ms_rating)),fr.mae,rtol=1e-10,atol=1e-10)
            if part.ms_rating.gt(2).nunique()>1:
                assert np.isclose(roc_auc_score(part.ms_rating.gt(2),part.prediction),fr.binary_auc,rtol=1e-10,atol=1e-10)
        checks.append(dict(file=pred_file,method=method,model=model,events=len(group),metrics=values))
    return predictions
primary=verify('oof_predictions.csv','metrics.csv','fold_metrics.csv')
sensitivity=verify('double_holdout_oof_predictions.csv','double_holdout_metrics.csv','double_holdout_fold_metrics.csv')
ridge_sensitivity=verify('ridge_double_holdout_oof_predictions.csv','ridge_double_holdout_metrics.csv','ridge_double_holdout_fold_metrics.csv')
manifest=json.loads((out/'training_manifest.json').read_text(encoding='utf-8'))
assert manifest['status']=='complete'
config=manifest['config']
assert len(primary.groupby(['method','model']))==len(config['methods'])*len(config['models'])+2
result=dict(status='passed',common_events=len(cohort),models_and_baselines_checked=len(checks),checks=checks,
            scope='Independent saved OOF metric and event/fold identity recalculation; no estimator refit')
(out/'output_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False))
