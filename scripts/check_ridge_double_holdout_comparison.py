"""Follow-up Ridge double holdout, added after observing primary comparisons.

Same inputs/parameters; exploratory sensitivity, not independent confirmation.
"""
from pathlib import Path
import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from train_rating_aggregation_comparison import load_methods,fit,metric_row,DOUBLE_HOLDOUT_METHODS,ROOT
from train_param_rating_model import digest,save_json

tag='rating_aggregation_comparison_2026-10-10'
out=ROOT/'results'/tag
cfg_path=ROOT/'configs'/f'{tag}.json'
cfg=json.loads(cfg_path.read_text(encoding='utf-8'))
frames,cohort,inputs=load_methods(cfg,ROOT/'features'/tag)
assert cohort.to_csv(index=False,float_format='%.17g')==pd.read_csv(out/'cohort.csv',dtype={c:str for c in ['game_name','session_id','participant_id','rating_event_id']},encoding='utf-8-sig').to_csv(index=False,float_format='%.17g')
plan=dict(status='planned',added_after_primary_results=True,reason='Ridge outperformed LightGBM in primary comparison; examine same additional protocol for all four registered input groups',
          claim='exploratory sensitivity, not independent winner confirmation',methods=DOUBLE_HOLDOUT_METHODS,model='ridge',alpha=10.,
          created_utc=datetime.now(timezone.utc).isoformat(),config_sha256=digest(cfg_path),cohort_sha256=digest(out/'cohort.csv'),
          script_sha256=digest(Path(__file__)),original_trainer_sha256=digest(ROOT/'scripts/train_rating_aggregation_comparison.py'))
save_json(out/'ridge_double_holdout_followup_manifest.json',plan)
y=cohort.ms_rating.to_numpy(float);games=cohort.game_name.to_numpy();parts=[];folds=[];audits=[]
for fold,(train,test) in enumerate(LeaveOneGroupOut().split(cohort,y,games),1):
    people=set(cohort.iloc[test].participant_id.dropna())-{'unknown','','nan'}
    keep=~cohort.iloc[train].participant_id.isin(people).to_numpy();purged=train[keep]
    assert not (set(cohort.iloc[purged].participant_id)&people)
    assert not (set(cohort.iloc[purged].game_name)&set(cohort.iloc[test].game_name))
    for method in DOUBLE_HOLDOUT_METHODS:
        pipe,detail=fit(frames[method].iloc[purged],y[purged],inputs[method]['requested_feature_columns'],'ridge')
        assert pipe is not None
        rows=cohort.iloc[test].copy();rows['prediction']=pipe.predict(frames[method].iloc[test][detail['input_feature_cols']])
        rows['method']=method;rows['model']='ridge';rows['fold']=fold;rows['held_out_game']=games[test[0]]
        rows['protocol']='leave_game_and_test_participants_out';rows['absolute_error']=(rows.prediction-rows.ms_rating).abs();parts.append(rows)
        metrics=metric_row(rows,method,'ridge',fold,games[test[0]]);metrics['protocol']='leave_game_and_test_participants_out'
        metrics.update(train_events_after_purge=len(purged),removed_train_events=len(train)-len(purged));folds.append(metrics)
        audits.append(dict(method=method,fold=fold,held_out_game=games[test[0]],train_events=len(purged),removed=len(train)-len(purged),**detail))
pred=pd.concat(parts,ignore_index=True);summary=[]
for method,g in pred.groupby('method',sort=False):
    assert len(g)==len(cohort) and g.rating_event_id.is_unique
    row=metric_row(g,method,'ridge');row['protocol']='leave_game_and_test_participants_out';summary.append(row)
pred.to_csv(out/'ridge_double_holdout_oof_predictions.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
pd.DataFrame(folds).to_csv(out/'ridge_double_holdout_fold_metrics.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
pd.DataFrame(summary).to_csv(out/'ridge_double_holdout_metrics.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
plan.update(status='complete',fold_audits=audits,metrics=summary)
save_json(out/'ridge_double_holdout_followup_manifest.json',plan)
print(pd.DataFrame(summary)[['method','model','mae','rmse','spearman','binary_auc']].to_string(index=False))
