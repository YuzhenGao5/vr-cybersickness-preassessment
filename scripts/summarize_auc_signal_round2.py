#!/usr/bin/env python3
"""Summarize real frozen outputs; never train, choose parameters, or alter scores."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
TAG='auc_signal_round2_2026-10-10'
RES=ROOT/'results'/TAG
FEATURES=ROOT/'features'/TAG
NAMES={'scene_multiscale':'多尺度相机/场景','scene_head_multiscale':'场景+头部',
       'robust_scene_head':'稳健/累计场景+头部','robust_joint':'稳健/累计+控制器',
       'elapsed_only':'仅体验时长','nested_selected':'内层自动选择','baseline':'训练先验基线'}

def fmt(v):
    if v is None or pd.isna(v):return '无法计算'
    return f'{float(v):.4f}'

def table(frame,columns):
    lines=['| '+' | '.join(label for key,label in columns)+' |','|'+'|'.join('---' for _ in columns)+'|']
    for row in frame.to_dict('records'):
        cells=[]
        for key,label in columns:
            value=row.get(key)
            if key=='method':value=NAMES.get(value,value)
            if isinstance(value,(float,np.floating)):value=fmt(value)
            cells.append(str(value).replace('|','\\|').replace('\n',' '))
        lines.append('| '+' | '.join(cells)+' |')
    return '\n'.join(lines)

def main():
    manifest=json.loads((RES/'training_manifest.json').read_text(encoding='utf-8'))
    check=json.loads((RES/'independent_check.json').read_text(encoding='utf-8'))
    if manifest['status']!='complete' or check['status']!='passed':raise ValueError('training/check incomplete')
    primary=pd.read_csv(RES/'metrics.csv');double=pd.read_csv(RES/'double_holdout_metrics.csv')
    fold=pd.read_csv(RES/'fold_metrics.csv');doublefold=pd.read_csv(RES/'double_holdout_fold_metrics.csv')
    old=pd.read_csv(RES/'previous_regression_references.csv')
    joined=primary.merge(double[['method','model','binary_auc','mean_valid_fold_auc']],on=['method','model'],suffixes=('','_double'),validate='one_to_one')
    joined.to_csv(RES/'final_comparison.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    candidate=joined[joined.method.ne('baseline')];best=candidate.loc[candidate.binary_auc.idxmax()]
    strictbest=candidate.loc[candidate.binary_auc_double.idxmax()]
    selected=joined[joined.method.eq('nested_selected')].iloc[0]
    data=pd.read_csv(FEATURES/'events.csv');qc=pd.read_csv(FEATURES/'support_quality.csv')
    preparation=json.loads((FEATURES/'preparation_manifest.json').read_text(encoding='utf-8'))
    counters={k:sum(m['counters'].get(k,0) for m in preparation['sessions_metadata']) for k in ['position_jumps','derivative_interval_resets','invalid_camera_rows','rotation_rate_over_flag','invalid_head_quaternion']}
    cfg=json.loads((ROOT/'configs'/f'{TAG}.json').read_text(encoding='utf-8'))
    feature_summary=[]
    for name,spec in cfg['methods'].items():
        a=data[spec['feature_columns']];constant=a.nunique(dropna=True)<=1
        feature_summary.append(dict(method=name,registered=len(a.columns),global_constant=int(constant.sum()),
           missing_rate=float(a.isna().sum().sum()/a.size),all_missing=int(a.isna().all().sum())))
    pd.DataFrame(feature_summary).to_csv(RES/'feature_quality_summary.csv',index=False,encoding='utf-8-sig')
    increments=check.get('paired_participant_bootstrap_vs_elapsed',[])
    for row in increments:
        for key in ['protocol','method','model']:row.setdefault(key,'unknown')
    increment_table=pd.DataFrame([r for r in increments if r['model']=='logistic' and r['method'] in ['scene_multiscale','scene_head_multiscale','robust_scene_head','robust_joint']])
    importance_path=RES/'feature_importance.csv'
    importance=pd.read_csv(importance_path).head(10) if importance_path.exists() else None
    bestpass=float(best.binary_auc)>=.70
    selectedpass=float(selected.binary_auc)>=.70
    lines=['# 第二轮方案对照与 AUC 目标判断','',
      '**探索性，非最终结论。** 所有数字来自固定真实数据和折外预测。未下载新数据、未训练视频模型、未执行 git commit。','',
      '## 结果','',
      f"目标：AUC≥0.70。观察到的最高合并 OOF AUC **{fmt(best.binary_auc)}**，方案 **{NAMES.get(best.method,best.method)} / {best.model}**；游戏折 AUC 均值 {fmt(best.mean_valid_fold_auc)}。{'这一组合达到合并 AUC 目标。' if bestpass else '本轮没有达到合并 AUC 目标。'}",
      f"同一组合在隔离测试参与者后合并AUC降至 **{fmt(best.binary_auc_double)}**。较严格评估中最高组合是 {NAMES.get(strictbest.method,strictbest.method)} / {strictbest.model}，合并AUC **{fmt(strictbest.binary_auc_double)}**、游戏均值 {fmt(strictbest.mean_valid_fold_auc_double)}。",
      f"完整的内层自动选择流程：合并 AUC **{fmt(selected.binary_auc)}**，游戏折均值 **{fmt(selected.mean_valid_fold_auc)}**；隔离测试参与者后分别为 **{fmt(selected.binary_auc_double)} / {fmt(selected.mean_valid_fold_auc_double)}**。{'合并值达到0.70。' if selectedpass else '合并值未达到0.70。'}",
      f"该自动选择流程的内层平均AUC为 {fmt(fold.loc[fold.method.eq('nested_selected'),'inner_mean_game_auc'].mean())}，外层游戏均值为 {fmt(selected.mean_valid_fold_auc)}；训练游戏中的较高结果未稳定转移到未见游戏。",
      f"最高观察组合的准确率 {fmt(best.accuracy)}、F1 macro {fmt(best.f1_macro)}、AP {fmt(best.average_precision)}、Brier {fmt(best.brier)}；AUC的session簇区间 [{fmt(best.auc_ci_low)}, {fmt(best.auc_ci_high)}] 是现有游戏条件下的探索区间，区间包含0.70不等于已达到0.70。",
      '最高单组合按已报告外层结果描述，不作为独立无偏获胜证明；内层自动选择的每一折方案只由训练游戏选择，完整保留其表现。两类AUC是不同统计，不能互换。','',
      '## 数据与有理由的处理','',
      f"- 同一共同队列：{len(data):,} 次真实评分，{data.session_id.nunique()} session、{data.game_name.nunique()} 游戏、{data.participant_id.nunique()} 个现代数据参与者编号。不是1,024名被试，也不是全量VR.net。",
      f"- 评分分布：{data.ms_rating.value_counts().sort_index().to_dict()}；固定标签 ≤2 为0，>2 为1。负例 {int((data.ms_rating<=2).sum())}，正例 {int((data.ms_rating>2).sum())}。",
      '- 每个真实评分只保留一个样本。全部输入在评分之前，禁止真实历史/当前/未来评分作为特征；比较过程中不改队列、不选种子、不调评分阈值。',
      '- 直接分类匹配本轮AUC目标；原评分回归作为独立参考保留，本轮不声称重新得到准确的连续评分。',
      '- 正值固定log1p，训练折中位数补缺/常量筛除，逻辑回归训练折标准化；训练折session等权，必要时加训练折类别平衡。',
      '- 过去5/30/60秒的时间加权均值/标准差/中位数/P90，加固定趋势和自session开始的运动积分；Σ速度×有效dt，缺失期不补零，不按行数累加。',
      f"- 检测并切断相机位置跳变 {counters['position_jumps']} 处、间隔重置 {counters['derivative_interval_resets']} 处。坏相机矩阵 {counters['invalid_camera_rows']}；超过720°/s的旋转率 {counters['rotation_rate_over_flag']} 处仅标记，未按目标AUC裁剪。",
      f"- 头部四元数异常 {counters['invalid_head_quaternion']} 处；夹角采用归一化四元数绝对内积，避免符号翻转造出假运动。",
      f"- 累计相机有效覆盖率中位数 {fmt(qc.cumulative_camera_speed_coverage.median())}，最小 {fmt(qc.cumulative_camera_speed_coverage.min())}；头部位置覆盖率中位数 {fmt(qc.cumulative_head_pose_speed_coverage.median())}，最小 {fmt(qc.cumulative_head_pose_speed_coverage.min())}。积分只代表有效观测区间，未按缺失比例外推；低覆盖会低估累计量。",
      '- FOV近乎恒定仍如实保留，由训练折去掉常量，不人为制造FOV差异；source单位未认证为米，不称m/s。眼动编码未完全认证，本轮排除。',
      '- 数据表保留辅助审计列和上轮中间字段，但模型只能读取配置列出的白名单，不使用nearest_*含评分后观测的字段、QC、ID或标签。','',
      '## 全部模型对照','',
      table(joined,[('method','输入'),('model','模型'),('binary_auc','合并AUC'),('mean_valid_fold_auc','游戏平均AUC'),('binary_auc_double','游戏+参与者AUC'),('mean_valid_fold_auc_double','隔离游戏平均AUC'),('average_precision','AP'),('f1_macro','F1 macro'),('brier','Brier')]),'',
      'AP的无技巧参考是正例比例；输出未做独立校准，类别平衡后的分数不能直接当成可靠的个人患病概率。','',
      '## 与上轮评分回归比较','',
      table(old,[('reference','参考'),('binary_auc','合并AUC'),('mean_valid_fold_auc','游戏平均AUC'),('average_precision','AP')]),'',
      '目标和训练方法不同；这部分只比较相同真实事件的排名，不把回归分数当概率。','',
      '## 内层自动选择的逐游戏结果','',
      table(fold[fold.method.eq('nested_selected')],[('held_out_game','游戏'),('events','评分数'),('positive_events','正例'),('binary_auc','AUC'),('f1_macro','F1 macro'),('selected_input_method','内层所选输入'),('selected_model_family','内层所选模型')]),'',
      '内层3折GroupKFold按游戏分组，各验证游戏单独计算AUC后等权平均选候选；外层Leave-One-Game-Out。另在外层和内层同时去除验证参与者。固定16组参数、5组输入，不按外层结果继续追加方案。','',
      '## 与仅时长的增量','',
      '独立检查JSON提供各输入相对同模型族elapsed_only的配对参与者簇bootstrap。固定OOF、不重训、条件于已有游戏；不能把其区间解释为新游戏泛化保证。',
      table(increment_table,[('protocol','协议'),('method','输入'),('auc_difference','ΔAUC'),('difference_ci_low','区间下界'),('difference_ci_high','上界'),('mean_valid_game_auc_difference','游戏均值ΔAUC')]),
      '详见 independent_check.json 的 paired_participant_bootstrap_vs_elapsed。增量区间跨0时，不能确认新增特征优于时长。','',
      '## 特征解释','']
    if importance is not None:
        explained=manifest['full_models']['nested_selected/selected']
        lines += [f"归因对象：全量数据内层选择的 {explained['selected_input_method']} / {explained['selected_model_family']}；这与对照表中观察到最高外层AUC的组合可能不同，不能互用其SHAP值。",
                  '使用最终全量拟合模型的SHAP，仅用于描述模型，非独立验证，非因果。',table(importance,[('rank','排名'),('feature','特征'),('mean_abs_shap','平均绝对SHAP'),('raw_feature_shap_spearman','值与SHAP相关'),('direction','方向')])]
    else:lines += ['查看 feature_interpretation.md；若计算不可用，保留明确原因，不编造贡献值。']
    lines += ['', '## 对项目的判断与下一步','',
      '1. 这一轮检查可否从现有运行数据区分真实评分>2。评分前头动/控制器属于用户交互或反应数据；矩阵相机也可能含头动，不能宣称都是可编辑Unity设计参数。',
      '2. 达到0.70只能作为继续投入的探索证据。已经反复查看同一批游戏，即使嵌套交叉验证也不能消除以前方案选择的影响；可靠声明需未参与选择的新游戏/参与者。',
      '3. 若仅时长已同样强，应优先解释累积暴露/评分协议；若加入头部或控制器有效，应明确该模型需要用户运行数据，不能直接交付无人开发时Unity引擎。',
      '4. 现有FOV变化不足、物体数为渲染矩阵条目、真实位置单位与运动来源尚需认证。若继续建设纯设计引擎，先建立可控导航路径和可编辑参数与这些测量的对应，随后独立验证修改建议的效果。',
      '5. 当前不追加数据下载或视频训练。先据此对照决定下一轮；不通过制造样本/阈值变化来冲AUC。','',
      f"独立检查：{check['status']}，{check['checks_passed']}项通过，{check['checks_failed']}项失败。数据/源/代码哈希、逐折身份、参数选择和全部OOF指标均保留。原PROTOCOL三分类门控未被本轮二分类替代。",
      '', '## 方法依据','',
      '[嵌套交叉验证](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)用于将模型选择限制于训练数据；[LightGBM官方权重说明](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)提示加权分数的概率校准问题。时间窗口、积分和稳健特征是本项目预登记假设，不声称文献认证其效果。',
      '', '输出：final_comparison.csv、report.md、逐折/OOF/内层选择文件、模型bundle；完整原始结果不删。']
    (RES/'final_comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # All input/model pairs retained. This visual displays both protocols.
    plot=joined[joined.method.ne('baseline')].copy()
    plot['label']=plot.method+' / '+plot.model
    plt.rcParams.update({'font.size':9})
    fig,axes=plt.subplots(1,2,figsize=(14,9),sharey=True)
    y=np.arange(len(plot))
    for ax,pcol,fcol,title in [(axes[0],'binary_auc','mean_valid_fold_auc','Unseen game'),(axes[1],'binary_auc_double','mean_valid_fold_auc_double','Unseen game + test participant purge')]:
        ax.scatter(plot[pcol],y,label='Pooled OOF',s=37,color='#1864ab')
        ax.scatter(plot[fcol],y,label='Equal game mean',s=37,marker='x',color='#d9480f')
        ax.axvline(.7,color='#777',linestyle='--',linewidth=1);ax.axvline(.5,color='#bbb',linewidth=.7)
        ax.set_xlim(max(0,min(.35,float(plot[[pcol,fcol]].min().min())-.04)),min(1,max(.85,float(plot[[pcol,fcol]].max().max())+.04)))
        ax.set_xlabel('AUC');ax.set_title(title);ax.grid(axis='x',alpha=.15)
        ax.set_yticks(y);ax.set_yticklabels(plot.label)
    axes[0].invert_yaxis()
    axes[0].legend(loc='lower left');fig.suptitle('Exploratory classification: 1024 real ratings, 86 sessions, 12 games')
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(RES/'comparison.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(best_observed=best[['method','model','binary_auc','mean_valid_fold_auc','binary_auc_double']].to_dict(),nested_selected=selected[['binary_auc','mean_valid_fold_auc','binary_auc_double','mean_valid_fold_auc_double']].to_dict()),ensure_ascii=False))
    print(RES/'final_comparison.md')

if __name__=='__main__':main()
