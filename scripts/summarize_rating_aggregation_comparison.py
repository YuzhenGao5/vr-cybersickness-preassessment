"""Create the user-facing comparison from saved, checked predictions only."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_rating_aggregation_comparison import delta_bootstrap,DESCRIPTIONS

root=Path(__file__).resolve().parents[1];tag='rating_aggregation_comparison_2026-10-10'
out=root/'results'/tag
m=pd.read_csv(out/'metrics.csv');r=pd.read_csv(out/'ridge_double_holdout_metrics.csv')
groups=pd.read_csv(out/'score_group_metrics.csv');pred=pd.read_csv(out/'oof_predictions.csv')
manifest=json.loads((out/'training_manifest.json').read_text(encoding='utf-8'))
assert json.loads((out/'output_check.json').read_text(encoding='utf-8'))['status']=='passed'
def pick(method,model='ridge'):return m[(m.method==method)&(m.model==model)].iloc[0]
def fmt(v):return '不可计算' if pd.isna(v) else f'{v:.4f}'
def md_table(frame,cols,labels):
    lines=['| '+' | '.join(labels)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for _,row in frame.iterrows():
        lines.append('| '+' | '.join(fmt(row[c]) if isinstance(row[c],(float,np.floating)) else str(row[c]) for c in cols)+' |')
    return lines

extra=[]
for method,ref in [('multi_scene_head','elapsed_only'),('multi_scene_head','multi_scene'),('multi_scene','elapsed_only')]:
    a=pred[(pred.method==method)&(pred.model=='ridge')];b=pred[(pred.method==ref)&(pred.model=='ridge')]
    d=delta_bootstrap(a,b,1000,42)
    extra.append(dict(method=method,reference=ref,model='ridge',mae_improvement=d['mae_improvement_over_nearest_mean'],
                      ci_low=d['nearest_mean_mae_improvement_ci_low'],ci_high=d['nearest_mean_mae_improvement_ci_high'],
                      posthoc_diagnostic=True,draws=1000,seed=42))
pd.DataFrame(extra).to_csv(out/'incremental_comparison.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
head=pick('multi_scene_head');time=pick('elapsed_only');scene=pick('multi_scene')
head_sens=r[r.method=='multi_scene_head'].iloc[0]
main_names=['nearest_mean','nearest_pooled','past5_scene','past30_mean_only','past30_scene','past60_scene',
            'multi_scene','multi_user','multi_joint','multi_joint_noelapsed','multi_scene_head','elapsed_only']
table_rows=[]
for name in main_names:
    for model in ['lightgbm','ridge']:
        row=pick(name,model).to_dict();row['description']=DESCRIPTIONS[name];table_rows.append(row)
labels=['Window mean (diagnostic)','Window pooled (diagnostic)','Past 5s scene','Past 30s mean','Past 30s scene','Past 60s scene',
        'Multi scene + time','Multi user + time','Multi joint + time','Multi joint, no time','Multi scene + head + time','Time only']
fig,axes=plt.subplots(1,2,figsize=(12,7.4),sharey=True)
positions=np.arange(len(main_names))
for ax,col,title in zip(axes,['mae','binary_auc'],['MAE (lower is better)','Pooled OOF AUC for score > 2']):
    for model,offset,color in [('lightgbm',-.12,'#4777b2'),('ridge',.12,'#cc6d32')]:
        ax.scatter([pick(n,model)[col] for n in main_names],positions+offset,label=model,color=color,s=34)
    ax.set_title(title);ax.grid(axis='x',alpha=.22);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_yticks(positions,labels);axes[0].invert_yaxis()
fig.legend(*axes[1].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False)
axes[1].axvline(.5,color='gray',linestyle='--',linewidth=1)
fig.suptitle('Exploratory event-level comparison | 1,024 reports | 12 games | LOGO')
fig.tight_layout(rect=(0,.055,1,.97));fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)

lines=['# 多种评分事件聚合与联合输入：最终对照', '',
       '**探索性，非最终结论。存在弱排序信号，但高评分仍明显低估；未证实用户交互带来稳定的独立增益。**','',
       '## 数据与执行','',
       '- 从本地86个session、12个游戏的433,677条原始运行记录重新提取；未下载新数据，未读取视频，未执行Git提交。',
       '- 原轮1,039个真实评分事件均找到，共同有效1,024个（69个参与者编号）；排除15个，其中4个评分在session起始时刻，其余11个评分前5秒没有场景观测。',
       '- 一行对应一次真实评分，不复制标签生成新的评分事件；原始评分保持不变。分布：1分584、2分241、3分142、4分47、5分10。',
       '- 主比较固定12种方法、LightGBM和Ridge各一种参数，另有均值/中位数基线；同一事件集合、同一游戏留出12折。训练折内补缺和常量筛选，Ridge训练折内缩放。',
       '- 评分前窗口是[t-W,t)，W=5/30/60秒，排除评分时刻及之后的观测。最近窗口池含评分后记录，只作旧口径诊断。',
       '- 场景测量：相机速度、三维向量加速度、旋转率、对象矩阵条目数及FOV；FOV恒定在训练中剔除，无原始flow/depth字段。头动：head_vel/head_angvel范数；控制器：扳机/握持均值、摇杆幅值及变化率。gaze和face未纳入。',
       '- elapsed是实际已体验时长，不是Unity设计预设时长；相机运动混合头动和导航，坐标与头动速度单位未认证。','',
       '## 主要发现','',
       f'1. 单纯合并最近窗口不能稳定改善。最近均值LightGBM AUC={fmt(pick("nearest_mean","lightgbm").binary_auc)}；均值/最大值/标准差池LightGBM AUC={fmt(pick("nearest_pooled","lightgbm").binary_auc)}。这些不是严格历史预测。',
       '2. 多尺度与简单正则线性模型更有苗头。多尺度场景Ridge合并AUC0.6374；场景+头动+时长Ridge具有主表最低模型MAE0.7410、最高Spearman0.2065，AUC0.6369。两个方法并非各项全面胜出。',
       '3. 全部用户字段没有稳定增益：多尺度场景+全部用户Ridge AUC0.6036、MAE0.7581；加入控制器不如场景+头动。这不证明控制器没有信息，仅代表当前统计量与划分的结果。',
       f'4. 仅体验时长Ridge已达到MAE={fmt(time.mae)}、rho={fmt(time.spearman)}、AUC={fmt(time.binary_auc)}，其RMSE={fmt(time.rmse)}优于场景+头动的{fmt(head.rmse)}。增加场景/头动后的误差改善有限。',
       '5. 中位数基线始终预测1分，MAE0.6895仍低于所有模型，但RMSE1.1659更差、AUC0.5、相关不可计算；不能只凭MAE宣称预测质量可靠。','',
       '## 全部主对照','',
       'MAE/RMSE越低越好，相关/AUC越高越好。AUC是固定实际评分>2的辅助排序指标，不是准确率或原三分类macro AUC。下表全部模型均按1,024次真实评分评估。','']
lines+=md_table(pd.DataFrame(table_rows),['description','model','mae','rmse','spearman','binary_auc'],['方法','模型','MAE','RMSE','Spearman','合并OOF AUC'])
lines+=['','## 简单基线','']+md_table(m[m.method=='baseline'],['model','mae','rmse','spearman','binary_auc'],['基线','MAE','RMSE','Spearman','合并OOF AUC'])
lines+=['','## 同时排除测试游戏参与者','',
        '主LOGO有6/12折共享参与者编号。下表留出游戏后，还从训练删除该测试游戏所有已知参与者的记录。LightGBM方案在主训练前登记；Ridge是在看到主比较后增加的探索性敏感性，不是新游戏上的独立确认测试。','']
double=pd.concat([pd.read_csv(out/'double_holdout_metrics.csv').query('method != "baseline"'),r],ignore_index=True)
lines+=md_table(double,['method','model','mae','rmse','spearman','binary_auc'],['方法','模型','MAE','RMSE','Spearman','合并OOF AUC'])
lines+=['',f'场景+头动Ridge在此协议中MAE={fmt(head_sens.mae)}、rho={fmt(head_sens.spearman)}、AUC={fmt(head_sens.binary_auc)}；多尺度场景Ridge AUC0.6245。用户输入没有造成明显跨参与者提升。','',
        '## 增量检查与高分错误','',
        '以下1000次session配对重采样基于固定OOF预测，是事后诊断，条件于这12个游戏，没有重训，不覆盖新游戏或全部跨session同人依赖。正值表示较参考模型MAE更小。','']
lines+=md_table(pd.DataFrame(extra),['method','reference','mae_improvement','ci_low','ci_high'],['方法(Ridge)','参考(Ridge)','MAE改善','区间下界','区间上界'])
lines+=['','加入头动相对于多尺度场景的MAE改善约0.0086，区间跨0；相对于仅时长改善约0.0222，区间也跨0。因此不能宣布头动提供了稳定的独立增益。','',
        '场景+头动Ridge按真实评分分组：','']
lines+=md_table(groups[(groups.method=='multi_scene_head')&(groups.model=='ridge')],['actual_ms_rating','events','mean_prediction','mae'],['真实评分','次数','平均预测','MAE'])
lines+=['','真实3/4/5分的平均预测均约1.8，高分幅度仍严重压缩。MAE改善不能掩盖这个问题。','',
        '## 建议','',
        '- 保留多尺度场景Ridge与场景+头动Ridge作为后续基线，先检查高分报告的时间上下文和标签覆盖；不以当前结果声称模型有效。',
        '- 下一步优先验证可用gaze的可靠解码及累计暴露特征，再在相同事件、相同留出协议中做增量对照；是否改进需要实测。',
        '- 当前联合模型需要实际用户遥测才能运行；这批运行参数尚不构成独立Unity设计设置，不能直接当作无人开发期评估引擎。',
        '- 方法是在同一批游戏上比较的，本轮最佳成绩具有选择偏差；真正推广效果需要未参与路线选择的游戏/参与者验证。','',
        '## 文件与复现','',
        '- 全部主指标：metrics.csv；各游戏折：fold_metrics.csv；预测：oof_predictions.csv；模型：models/rating_aggregation_comparison_2026-10-10/。',
        '- 详细主训练报告：[report.md](report.md)。额外Ridge敏感性：ridge_double_holdout_*；增量检查：incremental_comparison.csv。',
        '- 独立核查：preparation_check.json与output_check.json；36组主/敏感性/基线指标从保存OOF独立重算一致。',
        '- 已冻结配置：configs/rating_aggregation_comparison_2026-10-10.json；原协议补充说明：audit/rating_aggregation_comparison_2026-10-10.md。',
        '- 提取：scripts/prepare_rating_aggregation_comparison.py（--init、3个分区、--finalize）；主训练：scripts/train_rating_aggregation_comparison.py；追加Ridge：scripts/check_ridge_double_holdout_comparison.py。',
        f'- 图表：![探索性比较]({(out/"comparison.png").as_posix()})','']
(out/'final_comparison.md').write_text('\n'.join(lines),encoding='utf-8')
print(out/'final_comparison.md')
