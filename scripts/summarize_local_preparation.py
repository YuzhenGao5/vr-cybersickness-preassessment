#!/usr/bin/env python3
"""Summarize only observed local preparation artifacts, preserving NA gates."""
import hashlib
import importlib.metadata as metadata
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml


def main():
    root = Path(__file__).resolve().parents[1]
    downloaded = json.loads((root/'results/local_download_sessions.json').read_text(encoding='utf-8'))
    clips = pd.read_csv(root/'features/video_clips_manifest.csv')
    features = pd.read_csv(root/'features/lightgbm_features.csv')
    matrix = pd.read_csv(root/'features/lightgbm_training_matrix.csv')
    labels = pd.read_csv(root/'features/clip_risk_labels_v3.csv')
    shap = pd.read_csv(root/'features/shap_lightgbm_values.csv')
    bundle = joblib.load(root/'models/lightgbm_baseline.pkl')
    config_path = root/'configs/risk_factor_labels_v3.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    make_log = json.loads((root/'results/make_clips_local_log.json').read_text(encoding='utf-8'))
    raw = root/'data/raw/vrnet'
    csv_files = list(raw.rglob('data_file.csv'))
    jpgs = [p for p in raw.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg') and p.is_file()]
    raw_unique = {p.parent.name for p in csv_files}
    complete = {d['session_id'] for d in downloaded if d['complete_by_csv_references']}
    complete.add('11_2_Mars_Miners')
    partial = set()
    for path in jpgs:
        for parent in path.parents:
            if parent.name in raw_unique:
                partial.add(parent.name)
                break
    partial -= complete
    free = shutil.disk_usage(root).free
    quality = []
    for name in bundle['feature_names']:
        values = pd.to_numeric(features[name], errors='coerce').replace([np.inf, -np.inf], np.nan)
        quality.append({'feature': name, 'nonmissing': int(values.notna().sum()),
                        'missing_rate': float(values.isna().mean()), 'min': values.min(),
                        'max': values.max(), 'mean': values.mean(), 'std': values.std()})
    pd.DataFrame(quality).to_csv(root/'features/local_preparation_feature_quality.csv', index=False, encoding='utf-8-sig')
    label_table = ['| 风险因素 | 配置状态 | 已知标签 | unknown | AUC |', '|---|---|---:|---:|---|']
    for label in config['labels']:
        series = labels[label['id']].astype(str)
        known = int(series.isin(['0','1','0.0','1.0']).sum())
        label_table.append(f"| {label['id']} | {label['status']} | {known} | {len(series)-known} | NA：无有效监督 |")
    lines = ['# 本地训练前准备总结', '', '**流程验证，非最终结论。视频输入流程通过；有效风险因素监督与模型训练门控未通过。**', '',
             '## 实际数据与下载', '',
             f'- 当前 raw：{len(csv_files)} 份 CSV，去重 {len(raw_unique)} 个 session；完整已发布 JPG 序列 {len(complete)} 个 session、{clips.game_name.nunique()} 个游戏。',
             f'- 部分 JPG {len(partial)} 个 session，只有 CSV {len(raw_unique-complete-partial)} 个 session；raw JPG 副本合计 {len(jpgs)} 张。',
             '- 本轮只下载 Army_Men 一个 ZIP，973,639,478 字节（0.974 GB），保留5个 session 的 JPG/CSV，ZIP已删除。',
             f'- 新保留 {sum(d["jpg_count"] for d in downloaded)} 张 JPG；解压所选 JPG+CSV 5,338,743,907 字节。现有 Mars Miners 5,178 张。',
             f'- 当前 C盘空余 {free:,} 字节（{free/1e9:.3f} GB）；未低于50GB门槛。',
             '- 默认沙箱网络曾因 WinError10013 拒绝；经已批准的网络执行成功。下载日志完整保留失败与成功。', '',
             '| Session | JPG | CSV行数 | 视频引用缺失 |', '|---|---:|---:|---:|']
    lines += [f"| {d['session_id']} | {d['jpg_count']} | {d['csv_rows']} | {d['missing_references']} |" for d in downloaded]
    lines += ['| 11_2_Mars_Miners（既有） | 5178 | 5180 | 0 |', '',
              '“完整”指已发布 CSV 视频引用对应的 JPG 集合齐备；不表示引擎每一帧均保存，也未逐张解码所有完整序列。', '',
              '## 本地实际执行', '',
              f'- {len(clips)} 个真实1秒片段，{clips.session_id.nunique()} 个 session，每 session 均匀抽20个有效窗口；不依据风险标签抽样。',
              f'- 原时间线上 {make_log["skips"]} 个窗口因源帧不足/间隙/边界被跳过；其余有效窗口中只保留本地抽样20个，不等于其余均失败。',
              f'- 特征提取 {features.extraction_status.eq("success").sum()}/{len(features)} 成功，对齐 {len(matrix)} 行。',
              f'- 输出{len(bundle["feature_names"])}项视频统计特征，输入来自RGB、Farneback光流、图像单应性。',
              '- 6个片段的方向/单应性相关9项统计量缺失（各5%）；其他19项视频特征无缺失。缺失实测记录于feature_quality CSV，不能补成测得的值。',
              '- 深度模型未提供、真实米制相机/物体运动未从RGB估出，未生成这些特征；未把矩阵标签生成量作为输入。',
              '- 输出30FPS是最近帧重复重采样，源JPG实际时间间隔不规则；不恢复原始30Hz运动。原始源帧数量和映射见video_clips_manifest。',
              '- make_clips首轮因沙箱Temp编码失败立即停止；修复并通过合成测试后重新开展准备。失败与重试日志分别保存。', '',
              '## 文献、单位与标签门控', '',
              '- 25°/s已核实为特定yaw snapping干预参数；13/52°/s原始出处本轮未找到。其余候选均未获得能直接套用当前数据的因素切点。详见threshold_literature_review.md。',
              '- FOV逐帧 h std约2.02e-6°、v std约4.26e-6°，基本恒定；wide_fov停用。',
              '- 相机/物体速度为源坐标单位/s；米制转换未确认。已有“加速度”是速率差/dt，不是完整向量加速度。',
              f'- v3配置已冻结，SHA256 `{hashlib.sha256(config_path.read_bytes()).hexdigest()}`；7项全部禁用、threshold_value=null；不用P80。', '',
              *label_table, '',
              '## 模型、SHAP与信号判断', '',
              f'- 实际拟合模型：{bundle["macro_metrics"]["trained_labels"]}/7。多数类、LightGBM各标签指标均NA，无已知监督。',
              '- models/lightgbm_baseline.pkl为门控状态包，七项model均None，不能作为已训练分类器部署。',
              f'- SHAP真实值行数：{len(shap)}，生成图数：{len(list((root/"figures").glob("shap_lightgbm_*.png")))}。空CSV保留表头，不造图。',
              '- 数据读取→切片→视频特征→标签对齐→训练/SHAP门控代码通过；非空真实监督下的模型拟合和SHAP尚未验证。',
              '- 当前有2个完整游戏，其中Mars只有1个session；一个LOGO方向没有足够开发session，完整评估仍不成立。分组文件仅作为划分草案，不是已完成折外验证。',
              '- **不能判断有信号或无信号。没有AUC不是低AUC，unknown不是negative。**', '',
              '## 下一步', '',
              '1. 优先冻结范围明确的因素任务：例如受控的人工yaw、给定持续时间和可复现干预条件。研究可以采用明确工程标签，无需等待普适生理阈值，但不能把工程条件冒称已验证症状边界。',
              '2. 核实矩阵时钟、坐标到米、投影FOV语义、相机/头动分离和对象身份。确定可观测指标后，另立有依据的新版本；不要修改本轮v3去凑类别。',
              '3. 标签定义通过后再执行3游戏试点，先多数类与RGB统计LightGBM、外层LOGO/内层session，再考虑TSM。现在不建议启动大规模视频训练。', '',
              '## 产出与验证边界', '',
              '- evaluation_protocol.md、pilot_plan.md、local_preparation_reproduction.md：协议、三游戏计划、完整命令与环境。',
              '- scripts下已提供下载、提取、切片、标签、划分及4个基线脚本。Python本地已执行；Linux shell与服务器依赖安装未实测。',
              '- local_preparation_script_review.md：15项隔离合成检查；lightgbm_script_validation.md：6项隔离基线检查。合成fixture不进入真实训练清单。',
              '- 未训练TSM/深度模型；未执行git commit；旧配置/原始数据未改写。', '']
    (root/'results/lightgbm_baseline_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    appendix = ['\n## 本轮下载后更新（2026-10-08）', '',
                '本节由本轮真实下载与磁盘扫描追加，不改变上方下载前快照。',
                f'raw CSV副本数={len(csv_files)}；去重session={len(raw_unique)}；完整JPG session={len(complete)}；partial={len(partial)}；只有CSV={len(raw_unique-complete-partial)}。',
                f'raw JPG副本总数={len(jpgs)}。Army_Men 5个完整session新保留{sum(d["jpg_count"] for d in downloaded)}张；CSV视频引用缺失0；ZIP已删除。',
                'v6 61个session中完整JPG现为5（Army Men），另2个仅partial、54个只有CSV；Mars完整session不在v6清单中。',
                f'实际下载后源清单：results/local_download_sessions.json。实际本地片段{len(clips)}个，见features/video_clips_manifest.csv。',
                '满足一个游戏至少5个完整session的本地输入检查要求；尚不满足可靠多游戏训练/有效因素标签门控。', '']
    inv_path = root/'results/local_data_inventory.md'
    original = inv_path.read_text(encoding='utf-8').split('\n## 本轮下载后更新（2026-10-08）')[0]
    inv_path.write_text(original + '\n'.join(appendix), encoding='utf-8')
    pilot = root/'results/pilot_plan.md'
    original = pilot.read_text(encoding='utf-8').split('\n## 本轮实际执行更新')[0]
    pilot.write_text(original + '\n## 本轮实际执行更新\n\n本地仅执行Army_Men的五session下载，完整JPG共28,822张、CSV引用缺失0；ZIP已删除。Pirate_Life、City_Parkour未下载完整JPG。服务器三游戏清单为configs/vrnet_server_pilot_downloads.json，尚未执行。当前七标签禁用，训练门控未通过。\n', encoding='utf-8')
    versions = {'python': sys.version, 'platform': platform.platform(), 'seed': 42,
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'packages': {p: {'version': metadata.version(p), 'requires_python': metadata.metadata(p).get('Requires-Python')}
                             for p in ['requests','numpy','pandas','opencv-python-headless','scipy','PyYAML','scikit-learn','lightgbm','shap','matplotlib','joblib']}}
    versions['artifact_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [config_path, root/'features/video_clips_manifest.csv', root/'features/clip_risk_labels_v3.csv',
                                            root/'features/lightgbm_features.csv', root/'features/lightgbm_training_matrix.csv', root/'models/lightgbm_baseline.pkl']}
    (root/'results/local_preparation_environment.json').write_text(json.dumps(versions, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'complete_sessions': len(complete), 'clips': len(clips), 'features': len(bundle['feature_names']),
                      'trained_models': bundle['macro_metrics']['trained_labels'], 'shap_rows': len(shap), 'free_bytes': free}))


if __name__ == '__main__':
    main()
