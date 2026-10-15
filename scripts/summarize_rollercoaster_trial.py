"""Write the source/coverage/conclusion report from measured trial artifacts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def main():
    manifests = read('results/rollercoaster_download_manifest.json')
    metadata = read('results/rollercoaster_7z_inspection.json')
    audit = read('results/rollercoaster_depth_rgb_session_audit.json')
    metrics = read('results/rollercoaster_depth_rgb_metrics.json')
    scene = [m for m in manifests if m['name'] != 'SelfReportData.zip' and m['status'] == 'success']
    total_bytes = sum(m['archive_bytes'] for m in manifests if m['status'] == 'success') + metadata['archive_bytes']
    lines = ['# 过山车数据定位与小模型试验', '', '检查日期：2026-10-10（Pacific/Auckland）；下载日志保留UTC时间。', '',
             '**探索性流程验证，非最终结论。没有执行 git commit。**', '',
             '## 找到的是哪个版本', '',
             '- 此前82游戏/798sessions公共版本的索引不含明确过山车游戏名，常见文件是data_file.csv＋RGB JPG。此前CSV/RGB实验不能算原论文过山车实验复现。',
             '- 从[第一作者官方旧站源码](https://github.com/elliott-wen/vrnet/blob/master/index.md)找到[原始数据卡](https://docs.google.com/document/d/1CgAWHqVTw4IPT6B586dZRwnLgAdpZLAuFYt1iBgrWtQ/edit?usp=sharing)和[原始访问表](https://docs.google.com/forms/d/e/1FAIpQLSf8PsMjxFcNhkHMI3vZ3_kC0rIzSVQu_sI7Mt9wGwzTJt59Ew/viewform)。公开表单文本给出[原始VRNetDataCollection目录](https://drive.google.com/drive/folders/1N8FYZCvqS5kJHUN1wh_bRDEdFV76Jjxf)。没有填写/提交表单或签署额外协议。',
             '- 数据卡描述10游戏/36参与者/62sessions/12.1小时的旧版，并把Epic Roller Coaster列为15sessions；不能把该旧版说明套到新82游戏包。',
             '- [Epic原始目录](https://drive.google.com/drive/folders/1Nwd8JhgIxBgZwJLSuE9qIg5yb2D8rmbC)实际公开列出5个数据ZIP和一个元数据7z。它们已逐个成功下载，保留真实文件，并在完整提取后删除压缩包。', '',
             '## 下载和文件证据', '', '| 文件 | 压缩字节数 | 解压字节数 | 状态 |', '|---|---:|---:|---|']
    for m in manifests:
        if m['status'] == 'success':
            lines.append(f'| [{m["name"]}]({m["source_url"]}) | {m["archive_bytes"]} | {m["uncompressed_bytes"]} | 成功，CRC完整读取，压缩包已删 |')
    lines.append(f'| [Epic RollerCoaster.7z]({metadata["source"]}) | {metadata["archive_bytes"]} | {metadata["expected_uncompressed_bytes"]} | 成功，25个CSV，压缩包已删 |')
    lines += ['', f'- 实际下载合计 {total_bytes} B（约{total_bytes / 1e9:.3f} GB），包含评分包、5个图像ZIP和元数据7z；没有下载129GB全库。最大单文件{max(m["archive_bytes"] for m in scene)} B，未超过5GB；一次一个包。',
              f'- 本地5次录制：{sum(a["rgb_files"] for a in audit)}张RGB、{sum(a["depth_files"] for a in audit)}张深度JPEG、{sum(a["motion_files"] for a in audit)}个motion ZFP。图像记录时间合计{sum(a["duration_sec"] for a in audit):.3f}秒（约{sum(a["duration_sec"] for a in audit) / 60:.2f}分钟）。',
              '- 元数据7z包含每次录制的camera/control/light/object/pose.csv，路径在data/raw/vrnet/rollercoaster_raw/derived_csv/；不是缺失的10个session，也没有新增评分或纹理解码配置。',
              '- gamedata.log与voice.csv保留在各ZIP解压目录。ZIP SHA256、全部成员CRC/大小、7z CSV SHA256见rollercoaster_download_manifest.json、rollercoaster_zip_members.csv.gz、rollercoaster_7z_inspection.json。',
              '- 网络确实允许访问这些公开资源；早先旧域名DNS、某些工具/大范围查询失败记录保留在来源审计，没有把它们视为所有来源不可访问。', '',
              '## 原始评分与实际支持', '',
              '- SelfReportData包109个物理文件中，55是macOS元数据，2是notebook checkpoints；仅52个实际session CSV，不能把108个.csv扩展名文件称为108个session。',
              '- Epic对应P17–P21五个评分CSV，共144条原始记录：manual23、auto121。原始分值1–5；无分档、无补分。',
              '- manual均值2.695652、中位数3、总体标准差1.365245、最大5，四项舍入后与数据卡完全吻合；这是manual作为主对照的推断依据，不证明它是金标准。auto单列敏感性。',
              '- 两组manual内部近邻重复仍保留，全部source合并时还有近邻冲突；原始文件未被改写。全部事件按参与者整体留出，记录数量不等于独立人数或独立暴露实验数。',
              '- 评分是Unix微秒，native图像日志是Unix毫秒；整数统一为微秒，严格取评分前3秒[t-3,t)，没有未来帧。不能用30fps或FrameID猜时钟。',
              f'- 144条记录中{metrics["status_counts"].get("ok", 0)}个窗口通过预先固定质量门槛，{metrics["status_counts"].get("skipped", 0)}个被跳过；manual有效{metrics["cohorts"]["manual"]["n"]}，auto有效{metrics["cohorts"]["auto"]["n"]}。详细跳过原因见提取报告。', '',
              '## 模型与结果', '',
              '- 输入57个预先定义统计量：33个RGB图像/推断光流统计量＋24个原生深度JPEG编码统计量。深度非米制，原生motion因producer配置未知未解码，也未作为输入。',
              '- Leave-One-Participant-Out，5折；同一人全部记录同折，填补/标准化/预测范围仅使用训练折。mean/median、elapsed Ridge、RGB＋depth Ridge/LightGBM、RGB-only和depth-only LightGBM均用相同source事件。',
              '- min_child_samples=5在第一次fit前固定：少量manual无法用旧参数20形成两叶。没有根据验证指标调参数；因此和City Parkour实验的差异不是只更换数据的受控对照。',
              '- 保存全部折外预测和原始/训练范围裁剪后预测。下表为裁剪后pooled OOF；相关系数仅作描述，没有把多次评分当独立事件计算p值。', '',
              '| source | 合格记录/参与者 | 方法 | MAE ↓ | RMSE ↓ | Pearson | Spearman |',
              '|---|---|---|---:|---:|---:|---:|']
    names = {'train_mean': '训练均值', 'train_median': '训练中位数', 'elapsed_ridge': '仅经过时长Ridge',
             'native_lightgbm': 'RGB＋深度LightGBM', 'native_ridge': 'RGB＋深度Ridge'}
    for source, c in metrics['cohorts'].items():
        for method in names:
            m = c['methods'][method]['pooled_oof']
            corr = ['NA' if m[k] is None else f'{m[k]:.4f}' for k in ('pearson', 'spearman')]
            lines.append(f'| {source} | {c["n"]}/{c["n_groups"]} | {names[method]} | {m["mae"]:.4f} | {m["rmse"]:.4f} | {corr[0]} | {corr[1]} |')
    lines += ['', '## 结论与下一步', '',
              '1. 数据确实存在，并已成功取得公开目录的5个过山车录制和25个派生元数据CSV。原生深度可读取，评分/帧时间可以按实际记录对齐；原生运动流解码仍待确认。',
              '2. 当前57个统计量的小模型未显示优于猜训练均值/中位数的评分预测收益。manual及auto两个source均未显示收益，且拟合与留出误差差距较大；仅5人，结果不能推广到VR.net全部数据或模型架构。',
              '3. [论文§4.3](https://arxiv.org/html/2306.03381v1)用471次评分、评分前3秒的原生flow/depth序列和双分支3D CNN。本次尚未取得缺失P27–P36、471事件清单或5轨道映射，没有完成该实验复现。论文0.70 accuracy未给定义，不能拿它和本次MAE/AUC混比，也不能凭当前结果判断论文不可信。',
              '4. 优先向作者确认471事件版本、auto/manual/纠错规则、motion ZFP及depth编码、track/participant split，再决定视频模型训练。联系草稿已更新但未发送。',
              '5. 若后续只做开发时风险因素识别，可另立实验从相机/物体元数据生成有证据的因素标签，并以画面为输入；它是另一个任务，不能把这次评分回归结论或规则标签直接改称人体风险验证。',
              '6. 原PROTOCOL的三分类AUC门控未评估，没有改标签阈值，没有Git提交，没有宣称Unity开发时风险或设计建议已经验证。', '',
              '## 产出索引', '',
              '- 结果：rollercoaster_depth_rgb_training_report.md / metrics.json / fold_metrics.csv。',
              '- 特征：features/rollercoaster_depth_rgb_event_features.csv、feature_statistics.csv、oof_predictions.csv。',
              '- 模型：models/rollercoaster_depth_rgb.pkl（全量演示拟合，不是已经验证的风险引擎）。',
              '- 下载和来源：rollercoaster_download_log.txt、download_manifest.json、rollercoaster_sources/。',
              '- 原始评分/格式审核：rollercoaster_selfreport_check.md、rollercoaster_native_format_check.md；原始表和日志全部保留。',
              '- 预先冻结的提取/训练定义在audit/rollercoaster_depth_rgb_*_definition.json。',
              '- 许可：旧作者站声明CC-BY-NC-SA，公开访问表含非商业研究/教学条款；以官方来源及版本为准。DataCard模板页脚的CC-BY-SA不是数据集许可。未提交/代签额外协议。', '']
    (ROOT / 'results/rollercoaster_data_check.md').write_text('\n'.join(lines), encoding='utf-8')
    text = '\n'.join(lines)
    print('WROTE results/rollercoaster_data_check.md', len(text), 'characters')


if __name__ == '__main__':
    main()
