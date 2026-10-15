# 数据集、放置方式与许可边界

本仓库不分发完整原始参与者数据，也不发布 VR.net 的逐参与者/逐事件特征表。两个 Unity
runtime JSON 会分发经处理的 EyeNavGS 伪名轨迹 seed windows；具体边界见下文。下面记录
2026-10-15 快照实际使用、尝试或评估过的数据，以及新环境应如何恢复。下载和使用前仍需
阅读每个上游项目的最新许可和访问条款。

## 1. EyeNavGS — 当前 Normal HMD 的训练来源

用途：学习普通参与者的 6-DoF HMD 时序，训练 Normal HMD v4 低层运动先验。

上游：

- 项目页：<https://symmru.github.io/EyeNavGS/>
- Rutgers：<https://github.com/symmru/EyeNavGS_Rutgers_Dataset>
- NTHU：<https://github.com/sawalee0811/EyeNavGS_NTHU_Dataset>
- 论文：<https://arxiv.org/abs/2506.02380>

本地审计固定版本：

| Site | Commit | Participants | Traces |
|---|---|---:|---:|
| Rutgers | `7f264fda0223cbc74e191f115ef9fd3b580d8cf4` | 22 | 264 |
| NTHU | `608790e7d932534f18a2a124887605d1f6c85384` | 24 | 288 |

合计 46 人、12 场景、552 条 canonical trace、3,862,257 行、1,931,091 个有效 stereo pair；
本地检查目录共 570 个文件、537,607,804 bytes。许可：Apache License 2.0。

建议目录：

```text
data/raw/eyenavgS/
├─ rutgers/EyeNavGS_Rutgers_Dataset/
└─ nthu/EyeNavGS_NTHU_Dataset/
```

下载示例：

```powershell
git clone https://github.com/symmru/EyeNavGS_Rutgers_Dataset.git `
  data/raw/eyenavgS/rutgers/EyeNavGS_Rutgers_Dataset
git clone https://github.com/sawalee0811/EyeNavGS_NTHU_Dataset.git `
  data/raw/eyenavgS/nthu/EyeNavGS_NTHU_Dataset
python scripts/check_eyenavgS.py --deep
```

训练/导出入口见 `docs/REPRODUCIBILITY.md`。

### 仓库中仍有哪一小部分派生数据

`demo/UnityNormalHmdAgentDemo/Assets/Resources/NormalHmdAgentModel.json` 含模型权重、归一化参数、
golden vector，以及 414 个处理后的 validation seed windows（来自 104 条轨迹、7 个伪名参与者，
每个 64×9）和公开数据集伪 ID。
这些 seed 用于不同随机种子的初始状态。它不是完整原始 CSV，但属于 EyeNavGS 派生片段，随
Apache-2.0 归因发布。详见 `THIRD_PARTY_NOTICES.md`。

历史对照项目 `demo/UnityRandomAgentDemo/Assets/Resources/EyeNavAgentModel.json` 还包含 504 个
处理后的初始化 windows（84 条轨迹、7 个伪名参与者）。它不接入当前 Generic Agent，但同样
属于随 Apache-2.0 归因发布的派生 runtime asset。

## 2. VR.net — 离线晕动关联模型

用途：探索运行时相机/场景/真人头动统计与真实 `MS_rating` 的关联；未接入当前 Unity runtime。

来源信息：

- 论文：<https://doi.org/10.1109/TVCG.2024.3372044>
- 预印本：<https://arxiv.org/abs/2306.03381>
- VRHook 项目页：<https://ahlab.org/projects/vrhook/>
- 作者页面链接的公开 Drive 索引：
  <https://drive.google.com/drive/folders/1LLo4OE83Zeso8MkEXSvW9VrdjiqqlFRR>

本快照模型只使用本地子集：12 games、86 sessions、69 个 participant ID、1,024 个评分事件；
`MS_rating > 2` 为 199 条，`<= 2` 为 825 条。工作目录约 79.97 GB，包含下载、解压子集和中间
缓存，**不能把该体积描述成官方唯一数据包大小**。

推荐目录：

```text
data/raw/vrnet/
```

注意事项：

- 当前公开索引为 82 games / 798 sessions，论文摘要称 100 games / 165 小时 / 500 evaluators，
  预印本又描述更早版本；必须明确版本，不要混用规模。
- 现代公开 Drive 包未找到足够清晰的再分发许可证。本仓库不上传原始包、逐事件表、OOF 预测
  或 pickle。
- 旧 Epic roller-coaster 包是另一版本；5 个录制、144 次评分并不是论文中 471 事件的复现。
- 一个 0–1 模型输出只是 `MS_rating > 2` 的未校准分类倾向，不是个体概率或医疗诊断。

## 3. Habitat-Web ObjectNav THDA 70k — 离线真人导航先验

用途：训练六动作 pose/goal/history MLP；当前没有接入 Unity Agent。

上游：

- 官方训练代码：<https://github.com/Ram81/habitat-imitation-baselines>
- Human demonstrations：<https://huggingface.co/datasets/axel81/habitat-web>
- 论文：<https://arxiv.org/abs/2204.03514>

数据集许可：Creative Commons Attribution-NonCommercial 4.0。它限制商业使用；代码许可证与
demonstration 数据许可证不能混为一谈。

本地实验读取 `objectnav_mp3d_thda_70k`：56 Matterport scenes、70,176 episodes，60,085 个
usable goals；按 scene 39/8/9 切分，采样 600k/120k/120k 动作记录。原始和 prepared 数据都不
上传，runtime prior 权重也不随当前快照发布。

配置默认使用仓库内被 Git 排除的 `data/external/...` 与 `data/derived/...` 目录。若数据放在
其他位置，可修改 `configs/habitat_decision_prior_v1.json` 的 `data.data_root` 和 `prepared_dir`。

## 4. 已调查但没有形成当前模型的数据

| 数据 | 当前状态 | 不可声称的内容 |
|---|---|---|
| VRWalking / Mazed and Confused | 本地仅发现 32 行 CSV；视频/许可未确认；没有完成模型训练 | 不能说已用其训练晕动 Agent |
| CYRE | 没有找到公开 raw 包和明确数据许可 | 不能说已下载或使用 |
| MazeSick / RelaxVR | 需要申请；未提交申请/未下载 | 不能说本地可用 |
| Simulation 2021 | 需要申请；未提交申请/未下载 | 不能说本地可用 |

## 5. Git/隐私规则

以下内容必须保留在 Git 外：

```text
data/**                 # 除 data/cyre_status.md 和 data/test_scenes 合成 fixture
features/**             # 可能含 participant/session IDs
results/**              # 本地逐帧/逐事件/日志；仅 .gitkeep 和合成 unity_semantic_fixture
audit/**                # 可能含 session 映射
models/**/*.pth|pkl     # checkpoint 与未知再分发许可模型
tmp/**
Unity Library/Build/staging/Player/EXE
```

不要用 `git add -f` 绕过这些边界。若需要发布新的聚合结果，应去除 participant/session ID、
绝对路径、下载 token 和原始行，把最小摘要写到新的 `release/<date>.json`，并进行人工复核。
