# Unity VR Cybersickness Preassessment Agent

> Research prototype snapshot: **2026-10-15**  
> An auditable pipeline that parses a Unity project, compiles an Agent contract,
> builds an isolated first-person test Player, explores scene content, and
> exports motion/interaction evidence for future cybersickness assessment.

本项目的最终目标是：在一个新的 Unity VR 内容发布前，让“虚拟参与者”自动理解、
探索并体验场景，收集相机运动、头动、路径、交互和渲染证据，再给开发者提供可核验的
晕动风险线索与修改建议。它是研究和设计辅助工具，**不是医疗诊断系统**。

## 先看结论

当前版本已经走通下面这条工程闭环：

```text
Unity Scene / Prefab / C# / ProjectSettings
  -> 静态证据与可选 LLM 叙述
  -> 确定性 Scene-Agent Contract
  -> 隔离 staging 构建（不修改源项目）
  -> 高层 Brain 选目标、停走、注视、交互和重规划
  -> EyeNavGS Normal HMD v4 生成第一人称低层运动
  -> 碰撞、地面和交互事务安全层
  -> 内容完成制验收、遥测、截图、体验报告和 ZIP 证据包
```

但必须区分三件事：

| 模块 | 当前状态 | 可以怎样表述 |
|---|---|---|
| Normal HMD v4 | 已训练并接入 Unity | EyeNavGS 派生的头显/根运动先验；平滑部署验收通过，但完整真人分布验收仍未通过 |
| Unity Decision Brain v0 | 已在 `Model research` 模式闭环运行 | 规则教师蒸馏的 `smoke_only` 模型，不是已证明像真人的决策大脑 |
| VR.net 分类器 | 已完成离线探索实验，尚未接入 Demo | AUC 0.6819 的探索性关联模型；当前 Agent 报告中的 exposure proxy 不是晕动概率 |

所以，当前系统能自动采集并排序“哪里值得检查”的暴露证据；它还不能自动给出经过
独立验证、校准的个人晕动概率，也不能从相关性直接得出设计修改的因果效果。

## 文档入口

- [完整项目与版本报告](docs/PROJECT_REPORT_2026-10-15.md)：给研究者、开发者和评审阅读。
- [无记忆 AI 交接说明](docs/AI_HANDOFF.md)：上下文、文件地图、事实口径、下一步任务和禁止误称。
- [数据集与许可](docs/DATASETS.md)：EyeNavGS、VR.net、Habitat-Web 及未采用候选。
- [复现与验证](docs/REPRODUCIBILITY.md)：环境、测试、训练、构建和验收命令。
- [Scene Agent Terminal 使用说明](demo/SCENE_AGENT_TERMINAL.md)：界面和四步操作。
- [机器可读发布清单](release/research_snapshot_2026-10-15.json)：模型、指标、哈希和验收摘要。
- [第三方归因](THIRD_PARTY_NOTICES.md) 与 [项目许可状态](LICENSE_STATUS.md)。
- [继续开发约定](CONTRIBUTING.md)：数据、证据、测试和声明要求。

## 最快开始

### 1. 环境

- Windows 10/11；
- Unity Editor `6000.2.9f1`（当前构建脚本按该版本验证）；
- Python 3.10+；Normal HMD 训练环境使用 Python 3.12；
- PowerShell 7 推荐，Windows PowerShell 5.1 也受支持；
- 可选 NVIDIA CUDA，用于重新训练；查看/构建现有 Unity 导出不要求重新训练。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_release_package.py
python -m unittest `
  scripts.test_unity_parser `
  scripts.test_compile_scene_agent_contract `
  scripts.test_generic_scene_agent_collision `
  scripts.test_generic_scene_agent_interaction_lock `
  scripts.test_scene_agent_terminal_acceptance -v
```

### 2. 直接打开 Unity 源项目

用 Unity Hub 打开：

```text
demo/UnityNormalHmdAgentDemo
```

当前运行所需的两个小型 JSON 导出已包含在仓库中：

- `demo/UnityNormalHmdAgentDemo/Assets/Resources/NormalHmdAgentModel.json`
- `models/unity_decision_brain_v0_unity.json`

第一个文件包含模型参数以及 EyeNavGS 派生的短 seed windows；这不是完整原始数据集，
但也不能称为“完全不含数据片段”。来源和 Apache-2.0 归因见第三方说明。

### 3. 从源码生成终端

```powershell
.\demo\build_scene_agent_terminal.ps1
```

生成的 `demo/SceneAgentTerminalBuild/SceneAgentTerminal.exe` 是本机构建产物，按设计不进 Git。
打开终端后按以下顺序操作：

1. 选择 Unity 项目和 Scene；
2. `AI 解析 + 生成契约`；
3. `隔离构建 Agent`；
4. `启动 Agent 体验`，完成后导出证据。

未知项目并不保证零适配成功。项目必须可编译；复杂 NavMesh、Addressables、运行时生成内容、
XR/网络插件和自定义玩法可能需要适配器。改变游戏状态的方法只有进入中央 reviewed binding
白名单并具有可验证后置条件后才允许调用。

## 仓库内容

```text
configs/      冻结配置、Prompt、schema 和 reviewed-binding 规则
data/         原始数据位置（Git 排除）；只提交可公开的数据可用性审计和合成 parser fixture
demo/         Terminal、通用 Agent bridge、Unity 源项目和构建脚本
docs/         人类报告、AI 交接、数据与复现说明
knowledge/    已核验的模型/数据语义说明
models/       仅提交审查后的 metadata / Unity JSON；二进制 checkpoint 排除
release/      脱敏、机器可读的发布验收摘要
scripts/      解析、预处理、训练、导出、验证和测试代码
```

`data/`、`features/`、大部分 `results/`、`.pth/.pkl`、Unity `Library/`、隔离 staging、
Player 和 EXE 均不会上传。这样既避免发布参与者/session 明细和未知许可数据，也避免把
数 GB 构建缓存写进 Git 历史。需要重现实验时按 [数据集说明](docs/DATASETS.md) 放置数据。

## 权威快照结果

- 最终 Rule 与 Model Player 均在 `317.304 s` 完成 `10/10` 目标，状态
  `passed / content_complete`；Model 模式 40 次推理、0 fallback。
- 碰撞 fixture：4/4 完成，471 次安全绕行调整、0 overlap；强制重叠负例按预期失败。
- 异步交互 fixture：next-frame 与 coroutine 后置条件均跨帧验证；BodyRoot 锁违规为 0，
  HeadLocal 在交互期间继续运动；伪造既有状态的失败方法不会被误判成功。
- 发布前完整 Python 回归：88/88 passed；Unity `6000.2.9f1` 最终构建无错误。

旧的 `59.994 s / 8 of 10 / 33 inference` handoff 结果已被上述最终验收取代。
详细证据和原始本地路径映射见完整报告；GitHub 只保存脱敏聚合摘要。

## 贡献与许可

仓库当前**没有项目级开源许可证**。公开可读不等于获得复制、修改或再分发授权；继续使用
前请与项目所有者确认。第三方 EyeNavGS 派生资产仍遵循 Apache-2.0 及其归因要求。
详见 [LICENSE_STATUS.md](LICENSE_STATUS.md) 和 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
