# Unity 随机真人头动 Demo

这是一个第一人称 Windows Unity 演示：它把真实 EyeNavGS 数据训练得到的短时 6-DoF 头动 LSTM 直接导出为 Unity 资源，并在 Player 内用纯 C# 推理。默认情况下，模型直接控制体验者 Camera 的位置与旋转；场景中的胶囊人形只是切换到观察视角时用于查看运动的替身。运行时不需要 Python、PyTorch、Sentis，也不会在模型加载失败时退回随机游走。

## 运行位置

- 已构建程序：`demo/UnityRandomAgentDemoBuild/RandomHumanAgentDemo.exe`
- Unity 工程：`demo/UnityRandomAgentDemo`
- 启动场景：`Assets/Scenes/AuroraMotionCourtyard.unity`
- 模型资源：`Assets/Resources/EyeNavAgentModel.json`

请保留 `UnityRandomAgentDemoBuild` 目录中的 EXE、`RandomHumanAgentDemo_Data`、`UnityPlayer.dll` 等文件在一起，不要只复制 EXE。直接双击 EXE 即可运行。

## 模型确实经过训练的证据

- 数据源是真实 EyeNavGS 数据，训练标记为 `synthetic=false`。
- 任务是根据最近 8 帧局部线速度和角速度预测下一帧 6-DoF 运动；不是预录轨迹回放。
- 网络为 2 层 LSTM，hidden size 64，共 52,102 个学习参数；训练 30 个 epoch，验证集 normalized MSE 从 0.20952 降至最佳 0.17581。
- 数据按参与者隔离：32 名训练、7 名验证、7 名独立测试；测试集有 32,256 个窗口、84 条轨迹。
- 独立测试集线速度向量 MAE 为 0.06138 m/s，优于“保持上一帧速度”基线的 0.07135 m/s；单步姿态误差为 0.34131°，优于基线的 0.37229°。
- 源 checkpoint 为 `models/exploration_model.pth`，SHA-256 为 `eaa3fedd87a6163c8b64b30fd8f3da4601d73d4c1eb17856d9adeb39c2467c68`。
- Unity 启动时会校验导出资源哈希、checkpoint 身份和 PyTorch golden output。当前 C# 与 PyTorch 自检最大绝对误差为 `2.9802322387695315e-8`；任一校验失败都会停止运动。

详细训练记录见 `models/exploration_model_metadata.json`、`results/exploration_training_metrics.json` 和 `results/exploration_training_history.csv`。

## 每个 episode 如何随机、如何复现

无 `--seed` 参数时，程序以系统加密随机数生成一个非负 `runSeed`。这个种子驱动以下三项选择：

1. 从 504 个 held-out 8 帧历史窗口中选一个起始历史；种子库覆盖 7 名未参与训练的测试参与者、84 条参与者×场景轨迹和 12 个场景。
2. 在舞台中心附近随机选择出生位置。
3. 随机选择初始水平朝向。

之后的 240 帧动作全部由训练后的 LSTM 在线闭环生成。模型输出没有人为随机噪声，也不是随机游走。相同 `runSeed` 会得到相同的首个 seed window、出生位置、朝向和模型轨迹，例如：

```powershell
.\demo\UnityRandomAgentDemoBuild\RandomHumanAgentDemo.exe --seed=12345
```

`Replay same seed` 可立即重放当前种子。`New random person` 和每个 episode 结束后的自动切换会生成新种子。场地边缘的反射只是环境安全约束，会单独写入日志；它不是模型学会的碰撞规避。

## 界面控制

- `Pause` / `Resume`：暂停或继续模拟。
- `New random person`：立刻结束当前 episode，并随机选择新的模拟参与者起点。
- `Replay same seed`：使用当前 `runSeed` 从头复现。
- `Camera: First Person / Follow / Top / Orbit`：循环切换第一人称和三个观察/调试视角；默认是模型直接驱动的 `First Person`。
- `0.5× / 1× / 2×`：调整模拟播放速度，不改变模型固定步长。
- `Copy session log path`：把当前 episode 的日志目录复制到剪贴板。

右侧面板还会明确显示 `AI CONTROLS THIS FIRST-PERSON CAMERA`，并显示模型校验状态、checkpoint 短哈希、PyTorch↔Unity 误差、held-out 测试指标、当前种子、seed 来源、生成帧数、速度和移动距离。切换到观察视角后，Camera 与头部暂时分离以便观察，但 AI 仍继续驱动模拟参与者的头部位姿。

## 运行日志

Windows Player 默认把每个 episode 写入：

```text
%USERPROFILE%\AppData\LocalLow\OpenAI Research Prototype\EyeNavGS Random Human Motion Demo\Sessions\<UTC时间>_seed_<runSeed>\
```

每个目录包含：

- `manifest.json`：种子、held-out 窗口、模型/数据集哈希、固定步长和自检误差。
- `telemetry.csv`：每步原始模型动作、边界修正后动作、位置、姿态及边界反射标志。
- `summary.json`：结束原因、步数、推理次数、模拟时长、移动距离及 telemetry SHA-256。

`summary.json` 会在正常换 episode、重放或退出时完成；强制终止进程可能使最后一个 episode 缺少 summary 或末尾尚未刷新的 telemetry 行。

## 科学边界

- 一个 episode 只有 240 个模型步，固定步长约 0.027 秒，总计约 **6.48 秒**。短 episode 是有意设计，因为该一步预测模型在长期自回归闭环中会逐渐退化和趋向饱和。
- 这是短时真人头动行为先验，**不是长期自主导航模型**：它没有场景图像、深度、NavMesh、目标或障碍物输入，也不能据此规划、探索或避障。
- 舞台边界反射、出生点和视角属于演示环境逻辑，不是训练模型学到的能力。
- 这也**不是晕动症评分或风险预测模型**，不会输出 FMS、SSQ、风险分数或临床结论。测试 MAE 只衡量下一帧运动预测，不能解释为导航成功率或晕动预测性能。
- 因此，不应把连续多个短 episode 描述为同一人物的可信长期运动，也不应把轨迹用于临床判断。

## 从源文件重建 Windows Player

前提：

- Windows 上安装 Unity Editor `6000.2.9f1` 及 Windows Build Support；默认路径为 `C:\Program Files\Unity\Hub\Editor\6000.2.9f1\Editor\Unity.exe`。
- 仓库 `.venv` 已安装项目依赖，并存在 checkpoint、held-out 测试数据和训练指标文件。
- 构建前关闭正在打开此工程的 Unity Editor。

在仓库根目录执行：

```powershell
.\demo\build_unity_random_agent.ps1
```

脚本先运行 `scripts/export_unity_random_agent.py`，从已训练 checkpoint 重新生成模型 JSON、完整性常量和导出报告；随后用 Unity 6000.2.9f1 的 `-batchmode` 调用 `RandomAgentDemoEditor.DemoBuilder.BuildWindows`。默认输出为：

- Player：`demo/UnityRandomAgentDemoBuild/RandomHumanAgentDemo.exe`
- 导出报告：`results/unity_random_agent_export_report.json`
- Unity 构建日志：`results/unity_random_agent_build.log`
- 构建状态：`results/unity_random_agent_build_status.json`

需要换输出目录时可传绝对或相对路径：

```powershell
.\demo\build_unity_random_agent.ps1 -BuildDirectory .\artifacts\UnityRandomAgentDemoBuild
```

导出器、Unity、构建报告验证或目标 EXE 任一步失败时，脚本都会以非零状态退出。
