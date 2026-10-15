# Scene-Aware Normal HMD Agent Demo

## 交付物

- Windows Demo：`demo/UnitySceneAwareAgentDemoBuild/SceneAwareAgentDemo.exe`
- 构建脚本：`demo/build_unity_scene_aware_agent.ps1`
- 汇总验收：`results/scene_aware_agent_acceptance.json`
- 构建状态：`results/scene_aware_agent_build_status.json`

不要只复制 `.exe`。如需移动 Demo，应完整复制 `UnitySceneAwareAgentDemoBuild` 文件夹。

## 这个版本完成了什么

```text
设计时场景语义清单
  -> 节点/边/目标/可交互 affordance
  -> Dijkstra 高层路径规划
  -> Navigate / LookAt / Activate / Observe / Verify 技能
  -> 规则引导的 BodyRoot + EyeNavGS 训练的 Normal HMD HeadLocal
  -> Unity 引擎状态验证
  -> action ledger + 37 Hz scene telemetry + 未校准暴露代理
```

左右两侧 UI 会明确区分来源：

- **训练得到**：Normal HMD v4 的头显局部位置/旋转时序，以及作为速度参考的 locomotion prior。
- **规则实现**：语义图、目标选择、Dijkstra、技能状态机、BodyRoot 路线引导和舒适执行滤波。
- **不是人体结论**：界面中的 exposure proxy 是工程调试信号，不是 VR.net 概率、个人晕动概率或医学诊断。

现有 v4 没有“目标位置/地图/交互状态”输入，所以这个版本没有把规则导航冒充成模型学会的导航。规则层只覆盖实际执行的三个 BodyRoot 通道，以及过山车场景中显式标注的载具平台运动；六个 HeadLocal 通道和 LSTM 循环状态仍由已验证训练模型连续生成。日志同时保留 learned Root 和 executed Root，便于审计差异。

构建固定使用已通过 16 条验证轨迹、每条 60.021 秒校准的 SmoothWalk 推理配置：Root innovation `std=0.05`、`rho=0.99`、roll clamp `0.10 rad`、noise scale `1.0`。这些是部署平滑参数，不是重新训练得到的新 checkpoint；构建脚本即使使用 `-SkipExport` 也会核对配置与校准文件哈希，避免误打包为较晃的默认 profile。

## 四个场景

1. **Calm Door Baseline**：稳定地平线参照；走到按钮、注视、按下、等待门完全打开、穿过并观察。
2. **Multi-turn Corridor**：沿显式图经过多个 90° 转弯，到出口开关、验证门打开并观察出口。
3. **Dynamic Visual Motion Lab**：启动旋转并上下运动的视觉装置，然后移动到观察点采集场景变化后的相机/头部时序。
4. **Semantic Roller Coaster**：导航到车站、上车并确认安全锁、启动载具、沿闭合起伏轨道乘坐、回站并验证完成状态。载具轨迹是显式规则，乘坐期间的 HeadLocal 仍来自训练模型。

这些场景是合成的系统测试，不是“已证明会/不会引起真人晕动”的实验场景。

## 使用方法

双击 `SceneAwareAgentDemo.exe`。默认第一人称视角，AI 控制的就是 `ParticipantCamera`。

- `1 / 2 / 3 / 4` 或右侧按钮：切换四个场景。
- `R`：新 seed 重新运行当前场景；头显轨迹会变化。
- `C`：第一人称 / 外部观察相机切换。
- `Space`：暂停 / 继续。
- 左侧 `New random person`：重新采样 Normal HMD seed。
- 左侧 `Copy log folder`：复制当前日志根目录。

为避免转弯和技能切换时突然改变速度，规则执行器限制平面加速度为 `1.8 m/s²`，并限制转向角加速度。它是明确标注的舒适执行滤波，不属于 EyeNavGS 学习结果。

## 每次运行的证据

每次运行会生成：

- `scene_manifest.json`：场景 ID、语义节点、边、目标、affordance 和成功状态。
- `action_ledger.jsonl`：每次计划、技能切换、交互命令及 Unity 状态验证证据。
- `scene_telemetry.csv`：37 Hz 的 BodyRoot、HeadLocal、Camera、learned/executed Root、相机速度/角速度/加速度、动态场景强度和 exposure proxy。
- `scene_acceptance.json`：时长、覆盖、目标和交互完成情况。

门的 `press` 只先记为 `accepted`；只有 Unity 状态到达 `door_open_fraction=1.000` 才记为成功。动态装置也必须返回 `dynamic_rig_running=true`。

## 自动验收结果

Unity `6000.2.9f1` 构建为 0 warning / 0 error。四个固定 seed 均独立运行 60.021 秒：

| 场景 | 遥测 | 目标 | 交互 | 覆盖率 | exposure proxy 均值 |
|---|---:|---:|---:|---:|---:|
| Calm Door | 2,223 行 | 2/2 | 1/1 | 100% | 5.54 |
| Multi-turn Corridor | 2,223 行 | 2/2 | 1/1 | 100% | 7.44 |
| Motion Lab | 2,223 行 | 2/2 | 1/1 | 100% | 15.18 |
| Semantic Roller Coaster | 2,223 行 | 3/3 | 2/2 | 100% | 15.31 |

前三条步行路径的实测最大平面加速度分别约为 `1.800 / 1.801 / 1.800 m/s²`。过山车场景记录到 445 行载具运动样本，最高速度 `1.552 m/s`，载具垂直范围 `1.150 m`，同期相机垂直范围 `1.149 m`。Motion Lab 和过山车的代理均值高于基线只用于验证管线是否对动态场景变化有响应，不能用作效度或因果结论。

## 重新构建

在仓库根目录运行：

```powershell
.\demo\build_unity_scene_aware_agent.ps1
```

它会重新导出/校验真实训练模型、构建独立 Windows Player，并运行四个 60 秒验收。开发时若模型导出没有变化，可用 `-SkipExport`；仅做构建调试可再加 `-SkipTests`。

## 尚未完成的产品化部分

- EXE 内没有在线调用 LLM。当前四个测试场景把设计时语义理解编译成显式、可查询的运行时契约；这是为了可复现和安全执行。
- 任意 Unity 项目仍需把现有 `scripts/unity_parser.py` 的静态结果映射为带世界坐标、可达边和稳定 ID 的 runtime graph。
- 未标注的项目交互不能靠猜。项目需提供白名单 `AIAffordance`/adapter、前置条件和成功状态。
- 当前是 waypoint graph，不是任意复杂场景的 NavMesh、碰撞和动态重规划。
- 当前 BodyRoot 路线引导是规则层。要让走路本身也由训练模型按目标生成，需要用含身体/控制输入的数据训练 goal-conditioned locomotion policy；EyeNavGS 只有 HMD 轨迹，不能证明真实脚步或身体姿态。
- VR.net 探索模型需要 5/30/60 秒窗口，且当前最佳 AUC 结果仍未校准。正式接入前应完成单位对齐、窗口 coverage、跨游戏/参与者验证和反事实同路径重放。

下一阶段应先把真实项目解析结果编译成上述 runtime graph，再加入 NavMesh/稳定交互 adapter；闭环稳定后，才训练 goal-conditioned BodyRoot，并把 VR.net 特征模型作为版本化的离线/运行时评分层接入。
