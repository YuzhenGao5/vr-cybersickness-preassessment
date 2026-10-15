# Unity Scene Agent Terminal

## 现在走通的流程

终端把原来分开的“解析”和“运行预制 Demo”接成了一条有证据门控的四步流水线：

```text
Unity 项目、Scene、C# / Prefab / ProjectSettings
  → 只读 AI 场景理解报告
  → 确定性 Agent 契约（坐标、目标、约束、可验证交互候选）
  → 隔离 staging 副本中注入 Generic Brain + Normal HMD v4
  → 运行时对象落地、路径/目标决策、停走/注视/交互技能调度
  → 直到场景内容完成的第一人称验收、体验报告和 ZIP 证据包
```

源 Unity 项目不会被注入或修改。每个构建都在
`demo/UnityGenericSceneAgentProjects/` 的隔离副本中完成；Player 位于
`demo/GenericSceneAgentBuilds/`。项目与 Scene 的绝对身份会参与稳定哈希，
因此同名项目或同名 Scene 不会共用构建证据。

## 四步操作

交付入口：

```text
demo/SceneAgentTerminalBuild/SceneAgentTerminal.exe
```

若尚未生成 EXE，在仓库根目录运行：

```powershell
.\demo\build_scene_agent_terminal.ps1
```

打开终端后：

先选择 **Decision Brain 模式**：

- **Rule**：确定性规则大脑，始终可选；Unity 扁平模型缺失或无效时仍可解析、构建和运行。
- **Shadow**：需要 `models/unity_decision_brain_v0_unity.json`。模型与 Rule 同时推理并记录分歧，但执行权仍属于 Rule。
- **Model research**：让模型取得闭环决策权。任何不是 `production_candidate` 的模型都必须再勾选研究覆盖；当前模型是 `smoke_only`，这不是生产模式。

模型缺失只会使 Shadow / Model research 明确显示不可用，不会把 Rule、AI 解析或整个软件一起禁用。

1. **AI 解析 + 生成契约**：读取所选项目的 Scene、脚本、Prefab 和工程设置。终端内“AI 场景理解报告”页会显示完整报告，并明确本次使用 `codex_cli` 还是确定性回退。LLM 只能提出语义与规划建议；它不能直接授权任意 C# 方法调用。
2. **隔离构建 Agent**：复制 `Assets / Packages / ProjectSettings` 到 staging，注入 Generic Brain、Normal HMD v4 模型和已编译契约，然后构建 Player。构建器会运行到所有已落地目标完成且 required interaction 已验证，再检查第一人称 Camera、截图与非黑屏/非洋红渲染。全部通过后场景才变成“Agent 可体验”。
3. **启动 Agent 体验**：每次使用新 seed。Brain 会在运行时核对真实对象，选择下一个目标，决定走/停/注视/观察/交互/重规划；运动技能生成 HeadLocal 和速度包络，规划器只改变任务方向。旁边的“内容完成复验”可再生成一份独立验收记录。
4. **导出结果 ZIP**：只允许导出已经得到 `content_complete` 的运行。ZIP 同时包含 AI 报告、静态证据、契约、Brain 决策账本、运动遥测、体验报告以及对应构建/渲染验收文件。

“添加项目目录…”会永久记录额外搜索根目录。隔离 staging 和构建目录不会被再次识别为用户项目。

## Agent 的“大脑”和技能

Generic Brain 是显式、可审计的高层执行器，目前包含：

- `SceneUnderstanding`：读取契约，并扫描运行时对象、Collider 和 NavMesh；
- `Plan`：根据优先级、距离、新颖度和已覆盖目标选择下一目标；
- `Navigate / Recover`：有 NavMesh 时使用它，没有时使用经碰撞/地面约束的直接规划和重规划；
- `LookAt / Observe`：到达后停下并观察，不让移动技能一直向前冲；
- `Interact`：只调用单独人工审核、零参数、在契约 allowlist 中的方法；
- `Verify`：读取声明的 Unity 状态成员，只有状态变化符合成功条件才计为交互成功；
- `Complete`：写出覆盖率、失败原因、运动统计与修改建议。

通用运行不再以 60 秒作为成功条件。全部内容提前完成就提前结束；远目标只要距离持续缩短就不会被误判为超时。连续 45 秒没有实质进展、内容确认不可达或达到 900 秒模拟安全上限时会明确失败，不会冒充完成。

导航指令在积分到 BodyRoot 之前统一通过全身胶囊 sweep；被挡时先稳定绕行，无法绕行则立即清零惯性并重规划。即使 Agent 静止或正在交互锁内，也会逐步检查动态障碍是否侵入胶囊；任何侵入都会令验收失败。地面投影会排除 Trigger、参与者自身 Collider 和过陡表面，并保留脚边可行走坡道。交互前会先停稳 0.35 秒，从调用到交互后 Observe/Verify 全程锁住 BodyRoot 平移、yaw 和地面高度基线；后置状态从下一渲染帧开始轮询，返回 `IEnumerator` 的玩法方法会作为 Unity coroutine 执行；EyeNavGS HeadLocal 仍继续生成。验收文件会记录锁定帧、非零运动违规与碰撞恢复证据。

训练的 Unity Decision Brain v0 现在可以通过同一运行时以三种权限接入。Player 参数为
`--decision-brain-mode=rule|shadow|model`；非 `production_candidate` 模型取得控制权还要求
`--decision-brain-research-override`。Shadow/Model 的每次前向推理写入
`decision_brain_inference.jsonl`；验收报告分别记录请求模式、实际 authority、模型状态、模型 SHA-256、
schema、dataStatus、golden parity、有限 logits、推理次数、回退次数和与规则决策的分歧次数。Model 模式遇到缺失/坏模型会 fail closed，绝不会静默冒充模型后改用 Rule。

Normal HMD v4 是被 Brain 调用的运动技能，不是“大脑本身”。EyeNavGS 训练得到的是局部头显
6-DoF 时序和 BodyRoot 速度包络；目标选择、地图理解、转向到目标、什么时候停以及调用哪个交互，
由契约落地后的高层 Brain 决定。这避免把规则规划或自定义玩法误称为 EyeNavGS 学到的能力。

## 通用场景支持边界

“任何场景”在这个版本中的准确含义是：终端可以扫描和解析普通文本序列化的 Unity Scene，
并尝试在隔离副本中构建通用 Agent；不是保证任意第三方工程一次就能编译并完成所有玩法。

当前自动构建已针对 Unity `6000.2.9f1` 验证。版本不一致、缺包、平台专属插件或项目本身编译失败时，
终端仍能生成静态报告，但不会把它标成“Agent 可体验”。运行时创建的对象可由 live inventory 再发现；
没有 NavMesh 时可做保守 fallback，但复杂多层地图仍可能需要项目专用路径适配器。
隔离构建优先使用 PowerShell 7 (`pwsh`)；未安装时会自动使用 Windows 自带的
`powershell.exe`。构建脚本对含中文的 JSON 契约显式使用 UTF-8 解码，两种宿主均受支持。

未知玩法不会被猜测。只有项目源码中发现的方法候选还不够：必须有单独的 reviewed binding、
安全白名单和成功状态，Brain 才会调用。否则 Agent 仍可走近并观察，同时在报告中明确写“未验证交互”。
涉及真实参与者实验和数据写入的项目（当前包括 `Teach2SelfPrototype`）默认只读解析，除非先建立
不会污染研究数据的测试身份和 sandbox。

终端只信任 `configs/scene_agent_reviewed_bindings/` 下状态为 `approved` 的中央清单。清单必须锁定
项目绝对路径、Scene 路径和 Scene SHA-256；每个方法绑定还要记录 `reviewed_by`、
`reviewed_at_utc`、`script_path`、脚本 SHA-256、public 零参数实例方法以及可核验的成功成员。脚本
方法体变化即撤销旧授权。LLM 输出和被测试项目自己的文件都不能成为
授权源。清单路径、SHA-256 和绑定数会写入解析索引、build status、运行 manifest 和导出包；清单被
修改、撤销或删除后，旧 Player 会立即失去 `agent_ready` 资格。没有清单时 build status 明确记录
空路径/空哈希/`count=0`，Agent 严格为 observe-only。

唯一窄特例是经同一中央清单逐目标审核的 `Button` / `UnityEngine.UI.Button.Press`：因为它由
运行时适配器触发 `onClick`，不要求项目源码里声明一个同名 `Press` 方法，但静态契约必须同时
具有该目标的 Button 组件证据和 affordance。Scene/脚本哈希、审核人、审核时间和成功成员仍然
全部必填；这不会泛化为任意 UnityEvent 或任意方法调用。

`UnityRandomAgentDemo / AuroraMotionCourtyard` 有一个明确记录的兼容性 profile：隔离 Player 会停用
源 Demo 自带的 `RandomAgentController`，防止它与新 Brain 同时控制 Camera 并显示双 HUD。这只是
“source player conflict suppression”，不是给通用 Agent 预设路线；未知项目不会自动禁用任意 gameplay manager。

保留的已验收参考包括四个 SceneAware 场景和 `HeartislandUnity / ResonanceValley` 专用 adapter。
它们继续可用，但界面仍显示真实 `.unity` Scene；参考项不会再把项目中的其他 Scene 隐藏掉。

## 证据与输出

```text
results/scene_agent_terminal/
├─ semantic/   # 项目证据、Scene CSV、AI 报告、Agent 契约、契约状态与索引
├─ runs/       # 每次新 seed 的 Brain/运动遥测、日志与内容完成报告
└─ exports/    # 完整 ZIP 证据包

results/generic_scene_agent/
├─ *_build_status.json       # 源项目/Scene/契约/模型哈希和构建状态
├─ *_runtime_contract.json   # 运行时最小契约
├─ *_acceptance.json         # 内容完成、目标覆盖与交互验证
├─ *_camera_audit.json       # 第一人称 Camera/AudioListener/渲染后端
├─ *_first_person.png        # 实际 Player 截图
├─ *_render_validation.json  # 黑屏、尺寸、颜色/亮度方差检查
└─ *_runs/                   # runtime inventory、brain_decisions、telemetry、experience_report
```

隔离构建从 `models/unity_decision_brain_v0_unity.json` 读取
`unity-decision-brain-unity-flat-1.0`，复制到
`Assets/Resources/UnityDecisionBrainModel.json`（运行时 `Resources.Load`）和
`Assets/StreamingAssets/UnityDecisionBrainModel.json`（原始审计副本）。build status 用独立的
`decisionBrainModelSha256/schemaVersion/dataStatus/status` 字段记录它，并另存
`normalHmdModelSha256`；两种模型哈希不会混用。mode、模型哈希和研究覆盖也进入构建 identity，避免覆盖另一种模式的旧 Player。

通用运行目录中的关键文件：

- `runtime_scene_inventory.json`：契约目标与实际 Unity 对象的匹配情况；
- `brain_decisions.jsonl`：每次状态切换、选目标、停走、交互、验证和重规划；
- `generic_agent_telemetry.csv`：约 37 Hz 的 BodyRoot、Camera、训练先验与实际执行运动；
- `generic_agent_acceptance.json`：实际模拟时长、内容完成状态、覆盖、交互和失败门槛；
- `experience_report.md/json`：体验总结、暴露代理量和场景修改建议；
- `target_exposure_summary.md/json`：按 Brain 当前目标聚合的样本数、停留时间、速度、角速度、加速度和工程代理量，用于决定先检查哪个区域；它不是 VR.net 概率或因果归因；
- `normal_hmd/`：Normal HMD 模型自己的原始运动记录。

终端会在启动和导出前重新核对源 Scene SHA-256、AI 契约 SHA-256、运行时契约 SHA-256、
Player 构建时间、内容完成验收、第一人称 Camera 和渲染检查。任一项过期或指向别的项目/Scene，
旧 Player 都不会继续显示为可运行。

## 结果解释边界

- 这些轨迹是 Agent 模拟数据，不是真人参与者数据。
- Normal HMD v4 提供学习到的运动先验，但当前分布验收尚不能证明与所有真人等价。
- Brain 是“LLM 辅助契约 + 确定性规划/验证”，不是训练好的端到端视觉导航网络。
- 体验报告中的 exposure proxy 是未校准的工程信号，不是 VR.net 概率、个人晕动概率或医学诊断。
- 静态报告发现按钮/过山车脚本，不等于 Agent 已操作它；只有决策账本和 Unity 状态验证才能证明成功。
