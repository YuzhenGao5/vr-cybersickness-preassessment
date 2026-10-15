# Unity VR 晕动症事前评估 Agent：项目与版本完整报告

版本日期：2026-10-15  
版本定位：研究原型快照  
适用读者：合作研究者、HCI/VR 评审、Unity 开发者、接手工程师及没有历史记忆的 AI

## 1. 一句话说明

当前版本完成了一个可审计的 **Unity 项目解析 → Agent 契约 → 隔离构建 → 场景感知探索 →
安全交互 → 遥测与内容完成验收** 原型。EyeNavGS 模型提供较平滑的第一人称头/根运动先验，
Unity Decision Brain v0 已在研究模式中真实闭环运行；但是 VR.net AUC 0.6819 分类器仍只存在于
离线实验，尚未接入 Agent。因此系统现在能自动采集并排序暴露证据，不能宣称已经自动输出
经过验证的 VR 晕动概率、医学风险或因果设计建议。

## 2. 整个项目最终要做什么

目标用户是 VR 内容研究者和 Unity 开发者。理想工作流如下：

1. 开发者选择一个新 Unity 项目和 Scene。
2. 系统读取 `.unity`、Prefab、C#、Packages 和 ProjectSettings，建立可追溯的场景语义图。
3. 系统把语义图编译为 Agent 可执行契约：可观察目标、探索优先级、坐标、约束、交互候选和成功条件。
4. 一个高层“脑”决定下一步去哪、何时走/停/观察/交互/重规划。
5. 低层虚拟参与者运动技能生成类似普通真人的 BodyRoot 和 HMD 运动。
6. Agent 在隔离副本里体验内容；不修改开发者原工程，不污染真实研究数据。
7. 系统记录相机、头显、路径、场景对象、交互、视觉运动和暴露时间序列。
8. 未来的风险 critic 对轨迹和场景变化进行校准评分，并把高风险片段关联到可编辑设计因素。
9. 开发者修改速度、加速度、相机控制、参考框、交互节奏或场景结构，再自动回归比较。

最终想形成的是“设计时评估—修改—重测”的闭环，而不是在用户已经不适后才做运行时缓解。

## 3. 本项目针对的研究 Gap

| Gap | 本项目的应对 | 当前成熟度 |
|---|---|---|
| Unity 源代码语义与运行时体验脱节 | 静态解析器生成带 source reference 的报告和契约，再由运行时 grounding 核对对象 | 已形成原型 |
| 普通自动测试只检查功能，不模拟第一人称运动暴露 | EyeNavGS Normal HMD v4 生成头显/根动态并输出约 37 Hz 遥测 | 已接入，分布仍需改进 |
| 自由探索缺少可解释的高层控制 | 显式 Brain 状态机 + 可选 Decision Brain 模型 + 安全 mask | 工程闭环完成，模型仅 smoke |
| 自动交互容易误调用任意 C# 方法 | 中央 reviewed binding、零参数限制、脚本/Scene 哈希和后置状态验证 | 机制已实现；生产 binding 仍为空 |
| 固定 60 秒会在内容未完成时误判通过 | 内容完成制，15 秒证据下限、45 秒无进展、900 秒安全上限 | 已实现并真实验收 |
| 移动、交互期间的碰撞和惯性不安全 | 全高胶囊 sweep、稳定侧向绕行、惯性 hard-stop、持续 overlap 监测 | 已通过正/负 fixture |
| 晕动预测常与场景设计参数混淆 | 明确区分运行路径/用户反应特征、工程 proxy 和可编辑设计变量 | 科学口径已明确，最终 critic 未接入 |
| 工具结果难以审计与复现 | 契约、哈希、决策账本、context/outcome、截图、验收 JSON 和导出 ZIP | 已实现 |

它与“数字人”有关，但当前更准确的名称是 **embodied first-person test agent / 虚拟参与者**。
系统有 BodyRoot、HeadLocal、第一人称 Camera、停走/注视/交互和行为记录；没有全身 avatar、
四肢、足接触、步态生物力学、身体尺寸多样性或社交行为，BodyRoot 也是从 HMD 推导的 proxy。

## 4. 当前总体架构

```text
Unity Scene/Prefab/C#/settings
  -> 确定性静态解析 + 可选 LLM 叙述
  -> 带来源证据的 Scene-Agent Contract
  -> 隔离 Unity staging 和所选 Scene build
  -> Generic Brain: Scan/Plan/Navigate/Look/Interact/Verify/Replan
       -> Normal HMD v4: EyeNavGS 派生的 BodyRoot + HeadLocal 运动
       -> 安全层: NavMesh/碰撞/地面/交互 allowlist/跨帧后置条件
  -> telemetry/decisions/context/outcome/screenshot/acceptance
  -> 当前工程 exposure proxy；未来版本化、校准的风险 critic
```

核心设计原则是职责分离：

- LLM 可以概括代码和提出探索建议，但不能授权方法调用。
- 训练模型可以给目标/技能打分，但不能绕过 NavMesh、碰撞、cooldown 或交互白名单。
- Normal HMD 负责“怎么动得像一些真人轨迹”，不决定“要去哪里”。
- 风险分数不能反向伪装成训练标签或因果解释。

## 5. Unity 场景解析和契约

### 5.1 静态解析

主要实现：

- `scripts/unity_parser.py`：项目级证据解析，schema `1.1.0`；
- `scripts/parse_unity_scene.py`：相机/FOV、粒子、灯光、对象、层级、参考框等场景量；
- `scripts/compile_scene_agent_contract.py`：生成 `scene-agent-contract-1.0`；
- `scripts/materialize_generic_runtime_contract.py`：生成 `generic-scene-agent-contract-1.0`。

解析器读取文本序列化的 Scene、Prefab、脚本、Packages 和 ProjectSettings，生成层级、组件、
脚本证据、位置、风险候选、交互候选、优先级、注意事项和不确定项。可选 LLM/Codex 后端只补充
叙述与候选排序；稳定 ID、坐标、方法签名、哈希、成功谓词和运行权限仍由确定性编译器控制。

静态解析无法证明：运行时生成对象是否出现、Addressables/字符串加载的内容、一个目标是否可达、
方法调用是否成功，或一个关键词是否真正产生晕动。这些只能作为运行时待验证假设。

### 5.2 交互信任根

`configs/scene_agent_reviewed_bindings/` 是唯一中央信任根。一个自动交互至少要固定：

- 项目绝对身份和 Scene 路径；
- Scene SHA-256、脚本相对路径和脚本 SHA-256；
- 审核人、审核时间和 `approved` 状态；
- public、实例、零参数方法；
- 可读取的后置状态成员和期望值；
- 是否属于 required interaction。

项目自身文件、LLM 输出和反射发现都不能单独授予调用权限。脚本内容变化会使旧授权失效。
当前仓库的生产清单只有说明，没有实际项目批准项；所以最终 Garden 验收是导航/观察 10 个目标、
0 个声明交互。交互功能由专用正/负 fixture 验证，不能说 Garden 自己验证过玩法交互。

## 6. Generic Scene Agent Brain

主要实现位于 `demo/generic_scene_agent/`。运行时会盘点真实 GameObject、Collider 和 NavMesh，
把静态目标和 runtime-discovered observation 目标落地，然后循环执行：

- `Scan`：解析契约并核对对象；
- `Navigate`：选择未完成、合法、可达的目标；
- `LookAt` / `Observe`：到达后停止 BodyRoot 并观察；
- `Interact`：只执行获批绑定；
- `Verify`：跨帧检查后置状态；
- `Replan`：无进展、碰撞或目标失效时重新规划；
- `Complete`：只在内容条件满足时结束并输出报告。

有 baked NavMesh 时沿 path corner 前进；没有时明确记录为
`collision-checked-direct-steering(no-NavMesh)`。后者是带碰撞检查的局部直线引导，不能称为
完整全局路径规划。

### 6.1 时长与完成条件

通用 Agent 不再以 60 秒自动通过：

- 全部已落地目标 visited；
- 全部 required interaction verified；
- 至少 15 秒证据；
- 连续 45 秒无实质进展则 `stalled`；
- required interaction 不可成功或内容确认耗尽则 `content_exhausted`；
- 900 秒是安全上限，命中时 `safety_timeout`，不是成功。

单独的 Normal HMD 展示程序仍可固定生成 60 秒轨迹；它与通用内容完成验收不是一件事。

### 6.2 碰撞和地面

最终实际积分速度先经过完整身高 BodyRoot capsule sweep。正面被挡时尝试稳定的同侧绕行；
所有候选均被挡时立即清除旧平滑速度，避免惯性继续推墙，然后触发有限重规划。Agent 静止或
处在交互锁内时仍每步做 overlap 检查，动态障碍进入胶囊体也会令验收失败。

地面投影与运行时跟随共用判定：排除 trigger、自身 collider 和过陡表面，同时允许正常坡道。
记录的安全模式为：

```text
full-height-capsule-sweep+stable-side-steering+inertia-hard-stop
```

### 6.3 交互事务

调用前先用 0.35 秒让 BodyRoot 停稳。从调用开始到 Observe/Verify 完成，BodyRoot 的 XZ、yaw
和 ground baseline 被锁定；HeadLocal 继续生成，避免第一人称头部完全冻结。验证至少从下一
rendered frame 开始，返回 `IEnumerator` 的方法由 Unity coroutine 执行。

required 方法返回失败或后置条件超时后 fail closed，只调用一次，避免反复触发非幂等玩法。
方法调用前就已经等于期望值的状态不能替代本次调用成功证据。

## 7. 训练模型与数据

### 7.1 Normal HMD Agent v4

用途：低层普通头显/根运动技能，不是高层决策大脑，也不是晕动人群模型。

数据：EyeNavGS Rutgers + NTHU，46 人、12 场景、552 条 canonical trajectory、
1,931,091 个有效双眼姿态对。participant-disjoint split 为 32/7/7；训练/验证/测试片段为
25,546 / 5,927 / 5,592。原始数据不在 GitHub。

状态是 9 维：

```text
root right velocity, root forward velocity, root yaw rate,
HeadLocal right/up/forward, HeadLocal rotation-vector x/y/z
```

模型是三个独立的一层 LSTM branch，hidden size 96，共 125,689 参数：

- Root branch：连续 root 平移速度和 yaw rate；
- Horizontal branch：x/z/yaw/roll 的事件 onset、端点分布和时长；
- Vertical branch：y/pitch 的事件 onset、端点分布和时长。

事件开始时只采样一次端点和时长，再用五次 minimum-jerk 曲线
`10u^3 - 15u^4 + 6u^5` 插值；没有逐帧独立 HeadLocal 噪声。这正是消除旧版本上下/左右碎抖
的关键。模型保留 64 步、约 1.728 秒历史；训练 22 epoch，CUDA+AMP，checkpoint SHA-256：

```text
ad667a54dca8072cda0c684c4e15047e279938b702cd6be1ff7806d7449bad5b
```

真实边界：

- test onset AUROC：horizontal `0.7137`，vertical `0.6596`；开发中已反复查看 test，不能再称
  最终 untouched holdout；
- 16×60 秒 rollout 的整体真人分布验收 **未通过**：水平/垂直 event occupancy 为
  `0.1464 / 0.1484`，低于真人验证区间；
- 部署平滑验收通过：线性 jerk P95 从 `42.985` 降至 `9.6436 m/s^3`，yaw jerk P95 从
  `74.063` 降至 `15.812 deg/s^3`；
- `normal_hmd_v4_smooth_root.pth` 微调候选被拒绝；当前平滑来自 base v4 加校准后的运行时
  dynamics profile，不能声称产生了一个获批的新 smooth checkpoint。

Unity runtime JSON 还包含 414 个 EyeNavGS validation seed windows 和数据集伪 ID，用于不同 seed
初始化。因此仓库没有发布完整原始数据，但确实发布了受 Apache-2.0 约束的处理后小片段。

### 7.2 Unity Decision Brain v0

用途：对合法候选目标排序，并在 `Navigate / LookAt / Observe / Interact / Replan / Stop`
六个宏技能中打分。安全 mask 永远位于模型之上。

训练数据来自 40 次真实 Unity runtime run、919 个 context/outcome：245 个目标选择、674 个技能
切换。**所有标签来自规则 Brain bootstrap，真人 Unity 决策样本为 0**。数据只覆盖两个项目、
三个 scene group；test 和 validation 仍共享项目代码，不是未知项目泛化验证。

输入：21 维全局状态 + 每候选 29 维特征；不输入 ID、绝对坐标、未来 outcome 或规则 utility。
模型是 permutation-equivariant candidate set ranker，共 28,663 参数。

test 指标：

| 指标 | 数值 |
|---|---:|
| Target Top-1 | 0.5568 |
| Target Top-3 | 0.9205 |
| MRR | 0.7453 |
| NDCG | 0.8099 |
| Rule-utility Top-1 baseline | 0.6250 |
| Skill macro-F1 | 0.6667 |

目标 Top-1 比规则 utility baseline 低 0.0682；skill 的第一合法技能 baseline accuracy 为 1.0，
而训练集中没有 Interact 标签。因此模型必须保持：

```text
dataStatus = smoke_only
humanLikeClaimAllowed = false
```

运行模式：

- `Rule`：规则执行；
- `Shadow`：模型同时推理和记录分歧，Rule 仍执行；
- `Model research`：显式研究 override 后由模型执行；坏模型或非法输出 fail closed。

### 7.3 Habitat-Web 导航先验

这是一个尚未接入 Unity 的独立真人导航实验。数据为 Habitat-Web ObjectNav THDA 70k，
56 Matterport scenes，以 39/8/9 scene-disjoint split；70,176 episodes 中 60,085 个有可用目标，
最终 reservoir 为 600k / 120k / 120k。

模型输入是 pose、相对目标 waypoint 和历史动作，输出 forward/turn/look/stop 六类动作；
64-64 MLP，共 6,534 参数。test accuracy `0.6825`、balanced accuracy `0.5651`、macro-F1
`0.4050`；repeat-previous baseline accuracy `0.7205`。它没有 RGB、depth、地图、Unity 语义或
交互输入，`production_model=false`、`closed_loop=false`、`Unity integration=false`。

### 7.4 VR.net 晕动分类器

本地实验子集为 12 games、86 sessions、69 个 participant ID、1,024 次真实评分事件。
二分类标签：`MS_rating <= 2` 为 0（825 条），`MS_rating > 2` 为 1（199 条）。

每个评分时刻只使用过去 5/30/60 秒：camera speed、acceleration、rotation rate、
object-matrix count、head speed、head angular speed 的 mean/max/std，再加 exposure elapsed。
有效特征 55 个；保存的 pipeline 仍要求 73 个 raw columns，因为 18 个 FOV 统计在 pipeline 内
被近常量筛除。

探索中最高的 `scene_head_multiscale` LogisticRegression 结果：

| 评估 | Pooled OOF AUC | Mean fold AUC | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| Leave-One-Game-Out | 0.6819 | 0.6696 | 0.6553 | 0.5686 |
| 同时排除测试参与者 | 0.6082 | 0.6440 | 0.6563 | 0.5486 |

`0.6819` 是多个探索方案中的最好观察结果，存在选择偏差；完整 nested-selected 结果只有
`0.5660`。分数表示模型对“评分 > 2”的倾向，既不是 68.19% 准确率，也不是校准后的个人概率。

更重要的是，它依赖真实运行路径和真人头动。头动可能是对不适的反应，不是开发者可直接修改的
设计变量。该 pickle 当前没有被 Terminal 或 Unity runtime 调用。现有体验报告用的是一个基于
camera speed/angular speed/acceleration 的未校准工程 proxy，只能用于片段排序。

## 8. 当前 Demo 和 UI 做了什么

入口源码：`demo/scene_agent_terminal.py`。从源码打包：

```powershell
.\demo\build_scene_agent_terminal.ps1
```

终端支持：

1. 扫描/添加 Unity 项目目录并列出真实 Scene；
2. 在 UI 中展示 AI/规则解析报告和契约状态；
3. 选择 Rule、Shadow 或 Model research；
4. 在隔离目录复制 `Assets / Packages / ProjectSettings`；
5. 注入通用 Brain、Normal HMD runtime 和最小契约；
6. 构建所选 Scene，校验源 Scene 前后 SHA 一致；
7. 运行内容完成验收、Camera audit、D3D11 截图和非黑屏/非洋红验证；
8. 运行不同 seed 的体验并导出 ZIP 证据。

异步工作期间终端会显示 heartbeat，并在结束后恢复按钮。Shadow/Model 的模型不可用不会禁用
Rule 和静态解析。每个构建 identity 包含项目绝对身份、Scene、模式、模型哈希和 override，避免
误用别的项目或旧 Player。

每次运行的本地产物包括：

```text
runtime_scene_inventory.json
brain_decisions.jsonl
decision_contexts.jsonl
decision_outcomes.jsonl
decision_brain_inference.jsonl
generic_agent_telemetry.csv
generic_agent_acceptance.json
experience_report.md / .json
target_exposure_summary.md / .json
normal_hmd/...
```

这些结果默认不进 Git，因为可能含绝对路径、session 标识或大量逐帧记录。公开仓库只保留脱敏
聚合结果 `release/research_snapshot_2026-10-15.json`。

## 9. 最终验收结果

所有最终 fixture 使用同一个 `GenericSceneAgentBrain.cs`：

```text
b070363b572e3fc70b3a52528923ed091e68ed496a0c0888c0b2af657d1a4196
```

### 9.1 生产参考 Scene

| 模式 | 状态 | 模拟时间 | 内容 | 模型证据 |
|---|---|---:|---:|---|
| Rule | passed / content_complete | 317.304 s | 10/10 | authority=rule |
| Model research | passed / content_complete | 317.304 s | 10/10 | 40 inference、0 fallback、2 disagreement |

Model run 同时满足：`golden self-test=true`、`all logits finite=true`、`dataStatus=smoke_only`、
显式 research override、collision overlap=0、interaction lock violation=0。

### 9.2 碰撞 fixture

- 正例：`77.895 s`，4/4 目标，1 个 required interaction verified，471 次安全绕行调整，
  0 hard-stop、0 overlap；15°/30°坡均到达，trigger、80°陡面和 participant decoy 被正确过滤。
- 负例：在 interaction lock 时强制把 Box 插入胶囊体，检测 1 个 overlap episode，
  `collisionSafetyPassed=false`，验收按预期失败并退出 3。

### 9.3 异步交互 fixture

- 正例：`34.533 s`，2/2 required interactions；next-frame 与 coroutine 都在 invocation 后一帧
  才满足；lock 2/2、BodyRoot violation 0、HeadLocal 在锁内活动。
- 负例：方法返回 false 且状态预先伪装为 expected；仍只调用 1 次，失败目标未 visited/verified，
  `8.667 s` 以 `content_exhausted` 终止，没有空耗到 900 秒。

### 9.4 自动化

- 发布前 Python/源码/数据管线完整回归：88/88 passed；较早的最终 Unity 修复报告记录为 74/74；
- Rule、Model、collision、async fixture 均用 Unity `6000.2.9f1` 构建；
- 最终生产构建 0 errors；
- Terminal source self-test 与 packaged self-test 均 passed；
- 第一人称 render validation passed。

旧的 `results/decision_brain_demo_handoff_2026-10-15.md` 中 `59.994 s / 8 of 10 /
33 inference` 是中间版本，已被这里的 `317.304 s / 10 of 10 / 40 inference` 最终验收取代。

## 10. “可以放进任何 Unity 场景吗”的准确答案

可以做的是：对常规文本序列化 Unity 工程自动扫描、解析、生成最小契约、在隔离副本中构建，
再用 runtime inventory 尝试探索。

不能保证的是：任意 Unity 版本、缺包工程、平台专属插件、复杂 XR/网络初始化、Addressables、
程序生成、多层 NavMesh、需要参数的方法、隐藏状态机或有真实副作用的研究流程一次成功。

边界如下：

- 项目本身必须能在指定 Unity 版本编译；
- 没有 NavMesh 时只有局部碰撞检查转向；
- 未审核玩法只能走近观察，不能猜测调用；
- runtime-generated 对象可以被发现，但不等于其完整玩法被理解；
- 真正第三方项目应先在 sandbox 用测试身份运行；
- SceneAware 合成场景和 HeartIsland 专用 adapter 是参考，不是“零适配征服任意项目”的证据。

## 11. 对 HCI 研究够不够

作为 HCI 系统原型、方法 demo、pilot、自动化回归和数据采集基础，当前版本已经有研究价值：

- 模块边界清楚；
- 训练来源、规则来源和运行证据可追溯；
- 有真实 Unity Player 正/负验收；
- 能对场景和目标形成逐时序证据；
- 能明确失败，而不是把未完成冒充成功。

但它还不足以支持下面的论文结论：

- “Agent 已经像真人一样探索任意场景”；
- “系统能预测一个新用户是否会晕”；
- “某个 Unity 参数变化会因果降低晕动”；
- “晕动人群行为模型已经训练完成”；
- “Decision Brain 优于规则基线”；
- “AUC 0.6819 是独立最终验证”。

要支持这些结论，需要人类被试对照、真正独立项目/场景、预注册 holdout、风险校准、置信区间、
失败分析和因果/干预实验。

## 12. 数据、隐私和许可证

不随 GitHub 发布：

- EyeNavGS 完整 CSV；
- VR.net 原始包、本地 session、逐事件 features 和 OOF predictions；
- Habitat-Web 原始/处理数据；
- `.pth/.pkl` checkpoint；
- Unity Library、staging、Player、EXE、截图日志和逐帧结果。

发布的 Normal HMD Unity JSON 含处理后的 EyeNavGS seed windows，按 Apache-2.0 归因。
VR.net 当前公开包的再分发许可没有得到足够确认，所以连 pickle 也不发布。Habitat-Web prior 没有
接入当前 runtime，其权重也不随本快照发布。

项目自身目前没有开源许可证。公开可读不构成复用授权，第三方资产许可也不能替代项目级许可。

## 13. 仓库地图

| 路径 | 作用 |
|---|---|
| `demo/scene_agent_terminal.py` | 桌面终端 UI 与流水线编排 |
| `demo/build_generic_scene_agent.ps1` | 隔离复制、注入、Unity build 和验收 |
| `demo/generic_scene_agent/` | Generic Brain、contract、builder、decision runtime、render probe |
| `demo/UnityNormalHmdAgentDemo/` | 当前 Normal HMD Unity 源项目与 runtime JSON |
| `scripts/unity_parser.py` | 静态项目语义解析 |
| `scripts/compile_scene_agent_contract.py` | 分析器到 Agent 契约 |
| `scripts/materialize_generic_runtime_contract.py` | 契约到 Unity runtime schema |
| `scripts/normal_hmd_v4_*` | Normal HMD v4 数据、训练、导出和验证 |
| `scripts/*unity_decision*` | Decision Brain 数据、训练、policy 和 Unity export |
| `scripts/*auc_signal_round2*` | VR.net AUC 0.6819 实验 |
| `configs/` | 配置、schema、prompt、reviewed-binding 规则 |
| `models/normal_hmd_v4_metadata.json` | v4 训练 provenance 与架构元数据 |
| `models/unity_decision_brain_v0_unity.json` | C# 可加载的 smoke-only 决策模型 |
| `release/research_snapshot_2026-10-15.json` | 脱敏机器可读验收摘要 |

## 14. 下一步优先级

### P0：接入风险 critic，但不扩大结论

把 `generic_agent_telemetry.csv` 映射到 VR.net pipeline 要求的 73 个 raw columns，逐项核验单位、
采样、窗口覆盖、object count 定义和缺失机制。先做离线评分与一致性测试，再将版本化 critic 接入
Terminal。输出必须叫“未校准分类分数”，并保留工程 proxy 作为单独字段。

### P0：建立真正的跨项目验证集

至少收集 6 个独立 Unity project/scene family，覆盖 NavMesh、遮挡、动态障碍、异步交互、
多阶段玩法和 Stop。当前三个 group 且共享代码不足以证明泛化。

### P1：训练真正的决策 Brain

收集人类或专家 Unity 探索示范/纠正，重点补齐 Interact、碰撞恢复、visibility-positive、NavMesh、
停止和目标放弃。保留 `sourceKind`，不能把 model rollout 静默当成人类标签。

### P1：改进 Normal HMD 长序列

提高水平/垂直 event occupancy、持续时间和联合动力学，冻结新的 participant holdout，并进行真人
主观自然度 pilot。不要再次用已看过的 test 当最终结论。

### P2：晕动敏感人群 Agent

EyeNavGS 没有晕动标签，VR.net 当前模型又混合了场景运动和真人反应。要做“易晕人群模拟”，
需要带 susceptibility、连续症状、行为和 HMD 时序的配对数据，建立 normal 与 susceptible 条件模型，
而不是直接把 VR.net 分类标签微调进普通运动生成器。

### P2：工程发布

- Terminal UI 暴露 `-UnityEditor` 配置或增加安全自动发现（底层 build script 已支持该参数）；
- 为项目选择正式许可证；
- 在 CI 上增加 Windows + Unity license 可用时的构建验证；
- 将 EXE/Player 放 GitHub Release，而不是 Git 历史；
- 为 reviewed bindings 提供审核工具和 schema validator。

## 15. 最终口径表

| 问题 | 正确答案 |
|---|---|
| Agent 的运动是不是训练的？ | 部分是。Normal HMD 的 root/head 时序来自 EyeNavGS；minimum-jerk、采样、smoothing、clip、路径和安全层是工程逻辑。 |
| 大脑是不是训练的？ | v0 训练过，但标签全由规则教师生成，仍是 smoke-only；Rule Brain 仍是可靠基线。 |
| 能不能交互？ | 机制可以；必须有独立 reviewed binding 和可核验状态。当前生产清单为空。 |
| 能不能自动体验到内容结束？ | 对契约中可落地、可完成的内容可以；不再固定 60 秒。未知隐藏玩法不保证。 |
| 能不能放入任何 Scene？ | 可以尝试普通工程；不能保证零适配或所有项目构建/玩法成功。 |
| 能不能自动预测眩晕？ | 目前不能给出经验证概率。只能采集 Agent 暴露数据和未校准工程 proxy；VR.net 模型尚未接入。 |
| 是不是完整数字人？ | 不是。它是第一人称虚拟参与者，没有完整身体/步态/社交模型。 |
| 数据是否上传？ | 完整原始数据和 VR.net 逐参与者/逐事件特征表不上传；发布代码、聚合摘要，以及带明确归因、仍含伪名轨迹 ID 的小型 EyeNavGS 派生 runtime windows。 |

本报告是 2026-10-15 快照的权威总述。如历史文档、旧截图或旧 Player 与本报告冲突，以本报告和
`release/research_snapshot_2026-10-15.json` 为准；任何新实验应新建版本化快照，不能覆盖本版本口径。
