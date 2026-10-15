# EyeNavGS 6-DoF Motion Imitation Demo

## 快速开始

双击 `AgentExplorationDemo.exe` 可打开实时动画。左图显示短时 6-DoF 运动模型积分得到的轨迹，右图显示局部线速度和角速度。

这是一项**真人头部运动行为克隆演示**，不是已经具备场景理解、目标规划、碰撞检测或风险区规避能力的自主探索 Agent。当前模型只读取最近 8 步局部线速度/角速度，不读取右侧叠加的 Unity 对象或风险区。

## 数据与模型

- 数据：EyeNavGS 官方 Rutgers 与 NTHU 数据，非合成数据。
- 覆盖：46 名参与者、12 个场景、552 条标准轨迹。
- 预处理：同一时刻的左右眼记录配对；位置取双眼中点作为近似头部中心；四元数对齐符号后平均。
- 输入：最近 8 步的局部线速度（m/s）和局部角速度（rad/s），形状为 `8 × 6`。
- 输出：下一步 3 维局部线速度与 3 维局部角速度。
- 划分：先按参与者划分，再生成窗口；训练/验证/测试分别为 32/7/7 名参与者。
- 训练样本：147,456；验证样本：32,256；测试样本：32,256。

独立测试集结果（7 名未进入训练的参与者）：

| 指标 | LSTM | 上一步速度保持基线 |
|---|---:|---:|
| 线速度向量 MAE | 0.06138 m/s | 0.07135 m/s |
| 角速度向量 MAE | 15.3220 deg/s | 16.7267 deg/s |
| 单步位置误差 | 1.3287 mm | 1.5092 mm |
| 单步姿态测地误差 | 0.34131° | 0.37229° |

这些是单步运动预测指标，不是探索覆盖率、避障成功率，也不是晕动症风险预测指标。

## 场景叠加层

演示捆绑 `results/unity_parse_result.json` 作为视觉上下文。当前文件来自合成 Unity 语义 fixture，且坐标会缩放到演示舞台中；它只用于显示。模型不会看到这些对象/风险区，也不会据此改变动作。未知风险半径仅使用明确标注的显示半径，不能解释为真实风险范围。

## 输出文件

- `agent_exploration.gif`：演示动画。
- `agent_trajectory_static.png`：静态轨迹与速度曲线。
- `AgentExplorationDemo.exe`：PyInstaller one-file Windows 程序。

EXE 默认只显示窗口，不会在打包资源临时目录中写文件。需要导出时，可从终端运行：

```powershell
.\demo\AgentExplorationDemo.exe --save-gif --save-static --output-dir .\demo
```

首次启动较慢属于正常现象：one-file 程序需要先解压 PyTorch、Matplotlib 等运行库。

当前构建已完成验证：文件大小为 200,488,053 字节（约 191.2 MiB），SHA-256 为 `d4cb95ac000afeab74e592b744298317121a3c51f98a234bdcacd3251446111f`。它已从仓库外目录通过内嵌资源/推理自检，并以无参数方式启动出可响应的窗口。

## 从源码运行

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe .\demo\agent_demo.py
```

生成 GIF 和静态图但不打开窗口：

```powershell
.\.venv\Scripts\python.exe .\demo\agent_demo.py --no-show --save-gif --save-static --output-dir .\demo
```

仅检查模型、资源和一次推理：

```powershell
.\.venv\Scripts\python.exe .\demo\agent_demo.py --self-test
```

## 数据检查、预处理与重新训练

```powershell
.\.venv\Scripts\python.exe .\scripts\check_eyenavgS.py --deep
.\.venv\Scripts\python.exe .\scripts\prepare_exploration_data.py --source eyenavgs
.\.venv\Scripts\python.exe .\scripts\train_exploration.py
```

检查报告、预处理报告和完整训练指标分别位于：

- `results/eyenavgs_check.json`
- `results/exploration_data_report.json`
- `results/exploration_training_metrics.json`

## 重新打包

```powershell
.\.venv\Scripts\python.exe -m pip install pyinstaller
.\.venv\Scripts\python.exe .\demo\build_exe.py
```

构建脚本只把探索模型、模型元数据和指定的 Unity 解析 JSON 作为数据资源加入 EXE，并会从仓库外的临时目录运行 `--self-test`。结果记录在 `results/exe_build_status.json`，构建日志记录在 `results/exe_build.log`。

## 下一阶段

若目标是“在任意 Unity 项目中自由探索并为晕动症评估收集数据”，下一版需要增加 Unity 运行时桥接和场景条件输入，例如深度/碰撞传感器、可行走网格、目标或覆盖状态、风险语义，以及线速度/角速度控制接口；再用真实轨迹预训练并用场景交互数据进行模仿学习或强化学习。当前模型可以作为“人类运动先验”，不能单独承担导航策略。
