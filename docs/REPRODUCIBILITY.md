# 复现、测试和继续开发

本说明把“无需数据即可验证的源码”、“需要公开数据重新训练的模型”和“只在原工作站保存的
本地证据”分开。完整数值口径见 `docs/PROJECT_REPORT_2026-10-15.md`。

## 1. 已验证环境

| 组件 | 快照环境 |
|---|---|
| OS | Windows 10/11 |
| Unity | `6000.2.9f1` |
| 通用 Python 环境 | Python 3.10+，`requirements.txt` |
| Normal HMD 环境 | Python 3.12.10，NumPy 1.26.4，PyTorch 2.7.0+cu128 |
| Normal HMD 采样间隔 | 0.027 s |
| PowerShell | PowerShell 7 推荐；Windows PowerShell 5.1 已处理 UTF-8 contract |

`demo/build_generic_scene_agent.ps1` 的默认 Unity 路径为：

```text
C:\Program Files\Unity\Hub\Editor\6000.2.9f1\Editor\Unity.exe
```

如果安装位置不同，调用时传 `-UnityEditor 'D:\path\to\Unity.exe'`。Terminal UI 当前仍使用默认路径；
后续可增加安全自动发现。不要把新路径写死成另一个个人目录。

## 2. 安装

通用研究环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_imports.py
```

Normal HMD 独立环境：

```powershell
py -3.12 -m venv .venv-normal-hmd
.\.venv-normal-hmd\Scripts\Activate.ps1
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-normal-hmd.txt
```

没有兼容 NVIDIA GPU 时可安装 CPU 版 PyTorch；训练会变慢，但 Unity 已导出的 JSON 不要求 Python
或 GPU 才能在 Player 中推理。

## 3. Clone 后不需要研究数据的检查

发布内容安全检查：

```powershell
python scripts/check_release_package.py
```

轻量源码测试：

```powershell
python -m unittest `
  scripts.test_unity_parser `
  scripts.test_compile_scene_agent_contract `
  scripts.test_generic_scene_agent_collision `
  scripts.test_generic_scene_agent_interaction_lock `
  scripts.test_scene_agent_terminal_acceptance -v
```

安装全部 Python 依赖后运行所有 test module：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s scripts -p "test_*.py" -v
```

2026-10-15 发布前完整重跑结果是 88/88 passed；较早的 Unity 修复验收报告记录为 74/74。
GitHub clone 不包含本地 Unity Player 结果，
所以这条数字是快照证据，不应把没有重新执行的环境写成“再次通过”。

## 4. Terminal 自检与打包

源码自检：

```powershell
.\.venv\Scripts\python.exe demo/scene_agent_terminal.py `
  --self-test `
  --self-test-output tmp/terminal_source_self_test.json
```

生成 Windows one-file UI，并自动运行 source/packaged self-test：

```powershell
.\demo\build_scene_agent_terminal.ps1
```

输出位于 `demo/SceneAgentTerminalBuild/`，按设计被 Git 忽略。EXE 和 Player 应作为版本化 Release
附件分发，而不是写入 Git 历史。

## 5. Unity 源 Demo

用 Unity Hub 打开：

```text
demo/UnityNormalHmdAgentDemo
```

ProjectVersion 固定为 `6000.2.9f1`。仓库包含 Normal HMD runtime JSON，因此不需要 `.pth` 即可
加载当前 v4。Decision Brain 的 Unity-flat JSON 位于 `models/unity_decision_brain_v0_unity.json`。

构建一个外部场景的命令示例：

```powershell
.\demo\build_generic_scene_agent.ps1 `
  -SourceProject 'C:\path\to\UnityProject' `
  -CompiledContract '.\path\to\scene_agent_contract.json' `
  -ScenePath 'Assets/Scenes/ChosenScene.unity' `
  -DecisionBrainMode rule `
  -Seed 2026101501
```

Shadow 模式：

```powershell
.\demo\build_generic_scene_agent.ps1 `
  -SourceProject 'C:\path\to\UnityProject' `
  -CompiledContract '.\path\to\scene_agent_contract.json' `
  -ScenePath 'Assets/Scenes/ChosenScene.unity' `
  -DecisionBrainMode shadow `
  -DecisionBrainModel '.\models\unity_decision_brain_v0_unity.json'
```

当前 smoke-only 模型取得控制权必须同时传：

```text
-DecisionBrainMode model -DecisionBrainResearchOverride
```

不要把该 override 放进生产默认值。Model 模式缺模型、schema 不匹配、golden parity 失败或 logits
非有限时必须 fail closed。

## 6. 场景解析

确定性项目解析的入口和参数以脚本帮助为准：

```powershell
python scripts/unity_parser.py --help
python scripts/compile_scene_agent_contract.py --help
python scripts/materialize_generic_runtime_contract.py --help
```

可选 LLM 叙述需要在本机环境变量中设置凭据；不要把 API key 写入配置、日志或 commit。LLM 关闭时
确定性后端仍应生成可用报告和契约。

## 7. Normal HMD v4 重训

先按 `docs/DATASETS.md` 放置 EyeNavGS，然后：

```powershell
.\.venv-normal-hmd\Scripts\python.exe scripts/check_eyenavgS.py --deep
.\.venv-normal-hmd\Scripts\python.exe scripts/normal_hmd_v4_train.py --help
.\.venv-normal-hmd\Scripts\python.exe scripts/normal_hmd_v4_export.py --help
.\.venv-normal-hmd\Scripts\python.exe scripts/normal_hmd_v4_validate.py --help
```

不要在未读 `--help` 和配置前直接覆盖 `models/normal_hmd_v4.pth` 或 Unity runtime JSON。新训练必须：

- 重新冻结 participant-disjoint split；
- 保留新的真正 untouched holdout；
- 记录 dataset fingerprint、checkpoint hash、seed 和环境；
- 同时报告 predictive metrics、60 秒 closed-loop 分布和 Unity dynamics；
- 失败门槛也写进报告，不只保存最好指标。

## 8. Unity Decision Brain 重训

```powershell
.\.venv-normal-hmd\Scripts\python.exe scripts/prepare_unity_decision_data.py --help
.\.venv-normal-hmd\Scripts\python.exe scripts/train_unity_decision_brain.py --help
.\.venv-normal-hmd\Scripts\python.exe scripts/export_unity_decision_brain.py --help
.\.venv-normal-hmd\Scripts\python.exe -m unittest scripts.test_unity_decision_pipeline -v
```

数据的 `sourceKind` 必须保留。Rule/Shadow 运行是 `rule_bootstrap`，模型运行是 `model_rollout`；
它们都不是 human demonstration。模型 rollout 不得静默加入允许的训练源。收集到真实人类控制时，
使用独立 provenance，并按 project/scene family 做无泄漏切分。

## 9. Habitat-Web prior

修改 `configs/habitat_decision_prior_v1.json` 中的路径后：

```powershell
.\.venv-normal-hmd\Scripts\python.exe scripts/prepare_habitat_decision_data.py `
  --config configs/habitat_decision_prior_v1.json
.\.venv-normal-hmd\Scripts\python.exe scripts/train_habitat_decision_prior.py `
  --config configs/habitat_decision_prior_v1.json --device cuda
.\.venv-normal-hmd\Scripts\python.exe -m unittest scripts.test_habitat_decision_pipeline -v
```

该模型的离线成功不能被写成 Unity integration 成功。

## 10. VR.net AUC 实验

本仓库保留脚本和冻结配置，但不保留数据、event table、OOF prediction 或 pickle：

```powershell
python scripts/prepare_auc_signal_round2.py --help
python scripts/train_auc_signal_round2.py --help
python scripts/check_auc_signal_round2.py --help
```

保存的本地 pipeline 要求 73 列 raw input，内部保留 55 列。不要提前手动 log/standardize，也不要
把 55 列直接传给旧 bundle。任何重新训练都要同时报告 LOGO、participant-excluded double holdout、
nested selection 和 label prevalence。

## 11. 发布检查清单

在 commit/push 前：

1. `git status --short --ignored` 检查数据和构建确实被忽略；
2. `python scripts/check_release_package.py`；
3. 只暂存明确的源码/文档/配置/小型 runtime JSON；
4. `git diff --cached --stat` 和 `git diff --cached`；
5. 检查没有 participant/session 表、绝对个人路径、token、EXE、pickle 或 >50 MiB 文件；
6. 更新新的 dated report 和 `release/*.json`，不要覆盖历史结果；
7. 项目许可证未决定前，不要声称这是开源项目。
