# VR Cybersickness Preassessment

面向 VR 研究人员的预暴露晕动症风险评估与缓解建议引擎研究工作区。项目范围、假设、分组验证、主要指标、标签规则和继续门槛已冻结在 [`PROTOCOL.md`](PROTOCOL.md)。本项目不用于医疗诊断。

## 当前状态

目前为研究协议与环境骨架；尚未配置数据集或运行实验。任何实验开始前，先检查 `PROTOCOL.md` 并按 [`REPLICATION.md`](REPLICATION.md) 记录数据、标签、基线与执行环境。

## 环境

- Python 3.10 或更高版本
- `requirements.txt` 中固定版本的 Python 依赖
- LaTeX 发行版（例如 TeX Live 或 MiKTeX），仅在编译论文或报告时需要；它不是 pip 包，也不由 `requirements.txt` 安装

创建隔离环境并安装 Python 依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

安装后运行 `python scripts/check_imports.py` 检查 Python 包导入。该检查只验证环境，不运行研究实验。

## 目录

```text
paper/       论文与报告草稿
data/        原始及处理后的数据（默认不纳入版本控制）
features/    特征表及特征定义
models/      模型配置与产物
results/     指标、折外预测和分析结果
figures/     图表
refs/        文献与参考资料
scripts/     环境检查及可复现执行脚本
audit/       协议偏离、数据来源和执行审计记录
```

## 开始前

1. 核实并填写数据来源、标签来源和风险分档。
2. 在 `REPLICATION.md` 中登记基线、参数、随机种子、预处理和执行命令。
3. 确认 `PROTOCOL.md` 已提交并冻结。
4. 按协议执行 subject LOSO 和 leave-one-scene-out；不得在测试折上做选择。
