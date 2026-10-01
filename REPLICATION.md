# 复现记录

本文件记录数据、运行环境、实现细节及偏离情况。运行实验前填写所有适用字段；未知信息明确写为“待确认”，不得猜测。

## 协议

- 协议版本／提交：
- 执行日期与负责人：
- 协议偏离及 `audit/` 记录链接：

## 数据与标签

- 数据集名称、版本、来源与访问日期：
- 数据许可／访问限制：
- 样本单位及样本数：
- 参与者数量、场景数量：
- 标签来源（实测问卷／自评／代理标签）：
- 若为代理标签，醒目标注代理规则、依据与局限：
- 量表、测量时点、单位、低／中／高分档及依据：
- 缺失值与排除规则：

## 特征与预处理

- 光流、深度、运动轴数、参考框架的操作定义及单位：
- 视频抽帧率、图像尺寸、深度来源（如适用）：
- 特征汇总方式及类别编码：
- 归一化、缺失值处理、特征选择步骤：
- 防止参与者／场景信息泄漏的措施：

## 验证、基线与指标

- 分组键与 LOSO 折数：
- leave-one-scene-out 场景列表与折数：
- 预登记基线模型、参数、输入特征：
- 随机种子：
- 内层调参策略（如适用）：
- AUC 定义：macro one-vs-rest；门槛：0.60：
- F1：macro；Accuracy：总体；MAE／RMSE：低=0、中=1、高=2：
- 类别缺失导致指标不可计算时的处理：
- 基线门控结果与决策（AUC < 0.60 停止并诊断；AUC ≥ 0.60 继续）：

## 随机种子

- 全局种子：42（Python `random`、NumPy、PyTorch、LightGBM；使用 XGBoost 时也设为 42）。

## 软件与执行

- 已验证环境：Windows x64；官方用户级 Python 3.12.10；`requirements.txt` 安装成功；导入检查 10/10 通过。
- Python：3.12.10。
- OpenCV：`opencv-python-headless` 4.10.0.84（`cv2` 4.10.0）。
- scikit-learn：1.5.2。
- LightGBM：4.5.0。
- XGBoost：2.1.2。
- PyTorch：2.5.1+cpu。
- SHAP：0.46.0。
- matplotlib：3.9.2。
- pandas：2.2.3。
- NumPy：1.26.4。
- PyYAML：6.0.2。
- CUDA：N/A（`torch.version.cuda` 为 `None`；CPU-only PyTorch 构建）。
- 安装命令：`python -m pip install -r requirements.txt`。
- `requirements.txt` 校验结果或环境锁文件：固定依赖导入检查通过；10/10。
- LaTeX 发行版／版本（若使用）：MiKTeX 25.12；`pdflatex` 4.23 已通过绝对路径验证。安装器已将用户级 MiKTeX bin 目录加入 PATH；当前 Codex 进程未刷新环境变量，重启 Codex 后可直接调用 `pdflatex`。
- 代码提交：
- 完整执行命令：
- 输出路径及文件校验值（如适用）：

## 结果与审计

- LOSO 结果：
- leave-one-scene-out 结果：
- 折外预测和图表路径：
- 相关检验效应量、p 值、置信区间及多重检验校正：
- 失败、限制或未复现项：
- 协议偏离记录：
