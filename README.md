# SDGL: Selective Disentangled Gradient Learning


SDGL 在 **DGL（Disentangled Gradient Learning, ICCV 2025）** 的开源代码基础上，
提出一种 **选择性梯度截断（Selective Gradient Truncation, SDGL）** 方法，
在 CREMA-D 数据集上验证改进效果，并对方法的收益边界与局限性做了客观分析。

> **致谢**：本项目基于原作者 [shicaiwei](mailto:shicaiwei@std.uestc.edu.cn) 的
> DGL 开源代码（https://github.com/ICCV2025-GDL）改进而来，核心数据预处理、骨干网络、
> 融合模块等大量代码沿用原仓库。在此向原作者表示感谢。原论文：
> *Boosting Multimodal Learning via Disentangled Gradient Learning* ([arXiv:2507.10213](https://arxiv.org/pdf/2507.10213))。

---

## 1. 背景与动机

DGL 通过**完全截断**融合损失到单模态编码器的梯度，来缓解多模态联合训练中的梯度冲突。
但完全截断也可能丢失一部分**与单模态优化方向一致、对特征学习有益**的融合梯度。

**SDGL 的核心想法**：不再"一刀切"地截断，而是用特征级余弦相似度作为门控，
**选择性保留**方向一致的融合梯度：

$$
g_m^{Final} = \alpha \, g_m^{Uni} + \beta_m \, g_m^{Multi}, \qquad
\beta_m = \begin{cases} \lambda_m, & s_m > \tau_m \\ 0, & s_m \le \tau_m \end{cases}
$$

其中 $s_m$ 是单模态梯度与融合梯度的余弦相似度。DGL 相当于 $\beta_m \equiv 0$ 的特例。

本项目进一步将对称门控（两模态共享 $\tau,\lambda$）解耦为**非对称门控**
（$\tau_a,\lambda_a,\tau_v,\lambda_v$ 独立），以适配不同模态的收敛特性。

---

## 2. 主要改动（相对原 DGL 代码）

| 文件 | 改动 |
|---|---|
| `main_dgl.py` | 新增 `train_epoch_sdgl()`：手动用 `autograd.grad` 计算特征级梯度，按相似度门控合并后赋值给参数 |
| `models/basic_model.py` | 新增 `AVClassifier_SDGL`：前向额外返回音频/视觉特征 $z_a, z_v$ |
| `models/fusion_modules.py` | 新增 `ConcatFusion_SDGL`：去掉原 DGL 的 `.detach()`，改为由外部门控控制梯度流 |

DGL 原始训练逻辑（`train_epoch`、`AVClassifier_DGL`、`ConcatFusion_DGL`）保持不变，作为基线对照。

---

## 3. 训练环境

- Ubuntu 20.04 
- CUDA 11.3，PyTorch 1.11.0，Python 3.8
- 主干网络 ResNet18
- RTX 4090D
- **说明**：实验结果显示，不同硬件或训练环境可能导致结果存在偏差。

---

## 4. 数据准备

采用 CREMA-D 数据集，数据处理沿用原仓库流程：

1. 下载 [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) 原始数据。
2. 抽帧：`python data/CREMAD/video_preprocessing.py`（按原仓库说明）。
3. 将处理后的数据放入 `train_test_data/` 目录（或建立软链接）。
> 原作者已经准备好处理后的数据集，分别是如下链接[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D),[Kinetics-Sounds](https://github.com/cvdfoundation/kinetics-dataset)，[VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/)


