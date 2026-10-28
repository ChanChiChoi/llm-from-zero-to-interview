# Transformer 残差与归一化：事实和证据

检索日期：2026-10-28。

## LayerNorm 的动机和计算范围

- Ba、Kiros、Hinton 的原始论文《Layer Normalization》于 2016-07-21 提交 arXiv。摘要对比 BatchNorm 与 LayerNorm：BatchNorm 从一个 mini-batch 中各样本的神经元输入统计均值和方差；LayerNorm 则从单个训练样本的该层输入统计均值和方差。论文还说明可学习的 bias/gain 在归一化后应用，并指出循环网络可逐时间步单独计算统计量，训练与测试阶段采用相同计算。来源：[arXiv 摘要](https://arxiv.org/abs/1607.06450)，实际阅读摘要。证据等级：原始论文摘要中的方法定义。边界：不单独证明 LayerNorm 在所有网络上都优于 BatchNorm，也不描述现代 Transformer 的每种实现配置。
- PyTorch 2.14 `torch.nn.LayerNorm` 文档规定，统计量沿 `normalized_shape` 指定的末尾维度计算；方差使用 biased estimator，即 `correction=0`；接口默认 `eps=1e-5`，并为指定 shape 的每个元素提供可学习 weight/bias。来源：[PyTorch 2.14 LayerNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html)，读取定义、参数和 NLP 示例。证据等级：固定版本官方 API 文档。边界：只说明该版本公开接口，不外推到其他框架、kernel 或模型配置。

## RMSNorm 的假设

- Zhang 与 Sennrich 的《Root Mean Square Layer Normalization》于 2019-10-16 提交 arXiv，论文发表于 NeurIPS 2019。摘要提出 LayerNorm 的重新中心化不变性可能并非必要，并以均方根重标定来省略去均值步骤；论文作者报告若干任务和网络架构上的实验结果。来源：[arXiv 摘要](https://arxiv.org/abs/1910.07467)，实际阅读摘要。证据等级：原始论文方法假设和作者实验报告。边界：实验报告不证明所有模型、任务或参数预算下 RMSNorm 都优于 LayerNorm；摘要中的运行时间数字不能脱离模型与实验条件外推。
- PyTorch 2.14 `torch.nn.RMSNorm` 文档给出 RMS 分母为平方均值加 epsilon 后开方；其 `normalized_shape` 指定末尾归一化维度。文档中 `eps=None` 会按 opmath 计算类型选取 machine epsilon；fp16/bf16/fp32 输入使用 float32 的 machine epsilon，fp64 使用 float64 的 machine epsilon。来源：[PyTorch 2.14 RMSNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.RMSNorm.html)，读取公式、normalized_shape 与 eps 参数。证据等级：固定版本官方 API 文档。边界：这是 PyTorch v2.14 的默认接口行为，其他版本和实现可以有不同设置。

## Pre-LN 与 Post-LN 的分析边界

- Vaswani 等人的原始 Transformer 论文 v7 §3 描述子层输出与残差相加后执行 LayerNorm。来源：[论文 HTML §3.1](https://arxiv.org/html/1706.03762v7#S3.SS1)。证据等级：原始论文架构定义。边界：不能据此推断 decoder-only Transformer 或现代模型都逐项采用原始架构。
- Xiong 等人的《On Layer Normalization in the Transformer Architecture》v2 于 2020-06-29 修订，发表于 ICML 2020。摘要报告其 mean-field 理论分析中，原始 Post-LN 架构在初始化时靠近输出层的预期梯度较大，而其分析的 Pre-LN 梯度较平稳；作者报告 Pre-LN 在其一系列应用实验中可不使用 warm-up 并达到与 baseline 可比的结果。来源：[arXiv v2 摘要](https://arxiv.org/abs/2002.04745v2)，实际阅读摘要和版本历史。证据等级：原始论文的理论分析与作者实验报告。边界：结论受该论文架构、初始化、mean-field 假设和实验配置限制，不构成所有模型训练都应使用 Pre-LN 或取消 warm-up 的规则。

## 未验证边界

- 未运行或测量任何真实 Transformer；未比较 LayerNorm/RMSNorm kernel 的吞吐、低精度误差或训练效果。
- 正文四维向量、残差输入/更新和仿射参数均为教学输入；不代表模型观测值。
- 对具体模型做复现或排障时，仍需确认其 Norm 维度、epsilon、仿射参数、精度路径、残差缩放和放置顺序。
