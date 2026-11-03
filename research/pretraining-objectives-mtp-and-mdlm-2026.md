# 多 token 预测与掩码扩散语言模型的预训练目标证据（2026）

检索日期：2026-10-30

本记录汇总多 token prediction（MTP）和 masked diffusion language model（MDLM）的原始论文定义与证据边界。MTP 描述的是一种训练目标及其实现路线；MDLM 描述的是一类离散扩散训练/采样方案。二者均不能仅凭名称推定统一的网络架构、推理速度或下游收益。

## Multi-token Prediction

### 来源一：共享主干与并行未来预测头

- 来源：[Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/html/2404.19737v1)，arXiv v1（2024-04-30）。
- 实际阅读范围：摘要、§1、§2 的目标和结构、memory-efficient implementation、inference，以及 §3 实验介绍。
- 来源支持的事实：方法在每个训练位置取共享 Transformer 主干表示，再由彼此独立的输出 head 并行预测若干个不同距离的未来 token；论文的实现使用共享 unembedding。主 next-token head 可以按普通自回归路线推理，其余预测 head 可作为 self-speculative decoding 的候选来源。论文说明顺序执行各未来 head 的 forward/backward 可以降低同时驻留的大词表 logits 与梯度内存。
- 证据等级：原始论文的方法定义和作者实验报告（source-backed）。
- 适用边界：效果受模型规模、预测距离、数据、训练阶段和下游评测影响；自然语言、代码与算法任务并不呈现完全一致的变化。论文报告的采样加速依赖其模型、验证方法和测量设置，不能推成 MTP 通用加速保证。共享主干并行 head 只是一个 MTP 实现，并不定义全部 MTP 架构。

### 来源二：串联 MTP 模块

- 来源：[DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437v2)，arXiv v2（2025-02-18）。
- 实际阅读范围：§2.2 的 MTP modules、training objective 与 inference；§5.4.3 的 MTP acceptance-rate/TPS 评测说明。
- 来源支持的事实：DeepSeek-V3 报告中的 MTP 以多个串联模块逐深度预测未来 token，并维持预测深度之间的因果链；每层模块把前一预测深度的表示与 teacher-forced ground-truth token 的 embedding 结合，再进入 Transformer block。embedding 与输出 head 和主模型共享。各深度以交叉熵形成损失，再跨深度取平均并乘权重，作为额外训练目标。推理时可丢弃 MTP 模块，仅保留主模型，也可将模块用于 speculative decoding。作者在该模型报告中给出第二个预测 token 的接受率和 TPS 结果。
- 证据等级：技术报告中的架构/目标定义及作者特定模型实验（source-backed）。
- 适用边界：串联模块与前述共享主干、独立并行 head 不是同一实现；各损失项的数量、权重和有效目标分母须按具体配方确认。报告中的接受率和 TPS 仅适用于 DeepSeek-V3、其候选验证与部署设置，不支持对其他模型或负载作同等预测。

### 综合边界

“多预测几个未来 token”会使较远距离的目标也向共享表示提供监督，但总损失如何归一、每个预测距离有多少有效 label、各距离权重如何选择，都是配方的一部分。MTP 训练目标和推理阶段接受候选的 speculative decoding 是两个相关但不同的环节；训练中增加未来目标并不意味着推理时一定保留这些 head、接受固定数量 token 或降低端到端延迟。不同预测 head 可能对同一真实 token 提供不同条件上下文下的监督，因此不能只数“覆盖了多少 token”来代替核对目标和分母。

## Masked Diffusion Language Models

### 来源：吸收态掩码扩散

- 来源：[Simple and Effective Masked Diffusion Language Models](https://arxiv.org/html/2406.07524v2)，arXiv v2（2024-11-10）。
- 实际阅读范围：摘要、§2.2、§3.1—§3.5（离散/掩码前向过程、reverse unmasking、SUBS 参数化和加权目标）、§4 采样与 §5 语言模型实验概述。
- 来源支持的事实：论文采用 absorbing-mask forward process，随着噪声步推进，干净 token 有一定概率转为 `[MASK]`；一旦进入 mask 状态，在后续前向时间中保持掩码。模型按带噪序列与噪声条件估计原 token；其训练目标可写为一组 masked-LM 交叉熵的加权组合，并由离散扩散的变分似然界导出。生成通过 reverse unmasking 进行，论文还研究 semi-autoregressive sampler。
- 证据等级：原始论文方法、变分目标和作者 benchmark 报告（source-backed）。
- 适用边界：这些机制属于该论文定义的 masked diffusion 路线，不应外推到所有离散扩散噪声过程。损失包含噪声时间步及其权重，不能与普通 causal token-mean loss 直接比裸数值。论文的语言建模结果绑定其模型、语料、训练 recipe、采样方式和评估口径；能够在一轮中处理多个位置不等于端到端延迟必然更低。

## 未解决或不宜泛化的部分

- MTP 的最佳预测距离数、head/module 结构、loss 权重、尾部 mask 和训练阶段选择没有跨模型、跨任务的统一配方。
- MTP 的辅助训练收益、候选接受率和推理收益需要分别测量；验证开销、回滚、批量和服务实现会改变实际结果。
- MDLM 的噪声过程、目标权重、条件架构与 reverse sampler 是相互关联的设计变量；不能把某一个 mask-loss 公式当成所有离散扩散模型的通用目标。
- 本记录中的机制和实验结果属于原论文的方法定义与作者报告（source-backed）；这些来源本身不能支持跨实现的普遍性能结论。
