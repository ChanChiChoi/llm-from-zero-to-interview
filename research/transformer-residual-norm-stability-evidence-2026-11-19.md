# Transformer residual、归一化与训练稳定性证据

检索与原文核验日期：2026-11-19。范围：深层网络的 residual learning、LayerNorm 与 RMSNorm 定义、Transformer 中的 norm placement、warmup 相关证据，以及 residual scaling / initialization 的代表性方案。教学组织资料只用于观察讲解次序，不用于宣称学习效果。

## 检索范围

- 学科检索词：`On Layer Normalization in the Transformer Architecture warmup Pre-LN Post-LN Xiong 2020`；`DeepNet Scaling Transformers to 1000 Layers DeepNorm residual scaling initialization`；`Understanding the Difficulty of Training Transformers Admin initialization residual branch`；`RMSNorm Root Mean Square Layer Normalization original paper re-centering re-scaling`；`Layer Normalization Ba Kiros Hinton single training case`；`NormFormer Improved Transformer Pretraining with Extra Normalization`；`Peri-LN Revisiting Normalization Layer in the Transformer Architecture Kim ICML 2025`；`Stability of Transformers under Layer Normalization Kan 2026 Pre-LN`。
- 教学组织检索词：`Stanford CS224N lecture notes Transformer residual connections layer normalization Pre-LN Post-LN`。
- 资料类型：原始论文、EMNLP/NeurIPS/ICML 论文页面、arXiv 正文，以及 Stanford CS224N 课程讲义草稿。
- 未覆盖：不同训练框架下的 kernel 速度、特定 frontier 模型未公开的 norm/initialization recipe、目标硬件上的稳定性实测。没有运行论文代码或训练模型。

## 来源卡与证据边界

| 来源 | 原文阅读位置 | 可支持 | 不可推出 |
| --- | --- | --- | --- |
| He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), 2015 | §§3.1–3.2 | 把目标映射写成 $\mathcal{F}(x)+x$ ；作者提出残差映射可能更容易优化，并用图像分类实验验证深残差网络 | 图像任务结果本身证明 Transformer 的 warmup 或 norm placement 结论 |
| Ba et al., [Layer Normalization](https://arxiv.org/abs/1607.06450), 2016 | 摘要、原文 §2 与结论 | 从单个 training case 的层内神经元输入计算均值和方差；使用可学习 gain/bias；训练与测试计算一致 | Transformer 框架的归一化轴只凭这篇通用论文即可确定 |
| Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017 | §3.1 | 原始 Transformer 在每个 attention/FFN 子层外先做 residual add，再做 LayerNorm | 所有现代 Transformer 均保留同一 norm placement |
| Zhang & Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467), 2019 | §4 与论文摘要 | RMSNorm 按 RMS 缩放，不减均值；作者假设 re-centering invariance 可省略，并讨论较低计算开销 | 所有任务、框架和硬件上都比 LayerNorm 更快或效果相同 |
| Xiong et al., [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745), ICML 2020 | §§3.1–3.3、实验讨论 | 在其初始化/mean-field 分析中，Post-LN 输出附近部分参数期望梯度较大；Pre-LN 初始化梯度较平稳；其 IWSLT14、WMT14 与 BERT 实验显示指定 Pre-LN 设置可去掉 warmup 并取得可比结果 | 任意深度、数据、optimizer、精度或 LLM recipe 都可以安全删除 warmup |
| Liu et al., [Understanding the Difficulty of Training Transformers](https://aclanthology.org/2020.emnlp-main.463/), EMNLP 2020 | 摘要、§1、§3 | 作者认为梯度不平衡不是训练不稳定的根因；较强 residual-branch dependency 会放大参数扰动，较弱依赖又会限制模型潜力；Admin 通过 initialization 控制早期依赖，并在指定英德/英法翻译设置中验证 | 该分析说明单看 gradient norm 即可诊断稳定性，或 Admin 在所有模型中都优于其他初始化 |
| Shleifer et al., [NormFormer](https://arxiv.org/abs/2110.09456), ICLR 2022 | 摘要、§§2–4 | 该方案针对 Pre-LN 梯度幅度层间失配，加入 attention 后 LayerNorm、head-wise scaling 和 FFN 第一线性层后 LayerNorm；论文在 125M–2.7B causal/masked LM 及其等训练时间设置中报告改进 | 额外 norm 在所有模型尺度或训练预算下都会提高质量或速度 |
| Wang et al., [DeepNet: Scaling Transformers to 1,000 Layers](https://arxiv.org/abs/2203.00555), 2022 | §§2–4、§6 | DeepNorm 在 residual 加法和初始化中使用共同设计的缩放；作者在多语机器翻译中训练到 1,000 层，并在 200 层大模型上报告指定 BLEU 结果 | 该深度/成绩可直接迁移到 decoder-only 预训练或任意硬件 |
| Touvron et al., [LLaMA](https://arxiv.org/abs/2302.13971), 2023 | §2.2 | LLaMA 报告对子层输入做 normalization，并使用 RMSNorm | LLaMA 设计可单独证明 RMSNorm 或 Pre-LN 的因果收益 |
| Kim et al., [Peri-LN: Revisiting Normalization Layer in the Transformer Architecture](https://proceedings.mlr.press/v267/kim25u.html), ICML 2025 | 原文 §§1、3.3、4.1–4.3 | 作者把已有的子层输入/输出外围归一化做法命名为 Peri-LN；比较 400M、1.5B、3.2B 模型，每个模型训练 30B tokens、5 个随机种子，使用 DCLM-baseline，并报告方差、梯度与收敛方面的结果 | 不能将这些特定数据、训练预算和模型规模上的作者报告外推到所有架构；也不应把 Peri-LN 术语命名误写成所有外围归一化实现的首创 |
| Kan et al., [Stability of Transformers under Layer Normalization](https://arxiv.org/abs/2510.09904), v2 2026-08-07 | §§I–III、V–VI | 在离散时间最优控制框架中后续分析 Post-/Pre-/Peri-LN 和 residual-step scaling；实验使用 GPT-2 家族（约 100M–1.5B）、OpenWebText 与 H200 | 预印本的理论假设与 GPT-2 实验不证明所有生产模型都应改用 Peri-LN，也不等于 Kim et al. 的 2025 实验 |
| Stanford CS224N, [Note 10: Self-Attention & Transformers](https://cs224n.stanford.edu/readings/cs224n-self-attention-transformers-2023_draft.pdf), 2023 draft | §§3.2–3.3 | 先界定 LayerNorm 的统计对象和 Transformer 中的 token/hidden 轴，再讲 residual 与 Pre/Post-Norm 方程的一种课程组织方式 | 草稿次序是唯一最佳教学方案，或已被学习效果研究证明 |

## 可复用的技术关系

1. Residual block 把输入与分支输出相加： $y=x+F(x)$ 。在局部微分中 Jacobian 含 identity 项；这提供直接传播分量，但不能保证整个深层网络的激活、梯度或参数更新始终有界。
2. 对标准 Transformer hidden vector $x\in\mathbb{R}^{D}$ ，LayerNorm 在该 token 的 hidden 维计算 $\mu=\frac{1}{D}\sum_i x_i$ 和 $\sigma^2=\frac{1}{D}\sum_i(x_i-\mu)^2$ ，再应用可学习的逐维 gain/bias。batch 与序列中的其他 token 不参与这组统计。
3. RMSNorm 使用 $\mathrm{RMS}(x)=\sqrt{\frac{1}{D}\sum_i x_i^2+\epsilon}$ ，将 $x$ 除以 RMS 后乘逐维 gain；标准形式不减均值。具体实现的 gain、bias 和 epsilon 约定须看该实现。
4. Post-LN 与 Pre-LN 既改变子层看到的输入，也改变 residual 路径穿过 normalization 的位置。对 $y=x+F(\mathrm{Norm}(x))$ ，局部 Jacobian 有显式 $I$ 项；对 $y=\mathrm{Norm}(x+F(x))$ ，归一化位于加法结果之后。该局部式不等同于整模型训练必然稳定。
5. Xiong et al. 的 warmup 论证针对初始化处梯度与其论文实验；“Pre-LN 在论文设置中无需 warmup”不等于现代训练中 warmup 可删除。学习率、初始化、optimizer、batch、深度、精度、数据和架构需一起核验。
6. Admin、NormFormer、DeepNorm 与 Peri-LN 的着力点不同：residual-branch dependency 初始化、Pre-LN 内部梯度幅度、深层 residual/init 缩放、norm placement 与 residual step。它们对应各自论文实验范围，不能按普遍排行榜排序。

## 未验证边界

- 所有结果均引用作者报告；没有运行代码、重新训练或测量硬件吞吐。
- RMSNorm 的运算路径较简洁不保证目标框架上的 wall-clock 更快；fused kernel、dtype、shape 与设备会影响速度。
- Peri-LN 由 Kim et al. 2025 年 ICML 论文命名和系统研究；该文指出一些开源模型此前已采用相近的外围归一化放置。Kan et al. 2026 v2 是独立的后续 arXiv 预印本，不是 Peri-LN 的提出论文；其最优控制理论结论依赖建模条件，GPT-2 实验也不是跨架构、跨数据的普遍结论。
- 训练不稳定的一个信号不足以识别原因。Activation/gradient/update trace 应绑定模型、初始化、optimizer、learning-rate schedule、precision、数据和训练阶段；未固定这些条件的 toy 阈值不能作为真实训练门禁。
