# 序列架构的信息流：递归状态、卷积感受野与可寻址历史

研究日期：2026-11-23。用途：第二十一册第三十章《SSM、RNN、CNN、Attention 的信息流对比》的范围确认、关键机制核验和解释路径设计。第 29 章承担固定线性时不变 SSM 的基础数学；第 31 章展开 HiPPO/S4；第 32 章展开 Mamba 输入选择性与 scan；第 42 章负责替代架构的复杂度、表达力、硬件和生态选型；第 43 章负责 Attention 与 SSM 混合架构。

## 研究问题与检索

本章需要解释四类序列层怎样让当前位置获得历史信息：历史以什么形式保留、哪些过去位置存在计算路径、训练时各位置怎样计算，以及自回归生成时每步要携带什么状态。比较范围限定为因果序列预测下的 RNN、因果时序 CNN、固定参数 LTI SSM 与 causal self-attention；不扩展到 retrieval 系统、完整服务治理、端到端选型 benchmark 或混合模型接口。

检索词包括 `Attention Is All You Need path length recurrent convolutional layers`、`TCN causal convolution receptive field dilation sequence modeling`、`RNN unrolled recurrent state sequence modeling notes`、`S4 convolution recurrent representation training inference`、`Annotated Transformer sequence positions parallel` 和 `Annotated S4 CNN RNN training inference`。首轮并行查阅 Transformer 与 TCN 原论文、SSM 原论文和教学性材料；再核对其比较的假设及适用边界。未将特定论文的任务得分、训练时长或硬件速度写成架构的一般性质。

检索于 2026-11-23 通过网页搜索发现资料，并在 arXiv、NeurIPS Proceedings、Dive into Deep Learning、Harvard NLP 与《The Annotated S4》页面直接核对原文。检索词以英文为主，时间范围覆盖相关基础论文（2017—2022）及截至检索日可访问的教学材料；筛选限于可直接说明四种信息流机制、路径或教学解释的来源，不扫描模型产品/服务性能更新，也不把论文 benchmark 当作普遍架构比较证据。

## 核心发现

1. **比较必须先固定可见性。** 本章默认因果预测：时刻 $t$ 不能读取未来输入。时序 CNN 是否因果取决于卷积与 padding 设计；attention 是否能访问某个位置取决于 mask 和该位置是否仍在上下文中。原始 Transformer 论文的路径和复杂度表是针对其明确的层定义与符号假设，不直接决定现代系统的墙钟速度。
2. **RNN 与固定 LTI SSM 都通过状态递归传递历史，但不是同一种模型。** 常见非线性 RNN 用前一时刻的隐状态和当前输入计算新状态；固定 LTI SSM 使用线性状态转移。对固定 LTI SSM，在适当初态条件下可展开为同一个卷积核作用于整段输入。这个等价解释的是算子，不意味着所有 SSM 都是固定卷积，也不保证任意硬件上的速度结论。
3. **因果 CNN 的可访问历史受结构感受野约束。** 单层只连接当前层的有限位置；层叠与 dilation 扩大可达范围。因果卷积可以沿序列位置并行计算已知输入的各个输出，但有限层和有限感受野时，范围外输入与该输出之间没有路径。序列生成仍要遵守生成顺序，不能把训练并行直接等同于自回归解码并行。
4. **因果 self-attention 为每个 query 提供显式的、按内容加权的可见历史表示。** 若 mask 允许，任意过去位置可以在一层内成为直接读入对象；这只说明图上有边，并不保证分数选择正确、信息跨层不丢或答案可靠。显式逐位置 K/V 访问与递归状态压缩是不同的信息路径。
5. **训练算图、算子形式和生成缓存应分开比较。** RNN 的通用递归更新有跨时间依赖；完整序列卷积和 attention 可并行处理位置。固定 LTI SSM 可选递归或卷积/scan 路径；输入依赖的 selective SSM 需要另外的 scan 计算。本章只讲结构性区别，不把“更并行”改写成普遍更快。
6. **生成期携带的状态也不同。** RNN 与按递归方式执行的 SSM 携带每层固定形状的状态；因果 CNN 可缓存各层后续卷积所需的有限历史激活；attention 通常缓存已处理位置的 K/V，完整上下文下随历史长度增长。实际容量还取决于层数、维数、head 数、窗口和实现，不在本章给通用显存数值。

## 来源与使用范围

| 来源 | 阅读位置 | 支持内容 | 限制与本章用法 |
| --- | --- | --- | --- |
| Ashish Vaswani 等，[Attention Is All You Need](https://arxiv.org/abs/1706.03762)，NeurIPS 2017 | §4 Why Self-Attention，Table 1 | 论文定义下 self-attention、recurrent、convolutional layer 的计算复杂度、顺序操作数和长程路径长度对照；解释因果与位置路径 | 用于架构信息流与依赖路径的原始依据。表中的数值复杂度依赖论文符号和层假设，不外推到现代 kernel、serving 吞吐或所有 CNN/attention 变体。 |
| Shaojie Bai、J. Zico Kolter、Vladlen Koltun，[An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271)，arXiv:1803.01271 | §3.2 Causal Convolutions、§3.3 Dilated Convolutions、§3.5 Discussion、Figure 1 | 因果卷积的可见输入边界、dilation 的历史覆盖扩展，以及 TCN 在完整已知序列上的并行计算 | 用于解释因果性、感受野与位置并行。论文的 TCN/LSTM/GRU 实验只支持作者的任务、模型和实现设置，不转写为 CNN 普遍优于 RNN 的结论。 |
| Albert Gu、Karan Goel、Christopher Ré，[Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396)，ICLR 2022 | §2.3 Recurrent Representation、§2.4 Convolutional Representation | 离散固定 LTI SSM 的递归与卷积表达、初态条件、卷积核定义与计算非平凡性 | 作为第 29 章之后的简短承接；S4 的结构化参数化和 kernel 计算留给第 31 章。不能把该结论泛化到输入依赖的状态更新。 |
| Dive into Deep Learning，[Recurrent Neural Networks](https://d2l.ai/chapter_recurrent-neural-networks/)，在线教材 1.0.3 | 第 9 章开篇的循环图展开说明 | 以跨时间共享参数、相邻状态边解释 RNN 的递归状态 | 作为面向读者的概念核对与教学组织材料，不替代具体 RNN 变体的机制说明。 |
| Harvard NLP，[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) | self-attention 的 encoder/decoder 可见位置和实现串讲 | 说明 query/key/value 读写角色以及因果 decoder 在当前位置可见的前缀 | 教学性实现说明；方法事实仍回到 Transformer 原论文，代码实现不作为现代 Transformer 的唯一范式。 |
| Sasha Rush、Sidd Karamcheti，[The Annotated S4](https://srush.github.io/annotated-s4/) | Part 1 的递归与卷积表示；Part 3 的 S4 CNN/RNN Layer | 将 S4 的全序列 convolution training view 与递归 decoding view 对齐，帮助解释同一层的两种计算路径 | 特定教程实现用于教学；不能把其代码、实验或“CNN/RNN”命名直接推广到所有 SSM。 |
