# 第 54 章 LLaMA 系列：RoPE、RMSNorm、SwiGLU 与开源 LLM 基线

## 54.1 原始 LLaMA 建立了怎样的稠密基线

2023 年的 LLaMA 论文提出一组从 $7\mathrm{B}$ 到 $65\mathrm{B}$ 参数的基础语言模型，重点是让较小模型通过充分训练达到有竞争力的结果，并使用公开可获得的数据集。论文将模型提供给研究社区，使更多研究者能够检查权重、复现实验或在此基础上继续研究。[原始 LLaMA 论文](https://arxiv.org/abs/2302.13971)

理解这段历史时，先把“架构配方”和“生态影响”分开。LLaMA 没有提出 decoder-only Transformer，也没有发明 RoPE、RMSNorm 或 SwiGLU；它的影响在于把几项已有设计组合进一套表现有竞争力、后来容易被社区复用的稠密基线。原始配方使用预归一化、RMSNorm、SwiGLU 和 RoPE，仍然沿着标准自回归 Transformer 的注意力与前馈层组织信息。

“开放”也要具体到开放了什么。原始论文说明模型提供给研究社区，并使用公开数据集；这不等同于训练语料、完整清洗规则、全部训练产物和无限制使用权都与权重同时公开。后续版本的权重获取、许可和用途条款也按版本变化。把 LLaMA 简称为开源基线时，应避免让读者误以为完整训练因果链都可复现。

## 54.2 三个常见组件改变的是不同路径

LLaMA 论文介绍这些组件时明确援引了此前工作：预归一化与 GPT-3 的做法相关，SwiGLU 已用于 PaLM，RoPE 则沿用 GPT-Neo 等模型探索的旋转位置表示。因此，讲解重点应是它们在 block 中承担什么作用，以及为何不能只凭组件名称断定模型性能，而不是把它们描述成 LLaMA 首创。

| 组件 | 所在位置 | 改变的路径 | 需要保留的边界 |
| --- | --- | --- | --- |
| Pre-Norm 与 RMSNorm | 注意力和 FFN 子层输入侧 | 先规范化子层读取的表示；RMSNorm 不做均值中心化 | 训练稳定性还受初始化、优化器和残差设计影响；细节见[第六章](06-residual-layernorm-rmsnorm-prenorm与训练稳定性.md) |
| RoPE | 每层注意力中的 query/key 位置表示 | 让注意力点积携带由位置差异产生的相位关系 | 不等于无限上下文或长距离可靠检索；位置映射与外推见[第七章](07-position-encoding总览-sinusoidal-learned-relative-rope-alibi.md)和[第十八章](18-RoPE与长上下文位置外推.md) |
| SwiGLU | Transformer block 的前馈子层 | 用门控支路调节另一支路的信息通过 | 与普通 FFN 的比较受中间维度和训练预算影响；不构成单模块必胜保证 |

这三个选择分属归一化、位置表示和前馈变换，合在一起也不会自动成为一条不可拆分的“LLaMA 定律”。例如，改变 RoPE 频率或上下文训练长度，影响的是位置使用范围；改变 SwiGLU 中间维度，影响参数量和前馈计算；改成 GQA，则改变 query 与 KV heads 的共享关系和 KV 状态规模。注意力分组的定义见[第四章](04-mha-mqa-gqa-mla的演进与trade-off.md)，本章只追踪它何时进入 LLaMA 系列。

## 54.3 LLaMA 系列沿用基线，也逐版调整

“LLaMA 架构”不是跨版本不变的一套配置。论文和官方模型目录给出的几个阶段，可以概括为：

| 版本 | 公开模型范围 | 架构演进 | 适用边界 |
| --- | --- | --- | --- |
| LLaMA（2023） | 标称 $7\mathrm{B}$、$13\mathrm{B}$、$33\mathrm{B}$、$65\mathrm{B}$ | 稠密 decoder-only；使用 Pre-Norm/RMSNorm、SwiGLU 与 RoPE | 代表原始研究配方，不代表所有后续版本 |
| Llama 2（2023） | 发布 $7\mathrm{B}$、$13\mathrm{B}$、$70\mathrm{B}$ | 继续采用标准 Transformer 主干；上下文从 $2\mathrm{k}$ 扩到 $4\mathrm{k}$，并在较大模型中采用 GQA | 论文把数据、上下文和 GQA 等多项变化一起引入，不能将整体结果归因于某一项 |
| Llama 3 / 3.1（2024） | Llama 3 初发为 $8\mathrm{B}$、$70\mathrm{B}$；技术报告也记录后续的 $405\mathrm{B}$ 模型；3.1 官方目录列出 $8\mathrm{B}$、$70\mathrm{B}$、$405\mathrm{B}$ | 仍采用稠密 decoder-only 与 GQA；3.1 将官方 context length 扩至 $128\mathrm{k}$ | 训练长度、配置字段和任务有效长度不是同一概念；长上下文能力须按任务测量 |
| Llama 3.2 / 3.3 | [官方目录](https://github.com/meta-llama/llama-models)列出小型文本模型、视觉版本和 $70\mathrm{B}$ 指令模型 | 家族扩展到不同尺寸和模态用途 | 具体结构与接口要回到对应模型卡，不能从系列名推断 |

Llama 2 论文称其将上下文加倍，并采用 GQA 改善较大模型的推理扩展性；论文表中发布的 $70\mathrm{B}$ 版本使用 GQA。Llama 3 技术报告将最大的 $405\mathrm{B}$ 模型描述为稠密 Transformer；Meta 对 Llama 3.1 的发布说明也称采用标准 decoder-only Transformer，而没有选择 MoE，以保持训练稳定性。[Llama 2 论文](https://arxiv.org/abs/2307.09288)、[Llama 3 技术报告](https://arxiv.org/abs/2407.21783)、[Llama 3.1 发布说明](https://ai.meta.com/blog/meta-llama-3-1)

这条演进说明，基线的延续并不意味着架构停滞：同一稠密主干可以逐步调整 KV 头组织和上下文训练，而家族也可以增加不同尺寸和模态用途的分支。数据配比、tokenizer、预训练规模和后训练如何成熟由[第五十七章](57-Qwen-LLaMA3-Gemma训练成熟化.md)主讲；本章只保留足以理解架构路径的版本差异。逐个 checkpoint 的层数、头数、词表和位置字段则见[第十七章](17-LLaMA-GPT-Mistral架构对比.md)。

## 54.4 Llama 4 已超出经典稠密配方

截至 2026-11-24，Meta 公开的 Llama 模型目录将 Llama 4 列为后续家族版本，并列出 Scout 与 Maverick 两个模型及其上下文声明。[Meta Llama 模型目录](https://github.com/meta-llama/llama-models)。官方 Llama 4 模型卡把这两个模型描述为使用 MoE 的自回归模型，并采用 early fusion 支持原生多模态：Scout 有 $16$ 个专家、总参数 $109\mathrm{B}$、激活参数 $17\mathrm{B}$；Maverick 有 $128$ 个专家、总参数 $400\mathrm{B}$、激活参数 $17\mathrm{B}$。模型卡列出的 context length 分别为 $10\mathrm{M}$ 和 $1\mathrm{M}$。[Llama 4 官方模型卡](https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md)

这些字段说明 Llama 4 已不能被概括为“稠密 decoder + RoPE/RMSNorm/SwiGLU”的原始基线。总参数与激活参数也要分开看：MoE 每个 token 只经过被路由的部分专家，不能用总参数量直接代替每 token 激活计算；对应成本与通信边界见[第二十一章](21-MoE路线Switch-GShard-Mixtral-DeepSeek.md)和[第二十二章](22-Dense与MoE训练服务trade-off.md)。模型卡中的百万级 context length 是发布方字段，不证明任意任务都能在该长度上可靠检索，也不表示延迟、显存和评测质量与短上下文相同。

Llama 4 的原生多模态输入与 early-fusion 披露属于另一个架构分支，不能把该信息倒推到 LLaMA、Llama 2 或 Llama 3 的文本模型。多模态融合方式的完整概念见[第十五册统一多模态模型章节](../../book-15-multimodal-generative-models/chapters/10-统一多模态模型.md)。

## 54.5 把 LLaMA 当作可追溯的基线

使用 LLaMA 家族作比较起点时，先回答几个具体问题：使用的是哪个版本、哪个尺寸、基础模型还是指令版本？它是稠密模型还是 MoE？位置、归一化、FFN 和 attention 字段来自论文、模型卡还是某个固定配置？所谓上下文长度是发布字段、训练序列长度还是经过任务验证的有效范围？权重对应的许可和可用范围是什么？

这组问题把“系列基线”变成可复查对象。遇到没有公开的信息，应保留未知；不能从 LLaMA 品牌名、社区实现默认值或某个版本的结构补齐另一个版本。可下载的权重有助于检查具体 checkpoint，但不会自动公开训练语料、清洗细节、未发布实验或完整服务实现。许可也应按版本核对，而不能把所有 LLaMA 发布归成相同条件。

因此，LLaMA 的架构遗产不是一张永远固定的组件清单，而是一条可观察的演进线：原始稠密模型把几项已有高效设计组合成开源研究基线；Llama 2 和 3 在主干上逐步扩展上下文与 KV 组织；Llama 4 则进入 MoE 和原生多模态。理解这条线时，先锁定版本，再区分架构事实、训练选择、权重开放程度和模型卡声明。RMSNorm、RoPE、GQA、MoE 和多模态融合的机制分别回链对应专题，不把家族名当作解释。
