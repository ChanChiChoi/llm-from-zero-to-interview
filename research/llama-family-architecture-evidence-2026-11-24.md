# LLaMA 家族架构基线与版本演进研究记录

## 研究问题与范围

核对原始 LLaMA 论文中稠密 decoder 配方的来源与组件位置，梳理 Llama 2、Llama 3/3.1 对该基线的延续和调整，并检查 Llama 4 如何超出经典稠密 recipe。官方模型目录在 2026-11-24 检索，用于限定当前公开版本列表；本记录不把模型卡的最大上下文字段解释为任务级可靠性，也不尝试完整盘点每个版本的训练或服务实现。

检索词包括：LLaMA architecture RMSNorm RoPE SwiGLU original paper; Llama 2 context GQA architecture changes; Llama 3 405B dense transformer GQA; Llama 3.1 standard decoder-only not MoE context 128K; Meta Llama 4 Scout Maverick experts active total context model card; Meta official Llama model list current license.

## 主要发现

1. **原始 LLaMA 组合的是已有架构改进。** Touvron 等人的 2023 年论文介绍 $7\mathrm{B}$ 至 $65\mathrm{B}$ 基础模型，使用公开可获得数据集。架构部分称子层输入采用 Pre-Norm/RMSNorm，并明确将其联系到 GPT-3 预归一化、将 SwiGLU 联系到 PaLM、将 RoPE 联系到 GPT-Neo 等已有工作。因此证据支持“将既有组件组合成可复用的公开研究基线”，不支持“LLaMA 发明这些组件”。
2. **Llama 2 延续稠密主干并加入上下文与 KV 组织调整。** 论文报告发布 $7\mathrm{B}$、$13\mathrm{B}$ 和 $70\mathrm{B}$ 模型；将上下文长度从 $2\mathrm{k}$ 提至 $4\mathrm{k}$，采用新数据组合，并在较大模型使用 GQA。论文表中的发布版 $70\mathrm{B}$ 使用 GQA；作者将它用于改善较大模型的推理扩展性。由于同时调整了数据、训练 token 和上下文，整体性能不能归因于单独的 GQA。
3. **Llama 3/3.1 保持稠密 Transformer 路线。** Llama 3 技术报告摘要描述 $405\mathrm{B}$ 稠密 Transformer 和最高 $128\mathrm{k}$ context；Meta 的 Llama 3.1 官方发布页称采用标准 decoder-only 架构而选择不采用 MoE，发布的 $8\mathrm{B}$、$70\mathrm{B}$、$405\mathrm{B}$ 版本把上下文声明扩展至 $128\mathrm{k}$。技术报告、发布页和模型卡是不同版本/字段来源，引用时应保留各自口径。
4. **Llama 3.2/3.3 扩展了家族用途与尺寸。** Meta 当前官方模型目录列出 $1\mathrm{B}$、$3\mathrm{B}$ 文本模型、$11\mathrm{B}$/$90\mathrm{B}$ Vision，以及 Llama 3.3 $70\mathrm{B}$ 等版本。仅凭系列名无法推出每个分支的 block 和模态路径，应回到该版本模型卡。
5. **Llama 4 已改变经典基线的架构类别。** 官方 Llama 4 模型卡称 Scout 与 Maverick 是带 early fusion 的原生多模态 MoE 自回归模型。Scout 列出 $16$ 个专家、$109\mathrm{B}$ 总参数、$17\mathrm{B}$ 激活参数和 $10\mathrm{M}$ context；Maverick 列出 $128$ 个专家、$400\mathrm{B}$ 总参数、$17\mathrm{B}$ 激活参数和 $1\mathrm{M}$ context。这些是模型卡公布字段；不能将最大 context 宣称外推为所有任务的可靠检索长度、延迟或显存保证。
6. **家族索引和许可要按版本核对。** Meta 的公开模型目录目前列出 Llama 2、3、3.1、3.2、3.3 和 4，各自关联不同模型卡、Use Policy 和 License；下载指引也要求接受相应许可。可获得权重不自动等于训练数据、全部清洗步骤和使用权利都无限开放。
7. **教学分工应区分架构历史与具体配置读取。** 第五十四章适合呈现家族层面的架构演进；第二十一册第十七章已按 GPT、LLaMA 和 Mistral 的具体公开版本解释配置字段，第五十七章讲开源训练数据和工程成熟度。RMSNorm、RoPE、GQA、MoE 和多模态融合的完整机制应回链专题章节。

## 来源与证据边界

| 来源 | 类型与阅读位置 | 支持的内容 | 不支持的外推 |
| --- | --- | --- | --- |
| [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)（Touvron 等，2023） | 原始论文；摘要、§2.2、模型规模表 | 模型规模、公开数据集主张、Pre-Norm/RMSNorm、SwiGLU、RoPE 的组合及其先前工作来源 | 这些组件由 LLaMA 首创，或所有后续 LLaMA 版本完全相同 |
| [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)（Touvron 等，2023） | 原始论文；摘要、§2、Table 1、Appendix A.2.1 | $7\mathrm{B}$/$13\mathrm{B}$/$70\mathrm{B}$ 版本、$4\mathrm{k}$ 上下文、较大模型 GQA、$2\mathrm{T}$ 预训练 token | 将合并变化后的总体分数归因于单一架构部件，或将未发布的 $34\mathrm{B}$ 变体视为公开发行版 |
| [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)（Grattafiori 等，2024，v3） | 原始技术报告；摘要、架构与模型表 | $405\mathrm{B}$ 稠密 Transformer、最高 $128\mathrm{k}$ context 及 Llama 3 系列公开信息 | 所有 Llama 版本统一拥有同一上下文或模态接口 |
| [Introducing Llama 3.1](https://ai.meta.com/blog/meta-llama-3-1/)（Meta，2024-07-23） | 官方发布说明；Model Architecture、Instruction and chat fine-tuning | Llama 3.1 的 $8\mathrm{B}$/$70\mathrm{B}$/$405\mathrm{B}$、$128\mathrm{k}$ 上下文、标准 decoder-only 且未采用 MoE 的发布期说明 | 没有报告的低层 block 字段，或该说明之后全部模型版本的架构 |
| [Meta Llama Models repository](https://github.com/meta-llama/llama-models)（访问于 2026-11-24） | 官方型号目录；模型、日期、尺寸、上下文、Tokenizer、Use Policy、License 表 | 当前目录列出的 Llama 2—4 版本、分支尺寸与上下文字段；按版查找许可/模型卡的入口 | 最大上下文字段等同可验证的实际任务能力，或许可在各版间一致 |
| [Llama 4 official model card](https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md)（Meta，发布日 2025-04-05） | 官方模型卡；Model Information 与参数/上下文表 | Scout/Maverick 的 MoE、early fusion、总/激活参数、专家数、模态与模型卡 context 字段 | 由模型卡的 context 或能力简介推出长上下文下的任务质量、成本或完整架构实现 |

## 未解决边界

公开论文、型号目录和模型卡的层级不同；家族版本名称不保证全部字段相同。模型卡中的最大上下文是发布方披露，不等同于训练长度、单次请求上限、不同 runtime 的实现能力或任务有效长度。开放权重也不能单独重建数据清洗、训练过程、失败实验或安全策略。
