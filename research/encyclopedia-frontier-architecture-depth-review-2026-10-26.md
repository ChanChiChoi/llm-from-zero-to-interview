# 第四册第十九章：前沿架构与模型更新研究记录

核验日期：2026-10-26。正文：[`19-frontier-architecture-updates.md`](../book-04-llm-encyclopedia/chapters/19-frontier-architecture-updates.md)。本章是架构路线与厂商快照索引，不是某一家模型的统一架构论文。研究重点是把容易混为一谈的深度聚合、序列状态、显式 KV 压缩和多路残差拆开，并明确模型披露、论文实验和 Serving 测量各自能够证明什么。

## 本轮研究问题

1. AttnRes、KDA、CSA/HCA、mHC 改变的张量对象和坐标轴分别是什么？它们能否互相替代，或能否被组合？
2. 深度方向的可学习聚合有哪些前序方案；AttnRes 的新增内容是什么，是否应称为首创？
3. 线性递归状态、压缩 KV 与稀疏索引分别如何降低长序列成本，又分别会丢失哪类信息？
4. HC 与 mHC 分别解决什么问题；双随机约束能保证什么，不能保证什么？
5. 各论文报告的性能数字能否横向排名，哪些是算法消融，哪些是全模型/系统的联合结果？

## 资料与证据

| 资料 | 实际查看内容 | 支持的正文判断 | 不能支持的判断 |
|---|---|---|---|
| [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732)，2022 版 | 章节目录、引言和效率分类；将稀疏、低秩/核方法、压缩与循环等路线分开，并把效率分成计算和内存 | “高效注意力”不是单一机制；长序列要分开记 FLOPs、内存、召回和实现条件 | 该综述早于 KDA、CSA/HCA、AttnRes、mHC，不能用来证明这些新方法的细节或当前优势 |
| [On the Resurgence of Recurrent Models for Long Sequences](https://arxiv.org/abs/2402.08132)，2024 版 | 摘要及 recurrent state、attention/recurrent hybrid、长序列研究缺口相关段落 | 固定状态可降低每步历史访问成本，RNN/SSM 与 attention hybrid 有不同并行和记忆取舍 | 不能证明 KDA 的逐通道门、Kimi Linear 的配置或新模型 benchmark |
| [DenseFormer](https://arxiv.org/abs/2402.02622)，NeurIPS 2024 | 原论文摘要和 depth-weighted averaging 方案 | 深度方向加权聚合早于 AttnRes；方法动机是跨层复用与信息流 | 不证明 DCA、AttnRes 的具体公式或 Kimi 模型配置 |
| [DeepCrossAttention](https://proceedings.mlr.press/v267/heddes25a.html)，ICML 2025 | 论文摘要、正文结果表与相关工作；查看其 same-quality/up-to-3x-faster 声明及语言建模设置 | DCA 以输入相关权重和 depth-wise cross-attention 重组层输出；论文报告必须绑定其任务、模型与训练预算 | 不能把“最高 3 倍”转写为 AttnRes、任意推理服务或所有模型的吞吐收益 |
| [Attention Residuals](https://arxiv.org/abs/2603.15031)，2026 | arXiv 摘要；与本地深度专题 [`第 75 章`](../book-21-transformer-architecture-evolution/chapters/75-attention-residuals深度方向注意力.md) 中的原论文机制和源码阅读记录交叉核对 | Full AttnRes 对先前层表示做 softmax 深度聚合；Block AttnRes 用块内累计、块间选择减少存储/通信；论文报告 scaling-law 实验，并在 48B/3B active、1.4T tokens 的 Kimi Linear 配置中评估 | Kimi K3 的发布提及不能替代 K3 自己的 report/config；论文本身的实验数字不能推成 K3 完整配置或通用服务收益 |
| [Kimi Linear](https://arxiv.org/abs/2510.26692)，2025 | 原论文 HTML 的机制、128K long-context benchmark 与表 5；交叉核对本地 [`KDA 研究摘记`](model-update-2026-09/kda-source-notes.md) 和 [`第 76 章`](../book-21-transformer-architecture-evolution/chapters/76-kda从delta规则到线性注意力.md) | KDA 的 `alpha` 是逐通道衰减、`beta` 控制 delta 更新；模型把 KDA 与全局 MLA 混合。128K 套件中作者报告平均 54.5、RULER 84.3、RepoQA 68.5；MLA baseline 平均 52.2 | 这些是训练完成的混合架构/基准结果，不是单独替换 KDA kernel 的因果消融，也不代表 Kimi K3 配置或所有任务的稳定领先 |
| [DeepSeek V4 Technical Report](https://arxiv.org/abs/2606.19348)，2026 | 原论文摘要、架构 §2、异构 cache 和推理框架 §3.5；与本地 [`第 77 章`](../book-21-transformer-architecture-evolution/chapters/77-csa-hca从压缩kv到百万上下文.md)、[`第 78 章`](../book-21-transformer-architecture-evolution/chapters/78-mhc双随机残差连接.md) 对照 | CSA 是压缩后稀疏选择；HCA 更强压缩后 dense read；模型报告 1M 设置下 V4-Pro 为 V3.2 的 27% 单 token FLOPs/10% KV cache，V4-Flash 为 10%/7%；Serving 维护多类异构缓存状态 | FLOPs/cache 比例是模型和系统联合设计的发布方报告，不是 CSA/HCA 单独因果增益，也不是质量、召回、并发或所有 GPU 速度保证 |
| [Hyper-Connections](https://arxiv.org/abs/2409.19606)，2025 v3 | 原论文 HTML 的问题定义、输入/输出/残差连接矩阵、多路扩展和模型实验 | HC 扩展残差流并学习深度与宽度连接，目标是改善固定残差权重与表示/梯度问题 | HC 的多流扩展不能与深度历史 attention 混为一谈；论文报告的 OLMoE 结果是指定设置 |
| [mHC](https://arxiv.org/abs/2512.24880)，2025 | 原论文 HTML 的 Birkhoff polytope、双随机约束、矩阵乘法闭合和 infrastructure 部分 | mHC 将残差流映射投影到双随机矩阵集合；有限 Sinkhorn 迭代近似该约束；`n=4` 大规模设置报告 6.7% 时间开销 | 不保证整个非线性模型范数、训练稳定性或推理性能；6.7% 不是通用 overhead |

综述用于建立分类，近年的代表论文用于校验具体机制和数字；厂商模型卡/技术报告只支撑其公开版本。架构论文、固定 config、Serving 源码、microbenchmark 与用户端服务结果需要分层记录。没有在本轮重新加载完整权重或复现实验。

## 新增的方案对照与具体推演

### 改变对象

| 路线 | 它想解决的约束 | 状态如何变化 | 主要新风险 |
|---|---|---|---|
| DenseFormer / DCA / AttnRes | 深层网络不能显式挑选哪些早期表示继续影响当前层 | 在深度坐标上加权历史层输出；AttnRes 使用 softmax，Block AttnRes 在块边界保留跨块状态 | 历史激活保存、读取、流水线通信和块粒度折衷 |
| Gated DeltaNet / KDA | 每个 token 保留完整历史 KV 的内存随序列增长 | 对固定形状 `S_t` 做衰减、delta 修正和写入；KDA 细化逐通道衰减 | 有限状态覆盖/忘记；训练 chunk 与 decode recurrent kernel 不同；远端精确取回能力有限 |
| DSA / CSA / HCA | 百万上下文的显式缓存和每 query 读取量太大 | 先沿 token 维压缩 KV；CSA 再按 query 稀疏选择；HCA 在更强压缩表示上 dense read | 块内压缩损失、top-k 漏选、局部窗口、压缩尾部和 cache layout 均需分别验收 |
| HC / mHC | 多路 residual stream 的自由映射可能产生不稳定或不均匀混合 | HC 学习多流连接；mHC 将 residual mixing 限制到双随机矩阵集合 | 多流激活/通信与 Sinkhorn 计算；双随机只约束映射，不等同于整体范数守恒 |

### CSA 与 HCA 的玩具例子

16 个原始位置按连续 4 个汇成一个条目，得到 4 个压缩条目。若 CSA 风格路径取 top-2，就只让核心注意力读两个候选；相关定义若所在块未入选，问题是索引召回。若 HCA 风格路径更强地压缩后把全部压缩条目 dense 读完，就没有 top-k 漏选这一环，但原文的块内差异可能已经在压缩时丢失。这一数字例子只隔离“压缩”和“选择”两个步骤，实际模型的压缩倍率、稀疏度和局部窗口必须读其固定报告/config；不能把玩具参数当成 V4 配置。

## 旧知识点去向（coverage map）

| 原章内容 | 本轮判断 | 修订后去向 |
|---|---|---|
| Attention Residuals / AttnRes | 核心候选，保留并深化；纠正“新架构名即首创”的误读 | 原小节留在原位；新加深度轴对照、DenseFormer/DCA 前序关系、实验范围；机制主讲仍在第二十一册第 75 章 |
| Kimi Delta Attention / KDA | 核心候选，保留并深化；区分有限状态收益和遗忘/精确检索代价 | 原小节留在原位；新加与 KDA 的作用轴/证据比较；公式、代码和混合 MLA 主讲仍在第 76 章 |
| CSA / HCA | 核心候选，保留并深化；把 compression error 与 top-k recall 拆成两个故障源 | 原小节留在原位；新增 16→4→top-2 的教学推演、1M 系统指标证据边界；完整 cache 生命周期在第 77 章 |
| mHC | 核心候选，保留并深化；与 HC、AttnRes、GR 分层区分 | 原小节留在原位；增加 HC→mHC 的机制谱系、6.7% overhead 条件；公式和 Sinkhorn toy 在第 78 章 |
| GPT-6 Astra；GLM-5.3；Kimi K3；GLM-5.3-Flash；GLM-5.2；Mistral Small 4；Step 3.5 Flash | 主流模型/机制/接口快照，全部保留；不从接口字段推断架构 | 对应章节仍在原顺序；保留第六册第 18 章、第十六册第 20 章、第十七册第 15 章、第二十一册第 79/80/84/86/88 章等深度入口和模型研究笔记 |
| Grok 4.6；Claude Opus 5；Claude Fable 5.1；DeepSeek V4.1-Flash；DeepSeek V4 Flash Vision；Claude Sonnet 5；Claude Haiku 4.5；DeepSeek-R1-0528 | 产品/API/安全/Serving 事实按类型保留；不是独立基础架构论文 | 各原有条目、官方模型/API 来源和关联研究记录均保留；动态接口与价格仍以单项来源日期为准 |
| K2 Horizon MoVA 36B/A4B；Qwen3.8；GPT-OSS；Gemini 3.5 Flash-Lite；Grok 4.7；K2 Horizon 3.7B；Kimi K2.7 Code；Qwen3-Omni；Qwen3-VL | 架构/模型/系统条目均保留；将并行出现的技术限定到具体版本 | 原有小节顺序不变；保留第二十一册第 82/83/87/91/92/94 章和已有关联研究笔记；本章的新表只解释它们属于哪条架构轴，不代替各专题 |

除标题改为同时覆盖架构机制与模型快照、以及前四个方法段落的来源入口外，本轮未删减或移动任何模型条目。低优先级的服务快照仍保留其出处与未知项，不作为已验证的内部结构。

## 配套同步

- `PAPERS.md` 增加 DenseFormer、DCA、HC、mHC 的前序研究入口；KDA、AttnRes、DeepSeek V4 保留已有论文索引。
- `GLOSSARY_EN_ZH.md` 增加 DenseFormer、DCA、HC，已有 AttnRes/KDA/CSA/HCA/mHC 条目保留。
- `INTERVIEW_BANK.md` 增加一道区分深度聚合与多路残差混合的比较题，已有各方法专项追问保留。
- `EXERCISES.md` 已有四种核心路线的实现/消融任务，本轮没有重复复制。

## 尚未完成的独立核验

1. 各架构论文没有在同一数据、参数/FLOPs 预算、tokenizer、训练 token、硬件和 Serving engine 下做统一对照，因此本章只比较机制和各自的证据边界，不形成跨论文排名。
2. 模型快照章节包含动态价格、context、effort、模型 ID 和部署路由；本轮没有把所有产品页重新当作实时 API 手册逐项复核，具体数据需沿各条来源日期和对应模型研究笔记查验。
3. 没有运行章节中的 Python 示例、加载模型、训练、GPU profiling、公式渲染或 PDF 构建。前四个架构条目本轮只做来源、机制和静态结构核验；可执行示例在第二十一册相应专题中。
4. DeepSeek V4 的 FLOPs/cache 比例、Kimi Linear 长上下文分数、DCA 速度声明、mHC wall-time 均为论文作者报告；没有本地独立复现。
