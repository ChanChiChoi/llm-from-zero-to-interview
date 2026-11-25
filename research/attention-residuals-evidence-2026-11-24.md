# Attention Residuals / AttnRes 机制与证据

**检索日期：** 2026-11-24 至 2026-11-25  
**主题：** Full AttnRes、Block AttnRes、训练与推理开销、Kimi K3 的配置实例及 2026 年后续变体。  
**常用检索词：** `Attention Residuals AttnRes paper equations`; `Block AttnRes pseudocode two-phase inference online softmax`; `Kimi K3 AttnRes block size config`; `AttnRes tutorial depth-wise attention`; `Low-Rank Attention Residuals`; `Role-Decoupled Attention Residuals`; `Multi-Stream Attention Residuals`。

## 可核验的机制

在 PreNorm Transformer 中，将每次 attention 或 FFN 子层看作一次深度更新：token embedding 是一个独立来源，后续每个来源是子层输出增量。普通残差把可见来源以固定单位权重累加成一个流。AttnRes 为目标读入位置设置可学习伪查询，以历史来源的 RMSNorm 结果计算深度 softmax；同一来源向量同时作为打分 key 和加权 value，RMSNorm 只用于打分，原值进入加权和。伪查询对 token 输入独立，但历史向量依赖 token，所以深度权重仍可随 token 改变。伪查询零初始化令初始 logits 全为零，因而是来源均匀平均；它不是标准残差的恒等初始化。

Block AttnRes 将连续深度更新分成块：块内更新仍以普通加法形成 partial sum，块间只保存已完成块的表示。每个读入位置在深度 softmax 中选择 embedding、已完成块及（块内首个更新之后）当前块 partial sum。attention 和 MLP 子层分别读入并写回；模型输出端还会对可用块表示执行最终深度读入。它缩小需要保存和跨阶段传递的来源数量，但没有跳过 Transformer 子层的 early exit 语义。

原始论文使用 Full AttnRes 的每 token 算术量约 $O(L^2d)$、历史存储约 $O(Ld)$ 作为分析口径；Block AttnRes 将保留来源量降至 $O(Nd)$，而读取成本仍需按深度更新数 $L$ 和块数 $N$ 估算，不能将来源数量直接写成延迟。两阶段推理将同块伪查询批量对已完成块求值，再逐层加入当前 partial sum 并用 online softmax 合并；流水线训练通过跨 stage 缓存减少重复传输。作者报告的开销与收益均应限于论文模型、并行方案、硬件、负载和评测条件。

## 来源及支持范围

| 来源 | 版本/位置 | 支持内容 | 不支持的外推 |
| --- | --- | --- | --- |
| Kimi Team, [Attention Residuals](https://arxiv.org/pdf/2603.15031) | arXiv v1，2026-03-16；§1–5、公式 1–6、Algorithm 1、Tables 2–4、Appendix B | 固定残差问题；Full/Block 定义；伪查询、RMSNorm、零初始化；块边界、两阶段在线 softmax、流水线缓存；缩放律与 48B Kimi Linear 实验 | 其他模型/数据/硬件上的收益；Kimi K3 质量成绩的 AttnRes 单因素归因；作者报告之外的生产时延保证 |
| MoonshotAI [官方 Attention-Residuals 仓库](https://github.com/MoonshotAI/Attention-Residuals) | `master` 分支，2026-11-24 查阅；README 的 equations、Block AttnRes PyTorch-style pseudocode 与结果表 | 官方图示和伪代码有助于检查来源、partial sum、attention/MLP 两次读取 | 仓库 README 可变，且示例为伪代码；不作固定 revision 源码、可运行实现或复现证据 |
| Sebastian Raschka, [Attention Residuals (AttnRes)](https://sebastianraschka.com/llm-architecture-gallery/attention-residuals/) | 架构图谱页面，2026-11-24 查阅；“From a fixed sum”至“Experiments” | 逐 sublayer 的讲解、深度轴与 token 轴区分、伪查询共享与 token 相关权重、标准残差/AttnRes/mHC 对照 | 不能把解释材料当作独立实验验证 |
| Simo Wu, [Attention Residuals: Making the “+” Learned](https://wusimo.github.io/blog/2026/attention-residuals/) | 2026-05-17，Post 2；2026-11-24 查阅 | 用残差流和 source bank 解释 depth retrieval；帮助发现易混术语 | 文中把零查询初始化写成可让标准 residual 恒等起步的 one-hot 初始化，与 Kimi 论文的零初始化→均匀权重不同；不采用该结论，也不采用其“跳层/自适应计算”外推 |
| MoonshotAI, [Kimi K3 technical report v2](https://arxiv.org/html/2607.24653v2) | arXiv v2，2026-07-27；§2.2、§5.4.2 | K3 的 Block AttnRes 公式、embedding 来源、8 个 block × 12 layer size（末块不满）、在线 softmax，以及特定 prefill/decode kernel 路径 | K3 整体模型成绩不能归因于 AttnRes；同一个 block 数/宽度不能迁移到原论文或其他模型 |
| MoonshotAI, [Kimi-K3 fixed config](https://huggingface.co/moonshotai/Kimi-K3/blob/2496450e92e425c886db095102a52a6682ca3970/config.json) | revision `2496450e92e425c886db095102a52a6682ca3970` | 该配置 `attn_res_block_size=12` | 该字段不能单独证明所有 K3 revision 或其他实现均采用相同执行细节 |
| AMD, [Kimi K3 on AMD Instinct GPUs with TokenSpeed](https://www.amd.com/en/developer/resources/technical-articles/2026/kimi-k3-on-amd-instinct-gpus-with-tokenspeed.html) | MI355X 工程文章，2026-11-24 查阅；§3.1–3.4 | 以 K3 为例的 score/blend/normalize 单遍 kernel、register pressure、TP all-reduce 融合与 token-row 分工，展示机制如何被系统实现约束 | AMD 所述 kernel/端到端测量不可外推为一般 AttnRes 加速或独立复现 |
| Jonathan Su, [Low-Rank Attention Residuals](https://arxiv.org/abs/2607.09694) | arXiv v2，2026-09-22；摘要、§4.9 | 保留 full-width values、以低秩 key 做路由的设计；特定 0.5B/1B/4B 训练配置和吞吐对照 | 不证明任意维数、模型或 kernel 都会获益；与原论文不是同一实验协议 |
| Kehan Wang, [Role-Decoupled Attention Residuals](https://arxiv.org/pdf/2608.01075) | arXiv v1，2026-08-02；摘要与限制表 | QK 匹配路径和 value 内容路径拆分；五组配对 seed、120M/343M、2B token 的 NLL 结果 | 作者明确指出仍需更长训练、更多数据与更强 baseline；不把小规模结论写作通用胜出 |
| Qin et al., [Multi-Stream Attention Residuals](https://arxiv.org/pdf/2610.13630) | arXiv v1，2026-10-09；摘要、§4.2、§5、Appendix A/C | 多尺度 causal depthwise convolution 生成多流视图，再分别沿深度读取；18B/28B MoE 设置与作者报告的 12 项下游结果 | 不跨论文直接比较结果；预印本声明和作者配置不等于独立复现或行业共识 |

## 证据解释边界

- Full 与 Block 的 `L`、`N`、`S` 需区分：`N` 是 block 数，`S` 是每个 block 的深度宽度。原 AttnRes 论文缩放实验约取 8 个块；其 48B Kimi Linear 实验以 54 个 residual sublayers、每块 6 个 sublayers 形成 9 个块；Kimi K3 报告则给出 8 个块、12-layer size 和 partial final block。它们不是同一个固定参数。
- 原论文的 1.25× compute 等效、GPQA-Diamond/HumanEval 等变化都绑定论文的模型、训练 token、baseline、评测与 compute 口径。AttnRes 被加入 Kimi K3 后，K3 同时改变 KDA、MLA、MoE、量化、训练与 serving，模型级结果不可单独归因于 AttnRes。
- AMD 的 Kimi K3 kernel 文章是特定硬件及团队实现经验；其与 naive PyTorch 的 kernel 比较不等价于对标准 residual 的端到端架构对照。
- 三个后续预印本分别改变 routing key 维数、QK/V 深度路线或 depth streams。应将它们作为研究线索与各自证据范围说明，不能混为 AttnRes 的原始机制或同条件排行榜。
