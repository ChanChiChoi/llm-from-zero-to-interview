# 混合架构中的显式 KV 路径与按层容量：研究证据

检索日期：2026-11-24。研究范围：解释 hybrid 序列模型中显式逐 token KV、局部 KV、逐 token latent 和递归 state 的差别；核对层间交错、层内混合和跨层 sharing 各自改变的对象；确定怎样把架构层表转成显式 KV payload 估算及路径评估问题。未训练模型、未运行代码或 benchmark。

## 本轮检索与结论

- 关键词包括 `hybrid attention KV cache`, `GDN full attention layers`, `KDA MLA cache`, `hybrid head KV sharing`, `KV cache size recurrent state`, `hybrid architecture lecture notes` 与对应中英文术语；核对论文、官方团队发布说明和课程讲义。
- 全局 attention 保存的普通 K/V 可逐 token、逐层增长；局部 attention 只需为本层保留其可见窗口；MLA 仍可对历史 token 位置进行 attention，但每个位置存的是压缩 latent；递归模型保存随输入更新的状态，不等同于逐位置的 K/V 表。
- 层间交错、层内混合、局部/全局/稀疏可见范围、跨层 KV sharing、latent 压缩分别改变网络放置、分支计算、可见 token 集、重复副本或每 token 表示维度，不能统称为同一种“减少 KV”。
- 在普通全维 K/V 路径上，可按显式 K/V 层逐层求和：每层贡献与实际保留 token 数、KV head 数、head dimension 和 element bytes 成正比。只按全局层占总层数的比例估算，必须声明各层形状、dtype、请求长度相同的简化条件。
- 论文中直接检索/复制的结果绑定特定架构、训练设置和任务。显式读取路径本身不保证模型一定找到或正确解释证据；整体模型质量也不能单独归因于某个 cache 路径。

### QSA 的计算选择与 KV 保留边界

复核：[On the Design of Qwen3.8-Next Architecture](https://arxiv.org/html/2608.30320v1)，§2.1.2（特别是 Motivation、Compressed Lightweight Indexer、Implementation Details 和 Efficiency Analysis）。报告描述 QSA 将 indexer keys 按 micro-block 平均压缩，按 query 对 block 打分，再选择 top-$K$ block 并展开成 token 位置供 sparse core attention 使用；示例配置使用 $K=2048$、压缩比 $r=4$。这说明它改变索引与每个 query 的 attention 计算集合。报告没有在该节规定主 K/V 是否完整驻留、按需重算或采用另一种 serving 保留策略，故本记录不由 sparse read 推断 main KV payload 缩减。准确区分了 indexer 表示、主 K/V 和 query 选择集合之后，才能判断具体实现的内存账本。

## 主要技术来源

### Qwen Team：Qwen3.8-Next 架构报告

- 来源：[On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability](https://arxiv.org/abs/2608.30320)，arXiv v1，2026-08-31。
- 阅读位置：摘要、§2.1.1 Motivation/Architecture Ablation、§2.1.2 Qwen Sparse Attention。
- 支持：报告称基础 Qwen3.8-Flash-Next 每四层采用三层 GDN 和一层 full attention；GDN 以固定形状的递归状态压缩 prefix，full attention 提供逐 token 的 direct retrieval。其架构消融在所测九项 benchmark 上对比 full-attention、SWA hybrid 和 GDN hybrid；作者提醒结果本身不能隔离所有改变的因果。报告还称 continued pretraining 时以 QSA 替换 full-attention 层。
- 边界：报告中的比例、训练阶段、checkpoint 和 benchmark 结果属于该模型与实验设置，不是任意 hybrid 模型的最优层比；QSA 的稀疏读取不能据此称为删除或压缩完整 K/V payload。

### Alibaba Cloud Community：Qwen3.8-Flash-Next 发布说明

- 来源：[Qwen3.8-Flash-Next: A New Architecture, Towards Ultimate Cost-Efficiency](https://www.alibabacloud.com/blog/qwen-3-8-flash-next-a-new-architecture-towards-ultimate-cost-efficiency_603501)，2026-08-27。
- 阅读位置：Model Architecture / “Attention: GDN + QSA for Efficient Memory and Precise Retrieval”。
- 用途：该团队发布说明以“GDN 压缩历史、QSA 检索所选重要上下文”解释 hybrid 设计，适合作为教学组织入口。结构和实验事实仍以 Qwen 技术报告为准。
- 边界：这是发布方说明，不是独立复现；页面中的 serving 速度数字仅适用于其指定 1M-token 配置、硬件和 prefix-cache 条件，本章不引用为通用性能。

### Moonshot AI：Kimi K3 技术报告

- 来源：[Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653)，arXiv v2，2026-08-07。
- 阅读位置：§2.1 Hybrid Attention、§2.1.2 Gated MLA，以及 §5.4.1 Unified cache layout for hybrid KDA–MLA attention。
- 支持：报告称每个主体 block 包含三层 KDA 和一层 Gated MLA，另在主干末端放置 Gated MLA；其 Gated MLA 对逐 token KV latent 做全局 attention，而 KDA 使用固定大小递归状态。报告明确两种状态大小和生命周期不同。
- 边界：MLA latent 不是普通完整 head-specific K/V；Kimi K3 的结构不能推广成所有 MLA 或 hybrid 模型配置。§5.4 的 prefix cache layout、scheduler 和 serving 设计属于系统实现，本章只用它确认两种表示不同。

### Dong et al.：Hymba hybrid-head

- 来源：[Hymba: A Hybrid-head Architecture for Small Language Models](https://arxiv.org/abs/2411.13676)，arXiv v1，2024-11-20；发表于 ICLR 2025。
- 阅读位置：§2.1 Fused Hybrid-Head Module、§2.2 KV Cache Optimization、附录 B 的 architecture ablations。
- 支持：attention 与 SSM heads 可在同一层并行；作者另行研究 local/global attention 和相邻层 KV sharing。论文实验显示这些选择分别改变 cache、recall 与 throughput。
- 边界：论文中的数字来自 Hymba 小模型、指定训练和 A100 测试设置；组合后的性能不能归因于单一 hybrid head、window 或 KV-sharing 技术。

### De et al.：Griffin cache accounting 与 copy/retrieval

- 来源：[Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models](https://arxiv.org/abs/2402.19427)，arXiv v1，2024-02-29。
- 阅读位置：§6.2 Copy and retrieval capabilities、附录 F.4 Cache sizes。
- 支持：论文分别计算 full MQA KV、local attention cache 和 recurrent state 的大小，并在特定 synthetic copy/retrieval 与 phonebook lookup 设置中比较 Hawk/Griffin 和 Transformer。
- 边界：结果取决于论文的模型、训练预算、窗口与任务；不能推广为“所有 recurrent model 都无法记住远处细节”或“加少数 attention layer 必定提高任意任务质量”。

## 教学组织材料

- UC Berkeley Scalable AI, [Lecture 3: Architectures to Break Bottlenecks](https://scalable-ai.eecs.berkeley.edu/S2026/assets/lecture_slides/lecture3.pdf)，Spring 2026，阅读 hybrid stacks / Nemotron 3 页及总结。讲义把少数 attention layers 的 KV state 与其余序列层的 SSM state 分开，并用不同架构轴组织 hybrid、MLA 和 sparse attention；用于设计表述，不用作模型特定证据。
- 第十一章现有 Stanford MS&E 319 KV caching notes 和 Chapter 20 的 PagedAttention 论文分别作为常规 KV 机制与物理分页的入口；本章不重讲它们的公式推导或 serving 实现。

## 采纳范围与未验证项

- 本章采用“表示类型—层位置/可见范围—每层 payload—任务证据”的组织方式，强调普通 KV 与 latent/state 分账。
- 不把某个模型的 attention 比例写成通用经验，不把模型级 benchmark 当作路径级消融，不把 theoretical payload 写成实际显存、吞吐或延迟。
- 未进行本地模型、kernel、服务、代码或 benchmark 复现；所有实验数值仅在确有教学价值且注明来源设置时引用。
