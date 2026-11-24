# MoE、SSM 与 Attention 组合架构研究记录

## 研究问题与范围

- 日期：2026-11-23。
- 问题：MoE、SSM 与 Attention 在公开语言模型架构中分别放在哪里，组合形态有什么差别；现有消融能支持哪些关于组合收益的判断？
- 范围：第二十一册第四十七章。比较 MoE 放在混合序列架构的 FFN 位置、用 MoE-FFN 扩展 Mamba，以及把专家化放进 Mamba 内部投影并与 Attention 混合的公开路线。第 43 章主讲 Attention–SSM 融合粒度，第 21—22 章主讲 MoE 路由和系统成本；本章只说明两类架构选择怎样相接。
- 读者结果：能区分序列混合器（Attention/SSM）与条件参数路径（MoE），追踪代表性组合中的专家化位置，并根据消融和测量条件判断“组合有益”主张的范围。
- 检索词：`Jamba hybrid Transformer Mamba MoE architecture ablation`、`BlackMamba MoE SSM architecture`、`Routing Mamba MoE projection NeurIPS 2025`、`MoE inside Mamba projections hybrid Samba`、`Jamba architecture explanation AI21`；中文检索“MoE SSM Attention 组合架构”“Jamba 架构消融”“Mamba 投影专家化”。
- 来源范围：优先核对 Jamba 的 ICLR 2025 正式论文、BlackMamba 原始预印本、Routing Mamba 的 NeurIPS 2025 正式论文；同时阅读 AI21 发布方的 Jamba 解释文章，作为结构讲解入口。搜索截至本日；没有发现足以支持统一“未来最优组合”的证据。

## 研究发现

1. **MoE 与 Attention/SSM 位于不同的结构维度。** Attention 和 SSM 决定序列表示如何混合、历史以什么路径进入当前计算；MoE 通常把一个 FFN 替换为按 token 选择的专家路径，改变条件计算和参数容量。MoE 也可以专家化 SSM 内部投影，因此“三者相加”不是固定 block 或通用成本公式，必须说明具体插入位置和路由范围。
2. **Jamba 将 MoE-FFN 放进层间 Attention–Mamba 主干。** ICLR 2025 论文以重复 block 表达架构自由度：每 block 八层、Attention:Mamba 为 1:7，部分 MLP 每隔两层替换为 MoE，每个 MoE 层有 16 个专家、每 token top-2。1.3B、250B token 的消融中，1:3 与 1:7 hybrid 结果相近，作者选择计算更省的 1:7；7B、50B token 的比较中，hybrid 优于论文中的纯 Attention/Mamba 控制，增加 MoE 后若干 benchmark 和 log-prob 指标继续改善。结论属于论文模型、数据、训练预算和评测；不能解释成 MoE 对任意混合架构都带来同等增益，也不重复第 43 章的混合比例专题。
3. **BlackMamba 展示不保留 Attention 的两路组合。** 论文用 Mamba 取代 Transformer Attention，并以 routed MLP experts 替换普通 MLP，研究 340M/1.5B 与 630M/2.8B（active/total）模型，在自建语料上训练 300B tokens。它说明 SSM 与 MoE 可组合成无显式 Attention 的主干；论文的质量、FLOPs 和生成速度比较绑定模型、语料、实现与基线，不能视为大规模生产模型的普遍成本结论。
4. **Routing Mamba 把专家化移入 SSM 投影路径。** NeurIPS 2025 论文将稀疏专家用于选定 Mamba 线性投影，以共享路由协调相关子模块；作者报告，对 Mamba 投影逐项独立套用普通 MoE 常不能改善效果，甚至可能退化，而共享路由、选择性专家化可以提升其设置下的表现。该方法也在含 Mamba 与滑动窗口 Attention 的 Samba 上测试。论文一项对照将 RoM(Conv, Gate, Out) 与 Samba expand=4 比较：在 SlimPajama 20B tokens、8×A100、4K 序列训练设置的表中，验证困惑度约为 9.77 对 9.78，单次 4K forward 的 FLOPs 为 4.74T 对 6.13T；这是特定配置的约 23% 差值，不是普适加速保证。另一个 FFN-MoE 对照中，加入 RoM 的变体与相近 active 参数基线平均任务准确率接近（50.1% 对 50.2%），不能表述为一致的质量增益。
5. **组合位置决定应当怎样隔离因果。** 至少要区分“Attention/SSM 混合器的变化”“FFN-MoE 的增量”和“SSM 投影专家化的增量”；只把三类组件放在同一模型中，无法证明互补或协同。Jamba 提供混合主干与 FFN-MoE 的独立消融，Routing Mamba 提供投影位置、共享路由和混合 Samba 的具体比较。不同来源的分数不能拼成一个排名。

## 直接证据来源

### 原始论文与正式论文页

- **Opher Lieber et al., “Jamba: A Hybrid Transformer-Mamba Language Model” (ICLR 2025)**，https://proceedings.iclr.cc/paper_files/paper/2025/file/a9ed43fa31dc8b4a7d7a673d713dcb5f-Paper-Conference.pdf。阅读 §2—3 与 Appendix C.1、C.4。支持 Jamba 的层级组合、MoE 插入位置和配置自由度，以及 1.3B/250B-token 与 7B/50B-token 消融。其长上下文 KV、吞吐、质量对照仍受模型、软件、硬件及报告方法限制。
- **Quentin Anthony et al., “BlackMamba: Mixture of Experts for State-Space Models” (arXiv 预印本, 2024)**，https://arxiv.org/abs/2402.01771。阅读 §V—VII。支持 attention-free Mamba 与 routed MLP 的 block 组合、训练规模、数据和作者报告的效率/评估结果；不作为大规模生产场景的一般性能证据。
- **Zheng Zhan et al., “Routing Mamba: Scaling State Space Models with Mixture-of-Experts Projection” (NeurIPS 2025)**，正式论文页：https://proceedings.neurips.cc/paper_files/paper/2025/hash/79bc44e53e8a981ac1510d4c24ba3d71-Abstract-Conference.html；论文：https://papers.nips.cc/paper_files/paper/2025/file/79bc44e53e8a981ac1510d4c24ba3d71-Paper-Conference.pdf。阅读 §4—5、Table 1—2 及 Appendix A.5。支持选定 Mamba 投影的共享路由专家化、MoE 位置消融、Samba 混合模型对照及作者报告的 23% FLOPs 差值。结果绑定 SlimPajama、20B token、8×A100、对应 active/total 参数、序列长度和评测口径。

### 讲解与架构说明材料

- **AI21, “Introducing Jamba: AI21's Groundbreaking SSM-Transformer Model”**，https://www.ai21.com/blog/announcing-jamba/。阅读架构图附近的 block 说明；用来理解 Attention/Mamba layer 与 MoE-MLP 的层级关系。它是发布方解释，涉及效果和“优势”时以正式论文的对照和实验条件为准。
- **Databricks, “Pioneering a Hybrid SSM Transformer Architecture” (Barak Lenz 演讲)**，https://www.youtube.com/watch?v=dOR7GLSpEGg。作为 Jamba 作者团队对组合设计的教学入口；演讲宣传中的性能表述未作为独立证据。

## 证据边界与待核事项

- “MoE + SSM + Attention”不是唯一的 block 定义：专家可以替换 MLP，也可以作用于 SSM 的部分投影；混合器、专家路由和层次顺序需按具体论文描述。
- Jamba 的公开消融比产品总榜更适合说明模块增量；BlackMamba 不是与 Jamba 相同模型/数据的直接对照；Routing Mamba 的表格属于其自有训练栈和实验设置。不可跨论文比较分数、延迟或成本。
- Routing Mamba 的 “约 23% FLOPs”指特定 Samba/RoM 配置的一次 4K forward 计算量对照，不等同于端到端训练或 serving 延迟节省。
- 未运行本地模型、代码、API 或 benchmark；本章不预测哪种组合将成为通用最优，也不提供服务调度、状态迁移或生产回滚方案。
