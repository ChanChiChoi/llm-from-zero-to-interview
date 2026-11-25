# CSA/HCA 压缩注意力机制研究记录

研究日期：2026-11-24。问题范围是 DeepSeek-V4 技术报告中 Compressed Sparse Attention（CSA）与 Heavily Compressed Attention（HCA）如何形成压缩 KV 条目、如何把它们交给每个 query 读取，以及两种路径的计算/信息损失差别。第 74 章负责机制原理；第 77 章保留 V4 具体模型配置、百万 token 结果、参考实现和 serving/kernel 证据。

本轮检索词：`DeepSeek V4 technical report CSA HCA Compressed Sparse Attention Heavily Compressed Attention arXiv 2606.19348`、`site:github.com/deepseek-ai DeepSeek V4 model card CSA HCA compressor indexer`、`DeepSeek V4 CSA HCA explained compression attention tutorial blog`、`DeepSeek V4 CSA HCA lecture notes compressed KV attention mechanism`。检索并阅读了 arXiv 正式报告、Hugging Face Transformers 的 V4 架构文档、Stanford MS&E 319 Lecture 5 notes，以及一份逐式对照论文和公开实现的研究者技术笔记；后两类用于教学解释与交叉检查，不替代作者原文。

## 核心机制

### 压缩算子

CSA 与 HCA 不只是从原始 token KV 中挑出少数索引；它们把一个或多个 token 的投影 KV 表示训练成新的压缩条目，后续 attention 直接把条目作为 key/value 读取。压缩器对输入 hidden state 分别生成 content 投影和 gate logits，并加入可学习的位置偏置。softmax 沿块内位置归一化，且每个 KV 特征维使用自己的位置权重；因此它不是普通均值，也不是每个 token 只有一个共享标量权重。训练损失通过后续读取路径训练这些投影和 pooling 权重。

### CSA：重叠压缩，再检索压缩条目

报告 §2.3.1 的 CSA 对每个压缩条目使用两套 content/gate 投影：当前长度为 $m$ 的块走 $a$ stream，前一个块走 $b$ stream；两段合起来最多覆盖 $2m$ 个 token 位置，但输出每次向前移动 $m$，所以条目数约为 $n/m$，不是 $n/(2m)$。序列起点没有前块时以零权重 padding。报告称用同一压缩模式形成另一组 compressed indexer keys。

Lightning Indexer 从当前 query 的低秩表示生成 indexer query heads 和可学习 head 权重，对可用的压缩 indexer keys 算加权 ReLU 相似分数，再选 top-$k$。主 attention 读取的是被选中的 compressed KV 条目，不是把 top-$k$ 当成原始 token 的 gold evidence mask。该路径仍需要为候选压缩条目打分；主 attention 只对被选条目执行 core attention。

### HCA：非重叠重压缩，密集读取压缩域

报告 §2.3.2 的 HCA 使用一套 content/gate 投影和更大的压缩跨度 $m'\gg m$，对互不重叠的块做 feature-wise gated pooling。每 $m'$ 个输入位置形成一个新 KV 条目，长程序列长度约降为 $n/m'$。HCA 没有 indexer，也没有 top-$k$ 选择；它对所有因果上已经完成的压缩条目执行 dense attention。因此“dense”描述的是压缩后条目集合，不是逐 token 原始历史。

### 共同读取路径与因果边界

CSA/HCA 的压缩条目同时作为 key 与 value，由多 query heads 共享；当前 query 的未压缩 local sliding window 与长程压缩条目一并进入核心 attention。作者报告所述结构并非两条分别 softmax 后再用门控合并的独立分支。一个压缩块未完成前不能成为可读取的完整条目；局部原始 KV 同时保留近期细节和未完成块内因果可见性。RMSNorm、部分 RoPE/输出反旋转、attention sink 与 grouped output projection 是周边结构细节，不构成 CSA 与 HCA 的定义差别；位置处理和逐层 KV 布局由第 77 章及相关章节承接。

## 可作的对比与不能外推之处

若原序列长度为 $n$，忽略块边界时 CSA 保存约 $n/m$ 个 compressed entries，主 attention 对每个 query 读取至多 $k$ 个历史压缩条目并另读局部窗口；其 indexer 的候选打分对象仍随 $n/m$ 增长。HCA 保存约 $n/m'$ 个长程条目，每个 query 读取所有已完成条目并另读局部窗口。这是由结构推得的条目数与读取规模；不能直接当成实际 FLOPs、显存字节或延迟，因为 entry width、索引表示、低精度格式、kernel、batch 和 cache 布局也影响成本。

CSA 同时面对压缩表征损失与 indexer 漏选；HCA 不做 top-$k$ 漏选，但高压缩率下仍可能把 query 需要的区分信息混合掉。两者共用的近期原始窗口可缓解局部细节损失，但不恢复被压缩历史中的任意原始 token。应分别检查压缩表示误差、检索覆盖和实际 kernel 访问，而不能只报告压缩比例或模型 context-length 字段。

DeepSeek-V4 报告关于百万 token KV/FLOPs 的数字属于特定模型、层配置、精度和 baseline；模型整体的 long-context benchmark 也不能单独证明 CSA 或 HCA 的因果收益。具体 Flash/Pro 的层数、压缩率、top-k、partial RoPE、模型级比较和公开 inference code 条件留在第 77 章，不能复制成 CSA/HCA 的通用常数。

## 来源与证据边界

| 来源 | 本轮阅读位置 | 支持范围 | 边界 |
| --- | --- | --- | --- |
| DeepSeek-AI, [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence, arXiv v1](https://arxiv.org/html/2606.19348v1)（2026-04-26） | §2.3.1 CSA，Equations 9–19；§2.3.2 HCA，Equations 20–26；§2.3.3–2.3.4 的共同结构和效率讨论；§3.5.1 KV cache 结构 | 原始方法定义、CSA 重叠压缩与 lightning indexer、HCA 非重叠 dense 路径、共同 local window/共享 KV、作者报告的配置和效率主张 | 作者技术报告；V4 结构与效率结果不可外推成所有压缩注意力或任意硬件的表现。报告的 latency/FLOPs 不是本地复现 |
| Stanford MS&E 319, [Lecture 5: KV Caching, Sparse Attention, and Compression](https://web.stanford.edu/class/msande319/lectures/lecture-5-notes.pdf)（本轮于 2026-11-24 访问） | Definition 10、Algorithm 4、Example 11、§3.3–§4 | 解释 learned coordinatewise pooling、整块完成后才可读、压缩与选择两类成本、local/compressed candidate 组合的教学路径 | 课程讲义的简化模型用于讲解；V4 特有的两 stream overlap、indexer 和实现细节以 DeepSeek 报告为准 |
| zyeric, [CSA And HCA: Definition, State, And Cost Model](https://github.com/zyeric/v4-attn-notes/blob/main/docs/notes/01-csa-hca-definition.md)（GitHub `main`，访问日期 2026-11-24） | §§1–5，尤其 overlap、pooling axes、indexer entries 与 candidate set | 便于追踪张量、state 和 cost 的解释；其关键机制描述与报告对应 | mutable `main` 的研究者笔记而非官方来源；未独立证明训练效果、总体性能或设计动机 |
| Hugging Face Transformers, [DeepSeek-V4 model documentation](https://huggingface.co/docs/transformers/model_doc/deepseek_v4)（本轮于 2026-11-24 访问） | Architecture / Hybrid attention | 当前 Transformers 文档中关于 CSA/HCA 配置字段、压缩倍率及 local branch 的实现说明 | 框架文档对应具体库实现与默认配置，不替代模型方法定义或官方服务性能测量 |

原论文明确说明 HCA 的更大非重叠 compression rate 和 CSA 的 compressor/indexer，但未给 overlap 设计提供单独消融以证明其必要性；因此正文描述数据流，不把 overlap 的动机或收益写成已验证结论。在本轮查阅的来源中，未找到独立同规模 CSA/HCA 消融来隔离压缩、索引和层间 schedule 的因果贡献。
