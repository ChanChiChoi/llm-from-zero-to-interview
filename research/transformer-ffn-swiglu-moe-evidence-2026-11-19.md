# Transformer FFN、SwiGLU 与 MoE 基础证据

检索与原文核验日期：2026-11-19。研究范围：Transformer 中 position-wise FFN 的定义、GLU/SwiGLU 形式与参数比较、MoE 的基本路由/参数口径，以及这些主题的公开教学组织。此记录只保存技术发现和来源边界；第二十一册第 5 章的结构决定、修改和交接见 `progress/book-21-transformer-architecture-evolution/chapter-05-audit.md`。

## 检索范围

- 学科资料检索词：`GLU Variants Improve Transformer Shazeer 2020 SwiGLU T5 C4`；`Switch Transformers JMLR top-1 capacity factor`；`DeepSeekMoE ACL 2024 shared fine-grained experts`；`Attention Is All You Need position-wise feed-forward networks`；`PaLM SwiGLU three matrices`；`LLaMA SwiGLU 2/3 4d`；`GShard top-2 every other FFN`；`Mixtral 8x7B 13B active parameters`。
- 教学组织检索词：`Stanford MS&E 319 Autumn 2026 Lecture 4 Mixture of Experts notes`；另查 Stanford CS336 的 MoE 单元，未获得足以核对具体讲义次序的直接课程材料，不据此作判断。
- 资料类型：原始论文/技术报告、JMLR 与 ACL 论文页面、Stanford MS&E 319 课程大纲及 Lecture 4 讲义。英文资料；搜索结果仅用于发现入口，关键结论按下列原文位置核对。
- 未覆盖：闭源模型未公开的专家规模、路由配方和生产 kernel；当前硬件上的实际吞吐/延迟；超出本章基础目标的 2026 模型榜单。

## 来源卡与边界

| 来源 | 原文阅读位置 | 可支持 | 不可推出 |
| --- | --- | --- | --- |
| Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017 | §3.3 | 原始 Transformer 的 position-wise FFN 对每个位置分别、以同一参数应用两层全连接变换与 ReLU | 现代 decoder-only 模型都采用相同 FFN 宽度或激活函数 |
| Shazeer, [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202), 2020 | §§2–3.3，特别是等参数/等计算的 T5 变体及 held-out C4 与微调结果 | ReGLU、GEGLU、SwiGLU 的门控形式；三矩阵结构；为匹配参数和计算量将中间维度从 $d_{\mathrm{ff}}=3072$ 调到 $d_{\mathrm{ff}}=2048$；在该 T5 序列到序列、C4 span-filling 设置中 GEGLU/SwiGLU 的 held-out perplexity 较优，微调任务结果有噪声 | SwiGLU 对所有架构、数据、训练设置或 decoder-only LLM 都必然优于 ReLU/GELU |
| Chowdhery et al., [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311), 2022 | §2 | PaLM 采用 SwiGLU，指出其 MLP 使用三组矩阵，并援引 Shazeer 的 compute-equivalent 对照 | PaLM 中的架构组合结果能单独证明 SwiGLU 的因果收益 |
| Touvron et al., [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971), 2023 | §2.2 | LLaMA 使用 SwiGLU，FFN 中间维度设为 PaLM 常用 $4d$ 的约 $2/3$ | 这是跨模型的精确等质量、等延迟或普遍最优比例 |
| Shazeer et al., [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538), 2017 | §§1.2、2、5 | 稀疏门控 MoE 的基本想法、router 组合 expert 输出和负载/带宽挑战；代表实验包括 LSTM 语言建模与机器翻译 | 该论文已直接证明现代 Transformer MoE 的所有结构或性能结论 |
| Lepikhin et al., [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668), 2020 | §§2.1–2.2、§4 | 在 Transformer 中将隔层 FFN 替换为 position-wise MoE，采用 top-2 变体并结合自动分片；其实验面向大规模多语机器翻译/TPU 设置 | GShard 的拓扑、层间隔和质量/吞吐结果适用于所有 Transformer 与设备 |
| Fedus et al., [Switch Transformers](https://www.jmlr.org/papers/v23/21-0998.html), 2022 | §§2.1–2.2 与实验部分 | top-1 router、capacity、溢出 token assignment 处理、负载均衡和低精度训练的作者方案；结果绑定 T5/C4/多语任务及其训练资源 | top-1 一定比 top-2 更快，或 token dropping/capacity 采用同一通用策略 |
| Jiang et al., [Mixtral of Experts](https://arxiv.org/abs/2401.04088), 2024 | §2.1、§3 | Mixtral 8x7B 在每个 MoE 层使用 8 个 FFN expert、每 token top-2；expert 函数采用 SwiGLU；作者报告约 47B sparse total、13B active，并指出服务内存与路由/设备利用仍有成本 | `2/8` 可直接当作整模型 active/total 参数比，或 active 参数等于实际内存、FLOPs、时延 |
| Dai et al., [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://aclanthology.org/2024.acl-long.70/), ACL 2024 | ACL Anthology 元数据及摘要 | 摘要明确提出细粒度拆分 routed experts 与隔离 shared experts 两种专家专业化策略，并报告指定 2B/16B 实验的结果 | 摘要结果不能证明细粒度/共享专家在所有任务和硬件上普遍更优；本轮没有把摘要数字用于正文 |
| Stanford MS&E 319, [课程大纲](https://web.stanford.edu/class/msande319/) 与 [Lecture 4: Mixtures of Experts](https://web.stanford.edu/class/msande319/lectures/lecture-4-notes.pdf), 2026 | 大纲 Lecture 4；讲义 §1.1、后续 router、load-balance、capacity 与 expert-placement 部分 | 一个面向已具备深度学习、数学和算法前置的高阶课程，先定义稀疏 MoE 与参数口径，再讲 router/train、负载均衡、capacity 和设备放置；active 与 total 参数被明确区分 | 该课程顺序是所有初学者的唯一最佳顺序，或已由学习效果研究证明 |

## 可复用的技术关系

1. Dense FFN 对每个位置独立使用相同参数，张量形状可写作 $[B,T,D]\rightarrow[B,T,M]\rightarrow[B,T,D]$。忽略 bias 时，两矩阵 FFN 的参数量约为 $2DM$。
2. GLU 变体用两条投影形成相乘的通道表示，再经 down projection 回到模型宽度；SwiGLU 在 gate 分支使用 SiLU/Swish。若两条上投影宽度同为 $M$，忽略 bias 时有三组 $D\times M$/$M\times D$ 矩阵，参数量约为 $3DM$。这说明相同 $M$ 下不能与两矩阵 FFN 直接比较。
3. 以 dense $M=4D$ 为参数参照，SwiGLU 的近似等参数宽度为 $M\approx8D/3$。例如 $D=4096$ 时，dense $M=16384$ 为 134,217,728 个矩阵参数；SwiGLU $M=11008$ 为 135,266,304 个矩阵参数，约为前者的 1.008 倍。这个算术只匹配参数量，不匹配模型质量、延迟或硬件吞吐。
4. MoE 层在输入 token 上由 router 选取专家并合并其输出；normalized weights 是常见形式之一，但权重归一化及路由实现并非唯一规范。FFN/SwiGLU 与 MoE 属于不同架构轴：Mixtral 的 routed expert 本身就是 SwiGLU FFN。
5. 总参数与 token-selected 参数应和存储、计算、通信、调度成本分账。专家权重需要由整套分布式模型表示/保存，但不意味着都驻留在单张设备上；每 token active 参数也不是 FLOPs 或端到端延迟的代名词。
6. 容量超额按 token-expert assignments 计数，不等同于唯一 token 数；同一个 token 可有多个 expert assignment。具体 assignment 被丢弃后 token 是否还由另一条路径贡献，取决于路由和容量实现。

## 未验证边界

- 本轮不运行书中 Python 片段、框架、模型或 benchmark；参数示例为可复算的教学算术。
- 没有验证 Mixtral 或其他模型在任一目标 serving stack 上的 GPU 内存、吞吐或尾延迟。
- 课程资料仅用于观察教学组织，未找到可证明该组织顺序对本书读者有因果学习收益的研究。
