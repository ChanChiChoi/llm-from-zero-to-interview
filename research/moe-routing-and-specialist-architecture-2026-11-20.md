# MoE 路由与专家结构路线：第二十一章研究记录

检索与原文核验日期：2026-11-20。研究问题是怎样区分 MoE Transformer 中 token 路由、expert 组合与专家结构路线，并确定 GShard、Switch、Mixtral 和 DeepSeekMoE 各自能支持到什么结论。第二十一册第五章已有的 FFN、参数账本、负载均衡、z-loss 与 capacity 公式证据见[Transformer FFN、SwiGLU 与 MoE 基础证据](transformer-ffn-swiglu-moe-evidence-2026-11-19.md)，本记录只补路线比较、替代路由和教学组织材料。

## 检索范围

- 研究线索："GShard variant top-2 gating position-wise MoE"；"Switch Transformer top-1 routing expert capacity"；"Mixtral 8x7B two experts each layer"；"DeepSeekMoE fine-grained expert segmentation shared expert ACL 2024"；"Expert Choice Routing token choice variable experts per token"；"DeepSeek-V2 MLA DeepSeekMoE architecture".
- 教学线索："mixture of experts routing tutorial lecture notes top-k token choice expert choice capacity beginner"；"site:cs336.stanford.edu mixture of experts MoE routing lecture notes"；"MoE routing in depth tutorial Mixtral shared experts DeepSeekMoE blog".
- 资料范围：原始论文、ACL Anthology 正式论文页、CMU 与 Stanford 课程讲义、研究者工程博客。搜索结果只用于定位资料；路线定义和模型披露按下表所列正文位置复核。
- 未研究：目标硬件上的训练/推理性能、当前服务框架的部署结果、封闭模型的 expert 配置、专家是否具有人类可读的领域语义。

## 来源卡与边界

| 来源 | 版本与实际阅读位置 | 支持 | 不支持 |
| --- | --- | --- | --- |
| Lepikhin et al., [GShard](https://arxiv.org/html/2006.16668) | arXiv v1，2020-06-30；摘要、§§2.1–2.2、§3 | 在其 Transformer 翻译模型中以 position-wise MoE 替换隔层 FFN，使用一种 top-2 gating 变体；GShard 的自动分片、XLA 编译器和 SPMD 执行属于该方案的系统组成 | 该 top-2 细节、设备布局、千亿参数规模或翻译结果适用于所有 MoE 模型 |
| Fedus et al., [Switch Transformers](https://arxiv.org/html/2101.03961) | arXiv v3，2022-06-16；§§2.1–2.3、附录伪代码 | Switch 用 top-1 路由简化每个 token 的选择；论文还具体定义 gate、capacity 与溢出处理 | top-1 在任意模型、batch、硬件上必然更快或质量相同 |
| Jiang et al., [Mixtral of Experts](https://arxiv.org/html/2401.04088) | arXiv v1，2024-01-08；摘要、§2.1、§5 routing analysis | Mixtral 8x7B 每个 MoE 层包含 8 个 FFN expert，router 为每个 token 选 2 个并组合输出；选择结果可随 token 和层改变 | 论文性能数字可跨模型、数据、harness 或硬件外推 |
| Dai et al., [DeepSeekMoE](https://aclanthology.org/2024.acl-long.70/) | ACL 2024 Long Paper 70；§§3.1–3.2、§4.4–4.5 | 两项架构策略是细粒度拆分 routed experts 与隔离始终激活的 shared experts；论文在固定总参数和激活参数的特定消融中报告了结果 | 论文提出的“ultimate specialization”或消融结果证明这两项策略对所有模型和任务普遍更优 |
| Zhou et al., [Mixture-of-Experts with Expert Choice Routing](https://arxiv.org/html/2202.09368) | arXiv v2，2022-10-14；摘要、§§3.1–3.3 | 路由方向可反过来：每个 expert 选择固定 bucket 的 token，因此每个 token 最终可接收不同数量的 experts | 这与 token-choice top-k 是同一种机制，或其训练/下游结果适用于当前所有 LLM |
| DeepSeek-AI, [DeepSeek-V2](https://arxiv.org/html/2405.04434v5) | arXiv v5，2024-06-19；摘要、§§2.1–2.2 | DeepSeek-V2 同时使用 MLA 与 DeepSeekMoE：前者改变 attention/KV 表示，后者采用稀疏 expert 计算 | 联合模型结果可以被单独归因给 MoE 或 MLA |
| CMU 15-780 Graduate AI, [Lecture 19: Mixture of Experts](https://www.cs.cmu.edu/~aditirag/teaching/15780S26_files/Lecture19.pdf) | Spring 2026 讲义；路线图、token-choice 过程、worked example、routing alternatives | 展示了从 FFN 替换、router 选择到 expert-choice 对比的讲解次序，以及一个完整的小型路由演算 | 该讲法对所有读者都更有效，或其图示替代原始论文核验 |
| Stanford MS&E 319, [Lecture 4: Mixtures of Experts](https://web.stanford.edu/class/msande319/lectures/lecture-4-notes.pdf) | 2026 课程讲义；§1.1、top-k routing 部分 | 将稀疏路由输出、active 参数与 expert feature transformation 连起来，并提醒专家学到的特征不必对应“数学/历史”等语义主题 | expert 已形成清晰、可命名的语义分工 |
| Fu-Yun Wang, [Mixture of Experts: How Routing Becomes GPU Computation](https://g-u-n.github.io/blogs/mixture-of-experts.html) | 2026 工程笔记；“Routing”及“Packing”部分 | token-expert pair、按 expert 分组、保存反向置换并还原 token 顺序，是连接路由定义与矩阵计算的解释路径 | 该教学实现代表任一框架、硬件或生产 serving kernel |

## 可复用的技术关系

1. 本章涉及的 Transformer MoE 路线通常把一个或多个 position-wise FFN 替换成 expert FFN 集合；router 对每层的 token hidden state 作选择。它不是按用户请求或整段 prompt 固定选择一个领域模型。
2. token-choice top-k 中，token 先确定候选 experts；容量限制可能影响某项 assignment 是否执行，溢出策略由论文/实现定义。expert-choice 则由每个 expert 选择 token bucket，token 获得的 expert 数量可变。
3. top-k 输出组合权重不是统一约定。Switch 对全 expert logits 做 softmax 后取 top-1 gate；GShard 的 top-2 变体包含自己的归一化和第二路 dispatch 规则；正文应使用模型定义的 combine weight，不写成所有模型都对已选权重重新归一化。
4. GShard、Switch 与 Mixtral 的 route 选择数不能脱离各自具体模型和实验配方比较。GShard 还包含自动分片/编译系统设计；Switch 论文的简化结果绑定其设置；Mixtral 论文明确报告每层 8 个 FFN expert、每 token 选 2 个。
5. DeepSeekMoE 的细粒度拆分是把 FFN expert 切为更窄的子 expert 并相应增加激活数量；shared expert 则对每个 token 固定激活。论文的 fixed-total/fixed-active ablation 支持其模型设置下的结论，不能推出通用性能规律。
6. DeepSeek-V2 将 MoE 与 MLA 组合在一个模型中。MoE 的 token-to-FFN 路由与 MLA 对 attention/KV 表示的改造是不同结构轴，应分别解释和归因。

## 教学组织发现

- CMU 15-780 的讲义由稀疏 FFN 进入 token-choice、完整路由演算，再对照 expert-choice 和系统路线；Stanford MS&E 319 先定义稀疏输出与参数口径，再讲 router 和负载。两者适合帮助搭建“层定位—token 流程—路线差异—证据边界”的读者路径。
- 工程笔记用 token-expert pair 和反向置换解释 dispatch/combine，能补足从数学选择到批量 expert 计算的对象映射；本章只吸收这条解释关系，不引入 grouped-GEMM 教程。
- 这些课程和笔记只证明作者在具体材料中采用了相应组织方式；没有学习效果研究证明它对本书所有读者最优。

## 未验证边界

- 本轮不运行代码、模型、框架 API、GPU kernel、benchmark 或硬件性能实验。
- 不依据 expert id、router histogram 或论文术语推断现实模型中的专家对应数学、代码、语言等固定语义类别。
- 章节中关于方法收益的文字仅描述作者论文在已标注模型与实验条件下报告的结果；Dense/MoE 成本比较、训练 recipe、并行效率和服务 SLO 由相邻章节与第五册对应章节承接。
