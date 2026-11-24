# Mistral 7B 与 Mixtral 8x7B 架构证据

## 研究范围与检索记录

- **研究日期：** 2026-11-24。
- **问题：** 核验 Mistral 7B v0.1 的 GQA/SWA、Mixtral 8x7B 的 FFN MoE 配置、公开参数及论文评测主张，并寻找适合解释稀疏专家和资源口径的教学材料。
- **范围：** 固定在 Mistral 7B v0.1 论文、Mixtral 8x7B 论文和其官方权重发布说明；不把家族名推广到后续版本，不复测速度或质量。
- **检索：** Web search，查询词包括 `Mistral 7B arXiv 2310.06825 grouped-query sliding-window 4096 benchmark`、`Mixtral of Experts arXiv 2401.04088 47B 13B active 32k top 2 experts`、`Mistral 7B explained GQA sliding window attention blog`、`Mixtral 8x7B explained sparse mixture of experts top 2 tokens blog`。随后打开原论文、Mistral 官方发布页、Hugging Face 技术文章和 NVIDIA 技术博客核对原文。

## 可支持的事实

### Mistral 7B v0.1

- Jiang 等人的 [Mistral 7B 论文](https://arxiv.org/abs/2310.06825)于 2023-10-10 发布，明确对象为 Mistral 7B v0.1。论文 Table 1 列出 $32$ 个 query heads、$8$ 个 K/V heads、$4{,}096$ 的 sliding window 和 $8{,}192$ 的 `context_len`。这些是不同字段，不能简化为一个“有效上下文长度”。
- 论文采用 grouped-query attention（GQA）和 sliding-window attention（SWA）。GQA 通过较少 K/V heads 降低解码期内存需求；SWA 限制单层的直接可见范围。论文按其层数与窗口给出理论跨层传播跨度，但理论路径不证明任务级远程检索有效。
- 论文称基础模型在其测试基准上超过 Llama 2 13B，并在推理、数学和代码生成任务上超过 LLaMA 1 34B；这些结论对应原论文测试和对照，不是小模型的普遍规律。论文说明其基准比较使用作者自己的评估 pipeline 重跑对照。
- 同一论文称指令微调版本在作者报告的人类和自动基准上超过 Llama 2 13B Chat。该比较不用于推断基础模型，也不说明单个架构组件的因果效果。
- 论文声明模型按 Apache 2.0 发布；这说明发布许可，不替代对训练数据、权重 revision 或后续模型的逐项核对。

### Mixtral 8x7B

- Jiang 等人的 [Mixtral of Experts 论文](https://arxiv.org/abs/2401.04088)于 2024-01-08 发布。论文描述其主干与 Mistral 7B 相同，但每层 FFN 替换为 MoE；每个 token 在每层由 router 选两个专家并合并输出。Architecture Table 列出 $8$ 个 experts、`top_k_experts=2`、`context_len=32768`。
- 论文摘要以约 $47\mathrm{B}$ 参数可供每个 token 使用、每 token 推理激活约 $13\mathrm{B}$ 参数概括模型规模。官方 [Mixtral 发布说明](https://mistral.ai/news/mixtral-of-experts/)给出更精确的 $46.7\mathrm{B}$ 总参数与 $12.9\mathrm{B}$ 每 token 激活参数；“总参数”和“激活参数”不是同一口径。论文还明确说 serving 内存成本随总稀疏参数规模增长，路由及多专家设备负载会带来开销，因此不能由 active parameters 直接推出显存、通信或 wall-clock 性能。
- Mistral 官方发布说明日期为 2023-12-11，权重按 Apache 2.0 提供；这解释了官方发布在 2023 年、arXiv 论文在 2024 年的日期差异。官方页报告与 Llama 2 70B、GPT-3.5 多数 benchmark 的总体比较，但这些仍是发布方的特定评测结论。
- Mixtral 论文摘要概括所测 benchmark 的总体表现；论文正文的 Table 3 按指标列分数，个别指标不高于每个对照模型。正文应表述为作者报告的总体/多数指标结果，并让读者区分摘要概括与逐项数据。论文还说明部分协议与 Llama 2 论文不同，例如 MBPP 使用人工核验子集，TriviaQA 不提供 Wikipedia 上下文。

## 教学与工程材料

- [Hugging Face：Welcome Mixtral](https://huggingface.co/blog/mixtral)解释 FFN 专家替换与共享主干，提醒“8x7B”不是 8 个完整的 7B 独立模型；文章还指出预训练数据集规模、构成和预处理细节未公开。该文适合教学和公开信息边界，不作为独立速度评测。
- [NVIDIA：Applying Mixture of Experts in LLM Architectures](https://developer.nvidia.com/blog/applying-mixture-of-experts-in-llm-architectures/)用 Mixtral 说明 MLP 专家化与 token 路由，并讨论负载分布；其中观察来自该博客自己的特定输入/层实验，未用作 Mixtral 通用负载结论。
- 官方 Mixtral 发布页称预训练数据来自开放 Web，但未披露完整数据规模、构成及预处理细节。不得把“数据来自开放 Web”改写为训练语料完全公开。

## 编辑使用边界

- 将 GQA、SWA、FFN MoE 分成不同路径：K/V head 共享、attention 单层可见范围、每 token 的 FFN 专家选择。
- 将模型总参数、每 token 激活参数、常驻权重、路由/通信开销分开叙述。总/激活数字沿用论文取整值，同时记录发布页精确报告值。
- 不从论文 benchmark 结果推导 GQA、SWA 或 MoE 的独立因果贡献；单模块贡献需要其他条件匹配的消融。
- 不声称 Mixtral 每项 benchmark 都胜过每个对照模型，也不把论文的性能结论推广到任意提示、模型版本、硬件或服务栈。
- 对固定版本配置比较、窗口信息传播、KV cache、MoE 路由和系统成本的完整解释分别回链第二十一册第 17、19、11、21、22 章；第五十五章聚焦两个历史模型案例及证据边界。
