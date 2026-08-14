# 第十六章：Decoder-only 为什么成为通用 LLM 主流

## 16.1 从三种预训练目标开始

语言模型架构大致有三种典型目标。Encoder-only 模型用双向上下文学习表示，适合分类和抽取；Encoder-decoder 模型把输入编码后再生成输出，适合翻译、摘要和条件生成；Decoder-only 模型只用因果 mask，按从左到右的顺序预测下一个 token。

Decoder-only 的胜利不是因为其他架构不能生成，而是因为它把大量不同任务统一成了一个目标：给定前缀，预测后续 token。网页、代码、对话、工具调用和结构化输出都可以序列化成 token 流，用同一个训练接口吸收。

## 16.2 小白直觉：把所有任务改写成“续写”

翻译可以写成：

```text
Translate English to Chinese:
The cat sleeps.
中文：猫在睡觉。
```

问答可以写成：

```text
问题：水的化学式是什么？
答案：H2O
```

工具调用可以写成带 schema 的序列：模型先输出调用名和参数，工具返回结果，再把结果接回上下文。模型看到的是不同 token 格式，但底层目标仍是预测下一个 token。

## 16.3 因果语言模型的公式

给定 token 序列 `x_{1:T}`，decoder-only 训练最大化：

```math
\log P_\theta(x_{1:T})=\sum_{t=1}^{T}\log P_\theta(x_t\mid x_{<t})
```

attention mask 为：

```math
M_{ij}=\begin{cases}0,&j\le i\\-\infty,&j>i\end{cases}
```

因此第 `i` 个位置不能读取未来 token。训练时可以把整个序列并行送入 GPU，因为 mask 只限制信息流，不要求硬件逐位置执行；推理时新 token 依赖前缀，才需要自回归循环和 KV cache。

## 16.4 因果 mask 的最小实现

```python
import torch


def causal_attention(q, k, v):
    # q/k/v: [batch, heads, seq, head_dim]
    d = q.size(-1)
    scores = q @ k.transpose(-2, -1) / d**0.5
    seq = q.size(-2)
    mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ v, weights


q = k = v = torch.randn(1, 2, 4, 8)
out, weights = causal_attention(q, k, v)
print(out.shape, weights[0, 0].triu(1).abs().max().item())
```

输出 shape 是 `[1, 2, 4, 8]`，上三角权重应接近 0。工程实现还要处理 padding mask、prefix mask、不同 query/key 长度、半精度 `-inf` 和 fused kernel；这段代码只用于验证信息方向。

## 16.5 为什么统一目标有利于规模化

第一，数据来源广。只要文本能表示成序列，就能进入 next-token prediction，不必为每个任务制作单独标签。第二，目标稳定。训练系统可以复用同一套并行、混合精度、分布式和评估流程。第三，能力可以在上下文中组合：instruction、示例、文档和工具结果都成为前缀的一部分。

这并不意味着 next-token prediction 自动产生可靠推理。目标函数奖励的是分布中高概率的 token，而不是事实正确、引用完整或工具执行成功。后训练、验证器、检索和协议约束是把通用生成能力变成可靠系统的必要层。

## 16.6 Decoder-only 与 Encoder-decoder 的结构差异

Encoder-decoder 有两条信息流：encoder 可以双向读取源输入，decoder 用 causal self-attention 生成目标，并通过 cross-attention 读取 encoder 表示。对长输入、短输出的翻译和摘要，cross-attention 可以把源端表示与生成端分开管理。

Decoder-only 把输入和输出拼在一条序列里，减少了结构种类和训练目标，但输入 token 会进入生成时的上下文，KV cache 和长度管理更直接。对于大量不同任务的统一接口，decoder-only 更简单；对于严格的条件生成、双向编码或输入输出长度差异很大的任务，encoder-decoder 仍有合理性。

| 维度 | Encoder-only | Encoder-decoder | Decoder-only |
|---|---|---|---|
| 主要目标 | masked/表示学习 | 条件生成 | next-token |
| 注意力 | 通常双向 | 编码双向、解码因果 | 因果 |
| 生成 | 需额外 decoder | 原生 | 原生 |
| 任务统一 | 中等 | 强条件生成 | 强序列化统一 |
| 在线 cache | 主要编码表示 | encoder + decoder | decoder KV |

## 16.7 Prefix LM、填空和双向需求

Decoder-only 不是只能严格左到右。Prefix LM 可以让一段前缀内部双向可见，后续目标仍因果生成；代码补全可能用已知前缀和后缀做 infilling；工具调用可以把不可生成的 observation 作为前缀重新注入。这些模式改变的是 attention mask 和数据格式，不一定改变 block 的基本形式。

真正需要双向理解时，不能只在 prompt 里写“请同时看前后文”而期待 mask 自动改变。要确认训练时的 mask、position、损失区域和推理模板一致。否则模型可能把答案泄露给训练目标，或者在推理时无法读取未来片段。

## 16.8 从 base model 到 assistant model

base decoder-only 模型只学习语言分布。要成为可用助手，通常还要经历 instruction tuning、偏好优化、安全训练和工具协议适配。对话模板中的 system、user、assistant、tool token 不只是文字标签，它们可能参与 loss mask、角色边界和 stop condition。

一个简单的监督目标是只对 assistant 输出计算损失：

```math
\mathcal{L}_{\mathrm{assistant}}=-\frac{1}{|Y|}\sum_{t\in Y}\log P_\theta(y_t\mid x_{\le t})
```

如果把用户问题和工具返回也无差别训练，模型可能学会复述 observation，而不是遵循角色边界。相反，过度屏蔽输入也可能让模型学不到如何使用上下文。loss mask 是协议设计的一部分。

## 16.9 训练和推理的工程取舍

训练中，decoder-only 的序列拼接和 packing 利于提高 token 利用率，但文档边界、样本泄漏和 position reset 要处理正确。推理中，长 prompt 先经过 prefill，后续 decode 依赖 KV cache；输入输出混合、工具调用和多轮对话会让上下文不断增长。

模型统一不等于系统统一。不同模型的 tokenizer、chat template、stop token、tool schema、reasoning 字段和上下文位置可能不同。用 OpenAI-compatible API 只能说明请求形状相似，不能保证概率、模板和特殊 token 语义相同。

## 16.10 机制与边界：为什么 decoder-only 的扩展性强

在固定 block 结构下，可以独立扩大层数、宽度、训练 token、上下文、并行度和后训练数据。scaling law 研究让参数、数据和计算预算能够被量化规划；但 compute-optimal 结果依赖模型族、数据质量和训练范围，不是永远的比例定律。

decoder-only 的另一优势是把 ICL 变成一等能力：任务示例、标签空间和外部证据以 token 形式进入相同前向路径。这为 RAG、Agent 和结构化生成提供了统一接口，也把上下文污染、提示注入和长上下文成本引入核心系统。

## 16.11 面试追问、误区与练习

**问：为什么大模型大多使用 decoder-only，而不是 encoder-decoder？**

标准回答：decoder-only 用 causal next-token prediction 统一文本、代码、对话和工具序列，数据和系统扩展简单，并天然支持生成与 ICL；encoder-decoder 在条件生成和输入输出解耦上仍有优势，所以不是理论上被淘汰。

**问：训练时能并行，推理时为什么不能并行生成整段？**

标准回答：训练目标中的真实前缀已知，可以并行计算所有位置；生成时后一个 token 依赖前一个采样结果，只能逐步解码，除非使用 speculative decoding、multi-token prediction 或其他近似路径。

常见误区包括把 causal mask 说成“计算不能并行”、把 decoder-only 说成没有 encoder 信息、把模型名称当作架构证明，以及忽略模板和 loss mask。

练习：把分类、翻译、工具调用和代码补全各写成 decoder-only 序列，画出每个 token 的可见范围，并指出哪些 token 参与 loss。

## 16.12 统一接口带来的协议成本

把任务都序列化成 token 流很灵活，却把协议正确性交给了模板和解码器。角色标记、工具 JSON、停止 token、引用格式和多模态占位符任何一处不一致，都可能让同一模型表现得像换了一个模型。

因此 decoder-only 的统一接口需要一份版本化模板：

~~~text
system_format
user_format
assistant_prefix
tool_call_format
tool_result_format
stop_tokens
tokenizer_revision
~~~

训练、离线评估和线上服务必须使用同一套字段，或明确适配层。只比较裸 prompt 的输出而不记录模板，无法解释很多迁移回归。

## 16.13 生成模型和判别式使用方式

decoder-only 可以通过生成标签完成分类，也可以通过 log probability 比较候选答案。两种用法的成本和错误不同：生成受格式和采样影响，候选比较需要对齐长度、tokenizer 和 label prior。

对于结构化输出，最好使用 grammar 或 constrained decoding 验证 schema；对于开放问答，仍需事实和来源验证。统一生成接口并不意味着所有任务都应该用自由文本解码。

## 16.14 统一目标不等于统一能力

Decoder-only 的 next-token objective 很统一，但不同任务对目标函数和解码协议的要求并不相同。续写、问答、代码补全、分类和工具调用都可以被写成 token 序列，却可能需要不同的 loss mask、special token、停止规则和评估器。

例如分类可以让模型生成标签，也可以比较候选标签的条件 log probability。前者容易接入通用 API，但会受到格式和采样影响；后者更接近判别式评分，却要处理候选长度、tokenizer 和类别先验。工具调用还需要区分“决定调用哪个工具”和“生成合法参数”两个错误来源。

## 16.15 从概率目标到服务协议

自回归模型学习的是：

~~~math
\mathcal{L}(\theta)
=-\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t})
~~~

线上系统真正执行的却是模板拼接、tokenize、prefill、解码、停止和后处理。任一环节改变，条件分布就不再是训练时的同一个条件分布。一个看似相同的用户问题，若 system prompt、角色 token 或 assistant 起始标记不同，模型看到的上下文已经不同。

因此模型版本应和 tokenizer、chat template、stop token、工具 schema、grammar、max output 和 adapter 一起版本化。模型能力评估如果只保存自然语言 prompt，而不保存序列化后的 token 和协议字段，无法可靠复现。

## 16.16 Decoder-only 的边界与补偿层

Decoder-only 主干擅长把多种信息串成统一序列，但它不会自动提供事实更新、权限过滤、精确引用、长时记忆或结构化事务。RAG、memory、tool calling、constrained decoding、verifier 和 serving scheduler 都是在补偿这些边界。

这并不意味着 decoder-only 设计错误。统一主干降低了模型生态的复杂度，外部层则把变化快、需要审计或有副作用的能力放在更容易控制的位置。系统设计时应明确哪些能力由模型承担，哪些能力由协议、检索和执行器承担。

## 16.17 统一续写目标带来的数据责任

Decoder-only 的目标形式统一，不代表数据天然适合混合训练。网页正文、代码、对话、工具轨迹、表格和多模态占位符的格式不同；如果只把它们拼成字符串，模型可能学会错误的角色边界、工具参数或文档分隔方式。训练数据需要记录来源、模板、loss mask、特殊 token 和任务标签。

同一个 token 在不同位置的监督含义也可能不同。用户问题通常作为条件，不应被当作要预测的答案；assistant 输出、工具调用参数和工具结果可能有不同 loss 权重。最小数据审计要检查：渲染后的 token 序列、assistant-only mask、EOS、tool call marker、padding 和多轮边界。

## 16.18 Decoder-only 不是所有任务的最优结构

Encoder-decoder 在输入和输出边界清晰、需要双向编码源文本的翻译、摘要和条件生成任务中仍然自然；encoder-only 在分类、检索和 token-level understanding 中更直接。Decoder-only 的优势是统一接口、规模化生态和生成能力，不是它在每一个任务上都拥有最低成本。

选择架构时应比较全链路：训练目标、输入长度、输出长度、推理 cache、微调数据、评估和部署。一个输入很长、输出很短的抽取任务，使用 decoder-only 可能在 prefill 和 KV 上付出额外成本；一个需要连续工具调用和开放生成的 Agent，decoder-only 的统一 token 协议则更有价值。

## 16.19 从语言概率到服务协议

模型训练的是 token 条件概率，服务提供的却是 message、tool call、stream event、usage 和错误码。中间必须有 tokenizer、chat template、parser、grammar、权限和状态管理。模型能够续写 JSON，不等于客户端能安全执行其中的动作；模型输出自然语言，也不等于它遵守了产品的角色和引用协议。

因此 decoder-only 的“通用性”需要三层验收条件：语言概率是否合理，结构化输出是否符合 schema，外部动作是否通过 policy 和幂等执行。任何一层失败，都不能用“模型会续写”解释为模型本身的问题或产品已经兼容。

## 16.20 信息流视角的取舍

GPT-3 的规模化 decoder-only 与 ICL 证据见 *Language Models are Few-Shot Learners*（https://arxiv.org/abs/2005.14165）；原始 Transformer 见 https://arxiv.org/abs/1706.03762。具体模型的层数、训练数据、模板和损失设计以对应模型卡或技术报告为准。

Decoder-only 成为通用 LLM 主流，是“统一序列目标 + 可扩展系统 + 上下文组合能力”的结果，而不是一句“生成模型更方便”可以解释完的。它的成本和可靠性问题，需要由后续的 cache、检索、对齐和 serving 技术共同承担。

## 16.21 从信息流而不是模型名字判断架构

读者看到 GPT、Llama、Qwen 或某个新模型的名称时，不能仅凭名称判断它是 decoder-only，也不能把“支持对话”当成架构证据。真正需要确认的是四件事：输入 token 的可见范围、训练损失落在哪些位置、生成时状态如何递推，以及输入和输出是否经过同一个主干。

例如，一个模型可能在主干上使用 causal decoder，却在特定阶段加入 prefix bidirectional mask、视觉编码器、cross-attention 或外部 memory。此时“decoder-only”描述的是主干的基本信息流，不代表系统中没有任何编码器或其他模块。架构分析要画出一条具体路径：

```text
raw input -> tokenizer/encoder -> decoder blocks -> logits
                                      -> KV/state cache
tool result --------------------------^
```

如果工具结果在下一轮被重新序列化为 token，它属于新的前缀；如果它只写入外部 memory，再由检索模块选择性注入，它就不是主干天然拥有的长时记忆。把这两种路径混为一谈，会错误估计上下文成本和能力来源。

## 16.22 Packing、loss mask 与跨样本泄漏

Decoder-only 训练常把多个短样本 packing 到同一条长序列中，以减少 padding 浪费。packing 只有在 attention mask 和 position 处理正确时才等价于独立样本。若样本 B 能读到样本 A 的 token，模型会得到不应存在的额外上下文；若 position 没有 reset，样本边界附近还会产生训练时不存在的相对距离。

设一个 packed sequence 包含样本 A 和 B，位置集合分别为 `I_A`、`I_B`。独立样本要求：

```math
M_{ij}=-\infty,
\quad i\in I_B,\ j\in I_A,
```

除非数据协议明确允许 B 读取 A。对 assistant-only loss，还要同时满足 `L_t=1` 只出现在可监督的 assistant token 上。实际审计不能只打印一个 batch shape，应检查随机位置的可见矩阵、position id、loss mask 和 EOS。一个简单的单元测试是把 A 的内容替换成随机字符串，若 B 的 loss 或 logits 明显改变，就要确认这种依赖是否被允许。

这也是 decoder-only “统一序列”所带来的责任：序列化越灵活，边界越容易被悄悄破坏。训练脚本、数据 collator 和推理模板必须共享同一份边界定义。

## 16.23 长度扩展同时改变算力、显存和统计分布

把 context length 从 8K 增加到 128K，不只是把一个配置乘以 16。prefill 的 attention 计算、KV cache、位置分布、训练样本构造和评测任务都会改变。对标准多头注意力，单层全量 prefill 的 score 计算量近似为：

```math
C_{\mathrm{attn}}\propto B\,T^2\,d,
```

而 decode 阶段每生成一个 token，读取历史 KV 的成本近似随 `T` 增长。若每层每个 token 的 KV 元素数为 `2 n_{kv} d_h`，KV cache 的字节数可粗略写成：

```math
M_{\mathrm{KV}}
\approx B\,T\,L\,(2n_{kv}d_h)\,b,
```

其中 `L` 是层数，`b` 是每个元素的字节数。MQA/GQA 可以减小 `n_{kv}`，但不能消除长 prompt 的 prefill 计算，也不能保证模型在更长位置上保持准确引用。

更长的训练序列还会改变样本统计：短样本比例下降，文档边界和长距离依赖出现频率上升，batch 中每个 step 能看到的独立样本数下降。长度扩展的实验应同时报告 token throughput、有效样本数、长程任务、KV 峰值和训练稳定性，而不是只看 max position 配置是否被接受。

## 16.24 一个架构选择的 worked example

设团队要做一个企业知识助手：输入通常是 30K token 的文档，输出是 500 token 的带引用答案，偶尔还要调用只读数据库。候选方案是 decoder-only + RAG、encoder-decoder，或 encoder-only 检索器加一个小 decoder。

如果目标是统一对话、工具调用和开放式追问，decoder-only 的协议和生态成本最低；但长文档不能无条件塞进上下文，需要先做权限过滤、召回和证据组装。若主要任务是固定格式抽取，输入远长于输出，encoder-decoder 或专用 encoder 可能减少生成侧开销。若系统还要做召回，encoder-only 的向量表示可以独立缓存，降低重复计算。

比较时可以建立一个任务—资源表：

| 方案 | 主要优势 | 主要成本 | 必测验收条件 |
| --- | --- | --- | --- |
| decoder-only + RAG | 对话、工具和格式统一 | prefill、KV、模板复杂 | 引用支持、工具协议、TTFT |
| encoder-decoder | 输入输出边界清晰 | 生态和 serving 适配 | 长输入质量、cross-attention 成本 |
| encoder + 小 decoder | 检索/抽取可专门优化 | 系统组件更多 | 召回到生成的接口一致性 |

这个例子没有一个脱离任务的“最佳架构”。架构优劣是全链路函数：

```math
U=Q-\lambda_1 C-\lambda_2 L-\lambda_3 R,
```

其中 `Q` 是经过切片和安全验收条件后的质量，`C` 是计算与存储成本，`L` 是延迟，`R` 是协议和治理风险。公式只是帮助把讨论从“哪个模型更先进”拉回到可测量的系统决策。

## 16.25 复现 decoder-only 结论时要保留哪些证据

一个可复现的架构实验至少需要保存 tokenizer revision、chat template、attention mask、position ids、loss mask、context length、训练/推理 batch、dtype、KV layout、采样参数和评测 harness。只保存自然语言 prompt，无法证明两次实验看到的是相同 token 序列。

质量评估要覆盖 next-token loss 之外的任务：局部续写、跨段复制、冲突证据选择、结构化输出、工具参数和拒答。若只测 perplexity，可能看不出模板错位和外部动作错误。serving 评估则要将 prefill、decode、queue、cache、stream 和后处理分开计时。

因此，decoder-only 的学习重点不是背诵“它成为主流”，而是能从信息流推导训练目标，从训练目标推导协议约束，再从协议约束推导系统成本和失效模式。只有这条链闭合，读者才真正理解了它为什么成功、又为什么需要那么多补偿层。
