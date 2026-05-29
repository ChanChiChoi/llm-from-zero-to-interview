# 第十二章：In-Context Learning 与显式 token 检索能力

## 12.1 本章定位

前面几章讲了 Transformer 的核心优势和瓶颈：并行性、O(n^2) attention、KV cache 和长上下文推理成本。本章进入一个更偏“能力机制”的主题：In-Context Learning，简称 ICL。

GPT-3 之后，大家意识到大模型不一定每个任务都要 fine-tuning。只要在 prompt 里放任务说明、少量示例、相关文档或工具返回结果，模型就可能在不更新参数的情况下完成任务。

例如：

```text
English: cat
French: chat

English: dog
French: chien

English: book
French:
```

模型看到前两个例子后，可以推断这是英文到法文翻译任务，然后输出 `livre`。

这就是 In-Context Learning 的直观形式：

```text
模型把上下文当作临时任务说明、临时数据集和临时工作记忆。
```

本章要回答的问题是：

1. In-Context Learning 到底是什么。
2. 它和 fine-tuning、RAG、prompt engineering 有什么区别。
3. 为什么 attention 对 ICL 很关键。
4. “显式 token 检索能力”是什么意思。
5. 示例、标签空间、输入分布、格式分别起什么作用。
6. 为什么 ICL 不等于真正参数更新。
7. ICL 为什么会受位置、顺序、干扰信息和上下文长度影响。
8. 为什么显式检索能力是 Transformer 与 Mamba/SSM 等替代架构比较时的关键问题。

本章的核心观点是：

```text
ICL 是 Transformer 把上下文窗口变成运行时工作空间的能力；attention 让模型可以显式读取 prompt 中的示例、标签、约束和证据 token。
```

## 12.2 资料来源和可信边界

本章主要参考以下公开资料：

1. Brown et al., 2020, *Language Models are Few-Shot Learners*。GPT-3 展示大规模自回归语言模型可以通过 prompt 中的 few-shot demonstrations 完成任务。
2. Min et al., 2022, *Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?*。研究 demonstrations 中哪些因素影响 ICL，指出真实标签不总是必要，label space、input distribution 和 format 很关键。
3. Dai et al., 2022/2023, *Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers*。从 meta-optimizer 角度解释 ICL，提出 Transformer attention 与梯度下降之间的类比。
4. Dong et al., 2023, *A Survey on In-context Learning*。系统综述 ICL 的定义、技术、应用和挑战。
5. Liu et al., 2023, *Lost in the Middle*。说明长上下文模型虽然能接收长输入，但对中间位置相关信息利用不稳定。
6. Guu et al., 2020, *REALM: Retrieval-Augmented Language Model Pre-Training*。代表把外部检索与语言模型结合的早期路线，用来区分上下文内部检索和外部 RAG。
7. mechanistic interpretability 中关于 induction heads 的研究。它说明某些 attention head 会实现类似“看到 A 后跟 B，当前再看到 A 时预测 B”的复制/归纳模式。本章只讲直觉，不把 mechanistic 结论绝对化。

需要说明的是，ICL 的机制仍是开放研究问题。不同论文从不同角度解释：模式匹配、贝叶斯推断、隐式梯度下降、检索、归纳头、数据分布记忆等都能解释一部分现象。本章重点建立工程和面试中实用的理解框架。

## 12.3 什么是 In-Context Learning

In-Context Learning 指的是：

```text
模型在推理时仅通过输入上下文中的任务描述、示例或证据，临时适配当前任务，不更新模型参数。
```

关键点有三个。

第一，不更新参数。

```text
模型权重固定，没有反向传播，没有 optimizer step。
```

第二，任务信息在 prompt 中。

```text
任务说明、格式、标签、示例、文档证据都以 token 形式放进上下文。
```

第三，适配发生在前向计算中。

```text
模型通过 attention 和深层非线性计算，把上下文信息转化为当前输出条件。
```

所以 ICL 可以看作一种运行时能力：

```text
参数里存储通用能力，上下文里提供临时任务配置。
```

## 12.4 Zero-shot、One-shot、Few-shot

ICL 常见形式包括 zero-shot、one-shot 和 few-shot。

Zero-shot：

```text
只给任务说明，不给示例。
```

例如：

```text
请判断下面评论是正面还是负面：
“这个电影节奏太慢了。”
```

One-shot：

```text
给一个示例。
```

例如：

```text
评论：这个手机很好用。
情感：正面

评论：这个电影节奏太慢了。
情感：
```

Few-shot：

```text
给多个示例。
```

few-shot 示例能提供更多信息：

1. 任务格式。
2. 标签空间。
3. 输入分布。
4. 输出风格。
5. 边界样例。
6. 隐含规则。

GPT-3 的冲击在于，它展示了模型规模足够大时，few-shot prompt 可以在很多任务上接近或超过一些传统 fine-tuning baseline。

## 12.5 ICL 和 Fine-tuning 的区别

Fine-tuning 是参数更新：

```text
训练样本 -> 计算 loss -> 反向传播 -> 更新参数
```

ICL 是上下文条件化：

```text
示例和任务说明放进 prompt -> 前向推理 -> 输出答案
```

对比：

```text
Fine-tuning：慢，需训练数据和算力，结果固化到参数中。
ICL：快，不更新参数，任务配置保留在上下文中。
```

ICL 的优点：

1. 快速切换任务。
2. 不需要训练流程。
3. 可解释性较好，因为示例和规则在 prompt 中可见。
4. 适合用户临时指令、工具结果、文档证据。
5. 可以和 RAG、Agent、workflow 结合。

ICL 的缺点：

1. 受上下文长度限制。
2. 每次调用都要携带示例，增加 token 成本。
3. 对 prompt 格式、顺序、示例选择敏感。
4. 容易被干扰 token 影响。
5. 不一定能学习真正复杂的新技能。

## 12.6 ICL 和 Prompt Engineering 的关系

Prompt engineering 是设计输入上下文的方法。

ICL 是模型利用上下文适配任务的能力。

两者关系是：

```text
Prompt engineering 负责把任务信息组织成 token。
ICL 负责让模型在前向计算中利用这些 token。
```

例如一个好的 few-shot prompt 会考虑：

1. 示例是否代表目标分布。
2. 标签是否清晰。
3. 输出格式是否一致。
4. 示例顺序是否合理。
5. 任务说明是否和示例一致。
6. 是否避免无关干扰信息。
7. 关键证据是否放在模型容易利用的位置。

Prompt engineering 不是玄学堆词，而是在给 ICL 提供更清晰的运行时数据结构。

## 12.7 ICL 和 RAG 的区别

RAG 是 Retrieval-Augmented Generation。

典型流程是：

```text
用户问题 -> 外部检索系统找文档 -> 文档放入 prompt -> LLM 基于上下文回答
```

ICL 和 RAG 的关系：

```text
RAG 提供外部资料进入上下文。
ICL/attention 在上下文内部读取这些资料并生成答案。
```

所以 RAG 可以看成一种 ICL 输入构造方式。

区别在于：

```text
ICL 强调模型如何利用上下文完成任务。
RAG 强调如何从外部知识库找相关内容并放进上下文。
```

Self-Attention 像上下文内部的可微分检索；RAG 是上下文外部的显式检索。

二者配合后：

```text
外部检索扩展知识边界。
内部 attention 决定哪些 token 被真正使用。
```

## 12.8 为什么 Attention 对 ICL 很关键

ICL 需要模型在当前输入中找到相关信息。

例如 few-shot 分类：

```text
输入：这家餐厅太棒了。
标签：正面

输入：服务很差，再也不来了。
标签：负面

输入：味道不错，但价格太贵。
标签：
```

模型需要读取：

1. 示例中的输入文本。
2. 示例对应标签。
3. 当前输入。
4. 输入和标签的格式关系。

Attention 提供了这种 token-to-token 路由能力。

当前要预测标签的位置，可以关注前面示例中的：

1. `标签：正面`。
2. `标签：负面`。
3. 相似情感表达。
4. 分隔符和格式 token。

如果没有 attention 的上下文内显式读取能力，模型很难稳定利用 prompt 中任意位置的示例和证据。

## 12.9 显式 Token 检索能力是什么意思

这里的“显式 token 检索能力”不是指外部向量数据库检索，而是指：

```text
模型能在上下文 token 中定位、读取和使用相关 token。
```

例如：

1. 从前文找出变量定义。
2. 从 few-shot 示例中找相似输入和对应标签。
3. 从文档证据中找答案句。
4. 从工具返回 JSON 中找字段值。
5. 从代码上下文中找函数签名。
6. 从多轮对话中找用户最新约束。

这是一种内部检索能力。

它依赖：

1. attention score 是否把相关 token 权重拉高。
2. 位置编码是否支持距离关系。
3. 训练中是否见过类似格式。
4. 模型是否能抗干扰。
5. 后续层是否保留和加工检索到的信息。

显式 token 检索能力是 Transformer 的强项，也是长上下文任务的基础。

## 12.10 示例到底提供了什么

Min et al. 的研究指出，demonstrations 中真实标签不总是必要。随机替换标签后，在一些分类和多选任务上性能下降没有想象中那么大。

这说明 few-shot 示例提供的不只是“正确样本”。它还提供：

1. Label space。
2. Input distribution。
3. Output format。
4. Task framing。
5. Token pattern。

例如：

```text
评论：...
情感：正面/负面
```

即使示例标签被打乱，模型仍然知道：

```text
这是情感分类任务。
候选标签是正面和负面。
输出应该是一个标签，而不是长篇解释。
```

当然，这不代表标签永远不重要。在需要学习新映射、特殊标签含义、复杂规则或结构化输出时，正确示例会非常重要。

更准确的结论是：

```text
ICL 中示例同时提供格式、分布、标签空间和任务映射；不同任务对这些因素的依赖不同。
```

## 12.11 Induction Head 的直觉

Mechanistic interpretability 中常讨论 induction heads。

一个简化例子：

```text
A -> B
...
A -> ?
```

如果模型在上下文前面看到 `A` 后面跟着 `B`，当后面再次看到 `A` 时，它可能预测 `B`。

这类似一种上下文内复制和归纳机制。

直觉上，induction head 做的事情像：

```text
找到当前 token 在前文中出现的位置。
读取那个位置后面的 token。
把后继 token 用于当前预测。
```

这和 ICL 很相关，因为很多 few-shot 任务都可以看作：

```text
在上下文中找模式 -> 把模式应用到当前输入
```

但要注意，真实模型中的 ICL 不可能只靠一个 induction head 完成。它涉及多层、多头、FFN、residual stream 和训练数据分布。

## 12.12 ICL 作为隐式优化器

另一种解释是把模型看成 meta-optimizer。

Dai et al. 的工作从理论和经验角度讨论：Transformer attention 与梯度下降存在某种 dual form，可以把 ICL 理解成隐式 fine-tuning。

直觉是：

```text
示例相当于临时训练数据。
模型前向计算中产生某种“任务适配”的中间表示。
当前预测基于这些示例产生的临时适配结果。
```

这不是说模型真的执行了标准反向传播。

更稳妥的表达是：

```text
ICL 的行为在某些任务上类似 fine-tuning，但实现机制是在固定参数的前向计算中完成的。
```

面试中不要把“ICL 等于梯度下降”说得过满。应该说：这是一个有启发的理论视角，可以解释部分现象，但 ICL 还有模式匹配、检索、先验记忆等多种机制共同参与。

## 12.13 上下文是临时程序

可以把 prompt 看成一种临时程序。

例如：

```text
你是一个严格的 JSON 转换器。
把输入文本转换成 {"name": ..., "age": ...}。

输入：Alice is 20 years old.
输出：{"name": "Alice", "age": 20}

输入：Bob is 35 years old.
输出：
```

这里上下文提供了：

1. 角色约束。
2. 输出 schema。
3. 示例映射。
4. 当前输入。

模型不是永久学会了这个新 schema，而是在当前上下文里执行这个临时程序。

这也是为什么 agent、tool calling、RAG、workflow 都大量依赖 prompt：它们把运行时状态、工具说明、历史轨迹、候选文档和输出格式都编码成 token。

## 12.14 为什么 ICL 受示例顺序影响

ICL 对示例顺序敏感。

原因包括：

1. Causal LM 从左到右读取上下文。
2. 越靠近输出的位置通常更容易被利用。
3. 示例之间可能产生格式或语义干扰。
4. 模型可能对首尾位置有偏置。
5. 近期示例可能覆盖早期示例。

例如同样几个分类示例，顺序不同，最终预测可能不同。

这说明 ICL 不是理想化的统计学习算法，它受 Transformer 架构、位置编码、训练分布和 prompt 格式共同影响。

工程上常用策略包括：

1. 把最相关示例放在靠近 query 的位置。
2. 保持示例格式一致。
3. 避免标签分布严重偏斜。
4. 使用清晰分隔符。
5. 对候选 prompt 做验证集选择。

## 12.15 为什么 ICL 受干扰信息影响

上下文窗口不是无损数据库。

放入更多 token 可能带来更多证据，也可能带来更多噪声。

干扰信息会影响：

1. Attention 权重竞争。
2. 输出格式判断。
3. 任务边界识别。
4. 标签先验。
5. 多证据整合。
6. 幻觉和错误引用。

例如 RAG 中 top-k 过大，把很多无关 chunk 放进 prompt，模型可能反而回答更差。

所以长上下文不是越长越好。更合理的目标是：

```text
把相关、完整、低噪声、格式清晰的信息放入上下文。
```

## 12.16 Lost in the Middle 与显式检索失败

Lost in the Middle 研究显示，相关信息放在上下文开头或结尾时，模型表现往往更好；放在中间时，性能可能明显下降。

这说明：

```text
token 可见 != token 会被稳定使用
```

对 ICL 和显式 token 检索来说，这非常关键。

模型可能能接收 100K token，但如果证据句在中间位置被忽略，那么长上下文能力并不可靠。

原因可能包括：

1. 训练中长距离精确检索样本不足。
2. 位置编码外推或位置偏置。
3. attention 权重被首尾或近邻信息吸引。
4. 中间信息在多层传播中被稀释。
5. 干扰文档太多。

后续第 13 章会专门展开长距离依赖、精确检索和 lost-in-the-middle。

## 12.17 ICL、RAG 和 Long Context 的组合

现代系统经常把三者结合。

```text
RAG：从外部找文档。
Long context：把更多文档、工具结果、历史轨迹放进 prompt。
ICL：模型在 prompt 内读取、归纳、执行任务。
```

例如企业问答：

```text
用户问题 -> 检索相关制度文档 -> 放入 prompt -> 模型根据文档回答并引用
```

这里模型需要：

1. 读取用户问题。
2. 在上下文文档中找证据。
3. 忽略无关 chunk。
4. 按要求输出答案和引用。
5. 在资料不足时拒答。

这不是单纯“语言生成”，而是上下文内任务执行。

## 12.18 为什么 ICL 是替代架构的重要考题

很多替代架构，例如 SSM、Mamba、RWKV、RetNet，都试图降低 Transformer 的长序列成本。

但它们要替代 Transformer，需要回答一个关键问题：

```text
能否像 full attention 一样，在上下文中精确读取任意位置的示例、证据和绑定关系？
```

Transformer 的 full attention 有天然优势：

```text
任意 token pair 可以直接交互。
```

状态空间或 recurrent 类模型通常更强调压缩历史到固定或有限状态。压缩可以提高效率，但可能损失精确 token-level retrieval。

因此比较架构时，不只看 perplexity 或吞吐，还要看：

1. few-shot ICL。
2. copy/retrieval。
3. passkey retrieval。
4. multi-document QA。
5. 代码变量引用。
6. 工具输出字段读取。
7. 长上下文中间证据利用。

这就是为什么 ICL 和显式 token 检索能力是下一代架构能否取代 Transformer 的关键评估项。

## 12.19 面向专家：ICL 的多机制解释

ICL 不是单一机制。

可以把它拆成多种成分：

1. 格式识别：识别输入输出模板。
2. 标签空间识别：知道候选标签有哪些。
3. 分布适配：理解当前样本来自什么分布。
4. 最近邻检索：找到相似示例。
5. 规则归纳：从示例中归纳映射。
6. 复制机制：把前文某些 token 复制到输出。
7. 先验调用：使用预训练中学到的任务知识。
8. 隐式优化：表现得像在上下文中做临时适配。

不同任务依赖不同机制。

例如：

```text
情感分类：标签空间、格式和语义先验很重要。
人工构造映射：正确示例和绑定关系很重要。
代码补全：局部检索、变量绑定、语法模式很重要。
RAG 问答：证据检索、引用和抗干扰很重要。
```

所以讨论 ICL 时要避免单因解释。

## 12.20 面向专家：显式检索和隐式知识的分工

模型回答问题可能有两类来源。

第一，参数中的隐式知识：

```text
预训练中学到的事实、语言模式、推理模板。
```

第二，上下文中的显式 token：

```text
prompt、文档、示例、工具返回、用户约束。
```

很多失败来自二者冲突。

例如文档说：

```text
退款期限是 14 天。
```

但模型参数中常见模式可能是：

```text
退款期限是 30 天。
```

一个好的 RAG/ICL 模型应该优先使用上下文证据，而不是凭参数先验回答。

这就要求模型具备：

1. 上下文证据定位。
2. 参数知识抑制。
3. 冲突检测。
4. 引用约束遵守。
5. 资料不足拒答。

这也是企业 RAG 和长上下文问答中非常关键的能力。

## 12.21 常见误区

### 误区 1：ICL 就是模型在推理时 fine-tuning

不准确。ICL 没有参数更新。它可能表现得像临时适配，但实现是在固定参数的前向计算中完成。

### 误区 2：Few-shot 示例越多越好

不一定。更多示例可能带来更多格式和分布信息，也可能引入干扰、增加成本、挤占证据空间。

### 误区 3：示例标签错了，ICL 就一定完全失效

不一定。一些任务中 demonstrations 主要提供标签空间、输入分布和格式。但对新映射或复杂规则，正确标签仍然很重要。

### 误区 4：长上下文模型一定能利用所有 token

不一定。Lost in the Middle 表明模型对中间位置证据的利用可能不稳定。

### 误区 5：RAG 检索到正确文档就一定回答正确

不一定。检索只是把文档放进上下文，模型还要在内部 attention 中正确读取、整合和引用。

### 误区 6：ICL 是纯 prompt 技巧，和架构无关

不对。ICL 强依赖 attention、位置编码、上下文长度、训练分布和模型规模。

## 12.22 面试高频问题

### 题 1：什么是 In-Context Learning？

参考回答：

```text
In-Context Learning 指模型在不更新参数的情况下，仅通过 prompt 中的任务说明、示例、标签或证据来临时适配任务并输出答案。它发生在前向推理过程中，模型权重固定，上下文充当临时任务配置、临时数据集和工作记忆。
```

### 题 2：ICL 和 fine-tuning 有什么区别？

参考回答：

```text
Fine-tuning 会用训练样本计算 loss 并通过反向传播更新模型参数，任务能力固化到权重里。ICL 不更新参数，而是把示例和任务说明放进上下文，模型通过 attention 在当前前向计算中利用这些 token。ICL 切换任务快，但受上下文长度、prompt 格式和干扰信息影响。
```

### 题 3：为什么 attention 对 ICL 很重要？

参考回答：

```text
ICL 需要模型从 prompt 中读取相关示例、标签、格式和证据。Self-Attention 让当前预测位置可以直接关注上下文中任意 token，从 few-shot 示例中找到相似输入和对应输出，或从文档中找到证据句。因此 attention 提供了上下文内部的显式 token 检索和信息路由能力。
```

### 题 4：Few-shot 示例提供了什么信息？

参考回答：

```text
Few-shot 示例不只提供正确答案，还提供任务格式、标签空间、输入分布、输出风格和输入输出映射。研究发现某些分类任务中随机替换示例标签仍能保留部分效果，说明格式、标签空间和分布信息很重要。但对于新规则、新标签含义或复杂映射，正确示例仍然关键。
```

### 题 5：ICL 和 RAG 有什么关系？

参考回答：

```text
RAG 是从外部知识库检索文档并放入 prompt，ICL 是模型在上下文中利用这些 token 完成任务。可以说 RAG 提供外部记忆，Transformer attention 在扩展后的上下文中做内部检索和生成。检索到正确文档不等于回答正确，因为模型还要能正确读取、整合和引用证据。
```

### 题 6：什么是显式 token 检索能力？

参考回答：

```text
显式 token 检索能力指模型能在上下文 token 中定位并使用相关信息，例如找到变量定义、few-shot 示例标签、文档证据句、工具返回 JSON 字段或用户约束。这是上下文内部的检索能力，依赖 attention、位置编码、训练分布和模型抗干扰能力。
```

### 题 7：为什么长上下文不等于强 ICL？

参考回答：

```text
长上下文只是让更多 token 可见，不保证模型能稳定利用所有 token。模型可能受位置偏置、干扰信息、训练分布和 attention 竞争影响，忽略中间证据。Lost in the Middle 就显示相关信息放在上下文中间时，模型表现可能下降。因此长上下文能力要看有效利用能力，而不是只看最大长度。
```

### 题 8：为什么 ICL 是替代架构的重要评估项？

参考回答：

```text
Transformer 的 full attention 让任意 token pair 可以直接交互，因此很擅长从上下文中读取示例、证据和绑定关系。很多替代架构为了线性复杂度会压缩历史状态，可能影响精确 token-level retrieval。要替代 Transformer，不仅要看速度和 perplexity，还要验证 few-shot ICL、复制、passkey retrieval、多文档 QA、代码变量引用等能力。
```

## 12.23 小练习

1. 用自己的话解释 zero-shot、one-shot、few-shot 的区别。
2. 设计一个 few-shot prompt，让模型把自然语言转换成 JSON。
3. 比较 ICL、fine-tuning、RAG 的区别。
4. 举例说明 few-shot 示例中的 label space、input distribution、format 分别是什么。
5. 思考为什么随机标签在某些 ICL 任务中仍能保留部分效果。
6. 画出 RAG + ICL 的流程：外部检索、context construction、内部 attention、答案生成。
7. 设计一个测试显式 token 检索能力的小任务，例如从长文本中找 key-value。
8. 阅读 Lost in the Middle 摘要，解释为什么“可见”不等于“会用”。

## 12.24 本章总结

本章讲了 In-Context Learning 与显式 token 检索能力。

核心结论：

1. ICL 是模型在不更新参数的情况下，通过上下文中的说明、示例和证据临时适配任务。
2. Prompt 是运行时任务配置，context window 是模型的临时工作空间。
3. Attention 让模型可以显式读取上下文中的示例、标签、格式、证据和工具结果。
4. Few-shot 示例提供的不只是正确答案，还包括标签空间、输入分布、输出格式和任务 framing。
5. ICL 可能表现得像隐式优化，但它不是标准参数更新。
6. RAG 是外部检索，ICL/attention 是上下文内部检索和使用，两者可以组合。
7. 长上下文不等于强 ICL，模型可能受位置偏置、干扰信息和 lost-in-the-middle 影响。
8. 显式 token 检索能力是评估 Transformer 和替代架构的重要维度。

下一章会进入长距离依赖、精确检索与 lost-in-the-middle，进一步分析为什么模型能看到长上下文却不一定能稳定用好长上下文。
