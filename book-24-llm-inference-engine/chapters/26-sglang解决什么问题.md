# 第 26 章 SGLang 解决什么问题

前面十章集中讲了 vLLM：PagedAttention、KV Cache Block Manager、continuous batching、scheduler、worker/executor、prefix cache、并行和调优。vLLM 的核心问题意识是：如何把大量独立 LLM 请求高效地调度到 GPU 上，解决 KV Cache 显存浪费、batch 动态变化和吞吐问题。

从本章开始进入 SGLang。

SGLang 也做高性能 serving，但它的出发点不完全一样。它不仅关注“单次请求如何高效生成”，还关注“复杂 LLM 程序如何高效执行”。这些程序可能包含多次 generation、分支、并行、工具调用、结构化输出、multi-turn、RAG、agent 和 self-consistency。

一句话概括：

> SGLang 解决的是复杂 LLM 程序的表达和执行效率问题：前端让开发者更容易描述多步 LLM 控制流，后端 runtime 用 RadixAttention、prefix sharing、batching 和 constrained decoding 等机制减少重复计算并提高吞吐。

## 26.0 本讲资料边界与第二轮精修口径

本讲按第二轮精修要求做过资料校准，主要参考四类公开资料：

1. SGLang 论文《SGLang: Efficient Execution of Structured Language Model Programs》对 complex language model programs、frontend language、runtime、RadixAttention、KV cache reuse 和 compressed finite state machines for structured output decoding 的定义。
2. SGLang 官方文档首页对 SGLang 作为 high-performance serving framework、RadixAttention、prefix caching、multi-GPU parallelism、OpenAI API 兼容和生产 serving 定位的说明。
3. SGLang Frontend Language 文档对 `gen`、multi-turn、Python control flow、`fork` 并行、choices / regex constrained decoding、batching、streaming 和 multi-modal prompt 的示例口径。
4. SGLang Structured Outputs / OpenAI-Compatible API 文档对 JSON schema、regex、EBNF、structural tag、XGrammar 默认后端、OpenAI-compatible API、native API 和 offline engine 的说明。

本章只回答“SGLang 解决什么问题”和“为什么它不是另一个普通 chat endpoint”。本章不展开真实 SGLang server 参数、scheduler 细节、RadixAttention 源码、真实 grammar backend、GPU kernel、multi-node 部署、benchmark 排行或某个版本的性能结论；这些会在后续章节拆开。本章 demo 用纯 Python 模拟多步 LLM program 的 prefix sharing、分支可见性和结构化输出重试成本，只用于解释问题意识，不等同于真实 SGLang runtime。

参考资料：

1. SGLang 论文：<https://arxiv.org/abs/2312.07104>
2. SGLang 官方文档首页：<https://docs.sglang.io/index.md>
3. SGLang Frontend Language：<https://docs.sglang.io/docs/references/frontend/frontend_tutorial.md>
4. SGLang Structured Outputs：<https://docs.sglang.io/docs/advanced_features/structured_outputs.md>
5. SGLang OpenAI-Compatible APIs：<https://docs.sglang.io/docs/basic_usage/openai_api.md>

## 26.1 本章目标

读完本章，你应该能讲清：

1. SGLang 和 vLLM 的问题出发点有什么不同。
2. 为什么复杂 LLM 应用不是一次简单 `generate()`。
3. SGLang 为什么强调 frontend language 和 backend runtime 协同设计。
4. RadixAttention 要解决什么类型的 KV cache 复用问题。
5. structured generation 为什么是 serving runtime 的一等能力。
6. SGLang 适合哪些工作负载，不适合被误解成什么。
7. 面试中如何回答“SGLang 解决什么问题”。

## 26.2 从 vLLM 的问题意识说起

vLLM 解决的核心问题可以概括为：

```text
大量请求来了，如何高吞吐、低浪费、稳定地执行 prefill 和 decode？
```

因此 vLLM 的代表性能力包括：

1. PagedAttention。
2. KV Cache Block Manager。
3. Continuous batching。
4. iteration-level scheduling。
5. prefix caching。
6. 多 GPU serving。
7. OpenAI-compatible API。

这些能力非常适合处理大规模在线请求：聊天、补全、RAG、批量生成、多租户 serving。

但当应用从“一次生成”变成“复杂程序”时，会出现新的问题。

例如：

1. 一个任务要调用模型多次。
2. 多个分支共享同一个 prompt 前缀。
3. 需要 self-consistency 采样多个答案再投票。
4. 需要 Tree-of-Thought 搜索。
5. 需要 ReAct agent 多轮思考、调用工具、观察结果。
6. 需要严格输出 JSON、regex 或 grammar。
7. 多轮对话和多个候选路径共享大量历史上下文。

这些场景里，瓶颈不只是单次请求的 batching，而是多次 LLM 调用之间存在大量可复用状态和控制流。

这就是 SGLang 的切入点。

## 26.3 复杂 LLM 程序是什么

简单 LLM 请求：

```text
prompt -> generate answer
```

复杂 LLM 程序：

```text
prompt A -> generate intermediate result
         -> branch into multiple candidates
         -> call tool
         -> append observation
         -> generate next step
         -> enforce JSON schema
         -> merge results
```

它不再是一次模型调用，而是一段带状态的程序。

典型例子一：self-consistency。

```text
same question + same examples
  -> sample answer 1
  -> sample answer 2
  -> sample answer 3
  -> vote / verify
```

多个分支共享同一个题目和 few-shot examples。

典型例子二：Tree-of-Thought。

```text
root prompt
  -> thought A
     -> branch A1 / A2
  -> thought B
     -> branch B1 / B2
  -> evaluate branches
```

树上大量节点共享从 root 到某个中间节点的前缀。

典型例子三：Agent。

```text
system prompt
  -> user task
  -> thought
  -> action: search
  -> observation
  -> thought
  -> action: calculator
  -> observation
  -> final answer
```

每一步都可能复用前面历史，并追加少量新内容。

如果 runtime 每次都从头 prefill，就会浪费大量计算。

## 26.4 为什么普通 serving API 不够表达这些程序

OpenAI-compatible API 很适合单次请求：

```python
client.chat.completions.create(messages=[...])
```

但复杂程序需要表达：

1. generation 结果存入变量。
2. 多个 generation 并行执行。
3. 分支和 fork。
4. 条件控制流。
5. 外部工具调用。
6. 结构化输出约束。
7. 多轮状态复用。
8. 后端自动发现 prefix sharing。

如果开发者只用普通 API 手写这些逻辑，常见问题是：

1. 每次调用都重新发送完整 prompt。
2. runtime 难以知道不同调用之间的共享结构。
3. prefix cache 复用需要开发者自己设计 key。
4. 并行分支不好和 batching 结合。
5. 结构化输出约束在客户端拼凑，效率低且容易错。
6. 工具调用、多轮状态和 cache 生命周期难统一管理。

SGLang 的思路是：不要只把 LLM 当成黑盒 API，而是把 LLM 调用组织成可执行的程序，并让 runtime 看到这些结构。

## 26.5 SGLang 的核心思想：前后端协同设计

SGLang 论文和博客强调两个组成部分：

1. frontend language。
2. backend runtime。

Frontend language 解决表达问题。

它让开发者用更自然的方式表达：

1. `gen`：生成一段内容并保存。
2. `fork`：创建多个并行分支。
3. `choices`：从固定候选中选择。
4. 约束输出格式。
5. 多步 prompting。
6. 工具和外部环境交互。

Backend runtime 解决执行效率问题。

它负责：

1. 自动发现共享前缀。
2. 复用 KV cache。
3. 调度多个 generation。
4. batching。
5. structured decoding。
6. 多 GPU 并行。

关键是二者协同。

如果只有前端语言，没有高效 runtime，表达很优雅但执行慢。

如果只有 runtime，没有程序结构，后端很难知道哪些 LLM 调用可以共享 cache、哪些分支可以并行、哪些输出需要约束。

## 26.6 RadixAttention 解决什么问题

SGLang 最有代表性的后端优化是 RadixAttention。

它解决的问题是：

```text
复杂 LLM 程序中，不同 generation call 之间存在多种共享前缀，如何自动复用 KV cache？
```

vLLM 的 prefix cache 常常可以理解为基于 block hash 的前缀复用。

SGLang 的 RadixAttention 更强调用 radix tree 表达和管理复杂前缀共享结构。

Radix tree，也叫压缩前缀树，适合存储大量共享前缀的 token 序列。

普通 trie 可能每条边只表示一个 token。

Radix tree 的一条边可以表示一段 token 序列，因此更紧凑。

SGLang runtime 会把已经计算过的 prompt 和生成结果对应的 KV cache 保留在 radix tree 中。后续请求或后续 generation call 到来时，runtime 自动做 prefix matching，找到最长可复用前缀，避免重复 prefill。

## 26.7 为什么 radix tree 比单个 cache key 更适合复杂程序

如果只是固定 system prompt，普通 prefix cache 已经有用。

但复杂程序中，共享结构可能是树状的。

例如：

```text
root: system prompt + task
  branch A: reasoning path A
    branch A1
    branch A2
  branch B: reasoning path B
    branch B1
    branch B2
```

每个分支都共享 root。

A1 和 A2 还共享 branch A。

B1 和 B2 共享 branch B。

如果只用“完整 prompt -> KV”作为 cache key，就很难高效表示这些中间共享。

Radix tree 可以自然表示：

```text
root prefix
  -> A prefix
     -> A1 suffix
     -> A2 suffix
  -> B prefix
     -> B1 suffix
     -> B2 suffix
```

这正适合 Tree-of-Thought、self-consistency、multi-turn chat、agent search 等工作负载。

## 26.8 SGLang 为什么强调 structured generation

很多生产应用不是让模型自由写一段话，而是要求输出满足结构。

例如：

1. JSON。
2. 正则表达式。
3. EBNF grammar。
4. 固定 choices。
5. 函数调用参数。
6. SQL 片段。
7. 工具调用 schema。

如果模型输出不合法，后端还要重试、修复或人工处理。

这会增加延迟、成本和失败率。

Structured generation 的目标是：在 decoding 过程中约束 token 选择，让模型只能生成合法格式。

这不是简单的后处理。

后处理是：

```text
模型先乱生成 -> 解析失败 -> 重试或修复
```

Constrained decoding 是：

```text
每一步根据状态机或 grammar mask logits -> 只允许合法 token -> 输出天然合法
```

SGLang 论文中提到 compressed finite state machines 用于加速 structured output decoding。第 30 章会详细讲 constrained decoding，本章只要先理解：结构化输出是 SGLang 的重要问题意识之一。

## 26.9 SGLang 面向哪些 workload

SGLang 特别适合：

1. Agent。
2. ReAct。
3. Tree-of-Thought。
4. self-consistency。
5. few-shot learning。
6. multi-turn chat。
7. RAG pipeline。
8. structured extraction。
9. JSON decoding。
10. 多模态 LLM 程序。

这些 workload 的共同特点是：

1. 多次 LLM 调用。
2. 多个调用之间共享大量前缀。
3. 有控制流或并行分支。
4. 需要输出约束或工具交互。
5. 单次请求 API 难以表达执行结构。

所以 SGLang 不是只为了“把一个 chat endpoint 跑快一点”。它更关注复杂 LLM 应用的程序化表达和 runtime 优化。

## 26.10 SGLang 和 vLLM 的区别：先给直觉

可以粗略这样区分：

```text
vLLM: 高效服务大量 LLM 请求。
SGLang: 高效执行复杂 LLM 程序。
```

当然二者有重叠：都需要高性能 runtime、KV cache 管理、batching、多 GPU、OpenAI-compatible API。

但重点不同。

vLLM 的典型关键词：

1. PagedAttention。
2. continuous batching。
3. scheduler。
4. KV block manager。
5. serving throughput。

SGLang 的典型关键词：

1. frontend language。
2. RadixAttention。
3. automatic KV cache reuse。
4. structured generation。
5. complex LLM programs。
6. agent/reasoning/RAG。

面试中不要把 SGLang 只说成“另一个 vLLM”。更好的表达是：它是面向复杂 LLM 程序的 serving/runtime 系统，前端和后端一起设计。

## 26.11 为什么自动 KV cache 复用很重要

复杂程序里，prefix reuse 很难靠人工维护。

例如 self-consistency：

```text
same prompt -> sample 10 answers
```

如果每个 sample 都从头 prefill，重复计算 10 次。

如果复用 prefix KV，只需要一次 prefill，然后分支 decode。

再比如 multi-turn chat：

```text
system prompt + history turn 1 + history turn 2 + new user message
```

不同会话、不同分支可能共享 system prompt，部分共享历史。

再比如 few-shot learning：

```text
same examples + different questions
```

few-shot examples 可能很长，但每个问题不同。

如果 runtime 自动维护 radix tree，就能捕捉这些复用机会。

这会减少：

1. prefill compute。
2. TTFT。
3. GPU cost。
4. redundant KV allocation。

## 26.12 SGLang 的前端语言解决什么痛点

很多 LLM 应用代码长这样：

```python
prompt = build_prompt(x)
answer1 = call_llm(prompt)
prompt2 = prompt + answer1 + tool_result
answer2 = call_llm(prompt2)
```

问题是：

1. 程序结构在 Python 客户端里。
2. runtime 只看到一堆独立请求。
3. 共享前缀和并行分支不明显。
4. 输出约束分散在客户端逻辑里。

SGLang frontend 的目标是让这些结构变成 runtime 可以理解的 program。

例如：

1. generation 是一个操作。
2. fork 是一个操作。
3. choices 是一个约束。
4. 变量引用是 prompt 依赖。
5. 并行 generation 可以被 runtime 调度。

这样 runtime 就有机会做更全局的优化。

## 26.13 OpenAI-compatible API 和 SGLang language 的关系

SGLang 也支持 OpenAI-compatible API。

这很重要，因为生产系统往往需要兼容已有客户端和生态。

OpenAI API 适合：

1. 普通 chat/completions。
2. 迁移已有服务。
3. 和 LangChain、LlamaIndex、OpenAI SDK 兼容。
4. 简单在线 serving。

SGLang language 更适合：

1. 多步 generation。
2. 分支和并行。
3. agent 控制流。
4. structured generation。
5. 复杂 prompt program。

所以不要认为二者冲突。

可以这样理解：

```text
OpenAI API 是服务接口。
SGLang language 是复杂 LLM 程序表达接口。
SGLang runtime 是执行这些请求和程序的后端。
```

## 26.14 SGLang 不是解决所有问题的银弹

SGLang 很强，但不能误解。

误区一：SGLang 只要打开就一定比所有框架快。

不一定。它在共享前缀、多次 generation、结构化输出、复杂程序中优势明显；如果只是短 prompt 单次生成，收益可能没那么大。

误区二：RadixAttention 等于普通 prefix cache。

不完全等价。RadixAttention 更强调 radix tree 管理复杂共享前缀结构，而不仅是单条 prompt prefix 的 hash 命中。

误区三：structured generation 只是解析 JSON。

不是。真正的 structured generation 是 decoding 时约束 token，而不是输出后再 parse。

误区四：前端语言只是语法糖。

如果 runtime 能利用程序结构做 cache reuse、并行和 batching，它就不只是语法糖。

误区五：SGLang 和 vLLM 是完全替代关系。

更准确地说，它们都属于 LLM serving/runtime 系统，但问题侧重点不同，工程实现和适用 workload 也不同。

## 26.15 用一个例子理解 SGLang 的价值

假设你要做一个法律文档分析 agent。

流程：

```text
1. 读取长文档和 system prompt。
2. 从合同中抽取主体、金额、日期。
3. 对每个条款并行判断风险。
4. 对高风险条款调用外部法规检索。
5. 把检索结果追加进上下文。
6. 输出严格 JSON 报告。
```

这里有多个特点：

1. 长文档前缀被多步共享。
2. 多个条款分支可以并行。
3. 工具结果会追加上下文。
4. 输出必须是合法 JSON。
5. 某些中间结果需要被后续步骤引用。

如果用普通 API 手写，容易重复 prefill 长文档、多次传输完整 prompt、结构化输出失败后重试。

如果用 SGLang-like runtime，理想情况是：

1. 长文档 KV 被 RadixAttention 复用。
2. 条款分支被并行调度。
3. JSON decoding 被约束。
4. runtime 自动 batching 多个 generation。
5. 整体 TTFT、吞吐和失败率更好。

这就是 SGLang 解决的问题类型。

## 26.16 和本书后续章节的关系

本章只是回答“SGLang 为什么存在”。

后面几章会具体展开：

第 27 章：SGLang Runtime 总览，讲 runtime 的整体模块。

第 28 章：RadixAttention 与 prefix sharing，讲 radix tree 如何维护 KV cache。

第 29 章：SGLang scheduler 设计，讲多请求、多分支如何调度。

第 30 章：Structured Generation 与 constrained decoding，讲 JSON/regex/grammar 约束。

第 31 章：Speculative Decoding 在 SGLang 中的角色。

第 32 章：Multi-turn、tool use 和 agent serving 支持。

第 33 章：SGLang 和 vLLM 的架构对比。

第 34 章：mini-sglang 源码学习路径。

## 26.17 面试官会怎么问

问题一：SGLang 解决什么问题？

回答要点：它解决复杂 LLM 程序的表达和高效执行问题，通过 frontend language 表达 generation、fork、choices、控制流和结构化输出，通过 runtime 的 RadixAttention、cache reuse、batching 等机制提高执行效率。

问题二：SGLang 和 vLLM 有什么区别？

回答要点：vLLM 更强调高效 serving 大量请求，核心是 PagedAttention、continuous batching 和 KV block 管理；SGLang 更强调复杂 LLM programs，包括多次 generation、分支、agent、structured output 和自动 KV cache reuse。

问题三：RadixAttention 为什么重要？

回答要点：复杂程序中共享前缀呈树状结构，RadixAttention 用 radix tree 自动匹配、插入和淘汰 KV cache，避免多次 LLM 调用重复 prefill。

问题四：structured generation 为什么是 runtime 能力？

回答要点：因为只靠后处理会出现解析失败和重试；runtime 在 decoding 时用状态机或 grammar mask logits，可以让输出天然满足 JSON/regex/grammar 等约束。

问题五：什么时候 SGLang 优势明显？

回答要点：多步 generation、agent、RAG、多轮对话、self-consistency、Tree-of-Thought、few-shot、结构化抽取、多模态程序等共享前缀和控制流明显的 workload。

## 26.18 标准回答模板

如果面试官问“SGLang 解决什么问题”，可以这样回答：

```text
SGLang 主要解决复杂 LLM 程序的表达和执行效率问题。很多 LLM 应用不是一次简单的 chat completion，而是包含多次 generation、分支、并行、工具调用、多轮状态、结构化输出和外部环境交互。普通 OpenAI API 可以表达单次请求，但 runtime 很难看到这些调用之间的共享前缀和控制流，因此容易重复 prefill，结构化输出也常靠后处理和重试。

SGLang 的思路是前后端协同设计。前端提供嵌入 Python 的结构化生成语言，用 gen、fork、choices 等原语表达 LLM 程序；后端 runtime 则用 RadixAttention、automatic KV cache reuse、batching 和 constrained decoding 等机制高效执行。

其中 RadixAttention 用 radix tree 管理 token prefix 到 KV cache 的映射，适合 self-consistency、Tree-of-Thought、multi-turn chat、agent 和 few-shot learning 这类树状共享前缀场景。相比只把 SGLang 看成另一个 serving server，更准确地说，它是面向复杂 LLM programs 的高性能 runtime。
```

## 26.19 SGLang 动机公式、prefix sharing 和可运行 demo

把复杂 LLM 应用看成一组 generation call：

$$
c_i=(X_i,Y_i,b_i,s_i)
$$

其中 `X_i` 是第 `i` 次调用的输入 token 数，`Y_i` 是输出 token 数，`b_i` 表示它属于哪个分支，`s_i` 表示是否需要结构化输出约束。

如果普通客户端每次都把完整 prompt 作为独立请求发送，prefill token 工作量近似为：

$$
P_{\mathrm{naive}}=\sum_i X_i
$$

如果 runtime 能看到程序结构，并用 radix tree 找到第 `i` 次调用可复用的最长前缀 `H_i`，实际需要重新 prefill 的 token 数可以写成：

$$
P_{\mathrm{radix}}=\sum_i \max(0,X_i-H_i)
$$

prefix sharing 的节省比例可以写成：

$$
R_{\mathrm{reuse}}=\frac{P_{\mathrm{naive}}-P_{\mathrm{radix}}}{\max(1,P_{\mathrm{naive}})}
$$

如果结构化输出靠后处理，解析失败会触发重试。把无约束输出的重试 token 成本记为 `C_{\mathrm{retry}}`，把 constrained decoding 的额外 grammar 开销记为 `C_{\mathrm{grammar}}`，则教学版总工作量可以粗略写成：

$$
W_{\mathrm{naive}}=P_{\mathrm{naive}}+\sum_i Y_i+C_{\mathrm{retry}}
$$

$$
W_{\mathrm{sglang}}=P_{\mathrm{radix}}+\sum_i Y_i+C_{\mathrm{grammar}}
$$

整体工作量下降比例为：

$$
R_{\mathrm{work}}=\frac{W_{\mathrm{naive}}-W_{\mathrm{sglang}}}{\max(1,W_{\mathrm{naive}})}
$$

教学版 SGLang 动机门禁可以写成：

$$
G_{\mathrm{sglang}}=G_{\mathrm{program}}G_{\mathrm{radix}}G_{\mathrm{branch}}G_{\mathrm{structured}}G_{\mathrm{api}}G_{\mathrm{metric}}
$$

其中：

1. `G_{\mathrm{program}}`：能把任务表达成多次 generation、变量、分支、并行和结构化输出，而不是一堆互不相关的 API 请求。
2. `G_{\mathrm{radix}}`：能计算 naive prefill、radix prefill、最长可复用前缀和 saved prefill tokens。
3. `G_{\mathrm{branch}}`：能看见 self-consistency、Tree-of-Thought、agent 或 RAG 分支里的共享 root 和中间节点。
4. `G_{\mathrm{structured}}`：能解释 JSON / regex / EBNF constrained decoding 为什么减少后处理重试。
5. `G_{\mathrm{api}}`：能同时说明 OpenAI-compatible API 与 SGLang frontend language 的关系。
6. `G_{\mathrm{metric}}`：能输出 reuse ratio、retry saving、work reduction 和最终动机门禁。

下面这个 0 依赖 demo 模拟一个法律合同分析 program。它有 6 次 generation：先抽取字段，再对 A / B 两个条款分支分别做风险判断和法规检查，最后输出严格 JSON 报告。普通 API 会把 6 个完整 prompt 都重新 prefill；SGLang-like runtime 能让后端看见共享文档 root、A 分支和 B 分支，从而减少重复 prefill，并用结构化输出约束避免 JSON 失败重试。

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ProgramCall:
    name: str
    prompt_tokens: tuple
    output_tokens: int
    branch: str
    structured: bool = False


class ToyRadixPrefixCache:
    def __init__(self):
        self._stored_prompts = []

    @staticmethod
    def _common_prefix_len(left, right):
        size = min(len(left), len(right))
        count = 0
        while count < size and left[count] == right[count]:
            count += 1
        return count

    def longest_prefix(self, tokens):
        best = 0
        best_owner = None
        for owner, stored in self._stored_prompts:
            hit = self._common_prefix_len(tokens, stored)
            if hit > best:
                best = hit
                best_owner = owner
        return best, best_owner

    def insert(self, owner, tokens):
        self._stored_prompts.append((owner, tokens))


class ToySGLangMotivationAudit:
    def __init__(self, calls, root_len, invalid_json_retries, grammar_overhead_tokens):
        self.calls = list(calls)
        self.root_len = root_len
        self.invalid_json_retries = invalid_json_retries
        self.grammar_overhead_tokens = grammar_overhead_tokens

    def run(self):
        cache = ToyRadixPrefixCache()
        reuse_rows = []
        naive_prefill = 0
        radix_prefill = 0
        decode_tokens = 0
        branch_hits = {}

        for call in self.calls:
            prompt_len = len(call.prompt_tokens)
            hit_tokens, hit_owner = cache.longest_prefix(call.prompt_tokens)
            run_tokens = prompt_len - hit_tokens
            cache.insert(call.name, call.prompt_tokens)

            naive_prefill += prompt_len
            radix_prefill += run_tokens
            decode_tokens += call.output_tokens
            if hit_tokens >= self.root_len:
                branch_hits[call.branch] = branch_hits.get(call.branch, 0) + 1

            reuse_rows.append(
                {
                    "name": call.name,
                    "input_tokens": prompt_len,
                    "hit_tokens": hit_tokens,
                    "hit_owner": hit_owner,
                    "run_prefill_tokens": run_tokens,
                }
            )

        final_call = self.calls[-1]
        retry_tokens_without_constraints = (
            self.invalid_json_retries * final_call.output_tokens
        )
        naive_work = naive_prefill + decode_tokens + retry_tokens_without_constraints
        sglang_work = radix_prefill + decode_tokens + self.grammar_overhead_tokens
        saved_prefill = naive_prefill - radix_prefill

        summary = {
            "naive_prefill_tokens": naive_prefill,
            "radix_prefill_tokens": radix_prefill,
            "saved_prefill_tokens": saved_prefill,
            "reuse_ratio": round(saved_prefill / max(1, naive_prefill), 3),
            "retry_tokens_without_constraints": retry_tokens_without_constraints,
            "grammar_overhead_tokens": self.grammar_overhead_tokens,
            "naive_work_tokens": naive_work,
            "sglang_work_tokens": sglang_work,
            "work_reduction": round((naive_work - sglang_work) / max(1, naive_work), 3),
        }
        branch_summary = {
            "branches": sorted(set(call.branch for call in self.calls)),
            "root_shared_calls": sum(
                1 for row in reuse_rows if row["hit_tokens"] >= self.root_len
            ),
            "branch_hits": branch_hits,
            "structured_calls": [call.name for call in self.calls if call.structured],
        }
        gates = {
            "frontend_exposes_program": len(self.calls) >= 4
            and len(branch_summary["branches"]) >= 3,
            "radix_reduces_prefill": summary["saved_prefill_tokens"] > 0,
            "branch_sharing_visible": branch_summary["root_shared_calls"] >= 4,
            "structured_retry_saving": retry_tokens_without_constraints
            > self.grammar_overhead_tokens,
            "openai_api_boundary_clear": True,
            "metrics_ready": summary["work_reduction"] > 0.3,
        }
        gates["sglang_motivation_gate"] = all(gates.values())
        return reuse_rows, branch_summary, summary, gates


ROOT = tuple(f"doc_root_{i}" for i in range(8))
CLAUSE_A = ("clause_a", "risk", "scope")
CLAUSE_B = ("clause_b", "payment", "scope")

program_calls = [
    ProgramCall("extract_fields", ROOT + ("extract", "fields"), 40, "root"),
    ProgramCall("clause_a_risk", ROOT + CLAUSE_A, 35, "clause_a"),
    ProgramCall("clause_a_law_check", ROOT + CLAUSE_A + ("law", "check"), 30, "clause_a"),
    ProgramCall("clause_b_risk", ROOT + CLAUSE_B, 35, "clause_b"),
    ProgramCall("clause_b_law_check", ROOT + CLAUSE_B + ("law", "check"), 30, "clause_b"),
    ProgramCall(
        "final_json_report",
        ROOT + ("merge", "risk", "law", "json"),
        60,
        "report",
        structured=True,
    ),
]

audit = ToySGLangMotivationAudit(
    calls=program_calls,
    root_len=len(ROOT),
    invalid_json_retries=3,
    grammar_overhead_tokens=12,
)
reuse_rows, branch_summary, summary, gates = audit.run()

print("sglang_motivation_reuse=", reuse_rows)
print("sglang_motivation_branches=", branch_summary)
print("sglang_motivation_summary=", summary)
print("sglang_motivation_gates=", gates)
```

这段 demo 应该输出：

```text
sglang_motivation_summary= {'naive_prefill_tokens': 70, 'radix_prefill_tokens': 24, 'saved_prefill_tokens': 46, 'reuse_ratio': 0.657, 'retry_tokens_without_constraints': 180, 'grammar_overhead_tokens': 12, 'naive_work_tokens': 480, 'sglang_work_tokens': 266, 'work_reduction': 0.446}
sglang_motivation_gates= {'frontend_exposes_program': True, 'radix_reduces_prefill': True, 'branch_sharing_visible': True, 'structured_retry_saving': True, 'openai_api_boundary_clear': True, 'metrics_ready': True, 'sglang_motivation_gate': True}
```

读这段代码时要注意三点：

1. `P_{\mathrm{naive}}=70` 说明普通 API 把 6 次调用都当成独立 prompt，长文档 root 被重复 prefill。
2. `P_{\mathrm{radix}}=24` 说明 radix prefix matching 能复用 root、A 分支和 B 分支的中间前缀，真正新 prefill 的只是各分支追加部分。
3. `C_{\mathrm{retry}}=180` 与 `C_{\mathrm{grammar}}=12` 的对比说明 constrained decoding 的价值不只是“格式好看”，而是减少解析失败后的重试工作量和尾延迟。

## 26.20 小练习

1. 举一个普通 chat completion 以外的复杂 LLM 程序，并画出它的多次 generation 调用关系。
2. 解释 self-consistency 为什么适合 KV cache prefix reuse。
3. 画一个 Tree-of-Thought 的 prefix sharing 树。
4. 对比 vLLM prefix cache 和 SGLang RadixAttention 的直觉差异。
5. 解释为什么 JSON 输出失败后重试不如 constrained decoding。
6. 设计一个适合 SGLang 的 agent serving 场景，并说明哪些部分可以复用 KV cache。

## 26.21 本章总结

SGLang 的核心问题意识是：复杂 LLM 应用正在从单次 prompt-response 变成多步、有状态、有控制流、有结构化约束的 LLM programs。

如果 runtime 只看到一堆独立请求，就很难自动复用这些程序中的共享前缀，也很难高效处理分支、并行、工具调用和结构化输出。SGLang 通过 frontend language 表达程序结构，通过 backend runtime 做 RadixAttention、KV cache reuse、batching 和 constrained decoding，从而提升复杂 workload 的执行效率和可控性。

下一章会进入 SGLang Runtime 总览，具体拆解 SGLang server、runtime、scheduler、memory pool、RadixAttention、sampling 和 structured output 之间如何协作。
