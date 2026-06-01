# 第 33 章 SGLang 和 vLLM 的架构对比

上一章讲了 Multi-turn、tool use 和 agent serving：SGLang 的重点不是只服务一次 chat completion，而是高效执行多轮、有状态、有工具、有分支的复杂 LLM programs。

到这里，本书第三部分和第四部分已经分别讲完了 vLLM 和 SGLang 的核心主线。

vLLM 主线是：PagedAttention、KV Cache Block Manager、continuous batching、scheduler、worker/executor、prefix cache、并行和调优。

SGLang 主线是：frontend/runtime 协同、RadixAttention、SGLang Runtime、scheduler、structured generation、speculative decoding、tool use 和 agent serving。

本章做一次系统对比。

一句话概括：

> vLLM 更像面向通用高并发 LLM serving 的高性能执行引擎，核心是 KV cache 分页管理和 continuous batching；SGLang 更像面向复杂 LLM programs 的高性能 runtime，核心是前后端协同、RadixAttention、structured output 和多步程序执行。

## 33.1 本章目标

读完本章，你应该能讲清：

1. vLLM 和 SGLang 的共同底层问题是什么。
2. 二者的问题出发点有什么不同。
3. PagedAttention 和 RadixAttention 的直觉差异。
4. 两者 scheduler 都要解决什么，SGLang 额外考虑什么。
5. structured output、tool use、agent workload 为什么更像 SGLang 的主战场。
6. 工程选型时如何判断更适合 vLLM、SGLang 或组合方案。
7. 面试中如何避免把二者说成简单替代关系。

## 33.2 共同底层问题

先不要急着区分。

vLLM 和 SGLang 都在解决同一个大问题：

```text
如何在 GPU 上高效服务大语言模型自回归生成？
```

它们共同面对：

1. Prefill 和 decode 分离。
2. KV cache 显存增长。
3. 长短请求混合。
4. batch 动态变化。
5. TTFT、TPOT、吞吐之间的权衡。
6. Streaming 输出。
7. Sampling 参数。
8. 多 GPU 并行。
9. 线上稳定性和监控。

所以两者不是完全不同物种。

从 serving engine 基础能力看，它们都有类似模块：

```text
API server
  -> request parsing
  -> tokenizer
  -> scheduler
  -> KV cache manager
  -> model runner
  -> sampler
  -> output streamer
```

差别在于：它们对“最重要的问题”排序不同。

## 33.3 问题出发点对比

vLLM 的核心问题意识：

```text
大量请求来了，如何高吞吐、低显存浪费、稳定地执行 prefill/decode？
```

SGLang 的核心问题意识：

```text
复杂 LLM 程序来了，如何表达多步控制流，并高效复用其中的共享状态？
```

因此 vLLM 更强调：

1. PagedAttention。
2. KV block 管理。
3. Continuous batching。
4. OpenAI-compatible serving。
5. 通用在线推理吞吐。

SGLang 更强调：

1. Frontend language。
2. Runtime 执行复杂 LLM program。
3. RadixAttention。
4. Structured generation。
5. Multi-turn、tool use、agent serving。

这不是谁更先进的问题，而是设计重心不同。

## 33.4 一张总表

先用表建立整体印象。

| 维度 | vLLM | SGLang |
| --- | --- | --- |
| 核心定位 | 通用高性能 LLM serving engine | 面向复杂 LLM programs 的高性能 runtime |
| 代表机制 | PagedAttention | RadixAttention |
| 核心痛点 | KV cache 显存碎片、动态 batching | 多次 generation 之间的 prefix sharing 和控制流 |
| API 生态 | OpenAI-compatible API 强 | OpenAI-compatible、native API、offline engine、frontend language |
| Cache 重点 | block-based KV cache 管理 | radix tree 管理复杂 prefix sharing |
| Scheduler 重点 | prefill/decode、token/KV budget、continuous batching | 以上 + cache hit length、复杂程序分支、grammar/tool workload |
| Structured output | 支持能力逐步增强 | 是核心问题意识之一 |
| Agent workload | 可以服务，但不是最初核心表达层 | 更强调 multi-turn、tool use、agent serving |
| 学习价值 | 理解现代 serving engine 基础 | 理解复杂 LLM program runtime |
| 典型场景 | 通用 chat、completion、RAG、高并发 serving | agent、self-consistency、ToT、structured extraction、多轮复杂程序 |

这张表是直觉，不是绝对边界。两个项目都在快速演进，很多能力会互相靠近。

## 33.5 PagedAttention vs RadixAttention

这两个名字很容易混在一起。

PagedAttention 解决的核心问题是：

```text
KV cache 如何像分页内存一样管理，减少显存碎片和连续分配要求？
```

它关注的是物理内存布局：

```text
logical token blocks -> physical KV blocks
```

RadixAttention 解决的核心问题是：

```text
不同请求或 generation call 的 token prefix 如何自动匹配和复用？
```

它关注的是 prefix 共享结构：

```text
token prefix tree -> cached KV references
```

可以这样记：

```text
PagedAttention: KV cache 怎么放
RadixAttention: 哪些 token prefix 的 KV cache 可以复用
```

当然，真实系统里两者都需要和底层 KV layout 配合。RadixAttention 找到了可复用 prefix，也要依赖 memory pool 或 paged layout 来引用实际 KV。

## 33.6 Prefix cache 对比

vLLM 的 prefix caching 常见直觉是基于 block hash：

```text
block_hash = hash(parent_hash, block_tokens, extra_hashes)
```

新请求从左到右匹配 full blocks，命中就复用对应 KV blocks。

这种方式适合大量请求共享固定前缀，例如：

```text
[system prompt][RAG document][question A]
[system prompt][RAG document][question B]
```

SGLang 的 RadixAttention 更强调树状共享：

```text
root prompt
  -> branch A
      -> branch A1
      -> branch A2
  -> branch B
      -> branch B1
      -> branch B2
```

它适合 self-consistency、Tree-of-Thought、agent search、多轮 chat 等复杂结构。

对比：

| 维度 | vLLM prefix cache | SGLang RadixAttention |
| --- | --- | --- |
| 基本结构 | block hash | radix tree |
| 复用单位 | full KV blocks | token prefix path 对应 KV |
| 典型共享 | 线性前缀 | 树状前缀 |
| 代表场景 | 固定 system/RAG 前缀 | 多分支、多轮、多 generation |
| 关注点 | block 命中和生命周期 | longest prefix match、split、insert、evict |

## 33.7 Scheduler 对比

两者 scheduler 都要处理：

1. Waiting queue。
2. Running requests。
3. Prefill 和 decode 混合。
4. Token budget。
5. Sequence budget。
6. KV cache budget。
7. Streaming。
8. Abort 和 cleanup。

vLLM scheduler 的典型问题是：

```text
本轮哪些 waiting 请求 prefill？
哪些 running 请求 decode？
KV blocks 是否够？
如何控制 batch tokens 和 sequence 数？
```

SGLang scheduler 也要回答这些问题，但还要额外考虑：

1. RadixAttention cache hit length。
2. Uncached suffix cost。
3. Cache-aware admission。
4. Radix tree node pin/evict。
5. Structured output grammar state。
6. Tool call 和 frontend program 分支。
7. Agent 中动态派生的 generation。
8. Speculative decoding 的 draft/verify 成本。

所以可以说：

```text
SGLang scheduler 继承了 continuous batching 的问题，又叠加了复杂程序执行的问题。
```

## 33.8 Request 模型对比

vLLM 中，一个请求通常可以理解为一个或一组 sequence：

```text
prompt tokens + generated tokens + block table + sampling params
```

SGLang 中，一个 generation 请求也有类似状态，但它可能来自更大的 LLM program：

```text
frontend program
  -> gen node 1
  -> fork branch
  -> gen node 2A / 2B
  -> tool call
  -> gen final
```

也就是说，vLLM 更常从“独立请求流”视角组织系统。

SGLang 更常从“程序执行图或多步调用”视角组织系统。

这会影响：

1. Cache 复用方式。
2. Scheduler 成本估计。
3. Request 生命周期。
4. Output 结构。
5. 工具和外部环境交互。

## 33.9 API 和入口对比

vLLM 的强项之一是 OpenAI-compatible API 和易部署的 serving 体验。

典型用法：

```text
launch server -> /v1/chat/completions -> streaming response
```

SGLang 也支持 OpenAI-compatible API，但入口更多：

1. OpenAI-compatible API。
2. Native `/generate` API。
3. Offline Engine API。
4. SGLang frontend language。

这些入口反映了 SGLang 的定位：不仅是通用 chat server，也希望作为复杂 LLM program 的执行 runtime。

如果你的应用完全是普通 chat completion，OpenAI-compatible API 就够。

如果你需要在 Python 中组织多步 generation、fork、choices、structured output 和工具交互，SGLang 的 frontend/runtime 组合更有表达力。

## 33.10 Structured output 对比

结构化输出两边都可以做，但 SGLang 把它放在更核心的位置讲。

原因是 SGLang 面向复杂程序，而复杂程序经常需要：

1. JSON 抽取结果。
2. choices。
3. tool arguments。
4. EBNF grammar。
5. structural tag。
6. agent action schema。

SGLang structured generation 的 runtime 视角是：

```text
constraint spec -> grammar backend -> grammar state -> valid token mask -> sampler
```

vLLM 也可以支持 structured outputs，但从本书的讲解主线看，vLLM 部分更集中在 KV memory 和 scheduling，SGLang 部分更集中在 structured generation 和复杂程序控制。

面试时可以说：

```text
Structured output 不是 SGLang 独有，但它和 SGLang 的复杂 LLM program 定位更强相关。
```

## 33.11 Tool use 和 agent 对比

vLLM 可以作为 agent 后端模型服务。

很多 agent framework 完全可以调用 vLLM 的 OpenAI-compatible API。

但这种模式下，agent 逻辑通常在外部框架：

```text
agent framework
  -> vLLM chat completion
  -> parse tool call
  -> execute tool
  -> vLLM chat completion
```

SGLang 更强调 runtime 与 LLM program 结构结合：

```text
SGLang frontend / native runtime
  -> multi-generation
  -> fork / choices
  -> structured output
  -> RadixAttention reuse
  -> tool / agent serving support
```

区别不是“vLLM 不能做 agent”，而是：

```text
vLLM 更常作为 agent 的模型服务后端
SGLang 更强调把复杂 agent-like LLM program 的执行效率纳入 runtime 设计
```

## 33.12 Speculative decoding 对比

Speculative decoding 不是 SGLang 独有，很多高性能 serving engine 都在支持。

它解决的是 decode 阶段主模型逐 token 串行的问题。

SGLang 文档中提供了多种 speculative 路径：

1. EAGLE。
2. EAGLE3。
3. MTP。
4. DFLASH。
5. Standalone draft model。
6. NGRAM。
7. Adaptive speculative decoding。

vLLM 生态中也有 speculative decoding 相关能力。

对比时不要说“speculative decoding 是 SGLang 和 vLLM 的根本区别”。

更准确的说法是：

```text
Speculative decoding 是现代 serving runtime 共同追求的 decode 优化方向；SGLang 在其 runtime 中提供了多种实现路径，并需要和 RadixAttention、scheduler、structured output 组合。
```

## 33.13 并行和多 GPU 对比

两者都需要支持多 GPU serving。

常见并行包括：

1. Tensor Parallel。
2. Pipeline Parallel。
3. Data Parallel。
4. Expert Parallel。
5. 多副本部署。

vLLM 的多 GPU serving 更常围绕通用模型服务吞吐、KV block 管理、worker/executor 和并行配置展开。

SGLang 也支持多 GPU 和多硬件后端，并且在复杂 workload 下还要考虑：

1. Prefix cache locality。
2. Session-aware routing。
3. Agent 多轮请求打到同一 replica。
4. Tool 等待阶段不占 GPU。
5. Structured output 和 speculative decoding 的额外资源。

多 GPU 层面没有简单的“谁更好”。要看模型、硬件、并行方式、请求长度分布和 workload 类型。

## 33.14 性能调优思路对比

vLLM 调优常围绕：

1. `max_num_batched_tokens`。
2. `max_num_seqs`。
3. `gpu_memory_utilization`。
4. `max_model_len`。
5. chunked prefill。
6. prefix cache。
7. preemption。
8. TP/PP/DP。
9. TTFT、TPOT、KV blocks、GPU 利用率。

SGLang 调优除了这些通用指标，还要看：

1. RadixAttention hit length。
2. Saved prefill tokens。
3. Radix tree eviction。
4. Grammar mask latency。
5. Tool parser 失败率。
6. Agent rounds。
7. Speculative accept length。
8. Draft/verify latency。
9. Session cache locality。

也就是说：

```text
vLLM 调优更偏通用 serving engine 指标
SGLang 调优更要结合复杂程序结构和 runtime 特性
```

## 33.15 选型：什么时候优先 vLLM

优先考虑 vLLM 的场景：

1. 主要是标准 chat/completion API。
2. 需要快速上线 OpenAI-compatible 服务。
3. 高并发通用文本生成。
4. workload 以独立请求为主。
5. 团队更熟悉 vLLM 生态。
6. 重点瓶颈是 KV cache 显存和 continuous batching。
7. 希望使用成熟的通用 serving 路线。

例如：

```text
企业内部通用 Chat API
RAG 问答服务
批量摘要生成
多租户文本生成平台
```

这些场景 vLLM 很自然。

## 33.16 选型：什么时候优先 SGLang

优先考虑 SGLang 的场景：

1. 多步 LLM program。
2. Self-consistency 或多候选采样。
3. Tree-of-Thought 或搜索式推理。
4. Agent serving。
5. 多轮工具调用。
6. 复杂 structured output。
7. 大量共享 prefix 的分支 workload。
8. 希望用 frontend language 表达 LLM 控制流。
9. 需要 offline engine 批处理和自定义 server。

例如：

```text
法律文档分析 agent
代码修复 agent
多候选推理评测
结构化抽取流水线
复杂工具调用系统
```

这些场景更能体现 SGLang 的设计重心。

## 33.17 也可以组合使用

实际工程里，不一定只能选一个。

可能的组合：

1. 普通 chat 流量走 vLLM。
2. 复杂 agent 流量走 SGLang。
3. 离线评测用 SGLang offline engine。
4. 平台层统一路由，根据 workload 选择 backend。
5. 不同模型或业务线使用不同 runtime。

关键是平台层要定义清楚边界：

```text
请求类型 -> 路由策略 -> runtime backend -> 监控指标 -> 回滚方案
```

不要为了统一技术栈，把所有 workload 强行塞进同一个 runtime。

## 33.18 面试中容易犯的错误

错误一：说 SGLang 完全替代 vLLM。

不准确。两者设计重心不同，都在演进。很多场景 vLLM 仍然是非常自然的选择。

错误二：说 vLLM 不能做 agent。

不准确。vLLM 可以作为 agent 的模型 serving 后端，只是 agent 控制流通常在外部框架。

错误三：把 PagedAttention 和 RadixAttention 混为一谈。

PagedAttention 关注 KV cache 分页布局，RadixAttention 关注 prefix sharing 的树状复用。

错误四：只比较 benchmark 数字。

Benchmark 依赖模型、硬件、prompt 长度、输出长度、batch、并行、采样和 workload。架构对比要先看问题场景。

错误五：认为 structured output 是后处理。

在 SGLang 的 structured generation 中，约束可以进入 decoding，每步用 grammar mask 限制合法 token。

## 33.19 面试官会怎么问

问题一：SGLang 和 vLLM 最大区别是什么？

回答要点：vLLM 更强调高并发通用 LLM serving，核心是 PagedAttention、KV block 管理和 continuous batching；SGLang 更强调复杂 LLM programs，核心是 frontend/runtime 协同、RadixAttention、structured output 和 agent/multi-turn 支持。

问题二：PagedAttention 和 RadixAttention 有什么区别？

回答要点：PagedAttention 解决 KV cache 物理内存分页和碎片问题；RadixAttention 用 radix tree 管理 token prefix 到 KV cache 的映射，解决复杂程序中共享前缀自动复用问题。

问题三：vLLM 能不能做 agent serving？

回答要点：可以作为 agent 的模型服务后端。但 agent 控制流、工具调用解析、状态管理通常在外部框架；SGLang 更强调把复杂 LLM program 的表达和执行效率纳入 runtime 设计。

问题四：什么时候选 SGLang？

回答要点：多步 generation、分支搜索、self-consistency、Tree-of-Thought、multi-turn、tool use、agent、复杂 structured output、共享 prefix 明显的 workload。

问题五：什么时候选 vLLM？

回答要点：标准 OpenAI-compatible chat/completion、高并发通用 serving、独立请求流、主要瓶颈在 KV cache 管理和 continuous batching 的场景。

## 33.20 标准回答模板

如果面试官问“对比一下 SGLang 和 vLLM”，可以这样回答：

```text
vLLM 和 SGLang 都是高性能 LLM serving/runtime 系统，都要处理 prefill、decode、KV cache、batching、scheduler、sampling 和 streaming。但它们的问题出发点不同。

vLLM 更像通用高并发 LLM serving engine。它的核心创新是 PagedAttention，用类似分页内存的方式管理 KV cache，配合 continuous batching 和 iteration-level scheduling，提高显存利用率和吞吐，适合大量独立 chat/completion/RAG 请求。

SGLang 更强调复杂 LLM programs 的表达和执行效率。很多应用不是一次 generate，而是包含多次 generation、fork、self-consistency、Tree-of-Thought、multi-turn、tool use、agent 和 structured output。SGLang 通过 frontend language 表达这些控制流，通过 runtime 的 RadixAttention、scheduler、structured generation 和 speculative decoding 高效执行。

PagedAttention 和 RadixAttention 也不是一回事。PagedAttention 关注 KV cache 物理布局和 block 管理，RadixAttention 关注 token prefix 的树状共享和自动 KV 复用。

所以我不会说 SGLang 简单替代 vLLM。标准高并发 OpenAI-compatible serving 很适合 vLLM；复杂 agent、结构化输出、多分支推理和共享 prefix 明显的 workload 更能体现 SGLang 优势。实际工程也可以平台层路由，让不同 workload 使用不同 runtime。
```

## 33.21 小练习

1. 用一张表对比 PagedAttention 和 RadixAttention。
2. 给出 3 个更适合 vLLM 的 workload，说明原因。
3. 给出 3 个更适合 SGLang 的 workload，说明原因。
4. 设计一个平台层路由策略，把普通 chat 和 agent 任务路由到不同 runtime。
5. 解释为什么 benchmark 不能脱离 workload 讨论。
6. 画出 vLLM 和 SGLang 都共有的 serving engine 模块图。
7. 面试中用 2 分钟讲清 SGLang 和 vLLM 的区别。

## 33.22 本章总结

vLLM 和 SGLang 都是现代 LLM 推理系统的重要代表。

vLLM 的主线是高并发通用 serving：PagedAttention、KV block manager、continuous batching、scheduler、worker/executor 和调优。

SGLang 的主线是复杂 LLM program runtime：frontend/runtime 协同、RadixAttention、cache-aware scheduler、structured generation、speculative decoding、tool use 和 agent serving。

理解二者的最好方式不是问“谁更强”，而是问“我的 workload 是什么，瓶颈在哪里，runtime 能看到哪些结构，哪些优化真正能生效”。

下一章会进入 mini-sglang 源码学习路径，把 SGLang 这部分的概念落到一个更可读的源码路线中。
