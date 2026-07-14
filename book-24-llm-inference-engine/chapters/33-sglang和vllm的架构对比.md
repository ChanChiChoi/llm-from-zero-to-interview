# 第 33 章 SGLang 和 vLLM 的架构对比

上一章讲了 Multi-turn、tool use 和 agent serving：SGLang 的重点不是只服务一次 chat completion，而是高效执行多轮、有状态、有工具、有分支的复杂 LLM programs。

到这里，本书第三部分和第四部分已经分别讲完了 vLLM 和 SGLang 的核心主线。

vLLM 主线是：PagedAttention、KV Cache Block Manager、continuous batching、scheduler、worker/executor、prefix cache、并行和调优。

SGLang 主线是：frontend/runtime 协同、RadixAttention、SGLang Runtime、scheduler、structured generation、speculative decoding、tool use 和 agent serving。

本章做一次系统对比。

一句话概括：

> vLLM 更像面向通用高并发 LLM serving 的高性能执行引擎，核心是 KV cache 分页管理和 continuous batching；SGLang 更像面向复杂 LLM programs 的高性能 runtime，核心是前后端协同、RadixAttention、structured output 和多步程序执行。

## 33.0 本讲资料边界与第二轮精修口径

本讲第二轮精修时，主要参考五类公开资料：

1. vLLM / PagedAttention 论文，对 KV cache 动态增长、显存碎片、logical block 到 physical block 映射、block sharing 和 high-throughput serving 的问题背景给出基础口径。
2. vLLM 官方文档和本书第 16 到 25 章，对 OpenAI-compatible serving、PagedAttention、KV block manager、continuous batching、scheduler、worker / executor、prefix caching、并行和性能调优的工程口径做内部对齐。
3. SGLang 论文，对 frontend language、runtime、RadixAttention、structured output decoding、复杂 language model programs 和多步 / 分支 workload 的系统分层给出基础口径。
4. SGLang 官方文档和本书第 26 到 32 章，对 OpenAI-compatible API、native / offline engine、RadixAttention、scheduler、structured generation、speculative decoding、tool use 和 agent serving 做内部对齐。
5. 近年 serving 系统相关论文和文档，对 PagedAttention 后续 trade-off、prefix cache locality、agent serving、多 GPU routing、structured output 和 workload-sensitive routing 的边界做校准。

本章的边界也要说清：

1. 本章不是 benchmark 排名，也不写“谁全面替代谁”。不同版本、硬件、模型、attention backend、并行配置和 workload 会显著改变性能结果。
2. 本章只讲架构重心和选型方法：vLLM-like runtime 更偏通用高并发 serving 和 KV block 管理，SGLang-like runtime 更偏复杂 LLM program 的表达与执行。
3. PagedAttention 和 RadixAttention 不在同一抽象层：前者主要回答 KV cache 物理块如何管理，后者主要回答 token prefix / program trajectory 如何匹配和复用。
4. vLLM 可以作为 agent framework 的模型后端；SGLang 的优势是让部分复杂程序结构对 runtime 更可见。不能把“runtime 能不能服务 agent”误解成二选一。
5. 本章 demo 是教学版 workload router / architecture comparator，用 toy metrics 说明独立 chat、共享 RAG、结构化抽取和 agent tree 的不同适配性，不代表真实性能数字。

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

## 33.21 架构对比公式、workload router 和可运行 demo

做架构对比时，不要只说“功能多”或“benchmark 高”。更好的方式是把 workload 抽象出来：

```math
w_i=(c_i,P_i,O_i,H_i,B_i,S_i,T_i)
```

其中 `c_i` 是同类 generation calls 数量，`P_i` 是每次 prompt tokens，`O_i` 是 output tokens，`H_i` 是共享 prefix tokens，`B_i` 是分支因子，`S_i` 表示是否需要 structured output，`T_i` 表示是否包含 tool / agent 轨迹。

如果没有任何 cache，prefill token 数是：

```math
P_{\mathrm{naive}}=\sum_i c_iP_i
```

vLLM-like prefix cache 常以完整 KV block 命中为基础。给定 block size `b`，共享 prefix 中能稳定复用的完整 block token 数可以写成：

```math
H_i^{\mathrm{block}}=b\left\lfloor\frac{H_i}{b}\right\rfloor
```

对应的简化 prefill 节省是：

```math
S_{\mathrm{vllm}}=\sum_i (c_i-1)H_i^{\mathrm{block}}
```

SGLang-like runtime 除了线性共享 prefix，还要看多步程序和分支轨迹。一个简化写法是：

```math
S_{\mathrm{sglang}}=\sum_i (c_i-1)H_i+\sum_i \max(0,B_i-1)H_i^{\mathrm{branch}}
```

其中 `H_branch` 表示多分支 program 中额外共享的 root / trajectory token。这个式子只是教学抽象，真实系统还会受 page size、KV layout、eviction、scheduler、routing 和安全隔离影响。

运行时选择不能只看 cache，还要看 workload fit：

```math
F(r,w_i)=F_{\mathrm{serving}}+F_{\mathrm{cache}}+F_{\mathrm{program}}+F_{\mathrm{grammar}}+F_{\mathrm{agent}}
```

`r` 是 runtime 类型。独立高并发 chat 通常让 `F_serving` 权重大；structured output、tool use、multi-turn 和分支搜索会提高 `F_program`、`F_grammar` 和 `F_agent` 的权重。

架构对比门禁可以写成：

```math
G_{\mathrm{arch}}=G_{\mathrm{core}}G_{\mathrm{paged}}G_{\mathrm{radix}}G_{\mathrm{vllm}}G_{\mathrm{sglang}}G_{\mathrm{router}}G_{\mathrm{metric}}
```

这些因子分别要求共同 serving core、PagedAttention / block cache、RadixAttention / program reuse、vLLM 不被错误替代、SGLang 的复杂程序优势可见、平台层 workload router 可解释、指标可复盘。

下面是一个 0 依赖 toy demo。它把四类 workload 放进同一个对比器：

1. `independent_chat`：大量独立短 chat，更偏通用 serving。
2. `rag_shared_doc`：共享 RAG 文档，vLLM-like full block prefix cache 和 SGLang-like prefix reuse 都能受益。
3. `json_extraction`：结构化抽取，多次 generation 和 structured output 更偏 SGLang-like program runtime。
4. `agent_tree`：agent / branch trajectory，多分支 prefix sharing 更能体现 RadixAttention 的价值。

```python
from dataclasses import dataclass


@dataclass
class Workload:
    name: str
    calls: int
    prompt_tokens: int
    output_tokens: int
    shared_prefix: int
    branch_factor: int = 1
    structured: bool = False
    tool_or_agent: bool = False
    independent: bool = True


class ToyArchitectureComparator:
    def __init__(self, workloads, block_size=4):
        self.workloads = workloads
        self.block_size = block_size

    def _full_block_hit(self, tokens):
        return self.block_size * (tokens // self.block_size)

    def _vllm_reuse(self, workload):
        if workload.calls <= 1:
            return 0
        return self._full_block_hit(workload.shared_prefix) * (workload.calls - 1)

    def _sglang_reuse(self, workload):
        if workload.calls <= 1:
            return 0
        base = workload.shared_prefix * (workload.calls - 1)
        branch_bonus = 0
        if workload.branch_factor > 1:
            branch_bonus = (workload.branch_factor - 1) * self._full_block_hit(workload.shared_prefix // 2)
        return base + branch_bonus

    def _fit_scores(self, workload):
        vllm = 0
        sglang = 0
        if workload.independent:
            vllm += 4
            sglang += 2
        if workload.shared_prefix > 0:
            vllm += 1
            sglang += 1
        if workload.calls > 1 and not workload.independent:
            sglang += 1
        if workload.branch_factor > 1:
            sglang += 3
        if workload.structured:
            vllm += 1
            sglang += 2
        if workload.tool_or_agent:
            vllm += 1
            sglang += 3
        return vllm, sglang

    def run(self):
        rows = []
        totals = {
            "prompt_tokens_if_naive": 0,
            "vllm_run_prefill": 0,
            "sglang_run_prefill": 0,
            "vllm_saved_prefill": 0,
            "sglang_saved_prefill": 0,
            "vllm_wins": 0,
            "sglang_wins": 0,
            "mixed_route": 0,
        }
        for workload in self.workloads:
            naive = workload.calls * workload.prompt_tokens
            v_saved = self._vllm_reuse(workload)
            s_saved = self._sglang_reuse(workload)
            v_run = max(0, naive - v_saved)
            s_run = max(0, naive - s_saved)
            v_fit, s_fit = self._fit_scores(workload)
            if v_fit > s_fit:
                route = "vllm"
                totals["vllm_wins"] += 1
            elif s_fit > v_fit:
                route = "sglang"
                totals["sglang_wins"] += 1
            else:
                route = "either"
                totals["mixed_route"] += 1
            rows.append(
                {
                    "workload": workload.name,
                    "calls": workload.calls,
                    "naive_prefill": naive,
                    "vllm_saved": v_saved,
                    "sglang_saved": s_saved,
                    "vllm_run": v_run,
                    "sglang_run": s_run,
                    "vllm_fit": v_fit,
                    "sglang_fit": s_fit,
                    "recommended": route,
                }
            )
            totals["prompt_tokens_if_naive"] += naive
            totals["vllm_run_prefill"] += v_run
            totals["sglang_run_prefill"] += s_run
            totals["vllm_saved_prefill"] += v_saved
            totals["sglang_saved_prefill"] += s_saved

        totals["vllm_saved_ratio"] = round(totals["vllm_saved_prefill"] / max(1, totals["prompt_tokens_if_naive"]), 3)
        totals["sglang_saved_ratio"] = round(totals["sglang_saved_prefill"] / max(1, totals["prompt_tokens_if_naive"]), 3)
        totals["sglang_extra_saved"] = totals["sglang_saved_prefill"] - totals["vllm_saved_prefill"]
        gates = {
            "common_serving_core_visible": all(row["naive_prefill"] > 0 for row in rows),
            "paged_prefix_cache_visible": totals["vllm_saved_prefill"] > 0,
            "radix_program_reuse_visible": totals["sglang_saved_prefill"] > totals["vllm_saved_prefill"],
            "vllm_not_replaced": totals["vllm_wins"] >= 1,
            "sglang_program_fit_visible": totals["sglang_wins"] >= 2,
            "workload_router_needed": len({row["recommended"] for row in rows}) >= 2,
            "metrics_ready": totals["sglang_saved_ratio"] > totals["vllm_saved_ratio"],
        }
        gates["architecture_comparison_gate"] = all(gates.values())
        return rows, totals, gates


workloads = [
    Workload("independent_chat", calls=8, prompt_tokens=20, output_tokens=16, shared_prefix=0, independent=True),
    Workload("rag_shared_doc", calls=4, prompt_tokens=80, output_tokens=24, shared_prefix=56, independent=True),
    Workload("json_extraction", calls=6, prompt_tokens=44, output_tokens=12, shared_prefix=32, structured=True, independent=False),
    Workload(
        "agent_tree",
        calls=5,
        prompt_tokens=72,
        output_tokens=18,
        shared_prefix=48,
        branch_factor=3,
        structured=True,
        tool_or_agent=True,
        independent=False,
    ),
]

rows, summary, gates = ToyArchitectureComparator(workloads).run()
print("architecture_rows=", rows)
print("architecture_summary=", summary)
print("architecture_gates=", gates)
```

运行后可以看到类似输出：

```text
architecture_summary= {'prompt_tokens_if_naive': 1104, 'vllm_run_prefill': 584, 'sglang_run_prefill': 536, 'vllm_saved_prefill': 520, 'sglang_saved_prefill': 568, 'vllm_wins': 2, 'sglang_wins': 2, 'mixed_route': 0, 'vllm_saved_ratio': 0.471, 'sglang_saved_ratio': 0.514, 'sglang_extra_saved': 48}
architecture_gates= {'common_serving_core_visible': True, 'paged_prefix_cache_visible': True, 'radix_program_reuse_visible': True, 'vllm_not_replaced': True, 'sglang_program_fit_visible': True, 'workload_router_needed': True, 'metrics_ready': True, 'architecture_comparison_gate': True}
```

这个 demo 的重点是：

1. 两者都有共同 serving core，不能把一个说成“会 serving”，另一个说成“不会 serving”。
2. vLLM-like block prefix cache 在共享 RAG 文档上能明显节省 prefill。
3. SGLang-like program runtime 在 structured extraction 和 agent tree 里能看到更多 program structure。
4. 架构选型要看 workload，不是把所有请求都塞进一个结论。
5. 平台层可以用 workload router，把独立 chat / RAG / agent / structured extraction 分流到更合适的 backend。

## 33.22 小练习

1. 用一张表对比 PagedAttention 和 RadixAttention。
2. 给出 3 个更适合 vLLM 的 workload，说明原因。
3. 给出 3 个更适合 SGLang 的 workload，说明原因。
4. 设计一个平台层路由策略，把普通 chat 和 agent 任务路由到不同 runtime。
5. 解释为什么 benchmark 不能脱离 workload 讨论。
6. 画出 vLLM 和 SGLang 都共有的 serving engine 模块图。
7. 面试中用 2 分钟讲清 SGLang 和 vLLM 的区别。

## 33.23 本章总结

vLLM 和 SGLang 都是现代 LLM 推理系统的重要代表。

vLLM 的主线是高并发通用 serving：PagedAttention、KV block manager、continuous batching、scheduler、worker/executor 和调优。

SGLang 的主线是复杂 LLM program runtime：frontend/runtime 协同、RadixAttention、cache-aware scheduler、structured generation、speculative decoding、tool use 和 agent serving。

理解二者的最好方式不是问“谁更强”，而是问“我的 workload 是什么，瓶颈在哪里，runtime 能看到哪些结构，哪些优化真正能生效”。

下一章会进入 mini-sglang 源码学习路径，把 SGLang 这部分的概念落到一个更可读的源码路线中。
