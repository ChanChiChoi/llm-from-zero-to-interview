# 第 44 章 nano-vLLM 学习路线

前面五部分已经讲完了推理框架的核心概念：generate loop、KV Cache、batching、scheduler、PagedAttention、vLLM、SGLang、prefix cache、PD 分离和跨节点 serving。

从这一章开始进入第六部分：教学项目源码路线与升级实战。

这一部分的目标不是把生产级 vLLM 或 SGLang 的每一行代码都读完，而是借助更小的项目建立源码阅读骨架。

`nano-vLLM` 是一个很适合入门的项目，因为它有几个特点：

1. 代码量小，核心逻辑集中。
2. 接口风格接近 vLLM。
3. 包含 engine、scheduler、sequence、block manager、model runner、attention、sampler 等关键模块。
4. 支持 prefix caching、tensor parallel、torch compilation、CUDA graph 等优化方向。
5. 适合拿来做二次改造练习。

一句话概括：

> 学 nano-vLLM 的重点不是记住每个类名，而是把一个轻量实现和生产级 serving engine 的模块边界对应起来，知道请求如何进入 engine、如何被 scheduler 组织、KV block 如何分配、model runner 如何执行 prefill/decode，以及它距离生产系统还缺什么。

## 44.0 本讲资料边界与第二轮精修口径

本讲第二轮精修前，先按 `WRITING_PLAN.md` 对公开资料做校准：参考 `GeeeekExplorer/nano-vllm` 官方仓库和 README 对 nano-vLLM 作为轻量 vLLM 实现、约 1200 行 Python、包含 Prefix Caching、Tensor Parallelism、Torch compilation 和 CUDA graph 等学习点的说明；参考仓库中 `nanovllm/engine` 目录对 `llm_engine.py`、`scheduler.py`、`sequence.py`、`block_manager.py`、`model_runner.py` 等模块边界的呈现；并结合本书前面关于 generate loop、KV cache、block manager、continuous batching、vLLM scheduler、prefix cache、TP 和 PD 分离的章节。

本讲只讲“如何用 nano-vLLM 建立 vLLM-like serving engine 的源码阅读骨架”：模块地图、阅读顺序、请求生命周期、KV block lifecycle、model runner batch contract、sampler、可观测实验和改造练习。不把某个 GitHub 提交的源码行号、函数签名、内部字段名、benchmark 数字、真实 CUDA graph 行为、生产级 vLLM 的完整 worker/executor 设计或某个版本的默认实现写成通用结论。

本讲新增 demo 是教学版 nano-vLLM source auditor：用 0 依赖 Python 模拟源码阅读笔记和实验 trace，检查 example、LLM、sampling params、sequence、engine、scheduler、block manager、model runner、attention、sampler 和模型结构是否覆盖；同时检查 request state、waiting/running queue、KV block table、free blocks、slot mapping、logits、sampled token 和 finished cleanup 是否能被实验观察到。

## 44.1 本章目标

读完本章，你应该能讲清：

1. 为什么 nano-vLLM 适合作为 vLLM-like 源码入门。
2. 阅读 nano-vLLM 应该按什么顺序。
3. `LLM`、`LLMEngine`、`Scheduler`、`Sequence`、`BlockManager`、`ModelRunner` 分别对应推理框架里的什么模块。
4. 如何从一次 `generate()` 调用追踪到 prefill、decode、KV cache 和 sampling。
5. 如何通过 nano-vLLM 理解 continuous batching 和 block-based KV 管理。
6. nano-vLLM 和生产级 vLLM 的差距在哪里。
7. 如何基于 nano-vLLM 设计改造练习和面试项目。

## 44.2 先明确学习目标

读教学项目源码，最忌讳两件事。

第一，像读论文一样从头到尾逐行看。

这样很容易陷入工具函数和实现细节，忘记主线。

第二，直接拿生产级框架的复杂度要求教学项目。

教学项目不会完整覆盖：

1. 多租户。
2. 分布式控制面。
3. 复杂异步 streaming。
4. 完整 OpenAI-compatible API。
5. Prefix cache 的所有边界情况。
6. 生产级 metrics、logs、traces。
7. Worker 崩溃恢复。
8. 跨节点 PD 分离。

所以读 nano-vLLM 的目标应该是：

```text
用最小代码看清 vLLM-like engine 的骨架。
```

具体要回答五个问题：

1. 请求对象怎么表示？
2. 请求队列怎么调度？
3. KV cache 怎么按 block 管理？
4. prefill 和 decode 怎么共用 model runner？
5. 一轮 engine step 里到底发生了什么？

## 44.3 推荐先看项目结构

当前 nano-vLLM 的核心目录大致是：

```text
nanovllm/
  __init__.py
  config.py
  llm.py
  sampling_params.py
  engine/
    llm_engine.py
    scheduler.py
    sequence.py
    block_manager.py
    model_runner.py
  layers/
    attention.py
    sampler.py
    linear.py
    layernorm.py
    rotary_embedding.py
    embed_head.py
  models/
    qwen3.py
  utils/
```

这正好对应一个推理框架的分层：

```text
user API
  -> LLM wrapper
  -> engine
  -> scheduler
  -> sequence state
  -> block manager
  -> model runner
  -> model/layers
```

可以先建立映射表：

| nano-vLLM 模块 | 推理框架概念 | 重点问题 |
|---|---|---|
| `llm.py` | 用户入口 | `generate()` 如何把 prompt 变成请求 |
| `sampling_params.py` | 采样参数 | temperature、max tokens 等如何传递 |
| `llm_engine.py` | engine 主循环 | 请求如何进入队列，何时 step |
| `scheduler.py` | 调度器 | waiting/running 如何组成 batch |
| `sequence.py` | 请求状态 | prompt、output、block table、完成状态怎么保存 |
| `block_manager.py` | KV block 管理 | block 如何分配、复用和释放 |
| `model_runner.py` | 模型执行器 | prefill/decode batch 如何执行 |
| `attention.py` | attention kernel 封装 | 如何使用 KV cache 做注意力 |
| `sampler.py` | token 采样 | logits 如何变成 next token |
| `qwen3.py` | 模型结构 | 权重、layer、forward 如何组织 |

这张表比记住某个函数名更重要。

因为项目可能迭代，但模块职责不会轻易变。

## 44.4 推荐阅读顺序

不要从 `models/qwen3.py` 开始。

模型结构代码很容易把你带回 Transformer 细节，但这本书关注的是 serving engine。

推荐顺序是：

1. `example.py`：看用户如何调用。
2. `llm.py`：看外部 API 如何封装 engine。
3. `sampling_params.py`：看采样参数如何表达。
4. `engine/sequence.py`：看一个请求在 engine 内部如何表示。
5. `engine/llm_engine.py`：看 engine 主流程。
6. `engine/scheduler.py`：看请求如何被选入 batch。
7. `engine/block_manager.py`：看 KV block 如何管理。
8. `engine/model_runner.py`：看 prefill/decode 如何执行。
9. `layers/attention.py`：看 KV cache 如何参与 attention。
10. `layers/sampler.py`：看 logits 如何采样成 token。
11. `models/qwen3.py`：最后再看模型 forward 细节。

这个顺序是从请求生命周期往下读。

也就是：

```text
API -> request -> engine -> scheduler -> KV -> model runner -> attention -> sampler
```

这和线上 debug 的顺序一致。

线上排查推理问题时，你通常也不会先看模型层，而是先问：请求有没有进来？排队多久？是否被 schedule？KV 是否分配？prefill/decode 哪个慢？

## 44.5 第一遍：只追踪一次 generate

第一遍阅读只做一件事：追踪一次 `generate()`。

你应该画出这样的路径：

```text
prompts
  -> LLM.generate()
  -> tokenize
  -> create Sequence
  -> LLMEngine.add_request()
  -> engine loop
  -> Scheduler.schedule()
  -> ModelRunner.execute()
  -> Sampler
  -> append token
  -> finish or continue
  -> detokenize outputs
```

这一遍先不要纠结：

1. CUDA graph 怎么做。
2. tensor parallel 怎么切。
3. attention kernel 具体怎么优化。
4. prefix cache 细节。
5. 编译优化参数。

只需要搞清楚：

```text
一个 prompt 如何变成一个持续生成 token 的 Sequence。
```

建议边读边记录四个对象：

1. 用户看到的 request。
2. engine 内部的 sequence。
3. scheduler 看到的 waiting/running item。
4. model runner 看到的 batch。

很多推理框架面试题，本质都是在问这四个视角之间如何转换。

## 44.6 第二遍：读 Sequence

`Sequence` 是理解 serving engine 的关键。

一个 sequence 通常要保存：

1. prompt token ids。
2. generated token ids。
3. 当前长度。
4. 采样参数。
5. 是否完成。
6. finish reason。
7. KV block 映射。
8. prefix cache 相关状态。

在 naive generate 里，请求状态可能只是一个 `input_ids` tensor。

但在 serving engine 里，请求状态必须长期存在，因为 decode 是多轮执行：

```text
step 1 -> append token
step 2 -> append token
step 3 -> append token
...
```

每一轮都要知道：

```text
这条序列当前在哪里？
它的 KV 在哪些 block？
它是否已经达到 max_tokens？
它是否遇到 eos？
它下一轮还能不能进入 batch？
```

读 `sequence.py` 时，重点看：

1. prompt 和 output 是否分开保存。
2. token append 在哪里发生。
3. sequence length 如何计算。
4. finish 条件如何判断。
5. block table 或 block id 如何挂在 sequence 上。

面试中如果让你“设计一个推理请求对象”，你的答案就应该从这里抽象出来。

## 44.7 第三遍：读 Scheduler

Scheduler 是 serving engine 的大脑。

教学项目里的 scheduler 通常比生产级简单，但仍然能体现核心思想：

```text
从 waiting 队列和 running 队列里选择一批 sequence，让 model runner 执行下一步。
```

读 scheduler 时，重点看三件事。

第一，waiting 和 running 如何区分。

```text
waiting: 还没完成 prefill，或还没被加入 active set
running: 已经有 KV，正在 decode
```

第二，prefill 和 decode 如何调度。

有些实现会把 prefill 和 decode 放在同一个 schedule 结果里，有些会分开。

关键是识别：

```text
哪些 sequence 要跑 prefill？
哪些 sequence 要跑 decode？
```

第三，batch 限制来自哪里。

常见限制包括：

1. 最大 batch size。
2. 最大 token 数。
3. KV block 可用量。
4. 最大 running sequence 数。
5. prompt 长度。

Scheduler 不只是“凑 batch”。

它要在吞吐和延迟之间做取舍：

```text
prefill 太多 -> decode TPOT 抖动
decode 优先 -> 新请求 TTFT 变差
batch 太大 -> 单步慢
batch 太小 -> GPU 利用率低
```

这正好对应前面 continuous batching 和 PD 分离的内容。

## 44.8 第四遍：读 Block Manager

`BlockManager` 对应 vLLM 最核心的思想之一：用 block 管理 KV cache。

不要把它理解成普通显存分配器。

它解决的是：

```text
大量请求的 KV cache 长度不同、生命周期不同，如何避免整段连续显存分配带来的浪费和碎片？
```

读 `block_manager.py` 时，重点看：

1. block size 是多少。
2. 总 block 数如何计算。
3. free block 如何维护。
4. sequence 如何申请 block。
5. decode 生成新 token 时是否需要追加 block。
6. 请求结束时如何释放 block。
7. prefix cache 命中时 block 如何复用。

可以把它和第 18 章对应起来：

```text
Sequence -> logical token positions
BlockManager -> physical KV blocks
block table -> logical-to-physical mapping
```

这也是面试最常问的部分。

如果你能从 nano-vLLM 的 block manager 讲到生产级 vLLM 的 PagedAttention，就说明已经抓住了主线。

## 44.9 第五遍：读 Model Runner

`ModelRunner` 是 engine 和模型之间的桥。

它通常负责：

1. 加载模型。
2. 准备输入 tensor。
3. 准备 position ids。
4. 准备 slot mapping 或 block table。
5. 调用模型 forward。
6. 接收 logits。
7. 调用 sampler。
8. 返回 next token。

读 `model_runner.py` 时，不要只看 forward。

重点是看 batch 如何从高层对象变成 tensor：

```text
Sequence list
  -> input token ids
  -> positions
  -> block tables / slot mapping
  -> model forward
  -> logits
  -> sampled token ids
```

同时要区分 prefill 和 decode 的输入形态：

```text
prefill: one request may contribute many prompt tokens
decode: one running request usually contributes one new token per step
```

如果一个实现没有非常显式地区分 prefill/decode，也要在阅读时主动标注出来。

因为生产级优化几乎都围绕这个差异展开。

## 44.10 第六遍：读 Attention

`layers/attention.py` 是连接模型计算和 KV cache 的地方。

读 attention 时，重点不是重新学习 attention 公式，而是搞清楚 KV cache 如何被访问。

关键问题包括：

1. 当前 token 的 Q 从哪里来。
2. K/V 如何写入 cache。
3. 历史 K/V 如何从 block 中读出。
4. prefill 和 decode 是否走不同路径。
5. block table 如何参与 attention。
6. attention 输出如何回到模型后续层。

可以用一句话理解：

```text
ModelRunner 负责把 batch 描述清楚，Attention 负责按这些描述读写 KV 并完成计算。
```

如果你只看模型层代码，很容易以为 KV cache 是普通 tensor。

但 serving engine 里的 KV cache 是被 block manager、scheduler 和 attention kernel 共同管理的系统资源。

## 44.11 第七遍：读 Sampler

Sampler 代码通常很短，但不要忽略它。

因为推理服务的输出行为由 sampler 决定。

读 sampler 时，重点看：

1. temperature 如何使用。
2. greedy 和 sampling 如何切换。
3. top-k、top-p 是否支持。
4. logits shape 是什么。
5. 每个 sequence 是否可以有不同 sampling params。
6. 采样结果如何写回 sequence。

生产级系统里，sampler 还会涉及：

1. repetition penalty。
2. presence/frequency penalty。
3. logprobs。
4. stop words。
5. structured output 约束。
6. speculative decoding 验证。

nano-vLLM 的 sampler 可以作为最小入口。

后续如果你要扩展功能，可以从这里开始。

## 44.12 第八遍：读模型结构

最后再看 `models/qwen3.py`。

这一部分用于回答：

```text
serving engine 准备好的 input、position、KV cache metadata，最终如何进入具体模型 forward？
```

读模型结构时，重点看：

1. embedding。
2. decoder layers。
3. attention 层如何接收 position 和 KV cache。
4. MLP。
5. norm。
6. lm head。
7. 权重加载。

这部分和普通 Transformer 实现相似，但 serving 场景多了几个关键点：

1. 不再每次传完整上下文。
2. KV cache 是外部管理的。
3. decode 时通常只输入新 token。
4. position 必须和已有 KV 对齐。
5. tensor parallel 会改变 linear 层和权重加载方式。

所以读模型结构时，要一直带着 serving 问题看。

## 44.13 把 nano-vLLM 映射回本书前面章节

可以把 nano-vLLM 作为前面章节的实物索引。

| 本书章节 | nano-vLLM 观察点 |
|---|---|
| 第 7 章最小 generate loop | `LLM.generate()` 和 engine loop |
| 第 8 章采样 | `sampling_params.py`、`sampler.py` |
| 第 9 章 KV Cache | `block_manager.py`、`attention.py` |
| 第 10 章 batched prefill | model runner 如何组织 prompt batch |
| 第 11 章 batched decode | running sequence 如何逐步 decode |
| 第 12 章 scheduler | `scheduler.py` |
| 第 17 章 PagedAttention | block table 和 attention 访问 KV |
| 第 18 章 Block Manager | `block_manager.py` |
| 第 19 章 Continuous Batching | engine step + waiting/running 调度 |
| 第 20 章 vLLM 请求调度流程 | 从 generate 到 schedule 到 execute |
| 第 23 章 Prefix Cache | prefix block 复用逻辑 |
| 第 24 章 Serving 并行 | tensor parallel 相关代码 |

这样读源码就不是孤立阅读，而是把抽象概念落回代码。

## 44.14 一轮 Engine Step 应该怎么理解

读完核心模块后，要能讲清一轮 engine step。

一个简化版本是：

```text
1. scheduler 查看 waiting/running sequences
2. 判断哪些请求可以 prefill，哪些请求可以 decode
3. 为新请求分配 KV blocks
4. 为 batch 准备 input ids、positions、block tables
5. model runner 执行 forward
6. sampler 产生 next token
7. sequence 追加 token
8. 判断是否 finished
9. finished sequence 释放 KV blocks
10. 未完成 sequence 留在 running 队列等待下一轮
```

这就是 continuous batching 的核心。

它和 naive generate 的差别是：

```text
naive generate: 一个请求完整生成完再处理下一个
serving engine: 每轮从多个请求中动态组成 batch，只推进一步或一段
```

如果你能用 nano-vLLM 的代码解释这十步，就说明源码主线已经打通。

## 44.15 读 Prefix Cache 时看什么

nano-vLLM 支持 prefix caching，这是连接 vLLM 和 SGLang 思想的重要入口。

读 prefix cache 时，重点看：

1. prefix 如何被 hash。
2. cache 命中后复用的是 token、hidden state 还是 KV block。
3. 命中的 block 如何挂到当前 sequence。
4. 未命中的后缀如何继续 prefill。
5. 请求结束后 block 是否还能保留。
6. cache eviction 如何做。

要特别注意 ownership：

```text
一个 block 是当前 sequence 独占，还是 prefix cache 共享？
释放 sequence 时是否真的释放物理 block？
```

这能帮助你理解为什么生产级 prefix cache 需要引用计数、pin/unpin、eviction policy 和内存压力控制。

## 44.16 读 Tensor Parallel 时看什么

nano-vLLM 也可以作为 tensor parallel 入门。

但这里不要先陷入分布式通信细节。

先看三个问题：

1. 哪些 linear 层被切分。
2. 权重加载时如何按 rank 取 shard。
3. forward 中哪里需要通信或聚合。

常见切分包括：

```text
qkv projection: column parallel
output projection: row parallel
mlp up/gate: column parallel
mlp down: row parallel
lm head: vocab parallel or replicated
```

把它和第 24 章联系起来：

```text
TP 提升单 replica 可承载模型规模，但会引入每层通信；serving 中 decode 高频，跨节点 TP 尤其危险。
```

教学项目通常不会覆盖生产级所有通信优化，但足够理解 TP 如何进入模型层。

## 44.17 nano-vLLM 和生产级 vLLM 的差距

nano-vLLM 的价值在于小，不在于完整。

它和生产级 vLLM 至少有这些差距：

1. API server 能力不足。
2. 异步 streaming 能力不足。
3. 多模型、多 LoRA、多租户能力不足。
4. 调度策略相对简单。
5. Prefix cache 边界情况更少。
6. Metrics、logs、traces 不完整。
7. Worker/executor 架构不完整。
8. 分布式部署和故障恢复不完整。
9. 生产级内存治理不足。
10. 插件化 kernel 和硬件后端适配不足。

但它保留了最核心的骨架：

```text
sequence state + scheduler + block manager + model runner + attention + sampler
```

这正是入门应该先掌握的部分。

## 44.18 适合做的改造练习

读完 nano-vLLM 后，最好的学习方式是改代码。

推荐练习一：增加 TTFT/TPOT 统计。

记录每个 sequence 的：

1. request arrived time。
2. prefill start/end。
3. first token time。
4. 每个 decode token 时间。
5. finish time。

推荐练习二：打印 scheduler 决策。

每轮 step 输出：

1. waiting 数量。
2. running 数量。
3. prefill batch。
4. decode batch。
5. KV free blocks。

推荐练习三：实现简单 max token budget。

限制每轮 prefill token 数，观察 TTFT 和 TPOT 变化。

推荐练习四：实现 decode-first 策略。

优先保证 running sequence 的 decode，再插入 prefill，观察新请求饥饿问题。

推荐练习五：实现简单 chunked prefill。

把长 prompt 拆成多个 chunk，避免一次 prefill 长时间阻塞 decode。

推荐练习六：扩展 sampler。

加入 top-k、top-p、repetition penalty 或 stop token 逻辑。

推荐练习七：增加 block 使用可视化。

输出每个 sequence 占用哪些 block，结束后哪些 block 被释放。

推荐练习八：模拟 KV 泄漏。

故意不释放 finished sequence 的 block，观察可用 block 如何下降。

推荐练习九：实现最小 HTTP API。

把离线 `generate()` 包一层服务接口，理解 offline inference 和 online serving 的差异。

推荐练习十：抽象 prefill/decode 接口。

为后续 PD 分离做准备，让 model runner 的 prefill 和 decode 更显式。

## 44.19 适合写进简历的项目描述

如果你基于 nano-vLLM 做过改造，可以这样写项目描述：

```text
基于 nano-vLLM 实现了一个轻量级 LLM serving engine 学习项目，阅读并改造 engine、scheduler、block manager、model runner 和 sampler 模块；增加 TTFT/TPOT 指标、scheduler 决策日志和 KV block 使用统计；实现了简单 chunked prefill 和 decode-first 调度策略，用压测对比不同策略下的首 token 延迟、decode 抖动和 KV 利用率。
```

更偏系统方向可以写：

```text
在 nano-vLLM 基础上抽象 request state、KV metadata 和 prefill/decode 执行接口，模拟 vLLM-like continuous batching 流程，并设计了从单机 engine 演进到 PD 分离的模块边界，包括 prefill scheduler、decode scheduler、KV transfer handle 和 router 状态机。
```

面试时不要只说“我读过 nano-vLLM”。

要说清：

1. 你读了哪些模块。
2. 你画出了什么请求路径。
3. 你改了什么策略。
4. 你压测了什么指标。
5. 你发现了什么 tradeoff。

## 44.20 面试官会怎么问

问题一：nano-vLLM 适合用来学什么？

回答要点：适合学习 vLLM-like serving engine 的核心骨架，包括 LLM API、engine loop、sequence state、scheduler、KV block manager、model runner、attention 和 sampler。它代码量小，便于从一次 generate 调用追踪到 prefill、decode、KV cache 和 sampling。

问题二：你会按什么顺序读 nano-vLLM？

回答要点：先看 example 和 LLM.generate，再看 sampling params 和 sequence 表示；然后看 llm_engine 主循环、scheduler、block_manager、model_runner；最后看 attention、sampler 和具体模型结构。顺序是沿请求生命周期从上到下读。

问题三：nano-vLLM 里的 block manager 解决什么问题？

回答要点：它用固定大小 block 管理不同长度、不同生命周期请求的 KV cache，避免为每个请求分配连续大段 KV，降低显存碎片和浪费。Sequence 记录逻辑 token，block manager 维护物理 block，attention 通过 block table 读写 KV。

问题四：nano-vLLM 和生产级 vLLM 差在哪里？

回答要点：nano-vLLM 保留核心骨架，但缺少生产级 API server、异步 streaming、多租户、多模型、复杂调度、完整观测、worker/executor 架构、分布式部署、故障恢复和更复杂的内存治理。

问题五：如何基于 nano-vLLM 做一个有价值的项目？

回答要点：可以增加 TTFT/TPOT 指标、scheduler 日志、KV block 可视化，实现 token budget、decode-first、chunked prefill、sampler 扩展或最小 HTTP API，并通过压测比较不同调度策略对吞吐、TTFT、TPOT 和 KV 利用率的影响。

## 44.21 标准回答模板

如果面试官问“你怎么通过 nano-vLLM 学 vLLM”，可以这样回答：

```text
我会把 nano-vLLM 当成 vLLM-like serving engine 的最小骨架来看，而不是只看模型 forward。阅读顺序是沿请求生命周期走：先看 example 和 LLM.generate，理解用户 prompt 如何进入系统；再看 Sequence，理解请求状态如何保存 prompt token、output token、finish 状态和 KV block；然后看 LLMEngine 和 Scheduler，理解 waiting/running 队列如何组成 batch；接着看 BlockManager，理解 KV cache 如何按固定大小 block 分配、复用和释放；最后看 ModelRunner、Attention 和 Sampler，理解 batch 如何变成 tensor，如何读写 KV cache，logits 如何变成 next token。

读完后我会重点总结一轮 engine step：scheduler 选择 prefill/decode 请求，block manager 分配或追加 KV block，model runner 准备 input ids、positions 和 block table，模型 forward 后 sampler 采样 token，sequence 追加 token，完成的请求释放 KV，未完成的请求继续留在 running 队列。

nano-vLLM 和生产级 vLLM 的差距主要在生产能力上，比如 API server、异步 streaming、多租户、多模型、worker/executor、复杂调度、metrics、故障恢复和分布式部署。但它保留了 sequence、scheduler、block manager、model runner、attention、sampler 这些核心模块，所以非常适合入门。

如果要做项目，我会在 nano-vLLM 上增加 TTFT/TPOT 统计、scheduler 决策日志和 KV block 使用可视化，再实现简单 token budget、decode-first 或 chunked prefill，用压测比较不同策略对首 token 延迟、decode 抖动、吞吐和显存利用率的影响。
```

## 44.22 nano-vLLM 源码覆盖率、实验门禁和可运行 demo

先把源码学习对象抽象成模块集合：

```math
\mathcal{M}=\{m_1,m_2,\ldots,m_N\}
```

阅读覆盖率可以写成：

```math
C_{\mathrm{module}}=\frac{|M_{\mathrm{seen}}\cap M_{\mathrm{req}}|}{\max(1,|M_{\mathrm{req}}|)}
```

资源生命周期覆盖率可以写成：

```math
C_{\mathrm{resource}}=\frac{|R_{\mathrm{seen}}\cap R_{\mathrm{req}}|}{\max(1,|R_{\mathrm{req}}|)}
```

实验覆盖率可以写成：

```math
C_{\mathrm{experiment}}=\frac{|E_{\mathrm{done}}\cap E_{\mathrm{req}}|}{\max(1,|E_{\mathrm{req}}|)}
```

最终门禁：

```math
G_{\mathrm{nano}}=G_{\mathrm{module}}G_{\mathrm{path}}G_{\mathrm{step}}G_{\mathrm{state}}G_{\mathrm{kv}}G_{\mathrm{runner}}G_{\mathrm{experiment}}G_{\mathrm{signal}}
```

这组公式的作用，是避免“读过 nano-vLLM”停在文件名层面。你需要证明自己知道：

1. 哪些模块构成 request lifecycle。
2. 哪些状态和资源在模块之间传递。
3. 哪些实验能观察到这些状态变化。
4. 哪些点属于教学项目边界，不应该夸大成生产级能力。

下面的 demo 模拟两类输入：

1. `ModuleNote`：每个源码模块的 layer、涉及资源和可观察信号。
2. `ExperimentTrace`：每个实验触达哪些模块、观察到哪些信号。

```python
from dataclasses import dataclass


@dataclass
class ModuleNote:
    name: str
    layer: str
    resources: tuple
    signals: tuple


@dataclass
class ExperimentTrace:
    name: str
    touched_modules: tuple
    observed_signals: tuple


class ToyNanoVLLMSourceAuditor:
    def __init__(self):
        self.required_modules = [
            "example",
            "llm",
            "sampling_params",
            "sequence",
            "llm_engine",
            "scheduler",
            "block_manager",
            "model_runner",
            "attention",
            "sampler",
            "qwen3_model",
        ]
        self.required_resources = [
            "request_state",
            "waiting_queue",
            "running_queue",
            "kv_block_table",
            "free_blocks",
            "input_ids",
            "positions",
            "slot_mapping",
            "logits",
            "sampled_token",
            "finished_cleanup",
        ]
        self.required_experiments = [
            "trace_generate",
            "prefill_decode_step",
            "scheduler_budget",
            "kv_block_reuse",
            "prefix_cache_hit",
            "decode_finish_cleanup",
            "sampler_params",
        ]

    def coverage(self, seen, required):
        return round(len(set(seen) & set(required)) / max(1, len(required)), 3)

    def audit(self, module_notes, experiments):
        module_names = [note.name for note in module_notes]
        resources_seen = sorted({resource for note in module_notes for resource in note.resources})
        experiment_names = [experiment.name for experiment in experiments]
        observed_signals = sorted(
            {signal for note in module_notes for signal in note.signals}
            | {signal for experiment in experiments for signal in experiment.observed_signals}
        )
        module_positions = {name: index for index, name in enumerate(module_names)}
        lifecycle_order_ok = all(
            module_positions[self.required_modules[i]] < module_positions[self.required_modules[i + 1]]
            for i in range(len(self.required_modules) - 1)
        )
        touched_runtime_modules = sorted(
            {module for experiment in experiments for module in experiment.touched_modules}
        )
        summary = {
            "module_coverage": self.coverage(module_names, self.required_modules),
            "resource_coverage": self.coverage(resources_seen, self.required_resources),
            "experiment_coverage": self.coverage(experiment_names, self.required_experiments),
            "lifecycle_order_ok": lifecycle_order_ok,
            "runtime_module_touch": self.coverage(touched_runtime_modules, self.required_modules),
            "observed_signal_count": len(observed_signals),
            "missing_modules": sorted(set(self.required_modules) - set(module_names)),
            "missing_resources": sorted(set(self.required_resources) - set(resources_seen)),
            "missing_experiments": sorted(set(self.required_experiments) - set(experiment_names)),
        }
        gates = {
            "module_map_complete": summary["module_coverage"] == 1.0,
            "generate_path_visible": "trace_generate" in experiment_names,
            "engine_step_visible": "prefill_decode_step" in experiment_names,
            "sequence_state_visible": "request_state" in resources_seen,
            "scheduler_decision_visible": "scheduler_budget" in experiment_names,
            "kv_lifecycle_visible": (
                "kv_block_reuse" in experiment_names and "finished_cleanup" in resources_seen
            ),
            "runner_batch_contract_visible": all(
                resource in resources_seen for resource in ["input_ids", "positions", "slot_mapping"]
            ),
            "experiments_cover_core_paths": summary["experiment_coverage"] == 1.0,
            "signals_observable": summary["observed_signal_count"] >= 12,
        }
        gates["nano_vllm_source_gate"] = all(gates.values())
        return summary, gates


module_notes = [
    ModuleNote("example", "entry", ("request_state",), ("prompt_list",)),
    ModuleNote("llm", "api", ("request_state",), ("generate_call",)),
    ModuleNote("sampling_params", "api", ("sampled_token",), ("temperature", "max_tokens")),
    ModuleNote(
        "sequence",
        "state",
        ("request_state", "kv_block_table", "sampled_token"),
        ("token_append", "finish_reason"),
    ),
    ModuleNote(
        "llm_engine",
        "engine",
        ("waiting_queue", "running_queue"),
        ("engine_step", "prefill_decode_flag"),
    ),
    ModuleNote(
        "scheduler",
        "scheduler",
        ("waiting_queue", "running_queue", "free_blocks"),
        ("scheduled_tokens", "preempted_seq"),
    ),
    ModuleNote(
        "block_manager",
        "kv",
        ("kv_block_table", "free_blocks", "finished_cleanup"),
        ("block_ref_count", "prefix_hash"),
    ),
    ModuleNote(
        "model_runner",
        "runner",
        ("input_ids", "positions", "slot_mapping", "logits"),
        ("prefill_batch", "decode_batch"),
    ),
    ModuleNote("attention", "layer", ("kv_block_table", "slot_mapping"), ("kv_read", "kv_write")),
    ModuleNote("sampler", "layer", ("logits", "sampled_token"), ("next_token",)),
    ModuleNote("qwen3_model", "model", ("input_ids", "positions", "logits"), ("forward_call",)),
]
experiments = [
    ExperimentTrace("trace_generate", ("example", "llm", "llm_engine"), ("generate_call", "engine_step")),
    ExperimentTrace(
        "prefill_decode_step",
        ("llm_engine", "scheduler", "model_runner"),
        ("prefill_batch", "decode_batch"),
    ),
    ExperimentTrace("scheduler_budget", ("scheduler", "block_manager"), ("scheduled_tokens", "preempted_seq")),
    ExperimentTrace(
        "kv_block_reuse",
        ("sequence", "block_manager", "attention"),
        ("block_ref_count", "kv_read", "kv_write"),
    ),
    ExperimentTrace("prefix_cache_hit", ("block_manager", "scheduler"), ("prefix_hash", "scheduled_tokens")),
    ExperimentTrace(
        "decode_finish_cleanup",
        ("sequence", "scheduler", "block_manager"),
        ("finish_reason", "block_ref_count"),
    ),
    ExperimentTrace("sampler_params", ("sampling_params", "sampler"), ("temperature", "next_token")),
]
summary, gates = ToyNanoVLLMSourceAuditor().audit(module_notes, experiments)
print("nano_vllm_source_summary=", summary)
print("nano_vllm_source_gates=", gates)
```

一次运行的核心输出类似：

```text
nano_vllm_source_summary= {'module_coverage': 1.0, 'resource_coverage': 1.0, 'experiment_coverage': 1.0, 'lifecycle_order_ok': True, 'runtime_module_touch': 0.909, 'observed_signal_count': 18, 'missing_modules': [], 'missing_resources': [], 'missing_experiments': []}
nano_vllm_source_gates= {'module_map_complete': True, 'generate_path_visible': True, 'engine_step_visible': True, 'sequence_state_visible': True, 'scheduler_decision_visible': True, 'kv_lifecycle_visible': True, 'runner_batch_contract_visible': True, 'experiments_cover_core_paths': True, 'signals_observable': True, 'nano_vllm_source_gate': True}
```

这个 demo 说明，源码学习不能只写“我读了 `scheduler.py` 和 `block_manager.py`”。更可靠的说法是：

1. 我有模块地图：入口、API、sequence、engine、scheduler、KV、runner、attention、sampler、model。
2. 我有 request lifecycle：从 `generate()` 到 engine step，再到 sampler 和 sequence append。
3. 我有资源生命周期：KV block allocate、reuse、read/write、finish cleanup。
4. 我有实验信号：generate trace、prefill/decode step、scheduler budget、prefix cache hit、sampler params。
5. 我知道教学项目边界：它适合建立骨架，不代表生产级 vLLM 的完整控制面和故障恢复。

所以本章最终门禁是 `nano_vllm_source_gate`：只有模块、路径、engine step、sequence state、scheduler decision、KV lifecycle、runner batch contract、实验覆盖和信号可观测都成立，才算真正把 nano-vLLM 读成了可复盘的推理框架源码路线。

## 44.23 小练习

1. 画出 nano-vLLM 从 `LLM.generate()` 到返回文本的完整调用路径。
2. 总结 `Sequence` 中哪些字段属于请求输入，哪些字段属于执行状态，哪些字段属于 KV 状态。
3. 画出 scheduler 中 waiting/running 队列的状态变化。
4. 解释 block manager 如何为一个新请求分配 KV block。
5. 解释 decode 生成新 token 时，sequence、block manager 和 attention 分别发生什么变化。
6. 给 nano-vLLM 增加 TTFT、TPOT 和总 latency 统计。
7. 给 scheduler 增加一轮 step 的 debug 日志。
8. 实现一个简单的 prefill token budget，并压测不同 budget 的效果。
9. 实现一个简单 chunked prefill，并观察 decode TPOT 是否更稳定。
10. 写一个面试回答：nano-vLLM 和生产级 vLLM 的核心差距是什么？

## 44.24 本章总结

nano-vLLM 是学习 vLLM-like 推理框架的好入口。

它的价值不是完整复刻生产级 vLLM，而是用较小代码展示 serving engine 的核心骨架：`LLM` API、engine loop、sequence state、scheduler、block manager、model runner、attention 和 sampler。

阅读 nano-vLLM 时，应该沿请求生命周期从上到下读，而不是从模型层开始逐行看。第一遍追踪一次 `generate()`，第二遍看 `Sequence`，第三遍看 `Scheduler`，第四遍看 `BlockManager`，第五遍看 `ModelRunner`，最后再看 attention、sampler 和模型结构。

真正掌握 nano-vLLM 的标志，是你能讲清一轮 engine step 中 scheduler、KV block、model runner、attention、sampler 和 sequence state 如何协作。

下一章会继续看另一个教学项目：`tiny-llm`。它的侧重点会更偏从极简 LLM 实现理解模型、推理和工程结构之间的关系。
