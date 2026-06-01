# 第 27 章 SGLang Runtime 总览

上一章讲了 SGLang 解决什么问题：它不是只把一次 `generate()` 跑快，而是让复杂 LLM programs 的表达和执行更高效。

从本章开始，我们进入 SGLang runtime 的内部结构。

如果只从外部使用 SGLang，你看到的可能是：

```bash
python -m sglang.launch_server --model-path <MODEL>
```

或者：

```python
import sglang as sgl

llm = sgl.Engine(model_path="<MODEL>")
outputs = llm.generate(prompts, sampling_params)
```

但从内部看，SGLang runtime 要完成一整条复杂链路：接收请求、解析参数、tokenize、匹配 prefix cache、分配 KV cache、调度 prefill/decode、执行模型、采样、结构化约束、streaming、释放或保留 cache。

一句话概括：

> SGLang Runtime 是把 OpenAI-compatible API、native `/generate`、offline engine 和 SGLang frontend 程序统一执行到高性能模型引擎上的系统层；它的核心模块包括入口服务、请求状态、scheduler、RadixAttention、memory pool、model runner、sampler、grammar backend 和 streaming 输出。

## 27.1 本章目标

读完本章，你应该能讲清：

1. SGLang Runtime 在整个 SGLang 系统中的位置。
2. OpenAI-compatible API、native API、offline engine 和 frontend language 的关系。
3. 一个请求在 SGLang Runtime 内部会经过哪些模块。
4. Scheduler 为什么要同时理解 prefill、decode、cache 和 grammar。
5. RadixAttention、memory pool、model runner、sampler 分别负责什么。
6. Structured output 在 runtime 中插入在哪里。
7. 面试中如何画出 SGLang Runtime 总体架构图。

## 27.2 先给一张总览图

SGLang Runtime 可以先按下面这张图理解：

```text
Client / Application
  |
  |-- OpenAI-compatible API
  |-- Native /generate API
  |-- Offline Engine API
  |-- SGLang frontend program
  v
API Server / Engine Frontend
  |
  |-- request parsing
  |-- chat template / tokenize
  |-- sampling params
  |-- structured output constraints
  v
Runtime Scheduler
  |
  |-- waiting queue
  |-- running batch
  |-- prefill/decode scheduling
  |-- cache-aware scheduling
  v
Memory and Cache Layer
  |
  |-- KV cache memory pool
  |-- request-to-token mapping
  |-- RadixAttention prefix tree
  |-- eviction / reuse
  v
Model Execution Layer
  |
  |-- model runner
  |-- attention backend
  |-- tensor parallel / data parallel
  |-- CUDA graph / kernels
  v
Sampling and Output Layer
  |
  |-- logits processing
  |-- temperature / top-p / top-k
  |-- grammar mask / constrained decoding
  |-- detokenization
  |-- streaming response
```

这张图有两个重点。

第一，SGLang Runtime 不只是 HTTP server。HTTP server 只是入口，真正复杂的是后面的调度、cache、模型执行和采样。

第二，SGLang Runtime 也不只是 RadixAttention。RadixAttention 是非常关键的 cache reuse 机制，但它必须和 scheduler、memory pool、model runner、sampler 配合，才能真正提升端到端吞吐和延迟。

## 27.3 Runtime 在 SGLang 系统中的位置

SGLang 可以分成两层来看：

```text
Frontend language / API layer
  -> 负责表达任务

Backend runtime / SRT
  -> 负责高效执行任务
```

前端层关注：

1. 用户如何写 prompt。
2. 如何表达多次 `gen`。
3. 如何表达 `fork`、`choices`、并行和控制流。
4. 如何表达 JSON、regex、EBNF、tool call 等结构化约束。
5. 如何接入 OpenAI-compatible client。

后端 runtime 关注：

1. 怎么把请求合并成 batch。
2. 怎么减少重复 prefill。
3. 怎么管理 KV cache 显存。
4. 怎么调度 prefill 和 decode。
5. 怎么调用模型 forward。
6. 怎么采样下一个 token。
7. 怎么保证 structured output 约束。
8. 怎么把 token 流式返回给用户。

因此，SGLang 的关键设计不是“前端语法好用”或“后端 serving 快”二选一，而是前后端协同。

如果前端能表达复杂程序，但后端看不到共享前缀和并行关系，性能很难好。

如果后端很快，但前端只能提交一堆互相独立的字符串请求，runtime 也很难知道这些请求之间有哪些可以复用。

## 27.4 SRT 是什么

SGLang Runtime 常被简称为 SRT，可以理解为 SGLang 的 serving runtime。

它承担的职责包括：

1. 启动模型服务。
2. 加载模型权重和 tokenizer。
3. 暴露 OpenAI-compatible API 和 native API。
4. 支持 offline engine 直接调用。
5. 接收文本、token ids、多模态输入等请求。
6. 维护请求状态。
7. 做 continuous batching。
8. 管理 KV cache 和 prefix cache。
9. 执行模型 forward。
10. 采样 token。
11. 支持 structured output。
12. 支持 streaming。
13. 输出指标和日志。

注意这里的 runtime 和“Python 解释器运行一段 SGLang frontend 函数”不是同一个层级。

Frontend language 可以生成很多后端请求，SRT 负责把这些请求高效跑在 GPU 上。

## 27.5 四种入口：OpenAI API、native API、offline engine、frontend language

SGLang Runtime 常见有四类入口。

第一类是 OpenAI-compatible API。

这类入口适合把 SGLang 当成替代 OpenAI API 的本地服务：

```text
client.chat.completions.create(...)
  -> /v1/chat/completions
  -> SGLang server
```

它的优点是生态兼容好，应用层迁移成本低。

第二类是 native `/generate` API。

它更接近 runtime 的底层能力，常见请求形态是：

```json
{
  "text": "The capital of France is",
  "sampling_params": {
    "temperature": 0,
    "max_new_tokens": 32
  },
  "stream": true
}
```

native API 可以直接传 `text`、`input_ids`、`sampling_params`、`stream`、structured output 参数等。

第三类是 offline engine。

它不需要 HTTP server，直接在 Python 进程中调用：

```python
llm = sgl.Engine(model_path="qwen/qwen2.5-0.5b-instruct")
outputs = llm.generate(prompts, sampling_params)
```

offline engine 适合：

1. 离线批处理。
2. 数据生成。
3. evaluation。
4. 自定义 server。
5. 不想引入 HTTP 开销的场景。

第四类是 SGLang frontend language。

它面向复杂 LLM programs，比如 `gen`、`fork`、`choices`、多轮状态和工具交互。前端程序会把多步生成任务提交给 runtime 执行，runtime 负责复用 cache 和调度。

这四类入口看起来不同，但最后都会落到 runtime 的核心执行链路。

## 27.6 请求进入 runtime 后发生什么

一个文本生成请求进入 SGLang Runtime 后，大致会经历这些步骤：

```text
1. API 接收请求
2. 解析 prompt、messages、sampling 参数
3. 应用 chat template
4. tokenize 或接收 input_ids
5. 构造 request state
6. 进入 waiting queue
7. scheduler 选择请求
8. RadixAttention 查找可复用 prefix
9. memory pool 分配 KV cache 空间
10. model runner 执行 prefill
11. sampler 采样第一个 token
12. 请求进入 decode 阶段
13. scheduler 每轮组织 running batch
14. model runner 执行 decode forward
15. sampler 采样下一个 token
16. detokenizer / streamer 返回增量文本
17. 命中 stop 条件后结束
18. 释放或保留 KV cache
```

这条链路和 vLLM-like runtime 很像，但 SGLang 有几个额外重点：

1. 更强调跨请求、跨 generation 调用的 prefix cache reuse。
2. 更强调复杂程序产生的并行和分支。
3. 更强调 structured output 是 runtime decoding 的一部分。
4. 更强调 native API、offline engine、frontend language 共享同一套执行后端。

## 27.7 Request state：请求不是一段字符串

在 runtime 内部，请求不是简单的字符串，而是一份动态状态。

它至少包含：

1. request id。
2. 原始文本或 messages。
3. token ids。
4. prompt 长度。
5. 已生成 token 列表。
6. sampling parameters。
7. stop 条件。
8. 是否 stream。
9. structured output 约束。
10. 当前阶段：waiting、prefill、decode、finished。
11. 已分配的 KV cache 位置。
12. prefix cache 命中信息。
13. 输出缓冲区。
14. 错误或取消状态。

为什么要维护这么多状态？

因为 decode 是逐 token 进行的。一个请求不会一次 forward 完就结束，而是在很多 scheduler iteration 中不断被调度、生成、暂停、继续和最终释放。

可以把请求理解成一个小型状态机：

```text
WAITING
  -> PREFILLING
  -> DECODING
  -> FINISHED

也可能：

WAITING
  -> PREFILLING
  -> PREEMPTED / WAITING
  -> DECODING
  -> CANCELLED
```

SGLang 的复杂之处在于：这个状态机还要和 prefix cache、grammar state、frontend program state 结合。

## 27.8 Tokenization 和 chat template

OpenAI-compatible API 通常接收的是 `messages`：

```json
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Explain KV cache."}
]
```

但模型真正执行的是 token ids。

中间通常要经过：

```text
messages
  -> chat template
  -> prompt text
  -> tokenizer
  -> input_ids
```

这个阶段看似普通，但对 cache reuse 很重要。

如果同一个对话历史，因为 chat template、空格、特殊 token、system prompt 拼接方式不同，生成出的 token 序列不同，那么 prefix cache 就无法命中。

所以线上使用 SGLang 或任何 prefix-cache runtime 时，要注意：

1. 同一模型使用稳定的 chat template。
2. system prompt 不要频繁注入随机字段。
3. RAG prompt 的固定部分尽量固定。
4. few-shot examples 的顺序和格式保持一致。
5. 如果直接传 `input_ids`，要确保上层 tokenizer 和 runtime tokenizer 语义一致。

Prefix cache 的单位不是字符串相似，而是 token prefix 完全匹配。

## 27.9 Sampling params：生成行为如何进入 runtime

SGLang 的 sampling 参数会影响每一步 decode 的 token 选择。

常见参数包括：

1. `max_new_tokens`。
2. `temperature`。
3. `top_p`。
4. `top_k`。
5. `min_p`。
6. `stop`。
7. `stop_token_ids`。
8. `frequency_penalty`。
9. `presence_penalty`。
10. `repetition_penalty`。
11. `json_schema`。
12. `regex`。
13. `ebnf`。
14. `structural_tag`。

这些参数不是只在最后处理文本时用，而是会影响 runtime 内部执行。

例如：

1. `max_new_tokens` 决定请求最多占用多少 decode 轮次和 KV cache。
2. `temperature/top_p/top_k` 决定 sampler 如何从 logits 采样。
3. `stop` 和 `stop_token_ids` 决定何时终止请求。
4. `json_schema/regex/ebnf` 会引入 grammar state 和 logits mask。
5. `stream` 决定每轮生成后是否要把增量输出推给客户端。

所以 scheduler 做决策时，不能只看 prompt 长度，也要考虑输出长度预算、结构化约束和请求是否接近完成。

## 27.10 Scheduler 的核心职责

Scheduler 是 runtime 的中枢。

它要回答的问题是：

```text
下一次模型 forward，应该跑哪些请求、跑哪些 token、用多少 cache、是否允许新请求进入？
```

在 SGLang Runtime 中，scheduler 通常要处理：

1. 等待队列中的新请求。
2. 已经完成 prefill、正在 decode 的请求。
3. 长 prompt prefill 对 decode 的影响。
4. KV cache 剩余空间。
5. prefix cache 命中和复用。
6. batch token budget。
7. batch sequence budget。
8. structured output 的额外开销。
9. streaming 的输出节奏。
10. 请求取消和异常。

可以把 scheduler 想成一个资源分配器：

```text
GPU 计算预算
KV cache 显存预算
每轮 batch 大小预算
请求延迟预算
cache reuse 机会
```

它要在这些预算之间做折中。

## 27.11 Prefill 调度

Prefill 调度处理新请求的输入 prompt。

如果没有 prefix cache，prompt 有多少 token，prefill 就要计算多少 token。

如果命中 RadixAttention，则只需要计算未命中的 suffix。

例如：

```text
cached prefix:  [A B C D]
new prompt:     [A B C D E F]

需要 prefill:           [E F]
```

这对 TTFT 非常关键。

一个长 prompt 如果能命中大段前缀，首 token 延迟会明显下降，GPU 计算也减少。

Prefill 调度要注意几个问题：

1. 长 prompt 不能无限占用一次 iteration。
2. cache 命中请求和 cache 未命中请求的成本不同。
3. 多个新请求可以组成 batched prefill。
4. 长 prefill 可能影响正在 decode 的请求 TPOT。
5. chunked prefill 或类似机制可以缓解长 prompt 阻塞。

SGLang 的 cache-aware scheduling 会尽量利用 RadixAttention 带来的命中机会，但仍然要受显存和 token budget 限制。

## 27.12 Decode 调度

Decode 调度处理已经进入生成阶段的请求。

每轮 decode 通常每个请求只新增一个 token：

```text
running requests: R1, R2, R3, R4
this step input:  last token of each request
output:           next token for each request
```

Decode 阶段的挑战是动态性很强：

1. 有的请求很快遇到 stop。
2. 有的请求要生成很长。
3. 有的请求需要 grammar mask。
4. 有的请求客户端断开。
5. 有的新请求正在等待 prefill。
6. 每个请求的上下文长度不断增长。

Continuous batching 的核心就是：每一轮都重新组织 batch，让完成的请求退出，让新的请求进入，尽量保持 GPU 忙碌。

SGLang Runtime 也需要类似的 iteration-level scheduling，只是它还要额外考虑 prefix cache、structured output、frontend program 产生的多分支请求。

## 27.13 Memory pool：KV cache 从哪里来

LLM serving 中，KV cache 是最重要的动态显存资源。

SGLang Runtime 需要为请求分配 KV cache 空间，用来保存每层 attention 的 key/value。

从概念上看，memory pool 要做这些事：

1. 初始化时预留一批 KV cache 显存。
2. 为新 token 分配 cache slot。
3. 建立 request token 到 cache slot 的映射。
4. 在 decode 时追加新 token 的 KV。
5. 请求结束后释放不需要保留的 cache。
6. 对可复用 prefix，保留 cache 并挂到 prefix tree。
7. 显存不足时触发 eviction 或调度限制。

普通请求结束后，KV cache 可以释放。

但 SGLang 的 RadixAttention 会尽量保留有价值的 prefix cache。这意味着“请求结束”不等于“所有 KV 立刻释放”。

这也是 SGLang 和简单 KV cache 管理不同的地方：

```text
普通 serving：
请求结束 -> KV cache 释放

SGLang with RadixAttention：
请求结束 -> 有价值的 prefix 可能保留 -> 后续请求复用 -> 显存不足时再淘汰
```

因此 memory pool 必须和 RadixAttention 的 eviction 策略协同。

## 27.14 RadixAttention 在 runtime 中的位置

RadixAttention 不是 attention kernel 本身，而是一套自动 KV cache reuse 机制。

它在 runtime 中大致位于：

```text
tokenized prompt
  -> prefix tree lookup
  -> find longest cached prefix
  -> reuse matched KV cache
  -> prefill only unmatched suffix
  -> insert new prompt/output into tree
  -> evict when memory pressure exists
```

RadixAttention 管理的是：

```text
token prefix -> KV cache tensors / cache slots
```

为什么用 radix tree？

因为复杂 LLM programs 的共享关系经常是树状的：

```text
system prompt
  -> few-shot examples
      -> question 1
      -> question 2
      -> question 3

chat history
  -> branch answer A
  -> branch answer B
  -> branch answer C

tree-of-thought root
  -> thought path 1
  -> thought path 2
      -> subpath 2.1
      -> subpath 2.2
```

如果只做简单 exact prompt cache，很难覆盖这些部分共享。

Radix tree 可以高效做 longest prefix match，并在新请求插入时拆分节点，让多个请求共享公共前缀。

## 27.15 Model runner 负责什么

Model runner 是真正执行模型 forward 的模块。

它负责把 scheduler 选出的 batch 转成模型可执行的张量，然后调用 attention、MLP、norm、logits head 等计算。

概念上包括：

1. 准备 input ids 或 input embeddings。
2. 准备 position ids。
3. 准备 attention metadata。
4. 准备 KV cache 读写位置。
5. 调用 transformer forward。
6. 取出最后位置 logits。
7. 处理多 GPU 通信。
8. 返回 logits 给 sampler。

Prefill 和 decode 对 model runner 的要求不同。

Prefill 输入是多个 prompt token，计算更大：

```text
[prompt token 1, prompt token 2, ..., prompt token N]
```

Decode 输入通常是每个请求一个新 token：

```text
[last_token_of_R1, last_token_of_R2, ..., last_token_of_Rk]
```

所以 model runner 需要支持两类 metadata：

1. prefill metadata。
2. decode metadata。

并且它要和 attention backend 约定好 KV cache 的布局和索引方式。

## 27.16 Attention backend 和 kernel

Runtime 的性能很大程度取决于 attention backend。

Attention backend 要解决：

1. prefill attention 怎么算。
2. decode attention 怎么读 KV cache。
3. 不同请求长度如何组织。
4. KV cache layout 如何寻址。
5. 多模态 token 是否有特殊处理。
6. 是否支持 flash attention、paged attention、flashinfer 等实现。
7. 是否支持 CUDA graph 或其他图捕获优化。

对学习者来说，不需要一开始就记住所有 backend 名称和实现细节。

更重要的是理解分层：

```text
Scheduler 决定跑哪些请求
Memory layer 决定 KV 在哪里
Model runner 准备计算 metadata
Attention backend 按 metadata 高效执行 attention
```

如果这几层边界混在一起，阅读源码会很痛苦。

## 27.17 Sampler：logits 到 token

模型 forward 输出 logits 后，还没有生成 token。

Sampler 负责把 logits 变成下一个 token。

典型流程是：

```text
logits
  -> repetition / frequency / presence penalty
  -> grammar mask or custom logit processor
  -> temperature scaling
  -> top-k / top-p / min-p filtering
  -> sample or greedy choose
  -> next token id
```

如果 `temperature = 0`，通常接近 greedy decoding。

如果设置 `top_p`、`top_k`，sampler 会限制候选 token 集合。

如果设置 structured output，sampler 还必须考虑当前 grammar state 下哪些 token 合法。

这说明 structured output 不是“生成完再检查”，而是会进入 token selection 本身。

## 27.18 Structured output 在 runtime 里的位置

SGLang 支持用 JSON schema、regex、EBNF、structural tag 等方式约束输出。

runtime 视角下，它大致是这样工作的：

```text
request sampling params
  -> parse json_schema / regex / ebnf
  -> build grammar state
  -> each decode step:
       logits
       -> compute valid token mask
       -> mask invalid tokens
       -> sample valid next token
       -> update grammar state
```

这样做的好处是：

1. 输出天然满足约束的概率更高。
2. 减少生成后 JSON parse 失败。
3. 减少后处理和重试。
4. 工具调用格式更可控。
5. 可以和 OpenAI-compatible API 的 `response_format` 对齐。

但它也有成本：

1. 每个请求可能有自己的 grammar state。
2. 每轮 decode 需要计算合法 token mask。
3. 复杂 schema 可能增加 CPU 或 GPU 侧开销。
4. 约束过强可能影响模型输出质量。

所以 structured output 是 correctness 和 performance 的折中，不是免费功能。

## 27.19 Streaming 输出

Streaming 是 runtime 输出层的重要能力。

生成 token 后，runtime 需要把 token 转成文本增量，并通过 SSE、HTTP chunk 或其他协议返回。

需要处理的问题包括：

1. token 到文本的增量 detokenization。
2. Unicode 边界。
3. stop string 跨 token 匹配。
4. special token 是否跳过。
5. 客户端断开。
6. backpressure。
7. finish reason。
8. usage 统计。

Streaming 和 scheduler 也有关。

如果 decode step 抖动很大，用户看到的 streaming 就会一卡一卡。

如果 prefill 长请求长时间占用 GPU，正在 streaming 的请求也可能出现输出间隔变长。

所以优化 streaming 体验，不只是改 HTTP flush，而是要看 scheduler、decode step time 和 batch 组成。

## 27.20 Offline engine 为什么重要

很多人只关注在线 server，但 SGLang 的 offline engine 也很重要。

Offline engine 直接调用 runtime，不经过 HTTP server：

```text
Python process
  -> sgl.Engine
  -> runtime scheduler
  -> model runner
  -> outputs
```

它适合：

1. 离线数据合成。
2. 批量评测。
3. RL rollout。
4. 自定义服务框架。
5. notebook 或实验脚本。
6. 避免 HTTP 序列化和网络开销。

从架构上看，offline engine 的价值是把 runtime 能力从“必须通过 HTTP server 使用”中解耦出来。

这对工程团队很重要。你可以用同一套底层 runtime 同时服务在线和离线场景，而不是维护两套推理逻辑。

## 27.21 SGLang Runtime 和 vLLM Runtime 的相似点

SGLang Runtime 和 vLLM Runtime 有很多相似点。

共同点包括：

1. 都面向高性能 LLM serving。
2. 都要做 prefill 和 decode。
3. 都要做 continuous batching。
4. 都要管理 KV cache。
5. 都要支持 streaming。
6. 都要处理 sampling 参数。
7. 都要支持 OpenAI-compatible API。
8. 都要面对显存、吞吐、延迟和调度折中。

所以不要把两者理解成完全不同的系统。

它们共享同一个底层问题：

```text
如何在 GPU 上高效执行大量自回归生成请求？
```

## 27.22 SGLang Runtime 和 vLLM Runtime 的差异点

差异主要在问题重心。

vLLM 更突出：

1. PagedAttention。
2. KV block 管理。
3. high-throughput serving。
4. OpenAI-compatible API。
5. 大量独立请求的调度效率。

SGLang 更突出：

1. complex LLM programs。
2. frontend language 和 runtime 协同。
3. RadixAttention 自动 prefix reuse。
4. multi-call、branch、parallel、agent workload。
5. structured output 和 constrained decoding。
6. offline engine 和 native API 对复杂任务的支持。

可以这样记：

```text
vLLM: 高效执行大量生成请求
SGLang: 高效执行复杂 LLM programs，同时也支持高性能 serving
```

这不是绝对边界。两个项目都在演进，也会互相吸收能力。面试中不要把它们说成“谁完全替代谁”，而要说清楚设计重心和适用 workload。

## 27.23 一个请求的端到端例子

假设用户发送一个结构化抽取请求：

```json
{
  "text": "Extract the city and population from the following paragraph...",
  "sampling_params": {
    "temperature": 0,
    "max_new_tokens": 128,
    "json_schema": "{...}"
  },
  "stream": false
}
```

runtime 内部可以这样理解：

```text
API server
  -> parse request
  -> tokenize text
  -> parse json_schema
  -> create request state

Scheduler
  -> put request into waiting queue
  -> check token budget and KV budget
  -> select request for prefill

RadixAttention
  -> search token prefix
  -> reuse matched KV if any
  -> decide suffix to prefill

Memory pool
  -> allocate KV slots for uncached tokens

Model runner
  -> run prefill
  -> produce logits for first output token

Sampler + grammar backend
  -> apply JSON grammar mask
  -> choose first token
  -> update grammar state

Decode loop
  -> scheduler selects request each iteration
  -> model runner runs one-token decode
  -> grammar mask filters invalid tokens
  -> sampler chooses next token
  -> stop when JSON complete or max_new_tokens reached

Output layer
  -> detokenize
  -> return JSON text and usage

Cache layer
  -> keep useful prefix if policy allows
  -> release non-reusable temporary cache
```

这个例子说明：一个看似简单的 JSON 输出请求，会经过 scheduler、cache、model runner、sampler 和 grammar backend 多个模块。

## 27.24 SGLang Runtime 的关键指标

理解 runtime 架构后，指标也更容易拆。

入口层指标：

1. request QPS。
2. request parse latency。
3. tokenize latency。
4. HTTP queue time。
5. error rate。

调度层指标：

1. waiting queue length。
2. running request count。
3. scheduled prefill tokens。
4. scheduled decode tokens。
5. batch size。
6. preemption 或 reject 次数。

cache 层指标：

1. KV cache usage。
2. prefix cache hit rate。
3. matched prefix length。
4. eviction count。
5. memory pool free slots。

模型执行层指标：

1. prefill latency。
2. decode step latency。
3. input tokens/s。
4. output tokens/s。
5. GPU utilization。
6. memory bandwidth pressure。

输出层指标：

1. TTFT。
2. TPOT。
3. E2E latency。
4. streaming interval。
5. finish reason 分布。
6. structured output failure 或 retry 次数。

如果线上服务慢，不要只看 QPS。要按这几层拆，才能知道慢在 tokenize、queue、prefill、decode、grammar mask、streaming 还是网络。

## 27.25 常见误解

误解一：SGLang Runtime 就是一个 OpenAI API server。

不对。OpenAI-compatible API 只是入口之一，runtime 的核心是调度、cache、模型执行和采样。

误解二：SGLang 的核心只有 RadixAttention。

不对。RadixAttention 很重要，但没有 scheduler、memory pool、model runner 和 sampler 配合，无法单独构成 serving runtime。

误解三：structured output 是后处理。

不准确。在 SGLang Runtime 中，JSON schema、regex、EBNF 等约束可以进入 constrained decoding，在每一步采样前限制合法 token。

误解四：prefix cache 命中看字符串相似度。

不对。真正匹配的是 token prefix。字符串看起来相似，如果模板、空格或 special token 不同，也可能无法复用。

误解五：offline engine 只是 demo。

不对。offline engine 对批处理、评测、RL rollout 和自定义 server 很有价值。

## 27.26 面试官会怎么问

问题一：请画一下 SGLang Runtime 的整体架构。

回答要点：入口包括 OpenAI-compatible API、native `/generate`、offline engine 和 frontend language；进入 runtime 后经过 request parsing、tokenization、scheduler、RadixAttention、memory pool、model runner、sampler、grammar backend 和 streaming output。

问题二：SGLang Runtime 和普通 HTTP 模型服务有什么区别？

回答要点：普通 HTTP 服务只负责接请求和返回结果，SGLang Runtime 是 GPU 执行引擎，需要处理 continuous batching、KV cache、prefix reuse、structured decoding、streaming 和多请求调度。

问题三：RadixAttention 在 runtime 哪个阶段生效？

回答要点：在请求 tokenized 后、prefill 前，runtime 用 radix tree 查找最长可复用 prefix，只对未命中 suffix 做 prefill；请求结束或生成过程中，新的 token 序列和 KV cache 会插入 prefix tree，显存不足时按策略淘汰。

问题四：structured output 为什么属于 runtime？

回答要点：因为它不是生成完再 parse，而是在每一步 decode 时根据 JSON schema、regex 或 EBNF 生成合法 token mask，作用在 logits/sampling 之前。

问题五：SGLang 的 offline engine 有什么意义？

回答要点：它绕过 HTTP server，直接使用 runtime scheduler 和 model runner，适合离线批处理、评测、RL rollout、自定义 server 和实验脚本。

## 27.27 标准回答模板

如果面试官问“SGLang Runtime 是怎么工作的”，可以这样回答：

```text
SGLang Runtime 可以理解为 SGLang 的高性能执行后端。它对外支持 OpenAI-compatible API、native /generate API、offline engine，也可以承接 SGLang frontend language 产生的复杂 LLM program 请求。

一个请求进入后，server 会先解析 messages、prompt、sampling 参数和 structured output 约束，然后应用 chat template 和 tokenizer 得到 token ids，并构造 request state。请求进入 scheduler 后，scheduler 会根据 waiting queue、running batch、prefill/decode token budget 和 KV cache 空间决定下一轮执行哪些请求。

在 prefill 前，RadixAttention 会用 radix tree 查找最长 token prefix，复用已有 KV cache，只计算未命中的 suffix。Memory pool 负责分配和回收 KV cache slot。Model runner 根据 scheduler 给出的 batch 和 KV metadata 执行 prefill 或 decode forward，拿到 logits 后交给 sampler。Sampler 会应用 temperature、top-p、top-k、penalty 等参数；如果有 JSON schema、regex 或 EBNF，还会通过 grammar backend 做 constrained decoding，mask 掉非法 token。

生成 token 后，runtime 更新请求状态、KV cache、grammar state，并按需要 stream 给客户端。请求结束后，普通临时 cache 会释放，有复用价值的 prefix cache 可能保留在 RadixAttention 的树结构里，后续请求可以继续命中。
```

## 27.28 小练习

1. 画出一个 SGLang Runtime 请求从 API server 到 streaming output 的完整链路。
2. 解释为什么 tokenizer 和 chat template 会影响 prefix cache 命中率。
3. 对比 prefill 调度和 decode 调度分别关注什么。
4. 说明 RadixAttention 为什么需要和 memory pool 协同。
5. 写一个 JSON schema constrained decoding 请求，并标出 grammar backend 在 runtime 中的位置。
6. 解释 offline engine 和 HTTP server 入口有什么不同、底层又有什么相同。
7. 假设 TTFT 高但 TPOT 正常，按 SGLang Runtime 模块列出排查路径。

## 27.29 本章总结

SGLang Runtime 是 SGLang 的执行核心。它不是简单的 HTTP wrapper，也不是单独一个 RadixAttention 数据结构，而是一套完整的 LLM serving engine。

它把不同入口的请求统一落到一条执行链路：解析请求、tokenize、构造 request state、进入 scheduler、查找 prefix cache、分配 KV cache、执行 prefill/decode、采样 token、应用 structured output 约束、streaming 输出、释放或保留 cache。

理解这条链路后，后面的章节就更容易展开：第 28 章会专门深入 RadixAttention 与 prefix sharing，第 29 章会拆 SGLang scheduler 设计，第 30 章会深入 structured generation 与 constrained decoding。
