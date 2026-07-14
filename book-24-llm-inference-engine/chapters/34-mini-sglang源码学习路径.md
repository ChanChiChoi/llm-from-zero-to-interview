# 第 34 章 mini-sglang 源码学习路径

上一章对比了 SGLang 和 vLLM 的架构：vLLM 更强调通用高并发 serving，SGLang 更强调复杂 LLM programs 的表达和高效执行。

本章把 SGLang 这部分收束到源码学习路径。

这里说的 `mini-sglang` 指的是 SGLang 官方教学项目，目标不是覆盖完整生产系统，而是用更小、更可读的代码帮助你理解 SGLang 的核心设计。它适合在读真实 SGLang 源码之前作为过渡。

项目链接：

```text
https://github.com/sgl-project/mini-sglang
```

一句话概括：

> 读 mini-sglang 不要从文件名开始，而要带着 runtime 主链路去读：入口 API、请求状态、scheduler、KV cache、RadixAttention、model runner、sampler、structured output 和 streaming，每读一个模块都问它在请求生命周期中解决什么问题。

## 34.0 本讲资料边界与第二轮精修口径

本讲第二轮精修时，主要参考三类资料：

1. `sgl-project/mini-sglang` 官方仓库和 README。README 将 mini-sglang 定位为约 5000 行 Python 的紧凑 SGLang 实现，并列出 Radix Cache、Chunked Prefill、Overlap Scheduling、Tensor Parallelism、FlashAttention / FlashInfer backend 等核心学习点。
2. mini-sglang 仓库中的 system architecture / structure 文档。它把教学实现拆成 API server、request / tokenizer、scheduler、engine、KV cache、Radix cache、attention backend、message / protocol 等模块，适合映射到本书前面讲过的 runtime 生命周期。
3. SGLang 论文和官方文档，以及本书第 26 到 33 章对 SGLang runtime、RadixAttention、scheduler、structured output、speculative decoding、agent serving 和 vLLM 对比的内部口径。

本章的边界也要说清：

1. 本章是源码学习方法，不是完整源码逐行讲解；文件名、类名和实现细节可能随 mini-sglang 版本变化。
2. 本章不把 mini-sglang 当生产级 SGLang。教学实现的价值是帮助建立 runtime 骨架，真实 SGLang 还包含更多模型、硬件后端、分布式、grammar backend、tool parser、speculative decoding 和生产监控细节。
3. 本章新增 demo 是源码路径审计器，用来检查阅读笔记是否覆盖模块、生命周期、资源和实验，而不是执行真实 mini-sglang。
4. 读源码的验收标准不是“看过多少文件”，而是能把模块输入输出、状态变化、KV 生命周期、scheduler 输出和实验观察映射到 TTFT、TPOT、prefix hit、KV pressure、grammar mask 和 streaming 指标。

## 34.1 本章目标

读完本章，你应该能讲清：

1. 为什么要先读 mini-sglang，而不是直接啃完整 SGLang。
2. 阅读 mini-sglang 前应该具备哪些概念地图。
3. 如何按请求生命周期读源码。
4. 如何把代码模块映射到 SGLang Runtime 总览。
5. 如何重点观察 RadixAttention、scheduler、structured output。
6. 如何做源码笔记和调试实验。
7. 面试中如何用 mini-sglang 证明你理解推理框架。

## 34.2 为什么需要教学项目

真实 SGLang 是生产级项目。

它会包含大量工程细节：

1. 多模型支持。
2. 多硬件后端。
3. 多 GPU 并行。
4. CUDA graph。
5. attention backend。
6. grammar backend。
7. speculative decoding。
8. tool parser。
9. metrics 和 tracing。
10. 分布式部署。

这些都重要，但初学源码时很容易迷路。

教学项目的价值是把主线缩小：

```text
先看清请求如何跑通，再看生产系统如何扩展。
```

mini-sglang 的学习目标不是“记住所有类名”，而是建立一套源码阅读坐标系。

## 34.3 阅读前的概念地图

读 mini-sglang 前，建议先把前面几章的概念串起来。

核心链路：

```text
request
  -> tokenize
  -> scheduler
  -> prefix cache / RadixAttention
  -> KV cache allocation
  -> prefill
  -> decode loop
  -> sampling
  -> output / streaming
```

核心状态：

```text
WAITING
  -> PREFILLING
  -> DECODING
  -> FINISHED / ABORTED
```

核心资源：

```text
GPU compute
KV cache memory
token budget
sequence budget
prefix cache entries
grammar state
```

核心优化：

```text
continuous batching
RadixAttention
structured generation
speculative decoding
```

读源码时，每看到一个函数，就把它放回这几张图里。

## 34.4 不建议的阅读方式

不建议这样读：

```text
打开 repo -> 从第一个文件开始 -> 每个类都看 -> 试图一次看懂全部
```

这样很容易陷入：

1. 文件名很多，不知道主线。
2. 配置细节太多，忘了请求生命周期。
3. 看到 scheduler 不知道它的输入输出。
4. 看到 cache 不知道它和 model runner 的关系。
5. 看到 frontend 语法不清楚 runtime 如何执行。

更好的方式是：

```text
先跑通最小例子
  -> 找入口
  -> 跟一个请求
  -> 画状态机
  -> 画模块关系
  -> 再读每个优化点
```

## 34.5 推荐阅读顺序

建议按 8 步读。

第一步：启动和最小请求。

第二步：API 或 Engine 入口。

第三步：请求对象和状态机。

第四步：scheduler 主循环。

第五步：KV cache 和 memory pool。

第六步：RadixAttention / prefix sharing。

第七步：model runner、prefill、decode、sampler。

第八步：structured output、tool use、streaming 等扩展。

每一步都要回答三个问题：

1. 这个模块输入是什么？
2. 输出是什么？
3. 它改变了哪些状态或资源？

## 34.6 第一步：跑通最小请求

如果你拿到 mini-sglang 源码，第一步不是读代码，而是跑通最小请求。

目标是看到：

```text
prompt -> output text
```

你要记录：

1. 启动命令。
2. 模型路径。
3. 请求格式。
4. 输出格式。
5. 是否支持 streaming。
6. 日志里出现了哪些模块名。

如果项目提供 HTTP server，就发一个最小 `/generate` 或 chat 请求。

如果项目提供 offline engine，就直接在 Python 里调用 generate。

这一阶段不要优化性能，也不要改代码。

目标只是确定主链路可运行。

## 34.7 第二步：找到入口层

入口层通常负责：

1. 接收请求。
2. 解析 prompt 或 messages。
3. 解析 sampling params。
4. 调 tokenizer。
5. 构造内部 request。
6. 把请求交给 runtime。

你要找的问题：

```text
外部 JSON 请求在哪里变成内部 Request 对象？
```

建议做一张表：

| 外部字段 | 内部字段 | 作用 |
| --- | --- | --- |
| prompt / text | input_ids | 模型输入 token |
| max_new_tokens | output budget | 生成上限 |
| temperature | sampling params | 控制采样 |
| stream | output mode | 是否流式返回 |
| json_schema | grammar constraint | 结构化输出 |

入口层看懂后，后面的 scheduler 才有上下文。

## 34.8 第三步：请求对象和状态机

接下来找 Request 类或类似结构。

重点字段通常包括：

1. request id。
2. input ids。
3. output ids。
4. sampling params。
5. status。
6. arrival time。
7. prompt length。
8. num computed tokens。
9. KV cache references。
10. prefix cache hit info。
11. finish reason。

你要画出状态变化：

```text
WAITING -> PREFILLING -> DECODING -> FINISHED
```

并回答：

1. 请求什么时候进入 waiting？
2. 什么时候算完成 prefill？
3. 什么时候进入 decode？
4. stop 条件在哪里判断？
5. 资源在哪里释放？

如果只看模型 forward，不看 request state，就读不懂 serving engine。

## 34.9 第四步：scheduler 主循环

Scheduler 是最重要的模块之一。

你要找到类似下面的主循环：

```python
while True:
    add_new_requests()
    schedule = scheduler.schedule()
    outputs = model_runner.forward(schedule)
    update_states(outputs)
```

重点看 scheduler 的输入：

1. waiting queue。
2. running requests。
3. token budget。
4. sequence budget。
5. KV cache 状态。
6. prefix cache 命中。

重点看 scheduler 的输出：

1. scheduled prefill requests。
2. scheduled decode requests。
3. 每个请求本轮处理 token 数。
4. KV allocation plan。
5. attention metadata。

读 scheduler 时不要只问“它用了什么算法”，要问：

```text
它如何在 TTFT、TPOT、吞吐和显存之间取舍？
```

## 34.10 第五步：KV cache 和 memory pool

KV cache 是 runtime 的核心资源。

你要找：

1. KV cache 初始化在哪里。
2. 每个 token 的 KV 写到哪里。
3. request 如何记录自己的 KV 位置。
4. decode 如何读取历史 KV。
5. 请求结束后如何释放。
6. prefix cache 是否会保留部分 KV。
7. ref count 或 pin 机制在哪里。

可以画一张映射图：

```text
request token positions
  -> logical positions
  -> KV slots / pages
  -> actual K/V tensors
```

如果 mini-sglang 为了教学简化了 paged layout，也没关系。

你要理解的是生命周期：

```text
allocate -> write -> read -> append -> release / cache
```

## 34.11 第六步：RadixAttention

RadixAttention 是 mini-sglang 最值得重点看的部分之一。

你要找这些操作：

1. match prefix。
2. longest prefix length。
3. insert new path。
4. split node。
5. evict node。
6. pin / unpin cached KV。
7. LRU 或访问时间更新。

建议手工构造三个请求：

```text
R1: A B C D
R2: A B E F
R3: A B E G
```

然后观察 radix tree 如何变化：

```text
root
  -> [A B]
      -> [C D]
      -> [E]
          -> [F]
          -> [G]
```

如果源码里没有可视化工具，可以自己加 debug print。

重点不是打印漂亮，而是确认你理解 split 和 insert。

## 34.12 第七步：model runner

Model runner 把 scheduler 的计划转成模型 forward。

你要看：

1. Prefill 输入怎么构造。
2. Decode 输入怎么构造。
3. Position ids 怎么处理。
4. KV slot mapping 怎么传给 attention。
5. Logits 从哪里取。
6. Sampler 在哪里调用。
7. 输出 token 如何回写 request。

Prefill 和 decode 要分开看。

Prefill：

```text
多个 prompt tokens -> 写一批 KV -> 产生 first token logits
```

Decode：

```text
每个 request 一个新 token -> 读历史 KV -> 写新 token KV -> 产生 next logits
```

如果这个边界看清楚，后面读完整 SGLang 的 attention backend 会容易很多。

## 34.13 第八步：sampler

Sampler 负责 logits 到 token。

你要看：

1. Greedy 在哪里实现。
2. Temperature 在哪里处理。
3. Top-k / top-p 在哪里处理。
4. Stop token 在哪里判断。
5. 多请求不同 sampling params 怎么处理。
6. Structured output mask 是否进入 sampler。

普通 sampler 流程：

```text
logits
  -> penalty
  -> temperature
  -> top-k/top-p
  -> sample
```

Structured sampler 流程：

```text
logits
  -> grammar mask
  -> sampling filters
  -> sample
  -> update grammar state
```

如果 mini-sglang 没有完整 structured output，也可以记录它缺失在哪里，以及完整 SGLang 会如何补上。

## 34.14 Structured output 源码怎么读

如果 mini-sglang 包含 structured output，建议这样读：

1. JSON schema 或 regex 在入口哪里解析。
2. Grammar object 在哪里创建。
3. Grammar state 是否按 request 保存。
4. 每步 valid token mask 在哪里计算。
5. Mask 如何作用到 logits。
6. Grammar state 如何 accept token。
7. 请求结束时如何清理 grammar state。

你要把代码映射到第 30 章的流程：

```text
constraint spec
  -> grammar backend
  -> grammar state
  -> valid token mask
  -> sampler
```

如果源码里用的是简化 grammar，也要抓住这个接口边界。

## 34.15 Tool use 源码怎么读

如果 mini-sglang 包含 tool use 或 tool parser，重点看：

1. Tools schema 如何进入请求。
2. Chat template 如何渲染 tools。
3. 模型输出在哪里解析成 tool calls。
4. Streaming tool call arguments 如何拼接。
5. Tool result 如何作为 message 回灌。
6. Tool parser 和 structured output 是否分离。

注意工具执行通常不属于 runtime 核心。

源码里可能只做 parser，不真的调用外部 API。

这是合理的，因为真实工具执行应该在应用层处理权限、超时和安全。

## 34.16 Streaming 源码怎么读

Streaming 是很多教学项目容易低估的部分。

你要看：

1. Token id 如何 detokenize 成文本。
2. 增量文本如何避免重复。
3. Unicode 边界如何处理。
4. stop string 是否跨 token 检查。
5. 客户端断开如何 abort。
6. finish event 如何发送。

Streaming 和 scheduler 是相互影响的。

如果 scheduler 长时间不推进 decode，streaming 就会卡顿。

所以读 streaming 时要回到 TPOT 和 decode step latency。

## 34.17 推荐做的 8 个实验

只读源码不够，建议做小实验。

实验一：最小 prompt 生成。

确认请求生命周期。

实验二：两个共享 system prompt 的请求。

观察 prefix cache 命中。

实验三：三个分支 prompt。

观察 radix tree split。

实验四：长 prompt + 短 prompt 混合。

观察 scheduler 如何安排 prefill/decode。

实验五：max_new_tokens 很小。

观察请求何时 finished 和释放资源。

实验六：streaming 请求。

观察 token 输出节奏。

实验七：结构化输出。

观察 grammar mask 或等价约束在哪里生效。

实验八：模拟 abort。

观察 KV cache 和 request state 是否清理。

每个实验都要记录：

1. 输入。
2. 关键日志。
3. 状态变化。
4. KV cache 变化。
5. 输出。
6. 你学到的模块边界。

## 34.18 源码笔记模板

建议每个模块用同一个模板记笔记。

```text
模块名：Scheduler

位置：...

解决的问题：每个 engine step 选择 prefill/decode 请求

输入：waiting queue、running requests、token budget、KV state

输出：scheduled batch、KV allocation、metadata

核心状态：request.status、num_computed_tokens、KV refs

关键函数：...

我画的流程图：...

还没看懂的问题：...
```

这种笔记比摘抄源码有用。

面试时你能用它讲清楚模块职责。

## 34.19 如何从 mini-sglang 过渡到真实 SGLang

读完 mini-sglang 后，再读真实 SGLang。

过渡方式：

1. 用同一条请求生命周期找入口。
2. 找真实 SRT 的 request 数据结构。
3. 找 scheduler 主循环。
4. 找 memory pool 和 KV cache 管理。
5. 找 RadixAttention 实现。
6. 找 grammar backend 接入点。
7. 找 speculative decoding worker。
8. 找 OpenAI API 和 native API 映射。

你会发现真实项目复杂很多，但主线不变。

复杂性通常来自：

1. 更多模型。
2. 更多后端。
3. 更多并行方式。
4. 更多配置。
5. 更多异常处理。
6. 更多性能优化。

不要被复杂性吓到，先把它挂到 mini-sglang 的骨架上。

## 34.20 和本书前面章节的对应关系

可以把 mini-sglang 模块映射到本书章节。

| 源码模块 | 对应章节 | 重点问题 |
| --- | --- | --- |
| API / Engine | 第 14、27 章 | 请求如何进入 runtime |
| Request state | 第 12、20、27 章 | 状态机和生命周期 |
| Scheduler | 第 19、20、29 章 | prefill/decode 调度 |
| KV cache | 第 9、18、21 章 | KV 分配、追加、释放 |
| RadixAttention | 第 28 章 | prefix sharing |
| Sampler | 第 8、30 章 | logits 到 token |
| Structured output | 第 30 章 | grammar mask |
| Tool use | 第 32 章 | tool parser 和回灌 |
| Speculative decoding | 第 31 章 | draft/verify/accept |
| Streaming | 第 13、20 章 | 增量输出和 abort |

这张表可以作为源码阅读索引。

## 34.21 面试中怎么讲 mini-sglang

面试官不一定关心你背了多少文件名。

更关心你是否能从源码中抽象系统设计。

你可以这样讲：

1. 我先跑通最小请求。
2. 然后沿请求生命周期看入口、request state、scheduler、KV cache、model runner、sampler、output。
3. 我重点看了 RadixAttention 如何做 longest prefix match、insert、split 和 eviction。
4. 我用几个共享前缀请求验证了 radix tree 如何变化。
5. 我对比了 mini-sglang 和完整 SGLang，理解教学版省略了哪些生产能力。
6. 我能把源码模块映射到 TTFT、TPOT、KV cache、prefix hit 和 structured output 指标。

这比说“我看过 SGLang 源码”更可信。

## 34.22 常见误区

误区一：一上来读完整 SGLang 全部源码。

容易迷路。先用 mini-sglang 建立骨架。

误区二：只看 model forward。

Serving engine 的难点在 scheduler、KV cache、streaming 和状态管理，不只是 forward。

误区三：只记类名。

类名会变，模块职责和请求生命周期更稳定。

误区四：不跑实验。

不跑共享前缀、abort、streaming 等实验，很难真正理解 runtime 行为。

误区五：把教学实现当生产实现。

mini-sglang 是学习路线，不是完整生产系统。要知道它省略了什么。

## 34.23 面试官会怎么问

问题一：你会怎么读 mini-sglang 源码？

回答要点：先跑通最小请求，再沿 request lifecycle 阅读入口、request state、scheduler、KV cache、RadixAttention、model runner、sampler、output，不从文件顺序乱读。

问题二：读 scheduler 时看什么？

回答要点：看 waiting/running 输入、token budget、sequence budget、KV budget、cache hit length、输出的 prefill/decode batch 和 execution metadata。

问题三：读 RadixAttention 时看什么？

回答要点：看 longest prefix match、node split、insert、eviction、KV ref、pin/unpin，以及它如何影响 prefill suffix length。

问题四：mini-sglang 和完整 SGLang 的关系？

回答要点：mini-sglang 是教学项目，用更小代码解释核心 runtime；完整 SGLang 有更多模型、后端、并行、grammar、speculative decoding、tool parser 和生产工程细节。

问题五：怎么证明你真的看懂了？

回答要点：能画请求生命周期图，能解释状态变化和 KV cache 生命周期，能用共享前缀实验展示 radix tree 变化，能把源码模块和 TTFT/TPOT/cache hit 指标对应起来。

## 34.24 标准回答模板

如果面试官问“你如何学习 mini-sglang 源码”，可以这样回答：

```text
我不会从文件列表顺序读，而是先跑通一个最小 generate 请求，然后沿请求生命周期读源码。第一步看外部请求在哪里被解析、tokenize，并变成内部 request state；第二步看 request 状态机，比如 waiting、prefill、decode、finished；第三步看 scheduler 每个 engine step 如何根据 waiting queue、running requests、token budget、KV budget 和 prefix cache 命中生成执行计划；第四步看 KV cache 如何分配、追加、释放和保留；第五步重点看 RadixAttention 如何做 longest prefix match、split、insert 和 eviction；最后看 model runner、sampler、streaming 和 structured output 如何接上。

我会用几个实验验证理解，比如构造三个共享前缀请求观察 radix tree split，构造长短 prompt 混合观察 scheduler，构造 abort 请求观察 KV 是否释放。读 mini-sglang 的目标不是背类名，而是建立 SGLang Runtime 的骨架，再用这个骨架去读完整 SGLang 源码。
```

## 34.25 源码阅读覆盖率、实验门禁和可运行 demo

为了避免“我看过源码”变成空话，可以把源码阅读笔记当成一个审计对象。设模块集合为：

```math
\mathcal{M}=\{m_1,m_2,\ldots,m_N\}
```

必读模块集合为 `M_req`，已完成笔记的模块集合为 `M_seen`，模块覆盖率是：

```math
C_{\mathrm{module}}=\frac{|M_{\mathrm{seen}}\cap M_{\mathrm{req}}|}{\max(1,|M_{\mathrm{req}}|)}
```

资源生命周期也要覆盖。设必查资源集合为 `R_req`，已在笔记中解释清楚的资源集合为 `R_seen`：

```math
C_{\mathrm{resource}}=\frac{|R_{\mathrm{seen}}\cap R_{\mathrm{req}}|}{\max(1,|R_{\mathrm{req}}|)}
```

只读不跑实验不够。设必做实验集合为 `E_req`，已完成实验集合为 `E_done`：

```math
C_{\mathrm{experiment}}=\frac{|E_{\mathrm{done}}\cap E_{\mathrm{req}}|}{\max(1,|E_{\mathrm{req}}|)}
```

源码学习门禁可以写成：

```math
G_{\mathrm{source}}=G_{\mathrm{module}}G_{\mathrm{order}}G_{\mathrm{resource}}G_{\mathrm{experiment}}G_{\mathrm{signal}}G_{\mathrm{note}}
```

这些因子分别要求模块地图完整、请求生命周期顺序可见、资源 allocate / read / write / release / cache 可复盘、实验覆盖核心路径、观察信号足够、笔记能回答模块输入输出和状态变化。

下面是一个 0 依赖源码路径审计 demo。它不执行真实 mini-sglang，而是检查你的阅读笔记是否覆盖了必读模块、资源生命周期和实验路径。

```python
from dataclasses import dataclass


@dataclass
class ModuleNote:
    name: str
    lifecycle_stage: str
    inputs: tuple
    outputs: tuple
    states: tuple
    resources: tuple


@dataclass
class ExperimentTrace:
    name: str
    modules_touched: tuple
    observed: tuple


class ToySourcePathAuditor:
    REQUIRED_MODULES = {
        "api_server",
        "tokenizer",
        "request_state",
        "scheduler",
        "kv_cache",
        "radix_cache",
        "engine",
        "attention_backend",
        "sampler",
        "detokenizer",
        "streaming",
        "cleanup",
    }
    REQUIRED_RESOURCES = {
        "token_budget",
        "sequence_budget",
        "kv_allocate",
        "kv_write",
        "kv_read",
        "kv_release",
        "prefix_cache",
    }
    REQUIRED_EXPERIMENTS = {
        "minimal_generate",
        "shared_prefix",
        "radix_split",
        "mixed_prefill_decode",
        "max_tokens_finish",
        "streaming",
        "structured_output",
        "abort_cleanup",
    }
    EXPECTED_ORDER = ["api_server", "tokenizer", "request_state", "scheduler", "kv_cache", "engine", "sampler", "detokenizer"]

    def __init__(self, modules, experiments):
        self.modules = modules
        self.experiments = experiments

    def _order_score(self):
        positions = {note.name: index for index, note in enumerate(self.modules)}
        seen = [positions[name] for name in self.EXPECTED_ORDER if name in positions]
        return seen == sorted(seen) and len(seen) == len(self.EXPECTED_ORDER)

    def run(self):
        module_names = {note.name for note in self.modules}
        resource_names = {resource for note in self.modules for resource in note.resources}
        experiment_names = {exp.name for exp in self.experiments}
        touched_modules = {module for exp in self.experiments for module in exp.modules_touched}
        observed_signals = {signal for exp in self.experiments for signal in exp.observed}
        summary = {
            "module_coverage": round(len(module_names & self.REQUIRED_MODULES) / len(self.REQUIRED_MODULES), 3),
            "resource_coverage": round(len(resource_names & self.REQUIRED_RESOURCES) / len(self.REQUIRED_RESOURCES), 3),
            "experiment_coverage": round(len(experiment_names & self.REQUIRED_EXPERIMENTS) / len(self.REQUIRED_EXPERIMENTS), 3),
            "lifecycle_order_ok": self._order_score(),
            "modules_touched_by_experiments": round(len(touched_modules & self.REQUIRED_MODULES) / len(self.REQUIRED_MODULES), 3),
            "observed_signal_count": len(observed_signals),
            "missing_modules": sorted(self.REQUIRED_MODULES - module_names),
            "missing_resources": sorted(self.REQUIRED_RESOURCES - resource_names),
            "missing_experiments": sorted(self.REQUIRED_EXPERIMENTS - experiment_names),
        }
        gates = {
            "module_map_complete": summary["module_coverage"] == 1.0,
            "lifecycle_order_visible": summary["lifecycle_order_ok"],
            "resource_lifecycle_visible": summary["resource_coverage"] == 1.0,
            "experiments_cover_core_paths": summary["experiment_coverage"] == 1.0,
            "experiments_touch_runtime": summary["modules_touched_by_experiments"] >= 0.8,
            "signals_are_observable": summary["observed_signal_count"] >= 10,
            "source_path_notes_ready": not summary["missing_modules"] and not summary["missing_resources"],
        }
        gates["source_path_gate"] = all(gates.values())
        return summary, gates


modules = [
    ModuleNote("api_server", "entry", ("json_request",), ("raw_prompt",), ("request_id",), ()),
    ModuleNote("tokenizer", "entry", ("raw_prompt",), ("input_ids",), ("prompt_len",), ()),
    ModuleNote("request_state", "runtime", ("input_ids",), ("waiting_request",), ("WAITING", "PREFILLING", "DECODING", "FINISHED"), ()),
    ModuleNote("scheduler", "runtime", ("waiting_queue", "running_set"), ("schedule_plan",), ("phase",), ("token_budget", "sequence_budget")),
    ModuleNote("kv_cache", "memory", ("schedule_plan",), ("slot_mapping",), ("kv_refs",), ("kv_allocate", "kv_write", "kv_read", "kv_release")),
    ModuleNote("radix_cache", "memory", ("input_ids",), ("prefix_hit",), ("ref_count",), ("prefix_cache",)),
    ModuleNote("engine", "compute", ("schedule_plan", "slot_mapping"), ("logits",), ("prefill", "decode"), ()),
    ModuleNote("attention_backend", "compute", ("qkv", "slot_mapping"), ("context",), ("metadata",), ()),
    ModuleNote("sampler", "decode", ("logits",), ("next_token",), ("sampling_params",), ()),
    ModuleNote("detokenizer", "output", ("next_token",), ("text_delta",), ("offset",), ()),
    ModuleNote("streaming", "output", ("text_delta",), ("event",), ("finish_reason",), ()),
    ModuleNote("cleanup", "output", ("finish_reason",), ("released",), ("FINISHED", "ABORTED"), ("kv_release",)),
]

experiments = [
    ExperimentTrace("minimal_generate", ("api_server", "tokenizer", "scheduler", "engine", "sampler", "detokenizer"), ("output_text", "finish_reason")),
    ExperimentTrace("shared_prefix", ("radix_cache", "kv_cache", "scheduler"), ("prefix_hit", "saved_prefill")),
    ExperimentTrace("radix_split", ("radix_cache",), ("node_split", "ref_count")),
    ExperimentTrace("mixed_prefill_decode", ("scheduler", "engine", "kv_cache"), ("prefill_rows", "decode_rows")),
    ExperimentTrace("max_tokens_finish", ("request_state", "cleanup"), ("FINISHED", "kv_release")),
    ExperimentTrace("streaming", ("detokenizer", "streaming", "scheduler"), ("text_delta", "stream_event")),
    ExperimentTrace("structured_output", ("sampler", "request_state"), ("grammar_mask", "valid_token")),
    ExperimentTrace("abort_cleanup", ("request_state", "cleanup", "kv_cache"), ("ABORTED", "kv_release")),
]

summary, gates = ToySourcePathAuditor(modules, experiments).run()
print("source_path_summary=", summary)
print("source_path_gates=", gates)
```

运行后可以看到类似输出：

```text
source_path_summary= {'module_coverage': 1.0, 'resource_coverage': 1.0, 'experiment_coverage': 1.0, 'lifecycle_order_ok': True, 'modules_touched_by_experiments': 0.917, 'observed_signal_count': 15, 'missing_modules': [], 'missing_resources': [], 'missing_experiments': []}
source_path_gates= {'module_map_complete': True, 'lifecycle_order_visible': True, 'resource_lifecycle_visible': True, 'experiments_cover_core_paths': True, 'experiments_touch_runtime': True, 'signals_are_observable': True, 'source_path_notes_ready': True, 'source_path_gate': True}
```

这个 demo 的用法是：你读 mini-sglang 时，可以把真实模块名、实验记录和观察到的信号填进去。如果 `resource_coverage` 或 `experiment_coverage` 不达标，说明你还没有真正看清 runtime 的资源生命周期或关键路径。

## 34.26 小练习

1. 克隆 mini-sglang，跑通一个最小 generate 请求，并记录调用栈。
2. 找到内部 Request 数据结构，列出 10 个关键字段。
3. 画出 scheduler 的输入和输出。
4. 构造三个共享前缀请求，观察 RadixAttention 树结构变化。
5. 找出 KV cache allocate、append、release 的代码位置。
6. 找出 sampler 中 temperature、top-k、top-p 或 greedy 的实现。
7. 写一页笔记：mini-sglang 省略了完整 SGLang 的哪些生产能力。

## 34.27 本章总结

mini-sglang 的价值在于帮你用较小代码建立 SGLang Runtime 的骨架。

读源码时要始终沿请求生命周期走：入口、request state、scheduler、KV cache、RadixAttention、model runner、sampler、structured output、streaming 和 cleanup。

当你能把每个源码模块映射到 TTFT、TPOT、KV cache、prefix hit、grammar mask 和 streaming 指标时，就说明你不是只看过代码，而是真正理解了 serving runtime。

下一章开始进入第五部分：PD 分离与高级 Serving 架构。我们会先讲 Prefill 和 Decode 的资源画像为什么不同，为后面的 PD 分离、KV cache 迁移、多级缓存和跨节点 serving 做铺垫。
