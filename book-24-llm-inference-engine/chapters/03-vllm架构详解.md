# 第 3 章 推理请求的生命周期

前两章分别讲了 serving engine 的整体定位，以及从 `model.generate()` 演进到推理框架需要补哪些系统能力。本章继续把视角拉到一个具体请求：它从进入 API Server 到最后释放资源，中间会经历哪些阶段？

理解请求生命周期，是理解 vLLM、SGLang、TGI 这类系统的基础。因为 scheduler、KV Cache Manager、streaming、metrics、timeout、abort、finish reason，本质上都是围绕请求生命周期展开的。

一句话概括：

> 推理请求不是一次函数调用，而是一个带状态、带资源、带输出通道、可被调度和取消的生命周期对象。

## 3.1 为什么要讲生命周期

很多初学者看 serving engine 时，会先记模块名：API Server、Engine、Scheduler、Worker、Executor、Block Manager。这样容易只记住结构，没理解数据怎么流动。

更好的方法是从一个请求出发，追踪它的完整路径：

```text
客户端请求
  -> API Server
  -> 参数校验和 tokenization
  -> request object
  -> waiting queue
  -> scheduler
  -> prefill
  -> decode loop
  -> streaming output
  -> finish / abort / fail
  -> resource cleanup
  -> metrics / trace
```

只要这条链路清楚，后面看 continuous batching、PagedAttention、PD 分离和 prefix cache 都会更容易。

## 3.2 入口：客户端请求

一个典型聊天补全请求可能包含：

```json
{
  "model": "example-llm",
  "messages": [
    {"role": "user", "content": "解释一下 KV Cache"}
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": true
}
```

从业务角度看，这是一次普通 API 调用。从 engine 角度看，它马上会变成一个需要长期维护状态的对象。

入口层通常要处理：

1. 协议解析，比如 HTTP、gRPC、SSE 或 WebSocket。
2. 鉴权和租户信息。
3. 模型名映射。
4. 参数合法性检查。
5. 请求大小限制。
6. 超时时间和优先级。
7. 是否流式输出。

入口层不应该直接调用模型。它更像网关和请求构造器，负责把外部请求转成 engine 内部统一格式。

## 3.3 参数规范化

不同客户端会传不同参数。engine 通常需要把它们规范化成内部配置。

常见参数包括：

1. `max_tokens`：最多生成多少 token。
2. `temperature`：采样随机性。
3. `top_p` / `top_k`：采样候选集合。
4. `stop`：停止词或停止序列。
5. `stream`：是否增量返回。
6. `seed`：是否需要可复现采样。
7. `presence_penalty` / `frequency_penalty`：重复惩罚。
8. `logprobs`：是否返回 token 概率信息。

规范化要解决三个问题：

1. 默认值是什么。
2. 参数之间是否冲突。
3. 参数是否超出系统能力。

例如，用户传了 `max_tokens=100000`，但服务只允许最多生成 4096 个 token，就必须在入口阶段拒绝或截断。否则请求进入 scheduler 后才发现无法执行，会浪费队列和显存资源。

## 3.4 Prompt 构造和 tokenization

聊天请求通常不是直接拿用户文本送进模型，而要先套聊天模板。

例如 messages 会被转换成类似：

```text
<system>你是一个有帮助的助手</system>
<user>解释一下 KV Cache</user>
<assistant>
```

然后 tokenizer 把文本转成 token ids。

这一阶段要注意：

1. 不同模型的 chat template 不同。
2. tokenizer 版本必须和模型权重匹配。
3. token 数决定 prefill 成本。
4. prompt 太长可能超过上下文窗口。
5. stop words 需要和 token 边界协调。

tokenization 不是纯文本处理小事。生产系统里，token 数会影响路由、限流、计费、调度和显存预估。

## 3.5 创建 Request 对象

进入 engine 后，外部请求通常会被封装成内部 request object。

一个简化结构如下：

```python
class Request:
    def __init__(self, request_id, input_ids, sampling_params):
        self.request_id = request_id
        self.input_ids = input_ids
        self.output_ids = []
        self.sampling_params = sampling_params
        self.status = "WAITING"
        self.arrival_time = now()
        self.first_token_time = None
        self.finished_time = None
        self.kv_blocks = []
        self.stream = None
        self.error = None
```

真实系统里的字段更多，但核心信息差不多：

1. 请求身份：`request_id`、tenant、model。
2. 输入输出：`input_ids`、`output_ids`。
3. 采样配置：temperature、top-p、stop 等。
4. 生命周期状态：waiting、prefilling、decoding、finished。
5. 资源引用：KV Cache block、worker、device。
6. 输出通道：stream handle、回调或队列。
7. 指标时间戳：到达、首 token、完成。

注意，request object 不只是数据包。它是 scheduler 和 executor 共同操作的状态对象。

## 3.6 进入 waiting queue

请求创建后，通常不会马上执行，而是进入 waiting queue。

waiting queue 的作用是把“请求到达”和“模型执行”解耦。

如果没有队列，每来一个请求就直接调用模型，会导致：

1. 无法组成 batch。
2. 无法做公平调度。
3. 无法控制显存。
4. 无法统一处理优先级。
5. 无法在高峰期保护系统。

队列里常见的排序策略包括：

1. FIFO：先来先服务，简单但不一定最优。
2. Shortest prompt first：短 prompt 优先，降低平均 TTFT，但可能饿死长请求。
3. Priority-based：高优先级租户或实时业务优先。
4. Deadline-aware：接近超时的请求优先。
5. Token-budget-based：按本轮 token 预算选择请求。

生产系统往往不会只用单一策略，而是结合公平性、SLO 和资源预算。

## 3.7 Scheduler 决定谁上 GPU

Scheduler 是生命周期里的核心决策点。它每一轮会决定哪些请求进入模型执行。

一个简化调度输入包括：

1. waiting queue 中的新请求。
2. running set 中正在 decode 的请求。
3. 当前可用 KV Cache 块数。
4. 最大 batch token 数。
5. 最大并发序列数。
6. 每个请求的状态、长度和优先级。

调度输出通常是一个 batch：

```text
本轮 prefill: request_7, request_8
本轮 decode: request_1, request_2, request_5
```

Scheduler 要处理的 trade-off 很多：

1. 新请求 prefill 多一些，TTFT 可能更好，但会挤占 decode。
2. decode 请求多一些，已有用户输出更平滑，但新请求等待更久。
3. 长 prompt 直接 prefill，可能造成大抖动。
4. 短请求优先，平均延迟好，但公平性可能差。
5. batch 过大，吞吐高但单请求延迟可能变差。

这就是推理框架比 `generate()` 难的地方：它不是做一次最优，而是在每轮 iteration 做动态权衡。

## 3.8 Prefill 阶段

当请求第一次被调度执行时，会进入 prefill 阶段。

Prefill 做的是：

1. 读取 prompt token。
2. 执行模型前向。
3. 为每层生成并保存 KV Cache。
4. 得到最后一个位置的 logits。
5. 采样第一个输出 token。

简化流程如下：

```text
input_ids -> model forward -> KV Cache -> logits -> sample first token
```

Prefill 的结束时间直接决定 TTFT，也就是用户看到第一个 token 的等待时间。

长 prompt 的 prefill 可能非常重。为了避免它一次占用过多计算资源，一些系统会使用 chunked prefill，把长 prompt 分块处理，让 decode 请求有机会穿插执行。

## 3.9 Decode 阶段

prefill 完成后，请求进入 decode 阶段。decode 每轮通常只为每个请求生成一个 token。

简化流程如下：

```text
last_token + KV Cache -> model forward -> logits -> sample next token -> append
```

Decode 阶段会重复很多轮，直到满足停止条件。

停止条件可能包括：

1. 生成了 EOS token。
2. 达到 `max_tokens`。
3. 命中 stop words。
4. 客户端取消。
5. 请求超时。
6. engine 主动中止。
7. 执行错误。

Decode 的关键指标是 TPOT，也就是每个输出 token 的平均耗时。用户体验里的“输出是否顺滑”，主要由 decode 阶段决定。

## 3.10 Streaming 输出

如果请求启用了 streaming，engine 在每次生成 token 后都要尽快把增量文本推给客户端。

流程大致是：

```text
token id -> detokenize -> text chunk -> stream channel -> client
```

这里有几个细节：

1. 一个 token 不一定对应完整字符。
2. 多个 token 合起来才可能形成自然文本片段。
3. stop sequence 可能跨 token 出现。
4. 客户端断开时，engine 必须释放资源。
5. 流式通道不能阻塞模型执行主循环太久。

因此很多系统会把模型执行和网络输出解耦，用异步队列或回调把 token 传给输出层。

Streaming 的难点不是“能不能返回 token”，而是“返回 token 的同时不破坏调度、资源释放和错误处理”。

## 3.11 Finish Reason

请求结束时，系统通常需要返回 finish reason。

常见类型包括：

1. `stop`：正常遇到停止符或 stop sequence。
2. `length`：达到最大生成长度。
3. `abort`：用户取消。
4. `timeout`：请求超时。
5. `error`：系统错误。
6. `content_filter`：输出被策略拦截。

finish reason 很重要，因为它告诉上游业务这次输出是否完整。

例如，`length` 说明模型可能还没说完，只是被最大长度截断。`abort` 说明不是模型结束，而是连接或用户行为导致中止。`error` 需要上游决定是否重试。

## 3.12 资源释放

请求结束后，engine 必须释放资源，尤其是 KV Cache。

需要清理的资源包括：

1. GPU KV Cache block。
2. CPU staging buffer。
3. 输出 stream。
4. request object。
5. tracing span。
6. 临时 token buffer。
7. 队列和索引中的引用。

资源释放如果不彻底，会造成隐蔽问题：

1. 显存慢慢上涨。
2. 可用 block 越来越少。
3. scheduler 误以为资源不足。
4. 已取消请求继续占用 decode slot。
5. 长时间运行后吞吐下降。

推理服务是长生命周期进程，轻微泄漏也会在高 QPS 下快速放大。

## 3.13 Metrics 和 Trace

每个请求都应该留下可观测信息。

常见指标包括：

1. request arrival time。
2. queue waiting time。
3. prefill latency。
4. TTFT。
5. decode token count。
6. TPOT。
7. total latency。
8. input tokens。
9. output tokens。
10. finish reason。
11. error type。
12. KV Cache block 使用量。

trace 则用于回答更细的问题：

1. 请求在哪个阶段最慢。
2. 是排队慢、prefill 慢，还是 decode 慢。
3. 是否被抢占或等待 KV Cache。
4. 是否发生过 stream 阻塞。
5. 哪个 worker 执行了请求。

没有这些数据，线上问题只能靠猜。面试里如果问“TTFT 变差怎么排查”，生命周期指标就是排查入口。

## 3.14 Abort 和 Timeout

生产系统必须认真处理取消和超时。

用户取消请求时，engine 不能只关闭连接，还要：

1. 标记 request 状态为 aborted。
2. 从 waiting queue 或 running set 移除。
3. 释放 KV Cache。
4. 停止继续 streaming。
5. 记录 finish reason。
6. 避免下一轮 scheduler 再选中它。

超时也类似。区别是超时可能来自系统策略，而不是用户主动取消。

常见超时包括：

1. 排队超时。
2. 首 token 超时。
3. 总请求超时。
4. stream 空闲超时。

很多 serving bug 都来自取消处理不完整：请求已经断开，但 decode loop 还在生成；或者请求被移出队列，但 KV Cache 没释放。

## 3.15 生命周期状态机

可以用下面的状态机总结本章：

```text
RECEIVED
  -> VALIDATED
  -> TOKENIZED
  -> WAITING
  -> PREFILLING
  -> DECODING
  -> FINISHED

任意中间状态 -> ABORTED
任意中间状态 -> FAILED
WAITING/PREFILLING/DECODING -> TIMEOUT
```

每个状态都对应明确动作：

1. RECEIVED：接收到外部请求。
2. VALIDATED：参数校验完成。
3. TOKENIZED：prompt 已转 token。
4. WAITING：进入等待队列。
5. PREFILLING：执行 prompt 前向并建立 KV Cache。
6. DECODING：逐 token 生成。
7. FINISHED：正常结束。
8. ABORTED：用户取消。
9. FAILED：系统错误。
10. TIMEOUT：超时中止。

状态机的价值在于让复杂情况有规则可循。无论正常完成、失败、取消还是超时，都能走到统一清理路径。

## 3.16 面试追问

1. 一个 LLM 请求从进入 API Server 到返回结果，中间有哪些阶段？
2. 为什么请求进入 engine 后不能直接执行，而要进入 waiting queue？
3. Scheduler 在请求生命周期里做什么决策？
4. TTFT 对应生命周期中的哪个阶段？
5. TPOT 对应生命周期中的哪个阶段？
6. 客户端中途断开连接，engine 应该如何处理？
7. finish reason 有哪些类型，为什么重要？
8. 线上 TTFT 突然升高，你会从哪些生命周期指标排查？

参考回答思路：

1. 先按链路说：接入、校验、tokenization、排队、调度、prefill、decode、stream、结束、清理、指标。
2. 再强调请求是有状态对象，不是一次函数调用。
3. 然后说明 prefill 影响 TTFT，decode 影响 TPOT。
4. 最后补充异常路径：abort、timeout、fail 都必须释放 KV Cache 并记录 finish reason。

## 3.17 小练习

1. 画一张请求生命周期状态机，至少包含 waiting、prefilling、decoding、finished、aborted、failed。
2. 设计一个 `Request` 类，写出你认为最重要的 10 个字段。
3. 如果请求在 waiting queue 中超时，应该释放哪些资源？
4. 如果请求已经进入 decode 阶段后客户端断开，engine 应该如何停止它？
5. 设计一组指标，用来判断 TTFT 变差到底是排队导致还是 prefill 导致。

## 3.18 本章小结

本章从一个请求出发，完整走了一遍 LLM 推理请求的生命周期。

推理请求会经历接入、参数规范化、prompt 构造、tokenization、创建 request object、进入 waiting queue、scheduler 调度、prefill、decode、streaming、结束、资源释放和指标记录。正常结束只是其中一种路径，取消、超时和失败同样必须被纳入状态机。

下一章我们会进一步深入生命周期中最关键的四个概念：Prefill、Decode、KV Cache 和 Token Streaming。它们决定了大模型推理的性能边界，也是后续理解 vLLM、SGLang 和 PD 分离的基础。
