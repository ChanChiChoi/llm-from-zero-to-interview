# 第 13 章 实现 token streaming

前面几章已经让 engine 能排队、调度、prefill 和 decode。但到目前为止，用户通常要等请求完整生成结束后才能拿到答案。本章实现 token streaming：每生成一个 token，就尽快把增量文本返回给调用方。

Streaming 是 LLM 产品体验的核心能力。它不一定缩短总生成时间，但能显著降低用户感知等待，让用户尽早看到模型开始回答。

一句话概括：

> Token streaming 把 decode loop 的每一步输出变成增量事件，而不是等完整答案生成完再一次性返回。

## 13.1 为什么需要 streaming

非流式模式是：

```text
请求进入 -> prefill -> decode 100 个 token -> 返回完整文本
```

用户看到的是长时间空白，然后一次性出现答案。

流式模式是：

```text
请求进入 -> prefill -> token1 返回 -> token2 返回 -> token3 返回 -> ... -> 结束事件
```

用户更早看到响应。

在指标上：

1. TTFT 决定多久看到第一个片段。
2. TPOT 决定后续片段生成速度。
3. streaming 抖动决定输出是否平滑。

所以 streaming 不只是接口形式，而是直接影响产品体验。

## 13.2 streaming 在 engine 中的位置

在 generate loop 里，streaming 插在 token 生成之后：

```text
model forward -> sample token -> append token -> detokenize -> push chunk
```

在 engine 主循环中，它通常发生在 decode step 后：

```python
token_id = next_token_ids[i].item()
request.append_token(token_id)
streamer.push_token(request, token_id)
```

如果是 prefill 产生的首 token，也应该 streaming：

```python
first_token = prefill_next_token
streamer.push_token(request, first_token)
```

否则用户会错过第一个输出片段。

## 13.3 最小 Streamer 接口

先定义一个最小 streamer。

```python
class Streamer:
    def __init__(self, tokenizer_wrapper):
        self.tokenizer_wrapper = tokenizer_wrapper

    def push_token(self, request, token_id):
        text = self.tokenizer_wrapper.decode([token_id])
        print(f"{request.request_id}: {text}", end="", flush=True)

    def finish(self, request):
        print(f"\n{request.request_id} finished: {request.finish_reason}")
```

这个版本只是打印到终端，不是生产实现。

但它体现了 streaming 的最小动作：token 生成后，马上解码并输出。

## 13.4 为什么不能简单 decode 单个 token

上面的实现有问题：很多 tokenizer 的 token 到文本不是严格一对一自然字符。

例如：

1. token 可能包含前导空格。
2. 一个中文字符通常可能是一个 token，但不总是。
3. 某些 Unicode 字符可能跨 token。
4. BPE token 可能只是一个词的一部分。
5. special token 不应该直接输出。

所以生产 streaming 通常不是简单 `decode([token_id])`，而是维护一个增量解码状态。

教学版可以先接受简化，但必须知道真实系统要处理 tokenization 边界。

## 13.5 基于完整输出的增量差分

一个简单且常用的教学方法是：每次 decode 当前完整输出，然后和上一次输出做差分。

```python
class TextStreamer:
    def __init__(self, tokenizer_wrapper):
        self.tokenizer_wrapper = tokenizer_wrapper
        self.printed_text = {}

    def push_tokens(self, request):
        full_text = self.tokenizer_wrapper.decode(request.output_ids)
        old_text = self.printed_text.get(request.request_id, "")
        delta = full_text[len(old_text):]
        self.printed_text[request.request_id] = full_text

        if delta:
            self.send_delta(request, delta)

    def send_delta(self, request, delta):
        print(f"{request.request_id}: {delta}", end="", flush=True)
```

这种方法实现简单，能避免很多单 token decode 的边界问题。

缺点是每轮都 decode 完整输出，长输出时效率较差。

生产系统会使用更高效的增量 detokenizer。

## 13.6 StreamingEvent

生产中最好不要直接传字符串，而是定义事件。

```python
class StreamingEvent:
    def __init__(self, request_id, delta="", finish_reason=None, error=None):
        self.request_id = request_id
        self.delta = delta
        self.finish_reason = finish_reason
        self.error = error
```

事件类型可以包括：

1. token delta。
2. finish event。
3. error event。
4. heartbeat。
5. usage event。

这样 HTTP API、WebSocket、SSE、测试代码都可以消费同一种事件。

## 13.7 每个请求一个输出队列

为了把模型执行和网络输出解耦，可以给每个请求一个 queue。

```python
from queue import Queue


class RequestState:
    def __init__(self, request_id, prompt, max_new_tokens=128):
        self.request_id = request_id
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.output_ids = []
        self.stream_queue = Queue()
        self.finished = False
        self.finish_reason = None
```

Streamer 把事件放入队列：

```python
def send_delta(self, request, delta):
    request.stream_queue.put(StreamingEvent(request.request_id, delta=delta))
```

请求完成时：

```python
def finish(self, request):
    request.stream_queue.put(
        StreamingEvent(request.request_id, finish_reason=request.finish_reason)
    )
```

这样 engine 负责生成，API 层负责从 queue 里取事件并写给客户端。

## 13.8 在 prefill 中推送首 token

第 12 章的 `run_prefill` 可以改成：

```python
def run_prefill(self, requests):
    prompts = [request.prompt for request in requests]
    result = batched_prefill(
        self.tokenizer_wrapper,
        self.model_wrapper,
        self.sampler,
        prompts,
    )

    next_token_ids = result["next_token_ids"]
    past_key_values = result["past_key_values"]

    for i, request in enumerate(requests):
        token_id = next_token_ids[i].item()
        request.append_token(token_id)
        request.status = "DECODING"
        request.first_token_time = now()
        self.streamer.push_tokens(request)

    self.decode_groups.append(DecodeGroup(requests, past_key_values))
    self.running_requests.extend(requests)
```

首 token 推送出去，TTFT 才真正体现在用户侧。

## 13.9 在 decode 中推送增量

decode step 后：

```python
def run_decode_group(self, group):
    last_token_ids = torch.tensor(
        [[r.next_token_id] for r in group.requests],
        device=self.model_wrapper.device,
    )

    next_token_ids, group.past_key_values = batched_decode_step(
        self.model_wrapper,
        self.sampler,
        last_token_ids,
        group.past_key_values,
    )

    for i, request in enumerate(group.requests):
        if request.finished:
            continue

        token_id = next_token_ids[i].item()
        request.append_token(token_id)
        self.streamer.push_tokens(request)

        if request.should_stop(self.tokenizer_wrapper.eos_token_id):
            self.streamer.finish(request)
```

这里顺序很重要：

1. append token。
2. push delta。
3. 判断是否结束。
4. 推送 finish event。

如果先结束再 push，可能漏掉最后一个 token。

## 13.10 finish event

流式响应必须有结束事件。

否则客户端不知道是正常结束、网络断开，还是服务端异常。

finish event 至少应该包含：

1. request id。
2. finish reason。
3. usage 统计，比如 input tokens 和 output tokens。
4. 可选错误信息。

示例：

```json
{
  "request_id": "r1",
  "delta": "",
  "finish_reason": "stop",
  "usage": {
    "output_tokens": 128
  }
}
```

没有 finish event 的 streaming 是不完整的。

## 13.11 stop sequence 和 streaming

stop sequence 会让 streaming 变复杂。

假设 stop sequence 是：

```text
"</answer>"
```

它可能跨多个 token 出现。

如果每生成一个 token 就立刻输出，可能会把 stop sequence 的一部分发给客户端。

解决办法通常是维护一个小缓冲区：

1. 新 token 先进入 buffer。
2. 检查 buffer 是否可能包含 stop sequence。
3. 确认安全的部分再输出。
4. 如果命中 stop，截断 stop sequence 并结束。

教学版可以暂时只处理 EOS，但面试和生产设计必须知道 stop sequence 的跨 token 问题。

## 13.12 客户端取消

Streaming 场景中，客户端可能中途断开。

断开后 engine 必须：

1. 标记 request 为 aborted。
2. 停止继续生成。
3. 从 running set 中移除。
4. 释放 KV Cache。
5. 发送或记录 finish reason。
6. 记录指标。

如果只关闭网络连接，但 engine 还在 decode，这个请求就会继续消耗 GPU。

教学版可以给 request 加一个字段：

```python
request.aborted = True
```

在 scheduler 或 decode loop 中跳过它。

## 13.13 backpressure

如果客户端读取很慢，stream queue 可能积压。

这叫 backpressure。

简单做法是限制队列长度：

```python
request.stream_queue = Queue(maxsize=128)
```

如果队列满了，可以选择：

1. 阻塞 engine，等待客户端读取。
2. 中止请求。
3. 丢弃部分中间事件。
4. 把网络输出放到独立线程或异步任务。

生产系统通常不希望网络慢客户端阻塞模型执行主循环。因此 engine 和 API 输出层要解耦。

## 13.14 SSE 的直觉

HTTP streaming 常用 SSE，也就是 Server-Sent Events。

SSE 返回类似：

```text
data: {"delta":"Hello"}

data: {"delta":" world"}

data: {"finish_reason":"stop"}

```

API 层从 request 的 stream queue 里不断取事件，然后写给 HTTP response。

本章不实现完整 HTTP，下一章会讲最小 API 服务。本章只要先把 engine 内部的 token event 做出来。

## 13.15 Streaming 和指标

streaming 会带来更细的指标：

1. first token time。
2. 每个 token 的时间戳。
3. token 间隔分布。
4. stream queue 长度。
5. 客户端取消率。
6. finish reason 分布。
7. backpressure 次数。

这些指标能帮助排查：

1. 为什么首 token 慢。
2. 为什么输出一顿一顿。
3. 是模型 decode 慢，还是网络输出慢。
4. 哪些请求经常被用户中途取消。

## 13.16 最小代码骨架

核心组件可以整理成：

```python
class StreamingEvent:
    def __init__(self, request_id, delta="", finish_reason=None, error=None):
        self.request_id = request_id
        self.delta = delta
        self.finish_reason = finish_reason
        self.error = error


class TextStreamer:
    def __init__(self, tokenizer_wrapper):
        self.tokenizer_wrapper = tokenizer_wrapper
        self.printed_text = {}

    def push_tokens(self, request):
        full_text = self.tokenizer_wrapper.decode(request.output_ids)
        old_text = self.printed_text.get(request.request_id, "")
        delta = full_text[len(old_text):]
        self.printed_text[request.request_id] = full_text
        if delta:
            request.stream_queue.put(
                StreamingEvent(request.request_id, delta=delta)
            )

    def finish(self, request):
        request.stream_queue.put(
            StreamingEvent(
                request.request_id,
                finish_reason=request.finish_reason,
            )
        )
```

engine 只需要在 prefill 和 decode 生成 token 后调用 `push_tokens`。

## 13.17 常见误区

误区一：streaming 就是每个 token `print` 一下。

真实 streaming 要处理 detokenization、事件格式、结束事件、取消、backpressure 和错误。

误区二：只要首 token 生成了，用户就一定能看到。

还要看输出通道是否及时 flush，API 层是否阻塞，客户端是否正常读取。

误区三：finish event 可有可无。

没有结束事件，客户端无法区分正常结束和异常断开。

误区四：客户端断开只影响网络层。

断开必须反馈给 engine，否则 GPU 可能继续为已取消请求生成。

误区五：单 token decode 一定等于单字符输出。

token 和自然字符不一定一一对应，增量解码要处理边界问题。

## 13.18 面试追问

1. Token streaming 在 decode loop 的哪个位置发生？
2. 为什么 streaming 能改善用户体验，即使总耗时不变？
3. 为什么不能简单地 `decode([token_id])` 就输出？
4. finish event 应该包含哪些信息？
5. stop sequence 跨 token 时怎么处理？
6. 客户端取消后 engine 应该做什么？
7. backpressure 是什么，为什么会影响 serving engine？
8. 如何判断输出卡顿是 decode 慢还是网络 streaming 慢？

参考回答思路：

1. 先说 streaming 在 token 生成后、请求结束前，把增量文本推给客户端。
2. 再说它降低感知等待，TTFT 和 token 间隔决定体验。
3. 然后补工程细节：增量 detokenization、finish event、stop sequence、取消、backpressure。
4. 最后讲指标：first token time、token 间隔、stream queue、取消率。

## 13.19 小练习

1. 在第 12 章 MiniEngine 中加入 `TextStreamer`。
2. 每次 prefill 产生首 token 后，向 stream queue 推送 delta。
3. 每次 decode 产生新 token 后，向 stream queue 推送 delta。
4. 请求结束时推送 finish event。
5. 模拟客户端不读取 queue，观察 backpressure 可能带来的问题。

## 13.20 本章小结

本章实现了 token streaming。

Streaming 把 decode loop 的每一步输出变成增量事件。它需要处理增量 detokenization、事件队列、finish event、客户端取消、backpressure 和指标。对用户来说，streaming 让模型尽早开始回答；对 engine 来说，它让请求生命周期多了输出通道和取消反馈。

下一章我们会把当前 MiniEngine 包成一个最小 HTTP API 服务，让外部客户端能够通过接口提交请求并接收流式输出。
