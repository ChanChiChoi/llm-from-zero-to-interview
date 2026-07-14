# 第 13 章 实现 token streaming

前面几章已经让 engine 能排队、调度、prefill 和 decode。但到目前为止，用户通常要等请求完整生成结束后才能拿到答案。本章实现 token streaming：每生成一个 token，就尽快把增量文本返回给调用方。

Streaming 是 LLM 产品体验的核心能力。它不一定缩短总生成时间，但能显著降低用户感知等待，让用户尽早看到模型开始回答。

一句话概括：

> Token streaming 把 decode loop 的每一步输出变成增量事件，而不是等完整答案生成完再一次性返回。

## 13.0 本讲资料边界与第二轮精修口径

本讲只实现 engine 内部的教学版 token streaming。它会覆盖增量 detokenize、streaming event、finish event、stop sequence 缓冲、客户端取消、慢客户端 backpressure 和最小可运行 demo，但不实现完整 HTTP server、鉴权、SSE 长连接、WebSocket、OpenAI-compatible API、异步 worker、跨进程队列或生产级 Unicode / BPE 增量解码器。

资料校准口径：

1. Hugging Face Transformers 提供 `TextStreamer`、`TextIteratorStreamer` 和 `AsyncTextIteratorStreamer`。公开文档强调 streamer 会在 token 形成可输出文本时输出，并可把可打印文本放进队列供下游应用消费。
2. OpenAI API streaming 文档强调 streaming 可以在完整响应生成完之前开始处理输出，Responses API 使用 typed semantic events，Chat Completions streaming 中增量内容放在 `delta` 字段中。
3. WHATWG HTML 标准的 Server-Sent Events 定义了 `text/event-stream`、`EventSource`、`data:` 行和事件分隔等线格式；本章只借用 SSE 直觉，不实现 HTTP 传输层。
4. 本章 demo 用纯 Python list 模拟 stream queue，用 toy tokenizer 模拟 BPE 边界和 stop sequence 跨 token，不绑定任何具体 serving runtime。

参考资料：

1. Hugging Face Transformers generation streamers：<https://huggingface.co/docs/transformers/main/en/internal/generation_utils#streamers>
2. OpenAI API streaming responses：<https://platform.openai.com/docs/guides/streaming-responses>
3. WHATWG HTML Server-sent events：<https://html.spec.whatwg.org/multipage/server-sent-events.html>

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

## 13.2 Streaming 指标公式

假设请求 `i` 的到达时间是 `a_i`，第一个可见 chunk 发出的时间是 `s_{i,1}`，第 `j` 个 chunk 发出的时间是 `s_{i,j}`，一共发出 `M_i` 个文本 chunk。

用户侧 TTFT：

```math
T_{\mathrm{ttft},i}=s_{i,1}-a_i
```

相邻 chunk 间隔：

```math
\Delta_{i,j}=s_{i,j}-s_{i,j-1},\quad j=2,\ldots,M_i
```

平均 token / chunk 间隔：

```math
T_{\mathrm{tpot},i}=\frac{1}{M_i-1}\sum_{j=2}^{M_i}\Delta_{i,j}
```

stream queue backlog：

```math
Q_{i,t}=E_{i,t}^{\mathrm{produced}}-E_{i,t}^{\mathrm{consumed}}
```

backpressure 门禁：

```math
G_{\mathrm{bp}}=\mathbf{1}[Q_{i,t}\le Q_{\max}]
```

这几个量把 streaming 拆成三类问题：

1. 首 token 是否尽早可见。
2. 后续 token 是否平滑。
3. 网络或客户端是否慢到反压 engine。

## 13.3 streaming 在 engine 中的位置

在 generate loop 里，streaming 插在 token 生成之后：

```text
model forward -> sample token -> append token -> detokenize -> push event
```

在 engine 主循环中，它通常发生在 decode step 后：

```python
token_id = next_token_ids[i].item()
request.append_token(token_id)
streamer.push_tokens(request)
```

如果是 prefill 产生的首 token，也应该 streaming：

```python
first_token = prefill_next_token
request.append_token(first_token)
streamer.push_tokens(request)
```

否则用户会错过第一个输出片段，TTFT 也无法真实反映到客户端。

## 13.4 最小 Streamer 接口

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

## 13.5 为什么不能简单 decode 单个 token

上面的实现有问题：很多 tokenizer 的 token 到文本不是严格一对一自然字符。

例如：

1. token 可能包含前导空格。
2. 一个中文字符通常可能是一个 token，但不总是。
3. 某些 Unicode 字符可能跨 token。
4. BPE token 可能只是一个词的一部分。
5. special token 不应该直接输出。

所以生产 streaming 通常不是简单 `decode([token_id])`，而是维护一个增量解码状态。

教学版可以先接受简化，但必须知道真实系统要处理 tokenization 边界。

## 13.6 基于完整输出的增量差分

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
        request.stream_queue.put(StreamingEvent(request.request_id, "delta", delta=delta))
```

这种方法实现简单，能避免很多单 token decode 的边界问题。

缺点是每轮都 decode 完整输出，长输出时效率较差。生产系统会使用更高效的增量 detokenizer。

## 13.7 StreamingEvent

生产中最好不要直接传字符串，而是定义事件。

```python
class StreamingEvent:
    def __init__(self, request_id, event_type, delta="", finish_reason=None, error=None):
        self.request_id = request_id
        self.event_type = event_type
        self.delta = delta
        self.finish_reason = finish_reason
        self.error = error
```

事件类型可以包括：

1. `delta`：增量文本。
2. `finish`：正常结束、长度结束、取消或错误结束。
3. `error`：执行异常。
4. `heartbeat`：长连接保活。
5. `usage`：token 统计或计费信息。

这样 HTTP API、WebSocket、SSE、测试代码都可以消费同一种事件。

## 13.8 每个请求一个输出队列

为了把模型执行和网络输出解耦，可以给每个请求一个 queue。

```python
from queue import Queue


class RequestState:
    def __init__(self, request_id, prompt, max_new_tokens=128):
        self.request_id = request_id
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.output_ids = []
        self.stream_queue = Queue(maxsize=128)
        self.finished = False
        self.finish_reason = None
```

Streamer 把事件放入队列：

```python
def send_delta(self, request, delta):
    request.stream_queue.put(StreamingEvent(request.request_id, "delta", delta=delta))
```

请求完成时：

```python
def finish(self, request):
    request.stream_queue.put(
        StreamingEvent(request.request_id, "finish", finish_reason=request.finish_reason)
    )
```

这样 engine 负责生成，API 层负责从 queue 里取事件并写给客户端。

## 13.9 在 prefill 中推送首 token

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

## 13.10 在 decode 中推送增量

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

## 13.11 finish event

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
  "event_type": "finish",
  "delta": "",
  "finish_reason": "eos",
  "usage": {
    "input_tokens": 12,
    "output_tokens": 128
  }
}
```

没有 finish event 的 streaming 是不完整的。

## 13.12 stop sequence 和 streaming

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

## 13.13 客户端取消

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
request.finish_reason = "client_cancelled"
```

在 scheduler 或 decode loop 中跳过它。

## 13.14 backpressure

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

## 13.15 SSE 的直觉

HTTP streaming 常用 SSE，也就是 Server-Sent Events。

SSE 返回类似：

```text
event: delta
data: {"delta":"Hello"}

event: delta
data: {"delta":" world"}

event: finish
data: {"finish_reason":"eos"}

```

API 层从 request 的 stream queue 里不断取事件，然后写给 HTTP response。

本章不实现完整 HTTP，下一章会讲最小 API 服务。本章只要先把 engine 内部的 token event 做出来。

## 13.16 Streaming 和指标

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

## 13.17 Token streaming 公式、事件和可运行 demo

下面的 demo 不依赖外部库。它模拟 4 个请求：

1. `normal`：正常输出 `Hello world!`，最后 EOS。
2. `stop`：目标文本里出现跨 token 的 `END` stop sequence，streaming 只输出 `OK`，不泄露 stop sequence。
3. `cancel`：客户端在一个 chunk 后取消，engine 发送 `client_cancelled` finish event。
4. `slow`：stream queue 太小，第二个 delta 触发 backpressure。

```python
from dataclasses import dataclass, field


@dataclass
class StreamingEvent:
    request_id: str
    event_type: str
    delta: str = ""
    finish_reason: str | None = None
    usage: dict | None = None
    error: str | None = None
    t: int | None = None


@dataclass
class RequestState:
    request_id: str
    input_tokens: int
    planned_token_ids: list[int]
    max_stream_events: int = 8
    stop_sequence: str | None = None
    cancel_at_step: int | None = None
    output_ids: list[int] = field(default_factory=list)
    stream_queue: list[StreamingEvent] = field(default_factory=list)
    status: str = "DECODING"
    finish_reason: str | None = None
    first_token_time: int | None = None
    token_times: list[int] = field(default_factory=list)
    backpressure_events: int = 0

    def put_delta_event(self, event):
        if len(self.stream_queue) >= self.max_stream_events:
            self.backpressure_events += 1
            self.status = "FINISHED"
            self.finish_reason = "backpressure"
            return False
        self.stream_queue.append(event)
        return True

    @property
    def finished(self):
        return self.status == "FINISHED"


class ToyTokenizer:
    def __init__(self):
        self.id_to_text = {
            0: "<eos>",
            1: "Hel",
            2: "lo",
            3: " ",
            4: "world",
            5: "!",
            6: "O",
            7: "K",
            8: "E",
            9: "ND",
            10: "slow",
            11: " client",
            12: " keeps",
        }
        self.eos_token_id = 0

    def decode(self, token_ids):
        return "".join(
            self.id_to_text[token_id]
            for token_id in token_ids
            if token_id != self.eos_token_id
        )


class TextStreamer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.printed_text = {}
        self.trace = []

    def push_tokens(self, request, now):
        full_text = self.tokenizer.decode(request.output_ids)
        safe_full_text = full_text
        finish_reason = None
        stop = request.stop_sequence

        if stop:
            stop_at = full_text.find(stop)
            if stop_at >= 0:
                safe_full_text = full_text[:stop_at]
                finish_reason = "stop_sequence"
            else:
                holdback = max(0, len(stop) - 1)
                if len(full_text) > holdback:
                    safe_full_text = full_text[:-holdback]
                else:
                    safe_full_text = ""

        old_text = self.printed_text.get(request.request_id, "")
        delta = safe_full_text[len(old_text):]
        if delta:
            ok = request.put_delta_event(
                StreamingEvent(request.request_id, "delta", delta=delta, t=now)
            )
            if not ok:
                self.finish(request, now, error="stream_queue_full")
                self.trace.append((now, request.request_id, "backpressure", len(request.stream_queue)))
                return False
            self.printed_text[request.request_id] = safe_full_text
            self.trace.append((now, request.request_id, "delta", delta))

        if finish_reason:
            request.status = "FINISHED"
            request.finish_reason = finish_reason
            self.finish(request, now)
        return True

    def finish(self, request, now, error=None):
        usage = {
            "input_tokens": request.input_tokens,
            "output_tokens": len(request.output_ids),
        }
        request.stream_queue.append(
            StreamingEvent(
                request.request_id,
                "finish",
                finish_reason=request.finish_reason,
                usage=usage,
                error=error,
                t=now,
            )
        )
        self.trace.append((now, request.request_id, "finish", request.finish_reason))


def run_streaming_demo():
    tokenizer = ToyTokenizer()
    streamer = TextStreamer(tokenizer)
    requests = [
        RequestState("normal", 3, [1, 2, 3, 4, 5, 0]),
        RequestState("stop", 2, [6, 7, 8, 9, 5], stop_sequence="END"),
        RequestState("cancel", 1, [10, 11, 12, 0], cancel_at_step=1),
        RequestState("slow", 1, [10, 11, 12, 0], max_stream_events=1),
    ]

    now = 0
    active = list(requests)
    while active:
        next_active = []
        for request in active:
            if request.cancel_at_step is not None and len(request.output_ids) == request.cancel_at_step:
                request.status = "FINISHED"
                request.finish_reason = "client_cancelled"
                streamer.finish(request, now)
                continue

            token_id = request.planned_token_ids[len(request.output_ids)]
            request.output_ids.append(token_id)
            if request.first_token_time is None:
                request.first_token_time = now
            request.token_times.append(now)

            if token_id == tokenizer.eos_token_id:
                request.status = "FINISHED"
                request.finish_reason = "eos"
                streamer.push_tokens(request, now)
                streamer.finish(request, now)
                continue

            keep_running = streamer.push_tokens(request, now)
            if keep_running and not request.finished and len(request.output_ids) < len(request.planned_token_ids):
                next_active.append(request)
        active = next_active
        now += 1

    summary = {}
    for request in requests:
        deltas = [event.delta for event in request.stream_queue if event.event_type == "delta"]
        finish_events = [event for event in request.stream_queue if event.event_type == "finish"]
        intervals = [b - a for a, b in zip(request.token_times, request.token_times[1:])]
        summary[request.request_id] = {
            "deltas": deltas,
            "finish_reason": request.finish_reason,
            "finish_events": len(finish_events),
            "ttft_steps": request.first_token_time + 1 if request.first_token_time is not None else None,
            "token_intervals": intervals,
            "queue_len": len(request.stream_queue),
            "backpressure_events": request.backpressure_events,
        }

    stop_text = "".join(summary["stop"]["deltas"])
    gates = {
        "first_token_streamed": summary["normal"]["deltas"][0] == "Hel",
        "full_text_reconstructed": "".join(summary["normal"]["deltas"]) == "Hello world!",
        "finish_event_present": all(value["finish_events"] == 1 for value in summary.values()),
        "stop_sequence_hidden": "END" not in stop_text and stop_text == "OK",
        "client_cancelled": summary["cancel"]["finish_reason"] == "client_cancelled",
        "backpressure_detected": summary["slow"]["backpressure_events"] == 1,
    }
    gates["streaming_gate"] = all(gates.values())
    return summary, streamer.trace, gates


summary, trace, gates = run_streaming_demo()
print("stream_summary=", summary)
print("trace_tail=", trace[-8:])
print("streaming_gates=", gates)
```

一组稳定输出：

```text
stream_summary= {'normal': {'deltas': ['Hel', 'lo', ' ', 'world', '!'], 'finish_reason': 'eos', 'finish_events': 1, 'ttft_steps': 1, 'token_intervals': [1, 1, 1, 1, 1], 'queue_len': 6, 'backpressure_events': 0}, 'stop': {'deltas': ['O', 'K'], 'finish_reason': 'stop_sequence', 'finish_events': 1, 'ttft_steps': 1, 'token_intervals': [1, 1, 1], 'queue_len': 3, 'backpressure_events': 0}, 'cancel': {'deltas': ['slow'], 'finish_reason': 'client_cancelled', 'finish_events': 1, 'ttft_steps': 1, 'token_intervals': [], 'queue_len': 2, 'backpressure_events': 0}, 'slow': {'deltas': ['slow'], 'finish_reason': 'backpressure', 'finish_events': 1, 'ttft_steps': 1, 'token_intervals': [1], 'queue_len': 2, 'backpressure_events': 1}}
trace_tail= [(1, 'slow', 'backpressure', 2), (2, 'normal', 'delta', ' '), (2, 'stop', 'delta', 'O'), (3, 'normal', 'delta', 'world'), (3, 'stop', 'delta', 'K'), (3, 'stop', 'finish', 'stop_sequence'), (4, 'normal', 'delta', '!'), (5, 'normal', 'finish', 'eos')]
streaming_gates= {'first_token_streamed': True, 'full_text_reconstructed': True, 'finish_event_present': True, 'stop_sequence_hidden': True, 'client_cancelled': True, 'backpressure_detected': True, 'streaming_gate': True}
```

这个 demo 的关键证据：

1. `normal` 的首个 chunk 是 `Hel`，说明首 token 被推送。
2. `normal` 的 deltas 拼回 `Hello world!`，说明增量文本可重构。
3. 每个请求都有一个 finish event。
4. `stop` 的 planned tokens 中包含 `E` 和 `ND`，但输出只到 `OK`，说明 stop sequence 没被泄露。
5. `cancel` 的 finish reason 是 `client_cancelled`。
6. `slow` 触发 `backpressure`，说明慢客户端不会被忽略。

## 13.18 最小代码骨架

核心组件可以整理成：

```python
class StreamingEvent:
    def __init__(self, request_id, event_type, delta="", finish_reason=None):
        self.request_id = request_id
        self.event_type = event_type
        self.delta = delta
        self.finish_reason = finish_reason


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
                StreamingEvent(request.request_id, "delta", delta=delta)
            )

    def finish(self, request):
        request.stream_queue.put(
            StreamingEvent(
                request.request_id,
                "finish",
                finish_reason=request.finish_reason,
            )
        )
```

engine 只需要在 prefill 和 decode 生成 token 后调用 `push_tokens`，在请求结束、取消、超时或错误时调用 `finish`。

## 13.19 常见误区

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

## 13.20 面试追问

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

## 13.21 小练习

1. 在第 12 章 MiniEngine 中加入 `TextStreamer`。
2. 每次 prefill 产生首 token 后，向 stream queue 推送 delta。
3. 每次 decode 产生新 token 后，向 stream queue 推送 delta。
4. 请求结束时推送 finish event。
5. 模拟客户端不读取 queue，观察 backpressure 可能带来的问题。
6. 构造一个 stop sequence 跨 token 的例子，验证不会把 stop sequence 前缀泄露给客户端。

## 13.22 本章小结

本章实现了 token streaming。

Streaming 把 decode loop 的每一步输出变成增量事件。它需要处理增量 detokenization、事件队列、finish event、客户端取消、backpressure、stop sequence 和指标。对用户来说，streaming 让模型尽早开始回答；对 engine 来说，它让请求生命周期多了输出通道和取消反馈。

下一章我们会把当前 MiniEngine 包成一个最小 HTTP API 服务，让外部客户端能够通过接口提交请求并接收流式输出。
