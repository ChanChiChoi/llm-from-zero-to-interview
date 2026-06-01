# 第 14 章 实现最小 HTTP API 服务

前面我们已经实现了 MiniEngine 的核心链路：请求队列、scheduler、prefill、decode 和 token streaming。但这些还停留在内部对象调用。本章把 MiniEngine 包成最小 HTTP API 服务，让外部客户端可以提交请求并接收结果。

这一章的目标不是做完整 OpenAI-compatible server，而是理解 API 层和 engine 层如何解耦：API 层负责接入、参数校验和连接管理，engine 层负责模型执行和调度。

一句话概括：

> 最小 HTTP API 服务是 serving engine 的外部入口，它把 JSON 请求转换成 RequestState，并把 engine 的输出事件返回给客户端。

## 14.1 HTTP API 在架构中的位置

整体结构可以写成：

```text
Client
  -> HTTP API Server
  -> RequestState
  -> MiniEngine waiting queue
  -> Scheduler / Prefill / Decode
  -> StreamQueue
  -> HTTP Response
```

API Server 不应该直接调用模型 forward。它应该把请求交给 engine，然后从 stream queue 或结果对象中读取输出。

这样做的好处是：

1. API 层和模型执行层解耦。
2. engine 可以独立调度多个请求。
3. streaming 输出可以通过队列异步返回。
4. 后续可以替换 HTTP 为 gRPC、WebSocket 或内部 RPC。

## 14.2 最小接口设计

我们先设计一个简单接口：

```text
POST /generate
```

请求体：

```json
{
  "prompt": "Explain KV Cache",
  "max_new_tokens": 128,
  "temperature": 0.7,
  "stream": false
}
```

非流式响应：

```json
{
  "request_id": "r-123",
  "text": "...",
  "finish_reason": "stop",
  "usage": {
    "output_tokens": 128
  }
}
```

流式响应则逐段返回 delta。

## 14.3 API 层应该校验什么

最小校验包括：

1. `prompt` 是否存在且为字符串。
2. `max_new_tokens` 是否为正整数。
3. `temperature` 是否合法。
4. `top_p` 是否在合理范围。
5. 请求体大小是否超限。
6. prompt token 数是否超过上下文窗口。
7. 当前队列是否已满。

不要把所有错误都留给 engine。越早拒绝非法请求，越少浪费 GPU 资源。

## 14.4 FastAPI 最小骨架

教学版可以用 FastAPI 表达 API 层。

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid


app = FastAPI()


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    stream: bool = False


class GenerateResponse(BaseModel):
    request_id: str
    text: str
    finish_reason: str
    output_tokens: int
```

然后定义 endpoint：

```python
@app.post("/generate")
def generate(req: GenerateRequest):
    if not req.prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    request_id = str(uuid.uuid4())
    request_state = RequestState(
        request_id=request_id,
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
    )

    engine.add_request(request_state)
    result = wait_until_finished(request_state)

    return GenerateResponse(
        request_id=request_id,
        text=result.text,
        finish_reason=request_state.finish_reason,
        output_tokens=len(request_state.output_ids),
    )
```

这只是同步版本，适合理解接口，不适合高并发生产。

## 14.5 engine 后台循环

API 层把请求放入 queue 后，必须有后台循环持续调用 `engine.step()`。

最小写法：

```python
import threading
import time


def engine_loop():
    while True:
        engine.step()
        time.sleep(0.001)


threading.Thread(target=engine_loop, daemon=True).start()
```

这样 API 请求只负责入队，后台线程负责调度和模型执行。

生产系统里会使用更严谨的事件循环、异步队列、worker 线程或独立 engine process。

## 14.6 等待非流式结果

非流式模式下，HTTP handler 可以等待请求完成。

```python
def wait_until_finished(request_state, timeout=60):
    start = time.time()
    while not request_state.finished:
        if time.time() - start > timeout:
            request_state.finished = True
            request_state.finish_reason = "timeout"
            raise HTTPException(status_code=504, detail="request timeout")
        time.sleep(0.01)

    text = tokenizer_wrapper.decode(request_state.output_ids)
    return type("Result", (), {"text": text})
```

这个实现很粗糙，但说明了非流式接口的本质：等待 engine 生成完，再一次性返回完整文本。

缺点是：

1. HTTP 连接占用时间长。
2. 用户长时间看不到输出。
3. handler 线程被阻塞。
4. 超时和取消处理更复杂。

## 14.7 SSE 流式响应

流式 API 可以用 SSE 表达。

FastAPI 中可以返回 `StreamingResponse`：

```python
from fastapi.responses import StreamingResponse
import json


def sse_format(data):
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
```

流式生成器：

```python
def stream_events(request_state):
    while True:
        event = request_state.stream_queue.get()

        if event.error:
            yield sse_format({"error": event.error})
            break

        if event.finish_reason is not None:
            yield sse_format({
                "request_id": event.request_id,
                "delta": "",
                "finish_reason": event.finish_reason,
            })
            break

        yield sse_format({
            "request_id": event.request_id,
            "delta": event.delta,
            "finish_reason": None,
        })
```

endpoint：

```python
@app.post("/generate_stream")
def generate_stream(req: GenerateRequest):
    request_id = str(uuid.uuid4())
    request_state = RequestState(
        request_id=request_id,
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
    )

    engine.add_request(request_state)
    return StreamingResponse(
        stream_events(request_state),
        media_type="text/event-stream",
    )
```

这就是最小 streaming API。

## 14.8 请求取消如何传回 engine

HTTP 客户端可能中途断开。API 层应该把取消信号传给 engine。

教学版可以在生成器退出时标记：

```python
def stream_events(request_state):
    try:
        while True:
            event = request_state.stream_queue.get()
            if event.finish_reason is not None:
                yield sse_format({"finish_reason": event.finish_reason})
                break
            yield sse_format({"delta": event.delta})
    finally:
        if not request_state.finished:
            request_state.aborted = True
            request_state.finish_reason = "abort"
```

engine 的 scheduler 或 decode loop 应该检查 `request_state.aborted`，停止继续生成并释放资源。

如果取消只停在 API 层，GPU 会继续做无用工作。

## 14.9 API 层和 sampling params

第 8 章我们实现了 sampling params。HTTP 请求里的参数应该传入 RequestState。

```python
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int | None = None
    stream: bool = False
```

RequestState 可以保存：

```python
request_state.sampling_params = SamplingParams(
    temperature=req.temperature,
    top_p=req.top_p,
    top_k=req.top_k,
)
```

教学版 engine 可能仍然共用一个 sampler。生产 engine 需要 per-request sampling params。

## 14.10 API 层的并发风险

最小 HTTP 服务有很多并发风险：

1. 多个 handler 同时访问 engine。
2. waiting queue 需要线程安全。
3. request 状态被 engine 和 API 同时读写。
4. stream_queue 可能阻塞。
5. engine_loop 异常后请求永远等待。
6. 模型执行不能被多个线程同时乱调用。

教学版可以用锁保护入队：

```python
engine_lock = threading.Lock()

with engine_lock:
    engine.add_request(request_state)
```

生产版通常会让 engine 单线程消费队列，API 层通过线程安全或异步队列提交请求。

## 14.11 队列满了怎么办

API 层必须有 admission control。

如果 waiting queue 已经太长，还继续接收请求，会导致所有请求 TTFT 变差。

最小策略：

```python
if len(engine.waiting_queue) > MAX_QUEUE_SIZE:
    raise HTTPException(status_code=429, detail="server is busy")
```

更完整策略包括：

1. 按租户限流。
2. 按 prompt token 数限流。
3. 按预估 KV Cache 限流。
4. 按优先级队列接入。
5. 返回排队位置或建议重试时间。

如果没有 admission control，高峰期系统可能从慢变成雪崩。

## 14.12 OpenAI-compatible API 的差异

真实项目常提供 OpenAI-compatible API，例如：

```text
POST /v1/chat/completions
POST /v1/completions
```

它比本章接口复杂，因为要处理：

1. `messages` 和 chat template。
2. `choices` 格式。
3. `delta` 格式。
4. `usage` 统计。
5. `stop`、`logprobs`、`seed` 等参数。
6. 多模型字段。
7. 错误码兼容。

本章先用 `/generate` 是为了专注 engine 和 API 的连接。

理解后再做兼容层，会清楚很多。

## 14.13 最小服务文件结构

一个教学项目可以这样拆：

```text
mini_engine/
  engine.py        MiniEngine, Scheduler, RequestQueue
  request.py       RequestState, StreamingEvent
  model.py         ModelWrapper
  tokenizer.py     TokenizerWrapper
  sampler.py       Sampler, SamplingParams
  api.py           FastAPI endpoints
```

不要一开始把所有代码写进一个 `server.py`。拆分不是为了复杂，而是为了让每个模块职责清楚。

## 14.14 常见误区

误区一：HTTP API 直接调用 `model.generate()` 就是 serving engine。

这只是 Web demo，不具备请求队列、scheduler、KV Cache 管理和 streaming 生命周期。

误区二：非流式接口更简单，所以生产只做非流式就好。

非流式用户体验差，长请求更容易超时，也不利于取消反馈。

误区三：API 层不需要关心队列长度。

API 层必须做 admission control，否则过载时会拖垮系统。

误区四：客户端断开只要 HTTP 连接关闭就行。

必须通知 engine 取消请求并释放资源。

误区五：OpenAI-compatible 只是改字段名。

兼容接口还涉及 messages、stream delta、usage、finish reason、错误码和参数语义。

## 14.15 面试追问

1. HTTP API 层和 serving engine 层应该如何解耦？
2. 非流式 `/generate` 的执行流程是什么？
3. 流式 SSE 接口如何从 engine 获取 token delta？
4. 客户端断开时如何通知 engine？
5. 为什么 API 层要做 admission control？
6. 多线程 API 访问 engine 有哪些并发风险？
7. OpenAI-compatible API 比简单 `/generate` 多了哪些复杂度？
8. 为什么不建议 API handler 直接调用模型 forward？

参考回答思路：

1. 先说 API 层负责协议、校验、连接和响应，engine 层负责调度和模型执行。
2. 再说请求转成 RequestState 入队，engine 后台 loop 消费队列。
3. 然后说明 streaming 通过 stream queue 或事件流返回。
4. 最后补取消、限流、线程安全和 OpenAI-compatible 兼容细节。

## 14.16 小练习

1. 用 FastAPI 写一个 `/generate` endpoint，把 prompt 转成 RequestState。
2. 写一个后台线程循环调用 `engine.step()`。
3. 实现非流式等待，直到 request finished 后返回完整文本。
4. 实现 SSE streaming，从 `stream_queue` 中读取事件。
5. 给 API 加上最大队列长度限制，超过后返回 429。

## 14.17 本章小结

本章把 MiniEngine 包成了最小 HTTP API 服务。

HTTP API 层负责接收 JSON 请求、校验参数、创建 RequestState、提交到 engine，并把非流式结果或流式事件返回给客户端。engine 层继续负责队列、调度、prefill、decode 和 streaming。两层通过请求对象和 stream queue 解耦。

下一章我们会做压测，系统分析 TTFT、TPOT、吞吐和显存，看看这个最小推理框架的瓶颈在哪里。
