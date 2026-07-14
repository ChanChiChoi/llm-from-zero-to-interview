# 第 14 章 实现最小 HTTP API 服务

前面我们已经实现了 MiniEngine 的核心链路：请求队列、scheduler、prefill、decode 和 token streaming。但这些还停留在内部对象调用。本章把 MiniEngine 包成最小 HTTP API 服务，让外部客户端可以提交请求并接收结果。

这一章的目标不是做完整 OpenAI-compatible server，而是理解 API 层和 engine 层如何解耦：API 层负责接入、参数校验和连接管理，engine 层负责模型执行和调度。

一句话概括：

> 最小 HTTP API 服务是 serving engine 的外部入口，它把 JSON 请求转换成 RequestState，并把 engine 的输出事件返回给客户端。

## 14.0 本讲资料边界与第二轮精修口径

本讲只实现教学版最小 HTTP API 服务。它覆盖请求模型、参数校验、同步 `/generate`、SSE 风格流式响应、队列准入、错误返回、客户端取消、engine cleanup 和最小可运行 demo，但不实现鉴权、租户配额、TLS、反向代理、OpenAPI schema 完整治理、WebSocket、gRPC、跨进程 engine、分布式 worker、生产级日志脱敏或完整 OpenAI-compatible server。

资料校准口径：

1. FastAPI 官方文档用 Pydantic model 声明 request body，框架会读取 JSON、做类型转换、校验数据，并把 schema 用于 OpenAPI 文档。
2. FastAPI / Starlette 的 `StreamingResponse` 可以接收普通 generator 或 async generator，把响应体逐段流式返回；长流式 generator 要能让出控制权，才能被取消。
3. WHATWG HTML Server-Sent Events 标准定义了 `text/event-stream`、`event:`、`data:`、空行分隔、UTF-8 编码和 `EventSource` 消费方式。
4. OpenAI API streaming 文档当前推荐 Responses API 的 typed semantic events；Chat Completions streaming 仍使用 data-only SSE，增量内容通过 `delta` 字段返回。本章只借鉴这些接口形态，不复刻完整字段。
5. 本章 demo 用纯 Python 模拟 HTTP handler、engine queue、SSE frame 和取消清理，不依赖 FastAPI，也不启动本地服务。

参考资料：

1. FastAPI Request Body：<https://fastapi.tiangolo.com/tutorial/body/>
2. FastAPI `StreamingResponse`：<https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse>
3. Starlette `StreamingResponse`：<https://www.starlette.io/responses/#streamingresponse>
4. WHATWG HTML Server-Sent Events：<https://html.spec.whatwg.org/multipage/server-sent-events.html>
5. OpenAI API streaming responses：<https://platform.openai.com/docs/guides/streaming-responses>

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

## 14.2 最小接口设计与 API 门禁公式

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

把 API 层写成可验收系统，而不是只写一个 handler，可以先定义几个门禁。

请求合法性门禁：

```math
G_{\mathrm{valid}}=G_{\mathrm{prompt}}G_{\mathrm{tokens}}G_{\mathrm{sampling}}
```

其中 `G_{\mathrm{prompt}}` 检查 prompt 是非空字符串，`G_{\mathrm{tokens}}` 检查 `max_new_tokens` 在允许范围内，`G_{\mathrm{sampling}}` 检查 temperature、top-p、top-k 等参数不越界。

队列准入条件：

```math
Q_t<Q_{\max}
```

这里 `Q_t` 是当前等待队列长度，`Q_{\max}` 是 API 层允许提交给 engine 的最大等待队列长度。超过时应返回 429，而不是继续把请求塞进 engine。

HTTP 成功率和错误率：

```math
R_{2xx}=\frac{N_{2xx}}{N},\quad
R_{4xx}=\frac{N_{4xx}}{N},\quad
R_{5xx}=\frac{N_{5xx}}{N}
```

`4xx` 通常表示客户端输入、配额或准入失败，`5xx` 才表示服务端内部失败。把参数错误都变成 `500`，会让线上告警和容量判断失真。

取消清理门禁：

```math
G_{\mathrm{cancel}}=G_{\mathrm{abort}}G_{\mathrm{kvfree}}G_{\mathrm{finish}}
```

`G_{\mathrm{abort}}` 表示取消信号能传回 engine，`G_{\mathrm{kvfree}}` 表示 KV cache / running slot 被释放，`G_{\mathrm{finish}}` 表示客户端或日志能看到明确 finish reason。

最小 API 总门禁：

```math
G_{\mathrm{api}}=G_{\mathrm{valid}}G_{\mathrm{admit}}G_{\mathrm{sync}}G_{\mathrm{stream}}G_{\mathrm{cancel}}
```

一个最小 HTTP API 通过这个门禁，才算真正把请求接入、同步响应、流式响应、过载拒绝和取消清理连成闭环。

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


def sse_format(event_name, data):
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event_name}\ndata: {payload}\n\n"
```

SSE 的关键不是“每次写一行字符串”，而是线格式稳定：

1. `Content-Type` 应是 `text/event-stream`。
2. 一个事件可以包含 `event:` 和一行或多行 `data:`。
3. 事件之间用空行分隔。
4. 客户端断开时，服务端 generator 必须退出，并把取消信号传回 engine。

流式生成器：

```python
def stream_events(request_state):
    while True:
        event = request_state.stream_queue.get()

        if event.error:
            yield sse_format("error", {"error": event.error})
            break

        if event.finish_reason is not None:
            yield sse_format("finish", {
                "request_id": event.request_id,
                "delta": "",
                "finish_reason": event.finish_reason,
            })
            break

        yield sse_format("delta", {
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
                yield sse_format("finish", {"finish_reason": event.finish_reason})
                break
            yield sse_format("delta", {"delta": event.delta})
    finally:
        if not request_state.finished:
            request_state.aborted = True
            request_state.finish_reason = "abort"
```

engine 的 scheduler 或 decode loop 应该检查 `request_state.aborted`，停止继续生成并释放资源。

如果取消只停在 API 层，GPU 会继续做无用工作。

生产实现还要区分几种取消来源：

1. 客户端主动断开连接。
2. API 层等待超时。
3. engine 内部错误导致请求终止。
4. admission 或限流阶段直接拒绝。

这些路径最后都应该进入同一个 cleanup 收敛点：移出 waiting / running 集合，释放 KV cache，写 finish reason，更新 metrics 和 trace。

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
POST /v1/responses
POST /v1/chat/completions
POST /v1/completions
```

它比本章接口复杂，因为要处理：

1. `messages` 和 chat template。
2. Responses API 的 typed semantic events。
3. Chat Completions 的 `choices` 和 `delta` 格式。
4. `usage` 统计和流式 usage 汇总。
5. `stop`、`logprobs`、`seed`、tool call 等参数。
6. 多模型字段、模型别名和兼容版本。
7. 错误码、错误 body、finish reason 和取消语义兼容。

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

## 14.14 HTTP API 公式、接口和可运行 demo

下面的 demo 不依赖外部库，也不启动真实 HTTP server。它模拟一个最小 API 层：

1. 校验 JSON 请求，非法 prompt 返回 400。
2. 调用 engine 准入，队列满返回 429。
3. 非流式 `/generate` 一次性返回完整文本。
4. 流式 `/generate_stream` 返回 SSE frame。
5. 客户端取消时把 abort 信号传回 engine，并释放 KV。

```python
from dataclasses import dataclass, field
import json


@dataclass
class GenerateRequest:
    prompt: str
    max_new_tokens: int = 16
    temperature: float = 0.7
    top_p: float = 1.0
    stream: bool = False


@dataclass
class RequestState:
    request_id: str
    prompt: str
    max_new_tokens: int
    temperature: float
    top_p: float
    stream: bool
    output_tokens: list[str] = field(default_factory=list)
    status: str = "WAITING"
    finish_reason: str | None = None
    kv_allocated: bool = True


class ToyEngine:
    def __init__(self, max_queue_size=2):
        self.max_queue_size = max_queue_size
        self.waiting_queue = []
        self.released_kv = []
        self.trace = []

    def can_accept(self):
        return len(self.waiting_queue) < self.max_queue_size

    def add_request(self, state):
        if not self.can_accept():
            return False
        self.waiting_queue.append(state)
        self.trace.append((state.request_id, "queued"))
        return True

    def planned_text(self, state):
        prompt = state.prompt.lower()
        if "sync" in prompt:
            return "SYNC"
        if "stream" in prompt:
            return "API!"
        if "cancel" in prompt:
            return "CANCEL"
        return "OK"

    def complete(self, state):
        state.status = "RUNNING"
        text = self.planned_text(state)[:state.max_new_tokens]
        state.output_tokens.extend(text)
        state.finish_reason = "length"
        state.status = "FINISHED"
        self.release_kv(state)
        return text

    def release_kv(self, state):
        if state.kv_allocated:
            state.kv_allocated = False
            self.released_kv.append(state.request_id)
        if state in self.waiting_queue:
            self.waiting_queue.remove(state)
        self.trace.append((state.request_id, "released"))

    def cancel(self, state):
        state.status = "FINISHED"
        state.finish_reason = "client_cancelled"
        self.release_kv(state)
        self.trace.append((state.request_id, "cancelled"))


class MinimalHTTPAPI:
    def __init__(self, engine):
        self.engine = engine
        self.next_id = 1

    def error(self, status_code, message):
        return {
            "status_code": status_code,
            "body": {"error": {"message": message}},
        }

    def validate(self, payload):
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return None, self.error(400, "prompt must be a non-empty string")

        max_new_tokens = payload.get("max_new_tokens", 16)
        if not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 64:
            return None, self.error(400, "max_new_tokens must be an integer in [1, 64]")

        temperature = payload.get("temperature", 0.7)
        if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
            return None, self.error(400, "temperature must be in [0, 2]")

        top_p = payload.get("top_p", 1.0)
        if not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
            return None, self.error(400, "top_p must be in (0, 1]")

        stream = payload.get("stream", False)
        if not isinstance(stream, bool):
            return None, self.error(400, "stream must be a boolean")

        return GenerateRequest(
            prompt=prompt.strip(),
            max_new_tokens=max_new_tokens,
            temperature=float(temperature),
            top_p=float(top_p),
            stream=stream,
        ), None

    def new_request_state(self, request):
        request_id = f"req-{self.next_id:04d}"
        self.next_id += 1
        return RequestState(
            request_id=request_id,
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=request.stream,
        )

    def submit_or_429(self, payload):
        request, error_response = self.validate(payload)
        if error_response is not None:
            return None, error_response
        if not self.engine.can_accept():
            return None, self.error(429, "server is busy")

        state = self.new_request_state(request)
        self.engine.add_request(state)
        return state, None

    def generate(self, payload):
        state, error_response = self.submit_or_429(payload)
        if error_response is not None:
            return error_response

        text = self.engine.complete(state)
        return {
            "status_code": 200,
            "body": {
                "request_id": state.request_id,
                "text": text,
                "finish_reason": state.finish_reason,
                "usage": {"output_tokens": len(state.output_tokens)},
            },
        }

    def sse_frame(self, event_name, payload):
        data = json.dumps(payload, ensure_ascii=False)
        return f"event: {event_name}\ndata: {data}\n\n"

    def generate_stream(self, payload):
        state, error_response = self.submit_or_429(payload)
        if error_response is not None:
            yield self.sse_frame("error", error_response["body"])
            return

        text = self.engine.planned_text(state)[:state.max_new_tokens]
        state.status = "RUNNING"
        for token in text:
            state.output_tokens.append(token)
            yield self.sse_frame("delta", {"delta": token, "event": "delta"})

        state.finish_reason = "length"
        state.status = "FINISHED"
        yield self.sse_frame(
            "finish",
            {
                "event": "finish",
                "finish_reason": state.finish_reason,
                "request_id": state.request_id,
                "usage": {"output_tokens": len(state.output_tokens)},
            },
        )
        self.engine.release_kv(state)

    def stream_until_cancel(self, payload, cancel_after_frames):
        state, error_response = self.submit_or_429(payload)
        if error_response is not None:
            return {"finish_reason": "not_started", "frames": []}

        frames = []
        text = self.engine.planned_text(state)[:state.max_new_tokens]
        state.status = "RUNNING"
        for token in text:
            state.output_tokens.append(token)
            frames.append(self.sse_frame("delta", {"delta": token, "event": "delta"}))
            if len(frames) >= cancel_after_frames:
                self.engine.cancel(state)
                frames.append(
                    self.sse_frame(
                        "finish",
                        {
                            "event": "finish",
                            "finish_reason": state.finish_reason,
                            "request_id": state.request_id,
                        },
                    )
                )
                break
        return {"finish_reason": state.finish_reason, "frames": frames}


def run_http_api_demo():
    engine = ToyEngine(max_queue_size=2)
    api = MinimalHTTPAPI(engine)

    sync_response = api.generate(
        {"prompt": "sync request", "max_new_tokens": 4, "temperature": 0.0}
    )
    stream_frames = list(
        api.generate_stream({"prompt": "stream request", "max_new_tokens": 4, "stream": True})
    )
    invalid_response = api.generate({"prompt": "", "max_new_tokens": 4})

    engine.waiting_queue = [
        RequestState("busy-a", "queued", 1, 0.7, 1.0, False),
        RequestState("busy-b", "queued", 1, 0.7, 1.0, False),
    ]
    busy_response = api.generate({"prompt": "busy request", "max_new_tokens": 4})
    engine.waiting_queue = []

    cancel_result = api.stream_until_cancel(
        {"prompt": "cancel request", "max_new_tokens": 6, "stream": True},
        cancel_after_frames=1,
    )

    summary = {
        "sync_response": sync_response,
        "stream_frame_count": len(stream_frames),
        "stream_first_frame": stream_frames[0].strip(),
        "invalid_response": invalid_response,
        "busy_response": busy_response,
        "cancel_finish_reason": cancel_result["finish_reason"],
        "kv_released": engine.released_kv,
    }
    gates = {
        "sync_200": sync_response["status_code"] == 200
        and sync_response["body"]["text"] == "SYNC",
        "stream_sse_frames": len(stream_frames) == 5
        and stream_frames[0].startswith("event: delta\n")
        and stream_frames[-1].startswith("event: finish\n"),
        "invalid_400": invalid_response["status_code"] == 400,
        "queue_full_429": busy_response["status_code"] == 429,
        "cancel_cleanup": cancel_result["finish_reason"] == "client_cancelled"
        and "req-0003" in engine.released_kv,
    }
    gates["http_api_gate"] = all(gates.values())
    return summary, gates


summary, gates = run_http_api_demo()
print("api_summary=", summary)
print("http_api_gates=", gates)
```

一组稳定输出：

```text
api_summary= {'sync_response': {'status_code': 200, 'body': {'request_id': 'req-0001', 'text': 'SYNC', 'finish_reason': 'length', 'usage': {'output_tokens': 4}}}, 'stream_frame_count': 5, 'stream_first_frame': 'event: delta\ndata: {"delta": "A", "event": "delta"}', 'invalid_response': {'status_code': 400, 'body': {'error': {'message': 'prompt must be a non-empty string'}}}, 'busy_response': {'status_code': 429, 'body': {'error': {'message': 'server is busy'}}}, 'cancel_finish_reason': 'client_cancelled', 'kv_released': ['req-0001', 'req-0002', 'req-0003']}
http_api_gates= {'sync_200': True, 'stream_sse_frames': True, 'invalid_400': True, 'queue_full_429': True, 'cancel_cleanup': True, 'http_api_gate': True}
```

这个 demo 的关键证据：

1. `sync_response` 返回 200、完整文本、finish reason 和 usage。
2. `stream_frame_count` 是 5，说明 4 个 delta 加 1 个 finish event 都被写成 SSE frame。
3. 空 prompt 返回 400，不进入 engine。
4. 队列满返回 429，不创建新 request id。
5. 取消请求的 finish reason 是 `client_cancelled`，并且 `kv_released` 包含对应 request id。

## 14.15 常见误区

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

## 14.16 面试追问

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

## 14.17 小练习

1. 用 FastAPI 写一个 `/generate` endpoint，把 prompt 转成 RequestState。
2. 写一个后台线程循环调用 `engine.step()`。
3. 实现非流式等待，直到 request finished 后返回完整文本。
4. 实现 SSE streaming，从 `stream_queue` 中读取事件。
5. 给 API 加上最大队列长度限制，超过后返回 429。

## 14.18 本章小结

本章把 MiniEngine 包成了最小 HTTP API 服务。

HTTP API 层负责接收 JSON 请求、校验参数、创建 RequestState、提交到 engine，并把非流式结果或流式事件返回给客户端。engine 层继续负责队列、调度、prefill、decode 和 streaming。两层通过请求对象和 stream queue 解耦。

下一章我们会做压测，系统分析 TTFT、TPOT、吞吐和显存，看看这个最小推理框架的瓶颈在哪里。
