# 第 57 章 OpenAI-compatible API、流式协议、错误码和限流鉴权

第 56 章我们讨论了多 GPU worker 内部的 tensor parallel、pipeline parallel 和分布式 KV cache。

到这里，一个 LLM serving engine 的内部能力已经比较完整：

1. continuous batching。
2. PagedAttention 和 KV block manager。
3. prefix cache。
4. preemption。
5. benchmark 和 trace。
6. 异步 engine loop。
7. 多 worker router。
8. 多 GPU parallel group。

但这些还只是内部系统。

如果要真正对外提供服务，还需要一个稳定的 API 层。

用户不会直接调用 scheduler，也不会直接操作 KV cache。

用户看到的是 HTTP endpoint、JSON request、streaming response、错误码、鉴权和限流。

这一层做不好，engine 再强也很难被可靠使用。

本章讨论如何把 inference engine 包装成 OpenAI-compatible serving API。

## 57.0 本讲资料边界与第二轮精修口径

本章按第二轮精修口径，只讲教学版 serving engine 如何暴露 OpenAI-compatible API 的稳定工程边界。

公开资料校准主要参考四类口径：

1. OpenAI Chat Completions API reference 对 `messages`、`choices`、`finish_reason`、stream chunk、`stream_options.include_usage` 等字段的公开说明。
2. OpenAI streaming guide 对 `stream=true`、HTTP streaming over server-sent events、Chat Completions / legacy Completions 增量 chunk 的公开说明。
3. OpenAI error codes guide 对 401、403、429、500、503 等错误含义，以及认证、额度、rate limit 和 overload 的公开说明。
4. OpenAI rate limits guide 对 RPM、RPD、TPM、TPD、IPM、audio minutes 等多维速率限制，以及 synchronous request 的 RPM / TPM 区分的公开说明。

本章不复刻 OpenAI 官方 API 的全部字段、模型行为、SDK 实现、Responses API 完整事件流、tool calls、vision / audio content parts、project / organization 权限体系、真实 billing quota、真实风控策略或官方错误消息。我们只实现一个教学版兼容子集：

```text
Bearer auth -> JSON validation -> model permission -> chat template -> token budget -> RPM / TPM / concurrency limit -> EngineRequest -> non-stream response or SSE stream -> error response with request_id
```

第二轮新增 demo 的验收重点是：兼容层要诚实声明支持字段；不支持字段不能静默吞掉；streaming chunk 要可解析且以 `[DONE]` 结束；鉴权、权限和限流要在昂贵 tokenization / engine admission 前完成；rate limit 与系统 admission control 要区分；错误响应必须有稳定的 `type`、`code`、`param`、`message` 和 `request_id`。

## 57.1 本章目标

读完本章，你应该能讲清：

1. 为什么很多 serving 系统要兼容 OpenAI API。
2. `/v1/chat/completions` 和 `/v1/completions` 的核心差异。
3. chat messages 如何转成 prompt 和 token ids。
4. SSE streaming 协议如何组织增量输出。
5. 非流式和流式接口的状态管理差异。
6. 错误码应该如何分层。
7. 鉴权、配额和限流应该放在哪些位置。
8. request id、trace id、idempotency key 有什么作用。
9. API 层如何把请求安全地交给 router 和 engine。

本章重点不是复刻某个厂商的所有字段。

重点是理解一个生产级 LLM API 层应该具备哪些工程边界。

## 57.2 为什么要兼容 OpenAI API

OpenAI-compatible API 已经成为事实标准之一。

很多 SDK、agent framework、评测工具、IDE 插件和业务系统默认支持它。

如果你的 serving 系统兼容这些接口，用户接入成本会低很多。

兼容的收益包括：

1. 可以直接使用 OpenAI SDK。
2. 可以接入 LangChain、LlamaIndex、AutoGen 等框架。
3. 可以复用已有评测脚本。
4. 可以让业务迁移更容易。
5. 可以降低 API 文档解释成本。

但兼容不等于盲目复制。

你需要明确：

1. 哪些字段支持。
2. 哪些字段忽略。
3. 哪些字段不支持并返回错误。
4. streaming 格式是否严格兼容。
5. 错误响应格式是否稳定。

不要让 API 看起来兼容，实际行为却不兼容。

这会让用户 debug 非常痛苦。

## 57.3 常见 endpoint

最常见的 endpoint 包括：

```text
GET  /v1/models
POST /v1/completions
POST /v1/chat/completions
POST /v1/embeddings
```

对生成式 LLM serving 来说，最重要的是：

```text
POST /v1/chat/completions
```

它接收 messages，返回 assistant message。

一个最小请求：

```json
{
  "model": "tiny-llm-chat",
  "messages": [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user", "content": "解释一下 KV cache。"}
  ],
  "max_tokens": 256,
  "temperature": 0.7,
  "stream": true
}
```

一个最小非流式响应：

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "tiny-llm-chat",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "KV cache 是推理时缓存历史 token 的 key/value..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 18,
    "completion_tokens": 64,
    "total_tokens": 82
  }
}
```

## 57.4 Chat Completions 和 Completions

`/v1/completions` 接收的是 prompt：

```json
{
  "model": "tiny-llm",
  "prompt": "The capital of France is",
  "max_tokens": 16
}
```

`/v1/chat/completions` 接收的是 messages：

```json
{
  "model": "tiny-llm-chat",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is the capital of France?"}
  ]
}
```

对 engine 来说，两者最终都会变成 token ids。

区别在 API 层：

1. completions 直接把 prompt tokenize。
2. chat completions 需要先套 chat template。
3. chat messages 有 role 和多轮上下文。
4. chat 可能包含 tool calls、multi-modal content 等复杂结构。

可以把转换路径理解为：

```text
HTTP JSON
  -> validate request
  -> normalize fields
  -> render prompt text by template
  -> tokenize
  -> route to worker
  -> stream or collect output
```

## 57.5 Chat template

chat model 训练时通常使用固定格式。

例如：

```text
<|system|>
你是一个有帮助的助手。
<|user|>
解释一下 KV cache。
<|assistant|>
```

API 层要把 messages 渲染成模型期望的 prompt。

伪代码：

```python
def render_chat_prompt(messages: list[dict]) -> str:
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"<|system|>\n{content}\n")
        elif role == "user":
            parts.append(f"<|user|>\n{content}\n")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}\n")
        else:
            raise BadRequestError(f"unsupported role: {role}")
    parts.append("<|assistant|>\n")
    return "".join(parts)
```

真实系统不要手写错模板。

应该使用 tokenizer 或模型配置里定义的 chat template。

因为模板细节会影响模型输出质量。

一个多余空格、缺少 assistant prompt、错误 role token，都可能让模型行为异常。

## 57.6 请求字段校验

API 层必须在请求进入 engine 前做严格校验。

常见字段包括：

```text
model
messages / prompt
max_tokens
temperature
top_p
top_k
stop
stream
n
seed
presence_penalty
frequency_penalty
logprobs
user
```

校验要回答：

1. 字段类型是否正确。
2. 数值范围是否合理。
3. model 是否存在。
4. prompt/messages 是否为空。
5. prompt tokens 是否超过 max_model_len。
6. max_tokens 是否超过服务限制。
7. 不支持的字段如何处理。
8. stop sequence 数量和长度是否受限。

例如：

```python
def validate_sampling_params(req):
    if req.max_tokens <= 0:
        raise BadRequestError("max_tokens must be positive")
    if req.max_tokens > 4096:
        raise BadRequestError("max_tokens exceeds limit")
    if not (0.0 <= req.temperature <= 2.0):
        raise BadRequestError("temperature must be between 0 and 2")
    if not (0.0 < req.top_p <= 1.0):
        raise BadRequestError("top_p must be in (0, 1]")
```

坏请求应该尽早失败。

不要让明显非法请求进入 tokenizer、router 或 GPU engine。

## 57.7 从 API request 到 EngineRequest

API request 不应该直接作为 engine 内部对象。

更好的做法是转换成稳定的内部结构：

```python
from dataclasses import dataclass


@dataclass
class SamplingParams:
    max_tokens: int
    temperature: float
    top_p: float
    stop: list[str]
    seed: int | None


@dataclass
class EngineRequest:
    request_id: str
    model: str
    prompt: str
    prompt_token_ids: list[int]
    sampling: SamplingParams
    stream: bool
    tenant_id: str
    user_id: str | None
    trace_id: str
    created_at: float
```

这样 API 层和 engine 层解耦。

以后 API 新增字段，不一定要污染 scheduler。

engine 只关心真正影响生成的参数和调度需要的元信息。

## 57.8 非流式响应

非流式接口比较简单。

API server 把请求交给 engine，然后等待完整输出。

```text
request -> engine -> collect all tokens -> detokenize -> JSON response
```

伪代码：

```python
async def chat_completions(req):
    engine_req = await build_engine_request(req)
    result = await engine.generate(engine_req)
    return build_chat_completion_response(engine_req, result)
```

非流式响应适合：

1. 短文本。
2. 后台任务。
3. eval。
4. 不需要边生成边展示的场景。

缺点是用户必须等全部 token 生成完才收到响应。

长输出场景下，TTFT 体验很差。

## 57.9 流式响应

流式接口会在生成过程中逐步返回 token delta。

OpenAI-compatible streaming 通常使用 SSE：

```text
Content-Type: text/event-stream
```

每个事件形如：

```text
data: {json}\n\n
```

最后发送：

```text
data: [DONE]\n\n
```

一个 chat completion chunk 示例：

```text
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1710000000,"model":"tiny-llm-chat","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1710000000,"model":"tiny-llm-chat","choices":[{"index":0,"delta":{"content":"KV"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1710000000,"model":"tiny-llm-chat","choices":[{"index":0,"delta":{"content":" cache"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1710000000,"model":"tiny-llm-chat","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

```

注意第一个 chunk 通常会发送 role。

后续 chunk 发送 content delta。

结束 chunk 发送 finish_reason。

## 57.10 SSE 的几个坑

SSE 看起来简单，但有很多细节。

### 57.10.1 每个事件要以空行结束

SSE 事件之间用空行分隔。

也就是：

```text
data: {...}\n\n
```

少一个换行，客户端可能一直等不到事件。

### 57.10.2 不要缓存响应

应该设置类似 header：

```text
Cache-Control: no-cache
Connection: keep-alive
Content-Type: text/event-stream
```

如果经过 nginx 或网关，还要避免 proxy buffering。

否则 token 已经生成了，但客户端迟迟收不到。

### 57.10.3 处理 client disconnect

流式连接可能随时断开。

断开后必须 cancel engine request。

否则会产生僵尸请求。

这是第 54 章强调过的问题。

### 57.10.4 delta 不是完整文本

stream chunk 里的 `delta.content` 是增量文本。

客户端需要自己拼接。

服务端不要每次都发送完整历史文本。

否则带宽会随输出长度二次增长。

## 57.11 finish_reason

`finish_reason` 告诉用户生成为什么结束。

常见值包括：

```text
stop
length
content_filter
tool_calls
error
```

最常见的是：

1. `stop`：遇到 EOS 或 stop sequence。
2. `length`：达到 max_tokens。
3. `error`：生成中发生错误。

engine 内部可以有更细的结束原因：

```python
class FinishReason:
    EOS = "eos"
    STOP_SEQUENCE = "stop_sequence"
    MAX_TOKENS = "max_tokens"
    CANCELLED = "cancelled"
    ERROR = "error"
```

API 层再映射到对外协议：

```python
def to_api_finish_reason(reason):
    if reason in ("eos", "stop_sequence"):
        return "stop"
    if reason == "max_tokens":
        return "length"
    if reason == "error":
        return "error"
    return "stop"
```

不要把内部 enum 原样暴露给用户。

内部实现可以变，对外协议应该稳定。

## 57.12 Usage 统计

非流式响应通常包含 usage：

```json
"usage": {
  "prompt_tokens": 18,
  "completion_tokens": 64,
  "total_tokens": 82
}
```

这些数字看似简单，但很重要。

它们用于：

1. 计费。
2. 配额。
3. 成本分析。
4. debug prompt 长度。
5. benchmark。

prompt_tokens 应该来自 tokenizer 后的长度。

completion_tokens 应该来自实际生成 token 数。

如果 stream 模式也要返回 usage，可以在最终 chunk 附带，或者提供额外配置。

但要文档化行为。

## 57.13 错误响应格式

错误响应必须稳定。

一个 OpenAI-compatible 风格错误可以长这样：

```json
{
  "error": {
    "message": "max_tokens exceeds limit",
    "type": "invalid_request_error",
    "param": "max_tokens",
    "code": "max_tokens_exceeded"
  }
}
```

建议至少包含：

1. message：给人看的错误说明。
2. type：错误大类。
3. param：相关字段，可为空。
4. code：稳定机器可读错误码。
5. request_id：方便排查。

不要只返回：

```json
{"detail":"error"}
```

这对调用方和排障都不够。

## 57.14 错误码分层

常见 HTTP 状态码和错误类型：

```text
400 invalid_request_error
401 authentication_error
403 permission_denied
404 model_not_found
408 request_timeout
409 conflict_error
413 context_length_exceeded
429 rate_limit_error
499 client_closed_request
500 internal_server_error
503 service_unavailable
```

### 57.14.1 400 Bad Request

请求字段非法。

例如：

1. temperature 超范围。
2. messages 为空。
3. role 不合法。
4. stop 类型错误。

### 57.14.2 401 Unauthorized

缺少 API key 或 API key 无效。

### 57.14.3 403 Forbidden

认证成功，但没有权限访问该模型或 tenant。

### 57.14.4 413 Context Length Exceeded

prompt tokens + max_tokens 超过模型或服务限制。

这个错误很常见，应该清楚告诉用户限制是多少。

### 57.14.5 429 Rate Limit

租户或用户超过 QPS、并发、token rate 或配额限制。

### 57.14.6 503 Service Unavailable

服务过载、没有健康 worker、GPU 不可用、模型未加载。

429 和 503 要区分。

429 通常是用户额度或速率问题。

503 通常是服务端当前不可用或过载。

## 57.15 流式错误怎么处理

流式请求的错误分两类。

### 57.15.1 stream 开始前错误

如果还没发送任何 SSE event，可以直接返回普通 JSON 错误和对应 HTTP status。

例如参数校验失败：

```text
HTTP 400
{"error": ...}
```

### 57.15.2 stream 开始后错误

如果已经发送了部分 token，HTTP status 已经是 200。

这时不能再改成 500。

只能在 stream 里发送错误事件，或发送结束 chunk。

例如：

```text
data: {"error":{"message":"worker failed","type":"server_error","code":"worker_failed"}}

data: [DONE]

```

这也是为什么 worker failure 后，已经输出过 token 的请求不能透明重试。

API 协议层也已经进入 streaming 状态。

## 57.16 鉴权

API 层通常用 bearer token：

```text
Authorization: Bearer sk-xxx
```

鉴权要在任何昂贵操作之前完成。

顺序应该是：

```text
parse headers
  -> authenticate api key
  -> identify tenant/user
  -> check permission
  -> parse body
  -> validate request
  -> tokenize
  -> route
```

鉴权结果可以转换成上下文：

```python
@dataclass
class AuthContext:
    tenant_id: str
    user_id: str | None
    api_key_id: str
    allowed_models: set[str]
    qps_limit: int
    token_per_minute_limit: int
```

后续限流、审计和计费都依赖这个上下文。

不要把原始 API key 写进日志。

日志里最多记录 api_key_id 或 hash 后的标识。

## 57.17 权限模型

最小权限模型可以按 model 控制：

```python
def check_model_permission(auth: AuthContext, model: str) -> None:
    if model not in auth.allowed_models:
        raise PermissionDeniedError(f"no access to model: {model}")
```

更复杂的系统还会控制：

1. 最大上下文长度。
2. 最大 max_tokens。
3. 是否允许 streaming。
4. 是否允许 tool calls。
5. 是否允许某些 LoRA adapter。
6. 每分钟 token 配额。
7. 每日或每月预算。

权限检查应该在 tokenization 前尽量完成。

但 context length 这种检查必须等 tokenization 后才能准确判断。

## 57.18 限流维度

LLM API 的限流不能只看 QPS。

因为请求成本差异巨大。

常见限流维度包括：

1. requests per minute。
2. tokens per minute。
3. concurrent requests。
4. concurrent tokens。
5. prompt tokens per minute。
6. completion tokens per minute。
7. daily budget。
8. per-model quota。

一个请求可能 QPS 很低，但 tokens 很高。

例如用户每分钟只发 2 个请求，但每个 prompt 100k tokens。

如果只按 QPS 限流，系统仍然会被打爆。

## 57.19 token-based rate limit

可以用 token bucket 做 tokens per minute 限制。

请求进入前先估算成本：

```text
estimated_tokens = prompt_tokens + max_tokens
```

如果 token bucket 不够，就返回 429。

伪代码：

```python
def check_token_limit(auth, prompt_tokens: int, max_tokens: int):
    estimated = prompt_tokens + max_tokens
    bucket = token_buckets[auth.tenant_id]
    if not bucket.try_consume(estimated):
        raise RateLimitError("token rate limit exceeded")
```

请求完成后，可以根据实际 completion_tokens 做修正。

但 admission 时必须用估算值。

否则用户可以用很大的 max_tokens 占住资源。

## 57.20 concurrency limit

并发限制用于保护 engine 的运行资源。

例如：

```python
def acquire_concurrency_slot(auth):
    limiter = concurrency_limiters[auth.tenant_id]
    if not limiter.try_acquire():
        raise RateLimitError("too many concurrent requests")


def release_concurrency_slot(auth):
    concurrency_limiters[auth.tenant_id].release()
```

必须确保任何结束路径都会 release：

1. 正常完成。
2. 生成错误。
3. client disconnect。
4. timeout。
5. validation 后失败。

通常用 `try/finally` 包住请求生命周期。

并发 slot 泄漏会导致租户永久无法发请求。

## 57.21 admission control 和 rate limit 的区别

rate limit 是用户或租户维度的限制。

admission control 是系统容量维度的限制。

二者都可能拒绝请求，但含义不同。

```text
rate limit: 你这个 tenant 超额度了。
admission control: 系统现在接不下这个请求。
```

前者通常返回 429。

后者可以返回 503 或 429，取决于产品语义。

建议明确区分错误码：

```text
rate_limit_exceeded
engine_overloaded
no_healthy_worker
kv_capacity_exceeded
```

这样用户和运维都能知道该扩容、降速还是改请求参数。

## 57.22 request_id、trace_id 和 idempotency_key

### 57.22.1 request_id

每个请求都应该有服务端生成的 request_id。

它用于：

1. 日志串联。
2. metrics label。
3. 返回给用户报障。
4. router 映射。
5. cancel。

例如：

```text
chatcmpl-9f2a...
```

### 57.22.2 trace_id

trace_id 用于跨服务链路追踪。

一个用户请求可能经过：

```text
gateway -> api server -> tokenizer -> router -> worker -> output stream
```

所有日志都应该带 trace_id。

如果上游传了 trace header，可以继承。

否则服务端生成。

### 57.22.3 idempotency_key

idempotency key 用于处理客户端重试。

例如网络超时后，客户端不知道请求是否已经被服务端接收。

如果它带着同一个 idempotency key 再发一次，服务端可以避免重复执行。

但生成式请求的幂等并不简单。

如果第一次请求已经开始 streaming，第二次请求应该如何返回？

最小策略可以是：

1. 非流式请求：缓存最终结果一段时间。
2. 流式请求：只在 stream 未开始前允许幂等复用。
3. 已开始 stream 的请求：返回 conflict 或让客户端重新请求。

不要承诺做不到的强幂等。

## 57.23 日志和隐私

LLM API 日志很敏感。

prompt 里可能有用户隐私、代码、密钥或商业数据。

日志策略要明确：

1. 默认不要记录完整 prompt。
2. 可以记录 token counts、model、tenant、latency、finish_reason。
3. 如果需要 debug prompt，必须受权限控制。
4. API key 必须脱敏。
5. 错误日志不要包含过长 content。
6. trace 里避免记录原始用户文本。

一个安全的 access log 可以是：

```text
request_id=chatcmpl-abc tenant=t1 model=m1 prompt_tokens=128 completion_tokens=64 ttft_ms=320 e2e_ms=2800 status=200 finish=stop
```

这足够用于大部分运维分析。

## 57.24 API 层指标

除了 engine 指标，API 层还要有自己的指标：

```text
http_requests_total{endpoint,status}
http_request_duration_ms{endpoint}
auth_failure_total
permission_denied_total
rate_limit_reject_total{reason}
request_validation_error_total{param}
context_length_exceeded_total
stream_started_total
stream_client_disconnect_total
stream_error_after_start_total
tokens_prompt_total{model,tenant}
tokens_completion_total{model,tenant}
```

这些指标能区分：

1. 是 API 层拒绝多。
2. 还是 router/engine 过载。
3. 是用户请求非法。
4. 还是模型生成失败。
5. 是客户端主动断开。
6. 还是服务端 streaming 错误。

不要只看 engine tokens/s。

产品化服务必须看 API 层成功率和错误分布。

## 57.25 OpenAI-compatible 不等于完全相同

兼容 API 时，要诚实声明差异。

例如：

```text
Supported:
  model
  messages
  max_tokens
  temperature
  top_p
  stop
  stream
  seed

Ignored:
  user

Unsupported:
  logprobs
  response_format
  tools
  parallel_tool_calls
```

对 unsupported 字段，建议返回 400，而不是默默忽略。

对 ignored 字段，也要文档化。

默默忽略重要字段会制造错误预期。

比如用户传了 `response_format={"type":"json_object"}`，以为服务一定输出 JSON。

如果服务端实际忽略，就可能造成生产事故。

## 57.26 最小请求处理流程

把本章串起来，一个请求处理流程可以是：

```text
1. 生成 request_id / trace_id
2. 解析 Authorization header
3. 鉴权，得到 AuthContext
4. 解析 JSON body
5. 校验字段类型和范围
6. 检查 model 权限
7. 渲染 chat template
8. tokenization
9. 检查 context length
10. 检查 token rate limit
11. 获取 concurrency slot
12. 构造 EngineRequest
13. 全局 admission control
14. router 选择 worker
15. 非流式：等待完整结果并返回 JSON
16. 流式：建立 SSE，逐 chunk 返回 delta
17. 结束时记录 usage、metrics、日志
18. 任何路径都释放 concurrency slot
```

这个顺序不是唯一的。

但几个原则不变：

1. 便宜检查放前面。
2. 昂贵操作放后面。
3. 鉴权先于 tokenization。
4. context length 必须基于真实 token 数。
5. concurrency slot 必须可靠释放。
6. client disconnect 必须 cancel engine request。

## 57.27 常见错误

错误一：没有严格校验请求字段。

```text
结果：非法参数进入 engine，错误变成内部异常。
```

错误二：chat template 写错。

```text
结果：模型输出质量异常，但很难从 engine 指标看出来。
```

错误三：SSE chunk 少了空行。

```text
结果：客户端一直收不到增量事件。
```

错误四：stream 断开后没有 cancel。

```text
结果：僵尸请求继续占用 KV cache 和 decode slots。
```

错误五：只按 QPS 限流。

```text
结果：少量超长 prompt 请求仍然可以打爆系统。
```

错误六：并发 slot 没有 finally release。

```text
结果：租户并发额度泄漏，后续请求全部被拒绝。
```

错误七：把 API key 写进日志。

```text
结果：严重安全事故。
```

错误八：unsupported 字段被默默忽略。

```text
结果：用户以为功能生效，实际没有，业务语义出错。
```

错误九：stream 开始后还试图修改 HTTP status。

```text
结果：协议行为混乱，客户端无法可靠处理错误。
```

错误十：错误码只有 500。

```text
结果：用户、运维和自动重试逻辑都无法判断问题类型。
```

## 57.28 面试高频问题

问题一：为什么 LLM serving 要兼容 OpenAI API？

回答要点：OpenAI API 是事实标准之一，兼容后可以复用 SDK、agent framework、评测工具和业务接入代码。但兼容要明确支持、忽略和不支持的字段，不能看起来兼容实际语义不一致。

问题二：chat completions 请求进入 engine 前要做哪些步骤？

回答要点：先鉴权和权限检查，再解析和校验 JSON 字段，把 messages 用模型 chat template 渲染成 prompt，tokenize，检查 context length 和 rate limit，构造内部 EngineRequest，经过 admission control 和 router 选择 worker，最后进入 engine。

问题三：SSE streaming 有哪些坑？

回答要点：每个 event 要用 `data: ...\n\n` 结束，要设置 `text/event-stream` 和禁用缓存，要处理 proxy buffering，要检测 client disconnect 并 cancel engine request，delta 只发送增量文本，stream 开始后不能再修改 HTTP status。

问题四：LLM API 限流为什么不能只看 QPS？

回答要点：LLM 请求成本差异很大，prompt tokens、max_tokens、并发和模型类型都会影响 GPU 和 KV cache 压力。低 QPS 的超长 prompt 也能打爆系统，所以需要 requests per minute、tokens per minute、concurrent requests、context length、per-model quota 等多维限流。

问题五：rate limit 和 admission control 有什么区别？

回答要点：rate limit 是租户或用户维度，表示用户超过额度或速率，通常返回 429。admission control 是系统容量维度，表示当前 worker、KV cache 或队列无法承载，可能返回 503 或特定 overload 错误。二者都拒绝请求，但含义和处理方式不同。

## 57.29 标准回答模板

如果面试官问“你会怎么把 LLM engine 包装成 OpenAI-compatible API”，可以这样回答：

```text
我会把 API 层和 engine 层明确解耦。API 层提供 /v1/chat/completions、/v1/completions、/v1/models 等 endpoint，兼容常见字段，比如 model、messages、max_tokens、temperature、top_p、stop、stream。对于不支持的字段要明确返回 400 或文档说明，不能默默忽略重要语义。

请求进入 engine 前会经过一条清晰流水线：生成 request_id 和 trace_id，鉴权得到 tenant/user，上下文里检查模型权限，解析和校验 JSON，使用模型 chat template 把 messages 渲染成 prompt，tokenize，检查 context length，做 token-based rate limit 和 concurrency limit，然后构造内部 EngineRequest，交给 admission control 和 router。engine 内部不直接依赖原始 HTTP request。

流式响应我会用 SSE，Content-Type 是 text/event-stream，每个 chunk 用 data: JSON 加空行，最后发送 data: [DONE]。第一个 chunk 可以发送 assistant role，后续发送 delta.content，结束时发送 finish_reason。要特别处理 client disconnect，一旦断开就 cancel engine request，释放 KV cache 和并发 slot。stream 开始后的错误不能再改 HTTP status，只能通过 stream error event 或结束 chunk 表达。

限流上不会只看 QPS，因为 LLM 请求成本差异很大。我会同时做 requests per minute、tokens per minute、concurrent requests、max context length、per-model quota，并区分 rate limit 和系统 admission control。错误响应要有稳定的 type、code、param、message 和 request_id，方便 SDK、自动重试和运维排查。
```

## 57.30 API Compatibility 公式、错误门禁和可运行 demo

OpenAI-compatible API 层可以先把一次请求抽象成：

```math
a_i=(k_i,m_i,p_i,o_i,s_i,u_i,c_i,r_i)
```

其中 `k_i` 是 API key，`m_i` 是 model，`p_i` 是 prompt tokens，`o_i` 是 max output tokens，`s_i` 表示是否 stream，`u_i` 是 user / tenant，`c_i` 是 concurrency slot，`r_i` 是 request id。

上下文长度门禁：

```math
G_{\mathrm{ctx},i}=\mathbf{1}[p_i+o_i\le L_{m_i}^{\max}]
```

RPM 和 TPM 限流可以写成：

```math
R_u^{\mathrm{req}}(t)\le R_u^{\max}
```

```math
R_u^{\mathrm{tok}}(t)+p_i+o_i\le T_u^{\max}
```

并发门禁：

```math
C_u(t)<C_u^{\max}
```

Usage 统计：

```math
U_i=p_i+y_i
```

其中 `y_i` 是实际输出 token 数。

最终 API 层门禁：

```math
G_{\mathrm{api}}=G_{\mathrm{auth}}G_{\mathrm{schema}}G_{\mathrm{model}}G_{\mathrm{ctx}}G_{\mathrm{rate}}G_{\mathrm{stream}}G_{\mathrm{error}}G_{\mathrm{privacy}}
```

下面这个 0 依赖 demo 模拟一个最小 OpenAI-compatible Chat Completions API 子集：

```python
from dataclasses import dataclass
import json


@dataclass
class APITenant:
    tenant_id: str
    api_key: str
    allowed_models: set[str]
    rpm_limit: int
    tpm_limit: int
    concurrency_limit: int
    used_requests: int = 0
    used_tokens: int = 0
    active_requests: int = 0


@dataclass
class EngineRequest:
    request_id: str
    tenant_id: str
    model: str
    prompt: str
    prompt_tokens: int
    max_tokens: int
    temperature: float
    stream: bool


class ToyOpenAICompatibleAPI:
    def __init__(self):
        self.models = {"tiny-chat": {"max_context_tokens": 128}}
        self.tenants = {
            "sk-ok": APITenant("tenant-a", "sk-ok", {"tiny-chat"}, 10, 220, 2),
            "sk-rate": APITenant("tenant-rate", "sk-rate", {"tiny-chat"}, 1, 220, 2, 1, 0),
            "sk-busy": APITenant("tenant-busy", "sk-busy", {"tiny-chat"}, 10, 220, 1, 0, 0, 1),
            "sk-nomodel": APITenant("tenant-no-model", "sk-nomodel", set(), 10, 220, 2),
        }
        self.supported_fields = {
            "model",
            "messages",
            "max_tokens",
            "temperature",
            "top_p",
            "stop",
            "stream",
            "stream_options",
            "user",
        }
        self.metrics = {
            "auth_fail_total": 0,
            "bad_request_total": 0,
            "rate_limit_total": 0,
            "admission_reject_total": 0,
            "stream_response_total": 0,
            "non_stream_response_total": 0,
        }
        self.next_id = 1

    def new_request_id(self) -> str:
        request_id = f"req-{self.next_id:04d}"
        self.next_id += 1
        return request_id

    def error(self, status: int, request_id: str, message: str, err_type: str, code: str, param=None):
        if status == 401:
            self.metrics["auth_fail_total"] += 1
        elif status == 429:
            self.metrics["rate_limit_total"] += 1
        elif status == 503:
            self.metrics["admission_reject_total"] += 1
        else:
            self.metrics["bad_request_total"] += 1
        return status, {
            "error": {
                "message": message,
                "type": err_type,
                "param": param,
                "code": code,
            },
            "request_id": request_id,
        }

    def authenticate(self, headers: dict[str, str], request_id: str):
        auth = headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return None, self.error(401, request_id, "missing bearer token", "authentication_error", "missing_api_key")
        key = auth.removeprefix("Bearer ").strip()
        tenant = self.tenants.get(key)
        if tenant is None:
            return None, self.error(401, request_id, "invalid API key", "authentication_error", "invalid_api_key")
        return tenant, None

    def validate_schema(self, body: dict, request_id: str):
        unsupported = sorted(set(body) - self.supported_fields)
        if unsupported:
            return self.error(400, request_id, "unsupported request field", "invalid_request_error", "unsupported_field", unsupported[0])
        if body.get("model") not in self.models:
            return self.error(400, request_id, "unknown model", "invalid_request_error", "model_not_found", "model")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return self.error(400, request_id, "messages must be a non-empty list", "invalid_request_error", "invalid_messages", "messages")
        for index, message in enumerate(messages):
            if message.get("role") not in {"system", "user", "assistant"}:
                return self.error(400, request_id, "unsupported role", "invalid_request_error", "unsupported_role", f"messages.{index}.role")
            if not isinstance(message.get("content"), str) or not message["content"]:
                return self.error(400, request_id, "content must be a non-empty string", "invalid_request_error", "invalid_content", f"messages.{index}.content")
        max_tokens = body.get("max_tokens", 16)
        if not isinstance(max_tokens, int) or not 1 <= max_tokens <= 64:
            return self.error(400, request_id, "max_tokens out of supported range", "invalid_request_error", "invalid_max_tokens", "max_tokens")
        temperature = body.get("temperature", 1.0)
        if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
            return self.error(400, request_id, "temperature out of range", "invalid_request_error", "invalid_temperature", "temperature")
        return None

    def check_model_permission(self, tenant: APITenant, model: str, request_id: str):
        if model not in tenant.allowed_models:
            return self.error(403, request_id, "model not allowed for tenant", "permission_error", "model_not_allowed", "model")
        return None

    def render_chat_prompt(self, messages: list[dict]) -> str:
        parts = []
        for message in messages:
            parts.append(f"<|{message['role']}|>\n{message['content']}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    def count_tokens(self, text: str) -> int:
        return max(1, (len(text) + 3) // 4)

    def check_rate_limits(self, tenant: APITenant, needed_tokens: int, request_id: str):
        if tenant.used_requests + 1 > tenant.rpm_limit:
            return self.error(429, request_id, "rate limit exceeded", "rate_limit_error", "requests_per_minute")
        if tenant.used_tokens + needed_tokens > tenant.tpm_limit:
            return self.error(429, request_id, "token rate limit exceeded", "rate_limit_error", "tokens_per_minute")
        if tenant.active_requests >= tenant.concurrency_limit:
            return self.error(429, request_id, "concurrency limit exceeded", "rate_limit_error", "concurrent_requests")
        return None

    def build_engine_request(self, tenant: APITenant, body: dict, request_id: str):
        prompt = self.render_chat_prompt(body["messages"])
        prompt_tokens = self.count_tokens(prompt)
        max_tokens = body.get("max_tokens", 16)
        model_config = self.models[body["model"]]
        if prompt_tokens + max_tokens > model_config["max_context_tokens"]:
            return None, self.error(400, request_id, "context length exceeded", "invalid_request_error", "context_length_exceeded", "messages")
        limit_error = self.check_rate_limits(tenant, prompt_tokens + max_tokens, request_id)
        if limit_error is not None:
            return None, limit_error
        engine_request = EngineRequest(
            request_id=request_id,
            tenant_id=tenant.tenant_id,
            model=body["model"],
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            temperature=float(body.get("temperature", 1.0)),
            stream=bool(body.get("stream", False)),
        )
        return engine_request, None

    def fake_engine_tokens(self, max_tokens: int) -> list[str]:
        return ["KV", " cache", " stores", " keys"][:max_tokens]

    def usage(self, engine_request: EngineRequest, completion_tokens: int) -> dict:
        return {
            "prompt_tokens": engine_request.prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": engine_request.prompt_tokens + completion_tokens,
        }

    def non_stream_response(self, engine_request: EngineRequest, tokens: list[str]) -> dict:
        self.metrics["non_stream_response_total"] += 1
        return {
            "id": "chatcmpl-" + engine_request.request_id,
            "object": "chat.completion",
            "created": 1710000000,
            "model": engine_request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(tokens)},
                    "finish_reason": "stop",
                }
            ],
            "usage": self.usage(engine_request, len(tokens)),
            "request_id": engine_request.request_id,
        }

    def stream_response(self, engine_request: EngineRequest, tokens: list[str], include_usage: bool) -> list[str]:
        self.metrics["stream_response_total"] += 1
        chunks = []
        base = {
            "id": "chatcmpl-" + engine_request.request_id,
            "object": "chat.completion.chunk",
            "created": 1710000000,
            "model": engine_request.model,
        }
        role_chunk = dict(base)
        role_chunk["choices"] = [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
        chunks.append("data: " + json.dumps(role_chunk, ensure_ascii=False) + "\n\n")
        for token in tokens:
            token_chunk = dict(base)
            token_chunk["choices"] = [{"index": 0, "delta": {"content": token}, "finish_reason": None}]
            chunks.append("data: " + json.dumps(token_chunk, ensure_ascii=False) + "\n\n")
        finish_chunk = dict(base)
        finish_chunk["choices"] = [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        if include_usage:
            finish_chunk["usage"] = self.usage(engine_request, len(tokens))
        chunks.append("data: " + json.dumps(finish_chunk, ensure_ascii=False) + "\n\n")
        chunks.append("data: [DONE]\n\n")
        return chunks

    def handle_chat_completions(self, headers: dict[str, str], body: dict):
        request_id = self.new_request_id()
        tenant, auth_error = self.authenticate(headers, request_id)
        if auth_error is not None:
            return auth_error
        schema_error = self.validate_schema(body, request_id)
        if schema_error is not None:
            return schema_error
        permission_error = self.check_model_permission(tenant, body["model"], request_id)
        if permission_error is not None:
            return permission_error
        engine_request, build_error = self.build_engine_request(tenant, body, request_id)
        if build_error is not None:
            return build_error

        tenant.used_requests += 1
        tenant.used_tokens += engine_request.prompt_tokens + engine_request.max_tokens
        tenant.active_requests += 1
        try:
            tokens = self.fake_engine_tokens(engine_request.max_tokens)
            if engine_request.stream:
                include_usage = bool(body.get("stream_options", {}).get("include_usage", False))
                return 200, self.stream_response(engine_request, tokens, include_usage)
            return 200, self.non_stream_response(engine_request, tokens)
        finally:
            tenant.active_requests -= 1


api = ToyOpenAICompatibleAPI()
headers_ok = {"Authorization": "Bearer sk-ok"}
base_body = {
    "model": "tiny-chat",
    "messages": [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Explain KV cache."},
    ],
    "max_tokens": 4,
    "temperature": 0.7,
}

status_ok, non_stream = api.handle_chat_completions(headers_ok, dict(base_body))
stream_body = dict(base_body)
stream_body["stream"] = True
stream_body["stream_options"] = {"include_usage": True}
status_stream, stream_chunks = api.handle_chat_completions(headers_ok, stream_body)
status_auth, auth_error = api.handle_chat_completions({}, dict(base_body))
bad_field = dict(base_body)
bad_field["logprobs"] = True
status_bad_field, bad_field_error = api.handle_chat_completions(headers_ok, bad_field)
status_permission, permission_error = api.handle_chat_completions({"Authorization": "Bearer sk-nomodel"}, dict(base_body))
long_body = dict(base_body)
long_body["messages"] = [{"role": "user", "content": "x" * 700}]
status_context, context_error = api.handle_chat_completions(headers_ok, long_body)
status_rate, rate_error = api.handle_chat_completions({"Authorization": "Bearer sk-rate"}, dict(base_body))
status_busy, busy_error = api.handle_chat_completions({"Authorization": "Bearer sk-busy"}, dict(base_body))

parsed_stream_objects = []
for chunk in stream_chunks:
    payload = chunk.removeprefix("data: ").strip()
    if payload != "[DONE]":
        parsed_stream_objects.append(json.loads(payload))

summary = {
    "non_stream_status": status_ok,
    "non_stream_content": non_stream["choices"][0]["message"]["content"],
    "non_stream_usage": non_stream["usage"],
    "stream_status": status_stream,
    "stream_chunk_count": len(stream_chunks),
    "stream_first_delta": parsed_stream_objects[0]["choices"][0]["delta"],
    "stream_last_payload": stream_chunks[-1].strip(),
    "stream_usage": parsed_stream_objects[-1].get("usage"),
    "auth_error": (status_auth, auth_error["error"]["code"]),
    "bad_field_error": (status_bad_field, bad_field_error["error"]["code"], bad_field_error["error"]["param"]),
    "permission_error": (status_permission, permission_error["error"]["code"]),
    "context_error": (status_context, context_error["error"]["code"]),
    "rate_error": (status_rate, rate_error["error"]["code"]),
    "busy_error": (status_busy, busy_error["error"]["code"]),
    "metrics": api.metrics,
}

gates = {
    "auth_ready": summary["auth_error"] == (401, "missing_api_key"),
    "schema_ready": summary["bad_field_error"] == (400, "unsupported_field", "logprobs"),
    "model_permission_ready": summary["permission_error"] == (403, "model_not_allowed"),
    "context_ready": summary["context_error"] == (400, "context_length_exceeded"),
    "rate_limit_ready": summary["rate_error"] == (429, "requests_per_minute")
    and summary["busy_error"] == (429, "concurrent_requests"),
    "non_stream_ready": summary["non_stream_status"] == 200
    and summary["non_stream_content"] == "KV cache stores keys",
    "stream_ready": summary["stream_status"] == 200
    and summary["stream_first_delta"] == {"role": "assistant"}
    and summary["stream_last_payload"] == "data: [DONE]",
    "usage_ready": summary["stream_usage"] is not None
    and summary["non_stream_usage"]["completion_tokens"] == 4,
    "error_shape_ready": all(
        key in auth_error["error"] for key in ["message", "type", "param", "code"]
    )
    and "request_id" in auth_error,
    "privacy_ready": "sk-" not in json.dumps(summary, ensure_ascii=False),
}
gates["openai_compatible_api_gate"] = all(gates.values())

print("openai_compatible_api_summary=", summary)
print("openai_compatible_api_gates=", gates)
```

一次运行的核心输出类似：

```text
openai_compatible_api_summary= {'non_stream_status': 200, 'non_stream_content': 'KV cache stores keys', 'non_stream_usage': {'prompt_tokens': 16, 'completion_tokens': 4, 'total_tokens': 20}, 'stream_status': 200, 'stream_chunk_count': 7, 'stream_first_delta': {'role': 'assistant'}, 'stream_last_payload': 'data: [DONE]', 'stream_usage': {'prompt_tokens': 16, 'completion_tokens': 4, 'total_tokens': 20}, 'auth_error': (401, 'missing_api_key'), 'bad_field_error': (400, 'unsupported_field', 'logprobs'), 'permission_error': (403, 'model_not_allowed'), 'context_error': (400, 'context_length_exceeded'), 'rate_error': (429, 'requests_per_minute'), 'busy_error': (429, 'concurrent_requests'), 'metrics': {'auth_fail_total': 1, 'bad_request_total': 3, 'rate_limit_total': 2, 'admission_reject_total': 0, 'stream_response_total': 1, 'non_stream_response_total': 1}}
openai_compatible_api_gates= {'auth_ready': True, 'schema_ready': True, 'model_permission_ready': True, 'context_ready': True, 'rate_limit_ready': True, 'non_stream_ready': True, 'stream_ready': True, 'usage_ready': True, 'error_shape_ready': True, 'privacy_ready': True, 'openai_compatible_api_gate': True}
```

这个 demo 证明了几个关键点：

1. API key、schema、model permission、context length 和 rate limit 都在进入 engine 前完成。
2. 不支持字段 `logprobs` 被明确拒绝，不会静默忽略。
3. 非流式响应包含 `choices`、assistant message、`finish_reason`、usage 和 request id。
4. 流式响应是可解析的 SSE chunk，第一个 chunk 发送 assistant role，最后发送 `data: [DONE]`。
5. 错误响应结构稳定，包含 `message`、`type`、`param`、`code` 和 `request_id`。
6. RPM、并发和 context 约束的错误语义不同，不要混成一个 overload。
7. 摘要和日志里不能泄露 API key。

## 57.31 小练习

1. 实现 `/v1/models`，返回当前可用模型列表。
2. 定义 `ChatCompletionRequest` 和 `ChatCompletionResponse`。
3. 实现 chat messages 字段校验。
4. 使用模型 chat template 渲染 prompt。
5. 把 API request 转成内部 `EngineRequest`。
6. 实现非流式 chat completion 响应。
7. 实现 SSE streaming chunk 格式。
8. 在第一个 streaming chunk 里发送 assistant role。
9. 在最后发送 finish_reason 和 `[DONE]`。
10. 实现 client disconnect 后 cancel engine request。
11. 实现 OpenAI-style error response。
12. 增加 bearer token 鉴权。
13. 增加 per-tenant QPS limit。
14. 增加 token-based rate limit。
15. 增加 concurrency slot，并用 finally 保证释放。
16. 增加 request_id 和 trace_id 到日志。
17. 确保日志不打印 API key 和完整 prompt。
18. 写一个 stream benchmark，统计 TTFT 和 client disconnect。

## 57.32 本章总结

serving engine 的内部能力最终要通过 API 层交付给用户。

OpenAI-compatible API 能显著降低接入成本，因为大量 SDK、框架和工具已经支持它。

但兼容必须诚实。

支持、忽略和不支持的字段都要明确。

chat completions 的核心是把 messages 按模型 chat template 渲染成 prompt，再 tokenize 后交给 engine。

chat template 的细节会直接影响模型输出质量。

流式响应通常使用 SSE。

每个 chunk 都是 `data: JSON` 加空行，最后发送 `[DONE]`。

streaming 最大的工程点是 client disconnect、proxy buffering、增量 detokenization 和 stream 开始后的错误处理。

错误码、鉴权和限流是产品化 serving 的基础。

错误响应要稳定可解析。

鉴权要在昂贵操作前完成。

限流不能只看 QPS，还要看 tokens、并发、context length 和模型维度。

rate limit 和 admission control 要区分。

一个成熟 API 层还应该有 request_id、trace_id、usage、审计日志和隐私保护。

下一章可以继续讨论：如何把 serving 系统部署到生产环境，包括容器镜像、模型加载、健康检查、滚动升级和灰度发布。
