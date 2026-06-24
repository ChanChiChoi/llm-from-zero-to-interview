# 第 4 章 Prefill、Decode、KV Cache 和 Token Streaming

上一章我们从请求生命周期出发，走完了接入、排队、调度、执行、输出和资源释放。本章把生命周期中最关键的四个概念单独拎出来：Prefill、Decode、KV Cache 和 Token Streaming。

这四个概念决定了 LLM serving 的基本性能边界。后面无论讲 vLLM 的 PagedAttention、SGLang 的 RadixAttention，还是 PD 分离、chunked prefill、多级 KV Cache，本质上都是围绕它们做优化。

一句话概括：

> Prefill 决定首 token 等多久，Decode 决定后续 token 多快，KV Cache 决定显存和并发，Token Streaming 决定用户如何感知生成过程。

## 4.0 本讲资料边界与第二轮精修口径

本章讲的是 Prefill、Decode、KV Cache 和 Token Streaming 的稳定机制，不是 SGLang 源码导读。第二轮精修时，本章按下面口径处理：

1. KV Cache 的基础定义参考 Hugging Face Transformers cache 文档：cache 用来复用自回归生成中的历史 key / value，避免每轮重复计算历史 token。
2. request lifecycle、scheduler、KV cache manager 和 worker 的模块边界参考 vLLM、TGI 和 Triton 等公开 serving 抽象，但本章不绑定具体 runtime 的内部字段名。
3. PagedAttention、RadixAttention、prefix cache、chunked prefill、PD 分离和多级 KV cache 只在本章点到为止，后续章节会单独展开。
4. 本章公式用 toy step 和 MiB 估算解释资源趋势，不把数值当作真实 benchmark。
5. demo 只验证 prefill / decode / KV / streaming 四者如何串起来，不依赖真实 GPU、真实 tokenizer 或模型权重。

所以，本章的目标是让你能用公式和最小代码解释：为什么长 prompt 拉高 TTFT，为什么 decode 影响 TPOT，为什么 KV cache 会卡住并发，以及为什么 streaming 必须和资源清理绑定。

## 4.1 为什么这四个概念最重要

大模型推理看起来像一次 `generate()` 调用，但内部不是一次性计算完整答案。

它更像下面这条流水线：

```text
prompt tokens
  -> prefill
  -> KV Cache
  -> first token
  -> decode one token
  -> stream one chunk
  -> decode next token
  -> stream next chunk
  -> ...
  -> finish
```

如果你只知道“模型输入 prompt，输出答案”，就很难理解推理性能为什么复杂。

如果你知道 prefill、decode、KV cache 和 streaming，就能解释很多线上现象：

1. 为什么长 prompt 首 token 慢。
2. 为什么输出过程中有时一卡一卡的。
3. 为什么显存够放模型权重，却撑不住高并发。
4. 为什么 batch 越大不一定体验越好。
5. 为什么 PD 分离要把 prefill 和 decode 放到不同资源池。

## 4.2 Prefill 是什么

Prefill 是处理输入 prompt 的阶段。

如果用户输入了 2000 个 token，模型需要先对这 2000 个 token 做一次前向计算，得到每一层的 KV Cache，并产生下一个 token 的 logits。

简化流程：

```text
input_ids[0:prompt_len]
  -> embedding
  -> transformer layers
  -> logits at last position
  -> sample first output token
```

Prefill 的输入长度通常等于 prompt 长度，计算量和 prompt token 数强相关。

对用户来说，prefill 最直接影响的是 TTFT，也就是 time to first token。

如果 TTFT 很高，用户会觉得服务“半天没反应”。即使后面 decode 很快，体验也已经被首 token 等待拖差了。

## 4.3 Prefill 的计算特点

Prefill 阶段通常有这些特点：

1. 一次处理多个 prompt token。
2. 矩阵乘法规模较大。
3. 更容易利用 GPU 算力。
4. 计算量随 prompt 长度上升。
5. 会产生大量 KV Cache 写入。

从资源角度看，prefill 更偏 compute-bound，也就是更依赖 GPU 计算能力。当然这不是绝对的，长上下文、注意力实现和 cache 写入也会带来内存压力。

Prefill 的主要优化方向包括：

1. 提升 batched prefill 的吞吐。
2. 控制单个长 prompt 对系统的冲击。
3. 使用 chunked prefill 把长 prompt 拆成多段。
4. 复用 prefix cache，避免重复 prefill。
5. 在 PD 分离架构中把 prefill 放到更适合大计算的资源池。

## 4.4 Decode 是什么

Decode 是逐 token 生成阶段。

prefill 之后，模型已经有了 prompt 的 KV Cache。此后每一轮只输入上一个 token，并读取历史 KV Cache，生成下一个 token。

简化流程：

```text
last_token + past KV Cache
  -> transformer layers
  -> logits
  -> sample next token
  -> append token
  -> update KV Cache
```

Decode 会重复很多轮。用户看到的答案就是 decode 一轮一轮积累出来的。

如果生成 500 个 token，decode 至少要进行 500 轮模型前向。每轮可能很小，但重复次数很多。

因此 decode 对 TPOT 影响很大，也就是 time per output token。

## 4.5 Decode 的资源特点

Decode 阶段通常有这些特点：

1. 每个请求每轮只处理一个新 token。
2. 单轮计算量比 prefill 小。
3. 需要频繁读取 KV Cache。
4. 对内存带宽和调度开销敏感。
5. batch 中请求数量会动态变化。

Decode 常被认为更偏 memory-bound，因为模型需要不断读取历史 KV Cache。上下文越长，KV Cache 越大，读取压力越明显。

Decode 的主要优化方向包括：

1. continuous batching，提高每轮执行的有效请求数。
2. 高效 KV Cache 布局，减少碎片和无效读取。
3. kernel 优化，降低小 batch decode 的开销。
4. speculative decoding，减少主模型 decode 轮数。
5. 在 PD 分离架构中使用适合 decode 的资源池。

## 4.6 Prefill 和 Decode 的差异

可以用一张表总结：

| 维度 | Prefill | Decode |
| --- | --- | --- |
| 输入 | prompt 中的多个 token | 上一轮生成的 token |
| 输出 | 第一个输出 token 和 KV Cache | 下一个输出 token 和更新后的 KV Cache |
| 频率 | 每个请求通常一次 | 每个输出 token 一次 |
| 主要影响 | TTFT | TPOT、持续吞吐 |
| 资源倾向 | 更偏大计算 | 更偏 cache 读取和调度 |
| 优化重点 | 长 prompt、prefix 复用、chunking | continuous batching、cache 管理、decode kernel |

这个差异非常重要。很多系统设计问题都来自 prefill 和 decode 混在一起带来的冲突。

例如，一个长 prompt prefill 可能占用大量计算资源，导致正在 decode 的请求输出卡顿。反过来，如果系统一直优先 decode，新请求的 TTFT 又会变差。

Scheduler 要做的，就是在 prefill 和 decode 之间不断做权衡。

## 4.7 KV Cache 是什么

Transformer 自注意力中，每个 token 会产生 key 和 value。生成下一个 token 时，模型需要关注历史 token。如果每轮都重新计算历史 token 的 key/value，成本会非常高。

KV Cache 的作用就是保存历史 token 的 key/value，让 decode 阶段只计算新 token。

没有 KV Cache：

```text
第 1 轮：计算 prompt + token1
第 2 轮：重新计算 prompt + token1 + token2
第 3 轮：重新计算 prompt + token1 + token2 + token3
```

有 KV Cache：

```text
prefill：计算 prompt，保存 KV
decode 1：只计算 token1，读取历史 KV
decode 2：只计算 token2，读取历史 KV
decode 3：只计算 token3，读取历史 KV
```

KV Cache 是自回归推理能高效运行的关键。

## 4.8 KV Cache 为什么占显存

KV Cache 保存的是每层、每个历史 token、每个 attention head 的 key/value。

粗略看，它和这些因素正相关：

1. 层数。
2. hidden size 或 head dimension。
3. attention head 数量。
4. batch 中请求数。
5. 每个请求的上下文长度。
6. cache 的数据类型，比如 FP16、BF16、FP8。

可以记一个直觉：模型权重是相对固定的，KV Cache 会随着并发和上下文长度动态增长。

这解释了一个常见现象：模型权重明明能放进 GPU，为什么一上并发就 OOM？原因往往不是权重变大了，而是 KV Cache 占满了剩余显存。

## 4.9 KV Cache 管理的难点

单请求时，KV Cache 可以简单地随请求增长。多请求 serving 中，它变成了内存管理问题。

难点包括：

1. 分配：新请求进入时，如何分配 cache 空间。
2. 扩容：decode 每生成一个 token，都要追加 cache。
3. 释放：请求结束后必须归还 cache。
4. 碎片：不同请求长度不同，连续分配容易浪费。
5. 抢占：资源不足时是否暂停或重算部分请求。
6. 复用：相同 prefix 能否共享 cache。
7. 迁移：PD 分离或多级缓存中，cache 是否要跨设备移动。

vLLM 的 PagedAttention 就是为了解决 KV Cache 连续分配和碎片问题。它把 KV Cache 分成块，类似操作系统里的分页内存，让请求的逻辑 token 序列可以映射到非连续物理块。

## 4.10 Token Streaming 是什么

Token Streaming 是边生成边返回。

非流式输出是：

```text
生成完整答案 -> 一次性返回
```

流式输出是：

```text
生成 token1 -> 返回片段1
生成 token2 -> 返回片段2
生成 token3 -> 返回片段3
```

对用户体验来说，streaming 非常重要。哪怕总耗时一样，如果用户能很快看到第一个 token，并持续看到内容增长，就会感觉系统更快。

Streaming 和 TTFT、TPOT 都有关：

1. TTFT 决定多久看到第一段输出。
2. TPOT 决定后续输出是否顺滑。
3. 抖动决定输出是否一卡一卡。

## 4.11 Streaming 的工程细节

Token Streaming 不是简单 `print(token)`。

工程上要处理：

1. token 到文本的增量解码。
2. Unicode 字符边界。
3. stop sequence 跨 token 匹配。
4. SSE、WebSocket 或 gRPC stream。
5. 客户端断开检测。
6. backpressure，也就是客户端读得慢。
7. 错误和结束事件。
8. usage 统计和 finish reason。

一个典型流式事件可能包括：

```json
{
  "id": "request-123",
  "delta": "Cache",
  "finish_reason": null
}
```

结束时再返回：

```json
{
  "id": "request-123",
  "delta": "",
  "finish_reason": "stop"
}
```

如果客户端中途断开，engine 必须把请求标记为 aborted，并释放 KV Cache。否则看起来用户已经走了，GPU 还在继续为这个请求生成。

## 4.12 四者如何串起来

把四个概念放到同一条链路中：

```text
1. Prompt tokenization
2. Prefill prompt tokens
3. 建立 KV Cache
4. 采样第一个 token
5. Streaming 返回第一个文本片段
6. Decode 下一 token
7. 更新 KV Cache
8. Streaming 返回下一文本片段
9. 重复 6-8
10. 请求结束，释放 KV Cache
```

可以看到，KV Cache 贯穿 prefill 和 decode，streaming 则把 decode 结果及时暴露给客户端。

Prefill、decode、cache、streaming 不是四个孤立模块，而是一条请求执行链路中的不同职责。

## 4.13 对 Scheduler 的影响

Scheduler 必须理解这些阶段，否则无法做出合理决策。

它要知道：

1. 新请求 prefill 会消耗多少 token budget。
2. running 请求 decode 需要多少 cache。
3. 当前 KV Cache 是否足够容纳新请求。
4. 哪些请求已经接近结束。
5. 哪些请求启用了 streaming，需要输出平滑。
6. 长 prefill 是否应该拆成 chunk。

一个简单但实用的调度目标是：

1. 不让新请求饿死，控制 TTFT。
2. 不让老请求卡住，控制 TPOT。
3. 不超出 KV Cache 显存预算。
4. 尽量让 GPU 每轮都有足够有效工作。

这也是为什么推理框架的 scheduler 比普通任务队列复杂得多。

## 4.14 常见性能现象解释

现象一：prompt 变长后，首 token 明显变慢。

原因通常是 prefill token 数增加，prefill 延迟上升。

现象二：输出前面快，后面越来越慢。

原因可能是上下文越来越长，KV Cache 读取压力增大，也可能是 batch 内其他请求影响。

现象三：并发上来后显存爆掉。

原因通常是 KV Cache 随并发和上下文长度增长。

现象四：总耗时没变，但用户觉得更快了。

原因可能是 TTFT 降低或 streaming 更平滑。

现象五：GPU 利用率不低，但用户输出一卡一卡。

原因可能是 decode scheduling 不稳定，或者流式输出通道阻塞。

## 4.15 常见误区

误区一：Prefill 和 Decode 只是代码里的两个阶段，不影响架构。

实际上它们资源画像不同，直接影响 scheduler、PD 分离、扩缩容和性能调优。

误区二：KV Cache 只是优化项，不是必须关注的核心。

没有 KV Cache，自回归推理成本会非常高；KV Cache 管理不好，高并发 serving 会很快遇到显存瓶颈。

误区三：Streaming 只影响前端体验。

Streaming 还影响连接管理、取消、backpressure、finish reason 和资源释放。

误区四：TTFT 只和模型大小有关。

TTFT 还和排队、prompt 长度、prefill batch、prefix cache、调度策略和系统负载有关。

误区五：TPOT 只和 GPU 算力有关。

TPOT 还受 KV Cache 读取、batch 动态变化、kernel、调度开销和输出通道影响。

## 4.16 阶段指标、KV 公式和可运行 demo

可以把一个请求的阶段画像写成：

```math
P_i=(N_{\mathrm{prompt},i},N_{\mathrm{out},i},S_{\mathrm{prefill},i},S_{\mathrm{decode},i},K_i,Q_i,F_i)
```

其中 `N_prompt` 是输入 token 数，`N_out` 是输出 token 数，`S_prefill` 是 prefill step，`S_decode` 是 decode step，`K_i` 是 KV cache 占用，`Q_i` 是 streaming backlog，`F_i` 是 finish reason。

在 toy engine 里，可以把 prefill 近似成按 token 吞吐切块：

```math
S_{\mathrm{prefill},i}=\left\lceil \frac{N_{\mathrm{prompt},i}}{R_{\mathrm{prefill}}}\right\rceil
```

decode 每轮生成一个 token 时：

```math
S_{\mathrm{decode},i}=N_{\mathrm{out},i}
```

因此 toy TTFT 和 TPOT 可以写成：

```math
S_{\mathrm{ttft},i}=S_{\mathrm{queue},i}+S_{\mathrm{prefill},i}+1
```

```math
S_{\mathrm{tpot},i}=\frac{S_{\mathrm{decode},i}}{\max(N_{\mathrm{out},i}-1,1)}
```

KV cache 显存估算要包含 K 和 V 两份状态：

```math
M_{\mathrm{kv},i}=2L(N_{\mathrm{prompt},i}+N_{\mathrm{generated},i})H_{\mathrm{kv}}D_hb
```

其中 `L` 是层数，`H_kv` 是 KV head 数，`D_h` 是 head dimension，`b` 是每个元素的字节数。总 KV pressure 可以写成：

```math
P_{\mathrm{kv}}=\frac{\sum_i M_{\mathrm{kv},i}}{B_{\mathrm{kv}}}
```

streaming backlog 可以用增量事件描述：

```math
Q_{\mathrm{stream}}(t)=Q_{\mathrm{stream}}(t-1)+C_{\mathrm{emit}}(t)-C_{\mathrm{flush}}(t)
```

其中 `C_emit` 是 decode 生成的 chunk 数，`C_flush` 是客户端已经消费的 chunk 数。最终阶段门禁可以写成：

```math
G_{\mathrm{phase}}=G_{\mathrm{prefill}}G_{\mathrm{decode}}G_{\mathrm{kv}}G_{\mathrm{stream}}G_{\mathrm{cleanup}}
```

下面这个 demo 用 0 依赖代码把四个阶段串起来。它不会模拟真实 attention，只用 token 数和配置估算阶段指标，让你能看见长 prompt、输出长度、KV cache 和 streaming backlog 如何共同影响 serving。

```python
from dataclasses import dataclass, field
from math import ceil


@dataclass
class ToyRequest:
    request_id: str
    prompt_tokens: int
    output_tokens: int
    client_read_every: int = 1
    stream_buffer_capacity: int = 4
    stream_chunks: list = field(default_factory=list)
    flushed_chunks: list = field(default_factory=list)


class MiniPrefillDecodeKVStreamingAudit:
    def __init__(
        self,
        layers=32,
        kv_heads=8,
        head_dim=128,
        dtype_bytes=2,
        prefill_tokens_per_step=1024,
        kv_budget_mib=2048,
    ):
        self.layers = layers
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.dtype_bytes = dtype_bytes
        self.prefill_tokens_per_step = prefill_tokens_per_step
        self.kv_budget_mib = kv_budget_mib
        self.events = []

    def kv_mib(self, token_count):
        bytes_used = 2 * self.layers * token_count * self.kv_heads * self.head_dim * self.dtype_bytes
        return bytes_used / (1024 * 1024)

    def run_request(self, request):
        prefill_steps = ceil(request.prompt_tokens / self.prefill_tokens_per_step)
        decode_steps = request.output_tokens
        ttft_steps = prefill_steps + 1
        tpot_steps = decode_steps / max(request.output_tokens - 1, 1)
        max_stream_backlog = 0
        backpressure_events = 0

        self.events.append((request.request_id, "prefill", request.prompt_tokens, prefill_steps))
        for step in range(1, request.output_tokens + 1):
            chunk = f"{request.request_id}_tok{step}"
            request.stream_chunks.append(chunk)
            self.events.append((request.request_id, "decode", step, chunk))
            if len(request.stream_chunks) > request.stream_buffer_capacity:
                backpressure_events += 1
                request.flushed_chunks.append(request.stream_chunks.pop(0))
            if step % request.client_read_every == 0:
                request.flushed_chunks.extend(request.stream_chunks)
                request.stream_chunks.clear()
            max_stream_backlog = max(max_stream_backlog, len(request.stream_chunks))

        request.flushed_chunks.extend(request.stream_chunks)
        request.stream_chunks.clear()
        kv_tokens = request.prompt_tokens + request.output_tokens
        return {
            "prompt_tokens": request.prompt_tokens,
            "output_tokens": request.output_tokens,
            "prefill_steps": prefill_steps,
            "decode_steps": decode_steps,
            "ttft_steps": ttft_steps,
            "tpot_steps": round(tpot_steps, 3),
            "kv_mib": round(self.kv_mib(kv_tokens), 3),
            "stream_chunks": len(request.flushed_chunks),
            "max_stream_backlog": max_stream_backlog,
            "backpressure_events": backpressure_events,
            "finish_reason": "length",
        }

    def run(self, requests):
        per_request = {}
        for request in requests:
            per_request[request.request_id] = self.run_request(request)

        total_prompt = sum(item["prompt_tokens"] for item in per_request.values())
        total_output = sum(item["output_tokens"] for item in per_request.values())
        total_kv_mib = round(sum(item["kv_mib"] for item in per_request.values()), 3)
        max_ttft = max(item["ttft_steps"] for item in per_request.values())
        max_tpot = max(item["tpot_steps"] for item in per_request.values())
        return {
            "per_request": per_request,
            "metrics": {
                "total_prompt_tokens": total_prompt,
                "total_output_tokens": total_output,
                "sum_prefill_steps": sum(item["prefill_steps"] for item in per_request.values()),
                "sum_decode_steps": sum(item["decode_steps"] for item in per_request.values()),
                "p95_ttft_steps": max_ttft,
                "p95_tpot_steps": max_tpot,
                "total_kv_mib": total_kv_mib,
                "kv_pressure": round(total_kv_mib / self.kv_budget_mib, 3),
                "stream_events": sum(item["stream_chunks"] for item in per_request.values()),
                "backpressure_events": sum(item["backpressure_events"] for item in per_request.values()),
                "event_tail": self.events[-6:],
            },
        }


def audit_phases(report, kv_pressure_limit=0.8):
    gates = {
        "prefill_accounting": all(item["prefill_steps"] > 0 for item in report["per_request"].values()),
        "decode_accounting": all(item["decode_steps"] == item["output_tokens"] for item in report["per_request"].values()),
        "kv_budget": report["metrics"]["kv_pressure"] <= kv_pressure_limit,
        "streaming_chunks": report["metrics"]["stream_events"] == report["metrics"]["total_output_tokens"],
        "cleanup_finish_reason": all(item["finish_reason"] for item in report["per_request"].values()),
    }
    return {
        "gates": gates,
        "phase_gate": all(gates.values()),
    }


requests = [
    ToyRequest("short_chat", prompt_tokens=512, output_tokens=4),
    ToyRequest("long_context", prompt_tokens=4096, output_tokens=3, client_read_every=2),
    ToyRequest("code_gen", prompt_tokens=2048, output_tokens=5),
]

audit_runner = MiniPrefillDecodeKVStreamingAudit()
report = audit_runner.run(requests)
audit = audit_phases(report)

print("per_request=", report["per_request"])
print("metrics=", report["metrics"])
print("audit=", audit)
```

这段代码的重点不是数值本身，而是验证四个阶段的因果链：prompt 越长，prefill steps 和 TTFT 越高；输出越长，decode steps 越多；活跃上下文越长，KV MiB 和 KV pressure 越高；streaming chunks 必须和输出 token、finish reason、cleanup 一起统计。

## 4.17 面试追问

1. Prefill 和 Decode 的区别是什么？
2. 为什么 Prefill 主要影响 TTFT，Decode 主要影响 TPOT？
3. KV Cache 为什么能加速自回归生成？
4. KV Cache 为什么会成为高并发 serving 的显存瓶颈？
5. Streaming 输出有哪些工程难点？
6. 为什么长 prompt 会拖慢首 token？
7. 为什么输出过程中可能一卡一卡？
8. 如果显存充足但 TPOT 很差，你会从哪些方向排查？

参考回答思路：

1. 先定义 prefill、decode、KV cache 和 streaming。
2. 再说明资源画像：prefill 偏大计算，decode 更受 KV Cache 和调度影响。
3. 然后关联指标：prefill 对 TTFT 关键，decode 对 TPOT 关键，KV Cache 对并发关键，streaming 对感知体验关键。
4. 最后补工程 trade-off：scheduler 要在 TTFT、TPOT、吞吐、显存和公平性之间平衡。

## 4.18 小练习

1. 画出一个请求从 prefill 到 decode 再到 streaming 的流程图。
2. 假设 prompt 长度从 1k 增加到 8k，TTFT 可能发生什么变化？为什么？
3. 假设并发从 10 增加到 100，KV Cache 会带来什么压力？
4. 写一段伪代码，表示 decode 一轮后把 token 推送到 stream。
5. 解释为什么客户端断开后必须释放 KV Cache。

## 4.19 本章小结

本章讲清了 LLM serving 中最重要的四个执行概念。

Prefill 处理输入 prompt，决定首 token 等待；Decode 逐 token 生成，决定持续输出速度；KV Cache 复用历史 key/value，决定显存和并发边界；Token Streaming 把生成结果增量返回给用户，决定感知体验和连接生命周期。

下一章我们会把这些概念转成可量化指标，系统讲 TTFT、TPOT、吞吐、并发、显存和成本之间的关系。
