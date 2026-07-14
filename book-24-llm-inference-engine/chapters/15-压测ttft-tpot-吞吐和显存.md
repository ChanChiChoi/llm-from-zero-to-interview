# 第 15 章 压测 TTFT、TPOT、吞吐和显存

第二部分前面几章从 0 搭了一个最小推理框架：tokenizer、model wrapper、generate loop、sampling、KV Cache、batched prefill、batched decode、scheduler、streaming 和 HTTP API。本章做阶段性收尾：如何压测这个系统，如何看 TTFT、TPOT、吞吐和显存。

做推理框架不能只说“能跑”。能跑只是第一步。真正的工程问题是：首 token 多久返回？后续 token 多快？并发上来后吞吐如何？显存为什么涨？瓶颈在 prefill、decode、scheduler，还是 streaming？

一句话概括：

> 压测的目的不是得到一个漂亮 QPS，而是把请求生命周期拆成可观测指标，定位 serving engine 的真实性能瓶颈。

## 15.0 本讲资料边界与第二轮精修口径

本讲只做教学版 LLM serving 压测审计。它覆盖 TTFT、TPOT / inter-token latency、E2E latency、input / output tokens/s、queue length、active requests、KV cache 显存、allocated / reserved memory、workload 描述、瓶颈归因和最小可运行 demo，但不实现真实 GPU benchmark、压测平台、分布式压测、Prometheus / OpenTelemetry 接入、真实模型加载、生产级 autoscaling、租户限流或完整容量规划。

资料校准口径：

1. vLLM 公开 metrics 文档把 time to first token、time per output token、end-to-end request latency、request queue / prefill / decode 时间、running / waiting / swapped 请求数和 KV cache usage 作为 serving 观测指标。
2. NVIDIA GenAI-Perf 面向生成式模型的压测报告包含 request latency、time to first token、inter token latency、input / output sequence length、request throughput 和 output token throughput 等指标。
3. PyTorch CUDA memory 文档区分 `memory_allocated()` 与 `memory_reserved()`；reserved memory 可能来自 caching allocator，不等同于仍被张量实际占用。
4. 本章 demo 用手工 toy trace 模拟请求到达、排队、prefill、decode、finish 和显存采样，不绑定 vLLM、Triton、TGI、SGLang 或某个 GPU。

参考资料：

1. vLLM metrics：<https://docs.vllm.ai/en/latest/design/metrics.html>
2. NVIDIA GenAI-Perf：<https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_benchmark/genai_perf.html>
3. PyTorch CUDA memory management：<https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management>

## 15.1 压测要回答什么问题

LLM serving 压测至少要回答：

1. 单请求延迟是多少。
2. 并发增加后 TTFT 如何变化。
3. TPOT 是否稳定。
4. 每秒能处理多少 input tokens。
5. 每秒能生成多少 output tokens。
6. waiting queue 是否增长。
7. GPU 显存是否接近上限。
8. KV Cache 占用是否随并发线性增长。
9. p95/p99 是否出现长尾。
10. 错误、超时、取消是否增多。

如果压测只输出一个 QPS，就没有抓住 LLM serving 的重点。

## 15.2 指标定义回顾

先明确几个核心指标。设请求 `i` 的到达时间为 `a_i`，进入 prefill 的时间为 `p_i`，第一个对客户端可见的 token 时间为 `f_i`，第 `j` 个输出 token 的时间为 `s_{i,j}`，请求结束时间为 `e_i`。输入 token 数为 `x_i`，输出 token 数为 `y_i`。

TTFT：

```math
T_{\mathrm{ttft},i}=f_i-a_i
```

它衡量用户等多久看到第一个 token，通常包含排队、tokenization、prefill、首 token 采样和首次 flush。

TPOT：

```math
T_{\mathrm{tpot},i}=\frac{s_{i,y_i}-s_{i,1}}{y_i-1},\quad y_i>1
```

如果 `y_i=1`，TPOT 可以记为 0 或不纳入 TPOT 分位数。报告里必须写清楚定义。

E2E latency：

```math
T_{\mathrm{e2e},i}=e_i-a_i
```

queue wait：

```math
W_i=p_i-a_i
```

压测窗口从 `t_0` 到 `t_1`，窗口长度为：

```math
D=t_1-t_0
```

输入吞吐、输出吞吐和总 token 吞吐：

```math
Q_{\mathrm{in}}=\frac{\sum_i x_i}{D},\quad
Q_{\mathrm{out}}=\frac{\sum_i y_i}{D},\quad
Q_{\mathrm{tok}}=\frac{\sum_i(x_i+y_i)}{D}
```

KV Cache 显存粗估：

```math
M_{\mathrm{kv}}=2L H_{\mathrm{kv}}D_h b\sum_i T_i
```

其中 `L` 是层数，`H_{\mathrm{kv}}` 是 KV head 数，`D_h` 是 head dim，`b` 是每个元素字节数，`T_i` 是请求 `i` 当前占用的 cache token 数。

总显存可以粗略拆成：

```math
M_{\mathrm{total}}=M_{\mathrm{weight}}+M_{\mathrm{kv}}+M_{\mathrm{work}}+M_{\mathrm{runtime}}
```

其中 `M_{\mathrm{work}}` 是激活、临时 buffer、CUDA graph 或 kernel workspace 等工作区，`M_{\mathrm{runtime}}` 是 runtime 和 allocator 额外开销。

压测门禁可以写成：

```math
G_{\mathrm{bench}}=G_{\mathrm{trace}}G_{\mathrm{latency}}G_{\mathrm{throughput}}G_{\mathrm{kv}}G_{\mathrm{cleanup}}G_{\mathrm{bottleneck}}
```

这些指标要分开看，不能互相替代。一个系统可能 `Q_{\mathrm{out}}` 很高，但 `T_{\mathrm{ttft}}` p99 很差；也可能延迟达标，但 `M_{\mathrm{kv}}` 已经接近 OOM。

## 15.3 在 RequestState 中记录时间

先给 request 增加时间字段。

```python
class RequestState:
    def __init__(self, request_id, prompt, max_new_tokens=128):
        self.request_id = request_id
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens

        self.arrival_time = now()
        self.prefill_start_time = None
        self.first_token_time = None
        self.finished_time = None

        self.decode_token_timestamps = []
        self.input_token_count = 0
        self.output_ids = []
        self.finish_reason = None
```

然后定义：

```python
def ttft(request):
    return request.first_token_time - request.arrival_time


def total_latency(request):
    return request.finished_time - request.arrival_time


def tpot(request):
    if len(request.decode_token_timestamps) <= 1:
        return 0
    return (request.decode_token_timestamps[-1] - request.decode_token_timestamps[0]) / (len(request.decode_token_timestamps) - 1)
```

注意，TPOT 的定义在不同系统里可能略有差异。有的从首 token 后开始算，有的按所有 output token 间隔算。压测报告里要写清楚定义。

## 15.4 在 engine 中埋点

请求入队时：

```python
request.arrival_time = now()
```

prefill 开始时：

```python
request.prefill_start_time = now()
```

首 token 生成时：

```python
request.first_token_time = now()
request.decode_token_timestamps.append(request.first_token_time)
```

每个 decode token 生成时：

```python
request.decode_token_timestamps.append(now())
```

请求结束时：

```python
request.finished_time = now()
```

这些埋点要放在生命周期的正确位置，否则指标会偏。

例如，如果 first_token_time 记录在 token 生成后但 streaming flush 前，那么它是 engine 首 token 时间，不是用户实际收到首 token 的时间。生产系统最好区分 engine TTFT 和 client-observed TTFT。

## 15.5 统计分位数

平均值不够，要看 p50、p95、p99。

```python
def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    k = int((len(values) - 1) * p / 100)
    return values[k]
```

统计：

```python
ttfts = [ttft(r) for r in finished_requests]
tpots = [tpot(r) for r in finished_requests]
latencies = [total_latency(r) for r in finished_requests]

report = {
    "ttft_p50": percentile(ttfts, 50),
    "ttft_p95": percentile(ttfts, 95),
    "ttft_p99": percentile(ttfts, 99),
    "tpot_p50": percentile(tpots, 50),
    "tpot_p95": percentile(tpots, 95),
    "latency_p95": percentile(latencies, 95),
}
```

真实实现可以用更准确的分位数算法或指标系统。

## 15.6 统计 tokens/s

压测开始和结束时间：

```python
duration = end_time - start_time
```

输入 token：

```python
input_tokens = sum(r.input_token_count for r in finished_requests)
```

输出 token：

```python
output_tokens = sum(len(r.output_ids) for r in finished_requests)
```

吞吐：

```python
input_tokens_per_second = input_tokens / duration
output_tokens_per_second = output_tokens / duration
total_tokens_per_second = (input_tokens + output_tokens) / duration
```

LLM serving 报告里，output tokens/s 往往比 QPS 更有解释力。因为输出长度决定 decode 轮数，也直接影响成本。

## 15.7 统计 queue 和 active requests

engine 每轮 step 可以记录：

```python
metrics.queue_length.append(len(engine.waiting_queue))
metrics.running_requests.append(len(engine.running_requests))
metrics.decode_groups.append(len(engine.decode_groups))
```

这些指标帮助判断：

1. TTFT 高是不是因为 queue 堆积。
2. running requests 是否达到上限。
3. decode group 是否过多导致吞吐差。
4. scheduler 是否没有及时接入新请求。

如果 TTFT p99 高，同时 queue length p99 也高，问题可能在 admission、扩容或 scheduler。

## 15.8 显存观测

PyTorch 可以读取 CUDA 显存：

```python
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
```

记录每轮显存：

```python
metrics.gpu_allocated_gb.append(allocated)
metrics.gpu_reserved_gb.append(reserved)
```

还可以在关键阶段记录：

1. 模型加载后。
2. prefill 后。
3. decode 中。
4. 请求结束后。
5. cleanup 后。

如果请求结束后 allocated 不下降，可能是 cache 没释放、引用没清理，或 PyTorch caching allocator 保留了 reserved memory。要区分 allocated 和 reserved。

## 15.9 KV Cache 显存的直觉压测

可以设计一个实验：固定模型和输出长度，逐步增加并发。

```text
并发 1, 2, 4, 8, 16, 32
prompt 长度固定 1024
max_new_tokens 固定 128
```

观察显存增长。

预期现象：

1. 模型权重占用固定。
2. 并发越高，KV Cache 越多。
3. decode 越长，cache 序列长度越长。
4. 到某个并发后出现 OOM 或 admission 拒绝。

这个实验能帮助理解：为什么模型能加载，不代表能高并发 serving。

## 15.10 Workload 设计

压测 workload 不能只用一个短 prompt。

至少要定义：

1. 请求总数。
2. 并发数。
3. 到达模式。
4. 输入 token 长度分布。
5. 输出 token 长度分布。
6. 是否 streaming。
7. sampling 参数。
8. 是否有共享 prefix。
9. 超时时间。

示例 workload：

```text
总请求数：1000
并发：32
输入长度：50% 为 256 tokens，30% 为 1024 tokens，20% 为 4096 tokens
输出长度：平均 256 tokens
streaming：开启
temperature：0.7
```

这样的报告比“压 100 QPS”更有意义。

## 15.11 到达模式

常见到达模式：

1. 固定并发：始终保持 N 个请求在飞。
2. 固定 QPS：每秒发固定数量请求。
3. 泊松到达：模拟更自然的随机流量。
4. 突发流量：短时间涌入大量请求。
5. 阶梯压测：逐步提高并发或 QPS。

不同模式暴露不同问题。

固定并发适合测系统稳定吞吐。突发流量适合看 queue 和 TTFT 长尾。阶梯压测适合找拐点。

## 15.12 一个最小压测脚本思路

伪代码：

```python
def run_benchmark(client, prompts, concurrency):
    results = []
    start_time = now()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(client.generate, prompt) for prompt in prompts]
        for future in futures:
            results.append(future.result())

    end_time = now()
    return summarize(results, start_time, end_time)
```

每个 result 要包含：

1. request_id。
2. input_tokens。
3. output_tokens。
4. ttft。
5. tpot。
6. total_latency。
7. finish_reason。
8. error。

如果是 streaming client，要在收到第一个 delta 时记录 TTFT，在收到 finish event 时记录总延迟。

## 15.13 如何分析压测结果

情况一：TTFT 高，TPOT 正常。

优先排查：

1. waiting queue。
2. prefill latency。
3. prompt 长度。
4. scheduler 是否过度优先 decode。
5. prefix cache 是否失效。

情况二：TTFT 正常，TPOT 高。

优先排查：

1. decode batch。
2. KV Cache 读取。
3. decode group 过多。
4. GPU 显存带宽。
5. streaming backpressure。

情况三：tokens/s 高，但 p99 很差。

说明吞吐不错，但长尾体验差。要看请求长度分布、queue、长 prompt 和 scheduler 公平性。

情况四：显存持续上涨。

排查 request cleanup、cache 引用、finished group 是否释放、stream queue 是否持有 request。

## 15.14 最小框架的预期瓶颈

我们这个教学版 MiniEngine 预计会有很多瓶颈：

1. decode group 不能合并，吞吐低。
2. finished 请求可能仍占用 group cache。
3. 没有真正 KV Cache Manager。
4. 没有 PagedAttention，cache 可能浪费。
5. scheduler 只按请求数限流。
6. streamer 用完整 decode 差分，长输出效率低。
7. HTTP handler 和 engine 并发模型粗糙。
8. 没有 optimized kernel。

这些瓶颈不是失败，而是学习路线。后面讲 vLLM 和 SGLang 时，就是看真实系统如何解决这些问题。

## 15.15 报告模板

一次压测报告可以包含：

```text
模型：xxx
硬件：1 x A100 80GB
并发：32
请求数：1000
输入长度分布：p50=512, p95=4096
输出长度分布：p50=128, p95=512

TTFT: p50=..., p95=..., p99=...
TPOT: p50=..., p95=..., p99=...
Total latency: p50=..., p95=..., p99=...

Input tokens/s: ...
Output tokens/s: ...
Queue length p95: ...
Active requests p95: ...
GPU allocated max: ... GB
Error rate: ...
Timeout rate: ...
```

报告要同时包含 workload、硬件、模型、指标和结论。没有 workload 描述的性能数字不可比较。

## 15.16 压测公式、指标表和可运行 demo

下面的 demo 不依赖外部库。它用 4 条 toy request trace 模拟一次小压测：

1. `r1` 和 `r2` 是短请求。
2. `r3_long_prompt` 是长 prompt，请求排队后 prefill 很慢。
3. `r4_burst` 是突发请求，queue wait 较高但 decode 正常。
4. 显存按 toy KV cache 公式估算，最后检查 cleanup 后只剩模型权重。

```python
from dataclasses import dataclass
from math import ceil


@dataclass
class RequestTrace:
    request_id: str
    arrival_ms: int
    scheduled_ms: int
    first_token_ms: int
    output_token_times_ms: list[int]
    finish_ms: int
    input_tokens: int
    finish_reason: str = "length"


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, ceil(len(ordered) * p / 100) - 1)
    return ordered[min(index, len(ordered) - 1)]


def request_metrics(trace):
    output_tokens = len(trace.output_token_times_ms)
    if output_tokens > 1:
        tpot = (trace.output_token_times_ms[-1] - trace.output_token_times_ms[0]) / (
            output_tokens - 1
        )
    else:
        tpot = 0.0
    return {
        "ttft_ms": trace.first_token_ms - trace.arrival_ms,
        "queue_wait_ms": trace.scheduled_ms - trace.arrival_ms,
        "prefill_ms": trace.first_token_ms - trace.scheduled_ms,
        "tpot_ms": round(tpot, 1),
        "e2e_ms": trace.finish_ms - trace.arrival_ms,
        "input_tokens": trace.input_tokens,
        "output_tokens": output_tokens,
        "finish_reason": trace.finish_reason,
    }


def active_tokens_at(trace, t):
    if not (trace.arrival_ms <= t < trace.finish_ms):
        return 0
    emitted = sum(1 for token_t in trace.output_token_times_ms if token_t <= t)
    return trace.input_tokens + emitted


def run_benchmark_demo():
    traces = [
        RequestTrace("r1", 0, 0, 90, [90, 120, 150, 180], 190, 128),
        RequestTrace("r2", 0, 20, 130, [130, 165, 200, 235, 270, 305], 320, 256),
        RequestTrace("r3_long_prompt", 50, 200, 760, [760, 800, 840], 850, 4096),
        RequestTrace("r4_burst", 80, 220, 290, [290, 335, 380, 425, 470], 500, 512),
    ]
    queue_samples = [2, 3, 2, 2, 1, 1, 0]
    active_samples = [0, 2, 3, 1, 1, 0]

    per_request = {trace.request_id: request_metrics(trace) for trace in traces}
    ttfts = [item["ttft_ms"] for item in per_request.values()]
    tpots = [item["tpot_ms"] for item in per_request.values()]
    e2es = [item["e2e_ms"] for item in per_request.values()]
    start_ms = min(trace.arrival_ms for trace in traces)
    end_ms = max(trace.finish_ms for trace in traces)
    duration_s = (end_ms - start_ms) / 1000
    total_input = sum(trace.input_tokens for trace in traces)
    total_output = sum(len(trace.output_token_times_ms) for trace in traces)

    layers = 16
    kv_heads = 8
    head_dim = 128
    dtype_bytes = 2
    kv_mib_per_token = 2 * layers * kv_heads * head_dim * dtype_bytes / (1024 * 1024)
    model_mib = 14000.0
    memory_sample_times = [0, 100, 250, 500, 760, 860]
    kv_mib_samples = []
    for t in memory_sample_times:
        active_tokens = sum(active_tokens_at(trace, t) for trace in traces)
        kv_mib_samples.append(round(active_tokens * kv_mib_per_token, 2))
    kv_peak_mib = round(model_mib + max(kv_mib_samples), 2)
    kv_after_cleanup_mib = model_mib + kv_mib_samples[-1]

    slo = {
        "ttft": percentile(ttfts, 95) <= 500,
        "tpot": percentile(tpots, 95) <= 50,
        "kv": kv_peak_mib <= 14500,
    }
    if not slo["ttft"] and percentile(queue_samples, 95) >= 2:
        bottleneck = "queue_or_prefill"
    elif not slo["tpot"]:
        bottleneck = "decode_or_streaming"
    elif not slo["kv"]:
        bottleneck = "kv_capacity"
    else:
        bottleneck = "within_toy_slo"

    summary = {
        "ttft_ms": {
            "p50": percentile(ttfts, 50),
            "p95": percentile(ttfts, 95),
            "p99": percentile(ttfts, 99),
        },
        "tpot_ms": {
            "p50": percentile(tpots, 50),
            "p95": percentile(tpots, 95),
        },
        "e2e_ms": {"p95": percentile(e2es, 95)},
        "input_tokens_per_s": round(total_input / duration_s, 1),
        "output_tokens_per_s": round(total_output / duration_s, 1),
        "queue_p95": percentile(queue_samples, 95),
        "active_p95": percentile(active_samples, 95),
        "kv_peak_mib": kv_peak_mib,
        "kv_after_cleanup_mib": kv_after_cleanup_mib,
        "slo_pass": slo,
        "bottleneck": bottleneck,
    }
    gates = {
        "request_metrics_complete": all(
            key in next(iter(per_request.values()))
            for key in ["ttft_ms", "tpot_ms", "e2e_ms", "input_tokens", "output_tokens"]
        ),
        "tail_latency_detected": summary["ttft_ms"]["p95"] > summary["ttft_ms"]["p50"],
        "throughput_computed": summary["input_tokens_per_s"] > 0
        and summary["output_tokens_per_s"] > 0,
        "kv_peak_tracked": summary["kv_peak_mib"] > summary["kv_after_cleanup_mib"],
        "cleanup_checked": summary["kv_after_cleanup_mib"] == model_mib,
        "bottleneck_classified": summary["bottleneck"] == "queue_or_prefill",
    }
    gates["benchmark_gate"] = all(gates.values())
    return per_request, summary, gates


per_request, summary, gates = run_benchmark_demo()
print("benchmark_per_request=", per_request)
print("benchmark_summary=", summary)
print("benchmark_gates=", gates)
```

一组稳定输出：

```text
benchmark_per_request= {'r1': {'ttft_ms': 90, 'queue_wait_ms': 0, 'prefill_ms': 90, 'tpot_ms': 30.0, 'e2e_ms': 190, 'input_tokens': 128, 'output_tokens': 4, 'finish_reason': 'length'}, 'r2': {'ttft_ms': 130, 'queue_wait_ms': 20, 'prefill_ms': 110, 'tpot_ms': 35.0, 'e2e_ms': 320, 'input_tokens': 256, 'output_tokens': 6, 'finish_reason': 'length'}, 'r3_long_prompt': {'ttft_ms': 710, 'queue_wait_ms': 150, 'prefill_ms': 560, 'tpot_ms': 40.0, 'e2e_ms': 800, 'input_tokens': 4096, 'output_tokens': 3, 'finish_reason': 'length'}, 'r4_burst': {'ttft_ms': 210, 'queue_wait_ms': 140, 'prefill_ms': 70, 'tpot_ms': 45.0, 'e2e_ms': 420, 'input_tokens': 512, 'output_tokens': 5, 'finish_reason': 'length'}}
benchmark_summary= {'ttft_ms': {'p50': 130, 'p95': 710, 'p99': 710}, 'tpot_ms': {'p50': 35.0, 'p95': 45.0}, 'e2e_ms': {'p95': 800}, 'input_tokens_per_s': 5872.9, 'output_tokens_per_s': 21.2, 'queue_p95': 3, 'active_p95': 3, 'kv_peak_mib': 14312.06, 'kv_after_cleanup_mib': 14000.0, 'slo_pass': {'ttft': False, 'tpot': True, 'kv': True}, 'bottleneck': 'queue_or_prefill'}
benchmark_gates= {'request_metrics_complete': True, 'tail_latency_detected': True, 'throughput_computed': True, 'kv_peak_tracked': True, 'cleanup_checked': True, 'bottleneck_classified': True, 'benchmark_gate': True}
```

这个 demo 的关键证据：

1. 每个请求都能拆出 `ttft_ms`、`queue_wait_ms`、`prefill_ms`、`tpot_ms`、`e2e_ms` 和 token 数。
2. `ttft_ms.p95=710` 明显高于 p50，说明平均值会掩盖长 prompt 或 burst 导致的长尾。
3. `tpot_ms.p95=45.0` 仍然达标，说明问题不在 decode token 间隔。
4. `input_tokens_per_s` 和 `output_tokens_per_s` 同时给出，避免只看 QPS。
5. `kv_peak_mib` 高于 cleanup 后显存，说明并发期间 KV cache 增长可见；`kv_after_cleanup_mib=14000.0` 表明 toy 请求结束后 KV 已释放。
6. `bottleneck='queue_or_prefill'`，说明压测报告应给出归因，而不是只列数字。

## 15.17 常见误区

误区一：压测只看 QPS。

LLM 请求 token 长度差异极大，QPS 不足以描述负载。

误区二：只测短 prompt。

短 prompt 无法暴露长上下文 prefill、KV Cache 和显存问题。

误区三：只看平均延迟。

p95/p99 更能反映线上体验和长尾问题。

误区四：GPU 利用率高就说明服务好。

GPU 忙可能是有效吞吐，也可能是排队、padding 浪费或长请求堵塞。

误区五：显存 reserved 高就是泄漏。

PyTorch caching allocator 会保留 reserved memory，要结合 allocated 和请求生命周期判断。

## 15.18 面试追问

1. LLM serving 压测为什么不能只看 QPS？
2. TTFT 和 TPOT 分别应该在什么时候记录？
3. 如何统计 output tokens/s？
4. 如何设计一个合理 workload？
5. TTFT 高但 TPOT 正常说明什么？
6. TPOT 高但 TTFT 正常说明什么？
7. 显存随并发增加的主要原因是什么？
8. 如何判断是 engine 慢还是 streaming 输出慢？

参考回答思路：

1. 先说 LLM 请求按 token 消耗资源，所以要看 token 分布和 tokens/s。
2. 再说 TTFT 对应排队和 prefill，TPOT 对应 decode 和 streaming。
3. 然后说 workload 要包含输入输出长度、并发、到达模式和是否 streaming。
4. 最后讲排查路径：queue、prefill、decode、KV Cache、GPU、stream queue。

## 15.19 小练习

1. 给 MiniEngine 加上 TTFT、TPOT 和 total latency 统计。
2. 写一个固定并发压测脚本，发送 100 个请求。
3. 构造短 prompt 和长 prompt 两组 workload，对比 TTFT。
4. 固定 prompt 长度，逐步提高并发，观察显存变化。
5. 模拟客户端读 stream 很慢，观察 token 间隔和 stream queue。

## 15.20 本章小结

本章完成了第二部分的性能收尾。

我们给 MiniEngine 加入了指标视角：TTFT、TPOT、tokens/s、queue length、active requests 和显存。压测不是为了得到单一 QPS，而是为了把请求生命周期拆开，定位瓶颈在排队、prefill、decode、KV Cache、scheduler 还是 streaming。

从下一章开始进入第三部分：vLLM 架构详解。你会看到 vLLM 如何系统性解决我们这个 MiniEngine 暴露出来的问题，尤其是 PagedAttention、KV Cache Block Manager 和 continuous batching。
