# 第 15 章 压测 TTFT、TPOT、吞吐和显存

第二部分前面几章从 0 搭了一个最小推理框架：tokenizer、model wrapper、generate loop、sampling、KV Cache、batched prefill、batched decode、scheduler、streaming 和 HTTP API。本章做阶段性收尾：如何压测这个系统，如何看 TTFT、TPOT、吞吐和显存。

做推理框架不能只说“能跑”。能跑只是第一步。真正的工程问题是：首 token 多久返回？后续 token 多快？并发上来后吞吐如何？显存为什么涨？瓶颈在 prefill、decode、scheduler，还是 streaming？

一句话概括：

> 压测的目的不是得到一个漂亮 QPS，而是把请求生命周期拆成可观测指标，定位 serving engine 的真实性能瓶颈。

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

先明确几个核心指标。

TTFT：

```text
Time To First Token = 请求到达时间到首个输出 token 返回时间
```

TPOT：

```text
Time Per Output Token = decode 阶段平均每个输出 token 的时间
```

Output tokens/s：

```text
单位时间内生成的输出 token 数
```

Input tokens/s：

```text
单位时间内处理的 prompt token 数
```

显存：

```text
模型权重 + KV Cache + 临时 buffer + 运行时开销
```

这些指标要分开看，不能互相替代。

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

## 15.16 常见误区

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

## 15.17 面试追问

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

## 15.18 小练习

1. 给 MiniEngine 加上 TTFT、TPOT 和 total latency 统计。
2. 写一个固定并发压测脚本，发送 100 个请求。
3. 构造短 prompt 和长 prompt 两组 workload，对比 TTFT。
4. 固定 prompt 长度，逐步提高并发，观察显存变化。
5. 模拟客户端读 stream 很慢，观察 token 间隔和 stream queue。

## 15.19 本章小结

本章完成了第二部分的性能收尾。

我们给 MiniEngine 加入了指标视角：TTFT、TPOT、tokens/s、queue length、active requests 和显存。压测不是为了得到单一 QPS，而是为了把请求生命周期拆开，定位瓶颈在排队、prefill、decode、KV Cache、scheduler 还是 streaming。

从下一章开始进入第三部分：vLLM 架构详解。你会看到 vLLM 如何系统性解决我们这个 MiniEngine 暴露出来的问题，尤其是 PagedAttention、KV Cache Block Manager 和 continuous batching。
