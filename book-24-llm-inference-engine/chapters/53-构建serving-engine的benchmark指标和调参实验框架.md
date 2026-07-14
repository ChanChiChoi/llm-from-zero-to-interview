# 第 53 章 构建 serving engine 的 benchmark、指标和调参实验框架

第 52 章我们把 scheduler、paged KV cache、prefix cache 和 preemption 串成了一个完整 engine loop。

到这一步，mini engine 已经不再只是能跑 demo。

它开始具备真实 serving engine 的核心复杂度。

但工程上还有一个更重要的问题：

```text
每次改 scheduler、KV cache、prefix cache 或 preemption 之后，怎么知道系统真的变好了？
```

只看单条请求的输出是否正确远远不够。

serving engine 的优化目标通常不是“某一次 forward 更快”，而是：

1. 首 token 更快。
2. 流式输出更稳定。
3. 总吞吐更高。
4. 显存使用更可控。
5. 高并发和长上下文下不 OOM。
6. prefix cache 命中时确实减少 prefill 成本。
7. KV 压力下 preemption 不会把尾延迟打爆。

这些都需要 benchmark、指标和可重复实验来回答。

本章就把 mini engine 升级成一个可压测、可观测、可调参、可回归的实验框架。

## 53.0 本讲资料边界与第二轮精修口径

本章按第二轮精修口径，只讲教学版 serving engine 的 benchmark 指标、workload、trace、参数扫描和回归门禁。

公开资料校准主要参考四类口径：

1. vLLM benchmarking / optimization 文档对 serving benchmark、chunked prefill、decode 优先、`max_num_batched_tokens`、`max_num_seqs`、KV cache usage 和 preemption 指标的公开说明。
2. vLLM prefix caching 设计文档对 prefix cache hit tokens、cached blocks、ref count、free queue 和 eviction 观测点的公开说明。
3. SGLang benchmark / profiling 文档对在线 serving、吞吐、延迟、profiling 和调参实验的公开口径。
4. NVIDIA GenAI-Perf / TensorRT-LLM benchmark 资料对 TTFT、inter-token latency、request latency、throughput、concurrency 和 streaming 指标的公开口径。

本章不追求复现任何公开 benchmark 数字，不绑定真实 GPU 型号、模型权重、框架版本、压测工具 CLI、Prometheus / Grafana 部署、CUDA profiling 细节或生产容量结论。我们只验证一个最小闭环：

```text
workload spec -> benchmark config -> request trace -> engine step trace -> summary metrics -> regression comparison -> tuning decision
```

第二轮新增 demo 的验收重点是：

```text
是否覆盖 short / long / shared prefix / KV pressure workload；
TTFT、TPOT、E2E、throughput 是否从请求 trace 推导；
queue、active requests、KV peak、cleanup 是否从 step trace 推导；
baseline 和 candidate 是否使用可复现实验指纹；
prefix cache 收益是否同时看 hit tokens 和 TTFT；
preemption 是否作为风险指标而不是收益指标；
调参结论是否同时检查用户延迟、吞吐、KV 压力和失败请求。
```

## 53.1 本章目标

读完本章，你应该能讲清：

1. serving engine benchmark 要测哪些指标。
2. TTFT、TPOT、E2E latency、throughput 分别代表什么。
3. 为什么平均值经常误导 serving 系统优化。
4. 如何设计短请求、长请求、共享 prefix、长短混合和 KV 压力场景。
5. 如何给 scheduler、KV cache、prefix cache 和 preemption 增加指标。
6. 如何记录每轮 engine step 的结构化 trace。
7. 如何做参数扫描和对比实验。
8. 如何判断一次优化是有效优化还是局部作弊。
9. 面试里如何讲 serving benchmark 和性能调优方法论。

这一章的重点不是追求某个漂亮数字。

重点是建立一套能长期使用的实验纪律。

## 53.2 为什么 benchmark 是 serving engine 的核心模块

很多人写推理框架时，会先关注 kernel、attention、KV cache 和 scheduler。

这些当然重要。

但如果没有 benchmark，优化很快会变成猜谜。

比如你改了 scheduler，让 prefill 更积极。

你可能看到整体吞吐上涨了。

但同时也可能发生：

1. decode 被 prefill 挤压。
2. TPOT 抖动变大。
3. streaming 体验变差。
4. 长请求占满 KV blocks。
5. 新请求 TTFT 变高。
6. preemption 次数增加。
7. p99 E2E latency 恶化。

如果只看总 tokens/s，你会以为系统变好了。

但用户可能觉得它变慢了。

再比如你启用了 prefix cache。

你可能看到共享 prompt 场景的 TTFT 下降。

但如果 cache eviction 策略不好，可能又会导致：

1. cached blocks 长期占用显存。
2. active 请求可用 KV blocks 变少。
3. preemption 变多。
4. 无共享 prefix 的 workload 反而变差。

所以 benchmark 不应该是项目最后补的脚本。

它应该是 serving engine 的核心模块之一。

## 53.3 四个最基础的用户侧指标

先从用户真正感受到的指标开始。

### 53.3.1 TTFT

TTFT 是 time to first token。

它表示：

```text
请求提交到第一个输出 token 返回之间的时间。
```

TTFT 主要受这些因素影响：

1. waiting queue 排队时间。
2. tokenization 时间。
3. prefix cache lookup 时间。
4. prefill 计算时间。
5. chunked prefill 是否被拆成多轮。
6. scheduler 是否让新请求及时进入 batch。
7. GPU 是否被已有 decode 或长 prefill 占满。

对于聊天机器人，TTFT 非常关键。

用户通常可以接受模型慢慢输出，但很难接受很久没有任何响应。

### 53.3.2 TPOT

TPOT 是 time per output token。

它表示：

```text
从第一个 token 之后，后续每个输出 token 的平均间隔。
```

更严格地说，我们通常会记录每两个相邻输出 token 之间的 inter-token latency。

然后统计平均值、p50、p90、p99。

TPOT 主要受这些因素影响：

1. decode batch 大小。
2. attention kernel 性能。
3. KV cache 读写效率。
4. prefill 是否挤占 decode。
5. scheduler 是否优先保障 running 请求。
6. GPU 上是否有过多小 batch 或碎片化 batch。

如果 TPOT 抖动很大，用户会感到输出一卡一卡的。

### 53.3.3 E2E latency

E2E latency 是 end-to-end latency。

它表示：

```text
请求提交到请求完成之间的总时间。
```

它包括 TTFT、所有 decode step、排队、调度、网络、后处理等时间。

E2E latency 对离线任务、agent 长链路和批处理任务很重要。

但它不是唯一指标。

两个请求的 E2E latency 可能相同，但用户体验完全不同：

```text
请求 A：很快出首 token，然后稳定流式输出。
请求 B：等很久才出首 token，然后一次性吐完。
```

对交互式应用来说，请求 A 通常更好。

### 53.3.4 Throughput

throughput 可以有多种定义：

1. requests/s。
2. input tokens/s。
3. output tokens/s。
4. total tokens/s。
5. successful requests/s。

serving engine 里最常见的是 output tokens/s 和 total tokens/s。

但吞吐必须和延迟一起看。

一个系统可以通过无限排队把 GPU 打满，让 tokens/s 很高。

但用户 TTFT 和 E2E latency 会非常差。

所以 benchmark 报告里不能只放吞吐。

至少要同时放：

1. TTFT p50/p90/p99。
2. TPOT p50/p90/p99。
3. E2E p50/p90/p99。
4. output tokens/s。
5. request success rate。

## 53.4 平均值为什么危险

很多初学 benchmark 会写：

```python
avg_latency = sum(latencies) / len(latencies)
```

这个数字不是没用。

但在 serving 系统里，它经常掩盖问题。

举个例子：

```text
99 个请求延迟 1s，1 个请求延迟 60s。
平均延迟是 1.59s。
```

平均值看起来还行。

但那个 60s 的请求可能已经超时、取消或导致用户流失。

serving engine 必须关注尾延迟。

常见统计包括：

1. p50：一半请求低于这个值。
2. p90：90% 请求低于这个值。
3. p95：95% 请求低于这个值。
4. p99：99% 请求低于这个值。
5. max：最坏请求。

调 scheduler 时尤其要看 p99。

因为很多策略的副作用只会体现在少数 unlucky request 上。

比如：

1. 长请求反复被 preempt。
2. 某些请求一直排在 waiting queue。
3. prefix cache eviction 导致少数请求完全 miss。
4. decode 被大 prefill chunk 间歇性阻塞。

平均值会把这些问题稀释掉。

尾延迟会把它们暴露出来。

## 53.5 engine 内部指标

用户侧指标告诉我们“系统表现怎么样”。

内部指标告诉我们“为什么会这样”。

一个教学版 mini engine 至少应该记录以下内部指标。

### 53.5.1 scheduler 指标

```text
scheduler_step_total
scheduler_step_duration_ms
waiting_queue_size
running_queue_size
swapped_queue_size
scheduled_prefill_requests
scheduled_decode_requests
scheduled_prefill_tokens
scheduled_decode_tokens
num_batched_tokens
num_batched_requests
```

这些指标用来回答：

1. 每轮调度花了多久。
2. waiting queue 是否持续积压。
3. running queue 是否被维持在合理大小。
4. prefill 和 decode 的比例是否合理。
5. batch 是否太小导致 GPU 利用率低。
6. batch 是否太大导致延迟恶化。

### 53.5.2 KV cache 指标

```text
kv_total_blocks
kv_free_blocks
kv_used_blocks
kv_cached_blocks
kv_active_blocks
kv_allocation_success_total
kv_allocation_failure_total
kv_eviction_total
kv_fragmentation_estimate
```

这些指标用来回答：

1. KV block pool 是否经常被打满。
2. allocation failure 是否发生在高并发或长上下文下。
3. cached blocks 是否挤压 active blocks。
4. eviction 是否过于频繁。
5. block size 是否选择得不合理。

### 53.5.3 prefix cache 指标

```text
prefix_cache_lookup_total
prefix_cache_hit_total
prefix_cache_miss_total
prefix_cache_hit_tokens_total
prefix_cache_reused_blocks_total
prefix_cache_evicted_blocks_total
prefix_cache_active_ref_blocks
```

这些指标用来回答：

1. prefix cache 是否真的命中。
2. 命中后节省了多少 prefill tokens。
3. cache blocks 是否被 active request 引用。
4. eviction 是否误删了活跃 block。
5. cache 命中收益是否抵消了 cache 管理开销。

### 53.5.4 preemption 指标

```text
preemption_total
preemption_recompute_total
preemption_swap_total
preempted_requests_current
recompute_tokens_total
swap_out_bytes_total
swap_in_bytes_total
repeated_preemption_total
```

这些指标用来回答：

1. KV 压力是否已经过高。
2. 哪些 workload 触发 preemption。
3. 同一个请求是否被反复 preempt。
4. recompute 带来了多少额外计算。
5. swap 是否被 CPU/GPU 带宽限制。

要记住：

```text
preemption 是鲁棒性机制，不是性能优化指标。
```

preemption_total 高通常不是好事。

它说明系统正在资源压力下艰难维持。

## 53.6 请求级 trace

聚合指标适合看趋势。

但定位具体 bug 时，还需要请求级 trace。

一个 request trace 可以记录：

```python
@dataclass
class RequestTrace:
    request_id: str
    arrival_time: float
    first_scheduled_time: float | None = None
    first_token_time: float | None = None
    finish_time: float | None = None

    prompt_len: int = 0
    output_len: int = 0
    num_prefill_steps: int = 0
    num_decode_steps: int = 0
    num_preemptions: int = 0
    num_recompute_tokens: int = 0

    prefix_cache_hit_tokens: int = 0
    prefix_cache_reused_blocks: int = 0
```

完成请求后可以计算：

```python
def summarize_trace(trace: RequestTrace) -> dict:
    ttft = None
    if trace.first_token_time is not None:
        ttft = trace.first_token_time - trace.arrival_time

    e2e = None
    if trace.finish_time is not None:
        e2e = trace.finish_time - trace.arrival_time

    return {
        "request_id": trace.request_id,
        "prompt_len": trace.prompt_len,
        "output_len": trace.output_len,
        "ttft_ms": None if ttft is None else ttft * 1000,
        "e2e_ms": None if e2e is None else e2e * 1000,
        "num_prefill_steps": trace.num_prefill_steps,
        "num_decode_steps": trace.num_decode_steps,
        "num_preemptions": trace.num_preemptions,
        "prefix_cache_hit_tokens": trace.prefix_cache_hit_tokens,
    }
```

请求级 trace 的价值在于可以解释尾延迟。

比如某个请求 p99 E2E latency 很高。

你可以检查：

1. 它是不是排队太久。
2. 它是不是 prompt 太长。
3. 它是不是被 chunked prefill 拆了太多轮。
4. 它是不是 prefix cache miss。
5. 它是不是被 preempt 过。
6. 它是不是在 decode 阶段被大 prefill 挤压。

如果没有 trace，只看最终 latency 很难定位根因。

## 53.7 engine step 级 trace

除了请求级 trace，还应该记录每轮 engine step 的调度结果。

例如：

```python
@dataclass
class EngineStepTrace:
    step_id: int
    timestamp: float
    waiting_queue_size: int
    running_queue_size: int
    free_blocks_before: int
    free_blocks_after: int
    scheduled_prefill_tokens: int
    scheduled_decode_tokens: int
    scheduled_requests: int
    prefix_cache_hits: int
    evicted_blocks: int
    preempted_requests: int
    step_duration_ms: float
```

每轮输出一行 JSONL：

```json
{"step": 128, "waiting": 14, "running": 32, "free_blocks_before": 9, "prefill_tokens": 512, "decode_tokens": 31, "preempted": 1}
```

JSONL 的好处是简单。

你可以直接用脚本做后处理，也可以导入可视化工具。

engine step trace 适合回答：

1. 哪一轮开始 waiting queue 积压。
2. 哪一轮 free blocks 快用完。
3. 哪一轮发生 eviction。
4. 哪一轮发生 preemption。
5. 哪一轮 prefill tokens 太多导致 decode 抖动。

这类 trace 对调 scheduler 特别有用。

## 53.8 最小指标收集器

教学项目里不一定一开始就接 Prometheus。

可以先写一个内存版 metrics collector。

```python
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Metrics:
    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    gauges: dict[str, float] = field(default_factory=dict)
    histograms: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def inc(self, name: str, value: int = 1) -> None:
        self.counters[name] += value

    def set_gauge(self, name: str, value: float) -> None:
        self.gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        self.histograms[name].append(value)
```

再写一个 percentile 函数：

```python
def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = int((len(values) - 1) * p)
    return values[index]
```

输出 summary：

```python
def summarize_histogram(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "count": len(values),
        "avg": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }
```

这个实现很粗糙，但足够教学使用。

真实线上系统会用更专业的 histogram 实现，避免保存所有样本。

## 53.9 benchmark workload 设计

benchmark 最大的坑是 workload 不真实。

只测一种 prompt 长度、一个并发数、一个输出长度，很容易得到误导结论。

至少要设计五类 workload。

### 53.9.1 短请求 baseline

短请求用于验证基础吞吐和调度开销。

典型配置：

```text
prompt_len: 32-128
output_len: 32-128
concurrency: 1, 4, 8, 16, 32
shared_prefix: false
```

它主要观察：

1. 小 batch 下 engine overhead。
2. batch 变大后的吞吐增长。
3. TTFT 是否随并发线性恶化。
4. decode TPOT 是否稳定。

如果短请求 baseline 都不稳定，先不要急着测复杂场景。

### 53.9.2 长 prompt 场景

长 prompt 用于测试 prefill、chunked prefill 和 KV block 分配。

典型配置：

```text
prompt_len: 2048-8192
output_len: 64-256
concurrency: 1, 4, 8, 16
shared_prefix: false
```

它主要观察：

1. prefill 是否导致 TTFT 很高。
2. chunked prefill 是否改善 decode 抖动。
3. KV blocks 是否快速耗尽。
4. long prompt 是否挤压短请求。

### 53.9.3 共享 prefix 场景

共享 prefix 用于测试 prefix cache。

典型配置：

```text
system_prompt_len: 1024-4096
unique_suffix_len: 32-256
output_len: 64-256
shared_prefix_ratio: 0.5-1.0
```

它主要观察：

1. prefix_cache_hit_total 是否上升。
2. prefix_cache_hit_tokens_total 是否符合预期。
3. TTFT 是否下降。
4. cached blocks 是否挤压 active requests。
5. eviction 是否过早发生。

一个好的 prefix cache benchmark 必须同时包含命中和不命中的请求。

否则你只能证明“理想情况下它有用”。

### 53.9.4 长短混合场景

长短混合用于测试 scheduler 公平性。

典型配置：

```text
short_prompt_len: 64
short_output_len: 64
long_prompt_len: 4096
long_output_len: 256
long_request_ratio: 0.1-0.3
concurrency: 16-64
```

它主要观察：

1. 短请求 TTFT 是否被长请求拖垮。
2. 长请求是否长期饥饿。
3. chunked prefill 是否有效。
4. decode 优先策略是否保护 streaming。
5. p99 是否明显恶化。

这个场景非常接近真实线上服务。

因为实际请求长度往往是重尾分布。

### 53.9.5 KV 压力场景

KV 压力场景用于测试 eviction 和 preemption。

典型配置：

```text
prompt_len: 2048-8192
output_len: 512-2048
concurrency: high
kv_blocks: deliberately limited
prefix_cache: optional
```

它主要观察：

1. free_blocks 是否接近 0。
2. allocation failure 是否被正确处理。
3. eviction 是否先于 preemption。
4. preemption_total 是否上升。
5. repeated_preemption_total 是否可控。
6. 系统是否避免 OOM。

这个场景不是为了追求好看结果。

它是为了故意打爆系统，验证鲁棒性。

## 53.10 workload 生成器

可以定义一个简单请求规格：

```python
@dataclass
class RequestSpec:
    request_id: str
    arrival_time: float
    prompt_len: int
    output_len: int
    shared_prefix_id: str | None = None
```

再写 workload 生成器：

```python
import random


def generate_short_requests(num_requests: int, qps: float) -> list[RequestSpec]:
    requests = []
    now = 0.0
    for i in range(num_requests):
        now += random.expovariate(qps)
        requests.append(RequestSpec(
            request_id=f"req-{i}",
            arrival_time=now,
            prompt_len=random.randint(32, 128),
            output_len=random.randint(32, 128),
        ))
    return requests
```

这里用 `expovariate(qps)` 模拟泊松到达。

真实线上请求通常不是完全泊松分布，但它比固定间隔更接近实际突发情况。

共享 prefix 可以这样生成：

```python
def generate_shared_prefix_requests(
    num_requests: int,
    qps: float,
    shared_prefix_ratio: float,
) -> list[RequestSpec]:
    requests = []
    now = 0.0
    for i in range(num_requests):
        now += random.expovariate(qps)
        use_shared = random.random() < shared_prefix_ratio
        requests.append(RequestSpec(
            request_id=f"req-{i}",
            arrival_time=now,
            prompt_len=4096 if use_shared else random.randint(256, 1024),
            output_len=random.randint(64, 256),
            shared_prefix_id="system-prompt-a" if use_shared else None,
        ))
    return requests
```

教学项目里 prompt 可以先用虚拟 token 表示。

也就是不一定真的构造自然语言文本，而是直接构造 token ids。

这样 benchmark 更稳定，也更容易控制长度。

## 53.11 open-loop 和 closed-loop 压测

serving benchmark 有两种常见模式。

### 53.11.1 closed-loop

closed-loop 是：

```text
固定并发数。一个请求完成后，再发下一个请求。
```

它像这样：

```python
while benchmark_not_done:
    while inflight < concurrency:
        send_request()
    wait_one_request_finish()
```

closed-loop 简单，适合测系统在固定并发下的稳定吞吐。

但它有一个问题：

```text
系统变慢时，发送请求的速度也会自动变慢。
```

因此它不容易暴露排队崩溃。

### 53.11.2 open-loop

open-loop 是：

```text
按预先设定的到达时间发送请求，不管前面的请求是否完成。
```

它更接近真实线上流量。

如果系统处理不过来，waiting queue 会堆积，TTFT 和 E2E 会上升。

open-loop 适合回答：

1. 某个 QPS 下系统是否稳定。
2. 排队是否持续增长。
3. 系统饱和点在哪里。
4. admission control 是否生效。

缺点是实现稍复杂，并且更容易把系统打崩。

教学项目建议两种都做：

1. closed-loop 用于快速本地回归。
2. open-loop 用于容量评估和压力测试。

## 53.12 参数扫描

serving engine 有很多关键参数：

```text
max_num_seqs
max_num_batched_tokens
max_model_len
block_size
gpu_memory_utilization
enable_chunked_prefill
max_prefill_chunk_tokens
enable_prefix_cache
prefix_cache_max_blocks
preemption_mode
```

不要手工改一个参数跑一次。

应该写参数扫描脚本。

例如：

```python
def run_sweep():
    for max_num_seqs in [8, 16, 32, 64]:
        for max_tokens in [512, 1024, 2048, 4096]:
            config = EngineConfig(
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_tokens,
            )
            result = run_benchmark(config, workload="mixed")
            save_result(config, result)
```

每次实验至少记录：

1. git commit。
2. 模型名称。
3. GPU 型号。
4. batch 参数。
5. workload 配置。
6. 随机种子。
7. benchmark 开始和结束时间。
8. 指标 summary。
9. 原始 trace 文件路径。

没有这些元信息，实验结果很难复现。

## 53.13 如何比较两次实验

比较实验时，不要只问：

```text
吞吐有没有涨？
```

应该按层次比较。

第一层：正确性。

```text
请求是否全部成功？输出长度是否符合预期？是否 OOM？是否有异常？
```

第二层：用户体验。

```text
TTFT、TPOT、E2E 的 p50/p90/p99 是否改善？
```

第三层：资源效率。

```text
tokens/s 是否提升？GPU 利用率是否更高？KV blocks 是否使用更充分？
```

第四层：副作用。

```text
preemption 是否增加？eviction 是否增加？allocation failure 是否增加？尾延迟是否恶化？
```

一次优化可以接受某些 trade-off。

但必须把 trade-off 说清楚。

例如：

```text
max_num_batched_tokens 从 1024 增加到 4096 后，output tokens/s 提升 18%，但 TTFT p99 从 1.2s 上升到 2.8s。这个配置适合离线批处理，不适合交互式聊天。
```

这才是工程上有用的结论。

## 53.14 常见调参方向

### 53.14.1 TTFT 太高

可能原因：

1. waiting queue 积压。
2. prefill chunk 太大。
3. max_num_seqs 太小，新请求进不来。
4. 长 prompt 阻塞短 prompt。
5. prefix cache 没命中。
6. token budget 被 decode 或其他 prefill 用完。

常见调整：

1. 启用 chunked prefill。
2. 降低 max_prefill_chunk_tokens。
3. 增加 max_num_seqs。
4. 对短请求做优先级保护。
5. 优化 prefix cache lookup 和命中策略。

### 53.14.2 TPOT 抖动大

可能原因：

1. decode 没有被优先调度。
2. 大 prefill chunk 抢占 GPU。
3. batch size 波动太大。
4. KV cache 访问不稳定。
5. preemption 或 swap 干扰 decode。

常见调整：

1. decode 优先。
2. 限制每轮 prefill token budget。
3. 控制 running queue 大小。
4. 减少 swap。
5. 优化 attention metadata 构造。

### 53.14.3 吞吐低

可能原因：

1. batch 太小。
2. max_num_batched_tokens 太低。
3. max_num_seqs 太低。
4. CPU 侧调度或 tokenization 成瓶颈。
5. GPU 等待数据。
6. prefix cache 没有减少实际计算。

常见调整：

1. 增加 max_num_batched_tokens。
2. 增加 max_num_seqs。
3. 合并小请求。
4. 减少 Python 调度开销。
5. 异步化 tokenization 和输出处理。

### 53.14.4 preemption 过多

可能原因：

1. KV blocks 不够。
2. max_num_seqs 太高。
3. max_num_batched_tokens 太激进。
4. 长上下文太多。
5. prefix cache 占用过多 blocks。
6. admission control 缺失。

常见调整：

1. 降低并发。
2. 限制 max_model_len。
3. 降低 max_num_seqs。
4. 改善 prefix cache eviction。
5. 启用更保守的 admission control。
6. 增加 KV cache budget。

## 53.15 benchmark 报告模板

一次完整 benchmark 报告可以长这样：

```text
Experiment: mixed-long-short-v1
Model: tiny-llm-1b
GPU: A100-80G
Commit: abc1234
Seed: 42

Config:
  max_num_seqs: 32
  max_num_batched_tokens: 2048
  block_size: 16
  chunked_prefill: true
  max_prefill_chunk_tokens: 512
  prefix_cache: true
  preemption_mode: recompute

Workload:
  requests: 1000
  arrival: open-loop qps=8
  short_ratio: 0.8
  long_ratio: 0.2
  shared_prefix_ratio: 0.5

Results:
  success_rate: 100%
  output_tokens_per_sec: 18420
  total_tokens_per_sec: 32780

  ttft_ms: avg=420 p50=310 p90=870 p99=2100
  tpot_ms: avg=42 p50=38 p90=61 p99=120
  e2e_ms: avg=4300 p50=2800 p90=9200 p99=18000

Internal:
  prefix_cache_hit_rate: 62%
  prefix_cache_hit_tokens_total: 1.8M
  kv_eviction_total: 120
  preemption_total: 3
  allocation_failure_total: 0

Conclusion:
  Prefix cache significantly improves shared-prefix TTFT.
  p99 E2E still high for long requests under qps=8.
  Next experiment should reduce max_prefill_chunk_tokens to protect decode latency.
```

报告里一定要有 conclusion。

没有结论的 benchmark 只是数字堆砌。

## 53.16 最容易犯的 benchmark 错误

错误一：只测单请求。

```text
结果：看不到 continuous batching、排队、KV 压力和 scheduler trade-off。
```

错误二：只看平均延迟。

```text
结果：p99 问题被掩盖。
```

错误三：只看 tokens/s。

```text
结果：可能牺牲 TTFT 和 TPOT 换吞吐。
```

错误四：没有固定随机种子。

```text
结果：两次实验 workload 不一样，结论不可信。
```

错误五：没有记录配置和 commit。

```text
结果：数字无法复现。
```

错误六：只测理想共享 prefix。

```text
结果：prefix cache 看起来收益巨大，但真实混合流量下收益不稳定。
```

错误七：没有 warmup。

```text
结果：首次 kernel 初始化、内存分配、cache cold start 污染数据。
```

错误八：把 preemption 当成成功指标。

```text
结果：系统虽然没 OOM，但尾延迟可能已经不可接受。
```

错误九：没有区分输入 tokens/s 和输出 tokens/s。

```text
结果：长 prompt workload 下吞吐数字容易被误读。
```

错误十：没有失败请求统计。

```text
结果：只统计成功请求会让系统看起来比真实情况更好。
```

## 53.17 面试高频问题

问题一：LLM serving benchmark 你会看哪些指标？

回答要点：用户侧看 TTFT、TPOT、E2E latency、request success rate、output tokens/s 和 total tokens/s，并且要看 p50/p90/p99。系统侧看 waiting/running queue、scheduled tokens、KV free/used/cached blocks、prefix cache hit tokens、evictions、preemptions 和 allocation failures。

问题二：为什么不能只看吞吐？

回答要点：吞吐可以通过扩大 batch 或增加排队来提高，但这可能显著恶化 TTFT、TPOT 和 p99 E2E latency。交互式 serving 需要同时优化吞吐和延迟，尤其要保护首 token 和流式输出稳定性。

问题三：open-loop 和 closed-loop benchmark 有什么区别？

回答要点：closed-loop 固定并发，请求完成后再发新请求，适合测固定并发下的稳定吞吐；open-loop 按固定到达率发请求，不管前面是否完成，更接近真实线上流量，能暴露排队积压和系统饱和点。

问题四：如何验证 prefix cache 真的有效？

回答要点：要构造共享 prefix 和非共享 prefix 的混合 workload，观察 hit rate、hit tokens、reused blocks、TTFT 改善和 prefill tokens 减少。同时要看 cached blocks 是否挤压 active KV，eviction 和 preemption 是否增加。

问题五：preemption 变多说明系统更鲁棒了吗？

回答要点：不一定。preemption 可以避免 OOM，是鲁棒性兜底机制，但频繁 preemption 通常说明 KV budget、并发参数、上下文长度或 prefix cache 占用有问题。评估时要看 repeated preemption、recompute tokens、尾延迟和成功率。

## 53.18 标准回答模板

如果面试官问“你会怎么做 LLM serving benchmark 和调优”，可以这样回答：

```text
我会先把指标分成用户侧和系统侧。用户侧重点看 TTFT、TPOT、E2E latency、success rate、output tokens/s 和 total tokens/s，并且一定看 p50/p90/p99，而不是只看平均值。系统侧会记录 waiting/running queue、每轮 scheduled prefill/decode tokens、KV free/used/cached blocks、prefix cache hit tokens、eviction、preemption、allocation failure 等指标。

workload 不能只测单请求。我会至少设计短请求 baseline、长 prompt、共享 prefix、长短混合和 KV 压力场景。短请求看基础开销和吞吐，长 prompt 看 prefill 和 chunked prefill，共享 prefix 看 cache 收益，长短混合看 scheduler 公平性，KV 压力场景故意打爆 block pool 来验证 eviction 和 preemption。

压测模式上我会同时做 closed-loop 和 open-loop。closed-loop 用固定并发做快速回归，open-loop 用固定到达率观察排队、饱和点和 admission control。每次实验都记录模型、GPU、commit、配置、随机种子、workload 和原始 trace，保证可复现。

调优时不会只追求 tokens/s。比如增大 max_num_batched_tokens 可能提高吞吐，但也可能恶化 TTFT 和 TPOT；启用 prefix cache 可能降低共享 prompt 的 TTFT，但也可能占用 KV blocks 导致 eviction 或 preemption。最终要根据业务场景选择配置：交互式聊天更重视 TTFT 和 TPOT，离线批处理可以更偏向吞吐。
```

## 53.19 Benchmark Framework 公式、回归门禁和可运行 demo

一次 benchmark 样本可以抽象成：

```math
r_i=(a_i,f_i,z_i,e_i,p_i,o_i,q_i,k_i,h_i,s_i)
```

其中 `a_i` 是到达时间，`f_i` 是首 token 时间，`z_i` 是输出 token 时间序列，`e_i` 是完成时间，`p_i` 是 prompt tokens，`o_i` 是 output tokens，`q_i` 是排队时间，`k_i` 是 KV 峰值，`h_i` 是 prefix hit tokens，`s_i` 是成功标记。

首 token 延迟：

```math
L_i^{\mathrm{first}}=f_i-a_i
```

平均 inter-token latency：

```math
L_i^{\mathrm{tok}}=\frac{e_i-f_i}{\max(1,o_i-1)}
```

输出吞吐：

```math
\Theta_{\mathrm{out}}=\frac{\sum_i o_i}{\max_i e_i-\min_i a_i}
```

prefix token 命中率：

```math
R_{\mathrm{hit}}=\frac{\sum_i h_i}{\max(1,\sum_i p_i)}
```

candidate 相对 baseline 的吞吐变化：

```math
\Delta_{\mathrm{out}}=\frac{\Theta_{\mathrm{out}}^{\mathrm{cand}}-\Theta_{\mathrm{out}}^{\mathrm{base}}}{\max(1,\Theta_{\mathrm{out}}^{\mathrm{base}})}
```

最终 benchmark framework 门禁：

```math
G_{\mathrm{benchfw}}=G_{\mathrm{workload}}G_{\mathrm{trace}}G_{\mathrm{slo}}G_{\mathrm{throughput}}G_{\mathrm{kv}}G_{\mathrm{cache}}G_{\mathrm{preempt}}G_{\mathrm{repro}}G_{\mathrm{decision}}
```

下面这个 0 依赖 demo 不调用真实模型，只审计两组 toy benchmark trace：

1. baseline 没有 prefix cache，长 prompt 和 KV pressure 场景触发多次 preemption。
2. candidate 开启 prefix cache 和更合理的 chunked prefill 后，TTFT / TPOT / E2E p95 改善，output tokens/s 提升。
3. 框架同时检查 workload 覆盖、实验指纹、SLO、吞吐、KV peak、cleanup、prefix 收益、preemption 风险和最终调参结论。

```python
from dataclasses import dataclass


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    index = int((len(values) - 1) * p)
    return values[index]


def hist(values):
    return {
        "avg": round(sum(values) / len(values), 2),
        "p50": round(percentile(values, 0.50), 2),
        "p90": round(percentile(values, 0.90), 2),
        "p95": round(percentile(values, 0.95), 2),
        "max": round(max(values), 2),
    }


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    max_num_seqs: int
    max_num_batched_tokens: int
    chunked_prefill: bool
    prefix_cache: bool
    preemption_mode: str
    seed: int

    def fingerprint(self):
        return (
            self.name,
            self.max_num_seqs,
            self.max_num_batched_tokens,
            self.chunked_prefill,
            self.prefix_cache,
            self.preemption_mode,
            self.seed,
        )


@dataclass
class RequestTrace:
    request_id: str
    profile: str
    arrival_ms: float
    first_token_ms: float
    finish_ms: float
    prompt_tokens: int
    output_tokens: int
    queue_wait_ms: float
    kv_peak_blocks: int
    prefix_hit_tokens: int = 0
    preemptions: int = 0
    success: bool = True

    def ttft_ms(self):
        return self.first_token_ms - self.arrival_ms

    def tpot_ms(self):
        return (self.finish_ms - self.first_token_ms) / max(1, self.output_tokens - 1)

    def e2e_ms(self):
        return self.finish_ms - self.arrival_ms


class ToyServingBenchmarkFramework:
    def __init__(self, config, traces, step_trace):
        self.config = config
        self.traces = traces
        self.step_trace = step_trace

    def classify_bottleneck(self, summary):
        if summary["failure_count"] > 0:
            return "failure"
        if summary["preemption_total"] > 1 or summary["kv_peak_blocks"] > 20:
            return "kv_capacity"
        if summary["ttft_ms"]["p95"] > 700:
            return "queue_prefill"
        if summary["tpot_ms"]["p95"] > 90:
            return "decode_streaming"
        return "balanced"

    def summarize(self):
        ok = [trace for trace in self.traces if trace.success]
        duration_s = (max(t.finish_ms for t in ok) - min(t.arrival_ms for t in ok)) / 1000.0
        prompt_tokens = sum(t.prompt_tokens for t in ok)
        output_tokens = sum(t.output_tokens for t in ok)
        summary = {
            "fingerprint": self.config.fingerprint(),
            "profiles": sorted({t.profile for t in ok}),
            "request_count": len(self.traces),
            "success_rate": round(len(ok) / len(self.traces), 3),
            "ttft_ms": hist([t.ttft_ms() for t in ok]),
            "tpot_ms": hist([t.tpot_ms() for t in ok]),
            "e2e_ms": hist([t.e2e_ms() for t in ok]),
            "output_tokens_per_s": round(output_tokens / duration_s, 2),
            "total_tokens_per_s": round((prompt_tokens + output_tokens) / duration_s, 2),
            "prefix_hit_rate": round(sum(t.prefix_hit_tokens for t in ok) / max(1, prompt_tokens), 3),
            "preemption_total": sum(t.preemptions for t in ok),
            "queue_p95_ms": round(percentile([s["waiting"] for s in self.step_trace], 0.95), 2),
            "active_p95": round(percentile([s["active"] for s in self.step_trace], 0.95), 2),
            "kv_peak_blocks": max(s["kv_blocks"] for s in self.step_trace),
            "kv_after_cleanup": self.step_trace[-1]["kv_blocks"],
            "failure_count": len(self.traces) - len(ok),
        }
        summary["bottleneck"] = self.classify_bottleneck(summary)
        return summary

    def request_rows(self):
        return [
            (
                t.request_id,
                round(t.ttft_ms(), 2),
                round(t.tpot_ms(), 2),
                round(t.e2e_ms(), 2),
                t.prefix_hit_tokens,
                t.preemptions,
            )
            for t in self.traces
        ]


baseline_config = BenchmarkConfig("baseline", 16, 1024, False, False, "recompute", 42)
candidate_config = BenchmarkConfig("candidate", 32, 2048, True, True, "recompute", 42)

baseline_traces = [
    RequestTrace("short_a", "short", 0, 120, 360, 64, 8, 40, 6),
    RequestTrace("short_b", "short", 50, 210, 470, 96, 8, 70, 8),
    RequestTrace("long_a", "long", 100, 900, 1500, 2048, 8, 180, 18, preemptions=1),
    RequestTrace("shared_a", "shared_prefix", 150, 760, 1100, 1024, 8, 100, 12),
    RequestTrace("shared_b", "shared_prefix", 200, 820, 1160, 1088, 8, 110, 13),
    RequestTrace("kv_hot", "kv_pressure", 250, 1300, 2100, 1536, 8, 260, 22, preemptions=2),
]
candidate_traces = [
    RequestTrace("short_a", "short", 0, 90, 310, 64, 8, 20, 5),
    RequestTrace("short_b", "short", 50, 160, 380, 96, 8, 30, 7),
    RequestTrace("long_a", "long", 100, 620, 1180, 2048, 8, 90, 16),
    RequestTrace("shared_a", "shared_prefix", 150, 420, 760, 1024, 8, 60, 10, prefix_hit_tokens=768),
    RequestTrace("shared_b", "shared_prefix", 200, 390, 700, 1088, 8, 40, 10, prefix_hit_tokens=1024),
    RequestTrace("kv_hot", "kv_pressure", 250, 820, 1500, 1536, 8, 140, 18, preemptions=1),
]
baseline_steps = [
    {"waiting": 3, "active": 2, "kv_blocks": 12},
    {"waiting": 4, "active": 4, "kv_blocks": 22},
    {"waiting": 2, "active": 3, "kv_blocks": 20},
    {"waiting": 0, "active": 1, "kv_blocks": 0},
]
candidate_steps = [
    {"waiting": 2, "active": 2, "kv_blocks": 10},
    {"waiting": 3, "active": 4, "kv_blocks": 18},
    {"waiting": 1, "active": 4, "kv_blocks": 17},
    {"waiting": 0, "active": 0, "kv_blocks": 0},
]

baseline = ToyServingBenchmarkFramework(baseline_config, baseline_traces, baseline_steps)
candidate = ToyServingBenchmarkFramework(candidate_config, candidate_traces, candidate_steps)
base_summary = baseline.summarize()
cand_summary = candidate.summarize()

comparison = {
    "ttft_p95_delta_ms": round(cand_summary["ttft_ms"]["p95"] - base_summary["ttft_ms"]["p95"], 2),
    "tpot_p95_delta_ms": round(cand_summary["tpot_ms"]["p95"] - base_summary["tpot_ms"]["p95"], 2),
    "e2e_p95_delta_ms": round(cand_summary["e2e_ms"]["p95"] - base_summary["e2e_ms"]["p95"], 2),
    "output_tps_delta_ratio": round(
        (cand_summary["output_tokens_per_s"] - base_summary["output_tokens_per_s"])
        / max(1, base_summary["output_tokens_per_s"]),
        3,
    ),
    "preemption_delta": cand_summary["preemption_total"] - base_summary["preemption_total"],
    "bottleneck_change": (base_summary["bottleneck"], cand_summary["bottleneck"]),
}

required_profiles = {"short", "long", "shared_prefix", "kv_pressure"}
gates = {
    "workload_coverage_ready": set(cand_summary["profiles"]) == required_profiles,
    "trace_metrics_ready": cand_summary["ttft_ms"]["p95"] == 520
    and cand_summary["tpot_ms"]["p95"] == 80,
    "slo_ready": cand_summary["ttft_ms"]["p95"] <= 700
    and cand_summary["tpot_ms"]["p95"] <= 90
    and cand_summary["success_rate"] == 1.0,
    "throughput_regression_ready": comparison["output_tps_delta_ratio"] >= 0.25,
    "kv_cleanup_ready": cand_summary["kv_peak_blocks"] <= 20
    and cand_summary["kv_after_cleanup"] == 0,
    "prefix_effect_ready": cand_summary["prefix_hit_rate"] > base_summary["prefix_hit_rate"],
    "preemption_risk_ready": cand_summary["preemption_total"] < base_summary["preemption_total"],
    "reproducibility_ready": baseline_config.seed == candidate_config.seed
    and base_summary["fingerprint"][0] == "baseline"
    and cand_summary["fingerprint"][0] == "candidate",
    "decision_ready": comparison["bottleneck_change"] == ("kv_capacity", "balanced"),
}
gates["serving_benchmark_framework_gate"] = all(gates.values())

summary = {
    "baseline": base_summary,
    "candidate": cand_summary,
    "candidate_request_rows": candidate.request_rows(),
    "comparison": comparison,
    "decision": "accept_candidate_for_interactive_serving",
}

print("serving_benchmark_framework_summary=", summary)
print("serving_benchmark_framework_gates=", gates)
```

一次运行的核心输出类似：

```text
serving_benchmark_framework_summary= {'baseline': {'fingerprint': ('baseline', 16, 1024, False, False, 'recompute', 42), 'profiles': ['kv_pressure', 'long', 'shared_prefix', 'short'], 'request_count': 6, 'success_rate': 1.0, 'ttft_ms': {'avg': 560.0, 'p50': 610, 'p90': 800, 'p95': 800, 'max': 1050}, 'tpot_ms': {'avg': 61.43, 'p50': 48.57, 'p90': 85.71, 'p95': 85.71, 'max': 114.29}, 'e2e_ms': {'avg': 990.0, 'p50': 950, 'p90': 1400, 'p95': 1400, 'max': 1850}, 'output_tokens_per_s': 22.86, 'total_tokens_per_s': 2811.43, 'prefix_hit_rate': 0.0, 'preemption_total': 3, 'queue_p95_ms': 3, 'active_p95': 3, 'kv_peak_blocks': 22, 'kv_after_cleanup': 0, 'failure_count': 0, 'bottleneck': 'kv_capacity'}, 'candidate': {'fingerprint': ('candidate', 32, 2048, True, True, 'recompute', 42), 'profiles': ['kv_pressure', 'long', 'shared_prefix', 'short'], 'request_count': 6, 'success_rate': 1.0, 'ttft_ms': {'avg': 291.67, 'p50': 190, 'p90': 520, 'p95': 520, 'max': 570}, 'tpot_ms': {'avg': 55.48, 'p50': 44.29, 'p90': 80.0, 'p95': 80.0, 'max': 97.14}, 'e2e_ms': {'avg': 680.0, 'p50': 500, 'p90': 1080, 'p95': 1080, 'max': 1250}, 'output_tokens_per_s': 32.0, 'total_tokens_per_s': 3936.0, 'prefix_hit_rate': 0.306, 'preemption_total': 1, 'queue_p95_ms': 2, 'active_p95': 4, 'kv_peak_blocks': 18, 'kv_after_cleanup': 0, 'failure_count': 0, 'bottleneck': 'balanced'}, 'candidate_request_rows': [('short_a', 90, 31.43, 310, 0, 0), ('short_b', 110, 31.43, 330, 0, 0), ('long_a', 520, 80.0, 1080, 0, 0), ('shared_a', 270, 48.57, 610, 768, 0), ('shared_b', 190, 44.29, 500, 1024, 0), ('kv_hot', 570, 97.14, 1250, 0, 1)], 'comparison': {'ttft_p95_delta_ms': -280, 'tpot_p95_delta_ms': -5.71, 'e2e_p95_delta_ms': -320, 'output_tps_delta_ratio': 0.4, 'preemption_delta': -2, 'bottleneck_change': ('kv_capacity', 'balanced')}, 'decision': 'accept_candidate_for_interactive_serving'}
serving_benchmark_framework_gates= {'workload_coverage_ready': True, 'trace_metrics_ready': True, 'slo_ready': True, 'throughput_regression_ready': True, 'kv_cleanup_ready': True, 'prefix_effect_ready': True, 'preemption_risk_ready': True, 'reproducibility_ready': True, 'decision_ready': True, 'serving_benchmark_framework_gate': True}
```

这个 demo 证明了几个关键点：

1. benchmark framework 不是只算 QPS，而是把 workload、config、request trace、step trace 和调参结论连起来。
2. candidate 的吞吐提升只有在 TTFT、TPOT、E2E、KV peak、cleanup、preemption 和失败请求都不过线时才算有效。
3. prefix cache 的收益不能只看 hit rate，要看 shared-prefix 请求 TTFT 是否改善，以及 KV pressure 是否可控。
4. preemption_total 下降是风险收敛信号；preemption_total 上升不能被包装成性能收益。
5. 实验指纹必须包含关键参数和 seed，否则两次 benchmark 没有可比较性。

## 53.20 小练习

1. 给 mini engine 增加 TTFT、TPOT、E2E latency 统计。
2. 实现一个内存版 `Metrics` collector。
3. 给 scheduler 增加 waiting/running queue size 指标。
4. 给 BlockManager 增加 free/used/cached blocks 指标。
5. 给 prefix cache 增加 hit tokens 和 reused blocks 指标。
6. 给 preemption 增加 repeated preemption 指标。
7. 实现请求级 `RequestTrace`。
8. 实现 engine step 级 JSONL trace。
9. 写一个 short request closed-loop benchmark。
10. 写一个 open-loop QPS benchmark。
11. 构造共享 prefix workload，验证 TTFT 是否下降。
12. 构造长短混合 workload，观察短请求 p99 TTFT。
13. 构造 KV block pool 打爆 workload，验证 preemption 是否触发。
14. 写一个参数扫描脚本，对比不同 `max_num_batched_tokens`。
15. 写一个 benchmark report markdown 模板。

## 53.21 本章总结

serving engine 的优化必须靠 benchmark 闭环。

只看单请求、平均延迟或 tokens/s 都不够。

一个可靠的 benchmark 框架至少要包含用户侧指标、系统侧指标、请求级 trace、engine step trace、可重复 workload、参数扫描和实验报告。

TTFT 反映首 token 体验。

TPOT 反映流式输出稳定性。

E2E latency 反映请求总完成时间。

throughput 反映系统处理能力。

它们必须一起看。

workload 也必须覆盖短请求、长 prompt、共享 prefix、长短混合和 KV 压力场景。

否则很容易优化了一个理想场景，却伤害真实线上流量。

从工程习惯上，每次改 scheduler、KV cache、prefix cache、preemption 或调参，都应该跑 benchmark，记录配置和 trace，再用数据判断 trade-off。

下一章可以继续讨论：当 benchmark 发现 CPU 调度、tokenization、输出处理或网络成为瓶颈时，如何把 mini engine 从同步 loop 升级成异步 serving 架构。
