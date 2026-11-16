# 第 6 章（第二篇）：Prefill 与 Decode——先把等待时间量对

上一段说明了模型为什么保存 KV Cache。现在换成用户视角：一次回答为什么首 token 要等一会儿，后面的 token 又按另一种节奏到达？服务把 prompt 分成 prefill 和逐步 decode 两段处理，但用户实际等待还可能包含排队、分词、网络和客户端缓冲。

本篇先定义 TTFT、TPOT、ITL 和吞吐，再把这些量放进一个时间账本。重点是确认计时起止点、分母和失败请求口径；如果这些没对齐，性能比较没有意义。算出的瓶颈会交给下一篇的分页缓存与调度分析。

## 6.2 Prefill/Decode：把延迟和吞吐拆成可测量的部分

### 6.2.1 从一次请求的时间线开始

用户发送 prompt 后，服务先要把已有文本整体处理一遍，这一步叫 prefill。prefill 可以在序列维度上并行计算，主要压力来自长 prompt 的矩阵乘法、注意力和 KV 写入。

第一个输出 token 生成前，服务还需要完成排队、tokenize、prefill、调度和采样，所以用户感受到的首 token 时间通常明显高于后续 token 之间的间隔。后续 decode 每轮只生成少量新 token，却要读取越来越长的 KV Cache，常常更受显存带宽、kernel launch 和调度开销影响。

```text
请求到达 ── 排队 ── tokenize ── prefill ── 首 token ── decode ── 结束
             │                         │          │
             │                         │          └─ 每 token 间隔
             └─ 用户可见 TTFT 包含哪些部分取决于计时点
```

所以“平均每秒生成多少 token”无法单独描述体验。短回答可能首 token 很快但总吞吐低，长 prompt 可能首 token 较慢但之后每 token 稳定；两种请求不能只用一个平均数比较。

### 6.2.2 四个基本指标

设请求进入服务的时间为 $t_{\mathrm{req}}$，首个可交付输出 token 的时间为 $t_1$，第 $n$ 个输出 token 到达时间为 $t_n$。

用户可见的首 token 延迟可以写成：

```math
\mathrm{TTFT}_{\mathrm{user}}
=
t_1-t_{\mathrm{req}}
```

如果只想测模型计算，不应把排队和网络混进来，可以另记：

```math
\mathrm{TTFT}_{\mathrm{model}}
=
t_{\mathrm{prefill,end}}-t_{\mathrm{prefill,start}}
```

两者都合理，但名字和采样点必须明确。逐 token 延迟可以定义为：

```math
\mathrm{TPOT}
=
\frac{t_n-t_1}{n-1},
\qquad n>1
```

相邻 token 的延迟序列更能发现抖动：

```math
\mathrm{ITL}_i=t_i-t_{i-1},
\qquad i=2,\ldots,n
```

服务端指标还会把这两个量具体化。vLLM v0.32.0 文档区分 `vllm:inter_token_latency_seconds`（流式输出事件之间的间隔）与 `vllm:request_time_per_output_token_seconds`（按请求计算的 TPOT，近似为 `(end-to-end latency - TTFT) / (output_tokens - 1)`）。当一次流式事件打包多个 token、或汇总时采用不同权重，两者会不同；该版本还说明，输出不超过一个 token 的请求在 Prometheus TPOT 指标中记为 0，而 `vllm bench serve` 的 TPOT 统计会排除这类请求。指标名称、分母和特殊值会随版本变化，部署报告应绑定 vLLM 版本并核对[v0.32.0 生产指标说明](https://docs.vllm.ai/en/v0.32.0/usage/metrics/)及[延迟指标定义](https://docs.vllm.ai/en/v0.32.0/design/metrics/)。

输出吞吐通常记为：

```math
\mathrm{throughput}
=
\frac{\sum_i \mathrm{output\_tokens}_i}
{\mathrm{wall\_clock\_seconds}}
```

这里的 $\mathrm{wall\_clock\_seconds}$ 必须是同一次负载实验中的正数经过时间；输出 token 数为非负整数。如果没有完成的请求或计时区间为零，应报告为无效样本，不应用一个默认值把分母藏起来。

把几个指标放在同一条请求上看：从请求到首 token 用时 0.30 秒，端到端用时 2.00 秒，共生成 40 个 token。TPOT 用首 token 之后的 39 个间隔作分母，约为 $(2.00-0.30)/39=0.0436$ 秒，即 43.6 ms/token；端到端输出速率则是 $40/2.00=20$ token/s。它们回答不同问题，所以不能互相替代。

如果服务有排队时间、输入 token 和输出 token，报告至少应分别给出：

```text
request rate：每秒进入多少请求
input tokens/s：每秒处理多少输入 token
output tokens/s：每秒产生多少输出 token
TTFT 的 p50/p95/p99
TPOT 或 ITL 的 p50/p95/p99
完成率、超时率、取消率和错误率
```

### 6.2.3 一个简单的时间账本

对单个请求，可以把端到端时间近似拆成：

```math
T_{\mathrm{e2e}}
=
T_{\mathrm{queue}}
+T_{\mathrm{tokenize}}
+T_{\mathrm{prefill}}
+T_{\mathrm{decode}}
+T_{\mathrm{network}}
```

如果输出 $N_{\mathrm{out}}$ 个 token，decode 阶段还可以近似写成：

```math
T_{\mathrm{decode}}
\approx
N_{\mathrm{out}}\cdot T_{\mathrm{step}}
```

这里 $T_{\mathrm{step}}$ 并不是常数。随着 batch 中序列长度变化，调度、KV 读取和 kernel 形状都会变化；如果使用 speculative decoding，单位“输出 token”还不再对应一次 target 单 token forward。

并发提高时，单请求 latency 和系统总 throughput 可能朝相反方向变化。一个请求独占 GPU 时，batch 太小导致硬件利用率不足；并发很多时，continuous batching 能提高吞吐，但排队和共享 batch 也会增加 p95。

### 6.2.4 为什么 CUDA 异步会让计时失真

GPU kernel 发射通常是异步的。CPU 记录了结束时间，并不表示 GPU 已经完成最后一个 kernel。如果直接使用 Python 的 time.perf_counter 包住模型调用，测到的可能只是 kernel launch 时间。

PyTorch 官方提供了同步和 CUDA Event 等方式。一个依赖 PyTorch 的最小计时片段如下：

> 代码状态：静态审阅，未运行（2026-10-29）
```python
import torch


def timed_model_call(model, input_ids, stream=None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this timing example")
    if not isinstance(input_ids, torch.Tensor) or not input_ids.is_cuda:
        raise ValueError("input_ids must be a CUDA tensor")
    if stream is None:
        stream = torch.cuda.current_stream(device=input_ids.device)

    stream.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start_event.record()
        result = model(input_ids)
        end_event.record()
    end_event.synchronize()
    milliseconds = start_event.elapsed_time(end_event)
    return result, milliseconds / 1000.0


# model and input_ids must be created on the same CUDA device. If the
# model launches work on another stream, that stream must be synchronized too.
result, elapsed_seconds = timed_model_call(model, input_ids)
```
这里的 model 和 input_ids 是上下文变量，函数使用同一条 CUDA stream 记录模型调用的起止。它不能在没有模型和 CUDA 设备的环境直接运行；如果 model 内部还使用了其他 stream，就要把那些 stream 也纳入同步边界。

基准测试至少要包含 warmup。第一次执行可能触发 CUDA context 初始化、权重加载、kernel 选择、内存分配、编译或 graph capture；如果把它和稳态请求混在一起，p50 和 p95 都会失真。

### 6.2.5 Prefill 和 Decode 的资源画像

Prefill 通常有较大的输入矩阵，比较容易利用矩阵乘法单元；Decode 的 token 数少，batch 形状碎片化，常常需要频繁读取权重和 KV Cache。不能简单把“prefill 快”或“decode 快”当成绝对结论，因为硬件、量化 kernel、序列长度和 batch 都会改变结果。

注意力的理论 score cell 数量可以写成：

```math
C_{\mathrm{attn}}
=
B\cdot n_h\cdot T_q\cdot T_k
```

在 prefill 中，$T_q$ 和 $T_k$ 都可能接近 prompt 长度；在单 token decode 中，$T_q\approx1$，$T_k$ 是当前历史长度。即使 FlashAttention 不把完整 $T_q\times T_k$ 矩阵落到显存，依赖关系的规模仍然随这些维度增长。

KV 读取的元素量还可以粗略写成：

```math
E_{\mathrm{KV,read}}
\propto
B\cdot L\cdot T_k\cdot n_{\mathrm{kv}}\cdot d_h
```

这解释了为什么 GQA、KV Cache 量化、paged layout、prefix sharing 和 batch 调度会直接影响 decode 性能。它们减少的是历史状态的存储或读取压力，但不会自动解决所有矩阵乘法和采样开销。

### 6.2.6 无依赖延迟账本 demo

下面的 demo 用确定性数字模拟不同请求的时间组成。它不调用模型，只帮助检查指标计算和 p95 的定义。

> 代码状态：静态审阅，未运行（2026-10-29）
```python
import math


REQUESTS = [
    {"queue": 2.0, "tokenize": 1.0, "prefill": 18.0, "decode": 40.0, "network": 2.0},
    {"queue": 8.0, "tokenize": 1.0, "prefill": 36.0, "decode": 54.0, "network": 3.0},
    {"queue": 1.0, "tokenize": 1.0, "prefill": 12.0, "decode": 24.0, "network": 2.0},
    {"queue": 12.0, "tokenize": 2.0, "prefill": 48.0, "decode": 72.0, "network": 4.0},
]


COMPONENTS = ("queue", "tokenize", "prefill", "decode", "network")


def checked_total(item):
    values = [item[name] for name in COMPONENTS]
    if any(
        not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in values
    ):
        raise ValueError("every duration must be finite and non-negative")
    return sum(values)


def percentile(values, quantile):
    if not values:
        raise ValueError("percentile requires at least one sample")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


if not REQUESTS:
    raise ValueError("at least one completed request is required")
e2e = [checked_total(item) for item in REQUESTS]
ttft = [
    item["queue"] + item["tokenize"] + item["prefill"] + item["network"]
    for item in REQUESTS
]
p95_ttft = percentile(ttft, 0.95)

print("ttft_ms=", [round(value, 2) for value in ttft])
print("e2e_ms=", [round(value, 2) for value in e2e])
print("ttft_p95_ms=", round(p95_ttft, 2))
assert all(total >= first for total, first in zip(e2e, ttft))
```
真实压测通常需要更多样本，p95 应使用事先约定的分位数算法，不能用“取最大值”或隐式的索引规则替代。样本数、失败请求是否排除、分位数算法和负载生成器的时间戳来源都应记录。

### 6.2.7 Chunked Prefill 与调度取舍

长 prompt 的 prefill 如果一次占满 GPU，正在 decode 的短请求可能长时间得不到调度，表现为 TPOT 抖动。Chunked prefill 把长 prompt 切成多个片段，与 decode token 交替安排：

```text
没有 chunked prefill：
  长 prompt prefill ────────────────
  短请求 decode                    等待

有 chunked prefill：
  prompt chunk 1 → decode → prompt chunk 2 → decode → ...
```

chunk 大小过大，仍然会阻塞 decode；过小，则 kernel launch 和调度开销增加，prefill 的矩阵效率下降。应在固定请求分布下扫描 chunk size，观察 TTFT、TPOT、吞吐和 p99，而不是凭单个请求的结果选择。

再进一步，可以把 Prefill 与 Decode 放到不同 worker：前者偏好大批量并行计算，后者需要持续读取 KV 并逐步产出 token。解耦后能分别配置资源，但每次交接都要传递 KV 状态，网络和拓扑可能抵消调度收益。vLLM 2026 年工程说明强调，这类架构的目标通常是在 TTFT/ITL 约束下提高 goodput，不保证固定 GPU 数下的原始 token 吞吐更高；其 L40S/Qwen2.5-7B 单机数据绑定约 8K 输入、256 输出、Poisson 到达、prefix cache 关闭与指定传输代理。若目标机器缺少高效 GPU peer-to-peer，KV 传输还可能推高 TTFT。见 [vLLM Prefill/Decode 解耦工程说明](https://vllm-project.github.io/2026/09/29/disaggregated-serving-guide.html)。

### 6.2.8 失败模式：指标好看但体验变差

测量出问题时，先按现象查计时边界，不要马上归因给模型：

| 观察到的现象 | 先核对什么 |
| --- | --- |
| TTFT 高 | 队列、tokenize、prompt 长度、prefill 和客户端到达时间；确认报告的是用户 TTFT 还是模型阶段时间 |
| TPOT 抖动 | 活跃序列、KV 读取、chunked prefill、调度、流式事件粒度和网络缓冲 |
| GPU 调用时间异常短 | CUDA 是否异步返回；起止 event 是否记录在实际执行工作的 stream 上 |
| 输出吞吐高但用户体验差 | TTFT/尾延迟、超时与取消、错误重试、输出长度和完成率是否被省略 |
| 两个方案无法公平比较 | 模型/tokenizer、prompt 分布、并行拓扑、量化 kernel、缓存命中、warmup 和停止条件是否一致 |

分位数也要连着样本数与算法读。假设 10 个请求中 9 个 TTFT 为 0.3 秒、1 个为 3.0 秒，均值是 0.57 秒；若采用 nearest-rank，P95 是 3.0 秒。换成插值算法，P95 会得到另一个数。因此报告应固定算法并保留请求级样本，不能把不同工具给出的 P95 当成同一口径。

诊断时先沿请求时间线查看队列、GPU、CPU、显存和网络 trace，再判断模型或 runtime 是否是瓶颈。p95 变差可能来自长 prompt、排队、锁竞争或网络，不只来自 kernel。

### 6.2.9 资料与证据边界

- PyTorch CUDA 同步（2.14.0）：<https://docs.pytorch.org/docs/2.14/generated/torch.cuda.synchronize.html>
- PyTorch CUDA Event（2.14.0）：<https://docs.pytorch.org/docs/2.14/generated/torch.cuda.Event.html>
- PyTorch Profiler（2.14.0）：<https://docs.pytorch.org/docs/2.14/profiler.html>
- vLLM metrics（v0.32.0）：<https://docs.vllm.ai/en/v0.32.0/usage/metrics/>
- NVIDIA CUDA C Best Practices Guide（13.0）：<https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-best-practices-guide/index.html>

延迟公式是测量定义，不是某个框架的专属接口。官方文档可以支持同步、event 和 metrics 的使用方式；具体 p95、吞吐和显存数字必须来自目标硬件、目标版本和代表性负载。

本篇已经把“等了多久”拆成可比较的量。下一篇从多个请求同时运行的角度，解释 KV block、continuous batching 和服务协议怎样改变这条时间线。
