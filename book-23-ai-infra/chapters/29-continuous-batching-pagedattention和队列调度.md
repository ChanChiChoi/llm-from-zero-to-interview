# 第 29 章 Continuous Batching、PagedAttention 和队列调度

上一章讲了 Prefill、Decode 和 KV Cache。本章继续深入高并发推理中的三个关键机制：Continuous Batching、PagedAttention 和队列调度。

如果说上一章回答的是“推理请求消耗什么资源”，本章回答的是“runtime 怎样把很多请求组织起来，提高吞吐并控制延迟”。

先记住一句话：

> 大模型推理的核心挑战不是单个请求能不能跑，而是大量长短不一的请求如何在有限 GPU 和显存下公平、高效、低延迟地一起跑。

## 29.0 本讲资料边界与第二轮精修口径

本讲按 `WRITING_PLAN.md` 的第二轮要求做过资料校准。重点参考的是 vLLM 的 PagedAttention / KV cache block / scheduler 公开资料和论文口径，TensorRT-LLM 对 in-flight batching、paged KV cache、chunked prefill 和调度策略的工程说明，SGLang 对 continuous batching、RadixAttention / cache 和 serving runtime 的公开说明，以及 Hugging Face TGI 对 streaming、PagedAttention 和 Prometheus 指标的公开说明。

这些资料共同指向一个稳定事实：LLM serving 的 batching 不是传统“凑一批请求然后一起跑完”，而是围绕 decode iteration 动态维护 running set、waiting queue、prefill chunk、KV block、streaming 连接和租户配额。PagedAttention 也不是新的模型结构或新的 attention 数学公式，而是把逻辑连续的 KV cache 映射到物理上可分散管理的 blocks，从而降低碎片、支持动态调度和 prefix / block 级复用。

本章只抽象截至 2026-06 仍稳定的调度和容量画像口径，不把某个 runtime 的默认 block size、调度参数、指标字段名、benchmark 数值或云实例配置写成通用标准。正文中的公式用于面试表达、容量估算和系统设计推导；真实上线仍要用目标模型、tokenizer、runtime、硬件、量化方式、请求 token 分布和 SLO 实测校准。

## 29.1 为什么需要 batching

GPU 擅长并行计算。如果一次只处理一个请求，GPU 很可能吃不满。

Batching 的基本思想是：把多个请求合并成一个 batch，一起执行模型前向计算。

这样可以：

1. 提高 GPU 利用率。
2. 提高 tokens/s。
3. 降低单位 token 成本。
4. 减少调度开销。

但 batching 不是越大越好。batch 越大，请求可能等待越久，TTFT 可能变差，显存占用也会增加。

推理系统要在吞吐和延迟之间做权衡。

## 29.2 静态 batching 的问题

传统 batching 可以理解为：收集一批请求，凑够 batch 后一起执行。

例如：

```text
等待 8 个请求 -> 组成 batch -> 一起执行 -> 全部结束 -> 下一批
```

这种方式对分类模型、embedding 模型、CV 模型比较常见。

但对生成式大模型有明显问题。

原因是每个请求输出长度不同：

1. 有的请求生成 20 token 就结束。
2. 有的请求生成 2000 token 才结束。
3. 有的请求中途被用户取消。
4. 有的请求遇到 stop sequence 提前结束。

如果静态 batch 要等所有请求结束，短请求会被长请求拖住。

## 29.3 Continuous Batching 的直觉

Continuous Batching 也叫动态 batching 或 iteration-level batching。

它的核心思想是：batch 不是一次固定到结束，而是在每个 decode step 动态调整。

可以理解为：

```text
step 1: A B C D 在 batch 中
step 2: A B C D 继续生成
step 3: A 完成，E 加入
step 4: B 完成，F 加入
step 5: C D E F 继续生成
```

请求可以动态进入，也可以动态退出。

这非常适合大模型生成，因为输出长度天然不一致。

## 29.4 Continuous Batching 的收益

Continuous Batching 的收益包括：

1. 减少 GPU 空转。
2. 短请求不用一直等长请求。
3. 提高 decode 阶段吞吐。
4. 更好处理不同输出长度。
5. 支持高并发在线服务。

它把“按请求批处理”变成了“按生成步调度”。

这就是 LLM runtime 和传统模型 serving 很不一样的地方。

## 29.5 Continuous Batching 的代价

Continuous Batching 不是免费的。

它会带来：

1. 调度器更复杂。
2. KV cache 管理更复杂。
3. 每个 step 都要维护活跃请求集合。
4. 请求取消和超时处理更复杂。
5. streaming 返回顺序管理更复杂。
6. 指标统计更复杂。

因此 runtime 需要一个成熟的 scheduler 和 KV cache manager。

## 29.6 PagedAttention 解决什么问题

上一章已经讲过，KV cache 会占用大量显存。

不同请求长度不同，如果每个请求都预留连续显存空间，会出现两个问题：

1. 显存浪费。
2. 显存碎片。

PagedAttention 的直觉是：把 KV cache 拆成固定大小的 block，用类似页表的结构管理。

逻辑序列不需要对应一段连续物理显存，而是映射到多个 KV block。

如果 block size 是 $s_{\mathrm{blk}}$，请求 `i` 当前上下文长度是 $L_i$，则它大约需要：

```math
N_{\mathrm{blk},i}=\left\lceil \frac{L_i}{s_{\mathrm{blk}}}\right\rceil
```

个 KV blocks。若所有活跃请求集合为 $\mathcal{R}$，则总 block 占用为：

```math
N_{\mathrm{blk,total}}=\sum_{i\in \mathcal{R}} N_{\mathrm{blk},i}
```

分页管理仍然会有尾部浪费，粗略浪费率可以写成：

```math
R_{\mathrm{waste}}=
\frac{\sum_{i\in \mathcal{R}}\left(N_{\mathrm{blk},i}s_{\mathrm{blk}}-L_i\right)}
{\sum_{i\in \mathcal{R}}N_{\mathrm{blk},i}s_{\mathrm{blk}}}
```

这说明 block size 不是越小越好，也不是越大越好。小 block 降低尾部浪费，但 block table、元数据和访存管理更复杂；大 block 管理简单，但短请求和请求尾部会浪费更多 KV 空间。

这样能更好支持：

1. 不同长度请求。
2. 动态进入和退出。
3. prefix 共享。
4. 更高并发。
5. 更低显存浪费。

## 29.7 PagedAttention 和 Continuous Batching 的关系

Continuous Batching 要求请求可以动态加入和退出。

这意味着 KV cache 也要能动态分配、释放和复用。

PagedAttention 提供了更灵活的 KV cache 管理方式，让 continuous batching 更容易高效运行。

可以这样理解：

1. Continuous Batching 解决“请求如何动态组 batch”。
2. PagedAttention 解决“动态 batch 下 KV cache 如何高效管理”。
3. Scheduler 解决“谁先执行、谁等待、谁被拒绝”。

三者经常一起出现。

## 29.8 队列调度为什么重要

高并发推理服务不是请求来了就立刻执行。

请求通常会进入队列。

队列调度决定：

1. 哪些请求先执行。
2. 哪些请求进入 prefill。
3. 哪些请求进入 decode。
4. batch 里放多少请求。
5. 长请求是否要限速。
6. 高优先级请求是否插队。
7. 低优先级请求是否等待或拒绝。

调度策略会直接影响 TTFT、TPOT、吞吐、公平性和成本。

## 29.9 一个简单调度流程

一个 runtime scheduler 可能这样工作：

```text
1. 接收新请求
2. 估算输入 token 和最大输出 token
3. 检查上下文长度限制
4. 检查 KV cache 是否有空间
5. 将请求放入 waiting queue
6. 选择部分请求进入 prefill
7. 将 prefill 完成的请求放入 running set
8. 对 running set 执行 decode step
9. 完成的请求释放 KV cache
10. 新请求补入 batch
```

这个流程看似简单，但每一步都有工程细节。

## 29.10 调度目标之间的冲突

推理调度有多个目标：

1. 高吞吐。
2. 低 TTFT。
3. 低 TPOT。
4. 低 p99 延迟。
5. 高 GPU 利用率。
6. 高公平性。
7. 低拒绝率。
8. 低成本。

这些目标经常冲突。

例如：

1. 更大 batch 提高吞吐，但可能增加等待时间。
2. 优先短请求降低平均延迟，但可能饿死长请求。
3. 优先高价值租户提升业务收益，但影响公平性。
4. 尽量填满 GPU 提高利用率，但可能拉高 p99。

所以调度不是一个“最佳参数”，而是一组策略权衡。

## 29.11 常见队列策略

常见调度策略包括：

1. FIFO。
2. shortest-job-first。
3. priority queue。
4. deadline-aware scheduling。
5. tenant-aware scheduling。
6. prefill/decode 分队列。
7. 长短请求分队列。
8. token budget scheduling。
9. fairness scheduling。

FIFO 最简单，但容易被长请求拖慢。

shortest-job-first 可以降低平均延迟，但可能伤害长请求公平性。

priority queue 适合多租户和业务优先级，但要防止低优先级长期饥饿。

token budget scheduling 更适合 LLM，因为 token 数比请求数更能代表负载。

## 29.12 Token Budget 调度

Token budget 调度的核心是：每轮执行不只限制请求数，还限制 token 预算。

例如：

```text
max_batch_tokens = 8192
max_active_sequences = 64
```

可以把一轮调度写成：

```math
\sum_{i\in \mathcal{P}_t} n_i^{\mathrm{prefill}}+\sum_{i\in \mathcal{D}_t} n_i^{\mathrm{decode}}\le B_{\mathrm{tok}}
```

其中 $\mathcal{P}_t$ 是第 `t` 轮被选中做 prefill 的请求集合，$\mathcal{D}_t$ 是第 `t` 轮被选中做 decode 的请求集合，$n_i^{\mathrm{prefill}}$ 是本轮给请求 `i` 处理的 prompt token 数，$n_i^{\mathrm{decode}}$ 通常是 1 或少量输出 token，$B_{\mathrm{tok}}$ 是本轮 token budget。

同时还要限制活跃序列数：

```math
|\mathcal{R}_t|\le B_{\mathrm{seq}}
```

其中 $\mathcal{R}_t$ 是 running set，$B_{\mathrm{seq}}$ 是最大活跃 sequence 数。直觉上，`max_batch_tokens` 控制本轮算力和排队压力，`max_active_sequences` 控制 decode 并发和 KV cache 压力。只看 batch size 会低估长 prompt 和长上下文请求的成本。

调度器选择请求时同时考虑：

1. 输入 token 数。
2. 当前上下文长度。
3. 预计输出 token 数。
4. KV cache 剩余空间。
5. 活跃 sequence 数。

这样比单纯按 batch size 更合理。

因为一个长上下文请求可能比几十个短请求还重。

## 29.13 长短请求混跑的问题

长请求和短请求混跑会带来 head-of-line blocking。

问题表现：

1. 短请求 TTFT 被拉高。
2. 长请求占用 KV cache 很久。
3. batch 内长度差异导致调度效率下降。
4. p99 延迟变差。

常见解决方案：

1. 长短请求分队列。
2. 长上下文专用实例。
3. 对长请求限流。
4. 对短请求保留容量。
5. 按 token 预算调度。
6. 设置最大上下文长度和最大输出长度。

生产系统通常不会让所有请求无差别混在一个队列里。

## 29.14 Prefill 和 Decode 的调度冲突

Prefill 和 decode 会争用 GPU。

如果 prefill 太多，decode 可能变慢，TPOT 变差。

如果 decode 占满资源，新请求 prefill 进不来，TTFT 变差。

调度器要在两者之间平衡。

常见方法包括：

1. 限制每轮 prefill token 数。
2. 限制新请求进入速度。
3. 给 decode 保留执行窗口。
4. prefill/decode 分离资源池。
5. 对长 prefill 请求单独调度。

如果业务要求流式输出顺滑，decode 不能被 prefill 长时间阻塞。

## 29.15 公平性调度

多租户推理平台必须考虑公平性。

否则一个租户可能用长上下文、大并发、大输出占满所有资源。

公平性可以从多个维度定义：

1. 请求数公平。
2. token 数公平。
3. GPU 时间公平。
4. 成本公平。
5. SLO 公平。

对于大模型，token 数和 GPU 时间通常比请求数更合理。

平台可以设置：

1. 租户级并发上限。
2. 租户级 token/s 上限。
3. 租户级上下文长度上限。
4. 租户级优先级。
5. 租户级预算。

如果用 token 作为公平性口径，租户 `u` 在一个时间窗口内的资源占比可以粗略写成：

```math
S_u=\frac{N_u^{\mathrm{in}}+\alpha N_u^{\mathrm{out}}}{\sum_v\left(N_v^{\mathrm{in}}+\alpha N_v^{\mathrm{out}}\right)}
```

其中 $N_u^{\mathrm{in}}$ 是租户 `u` 的输入 token 数，$N_u^{\mathrm{out}}$ 是输出 token 数，$\alpha$ 是输出 token 相对输入 token 的成本权重。实际系统还会把 KV cache、GPU seconds、优先级和 SLO 合进来，但这个公式能提醒你：请求数公平通常不等于资源公平。

公平性不是道德问题，而是稳定性问题。

## 29.16 Admission Control

Admission control 是决定请求是否允许进入系统。

如果系统已经没有足够资源，盲目接收请求只会让所有请求都变慢。

Admission control 可以检查：

1. 队列长度。
2. 等待时间。
3. KV cache 剩余空间。
4. GPU 负载。
5. 租户配额。
6. 请求 token 数。
7. 当前 p99 延迟。
8. 错误率和超时率。

处理方式包括：

1. 接收。
2. 排队。
3. 降级。
4. 重定向到其他实例。
5. 返回限流错误。

对在线系统来说，及时拒绝比无限排队更健康。

## 29.17 Streaming 对调度的影响

大模型服务常用 streaming 返回。

Streaming 会影响调度和资源管理：

1. 请求连接持续时间更长。
2. decode 过程需要持续输出 token。
3. 客户端断开要及时释放资源。
4. 慢客户端可能影响发送缓冲。
5. 指标要区分生成耗时和传输耗时。

runtime 要能处理请求取消和客户端断开，否则 KV cache 和运行资源可能泄漏。

## 29.18 指标设计

调度相关指标至少包括：

1. waiting queue length。
2. running requests。
3. active sequences。
4. scheduled tokens per step。
5. prefill tokens/s。
6. decode tokens/s。
7. batch size 分布。
8. batch token 分布。
9. KV cache 使用率。
10. admission reject count。
11. queue wait time。
12. TTFT。
13. TPOT。
14. p95 / p99 latency。
15. tenant-level token usage。

没有这些指标，调度策略只能靠猜。

## 29.19 常见故障模式

Continuous batching 和队列调度常见故障包括：

1. 长请求拖慢所有短请求。
2. batch 太大导致 TTFT 升高。
3. batch 太小导致 GPU 利用率低。
4. KV cache 接近水位后频繁拒绝请求。
5. 高优先级请求插队导致低优先级饥饿。
6. prefill 抢占 decode，streaming 输出卡顿。
7. 请求取消后资源没有释放。
8. 指标只看 QPS，无法解释 p99 抖动。

这些问题都不是简单加 GPU 就能解决的。

## 29.20 Continuous Batching / PagedAttention 调度审计指标与最小 demo

做推理调度系统设计时，可以把一个请求样本抽象成：

```math
s_i=(a_i,u_i,p_i,x_i,y_i,q_i,r_i,b_i,k_i,c_i,m_i,z_i)
```

其中 $a_i$ 是到达时间，$u_i$ 是租户，$p_i$ 是优先级，$x_i$ 是输入 token 数，$y_i$ 是最大输出 token 数，$q_i$ 是队列等待策略，$r_i$ 是 running / waiting / finished 状态，$b_i$ 是 KV block 占用，$k_i$ 是是否命中 prefix / KV 复用，$c_i$ 是取消或超时处理，$m_i$ 是指标记录，$z_i$ 是最终门禁结果。

对每个审计维度 `j` 定义一个检查函数 $g_j(s_i)\in\{0,1\}$，覆盖率可以写成：

```math
C_j=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[g_j(s_i)=1]
```

最终可以用一个调度门禁收束：

```math
G_{\mathrm{sched}}=\mathbf{1}\left[\min_j C_j\ge \tau_j \land T_{\mathrm{ttft,p95}}\le B_{\mathrm{ttft}} \land T_{\mathrm{tpot,p95}}\le B_{\mathrm{tpot}} \land R_{\mathrm{kv}}\le \rho_{\mathrm{kv}} \land R_{\mathrm{reject}}\le \rho_{\mathrm{reject}} \land P_0=0\right]
```

下面这个 0 依赖 demo 做两件事：第一，模拟一个小型 continuous batching scheduler，展示请求动态到达、短请求优先、长 prompt 被 chunked prefill、running set 动态退出、KV block 被释放；第二，用审计表检查一个推理调度系统是否覆盖 arrival profile、continuous batching、token budget、prefill/decode 平衡、PagedAttention block、KV admission、长短请求隔离、多租户公平、取消释放、streaming 背压、指标和最终门禁。

```python
# Continuous batching / PagedAttention scheduler audit: 0-dependency teaching demo.
from copy import deepcopy
from math import ceil


GATES = [
    "request_arrival_profile",
    "continuous_batching_core",
    "token_budget_policy",
    "prefill_decode_balance",
    "paged_kv_block_management",
    "kv_capacity_admission",
    "long_short_isolation",
    "tenant_priority_fairness",
    "cancellation_cleanup",
    "streaming_backpressure",
    "observability_metrics",
    "scheduler_gate",
]


def has_keys(obj, keys):
    return isinstance(obj, dict) and all(obj.get(k) for k in keys)


def percentile(values, p):
    values = sorted(values)
    idx = max(0, min(len(values) - 1, int(len(values) * p + 0.999999) - 1))
    return values[idx]


def gate_results(profile):
    metrics = set(profile.get("metrics", []))
    results = {
        "request_arrival_profile": has_keys(profile.get("arrival"), ["input_tokens", "output_tokens", "arrival_time", "tenant"]),
        "continuous_batching_core": has_keys(profile.get("batching"), ["dynamic_join", "dynamic_exit", "iteration_level", "max_active_sequences"]),
        "token_budget_policy": has_keys(profile.get("token_budget"), ["max_batch_tokens", "max_prefill_chunk", "decode_token_budget"]),
        "prefill_decode_balance": has_keys(profile.get("phase_policy"), ["decode_first", "chunked_prefill", "prefill_queue", "decode_queue"]),
        "paged_kv_block_management": has_keys(profile.get("paged_kv"), ["block_size", "block_table", "free_list", "fragmentation_metric"]),
        "kv_capacity_admission": has_keys(profile.get("admission"), ["kv_block_budget", "queue_timeout", "reject_policy", "watermark"]),
        "long_short_isolation": has_keys(profile.get("long_short"), ["long_prompt_threshold", "separate_queue", "reserved_short_capacity"]),
        "tenant_priority_fairness": has_keys(profile.get("tenant"), ["quota", "priority", "fairness_metric", "cost_attribution"]),
        "cancellation_cleanup": has_keys(profile.get("cancel"), ["cancel_signal", "release_blocks", "timeout_release"]),
        "streaming_backpressure": has_keys(profile.get("streaming"), ["flush_metric", "backpressure", "client_disconnect"]),
        "observability_metrics": {"queue_wait", "ttft", "tpot", "scheduled_tokens", "active_sequences", "kv_blocks", "kv_waste", "rejects", "cancellations", "tenant_tokens"}.issubset(metrics),
        "scheduler_gate": bool(profile.get("final_gate")),
    }
    return results


def audit_scheduler_profiles(profiles):
    failed_cases = []
    failed_gates = set()
    for profile in profiles:
        bad = [name for name, ok in gate_results(profile).items() if not ok]
        if bad:
            failed_cases.append(profile["case"])
            failed_gates.update(bad)

    n = len(profiles)
    metrics = {gate: round(sum(gate_results(p)[gate] for p in profiles) / n, 3) for gate in GATES}
    return {
        "metrics": metrics,
        "failed_cases": failed_cases,
        "failed_gates": [g for g in GATES if g in failed_gates],
        "hard_blocker_count": len(failed_gates),
        "scheduler_gate_pass": not failed_cases,
    }


complete = {
    "case": "complete_scheduler_profile_ok",
    "arrival": {"input_tokens": "histogram", "output_tokens": "histogram", "arrival_time": "timestamp", "tenant": "tenant_id"},
    "batching": {"dynamic_join": True, "dynamic_exit": True, "iteration_level": True, "max_active_sequences": 64},
    "token_budget": {"max_batch_tokens": 4096, "max_prefill_chunk": 1024, "decode_token_budget": 256},
    "phase_policy": {"decode_first": True, "chunked_prefill": True, "prefill_queue": "weighted", "decode_queue": "running_set"},
    "paged_kv": {"block_size": 16, "block_table": True, "free_list": True, "fragmentation_metric": True},
    "admission": {"kv_block_budget": 4096, "queue_timeout": 30, "reject_policy": "fast_fail_or_degrade", "watermark": 0.90},
    "long_short": {"long_prompt_threshold": 4096, "separate_queue": True, "reserved_short_capacity": True},
    "tenant": {"quota": True, "priority": True, "fairness_metric": "token_share", "cost_attribution": True},
    "cancel": {"cancel_signal": True, "release_blocks": True, "timeout_release": True},
    "streaming": {"flush_metric": True, "backpressure": True, "client_disconnect": True},
    "metrics": ["queue_wait", "ttft", "tpot", "scheduled_tokens", "active_sequences", "kv_blocks", "kv_waste", "rejects", "cancellations", "tenant_tokens"],
    "final_gate": True,
}


def make_bad(case, mutator):
    item = deepcopy(complete)
    item["case"] = case
    mutator(item)
    return item


profiles = [complete]
profiles.append(make_bad("arrival_profile_missing_bad", lambda p: p["arrival"].pop("output_tokens")))
profiles.append(make_bad("static_batching_bad", lambda p: p["batching"].pop("dynamic_exit")))
profiles.append(make_bad("token_budget_missing_bad", lambda p: p["token_budget"].pop("max_batch_tokens")))
profiles.append(make_bad("prefill_decode_unbalanced_bad", lambda p: p["phase_policy"].pop("chunked_prefill")))
profiles.append(make_bad("paged_kv_no_freelist_bad", lambda p: p["paged_kv"].pop("free_list")))
profiles.append(make_bad("kv_admission_missing_bad", lambda p: p["admission"].pop("watermark")))
profiles.append(make_bad("long_short_mixed_bad", lambda p: p["long_short"].pop("separate_queue")))
profiles.append(make_bad("tenant_fairness_missing_bad", lambda p: p["tenant"].pop("fairness_metric")))
profiles.append(make_bad("cancel_leaks_blocks_bad", lambda p: p["cancel"].pop("release_blocks")))
profiles.append(make_bad("streaming_backpressure_missing_bad", lambda p: p["streaming"].pop("backpressure")))
profiles.append(make_bad("scheduler_metrics_missing_bad", lambda p: p["metrics"].remove("kv_waste")))
profiles.append(make_bad("scheduler_gate_missing_bad", lambda p: p.update({"final_gate": False})))


def blocks_for(tokens, block_size):
    return int(ceil(tokens / block_size)) if tokens > 0 else 0


def simulate_continuous_batching(requests, max_steps=14, max_batch_tokens=1024, max_active=3, block_size=16, kv_block_budget=960, max_prefill_chunk=384):
    reqs = {r["id"]: dict(r, prompt_done=0, output_done=0, first_scheduled=None, first_decode=None, finish_step=None, blocks=0) for r in requests}
    waiting = []
    running = []
    free_blocks = kv_block_budget
    trace = []
    finished_order = []

    def allocate(req, new_tokens):
        nonlocal free_blocks
        need = blocks_for(new_tokens, block_size)
        delta = need - req["blocks"]
        if delta <= 0:
            return True
        if delta > free_blocks:
            return False
        req["blocks"] = need
        free_blocks -= delta
        return True

    for step in range(max_steps):
        for req in reqs.values():
            if req["arrival"] == step:
                waiting.append(req["id"])

        budget = max_batch_tokens
        actions = []

        for rid in list(running):
            if budget <= 0:
                break
            req = reqs[rid]
            if not allocate(req, req["prompt_done"] + req["output_done"] + 1):
                continue
            req["output_done"] += 1
            req["first_decode"] = req["first_decode"] if req["first_decode"] is not None else step
            budget -= 1
            actions.append(f"decode:{rid}")
            if req["output_done"] >= req["max_output"]:
                running.remove(rid)
                free_blocks += req["blocks"]
                req["blocks"] = 0
                req["finish_step"] = step
                finished_order.append(rid)
                actions.append(f"finish:{rid}")

        def waiting_key(rid):
            req = reqs[rid]
            is_long = req["input_tokens"] >= 2048
            return (req["priority"], is_long, req["arrival"], req["input_tokens"])

        for rid in sorted(list(waiting), key=waiting_key):
            if budget <= 0 or len(running) >= max_active:
                break
            req = reqs[rid]
            remaining = req["input_tokens"] - req["prompt_done"]
            chunk = min(remaining, max_prefill_chunk, budget)
            if chunk <= 0:
                continue
            if not allocate(req, req["prompt_done"] + chunk):
                continue
            req["prompt_done"] += chunk
            req["first_scheduled"] = req["first_scheduled"] if req["first_scheduled"] is not None else step
            budget -= chunk
            actions.append(f"prefill:{rid}:{chunk}")
            if req["prompt_done"] >= req["input_tokens"]:
                waiting.remove(rid)
                running.append(rid)
                actions.append(f"running:{rid}")

        active_ids = waiting + running
        used_blocks = kv_block_budget - free_blocks
        capacity_tokens = sum(reqs[rid]["blocks"] * block_size for rid in active_ids)
        actual_tokens = sum(reqs[rid]["prompt_done"] + reqs[rid]["output_done"] for rid in active_ids)
        waste = (capacity_tokens - actual_tokens) / capacity_tokens if capacity_tokens else 0.0
        trace.append({
            "step": step,
            "actions": actions,
            "running": list(running),
            "waiting": list(waiting),
            "used_blocks": used_blocks,
            "kv_waste": round(waste, 3),
        })
        if not waiting and not running and all(r["arrival"] <= step for r in reqs.values()):
            break

    queue_waits = [r["first_scheduled"] - r["arrival"] for r in reqs.values() if r["first_scheduled"] is not None]
    ttft_steps = [r["first_decode"] - r["arrival"] for r in reqs.values() if r["first_decode"] is not None]
    active_counts = [len(x["running"]) + len(x["waiting"]) for x in trace]
    long_prompt_chunks = sum(1 for x in trace for a in x["actions"] if a.startswith("prefill:B") or a.startswith("prefill:E"))
    return {
        "iterations": len(trace),
        "finished_order": finished_order,
        "avg_queue_wait_steps": round(sum(queue_waits) / len(queue_waits), 2),
        "p95_ttft_steps": percentile(ttft_steps, 0.95),
        "p95_active_sequences": percentile(active_counts, 0.95),
        "max_kv_blocks_used": max(x["used_blocks"] for x in trace),
        "max_kv_waste_ratio": max(x["kv_waste"] for x in trace),
        "long_prompt_chunks": long_prompt_chunks,
        "trace_tail": trace[-4:],
    }


requests = [
    {"id": "A", "tenant": "free", "arrival": 0, "input_tokens": 512, "max_output": 3, "priority": 2},
    {"id": "B", "tenant": "pro", "arrival": 0, "input_tokens": 2304, "max_output": 4, "priority": 2},
    {"id": "C", "tenant": "free", "arrival": 1, "input_tokens": 384, "max_output": 2, "priority": 2},
    {"id": "D", "tenant": "enterprise", "arrival": 1, "input_tokens": 768, "max_output": 3, "priority": 0},
    {"id": "E", "tenant": "pro", "arrival": 2, "input_tokens": 2048, "max_output": 2, "priority": 1},
]

scheduler_examples = simulate_continuous_batching(requests)
audit = audit_scheduler_profiles(profiles)
smoke = {
    "complete_case_passes": not any(v is False for v in gate_results(complete).values()),
    "caught_static_batching": "static_batching_bad" in audit["failed_cases"],
    "caught_paged_kv_gap": "paged_kv_no_freelist_bad" in audit["failed_cases"],
    "caught_cancel_leak": "cancel_leaks_blocks_bad" in audit["failed_cases"],
    "caught_metrics_gap": "scheduler_metrics_missing_bad" in audit["failed_cases"],
    "caught_gate_gap": "scheduler_gate_missing_bad" in audit["failed_cases"],
}

print(f"scheduler_examples={scheduler_examples}")
print(f"smoke={smoke}")
print(f"metrics={audit['metrics']}")
print(f"hard_blocker_count={audit['hard_blocker_count']}")
print(f"failed_cases={audit['failed_cases']}")
print(f"failed_gates={audit['failed_gates']}")
print(f"scheduler_gate_pass={audit['scheduler_gate_pass']}")
```

这个 demo 想说明：Continuous Batching、PagedAttention 和队列调度不是三个孤立名词。请求动态加入退出需要 scheduler；scheduler 要靠 token budget 和 phase policy 平衡 TTFT / TPOT；动态请求又要求 KV block 能按需分配、释放和统计碎片；多租户环境还必须把长短请求隔离、公平性、取消释放、streaming 背压和最终上线门禁放进同一个审计表。

## 29.21 面试常见追问

问题一：Continuous Batching 和普通 batching 有什么区别？

可以回答：普通 batching 通常一批请求固定执行到结束；Continuous Batching 在每个 decode step 动态调整活跃请求集合，完成的请求退出，新请求加入，更适合输出长度不一致的生成式模型。

问题二：PagedAttention 解决什么问题？

可以回答：它通过类似分页的方式管理 KV cache，减少显存浪费和碎片，提高长短不一请求下的并发能力，也更适合 dynamic batching。

问题三：为什么调度不能只按请求数？

可以回答：大模型请求的 token 数差异巨大，长 prompt 或长输出请求的资源消耗远高于短请求。按 token budget、KV cache 和 SLO 调度更合理。

问题四：如何避免长请求拖慢短请求？

可以回答：可以做长短请求分队列、长上下文专用实例、token budget 调度、限制最大上下文和输出长度，并为短请求保留容量。

## 29.22 小练习

1. 为什么生成式模型不适合简单静态 batching？
2. Continuous Batching 的核心思想是什么？
3. PagedAttention 和 KV cache 管理有什么关系？
4. 为什么 token budget 比 batch size 更适合 LLM 调度？
5. Prefill 和 decode 会如何互相影响？
6. 多租户推理平台如何设计公平性？
7. Admission control 为什么重要？
8. 如果 p99 延迟突然升高，你会查看哪些调度指标？

## 29.23 本章小结

本章讲了 Continuous Batching、PagedAttention 和队列调度。

你需要记住：

1. Batching 提高吞吐，但会影响延迟。
2. 静态 batching 不适合输出长度差异大的生成式模型。
3. Continuous Batching 允许请求按 decode step 动态加入和退出。
4. PagedAttention 通过分页式 KV cache 管理减少显存浪费和碎片。
5. 调度策略要同时考虑 TTFT、TPOT、吞吐、KV cache、公平性和成本。
6. 大模型调度应更多关注 token budget，而不是只看请求数。
7. Admission control、长短请求分队列和多租户公平性是生产推理平台的关键能力。

下一章我们会讲模型路由：模型选择、能力路由、成本路由和降级路由。
