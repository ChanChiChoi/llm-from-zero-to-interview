# 第 28 章 Prefill、Decode、KV Cache 与推理资源画像

上一章讲了模型服务运行时。本章继续往下拆，讲大模型推理中最关键的三个概念：Prefill、Decode 和 KV Cache。

很多人说推理慢、吞吐低、GPU 利用率不高，但说不清慢在哪里。真正要理解推理平台，必须把一次生成请求拆成阶段，分别分析它们消耗什么资源。

先记住一句话：

> 大模型推理不是一次普通函数调用，而是由 prefill、decode 和 KV cache 管理共同决定的在线计算过程。

## 28.0 本讲资料边界与第二轮精修口径

本讲按 `WRITING_PLAN.md` 的第二轮要求做过资料校准。重点参考的是 vLLM 官方文档中 PagedAttention、KV cache block、prefix caching、TTFT / TPOT / prefill / decode / KV 指标的公开口径；TensorRT-LLM 官方文档中 paged KV cache、in-flight batching、chunked prefill、KV cache reuse、KV cache offload 和 disaggregated serving 的工程边界；SGLang 官方文档中 RadixAttention / cache、continuous batching、PD disaggregation 和 structured serving 的边界；以及 Hugging Face TGI 文档中 streaming、PagedAttention 和 Prometheus 指标的说明。

这些资料共同指向一个稳定事实：LLM 推理不是“一个 batch forward”这么简单。Prefill 处理输入上下文，通常决定首 token 等待；decode 逐 token 生成，通常决定输出流畅度；KV cache 是连接两者的运行时状态，决定显存容量、并发上限、长上下文成本和调度风险。

本章只抽象截至 2026-06 仍稳定的资源画像口径，不把某个框架的具体指标名、默认 block size、benchmark 数值、显卡型号或云实例配置写成通用标准。正文中的公式用于容量估算和面试表达，真实上线仍要用目标模型、tokenizer、runtime、硬件、量化方式和流量分布实测校准。

## 28.1 一次大模型生成请求发生了什么

用户发起一个请求：

```text
请解释 Transformer 的 attention 机制。
```

模型返回一段文本：

```text
Attention 机制可以理解为...
```

从 runtime 内部看，大致发生了这些步骤：

1. 接收请求。
2. tokenizer 把文本转成 token。
3. 模型处理输入 token。
4. 生成第一枚输出 token。
5. 继续逐个生成后续 token。
6. 达到停止条件后结束。
7. 回收 KV cache。

其中第 3 步主要对应 prefill，第 4 到第 6 步主要对应 decode。

## 28.2 Prefill 是什么

Prefill 是处理输入 prompt 的阶段。

假设输入有 2000 个 token，模型需要一次性处理这 2000 个 token，并为每一层 attention 生成对应的 KV cache。

Prefill 的输入是完整 prompt。

Prefill 的输出包括：

1. 下一 token 的 logits。
2. 每一层的 KV cache。

可以把 prefill 理解为：模型先读完整个题目，建立上下文。

## 28.3 Decode 是什么

Decode 是逐 token 生成输出的阶段。

生成第一个 token 后，模型会把这个 token 加入上下文，再生成下一个 token。这个过程不断重复，直到遇到停止符、达到最大输出长度，或者请求超时。

Decode 的特点是：

1. 每一步通常只生成一个 token。
2. 每一步都要读取已有 KV cache。
3. 输出越长，decode 步数越多。
4. streaming 返回主要发生在 decode 阶段。

可以把 decode 理解为：模型一边想，一边一个词一个词往外说。

## 28.4 Prefill 和 Decode 的核心差异

Prefill 和 decode 的资源特征不同。

| 阶段 | 主要处理对象 | 主要瓶颈 | 延迟影响 | 吞吐影响 |
| --- | --- | --- | --- | --- |
| Prefill | 输入 token | 计算 | 影响 TTFT | 影响输入吞吐 |
| Decode | 输出 token | 显存带宽、KV cache 访问 | 影响 TPOT | 影响输出吞吐 |

这张表非常重要。

如果输入很长，TTFT 可能很高。

如果输出很长，整体延迟和 GPU 占用时间会很高。

如果并发很高，KV cache 会成为主要瓶颈。

## 28.5 TTFT 和 TPOT

TTFT 是 Time To First Token，即从请求进入到第一个 token 返回的时间。

TTFT 通常包括：

1. 网关耗时。
2. 路由耗时。
3. 排队耗时。
4. tokenizer 耗时。
5. prefill 耗时。
6. 第一个 token 采样耗时。

TPOT 是 Time Per Output Token，即平均每生成一个输出 token 的耗时。

TPOT 通常受 decode 阶段影响更大。

用户体验上：

1. TTFT 决定“多久开始响应”。
2. TPOT 决定“输出流不流畅”。
3. 总延迟约等于 TTFT 加上输出 token 数乘以 TPOT。

粗略表达：

```math
T_{\mathrm{ttft}}=T_{\mathrm{gateway}}+T_{\mathrm{route}}+T_{\mathrm{queue}}+T_{\mathrm{tokenize}}+T_{\mathrm{prefill}}+T_{\mathrm{first}}
```

```math
T_{\mathrm{tpot}}=\frac{T_{\mathrm{decode}}+T_{\mathrm{sample}}+T_{\mathrm{stream}}}{N_{\mathrm{out}}}
```

端到端延迟可以粗略写成：

```math
T_{\mathrm{e2e}}\approx T_{\mathrm{ttft}}+N_{\mathrm{out}}T_{\mathrm{tpot}}
```

真实系统还要考虑网络、队列、sampling、后处理和 streaming flush。

## 28.6 KV Cache 是什么

Transformer 每层 attention 会计算 Query、Key、Value。

在自回归生成中，历史 token 的 Key 和 Value 不需要每一步重复计算，可以缓存起来。这就是 KV cache。

KV cache 的作用是：

1. 减少重复计算。
2. 加速 decode。
3. 支持长上下文生成。

没有 KV cache，每生成一个 token 都要重新处理全部历史上下文，成本会高得不可接受。

## 28.7 KV Cache 为什么占显存

KV cache 会保存每一层、每个 token、每个 head 的 Key 和 Value。

它的规模大致和这些因素相关：

1. batch size。
2. 序列长度。
3. 模型层数。
4. hidden size。
5. attention head 数。
6. 数据类型精度。

粗略理解：

```math
M_{\mathrm{kv}}=2LBSH_{\mathrm{kv}}D_hB_{\mathrm{elem}}
```

其中 `L` 是层数，`B` 是活跃 batch 或活跃序列数，`S` 是当前上下文 token 数，`H_kv` 是 KV head 数，`D_h` 是每个 head 的维度，`B_elem` 是每个元素的字节数。前面的 `2` 来自 Key 和 Value 两份缓存。

如果只看单个请求，可以写成：

```math
M_{\mathrm{kv,req}}=2LSH_{\mathrm{kv}}D_hB_{\mathrm{elem}}
```

如果可用于 KV cache 的显存预算是 `B_kv`，粗略并发上限是：

```math
N_{\mathrm{kv}}=\left\lfloor \frac{B_{\mathrm{kv}}}{M_{\mathrm{kv,req}}}\right\rfloor
```

这仍然是容量估算，不是精确 profiler。真实系统还要考虑 block size、碎片、prefix cache、请求取消、padding、量化、MLA / GQA / MQA 结构差异和 runtime 预留显存。

长上下文和高并发都会快速放大 KV cache 显存占用。

## 28.8 一个直观例子

假设有两个请求：

请求 A：输入 200 token，输出 100 token。

请求 B：输入 20000 token，输出 100 token。

它们的输出长度一样，但 B 的 TTFT 会明显更高，因为 prefill 要处理更长输入。

再看两个请求：

请求 C：输入 500 token，输出 50 token。

请求 D：输入 500 token，输出 2000 token。

C 和 D 的 prefill 成本接近，但 D 的 decode 时间远长于 C，还会更长时间占用 KV cache 和 GPU 资源。

所以不能只用 QPS 衡量推理负载。

必须看 token 维度。

## 28.9 推理资源画像

推理资源画像就是描述不同请求消耗哪些资源。

至少要看：

1. 输入 token 数。
2. 输出 token 数。
3. 上下文长度。
4. 并发数。
5. batch size。
6. prefill latency。
7. decode latency。
8. KV cache 占用。
9. GPU 利用率。
10. 显存带宽。
11. 网络和 streaming 开销。

一个推理平台如果不知道请求画像，就无法做正确调度。

## 28.10 为什么 QPS 不够用

两个系统都处理 100 QPS，但负载可能完全不同。

系统 A：每个请求输入 100 token，输出 50 token。

系统 B：每个请求输入 8000 token，输出 1000 token。

它们的 QPS 一样，但系统 B 的计算量、显存占用和响应时间远高于系统 A。

所以大模型推理更应该关注：

1. input tokens/s。
2. output tokens/s。
3. active sequences。
4. total tokens in KV cache。
5. cost per token。
6. SLO 下可承载吞吐。

QPS 只是表层指标。

## 28.11 Prefill 的优化方向

Prefill 优化常见方向：

1. 提高矩阵计算效率。
2. 使用更高效 kernel。
3. 优化 attention 实现。
4. prompt cache。
5. prefix cache。
6. 输入截断或压缩。
7. 分离 prefill 和 decode 资源池。
8. 对长 prompt 请求单独调度。

长上下文场景中，prefill 往往是 TTFT 的主要来源。

如果业务强依赖长文档、RAG、代码仓库上下文或多轮对话，prefill 优化会非常关键。

## 28.12 Decode 的优化方向

Decode 优化常见方向：

1. continuous batching。
2. PagedAttention。
3. KV cache 复用。
4. KV cache 量化。
5. speculative decoding。
6. 更高效 sampling。
7. 限制最大输出长度。
8. 分离短输出和长输出请求。
9. 优化 streaming flush 策略。

Decode 阶段的特点是每步计算粒度小、步骤多、频繁访问 KV cache。

这也是为什么 decode 不一定能把 GPU 算力打满，显存带宽和调度开销可能成为瓶颈。

## 28.13 KV Cache 的管理策略

KV cache 管理常见问题：

1. 如何分配。
2. 如何释放。
3. 如何避免碎片。
4. 如何处理长上下文。
5. 如何处理请求取消。
6. 如何处理超时。
7. 如何做前缀复用。
8. 如何在显存不足时驱逐。

常见策略包括：

1. 分块管理。
2. 分页管理。
3. 引用计数。
4. prefix cache。
5. LRU 驱逐。
6. 按租户或优先级隔离。
7. 显存水位控制。

KV cache 管理不好，系统会出现延迟抖动和 OOM。

## 28.14 PagedAttention 的直觉

PagedAttention 的直觉类似操作系统分页。

传统方式可能为每个请求预留连续 KV cache 空间，容易浪费和碎片化。

PagedAttention 把 KV cache 拆成固定大小的块，通过映射表管理逻辑序列和物理块。

好处是：

1. 减少显存浪费。
2. 支持更高并发。
3. 更容易复用前缀。
4. 更好处理不同长度请求。

你不需要一开始就掌握所有实现细节，但要理解它解决的是 KV cache 显存管理问题。

## 28.15 长上下文的特殊问题

长上下文会同时放大 prefill 和 KV cache 压力。

它会带来：

1. TTFT 增加。
2. KV cache 显存占用增加。
3. batch 中长度差异更大。
4. 短请求被长请求影响。
5. 成本显著上升。
6. p99 延迟变差。

长上下文优化手段包括：

1. prompt 压缩。
2. RAG 召回控制。
3. sliding window attention。
4. prefix cache。
5. 长短请求分队列。
6. 长上下文专用实例。
7. 限制上下文配额。

长上下文不是免费能力，平台必须显式治理。

## 28.16 多租户下的 KV Cache 问题

多租户环境中，KV cache 还涉及公平性和隔离。

问题包括：

1. 一个租户的长请求占满显存。
2. 高优先级业务被低优先级业务阻塞。
3. KV cache 驱逐影响用户体验。
4. cache 复用可能带来数据隔离风险。
5. 成本归因不清。

平台应该支持：

1. 租户级上下文长度限制。
2. 租户级并发限制。
3. 租户级 token 配额。
4. 优先级队列。
5. cache 隔离策略。
6. 详细用量统计。

多租户推理平台不能只追求整体吞吐，还要保证公平性和安全隔离。

## 28.17 Prefill/Decode 分离

有些系统会把 prefill 和 decode 分离到不同资源池。

原因是两者资源特征不同：

1. Prefill 更偏计算密集。
2. Decode 更偏显存带宽和 KV cache。
3. Prefill 请求长度差异大。
4. Decode 持续时间受输出长度影响。

分离后的可能架构：

```text
Request -> Prefill Pool -> KV Transfer -> Decode Pool -> Streaming Response
```

好处：

1. 更细粒度调度资源。
2. 长 prompt 不一定阻塞 decode。
3. 可按阶段独立扩缩容。
4. 更适合超大规模推理集群。

代价：

1. 系统复杂度增加。
2. KV cache 传输成本增加。
3. 调度和一致性更难。
4. 故障排查更复杂。

所以它更适合大规模和高优化场景，不一定适合所有团队。

## 28.18 如何定位推理慢

当用户说“模型慢”时，不要直接说加 GPU。

应该按阶段排查：

1. 请求是否在网关排队？
2. router 是否选择了拥塞实例？
3. runtime 队列长度是否过高？
4. 输入 token 是否变长？
5. prefill latency 是否上升？
6. output token 是否变长？
7. TPOT 是否上升？
8. KV cache 是否接近水位？
9. GPU 显存是否不足？
10. batch 策略是否变化？
11. streaming 是否被网络或客户端拖慢？

只有定位到阶段，才能做正确优化。

## 28.19 Prefill/Decode/KV 资源画像审计指标与最小 demo

如果要把这一章落到平台工程，最核心的动作是建立“推理资源画像”：每个请求不只记录 QPS，还要记录输入 token、输出 token、TTFT、TPOT、prefill、decode、KV cache、prefix cache、长上下文、租户和成本。

可以把一个推理资源画像样本写成：

```math
p_i=(x_i,y_i,c_i,q_i,f_i,d_i,k_i,b_i,r_i,s_i,t_i,o_i,z_i)
```

其中 `x_i` 是输入 token 数，`y_i` 是输出 token 数，`c_i` 是上下文长度，`q_i` 是 queue 和 scheduler 状态，`f_i` 是 prefill 阶段，`d_i` 是 decode 阶段，`k_i` 是 KV cache 状态，`b_i` 是 batching 状态，`r_i` 是 prefix / prompt cache 复用，`s_i` 是 streaming 状态，`t_i` 是租户和优先级，`o_i` 是观测指标，`z_i` 是最终门禁。

统一覆盖率可以写成：

```math
C_j=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[g_j(p_i)=1]
```

Prefill 阶段的 attention 交互量可以粗略理解为：

```math
A_{\mathrm{prefill}}\propto LBH_qS_{\mathrm{in}}^2
```

Decode 阶段对历史上下文的访问可以粗略理解为：

```math
A_{\mathrm{decode}}\propto LBH_qS_{\mathrm{ctx}}N_{\mathrm{out}}
```

其中 `H_q` 是 query head 数。这个表达不是 kernel 级 FLOPs 公式，而是帮助理解：长输入会显著推高 prefill，长输出会让 decode 访问 KV cache 的次数变多。

KV pressure 可以写成：

```math
R_{\mathrm{kv}}=\frac{M_{\mathrm{kv}}}{B_{\mathrm{kv}}}
```

prefix cache 的收益可以用节省的 prefill token 或 prefill 时间表示：

```math
R_{\mathrm{prefix}}=\frac{N_{\mathrm{hit\_tokens}}}{N_{\mathrm{input\_tokens}}}
```

最终资源画像门禁可以写成：

```math
G_{\mathrm{resource}}=\mathbf{1}\left[\min_j C_j\ge \tau_j \land T_{\mathrm{ttft,p95}}\le B_{\mathrm{ttft}} \land T_{\mathrm{tpot,p95}}\le B_{\mathrm{tpot}} \land R_{\mathrm{kv}}\le \rho_{\mathrm{kv}} \land P_0=0\right]
```

下面是一个 0 依赖 demo。它不会模拟真实 GPU kernel，而是把 Prefill、Decode、KV cache 和资源画像门禁变成可运行的审计表。

```python
# Prefill / Decode / KV Resource Profile Audit: 0-dependency teaching demo.
from copy import deepcopy


GATES = [
    "request_token_profile",
    "prefill_phase_accounting",
    "decode_phase_accounting",
    "ttft_tpot_contract",
    "kv_cache_formula_fit",
    "kv_capacity_admission",
    "paged_block_management",
    "prefix_cache_reuse",
    "long_context_policy",
    "tenant_kv_isolation",
    "continuous_batching_policy",
    "pd_disaggregation_fit",
    "streaming_backpressure_fit",
    "observability_phase_metrics",
    "cost_capacity_model",
    "resource_profile_gate",
]


def has_keys(obj, keys):
    return isinstance(obj, dict) and all(obj.get(k) for k in keys)


def percentile(values, p):
    values = sorted(values)
    idx = max(0, min(len(values) - 1, int(len(values) * p + 0.999999) - 1))
    return values[idx]


def kv_cache_mib(layers, batch, seq, kv_heads, head_dim, dtype_bytes):
    total_bytes = 2 * layers * batch * seq * kv_heads * head_dim * dtype_bytes
    return total_bytes / (1024 * 1024)


def per_token_kib(layers, kv_heads, head_dim, dtype_bytes):
    return 2 * layers * kv_heads * head_dim * dtype_bytes / 1024


def gate_results(profile):
    metrics = set(profile.get("metrics", []))
    kv = profile.get("kv_cache", {})
    results = {
        "request_token_profile": has_keys(profile.get("request_tokens"), ["input_distribution", "output_distribution", "context_length", "token_budget"]),
        "prefill_phase_accounting": has_keys(profile.get("prefill"), ["measured", "input_tokens", "chunking_policy", "prefill_queue"]),
        "decode_phase_accounting": has_keys(profile.get("decode"), ["measured", "output_tokens", "step_metrics", "decode_queue"]),
        "ttft_tpot_contract": has_keys(profile.get("slo"), ["ttft_p95", "tpot_p95", "e2e_p99"]),
        "kv_cache_formula_fit": has_keys(kv, ["layers", "kv_heads", "head_dim", "dtype_bytes", "active_sequences"]),
        "kv_capacity_admission": has_keys(kv, ["kv_budget_mib", "admission_control"]),
        "paged_block_management": has_keys(profile.get("paged"), ["block_size", "block_table", "free_list", "fragmentation_metric"]),
        "prefix_cache_reuse": has_keys(profile.get("prefix_cache"), ["cache_key", "hit_tokens_metric", "tenant_isolation"]),
        "long_context_policy": has_keys(profile.get("long_context"), ["max_context", "long_short_queue", "truncation_or_compression", "rag_budget"]),
        "tenant_kv_isolation": has_keys(profile.get("tenant"), ["tenant_quota", "priority", "cache_isolation", "cost_attribution"]),
        "continuous_batching_policy": has_keys(profile.get("batching"), ["continuous", "max_batch_tokens", "dynamic_join", "dynamic_exit"]),
        "pd_disaggregation_fit": has_keys(profile.get("pd"), ["enabled_or_rejected", "kv_transfer_budget", "failure_boundary"]),
        "streaming_backpressure_fit": has_keys(profile.get("streaming"), ["backpressure", "cancel", "flush_metric"]),
        "observability_phase_metrics": {"ttft", "tpot", "prefill", "decode", "queue", "kv_mib", "kv_pressure", "prefix_hit", "oom", "stream_flush"}.issubset(metrics),
        "cost_capacity_model": has_keys(profile.get("cost"), ["gpu_hour_usd", "tokens_in", "tokens_out", "kv_mib", "cache_saved_tokens"]),
        "resource_profile_gate": bool(profile.get("final_gate")),
    }
    return results


def audit_resource_profile(profiles):
    failed_cases = []
    failed_gates = set()
    for profile in profiles:
        bad = [name for name, ok in gate_results(profile).items() if not ok]
        if bad:
            failed_cases.append(profile["case"])
            failed_gates.update(bad)

    n = len(profiles)
    metrics = {}
    for gate in GATES:
        metrics[gate] = round(sum(gate_results(p)[gate] for p in profiles) / n, 3)

    return {
        "metrics": metrics,
        "failed_cases": failed_cases,
        "failed_gates": [g for g in GATES if g in failed_gates],
        "hard_blocker_count": len(failed_gates),
        "resource_profile_gate_pass": not failed_cases,
    }


complete = {
    "case": "complete_resource_profile_ok",
    "request_tokens": {
        "input_distribution": "prod_prompt_histogram",
        "output_distribution": "prod_output_histogram",
        "context_length": 4096,
        "token_budget": "tenant_model_budget",
    },
    "prefill": {
        "measured": True,
        "input_tokens": 8192,
        "chunking_policy": "chunk_long_prefill",
        "prefill_queue": "separate_or_weighted",
    },
    "decode": {
        "measured": True,
        "output_tokens": 512,
        "step_metrics": "per_token_latency",
        "decode_queue": "continuous_batching",
    },
    "slo": {"ttft_p95": 1800, "tpot_p95": 35, "e2e_p99": 6000},
    "kv_cache": {
        "layers": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "dtype_bytes": 2,
        "active_sequences": 12,
        "kv_budget_mib": 24576,
        "admission_control": True,
    },
    "paged": {
        "block_size": 16,
        "block_table": True,
        "free_list": True,
        "fragmentation_metric": True,
    },
    "prefix_cache": {
        "cache_key": "tenant_model_template_prefix",
        "hit_tokens_metric": True,
        "tenant_isolation": True,
    },
    "long_context": {
        "max_context": 32768,
        "long_short_queue": True,
        "truncation_or_compression": True,
        "rag_budget": True,
    },
    "tenant": {
        "tenant_quota": True,
        "priority": True,
        "cache_isolation": True,
        "cost_attribution": True,
    },
    "batching": {
        "continuous": True,
        "max_batch_tokens": 8192,
        "dynamic_join": True,
        "dynamic_exit": True,
    },
    "pd": {
        "enabled_or_rejected": "rejected_for_small_scale",
        "kv_transfer_budget": True,
        "failure_boundary": True,
    },
    "streaming": {"backpressure": True, "cancel": True, "flush_metric": True},
    "metrics": ["ttft", "tpot", "prefill", "decode", "queue", "kv_mib", "kv_pressure", "prefix_hit", "oom", "stream_flush"],
    "cost": {
        "gpu_hour_usd": 1.20,
        "tokens_in": 55000,
        "tokens_out": 12000,
        "kv_mib": 6144,
        "cache_saved_tokens": 8000,
    },
    "final_gate": True,
}


def make_bad(case, mutator):
    item = deepcopy(complete)
    item["case"] = case
    mutator(item)
    return item


profiles = [complete]
profiles.append(make_bad("token_profile_missing_bad", lambda p: p["request_tokens"].pop("output_distribution")))
profiles.append(make_bad("prefill_unmeasured_bad", lambda p: p["prefill"].pop("chunking_policy")))
profiles.append(make_bad("decode_unmeasured_bad", lambda p: p["decode"].pop("step_metrics")))
profiles.append(make_bad("slo_missing_tpot_bad", lambda p: p["slo"].pop("tpot_p95")))
profiles.append(make_bad("kv_formula_gap_bad", lambda p: p["kv_cache"].pop("kv_heads")))
profiles.append(make_bad("kv_admission_missing_bad", lambda p: p["kv_cache"].pop("admission_control")))
profiles.append(make_bad("paged_block_missing_bad", lambda p: p["paged"].pop("block_table")))
profiles.append(make_bad("prefix_cache_unsafe_bad", lambda p: p["prefix_cache"].pop("tenant_isolation")))
profiles.append(make_bad("long_context_unbounded_bad", lambda p: p["long_context"].pop("max_context")))
profiles.append(make_bad("tenant_quota_missing_bad", lambda p: p["tenant"].pop("tenant_quota")))
profiles.append(make_bad("continuous_batching_gap_bad", lambda p: p["batching"].pop("dynamic_exit")))
profiles.append(make_bad("pd_disaggregation_unbounded_bad", lambda p: p["pd"].pop("kv_transfer_budget")))
profiles.append(make_bad("streaming_backpressure_gap_bad", lambda p: p["streaming"].pop("backpressure")))
profiles.append(make_bad("observability_phase_gap_bad", lambda p: p["metrics"].remove("prefill")))
profiles.append(make_bad("cost_capacity_gap_bad", lambda p: p["cost"].pop("kv_mib")))
profiles.append(make_bad("resource_profile_gate_missing_bad", lambda p: p.update({"final_gate": False})))

latency_samples = [
    {"queue_ms": 40, "tokenize_ms": 20, "prefill_ms": 260, "first_ms": 30, "decode_ms": 900, "sample_ms": 90, "stream_ms": 60, "input_tokens": 2048, "output_tokens": 60},
    {"queue_ms": 65, "tokenize_ms": 30, "prefill_ms": 420, "first_ms": 35, "decode_ms": 1250, "sample_ms": 120, "stream_ms": 85, "input_tokens": 4096, "output_tokens": 75},
    {"queue_ms": 85, "tokenize_ms": 42, "prefill_ms": 610, "first_ms": 40, "decode_ms": 1680, "sample_ms": 150, "stream_ms": 110, "input_tokens": 8192, "output_tokens": 90},
    {"queue_ms": 115, "tokenize_ms": 55, "prefill_ms": 950, "first_ms": 50, "decode_ms": 2100, "sample_ms": 200, "stream_ms": 150, "input_tokens": 12000, "output_tokens": 100},
    {"queue_ms": 140, "tokenize_ms": 70, "prefill_ms": 1300, "first_ms": 55, "decode_ms": 2600, "sample_ms": 240, "stream_ms": 200, "input_tokens": 16000, "output_tokens": 120},
]

ttfts = [x["queue_ms"] + x["tokenize_ms"] + x["prefill_ms"] + x["first_ms"] for x in latency_samples]
tpots = [(x["decode_ms"] + x["sample_ms"] + x["stream_ms"]) / x["output_tokens"] for x in latency_samples]
e2e = [ttfts[i] + latency_samples[i]["output_tokens"] * tpots[i] for i in range(len(latency_samples))]
prefill_tps = 1000 * sum(x["input_tokens"] for x in latency_samples) / sum(x["prefill_ms"] for x in latency_samples)
decode_tps = 1000 * sum(x["output_tokens"] for x in latency_samples) / sum(x["decode_ms"] for x in latency_samples)
kv_mib = kv_cache_mib(32, 12, 4096, 8, 128, 2)
per_request_kv_mib = kv_cache_mib(32, 1, 4096, 8, 128, 2)
cache_saved_prefill_ms = 1000 * complete["cost"]["cache_saved_tokens"] / prefill_tps
cost_per_1k = 1000 * (1.20 + 0.05 + 0.03) / (complete["cost"]["tokens_in"] + complete["cost"]["tokens_out"])
audit = audit_resource_profile(profiles)

resource_profile_examples = {
    "ttft_p95_ms": percentile(ttfts, 0.95),
    "tpot_p95_ms": round(percentile(tpots, 0.95), 1),
    "e2e_p99_ms": round(percentile(e2e, 0.99), 1),
    "prefill_tokens_per_second": round(prefill_tps, 1),
    "decode_output_tokens_per_second": round(decode_tps, 1),
    "kv_cache_mib": round(kv_mib, 1),
    "kv_pressure": round(kv_mib / complete["kv_cache"]["kv_budget_mib"], 3),
    "per_token_kv_kib": round(per_token_kib(32, 8, 128, 2), 1),
    "kv_concurrency_limit": int(complete["kv_cache"]["kv_budget_mib"] // per_request_kv_mib),
    "prefix_saved_prefill_ms": round(cache_saved_prefill_ms, 1),
    "cost_per_1k_tokens": round(cost_per_1k, 4),
}
smoke = {
    "complete_case_passes": not any(v is False for v in gate_results(complete).values()),
    "caught_prefill_gap": "prefill_unmeasured_bad" in audit["failed_cases"],
    "caught_decode_gap": "decode_unmeasured_bad" in audit["failed_cases"],
    "caught_kv_gap": "kv_admission_missing_bad" in audit["failed_cases"],
    "caught_long_context_gap": "long_context_unbounded_bad" in audit["failed_cases"],
    "caught_gate_gap": "resource_profile_gate_missing_bad" in audit["failed_cases"],
}

print(f"resource_profile_examples={resource_profile_examples}")
print(f"smoke={smoke}")
print(f"metrics={audit['metrics']}")
print(f"hard_blocker_count={audit['hard_blocker_count']}")
print(f"failed_cases={audit['failed_cases']}")
print(f"failed_gates={audit['failed_gates']}")
print(f"resource_profile_gate_pass={audit['resource_profile_gate_pass']}")
```

这个 demo 想说明：推理慢不是一个单点指标问题。你必须同时证明输入 / 输出 token 分布、prefill、decode、TTFT、TPOT、KV cache、分页块管理、prefix cache、长上下文、多租户、continuous batching、PD 分离、streaming、observability 和成本容量都可观测、可解释、可门禁。

## 28.20 面试常见追问

问题一：为什么长 prompt 会影响 TTFT？

可以回答：长 prompt 会增加 prefill 计算量，模型需要处理全部输入 token 并生成 KV cache，所以第一个 token 返回前的时间会变长。

问题二：为什么长输出会影响总延迟？

可以回答：输出是逐 token decode 的，输出越长 decode 步数越多，整体延迟约等于 TTFT 加 output tokens 乘以 TPOT。

问题三：为什么 KV cache 会成为瓶颈？

可以回答：KV cache 与并发数、序列长度、层数和 hidden size 相关，会占用大量显存。高并发和长上下文下，显存容量、显存带宽和 cache 管理都会成为瓶颈。

问题四：为什么不能只用 QPS 评估推理系统？

可以回答：不同请求 token 数差异巨大，同样 QPS 下输入输出 token 数可能相差几十倍。应该结合 input tokens/s、output tokens/s、TTFT、TPOT、KV cache 和成本来评估。

## 28.21 小练习

1. Prefill 和 decode 的区别是什么？
2. TTFT 和 TPOT 分别受哪些因素影响？
3. 为什么 KV cache 能加速 decode？
4. 为什么 KV cache 会占大量显存？
5. 长上下文会带来哪些平台问题？
6. 为什么高 QPS 不一定代表高负载？
7. Prefill/decode 分离有什么收益和代价？
8. 用户反馈“模型慢”时，你会按什么顺序排查？
9. 写一个推理资源画像表，至少包含 input tokens、output tokens、TTFT、TPOT、prefill、decode、KV cache、prefix cache、tenant、streaming 和 cost。
10. 为什么 prefix cache 命中率不能单独证明成本下降，还要看命中 token 数、节省 prefill 时间、权限隔离和错误复用风险？

## 28.22 本章小结

本章讲了 Prefill、Decode、KV Cache 和推理资源画像。

你需要记住：

1. Prefill 处理输入 prompt，主要影响 TTFT。
2. Decode 逐 token 生成输出，主要影响 TPOT 和总延迟。
3. KV cache 避免重复计算，但会消耗大量显存。
4. 长上下文会同时放大 prefill 成本和 KV cache 压力。
5. 大模型推理不能只看 QPS，要看 token 吞吐、延迟、cache 和成本。
6. 推理优化必须按阶段定位瓶颈，而不是盲目加机器。
7. 生产推理平台需要用资源画像门禁同时约束 prefill、decode、KV cache、长上下文、多租户、streaming 和成本。

下一章我们会继续讲 Continuous Batching、PagedAttention 和队列调度，理解 runtime 如何在高并发下提升吞吐并控制延迟。
