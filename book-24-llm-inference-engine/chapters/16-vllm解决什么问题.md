# 第 16 章 vLLM 解决什么问题

从本章开始进入第三部分：vLLM 架构详解。前面我们从 0 搭了一个 MiniEngine，也看到了它的瓶颈：KV Cache 管理粗糙、decode group 不能动态合并、finished 请求释放不彻底、padding 浪费、scheduler 简单、吞吐和显存都不理想。

vLLM 的价值，就是系统性解决这些 serving engine 的核心问题，尤其是 KV Cache 管理和 continuous batching。

一句话概括：

> vLLM 解决的核心问题是：如何在高并发 LLM serving 中高效管理 KV Cache，并通过 iteration-level scheduling 提高吞吐、降低显存浪费。

## 16.0 本讲资料边界与第二轮精修口径

本讲只解释 vLLM 为什么出现、主要解决哪些 serving engine 问题。它覆盖 naive MiniEngine 的 KV 预留浪费、连续内存约束、finished 请求释放、固定 batch decode 浪费、continuous batching 动机、PagedAttention 直觉和最小可运行问题归因 demo，但不展开 vLLM 源码细节、真实 block manager API、调度器参数全集、多卡 worker / executor、prefix caching、speculative decoding、生产部署和 OpenAI-compatible server 配置。

资料校准口径：

1. vLLM 论文与项目文档把 PagedAttention 作为核心机制，用 block-based KV cache 管理降低内存浪费，并支持更高吞吐 serving。
2. vLLM optimization 文档强调调优时要同时关注并发请求数、batch tokens、KV cache 容量、preemption、chunked prefill 和 decode / prefill 平衡。
3. vLLM metrics 文档把 running / waiting 请求数、KV cache usage、TTFT、TPOT、queue / prefill / decode 时间纳入观测，说明 vLLM 不是单个 kernel，而是一套 request scheduling + memory management + execution engine。
4. Orca 的 iteration-level scheduling 思想是 continuous batching 的重要背景：动态请求流不应被固定 batch 生命周期锁死。
5. 本章 demo 用 token 数模拟连续 KV 预留、paged block 分配、finished 请求释放和固定 batch / continuous batching 行数，不等同于真实 vLLM 性能。

参考资料：

1. vLLM paper / PagedAttention：<https://arxiv.org/abs/2309.06180>
2. vLLM documentation：<https://docs.vllm.ai/>
3. vLLM optimization and tuning：<https://docs.vllm.ai/en/latest/configuration/optimization.html>
4. vLLM metrics：<https://docs.vllm.ai/en/latest/design/metrics.html>
5. Orca iteration-level scheduling：<https://www.usenix.org/conference/osdi22/presentation/yu>

## 16.1 为什么要从 MiniEngine 过渡到 vLLM

第二部分的 MiniEngine 已经具备基本链路：

1. tokenizer。
2. model wrapper。
3. sampling。
4. KV Cache。
5. batched prefill。
6. batched decode。
7. request queue。
8. scheduler。
9. streaming。
10. HTTP API。

但它离生产级 serving engine 还有很远。

最典型的问题是：

1. KV Cache 直接依赖连续张量，难以动态管理。
2. finished 请求释放 cache 不优雅。
3. 新请求难以加入已有 decode batch。
4. 固定 batch 容易浪费计算。
5. 长短请求混合导致调度困难。
6. 显存碎片和预留浪费严重。
7. scheduler 缺少 token budget 和 cache budget。

vLLM 就是围绕这些问题设计出来的。

## 16.2 vLLM 的核心定位

vLLM 是一个面向大语言模型高吞吐推理的 serving engine。

它重点关注：

1. 高效的 KV Cache 管理。
2. PagedAttention。
3. Continuous Batching。
4. 高吞吐模型执行。
5. OpenAI-compatible serving。
6. 多模型和多并行配置支持。
7. Prefix caching 等推理优化。

如果用本书前面的三层边界来说，vLLM 主要属于 serving engine 层。

它不是完整 AI Infra，也不是完整企业推理平台。它可以提供 HTTP server 和 API 兼容能力，但它的核心价值仍然在模型执行、调度和内存管理。

## 16.3 vLLM 之前的痛点

传统推理实现容易遇到几个问题。

痛点一：KV Cache 显存浪费。

不同请求长度不同，如果每个请求预留一块连续 KV Cache，很容易出现内部浪费和碎片。

痛点二：batch 不够动态。

如果 batch 必须等一组请求全部结束，短请求会被长请求拖住，新请求也不能及时加入。

痛点三：显存限制并发。

模型权重固定，但 KV Cache 随并发和上下文增长。显存管理不好，最大并发会很低。

痛点四：吞吐和延迟难平衡。

大 batch 吞吐高但 TTFT 可能差，小 batch 延迟好但 GPU 利用率低。

痛点五：长上下文更难服务。

上下文越长，KV Cache 越大，调度和内存管理越关键。

vLLM 的架构创新主要围绕前两个痛点展开：KV Cache 和动态 batching。

可以先用几个简化公式看动机。设请求 `i` 的 prompt token 数为 `x_i`，最大输出 token 预算为 `m_i`，实际输出 token 数为 `y_i`。naive 连续 KV 预留 token 数可以写成：

```math
C_{\mathrm{reserve}}=\sum_i (x_i+m_i)
```

实际使用 token 数是：

```math
C_{\mathrm{used}}=\sum_i (x_i+y_i)
```

连续预留浪费率：

```math
R_{\mathrm{naive}}=\frac{C_{\mathrm{reserve}}-C_{\mathrm{used}}}{C_{\mathrm{reserve}}}
```

如果按 block size `S` 分块，paged KV 需要的 block 数为：

```math
B_i=\left\lceil\frac{x_i+y_i}{S}\right\rceil
```

paged KV 分配 token 数：

```math
C_{\mathrm{paged}}=S\sum_i B_i
```

paged KV 的块内浪费率：

```math
R_{\mathrm{paged}}=\frac{C_{\mathrm{paged}}-C_{\mathrm{used}}}{C_{\mathrm{paged}}}
```

固定 batch decode 的行数近似为：

```math
W_{\mathrm{static}}=N\max_i y_i
```

continuous batching 只处理仍然 active 的请求，decode 行数近似为：

```math
W_{\mathrm{cont}}=\sum_{t=1}^{T}A_t
```

其中 `A_t` 是第 `t` 轮仍在 decode 的请求数。vLLM 的动机门禁可以写成：

```math
G_{\mathrm{vllm}}=G_{\mathrm{kvblock}}G_{\mathrm{sched}}G_{\mathrm{reuse}}G_{\mathrm{cleanup}}G_{\mathrm{metrics}}
```

这些公式不等价于真实 vLLM 内部实现，但能解释它为什么要同时解决 KV block 管理、动态调度、block 复用、请求结束清理和指标观测。

## 16.4 PagedAttention 解决什么

PagedAttention 是 vLLM 最著名的机制。

它的直觉来自操作系统虚拟内存分页：不要要求一个请求的 KV Cache 在物理显存中连续存放，而是把 KV Cache 切成固定大小的 block，通过映射表把逻辑 token 位置映射到物理 block。

简化理解：

```text
request logical tokens: 0, 1, 2, 3, 4, 5, ...
logical blocks:         block 0, block 1, ...
physical blocks:        GPU block 17, GPU block 5, ...
```

好处是：

1. 减少连续分配要求。
2. 降低显存碎片。
3. 方便动态增长和释放。
4. 方便多个请求共享 prefix block。
5. 提升高并发场景下的显存利用率。

PagedAttention 不只是一个 attention kernel，而是一套 KV Cache 内存管理思想。

## 16.5 Continuous Batching 解决什么

Continuous batching，也叫 iteration-level scheduling，解决动态请求流问题。

传统 batch：

```text
batch A 开始 -> batch A 全部结束 -> batch B 开始
```

Continuous batching：

```text
iteration 1: r1, r2
iteration 2: r1, r2, r3
iteration 3: r2, r3, r4
iteration 4: r3, r4
```

每一轮 decode 都可以：

1. 移除完成请求。
2. 加入新请求。
3. 按 token budget 控制 prefill。
4. 按 cache budget 控制并发。
5. 让 GPU 尽量保持有效工作。

这比第二部分的固定 decode group 更灵活。

## 16.6 vLLM 和 MiniEngine 的对比

可以这样对比：

| 能力 | MiniEngine | vLLM |
| --- | --- | --- |
| KV Cache | 简单 `past_key_values` | block-based cache manager |
| 内存布局 | 连续 batch cache | PagedAttention |
| batch | 固定 group | continuous batching |
| 调度 | 简单 FIFO / 请求数限制 | token/cache budget + iteration-level scheduling |
| 请求退出 | 简单 finished 标记 | 释放 cache block，动态移出 |
| 新请求加入 | 新建 group | 可进入后续 iteration |
| 显存效率 | 较低 | 更高 |
| 生产能力 | 教学 | 生产级 serving engine |

这个对比能帮助你理解 vLLM 不是“更复杂的 generate”，而是把 MiniEngine 的痛点系统工程化解决。

## 16.7 vLLM 不只是 PagedAttention

很多人提到 vLLM 只记得 PagedAttention。

但真实 vLLM 还包括：

1. 请求生命周期管理。
2. Scheduler。
3. Block Manager。
4. Worker 和 Executor。
5. Model Runner。
6. Sampling。
7. Prefix Caching。
8. 多卡并行支持。
9. API Server。
10. Metrics 和配置系统。

PagedAttention 是核心创新，但不是全部系统。

面试里只说“vLLM 用 PagedAttention 提升吞吐”是不够的。要能讲请求如何进入、如何分配 block、如何调度 prefill/decode、如何释放 cache。

## 16.8 vLLM 的请求视角

从请求生命周期看，vLLM 大致做这些事：

```text
外部请求
  -> API 层解析
  -> 创建内部 sequence / sequence group
  -> 进入 waiting queue
  -> scheduler 选择 prefill 或 decode
  -> block manager 分配 KV blocks
  -> worker 执行模型 forward
  -> sampler 生成 token
  -> 更新 sequence 状态
  -> stream / return output
  -> finished 后释放 blocks
```

这和我们前面写的 MiniEngine 生命周期相同，只是每个环节更精细、更高效。

学习 vLLM 时，建议始终用这条链路做地图。

## 16.9 为什么 KV Cache 是 vLLM 的核心

LLM serving 的显存大头通常包括模型权重和 KV Cache。

模型权重加载后相对固定，KV Cache 随请求动态变化。

高并发时，真正决定能容纳多少请求的，经常是 KV Cache，而不是权重。

如果 KV Cache 管理粗糙：

1. 显存碎片会降低可用容量。
2. 长请求会占用大量连续空间。
3. 短请求结束后释放的空间难复用。
4. 并发上不去。
5. OOM 更容易发生。

vLLM 把 KV Cache 当成一级资源来管理，这是它和 naive serving 的根本区别。

## 16.10 为什么调度是 vLLM 的核心

只有 KV Cache 管理还不够。请求还需要被合理调度。

Scheduler 要决定：

1. 本轮处理哪些 waiting 请求。
2. 本轮继续 decode 哪些 running 请求。
3. prefill 和 decode 如何混合。
4. token budget 如何分配。
5. cache block 是否足够。
6. 请求完成后如何释放资源。
7. 长请求是否会拖慢短请求。

调度直接影响：

1. TTFT。
2. TPOT。
3. tokens/s。
4. GPU 利用率。
5. 公平性。
6. 显存稳定性。

因此 vLLM 的 scheduler 和 block manager 要一起理解，不能割裂看。

## 16.11 vLLM 适合什么场景

vLLM 适合：

1. 高并发在线文本生成。
2. OpenAI-compatible LLM serving。
3. 多用户共享模型服务。
4. 长短请求混合负载。
5. 需要 streaming 的聊天服务。
6. 需要较高吞吐和较好显存利用率的场景。

它不一定是所有场景的唯一选择。

例如，极致低延迟单请求、特定硬件上的 TensorRT-LLM 优化、复杂结构化生成和 agent runtime，也可能选择其他框架或组合方案。

工程上不要迷信某个框架，要理解它解决的问题和适用边界。

## 16.12 vLLM 与其他 runtime 的关系

常见 serving runtime 有：

1. vLLM。
2. SGLang。
3. TGI。
4. TensorRT-LLM。
5. Triton + 自研 backend。
6. llama.cpp 等本地推理 runtime。

vLLM 的优势在于通用 LLM serving、PagedAttention、continuous batching、生态和易用性。

SGLang 更强调 runtime、prefix sharing、structured generation、agent serving 等能力。

TensorRT-LLM 更偏 NVIDIA 生态内的高性能优化。

TGI 是 Hugging Face 生态常见 serving 方案。

本书先讲 vLLM，是因为它非常适合作为理解现代 serving engine 的主线样本。

## 16.13 如何阅读 vLLM 源码

阅读 vLLM 源码不要从所有文件开始。

建议按问题读：

1. 请求如何进入系统。
2. 内部如何表示 sequence。
3. scheduler 如何选择请求。
4. block manager 如何分配 KV Cache。
5. worker 如何执行模型。
6. sampler 如何返回 token。
7. 请求如何完成并释放 block。

读源码时要把类名放回生命周期，而不是孤立记忆。

例如看到 block table，就问：它是不是把逻辑 token block 映射到物理 KV block？看到 scheduler，就问：它这轮选择了哪些 prefill 和 decode？

## 16.14 vLLM 动机公式、内存浪费和可运行 demo

下面的 demo 不依赖外部库。它不实现真实 vLLM，而是用 toy request profiles 对比两件事：

1. naive serving 为每个请求连续预留 `prompt + max_new_tokens`，如果实际输出更短，会浪费 KV token。
2. paged KV cache 只按实际增长分配固定大小 block，finished / cancelled 请求释放 block 后可复用。
3. 固定 batch decode 要让短请求陪长请求跑到最长输出长度，continuous batching 只处理仍 active 的请求。

```python
from dataclasses import dataclass
from math import ceil


@dataclass
class RequestProfile:
    request_id: str
    prompt_tokens: int
    max_new_tokens: int
    actual_new_tokens: int
    finished: bool = True


def block_count(tokens, block_size):
    return ceil(tokens / block_size)


def run_vllm_motivation_demo():
    block_size = 16
    requests = [
        RequestProfile("r_short", 127, 128, 16),
        RequestProfile("r_chat", 513, 256, 92),
        RequestProfile("r_long", 2049, 512, 401),
        RequestProfile("r_cancel", 1023, 512, 34),
    ]

    rows = {}
    next_block = 0
    free_after_cancel = []
    for req in requests:
        reserved_tokens = req.prompt_tokens + req.max_new_tokens
        used_tokens = req.prompt_tokens + req.actual_new_tokens
        blocks = block_count(used_tokens, block_size)
        physical_blocks = list(range(next_block, next_block + blocks))
        next_block += blocks
        allocated_tokens = blocks * block_size
        rows[req.request_id] = {
            "reserved_tokens": reserved_tokens,
            "used_tokens": used_tokens,
            "paged_blocks": blocks,
            "paged_allocated_tokens": allocated_tokens,
            "logical_to_physical_head": physical_blocks[:3],
        }
        if req.request_id == "r_cancel":
            free_after_cancel = physical_blocks

    new_request_blocks = free_after_cancel[: block_count(64, block_size)]
    naive_reserved = sum(item["reserved_tokens"] for item in rows.values())
    actual_used = sum(item["used_tokens"] for item in rows.values())
    paged_allocated = sum(item["paged_allocated_tokens"] for item in rows.values())
    naive_waste_ratio = round((naive_reserved - actual_used) / naive_reserved, 3)
    paged_waste_ratio = round((paged_allocated - actual_used) / paged_allocated, 3)

    max_decode_steps = max(req.actual_new_tokens for req in requests)
    static_decode_rows = len(requests) * max_decode_steps
    continuous_decode_rows = sum(req.actual_new_tokens for req in requests)
    saved_rows = static_decode_rows - continuous_decode_rows
    row_saving_ratio = round(saved_rows / static_decode_rows, 3)

    memory_summary = {
        "naive_reserved_tokens": naive_reserved,
        "actual_used_tokens": actual_used,
        "paged_allocated_tokens": paged_allocated,
        "naive_waste_ratio": naive_waste_ratio,
        "paged_waste_ratio": paged_waste_ratio,
        "freed_blocks_after_cancel": len(free_after_cancel),
        "new_request_reused_blocks": new_request_blocks,
    }
    batch_summary = {
        "static_decode_rows": static_decode_rows,
        "continuous_decode_rows": continuous_decode_rows,
        "saved_rows": saved_rows,
        "row_saving_ratio": row_saving_ratio,
    }
    gates = {
        "kv_waste_reduced": paged_waste_ratio < naive_waste_ratio,
        "block_reuse_visible": new_request_blocks == free_after_cancel[:4],
        "continuous_saves_rows": saved_rows > 0,
        "finished_cleanup_visible": len(free_after_cancel) == rows["r_cancel"]["paged_blocks"],
        "motivation_classified": naive_waste_ratio > 0.1 and row_saving_ratio > 0.5,
    }
    gates["vllm_motivation_gate"] = all(gates.values())
    return rows, memory_summary, batch_summary, gates


rows, memory_summary, batch_summary, gates = run_vllm_motivation_demo()
print("request_memory_rows=", rows)
print("memory_summary=", memory_summary)
print("batch_summary=", batch_summary)
print("vllm_motivation_gates=", gates)
```

一组稳定输出：

```text
request_memory_rows= {'r_short': {'reserved_tokens': 255, 'used_tokens': 143, 'paged_blocks': 9, 'paged_allocated_tokens': 144, 'logical_to_physical_head': [0, 1, 2]}, 'r_chat': {'reserved_tokens': 769, 'used_tokens': 605, 'paged_blocks': 38, 'paged_allocated_tokens': 608, 'logical_to_physical_head': [9, 10, 11]}, 'r_long': {'reserved_tokens': 2561, 'used_tokens': 2450, 'paged_blocks': 154, 'paged_allocated_tokens': 2464, 'logical_to_physical_head': [47, 48, 49]}, 'r_cancel': {'reserved_tokens': 1535, 'used_tokens': 1057, 'paged_blocks': 67, 'paged_allocated_tokens': 1072, 'logical_to_physical_head': [201, 202, 203]}}
memory_summary= {'naive_reserved_tokens': 5120, 'actual_used_tokens': 4255, 'paged_allocated_tokens': 4288, 'naive_waste_ratio': 0.169, 'paged_waste_ratio': 0.008, 'freed_blocks_after_cancel': 67, 'new_request_reused_blocks': [201, 202, 203, 204]}
batch_summary= {'static_decode_rows': 1604, 'continuous_decode_rows': 543, 'saved_rows': 1061, 'row_saving_ratio': 0.661}
vllm_motivation_gates= {'kv_waste_reduced': True, 'block_reuse_visible': True, 'continuous_saves_rows': True, 'finished_cleanup_visible': True, 'motivation_classified': True, 'vllm_motivation_gate': True}
```

这个 demo 的关键证据：

1. naive 连续预留需要 `5120` 个 token 容量，实际只用了 `4255`，浪费率 `0.169`。
2. paged block 分配只需要 `4288` 个 token 容量，块内浪费率 `0.008`。
3. `r_cancel` 结束后释放了 `67` 个 block，新请求可以复用 `[201,202,203,204]`。
4. 固定 batch decode 需要 `1604` 行，continuous batching 近似只需要 `543` 行。
5. 这同时解释了为什么 vLLM 不能只理解成 PagedAttention kernel；它的动机是 KV block、调度、复用、cleanup 和 metrics 的系统组合。

## 16.15 常见误区

误区一：vLLM 只是比 `model.generate()` 快。

vLLM 的核心是 serving engine：KV Cache 管理、调度、batching、worker/executor 和 streaming，不是简单加速单次 generate。

误区二：PagedAttention 就是全部 vLLM。

PagedAttention 很重要，但还要理解 scheduler、block manager、worker、sampler 和请求生命周期。

误区三：continuous batching 就是普通 batch。

普通 batch 是固定批次，continuous batching 是每轮动态加入和移除请求。

误区四：显存够放模型就够了。

高并发 serving 还要考虑 KV Cache，vLLM 的很多设计正是为了解决 cache 显存问题。

误区五：用了 vLLM 就不需要平台层。

vLLM 是 engine，生产还需要推理平台做路由、限流、灰度、观测、成本和多租户治理。

## 16.16 面试追问

1. vLLM 主要解决什么问题？
2. 为什么 KV Cache 管理是 LLM serving 的核心？
3. PagedAttention 的直觉是什么？
4. Continuous batching 和静态 batching 有什么区别？
5. vLLM 和一个 naive MiniEngine 的差距在哪里？
6. 为什么 scheduler 和 block manager 要一起理解？
7. vLLM 是否等于完整推理平台？为什么？
8. 什么场景下你会考虑 vLLM，什么场景下可能考虑其他 runtime？

参考回答思路：

1. 先说 vLLM 是高吞吐 LLM serving engine。
2. 再说两个核心：PagedAttention 管 KV Cache，continuous batching 管动态请求调度。
3. 然后解释它解决显存碎片、并发、吞吐和动态 batch 问题。
4. 最后补充边界：vLLM 是 engine，不等于完整平台，仍需平台和 infra 配合。

## 16.17 小练习

1. 用一张图画出 MiniEngine 和 vLLM 在 KV Cache 管理上的差异。
2. 解释为什么固定 decode group 会限制吞吐。
3. 写出 continuous batching 的 4 轮示例，包含新请求加入和旧请求退出。
4. 列出 vLLM 中你最想读的 5 个模块，并说明每个模块解决什么问题。
5. 思考如果没有 PagedAttention，高并发长上下文 serving 会遇到哪些问题。

## 16.18 本章小结

本章开启 vLLM 架构详解。

vLLM 解决的不是“如何调用模型生成文本”这么简单的问题，而是高并发 LLM serving 的系统问题：KV Cache 如何高效管理，请求如何动态调度，finished 请求如何释放资源，新请求如何加入运行中的 batch，显存如何更高效利用。

下一章我们会深入 PagedAttention，讲清它如何借鉴分页内存思想，把 KV Cache 从连续大张量管理升级成 block-based 管理。
