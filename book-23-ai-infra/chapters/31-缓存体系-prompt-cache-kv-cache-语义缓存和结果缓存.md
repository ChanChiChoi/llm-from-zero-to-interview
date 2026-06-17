# 第 31 章 缓存体系：prompt cache、KV cache、语义缓存和结果缓存

上一章讲了模型路由。本章讲推理平台中另一个非常关键的能力：缓存体系。

大模型推理成本高、延迟高，很多请求又存在重复、相似或共享前缀。如果完全每次从头算，GPU 成本会迅速失控。缓存的目标就是：在不明显损害正确性和安全性的前提下，减少重复计算、降低延迟和成本。

先记住一句话：

> 推理缓存不是一个缓存，而是一组位于不同层次、解决不同问题的缓存机制。

## 31.0 本讲资料边界与第二轮精修口径

本讲按 `WRITING_PLAN.md` 的第二轮要求做过资料校准。重点参考的是 OpenAI Prompt Caching 对长 prompt 前缀复用的公开说明，vLLM Automatic Prefix Caching 对 KV block hash、prefix reuse 和 cache hit 的工程口径，TensorRT-LLM 对 paged KV cache、KV cache reuse、retention 和 eviction 的公开说明，以及 RedisVL SemanticCache 对 embedding 相似检索、阈值、TTL 和过滤条件的工程抽象。

这些资料共同指向一个稳定事实：大模型推理缓存不是单个 Redis key-value 缓存，而是跨平台层、runtime 层和业务层的一组缓存策略。prompt / prefix cache 主要减少重复 prefill，KV cache 主要管理生成过程中的 attention 状态，语义缓存用相似检索减少近似重复请求，结果缓存则复用完整响应。四者的 key、生命周期、隔离风险和收益指标都不同。

本章只抽象截至 2026-06 仍稳定的缓存体系设计口径，不把某个云服务的具体折扣、最小 token 门槛、框架默认 block size、向量库参数或缓存字段名写成通用标准。正文公式用于面试表达、容量估算和策略审计；真实上线仍要用目标模型、tokenizer、runtime、请求分布、租户权限、知识库版本、采样参数和安全策略实测校准。

## 31.1 为什么大模型推理需要缓存

推理平台里的缓存主要解决四类问题：

1. 降低 TTFT。
2. 降低 TPOT。
3. 降低 GPU 计算成本。
4. 降低重复请求的端到端延迟。

常见重复来源包括：

1. 相同系统 prompt。
2. 相同文档前缀。
3. 相同 RAG 检索结果。
4. 相似用户问题。
5. 热门 FAQ。
6. 重试请求。
7. 多 Agent 复用同一上下文。

没有缓存，平台会反复为相同或相似内容付费。

## 31.2 四类常见缓存

大模型推理平台常见缓存包括：

1. prompt cache。
2. KV cache。
3. 语义缓存。
4. 结果缓存。

它们的位置不同，命中条件不同，风险也不同。

| 缓存类型 | 缓存内容 | 主要收益 | 主要风险 |
| --- | --- | --- | --- |
| prompt cache | prompt 前缀或 tokenized prompt | 降低 prefill 成本 | 前缀匹配和权限隔离 |
| KV cache | attention 的 Key/Value | 加速 decode 和上下文复用 | 显存占用和隔离风险 |
| 语义缓存 | 相似问题到答案或中间结果 | 减少近似重复请求 | 误命中导致答非所问 |
| 结果缓存 | 完整请求到完整结果 | 最快返回 | 过期、个性化和安全风险 |

## 31.3 Prompt Cache 是什么

Prompt cache 缓存的是 prompt 的可复用部分。

例如很多请求都有相同系统提示词：

```text
你是一个企业客服助手，请遵守以下规则...
```

或者 RAG 场景中多个请求共享相同文档前缀：

```text
以下是产品手册内容：...
请基于手册回答用户问题：...
```

如果前缀相同，runtime 或平台可以复用前缀处理结果，从而减少 prefill 开销。

Prompt cache 主要改善 TTFT。

## 31.4 Prompt Cache 的命中条件

Prompt cache 通常要求：

1. prompt 前缀完全一致，或满足某种规范化后一致。
2. tokenizer 一致。
3. 模型版本一致。
4. 推理配置兼容。
5. 权限边界一致。
6. 安全策略一致。

模型版本不同，KV 或 prompt 处理结果通常不能直接复用。

租户不同，也不能随便复用，因为 prompt 里可能包含私有信息。

## 31.5 Prefix Cache

Prefix cache 是 prompt cache 的常见形式，缓存共享前缀。

适合场景：

1. 固定 system prompt。
2. 固定工具说明。
3. 固定长文档。
4. 固定代码上下文。
5. 多轮对话共享历史。

例如两个请求：

```text
prefix: 公司产品手册全文
query A: 这个产品如何退款？
query B: 这个产品支持哪些地区？
```

如果产品手册部分相同，就可以复用前缀计算。

Prefix cache 对长上下文尤其有价值。

## 31.6 KV Cache 回顾

KV cache 是 Transformer attention 中 Key 和 Value 的缓存。

在生成过程中，历史 token 的 Key 和 Value 不需要每一步重复计算，所以缓存起来。

KV cache 的主要作用是：

1. 加速 decode。
2. 支持长上下文。
3. 支持 continuous batching。
4. 支持 prefix 复用。

但它也会带来显存压力。

KV cache 更接近 runtime 内部缓存，而不是传统意义上的业务缓存。

## 31.7 KV Cache 的生命周期

一个请求的 KV cache 生命周期通常是：

```text
请求进入 -> prefill 生成 KV cache -> decode 持续追加 -> 请求结束 -> 释放 KV cache
```

如果启用 prefix cache，部分 KV cache 可能在请求结束后继续保留，供后续请求复用。

生命周期管理要处理：

1. 正常结束。
2. 超时结束。
3. 用户取消。
4. 客户端断开。
5. runtime 异常。
6. 显存水位过高。
7. cache 驱逐。

如果释放不及时，会造成显存泄漏。

## 31.8 语义缓存是什么

语义缓存不是要求请求完全相同，而是判断问题语义相似。

例如：

```text
问题 A：怎么申请退款？
问题 B：我想退货退款，流程是什么？
```

这两个问题字面不同，但语义接近。语义缓存可能直接返回已有答案，或者复用中间结果。

实现方式通常是：

1. 对用户问题做 embedding。
2. 在向量索引中检索相似历史请求。
3. 如果相似度超过阈值，返回缓存答案或候选答案。
4. 如果不够相似，走正常模型推理。

## 31.9 语义缓存的收益和风险

语义缓存的收益：

1. 降低重复问题成本。
2. 显著降低延迟。
3. 对 FAQ 类场景很有效。
4. 可以减少大模型调用次数。

语义缓存的风险：

1. 相似但不等价。
2. 答案过期。
3. 用户上下文不同。
4. 权限不同。
5. 个性化需求不同。
6. 安全策略不同。

例如“我的订单能退吗”和“退款规则是什么”很相似，但前者可能需要用户订单信息，不能直接复用通用答案。

## 31.10 结果缓存是什么

结果缓存缓存完整请求到完整输出。

例如：

```text
cache_key = hash(model + version + prompt + parameters)
cache_value = generated_answer
```

如果下次完全相同的请求到来，可以直接返回结果。

结果缓存适合：

1. 确定性生成。
2. FAQ。
3. 后台批处理。
4. 重试请求。
5. 固定 prompt 的分类或抽取任务。

如果 temperature 较高、输出本身要求多样性，结果缓存就不一定适合。

## 31.11 缓存 Key 如何设计

缓存 key 设计非常关键。

常见字段包括：

1. 模型名称。
2. 模型版本。
3. tokenizer 版本。
4. prompt 或 prompt hash。
5. system prompt hash。
6. 参数配置。
7. temperature。
8. top_p。
9. max_tokens。
10. stop sequence。
11. 租户 ID。
12. 权限域。
13. 安全策略版本。
14. 数据版本。

不要只用 prompt 文本做 key。

否则模型版本、参数、权限或安全策略变化后，可能错误命中旧结果。

## 31.12 缓存粒度

缓存粒度可以从粗到细：

1. 完整响应缓存。
2. 中间检索结果缓存。
3. prompt 前缀缓存。
4. tokenizer 结果缓存。
5. KV cache block 缓存。

粒度越粗，命中后收益越大，但命中条件更严格，错误风险也更高。

粒度越细，复用机会更多，但管理复杂度更高。

平台通常会组合多层缓存，而不是只依赖一种。

## 31.13 缓存与安全隔离

缓存最容易被忽视的是安全隔离。

必须避免：

1. A 租户命中 B 租户缓存。
2. 普通用户命中管理员上下文缓存。
3. 低权限用户拿到高权限检索结果。
4. 私有文档前缀被跨用户复用。
5. 敏感输出被结果缓存长期保存。

安全策略：

1. cache key 包含租户和权限域。
2. 私有数据默认不跨租户复用。
3. 敏感数据设置短 TTL 或不缓存。
4. 缓存内容加密存储。
5. 访问缓存也要审计。
6. 明确 cache purge 机制。

缓存是性能优化，但不能破坏数据边界。

## 31.14 缓存与一致性

缓存可能返回过期答案。

常见过期来源：

1. 模型版本升级。
2. prompt 模板升级。
3. 文档知识库更新。
4. 安全策略更新。
5. 业务规则更新。
6. 用户权限变化。

应对方式：

1. cache key 加版本。
2. 设置 TTL。
3. 主动失效。
4. 按数据版本隔离。
5. 高风险场景禁用结果缓存。
6. 对命中结果做二次校验。

缓存一致性不是数据库一致性，但同样需要治理。

## 31.15 缓存与随机性

大模型生成可能有随机性。

如果 temperature 大于 0，同一个 prompt 可能生成不同答案。

这时结果缓存要谨慎。

可选策略：

1. 只缓存 temperature 为 0 的请求。
2. cache key 包含 sampling 参数。
3. 对创意生成禁用结果缓存。
4. 对分类、抽取、改写等任务启用缓存。
5. 对缓存命中返回标记。

缓存适合确定性或近似确定性任务，不适合所有生成任务。

## 31.16 缓存与流式输出

流式输出下，结果缓存有两种方式：

1. 缓存完整结果，下次模拟 streaming 分片返回。
2. 不缓存完整结果，只缓存前缀或 KV。

第一种对用户体验更一致，但要注意：

1. chunk 节奏是否需要模拟。
2. 是否暴露缓存命中。
3. 客户端取消时如何计费。
4. trace 如何记录。

流式场景中，prompt cache 和 KV cache 通常比完整结果缓存更常用。

## 31.17 缓存驱逐策略

缓存空间有限，需要驱逐。

常见策略：

1. LRU。
2. LFU。
3. TTL。
4. 按成本收益驱逐。
5. 按租户配额驱逐。
6. 按安全等级驱逐。
7. 按模型版本驱逐。

KV cache 驱逐尤其复杂，因为它占的是 GPU 显存，资源昂贵且变化快。

结果缓存和语义缓存通常可以放在内存、Redis、向量库或对象存储中，驱逐策略相对灵活。

## 31.18 缓存指标

缓存体系至少要观察：

1. hit rate。
2. miss rate。
3. cache latency。
4. saved tokens。
5. saved GPU time。
6. cache memory usage。
7. KV cache usage。
8. eviction count。
9. stale hit count。
10. semantic false hit rate。
11. tenant-level hit rate。
12. cost saved。

只看 hit rate 不够。

一个缓存命中率很高，但命中的都是低成本短请求，实际收益可能不大。

更应该关注节省了多少 token、GPU 时间和成本。

## 31.19 缓存策略如何落地

一个实际平台可以这样设计缓存策略：

1. 对 system prompt 和工具说明启用 prefix cache。
2. 对固定知识库版本启用 prompt cache。
3. 对 FAQ 场景启用语义缓存。
4. 对 temperature 为 0 的分类和抽取任务启用结果缓存。
5. 对私有数据默认只在租户内缓存。
6. 对高风险请求禁用跨请求缓存。
7. 对所有缓存命中记录 trace。
8. 定期评估缓存误命中和成本收益。

缓存策略应该按场景开启，而不是全局一刀切。

## 31.20 缓存体系审计指标与最小 demo

第二轮精修时，需要把缓存体系从“哪些地方可以缓存”升级成“哪些缓存可以安全命中、命中后节省多少、错误命中如何阻断、最终能否过门禁”。

可以把一次缓存查找写成：

```math
c_i=(r_i,m_i,v_i,t_i,p_i,d_i,u_i,a_i,s_i,\tau_i,z_i)
```

其中，`r_i` 是请求内容或请求 hash，`m_i` 是模型名，`v_i` 是模型版本，`t_i` 是 tokenizer 版本，`p_i` 是生成参数，`d_i` 是数据或知识库版本，`u_i` 是租户，`a_i` 是权限域，`s_i` 是安全策略版本，`\tau_i` 是 TTL / 时间戳信息，`z_i` 是 trace 和审计字段。

缓存命中率可以写成：

```math
H_k=\frac{N_{\mathrm{hit},k}}{N_{\mathrm{lookup},k}}
```

其中，`k` 表示缓存层，例如 result、semantic、prefix 或 KV。只看 `H_k` 不够，因为一个高命中率的短请求缓存可能节省不了多少成本。

更有意义的是 token 节省量：

```math
S_{\mathrm{tok}}=\sum_{i=1}^{N}\mathbf{1}[h_i=1](n_i^{\mathrm{save}}+o_i^{\mathrm{save}})
```

其中，`h_i` 表示第 `i` 次查找是否命中，`n_i^{\mathrm{save}}` 是节省的输入 token 或 prefill token，`o_i^{\mathrm{save}}` 是节省的输出 token。prefix cache 主要节省输入 token 的 prefill，结果缓存和安全的语义缓存可能同时节省输入和输出。

成本收益可以写成：

```math
K_{\mathrm{save}}=\sum_i \frac{n_i^{\mathrm{save}}p_{\mathrm{in}}+o_i^{\mathrm{save}}p_{\mathrm{out}}}{1000}-K_{\mathrm{cache}}
```

其中，`p_{\mathrm{in}}` 和 `p_{\mathrm{out}}` 是每千 input / output token 成本，`K_{\mathrm{cache}}` 是缓存存储、向量检索、网络、反序列化和运维成本。这个公式强调：缓存收益要扣掉缓存系统自身成本。

prefix cache 的 prefill 时间收益可以粗略写成：

```math
T_{\mathrm{prefill,save}}=\frac{N_{\mathrm{prefix,hit}}}{Q_{\mathrm{prefill}}}
```

其中，`N_{\mathrm{prefix,hit}}` 是命中的前缀 token 数，`Q_{\mathrm{prefill}}` 是 prefill tokens/s。真实系统还要考虑 batch、硬件、KV block 布局、cache read latency 和调度等待。

语义缓存需要单独监控误命中率：

```math
R_{\mathrm{false}}=\frac{N_{\mathrm{false\_semantic\_hit}}}{N_{\mathrm{semantic\_hit}}}
```

其中，`N_{\mathrm{false\_semantic\_hit}}` 是相似但不等价、权限不一致、上下文不同或知识过期的语义命中数。语义缓存的关键不是“相似度越高越好”，而是相似度、权限、数据版本、任务类型和风险等级共同过线。

过期命中率可以写成：

```math
R_{\mathrm{stale}}=\frac{N_{\mathrm{stale\_hit}}}{N_{\mathrm{hit}}}
```

租户或权限隔离违规率可以写成：

```math
R_{\mathrm{iso}}=\frac{N_{\mathrm{cross\_domain\_hit}}}{N_{\mathrm{hit}}}
```

这两个指标通常应该接近 0。缓存事故里，过期答案和跨权限命中比低命中率更危险。

最终缓存门禁可以写成：

```math
G_{\mathrm{cache}}=\mathbf{1}\left[\min_j C_j\ge\tau_j \land R_{\mathrm{false}}\le\rho_{\mathrm{false}} \land R_{\mathrm{stale}}\le\rho_{\mathrm{stale}} \land R_{\mathrm{iso}}=0 \land K_{\mathrm{save}}>0 \land P_0=0\right]
```

其中，`C_j` 是各审计维度覆盖率，`P_0` 是未关闭的 P0 风险数。缓存门禁强调：缓存必须同时证明收益、正确性、隔离、一致性和可审计。

下面这个 0 依赖 Python demo 演示一个简化缓存审计器：先查 result cache，再查 prefix cache，最后查 semantic cache；同时阻断跨租户语义命中、版本过期结果命中和非确定性生成的结果缓存。

```python
from math import sqrt

CHECKS = [
    "request_cache_profile",
    "cache_layer_boundary",
    "key_version_fingerprint",
    "prompt_prefix_reuse",
    "kv_lifecycle_management",
    "semantic_similarity_guard",
    "result_cache_determinism",
    "tenant_permission_isolation",
    "ttl_staleness_control",
    "eviction_quota_policy",
    "streaming_cache_policy",
    "cache_observability_metrics",
    "cost_latency_savings",
    "cache_governance_gate",
]

POLICY = {
    "now_min": 1000,
    "semantic_threshold": 0.92,
    "prefill_tokens_per_second": 12000,
    "input_cost_per_1k": 0.6,
    "output_cost_per_1k": 1.8,
    "cache_read_ms": {"result": 12, "prefix": 5, "semantic": 18},
}

RESULT_CACHE = [
    {
        "prompt_hash": "faq_refund_v1",
        "model": "chat-large",
        "model_version": "v3",
        "tokenizer": "tok-a",
        "params_hash": "temp0_top1_max80",
        "tenant": "public",
        "permission_domain": "faq_public",
        "safety_policy": "safe-2026-06",
        "data_version": "kb-42",
        "created_min": 980,
        "ttl_min": 60,
        "answer_tokens": 76,
    },
    {
        "prompt_hash": "policy_shipping_v1",
        "model": "chat-large",
        "model_version": "v3",
        "tokenizer": "tok-a",
        "params_hash": "temp0_top1_max120",
        "tenant": "public",
        "permission_domain": "faq_public",
        "safety_policy": "safe-2026-06",
        "data_version": "kb-41",
        "created_min": 950,
        "ttl_min": 30,
        "answer_tokens": 90,
    },
]

PREFIX_CACHE = [
    {
        "prefix_hash": "handbook_v7_prefix",
        "model": "chat-large",
        "model_version": "v3",
        "tokenizer": "tok-a",
        "tenant": "enterprise-a",
        "permission_domain": "handbook_reader",
        "safety_policy": "safe-2026-06",
        "data_version": "doc-v7",
        "cached_tokens": 1600,
        "created_min": 990,
        "ttl_min": 120,
    }
]

SEMANTIC_CACHE = [
    {
        "query_id": "refund_process",
        "embedding": [1.0, 0.0, 0.0],
        "tenant": "public",
        "permission_domain": "faq_public",
        "data_version": "kb-42",
        "created_min": 970,
        "ttl_min": 90,
        "saved_input_tokens": 140,
        "saved_output_tokens": 70,
    },
    {
        "query_id": "enterprise_salary_policy",
        "embedding": [0.0, 1.0, 0.0],
        "tenant": "enterprise-a",
        "permission_domain": "hr_private",
        "data_version": "hr-9",
        "created_min": 990,
        "ttl_min": 45,
        "saved_input_tokens": 900,
        "saved_output_tokens": 120,
    },
]

REQUESTS = [
    {
        "id": "exact_faq_retry",
        "prompt_hash": "faq_refund_v1",
        "prefix_hash": None,
        "embedding": [0.97, 0.03, 0.0],
        "model": "chat-large",
        "model_version": "v3",
        "tokenizer": "tok-a",
        "params_hash": "temp0_top1_max80",
        "temperature": 0.0,
        "tenant": "public",
        "permission_domain": "faq_public",
        "safety_policy": "safe-2026-06",
        "data_version": "kb-42",
        "input_tokens": 180,
        "expected_output_tokens": 80,
        "stream": False,
        "sensitive": False,
    },
    {
        "id": "shared_doc_query",
        "prompt_hash": "handbook_query_a",
        "prefix_hash": "handbook_v7_prefix",
        "embedding": [0.20, 0.10, 0.97],
        "model": "chat-large",
        "model_version": "v3",
        "tokenizer": "tok-a",
        "params_hash": "temp0_top1_max200",
        "temperature": 0.0,
        "tenant": "enterprise-a",
        "permission_domain": "handbook_reader",
        "safety_policy": "safe-2026-06",
        "data_version": "doc-v7",
        "input_tokens": 2100,
        "expected_output_tokens": 160,
        "stream": True,
        "sensitive": False,
    },
    {
        "id": "semantic_faq",
        "prompt_hash": "refund_wording_new",
        "prefix_hash": None,
        "embedding": [0.96, 0.05, 0.0],
        "model": "chat-large",
        "model_version": "v3",
        "tokenizer": "tok-a",
        "params_hash": "temp0_top1_max80",
        "temperature": 0.0,
        "tenant": "public",
        "permission_domain": "faq_public",
        "safety_policy": "safe-2026-06",
        "data_version": "kb-42",
        "input_tokens": 130,
        "expected_output_tokens": 70,
        "stream": False,
        "sensitive": False,
    },
    {
        "id": "private_cross_tenant",
        "prompt_hash": "salary_question",
        "prefix_hash": None,
        "embedding": [0.01, 0.99, 0.0],
        "model": "chat-large",
        "model_version": "v3",
        "tokenizer": "tok-a",
        "params_hash": "temp0_top1_max120",
        "temperature": 0.0,
        "tenant": "enterprise-b",
        "permission_domain": "hr_private",
        "safety_policy": "safe-2026-06",
        "data_version": "hr-9",
        "input_tokens": 760,
        "expected_output_tokens": 110,
        "stream": False,
        "sensitive": True,
    },
    {
        "id": "stale_policy",
        "prompt_hash": "policy_shipping_v1",
        "prefix_hash": None,
        "embedding": [0.1, 0.2, 0.9],
        "model": "chat-large",
        "model_version": "v3",
        "tokenizer": "tok-a",
        "params_hash": "temp0_top1_max120",
        "temperature": 0.0,
        "tenant": "public",
        "permission_domain": "faq_public",
        "safety_policy": "safe-2026-06",
        "data_version": "kb-42",
        "input_tokens": 220,
        "expected_output_tokens": 95,
        "stream": False,
        "sensitive": False,
    },
    {
        "id": "creative_generation",
        "prompt_hash": "creative_copy_variant",
        "prefix_hash": None,
        "embedding": [0.2, 0.2, 0.8],
        "model": "chat-large",
        "model_version": "v3",
        "tokenizer": "tok-a",
        "params_hash": "temp08_top09_max200",
        "temperature": 0.8,
        "tenant": "public",
        "permission_domain": "marketing_public",
        "safety_policy": "safe-2026-06",
        "data_version": "kb-42",
        "input_tokens": 240,
        "expected_output_tokens": 160,
        "stream": False,
        "sensitive": False,
    },
]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sqrt(sum(x * x for x in a))
    nb = sqrt(sum(y * y for y in b))
    return dot / max(na * nb, 1e-9)


def is_fresh(entry):
    return POLICY["now_min"] - entry["created_min"] <= entry["ttl_min"]


def result_key_match(req, entry):
    fields = [
        "prompt_hash",
        "model",
        "model_version",
        "tokenizer",
        "params_hash",
        "tenant",
        "permission_domain",
        "safety_policy",
        "data_version",
    ]
    return all(req[field] == entry[field] for field in fields)


def stale_result_candidate(req, entry):
    return (
        req["prompt_hash"] == entry["prompt_hash"]
        and req["model"] == entry["model"]
        and req["model_version"] == entry["model_version"]
        and req["tenant"] == entry["tenant"]
        and req["permission_domain"] == entry["permission_domain"]
        and req["data_version"] != entry["data_version"]
    )


def prefix_match(req, entry):
    fields = [
        "prefix_hash",
        "model",
        "model_version",
        "tokenizer",
        "tenant",
        "permission_domain",
        "safety_policy",
        "data_version",
    ]
    return req["prefix_hash"] and all(req[field] == entry[field] for field in fields)


def saved_cost(input_tokens, output_tokens):
    return (
        input_tokens * POLICY["input_cost_per_1k"]
        + output_tokens * POLICY["output_cost_per_1k"]
    ) / 1000.0


def evaluate_cache(req):
    trace = {
        "request_id": req["id"],
        "tenant": req["tenant"],
        "permission_domain": req["permission_domain"],
        "model_version": req["model_version"],
        "data_version": req["data_version"],
        "decision": None,
        "layer": None,
        "saved_input_tokens": 0,
        "saved_output_tokens": 0,
        "cache_latency_ms": 0,
        "blocked_reason": None,
    }

    if req["temperature"] == 0.0:
        for entry in RESULT_CACHE:
            if result_key_match(req, entry):
                if is_fresh(entry):
                    trace.update({
                        "decision": "result_cache_hit",
                        "layer": "result",
                        "saved_input_tokens": req["input_tokens"],
                        "saved_output_tokens": min(req["expected_output_tokens"], entry["answer_tokens"]),
                        "cache_latency_ms": POLICY["cache_read_ms"]["result"],
                    })
                    return trace
                trace.update({"decision": "stale_result_blocked", "layer": "result", "blocked_reason": "ttl_expired"})
                return trace
            if stale_result_candidate(req, entry):
                trace.update({"decision": "stale_result_blocked", "layer": "result", "blocked_reason": "data_version_mismatch"})
                return trace
    else:
        trace.update({"blocked_reason": "non_deterministic_sampling"})

    for entry in PREFIX_CACHE:
        if prefix_match(req, entry):
            if is_fresh(entry):
                trace.update({
                    "decision": "prefix_cache_hit",
                    "layer": "prefix",
                    "saved_input_tokens": min(req["input_tokens"], entry["cached_tokens"]),
                    "cache_latency_ms": POLICY["cache_read_ms"]["prefix"],
                })
                return trace
            trace.update({"decision": "stale_prefix_blocked", "layer": "prefix", "blocked_reason": "ttl_expired"})
            return trace

    best = None
    for entry in SEMANTIC_CACHE:
        sim = cosine(req["embedding"], entry["embedding"])
        if sim >= POLICY["semantic_threshold"]:
            if req["tenant"] != entry["tenant"] or req["permission_domain"] != entry["permission_domain"]:
                trace.update({
                    "decision": "semantic_isolation_blocked",
                    "layer": "semantic",
                    "blocked_reason": "tenant_or_permission_mismatch",
                })
                return trace
            if req["data_version"] != entry["data_version"] or not is_fresh(entry):
                trace.update({"decision": "semantic_stale_blocked", "layer": "semantic", "blocked_reason": "stale_semantic_entry"})
                return trace
            if best is None or sim > best[0]:
                best = (sim, entry)
    if best is not None and not req["sensitive"]:
        _, entry = best
        trace.update({
            "decision": "semantic_cache_hit",
            "layer": "semantic",
            "saved_input_tokens": min(req["input_tokens"], entry["saved_input_tokens"]),
            "saved_output_tokens": min(req["expected_output_tokens"], entry["saved_output_tokens"]),
            "cache_latency_ms": POLICY["cache_read_ms"]["semantic"],
        })
        return trace

    trace["decision"] = "cache_miss"
    trace["layer"] = "none"
    return trace


def build_profile_cases():
    complete = {"name": "complete_cache_case"}
    complete.update({check: True for check in CHECKS})
    cases = [complete]
    bad_names = [
        "request_cache_profile_missing_bad",
        "cache_layer_blurred_bad",
        "key_version_missing_bad",
        "prompt_prefix_disabled_bad",
        "kv_lifecycle_leak_bad",
        "semantic_threshold_missing_bad",
        "result_cache_random_bad",
        "tenant_isolation_missing_bad",
        "ttl_staleness_missing_bad",
        "eviction_quota_missing_bad",
        "streaming_policy_missing_bad",
        "cache_metrics_missing_bad",
        "savings_model_missing_bad",
        "cache_governance_gate_missing_bad",
    ]
    for bad_name, failed_check in zip(bad_names, CHECKS):
        case = {"name": bad_name}
        case.update({check: True for check in CHECKS})
        case[failed_check] = False
        cases.append(case)
    return cases


def audit_profiles(cases):
    metrics = {check: round(sum(1 for case in cases if case.get(check)) / len(cases), 3) for check in CHECKS}
    failed_cases = [case["name"] for case in cases if not all(case.get(check) for check in CHECKS)]
    failed_gates = [check for check, value in metrics.items() if value < 1.0]
    hard_blockers = sum(1 for case in cases if case["name"].endswith("_bad"))
    gate_pass = min(metrics.values()) >= 0.95 and hard_blockers == 0
    return metrics, failed_cases, failed_gates, hard_blockers, gate_pass


traces = [evaluate_cache(req) for req in REQUESTS]
decisions = {trace["request_id"]: trace["decision"] for trace in traces}
layer_hits = {
    "result": sum(1 for trace in traces if trace["layer"] == "result" and trace["decision"].endswith("hit")),
    "prefix": sum(1 for trace in traces if trace["layer"] == "prefix" and trace["decision"].endswith("hit")),
    "semantic": sum(1 for trace in traces if trace["layer"] == "semantic" and trace["decision"].endswith("hit")),
}
saved_input = sum(trace["saved_input_tokens"] for trace in traces)
saved_output = sum(trace["saved_output_tokens"] for trace in traces)
cache_cost_usd = 0.003
net_saved_cost = round(saved_cost(saved_input, saved_output) - cache_cost_usd, 4)
saved_prefill_ms = round(saved_input / POLICY["prefill_tokens_per_second"] * 1000, 1)
required_trace = {
    "request_id",
    "tenant",
    "permission_domain",
    "model_version",
    "data_version",
    "decision",
    "layer",
    "saved_input_tokens",
    "saved_output_tokens",
    "cache_latency_ms",
    "blocked_reason",
}
trace_coverage = sum(1 for trace in traces if required_trace.issubset(trace)) / len(traces)
blocked = [trace["request_id"] for trace in traces if "blocked" in trace["decision"]]
metrics, failed_cases, failed_gates, hard_blockers, gate_pass = audit_profiles(build_profile_cases())

print("cache_decisions=", decisions)
print("cache_layer_hits=", layer_hits)
print("blocked_cache_reuses=", blocked)
print("saved_tokens=", {"input": saved_input, "output": saved_output})
print("saved_prefill_ms=", saved_prefill_ms)
print("net_saved_cost=", net_saved_cost)
print("cache_trace_coverage=", round(trace_coverage, 3))
print("metrics=", metrics)
print("hard_blocker_count=", hard_blockers)
print("failed_cases=", failed_cases)
print("failed_gates=", failed_gates)
print("cache_governance_gate_pass=", gate_pass)
```

输出示例：

```text
cache_decisions= {'exact_faq_retry': 'result_cache_hit', 'shared_doc_query': 'prefix_cache_hit', 'semantic_faq': 'semantic_cache_hit', 'private_cross_tenant': 'semantic_isolation_blocked', 'stale_policy': 'stale_result_blocked', 'creative_generation': 'cache_miss'}
cache_layer_hits= {'result': 1, 'prefix': 1, 'semantic': 1}
blocked_cache_reuses= ['private_cross_tenant', 'stale_policy']
saved_tokens= {'input': 1910, 'output': 146}
saved_prefill_ms= 159.2
net_saved_cost= 1.4058
cache_trace_coverage= 1.0
metrics= {'request_cache_profile': 0.933, 'cache_layer_boundary': 0.933, 'key_version_fingerprint': 0.933, 'prompt_prefix_reuse': 0.933, 'kv_lifecycle_management': 0.933, 'semantic_similarity_guard': 0.933, 'result_cache_determinism': 0.933, 'tenant_permission_isolation': 0.933, 'ttl_staleness_control': 0.933, 'eviction_quota_policy': 0.933, 'streaming_cache_policy': 0.933, 'cache_observability_metrics': 0.933, 'cost_latency_savings': 0.933, 'cache_governance_gate': 0.933}
hard_blocker_count= 14
cache_governance_gate_pass= False
```

这个 demo 的重点不是复刻某个生产缓存系统，而是训练一套面试思维：缓存命中前必须检查模型、版本、tokenizer、参数、租户、权限、安全策略、数据版本和 TTL；语义相似不能绕过权限；随机生成不能随便结果缓存；缓存收益要同时看 token、TTFT、成本和误命中风险。

## 31.21 常见误区

误区一：缓存命中率越高越好。

高命中率不一定等于高收益，还要看节省的 token 和成本。

误区二：语义相似就可以直接返回。

语义相似不代表答案等价，尤其涉及用户、权限、时间和业务状态时。

误区三：KV cache 和结果缓存是一回事。

KV cache 是 runtime 内部 attention 缓存，结果缓存是业务层响应缓存，完全不是同一层东西。

误区四：缓存只是性能问题。

缓存同时是安全、权限、一致性和审计问题。

## 31.22 面试常见追问

问题一：prompt cache 和 KV cache 有什么区别？

可以回答：prompt cache 更关注复用 prompt 前缀处理结果，主要降低 prefill 成本；KV cache 是 attention 中 Key/Value 的缓存，主要用于加速 decode 和上下文复用，通常由 runtime 管理。

问题二：语义缓存有什么风险？

可以回答：语义缓存可能把相似但不等价的问题当成同一问题，导致答非所问；还可能因为权限、用户上下文、知识过期或安全策略不同而错误返回。

问题三：缓存 key 应该包含什么？

可以回答：至少包含模型和版本、prompt hash、参数、tokenizer、租户或权限域、安全策略版本和数据版本，避免跨模型、跨权限或跨版本误命中。

问题四：如何评估缓存收益？

可以回答：不能只看命中率，还要看节省的 input/output tokens、GPU 时间、TTFT、端到端延迟、成本，以及误命中和过期命中风险。

问题五：语义缓存上线前要设置哪些门禁？

可以回答：至少要设置相似度阈值、任务类型一致、租户和权限域一致、数据版本一致、TTL 未过期、敏感请求禁用或二次校验、误命中率评估、trace 完整和人工复核样本。

问题六：为什么结果缓存通常要求采样参数进入 key？

可以回答：temperature、top_p、max_tokens、stop sequence 等参数会改变输出分布和截断边界。如果 key 不包含这些参数，同一个 prompt 可能错误命中不同生成配置下的结果。

## 31.23 小练习

1. Prompt cache、KV cache、语义缓存和结果缓存分别缓存什么？
2. 为什么 prompt cache 通常要求模型版本和 tokenizer 一致？
3. 语义缓存为什么可能误命中？
4. 结果缓存适合哪些任务？
5. 缓存 key 为什么要包含租户和权限域？
6. 缓存如何处理模型版本升级？
7. 为什么只看 cache hit rate 不够？
8. 如何为企业知识库问答设计缓存策略？
9. 写出一个缓存 key schema，要求覆盖模型、参数、租户、权限、安全策略和数据版本。
10. 设计一个语义缓存误命中评估集，至少包含相似但不等价、跨权限、过期知识和个性化上下文四类样本。
11. 用公式估算 prefix cache 节省的 prefill token、TTFT 和成本。
12. 为一个多租户推理平台设计缓存 trace 字段。

## 31.24 本章小结

本章讲了大模型推理平台的缓存体系。

你需要记住：

1. 推理缓存是一组多层机制，不是单个 Redis 缓存。
2. Prompt cache 主要降低 prefill 成本，KV cache 主要加速 decode 和上下文复用。
3. 语义缓存能降低相似请求成本，但有误命中风险。
4. 结果缓存返回最快，但对版本、参数、权限、一致性和随机性要求最严格。
5. 缓存 key 必须包含模型、版本、参数、权限、安全和数据版本等信息。
6. 缓存必须同时考虑性能、成本、安全、隔离、一致性和审计。
7. 缓存门禁要同时检查请求画像、缓存层边界、key 版本指纹、prefix / KV 生命周期、语义相似阈值、结果缓存确定性、租户隔离、TTL、驱逐、streaming、指标和成本收益。

下一章我们会讲自动扩缩容：QPS、延迟、队列长度和 GPU 利用率。
