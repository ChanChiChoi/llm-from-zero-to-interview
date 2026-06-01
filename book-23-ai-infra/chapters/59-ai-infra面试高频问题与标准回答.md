# 第 59 章 AI Infra 面试高频问题与标准回答

上一章设计了 AI Infra 可观测性平台。本章整理 AI Infra 面试中的高频问题与标准回答。

这章不是重新讲知识点，而是帮助你把前面 58 章压缩成面试可用的回答模板。面试时最重要的不是背术语，而是能把问题拆开，讲清楚目标、架构、关键机制、trade-off 和生产治理。

先记住一句话：

> AI Infra 面试的核心，是证明你既懂大模型任务特性，又懂分布式系统、GPU 集群、平台工程和生产治理。

## 59.1 AI Infra 是什么

问题：什么是 AI Infra？它和 MLOps、LLMOps 有什么区别？

标准回答：

```text
AI Infra 是支撑大模型训练、推理、评估和应用运行的基础设施与平台工程体系，覆盖 GPU 集群、网络、存储、调度、训练平台、推理平台、数据平台、模型仓库、评估平台、可观测性、安全和成本治理。

MLOps 更关注传统机器学习生命周期管理，例如数据、训练、部署、监控。LLMOps 更关注大模型应用生命周期，例如 prompt、RAG、Agent、评估、发布、trace 和成本。AI Infra 更底层，提供算力、平台和生产治理能力。
```

追问重点：AI Infra 不是单个工具，而是一套平台体系。

## 59.2 训练平台怎么设计

问题：请设计一个大模型训练平台。

标准回答：

```text
我会把训练平台设计成 TrainingJob 生命周期管理系统。用户通过 Web、CLI、SDK 或 API 提交 TrainingJob，配置镜像、代码版本、数据版本、资源、launcher、checkpoint 和重试策略。

平台先做身份、权限、配额、数据访问和配置校验，然后进入队列。Scheduler 根据 GPU 类型、优先级、配额、gang scheduling、拓扑和资源碎片做调度。Launcher Controller 负责 torchrun、DeepSpeed、Megatron 或 Ray 的分布式启动。

运行中采集日志、metrics、events、loss、tokens/s、GPU 利用率、data loader time 和 checkpoint 信息，并自动写入 experiment tracking。Checkpoint 支持保留策略、校验和断点恢复。任务完成后将候选 checkpoint 注册到模型仓库并触发评估。
```

关键词：TrainingJob、gang scheduling、launcher、checkpoint、experiment tracking、model registry。

## 59.3 为什么 TrainingJob 不是普通 K8s Job

问题：TrainingJob 和 Kubernetes Job 有什么区别？

标准回答：

```text
K8s Job 只是通用批任务抽象，而 TrainingJob 需要表达训练语义，包括数据版本、代码版本、镜像、分布式 launcher、GPU 拓扑、checkpoint 策略、重试策略、实验追踪、模型产物、权限和成本归因。

大模型训练还需要 gang scheduling、rank/world size 注入、NCCL 环境、checkpoint 恢复和多租户队列，这些不是普通 K8s Job 自带的能力。
```

扣分点：只说“TrainingJob 是一个 YAML”。

## 59.4 为什么需要 Gang Scheduling

问题：大模型训练为什么需要 gang scheduling？

标准回答：

```text
多机多卡训练要求所有 worker 同时启动并组成同一个分布式作业。如果只启动一部分 worker，任务无法正常运行，还会占住 GPU 资源。

Gang scheduling 要求资源足够时整体调度，资源不足时整体排队，启动失败时整体回滚。它能避免部分 worker 占资源但任务无法执行的问题。
```

延伸：gang scheduling 和资源碎片、拓扑感知经常一起考。

## 59.5 GPU 集群调度怎么设计

问题：请设计一个多租户 GPU 集群调度系统。

标准回答：

```text
我会设计队列、配额和拓扑感知的调度控制平面。用户提交任务后，平台做身份、权限、队列和配额校验，然后进入 Queue Manager。

Scheduler 维护 Resource Snapshot，包括 GPU 类型、数量、健康状态、拓扑、CPU、内存、网络和本地盘。Policy Engine 根据租户配额、优先级、公平性和任务类型选择候选任务，Placement Engine 根据 gang scheduling、拓扑、资源碎片和数据本地性选择节点。

如果高优先级任务资源不足，可以通过 Preemption Engine 抢占低优先级、可恢复任务。低优先级队列用于填充空闲 GPU。系统还要提供调度事件、队列等待、配额使用、资源碎片、GPU 利用率和成本归因。
```

关键词：queue、quota、topology-aware、gang scheduling、preemption、fairness。

## 59.6 GPU 利用率低怎么排查

问题：训练任务 GPU 利用率低，你怎么排查？

标准回答：

```text
我会先拆分 step time，看时间花在 data loading、forward、backward、optimizer、communication 还是 checkpoint。GPU 利用率低不一定是 GPU 问题，可能是数据读取慢、CPU preprocessing 慢、batch 太小、NCCL 通信等待、checkpoint 太频繁、eval 太频繁或负载不均衡。

具体看 dataloader time、GPU idle time、storage throughput、CPU 利用率、NCCL latency、tokens/s、batch size 和 per-rank logs。定位阶段后再做对应优化。
```

重点：不要直接说“加 batch”。

## 59.7 Loss NaN 怎么排查

问题：训练 loss 出现 NaN 怎么排查？

标准回答：

```text
先定位 NaN 出现的 step，保存对应 batch，检查输入、label、mask、tokenizer 和异常样本。再看 learning rate、gradient norm、precision、loss scale、logits 和 gradient 是否出现 inf。

常见原因包括学习率过大、梯度爆炸、FP16 溢出、loss scale 不合适、数据异常、自定义 loss 除零、checkpoint resume 状态损坏。可以尝试降低学习率、开启 gradient clipping、切 BF16、检查 loss scale，并用单卡小 batch 做最小复现。
```

重点：NaN 不要只靠跳过 batch 掩盖。

## 59.8 推理平台怎么设计

问题：请设计一个高并发大模型推理平台。

标准回答：

```text
我会把平台分为治理层和 runtime 执行层。请求进入 API Gateway 后做认证、限流和 request ID 注入，再经过安全检查和 Model Router。Router 根据模型能力、租户权限、成本、SLO、灰度规则和 endpoint 负载选择模型版本和 endpoint。

Cache Layer 尝试命中 prompt cache、semantic cache 或 result cache。Admission Controller 根据队列、KV cache 和配额判断是否接收。Runtime 层使用 vLLM/TGI/Triton/SGLang 等，支持 prefill、decode、continuous batching、KV cache 和 streaming。

扩缩容结合 input/output tokens/s、queue wait、TTFT、TPOT、KV cache、GPU 和 timeout。稳定性上做限流、熔断、timeout budget、fallback 和降级。发布上支持模型、prompt、runtime 和路由策略灰度与回滚。
```

关键词：Model Router、KV cache、continuous batching、TTFT、TPOT、autoscaling。

## 59.9 为什么不能只看 QPS

问题：LLM 推理为什么不能只用 QPS 衡量负载？

标准回答：

```text
因为大模型请求的输入和输出 token 数差异巨大。同样 10 QPS，一个请求可能是 100 input tokens + 50 output tokens，另一个可能是 8000 input tokens + 2000 output tokens，资源消耗完全不同。

LLM 推理更应该看 input tokens/s、output tokens/s、active sequences、KV cache usage、TTFT、TPOT、queue length 和 cost per 1k tokens。
```

重点：token 是推理负载的基本单位。

## 59.10 TTFT 和 TPOT

问题：TTFT 和 TPOT 分别是什么？怎么优化？

标准回答：

```text
TTFT 是 Time To First Token，表示从请求进入到第一个 token 返回的时间，主要受 gateway、router、queue、tokenizer、prefill 和输入长度影响。

TPOT 是 Time Per Output Token，表示每生成一个输出 token 的时间，主要受 decode、KV cache、batching、显存带宽、active sequences 和 runtime 影响。

优化 TTFT 可以做 prompt cache、减少排队、优化 prefill、长短请求分队列。优化 TPOT 可以做 continuous batching、KV cache 优化、量化、speculative decoding 和 runtime 优化。
```

## 59.11 KV Cache 为什么重要

问题：KV cache 是什么？为什么会成为瓶颈？

标准回答：

```text
KV cache 保存 Transformer attention 中历史 token 的 Key 和 Value，避免 decode 时每一步重新计算完整上下文。

它能加速生成，但会占用大量显存，显存占用和 batch size、sequence length、层数、hidden size、数据类型有关。长上下文、高并发和大模型会快速放大 KV cache 压力，导致 OOM、admission reject、p99 抖动和吞吐下降。
```

## 59.12 Continuous Batching

问题：Continuous batching 和普通 batching 有什么区别？

标准回答：

```text
普通 batching 通常一批请求固定执行到结束，而生成式大模型每个请求输出长度不同，短请求会被长请求拖住。

Continuous batching 在每个 decode step 动态调整 batch，完成的请求退出，新请求加入，更适合生成式模型。它能提高 GPU 利用率和 output tokens/s，但会增加 scheduler 和 KV cache 管理复杂度。
```

## 59.13 模型路由和负载均衡

问题：模型路由和普通负载均衡有什么区别？

标准回答：

```text
普通负载均衡是在等价实例之间分流。模型路由要先在不同能力、成本、延迟、安全等级、上下文长度和版本的模型之间选择，再选择具体 endpoint。

路由依据包括任务类型、租户权限、成本预算、SLO、灰度规则、endpoint 健康、queue length、KV cache 水位和 fallback chain。
```

## 59.14 自动扩缩容怎么设计

问题：LLM 推理自动扩缩容怎么设计？

标准回答：

```text
不能只看 CPU、QPS 或 GPU 利用率。应结合 QPS、input/output tokens/s、queue length、queue wait time、TTFT、TPOT、p99、KV cache usage、GPU utilization、timeout rate。

还要考虑冷启动，因为大模型实例需要拉镜像、加载权重、初始化 runtime、分配显存和预热 kernel。可以使用 warm pool、权重本地缓存、预测扩容和保守缩容。缩容前要 draining，避免中断 streaming 请求。
```

## 59.15 RAG 平台怎么设计

问题：企业 RAG 平台怎么设计？

标准回答：

```text
RAG 平台包括 Knowledge Base、Document Store、Chunk Store、Embedding Store、Vector Index、Retrieval Service、Reranker、Prompt Assembly、Citation 和 Retrieval Trace。

文档进入平台后经过解析、清洗、chunking、embedding 和索引构建。检索时必须根据用户身份做权限过滤，再 rerank，最后组装 prompt。平台要记录 query、embedding model、index version、filters、top-k、rerank scores、进入 prompt 的 chunks 和 citation。
```

重点：企业 RAG 的核心是权限和 trace，不只是向量数据库。

## 59.16 Agent 平台怎么设计

问题：Agent 平台怎么设计？

标准回答：

```text
Agent 平台需要管理 Agent Definition、Prompt、Tool Registry、Tool Permission、Memory、Run State 和 Execution Trace。

Agent Runtime 执行多步任务，控制最大 step、超时、工具调用、用户确认和错误处理。Tool Definition 要版本化，包括 schema、endpoint、timeout、retry policy、权限和副作用等级。高风险工具需要权限校验、用户确认、审计和 human-in-the-loop。

平台必须记录 tool call trace 和 execution trace，用于调试、回放、评估、审计和成本归因。
```

## 59.17 模型仓库怎么设计

问题：模型仓库与发布系统怎么设计？

标准回答：

```text
模型仓库管理 Model、ModelVersion、权重、tokenizer、config、adapter、量化版本、eval report、权限、血缘和发布状态。训练平台产出 checkpoint 后注册成 ModelVersion candidate，做 artifact checksum、格式、安全和 runtime 兼容性校验。

评估平台写回 EvalReport，Quality Gate 根据质量、安全、延迟、成本和关键 slice 决定是否允许发布。发布系统通过 registered、validated、evaluated、approved、staged、canary、production 等状态机管理发布，并生成 ReleaseManifest，支持灰度、A/B 和回滚。
```

## 59.18 评估平台怎么设计

问题：大模型评估平台怎么设计？

标准回答：

```text
评估平台包括 Eval Dataset Registry、Eval Suite Registry、Eval Job Scheduler、Inference Runner、Metric Engine、LLM Judge、Human Review、Sample Result Store、Report Generator 和 Quality Gate。

Eval dataset、prompt、metric definition、judge model、inference config 都要版本化。平台要支持离线批量评测、LLM-as-Judge、人工评测、slice evaluation、sample-level failure analysis、在线 A/B 和发布门禁。
```

重点：评估平台不是 benchmark 脚本。

## 59.19 可观测性平台怎么设计

问题：AI Infra 可观测性平台怎么设计？

标准回答：

```text
我会设计统一收集 metrics、logs、traces、events、cost 和 model quality signals 的平台。数据来源包括训练平台、推理平台、GPU 集群、K8s、runtime、数据平台、评估平台、RAG/Agent 和发布系统。

通过 request ID、job ID、model version、tenant、deployment 等统一 ID 做关联，支持从告警跳到 trace、日志、事件、发布记录和成本记录。训练侧看 loss、tokens/s、GPU、data loader、NCCL、checkpoint；推理侧看 TTFT、TPOT、p99、queue、KV cache、tokens/s、cache hit 和成本；RAG/Agent 侧看 retrieval trace、tool call trace 和 execution trace。
```

## 59.20 成本治理怎么做

问题：AI Infra 成本治理怎么做？

标准回答：

```text
成本治理要先做用量采集和归因。按 tenant、team、project、model、endpoint、training job、experiment run 统计 GPU hours、storage、network、inference tokens、eval cost 和 trace/log cost。

训练看 GPU hours、failed job cost、tokens/GPU hour、idle GPU hours。推理看 cost per request、cost per 1k tokens、cache saved cost、warm pool cost。优化手段包括小模型优先、模型路由、缓存、量化、batching、自动扩缩容、checkpoint retention、冷热存储和预算配额。
```

## 59.21 安全治理怎么做

问题：AI Infra 安全治理包括什么？

标准回答：

```text
包括身份认证、权限控制、密钥管理、数据合规、多租户隔离、模型访问控制、供应链安全、运行时安全、日志 trace 安全、审计和安全事件响应。

训练任务使用 service account 和短期凭证，不使用长期高权限密钥。RAG 检索必须在检索阶段做权限过滤。Agent 高风险工具调用要权限校验、用户确认和审计。模型 artifact 要做 checksum、签名、格式和安全扫描。日志和 trace 要脱敏、限权、TTL 和审计。
```

## 59.22 多租户隔离怎么做

问题：AI Infra 多租户隔离包括哪些方面？

标准回答：

```text
多租户隔离不仅是 namespace。它包括身份隔离、资源配额、队列隔离、数据权限、模型访问权限、网络隔离、secret 隔离、日志和 trace 隔离、缓存隔离、成本归因和审计。

例如 RAG 中 chunk 和 embedding 要带权限标签，检索时强制过滤；推理缓存 key 要包含租户和权限域；训练任务读取 dataset 前要做权限校验。
```

## 59.23 训练故障定位

问题：训练任务 hang 住怎么排查？

标准回答：

```text
先看所有 rank 最后一条日志、step 是否停止增长、GPU 利用率是否部分为 0、dataloader worker 状态、NCCL 日志、节点事件和存储网络指标。

Hang 常见原因是某个 rank 卡在 dataloader、某个 rank 先 OOM 或崩溃、NCCL collective 不一致、数据读取阻塞、checkpoint 保存阻塞或 barrier 使用不当。重点找最早异常 rank，而不是最后 timeout 的 rank。
```

## 59.24 推理故障定位

问题：推理 p99 突然升高怎么排查？

标准回答：

```text
先确定影响范围：模型、租户、endpoint、地域、请求类型。再按模型、租户、endpoint、输入长度、输出长度切分 p99。查看慢请求 trace，拆 gateway、router、cache、queue、prefill、decode、streaming。

常见原因包括长请求混入、某个租户突增、长短请求混跑、某个实例拥塞、KV cache 水位高、cache 命中率下降、tool 调用慢、发布变更或 autoscaling 来不及。
```

## 59.25 SLO 和错误预算

问题：AI Infra 如何设计 SLO 和错误预算？

标准回答：

```text
先定义 SLI，例如推理成功率、TTFT、TPOT、p99、timeout rate，训练任务成功率、调度等待时间、checkpoint 恢复成功率。再定义 SLO，例如 99.9% 请求成功、95% TTFT 小于 1 秒。

错误预算是 SLO 允许的失败空间，用于平衡稳定性和迭代速度。预算消耗过快时应降低发布风险、暂停高风险变更或优先修复可靠性问题。
```

## 59.26 回答系统设计题的结构

面试中回答 AI Infra 系统设计题，可以按这个结构：

1. 需求澄清。
2. 核心目标。
3. 核心对象。
4. 总体架构。
5. 请求或任务链路。
6. 调度和资源。
7. 数据和 artifact。
8. 容错和稳定性。
9. 可观测性。
10. 安全和权限。
11. 成本治理。
12. Trade-off。

这个结构适用于训练平台、推理平台、LLMOps、评估平台、调度系统等题目。

## 59.27 高频扣分点总结

常见扣分点：

1. 只说工具名，不讲系统链路。
2. 只看 QPS，不看 token。
3. 只看 GPU 利用率，不看有效产出。
4. 不提 checkpoint 和恢复。
5. 不提 gang scheduling。
6. 不提 KV cache。
7. 不提 TTFT/TPOT。
8. 不提多租户权限。
9. 不提 trace 和可观测性。
10. 不提成本治理。
11. 不提灰度和回滚。
12. 不讲 trade-off。

面试官通常不是要你背全，而是看你能不能抓住核心矛盾。

## 59.28 小练习

1. 用 2 分钟回答“设计一个训练平台”。
2. 用 2 分钟回答“设计一个推理平台”。
3. 用 1 分钟解释 TTFT、TPOT 和 KV cache。
4. 用 1 分钟解释 gang scheduling 和资源碎片。
5. 用 2 分钟回答“推理 p99 升高怎么排查”。
6. 用 2 分钟回答“AI Infra 成本治理怎么做”。
7. 用 2 分钟回答“企业 RAG 如何防止越权”。
8. 用 3 分钟回答“设计一个 LLMOps 平台”。

## 59.29 本章小结

本章整理了 AI Infra 面试高频问题与标准回答。

你需要记住：

1. AI Infra 面试重在系统性，不是单点工具。
2. 训练平台抓 TrainingJob、gang scheduling、checkpoint、实验追踪。
3. 推理平台抓 TTFT、TPOT、KV cache、continuous batching、路由、缓存和扩缩容。
4. 企业平台抓多租户、安全、评估、发布、trace 和成本治理。
5. 回答开放题时要讲目标、架构、链路、故障、可观测性、安全、成本和 trade-off。

下一章是第二十三册最后一章，我们会讲 AI Infra 未来趋势：Serverless GPU、异构算力、边缘推理和自治运维。
