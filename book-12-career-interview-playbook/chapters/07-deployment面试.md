# 第七章：Deployment 面试

Deployment 面试考的不是你会不会说“KV Cache、vLLM、量化、batching、RAG、SLA、成本”这些词，而是你能不能把一个模型从 checkpoint 变成稳定、低延迟、可观测、可扩展、可控成本的线上服务。

很多候选人在部署面试里容易失分：知道 KV Cache 能加速，但不会估算显存；知道量化能省显存，但不知道 INT4 不一定更快；知道 vLLM 和 PagedAttention，但讲不清它解决的是 KV Cache 管理和动态调度问题；知道 RAG，但把它说成“向量数据库加大模型”；知道延迟指标，但分不清 TTFT、TPOT、吞吐和尾延迟。

大模型部署是系统工程。它的核心不是“把模型加载起来能回答”，而是在真实流量下同时满足质量、延迟、吞吐、成本、安全和可靠性约束。

本章重点：推理服务架构、prefill/decode、KV Cache、batching、vLLM/PagedAttention、量化、RAG/Agent 部署、监控排障、成本估算。

## 7.1 Deployment 面试到底考什么

Deployment 面试通常围绕八条主线：

1. 推理流程：模型从 checkpoint 到线上服务的完整链路。
2. 延迟与吞吐：TTFT、TPOT、QPS、tokens/s、P95/P99 如何理解和优化。
3. KV Cache：为什么需要、显存如何估算、长上下文和高并发下为什么成为瓶颈。
4. 调度与 batching：dynamic batching、continuous batching、chunked prefill、token budget。
5. 推理引擎：vLLM、SGLang、PagedAttention、prefix cache、speculative decoding 的作用边界。
6. 量化与压缩：PTQ、QAT、GPTQ、AWQ、weight-only quantization、KV Cache quantization。
7. 应用系统：RAG、Agent、function calling、多模型路由、安全审核和权限控制。
8. 生产治理：监控、告警、灰度、回滚、容量规划、成本和故障排查。

这些主题可以用一句话串起来：部署团队把模型权重、tokenizer、prompt template 和推理引擎组合成服务，通过路由、调度、缓存、量化和监控，在有限 GPU 资源下稳定服务不同长度、不同优先级、不同风险等级的请求。

面试官想看到的是你能不能回答这些问题：

1. 用户说“首字很慢”，你怎么排查？
2. tokens/s 看起来很高，但用户仍觉得卡，为什么？
3. 高并发下为什么 KV Cache 会 OOM？
4. vLLM 的 PagedAttention 解决了什么，不解决什么？
5. INT4 模型为什么可能质量变差或速度不升？
6. RAG 系统回答错了，如何判断是检索错、重排错、上下文错还是生成错？
7. 如果成本太高，你会从哪些层面优化？

## 7.2 回答 Deployment 题的通用结构

Deployment 面试建议使用“五步结构”：

1. 先定义目标：是离线评测、内部 demo、生产 API、端侧部署，还是企业私有化部署。
2. 再说请求路径：gateway、router、tokenizer、scheduler、GPU worker、streamer、日志监控。
3. 解释核心瓶颈：prefill、decode、KV Cache、显存、带宽、batching、网络和 CPU 前处理。
4. 讲优化手段：量化、continuous batching、prefix cache、speculative decoding、模型路由、prompt 压缩。
5. 给排障和评估：质量、延迟、吞吐、错误率、成本、安全和可观测性。

例如问“如何部署一个大模型服务”，不要只说：

```text
用 vLLM 启动模型，套一个 API，然后压测。
```

更好的回答是：

```text
我会先明确服务目标，包括模型大小、上下文长度、请求量、SLA、成本和安全要求。系统上会包含 API gateway、鉴权限流、请求路由、tokenizer、推理调度器、GPU worker、KV Cache manager、流式输出和监控。推理阶段要区分 prefill 和 decode：prefill 主要影响 TTFT，decode 主要影响 TPOT 和输出吞吐。高并发下要重点管理 KV Cache 和动态 batching，可以使用 vLLM 或 SGLang 这类引擎。上线前要评估质量、延迟、P95/P99、显存、错误率、安全拒答和成本，并做灰度、回滚和容量规划。
```

这个回答体现了完整工程闭环。

## 7.3 从 checkpoint 到线上服务

模型部署不是只加载权重。一个可用的大模型服务通常包括：

1. 模型权重：base、instruct、量化版本或 LoRA merge 后版本。
2. Tokenizer：必须和训练时一致，否则 token 序列分布会错。
3. Chat template：system、user、assistant、特殊 token 和轮次格式。
4. 推理引擎：Transformers、vLLM、SGLang、TensorRT-LLM 或自研 serving engine。
5. API 层：REST、OpenAI-compatible API、SSE 流式输出或 gRPC。
6. 调度层：batching、优先级、队列、超时、取消、资源预算。
7. 安全层：鉴权、限流、内容安全、工具权限、审计日志。
8. 观测层：指标、日志、trace、bad case、成本归因。

面试中如果问“为什么 tokenizer 和 chat template 很重要”，可以回答：

```text
模型看到的是 token 序列，不是人类看到的 messages 对象。Tokenizer 决定文本如何切成 token，chat template 决定 system、user、assistant 和特殊 token 如何排列。如果训练和推理模板不一致，模型会遇到分布偏移，轻则格式不稳定，重则拒答、角色混乱或工具调用失败。
```

上线前至少要验证：

1. 同一 prompt 在离线评测和线上服务中 tokenization 是否一致。
2. system prompt 和 chat template 是否重复或缺失。
3. stop token 和 EOS 是否正确。
4. max context、max output tokens 和截断策略是否符合预期。
5. 流式输出中断、客户端断开和超时是否正确释放资源。

## 7.4 Prefill 与 Decode

LLM 推理可以粗略分成两个阶段：prefill 和 decode。

Prefill 阶段处理用户输入的完整 prompt，一次前向计算所有输入 token 的 hidden states，并为每一层生成 KV Cache。Decode 阶段逐 token 生成输出，每一步只输入新生成的 token，同时复用历史 KV Cache。

两者瓶颈不同：

| 阶段 | 输入特点 | 主要指标 | 常见瓶颈 |
| --- | --- | --- | --- |
| Prefill | 一次处理较长 prompt | TTFT | 计算量、长 prompt、attention kernel、tokenizer、排队 |
| Decode | 每步生成一个或少量 token | TPOT、tokens/s | 权重读取、KV Cache 访问、batching、显存带宽 |

面试时要能解释：

```text
TTFT 主要受排队、tokenization、prefill 计算、prefix cache 命中和调度策略影响；TPOT 主要受 decode 阶段的模型 forward、KV Cache 访问、batch 大小、显存带宽和引擎调度影响。所以优化首 token 和优化持续输出速度，手段不完全一样。
```

常见追问：“为什么 output token 通常比 input token 贵？”

可以回答：

```text
输入 token 在 prefill 阶段可以并行处理，而输出 token 需要自回归逐步生成。每生成一个 token，都要占用一次模型 forward，并访问权重和历史 KV Cache。输出越长，占用 GPU 的时间越长，所以输出 token 的边际成本通常更高。
```

## 7.5 KV Cache：推理服务的核心状态

KV Cache 的作用是避免 decode 阶段重复计算历史 token 的 Key 和 Value。

没有 KV Cache 时，每生成一个新 token，都要重新处理整个上下文，代价随序列长度不断增加。有了 KV Cache 后，每一步只计算新 token 的 Q/K/V，并让新 token 的 Q 去 attend 历史 K/V。

面试中要能说清：

1. KV Cache 加速的是 decode 阶段。
2. KV Cache 是每个请求独有的运行时状态，不是模型参数。
3. KV Cache 大小随层数、batch、上下文长度、KV head 数、head dim 和精度增长。
4. 长上下文和高并发下，KV Cache 可能比权重更先成为显存瓶颈。

粗略估算公式：

```text
KV Cache bytes ≈ 2 * num_layers * batch_size * seq_len * num_kv_heads * head_dim * bytes_per_element
```

其中 `2` 表示 K 和 V 两份缓存。

如果面试官问“为什么 GQA/MQA 能降低 KV Cache 显存”，可以回答：

```text
MHA 中每个 query head 都有独立 K/V head，KV Cache 规模和 head 数相关。MQA 让所有 query heads 共享一组 K/V，GQA 让一组 query heads 共享一组 K/V，因此 num_kv_heads 更少，KV Cache 显存和读写带宽都会下降。代价是表达能力和质量可能受影响，需要训练阶段配合。
```

## 7.6 PagedAttention 与 vLLM

高并发 serving 中，请求长度不同、输出长度不同、到达和结束时间也不同。如果每个请求都连续预留一大块 KV Cache，会产生大量内部碎片和外部碎片。

PagedAttention 的直觉是借鉴操作系统分页：逻辑上连续的 KV Cache，不要求物理显存连续，而是切成固定大小 blocks，通过 block table 做映射。

面试回答要准确：

```text
PagedAttention 不是新的 attention 建模算法，也不改变 attention 的数学定义。它主要是 KV Cache 管理方法，把请求的逻辑 token block 映射到物理 KV blocks，降低显存碎片，并支持请求动态增长、释放和 continuous batching。
```

vLLM 的价值不只是 PagedAttention。面试中可以概括为：

1. 更高效的 KV Cache block 管理。
2. 支持 continuous batching，提高吞吐。
3. 支持 OpenAI-compatible API 和常见模型接入。
4. 支持 prefix cache、量化、分布式推理等工程能力。

但也要说边界：

1. 它不能自动修复模型质量问题。
2. 它不能让显存无限大，长上下文和高并发仍受 KV Cache budget 限制。
3. 它不是所有场景都比专用 TensorRT 或自研 kernel 更优。
4. 真实收益取决于模型、请求长度分布、硬件、batching 和业务 SLA。

## 7.7 Dynamic Batching 与 Continuous Batching

普通 dynamic batching 通常是在请求进入模型前等待一小段时间，把多个请求凑成 batch 一起执行。

LLM 自回归生成更复杂，因为每个请求输出长度不同。静态 batch 会让短请求等长请求，GPU 也无法及时接纳新请求。

Continuous batching，也叫 in-flight batching，是在每个 decode iteration 动态更新 batch：完成的请求退出，新请求进入，活跃请求继续生成。

面试回答：

```text
Continuous batching 适合 LLM，因为生成过程是逐 token 迭代的，不同请求输出长度差异很大。它允许请求在 decode 过程中动态加入和退出，提高 GPU 利用率，降低短请求被长请求阻塞的浪费。但它要求调度器能管理 KV Cache、token budget、优先级、超时和取消。
```

调度器不能只控制请求数，还要控制：

1. 每轮 token budget。
2. 活跃 batch size。
3. KV Cache block 占用。
4. Prefill 和 decode 的比例。
5. 请求优先级和超时。
6. 长请求与短请求隔离策略。

## 7.8 Chunked Prefill：长 prompt 怎么调度

长 prompt 的 prefill 可能一次占用 GPU 较长时间。如果服务正在给很多用户流式输出，突然进入一个超长 prompt，会让已有请求的 decode 变慢，用户看到 token 间隔变长。

Chunked prefill 的思想是把长 prompt 切成多个 chunk，穿插在 decode iteration 中执行，避免长 prefill 独占 GPU。

回答时要讲 trade-off：

```text
Chunked prefill 可以降低长 prompt 对已有 decode 请求的阻塞，改善流式输出稳定性。但它也可能增加调度复杂度，并让单个长 prompt 的 TTFT 变长。是否启用、chunk 多大，要根据业务更重视新请求首 token 还是已有请求 TPOT 来调。
```

这类题的重点不是背术语，而是能解释调度目标：TTFT、TPOT、吞吐、公平性和成本之间存在冲突。

## 7.9 量化：省显存不等于一定更快

量化把 FP16/BF16 权重或激活映射到 INT8、INT4 等低精度表示，目标是降低显存、带宽和成本。

常见类型包括：

1. Weight-only quantization：只量化权重，激活保留较高精度。
2. PTQ：训练后量化，成本低，依赖校准数据。
3. QAT：量化感知训练，质量更稳但成本更高。
4. GPTQ：利用近似二阶信息做权重量化误差补偿。
5. AWQ：根据激活统计保护重要权重或通道。
6. KV Cache quantization：量化运行时 KV Cache，降低长上下文和高并发显存压力。

面试中不要说“INT4 一定更快”。更准确的回答是：

```text
INT4 通常能显著降低权重显存，但速度是否提升取决于硬件和 kernel 是否高效、反量化开销、batch 大小、瓶颈是否在权重带宽，以及 KV Cache 是否成为主要瓶颈。如果系统瓶颈在 KV Cache、调度或网络，权重量化未必明显提升 TPOT。
```

量化上线前要评估：

1. 通用 benchmark 和业务样本质量。
2. 格式稳定性，如 JSON、代码、工具调用参数。
3. 安全拒答和越狱鲁棒性。
4. 长上下文、RAG、数学、代码、多轮对话。
5. TTFT、TPOT、吞吐、P95/P99。
6. 显存、OOM 率和并发上限。
7. 与 FP16/BF16 baseline 的差异。

如果问“perplexity 变化很小，为什么线上仍可能出问题”，可以回答：

```text
Perplexity 是平均 token loss，可能掩盖格式、工具调用、安全拒答和长上下文检索这些局部能力变化。量化误差可能不明显影响平均 loss，但会让 JSON 少括号、函数参数错、代码边界错或安全回答不稳定，所以需要任务级和业务级评估。
```

## 7.10 Speculative Decoding 与多 token 预测

Speculative decoding 的核心思想是：用小而快的 draft model 先提出多个候选 token，再用 target model 一次 forward 验证这些候选。如果候选被接受，一次大模型调用可以推进多个 token。

面试回答：

```text
Speculative decoding 的目标是在保持 target model 输出分布基本不变的前提下减少大模型 decode 次数。收益取决于 draft model 速度、接受率、候选长度、target 验证效率和任务分布。如果 draft 太弱、太慢或接受率低，收益可能很小甚至变差。
```

Medusa、EAGLE 等多 token 预测方法也可以放在 proposal-verification 框架下理解：

1. Medusa 用额外 heads 从当前 hidden state 预测未来 token 候选。
2. EAGLE 更强调在 feature 或 hidden state 层面预测未来状态。
3. 最终仍需要 target model 或原模型验证，不能简单把多个 token 直接拼上。

上线时不能只看接受率，还要看：

1. 提案模型或额外 heads 的计算成本。
2. 候选树验证成本。
3. KV Cache 和显存占用。
4. 对 continuous batching 的影响。
5. 不同任务、温度和输出长度下的收益。
6. P95/P99 是否变差。

## 7.11 RAG 部署不是向量数据库加 LLM

企业 RAG 系统通常分为离线和在线两条链路。

离线链路包括：

1. 文档接入。
2. 解析、OCR、表格抽取。
3. 清洗和去重。
4. Chunking。
5. Metadata 和权限标签。
6. Embedding。
7. 向量索引和关键词索引。

在线链路包括：

1. Query rewrite。
2. Hybrid retrieval。
3. Rerank。
4. Context construction。
5. LLM generation。
6. Citation 和 groundedness 检查。
7. 日志、反馈和 bad case 回流。

面试回答：

```text
RAG 不是简单接一个向量数据库。向量检索只是召回组件，真实系统还要处理文档解析、chunking、metadata、权限过滤、混合检索、reranker、上下文构造、引用、拒答、评估和错误归因。
```

如果 RAG 回答错了，排查路径是：

1. 正确文档是否入库。
2. 文档解析是否正确，表格、图片、标题是否丢失。
3. Chunk 是否语义完整。
4. Retriever 是否召回正确 chunk。
5. Reranker 是否把正确 chunk 排到前面。
6. Context construction 是否截断或被噪声淹没。
7. Prompt 是否要求基于证据回答和资料不足拒答。
8. LLM 是否忽略证据或编造引用。
9. 权限过滤是否误杀或漏放。

RAG 面试的关键是错误归因，不是只说“换更好的 embedding model”。

## 7.12 Agent 与工具调用部署

Agent 部署比普通 chat 更复杂，因为模型输出可能触发真实工具、API、数据库或外部系统。

一个工具调用系统通常包括：

1. Tool registry：工具名、描述、schema、权限、超时、风险等级。
2. Planner 或 policy：决定是否调用工具、调用哪个工具。
3. 参数校验：类型、枚举、范围、必填字段。
4. 权限检查：用户身份、资源权限、操作级别。
5. 执行器：真实调用 API、数据库、浏览器或代码环境。
6. 错误处理：超时、失败、重试、追问用户。
7. 审计日志：记录工具调用 trace。
8. 安全控制：高风险操作二次确认、防 prompt injection。

面试中要强调：

```text
Function calling 不只是让模型输出 JSON。完整系统必须做 schema 约束、参数校验、权限控制、执行隔离、错误处理、审计日志和安全策略。高风险写操作不能只靠模型自觉，必须由系统层做二次确认和权限边界。
```

常见风险包括：

1. 模型选错工具。
2. 参数缺失或编造参数。
3. 工具返回失败时模型编造结果。
4. Prompt injection 诱导越权调用。
5. 多步调用中状态错乱。
6. 高风险操作缺少确认。

## 7.13 延迟、吞吐和用户体验

Deployment 面试里必须区分几个指标：

1. TTFT：time to first token，首 token 延迟。
2. TPOT：time per output token，每输出 token 延迟。
3. E2E latency：端到端完成时间。
4. Throughput：tokens/s 或 requests/s。
5. QPS：每秒请求数。
6. P95/P99：尾延迟。
7. Error rate：超时、OOM、取消、内部错误。

“tokens/s 高但用户觉得卡”可能来自：

1. TTFT 很高，用户等首字太久。
2. 平均 tokens/s 高但 P99 很差。
3. 流式输出不均匀，token 间隔抖动。
4. 长请求阻塞短请求。
5. 网络、网关或客户端渲染慢。
6. safety filter、RAG 检索或工具调用前置耗时高。
7. 输出太长，整体完成时间高。

面试回答：

```text
我不会只看平均 tokens/s。用户体验更受 TTFT、流式稳定性和尾延迟影响。排查时会把链路拆成网关排队、tokenizer、RAG 或工具前处理、prefill、decode、streaming、网络和客户端渲染，分别看 P50/P95/P99。
```

## 7.14 线上故障怎么排查

部署面试经常问线上问题。回答时要先分类，再定位。

### 7.14.1 TTFT 变高

排查路径：

1. 请求排队是否变长。
2. Router 是否把流量打到热点 worker。
3. Prompt 是否变长。
4. Tokenizer CPU 是否成为瓶颈。
5. RAG 检索、rerank 或工具调用是否变慢。
6. Prefill tokens/s 是否下降。
7. Prefix cache 命中率是否下降。
8. 是否有长 prompt 抢占 prefill budget。

### 7.14.2 TPOT 变高

排查路径：

1. Running batch 是否过大。
2. KV Cache 访问是否成为瓶颈。
3. 显存是否接近上限导致频繁调度失败。
4. 是否启用了低效量化 kernel。
5. GPU utilization 和 memory bandwidth 是否异常。
6. Decode 调度是否被大量 prefill 打断。
7. Speculative decoding 接受率是否下降。

### 7.14.3 OOM 增加

排查路径：

1. 上下文长度是否变长。
2. 并发是否增加。
3. max output tokens 是否过大。
4. KV Cache block 是否泄漏。
5. 请求取消后资源是否释放。
6. 模型版本、量化配置或 KV dtype 是否变化。
7. Prefix cache 或 LoRA adapter 是否占用额外显存。

### 7.14.4 回答质量突然下降

排查路径：

1. 模型版本是否切换。
2. Tokenizer 或 chat template 是否变化。
3. system prompt 是否被覆盖。
4. RAG 索引、embedding model、reranker 是否更新。
5. Decoding 参数是否变化。
6. 量化版本是否替换。
7. 安全策略是否过度拒答。

## 7.15 成本优化：不要只说换小模型

推理成本由多个因素决定：

1. 模型大小。
2. 输入 token 数。
3. 输出 token 数。
4. 上下文长度。
5. 并发和 SLA。
6. KV Cache 显存。
7. GPU 类型和利用率。
8. 推理引擎效率。
9. 量化和 batching。
10. 冗余、灰度和故障切换。

降本可以从四层做：

1. 模型层：小模型、蒸馏、量化、LoRA merge、模型路由。
2. 引擎层：continuous batching、PagedAttention、prefix cache、speculative decoding、kernel 优化。
3. 产品层：prompt 压缩、限制 max tokens、缓存、摘要、减少无效调用。
4. 系统层：容量规划、峰谷调度、冷热模型分层、优先级队列、自动扩缩容。

面试回答要体现单位有效任务成本：

```text
最便宜的模型不一定最划算。要看 cost per successful answer。如果小模型单次调用便宜，但成功率低、重试多、人工介入多，最终有效任务成本可能更高。成本优化要同时看质量、延迟、成功率和用户体验。
```

## 7.16 高频题：如何设计一个 LLM Serving 系统

可以用下面模板回答：

```text
第一，我会先明确目标：模型规模、上下文长度、请求量、SLA、成本、安全和部署环境。

第二，设计入口层：API gateway 做鉴权、限流、配额、请求大小检查和日志；支持流式输出和取消请求。

第三，设计路由层：根据模型、租户、token 预算、GPU 负载、KV Cache 余量和优先级选择 worker，而不是只按请求数负载均衡。

第四，设计推理层：使用 vLLM、SGLang 或自研引擎，支持 tokenizer、prefill/decode、continuous batching、KV Cache 管理、prefix cache、量化和流式返回。

第五，设计可靠性：超时、重试、降级模型、灰度发布、版本回滚、健康检查和故障隔离。

第六，设计观测体系：监控 TTFT、TPOT、P95/P99、tokens/s、queue time、KV blocks、OOM、error rate、GPU utilization、成本和质量 bad case。

第七，设计安全治理：内容安全、权限控制、prompt injection 防护、工具调用审计、用户数据隔离和合规日志。
```

这个模板适合大多数推理服务系统设计题。

## 7.17 高频题：如何优化首 token 延迟

首 token 延迟通常来自排队、前处理和 prefill。

优化方向：

1. 减少排队：更好的路由、优先级、扩容、隔离长短请求。
2. 减少 prompt：prompt 压缩、历史对话摘要、减少无效 system prompt。
3. 优化 prefill：高效 attention kernel、chunked prefill、prefix cache。
4. 优化前处理：tokenizer 并行、RAG/rerank 降延迟、缓存检索结果。
5. 控制长上下文：限制最大输入、分桶调度、长请求单独队列。

回答模板：

```text
我会先把 TTFT 拆成 queue time、tokenization、RAG/tool 前处理、prefill、调度等待和网络返回。不同部分对应不同优化：排队高就优化路由和扩容，prompt 长就压缩或缓存，prefill 慢就看 kernel 和 chunked prefill，RAG 慢就看检索和 rerank。不能只笼统说换更快模型。
```

## 7.18 高频题：如何做容量规划

容量规划不能只看 QPS，要看 token 分布。

需要统计：

1. 输入 token 分布：P50、P90、P99。
2. 输出 token 分布：P50、P90、P99。
3. 峰值并发。
4. 目标 TTFT 和 TPOT。
5. 每张 GPU 的 prefill tokens/s 和 decode tokens/s。
6. KV Cache 可用显存和最大并发上下文。
7. 冗余系数和故障切换容量。

面试回答：

```text
LLM 容量规划要按 token 负载做，而不是只按请求数。两个请求的成本可能因为 prompt 长度和输出长度差几十倍。估算时我会用业务 token 分布、目标 SLA、单卡 profiling、KV Cache 显存预算和峰值并发，计算所需 GPU 数，并预留灰度、故障和流量尖峰冗余。
```

## 7.19 高频题：RAG 系统上线前评估什么

RAG 上线前要同时评估检索、生成和系统。

检索指标：

1. Recall@k。
2. Precision@k。
3. MRR。
4. nDCG。
5. Rerank latency。

生成指标：

1. Answer correctness。
2. Faithfulness。
3. Groundedness。
4. Citation accuracy。
5. Unsupported claim rate。
6. Abstention accuracy。

系统指标：

1. 端到端延迟。
2. 检索和生成成本。
3. 权限过滤正确率。
4. 文档新鲜度。
5. bad case 回流效率。

回答模板：

```text
我不会只评估最终答案分数。RAG 要分层评估：先看 retriever 是否召回正确证据，再看 reranker 是否排序正确，再看最终 context 是否包含证据，最后看 LLM 是否基于证据回答并给出正确引用。否则答案错了无法定位问题。
```

## 7.20 面试中的常见失分点

Deployment 面试常见失分点包括：

1. 把模型能跑起来等同于能上线。
2. 分不清 prefill 和 decode 的瓶颈。
3. 只说平均 latency，不看 TTFT、TPOT、P95/P99。
4. 不会估算 KV Cache 显存。
5. 把 PagedAttention 说成新的 attention 算法。
6. 认为 INT4 一定更快、质量一定可接受。
7. 只会说“用 vLLM”，讲不清调度和 KV 管理。
8. RAG 错误只会换 embedding，不会做错误归因。
9. Agent 工具调用只讲 JSON，不讲权限、审计和安全。
10. 成本优化只说换小模型，不看成功率和重试成本。
11. 忽略 tokenizer、chat template 和 stop token 的一致性。
12. 不会从监控指标定位线上问题。

## 7.21 一套完整 Deployment 面试回答模板

如果被问开放题：“请你设计一个 ChatGPT 类大模型推理服务”，可以这样组织：

```text
第一，我会先明确业务目标和约束，包括模型大小、上下文长度、QPS、输入输出 token 分布、TTFT/TPOT SLA、成本、安全和合规要求。

第二，设计整体架构。入口是 API gateway，负责鉴权、限流、请求校验和流式连接；后面是 router，根据模型版本、租户、token 预算、GPU 负载和 KV Cache 余量选择 worker；推理层使用 vLLM、SGLang 或自研 engine，负责 tokenizer、prefill、decode、continuous batching、KV Cache 管理和 streaming。

第三，优化核心瓶颈。Prefill 影响首 token，可以通过 prompt 压缩、prefix cache、chunked prefill 和高效 attention kernel 优化；decode 影响持续输出，可以通过 batching、量化、KV Cache 管理、speculative decoding 和调度策略优化。

第四，保证质量和安全。上线前比较 FP16 baseline 和量化版本，评估通用能力、业务样本、格式稳定性、安全拒答和长上下文；如果接入 RAG 或工具，还要做检索评估、引用准确率、权限控制和工具调用审计。

第五，设计可观测性和可靠性。监控 queue time、TTFT、TPOT、P95/P99、tokens/s、KV blocks、OOM、error rate、GPU utilization、RAG latency、工具调用失败率和成本；支持灰度、回滚、降级、限流和故障隔离。

第六，做容量规划和成本治理。根据 token 分布、单卡 profiling、KV Cache 显存和目标 SLA 估算 GPU 数；通过模型路由、量化、缓存、prompt 压缩、max tokens 控制和峰谷调度降低单位有效任务成本。
```

## 7.22 准备清单

准备 Deployment 面试时，至少要能回答下面的问题：

1. 从 checkpoint 到线上服务需要哪些组件？
2. Tokenizer 和 chat template 为什么必须一致？
3. Prefill 和 decode 的区别是什么？
4. TTFT、TPOT、吞吐、P95/P99 分别衡量什么？
5. KV Cache 为什么能加速推理？显存如何估算？
6. MHA、MQA、GQA 对 KV Cache 有什么影响？
7. PagedAttention 解决什么问题？它是不是新的 attention 算法？
8. Continuous batching 和普通 dynamic batching 有什么区别？
9. Chunked prefill 解决什么问题，代价是什么？
10. INT8、INT4、GPTQ、AWQ、KV Cache 量化分别要注意什么？
11. Speculative decoding 的收益由什么决定？
12. RAG 回答错了怎么做错误归因？
13. Function calling 上线要做哪些权限和安全控制？
14. TTFT 变高、TPOT 变高、OOM 增加分别怎么排查？
15. 如何估算推理服务所需 GPU 数？
16. 如何降低推理成本但不明显损伤质量？
17. 如何设计大模型 serving 的监控 dashboard？
18. 如何做模型版本灰度、回滚和降级？

这一章的核心不是让你背完所有部署系统名词，而是让你建立一套生产思维：模型质量只是起点，线上服务还要在真实流量、真实成本、真实故障和真实安全风险下稳定运行。
