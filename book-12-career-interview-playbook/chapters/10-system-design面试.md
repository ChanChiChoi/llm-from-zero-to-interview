# 第十章：System Design 面试

System Design 面试考的不是你能不能画出一张复杂架构图，而是你能不能在一个开放、模糊、有约束的问题里，主动澄清目标，拆出核心模块，识别瓶颈和 trade-off，并给出可以落地、可以评估、可以演进的系统方案。

大模型系统设计和传统后端系统设计不同。它不仅有流量、存储、缓存、消息队列和可用性问题，还多了模型质量、推理延迟、token 成本、RAG 事实性、Agent 工具安全、评估闭环、模型灰度、prompt 版本和安全治理等问题。

很多候选人这一关失分，不是因为完全不会系统设计，而是回答没有主线：一上来堆 API Gateway、Redis、Kafka、向量数据库、Kubernetes、Prometheus，却讲不清请求怎么流动、核心瓶颈是什么、为什么这样拆模块、系统如何评估、失败时如何降级。

本章重点：系统设计答题流程、ChatGPT 服务、训练平台、推理平台、RAG、Agent、多模态助手、评估平台、安全和成本治理。

## 10.1 System Design 面试到底考什么

大模型系统设计面试通常考六类能力：

1. 需求澄清：能否把“设计一个 ChatGPT”这种宽题变成有边界的问题。
2. 架构拆解：能否拆出在线链路、离线链路、模型服务、数据服务、安全和观测。
3. 核心机制：是否理解 RAG、推理调度、KV Cache、工具调用、评估平台等大模型特有模块。
4. Trade-off：能否解释质量、延迟、吞吐、成本、安全和可维护性之间的取舍。
5. 故障意识：能否处理超时、OOM、检索失败、模型幻觉、权限泄露、缓存错误和评估污染。
6. 演进能力：能否从 MVP 讲到生产化，再讲到持续优化和反馈闭环。

面试官想看到的是你能不能像负责过系统的人一样思考，而不是背一个标准答案。

## 10.2 一套通用答题流程

遇到任何大模型系统设计题，都可以按六步回答。

第一步，澄清需求。

可以问：

1. 面向 C 端、企业内部、开发者 API，还是研究平台？
2. 支持文本，还是还要图片、语音、视频？
3. 是否需要 RAG、工具调用、多轮对话和记忆？
4. 流量规模、文档规模、模型规模大概是多少？
5. 主要优化目标是质量、延迟、成本、安全还是可用性？
6. 是否有多租户、权限隔离、合规和审计要求？

第二步，列功能需求和非功能需求。

功能需求是系统要做什么。非功能需求是系统要做到什么程度。

例如设计 RAG，功能需求包括文档上传、解析、切分、索引、检索、重排、生成、引用和反馈；非功能需求包括权限安全、低延迟、高召回、答案可追溯、知识更新及时和成本可控。

第三步，给高层架构。

先给主链路，不要直接陷入数据库字段。

常用骨架：

```text
Client -> Gateway -> Orchestrator -> Retrieval/Tool/Cache -> Model Serving -> Guard -> Response -> Logs/Eval
```

第四步，深入核心模块。

不同题目的核心模块不同：

1. ChatGPT 服务：会话、上下文、推理、流式输出、安全。
2. 推理平台：路由、调度、batching、KV Cache、模型版本、GPU 利用率。
3. RAG：文档解析、chunk、检索、rerank、引用、权限。
4. Agent：工具注册、规划、状态、权限、失败恢复、审计。
5. 评估平台：数据集、模型输出、指标、judge、人工评审、回归门禁。

第五步，讲权衡和失败场景。

系统设计没有唯一答案。你需要主动讲：为什么这样设计、有什么代价、失败时怎么办。

第六步，讲演进路线。

可以分三阶段：

1. MVP：打通核心链路。
2. 生产化：补监控、评估、安全、灰度、回滚。
3. 优化：做成本治理、自动路由、数据闭环和持续评估。

## 10.3 大模型系统的通用架构积木

很多系统题底层积木相似。

### 10.3.1 接入层

接入层包括 API Gateway、鉴权、限流、租户识别、请求大小限制、基础日志。

大模型系统里，限流不能只按 request 数，还要按 input tokens、output tokens、并发数、模型等级和租户配额。

### 10.3.2 编排层

编排层决定一次请求怎么执行：是否需要 RAG、是否调用工具、用哪个模型、是否走安全审核、是否命中缓存。

大模型应用的复杂度通常在编排层，而不是单个模型调用。

### 10.3.3 模型服务层

模型服务层负责推理，核心包括模型加载、tokenizer、chat template、prefill/decode、KV Cache、batching、量化、流式输出和超时取消。

### 10.3.4 数据与存储层

不同系统需要不同存储：

1. Chat 服务需要会话存储和用户反馈。
2. RAG 需要对象存储、元数据、向量索引、倒排索引。
3. Agent 需要工具状态、执行 trace 和审计日志。
4. 评估平台需要数据集版本、实验结果和人工标注。

### 10.3.5 安全与权限层

安全不是外挂模块，而是全链路能力。包括输入审核、输出审核、RAG 权限过滤、工具最小权限、PII 检测、prompt injection 防护和审计。

### 10.3.6 观测与评估层

观测指标包括 latency、TTFT、TPOT、tokens/s、error rate、OOM、cost、user feedback、safety violation、RAG citation accuracy、tool success rate。

没有观测和评估，就无法判断系统变更是否真的变好。

## 10.4 设计 ChatGPT 类服务

题目可以这样澄清：

```text
我先按一个文本为主的 ChatGPT 类服务设计，支持多轮对话、流式输出、上下文管理、安全审核、可选 RAG 和工具调用，目标是低延迟、高可用、成本可控和安全合规。
```

功能需求：

1. 用户登录和会话管理。
2. 多轮对话。
3. 流式输出。
4. 上下文构建和历史压缩。
5. 模型推理。
6. 安全审核。
7. 用户反馈。
8. 模型灰度和回滚。

高层架构：

```text
Web/App
  -> API Gateway
  -> Auth / Rate Limit
  -> Conversation Service
  -> Request Orchestrator
  -> Context Builder
  -> Safety Input Guard
  -> Retrieval / Tool / Cache
  -> Model Router
  -> Model Serving
  -> Safety Output Guard
  -> Streaming Response
  -> Logs / Metrics / Evaluation
```

核心模块：

1. Conversation Service 保存会话、消息、上下文摘要和用户偏好。
2. Context Builder 决定哪些历史进入 prompt，避免无限堆上下文。
3. Model Router 根据任务难度、上下文长度、租户等级和成本选择模型。
4. Safety Guard 做输入输出安全检查。
5. Streaming Response 降低体感延迟。
6. Feedback Loop 收集点赞、点踩、重试、追问和人工标注。

关键 trade-off：

1. 上下文越长，理解越好，但成本和延迟越高。
2. 安全越严格，风险越低，但误拒可能增加。
3. 强模型质量更好，但成本更高。
4. 流式输出体验更好，但输出审核更复杂。

常见追问：如何处理超长历史？

回答：

```text
我会结合滑动窗口、会话摘要、重要消息保留、用户显式 pin、历史检索和任务相关性筛选。不能简单保留最后 N 条，因为早期约束可能很重要。
```

## 10.5 设计大模型推理平台

题目可以这样澄清：

```text
我先按一个服务多业务、多模型的在线推理平台设计，支持模型注册、版本管理、部署、路由、动态 batching、流式输出、GPU 调度、监控、灰度和回滚。目标是提高 GPU 利用率，同时满足不同业务 SLA。
```

高层架构：

```text
Business Service
  -> Inference Gateway
  -> Auth / Quota / Rate Limit
  -> Model Router
  -> Request Scheduler
  -> Runtime Workers
  -> GPU Cluster
  -> Metrics / Logs / Traces
  -> Model Registry / Config Center
```

核心模块：

1. Model Registry 保存模型版本、tokenizer、runtime config、资源需求和能力标签。
2. Router 根据模型、租户、token 预算、延迟 SLA 和负载选择 worker。
3. Scheduler 控制 waiting queue、running batch、prefill/decode、token budget 和 KV Cache budget。
4. Runtime Worker 加载模型，执行推理，管理 KV Cache，流式返回。
5. Observability 记录 TTFT、TPOT、queue time、tokens/s、OOM、error rate、GPU utilization。

关键 trade-off：

1. Batch 越大吞吐越高，但等待时间更长。
2. 副本越多可用性越高，但显存成本越高。
3. 量化降低成本，但可能影响质量和格式稳定性。
4. 多租户共享提高利用率，但隔离和公平性更难。

常见追问：延迟变高怎么排查？

回答：

```text
我会把延迟拆成 gateway、queue、tokenization、prefill、decode、output guard、network 几段。如果 TTFT 高，重点看排队、prompt 长度、prefill 和 RAG/tool 前处理；如果 TPOT 高，重点看 decode batch、KV Cache、显存带宽、量化 kernel 和调度策略。
```

## 10.6 设计训练平台

训练平台题考察你是否理解大模型训练不是单个脚本，而是数据、调度、分布式训练、checkpoint、监控、评估和成本的系统。

题目澄清：

```text
我先按一个支持预训练、继续预训练和 SFT 的训练平台设计。平台支持数据集管理、训练任务提交、资源调度、分布式训练、checkpoint、监控、评估、实验追踪和失败恢复。
```

高层架构：

```text
User / Researcher
  -> Experiment Portal / CLI
  -> Config & Dataset Registry
  -> Job Scheduler
  -> Training Workers on GPU Cluster
  -> Checkpoint Store
  -> Metrics / Logs / Traces
  -> Evaluation Pipeline
  -> Experiment Tracker
```

核心模块：

1. Dataset Registry 管理数据版本、来源、过滤策略、token 数和权限。
2. Config System 管理模型、优化器、batch、学习率、并行策略和随机种子。
3. Scheduler 分配 GPU、处理队列、优先级、抢占和失败重试。
4. Training Runtime 支持 DDP、FSDP、ZeRO、tensor parallel、pipeline parallel。
5. Checkpoint System 保存模型、optimizer、scheduler、RNG、step、sampler 状态。
6. Monitoring 监控 loss、gradient norm、learning rate、throughput、MFU、GPU utilization、NaN/Inf。
7. Evaluation Pipeline 定期跑 validation loss、benchmark、安全和业务评估。

关键 trade-off：

1. Checkpoint 越频繁，恢复损失越小，但 I/O 成本越高。
2. 更复杂并行策略能训练更大模型，但通信和调试更难。
3. 数据加载越复杂，质量越高，但吞吐可能受影响。
4. 自动恢复能节省成本，但必须防止 silent bug。

常见追问：训练 loss spike 怎么接入平台治理？

回答：

```text
平台要记录 step、rank、batch id、数据 shard、learning rate、gradient norm、NaN/Inf、checkpoint 版本和代码版本。发现 spike 后先判断是全局还是局部、单点还是持续，再支持回滚到稳定 checkpoint，并把异常 batch 和日志保留下来用于排查。
```

## 10.7 设计 RAG 系统

题目澄清：

```text
我先按一个企业知识库 RAG 系统设计，支持文档上传、解析、切分、索引、权限控制、混合检索、rerank、答案生成、引用溯源、反馈和知识库更新。
```

离线链路：

```text
Upload -> Parse -> Clean -> Chunk -> Metadata/Permission -> Embed -> Index -> Version
```

在线链路：

```text
Query -> Rewrite -> Permission Filter -> Hybrid Retrieve -> Rerank -> Context Builder -> LLM -> Citation -> Feedback
```

核心模块：

1. 文档解析处理 PDF、Word、网页、表格、图片 OCR 和代码。
2. Chunking 根据标题、段落、表格、代码块和语义边界切分。
3. Metadata 保存权限、来源、版本、时间、部门和文档类型。
4. Hybrid Retrieval 结合向量检索和 BM25。
5. Reranker 提高最终上下文精度。
6. Context Builder 控制 token budget、去重、排序和引用。
7. Citation 检查答案声明是否被证据支持。

关键 trade-off：

1. Chunk 太小语义不完整，太大噪声多。
2. Top-k 太小漏召回，太大增加噪声和成本。
3. 强 reranker 质量好但延迟高。
4. 严格引用减少幻觉，但可能降低回答流畅性。

常见追问：RAG 回答错了怎么排查？

回答：

```text
我会做错误归因：先看正确文档是否入库，再看解析和 chunk 是否正确，然后看 retriever 是否召回、reranker 是否排前、context 是否包含证据，最后看 LLM 是否忽略证据、编造引用或资料不足时没有拒答。
```

## 10.8 设计 Agent 平台

Agent 平台题的核心不是“让模型自动规划”，而是让工具调用可靠、安全、可观测、可恢复。

题目澄清：

```text
我先按一个企业 Agent 平台设计，支持工具注册、权限控制、任务规划、工具执行、状态管理、失败恢复、trace 日志、评估和安全审计。
```

高层架构：

```text
User Request
  -> Gateway / Auth
  -> Agent Orchestrator
  -> Planner / Tool Selector
  -> Tool Registry
  -> Permission & Policy Check
  -> Tool Executor / Sandbox
  -> State Store / Memory
  -> LLM Response
  -> Trace / Audit / Evaluation
```

核心模块：

1. Tool Registry 记录工具名、描述、schema、权限、风险等级和超时。
2. Planner 决定是否调用工具以及调用顺序。
3. Permission Check 做用户、租户、资源和操作级别校验。
4. Executor 执行工具，处理超时、重试、错误和幂等。
5. State Store 保存任务状态、工具结果和中间计划。
6. Trace 记录每一步 reasoning 摘要、tool call、参数、结果和错误。
7. Evaluation 测工具选择、参数正确率、任务成功率、安全违规率。

关键 trade-off：

1. Agent 自主性越强，能力越强，但风险越高。
2. 工具越多，覆盖越广，但选择错误率和权限复杂度上升。
3. 多步计划能解决复杂任务，但延迟、成本和失败概率增加。
4. 沙箱越严格越安全，但工具能力可能受限。

常见追问：如何防 prompt injection？

回答：

```text
外部网页、邮件、文档和工具返回都要视为 untrusted content，不能直接覆盖 system instruction 或触发高权限工具。工具调用必须由系统做 schema 校验、权限检查和高风险二次确认，不能只相信模型输出。
```

## 10.9 设计多模态助手

多模态助手题要同时覆盖图片理解、文本对话、RAG、工具、安全和延迟。

题目澄清：

```text
我先按一个支持图片输入和文本对话的多模态助手设计，支持图片问答、OCR、图表解释、截图理解、可选 RAG 和安全审核。目标是准确理解视觉内容，并给出可追溯、低延迟、安全的回答。
```

高层架构：

```text
Image/Text Input
  -> Gateway
  -> Image Preprocessor / OCR
  -> Vision Encoder / VLM
  -> Context Builder
  -> Optional Retrieval / Tool
  -> LLM / VLM Serving
  -> Safety & Privacy Guard
  -> Response / Citation
  -> Logs / Evaluation
```

核心模块：

1. 图片预处理：尺寸、格式、压缩、OCR、隐私检测。
2. VLM 服务：vision encoder、connector、LLM 或统一多模态模型。
3. 多模态 RAG：图片 caption、OCR、metadata、文本和图像向量索引。
4. 安全：人脸、证件、屏幕截图、OCR prompt injection、版权和危险内容。
5. 评估：OCR、图表、计数、空间关系、视觉幻觉、多图推理。

关键 trade-off：

1. 高分辨率保留细节，但 token 和延迟成本更高。
2. OCR 专用模型更准，但链路更复杂。
3. VLM 端到端更统一，但错误归因更难。
4. 多模态安全比文本更复杂，因为图像里也可能包含指令和敏感信息。

## 10.10 设计评估平台

评估平台题非常适合展示你对大模型迭代闭环的理解。

题目澄清：

```text
我先按一个支持模型、prompt、RAG、Agent 和 safety 的评估平台设计。平台支持数据集管理、批量生成、自动指标、LLM-as-a-judge、人工评审、回归测试、A/B 实验和发布门禁。
```

高层架构：

```text
Dataset Registry
  -> Eval Job Scheduler
  -> Model Runner
  -> Metric Calculator
  -> Judge Model Service
  -> Human Review Tool
  -> Result Store
  -> Report Dashboard
  -> Regression Gate
```

核心模块：

1. Dataset Registry 管理数据集版本、来源、任务类型、难度、标签和权限。
2. Model Runner 记录模型、prompt、解码参数、工具版本和知识库版本。
3. Metric Calculator 支持准确率、F1、EM、代码测试、retrieval recall、citation accuracy。
4. Judge Service 用 rubric 做 pairwise 或 scalar 评估。
5. Human Review 处理高风险和主观任务。
6. Regression Gate 接入发布流程，阻止质量或安全回退。

关键 trade-off：

1. 自动评测快但可能不符合人类偏好。
2. 人工评审准确但贵且慢。
3. LLM judge 可扩展但有位置偏差、长度偏差和模型偏好。
4. 固定评测集可比较但会过拟合，动态评测集更真实但历史可比性差。

常见追问：如何避免评测污染？

回答：

```text
评测集要分公开集、内部保密集、动态集和线上真实样本集。核心集限制访问，只展示聚合结果。训练数据和评测数据做近重复检测，模型发布前跑保密回归集，线上再用真实反馈校验。
```

## 10.11 设计安全审核系统

安全审核系统题考察你是否能把 safety 从模型能力变成系统能力。

高层架构：

```text
Request
  -> Input Classifier / Policy Engine
  -> Risk Router
  -> Model / Tool / Retrieval
  -> Output Classifier
  -> Human Review for High Risk
  -> Audit Logs / Red Team Regression
```

核心设计：

1. 风险 taxonomy：有害内容、隐私、版权、prompt injection、工具滥用、高风险建议。
2. 输入审核：判断请求风险和允许的响应等级。
3. 输出审核：检查模型是否泄露、违规或过度自信。
4. Policy Engine：按地区、业务、用户等级和风险类别执行策略。
5. Human Review：处理高风险或低置信样本。
6. Regression Suite：红队样本和线上事故进入回归集。

关键 trade-off：

1. 拦截越严，风险越低，但误杀越高。
2. 人工审核更可靠，但延迟和成本高。
3. 安全策略透明有助于用户理解，但也可能被滥用者利用。

## 10.12 面试追问怎么应对

系统设计面试后半段通常会不断追问。

### 10.12.1 流量扩大十倍怎么办

回答方向：

1. 水平扩展 gateway、orchestrator 和 stateless 服务。
2. 增加模型副本和 GPU worker。
3. 按 token 而不是请求数负载均衡。
4. 缓存热点 prompt、检索结果和响应。
5. 大小模型路由和降级。
6. 按租户限流和优先级队列。

### 10.12.2 成本太高怎么办

回答方向：

1. 模型路由：简单请求用小模型，复杂请求用大模型。
2. Prompt 压缩和上下文裁剪。
3. 限制 max output tokens。
4. Prefix cache、response cache、retrieval cache。
5. 量化、batching、speculative decoding。
6. 离线预计算 embedding 和热门答案。
7. 监控 cost per successful answer。

### 10.12.3 质量下降怎么办

回答方向：

1. 确认模型、prompt、template、tokenizer、知识库版本是否变化。
2. 用评估平台复现问题。
3. 做 bad case 分类。
4. 区分检索问题、模型问题、工具问题、安全误杀和数据过期。
5. 回滚或灰度修复。

### 10.12.4 数据泄露怎么办

回答方向：

1. 立即限制相关功能和访问。
2. 定位泄露路径：RAG 权限、日志、缓存、工具、模型输出。
3. 保留审计日志。
4. 修复权限和脱敏策略。
5. 通知合规流程。
6. 加入回归测试和监控。

## 10.13 常见失分点

System Design 面试常见失分点包括：

1. 一上来画架构，不澄清需求。
2. 只堆组件名，不讲数据流。
3. 只讲在线链路，不讲离线索引、评估和反馈。
4. 只讲模型，不讲 tokenizer、prompt、版本、缓存和监控。
5. 设计 RAG 时忽略权限和引用。
6. 设计 Agent 时忽略工具权限、审计和失败恢复。
7. 设计推理平台时不讲 batching、KV Cache、TTFT、TPOT。
8. 不讲安全、隐私和合规。
9. 不讲成本估算和降本策略。
10. 不讲灰度、回滚和发布门禁。
11. 不讲指标，无法判断系统是否成功。
12. 遇到追问只加组件，不解释 trade-off。

## 10.14 高频题回答模板

### 10.14.1 设计 ChatGPT 服务

```text
我会先澄清用户、规模、SLA、是否需要 RAG 和工具。架构上分为 gateway、会话服务、上下文构建、编排层、模型路由、推理服务、安全审核、流式返回和观测评估。核心难点是上下文管理、低延迟推理、安全、成本和模型迭代。MVP 先支持多轮文本和流式输出，生产化后补 RAG、工具、反馈闭环、灰度和回滚。
```

### 10.14.2 设计 RAG 系统

```text
我会拆成离线文档链路和在线查询链路。离线链路做解析、清洗、chunk、metadata、权限、embedding 和索引；在线链路做 query rewrite、权限过滤、hybrid retrieval、rerank、context construction、LLM 生成、citation 和反馈。核心 trade-off 是召回率、上下文噪声、延迟、权限和引用准确性。
```

### 10.14.3 设计推理平台

```text
推理平台要支持模型注册、版本管理、路由、调度、GPU worker、batching、KV Cache、流式输出、监控和灰度。路由不能只按请求数，要按 token 预算、模型、SLA、GPU 负载和 KV Cache 余量。核心指标是 TTFT、TPOT、吞吐、P95/P99、OOM、GPU 利用率和成本。
```

### 10.14.4 设计评估平台

```text
评估平台要管理数据集版本、模型和 prompt 版本、批量生成、自动指标、LLM-as-judge、人工评审、回归测试和报告。所有评测结果必须记录模型版本、解码参数、工具版本和知识库版本，保证可复现。上线前用 regression gate 防止质量、安全和成本回退。
```

## 10.15 一套完整 System Design 回答模板

如果被问任意大模型系统设计题，可以用下面模板组织：

```text
第一，我会先澄清场景和目标，包括用户是谁、输入输出是什么、是否需要 RAG/工具/多模态、流量规模、SLA、安全和成本约束。

第二，我会列功能需求和非功能需求。功能需求描述系统做什么，非功能需求包括延迟、吞吐、可用性、权限、隐私、可观测、可评估和成本。

第三，我会给高层架构。通常包括 gateway、auth/rate limit、orchestrator、retrieval/tool/cache、model serving、safety guard、storage、observability 和 evaluation。

第四，我会深入核心模块。不同题目重点不同：RAG 讲文档和检索，推理平台讲调度和 KV Cache，Agent 讲工具和权限，评估平台讲数据集和门禁。

第五，我会讲 trade-off 和失败场景，例如质量和成本、延迟和准确率、安全和误杀、缓存和隐私、召回和噪声，并给出降级、回滚和监控方案。

第六，我会讲演进路线。第一阶段打通 MVP，第二阶段补齐监控、安全和评估，第三阶段做自动路由、成本优化和反馈闭环。
```

## 10.16 面试前一页速记

澄清问题：

1. 用户是谁？
2. 输入输出是什么？
3. 是否需要多轮、RAG、工具、多模态？
4. 规模和 SLA 是什么？
5. 安全、隐私、权限和成本要求是什么？

架构主线：

```text
Gateway -> Auth/RateLimit -> Orchestrator -> Retrieval/Tool/Cache -> Model Serving -> Safety Guard -> Response -> Logs/Eval
```

必讲指标：

1. 延迟：TTFT、TPOT、P95、P99。
2. 吞吐：QPS、tokens/s、GPU utilization。
3. 质量：任务成功率、人工胜率、用户满意度。
4. RAG：Recall@k、MRR、citation accuracy、faithfulness。
5. Agent：tool selection accuracy、argument accuracy、task success rate。
6. 安全：harmful compliance、over-refusal、prompt injection success rate。
7. 成本：每请求成本、每百万 token 成本、GPU 小时。

必讲能力：

1. 权限隔离。
2. 安全审核。
3. 监控告警。
4. 评估体系。
5. 灰度回滚。
6. 成本优化。
7. 故障降级。
8. 数据和反馈闭环。

这一章的核心不是让你背一个架构图，而是让你形成一种面试现场的系统思维：先定边界，再画主链路，深入核心模块，主动讲 trade-off，最后用指标和演进路线收束。
