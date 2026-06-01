# 第 54 章 设计一个企业级 LLMOps 平台

上一章设计了高并发大模型推理平台。本章继续系统设计题：设计一个企业级 LLMOps 平台。

LLMOps 平台关注的是大模型应用从开发、评估、发布、观测、治理到迭代的完整生命周期。它不是单纯训练平台，也不是单纯推理平台，而是把模型、prompt、RAG、Agent、工具、安全、评估、发布和成本整合起来的企业级平台。

先记住一句话：

> 企业级 LLMOps 平台的核心，是让大模型应用可开发、可评估、可发布、可观测、可治理、可审计、可持续优化。

## 54.1 题目理解

面试题可能这样问：

1. 请设计一个企业内部 LLMOps 平台。
2. 请设计一个支持 prompt、RAG、Agent、评估和发布的大模型应用平台。
3. 请设计一个多团队共享的大模型应用生命周期管理平台。
4. 请设计一个统一管理模型调用、知识库、工具和 trace 的平台。

这类题重点不是模型训练，而是大模型应用落地和治理。

## 54.2 LLMOps 和 MLOps 的区别

MLOps 更关注传统机器学习生命周期：

1. 数据。
2. 特征。
3. 训练。
4. 部署。
5. 监控。

LLMOps 额外关注：

1. Prompt 版本。
2. RAG 知识库。
3. Agent 工具调用。
4. 模型路由。
5. 输出安全。
6. 用户反馈。
7. LLM-as-Judge。
8. Trace 回放。
9. Token 成本。
10. 多模型能力治理。

LLMOps 更偏“大模型应用生命周期管理”。

## 54.3 需求澄清

设计前先问：

1. 平台服务哪些用户？算法、应用开发、业务团队、运维？
2. 是否接入多个基础模型和外部模型？
3. 是否支持 prompt 管理？
4. 是否支持 RAG 知识库？
5. 是否支持 Agent 和工具？
6. 是否支持评估和 A/B 测试？
7. 是否需要多租户隔离？
8. 是否有安全、合规和审计要求？
9. 是否需要成本预算和计量？
10. 是否对外提供 API/SDK？

企业级平台必须明确用户角色和治理要求。

## 54.4 核心目标

企业级 LLMOps 平台目标：

1. 统一模型接入和调用。
2. 管理 prompt 版本和发布。
3. 管理知识库和 RAG 链路。
4. 管理 Agent 和工具定义。
5. 支持评估、实验和对比。
6. 支持灰度、A/B 测试和回滚。
7. 支持 trace、日志和可观测性。
8. 支持权限、安全和审计。
9. 支持 token 成本和预算治理。
10. 提供 Web Console、API、SDK 和 CLI。

LLMOps 平台本质上是企业大模型应用的控制平面。

## 54.5 总体架构

可以设计如下架构：

```text
Users / Apps / Agents
  -> LLMOps API Gateway
  -> Auth / Tenant / Quota
  -> Model Gateway / Router
  -> Prompt Registry
  -> Knowledge Base / RAG Service
  -> Tool Registry / Agent Runtime
  -> Eval Platform
  -> Release / Experiment System
  -> Trace / Observability
  -> Cost / Audit / Governance
  -> Model Registry / Inference Platform
```

底层依赖推理平台、模型仓库、向量索引、评估平台和可观测性平台。

LLMOps 把这些能力包装成应用开发者能使用的产品。

## 54.6 Model Gateway

Model Gateway 是统一模型调用入口。

能力包括：

1. 多模型接入。
2. OpenAI-compatible API。
3. 模型路由。
4. 租户鉴权。
5. 限流配额。
6. 统一日志和 trace。
7. token 计量。
8. fallback 和降级。

它让应用不直接绑定某个模型供应商或某个内部 endpoint。

## 54.7 Prompt Registry

Prompt Registry 管理 prompt 模板。

它应支持：

1. prompt 创建。
2. prompt 版本。
3. 变量 schema。
4. prompt 预览。
5. 测试样例。
6. 评估结果。
7. 发布状态。
8. 灰度和回滚。
9. 权限控制。
10. 审计记录。

Prompt 是大模型应用的重要资产。

不要把 prompt 当成代码里的硬编码字符串。

## 54.8 Prompt 发布流程

Prompt 发布可以类似模型发布：

```text
draft -> reviewed -> evaluated -> staged -> canary -> production -> deprecated
```

发布前检查：

1. 变量是否完整。
2. 是否通过格式测试。
3. 是否通过安全测试。
4. 是否通过评估集。
5. 是否影响工具调用。
6. 是否有回滚版本。

Prompt 修改可能导致输出格式、拒答率和工具调用行为变化，因此必须可灰度和回滚。

## 54.9 Knowledge Base 管理

RAG 知识库管理包括：

1. 知识库创建。
2. 文档导入。
3. 文档解析。
4. chunking。
5. embedding。
6. 向量索引。
7. 权限同步。
8. 索引版本。
9. 检索配置。
10. citation。

企业知识库必须支持权限过滤。

不能让用户检索到自己无权访问的文档。

## 54.10 RAG Pipeline

RAG 请求链路：

```text
user query
  -> query rewrite
  -> retrieval
  -> metadata permission filter
  -> rerank
  -> prompt assembly
  -> model generation
  -> citation
  -> trace
```

平台要记录 retrieval trace 和 prompt assembly trace。

否则无法解释为什么模型引用了某些文档，或者为什么没有召回正确知识。

## 54.11 Tool Registry

Tool Registry 管理可供 Agent 调用的工具。

Tool Definition 包括：

1. tool name。
2. version。
3. description。
4. input schema。
5. output schema。
6. endpoint。
7. timeout。
8. retry policy。
9. permission requirement。
10. side effect level。

工具描述会直接影响模型调用行为，因此工具也需要版本化、评估和灰度。

## 54.12 Agent Runtime

Agent Runtime 负责任务执行。

它需要支持：

1. agent definition。
2. 多步执行。
3. 工具调用。
4. memory。
5. run state。
6. 超时。
7. step limit。
8. human confirmation。
9. execution trace。
10. error handling。

Agent 不是一次模型调用，而是一个多步状态机。

平台必须限制无限循环和高风险工具调用。

## 54.13 权限和安全

LLMOps 平台安全包括：

1. 模型访问权限。
2. prompt 修改权限。
3. 知识库访问权限。
4. 工具调用权限。
5. trace 查看权限。
6. secret 管理。
7. 输入输出安全检查。
8. 审计日志。

特别注意：应用开发者不一定能访问所有模型、所有知识库、所有工具。

权限要跟租户、项目、环境和数据等级绑定。

## 54.14 Eval Platform 集成

LLMOps 平台必须集成评估。

评估对象包括：

1. 模型版本。
2. prompt 版本。
3. RAG 配置。
4. 知识库版本。
5. 工具定义版本。
6. Agent 配置。

评估方式：

1. 离线评估。
2. LLM-as-Judge。
3. 人工评测。
4. 在线 A/B 测试。
5. 回归测试。

没有评估，LLMOps 只能靠感觉发布。

## 54.15 Release System

发布系统管理应用版本。

一个 LLM 应用版本可能包括：

1. model route policy。
2. prompt version。
3. RAG config。
4. knowledge base version。
5. tool versions。
6. safety policy。
7. generation config。

这些组合应该形成 release manifest。

发布支持：

1. 灰度。
2. A/B 测试。
3. 回滚。
4. 审批。
5. 变更记录。

## 54.16 Trace 和可观测性

LLMOps trace 应覆盖：

1. model call trace。
2. prompt assembly trace。
3. retrieval trace。
4. rerank trace。
5. tool call trace。
6. agent execution trace。
7. safety events。
8. cost trace。

核心指标：

1. latency。
2. TTFT。
3. TPOT。
4. token usage。
5. cost。
6. cache hit rate。
7. retrieval hit rate。
8. tool success rate。
9. user feedback。
10. safety violation rate。

大模型应用没有 trace 就很难 debug。

## 54.17 用户反馈闭环

企业 LLMOps 平台要支持反馈收集。

反馈类型：

1. 点赞点踩。
2. 用户评论。
3. 人工接管。
4. 任务成功失败。
5. 引用是否有用。
6. 工具调用是否正确。
7. 安全投诉。

反馈用于：

1. 评估数据构建。
2. 回归集更新。
3. prompt 优化。
4. RAG 调优。
5. 模型选择。
6. 安全规则改进。

反馈要和 trace 关联。

## 54.18 成本治理

LLMOps 成本治理关注：

1. token usage。
2. cost per app。
3. cost per tenant。
4. cost per user。
5. cost per model。
6. RAG 检索成本。
7. rerank 成本。
8. tool API 成本。
9. agent run 成本。

Agent 特别容易成本失控，因为它会多轮调用模型和工具。

平台要支持 budget、quota、step limit、max tokens 和 cost alert。

## 54.19 多环境管理

企业平台通常有多个环境：

1. dev。
2. staging。
3. production。

LLM 应用在不同环境中应有独立配置：

1. 模型版本。
2. prompt 版本。
3. 知识库版本。
4. 工具 endpoint。
5. 安全策略。
6. 成本配额。

从 dev 到 production 应通过发布流程，而不是手工复制配置。

## 54.20 Web Console 设计

Console 功能可以包括：

1. 应用列表。
2. 模型网关配置。
3. Prompt 管理。
4. 知识库管理。
5. 工具管理。
6. Agent 配置。
7. 评估任务。
8. 发布和灰度。
9. Trace 查看。
10. 成本和配额。
11. 权限和审计。

不同角色看到的页面不同。

应用开发者关注调试和发布，平台管理员关注安全、成本和稳定性。

## 54.21 API / SDK / CLI

平台应提供统一接口。

SDK 示例能力：

1. 调用模型。
2. 获取 prompt version。
3. 查询知识库。
4. 调用 agent。
5. 上传反馈。
6. 获取 trace ID。

CLI 示例：

```bash
llmops prompt publish customer-bot --version v12
llmops kb sync product-docs
llmops eval run customer-bot --suite regression
llmops release canary customer-bot --percent 5
llmops trace get <trace-id>
```

API、SDK、CLI 和 Console 应基于同一套资源模型。

## 54.22 核心数据模型

核心对象：

1. Application。
2. Environment。
3. PromptTemplate。
4. KnowledgeBase。
5. ToolDefinition。
6. AgentDefinition。
7. EvalSuite。
8. ReleaseManifest。
9. Trace。
10. Feedback。
11. Policy。
12. CostRecord。

这些对象构成 LLMOps 平台的控制平面。

## 54.23 关键 trade-off

LLMOps 平台 trade-off：

1. 平台统一治理 vs 团队灵活性。
2. Trace 完整性 vs 隐私和成本。
3. Prompt 快速迭代 vs 发布风险。
4. RAG 召回更多内容 vs 权限和噪声风险。
5. Agent 自动化能力 vs 工具副作用风险。
6. 多模型路由优化成本 vs 输出一致性。
7. 强审批流程 vs 研发效率。

企业平台的难点往往不是技术组件，而是治理边界和协作流程。

## 54.24 面试回答模板

可以这样回答：

```text
我会把企业级 LLMOps 平台设计成大模型应用生命周期管理平台，覆盖模型调用、prompt、RAG、Agent、评估、发布、观测、安全和成本。

架构上包括 Model Gateway、Prompt Registry、Knowledge Base/RAG Service、Tool Registry、Agent Runtime、Eval Platform、Release System、Trace/Observability、Cost/Audit/Governance。底层对接模型仓库和推理平台。

应用开发者通过 Web Console、API、SDK 或 CLI 管理 prompt、知识库、工具和 agent 配置。每次发布生成 release manifest，包含模型路由、prompt 版本、RAG 配置、工具版本、安全策略和推理参数。发布前跑评估和回归，发布时支持灰度、A/B 测试和回滚。

运行中记录 model call、retrieval、prompt assembly、tool call 和 agent execution trace，并采集 token、latency、cost、feedback 和 safety 指标。平台通过租户权限、模型访问控制、知识库 ACL、工具权限、审计和预算配额保证企业治理。
```

这个回答体现了企业级 LLMOps 的完整边界。

## 54.25 常见扣分点

扣分点一：只说模型部署。

LLMOps 不只是部署模型，还包括 prompt、RAG、Agent、评估、发布和治理。

扣分点二：不提 prompt 版本。

Prompt 是应用行为的重要组成，必须版本化和灰度。

扣分点三：不提知识库权限。

企业 RAG 的核心风险是越权检索。

扣分点四：不提 trace。

没有 trace，RAG/Agent 应用无法调试和审计。

扣分点五：不提成本。

企业平台需要 token 计量、预算、配额和成本归因。

## 54.26 小练习

1. LLMOps 和 MLOps 的核心区别是什么？
2. Prompt Registry 应该支持哪些能力？
3. 一个 LLM 应用 release manifest 应包含哪些内容？
4. 企业 RAG 平台如何做权限控制？
5. Agent Runtime 为什么需要 step limit 和 human confirmation？
6. LLMOps trace 应覆盖哪些阶段？
7. 用户反馈如何进入模型和应用迭代闭环？
8. 如何设计 LLMOps 平台的成本治理？

## 54.27 本章小结

本章系统设计了一个企业级 LLMOps 平台。

你需要记住：

1. LLMOps 关注大模型应用生命周期，而不只是模型训练或部署。
2. 企业级 LLMOps 平台要管理模型调用、prompt、RAG、Agent、工具、评估、发布、trace、安全和成本。
3. Prompt、知识库、工具和 Agent 配置都要版本化、评估、灰度和回滚。
4. RAG/Agent 场景中，trace、权限和审计是核心能力。
5. 企业平台要在研发效率、治理、安全、成本和灵活性之间做平衡。

下一章我们会设计一个多租户 GPU 集群调度系统。
