# Agent 基础概念与风险资料

记录日期：2026-11-13。此文件保存本轮联网核对的新知识点及来源证据，不记录书籍审阅决定或修改历史。

## 来源与阅读位置

### ReAct

- 来源：[arXiv:2210.03629 v3](https://arxiv.org/abs/2210.03629)，2023-03-10 修订；2026-11-13 阅读摘要与版本信息。
- 类型/证据：研究论文，支持作者提出的 interleaved reasoning/action 方法和论文任务设置。
- 支持的事实：作者把 reasoning trace 与 task-specific action 交错生成，用前者跟踪/更新计划并处理异常，用后者与外部来源或环境交互。论文在 HotpotQA、FEVER、ALFWorld 和 WebShop 等指定任务中做了实验，并报告其设置下相对相应基线的结果。
- 边界：这些实验不定义行业中所有 Agent 的必要条件，也不证明任意模型、工具、Harness 或生产任务都得到相同收益；正文引用时不把论文结论外推至未测设置。

### Toolformer

- 来源：[arXiv:2302.04761 v1](https://arxiv.org/abs/2302.04761)，2023-02-09；2026-11-13 阅读摘要与方法定位。
- 类型/证据：研究论文，支持作者提出的工具调用训练方法和其论文实验。
- 支持的事实：Toolformer 研究模型怎样学习决定调用哪个 API、何时调用、传入什么参数，以及怎样将返回结果用于后续 token 预测；作者描述了基于少量 API demonstrations 的自监督数据构造，并测试 calculator、QA、search、translation、calendar 等 API。
- 边界：这是特定训练方法和实验设置，不是通用 Agent runtime 架构，也不证明外部动作安全或工具结果真实。

### MRKL Systems

- 来源：[arXiv:2205.00445 v1](https://arxiv.org/abs/2205.00445)，2022-05-01；2026-11-13 阅读摘要与论文标题/版本信息。
- 类型/证据：研究论文与系统设计讨论，支持其模块化神经—符号架构描述。
- 支持的事实：论文讨论以多个神经模型与离散知识/推理模块配合的大模型系统架构，并介绍 Jurassic-X 的实现背景。
- 边界：该论文讨论的架构和实现不是现代 Agent 产品的统一标准，也不支持未披露产品的内部设计推断。

### OpenAI Agents SDK 文档

- 来源：[官方 Python SDK 文档](https://openai.github.io/openai-agents-python/)，动态文档；2026-11-13 阅读目录、Intro 和工具/guardrail/orchestration/trace 入口。
- 类型/证据：官方产品/API 文档，支持该 SDK 当时公开的概念和接口组织。
- 支持的事实：文档按 Agents、Tools、Guardrails、Running agents、Orchestration、Handoffs、Sessions、Context management、MCP、Tracing 等公开概念组织内容。
- 边界：动态 SDK 文档只说明公开接口，不表示模型内部结构或所有 Agent 系统都采用同样组件；后续引用具体 API 必须固定文档/SDK 版本。

### NIST AI RMF Generative AI Profile

- 来源：[NIST AI 600-1 页面](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence)，2024-07-26 发布，页面 2026-04-08 更新；2026-11-13 阅读简介和版本信息。
- 类型/证据：官方风险管理资料，支持组织级生成式 AI 风险管理背景。
- 支持的事实：该文件是面向生成式 AI 的跨行业 profile 和 AI RMF 1.0 配套资料，鼓励组织将可信性考虑纳入 AI 产品、服务和系统的设计、开发、使用与评估。
- 边界：它是自愿使用的风险管理资料，不是 Agent 专属运行规范、产品验收标准或某项业务的安全评审。

### OWASP Top 10 for LLM Applications

- 来源：[OWASP GenAI Security Project 页面](https://genai.owasp.org/llm-top-10/)，当前页面列出 2025 Top 10；2026-11-13 阅读版本目录和风险列表。
- 类型/证据：OWASP 安全风险分类项目，支持风险类别入口。
- 支持的事实：当前页面列出 2025 版风险类别，包括 Prompt Injection、Sensitive Information Disclosure、Supply Chain、Improper Output Handling、Excessive Agency 和 Unbounded Consumption 等。
- 边界：风险分类是审查入口，不是某系统已发生漏洞的证据或完整威胁模型；系统安全仍需按具体权限、数据和环境分析。

## 核验边界

- 本轮只核对论文摘要/版本和官方文档的公开范围，没有复现论文、SDK 示例或生产工作流。
- 章内报销账单的政策阈值和计算仅作为合成教学数据；不能用作真实公司的报销规则。
- 本笔记只保存来源支持的知识和限制；章节内容去向、取舍与精修结果写入 progress/。
