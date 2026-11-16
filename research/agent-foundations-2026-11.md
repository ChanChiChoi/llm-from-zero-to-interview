# Agent 基础概念与风险资料

首次记录：2026-11-13；本章精修时复核：2026-11-16。此文件保存联网核对的新知识点及来源证据，不记录书籍审阅决定或修改历史。

## 来源与阅读位置

### ReAct

- 来源：[arXiv:2210.03629 v3](https://arxiv.org/abs/2210.03629)，2023-03-10 修订；2026-11-13 阅读摘要与版本信息，2026-11-16 复核引用边界。
- 类型/证据：研究论文，支持作者提出的 interleaved reasoning/action 方法和论文任务设置。
- 支持的事实：作者把 reasoning trace 与 task-specific action 交错生成，用前者跟踪/更新计划并处理异常，用后者与外部来源或环境交互。论文在 HotpotQA、FEVER、ALFWorld 和 WebShop 等指定任务中做了实验，并报告其设置下相对相应基线的结果。
- 边界：这些实验不定义行业中所有 Agent 的必要条件，也不证明任意模型、工具、Harness 或生产任务都得到相同收益；正文引用时不把论文结论外推至未测设置。

### Toolformer

- 来源：[arXiv:2302.04761 v1](https://arxiv.org/abs/2302.04761)，2023-02-09；2026-11-13 阅读摘要与方法定位，2026-11-16 复核引用边界。
- 类型/证据：研究论文，支持作者提出的工具调用训练方法和其论文实验。
- 支持的事实：Toolformer 研究模型怎样学习决定调用哪个 API、何时调用、传入什么参数，以及怎样将返回结果用于后续 token 预测；作者描述了基于少量 API demonstrations 的自监督数据构造，并测试 calculator、QA、search、translation、calendar 等 API。
- 边界：这是特定训练方法和实验设置，不是通用 Agent runtime 架构，也不证明外部动作安全或工具结果真实。

### MRKL Systems

- 来源：[arXiv:2205.00445 v1](https://arxiv.org/abs/2205.00445)，2022-05-01；2026-11-13 阅读摘要与论文标题/版本信息，2026-11-16 复核引用边界。
- 类型/证据：研究论文与系统设计讨论，支持其模块化神经—符号架构描述。
- 支持的事实：论文讨论以多个神经模型与离散知识/推理模块配合的大模型系统架构，并介绍 Jurassic-X 的实现背景。
- 边界：该论文讨论的架构和实现不是现代 Agent 产品的统一标准，也不支持未披露产品的内部设计推断。

### Plan-and-Solve Prompting

- 来源：[arXiv:2305.04091 v3](https://arxiv.org/abs/2305.04091)，2023-05-26 修订，页面标注 ACL 2023；2026-11-16 阅读摘要与版本信息。
- 类型/证据：研究论文，支持作者提出的提示策略和论文评估范围。
- 支持的事实：Plan-and-Solve 由先把任务拆为子任务、再按计划执行组成；PS+ 增加更详细的指令以改善推理步骤。摘要报告在十个数据集、三类推理问题上进行了评估。
- 边界：论文讨论的是多步推理提示方法，不定义 Agent 工具调用 API 或通用环境交互协议；这表示方法描述范围有限，不等于证明该方法不能与工具组合。论文结果绑定其模型、数据和任务设置。

### LLM+P

- 来源：[arXiv:2304.11477 v3](https://arxiv.org/abs/2304.11477)，2023-09-27 修订；2026-11-16 阅读摘要与版本信息。
- 类型/证据：研究论文，支持作者提出的 LLM 与经典规划器组合流程及其论文实验范围。
- 支持的事实：LLM+P 将自然语言规划问题转成 PDDL 描述，调用经典规划器求解，再把规划结果译回自然语言；摘要报告其在作者设定的多个规划基准问题上的结果。
- 边界：规划正确性依赖问题和领域描述的转换质量、PDDL 表达能力及规划器约束；论文本身不证明开放环境中工具状态、权限或副作用可以被同一流程自动保证。论文实验结果不应外推到任意业务任务。

### Tree of Thoughts

- 来源：[arXiv:2305.10601 v2](https://arxiv.org/abs/2305.10601)，2023-12-03 修订，页面标注 NeurIPS 2023 camera-ready；2026-11-16 阅读摘要与版本信息。
- 类型/证据：研究论文，支持作者提出的推理时搜索框架和指定实验任务。
- 支持的事实：Tree of Thoughts 将连贯文本推理片段作为中间节点，允许模型生成多个候选路径、评估候选、前瞻或回溯；论文报告 Game of 24、创意写作和 Mini Crosswords 三项任务的实验。
- 边界：这是指定任务上的推理时搜索方案；“thought”树不自动等同外部环境的任务 DAG、工具执行器或真实状态观察。论文结果不证明所有任务都适合扩大搜索。

### Reflexion

- 来源：[arXiv:2303.11366 v4](https://arxiv.org/abs/2303.11366)，2023-10-10 修订；2026-11-16 阅读摘要与版本信息。
- 类型/证据：研究论文，支持作者提出的语言反馈框架和论文所报告的任务范围。
- 支持的事实：Reflexion 不通过更新模型权重实现反馈，而是将任务反馈转成语言反思，存入 episodic memory，供后续尝试决策时使用；论文报告其在序列决策、代码和语言推理任务上的实验。
- 边界：这是论文框架及其实验，不代表所有 Agent 都需要跨尝试记忆，也不证明加入反思文本会在任意任务或预算下提升表现。

### Pre-Act

- 来源：[arXiv:2505.09970 v2](https://arxiv.org/abs/2505.09970)，2025-05-19 修订；2026-11-16 阅读摘要以及 HTML 版第 3.1、4.1、4.3 节和相应用例表。
- 类型/证据：研究预印本；作者报告的方法、数据集与评估设计。
- 支持的事实：Pre-Act 为请求生成多步计划和逐步推理；每步执行后把先前动作及工具输出纳入上下文，细化后续步骤。论文用 turn-level 和 end-to-end 两级评估；数据包括 Glaive Function-Calling v2、作者的专有数据集和 Almita。作者从 Almita 的 18 个用例中人工筛出 5 个用于端到端评估，排除相似用例及缺少工具或流程信息的用例；端到端环境使用 GPT-4 模拟用户，并以 GPT-4 评判目标完成与进度。
- 边界：端到端样本仅五个且经过人工筛选，模拟用户、LLM judge、专有训练数据和指定模型限制了独立复现与外推范围。论文报告的改进不能直接推广为 Pre-Act 对 ReAct 的普遍优势。

### OpenAI Agents SDK 文档

- 来源：[官方 Python SDK 文档](https://openai.github.io/openai-agents-python/)，动态文档；2026-11-13 阅读目录、Intro 和工具/guardrail/orchestration/trace 入口，2026-11-16 复核目录。
- 类型/证据：官方产品/API 文档，支持该 SDK 当时公开的概念和接口组织。
- 支持的事实：文档按 Agents、Tools、Guardrails、Running agents、Orchestration、Handoffs、Sessions、Context management、MCP、Tracing 等公开概念组织内容。
- 边界：动态 SDK 文档只说明公开接口，不表示模型内部结构或所有 Agent 系统都采用同样组件；后续引用具体 API 必须固定文档/SDK 版本。

#### Tracing 文档补充

- 来源：[Tracing 文档](https://openai.github.io/openai-agents-python/tracing/)，动态官方文档；2026-11-16 阅读 Traces and spans、Default tracing、Sensitive data、Custom tracing processors 段落。
- 类型/证据：官方 SDK 文档，支持当日公开的默认 trace/span 组成和敏感数据选项。
- 支持的事实：文档说明 Runner 工作流会建立 trace，并可为调用、模型轮次、Agent、generation、function、guardrail 和 handoff 生成 span。generation/function span 可能保存模型或函数输入/输出；`trace_include_sensitive_data` 当时默认为 `True`，可通过运行配置关闭；默认 tracing 本身也可禁用。
- 边界：文档是动态的、仅对应 OpenAI Agents SDK；span 内容受配置和运行路径影响。不能据此推断其他 tracing 系统的默认值，也不能把 SDK trace 当成完整业务状态或安全审计记录。

### NIST AI RMF Generative AI Profile

- 来源：[NIST AI 600-1 页面](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence)，2024-07-26 发布，页面 2026-04-08 更新；2026-11-13 阅读简介和版本信息，2026-11-16 复核页面状态。
- 类型/证据：官方风险管理资料，支持组织级生成式 AI 风险管理背景。
- 支持的事实：该文件是面向生成式 AI 的跨行业 profile 和 AI RMF 1.0 配套资料，鼓励组织将可信性考虑纳入 AI 产品、服务和系统的设计、开发、使用与评估。
- 边界：它是自愿使用的风险管理资料，不是 Agent 专属运行规范、产品验收标准或某项业务的安全评审。

### OWASP Top 10 for LLM Applications

- 来源：[OWASP GenAI Security Project 页面](https://genai.owasp.org/llm-top-10/)，当前页面列出 2025 Top 10；2026-11-13 阅读版本目录和风险列表，2026-11-16 复核页面状态。
- 类型/证据：OWASP 安全风险分类项目，支持风险类别入口。
- 支持的事实：当前页面列出 2025 版风险类别，包括 Prompt Injection、Sensitive Information Disclosure、Supply Chain、Improper Output Handling、Excessive Agency 和 Unbounded Consumption 等。
- 边界：风险分类是审查入口，不是某系统已发生漏洞的证据或完整威胁模型；系统安全仍需按具体权限、数据和环境分析。

### OpenAI Function Calling 与 Structured Outputs

- 来源：[OpenAI API Function Calling 指南](https://developers.openai.com/api/docs/guides/function-calling) 与 [Structured Outputs 指南](https://developers.openai.com/api/docs/guides/structured-outputs)，动态官方文档，2026-11-16 阅读。
- 类型/证据：当前公开 API 文档，支持该日期可见的接口参数和 schema 约束；不是所有厂商的通用协议定义。
- 支持的事实：Function Calling 指南将 `tool_choice` 的公开选择方式列为自动选择、要求调用、指定函数和限定可用工具子集；设 `strict: true` 时，参数对象需要 `additionalProperties: false`，所有 properties 字段都要列为 required，可选字段可用允许 `null` 的类型表达。若未显式设置 strict，Responses API 会尝试规范化兼容 schema，无法兼容时可能回退到 best-effort；Chat Completions 默认行为不同。
- 边界：这些约束绑定 OpenAI 的具体 API surface 和当前动态文档；不能外推到其他厂商、所有 JSON Schema 功能或未来版本。结构约束只约束形状，不证明工具选择、参数语义、权限、确认或服务执行结果正确。

## 核验边界

- 本轮只核对论文摘要/版本和官方文档的公开范围，没有复现论文、SDK 示例或生产工作流。
- 章内报销账单的政策阈值和计算仅作为合成教学数据；不能用作真实公司的报销规则。
- 本笔记只保存来源支持的知识和限制；章节内容去向、取舍与精修结果写入 progress/。
