# 第十二章：Agent 面试题

本章是第十七册的综合面试题。前面章节已经讲过 Agent 总览、tool use、function calling、ReAct、planning、memory、Agentic RAG、Code Agent、Browser Agent、Multi-Agent、评估和安全。本章把这些内容整理成面试中可以直接使用的回答。

Agent 面试回答要避免两个极端：一是只说“Agent 就是能调用工具的大模型”，过于浅；二是堆砌 ReAct、planner、memory、multi-agent 等术语，但不讲工程边界。高质量回答要能说明：Agent 解决什么问题，系统怎么设计，如何评估，如何控制风险。

## 0. 本讲资料边界与第二轮精修口径

本讲第二轮精修前，已按 `WRITING_PLAN.md` 联网核对 OpenAI Agents SDK 的 tools / guardrails / tracing 公开文档、OpenAI Evals、OpenAI Model Spec 的指令层级口径，以及 SWE-bench、WebArena、OSWorld、AgentBench、GAIA 和 tau-bench 等公开 Agent 评估资料。

本章定位是第十七册的面试总复盘，不重新展开第 1-11 章的全部细节。正文重点是把 Agent 架构、工具调用、planning、memory、RAG、Code Agent、Browser / Computer Use、Multi-Agent、评估和安全，组织成可在面试中稳定表达、可自查、可迭代的回答框架。

本章只做防御性、教学性和面试表达层面的总结，不提供可复用注入提示、绕过权限、规避审计、高风险自动操作或破坏系统的方法。

## 12.0 Agent 面试总框架

Agent 面试的标准答案不要从“模型很聪明”开始，而要从系统闭环开始：

```text
我会把 Agent 看成目标驱动的多步任务执行系统。它有 goal、state、action、tool、observation、memory、controller、evaluator 和 logger。模型可以提出下一步动作，但动作是否执行要经过 schema 校验、权限门禁、预算控制和高风险确认。评估时不能只看最终答案，要看 trace 中工具选择、参数、observation 使用、状态更新、错误恢复、成本和安全。
```

这段话可以拆成 8 个必须覆盖的维度：

1. 任务目标：Agent 解决需要多步行动、外部工具和环境反馈的问题。
2. 系统结构：goal、state、tool registry、executor、observation、controller、memory 和 logger。
3. 工具调用：schema、参数校验、权限、执行、错误恢复和 trace。
4. 规划与状态：任务分解、动态重规划、状态更新、停止条件和预算。
5. 特殊场景：Agentic RAG、Code Agent、Browser / Computer Use、Multi-Agent。
6. 评估指标：任务成功、部分成功、工具、参数、trace、summary、成本和延迟。
7. 安全控制：最小权限、沙箱、不可信内容、人工确认、数据流和审计。
8. 项目表达：baseline、指标、bad case、个人贡献、取舍和下一步改进。

## 12.0.1 关键公式与 Agent 面试自评指标

面试准备可以抽象成一组回答样本。第 `i` 个回答记为：

```math
a_i=(q_i,C_i,F_i,D_i,M_i,S_i,E_i,P_i,T_i,R_i)
```

变量含义：

1. `q_i` 是问题。
2. `C_i` 是回答覆盖的概念集合。
3. `F_i` 是回答覆盖的公式或指标集合。
4. `D_i` 是回答能举出的 demo 或代码集合。
5. `M_i` 是 trace、成本、延迟、成功率等评估指标集合。
6. `S_i` 是安全边界集合。
7. `E_i` 是评估方法集合。
8. `P_i` 是项目证据集合。
9. `T_i` 是 trade-off 集合。
10. `R_i` 是红旗问题集合，例如“只背术语”“没有指标”“没有安全边界”“夸大项目”。

概念覆盖率：

```math
R_{\mathrm{concept}}=\frac{|C_i\cap C_i^\star|}{|C_i^\star|}
```

公式覆盖率：

```math
R_{\mathrm{formula}}=\frac{|F_i\cap F_i^\star|}{|F_i^\star|}
```

demo 覆盖率：

```math
R_{\mathrm{demo}}=\frac{|D_i\cap D_i^\star|}{|D_i^\star|}
```

trace 指标覆盖率：

```math
R_{\mathrm{metric}}=\frac{|M_i\cap M_i^\star|}{|M_i^\star|}
```

项目证据得分：

```math
S_{\mathrm{project}}=\frac{|P_i\cap P_i^\star|}{|P_i^\star|}
```

单题面试得分可以写成加权和：

```math
S_i=w_cR_{\mathrm{concept}}+w_fR_{\mathrm{formula}}+w_dR_{\mathrm{demo}}+w_mR_{\mathrm{metric}}+w_sR_{\mathrm{safety}}+w_eR_{\mathrm{eval}}+w_pS_{\mathrm{project}}+w_tR_{\mathrm{trade}}
```

其中各权重相加为 1。准备度门禁可以写成：

```math
G_{\mathrm{agentint}}=\mathbf{1}[\bar S\ge \tau_s \land \min_i S_i\ge \tau_m \land N_{\mathrm{red}}=0 \land R_{\mathrm{safety}}\ge \tau_{\mathrm{safe}}]
```

直觉：Agent 面试不是背 25 道题，而是证明你能用结构、公式、demo、评估、安全和项目证据支撑回答。

## 12.1 什么是 Agent

回答要点：

```text
Agent 是由 LLM 驱动、围绕目标进行多步决策，并通过工具或环境反馈执行任务的系统。它通常包含 goal、state、tool、observation、planner、memory、controller 和 logger。和普通聊天模型相比，Agent 不只是生成答案，还会选择动作、调用工具、观察结果、更新状态并决定下一步。
```

补充说明：

```text
Agent 不是单个模型名，而是一种系统架构。判断一个系统是不是 Agent，关键看它是否有目标驱动、多步执行、工具调用、状态管理和反馈闭环。
```

## 12.2 Agent 和普通 Chatbot 有什么区别

回答要点：

```text
普通 chatbot 主要根据输入生成回复，流程通常是 user input 到 model response。Agent 是任务执行系统，会维护状态、拆解任务、调用工具、读取 observation，并在多轮循环中完成目标。Agent 更适合需要外部信息、工具执行和动态决策的任务，但也更难控制成本、安全和稳定性。
```

## 12.3 Agent 和 Workflow 如何取舍

回答要点：

```text
Workflow 是预定义流程，稳定、可控、容易测试，适合流程明确的任务。Agent 更灵活，适合信息不完整、需要动态决策和工具反馈的任务。生产系统里常用混合架构：外层 workflow 控制关键流程，局部复杂步骤交给 Agent 处理，这样既保留可控性，又利用模型的灵活性。
```

## 12.4 如何设计工具调用系统

回答要点：

```text
我会先建立 tool registry，为每个工具定义名称、描述、参数 schema、返回 schema、权限等级、超时和重试策略。模型负责选择工具和生成参数，系统在执行前做参数校验和权限检查。工具返回 observation 后，Agent 决定继续调用、重试、请求用户补充还是停止。系统还需要日志、错误恢复、速率限制和工具输出注入防护。
```

## 12.5 Function Calling 解决什么问题

回答要点：

```text
Function calling 解决的是工具调用的结构化和可控性问题。系统把工具名称、描述和参数 schema 提供给模型，模型生成符合 schema 的调用，而不是自由写自然语言命令。这样更容易解析、校验、执行、记录和评估，也能减少参数错误和工具误用。但它不能替代权限控制和安全策略。
```

## 12.6 ReAct 是什么

回答要点：

```text
ReAct 是 reasoning 和 action 交替的 Agent 框架。模型每一步根据当前状态思考下一步，调用工具或执行动作，然后根据 observation 更新判断。它适合搜索、代码调试、浏览器操作和复杂工具调用任务。工程上需要控制最大步数、工具权限、错误恢复和 trace 日志。
```

## 12.7 Plan-Act-Observe 如何落地

回答要点：

```text
我会让 Agent 先根据目标生成一个可修改计划，然后每次执行一个或少量 action。工具返回 observation 后，系统更新 state，并让模型判断计划是否需要调整。工程上需要 action schema、tool executor、observation parser、step limit、budget manager、error recovery 和 trace logger。高风险 action 需要权限检查或用户确认。
```

## 12.8 Agent 如何做任务分解

回答要点：

```text
任务分解首先要明确目标、约束和完成标准，然后拆成可执行、可验证的子目标。每个子目标要有输入输出和验收标准，再根据依赖关系安排顺序。执行过程中根据 observation 更新计划。如果子任务失败，系统要能重试、换工具、请求用户补充、降级或人工接管。
```

## 12.9 一次性规划和动态规划如何取舍

回答要点：

```text
一次性规划适合流程稳定、目标明确、需要用户确认的任务，优点是结构清晰、成本可控。动态规划适合信息不完整、环境反馈强、执行结果决定下一步的任务，例如调试和浏览器操作。实际系统通常先生成粗计划，再在每一步根据 observation 修正，既保留全局方向，又避免僵化执行错误计划。
```

## 12.10 Agent Memory 如何设计

回答要点：

```text
我会把 memory 分成短期和长期。短期 memory 维护当前任务状态，例如目标、计划、工具结果和失败尝试；长期 memory 保存稳定用户偏好、项目事实和历史经验。系统需要写入策略、检索策略、更新和删除机制，并记录来源、时间和置信度。安全上要做用户隔离、敏感信息过滤和可查看可删除，避免 memory 污染和隐私泄露。
```

## 12.11 Memory 和 RAG 的区别

回答要点：

```text
RAG 通常检索外部知识库，解决知识获取问题；memory 检索用户、任务和历史行为，解决上下文连续性和个性化问题。二者技术上都可以用向量检索，但 memory 更强调写入、更新、遗忘、权限隔离和隐私控制。Agent 系统里 memory 可以复用 RAG 基础设施，但需要更严格的治理。
```

## 12.12 Agentic RAG 和普通 RAG 的区别

回答要点：

```text
普通 RAG 通常是用户问题检索一次，然后基于 top-k 文档生成答案。Agentic RAG 把检索放进 Agent 循环，模型会主动判断是否需要检索、如何改写查询、是否多轮检索、如何验证证据以及什么时候停止。它适合复杂、多跳、需要引用和证据验证的问题，但成本更高，也需要更强的循环控制和评估。
```

## 12.13 如何设计可靠的 Agentic RAG

回答要点：

```text
我会让 Agent 先理解问题并拆成子问题，再选择检索工具和生成查询。每轮检索后，系统阅读证据、判断是否支持结论、识别缺口和冲突，并决定继续检索还是停止。工程上要做查询重写、reranking、引用校验、权限控制、成本限制和 trace 日志。评估时看答案正确率、引用准确率、证据支持率和检索成本。
```

## 12.14 Code Agent 和普通代码生成有什么区别

回答要点：

```text
普通代码生成通常根据 prompt 生成一段代码，缺少仓库上下文和执行反馈。Code Agent 会在真实仓库中完成任务，包括理解项目结构、搜索相关文件、做最小修改、运行测试、读取错误并迭代修复。它更接近工程执行系统，所以需要工具权限、测试验证、日志审计和安全边界。
```

## 12.15 如何设计可靠的 Code Agent

回答要点：

```text
我会让 Code Agent 先理解任务和仓库结构，再通过搜索定位相关文件，修改前读取上下文，遵循最小修改原则。修改后运行相关测试、lint 或 build，根据失败反馈继续 debug。系统层面要有文件编辑工具、命令执行沙箱、权限控制、超时、trace 日志和禁止危险命令的策略。评估时看任务成功率、测试通过率、无关改动比例和安全违规率。
```

## 12.16 Browser Agent 和 API Tool Use 如何取舍

回答要点：

```text
如果有稳定、安全、权限清晰的 API，我会优先使用 API tool，因为它结构化、可验证、鲁棒性更好。Browser agent 适合没有 API、需要操作现有网页或跨系统流程的场景。但浏览器 UI 更脆弱，容易受页面变化、弹窗、注入和登录状态影响，因此需要更强的观察、错误恢复和高风险操作确认。
```

## 12.17 如何保证 Computer Use Agent 安全

回答要点：

```text
我会从权限、环境和操作三层控制。权限上使用最小权限和用户隔离；环境上放在沙箱或受控浏览器中，限制文件系统、剪贴板和网络访问；操作上对支付、删除、发送、提交等高风险动作要求用户确认，并记录完整 trace。还要防 prompt injection，把网页或屏幕内容视为不可信数据。
```

## 12.18 什么时候需要 Multi-Agent

回答要点：

```text
当任务复杂、需要多个专业角色、可以并行处理、需要独立审查或不同视角时，可以考虑 Multi-Agent。例如研究报告、代码开发加 review、复杂规划和多来源验证。但简单任务不适合 Multi-Agent，因为通信和协调成本会超过收益。是否使用多 Agent 要看它是否真正提升任务成功率、验证质量或并行效率。
```

## 12.19 如何设计 Multi-Agent 系统

回答要点：

```text
我会先明确任务是否需要多 Agent，然后设计 coordinator、角色分工、工具权限和通信协议。每个 Agent 只拿到完成任务所需的上下文和工具。共享状态通过结构化 blackboard 管理，冲突通过证据、工具验证、judge 或人工确认解决。系统还要记录 trace，评估任务成功率、成本、通信轮数、冲突解决和安全违规率。
```

## 12.20 如何评估 Agent 系统

回答要点：

```text
我会从任务成功率、工具调用质量、步骤效率、错误恢复、安全和成本几个维度评估。首先定义每类任务的验收标准，再记录完整 trace，检查工具选择、参数、observation 使用和最终结果是否一致。可验证任务用自动评估，例如测试通过或页面状态；开放任务结合人工 rubric 和 LLM judge。还要报告 token、延迟、工具调用次数和安全违规率。
```

## 12.21 为什么 Agent 不能只看最终答案

回答要点：

```text
因为 Agent 的价值和风险都在执行过程中。最终答案看起来正确，不代表它真的运行了测试、没有越权调用工具、没有误删文件、没有忽略错误反馈。评估 Agent 必须看 trace，包括每一步 action、工具参数、observation、失败恢复和最终总结是否忠实于实际执行。
```

## 12.22 Agent 安全如何设计

回答要点：

```text
我会从权限、执行环境、数据流和审计四层设计。权限上遵循最小权限，区分只读和写入工具，高风险动作需要人工确认；执行环境上使用沙箱限制文件、网络、CPU、内存和命令；数据流上防止敏感信息泄露和工具输出注入；审计上记录完整 trace、工具参数、权限检查和用户确认。关键是安全由系统层强制，而不是只靠模型提示词。
```

## 12.23 如何防 Prompt Injection

回答要点：

```text
首先要把用户输入、网页、文档和工具返回都视为不可信数据，不能让它们覆盖系统指令。系统应隔离指令和数据，对工具输出做引用隔离，高风险动作必须经过权限检查和用户确认。对于 RAG 和 browser agent，还要记录内容来源，避免外部文档中的恶意指令影响 Agent 行为。
```

## 12.24 设计一个完整 Agent 系统

回答框架：

```text
我会把系统拆成 router、planner、state manager、tool registry、executor、observation handler、memory、controller、evaluator 和 logger。Router 判断任务类型；planner 拆解任务；state manager 维护状态；tool registry 管理工具 schema 和权限；executor 执行动作；observation handler 解析反馈；memory 保存短期和长期信息；controller 管理预算、停止和错误恢复；evaluator 判断任务是否完成；logger 记录 trace 和审计。
```

补充：

```text
生产系统不会让模型自由调用所有工具，而是用 workflow、权限、预算、人工确认和安全策略控制 Agent 行为。
```

## 12.25 高频追问

追问：Agent 是不是越自主越好？

```text
不是。自主性越高，风险和不可控性越高。生产系统需要在自主性和可控性之间平衡，简单任务用 workflow，高风险动作需要确认，关键步骤要可验证和可审计。
```

追问：工具越多越好吗？

```text
不是。工具越多，选择难度、权限风险和维护成本越高。工具应该有清晰 schema、边界和权限，Agent 只拿到当前任务需要的工具。
```

追问：Multi-Agent 一定比 Single Agent 强吗？

```text
不一定。Multi-Agent 可以带来并行、分工和互检，但也会增加通信成本和冲突。只有当任务复杂、需要多角色或验证收益明显时才值得使用。
```

追问：Agent 失败时怎么办？

```text
要根据失败类型处理。参数错误可以修正重试，权限不足需要请求授权或降级，工具不可用可以换工具，目标不明确要问用户，达到预算上限要停止并报告已完成和未完成内容。
```

## 12.26 面试回答的常见红旗

红旗 1：只说“Agent 是会调用工具的 LLM”。

修正：补上 goal、state、action、observation、controller、trace、evaluation 和 safety。

红旗 2：把 function calling 当成安全机制。

修正：function calling 只是结构化接口，仍然需要 schema 校验、业务校验、权限、确认、沙箱和审计。

红旗 3：只讲 ReAct 或 planning，不讲停止条件。

修正：Agent loop 必须有 step limit、budget、success criteria、blocked action recovery 和 stop correctness。

红旗 4：项目表达只有“做了一个 Agent”。

修正：改成“做了什么任务、baseline 是什么、提升了什么指标、失败样本是什么、我负责了什么、为什么这样取舍”。

红旗 5：评估只报最终成功率。

修正：补充 trace completeness、tool accuracy、argument validity、observation use、state update、summary faithfulness、claim support、cost、latency 和 safety gate。

红旗 6：安全只说“加 prompt 防护”。

修正：强调系统层最小权限、工具权限矩阵、不可信内容边界、数据流门禁、沙箱、人工确认、dry-run 和 audit log。

## 12.27 最小可运行 Agent interview readiness demo

下面这个 demo 不调用外部模型，而是模拟 5 道 Agent 面试题的回答记录，检查每题是否覆盖概念、公式 / 指标、demo、trace 指标、安全边界、评估口径、项目证据和 trade-off。

它演示的问题是：面试复盘不能只写“这题不会”，而要把每个弱点绑定到缺失概念、缺失公式、缺失 demo 或缺失项目证据。

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class AnswerRecord:
    question_id: str
    title: str
    concepts: set
    formulas: set
    demos: set
    trace_metrics: set
    safety: set
    evaluation: set
    project_evidence: set
    tradeoffs: set
    red_flags: set


expected = {
    "q1": {
        "concepts": {"goal", "state", "action", "observation", "controller", "trace"},
        "formulas": {"agent_gate"},
        "demos": {"agent_trace_audit"},
        "trace_metrics": {"task_success_rate", "observation_use_rate", "state_update_coverage"},
        "safety": {"permission_gate"},
        "evaluation": {"trace_eval"},
        "project_evidence": set(),
        "tradeoffs": {"agent_vs_workflow"},
    },
    "q2": {
        "concepts": {"tool_registry", "tool_executor", "schema", "function_calling"},
        "formulas": {"tool_selection_accuracy", "argument_validity"},
        "demos": {"tool_call_audit"},
        "trace_metrics": {"schema_valid_rate", "execution_success_rate", "error_recovery_rate"},
        "safety": {"least_privilege", "human_confirmation"},
        "evaluation": {"tool_eval"},
        "project_evidence": set(),
        "tradeoffs": {"schema_vs_business_rule"},
    },
    "q3": {
        "concepts": {"agentic_rag", "retrieval_controller", "evidence_state", "citation"},
        "formulas": {"context_precision", "citation_accuracy"},
        "demos": {"agentic_rag_audit"},
        "trace_metrics": {"new_evidence_gain", "claim_support_rate"},
        "safety": {"untrusted_content", "permission_filter"},
        "evaluation": {"rag_eval"},
        "project_evidence": {"bad_cases", "metrics"},
        "tradeoffs": {"quality_vs_cost"},
    },
    "q4": {
        "concepts": {"code_agent", "repo_understanding", "minimal_patch", "test_feedback"},
        "formulas": {"patch_localization", "validation_coverage"},
        "demos": {"code_agent_audit"},
        "trace_metrics": {"test_pass_rate", "unrelated_change_rate", "command_success_rate"},
        "safety": {"code_sandbox", "user_change_protection"},
        "evaluation": {"programmatic_eval"},
        "project_evidence": {"tests", "owned_work", "bad_cases"},
        "tradeoffs": {"minimal_patch_vs_refactor"},
    },
    "q5": {
        "concepts": {"agent_evaluation", "agent_safety", "multi_agent", "ui_agent"},
        "formulas": {"summary_faithfulness", "unauthorized_action_rate", "agent_safety_gate"},
        "demos": {"agent_eval_audit", "agent_safety_audit"},
        "trace_metrics": {"trace_completeness", "claim_support_rate", "p95_latency", "cost_per_success"},
        "safety": {"least_privilege", "sandbox", "data_flow_guard", "audit_log"},
        "evaluation": {"regression_suite", "human_rubric"},
        "project_evidence": {"baseline", "metrics", "bad_cases"},
        "tradeoffs": {"autonomy_vs_control", "single_vs_multi"},
    },
}


answers = [
    AnswerRecord(
        "q1",
        "Agent 和普通应用区别",
        {"goal", "state", "action", "observation", "controller", "trace"},
        {"agent_gate"},
        {"agent_trace_audit"},
        {"task_success_rate", "observation_use_rate", "state_update_coverage"},
        {"permission_gate"},
        {"trace_eval"},
        set(),
        {"agent_vs_workflow"},
        set(),
    ),
    AnswerRecord(
        "q2",
        "工具调用系统设计",
        {"tool_registry", "tool_executor", "schema", "function_calling"},
        {"tool_selection_accuracy"},
        {"tool_call_audit"},
        {"schema_valid_rate", "execution_success_rate"},
        {"least_privilege"},
        {"tool_eval"},
        set(),
        {"schema_vs_business_rule"},
        {"missing_confirmation"},
    ),
    AnswerRecord(
        "q3",
        "Agentic RAG",
        {"agentic_rag", "retrieval_controller", "evidence_state", "citation"},
        {"context_precision", "citation_accuracy"},
        {"agentic_rag_audit"},
        {"new_evidence_gain", "claim_support_rate"},
        {"untrusted_content", "permission_filter"},
        {"rag_eval"},
        {"bad_cases", "metrics"},
        {"quality_vs_cost"},
        set(),
    ),
    AnswerRecord(
        "q4",
        "Code Agent 项目深挖",
        {"code_agent", "repo_understanding", "minimal_patch", "test_feedback"},
        {"patch_localization", "validation_coverage"},
        {"code_agent_audit"},
        {"test_pass_rate", "unrelated_change_rate", "command_success_rate"},
        {"code_sandbox", "user_change_protection"},
        {"programmatic_eval"},
        {"tests", "owned_work"},
        {"minimal_patch_vs_refactor"},
        {"weak_project_evidence"},
    ),
    AnswerRecord(
        "q5",
        "Agent 评估与安全",
        {"agent_evaluation", "agent_safety", "multi_agent"},
        {"summary_faithfulness", "unauthorized_action_rate"},
        {"agent_eval_audit"},
        {"trace_completeness", "claim_support_rate", "p95_latency"},
        {"least_privilege", "sandbox", "audit_log"},
        {"regression_suite"},
        {"baseline", "metrics"},
        {"autonomy_vs_control"},
        set(),
    ),
]


def coverage(got, want):
    if not want:
        return 1.0
    return round(len(got & want) / len(want), 3)


def union(records, field):
    values = set()
    for record in records:
        values |= getattr(record, field)
    return values


def expected_union(field):
    values = set()
    for spec in expected.values():
        values |= spec[field]
    return values


weights = {
    "concepts": 0.18,
    "formulas": 0.14,
    "demos": 0.12,
    "trace_metrics": 0.16,
    "safety": 0.16,
    "evaluation": 0.10,
    "project_evidence": 0.08,
    "tradeoffs": 0.06,
}

question_scores = {}
weak_questions = []
revision_plan = {}

for answer in answers:
    spec = expected[answer.question_id]
    parts = {field: coverage(getattr(answer, field), spec[field]) for field in weights}
    score = round(sum(weights[field] * parts[field] for field in weights), 3)
    question_scores[answer.question_id] = score
    missing = {
        field: sorted(spec[field] - getattr(answer, field))
        for field in weights
        if spec[field] - getattr(answer, field)
    }
    if score < 0.85 or answer.red_flags:
        weak_questions.append(answer.question_id)
        revision_plan[answer.question_id] = {
            "missing": missing,
            "red_flags": sorted(answer.red_flags),
            "next_action": "补一个公式、一个 demo、一个 bad case 和一个 3 分钟回答模板",
        }

overall = {
    "concept_coverage": coverage(union(answers, "concepts"), expected_union("concepts")),
    "formula_coverage": coverage(union(answers, "formulas"), expected_union("formulas")),
    "demo_coverage": coverage(union(answers, "demos"), expected_union("demos")),
    "trace_metric_coverage": coverage(union(answers, "trace_metrics"), expected_union("trace_metrics")),
    "safety_coverage": coverage(union(answers, "safety"), expected_union("safety")),
    "evaluation_coverage": coverage(union(answers, "evaluation"), expected_union("evaluation")),
    "project_evidence_score": coverage(union(answers, "project_evidence"), expected_union("project_evidence")),
    "tradeoff_score": coverage(union(answers, "tradeoffs"), expected_union("tradeoffs")),
}

red_flags = sorted({flag for answer in answers for flag in answer.red_flags})
average_score = round(sum(question_scores.values()) / len(question_scores), 3)
readiness_gate = (
    average_score >= 0.85
    and min(question_scores.values()) >= 0.75
    and overall["safety_coverage"] >= 0.90
    and not red_flags
)

print(f"question_scores={question_scores}")
print(f"overall={overall}")
print(f"red_flags={red_flags}")
print(f"weak_questions={weak_questions}")
print(f"average_score={average_score}")
print(f"readiness_gate={readiness_gate}")
print(f"revision_plan={revision_plan}")
```

输出示例：

```text
question_scores={'q1': 1.0, 'q2': 0.797, 'q3': 1.0, 'q4': 0.973, 'q5': 0.662}
overall={'concept_coverage': 0.955, 'formula_coverage': 0.8, 'demo_coverage': 0.833, 'trace_metric_coverage': 0.857, 'safety_coverage': 0.8, 'evaluation_coverage': 0.833, 'project_evidence_score': 1.0, 'tradeoff_score': 0.833}
red_flags=['missing_confirmation', 'weak_project_evidence']
weak_questions=['q2', 'q4', 'q5']
average_score=0.886
readiness_gate=False
revision_plan={'q2': {'missing': {'formulas': ['argument_validity'], 'trace_metrics': ['error_recovery_rate'], 'safety': ['human_confirmation']}, 'red_flags': ['missing_confirmation'], 'next_action': '补一个公式、一个 demo、一个 bad case 和一个 3 分钟回答模板'}, 'q4': {'missing': {'project_evidence': ['bad_cases']}, 'red_flags': ['weak_project_evidence'], 'next_action': '补一个公式、一个 demo、一个 bad case 和一个 3 分钟回答模板'}, 'q5': {'missing': {'concepts': ['ui_agent'], 'formulas': ['agent_safety_gate'], 'demos': ['agent_safety_audit'], 'trace_metrics': ['cost_per_success'], 'safety': ['data_flow_guard'], 'evaluation': ['human_rubric'], 'project_evidence': ['bad_cases'], 'tradeoffs': ['single_vs_multi']}, 'red_flags': [], 'next_action': '补一个公式、一个 demo、一个 bad case 和一个 3 分钟回答模板'}}
```

这个 demo 的 `readiness_gate=False` 不是程序错误，而是在提醒面试准备还存在三个阻断点：工具调用题缺少高风险确认口径，Code Agent 项目证据缺少 bad case，评估与安全题缺少 safety gate、成本指标、数据流门禁、人评口径和 single-vs-multi 取舍。

## 12.28 第十七册总复盘清单

完成第十七册后，建议用下面的清单做最后一轮自查：

1. 能用 3 分钟讲清楚 Agent 和普通 LLM 应用、RAG、workflow 的区别。
2. 能画出 Agent loop，并解释 goal、state、action、observation、controller 和 trace。
3. 能设计 tool registry、tool schema、tool executor、权限检查和错误恢复。
4. 能解释 ReAct、Plan-Act-Observe、任务分解、动态重规划和停止条件。
5. 能区分 short-term memory、long-term memory、RAG 和 memory pollution。
6. 能讲 Agentic RAG 的 retrieval controller、evidence state、query drift、citation 和评估。
7. 能讲 Code Agent 如何理解仓库、做最小 patch、运行测试和保护用户改动。
8. 能讲 Browser / Computer Use Agent 的 observation、GUI action、误点击、表单、状态重观测和高风险确认。
9. 能讲 Multi-Agent 的角色、coordinator、blackboard、通信协议、冲突、单 Agent baseline 和成本。
10. 能列出 Agent 评估指标：task success、partial score、tool accuracy、argument validity、trace completeness、summary faithfulness、claim support、recovery、cost、P95 latency 和 regression suite。
11. 能列出 Agent 安全指标：least privilege、permission matrix、untrusted content boundary、data flow guard、sandbox、human confirmation、dry-run、audit log 和 safety gate。
12. 能把自己的 Agent 项目讲成 baseline、指标、trace、bad case、个人贡献、trade-off 和下一步改进，而不是只讲“做了一个智能体”。

## 12.29 本章小结

Agent 面试的核心是把“大模型 + 工具”讲成一个可落地、可评估、可控制的系统。一个完整答案应该覆盖目标、状态、工具、观察、规划、memory、执行、评估和安全。

到这里，第十七册《Agent 与工具调用专题》的第二轮阶段性精修完成。本册的主线是：Agent 如何从回答问题走向执行任务，如何通过工具调用、ReAct、规划、memory、RAG、代码和浏览器操作完成复杂目标，以及如何用 trace、评估指标和系统层安全控制这些能力。
