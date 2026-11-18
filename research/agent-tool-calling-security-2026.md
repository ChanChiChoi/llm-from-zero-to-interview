# Agent 工具调用、安全确认与评测：来源证据

创建于 2026-10-30；最后更新于 2026-11-18。本文只记录公开资料支持的接口约束、授权建议和评测构造；产品接口与仓库文档为动态页面时，以访问日期和具体 API 模式限定适用范围。

## 工具协议与 schema

- **OpenAI Function Calling**：[官方文档](https://developers.openai.com/api/docs/guides/function-calling)，访问日期 2026-10-30；阅读 tool-call/result 往返、call id、parallel function calls 和 strict mode 部分。
- **来源支持的事实**：API 响应可包含零个、一个或多个函数调用；调用对象带有供结果回传关联的 id。strict mode 要求每个 parameters 对象把 `additionalProperties` 设为 `false`，并把 `properties` 中的所有字段列入 `required`；可选字段通过将 `null` 加入类型表示。文档区分 Responses 与 Chat Completions 的默认行为，因此不能把 strict mode 规则外推为所有 API、供应商或调用模式的统一契约。
- **证据边界**：此页面描述 OpenAI 当前公开 API；不证明任意模型都能选对工具，也不表示 schema 通过就满足业务授权。

- **Anthropic Tool Use**：[How tool use works](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works)，访问日期 2026-10-30；阅读 client-executed 与 server-executed tool 的执行责任说明。
- **来源支持的事实**：用户定义的工具由应用提供 schema、执行返回的 `tool_use`，再将结果作为 `tool_result` 回传；若使用服务器执行的工具，应用的执行/回传责任不同。供应商定义的工具协议不应被视为跨平台通用调用对象。
- **证据边界**：页面属于动态产品文档；实际字段、执行方和循环行为须按 API 与工具类型确认。

## 行动—观察轨迹与 Agent 评估

- **ReAct**：[原论文](https://arxiv.org/abs/2210.03629)，2022 年论文页面，访问日期 2026-10-30；阅读摘要及方法说明。
- **来源支持的事实**：论文研究模型交错生成任务相关推理轨迹与行动，并通过外部知识库或环境取得观察；方法案例采用多步 thought/action/observation 轨迹。
- **证据边界**：该论文支持研究方法的定义，不规定现代服务 API、生产 trace 格式或应公开何种内部推理文本。

- **OpenAI Agent Evals**：[Evaluate agent workflows](https://developers.openai.com/api/docs/guides/agent-evals)，访问日期 2026-10-30；阅读 trace grading 和 datasets/eval runs 部分。
- **来源支持的事实**：文档将 trace 描述为一次 workflow 中模型调用、工具调用、guardrails 和 handoffs 的端到端记录，并建议在需要可重复对比时使用数据集和 eval runs。
- **证据边界**：该文档是动态产品指南；产品评测工具、保留策略与可用功能可能变化，不能据此声称某个项目的 trace 天然安全。

- **AgentDojo**：[论文](https://arxiv.org/abs/2406.13352)，2024；阅读摘要。
- **来源支持的事实**：该评估环境研究工具返回不可信外部内容时的间接提示注入，包含任务、攻击与防御的可扩展评测设计。
- **证据边界**：论文设置和其攻击成功率不等于其他模型、工具、权限和部署环境的安全结果。

- **ToolSandbox**：[论文](https://arxiv.org/abs/2408.04682)，2024；阅读设计与评估部分。
- **来源支持的事实**：该 benchmark 支持有状态工具执行和对话式多步任务；以轨迹中的 milestones 与不得发生的 minefields 检查任务过程，覆盖工具状态依赖、并行调用顺序和信息不足场景。
- **证据边界**：benchmark 分数只适用于其模型、场景、工具和匹配规则；只能借用测试构造，不应横向当作本地 Agent 的成功率或安全率。

## 高风险动作授权与日志

- **OWASP Transaction Authorization Cheat Sheet**：[官方指南](https://cheatsheetseries.owasp.org/cheatsheets/Transaction_Authorization_Cheat_Sheet.html)，访问日期 2026-10-30；阅读交易数据展示、服务端强制、授权状态转换、一次性使用、有效期和执行前检查部分。
- **来源支持的事实**：指南要求用户能识别并确认关键交易数据，授权在服务端实施，交易数据在授权期间不得被篡改，授权流程按允许的状态顺序推进，并在执行前做最终授权检查；授权凭据应限时、按操作唯一。
- **证据等级与边界**：source-backed（OWASP 开发安全指南）。把这些要求组合成 Agent 的服务端 `approval_id` 与动作摘要记录，是本项目的工程化设计推论。共享状态中的原子检查/消费可用于处理并发与重放，但不是该指南明确规定的 Agent 协议或专门原子操作要求。

- **OWASP Logging Cheat Sheet**：[官方指南](https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html)，访问日期 2026-10-30；阅读需要移除、遮蔽、散列或加密的日志字段，以及事件关联和访问控制部分。
- **来源支持的事实**：日志字段应由采集目的决定；访问令牌、密码、密钥、敏感个人数据、支付信息和高等级数据通常不应直接写入日志，需要删除、遮蔽、散列或加密，并限制日志访问。
- **证据边界**：指南不能代替具体系统的数据分类、授权、保留期限和审计设计；简单字段名单无法发现嵌在自由文本中的敏感信息。

- **OpenTelemetry GenAI spans**：[语义约定页面](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)，`main` 分支动态页面，访问日期 2026-10-30；阅读工具参数/结果属性与 buffered content capture 部分。
- **来源支持的事实**：GenAI 工具参数和结果字段可能包含敏感信息；完整系统指令、用户消息和模型输出通常既敏感又较大，当前约定建议默认不采集内容，按明确配置选择采集或存外部并在 span 记录引用。
- **证据边界**：语义约定处于持续演进状态，且不会自动实施脱敏、访问控制、留存或合规保障。实现时还须固定具体语义约定和 SDK 版本。

## MCP、A2A 与网络/文件边界

- **Model Context Protocol 规范**：[2026-07-28 官方规范](https://modelcontextprotocol.io/specification/2026-07-28)，访问日期 2026-10-30；阅读概览、安全与生命周期说明。
- **来源支持的事实**：该版本将协议消息定义为 JSON-RPC 2.0；概览描述自包含、无状态请求和按请求进行能力协商。Server 可提供 Tools、Resources、Prompts，Client 可提供 Elicitation；Tasks、Skills over MCP、MCP Apps 列为可选扩展。规范安全原则要求 Host 给用户提供数据暴露与工具动作的控制，但规范本身不能替实现落实这些控制。
- **证据边界**：上述结论绑定 2026-07-28 规范快照；不能据此推断更早/更新版或某个 SDK 的支持情况。Roots 等旧版或可选能力须按目标版本和实现分别确认。

- **Agent2Agent (A2A)**：[v1.0.1 规范](https://a2a-protocol.org/v1.0.1/specification/)、[v1.0 迁移说明](https://a2a-protocol.org/latest/whats-new-v1/) 和[项目 release 记录](https://github.com/a2aproject/A2A/releases)，访问日期 2026-10-30。
- **来源支持的事实**：项目 release 记录列出 v1.0.1；规范的协议版本协商使用 Major.Minor，因此该补丁版的 wire protocol version 是 `1.0`。v1.0 迁移资料记录了结构变化，包括移除 Part/流事件中的内嵌 `kind` 判别字段以及更新枚举值；v1.0 AgentInterface 声明 transport binding 与 `protocolVersion`，HTTP 请求使用 `A2A-Version`。
- **证据边界**：迁移摘要与规范绑定 A2A 版本；接入时需锁定实际实现版本并做兼容测试。Agent Card 声明能力不代表调用方拥有业务授权，也不证明目标任务可被完成。

- **OWASP SSRF Prevention Cheat Sheet**：[官方指南](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html)，访问日期 2026-10-30；阅读目标 allowlist、重定向、DNS 与网络层控制建议。
- **来源支持的事实**：对已知受信目标优先采用 allowlist；限制或禁用自动重定向，避免绕过目标校验；除应用层验证外，使用网络层出站规则形成纵深防御。若确需访问任意公网目标，还需处理解析到私有/特殊地址的情况。
- **证据边界**：具体网络地址、云元数据端点、DNS 行为和代理链依部署环境而异；单次字符串校验不等于连接目标已被安全约束。

- **OWASP Path Traversal**：[社区攻击说明](https://community.owasp.org/attacks/Path_Traversal)，访问日期 2026-10-30；阅读相对路径片段和绝对路径示例。
- **来源支持的事实**：路径穿越可利用 `../` 等相对路径片段，也可通过绝对路径绕过预期目录约束；因此仅过滤某个固定字符串不足以确保文件访问被限制在根目录内。
- **证据边界**：具体路径规范化、符号链接、大小写和竞态行为取决于操作系统与文件 API；应用还需要在实际打开目标时落实目录边界。

- **Anthropic, Building effective agents**：[工程文章](https://www.anthropic.com/engineering/building-effective-agents)，发布于 2024-12-19，访问日期 2026-10-30。
- **来源支持的事实**：文章现有页面提示，2024 年描述的 tooling landscape 已发生变化；其 workflow 与更开放的 agent loop 区分仍可用于架构讨论。
- **证据边界**：该文章是特定时期的工程经验，不是当前框架 API 或跨模型的普遍定律。

## Programmatic Tool Calling 当前 API 边界

- **来源与版本**：[OpenAI Programmatic Tool Calling 官方文档](https://developers.openai.com/api/docs/guides/tools-programmatic-tool-calling)，访问日期 2026-11-02；阅读 tool 类型、路由、runtime、选择建议、Responses API 和 Agents API 部分。
- **来源支持的事实**：文档描述模型生成 JavaScript，在新建的隔离 V8 runtime 中协调已启用工具，可使用循环、条件和并发并处理中间结果；Responses API 通过 `programmatic_tool_calling` 工具启用，并用 `allowed_callers` 选择工具可由模型直接调用、程序调用或两种方式调用。Agents API 管理 Agent loop 并默认启用该能力。官方建议在控制流可预测且代码能返回较小结构化结果时使用；单次调用、需要模型依据每个新结果作语义判断、审批敏感写操作或需保留原生引用/产物的阶段更适合直接调用。当前文档说明该生成程序 runtime 不提供 Node.js、包安装、直接网络、通用文件系统、子进程或跨执行持久 JavaScript 状态。
- **证据等级与边界**：source-backed（官方动态 API 文档）；接口与默认行为绑定访问日和指定 API，需按目标模型、工具类型和 API 版本复核。程序 runtime 的隔离不会自动改变工具实际运行的宿主环境，也不替应用侧执行身份校验、权限、审批、幂等或数据保留控制。

## 间接注入的检索可达性研究

- **来源与版本**：[Chang et al., “Overcoming the Retrieval Barrier: Indirect Prompt Injection in the Wild for LLM Systems”](https://www.usenix.org/conference/usenixsecurity26/presentation/chang-hongyan)，USENIX Security 2026，访问日期 2026-11-02；阅读官方会议页面摘要与威胁/结果范围。
- **来源支持的事实**：作者将恶意内容拆成保证触发检索的 trigger fragment 与承载攻击目标的 attack fragment；论文页面报告其黑盒方法在 11 个 benchmark、8 种 embedding 模型上近乎 100% 检索到攻击内容。在一个自然查询触发的邮件摘要多 Agent 设置中，单封投毒邮件使 GPT-4o 外泄 SSH key 的成功率超过 80%；页面还称所测防御不足以阻止恶意文本被检索。
- **证据等级与边界**：source-backed（USENIX 会议论文页面）；检索率和端到端成功率分别受其 benchmark、embedding、系统、提示、任务及威胁模型限制，不代表所有 RAG/Agent 的一般攻击概率、所有防御失效率或真实流量事件率。

## ASR 与任务效用的联合评估

- **来源与版本**：[Zarif et al., “Ignored, Not Refused: What Low Prompt-Injection Success Rates Hide in Small Tool-Using Agents”](https://arxiv.org/abs/2610.23254)，arXiv:2610.23254v1，2026-10-16；访问日期 2026-11-02。
- **阅读位置**：摘要、§3 Study Design、§4.1–4.3 Results、§5 Discussion/Conclusion、Limitations，以及 Appendix D 的人工校验和 benchmark agreement。
- **来源支持的事实**：作者审计 11 个 2–14B 开放权重工具 Agent 在 InjecAgent 与 AgentDojo 的约 33,000 条轨迹，并用一个 LLM judge 对失败攻击做行为分类。论文报告失败攻击多数被忽略而非明确拒答；提示词允许工具输出中的外部指令时，在 InjecAgent 上的汇总 ASR 明显升高。作者提出 Attack–Task Outcome Matrix（ATOM），在其 16,273 条 AgentDojo 轨迹上分别报告安全且完成 67.2%、安全但未完成 21.7%、不安全但完成 3.8%、不安全且被劫持 7.3%；因此该审计中的 ASR 为 11.1%，任务完成率为 71.0%，两个边际量会掩盖 21.7% 安全但未完成的轨迹。
- **证据等级与边界**：source-backed（arXiv v1 预印本）。行为主结果来自一个 LLM judge；两位作者对跨模型和设置分层抽取的 150 条轨迹（两 benchmark 各 75 条）做人审，四类分组与 judge 的一致率为 96%，Cohen's kappa 为 0.835，仍不是对全部 33,000 条轨迹的人工复核。任务完成只在 AgentDojo 测量，策略提示条件只在 InjecAgent 测量其 ASR。研究使用静态注入的合成单 Agent benchmark、小型开放权重 Agent 和特定工具脚手架；作者明确未验证自适应/白盒攻击、多 Agent 系统或线上部署。以上比例不代表生产发生率或一般 Agent 水平，只支持把攻击结果与正常任务效用、无效动作和样本边界联合报告。

## 跨会话记忆写入面与 MPBench

- **来源与版本**：[Dash et al., “From Untrusted Input to Trusted Memory: A Systematic Study of Memory Poisoning Attacks in LLM Agents”](https://arxiv.org/abs/2606.04329)，arXiv:2606.04329v2，v1 提交于 2026-06-03，v2 修订于 2026-06-18；访问日期 2026-11-03。
- **阅读位置**：摘要、§§1–4.5、§5 讨论/限制，以及附录 A 的漏洞与攻击路径概述。
- **来源支持的事实**：论文将长期记忆写入分为 explicit instruction-executed、system prompt-driven、compaction-driven 和 experience-to-procedure 四条通道；以 MPBench 对 OpenClaw 和 HERMES 两种 Agent 系统评估，实验使用 GPT-OSS-120B、默认系统提示和记忆写入配置。数据集含 3,240 个攻击样例和 2,997 个良性样例，覆盖六类攻击和七个领域。作者报告平均写入 ASR 50.46%、记忆写入成功后的检索影响率 RSR 41.05%；RSR 的分母是写入成功子集，不是所有输入。两种 Agent 的平均 ASR/RSR差别明显，且其记忆写入、自动检索与压缩策略不同。
- **证据等级与边界**：source-backed（arXiv v2 预印本）。作者明确限制包括只测试一个模型、两个 Agent 系统，以及部分场景把外部内容作为带标签的上下文块而未完整模拟真实工具检索链。数字不是其他记忆架构或线上部署的事故率；LLM judge 判定语义写入/召回影响的准确度也不能视为无误。该研究支持区分单轮输入过滤与长期记忆写入/召回门禁，不支持对所有 Agent 的泛化风险结论。

## CounterSteer：模型内部注入缓解及参数来源边界

- **来源与版本**：[Russinovich, “CounterSteer: Suppressing Indirect Prompt Injection with Activation Steering”](https://arxiv.org/abs/2609.36570)，arXiv:2609.36570v1，提交于 2026-09-29；Microsoft Research 页面同于 2026-09 发布；访问日期 2026-11-03。
- **阅读位置**：摘要、§§II–V 的 threat model/评估，§VI mechanism，§VII limitations 和结论。
- **来源支持的事实**：该工作提出逐模型拟合残差流方向并在 prefill 时对 tool-result token 做常驻激活干预。作者在五个开源权重模型（8B–106B）上报告指定 holdout attack success 从 0.21–1.00 降到 0.00–0.17；AgentDojo compromise rate 从 0.10–0.49 降到 0.006–0.079，benign utility 依模型和判定器而异。该工作把 tool hijack、parameter manipulation 和 delegated authority 分开；针对参数污染的 best-of-24 query search 在 18 个开发样本中仍击穿 13 个样本，作者据此提出参数来源控制作为互补层。
- **证据等级与边界**：source-backed（单作者 arXiv v1 预印本及作者所属机构页面）。方法需要白盒 serving、可标记 tool-result spans 和逐模型拟合；论文自己报告效用成本、模型/攻击面不完整与仍未测试的自适应范围。具体数值绑定其任务、基线、模型和判分口径，不证明第三方 API、用户消息、任意 RAG 或生产系统可复现同等效果。它支持“模型侧缓解不能替代服务端参数来源和授权检查”，不是已验证的通用防火墙。

## 重试语义与执行隔离配置

- **RFC 9110 §9.2.2 幂等方法**：[RFC Editor 正式文本](https://www.rfc-editor.org/rfc/rfc9110#section-9.2.2)，访问日期 2026-11-05；阅读幂等定义、通信失败时的重试理由及对非幂等请求的 SHOULD NOT 限制。
- **来源支持的事实**：HTTP 幂等性描述重复相同请求对服务器预期效果的关系；客户端在无法读取响应的通信失败后可以安全重试幂等请求，但不应自动重试非幂等请求，除非能确认请求语义实际幂等或确认原请求未生效。
- **证据边界**：RFC 定义 HTTP 方法语义，不为工单、付款、通知等业务动作自动提供幂等键、结果查询、去重保留时间或补偿机制；应用需另设操作合同。

- **AWS SDK retry behavior**：[官方文档](https://docs.aws.amazon.com/sdkref/latest/guide/feature-retry-behavior.html)，访问日期 2026-11-05；阅读错误分类、退避公式、worked examples 与 full-jitter 说明。
- **来源支持的事实**：该页描述 AWS SDK 的一种按 transient/throttling 分类的退避实现，延迟公式为均匀随机乘数乘以 min(cap, base_delay × 2^retry)，并提供 retry quota 限制重试流量。页面注明其所述 2026 新行为当前需设置 AWS_NEW_RETRIES_2026=true opt in，且 SDK 与服务可能有各自支持和默认值。
- **证据等级与边界**：source-backed（动态官方 SDK 文档）；公式可作一种 full-jitter 教学形式，不能外推为其他 SDK、Agent 工具或自建 executor 的默认策略。请求是否允许重试仍取决于错误类型和动作的幂等语义。

- **Kubernetes Pod Security Standards**：[官方文档](https://kubernetes.io/docs/concepts/security/pod-security-standards/)，访问日期 2026-11-05；当前导航指向 Kubernetes v1.37，页面标注 2026-10-16 更新；阅读三档策略总览及 Restricted policy 的说明和相关字段约束。
- **来源支持的事实**：文档定义 Privileged、Baseline 和 Restricted 三种累积策略；Restricted 面向采用当前 Pod hardening practices 的安全关键应用和较低信任用户，限制特权容器、HostPath、特权提升、root 身份及 Linux capabilities 等 Pod 字段。
- **证据等级与边界**：source-backed（Kubernetes 官方配置标准）；Pod policy 约束 Kubernetes 对 Pod 配置的接受范围，不是任意代码逃逸、内核漏洞、多租户数据隔离或微虚拟机比较的实测证明。关于更强隔离技术的选择仍需依据部署威胁模型判断。

## 2026-11-13 工具往返、thinking 交错与 MCP 当前协议面

- **OpenAI Responses Function Calling**：[官方文档](https://developers.openai.com/api/docs/guides/function-calling)，2026-11-13 访问；阅读术语定义、tool calling flow、function-call output 与多轮继续部分。Responses 将调用和结果表示为不同类型的 item，结果通过原 `call_id` 关联；应用执行 client-owned function 后回传结果并继续请求，直到最终响应或下一次工具调用。`call_id` 只关联 API 项，不代表业务授权。文档同时包括托管工具，实际执行责任需按工具类型区分；接口规则不外推到其他 provider 或 API surface。
- **Anthropic Messages 工具往返**：[Handle tool calls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls) 与 [Thinking](https://platform.claude.com/docs/en/build-with-claude/thinking)，2026-11-13 访问；阅读 `tool_use`/`tool_result`、`tool_use_id`、`is_error`、结果块顺序、thinking block 回传和 interleaved thinking 可用范围。client tool 由应用执行，工具结果需与原 tool-use ID 对应；工具循环中的 thinking block 必须按原顺序回传，不能编辑或局部删除。interleaved thinking 是否可用以及是否需 beta header 依模型/模式而变，且该文档所述能力限 Messages API。以上均为当前动态 API 合同，不表明隐藏推理可见，也不赋予应用授权。
- **Model Context Protocol**：阅读[2026-07-28 版官方规范](https://modelcontextprotocol.io/specification/2026-07-28)、[发布说明](https://blog.modelcontextprotocol.io/posts/2026-07-28/)及 [Tasks extension 2026-07-28](https://tasks.extensions.modelcontextprotocol.io/specification/2026-07-28/tasks)，访问日期 2026-11-13。正式版本把核心改为自包含无协议会话请求，移除 `initialize`/`initialized` 握手和 `Mcp-Session-Id`；请求携带版本与 client 元数据，需要预先发现能力时可调用 `server/discover`。Multi Round-Trip Requests 将工具中途需要的 client 输入随原请求的响应/更新流转。Tasks 从核心实验功能移入 `io.modelcontextprotocol/tasks` 扩展，当前适用 `tools/call`，由 server 决定是否返回任务句柄；客户端通过 `tasks/get`、`tasks/update`、`tasks/cancel` 查询、更新和取消。取消语义为协作式且最终状态可能不是 `cancelled`。应用仍可保存外部业务状态；无协议会话不等于无状态业务。
- **证据等级与边界**：source-backed（供应商动态官方文档、MCP 版本化规范和扩展规范）。OpenAI/Anthropic 字段绑定其 API 与模型版本；MCP 事实绑定 2026-07-28 规范和 Tasks 扩展，不证明每个 SDK/host 已实现新版功能。API 文档不揭示模型内部策略，不替代 host 的业务授权、幂等执行、结果验证和状态恢复。

## 2026-11-13 工具返回内容与 Agent 安全分类

- **来源与版本**：OWASP GenAI Security Project [Top 10 for LLM Applications](https://genai.owasp.org/initiatives/top-10-for-llm-and-genai/)，2025 版页面，访问日期 2026-11-13；Anthropic [Handle tool calls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls)，访问日期 2026-11-13；MCP [Security Best Practices](https://modelcontextprotocol.io/docs/2026-07-28/tutorials/security/security_best_practices)，对应 2026-07-28 页面，访问日期 2026-11-13。
- **实际阅读范围**：OWASP 当前 LLM01 Prompt Injection、LLM06 Excessive Agency 列表；Anthropic 对网页、邮件、上传文件等 tool results 的 untrusted-content 提醒；MCP 安全页面目录中的 confused deputy、scope minimization、token passthrough、SSRF 和 state-handle hijacking 风险。
- **来源支持的事实**：OWASP 2025 Top 10 将 Prompt Injection 与 Excessive Agency 列为风险类别；Anthropic 要求将不受控制的工具结果作为不可信内容处理；MCP 独立列出其实现特定的授权、委托、scope 和资源访问安全风险与缓解指导。
- **证据等级与边界**：source-backed（OWASP 风险分类和官方协议/供应商安全文档）。这些资料用于提出威胁模型与控制点，不是目标 Agent 的攻击成功率或生产风险率测量；具体缓解仍需在目标工具、权限和执行环境中测试。

## 2026-11-16 工具 schema、工具往返与 trace grading 当前页面复核

- **OpenAI Function Calling**：[官方文档](https://developers.openai.com/api/docs/guides/function-calling)，访问日期 2026-11-16；复核 strict mode、Responses/Chat Completions 默认行为和并行函数调用部分。
- **来源支持的事实**：strict mode 要求 parameters 中的对象设置 additionalProperties=false，properties 中的字段都列为 required；Responses 请求会在可兼容时尝试规范化到 strict mode，不兼容时可能回退到 best-effort，并在响应中显示 strict=false；Chat Completions 默认仍为 non-strict。文档还说明可通过 parallel_tool_calls=false 将一轮限制为零个或一个函数调用。
- **适用边界**：这些默认值和字段约束绑定 OpenAI 当前 API surface；schema 符合要求不等于调用正确、业务合法或已获授权。接口允许多调用也不代表工具之间没有依赖，执行器仍要负责预算、并发和先后顺序。

- **Anthropic Handle tool calls**：[官方文档](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls)，访问日期 2026-11-16；复核 client tool 的 tool_use/tool_result 往返、错误标记和 server tool 差异。
- **来源支持的事实**：客户端工具的 tool_use 块包含 id、name 和 input；应用执行后通过 tool_use_id 关联 tool_result，可使用 is_error 表示工具错误。文档对结果块位置有顺序约束；服务端工具具有不同的执行责任和回传规则。
- **适用边界**：此结构属于 Anthropic Messages API 与相应工具类型，不能当作跨供应商的统一 wire format；ID 用于请求和结果关联，不授予业务权限。

- **OpenAI Evaluate agent workflows**：[官方文档](https://developers.openai.com/api/docs/guides/agent-evals)，访问日期 2026-11-16；复核 trace grading 的说明。
- **来源支持的事实**：文档将单次 workflow trace 描述为模型调用、工具调用、guardrails 和 handoffs 的端到端记录，并建议用结构化 grader 查找工作流级失败和回归。
- **适用边界**：产品 trace 与 grader 不自动提供脱敏、访问控制或正确性保证；评分仍需结合任务定义、人工复核和本地固定样例。
- **ReAct 原论文**：[arXiv 页面](https://arxiv.org/abs/2210.03629)，2026-11-16 复核当前页面；页面指向 v3 camera-ready 版本，摘要描述模型交错生成任务相关轨迹和动作，并从外部知识库或环境取得观察。该论文支持方法背景，不规定现代工具 API、生产轨迹 schema 或内部推理披露策略。
- **OWASP Transaction Authorization 与 Logging Cheat Sheets**：[交易授权](https://cheatsheetseries.owasp.org/cheatsheets/Transaction_Authorization_Cheat_Sheet.html)和[日志指南](https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html)，2026-11-16 复核。交易授权指南支持确认关键交易数据、服务端执行授权、限制有效期、按操作唯一并在执行前检查；原子检查/消费和 Agent approval record 属工程推导。日志指南建议对访问令牌、密钥、敏感个人数据等移除、遮蔽、散列或加密。
- **OpenTelemetry GenAI spans**：[当前语义约定](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)，main 分支动态页面，2026-11-16 复核。页面建议默认不采集完整系统指令、输入和输出，可由明确配置选择采集；约定本身不落实脱敏、访问控制或留存策略。

## 2026-11-17 Structured Outputs 边界复核

- **来源与版本**：OpenAI [Structured Outputs 官方文档](https://developers.openai.com/api/docs/guides/structured-outputs)，动态 API 文档，访问日期 2026-11-17；阅读 schema adherence、refusal/incomplete response 和 business rule 说明。
- **来源支持的事实**：文档说明 Structured Outputs 可约束非拒绝的完整响应符合所提供且受支持的 JSON Schema；应用仍需处理拒绝、不完整输出和 API 错误，并验证 schema 未表达的业务规则。
- **证据等级与边界**：source-backed（OpenAI 当前公开 API 行为）。严格 schema 只约束其支持范围内的结构，不证明语义正确、动作获得授权或其他 provider 采用相同保证；API 细节应按访问时版本复核。

## 2026-11-17 间接提示注入结果与轨迹隐私复核

- **研究来源**：[Ignored, Not Refused: What Low Prompt-Injection Success Rates Hide in Small Tool-Using Agents](https://arxiv.org/abs/2610.23254)，v1，2026-10-16 提交；2026-11-17 阅读摘要、研究设计、结果和局限。
- **实际阅读位置**：研究设计与威胁模型、AgentDojo/InjecAgent 样本说明、失败行为分类、Attack–Task Outcome Matrix、局限。
- **来源支持的事实**：作者对 11 个 2B—14B 开源 Agent 在两个合成单 Agent benchmark 中的 33,000 条轨迹做 LLM-judge 行为审计；报告 benchmark 记为“攻击失败”的轨迹可进一步分为忽略注入、显式拒绝、机械故障或部分/完整服从。作者在其样本中观察到，低攻击成功率不必然表示显式拒绝或完成了用户任务，并建议同时报告行为和任务结果。
- **证据等级与边界**：当前按 arXiv 预印本记录，不据此宣称独立复现或生产验证。研究只使用静态注入、两个合成单 Agent benchmark、小型开源模型和指定工具脚手架，采用一个经人工标签核验的 LLM judge；没有证明对自适应/白盒攻击、多 Agent 或真实生产环境的韧性。论文中的比例不能外推到其他模型、部署或任务。

- **OpenAI Agents SDK Tracing**：[官方动态文档](https://openai.github.io/openai-agents-python/tracing/)，2026-11-17 访问；阅读敏感数据、trace processors 与 redaction before export 部分。
- **来源支持的事实**：当前文档写明 `trace_include_sensitive_data` 默认值为 `True`；通过 `add_trace_processor()` 增加的处理器不会移除默认 exporter。处理器回调异常会被捕获并继续调用其他 processor，因此单独注册的脱敏处理器失败时，不能阻止另一个 exporter 收到原始数据。文档建议依赖脱敏成功时，在同一应用自有 exporter 内完成脱敏和发送，并在脱敏失败时丢弃该批次。
- **证据等级与边界**：source-backed（访问日官方 SDK 文档）；此结论绑定动态页面所描述的 OpenAI Agents SDK 实现，不代表其他 tracing SDK。不能据此确认任一本地部署的版本、配置、exporter 顺序或数据已经安全脱敏，需核实实际锁定版本和运行配置。

- **OpenTelemetry GenAI spans**：[开发中的语义约定](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)，动态 `main` 页面，2026-11-17 访问；复核 status、content capture configuration 与 external storage 部分。
- **来源支持的事实**：页面将该约定标为 Development；模型指令、输入、输出和工具调用参数/结果内容需要显式选择采集，默认 `NO_CONTENT`。该约定不规定外部对象的访问机制、留存期限或存储布局。
- **证据等级与边界**：source-backed（动态开发中语义约定）；需固定访问时 commit/SDK 才能复核具体实现。约定本身不落实脱敏、权限、留存或目标系统的合规义务。

## 2026-11-18 工具目录、schema、结果与执行边界复核

- **OpenAI Function Calling**：[官方文档](https://developers.openai.com/api/docs/guides/function-calling)，2026-11-18 访问；阅读 “How it works” 与 “The tool calling flow”、strict mode、tool choice、并行调用和结果格式部分。
- **来源支持的事实**：模型响应可包含工具调用；应用执行其自定义函数后将结果与原调用关联并继续请求模型。结果可以按应用选择的字符串格式返回，例子包含结构化 JSON 和纯文本。strict=true 约束受支持 schema 范围内的参数格式；对象需要 additionalProperties=false，所有 properties 均列为 required，可选字段可包含 null。文档说明 Responses 与 Chat Completions 的默认 strict 行为不同，也允许零个、一个或多个调用并通过 tool choice/allowed tools 限制当前可调用集合。
- **证据等级与边界**：source-backed（OpenAI 当前动态 API 文档）。执行和字段结论绑定相应 API 与工具类型；schema 符合不证明语义正确、业务授权、目标安全或任务已验收，也不能外推至其他 provider。

- **OpenAI Agents SDK Tools**：[官方工具文档](https://openai.github.io/openai-agents-python/tools/)，2026-11-18 访问；阅读 hosted tool search、Function tools、工具 schema 生成和直接调用包装函数的说明。
- **来源支持的事实**：SDK 可根据 Python 函数名、docstring 和参数类型生成工具名、描述及 JSON schema；文档明确指出直接调用 decorated function 的 wrapped callable 会绕过该 SDK runtime 的 schema validation、context injection、guardrails、timeouts、failure handling 和 tracing。工具搜索/deferred loading 可延迟暴露部分工具表面，文档将其列为 Responses API 的特定能力。
- **证据等级与边界**：source-backed（当前 SDK 文档）。绕过的是所述 SDK runtime pipeline；不能据此声称所有框架具有相同包装层、工具执行责任或发现能力。

- **Model Context Protocol Tools**：[2026-07-28 版 server/tools 规范](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)，2026-11-18 访问；阅读用户交互模型、Tool 数据类型、structured content、output schema、错误处理及安全考虑部分。
- **来源支持的事实**：该版本 Tool 定义包含 name、description、inputSchema，并允许可选 outputSchema 与 annotations；客户端必须将来自非可信 server 的 annotations 视为不可信。声明 outputSchema 时 server 必须返回符合 schema 的结构化结果，client 应进行验证。规范将协议请求结构错误与工具执行错误分开；后者可通过 isError 和可操作内容返回，并建议 client 把执行错误交给模型以便自我纠错。规范指出 MCP 不强制单一 UI 交互模式；在人机确认、显示输入、校验结果、超时和日志方面使用 SHOULD 建议，同时对 server 输入校验、访问控制、速率限制和输出净化规定 MUST 要求。
- **证据等级与边界**：source-backed（固定版本的官方协议规范）。MUST/SHOULD 只表示该版本协议的规范性要求；不能证明具体 host/server 已实现，也不意味着协议自身完成业务授权、沙箱或数据保留控制。

- **未验证项与分歧边界**：当前三类来源共同支持“工具定义、运行时执行、权限策略和任务验收不是一个环节”，但没有规定跨供应商统一的 Registry 字段、错误枚举、权限等级或 trace 格式。具体实现须按 provider/API、MCP 版本、host 和执行器分别复核。上述事实均未由本地代码或生产运行验证。

## 2026-11-18 工作区路径、符号链接与 Git patch/diff 语义复核

- **OWASP Path Traversal Prevention**：[官方指南](https://cheatsheetseries.owasp.org/cheatsheets/Path_Traversal_Prevention_Cheat_Sheet.html)，2026-11-18 访问；阅读路径解码/规范化、固定根目录、符号链接和 TOCTOU 部分。
- **来源支持的事实**：指南建议在固定 base directory 下解析规范化路径，检查解析结果仍在该目录内并使用已检查的路径；若检查与打开之间可替换 symlink，仍会有 TOCTOU 风险。指南举例说明使用能在打开时强制根目录约束的 API（如 Linux `openat2` 或 Go `os.Root`）以及最小权限。
- **证据等级与边界**：source-backed（OWASP 安全指南）。具体安全保证由平台和文件 API 实现决定；字符串规范化本身不能证明随后打开的文件对象未发生替换。

- **Linux pathname lookup**：[Linux 6.15 内核文档](https://docs.kernel.org/6.15/filesystems/path-lookup.html)，2026-11-18 访问；阅读 openat2 resolution flags 与 pathname lookup race 部分。
- **来源支持的事实**：文档说明 `openat2` 的 `resolve` 字段向路径查找提供组件级约束；内核内部的 `LOOKUP_BENEATH` 阻止解析结果越过起始点，`LOOKUP_NO_SYMLINKS` 阻止符号链接遍历，并使用 rename/mount 检查发现特定竞态。对外 API 名称 `RESOLVE_BENEATH` 的例子由 OWASP 指南明确列出。
- **证据等级与边界**：source-backed（Linux 6.15 内核文档）；文中的 `LOOKUP_*` 是 Linux 内部路径查找语义，不是跨平台文件 API 合同，也不证明某个 Agent 执行器已经采用相应的打开约束。

- **Git apply**：[官方命令文档](https://git-scm.com/docs/git-apply)，2026-11-18 访问；阅读 `--check`、默认失败行为、`--reject`、`--reverse` 和 `--3way`。
- **来源支持的事实**：`--check` 检查 patch 是否可应用而不写入；默认行为下无法应用的 hunk 会使整个 patch 被拒绝且不修改工作树；`--reject` 可让可匹配 hunk 部分应用并把失败 hunk 写入 `.rej`；`--reverse` 尝试反向应用 patch。三方应用可能留下冲突标记。
- **证据等级与边界**：source-backed（Git 当前命令文档）；上述行为只描述 Git `apply`，不能外推为所有文件编辑工具的原子性、冲突或回滚保证。patch 可应用也不证明语义正确。

- **Git diff 与 gitignore**：[git diff](https://git-scm.com/docs/git-diff) 和 [gitignore](https://git-scm.com/docs/gitignore) 官方文档，2026-11-18 访问；阅读工作区/暂存区比较及 tracked-file ignore 说明。
- **来源支持的事实**：`git diff` 默认比较工作区与 index，`git diff --cached` 比较 index 与提交基线；不同选项可以选择其他比较对象。`.gitignore` 规则针对未跟踪文件，已经被 Git 跟踪的文件不受该规则影响。
- **证据等级与边界**：source-backed（Git 命令与配置文档）。单一 diff 视图不能自动区分作者，也不涵盖所有未跟踪状态；`.gitignore` 是版本控制排除机制，不是运行时文件访问控制。

## 2026-11-18 终端命令执行、隔离、资源与输出边界复核

- **命令权限匹配**：[Claude Code 权限文档](https://code.claude.com/docs/en/permissions)，2026-11-18 访问；阅读 Bash 规则、复合命令、包装器和规则边界部分。
- **来源支持的事实**：该产品的 Bash 规则匹配完整命令文本；文档说明其会识别一组 shell 分隔符并分别检查子命令，也会剥除特定包装器，但相同程序通过绝对路径或 `sh -c` 等形式调用时可能不再匹配同一规则。文档明确指出命令字符串规则不是围绕程序行为的安全边界，参数模式也可能被选项、协议、重定向、变量和 URL 跳转改变。
- **证据等级与边界**：source-backed（Claude Code 当前产品文档）。这些是该产品公开的匹配行为和警告，不定义通用命令解析器，也不证明其他 harness 的规则相同。

- **命令沙箱与权限层**：[Claude Code 沙箱文档](https://code.claude.com/docs/en/sandboxing)，2026-11-18 访问；阅读 OS-level enforcement、permissions relationship 和 security limitations。
- **来源支持的事实**：该产品说明权限规则在工具执行前决定工具调用是否允许，操作系统沙箱在命令运行时限制文件系统和网络访问，并作用于 Bash/PowerShell/Monitor 命令及其子进程；当前产品在 macOS 使用 Seatbelt、Linux/WSL 使用 bubblewrap。其沙箱仅覆盖 shell 命令，文件工具和 MCP 等工具在沙箱外运行。文档还指出域名代理默认不终止或检查 TLS，允许宽泛域名有外传风险；放行 Docker socket 可能触达宿主机。
- **证据等级与边界**：source-backed（Claude Code 当前产品文档）。平台、版本和工具覆盖范围均绑定该产品；不能从“开了沙箱”推出所有进程、工具和网络都已隔离。

- **OpenAI Agents API 环境安全**：[官方 Sandbox security 文档](https://developers.openai.com/api/docs/guides/agents-api/environments/security)，2026-11-18 访问；阅读隔离计算、网络访问、automatic review 和凭据分离部分。
- **来源支持的事实**：文档称 Agent 生成的代码可以访问环境可用的文件、凭据和网络，建议隔离工作负载、仅允许已批准的出站目标，并指出 automatic review 不替代环境网络限制和凭据控制。它要求应用 API key 留在环境之外；其 executor environment key 虽然 scope 受限，生成代码仍可读取该环境变量；第三方凭据建议由可信 proxy、server、vault 或调用工具所在应用代为提供。
- **证据等级与边界**：source-backed（OpenAI 当前 Agents API 产品文档）。key scope、review profile 与网络职责是该 API 的公开合同，不能推为其他执行器的保证。

- **Docker Engine 安全**：[官方 Engine security 文档](https://docs.docker.com/engine/security/)，2026-11-18 访问；阅读 namespaces、cgroups、daemon attack surface 和容器配置部分。
- **来源支持的事实**：文档将内核 namespace/cgroup、Docker daemon 攻击面、容器配置缺口和内核 hardening 作为不同安全审查面；说明 cgroup 可进行资源核算与限制，而 Docker daemon 的特权能力和宿主目录挂载可能扩大容器对宿主文件系统的影响。
- **证据等级与边界**：source-backed（Docker 当前 Engine 文档）。文档用于说明容器配置及 daemon 权限仍属威胁面，不能把容器视为自动等价于强隔离。

- **Linux cgroup v2 资源接口**：[Linux 6.18 cgroup v2 文档](https://docs.kernel.org/6.18/admin-guide/cgroup-v2.html)，2026-11-18 访问；阅读 `cpu.max`、`memory.max` 和资源事件部分。
- **来源支持的事实**：`cpu.max` 用 MAX/PERIOD 表示非 root cgroup 的 CPU 带宽上限，具体作用受调度器类型限制；`memory.max` 是主要内存硬限制机制，达到上限且不能回收时会在该 cgroup 触发 OOM 处理，使用量在特定条件下可能暂时超过设定值。文档同时提供资源压力和 OOM 事件计数接口。
- **证据等级与边界**：source-backed（固定版本 Linux 内核文档）。这些是 Linux cgroup v2 语义，不代表所有 OS 的资源控制接口或严格绝不超限保证。

- **OWASP LLM06:2025 Excessive Agency**：[官方风险说明](https://genai.owasp.org/llmrisk/llm062025-excessive-agency/?lang=us)，2026-11-18 访问；阅读风险成因和预防建议。
- **来源支持的事实**：该风险由功能、权限或自主性过度引起，触发因素包括模型误输出和直接/间接 prompt injection。建议缩减扩展功能和权限，能用窄能力扩展时避免开放式 shell/URL 扩展，高影响动作要求人工批准，并在下游系统实施授权；日志、监控和 rate limiting 用于限制损害，不单独预防过度授权。
- **证据等级与边界**：source-backed（OWASP 2025 风险说明及建议）；建议不是特定产品接口合同，也不能单独证明某一控制已经充分。

- **OWASP LLM10:2025 Unbounded Consumption**：[官方风险说明](https://genai.owasp.org/llmrisk/llm102025-unbounded-consumption/)，2026-11-18 访问；阅读资源耗尽案例和缓解建议。
- **来源支持的事实**：过量、无控制的推理可能造成拒绝服务、经济损失、模型提取或服务退化；指南列出输入验证、rate limit/用户配额、动态资源管理、timeout/throttling、sandbox 和监控等缓解项。
- **证据等级与边界**：source-backed（OWASP 2025 风险说明及建议）。不为终端执行器提供通用 timeout 数值、cgroup 配置或发布阈值。

- **OWASP SSRF Prevention Cheat Sheet**：[官方指南](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html)，2026-11-18 访问；阅读域名 allowlist、DNS rebinding 与连接目标校验部分。
- **来源支持的事实**：在服务端请求伪造防护语境中，域名 allowlist 本身不能阻止 DNS rebinding；需要校验解析后的目标 IP，并保证 HTTP 客户端连接到已校验地址；校验和连接之间再次未受控解析，以及重定向、重试或 fallback，可能绕过目的地址规则。
- **证据等级与边界**：source-backed（OWASP SSRF 指南）。该建议针对处理不可信目标 URL 的 SSRF 路径，不是所有 shell 出站网络的完整安全方案。

- **Python subprocess 超时语义**：[Python 3.15.2 官方 subprocess 文档](https://docs.python.org/3/library/subprocess.html)，2026-11-18 访问；阅读 `subprocess.run` 与 `timeout` 参数说明。
- **来源支持的事实**：`subprocess.run(timeout=...)` 在超时后会终止并等待所启动的 child process，然后重新抛出 `TimeoutExpired`；部分平台的初始进程创建本身不可中断，因此异常可能晚于 timeout 才出现。
- **证据等级与边界**：source-backed（Python 3.15.2 API 文档）。该说明针对所启动的 child process，不构成进程后代树、OS job、容器或外部副作用均已取消的证明。

- **OWASP Logging Cheat Sheet**：[官方指南](https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html)，2026-11-18 访问；阅读 event attributes、data to exclude 和 log confidentiality 部分。
- **来源支持的事实**：日志指南要求按用途选取事件字段，并识别应排除或脱敏的敏感数据；其威胁模型涵盖日志的机密性、完整性、可用性与问责性。
- **证据等级与边界**：source-backed（OWASP 日志指南）。具体字段、保留周期、访问控制和脱敏方法须按系统与数据分类确定。

## 2026-11-18 Shell 参数化、npm lifecycle 与 Git 破坏性命令语义复核

- **OWASP OS Command Injection Defense**：[官方 Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/OS_Command_Injection_Defense_Cheat_Sheet.html)，2026-11-18 访问；阅读优先使用库/API、参数化和输入校验部分。
- **来源支持的事实**：指南把避免直接调用 OS 命令、优先使用用途更窄的内建库/API，列为主要防御；无法避免时，应使用结构化参数化以分离命令与数据，并与输入校验一起使用。
- **证据等级与边界**：source-backed（OWASP 安全指南）。分离参数降低 shell 元字符注入风险，不证明目标程序参数本身无危险，也不替代权限、沙箱和资源限制。

- **npm 生命周期**：[npm CLI v11 Scripts 文档](https://docs.npmjs.com/cli/v11/using-npm/scripts/)，2026-11-18 访问；阅读 `npm install`、`npm ci` 与用户脚本执行顺序。
- **来源支持的事实**：文档列出 `npm install` 和 `npm ci` 的生命周期事件，包括 `preinstall`、`install`、`postinstall`、`prepare` 等；`npm run <user-defined>` 会依次运行同名脚本的 `pre`、主体和 `post` 生命周期函数。执行这些命令不等于只进行文件下载或只读查询。
- **证据等级与边界**：source-backed（npm CLI v11 文档）。实际脚本由项目和依赖定义；具体行为还与 npm 版本、配置和项目内容有关。

- **Git reset 与 Git clean**：[git reset](https://git-scm.com/docs/git-reset) 和 [git clean](https://git-scm.com/docs/git-clean) 官方命令文档，2026-11-18 访问；阅读 hard reset 与未跟踪文件移除说明。
- **来源支持的事实**：`git reset --hard` 会重置 index 和工作树状态，官方示例说明它会丢弃本地改动；`git clean` 递归删除工作树中的未跟踪文件，`-x` 还会包含被忽略文件。
- **证据等级与边界**：source-backed（Git 命令文档）。具体受影响对象取决于参数、工作目录、配置和当前仓库状态；命令名本身不能说明用户有哪些未提交或未跟踪内容。

## 2026-11-18 Linux 进程组脱离与 cgroup v2 清理接口复核

- **Linux `setsid(2)`**：[Linux man-pages](https://man7.org/linux/man-pages/man2/setsid.2.html)，man-pages 6.19，2026-11-18 访问；阅读 `setsid()` 的 session/process-group 语义。
- **来源支持的事实**：满足条件的进程可调用 `setsid()` 创建新 session，并成为新 session/process group 的 leader；其后代继承 session ID，直到显式改变。由此可推断，单靠向原 process group 发信号，不能作为“任意后代均已退出”的普遍证明。
- **证据等级与边界**：source-backed（Linux man-pages）。session/process-group 行为属于 POSIX/Linux 进程接口；这里只说明其对进程组清理的影响，不代表任意执行器都允许或能阻止会话变化。

- **Linux cgroup v2 kill 与状态**：[Linux 6.18 cgroup v2 文档](https://docs.kernel.org/6.18/admin-guide/cgroup-v2.html)，2026-11-18 访问；阅读 `cgroup.kill` 和 `cgroup.events`。
- **来源支持的事实**：非 root cgroup 上的 `cgroup.kill` 写入 `1` 会通过 SIGKILL 终止该 cgroup 树中的进程；文档说明它会处理并发 fork 并防止迁移竞态。`cgroup.events` 中的 `populated` 在本 cgroup 或其后代存在存活进程时为 1，否则为 0。
- **证据等级与边界**：source-backed（固定版本 Linux 内核文档）。接口仅适用 Linux cgroup v2；SIGKILL 是强制终止，不是优雅清理，也不证明命令此前没有产生文件、网络或业务副作用。

## 2026-11-18 权限授权、审批绑定与工具覆盖范围复核

- **OWASP LLM06:2025 Excessive Agency**：[官方风险说明](https://genai.owasp.org/llmrisk/llm062025-excessive-agency/?lang=us)，2026-11-18 再次打开；阅读 excessive functionality / permissions / autonomy、least privilege、high-impact approval 与 downstream authorization 部分。
- **来源支持的事实**：风险来自超出任务需要的扩展功能、权限或自主程度；可能由模型误输出或直接/间接 prompt injection 触发。建议尽量缩小工具数量与每个工具的功能，避免不必要的开放式 Shell/URL 扩展，在高影响动作前加入人工批准，并由下游系统验证授权。
- **证据边界**：source-backed（OWASP 2025 风险说明）；建议不规定 Agent 的统一权限 schema，不证明某产品实现符合这些建议。日志、监控和限速可帮助发现或限制影响，但不能代替执行前授权。

- **OWASP Transaction Authorization Cheat Sheet**：[官方指南](https://cheatsheetseries.owasp.org/cheatsheets/Transaction_Authorization_Cheat_Sheet.html)，2026-11-18 再次打开；阅读关键交易字段可识别、服务端生成与保存、流程顺序、交易数据变更和执行前验证部分。
- **来源支持的事实**：用户需要识别并确认对交易有意义的数据；授权步骤应由服务端按顺序强制；授权期间更改交易数据需要使先前授权失效或重启流程，并在执行前核验。
- **证据等级与边界**：source-backed（OWASP 开发安全指南）。把动作、目标、版本摘要绑定到 Agent approval ID，并设计短时、单次使用、变更后重新审批，是将交易授权原则迁移到 Agent 的工程设计推论，不是 OWASP 定义的 Agent 标准协议。

- **Claude Code Configure permissions 与 sandbox**：[权限规则](https://code.claude.com/docs/en/permissions)和[沙箱说明](https://code.claude.com/docs/en/sandboxing)，2026-11-18 访问；阅读规则顺序、模型与运行时边界、命令沙箱覆盖范围和沙箱外工具列表。
- **来源支持的事实**：该产品说明 permission rule 由 Claude Code 执行而不是模型执行，并按 deny、ask、allow 顺序匹配；其沙箱限制指定的 Shell/PowerShell/Monitor 命令及子进程，但内建文件与 Web 工具、MCP server、hooks、部分辅助进程不因此自动进入同一沙箱。
- **证据等级与边界**：source-backed（Claude Code 当前产品文档）。规则优先级、平台实现和覆盖面只适用于所访问版本；不能外推为其他 Agent 的通用合同或权限最佳实践。

- **OpenAI Agents API Sandbox security**：[官方文档](https://developers.openai.com/api/docs/guides/agents-api/environments/security)，2026-11-18 访问；阅读自动 review、网络限制和凭据边界。
- **来源支持的事实**：该文档说明生成代码能够访问环境可用的文件、凭据和网络；automatic review 不替代出站网络限制与凭据控制；应用 API key 应留在环境之外，环境凭据仍可能被环境内生成代码读取。
- **证据等级与边界**：source-backed（OpenAI 当前 Agents API 文档）。这是该服务的合同说明，不代表其他 provider 或执行器具有相同环境结构或密钥隔离行为。

- **Docker Engine security**：[官方文档](https://docs.docker.com/engine/security/)，2026-11-18 访问；阅读 namespaces、cgroups、daemon attack surface 和共享宿主目录部分。
- **来源支持的事实**：文档将内核 namespaces/cgroups、Docker daemon 攻击面、容器配置和内核 hardening 作为独立审查面；共享宿主目录可以让容器进程影响宿主文件，cgroups 负责资源核算/限制而不负责数据隔离。
- **证据等级与边界**：source-backed（Docker Engine 文档）。不能仅由“使用 Docker”推出具体部署已具备最小权限或强租户隔离。

- **OWASP Logging Cheat Sheet**：[官方指南](https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html)，2026-11-18 再次打开；阅读事件字段、敏感数据排除、日志机密性和访问保护部分。
- **来源支持的事实**：日志事件按用途确定字段；敏感值应依数据与法律要求排除、遮蔽、散列或加密，并保护日志的机密性、完整性、可用性和问责能力。
- **证据等级与边界**：source-backed（OWASP 日志指南）。具体保留策略、合法依据、字段分类和脱敏措施需要按系统确定，不能只靠通用字段名单。

## 2026-11-18 Agent Trace 隐私、处理器与跨服务上下文复核

- **OpenAI Agents SDK tracing**：[官方 tracing 文档](https://openai.github.io/openai-agents-python/tracing/)，2026-11-18 访问；阅读 sensitive data、add/set trace processors 和 redaction before export。
- **来源支持的事实**：该动态文档说明 generation/function spans 可捕获模型和工具输入输出；访问时 trace_include_sensitive_data 默认值为 True。add_trace_processor 增加一个处理器但保留默认 OpenAI exporter；默认 provider 捕获处理器回调异常后仍继续其他处理器，因此独立脱敏处理器失败不保证阻止后续 exporter 获取原数据。文档建议需要脱敏成功才能发送时，用 set_trace_processors 替换默认处理链，在同一个应用自有 exporter 内完成复制、脱敏和发送，并在序列化、复制或脱敏失败时丢弃该批次。
- **证据等级与边界**：source-backed（OpenAI Agents SDK 当前动态文档，访问日期 2026-11-18）。接口、默认值和处理器语义绑定访问版本；未运行本地 SDK，也未验证其他 exporter / SDK 的行为。

- **OpenTelemetry GenAI spans**：[语义约定](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)，动态 main 页面，2026-11-18 访问；阅读内容捕获、外部存储引用和配置值。
- **来源支持的事实**：页面状态为 Development；当前草案建议默认不记录模型指令、用户输入和输出，内容捕获配置的默认值为 NO_CONTENT，并要求内容采集由显式 opt-in 控制。草案也描述可将大内容另存，再在 span 中放引用，以便独立配置访问控制。
- **证据等级与边界**：source-backed（OpenTelemetry GenAI 动态开发中语义约定）。约定不替 instrumentation、SDK、collector 或厂商 exporter 关闭内容采集，也不规定 URI 权限、保留期限、对象结构或脱敏实现。

- **OpenTelemetry Traces 概念文档**：[官方文档](https://opentelemetry.io/docs/concepts/signals/traces/)，2026-11-18 访问；阅读 span、event、link、status 与父子关系。
- **来源支持的事实**：文档将 span 定义为带开始/结束时间的操作，span event 是记录 span 持续期间某个单点事件的结构化信息；parent span 表示嵌套子操作；span links 可以关联异步或跨 trace 的因果操作。文档区分 Unset、Error 和 Ok 状态。
- **证据边界**：source-backed（OpenTelemetry 官方概念文档）。具体 Agent span 名、task 状态和业务验收字段需要由应用与所用语义约定另行定义。

- **W3C Trace Context**：[Recommendation](https://www.w3.org/TR/trace-context/)，2021-11-23 发布，2026-11-18 访问；阅读 traceparent / tracestate 传播、隐私和公开入口 DoS 风险。
- **来源支持的事实**：规范定义 HTTP 头传播 trace 和 vendor-specific correlation context；traceparent / tracestate 仅用于链路关联，不得包含个人身份或其他敏感信息。规范指出公共 API 若盲目遵循调用方采样标志，可能增加 tracing 负载或成本；调用方可按认证状态或记录配额治理。
- **证据等级与边界**：source-backed（W3C Recommendation）。这类 header 用于相关追踪，不是认证、授权或任务身份；跨其他协议的传播需要对应扩展规范。

## 2026-11-18 Agent Trace 采样与指标基数复核

- **OpenTelemetry Sampling**：[官方概念文档](https://opentelemetry.io/docs/concepts/sampling/)，2026-11-18 访问；阅读 head sampling、tail sampling 与取舍。
- **来源支持的事实**：head sampling 在 trace 开始较早决定是否保留，不能依据尚未发生的整条 trace 结果保证采样错误轨迹；tail sampling 等待观察大部分或全部 spans 后，可按错误、总延迟等条件选择 trace，但需要有状态的缓冲与处理，实施、运行和容量管理更复杂。两者可以组合，但前序 head sampling 丢弃的 trace 不会进入后续 tail decision。
- **证据边界**：source-backed（OpenTelemetry 官方概念文档，2026-11-18 访问）。采样策略、保留率和错误轨迹覆盖取决于具体 SDK、collector、backend 和负载，文档没有规定 Agent 系统的固定比例。

- **OpenTelemetry Metrics**：[官方概念文档](https://opentelemetry.io/docs/concepts/signals/metrics/)，2026-11-18 访问；阅读 metric attribute combinations 与 cardinality limit。
- **来源支持的事实**：指标的 cardinality 是不同属性组合数；OpenTelemetry Metrics SDK 为每种组合维护聚合状态，所以 user ID、原始 URL 路径等高基数维度会增加内存成本。当前文档记载 SDK 默认 cardinality limit 为 2000；超过上限时额外测量会折叠到 overflow 点，保留总量但丢失相关属性，按这些属性过滤或分组可能低计数。
- **证据边界**：source-backed（OpenTelemetry 当前概念文档，2026-11-18 访问）。2000 是该文档描述的 SDK 默认值，不代表所有 OTel 版本、后端或 Agent 指标系统的固定限制；设计指标维度时仍须核对实际 SDK 配置。
