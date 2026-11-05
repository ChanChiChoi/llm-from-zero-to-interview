# Agent 工具调用、安全确认与评测：来源证据

创建于 2026-10-30；最后更新于 2026-11-05。本文只记录公开资料支持的接口约束、授权建议和评测构造；产品接口与仓库文档为动态页面时，以访问日期和具体 API 模式限定适用范围。

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
- **来源支持的事实**：指南要求用户能识别并确认关键交易数据，授权在服务端实施，交易数据在授权期间不得被篡改，授权凭据应限时、按操作唯一且执行前在共享状态中原子检查/消费。
- **证据等级与边界**：source-backed（OWASP 开发安全指南）。将这些要求组合成 Agent 的服务端 `approval_id` 与动作摘要记录，是本项目的工程化设计推论，不是 OWASP 规定的专用 Agent 协议。

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
