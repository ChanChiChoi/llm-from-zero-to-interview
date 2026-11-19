# 第八章《权限模型与安全沙箱》精修审计

审计日期：2026-11-18。状态：完成章级扫描、正文重写、配套同步和静态文档检查；第二十册第 9—24 章待逐章处理。

## Reader path 与章节结果

- **读者前置：** 理解 Agent 会提出工具调用，且工具执行能产生文件、进程和远端副作用；了解基本文件、Shell、Git 与工具调用。
- **书册贡献：** 在工具、文件、终端、上下文之后，把权限决定、人工确认、业务授权、OS 隔离与审计放到同一条执行路径中，明确它们保护的对象和证据边界。
- **中心问题：** 模型提出动作后，哪些可信检查决定它能否接触真实资源；获准代码运行时能访问什么；执行结束后如何证明发生了什么？
- **预期结果：** 读者能构造主体/任务/动作/资源/环境/副作用组成的权限请求，解释 allow/ask/deny 与下游授权的分工，设计绑定具体动作的确认，逐条检查文件、Shell、网络和外部工具入口，说明沙箱与容器的控制边界，并用明确分母审阅拒绝、确认和隔离是否落实。
- **主要任务与讲解路径：** 系统/工程机制解释。贯穿一项明确标为合成的显示名修复任务，沿源码读取、补丁写入、局部测试、Secret 访问、外发请求和 Git push 追踪决策、边界与实际副作用。

## 章级扫描与候选去向

扫描日期：2026-11-18。扫描问题：如何让模型的动作提议经过可信运行时和下游服务授权；用户确认怎样绑定将要执行的具体动作；OS 沙箱、容器、网络/凭据限制覆盖哪些执行路径；如何防止外部文本改变权限来源；哪些审计证据能证明控制实际生效？资料类型：OWASP LLM06:2025 与 Transaction Authorization / Logging Cheat Sheet、Claude Code permissions/sandbox 文档、OpenAI Agents API sandbox security 文档、Docker Engine security 文档，以及 OWASP Path Traversal、SSRF 指南。

| 候选主题 | 证据与适配性 | 去向与判断 |
| --- | --- | --- |
| 权限决策与 prompt 的差异 | OWASP Excessive Agency 强调缩减功能/权限并由下游授权；产品文档显示具体 permission engine 在模型之外 | 保留 8.2—8.5，按可信主体、任务、动作、资源、条件说明；产品行为仅作有版本边界的例子 |
| 人工确认与权限/业务授权的区别 | OWASP 交易授权要求用户看见关键数据，服务端按顺序核验并在交易变化时重新授权 | 保留 8.6；将 Agent approval ID、短时和单次使用明确标为工程推论 |
| 文件、命令、网络、Secret、Git 和外部工具 | 与第 4—7 章相接；同一资源可能有多个工具入口，命令字符串和提示标签不能代替运行时强制 | 保留 8.7—9、8.12—13；引用第五、六章细节，不重复完整命令和路径教程 |
| 沙箱、容器和资源边界 | 产品文档限定各自具体工具覆盖范围；Docker 文档区分 namespace/cgroup、daemon 和挂载风险 | 重组为 8.10—12；说明控制对象、配置缺口、执行路径覆盖与资源限制，不把容器或沙箱 ID说成已隔离证据 |
| 审计、日志隐私与指标 | OWASP logging 指南说明日志可能含敏感信息；原稿 8.19 的比例定义与 Python auditor 把主观标签和固定门槛混成统一 gate | 保留 8.14 和 8.16；改为动作/审批/环境/结果的可追溯字段，以及四项独立诊断比例、明确分母与 N/A |
| UX、企业策略、面试与练习 | 原稿的分级、组织治理和面试主题仍服务读者；旧练习要求复刻长 auditor | 保留 8.15、8.17—18，练习转为合成事件轨迹分析和手算 |

## 覆盖图与原稿主题去向

| 原稿主题 | 本轮处理与正文位置 | 去向理由 |
| --- | --- | --- |
| 权限为什么重要、权限维度、最小权限、策略矩阵 | 8.1—5 以同一任务说明动作请求和分资源策略 | 把资源与动作清单转成读者能跟踪的权限决定 |
| 用户确认、dry run、不可逆动作 | 8.6 展示确认信息，并说明审批须与目标、版本和实际动作一致 | 保留旧确认 UX 内容，增加交易授权来源支持与 Agent 工程推论边界 |
| 文件、命令、网络、Secret、Git 权限 | 8.7—9 承接第 5、6 章，补齐执行入口一致性、下游身份和凭据代理边界 | 避免将每类工具分开列表后误以为每条路径天然受同一策略控制 |
| 沙箱形态与限制 | 8.10—12 重组文件、网络、身份、凭据、资源、生命周期，并检查容器 daemon/宿主挂载和沙箱外工具 | 解释运行时到底限制什么，不将容器等同于完整威胁模型 |
| Prompt Injection 与越权 | 8.13 以 README 请求外发为贯穿案例，区分来源标签和实际执行控制 | 说明模型可能受诱导，但授权来源与资源边界由可信组件决定 |
| Trace、企业权限和真实坑 | 8.14—15 整合为审计字段、日志隐私、策略层级及确认体验 | 保留实务责任与治理内容，避免空泛经验清单 |
| 原 8.19 指标与约 350 行 Python auditor | 删除固定阈值、G_perm 聚合判断和程序输出；8.16 改为六项合成权限轨迹和四个独立比例的手算 | 旧 demo 对模拟字段而非真实执行边界作结论，通用 gate 缺乏适用范围和证据支撑 |
| 原 8.20—8.23 面试题与练习 | 8.17—18 保留三个核心面试回答并重写 35L 练习 | 将实现审计脚本改为读者能检查权限决策、实际副作用和边界的事件审阅 |

## Evidence、示例与呈现状态

- **evidence：** 2026-11-18 打开 OWASP [LLM06:2025 Excessive Agency](https://genai.owasp.org/llmrisk/llm062025-excessive-agency/?lang=us)、[Transaction Authorization](https://cheatsheetseries.owasp.org/cheatsheets/Transaction_Authorization_Cheat_Sheet.html)、[Logging](https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html)；Anthropic [Claude Code permissions](https://code.claude.com/docs/en/permissions) 与 [sandbox](https://code.claude.com/docs/en/sandboxing)；OpenAI [Agents API Sandbox security](https://developers.openai.com/api/docs/guides/agents-api/environments/security)；Docker [Engine security](https://docs.docker.com/engine/security/)；OWASP [Path Traversal](https://cheatsheetseries.owasp.org/cheatsheets/Path_Traversal_Prevention_Cheat_Sheet.html) 与 [SSRF](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html)。来源事实、日期、适用范围和 Agent 审批绑定的工程推论写入[Agent 工具调用与安全确认来源记录](../../research/agent-tool-calling-security-2026.md)。
- **example_status：** 8.1—8.16 复用一项合成显示名修复任务。8.16 六项请求的结果、rev-17 和 $4/5$ 测试回执均为教学构造，非真实运行；四项比例分别为策略覆盖 $6/6$、拒绝后无副作用 $2/3$、已执行审批动作精确匹配 $1/1$、必须隔离的执行覆盖 $1/1$。连续复读时补入测试执行环境的配置核验证据，使 $1/1$ 的分子有明确依据。每项只支持当前样例的诊断，不形成通用安全分数。
- **code_status：** 删除旧 Python 权限沙箱 auditor；正文没有可执行代码，也未运行命令、测试、模型、API 或 benchmark。
- **formula_status：** 保留四个行内比例和必要手算，分子/分母及零分母 N/A 均有解释；没有固定阈值或聚合 gate。未核验 GitHub MathJax 实际渲染。
- **visual_status：** 用权限矩阵、控制面表、访问路径表和合成事件账本表达关系；未新增 Mermaid 或图片。
- **link_status：** 正文 9 个唯一外部官方来源与第 5、6、9 章三个本地章节入口经静态目标扫描；未做全站外链可达性检查。
- **sync_status：** 重写 EXERCISES.md 35L 为无代码的合成权限事件审阅与手算；重写 INTERVIEW_BANK.md 40J，移除旧指标与 permission sandbox gate。第二十册目录摘要已有工具可见性、权限、用户确认、业务授权和环境隔离内容，仍准确；未更新简介、目录、plan.md 或 PDF。第二十册架构审计已记录第 1—8 章完成。

## 遗留风险与下一入口

- 2026-11-19 连续复读补充区分角色访问控制与属性访问控制，并为合成账本中的隔离覆盖率增加环境配置核验事实；sandbox ID 仍只作环境关联标识，不单独作为隔离证据。没有新增外部事实，11 月 18 日来源扫描仍适用。
- 没有目标读者试读；公式页面渲染和所有外链可达性未核验。
- 本章引用的 Claude Code、OpenAI Agents API 和 Docker 产品文档绑定 2026-11-18 访问时的公开合同；具体平台、工具覆盖、默认值和部署配置后续需要按目标环境复核。
- 合成账本用于展示定义明确的诊断口径；生产安全审计仍须使用实际执行证据、覆盖所有工具路径与主体，并检查环境与下游状态。
- 下一入口为[第九章《Trace、日志、回放与可观测性》](../../book-20-agent-harness-runtime/chapters/09-trace日志回放与可观测性.md)，将权限策略、确认事件、沙箱身份与执行回执接入完整运行轨迹。
