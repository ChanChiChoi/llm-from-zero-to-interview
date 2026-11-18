# 第二十册第十六章精修审计

日期：2026-11-18  
章节：[第十六章《Harness 实战坑与面试题》](../../book-20-agent-harness-runtime/chapters/16-harness实战坑与面试题.md)  
状态：章级扫描、正文重构与配套同步已完成；第二十册第 17—24 章及全套书精修 goal 继续 active。

## 章节契约与读者路径

本章承接第 1—15 章的 Runtime、工具/文件/命令、上下文、权限、Trace、Evaluation、协议集成和系统设计。它不再讲一遍系统应该包含哪些模块，而是回答：一次运行出现错误、危险、卡住、成本异常，或“测试通过但交付不合约”时，如何把用户契约、运行轨迹、真实状态和验收证据对齐，找到最早偏离并安全恢复？

读者进入本章时应能理解工具提议、策略决定、实际派发、工作区 diff、测试结果和任务完成之间的差异。离开时应能复述任务约束、按因果顺序重建事件、标注 unknown、核对副作用状态、区分执行故障与验证器漏检，并把失败保存为有边界的回归样例。

## 章级扫描与候选去向

检索日期：2026-11-18。检索问题：Agent/Coding Agent 故障怎样从可复查轨迹定位到最早偏离；重试前如何判断副作用状态；trace 缺失和评估变化分别能证明什么；怎样把失败样本转成回归。查询覆盖 OpenAI Agents SDK 的错误、timeout、replay-safety、guardrails 和 tracing；Google ADK 的日志及环境模拟；Google Agents CLI 的 debug/evaluation 流程；LangGraph fault tolerance；OpenTelemetry error recording；Google SRE incident response。来源、版本、直接阅读范围和限制见[Agent Harness 故障诊断、重试与运行证据记录](../../research/agent-harness-failure-diagnostics-evidence-2026-11-18.md)。

| 外部主题 | 读者结果 | 正文去向 | 证据与边界 |
|---|---|---|---|
| 任务要求、工具提议、执行事实和 verifier | 区分提议/派发/实际副作用/任务验收。 | 16.2—16.5 | 以标注为教学构造的版本冲突时间线演示；不代表某产品的真实轨迹。 |
| timeout、重试、幂等与丢失回执 | 区分未派发、拒绝、已提交、unknown，按状态选择下一步。 | 16.6 | OpenAI Agents SDK 与 LangGraph 各自框架的语义示例，不推广成通用合同。 |
| Guardrail 时机和日志隐私 | 判断输入/工具检查发生在哪个执行阶段，并限制敏感内容采集。 | 16.4 | OpenAI SDK 当前 guardrail 文档和 Google ADK 当前日志文档；实现与配置会变化，需按部署版本核验。 |
| Trace 缺失与异步导出 | 知道 dashboard 空白不证明执行缺席，检查 sampling/queue/flush/export/auth/retention。 | 16.4 | OpenAI Agents SDK 当前 tracing 文档的后台批量导出语义；不能据此断定所有 exporter 的表现。 |
| 失败复现与回归边界 | 保存具体失败，比较修复前后，并区分固定模拟和真实集成。 | 16.7、16.9 | Google Agents CLI 文档是特定工具的 debug loop；ADK v1.24.0 environment simulation 是版本化框架能力。 |
| 遥测错误语义和生产事故复盘 | 将中间重试错误与最终 operation 状态分开，并保存协作记录。 | 16.7 | OpenTelemetry Recording Errors 页面标注 Development；Google SRE 材料是其组织实践，不构成普遍模板。 |
| 十类运行故障的诊断入口 | 从症状找到需要核对的证据，而非记一张 gate 清单。 | 16.4 | 原稿主题与前十五章对应机制保留为诊断线索；不声称类别穷尽或“最常见”。 |

## 原稿知识覆盖与去向

| 原稿内容 | 本轮处理 |
|---|---|
| 工具权限过大 | 保留为权限/审批/真实派发故障，核对身份范围、策略决定、审批对象和副作用。 |
| 大文件和工具输出挤压上下文 | 保留为 context snapshot、来源新鲜度、压缩、截断和工具返回诊断。 |
| 文件编辑覆盖并发改动 | 作为贯穿案例，建立 h0 读取、h1 用户并发保存、过期提案应用为 h2、简单测试通过的因果链。 |
| 终端命令副作用与挂起 | 保留实际命令、cwd、脱敏环境、进程存活、timeout/cancel 和远端状态核查。 |
| 外部内容中的 Prompt Injection | 保留来源、信任标记、模型输入、工具提议、策略决定与实际效果的分层；补充 OpenAI SDK guardrail 时机的专属语义及边界。 |
| Trace 不完整 | 保留 instrumentation、sampling、队列、flush、export、认证/网络、查询和保留检查；说明 dashboard 空白不是未运行证据。 |
| Evaluation 不可比 | 保留任务、model/provider、prompt、工具、环境、verifier、重复与随机性条件；统计和归因深度交第十九章/第七册。 |
| Agent 循环和成本失控 | 保留重复语义调用、状态是否推进、预算与停止条件的检查，不构造通用成本阈值。 |
| MCP Server/工具过多 | 保留实际暴露能力、工具描述/名称冲突、筛选和上下文快照；按任务检查选择，不把 server 数量当根因。 |
| 本地与云端环境不一致 | 保留 revision、镜像、依赖、OS、cwd、非秘密配置、网络策略与测试数据对齐。 |
| 十一项 coverage 比率、聚合 G_pitfall 和 Python audit demo | 删除。覆盖率不能证明状态正确或约束已满足；改用时间线、证据分层、症状矩阵和状态恢复决策表，无新汇总 gate。 |
| 十个宽泛知识/系统设计/产品比较问答 | 删除重复问答并回链对应主讲章；新增面向并发覆盖、超时副作用、循环、注入、trace 缺失和评估差异的诊断题，以及五步作答示例。 |
| 七项跨章练习 | 统一改为一个可复核的并发编辑事故 worksheet：契约、时间线、证据边界、unknown、恢复、特殊 query 与并发回归。设计权限、Trace UI、MCP injection、evaluation task、local/cloud threat model 和系统设计答题等读者结果分别由第 8、9、14、10/19、11—13、15 章承担，本章提供故障时的证据诊断入口。 |

## 正文和呈现状态

- **reader_path：** 章节任务 → 并发版本冲突案例 → 契约/轨迹/状态/验收四类事实 → 十类症状与证据 → 最早偏离 → timeout 与副作用状态表 → 失败固化为回归 → 面试回答和练习。
- **coverage_map：** 核心结果是证据排序、最早偏离判断和安全恢复；支撑为 trace 交付、守护检查位置、固定评估与生产事故记录。旧十类故障全部保留；系统设计、产品架构、统计评估及 checkpoint 专题明确链接回/后续专章。
- **example_status：** 采用明确标记为教学构造的登录重定向任务，并用一个 query 输入显示编码前后外层参数边界怎样变化。h0 是 Agent 读取版本；h1 是并发用户保存；基于 h0 的整文件提案无版本条件地应用到 h1 后生成 h2。/reports 简单路径测试只验证自身覆盖面；用户约束、实际 diff、commit 状态和 verifier 各自有独立证据。案例不表示任何真实产品、事故频率或实测行为。
- **code_status：** 不适用。任务是故障证据诊断和面试表达，不承诺读者编写审计程序；移除原约 400 行 demo 后用时间线、表格和人工 worksheet 教授同一诊断目标。未运行书中代码。
- **formula_status：** 不适用。按更新后的模块 04 §6.2，本章的因果轨迹和状态判断用事件表/决策表即可准确表达，没有需要公式才能推广的关系；移除任意 coverage/阈值和综合 gate。
- **visual_status：** 使用时间线表、四类证据表、带具体代表事件的症状—证据矩阵和恢复状态表；这些表对应顺序、分类和分支判断。无 Mermaid/公式渲染项。复核时发现模块 02 增补讲解锚点、总纲增补“小白模式”要求，均已完整重读；据此补足 query 输入/输出、任务契约/Trace/verifier/unknown 的解释及各故障行的代表性观察对象。
- **evidence_status：** 重要框架事实在正文贴近结论引用，并限定 SDK/framework/version；完整来源卡及未知边界在 research 记录。合成轨迹与现实观察分开。
- **link_status：** 本地链接连接第 15、17、19 章及 research 记录；外部来源直达 OpenAI、Google、LangGraph、OpenTelemetry 和 Google SRE 官方页。research 记录保存 10 个来源页的访问日期、直接核对范围和证据边界。
- **sync_status：** 已将 EXERCISES.md 的 35T 改为并发编辑事故复盘；将 INTERVIEW_BANK.md 的 40R 改为“测试通过但覆盖并发编辑”的证据诊断；更新本册目录第 16 章摘要。简介不含需逐章摘要的对应项；未改 plan.md 或 PDF。
- **open_risks：** 未进行目标读者试读；未验证所有外部链接长期可达；无 Mermaid/MathJax 页面渲染项；官方 SDK 文档持续更新，涉及具体执行语义时需在后续维护中检查当前部署版本。

## 静态验收

- H2 标题 16.1—16.10 连续；无代码围栏、行内公式或可执行代码；章节、目录和审计共检查 41 个相对 Markdown 链接目标，缺失 0。
- 正文、练习 35T、面试题 40R 中搜索旧 G_pitfall、harness_pitfall_gate、failed_gates、审计 demo 和 0 依赖残留均无命中；git diff --check 通过。
- 未运行书中代码、自动化测试、模型/API/provider/产品、benchmark 或 PDF 构建；无 Mermaid/MathJax 页面渲染项；未做目标读者试读或所有外链长期可达性检查。

## 下一入口

[第二十册第十七章《Long-Running Agent 与 Checkpoint》](../../book-20-agent-harness-runtime/chapters/17-long-running-agent与checkpoint.md)。第十六章不展开 checkpoint/幂等内部机制；接续时按用户要求先重读《写作规范/总纲.md》，再重读本章适用模块及当前 progress handoff。
