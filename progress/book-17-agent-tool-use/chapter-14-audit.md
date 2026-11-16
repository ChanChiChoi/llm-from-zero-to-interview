# 第十四章《Agent Swarm 并行协作》精修审计

审阅日期：2026-11-16。状态：章节主线、公式例子、示意图、代码状态、模型/runtime 案例和来源边界已整理；§14.20 demo 静态审阅、未运行。GitHub Mermaid/MathJax 呈现、外链全量可达性和代码执行仍属于最终门禁。

## scope 与 reader_path

- **范围：** 完整复读第十四章正文（当前 800 行），对照本册简介/目录、第九章协作选择、第十章评估、第十一章安全、第十三章环境契约及相关 research/PAPERS 资料。章节职责保持为 Swarm 的任务图、artifact、隔离、合并、协调、恢复和成本，不重复第九章的协作适用性主讲。
- **读者前置：** 读者已知道多 Agent 是一种协作方式，而不是模型内部结构；第九章提供选择协作的条件和单 Agent baseline，第十、十一、十三章提供评估、安全及环境/verifier 契约。
- **本章贡献：** 读者能把并行执行设计成有任务依赖、owner、版本化 artifact、隔离 workspace、冲突处理、单一外部执行器、可恢复状态、预算和独立验收的系统，并用同条件消融证明是否有净收益。
- **阅读路径：** 先判断哪些工作可并行，再用 DAG 和角色合同固定依赖与权限；接着设计消息、artifact、共享索引和隔离 workspace；然后处理合并冲突、协调、租约、取消、幂等、副作用恢复；最后计算通信/关键路径/预算/净收益并通过 demo 检查候选证据与外部提交门禁。Kimi K2.6 与 Grok 4.20 是通用主线之后的版本化厂商案例。

## chapter_scan 与 evidence

- **日期与问题：** 2026-11-16；检查并行是否真实缩短关键路径、worker 输出怎样保持独立和可追溯、冲突与取消如何恢复、外部副作用如何防重、预算是否包含合并/验证成本，以及产品案例中外部 Agent 数与模型内部 MoE 专家是否被混称。
- **候选及去向：**

| 候选 | 判断与处理 | 证据/边界 |
|---|---|---|
| Swarm 拆分条件、固定 workflow 对照、任务 DAG | 本章核心；定义可并行数据依赖与必须串行的副作用依赖。 | 架构讲解与合成合同助手例；不主张 Swarm 一般优于单 Agent。 |
| 角色/权限契约、artifact、blackboard 和 workspace 隔离 | 本章核心；区分消息、证据、假设、候选 patch 与已验收产物。 | 设计性知识；真实 provenance 和权限须由目标系统核实。 |
| 冲突合并、coordinator、租约、取消、幂等与失败恢复 | 本章核心；明确 `unknown` 不能盲目重试，并区分 canceled、rolled back 与外部副作用状态。 | 工程机制；exactly-once 被限定为业务层幂等表达，不承诺网络 exactly-once。 |
| 通信成本、critical path、预算与 NetLift | 本章核心；将 worker 时间、协调、通信、合并、验证、浪费和风险放入统一对照。 | 公式是教学分析模型；需要一致的单位/归一化与实测输入，不是通用评分标准。 |
| Kimi K2.6 Agent Swarm | 作为产品方案例，区分 300 sub-agents/4,000 coordinated steps 与 384 routed experts/top-8，并把 tool calls/长时运行案例另列为厂商展示。 | Moonshot 官方博客、模型卡和固定 config；没有独立并行对照，不推断这些案例都使用最大 swarm。 |
| Grok 4.20 Multi-agent | 作为 provider runtime 案例，记录 beta、leader 输出、4/16 配置映射与加密的子 Agent 状态。 | xAI 官方动态文档，2026-11-16 核阅；字段只支持该 endpoint 合同，不外推一般模型能力或生产 SLO。 |
| 协议和大型 Runtime 实现 | 本章保留架构边界，深入运行时/协议设计交第二十、二十二册。 | 沿本册分工和章节链接处理，避免重复专章。 |

- **研究证据：** Kimi K2.6 与 Grok 4.20 的来源、阅读位置、revision、支持事实及发布方/动态文档边界记录于[Agent Swarm 产品与 Multi-agent runtime 来源记录](../../research/agent-swarm-runtime-evidence-2026-11.md)。原有多 Agent benchmark 和实验资料仍在[Agent 环境与多 Agent 评估资料](../../research/agent-environments-multi-agent-2026-11.md)。本审计不把进度写入 research。
- **结构决定：** 保留主干和现有编号；把 Kimi K2.6/Grok 4.20 明确为通用主线后的特定产品/runtime 案例，不将它们作为协作架构的普遍定义。正文说明“Agent Swarm”在本章前半作通用架构名、Kimi 案例使用厂商的具体描述。

## coverage_map

| 知识点 | 对读者结果的作用 | 深度与去向 |
|---|---|---|
| 单 Agent/workflow/swarm 边界、任务 DAG 与角色合同 | 能判断并行点、依赖、owner、输入 revision 和验收责任。 | 核心；用合同助手流程图和代码 swarm 流程示意。 |
| 消息、artifact、证据独立性与 workspace 隔离 | 能合并可追溯产物，避免共享状态污染和多数票误判。 | 核心；有 claim/evidence schema、来源组和隔离案例。 |
| coordinator、状态、lease、cancel、幂等与恢复 | 能处理超时、部分完成、unknown、重试和已提交副作用。 | 核心；特别区分候选、verified、blocked、cancelled 和 rollback。 |
| 通信、关键路径、预算与风险调整收益 | 能比较相同任务和预算下的质量、成本、延迟、风险和治理负担。 | 核心；公式配有成本、延迟、预算和 NetLift 的合成数值例子。 |
| 可观测性、独立 verifier 与 demo | 能追溯 artifact 到 worker、来源、版本和外部验收，并理解 demo 覆盖边界。 | 核心实作；静态代码只演示证据分组和 fail-closed gate。 |
| Kimi K2.6 与 Grok 4.20 | 能分辨 MoE 路由与外部 swarm、provider effort 与 Agent 数映射、leader 汇总与隐藏状态。 | 版本化案例；使用官方资料并明确厂商自报/beta 边界。 |

## examples、formula 与 code 状态

- **贯穿案例：** 合同任务的 contract_read、policy_read、permission_check 可并行，rule_check 等三者到位后再准备待审核工单；只有受控 executor 写入并由外部 verifier 回读，才算得到可报告结果。新增 Mermaid DAG 表示该依赖关系。
- **数值案例：**
  - 冲突优先级合成分数：在 0—1 教学尺度和给定权重下，手算 `Q(A)=0.7625`、`Q(B)=0.5925`；只用于复核排序分数的算术，不是资料质量校准。
  - 租约/幂等轨迹按同一 key 与 action hash 区分返回已有结果、拒绝 key conflict、unknown 时 hold、权限不满足时拒绝，以及状态确认且 `A(a)=1` 才执行一次。
  - 四个 worker 各 10 秒、加 dispatch/消息/合并/验证后的 wall-clock 近似 24 秒；用于说明并行省下的计算等待可能被串行协调抵消。
  - 预算合成例中 3 个 worker 各 20,000 token、合并 15,000、验证 20,000，共 95,000/100,000 token；单位一致并保留 5,000 余量。
  - NetLift 用归一化合成输入手算为 `0.01`；安全硬门仍优先于正的软收益分。
- **代码：** §14.20 线程池 demo 未运行，状态已标“静态审阅，未运行（2026-11-16）”。静态追踪输出应为 `artifact_count=3`、`source_groups=3`、`independent_support=3`、`merged_status=agree_independent`、`commit_allowed=False`。共享上游样例应为 `independent_support=0`/`agree_shared_source`；空输入为 `insufficient_evidence`；重复 task_id 和空 candidate evidence 会抛出对应异常。章节明确指出该 toy gate 在所有路径下都保持关闭，因为没有实现 verifier-to-verified 转换；这些判断不是执行结果，真实来源独立性仍需审计。
- **代码边界：** 本轮只改了 §14.20 标题、状态和说明，没有改可执行语句或断言；没有运行 demo、测试、模型或 benchmark。标准库线程池示例不覆盖持久化队列、真实租约、sandbox、并发写入或生产事务。
- **公式与图：** 14.3 Mermaid 使用引号标签和管道箭头；已静态检查源码，GitHub 实际渲染未确认。公式中变量、归一化和硬风险边界已补说明；静态搜索未发现 `\operatorname`、`\(...\)` 或尖括号下标风险，MathJax 实际呈现待验。

## link_status 与 sync_status

- 新增本地回链：第九章（协作适用性与 baseline）、第十章（指标/预算/不确定性）、第十一章（授权与注入边界）、第十三章（环境/reset/verifier）；Kimi 案例链接第二十一册第 90 章。目标文件均存在。
- §14.23 补入 Kimi K2.6 官方博客/模型卡/固定 config 与 xAI Grok 4.20 模型文档/Multi-agent 文档入口；访问日期和来源限制同步进 research。`PAPERS.md` 修正并加入 Kimi/Grok 对应 research 链接。没有发现需要新增 glossary 或练习条目的稳定新术语。
- 章节可见职责和阅读顺序未变，故简介/目录摘要无需改动；全书架构交接将在本章静态门禁后推进到第十五章。

## open_risks 与 handoff

- Demo 未执行；若后续用户要求运行或写作门禁需要动态验证，需在 Python 标准库环境按模块 04 记录命令、输出和边界，不把静态推演写成已运行。
- Mermaid/MathJax 的 GitHub 页面呈现、全部外链可达性仍待终验。xAI 文档明确标注 beta，字段可能变化；Kimi 博客中的规模/案例是厂商发布内容，不是独立对照实验。
- **下一入口：** 第十五章《Kimi K3 发布证据与长任务 Harness》，核对 Kimi K3/K3.1 Pro 版本、模型/协议状态、Harness 与 benchmark 条件，并与本章的外部 Agent 编排案例区分。第十七册后续仍需整册连续复读和最终门禁。
