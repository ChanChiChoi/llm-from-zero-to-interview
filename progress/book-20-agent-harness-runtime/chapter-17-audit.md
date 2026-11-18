# 第二十册第十七章精修审计

日期：2026-11-18  
章节：[第十七章《Long-Running Agent 与 Checkpoint》](../../book-20-agent-harness-runtime/chapters/17-long-running-agent与checkpoint.md)  
状态：章级扫描、正文精修、目录和配套同步完成；第二十册第 18—24 章及全套书精修 goal 继续 active。

## 章节契约与读者路径

本章承接第七章的上下文与任务状态、第九章的 Trace 证据、第十五章的运行时设计和第十六章的故障诊断。它进一步回答：worker 中断后，Runtime 凭哪些持久记录和外部事实判断任务从哪里继续；某项副作用可能已发生但回执丢失时，什么条件允许查询或重试，什么情况下必须保留 unknown 或交人工。

读者进入时应能区分工具提议、执行回执、工作区和任务验收。读完后应能拆分对话历史、任务 checkpoint、Trace、workspace 与服务端状态；为 effect 持久化 intent 和稳定操作身份；按服务合同处理幂等、unknown 和在途请求；用 lease/fencing、版本、预算和独立 verifier 设计恢复路径。

## 章级扫描与候选去向

检索日期：2026-11-18。检索问题：进程、连接或执行阶段中断后，什么可以重放；本地无回执时怎样判断外部动作是否已发生；框架 checkpoint/retry 到底保存和重跑什么；如何验证恢复没有重复副作用。英文检索词包括 `durable execution workflow replay activity retry idempotency`、`LangGraph persistence checkpoint durability fault tolerance`、`transactional outbox duplicate delivery`、`HTTP idempotent method`、`agent checkpoint resume conformance`、`state-verified tool fault recovery evaluation` 和 `fencing token expired lease stale writer`。领域资料与组织样本的来源卡、支持范围和未验证边界见[checkpoint 与恢复研究记录](../../research/long-running-agent-checkpoint-recovery-evidence-2026-11-18.md)。

| 外部主题 | 读者结果 | 正文去向与判断 |
|---|---|---|
| Task、对话历史、Trace、checkpoint 与外部现实状态 | 说明每种记录回答什么、不能证明什么。 | 17.1—17.3、17.11；用同一 `task-42` 贯穿，不把 checkpoint 当聊天快照或远端回滚。 |
| 远端副作用、持久 intent、outbox 和服务方幂等合同 | 确定何时写 intent、何时派发、回执丢失后怎样核查。 | 17.4—17.5；使用 RFC 9110、AWS outbox 和特定 DynamoDB token 作为有范围的例证，不推广成统一 API 合同。 |
| Worker 所有权、租约失效与 fencing | 解释为何租约过期不等于旧 worker 停止，以及由谁拒绝旧世代写入。 | 17.3、17.6、17.9；依据 Chubby 与 etcd 原始材料说明 fencing 必须在接受写入/副作用的边界核验；第三方不接 token 时，在途副作用仍需查询或对账。 |
| Workflow replay、Activity/node retry 和 checkpoint durability | 按框架合同核对持久化单位、重放位置和副作用重试。 | 17.7、17.10；Temporal/LangGraph 事实限定到其文档、SDK、版本、后端和配置，不假定 checkpoint 自动提供 exactly-once。 |
| Schema、工具、策略、workspace 与预算版本迁移 | 判断哪些回执或产物可复用、哪些必须重新验收。 | 17.3、17.7—17.8；不把反序列化成功视为语义兼容，预算用明确教学数值演示恢复时延续累计消耗。 |
| 故障注入与恢复评估 | 检查中断窗口、外部最终状态、unknown 和恢复耗时，避免删去无法确认的样本。 | 17.9；AgentFaultBench 只作为单作者预印本中的测试设计样本，不作为生产表现证明。 |
| checkpoint conformance 与教学组织 | 把“支持 resume”拆成可测试合同，并从故障事实连到恢复语义。 | 17.10；Resume Means Resume 是单作者未同行评审预印本；MIT 6.5840 为研究生系统课样本，只借鉴结构，不推断普遍教学效果。 |

本轮没有发现需要增设独立章节的新主题。Context Folding、workspace 所有权和 provider conversation state 各由第十八、二十一、二十/二十三章主讲；本章只说明它们与任务恢复的边界。

## 原稿知识覆盖与去向

原稿含 24 个小节，包含重要知识，但多次重复状态机、checkpoint 字段、错误处理和恢复结论；旧长篇 audit demo 与任意综合指标又把读者注意力从“外部动作可能已发生但没有回执”移开。本轮重组为 12 节，按一个中断案例、状态证据、写入协议、恢复边界、版本/预算、故障验证和总结排序。

| 原稿知识 | 本轮处理 |
|---|---|
| checkpoint 字段、父版本、游标与持久时点 | 保留于 17.3—17.4，以任务身份、契约、状态游标、Artifact、effect ledger、版本、预算、owner 和完整性来源说明恢复作用。 |
| 时间线、状态机、intent/commit 与幂等 | 合并进 17.1、17.4—17.5；明确 `not_dispatched`、`intent_saved`、`unknown`、`committed`、`confirmed_failed` 的证据前提。 |
| 跨日续跑和任务进度 | 由 `task-42` 全章贯穿；以阶段游标、job ID、workspace revision 和待决 effect 表示进度，不用生成 token 数代替任务状态。 |
| workflow/schema、工具/模型/策略版本及租约 | 保留于 17.3、17.6—17.7；补入 lease 到期与旧 worker 停止不是一回事，以及 fencing 接收端约束。 |
| 预算、频率、资源保留与取消 | 保留于 17.8；用合成预算演示恢复不重置已用量，说明取消不能撤回已派发远端操作，并把 workspace 保留/租户范围交第 21 章。 |
| 故障注入、恢复评估、SLO/指标分母 | 保留于 17.9，以故障窗口表及计划中断数、事实已知样本、unknown、重复副作用和恢复耗时等分项观察表达；不给出通用数值目标或总分。 |
| 任意公式 `K_t`、`S`、`G_resume`、`D_duplicate`、`P_task`，人为阈值/coverage gate 与约 400 行旧 audit demo | 删除；用具体操作状态、时序、失败证据和表格解释，不以另一组公式或通用 gate 替换。 |

## 正文结构、例子与章节交接

- **reader_path：** `ci-884` 已完成但任务记录未更新，以及 `review-481` 已创建但回执丢失 → 区分四类记录 → 设计 checkpoint → 持久 intent 与 effect ID → 根据状态核查或重试 → 用 fencing 排除过期 worker → 校验版本、预算和恢复证据 → 注入中断并检查真实外部状态 → 回到任务验收。
- **example_status：** `task-42`、`r41/r42/r43`、`ci-884`、`review-481` 均明确标作教学构造。说明 CI job ID 可查是本例给定合同；代码托管服务能否按操作身份查询或幂等重放需由实际合同决定。叙述者知道 review 创建成功而 Runtime 不知道，以此表现知识边界。另用 checkpoint 序号 18 与租约代数 7/8 区分两个计数，并以 100,000/72,000/28,000 个 token 演示预算延续。
- **章节分工：** 第十五章负责系统级状态所有权和恢复设计；第十六章负责从故障轨迹定位最早偏离；本章深入 checkpoint/effect 恢复。第十八章主讲上下文折叠与来源保真；第二十、二十三章主讲 Responses API/Gemini 状态合同；第二十一章主讲 workspace、revision 和外部状态所有权。本章只链接并说明前后关系，不重复主讲。
- **code_status：** 不适用。章节结果是设计和核查恢复协议，不承诺实现可运行程序；未提供代码、未运行 Agent、框架或服务恢复 probe。
- **formula_status：** 不适用。状态、执行顺序和故障窗口用表格与文字可准确表达；删除任意合并公式和 gate，没有以新公式替代。
- **visual_status：** 采用中断对照表、checkpoint 字段表、intent/dispatch 时序表、effect 状态表、版本迁移表和 fault-injection 表。内容分别承担分类、状态变化、步骤顺序和边界比较；无需 Mermaid、数学图或数据图。
- **evidence_status：** Temporal、LangGraph、AWS、IETF、DynamoDB、Google Chubby、etcd 与 MIT 原始/官方资料支撑相应的特定合同或组织案例；RFC/服务行为、框架语义和本书工程建议分开表述。AgentFaultBench、Resume Means Resume 明确为单作者 arXiv 预印本，未声称独立复现或生产普遍性。证据明细在 research 文件。
- **sync_status：** 更新本册[目录第十七章摘要](../../book-20-agent-harness-runtime/目录.md)；新增练习 35EL 和面试题 40AA。简介无需变更；未改 `plan.md`、PDF、术语表或知识图谱。

## 静态验收与遗留风险

- H2 标题 `17.1`—`17.12` 连续；章节没有代码围栏、行内/块级公式或可执行代码；旧任意公式、综合 gate 与 audit demo 标记无残留。目录摘要和配套题与当前读者结果一致。
- 静态链接检查：章节 34 个、目录 31 个、research 18 个、审计 4 个 Markdown 链接的本地目标均存在；练习 35EL、全书交接和架构地图新增的相对链接目标也已核对，无缺失。`git diff --check` 通过。
- 未核验所有外链长期可达性；滚动更新的 Temporal/LangGraph 页面及实际部署 API 合同需要按使用版本再核对。没有真实服务的故障注入、框架恢复 probe 或目标读者试读；预印本结论尚需独立复现。无 GitHub 图表/公式渲染项。
- 未运行书中代码、自动化测试、模型/API、benchmark 或 PDF 构建；未更新 PDF 或 `plan.md`。

## 下一入口

[第二十册第十八章《Context Folding 上下文折叠》](../../book-20-agent-harness-runtime/chapters/18-context-folding上下文折叠.md)。继续区分压缩后的模型上下文、可追溯来源和本章已经定义的持久任务状态，并按其研究记录和章节契约开展精修。
