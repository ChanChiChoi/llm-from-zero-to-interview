# 第二十册整册连续复读记录

开始日期：2026-11-19  
状态：进行中；第 24/24 章已复读，整册终验尚未完成。

## 复读路径与完成规则

按《目录.md》顺序复读第 1—24 章，并用《简介.md》中的系统学习、搭建、长任务、评估和 API 查询路线检查承诺是否兑现。章节正文以当前内容为准，单篇审计用于追溯修改与证据，不代替连续阅读。全部章节读完后，再终验简介/目录、跨章依赖与术语、配套入口、本地链接、代码状态和目标平台图表/公式呈现；未完成项继续记录，不因首轮精修和局部静态检查而标记整册完成。

## 已完成范围

| 阅读顺序 | 章节 | 状态 | 连续复读观察 |
|---|---|---|---|
| 1 | Harness 总览 | 已读 | 从合成登录修复任务建立“模型提议—策略决定—实际副作用—任务状态/trace—独立验收”的主线。明确 Harness 与 Runtime 是本书工作定义、产品术语可能不同；用户确认、执行隔离、trace、当前状态、工具成功和任务成功各有不同证据边界。评估章节提前提示条件固定、分母明确和不拼万能总分；第 1 章的控制循环与目录第 2 章组件/状态转移职责衔接清楚。没有发现需要改正文、简介或目录的问题。 |
| 2 | Agent Runtime 架构 | 已读 | 用明确标注的恢复分支承接第 1 章闭环，并声明它不是同一次连续运行；将 session、task、workspace、checkpoint、trace 和 evaluation record 按状态所有权及可证明事实区分。组件链从 Context Builder、Adapter、Parser/Registry/Permission 到执行回执、State Store、Trace、Error Handler 和 verifier，状态机再把等待审批、超时 unknown、取消与终态接上。Runtime 机制和 provider 合同的分工也回链第 20—24 章。没有发现会造成第 1 章定义冲突或目录职责重叠的问题。 |
| 3 | Coding Agent 工作流 | 已读并修正 | 从需求澄清、仓库探索、计划、小步修改到验证和交付，复用前两章的任务/工作区状态与副作用证据。§3.15 重述与第 1—2 章相同的教学任务，但明确是重新构造的完整轨迹，未把前章合成事件冒充连续日志。复读发现并修正三处：合成测试结果 `2 passed` 曾被误写为 `1 passed`；检查通过比例的分子/分母单位原不一致；“危险命令”和“三次失败”练习容易暗示固定黑名单/阈值，现改为依据任务、环境与诊断证据判断。未发现改变章节职责或需补外部研究的新主题。 |
| 4 | 工具系统与 Tool Registry | 已读并修正 | 沿模型提议、schema、策略/确认、执行回执和任务验收的层次，承接前三章的 action/状态/证据口径。4.9 明确为同一任务目标的独立教学分支，避免将各章示例当作同一组真实运行事件。复读发现并修正 4.15.1 的公式变量和阶段分母，零样本改报 N/A；另将 YAML 中的 `idempotency: compare_expected_revision` 改为 `revision_precondition`，避免把文件版本条件误作通用幂等保证。未发现需改动本册职责或增加外部研究的新主题。 |
| 5 | 文件系统与代码编辑 | 已读并修正 | 承接第四章工具合同，把路径解析、文件版本、patch、diff、验证和回滚接成工作区编辑闭环。复读发现 §5.19.1 的 stale-write 统计漏计了回滚冲突：账本中的过期 patch 与过期 rollback 都应进入写操作口径，拦截率由 1/1 修正为 2/2，并分别报告两类切片各 1/1；另把合成测试比例改成数学定界符。目录与章节职责不变，沿用 2026-11-18 的研究扫描。 |
| 6 | 终端执行与命令安全 | 已读 | 沿第五章显示名修改任务构造独立的假设执行分支，依次区分命令提议、策略、确认、隔离、进程回执与任务验收；对齐第五章路径/版本边界，并将输出截断与不可信结果交给第七章上下文管理。正文中的命令、退出码与测试结果都明确为合成数据。11 月 18 日的章级来源扫描仍覆盖其版本事实，本轮未发现需要改正文、简介、目录或新增研究的问题。 |
| 7 | 上下文管理与记忆压缩 | 已读并修正 | 以同一合成显示名任务串起来源选择、日志摘要、task state、记忆、过期信息与信任边界；核对 7.18.1 表格及逐步计算，预算、召回、精确率、约束保留和 stale 比率都能复算。复读补齐正文中 $4/5$、$7/7$、$0/1$ 等比例的数学呈现，并把 §7.6 的文件摘要例子改回当前任务，减少无关场景切换。11 月 18 日研究扫描仍适用，未新增外部事实。 |
| 8 | 权限模型与安全沙箱 | 已读并修正 | 沿显示名任务复核策略、确认、业务授权、沙箱和审计的不同职责。复读发现隔离覆盖率原写为 1/1，但合成账本只给出 sandbox ID，缺少环境配置证据；已补入工作区副本、非特权身份、无凭据、阻断公网和资源限制的配置核验，并注明 ID 只关联环境。顺带简释角色访问控制与属性访问控制。章级扫描仍适用。 |
| 9 | Trace、日志、回放与可观测性 | 已读并修正 | 沿权限决定、补丁版本、测试回执、Artifact 引用、拒绝和待审批事件追踪显示名修复任务；核对 trace/span/event/log/metric/artifact 和 task/session/run 的边界、三种回放条件及数据导出链路。复读将时间线、失败排查和采样示例里的 4/5、5/5、4/5、3/4、2/3 从行内代码改成 MathJax 数学表达，语义和合成计数不变。11 月 18 日研究扫描仍适用。 |
| 10 | Evaluation Harness | 已读并修正 | 核对 fixture、attempt、system configuration、experiment 的单位，确认 A—D 在相同条件下配对；E 在启动前环境预检失败，只进入环境报告，不进入 Agent 成功率。复读把 2/4、3/4、50%、75% 和环境覆盖率 4/5 改为数学表达，并将指标表裸除号改写成各自分子/分母定义。未新增研究主题。 |
| 11 | Claude Code 架构分析 | 已读并修正 | 沿第十章的独立验收证据，复读公开产品案例如何区分官方合同、单次可观察行为、分析者架构抽象与未知内部；逐项核对产品表面/执行环境、上下文与持久状态、权限与 OS 沙箱、扩展点、遥测和任务验收，并确认与第 10、12、13 章分工相接。修正合成任务文件路径、状态码及命令/配置名的代码格式，澄清权限规则优先级，补正 OpenTelemetry 的定义为可观测性框架与工具集并加官方定义来源；沿用 2026-11-18 的 Claude Code 文档扫描，另于 2026-11-19 核对 OpenTelemetry 官方定义。 |
| 12 | OpenCode 架构分析 | 已读并修正 | 固定 OpenCode v2.3.2 / `c0ad18b`，沿认证修复任务复读配置与指令、Agent/工具、权限求值、宿主执行、MCP/插件、Server/API/SDK、Session/Snapshot/compaction 和独立验收。确认与第 11 章公开产品合同案例形成证据对照，第 13 章承接横向比较。修正状态码、路径、版本/提交、API 字段/规则值与程序包名的代码格式；依据固定源码和权限文档，澄清最后匹配规则、已保存批准与显式 `deny`、插件 hook 的求值顺序。复用 2026-11-18 的版本固定扫描，没有运行产品或代码。 |
| 13 | Codex 与主流 Coding Agent 对比 | 已读并修正 | 沿第 11 章闭源公开合同和第 12 章固定版本源码案例，检查比较单位、产品 surface、Agent/harness、执行环境与 verifier，并复走同一项合成认证修复任务。确认正文把产品选择与 harness 因果研究分开，且不做产品排名。将 Aider repo map 的用途表述收窄到结构摘要与文件定位意图，明确没有本章效果测量；把“工具越多风险越大”改成需要检查的授权/隔离/审计路径；移除研究来源未承接的 OpenHands SDK 选型例，改为固定 Agent/runtime/backend/revision；统一状态码、路径与 `git push` 的代码呈现。沿用 2026-11-18 资料扫描，未运行产品或实验。 |
| 14 | MCP、A2A 与 Harness 集成 | 已读并修正 | 沿第十三章产品边界复读协议能力如何进入 Runtime 的工具、身份、权限、状态、Trace 和验收路径，并确认完整协议语义仍由第二十二册主讲。依据 A2A v1.0.1 官方规范澄清远端 Task ID 的字段命名：Task 对象和 GetTask/CancelTask 参数写作 `id`，Message/更新事件用 `taskId` 引用，并与外层 JSON-RPC 请求 `id` 区分。未改变章节职责或目录摘要；未运行 SDK/服务。 |
| 15 | Agent Harness 系统设计 | 已读并修正 | 沿第十四章协议边界复读端到端设计路径。明确本章 `Task` 是本地业务任务，与 MCP/A2A 协议任务分开；将客户端重试改为稳定幂等键映射，说明传输层请求 ID 不自动保证幂等；在架构图中显式分出单任务 verifier 与离线 Evaluation Runner，并把验收条件、实际 Artifact 和结论流转接入任务状态。部署形态、容量教学算例和跨章分工未发现其他断点；未运行代码或服务。 |
| 16 | Harness 故障诊断与面试题 | 已读并修正 | 对照第十五章设计方案复读并发编辑事故，确认过期写入是最早偏离，简单路径测试遗漏任务约束是独立 verifier 缺口；再检查 timeout/unknown、guardrail 位置、trace 导出、评估条件、循环、MCP 工具选择和环境差异的证据路径。统一教学路径、URL/query 与版本标签的代码格式，未改变事实结论；复用 2026-11-18 研究扫描，未运行 SDK/服务/代码。 |
| 17 | Long-Running Agent 与 Checkpoint | 已读并修正 | 沿 `task-42` 复走 CI 回执迟写和代码审查响应丢失两种中断，确认按稳定 job ID 查询的 CI 结果与无回执的外部创建动作需要不同恢复路径；checkpoint、Trace、workspace 和服务端事实各自能证明什么没有混淆。核对 intent/outbox、服务方幂等合同、`unknown`、lease/fencing、版本迁移、累计预算、故障注入和独立验收；按规范将 §17.8 的 token 余额关系补成 MathJax 算式。第十六章负责定位偏离，第十八章负责输入压缩，第 20/23 章负责 provider 状态合同，第 21 章负责 workspace 所有权。复用 2026-11-18 研究扫描，未改变示例假设和技术结论。 |
| 18 | Context Folding 上下文折叠 | 已读并修正 | 沿 `task-42` 核对上下文输入、transcript、checkpoint、Trace、workspace 与服务端事实的边界；确认折叠摘要保留用户约束、授权范围、revision 绑定的 CI 结果、来源入口和 `unknown` effect，同时让原始材料仍可按权限回查。对照第七章输入构造、第十七章任务恢复、第二十/二十三章 provider 合同和第二十一章 workspace 所有权，章节职责衔接清楚。按规范把 §18.6 的五组 token 预算运算写成 MathJax 算式；复用 2026-11-18 研究扫描，未改变算例数值或来源结论。 |
| 19 | Harness-Aware Evaluation | 已读并修正 | 核对同一任务配置中模型、Agent harness、环境、评测 harness 与预算的比较单位；教学矩阵能区分平均主效应和模型—harness 配对交互。任务结果、harness 履约、评测装置故障分开记录；Trace 只支持提出归因假设，仍需事件/状态证据、竞争解释和受控验证。复核论文各自的任务、配置、运行次数和成本边界，按规范将矩阵比例及百分数/百分点/倍数改成 MathJax。复用 2026-11-18 研究扫描；与第 9/10 章证据和评估、第 18 章压缩、第七册统计设计分工清楚。 |
| 20 | Responses API | 已读并修正 | 沿 Response 封套、typed output item、一次自定义工具往返、SSE 事件、三种对话续接方式、保留边界、后台处理与 usage 复读；与第十七章任务副作用恢复、第十八章上下文折叠、第十九章评估及第二十一章 workspace 分工一致。11 月 19 日重查 OpenAI 官方 Background mode 与 Data controls 后，补齐后台流仅适用于默认 `soonest` 窗口、`2h`/`8h` 不启用 streaming、断线续读与约 10 分钟临时磁盘暂存的条件。未调用 API/SDK。 |
| 21 | Persistent Workspace 与状态边界 | 已读并修正 | 沿 `task-42` 的 r17→r18→r19 检查对话、检查点、workspace、CI 和外部服务各自能证明什么；核对 Git 的 HEAD/index/working tree、派生索引新鲜度、并发编辑后的验证目标与 artifact 交接。与第 5、7—10、17—20 章的编辑、记忆、权限、评估、checkpoint 和 provider state 边界吻合。复核 Agents API 官方页面后，补充多 session 共享 environment 只适用于新建 standalone environments，旧环境可能保留单 session 生命周期。 |
| 22 | OpenAI-Compatible API 边界 | 已读并修正 | 沿调用方迁移路径区分 SDK/传输、endpoint/schema 与实际服务行为；复核 Anthropic、Gemini、DeepSeek、vLLM、Alibaba、Hugging Face 等官方文档及 API 转换研究。发现 Hugging Face 当前文档新增 Responses API (beta)，修正“其兼容端点仅面向 Chat Completion”的过宽表述，改为说明两个独立接口面及 beta/未 probe 边界。与第 20 章原生 Responses、第 23/24 章 provider 状态、第 24 册服务端实现分工清楚。 |
| 23 | Gemini Interactions 与 thought-state 回放 | 已读并修正 | 沿工单 742 的中断过程区分 Interaction 历史、SSE 事件恢复、长解码续接和宿主外部动作；复核五类 ID/token、逐轮配置、stateless thought/signature 回放、function `id`/`call_id`、Interaction status、存储期限与删除边界。重新打开当前 Google 官方 overview、API v1 reference、Thinking、Streaming、Background execution 文档及 Python SDK releases：关键字段和状态仍与正文一致；Thinking guide 仍称 thought signature 必需，而 API v1 reference 未标 required/optional，正文保留此证据边界。无需改动正文；未调用 endpoint 或运行 SDK。 |
| 24 | Claude Sonnet 5.5 thinking-state 与安全路由 | 已读并修正 | 沿工单 742 的恢复过程复核 `between_tools`、原样 block 回放、工具提议/宿主授权、refusal/fallback 路由与任务验收。当前官方兼容表纠正了“Sonnet 5.5 不读取 Fable/Mythos 系列 blocks”的过宽概括：方向和版本有差异，Claude API 表将 Fable/Mythos 5.1→Sonnet 5.5 标为 limited support，将 Sonnet 5.5→Fable/Mythos 5.2、Opus 5.5 标为可读取；跨账户和其他 provider surface 仍需分别核验。前缀默认行为的 Platform/Cookbook 文档差异保留。 |

## 累计跨章观察与待处理

- 连续阅读已确认第 1—24 章从全局职责、Runtime 状态、工程工作流到工具合同、文件编辑、进程执行、上下文选择与折叠、权限/隔离、运行证据、任务评估、产品案例、协议集成、系统设计、故障诊断、长任务恢复、harness 评估、provider API 合同、持久 workspace 状态归属、API 兼容边界、Gemini Interaction 状态和 Claude thinking-state 路由逐层推进；第 2—24 章使用的恢复/调用例子均标清为独立教学分支。
- 第 1 章伪代码明确标为非执行；第 3—24 章的案例状态均为教学构造。本轮未运行任何书中代码或测试。
- 目标平台公式/图表呈现和外链全量可达性仍按本册架构审计的风险记录处理，不能由本地复读替代。

## 下一入口

《简介.md》《目录.md》、跨章路径、代码状态和本地链接已完成整册静态复核；目标平台公式/图表渲染、历史外链可达性与共享索引的失效链接列入全书最终门禁。下一入口为第二十一册架构审视；这些未关闭项在对应书册处理或全书终验时回收。第 24 章连续复读发现的 thinking-block 兼容表修正见[第二十四章审计](chapter-24-audit.md)。不更新 PDF 或 `plan.md`。
