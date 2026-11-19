# 第九章《Trace、日志、回放与可观测性》精修审计

审计日期：2026-11-18。状态：完成章级扫描、正文重构、题库/练习同步和静态文档检查；第二十册第 10—24 章待逐章处理。

## Reader path 与章节结果

- **读者前置：** 理解一次 Agent 运行会经过上下文构建、模型调用、工具执行、权限判断、文件/命令副作用和任务验收；会读常见日志与 Git revision。第 1—8 章已建立 runtime、工具、文件、命令、上下文和权限边界。
- **书册贡献：** 将前文分散的模型提议、策略决定、工具副作用、状态变化与验收回执连成可追查的证据关系，为第十章 Evaluation Harness 和后续 checkpoint、context folding、harness-aware evaluation 提供运行观测入口。
- **中心问题：** 怎样用足以解释一次 Agent run 的最小记录定位行为与失败，同时避免把敏感内容不加区分地写入 telemetry；回放能复现哪些对象、还缺哪些条件？
- **预期结果：** 读者能区分 trace/span/event/log/metric/artifact，定义 trace/session/task/run/span ID 的作用域，沿证据定位首个失败点，说明三种回放模式的前提与副作用边界，并设计按用途采集、脱敏、采样、保留和审阅指标的方案。
- **主要任务与讲解路径：** 系统/工程机制解释。用一项明确标作教学构造的显示名修复任务追踪模型动作提议、权限结果、补丁 revision、测试失败/修正/通过、.env 读取拒绝和未获批的 push 请求；该任务不是本地真实执行。

## 章级扫描与候选去向

扫描日期：2026-11-18。问题：怎样将运行、子操作、状态变化和工具副作用关联为可检索证据？Trace 与业务任务、session、回放分别是什么关系？哪些模型/工具内容默认不应进入遥测？SDK 处理器与 exporter 组合是否会让脱敏失败后原始数据继续外发？head/tail sampling 怎样影响错误 trace 的可见性？metric cardinality 有何成本？资料类型为 OpenTelemetry 概念文档与语义约定、W3C Recommendation、OpenAI Agents SDK 官方文档及 OWASP 日志指南。

| 候选主题 | 本章处理 | 判断 |
| --- | --- | --- |
| Trace、Span、Event、Log、Metric、Artifact 职责 | 9.2 以表格对齐信号对象，再说明父子 spans、事件、link 和 Artifact 权限 | 核心基础，保留并用同一任务解释；不将事件误称为 trace 的唯一组成，也不把 Artifact 塞回 span 正文 |
| Trace / session / task / run ID 混用 | 9.3 说明样例范围；9.6 将 ID、尝试、版本与权限记录放进字段设计 | 保留并限定为应用定义；不同运行可关联同一任务，关联 ID 不是认证或授权凭证 |
| 记录全部上下文和模型/工具输出 | 9.6、9.11—12 改为最小低敏元数据、按需 opt in、外置 Artifact、导出前 fail-closed | 删除原“完整 trace 一律应记录所有输入输出”倾向；完整内容可能超出访问、隐私、存储与合规边界 |
| Replay 及确定性 | 9.8—9.9 区分只读查看、固定回执模拟和隔离重跑，重列输入、依赖、权限、模型、网络与副作用条件 | 核心主题保留；历史审批不复用，真实写操作不能由 trace 直接自动重放 |
| Debug workflow、恢复、错误归因 | 9.7 沿测试失败输出截断、Artifact、第二次模型输入、rev-18→rev-19、验收回执追踪第一处失败 | 保留；将观测事实与调查者 root-cause 判断分开，未知副作用状态不强行归因 |
| 隐私脱敏、保留、可观测性 UI | 9.10—13 说明数据分层、访问隔离、处理顺序、trace context 与 UI 内容权限 | 保留并更新；废止“默认脱敏即安全”的无条件表述，SDK 默认与 exporter 行为按具体版本界定 |
| 可观测性指标与 Replay gate | 9.14、9.16 只给明确定义的示例分子/分母和用途矩阵 | 删除 12 项合并审计和 G_trace；统计信号与任务验收互不汇总，不设通用发布阈值 |
| Sampling、指标基数 | 9.14 说明 head/tail sampling 的可见性/运行成本差异，以及高基数标签成本 | 新增章级主题；不规定固定采样率和跨实现 cardinality 上限 |
| Trace 和 Evaluation Harness | 9.15 说明 trace 是运行观察材料，不是验证器、任务集或成功定义 | 保留边界；系统性 eval 过程交第十章，不重复统计方法教材 |
| 面试与练习 | 9.17—18 回答三道核心设计问题，练习 35M/40K 改为事件审阅、独立分母、回放边界和导出失败审计 | 保留学习入口，删除长 Python auditor 和累积指标清单 |

## 原稿重要知识点的去向

| 原稿知识点 | 新稿位置 | 处理理由 |
| --- | --- | --- |
| Agent 多步系统、最终答案不足以复盘 | 9.1—3 | 用具体修复任务说明最终文本和执行事实不同 |
| ID、版本、任务状态、模型和工具字段 | 9.3、9.6 | 保留有排障/恢复用途的字段，区分 scope 与敏感内容 |
| Trace schema / step 粒度 | 9.4、9.6 | 从模板字段清单改成 span/event 职责与证据目的 |
| Replay 定义、难点、环境和非确定性 | 9.8—9.9 | 分三种模式讲，分别限定真实副作用和可以声称的复现范围 |
| 分层 debug 流程 | 9.7 | 用同一时间线追首个偏离、失败回执与验收 |
| Task/model/tool/security/quality 指标 | 9.14、9.16 | 保留信号类别，缩减为 3 个手算分项、明确分母、N/A/unknown |
| 日志过少/过多、UI、真实坑 | 9.10、9.14 | 归入数据保留、采样、使用者权限和局限解释，不重复枚举口号 |
| 原 9.16 的 12 项比例、G_trace 和约 500 行 Python auditor | 删除，意图迁到 9.14、9.16 与练习 | 分母与适用对象混杂，指标全部通过硬门禁缺少证据依据；程序审计合成字段而非验证真实导出/副作用 |
| 原面试题、8 项章末练习 | 9.17—18 | 题目聚焦实际设计判断和事实边界 |

## Evidence、示例与呈现状态

- **evidence：** 2026-11-18 打开 OpenTelemetry [Traces](https://opentelemetry.io/docs/concepts/signals/traces/)、[Metrics](https://opentelemetry.io/docs/concepts/signals/metrics/)、[Sampling](https://opentelemetry.io/docs/concepts/sampling/) 文档；W3C [Trace Context Recommendation](https://www.w3.org/TR/trace-context/)；OpenAI Agents SDK [Tracing](https://openai.github.io/openai-agents-python/tracing/)；OpenTelemetry GenAI [span conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md)（状态 Development）。OpenTelemetry 当前旧文档已提示 GenAI 约定迁至独立 repo；事实限定为访问时 main 草案。隐私日志边界另参 OWASP [Logging Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html)。新增来源事实及版本/证据边界见[Agent 工具调用与安全研究记录](../../research/agent-tool-calling-security-2026.md)。
- **evidence boundaries：** OpenAI SDK 的 `trace_include_sensitive_data` 当前默认值、`add_trace_processor()` 保留默认 exporter、单 processor 异常不阻断后续处理器，均为动态官方文档对该 SDK 的描述；未运行本地 SDK，需按 deployment lockfile 核验。GenAI semantic conventions 为 Development 草案；不代表所有 instrumentation 实际关闭内容采集。OpenTelemetry metrics 默认 cardinality limit 2000 只记入 research note，正文仅陈述其成本机制，不把此版本默认值外推为全系统约束。
- **example_status：** 9.1—9.9 使用明确标注的合成显示名修复 trace，包含 rev-17→rev-18 的首轮失败、输出截断、通过受控 Artifact 继续诊断、rev-18→rev-19 的修正和 $5/5$ 局部测试通过；另含 `.env` 在执行器派发前被拒、push 待审批未执行。数据均为教学构造，不是本地实际执行证据。
- **code_status：** 移除旧 Python auditor；章节中无可执行代码，不运行代码、测试、模型、API 或 benchmark。
- **formula_status：** 删去原聚合公式和固定 Gate；保留 $4/5$、$5/5$ 作为合成回执，并展示三个按不同对象和分母定义的简单比例，无统一阈值。未核验 GitHub MathJax 页面渲染。
- **visual_status：** 使用 signal 对照表、事件时间线、状态层次和 replay 模式表帮助读者观察关系；无需新增 Mermaid 或图片。
- **link_status：** 已静态核对第八、十、十七至十九章的本地链接目标；外部来源页面在本轮打开或在同日研究记录复核。未完成全站外链可达性检查。
- **sync_status：** 重写[练习 35M](../../EXERCISES.md)为合成 trace 审阅、ID/版本、三类 replay、独立比例和 fail-closed 检查；更新[面试题 40K](../../INTERVIEW_BANK.md)，移除 12 项指标和 trace replay gate。第二十册目录摘要仍准确；未改简介、目录、plan.md 或 PDF。

## 遗留风险与下一入口

- 2026-11-19 连续复读将时间线、失败排查路径和采样示例里的比例从行内代码改为 MathJax 数学表达；本轮未改变合成计数、运行结论或来源依据。
- 未核验部署环境中具体 OpenAI Agents SDK、OTel SDK/Collector/exporter 的锁定版本与配置；正文将动态默认值写成当前文档事实而不是本地行为证明。
- 没有目标读者试读；GitHub 外链可达性、页面图表/公式渲染及 trace UI 目标产品验证未完成。
- 合成示例只展示可审阅的证据关系；生产系统仍需核对实际执行器回执、外部状态、权限与 Artifact ACL，不能将 trace 内容完整度直接作为副作用不存在或任务成功证明。
- **下一入口：** [第十章《Evaluation Harness》](../../book-20-agent-harness-runtime/chapters/10-evaluation-harness.md)。本章提交 trace 和 Artifact 证据边界，下章转向任务集、环境重置、验收器与结果比较。
