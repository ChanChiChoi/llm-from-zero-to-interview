# 第二十四章精修审计

## 状态与范围

- 状态：第二十册第 24 章完成本轮章级研究、正文精修和配套同步；第二十册整册连续复读、整册门禁和全书精修 goal 尚未完成。
- 主题：以 Claude API Messages API model ID `claude-sonnet-5-5` 为例，讲清工具回合的 API 状态、宿主执行事实、refusal/fallback 路由和任务验收之间的边界。
- 目标读者：理解 Agent 工具循环和消息 API 基本结构的读者；首次需要理解的 `thinking`、`tool_use`、`tool_result` 与 refusal 字段在正文内解释。
- 读者结果：读者能复走一轮工具调用，区分对话块、外部副作用和验收证据；能按模型/前缀/账号条件判断 thinking block 是否可续接；能从 refusal 响应确认类别、fallback 和实际答复模型。
- 研究日期：2026-11-19。用户要求专注修书；没有复查代理设置，没有构建或更新 PDF，也没有修改 `plan.md`。

## 研究范围与组织参考

- 外部主题：Messages API typed blocks 与工具往返、`between_tools` 和 effort、thinking-block 回放/模型兼容/前缀/账号绑定、forced tool-choice 约束、HTTP 200 refusal、类别化默认 fallback、实际模型追踪、发布方评测与 monitorability 证据。
- 主要一手材料：Anthropic Sonnet 5.5 What's New、Thinking、Preserved thinking、Prefix checks、Prompting Claude Sonnet 5.5、Tool use、Refusals and fallback、Transparency Hub；来源卡和支持范围见[研究记录](../../research/model-update-2026-09/claude-sonnet-5.5-source-notes.md)。Monitorability 结论来自 Guan et al. 的 PMLR 2026 论文及[安全评测研究记录](../../research/ai-safety-alignment-architecture-scan-2026-11-03.md)。
- 组织参考：Diátaxis 将教程、操作指南、技术参考和解释对应到不同读者需要；这里借用其读者任务区分，用一条工具轨迹解释机制，再呈现版本条件和路由，不把框架建议当成学习效果证据。
- 证据边界：API 文档描述公开合同，不证明每个托管 provider 的实测行为。当前 Platform 前缀指南称 Sonnet 5.5 的未显式配置行为“可能 400”，官方 Cookbook 仍按 2026-08-31 账号创建时间说明默认值；正文和来源记录保留这一官方文档差异，不推广统一日期阈值。没有调用真实 endpoint、SDK、benchmark 或模型。

## 读者路径

1. 用合成工单 742 沿 `r17` 读取、补丁与测试、暂停后变为 `r19`、重新验收的过程，看到 `thinking` / `tool_use` / `tool_result` 如何接成一轮 Agent 请求，并区分 API history、宿主副作用和任务验收。
2. 解释 `between_tools` 关闭 upfront thinking 的含义、合法 effort 和显示造成的“静默”现象。
3. 将 thinking block 当作有类型的续接状态，沿 model/prefix/account 三种独立绑定判断回放边界；指出长期历史仍占上下文。
4. 区分 `tool_use` 提议与授权执行，并说明 Sonnet 5.5 的 forced tool-choice 400 条件。
5. 以 `stop_reason=refusal` 判断安全拒答，按类别识别默认 fallback，再从响应/流事件记录实际答复模型。
6. 收束到发布方评测和 monitorability 的证据边界，回到独立 verifier 与外部状态。

## 原稿内容去向

| 原稿部分 | 处理 | 去向与理由 |
| --- | --- | --- |
| 24.1 AA 配置、价格、context 与 DataCurve 缺少精确行 | 删除榜单和服务报价表述；保留准确 Messages API model ID | 本章读者结果是运行时状态合同；model–Harness 分数归第 19 章，避免把一次榜单快照当作章节定位 |
| 24.2 `between_tools` 请求差异 | 保留并重写 | 补充无工具任务、progress block、空 thinking 文本、effort 与 `max_tokens` 的区别 |
| 24.3 thinking binding 与 replay | 保留并更正 | 解释方向性模型兼容、前缀和账号三类条件；删除原稿把账号创建日期写成统一默认规则的断言，改为如实记录两份官方文档的差异 |
| 24.4 tool choice、Computer Use、Advisor、compaction、inline-tools | 保留 forced-tool-choice 行为；删除其他迁移清单 | forced tool choice 直接影响模型动作接口且会 400；Computer Use/Advisor 和 beta inline-tools 细节易时效且超出中心读者结果；compaction 回第 18 章，工具注册/权限回第 4/8/17 章 |
| 24.5 cyber safeguard 内部三阶段与类别 fallback | 保留 refusal 状态、类别化 fallback 和 actual-model 追踪；内部 classifier recipe 不放正文主线 | API 路由合同能指导宿主处理；发布方对内部控制的披露不等于内部实现完整公开，也不是 host permission |
| 24.6 System Card 与多组 benchmark 数字 | 删除细项表格；保留发布方证据边界和实际路由归因 | 详细评测条件与跨配置比较由第 19 章承担；旧来源记录仍保存原先核对的发布方数字，但本章不将其重复成主线；若未来恢复数值，应重读 System Card 原件和方法条件 |
| 24.6 CoT monitorability | 保留为一段有范围的证据说明 | 论文支持“所测任务中 CoT 通常较 action-only 有信息但不完美”，不支持内部推理忠实性、授权或副作用前拦截 |
| 24.7 面试回答框架 | 正文结尾改为回收工单主线 | 同步为 canonical 面试题 40AH 与练习 35ES，避免正文重复配套答案 |
| 第 21 章移出的 Claude Fable 5.1 thinking-state 候选 | 收束为 Sonnet 5.5 model-binding 的方向性兼容边界 | 读者所需的是 producer→consumer→provider 方向判断；没有形成独立于本章的读者结果，不另开专题 |

## 配套同步

- 更新第二十册[目录摘要](../../book-20-agent-harness-runtime/目录.md)、[书系入口说明](../../BOOK_SERIES.md)、[知识图谱](../../KNOWLEDGE_GRAPH.md)与[架构审计](architecture-audit.md)。
- 更新练习 35ES 和面试题 40AH；改正旧练习、面试答法与知识图谱中关于固定账号创建日期阈值的表述，并保留来源分歧。
- 研究事实和来源边界记入 `research/model-update-2026-09/claude-sonnet-5.5-source-notes.md`；章节范围、删改去向和进度只记在本 `progress/` 文件。

## 验收与遗留风险

- 静态验收：H2 `24.1`—`24.5` 连续；本章、目录、章节审计、架构审计和 Sonnet 5.5 研究记录共检查 68 个本地 Markdown 链接，缺失 0；练习 `35ES` 与面试题 `40AH` 各唯一；`git diff --check` 通过。工单轨迹已沿 `r17`、`r18`、`r19` 复走，明确测试回执必须对应当前快照。章节没有公式、Mermaid 或可执行代码；未做 GitHub 页面渲染、外链长期可达性或目标读者试读。
- 遗留边界：前缀 mismatch 的官方默认值仍有文档描述差异；目标 provider/account 若要上线，需要按该环境最新合同核验。未做 endpoint probe、目标读者试读和 GitHub 页面外链可达性复查。
- 本轮未运行书中代码、SDK、真实 API、自动化测试、benchmark 或 PDF 构建；未更新 PDF 或 `plan.md`。第二十册整册复读和全书目标继续 active。
