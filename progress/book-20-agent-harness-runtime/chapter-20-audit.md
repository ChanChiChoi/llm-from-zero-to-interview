# 第二十册第二十章精修审计

日期：2026-11-18  
章节：[第二十章《Responses API》](../../book-20-agent-harness-runtime/chapters/20-responses-api.md)  
状态：章级扫描、正文重写、目录与配套同步完成；第二十册第二十一至二十四章及全套书精修 goal 继续 active。

## 章节契约与边界

本章是第二十册 Provider/API 查询路径中的 OpenAI 专题。读者从已了解模型调用、JSON 和基本工具执行出发，读完应能解析 Response 外层状态与 output items、复走一次自定义 function call 往返、辨认 SSE 与后台响应的不同终态，并按需要选择对话续接和保留方式。

本章只说明 OpenAI 官方公开 Responses API 合同，不把其中字段、保留期、事件名称和模型资格推广成跨 provider 规则。应用任务状态、幂等恢复和外部副作用由第十七章负责；上下文折叠见第十八章；持久 workspace、artifact 与外部事实由第二十一章负责；跨 provider 兼容性由第二十二章负责。Response 的 completed 不等于工具完成或业务任务验收通过。

## 章级外部扫描与教学组织

研究日期：2026-11-18。检索范围、检索式、直接打开的官方页面、主题事实和适用边界见[OpenAI Responses API 当前合同记录](../../research/openai-responses-api-contract-evidence-2026-11-18.md)。使用 Web Search 后直接读取 OpenAI 官方 API 文档与 API reference 原文；API 事实只使用 OpenAI 官方域名资料。

| 外部主题 | 官方资料支持的内容 | 本章安排 |
|---|---|---|
| Response envelope 与 typed output | Create reference 列出 response status、output item union、usage 和 request fields；output 不保证只有一个文本 message。 | 20.2—20.3 先区分外层状态和 item 类型，纠正直接取首个文本字段的旧式假设。 |
| Custom function call 往返 | Function Calling 指南说明 function_call、call_id、应用执行、function_call_output、并行调用与增量参数。 | 20.4 用一次 read_file 教学构造走完整调用回路；执行权限和副作用仍归宿主。 |
| SSE 事件 | 官方 streaming guide 与 event reference 定义语义化事件及序号。 | 20.5 解释 item 生命周期和参数 delta；不把 sequence_number 当成应用执行回执。 |
| Conversation state | 官方指南区分手工 replay、previous_response_id 和 Conversations API；顶层 instructions 的续接与 token 计量各有合同。 | 20.6 对照三种路线，并与任务 checkpoint / workspace 分开。 |
| Retention 与数据控制 | 官方文档说明 Response 一般 30 天保留、store、Conversation 与 ZDR/MAM/媒体/后台处理的不同边界。 | 20.7 说明 store 字段不能概括所有处理和第三方路径；保留动态边界，不提供合规结论。 |
| Background、取消与续读 | 官方 background guide 和 cancel reference 支持后台状态查询、webhook、background-only cancel 及后台 stream 的 starting_after 恢复。 | 20.8 收窄到明确支持范围，不推广为所有同步流或工具副作用的取消/恢复合同。 |
| Structured output、usage 与迁移 | 官方指南说明 text.format / JSON Schema、output 读取与 Chat Completions 的迁移差异和 usage 字段。 | 20.3、20.9—20.10 交代格式边界、动态价格和迁移测试维度，不把 usage 当作全系统账单。 |

教学组织复核读取 CMU 10601 LLM tool-use 课件、University of Mannheim IE685 课件版本 2026-02-27 和 CMU Eberly Center Learning Objectives 指南。它们支持用一次可追踪的“模型提议—宿主执行—结果回传”过程组织章节，并把读者结果写成可观察动作；这些课程/指南是组织样本，不是普遍学习效果证据。详细来源及其边界见 research 记录。

没有发现需要新增独立章节的主题。本章作为 OpenAI API 案例独立交付“读懂一份 Response、一次工具往返和状态续接合同”的能力；通用 Runtime 和跨 provider API 兼容已在邻章/第二十二册有主讲位置。

## 原稿知识点去向

| 原稿范围 | 主要内容 | 处理 |
|---|---|---|
| 20.1—20.4 | message 与 item、多模态动机、工具调用、代码 Agent 请求 | 保留其读者问题，重写为 Response 封套、item union、tool call 提议和宿主回传流程；移除“传统接口不能表达 Agent”的绝对化对比。 |
| 20.5、20.12.4、20.13、20.18、20.21 | response/conversation/task state、完整性和续接 | API context continuation 归 20.6—20.7；checkpoint、effect 状态和 workspace ownership 回链第十七、二十一章；删除通用协议层与执行层状态图混成一层的写法。 |
| 20.6、20.12.2、20.14 | 流事件、断线、乱序、重放 | 真实 Responses SSE 事件和 background starting_after 归 20.5、20.8；通用 ACK、event ID 去重协议及任意同步流恢复要求移除，因官方 API 来源不支持这些通用保证。 |
| 20.7、20.12.3、20.19、20.22 | Chat 接口迁移 | 合并到 20.10，保留 endpoint/input/output、工具回传、text.format、状态与流式映射；旧迁移清单中无来源的字段断言删去。 |
| 20.8、20.20 | coding Agent 执行边界 | 合并进 20.4 的单条教学构造，强调自定义工具由宿主执行；业务授权、workspace 和任务验收交回通用章节。 |
| 20.9、20.12.1、20.24 | 错误、取消、阶段状态和灰度 | API 的 response status/error/incomplete 与后台取消归 20.2、20.8—20.9；通用 effect 状态、发布灰度流程移出本章。 |
| 20.10、20.17 | API 测试与验收案例 | 改为 20.10 的迁移对照维度，不宣称固定 golden suite 或“上线协议门槛”是官方合同。 |
| 20.11—20.12、20.25 | 误区、面试回答、练习和阶段总结 | 有效误区并入 20.2、20.5—20.9；本章末面试回答归 20.11；练习转入 EXERCISES.md 的 35EO。 |
| 20.12.5、20.13、20.15、20.17 | 自拟完整性乘积 gate、状态封套、成本公式与汇总 gate | 删除任意综合公式；保留协议状态与业务状态分开的解释，以及官方 usage 与动态价格查询路径。 |
| 20.16、20.23 | capability、权限、数据保留、多模态预算 | 官方 Responses 数据保留收束至 20.7；能力/权限通用检查回链第四、八、二十二章；空白的 20.23 删除。 |
| 20.26—20.29 | GPT-5.6 Luna、GPT-6 Sol/Luna、model alias 与 leaderboard snapshots | 删除模型发布和排行榜身份分析，因其不属于本章 Responses API 核心职责；回链第十九章及对应模型资料。原稿的 gpt-6-luna、gpt-5.5 source notes 与 gpt6_sol_contract_audit.py 目标不存在，相关模型快照、榜单结论和 toy audit 运行声明不再保留。 |

本轮没有只删除旧的通用 gate，而是把读者仍需理解的工具往返、流式、状态续接、保留和错误边界分别交给当前官方 API 合同或本册的通用主讲章。

## 正文主线、同步与静态验收

正文调整为 20.1—20.11 连续路径：为何使用 typed Response → 外层状态 → item → 自定义工具回路 → SSE → conversation state → retention → background/cancel → usage/errors → migration → interview/practice。随后依据官方 state guide 补充：<code>store=false</code> 的无状态多轮示例由应用重放累计 history；<code>previous_response_id</code> 示例是 API 响应链路线，正文未把两者写成普遍互斥，也未声称 ID 可代替应用历史。章节没有可执行代码块或数学公式；表格用于并列 item 责任、事件含义和续接方式。

已同步第二十册目录第 20 章摘要、练习 35EO、面试题 40AD 和本册架构地图下一入口。简介没有逐章摘要，无需修改；plan.md、README 和 PDF 未修改。新知识证据只写入 research；书稿去向与进度写入本 progress 文件。

静态验收：
- H2 编号 20.1—20.11 连续；无空白旧小节残留。
- 本章无代码围栏或数学公式；示例用职责表呈现，不承诺可执行 API 操作。
- 本章、目录、练习、面试题、架构审计、research 记录和本审计中的本地 Markdown 链接目标已逐项静态检查，缺失为 0。
- 第 20 章旧稿三个缺失 source-note / audit-script 目标均已移除；第二十册其他章节剩余缺失目标列于 architecture-audit.md。
- 已执行 git diff --check，未发现空白错误。

## 连续复读补充（2026-11-19）

沿第二十册目录复读本章，并将第 20 章与第十七章副作用恢复、第十八章上下文折叠、第十九章评估及第二十一章 workspace 主讲边界对照。章节职责和现有章节路线保持稳定；读者能够区分 API Response 终态、工具执行回执、对话续接与任务验收。

针对快速变化的官方 API 合同，重新访问 OpenAI Background mode 和 Data controls 文档后，在正文 §20.7 补充默认 <code>soonest</code> 后台完成窗口约 10 分钟的磁盘暂存，以及 Modified Abuse Monitoring 下标准保留期的 <code>store=true</code> 条件；在 §20.8 写明只有默认窗口支持创建时的后台 streaming，<code>2h</code>/<code>8h</code> 请求不启用 streaming，及未创建为流式响应时不能事后流式读取。证据事实与边界补入[研究记录](../../research/openai-responses-api-contract-evidence-2026-11-18.md)。未运行 API、SDK 或书中代码。

静态复核：H2 标题仍为 20.1—20.11 连续；无代码围栏或行内数学表达式；相关本地 Markdown 链接在本轮检查中缺失 0；<code>git diff --check</code> 通过。目标读者试读、GitHub 页面核验和全量外链可达性仍属于整册终验。

## 开放风险与下一入口

- Responses API 页面、模型支持、保留边界、background options 和价格会更新；本轮按 2026-11-19 官方页面复核的条件补充正文，后续维护仍需重新核验。
- 没有运行 API 请求、SDK、示例代码、benchmark 或自动化测试；没有做目标读者试读、GitHub 页面渲染或所有外链的长期可达性检查。
- 下一入口：[第二十一章《Persistent Workspace》](../../book-20-agent-harness-runtime/chapters/21-persistent-workspace与状态边界.md)。本章精修完成；第二十册第 21—24 章及全套书精修 goal 继续 active。
