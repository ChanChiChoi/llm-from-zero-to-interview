# Agent Harness 故障诊断、重试与运行证据

检索日期：2026-11-18  
主题：Agent/Coding Agent 故障怎样从一条可复查运行轨迹定位到最早偏离，并区分重试、恢复、策略拒绝、外部副作用、观测和评估问题。

## 检索范围与方法

本轮搜索覆盖 Agent SDK 的错误、timeout、重试和 replay safety；trace 导出与敏感数据；工具 guardrail 的执行位置；LangGraph 的 retry、timeout、恢复和副作用边界；Google ADK 的日志、环境模拟及失败迭代；OpenTelemetry 错误语义；生产 incident 的组织记录。检索使用网页搜索，并直接打开 OpenAI Agents SDK、Google ADK/Agents CLI、LangGraph、OpenTelemetry 和 Google SRE 官方文档。官方页面多为持续更新文档，除来源明确给出的版本外，不推断固定 SDK 版本；记录访问日期而非声称复现了某个版本行为。

主要查询组：

- `OpenAI Agents SDK tracing debugging errors retries guardrails`
- `Google ADK debugging evaluation logging observability errors`
- `LangGraph fault tolerance retries idempotency checkpoint errors`
- `OpenTelemetry recording errors handled retried exceptions`
- `Google SRE incident response working record postmortem`

## 来源与证据

| 来源 | 实际核对内容 | 可支持的结论 | 不支持的结论 / 限制 |
|---|---|---|---|
| [OpenAI Agents SDK Models](https://openai.github.io/openai-agents-python/models/)，访问 2026-11-18 | Model-call timeout 与 runner-managed retries 段落 | 当前 Python SDK 的 timeout 限制单次模型调用 attempt，不覆盖整个 agent run、function tool 或 retry backoff；SDK 重试需配置并遵守 replay-safety 判断。文档区分 `safe`、`unsafe`、`unknown`，并说明 abort、已有流式输出及某些本地副作用请求不应重放。 | 这是该 SDK 当前公开合同，不代表其他框架或所有 provider 的 timeout、取消或 retry 语义。timeout 也不能单独证明 provider 没有处理请求。 |
| [OpenAI Agents SDK Guardrails](https://openai.github.io/openai-agents-python/guardrails/)，访问 2026-11-18 | input、output、tool guardrail 的工作流位置与 execution modes | Agent input guardrail 只运行于链首 agent；output guardrail 运行于最终回答 agent；tool guardrail 按工具调用运行。并行 input guardrail 可在模型工作已启动后才发现问题；blocking mode 会等待 guardrail 完成后再启动 agent。 | 说明 SDK 的 guardrail 位置和并行语义，不构成所有 Harness 的安全边界；tool guardrail 与 approval 的相对执行顺序还受该 SDK 配置影响。 |
| [OpenAI Agents SDK Tracing](https://openai.github.io/openai-agents-python/tracing/)，访问 2026-11-18 | span 范围、后台 batch export、`flush_traces()`、敏感数据配置 | 当前 SDK 默认批量后台导出 trace；长任务结束后 trace 可能尚未立即出现在 dashboard，要求即时交付时可在 trace 关闭后 flush。generation/function spans 可含输入输出，需按诊断需要控制敏感内容采集。 | dashboard 暂无记录不证明运行未发生；此时也不能只凭 SDK exporter 行为认定 trace 一定丢失。配置名称和默认值是易变的 SDK 合同。 |
| [Google ADK Logging](https://adk.dev/observability/logging/)，访问 2026-11-18 | production log level、prompt 内容捕获、debug output 警告 | ADK 文档提醒 DEBUG 日志可能很详细且含敏感信息；完整 prompt/response 捕获和 debug plugin 输出可能包含原始 prompt、tool args、session state，应按敏感资料保护。 | 这是 ADK 当前日志实现说明，不推出其他 SDK 的默认采集行为，也不支持把所有原始内容都记录进生产 trace。 |
| [Google ADK Environment Simulation](https://adk.dev/evaluate/environment_simulation/)，访问 2026-11-18 | 支持版本、tool-call interception、error/latency injection、seed | 该页说明 ADK Python v1.24.0 提供在 evaluation 中拦截工具调用并替换受控响应的能力，可注入错误/延迟并用 seed 固定概率注入，帮助构造外部依赖故障的可重复场景。 | 这是 ADK 功能合同。带 LLM 的 mock strategy 仍可能引入随机性；不能将模拟运行当作真实外部服务集成验证。 |
| [Google Agents CLI: Debug a failing agent](https://google.github.io/agents-cli/guide/debugging/)、[Evaluation guide](https://google.github.io/agents-cli/guide/evaluation/)，页面标示 2026-11-04 更新，访问 2026-11-18 | 单例复现、保存 eval case、查看 metric 与 trace、修正并对比 | 官方开发指南给出“复现一个输入—把失败固化成 eval case—读失败指标和 trace—修复—重跑并比较”的具体开发循环。页面也区分现成与实验性 CLI 子命令。 | 这代表该 CLI 的开发工作流示例，不证明所有项目都应使用同一工具、metric 或阈值，也不证明这一顺序具有普遍学习效果。 |
| [LangGraph Fault Tolerance](https://docs.langchain.com/oss/python/langgraph/fault-tolerance)，访问 2026-11-18 | retries、idempotency、timeout、error handling、missing acknowledgement、recoverable/unexpected errors | 文档说明 retry 会从 node 开头重跑，node execution 因而是 at-least-once；checkpoints 不包含节点内部产生的外部副作用。未收到 provider acknowledgement 时，外部操作状态仍是 unknown，应查询状态；不能把本地无记录当成未执行。已知可由模型修正的 tool/parse error 可作为应用流程返回给模型；无法处理的意外错误应暴露并使 run 失败。 | 这是 LangGraph 节点和 checkpoint 的语义，不能泛化到其他运行时；timeout/error handler 部分要求 `langgraph>=1.2`。幂等键是否去重取决于外部 provider 是否实际支持和遵守。 |
| [OpenTelemetry: Recording errors](https://opentelemetry.io/docs/specs/semconv/general/recording-errors/) 与 [Exceptions in logs](https://opentelemetry.io/docs/specs/semconv/exceptions/exceptions-logs/)，访问 2026-11-18 | operation-level error 分类、retried/handled 错误、exception status | 语义约定强调 error 要按被观测操作的语境判定；例如 404 可能是失败也可能是预期探测。开发阶段的通用错误记录建议指出，已被处理或重试后成功的失败，不应把最终成功 operation 的 span/metric 记成失败。 | `Recording errors` 页面标为 Development；相关 exception 记录约定仍在演进。它不是针对 Agent 的成熟专用指标定义，正文应把它作为遥测口径参考而非运行时成功标准。 |
| [Google SRE Workbook: Incident Response](https://sre.google/workbook/incident-response/) 与 [Incident Management Guide](https://sre.google/resources/practices-and-processes/incident-management-guide/)，访问 2026-11-18 | incident 管理与恢复的区分、协作角色、working record、blameless postmortem | Google SRE 的生产案例区分“恢复服务”和“协调响应”，并强调预先明确角色、持续记录 debug/mitigation、复盘并分配后续行动。它支持把事故过程和事后学习纳入工程实践。 | 这些材料记录 Google/PagerDuty 的组织实践，不证明特定组织模板普遍最优，也不等同于单个 Agent run 的技术排障步骤。 |

## 证据边界与仍未知内容

- 跨框架共有的只是需要辨认单次 attempt、运行、任务结果、持久状态、外部副作用和观测记录之间的边界；重试时机、超时范围、异常类型和 trace 导出行为必须按固定框架/provider 版本确认。
- trace 缺失可能来自未埋点、采样、buffer 尚未 flush、导出失败、权限/配置或保留策略；现有文档不能仅凭 dashboard 空白确定本地根因。
- 日志和 trace 的诊断价值不构成收集原始 prompt、token、文件内容、命令环境变量或凭证的理由。各平台的脱敏能力与策略不同，部署方仍须核验实际数据流。
- 使用合成任务的事故复走只能说明一条机制路径；它不证明真实用户事件的发生频率，也不支持“最常见故障”的总体排序。
- 将“保存失败样本并回归”与“确定性复现”分开：mock/snapshot 可锁定部分外部输入，但真实模型、provider、工具和环境仍可能变化。
