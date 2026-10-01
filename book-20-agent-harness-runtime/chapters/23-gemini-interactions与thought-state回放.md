# 第 23 章 Gemini Interactions：跨轮状态、签名回放与工具责任链

## 23.1 长任务的状态放在哪里

Agent 调用模型、执行工具、收到结果，再继续推理。跨轮继续时，系统至少要回答：历史由谁保存？哪些参数会继承？断线重试如何避免重复副作用？何时删除交互数据？

Gemini Interactions API 把一轮交互暴露为有序 steps，并提供服务端存储和 `previous_interaction_id` 延续能力。它提供 API 级历史管理与交互可观测性，不等于模型拥有永久记忆，也不替宿主保存文件、权限、工具执行结果或业务状态。

本章以已出现在 Artificial Analysis 和 DataCurve DeepSWE 的 `Gemini 3.8 Flash` 为锚点。这里记录的是 Google API/runtime 契约，不能由这些字段反推参数量、Transformer 结构、训练配方或内部 verifier。来源与榜单配置见[研究笔记](../../research/model-update-2026-09/gemini-3.8-flash-source-notes.md)。

## 23.2 Interaction 是一轮执行记录

一次 Interaction 可以包含用户输入、thought、内置工具调用/结果、自定义 function call/response 和最终 model output。流式调用还会产生创建、step 开始、增量、step 结束和完成等事件：

```text
user_input -> thought -> tool_call -> tool_result -> thought -> model_output
```

有序 steps 让运行时可以区分模型规划、工具执行、结果回灌和最终输出。只保存答案文本会丢掉归因所需的状态，也无法解释工具成本、重复动作和最终成功。

API step 不是完整生产 trace。宿主仍需记录授权决策、环境版本、副作用回执和最终 artifact；模型生成 `tool_call` 不代表调用已经执行或获得权限。

## 23.3 Stateful 与 stateless：两种状态所有权

| 维度 | Stateful continuation | Stateless replay |
| --- | --- | --- |
| 历史持有人 | 服务端依据 `previous_interaction_id` 取回已存储历史 | 客户端保存并提交所需历史 steps |
| 延续方式 | 引用前一轮完成的 Interaction ID | 原序提交需继续的历史内容 |
| 主要优点 | 请求体较小，服务端维护交互历史 | 客户端控制持久化、审计和跨系统恢复 |
| 主要风险 | 历史有保留期；删除或过期后不能继续 | 容易漏字段、重排 steps 或重复执行工具 |
| 每轮仍要显式传入 | tools、system instruction、generation config | tools、system instruction、generation config |

关键边界：`previous_interaction_id` 继承 conversation history，不是完整 runtime snapshot。`tools`、`system_instruction` 和 `generation_config`（包括 thinking level、temperature）按当前 Interaction 生效，下一轮必须重新指定所需配置。

```text
stateful: create(turn_1, tools=A, config=X) -> id=I1
          create(turn_2, previous_interaction_id=I1, tools=B, config=Y)
          # I1 提供历史；B/Y 是本轮显式配置

stateless: client 保存完整原序历史与 opaque fields
           create(turn_2, replayed_history=H1, tools=B, config=Y)
```

应用任务状态应另有权威记录，例如 task ID、workspace revision、权限、待执行副作用和 verifier 状态。Interaction ID 是服务资源引用，不能代替任务数据库或 checkpoint。

## 23.4 Thought summary 与 signature：保留协议状态，不读取隐藏思维

- `summary` 是可选的开发者可见摘要，不是完整 chain-of-thought；某些响应可能没有 summary。
- `signature` 是用于协议延续的 opaque 字段。调用方不应尝试解密、重建或改写，也不能把它展示为可读推理。
- Stateless replay 要保留实际返回的兼容状态块和必要工具上下文；摘要不能替代原始状态。

官方资料存在多处需要保留的范围差异，不能把不同 API 的名字和字段表强行合并：

| 资料 | 自定义函数 step / 关联 ID | signature 声明 |
|---|---|---|
| Thinking prose | Interactions 标准 function call | signature 限于 thought 与 built-in tool；排除标准 function call；同时称每个 thought signature 必有 |
| Tool combination prose | `function_call` 与 `function_response`，两者按 `id` 配对 | 称 Gemini 3+ 所有 tool call/result（包括自定义 function）都有 signature |
| 当前 Interactions API reference | `function_call.id` → `function_result.call_id` | `FunctionCallStep`/`FunctionResultStep` 不声明 signature；`ThoughtStep.signature` 标为 optional |
| Google Gen AI Python SDK | 与当前 API reference 相同 | 生成 schema 不声明 custom-function signature，thought signature optional；Google Search call/result signature optional |

2026-09-28 通过 7890 复核，Google overview 现在链接到 [`Interactions API reference`](https://ai.google.dev/api/interactions-api)；旧 `/api/interactions` 路径返回 404。SDK 当前 `main` 最新可见提交固定为 [`6d012889752f65c1a51d0ad6e5970fc97d19c4ca`](https://github.com/googleapis/python-genai/tree/6d012889752f65c1a51d0ad6e5970fc97d19c4ca)，相关生成字段文件与 9 月 24 日 revision `4742c9a5c213a587add126a500a271824e2f0add` 的 SHA 完全相同。当前正式 schema 因而更支持 Thinking 的窄口径，但它只说明 typed fields；SDK `BaseModel extra="allow"` 会保留 extra fields，lenient open union 以 `UnknownStep.raw` 保存未知 step，所以不能把字段未声明解释成服务端禁止或绝不返回。

工程上以**具体 endpoint/version 的 raw response** 为准：保存所有实际返回的 opaque/extra fields，不自行伪造或剥除；按 Interactions 的 `function_call.id` → `function_result.call_id` 关联，不把 Tool combination 的 `function_response.id` 或 GenerateContent 的 `functionResponse` 名称直接移植进另一套 wire schema。当前 API reference 对 `ThoughtStep.signature` optional 与 Thinking prose“必有”仍不一致；没有真实 endpoint/API key 时应明确标成未裁决，而不是选一条文档冒充已验证的服务端行为。

这里的 `BaseModel` 要按生成包范围精确引用：Interactions client 使用 `google/genai/_gaos/interactions.py` 与 `google/genai/_gaos/types/interactions/*`；其 `google/genai/_gaos/types/basemodel.py` 才是 `extra="allow"` 的 typed model base。仓库中另一个 `google/genai/_common.py` 的 `BaseModel` 配置 `extra="forbid"`，属于不同 SDK 类型模块，不能据此否定或泛化 Interactions 的模型行为。Interactions 的 `FunctionCallStep`/`FunctionResultStep` typed fields 未声明 signature，`ThoughtStep.signature` 是 optional；Step union 用 lenient `parse_open_union`，将未知 payload 放进 `UnknownStep.raw`。详细文件路径、revision 与 SHA 见来源笔记 §19。

本章 replay toy 的 `thought.signature` 默认必需门禁是**宿主的保守 stateless-replay 策略**，不是 SDK schema 的 required 字段：schema profile 可接受该字段缺省，但应用仍应原样保存实际收到的签名，并由 capability probe 决定缺失状态的恢复策略。schema optional 也不证明真实 endpoint 一定省略签名。

## 23.5 SSE 恢复与工具副作用：事件重连不保证恰好执行一次

事件消费、工具执行和 Interaction 状态不是一个原子事务：

```text
收到 tool_call -> 宿主授权 -> 工具产生副作用
            -> 宿主写回执行回执 -> SSE 断开/客户端重连
```

若系统在写回执前崩溃，恢复逻辑可能再次执行同一调用。Harness 应建立独立执行账本，记录 task/interaction/step 或 call ID、工具名、参数哈希、策略决策、执行状态、回执、结果哈希、环境 revision 和幂等键。

将宿主幂等键绑定到任务与调用身份；未知副作用要先查询状态或进入人工恢复，不能盲目重试。工具结果按 call ID 回填，最终由独立 verifier 检查 artifact，而不是以模型回复作为验收。SSE 事件顺序与重放语义应按当前 schema/SDK 探测，不假设网络重连会让工具恰好执行一次。

## 23.6 工具上下文循环与权限边界

Google 文档把 Search、Maps、URL Context、File Search 等描述为服务端工具；Code Execution 有自己的结果步骤；自定义 function calling 与 Computer Use 则需要客户端/宿主执行真实动作。一个 Interaction 中可以交错出现工具意图、观察结果和后续输出。

工具目录不等于宿主授权。宿主仍需校验租户、资源、参数、allowlist、审批、超时、出站网络、幂等和业务结果。结构化输出只约束形状，不证明语义正确。

审计时分开记录：模型提出的调用、策略是否允许、执行器与环境、工具回执、verifier 是否接受产物。这个分层与通用[工具 registry](04-工具系统与tool-registry.md)、[权限沙箱](08-权限模型与安全沙箱.md)和[状态边界](21-persistent-workspace与状态边界.md)一致。

### 视频处理 step：Provider-managed 的媒体读取闭环

Gemini 3.8 Flash 的官方视频文档描述了专用 `processing_call` / `processing_result` steps：前者请求加载指定视频片段或 transcript（有 `id`），后者以 `call_id` 关联返回内容。它们让动态媒体选择可见于 Interaction trace，但不等于应用自定义 function，也不能据此推断 endpoint 的普通 function signature 规则。应用应保留媒体对象版本、时间范围、处理请求/结果及答案引用；服务端提供媒体处理 step，也不替宿主做租户授权和资源访问控制。

静态路径默认每秒取一帧；agentic 路径按问题动态读取帧、transcript 或音频，长视频可减少输入 token，但增加导航思考和处理往返。Usage 需区分 `total_thought_tokens` 与 `total_tool_use_tokens`。Google 报告的“最多约 88% 少用 token、约 7% 质量提升”是文档中的长视频比较描述；短于 5 分钟的片段可能因导航而增加 TTFT，自定义 FPS/区间也只支持 static。具体应用评测和宿主责任链见[第十七册 Code Agent 视频案例](../../book-17-agent-tool-use/chapters/07-code-agent.md#agentic-video从整段采帧到问题驱动的时间轴检索)及[研究笔记 §20](../../research/model-update-2026-09/gemini-3.8-flash-source-notes.md#20-2026-09-28-gemini-38-flashagentic-video-的按需证据读取)。

## 23.7 Thinking 预算、硬截止线与 `incomplete`

Gemini 3.8 Flash 官方页将 stable alias 的输入上限列为 `1,048,576`、输出上限列为 `65,536`；Thinking 文档支持 `low`、`medium`、`high`，`minimal` 不在该模型的合法配置中。档位是请求级运行控制，不是不同 checkpoint。

`max_output_tokens` 可能同时消耗 thought tokens 与可见输出预算。它是硬截止线，不等于降低 thinking level：在思考阶段触顶时，Interaction 可能以 `incomplete` 结束，最终输出为空或截断，但已生成 thought tokens 仍可能计费。评测应分开记录 thinking level、max output、thought/output usage、工具轮次、重试和完成状态。

降低成本/延迟时，应做按任务的 thinking-level 路由和预算实验，而不是把硬上限压得过低。low/medium/high 对比要固定模型版本、任务、工具与 verifier，并报告 artifact 成功、TTFT/TPOT、工具失败和单位成功成本。

## 23.8 存储、删除、缓存与长期记忆

已核验的文档快照描述 Interactions 默认 `store=true`；当时付费层默认保留 55 天、免费层 1 天，付费项目可配置保留窗口并按 ID 删除。`store=false` 会关闭服务端存储，不能与 `background=true` 或后续 `previous_interaction_id` 延续一起使用。保留期限是会变化的服务合同，部署前应按账户、区域和当前产品文档复核。

| 概念 | 所属层 | 不能混同为 |
| --- | --- | --- |
| Interaction retention | 服务端 API 资源保留/删除 | 应用永久记忆 |
| 应用 memory | 宿主维护的长期知识 | 模型会话历史 |
| implicit caching | 服务侧重复前缀复用 | GPU KV cache |
| GPU KV cache | 单次生成的注意力状态 | 应用长期记忆 |
| thought signature | 协议回放所需 opaque 字段 | 可读思维链 |

`store=false` 不会自动替宿主保存 history；`store=true` 也不表示应用可把历史当永久记忆。删除 Interaction 后，还要分别检查应用日志、workspace、工具产物和外部系统记录的保留/清理策略。

## 23.9 最小 harness 与 capability probe

将 provider history 与应用任务状态分成两份账本：

| Provider Interaction ledger | Host task ledger |
| --- | --- |
| model ID、Interaction ID、previous ID | task/tenant/workspace ID 与 revision |
| 原始 step/event 顺序、opaque fields | 工具 schema、版本、权限、审批 |
| store、保留策略、删除状态 | call ID、幂等键、执行器、副作用回执 |
| thinking level、输出上限、usage | verifier、最终 artifact、验收状态 |
| endpoint、SDK、API version | 去重、重试预算、错误与恢复决定 |

最小 capability probe 应覆盖：

1. Stateful continuation 恢复哪些 history，工具和 generation config 是否需要重发；
2. store=false、background、delete 和过期 ID 的错误行为；
3. Stateless thought/tool history 中 opaque fields 的实际位置；
4. `function_call.id` 与 `function_result.call_id` 的关联，以及实际 signature/extra 字段；
5. SSE 断线、重复事件、工具超时和副作用幂等；
6. max output 触顶、incomplete、thought/visible usage；
7. 当前账户的 retention、删除和日志配置。

没有真实 endpoint 授权时，可以用合成协议测试验证 harness 的失败路径；结果必须标记为 local toy，不能冒充 Gemini 服务端行为。

## 23.10 面试回答模板与常见误区

回答“如何让多轮 Gemini Agent 在断线后可靠恢复”时，按四层组织：

1. **History**：选择 stateful previous ID 或客户端 stateless replay，明确谁持有历史；
2. **Configuration**：每轮重新声明工具、system instruction 和 generation config；
3. **Execution**：保存原始 steps/signatures 与 call/result ID；工具由宿主授权、幂等执行并记录回执；
4. **Verification**：从真实 workspace/外部系统读取状态，用独立 verifier 验 artifact；将 incomplete、失败和未知副作用留在未完成状态。

常见误区包括：把 previous ID 当完整 checkpoint；把 summary 当完整推理、signature 当可读文本；静默合并官方文档的字段范围差异；混淆服务端历史、implicit cache、GPU KV cache 与长期 memory；SSE 重连后无条件重试工具；只检查最终文本；把 API/runtime 行为写成 Gemini 3.8 的内部架构或训练算法。

## 23.11 小结与证据边界

Interactions 将一次 Agent turn 表示为可追踪 steps，并提供 stateful history continuation。可靠性仍由 harness 补齐：保存配置和环境版本、保留 opaque state、分离模型意图与工具权限、对副作用做幂等处理、从外部系统恢复事实，并通过独立 verifier 验收。

本章属于官方 API/runtime 契约与系统设计建议。公开资料没有确认 Gemini 3.8 的参数规模、内部架构、训练配方、signature 编码或服务端 exactly-once 语义。

## 23.12 参考资料

- [Gemini 3.8 Flash 模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash)
- [Thinking](https://ai.google.dev/gemini-api/docs/thinking)
- [Interactions API](https://ai.google.dev/gemini-api/docs/interactions-overview)
- [Thought signatures](https://ai.google.dev/gemini-api/docs/thought-signatures)
- [Tool combination](https://ai.google.dev/gemini-api/docs/tool-combination)
- [Google Gen AI Python SDK pinned revision `6d012889`](https://github.com/googleapis/python-genai/tree/6d012889752f65c1a51d0ad6e5970fc97d19c4ca)
- [Gemini 3.8 Flash 来源摘记与快照哈希](../../research/model-update-2026-09/gemini-3.8-flash-source-notes.md)
