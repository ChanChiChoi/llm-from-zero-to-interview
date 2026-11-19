# Claude Sonnet 5.5：工具循环中的状态回放与安全路由

本章以 Claude API 的 Messages API model ID `claude-sonnet-5-5` 为例，回答一个 Agent 运行时问题：工具调用暂停、对话续接或安全拒答之后，宿主怎样知道哪些状态要保留、实际由哪个模型作答、任务是否真的完成？以下时效性接口事实按 2026-11-19 查阅的官方文档表述；其他托管平台、账户和模型组合可能有差异。

先看一条教学构造的工单：工单 742 要求修复日期解析问题、运行测试，但不要提交代码。Messages API 不只返回一整段文本，而是返回带类型的内容块，例如 `thinking`、`tool_use` 和 `text`。这里的 `read_file`、`write_patch`、`run_tests` 是宿主应用执行的 client tools；Anthropic 自己运行的 server tools 属于另一种执行位置。

| 时点 | 发生了什么 | 这份记录能证明什么 |
| --- | --- | --- |
| 请求 | 宿主把工单要求、`read_file`/`write_patch`/`run_tests` 工具定义和对话历史发给 Sonnet 5.5；起始工作区快照记为 `r17` | 模型收到哪些目标和工具说明 |
| 读取 | 模型用 `tool_use` 提出 `read_file`；宿主核对权限后读取 `r17`，并用 `tool_use.id` 与 `tool_result.tool_use_id` 配对返回结果 | 模型提出了读取动作；宿主的执行记录证明读取发生在 `r17` |
| 编辑与测试 | 模型基于读取结果提出补丁并请求 `run_tests`；宿主检查补丁范围后写入，形成 `r18`，再在 `r18` 上运行测试并保存回执，不提交代码 | 模型提议本身不能证明已执行；宿主记录确认补丁和测试对应 `r18` |
| 中断 | 工单暂停时，用户又改动工作区，当前快照成为 `r19`；API 历史前缀仍可保持原样 | thinking block 可能仍符合 API 前缀合同，但 `r17` 的读取结果和 `r18` 的测试回执都不能证明 `r19` 的状态 |
| 恢复与验收 | 宿主检查 `r19` 的实际 diff，在当前快照重新运行测试，并确认没有提交 | 只有当前快照上的 diff、测试回执和提交状态共同支持任务验收 |

该例中的每一步都归不同状态所有者：API 保存的对话历史不能代替工作区快照，`tool_use` 与 `tool_result` 的 ID 配对只关联一次请求和结果，不能代替宿主授权或当前执行回执；测试通过也要绑定实际测试过的快照。

因此，API 历史不是工作区快照，模型的“已完成”也不是执行回执。第二十册第 17 章讲长任务恢复与外部副作用，第 21 章讲 workspace 状态；本章只看 Sonnet 5.5 的接口如何影响这条状态链。

## 24.1 `between_tools` 关闭了什么

Sonnet 5.5 的 Messages API 默认启用 adaptive thinking。旧客户端若发送 `thinking: {"type":"disabled"}` 会收到 400；要关闭工具调用前的 upfront thinking，应改用 `thinking: {"type":"between_tools"}`。这个设置只关闭工具调用前的思考阶段，不表示工具之间绝不会返回进度消息。

| 设置 | Sonnet 5.5 的公开行为 | 客户端需要注意 |
| --- | --- | --- |
| `thinking.type=disabled` | 请求被拒绝，返回 400 | 不能把旧模型配置原样迁来 |
| `thinking.type=between_tools`，effort 为 low、medium 或 high | 接受；不做 upfront thinking | 工具间较长的进度消息仍可能以 `thinking` block 返回 |
| `between_tools`，effort 为 xhigh 或 max | 返回 400 | 这组 effort 不兼容该模式 |
| `between_tools`，同时设置 `display`、`budget_tokens` 或 `block_binding` | 返回 400 | 这些额外 thinking 字段不属于该模式的请求合同 |

没有工具可调用时，`between_tools` 会直接生成文本答复。相反，默认 `display="omitted"` 时，返回的 thinking block 可能带有空的可读文本；如果界面只渲染 `text` block，用户会觉得长任务“没反应”，但这不等同于 API 报错。客户端应按 block 类型处理消息，并把模型进度与可见界面分开设计。

`effort` 控制模型投入多少工作，不是精确 token 预算；`max_tokens` 才是本轮输出的硬上限，而且会被 thinking 与最终答复共同消耗。Sonnet 5.5 重新校准了 effort 的行为，不能假设 Sonnet 5 上的同一个标签会产生同等成本或思考深度。迁移时应按自己的任务重新比较质量、延迟和成本。[Sonnet 5.5 迁移说明](https://platform.claude.com/docs/en/models/sonnet-5-5/whats-new-sonnet-5-5)、[Thinking 指南](https://platform.claude.com/docs/en/build-with-claude/thinking)和[专用提示指南](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-sonnet-5-5)分别说明了这些接口行为与显示差异。

## 24.2 thinking block 是需要按合同回放的状态

工具调用把一次 assistant turn 分成多段：模型先提出 `tool_use`，宿主返回 `tool_result`，模型再接着生成。工具结果不是新的独立任务；它接在刚才那次模型输出之后。对 Sonnet 5.5 的工具回合，宿主应把 API 返回的 assistant 内容块按原顺序、不改内容地带回去，包括 `thinking` 和 `redacted_thinking` block。thinking block 携带供接口校验和续接使用的签名状态；默认显示设置下其可读文本可能为空，redacted block 本身也不提供可读内容。只保留可见文本，或把 block 拆开重排，可能让续接丢掉原有状态或触发协议错误。

thinking block 的可用性有三种彼此独立的绑定：

| 绑定 | API 检查的关系 | 不匹配时可能发生什么 |
| --- | --- | --- |
| 模型绑定 | 目标模型能否读取生成该 block 的模型状态；这是有方向的关系，还受 provider surface 影响 | 请求可以成功，但不可读取的 block 会被丢弃 |
| 前缀绑定 | block 之前的 `system`、`tools` 与所有旧消息是否仍和生成时相同 | 对 Sonnet 5.5，失败可能返回 400，也可能没有 400；不能只看 HTTP 状态判断 |
| 账号绑定 | 生成 block 的账户与当前账户是否相同或关联 | 某些账户/托管组合会静默丢弃 block；部分 surface 可借 beta `input_transformations` 查看原因 |

模型兼容不是对称矩阵，还要按版本读表。当前 Claude API 的方向表把 Sonnet 5.5 生成的 blocks 标为可由 Fable 5.2、Mythos 5.2 和 Opus 5.5 读取；来自 Fable 5.1 或 Mythos 5.1 的 blocks 对 Sonnet 5.5 则标为 limited support，并非一律不可读。该表描述 Claude API；切到 Managed Agents 或其他云平台时，应按对应 surface 的现行文档核对来源模型、目标模型、账户和平台条件。尤其 Sonnet 5.5 的 blocks 仅在生成账户或关联账户中可用，其他账户的请求可能成功但会丢弃 blocks。[官方 preserved-thinking 指南](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking)列出当前方向与平台条件。

前缀变化也不只指 system prompt。重写、插入或删掉旧消息，或改动工具 schema，都可能使后续 thinking block 与原来的上下文不匹配。稳妥的续接方式是沿用原来的 system、tools 和历史，只在末尾追加新消息；若确实要改写历史，就不要把依赖旧前缀的 block 当成仍有效的状态。当前 [Platform 前缀检查指南](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking/prefix-checks)把 Sonnet 5.5 标为“可能返回 400”；[官方 Cookbook](https://platform.claude.com/cookbook/extended-thinking-extended-thinking-with-tool-use)仍以账号创建日期描述默认检查行为。两页对默认边界的说明粒度不一致，因此不要把某个日期推广成所有账户、模型和托管平台的通则；目标环境要按当前合同处理 400 与未报错两种结果。

Sonnet 5.5 默认保留先前轮次的 thinking blocks；这些状态会随历史进入上下文并占用窗口。`between_tools` 不会自动清除已有历史。长任务需要把这些块与普通消息一起纳入上下文管理，折叠和裁剪的条件见[第二十册第 18 章](18-context-folding上下文折叠.md)。还有一条工程边界：thinking block 是对话状态，不是任务的持久记忆，也不记录工具调用是否真的产生副作用。宿主应另外保存工具执行回执、工作区版本和验收结果；遇到回执丢失时，查询实际系统，不能靠重放 thinking 来推断写入是否成功。

## 24.3 `tool_use` 提议动作，不授予执行权限

Sonnet 5.5 不接受强制工具选择：`tool_choice` 为 `any` 或指定单个 `tool` 会返回 400；`auto` 与 `none` 才受支持。若需要限制参数形状，可使用工具 schema 的 `strict: true` 或 Structured Outputs，但 schema 合法不等于调用安全，也不等于模型必定选择某个工具。

回到工单 742：模型提出 `run_tests` 后，宿主仍要检查这项工具是否在任务允许范围、参数指向哪个仓库、当前权限是否允许执行，再由执行器运行并记录实际结果。模型输出是候选动作；授权、沙箱和执行回执属于宿主。工具注册与参数边界见[第二十册第 4 章](04-工具系统与tool-registry.md)，权限见[第 8 章](08-权限模型与安全沙箱.md)。[Sonnet 5.5 迁移说明](https://platform.claude.com/docs/en/models/sonnet-5-5/whats-new-sonnet-5-5)记录了强制工具调用的 400 条件；[Tool use 指南](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)说明调用方与工具执行器的消息流程。

## 24.4 refusal 是响应状态，fallback 是按类别配置的路由

安全拒答不是 HTTP 传输错误。Sonnet 5.5 可以返回正常的 HTTP 200 响应，同时将 `stop_reason` 设为 `refusal`，并在 `stop_details.category` 中报告触发的策略类别。这些标签分别覆盖网络安全伤害、化学/生物伤害、前沿模型开发、reasoning extraction 或其他政策领域；类别可能为空，因此客户端应先判断 `stop_reason`，再按可用的类别信息处理。类别说明路由原因，不替代应用自己的权限判断。

| 拒答类别 | Anthropic 当前默认 fallback 合同 |
| --- | --- |
| `cyber`、`frontier_llm` | 开启 beta `fallbacks="default"` 时，可能转由 Sonnet 5 继续处理 |
| `bio`、`reasoning_extraction`、`general_harms` | 不走这条默认自动 fallback |
| 429、过载或服务端错误 | 不是这条 safety fallback 的触发条件 |

默认 fallback 目前只在 Anthropic API 和 Anthropic 管理的 Google Cloud surface 提供，且需要对应 beta 配置；其他 provider 和产品要分别核对。若开启它，响应中的实际服务模型、fallback block 和 `usage.iterations` 能说明发生过哪些尝试。流式请求可能先以请求模型开始、随后才出现 fallback 标记，所以不能只保存最初的 `message_start`；中途切换时还要读 fallback block 的 `to.model` 和结束事件中的 `fallback_message` 记录。如果 refusal 在流中的某个 tool-use block 尚未闭合时触发，官方合同规定直接返回 refusal，不执行这条 server-side fallback。fallback 模型若也过载，原拒答可能仍会返回。上述规则见[Refusals and fallback](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback)；`between_tools` 请求在 2026-07-01 beta fallback 路由到 Sonnet 5 时还有 thinking 参数兼容处理，不能假设其他 fallback target 也接受同一组字段。

对宿主来说，这条路由不是“重试任意错误”的开关。日志至少应区分 requested model、实际回答的 model、拒答类别、fallback 是否执行及每次尝试；工具副作用仍由执行回执和外部系统确认。发现拒答后换了模型，也不自动改变调用者权限或任务边界。任务评分和 Model×Harness 归因见[第二十册第 19 章](19-harness-aware-evaluation.md)。

## 24.5 发布方评测说明发布方测了什么

Anthropic 的公开 [Transparency Hub](https://www.anthropic.com/transparency)把 Sonnet 5.5 的评估描述为特定模型与安全配置下的发布方测试，并将化学/生物能力和 safeguards 放在自己的风险框架中报告。这类材料能说明厂商如何评估和部署模型，不能证明你的宿主权限、工具执行或任务验收正确。

当某项评测启用工具、fallback、产品级 safeguards 或特定 verifier 时，结果对应的是那组配置下的系统。报告时应保留任务版本、harness、effort、工具权限、fallback 与实际服务模型；这与第二十册第 19 章的模型—Harness 归因原则一致。排行榜配置、另一版本模型的 Agent 分数以及发布方测试，都不能替代目标部署的任务证据。更完整的官方来源、版本范围与未解决边界见[Sonnet 5.5 研究记录](../../research/model-update-2026-09/claude-sonnet-5.5-source-notes.md)。

如果评测使用 thinking 或行动轨迹训练/运行监控器，还要分清两件事：轨迹里是否出现了可识别信号，以及监控器能否从信号中判断指定属性。Guan 等人提出的 monitorability 评测显示，在其研究任务中，CoT 监控通常比只看动作更有信息，但并不完美，也不是所有属性都同样容易识别。这不证明 thinking block 是内部过程的完整记录，也不能替代工具授权、执行前拦截和外部状态审计；研究的任务与监控协议见[论文](https://proceedings.mlr.press/v306/guan26h.html)及[安全评测研究记录](../../research/ai-safety-alignment-architecture-scan-2026-11-03.md)。

回到工单 742：宿主按原样保存需要续接的消息块，单独追踪文件变更、测试回执和提交状态；遇到 refusal 时记录类别与实际服务模型；最后由独立检查确认代码和测试属于同一工作区版本，而且没有提交。这样读者才能分清模型生成了什么、接口路由了谁、宿主实际做了什么，以及任务最终是否达到要求。
