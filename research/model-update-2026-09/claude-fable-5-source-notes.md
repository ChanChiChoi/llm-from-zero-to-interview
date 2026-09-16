# Claude Fable 5：排行榜锚点、Agent 运行时与安全 fallback

核验日期：2026-09-15。本笔记只把 Artificial Analysis 与 DataCurve DeepSWE 的条目作为模型锚点，再沿 Anthropic 官方模型页、发布页、API 文档和 system card 追踪面试相关技术。官方资料没有公开 Claude Fable 5 的参数规模、网络结构或完整训练报告；运行时协议、产品定位和发布方评测不能反推出这些内部事实。

## 1. 榜单锚点与归并口径

- Artificial Analysis 条目为 [Claude Fable 5](https://artificialanalysis.ai/models/claude-fable-5)，页面配置名是 `Claude Fable 5 (Adaptive Reasoning, Max Effort, Opus 4.8 Fallback)`，canonical slug 为 `claude-fable-5`，页面 `releaseDate` 字段为 `2026-06-09`。
- Artificial Analysis 详情页把该配置的 Intelligence Index 展示为约 50，精确值约为 `49.6994`；同时展示约 60.6 output tokens/s、约 88.23s TTFT、1M context 和约 `$8.75`/Intelligence Index task。它们是 Artificial Analysis 在自身任务和硬件上对“max + fallback”配置的测量，不是裸模型能力、通用延迟或永久价格。
- DataCurve [DeepSWE v1.1](https://deepswe.datacurve.ai/) 当前在线页面仍显示 113 个任务、91 个仓库、5 种语言和统一的 `mini-swe-agent` harness。Fable 5 的 `xhigh` 记录为 316/452 次通过，Pass@1 约 `70% ±3%`，Pass@4 约 88.5%，平均成本约 `$13.41`，输出约 80K tokens，约 68 个 Agent steps。
- 两个榜单的配置并不相同：Artificial Analysis 的条目是 `max` 且带 `Opus 4.8 Fallback`，DeepSWE 的记录是 `xhigh` + `mini-swe-agent`。不能把 49.7 与 70% 拼成一个排名，也不能把 fallback/工具/验证器的组合结果归因给基础模型。

DeepSWE 的观测对象应写成：

```text
result = F(base_model, model_revision, effort, fallback, mini_swe_agent,
           tools, task_set, verifier, timeout, retry, context_policy, provider)
```

本轮批量联网抓取期间，三个代理 `10.24.27.134:8098`、`10.24.27.134:7890` 和 `10.237.126.170:1234` 均曾成功访问目标页面；后续轻量复验中 `8098` 与 `1234` 对两个排行榜均返回 HTTP 200，`7890` 对 DataCurve 返回 200、对 Artificial Analysis 出现 TLS EOF。当前稳定入口优先使用 `10.237.126.170:1234`。临时文件和哈希如下，哈希只用于这次页面快照的复现线索：

| 页面 | 临时文件 | 大小 | SHA-256 |
|---|---|---:|---|
| Artificial Analysis 中文首页 | `/tmp/artificialanalysis-20260915.html` | 1,772,838 bytes | `1aa28f64778508223642522caffb86da7b9c34395da1030437a5ca1827f369c3` |
| Artificial Analysis Fable 5 详情 | `/tmp/artificialanalysis-claude-fable-5-20260915.html` | 3,531,469 bytes | `a69a5ea342984d9b4054bec82756fab3e67863589e7aaf4b31ca38ce3f9ba4d0` |
| DataCurve DeepSWE | `/tmp/deepswe-20260915.html` | 268,313 bytes | `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7` |

## 2. 官方模型身份与生命周期

主要来源是 Anthropic 的 [Claude Fable 5 模型页](https://platform.claude.com/docs/en/models/fable-5/overview)、[模型说明](https://platform.claude.com/docs/en/models/fable-5/introducing-claude-fable-5-and-claude-mythos-5) 和 [Models Overview](https://platform.claude.com/docs/en/models/overview)。

| 字段 | 官方当前记录 |
|---|---|
| API model ID | `claude-fable-5` |
| 状态 | Active (legacy)；官方页面建议迁移到 Fable 5.1 |
| 发布日期 | 2026-06-09 |
| 退休承诺 | 不早于 2027-06-09 |
| 上下文 / 单次最大输出 | 1M / 128K tokens |
| 输入输出 | 文本、图像 -> 文本 |
| 思考 | `adaptive (always on)`；不能用 `thinking.type=disabled` 关闭 |
| 默认 effort | `high` |
| 可靠知识与训练数据截止 | 2026-01 |
| 基础价格 | 输入 `$10`、输出 `$50` / MTok |
| 接入平台 | Claude API、Amazon Bedrock、Google Cloud、Microsoft Foundry、Claude Platform on AWS |

Fable 5 与 Mythos 5 的关系必须单独说明：Anthropic 官方资料称两者共享同一个 underlying model、规格和价格，但 Fable 5 带有可拒答的 safety classifiers，Mythos 5 面向 Project Glasswing 的受邀防御性网络安全流程，不是普通公开接入的独立模型。不要把 Mythos 5 的能力或安全边界倒灌为 Fable 5 的新架构结论。

## 3. 发布与安全时间线

官方 [发布公告](https://www.anthropic.com/news/claude-fable-5-mythos-5) 给出 2026-06-09 的联合发布，并把 Fable 5 定位为面向 demanding reasoning、软件工程、知识工作、视觉、记忆/长上下文和生命科学研究的通用模型。这些是发布方能力和产品定位描述；公告中的客户反馈、图表和演示不能替代固定条件的独立评测。

官方 [重新部署公告](https://www.anthropic.com/news/redeploying-fable-5) 补充了真实的生命周期事件：

1. 2026-06-12，Anthropic 因美国政府出口管制要求和无法实时可靠验证用户国籍，暂停了 Fable 5 与 Mythos 5 的访问。
2. 2026-06-30，Anthropic 公告称相关限制已解除；2026-07-01 起恢复 Fable 5 的全球可用性，并逐步恢复各云平台接入。
3. 重新部署伴随更新后的 cyber safety classifier；官方说明，触发该类安全判断的请求会被拒绝并改送 Opus 4.8。这个路径解释了 Artificial Analysis 条目中的 `Opus 4.8 Fallback`，但不代表所有任务都会 fallback，也不代表 Opus 4.8 是 Fable 5 的一部分。

官方 system card 为 [Claude Fable 5 & Claude Mythos 5 System Card](https://www.anthropic.com/claude-fable-5-mythos-5-system-card)。本轮已成功下载 PDF，文件约 26.96 MB，SHA-256 为 `f95d413845ad8624f384ba026963f2bad8f10f2626575bb45e823e3c2e0ca`。当前环境没有可用的 PDF 正文提取器，因此只把它登记为官方安全评估来源，不从目录、标题或 `strings` 输出扩写具体安全数值。

## 4. 面试技术主线

### 4.1 Always-on adaptive thinking 与 opaque reasoning state

Fable 5 只有 adaptive thinking 模式。模型可以根据任务决定是否思考以及思考深度，应用通过 `effort` 施加行为上的方向；`adaptive` 是 thinking mode，不是一个 `effort` 值。API 不返回 raw chain of thought：

- `thinking.display="summarized"` 返回可读的 reasoning summary；
- `thinking.display="omitted"`（Fable 5 默认）返回空的 `thinking` 字段，但仍返回 `signature`；
- beta 的 `display="updates"` 可以返回工具调用之间的短进度更新，让界面显示状态而不暴露 reasoning；
- `redacted_thinking` 是另一种加密块，不能只按 `type == "thinking"` 过滤。

`signature` 是服务端用于恢复思考连续性的加密状态，不是给应用解析的思维链。工具调用或多轮续接时，应把收到的 thinking/redacted thinking blocks 原样回传；修改、重排或删掉当前工具回合需要的块会导致协议失败，且将空的 omitted 文本误当成“没有思考”是不正确的。`display="omitted"` 主要减少流式传输延迟，不会免除完整 thinking token 的计费。

面试回答应把三件事分开：

1. thinking 是模型输出协议中的 opaque 状态块；
2. `effort` 是全响应的行为旋钮，影响 thinking、工具调用和函数参数；
3. `max_tokens` 才是单次请求的硬输出上限。

### 4.2 Effort sweep，而不是把档位当成模型版本

Fable 5 支持 `low`、`medium`、`high`、`xhigh` 和 `max`；默认 `high` 与省略 effort 等价。官方把它描述为 intelligence、latency 和 cost 的软权衡：高 effort 可能带来更深思考、更多工具调用和更完整验证，低 effort 往往产生更短的工具调用和更低成本，但不是固定的 token budget。

官方建议从 `high` 开始，对最敏感的长任务测 `xhigh` 或 `max`，对常规子任务测 `medium`/`low`。Fable 5 不支持 Fable 5.1 的 per-message effort beta；在 Fable 5 上切换后续请求的 top-level effort 会重新影响 prompt/cache 前缀。因此应在同一任务集上做 effort sweep，同时记录成功率、thinking tokens、visible output tokens、工具轮次、TTFT/TPOT、总延迟和单位成功成本。

### 4.3 拒答是业务状态，不是 HTTP 错误

Fable 5 的 safety classifier 拒答返回 HTTP 200，消息的 `stop_reason` 为 `"refusal"`，而不是 4xx/5xx。`stop_details.category` 可标识 `cyber`、`bio`、`frontier_llm`、`reasoning_extraction` 或 `general_harms` 等策略类别；`explanation` 是给人看的不稳定文本，不应作为机器解析协议。流式响应中途拒答时，已经收到的 partial output 也必须按不完整结果丢弃。

这会改变监控和重试设计：只看 HTTP error rate 会漏掉一类真实业务失败，应该单独统计 refusal rate、类别、是否成功 fallback、最终 served model 和用户可见结果。拒答前没有生成输出时通常不计 token 费用，但仍计入 rate limit；已经流出的中途拒答则按已使用输入和输出计费。

### 4.4 Fallback 是模型路由协议

Anthropic 文档提供三种 fallback 方式：

- Claude API beta 的 server-side fallback：传 `fallbacks="default"` 或最多三个自定义模型，API 在同一请求中按安全类别执行推荐路由；
- 任意支持平台上的 SDK middleware：客户端收到 refusal 后重试；
- 手动 retry：应用自己选择模型、保存状态并控制副作用。

只有 safety classifier refusal 会触发 server-side fallback；rate limit、overload 或普通 server error 不会自动被这个机制吸收。平台能力也不相同：server-side `fallbacks` 不适用于 Message Batches、Bedrock、Google Cloud 或 Microsoft Foundry，跨平台场景要使用客户端方案。

响应要同时检查顶层 `model`、`usage.iterations` 中的 `fallback_message` 和 `stop_reason`，不能只看请求中的 model 字符串。因为 fallback 可能换了模型，应用还需要重新检查工具权限、上下文限制、图像尺寸、内容策略、超时和输出 schema；fallback 不是“同一个模型继续执行”的保证。

### 4.5 Fallback credit：避免跨模型重写 prompt cache

Prompt cache 按模型隔离。Fable 5 拒答后把相同长前缀送给 Opus 4.8，目标模型的缓存需要重新写入。官方的 [Fallback credit](https://platform.claude.com/docs/en/build-with-claude/fallback-credit) 为手动 retry 提供一次性 opaque `fallback_credit_token`，并通过 `fallback_has_prefill_claim` 指示是否可以带着拒答模型的 partial output 继续。

因此完整链路是：读取 refusal 的 token 和 claim -> 选择 continuation 或原始 body -> 在同一 beta header 下把 token 交给 fallback -> 记录 redemption 是否成功。server-side fallback 和 SDK middleware 会自动应用 credit。面试中应指出，这解决的是跨模型 prompt-cache 写入成本，不是把 KV cache 或 reasoning state 跨模型共享。

### 4.6 Long-horizon prompting 与 Agent harness

Fable 5 专属提示指南把能力提升落到 harness 设计，而不是神秘化成“模型会自动完成一切”：

- 用目标、成功标准、证据要求、权限边界和停止条件表达 outcome；过程步骤可以交给模型选择，但副作用和审批必须由宿主控制；
- 长运行任务要让模型根据真实工具结果核对 progress claims，明确报告已验证、失败、跳过和待核验项；
- 独立任务可以并行派发给 subagents，长寿命 subagent 通过复用上下文和 cache reads 减少重复工作，但 orchestrator 仍要处理超时、取消、结果合并和权限；
- 用 memory system 记录跨会话的可复用经验，而不是把所有历史重新塞入上下文；必要时提供 `send_to_user` 之类的进度工具；
- 高 effort 下要显式限制无关重构、过度规划和未授权操作，避免把能力提升变成额外副作用。

这些是官方 prompting/harness 建议，不是 Fable 5 的网络结构或训练 loss 证据。

### 4.7 Memory tool：just-in-time context retrieval

官方 [Memory tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool) 是客户端执行的文件操作协议。Claude 请求在 `/memories` 下 view/create/update/delete，应用把逻辑路径映射到自己的文件系统或数据库，并把结果作为 `tool_result` 返回。模型不会因为拥有该工具就获得宿主文件系统权限。

面试重点是数据边界：handler 必须拒绝 `/memories` 之外的路径，防止 path traversal；必须做用户/租户隔离、大小和敏感信息控制、并发写入、删除与审计。记忆的价值是按需取回相关文件，降低每一轮 active context，而不是无限扩大 context window。

### 4.8 Programmatic tool calling：把批量工具循环放入代码执行

官方 [Programmatic tool calling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling) 允许 Claude 在 code execution sandbox 中写代码，把工具当作异步函数批量调用、循环、条件过滤和聚合。流程是：模型生成代码 -> 容器执行 -> 工具调用处暂停并返回 `tool_use` -> 应用提供结果 -> 容器继续 -> 只有最终代码输出回到 Claude 的上下文。

关键收益和边界：

- 中间 tool results 不进入模型上下文，减少 token 和模型往返；适合大结果集、批处理和多步检索；
- `allowed_callers`/`caller` 区分 direct 与 code-execution invocation，便于协议和观测，但官方明确提醒它不是安全边界；应用仍必须验证权限和输入；
- 容器有 `container`、`expires_at` 和复用生命周期；暂停的程序化工具调用约 4 分钟后可能超时，空闲容器约 5 分钟回收；
- MCP connector、computer/browser use 等工具集可能不支持 programmatic calling；`strict: true`、强制 `tool_choice` 和关闭并行工具调用也有兼容性限制；
- 官方页面给出约 38% billed input token reduction、20%--40% production traffic typical savings 等自报数字，同时也说明在只有一两个串行工具调用的 τ²-bench 场景可能不降成本，不能把收益宣传成普遍定律。

### 4.9 Compaction、context editing 与 task budget 的三层账本

三者解决的问题不同：

| 机制 | 解决的问题 | 关键协议/边界 |
|---|---|---|
| Server-side compaction | 长对话接近窗口上限时总结旧上下文 | beta `compact-2026-01-12`；默认 trigger 约 150K，最低 50K；返回 `compaction` block，后续必须原样带回 |
| Context editing | 精细删除旧 tool results 或 thinking blocks | beta `context-management-2025-06-27`；服务端清理；清理内容会影响 prompt cache，需看 `applied_edits` |
| Task budgets | 约束完整 Agent loop 的总工作量 | beta `task-budgets-2026-03-13`；包含 thinking、工具调用、工具结果和输出；是 advisory countdown，不是 `max_tokens` 硬限制 |

Compaction 并不是无损 transcript：摘要可能丢掉细节，应用应要求摘要保留已完成动作、关键 ID、活动假设、工具错误、未解决阻塞和下一目标。后续请求要把 `compaction` block 作为 assistant 内容保留；可在摘要处设置 cache breakpoint，并在 system prompt 末尾单独缓存，以减少一次压缩导致的整体重写。

Context editing 的 tool result clearing 会按时间顺序清理旧结果，可设置 `clear_at_least`、`exclude_tools` 和是否清理 tool inputs；清理是服务端行为，客户端可以继续保存完整历史。Fable/Mythos 类的默认 thinking preservation 与手动清理策略要分清，清理 thinking 或修改前缀可能破坏后续 signature 绑定。

Task budget 的 countdown 只对模型可见，应用不能把它当作 API 返回的精确剩余字段。它跨一个完整 Agent loop 计算，但 `max_tokens` 仍是每次请求的硬上限；budget 太小可能让模型提前缩小任务或拒绝开始。评测时应同时记总 token、每请求 max、compaction 次数、cache invalidation、成功率和最终 artifact。

## 5. 评测设计与常见误区

一个公平的 Fable 5 对照至少固定：model snapshot、effort、fallback policy、tool catalog、harness、任务集、verifier、超时、重试、硬件、上下文压缩、缓存策略和 provider。对 `max + Opus 4.8 fallback`，还要记录请求是由 Fable 5 还是 fallback model 服务、拒答类别、fallback 次数和最终答案质量。

可直接用于面试的回答：

1. **Fable 5 的 always-on adaptive thinking 是不是公开了新架构？** 不是。它是 API 的 thinking 模式和行为控制；官方没有公开层数、参数、MoE/稠密结构或训练 recipe。
2. **为什么 `display="omitted"` 仍然会收费？** 它隐藏或不流式传输可读 thinking，只改变可见性和传输延迟；服务端仍生成并计费完整 thinking token，signature 还可能用于后续连续性。
3. **为什么 refusal 是 HTTP 200？** 拒答是模型策略状态，不是传输或服务故障；应用要按 `stop_reason` 和 `stop_details` 处理，并单独监控。
4. **fallback 和 retry 有什么风险？** fallback 可能换模型、换上下文/工具限制和安全边界；要验证 served model、工具 schema、权限、幂等性和最终 artifact，不能盲目重放副作用。
5. **programmatic tool calling 为什么能省 token？** 中间结果在代码执行容器中被过滤和聚合，只有最终结果进入模型上下文；但容器生命周期、权限、工具结果 schema 和等待超时仍由应用负责。
6. **compaction、memory 和 task budget 是否都是“扩大上下文”？** 不是。compaction 压缩历史，memory 把持久知识按需取回，task budget 控制全 loop 的工作量；三者都可能改变成本、缓存和信息保真度。

## 6. 未公开内容与闭环状态

本轮可获取的 Anthropic 官方模型页、发布页、API 文档、prompting 指南和 system card 入口没有给出：

- 参数总量、激活参数、层数、专家数、稠密/MoE 结构或注意力变体；
- 预训练 token、数据配比、优化器、学习率、训练硬件和训练成本；
- SFT、RL/RLVR、verifier、蒸馏或安全后训练的完整配方；
- 可独立复现的 Fable 5 技术报告、完整 benchmark 数据或本文所述系统能力的外部复现。

因此 Fable 5 当前状态为 **资料级闭环**：两个排行榜中的可追溯锚点、官方模型页、发布与重新部署公告、API 周边文档、system card 入口和研究笔记均已具备；但没有独立公开架构/训练报告，不新增 Fable 5 专属架构章节。官方产品页的长任务、视觉、知识工作和安全改进作为发布方声明保留，不升级为本项目独立结论。

## 7. 书系映射

- 第十六册：always-on reasoning、summarized/omitted thinking、signature、effort sweep 与 refusal-aware evaluation。
- 第十七册：tool use、fallback 路由、programmatic tool calling、`caller`/`allowed_callers` 和 memory tool。
- 第二十册：模型、API、harness、权限、进度、重试、compaction、context editing、task budget 和副作用边界。
- 第六册与第二十四册：1M context、thinking/output token、prompt cache、TTFT/TPOT 和单位成功成本。
- 第七册：Artificial Analysis/DeepSWE 配置级评测的固定条件、fallback 归因和统计区间。

## 8. 官方来源清单

- [Claude Fable 5 overview](https://platform.claude.com/docs/en/models/fable-5/overview)
- [Introducing Claude Fable 5 and Claude Mythos 5](https://platform.claude.com/docs/en/models/fable-5/introducing-claude-fable-5-and-claude-mythos-5)
- [Prompting Claude Fable 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5)
- [Thinking](https://platform.claude.com/docs/en/build-with-claude/thinking)
- [Effort](https://platform.claude.com/docs/en/build-with-claude/effort)
- [Refusals and fallback](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback)
- [Fallback credit](https://platform.claude.com/docs/en/build-with-claude/fallback-credit)
- [Memory tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool)
- [Programmatic tool calling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling)
- [Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction)
- [Context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing)
- [Task budgets](https://platform.claude.com/docs/en/build-with-claude/task-budgets)
- [Claude Fable 5 and Claude Mythos 5 announcement](https://www.anthropic.com/news/claude-fable-5-mythos-5)
- [Redeploying Claude Fable 5](https://www.anthropic.com/news/redeploying-fable-5)
- [Claude Fable 5 & Claude Mythos 5 System Card](https://www.anthropic.com/claude-fable-5-mythos-5-system-card)
