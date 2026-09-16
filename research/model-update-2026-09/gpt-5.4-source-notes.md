# GPT-5.4：1M Context、工具搜索与可压缩 Agent 运行时资料摘记

核验日期：2026-09-15。本笔记只把 GPT-5.4 在 Artificial Analysis 与 DataCurve DeepSWE 中的条目作为模型锚点，再沿 OpenAI 官方模型页、专属指南和 API 文档追踪与面试相关的新技术。官方资料没有公开 GPT-5.4 的参数规模、网络结构或完整训练报告；API 字段、榜单分数和产品指南不能反推出这些内部事实。

## 1. 榜单锚点与归并口径

### 1.1 Artificial Analysis

- 2026-09-15 重新获取 [GPT-5.4 详情页](https://artificialanalysis.ai/models/gpt-5-4)，页面标题为 `GPT-5.4 (xhigh)`，canonical slug 为 `gpt-5-4`，页面 `releaseDate` 为 `2026-03-05`。
- 详情页当前将该配置标记为 deprecated，并指向 [GPT-5.5 (xhigh)](https://artificialanalysis.ai/models/gpt-5-5)；页面还说明 deprecated 模型只继续更新默认 10K input workload 的性能结果，其他 workload 结果属于历史数据。因此不能把当前页面的所有字段误读成持续更新的实时能力。
- 该配置的第三方字段为：Intelligence Index `38.9756`（estimated，页面展示约 39）、输出速度约 `143.44 tokens/s`、median TTFT 约 `93.69s`、context window `1,050,000`、文本/图像输入和文本输出。页面把开放性标为 proprietary，parameters 为 `null`；这不是 OpenAI 的参数规模披露。
- Artificial Analysis 详情页的文本价格字段为输入 `$2.50`、缓存输入 `$0.25`、输出 `$15.00`/每百万 token；价格、速度、TTFT 和指数都绑定页面配置、provider 与测量时点。
- 同一历史候选表还记录了 GPT-5.4 的 `low`、`Non-reasoning`、GPT-5.4 Pro `xhigh`，以及 2026-03-17 的 GPT-5.4 mini/nano 变体。它们保留为同一产品家族的不同配置/服务档位；本轮的主研究对象是 `gpt-5.4`，不把每个 effort 或小型号自动当成新的基础模型。

详情页快照：`/tmp/1234-aa-gpt54-20260915.html` 和 `/tmp/8098-aa-gpt54-20260915.html` 均为 `3,488,457` bytes，SHA-256 均为 `3438103fd097715e5a39d9ee7b3bc0710c23a251c013da3c1b9f8e05a35ada5e`。两个代理取得相同字节，说明本轮的详情页证据可复核；首页复验快照 `/tmp/aa-home-1234-20260915-r3.html` 与 `/tmp/aa-home-8098-20260915-r3.html` 均为 `1,773,715` bytes，SHA-256 为 `4e089738feed2330ffce50ba7cd4d58141641e647ddbf0e50e1d8c4a3b975cf0`。

### 1.2 DataCurve DeepSWE

DataCurve DeepSWE v1.1 的 2026-09-03 页面快照包含：

| 页面配置 | effort | 通过/尝试 | Pass@1 | Pass@4 | 平均成本/任务 | 平均输出 token | 平均 Agent steps |
|---|---|---:|---:|---:|---:|---:|---:|
| `mini_swe_agent_gpt_5_4_xhigh` | `xhigh` | 234/452 | 51.7699% | 77.8761% | `$5.6525` | 71,408.87 | 70.47 |

该行还记录了 113 个任务、4 次整套运行、95% run-to-run 区间约 `±1.5021` 个百分点、平均输入约 9.42M tokens 和平均 cache tokens 约 8.61M。页面整体范围为 113 个任务、91 个仓库、5 种语言，统一 harness 为 `mini-swe-agent`，结果由工具、环境和 verifier 共同产生。

因此 DeepSWE 的观测对象应写成：

```text
result = F(base_model, model_revision, effort, mini_swe_agent, tools,
           task_set, verifier, timeout, retry, context_policy, provider)
```

上面的 51.7699% 是 `gpt-5-4` + `xhigh` + `mini-swe-agent` + 工具/环境/verifier 的系统结果，不是 GPT-5.4 裸模型分数；成本、输出 token 和 Agent steps 也不能独立解释质量。DataCurve 当前快照没有 GPT-5.4 Pro、mini 或 nano 的主表行，不能将 base 行迁移给这些变体。

DataCurve 快照：`/tmp/1234-deepswe-20260915-r3.html` 和 `/tmp/8098-deepswe-20260915-r3.html` 均为 `268,313` bytes，SHA-256 均为 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。该页面的生成时间和排行榜更新时间仍属于 2026-09-03 快照，不是 2026-09-15 的新一轮 benchmark。

## 2. 官方模型身份与接口字段

主要依据是 [GPT-5.4 模型页](https://developers.openai.com/api/docs/models/gpt-5.4.md)、[Using GPT-5.4](https://developers.openai.com/api/docs/guides/latest-model/gpt-5.4.md) 和 [OpenAI Models](https://developers.openai.com/api/docs/models.md)。

| 模型/档位 | 官方定位 | 默认 snapshot | 输入/输出 | context / max output | reasoning effort |
|---|---|---|---:|---:|---|
| `gpt-5.4` | 复杂专业工作的旗舰通用模型，覆盖编码、推理、写作和工具使用 | `gpt-5.4-2026-03-05` | text/image → text | 1,050,000 / 128,000 | `none`（默认）、`low`、`medium`、`high`、`xhigh` |
| `gpt-5.4-pro` | 使用更多计算、面向更难问题的服务档位 | 由专属模型页定义 | 由专属模型页定义 | 同家族长上下文方向 | 不能直接用 base 的字段替代 |
| `gpt-5.4-mini` / `gpt-5.4-nano` | 更高吞吐、更低成本的编码或 Agent 工作负载 | 各自专属页面定义 | 家族指南给出用途差异 | 不能用 base 行自动迁移 | 作为关联变体单独核验 |

GPT-5.4 模型页还确认：

- 支持 Chat Completions、Responses 和 Batch；不支持 Realtime、Assistants、fine-tuning、embedding、音频、视频和图像生成等该页列出的其他端点。
- 支持 streaming、structured outputs、function calling、file search、file uploads、image input、web search 和 prompt caching；Responses 工具列表包括 `function_calling`、`web_search`、`file_search`、`tool_search`、`code_interpreter`、`hosted_shell`、`apply_patch`、`skills`、`computer_use` 和 MCP。
- 页面公开的是模型与 API 的兼容入口。工具的实际执行、网络/文件权限、审批、沙箱、超时、重试和回滚仍属于宿主运行时，不能因为模型页列出 `hosted_shell` 或 `computer_use` 就推断模型拥有任意主机权限。
- 官方模型页的知识截止字段为 `2025-08-31`；知识截止日期、排行榜 `releaseDate` 和 snapshot 日期承担不同证据责任，不能相互替代。

模型页快照：`/tmp/7890-openai-gpt54-md-20260915.html` 与 `/tmp/7890-platform-gpt54-20260915.html` 内容相同，均为 `4,196` bytes，SHA-256 为 `c30e86b38bc6ceacb3d6db269aa6ba4a09c3c5e1322bfaf90f924fddce4013a5`。本轮官方页面通过用户提供的 `7890` 代理获取；后续若代理恢复，应复验 snapshot 和价格字段。

## 3. GPT-5.4 的新技术与面试主线

### 3.1 1M context 不是单一容量数字

GPT-5.4 官方指南将 1M token context 描述为该代新增能力，适合把大型代码库、长文档集合或较长 Agent trajectory 放进单次请求。面试时要把三个概念分开：

1. `context window` 是单次请求可用的总窗口，输入、可见输出和 reasoning tokens 都会占用它。
2. `max output tokens` 是输出上限，不等于可以额外获得一个 128K 窗口。
3. 长上下文的可用性、延迟、缓存命中和单位成功成本是不同实验指标，不能由“能塞进去”推出“更便宜”或“更准确”。

官方模型页给出的价格快照是输入 `$2.5`、cached input `$0.25`、输出 `$15`/每百万 token。对 GPT-5.4 及 Pro，输入超过 272K 时，standard、batch 和 flex 的**整次 session**按输入 2 倍、输出 1.5 倍计价；这不是只对超过 272K 的那一段加价。区域处理还可能有 10% uplift。应把长上下文账本拆为：

```text
C_request = uncached_input + cached_input + output
            + tool_calls + service_or_region_surcharge
```

这个式子是面试用的成本分层，不是完整平台账单公式。真实实验必须固定 snapshot、effort、工具、输入长度、缓存策略、provider、服务模式和任务 verifier。

### 3.2 `reasoning.effort` 与 `text.verbosity` 是两个旋钮

`reasoning.effort` 控制模型在生成回答前投入多少 reasoning tokens。GPT-5.4 的 `none` 是默认低延迟档；需要更强的规划、搜索或多步工具决策时，可以逐步比较 `low`、`medium`、`high` 和 `xhigh`。官方指南强调应按任务形状和 eval 选择档位，而不是凭直觉认为 effort 越高越好。

`text.verbosity` 控制最终可见输出的长度和风格，GPT-5.4 默认 `medium`，可以使用 `low` 产生更紧凑的回答或代码。它不等于禁止模型推理：降低 verbosity 不必然减少 reasoning，增加 effort 也不必然让最终答案更长。评测至少分开记录 reasoning tokens、visible output tokens、端到端延迟和任务成功率。

当 GPT-5.4 的 effort 为 `none` 时，官方指南允许使用 `temperature`、`top_p` 和 `logprobs`；其他 reasoning effort 不应带这些字段。迁移时可用 `reasoning.effort` 控制推理深度、`text.verbosity` 控制可见表达、`max_output_tokens` 控制共享硬上限，不能把三个参数混作一个“模型聪明程度”开关。

### 3.3 Deferred `tool_search`：把工具面治理变成运行时状态

GPT-5.4 是 OpenAI 文档明确支持 `tool_search` 的起点。工具搜索的核心不是让模型获得更多权限，而是把大型工具目录从“请求一开始就注入所有 schema”改成“需要时发现并加载”：

1. 在 `tools` 中加入 `tool_search`。
2. 对不常用的 function 或 MCP server 设置 `defer_loading: true`。
3. 请求初始上下文只保留 namespace/server 的高层描述，或保留 deferred function 的名称和描述。
4. 模型运行时搜索相关工具，加载定义后再进行 function call。
5. 宿主负责真正执行、授权、审计、错误处理和结果回传。

官方建议优先用 namespace 或 MCP server 组织工具，并让每个 namespace 少于 10 个函数；namespace 的描述要能帮助模型判断里面有什么。Hosted tool search 由 OpenAI 在请求中搜索已声明的目录；client-executed tool search 则由模型发出 `tool_search_call`，应用自己查目录并返回 `tool_search_output`。

工具定义动态加载的收益是减少初始 token、降低工具选择噪声、改善大型工具面的延迟和缓存前缀；代价是多出了发现状态、schema 版本、权限审计和“搜索失败后如何恢复”的状态机。把 schema 延迟加载写成“模型自动拥有工具”是错误的安全结论。

### 3.4 Built-in computer use 与 build-run-verify-fix

官方 GPT-5.4 指南把 computer use 描述为：模型检查截图并返回结构化动作，由应用自己的 harness 执行。它适合浏览器/桌面这类人类可以通过界面完成的任务，例如导航、填表和验证变更是否生效。

面试回答应拆成三层：

- 模型层：从视觉输入生成动作意图或结构化 action。
- harness 层：把 action 翻译为浏览器/桌面操作，回传截图和错误。
- 安全层：隔离浏览器或 VM，对高影响动作保留人工确认，并限制网络、凭据和副作用。

前端专属指南进一步建议让模型先生成 mood board 或视觉选项，再选择资源；让它结合 Playwright 检查多个 viewport、导航流程、状态和渲染结果，形成 build—run—verify—fix 循环。这些是官方工程指导和模型训练目标描述，不是对内部视觉 encoder 或训练 loss 的披露。

### 3.5 Custom tools、CFG 和 `allowed_tools`

GPT-5.4 延续 GPT-5 的 custom tools：工具可以接收任意 raw text，而不局限于 JSON，例如代码、SQL、shell 命令、配置文件或长文本。它还支持用 Lark grammar 表达 context-free grammar（CFG），让 custom tool 的输出符合指定语法或 DSL。

这形成一个有用的面试对比：

| 机制 | 解决的问题 | 仍需由宿主负责的部分 |
|---|---|---|
| Structured Outputs | 让结构化输出符合 schema | 事实正确性、业务约束、执行安全 |
| Custom tool | 允许自由文本作为调用参数 | 注入防护、命令验证、资源限制 |
| CFG | 约束自由文本的语法/DSL | 语义合法性、权限、数据库或 shell 副作用 |
| `allowed_tools` | 从全量工具中限制当前可用子集 | 最终授权、审批、审计和执行 |

`allowed_tools` 让应用把“工具宇宙”与“这一轮允许使用的工具”分开：在 `tools` 中声明全部能力，在 `tool_choice` 的 `allowed_tools` 中以 `auto` 或 `required` 限制当前子集。它减少硬编码调用顺序，改善可预测性和缓存稳定性，也把权限边界显式化。

### 3.6 Tool preambles 与 `phase`：可见进度不是 chain of thought

Preamble 是工具调用前的简短、用户可见的意图说明，例如“我先读取日志，再核对修复”。GPT-5.4 指南把它放在 reasoning 之后、真实 tool call 之前，用于提高工具调用的可理解性、调试性和 steering；应用不应把 preamble 当作原始 chain of thought。

长任务或工具密集型 Responses 流程应保留 assistant message 的 `phase`：

| `phase` | 语义 |
|---|---|
| `commentary` | 工具调用前的 preamble 或中间进度 |
| `final_answer` | 已完成工作的最终答复 |

`phase` 不应加在 user message 上。使用 `previous_response_id` 时通常由服务端保留先前状态；手工 replay assistant history 时必须原样保留 `phase`。丢失该字段可能让中间 preamble 被当成最终答案，造成 Agent 提前停止。它是协议状态字段，不是新的 attention 结构或 reasoning loss。

### 3.7 Responses 状态、reasoning item 与 compaction

GPT-5.4 专属指南将 Responses API 作为多步工具任务的优先路径，并说明 Responses 能在轮次之间传递可回放的 reasoning context，从而可能减少重复 reasoning tokens、提高缓存命中并降低延迟。这里应区分：

- 可见对话历史：消息、工具调用和工具结果。
- reasoning item：API 可携带但 opaque/encrypted 的状态，不是公开的原始思维链。
- `previous_response_id` 或 Conversation：服务端状态引用。
- prompt cache：复用相同输入前缀产生的 KV states。
- compaction item：把长历史压缩为后续继续任务所需的机器状态。

Compaction 有两种官方路径：

- server-side compaction：在 Responses 请求中设置 `context_management` 和 `compact_threshold`；跨过阈值后服务端在流中返回 encrypted compaction item。
- standalone `/responses/compact`：应用提交仍在窗口内的完整上下文，取得下一轮应直接使用的 compacted context。

compaction item 是 opaque 的，不要把它当人写的摘要；standalone endpoint 的返回窗口也不能自行删改。stateless input-array chaining 要把 output items（包括 compaction item）接入下一轮；使用 `previous_response_id` 时通常只提交新 user message。压缩能延长有效任务轨迹，但不保证无损记忆，也可能改变 prompt prefix、降低第一次请求的 cache hit，因此要同时测质量、工具错误、重试、token、延迟和缓存。

### 3.8 Prompting 重点：目标、依赖、证据和停止条件

官方 GPT-5.4 指南建议把 prompt 从规定每一步操作，改成定义目标、成功标准、权限和停止条件。适合面试的最小研究契约是：

```text
目标：完成有证据的资料核验。
成功标准：每个结论绑定允许来源；缺证据时明确待核验；
          工具结果经过检查；达到停止条件后结束循环。
执行：先列 3—6 个子问题，再逐个检索，最后解决冲突并综合。
```

对工具任务，仍需显式写出前置依赖、错误恢复和高影响动作前的验证；对研究任务，官方给出的三遍模式是 plan → retrieve → synthesize。Structured Outputs 解决格式一致性，不替代 citation grounding、事实 verifier 或权限门禁。

## 4. 评测设计与面试回答抓手

### 4.1 推荐记录的完整配置

```text
model + snapshot + effort + verbosity + image_detail
+ tool catalog + tool_search mode + allowed_tools + phase replay
+ prompt/cache policy + compaction policy + harness
+ task set + verifier + timeout + retry + provider
```

这份记录同时覆盖两榜单的差异：Artificial Analysis 更偏模型/服务配置的 intelligence、速度、TTFT、context 和价格字段；DeepSWE 观察的是 Agent harness 中的代码任务完成率、成本、token 和 steps。不能用其中一项替代另一项。

### 4.2 典型面试问答

1. **1M context 是否意味着一次请求可以无条件输入 1M，再额外输出 128K？** 不是。context 是输入、输出和 reasoning 的总约束；max output 是另一个上限，长上下文还受 272K 计费/限流分界和实际 provider 行为影响。
2. **`tool_search` 是不是新的模型能力或权限系统？** 它是 deferred tool loading 的协议能力，解决工具 schema 膨胀；真正权限仍由 namespace/MCP 配置、宿主授权、审批和执行器决定。
3. **`phase` 和 `reasoning.effort` 有什么区别？** effort 是推理预算/行为控制，phase 是 assistant 输出处于中间 commentary 还是 final answer 的协议标记。
4. **为什么 computer use 必须有 harness？** 模型只返回动作，浏览器或桌面执行、截图回传、隔离、凭据控制和高影响审批都在应用侧。
5. **CFG 能保证 SQL 安全吗？** 不能。CFG 只能约束语法；SQL 的表权限、注入语义、行级访问和执行资源仍需 parser、allowlist、数据库权限和 verifier。
6. **compaction 是不是把历史总结成一段文本？** 不是。官方返回的是 opaque machine state；应用应按协议原样回传，不能自行编辑后当摘要使用。
7. **DataCurve 的 GPT-5.4 51.77% 能否与 Artificial Analysis 的 38.98 直接比较？** 不能。前者是 xhigh + mini-SWE-agent + 工具/环境/verifier 的系统结果，后者是第三方模型配置指数；指标、任务和观测对象不同。
8. **GPT-5.4 的 `xhigh` 是否证明内部用了某种 MoE 或搜索算法？** 不能。`xhigh` 只说明请求配置；公开资料没有给出参数、层、专家、训练数据或完整 reasoning 实现。

## 5. 未公开内容、证据边界与闭环状态

截至本次实际获取的 GPT-5.4 官方模型页、专属指南、Reasoning、Tools、Tool search、Compaction、Conversation state、Prompt caching、Agents 和 frontend guide，没有找到可用于确认以下事项的 GPT-5.4 专属公开资料：

- 参数总量、激活参数、层数、专家数量、稠密/MoE 结构或注意力变体；
- 预训练数据规模与配比、优化器、学习率、硬件集群或训练成本；
- SFT、RLHF/RLVR、DPO、verifier、蒸馏或安全后训练的完整 recipe；
- 可独立复现的 GPT-5.4 技术报告、system card、公开权重或完整 benchmark 复现。

可以确认的是 OpenAI 公开了 GPT-5.4 的 API snapshot、1M context、reasoning/verbosity 控制、tool search、computer use、custom tools、allowed tools、preambles、`phase`、Responses 状态和 compaction 等模型周边技术。不能从这些产品/运行时字段反推内部架构或训练方法；排行榜中的指数和 DeepSWE 分数也不能替代独立复现。

当前状态：**资料级闭环**。理由是 GPT-5.4 已有两个排行榜的可追溯锚点、官方模型页、专属指南、API 周边文档和研究笔记；由于没有独立架构/训练报告，不新增 GPT-5.4 专属正式章节。

书系映射：

- 第四册：模型家族、snapshot、官方字段与第三方榜单证据边界。
- 第六册与第二十四册：1.05M context、272K 整次请求价格门槛、prompt cache、reasoning/visible output 账本。
- 第七册：Artificial Analysis 与 DeepSWE 的观测对象、配置固定和公平评测。
- 第十六册：reasoning tokens、effort、verbosity 与“更高 effort 不必然更好”。
- 第十七册：tool search、custom tools、CFG、allowed_tools、preambles、Responses tool loop 和 `phase`。
- 第二十册：模型—协议—harness—执行器分层、computer use、compaction、checkpoint、审批与 verifier。

## 6. 来源清单

### 排行榜发现

- [Artificial Analysis GPT-5.4 (xhigh)](https://artificialanalysis.ai/models/gpt-5-4)、[GPT-5.4 low](https://artificialanalysis.ai/models/gpt-5-4-low)、[GPT-5.4 Non-reasoning](https://artificialanalysis.ai/models/gpt-5-4-non-reasoning)、[GPT-5.4 Pro](https://artificialanalysis.ai/models/gpt-5-4-pro)、[GPT-5.4 mini](https://artificialanalysis.ai/models/gpt-5-4-mini) 和 [GPT-5.4 nano](https://artificialanalysis.ai/models/gpt-5-4-nano)：历史表中的配置/服务档位。
- [DataCurve DeepSWE v1.1](https://deepswe.datacurve.ai/)：`mini_swe_agent_gpt_5_4_xhigh`，2026-09-03 页面快照。

### OpenAI 官方资料

- [GPT-5.4 模型页](https://developers.openai.com/api/docs/models/gpt-5.4.md) 与 [OpenAI Models](https://developers.openai.com/api/docs/models.md)
- [Using GPT-5.4](https://developers.openai.com/api/docs/guides/latest-model/gpt-5.4.md)
- [Reasoning models](https://developers.openai.com/api/docs/guides/reasoning.md)
- [Using tools](https://developers.openai.com/api/docs/guides/tools.md)
- [Tool search](https://developers.openai.com/api/docs/guides/tools-tool-search.md)
- [Compaction](https://developers.openai.com/api/docs/guides/compaction.md)
- [Conversation state](https://developers.openai.com/api/docs/guides/conversation-state.md)
- [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching.md)
- [Agents](https://developers.openai.com/api/docs/guides/agents.md)
- [Designing delightful frontends with GPT-5.4](https://developers.openai.com/blog/designing-delightful-frontends-with-gpt-5.4.md)

本轮未把 OpenAI 官方文档中指向的外部链接当作 GPT-5.4 新模型发现入口；它们只作为 GPT-5.4 已有锚点的周边技术提示。后续若需补参数、训练或技术报告，先检查官方新增资料，再在两个排行榜中保持同一锚点和版本语境。
