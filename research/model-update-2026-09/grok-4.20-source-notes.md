# Grok 4.20：多 Agent 研究运行时、推理状态与工具协议

核验日期：2026-09-20。本笔记只研究已经出现在 Artificial Analysis 的 `Grok 4.20 0309 v2 (Reasoning)`。DataCurve DeepSWE 当前没有精确的 `mini_swe_agent_grok_4_20_*` 行，因此不迁移 Grok 4.5 或 Grok 4.6 的 Agent 结果。xAI 官方文档、模型目录和 release notes 用于核验该榜单候选及扩展面试知识，不作为新的模型发现入口。

## 1. 榜单锚点与证据边界

### Artificial Analysis

[Grok 4.20 0309 v2](https://artificialanalysis.ai/models/grok-4-20) 的当前 canonical 页面标题为 `Grok 4.20 0309 v2 (Reasoning)`。本轮通过 `7890`、`8098` 和 `1234` 三条代理获取 HTTP 200，文件均为 513,564 bytes，SHA-256 均为 `2fc88248faf152f46f659310f300f2fe3e5b84e419babb9c8c7dc3752452cdd8`。

详情页当前字段为：

- `releaseDate: 2026-04-07`；这是 Artificial Analysis 的目录字段，不替代 xAI 的官方发布日期。
- Artificial Analysis Intelligence Index：`25.6550155187053`，且 `intelligenceIndexIsEstimated: true`；页面约显示 26。
- `contextWindowTokens: 2,000,000`；输入为 text/image，输出为 text；参数字段为空，开放性为 proprietary。
- 输入价格 `$1.25/M`、输出价格 `$2.50/M`、缓存命中价格 `$0.20/M`；页面性能数据源标为 first-party provider `SpaceXAI`。
- 输出速度中位数 `97.0079005937196` tokens/s；首 token 时间中位数 `22.709823743` s；端到端响应时间中位数 `27.864043110080814` s。
- 当前 Artificial Analysis 对该页面标记 `deprecated: true`、`deprecatedTo: grok-4-3`。这只是第三方目录的当前状态；xAI 当前文档仍列出 Grok 4.20 的模型名和别名，不能用目录状态替代 API capability probe。

这里存在一个必须保留的版本/提供方差异：Artificial Analysis 记录 2M context，而 xAI 当前模型页与 HTML 模型注册表记录 1M maximum prompt/context。它们可能对应不同的 snapshot、服务配置或第三方目录口径；本项目不把两者强行合并为一个无条件上下文结论。

### DataCurve DeepSWE

[DataCurve DeepSWE](https://deepswe.datacurve.ai/) 本轮通过三条代理均返回 HTTP 200，当前响应为 48,451 bytes，SHA-256 为 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。页面可见的 Grok 配置包括：

- `mini_swe_agent_grok_4_5_high`；
- `mini_swe_agent_grok_4_6_low`、`medium`、`high`、`xhigh`。

没有精确的 `mini_swe_agent_grok_4_20_*` 行。因此本轮不记录 Grok 4.20 的 Pass@1/Pass@4、成本、输出 token 或 Agent steps，也不把 Grok 4.6 的 `reasoning_effort` 配置迁移给 4.20。DataCurve 的每个结果都应绑定 `mini-swe-agent`、任务集、工具、环境、verifier、运行次数和模型配置。

## 2. xAI 官方身份与服务字段

主要来源是 [Grok 4.20 模型页](https://docs.x.ai/developers/models/grok-4.20) 及其 [Markdown 版本](https://docs.x.ai/developers/models/grok-4.20.md)。本轮模型页 HTML 为 396,958 bytes、SHA-256 `7997425c354b938ae78881b6c0b5d6a4efc9c10dfbf687c70828575c80fd3f34`；Markdown 为 1,564 bytes、SHA-256 `99057608aeb1a4e6ace5d87bf6c4b27b3bff4af5248babad475500da31fdc2e3`。

官方 Markdown 页面明确给出：

- 主 reasoning ID：`grok-4.20-0309-reasoning`；常用别名包括 `grok-4.20`、`grok-4.20-reasoning`、`grok-4.20-0309` 和 `grok-4.20-reasoning-latest`，另有 beta/experimental aliases。
- 模态为 text/image input -> text output；页面的 context window 为 1,000,000 tokens。
- 支持 function calling、structured outputs、reasoning 和 Batch API。
- 低于 200K prompt tokens 时，input/cached input/output 价格为 `$1.25/$0.20/$2.50` 每百万 token；达到 200K 后，整次请求按 `$2.50/$0.40/$5.00` 计费，而不是只对超出的部分加价。
- 页面限流为 37 requests/s、10,000,000 tokens/minute；可用区域为 `us-east-1`、`us-west-2`。

模型页 HTML 内的 `__XAI_PUBLIC_MODELS__` 模型注册表还给出同一模型在两个区域的公开服务字段：`maxPromptLength=1,000,000`、37 RPS、1,800 RPM、10M TPM、`longContextThreshold=200,000`、Batch 20% discount，以及 `algorithm: "grokSlop"`。后者只是 xAI 公开注册表字段；没有官方技术报告解释它，不能把名称反推成注意力、路由或训练算法。

注册表还区分了三个可观察的 model ID：

| 服务对象 | 官方 ID/别名 | 公开语义 |
|---|---|---|
| 普通 reasoning | `grok-4.20-0309-reasoning`；别名含 `grok-4.20` | reasoning、function calling、structured output；1M prompt；37 RPS/10M TPM |
| non-reasoning | `grok-4.20-0309-non-reasoning`；别名含 `grok-4.20-non-reasoning` | function calling、structured output；`algorithm` 字段为 `grokSlopNonThinking` |
| multi-agent | `grok-4.20-multi-agent-0309`；文档常用别名 `grok-4.20-multi-agent` | 多 Agent 研究；1M prompt；9 RPS/450 RPM/2.5M TPM |

`grok-4.20-multi-agent` 是一个独立服务对象，不应当把“4/16 个协作 Agent”写成 MoE expert 数量、模型层数或普通 Grok 4.20 的隐藏推理深度。

## 3. 官方发布入口与负面证据

xAI [Release Notes](https://docs.x.ai/developers/release-notes.md) 的 March 2026 条目写明 “Grok 4.20 and Grok 4.20 Multi-agent are live”，确认了这两个服务入口已经进入 API 文档，但没有公开参数量、层数、专家结构、训练数据规模或专属技术报告。本轮 release notes Markdown 快照为 16,645 bytes，SHA-256 `e18c3f31802784e0e0d39624124752bc1eadf8074a9a228cbcb94a8f0092ffa0`。

本轮通过可用代理访问以下可能的专属新闻入口，均为 HTTP 404：

- `https://x.ai/news/grok-4-20`
- `https://x.ai/news/grok-4.20`
- `https://x.ai/news/grok-4-20-multi-agent`

arXiv 的[标题精确检索](https://arxiv.org/search/?query=%22Grok+4.20%22&searchtype=title)返回 `title: "Grok 4.20" produced no results`。全文检索返回 7 篇论文，但它们只是把 Grok 4.20 作为被测模型、评审模型或背景模型，没有检出 xAI 发布的 Grok 4.20 专属技术报告。标题检索快照为 16,431 bytes、SHA-256 `8167aeb1ca952502146faeee421e4739ac320fb6b5e74c4d5d96a4181222b4c1`；全文检索快照为 49,517 bytes、SHA-256 `58ad679c9957f773794daa15c62bf997026636e9e66d9c4b8cc0da42ddabe33a`。

因此当前不能把 Grok 4.6 发布页中的 supplemental training、模型生成 SFT trajectories、agentic RL 或 self-testing 结论改名为 Grok 4.20 的训练事实；也不能因为 AA 有参数/速度/上下文字段就补写内部架构。

## 4. 新技术主线：多 Agent 研究运行时

[Multi Agent 文档](https://docs.x.ai/developers/model-capabilities/text/multi-agent.md)把 `grok-4.20-multi-agent` 定位为 Realtime Multi-agent Research beta。它公开描述的系统流程是：多个专门化 Agent 并行搜索和收集信息、分析与交叉核验、综合结果，最后由 leader agent 负责合成并返回最终答案。这个创新首先是 Agent orchestration/runtime 能力，不是公开的基础模型架构变化。

### 4.1 4 Agent 与 16 Agent

文档明确给出两档协作规模：

| API 表达 | 4-agent 配置 | 16-agent 配置 |
|---|---:|---:|
| xAI SDK | `agent_count=4` | `agent_count=16` |
| Responses/REST | `reasoning.effort=low` 或 `medium` | `reasoning.effort=high` 或 `xhigh` |
| 适用场景 | 快速、聚焦问题 | 多角度、复杂研究 |
| 代价 | 较少 token/延迟 | 更多 token/延迟 |

这里的 `reasoning.effort` 是 multi-agent 服务的 agent-count 映射，而不是普通 reasoning 模型的“思考深度”旋钮。面试中若把 `high` 直接解释为“每个 agent 多想一些”，就混淆了产品协议语义。

### 4.2 Leader、子 Agent 与加密状态

默认返回 leader 的工具调用和最终响应；子 Agent 的中间 reasoning、工具调用和输出不会以可读文本直接暴露。xAI SDK 的 `use_encrypted_content=True` 可以把这些状态以加密内容带回，用于后续多轮上下文恢复。它解决的是状态传输和隐私边界，不等于公开 chain-of-thought，也不等于客户端可以编辑子 Agent 轨迹。

内置工具包括 `web_search`、`x_search`、`code_execution` 和 `collections_search`。开启后，服务器负责执行 Agent loop，工具调用和工具成本还会叠加到多 Agent 的 token 成本上。官方文档同时支持不带内置工具的纯协作模式，说明“多 Agent”与“联网检索”是两个可组合的维度。

### 4.3 研究结果的可信性边界

多 Agent 并行会增加视角和覆盖面，但也会增加：重复检索、来源相关性、互相确认错误、工具成本、上下文汇总损失和 leader 过早收敛等风险。可审计的生产实现应记录：每个子 Agent 的任务分工、输入版本、检索来源、工具调用、失败/重试、汇总依据、最终引用和 verifier 结果，而不是只保存 leader 的最终文本。

## 5. Reasoning、状态、缓存与上下文压缩

### 5.1 普通 reasoning 与 multi-agent 必须分开

xAI 的通用 [Reasoning 文档](https://docs.x.ai/developers/model-capabilities/text/reasoning.md)详细列出的普通 effort 表主要针对 Grok 4.6/4.5；同一文档单独列出 `grok-4.20-multi-agent` 的 effort 是 4/16 agent-count 选择。当前材料足以确认 Grok 4.20 的 reasoning capability 和独立 multi-agent 语义，但没有一份专属页明确列出普通 `grok-4.20-0309-reasoning` 的全部 effort 档位和默认值。本项目不把 Grok 4.6 的 `low/medium/high/xhigh` 表自动迁移给普通 4.20。

### 5.2 Encrypted reasoning state

Reasoning 文档公开 `reasoning_tokens` usage，并允许通过 Responses API 的 `include: ["reasoning.encrypted_content"]` 返回加密 reasoning content。客户端只能把它作为 opaque state 原样回传，不能解析、裁剪或当作可读思维链。工程上要区分：

1. 可见回答与 `reasoning_tokens` 统计；
2. 供模型继续使用的加密协议状态；
3. Responses 的服务端 conversation state；
4. 应用自己的数据库、工具回执和 artifact 状态。

### 5.3 Context Compaction

[Context Compaction](https://docs.x.ai/developers/advanced-api-usage/context-compaction.md)提供 `POST /v1/responses/compact`。它把当前仍能放进 context 的长对话压成一个单独的 `type=compaction` item，其中 `encrypted_content` 是不可解析的 opaque blob。下一次请求必须把整个 output item 原样放在新 user turn 之前；不能删除、重排或手工拼接内部字段。

返回的 usage 还包括 pre-compaction `input_tokens`、compaction `output_tokens` 和 `dropped_message_count`。Compaction 可以降低后续输入成本和 TTFT，但不能挽救已经超出 context limit 的请求；是否值得压缩要比较压缩成本、缓存命中、信息损失、恢复成功率和长任务总成本。该 API 是 xAI serving/runtime 能力，不能写成 Grok 4.20 的神经网络结构。

### 5.4 Prompt caching

xAI 的 [Prompt Caching](https://docs.x.ai/developers/advanced-api-usage/prompt-caching.md)说明连续请求开头完全相同的消息会自动缓存；`x-grok-conv-id` 和 `prompt_cache_key` 可帮助把同一会话路由到有利于命中缓存的服务路径。缓存命中降低的是 provider 侧重复输入计算和计费，不等于永久 GPU KV cache、reasoning state 或“模型记住了用户”。

## 6. 工具协议：服务端工具、宿主函数与 MCP

### 6.1 两类工具与责任分层

[Tools Overview](https://docs.x.ai/developers/tools/overview.md)把工具分成：

- **built-in/server-side tools**：Web Search、X Search、Code Execution、Collections Search 等由 xAI 服务端执行，返回工具事件、结果和适用的 citations；
- **client-side function calling**：模型提出结构化 `function_call`，宿主负责权限检查、执行数据库/API/文件副作用，再用 `function_call_output` 回灌。

因此可靠 Agent 的状态机是：

```text
model proposal
  -> schema/permission/policy check
  -> executor or server-side tool
  -> call_id + result + provenance
  -> model continuation
  -> tests/verifier/artifact gate
```

模型产生 tool call 不代表动作已经执行，服务器成功返回工具结果也不等于最终 artifact 正确。

### 6.2 Function Calling 与并行调用

[Function Calling](https://docs.x.ai/developers/tools/function-calling.md)要求工具名称、描述和 JSON Schema。文档说明 streaming 时 function call 以完整调用在一个 chunk 中返回，而不是跨多个 chunk 拼接；默认开启 parallel function calling，一次响应可能包含多个调用，可用 `parallel_tool_calls=false` 关闭。`tool_choice` 支持 `auto`、`required`、`none` 或指定函数；当前文档还给出单次请求最多 350 个工具定义的边界。

并行调用的面试重点不是“更快”三个字，而是副作用语义：只适合彼此独立、幂等或有事务隔离的工具；若一个调用依赖另一个结果，必须由宿主拆成多轮。重试时要用 `call_id`、幂等键和执行日志防止重复扣款、重复写入或重复提交。

### 6.3 Web Search 与 X Search

- [Web Search](https://docs.x.ai/developers/tools/web-search.md)支持实时搜索和页面浏览；`allowed_domains`/`excluded_domains`各最多 5 个域名，另有网页图片理解和显式图片搜索。引用链接应作为独立证据保存，不应把搜索结果改写为模型的训练知识。
- [X Search](https://docs.x.ai/developers/tools/x-search.md)支持关键词、语义、用户和 thread fetch；可用 `allowed_x_handles`/`excluded_x_handles`（各最多 20 个）、日期范围、图片理解和视频理解过滤。文档的时间敏感提示写明 2026-09-21 起计费口径将改为按返回的 posts/profiles 计费，因此 Agent 成本账应记录文档核验日期。

### 6.4 Code Execution

[Code Execution](https://docs.x.ai/developers/tools/code-execution.md)让模型在服务端 sandbox 中实时写、运行和验证 Python，适合数值计算、统计、数据分析和结果校验。文档列出常用 NumPy、Pandas、Matplotlib、SciPy 等库，但明确这是 sandbox 环境；它不是宿主生产文件系统、网络或数据库权限。评测时仍需记录代码、输入数据、运行输出、超时、依赖和结果 verifier。

### 6.5 Remote MCP 最小权限

[Remote MCP Tools](https://docs.x.ai/developers/tools/remote-mcp.md)支持 Streaming HTTP/SSE；Responses API 配置包括 `server_url`、`server_label`、可选 `server_description`、`allowed_tools`、authorization 和 headers。若省略 `allowed_tools`，远程服务器公开的全部 tool definitions 会注入模型上下文；使用 allowlist 可以同时降低 schema token/context overhead 和误用风险。当前 OpenAI-compatible Responses 接口不支持 `require_approval` 与 `connector_id`，不能按其他 MCP 客户端的字段假设有人工审批语义。

### 6.6 混合工具与 max_turns

[Advanced Usage](https://docs.x.ai/developers/tools/advanced-usage.md)说明 server-side tools 可自动执行，而 client-side function call 会让请求暂停并把控制权交还宿主。`max_turns`限制单次请求内的 assistant/server-side tool turns，不直接限制单个 tool call 数；client-side tool 作为 checkpoint 后，新 follow-up request 会重新开始计数。这个边界适合面试追问：全局 Agent budget、单请求 turn budget、工具调用数量和宿主重试次数是四个不同账本。

## 7. 面试导向的技术拆解

### 问题一：Grok 4.20 的“多 Agent”究竟改变了什么？

可观察的变化是 runtime：一个 leader 调度 4 或 16 个协作 Agent，子 Agent 可并行搜索、分析和交叉核验，再由 leader 综合。公开资料没有证明基础模型使用了新的 MoE、层级注意力或多模型权重融合；回答应把“编排拓扑”和“神经网络架构”分开。

### 问题二：为什么 16 Agent 不一定比 4 Agent 更好？

它增加覆盖面和独立视角，也增加 token、工具调用、延迟、来源相关性和错误汇总风险。应固定问题、来源白名单、工具版本、预算和 verifier，分别测召回、事实一致性、引用覆盖、重复率、最终任务成功和单位成功成本。

### 问题三：2M 与 1M context 如何处理？

把 `source=Artificial Analysis / official xAI`、`snapshot`、`model_id`、`endpoint` 和计费/限流配置写进 capability manifest；启动时做实际上限探测，不把第三方目录数字覆盖官方服务字段。长上下文还要同时记录 cache、compaction、工具 schema/result 和输出预算，不能把 context window 当成可用工作记忆。

### 问题四：多 Agent 的结果如何审计？

为每个子 Agent 保存 role、任务分片、prompt revision、工具调用、来源、失败/重试和中间摘要；leader 只能引用可追踪证据；最终通过独立 verifier、schema 校验、事实交叉检查和 artifact gate。若只保存 leader 最终文本，就无法定位是搜索漏召回、子 Agent 幻觉、汇总丢证据还是 verifier 过宽。

### 问题五：MCP allowlist 为什么同时影响安全和成本？

工具定义会进入模型上下文。全量暴露扩大了可调用面、prompt injection/误调用风险和 schema token；`allowed_tools` 缩小工具集既减少 context overhead，也减少副作用面。但 allowlist 不是最终授权，宿主仍应做租户、资源、参数、审批、速率、审计和幂等控制。

## 8. 书系映射与闭环状态

1. 第十六册：reasoning token、opaque encrypted state、普通 effort 与 multi-agent agent-count 的语义差异。
2. 第十七册：leader/sub-agent DAG、server-side/client-side tools、parallel function calling、Remote MCP allowlist、权限、重试和 verifier。
3. 第二十册：搜索来源、引用、代码执行 sandbox、研究 Agent 的证据链和多 Agent 评测。
4. 第二十四册：1M/2M context discrepancy、prompt cache、compaction、TTFT/TPOT、tool-turn budget、区域限流和单位成功成本。
5. 第七册：AA Intelligence Index 与 DataCurve DeepSWE 的配置/harness 证据分层；Grok 4.20 没有精确 DataCurve 行，不迁移相邻版本结果。

当前状态：**资料级闭环（AA 单榜）**。已具备精确 Artificial Analysis 条目、DataCurve 精确行缺失证据、xAI 模型页/注册表、multi-agent/reasoning/compaction/tools/release notes 和论文负检索；没有独立 Grok 4.20 技术报告、公开权重、参数架构、完整训练/后训练 recipe、生产 kernel、硬件 profiling、线上 acceptance rate 或独立 Agent benchmark。因此不新增重复 Transformer 架构章节，先复用既有 reasoning、Agent serving、工具协议和评测章节。

## 9. 来源清单

### 榜单

- [Artificial Analysis: Grok 4.20 0309 v2](https://artificialanalysis.ai/models/grok-4-20)
- [DataCurve DeepSWE](https://deepswe.datacurve.ai/)

### xAI 官方文档

- [Grok 4.20 模型页](https://docs.x.ai/developers/models/grok-4.20) / [Markdown](https://docs.x.ai/developers/models/grok-4.20.md)
- [Reasoning](https://docs.x.ai/developers/model-capabilities/text/reasoning.md)
- [Multi Agent](https://docs.x.ai/developers/model-capabilities/text/multi-agent.md)
- [Context Compaction](https://docs.x.ai/developers/advanced-api-usage/context-compaction.md)
- [Release Notes](https://docs.x.ai/developers/release-notes.md)
- [Tools Overview](https://docs.x.ai/developers/tools/overview.md)、[Function Calling](https://docs.x.ai/developers/tools/function-calling.md)
- [Web Search](https://docs.x.ai/developers/tools/web-search.md)、[X Search](https://docs.x.ai/developers/tools/x-search.md)、[Code Execution](https://docs.x.ai/developers/tools/code-execution.md)
- [Remote MCP](https://docs.x.ai/developers/tools/remote-mcp.md)、[Citations](https://docs.x.ai/developers/tools/citations.md)、[Advanced Usage](https://docs.x.ai/developers/tools/advanced-usage.md)
- [Prompt Caching](https://docs.x.ai/developers/advanced-api-usage/prompt-caching.md)

### 论文与代码检索

- [arXiv 标题精确检索："Grok 4.20"](https://arxiv.org/search/?query=%22Grok+4.20%22&searchtype=title)：本轮无精确标题结果。
- [xAI Python SDK](https://github.com/xai-org/xai-sdk-python)：官方 SDK 入口；用于确认多 Agent、工具和 Responses 的调用生态，不等于 Grok 4.20 权重或训练实现。

