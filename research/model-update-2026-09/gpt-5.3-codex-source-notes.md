# GPT-5.3 Codex：模型、Codex harness 与长任务状态协议

核验日期：2026-09-18。本笔记只处理已经出现在 Artificial Analysis 中的 `GPT-5.3 Codex`。Artificial Analysis 用于发现候选；OpenAI 官方模型页、Codex Prompting Guide 和 API 运行时文档用于核验接口与周边技术。当前没有把相邻 GPT-5.x、其他 Codex 版本或 Agent harness 的评测结果迁移给这个锚点。

## 1. 两个排行榜中的锚点身份

| 来源 | 当前证据 | 解释边界 |
|---|---|---|
| [Artificial Analysis GPT-5.3 Codex](https://artificialanalysis.ai/models/gpt-5-3-codex) | 页面标题为 `GPT-5.3 Codex (xhigh)`；页面 release date 字段为 `2026-02-05`；Intelligence Index `32.5028174368983`，标为 estimated；context `400,000`；knowledge cutoff `2025-08-31`；proprietary，参数字段未公开 | 这些是第三方目录/配置字段，不能替代 OpenAI 的模型接口文档，也不能从指数或参数空值反推架构 |
| Artificial Analysis 详情快照 | `/tmp/gpt53codex-aa-detail-20260918.out`，`3,528,646` bytes，SHA-256 `565d91572b1a0bd9fb8f7f89f16c8beefbaadfdea79de5b229a9bd5a997b27ef` | 用于固定本轮页面身份；详情页的 effort 行仍属于配置，不是独立基础模型 |
| [DataCurve DeepSWE](https://deepswe.datacurve.ai/) | 当前快照中没有精确的 `mini_swe_agent_gpt_5_3_codex_*` 行 | 不能借用 GPT-5.5、GPT-5.6、GPT-5.4、GPT-5.3 Codex 以外的 Codex 或其他 GPT 配置的 Pass@1、成本、输出 token 或 Agent steps |
| 两榜单恢复快照 | Artificial Analysis `/zh`：`1,769,512` bytes，SHA-256 `d50456b4597b637829b46332b3991a8f6a004507341c3fc4d74bc390f42f3cff`；DataCurve：`268,571` bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870` | 证明本轮排行榜入口可访问；不能把页面集合或目录字段写成官方发布事实 |

因此，本轮对象是“Artificial Analysis 已发现、DataCurve 当前没有精确行的 GPT-5.3 Codex 配置锚点”。它可以进入官方资料核验，但不能因为官方文档中存在更宽泛的 Codex 介绍而新增其他模型候选。

## 2. OpenAI 官方模型页确认的接口事实

官方模型页：[GPT-5.3-Codex](https://developers.openai.com/api/docs/models/gpt-5.3-codex.md)。页面将它定位为当前最强的 agentic coding model，并说明它面向 Codex 或类似环境中的 agentic coding task。

已确认字段：

- 模型 ID 和默认 snapshot 都是 `gpt-5.3-codex`；公开 snapshot 列表没有另一个日期锁定版本。
- 输入支持 text、image，输出为 text；支持 `low`、`medium`、`high`、`xhigh` reasoning effort。
- context window 为 `400,000`；maximum input 为 `272,000`；maximum output 为 `128,000`。三者不能相加成可用并发，也不能把 maximum input 写成整个 context 的输入上限。
- 页面列出 knowledge cutoff `2025-08-31`。这不是 release date，也不是实时知识保证。
- 页面列出 proprietary，未给出参数规模、激活参数、层数、专家数或公开权重。
- API 端点只支持 Responses；Chat Completions、Batch、Fine-tuning、Realtime、Assistants 和其他表中端点均标为不支持。
- 模型页列出 streaming、structured outputs、function calling、image input、web search 和 prompt caching；Responses tools 列出 function calling、web search、hosted shell、skills。
- 本轮页面价格字段为 input `$1.75/M`、cached input `$0.175/M`、output `$14/M`。价格属于页面快照，实际接入应重新读取价格页和账户限额。

这里必须拆开三层含义：模型页的能力开关描述 API contract；Responses tool 列表描述宿主可以接入的工具面；真正的 shell、skill 或函数副作用仍由宿主执行器、权限、沙箱、审批、超时和审计层负责。

## 3. Codex Prompting Guide 暴露的工程技术线

官方 [Codex Prompting Guide](https://developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide.md) 不是 GPT-5.3 Codex 的架构论文，但它明确说明了模型与 harness 如何协同：

1. 模型强调更少的 reasoning tokens 和更高的长任务自治。官方建议交互式任务优先用 `medium` 平衡智能与速度；困难任务再考虑 `high` 或 `xhigh`。这仍是请求级 effort 选择，不是五个不同 checkpoint。
2. First-class compaction 被视为多小时 reasoning 和连续会话的基础能力。有效任务长度因此由模型预算、上下文状态、工具回执和压缩恢复共同决定，而不是只看 context window。
3. Codex harness 的关键提升点包括 autonomy/persistence、代码库探索、工具使用、`apply_patch`、固定 `workdir`、清晰的工具 schema、并行工具调用和恢复策略。模型质量不能脱离这些接口设计单独解释。
4. 指南建议迁移时避免要求模型在 rollout 前反复输出完整计划、preamble 或状态更新，否则可能让长任务提前停在“计划已完成”而没有执行。这里是 harness prompting 的建议，不能与应用层的可观测进度事件混为一谈。
5. 公开提示词和工具契约强调目标、成功标准、权限、依赖、停止条件、验证和 artifact，而不是把每一步都硬编码进 prompt。面试中应把它归类为 outcome-oriented agent design。
6. Codex 轨迹中的 assistant `phase` 用于区分中间 commentary/preamble 与最终 `final_answer`。它是输出协议状态，不是 chain-of-thought 暴露，也不是新的 attention 结构。

对面试最有价值的结论是：GPT-5.3 Codex 的“能力”至少要按 model、Responses protocol、Codex harness、tool executor、workspace 和 verifier 分层描述。只报告模型名称而不报告 harness，会把系统效果错误归因给裸模型。

## 4. Responses 状态、replay 与 compaction

### 4.1 完整 output item replay

官方 [Conversation state](https://developers.openai.com/api/docs/guides/conversation-state.md) 和相关 Responses 资料要求无状态 replay 时保留完整 output items。对 GPT-5.3 Codex，至少要区分：

- 可见 assistant message、tool call 和 tool result；
- 加密/opaque reasoning item；
- assistant `phase`；
- compaction item；
- 工具集合、权限决定和最终 artifact 引用。

只保存最终可见文本，可能丢掉下一轮生成所需的状态。`previous_response_id` 或 Conversations API 可以引用服务端状态，但它不是可读思维链，也不表示历史输入免费；无状态数组链路和服务端状态链路的恢复规则必须分别测试。

### 4.2 Server-side 与 standalone compaction

官方 [Compaction](https://developers.openai.com/api/docs/guides/compaction.md) 文档给出两条路径：

- server-side compaction：在请求中设置 `context_management` 与 `compact_threshold`，跨过阈值后服务端在流中返回 encrypted compaction item，并继续使用压缩后的状态；
- standalone `/responses/compact`：提交仍在窗口内的上下文，取得下一轮应直接使用的 canonical compacted context。

standalone 返回值不是可任意编辑的摘要字符串。若应用删除前置 item、工具回执、权限状态或 artifact 元数据，可能破坏恢复。正确的验收对象是“压缩前后的任务状态是否仍可继续”：目标、已执行副作用、未完成副作用、工具集合、权限、错误恢复和最终文件都要能回放。

### 4.3 Compaction 与缓存不是同一层

compaction 改变继续任务所需的状态表示；prompt cache 复用稳定输入前缀的 KV states。压缩后前缀发生变化，可能降低首次 cache hit；反过来，命中缓存也不代表 compaction 状态已经正确。评测至少记录 context tokens、reasoning tokens、visible output、tool rounds、compaction events、cache hit、恢复成功率、重复副作用和单位成功成本。

## 5. 工具契约与工具发现边界

GPT-5.3 Codex 官方模型页明确支持 function calling、web search、hosted shell 和 skills（通过 Responses）。这应拆成以下状态：

```text
model proposes call
  -> schema/parser validates
  -> host permission/approval decides
  -> executor runs in sandbox
  -> result/receipt returns
  -> model continues or finishes
```

模型产生 call 不等于宿主接受 call，宿主接受也不等于外部副作用已经提交。每个 tool call 应记录 schema version、source、arguments hash、permission decision、execution state、timeout/cancel、retry/idempotency、result provenance 和 artifact。

本轮另外读取的 [Tool search](https://developers.openai.com/api/docs/guides/tools-tool-search.md) 文档明确说明当前文档中的 `tool_search` 支持边界从 GPT-5.4 及之后模型开始变化。因此不能把 GPT-5.3 Codex 模型页的 `skills`/hosted shell/function calling 自动升级为 GPT-5.4+ 的 deferred `tool_search` 能力；工具搜索是另一层 schema discovery/protocol capability，需要单独记录模型支持矩阵、namespace、权限和缓存影响。

## 6. Prompt cache 与上下文预算

官方 [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching.md) 将缓存解释为可复用前缀的 KV states，而不是保存一段可以任意拼接的文本。模型 ID、工具名称/描述/schema/顺序、系统指令、消息历史和模板变化都可能破坏完整 rendered prefix 匹配。

对 GPT-5.3 Codex 的服务账本应至少区分：

```text
400K context window
272K maximum input
128K maximum output
reasoning tokens + visible output
tool schema/results + workspace state
cached prefix + uncached suffix
```

不要把 GPT-5.6 或其他更新模型的 cache threshold、TTL、费率和断点规则迁移到 GPT-5.3 Codex。当前能确认的是模型页列出的 input/cached-input/output 价格和官方缓存的前缀匹配语义；未确认的 provider 级线上接受率、硬件 KV bytes 和 kernel 性能保持待核验。

## 7. 适合面试的技术主线

1. **模型与 harness 分层**：GPT-5.3 Codex 的 agentic coding 定位不能单独解释 Codex 的长任务结果；要同时讨论 prompt、工具 schema、workspace、权限、shell、patch、测试和 verifier。
2. **Responses-only 迁移**：没有 Chat Completions/Batch 支持时，adapter 不能只改 model ID；需要迁移消息、tool call、状态引用、stream event、phase 和错误处理。
3. **Reasoning budget 与硬上限**：`reasoning.effort` 是行为/预算控制，`max_output_tokens`、maximum input 和 context window 是不同约束；评测要记录隐藏推理、可见输出、工具和恢复成本。
4. **Phase 是协议状态**：`commentary` 与 `final_answer` 影响 harness 是否继续工具循环；不能把中间进度当最终完成声明，也不能把 phase 当 chain-of-thought。
5. **Compaction 是可恢复状态协议**：opaque encrypted item 和 canonical context 不能按普通摘要自由裁剪；需要做目标、回执、权限、artifact 和重复副作用回归。
6. **Tool contract 是权限边界**：模型输出、宿主授权、工具执行和 artifact 提交是四个事件；schema 正确不等于业务安全，工具搜索到能力也不等于获得权限。
7. **指标必须绑定观测对象**：Artificial Analysis Intelligence Index、DataCurve Pass@1、官方 benchmark 和本地 toy audit 不能拼成一个裸模型分数。

## 8. 未公开内容与书系映射

截至本轮官方模型页、Codex Prompting Guide、Compaction、Conversation state、Tools、Agents 和 Prompt caching 资料，没有确认：

- 参数规模、激活参数、层数、专家数量、稠密/MoE 结构或注意力变体；
- 预训练数据、优化器、硬件集群、训练成本和完整后训练/RL recipe；
- GPT-5.3 Codex 专属 system card、独立技术报告、公开权重或可独立复现的训练 benchmark；
- hosted shell、skills、compaction、phase 背后的模型内部实现、线上 acceptance rate 和目标硬件 profiling。

因此当前状态为**资料级闭环**：有榜单候选、OpenAI 官方模型/API/提示/状态资料、研究笔记和配套映射，但没有足够公开架构/训练证据新增 GPT-5.3 Codex 专属 Transformer 章节。

书系映射：

- 第六册：Responses-only、400K/272K/128K 预算、reasoning/visible output 和 compaction 成本账本；
- 第七册：Artificial Analysis 与 DataCurve 的配置/Agent harness 观测对象边界；
- 第十六册：effort、reasoning token、可见输出和测试时计算预算；
- 第十七册：Codex harness、工具契约、hosted shell/skills、权限和 artifact；
- 第二十册：模型—协议—harness—执行器—verifier 分层、phase、replay、compaction 恢复；
- 第二十四册：multi-turn、tool use、session、prefix/cache、工具等待释放 GPU 和 task-level metrics。

## 9. 官方快照清单

本轮通过 `10.24.27.134:7890` 获取的官方 Markdown 快照如下；哈希只用于固定证据，不代表页面永久不变。

| 页面 | bytes | SHA-256 |
|---|---:|---|
| [GPT-5.3-Codex model page](https://developers.openai.com/api/docs/models/gpt-5.3-codex.md) | 3,580 | `6d228a9894bf1b4c38e029cb283e3bb85d410d8100c262f35c7e7ee4fb3d0b22` |
| [Codex Prompting Guide](https://developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide.md) | 41,694 | `314dec7a7ba1107569eeeba7244c24e90773b1da688d6ce88d89fe3db280b4f8` |
| [Compaction](https://developers.openai.com/api/docs/guides/compaction.md) | 14,272 | `73fd2fd1afd44bd6f29ce7bd86fd0ae3413c98ec00ae5476879e98171b0b60dd` |
| [Conversation state](https://developers.openai.com/api/docs/guides/conversation-state.md) | 28,787 | `c38bcd32ee9ff9904746a5a092c020e30d647d1598418bea93e949cce5021e14` |
| [Tools](https://developers.openai.com/api/docs/guides/tools.md) | 33,282 | `4722fa102070178c1a1d603e718c69575c7c1768fa901218eab359bafdcba341` |
| [Agents](https://developers.openai.com/api/docs/guides/agents.md) | 5,432 | `df4f61b609550619a3f9e445c66d28318b9f681affe1a819ef0098cde32bf43a` |
| [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching.md) | 45,605 | `ecced3016c57dcef3115dbda6f73ba6f153f99856f1dadac46d165c12611ab44` |

`tool_search` 快照为 `/tmp/gpt53codex-tools-20260918.out`，本轮只用于确认其支持矩阵与 GPT-5.3 Codex 的能力边界分开记录；不把 GPT-5.4+ 的 deferred tool loading 迁移给 GPT-5.3 Codex。
