# 新模型资料来源索引

核验日期：截至 2026-09-23。链接按“发现来源—官方资料—技术论文/实现”组织。榜单名称只用于发现候选，正式结论优先使用官方资料。

## GPT-6 Astra

- 排行榜发现：[Artificial Analysis GPT-6 Astra](https://artificialanalysis.ai/models/gpt-6-astra)；[DataCurve DeepSWE](https://deepswe.datacurve.ai/) 的 `gpt-6-astra` xhigh。2026-09-17 三条代理抓到的 AA 中文首页快照均为 1,782,207 bytes、SHA-256 `d547f7bde6adc164aaea60026d06e9a59178fc86d89c8ce5333fe8d28ffb755a`；DataCurve 均为 268,571 bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，模型集合与既有快照相比无重点厂商新增基础模型。
- 官方模型页：[developers.openai.com/api/docs/models/gpt-6-astra.md](https://developers.openai.com/api/docs/models/gpt-6-astra.md)
- 官方模型指南与运行时资料：[Using GPT-6 Astra](https://developers.openai.com/api/docs/guides/latest-model/gpt-6-astra.md)、[Reasoning](https://developers.openai.com/api/docs/guides/reasoning.md)、[Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching.md)、[Conversation state](https://developers.openai.com/api/docs/guides/conversation-state.md)、[Compaction](https://developers.openai.com/api/docs/guides/compaction.md)、[Tool search](https://developers.openai.com/api/docs/guides/tools-tool-search.md)、[Async tool calling](https://developers.openai.com/api/docs/guides/async-tool-calling.md)、[Mid-turn steering](https://developers.openai.com/api/docs/guides/steering.md)、[Misalignment monitoring](https://developers.openai.com/api/docs/guides/safety-checks/misalignment-monitoring.md)、[Agents](https://developers.openai.com/api/docs/guides/agents.md) 和 [Fast mode](https://developers.openai.com/api/docs/guides/fast-mode.md)。
- 官方开发者博客：[Rethinking skills and prompts for GPT-6 Astra](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra)：技能描述、`AGENTS.md`、任务提示、渐进披露、完成定义和长任务持续性的官方工程建议；博客用于扩展锚点，不作为新模型发现入口。
- 已核验：模型 ID、文本/图像输入、文本输出、1,050,000 context、922,000 maximum input、128,000 maximum output、`low`/`medium`/`high`/`xhigh`/`max` effort、Responses/Chat Completions/Batch 端点、工具支持、动态 `configuration_update`、加密 reasoning item 和通用 assistant `phase` 回放、server-side/standalone compaction、deferred tool search、缓存前缀边界、应用执行的 async function/custom tool、WebSocket `response.steer` 和安全监控覆盖边界。`phase` 当前专节以 GPT-5.5/GPT-5.4 为示例，不把它标成 GPT-6 专属能力。
- 研究笔记：[`gpt-6-astra-source-notes.md`](gpt-6-astra-source-notes.md)；正式章节：[`第六册第 18 章`](../../book-06-llm-deployment/chapters/18-gpt-6-astra长上下文与工具预算.md)。
- 本地教学实验：[`gpt6_agent_protocol_demo.py`](code/gpt6_agent_protocol_demo.py)；只验证 async tool、steering lineage 和 skill progressive disclosure 的状态账本，不调用 API、不代表真实模型性能。
- 待核验：发布日期、参数规模、训练架构、训练/后训练配方、system card、专属技术报告、生产 kernel 和独立 benchmark。

## GPT-5.6

- 排行榜发现：[Artificial Analysis Sol](https://artificialanalysis.ai/models/gpt-5-6-sol)、[Terra](https://artificialanalysis.ai/models/gpt-5-6-terra)、[Luna](https://artificialanalysis.ai/models/gpt-5-6-luna)；DataCurve DeepSWE v1.1 快照包含 [`gpt-5.6-sol`](https://deepswe.datacurve.ai/) 和 [`gpt-5.6-luna`](https://deepswe.datacurve.ai/)。
- 官方模型页：[GPT-5.6 Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol.md)、[GPT-5.6 Terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra.md)、[GPT-5.6 Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna.md)、[OpenAI Models](https://developers.openai.com/api/docs/models.md)
- 官方周边文档：[Reasoning](https://developers.openai.com/api/docs/guides/reasoning.md)、[Agents](https://developers.openai.com/api/docs/guides/agents.md)、[Tools](https://developers.openai.com/api/docs/guides/tools.md)、[Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching.md)、[Compaction](https://developers.openai.com/api/docs/guides/compaction.md)
- 官方开发者博客：[Codex as a platform](https://developers.openai.com/blog/codex-as-a-platform.md)、[Shell + Skills + Compaction](https://developers.openai.com/blog/skills-shell-tips.md)、[One year of Responses](https://developers.openai.com/blog/one-year-of-responses.md)
- 已核验：Sol/Terra/Luna 的官方层级与别名、1,050,000 context、922,000 maximum input、128,000 maximum output、text/image input、text output、reasoning tokens、`none`--`max` effort、Responses 的 `standard/pro` mode、跨轮 `reasoning.context`、工具/端点、272K 整次请求价格阈值，以及 GPT-5.6 专属的 prompt-cache 断点、1,024 token 最小前缀、30m TTL 和缓存费率字段。
- 工程层证据：官方博客把 retained reasoning、context compaction 和 Codex harness 作为系统层能力，展示 GPT-5.6 Sol 在特定 ARC-AGI-3 流程中的组合结果；不把该结果归因于裸模型。
- 待核验：OpenAI 未在本轮官方资料中公开参数规模、稠密/MoE 结构、注意力变体、训练数据、优化器、完整后训练配方、system card 或独立技术报告。研究笔记：[`gpt-5.6-source-notes.md`](gpt-5.6-source-notes.md)。

## GPT-5.5

- 排行榜发现：[Artificial Analysis GPT-5.5](https://artificialanalysis.ai/models/gpt-5-5) 的 `xhigh`、`high`、`medium`、`low` 和 `Non-reasoning` 配置，以及 [GPT-5.5 Pro](https://artificialanalysis.ai/models/gpt-5-5-pro)，榜单日期为 2026-04-23；[DataCurve DeepSWE](https://deepswe.datacurve.ai/) v1.1 快照包含 `gpt-5.5` `xhigh` 配置。
- Artificial Analysis 的历史表另有 [GPT-5.5 Instant (May)](https://artificialanalysis.ai/models/gpt-5-5-instant-05-26) 和 [GPT-5.5 Instant (June)](https://artificialanalysis.ai/models/gpt-5-5-instant-06-26) 条目；当前只有榜单级证据，不自动等同于 OpenAI API 的 `gpt-5.5`。
- 官方模型页：[GPT-5.5](https://developers.openai.com/api/docs/models/gpt-5.5.md)、[GPT-5.5 Pro](https://developers.openai.com/api/docs/models/gpt-5.5-pro.md)、[OpenAI Models](https://developers.openai.com/api/docs/models.md)。
- 官方专属指南：[Using GPT-5.5](https://developers.openai.com/api/docs/guides/latest-model/gpt-5.5.md)；周边 API 文档：[Reasoning](https://developers.openai.com/api/docs/guides/reasoning.md)、[Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching.md)、[Compaction](https://developers.openai.com/api/docs/guides/compaction.md)、[Tools](https://developers.openai.com/api/docs/guides/tools.md)、[Tool search](https://developers.openai.com/api/docs/guides/tools-tool-search.md)、[Images and vision](https://developers.openai.com/api/docs/guides/images-vision.md)、[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs.md)、[Conversation state](https://developers.openai.com/api/docs/guides/conversation-state.md) 和 [Background mode](https://developers.openai.com/api/docs/guides/background.md)。
- 已核验：`gpt-5.5`/`gpt-5.5-pro` 的 snapshot、1,050,000 context、128,000 max output、模态、effort、端点和价格边界；GPT-5.5 专属指南中的高效 reasoning、outcome-first prompting、工具选择、`text.verbosity`、图像 `detail`、Responses `phase` 回放和 compaction；5.5 与 5.6 的 prompt-cache 断点、key、TTL、统计和写入费率差异；Pro 的 Responses/Batch 与 background mode 边界。
- GPT-5.5 Instant 复核：May 条目详情 `3,548,855` bytes / SHA-256 `ed410b6cecbd8eb1dcaf65715547771bcb7435e444f6c5ba7ce815e16614e51e`，June 条目详情 `3,847,333` bytes / SHA-256 `9c6fa94b9c42a89eb7c8f59bad08a30c168bf0860971fbb666c4be47e6a05599`；June 的 AA Index `26.0135173401368`、约 130.9925 output tokens/s、400K context，May 已 deprecated。两个都是 AA 配置字段。
- 官方身份负证据：[GPT-5.5 模型页](https://developers.openai.com/api/docs/models/gpt-5.5.md)确认的是 `gpt-5.5`；本轮访问 [精确 `gpt-5.5-instant` 页面](https://developers.openai.com/api/docs/models/gpt-5.5-instant.md)返回 HTTP 404。DataCurve 当前没有 `gpt_5_5_instant` 精确 `mini_swe_agent` 行，因此不迁移 GPT-5.5 base 的 Agent 结果；Instant June 已完成本轮“榜单配置 + 官方身份负证据”核验，作为历史关联配置保留，不再阻塞下一锚点选择。
- DeepSWE 证据边界：`gpt-5.5` `xhigh` 的 67% ±6%、约 `$7.23`、46K 输出 token、82 steps 是模型配置 + `mini-swe-agent` + 工具/环境/verifier 的组合结果，不是基础模型裸分。
- 待核验：OpenAI 没有在本轮可获取的官方资料中公开 GPT-5.5 专属参数规模、稠密/MoE 结构、注意力变体、训练数据、完整后训练配方、system card 或独立技术报告。研究笔记：[`gpt-5.5-source-notes.md`](gpt-5.5-source-notes.md)。
- 当前状态：资料级闭环。没有独立架构/训练报告，因此暂不新增 GPT-5.5 专属正式章节；技术映射进入 Reasoning、Agent/工具协议、长上下文/Serving、评测和 harness 章节。

## GPT-5.4

- 排行榜发现：[Artificial Analysis GPT-5.4 (xhigh)](https://artificialanalysis.ai/models/gpt-5-4)、[GPT-5.4 low](https://artificialanalysis.ai/models/gpt-5-4-low)、[GPT-5.4 Non-reasoning](https://artificialanalysis.ai/models/gpt-5-4-non-reasoning) 和 [GPT-5.4 Pro](https://artificialanalysis.ai/models/gpt-5-4-pro)，历史榜单日期为 2026-03-05；Artificial Analysis 还记录了 2026-03-17 的 GPT-5.4 mini/nano 变体；[DataCurve DeepSWE](https://deepswe.datacurve.ai/) v1.1 快照包含 `mini_swe_agent_gpt_5_4_xhigh`。
- Artificial Analysis 2026-09-15 详情快照为 `3,488,457` bytes，SHA-256 `3438103fd097715e5a39d9ee7b3bc0710c23a251c013da3c1b9f8e05a35ada5e`；主配置为 `GPT-5.4 (xhigh)`，第三方字段为 Intelligence Index `38.9756`（estimated）、约 `143.44 tokens/s`、约 `93.69s` median TTFT、1,050,000 context，且当前标记 deprecated、指向 GPT-5.5。以上是第三方配置字段。
- DataCurve 快照为 `268,313` bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`；GPT-5.4 xhigh 为 234/452、Pass@1 `51.7699%`、Pass@4 `77.8761%`、平均成本 `$5.6525`、平均输出 `71,408.87` token、约 `70.47` Agent steps。数字绑定 `mini-swe-agent`、工具、环境、任务集和 verifier，不能写成裸模型能力。
- 官方模型页：[GPT-5.4](https://developers.openai.com/api/docs/models/gpt-5.4.md)；专属指南：[Using GPT-5.4](https://developers.openai.com/api/docs/guides/latest-model/gpt-5.4.md)；周边文档：[Reasoning](https://developers.openai.com/api/docs/guides/reasoning.md)、[Tools](https://developers.openai.com/api/docs/guides/tools.md)、[Tool search](https://developers.openai.com/api/docs/guides/tools-tool-search.md)、[Compaction](https://developers.openai.com/api/docs/guides/compaction.md)、[Conversation state](https://developers.openai.com/api/docs/guides/conversation-state.md)、[Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching.md) 和 [Agents](https://developers.openai.com/api/docs/guides/agents.md)。
- 已核验：`gpt-5.4-2026-03-05` snapshot、text/image → text、1,050,000 context、128,000 max output、2025-08-31 knowledge cutoff、`none`--`xhigh` effort、Responses/Chat Completions/Batch、1M 长上下文、272K 整次请求价格阈值、deferred `tool_search`、built-in computer use、custom tools/CFG、`allowed_tools`、tool preambles、assistant `phase`、Responses 状态回放和 native compaction。
- 已核验的面试主线是模型—协议—harness—执行器分层、reasoning/visible output 分账、工具 schema 延迟加载、截图动作执行与 build-run-verify-fix、opaque compaction state、研究 citation contract 和长上下文缓存/成本账本。官方资料未公开参数、架构、完整训练/后训练 recipe、system card、公开权重或独立技术报告。
- 当前状态：资料级闭环。研究笔记：[`gpt-5.4-source-notes.md`](gpt-5.4-source-notes.md)。GPT-5.4/Pro 已具备锚点级资料；mini/nano 已在精确 AA 条目和各自 OpenAI 模型页基础上单独闭环，但仍不迁移 base 的 Agent 评测或工具能力。

### GPT-5.4 mini / nano

- 排行榜发现：[Artificial Analysis GPT-5.4 mini](https://artificialanalysis.ai/models/gpt-5-4-mini) 与 [GPT-5.4 nano](https://artificialanalysis.ai/models/gpt-5-4-nano)，两个条目的 release date 字段均为 `2026-03-17`；DataCurve 当前只有 `mini_swe_agent_gpt_5_4_xhigh` 的 GPT-5.4 base 行，没有 `gpt_5_4_mini`/`gpt_5_4_nano` 精确模型 ID。
- AA 详情快照：mini `3,857,350` bytes / SHA-256 `a01a6da4a38077bc6309333d509bf303f8b1beb36729fd94f79fe7d3a8ffe0ee`；nano `3,859,704` bytes / SHA-256 `2e6ddfefc9a3483437f0fe2648ce23ff42fe4f8614cec0c365dfc69777a166b4`。Intelligence Index、速度、价格、deprecated/迁移提示和 context 目录字段保持 AA 第三方证据边界。
- 官方资料：[GPT-5.4 mini 模型页](https://developers.openai.com/api/docs/models/gpt-5.4-mini.md)、[GPT-5.4 nano 模型页](https://developers.openai.com/api/docs/models/gpt-5.4-nano.md)、[Using GPT-5.4](https://developers.openai.com/api/docs/guides/latest-model/gpt-5.4.md)、[Reasoning models](https://developers.openai.com/api/docs/guides/reasoning.md)。已核验各自 `2026-03-17` snapshot、400K context、272K maximum input、128K maximum output、`none`--`xhigh` effort、定位和官方价格。
- 能力边界：mini 页面当前列出 `tool_search`/`computer_use`，nano 页面当前未列出这两项；不能因同属 GPT-5.4 家族就复用完整 tool catalog。mini/nano 的 prompt contract、task-shape routing、capability probe 和 verifier 主线见 [`gpt-5.4-mini-nano-source-notes.md`](gpt-5.4-mini-nano-source-notes.md)。
- 当前状态：AA 单榜资料级闭环（关联档位）。官方没有公开 mini/nano 参数、内部架构、训练/后训练 recipe、system card、独立技术报告或精确 DataCurve Agent 结果；不新增重复 Transformer 章节。

## GPT-5.3 Codex

- 排行榜发现：[Artificial Analysis GPT-5.3 Codex](https://artificialanalysis.ai/models/gpt-5-3-codex)，页面标题为 `GPT-5.3 Codex (xhigh)`，AA release date 字段为 `2026-02-05`；详情快照 `/tmp/gpt53codex-aa-detail-20260918.out` 为 `3,528,646` bytes，SHA-256 `565d91572b1a0bd9fb8f7f89f16c8beefbaadfdea79de5b229a9bd5a997b27ef`。DataCurve 当前没有精确 `mini_swe_agent_gpt_5_3_codex_*` 行。
- Artificial Analysis 当前字段：Intelligence Index `32.5028174368983`（estimated）、context `400,000`、knowledge cutoff `2025-08-31`、proprietary；参数规模未公开。指数、release date 和目录字段均保持第三方证据边界。
- 官方模型页：[GPT-5.3-Codex](https://developers.openai.com/api/docs/models/gpt-5.3-codex.md)；官方提示资料：[Codex Prompting Guide](https://developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide.md)；运行时资料：[Compaction](https://developers.openai.com/api/docs/guides/compaction.md)、[Conversation state](https://developers.openai.com/api/docs/guides/conversation-state.md)、[Tools](https://developers.openai.com/api/docs/guides/tools.md)、[Agents](https://developers.openai.com/api/docs/guides/agents.md) 和 [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching.md)。
- 已核验：模型 ID `gpt-5.3-codex`、`low/medium/high/xhigh` effort、text/image input、text output、400K context、272K maximum input、128K maximum output、Responses-only、function calling/web search/hosted shell/skills、structured outputs、streaming 和当前页面价格字段。
- Codex 技术主线：较少 reasoning tokens、交互式任务优先 medium、困难任务使用 high/xhigh、长时自治、first-class compaction、Codex harness 的工具 schema/并行调用/`apply_patch`/固定工作目录，以及 assistant `phase` 的中间状态与最终答案区分。
- 状态协议主线：完整 Responses output item replay、加密 reasoning item、`previous_response_id` 与无状态链路的差异、server-side/standalone compaction、canonical context、KV prefix cache 和工具/权限/artifact 账本。`tool_search` 的 GPT-5.4+ 支持边界单独记录，不迁移给 GPT-5.3 Codex。
- 当前状态：资料级闭环；研究笔记：[`gpt-5.3-codex-source-notes.md`](gpt-5.3-codex-source-notes.md)。没有公开 GPT-5.3 Codex 专属参数、架构、训练 recipe、system card、技术报告、kernel 或线上 acceptance rate，不新增重复 Transformer 章节；内容映射到第六、七、十六、十七、二十和二十四册。

## Kimi K3

- 官方发布：[kimi.com/en/blog/kimi-k3](https://www.kimi.com/en/blog/kimi-k3)
- 排行榜发现：[Artificial Analysis](https://artificialanalysis.ai/models/kimi-k3)、[DeepSWE](https://deepswe.datacurve.ai/)。K3 的 `low/max` 是运行配置，DataCurve 的 `mini-swe-agent` 行是 Agent harness 结果，不是新的基础模型。
- 官方仓库：[MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3)、[K3 技术报告 PDF](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)、[K3 License](https://github.com/MoonshotAI/Kimi-K3/blob/main/LICENSE)
- 相关论文：[Kimi Linear, arXiv:2510.26692](https://arxiv.org/abs/2510.26692)
- Attention Residuals：[arXiv:2603.15031](https://arxiv.org/abs/2603.15031)、[官方仓库](https://github.com/MoonshotAI/Attention-Residuals)
- 已核验：2.8T total、104B activated、93 layers、69 KDA + 24 Gated MLA、3:1 KDA/Gated MLA、末尾 Gated MLA、8 个 Block AttnRes/每 block 12 层、896 routed experts/16 selected/2 shared、1,048,576 context、MoonViT-V2、MXFP4/MXFP8 QAT、SiTU-GLU、Quantile Balancing、Per-Head Muon、MoonEP/长轨迹 RL 基础设施和 XTM channel/tool protocol；关联论文仍用于解释通用机制。
- 许可证边界：K3 License 覆盖 weights、parameters、configuration、code 和 documentation，并含商业条件。README 声明完整权重已发布；本轮固定了官方 HF revision `f831ab66814297da540d832a5235f8e904f29d06` 与 `config.json`，但没有下载完整 safetensors，不能写成“本地已下载权重”。
- 实现/部署补证：[HF API metadata](https://huggingface.co/api/models/moonshotai/Kimi-K3)、[固定 `config.json`](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json)、[FlashKDA README](https://github.com/MoonshotAI/FlashKDA)、[vLLM K3 recipe](https://recipes.vllm.ai/moonshotai/Kimi-K3)。FlashKDA master commit 为 `7afb9f454f160a6c4bbc0999beca0a8c40a38934`；README 记录 SM90+/CUDA 12.9+/PyTorch 2.4+、`chunk_kda` backend、recurrent state、H20/GB200 benchmark。vLLM recipe 更新于 2026-09-10，要求 vLLM 0.29.0/K3-enabled nightly，并将 MLA cache 与 KDA state 纳入 hybrid KV manager。
- 2026-09-21 固定的 [safetensors index](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/model.safetensors.index.json) 为 `59,764,096` bytes、SHA-256 `a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd`；审计通过 `497,220` 个 tensor、96 个连续分片、93 个 layer id、92 个 expert-bearing layer、每层 896 experts 和 `weight_packed/weight_scale` 成对映射。`metadata.total_size=1,560,860,324,864` 是 MXFP4 分片 index 口径，不能与 HF API 的 U8/BF16/F32 参数统计 `2,779,931,837,184` 混写。
- 固定 revision 的 `modeling_kimi_linear.py` SHA-256 为 `9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a`；源码把 full-attention 的 `key_cache/value_cache` 与 KDA 的 `conv_states/recurrent_states` 分开维护，单 token decode 走 `fused_recurrent_kda`，chunk/prefill 走 `chunk_kda`，变长输入使用 `cu_seqlens`。零依赖校验器：[`kimi_k3_manifest_audit.py`](code/kimi_k3_manifest_audit.py)。
- 待核验：本地安装 stable wheel、完整训练数据/optimizer recipe、目标硬件 profiling、独立 benchmark 复现和线上 acceptance rate；vLLM `0.29.0` stable release/source entry 已由 PyPI metadata 与 v0.29.0 tag 固定，但 recipe 关于偶发 tool-call parser 不兼容的提示仍必须通过宿主 schema validation、retry、幂等和 verifier 处理。
- 教学落地：[`第十七册第 15 章`](../../book-17-agent-tool-use/chapters/15-kimi-k3发布证据与长任务harness.md)、[`第二十一册第 88 章`](../../book-21-transformer-architecture-evolution/chapters/88-kimi-k3-kda-stable-latentmoe与百万token-agent.md) 与 [`第二十四册第 32.45 节`](../../book-24-llm-inference-engine/chapters/32-multi-turn-tool-use和agent-serving支持.md)。当前状态：内容专题闭环 + 固定 manifest/config/runtime source evidence；新增 SGLang stable/main 对照，不等于完整权重已下载或生产 serving 已验收。

### 2026-09-21 Kimi K3 vLLM 0.29.0 stable artifact recheck

- [PyPI vLLM release metadata](https://pypi.org/pypi/vllm/json) 为 `253,652` bytes、SHA-256 `a232f3e31b111ebfb0d99cbe69b71da682a680ccb4cbf1c57f8cdab3801e5b94`，`info.version=0.29.0`；x86_64 wheel 为 `315,961,042` bytes、SHA-256 `09d48617fc2be9c6cdcd5db480651ab0d84817b257204f2cc2e3ecbb70bbb635`，aarch64 wheel 为 `310,033,787` bytes、SHA-256 `e6b0dfc2b6fd307315e9b34b73cd2bfe7b6b08958eda721828e61732bba426b`，两者均于 2026-09-09 上传。
- 公开 `v0.29.0` tag 的 stable source entry 已固定：`registry.py` 为 `63,102` bytes、SHA-256 `fef8293fe19cef01768a4c5bd8adc05e120202f1c1597eb264d090672894fbd7`；K3 package init 为 `1,444` bytes、SHA-256 `a8ce9cbcd0d8f45f27fe58e70f7dbe080da30886a744f21e60a8ee13ea955a0b`；NVIDIA K3 `model.py` 为 `87,313` bytes、SHA-256 `e74026a83c28cce7ee62889e9ed434d1fff6076dbd263d388aa379f4cf987a4d`。registry 含 K3/MTP/DSpark，package 具备 NVIDIA/ROCm/TPU 分支，NVIDIA model 含 K3 KDA/MLA/MoE 实现入口。
- 证据修正：K3 已不能再写成“stable upstream merge 未证明”或“只有 nightly 入口”。更准确的表述是：vLLM `0.29.0` stable release/source 已有 K3 实现入口；recipe 仍要求 K3-enabled nightly/image 和特定 CUDA/driver 组合，且本轮未安装 wheel、未加载完整 K3 权重、未做目标硬件 profiling、hybrid cache recovery 或线上 tool-call acceptance。

### 2026-09-22 Kimi K3 SGLang main/stable runtime 对照

说明：本节早先的代理失败句子属于首次尝试的历史状态；随后网络恢复并补齐了 commit history，最新结果见紧接其后的 commit history recheck 小节。

- [SGLang `main` K3 text](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_k3.py)：Git blob `383a6f47812bccd1cb91b76814cd0730ff945dd7`，171,101 bytes，SHA-256 `54069412bf3766e974c168985b4e848ecfc29aa4f02dbe27f8e037345faefc0e`。
- [SGLang `v0.5.20` K3 text](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/models/kimi_k3.py)：Git blob `b0ede48c88264d518351a66abf623f1bcf8a730e`，168,114 bytes，SHA-256 `7a3ef867394a2fd52b3a71a979c053b51e9f2310c35cd4c12ef3aa8e7be172e5`；tag commit `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`，发布时间 `2026-09-18T22:41:33Z`。
- [SGLang K3 vision](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/kimi_k3_vl.py) 在 `main`/`v0.5.20` 逐字节一致：Git blob `c423027994645e4a839ec47bd26bcc1b2a7cb012`，32,790 bytes，SHA-256 `2924c38f652a6ebff2ef79c49f2f336ba18723ea4b854d3ac14d95292988acdb`。
- `main` 文本路径显示 DP/SP token shard、MegaMoE/DeepEP/Mooncake/Ascend-FuseEP/MoRI A2A、可配置 shared-expert TP/reduce-scatter、SBO/NPU dual-stream、ModelSlim fused QKVG/packed expert loader 和 KDA fused decode capability gate；`v0.5.20` 已有 K3 基础文本/视觉实现与部分 EP/SBO 路径，但差异不能直接转换为 stable wheel 或目标硬件能力。
- 当前状态：**内容专题闭环 + HF/vLLM + SGLang stable/main source evidence**。完整权重加载、SGLang 依赖安装、KDA/MLA 双状态恢复、视觉数值正确性、目标硬件 profiling、tool/schema/idempotency/verifier acceptance 和生产 SLO 仍待核验。GitHub commit history 因本轮代理连接失败未固定，不把 mutable `main` 写成 release。

### 2026-09-22 Kimi K3 SGLang commit history recheck

- [SGLang commit history API](https://api.github.com/repos/sgl-project/sglang/commits?path=python/sglang/srt/models/kimi_k3.py&per_page=10)：通过 10.237.126.170:1234 获取，51,251 bytes，SHA-256 2f7d8402aa104fe9a5b7839e85c9eb56060e98f7aeb4aecb787005f775351e04，最新记录时间 2026-09-22T06:35:12Z。
- [当前 main K3 text](https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/models/kimi_k3.py)：重新抓取后 171,101 bytes，Git blob 383a6f47812bccd1cb91b76814cd0730ff945dd7，SHA-256 54069412bf3766e974c168985b4e848ecfc29aa4f02dbe27f8e037345faefc0e，与上一轮快照一致。
- 关键 commit：[8ac19cc](https://github.com/sgl-project/sglang/commit/8ac19cc19f8ade51ed203478f17c9a09c706d730) deferred KDA gate；[c4d3770](https://github.com/sgl-project/sglang/commit/c4d3770a6850171beb660576cd003e8001a3f75b) CUDA graph stream；[c2c3629](https://github.com/sgl-project/sglang/commit/c2c3629f2dc0d4fa9386e90ea1a63e6ed5d50580) O(1) expert lookup；[f4c2563](https://github.com/sgl-project/sglang/commit/f4c256354cc8a15d18b11f970f01a82d7394a715) PP/DCP/DSpark；[8ac39c6](https://github.com/sgl-project/sglang/commit/8ac39c66d837f6c91a496ba99a9dba7af1efa894) Ascend A5；[cb32dbc](https://github.com/sgl-project/sglang/commit/cb32dbc9e0c6a236ba6f2be3e9ceded0e8609d71) ROCm KDA input projection。
- 证据解释：commit history 说明 main 仍在修正状态、stream、loader、拓扑和设备分支；不证明 stable release、完整权重、目标硬件或线上 acceptance。

### 2026-09-21 vLLM upstream/runtime recheck

| 证据 | 快照与含义 |
|---|---|
| [vLLM stable supported-models](https://docs.vllm.ai/en/stable/models/supported_models/) | `738,169` bytes；SHA-256 `5420be70a431de9d6e25ba8322c27c1397239fc11643544b354591b8de80d069`；页面更新时间 2026-08-29，列出 `KimiK3ForConditionalGeneration`、`Kimi-K3` 和 `moonshotai/Kimi-K3`。 |
| [vLLM stable K3 API](https://docs.vllm.ai/en/stable/api/vllm/models/kimi_k3/) | `799,430` bytes；SHA-256 `d17b052921ae5ac2216453edf2034854f45a7d1e46a37ef90eff13e25e5cab5b`；页面更新时间 2026-09-09，暴露 `KimiK3ForConditionalGeneration` 与 `KimiK3MTP`。 |
| [vLLM latest supported-models](https://docs.vllm.ai/en/latest/models/supported_models/) | `792,208` bytes；SHA-256 `4604b83e2dffa3d6cf154003bf4aa1a29d3b7d7503ed278b556c90f7c29d098f`；latest 与 stable 单独记录，不互相替代。 |
| [vLLM main registry.py](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/registry.py) | `64,391` bytes；SHA-256 `64c80d8c6659833a9abf836180f2b7549903db2a664e22bc38e38c258a6f572e`；包含 K3、MTP、DSpark 等 registry entries，证明 main 源码入口存在。 |
| [K3 package `__init__.py`](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/kimi_k3/__init__.py) | `1,444` bytes；SHA-256 `a8ce9cbcd0d8f45f27fe58e70f7dbe080da30886a744f21e60a8ee13ea955a0b`；按平台分流 NVIDIA/ROCm，不能替代目标硬件运行测试。 |
| [K3 recipe YAML](https://raw.githubusercontent.com/vllm-project/recipes/main/models/moonshotai/Kimi-K3.yaml) | `19,414` bytes；SHA-256 `5b2bcd21b8dfe210ea3f3d514c6a5448e395847d35fcb1f137d3567233e9d227`；`date_updated=2026-09-10`、`min_vllm_version=0.29.0`，但仍是 `Pre-release`/K3-enabled nightly、CUDA 13/cu130、r580+ driver 路径。 |
| [FlashKDA README](https://github.com/MoonshotAI/FlashKDA) / [Atom](https://github.com/MoonshotAI/FlashKDA/commits/master.atom) | README `4,431` bytes / SHA-256 `fc56ca9a3cd1786d0ff526a12be21500d79ee671e5f0eeb8352fad23d3ba8fb2`；Atom `8,454` bytes / SHA-256 `2155ff08883c6240b3fb53920a8ecdabfd79d2045f41db35d15e4b9d699f07c5`；最新可见 commit 仍为 `7afb9f454f160a6c4bbc0999beca0a8c40a38934`。 |

证据结论：stable 文档/API、v0.29.0 release/source 与 main registry/package 已有 K3 入口，但 recipe 仍标为 pre-release/nightly；本轮没有安装 stable wheel、完整权重加载、目标硬件 profiling、hybrid cache recovery 或线上 tool-call acceptance 证据。不能把 docs/API/source entry 直接升级为生产 serving 验收。

## Kimi K2.7 Code

- 排行榜发现：[Artificial Analysis Kimi K2.7 Code](https://artificialanalysis.ai/models/kimi-k2-7-code)；[DataCurve DeepSWE v1.1](https://deepswe.datacurve.ai/) 的 `kimi-k2-7-code`、`mini-swe-agent`、`reasoning_effort:null` 配置。
- 官方资源：[Kimi K2.7 Code: Open-Source Agentic Coding Model](https://www.kimi.ai/resources/kimi-k2-7-code)；[Kimi API 快速开始](https://platform.kimi.com/docs/guide/kimi-k2-7-code-quickstart)。
- 官方模型卡：[moonshotai/Kimi-K2.7-Code](https://huggingface.co/moonshotai/Kimi-K2.7-Code)；固定 revision：`74797c9c62378b951a1f6fcf5c4631024e9b8bef`。
- 固定实现资料：[config.json](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/config.json)、[Modified MIT License](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/LICENSE)、[部署指南](https://huggingface.co/moonshotai/Kimi-K2.7-Code/blob/74797c9c62378b951a1f6fcf5c4631024e9b8bef/docs/deploy_guidance.md)。
- 已核验：1T 总参数、32B/token active、61 层、1 dense layer、384 routed experts、top-8、1 shared expert、MLA、SwiGLU、256K/262,144 context、MoonViT 400M、native INT4、always-on thinking、`preserve_thinking`、固定采样参数、`auto`/`none` tool choice、`reasoning_content` 回传、图像/视频输入和 vLLM/SGLang/KTransformers 部署边界。
- 官方 benchmark 仅按发布方自报记录；DataCurve 的 138/452、Pass@1/Pass@4、成本和 steps 绑定 `mini-swe-agent`、工具、任务集、verifier 和 4 次运行，不能与 Artificial Analysis 指数拼接成裸模型排名。
- 论文/报告边界：本轮官方资源页、模型卡、定向 arXiv 检索和 MoonshotAI 官方 repository API 查询没有检出 Kimi K2.7 Code 专属论文或独立技术报告；Kimi Linear、Attention Residuals、Mooncake 和 K2 Thinking 只能作为关联路线单独引用。
- 当前状态：资料级闭环；暂无 Kimi K2.7 Code 专属正式章节。完整证据、快照哈希、面试主线和待核验项见 [`kimi-k2.7-code-source-notes.md`](kimi-k2.7-code-source-notes.md)。

## GLM-5.2

- 排行榜发现：[Artificial Analysis GLM-5.2](https://artificialanalysis.ai/models/glm-5-2) 的 `max` 与 `non-reasoning` 条目；[DataCurve DeepSWE](https://deepswe.datacurve.ai/) 的 `mini_swe_agent_glm_5_2_high` 和 `mini_swe_agent_glm_5_2_max`。2026-09-16 AA 与 DataCurve 复验均返回 HTTP 200。
- 快照：AA `/tmp/glm52-aa-20260916.out`，`3,699,404` bytes，SHA-256 `25cf2438443a3ff1dac18964adb79ae98f24f41ed68d02ea4bd38e527a574244`；DataCurve `/tmp/glm52-ds-20260916.out`，`268,313` bytes，SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。
- DataCurve high/max 分别为 164/452、Pass@1 `36.2832%`、Pass@4 `68.1416%`、约 `$2.8355`/task、约 121.88 steps，以及 197/450、Pass@1 `43.7778%`、Pass@4 `76.9912%`、约 `$3.9199`/task、约 129.13 steps。结果绑定 `mini-swe-agent`、4 runs、任务集、工具、环境和 verifier，不是裸模型能力。
- 官方资料：[Z.ai GLM-5.2 文档](https://docs.z.ai/guides/llm/glm-5.2)、[Z.ai 文档索引](https://docs.z.ai/llms.txt)。文档快照 `/tmp/glm52-zai-doc-8098-20260916.out` 为 `525,141` bytes，SHA-256 `d66a48a8a7654f1b033abca4fc9270a57d52961aace20d1335f36e9ab33ba6ba`，`dateModified` 为 `2026-09-03T14:49:41.429Z`。
- 已核验：1M context、128K max output、thinking、function calling、context caching、structured output、MCP，以及面向 project-scale codebase、跨文件重构和 requirements—implementation—verification—deploy 的长周期 Agent 工作流描述。
- 证据边界：官方“lossless context”、数月 Coding Agent 专项训练和 benchmark/开发者案例均按发布方产品描述记录；不能升级为数学无损、完整训练 recipe 或独立复现。GLM-5.2 页面不展开 SAO 定义，现已由 arXiv:2607.07508 补充公开算法；GLM-5.3 专属 compaction 仍待核验。
- 当前状态：资料级闭环；正式专题见第二十一册第 86 章 [`GLM-5.2：IndexShare、MTP 与长轨迹 RL`](../../book-21-transformer-architecture-evolution/chapters/86-glm-5.2-indexshare-mtp与长轨迹rl.md)。研究笔记：[`glm-5.2-source-notes.md`](glm-5.2-source-notes.md)。

- 2026-09-17 补证：此前错误路径 `https://z.ai/blog/glm-5-2` 返回 404；正确博客为 [`https://z.ai/blog/glm-5.2`](https://z.ai/blog/glm-5.2)，正文资源 `glm-5.2-UFbrCk0E.js` 为 46,166 bytes，SHA-256 `c29e51551a1fb0100c681267da8330fa7a29df4e4160382b1b38cd8e7dc594db`，三条代理一致。新增公开线索包括 IndexShare、MTP speculative decoding、长上下文 serving、critic-based PPO 和 coding-agent anti-hack；详见研究笔记与第二十一册第 86 章。

## GLM-5

- 排行榜发现：[Artificial Analysis GLM-5](https://artificialanalysis.ai/models/glm-5) 的 `GLM-5 (Reasoning)`，并列出 `glm-5-non-reasoning`；本轮三条代理均返回 HTTP 200，详情快照为 `3,577,227` bytes，SHA-256 `57dbab2e4ef52e2c95c585bb1d0549f8044486903b77366fed1d5c534569c2a1`。
- DataCurve [DeepSWE](https://deepswe.datacurve.ai/) 本轮三条代理均返回 HTTP 200，快照 SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`；当前只检出 GLM-5.2、GLM-5.3 和 GLM-5.3-Flash 配置，没有精确 `GLM-5` 行。因此 GLM-5 是 AA 单榜发现，不能迁移相邻版本的 DeepSWE 分数。
- 官方资料：[Z.ai GLM-5 博客](https://z.ai/blog/glm-5)、[API 文档](https://docs.z.ai/guides/llm/glm-5)、[Hugging Face 模型卡](https://huggingface.co/zai-org/GLM-5)、[固定 README](https://huggingface.co/zai-org/GLM-5/raw/main/README.md)、[固定 config.json](https://huggingface.co/zai-org/GLM-5/raw/main/config.json)、[官方 GitHub](https://github.com/zai-org/GLM-5) 和 [`slime`](https://github.com/THUDM/slime)。
- 专属论文：[GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/abs/2602.15763)。模型卡和报告确认 744B total/40B active、28.5T 预训练 tokens、DSA、异步 RL 基础设施 `slime` 和长周期 Agent 定位；配置公开 `GlmMoeDsaForCausalLM`、78 层、256 routed experts、top-8、1 shared expert、`index_topk=2048` 和约 202K position。
- 面试主线：DSA 的 indexer/top-k 召回与端到端成本、MoE 总容量与 active compute 分离、rollout/trainer 解耦的 policy lag 与样本新鲜度、长轨迹 Agent RL 的 verifier/credit assignment，以及模型—harness—工具—环境—verifier 的评测分层。发布方 benchmark 不与 AA/DataCurve 拼成裸模型能力。
- 当前状态：内容专题闭环（AA 单榜）；没有精确 DataCurve 行，但已有官方模型卡、专属技术报告、API/部署资料、研究笔记和第二十一册第 89 章。2026-09-20 AA 快照为 `3,811,809` bytes、SHA-256 `0b9c56ff97a87b1dd0a006d1c300057f5ef20c3f65a19bcddf0f6803d8a94f8d`；DataCurve 快照为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，仍无精确 `mini_swe_agent_glm_5_*` 行。Z.ai 博客资源本轮为 `145,189` bytes、SHA-256 `99d27d6132c25e1b39fe26df0605ad1abd855e9b30b098996c64423093fb618f`；文档 Markdown 端点由代理返回 503，只记为线路失败。完整 DSA indexer 训练目标、生产 kernel、硬件 profiling、`slime` 调度与完整训练/后训练 recipe 仍待核验。研究笔记：[`glm-5-source-notes.md`](glm-5-source-notes.md)；正式专题：[`第二十一册第 89 章`](../../book-21-transformer-architecture-evolution/chapters/89-glm-5-dsa-slime与agentic-engineering.md)。

## GLM-5.3

- 官方文档：[docs.z.ai/guides/llm/glm-5.3](https://docs.z.ai/guides/llm/glm-5.3)
- 前代文档：[docs.z.ai/guides/llm/glm-5.2](https://docs.z.ai/guides/llm/glm-5.2)
- 排行榜发现：[Artificial Analysis](https://artificialanalysis.ai/models/glm-5-3)、[DeepSWE](https://deepswe.datacurve.ai/)
- 已核验：GLM-5.3 与 GLM-5.2 的基础模型/后训练关系、推理档位、1M 输入窗口、Agent 环境与验证器描述；SAO 论文的公开算法也已核验。
- 待核验：GLM-5.3 专属 compaction 的状态/序列化、完整技术报告、权重与许可证。
- 官方文档索引快照还列出 GLM-5.3-Flash 专属页面、迁移指南及流式/工具/缓存/结构化输出文档；当前只把它们记录为入口和接入范围，不把索引摘要当作 Flash 模型规格或 benchmark 证据。

## GLM-5.3-Flash

- 排行榜发现：[Artificial Analysis GLM-5.3-Flash](https://artificialanalysis.ai/models/glm-5-3-flash)，canonical slug `glm-5-3-flash`、`max` reasoning effort、页面 `releaseDate` 字段 `2026-08-26`；[DataCurve DeepSWE v1.1](https://deepswe.datacurve.ai/) 的 `mini_swe_agent_glm_5_3_flash_max` 配置。
- Artificial Analysis 本次复验快照：1M context、Intelligence Index `41.907366113455`、median output speed `114.22108687545 tokens/s`、median TTFT `2.45458272199994s`、约 `$0.15/$0.50/$0.026` 每百万 token；快照 SHA-256 `7800ff202ced5e5cc170d7f1858d8070cf8d41c47b3ab6bace60b75c596b6319`。这些是第三方配置字段。
- DataCurve 本次复验快照 SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`；配置为 `max`、`n_runs=4`、284/448、Pass@1 `63.392857%`、Pass@4 `84.955752%`、平均成本约 `$0.2409818562`、平均输出 `72829.77` token、平均 Agent steps `122.89`。结果绑定 `mini-swe-agent`、工具、任务集、环境和 verifier，不能写成裸模型能力。
- 2026-09-21 当前时点复验：Artificial Analysis 详情快照 `3,942,546` bytes，SHA-256 `42f880600d3637489a0c48ff27357fe7986e53510ad5ee016c7d6170bd201fae`；release date 仍为 `2026-08-26`、context `1,048,576`、Intelligence Index `41.807466113455`、median output speed `95.0131572798129 tokens/s`、cost per Intelligence Index task `0.2532595604307378`。与旧快照的指数/速度差异记录为第三方测量漂移，不解释成 checkpoint 或训练变化。DataCurve 精确行和指标未发生模型级变化，当前页面快照 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。
- 官方模型文档：[GLM-5.3-Flash](https://docs.z.ai/guides/vlm/glm-5.3-flash)、[Thinking Mode](https://docs.z.ai/guides/capabilities/thinking-mode)、[Streaming](https://docs.z.ai/guides/capabilities/stream-tool)、[Function Calling](https://docs.z.ai/guides/capabilities/function-calling)、[Context Caching](https://docs.z.ai/guides/capabilities/cache)、[Structured Output](https://docs.z.ai/guides/capabilities/struct-output)、[迁移指南](https://docs.z.ai/guides/overview/migrate-to-glm-new.md)。文档快照 SHA-256 `a127bf7eff2780aacebfc4ffdcadfac5820b75caeaafdb932da0c8942eee879f`。
- 官方博客：[GLM-5.3-Flash](https://z.ai/blog/glm-5.3-flash)：hybrid linear+sparse attention、IndexPool、mHC、30T multimodal corpus、visual self-judgment/test-time improvement、SGLang、ReplaySSM、W8A8、混合 cache quantization、Layer Split、EPD 和约 3× serving 的发布方描述。正文本次由页面 JS 加载，资源 SHA-256 `225196c63b5944629606d26982c0a43c4a8fbd6edb8a7a2e9bf8abfacd35fcbb`。
- 官方 API 文档的当前补证：模型文档 `23,795` bytes/SHA-256 `6abc795542c1eaa546704754b4dcba2160eadec3bdf715875b07105b90e027e9`；Thinking `6,890` bytes/SHA-256 `ce29adde77f3936425663cd346879354eccf9c242b00c72b50d08f21c8bca5d8`；tool streaming `5,370` bytes/SHA-256 `669288770dea7924944918c40a5a29149ec9d8df509e120bae28729eeda105ee`；function calling `18,911` bytes/SHA-256 `d79eaa5d1e3ae0c82e18d8c904da479542efe8b73eb49d755bd091012e3e8cbb`；context caching `17,788` bytes/SHA-256 `fd8593400c204e879c4c26a5f516a31c583c39fb4fd5e07467612c9ae66fd699`。这些快照用于记录 API 版本语境，不把 endpoint 字段升级成模型内部训练事实。
- 官方模型卡：[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a)，固定 revision `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`；[config.json](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/config.json)。README/config 公开 `Glm5NextForConditionalGeneration`、320B/18B、45 层、34 `linear_attention`/11 `deepseek_sparse_attention`、前 3 层 dense MLP、288 routed/top-8/1 shared、1M、IndexPool 和 mHC 字段；模型卡 front matter 标注 MIT。
- 视觉配置公开 24 层 vision module、hidden size 1024、image size 448、patch 14、temporal patch 2、输出投影 4096；输入为 video/image/text/file，输出为 text。官方博客/文档将视觉能力放入 observe—render/use—verify—refine coding loop，不能把它写成完备 verifier 或宿主权限。
- 关联实现：[SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash.md)（页面更新时间 `2026-09-21T12:21:10.188Z`，快照 `247,030` bytes，SHA-256 `fef983feab9a25311de8cf0b62c539b8450ef9acef9015a288ab6639ecd89019`）、[vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash)（2026-09-18 页面快照 SHA-256 `cfd032baa545b0749c171bebac7ad9fab30e3478ef20b5233f1ab44971539103`）、[Transformers GLM5-Next](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/glm5_next.md)、[KTransformers tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/GLM-5.3-Flash-Tutorial.md)。SGLang 记录 paged KV pool 与 KDA state pool、MTP 5/1/6、high-throughput 关闭 speculative、Blackwell/Hopper 的 KV/DSA backend 配对、视频/EPD 和 PD dummy-weight 门禁；vLLM recipe 记录 v0.29.0+、native FP8/MTP、硬件路线与依赖门槛；Transformers 明确不包含 MTP layer。这些都是实现/recipe 证据，不替代目标硬件 profiling。
- `GLM-5.3-FlashX` 是 Z.ai 文档中的关联服务入口，约 200 tokens/s，且不在本轮两个排行榜的新候选中；它不新增模型条目。`FlashX` 的 endpoint 配额/速度不能迁移成 Flash checkpoint 的架构或训练事实。
- 当前状态：内容专题闭环，并完成 serving/runtime 资料级补证；正式落点为 [第二十一册第 84 章](../../book-21-transformer-architecture-evolution/chapters/84-glm-5.3-flash混合注意力与视觉闭环.md)，研究笔记见 [`glm-5.3-flash-source-notes.md`](glm-5.3-flash-source-notes.md)。本地目标硬件 profiling、完整 kernel、PD load/accuracy、MTP acceptance、线上 tool acceptance 和独立 benchmark 仍待核验。
- 待核验：完整训练/后训练 recipe、生产 linear-attention/ReplaySSM/IndexPool/MoE/mHC/EPD kernel、真实 state/KV/indexer bytes、`index_topk=2048` 的最终可见 token 语义、硬件 profiling、线上 acceptance rate、API endpoint 差异和独立 benchmark。GLM-5 技术报告 [arXiv:2602.15763](https://arxiv.org/abs/2602.15763) 是模型卡引用的关联报告，不把其中未明确归属 Flash 的数字直接迁移。

### 2026-09-22 upstream runtime source 对照

- **SGLang stable tag**：[v0.5.20 `glm5_next.py`](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/models/glm5_next.py) 为 `61,466` bytes、SHA-256 `12c5157b07fb7c6d93f34e84c43a37866d2e382e703729e2205aed9f8961f9c2`；对应配置为 `12,026` bytes、SHA-256 `3b3c7aa3ae60e1cf59edf91e1c11a6aa49be7532340c8e2482f7f75ed859f3e0`。tag commit 为 `94602c9c2b7cbdb8efd5c52802dac6a1c180089e`。这证明固定 stable source 有 Flash 模型入口，不证明本机完整权重或目标硬件已运行。
- **SGLang main**：main tree commit `9d58189c12e4e14a7eea20f24f9aa7b17e221778` 的模型文件为 `68,097` bytes、SHA-256 `1cd324533aa0827e7542e27fc39c5901c2330823c4b3a32b79bcb26521dbcfec`。与 tag 相比可观察到 projection fusion、KDA projection/prefill metadata、mHC boundary fusion、AMD FP8/Quark MXFP4 和 B200/H200 gate 演进；`c8eb54c41da1`、`2fa6b94e3440`、`b44e2486824e` 只作为 mutable upstream history 记录。
- **vLLM main**：[`vllm/models/glm5next/`](https://github.com/vllm-project/vllm/tree/main/vllm/models/glm5next/) 已拆出 `attention.py`、`kda.py`、`model.py`、`mtp.py` 和 `common/sparse_indexer.py`。关键源码快照分别为 `24,145`/`a7554347a8e91215a9cf884f74bdcbaa4eb4a3870dbd016994b92d93293b1c57`、`31,136`/`37745b45892cb26d9193160c4f446191f6cc276031fe8f7de5dbb37784cf8e8b`、`51,591`/`9d30ec0bf2eb052b96c6fc995435979993a68cced3376734d0ea72002e0f5dfc`、`17,490`/`db158eec6731fb11e3072cd9d9a76f34f3267007a821277ce3463090eb8f85c0` 和 `6,294`/`a3ab1edda8490b8e21c1c240e07e8c8fcd0bb34246a9ed1f64acfe067d15067c`。实现包含 pool 粒度 indexer cache、未完成 pool 的 tail cache、独立 KDA state 和 MTP top-k/slot 复用。
- **vLLM stable tag 的负证据**：[v0.29.0 tree](https://github.com/vllm-project/vllm/tree/v0.29.0) 快照为 `1,979,822` bytes、SHA-256 `7131879ae9592d90738776d24ae213f789577317b2c5427f350761a87ca05070`，路径搜索未发现 `vllm/models/glm5next/` 或同名专属 common runtime 文件。因此 recipe 的 `v0.29.0+` 是部署门槛/路线声明，不能写成公开 `v0.29.0` tag 已包含 GLM5Next 专属实现。
- **证据层级**：SGLang `v0.5.20` 是固定 stable source entry，SGLang/vLLM `main` 是 mutable upstream implementation evidence，recipe 是特定版本/硬件部署路径；完整权重、目标硬件 profiling、PD/EPD recovery、MTP acceptance、工具/verifier 和生产 SLO 仍是独立门禁。

## DeepSeek V4

- Preview 公告：[DeepSeek V4 Preview](https://api-docs.deepseek.com/news/news260424)
- Pro GA：[DeepSeek V4 Pro GA](https://api-docs.deepseek.com/news/news260813)
- Flash Vision 实验版：[DeepSeek V4 Flash Vision](https://api-docs.deepseek.com/news/news260821)
- API 快速开始：[DeepSeek Quick Start](https://api-docs.deepseek.com/quick_start)
- 官方模型卡：[DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- 技术报告：[arXiv:2606.19348](https://arxiv.org/abs/2606.19348)
- 排行榜发现：[Artificial Analysis](https://artificialanalysis.ai/)、[DeepSWE](https://deepswe.datacurve.ai/)
- 已核验：版本日期、V4-Pro/Flash 参数量和激活量、1M context、CSA/HCA、mHC、Muon、FP4/FP8、GRPO 与 on-policy distillation。
- 待核验：各 benchmark 的独立复现、API 别名的长期稳定策略、不同快照的行为差异。

### DeepSeek V4 Pro 0813 当前活动锚点（2026-09-20）

- 排行榜发现：[Artificial Analysis DeepSeek V4 Pro](https://artificialanalysis.ai/models/deepseek-v4-pro) 的精确标题为 `DeepSeek V4 Pro 0813 (Reasoning, Max Effort)`，release date `2026-08-13`；AA 详情快照 3,934,926 bytes，SHA-256 `b14ce625eb337401f4faa63eed37f1b08fe109a693dcdf2a34e21924b6cddff1`。AA 字段为 Intelligence Index `35.9967791278402`、1,000,000 context、约 1.6T/49B total/active，并标记开放权重、MIT；这些目录和测量字段保持 AA 证据边界。
- DataCurve 精确行：[DeepSWE](https://deepswe.datacurve.ai/) 的 `mini_swe_agent_deepseek_v4_pro_max`，`reasoning_effort=max`，Pass@1 `62.831858%`、Pass@4 `88.495575%`、`n_runs=4`、平均成本 `$1.6660232187`、平均输出 `105998.9` token、平均 `154.71` Agent steps。结果绑定 `mini-swe-agent`、工具、任务集、环境和 verifier，不是裸模型能力。
- 官方资料：[V4 Pro GA 公告](https://api-docs.deepseek.com/news/news260813)、[Quick Start](https://api-docs.deepseek.com/quick_start)、[Responses API](https://api-docs.deepseek.com/guides/responses_api)、[Thinking](https://api-docs.deepseek.com/guides/thinking)、[Tool Calls](https://api-docs.deepseek.com/guides/tool_calls)、[模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)、[配置](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json)和 [技术报告](https://arxiv.org/abs/2606.19348)。
- 已核验：GA 公告的 `low/high/max` reasoning effort、原生 Responses API、Codex 优化；当前 API 名 `deepseek-v4-pro`；Responses stateless 边界（不支持 `previous_response_id`、`conversation`、`background`、`store`，支持 function tools/`apply_patch`，并行工具调用始终开启）；模型卡/报告的 CSA/HCA、mHC、Muon、MoE、FP4/FP8、32T+ 预训练、SFT+GRPO 专家培养与 on-policy distillation；配置的 61 层、384 routed/6 selected/1 shared、`q_lora_rank=1536`、`o_lora_rank=1024`、`index_topk=1024` 和 YaRN factor 16。
- 当前状态：内容专题闭环（双榜锚点）；不新增重复 Transformer 章节，复用第二十一册第 77/78 章及既有 DeepSeek V4/V4.1 章节。待核验完整独立复现、生产 kernel、目标硬件 profiling、线上 tool acceptance、完整训练超参和不同服务快照的行为差异。研究笔记：[`deepseek-v4-source-notes.md`](deepseek-v4-source-notes.md)。

#### 2026-09-20 官方 revision / encoding / inference 实现补证

- [HF revision `b5968e9190ef611bbf34a7229255be88a0e937c1`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/b5968e9190ef611bbf34a7229255be88a0e937c1) 的 API metadata 列出 64 个 safetensors 分片；metadata 总 storage 为 `1,598,839,674,782` bytes。固定 `README/config/encoding/inference` 文件哈希已记录在研究笔记；完整权重未下载。
- `inference/config.json` 的参考路径字段为 `n_hash_layers=3`、`sqrtsoftplus` routing、`window_size=128`、`index_n_heads=64`、`index_head_dim=128`、`hc_mult=4`、`hc_sinkhorn_iters=20`、`expert_dtype=fp4` 和 128/4 压缩布局；官方转换 README 的 `EXPERTS=384, MP=8` 只是示例并行配置。
- `inference/model.py`/`kernel.py` 公开了 gated KV compressor、overlap state、causal/top-k indexer、MLA + compressed sparse attention + local window、前三层 hash routing、top-6 + shared expert、MTP、Hyper-Connections/Sinkhorn，以及 FP8/FP4 quantization/GEMM 和 sparse online softmax。它是 reference implementation evidence，不是生产性能或本地完整推理证据。
- `encoding/encoding_dsv4.py` 是严格的 DSML prompt encoder/parser：tool role 合并为 `<tool_result>`，`<｜DSML｜invoke>`/`parameter` 区分 string/JSON，`<think>` 与 reasoning 保留策略均属于公开协议；不把它写成内部隐藏思维算法。AST 与无 CUDA round-trip 已通过，TileLang/完整权重 inference 未运行。

## DeepSeek V4.1-Flash

- 官方发布页：[DeepSeek-V4.1-Flash Release](https://api-docs.deepseek.com/news/news260910)
- 官方模型卡：[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
- 固定 revision：`dba1be0a40aa45a94ad051997016db3960a90277`
- 配置：[config.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json)
- 技术报告：[DeepSeek_V41_Tech_Report.pdf](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)
- 已核验：页面标注 `2026/09/10`；552B backbone、1M context、20 层 causal encoder + 20 层 decoder 的 CED、8B/16B prefill/decode 激活口径、SWA Bounded Replay、CSA2 的 `Full/Reindex/Reuse`、Hierarchical Sparse Indexer、FP4 main KV 与 890 bytes/global-KV-token、1 shared + 384 routed experts、6 routed experts/token、196B Engram、DSpark、DeepSeek-ViT 多模态路径、45T 预训练 token、64K 稀疏训练到 1M 扩展、`SFT -> RL -> OPD` 数据管线描述和 1--100 reasoning effort。
- API 页面还明确模型名为 `deepseek-flash`，并描述旧 alias 的临时兼容路由；这些是带日期的服务端行为，不能当作永久权重身份。
- 评测边界：MMLU-Pro/HumanEval/GSM8K/MMMU-Pro/DocVQA 及 Terminal-Bench、DeepSWE、AutomationBench、Agent's Last Exam 数字均来自模型卡自报；Agent 数字绑定模型 revision、effort、harness、工具、环境和 verifier，不与其他榜单直接合并。
- 快照身份：发布页 SHA-256 `420cbb7b5e8e97632fa45cb49cd2b5f22b57c8f9e125d1c34a22bd67bbc33705`；README SHA-256 `347c9db4e5506acb531cbc3b724407ab88e9af8781679152f0823d7bac16d251`；技术报告 PDF SHA-256 `ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d`。
- 已逐页提取并复核 51 页技术报告正文；报告补充了精确层排布、HSI 候选池、Single-Pass mHC、DSpark、FP4 量化位置、EPD/SWA 部署、训练设置和异步后训练细节。完整 production kernel、所有参数分片、线上接受率、目标硬件 profiling 和独立 benchmark 仍待核验；固定 HF 仓库的 reference kernel 已在下方单独记录。研究笔记：[`deepseek-v4.1-flash-source-notes.md`](deepseek-v4.1-flash-source-notes.md)。

#### 2026-09-20 固定 revision 的 reference implementation 补证

- [HF inference README](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/README.md)、[`inference/config.json`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/config.json)、[`model.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py)、[`kernel.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/kernel.py)、[`generate.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/generate.py) 和 [`convert.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/convert.py) 已按固定 revision 读取。源码覆盖 SWA ring、压缩 KV、两级 candidate/index top-k、sparse attention、MoE、Engram、Hyper-Connections/Sinkhorn、TileLang FP4/FP8 kernel 和多模态输入。
- HF API metadata 响应 SHA-256 为 `fb3aefa7794da9101d0253ccc4e6e36fbace841778f6857b1370cb2880235d63`，列出 48 个 safetensors 分片；完整权重没有下载。研究笔记登记了关键源码文件 bytes/SHA-256，避免把 `main` 页面内容与固定 revision 混用。
- `inference/README.md` 将路径定义为 readable reference implementation，`generate.py` 使用普通 autoregressive generation；虽然 `model.py` 暴露 DSpark `forward_spec`，参考生成入口没有完整 draft/verify/rollback scheduler。因此源码证据不能升级为生产吞吐或 speculative acceptance 结果。
- `evaluation/README.md` 与 `dsh-minimal.patch` 把 `mini-swe-agent`、`dsh-minimal`、环境和 verifier 分开；评测结果仍按 harness 组合记录。源码仅通过 Python 静态编译与无 CUDA encoding smoke test，未执行完整权重、TileLang 或 GPU 推理。
- 新增本地标准库教学脚本 [`deepseek_v41_cache_demo.py`](code/deepseek_v41_cache_demo.py)，用于分开测量 synthetic candidate-pool recall、conditional Top-K recall、端到端 recall 和 E2M1-like 分组误差；脚本 SHA-256 为 `6c65d44f2f092369ac8b0cdf83a58fe4c2dad38e46698e5d7f1badc209c9517e`。输出仅是教学实验，不是 V4.1 模型、生产 kernel 或硬件 benchmark 结果。

#### 2026-09-22 vLLM upstream main 的 V4.1/DSpark runtime 入口

- [vLLM main `registry.py`](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/registry.py) 快照为 `64,391` bytes / SHA-256 `64c80d8c6659833a9abf836180f2b7549903db2a664e22bc38e38c258a6f572e`，明确登记 `DeepseekV41ForCausalLM` 与 `DSparkV41DraftModel` 到 `vllm.models.deepseek_v41`。
- 正确包路径是 [`vllm/models/deepseek_v41/`](https://github.com/vllm-project/vllm/tree/main/vllm/models/deepseek_v41)，其 `__init__.py` 为 `625` bytes / `6b928f07c6f67f4fd1b599a143cd15a1309a177c877f7f08b29d5fa30db659fe`；本轮固定的 `quant_config.py`、NVIDIA/ROCm `vl_model.py` 与 `dspark.py` 文件大小和哈希详见研究笔记。
- `quant_config.py` 证实 `expert_dtype=fp4/fp8` 分支：FP4 走 MXFP4 + `ue8m0` scale，FP8 走 block-FP8 + float32 scale，并通过 `deepseek_v4_fp8` 路径接入。V4.1 vision wrapper 通过 `inputs_embeds` 注入 ViT/aligner 结果，保留 raw `input_ids` 供 `bias_vl` 路由，支持 encoder CUDA graph/ViT DP，并显式跳过 `mtp.*`，因此当前视觉 wrapper 不应与 MTP/DSpark draft heads 写成已合并能力。
- `dspark.py` 证实三层 draft 配置、目标 layer ids、`[max_num_batched_tokens, index_topk]` Top-K buffer、目标 checkpoint 的 `mtp.{0,1,2}.*` 权重、Markov/confidence head、sigmoid confidence、共享 embedding/lm head、context KV/SWA cache 插入和 FP4/FP8 scale 分支。它是 runtime code evidence，不是 acceptance、吞吐或质量结果。
- [vLLM `v0.29.0` stable `registry.py`](https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/vllm/model_executor/models/registry.py) 快照为 `63,102` bytes / `fef8293fe19cef01768a4c5bd8adc05e120202f1c1597eb264d090672894fbd7`；该版本有通用 V4/DSpark 入口，但没有本轮 `DeepseekV41ForCausalLM`/`DSparkV41DraftModel` 专用登记。结论是“main 已有专用入口，stable 0.29.0 专用入口未证实”，不能把 main 代码升级为 stable wheel 或生产 serving 验收。
- 完整权重、stable wheel 安装、目标 GPU/ROCm 加载、真实 Top-K recall、FP4 质量、speculative acceptance、EPD 调度、硬件 profiling、线上 tool acceptance 和独立 benchmark 仍待核验。完整证据见 [`deepseek-v4.1-flash-source-notes.md`](deepseek-v4.1-flash-source-notes.md)。

#### 2026-09-23 vLLM `v0.30.0` stable release/source evidence

- [vLLM v0.30.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.30.0) 于 `2026-09-22T05:20:54Z` 发布，tag commit 为 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`。固定 [release API](https://api.github.com/repos/vllm-project/vllm/releases/tags/v0.30.0) 为 `65,334` bytes / SHA-256 `bc5d0dee9296de133c54209afab0ae4eb9d2c7c3f331261f1dfdd4d0ab23a48`。
- [v0.30.0 registry](https://raw.githubusercontent.com/vllm-project/vllm/v0.30.0/vllm/model_executor/models/registry.py) 为 `64,420` bytes / SHA-256 `a08a98aaae52ced32226aa647f58d682ac600b9572651a9b377b97846bc99212`，已登记 `DeepseekV41ForCausalLM` 与 `DSparkV41DraftModel`。这证明 v0.30.0 stable release surface 有 V4.1 专用入口；v0.29.0 的 `63,102` bytes / `fef8293fe19cef01768a4c5bd8adc05e120202f1c1597eb264d090672894fbd7` 仍保留为历史负对照。
- v0.30.0 V4.1 package 快照：`__init__.py` `625` / `6b928f07c6f67f4fd1b599a143cd15a1309a177c877f7f08b29d5fa30db659fe`；`quant_config.py` `9,006` / `bfc500c4989607809577cbd10512b96e9162a7359ad407f772b7f695eaa34cd9`；NVIDIA `vl_model.py` `13,728` / `a5a3f477225990946093092d4781db181b59b52102aff2e0e345e631973817f1`；NVIDIA `dspark.py` `23,059` / `4d9c2bfa4c123aa5b95b637f24dc3d04748227455857bdc1376b27cda8af5954`；ROCm `vl_model.py` `13,736` / `6f3fcc8a5896432ef51f809348097e92c7782ab226adb5ecb32cdc599ce05af4`；ROCm `dspark.py` `22,731` / `a109e581711a74a7c5597b3f5a07d81ed05aac0ed2619dfa3f859050efc0b9d9`。
- release notes 记录了 V4.1 接入、SM100 FlashMLA V4.1 record 的 MXFP8 whole-KV、DeepGEMM Mega-mHC、mHC post/pre projection folding、Triton-fused input metadata、CPU-offloaded Engram async prefetch/DP sharding、DSpark draft-state folding、EPLB state isolation、strict tool parameters 的 XGrammar、Responses text parts 和 Vision-Exp image sentinel padding。这些是 vLLM release/runtime evidence，不是 DeepSeek 独立 benchmark 或本机运行结果。
- [PyPI `vllm/0.30.0` metadata](https://pypi.org/pypi/vllm/0.30.0/json) 为 `13,218` bytes / SHA-256 `43020551808911e4cabfca5ea71951766c25101b3817a10d306d88fe42d860b8`；x86_64 wheel `314,883,777` / `ef52ee58c410ead0b8afb190838fa4cbcb52075596f67862a03859d984966ac4`，aarch64 wheel `309,984,160` / `eb3e11bab695d085098579a6eda2d602419adec3826ebfbcce9a3ffa543eb62e`，sdist `42,432,229` / `5f8f4e890c042ffa1c3e103f81c35e2d96f60a0a175ac43a4adbae7006bef62b`。
- GitHub release API 的 CUDA 12.9 artifact 也已固定：x86_64 `vllm-0.30.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl` 为 `545,459,905` bytes / `e98cb69659bfcfc849cf11ce0781a7161d40b02b51a6c3636924a5909f2aabcc`；aarch64 对应 wheel 为 `519,981,036` bytes / `fdb57ab5fa1c3ac4c94a6eff52579df32cc9aab5da9863880d03e7167e088e77`。CUDA/架构/ABI/wheel 名称必须进入部署 manifest，不能只写版本号。
- 同一 release body 的 vLLM-wide serving 周边包括 Fast Start 的 per-GPU post-quantized TP-sharded weight cache + CUDA IPC、HiSparse 的 pinned-host sparse-MLA tier/热 buffer、Model Runner V2 dual-batch overlap，以及 speculative decoding 的 online adaptive verification。它们是 runtime release evidence，不是 DeepSeek V4.1 独立架构或 benchmark 结论。
- 状态更新：**内容专题 + HF reference implementation + vLLM `v0.30.0` stable release/source evidence + SGLang main + recipe protocol evidence（AA 单榜）**。完整权重、依赖安装、GPU/ROCm/NPU、真实 FP4 质量、candidate/index Top-K recall、DSpark acceptance/rollback、EPD、tool acceptance、目标硬件 profiling、独立 benchmark 和生产 SLO 仍待核验。

## K2 Horizon MoVA 36B/A4B

- 排行榜发现：[Artificial Analysis K2 Horizon MoVA 36B A4B](https://artificialanalysis.ai/models/k2-horizon-mova-36b-a4b)；[DataCurve DeepSWE](https://deepswe.datacurve.ai/) 本地 v1.1 快照未检出 K2，因此不把 DeepSWE 写成 K2 的发现证据。
- 官方模型卡：[IFM/K2-Horizon-MoVA-36B-A4B](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B)；固定 revision：`de2d2efb32ed7639b7140bccbefe131a0063a982`。
- 官方实现：[config.json](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B/blob/de2d2efb32ed7639b7140bccbefe131a0063a982/config.json)、[modeling_k2_horizon.py](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B/blob/de2d2efb32ed7639b7140bccbefe131a0063a982/modeling_k2_horizon.py)；部署参考：[SGLang K2 Horizon cookbook](https://docs.sglang.io/cookbook/autoregressive/IFM/K2-Horizon)。
- 已核验：约 36B 总参数、约 4B/token active proxy、48 层中前 3 层 dense、后 45 层 MoVA + MoE，32Q/8KV GQA，64 value experts/top-4，100 routed FFN experts/top-8 + 1 shared expert，524,288 context，BF16，以及 sigmoid router、selection-only bias、2.5 scaling 和 softplus attention gate。
- 训练与 serving 边界：模型卡披露 8K 到 512K 的分阶段训练和 TP=2/EP 部署参考；0.9B 同系列卡片的 MOPD 不能反推为 36B 配方，active 参数也不能直接换算显存或吞吐。
- 关联技术：[K2-Horizon-7B](https://huggingface.co/IFM/K2-Horizon-7B)、[K2-Horizon-7B-Uno adapter](https://huggingface.co/IFM/K2-Horizon-7B-Uno)、[Uno paper, arXiv:2609.04010](https://arxiv.org/abs/2609.04010)。Uno 是 K2 7B 的官方关联 adapter/论文路线，不是排行榜新增候选，也不是 36B 架构变体。
- 教学落地：[`第二十一册第 82 章`](../../book-21-transformer-architecture-evolution/chapters/82-k2-horizon-mova-36b-a4b与uno.md)；完整证据和待核验点见 [`k2-horizon-source-notes.md`](k2-horizon-source-notes.md)。

## DeepSeek-R1-0528

- 官方发布页：[DeepSeek-R1-0528 Release](https://api-docs.deepseek.com/news/news250528)
- 开源权重入口：[deepseek-ai/DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)
- 已核验（官方发布页快照）：发布日期 2025/05/28、页面自述的 benchmark/前端/幻觉改进方向、JSON output、function calling、API 使用方式不变，以及开源权重链接。
- 待核验：模型卡、参数规模、架构、训练/后训练配方、benchmark 图片数字、权重 revision/许可证和独立复现。
- 研究笔记：[`deepseek-r1-0528-source-notes.md`](deepseek-r1-0528-source-notes.md)。

## Qwen3.8

- Artificial Analysis 发现：[Qwen3.8 Max](https://artificialanalysis.ai/models/qwen3-8-max)（2026-08-03）、[Qwen3.8 2.4T A95B](https://artificialanalysis.ai/models/qwen3-8-2-4t-a95b)（2026-08-12）、[Qwen3.8 27B](https://artificialanalysis.ai/models/qwen3-8-27b)（2026-08-14）、[Qwen3.8-Flash-Next](https://artificialanalysis.ai/models/qwen3-8-flash-next)（2026-08-26）。
- DataCurve 辅助发现：[DeepSWE](https://deepswe.datacurve.ai/) 当前快照只检出 `qwen3.8-max`；不把它写成其他三个 Qwen3.8 变体的发现来源。
- 官方模型卡：[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)、[Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B)、[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)。
- 官方实现与报告：[Qwen3.8 GitHub](https://github.com/QwenLM/Qwen3.8)、[Flash-Next GitHub](https://github.com/QwenLM/Qwen3.8-Flash-Next)、[Flash-Next 技术报告](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)。
- 托管页面：[Qwen3.8-Max](https://www.qwencloud.com/models/qwen3.8-max)、[Qwen3.8-27B](https://www.qwencloud.com/models/qwen3.8-27b)、[Qwen3.8-Flash](https://www.qwencloud.com/models/qwen3.8-flash)。Max 是官方说明基于 A95B 的 hosted version，Flash 是 Flash-Next 的托管版本，均不新增为独立 open checkpoint。
- 已核验：27B 的 dense vision-language 结构、A95B 的 2.4T/95B active MoE 结构、Flash-Next 的 125B/6B active + 51B N-gram/4B MTP 账本，以及 GDN、Gated Attention/QSA、Gated Residual、N-gram Embedding、Muon/AdamW 和 thinking protocol。
- 研究笔记：[`qwen3.8-source-notes.md`](qwen3.8-source-notes.md)；教学落地：[`第二十一册第 83 章`](../../book-21-transformer-architecture-evolution/chapters/83-qwen3.8-qsa-gated-residual-n-gram-muon.md)。
- 待核验：完整生产 kernel、线上接受率、目标硬件 profiling、host-memory 预取的端到端收益、全系列训练/后训练配方和独立 benchmark 复现。Flash-Next 报告中的 loss、速度、稳定性和 benchmark 均为发布方自报。

## Qwen3.8 Max (0902) 服务 revision

- 排行榜发现：[Artificial Analysis Qwen3.8 Max](https://artificialanalysis.ai/models/qwen3-8-max)；当前页面标题为 `Qwen3.8 Max (0902)`，canonical 页面为 `qwen3-8-max`，release slug 为 `qwen3-8-max-0902`。2026-09-21 Artificial Analysis 详情快照为 `3,835,865` bytes，SHA-256 `e879b3b7dfe5beac76808114a200afa69ab7e078d61d1685cea4076c4d8e0541`，第三方 Intelligence Index `45.4152084980521`、median output speed 约 `37.275289705095 tokens/s`、cost per Intelligence Index task 约 `$5.408509428374016`、context 约 `984K`；本轮 `/zh` 首页为 `1,777,588` bytes、SHA-256 `3fa3fc0caa518a3617f5aaabf2618db26f6f68c5b9d28e50e245c1240530bdee`。这些是第三方目录/测量字段。
- DataCurve 辅助发现：[DeepSWE](https://deepswe.datacurve.ai/) 的 `mini_swe_agent_qwen3_8_max_xhigh` 为 258/449、Pass@1 `57.4610%`、Pass@4 `83.1858%`、平均成本约 `$3.7291`、平均输出 `95,075` tokens、平均 Agent steps `111.34`、4 runs；页面没有 `qwen3_8_max_0902` 精确行，因此不把这些数字绑定到 0902 revision。
- 官方服务页：[Qwen3.8-Max-0902](https://www.qwencloud.com/models/qwen3.8-max-0902)；官方 alias 为 `qwen3.8-max-2026-09-02`，页面称其为 `qwen3.8-max` 的 upgraded snapshot。页面 `last-modified` 更新为 `2026-09-21 11:01:05`；三条代理均取得 `98,992` bytes，但动态 trace/CSS/asset 字段导致哈希分别为 7890=`aef93a9892e52504043a81a941a2150a24a5210dd3daf769a85e72dbe5949dad`、8098=`8aa221b29ac0236eee10dc745c374e2c77362077e9d852789a6a9f3fd3c70270`、1234=`39cabb0a6291c417acf6c9904c2c769be89c4739016d41c3ceecfd6f10160e5e`。
- 官方运行时资料：[Thinking](https://docs.qwencloud.com/developer-guides/text-generation/thinking)、[Function Calling](https://docs.qwencloud.com/developer-guides/tool-calling/function-calling)、[Context Cache](https://docs.qwencloud.com/developer-guides/run-and-scale/context-cache)。已核验 `low/medium/xhigh` 的 `reasoning_effort`（默认 `xhigh`）、与 `thinking_budget` 互斥、thinking 下 `tool_choice` 只能为 `auto`/`none`、`MultiModalConversation`、explicit/implicit/session cache 和最小 1,024-token 缓存长度。
- 2026-09-21 文档示例已从旧的 `dashscope-intl.aliyuncs.com` 迁移到 `maas.qwencloudapi.com`：OpenAI-compatible mode 为 `https://maas.qwencloudapi.com/compatible-mode/v1`，DashScope API 为 `https://maas.qwencloudapi.com/api/v1`，Realtime WebSocket 也使用该 host。这是 provider adapter/transport 配置变更，不是模型能力或架构升级；历史 endpoint 只作为历史快照保留。
- [Dynamic Rate Limiting](https://docs.qwencloud.com/developer-guides/administration/dynamic-rate-limits) 快照为 `408,930` bytes、SHA-256 `ba010f2386fbb32c0b69bcbdde64efb520435cefc16475cb030b55253909949e`；`qwen3.8-max-0902` 的保证 TPM tiers 为 `1,500,000 / 1,500,000 / 1,500,000`。机制是 account+model 聚合、workspace override、月度 tier 和 soft TPM；这些是 hosted quota，不是模型吞吐、GPU capacity 或榜单评测结果。
- 官方服务页还给出 1M context、991K 普通最大输入、983K thinking 最大输入、131K 最大输出，以及当前 `$2/$6/$0.25/$2.50/$0.17` 每百万 token 的输入/输出/implicit-cache/explicit-create/explicit-read 价格字段。coding、工程规模项目、长周期 autonomous development、多工具 Agent 和视觉理解是产品升级描述，不是新架构证据。
- 当前解释：这是 hosted service 的 0902 revision/update，不新增独立 open checkpoint，也不新增 Qwen3.8 架构章节。参数、层排布、训练配方、专属技术报告、生产 kernel、线上接受率和硬件 profiling仍待核验。研究笔记：[`qwen3.8-max-0902-source-notes.md`](qwen3.8-max-0902-source-notes.md)；教学落点扩展第二十一册第 83 章和第二十四册工具 serving 章节。

## Qwen3.5-397B-A17B

- 榜单发现：[Artificial Analysis Qwen3.5 397B A17B](https://artificialanalysis.ai/models/qwen3-5-397b-a17b)；页面有 Reasoning/Non-reasoning 配置，按同一基础模型归并。DataCurve 当前没有精确 `mini_swe_agent_qwen3_5_397b_a17b_*` 行。
- Artificial Analysis 详情快照 `/tmp/qwen35-397-aa.html` 为 3,700,616 bytes，SHA-256 `04f09d032033c2705bc9a40c649ebc226f81eeef457adf11886f644490181719`；页面字段的 397B/17B、约 1M context 和 Intelligence Index 约 19.1073 均保持第三方目录证据边界。
- 官方模型卡：[Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)；固定配置：[config.json](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/raw/main/config.json)；官方仓库：[QwenLM/Qwen3.8](https://github.com/QwenLM/Qwen3.8)；发布博客：[Qwen3.5](https://qwen.ai/blog?id=qwen3.5)。
- 已核验公开字段：397B total/17B active、60 层、hidden 4096、15 组 `3 x Gated DeltaNet + 1 x Gated Attention`、512 experts、10 routed + 1 shared、原生 262,144 context/约 1.01M YaRN、vision encoder、MTP 和 8-GPU serving 示例。
- Qwen 官方还声明 early-fusion 多模态训练、trillions of multimodal tokens、million-agent RL environments、asynchronous RL framework、201 languages/dialects；这些是发布方自报的训练/产品声明，不升级为独立复现或完整 recipe。
- Qwen3.5-Plus 是官方模型卡声明的 hosted counterpart，不新增为独立 open checkpoint；不能把 Qwen3.8 的 QSA、Gated Residual、N-gram 或 Muon 细节反向迁移给 Qwen3.5。
- 当前状态：内容专题闭环。研究笔记：[`qwen3.5-397b-a17b-source-notes.md`](qwen3.5-397b-a17b-source-notes.md)；正式落点：第二十一册第 83 章新增 Qwen3.5 对比/前置小节。完整训练 recipe、kernel、MTP acceptance rate、视觉独立复现和 hosted/open 精确差异仍待核验。

## Gemini 3.7 Flash

- Artificial Analysis 发现：[Gemini 3.7 Flash high](https://artificialanalysis.ai/models/gemini-3-7-flash)、[medium](https://artificialanalysis.ai/models/gemini-3-7-flash-medium)、[low](https://artificialanalysis.ai/models/gemini-3-7-flash-low)，页面内嵌榜单 `releaseDate` 均为 2026-08-13。
- Artificial Analysis high 详情页本轮复验为 3,605,373 bytes，SHA-256 `d0c5128baac580dbc983b403719f44730decf3bd6c3a24484d29633f13ffbe4c`；第三方字段为 Intelligence Index `39.4295316404896`、约 `292.3389 tokens/s`、约 `10.2161s` input/TTFT 字段、1M context，输入/输出/cache hit 约 `$0.75/$3.75/$0.075` 每百万 token。medium/low 详情快照哈希见研究笔记。
- DataCurve DeepSWE v1.1 页面（2026-09-03）包含 `gemini-3.7-flash` low/medium/high，113 tasks、4 runs、统一 `mini-swe-agent`；Pass@1 分别为 `53.7611%/65.4867%/65.2655%`，平均成本约 `$1.8323/$2.0251/$2.1763`。这些是配置 + harness + 工具 + 环境 + verifier 的系统结果，不是裸模型分数；快照 SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。
- 官方模型页：[Gemini 3.7 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.7-flash)；[Google DeepMind Model Card](https://deepmind.google/models/model-cards/gemini-3-7-flash/)；[官方评测 PDF](https://storage.googleapis.com/deepmind-media/gemini/gemini_3-7_flash_model_evaluation.pdf)；[Frontier Safety Framework 报告](https://storage.googleapis.com/deepmind-media/gemini/gemini_3-7_flash_fsf_report.pdf)。
- 官方周边文档：[Thinking](https://ai.google.dev/gemini-api/docs/thinking)、[Interactions API](https://ai.google.dev/gemini-api/docs/interactions-overview)、[工具](https://ai.google.dev/gemini-api/docs/tools)、[工具组合](https://ai.google.dev/gemini-api/docs/tool-combination)、[视频理解](https://ai.google.dev/gemini-api/docs/video-understanding)、[Context caching](https://ai.google.dev/gemini-api/docs/context-caching) 和 [Computer Use](https://ai.google.dev/gemini-api/docs/computer-use)。
- 已核验：`gemini-3.7-flash`、text/image/video/audio/PDF 输入、text 输出、1,048,576 输入 token、65,536 输出 token、low/medium/high thinking、agentic video understanding、Interactions 的 interaction/step/state/background、implicit caching、tool context circulation、加密 signature、stateful/stateless 回放、built-in/custom tool 执行边界和视频 `processing_call`/`processing_result`。
- Model Card 公开的是核心 reasoning foundation 的算法改进、agentic video 与可调 thinking；架构、训练数据、软硬件信息指向 Gemini 3.6 Flash Model Card。当前没有 Gemini 3.7 独立参数/层结构/完整训练 recipe 或官方专属技术报告。
- arXiv 精确标题检索 [`title:"Gemini 3.7 Flash"`](https://arxiv.org/search/?query=%22Gemini+3.7+Flash%22&searchtype=title) 返回 0 个结果；全文检索找到 3 篇将其作为被测配置的外部论文（arXiv:2609.15983、2609.05232、2608.20563），不把它们写成 Google 技术报告。
- 当前状态：资料级闭环；暂无独立 Gemini 3.7 正式章节。面试映射进入 Reasoning、Agent/工具协议、视频多模态、长上下文/Serving、评测与安全章节。完整证据、快照哈希和待核验项见 [`gemini-3.7-flash-source-notes.md`](gemini-3.7-flash-source-notes.md)。

## Gemini 3.6 Flash

- 排行榜发现：[Artificial Analysis Gemini 3.6 Flash (high)](https://artificialanalysis.ai/models/gemini-3-6-flash)；[DataCurve DeepSWE v1.1](https://deepswe.datacurve.ai/) 的 `mini_swe_agent_gemini_3_6_flash_high`、`reasoning_effort: high`。两个排行榜都只按配置行记录，不把 `high` 另算成独立架构。
- Artificial Analysis 详情页本轮复验：榜单 `releaseDate` `2026-07-21`、Intelligence Index `34.3395678920714`、median output speed `192.733708656795 tokens/s`、TTFT `19.2238583065s`、1M context、约 `$0.15/$0.75/$3.75` cache-hit/input/output 每百万 token，以及约 `$0.9288`/Intelligence Index task。provider benchmark 页当前展示 `Google AI Studio` 一个 provider；上述均为第三方配置字段。
- DataCurve 原始行：211/452，Pass@1 `46.6814%`、Pass@4 `75.2212%`、平均成本 `$2.2095`、平均输出 `95,844.86` token、平均 Agent steps `116.73`，统一 `mini-swe-agent`、4 runs、113 tasks。结果绑定 harness、工具、任务环境和 verifier，不能写成裸模型能力。
- 官方资料：[Gemini 3.6 Flash API 模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3.6-flash)、[DeepMind Model Card](https://deepmind.google/models/model-cards/gemini-3-6-flash/)、[Model Card PDF](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-6-Flash-Model-Card.pdf)、[Gemini 3.5 Flash Model Card PDF（官方依赖资料）](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-5-Flash-Model-Card.pdf)。
- 已核验：`gemini-3.6-flash`、2026-07-21、基于 Gemini 3.5 Flash、text/image/video/audio/PDF 输入、text 输出、1,048,576 输入 token、65,536 输出 token、默认 `medium` 且支持 `minimal/low/medium/high` thinking、caching/code execution/File Search/function calling/Maps/Search grounding/structured outputs/URL context、Computer Use Preview、Batch/Flex/Priority inference。
- 官方周边文档：[Thinking](https://ai.google.dev/gemini-api/docs/thinking)、[Interactions API](https://ai.google.dev/gemini-api/docs/interactions-overview)、[工具组合](https://ai.google.dev/gemini-api/docs/tool-combination)、[视频理解](https://ai.google.dev/gemini-api/docs/video-understanding)、[Context caching](https://ai.google.dev/gemini-api/docs/caching)、[Computer Use](https://ai.google.dev/gemini-api/docs/computer-use)。文档明确把 Gemini 3.6 列入 Interactions、agentic video 和 4,096-token implicit caching 支持表；Gemini 3 工具组合通过 thought/tool signatures 和 tool context circulation 维持多轮连续性。
- Model Card 的架构、训练数据、数据处理、硬件和软件信息指向 Gemini 3.5 Flash；其 benchmark 与安全表、Frontier Safety 结论和 9 篇 arXiv 外部使用论文已逐项记录，但不能升级为 3.6 独有训练/架构事实。
- arXiv [精确标题检索](https://arxiv.org/search/?query=%22Gemini+3.6+Flash%22&searchtype=title) 返回 0 篇；[全文检索](https://arxiv.org/search/?query=%22Gemini+3.6+Flash%22&searchtype=all) 返回 9 篇外部使用/评测论文（2609.14973、2609.08402、2608.24921、2608.23061、2608.16663、2608.06361、2607.23976、2607.23893、2601.16755），不是 Gemini 3.6 专属技术报告。
- 当前状态：资料级闭环；暂无独立 Gemini 3.6 正式架构章节。面试映射进入 thinking budget、Interactions/state replay、工具签名、agentic video、长上下文/caching、Computer Use 权限和证据分层。完整快照哈希、论文标题和待核验项见 [`gemini-3.6-flash-source-notes.md`](gemini-3.6-flash-source-notes.md)。

## Gemini 3.1 Pro Preview

- 排行榜发现：[Artificial Analysis Gemini 3.1 Pro Preview](https://artificialanalysis.ai/models/gemini-3-1-pro-preview)；[DataCurve DeepSWE v1.1](https://deepswe.datacurve.ai/) 的 `mini_swe_agent_gemini_3_1_pro_preview_high`、`reasoning_effort: high`。2026-09-16 三条代理对详情页和 DataCurve 均返回完整 HTTP 200；同一页面快照逐字节一致。
- Artificial Analysis 2026-09-16 详情字段：第三方 `releaseDate` `2026-02-19`、Intelligence Index `30.3596656132261`、约 `108.4052 tokens/s`、约 `24.6088s` TTFT、1M context、约 `$2/$12/$0.20` input/output/cache-hit 每百万 token。当前页面标记 proprietary、parameters 为 null；这些是第三方目录/测量字段。
- DataCurve 原始行：`53/452`、Pass@1 `11.7257%`、Pass@4 `28.3186%`、平均成本约 `$2.1434`、平均输出 `28,368.88` token、平均 Agent steps `75.56`；页面为 113 tasks/91 repositories/5 languages/4 runs，统一 `mini-swe-agent`。结果绑定 harness、工具、任务集、环境和 verifier，不是裸模型能力。
- 官方资料：[Gemini 3.1 Pro API 模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview)、[Release Notes](https://ai.google.dev/gemini-api/docs/changelog)、[DeepMind Model Card](https://deepmind.google/models/model-cards/gemini-3-1-pro/)、[评测方法](https://deepmind.google/models/evals-methodology/gemini-3-1-pro)、[Gemini 3.1 Pro Model Card PDF](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-1-Pro-Model-Card.pdf)、[Gemini 3 Pro Model Card 依赖入口](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Pro-Model-Card.pdf)。
- 已核验：`gemini-3.1-pro-preview`、2026-02-19、proprietary、text/image/video/audio/PDF 输入、text 输出、1,048,576 输入 token、65,536 输出 token、caching/code execution/function calling/Search/Maps grounding/structured outputs；`gemini-3.1-pro-preview-customtools` 面向 bash + custom tools 的工具优先级，但不是独立基础模型。
- 官方 Thinking/工具资料：[Thinking](https://ai.google.dev/gemini-api/docs/thinking)、[Thought signatures 迁移页](https://ai.google.dev/gemini-api/docs/thought-signatures)、[工具组合](https://ai.google.dev/gemini-api/docs/tool-combination)、[Function calling](https://ai.google.dev/gemini-api/docs/function-calling)、[Long context](https://ai.google.dev/gemini-api/docs/long-context)、[Context caching](https://ai.google.dev/gemini-api/docs/caching)。面试主线为 `thinking_level` 与共同 output budget、加密 `signature`/`id` 回放、tool context circulation、built-in/custom tool 宿主分层、1M context 与缓存/召回/成本权衡。
- Model Card 将架构、训练数据、数据处理、硬件和软件资料指向 Gemini 3 Pro；评测覆盖 reasoning、multimodal、agentic tool use、multilingual 和 long-context。代表性发布方结果包括 HLE no-tools `44.4%`、ARC-AGI-2 `77.1%`、GPQA Diamond `94.3%`、SWE-Bench Pro `54.2%`、MCP Atlas `69.2%`、MRCR v2 128K/1M `84.9%/26.3%`；不与 AA/DeepSWE 分数拼接。
- 安全页记录 Text-to-Text `+0.10%`、Multilingual `+0.11%`、Image-to-Text `-0.33%`、Tone `+0.02%`、Unjustified-refusals `-0.08%`（相对 Gemini 3 Pro 的自动评测变化）；Frontier Safety 页面称 Cyber 达到 alert threshold 但未达到 CCL，且计入 inference cost 后 Deep Think 结果低于无 Deep Think 设置。以上是发布方安全评估，不是训练 recipe。
- arXiv [标题精确检索](https://arxiv.org/search/?query=%22Gemini+3.1+Pro%22&searchtype=title) 返回 0；[全文检索](https://arxiv.org/search/?query=%22Gemini+3.1+Pro%22&searchtype=all) 返回 198 篇外部使用/评测结果，没有检出 Google 发布的 Gemini 3.1 Pro 专属技术报告。完整快照哈希、证据边界与待核验项见 [`gemini-3.1-pro-preview-source-notes.md`](gemini-3.1-pro-preview-source-notes.md)。
- 当前状态：资料级闭环；暂无独立 Gemini 3.1 Pro 正式架构章节。内容映射到 Reasoning、Agent/工具协议、长上下文/Serving、多模态、评测与安全章节。

## Gemini 3.5 Flash

- 排行榜发现：[Artificial Analysis Gemini 3.5 Flash (high)](https://artificialanalysis.ai/models/gemini-3-5-flash)；[DataCurve DeepSWE v1.1](https://deepswe.datacurve.ai/) 的 `mini_swe_agent_gemini_3_5_flash_high`、`reasoning_effort: high`。Artificial Analysis 同一 release 还列出 `medium` 与 `minimal` 配置，按运行配置归并为一个基础模型。
- Artificial Analysis 2026-09-15 复验字段：第三方 `releaseDate` `2026-05-19`、Intelligence Index `32.9816033695905`、约 `221.8080 tokens/s`、TTFT `18.3631s`、1M context、约 `$1.5625`/Intelligence Index task；当前页面 serialized object 标为 deprecated 并指向 `gemini-3-6-flash`，页面只继续更新默认 10K 输入 workload。deprecated 是榜单目录状态，不替代 Google API 生命周期声明。
- DataCurve 原始行是 163/452、Pass@1 `36.0619%`、Pass@4 `63.7168%`、平均成本 `$3.4467`、平均输出 `75,730.19` token、平均 Agent steps `105.30`，统一 `mini-swe-agent`、4 runs、113 tasks。结果绑定 effort、harness、工具、任务环境和 verifier，不能写成裸模型能力。
- 官方资料：[Gemini 3.5 Flash API 模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash)、[What's new in Gemini 3.5 Flash](https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.5)、[DeepMind Model Card](https://deepmind.google/models/model-cards/gemini-3-5-flash/)、[Gemini 3 Flash Model Card（架构等依赖入口）](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Flash-Model-Card.pdf)。
- 已核验：`gemini-3.5-flash`、2026-05-19、1,048,576 输入 token、65,536 输出 token、text/image/video/audio/PDF 输入、text 输出、默认 `medium` 与 `minimal/low/medium/high` thinking、thought preservation、Interactions API、implicit caching、combined tool use、Computer Use Preview、Computer Use prompt-injection detection 和 Batch/Flex/Priority inference；What's New 页面还记录知识截止为 2025 年 1 月。
- 3.5 的主要面试主线是默认 thinking 从 Gemini 3 Flash Preview 的 `high` 改为 `medium`、`thinking_level`/legacy `thinking_budget` 互斥、带 thought signature 的跨轮 reasoning context、Interactions/state replay、tool context circulation 与宿主责任分层、4,096-token implicit caching，以及 Computer Use 的截图注入防护。当前视频文档的 agentic processing 支持列表明确列出 3.8/3.7/3.6 和 3.5 Flash-Lite，没有明确列出 3.5 Flash；不把 3.6 的 agentic video 结论迁移给 3.5 Flash。
- Model Card 的 architecture、training dataset、data processing、hardware、software 和主要限制均指向 Gemini 3 Flash Model Card；没有独立公开的 3.5 参数、层/专家结构、完整训练/后训练 recipe 或专属技术报告。Model Card 的 benchmark、安全和 Frontier Safety 数字保留为 Google 发布方证据。
- arXiv [精确标题检索](https://arxiv.org/search/?query=%22Gemini+3.5+Flash%22&searchtype=title) 返回 0 篇；[全文检索](https://arxiv.org/search/?query=%22Gemini+3.5+Flash%22&searchtype=all) 返回 30 篇外部使用/评测论文，没有检出 Gemini 3.5 Flash 专属技术报告。完整来源、快照哈希、Model Card 评测表和证据边界见 [`gemini-3.5-flash-source-notes.md`](gemini-3.5-flash-source-notes.md)。
- 当前状态：资料级闭环；暂无独立 Gemini 3.5 正式架构章节。面试映射进入 Reasoning、Agent/工具协议、Computer Use、长上下文/Serving、评测与安全章节。

## Gemini 3.8 Flash

- Artificial Analysis 发现：[Gemini 3.8 Flash high](https://artificialanalysis.ai/models/gemini-3-8-flash)、[medium](https://artificialanalysis.ai/models/gemini-3-8-flash-medium)、[low](https://artificialanalysis.ai/models/gemini-3-8-flash-low)，三条榜单记录日期均为 2026-09-02。
- DataCurve DeepSWE v1.1 发现：[DeepSWE](https://deepswe.datacurve.ai/) 当前快照包含 `gemini-3.8-flash` high 配置；Pass@1、成本、输出 token 和 Agent steps 只作为配置 + harness 结果记录。
- 官方模型页：[Gemini 3.8 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash)；最新模型/迁移页：[What's new in Gemini 3.8 Flash](https://ai.google.dev/gemini-api/docs/latest-model)；[Google DeepMind Model Card](https://deepmind.google/models/model-cards/gemini-3-8-flash/)；[官方发布博客](https://blog.google/innovation-and-ai/models-and-research/gemini-models/3-8-flash-and-3-8-flash-cyber/)；[官方评测 PDF](https://storage.googleapis.com/deepmind-media/gemini/gemini_3-8_flash_model_evaluation.pdf)。
- 已核验：`gemini-3.8-flash`、GA/stable、1,048,576 输入 token、65,536 输出 token、文本/图像/视频/音频/PDF 输入、文本输出、low/medium/high thinking、caching、function calling、structured outputs、Google Search/Maps grounding、URL Context、File Search、Code Execution 和 Computer Use（Preview）。
- 周边官方文档：[Thinking](https://ai.google.dev/gemini-api/docs/thinking)、[Thought signatures](https://ai.google.dev/gemini-api/docs/thought-signatures)、[Interactions API](https://ai.google.dev/gemini-api/docs/interactions-overview)、[工具组合](https://ai.google.dev/gemini-api/docs/tool-combination)、[Google Search grounding](https://ai.google.dev/gemini-api/docs/google-search)、[URL Context](https://ai.google.dev/gemini-api/docs/url-context)、[File Search](https://ai.google.dev/gemini-api/docs/file-search)、[Code Execution](https://ai.google.dev/gemini-api/docs/code-execution)、[Computer Use](https://ai.google.dev/gemini-api/docs/computer-use)、[Long context](https://ai.google.dev/gemini-api/docs/long-context) 和 [Structured outputs](https://ai.google.dev/gemini-api/docs/structured-output)。
- 官方最新模型页自述其在长任务中使用更小的 reasoning steps、迭代调用工具并验证结果；该描述作为公开系统行为记录，不升级为具体 RL、verifier 或网络架构事实。
- Model Card 将 3.8 的架构、训练数据及软硬件信息指向 Gemini 3.7 Model Card；目前没有独立公开的 3.8 参数规模、层结构、完整训练/后训练配方或独立论文证据。研究笔记：[`gemini-3.8-flash-source-notes.md`](gemini-3.8-flash-source-notes.md)。
- 当前状态：资料级闭环。暂无独立 Gemini 3.8 正式章节，技术映射进入 Reasoning、Agent/工具协议、Computer Use、长上下文/Serving 与评测章节。

## Grok 4.6

- 排行榜发现：[Artificial Analysis Grok 4.6](https://artificialanalysis.ai/models/grok-4-6) 的 `high`、`low`、`medium`、`xhigh` 配置；[DataCurve DeepSWE v1.1](https://deepswe.datacurve.ai/) 的 `mini_swe_agent_grok_4_6_low/medium/high/xhigh` 配置。
- Artificial Analysis 详情页快照确认第三方 `releaseDate: 2026-08-12`、Intelligence Index `44.4050073012592`、58.5035 tokens/s、40.8596s TTFT、500K context、`$2/$6` 和 proprietary/parameters null。两个代理抓到的快照均为 3,522,180 bytes，SHA-256 `8d6c96aa27f6db87537f3f1802ca07122d58385ee4b6d5e1a5ab84897f360383`；日期和指数仍是第三方榜单字段。
- DataCurve 页面更新时间为 2026-09-03，`generated_at` 为 `2026-09-03T22:24:37.984682+00:00`，规模为 113 tasks/91 repositories/5 languages，统一 harness 为 `mini-swe-agent`。四档原始 Pass@1 约为 low 41.648%、medium 67.478%、high 65.188%、xhigh 66.741%，平均成本约 `$1.0424/$3.4490/$4.3849/$5.4977`；全部绑定工具、任务集、环境、verifier 和 4 runs。快照 SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`。
- 官方发布：[Introducing Grok 4.6](https://x.ai/news/grok-4-6)，JSON-LD 发布日期为 2026-08-12。xAI 公开描述更长 supplemental training、模型生成 reasoning/技术数据、高质量工程数据、改进 optimizer/training recipe、Grok 4.5 生成 SFT trajectories、model-based checks、面向编码/知识工作/领域环境的 agentic RL，以及长轨迹 self-testing/verification；这些仍不是完整训练 recipe 或架构证明。
- 官方模型页：[Grok 4.6 overview](https://docs.x.ai/developers/grok-4-6)；[Markdown 版本](https://docs.x.ai/developers/grok-4-6.md)；[Reasoning](https://docs.x.ai/developers/model-capabilities/text/reasoning)；[Context Compaction](https://docs.x.ai/developers/advanced-api-usage/context-compaction)；[Function Calling](https://docs.x.ai/developers/tools/function-calling)；[Web Search](https://docs.x.ai/developers/tools/web-search)；[X Search](https://docs.x.ai/developers/tools/x-search)；[Remote MCP](https://docs.x.ai/developers/tools/remote-mcp)。
- 已核验：`grok-4.6`、500K context、2026-02-01 knowledge cutoff、text/image input、text output、无文本输出上限、`low/medium/high/xhigh`（默认 high）、Responses/Chat Completions、function calling、structured outputs、Web/X Search、Code Execution，以及 `prompt_cache_key`/`x-grok-conv-id` 和长循环 compaction 建议。
- xAI 发布页还给出 AA Intelligence 61、GDPVal-AA 1753、DeepSWE 1.1 65.9、CursorBench 3.2 69.9、FrontierCode 1.1 61.3；这些是 2026-08-12 发布页展示值，不与当前 AA 详情页或 2026-09-03 DataCurve 原始值直接拼接。
- 论文/报告边界：[arXiv 精确标题检索](https://arxiv.org/search/?query=%22Grok+4.6%22&searchtype=title) 本轮返回 0 个结果；当前公开入口未检出 Grok 4.6 专属论文、独立技术报告或公开权重代码。快照 SHA-256 `2aa9e76dfac307b35c58488f760af873adc01e168aaaaadb9dd96201a22cc90b`。
- 当前状态：资料级闭环；没有独立架构/完整训练报告，因此暂不新增 Grok 4.6 专属正式章节。面试主线为模型生成数据与 SFT 轨迹筛选、agentic RL、长任务自测/验证、reasoning opaque state、prompt cache/compaction、工具责任和 harness-aware evaluation。完整证据、快照哈希和待核验项见 [`grok-4.6-source-notes.md`](grok-4.6-source-notes.md)。

### 2026-09-21 当前时点排行榜复验

- Artificial Analysis 当前详情页快照 `/tmp/grok46-aa-20260921.out` 为 `3,859,075` bytes，SHA-256 `a23b19aceae3fee2b1a92421d21358eb5739043c86b6d24a81348d849f887e94`。当前 `Grok 4.6 (high)` 字段为 Intelligence Index `44.3113073012592`、median output speed `66.6843264403358 tokens/s`、TTFT `46.00s`（原始输入时间约 `45.9997646444999s`）、500K context、`$2/$6`；`releaseDate` 仍是第三方字段 `2026-08-12`。9 月 15 日详情页的 `44.4050073012592`、`58.5035284934629 tokens/s`、`40.8595908855s` 作为历史快照保留，不解释为模型 revision 或训练变化。
- DataCurve 当前快照 `/tmp/ds-current-1234.html` 为 `268,571` bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`；页面仍为 2026-09-03 更新、`generated_at=2026-09-03T22:24:37.984682+00:00`、113 tasks/91 repositories/5 languages、`mini-swe-agent`。四档 `mini_swe_agent_grok_4_6_*` 当前仍为 low/medium/high/xhigh 的 `41.648%/67.478%/65.188%/66.741%` Pass@1，平均成本 `$1.0424/$3.4490/$4.3849/$5.4977`，并绑定 4 runs、工具、环境和 verifier；没有新增 DataCurve 结果。
- 本轮重新尝试 xAI 发布公告、官方模型页、Markdown、Reasoning、Compaction、Tools 和 Remote MCP；`10.237.126.170:1234` 为 EOF/超时，`10.24.27.134:7890` 与 `10.24.27.134:8098` 为连接超时。已有 2026-09-15 官方快照继续作为历史证据，不能写成当前联网成功；当前状态仍为资料级闭环，不新增技术结论或专属架构章节。

## Grok 4.5

- 排行榜发现：[Artificial Analysis Grok 4.5](https://artificialanalysis.ai/models/grok-4-5) 的 `Grok 4.5 (high)`；[DataCurve DeepSWE v1.1](https://deepswe.datacurve.ai/) 的 `mini_swe_agent_grok_4_5_high`、`reasoning_effort: high`。
- 官方发布：[Introducing Grok 4.5](https://x.ai/news/grok-4-5)，发布日期为 2026-07-16。
- 官方模型页：[Grok 4.5 overview](https://docs.x.ai/developers/models/grok-4.5)；[Markdown 版本](https://docs.x.ai/developers/models/grok-4.5.md)；[Reasoning](https://docs.x.ai/developers/model-capabilities/text/reasoning)；[Generate Text](https://docs.x.ai/developers/model-capabilities/text/generate-text)；[Context Compaction](https://docs.x.ai/developers/advanced-api-usage/context-compaction)；[Tools Overview](https://docs.x.ai/developers/tools/overview)；[Function Calling](https://docs.x.ai/developers/tools/function-calling)；[Web Search](https://docs.x.ai/developers/tools/web-search)；[X Search](https://docs.x.ai/developers/tools/x-search)；[Remote MCP](https://docs.x.ai/developers/tools/remote-mcp)；[Release Notes](https://docs.x.ai/developers/release-notes)。
- 已核验：500K context、text/image input、text output、function calling、structured outputs、Batch API 不支持、150 RPS、50M TPM、区域、`low/medium/high/xhigh` 专属模型页字段和 200K prompt 价格阈值；Reasoning 文档同时称 Grok 4.5 的 `xhigh` 按 `high` 处理，官方页面存在规格不一致。
- 官方发布方描述：deduplication、quality scoring、domain-focused data selection、数以万计 GB300 GPU、数十万任务的 RL、automated/model-based grading 和 highly asynchronous agentic rollouts；这些不是完整训练配方或内部架构证明。
- 周边技术：opaque/encrypted reasoning state、30 天 Responses 状态、Context Compaction、server-side built-in tools、parallel function calling、Web/X Search、Remote MCP 的 `allowed_tools` 和工具责任分层；`grok-4.20-multi-agent` 的 4/16 agents 不归因于 Grok 4.5。
- 论文/报告边界：[arXiv 精确标题检索](https://arxiv.org/search/?query=%22Grok+4.5%22&searchtype=title) 本轮返回 0 个结果；当前公开入口未检出 Grok 4.5 专属论文或独立技术报告。
- 当前状态：资料级闭环；没有独立架构/训练报告，因此暂不新增 Grok 4.5 专属正式章节。完整证据、快照哈希、面试主线和待核验项见 [`grok-4.5-source-notes.md`](grok-4.5-source-notes.md)。

## Claude Opus 5

- 官方模型目录：[Anthropic Models Overview](https://platform.claude.com/docs/en/models/overview)
- 专属模型页：[Claude Opus 5 overview](https://platform.claude.com/docs/en/models/opus-5/overview)
- 官方关联资料：[Claude Opus 5 announcement](https://www.anthropic.com/research/claude-opus-5)、[system card](https://www.anthropic.com/claude-opus-5-system-card)、[Thinking](https://platform.claude.com/docs/en/build-with-claude/thinking)、[Effort](https://platform.claude.com/docs/en/build-with-claude/effort)、[Refusals and fallback](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback)、[Prompting Claude](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude)
- 榜单锚点：[Artificial Analysis Claude Opus 5](https://artificialanalysis.ai/models/claude-opus-5) 的 `max`；[DataCurve DeepSWE v1.1](https://deepswe.datacurve.ai/) 的 `mini_swe_agent_claude_opus_5_max` 及其他 effort 配置。
- 已核验：`claude-opus-5`、2026-07-24 页面发布日期字段、1M context、128K 普通最大输出、300K batch 最大输出、Claude API/Bedrock/Vertex/Foundry 平台、adaptive thinking、默认 `high`、`low/medium/high/xhigh/max` effort、$5/$25 输入输出价格、512-token 最小缓存 prompt、fallback 与中途工具/effort 变更协议。
- 榜单快照：Artificial Analysis Intelligence Index 约 `50.7002`、约 `50.07` output tokens/s、约 `46.50s` TTFT、约 `$5.8584`/task；DeepSWE max 为 327/444、Pass@1 约 `73.65% +/-4%`、Pass@4 约 `88.50%`、平均成本约 `$11.8376`、约 99 steps。所有数字绑定具体配置、harness、任务集、工具、provider 和 verifier，不能写成裸模型能力。
- 论文/报告检索：[arXiv 精确标题检索](https://arxiv.org/search/?query=%22Claude+Opus+5%22&searchtype=title) 返回 2 篇把 Opus 5 当被测模型的文章，没有 Opus 5 专属技术报告；Anthropic Research 页面也没有专属报告条目。System card 已登记但 PDF 正文当前无法稳定抽取。
- 仍待核验：参数规模、稠密/MoE 架构、训练配方、后训练算法、adaptive thinking 内部实现、完整安全数字和独立 benchmark 复现；官方 API 字段与榜单配置不能替代技术报告。
- 研究笔记：[`claude-opus-5-source-notes.md`](claude-opus-5-source-notes.md)。

## Claude Fable 5.1

- 专属模型页：[Claude Fable 5.1 overview](https://platform.claude.com/docs/en/models/fable-5-1/overview)
- 官方发布入口：[Claude Fable and Mythos 5.1](https://www.anthropic.com/claude-fable-and-mythos-5-1)
- 官方 System Card：[Claude Fable 5.1 and Claude Mythos 5.1 System Card](https://www.anthropic.com/claude-fable-5-1-mythos-5-1-system-card)
- 排行榜发现：[Artificial Analysis](https://artificialanalysis.ai/models/claude-fable-5-1)；本次 DataCurve DeepSWE 快照只有 Claude Fable 5，没有 Claude Fable 5.1，因此不把 DeepSWE 记为 Fable 5.1 的发现来源。
- 已核验：`claude-fable-5-1`、2026-09-01 发布字段、1M context、128K 最大输出、adaptive always-on thinking、默认 high effort、API/Bedrock/Vertex/Foundry/AWS 平台、输入/输出价格和 2026-06 知识/训练截止字段。
- 2026-09-23 当前复验：Artificial Analysis 详情为 `4,006,915` bytes、SHA-256 `d80b226b7a4756cac93d6a065f400f39c9f756b971b827ecf0efbd5a67bffaeb`；主配置 Intelligence Index `53.3549259623252`、median output speed `65.4865856934115 tokens/s`、median TTFT `298.446428812s`、context `1,000,000`、cost per Intelligence Index task `7.629706364004841`。AA 字段仍是第三方 provider/configuration 测量。
- 2026-09-21 复验：Artificial Analysis 详情为 `3,854,152` bytes、SHA-256 `bc83faa8117eebdd2ff28660800be4abf7016af7e10511cb4783f2ca12fbc02a`；主配置 Intelligence Index `53.3549259623252`、median output speed `68.7301566560472 tokens/s`、context `1,000,000`、cost per Intelligence Index task `7.629706364004841`。这些是第三方配置级测量。DataCurve 当前没有精确 `mini_swe_agent_claude_fable_5_1_*` 行，不能迁移 Fable 5 的 `316/452` 结果。
- 2026-09-23 官方续抓哈希：Anthropic 发布页 `534,503` bytes / `610f2e5cf15100fdc85edf0c4bee750878cdbf2db3520606aed088f757bd33f2`；AA 中文首页 `1,783,605` bytes / `7a954aed8916ec9c5c88380cfc159274573dc9547182032ca142f556c0fc2916`；DataCurve `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。System Card 原始 PDF 哈希未变。
- System Card 正文已用 [`pdf_text_extract.js`](code/pdf_text_extract.js) 提取到 `/tmp/fable51-system-card-extracted-20260923.txt`（`376,181` bytes，SHA-256 `d61d0a99b770ae104150ceb2ee04e0ee02fc4a728d7521ee0319b3742ed091e5`）。正文确认 Fable/Mythos 共享相同模型权重、差异在 safeguards；训练数据由公开互联网、公共/私有数据和合成数据组成，使用 deduplication/classification/ClaudeBot，knowledge cutoff 为 2026-06；RSP 判定为 CB-1、未达 CB-2，autonomy threat model 1 适用但 low、threat model 2 未达。能力和安全数字已绑定具体 snapshot、effort、tools、fallback、safeguards 与 verifier。
- 正文代表性结果：Terminal-Bench 4.0 `55.8%`、Terminal-Bench-Science 0.1 `52.6%`、CursorBench `73.4%`、OSWorld partial/strict `77.9%/41.7%`、GDPval-AA v2 `1853`、AutomationBench `31.4%`、ProgramBench `87.6%`；Gray Swan IPI `k=1/10/15` 为 `0.1%/0.7%/1.0%`，较强 Shade coding attacker 无 probes/启用 probes 为 `56.87%/12.80%`，browser auto `0/110`。这些是 Anthropic System Card 条件结果，不是本地复现或 Fable 裸模型分数；cyber 多数结果针对 Mythos/helpful-only 或关闭 safeguards 的配置。
- 页面/发布方自述：定位为 demanding reasoning 与 long-horizon agentic work；页面列出 preserved thinking、跨轮模型切换、per-message effort/turn-scoped system messages/工具间进度更新和 content provenance 等 beta/协议入口。迁移时必须处理 forced tool use 报错、旧模型读取 Fable 5.1 thinking blocks 失败、编辑历史 turn 使 thinking blocks 失效等 breaking changes。发布页还确认 Fable 5.1 与 Mythos 5.1 共享底模但 safeguards 不同、cache read 为 `$0.25/M`，并公开发布方 benchmark、安全策略和科学工作流案例。
- 论文检索：[arXiv 标题查询](https://arxiv.org/search/?query=%22Claude+Fable+5.1%22&searchtype=title) 返回 0 篇；[全文查询](https://arxiv.org/search/?query=%22Claude+Fable+5.1%22&searchtype=all) 返回 4 篇外部使用/评测论文，未发现 Fable 5.1 专属技术报告。仍待核验参数规模、稠密/MoE 架构、完整训练/后训练配方、独立 benchmark 复现、真实 API capability probe 和目标硬件/生产 SLO；System Card 正文安全数字本身已经解析，但仍属于发布方条件证据。
- 外部论文：[`2609.15597`](https://arxiv.org/abs/2609.15597)、[`2609.15494`](https://arxiv.org/abs/2609.15494)、[`2609.10420`](https://arxiv.org/abs/2609.10420)、[`2609.08847`](https://arxiv.org/abs/2609.08847)；这些论文把 Fable 5.1 作为协作工具或评测对象，不能当作 Anthropic 的模型技术报告。
- 研究笔记：[`claude-fable-5.1-source-notes.md`](claude-fable-5.1-source-notes.md)。

## Claude Fable 5

- 排行榜发现：[Artificial Analysis](https://artificialanalysis.ai/models/claude-fable-5)；[DataCurve DeepSWE v1.1](https://deepswe.datacurve.ai/) 的 `claude-fable-5` `xhigh` 配置。
- 官方模型页：[Claude Fable 5 overview](https://platform.claude.com/docs/en/models/fable-5/overview)、[模型说明](https://platform.claude.com/docs/en/models/fable-5/introducing-claude-fable-5-and-claude-mythos-5)、[Prompting](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5)。
- 官方 API/Agent 资料：[Thinking](https://platform.claude.com/docs/en/build-with-claude/thinking)、[Effort](https://platform.claude.com/docs/en/build-with-claude/effort)、[Refusals and fallback](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback)、[Fallback credit](https://platform.claude.com/docs/en/build-with-claude/fallback-credit)、[Memory tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool)、[Programmatic tool calling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling)、[Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction)、[Context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing) 和 [Task budgets](https://platform.claude.com/docs/en/build-with-claude/task-budgets)。
- 生命周期与安全：[Claude Fable 5 and Claude Mythos 5 announcement](https://www.anthropic.com/news/claude-fable-5-mythos-5)、[Redeploying Claude Fable 5](https://www.anthropic.com/news/redeploying-fable-5)、[System Card](https://www.anthropic.com/claude-fable-5-mythos-5-system-card)。
- 已核验：`claude-fable-5`、Artificial Analysis 的 `max + Opus 4.8 Fallback` 配置、DataCurve 的 `xhigh + mini-swe-agent` 组合、2026-06-09 发布字段、1M context、128K max output、adaptive always-on thinking、默认 `high`、拒答/fallback 协议，以及 compaction、context editing、memory、programmatic tool calling 和 task budgets 的运行时边界。
- 证据边界：两个榜单的分数和配置不能合并为裸模型能力；Anthropic 当前资料没有公开参数规模、内部架构、训练/后训练 recipe 或可独立复现的 Fable 5 技术报告。当前状态为资料级闭环，不新增独立架构章节。
- 研究笔记：[`claude-fable-5-source-notes.md`](claude-fable-5-source-notes.md)。

## Claude Opus 4.8

- 排行榜发现：[Artificial Analysis Claude Opus 4.8](https://artificialanalysis.ai/models/claude-opus-4-8)；[DataCurve DeepSWE v1.1](https://deepswe.datacurve.ai/) 的 `claude-opus-4.8` `xhigh` 与 `max` 配置。
- 官方发布：[Introducing Claude Opus 4.8](https://www.anthropic.com/news/claude-opus-4-8)；[Claude Opus 4.8 System Card](https://www.anthropic.com/claude-opus-4-8-system-card)。
- 官方关联工程资料：[Introducing dynamic workflows in Claude Code](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code)。
- 已核验：`claude-opus-4-8`、2026-05-28 发布日期、普通/fast mode 价格边界、默认 `high` effort、`xhigh`/`max` 使用建议、Messages API 任务中途 system entries，以及 Dynamic Workflows 的动态编排、并行 subagents、独立检查和长任务断点恢复描述。
- 评测边界：Artificial Analysis 的 `max` 条目约 41.99 Intelligence Index、1M context、约 56.1 output tokens/s 和约 `$4.08`/task；DeepSWE 的 `xhigh` 约 54.36% Pass@1、`max` 约 58.97% Pass@1。两者绑定不同配置、任务集、provider、harness、工具和 verifier，不能合并为裸模型分数。
- 证据边界：Anthropic 的公告、博客和 system card 入口没有公开参数规模、层/专家结构、训练/后训练 recipe 或可独立复现的 Opus 4.8 技术报告；Artificial Analysis 当前页面标记该条目 deprecated 并指向 `claude-opus-5`，因此保留为历史锚点，不写成当前推荐模型。
- 当前状态：资料级闭环，不新增 Opus 4.8 专属架构章节；研究笔记：[`claude-opus-4.8-source-notes.md`](claude-opus-4.8-source-notes.md)。

## Claude Sonnet 5

- 排行榜发现：[Artificial Analysis](https://artificialanalysis.ai/models/claude-sonnet-5)；[DataCurve DeepSWE](https://deepswe.datacurve.ai/)。
- 官方资料：[Anthropic Models Overview](https://platform.claude.com/docs/en/models/overview)、[Sonnet 5 overview](https://platform.claude.com/docs/en/models/sonnet-5/overview)、[发布公告](https://www.anthropic.com/news/claude-sonnet-5)、[System Card](https://www.anthropic.com/claude-sonnet-5-system-card)、[Adaptive thinking](https://platform.claude.com/docs/en/build-with-claude/thinking)、[Effort](https://platform.claude.com/docs/en/build-with-claude/effort)、[Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)、[Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction)、[Programmatic tool calling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling)。
- 2026-09-15 复核确认：`claude-sonnet-5`、`releaseDate: 2026-06-30`、proprietary/非 open weights、parameters 为 null、1M context、128K 普通/300K Batch 最大输出、Adaptive、默认 high effort、Fast、平台与价格字段。Artificial Analysis 历史主配置 `max` 的 Intelligence Index 为 `38.3576962882576`、约 80.00 output tokens/s、约 202.63s TTFT、约 `$5.0912`/task；这些是第三方配置级字段。
- 2026-09-21 当前快照复核：AA 主配置 `max` 为 Intelligence Index `38.1638712882576`、median output speed `86.1790204452512 tokens/s`、median TTFC `137.660255319s`；相对 9 月 15 日的漂移只作为 provider/测量时点变化记录，不解释为 revision 或训练变化。当前 AA 快照 `/tmp/aa-sonnet5-20260921.html` 为 3,872,014 bytes，SHA-256 `2bd51075cf1ad426140a25dddb097e62af0ef4d07308c00fbc8c022b9158318e`。
- DataCurve DeepSWE v1.1 当前为 113 tasks/91 repositories/5 languages、统一 `mini-swe-agent`；Sonnet 5 的 max/xhigh/high/medium/low Pass@1 分别为 53.846%/49.667%/48.230%/39.778%/30.512%，对应的平均成本约为 `$26.40/$11.89/$7.43/$4.08/$2.19`。所有数字绑定 effort、harness、工具、任务集、环境和 verifier，不能写成裸模型能力。
- 面试技术点：adaptive thinking 不接受手动 `thinking.type: "enabled"` + `budget_tokens`；effort 是行为信号而非严格预算，`max_tokens` 是思考/工具/文本共享的硬上限；thinking blocks/signatures 需按异构 `content` 回放；新 tokenizer 约增加 30% token；context awareness、`compact-2026-01-12` server-side compaction、`computer_toolset_20260801` 和 programmatic tool calling 共同构成长任务运行时。
- System Card/发布方结果已单独记录：SWE-bench Verified 85.2%、SWE-bench Pro 63.2%、Multilingual SWE-bench 78.3%、Terminal-Bench 2.1 80.4%、BrowseComp 84.7%、OSWorld-Verified 81.2%、GDPval-AA v2 Elo 1618、Toolathlon Pass@1 54.3%、AA-Briefcase Elo 1393；这些不替代独立复现。卡片还报告 adaptive+max、通常 5 trials、BrowseComp 10M token limit 和约 200K compaction 触发语义。
- 2026-09-21 System Card 深读补齐训练公开性与安全边界：训练数据只公开为包含公开互联网、公开/私有数据和合成数据的专有混合数据，并提及去重、分类、ClaudeBot 爬取和 post-training/fine-tuning，没有公开完整 recipe；RSP 中不跨 automated AI R&D threshold、Autonomy threat model 1 且 stealth rate 接近零、CB-2 未跨越；没有网络安全专项训练，报告 ExploitBench/OSS-Fuzz/CyberGym/Firefox 147 和 safeguards 影响。Claude Code 恶意请求拒答率 92.37%，computer-use 恶意任务拒答率 84.68%，Gray Swan IPI 去重后 1,130 个攻击、28 个场景。上述均是发布方 harness 证据，不是内部安全模块或裸模型能力证明。
- arXiv 精确标题检索截至 2026-09-21 返回 0 个结果，Anthropic Research 页面未检出 Sonnet 5 专属技术报告；参数、架构、完整训练/后训练 recipe、内部 adaptive-thinking 机制和独立 benchmark 复现仍待核验。当前状态：资料级闭环，不新增独立架构章节。
- 研究笔记：[`claude-sonnet-5-source-notes.md`](claude-sonnet-5-source-notes.md)。

## Claude Sonnet 4.6

- 排行榜发现：[Artificial Analysis Claude Sonnet 4.6 Adaptive](https://artificialanalysis.ai/models/claude-sonnet-4-6-adaptive)；[DataCurve DeepSWE](https://deepswe.datacurve.ai/) 的 `mini_swe_agent_claude_sonnet_4_6_high`。
- 官方资料：[Introducing Claude Sonnet 4.6](https://www.anthropic.com/news/claude-sonnet-4-6)、[Anthropic Models Overview](https://platform.claude.com/docs/en/models/overview)、[System Card](https://www.anthropic.com/claude-sonnet-4-6-system-card)、[Research](https://www.anthropic.com/research) 及 Thinking/Compaction/Agent 工具文档。
- 已核验：`claude-sonnet-4-6`、2026-02-17 发布日期、coding/computer use/long-context reasoning/agent planning 定位、1M context（发布时 beta）、adaptive/extended thinking、context compaction、tool search、web search/fetch、code execution、memory、programmatic tool calling 和 computer-use prompt-injection 风险边界。
- 评测边界：Artificial Analysis 的 adaptive/max 配置与 DataCurve 的 135/451、Pass@1 `29.9335%`、Pass@4 `56.6372%` 均是配置级或 Agent 系统结果，不能合并为裸模型能力。
- 2026-09-16 夜间中断复验：三条代理对 Artificial Analysis 首页、DataCurve 和 Sonnet 4.6 发布页均返回 HTTP 200；排行榜快照分别为 `1,773,553` 和 `268,313` bytes，三个代理逐字节一致。 `8098/1234` 的模型目录重定向到区域不可用页，`7890` 成功取得真实 `platform.claude.com` 目录（`728,089` bytes，SHA-256 `4c5ba69551075bbff00251943a1765c77be38f998d512fac67902067097cdfff`）。
- 证据边界：没有公开参数规模、内部架构、完整训练/后训练 recipe 或 Sonnet 4.6 专属技术报告；adaptive thinking 内部预算、compaction 摘要格式、tool search 质量和线上 computer-use 接受率仍待核验。当前状态为资料级闭环，不新增独立架构章节。
- 研究笔记：[`claude-sonnet-4.6-source-notes.md`](claude-sonnet-4.6-source-notes.md)。

## Claude Opus 4.6

- 排行榜发现：[Artificial Analysis Claude Opus 4.6 adaptive](https://artificialanalysis.ai/models/claude-opus-4-6-adaptive) 与 [Claude Opus 4.6](https://artificialanalysis.ai/models/claude-opus-4-6)；DataCurve DeepSWE v1.1 当前没有精确的 `mini_swe_agent_claude_opus_4_6_*` 行。
- AA 快照：adaptive `3,497,470` bytes，SHA-256 `0bcf25542c95489db50a83925caf5e32bab672e4f7134126009fb94937b6b32d`；基础页 `3,485,093` bytes，SHA-256 `544de10658d16de8ac25da44e6c6792ab51b64093a5a410f3eb3be1173e6d219`。页面字段为第三方 `releaseDate` `2026-02-05`、Intelligence Index 约 `32`（estimated）、1M context、约 37.7 output tokens/s、约 19.75s TTFT、`$5/$25` 每百万 token，参数未公开。
- 官方资料：[发布页](https://www.anthropic.com/news/claude-opus-4-6)、[System Card](https://www.anthropic.com/claude-opus-4-6-system-card)、[模型页 Markdown](https://platform.claude.com/docs/en/models/opus-4-6/overview.md)、[Adaptive thinking](https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking)、[Effort](https://platform.claude.com/docs/en/build-with-claude/effort)、[Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction)、[Computer use](https://platform.claude.com/docs/en/agents-and-tools/computer-use)、[Tool search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)。
- 已核验：1M context、128K 普通/300K Batch 最大输出、默认 high 与 `max/high/medium/low` effort、adaptive thinking、thinking signature 原样回传、`compact-2026-01-12` server-side compaction、约 150K 默认 trigger/50K 最低 trigger、regex/BM25 tool search、最多 5 个 `tool_reference`，以及仍使用 `computer_20251124` 的 computer use 执行器边界。
- 发布方自报的 1M retrieval、BigLaw Bench 和盲测结果不与 AA/DeepSWE 混合；arXiv 本轮没有 Opus 4.6 官方技术报告，检出的两篇论文只是外部使用/评测。完整证据见 [`claude-opus-4.6-source-notes.md`](claude-opus-4.6-source-notes.md)。
- 当前状态：资料级闭环；参数、架构、完整训练/后训练 recipe、adaptive 内部机制、compaction 编码、生产 kernel、硬件 profiling、线上 acceptance rate 和独立复现仍待核验，不新增独立架构章节。

## DeepSeek V4 Flash Vision

- 榜单发现：[Artificial Analysis DeepSeek V4 Flash Vision](https://artificialanalysis.ai/models/deepseek-v4-flash-vision) 的 `max` 配置；本次 [DataCurve DeepSWE](https://deepswe.datacurve.ai/) 快照没有同名行，不迁移 V4 Flash/Pro 的代码 Agent 结果。
- 历史官方发布：[DeepSeek-V4-Flash-Vision-Exp Release](https://api-docs.deepseek.com/news/news260821)，标注 2026/08/21；确认实验性多模态 API、混合文本/图像、Chat/Messages/Responses、base64/URL/Files API、DeepSeek Harness 0.1.1 和发布方的多模态 Agent 定位。
- 当前官方运行时：[Quick Start](https://api-docs.deepseek.com/quick_start)、[Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing)、[Vision](https://api-docs.deepseek.com/guides/vision)、[Files API](https://api-docs.deepseek.com/guides/files_api)、[Responses API](https://api-docs.deepseek.com/guides/responses_api)。当前主名为 `deepseek-flash`，版本标为 V4.1-Flash；旧 `deepseek-v4-flash`/`deepseek-v4-flash-vision-exp` 被说明为 retired 并由 V4.1-Flash 服务。
- 已核验周边技术：图片 detail/resize/token 预算、Files `file_id` 生命周期、Responses `function_call_output(input_image)`、流式事件回放、兼容接口静默忽略边界和峰谷价格。历史公告的 384 image tokens 与当前 guide 的约 1024 tokens/image 必须绑定日期、alias、detail 和 served model。
- 研究笔记：[`deepseek-v4-flash-vision-source-notes.md`](deepseek-v4-flash-vision-source-notes.md)；正式专题：[`第二十一册第 85 章`](../../book-21-transformer-architecture-evolution/chapters/85-deepseek-v4-flash-vision多模态api与路由账本.md)。
- 当前状态：资料级闭环。没有专属视觉架构/训练报告、独立 benchmark 复现或 DataCurve 同名行，因此不把 V4 家族报告自动写成 Vision 变体的完整结构事实。

## Claude Haiku 4.5

- 官方模型目录：[Anthropic Models Overview](https://platform.claude.com/docs/en/models/overview)
- 专属模型页：[Claude Haiku 4.5 overview](https://platform.claude.com/docs/en/models/haiku-4-5/overview)
- 官方关联资料：[announcement](https://www.anthropic.com/news/claude-haiku-4-5)、[system card](https://www.anthropic.com/claude-haiku-4-5-system-card)
- 已核验（官方页面快照）：`claude-haiku-4-5-20251001`、别名 `claude-haiku-4-5`、2025-10-15 发布字段、200K context、64K 最大输出、`fastest` 延迟字段、extended thinking、平台、价格和 2025-02/2025-07 cutoff 字段。
- 仍待核验：参数规模、稠密/MoE 架构、训练/后训练配方、完整推理机制、system card 全文和独立 benchmark 复现。
- 研究笔记：[`claude-haiku-4.5-source-notes.md`](claude-haiku-4.5-source-notes.md)。

## DeepSWE v1.1 排行榜快照

- 官方页面：[DeepSWE](https://deepswe.datacurve.ai/)
- 已保存快照：`/tmp/deepswe.html`；页面更新时间为 2026-09-03，数据对象生成时间为 `2026-09-03T22:24:37.984682+00:00`，显示 113 个任务、91 个仓库、5 种语言，并注明统一使用 [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)。
- 研究笔记：[`deepswe-snapshot-notes.md`](deepswe-snapshot-notes.md)。
- 证据边界：Pass@1、成本、输出 token 和 Agent steps 是模型配置 + harness + 工具 + verifier 的组合结果；不归因于基础模型，也不与 Artificial Analysis 或其他 harness 分数直接合并。

## Artificial Analysis 快照补充

- 本地快照：`/tmp/artificialanalysis.html`，采集日期按本轮记录为 2026-09-09；页面同时列出模型配置、`releaseDate`、开放性分类、参数字段和 Artificial Analysis Intelligence Index。
- 快照可复现标识：文件大小 1,863,708 bytes，mtime `2026-09-09 11:46:15 +0800`，SHA-256 `bd8cfd2e4120012b7f58fe48e6c37117667fe223077a4448b675de8bb91f51f3`。
- 快照中可见的指数示例：Claude Fable 5.1 max with fallback 约 53.37、GPT-6 Astra max 约 52.81、Claude Opus 5 max 约 50.70、GLM-5.3 max 约 44.86、Grok 4.6 high 约 44.41、Kimi K3 max 约 43.78、Gemini 3.8 Flash high 约 41.19。
- 解释边界：这些是第三方榜单配置和页面快照字段；`releaseDate` 不替代官方发布日期，`parameters` 可能是估算，指数不等于基础模型能力，也不能与 DeepSWE 或其他 harness 分数直接拼接。模型名、effort、fallback、任务集和快照日期必须一起记录。
- 例如快照中的 `parameters=753`（GLM-5.3 max）和 `parameters=2800`（Kimi K3 max）仅是第三方字段；未得到官方模型卡或技术报告支持前，不进入模型规格结论。

## DeepSeek V3.2

- 榜单发现：[Artificial Analysis DeepSeek V3.2](https://artificialanalysis.ai/models/deepseek-v3-2)；当前详情页为 `DeepSeek V3.2 (Non-reasoning)`，2025 年 12 月、128K context，第三方 Intelligence Index `16.043537719683`，第三方参数字段约 648B total/37B active。DataCurve 当前快照没有精确 V3.2 行。
- 官方资料：[DeepSeek V3.2 模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)、[技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/assets/paper.pdf)、[V3.2-Exp 仓库](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)、[DeepSeek API News 索引](https://api-docs.deepseek.com/news)。
- 已核验：DSA、scalable RL、large-scale agentic task synthesis、thinking with tools、工具调用模板、`developer` role 限定、V3.2-Speciale 不支持 tool calling、MIT license 和官方报告 PDF。报告正文还给出 DSA 的 dense warm-up/sparse training、2,048 KV top-k、约 2.1B/943.7B tokens、主 attention 复杂度边界、specialist distillation、GRPO 稳定化的 unbiased KL/off-policy masking/Keep Routing/Keep Sampling Mask，以及四类 Agent 任务规模与 verifier 闭环。
- 证据边界：`/news/news251201` 本轮返回 Docusaurus fallback；PDF 文本已用标准库抽取，但公式、图表和字体编码有失真，未把无法逐页视觉核对的内容扩写为事实。完整 DSA kernel 行为、参数/层排布、indexer recall 曲线、KV 字节账本、完整训练超参、线上 acceptance rate、硬件 profiling 和独立 benchmark 待核验。
- 当前状态：AA 单榜资料级闭环；2026-09-20 详情快照为 `3,625,720` bytes、SHA-256 `7d4fd97ed7f2efbc9ab2f001072b353f00e2476cb884abe0d8185f16d7562052`，两个代理逐字节一致；DataCurve 快照仍无精确 V3.2 行。研究笔记：[`deepseek-v3.2-source-notes.md`](deepseek-v3.2-source-notes.md)。本轮活动锚点为 V3.2，正式落点补强到第二十一册第 19 章 19.29--19.33。

### DeepSeek V3.2 官方实现与 serving 补证（2026-09-20）

- [V3.2 固定配置与 encoding](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/tree/a7e62ac04ecb2c0a54d736dc46601c5606cf10a6)：commit `a7e62ac04ecb2c0a54d736dc46601c5606cf10a6` 的 `config.json` 为 1,552 bytes、SHA-256 `c7fa8b191e9936d8e6a57d864baab82b792fae16a116416cdd3a75ba76bc5af1`，核验最终配置 `q_lora_rank=1536`、`kv_lora_rank=512`、61 层、前三层 dense、256 routed/top-8/1 shared、64 index heads、`index_topk=2048`、163840 positions、YaRN factor 40、BF16/FP8 字段；`encoding/encoding_dsv32.py` 为 14,317 bytes、SHA-256 `5e068c2ba2a6e5ebe37a49bb005650c507e7935d77a32f3f7c11ee071498b370`，核验 DSML function-call/result、role、think/reasoning 边界。
- [V3.2-Exp inference demo](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/main/inference)：参考 `model.py`/`kernel.py` 直接展示 FP8 indexer、non-interleaved indexer RoPE、causal/top-k、prefill MHA、decode MQA、latent KV/position cache 和 FP8 KV cache 路径。实验 config 的 `q_lora_rank=1536` 与固定最终 `config.json` 的同值字段仍需按 revision、实现路径和 artifact 分开记录。
- [DeepGEMM PR #200](https://github.com/deepseek-ai/DeepGEMM/pull/200)：FP8 MQA logits、paged MQA logits、SM90/SM100 实现与测试入口。
- [FlashMLA PR #98](https://github.com/deepseek-ai/FlashMLA/pull/98)：sparse prefill、sparse FP8 decode、SM90 sparse MLA、metadata/combine、测试和量化入口。
- [TileLang DeepSeek V3.2 examples](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32)：Lightning Indexer -> radix/histogram Top-k Selector -> Sparse MLA；含 pipelined producer/consumer、double buffering 和 FP8 sparse MLA 研究实现。
- [vLLM DeepSeek-V3.2-Exp recipe](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2-Exp.html)：记录 DeepGEMM 的 MoE/MQA logits 使用、`DP=8, EP=8, TP=1` 推荐、TP fallback、FP8/BF16 KV cache 选择、`max-num-seqs` 调整和 GSM8K recipe 结果。GSM8K 数字绑定 V3.2-Exp + vLLM + lm-eval，不是裸模型能力或 DataCurve 分数。
- 实现证据边界：官方代码覆盖了公开实验实现和特定 serving 路径，但没有证明最终 V3.2 的完整 production kernel、全 GPU 代际 profiling、indexer 召回曲线、线上 tool-call acceptance rate 或独立 benchmark；详见 [`deepseek-v3.2-source-notes.md`](deepseek-v3.2-source-notes.md)。

## Artificial Analysis 2026-09-14 实时快照

- 独立记录：[`artificial-analysis-2026-09-14-snapshot.md`](artificial-analysis-2026-09-14-snapshot.md)。本轮通过 `10.24.27.134:7890` 获取首页 HTTP 200，快照大小 1,771,961 bytes，SHA-256 `ebda1f3ff7fc1956dc629b82ab56400300683a8a134446882d12dfbb4d814423`。
- 页面解析到约 702 个配置条目、673 个唯一名称，最新 `releaseDate` 为 `2026-09-11`；相对 2026-09-09 页面结构新增 13 个 slug/别名，包含 DeepSeek V4.1 Flash、Agnes 3.0 Flash、Ling-3.0-flash-VL、K2 Horizon 系列和 `mbzuai` 目录项。
- DeepSeek V4.1-Flash 与 K2 Horizon MoVA 36B/A4B 已由相应官方资料升级为核验专题；截至本历史快照，Agnes、Ling、K2 Horizon 3.7B 和 `mbzuai` 仍停留在榜单发现层。K2 Horizon 3.7B 已在 2026-09-22 增补官方核验，见后文；第三方 `releaseDate`、参数、开放性和指数不替代官方日期、revision、许可证或独立 benchmark。

## Mistral Small 4

- 官方模型卡：[Mistral-Small-4-119B-2603](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603)
- 已核验：119B/6.5B MoE、128 experts/4 active、256K context、文本/图像输入、`reasoning_effort` 模式、Apache 2.0、EAGLE 草稿头和 NVFP4 检查点。
- 待核验：完整训练配方、专家负载均衡细节、独立技术报告和模型名中 `2603` 的发布日期含义。
- 教学章节：[`第二十一册第 79 章`](../../book-21-transformer-architecture-evolution/chapters/79-mistral-small-4混合推理与eagle量化.md)。

## Step 3.5 Flash

- 官方模型卡：[Step-3.5-Flash](https://huggingface.co/stepfun-ai/Step-3.5-Flash)
- 技术报告：[arXiv:2602.10604](https://arxiv.org/abs/2602.10604)
- 已核验：196.81B/约 11B active、45 层、256K context、288 routed + 1 shared experts、top-8 路由、MTP-3、3:1 SWA/full attention、Apache 2.0 及模型卡评测协议说明。
- 待核验：技术报告中的完整训练数据、RL 细节和不同后端的 MTP3 实现状态。
- 教学章节：[`第二十一册第 80 章`](../../book-21-transformer-architecture-evolution/chapters/80-step-3-5-flash的mtp与滑动窗口moe.md)。

## OpenAI gpt-oss

- 排行榜发现：[Artificial Analysis gpt-oss-120b](https://artificialanalysis.ai/models/gpt-oss-120b) 与 [gpt-oss-20b](https://artificialanalysis.ai/models/gpt-oss-20b)；两个页面均把 `high` 作为推理配置展示。DataCurve DeepSWE 当前没有精确的 `mini_swe_agent_gpt_oss_120b_*` 或 `mini_swe_agent_gpt_oss_20b_*` 行，因此不迁移其他 OpenAI/Codex 模型的 Agent 结果。
- 官方技术资料：[gpt-oss Model Card / arXiv:2508.10925](https://arxiv.org/abs/2508.10925)、[OpenAI gpt-oss GitHub](https://github.com/openai/gpt-oss)、[gpt-oss-120b Model Card](https://huggingface.co/openai/gpt-oss-120b)、[gpt-oss-20b Model Card](https://huggingface.co/openai/gpt-oss-20b)、[Harmony](https://github.com/openai/harmony)。
- 官方运行资料：[gpt-oss Cookbook](https://developers.openai.com/cookbook/topic/gpt-oss)、[Transformers](https://developers.openai.com/cookbook/articles/gpt-oss/run-transformers)、[vLLM](https://developers.openai.com/cookbook/articles/gpt-oss/run-vllm)、[实现兼容性验证](https://developers.openai.com/cookbook/articles/gpt-oss/verifying-implementations)、[raw CoT 处理](https://developers.openai.com/cookbook/articles/gpt-oss/handle-raw-cot)；这些资料用于核对 chat template、MXFP4、工具调用、raw CoT 状态回放和本地 serving 边界。9 月 22 日快照：验证文章 `336,633` bytes / `4f293d2bd51666966a8d3657893eb1d1093b4b3230a405c9178cbc7302d664b8`，raw CoT 文章 `338,262` bytes / `7330273ff59d6e1a344abe3f435ae421a20ebd775bda9122d22ff606f0782072`，专题页 `312,061` bytes / `8027d2c25fc0607e52783669a52914524edd4ea999335b1aae98836fc67d78a0`。
- 已核验：120B/20B 的总参数与 active/token 账本、128/32 experts 与 top-4、交替 sliding/full attention、GQA、RoPE/YaRN、MXFP4、`o200k_harmony` tokenizer、CoT RL、`low/medium/high` reasoning effort、Harmony channel/recipient/structured output 和 browser/Python/function 工具方向。
- 正式专题：[`第二十一册第 87 章`](../../book-21-transformer-architecture-evolution/chapters/87-gpt-oss开放权重moe-harmony与可变推理.md)；研究笔记：[`gpt-oss-source-notes.md`](gpt-oss-source-notes.md)。
- 当前状态：内容专题闭环 + 官方 provider 兼容性/raw CoT protocol evidence。完整训练/后训练 recipe、所有 MoE 负载均衡细节、MXFP4 kernel 与误差消融、目标硬件 profiling、线上接受率和 DataCurve 精确 Agent 评测仍待核验；不把两个尺寸拆成重复架构章节。兼容性 smoke test、AIME/GPQA/HealthBench 质量 eval、kernel/precision 和生产 acceptance 必须分层验收。

## Claude Opus 4.7

- 排行榜发现：[Artificial Analysis Claude Opus 4.7 adaptive/max](https://artificialanalysis.ai/models/claude-opus-4-7) 与 [Claude Opus 4.7 non-reasoning/high](https://artificialanalysis.ai/models/claude-opus-4-7-non-reasoning)；DataCurve DeepSWE v1.1 当前没有精确的 `mini_swe_agent_claude_opus_4_7_*` 行，因此不迁移 Opus 4.6、4.8、5 或其他 Claude 版本的 Agent 结果。
- 榜单字段：Artificial Analysis release date `2026-04-16`、1M context；max 配置 Intelligence Index 约 `40.6898`（estimated），non-reasoning/high 约 `30.9317`（estimated）。这些是第三方配置字段，不是参数或架构证据。
- 2026-09-21 runtime recheck：Artificial Analysis `/zh` 为 `1,777,588` bytes，SHA-256 `9a3ba0c9ba3f23c190f1607073262738c010f255e4b2d4d37a638c73ba409f6f`；max 详情页为 `3,798,671` bytes，SHA-256 `6f97fa78c44721bc9844e91eed41643dea73a736fd46f81a5743e6758f3bff38`。DataCurve 为 `268,571` bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`；三条代理均 HTTP 200 且逐字节一致。AA 详情页标记为 deprecated，但 Anthropic 官方页仍标为 Active (legacy)，两者按来源分别记录。
- Anthropic 官方资料：[发布页](https://www.anthropic.com/news/claude-opus-4-7)、[System Card](https://www.anthropic.com/claude-opus-4-7-system-card)、[模型页](https://platform.claude.com/docs/en/models/opus-4-7/overview.md)、[Effort](https://platform.claude.com/docs/en/build-with-claude/effort)、[Task budgets](https://platform.claude.com/docs/en/build-with-claude/task-budgets)、[Vision](https://platform.claude.com/docs/en/build-with-claude/vision)、[Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction) 和 [Tool search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)。
- 已核验的面试主线：`xhigh` 是位于 `high` 与 `max` 之间的 effort 档位；`effort`、task budget、`max_tokens` 分别对应 step、Agent loop 和单次请求预算；task budget 覆盖 thinking、工具调用、工具结果和输出；server-side compaction 不重置当前 turn 已消耗预算；更新 tokenizer 可能令相同输入变成旧版本的约 `1.0-1.35x` token；Opus 4.7 及以后支持最长边 `2576 px`、最多 `4784` visual tokens 的高分辨率视觉档位。
- 发布页公开的 cyber safeguards 和 Cyber Verification Program 属于部署/安全控制面事实，不能反推出内部训练算法或架构。当前没有公开参数规模、层/专家结构、训练 recipe、生产 kernel、硬件 profiling 或线上 acceptance rate。
- 研究笔记：[`claude-opus-4.7-source-notes.md`](claude-opus-4.7-source-notes.md)；当前状态：内容专题闭环（复用 Agent/tool 与 inference serving 章节），不新增独立 Transformer 架构章节。

## Gemini 3 Deep Think（配置级锚点）

- 排行榜发现：[Artificial Analysis Gemini 3 Deep Think](https://artificialanalysis.ai/models/gemini-3-deep-think)；DataCurve DeepSWE 当前没有精确 `mini_swe_agent_gemini_3_deep_think_*` 行。AA 详情快照 `/tmp/gdt-aa.html` 为 3,249,519 bytes，SHA-256 `b96e3a93bc9f7a882f71523cdc9cdf6023fa2b1536e72a93c579f942d31cd75f`。
- 官方归并核验：[Gemini 3.1 Pro API 模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview)、[Gemini 3.1 Pro Model Card](https://deepmind.google/models/model-cards/gemini-3-1-pro/)、[Gemini 3.1 Pro Model Card PDF](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-1-Pro-Model-Card.pdf)。这些资料把 Deep Think 作为推理设置/评测对象，未提供独立 Deep Think endpoint、权重或专属技术报告。
- 已核验的可观察技术：`thinking_level` 与共同 output budget、`gemini-3.1-pro-preview-customtools` endpoint variant、thought/tool `signature` 和 `id` 的 stateful/stateless 回放、1M context/caching，以及将 test-time compute、能力、延迟、成本和安全风险共同报告的评测边界。
- 研究笔记：[`gemini-3-deep-think-source-notes.md`](gemini-3-deep-think-source-notes.md)。当前状态：资料级闭环（配置级锚点）；不新增独立 Transformer 章节，不把 Gemini 3.1 Pro 的架构/训练资料升级为 Deep Think 独有事实。

## Gemini 3.5 Flash-Lite

- 排行榜发现：[Artificial Analysis Gemini 3.5 Flash-Lite](https://artificialanalysis.ai/models/gemini-3-5-flash-lite)；DataCurve DeepSWE 当前没有精确的 `mini_swe_agent_gemini_3_5_flash_lite_*` 行，不迁移 Gemini 3.5/3.6 Flash 的 Agent 结果。
- AA 2026-09-20 快照 `/tmp/gemini35lite-aa-20260920.html` 为 `3,845,070` bytes，SHA-256 `7de22f529d3b4e2dd440ef8a8e9447480e18dde08437fc508ece06767571e8ef`；第三方字段为 `releaseDate: 2026-07-21`、`isReasoning: true`、Intelligence Index `22.1685424839812`。
- 官方资料：[Google AI Developers 模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite)、[DeepMind Model Card](https://deepmind.google/models/model-cards/gemini-3-5-flash-lite/)、[Model Card PDF](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-5-Flash-Lite-Model-Card.pdf)、[Thinking](https://ai.google.dev/gemini-api/docs/thinking)、[Video understanding](https://ai.google.dev/gemini-api/docs/video-understanding)。
- 已核验：`gemini-3.5-flash-lite`、text/image/video/audio/PDF 输入、text 输出、`1,048,576` 输入 token、`65,536` 输出 token、caching/code execution/Computer Use Preview/File Search/function calling/Maps/Search grounding/structured output/thinking/URL context；默认 thinking 为 `minimal`，支持 `minimal/low/medium/high`。
- Model Card 将模型描述为低成本、低延迟、高吞吐的原生多模态 reasoning model，并明确写出 based on Gemini 3.1 Flash-Lite；架构、训练数据、数据处理、硬件和软件资料全部指向 3.1 Model Card，不能写成 3.5 Lite 独有架构。
- Gemini 视频文档把 3.5 Flash-Lite 列入 agentic video processing：区别 static 1 FPS 与按需时间轴浏览，并以 `processing_call`/`processing_result` 暴露处理步骤；发布方表格的 SWE-Bench Pro `54.2%`、Terminal-Bench 2.1 `54.0%`、GDM-MRCR v2 128K/1M `72.2%`/`21.3%` 等数字只绑定 Google 评测设置。
- 正确的 [What's new in Gemini 3.5 Flash](https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.5) 是 Flash 页面，不能把其 `medium` 默认值、GA 或 thought preservation 迁移给 Lite。完整哈希、证据边界和待核验项见 [`gemini-3.5-flash-lite-source-notes.md`](gemini-3.5-flash-lite-source-notes.md)。
- 当前状态：资料级闭环（AA 单榜）；不新增独立架构章节，内容映射到 Reasoning、Agent/工具协议、多模态视频、长上下文/Serving、评测与安全章节。

## Kimi K2.6：Native Multimodal、Agent Swarm 与 Vendor Verifier

- 排行榜发现：[Artificial Analysis Kimi K2.6](https://artificialanalysis.ai/models/kimi-k2-6)；2026-09-20 三代理详情快照 `/tmp/kimi26-aa-1234.html` 为 `3,938,140` bytes，SHA-256 `f24ad7d9cd3ddbb253470750dcf9f47aea3fc0d675c9f8f7eb8bdbcd50dcfe18`。DataCurve [DeepSWE](https://deepswe.datacurve.ai/) 快照为 `268,571` bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，没有精确 `mini_swe_agent_kimi_k2_6_*` 行。
- 官方模型资料：[Kimi K2.6 Hugging Face 模型卡](https://huggingface.co/moonshotai/Kimi-K2.6)、[固定 config.json](https://huggingface.co/moonshotai/Kimi-K2.6/raw/main/config.json)、[部署指南](https://huggingface.co/moonshotai/Kimi-K2.6/raw/main/docs/deploy_guidance.md)。固定 README revision 为 `7eb5002f6aadc958aed6a9177b7ed26bb94011bb`；模型卡公开 1T/32B、61 层、384 experts/top-8/1 shared、MLA、256K、MoonViT 和 native INT4。部署指南明确 K2.6 复用 K2.5 架构路线，vLLM/SGLang/KTransformers 均有入口。
- 官方 Agent/验收资料：[Kimi K2.6: Advancing Open-Source Coding](https://www.kimi.com/blog/kimi-k2-6)、[Kimi Vendor Verifier](https://www.kimi.com/blog/kimi-vendor-verifier.html)、[Thinking Models API guide](https://platform.moonshot.ai/docs/guide/use-kimi-k2-thinking-model)。博客公开 long-horizon coding、coding-driven design、最多 300 sub-agents/4,000 coordinated steps、proactive agents；KVV 覆盖参数 pre-flight、OCRBench、MMMU-Pro、AIME2025、K2VV ToolCall 和 SWE-Bench。
- 当前 Kimi Code harness 补证：[模型配置](https://www.kimi.com/code/docs/kimi-code/models.html) 当前列出 K3、K2.8 Preview、K2.7 Code HighSpeed 三类、4 个 model ID；[Agent/subagent](https://www.kimi.com/code/docs/kimi-code-cli/customization/agents.html)、[内置工具/AgentSwarm](https://www.kimi.com/code/docs/kimi-code-cli/reference/tools.html) 和[会话](https://www.kimi.com/code/docs/kimi-code-cli/guides/sessions.html)公开 main/subagent 隔离、128 subagent、2 小时超时、resume/model pool、聚合报告、并发控制、唯一工具调用约束、权限复核和 wire event replay。K2.8 未出现在本轮 AA/DataCurve 快照，因此只作周边文档负面证据，不新增候选或研究笔记。
- 研究笔记：[`kimi-k2.6-source-notes.md`](kimi-k2.6-source-notes.md)；正式专题：[`第二十一册第 90 章`](../../book-21-transformer-architecture-evolution/chapters/90-kimi-k2.6-native-multimodal-agent-swarm与推理验收.md)。当前状态为 **AA 单榜资料级闭环**；不迁移 K2.7 Code/K3 的 DeepSWE 结果，不把 Agent Swarm 数量写成模型内部专家结构，不把 KVV 验收结果写成模型能力分数。完整训练 recipe、生产 kernel、INT4 profiling、Agent Swarm coordinator、线上 acceptance 和独立 K2.6 Agent 复现仍待核验。

## GLM-5.1

- 排行榜发现：[Artificial Analysis GLM-5.1](https://artificialanalysis.ai/models/glm-5-1)，页面标题为 `GLM-5.1 (Reasoning)`，并列有 `glm-5-1-non-reasoning`；DataCurve [DeepSWE v1.1](https://deepswe.datacurve.ai/) 当前没有精确 `mini_swe_agent_glm_5_1_*` 行，因此不迁移 GLM-5/5.2/5.3/5.3-Flash 的 Agent 结果。
- 2026-09-20 三代理 Artificial Analysis 详情快照逐字节一致：`3,935,095` bytes，SHA-256 `f6b1ca673b777602013f13eb348a7c48773120e13684eadcbd7220b50b6c137d`。第三方字段约为 744B/40B、200K context、Intelligence Index `26.0585912980095`、约 39.9 tokens/s；这些是目录/provider 测量，不是官方参数或裸模型能力。
- 官方资料：[Z.ai GLM-5.1 文档](https://docs.z.ai/guides/llm/glm-5.1)、[Deep Thinking](https://docs.z.ai/guides/capabilities/thinking.md)、[Function Calling](https://docs.z.ai/guides/capabilities/function-calling.md)、[Context Caching](https://docs.z.ai/guides/capabilities/cache.md)、[Pricing](https://docs.z.ai/guides/overview/pricing.md)、[Release Notes](https://docs.z.ai/release-notes/new-released.md)、[Hugging Face 模型卡](https://huggingface.co/zai-org/GLM-5.1)和[配置](https://huggingface.co/zai-org/GLM-5.1/blob/main/config.json)。
- 已核验：200K/128K、`glm-5.1` API ID、MIT、78 层、256 routed/top-8/1 shared、前三层 dense 字段、`GlmMoeDsaForCausalLM`、`q_lora_rank=2048`、`kv_lora_rank=512`、`index_topk=2048`、202752 positions；以及长周期 8 小时 Agent、实验—分析—优化、multi-turn SFT/RL/process-quality evaluation framework、thinking/工具/缓存协议和本地 serving 入口。
- Z.ai 的 SWE-Bench Pro `58.4`、Linux desktop 655 iterations/6.9×、KernelBench Level 3 3.6× 对比 `torch.compile` max-autotune 1.49× 均是发布方自报，不能与 AA 或 DataCurve 拼接。2026-09-21 重新取得官方 [GLM-5.1 博客](https://z.ai/blog/glm-5.1) 前端资源：壳页面 598 bytes、SHA-256 `6fa12ef1d6f8bd1e834b4d8074da0035e36112641ba609d7855d31f92e2727db`，正文 JS 221,954 bytes、SHA-256 `0e2a4ae9177f44509ee54e9105127d17ff65f6d3b5f95294a7b3b3bdb125c08b`。博客新增可核验的 VectorDBBench 600+ iterations/6,000+ tool calls/21.5k QPS、KernelBench Level 3 的 H100/1,200-turn/双审计器设置和 8 小时无标量目标自评 harness；这些仍是发布方实验设置。arXiv 精确检索没有 GLM-5.1 专属技术报告，模型卡链接的是 GLM-5 报告。
- 当前状态：**AA 单榜资料级闭环**。不新增重复 Transformer 正式章节；完整参数独立披露、DSA indexer loss/recall、训练/后训练 recipe、过程质量 verifier、生产 kernel、硬件 profiling、8 小时 harness、线上 tool acceptance、精确 DataCurve 行和独立复现仍待核验。完整快照和证据边界见 [`glm-5.1-source-notes.md`](glm-5.1-source-notes.md)。

## Grok 4.20 0309 v2

- 排行榜发现：[Artificial Analysis Grok 4.20 0309 v2](https://artificialanalysis.ai/models/grok-4-20)；canonical 配置为 `Grok 4.20 0309 v2 (Reasoning)`。2026-09-20 通过三条代理获取的详情页均为 513,564 bytes、SHA-256 `2fc88248faf152f46f659310f300f2fe3e5b84e419babb9c8c7dc3752452cdd8`；第三方字段为 `releaseDate: 2026-04-07`、Intelligence Index `25.6550155187053`（estimated）、2M context、约 97.01 tokens/s、TTFT 22.71s 和 `$1.25/$2.50` input/output。页面当前还标记 deprecated -> `grok-4-3`，需与官方服务字段分栏。
- [DataCurve DeepSWE](https://deepswe.datacurve.ai/) 当前只有 Grok 4.5 high、Grok 4.6 low/medium/high/xhigh 行，没有精确 `mini_swe_agent_grok_4_20_*`；本轮不迁移相邻 Grok 版本的 Pass@1、成本、输出 token 或 Agent steps。
- 官方身份：[Grok 4.20 模型页](https://docs.x.ai/developers/models/grok-4.20)、[Markdown](https://docs.x.ai/developers/models/grok-4.20.md)、[模型目录](https://docs.x.ai/developers/models.md)；主 ID 为 `grok-4.20-0309-reasoning`，别名含 `grok-4.20`，另有 `grok-4.20-0309-non-reasoning` 与 `grok-4.20-multi-agent-0309`。xAI 页面记录 1M context/max prompt、text/image -> text、structured output/function calling/reasoning、Batch、200K long-context price threshold、37 RPS/10M TPM；与 AA 2M 字段的差异已保留。
- 官方周边：[Reasoning](https://docs.x.ai/developers/model-capabilities/text/reasoning.md)、[Multi Agent](https://docs.x.ai/developers/model-capabilities/text/multi-agent.md)、[Context Compaction](https://docs.x.ai/developers/advanced-api-usage/context-compaction.md)、[Tools Overview](https://docs.x.ai/developers/tools/overview.md)、[Function Calling](https://docs.x.ai/developers/tools/function-calling.md)、[Web Search](https://docs.x.ai/developers/tools/web-search.md)、[X Search](https://docs.x.ai/developers/tools/x-search.md)、[Code Execution](https://docs.x.ai/developers/tools/code-execution.md)、[Remote MCP](https://docs.x.ai/developers/tools/remote-mcp.md) 和 [Release Notes](https://docs.x.ai/developers/release-notes.md)。核心面试线是 leader/sub-agent 研究编排、4/16 agent-count、encrypted state、opaque compaction、server/client tool 分层、parallel function call、搜索引用、sandbox code execution 和 MCP allowlist。
- xAI Release Notes 的 March 2026 条目确认 Grok 4.20 与 Multi-agent 已 live；`x.ai/news/grok-4-20`、`grok-4.20` 和 `grok-4-20-multi-agent` 本轮均 HTTP 404。arXiv 标题精确检索无 `Grok 4.20` 专属报告；全文命中只是外部论文中的被测/背景模型。
- 研究笔记：[`grok-4.20-source-notes.md`](grok-4.20-source-notes.md)。当前状态：**AA 单榜资料级闭环**；没有独立参数/架构、训练 recipe、公开权重、生产 kernel、硬件 profiling、线上 acceptance 或精确 DataCurve Agent benchmark，因此不新增重复 Transformer 正式章节。

## 2026-09-20 Gemini 3.8 Flash 当前快照复验

- 本轮重新抓取 [Artificial Analysis `/zh`](https://artificialanalysis.ai/zh) 与 [DataCurve DeepSWE](https://deepswe.datacurve.ai/)；AA 首页 `1,777,695` bytes / SHA-256 `82b890a93772e857bfafd0aa0a96994d39afb4fcdd2a9fdf2d4e6ec01f9c7912`，DataCurve `268,571` bytes / SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，三条代理均 HTTP 200 且逐字节一致。
- Gemini 3.8 high/medium/low 详情页本轮分别为 `517,803` / `511,687` / `509,053` bytes，SHA-256 分别为 `0618989e412ae947b3a9f499c1cdae16e4ea3f5ee65a25def360249ae33985ef`、`d96907dd254166f038084e7586e30ab869fc8a3225501c37c958650859aec6a8`、`ef35f6f567518bd1b9e9e1bc17e4826ea479b7d1e7868bc35f8daf7c464ee5b3`；AA high 仍为 2026-09-02 release、约 1M context、FAQ 指数约 41。
- DataCurve 精确 high 行为 `gemini-3-8-flash` + `mini-swe-agent`：`n_runs=4`、`n_attempted=447`、Pass@1 `73.8255%`、Pass@4 `85.8408%`、平均成本 `$2.3623`、平均输出约 `143.2K` token、平均 `166.31` steps；这些数字绑定 harness、工具、任务环境和 verifier。
- Google 模型页、Thinking、Interactions 本轮 `7890` 响应分别为 `23,603` / `37,414` / `23,899` bytes，SHA-256 为 `1ef402096c255cc1a960f6981e31cc0a575c791e0afed3920678f523ea6ba4dd`、`47bfe9ed296f388f2aa8021c7431be7bc7de09ffd686598ec442813ef5bb70ae`、`9c9eac247c1c05d0382d5acb3b53fa96ee713d23bfc15d3eed8765505d69e4ea`。
- 当前状态不变：**资料级闭环**。本轮只复验榜单与官方文档，不新增 3.8 独立架构章节，也不把 `thinking`、1M context 或 Agent steps 写成内部训练事实。

## 2026-09-20 GLM-5.3 当前快照与后训练资料补证

- 榜单锚点：[Artificial Analysis GLM-5.3 (max)](https://artificialanalysis.ai/models/glm-5-3)；三代理详情快照 `3,928,329` bytes，SHA-256 `090279e870a3ebb72c7a69f24963f87fe10d63595642a4c61959b911f2a3c3a2`。AA 字段为 release `2026-08-18`、Intelligence Index `44.777392385614`、约 `72.1152 tokens/s`、TTFT 约 `2.99s`、1M context、`$1.40/$4.40`；均标作第三方/provider 配置字段。
- [DataCurve DeepSWE](https://deepswe.datacurve.ai/) 三代理快照 `268,571` bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。精确 `mini_swe_agent_glm_5_3_max` 行为 311/451、Pass@1 `68.9579%`、Pass@4 `87.6106%`、平均成本 `$3.9934`、平均输出 `80.4K` token、平均 `124.47` steps；数字绑定 harness、工具、环境和 verifier。
- 官方产品/迁移资料：[GLM-5.3 文档](https://docs.z.ai/guides/llm/glm-5.3)、[迁移指南](https://docs.z.ai/guides/overview/migrate-to-glm-new)、[Deep Thinking](https://docs.z.ai/guides/capabilities/thinking.md)、[Function Calling](https://docs.z.ai/guides/capabilities/function-calling.md)、[Tool Streaming](https://docs.z.ai/guides/capabilities/stream-tool.md)、[Context Caching](https://docs.z.ai/guides/capabilities/cache.md)、[Structured Output](https://docs.z.ai/guides/capabilities/struct-output.md)、[Chat Completion schema](https://docs.z.ai/api-reference/llm/chat-completion.md)。重点是强制 thinking、`low/high/max`、旧 disabled 请求失败、`tool_stream` delta 拼接、cached token usage 与工具 schema/业务校验边界。
- 官方发布资料：[GLM-5.3 release note](https://docs.z.ai/release-notes/new-released.md) 与 [博客](https://z.ai/blog/glm-5.3)；博客脚本资源本轮可读，补出同基座后训练、可执行长周期环境、无 reference verifier、oracle/no-op/unsolved-state 门禁、SAO with compaction、slime/Megatron/SGLang、training-rollout logprob `1e-7`、top-p mask/top-k/full-vocabulary OPD、R3-style 和 >2.3x RL throughput 等线索。发布方 benchmark 仍不当作独立复现。
- 官方模型卡与实现：[Hugging Face README](https://huggingface.co/zai-org/GLM-5.3) / [config](https://huggingface.co/zai-org/GLM-5.3/raw/main/config.json)、[slime](https://github.com/THUDM/slime)。关联论文为 [IndexCache](https://arxiv.org/abs/2603.12201) 与 [SAO](https://arxiv.org/abs/2607.07508)；SAO 直接论文结论绑定 GLM-5.2，不能自动升级为 GLM-5.3 专属算法。
- 当前状态：**双榜资料级闭环**。榜单、API、模型卡/config、后训练环境/验证器、关联论文/代码均已入库；独立 GLM-5.3 技术报告、完整 recipe、5.3 上 SAO/compaction 精确实现、production kernel、硬件 profiling、独立 benchmark 和线上 acceptance 仍待核验。

## 2026-09-21 DeepSeek V4.1-Flash：`deepseek-recipe` 协议实现补证

- 榜单锚点仍是 [Artificial Analysis DeepSeek V4.1-Flash](https://artificialanalysis.ai/models/deepseek-v4-1-flash)；DataCurve 当前没有精确 `mini_swe_agent_deepseek_v4_1_flash_*` 行。本节只补已确定锚点的官方周边实现，不从 GitHub 仓库另发现模型。
- 官方仓库：[DeepSeek `deepseek-recipe`](https://github.com/deepseek-ai/deepseek-recipe/tree/8cadfede7063c896b944e7bae05daa3549ae97ea)，固定 commit `8cadfede7063c896b944e7bae05daa3549ae97ea`，源码归档 3,996,119 bytes，SHA-256 `1116ca33e9dc62a913fb9214578c400f1704e6bca33487f4a4c31b32c67a21a6`；官方最新提交 Atom 快照为 2,280 bytes、SHA-256 `70e83744340349e637a6569557439a493d6ce5483540d172ea2a6560c255fa05`。GitHub API 本轮因匿名 rate limit 返回 403，但 raw README、官方 Atom 和 pinned archive 成功，故 commit 证据不依赖受限 API。
- README：[固定 raw README](https://raw.githubusercontent.com/deepseek-ai/deepseek-recipe/8cadfede7063c896b944e7bae05daa3549ae97ea/README.md)，5,261 bytes，SHA-256 `0cccc69baa118d7689fc3ff2c2e47ab652e05410b777744c43a424f4db5fc0af`；[streaming guide](https://github.com/deepseek-ai/deepseek-recipe/blob/8cadfede7063c896b944e7bae05daa3549ae97ea/docs/streaming.md) 和 [tokenizer guide](https://github.com/deepseek-ai/deepseek-recipe/blob/8cadfede7063c896b944e7bae05daa3549ae97ea/docs/tokenizer.md) 分别说明协议事件流和显式 tokenizer bridge。
- 源码证据：`deepseek-recipe/src/request/mod.rs` 的 `ConversationRequest` 把 conversation、inference options、parsing options、model 和 stream 分层；`stream/state_machine.rs` 以状态机增量识别 reasoning、DSML tool-call、JSON 和 stop sequence；`deepseek-recipe-encoding/src/v4/dsv41.rs` 固定 V4.1 的中途 system、带前导空格的 DSML 标签和 `low/high/max`→50/75/100 effort 映射；`deepseek-recipe-image` 负责 URL/data URL/bytes、quota、并发、重试和 OpenCV 预处理。
- 适配边界：仓库负责协议转换、prompt/token 编码和 output parsing；不负责模型 inference、HTTP transport、tool execution、权限、业务 verifier。README 还明确列出 `logprobs`、server-side `web_search`、JSON Schema/strict enforcement、`n>1`、`previous_response_id` storage 和 encrypted thinking 等未支持项，不能把这些 adapter 缺口误写成 V4.1 模型能力缺失。
- 安全边界：默认 image limits 为 600 images、32 MiB/image、64 MiB/request、8 concurrent sources；默认 fetcher 为 5 redirects、10s connect、60s request，但不做 SSRF/private-address filtering。`server-py`/`server-rs` 是 mock inference 示例，不能证明权重加载、CUDA/TileLang 或线上 SLO。
- 研究笔记与正式落点：[`deepseek-v4.1-flash-source-notes.md`](deepseek-v4.1-flash-source-notes.md) 与[第二十一册第 81 章](../../book-21-transformer-architecture-evolution/chapters/81-deepseek-v4.1-flash-causal-encoder-decoder.md) `81.10.4`。当前状态：**V4.1 内容专题 + 固定 recipe commit 协议证据补强（AA 单榜）**；完整权重、production kernel、真实 draft/verify/rollback、GPU profiling、线上 tool acceptance 和独立 benchmark 仍待核验。

## 2026-09-21 两榜当前时点复验：无新增重点模型

- 通过三条代理重新获取 [Artificial Analysis `/zh`](https://artificialanalysis.ai/zh)；当前快照为 `1,777,588` bytes，SHA-256 `10630c5152df60351ceff5819c90dfd104f1e9e502959eb36e470abae9ab2c9`。与 2026-09-20 快照相比页面字节和动态字段发生变化，但按 `models/<slug>` 归一化后，重点八家厂商的 canonical slug 集合没有新增或删除。
- 通过三条代理重新获取 [DataCurve DeepSWE](https://deepswe.datacurve.ai/)；三份文件均为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，逐字节一致，并与 2026-09-20 快照相同。当前重点厂商的 `mini_swe_agent_*` 配置集合也没有新增或删除；`mini_swe_agent_kimi_k2_7_code_default` 是已入库的 Kimi K2.7 Code，不是新模型。
- 本轮没有新的重点模型进入候选队列，也没有从官方目录、论文、HF 或 GitHub 另发现模型。排行榜快照只负责发现/复验；effort、fallback、provider、harness 和 verifier 仍按既有模型账本分开记录。
- 下一活动锚点切换为已在 Artificial Analysis 候选队列中的 `GLM-5.1`。它已有 AA 单榜资料级闭环，但没有精确 DataCurve 行和专属技术报告；后续只补 Z.ai 官方博客/发布说明、模型卡/配置、API/代码/论文负面证据及长周期 Agent 的评测边界，不新增重复 Transformer 章节。

## 2026-09-21 Claude Opus 5 当前时点复验

- 本轮沿既有 [Artificial Analysis Claude Opus 5](https://artificialanalysis.ai/models/claude-opus-5) 与 [DataCurve DeepSWE](https://deepswe.datacurve.ai/) 条目继续推进，没有从官方资料另发现模型。
- 当前时点重试 `10.24.27.134:7890`、`10.24.27.134:8098`、`10.237.126.170:1234` 均无法连接；本轮没有新的网页快照。研究笔记中的 2026-09-15 快照仍是历史缓存证据，不能改写为本轮联网成功。
- Opus 5 的官方资料面试主线继续包括 `thinking.display: "omitted"`、thinking block/signature 原样回放、`thinking disabled` 与 `xhigh/max` capability gate、refusal/fallback、fallback credit、工具/effort 中途变更、512-token cache、web fetch 宿主边界和 subagent 预算。参数/架构/训练配方/独立报告/精确本模型 DataCurve 评测仍待核验。

## 2026-09-21 Claude Opus 5 联网恢复后的新鲜快照

- 三条代理均恢复：Artificial Analysis `/zh` 7890 快照为 `1,778,568` bytes、SHA-256 `f4ba39ee5b5638def213f29fa5e9f6aac9954a26cb291f6a6064283f9b0eba93`；Opus 5 详情三条代理逐字节一致，为 `3,868,875` bytes、SHA-256 `e1710da9c158833bad7c1cc8b02dcfdd77d72ea8f6b5652274faa399dde7b615`。DataCurve 三条代理逐字节一致，为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`。
- 当前 AA max 字段为 Intelligence Index `50.7771115797629`、median output speed `60.707552496689 tokens/s`、median time to first chunk `46.6837206345s`、1M context 和约 `$5.8584`/task；这些是第三方配置/provider 字段。DataCurve max 仍为 327/444、Pass@1 `73.6486%`、Pass@4 `88.4956%`、平均成本 `$11.8376`，不迁移相邻 Claude 版本结果。
- Anthropic 模型页 1234 快照为 `461,560` bytes、SHA-256 `57d20b24a8d7961bd2ea76d71080035677ec27deac07991bcc73cc3d305a03b5`；发布页 8098 快照为 `352,773` bytes、SHA-256 `72490a50c0d5c96021954261ed4201d03c41e5134f2432647eed8ac58644c31f`。页面仍支持 1M/128K/300K、adaptive/high、五档 effort、工具中途变更和 automatic fallback；发布方 benchmark 和安全结论仍按自报证据处理。
- AA canonical slug 差异中没有新增八家重点厂商模型；其他新增项不进入本项目候选队列。

## 2026-09-21 GLM-5.3 官方博客正文与评测脚注补证

- 两榜当前时点复验没有新增八家重点厂商 canonical 模型。Artificial Analysis 当前榜单快照为 1,776,713 bytes，SHA-256 为 0d3b7e635e7c785fb94c5ee30c28e0b61fd42ea398f20c4d122e20f1634266c8；DataCurve 快照为 268,571 bytes，SHA-256 为 67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870。
- GLM-5.3 博客壳、正文 JS、source bundle 分别为 598、30,414、245,425 bytes；SHA-256 分别为 240cedb6d23b13b8bdd177e51410dbe1c7783fbd0cfca98be1e0af26688878c0、f809bf586f74b40a01ecd8d674f9ec3de469a5abeae125e2a849c9805605d0f3、e431f2c5e2590dd61672b259c32f49a81d6a0a0304e1311ea29b4b5a1e7b7d69。
- 正文补出 Z.ai Code Bench 相对 GLM-5.2 的 50% 发布方声明、Max/High effort 的完成率与 output-token efficiency、judge/reference-free verifier/solver trajectory/reward-shortcut audit，以及 CyberGym、ExploitBench、ExploitGym 的发现—验证—利用链分层。合作代码库统计为 269 个项目、2,436 个漏洞、1,097 个中高危发现；这些都是发布方证据，不能升级为独立复现。
- 评测脚注补充 DeepSWE 的 mini-swe-agent、temperature 0.95、top-p 1.0、400K context、6 小时；Terminal-Bench 3.0 与 ALE 的 Claude Code、隔离、turn/timeout、Tool Search 和官方 verifier；CyberGym/ExploitGym 的 no-web、域名白名单、single-run 和 TPS 时间归一化；ExploitBench 的 41 tasks、3 revisions、300 interaction rounds。
- 研究笔记：glm-5.3-source-notes.md。当前状态为双榜资料级闭环；上述均为 Z.ai 发布方或页面脚注证据，不能与 DataCurve/AA 分数拼成裸模型结论。仍待核验独立 GLM-5.3 报告、完整 recipe、5.3 专属 SAO/compaction 实现、production kernel、硬件 profiling、独立 benchmark 和线上 acceptance。

## 2026-09-21 Gemini 3.8 Flash Model Card 与长周期 Agent 复验

- 榜单入口仍是 [Artificial Analysis Gemini 3.8 high](https://artificialanalysis.ai/models/gemini-3-8-flash)、medium、low；本轮 `/zh` 为 `1,776,713` bytes / SHA-256 `0d3b7e635e7c785fb94c5ee30c28e0b61fd42ea398f20c4d122e20f1634266c8`，按 canonical slug 归一化后没有新增重点模型。high 详情为 `3,862,728` bytes / SHA-256 `cf66e756c191ab44aad94ff3ab2867f33ce90225d7af185b4252d5f1c91d5785`，当前 AA Intelligence Index `40.9262321765904`、速度 `328.820180257944` tokens/s、1M context、`$0.75/$3.75` 与 `$0.075` cache-hit 均为第三方字段。
- DataCurve 快照为 `268,571` bytes / SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`；精确 high 行为 `n_runs=4`、`n_attempted=447`、Pass@1 `73.8255%`、Pass@4 `85.8408%`、平均成本约 `$2.36`、平均 `166.31` steps，全部绑定 `mini-swe-agent`、工具、环境和 verifier。
- [Gemini 3.8 Flash Model Card](https://deepmind.google/models/model-cards/gemini-3-8-flash/) 本轮为 `156,591` bytes / SHA-256 `c09779a8eac8babcee393031fd1644cded00f7ee6f076224c87374271692f369`；明确 3.8 基于 3.7，1M 输入、64K 文本输出，架构/训练数据/数据处理/软硬件/评测方法均指向 3.7 Model Card。Model Card 的 HLE-Verified `54.9%` 和 multilingual safety `+5.4pp` 回归是发布方评测字段。
- [Gemini 3.8 Flash/Cyber 官方博客](https://blog.google/innovation-and-ai/models-and-research/gemini-models/3-8-flash-and-3-8-flash-cyber/) 本轮为 `407,603` bytes / SHA-256 `7a74091ed7600d91b00e170604633d6d98d7c20a0757591899f94f5af3049603`；公开描述 shared foundational intelligence、long-running agentic loops 递归评估/改进、网络安全训练、额外 reasoning steps 和迭代工具调用。它们是系统行为/发布方叙述，不是已公开的 RL、verifier 或网络结构。
- Google AI Developers 本轮经 1234 返回 503，7890/8098 超时；因此不更新旧文档快照哈希。当前仍为资料级闭环，不新增 3.8 专属 Transformer 章节，也不把 3.8 Flash Cyber 纳入候选盘点。完整研究笔记见 [`gemini-3.8-flash-source-notes.md`](gemini-3.8-flash-source-notes.md)。

## 2026-09-21 GPT-5.6 Luna 当前时点复验与 OpenAI 访问边界

- 继续沿两个排行榜已经发现的 `GPT-5.6 Luna` 推进，没有从 OpenAI 官方目录另发现模型。Artificial Analysis [详情页](https://artificialanalysis.ai/models/gpt-5-6-luna) HTTP 200，快照 `3,861,087` bytes、SHA-256 `00c856c1ecc7bb7d79363f4d2b6814e9cd15a6a02cb8f0c99060862dfbd99cac`；`max` 的第三方字段为 release `2026-07-09`、Intelligence Index `37.3244239690841`、median output speed `164.509614645781 tokens/s`、1M context、约 `$0.20/$1.20` input/output。
- DataCurve 快照 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`；精确 `mini_swe_agent_gpt_5_6_luna_max` 为 301/448、Pass@1 `67.1875%`、Pass@4 `90.2655%`、平均成本 `$0.6056`、平均输出约 `73.4K` token、平均 `101.68` steps。结果绑定 `mini-swe-agent`、工具、环境和 verifier。
- 本轮对 `developers.openai.com` 的模型索引、Luna 模型页、Reasoning 与 Prompt Caching 页面：1234 返回 HTTP `403`，7890/8098 超时，直连 DNS 失败。已有官方快照继续有效，但不把它们写成 2026-09-21 新鲜官方响应。
- 当前状态：**双榜资料级闭环**。GPT-5.6 的运行时 reasoning、persisted state、prompt caching、tool search、compaction 和 harness 资料已入库；参数、架构、完整训练配方、system card、独立报告和生产 profiling 仍待核验。

### DeepSeek V3.2：2026-09-21 当前时点复核

- Artificial Analysis DeepSeek V3.2：当前详情快照 3,638,730 bytes，SHA-256 4dbd4f6cb11f25092a221ef0490a910f91f43885330fe34cc38c212c4d7470c3；结构化条目为 Non-reasoning、2025-12-01、685B/37B、128K、Intelligence Index 16.043537719683、输入/输出 0.28/0.42 美元。以上都是第三方目录/provider 字段；当前对象没有可用 output-speed/TTFT 字段。
- Artificial Analysis /zh：当前排行榜快照 1,776,748 bytes，SHA-256 e73b156ffdc11dd391b48ac9c9f114f31bfab711ba9171a55f66c647cf22c73a；DataCurve DeepSWE 快照 268,571 bytes，SHA-256 67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870，没有精确 mini_swe_agent_deepseek_v3_2_* 行。
- DeepSeek-V3.2-Exp README：本轮快照 6,899 bytes，SHA-256 dffcdf358a42599945d49293a4f210dbe589141085207b76c081c9ace1f8fd74；补证 V3.1-Terminus 对齐对照、2025-11-17 indexer/MLA RoPE layout 修复、TileLang/DeepGEMM/FlashMLA 分层、SGLang dsv32 镜像和 tp=8, dp=8, enable-dp-attention 命令。
- vLLM DeepSeek-V3.2-Exp recipe：当前通过 1234 返回 HTTP 404；仅记为当前 URL/线路负证据，不据此否定已有 recipe 记录或推导 vLLM 没有实现。
- 当前状态仍为 AA 单榜资料级闭环；新鲜目录中的 648B 到 685B 差异按第三方页面漂移记录，不解释为模型 revision 或训练变化。

## 2026-09-22 DeepSeek V3.2 身份归并复核

- [Artificial Analysis `deepseek-v3-2-reasoning-0925`](https://artificialanalysis.ai/models/deepseek-v3-2-reasoning-0925)：页面标为 `V3.2 Exp (Reasoning)`，2025-09-29，deprecated，redirect 到 `deepseek-v3-2-reasoning`。
- [Artificial Analysis `deepseek-v3-2-0925`](https://artificialanalysis.ai/models/deepseek-v3-2-0925)：页面标为 `V3.2 Exp (Non-reasoning)`，deprecated，redirect 到 `deepseek-v3-2`。
- [Artificial Analysis `deepseek-v3-2-reasoning`](https://artificialanalysis.ai/models/deepseek-v3-2-reasoning) 与 [`deepseek-v3-2`](https://artificialanalysis.ai/models/deepseek-v3-2)：页面均显示 2025-12-01 的 V3.2 配置并标 deprecated，redirect 到 V4 Pro 0424。
- [Artificial Analysis `deepseek-v3-2-speciale`](https://artificialanalysis.ai/models/deepseek-v3-2-speciale)：V3.2 专项推理 checkpoint，deprecated，redirect 到 V4 Pro 0424；不能把专项 checkpoint 当作新的基础模型。
- DataCurve 2026-09-22 快照为 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，没有精确 `mini_swe_agent_deepseek_v3_2_*` 行。结论：上述对象属于 V3.2/V3.2-Exp/Speciale 的历史 revision、配置或 checkpoint，不新增 canonical 模型，也不迁移 V3.1/V4 的 Agent 分数。

## Grok 4.7

- 榜单发现入口：[Artificial Analysis Grok 4.7](https://artificialanalysis.ai/models/grok-4-7)；当前标题 `Grok 4.7 (xhigh)`，第三方 release date `2026-09-21`、Intelligence Index `46.4465506302286`、约 `38.7732 tokens/s`、500K context、Intelligence task cost 约 `$3.7383`，proprietary/parameters null。详情快照 `3,882,692` bytes、SHA-256 `e62290c7cc0d937afb8b8e4f08328c9935e6b40a4db9af3029579052b57baa5c`；/zh 首页快照 `1,785,528` bytes、SHA-256 `8d154f41216b952df387e689dbe9a5a254188b08fda9eba1791c4da9fabb1be8`。
- DataCurve [DeepSWE](https://deepswe.datacurve.ai/) 快照 `268,571` bytes、SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，没有精确 `mini_swe_agent_grok_4_7_*` 行；不迁移 Grok 4.6 的 low/medium/high/xhigh 结果。
- 官方：[Grok 4.7 模型页](https://docs.x.ai/developers/models/grok-4.7)，旧快照 `376,912` bytes / `c3ddd6b44b97fb4b527096ca69e4d9eacdca99e0e4e44427c9da5b81a615181e`；本轮 `7890` 刷新为 `376,913` bytes / `add926110deb683b8c90a126340a1f1fa4fc44a6aaa698ea5d3a7d68f537bdd5`；[发布页](https://x.ai/news/grok-4-7)本轮刷新为 `288,007` bytes / `88d0ce52f3c9edfe273d955a1b4bd2547202f5f6a226b14070729703d818f53b`，正文与旧快照一致。官方字段为 `grok-4.7`、text/image -> text、500K、`$2/$0.50/$6` input/cached/output、200K long-context price threshold、Batch unsupported、150 RPS/50M TPM/500K prompt、function calling、structured outputs、reasoning 和 `low/medium/high/xhigh`。
- [Reasoning](https://docs.x.ai/developers/model-capabilities/text/reasoning)：Grok 4.7 的 Responses 每次返回 `reasoning.encrypted_content`，还包括服务端工具的加密输出；后续请求原样回传。该 opaque state 不是可读 CoT；xAI 也公开 `response.reasoning_text.delta`/`response.reasoning_summary_text.delta` 的 summarized reasoning 流。服务端 thinking trace 的 rehydration 与 `store`/`previous_response_id` 存储行为分开，Chat Completions 不提供同样的 ciphertext。本轮 raw HTML `522,822` bytes、SHA-256 `ce3cd1997308094472d2cce8017eee510cf3441c7be2711029870c9d18d1cc87`。
- [Context Compaction](https://docs.x.ai/developers/advanced-api-usage/context-compaction)：`POST /v1/responses/compact` 返回单个 opaque `type=compaction` item；不得修改、裁剪或重排，且不能挽救已经超出 context limit 的请求。本轮 raw HTML `521,486` bytes、SHA-256 `2a58eb48a8ed56d5f747e20fdee8dc9a50cc5998507f3ac0f598afd3d0f48eb3`；正文与旧快照等价。
- [Function Calling](https://docs.x.ai/developers/tools/function-calling)、[Structured Outputs](https://docs.x.ai/developers/features/structured-outputs)、[Remote MCP](https://docs.x.ai/developers/tools/remote-mcp)：分别核验宿主执行、JSON Schema、`server_url`/`server_label`/`server_description`/`allowed_tools`/authorization/headers；Streaming HTTP/SSE 是当前 MCP transport，xAI SDK 使用 `allowed_tool_names`/`extra_headers`，OpenAI Responses API 当前不支持 `require_approval`/`connector_id`。`allowed_tools` 同时影响 schema context 成本和最小权限，但不替代宿主 policy、审计和 verifier。本轮 MCP raw HTML `475,038` bytes、SHA-256 `459b83b7522eaa9139677f9da9760593e56072d52fd63a1883ba6a370a13c705`。
- xAI 发布页公开更大 base、更长 RL run、困难长任务混合、长上下文管理、自验证和新的 safeguard stack；发布方 benchmark 包括 CursorBench 4.0 `46.3%`、DeepSWE v1.1 `71.0%`、EEBench `64.0%`、AA Briefcase v1.1 `1657`、Terminal-Bench 4.0 `38.0%`、Harvey `19.6%`、HealthBench Professional `56.7%`、GDPval `1695`，以及 LatchBio `62.4%`、HackerBench v0.3 危险 prompt 放行率 `3.3%`。这些数字均保留发布方 benchmark/harness/effort 口径，不与 AA/DataCurve 合并。
- [arXiv 精确标题检索](https://arxiv.org/search/?query=%22Grok+4.7%22&searchtype=title)无结果，快照 SHA-256 `8f8c4bb0f6c8a0346e28a56864a4aa4d630bdde9c536ee1320be29f84e6e9da6`。当前状态：AA 单榜内容专题闭环；参数、架构、完整训练 recipe、生产 kernel、硬件 profiling、线上 acceptance 和独立复现待核验。完整记录见 [`grok-4.7-source-notes.md`](grok-4.7-source-notes.md)。

## Qwen3-Omni 30B A3B：Thinker-Talker、AuT 与流式多模态

1. [Artificial Analysis instruct](https://artificialanalysis.ai/models/qwen3-omni-30b-a3b-instruct) 与 [reasoning](https://artificialanalysis.ai/models/qwen3-omni-30b-a3b-thinking)：模型发现入口和第三方页面字段。instruct 三代理快照均为 `3,833,501` bytes，SHA-256 `e62d2c5dac6af3b1b08efa94b58e321df8ac16a813de05d95ad7fabe5eedb316`；reasoning 快照为 `3,845,331` bytes，SHA-256 `22f186bdd723dda3647aca48c7cbbe2d91af1832f08217620c3dad39bff5c748`。DataCurve 当前没有精确 `mini_swe_agent_qwen3_omni_*` 行。
2. [Qwen3-Omni 官方 GitHub](https://github.com/QwenLM/Qwen3-Omni) 与 [raw README](https://raw.githubusercontent.com/QwenLM/Qwen3-Omni/main/README.md)：安装、推理、`disable_talker()`、vLLM Thinker 支持和当前音频输出边界。
3. [Instruct 模型卡](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)、[Instruct config](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/raw/main/config.json) 和 [Thinking config](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking/raw/main/config.json)：模型类、MoE/参数字段和变体边界；不把 Config 当成完整训练 recipe。
4. [Qwen3-Omni Technical Report](https://arxiv.org/abs/2509.17765)、[PDF](https://arxiv.org/pdf/2509.17765) 与 [源码](https://export.arxiv.org/e-print/2509.17765)：Thinker-Talker、AuT、视觉 encoder、TM-RoPE、Talker 多码本/MTP、Code2Wav、三阶段预训练和 Thinker/Talker 后训练。
5. [Qwen 官方博客](https://qwen.ai/blog?id=65f766fc2dcba7905c1cb69cc4cab90e94126bf4&from=research.latest-advancements-list) 与 [阿里云 Qwen-Omni 文档](https://help.aliyun.com/zh/model-studio/user-guide/qwen-omni)：发布方产品/服务入口和 API 约束；不据此推断未公开内部架构或线上 acceptance。
6. 完整研究底稿见 [`qwen3-omni-source-notes.md`](qwen3-omni-source-notes.md)，正式章节见 [`第二十一册第 91 章`](../../book-21-transformer-architecture-evolution/chapters/91-qwen3-omni-thinker-talker-aut与流式多模态.md)。

Qwen3-Omni 的理论首包、发布方 benchmark 和 AA 字段必须与本机/目标硬件实测分栏；Instruct、Thinking、Captioner 是同一模型家族内的不同 artifact，不作为三个新模型入口。

## K2 Horizon 3.7B：dense 长上下文对照

1. [Artificial Analysis K2 Horizon 3.7B](https://artificialanalysis.ai/models/k2-horizon-3-7b)：本轮唯一模型发现入口；第三方字段为 `releaseDate=2026-09-03`、reasoning、open weights、3.7B、524,288 context、Apache 2.0、AA Index `15.6103951330976`。三条代理详情页均为 `3,756,408` bytes，SHA-256 `72b4f94c55add582b0399333552e92b7b4aebd2f493c26830c00e09e58afc39d`。
2. [DataCurve DeepSWE](https://deepswe.datacurve.ai/)：当前 `268,571` bytes，SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`；没有精确 `mini_swe_agent_k2_horizon_3_7b_*` 或 K2 Horizon 行，不迁移其他 K2/模型的 Agent 分数。
3. [IFM/K2-Horizon-3.7B model card](https://huggingface.co/IFM/K2-Horizon-3.7B)：官方模型入口，当前 main revision `6360f705b2e57d542959e6a2e67ebeb95dae0373`，`lastModified=2026-09-21T01:40:06Z`。
4. 当前固定 artifact：`K2HorizonForCausalLM`、36 层、2560 hidden、32Q/8KV、head dim 128、524,288 position、default RoPE、BF16、`num_experts=0`、`mova_num_experts=0`；config/configuration/modeling/index 的 SHA-256 分别为 `a98a4dc771aadcbe03a390d825723a42eaee2682758b29b7d44341d4f33d8ab4`、`5c2f993c1053d9462ebea6dea416c897fddfbb4a5edd904e486936b20d4badc5`、`fb09e010956bd51cfa7d4055b4381cff34c9e06164066b49e3546f38b2e6242f`、`d3b5c4c42227590b76382a9bcc54f868a725bc3c46bea5cdc82e218494759599`。
5. 训练证据：模型卡给出 22.9T/8K pretraining、32K/128K/512K 分阶段 midtraining、512K SFT Phase 1/2，RL 分支包括 Math、Code、STEM-Code，merge 规则为 self-attention ISO merge、其他权重 RAM；这些是发布方高层披露，不是完整 recipe。
6. [vLLM recipe](https://recipes.vllm.ai/IFM/K2-Horizon-3.7B)：更新时间 `2026-09-02`，5.06B dense、512K、H200、`k2_horizon` reasoning/tool parser。[SGLang PR #37654](https://github.com/sgl-project/sglang/pull/37654)：TP1/BF16/FlashAttention-3/H200，文档引用 revision `c177771836a4c460743c00002c22483f6f18d1eb`；该 revision 当前 HF raw/API 404，只记录为旧部署 revision 不可解析。
7. 迁移证据 `/tmp/k2-migration-fixed-20260922.out`：`k2_aurora -> k2_horizon`、`K2HorizonForCausalLM`、copy、`weights_reencoded=false`、BF16、36 shards/327 tensors；它证明 artifact 迁移关系，不证明重新训练或所有 backend 等价。
8. 完整研究底稿见 [`k2-horizon-3.7b-source-notes.md`](k2-horizon-3.7b-source-notes.md)；与 36B MoVA、MOPD 和 Uno 的对照见第二十一册第 82 章 [`k2-horizon-mova-36b-a4b与uno.md`](../../book-21-transformer-architecture-evolution/chapters/82-k2-horizon-mova-36b-a4b与uno.md)。当前状态：**AA 单榜资料级闭环**。

## Qwen3-VL-235B-A22B：Interleaved-MRoPE、DeepStack 与视频时间戳

1. [Artificial Analysis instruct](https://artificialanalysis.ai/models/qwen3-vl-235b-a22b-instruct) 与 [reasoning](https://artificialanalysis.ai/models/qwen3-vl-235b-a22b-reasoning)：模型发现入口；两个页面归并为同一 Qwen3-VL 基础模型的运行配置。AA 当前字段约为 `235B total / 22B active`、`262,144` context；DataCurve 没有精确 `mini_swe_agent_qwen3_vl_*` 行，不迁移其他 Qwen 的 Agent 结果。
2. [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631)：核对 SigLIP2 vision encoder、两层 MLP merger、Interleaved-MRoPE、DeepStack、Video Timestamp、S0-S3 训练 curriculum、square-root normalized loss、SAPO/General RL 和 Thinking with Images。
3. [Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL)：核对官方 README、技术报告、模型卡、Cookbook、GUI/手机 Agent、OCR、视频理解、grounding 和 multimodal coding 入口。
4. [Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct) 与 [Thinking model card](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Thinking)：固定 revision/config；当前实现为 `Qwen3VLMoeForConditionalGeneration`，94 层、64Q/4KV、128 experts/top-8、262K position、视觉 depth 27、patch 16/temporal patch 2、DeepStack `[8,16,24]`。
5. [Transformers Qwen3-VL config](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/configuration_qwen3_vl.py)、[modeling](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py) 与 [processing](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/processing_qwen3_vl.py)：2026-09-22 raw main 快照确认三轴 position ids、视频 timestamp placeholder、DeepStack 前层注入；未固定 upstream commit，不能写成不可变版本或生产 kernel 证明。
6. 完整证据、快照哈希和待核验项见 [`qwen3-vl-source-notes.md`](qwen3-vl-source-notes.md)；正式专题见 [`第二十一册第 92 章`](../../book-21-transformer-architecture-evolution/chapters/92-qwen3-vl-interleaved-mrope-deepstack与视频时间戳.md)。

## 2026-09-22 Claude Opus 5 当前时点复验

- [Artificial Analysis Claude Opus 5](https://artificialanalysis.ai/models/claude-opus-5)：当前详情快照 `3,869,351` bytes，SHA-256 `c18260ab4ff331d5bd3305691db2d4b6051dc2ebe642aa1458c5b8fa2c367643`；AA 字段为 release `2026-07-24`、Intelligence Index `50.7771115797629`、`56.4471785104486 tokens/s`、TTFC `49.2490756305s`、1M context 和约 `$5.8584`/task。9 月 21 日的速度/TTFC 作为历史 provider 测量保留，不能解释为模型 revision。
- [DataCurve DeepSWE](https://deepswe.datacurve.ai/) 当前复验页面为 `268,571` bytes / `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`；精确 `mini_swe_agent_claude_opus_5_max` 仍为 327/444、Pass@1 `73.6486%`、Pass@4 `88.4956%`、约 `$11.8376`、约 117,566 output tokens、约 99.04 steps。没有把相邻 Claude 版本分数迁移到 Opus 5。
- [Anthropic Opus 5 模型页](https://platform.claude.com/docs/en/models/opus-5/overview)、[发布页](https://www.anthropic.com/research/claude-opus-5) 和 [System Card](https://www.anthropic.com/claude-opus-5-system-card)：补证 1M/128K/300K、adaptive/high、effort、thinking block/signature 回放、工具/effort 中途变更、fallback、缓存和发布方安全/评测边界。发布页快照 `352,846` bytes / `7bb18f8e14e20fe2651e4a8308947f53a9541b0dcce6b7549e2b2c244202dce5`；System Card `16,281,258` bytes / `0950dae1ba6b341e4f1a009e535e0f025625efeca21bd043a9a5ae148a3f2e6b`。
- 当前状态：**资料级闭环**。本轮只补当前时点证据，不新增 Opus 5 架构章节或内部训练结论。完整研究笔记见 [`claude-opus-5-source-notes.md`](claude-opus-5-source-notes.md)。

## 2026-09-22 Gemini 3.5 Flash-Lite 当前时点复验

- [Artificial Analysis Gemini 3.5 Flash-Lite](https://artificialanalysis.ai/models/gemini-3-5-flash-lite)：当前详情快照 `3,858,652` bytes / `bbe13cb52c85772080a105b1ec98c42bd71e7194dd67abbc26cb52f9ba1115d`；字段为 release `2026-07-21`、Intelligence Index `22.1685424839812`、`386.373295824977 tokens/s`、TTFC `11.2053542005s`、1M context 和约 `$0.1235`/task。它们是第三方/provider 字段。
- [DataCurve DeepSWE](https://deepswe.datacurve.ai/) 当前页面 `268,036` bytes / `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`，生成时间 `2026-09-22T06:27:15.860279+00:00`；没有精确 `mini_swe_agent_gemini_3_5_flash_lite_*` 行。相邻 `gemini_3_5_flash_high` 不迁移。
- [Google API 模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite)、[Thinking](https://ai.google.dev/gemini-api/docs/thinking)、[Video understanding](https://ai.google.dev/gemini-api/docs/video-understanding) 的新鲜快照分别为 `106,715` / `052846d394f282870f2334d4c079947b45cb3f83faa39de63eb338f78ca5a4a9`、`226,283` / `ab9d354068333aedd8199c5afafd23d36e6b0c6677df531de4d8804b48833951`、`262,508` / `7420a3b0b4baf5a6667371167c86fe8f178a339c9eb6bd1e5108daa789b4125e`。官方字段支持 1,048,576/65,536、`minimal/low/medium/high`、多模态输入、agentic video 和 `processing_call/result`。
- [DeepMind Model Card](https://deepmind.google/models/model-cards/gemini-3-5-flash-lite/) 明确 3.5 Flash-Lite 基于 3.1 Flash-Lite，并将 architecture/training data/hardware/software 指向前代 Model Card；不能把前代资料包装成 3.5 新技术。当前状态：**资料级闭环（AA 单榜）**，完整研究笔记见 [`gemini-3.5-flash-lite-source-notes.md`](gemini-3.5-flash-lite-source-notes.md)。

## 2026-09-22 GLM-5.1 当前时点复验

- [Artificial Analysis GLM-5.1](https://artificialanalysis.ai/models/glm-5-1)：三条代理详情页内容一致；当前快照 `3,971,543` bytes / SHA-256 `c806874d4eb235658b04ba6f527e1ca1db30b2c5e149e8690cd428fd5d2614c9`。页面字段为 `GLM-5.1 (Reasoning)`、release `April 2026`、200K context、Intelligence Index `26.0585912980095`、median output speed `37.1922485381327 tokens/s`、cost per Intelligence Index task `0.9217822401466147`。中文首页当前快照为 `1,798,627` bytes / `4254f2dad0222fd7147ae50047e3bfe0b382ec1aed1c788c94ef9e8f683242fe`。
- [DataCurve DeepSWE](https://deepswe.datacurve.ai/) 当前快照 `268,571` bytes / SHA-256 `67a6b5350a1bc986e814097a87928f5064ba6955ca78be44795e11b2413f2870`，没有精确 `mini_swe_agent_glm_5_1_*` 行；不迁移 GLM-5、GLM-5.2、GLM-5.3 或 GLM-5.3-Flash 的 Agent 结果。
- 官方资料继续使用 [Z.ai GLM-5.1 文档](https://docs.z.ai/guides/llm/glm-5.1)、[官方博客](https://z.ai/blog/glm-5.1)、[HF README/config](https://huggingface.co/zai-org/GLM-5.1)、[Thinking](https://docs.z.ai/guides/capabilities/thinking.md)、[Function Calling](https://docs.z.ai/guides/capabilities/function-calling.md) 和 [Context Caching](https://docs.z.ai/guides/capabilities/cache.md)。博客正文快照仍为 221,954 bytes / `0e2a4ae9177f44509ee54e9105127d17ff65f6d3b5f95294a7b3b3bdb125c08b`；本轮没有新增独立 GLM-5.1 arXiv 报告。
- 当前状态：**AA 单榜资料级闭环**。已具备精确榜单条目、官方文档/模型卡/config、博客、release note、API 协议和负面论文检索；完整参数独立披露、训练/后训练 recipe、indexer loss/recall、过程质量 verifier、生产 kernel、硬件 profiling、长周期 harness、线上 tool acceptance、精确 DataCurve 行和独立复现仍待核验。不新增重复 Transformer 正式章节。

## 2026-09-22 GPT-5.6 Luna 当前时点复验与官方运行时补证

- [Artificial Analysis GPT-5.6 Luna](https://artificialanalysis.ai/models/gpt-5-6-luna)：HTTP 200，当前详情快照 `3,861,318` bytes，SHA-256 `1b425d8f2a93f8418702ac7acbb62b26850b98fe20680f82205bed5fce38fed0`。`max` 的 release `2026-07-09`、Intelligence Index `37.3244239690841`、median output speed `158.728370482714 tokens/s`、cost per Intelligence Index task `0.17829726152289094`、1M context 和 `$0.20/$1.20` 是第三方/provider 字段；9 月 21 日的 `164.509614645781 tokens/s` 和旧页面哈希保留为历史测量。
- [DataCurve DeepSWE](https://deepswe.datacurve.ai/)：当前快照 `268,036` bytes，SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。精确 `mini_swe_agent_gpt_5_6_luna_max` 为 448 attempts、301 passed、Pass@1 `67.1875%`、Pass@4 `90.26548672566371%`、平均成本 `$0.6056233620535714`、平均输出 `73399.70758928571` tokens、平均 101.6808 steps、median peak context `201647`；继续标注为 `mini-swe-agent + tools + task environment + verifier` 系统结果，不迁移 Sol 或其他 GPT 版本成绩。
- [GPT-5.6 Luna model page](https://developers.openai.com/api/docs/models/gpt-5.6-luna.md)：`3,744` bytes / `1f425d8f2a93f8418702ac7acbb62b26850b98fe20680f82205bed5fce38fed0`；通过 `7890` HTTP 200。页面确认 cost-sensitive/high-volume 定位、1,050,000 context、922,000 maximum input、128,000 output、文本/图像输入、文本输出、`none`--`max` effort、Chat Completions/Responses/Batch 和 tool catalog。

## Qwen3.7 Plus

- 排行榜发现：[Artificial Analysis Qwen3.7 Plus](https://artificialanalysis.ai/models/qwen3-7-plus)。当前详情快照 `/tmp/q37-aa-20260922.out` 为 `3,859,009` bytes、SHA-256 `19e9b48bbc9c6d38fab3391bee6353cff5aaa2524bdf04589ae02bbad4a27040`；页面 release 为 June 2026，第三方 Intelligence Index `25.1622215821984`、median output speed `68.5428061089526 tokens/s`、cost per Intelligence Index task `0.32527343198119174`、约 1M context 和约 `$0.40/$1.60` input/output。
- DataCurve [DeepSWE](https://deepswe.datacurve.ai/) 当前快照为 `268,036` bytes、SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`，生成时间 `2026-09-22T06:27:15.860279+00:00`；没有精确 `mini_swe_agent_qwen3_7_plus_*` 行，不迁移其他 Qwen 的 Agent 结果。
- 官方模型文档：[qwen3.7-plus](https://www.alibabacloud.com/help/en/model-studio/qwen3-7-plus)，本地快照 `/tmp/q37-q37-doc-hyphen-20260922.out` 为 `44,386` bytes、SHA-256 `662ebedd3a538e22490e00c4eb29def16c0c5e9ff2de1d3f8fe7d7d334b43656`，页面更新时间 Sep 20, 2026。官方说明当前 alias 功能等价于 `qwen3.7-plus-2026-05-26`，定位为多模态交互式混合 Agent；输入 text/image/video，输出 text。
- 官方推荐模型页：[Model Studio models](https://www.alibabacloud.com/help/en/model-studio/models)，快照 `/tmp/q37-model-studio-20260922.out` 为 `27,344` bytes、SHA-256 `9df37ab1723f032187895b9961865f2831cb575147985a281310d0616f8f4fe8`，更新时间 Sep 22, 2026；在 text generation 和 vision model 列表中列出 `qwen3.7-plus`。
- API 接入资料：[OpenAI-compatible Chat](https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope)，快照 `/tmp/q37-q37-api-20260922.out` 为 `60,448` bytes、SHA-256 `ec3f4a6c3db9ea89fc4141e78cfcfaf389aafa67ff11efec664b3600aa40714`；记录 region-specific base URL、region-bound API key、OpenAI SDK/tool calling 接口和 workspace domain 迁移。
- 官方 capability table 支持多区域的 Function Calling、Structured Outputs、Web Search、Prefix Completion、Context Caching；Batch Inference 主要北京支持，Fine-tuning 标记为 unsupported；Virginia 的 US scope 另不支持 Structured Outputs/Web Search。Context limits 为 1M/991,808/131,072，thinking input 为 983,616，max chain-of-thought length 为 262,144。
- 当前状态：**AA 单榜资料级闭环**。官方文档支持产品/API/runtime 和 region/scope 边界，不能证明参数规模、dense/MoE、视觉 encoder、GUI policy、完整训练 recipe、公开权重、生产 kernel、真实移动设备成功率或独立 benchmark。研究笔记：[`qwen3.7-plus-source-notes.md`](qwen3.7-plus-source-notes.md)。
- [Reasoning](https://developers.openai.com/api/docs/guides/reasoning.md)：`70,253` bytes / `91604df954335250d16e33f3b07ebbfa3e6e2b7f821f0722ad7a69b9d586719a`；补证 GPT-5.6 `standard/pro` mode 与 effort 独立、跨轮 reasoning 默认行为和 `reasoning.context`/reasoning item 的回放边界。
- [Agents](https://developers.openai.com/api/docs/guides/agents.md)：`5,432` bytes / `df4f61b609550619a3f9e445c66d28318b9f681affe1a819ef0098cde32bf43a`；明确 Agents API 托管 Codex harness 和 session/turn/item 状态，Agents SDK 由应用控制 loop、deployment、storage、approval/runtime，Responses API 直接管理 response/history/tool loop；三类 session/conversation/sandbox 资源不混同。
- [Using tools](https://developers.openai.com/api/docs/guides/tools.md)：`33,282` bytes / `4722fa102070178c1a1d603e718c69575c7c1768fa901218eab359bafdcba341`；[Tool search](https://developers.openai.com/api/docs/guides/tools-tool-search.md)：`39,288` bytes / `9d6c3855a4cb722a98fb364618a852fb436e9822a36fe9b54650f75bbc8d1fd8`。tool search 支持 hosted 与 client-executed 两种路径：前者由 OpenAI 返回 `server` side 的 loaded subset，后者由应用执行搜索并以原 `call_id` 回传 `tool_search_output`；两者都把加载工具追加到上下文末端以尽量保护 cache prefix。
- [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching.md)：`47,097` bytes / `c70d858eecd09681cdc671a1a037e7d51916a793eb85c9dc08be240d1cb9b2d1`；[Compaction](https://developers.openai.com/api/docs/guides/compaction.md)：`14,272` bytes / `73fd2fd1afd44bd6f29ce7bd86fd0ae3413c98ec00ae5476879e98171b0b60dd`。GPT-5.6 最小可缓存前缀为 1,024 visible tokens，支持 implicit/explicit breakpoint、最多四次 cache writes、`30m` TTL；`context_management` compaction 会替换旧前缀，stateless chaining 要回放 output items，`previous_response_id` 模式不要手工裁剪。
- 资料状态：**双榜资料级闭环**。这些页面确认可观察的模型合同和 Agent runtime 协议，不公开 GPT-5.6 参数、内部注意力/MoE 结构、完整训练/后训练 recipe、生产 kernel 或独立技术报告；研究底稿为 [`gpt-5.6-source-notes.md`](gpt-5.6-source-notes.md)。

## 2026-09-22 DeepSeek V4.1-Flash 当前时点复验

- [Artificial Analysis 中文首页](https://artificialanalysis.ai/zh)：三条代理均 HTTP 200，当前快照 `1,762,420` bytes、SHA-256 `de45e583216f236e0eb6c0b69b65b251394eea04137368a056c43e072a97a6f6`；[DataCurve DeepSWE](https://deepswe.datacurve.ai/) 三条代理均 HTTP 200，当前快照 `268,036` bytes、SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。本轮没有发现新的重点厂商 canonical 模型。
- [Artificial Analysis DeepSeek V4.1-Flash](https://artificialanalysis.ai/models/deepseek-v4-1-flash)：详情页三条代理均 HTTP 200，快照 `3,948,908` bytes、SHA-256 `114cc90d1cb8125174d9464141cbfd0faeac76f52b132f78630c0375c9fcf8fe`；canonical slug 仍为 `deepseek-v4-1-flash`，AA Intelligence Index `39.456167472527`、约 1M context 和约 `$0.30/$1.20` 为第三方/provider 字段。
- [DeepSeek V4.1 发布页](https://api-docs.deepseek.com/news/news260910)：三条代理均 HTTP 200，快照 `27,838` bytes、SHA-256 `bea79d60a0712f1971554724c94e8ff2145a048d3c0e80326efa4bacd2bf8e11`。发布页与既有记录一致，没有产生新的 API 模型身份或路由 revision。
- [DataCurve DeepSWE](https://deepswe.datacurve.ai/) 当前没有精确 `mini_swe_agent_deepseek_v4_1_flash_*` 行；不迁移 V4 Pro/V4 Flash 或其他 DeepSeek 版本的 Agent 成绩。HF API 经 `10.24.27.134:7890` 取得 `6,714` bytes、SHA-256 `df3cb8b368d3a77a4eb3b96c8a4f85abfb3a199245bf9006a68d374a4b425ca3`，revision 仍为 `dba1be0a40aa45a94ad051997016db3960a90277`，`lastModified=2026-09-10T08:18:10Z`，仍为 48 个 safetensors 分片。
- HF 的 8098 线路返回 503/连接失败，1234 未成功；`deepseek-recipe` README 经 1234/7890 成功、8098 对 GitHub raw 出现 TLS EOF。它们是代理可达性边界，不是页面或 revision 不存在的负证据。
- 当前状态：**内容专题 + reference implementation + recipe protocol evidence（AA 单榜）**。本轮没有新的模型或权重 revision；完整权重加载、production kernel、candidate/index Top-K recall、真实 FP4 误差、DSpark 调度、EPD 调度、目标硬件 profiling、tool acceptance 和独立 benchmark 仍待核验。完整记录见 [`deepseek-v4.1-flash-source-notes.md`](deepseek-v4.1-flash-source-notes.md)。

## 2026-09-22 DeepSeek V4.1-Flash：SGLang main/stable runtime 对照

1. [SGLang main `deepseek_v4.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/deepseek_v4.py)、[`deepseek_v4_dspark.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/deepseek_v4_dspark.py)、[`deepseek_v41_vit.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/deepseek_v41_vit.py) 和 [`deepseek_v4_nextn.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/deepseek_v4_nextn.py) 由 Contents API 固定；源码快照分别为 `241,421` / `4164c354...71ded`、`47,640` / `61dc79f...d7b61`、`5,126` / `29f4d988...c94eb`、`9,459` / `a3ca101d...5d90b`，Git blob 分别为 `a3b8b610...`、`baebc2de...`、`d5777a4f...`、`6694abb6...`。
2. main 代码确认 V4.1 vision 的 TP/EP/DP 路径和 CP/PP/MoE A2A 限制、2D-RoPE ViT/Aligner、DSV4 sparse indexer/unified KV/FP8 路径，以及 DSpark 的 Markov/confidence head、`mtp.*` 映射和 draft stage 的 `vision_n_layers=0`。相关 main commits 为 `a6cf0581...` 和 `7fac84b6...`。
3. [SGLang `v0.5.20` `deepseek_v4.py`](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/models/deepseek_v4.py) 与 [`deepseek_v4_dspark.py`](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/models/deepseek_v4_dspark.py) 的解码源码分别为 `170,193` / `252f6176...0caa4` 和 `40,396` / `59ac079e...fd04`；tag commit `94602c9c...`，发布时间 `2026-09-18T22:41:33Z`。stable 没有 `deepseek_v41_vit.py`，相关文件中 `deepseek_v41`/`dsv41` 计数为 0。
4. 当前准确状态是 **SGLang main runtime implementation evidence**，不是 v0.5.20 stable V4.1 serving 证明；完整权重、目标硬件、视觉 draft/target verify、FP4/FP8 质量、DSpark acceptance/rollback、EPD 和生产 SLO 仍待核验。完整来源、哈希和代理边界见 [`deepseek-v4.1-flash-source-notes.md`](deepseek-v4.1-flash-source-notes.md)。

## 2026-09-22 GLM-5.3 标准 DSA 版：stable/main runtime source evidence

- 本节只补充已经由两个排行榜确认的标准 `GLM-5.3`，不把 `GLM-5.3-Flash` 的 `glm5_next`/linear-attention 实现迁移到标准版。最新两榜复验快照为 Artificial Analysis 中文首页 `1,762,420` bytes、SHA-256 `de45e583216f236e0eb6c0b69b65b251394eea04137368a056c43e072a97a6f6`，DataCurve DeepSWE `268,036` bytes、SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`；八家重点厂商没有新的 canonical 模型。
- [Hugging Face 固定 revision](https://huggingface.co/zai-org/GLM-5.3/tree/aca966e4e02791568aa6a4ced368624b3d897f42) 为 `aca966e4e02791568aa6a4ced368624b3d897f42`；[`config.json`](https://huggingface.co/zai-org/GLM-5.3/raw/aca966e4e02791568aa6a4ced368624b3d897f42/config.json) 为 `29,464` bytes、SHA-256 `3ac72612095574542f7fff847ada8e59d9199dd8af44bdf625d7e02615572e69`。它确认 `model_type=glm_moe_dsa`、`GlmMoeDsaForCausalLM`、78 层、前三层 dense、256 routed/top-8/1 shared、`q_lora_rank=2048`、`kv_lora_rank=512`、`index_topk=2048`、`index_topk_freq=4`、`index_share_for_mtp_iteration=true` 和 1,048,576 positions。
- [Transformers main](https://github.com/huggingface/transformers/tree/0bc252863a4e5c0e709893664cc57369ebcd7353) ref 为 `0bc252863a4e5c0e709893664cc57369ebcd7353`。[配置源码](https://github.com/huggingface/transformers/blob/0bc252863a4e5c0e709893664cc57369ebcd7353/src/transformers/models/glm_moe_dsa/configuration_glm_moe_dsa.py) 为 `7,918` bytes、blob `9759092416021956de13239d9c8e858d22a31b8d`；[模型源码](https://github.com/huggingface/transformers/blob/0bc252863a4e5c0e709893664cc57369ebcd7353/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py) 为 `37,868` bytes、blob `922ec9ee0e18cd6c3f8acca93f49465d35ae86f7`。实现明确区分 interleaved indexer RoPE、Full layer 的 top-k 计算和 Shared layer 对 `prev_topk_indices` 的复用，并构造 sparse attention mask。
- [vLLM main](https://github.com/vllm-project/vllm/tree/81d7293c2167e39f3ffddc9a82d633f94e8a1eaa) ref 为 `81d7293c2167e39f3ffddc9a82d633f94e8a1eaa`；[registry](https://github.com/vllm-project/vllm/blob/81d7293c2167e39f3ffddc9a82d633f94e8a1eaa/vllm/model_executor/models/registry.py) 将 `GlmMoeDsaForCausalLM` 路由到 `vllm.models.deepseek_v32`。[`deepseek_v2.py`](https://github.com/vllm-project/vllm/blob/81d7293c2167e39f3ffddc9a82d633f94e8a1eaa/vllm/model_executor/models/deepseek_v2.py) 为 `78,620` bytes、blob `ed702400be6c21bb2ee63a20ff438eeebad6ab3e`；[`deepseek_v32/attention.py`](https://github.com/vllm-project/vllm/blob/81d7293c2167e39f3ffddc9a82d633f94e8a1eaa/vllm/model_executor/models/deepseek_v32/attention.py) 为 `23,892` bytes、SHA-256 `9bbeeb3696c9848427bdd7c3573df15cdfa67931147ff673be12ed694c5cd326`。对照 [v0.29.0 registry](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/model_executor/models/registry.py)、[`deepseek_v2.py`](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/model_executor/models/deepseek_v2.py)（`77,732` bytes、blob `bc99509c7f45cb20739a215f5b32d7e7fb0af69a`）和 [`deepseek_v32/attention.py`](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/model_executor/models/deepseek_v32/attention.py)（`21,765` bytes、SHA-256 `bba67788421a33a6d7db8cb7a44f91c90d07cd348fb36887026662c5523c424f`），两者均沿 DeepSeek-V3.2 共用路径；main 额外出现 PCP/DCP、`SparseCacheRole.INDEXER`、HiSparse 与 logical top-k 协作。
- [SGLang main](https://github.com/sgl-project/sglang/tree/861b11f087af2822cb545ea1895059a721831014) ref 为 `861b11f087af2822cb545ea1895059a721831014`；[`deepseek_v2.py`](https://github.com/sgl-project/sglang/blob/861b11f087af2822cb545ea1895059a721831014/python/sglang/srt/models/deepseek_v2.py) 为 `141,378` bytes、blob `c2356c907b37ff47fc6b228363685e5b3d2b88f3`。对照 [`v0.5.20`](https://github.com/sgl-project/sglang/tree/v0.5.20) stable（annotated tag ref `d158602ff1d2cb953196c95158c488d503d2470c`）的同文件为 `130,545` bytes、blob `bae009ce8fb3f51cc00c635089624fe54ae65c0b`；两者均包含 `is_glm_moe_dsa`、`DeepseekV32ForCausalLM` 和跨层 top-k 状态。mutable main 的额外代码只能标为 upstream source evidence，不能替代 stable wheel 或本机验收。
- 当前证据等级为：**双榜资料级闭环 + stable/main runtime source evidence**。源码入口、registry 和 recipe 只能证明实现路径存在；完整权重加载、index/evidence recall、MLA/indexer cache 恢复、MTP acceptance、目标硬件 profiling、tool/verifier acceptance 和生产 SLO 仍待核验。标准版不包含 Flash 的 KDA、视觉、`RadixLinearAttention`、双 state pool 或 `glm5_next.py` 结论。

## 2026-09-23 Claude Opus 5.5

- [Artificial Analysis Claude Opus 5.5](https://artificialanalysis.ai/models/claude-opus-5-5)：三条代理详情页逐字节一致，`3,824,824` bytes / SHA-256 `ed037387bd96b9242985d77657b3cd094000f080854882e0c05ab04d4abb9904`；`releaseDate=2026-09-22`、`contextWindowTokens=1000000`、Intelligence Index `57.6223698102963`、`isOpenWeights=false`、`parameters=null`、input/output `$4/$20`、cache hit `$0.20`。这些是 AA 第三方/provider 字段。
- [Artificial Analysis 中文首页](https://artificialanalysis.ai/zh)：最新续抓三条用户代理逐字节一致，`1,783,592` bytes / SHA-256 `7880a715fd069d4e475e4d717b37971d1f6402e5492964bb3f58787524d0b5a4`；此前同日 `1,783,001` bytes 的 `2fd4bd...` 快照作为历史记录保留。首页前列出现 `Claude Opus 5.5`，并同时出现 `GPT-6 Sol`、`Grok 4.7` 等其他重点厂商 canonical 条目；本轮切换到 Opus 5.5，不把 effort 变体重复建档。
- [DataCurve DeepSWE](https://deepswe.datacurve.ai/)：`268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`；当前没有 `mini_swe_agent_claude_opus_5_5_*` 精确行，不迁移 Opus 5/Fable 5.1/其他 Claude 的 Agent 结果。
- [Anthropic Claude Opus 5.5 发布页](https://www.anthropic.com/claude-opus-5-5)：`619,938` bytes / SHA-256 `1b4b50a9df4c7f24d786b811c88f372ef066e4cc71ed7d43a5a1b3509136f293`；正文明确 2026-09-22、Claude 5.5 家族首个模型、长任务 coding、token/cache 成本、benchmark 条件、Cyber/Life Sciences Verification、fallback 和 preserved thinking anti-distillation。
- [Claude Opus 5.5 System Card](https://www.anthropic.com/claude-opus-5-5-system-card)：PDF `17,795,106` bytes / SHA-256 `7311c9c6bbb16d012f1c12c7418b05949fcf7ae3e30d2c40f22050074b2a7378`；已用 Node.js `zlib` 与 PDF ToUnicode/CMap 只读解析正文。训练数据边界为公开互联网、公开/私有、获许可用户和合成数据，knowledge cutoff 为 2026-06；评测多使用最终 snapshot，但部分章节使用早期/替代 snapshot 或关闭生产 safeguards，数字必须绑定配置。
- [Claude Opus 5.5 model page](https://platform.claude.com/docs/en/models/opus-5-5/overview.md)：7890 代理取得 Markdown `3,184` bytes / SHA-256 `aa9389d5f328023dea11650d26898b43b4e04969a4facd327881df145a307927`；确认 `claude-opus-5-5`、1M context、128K synchronous max output、always-on adaptive thinking、default `medium` effort、各平台 model ID、价格、512-token cache minimum 和 300K batch beta。1234/8098 的区域重定向仅保留为线路访问边界。
- [What's new in Claude Opus 5.5](https://platform.claude.com/docs/en/models/opus-5-5/whats-new-opus-5-5.md)：Markdown `5,972` bytes / SHA-256 `90fb6efa547a193cbf1eb4b836ef5310234da054f2e83abbe15ce41b0d4c5a6a`；确认 always-on thinking、强制工具调用错误、thinking block model/conversation binding、`computer_toolset_20260801`、progress-update blocks、inline tools、on-demand compaction、refusal/fallback 和 fast mode 契约。
- [Claude Opus 5.5 migration guide](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide.md)：Markdown `4,740` bytes / SHA-256 `0c8f717ab25184446431b1c579905a2458e4917258cfb71d66b65737eb3feab5`；固定从 Opus 5 迁移的 model ID、thinking、tool choice、computer use 和 block 回放变更。
- [Anthropic fast mode](https://platform.claude.com/docs/en/build-with-claude/fast-mode)：Markdown `6,962` bytes / SHA-256 `c23e9562b6c76c530dd95ca51578a360bbf0304166839fef2f8ec38db746e913`；确认同一模型的更快推理配置、最高约 2.5x OTPS、API-only research preview、独立限流、`usage.speed` 和 premium pricing。
- System Card 的能力评测包括 Terminal-Bench 4.0 `66.36%`（xhigh、Claude Code `--bare`、5 trials，约 2.5% requests fallback，影响约 10% trials）、ProgramBench `91.2%`（166 golden tasks）和 OSWorld 2.0 partial/strict `81.8%/48.7%`（108 tasks、1080p、最多 500 actions、5 runs、>100K tokens 后 server-side compaction）。五 Agent team 在同分 ProgramBench 下约 `2.7x` latency improvement；DRACO 在 `0.5x` latency budget 下约 `2.8x` speedup；这些是 derived latency/harness 结果，不是裸模型分数。
- System Card 的 RSP 为 CB-1、未达 CB-2；autonomy threat model 1 适用但整体 low，threat model 2 未达。CoBench 2.1 为 Opus 5.5 `55.8%`，Mythos 5.1 `53.4%`、Opus 5 `53.2%`；AECI `169.36`，global CI `[165.23, 177.05]`、local CI `[167.99, 171.24]`。新 fit 使用 374 benchmarks、7,985 observations、732 models，不能和旧 fit 直接比较。
- Cyber 评测在关闭 cyber safeguards 下报告 ExploitBench `13.99` flags、`91%` Cap%、完整 ACE `301/410 = 73.4%`、CyScenarioBench `67.6%`、ExploitGym 2h/6h `289/869`、`300/869`；不能写成生产 cyber 成绩。Claude Code malicious refusal `79.8%`、dual-use/benign success `99.8%`、malicious computer-use refusal `79.46%`；Gray Swan IPI k=1/10/15 为 `0.1%/0.7%/1.0%`，18% rollout fallback 到 Opus 4.8，1,310 个 fallback rollout 无成功攻击。
- alignment/robustness 还报告 sandbox escape `1.5%`（均 low severity）、模拟 registry 高风险场景约一半采取可能有害动作；早期 snapshot 粘贴恶意指令/不可见字符执行约 `52%/26%`，最终 default/max 约 `2%/7.4%`，产品 mitigations 后为零。早期 snapshot、关闭 safeguards、模拟环境和 fallback 都是证据条件，不能迁移为生产结论。
- fallback 规则按风险类别区分：biology/CB -> Opus 5，cyber -> Opus 4.8，frontier LLM kernel 等窄范围 AI R&D -> Opus 5；conventional weapons/high-yield explosives 与 distillation/hidden reasoning extraction 无 fallback，其他 provider 可能不同。

当前状态：**AA 单榜 + System Card 正文证据**；官方资料已支持 token efficiency、effort/cost 曲线、API compatibility、Agent harness、safeguard/fallback、反蒸馏和安全评测条件的面试主线，但不支持参数、架构、完整训练 recipe、DataCurve Agent 分数、System Card 数字的独立复现或生产 acceptance 结论。完整整理见 [`claude-opus-5.5-source-notes.md`](claude-opus-5.5-source-notes.md)。

## 2026-09-23 GPT-6 Sol

- [Artificial Analysis GPT-6 Sol](https://artificialanalysis.ai/models/gpt-6-sol)：三条代理详情页逐字节一致，`547,729` bytes / SHA-256 `88841ae9837f189a6ab739d8fa2bad6d142163b5b6486cab9181ebbfbc75f116`；标题为 `GPT-6 Sol (max)`，release 为 2026 年 9 月，max Intelligence Index `47.5276426437724`、median output speed `115.205383643174 tokens/s`、cost per Intelligence Index task `1.0564240894076389`。这些是 AA 第三方/provider 字段。
- [Artificial Analysis 中文首页](https://artificialanalysis.ai/zh)：`1,783,001` bytes / SHA-256 `2fd4bd27a526ee60d807b9b596f0f3a8a4faa60223d5a3a21fca1ea643ec2bbf`；当前首页展示 GPT-6 Sol、Claude Opus 5.5、Grok 4.7 等重点厂商 canonical 条目。本轮选择 GPT-6 Sol 作为活动锚点，不把 effort/provider 变体重复建档。
- [DataCurve DeepSWE](https://deepswe.datacurve.ai/)：`268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`；当前没有精确 `mini_swe_agent_gpt_6_sol_*` 行，不迁移 GPT-6 Astra、GPT-5.6 或其他 GPT 版本的 Agent 结果。
- [GPT-6 Sol model page](https://developers.openai.com/api/docs/models/gpt-6-sol.md)：`1,664` bytes / SHA-256 `5566589803b6f0ac79c00125a7c1af9291e46d5724c1c414f932d36ab92a8ad0`；确认 `gpt-6-sol`、complex coding/agentic workflows、text/image input、text output、1,050,000 context、922,000 maximum input、128,000 maximum output、2026-04-20 knowledge cutoff、`none`--`max` effort、Responses/Chat Completions/Batch、supported features/tools、272K whole-request pricing threshold、Batch/Flex 50% 和 Fast mode 2x。
- [Reasoning](https://developers.openai.com/api/docs/guides/reasoning.md)：`70,315` bytes / SHA-256 `93327c5eb19df7ecf8c0bd9d581f58052dee9f090c38bb48906110b08d0ce251`；GPT-6 family 支持 `standard/pro` mode，mode 与 effort 独立；`configuration_update` 可在标准单 Agent 会话中调整后续 effort，但不能与自动 compaction/truncation 组合，更新项不能相邻。
- [Agents](https://developers.openai.com/api/docs/guides/agents.md)：`5,432` bytes / SHA-256 `df4f61b609550619a3f9e445c66d28318b9f681affe1a819ef0098cde32bf43a`；区分 Agents API 托管 Codex harness、Agents SDK 应用控制 loop/deployment/storage/approval/runtime 和 Responses API 的 response/history/tool loop；session、conversation、sandbox 是不同资源。
- [Using tools](https://developers.openai.com/api/docs/guides/tools.md)：`33,282` bytes / SHA-256 `4722fa102070178c1a1d603e718c69575c7c1768fa901218eab359bafdcba341`；官方通用接口涵盖 web/file search、function calling、remote MCP、skills、shell、computer use、tool search 和 programmatic tool calling。模型页的工具支持只证明 capability surface，不证明模型内部架构。
- [Compaction](https://developers.openai.com/api/docs/guides/compaction.md)：`14,272` bytes / SHA-256 `73fd2fd1afd44bd6f29ce7bd86fd0ae3413c98ec00ae5476879e98171b0b60dd`；server-side compaction 由 `context_management.compact_threshold` 触发并返回 encrypted compaction item；standalone `/responses/compact` 返回 canonical next context，不能随意裁剪。stateless chaining 要回放 output items，`previous_response_id` chaining 不要手工 prune。
- 当前状态：**AA 单榜资料级闭环**。官方资料支持模型身份、预算、effort、端点、工具与 Agent runtime/compaction 协议；不支持参数规模、dense/MoE、attention 变体、训练/后训练 recipe、system card、公开权重、生产 kernel、目标硬件 profiling、DataCurve Agent 分数或独立 benchmark。完整整理见 [`gpt-6-sol-source-notes.md`](gpt-6-sol-source-notes.md)。

## 2026-09-23 GPT-6 Luna

- [Artificial Analysis GPT-6 Luna](https://artificialanalysis.ai/models/gpt-6-luna)：三个代理详情页逐字节一致，`3,992,084` bytes / SHA-256 `cda7ec1a8b90bfb76c85624f31cab242a0779732f68d65641b0dee5929a5d0f1`；标题为 `GPT-6 Luna (max)`，release date `2026-09-22`，max Intelligence Index `37.2559686869738`、median output speed `153.87508473888 tokens/s`、cost per Intelligence Index task `0.06809498628701058`。这些是 AA 第三方/provider 字段。
- [Artificial Analysis 中文首页](https://artificialanalysis.ai/zh)：三个代理逐字节一致，`1,782,868` bytes / SHA-256 `24182a9f96da6b6b96567bc2d069eddb553109a6bb323d07d79d900bb107770c`；当前重点厂商前列出现 `gpt-6-luna`，本轮按 canonical release 归并 effort/provider 变体。
- [DataCurve DeepSWE](https://deepswe.datacurve.ai/)：`268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`；当前没有精确 `mini_swe_agent_gpt_6_luna_*` 行，不迁移 Sol、Astra、GPT-5.6 或其他 GPT 的 Agent 结果。
- [GPT-6 Luna model page](https://developers.openai.com/api/docs/models/gpt-6-luna.md)：7890 代理 HTTP 200，`4,019` bytes / SHA-256 `561a86af72eced9a76a4e3a96c45ae7fff7782732514a8429b55ebddb4404945`；确认 `gpt-6-luna`、focused/high-volume 定位、text/image input、1,050,000 context、922,000 maximum input、128,000 maximum output、2026-05-18 knowledge cutoff、`none`--`max` effort、Responses/Chat Completions/Batch、工具目录和 `$0.10/$0.50` token 价格合同。1234/8098 返回 403，仅作为线路边界记录。
- [Reasoning](https://developers.openai.com/api/docs/guides/reasoning.md)、[Agents](https://developers.openai.com/api/docs/guides/agents.md)、[Using tools](https://developers.openai.com/api/docs/guides/tools.md) 和 [Compaction](https://developers.openai.com/api/docs/guides/compaction.md)：复用与 GPT-6 Sol 相同的 `70,315`/`5,432`/`33,282`/`14,272` bytes 快照和既有哈希；它们是 GPT-6 family runtime 证据，不改写成 Luna 专属内部算法。
- 当前状态：**AA 单榜资料级闭环**。完整记录见 [`gpt-6-luna-source-notes.md`](gpt-6-luna-source-notes.md)；参数、架构、完整训练 recipe、公开权重、生产 kernel、目标硬件 profiling、精确 DataCurve Agent 行、独立 benchmark 和 tool acceptance 仍待核验。

## 2026-09-23 DeepSeek V4.1-Flash 官方 API contract

- [Pricing](https://api-docs.deepseek.com/quick_start/pricing)：`23,149` bytes / SHA-256 `2fecee48bf6ad791bce38d1d5504d8ad5c8b0fd4da93e6dc198ae88ff1a4506a`；当前模型版本、价格和计费字段。
- [Rate Limit](https://api-docs.deepseek.com/quick_start/rate_limit)：`35,133` bytes / SHA-256 `1190b37c138b132b45ed3fa6e19861e7fa40be95a9a9c5a36104cc88ad26413a`；并发、速率与请求限制。
- [Error Codes](https://api-docs.deepseek.com/quick_start/error_codes)：`20,387` bytes / SHA-256 `0df2698a3c67c567476e476c75c74f69313b9025ded16eeb324514c52ae5094a`；错误类型与恢复入口。
- [Vision](https://api-docs.deepseek.com/guides/vision)：`78,174` bytes / SHA-256 `a805d8a40388ee238c626b83419c2cf786ac3002187c072699710132fc77dac7`；URL、`file_id`、请求体、图像数量和 image-token 预算边界。
- [Files](https://api-docs.deepseek.com/guides/files)：`61,828` bytes / SHA-256 `1020efca2be22faf0ec88f04aed7ee701b290c0d5ff0a465f7ec617314240345`；`purpose=user_data`、64 MiB 单文件、保存期、25 GiB/用户和 10,000 文件配额。
- [Responses API](https://api-docs.deepseek.com/guides/responses_api)：`56,250` bytes / SHA-256 `1719ac1b05e29579acd0cbc5ba0bcdb629cf7722eb551b5cd6d6d62c271e3ca2`；stateless、semantic SSE、递增 `sequence_number`、终态事件和不支持字段。
- [Tool Calls](https://api-docs.deepseek.com/guides/tool_calls)：`70,636` bytes / SHA-256 `5ee72ac00e5594bffac058cfef8872beb121b97e122f914c114a618f6a2b4027`；`function_call_output`/`custom_tool_call_output`、中途客户端工具回路和 tool/schema 边界。

本轮证据只升级到 **官方 API/provider contract**：`deepseek-flash` 对应 V4.1-Flash，1M context、384K max output 和最高 2500 concurrency；当前 Vision guide 的媒体限制与历史实验公告的 384 image tokens 必须按日期/alias 分账；Responses 以 semantic SSE 终态结束且不发送 `[DONE]`；`strict=true` 约束是 `/beta` API 能力，不代表 recipe adapter 或业务 verifier 已通过。API 页面不能替代完整权重、FP4/FP8 误差、Top-K recall、DSpark acceptance/rollback、EPD、硬件 profile 或生产 SLO。

### 2026-09-23 API contract toy

- [`deepseek_v41_api_contract_audit.py`](code/deepseek_v41_api_contract_audit.py)：标准库脚本，固定输入分块验证 semantic SSE 的 `sequence_number`/终态、strict object schema、工具权限、执行前 retry、idempotency 和独立 verifier；不访问网络、不调用付费 API、不下载权重。
- 运行结果：3 个 SSE 事件，文本 `contract audit`，终态 `response.completed`；首次调用 2 次尝试后完成，重复调用不增加 side effect；extra property 被 schema gate 拒绝，越权路径被 permission gate 拒绝，`network_called=false`。
- 证据等级仍是 **local protocol toy**。它不能升级真实 DeepSeek endpoint、alias/served-model 路由、`strict=true` 完整实现、工具权限、模型质量、FP4/FP8、DSpark、硬件或生产 SLO。

## 2026-09-23 Gemini 3.8 Flash：Interactions state、thinking budget 与 tool context

- 榜单发现：[Artificial Analysis Gemini 3.8 Flash](https://artificialanalysis.ai/models/gemini-3-8-flash)；[DataCurve DeepSWE](https://deepswe.datacurve.ai/) 的精确对象为 `gemini-3-8-flash` + `mini-swe-agent` + `high`。high 的 Agent 结果只绑定该 effort、harness、工具、环境和 verifier，不迁移到 low/medium。
- 官方模型页：[Gemini 3.8 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash)；运行时资料：[Thinking](https://ai.google.dev/gemini-api/docs/thinking)、[Interactions API](https://ai.google.dev/gemini-api/docs/interactions-overview)、[Tool combination](https://ai.google.dev/gemini-api/docs/tool-combination)、[Thought signatures](https://ai.google.dev/gemini-api/docs/thought-signatures)。
- 2026-09-23 `10.24.27.134:7890` 官方快照：model page `103,745` bytes / `e609ede21f99462207a0552cde54c8a7f7462e23a28a7515fb774c84f8f16f2b`；Thinking `226,287` / `a443bebc66c224284ea8f18ab8f6576fc4aebeb9d3419113be722da4bf7feef8`；Interactions `105,650` / `ee250076b4cb66072a5e98951cc2c7d2003b0ae6fce3b5cce398821443b0a553`；Tool combination `124,824` / `038adc9426510e7f15075d966186d7689817a06fd3c3be1c52100a3f397991ff`；Thought signatures `87,723` / `6b269d05196fef00a5837fb9aa4e42d31034acd072f6bb6064fd91b6a6c89e8c`。
- 已核验：stable alias `gemini-3.8-flash`、1,048,576 input tokens、65,536 output tokens、默认 `medium`、合法 `low/medium/high`；`max_output_tokens` 同时覆盖 thinking 与 visible output，触顶时可能 `incomplete`；`total_thought_tokens` 与 output tokens 应分账。
- Interactions contract：默认 `store=true`；付费/免费保留 `55/1` 天；`store=false` 不能配合 background execution 或后续 `previous_interaction_id`；可按 ID 删除；`previous_interaction_id` 只保留 conversation history，tools/system/generation config 是 interaction-scoped；stateful/stateless 都支持 implicit caching；混用模型时检查输出模态兼容性。
- Tool contract：Interactions 将 user input、thought、built-in/custom tool call/result 和 model output 组织成 steps；function call/result 以 `id` 对齐；Search/Maps/URL/File Search 为 server-side，Code Execution 有独立 server-side steps，Computer Use/custom function 为 client-side。工具组合要求 `validated`，不支持 `auto`。
- 证据冲突已显式保留：Thinking 页面把 signature 限定在 thought/built-in tool steps，Tool combination 页面又把 Gemini 3+ tool call/result signatures 描述为普遍存在；stateless 实现必须原样保存实际返回的 opaque fields，标准 function call 的 signature 需以具体 schema/SDK capability probe 为准。
- 本地教学实验：[`gemini_interactions_replay_demo.py`](code/gemini_interactions_replay_demo.py)，标准库合成 replay，验证 stateful/stateless、signature/id、SSE 终态、删除/保留期和 modality gate；不调用 API、不代表模型质量或生产 SLA。研究笔记：[`gemini-3.8-flash-source-notes.md`](gemini-3.8-flash-source-notes.md)。
- 当前状态：**资料级闭环**。独立 3.8 架构/训练报告、真实 API capability probe、thinking 消融、长上下文/cache billing、完整权重、目标硬件 profiling、独立 benchmark 和线上 acceptance 仍待核验；不新增重复 Transformer 正式章节。

## 2026-09-23 Kimi K3 当前时点复验

- [Artificial Analysis 中文首页](https://artificialanalysis.ai/zh)：三条代理逐字节一致，`1,783,605` bytes / SHA-256 `7a954aed8916ec9c5c88380cfc159274573dc9547182032ca142f556c0fc2916`。
- [DataCurve DeepSWE](https://deepswe.datacurve.ai/)：三条代理逐字节一致，`268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`；八家重点厂商按 canonical slug 去重后没有新增模型。
- [Artificial Analysis Kimi K3](https://artificialanalysis.ai/models/kimi-k3)：`4,080,091` bytes / SHA-256 `340a36ac5922f354aa42553fe3da2ce4acf4231bbb6af97fd6a88b974dbea4ee`；`max` Intelligence Index `43.5938229518782`、median output speed `36.9982439081499 tokens/s`、cost per Intelligence Index task `2.0001323004425493`、context `1M`。这些是第三方/provider 字段。
- DataCurve 精确对象为 `mini_swe_agent_kimi_k3_max`：`309/451`、Pass@1 `0.6851441241685144`、Pass@4 `0.8938053097345132`、平均成本 `$4.654682129933482`、平均输出 `81,499.84` tokens、平均 Agent steps `97.5876`、`4 runs`/`113 tasks`；只按 `mini-swe-agent + tools + environment + verifier` 系统结果引用，不迁移为裸模型分数。
- [Kimi K3 README](https://github.com/MoonshotAI/Kimi-K3/blob/main/README.md)：`45,004` bytes / SHA-256 `849a303d849486aac61d1f0c253e3a1148ca59e801be807151c2dc3cce57ccb2`，与 9 月 18 日快照逐字节一致。[HF metadata](https://huggingface.co/api/models/moonshotai/Kimi-K3)：`9,436` bytes / SHA-256 `6dfa5b1680074f4dfdd79a36b3342a191aacc32e15a9bd4a3895b98570773939`，revision 仍为 `f831ab66814297da540d832a5235f8e904f29d06`。
- SGLang `main` K3 文本当前为 `171,130` bytes / SHA-256 `9af22b45f8d8f8a5931c3bd60310316090973fd5cf9626aabd2a618a9e4baa6d`；较 9 月 22 日仅有 import/type annotation 变化，没有实质 runtime 技术变化。固定 config、vLLM recipe、FlashKDA 和双状态 serving 证据未漂移。
- 面试边界：`2.5x scaling efficiency` 是发布方声明；preserved thinking history 需要结构化原样回传；跨模型切换可能不稳定；excessive proactiveness 是官方限制；Kimi Code/Claude Code/Codex、H20/H100 和 compaction 混用使发布评测不可直接拼接。当前状态为 **内容专题闭环 + 当前时点榜单/官方 revision 复验**；完整权重、目标硬件 profile、hybrid cache recovery、tool/verifier acceptance 和生产 SLO 仍待核验。

## 2026-09-23 GLM-5.3：SAO 与 compaction 关联技术补证

- [Artificial Analysis 中文首页](https://artificialanalysis.ai/zh)：三条代理逐字节一致，`1,783,597` bytes / SHA-256 `ce87eece01c108877686e76d76f7e26d463416d0a6cd895e554e284549a286ad`；[DataCurve DeepSWE](https://deepswe.datacurve.ai/)：`268,036` bytes / SHA-256 `14436c31be1e50a0b62171e4aee4dd0ae0ce66b1e390af89c7e6e095ad59f1f1`。本轮没有从官方资料另发现模型，锚点仍是榜单已有的标准 `GLM-5.3`。
- [Artificial Analysis GLM-5.3 详情](https://artificialanalysis.ai/models/glm-5-3)：`4,074,151` bytes / SHA-256 `805918eef4c32b3cdeaf482f316b3311016b0523aef4a24886700e02e906ca73`；Intelligence Index `44.777392385614`、median output speed `57.0553482500823 tokens/s`、cost per Intelligence Index task `2.0056375150449584`、context `1,000,000`、`753B total / 40B active`、open weights、release `2026-08-18`。这些是 AA/provider 配置字段。
- DataCurve 精确对象为 `model=glm-5-3`、`harness=mini-swe-agent`、`reasoning_effort=max`、`config=mini_swe_agent_glm_5_3_max`：`311/451`、Pass@1 `0.6895787139689579`、Pass@4 `0.8761061946902655`、平均成本 `$3.9933584893126386`、平均输出 `80435.60975609756` token、平均 `124.47228381374723` Agent steps。它绑定工具、任务集、环境和 verifier，不能写成裸模型分数。
- [Z.ai GLM-5.3 Markdown](https://docs.z.ai/guides/llm/glm-5.3)：本地 `/tmp/glm53-md-20260923-7890.out`，`23,446` bytes / SHA-256 `9545c3d6fb1cabfa5951928bbe9a535e6958561d3bd6b5f4354cd5d3b0f6c929`；[GLM-5.2 Markdown](https://docs.z.ai/guides/llm/glm-5.2)：`25,293` bytes / SHA-256 `eeca971119068b0619137ce9c9d3c9362131baf16fdab0c067eefe5802919023`。官方文档确认 GLM-5.3 沿用 GLM-5.2 基础模型并继承 `SAO with compaction`，但没有公开 5.3 专属 compaction 实现。
- [SAO 论文 HTML](https://arxiv.org/html/2607.07508v1)：本地 `/tmp/sao-html-20260923-1234.out`，`190,454` bytes / SHA-256 `953b8968fa30d5f579a9650cc17d95521515ccac6cc7408b2a45c8126651822d`；[PDF](https://arxiv.org/pdf/2607.07508)：`664,828` bytes / SHA-256 `44c695be0428c666d06c914ba76c037e3ac77eeb5db0a81bbe239719c21bda48`。HTML 是本轮主要正文证据，论文直接实验对象包含 Qwen3-30B-A3B，摘要明确 SAO 部署到 GLM-5.2（750B-A40B）。
- 当前状态：**GLM-5.3 双榜资料级闭环 + SAO 关联论文算法证据**。SAO 的 single-rollout、DIS、双侧 token clipping、critic/value model、冻结 attention 和 Skip-Observation GAE 已进入研究资料；5.3 专属 compaction 状态、完整 recipe、完整权重、目标硬件 profile、独立 benchmark 和线上 tool acceptance 仍待核验。
- 本地教学实验：[`sao_async_rl_toy.py`](code/sao_async_rl_toy.py) 运行通过 `group_barrier_wait_total=11`、single-rollout wait `0`、极端 ratio mask、Skip-Observation GAE 对 observation 长度不变和 `K=2` critic proxy 收敛更快；证据等级为 **local protocol toy**，不代表真实 SAO、GLM-5.2/5.3 或生产 compaction。

## 2026-09-23 GLM-5.3 compaction contract audit

- 官方 [GLM-5.3 Markdown](https://docs.z.ai/guides/llm/glm-5.3) 的本地快照为 `23,446` bytes / SHA-256 `9545c3d6fb1cabfa5951928bbe9a535e6958561d3bd6b5f4354cd5d3b0f6c929`；博客正文资源为 `30,414` bytes / SHA-256 `f809bf586f74b40a01ecd8d674f9ec3de469a5abeae125e2a849c9805605d0f3`。两者都只支持“继承 SAO with compaction”的高层结论，不支持 5.3 专属状态格式、阈值或 recipe。
- 当前复抓的三条代理对 Z.ai 页面和百度诊断均为连接失败、HTTP `000`；这条线路事实不升级为页面不存在，也不改变两榜模型发现规则。
- [`glm53_compaction_contract_audit.py`](code/glm53_compaction_contract_audit.py) 是本地零依赖教学协议，覆盖 schema round-trip、tool lineage、permission/idempotency、pending effect、artifact/verifier、budget 和 cut marker；证据等级为 `local_protocol_toy`，不代表 Z.ai 或 GLM-5.3 生产行为。
