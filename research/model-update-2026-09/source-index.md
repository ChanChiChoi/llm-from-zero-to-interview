# 新模型资料来源索引

核验日期：2026-09-16。链接按“发现来源—官方资料—技术论文/实现”组织。榜单名称只用于发现候选，正式结论优先使用官方资料。

## GPT-6 Astra

- 官方模型页：[developers.openai.com/api/docs/models/gpt-6-astra.md](https://developers.openai.com/api/docs/models/gpt-6-astra.md)
- 已核验：模型 ID、输入输出模态、上下文/输入/输出预算、reasoning effort、工具和端点支持。
- 待核验：发布日期、参数规模、训练架构、技术报告。

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
- 当前状态：资料级闭环。研究笔记：[`gpt-5.4-source-notes.md`](gpt-5.4-source-notes.md)。GPT-5.4/Pro 已具备锚点级资料；mini/nano 只作为榜单和官方指南中的关联变体保留，不把它们自动升级为独立闭环。

## Kimi K3

- 官方发布：[kimi.com/en/blog/kimi-k3](https://www.kimi.com/en/blog/kimi-k3)
- 排行榜发现：[Artificial Analysis](https://artificialanalysis.ai/models/kimi-k3)、[DeepSWE](https://deepswe.datacurve.ai/)
- 相关论文：[Kimi Linear, arXiv:2510.26692](https://arxiv.org/abs/2510.26692)
- Attention Residuals：[arXiv:2603.15031](https://arxiv.org/abs/2603.15031)、[官方仓库](https://github.com/MoonshotAI/Attention-Residuals)
- 已核验：K3 发布文章披露的 KDA、AttnRes、Stable LatentMoE、量化和 Agent 限制；KDA 的独立论文机制；AttnRes 论文机制；发布文章中的接入方式、价格快照、Mooncake/缓存命中率自报和 benchmark effort/采样条件。
- 待核验：K3 完整技术报告、权重交付状态、精确配置和 FlashKDA 当前可用版本。
- 教学落地：[`第十七册第 15 章`](../../book-17-agent-tool-use/chapters/15-kimi-k3发布证据与长任务harness.md)，覆盖发布证据分层与长任务 harness。

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
- 证据边界：官方“lossless context”、数月 Coding Agent 专项训练和 benchmark/开发者案例均按发布方产品描述记录；不能升级为数学无损、完整训练 recipe 或独立复现。GLM-5.3 提到的 SAO with compaction 仍没有 GLM-5.2 原始定义。
- 当前状态：资料级闭环；暂无独立正式章节。研究笔记：[`glm-5.2-source-notes.md`](glm-5.2-source-notes.md)。

## GLM-5

- 排行榜发现：[Artificial Analysis GLM-5](https://artificialanalysis.ai/models/glm-5) 的 `GLM-5 (Reasoning)`，并列出 `glm-5-non-reasoning`；本轮三条代理均返回 HTTP 200，详情快照为 `3,577,227` bytes，SHA-256 `57dbab2e4ef52e2c95c585bb1d0549f8044486903b77366fed1d5c534569c2a1`。
- DataCurve [DeepSWE](https://deepswe.datacurve.ai/) 本轮三条代理均返回 HTTP 200，快照 SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`；当前只检出 GLM-5.2、GLM-5.3 和 GLM-5.3-Flash 配置，没有精确 `GLM-5` 行。因此 GLM-5 是 AA 单榜发现，不能迁移相邻版本的 DeepSWE 分数。
- 官方资料：[Z.ai GLM-5 博客](https://z.ai/blog/glm-5)、[API 文档](https://docs.z.ai/guides/llm/glm-5)、[Hugging Face 模型卡](https://huggingface.co/zai-org/GLM-5)、[固定 README](https://huggingface.co/zai-org/GLM-5/raw/main/README.md)、[固定 config.json](https://huggingface.co/zai-org/GLM-5/raw/main/config.json)、[官方 GitHub](https://github.com/zai-org/GLM-5) 和 [`slime`](https://github.com/THUDM/slime)。
- 专属论文：[GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/abs/2602.15763)。模型卡和报告确认 744B total/40B active、28.5T 预训练 tokens、DSA、异步 RL 基础设施 `slime` 和长周期 Agent 定位；配置公开 `GlmMoeDsaForCausalLM`、78 层、256 routed experts、top-8、1 shared expert、`index_topk=2048` 和约 202K position。
- 面试主线：DSA 的 indexer/top-k 召回与端到端成本、MoE 总容量与 active compute 分离、rollout/trainer 解耦的 policy lag 与样本新鲜度、长轨迹 Agent RL 的 verifier/credit assignment，以及模型—harness—工具—环境—verifier 的评测分层。发布方 benchmark 不与 AA/DataCurve 拼成裸模型能力。
- 当前状态：AA 单榜资料闭环；没有精确 DataCurve 行，但已有官方模型卡、专属技术报告、API/部署资料和研究笔记。完整 DSA indexer 训练目标、生产 kernel、硬件 profiling、`slime` 调度与完整训练/后训练 recipe 仍待核验。研究笔记：[`glm-5-source-notes.md`](glm-5-source-notes.md)。

## GLM-5.3

- 官方文档：[docs.z.ai/guides/llm/glm-5.3](https://docs.z.ai/guides/llm/glm-5.3)
- 前代文档：[docs.z.ai/guides/llm/glm-5.2](https://docs.z.ai/guides/llm/glm-5.2)
- 排行榜发现：[Artificial Analysis](https://artificialanalysis.ai/models/glm-5-3)、[DeepSWE](https://deepswe.datacurve.ai/)
- 已核验：GLM-5.3 与 GLM-5.2 的基础模型/后训练关系、推理档位、1M 输入窗口、Agent 环境与验证器描述。
- 待核验：SAO with compaction 的原始定义、完整技术报告、权重与许可证。
- 官方文档索引快照还列出 GLM-5.3-Flash 专属页面、迁移指南及流式/工具/缓存/结构化输出文档；当前只把它们记录为入口和接入范围，不把索引摘要当作 Flash 模型规格或 benchmark 证据。

## GLM-5.3-Flash

- 排行榜发现：[Artificial Analysis GLM-5.3-Flash](https://artificialanalysis.ai/models/glm-5-3-flash)，canonical slug `glm-5-3-flash`、`max` reasoning effort、页面 `releaseDate` 字段 `2026-08-26`；[DataCurve DeepSWE v1.1](https://deepswe.datacurve.ai/) 的 `mini_swe_agent_glm_5_3_flash_max` 配置。
- Artificial Analysis 本次复验快照：1M context、Intelligence Index `41.907366113455`、median output speed `114.22108687545 tokens/s`、median TTFT `2.45458272199994s`、约 `$0.15/$0.50/$0.026` 每百万 token；快照 SHA-256 `7800ff202ced5e5cc170d7f1858d8070cf8d41c47b3ab6bace60b75c596b6319`。这些是第三方配置字段。
- DataCurve 本次复验快照 SHA-256 `8fdbb59257d00cbb0772248bb602aafc1d20a51822388cd4eaf4625505182be7`；配置为 `max`、`n_runs=4`、284/448、Pass@1 `63.392857%`、Pass@4 `84.955752%`、平均成本约 `$0.2409818562`、平均输出 `72829.77` token、平均 Agent steps `122.89`。结果绑定 `mini-swe-agent`、工具、任务集、环境和 verifier，不能写成裸模型能力。
- 官方模型文档：[GLM-5.3-Flash](https://docs.z.ai/guides/vlm/glm-5.3-flash)、[Thinking Mode](https://docs.z.ai/guides/capabilities/thinking-mode)、[Streaming](https://docs.z.ai/guides/capabilities/stream-tool)、[Function Calling](https://docs.z.ai/guides/capabilities/function-calling)、[Context Caching](https://docs.z.ai/guides/capabilities/cache)、[Structured Output](https://docs.z.ai/guides/capabilities/struct-output)、[迁移指南](https://docs.z.ai/guides/overview/migrate-to-glm-new.md)。文档快照 SHA-256 `a127bf7eff2780aacebfc4ffdcadfac5820b75caeaafdb932da0c8942eee879f`。
- 官方博客：[GLM-5.3-Flash](https://z.ai/blog/glm-5.3-flash)：hybrid linear+sparse attention、IndexPool、mHC、30T multimodal corpus、visual self-judgment/test-time improvement、SGLang、ReplaySSM、W8A8、混合 cache quantization、Layer Split、EPD 和约 3× serving 的发布方描述。正文本次由页面 JS 加载，资源 SHA-256 `225196c63b5944629606d26982c0a43c4a8fbd6edb8a7a2e9bf8abfacd35fcbb`。
- 官方模型卡：[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a)，固定 revision `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`；[config.json](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/config.json)。README/config 公开 `Glm5NextForConditionalGeneration`、320B/18B、45 层、34 `linear_attention`/11 `deepseek_sparse_attention`、前 3 层 dense MLP、288 routed/top-8/1 shared、1M、IndexPool 和 mHC 字段；模型卡 front matter 标注 MIT。
- 视觉配置公开 24 层 vision module、hidden size 1024、image size 448、patch 14、temporal patch 2、输出投影 4096；输入为 video/image/text/file，输出为 text。官方博客/文档将视觉能力放入 observe—render/use—verify—refine coding loop，不能把它写成完备 verifier 或宿主权限。
- 关联实现：[SGLang cookbook](https://cookbook.sglang.io/autoregressive/GLM/GLM-5.3-Flash)、[vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash)、[Transformers GLM5-Next](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/glm5_next.md)、[KTransformers tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/GLM-5.3-Flash-Tutorial.md)。这些是部署入口，不替代目标硬件 profiling。
- 当前状态：内容专题闭环，正式落点为 [第二十一册第 84 章](../../book-21-transformer-architecture-evolution/chapters/84-glm-5.3-flash混合注意力与视觉闭环.md)，研究笔记见 [`glm-5.3-flash-source-notes.md`](glm-5.3-flash-source-notes.md)。
- 待核验：完整训练/后训练 recipe、生产 linear-attention/ReplaySSM/IndexPool/MoE/mHC/EPD kernel、真实 state/KV/indexer bytes、`index_topk=2048` 的最终可见 token 语义、硬件 profiling、线上 acceptance rate、API endpoint 差异和独立 benchmark。GLM-5 技术报告 [arXiv:2602.15763](https://arxiv.org/abs/2602.15763) 是模型卡引用的关联报告，不把其中未明确归属 Flash 的数字直接迁移。

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
- 已逐页提取并复核 51 页技术报告正文；报告补充了精确层排布、HSI 候选池、Single-Pass mHC、DSpark、FP4 量化位置、EPD/SWA 部署、训练设置和异步后训练细节。完整 kernel source、所有参数分片、线上接受率、目标硬件 profiling 和独立 benchmark 仍待核验。研究笔记：[`deepseek-v4.1-flash-source-notes.md`](deepseek-v4.1-flash-source-notes.md)。

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
- 2026-09-15 复验：Artificial Analysis `max + default fallback` 的 Intelligence Index 为 `53.3738`、输出速度约 `65.98 tokens/s`、TTFT 约 `212.01s`、成本约 `$7.63`/task；这些是第三方配置级测量。DataCurve 页面仍只有 Fable 5 行。
- 页面/发布方自述：定位为 demanding reasoning 与 long-horizon agentic work；页面列出 preserved thinking、跨轮模型切换、per-message effort/turn-scoped system messages/工具间进度更新等 beta 入口。发布页还确认 Fable 5.1 与 Mythos 5.1 共享底模但 safeguards 不同、cache read 为 `$0.25/M`，并公开了发布方 benchmark、安全策略和科学工作流案例。
- 论文检索：[arXiv 标题查询](https://arxiv.org/search/?query=%22Claude+Fable+5.1%22&searchtype=title) 返回 0 篇；[全文查询](https://arxiv.org/search/?query=%22Claude+Fable+5.1%22&searchtype=all) 返回 4 篇外部使用/评测论文，未发现 Fable 5.1 专属技术报告。仍待核验参数规模、稠密/MoE 架构、训练/后训练配方、System Card 正文安全数字和独立 benchmark 复现。
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
- 2026-09-15 复核确认：`claude-sonnet-5`、`releaseDate: 2026-06-30`、proprietary/非 open weights、parameters 为 null、1M context、128K 普通/300K Batch 最大输出、Adaptive、默认 high effort、Fast、平台与价格字段。Artificial Analysis 主配置 `max` 的 Intelligence Index 为 `38.3576962882576`、约 80.00 output tokens/s、约 202.63s TTFT、约 `$5.0912`/task；这些是第三方配置级字段。
- DataCurve DeepSWE v1.1 当前为 113 tasks/91 repositories/5 languages、统一 `mini-swe-agent`；Sonnet 5 的 max/xhigh/high/medium/low Pass@1 分别为 53.846%/49.667%/48.230%/39.778%/30.512%，对应的平均成本约为 `$26.40/$11.89/$7.43/$4.08/$2.19`。所有数字绑定 effort、harness、工具、任务集、环境和 verifier，不能写成裸模型能力。
- 面试技术点：adaptive thinking 不接受手动 `thinking.type: "enabled"` + `budget_tokens`；effort 是行为信号而非严格预算，`max_tokens` 是思考/工具/文本共享的硬上限；thinking blocks/signatures 需按异构 `content` 回放；新 tokenizer 约增加 30% token；context awareness、`compact-2026-01-12` server-side compaction、`computer_toolset_20260801` 和 programmatic tool calling 共同构成长任务运行时。
- System Card/发布方结果已单独记录：SWE-bench Verified 85.2%、SWE-bench Pro 63.2%、Multilingual SWE-bench 78.3%、Terminal-Bench 2.1 80.4%、BrowseComp 84.7%、OSWorld-Verified 81.2%、GDPval-AA v2 Elo 1618；这些不替代独立复现。
- arXiv 精确标题检索截至 2026-09-15 返回 0 个结果，Anthropic Research 页面未检出 Sonnet 5 专属技术报告；参数、架构、完整训练/后训练 recipe、内部 adaptive-thinking 机制和独立 benchmark 复现仍待核验。当前状态：资料级闭环，不新增独立架构章节。
- 研究笔记：[`claude-sonnet-5-source-notes.md`](claude-sonnet-5-source-notes.md)。

## Claude Sonnet 4.6

- 排行榜发现：[Artificial Analysis Claude Sonnet 4.6 Adaptive](https://artificialanalysis.ai/models/claude-sonnet-4-6-adaptive)；[DataCurve DeepSWE](https://deepswe.datacurve.ai/) 的 `mini_swe_agent_claude_sonnet_4_6_high`。
- 官方资料：[Introducing Claude Sonnet 4.6](https://www.anthropic.com/news/claude-sonnet-4-6)、[Anthropic Models Overview](https://platform.claude.com/docs/en/models/overview)、[System Card](https://www.anthropic.com/claude-sonnet-4-6-system-card)、[Research](https://www.anthropic.com/research) 及 Thinking/Compaction/Agent 工具文档。
- 已核验：`claude-sonnet-4-6`、2026-02-17 发布日期、coding/computer use/long-context reasoning/agent planning 定位、1M context（发布时 beta）、adaptive/extended thinking、context compaction、tool search、web search/fetch、code execution、memory、programmatic tool calling 和 computer-use prompt-injection 风险边界。
- 评测边界：Artificial Analysis 的 adaptive/max 配置与 DataCurve 的 135/451、Pass@1 `29.9335%`、Pass@4 `56.6372%` 均是配置级或 Agent 系统结果，不能合并为裸模型能力。
- 2026-09-16 夜间中断复验：三条代理对 Artificial Analysis 首页、DataCurve 和 Sonnet 4.6 发布页均返回 HTTP 200；排行榜快照分别为 `1,773,553` 和 `268,313` bytes，三个代理逐字节一致。 `8098/1234` 的模型目录重定向到区域不可用页，`7890` 成功取得真实 `platform.claude.com` 目录（`728,089` bytes，SHA-256 `4c5ba69551075bbff00251943a1765c77be38f998d512fac67902067097cdfff`）。
- 证据边界：没有公开参数规模、内部架构、完整训练/后训练 recipe 或 Sonnet 4.6 专属技术报告；adaptive thinking 内部预算、compaction 摘要格式、tool search 质量和线上 computer-use 接受率仍待核验。当前状态为资料级闭环，不新增独立架构章节。
- 研究笔记：[`claude-sonnet-4.6-source-notes.md`](claude-sonnet-4.6-source-notes.md)。

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

## Artificial Analysis 2026-09-14 实时快照

- 独立记录：[`artificial-analysis-2026-09-14-snapshot.md`](artificial-analysis-2026-09-14-snapshot.md)。本轮通过 `10.24.27.134:7890` 获取首页 HTTP 200，快照大小 1,771,961 bytes，SHA-256 `ebda1f3ff7fc1956dc629b82ab56400300683a8a134446882d12dfbb4d814423`。
- 页面解析到约 702 个配置条目、673 个唯一名称，最新 `releaseDate` 为 `2026-09-11`；相对 2026-09-09 页面结构新增 13 个 slug/别名，包含 DeepSeek V4.1 Flash、Agnes 3.0 Flash、Ling-3.0-flash-VL、K2 Horizon 系列和 `mbzuai` 目录项。
- DeepSeek V4.1-Flash 与 K2 Horizon MoVA 36B/A4B 已由相应官方资料升级为核验专题；Agnes、Ling、K2 Horizon 3.7B 和 `mbzuai` 仍停留在榜单发现层。第三方 `releaseDate`、参数、开放性和指数不替代官方日期、revision、许可证或独立 benchmark。

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
